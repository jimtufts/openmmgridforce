/* -------------------------------------------------------------------------- *
 *                              OpenMMGridForce                               *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2008-2012 Stanford University and the Authors.      *
 * Authors:                                                                   *
 * Contributors:                                                              *
 *                                                                            *
 * Permission is hereby granted, free of charge, to any person obtaining a    *
 * copy of this software and associated documentation files (the "Software"), *
 * to deal in the Software without restriction, including without limitation  *
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,   *
 * and/or sell copies of the Software, and to permit persons to whom the      *
 * Software is furnished to do so, subject to the following conditions:       *
 *                                                                            *
 * The above copyright notice and this permission notice shall be included in *
 * all copies or substantial portions of the Software.                        *
 *                                                                            *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR *
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,   *
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL    *
 * THE AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,    *
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR      *
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE  *
 * USE OR OTHER DEALINGS IN THE SOFTWARE.                                     *
 * -------------------------------------------------------------------------- */

#include "CudaIsolatedNonbondedKernels.h"
#include "CudaGridForceKernelSources.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/cuda/CudaBondedUtilities.h"
#include "openmm/cuda/CudaForceInfo.h"

#include <cstring>
#include <map>
#include <iostream>

using namespace GridForcePlugin;
using namespace OpenMM;
using namespace std;

CudaCalcIsolatedNonbondedForceKernel::~CudaCalcIsolatedNonbondedForceKernel() {
}

void CudaCalcIsolatedNonbondedForceKernel::initialize(const System& system, const IsolatedNonbondedForce& force) {
    cu.setAsCurrent();

    numAtoms = force.getNumAtoms();

    if (numAtoms == 0) {
        throw OpenMMException("IsolatedNonbondedForce: Must set number of atoms before initialization");
    }

    // Get particle indices
    h_particleIndices = force.getParticles();
    if ((int)h_particleIndices.size() != numAtoms) {
        throw OpenMMException("IsolatedNonbondedForce: Must call setParticles() with numAtoms indices");
    }

    // Allocate GPU arrays
    particleIndices.initialize<int>(cu, numAtoms, "isolatedNB_particleIndices");
    charges.initialize<float>(cu, numAtoms, "isolatedNB_charges");
    sigmas.initialize<float>(cu, numAtoms, "isolatedNB_sigmas");
    epsilons.initialize<float>(cu, numAtoms, "isolatedNB_epsilons");

    // Upload particle indices
    particleIndices.upload(h_particleIndices);

    // Upload parameters
    vector<float> h_charges(numAtoms);
    vector<float> h_sigmas(numAtoms);
    vector<float> h_epsilons(numAtoms);

    for (int i = 0; i < numAtoms; i++) {
        double charge, sigma, epsilon;
        force.getAtomParameters(i, charge, sigma, epsilon);
        h_charges[i] = (float)charge;
        h_sigmas[i] = (float)sigma;
        h_epsilons[i] = (float)epsilon;
    }

    charges.upload(h_charges);
    sigmas.upload(h_sigmas);
    epsilons.upload(h_epsilons);

    // Upload exclusions
    int numExclusions = force.getNumExclusions();
    if (numExclusions > 0) {
        vector<int2> h_exclusions(numExclusions);
        for (int i = 0; i < numExclusions; i++) {
            int atom1, atom2;
            force.getExclusion(i, atom1, atom2);
            h_exclusions[i] = make_int2(atom1, atom2);
        }
        exclusions.initialize<int2>(cu, numExclusions, "isolatedNB_exclusions");
        exclusions.upload(h_exclusions);
    } else {
        exclusions.initialize<int2>(cu, 1, "isolatedNB_exclusions");
    }

    // Upload exceptions (1-4 scaled interactions with custom parameters)
    int numExceptions = force.getNumExceptions();
    if (numExceptions > 0) {
        vector<int2> h_exceptions(numExceptions);
        vector<float3> h_exceptionParams(numExceptions);
        for (int i = 0; i < numExceptions; i++) {
            int atom1, atom2;
            double chargeProd, sigma, epsilon;
            force.getExceptionParameters(i, atom1, atom2, chargeProd, sigma, epsilon);
            h_exceptions[i] = make_int2(atom1, atom2);
            h_exceptionParams[i] = make_float3((float)chargeProd, (float)sigma, (float)epsilon);
        }
        exceptions.initialize<int2>(cu, numExceptions, "isolatedNB_exceptions");
        exceptionParams.initialize<float3>(cu, numExceptions, "isolatedNB_exceptionParams");
        exceptions.upload(h_exceptions);
        exceptionParams.upload(h_exceptionParams);
    } else {
        exceptions.initialize<int2>(cu, 1, "isolatedNB_exceptions");
        exceptionParams.initialize<float3>(cu, 1, "isolatedNB_exceptionParams");
    }

    // Load CUDA kernel
    map<string, string> defines;
    defines["NUM_ATOMS"] = cu.intToString(numAtoms);
    defines["NUM_EXCLUSIONS"] = cu.intToString(numExclusions);
    defines["NUM_EXCEPTIONS"] = cu.intToString(numExceptions);

    CUmodule module = cu.createModule(CudaGridForceKernelSources::gridForceKernel, defines);
    kernel = cu.getKernel(module, "computeIsolatedNonbonded");

    hasInitializedKernel = true;
}

double CudaCalcIsolatedNonbondedForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    if (!hasInitializedKernel) {
        return 0.0;
    }

    // Get the number of pairs
    int numPairs = (numAtoms * (numAtoms - 1)) / 2;
    if (numPairs == 0) {
        return 0.0;
    }

    // Set up kernel arguments
    int paddedNumAtoms = cu.getPaddedNumAtoms();
    CUdeviceptr posqPtr = cu.getPosq().getDevicePointer();
    CUdeviceptr forcePtr = cu.getLongForceBuffer().getDevicePointer();
    CUdeviceptr energyPtr = cu.getEnergyBuffer().getDevicePointer();
    CUdeviceptr particleIndicesPtr = particleIndices.getDevicePointer();
    CUdeviceptr chargesPtr = charges.getDevicePointer();
    CUdeviceptr sigmasPtr = sigmas.getDevicePointer();
    CUdeviceptr epsilonsPtr = epsilons.getDevicePointer();
    CUdeviceptr exclusionsPtr = exclusions.getDevicePointer();
    CUdeviceptr exceptionsPtr = exceptions.getDevicePointer();
    CUdeviceptr exceptionParamsPtr = exceptionParams.getDevicePointer();

    void* args[] = {
        &posqPtr,
        &forcePtr,
        &energyPtr,
        &particleIndicesPtr,
        &chargesPtr,
        &sigmasPtr,
        &epsilonsPtr,
        &exclusionsPtr,
        &exceptionsPtr,
        &exceptionParamsPtr,
        &numAtoms,
        &numPairs,
        &paddedNumAtoms,
        &includeEnergy
    };

    // Launch kernel
    int blockSize = 128;
    int numBlocks = (numPairs + blockSize - 1) / blockSize;
    cu.executeKernel(kernel, args, numBlocks * blockSize, blockSize);

    return 0.0;  // Energy is accumulated in the energy buffer
}

void CudaCalcIsolatedNonbondedForceKernel::copyParametersToContext(ContextImpl& context, const IsolatedNonbondedForce& force) {
    if (numAtoms != force.getNumAtoms()) {
        throw OpenMMException("Cannot update IsolatedNonbondedForce: number of atoms has changed");
    }

    // Update parameters
    vector<float> h_charges(numAtoms);
    vector<float> h_sigmas(numAtoms);
    vector<float> h_epsilons(numAtoms);

    for (int i = 0; i < numAtoms; i++) {
        double charge, sigma, epsilon;
        force.getAtomParameters(i, charge, sigma, epsilon);
        h_charges[i] = (float)charge;
        h_sigmas[i] = (float)sigma;
        h_epsilons[i] = (float)epsilon;
    }

    charges.upload(h_charges);
    sigmas.upload(h_sigmas);
    epsilons.upload(h_epsilons);

    cu.invalidateMolecules();
}

std::vector<double> CudaCalcIsolatedNonbondedForceKernel::computeHessian(ContextImpl& context) {
    if (!hasInitializedKernel) {
        throw OpenMMException("IsolatedNonbondedForce: must call execute() before computeHessian()");
    }

    // Get the number of pairs
    int numPairs = (numAtoms * (numAtoms - 1)) / 2;
    if (numPairs == 0) {
        return std::vector<double>(9 * numAtoms * numAtoms, 0.0);
    }

    // Initialize Hessian kernel if not already done
    if (hessianKernel == nullptr) {
        map<string, string> defines;
        defines["NUM_ATOMS"] = cu.intToString(numAtoms);
        defines["NUM_EXCLUSIONS"] = cu.intToString(exclusions.getSize() > 1 ? exclusions.getSize() : 0);
        defines["NUM_EXCEPTIONS"] = cu.intToString(exceptions.getSize() > 1 ? exceptions.getSize() : 0);

        CUmodule module = cu.createModule(CudaGridForceKernelSources::gridForceKernel, defines);
        hessianKernel = cu.getKernel(module, "computeIsolatedNonbondedHessians");
    }

    // Allocate Hessian buffer if needed (3N x 3N matrix, stored as 3x3 blocks)
    int hessianSize = 3 * numAtoms;
    int numBlocks3x3 = numAtoms * numAtoms;  // One 3x3 block per atom pair (i,j)
    if (hessianBuffer.getSize() != numBlocks3x3 * 9) {
        hessianBuffer.initialize<float>(cu, numBlocks3x3 * 9, "isolatedNB_hessian");
    }

    // Zero out the Hessian buffer
    cu.clearBuffer(hessianBuffer);

    // Set up kernel arguments
    int paddedNumAtoms = cu.getPaddedNumAtoms();
    CUdeviceptr posqPtr = cu.getPosq().getDevicePointer();
    CUdeviceptr particleIndicesPtr = particleIndices.getDevicePointer();
    CUdeviceptr sigmasPtr = sigmas.getDevicePointer();
    CUdeviceptr epsilonsPtr = epsilons.getDevicePointer();
    CUdeviceptr exclusionsPtr = exclusions.getDevicePointer();
    CUdeviceptr hessianPtr = hessianBuffer.getDevicePointer();

    void* args[] = {
        &posqPtr,
        &particleIndicesPtr,
        &sigmasPtr,
        &epsilonsPtr,
        &exclusionsPtr,
        &hessianPtr,
        &numAtoms,
        &numPairs,
        &paddedNumAtoms
    };

    // Launch kernel - one thread per pair
    int blockSize = 128;
    int numBlocksKernel = (numPairs + blockSize - 1) / blockSize;
    cu.executeKernel(hessianKernel, args, numBlocksKernel * blockSize, blockSize);

    // Download Hessian and convert to double
    vector<float> h_hessian(numBlocks3x3 * 9);
    hessianBuffer.download(h_hessian);

    // Convert to double and reformat as full 3N x 3N matrix
    vector<double> result(hessianSize * hessianSize, 0.0);
    for (int i = 0; i < numAtoms; i++) {
        for (int j = 0; j < numAtoms; j++) {
            int blockIdx = i * numAtoms + j;
            for (int di = 0; di < 3; di++) {
                for (int dj = 0; dj < 3; dj++) {
                    int row = 3 * i + di;
                    int col = 3 * j + dj;
                    result[row * hessianSize + col] = h_hessian[blockIdx * 9 + di * 3 + dj];
                }
            }
        }
    }

    return result;
}
