/* -------------------------------------------------------------------------- *
 *                              OpenMMGridForce                               *
 * -------------------------------------------------------------------------- *
 * CUDA implementation of BondedHessian using GPU kernels.                    *
 * -------------------------------------------------------------------------- */

#include "CudaBondedHessian.h"
#include "CudaGridForceKernelSources.h"
#include "openmm/HarmonicBondForce.h"
#include "openmm/HarmonicAngleForce.h"
#include "openmm/PeriodicTorsionForce.h"
#include "openmm/OpenMMException.h"
#include "openmm/cuda/CudaPlatform.h"
#include "openmm/internal/ContextImpl.h"

#include <map>
#include <cstring>

using namespace GridForcePlugin;
using namespace OpenMM;
using namespace std;

CudaBondedHessian::CudaBondedHessian()
    : cu(nullptr), initialized(false), numAtoms(0), numBonds(0), numAngles(0), numTorsions(0),
      bondHessianKernel(nullptr), angleHessianKernel(nullptr), torsionHessianKernel(nullptr),
      initHessianKernel(nullptr) {
}

CudaBondedHessian::~CudaBondedHessian() {
}

void CudaBondedHessian::initialize(const System& system, Context& context) {
    // Get CudaContext from the OpenMM Context
    ContextImpl& impl = *reinterpret_cast<ContextImpl*>(&context);
    cu = static_cast<CudaPlatform::PlatformData*>(impl.getPlatformData())->contexts[0];

    if (cu == nullptr) {
        throw OpenMMException("CudaBondedHessian: Context must use CUDA platform");
    }

    cu->setAsCurrent();
    numAtoms = system.getNumParticles();

    // Extract HarmonicBondForce parameters
    for (int i = 0; i < system.getNumForces(); i++) {
        const HarmonicBondForce* bondForce = dynamic_cast<const HarmonicBondForce*>(&system.getForce(i));
        if (bondForce != nullptr) {
            numBonds = bondForce->getNumBonds();
            if (numBonds > 0) {
                vector<int> h_bondAtoms(numBonds * 2);
                vector<float> h_bondParams(numBonds * 2);

                for (int j = 0; j < numBonds; j++) {
                    int atom1, atom2;
                    double length, k;
                    bondForce->getBondParameters(j, atom1, atom2, length, k);
                    h_bondAtoms[j * 2] = atom1;
                    h_bondAtoms[j * 2 + 1] = atom2;
                    h_bondParams[j * 2] = (float)k;
                    h_bondParams[j * 2 + 1] = (float)length;
                }

                bondAtoms.initialize<int>(*cu, numBonds * 2, "bondedHessian_bondAtoms");
                bondParams.initialize<float>(*cu, numBonds * 2, "bondedHessian_bondParams");
                bondAtoms.upload(h_bondAtoms);
                bondParams.upload(h_bondParams);
            }
            break;
        }
    }

    // Extract HarmonicAngleForce parameters
    for (int i = 0; i < system.getNumForces(); i++) {
        const HarmonicAngleForce* angleForce = dynamic_cast<const HarmonicAngleForce*>(&system.getForce(i));
        if (angleForce != nullptr) {
            numAngles = angleForce->getNumAngles();
            if (numAngles > 0) {
                vector<int> h_angleAtoms(numAngles * 3);
                vector<float> h_angleParams(numAngles * 2);

                for (int j = 0; j < numAngles; j++) {
                    int atom1, atom2, atom3;
                    double angle, k;
                    angleForce->getAngleParameters(j, atom1, atom2, atom3, angle, k);
                    h_angleAtoms[j * 3] = atom1;
                    h_angleAtoms[j * 3 + 1] = atom2;
                    h_angleAtoms[j * 3 + 2] = atom3;
                    h_angleParams[j * 2] = (float)k;
                    h_angleParams[j * 2 + 1] = (float)angle;
                }

                angleAtoms.initialize<int>(*cu, numAngles * 3, "bondedHessian_angleAtoms");
                angleParams.initialize<float>(*cu, numAngles * 2, "bondedHessian_angleParams");
                angleAtoms.upload(h_angleAtoms);
                angleParams.upload(h_angleParams);
            }
            break;
        }
    }

    // Extract PeriodicTorsionForce parameters
    for (int i = 0; i < system.getNumForces(); i++) {
        const PeriodicTorsionForce* torsionForce = dynamic_cast<const PeriodicTorsionForce*>(&system.getForce(i));
        if (torsionForce != nullptr) {
            numTorsions = torsionForce->getNumTorsions();
            if (numTorsions > 0) {
                vector<int> h_torsionAtoms(numTorsions * 4);
                vector<float> h_torsionParams(numTorsions * 3);

                for (int j = 0; j < numTorsions; j++) {
                    int atom1, atom2, atom3, atom4, periodicity;
                    double phase, k;
                    torsionForce->getTorsionParameters(j, atom1, atom2, atom3, atom4, periodicity, phase, k);
                    h_torsionAtoms[j * 4] = atom1;
                    h_torsionAtoms[j * 4 + 1] = atom2;
                    h_torsionAtoms[j * 4 + 2] = atom3;
                    h_torsionAtoms[j * 4 + 3] = atom4;
                    h_torsionParams[j * 3] = (float)periodicity;  // n
                    h_torsionParams[j * 3 + 1] = (float)k;
                    h_torsionParams[j * 3 + 2] = (float)phase;
                }

                torsionAtoms.initialize<int>(*cu, numTorsions * 4, "bondedHessian_torsionAtoms");
                torsionParams.initialize<float>(*cu, numTorsions * 3, "bondedHessian_torsionParams");
                torsionAtoms.upload(h_torsionAtoms);
                torsionParams.upload(h_torsionParams);
            }
            break;
        }
    }

    // Allocate Hessian buffer (3N x 3N) using fixed-point for deterministic accumulation
    int hessianSize = 3 * numAtoms;
    hessianBuffer.initialize<unsigned long long>(*cu, hessianSize * hessianSize, "bondedHessian_hessian");

    // Load CUDA kernels
    map<string, string> defines;
    defines["NUM_ATOMS"] = cu->intToString(numAtoms);

    CUmodule module = cu->createModule(
        CudaGridForceKernelSources::commonHeaders +
        CudaGridForceKernelSources::bondedHessianKernel, defines);
    bondHessianKernel = cu->getKernel(module, "computeBondHessians");
    angleHessianKernel = cu->getKernel(module, "computeAngleHessians");
    torsionHessianKernel = cu->getKernel(module, "computeTorsionHessians");
    initHessianKernel = cu->getKernel(module, "initializeHessian");

    initialized = true;
}

std::vector<double> CudaBondedHessian::computeHessian(Context& context) {
    if (!initialized) {
        throw OpenMMException("CudaBondedHessian: must call initialize() first");
    }

    cu->setAsCurrent();

    int hessianSize = 3 * numAtoms;
    int totalElements = hessianSize * hessianSize;

    // Zero out Hessian buffer
    {
        CUdeviceptr hessianPtr = hessianBuffer.getDevicePointer();
        void* args[] = {&hessianPtr, &totalElements};
        int blockSize = 256;
        int numBlocks = (totalElements + blockSize - 1) / blockSize;
        cu->executeKernel(initHessianKernel, args, numBlocks * blockSize, blockSize);
    }

    // Get positions
    CUdeviceptr posqPtr = cu->getPosq().getDevicePointer();
    CUdeviceptr hessianPtr = hessianBuffer.getDevicePointer();

    // Compute bond Hessians
    if (numBonds > 0) {
        CUdeviceptr bondAtomsPtr = bondAtoms.getDevicePointer();
        CUdeviceptr bondParamsPtr = bondParams.getDevicePointer();

        void* args[] = {
            &posqPtr,
            &bondAtomsPtr,
            &bondParamsPtr,
            &hessianPtr,
            &numBonds,
            &numAtoms
        };

        int blockSize = 128;
        int numBlocks = (numBonds + blockSize - 1) / blockSize;
        cu->executeKernel(bondHessianKernel, args, numBlocks * blockSize, blockSize);
    }

    // Compute angle Hessians
    if (numAngles > 0) {
        CUdeviceptr angleAtomsPtr = angleAtoms.getDevicePointer();
        CUdeviceptr angleParamsPtr = angleParams.getDevicePointer();

        void* args[] = {
            &posqPtr,
            &angleAtomsPtr,
            &angleParamsPtr,
            &hessianPtr,
            &numAngles,
            &numAtoms
        };

        int blockSize = 128;
        int numBlocks = (numAngles + blockSize - 1) / blockSize;
        cu->executeKernel(angleHessianKernel, args, numBlocks * blockSize, blockSize);
    }

    // Compute torsion Hessians
    if (numTorsions > 0) {
        CUdeviceptr torsionAtomsPtr = torsionAtoms.getDevicePointer();
        CUdeviceptr torsionParamsPtr = torsionParams.getDevicePointer();

        void* args[] = {
            &posqPtr,
            &torsionAtomsPtr,
            &torsionParamsPtr,
            &hessianPtr,
            &numTorsions,
            &numAtoms
        };

        int blockSize = 128;
        int numBlocks = (numTorsions + blockSize - 1) / blockSize;
        fprintf(stderr, "DEBUG: About to call torsionHessianKernel with %d torsions\n", numTorsions);
        cu->executeKernel(torsionHessianKernel, args, numBlocks * blockSize, blockSize);
        fprintf(stderr, "DEBUG: Finished torsionHessianKernel\n");
    }

    // Download fixed-point Hessian and convert to double
    // Scale factor must match HESSIAN_SCALE in bondedHessian.cu (0x1000000 = 16777216)
    const double HESSIAN_SCALE_INV = 1.0 / 16777216.0;

    vector<unsigned long long> h_hessian(totalElements);
    hessianBuffer.download(h_hessian);

    // Convert from fixed-point to double
    // Cast to signed long long first to recover negative values via two's complement
    vector<double> result(totalElements);
    for (int i = 0; i < totalElements; i++) {
        long long signedVal = static_cast<long long>(h_hessian[i]);
        result[i] = signedVal * HESSIAN_SCALE_INV;
    }

    return result;
}
