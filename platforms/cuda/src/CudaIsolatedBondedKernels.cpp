/* -------------------------------------------------------------------------- *
 *                              OpenMMGridForce                               *
 * -------------------------------------------------------------------------- *
 * CUDA platform implementation of IsolatedBondedForce kernel.                *
 * Computes harmonic bonds, angles, and periodic torsions for isolated        *
 * particle groups on GPU.                                                    *
 * -------------------------------------------------------------------------- */

#include "CudaIsolatedBondedKernels.h"
#include "CudaGridForceKernelSources.h"
#include "internal/BondedHessianAnalytical.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/cuda/CudaForceInfo.h"

#include <cstring>
#include <cmath>
#include <map>

using namespace GridForcePlugin;
using namespace OpenMM;
using namespace std;

CudaCalcIsolatedBondedForceKernel::~CudaCalcIsolatedBondedForceKernel() {
}

void CudaCalcIsolatedBondedForceKernel::initialize(const System& system, const IsolatedBondedForce& force) {
    cu.setAsCurrent();

    numAtoms = force.getNumAtoms();
    if (numAtoms == 0)
        throw OpenMMException("IsolatedBondedForce: Must set number of atoms before initialization");

    // Process particle groups
    int nGroups = force.getNumParticleGroups();
    if (nGroups > 0) {
        numParticleGroups = nGroups;
        h_particleIndices.resize(nGroups * numAtoms);
        for (int g = 0; g < nGroups; g++) {
            string name;
            vector<int> indices;
            force.getParticleGroup(g, name, indices);
            if ((int)indices.size() != numAtoms)
                throw OpenMMException("IsolatedBondedForce: particle group " + name + " has wrong number of indices");
            for (int i = 0; i < numAtoms; i++)
                h_particleIndices[g * numAtoms + i] = indices[i];
        }
    } else {
        throw OpenMMException("IsolatedBondedForce: Must add at least one particle group");
    }

    // Allocate and upload particle indices
    groupParticleIndices.initialize<int>(cu, numParticleGroups * numAtoms, "isolatedBonded_groupParticleIndices");
    groupParticleIndices.upload(h_particleIndices);

    // Per-group energy buffer
    groupEnergiesBuffer.initialize<float>(cu, numParticleGroups, "isolatedBonded_groupEnergies");
    groupEnergiesHost.resize(numParticleGroups, 0.0f);

    // Alchemical scaling
    globalScalingFactor = static_cast<float>(force.getGlobalScalingFactor());
    vector<float> h_groupScalings(numParticleGroups, 1.0f);
    for (int g = 0; g < nGroups; g++)
        h_groupScalings[g] = static_cast<float>(force.getGroupScalingFactor(g));
    groupScalingFactorsBuffer.initialize<float>(cu, numParticleGroups, "isolatedBonded_groupScalingFactors");
    groupScalingFactorsBuffer.upload(h_groupScalings);

    // Upload bond parameters
    numBonds = force.getNumBonds();
    h_bonds.resize(numBonds);
    if (numBonds > 0) {
        vector<int2> h_bondAtoms(numBonds);
        vector<float2> h_bondParams(numBonds);
        for (int i = 0; i < numBonds; i++) {
            int a1, a2;
            double length, k;
            force.getBondParameters(i, a1, a2, length, k);
            h_bondAtoms[i] = make_int2(a1, a2);
            h_bondParams[i] = make_float2((float)length, (float)k);
            h_bonds[i] = {a1, a2, length, k};
        }
        bondAtoms.initialize<int2>(cu, numBonds, "isolatedBonded_bondAtoms");
        bondParams.initialize<float2>(cu, numBonds, "isolatedBonded_bondParams");
        bondAtoms.upload(h_bondAtoms);
        bondParams.upload(h_bondParams);
    } else {
        bondAtoms.initialize<int2>(cu, 1, "isolatedBonded_bondAtoms");
        bondParams.initialize<float2>(cu, 1, "isolatedBonded_bondParams");
    }

    // Upload angle parameters
    numAngles = force.getNumAngles();
    h_angles.resize(numAngles);
    if (numAngles > 0) {
        vector<int4> h_angleAtoms(numAngles);
        vector<float2> h_angleParams(numAngles);
        for (int i = 0; i < numAngles; i++) {
            int a1, a2, a3;
            double angle, k;
            force.getAngleParameters(i, a1, a2, a3, angle, k);
            h_angleAtoms[i] = make_int4(a1, a2, a3, 0);
            h_angleParams[i] = make_float2((float)angle, (float)k);
            h_angles[i] = {a1, a2, a3, angle, k};
        }
        angleAtoms.initialize<int4>(cu, numAngles, "isolatedBonded_angleAtoms");
        angleParams.initialize<float2>(cu, numAngles, "isolatedBonded_angleParams");
        angleAtoms.upload(h_angleAtoms);
        angleParams.upload(h_angleParams);
    } else {
        angleAtoms.initialize<int4>(cu, 1, "isolatedBonded_angleAtoms");
        angleParams.initialize<float2>(cu, 1, "isolatedBonded_angleParams");
    }

    // Upload torsion parameters
    numTorsions = force.getNumTorsions();
    h_torsions.resize(numTorsions);
    if (numTorsions > 0) {
        vector<int4> h_torsionAtoms(numTorsions);
        vector<float4> h_torsionParams(numTorsions);
        for (int i = 0; i < numTorsions; i++) {
            int a1, a2, a3, a4, periodicity;
            double phase, k;
            force.getTorsionParameters(i, a1, a2, a3, a4, periodicity, phase, k);
            h_torsionAtoms[i] = make_int4(a1, a2, a3, a4);
            h_torsionParams[i] = make_float4((float)periodicity, (float)phase, (float)k, 0.0f);
            h_torsions[i] = {a1, a2, a3, a4, periodicity, phase, k};
        }
        torsionAtoms.initialize<int4>(cu, numTorsions, "isolatedBonded_torsionAtoms");
        torsionParams.initialize<float4>(cu, numTorsions, "isolatedBonded_torsionParams");
        torsionAtoms.upload(h_torsionAtoms);
        torsionParams.upload(h_torsionParams);
    } else {
        torsionAtoms.initialize<int4>(cu, 1, "isolatedBonded_torsionAtoms");
        torsionParams.initialize<float4>(cu, 1, "isolatedBonded_torsionParams");
    }

    // Load CUDA kernels
    map<string, string> defines;
    defines["NUM_ATOMS_BONDED"] = cu.intToString(numAtoms);
    defines["NUM_BONDS"] = cu.intToString(numBonds);
    defines["NUM_ANGLES"] = cu.intToString(numAngles);
    defines["NUM_TORSIONS"] = cu.intToString(numTorsions);

    CUmodule module = cu.createModule(CudaGridForceKernelSources::gridForceKernel, defines);
    if (numBonds > 0)
        bondKernel = cu.getKernel(module, "computeIsolatedBonds");
    if (numAngles > 0)
        angleKernel = cu.getKernel(module, "computeIsolatedAngles");
    if (numTorsions > 0)
        torsionKernel = cu.getKernel(module, "computeIsolatedTorsions");

    hasInitializedKernel = true;
}

double CudaCalcIsolatedBondedForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    if (!hasInitializedKernel)
        return 0.0;

    // Zero per-group energy buffer (only when energy is needed to avoid sync barriers)
    if (includeEnergy)
        cu.clearBuffer(groupEnergiesBuffer);

    int paddedNumAtoms = cu.getPaddedNumAtoms();
    CUdeviceptr posqPtr = cu.getPosq().getDevicePointer();
    CUdeviceptr forcePtr = cu.getLongForceBuffer().getDevicePointer();
    CUdeviceptr energyPtr = cu.getEnergyBuffer().getDevicePointer();
    CUdeviceptr groupIndicesPtr = groupParticleIndices.getDevicePointer();
    CUdeviceptr groupEnergiesPtr = groupEnergiesBuffer.getDevicePointer();
    CUdeviceptr groupScalingPtr = groupScalingFactorsBuffer.getDevicePointer();

    int blockSize = 128;

    // Launch bond kernel
    if (numBonds > 0) {
        int totalBondWork = numParticleGroups * numBonds;
        int numBlocks = (totalBondWork + blockSize - 1) / blockSize;

        CUdeviceptr bondAtomsPtr = bondAtoms.getDevicePointer();
        CUdeviceptr bondParamsPtr = bondParams.getDevicePointer();

        void* args[] = {
            &posqPtr, &forcePtr, &energyPtr,
            &groupIndicesPtr, &bondAtomsPtr, &bondParamsPtr,
            &groupEnergiesPtr, &groupScalingPtr,
            &globalScalingFactor, &numAtoms, &numBonds,
            &numParticleGroups, &paddedNumAtoms, &includeEnergy
        };
        cu.executeKernel(bondKernel, args, numBlocks * blockSize, blockSize);
    }

    // Launch angle kernel
    if (numAngles > 0) {
        int totalAngleWork = numParticleGroups * numAngles;
        int numBlocks = (totalAngleWork + blockSize - 1) / blockSize;

        CUdeviceptr angleAtomsPtr = angleAtoms.getDevicePointer();
        CUdeviceptr angleParamsPtr = angleParams.getDevicePointer();

        void* args[] = {
            &posqPtr, &forcePtr, &energyPtr,
            &groupIndicesPtr, &angleAtomsPtr, &angleParamsPtr,
            &groupEnergiesPtr, &groupScalingPtr,
            &globalScalingFactor, &numAtoms, &numAngles,
            &numParticleGroups, &paddedNumAtoms, &includeEnergy
        };
        cu.executeKernel(angleKernel, args, numBlocks * blockSize, blockSize);
    }

    // Launch torsion kernel
    if (numTorsions > 0) {
        int totalTorsionWork = numParticleGroups * numTorsions;
        int numBlocks = (totalTorsionWork + blockSize - 1) / blockSize;

        CUdeviceptr torsionAtomsPtr = torsionAtoms.getDevicePointer();
        CUdeviceptr torsionParamsPtr = torsionParams.getDevicePointer();

        void* args[] = {
            &posqPtr, &forcePtr, &energyPtr,
            &groupIndicesPtr, &torsionAtomsPtr, &torsionParamsPtr,
            &groupEnergiesPtr, &groupScalingPtr,
            &globalScalingFactor, &numAtoms, &numTorsions,
            &numParticleGroups, &paddedNumAtoms, &includeEnergy
        };
        cu.executeKernel(torsionKernel, args, numBlocks * blockSize, blockSize);
    }

    // Download per-group energies (only when energy is needed to avoid sync barriers)
    if (includeEnergy && !skipGroupEnergyDownload_)
        groupEnergiesBuffer.download(groupEnergiesHost);

    return 0.0;  // Energy accumulated in energy buffer
}

double CudaCalcIsolatedBondedForceKernel::getGroupEnergy(int groupIndex) const {
    if (groupIndex < 0 || groupIndex >= numParticleGroups)
        throw OpenMMException("IsolatedBondedForce: group index out of range");
    return static_cast<double>(groupEnergiesHost[groupIndex]);
}

void CudaCalcIsolatedBondedForceKernel::copyParametersToContext(ContextImpl& context, const IsolatedBondedForce& force) {
    if (numAtoms != force.getNumAtoms())
        throw OpenMMException("Cannot update IsolatedBondedForce: number of atoms has changed");

    // Update bond parameters
    if (numBonds > 0) {
        vector<float2> h_bondParams(numBonds);
        for (int i = 0; i < numBonds; i++) {
            int a1, a2;
            double length, k;
            force.getBondParameters(i, a1, a2, length, k);
            h_bondParams[i] = make_float2((float)length, (float)k);
        }
        bondParams.upload(h_bondParams);
    }

    // Update angle parameters
    if (numAngles > 0) {
        vector<float2> h_angleParams(numAngles);
        for (int i = 0; i < numAngles; i++) {
            int a1, a2, a3;
            double angle, k;
            force.getAngleParameters(i, a1, a2, a3, angle, k);
            h_angleParams[i] = make_float2((float)angle, (float)k);
        }
        angleParams.upload(h_angleParams);
    }

    // Update torsion parameters
    if (numTorsions > 0) {
        vector<float4> h_torsionParams(numTorsions);
        for (int i = 0; i < numTorsions; i++) {
            int a1, a2, a3, a4, periodicity;
            double phase, k;
            force.getTorsionParameters(i, a1, a2, a3, a4, periodicity, phase, k);
            h_torsionParams[i] = make_float4((float)periodicity, (float)phase, (float)k, 0.0f);
        }
        torsionParams.upload(h_torsionParams);
    }

    // Update scaling factors
    globalScalingFactor = static_cast<float>(force.getGlobalScalingFactor());
    int nGroups = force.getNumParticleGroups();
    if (nGroups > 0) {
        vector<float> h_groupScalings(numParticleGroups);
        for (int g = 0; g < nGroups; g++)
            h_groupScalings[g] = static_cast<float>(force.getGroupScalingFactor(g));
        groupScalingFactorsBuffer.upload(h_groupScalings);
    }

    cu.invalidateMolecules();
}

vector<double> CudaCalcIsolatedBondedForceKernel::computeHessian(ContextImpl& context, int groupIndex) {
    if (groupIndex < 0 || groupIndex >= numParticleGroups)
        throw OpenMMException("IsolatedBondedForce: group index out of range for computeHessian");

    // Download positions from GPU
    int totalSystemAtoms = cu.getNumAtoms();
    vector<float4> posq(cu.getPaddedNumAtoms());
    cu.getPosq().download(posq);

    // Extract group particle positions as Vec3
    vector<Vec3> groupPos(numAtoms);
    for (int i = 0; i < numAtoms; i++) {
        int sysIdx = h_particleIndices[groupIndex * numAtoms + i];
        groupPos[i] = Vec3(posq[sysIdx].x, posq[sysIdx].y, posq[sysIdx].z);
    }

    int N3 = 3 * numAtoms;
    vector<double> H(N3 * N3, 0.0);

    // Bond Hessians
    for (int b = 0; b < numBonds; b++) {
        int i = h_bonds[b].atom1;
        int j = h_bonds[b].atom2;
        double Hii[9], Hij[9];
        BondedHessianAnalytical::computeBondHessianBlock(
            groupPos[i], groupPos[j], h_bonds[b].k, h_bonds[b].length, Hii, Hij);
        BondedHessianAnalytical::addBlock(H, N3, i, i, Hii);
        BondedHessianAnalytical::addBlock(H, N3, j, j, Hii);
        BondedHessianAnalytical::addBlock(H, N3, i, j, Hij);
        BondedHessianAnalytical::addBlock(H, N3, j, i, Hij);
    }

    // Angle Hessians
    for (int a = 0; a < numAngles; a++) {
        int i = h_angles[a].atom1;
        int j = h_angles[a].atom2;
        int k = h_angles[a].atom3;
        double Ha[9][9];
        BondedHessianAnalytical::computeAngleHessian(
            groupPos[i], groupPos[j], groupPos[k],
            h_angles[a].k, h_angles[a].angle, Ha);
        int atoms[3] = {i, j, k};
        for (int ai = 0; ai < 3; ai++)
            for (int aj = 0; aj < 3; aj++)
                for (int di = 0; di < 3; di++)
                    for (int dj = 0; dj < 3; dj++)
                        H[(3*atoms[ai] + di) * N3 + (3*atoms[aj] + dj)] += Ha[3*ai + di][3*aj + dj];
    }

    // Torsion Hessians
    for (int t = 0; t < numTorsions; t++) {
        int i = h_torsions[t].atom1;
        int j = h_torsions[t].atom2;
        int k = h_torsions[t].atom3;
        int l = h_torsions[t].atom4;
        double Ht[12][12];
        BondedHessianAnalytical::computeTorsionHessian(
            groupPos[i], groupPos[j], groupPos[k], groupPos[l],
            h_torsions[t].periodicity, h_torsions[t].phase, h_torsions[t].k, Ht);
        int atoms[4] = {i, j, k, l};
        for (int ai = 0; ai < 4; ai++)
            for (int aj = 0; aj < 4; aj++)
                for (int di = 0; di < 3; di++)
                    for (int dj = 0; dj < 3; dj++)
                        H[(3*atoms[ai] + di) * N3 + (3*atoms[aj] + dj)] += Ht[3*ai + di][3*aj + dj];
    }

    // Symmetrize
    for (int i = 0; i < N3; i++) {
        for (int j = i + 1; j < N3; j++) {
            double avg = 0.5 * (H[i * N3 + j] + H[j * N3 + i]);
            H[i * N3 + j] = avg;
            H[j * N3 + i] = avg;
        }
    }

    return H;
}

vector<double> CudaCalcIsolatedBondedForceKernel::computeInternalForceConstants(ContextImpl& context, int groupIndex) {
    if (groupIndex < 0 || groupIndex >= numParticleGroups)
        throw OpenMMException("IsolatedBondedForce: group index out of range for computeInternalForceConstants");

    // Download positions from GPU
    vector<float4> posq(cu.getPaddedNumAtoms());
    cu.getPosq().download(posq);

    // Extract group particle positions as Vec3
    vector<Vec3> groupPos(numAtoms);
    for (int i = 0; i < numAtoms; i++) {
        int sysIdx = h_particleIndices[groupIndex * numAtoms + i];
        groupPos[i] = Vec3(posq[sysIdx].x, posq[sysIdx].y, posq[sysIdx].z);
    }

    int total = numBonds + numAngles + numTorsions;
    vector<double> constants(total);
    int idx = 0;

    for (int b = 0; b < numBonds; b++)
        constants[idx++] = h_bonds[b].k;

    for (int a = 0; a < numAngles; a++)
        constants[idx++] = h_angles[a].k;

    for (int t = 0; t < numTorsions; t++) {
        double phi = BondedHessianAnalytical::computeDihedralAngle(
            groupPos[h_torsions[t].atom1], groupPos[h_torsions[t].atom2],
            groupPos[h_torsions[t].atom3], groupPos[h_torsions[t].atom4]);
        int n = h_torsions[t].periodicity;
        constants[idx++] = -h_torsions[t].k * n * n * cos(n * phi - h_torsions[t].phase);
    }

    return constants;
}
