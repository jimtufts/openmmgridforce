/* -------------------------------------------------------------------------- *
 *                              OpenMMGridForce                               *
 * -------------------------------------------------------------------------- */

#include "CudaIsolatedSiteKernels.h"
#include "CudaGridForceKernelSources.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/OpenMMException.h"
#include "openmm/cuda/CudaContext.h"
#include "openmm/cuda/CudaArray.h"

using namespace OpenMM;
using namespace std;

// Energy buffers follow the context precision (mixed = double in mixed/double).
static int mixedEnergyElementSize(OpenMM::CudaContext& cu) {
    return (cu.getUseDoublePrecision() || cu.getUseMixedPrecision()) ? sizeof(double) : sizeof(float);
}
static void initMixedEnergyBuffer(OpenMM::CudaContext& cu, OpenMM::CudaArray& buf, int n, const char* name) {
    buf.initialize(cu, n, mixedEnergyElementSize(cu), name);
}
static void downloadMixedEnergy(OpenMM::CudaContext& cu, OpenMM::CudaArray& buf, std::vector<double>& out) {
    if (cu.getUseDoublePrecision() || cu.getUseMixedPrecision()) {
        buf.download(out);
    } else {
        std::vector<float> tmp(out.size());
        buf.download(tmp);
        out.assign(tmp.begin(), tmp.end());
    }
}

namespace GridForcePlugin {

CudaCalcIsolatedSiteForceKernel::~CudaCalcIsolatedSiteForceKernel() {
}

void CudaCalcIsolatedSiteForceKernel::initialize(
        const System& system, const IsolatedSiteForce& force) {

    cu.setAsCurrent();

    numAtoms = force.getNumAtoms();
    numParticleGroups = force.getNumParticleGroups();

    // Site parameters
    double cx, cy, cz;
    force.getSiteCenter(cx, cy, cz);
    centerX = cx;
    centerY = cy;
    centerZ = cz;
    maxRadius = force.getMaxRadius();
    forceConstantVal = force.getForceConstant();
    globalScalingFactor = (float)force.getGlobalScalingFactor();

    // Atom masses
    const vector<double>& massesD = force.getAtomMasses();
    vector<float> massesF(numAtoms);
    totalMass = 0.0;
    for (int i = 0; i < numAtoms; i++) {
        massesF[i] = (float)massesD[i];
        totalMass += massesD[i];
    }
    atomMasses.initialize<float>(cu, numAtoms, "isolatedSite_masses");
    atomMasses.upload(massesF);

    // Particle group indices
    h_particleIndices.resize(numParticleGroups * numAtoms);
    for (int g = 0; g < numParticleGroups; g++) {
        string name;
        vector<int> indices;
        force.getParticleGroup(g, name, indices);
        for (int i = 0; i < numAtoms; i++)
            h_particleIndices[g * numAtoms + i] = indices[i];
    }
    groupParticleIndices.initialize<int>(cu, numParticleGroups * numAtoms,
                                        "isolatedSite_groupParticleIndices");
    groupParticleIndices.upload(h_particleIndices);

    // Per-group energy buffer
    initMixedEnergyBuffer(cu, groupEnergiesBuffer, numParticleGroups, "isolatedSite_groupEnergies");
    groupEnergiesHost.resize(numParticleGroups, 0.0f);

    // Fixed-point energy accumulator (for pre-sm_60 GPU compatibility)
    fixedPointEnergyBuffer.initialize<unsigned long long>(cu, 1, "isolatedSite_fixedPointEnergy");

    // Per-group scaling factors
    vector<float> scalingFactors(numParticleGroups);
    for (int g = 0; g < numParticleGroups; g++)
        scalingFactors[g] = (float)force.getGroupScalingFactor(g);
    groupScalingFactorsBuffer.initialize<float>(cu, numParticleGroups,
                                                "isolatedSite_groupScalingFactors");
    groupScalingFactorsBuffer.upload(scalingFactors);

    // Compile CUDA kernel
    CUmodule module = cu.createModule(
        CudaGridForceKernelSources::commonHeaders +
        CudaGridForceKernelSources::isolatedSiteKernel);
    siteKernel = cu.getKernel(module, "computeIsolatedSiteRestraint");

    hasInitializedKernel = true;
}

double CudaCalcIsolatedSiteForceKernel::execute(
        ContextImpl& context, bool includeForces, bool includeEnergy) {

    if (!hasInitializedKernel) return 0.0;

    // Clear per-group energies and fixed-point accumulator
    if (includeEnergy) {
        cu.clearBuffer(groupEnergiesBuffer);
        cu.clearBuffer(fixedPointEnergyBuffer);
    }

    int paddedNumAtoms = cu.getPaddedNumAtoms();
    CUdeviceptr posqPtr = cu.getPosq().getDevicePointer();
    CUdeviceptr forcePtr = cu.getLongForceBuffer().getDevicePointer();
    CUdeviceptr fixedPointEnergyPtr = fixedPointEnergyBuffer.getDevicePointer();
    CUdeviceptr groupIndPtr = groupParticleIndices.getDevicePointer();
    CUdeviceptr massPtr = atomMasses.getDevicePointer();
    CUdeviceptr groupEPtr = groupEnergiesBuffer.getDevicePointer();
    CUdeviceptr groupSPtr = groupScalingFactorsBuffer.getDevicePointer();

    int blockSize = 128;
    // Grid-stride loop handles any totalWork, so we just need enough blocks
    // to saturate the GPU. The kernel loops over all (group, atom) pairs.
    int totalWork = numParticleGroups * numAtoms;
    int numBlocks = min((totalWork + blockSize - 1) / blockSize,
                        cu.getNumThreadBlocks());

    void* args[] = {
        &posqPtr, &forcePtr, &fixedPointEnergyPtr,
        &groupIndPtr, &massPtr, &groupEPtr, &groupSPtr,
        &globalScalingFactor,
        &centerX, &centerY, &centerZ,
        &maxRadius, &forceConstantVal, &totalMass,
        &numAtoms, &numParticleGroups, &paddedNumAtoms,
        &includeEnergy
    };
    cu.executeKernel(siteKernel, args, numBlocks * blockSize, blockSize);

    // Convert fixed-point energy and return
    if (includeEnergy) {
        unsigned long long fixedPointEnergyRaw;
        fixedPointEnergyBuffer.download(&fixedPointEnergyRaw);
        double energy = (long long)fixedPointEnergyRaw / (double)0x100000000;
        if (!skipGroupEnergyDownload_)
            downloadMixedEnergy(cu, groupEnergiesBuffer, groupEnergiesHost);
        return energy;
    }

    return 0.0;
}

void CudaCalcIsolatedSiteForceKernel::copyParametersToContext(
        ContextImpl& context, const IsolatedSiteForce& force) {

    cu.setAsCurrent();

    double cx, cy, cz;
    force.getSiteCenter(cx, cy, cz);
    centerX = cx;
    centerY = cy;
    centerZ = cz;
    maxRadius = force.getMaxRadius();
    forceConstantVal = force.getForceConstant();
    globalScalingFactor = (float)force.getGlobalScalingFactor();

    // Update scaling factors
    vector<float> scalingFactors(numParticleGroups);
    for (int g = 0; g < numParticleGroups; g++)
        scalingFactors[g] = (float)force.getGroupScalingFactor(g);
    groupScalingFactorsBuffer.upload(scalingFactors);
}

double CudaCalcIsolatedSiteForceKernel::getGroupEnergy(int groupIndex) const {
    if (groupIndex < 0 || groupIndex >= numParticleGroups)
        throw OpenMMException("IsolatedSiteForce: group index out of range");
    return static_cast<double>(groupEnergiesHost[groupIndex]);
}

}  // namespace GridForcePlugin
