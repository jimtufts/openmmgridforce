#ifndef OPENMM_CUDA_ISOLATEDSITEFORCE_KERNELS_H_
#define OPENMM_CUDA_ISOLATEDSITEFORCE_KERNELS_H_

/* -------------------------------------------------------------------------- *
 *                              OpenMMGridForce                               *
 * -------------------------------------------------------------------------- */

#include "IsolatedSiteForceKernels.h"
#include "openmm/cuda/CudaContext.h"
#include "openmm/cuda/CudaArray.h"
#include <vector>

namespace GridForcePlugin {

class CudaCalcIsolatedSiteForceKernel : public CalcIsolatedSiteForceKernel {
public:
    CudaCalcIsolatedSiteForceKernel(std::string name,
                                     const OpenMM::Platform& platform,
                                     OpenMM::CudaContext& cu) :
        CalcIsolatedSiteForceKernel(name, platform),
        hasInitializedKernel(false), cu(cu) {}

    ~CudaCalcIsolatedSiteForceKernel();

    void initialize(const OpenMM::System& system,
                   const IsolatedSiteForce& force) override;
    double execute(OpenMM::ContextImpl& context,
                  bool includeForces, bool includeEnergy) override;
    void copyParametersToContext(OpenMM::ContextImpl& context,
                                const IsolatedSiteForce& force) override;
    double getGroupEnergy(int groupIndex) const override;

private:
    bool hasInitializedKernel;
    int numAtoms;
    int numParticleGroups;
    OpenMM::CudaContext& cu;

    // CUDA kernel function
    CUfunction siteKernel;

    // GPU buffers
    OpenMM::CudaArray groupParticleIndices;   // [numGroups * numAtoms] int
    OpenMM::CudaArray atomMasses;             // [numAtoms] float
    OpenMM::CudaArray groupEnergiesBuffer;    // [numGroups] float
    OpenMM::CudaArray fixedPointEnergyBuffer; // [1] unsigned long long - fixed-point energy accumulator
    OpenMM::CudaArray groupScalingFactorsBuffer;  // [numGroups] float

    // Site geometry (passed as kernel args in double; positions/COM use the
    // context real type, so the restraint honors the selected precision).
    double centerX, centerY, centerZ;
    double maxRadius;
    double forceConstantVal;
    float globalScalingFactor;
    double totalMass;

    // Host-side caches
    mutable std::vector<double> groupEnergiesHost;
    std::vector<int> h_particleIndices;

    bool skipGroupEnergyDownload_ = false;
public:
    void setSkipGroupEnergyDownload(bool skip) override { skipGroupEnergyDownload_ = skip; }
    void* getGroupEnergyDevicePointer() override {
        return groupEnergiesBuffer.isInitialized()
            ? (void*)groupEnergiesBuffer.getDevicePointer() : nullptr;
    }
};

}  // namespace GridForcePlugin

#endif /*OPENMM_CUDA_ISOLATEDSITEFORCE_KERNELS_H_*/
