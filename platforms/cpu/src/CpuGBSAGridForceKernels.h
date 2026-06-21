#ifndef CPU_GBSA_GRID_FORCE_KERNELS_H_
#define CPU_GBSA_GRID_FORCE_KERNELS_H_

/* -------------------------------------------------------------------------- *
 *                              OpenMMGridForce                               *
 * -------------------------------------------------------------------------- *
 * CPU platform implementation of GBSAGridForce kernel.                       *
 * Distributes the particle groups across the platform thread pool; the       *
 * per-group physics is inherited from the Reference kernel.                  *
 * -------------------------------------------------------------------------- */

#include "ReferenceGBSAGridForceKernels.h"

namespace GridForcePlugin {

class CpuCalcGBSAGridForceKernel : public ReferenceCalcGBSAGridForceKernel {
public:
    CpuCalcGBSAGridForceKernel(std::string name, const OpenMM::Platform& platform)
        : ReferenceCalcGBSAGridForceKernel(name, platform) {}

protected:
    void runGroups(OpenMM::ContextImpl& context,
                   std::vector<OpenMM::Vec3>& posData,
                   std::vector<OpenMM::Vec3>& forceData,
                   bool includeForces, bool includeEnergy) override;
};

}  // namespace GridForcePlugin

#endif /* CPU_GBSA_GRID_FORCE_KERNELS_H_ */
