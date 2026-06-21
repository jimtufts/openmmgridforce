#ifndef CPU_ISOLATED_GBSA_KERNELS_H_
#define CPU_ISOLATED_GBSA_KERNELS_H_

/* -------------------------------------------------------------------------- *
 *                              OpenMMGridForce                               *
 * -------------------------------------------------------------------------- *
 * CPU platform implementation of IsolatedGBSAForce kernel.                   *
 * Distributes the particle groups across the platform thread pool; the       *
 * per-group physics is inherited from the Reference kernel.                  *
 * -------------------------------------------------------------------------- */

#include "ReferenceIsolatedGBSAKernels.h"

namespace GridForcePlugin {

class CpuCalcIsolatedGBSAForceKernel : public ReferenceCalcIsolatedGBSAForceKernel {
public:
    CpuCalcIsolatedGBSAForceKernel(std::string name, const OpenMM::Platform& platform)
        : ReferenceCalcIsolatedGBSAForceKernel(name, platform) {}

protected:
    void runGroups(OpenMM::ContextImpl& context,
                   std::vector<OpenMM::Vec3>& posData,
                   std::vector<OpenMM::Vec3>& forceData,
                   bool includeForces, bool includeEnergy) override;
};

}  // namespace GridForcePlugin

#endif /* CPU_ISOLATED_GBSA_KERNELS_H_ */
