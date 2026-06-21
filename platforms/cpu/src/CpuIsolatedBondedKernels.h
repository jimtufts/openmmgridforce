#ifndef CPU_ISOLATED_BONDED_KERNELS_H_
#define CPU_ISOLATED_BONDED_KERNELS_H_

/* -------------------------------------------------------------------------- *
 *                              OpenMMGridForce                               *
 * -------------------------------------------------------------------------- *
 * CPU platform implementation of IsolatedBondedForce kernel.                 *
 * Distributes the particle groups across the platform thread pool; the       *
 * per-group physics is inherited from the Reference kernel.                  *
 * -------------------------------------------------------------------------- */

#include "ReferenceIsolatedBondedKernels.h"

namespace GridForcePlugin {

class CpuCalcIsolatedBondedForceKernel : public ReferenceCalcIsolatedBondedForceKernel {
public:
    CpuCalcIsolatedBondedForceKernel(std::string name, const OpenMM::Platform& platform)
        : ReferenceCalcIsolatedBondedForceKernel(name, platform) {}

protected:
    void runGroups(OpenMM::ContextImpl& context,
                   std::vector<OpenMM::Vec3>& posData,
                   std::vector<OpenMM::Vec3>& forceData,
                   bool includeForces, bool includeEnergy) override;
};

}  // namespace GridForcePlugin

#endif /* CPU_ISOLATED_BONDED_KERNELS_H_ */
