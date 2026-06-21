#ifndef CPU_ISOLATED_NONBONDED_KERNELS_H_
#define CPU_ISOLATED_NONBONDED_KERNELS_H_

/* -------------------------------------------------------------------------- *
 *                              OpenMMGridForce                               *
 * -------------------------------------------------------------------------- *
 * CPU platform implementation of IsolatedNonbondedForce kernel.              *
 * Distributes the particle groups across the platform thread pool; the       *
 * per-group physics is inherited from the Reference kernel.                  *
 * -------------------------------------------------------------------------- */

#include "ReferenceIsolatedNonbondedKernels.h"

namespace GridForcePlugin {

class CpuCalcIsolatedNonbondedForceKernel : public ReferenceCalcIsolatedNonbondedForceKernel {
public:
    CpuCalcIsolatedNonbondedForceKernel(std::string name, const OpenMM::Platform& platform)
        : ReferenceCalcIsolatedNonbondedForceKernel(name, platform) {}

protected:
    void runGroups(OpenMM::ContextImpl& context,
                   std::vector<OpenMM::Vec3>& posData,
                   std::vector<OpenMM::Vec3>& forceData,
                   bool includeForces, bool includeEnergy) override;
};

}  // namespace GridForcePlugin

#endif /* CPU_ISOLATED_NONBONDED_KERNELS_H_ */
