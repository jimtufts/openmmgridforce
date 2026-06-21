#ifndef CPU_MULTIGROUPHMC_KERNELS_H_
#define CPU_MULTIGROUPHMC_KERNELS_H_

/* -------------------------------------------------------------------------- *
 *                              OpenMMGridForce                               *
 * -------------------------------------------------------------------------- *
 * CPU platform implementation of MultiGroupHMCIntegrator kernel.             *
 * Distributes the per-group work across the platform thread pool; per-group  *
 * RNG streams keep results identical for any thread count.                   *
 * -------------------------------------------------------------------------- */

#include "ReferenceMultiGroupHMCKernels.h"

namespace GridForcePlugin {

class CpuIntegrateMultiGroupHMCStepKernel : public ReferenceIntegrateMultiGroupHMCStepKernel {
public:
    CpuIntegrateMultiGroupHMCStepKernel(std::string name, const OpenMM::Platform& platform)
        : ReferenceIntegrateMultiGroupHMCStepKernel(name, platform) {}

protected:
    void forEachGroup(OpenMM::ContextImpl& context,
                      const std::function<void(int)>& body) override;
};

}  // namespace GridForcePlugin

#endif /* CPU_MULTIGROUPHMC_KERNELS_H_ */
