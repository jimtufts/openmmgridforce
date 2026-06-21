#ifndef CPU_MULTIGROUPNUTS_KERNELS_H_
#define CPU_MULTIGROUPNUTS_KERNELS_H_

/* -------------------------------------------------------------------------- *
 *                              OpenMMGridForce                               *
 * -------------------------------------------------------------------------- *
 * CPU platform implementation of MultiGroupNUTSIntegrator kernel.            *
 * Distributes the per-group work across the platform thread pool; per-group  *
 * RNG streams keep results identical for any thread count.                   *
 * -------------------------------------------------------------------------- */

#include "ReferenceMultiGroupNUTSKernels.h"

namespace GridForcePlugin {

class CpuIntegrateMultiGroupNUTSStepKernel : public ReferenceIntegrateMultiGroupNUTSStepKernel {
public:
    CpuIntegrateMultiGroupNUTSStepKernel(std::string name, const OpenMM::Platform& platform)
        : ReferenceIntegrateMultiGroupNUTSStepKernel(name, platform) {}

protected:
    void forEachGroup(OpenMM::ContextImpl& context,
                      const std::function<void(int)>& body) override;
};

}  // namespace GridForcePlugin

#endif /* CPU_MULTIGROUPNUTS_KERNELS_H_ */
