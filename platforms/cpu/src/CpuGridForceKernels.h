#ifndef CPU_GRIDFORCE_KERNELS_H_
#define CPU_GRIDFORCE_KERNELS_H_

/* -------------------------------------------------------------------------- *
 *                              OpenMMGridForce                               *
 * -------------------------------------------------------------------------- *
 * CPU platform implementation of GridForce kernel.                           *
 * Distributes the ligand atoms across the platform thread pool; the          *
 * per-atom physics is inherited from the Reference kernel.                   *
 * -------------------------------------------------------------------------- */

#include "ReferenceGridForceKernels.h"

namespace GridForcePlugin {

class CpuCalcGridForceKernel : public ReferenceCalcGridForceKernel {
public:
    CpuCalcGridForceKernel(std::string name, const OpenMM::Platform& platform)
        : ReferenceCalcGridForceKernel(name, platform) {}

protected:
    void runAtoms(OpenMM::ContextImpl& context,
                  std::vector<OpenMM::Vec3>& posData,
                  std::vector<OpenMM::Vec3>& forceData,
                  bool includeForces, bool includeEnergy) override;
};

}  // namespace GridForcePlugin

#endif /* CPU_GRIDFORCE_KERNELS_H_ */
