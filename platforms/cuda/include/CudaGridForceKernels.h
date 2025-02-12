#ifndef GRIDFORCE_CUDAKERNELS_H_
#define GRIDFORCE_CUDAKERNELS_H_

#include "GridForceKernels.h"
#include "CommonGridForceKernels.h"
#include "openmm/cuda/CudaPlatform.h"
#include "openmm/cuda/CudaContext.h"

using namespace OpenMM;

namespace GridForcePlugin {
  
class CudaCalcGridForceKernel : public CalcGridForceKernel {
public:
    CudaCalcGridForceKernel(std::string name, const Platform& platform, CudaContext& cc) : 
        CalcGridForceKernel(name, platform), hasInitializedKernel(false), context(cc) {
    }
    /**
     * Initialize the kernel.
     * 
     * @param system     the System this kernel will be applied to
     * @param force      the GridForce this kernel will be used for
     */
    void initialize(const OpenMM::System& system, const GridForce& force);
    /**
     * Execute the kernel to calculate the forces and/or energy.
     *
     * @param context        the context in which to execute this kernel
     * @param includeForces  true if forces should be calculated
     * @param includeEnergy  true if the energy should be calculated
     * @return the potential energy due to the force
     */
    double execute(OpenMM::ContextImpl& context, bool includeForces, bool includeEnergy);
    /**
     * Copy changed parameters over to a context.
     *
     * @param context    the context to copy parameters to
     * @param force      the GridForce to copy the parameters from
     */
    void copyParametersToContext(OpenMM::ContextImpl& context, const GridForce& force);

private:
    bool hasInitializedKernel;
    CudaContext& context;
    CudaArray g_counts;
    CudaArray g_spacing;
    CudaArray g_vals;
    CudaArray g_scaling_factors;
};

} // namespace GridForcePlugin

#endif /* GRIDFORCE_CUDAKERNELS_H_ */
