#ifndef GRIDFORCE_CUDAKERNELS_H_
#define GRIDFORCE_CUDAKERNELS_H_

#include "GridForceKernels.h"
#include "openmm/cuda/CudaPlatform.h"
#include "openmm/cuda/CudaContext.h"
#include "openmm/cuda/CudaArray.h"

using namespace OpenMM;

namespace GridForcePlugin {
  
class CudaCalcGridForceKernel : public CalcGridForceKernel {
public:
    CudaCalcGridForceKernel(std::string name, const Platform& platform, CudaContext& cc) : 
        CalcGridForceKernel(name, platform), context(cc), hasInitializedKernel(false) {
    }
    
    virtual void initialize(const System& system, const GridForce& force);
    virtual double execute(ContextImpl& context, bool includeForces, bool includeEnergy);
    virtual void copyParametersToContext(ContextImpl& context, const GridForce& force);

private:
    CudaContext& context;
    bool hasInitializedKernel;
    CudaArray g_counts;
    CudaArray g_spacing;
    CudaArray g_vals;
    CudaArray g_scaling_factors;
    CUfunction kernel;
    int numAtoms;
};

} // namespace GridForcePlugin

#endif /* GRIDFORCE_CUDAKERNELS_H_ */
