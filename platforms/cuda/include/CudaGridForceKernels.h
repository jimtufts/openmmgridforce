#ifndef GRIDFORCE_CUDAKERNELS_H_
#define GRIDFORCE_CUDAKERNELS_H_

#include "GridForceKernels.h"
#include "openmm/cuda/CudaPlatform.h"
#include "openmm/cuda/CudaContext.h"
#include "openmm/cuda/CudaArray.h"
#include <vector>

using namespace OpenMM;
using namespace std;

namespace GridForcePlugin {
  
class CudaCalcGridForceKernel : public CalcGridForceKernel {
public:
    CudaCalcGridForceKernel(std::string name, const Platform& platform, CudaContext& cc) : 
        CalcGridForceKernel(name, platform), context(cc), hasInitializedKernel(false), energyBuffer(NULL) {
    }
    
    ~CudaCalcGridForceKernel() {
        if (energyBuffer != NULL)
            delete energyBuffer;
    }
    
    void initialize(const System& system, const GridForce& force);
    double execute(ContextImpl& context, bool includeForces, bool includeEnergy);
    void copyParametersToContext(ContextImpl& context, const GridForce& force);

private:
    void initializeArrays(const vector<int>& counts, const vector<double>& spacing, 
                         const vector<double>& vals, const vector<double>& scaling_factors);
    bool arraysAreInitialized() const {
        return g_counts.isInitialized() && g_spacing.isInitialized() && 
               g_vals.isInitialized() && g_scaling_factors.isInitialized();
    }

    CudaContext& context;
    bool hasInitializedKernel;
    CudaArray g_counts;
    CudaArray g_spacing;
    CudaArray g_vals;
    CudaArray g_scaling_factors;
    CudaArray* energyBuffer;
    CUfunction kernel;
    int numAtoms;
    double g_inv_power;
};

} // namespace GridForcePlugin

#endif /* GRIDFORCE_CUDAKERNELS_H_ */
