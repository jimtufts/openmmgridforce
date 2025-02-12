#include "CudaGridForceKernels.h"
#include "CudaGridForceKernelSources.h"
#include "openmm/internal/ContextImpl.h"
#include <map>

using namespace GridForcePlugin;
using namespace OpenMM;
using namespace std;

void CudaCalcGridForceKernel::initialize(const System& system, const GridForce& force) {
    // Get grid parameters
    vector<int> g_counts;
    vector<double> g_spacing;
    vector<double> g_vals;
    vector<double> g_scaling_factors;
    force.getGridParameters(g_counts, g_spacing, g_vals, g_scaling_factors);
    
    numAtoms = system.getNumParticles();
    
    // Initialize arrays
    this->g_counts.initialize<int>(context, 3, "gridCounts");
    this->g_spacing.initialize<float>(context, 3, "gridSpacing");
    this->g_vals.initialize<float>(context, g_vals.size(), "gridValues");
    this->g_scaling_factors.initialize<float>(context, g_scaling_factors.size(), "scalingFactors");
    
    // Copy data to GPU
    vector<int> countsVec = {g_counts[0], g_counts[1], g_counts[2]};
    vector<float> spacingVec = {(float)g_spacing[0], (float)g_spacing[1], (float)g_spacing[2]};
    vector<float> valsFloat(g_vals.begin(), g_vals.end());
    vector<float> scalingFloat(g_scaling_factors.begin(), g_scaling_factors.end());
    
    this->g_counts.upload(countsVec);
    this->g_spacing.upload(spacingVec);
    this->g_vals.upload(valsFloat);
    this->g_scaling_factors.upload(scalingFloat);
    
    // Create kernel
    map<string, string> defines;
    defines["GRID_SIZE_X"] = context.intToString(g_counts[0]);
    defines["GRID_SIZE_Y"] = context.intToString(g_counts[1]);
    defines["GRID_SIZE_Z"] = context.intToString(g_counts[2]);
    defines["NUM_ATOMS"] = context.intToString(numAtoms);
    
    CUmodule module = context.createModule(CudaGridForceKernelSources::gridForce, defines);
    kernel = context.getKernel(module, "computeGridForce");
    
    hasInitializedKernel = true;
}

double CudaCalcGridForceKernel::execute(ContextImpl& contextImpl, bool includeForces, bool includeEnergy) {
    if (!hasInitializedKernel) {
        initialize(contextImpl.getSystem(), dynamic_cast<const GridForce&>(contextImpl.getSystem().getForce(0)));
    }
    
    void* args[] = {&context.getPosq().getDevicePointer(),
                    &context.getForce().getDevicePointer(),
                    &g_counts.getDevicePointer(),
                    &g_spacing.getDevicePointer(),
                    &g_vals.getDevicePointer(),
                    &g_scaling_factors.getDevicePointer(),
                    &includeEnergy,
                    &context.getEnergyBuffer().getDevicePointer()};
                    
    int threads = min(numAtoms, 256);
    int blocks = (numAtoms + threads - 1)/threads;
    context.executeKernel(kernel, args, threads, blocks);
    
    // Return energy
    if (includeEnergy) {
        double energy;
        context.getEnergyBuffer().download(&energy);
        return energy;
    }
    return 0.0;
}

void CudaCalcGridForceKernel::copyParametersToContext(ContextImpl& contextImpl, const GridForce& force) {
    // Get updated parameters
    vector<int> g_counts;
    vector<double> g_spacing;
    vector<double> g_vals;
    vector<double> g_scaling_factors;
    force.getGridParameters(g_counts, g_spacing, g_vals, g_scaling_factors);
    
    if (numAtoms != contextImpl.getSystem().getNumParticles())
        throw OpenMMException("updateParametersInContext: The number of particles has changed");
    
    // Update CUDA arrays
    vector<int> countsVec = {g_counts[0], g_counts[1], g_counts[2]};
    vector<float> spacingVec = {(float)g_spacing[0], (float)g_spacing[1], (float)g_spacing[2]};
    vector<float> valsFloat(g_vals.begin(), g_vals.end());
    vector<float> scalingFloat(g_scaling_factors.begin(), g_scaling_factors.end());
    
    this->g_counts.upload(countsVec);
    this->g_spacing.upload(spacingVec);
    this->g_vals.upload(valsFloat);
    this->g_scaling_factors.upload(scalingFloat);
}
