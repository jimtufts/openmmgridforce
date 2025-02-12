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
    
    // Create CUDA arrays and upload data
    g_counts_cuda.initialize<int>(context, 3, "gridCounts");
    g_spacing_cuda.initialize<float>(context, 3, "gridSpacing");
    g_vals_cuda.initialize<float>(context, g_vals.size(), "gridValues");
    g_scaling_factors_cuda.initialize<float>(context, g_scaling_factors.size(), "scalingFactors");
    
    // Copy data to GPU
    vector<int> countsVec = {g_counts[0], g_counts[1], g_counts[2]};
    vector<float> spacingVec = {(float)g_spacing[0], (float)g_spacing[1], (float)g_spacing[2]};
    vector<float> valsFloat(g_vals.begin(), g_vals.end());
    vector<float> scalingFloat(g_scaling_factors.begin(), g_scaling_factors.end());
    
    g_counts_cuda.upload(countsVec);
    g_spacing_cuda.upload(spacingVec);
    g_vals_cuda.upload(valsFloat);
    g_scaling_factors_cuda.upload(scalingFloat);
    
    // Define grid dimensions for optimal occupancy
    numAtoms = g_scaling_factors.size();
    map<string, string> defines;
    defines["GRID_SIZE_X"] = context.intToString(g_counts[0]);
    defines["GRID_SIZE_Y"] = context.intToString(g_counts[1]);
    defines["GRID_SIZE_Z"] = context.intToString(g_counts[2]);
    defines["NUM_ATOMS"] = context.intToString(numAtoms);
    
    // Create CUDA program
    CUmodule program = context.createModule(CudaGridForceKernelSources::gridForce, defines);
    kernel = context.getKernel(program, "computeGridForce");
}

double CudaCalcGridForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    if (!hasInitializedKernel) {
        hasInitializedKernel = true;
        initialize(context.getSystem(), dynamic_cast<const GridForce&>(context.getSystem().getForce(0)));
    }
    
    void* args[] = {&context.getPosq().getDevicePointer(),
                    &context.getForce().getDevicePointer(),
                    &g_counts_cuda.getDevicePointer(),
                    &g_spacing_cuda.getDevicePointer(),
                    &g_vals_cuda.getDevicePointer(),
                    &g_scaling_factors_cuda.getDevicePointer(),
                    &includeEnergy,
                    &context.getEnergyBuffer().getDevicePointer()};
                    
    int gridSize = min(context.getNumAtoms(), 256);
    int blockSize = (numAtoms + gridSize - 1)/gridSize;
    context.executeKernel(kernel, args, gridSize, blockSize);
    
    // Return energy
    if (includeEnergy) {
        double energy;
        context.getEnergyBuffer().download(&energy);
        return energy;
    }
    return 0.0;
}

void CudaCalcGridForceKernel::copyParametersToContext(ContextImpl& context, const GridForce& force) {
    if (numAtoms != force.getNumParticles())
        throw OpenMMException("updateParametersInContext: The number of particles has changed");
        
    // Get updated parameters
    vector<int> g_counts;
    vector<double> g_spacing;
    vector<double> g_vals;
    vector<double> g_scaling_factors;
    force.getGridParameters(g_counts, g_spacing, g_vals, g_scaling_factors);
    
    // Update CUDA arrays
    vector<int> countsVec = {g_counts[0], g_counts[1], g_counts[2]};
    vector<float> spacingVec = {(float)g_spacing[0], (float)g_spacing[1], (float)g_spacing[2]};
    vector<float> valsFloat(g_vals.begin(), g_vals.end());
    vector<float> scalingFloat(g_scaling_factors.begin(), g_scaling_factors.end());
    
    g_counts_cuda.upload(countsVec);
    g_spacing_cuda.upload(spacingVec);
    g_vals_cuda.upload(valsFloat);
    g_scaling_factors_cuda.upload(scalingFloat);
}
