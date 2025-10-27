#include "CudaGridForceKernels.h"
#include "CudaGridForceKernelSources.h"
#include "openmm/internal/ContextImpl.h"
#include <map>
#include <iostream>

using namespace GridForcePlugin;
using namespace OpenMM;
using namespace std;

void CudaCalcGridForceKernel::initializeArrays(const vector<int>& counts, 
                                              const vector<double>& spacing,
                                              const vector<double>& vals, 
                                              const vector<double>& scaling_factors) {
    // Clear existing arrays if they exist
    if (g_counts.isInitialized())
        g_counts.resize(0);
    if (g_spacing.isInitialized())
        g_spacing.resize(0);
    if (g_vals.isInitialized())
        g_vals.resize(0);
    if (g_scaling_factors.isInitialized())
        g_scaling_factors.resize(0);
        
    // Initialize arrays
    g_counts.initialize<int>(context, 3, "gridCounts");
    g_spacing.initialize<float>(context, 3, "gridSpacing");
    g_vals.initialize<float>(context, vals.size(), "gridValues");
    g_scaling_factors.initialize<float>(context, scaling_factors.size(), "scalingFactors");
    
    // Copy data to GPU
    vector<int> countsVec = {counts[0], counts[1], counts[2]};
    vector<float> spacingVec = {(float)spacing[0], (float)spacing[1], (float)spacing[2]};
    vector<float> valsFloat(vals.begin(), vals.end());
    vector<float> scalingFloat(scaling_factors.begin(), scaling_factors.end());
    
    g_counts.upload(countsVec);
    g_spacing.upload(spacingVec);
    g_vals.upload(valsFloat);
    g_scaling_factors.upload(scalingFloat);
}

void CudaCalcGridForceKernel::initialize(const System& system, const GridForce& force) {
    cout << "GridForceCUDA: Initializing CudaCalcGridForceKernel" << endl;

    // Get grid parameters
    vector<int> g_counts;
    vector<double> g_spacing;
    vector<double> g_vals;
    vector<double> g_scaling_factors;
    force.getGridParameters(g_counts, g_spacing, g_vals, g_scaling_factors);
    g_inv_power = force.getInvPower();
    
    numAtoms = system.getNumParticles();
    cout << "GridForceCUDA: System has " << numAtoms << " atoms" << endl;
    
    if (g_spacing.size() != 3 || g_counts.size() != 3) {
        throw OpenMMException("GridForce: Grid dimensions must be 3D");
    }
    
    if (g_vals.size() != g_counts[0] * g_counts[1] * g_counts[2]) {
        throw OpenMMException("GridForce: Number of grid values doesn't match grid dimensions");
    }
    
    if (g_scaling_factors.size() != numAtoms) {
        throw OpenMMException("GridForce: Number of scaling factors must match number of atoms");
    }
    
    if (!context.getContextIsValid()) {
        throw OpenMMException("CUDA context is not valid");
    }
    
    context.setAsCurrent();
    
    try {
        cout << "GridForceCUDA: Initializing arrays" << endl;
        cout << "GridForceCUDA: Grid dimensions: " << g_counts[0] << " x " << g_counts[1] << " x " << g_counts[2] << endl;
        
        // Initialize or reinitialize arrays
        initializeArrays(g_counts, g_spacing, g_vals, g_scaling_factors);
        
        cout << "GridForceCUDA: Creating kernel" << endl;
        // Create kernel
        map<string, string> defines;
        defines["GRID_SIZE_X"] = context.intToString(g_counts[0]);
        defines["GRID_SIZE_Y"] = context.intToString(g_counts[1]);
        defines["GRID_SIZE_Z"] = context.intToString(g_counts[2]);
        defines["NUM_ATOMS"] = context.intToString(numAtoms);
        
        CUmodule module = context.createModule(CudaGridForceKernelSources::gridForce, defines);
        kernel = context.getKernel(module, "computeGridForce");
        
        hasInitializedKernel = true;
        cout << "GridForceCUDA: Initialization complete" << endl;
    }
    catch (const exception& e) {
        cout << "GridForceCUDA: Error during initialization: " << e.what() << endl;
        throw;
    }
}

double CudaCalcGridForceKernel::execute(ContextImpl& contextImpl, bool includeForces, bool includeEnergy) {
    cout << "GridForceCUDA: Executing kernel" << endl;
    
    if (!hasInitializedKernel) {
        cout << "GridForceCUDA: First-time initialization" << endl;
        initialize(contextImpl.getSystem(), dynamic_cast<const GridForce&>(contextImpl.getSystem().getForce(0)));
    }
    
    if (!context.getContextIsValid()) {
        throw OpenMMException("CUDA context is not valid before kernel execution");
    }
    
    cout << "GridForceCUDA: Setting current context" << endl;
    context.setAsCurrent();
    
    if (numAtoms == 0) {
        return 0.0;
    }

    // Handle energy buffer
    if (includeEnergy) {
        cout << "GridForceCUDA: Setting up energy buffer" << endl;
        if (energyBuffer == NULL) {
            cout << "GridForceCUDA: Creating new energy buffer" << endl;
            energyBuffer = new CudaArray();
            energyBuffer->initialize<float>(context, 1, "gridForceEnergy");
        }
        float zero = 0.0f;
        energyBuffer->upload(&zero);
    }

    // Get device pointers with error checking
    cout << "GridForceCUDA: Getting device pointers" << endl;
    CUdeviceptr posPtr = context.getPosq().getDevicePointer();
    CUdeviceptr forcePtr = context.getForce().getDevicePointer();
    CUdeviceptr gridCountsPtr = g_counts.getDevicePointer();
    CUdeviceptr gridSpacingPtr = g_spacing.getDevicePointer();
    CUdeviceptr gridValsPtr = g_vals.getDevicePointer();
    CUdeviceptr scalingFactorsPtr = g_scaling_factors.getDevicePointer();
    CUdeviceptr energyPtr = (energyBuffer != NULL ? energyBuffer->getDevicePointer() : 0);

    // Check all pointers
    if (!posPtr || !forcePtr || !gridCountsPtr || !gridSpacingPtr || 
        !gridValsPtr || !scalingFactorsPtr || (includeEnergy && !energyPtr)) {
        throw OpenMMException("One or more required CUDA arrays not properly initialized");
    }
    
    float inv_power_float = (float)g_inv_power;
    void* args[] = {&posPtr,
                    &forcePtr,
                    &gridCountsPtr,
                    &gridSpacingPtr,
                    &gridValsPtr,
                    &scalingFactorsPtr,
                    &inv_power_float,
                    &includeEnergy,
                    &energyPtr};
    
    // Use a safe default value for max threads per block
    const int maxThreadsPerBlock = 256;  // This is a safe value for all CUDA devices
    int threads = min(numAtoms, maxThreadsPerBlock);
    int blocks = (numAtoms + threads - 1)/threads;
    
    cout << "GridForceCUDA: Launching kernel with " << blocks << " blocks and " << threads << " threads" << endl;
    cout << "GridForceCUDA: Number of atoms: " << numAtoms << endl;
    
    try {
        context.executeKernel(kernel, args, threads, blocks);
        cout << "GridForceCUDA: Kernel execution completed" << endl;
    }
    catch (const exception& e) {
        cout << "GridForceCUDA: Error during kernel execution: " << e.what() << endl;
        throw;
    }

    // Handle energy calculation
    if (includeEnergy && energyBuffer != NULL) {
        float energy = 0.0f;
        try {
            cout << "GridForceCUDA: Downloading energy buffer" << endl;
            energyBuffer->download(&energy);
            cout << "GridForceCUDA: Energy value: " << energy << endl;
            return energy;
        }
        catch (const exception& e) {
            cout << "GridForceCUDA: Error downloading energy: " << e.what() << endl;
            throw;
        }
    }
    return 0.0;
}

void CudaCalcGridForceKernel::copyParametersToContext(ContextImpl& contextImpl, const GridForce& force) {
    cout << "GridForceCUDA: Copying parameters to context" << endl;

    vector<int> g_counts;
    vector<double> g_spacing;
    vector<double> g_vals;
    vector<double> g_scaling_factors;
    force.getGridParameters(g_counts, g_spacing, g_vals, g_scaling_factors);
    g_inv_power = force.getInvPower();
    
    if (numAtoms != contextImpl.getSystem().getNumParticles())
        throw OpenMMException("updateParametersInContext: The number of particles has changed");
    
    if (g_spacing.size() != 3 || g_counts.size() != 3)
        throw OpenMMException("GridForce: Grid dimensions must be 3D");
    
    if (g_vals.size() != g_counts[0] * g_counts[1] * g_counts[2])
        throw OpenMMException("GridForce: Number of grid values doesn't match grid dimensions");
    
    if (g_scaling_factors.size() != numAtoms)
        throw OpenMMException("GridForce: Number of scaling factors must match number of atoms");
    
    context.setAsCurrent();
    
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
