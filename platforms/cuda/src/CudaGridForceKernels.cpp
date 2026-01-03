/* -------------------------------------------------------------------------- *
 *                              OpenMMGridForce                               *
 * -------------------------------------------------------------------------- */

#include "CudaGridForceKernels.h"
#include "CudaGridForceKernelSources.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/cuda/CudaContext.h"
#include "openmm/NonbondedForce.h"
#include <map>
#include <iostream>
#include <iomanip>
#include <string>
#include <cmath>
#include <algorithm>

using namespace GridForcePlugin;
using namespace OpenMM;
using namespace std;

CudaCalcGridForceKernel::~CudaCalcGridForceKernel() {
}

void CudaCalcGridForceKernel::initialize(const System& system, const GridForce& force) {
    cu.setAsCurrent();
    // Get grid parameters
    vector<int> counts_local;
    vector<double> spacing_local;
    vector<double> vals;
    vector<double> scaling_factors;
    force.getGridParameters(counts_local, spacing_local, vals, scaling_factors);
    double inv_power = force.getInvPower();
    double outOfBounds_k = force.getOutOfBoundsRestraint();
    double grid_cap = force.getGridCap();
    int interp_method = force.getInterpolationMethod();

    numAtoms = system.getNumParticles();

    // Store counts and spacing as member variables for generateGrid
    counts = counts_local;
    spacing = spacing_local;
//     std::cout << "CudaCalcGridForceKernel::initialize() received counts: ["
//               << counts[0] << ", " << counts[1] << ", " << counts[2] << "] = "
//               << (counts[0] * counts[1] * counts[2]) << " points" << std::endl;

    // Store grid cap, invPower, and invPowerMode BEFORE generateGrid() is called (it needs these values)
    gridCap = (float)grid_cap;
    invPower = (float)inv_power;
    invPowerMode = static_cast<int>(force.getInvPowerMode());

    // Store ligand atoms and derivative computation flag
    ligandAtoms = force.getLigandAtoms();
    computeDerivatives = force.getComputeDerivatives();

    // Auto-calculate scaling factors if enabled and not already provided
    if (force.getAutoCalculateScalingFactors() && scaling_factors.empty()) {
        std::string scalingProperty = force.getScalingProperty();
        if (scalingProperty.empty()) {
            throw OpenMMException("GridForce: Auto-calculate scaling factors enabled but no scaling property specified");
        }

        // Validate scaling property
        if (scalingProperty != "charge" && scalingProperty != "ljr" && scalingProperty != "lja") {
            throw OpenMMException("GridForce: Invalid scaling property '" + scalingProperty + "'. Must be 'charge', 'ljr', or 'lja'");
        }

        // Find NonbondedForce in the system
        const NonbondedForce* nonbondedForce = nullptr;
        for (int i = 0; i < system.getNumForces(); i++) {
            if (dynamic_cast<const NonbondedForce*>(&system.getForce(i)) != nullptr) {
                nonbondedForce = dynamic_cast<const NonbondedForce*>(&system.getForce(i));
                break;
            }
        }

        if (nonbondedForce == nullptr) {
            throw OpenMMException("GridForce: Auto-calculate scaling factors requires a NonbondedForce in the system");
        }

        // Extract scaling factors based on property
        scaling_factors.resize(numAtoms);
        for (int i = 0; i < numAtoms; i++) {
            double charge, sigma, epsilon;
            nonbondedForce->getParticleParameters(i, charge, sigma, epsilon);

            if (scalingProperty == "charge") {
                // For electrostatic grids: use charge directly
                scaling_factors[i] = charge;
            } else if (scalingProperty == "ljr") {
                // For LJ repulsive: sqrt(epsilon) * Rmin^6
                // where Rmin = 2^(1/6) * sigma (AMBER convention)
                double rmin = std::pow(2.0, 1.0/6.0) * sigma;
                scaling_factors[i] = std::sqrt(epsilon) * std::pow(rmin, 6.0);
            } else if (scalingProperty == "lja") {
                // For LJ attractive: sqrt(epsilon) * Rmin^3
                // where Rmin = 2^(1/6) * sigma (AMBER convention)
                double rmin = std::pow(2.0, 1.0/6.0) * sigma;
                scaling_factors[i] = std::sqrt(epsilon) * std::pow(rmin, 3.0);
            }
        }

        // Copy calculated scaling factors back to GridForce object
        const_cast<GridForce&>(force).setScalingFactors(scaling_factors);
    }

    // Auto-generate grid if enabled and grid values are empty
    if (force.getAutoGenerateGrid() && vals.empty()) {
        std::string gridType = force.getGridType();

        // Validate grid type
        if (gridType != "charge" && gridType != "ljr" && gridType != "lja") {
            throw OpenMMException("GridForce: Invalid grid type '" + gridType + "'. Must be 'charge', 'ljr', or 'lja'");
        }

        // Ensure grid counts and spacing are set
        if (counts.size() != 3 || spacing.size() != 3) {
            throw OpenMMException("GridForce: Grid counts and spacing must be set before auto-generation");
        }

        // Find NonbondedForce
        const NonbondedForce* nonbondedForce = nullptr;
        for (int i = 0; i < system.getNumForces(); i++) {
            if (dynamic_cast<const NonbondedForce*>(&system.getForce(i)) != nullptr) {
                nonbondedForce = dynamic_cast<const NonbondedForce*>(&system.getForce(i));
                break;
            }
        }

        if (nonbondedForce == nullptr) {
            throw OpenMMException("GridForce: Auto-grid generation requires a NonbondedForce in the system");
        }

        // Get receptor atoms and positions
        std::vector<int> receptorAtoms = force.getReceptorAtoms();
        std::vector<int> ligandAtoms = force.getLigandAtoms();
        const std::vector<Vec3>& receptorPositions = force.getReceptorPositions();

        // If receptorAtoms not specified, use all atoms except ligandAtoms
        if (receptorAtoms.empty()) {
            for (int i = 0; i < system.getNumParticles(); i++) {
                bool isLigand = std::find(ligandAtoms.begin(), ligandAtoms.end(), i) != ligandAtoms.end();
                if (!isLigand) {
                    receptorAtoms.push_back(i);
                }
            }
        }

        // Validate receptor positions
        if (receptorPositions.empty()) {
            throw OpenMMException("GridForce: Receptor positions must be set for auto-grid generation");
        }

        if (receptorPositions.size() < receptorAtoms.size()) {
            throw OpenMMException("GridForce: Not enough receptor positions provided");
        }

        // Get grid origin
        double ox, oy, oz;
        force.getGridOrigin(ox, oy, oz);

        // Generate grid and derivatives (if enabled)
        std::vector<double> derivatives;
        generateGrid(system, nonbondedForce, gridType, receptorAtoms, receptorPositions,
                     ox, oy, oz, vals, derivatives);

        // Copy generated values back to GridForce object so saveToFile() and getGridParameters() work
        const_cast<GridForce&>(force).setGridValues(vals);
        if (!derivatives.empty()) {
            const_cast<GridForce&>(force).setDerivatives(derivatives);
        }

        // Set invPowerMode based on whether transformation was applied
        if (invPower > 0.0f) {
            const_cast<GridForce&>(force).setInvPowerMode(InvPowerMode::STORED, invPower);
        } else {
            const_cast<GridForce&>(force).setInvPowerMode(InvPowerMode::NONE, 0.0);
        }
    }

    if (spacing.size() != 3 || counts.size() != 3) {
        throw OpenMMException("GridForce: Grid dimensions must be 3D");
    }

    if (vals.size() != counts[0] * counts[1] * counts[2]) {
        throw OpenMMException("GridForce: Number of grid values doesn't match grid dimensions");
    }

    // Skip scaling factor validation during auto-grid generation (no ligand atoms in system yet)
    // Only validate when using grids with actual ligand particles
    if (!force.getAutoGenerateGrid()) {
        // Check if we have the right number of scaling factors
        if (scaling_factors.size() > numAtoms) {
            throw OpenMMException("GridForce: Too many scaling factors provided");
        }
        // If we have fewer, verify the missing ones are virtual sites or dummy particles (mass=0)
        if (scaling_factors.size() < numAtoms) {
            for (int i = scaling_factors.size(); i < numAtoms; i++) {
                double mass = system.getParticleMass(i);
                if (mass != 0.0 && !system.isVirtualSite(i)) {
                    throw OpenMMException("GridForce: Missing scaling factor for particle " +
                                        std::to_string(i) + " (mass=" + std::to_string(mass) + ")");
                }
            }
            // Pad with zeros for verified dummy/virtual particles
            while (scaling_factors.size() < numAtoms) {
                scaling_factors.push_back(0.0);
            }
        }
    } else {
        // Auto-grid generation: pad scaling factors with zeros (no ligand atoms to scale)
        while (scaling_factors.size() < numAtoms) {
            scaling_factors.push_back(0.0);
        }
    }

    // Initialize arrays
    g_counts.initialize<int>(cu, 3, "gridCounts");
    g_spacing.initialize<float>(cu, 3, "gridSpacing");
    g_vals.initialize<float>(cu, vals.size(), "gridValues");
    g_scaling_factors.initialize<float>(cu, scaling_factors.size(), "scalingFactors");

    // Copy data to device
    vector<int> countsVec = {counts[0], counts[1], counts[2]};
    vector<float> spacingVec = {(float)spacing[0], (float)spacing[1], (float)spacing[2]};
    vector<float> valsFloat(vals.begin(), vals.end());
    vector<float> scalingFloat(scaling_factors.begin(), scaling_factors.end());

    g_counts.upload(countsVec);
    g_spacing.upload(spacingVec);
    g_vals.upload(valsFloat);
    g_scaling_factors.upload(scalingFloat);

    // Upload derivatives if they exist (for triquintic interpolation)
    if (force.hasDerivatives()) {
        vector<double> derivatives_vec = force.getDerivatives();
//         std::cout << "Uploading " << derivatives_vec.size() << " derivatives to GPU..." << std::endl;

        // Check for NaN/Inf values
        int nan_count = 0;
        int inf_count = 0;
        for (size_t i = 0; i < std::min(derivatives_vec.size(), (size_t)1000); i++) {
            if (std::isnan(derivatives_vec[i])) nan_count++;
            if (std::isinf(derivatives_vec[i])) inf_count++;
        }
//         std::cout << "  Checked first 1000 derivatives: " << nan_count << " NaN, " << inf_count << " Inf" << std::endl;

        // Always create fresh CudaArray to avoid reinitialization issues
        g_derivatives = CudaArray();
//         std::cout << "  Initializing CudaArray with size " << derivatives_vec.size() << "..." << std::endl;
        try {
            g_derivatives.initialize<float>(cu, derivatives_vec.size(), "gridDerivatives");
//             std::cout << "  CudaArray initialized successfully, size=" << g_derivatives.getSize() << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "  ERROR initializing CudaArray: " << e.what() << std::endl;
            throw;
        }

//         std::cout << "  Converting to float vector..." << std::endl;
        vector<float> derivativesFloat(derivatives_vec.begin(), derivatives_vec.end());
//         std::cout << "  Float vector size: " << derivativesFloat.size() << ", bytes: "
//                   << (derivativesFloat.size() * sizeof(float)) << std::endl;
//         std::cout << "  Setting CUDA context..." << std::endl;
        cu.setAsCurrent();
//         std::cout << "  Calling g_derivatives.upload()..." << std::endl;
        try {
            g_derivatives.upload(derivativesFloat);
//             std::cout << "  Upload successful!" << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "  ERROR during upload: " << e.what() << std::endl;
            std::cerr << "  Array size: " << g_derivatives.getSize() << std::endl;
            std::cerr << "  Vector size: " << derivativesFloat.size() << std::endl;
            throw;
        }
    }

    // Store out-of-bounds restraint and interpolation method (gridCap and invPower already set above)
    outOfBoundsRestraint = (float)outOfBounds_k;
    interpolationMethod = interp_method;

    // Store grid origin
    double ox, oy, oz;
    force.getGridOrigin(ox, oy, oz);
    originX = (float)ox;
    originY = (float)oy;
    originZ = (float)oz;

    // Compile kernel from .cu file
    map<string, string> defines;
    CUmodule module = cu.createModule(CudaGridForceKernelSources::gridForceKernel);
    kernel = cu.getKernel(module, "computeGridForce");

    hasInitializedKernel = true;
}

double CudaCalcGridForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    if (!hasInitializedKernel) {
        throw OpenMMException("CudaCalcGridForceKernel: Kernel not initialized before execution");
    }

    if (numAtoms == 0) {
        return 0.0;
    }
    // if (scalingCopy.size() > 0) {
    //     std::cout << "  scaling[0] = " << std::scientific << std::setprecision(6)
    //               << scalingCopy[0] << std::endl;
    // }

    // Set kernel arguments
    int paddedNumAtoms = cu.getPaddedNumAtoms();
    CUdeviceptr posqPtr = cu.getPosq().getDevicePointer();
    CUdeviceptr forcePtr = cu.getLongForceBuffer().getDevicePointer();
    CUdeviceptr countsPtr = g_counts.getDevicePointer();
    CUdeviceptr spacingPtr = g_spacing.getDevicePointer();
    CUdeviceptr valsPtr = g_vals.getDevicePointer();
    CUdeviceptr scalingPtr = g_scaling_factors.getDevicePointer();
    CUdeviceptr derivsPtr = (g_derivatives.isInitialized()) ? g_derivatives.getDevicePointer() : 0;
    CUdeviceptr energyPtr = cu.getEnergyBuffer().getDevicePointer();

    // Debug: Read force buffer BEFORE kernel execution
    // const int num_force_components = 3 * paddedNumAtoms;
    // std::vector<long long> forcesBefore(num_force_components);
    // cuMemcpyDtoH(forcesBefore.data(), forcePtr, num_force_components * sizeof(long long));

    // // Convert fixed-point to float for atom 0
    // double fx_before = forcesBefore[0] / (double)0x100000000;
    // double fy_before = forcesBefore[paddedNumAtoms] / (double)0x100000000;
    // double fz_before = forcesBefore[2*paddedNumAtoms] / (double)0x100000000;
    // std::cout << "  Force buffer BEFORE: atom0 = (" << fx_before << ", "
    //           << fy_before << ", " << fz_before << ")" << std::endl;

    void* args[] = {
        &posqPtr,
        &forcePtr,
        &countsPtr,
        &spacingPtr,
        &valsPtr,
        &scalingPtr,
        &invPower,
        &invPowerMode,
        &interpolationMethod,
        &outOfBoundsRestraint,
        &originX,
        &originY,
        &originZ,
        &derivsPtr,  // Derivatives for triquintic interpolation
        &energyPtr,
        &numAtoms,
        &paddedNumAtoms
    };

    // Execute kernel
    cu.executeKernel(kernel, args, numAtoms);

    // Debug: Read force buffer AFTER kernel execution
    // std::vector<long long> forcesAfter(num_force_components);
    // cuMemcpyDtoH(forcesAfter.data(), forcePtr, num_force_components * sizeof(long long));

    // // Convert fixed-point to float for atom 0
    // double fx_after = forcesAfter[0] / (double)0x100000000;
    // double fy_after = forcesAfter[paddedNumAtoms] / (double)0x100000000;
    // double fz_after = forcesAfter[2*paddedNumAtoms] / (double)0x100000000;
    // std::cout << "  Force buffer AFTER:  atom0 = (" << fx_after << ", "
    //           << fy_after << ", " << fz_after << ")" << std::endl;

    // // Show the delta
    // double delta_fx = fx_after - fx_before;
    // double delta_fy = fy_after - fy_before;
    // double delta_fz = fz_after - fz_before;
    // std::cout << "  Delta (this kernel):        = (" << delta_fx << ", "
    //           << delta_fy << ", " << delta_fz << ")" << std::endl;

    return 0.0;
}

void CudaCalcGridForceKernel::copyParametersToContext(ContextImpl& contextImpl, const GridForce& force) {
    vector<int> counts;
    vector<double> spacing;
    vector<double> vals;
    vector<double> scaling_factors;
    force.getGridParameters(counts, spacing, vals, scaling_factors);
    double inv_power = force.getInvPower();
    int inv_power_mode = static_cast<int>(force.getInvPowerMode());

    if (numAtoms != contextImpl.getSystem().getNumParticles())
        throw OpenMMException("updateParametersInContext: The number of particles has changed");

    if (spacing.size() != 3 || counts.size() != 3)
        throw OpenMMException("GridForce: Grid dimensions must be 3D");

    if (vals.size() != counts[0] * counts[1] * counts[2])
        throw OpenMMException("GridForce: Number of grid values doesn't match grid dimensions");

    // Skip scaling factor validation during auto-grid generation (no ligand atoms in system yet)
    // Only validate when using grids with actual ligand particles
    if (!force.getAutoGenerateGrid()) {
        // Check if we have the right number of scaling factors
        if (scaling_factors.size() > numAtoms)
            throw OpenMMException("GridForce: Too many scaling factors provided");
        // If we have fewer, verify the missing ones are virtual sites or dummy particles (mass=0)
        if (scaling_factors.size() < numAtoms) {
            const System& system = contextImpl.getSystem();
            for (int i = scaling_factors.size(); i < numAtoms; i++) {
                double mass = system.getParticleMass(i);
                if (mass != 0.0 && !system.isVirtualSite(i)) {
                    throw OpenMMException("GridForce: Missing scaling factor for particle " +
                                        std::to_string(i) + " (mass=" + std::to_string(mass) + ")");
                }
            }
            // Pad with zeros for verified dummy/virtual particles
            while (scaling_factors.size() < numAtoms) {
                scaling_factors.push_back(0.0);
            }
        }
    } else {
        // Auto-grid generation: pad scaling factors with zeros (no ligand atoms to scale)
        while (scaling_factors.size() < numAtoms) {
            scaling_factors.push_back(0.0);
        }
    }

    // Update arrays
    vector<int> countsVec = {counts[0], counts[1], counts[2]};
    vector<float> spacingVec = {(float)spacing[0], (float)spacing[1], (float)spacing[2]};
    vector<float> valsFloat(vals.begin(), vals.end());
    vector<float> scalingFloat(scaling_factors.begin(), scaling_factors.end());

    g_counts.upload(countsVec);
    g_spacing.upload(spacingVec);
    g_vals.upload(valsFloat);
    g_scaling_factors.upload(scalingFloat);

    invPower = (float)inv_power;
    invPowerMode = inv_power_mode;
}

void CudaCalcGridForceKernel::generateGrid(
    const System& system,
    const NonbondedForce* nonbondedForce,
    const std::string& gridType,
    const std::vector<int>& receptorAtoms,
    const std::vector<Vec3>& receptorPositions,
    double originX, double originY, double originZ,
    std::vector<double>& vals,
    std::vector<double>& derivatives) {

    cu.setAsCurrent();

    // Total grid points
    // Use size_t to avoid any potential overflow issues
    size_t c0_size = static_cast<size_t>(counts[0]);
    size_t c1_size = static_cast<size_t>(counts[1]);
    size_t c2_size = static_cast<size_t>(counts[2]);
//     std::cout << "DEBUG size_t calculation:" << std::endl;
//     std::cout << "  c0_size = " << c0_size << std::endl;
//     std::cout << "  c1_size = " << c1_size << std::endl;
//     std::cout << "  c2_size = " << c2_size << std::endl;
    size_t product01 = c0_size * c1_size;
//     std::cout << "  c0_size * c1_size = " << product01 << std::endl;
    size_t totalPoints_size = product01 * c2_size;
//     std::cout << "  totalPoints_size = " << totalPoints_size << std::endl;
    int totalPoints = static_cast<int>(totalPoints_size);
//     std::cout << "  totalPoints (int) = " << totalPoints << std::endl;
    vals.resize(totalPoints, 0.0);

    // Extract receptor atom parameters
    std::vector<float3> positions(receptorAtoms.size());
    std::vector<float> charges(receptorAtoms.size());
    std::vector<float> sigmas(receptorAtoms.size());
    std::vector<float> epsilons(receptorAtoms.size());

    for (size_t i = 0; i < receptorAtoms.size(); i++) {
        double q, sig, eps;
        nonbondedForce->getParticleParameters(receptorAtoms[i], q, sig, eps);
        positions[i] = make_float3(receptorPositions[i][0], receptorPositions[i][1], receptorPositions[i][2]);
        charges[i] = q;
        sigmas[i] = sig;
        epsilons[i] = eps;
    }

    // Map grid type to integer
    int gridTypeInt = 0;
    if (gridType == "charge") gridTypeInt = 0;
    else if (gridType == "ljr") gridTypeInt = 1;
    else if (gridType == "lja") gridTypeInt = 2;

    // Allocate GPU memory
    CudaArray receptorPos, receptorCharges, receptorSigmas, receptorEpsilons, gridVals;
    receptorPos.initialize<float3>(cu, receptorAtoms.size(), "receptorPositions");
    receptorCharges.initialize<float>(cu, receptorAtoms.size(), "receptorCharges");
    receptorSigmas.initialize<float>(cu, receptorAtoms.size(), "receptorSigmas");
    receptorEpsilons.initialize<float>(cu, receptorAtoms.size(), "receptorEpsilons");
    gridVals.initialize<float>(cu, totalPoints, "gridValues");

    // Upload data to GPU
    receptorPos.upload(positions);
    receptorCharges.upload(charges);
    receptorSigmas.upload(sigmas);
    receptorEpsilons.upload(epsilons);

    // Prepare grid counts and spacing for GPU
    vector<int> gridCountsVec = {counts[0], counts[1], counts[2]};
    vector<float> gridSpacingVec = {(float)spacing[0], (float)spacing[1], (float)spacing[2]};

    CudaArray d_gridCounts, d_gridSpacing;
    d_gridCounts.initialize<int>(cu, 3, "gridCounts");
    d_gridSpacing.initialize<float>(cu, 3, "gridSpacing");
    d_gridCounts.upload(gridCountsVec);
    d_gridSpacing.upload(gridSpacingVec);

    // Get kernel module
    CUmodule module = cu.createModule(CudaGridForceKernelSources::gridForceKernel);

    // Convert origin to float
    float originXf = (float)originX;
    float originYf = (float)originY;
    float originZf = (float)originZ;
    int numReceptorAtoms = receptorAtoms.size();

    int blockSize = 256;
    int gridSize = (totalPoints + blockSize - 1) / blockSize;
    CUresult result;

    // Decide whether to use analytical derivatives (RASPA3 tensor method) for all grid types
    bool useAnalyticalDerivatives = computeDerivatives;

    if (useAnalyticalDerivatives) {
        // Use RASPA3 tensor method with analytical derivatives for all grid types
        // This generates energy AND all 27 derivatives in one pass
//         std::cout << "Using analytical derivatives (RASPA3 tensor method) for " << gridType << " grid..." << std::endl;

        // Check if derivative array would exceed reasonable GPU memory limit (10 GB)
        size_t derivativeBytes = 27 * totalPoints * sizeof(float);
        const size_t MAX_DERIV_BYTES = 10LL * 1024 * 1024 * 1024;  // 10 GB

        if (derivativeBytes > MAX_DERIV_BYTES) {
            std::cerr << "WARNING: Derivative array would require "
                      << (derivativeBytes / (1024.0 * 1024 * 1024))
                      << " GB GPU memory, exceeding limit of "
                      << (MAX_DERIV_BYTES / (1024.0 * 1024 * 1024))
                      << " GB. Falling back to energy-only generation." << std::endl;
            useAnalyticalDerivatives = false;
        }
    }

    if (useAnalyticalDerivatives) {
        // Allocate GPU memory for grid data (27 values per point)
        CudaArray gridDataGPU;
        gridDataGPU.initialize<float>(cu, 27 * totalPoints, "gridDataWithDerivatives");

        // Get analytical derivative kernel
        CUfunction analyticalKernel = cu.getKernel(module, "generateGridWithAnalyticalDerivatives");

        // Map gridType for analytical kernel: 0=charge, 1=ljr, 2=lja
        int analyticalGridType = gridTypeInt;  // Mapping is the same
        float gridCapF = gridCap;
        float invPowerF = invPower;

        void* analyticalArgs[] = {
            &gridDataGPU.getDevicePointer(),
            &receptorPos.getDevicePointer(),
            &receptorCharges.getDevicePointer(),
            &receptorSigmas.getDevicePointer(),
            &receptorEpsilons.getDevicePointer(),
            &numReceptorAtoms,
            &analyticalGridType,
            &gridCapF,
            &invPowerF,
            &originXf,
            &originYf,
            &originZf,
            &d_gridCounts.getDevicePointer(),
            &d_gridSpacing.getDevicePointer(),
            &totalPoints
        };

        // Launch analytical derivative kernel
        result = cuLaunchKernel(analyticalKernel,
            gridSize, 1, 1,
            blockSize, 1, 1,
            0,
            cu.getCurrentStream(),
            analyticalArgs,
            NULL);

        if (result != CUDA_SUCCESS) {
            throw OpenMMException("Error launching analytical derivative kernel");
        }

        // Synchronize
        cuStreamSynchronize(cu.getCurrentStream());

        // Download all data (energy + derivatives)
        vector<float> gridDataFloat(27 * totalPoints);
        gridDataGPU.download(gridDataFloat);

        // Extract energy values - stored at [0 * totalPoints + point_idx]
        for (int i = 0; i < totalPoints; i++) {
            vals[i] = gridDataFloat[0 * totalPoints + i];
        }

        // Diagnostic: check for problematic values in grid energies
        int nan_count = 0, inf_count = 0;
        double min_val = vals[0], max_val = vals[0];
        for (int i = 0; i < totalPoints; i++) {
            if (std::isnan(vals[i])) nan_count++;
            if (std::isinf(vals[i])) inf_count++;
            if (std::isfinite(vals[i])) {
                min_val = std::min(min_val, vals[i]);
                max_val = std::max(max_val, vals[i]);
            }
        }
//         std::cout << "  Grid energy statistics: min=" << min_val << ", max=" << max_val
//                   << ", NaN=" << nan_count << ", Inf=" << inf_count << std::endl;

        // Store derivatives
        derivatives.resize(27 * totalPoints);
        for (int i = 0; i < 27 * totalPoints; i++) {
            derivatives[i] = gridDataFloat[i];
        }

//         std::cout << "Generated grid with " << totalPoints << " points and "
//                   << derivatives.size() << " derivative values using RASPA3 analytical method" << std::endl;

    } else {
        // Use original finite difference method
        CUfunction kernel = cu.getKernel(module, "generateGridKernel");
        float gridCapF = gridCap;

        void* args[] = {
            &gridVals.getDevicePointer(),
            &receptorPos.getDevicePointer(),
            &receptorCharges.getDevicePointer(),
            &receptorSigmas.getDevicePointer(),
            &receptorEpsilons.getDevicePointer(),
            &numReceptorAtoms,
            &gridTypeInt,
            &gridCapF,
            &invPower,
            &originXf,
            &originYf,
            &originZf,
            &d_gridCounts.getDevicePointer(),
            &d_gridSpacing.getDevicePointer(),
            &totalPoints
        };

        // Launch kernel
        result = cuLaunchKernel(kernel,
            gridSize, 1, 1,
            blockSize, 1, 1,
            0,
            cu.getCurrentStream(),
            args,
            NULL);

        if (result != CUDA_SUCCESS) {
            throw OpenMMException("Error launching grid generation kernel");
        }

        // Download results
        vector<float> gridValsFloat(totalPoints);
        gridVals.download(gridValsFloat);

        // Convert to double
        for (int i = 0; i < totalPoints; i++) {
            vals[i] = gridValsFloat[i];
        }
    }

    // Compute derivatives using finite differences if needed (for non-LJ grids or if analytical failed)
    if (computeDerivatives && !useAnalyticalDerivatives) {
        // Check if derivative array would exceed reasonable GPU memory limit (10 GB)
        size_t derivativeBytes = 27 * totalPoints * sizeof(float);
        const size_t MAX_DERIV_BYTES = 10LL * 1024 * 1024 * 1024;  // 10 GB

        if (derivativeBytes > MAX_DERIV_BYTES) {
            std::cerr << "WARNING: Derivative array would require "
                      << (derivativeBytes / (1024.0 * 1024 * 1024))
                      << " GB GPU memory, exceeding limit of "
                      << (MAX_DERIV_BYTES / (1024.0 * 1024 * 1024))
                      << " GB. Skipping derivative computation." << std::endl;
            std::cerr << "         Grid size: " << counts[0] << "×" << counts[1] << "×" << counts[2]
                      << " = " << totalPoints << " points × 27 derivatives" << std::endl;
            std::cerr << "         Consider using a smaller grid or Reference platform for derivative computation." << std::endl;
            derivatives.clear();  // Clear derivatives so hasDerivatives() returns false
            return;  // Skip derivative computation
        }

        // Allocate GPU memory for derivatives (27 values per grid point)
        CudaArray derivsGPU;
        try {
            derivsGPU.initialize<float>(cu, 27 * totalPoints, "gridDerivatives");
        } catch (const std::exception& e) {
            std::cerr << "ERROR: Failed to allocate GPU memory for derivatives: " << e.what() << std::endl;
            std::cerr << "       Requested: " << (derivativeBytes / (1024.0 * 1024 * 1024)) << " GB" << std::endl;
            throw;
        }

        // Get derivative computation kernel
        CUfunction derivKernel = cu.getKernel(module, "computeDerivativesKernel");

        void* derivArgs[] = {
            &derivsGPU.getDevicePointer(),
            &gridVals.getDevicePointer(),
            &d_gridCounts.getDevicePointer(),
            &d_gridSpacing.getDevicePointer(),
            &totalPoints
        };

        // Launch derivative kernel
        result = cuLaunchKernel(derivKernel,
            gridSize, 1, 1,
            blockSize, 1, 1,
            0,
            cu.getCurrentStream(),
            derivArgs,
            NULL);

        if (result != CUDA_SUCCESS) {
            throw OpenMMException("Error launching derivative computation kernel");
        }

        // Synchronize to ensure kernel completes before downloading
        cuStreamSynchronize(cu.getCurrentStream());

        // Download derivatives
        vector<float> derivsFloat(27 * totalPoints);
        derivsGPU.download(derivsFloat);

        // Convert to double and store in output parameter
        derivatives.resize(27 * totalPoints);
        for (int i = 0; i < 27 * totalPoints; i++) {
            derivatives[i] = derivsFloat[i];
        }

//         std::cout << "Computed " << derivatives.size() << " derivative values on GPU (grid: "
//                   << counts[0] << "×" << counts[1] << "×" << counts[2] << ")" << std::endl;
    }
}