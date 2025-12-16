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

    numAtoms = system.getNumParticles();

    // Store counts and spacing as member variables for generateGrid
    counts = counts_local;
    spacing = spacing_local;

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
                // For LJ repulsive: sqrt(epsilon) * (2*sigma)^6
                double diameter = 2.0 * sigma;
                scaling_factors[i] = std::sqrt(epsilon) * std::pow(diameter, 6.0);
            } else if (scalingProperty == "lja") {
                // For LJ attractive: sqrt(epsilon) * (2*sigma)^3
                double diameter = 2.0 * sigma;
                scaling_factors[i] = std::sqrt(epsilon) * std::pow(diameter, 3.0);
            }
        }
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

        // Generate grid
        generateGrid(system, nonbondedForce, gridType, receptorAtoms, receptorPositions,
                     ox, oy, oz, vals);
    }

    if (spacing.size() != 3 || counts.size() != 3) {
        throw OpenMMException("GridForce: Grid dimensions must be 3D");
    }

    if (vals.size() != counts[0] * counts[1] * counts[2]) {
        throw OpenMMException("GridForce: Number of grid values doesn't match grid dimensions");
    }

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

    // Store inv_power
    invPower = (float)inv_power;

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

    // Set kernel arguments
    int paddedNumAtoms = cu.getPaddedNumAtoms();
    CUdeviceptr posqPtr = cu.getPosq().getDevicePointer();
    CUdeviceptr forcePtr = cu.getLongForceBuffer().getDevicePointer();
    CUdeviceptr countsPtr = g_counts.getDevicePointer();
    CUdeviceptr spacingPtr = g_spacing.getDevicePointer();
    CUdeviceptr valsPtr = g_vals.getDevicePointer();
    CUdeviceptr scalingPtr = g_scaling_factors.getDevicePointer();
    CUdeviceptr energyPtr = cu.getEnergyBuffer().getDevicePointer();

    void* args[] = {
        &posqPtr,
        &forcePtr,
        &countsPtr,
        &spacingPtr,
        &valsPtr,
        &scalingPtr,
        &invPower,
        &energyPtr,
        &numAtoms,
        &paddedNumAtoms
    };

    // Execute kernel
    cu.executeKernel(kernel, args, numAtoms);

    return 0.0;
}

void CudaCalcGridForceKernel::copyParametersToContext(ContextImpl& contextImpl, const GridForce& force) {
    vector<int> counts;
    vector<double> spacing;
    vector<double> vals;
    vector<double> scaling_factors;
    force.getGridParameters(counts, spacing, vals, scaling_factors);
    double inv_power = force.getInvPower();

    if (numAtoms != contextImpl.getSystem().getNumParticles())
        throw OpenMMException("updateParametersInContext: The number of particles has changed");

    if (spacing.size() != 3 || counts.size() != 3)
        throw OpenMMException("GridForce: Grid dimensions must be 3D");

    if (vals.size() != counts[0] * counts[1] * counts[2])
        throw OpenMMException("GridForce: Number of grid values doesn't match grid dimensions");

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
}

void CudaCalcGridForceKernel::generateGrid(
    const System& system,
    const NonbondedForce* nonbondedForce,
    const std::string& gridType,
    const std::vector<int>& receptorAtoms,
    const std::vector<Vec3>& receptorPositions,
    double originX, double originY, double originZ,
    std::vector<double>& vals) {

    // Total grid points
    int totalPoints = counts[0] * counts[1] * counts[2];
    vals.resize(totalPoints, 0.0);

    // Extract receptor atom parameters
    std::vector<double> charges, sigmas, epsilons;
    for (int atomIdx : receptorAtoms) {
        double q, sig, eps;
        nonbondedForce->getParticleParameters(atomIdx, q, sig, eps);
        charges.push_back(q);
        sigmas.push_back(sig);
        epsilons.push_back(eps);
    }

    // Physics constants in OpenMM units
    const double COULOMB_CONST = 138.935456;  // kJ·nm/(mol·e²)
    const double U_MAX = 41840.0;  // 10000 kcal/mol * 4.184 kJ/kcal

    // For each grid point
    int idx = 0;
    for (int i = 0; i < counts[0]; i++) {
        for (int j = 0; j < counts[1]; j++) {
            for (int k = 0; k < counts[2]; k++) {
                // Grid point position (in nm)
                double gx = originX + i * spacing[0];
                double gy = originY + j * spacing[1];
                double gz = originZ + k * spacing[2];

                // Calculate contribution from each receptor atom
                double gridValue = 0.0;
                for (size_t atomIdx = 0; atomIdx < receptorAtoms.size(); atomIdx++) {
                    // Get atom position (in nm)
                    Vec3 atomPos = receptorPositions[atomIdx];

                    // Calculate distance
                    double dx = gx - atomPos[0];
                    double dy = gy - atomPos[1];
                    double dz = gz - atomPos[2];
                    double r2 = dx*dx + dy*dy + dz*dz;
                    double r = std::sqrt(r2);

                    // Avoid singularities at very small distances
                    if (r < 1e-6) {
                        r = 1e-6;
                    }

                    // Calculate contribution based on grid type
                    if (gridType == "charge") {
                        // Electrostatic potential: k * q / r
                        gridValue += COULOMB_CONST * charges[atomIdx] / r;
                    } else if (gridType == "ljr") {
                        // LJ repulsive: sqrt(epsilon) * diameter^6 / r^12
                        double diameter = 2.0 * sigmas[atomIdx];
                        gridValue += std::sqrt(epsilons[atomIdx]) * std::pow(diameter, 6.0) / std::pow(r, 12.0);
                    } else if (gridType == "lja") {
                        // LJ attractive: -2 * sqrt(epsilon) * diameter^3 / r^6
                        double diameter = 2.0 * sigmas[atomIdx];
                        gridValue += -2.0 * std::sqrt(epsilons[atomIdx]) * std::pow(diameter, 3.0) / std::pow(r, 6.0);
                    }
                }

                // Apply capping to avoid extreme values
                gridValue = U_MAX * std::tanh(gridValue / U_MAX);

                vals[idx++] = gridValue;
            }
        }
    }
}
