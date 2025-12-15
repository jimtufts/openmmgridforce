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

using namespace GridForcePlugin;
using namespace OpenMM;
using namespace std;

CudaCalcGridForceKernel::~CudaCalcGridForceKernel() {
}

void CudaCalcGridForceKernel::initialize(const System& system, const GridForce& force) {
    cu.setAsCurrent();
    // Get grid parameters
    vector<int> counts;
    vector<double> spacing;
    vector<double> vals;
    vector<double> scaling_factors;
    force.getGridParameters(counts, spacing, vals, scaling_factors);
    double inv_power = force.getInvPower();

    numAtoms = system.getNumParticles();

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
