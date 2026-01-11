/* -------------------------------------------------------------------------- *
 *                              OpenMMGridForce                               *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2014 Stanford University and the Authors.           *
 * Authors:                                                                   *
 * Contributors:                                                              *
 *                                                                            *
 * Permission is hereby granted, free of charge, to any person obtaining a    *
 * copy of this software and associated documentation files (the "Software"), *
 * to deal in the Software without restriction, including without limitation  *
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,   *
 * and/or sell copies of the Software, and to permit persons to whom the      *
 * Software is furnished to do so, subject to the following conditions:       *
 *                                                                            *
 * The above copyright notice and this permission notice shall be included in *
 * all copies or substantial portions of the Software.                        *
 *                                                                            *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR *
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,   *
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL    *
 * THE AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,    *
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR      *
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE  *
 * USE OR OTHER DEALINGS IN THE SOFTWARE.                                     *
 * -------------------------------------------------------------------------- */

#include "CommonGridForceKernels.h"
#include "CommonGridForceKernelSources.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/common/ComputeForceInfo.h"
#include "openmm/common/ContextSelector.h"
#include "openmm/NonbondedForce.h"
#include "IsolatedNonbondedForce.h"
#include <map>
#include <iostream>
#include <cstring>
#include <algorithm>
#include <cmath>

using namespace GridForcePlugin;
using namespace OpenMM;
using namespace std;

class GridForceInfo : public ComputeForceInfo {
public:
    GridForceInfo(int numAtoms) : numAtoms(numAtoms) {
    }
    int getNumParticleGroups() {
        return numAtoms;
    }
    void getParticlesInGroup(int index, vector<int>& particles) {
        particles.clear();
        particles.push_back(index);
    }
    bool areGroupsIdentical(int group1, int group2) {
        return true;
    }
private:
    int numAtoms;
};

void CommonCalcGridForceKernel::initialize(const System& system, const GridForce& force) {
    ContextSelector selector(cc);

    // Get grid parameters
    vector<int> counts_local;
    vector<double> spacing_local;
    vector<double> vals;
    vector<double> scaling_factors;
    force.getGridParameters(counts_local, spacing_local, vals, scaling_factors);
    double inv_power = force.getInvPower();
    double outOfBoundsRestraint = force.getOutOfBoundsRestraint();
    double gridCap = force.getGridCap();
    int interpolationMethod = force.getInterpolationMethod();

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

        // Find NonbondedForce or IsolatedNonbondedForce in the system
        const NonbondedForce* nonbondedForce = nullptr;
        const IsolatedNonbondedForce* isolatedNonbondedForce = nullptr;

        for (int i = 0; i < system.getNumForces(); i++) {
            if (dynamic_cast<const NonbondedForce*>(&system.getForce(i)) != nullptr) {
                nonbondedForce = dynamic_cast<const NonbondedForce*>(&system.getForce(i));
                break;
            } else if (dynamic_cast<const IsolatedNonbondedForce*>(&system.getForce(i)) != nullptr) {
                isolatedNonbondedForce = dynamic_cast<const IsolatedNonbondedForce*>(&system.getForce(i));
                // Keep searching in case there's a NonbondedForce (prefer that)
            }
        }

        if (nonbondedForce == nullptr && isolatedNonbondedForce == nullptr) {
            throw OpenMMException("GridForce: Auto-calculate scaling factors requires a NonbondedForce or IsolatedNonbondedForce in the system");
        }

        // Extract scaling factors based on property
        scaling_factors.resize(numAtoms);
        for (int i = 0; i < numAtoms; i++) {
            double charge, sigma, epsilon;

            // Get parameters from whichever force is available
            if (nonbondedForce != nullptr) {
                nonbondedForce->getParticleParameters(i, charge, sigma, epsilon);
            } else {
                isolatedNonbondedForce->getAtomParameters(i, charge, sigma, epsilon);
            }

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

        // Find NonbondedForce or IsolatedNonbondedForce
        const NonbondedForce* nonbondedForce = nullptr;
        const IsolatedNonbondedForce* isolatedNonbondedForce = nullptr;

        for (int i = 0; i < system.getNumForces(); i++) {
            if (dynamic_cast<const NonbondedForce*>(&system.getForce(i)) != nullptr) {
                nonbondedForce = dynamic_cast<const NonbondedForce*>(&system.getForce(i));
                break;
            } else if (dynamic_cast<const IsolatedNonbondedForce*>(&system.getForce(i)) != nullptr) {
                isolatedNonbondedForce = dynamic_cast<const IsolatedNonbondedForce*>(&system.getForce(i));
                // Keep searching in case there's a NonbondedForce (prefer that)
            }
        }

        if (nonbondedForce == nullptr && isolatedNonbondedForce == nullptr) {
            throw OpenMMException("GridForce: Auto-grid generation requires a NonbondedForce or IsolatedNonbondedForce in the system");
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
        generateGrid(system, nonbondedForce, isolatedNonbondedForce, gridType, receptorAtoms, receptorPositions,
                     ox, oy, oz, gridCap, inv_power, vals);

        // Copy generated values back to GridForce object so saveToFile() and getGridParameters() work
        const_cast<GridForce&>(force).setGridValues(vals);
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
    g_counts.initialize<int>(cc, 3, "gridCounts");
    g_spacing.initialize<float>(cc, 3, "gridSpacing");
    g_vals.initialize<float>(cc, vals.size(), "gridValues");
    g_scaling_factors.initialize<float>(cc, scaling_factors.size(), "scalingFactors");

    // Copy data to device
    vector<int> countsVec = {counts[0], counts[1], counts[2]};
    vector<float> spacingVec = {(float)spacing[0], (float)spacing[1], (float)spacing[2]};
    vector<float> valsFloat(vals.begin(), vals.end());
    vector<float> scalingFloat(scaling_factors.begin(), scaling_factors.end());

    g_counts.upload(countsVec);
    g_spacing.upload(spacingVec);
    g_vals.upload(valsFloat);
    g_scaling_factors.upload(scalingFloat);

    // Create kernel
    map<string, string> defines;
    defines["GRID_SIZE_X"] = cc.intToString(counts[0]);
    defines["GRID_SIZE_Y"] = cc.intToString(counts[1]);
    defines["GRID_SIZE_Z"] = cc.intToString(counts[2]);
    defines["NUM_ATOMS"] = cc.intToString(numAtoms);
    defines["PADDED_NUM_ATOMS"] = cc.intToString(cc.getPaddedNumAtoms());

    ComputeProgram program = cc.compileProgram(CommonGridForceKernelSources::gridForceSource, defines);
    computeKernel = program->createKernel("computeGridForce");

    // Set kernel arguments
    computeKernel->addArg(cc.getPosq());
    computeKernel->addArg(cc.getLongForceBuffer());
    computeKernel->addArg(g_counts);
    computeKernel->addArg(g_spacing);
    computeKernel->addArg(g_vals);
    computeKernel->addArg(g_scaling_factors);
    computeKernel->addArg((float)inv_power);
    computeKernel->addArg(interpolationMethod);
    computeKernel->addArg((float)outOfBoundsRestraint);
    computeKernel->addArg(cc.getEnergyBuffer());

    cc.addForce(new GridForceInfo(numAtoms));

    hasInitializedKernel = true;
}

double CommonCalcGridForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    if (!hasInitializedKernel) {
        throw OpenMMException("CommonCalcGridForceKernel: Kernel not initialized before execution");
    }

    if (numAtoms == 0) {
        return 0.0;
    }

    // Execute the kernel
    computeKernel->execute(numAtoms);

    return 0.0;
}

void CommonCalcGridForceKernel::copyParametersToContext(ContextImpl& contextImpl, const GridForce& force) {
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

    // Update inv_power parameter
    computeKernel->setArg(6, (float)inv_power);
}

void CommonCalcGridForceKernel::generateGrid(
    const System& system,
    const NonbondedForce* nonbondedForce,
    const IsolatedNonbondedForce* isolatedNonbondedForce,
    const std::string& gridType,
    const std::vector<int>& receptorAtoms,
    const std::vector<Vec3>& receptorPositions,
    double originX, double originY, double originZ,
    double gridCap,
    double invPower,
    std::vector<double>& vals) {

    // Total grid points
    int totalPoints = counts[0] * counts[1] * counts[2];
    vals.resize(totalPoints, 0.0);

    // Extract receptor atom parameters
    std::vector<double> charges, sigmas, epsilons;
    for (int atomIdx : receptorAtoms) {
        double q, sig, eps;
        if (nonbondedForce != nullptr) {
            nonbondedForce->getParticleParameters(atomIdx, q, sig, eps);
        } else {
            isolatedNonbondedForce->getAtomParameters(atomIdx, q, sig, eps);
        }
        charges.push_back(q);
        sigmas.push_back(sig);
        epsilons.push_back(eps);
    }

    // Physics constants in OpenMM units
    const double COULOMB_CONST = 138.935456;  // kJ·nm/(mol·e²)
    const double U_MAX = gridCap;  // Configurable capping threshold (kJ/mol)

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
                        // LJ repulsive: sqrt(epsilon) * Rmin^6 / r^12
                        // where Rmin = 2^(1/6) * sigma (AMBER convention)
                        double rmin = std::pow(2.0, 1.0/6.0) * sigmas[atomIdx];
                        gridValue += std::sqrt(epsilons[atomIdx]) * std::pow(rmin, 6.0) / std::pow(r, 12.0);
                    } else if (gridType == "lja") {
                        // LJ attractive: -2 * sqrt(epsilon) * Rmin^3 / r^6
                        // where Rmin = 2^(1/6) * sigma (AMBER convention)
                        double rmin = std::pow(2.0, 1.0/6.0) * sigmas[atomIdx];
                        gridValue += -2.0 * std::sqrt(epsilons[atomIdx]) * std::pow(rmin, 3.0) / std::pow(r, 6.0);
                    }
                }

                // Apply capping to avoid extreme values
                gridValue = U_MAX * std::tanh(gridValue / U_MAX);

                // Apply inverse power transformation if specified
                // Grid should store G^(1/n), kernel will apply ^n to recover G
                if (invPower > 0.0) {
                    gridValue = std::pow(gridValue, 1.0 / invPower);
                }

                vals[idx++] = gridValue;
            }
        }
    }
}

vector<double> CommonCalcGridForceKernel::getParticleGroupEnergies() {
    // Common platform does not support per-group energy tracking yet
    return vector<double>();
}

vector<double> CommonCalcGridForceKernel::getParticleAtomEnergies() {
    // Common platform does not support per-atom energy tracking yet
    return vector<double>();
}
