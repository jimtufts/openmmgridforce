/* -------------------------------------------------------------------------- *
 *                               OpenMMGridForce                              *
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

#include "ReferenceGridForceKernels.h"
#include "GridForce.h"

#include "openmm/OpenMMException.h"
#include "openmm/Vec3.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/reference/ReferencePlatform.h"
#include "openmm/NonbondedForce.h"

#include <algorithm>
#include <cmath>

using namespace OpenMM;
using namespace std;

namespace GridForcePlugin {

// The length unit is nm
static vector<Vec3> &extractPositions(ContextImpl &context) {
    ReferencePlatform::PlatformData *data = reinterpret_cast<ReferencePlatform::PlatformData *>(context.getPlatformData());
    return *((vector<Vec3> *)data->positions);
}

static vector<Vec3> &extractForces(ContextImpl &context) {
    ReferencePlatform::PlatformData *data = reinterpret_cast<ReferencePlatform::PlatformData *>(context.getPlatformData());
    return *((vector<Vec3> *)data->forces);
}

/*
    OpenMM Grid Force
*/
void ReferenceCalcGridForceKernel::initialize(const System &system,
                                              const GridForce &grid_force) {
    // Initialize Nonbond parameters.
    grid_force.getGridParameters(g_counts, g_spacing, g_vals, g_scaling_factors);
    g_inv_power = grid_force.getInvPower();

    // Auto-calculate scaling factors if enabled and not already provided
    if (grid_force.getAutoCalculateScalingFactors() && g_scaling_factors.empty()) {
        std::string scalingProperty = grid_force.getScalingProperty();
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
        int numAtoms = system.getNumParticles();
        g_scaling_factors.resize(numAtoms);
        for (int i = 0; i < numAtoms; i++) {
            double charge, sigma, epsilon;
            nonbondedForce->getParticleParameters(i, charge, sigma, epsilon);

            if (scalingProperty == "charge") {
                // For electrostatic grids: use charge directly
                g_scaling_factors[i] = charge;
            } else if (scalingProperty == "ljr") {
                // For LJ repulsive: sqrt(epsilon) * (2*sigma)^6
                double diameter = 2.0 * sigma;
                g_scaling_factors[i] = std::sqrt(epsilon) * std::pow(diameter, 6.0);
            } else if (scalingProperty == "lja") {
                // For LJ attractive: sqrt(epsilon) * (2*sigma)^3
                double diameter = 2.0 * sigma;
                g_scaling_factors[i] = std::sqrt(epsilon) * std::pow(diameter, 3.0);
            }
        }
    }

    // Auto-generate grid if enabled and grid values are empty
    if (grid_force.getAutoGenerateGrid() && g_vals.empty()) {
        std::string gridType = grid_force.getGridType();

        // Validate grid type
        if (gridType != "charge" && gridType != "ljr" && gridType != "lja") {
            throw OpenMMException("GridForce: Invalid grid type '" + gridType + "'. Must be 'charge', 'ljr', or 'lja'");
        }

        // Ensure grid counts and spacing are set
        if (g_counts.size() != 3 || g_spacing.size() != 3) {
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
        std::vector<int> receptorAtoms = grid_force.getReceptorAtoms();
        std::vector<int> ligandAtoms = grid_force.getLigandAtoms();
        const std::vector<Vec3>& receptorPositions = grid_force.getReceptorPositions();

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
        grid_force.getGridOrigin(ox, oy, oz);

        // Generate grid
        generateGrid(system, nonbondedForce, gridType, receptorAtoms, receptorPositions,
                     ox, oy, oz);
    }
}

void ReferenceCalcGridForceKernel::generateGrid(
    const System& system,
    const NonbondedForce* nonbondedForce,
    const std::string& gridType,
    const std::vector<int>& receptorAtoms,
    const std::vector<Vec3>& receptorPositions,
    double originX, double originY, double originZ) {

    // Total grid points
    int totalPoints = g_counts[0] * g_counts[1] * g_counts[2];
    g_vals.resize(totalPoints, 0.0);

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
    for (int i = 0; i < g_counts[0]; i++) {
        for (int j = 0; j < g_counts[1]; j++) {
            for (int k = 0; k < g_counts[2]; k++) {
                // Grid point position (in nm)
                double gx = originX + i * g_spacing[0];
                double gy = originY + j * g_spacing[1];
                double gz = originZ + k * g_spacing[2];

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

                g_vals[idx++] = gridValue;
            }
        }
    }
}

double ReferenceCalcGridForceKernel::execute(ContextImpl &context,
                                             bool includeForces,
                                             bool includeEnergy) {
    
    vector<Vec3> &posData = extractPositions(context);
    vector<Vec3> &forceData = extractForces(context);
    
    const int nyz = g_counts[1] * g_counts[2];
    Vec3 hCorner(g_spacing[0] * (g_counts[0] - 1),
                 g_spacing[1] * (g_counts[1] - 1),
                 g_spacing[2] * (g_counts[2] - 1));

    double energy = 0.0;
    int natom_lig = g_scaling_factors.size();

    for (int ia = 0; ia < natom_lig; ++ia) {
        if (g_scaling_factors[ia] == 0.0) continue;

        Vec3 pi = posData[ia];
        bool is_inside = true;
        for (int k = 0; k < 3; ++k) {
            if (pi[k] > 0.0 && pi[k] < hCorner[k])
                continue;
            else
                is_inside = false;
        }

        if (is_inside) {
            int ix = (int)(pi[0] / g_spacing[0]);
            int iy = (int)(pi[1] / g_spacing[1]);
            int iz = (int)(pi[2] / g_spacing[2]);

            int im = ix * nyz + iy * g_counts[2] + iz;
            int imp = im + g_counts[2];  // iy --> iy + 1
            int ip = im + nyz;           // (ix --> ix+1)
            int ipp = ip + g_counts[2];  // (ix, iy) --> (ix+1, iy+1)

            // Corners of the box surrounding the point
            double vmmm = g_vals[im];
            double vmmp = g_vals[im + 1];   // iz --> iz+1
            double vmpm = g_vals[imp];      // iy --> iy + 1
            double vmpp = g_vals[imp + 1];  // (iy,iz)-->(iy+1, iz+1)

            double vpmm = g_vals[ip];
            double vpmp = g_vals[ip + 1];
            double vppm = g_vals[ipp];
            double vppp = g_vals[ipp + 1];

            // Fraction within the box
            double fx = (pi[0] - ix * g_spacing[0]) / g_spacing[0];
            double fy = (pi[1] - iy * g_spacing[1]) / g_spacing[1];
            double fz = (pi[2] - iz * g_spacing[2]) / g_spacing[2];

            // Fraction ahead
            double ax = 1.0 - fx;
            double ay = 1.0 - fy;
            double az = 1.0 - fz;

            // Trillinear interpolation for energy
            double vmm = az * vmmm + fz * vmmp;
            double vmp = az * vmpm + fz * vmpp;
            double vpm = az * vpmm + fz * vpmp;
            double vpp = az * vppm + fz * vppp;

            double vm = ay * vmm + fy * vmp;
            double vp = ay * vpm + fy * vpp;

            // Get interpolated value (still on transformed scale if inv_power was used)
            double interpolated = ax * vm + fx * vp;

            // Apply inverse power transformation if specified
            // This reverses the grid transformation: (G^(1/n))^n = G
            if (g_inv_power > 0.0) {
                interpolated = pow(interpolated, g_inv_power);
            }

	        double enr = g_scaling_factors[ia] * interpolated;

            energy += enr;

            // x coordinate
            double dvdx = -vm + vp;
            // y coordinate
            double dvdy = (-vmm + vmp) * ax + (-vpm + vpp) * fx;
            // z coordinate
            double dvdz = ((-vmmm + vmmp) * ay + (-vmpm + vmpp) * fy) * ax +
                          ((-vpmm + vpmp) * ay + (-vppm + vppp) * fy) * fx;
            Vec3 grd(dvdx / g_spacing[0], dvdy / g_spacing[1], dvdz / g_spacing[2]);

            // Apply chain rule if inv_power is set
            // d/dx(f^n) = n * f^(n-1) * df/dx
            if (g_inv_power > 0.0) {
                double base_interpolated = ax * vm + fx * vp;  // Value before power transform
                double power_factor = g_inv_power * pow(base_interpolated, g_inv_power - 1.0);
                grd = grd * power_factor;
            }

            forceData[ia] -= g_scaling_factors[ia] * grd;
        } else {
            double kval = 10000.0;  // kJ/mol nm**2
            Vec3 grd(0.0, 0.0, 0.0);
            for (int k = 0; k < 3; k++) {
                double dev = 0.0;
                if (pi[k] < 0.0) {
                    dev = pi[k];
                } else if (pi[k] > hCorner[k]) {
                    dev = pi[k] - hCorner[k];
                }
                energy += 0.5 * kval * dev * dev;
                grd[k] = kval * dev;
            }

            forceData[ia] -= g_scaling_factors[ia] * grd;
        }
    }

    return static_cast<double>(energy);
}

void ReferenceCalcGridForceKernel::copyParametersToContext(ContextImpl &context,
                                                           const GridForce &grid_force) {
    grid_force.getGridParameters(g_counts, g_spacing, g_vals, g_scaling_factors);
    g_inv_power = grid_force.getInvPower();
}



}  // namespace AlGDockPlugin
