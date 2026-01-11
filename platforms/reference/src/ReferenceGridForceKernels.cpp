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
#include "TriquinticMatrix.h"

#include "openmm/OpenMMException.h"
#include "openmm/Vec3.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/reference/ReferencePlatform.h"
#include "openmm/NonbondedForce.h"
#include <iostream>
#include <iomanip>

#include <algorithm>
#include <cmath>

using namespace OpenMM;
using namespace std;

namespace GridForcePlugin {

// Cubic B-spline basis functions
// t is the fractional position within the grid cell [0,1]
inline double bspline_basis0(double t) { return (1.0 - t) * (1.0 - t) * (1.0 - t) / 6.0; }
inline double bspline_basis1(double t) { return (3.0 * t * t * t - 6.0 * t * t + 4.0) / 6.0; }
inline double bspline_basis2(double t) { return (-3.0 * t * t * t + 3.0 * t * t + 3.0 * t + 1.0) / 6.0; }
inline double bspline_basis3(double t) { return t * t * t / 6.0; }

// Derivatives of cubic B-spline basis functions
inline double bspline_deriv0(double t) { return -(1.0 - t) * (1.0 - t) / 2.0; }
inline double bspline_deriv1(double t) { return (3.0 * t * t - 4.0 * t) / 2.0; }
inline double bspline_deriv2(double t) { return (-3.0 * t * t + 2.0 * t + 1.0) / 2.0; }
inline double bspline_deriv3(double t) { return t * t / 2.0; }

// Cubic Hermite basis functions for tricubic interpolation
// These interpolate exactly through points while maintaining C1 continuity
// h00, h01 are for function values; h10, h11 are for derivatives
inline double hermite_h00(double t) { return (1.0 + 2.0*t) * (1.0 - t) * (1.0 - t); }  // Interpolates f(0)
inline double hermite_h10(double t) { return t * (1.0 - t) * (1.0 - t); }              // Scales f'(0)
inline double hermite_h01(double t) { return t * t * (3.0 - 2.0*t); }                  // Interpolates f(1)
inline double hermite_h11(double t) { return t * t * (t - 1.0); }                      // Scales f'(1)

// Derivatives of Hermite basis functions (for computing forces)
inline double hermite_dh00(double t) { return 6.0*t*t - 6.0*t; }
inline double hermite_dh10(double t) { return 3.0*t*t - 4.0*t + 1.0; }
inline double hermite_dh01(double t) { return -6.0*t*t + 6.0*t; }
inline double hermite_dh11(double t) { return 3.0*t*t - 2.0*t; }

// Quintic Hermite basis functions for C2 continuous interpolation
// These interpolate exactly through points with C2 continuity
// h00, h01 for function values; h10, h11 for 1st derivatives; h20, h21 for 2nd derivatives
inline double quintic_h00(double t) {
    double t2 = t*t, t3 = t2*t, t4 = t3*t, t5 = t4*t;
    return 1.0 - 10.0*t3 + 15.0*t4 - 6.0*t5;
}
inline double quintic_h01(double t) {
    double t2 = t*t, t3 = t2*t, t4 = t3*t, t5 = t4*t;
    return 10.0*t3 - 15.0*t4 + 6.0*t5;
}
inline double quintic_h10(double t) {
    double t2 = t*t, t3 = t2*t, t4 = t3*t, t5 = t4*t;
    return t - 6.0*t3 + 8.0*t4 - 3.0*t5;
}
inline double quintic_h11(double t) {
    double t2 = t*t, t3 = t2*t, t4 = t3*t, t5 = t4*t;
    return -4.0*t3 + 7.0*t4 - 3.0*t5;
}
inline double quintic_h20(double t) {
    double t2 = t*t, t3 = t2*t, t4 = t3*t, t5 = t4*t;
    return 0.5*t2 - 1.5*t3 + 1.5*t4 - 0.5*t5;
}
inline double quintic_h21(double t) {
    double t2 = t*t, t3 = t2*t, t4 = t3*t, t5 = t4*t;
    return 0.5*t3 - t4 + 0.5*t5;
}

// Derivatives of quintic Hermite basis functions (for computing forces)
inline double quintic_dh00(double t) {
    double t2 = t*t, t3 = t2*t, t4 = t3*t;
    return -30.0*t2 + 60.0*t3 - 30.0*t4;
}
inline double quintic_dh01(double t) {
    double t2 = t*t, t3 = t2*t, t4 = t3*t;
    return 30.0*t2 - 60.0*t3 + 30.0*t4;
}
inline double quintic_dh10(double t) {
    double t2 = t*t, t3 = t2*t, t4 = t3*t;
    return 1.0 - 18.0*t2 + 32.0*t3 - 15.0*t4;
}
inline double quintic_dh11(double t) {
    double t2 = t*t, t3 = t2*t, t4 = t3*t;
    return -12.0*t2 + 28.0*t3 - 15.0*t4;
}
inline double quintic_dh20(double t) {
    double t2 = t*t, t3 = t2*t, t4 = t3*t;
    return t - 4.5*t2 + 6.0*t3 - 2.5*t4;
}
inline double quintic_dh21(double t) {
    double t2 = t*t, t3 = t2*t, t4 = t3*t;
    return 1.5*t2 - 4.0*t3 + 2.5*t4;
}

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

    // Get ligand atom indices
    g_ligand_atoms = grid_force.getLigandAtoms();
    g_inv_power = grid_force.getInvPower();
    g_gridCap = grid_force.getGridCap();
    g_outOfBoundsRestraint = grid_force.getOutOfBoundsRestraint();
    g_interpolationMethod = grid_force.getInterpolationMethod();
    grid_force.getGridOrigin(g_origin_x, g_origin_y, g_origin_z);
    g_computeDerivatives = grid_force.getComputeDerivatives();
    g_derivatives = grid_force.getDerivatives();

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

        // Copy calculated scaling factors back to GridForce object
        const_cast<GridForce&>(grid_force).setScalingFactors(g_scaling_factors);
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

        // Copy generated values back to GridForce object so saveToFile() and getGridParameters() work
        const_cast<GridForce&>(grid_force).setGridValues(g_vals);

        // Copy derivatives back if they were computed
        if (!g_derivatives.empty()) {
            const_cast<GridForce&>(grid_force).setDerivatives(g_derivatives);
        }
    }
}

std::vector<double> ReferenceCalcGridForceKernel::computeDerivativesAtPoint(
    const std::vector<double>& rawGrid,
    int ix, int iy, int iz,
    double dx, double dy, double dz) const {

    const int nx = g_counts[0];
    const int ny = g_counts[1];
    const int nz = g_counts[2];
    const int nyz = ny * nz;

    // Helper to safely get grid value with boundary handling (clamp to boundary values)
    auto getVal = [&](int i, int j, int k) -> double {
        // Clamp indices to valid range instead of returning 0
        int ic = std::max(0, std::min(i, nx-1));
        int jc = std::max(0, std::min(j, ny-1));
        int kc = std::max(0, std::min(k, nz-1));
        return rawGrid[ic * nyz + jc * nz + kc];
    };

    double f = getVal(ix, iy, iz);

    // First derivatives (use one-sided differences at boundaries, centered otherwise)
    double dx_f, dy_f, dz_f;
    if (ix == 0) {
        dx_f = (getVal(ix+1, iy, iz) - f) / dx;  // Forward difference
    } else if (ix == nx-1) {
        dx_f = (f - getVal(ix-1, iy, iz)) / dx;  // Backward difference
    } else {
        dx_f = (getVal(ix+1, iy, iz) - getVal(ix-1, iy, iz)) / (2.0 * dx);  // Centered
    }

    if (iy == 0) {
        dy_f = (getVal(ix, iy+1, iz) - f) / dy;
    } else if (iy == ny-1) {
        dy_f = (f - getVal(ix, iy-1, iz)) / dy;
    } else {
        dy_f = (getVal(ix, iy+1, iz) - getVal(ix, iy-1, iz)) / (2.0 * dy);
    }

    if (iz == 0) {
        dz_f = (getVal(ix, iy, iz+1) - f) / dz;
    } else if (iz == nz-1) {
        dz_f = (f - getVal(ix, iy, iz-1)) / dz;
    } else {
        dz_f = (getVal(ix, iy, iz+1) - getVal(ix, iy, iz-1)) / (2.0 * dz);
    }

    // Second derivatives (pure) - use one-sided at boundaries
    double dxx_f, dyy_f, dzz_f;
    if (ix == 0) {
        dxx_f = (getVal(ix+2, iy, iz) - 2.0*getVal(ix+1, iy, iz) + f) / (dx * dx);
    } else if (ix == nx-1) {
        dxx_f = (f - 2.0*getVal(ix-1, iy, iz) + getVal(ix-2, iy, iz)) / (dx * dx);
    } else {
        dxx_f = (getVal(ix+1, iy, iz) - 2.0*f + getVal(ix-1, iy, iz)) / (dx * dx);
    }

    if (iy == 0) {
        dyy_f = (getVal(ix, iy+2, iz) - 2.0*getVal(ix, iy+1, iz) + f) / (dy * dy);
    } else if (iy == ny-1) {
        dyy_f = (f - 2.0*getVal(ix, iy-1, iz) + getVal(ix, iy-2, iz)) / (dy * dy);
    } else {
        dyy_f = (getVal(ix, iy+1, iz) - 2.0*f + getVal(ix, iy-1, iz)) / (dy * dy);
    }

    if (iz == 0) {
        dzz_f = (getVal(ix, iy, iz+2) - 2.0*getVal(ix, iy, iz+1) + f) / (dz * dz);
    } else if (iz == nz-1) {
        dzz_f = (f - 2.0*getVal(ix, iy, iz-1) + getVal(ix, iy, iz-2)) / (dz * dz);
    } else {
        dzz_f = (getVal(ix, iy, iz+1) - 2.0*f + getVal(ix, iy, iz-1)) / (dz * dz);
    }

    // Second derivatives (mixed)
    double dxy_f = (getVal(ix+1, iy+1, iz) - getVal(ix-1, iy+1, iz) -
                    getVal(ix+1, iy-1, iz) + getVal(ix-1, iy-1, iz)) / (4.0 * dx * dy);
    double dxz_f = (getVal(ix+1, iy, iz+1) - getVal(ix-1, iy, iz+1) -
                    getVal(ix+1, iy, iz-1) + getVal(ix-1, iy, iz-1)) / (4.0 * dx * dz);
    double dyz_f = (getVal(ix, iy+1, iz+1) - getVal(ix, iy-1, iz+1) -
                    getVal(ix, iy+1, iz-1) + getVal(ix, iy-1, iz-1)) / (4.0 * dy * dz);

    // Third derivatives
    double dxxy_f = (getVal(ix+1, iy+1, iz) - 2.0*getVal(ix, iy+1, iz) + getVal(ix-1, iy+1, iz) -
                     getVal(ix+1, iy-1, iz) + 2.0*getVal(ix, iy-1, iz) - getVal(ix-1, iy-1, iz)) /
                    (2.0 * dx * dx * dy);
    double dxxz_f = (getVal(ix+1, iy, iz+1) - 2.0*getVal(ix, iy, iz+1) + getVal(ix-1, iy, iz+1) -
                     getVal(ix+1, iy, iz-1) + 2.0*getVal(ix, iy, iz-1) - getVal(ix-1, iy, iz-1)) /
                    (2.0 * dx * dx * dz);
    double dxyy_f = (getVal(ix+1, iy+1, iz) - 2.0*getVal(ix+1, iy, iz) + getVal(ix+1, iy-1, iz) -
                     getVal(ix-1, iy+1, iz) + 2.0*getVal(ix-1, iy, iz) - getVal(ix-1, iy-1, iz)) /
                    (2.0 * dx * dy * dy);
    double dyyz_f = (getVal(ix, iy+1, iz+1) - 2.0*getVal(ix, iy, iz+1) + getVal(ix, iy-1, iz+1) -
                     getVal(ix, iy+1, iz-1) + 2.0*getVal(ix, iy, iz-1) - getVal(ix, iy-1, iz-1)) /
                    (2.0 * dy * dy * dz);
    double dxzz_f = (getVal(ix+1, iy, iz+1) - 2.0*getVal(ix+1, iy, iz) + getVal(ix+1, iy, iz-1) -
                     getVal(ix-1, iy, iz+1) + 2.0*getVal(ix-1, iy, iz) - getVal(ix-1, iy, iz-1)) /
                    (2.0 * dx * dz * dz);
    double dyzz_f = (getVal(ix, iy+1, iz+1) - 2.0*getVal(ix, iy+1, iz) + getVal(ix, iy+1, iz-1) -
                     getVal(ix, iy-1, iz+1) + 2.0*getVal(ix, iy-1, iz) - getVal(ix, iy-1, iz-1)) /
                    (2.0 * dy * dz * dz);
    double dxyz_f = (getVal(ix+1, iy+1, iz+1) - getVal(ix-1, iy+1, iz+1) -
                     getVal(ix+1, iy-1, iz+1) + getVal(ix-1, iy-1, iz+1) -
                     getVal(ix+1, iy+1, iz-1) + getVal(ix-1, iy+1, iz-1) +
                     getVal(ix+1, iy-1, iz-1) - getVal(ix-1, iy-1, iz-1)) / (8.0 * dx * dy * dz);

    // Fourth derivatives
    double dxxyy_f = (getVal(ix+1, iy+1, iz) - 2.0*getVal(ix, iy+1, iz) + getVal(ix-1, iy+1, iz) -
                      2.0*getVal(ix+1, iy, iz) + 4.0*f - 2.0*getVal(ix-1, iy, iz) +
                      getVal(ix+1, iy-1, iz) - 2.0*getVal(ix, iy-1, iz) + getVal(ix-1, iy-1, iz)) /
                     (dx * dx * dy * dy);
    double dxxzz_f = (getVal(ix+1, iy, iz+1) - 2.0*getVal(ix, iy, iz+1) + getVal(ix-1, iy, iz+1) -
                      2.0*getVal(ix+1, iy, iz) + 4.0*f - 2.0*getVal(ix-1, iy, iz) +
                      getVal(ix+1, iy, iz-1) - 2.0*getVal(ix, iy, iz-1) + getVal(ix-1, iy, iz-1)) /
                     (dx * dx * dz * dz);
    double dyyzz_f = (getVal(ix, iy+1, iz+1) - 2.0*getVal(ix, iy, iz+1) + getVal(ix, iy-1, iz+1) -
                      2.0*getVal(ix, iy+1, iz) + 4.0*f - 2.0*getVal(ix, iy-1, iz) +
                      getVal(ix, iy+1, iz-1) - 2.0*getVal(ix, iy, iz-1) + getVal(ix, iy-1, iz-1)) /
                     (dy * dy * dz * dz);

    // Fourth derivatives (mixed with 3 variables)
    double dxxyz_f = (getVal(ix+1, iy+1, iz+1) - 2.0*getVal(ix, iy+1, iz+1) + getVal(ix-1, iy+1, iz+1) -
                      getVal(ix+1, iy-1, iz+1) + 2.0*getVal(ix, iy-1, iz+1) - getVal(ix-1, iy-1, iz+1) -
                      getVal(ix+1, iy+1, iz-1) + 2.0*getVal(ix, iy+1, iz-1) - getVal(ix-1, iy+1, iz-1) +
                      getVal(ix+1, iy-1, iz-1) - 2.0*getVal(ix, iy-1, iz-1) + getVal(ix-1, iy-1, iz-1)) /
                     (4.0 * dx * dx * dy * dz);
    double dxyyz_f = (getVal(ix+1, iy+1, iz+1) - 2.0*getVal(ix+1, iy, iz+1) + getVal(ix+1, iy-1, iz+1) -
                      getVal(ix-1, iy+1, iz+1) + 2.0*getVal(ix-1, iy, iz+1) - getVal(ix-1, iy-1, iz+1) -
                      getVal(ix+1, iy+1, iz-1) + 2.0*getVal(ix+1, iy, iz-1) - getVal(ix+1, iy-1, iz-1) +
                      getVal(ix-1, iy+1, iz-1) - 2.0*getVal(ix-1, iy, iz-1) + getVal(ix-1, iy-1, iz-1)) /
                     (4.0 * dx * dy * dy * dz);
    double dxyzz_f = (getVal(ix+1, iy+1, iz+1) - 2.0*getVal(ix+1, iy+1, iz) + getVal(ix+1, iy+1, iz-1) -
                      getVal(ix-1, iy+1, iz+1) + 2.0*getVal(ix-1, iy+1, iz) - getVal(ix-1, iy+1, iz-1) -
                      getVal(ix+1, iy-1, iz+1) + 2.0*getVal(ix+1, iy-1, iz) - getVal(ix+1, iy-1, iz-1) +
                      getVal(ix-1, iy-1, iz+1) - 2.0*getVal(ix-1, iy-1, iz) + getVal(ix-1, iy-1, iz-1)) /
                     (4.0 * dx * dy * dz * dz);

    // Fifth derivatives
    double dxxyyz_f = (getVal(ix+1, iy+1, iz+1) - 2.0*getVal(ix, iy+1, iz+1) + getVal(ix-1, iy+1, iz+1) -
                       2.0*getVal(ix+1, iy, iz+1) + 4.0*getVal(ix, iy, iz+1) - 2.0*getVal(ix-1, iy, iz+1) +
                       getVal(ix+1, iy-1, iz+1) - 2.0*getVal(ix, iy-1, iz+1) + getVal(ix-1, iy-1, iz+1) -
                       getVal(ix+1, iy+1, iz-1) + 2.0*getVal(ix, iy+1, iz-1) - getVal(ix-1, iy+1, iz-1) +
                       2.0*getVal(ix+1, iy, iz-1) - 4.0*getVal(ix, iy, iz-1) + 2.0*getVal(ix-1, iy, iz-1) -
                       getVal(ix+1, iy-1, iz-1) + 2.0*getVal(ix, iy-1, iz-1) - getVal(ix-1, iy-1, iz-1)) /
                      (2.0 * dx * dx * dy * dy * dz);
    double dxxyzz_f = (getVal(ix+1, iy+1, iz+1) - 2.0*getVal(ix, iy+1, iz+1) + getVal(ix-1, iy+1, iz+1) -
                       2.0*getVal(ix+1, iy+1, iz) + 4.0*getVal(ix, iy+1, iz) - 2.0*getVal(ix-1, iy+1, iz) +
                       getVal(ix+1, iy+1, iz-1) - 2.0*getVal(ix, iy+1, iz-1) + getVal(ix-1, iy+1, iz-1) -
                       getVal(ix+1, iy-1, iz+1) + 2.0*getVal(ix, iy-1, iz+1) - getVal(ix-1, iy-1, iz+1) +
                       2.0*getVal(ix+1, iy-1, iz) - 4.0*getVal(ix, iy-1, iz) + 2.0*getVal(ix-1, iy-1, iz) -
                       getVal(ix+1, iy-1, iz-1) + 2.0*getVal(ix, iy-1, iz-1) - getVal(ix-1, iy-1, iz-1)) /
                      (2.0 * dx * dx * dy * dz * dz);
    double dxyyzz_f = (getVal(ix+1, iy+1, iz+1) - 2.0*getVal(ix+1, iy, iz+1) + getVal(ix+1, iy-1, iz+1) -
                       2.0*getVal(ix+1, iy+1, iz) + 4.0*getVal(ix+1, iy, iz) - 2.0*getVal(ix+1, iy-1, iz) +
                       getVal(ix+1, iy+1, iz-1) - 2.0*getVal(ix+1, iy, iz-1) + getVal(ix+1, iy-1, iz-1) -
                       getVal(ix-1, iy+1, iz+1) + 2.0*getVal(ix-1, iy, iz+1) - getVal(ix-1, iy-1, iz+1) +
                       2.0*getVal(ix-1, iy+1, iz) - 4.0*getVal(ix-1, iy, iz) + 2.0*getVal(ix-1, iy-1, iz) -
                       getVal(ix-1, iy+1, iz-1) + 2.0*getVal(ix-1, iy, iz-1) - getVal(ix-1, iy-1, iz-1)) /
                      (2.0 * dx * dy * dy * dz * dz);

    // Sixth derivative
    double dxxyyzz_f = (getVal(ix+1, iy+1, iz+1) - 2.0*getVal(ix, iy+1, iz+1) + getVal(ix-1, iy+1, iz+1) -
                        2.0*getVal(ix+1, iy, iz+1) + 4.0*getVal(ix, iy, iz+1) - 2.0*getVal(ix-1, iy, iz+1) +
                        getVal(ix+1, iy-1, iz+1) - 2.0*getVal(ix, iy-1, iz+1) + getVal(ix-1, iy-1, iz+1) -
                        2.0*getVal(ix+1, iy+1, iz) + 4.0*getVal(ix, iy+1, iz) - 2.0*getVal(ix-1, iy+1, iz) +
                        4.0*getVal(ix+1, iy, iz) - 8.0*f + 4.0*getVal(ix-1, iy, iz) -
                        2.0*getVal(ix+1, iy-1, iz) + 4.0*getVal(ix, iy-1, iz) - 2.0*getVal(ix-1, iy-1, iz) +
                        getVal(ix+1, iy+1, iz-1) - 2.0*getVal(ix, iy+1, iz-1) + getVal(ix-1, iy+1, iz-1) -
                        2.0*getVal(ix+1, iy, iz-1) + 4.0*getVal(ix, iy, iz-1) - 2.0*getVal(ix-1, iy, iz-1) +
                        getVal(ix+1, iy-1, iz-1) - 2.0*getVal(ix, iy-1, iz-1) + getVal(ix-1, iy-1, iz-1)) /
                       (dx * dx * dy * dy * dz * dz);

    // Return all 27 derivatives in order
    return {
        f,                              // 0: f
        dx_f, dy_f, dz_f,              // 1-3: first derivatives
        dxx_f, dyy_f, dzz_f,           // 4-6: second derivatives (pure)
        dxy_f, dxz_f, dyz_f,           // 7-9: second derivatives (mixed)
        dxxy_f, dxxz_f, dxyy_f, dyyz_f, dxzz_f, dyzz_f, dxyz_f,  // 10-16: third derivatives
        dxxyy_f, dxxzz_f, dyyzz_f, dxxyz_f, dxyyz_f, dxyzz_f,     // 17-22: fourth derivatives
        dxxyyz_f, dxxyzz_f, dxyyzz_f,  // 23-25: fifth derivatives
        dxxyyzz_f                       // 26: sixth derivative
    };
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
    const int nx = g_counts[0];
    const int ny = g_counts[1];
    const int nz = g_counts[2];
    const int nyz = ny * nz;

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
    const double U_MAX = g_gridCap;  // Configurable capping threshold

    // For each grid point, compute grid values
    int idx = 0;
    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            for (int k = 0; k < nz; k++) {
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

    // Compute derivatives if requested (using CAPPED grid values for stability)
    if (g_computeDerivatives) {
        g_derivatives.resize(27 * totalPoints, 0.0);

        int nOverlapPoints = 0;
        idx = 0;
        for (int i = 0; i < nx; i++) {
            for (int j = 0; j < ny; j++) {
                for (int k = 0; k < nz; k++) {
                    // Compute all 27 derivatives at this point from the capped grid
                    std::vector<double> derivs = computeDerivativesAtPoint(
                        g_vals, i, j, k,
                        g_spacing[0], g_spacing[1], g_spacing[2]
                    );

                    // Convert derivatives from physical coordinates to cell-local [0,1] coordinates
                    // Like RASPA3, we DIVIDE by spacing powers (not multiply!)
                    // This converts from physical nm^-n to cell-fractional coordinates
                    // Order: [f, fx,fy,fz, fxx,fxy,fxz,fyy,fyz,fzz, fxxy,fxxz,fxyy,fxyz,fxzz,fyyz,fyzz,
                    //         fxxyy,fxxzz,fyyzz,fxxyz,fxyyz,fxyzz, fxxyyz,fxxyzz,fxyyzz, fxxyyzz]

                    // Scaling factors for each derivative based on which variables it differentiates
                    double scaling[27];
                    scaling[0] = 1.0;  // f
                    scaling[1] = 1.0 / g_spacing[0];  // fx
                    scaling[2] = 1.0 / g_spacing[1];  // fy
                    scaling[3] = 1.0 / g_spacing[2];  // fz
                    scaling[4] = 1.0 / (g_spacing[0] * g_spacing[0]);  // fxx
                    scaling[5] = 1.0 / (g_spacing[0] * g_spacing[1]);  // fxy
                    scaling[6] = 1.0 / (g_spacing[0] * g_spacing[2]);  // fxz
                    scaling[7] = 1.0 / (g_spacing[1] * g_spacing[1]);  // fyy
                    scaling[8] = 1.0 / (g_spacing[1] * g_spacing[2]);  // fyz
                    scaling[9] = 1.0 / (g_spacing[2] * g_spacing[2]);  // fzz
                    scaling[10] = 1.0 / (g_spacing[0] * g_spacing[0] * g_spacing[1]);  // fxxy
                    scaling[11] = 1.0 / (g_spacing[0] * g_spacing[0] * g_spacing[2]);  // fxxz
                    scaling[12] = 1.0 / (g_spacing[0] * g_spacing[1] * g_spacing[1]);  // fxyy
                    scaling[13] = 1.0 / (g_spacing[0] * g_spacing[1] * g_spacing[2]);  // fxyz
                    scaling[14] = 1.0 / (g_spacing[0] * g_spacing[2] * g_spacing[2]);  // fxzz
                    scaling[15] = 1.0 / (g_spacing[1] * g_spacing[1] * g_spacing[2]);  // fyyz
                    scaling[16] = 1.0 / (g_spacing[1] * g_spacing[2] * g_spacing[2]);  // fyzz
                    scaling[17] = 1.0 / (g_spacing[0] * g_spacing[0] * g_spacing[1] * g_spacing[1]);  // fxxyy
                    scaling[18] = 1.0 / (g_spacing[0] * g_spacing[0] * g_spacing[2] * g_spacing[2]);  // fxxzz
                    scaling[19] = 1.0 / (g_spacing[1] * g_spacing[1] * g_spacing[2] * g_spacing[2]);  // fyyzz
                    scaling[20] = 1.0 / (g_spacing[0] * g_spacing[0] * g_spacing[1] * g_spacing[2]);  // fxxyz
                    scaling[21] = 1.0 / (g_spacing[0] * g_spacing[1] * g_spacing[1] * g_spacing[2]);  // fxyyz
                    scaling[22] = 1.0 / (g_spacing[0] * g_spacing[1] * g_spacing[2] * g_spacing[2]);  // fxyzz
                    scaling[23] = 1.0 / (g_spacing[0] * g_spacing[0] * g_spacing[1] * g_spacing[1] * g_spacing[2]);  // fxxyyz
                    scaling[24] = 1.0 / (g_spacing[0] * g_spacing[0] * g_spacing[1] * g_spacing[2] * g_spacing[2]);  // fxxyzz
                    scaling[25] = 1.0 / (g_spacing[0] * g_spacing[1] * g_spacing[1] * g_spacing[2] * g_spacing[2]);  // fxyyzz
                    scaling[26] = 1.0 / (g_spacing[0] * g_spacing[0] * g_spacing[1] * g_spacing[1] * g_spacing[2] * g_spacing[2]);  // fxxyyzz

                    // Debug: print raw derivatives for one point
                    static bool printed_derivs = false;
                    if (!printed_derivs && i == nx/2 && j == ny/2 && k == nz/2) {
                        printed_derivs = true;
                        std::cout << "Raw derivatives at grid center before scaling:" << std::endl;
                        std::cout << "  f = " << derivs[0] << std::endl;
                        std::cout << "  fx = " << derivs[1] << " (physical: kJ/mol/nm)" << std::endl;
                        std::cout << "  fy = " << derivs[2] << std::endl;
                        std::cout << "  fz = " << derivs[3] << std::endl;
                        std::cout << "  Grid spacing: " << g_spacing[0] << " nm" << std::endl;
                    }

                    // Store scaled derivatives
                    // Handle overlap regions like RASPA3: zero out higher derivatives if energy is capped
                    double gridValue = g_vals[idx];
                    bool isOverlap = (gridValue >= U_MAX * 0.999);  // Close to cap means we're in overlap region

                    if (isOverlap) nOverlapPoints++;

                    for (int d = 0; d < 27; d++) {
                        double val = derivs[d] * scaling[d];

                        // In overlap regions, clamp first derivatives and zero higher derivatives
                        if (isOverlap) {
                            if (d == 0) {
                                // Keep function value as is
                            } else if (d >= 1 && d <= 3) {
                                // Clamp first derivatives to reasonable values
                                val = std::max(-U_MAX, std::min(U_MAX, val));
                            } else {
                                // Zero out all second and higher derivatives
                                val = 0.0;
                            }
                        }

                        int deriv_idx = d * totalPoints + idx;
                        g_derivatives[deriv_idx] = val;
                    }

                    idx++;
                }
            }
        }

        std::cout << "Derivative computation: " << nOverlapPoints << " / " << totalPoints
                  << " points in overlap region (capped)" << std::endl;
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

    // Debug: print execution info
    static int exec_count = 0;
    static bool printed_header = false;
    if (!printed_header) {
        printed_header = true;
        std::cout << "\n=== EXECUTE FUNCTION DEBUG ===" << std::endl;
        std::cout << "Grid counts: " << g_counts[0] << "x" << g_counts[1] << "x" << g_counts[2] << std::endl;
        std::cout << "Grid spacing: " << g_spacing[0] << ", " << g_spacing[1] << ", " << g_spacing[2] << std::endl;
        std::cout << "Grid origin: " << g_origin_x << ", " << g_origin_y << ", " << g_origin_z << std::endl;
        std::cout << "Interpolation method: " << g_interpolationMethod << std::endl;
        std::cout << "Number of ligand atoms: " << natom_lig << std::endl;
        std::cout << "Scaling factors: ";
        for (int i = 0; i < natom_lig; i++) std::cout << g_scaling_factors[i] << " ";
        std::cout << std::endl;
        std::cout << "Has derivatives: " << (g_derivatives.empty() ? "NO" : "YES") << std::endl;
        if (!g_derivatives.empty()) {
            std::cout << "Derivative count: " << g_derivatives.size() << std::endl;
        }
    }
    exec_count++;

    for (int ia = 0; ia < natom_lig; ++ia) {
        // Get the actual particle index for this ligand atom
        int particle_idx = (g_ligand_atoms.empty()) ? ia : g_ligand_atoms[ia];

        // Transform position to grid coordinates (relative to origin)
        Vec3 pi_orig = posData[particle_idx];
        Vec3 pi(pi_orig[0] - g_origin_x, pi_orig[1] - g_origin_y, pi_orig[2] - g_origin_z);

        bool is_inside = true;
        for (int k = 0; k < 3; ++k) {
            if (pi[k] >= 0.0 && pi[k] <= hCorner[k])
                continue;
            else
                is_inside = false;
        }

        // Debug: print position info for first few atoms
        if (exec_count <= 2 && ia < 2) {
            std::cout << "Atom " << ia << ": pos_orig=(" << pi_orig[0] << "," << pi_orig[1] << "," << pi_orig[2] << ")"
                      << " -> pi=(" << pi[0] << "," << pi[1] << "," << pi[2] << ")"
                      << " is_inside=" << is_inside
                      << " scaling=" << g_scaling_factors[ia] << std::endl;
        }

        if (is_inside && g_scaling_factors[ia] != 0.0) {
            // Calculate base grid indices
            int ix = (int)(pi[0] / g_spacing[0]);
            int iy = (int)(pi[1] / g_spacing[1]);
            int iz = (int)(pi[2] / g_spacing[2]);

            // Fraction within the grid cell [0,1]
            double fx = (pi[0] / g_spacing[0]) - ix;
            double fy = (pi[1] / g_spacing[1]) - iy;
            double fz = (pi[2] / g_spacing[2]) - iz;

            double interpolated = 0.0;
            Vec3 grd(0.0, 0.0, 0.0);

            // Debug: print which interpolation path we're taking
            if (exec_count <= 2 && ia < 2) {
                std::cout << "  Cell indices: ix=" << ix << ", iy=" << iy << ", iz=" << iz
                          << ", fx=" << fx << ", fy=" << fy << ", fz=" << fz << std::endl;
                std::cout << "  Interpolation method: " << g_interpolationMethod << std::endl;
            }

            if (g_interpolationMethod == 1) {
                // CUBIC B-SPLINE INTERPOLATION (4x4x4 = 64 points)

                // Clamp indices to ensure we don't go out of bounds
                // For B-spline we need ix-1, ix, ix+1, ix+2
                int ix_start = std::max(0, ix - 1);
                int iy_start = std::max(0, iy - 1);
                int iz_start = std::max(0, iz - 1);

                int ix_end = std::min(g_counts[0] - 1, ix + 2);
                int iy_end = std::min(g_counts[1] - 1, iy + 2);
                int iz_end = std::min(g_counts[2] - 1, iz + 2);

                // Precompute basis functions for each dimension
                double bx[4] = {bspline_basis0(fx), bspline_basis1(fx), bspline_basis2(fx), bspline_basis3(fx)};
                double by[4] = {bspline_basis0(fy), bspline_basis1(fy), bspline_basis2(fy), bspline_basis3(fy)};
                double bz[4] = {bspline_basis0(fz), bspline_basis1(fz), bspline_basis2(fz), bspline_basis3(fz)};

                // Precompute derivatives for gradient calculation
                double dbx[4] = {bspline_deriv0(fx), bspline_deriv1(fx), bspline_deriv2(fx), bspline_deriv3(fx)};
                double dby[4] = {bspline_deriv0(fy), bspline_deriv1(fy), bspline_deriv2(fy), bspline_deriv3(fy)};
                double dbz[4] = {bspline_deriv0(fz), bspline_deriv1(fz), bspline_deriv2(fz), bspline_deriv3(fz)};

                // Tri-linear B-spline interpolation
                interpolated = 0.0;
                double dvdx = 0.0, dvdy = 0.0, dvdz = 0.0;

                for (int i = 0; i < 4; i++) {
                    int gx = std::min(std::max(ix - 1 + i, 0), g_counts[0] - 1);
                    for (int j = 0; j < 4; j++) {
                        int gy = std::min(std::max(iy - 1 + j, 0), g_counts[1] - 1);
                        for (int k = 0; k < 4; k++) {
                            int gz = std::min(std::max(iz - 1 + k, 0), g_counts[2] - 1);

                            // Get grid value
                            int gridIdx = gx * nyz + gy * g_counts[2] + gz;
                            double val = g_vals[gridIdx];

                            // Accumulate interpolated value
                            double weight = bx[i] * by[j] * bz[k];
                            interpolated += weight * val;

                            // Accumulate gradients
                            dvdx += dbx[i] * by[j] * bz[k] * val;
                            dvdy += bx[i] * dby[j] * bz[k] * val;
                            dvdz += bx[i] * by[j] * dbz[k] * val;
                        }
                    }
                }

                // Apply inverse power transformation if specified
                if (g_inv_power > 0.0) {
                    double base_interpolated = interpolated;
                    interpolated = pow(interpolated, g_inv_power);

                    // Apply chain rule to gradients: d/dx(f^n) = n * f^(n-1) * df/dx
                    double power_factor = g_inv_power * pow(base_interpolated, g_inv_power - 1.0);
                    dvdx *= power_factor;
                    dvdy *= power_factor;
                    dvdz *= power_factor;
                }

                // Convert gradients to forces (divide by spacing)
                grd = Vec3(dvdx / g_spacing[0], dvdy / g_spacing[1], dvdz / g_spacing[2]);

                // Energy and force
                energy += g_scaling_factors[ia] * interpolated;
                forceData[ia] -= g_scaling_factors[ia] * grd;

            } else if (g_interpolationMethod == 2) {
                // TRICUBIC HERMITE INTERPOLATION (2x2x2 cell with derivatives)
                // Uses cubic Hermite interpolation for exact interpolation with C1 continuity

                // Get 8 corner values of the cell
                int im = ix * nyz + iy * g_counts[2] + iz;
                int imp = im + g_counts[2];
                int ip = im + nyz;
                int ipp = ip + g_counts[2];

                double f000 = g_vals[im];
                double f001 = g_vals[im + 1];
                double f010 = g_vals[imp];
                double f011 = g_vals[imp + 1];
                double f100 = g_vals[ip];
                double f101 = g_vals[ip + 1];
                double f110 = g_vals[ipp];
                double f111 = g_vals[ipp + 1];

                // Estimate derivatives at corners using centered finite differences
                // For derivative in x-direction at each corner
                double dx000 = (ix > 0 && ix < g_counts[0]-1) ?
                    (g_vals[(ix+1)*nyz + iy*g_counts[2] + iz] - g_vals[(ix-1)*nyz + iy*g_counts[2] + iz]) / (2.0 * g_spacing[0]) : 0.0;
                double dx001 = (ix > 0 && ix < g_counts[0]-1) ?
                    (g_vals[(ix+1)*nyz + iy*g_counts[2] + iz+1] - g_vals[(ix-1)*nyz + iy*g_counts[2] + iz+1]) / (2.0 * g_spacing[0]) : 0.0;
                double dx010 = (ix > 0 && ix < g_counts[0]-1) ?
                    (g_vals[(ix+1)*nyz + (iy+1)*g_counts[2] + iz] - g_vals[(ix-1)*nyz + (iy+1)*g_counts[2] + iz]) / (2.0 * g_spacing[0]) : 0.0;
                double dx011 = (ix > 0 && ix < g_counts[0]-1) ?
                    (g_vals[(ix+1)*nyz + (iy+1)*g_counts[2] + iz+1] - g_vals[(ix-1)*nyz + (iy+1)*g_counts[2] + iz+1]) / (2.0 * g_spacing[0]) : 0.0;
                double dx100 = (ix > 0 && ix < g_counts[0]-1) ?
                    (g_vals[(ix+2)*nyz + iy*g_counts[2] + iz] - g_vals[ix*nyz + iy*g_counts[2] + iz]) / (2.0 * g_spacing[0]) : 0.0;
                double dx101 = (ix > 0 && ix < g_counts[0]-1) ?
                    (g_vals[(ix+2)*nyz + iy*g_counts[2] + iz+1] - g_vals[ix*nyz + iy*g_counts[2] + iz+1]) / (2.0 * g_spacing[0]) : 0.0;
                double dx110 = (ix > 0 && ix < g_counts[0]-1) ?
                    (g_vals[(ix+2)*nyz + (iy+1)*g_counts[2] + iz] - g_vals[ix*nyz + (iy+1)*g_counts[2] + iz]) / (2.0 * g_spacing[0]) : 0.0;
                double dx111 = (ix > 0 && ix < g_counts[0]-1) ?
                    (g_vals[(ix+2)*nyz + (iy+1)*g_counts[2] + iz+1] - g_vals[ix*nyz + (iy+1)*g_counts[2] + iz+1]) / (2.0 * g_spacing[0]) : 0.0;

                // Interpolate in x-direction first (8 1D interpolations -> 4 values)
                double h00_x = hermite_h00(fx), h01_x = hermite_h01(fx), h10_x = hermite_h10(fx), h11_x = hermite_h11(fx);
                double dh00_x = hermite_dh00(fx), dh01_x = hermite_dh01(fx), dh10_x = hermite_dh10(fx), dh11_x = hermite_dh11(fx);

                double v00 = h00_x * f000 + h01_x * f100 + h10_x * dx000 * g_spacing[0] + h11_x * dx100 * g_spacing[0];
                double v01 = h00_x * f001 + h01_x * f101 + h10_x * dx001 * g_spacing[0] + h11_x * dx101 * g_spacing[0];
                double v10 = h00_x * f010 + h01_x * f110 + h10_x * dx010 * g_spacing[0] + h11_x * dx110 * g_spacing[0];
                double v11 = h00_x * f011 + h01_x * f111 + h10_x * dx011 * g_spacing[0] + h11_x * dx111 * g_spacing[0];

                double dv00 = dh00_x * f000 + dh01_x * f100 + dh10_x * dx000 * g_spacing[0] + dh11_x * dx100 * g_spacing[0];
                double dv01 = dh00_x * f001 + dh01_x * f101 + dh10_x * dx001 * g_spacing[0] + dh11_x * dx101 * g_spacing[0];
                double dv10 = dh00_x * f010 + dh01_x * f110 + dh10_x * dx010 * g_spacing[0] + dh11_x * dx110 * g_spacing[0];
                double dv11 = dh00_x * f011 + dh01_x * f111 + dh10_x * dx011 * g_spacing[0] + dh11_x * dx111 * g_spacing[0];

                // Estimate y-derivatives for the interpolated values
                double dy00 = (iy > 0 && iy < g_counts[1]-1) ? (v10 - (h00_x * g_vals[im - g_counts[2]] + h01_x * g_vals[ip - g_counts[2]])) / g_spacing[1] : 0.0;
                double dy01 = (iy > 0 && iy < g_counts[1]-1) ? (v11 - (h00_x * g_vals[im + 1 - g_counts[2]] + h01_x * g_vals[ip + 1 - g_counts[2]])) / g_spacing[1] : 0.0;
                double dy10 = (iy > 0 && iy < g_counts[1]-1) ? ((h00_x * g_vals[im + 2*g_counts[2]] + h01_x * g_vals[ip + 2*g_counts[2]]) - v00) / g_spacing[1] : 0.0;
                double dy11 = (iy > 0 && iy < g_counts[1]-1) ? ((h00_x * g_vals[im + 1 + 2*g_counts[2]] + h01_x * g_vals[ip + 1 + 2*g_counts[2]]) - v01) / g_spacing[1] : 0.0;

                // Interpolate in y-direction (4 1D interpolations -> 2 values)
                double h00_y = hermite_h00(fy), h01_y = hermite_h01(fy), h10_y = hermite_h10(fy), h11_y = hermite_h11(fy);
                double dh00_y = hermite_dh00(fy), dh01_y = hermite_dh01(fy), dh10_y = hermite_dh10(fy), dh11_y = hermite_dh11(fy);

                double v0 = h00_y * v00 + h01_y * v10 + h10_y * dy00 * g_spacing[1] + h11_y * dy10 * g_spacing[1];
                double v1 = h00_y * v01 + h01_y * v11 + h10_y * dy01 * g_spacing[1] + h11_y * dy11 * g_spacing[1];

                double dvdx_0 = h00_y * dv00 + h01_y * dv10;
                double dvdx_1 = h00_y * dv01 + h01_y * dv11;
                double dvdy = (dh00_y * v00 + dh01_y * v10 + dh10_y * dy00 * g_spacing[1] + dh11_y * dy10 * g_spacing[1]);

                // Estimate z-derivatives
                double dz0 = (iz > 0 && iz < g_counts[2]-1) ? (v1 - (h00_y * (h00_x * g_vals[im - 1] + h01_x * g_vals[ip - 1]) + h01_y * (h00_x * g_vals[imp - 1] + h01_x * g_vals[ipp - 1]))) / g_spacing[2] : 0.0;
                double dz1 = (iz > 0 && iz < g_counts[2]-1) ? ((h00_y * (h00_x * g_vals[im + 2] + h01_x * g_vals[ip + 2]) + h01_y * (h00_x * g_vals[imp + 2] + h01_x * g_vals[ipp + 2])) - v0) / g_spacing[2] : 0.0;

                // Final interpolation in z-direction
                double h00_z = hermite_h00(fz), h01_z = hermite_h01(fz), h10_z = hermite_h10(fz), h11_z = hermite_h11(fz);
                double dh00_z = hermite_dh00(fz), dh01_z = hermite_dh01(fz), dh10_z = hermite_dh10(fz), dh11_z = hermite_dh11(fz);

                interpolated = h00_z * v0 + h01_z * v1 + h10_z * dz0 * g_spacing[2] + h11_z * dz1 * g_spacing[2];

                double dvdx = h00_z * dvdx_0 + h01_z * dvdx_1;
                double dvdz = dh00_z * v0 + dh01_z * v1 + dh10_z * dz0 * g_spacing[2] + dh11_z * dz1 * g_spacing[2];

                // Apply inverse power transformation if specified
                if (g_inv_power > 0.0) {
                    double base_interpolated = interpolated;
                    interpolated = pow(interpolated, g_inv_power);
                    double power_factor = g_inv_power * pow(base_interpolated, g_inv_power - 1.0);
                    dvdx *= power_factor;
                    dvdy *= power_factor;
                    dvdz *= power_factor;
                }

                // Convert gradients to forces
                grd = Vec3(dvdx / g_spacing[0], dvdy / g_spacing[1], dvdz / g_spacing[2]);

                // Energy and force
                energy += g_scaling_factors[ia] * interpolated;
                forceData[ia] -= g_scaling_factors[ia] * grd;

            } else if (g_interpolationMethod == 3) {
                // TRIQUINTIC HERMITE INTERPOLATION (C² continuous)
                static int triq_eval_count = 0;
                if (triq_eval_count < 5) {
                    triq_eval_count++;
                    std::cout << ">>> ENTERING TRIQUINTIC BRANCH (eval #" << triq_eval_count << ")" << std::endl;
                    std::cout << "    Ligand atom: " << ia << std::endl;
                    std::cout << "    Cell: (" << ix << "," << iy << "," << iz << ")" << std::endl;
                    std::cout << "    Local coords: (" << fx << "," << fy << "," << fz << ")" << std::endl;
                    std::cout << "    Derivatives empty? " << (g_derivatives.empty() ? "YES" : "NO") << std::endl;
                }
                // Uses tensor-product quintic Hermite interpolation with precomputed derivatives
                // Requires Version 2 grid format with 27 derivatives per point

                // Check if derivatives are available
                if (g_derivatives.empty()) {
                    std::cout << "ERROR: Derivatives are empty! Throwing exception..." << std::endl;
                    throw OpenMMException("GridForce: Triquintic interpolation (method=3) requires precomputed derivatives. Generate grid with setComputeDerivatives(True) or use a different interpolation method.");
                }

                // Get 8 corner indices of the enclosing cell
                int totalPoints = g_counts[0] * g_counts[1] * g_counts[2];

                // Corner indices in grid
                int x0 = ix, y0 = iy, z0 = iz;
                int x1 = ix + 1, y1 = iy + 1, z1 = iz + 1;

                // Gather 216 derivative values (27 derivatives × 8 corners)
                // Layout must match RASPA3: X[deriv_idx * 8 + corner_idx]
                // g_derivatives layout: [deriv_idx * totalPoints + (ix * nyz + iy * nz + iz)]
                std::vector<double> X(216);

                // Corners in order: (x0,y0,z0), (x1,y0,z0), (x0,y1,z0), (x1,y1,z0),
                //                   (x0,y0,z1), (x1,y0,z1), (x0,y1,z1), (x1,y1,z1)
                int corners[8][3] = {
                    {x0, y0, z0}, {x1, y0, z0}, {x0, y1, z0}, {x1, y1, z0},
                    {x0, y0, z1}, {x1, y0, z1}, {x0, y1, z1}, {x1, y1, z1}
                };

                // Gather in RASPA3 order: X[deriv_idx * 8 + corner_idx]
                for (int d = 0; d < 27; d++) {
                    for (int c = 0; c < 8; c++) {
                        int point_idx = corners[c][0] * nyz + corners[c][1] * g_counts[2] + corners[c][2];
                        X[d * 8 + c] = g_derivatives[d * totalPoints + point_idx];
                    }
                }

                // Compute polynomial coefficients: a = 0.125 * TRIQUINTIC_COEFFICIENTS * X
                std::vector<double> a(216, 0.0);
                const double scale = 0.125;  // 1/8 as specified by RASPA3
                for (int i = 0; i < 216; i++) {
                    for (int j = 0; j < 216; j++) {
                        a[i] += TRIQUINTIC_COEFFICIENTS[i][j] * X[j];
                    }
                    a[i] *= scale;
                }

                // Evaluate polynomial and its gradient
                // Position within cell [0,1]
                double s_x = fx;
                double s_y = fy;
                double s_z = fz;

                // Precompute powers
                double sx_pow[6], sy_pow[6], sz_pow[6];
                sx_pow[0] = sy_pow[0] = sz_pow[0] = 1.0;
                for (int p = 1; p < 6; p++) {
                    sx_pow[p] = sx_pow[p-1] * s_x;
                    sy_pow[p] = sy_pow[p-1] * s_y;
                    sz_pow[p] = sz_pow[p-1] * s_z;
                }

                // Evaluate polynomial: sum over i,j,k of a[i+6j+36k] * s_x^i * s_y^j * s_z^k
                double value = 0.0;
                double dvalue_dx = 0.0;
                double dvalue_dy = 0.0;
                double dvalue_dz = 0.0;

                for (int k = 0; k < 6; k++) {
                    for (int j = 0; j < 6; j++) {
                        for (int i = 0; i < 6; i++) {
                            int coeff_idx = i + 6*j + 36*k;
                            double coeff = a[coeff_idx];

                            // Value
                            value += coeff * sx_pow[i] * sy_pow[j] * sz_pow[k];

                            // Gradients (d/ds_x, d/ds_y, d/ds_z in local coordinates [0,1])
                            if (i > 0) dvalue_dx += coeff * i * sx_pow[i-1] * sy_pow[j] * sz_pow[k];
                            if (j > 0) dvalue_dy += coeff * j * sx_pow[i] * sy_pow[j-1] * sz_pow[k];
                            if (k > 0) dvalue_dz += coeff * k * sx_pow[i] * sy_pow[j] * sz_pow[k-1];
                        }
                    }
                }

                interpolated = value;

                // Convert gradients from local [0,1] coordinates to physical coordinates
                // Since we divided by spacing when storing, multiply by spacing to convert back
                // d/dx_physical = (d/ds_x) * grid_spacing
                double dvdx = dvalue_dx * g_spacing[0];
                double dvdy = dvalue_dy * g_spacing[1];
                double dvdz = dvalue_dz * g_spacing[2];

                // Apply inverse power transformation if specified
                if (g_inv_power > 0.0) {
                    double base_interpolated = interpolated;
                    interpolated = pow(interpolated, g_inv_power);
                    double power_factor = g_inv_power * pow(base_interpolated, g_inv_power - 1.0);
                    dvdx *= power_factor;
                    dvdy *= power_factor;
                    dvdz *= power_factor;
                }

                // Convert gradients to forces
                grd = Vec3(dvdx, dvdy, dvdz);

                // Energy and force
                energy += g_scaling_factors[ia] * interpolated;
                forceData[ia] -= g_scaling_factors[ia] * grd;

            } else {
                // TRILINEAR INTERPOLATION (default, 2x2x2 = 8 points)
                if (exec_count <= 2 && ia < 2) {
                    std::cout << ">>> USING TRILINEAR INTERPOLATION (default)" << std::endl;
                }

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

            // Fraction ahead (complement of fx, fy, fz)
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
            interpolated = ax * vm + fx * vp;

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
            grd = Vec3(dvdx / g_spacing[0], dvdy / g_spacing[1], dvdz / g_spacing[2]);

            // Apply chain rule if inv_power is set
            // d/dx(f^n) = n * f^(n-1) * df/dx
            if (g_inv_power > 0.0) {
                double base_interpolated = ax * vm + fx * vp;  // Value before power transform
                double power_factor = g_inv_power * pow(base_interpolated, g_inv_power - 1.0);
                grd = grd * power_factor;
            }

            forceData[ia] -= g_scaling_factors[ia] * grd;

            }  // End of if-else interpolation method selection

            // Debug: print energy contribution
            if (exec_count <= 2 && ia < 2) {
                double energy_contrib = g_scaling_factors[ia] * interpolated;
                std::cout << "  Interpolated value: " << interpolated << std::endl;
                std::cout << "  Energy contribution: " << energy_contrib
                          << " (scaling=" << g_scaling_factors[ia] << ")" << std::endl;
            }
        } else {
            // Out of bounds - apply restraint based on distance from grid boundaries
            // NOTE: This restraint is NOT scaled by scaling_factors - it applies uniformly
            // to all particles to keep them within the grid boundaries

            if (exec_count <= 2 && ia < 2) {
                std::cout << ">>> ATOM OUT OF BOUNDS" << std::endl;
                std::cout << "  Position: (" << pi[0] << "," << pi[1] << "," << pi[2] << ")" << std::endl;
                std::cout << "  Grid corner: (" << hCorner[0] << "," << hCorner[1] << "," << hCorner[2] << ")" << std::endl;
            }
            Vec3 grd(0.0, 0.0, 0.0);
            for (int k = 0; k < 3; k++) {
                double dev = 0.0;
                // Check distance from grid boundaries (in grid coordinates)
                if (pi[k] < 0.0) {
                    dev = pi[k];  // Negative distance from lower bound
                } else if (pi[k] > hCorner[k]) {
                    dev = pi[k] - hCorner[k];  // Positive distance from upper bound
                }
                energy += 0.5 * g_outOfBoundsRestraint * dev * dev;
                grd[k] = g_outOfBoundsRestraint * dev;
            }

            forceData[ia] -= grd;  // Don't scale the out-of-bounds restraint!
        }
    }

    return static_cast<double>(energy);
}

void ReferenceCalcGridForceKernel::copyParametersToContext(ContextImpl &context,
                                                           const GridForce &grid_force) {
    grid_force.getGridParameters(g_counts, g_spacing, g_vals, g_scaling_factors);
    g_inv_power = grid_force.getInvPower();
}

vector<double> ReferenceCalcGridForceKernel::getParticleGroupEnergies() {
    // Reference platform does not support per-group energy tracking yet
    return vector<double>();
}

vector<double> ReferenceCalcGridForceKernel::getParticleAtomEnergies() {
    // Reference platform does not support per-atom energy tracking yet
    return vector<double>();
}

}  // namespace AlGDockPlugin
