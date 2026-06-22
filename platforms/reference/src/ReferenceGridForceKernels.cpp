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
#include "TricubicMatrix.h"

#include "openmm/OpenMMException.h"
#include "openmm/HarmonicBondForce.h"
#include "openmm/HarmonicAngleForce.h"
#include "openmm/PeriodicTorsionForce.h"
#include "openmm/Vec3.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/reference/ReferencePlatform.h"
#include "openmm/NonbondedForce.h"
#include "IsolatedNonbondedForce.h"
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

// Second derivatives of cubic B-spline basis functions (match CUDA InterpolationBasis.cuh)
inline double bspline_deriv2_0(double t) { return 1.0 - t; }
inline double bspline_deriv2_1(double t) { return 3.0 * t - 2.0; }
inline double bspline_deriv2_2(double t) { return -3.0 * t + 1.0; }
inline double bspline_deriv2_3(double t) { return t; }

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

    // Get global alchemical scaling factor
    g_globalScalingFactor = grid_force.getGlobalScalingFactor();

    // Get per-group scaling factors and particle indices
    int numGroups = grid_force.getNumParticleGroups();
    g_groupScalingFactors.resize(numGroups);
    g_groupRuntimeCaps.resize(numGroups);
    g_groupParticleIndices.resize(numGroups);
    g_groupEnergies.resize(numGroups, 0.0);
    g_groupUnscaledEnergies.resize(numGroups, 0.0);
    g_atomToGroup.clear();
    for (int i = 0; i < numGroups; i++) {
        const auto& group = grid_force.getParticleGroup(i);
        g_groupScalingFactors[i] = group.groupScalingFactor;
        g_groupRuntimeCaps[i] = group.groupRuntimeCap;
        g_groupParticleIndices[i] = group.particleIndices;
        for (int atomIdx : group.particleIndices) {
            g_atomToGroup[atomIdx] = i;
        }
    }

    // Get ligand atom indices
    g_ligand_atoms = grid_force.getLigandAtoms();
    g_inv_power = grid_force.getInvPower();
    g_gridCap = grid_force.getGridCap();
    g_runtimeCap = grid_force.getRuntimeCap();
    g_outOfBoundsRestraint = grid_force.getOutOfBoundsRestraint();
    g_interpolationMethod = grid_force.getInterpolationMethod();
    grid_force.getGridOrigin(g_origin_x, g_origin_y, g_origin_z);

    // Compute effective bounds in grid-local coordinates
    if (grid_force.hasEffectiveBounds()) {
        double ebMinX, ebMinY, ebMinZ, ebMaxX, ebMaxY, ebMaxZ;
        grid_force.getEffectiveBounds(ebMinX, ebMinY, ebMinZ, ebMaxX, ebMaxY, ebMaxZ);
        g_effectiveMinX = ebMinX - g_origin_x;
        g_effectiveMinY = ebMinY - g_origin_y;
        g_effectiveMinZ = ebMinZ - g_origin_z;
        g_effectiveMaxX = ebMaxX - g_origin_x;
        g_effectiveMaxY = ebMaxY - g_origin_y;
        g_effectiveMaxZ = ebMaxZ - g_origin_z;
    } else {
        g_effectiveMinX = 0.0;
        g_effectiveMinY = 0.0;
        g_effectiveMinZ = 0.0;
        g_effectiveMaxX = g_spacing[0] * (g_counts[0] - 1);
        g_effectiveMaxY = g_spacing[1] * (g_counts[1] - 1);
        g_effectiveMaxZ = g_spacing[2] * (g_counts[2] - 1);
    }
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
        int numAtoms = system.getNumParticles();
        int numTemplateAtoms = (isolatedNonbondedForce != nullptr) ? isolatedNonbondedForce->getNumAtoms() : numAtoms;
        g_scaling_factors.resize(numAtoms);
        for (int i = 0; i < numAtoms; i++) {
            double charge, sigma, epsilon;

            // Get parameters from whichever force is available
            // IsolatedNonbondedForce uses template atoms (mod numTemplateAtoms)
            if (nonbondedForce != nullptr) {
                nonbondedForce->getParticleParameters(i, charge, sigma, epsilon);
            } else {
                isolatedNonbondedForce->getAtomParameters(i % numTemplateAtoms, charge, sigma, epsilon);
            }

            if (scalingProperty == "charge") {
                // For electrostatic grids: use charge directly
                g_scaling_factors[i] = charge;
            } else if (scalingProperty == "ljr") {
                // For LJ repulsive: sqrt(epsilon) * Rmin^6
                // where Rmin = 2^(1/6) * sigma (AMBER convention)
                double rmin = std::pow(2.0, 1.0/6.0) * sigma;
                g_scaling_factors[i] = std::sqrt(epsilon) * std::pow(rmin, 6.0);
            } else if (scalingProperty == "lja") {
                // For LJ attractive: sqrt(epsilon) * Rmin^3
                // where Rmin = 2^(1/6) * sigma (AMBER convention)
                double rmin = std::pow(2.0, 1.0/6.0) * sigma;
                g_scaling_factors[i] = std::sqrt(epsilon) * std::pow(rmin, 3.0);
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

        // Defer generation to the first execute(), where a Context (and the CPU
        // thread pool used by parallelFor) is available. System/Forces persist for
        // the Context lifetime, so capturing the pointers here is safe.
        genNeedsGrid_ = true;
        genSystem_ = &system;
        genNonbonded_ = nonbondedForce;
        genIsolated_ = isolatedNonbondedForce;
        genGridType_ = gridType;
        genReceptorAtoms_ = receptorAtoms;
        genReceptorPositions_ = receptorPositions;
        genOrigin_[0] = ox; genOrigin_[1] = oy; genOrigin_[2] = oz;

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
    const IsolatedNonbondedForce* isolatedNonbondedForce,
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
    const double U_MAX = g_gridCap;  // Configurable capping threshold

    // For each grid point, compute grid values. Points are independent (each writes
    // only g_vals[idx]), so parallelFor runs this serially on Reference and across
    // the thread pool on the CPU platform.
    parallelFor(totalPoints, [&](int idx) {
        int i = idx / nyz;
        int rem = idx % nyz;
        int j = rem / nz;
        int k = rem % nz;

        // Grid point position (in nm)
        double gx = originX + i * g_spacing[0];
        double gy = originY + j * g_spacing[1];
        double gz = originZ + k * g_spacing[2];

        // Calculate contribution from each receptor atom
        double gridValue = 0.0;
        for (size_t atomIdx = 0; atomIdx < receptorAtoms.size(); atomIdx++) {
            Vec3 atomPos = receptorPositions[atomIdx];
            double dx = gx - atomPos[0];
            double dy = gy - atomPos[1];
            double dz = gz - atomPos[2];
            double r = std::sqrt(dx*dx + dy*dy + dz*dz);
            if (r < 1e-6) r = 1e-6;  // avoid singularities

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
        g_vals[idx] = U_MAX * std::tanh(gridValue / U_MAX);
    });

    // Compute derivatives if requested (using CAPPED grid values for stability).
    // Reads the completed g_vals stencil (read-only) and writes disjoint output
    // slots per point, so parallelFor is safe.
    if (g_computeDerivatives) {
        g_derivatives.resize(27 * totalPoints, 0.0);

        parallelFor(totalPoints, [&](int idx) {
                    int i = idx / nyz;
                    int rem = idx % nyz;
                    int j = rem / nz;
                    int k = rem % nz;
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

                    // Store scaled derivatives.
                    // Handle overlap regions like RASPA3: zero out higher derivatives if energy is capped
                    double gridValue = g_vals[idx];
                    bool isOverlap = (gridValue >= U_MAX * 0.999);  // Close to cap means we're in overlap region

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
        });
    }
}

static inline void applyRuntimeCap(double cap, double& interpolated, Vec3& grd) {
    if (cap > 0.0) {
        // Tanh cap: f(v) = C * tanh(v/C), bounded by ±C
        // Gradient factor: sech²(v/C) = 1 - tanh²(v/C)
        double t = std::tanh(interpolated / cap);
        double sech2 = 1.0 - t * t;
        interpolated = cap * t;
        grd = grd * sech2;
    }
}

void ReferenceCalcGridForceKernel::computeAtom(int ia,
                                               vector<Vec3>& posData,
                                               vector<Vec3>& forceData,
                                               bool includeForces,
                                               bool includeEnergy,
                                               double& atomEnergy,
                                               double& atomUnscaled,
                                               int& groupIdx) {

    atomEnergy = 0.0;
    atomUnscaled = 0.0;
    groupIdx = -1;

    const int nyz = g_counts[1] * g_counts[2];

    {
        // Get the actual particle index for this ligand atom
        int particle_idx = (g_ligand_atoms.empty()) ? ia : g_ligand_atoms[ia];

        // Transform position to grid coordinates (relative to origin)
        Vec3 pi_orig = posData[particle_idx];
        Vec3 pi(pi_orig[0] - g_origin_x, pi_orig[1] - g_origin_y, pi_orig[2] - g_origin_z);

        // Check against effective bounds (defaults to full grid extent)
        double effMin[3] = {g_effectiveMinX, g_effectiveMinY, g_effectiveMinZ};
        double effMax[3] = {g_effectiveMaxX, g_effectiveMaxY, g_effectiveMaxZ};
        bool is_inside = true;
        for (int k = 0; k < 3; ++k) {
            if (pi[k] >= effMin[k] && pi[k] <= effMax[k])
                continue;
            else
                is_inside = false;
        }

        // Compute effective scaling factor including global and per-group alchemical scaling
        double groupScaling = 1.0;
        auto it = g_atomToGroup.find(particle_idx);
        if (it != g_atomToGroup.end()) {
            groupIdx = it->second;
            groupScaling = g_groupScalingFactors[groupIdx];
        }
        double effectiveScaling = g_globalScalingFactor * groupScaling * g_scaling_factors[ia];
        double unscaledScaling = g_globalScalingFactor * g_scaling_factors[ia];

        // Resolve effective runtime cap: per-group if available, else global
        double effectiveCap = g_runtimeCap;
        if (groupIdx >= 0 && groupIdx < (int)g_groupRuntimeCaps.size() && g_groupRuntimeCaps[groupIdx] > 0.0) {
            effectiveCap = g_groupRuntimeCaps[groupIdx];
        }

        if (is_inside && (effectiveScaling != 0.0 || (groupIdx >= 0 && unscaledScaling != 0.0))) {
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

                // Apply runtime cap (tanh capping after interpolation)
                applyRuntimeCap(effectiveCap, interpolated, grd);

                // Energy and force
                atomEnergy += effectiveScaling * interpolated;
                forceData[ia] -= effectiveScaling * grd;

            } else if (g_interpolationMethod == 2) {
                // LEKIEN-MARSDEN TRICUBIC HERMITE (C1, matches CUDA/RASPA3). Uses the
                // precomputed derivative grid (8 of the 27 derivatives per corner) and the
                // 64x64 coefficient matrix -- the same construction as the CUDA kernel and the
                // triquintic path below -- replacing the earlier finite-difference Hermite.
                if (g_derivatives.empty()) {
                    throw OpenMMException("GridForce: Tricubic interpolation (method=2) requires precomputed derivatives. Generate grid with setComputeDerivatives(True) or use a different interpolation method.");
                }
                int totalPoints = g_counts[0] * g_counts[1] * g_counts[2];
                int corners[8][3] = {
                    {ix, iy, iz}, {ix+1, iy, iz}, {ix, iy+1, iz}, {ix+1, iy+1, iz},
                    {ix, iy, iz+1}, {ix+1, iy, iz+1}, {ix, iy+1, iz+1}, {ix+1, iy+1, iz+1}
                };
                // Tricubic needs {f,fx,fy,fz,fxy,fxz,fyz,fxyz} from the RASPA3 27-derivative order.
                const int derivMap[8] = {0, 1, 2, 3, 5, 6, 8, 13};
                double X[64];
                for (int d = 0; d < 8; d++)
                    for (int c = 0; c < 8; c++) {
                        int point_idx = corners[c][0] * nyz + corners[c][1] * g_counts[2] + corners[c][2];
                        X[d * 8 + c] = g_derivatives[derivMap[d] * totalPoints + point_idx];
                    }
                double a[64];
                tricubicAssemble(X, a);
                double value, dvalue_dx, dvalue_dy, dvalue_dz;
                tricubicEvalVG(a, fx, fy, fz, &value, &dvalue_dx, &dvalue_dy, &dvalue_dz);
                interpolated = value;
                // Stored derivatives are divided by spacing^n (see generateGrid), so the unit-cell
                // gradient is multiplied by spacing to recover physical units (matches triquintic/CUDA).
                double dvdx = dvalue_dx * g_spacing[0];
                double dvdy = dvalue_dy * g_spacing[1];
                double dvdz = dvalue_dz * g_spacing[2];
                if (g_inv_power > 0.0) {
                    double base_interpolated = interpolated;
                    interpolated = pow(interpolated, g_inv_power);
                    double power_factor = g_inv_power * pow(base_interpolated, g_inv_power - 1.0);
                    dvdx *= power_factor; dvdy *= power_factor; dvdz *= power_factor;
                }
                grd = Vec3(dvdx, dvdy, dvdz);
                applyRuntimeCap(effectiveCap, interpolated, grd);
                atomEnergy += effectiveScaling * interpolated;
                forceData[ia] -= effectiveScaling * grd;
            } else if (g_interpolationMethod == 3) {
                // TRIQUINTIC HERMITE INTERPOLATION (C² continuous)
                // Uses tensor-product quintic Hermite interpolation with precomputed derivatives
                // Requires Version 2 grid format with 27 derivatives per point

                // Check if derivatives are available
                if (g_derivatives.empty()) {
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

                // Apply runtime cap (tanh capping after interpolation)
                applyRuntimeCap(effectiveCap, interpolated, grd);

                // Energy and force
                atomEnergy += effectiveScaling * interpolated;
                forceData[ia] -= effectiveScaling * grd;

            } else {
                // TRILINEAR INTERPOLATION (default, 2x2x2 = 8 points)

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

            // Apply runtime cap (tanh capping after interpolation)
            applyRuntimeCap(effectiveCap, interpolated, grd);

            atomEnergy += effectiveScaling * interpolated;
            forceData[ia] -= effectiveScaling * grd;

            }  // End of if-else interpolation method selection

            // Track unscaled per-group energy (omits group scaling factor)
            if (groupIdx >= 0) {
                atomUnscaled += unscaledScaling * interpolated;
            }
        } else {
            // Out of bounds - apply restraint based on distance from grid boundaries
            // NOTE: This restraint is NOT scaled by scaling_factors - it applies uniformly
            // to all particles to keep them within the grid boundaries
            double oobEnergy = 0.0;
            Vec3 grd(0.0, 0.0, 0.0);
            for (int k = 0; k < 3; k++) {
                double dev = 0.0;
                // Check distance from effective bounds
                if (pi[k] < effMin[k]) {
                    dev = pi[k] - effMin[k];  // Negative distance from lower bound
                } else if (pi[k] > effMax[k]) {
                    dev = pi[k] - effMax[k];  // Positive distance from upper bound
                }
                double oobTerm = 0.5 * g_outOfBoundsRestraint * dev * dev;
                atomEnergy += oobTerm;
                oobEnergy += oobTerm;
                grd[k] = g_outOfBoundsRestraint * dev;
            }

            forceData[ia] -= grd;  // Don't scale the out-of-bounds restraint!

            // Track OOB energy in unscaled buffer too (OOB is not group-scaled)
            if (groupIdx >= 0) {
                atomUnscaled += oobEnergy;
            }
        }
    }
}

void ReferenceCalcGridForceKernel::runAtoms(ContextImpl& context,
                                            vector<Vec3>& posData,
                                            vector<Vec3>& forceData,
                                            bool includeForces,
                                            bool includeEnergy) {
    int natom_lig = g_scaling_factors.size();
    g_atomEnergyContribution.resize(natom_lig);
    g_atomUnscaledContribution.resize(natom_lig);
    g_atomGroupIdx.resize(natom_lig);

    for (int ia = 0; ia < natom_lig; ++ia) {
        computeAtom(ia, posData, forceData, includeForces, includeEnergy,
                    g_atomEnergyContribution[ia], g_atomUnscaledContribution[ia],
                    g_atomGroupIdx[ia]);
    }
}

double ReferenceCalcGridForceKernel::execute(ContextImpl &context,
                                             bool includeForces,
                                             bool includeEnergy) {

    g_lastContext = &context;

    // Build the auto-generated grid on first use (deferred from initialize() so the
    // CPU thread pool is available for parallel generation).
    if (genNeedsGrid_ && g_vals.empty()) {
        generateGrid(*genSystem_, genNonbonded_, genIsolated_, genGridType_,
                     genReceptorAtoms_, genReceptorPositions_,
                     genOrigin_[0], genOrigin_[1], genOrigin_[2]);
        genNeedsGrid_ = false;
    }

    vector<Vec3> &posData = extractPositions(context);
    vector<Vec3> &forceData = extractForces(context);

    const int nyz = g_counts[1] * g_counts[2];
    Vec3 hCorner(g_spacing[0] * (g_counts[0] - 1),
                 g_spacing[1] * (g_counts[1] - 1),
                 g_spacing[2] * (g_counts[2] - 1));

    int natom_lig = g_scaling_factors.size();

    // Reset per-group energies
    for (size_t g = 0; g < g_groupEnergies.size(); g++) {
        g_groupEnergies[g] = 0.0;
    }
    for (size_t g = 0; g < g_groupUnscaledEnergies.size(); g++) {
        g_groupUnscaledEnergies[g] = 0.0;
    }

    runAtoms(context, posData, forceData, includeForces, includeEnergy);

    // Reduce the per-atom contributions serially in atom order so the result is
    // bit-for-bit identical to the original serial accumulation.
    double energy = 0.0;
    for (int ia = 0; ia < natom_lig; ++ia) {
        energy += g_atomEnergyContribution[ia];
        int gi = g_atomGroupIdx[ia];
        if (gi >= 0) {
            g_groupEnergies[gi] += g_atomEnergyContribution[ia];
            g_groupUnscaledEnergies[gi] += g_atomUnscaledContribution[ia];
        }
    }

    return static_cast<double>(energy);
}

void ReferenceCalcGridForceKernel::copyParametersToContext(ContextImpl &context,
                                                           const GridForce &grid_force) {
    grid_force.getGridParameters(g_counts, g_spacing, g_vals, g_scaling_factors);
    g_inv_power = grid_force.getInvPower();
    g_runtimeCap = grid_force.getRuntimeCap();
    g_globalScalingFactor = grid_force.getGlobalScalingFactor();

    // Update per-group scaling factors and runtime caps
    int numGroups = grid_force.getNumParticleGroups();
    g_groupScalingFactors.resize(numGroups);
    g_groupRuntimeCaps.resize(numGroups);
    for (int i = 0; i < numGroups; i++) {
        g_groupScalingFactors[i] = grid_force.getParticleGroupScalingFactor(i);
        g_groupRuntimeCaps[i] = grid_force.getParticleGroupRuntimeCap(i);
    }
}

vector<double> ReferenceCalcGridForceKernel::getParticleGroupEnergies() {
    return g_groupEnergies;
}

vector<double> ReferenceCalcGridForceKernel::getParticleGroupUnscaledEnergies() {
    return g_groupUnscaledEnergies;
}

vector<double> ReferenceCalcGridForceKernel::getParticleAtomEnergies() {
    // Reference platform does not support per-atom energy tracking yet
    return vector<double>();
}

vector<double> ReferenceCalcGridForceKernel::getParticleGroupAtomRawEnergies() {
    // Reference platform does not support per-atom raw energy tracking yet
    return vector<double>();
}

vector<int> ReferenceCalcGridForceKernel::getParticleOutOfBoundsFlags() {
    // Reference platform does not support per-atom out-of-bounds tracking yet
    return vector<int>();
}

// ============================================================================
// Per-atom Hessian (block-diagonal 3x3) implementation
// ============================================================================
//
// GridForce is a per-atom external potential, so the Hessian is block-diagonal:
// one symmetric 3x3 block per atom. Each block is d2E/dx_a dx_b for that atom,
// where E = effectiveScaling * cap(invPower(interp(pos))). The math mirrors the
// CUDA gridHessian.cu kernel and extends the Reference force path in execute():
// the same interpolant value and gradient are computed, plus the interpolant
// second derivative, then chained through the inv_power and runtime-cap
// transforms and multiplied by the effective scaling factor.
//
// Output layout matches CUDA getHessianBlocks(): 6 doubles per atom in order
// [xx, yy, zz, xy, xz, yz]. Out-of-bounds atoms get a zero block (the out-of-
// bounds restraint is a global quadratic that this per-atom grid Hessian does
// not include, matching the CUDA kernel which leaves the block at zero).

namespace {

// Symmetric 3x3 block stored as [xx, yy, zz, xy, xz, yz].
struct Block3 {
    double xx = 0.0, yy = 0.0, zz = 0.0, xy = 0.0, xz = 0.0, yz = 0.0;
};

// Apply the inv_power transform W = V^n (Reference semantics, n = g_inv_power > 0)
// to the value, gradient (gx,gy,gz) and Hessian block, all in physical units.
//   W'  = n * V^(n-1)
//   W'' = n*(n-1) * V^(n-2)
//   dW/dx           = W' * dV/dx
//   d2W/dx_a dx_b   = W'' * dV/dx_a dV/dx_b + W' * d2V/dx_a dx_b
inline void applyInvPowerChain(double invPower, double& v,
                               double& gx, double& gy, double& gz,
                               Block3& H) {
    double base = v;
    double w1 = invPower * std::pow(base, invPower - 1.0);
    double w2 = invPower * (invPower - 1.0) * std::pow(base, invPower - 2.0);

    H.xx = w2 * gx * gx + w1 * H.xx;
    H.yy = w2 * gy * gy + w1 * H.yy;
    H.zz = w2 * gz * gz + w1 * H.zz;
    H.xy = w2 * gx * gy + w1 * H.xy;
    H.xz = w2 * gx * gz + w1 * H.xz;
    H.yz = w2 * gy * gz + w1 * H.yz;

    gx *= w1;
    gy *= w1;
    gz *= w1;
    v = std::pow(base, invPower);
}

// Apply the runtime tanh cap Y = C * tanh(W/C) to value, gradient and Hessian.
//   Y'  = sech2(W/C)
//   Y'' = -(2/C) * sech2(W/C) * tanh(W/C)
//   d2Y/dx_a dx_b = Y' * d2W/dx_a dx_b + Y'' * dW/dx_a dW/dx_b
inline void applyRuntimeCapChain(double cap, double& v,
                                 double& gx, double& gy, double& gz,
                                 Block3& H) {
    double t = std::tanh(v / cap);
    double sech2 = 1.0 - t * t;
    double y1 = sech2;
    double y2 = -2.0 * t * sech2 / cap;

    H.xx = y1 * H.xx + y2 * gx * gx;
    H.yy = y1 * H.yy + y2 * gy * gy;
    H.zz = y1 * H.zz + y2 * gz * gz;
    H.xy = y1 * H.xy + y2 * gx * gy;
    H.xz = y1 * H.xz + y2 * gx * gz;
    H.yz = y1 * H.yz + y2 * gy * gz;

    gx *= y1;
    gy *= y1;
    gz *= y1;
    v = cap * t;
}

}  // namespace

void ReferenceCalcGridForceKernel::parallelFor(int count, const std::function<void(int)>& body) {
    for (int i = 0; i < count; i++)
        body(i);
}

void ReferenceCalcGridForceKernel::computeHessianForPositions(const std::vector<Vec3>& posData) {
    const int nyz = g_counts[1] * g_counts[2];
    const int natom_lig = (int)g_scaling_factors.size();
    const int nx = g_counts[0], ny = g_counts[1], nz = g_counts[2];

    g_hessianBlocks.assign(6 * natom_lig, 0.0);

    const double effMin[3] = {g_effectiveMinX, g_effectiveMinY, g_effectiveMinZ};
    const double effMax[3] = {g_effectiveMaxX, g_effectiveMaxY, g_effectiveMaxZ};

    parallelFor(natom_lig, [&](int ia) {
        int particle_idx = (g_ligand_atoms.empty()) ? ia : g_ligand_atoms[ia];

        Vec3 pi_orig = posData[particle_idx];
        Vec3 pi(pi_orig[0] - g_origin_x, pi_orig[1] - g_origin_y, pi_orig[2] - g_origin_z);

        bool is_inside = true;
        for (int k = 0; k < 3; ++k) {
            if (!(pi[k] >= effMin[k] && pi[k] <= effMax[k]))
                is_inside = false;
        }

        double groupScaling = 1.0;
        int groupIdx = -1;
        auto it = g_atomToGroup.find(particle_idx);
        if (it != g_atomToGroup.end()) {
            groupIdx = it->second;
            groupScaling = g_groupScalingFactors[groupIdx];
        }
        double effectiveScaling = g_globalScalingFactor * groupScaling * g_scaling_factors[ia];

        double effectiveCap = g_runtimeCap;
        if (groupIdx >= 0 && groupIdx < (int)g_groupRuntimeCaps.size() && g_groupRuntimeCaps[groupIdx] > 0.0) {
            effectiveCap = g_groupRuntimeCaps[groupIdx];
        }

        if (!is_inside || effectiveScaling == 0.0)
            return;  // leave the block at zero (matches CUDA)

        int ix = (int)(pi[0] / g_spacing[0]);
        int iy = (int)(pi[1] / g_spacing[1]);
        int iz = (int)(pi[2] / g_spacing[2]);
        double fx = (pi[0] / g_spacing[0]) - ix;
        double fy = (pi[1] / g_spacing[1]) - iy;
        double fz = (pi[2] / g_spacing[2]) - iz;

        double interpolated = 0.0;
        double gx = 0.0, gy = 0.0, gz = 0.0;     // physical gradient dV/dx
        Block3 H;                                 // physical Hessian d2V/dx dx

        if (g_interpolationMethod == 1) {
            // CUBIC B-SPLINE (4x4x4 stencil)
            double bx[4] = {bspline_basis0(fx), bspline_basis1(fx), bspline_basis2(fx), bspline_basis3(fx)};
            double by[4] = {bspline_basis0(fy), bspline_basis1(fy), bspline_basis2(fy), bspline_basis3(fy)};
            double bz[4] = {bspline_basis0(fz), bspline_basis1(fz), bspline_basis2(fz), bspline_basis3(fz)};
            double dbx[4] = {bspline_deriv0(fx), bspline_deriv1(fx), bspline_deriv2(fx), bspline_deriv3(fx)};
            double dby[4] = {bspline_deriv0(fy), bspline_deriv1(fy), bspline_deriv2(fy), bspline_deriv3(fy)};
            double dbz[4] = {bspline_deriv0(fz), bspline_deriv1(fz), bspline_deriv2(fz), bspline_deriv3(fz)};
            double d2bx[4] = {bspline_deriv2_0(fx), bspline_deriv2_1(fx), bspline_deriv2_2(fx), bspline_deriv2_3(fx)};
            double d2by[4] = {bspline_deriv2_0(fy), bspline_deriv2_1(fy), bspline_deriv2_2(fy), bspline_deriv2_3(fy)};
            double d2bz[4] = {bspline_deriv2_0(fz), bspline_deriv2_1(fz), bspline_deriv2_2(fz), bspline_deriv2_3(fz)};

            for (int i = 0; i < 4; i++) {
                int ggx = std::min(std::max(ix - 1 + i, 0), nx - 1);
                for (int j = 0; j < 4; j++) {
                    int ggy = std::min(std::max(iy - 1 + j, 0), ny - 1);
                    for (int k = 0; k < 4; k++) {
                        int ggz = std::min(std::max(iz - 1 + k, 0), nz - 1);
                        double val = g_vals[ggx * nyz + ggy * nz + ggz];
                        interpolated += bx[i] * by[j] * bz[k] * val;
                        gx += dbx[i] * by[j] * bz[k] * val;
                        gy += bx[i] * dby[j] * bz[k] * val;
                        gz += bx[i] * by[j] * dbz[k] * val;
                        H.xx += d2bx[i] * by[j] * bz[k] * val;
                        H.yy += bx[i] * d2by[j] * bz[k] * val;
                        H.zz += bx[i] * by[j] * d2bz[k] * val;
                        H.xy += dbx[i] * dby[j] * bz[k] * val;
                        H.xz += dbx[i] * by[j] * dbz[k] * val;
                        H.yz += bx[i] * dby[j] * dbz[k] * val;
                    }
                }
            }
            // unit-cell -> physical (grid values stored directly: divide by spacing)
            gx /= g_spacing[0]; gy /= g_spacing[1]; gz /= g_spacing[2];
            H.xx /= g_spacing[0] * g_spacing[0];
            H.yy /= g_spacing[1] * g_spacing[1];
            H.zz /= g_spacing[2] * g_spacing[2];
            H.xy /= g_spacing[0] * g_spacing[1];
            H.xz /= g_spacing[0] * g_spacing[2];
            H.yz /= g_spacing[1] * g_spacing[2];

        } else if (g_interpolationMethod == 2) {
            // Tricubic Hermite Hessian is unsupported, mirroring CUDA (computeGridHessian
            // handles only methods 1, 3, 4). The force path is C1, so its second
            // derivatives are discontinuous across cell faces; use triquintic (method 3)
            // when a grid Hessian is required.
            throw OpenMMException("GridForce: Hessian not supported for tricubic Hermite (method 2)");

        } else if (g_interpolationMethod == 3) {
            // TRIQUINTIC HERMITE. Mirrors the execute() assembly: build the 216
            // polynomial coefficients, evaluate value/gradient/Hessian in unit-cell
            // coords, then convert to physical units. Note the Reference stores
            // corner derivatives divided by spacing^n, so (matching the force path)
            // the unit-cell gradient is MULTIPLIED by spacing and the Hessian by
            // spacing^2 -- this yields the same physical-unit values as CUDA.
            if (g_derivatives.empty()) {
                throw OpenMMException("GridForce: Triquintic Hessian (method=3) requires precomputed derivatives.");
            }
            int totalPoints = nx * ny * nz;
            int x0 = ix, y0 = iy, z0 = iz, x1 = ix + 1, y1 = iy + 1, z1 = iz + 1;
            int corners[8][3] = {
                {x0, y0, z0}, {x1, y0, z0}, {x0, y1, z0}, {x1, y1, z0},
                {x0, y0, z1}, {x1, y0, z1}, {x0, y1, z1}, {x1, y1, z1}
            };
            std::vector<double> X(216);
            for (int d = 0; d < 27; d++) {
                for (int c = 0; c < 8; c++) {
                    int point_idx = corners[c][0] * nyz + corners[c][1] * nz + corners[c][2];
                    X[d * 8 + c] = g_derivatives[d * totalPoints + point_idx];
                }
            }
            std::vector<double> a(216, 0.0);
            const double scale = 0.125;
            for (int i = 0; i < 216; i++) {
                for (int j = 0; j < 216; j++)
                    a[i] += TRIQUINTIC_COEFFICIENTS[i][j] * X[j];
                a[i] *= scale;
            }
            double sx[6], sy[6], sz[6];
            sx[0] = sy[0] = sz[0] = 1.0;
            for (int p = 1; p < 6; p++) {
                sx[p] = sx[p-1] * fx;
                sy[p] = sy[p-1] * fy;
                sz[p] = sz[p-1] * fz;
            }
            double value = 0.0, dvx = 0.0, dvy = 0.0, dvz = 0.0;
            double hxx = 0.0, hyy = 0.0, hzz = 0.0, hxy = 0.0, hxz = 0.0, hyz = 0.0;
            for (int k = 0; k < 6; k++) {
                for (int j = 0; j < 6; j++) {
                    for (int i = 0; i < 6; i++) {
                        double c = a[i + 6*j + 36*k];
                        value += c * sx[i] * sy[j] * sz[k];
                        if (i > 0) dvx += c * i * sx[i-1] * sy[j] * sz[k];
                        if (j > 0) dvy += c * j * sx[i] * sy[j-1] * sz[k];
                        if (k > 0) dvz += c * k * sx[i] * sy[j] * sz[k-1];
                        if (i > 1) hxx += c * i * (i-1) * sx[i-2] * sy[j] * sz[k];
                        if (j > 1) hyy += c * j * (j-1) * sx[i] * sy[j-2] * sz[k];
                        if (k > 1) hzz += c * k * (k-1) * sx[i] * sy[j] * sz[k-2];
                        if (i > 0 && j > 0) hxy += c * i * j * sx[i-1] * sy[j-1] * sz[k];
                        if (i > 0 && k > 0) hxz += c * i * k * sx[i-1] * sy[j] * sz[k-1];
                        if (j > 0 && k > 0) hyz += c * j * k * sx[i] * sy[j-1] * sz[k-1];
                    }
                }
            }
            interpolated = value;
            // unit-cell -> physical (matches force-path convention: MULTIPLY by spacing)
            gx = dvx * g_spacing[0];
            gy = dvy * g_spacing[1];
            gz = dvz * g_spacing[2];
            H.xx = hxx * g_spacing[0] * g_spacing[0];
            H.yy = hyy * g_spacing[1] * g_spacing[1];
            H.zz = hzz * g_spacing[2] * g_spacing[2];
            H.xy = hxy * g_spacing[0] * g_spacing[1];
            H.xz = hxz * g_spacing[0] * g_spacing[2];
            H.yz = hyz * g_spacing[1] * g_spacing[2];

        } else {
            // TRILINEAR. The interpolant is multilinear, so all pure second
            // derivatives vanish; only the three mixed terms are nonzero within
            // the cell. Build from the 8 corner values directly.
            int im = ix * nyz + iy * nz + iz;
            int imp = im + nz;
            int ip = im + nyz;
            int ipp = ip + nz;
            double vmmm = g_vals[im],   vmmp = g_vals[im + 1];
            double vmpm = g_vals[imp],  vmpp = g_vals[imp + 1];
            double vpmm = g_vals[ip],   vpmp = g_vals[ip + 1];
            double vppm = g_vals[ipp],  vppp = g_vals[ipp + 1];
            double ax = 1.0 - fx, ay = 1.0 - fy, az = 1.0 - fz;

            double vmm = az * vmmm + fz * vmmp;
            double vmp = az * vmpm + fz * vmpp;
            double vpm = az * vpmm + fz * vpmp;
            double vpp = az * vppm + fz * vppp;
            double vm = ay * vmm + fy * vmp;
            double vp = ay * vpm + fy * vpp;
            interpolated = ax * vm + fx * vp;

            // fractional gradient (matches execute())
            double gfx = -vm + vp;
            double gfy = (-vmm + vmp) * ax + (-vpm + vpp) * fx;
            double gfz = ((-vmmm + vmmp) * ay + (-vmpm + vmpp) * fy) * ax +
                         ((-vpmm + vpmp) * ay + (-vppm + vppp) * fy) * fx;

            // fractional second derivatives (pure terms are zero)
            // d2/dfx dfy : derivative of gfx w.r.t. fy = -(dvm/dfy) + (dvp/dfy)
            double dvm_dfy = -vmm + vmp;     // d vm / dfy
            double dvp_dfy = -vpm + vpp;     // d vp / dfy
            double hfxy = -dvm_dfy + dvp_dfy;
            // d2/dfx dfz : derivative of gfx w.r.t fz
            double dvm_dfz = ay * (-vmmm + vmmp) + fy * (-vmpm + vmpp);
            double dvp_dfz = ay * (-vpmm + vpmp) + fy * (-vppm + vppp);
            double hfxz = -dvm_dfz + dvp_dfz;
            // d2/dfy dfz : derivative of gfy w.r.t fz
            double dvmm_dfz = -vmmm + vmmp;
            double dvmp_dfz = -vmpm + vmpp;
            double dvpm_dfz = -vpmm + vpmp;
            double dvpp_dfz = -vppm + vppp;
            double hfyz = (-dvmm_dfz + dvmp_dfz) * ax + (-dvpm_dfz + dvpp_dfz) * fx;

            gx = gfx / g_spacing[0];
            gy = gfy / g_spacing[1];
            gz = gfz / g_spacing[2];
            H.xx = 0.0; H.yy = 0.0; H.zz = 0.0;
            H.xy = hfxy / (g_spacing[0] * g_spacing[1]);
            H.xz = hfxz / (g_spacing[0] * g_spacing[2]);
            H.yz = hfyz / (g_spacing[1] * g_spacing[2]);
        }

        // Chain rule: inv_power transform W = V^n, then runtime tanh cap.
        // Both act on the physical-unit scalar field and its derivatives.
        if (g_inv_power > 0.0)
            applyInvPowerChain(g_inv_power, interpolated, gx, gy, gz, H);
        if (effectiveCap > 0.0)
            applyRuntimeCapChain(effectiveCap, interpolated, gx, gy, gz, H);

        // Multiply by the effective per-atom scaling factor.
        H.xx *= effectiveScaling; H.yy *= effectiveScaling; H.zz *= effectiveScaling;
        H.xy *= effectiveScaling; H.xz *= effectiveScaling; H.yz *= effectiveScaling;

        int off = ia * 6;
        g_hessianBlocks[off + 0] = H.xx;
        g_hessianBlocks[off + 1] = H.yy;
        g_hessianBlocks[off + 2] = H.zz;
        g_hessianBlocks[off + 3] = H.xy;
        g_hessianBlocks[off + 4] = H.xz;
        g_hessianBlocks[off + 5] = H.yz;
    });
}

void ReferenceCalcGridForceKernel::computeHessian() {
    // Supported: trilinear (0), cubic B-spline (1), triquintic Hermite (3).
    // Tricubic Hermite (2) is unsupported (mirrors CUDA computeGridHessian).
    if (g_interpolationMethod == 2) {
        throw OpenMMException("GridForce: Hessian not supported for tricubic Hermite (method 2)");
    }
    if (g_interpolationMethod != 0 && g_interpolationMethod != 1 && g_interpolationMethod != 3) {
        throw OpenMMException("GridForce: unsupported interpolation method for Hessian");
    }
    if (g_lastContext == nullptr)
        throw OpenMMException("GridForce: execute() must run before computeHessian()");
    computeHessianForPositions(extractPositions(*g_lastContext));
}

vector<double> ReferenceCalcGridForceKernel::getHessianBlocks() {
    return g_hessianBlocks;
}

void ReferenceCalcGridForceKernel::computeThirdDerivatives() {
    // CUDA supports third derivatives only for quintic B-spline (method 4), which
    // the Reference platform does not implement. Match CUDA's contract: error for
    // unsupported methods.
    throw OpenMMException("Third derivative computation only supported for quintic B-spline (method 4) interpolation");
}

vector<double> ReferenceCalcGridForceKernel::getThirdDerivativeBlocks() {
    return g_thirdDerivBlocks;
}

namespace {

// Analytical eigenvalues of a 3x3 symmetric matrix (Cardano), sorted ascending.
// Mirrors gridHessianAnalysis.cu eigenvalues_3x3_symmetric.
void eigenvalues3x3(double dxx, double dyy, double dzz,
                    double dxy, double dxz, double dyz, double lambda[3]) {
    double trace = dxx + dyy + dzz;
    double q = trace / 3.0;
    double a00 = dxx - q, a11 = dyy - q, a22 = dzz - q;
    double p2 = (a00*a00 + a11*a11 + a22*a22 + 2.0*(dxy*dxy + dxz*dxz + dyz*dyz)) / 6.0;
    double p = std::sqrt(p2);
    if (p < 1e-10) {
        lambda[0] = lambda[1] = lambda[2] = q;
        return;
    }
    double inv_p = 1.0 / p;
    double b00 = a00 * inv_p, b11 = a11 * inv_p, b22 = a22 * inv_p;
    double b01 = dxy * inv_p, b02 = dxz * inv_p, b12 = dyz * inv_p;
    double detB = b00 * (b11*b22 - b12*b12)
                - b01 * (b01*b22 - b12*b02)
                + b02 * (b01*b12 - b11*b02);
    double r = detB * 0.5;
    r = std::min(1.0, std::max(-1.0, r));
    double phi = std::acos(r) / 3.0;
    const double twoPi3 = 2.0 * 3.14159265358979323846 / 3.0;
    double eig0 = 2.0 * p * std::cos(phi);
    double eig1 = 2.0 * p * std::cos(phi - twoPi3);
    double eig2 = 2.0 * p * std::cos(phi + twoPi3);
    lambda[0] = eig2 + q;  // smallest
    lambda[1] = eig1 + q;  // middle
    lambda[2] = eig0 + q;  // largest
}

// Eigenvector for an eigenvalue via row cross products (matches CUDA).
void eigenvectorFor(double dxx, double dyy, double dzz,
                    double dxy, double dxz, double dyz,
                    double lambda, double v[3]) {
    double a00 = dxx - lambda, a11 = dyy - lambda, a22 = dzz - lambda;
    double v0 = dxy * dyz - a11 * dxz;
    double v1 = dxz * dxy - a00 * dyz;
    double v2 = a00 * a11 - dxy * dxy;
    double norm = std::sqrt(v0*v0 + v1*v1 + v2*v2);
    if (norm < 1e-10) {
        v0 = dxy * a22 - dyz * dxz;
        v1 = dxz * dxz - a00 * a22;
        v2 = a00 * dyz - dxz * dxy;
        norm = std::sqrt(v0*v0 + v1*v1 + v2*v2);
    }
    if (norm < 1e-10) {
        v0 = a11 * a22 - dyz * dyz;
        v1 = dyz * dxz - dxy * a22;
        v2 = dxy * dyz - a11 * dxz;
        norm = std::sqrt(v0*v0 + v1*v1 + v2*v2);
    }
    if (norm < 1e-10) {
        v[0] = 1.0; v[1] = 0.0; v[2] = 0.0;
        return;
    }
    double inv = 1.0 / norm;
    v[0] = v0 * inv; v[1] = v1 * inv; v[2] = v2 * inv;
}

}  // namespace

void ReferenceCalcGridForceKernel::analyzeHessian(float temperature) {
    if (g_interpolationMethod != 1 && g_interpolationMethod != 3) {
        throw OpenMMException("Hessian analysis only supported for bspline (method 1) and triquintic (method 3) interpolation");
    }
    if (g_hessianBlocks.empty()) {
        // Compute on demand from the last context, matching the GPU contract where
        // computeHessian() must precede analyzeHessian().
        if (g_lastContext == nullptr)
            throw OpenMMException("Must call computeHessian() before analyzeHessian()");
        computeHessianForPositions(extractPositions(*g_lastContext));
    }

    int numAtoms = (int)g_hessianBlocks.size() / 6;
    const double kB = 0.008314462618;  // kJ/(mol·K)
    const double kT = kB * (double)temperature;
    const double M_PI_D = 3.14159265358979323846;

    g_eigenvalues.assign(3 * numAtoms, 0.0);
    g_eigenvectors.assign(9 * numAtoms, 0.0);
    g_meanCurvature.assign(numAtoms, 0.0);
    g_totalCurvature.assign(numAtoms, 0.0);
    g_gaussianCurvature.assign(numAtoms, 0.0);
    g_fracAnisotropy.assign(numAtoms, 0.0);
    g_entropy.assign(numAtoms, 0.0);
    g_minEigenvalue.assign(numAtoms, 0.0);
    g_numNegative.assign(numAtoms, 0);
    g_totalEntropy = 0.0;

    for (int i = 0; i < numAtoms; i++) {
        double dxx = g_hessianBlocks[6*i + 0];
        double dyy = g_hessianBlocks[6*i + 1];
        double dzz = g_hessianBlocks[6*i + 2];
        double dxy = g_hessianBlocks[6*i + 3];
        double dxz = g_hessianBlocks[6*i + 4];
        double dyz = g_hessianBlocks[6*i + 5];

        double lambda[3];
        eigenvalues3x3(dxx, dyy, dzz, dxy, dxz, dyz, lambda);
        g_eigenvalues[3*i + 0] = lambda[0];
        g_eigenvalues[3*i + 1] = lambda[1];
        g_eigenvalues[3*i + 2] = lambda[2];

        for (int e = 0; e < 3; e++) {
            double v[3];
            eigenvectorFor(dxx, dyy, dzz, dxy, dxz, dyz, lambda[e], v);
            g_eigenvectors[9*i + 3*e + 0] = v[0];
            g_eigenvectors[9*i + 3*e + 1] = v[1];
            g_eigenvectors[9*i + 3*e + 2] = v[2];
        }

        double mean = (lambda[0] + lambda[1] + lambda[2]) / 3.0;
        double total = lambda[0] + lambda[1] + lambda[2];
        double gauss = lambda[0] * lambda[1] * lambda[2];
        g_meanCurvature[i] = mean;
        g_totalCurvature[i] = total;
        g_gaussianCurvature[i] = gauss;
        g_minEigenvalue[i] = lambda[0];
        g_numNegative[i] = (lambda[0] < 0.0) + (lambda[1] < 0.0) + (lambda[2] < 0.0);

        double diff0 = lambda[0] - mean, diff1 = lambda[1] - mean, diff2 = lambda[2] - mean;
        double num = diff0*diff0 + diff1*diff1 + diff2*diff2;
        double den = lambda[0]*lambda[0] + lambda[1]*lambda[1] + lambda[2]*lambda[2];
        double fa = 0.0;
        if (den > 1e-20)
            fa = std::sqrt(num / (2.0 * den));
        g_fracAnisotropy[i] = fa;

        if (lambda[0] > 1e-10 && lambda[1] > 1e-10 && lambda[2] > 1e-10) {
            double two_pi_kT = 2.0 * M_PI_D * kT;
            double S = 0.0;
            S += 0.5 * (1.0 + std::log(two_pi_kT / lambda[0]));
            S += 0.5 * (1.0 + std::log(two_pi_kT / lambda[1]));
            S += 0.5 * (1.0 + std::log(two_pi_kT / lambda[2]));
            g_entropy[i] = S;
            g_totalEntropy += S;
        } else {
            g_entropy[i] = std::nan("");
        }
    }
}

vector<double> ReferenceCalcGridForceKernel::getEigenvalues() { return g_eigenvalues; }
vector<double> ReferenceCalcGridForceKernel::getEigenvectors() { return g_eigenvectors; }
vector<double> ReferenceCalcGridForceKernel::getMeanCurvature() { return g_meanCurvature; }
vector<double> ReferenceCalcGridForceKernel::getTotalCurvature() { return g_totalCurvature; }
vector<double> ReferenceCalcGridForceKernel::getGaussianCurvature() { return g_gaussianCurvature; }
vector<double> ReferenceCalcGridForceKernel::getFracAnisotropy() { return g_fracAnisotropy; }
vector<double> ReferenceCalcGridForceKernel::getEntropy() { return g_entropy; }
vector<double> ReferenceCalcGridForceKernel::getMinEigenvalue() { return g_minEigenvalue; }
vector<int> ReferenceCalcGridForceKernel::getNumNegative() { return g_numNegative; }
double ReferenceCalcGridForceKernel::getTotalEntropy() { return g_totalEntropy; }

// ============================================================================
// ReferenceCalcBondedHessianKernel implementation
// ============================================================================

void ReferenceCalcBondedHessianKernel::initialize(const System& system) {
    numAtoms = system.getNumParticles();

    // Extract HarmonicBondForce parameters
    for (int i = 0; i < system.getNumForces(); i++) {
        const HarmonicBondForce* bondForce = dynamic_cast<const HarmonicBondForce*>(&system.getForce(i));
        if (bondForce != nullptr) {
            numBonds = bondForce->getNumBonds();
            bondAtoms.resize(2 * numBonds);
            bondLengths.resize(numBonds);
            bondKs.resize(numBonds);

            for (int j = 0; j < numBonds; j++) {
                int atom1, atom2;
                double length, k;
                bondForce->getBondParameters(j, atom1, atom2, length, k);
                bondAtoms[2*j] = atom1;
                bondAtoms[2*j + 1] = atom2;
                bondLengths[j] = length;
                bondKs[j] = k;
            }
            break;
        }
    }

    // Extract HarmonicAngleForce parameters
    for (int i = 0; i < system.getNumForces(); i++) {
        const HarmonicAngleForce* angleForce = dynamic_cast<const HarmonicAngleForce*>(&system.getForce(i));
        if (angleForce != nullptr) {
            numAngles = angleForce->getNumAngles();
            angleAtoms.resize(3 * numAngles);
            angleValues.resize(numAngles);
            angleKs.resize(numAngles);

            for (int j = 0; j < numAngles; j++) {
                int atom1, atom2, atom3;
                double angle, k;
                angleForce->getAngleParameters(j, atom1, atom2, atom3, angle, k);
                angleAtoms[3*j] = atom1;
                angleAtoms[3*j + 1] = atom2;
                angleAtoms[3*j + 2] = atom3;
                angleValues[j] = angle;
                angleKs[j] = k;
            }
            break;
        }
    }

    // Extract PeriodicTorsionForce parameters
    for (int i = 0; i < system.getNumForces(); i++) {
        const PeriodicTorsionForce* torsionForce = dynamic_cast<const PeriodicTorsionForce*>(&system.getForce(i));
        if (torsionForce != nullptr) {
            numTorsions = torsionForce->getNumTorsions();
            torsionAtoms.resize(4 * numTorsions);
            torsionPeriodicities.resize(numTorsions);
            torsionPhases.resize(numTorsions);
            torsionKs.resize(numTorsions);

            for (int j = 0; j < numTorsions; j++) {
                int atom1, atom2, atom3, atom4, periodicity;
                double phase, k;
                torsionForce->getTorsionParameters(j, atom1, atom2, atom3, atom4, periodicity, phase, k);
                torsionAtoms[4*j] = atom1;
                torsionAtoms[4*j + 1] = atom2;
                torsionAtoms[4*j + 2] = atom3;
                torsionAtoms[4*j + 3] = atom4;
                torsionPeriodicities[j] = periodicity;
                torsionPhases[j] = phase;
                torsionKs[j] = k;
            }
            break;
        }
    }
}

// Helper: compute bond Hessian block
static void computeRefBondHessianBlock(const Vec3& ri, const Vec3& rj, double k, double r0,
                                        double Hii[9], double Hij[9]) {
    Vec3 rij = rj - ri;
    double r = sqrt(rij[0]*rij[0] + rij[1]*rij[1] + rij[2]*rij[2]);
    if (r < 1e-10) r = 1e-10;

    double invR = 1.0 / r;
    double invR2 = invR * invR;
    double factor1 = k * (1.0 - r0 * invR);
    double factor2 = k * r0 * invR * invR2;

    for (int a = 0; a < 3; a++) {
        for (int b = 0; b < 3; b++) {
            double delta_ab = (a == b) ? 1.0 : 0.0;
            Hii[a*3 + b] = factor1 * delta_ab + factor2 * rij[a] * rij[b];
            Hij[a*3 + b] = -Hii[a*3 + b];
        }
    }
}

// Helper: compute analytical angle Hessian
static void computeRefAngleHessian(const Vec3& r1, const Vec3& r2, const Vec3& r3,
                                    double k, double theta0, double H[9][9]) {
    for (int i = 0; i < 9; i++)
        for (int j = 0; j < 9; j++)
            H[i][j] = 0.0;

    Vec3 b1, b3;
    for (int d = 0; d < 3; d++) {
        b1[d] = r1[d] - r2[d];
        b3[d] = r3[d] - r2[d];
    }

    double L1 = sqrt(b1[0]*b1[0] + b1[1]*b1[1] + b1[2]*b1[2]);
    double L3 = sqrt(b3[0]*b3[0] + b3[1]*b3[1] + b3[2]*b3[2]);
    if (L1 < 1e-10 || L3 < 1e-10) return;

    double invL1 = 1.0 / L1, invL3 = 1.0 / L3;
    double invL1_sq = invL1 * invL1, invL3_sq = invL3 * invL3;

    double e1[3], e3[3];
    for (int d = 0; d < 3; d++) {
        e1[d] = b1[d] * invL1;
        e3[d] = b3[d] * invL3;
    }

    double cos_theta = e1[0]*e3[0] + e1[1]*e3[1] + e1[2]*e3[2];
    cos_theta = max(-0.9999999, min(0.9999999, cos_theta));
    double theta = acos(cos_theta);
    double sin_theta = sin(theta);
    if (fabs(sin_theta) < 1e-10) return;

    double inv_sin = 1.0 / sin_theta;
    double cot_theta = cos_theta * inv_sin;
    double dtheta = theta - theta0;
    double dE_dtheta = k * dtheta;
    double d2E_dtheta2 = k;

    double v1[3], v3[3];
    for (int d = 0; d < 3; d++) {
        v1[d] = e3[d] - cos_theta * e1[d];
        v3[d] = e1[d] - cos_theta * e3[d];
    }

    double g1[3], g3[3], g2[3];
    for (int d = 0; d < 3; d++) {
        g1[d] = -inv_sin * invL1 * v1[d];
        g3[d] = -inv_sin * invL3 * v3[d];
        g2[d] = -(g1[d] + g3[d]);
    }

    double grad[9];
    for (int d = 0; d < 3; d++) {
        grad[d] = g1[d];
        grad[3+d] = g2[d];
        grad[6+d] = g3[d];
    }

    double P1[9], P3[9];
    for (int a = 0; a < 3; a++) {
        for (int b = 0; b < 3; b++) {
            double delta = (a == b) ? 1.0 : 0.0;
            P1[a*3+b] = delta - e1[a] * e1[b];
            P3[a*3+b] = delta - e3[a] * e3[b];
        }
    }

    double H_theta[81];
    for (int i = 0; i < 81; i++) H_theta[i] = 0.0;

    // d²θ/dr1 dr1
    for (int a = 0; a < 3; a++) {
        for (int b = 0; b < 3; b++) {
            H_theta[a*9 + b] = inv_sin * invL1_sq * (2.0 * v1[a] * e1[b] + cot_theta * v1[a] * v1[b] + cos_theta * P1[a*3+b]);
        }
    }

    // d²θ/dr3 dr3
    for (int a = 0; a < 3; a++) {
        for (int b = 0; b < 3; b++) {
            H_theta[(6+a)*9 + (6+b)] = inv_sin * invL3_sq * (2.0 * v3[a] * e3[b] + cot_theta * v3[a] * v3[b] + cos_theta * P3[a*3+b]);
        }
    }

    // d²θ/dr1 dr3
    for (int a = 0; a < 3; a++) {
        for (int b = 0; b < 3; b++) {
            double val = -inv_sin * invL1 * invL3 * (P3[a*3+b] - v3[a] * e1[b] - cot_theta * v1[a] * v3[b]);
            H_theta[a*9 + (6+b)] = val;
            H_theta[(6+b)*9 + a] = val;
        }
    }

    // d²θ/dr1 dr2 via chain rule
    for (int a = 0; a < 3; a++) {
        for (int b = 0; b < 3; b++) {
            double val = -H_theta[a*9 + b] - H_theta[a*9 + (6+b)];
            H_theta[a*9 + (3+b)] = val;
            H_theta[(3+b)*9 + a] = val;
        }
    }

    // d²θ/dr3 dr2
    for (int a = 0; a < 3; a++) {
        for (int b = 0; b < 3; b++) {
            double val = -H_theta[(6+a)*9 + (6+b)] - H_theta[(6+a)*9 + b];
            H_theta[(6+a)*9 + (3+b)] = val;
            H_theta[(3+b)*9 + (6+a)] = val;
        }
    }

    // d²θ/dr2 dr2
    for (int a = 0; a < 3; a++) {
        for (int b = 0; b < 3; b++) {
            H_theta[(3+a)*9 + (3+b)] = -H_theta[a*9 + (3+b)] - H_theta[(6+a)*9 + (3+b)];
        }
    }

    // Full Hessian: d²E/dθ² * grad ⊗ grad + dE/dθ * H_theta
    for (int i = 0; i < 9; i++) {
        for (int j = 0; j < 9; j++) {
            H[i][j] = d2E_dtheta2 * grad[i] * grad[j] + dE_dtheta * H_theta[i * 9 + j];
        }
    }
}

// Helper: compute torsion Hessian using Blondel-Karplus
static void computeRefTorsionHessian(const Vec3& p1, const Vec3& p2, const Vec3& p3, const Vec3& p4,
                                      int n, double phi0, double k, double H[12][12]) {
    for (int i = 0; i < 12; i++)
        for (int j = 0; j < 12; j++)
            H[i][j] = 0.0;

    Vec3 b1 = {p2[0]-p1[0], p2[1]-p1[1], p2[2]-p1[2]};
    Vec3 b2 = {p3[0]-p2[0], p3[1]-p2[1], p3[2]-p2[2]};
    Vec3 b3 = {p4[0]-p3[0], p4[1]-p3[1], p4[2]-p3[2]};

    Vec3 m = {b1[1]*b2[2] - b1[2]*b2[1], b1[2]*b2[0] - b1[0]*b2[2], b1[0]*b2[1] - b1[1]*b2[0]};
    Vec3 nv = {b2[1]*b3[2] - b2[2]*b3[1], b2[2]*b3[0] - b2[0]*b3[2], b2[0]*b3[1] - b2[1]*b3[0]};

    double m_sq = m[0]*m[0] + m[1]*m[1] + m[2]*m[2];
    double n_sq = nv[0]*nv[0] + nv[1]*nv[1] + nv[2]*nv[2];
    double b2_sq = b2[0]*b2[0] + b2[1]*b2[1] + b2[2]*b2[2];
    if (m_sq < 1e-20 || n_sq < 1e-20 || b2_sq < 1e-20) return;

    double b2_norm = sqrt(b2_sq);
    double m_norm = sqrt(m_sq), n_norm = sqrt(n_sq);

    Vec3 m_hat = {m[0]/m_norm, m[1]/m_norm, m[2]/m_norm};
    Vec3 n_hat = {nv[0]/n_norm, nv[1]/n_norm, nv[2]/n_norm};
    Vec3 b2_hat = {b2[0]/b2_norm, b2[1]/b2_norm, b2[2]/b2_norm};

    double cos_phi = m_hat[0]*n_hat[0] + m_hat[1]*n_hat[1] + m_hat[2]*n_hat[2];
    Vec3 mcb2 = {m_hat[1]*b2_hat[2] - m_hat[2]*b2_hat[1],
                 m_hat[2]*b2_hat[0] - m_hat[0]*b2_hat[2],
                 m_hat[0]*b2_hat[1] - m_hat[1]*b2_hat[0]};
    double sin_phi = mcb2[0]*n_hat[0] + mcb2[1]*n_hat[1] + mcb2[2]*n_hat[2];
    double phi = atan2(sin_phi, cos_phi);

    double d2E_dphi2 = -k * n * n * cos(n * phi - phi0);

    Vec3 dphi_dr1 = {b2_norm / m_sq * m[0], b2_norm / m_sq * m[1], b2_norm / m_sq * m[2]};
    Vec3 dphi_dr4 = {-b2_norm / n_sq * nv[0], -b2_norm / n_sq * nv[1], -b2_norm / n_sq * nv[2]};

    double b1b2 = b1[0]*b2[0] + b1[1]*b2[1] + b1[2]*b2[2];
    double b3b2 = b3[0]*b2[0] + b3[1]*b2[1] + b3[2]*b2[2];
    double alpha = b1b2 / b2_sq, beta = b3b2 / b2_sq;
    double c1 = -(1.0 + alpha), c4 = beta, d1 = alpha, d4 = -(1.0 + beta);

    Vec3 dphi_dr2 = {c1 * dphi_dr1[0] + c4 * dphi_dr4[0],
                     c1 * dphi_dr1[1] + c4 * dphi_dr4[1],
                     c1 * dphi_dr1[2] + c4 * dphi_dr4[2]};
    Vec3 dphi_dr3 = {d1 * dphi_dr1[0] + d4 * dphi_dr4[0],
                     d1 * dphi_dr1[1] + d4 * dphi_dr4[1],
                     d1 * dphi_dr1[2] + d4 * dphi_dr4[2]};

    Vec3 dphi[4] = {dphi_dr1, dphi_dr2, dphi_dr3, dphi_dr4};

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            for (int di = 0; di < 3; di++) {
                for (int dj = 0; dj < 3; dj++) {
                    H[3*i + di][3*j + dj] = d2E_dphi2 * dphi[i][di] * dphi[j][dj];
                }
            }
        }
    }
}

// Helper: add a 3x3 block to the Hessian matrix
static void addBlockToHessian(vector<double>& H, int N3, int i, int j, const double block[9]) {
    for (int di = 0; di < 3; di++) {
        for (int dj = 0; dj < 3; dj++) {
            H[(3*i + di) * N3 + (3*j + dj)] += block[di * 3 + dj];
        }
    }
}

std::vector<double> ReferenceCalcBondedHessianKernel::computeHessian(ContextImpl& context) {
    int N3 = 3 * numAtoms;
    vector<double> H(N3 * N3, 0.0);

    vector<Vec3>& positions = *static_cast<ReferencePlatform::PlatformData*>(
        context.getPlatformData())->positions;

    // Compute bond Hessians
    for (int b = 0; b < numBonds; b++) {
        int i = bondAtoms[2*b];
        int j = bondAtoms[2*b + 1];
        double Hii[9], Hij[9];
        computeRefBondHessianBlock(positions[i], positions[j], bondKs[b], bondLengths[b], Hii, Hij);
        addBlockToHessian(H, N3, i, i, Hii);
        addBlockToHessian(H, N3, j, j, Hii);
        addBlockToHessian(H, N3, i, j, Hij);
        addBlockToHessian(H, N3, j, i, Hij);
    }

    // Compute angle Hessians
    for (int a = 0; a < numAngles; a++) {
        int i = angleAtoms[3*a];
        int j = angleAtoms[3*a + 1];
        int k_idx = angleAtoms[3*a + 2];
        double Ha[9][9];
        computeRefAngleHessian(positions[i], positions[j], positions[k_idx], angleKs[a], angleValues[a], Ha);
        int atoms[3] = {i, j, k_idx};
        for (int ai = 0; ai < 3; ai++) {
            for (int aj = 0; aj < 3; aj++) {
                for (int di = 0; di < 3; di++) {
                    for (int dj = 0; dj < 3; dj++) {
                        H[(3*atoms[ai] + di) * N3 + (3*atoms[aj] + dj)] += Ha[3*ai + di][3*aj + dj];
                    }
                }
            }
        }
    }

    // Compute torsion Hessians
    for (int t = 0; t < numTorsions; t++) {
        int i = torsionAtoms[4*t];
        int j = torsionAtoms[4*t + 1];
        int k_idx = torsionAtoms[4*t + 2];
        int l = torsionAtoms[4*t + 3];
        double Ht[12][12];
        computeRefTorsionHessian(positions[i], positions[j], positions[k_idx], positions[l],
                                  torsionPeriodicities[t], torsionPhases[t], torsionKs[t], Ht);
        int atoms[4] = {i, j, k_idx, l};
        for (int ai = 0; ai < 4; ai++) {
            for (int aj = 0; aj < 4; aj++) {
                for (int di = 0; di < 3; di++) {
                    for (int dj = 0; dj < 3; dj++) {
                        H[(3*atoms[ai] + di) * N3 + (3*atoms[aj] + dj)] += Ht[3*ai + di][3*aj + dj];
                    }
                }
            }
        }
    }

    // Symmetrize
    for (int i = 0; i < N3; i++) {
        for (int j = i + 1; j < N3; j++) {
            double avg = 0.5 * (H[i * N3 + j] + H[j * N3 + i]);
            H[i * N3 + j] = avg;
            H[j * N3 + i] = avg;
        }
    }

    return H;
}

}  // namespace GridForcePlugin
