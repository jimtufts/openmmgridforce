#ifndef OPENMM_GRIDFORCE_H_
#define OPENMM_GRIDFORCE_H_

/* -------------------------------------------------------------------------- *
 *                              OpenMMGridForce                               *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2008-2012 Stanford University and the Authors.      *
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

#include <string>
#include <vector>
#include <memory>

#include "internal/windowsExportGridForce.h"
#include "openmm/Context.h"
#include "openmm/Force.h"
#include "openmm/Vec3.h"

using namespace OpenMM;

namespace GridForcePlugin {

/**
 * Inverse power transformation mode.
 * Controls how and when the inv_power transformation is applied to grid values.
 */
enum class InvPowerMode {
    /**
     * No transformation applied.
     * Grid values are used as-is, and no power transformation occurs during evaluation.
     */
    NONE = 0,

    /**
     * Transform grid values at initialization/runtime.
     * Grid values are transformed G -> G^(1/n) once after loading, before evaluation.
     * The evaluation kernel then applies ^n to recover original values.
     * Only valid for grids WITHOUT analytical derivatives.
     */
    RUNTIME = 1,

    /**
     * Grid values already have transformation stored.
     * Grid values are already G^(1/n) (from generation or prior transformation).
     * The evaluation kernel applies ^n to recover original values.
     * Compatible with analytical derivatives.
     */
    STORED = 2
};

/**
 * This class implements the AlGDock Nonbond interaction.
 */

class OPENMM_EXPORT_GRIDFORCE GridForce : public OpenMM::Force {
   public:
    /**
     * Create a GridForce.
     * @param spacing       the grid space
     * @param vals          the value at each grid
     */
    GridForce();

    /**
     * Get the force field parameters for a Nonbond Energy term
     * 
     */
    void addGridCounts(int nx, int ny, int nz);
    void addGridSpacing(double dx, double dy, double dz);  // length unit is 'nm'

    void addGridValue(double val);

    /**
     * Set all grid values at once. This is primarily used internally by kernels
     * after auto-generating grids to copy the values back to the GridForce object.
     *
     * @param vals  vector of grid values (must match grid dimensions)
     */
    void setGridValues(const std::vector<double>& vals);

    /**
     * Get all grid values. Returns a copy of the grid values vector.
     * Note: Only works for grids without analytical derivatives (e.g., trilinear).
     *
     * @return vector of grid values
     */
    const std::vector<double>& getGridValues() const;

    void addScalingFactor(double val);
    void setScalingFactor(int index, double val);

    /**
     * Set all scaling factors at once.
     */
    void setScalingFactors(const std::vector<double>& vals);

    /**
     * Enable or disable automatic calculation of scaling factors from the System.
     * When enabled, scaling factors will be extracted from the NonbondedForce
     * in the System based on the scalingProperty setting. When disabled (default),
     * scaling factors must be added manually using addScalingFactor().
     *
     * @param enable  if true, auto-calculate scaling factors; if false, use manual values
     */
    void setAutoCalculateScalingFactors(bool enable);

    /**
     * Get whether automatic scaling factor calculation is enabled.
     *
     * @return  true if auto-calculation is enabled, false otherwise
     */
    bool getAutoCalculateScalingFactors() const;

    /**
     * Set the property to use for automatic scaling factor calculation.
     * This is only used when autoCalculateScalingFactors is enabled.
     *
     * Supported values:
     * - "charge": Use particle charges (for electrostatic grids)
     * - "ljr": Use sqrt(epsilon) * (2*sigma)^6 (for LJ repulsive grids)
     * - "lja": Use sqrt(epsilon) * (2*sigma)^3 (for LJ attractive grids)
     *
     * @param property  the scaling property to use
     */
    void setScalingProperty(const std::string& property);

    /**
     * Get the current scaling property setting.
     *
     * @return  the scaling property name
     */
    const std::string& getScalingProperty() const;

    /**
     * Set the inverse power transformation mode and exponent.
     * This controls how and when the inv_power transformation is applied.
     *
     * Modes:
     * - NONE: No transformation (inv_power must be 0)
     * - RUNTIME: Transform grid values G -> G^(1/n) at initialization, then apply ^n during evaluation
     *            Only works for grids WITHOUT analytical derivatives
     * - STORED: Grid values already store G^(1/n), apply ^n during evaluation only
     *           Compatible with analytical derivatives
     *
     * @param mode       Transformation mode (see InvPowerMode enum)
     * @param inv_power  Exponent to apply (must be > 0 if mode != NONE)
     * @throws OpenMMException if validation fails (e.g., conflicting mode after loadFromFile)
     */
    void setInvPowerMode(InvPowerMode mode, double inv_power);

    /**
     * Get the current inverse power transformation mode.
     * @return  the current mode
     */
    InvPowerMode getInvPowerMode() const;

    /**
     * Apply inverse power transformation to grid values (in-place).
     * Transforms grid values: G -> sign(G) * |G|^(1/inv_power)
     *
     * Requirements:
     * - Mode must be RUNTIME
     * - Grid must NOT have analytical derivatives (hasDerivatives() == false)
     * - Must be called after grid is loaded but before first evaluation
     *
     * After successful transformation, mode is automatically updated to STORED.
     *
     * @throws OpenMMException if requirements are not met
     */
    void applyInvPowerTransformation();

    /**
     * Set the inverse power parameter for grid value transformation.
     *
     * DEPRECATED: Use setInvPowerMode() instead for explicit control over transformation timing.
     * This method is kept for backward compatibility and maps to setInvPowerMode(STORED, inv_power).
     *
     * When inv_power > 0, the interpolated grid value is raised to this power
     * before being multiplied by the scaling factor. This reverses the grid
     * transformation where grid values were stored as G^(1/inv_power).
     *
     * For example, if grids were transformed as G^(1/4), set inv_power=4.0
     * to compute: energy = scaling_factor * (interpolated)^4
     *
     * @param inv_power  exponent to apply to interpolated values (default: 0.0, meaning no transformation)
     */
    void setInvPower(double inv_power);

    /**
     * Get the current inverse power parameter.
     * @return  the inv_power value
     */
    double getInvPower() const;

    /**
     * Set the capping threshold for grid value saturation.
     * During grid generation, grid values are capped using: value = U_MAX * tanh(value / U_MAX)
     * This prevents extreme values at grid points very close to atoms, which improves
     * interpolation accuracy but can limit gradients in tight binding pockets.
     *
     * @param uMax  capping threshold in kJ/mol (default: 41840.0, equivalent to 10000 kcal/mol)
     */
    void setGridCap(double uMax);

    /**
     * Get the current grid capping threshold.
     * @return  the capping threshold in kJ/mol
     */
    double getGridCap() const;

    /**
     * Set the force constant for the harmonic restraint applied to atoms outside the grid.
     * When an atom is outside the grid bounds, a harmonic restrain force is applied
     * with energy: E = 0.5 * k * distance^2, where distance is the distance from the
     * nearest grid boundary.
     *
     * Set to 0.0 to disable out-of-bounds restraints (useful when using multiple grids
     * at different origins where atoms may legitimately be outside some grids).
     *
     * @param k  force constant in kJ/(mol*nm^2) (default: 10000.0)
     */
    void setOutOfBoundsRestraint(double k);

    /**
     * Get the current out-of-bounds restraint force constant.
     * @return  the restraint force constant in kJ/(mol*nm^2)
     */
    double getOutOfBoundsRestraint() const;

    /**
     * Set the interpolation method for grid value lookup.
     *
     * Supported methods:
     * - 0: Trilinear interpolation (default) - uses 2x2x2 grid points, fastest
     * - 1: Cubic B-spline interpolation - uses 4x4x4 grid points, smoother gradients
     * - 2: Tricubic interpolation - uses 4x4x4 grid points with derivatives, highest accuracy
     * - 3: Quintic Hermite interpolation - uses 6x6x6 grid points, smoothest but slowest
     *
     * @param method  interpolation method code (default: 0 for trilinear)
     */
    void setInterpolationMethod(int method);

    /**
     * Get the current interpolation method.
     * @return  interpolation method code
     */
    int getInterpolationMethod() const;

    /**
     * Enable or disable automatic grid generation from the System.
     * When enabled, the grid will be generated from NonbondedForce parameters
     * and receptor positions during kernel initialization.
     *
     * @param enable  if true, auto-generate grid; if false, use manual values
     */
    void setAutoGenerateGrid(bool enable);

    /**
     * Get whether automatic grid generation is enabled.
     *
     * @return  true if auto-generation is enabled, false otherwise
     */
    bool getAutoGenerateGrid() const;

    /**
     * Set the type of grid to generate.
     * Only used when autoGenerateGrid is enabled.
     *
     * Supported values:
     * - "charge": Electrostatic potential grid (kJ/(mol·e))
     * - "ljr": Lennard-Jones repulsive grid (kJ/mol)^(1/2)
     * - "lja": Lennard-Jones attractive grid (kJ/mol)^(1/2)
     *
     * @param type  the grid type to generate
     */
    void setGridType(const std::string& type);

    /**
     * Get the current grid type setting.
     *
     * @return  the grid type name
     */
    const std::string& getGridType() const;

    /**
     * Set the grid origin (default: 0,0,0).
     * The grid extends from the origin in positive directions.
     *
     * @param x  x-coordinate of grid origin (nm)
     * @param y  y-coordinate of grid origin (nm)
     * @param z  z-coordinate of grid origin (nm)
     */
    void setGridOrigin(double x, double y, double z);

    /**
     * Get the grid origin.
     *
     * @param x  output x-coordinate of grid origin (nm)
     * @param y  output y-coordinate of grid origin (nm)
     * @param z  output z-coordinate of grid origin (nm)
     */
    void getGridOrigin(double& x, double& y, double& z) const;

    /**
     * Enable or disable derivative computation for triquintic interpolation.
     * When enabled, grid generation will compute and store all 27 derivatives
     * at each grid point, required for proper C² continuous triquintic interpolation.
     * This creates Version 2 grid files which are 27× larger.
     *
     * @param compute  if true, compute derivatives; if false, store only function values
     */
    void setComputeDerivatives(bool compute);

    /**
     * Get whether derivative computation is enabled.
     *
     * @return  true if derivatives will be computed, false otherwise
     */
    bool getComputeDerivatives() const;

    /**
     * Check if the grid has precomputed derivatives.
     *
     * @return  true if derivatives are stored, false if only function values
     */
    bool hasDerivatives() const;

    /**
     * Get the derivative values (if computed).
     * Data is stored as a 4D array [27, nx, ny, nz] in C order (row-major).
     *
     * @return  vector of derivative values, or empty if derivatives not computed
     */
    const std::vector<double>& getDerivatives() const;

    /**
     * Set the derivative values. Used internally by kernels after grid generation.
     *
     * @param derivs  vector of derivative values [27, nx, ny, nz]
     */
    void setDerivatives(const std::vector<double>& derivs);

    /**
     * Set which particles this GridForce applies to during energy evaluation.
     * If not set (empty vector), the force applies to all particles in the System.
     * If set, only the specified particles experience the grid potential.
     *
     * This enables per-ligand grid energy evaluation in multi-ligand systems.
     * Each GridForce instance can be assigned to a specific set of particles,
     * allowing independent energy queries via force groups.
     *
     * @param particles  vector of particle indices (empty = all particles)
     */
    void setParticles(const std::vector<int>& particles);

    /**
     * Get which particles this GridForce applies to.
     *
     * @return  vector of particle indices (empty = all particles)
     */
    const std::vector<int>& getParticles() const;

    /**
     * Clear grid data from host memory (values and derivatives).
     * Call this after Context creation to free host memory when grid is cached on GPU.
     * Note: After calling this, saveToFile() will not work.
     */
    void clearGridData();

    /**
     * Set which atoms to include in grid calculation (receptor atoms).
     * If not set, all atoms except ligand atoms will be included.
     *
     * @param atomIndices  vector of atom indices to include
     */
    void setReceptorAtoms(const std::vector<int>& atomIndices);

    /**
     * Get the receptor atom indices.
     *
     * @return  vector of receptor atom indices
     */
    const std::vector<int>& getReceptorAtoms() const;

    /**
     * Set which atoms to exclude from grid calculation (ligand atoms).
     * If receptorAtoms is not set, the grid will include all atoms except these.
     *
     * @param atomIndices  vector of atom indices to exclude
     */
    void setLigandAtoms(const std::vector<int>& atomIndices);

    /**
     * Get the ligand atom indices.
     *
     * @return  vector of ligand atom indices
     */
    const std::vector<int>& getLigandAtoms() const;

    /**
     * Set the positions of receptor atoms for grid generation.
     * These positions should be in nanometers (OpenMM default units).
     * This must be called before adding the force to a System if auto-generation is enabled.
     *
     * @param positions  vector of Vec3 positions (nm)
     */
    void setReceptorPositions(const std::vector<Vec3>& positions);

    /**
     * Set the positions of receptor atoms from flat coordinate arrays.
     * Convenience method for Python - takes x, y, z as separate arrays.
     *
     * @param x  vector of x coordinates (nm)
     * @param y  vector of y coordinates (nm)
     * @param z  vector of z coordinates (nm)
     */
    void setReceptorPositionsFromArrays(const std::vector<double>& x,
                                        const std::vector<double>& y,
                                        const std::vector<double>& z);

    /**
     * Get the receptor positions.
     *
     * @return  vector of receptor positions (nm)
     */
    const std::vector<Vec3>& getReceptorPositions() const;

    /**
     * Load grid from a binary file.
     *
     * @param filename  path to grid file
     */
    void loadFromFile(const std::string& filename);

    /**
     * Save grid to a binary file.
     *
     * @param filename  path to output file
     */
    void saveToFile(const std::string& filename) const;

    void getGridParameters(std::vector<int> &g_counts,
                           std::vector<double> &g_spacing,
                           std::vector<double> &g_vals,
                           std::vector<double> &g_scaling_factors) const;

    /**
     *
     */
    void updateParametersInContext(Context &context);

   protected:
    ForceImpl *createImpl() const;

   private:
    std::vector<int> m_counts;
    std::vector<double> m_spacing;  // the length unit is 'nm'
    std::shared_ptr<std::vector<double>> m_vals;        // Shared grid values for memory efficiency
    std::vector<double> m_scaling_factors;
    double m_inv_power;
    InvPowerMode m_invPowerMode;     // Transformation mode (NONE, RUNTIME, or STORED)
    double m_gridCap;  // Capping threshold for grid values (kJ/mol)
    double m_outOfBoundsRestraint;  // Force constant for out-of-bounds harmonic restraint (kJ/mol/nm^2)
    int m_interpolationMethod;  // 0=trilinear, 1=cubic B-spline, 2=tricubic, 3=quintic Hermite
    bool m_autoCalculateScalingFactors;
    std::string m_scalingProperty;

    // Auto-generation parameters
    bool m_autoGenerateGrid;
    std::string m_gridType;              // "charge", "ljr", "lja"
    std::vector<double> m_gridOrigin;    // 3 elements: x, y, z (nm)
    std::vector<int> m_receptorAtoms;    // Indices of atoms to include
    std::vector<int> m_ligandAtoms;      // Indices of atoms to exclude
    std::vector<Vec3> m_receptorPositions; // Positions for grid generation (nm)

    // Derivative storage for triquintic interpolation
    bool m_computeDerivatives;           // Whether to compute derivatives during grid generation
    std::shared_ptr<std::vector<double>> m_derivatives;  // Shared derivatives [27, nx, ny, nz]

    // Particle filtering for multi-ligand evaluation
    std::vector<int> m_particles;        // Particle indices this force applies to (empty = all particles)
};

}  // namespace GridForcePlugin

#endif /*OPENMM_GRIDFORCE_H_*/
