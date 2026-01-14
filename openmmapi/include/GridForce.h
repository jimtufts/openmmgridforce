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

#include "GridForceTypes.h"
#include "GridData.h"
#include "CachedGridData.h"
#include "internal/windowsExportGridForce.h"
#include "openmm/Context.h"
#include "openmm/Force.h"
#include "openmm/Vec3.h"

using namespace OpenMM;

namespace GridForcePlugin {

/**
 * Represents a named group of particles for multi-ligand simulations.
 * Allows multiple ligands to share a single GridForce while maintaining
 * independent particle lists and per-group energy tracking.
 */
struct OPENMM_EXPORT_GRIDFORCE ParticleGroup {
    /**
     * Create a particle group.
     *
     * @param name              name of the group (e.g., "ligand1")
     * @param particleIndices   indices of particles in this group
     * @param scalingFactors    per-particle scaling factors (optional)
     */
    ParticleGroup(const std::string& name,
                  const std::vector<int>& particleIndices,
                  const std::vector<double>& scalingFactors = std::vector<double>())
        : name(name), particleIndices(particleIndices), scalingFactors(scalingFactors) {
        // If no scaling factors provided, default to 1.0 for all particles
        if (this->scalingFactors.empty()) {
            this->scalingFactors.resize(particleIndices.size(), 1.0);
        }
    }

    std::string name;                    // Group name for identification
    std::vector<int> particleIndices;    // Particle indices in this group
    std::vector<double> scalingFactors;  // Per-particle scaling factors
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
     * Construct a GridForce with shared GridData.
     * This constructor enables multiple GridForce instances to share the same grid data,
     * reducing memory usage for multi-ligand simulations.
     *
     * @param gridData  shared_ptr to GridData containing grid values and metadata
     */
    GridForce(std::shared_ptr<GridData> gridData);

    /**
     * Set the grid data for this force.
     * Allows explicit sharing of GridData across multiple GridForce instances.
     *
     * @param gridData  shared_ptr to GridData
     */
    void setGridData(std::shared_ptr<GridData> gridData);

    /**
     * Get the shared grid data.
     *
     * @return shared_ptr to GridData (may be null if not using GridData API)
     */
    std::shared_ptr<GridData> getGridData() const;

    /**
     * Get the cached grid data (used internally for GPU cache keys).
     *
     * @return shared_ptr to CachedGridData (may be null if not loaded from file)
     */
    std::shared_ptr<CachedGridData> getCachedGridData() const;

    /**
     * Set the cached grid data (used internally by kernels).
     *
     * @param cachedGridData shared_ptr to CachedGridData
     */
    void setCachedGridData(std::shared_ptr<CachedGridData> cachedGridData);

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
     * Enable tiled grid mode for memory-efficient large grids.
     * When enabled, the grid is divided into tiles that are streamed to
     * GPU memory on demand, allowing arbitrarily large grids with bounded
     * GPU memory usage.
     *
     * @param enable  if true, enable tiled mode
     * @param tileSize  size of each tile in grid points (default: 64)
     * @param memoryBudgetMB  GPU memory budget in MB (default: 2048)
     */
    void setTiledMode(bool enable, int tileSize = 64, int memoryBudgetMB = 2048);

    /**
     * Get whether tiled mode is enabled.
     * @return  true if tiled mode is enabled
     */
    bool getTiledMode() const;

    /**
     * Get the tile size (only valid when tiled mode is enabled).
     * @return  tile size in grid points
     */
    int getTileSize() const;

    /**
     * Get the GPU memory budget for tiled mode.
     * @return  memory budget in MB
     */
    int getMemoryBudgetMB() const;

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
     * Add a named particle group for multi-ligand simulations.
     * Each group has its own set of particles and scaling factors,
     * allowing multiple ligands to share a single GridForce instance.
     *
     * @param name              name for this group (e.g., "ligand1")
     * @param particleIndices   particle indices in this group
     * @param scalingFactors    per-particle scaling factors (optional, defaults to 1.0)
     * @return                  index of the added group
     */
    int addParticleGroup(const std::string& name,
                         const std::vector<int>& particleIndices,
                         const std::vector<double>& scalingFactors = std::vector<double>());

    /**
     * Get the number of particle groups.
     *
     * @return  number of particle groups
     */
    int getNumParticleGroups() const;

    /**
     * Get a particle group by index.
     *
     * @param index  index of the group
     * @return       const reference to the ParticleGroup
     */
    const ParticleGroup& getParticleGroup(int index) const;

    /**
     * Get a particle group by name.
     *
     * @param name  name of the group
     * @return      pointer to the ParticleGroup, or nullptr if not found
     */
    const ParticleGroup* getParticleGroupByName(const std::string& name) const;

    /**
     * Remove a particle group by index.
     *
     * @param index  index of the group to remove
     */
    void removeParticleGroup(int index);

    /**
     * Clear all particle groups.
     */
    void clearParticleGroups();

    /**
     * Get per-particle-group energies from the most recent evaluation.
     * Only available after evaluating a Context with particle groups.
     *
     * @param context  the Context to query
     * @return         vector of energies, one per particle group (empty if no groups)
     */
    std::vector<double> getParticleGroupEnergies(OpenMM::Context& context) const;

    /**
     * Get per-atom energies from the most recent evaluation.
     * Only available after evaluating a Context with particle groups.
     * Returns energies in the same order as particles were added to groups.
     *
     * @param context  the Context to query
     * @return         vector of per-atom energies (empty if no groups)
     */
    std::vector<double> getParticleAtomEnergies(OpenMM::Context& context) const;

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

    /**
     * Set output file for tiled grid generation.
     * When set and auto-generation is enabled, the grid will be generated
     * tile-by-tile directly to this file, avoiding the need to hold the
     * full grid in memory. This is useful for very large grids.
     *
     * @param filename  path to output tiled grid file
     * @param tileSize  size of tiles (default 32)
     */
    void setTiledOutputFile(const std::string& filename, int tileSize = 32);

    /**
     * Get the tiled output filename, or empty string if not set.
     */
    const std::string& getTiledOutputFile() const { return m_tiledOutputFile; }

    /**
     * Get the tile size for tiled output.
     */
    int getTiledOutputTileSize() const { return m_tiledOutputTileSize; }

    /**
     * Set input file for tiled grid evaluation.
     * When set, the grid will be loaded from this tiled file on demand
     * rather than from memory. This is useful for very large grids
     * that don't fit in GPU or host memory.
     *
     * When a tiled input file is specified:
     * - Tiles are loaded on-demand during force evaluation
     * - Only tiles containing particles are loaded to GPU
     * - LRU caching is used to manage GPU memory
     * - Tiled mode is automatically enabled
     *
     * @param filename  path to input tiled grid file (TiledGridData format)
     */
    void setTiledInputFile(const std::string& filename);

    /**
     * Get the tiled input filename, or empty string if not set.
     */
    const std::string& getTiledInputFile() const { return m_tiledInputFile; }

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

   public:
    /**
     * Internal: Set the System pointer for per-System cache scoping.
     * This is called by GridForceImpl during initialization.
     * Users should not call this directly.
     */
    void setSystemPointer(const void* systemPtr);

    /**
     * Internal: Get the System pointer.
     */
    const void* getSystemPointer() const;

   private:
    // Shared grid data container (when null, uses legacy storage below)
    std::shared_ptr<GridData> m_gridData;

    // Cached grid data with transformation state tracking
    std::shared_ptr<CachedGridData> m_cachedGridData;

    // System pointer for per-System cache scoping
    const void* m_systemPtr;

    // Grid storage (used when m_gridData is null for backward compatibility)
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

    // Named particle groups for multi-ligand workflows
    std::vector<ParticleGroup> m_particleGroups;  // Named groups of particles with individual scaling

    // Tiled mode parameters
    bool m_tiledMode;            // Whether to use tiled grid storage
    int m_tileSize;              // Tile size in grid points (default: 64)
    int m_memoryBudgetMB;        // GPU memory budget in MB (default: 2048)

    // Tiled file output (for generating directly to file)
    std::string m_tiledOutputFile;   // Output file for tiled generation
    int m_tiledOutputTileSize;       // Tile size for tiled output (default: 32)

    // Tiled file input (for loading tiles on demand during evaluation)
    std::string m_tiledInputFile;    // Input tiled grid file
};

}  // namespace GridForcePlugin

#endif /*OPENMM_GRIDFORCE_H_*/
