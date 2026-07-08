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
/**
 * Structure containing results from Hessian analysis.
 * Provides per-atom eigendecomposition, curvature metrics, and entropy estimates.
 */
struct OPENMM_EXPORT_GRIDFORCE HessianAnalysis {
    std::vector<double> eigenvalues;        // [3 * N] sorted ascending per atom
    std::vector<double> eigenvectors;       // [9 * N] (3 eigenvectors × 3 components per atom)
    std::vector<double> meanCurvature;      // [N] (λ1 + λ2 + λ3) / 3
    std::vector<double> totalCurvature;     // [N] λ1 + λ2 + λ3 (equals trace of Hessian)
    std::vector<double> gaussianCurvature;  // [N] λ1 * λ2 * λ3 (negative indicates saddle)
    std::vector<double> fracAnisotropy;     // [N] range [0,1] (0=isotropic, 1=linear)
    std::vector<double> entropy;            // [N] per-atom entropy in kB units, NaN for saddle points
    std::vector<double> minEigenvalue;      // [N] smallest eigenvalue per atom
    std::vector<int> numNegative;           // [N] count of negative eigenvalues (0-3)
    double totalEntropy;                    // Sum of per-atom entropies (excluding NaN)
};

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
        : name(name), particleIndices(particleIndices), scalingFactors(scalingFactors),
          groupScalingFactor(1.0), groupRuntimeCap(0.0) {}

    std::string name;                    // Group name for identification
    std::vector<int> particleIndices;    // Particle indices in this group
    std::vector<double> scalingFactors;  // Per-particle scaling factors
    double groupScalingFactor;           // Per-group alchemical scaling factor (default 1.0)
    double groupRuntimeCap;              // Per-group runtime cap (kJ/mol), 0 = use global cap
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
     * Set the global scaling factor that multiplies all per-particle scaling factors.
     * This is useful for alchemical free energy calculations where the entire grid
     * interaction needs to be scaled by a lambda/alpha parameter.
     *
     * The total scaling for each particle is: globalScalingFactor * scalingFactor[i]
     *
     * @param factor  the global scaling factor (default 1.0)
     */
    void setGlobalScalingFactor(double factor);

    /**
     * Get the global scaling factor.
     *
     * @return  the global scaling factor
     */
    double getGlobalScalingFactor() const;

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
     * Set the runtime capping threshold for interpolated grid values.
     * Applies tanh capping after interpolation (and after arcsinh decompression if active):
     *   value_capped = cap * tanh(value / cap)
     *   gradient_capped = gradient * sech^2(value / cap)
     *
     * This corrects B-spline interpolation overshoot for grids that are bounded by
     * construction (e.g. soft grids like sLJr/sELE where the potential IS a tanh function).
     * B-spline interpolation can overshoot beyond grid-node values; this re-enforces the
     * bound. For Hermite interpolation methods (tricubic/triquintic), this is unnecessary
     * since stored derivatives ensure the interpolant reproduces the capped function exactly.
     *
     * Typically set to the same value as setGridCap() for soft grids using B-spline.
     *
     * @param cap  capping threshold in kJ/mol (0 = disabled, default)
     */
    void setRuntimeCap(double cap);

    /**
     * Get the current runtime capping threshold.
     * @return  the runtime capping threshold in kJ/mol (0 = disabled)
     */
    double getRuntimeCap() const;

    /**
     * Enable V-space evaluation semantics for soft-LJ-style forces (sLJr).
     *
     * When true AND invPowerMode==STORED, the kernel:
     *   (1) back-transforms each corner v^(1/n) -> V (per-corner)
     *   (2) applies the runtime tanh cap in V-space
     *   (3) linearly interpolates the capped V values
     *   (4) SKIPS the post-interp back-transform
     *
     * This reproduces AlGDock reference's `trilinear_grid.c` semantics for
     * sLJr (tanh cap at a finite V value, linearly interpolated) while still
     * reading from a shared v^(1/n) STORED grid file (e.g. ljr.grid).
     *
     * Default: false (keeps current STORED behavior: cap in v^(1/n) space,
     * interp, then ^n back-transform).
     *
     * @param enabled  enable V-space per-corner evaluation path
     */
    void setEvaluateInVSpace(bool enabled);

    /**
     * Get the current V-space evaluation setting.
     */
    bool getEvaluateInVSpace() const;

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
     * Set effective evaluation bounds for this grid force.
     *
     * When set, atoms outside these bounds are treated as out-of-bounds even
     * if they are inside the actual grid data.  This is useful when multiple
     * grids have different extents (e.g., ELE at 6 nm vs LJr at 3 nm) and
     * you want all grids to apply OOB restraints at the smallest grid extent.
     *
     * Coordinates are in absolute (nm) space, the same frame as atom positions.
     * The effective bounds must be a subset of the actual grid bounds.
     *
     * Call clearEffectiveBounds() to revert to using the full grid extent.
     *
     * @param minX  lower X bound (nm)
     * @param minY  lower Y bound (nm)
     * @param minZ  lower Z bound (nm)
     * @param maxX  upper X bound (nm)
     * @param maxY  upper Y bound (nm)
     * @param maxZ  upper Z bound (nm)
     */
    void setEffectiveBounds(double minX, double minY, double minZ,
                            double maxX, double maxY, double maxZ);

    /**
     * Get the effective evaluation bounds.
     * If not set, returns the actual grid bounds (origin to origin + extent).
     */
    void getEffectiveBounds(double& minX, double& minY, double& minZ,
                            double& maxX, double& maxY, double& maxZ) const;

    /**
     * Check whether custom effective bounds have been set.
     */
    bool hasEffectiveBounds() const;

    /**
     * Clear effective bounds, reverting to the full grid extent.
     */
    void clearEffectiveBounds();

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
     * Set the arcsinh transformation scale for grid value compression.
     *
     * When set to a positive value, the arcsinh transform is applied to grid values
     * before B-spline prefiltering: g(x) = arcsinh(V(x) / scale). During evaluation,
     * the inverse sinh transform recovers the original values with chain rule for forces:
     *   V = scale * sinh(g_interp)
     *   dV/dr = scale * cosh(g_interp) * dg/dr
     *
     * This compresses extreme dynamic range (e.g., LJR grids near atoms) so that
     * B-spline prefiltering doesn't produce Gibbs-like ringing artifacts. Combined with
     * a high grid cap (or no cap), this achieves ~0.0001% interpolation accuracy.
     *
     * Only applicable to interpolation methods 0 (trilinear), 1 (tricubic B-spline),
     * and 4 (triquintic B-spline). Methods 2 and 3 (Hermite) are not supported.
     *
     * @param scale  arcsinh scale parameter (0.0 = disabled, > 0.0 = enabled)
     */
    void setArcsinhScale(double scale);

    /**
     * Get the current arcsinh transformation scale.
     * @return  the arcsinh scale (0.0 if disabled)
     */
    double getArcsinhScale() const;

    /**
     * Set Gaussian blur sigma for smoothing grid values before B-spline prefiltering.
     *
     * Applied after arcsinh transform (if any) and before the B-spline prefilter.
     * Smooths derivative discontinuities at cap boundaries to prevent Gibbs ringing
     * in the prefilter. Sigma is in grid-cell units (e.g. 1.5 = 1.5 grid spacings).
     *
     * @param sigma  blur sigma (0.0 = disabled, > 0.0 = enabled)
     */
    void setGaussianBlurSigma(double sigma);

    /**
     * Get the Gaussian blur sigma.
     * @return  the blur sigma (0.0 if disabled)
     */
    double getGaussianBlurSigma() const;

    /**
     * Set the B-spline prefilter order for grid generation.
     *
     * When set to a non-zero value, the B-spline prefilter is applied to grid
     * values at generation time, converting them from raw function values to
     * B-spline control points. This makes B-spline interpolation pass through
     * the original function values at grid nodes (interpolating rather than
     * approximating).
     *
     * Supported orders:
     * - 0: No prefilter (default) - grid values are used as-is
     * - 3: Cubic B-spline prefilter (tridiagonal solver) - use with interpolation method 1
     * - 5: Quintic B-spline prefilter (pentadiagonal solver) - use with quintic B-spline evaluation
     *
     * This setting only affects grid generation. The prefiltered control points
     * are stored in the grid file, so this is a one-time cost.
     *
     * @param order  B-spline degree (0, 3, or 5)
     */
    void setBSplinePrefilterOrder(int order);

    /**
     * Get the current B-spline prefilter order.
     * @return  B-spline degree (0, 3, or 5)
     */
    int getBSplinePrefilterOrder() const;

    /**
     * Set the adaptive regularization coefficient for B-spline prefiltering.
     *
     * When > 0, uses an adaptive regularized least-squares prefilter (PCG solver)
     * instead of the standard Thomas algorithm. This smooths Gibbs-like ringing
     * near singularities (e.g., receptor atoms in LJr grids) while preserving
     * exact interpolation in smooth regions.
     *
     * When == 0 (default), uses the standard Thomas algorithm (exact interpolation).
     *
     * Only applies to cubic B-spline prefilter (order 3).
     *
     * @param cReg  regularization coefficient (>= 0.0, default 0.0)
     */
    void setAdaptiveRegularization(double cReg);

    /**
     * Get the adaptive regularization coefficient.
     */
    double getAdaptiveRegularization() const;

    /**
     * Set the gradient threshold for adaptive regularization.
     *
     * Grid points with gradient magnitude below this threshold receive zero
     * regularization (preserving exact interpolation). Points above the
     * threshold receive regularization proportional to (|grad| - threshold).
     *
     * @param threshold  gradient magnitude threshold (>= 0.0, default 0.0)
     */
    void setRegularizationThreshold(double threshold);

    /**
     * Get the regularization gradient threshold.
     */
    double getRegularizationThreshold() const;

    /**
     * Set the PCG solver tolerance for adaptive prefiltering.
     * The solver stops when relative residual ||r|| / ||b|| < tolerance.
     *
     * @param tol  relative tolerance (default 1e-6)
     */
    void setPrefilterPCGTolerance(double tol);

    /**
     * Get the PCG solver tolerance.
     */
    double getPrefilterPCGTolerance() const;

    /**
     * Set the maximum PCG iterations for adaptive prefiltering.
     *
     * @param maxIter  maximum iterations (default 200)
     */
    void setPrefilterMaxIterations(int maxIter);

    /**
     * Get the maximum PCG iterations.
     */
    int getPrefilterMaxIterations() const;

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
     * Enable double-precision storage of the grid derivatives on the GPU.
     * Default false (float storage). When true, the 27 derivatives per point are
     * generated, uploaded, and read as double, removing the float-storage accuracy
     * limit of the triquintic interpolation. The kernel's compute precision still
     * follows the OpenMM context precision. Costs 2x derivative-grid VRAM.
     */
    void setUseDoubleStorage(bool useDouble);

    /** Get whether double-precision grid-derivative storage is enabled. */
    bool getUseDoubleStorage() const;

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
     * Set the alchemical scaling factor for a particle group.
     * This multiplies all per-particle scaling factors within the group,
     * enabling per-replica alchemical state control in multi-replica simulations.
     *
     * The total scaling for each particle is:
     *   globalScalingFactor * groupScalingFactor * scalingFactor[i]
     *
     * @param groupIndex  index of the particle group
     * @param factor      the group scaling factor (default 1.0)
     */
    void setParticleGroupScalingFactor(int groupIndex, double factor);

    /**
     * Get the alchemical scaling factor for a particle group.
     *
     * @param groupIndex  index of the particle group
     * @return            the group scaling factor
     */
    double getParticleGroupScalingFactor(int groupIndex) const;

    /**
     * Set the positions of particles in a specific group.
     * This modifies only the positions of the group's particles in the Context,
     * leaving all other particles unchanged. Essential for multi-replica simulations
     * where each group (replica) needs independent position management.
     *
     * @param context         the Context to modify
     * @param groupIndex      index of the particle group
     * @param positions       positions for the group's particles (must match group size)
     */
    void setParticleGroupPositions(OpenMM::Context& context, int groupIndex,
                                    const std::vector<OpenMM::Vec3>& positions) const;

    /**
     * Get the positions of particles in a specific group.
     *
     * @param context         the Context to query
     * @param groupIndex      index of the particle group
     * @return                positions of the group's particles
     */
    std::vector<OpenMM::Vec3> getParticleGroupPositions(OpenMM::Context& context, int groupIndex) const;

    /**
     * Swap positions between two particle groups.
     * This is used in replica exchange to swap configurations between replicas
     * without copying data through the host. Both groups must have the same
     * number of particles.
     *
     * @param context   the Context to modify
     * @param group1    index of the first particle group
     * @param group2    index of the second particle group
     */
    void swapParticleGroupPositions(OpenMM::Context& context, int group1, int group2) const;

    /**
     * Set the velocities of particles in a specific group.
     * Essential for HMC where each replica needs velocities drawn from
     * a Maxwell-Boltzmann distribution at its own temperature.
     *
     * @param context         the Context to modify
     * @param groupIndex      index of the particle group
     * @param velocities      velocities for the group's particles (must match group size)
     */
    void setParticleGroupVelocities(OpenMM::Context& context, int groupIndex,
                                     const std::vector<OpenMM::Vec3>& velocities) const;

    /**
     * Get the velocities of particles in a specific group.
     * Used to compute per-group kinetic energy for HMC accept/reject.
     *
     * @param context         the Context to query
     * @param groupIndex      index of the particle group
     * @return                velocities of the group's particles
     */
    std::vector<OpenMM::Vec3> getParticleGroupVelocities(OpenMM::Context& context, int groupIndex) const;

    /**
     * Set the positions of particles in a group from a flat coordinate array.
     * Format: [x0, y0, z0, x1, y1, z1, ...] with 3*N elements.
     *
     * @param context     the Context to modify
     * @param groupIndex  index of the particle group
     * @param coords      flat array of coordinates (nm), size = 3 * group size
     */
    void setParticleGroupPositionsFlat(OpenMM::Context& context, int groupIndex,
                                        const std::vector<double>& coords) const;

    /**
     * Get the positions of particles in a group as a flat coordinate array.
     * Returns: [x0, y0, z0, x1, y1, z1, ...] with 3*N elements.
     *
     * @param context     the Context to query
     * @param groupIndex  index of the particle group
     * @return            flat array of coordinates (nm), size = 3 * group size
     */
    std::vector<double> getParticleGroupPositionsFlat(OpenMM::Context& context, int groupIndex) const;

    /**
     * Set the velocities of particles in a group from a flat array.
     * Format: [vx0, vy0, vz0, vx1, vy1, vz1, ...] with 3*N elements.
     *
     * @param context     the Context to modify
     * @param groupIndex  index of the particle group
     * @param vels        flat array of velocities (nm/ps), size = 3 * group size
     */
    void setParticleGroupVelocitiesFlat(OpenMM::Context& context, int groupIndex,
                                         const std::vector<double>& vels) const;

    /**
     * Get the velocities of particles in a group as a flat array.
     * Returns: [vx0, vy0, vz0, vx1, vy1, vz1, ...] with 3*N elements.
     *
     * @param context     the Context to query
     * @param groupIndex  index of the particle group
     * @return            flat array of velocities (nm/ps), size = 3 * group size
     */
    std::vector<double> getParticleGroupVelocitiesFlat(OpenMM::Context& context, int groupIndex) const;

    /**
     * Get per-particle-group energies from the most recent evaluation.
     * Only available after evaluating a Context with particle groups.
     *
     * @param context  the Context to query
     * @return         vector of energies, one per particle group (empty if no groups)
     */
    std::vector<double> getParticleGroupEnergies(OpenMM::Context& context) const;

    /**
     * Get per-particle-group unscaled energies from the most recent evaluation.
     * Unscaled means globalScalingFactor * perParticleScale * interpolated
     * (no per-group alchemical scaling applied). This allows extracting
     * unscaled grid energies without setting all scaling to 1.0 and re-evaluating.
     *
     * @param context  the Context to query
     * @return         vector of unscaled energies, one per particle group (empty if no groups)
     */
    std::vector<double> getParticleGroupUnscaledEnergies(OpenMM::Context& context) const;

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
     * Get per-atom out-of-bounds flags from the most recent evaluation.
     * Only available after evaluating a Context with particle groups.
     * Returns flags in the same order as particles were added to groups.
     * Flag values: 0 = inside grid, 1 = outside grid.
     *
     * @param context  the Context to query
     * @return         vector of per-atom flags (empty if no groups)
     */
    std::vector<int> getParticleOutOfBoundsFlags(OpenMM::Context& context) const;

    // =========================================================================
    // Batch HMC operations (reduce Python→C++ round trips)
    // =========================================================================

    /**
     * Draw Maxwell-Boltzmann velocities for each particle group at its temperature
     * and set them in the Context. This performs a single getState + setVelocities
     * round trip regardless of the number of groups.
     *
     * @param context       the Context to modify
     * @param temperatures  per-group temperatures in Kelvin (length = numGroups)
     * @param masses        per-atom masses in amu for ONE group template (length = atoms_per_group)
     * @param seed          random seed (0 = use random device)
     */
    void drawAndSetGroupVelocities(OpenMM::Context& context,
                                    const std::vector<double>& temperatures,
                                    const std::vector<double>& masses,
                                    unsigned int seed = 0) const;

    /**
     * Compute per-group kinetic energy from current velocities.
     * Performs a single getState(Velocities) call.
     *
     * @param context  the Context to query
     * @param masses   per-atom masses in amu for ONE group template (length = atoms_per_group)
     * @return         vector of kinetic energies in kJ/mol, one per group
     */
    std::vector<double> computeGroupKineticEnergies(OpenMM::Context& context,
                                                     const std::vector<double>& masses) const;

    /**
     * Per-group Metropolis accept/reject. Restores rejected groups' positions
     * from the backup. Performs at most one getState + one setPositions.
     *
     * @param context          the Context to modify
     * @param positionsBackup  flat backup positions [K*N*3] in nm from before MD
     * @param pe_old           per-group potential energy before MD (kJ/mol)
     * @param pe_new           per-group potential energy after MD (kJ/mol)
     * @param ke_old           per-group kinetic energy before MD (kJ/mol)
     * @param ke_new           per-group kinetic energy after MD (kJ/mol)
     * @param temperatures     per-group temperatures in Kelvin
     * @param seed             random seed (0 = use random device)
     * @return                 vector of accept flags (1=accepted, 0=rejected)
     */
    std::vector<int> acceptRejectGroups(OpenMM::Context& context,
                                         const std::vector<double>& positionsBackup,
                                         const std::vector<double>& pe_old,
                                         const std::vector<double>& pe_new,
                                         const std::vector<double>& ke_old,
                                         const std::vector<double>& ke_new,
                                         const std::vector<double>& temperatures,
                                         unsigned int seed = 0) const;

    /**
     * Batch set all particle group scaling factors at once.
     * This is a convenience method that avoids K individual
     * setParticleGroupScalingFactor calls.
     *
     * @param factors  per-group scaling factors (length = numGroups)
     */
    void setAllParticleGroupScalingFactors(const std::vector<double>& factors);

    /**
     * Set the runtime cap for a particle group.
     * When > 0, overrides the global runtimeCap for this group.
     * The cap is applied as: E = cap * tanh(raw / cap).
     *
     * @param groupIndex  index of the particle group
     * @param cap         the runtime cap in kJ/mol (0 = use global)
     */
    void setParticleGroupRuntimeCap(int groupIndex, double cap);

    /**
     * Get the runtime cap for a particle group.
     *
     * @param groupIndex  index of the particle group
     * @return            the per-group runtime cap (0 = using global)
     */
    double getParticleGroupRuntimeCap(int groupIndex) const;

    /**
     * Batch set all particle group runtime caps at once.
     *
     * @param caps  per-group runtime caps (length = numGroups), 0 = use global
     */
    void setAllParticleGroupRuntimeCaps(const std::vector<double>& caps);

    /**
     * Get all particle group runtime caps.
     *
     * @return  per-group runtime caps (length = numGroups)
     */
    std::vector<double> getAllParticleGroupRuntimeCaps() const;

    /**
     * Get per-atom raw (pre-cap) energies from the most recent evaluation.
     * These are the interpolated grid values multiplied by globalScale * perParticleScale,
     * BEFORE the tanh runtime cap is applied. Useful for recomputing capped energies
     * at different cap values (e.g., for u_kln in alchemical free energy calculations).
     *
     * @param context  the Context to query
     * @return         vector of raw energies, one per atom across all groups (K*N values)
     */
    std::vector<double> getParticleGroupAtomRawEnergies(OpenMM::Context& context) const;

    /**
     * Compute Hessian (second derivative) blocks for each atom from the grid potential.
     * This computes the 3x3 Hessian block for each atom, storing 6 unique components
     * per atom: [d²V/dx², d²V/dy², d²V/dz², d²V/dxdy, d²V/dxdz, d²V/dydz].
     *
     * Must be called after execute() (e.g., after getState() with forces).
     * Only supported for bspline (method 1) and triquintic (method 3) interpolation.
     *
     * @param context    the Context for which to compute the Hessian
     */
    void computeHessian(OpenMM::Context& context) const;

    /**
     * Get the Hessian blocks computed by the most recent call to computeHessian().
     *
     * @param context    the Context from which to retrieve Hessian data
     * @return           vector of 6 components per atom: [dxx, dyy, dzz, dxy, dxz, dyz]
     *                   Total size is 6 * numAtoms. Units are kJ/(mol·nm²).
     */
    std::vector<double> getHessianBlocks(OpenMM::Context& context) const;

    /**
     * Compute third derivative blocks for each atom from grid potential.
     *
     * Stores 10 unique third-order partial derivative components per atom.
     * Only supported for quintic B-spline (method 4) interpolation (C4 continuity).
     *
     * @param context    the Context for which to compute third derivatives
     */
    void computeThirdDerivatives(OpenMM::Context& context) const;

    /**
     * Get the third derivative blocks computed by computeThirdDerivatives().
     *
     * @param context    the Context from which to retrieve data
     * @return           vector of 10 components per atom:
     *                   [d3xxx, d3yyy, d3zzz, d3xxy, d3xxz, d3xyy, d3xzz, d3yyz, d3yzz, d3xyz]
     *                   Total size is 10 * numAtoms. Units are kJ/(mol*nm^3).
     */
    std::vector<double> getThirdDerivativeBlocks(OpenMM::Context& context) const;

    /**
     * Analyze Hessian to compute per-atom eigenvalues, curvature metrics, and entropy.
     *
     * This performs eigendecomposition of each 3x3 Hessian block using Cardano's
     * analytical method, then computes derived metrics useful for binding site analysis
     * and normal modes approximations.
     *
     * Must be called after execute() (e.g., after getState() with forces).
     * Only supported for bspline (method 1) and triquintic (method 3) interpolation.
     *
     * @param context      the Context for analysis
     * @param temperature  temperature in Kelvin for entropy calculation (default: 300.0)
     * @return             HessianAnalysis structure with all computed metrics
     */
    HessianAnalysis analyzeHessian(OpenMM::Context& context, float temperature = 300.0f) const;

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
     * Check if grid values already have arcsinh+prefilter applied.
     * This is true for grids loaded from file (which were saved after transforms).
     *
     * @return  true if values are pre-transformed
     */
    bool getValuesPreTransformed() const { return m_valuesPreTransformed; }

    /**
     * Override the pre-transformed flag. Use after loadFromFile() when the
     * caller knows the loaded file contains raw (un-prefiltered) values, so
     * that a subsequent setBSplinePrefilterOrder() / setArcsinhScale() /
     * setGaussianBlurSigma() actually applies at runtime instead of being
     * silently skipped.
     *
     * @param flag  true if values are already transformed, false if raw
     */
    void setValuesPreTransformed(bool flag) { m_valuesPreTransformed = flag; }

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
    double m_globalScalingFactor;  // Multiplies all per-particle scaling factors (default 1.0)
    double m_inv_power;
    InvPowerMode m_invPowerMode;     // Transformation mode (NONE, RUNTIME, or STORED)
    double m_gridCap;  // Capping threshold for grid values (kJ/mol)
    double m_runtimeCap;  // Runtime capping threshold (kJ/mol), 0 = disabled
    bool m_evaluateInVSpace;  // Per-corner back-transform + V-space cap (for STORED grids)
    double m_outOfBoundsRestraint;  // Force constant for out-of-bounds harmonic restraint (kJ/mol/nm^2)
    bool m_hasEffectiveBounds;     // Whether custom effective bounds are set
    std::vector<double> m_effectiveBoundsMin;  // 3 elements: min x,y,z in absolute coordinates (nm)
    std::vector<double> m_effectiveBoundsMax;  // 3 elements: max x,y,z in absolute coordinates (nm)
    int m_interpolationMethod;  // 0=trilinear, 1=cubic B-spline, 2=tricubic, 3=quintic Hermite
    int m_bsplinePrefilterOrder;  // 0=none, 3=cubic, 5=quintic
    double m_arcsinhScale;  // 0.0=disabled, >0.0=arcsinh(V/scale) transform
    double m_blurSigma;    // 0.0=disabled, >0.0=Gaussian blur sigma in grid cells
    double m_adaptiveRegularization;   // 0.0=disabled, >0.0=adaptive reg strength
    double m_regularizationThreshold;  // gradient threshold for adaptive reg
    double m_prefilterPCGTolerance;    // PCG solver tolerance
    int m_prefilterMaxIterations;      // PCG max iterations
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
    bool m_useDoubleStorage;             // Store grid derivatives as double on the GPU
    std::shared_ptr<std::vector<double>> m_derivatives;  // Shared derivatives [27, nx, ny, nz]

    // Particle filtering for multi-ligand evaluation
    std::vector<int> m_particles;        // Particle indices this force applies to (empty = all particles)

    // Named particle groups for multi-ligand workflows
    std::vector<ParticleGroup> m_particleGroups;  // Named groups of particles with individual scaling

    // Tiled mode parameters
    bool m_valuesPreTransformed;  // Whether grid values already have arcsinh+prefilter applied (e.g., loaded from file)
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
