/* -------------------------------------------------------------------------- *
 *                              OpenMMGridForce                               *
 * -------------------------------------------------------------------------- *
 * GBSAGridForce: Grid-based Generalized Born solvation for ligand-receptor.
 *
 * Computes GB solvation energy using:
 * - Grid-interpolated HCT integrals from receptor (with binned correction)
 * - Pairwise HCT within ligand (O(N²))
 * - OBC-II Born radius correction
 * - Still equation for GB energy
 * - Optional ACE surface area term
 *
 * Supports multiple isolated ligands via particle groups, enabling efficient
 * batched evaluation of docking poses or conformations.
 * -------------------------------------------------------------------------- */

#ifndef OPENMM_GBSAGRIDFORCE_H_
#define OPENMM_GBSAGRIDFORCE_H_

#include "internal/windowsExportGridForce.h"
#include "DesolvationGrid.h"
#include "openmm/Force.h"
#include <vector>
#include <string>
#include <memory>

namespace GridForcePlugin {

/**
 * GBSAGridForce computes Generalized Born solvation energy using a hybrid
 * grid/pairwise approach optimized for ligand-receptor systems.
 *
 * The receptor contribution to the HCT (Hawkins-Cramer-Truhlar) integral is
 * precomputed on a grid and interpolated at runtime. A binned correction
 * formula provides exact results for any ligand atom radius.
 *
 * Ligand-ligand HCT contributions are computed pairwise, which is efficient
 * for small molecules (~50 atoms).
 *
 * This approach provides O(N_ligand) scaling instead of O(N_ligand × N_receptor)
 * for the receptor contribution, enabling rapid evaluation of many ligand poses.
 */
class OPENMM_EXPORT_GRIDFORCE GBSAGridForce : public OpenMM::Force {
public:
    // OBC-II parameters (Onufriev-Bashford-Case)
    static constexpr double OBC_ALPHA = 1.0;
    static constexpr double OBC_BETA = 0.8;
    static constexpr double OBC_GAMMA = 4.85;
    static constexpr double DIELECTRIC_OFFSET = 0.009;  // nm

    // Solvent parameters
    static constexpr double DEFAULT_SOLUTE_DIELECTRIC = 1.0;
    static constexpr double DEFAULT_SOLVENT_DIELECTRIC = 78.3;  // matches OpenMM GBSAOBCForce
    static constexpr double DEFAULT_SA_SURFACE_TENSION = 2.25936;  // kJ/mol/nm² (matches OpenMM)

    /**
     * Create a GBSAGridForce.
     */
    GBSAGridForce();

    /**
     * Get the number of atoms in the ligand template.
     */
    int getNumAtoms() const { return numAtoms; }

    /**
     * Set the number of atoms in the ligand template.
     * Must be called before setting atom parameters.
     */
    void setNumAtoms(int n);

    /**
     * Set which particle indices in the System this force applies to.
     */
    void setParticles(const std::vector<int>& particles);

    /**
     * Get the particle indices this force applies to.
     */
    const std::vector<int>& getParticles() const { return particles; }

    // ========== Atom Parameters ==========

    /**
     * Set parameters for an atom in the ligand template.
     *
     * @param index       Atom index (0 to numAtoms-1)
     * @param charge      Partial charge (elementary charge units)
     * @param radius      Intrinsic radius (nm) - NOT offset radius
     * @param scaleFactor OBC scale factor (typically 0.8 for H, 0.72 for C, etc.)
     */
    void setAtomParameters(int index, double charge, double radius, double scaleFactor);

    /**
     * Get parameters for an atom.
     */
    void getAtomParameters(int index, double& charge, double& radius, double& scaleFactor) const;

    // ========== Desolvation Grid ==========

    /**
     * Set the desolvation grid for receptor HCT computation.
     */
    void setDesolvationGrid(std::shared_ptr<DesolvationGrid> grid);

    /**
     * Get the desolvation grid.
     */
    std::shared_ptr<DesolvationGrid> getDesolvationGrid() const { return desolvationGrid; }

    /**
     * Load desolvation grid from file.
     */
    void loadDesolvationGrid(const std::string& filename);

    // ========== Auto Grid Generation ==========

    /**
     * Enable/disable automatic grid generation.
     * When enabled, the grid will be generated on GPU at initialization time
     * using the receptor parameters set via setReceptorAtoms/setReceptorPositions.
     */
    void setAutoGenerateGrid(bool enable);

    /**
     * Get whether auto grid generation is enabled.
     */
    bool getAutoGenerateGrid() const { return autoGenerateGrid; }

    /**
     * Set receptor atom indices for grid generation.
     */
    void setReceptorAtoms(const std::vector<int>& atoms);

    /**
     * Get receptor atom indices.
     */
    const std::vector<int>& getReceptorAtoms() const { return receptorAtoms; }

    /**
     * Set receptor atom positions for grid generation.
     * Positions should be in nm, flattened as [x0,y0,z0,x1,y1,z1,...].
     */
    void setReceptorPositions(const std::vector<double>& positions);

    /**
     * Get receptor positions.
     */
    const std::vector<double>& getReceptorPositions() const { return receptorPositions; }

    /**
     * Set receptor atom radii for grid generation.
     */
    void setReceptorRadii(const std::vector<double>& radii);

    /**
     * Get receptor radii.
     */
    const std::vector<double>& getReceptorRadii() const { return receptorRadii_; }

    /**
     * Set receptor atom scale factors for grid generation.
     */
    void setReceptorScaleFactors(const std::vector<double>& scales);

    /**
     * Get receptor scale factors.
     */
    const std::vector<double>& getReceptorScaleFactors() const { return receptorScaleFactors; }

    /**
     * Set the grid origin for auto-generation.
     */
    void setGridOrigin(double x, double y, double z);

    /**
     * Get the grid origin.
     */
    void getGridOrigin(double& x, double& y, double& z) const;

    /**
     * Set grid dimensions (counts) for auto-generation.
     */
    void setGridCounts(int nx, int ny, int nz);

    /**
     * Get grid dimensions.
     */
    void getGridCounts(int& nx, int& ny, int& nz) const;

    /**
     * Set grid spacing for auto-generation (nm).
     */
    void setGridSpacing(double spacing);

    /**
     * Get grid spacing.
     */
    double getGridSpacing() const { return gridSpacing_; }

    /**
     * Set probe radius for grid generation (nm).
     */
    void setProbeRadius(double radius);

    /**
     * Get probe radius.
     */
    double getProbeRadius() const { return probeRadius_; }

    /**
     * Set R thresholds for correction bins (nm).
     */
    void setRThresholds(const std::vector<double>& thresholds);

    /**
     * Get R thresholds.
     */
    const std::vector<double>& getRThresholds() const { return rThresholds_; }

    /**
     * Set whether to compute derivatives during grid generation (for tricubic/triquintic).
     */
    void setComputeGridDerivatives(bool compute);

    /**
     * Get whether derivatives will be computed.
     */
    bool getComputeGridDerivatives() const { return computeGridDerivatives; }

    // ========== KDE Smoothing Parameters ==========

    /**
     * Set KDE threshold for smooth cutoff (nm).
     * Atoms with |r - S| < threshold contribute with weight ≈ 1.
     * Default: 0.02 nm
     */
    void setKDEThreshold(double threshold);

    /**
     * Get KDE threshold.
     */
    double getKDEThreshold() const { return kdeThreshold_; }

    /**
     * Set KDE bandwidth for sigmoid smoothing (nm).
     * Smaller values give sharper cutoffs; larger values are smoother.
     * As bandwidth → 0, converges to hard-cutoff binned approach.
     * Default: 0.04 nm
     */
    void setKDEBandwidth(double bandwidth);

    /**
     * Get KDE bandwidth.
     */
    double getKDEBandwidth() const { return kdeBandwidth_; }

    /**
     * Set KDE epsilon_B smoothing parameter (nm).
     * Used in B grid: 0.5/sqrt(r² + ε²) instead of 0.5/r for numerical stability.
     * Default: 0.03 nm
     */
    void setKDEEpsilonB(double epsilon);

    /**
     * Get KDE epsilon_B.
     */
    double getKDEEpsilonB() const { return kdeEpsilonB_; }

    // ========== Exclusions ==========

    /**
     * Add an exclusion between two atoms.
     * Excluded pairs do not contribute to ligand-ligand HCT or GB energy.
     */
    void addExclusion(int atom1, int atom2);

    /**
     * Get the number of exclusions.
     */
    int getNumExclusions() const { return static_cast<int>(exclusions.size()); }

    /**
     * Get an exclusion pair.
     */
    void getExclusionParticles(int index, int& atom1, int& atom2) const;

    // ========== Alchemical Scaling ==========

    /**
     * Get the global scaling factor applied to all energy and force contributions.
     * Default is 1.0.
     */
    double getGlobalScalingFactor() const { return globalScalingFactor; }

    /**
     * Set the global scaling factor applied to all energy and force contributions.
     * Total scale = globalScalingFactor * groupScalingFactor.
     */
    void setGlobalScalingFactor(double factor) { globalScalingFactor = factor; }

    /**
     * Get the per-group scaling factor for a particle group.
     */
    double getGroupScalingFactor(int groupIndex) const;

    /**
     * Set the per-group scaling factor for a particle group.
     */
    void setGroupScalingFactor(int groupIndex, double factor);

    // ========== Particle Groups (Multi-ligand) ==========

    /**
     * Add a particle group for multi-ligand support.
     * Each group represents one ligand instance with its own set of particles.
     *
     * @param name            Group name for identification
     * @param particleIndices Indices of particles in this group (in System)
     * @return Index of the newly added group
     */
    int addParticleGroup(const std::string& name, const std::vector<int>& particleIndices);

    /**
     * Get the number of particle groups.
     */
    int getNumParticleGroups() const { return static_cast<int>(particleGroups.size()); }

    /**
     * Get particle group information.
     */
    void getParticleGroup(int index, std::string& name, std::vector<int>& particleIndices) const;

    // ========== Solvent Parameters ==========

    /**
     * Get the dielectric constant of the solute.
     */
    double getSoluteDielectric() const { return soluteDielectric; }

    /**
     * Set the dielectric constant of the solute.
     */
    void setSoluteDielectric(double dielectric);

    /**
     * Get the dielectric constant of the solvent.
     */
    double getSolventDielectric() const { return solventDielectric; }

    /**
     * Set the dielectric constant of the solvent.
     */
    void setSolventDielectric(double dielectric);

    // ========== Surface Area Term ==========

    /**
     * Get whether the surface area term is included.
     */
    bool getIncludeSurfaceArea() const { return includeSurfaceArea; }

    /**
     * Set whether to include the surface area term.
     */
    void setIncludeSurfaceArea(bool include);

    /**
     * Get the surface tension for the SA term (kJ/mol/nm²).
     */
    double getSurfaceTension() const { return surfaceTension; }

    /**
     * Set the surface tension for the SA term.
     */
    void setSurfaceTension(double tension);

    // ========== Interpolation Method ==========

    /**
     * Get the interpolation method used for grid evaluation.
     * 0 = Trilinear (default, C0 continuity)
     * 1 = B-spline (C2 continuity, less accurate)
     * 2 = Tricubic (C1 continuity, requires derivatives in grid)
     * 3 = Triquintic Hermite (C2 continuity, requires derivatives in grid)
     */
    int getInterpolationMethod() const { return interpolationMethod; }

    /**
     * Set the interpolation method for grid evaluation.
     * Methods 2 (tricubic) and 3 (triquintic) require the desolvation grid
     * to contain precomputed analytical derivatives.
     */
    void setInterpolationMethod(int method);

    /**
     * Set the B-spline prefilter order for grid generation.
     * When set, the B-spline prefilter is applied to grid values at generation time.
     * @param order  B-spline degree: 0 (none), 3 (cubic), or 5 (quintic)
     */
    void setBSplinePrefilterOrder(int order);

    /**
     * Get the current B-spline prefilter order.
     * @return  B-spline degree (0, 3, or 5)
     */
    int getBSplinePrefilterOrder() const { return bsplinePrefilterOrder; }

    // ========== Energy Reporting ==========

    /**
     * Get the total GB energy for a specific particle group (ligand + receptor desolvation).
     * Only valid after calling getState() on the Context.
     */
    double getGroupEnergy(int groupIndex) const;

    /**
     * Get the ligand desolvation energy for a particle group (GB + optional SA).
     * Only valid after calling getState() on the Context.
     */
    double getGroupLigandDesolvationEnergy(int groupIndex) const;

    /**
     * Get the Born radii for atoms in a particle group.
     * Only valid after calling getState() on the Context.
     */
    std::vector<double> getGroupBornRadii(int groupIndex) const;

    // ========== Hessian ==========

    /**
     * Compute the Hessian (second derivatives) via numerical finite differences.
     * Must be called after getState(getForces=True) so GBSA internal state is valid.
     */
    void computeHessian(OpenMM::Context& context) const;

    /**
     * Get per-atom 3x3 diagonal Hessian blocks.
     * Returns [6 * N] array: dxx, dyy, dzz, dxy, dxz, dyz per atom.
     * Only valid after computeHessian().
     */
    std::vector<double> getHessianBlocks(OpenMM::Context& context) const;

    /**
     * Get the full 3N x 3N Hessian matrix (row-major).
     * Captures cross-atom coupling through Born radii.
     * Only valid after computeHessian().
     */
    std::vector<double> getFullHessian(OpenMM::Context& context) const;

    // ========== OpenMM Force Interface ==========

    bool usesPeriodicBoundaryConditions() const override { return false; }

protected:
    OpenMM::ForceImpl* createImpl() const override;

private:
    int numAtoms;
    std::vector<int> particles;

    // Atom parameters
    std::vector<double> charges;
    std::vector<double> radii;
    std::vector<double> scaleFactors;

    // Exclusions
    std::vector<std::pair<int, int>> exclusions;

    // Particle groups for multi-ligand
    struct ParticleGroupInfo {
        std::string name;
        std::vector<int> particleIndices;
    };
    std::vector<ParticleGroupInfo> particleGroups;

    // Alchemical scaling
    double globalScalingFactor;
    std::vector<double> groupScalingFactors;

    // Desolvation grid
    std::shared_ptr<DesolvationGrid> desolvationGrid;

    // Auto grid generation
    bool autoGenerateGrid;
    std::vector<int> receptorAtoms;
    std::vector<double> receptorPositions;
    std::vector<double> receptorRadii_;
    std::vector<double> receptorScaleFactors;
    double gridOrigin[3];
    int gridCounts_[3];
    double gridSpacing_;
    double probeRadius_;
    std::vector<double> rThresholds_;
    bool computeGridDerivatives;

    // KDE smoothing parameters
    double kdeThreshold_;
    double kdeBandwidth_;
    double kdeEpsilonB_;

    // Solvent parameters
    double soluteDielectric;
    double solventDielectric;

    // Surface area term
    bool includeSurfaceArea;
    double surfaceTension;

    // Interpolation method (0=trilinear, 1=bspline, 2=tricubic, 3=triquintic)
    int interpolationMethod;

    // B-spline prefilter order (0=none, 3=cubic, 5=quintic)
    int bsplinePrefilterOrder;

    // Cached group energies (populated by kernel)
    mutable std::vector<double> groupEnergies;           // Total energy
    mutable std::vector<double> groupLigandEnergies;     // Ligand desolvation
    mutable std::vector<std::vector<double>> groupBornRadii;

    friend class GBSAGridForceImpl;
};

} // namespace GridForcePlugin

#endif // OPENMM_GBSAGRIDFORCE_H_
