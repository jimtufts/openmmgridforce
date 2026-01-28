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
    static constexpr double DEFAULT_SOLVENT_DIELECTRIC = 78.5;
    static constexpr double DEFAULT_SA_SURFACE_TENSION = 0.0054;  // kJ/mol/nm²

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

    // ========== Energy Reporting ==========

    /**
     * Get the GB energy for a specific particle group.
     * Only valid after calling getState() on the Context.
     */
    double getGroupEnergy(int groupIndex) const;

    /**
     * Get the Born radii for atoms in a particle group.
     * Only valid after calling getState() on the Context.
     */
    std::vector<double> getGroupBornRadii(int groupIndex) const;

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

    // Desolvation grid
    std::shared_ptr<DesolvationGrid> desolvationGrid;

    // Solvent parameters
    double soluteDielectric;
    double solventDielectric;

    // Surface area term
    bool includeSurfaceArea;
    double surfaceTension;

    // Interpolation method (0=trilinear, 1=bspline, 2=tricubic, 3=triquintic)
    int interpolationMethod;

    // Cached group energies (populated by kernel)
    mutable std::vector<double> groupEnergies;
    mutable std::vector<std::vector<double>> groupBornRadii;

    friend class GBSAGridForceImpl;
};

} // namespace GridForcePlugin

#endif // OPENMM_GBSAGRIDFORCE_H_
