#ifndef OPENMM_ISOLATEDGBSAFORCE_H_
#define OPENMM_ISOLATEDGBSAFORCE_H_

/* -------------------------------------------------------------------------- *
 *                              OpenMMGridForce                               *
 * -------------------------------------------------------------------------- *
 * IsolatedGBSAForce: Pairwise GBSA solvation for isolated particle groups.
 *
 * Unlike GBSAGridForce which uses grid-interpolated receptor HCT, this force
 * computes Born radii entirely from pairwise interactions. Supports three
 * receptor modes:
 *   - NONE: Ligand-only (no receptor)
 *   - GRID: Receptor HCT from desolvation grid (fast, for ligand Born radii)
 *   - PAIRWISE: Full pairwise receptor-ligand HCT (accurate, includes receptor
 *               desolvation which cannot be precomputed as a grid)
 *
 * Each particle group is an isolated ligand - no inter-group interactions.
 * This enables efficient batched evaluation of multiple ligand poses.
 *
 * Important: Unlike NonbondedForce, GBSA has NO exclusions. All atom pairs
 * contribute to Born radii, including bonded pairs. This is physically correct.
 * -------------------------------------------------------------------------- */

#include <string>
#include <vector>
#include <memory>

#include "internal/windowsExportGridForce.h"
#include "DesolvationGrid.h"
#include "openmm/Context.h"
#include "openmm/Force.h"
#include "openmm/Vec3.h"

namespace GridForcePlugin {

class OPENMM_EXPORT_GRIDFORCE IsolatedGBSAForce : public OpenMM::Force {
public:
    /**
     * GB method for computing Born radii.
     */
    enum GBMethod {
        HCT = 0,     /**< Raw HCT descreening (no OBC correction) */
        OBC_II = 1   /**< OBC-II with tanh correction (production) */
    };

    /**
     * Mode for receptor contributions to ligand Born radii.
     */
    enum ReceptorMode {
        NONE = 0,     /**< Ligand-only (no receptor) */
        GRID = 1,     /**< Receptor HCT from desolvation grid */
        PAIRWISE = 2  /**< Full pairwise receptor-ligand HCT */
    };

    // OBC-II parameters
    static constexpr double OBC_ALPHA = 1.0;
    static constexpr double OBC_BETA = 0.8;
    static constexpr double OBC_GAMMA = 4.85;
    static constexpr double DIELECTRIC_OFFSET = 0.009;  // nm

    // Default solvent parameters
    static constexpr double DEFAULT_SOLUTE_DIELECTRIC = 1.0;
    static constexpr double DEFAULT_SOLVENT_DIELECTRIC = 78.5;
    static constexpr double DEFAULT_SA_SURFACE_TENSION = 0.0054;  // kJ/mol/nm^2

    // Cutoff
    static constexpr double NO_CUTOFF = -1.0;  // Special value meaning no cutoff

    /**
     * Create an IsolatedGBSAForce.
     */
    IsolatedGBSAForce();

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
     * Set GBSA parameters for an atom in the ligand template.
     *
     * @param index       Atom index (0 to numAtoms-1)
     * @param charge      Partial charge (elementary charge units)
     * @param radius      Intrinsic radius (nm) - NOT offset radius
     * @param scaleFactor HCT scale factor (typically 0.8 for H, 0.72 for C, etc.)
     */
    void setAtomParameters(int index, double charge, double radius, double scaleFactor);

    /**
     * Get GBSA parameters for an atom.
     */
    void getAtomParameters(int index, double& charge, double& radius, double& scaleFactor) const;

    // ========== GB Method ==========

    /**
     * Get the GB method (HCT or OBC_II).
     */
    GBMethod getGBMethod() const { return gbMethod; }

    /**
     * Set the GB method. Default is OBC_II.
     */
    void setGBMethod(GBMethod method) { gbMethod = method; }

    // ========== Solvent Parameters ==========

    double getSoluteDielectric() const { return soluteDielectric; }
    void setSoluteDielectric(double dielectric);

    double getSolventDielectric() const { return solventDielectric; }
    void setSolventDielectric(double dielectric);

    // ========== Surface Area Term ==========

    bool getIncludeSurfaceArea() const { return includeSurfaceArea; }
    void setIncludeSurfaceArea(bool include) { includeSurfaceArea = include; }

    double getSurfaceTension() const { return surfaceTension; }
    void setSurfaceTension(double tension) { surfaceTension = tension; }

    // ========== Cutoff ==========

    /**
     * Get the cutoff distance for pairwise HCT calculations.
     * Pairs beyond this distance don't contribute to Born radii.
     * Returns NO_CUTOFF (-1.0) if no cutoff is used.
     */
    double getCutoffDistance() const { return cutoffDistance; }

    /**
     * Set the cutoff distance for pairwise HCT calculations (nm).
     * Set to NO_CUTOFF (-1.0) to disable cutoff (compute all pairs).
     * Default is NO_CUTOFF.
     *
     * Note: A cutoff of ~1.5-2.0 nm is typically sufficient for GBSA
     * as the HCT integral decays rapidly with distance.
     */
    void setCutoffDistance(double distance) { cutoffDistance = distance; }

    // ========== Receptor Mode ==========

    /**
     * Get the receptor mode (NONE, GRID, or PAIRWISE).
     */
    ReceptorMode getReceptorMode() const { return receptorMode; }

    /**
     * Set the receptor mode. Default is NONE.
     */
    void setReceptorMode(ReceptorMode mode) { receptorMode = mode; }

    // ========== Receptor Configuration (GRID mode) ==========

    /**
     * Set the desolvation grid for GRID mode.
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

    /**
     * Get/set interpolation method for grid lookups.
     * 0 = trilinear, 2 = tricubic, 3 = triquintic
     */
    int getInterpolationMethod() const { return interpolationMethod; }
    void setInterpolationMethod(int method);

    // ========== Receptor Configuration (PAIRWISE mode) ==========

    /**
     * Set the number of receptor atoms for PAIRWISE mode.
     */
    void setNumReceptorAtoms(int n);

    /**
     * Get the number of receptor atoms.
     */
    int getNumReceptorAtoms() const { return numReceptorAtoms; }

    /**
     * Set receptor atom parameters for PAIRWISE mode.
     *
     * @param index       Receptor atom index (0 to numReceptorAtoms-1)
     * @param charge      Partial charge (elementary charge units)
     * @param radius      Intrinsic radius (nm)
     * @param scaleFactor HCT scale factor
     */
    void setReceptorAtomParameters(int index, double charge, double radius, double scaleFactor);

    /**
     * Get receptor atom parameters.
     */
    void getReceptorAtomParameters(int index, double& charge, double& radius, double& scaleFactor) const;

    /**
     * Set receptor positions for PAIRWISE mode.
     * Positions are in nm, flattened as [x0,y0,z0,x1,y1,z1,...].
     */
    void setReceptorPositions(const std::vector<double>& positions);

    /**
     * Get receptor positions.
     */
    const std::vector<double>& getReceptorPositions() const { return receptorPositions; }

    // ========== Particle Groups (Multi-ligand) ==========

    /**
     * Add a particle group. Each group is an isolated ligand instance.
     *
     * @param name    Group name for identification
     * @param indices Particle indices for this group (must match template size)
     * @return Index of the newly added group
     */
    int addParticleGroup(const std::string& name, const std::vector<int>& indices);

    /**
     * Get the number of particle groups.
     */
    int getNumParticleGroups() const { return static_cast<int>(particleGroups.size()); }

    /**
     * Get particle group information.
     */
    void getParticleGroup(int index, std::string& name, std::vector<int>& indices) const;

    // ========== Energy Reporting ==========

    /**
     * Get the total GB energy for a particle group.
     * Only valid after calling getState() on the Context.
     */
    double getGroupEnergy(int groupIndex) const;

    /**
     * Get the ligand self-solvation energy (ligand-ligand GB only).
     */
    double getGroupLigandSelfEnergy(int groupIndex) const;

    /**
     * Get the receptor contribution to ligand solvation.
     * (Change in ligand GB energy due to receptor screening)
     */
    double getGroupReceptorContribution(int groupIndex) const;

    /**
     * Get the receptor desolvation energy (PAIRWISE mode only).
     * (Change in receptor GB energy due to ligand screening)
     */
    double getGroupReceptorDesolvation(int groupIndex) const;

    /**
     * Get the cross-term energy (receptor-ligand GB pairs, PAIRWISE mode only).
     * This is the solvent screening contribution to receptor-ligand electrostatics.
     * This is NOT the direct Coulomb - it's the implicit solvent correction.
     */
    double getGroupCrossTermEnergy(int groupIndex) const;

    /**
     * Get the Born radii for atoms in a particle group.
     */
    std::vector<double> getGroupBornRadii(int groupIndex) const;

    /**
     * Get per-atom GB energies for a particle group.
     */
    std::vector<double> getGroupAtomEnergies(int groupIndex) const;

    // ========== Parameter Updates ==========

    /**
     * Update parameters in a Context after they have changed.
     */
    void updateParametersInContext(OpenMM::Context& context);

    // ========== Hessian ==========

    /**
     * Compute the Hessian (second derivatives) for the GBSA force.
     *
     * @param context The Context containing current positions
     * @return Flattened 3N x 3N Hessian matrix, row-major order
     */
    std::vector<double> computeHessian(OpenMM::Context& context);

    // ========== OpenMM Force Interface ==========

    bool usesPeriodicBoundaryConditions() const override { return false; }

protected:
    OpenMM::ForceImpl* createImpl() const override;

private:
    // Template configuration
    int numAtoms;
    std::vector<int> particles;
    std::vector<double> charges;
    std::vector<double> radii;
    std::vector<double> scaleFactors;

    // GB method
    GBMethod gbMethod;

    // Solvent parameters
    double soluteDielectric;
    double solventDielectric;

    // Surface area term
    bool includeSurfaceArea;
    double surfaceTension;

    // Cutoff
    double cutoffDistance;

    // Receptor mode
    ReceptorMode receptorMode;

    // Grid mode configuration
    std::shared_ptr<DesolvationGrid> desolvationGrid;
    int interpolationMethod;

    // Pairwise mode configuration
    int numReceptorAtoms;
    std::vector<double> receptorCharges;
    std::vector<double> receptorRadii;
    std::vector<double> receptorScaleFactors;
    std::vector<double> receptorPositions;

    // Particle groups
    struct ParticleGroupInfo {
        std::string name;
        std::vector<int> indices;
    };
    std::vector<ParticleGroupInfo> particleGroups;

    // Cached results (mutable for const getters)
    mutable std::vector<double> groupEnergies;
    mutable std::vector<double> groupLigandSelfEnergies;
    mutable std::vector<double> groupReceptorContributions;
    mutable std::vector<double> groupReceptorDesolvations;
    mutable std::vector<double> groupCrossTermEnergies;
    mutable std::vector<std::vector<double>> groupBornRadii;
    mutable std::vector<std::vector<double>> groupAtomEnergies;

    friend class IsolatedGBSAForceImpl;
};

}  // namespace GridForcePlugin

#endif /*OPENMM_ISOLATEDGBSAFORCE_H_*/
