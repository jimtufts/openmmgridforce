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
#include "SolvationFieldGrid.h"
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

    /**
     * How the receptor-ligand GB cross term is evaluated in GRID mode.
     *
     * The cross term is the solvent screening of receptor-ligand
     * electrostatics. PAIRWISE mode computes it exactly, with receptor Born
     * radii re-solved for the pose. CROSS_EXACT is the same sum with the
     * receptor frozen at apo. CROSS_RADIUS_GRID reads the far field from a
     * grid sliced in the ligand Born radius and evaluates a near shell
     * pairwise, where it also re-solves the receptor radii -- so it is both
     * cheaper and closer to PAIRWISE than CROSS_EXACT.
     */
    enum CrossMode {
        CROSS_NONE = 0,        /**< Omit the cross term */
        CROSS_EXACT = 1,       /**< Sum over every receptor atom, O(N_lig * N_rec) */
        CROSS_RADIUS_GRID = 2  /**< Field lookup plus a near shell, O(N_lig * k) */
    };

    /**
     * How the receptor desolvation ("mirror") term is evaluated in GRID mode.
     *
     * This is the change in receptor GB energy caused by the ligand raising
     * nearby receptor Born radii -- what PAIRWISE reports as
     * getGroupReceptorDesolvation(). LINEAR_GRID linearizes the OBC-II
     * rescale about the apo receptor and collapses it into a field that is
     * read with one lookup per ligand atom.
     */
    enum MirrorMode {
        MIRROR_NONE = 0,        /**< Frozen receptor: the term is zero */
        MIRROR_LINEAR_GRID = 1  /**< Linear-response field plus a near shell */
    };

    /**
     * Storage precision for the analytical Hessian path.
     *
     * FLOAT matches GBSAGridForce and the production force kernels:
     * ~1e-5 relative noise floor, faster, runs at full speed on all CUDA
     * arches.
     *
     * DOUBLE eliminates float-summation noise (notably in J^T M J at
     * Mpro-scale pairwise sums). Runs at native speed on sm_60+ (Pascal
     * and newer: hardware atomicAdd(double*) available). On pre-sm_60
     * hardware (Maxwell, Kepler) the kernel falls back to a software
     * atomicCAS loop, which works but is ~50-100x slower than FLOAT —
     * use FLOAT on those arches unless precision is critical.
     */
    enum HessianPrecision {
        HESSIAN_FLOAT  = 0,
        HESSIAN_DOUBLE = 1
    };

    // OBC-II parameters
    static constexpr double OBC_ALPHA = 1.0;
    static constexpr double OBC_BETA = 0.8;
    static constexpr double OBC_GAMMA = 4.85;
    static constexpr double DIELECTRIC_OFFSET = 0.009;  // nm

    // Default solvent parameters
    static constexpr double DEFAULT_SOLUTE_DIELECTRIC = 1.0;
    static constexpr double DEFAULT_SOLVENT_DIELECTRIC = 78.3;  // matches OpenMM GBSAOBCForce
    static constexpr double DEFAULT_SA_SURFACE_TENSION = 2.25936;  // kJ/mol/nm^2 (matches OpenMM)

    // Cutoff
    static constexpr double NO_CUTOFF = -1.0;  // Special value meaning no cutoff

    // Receptor locality cutoff
    static constexpr double NO_LOCALITY_CUTOFF = -1.0;  // Special value: update all receptor atoms

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

    // ========== Receptor Locality Cutoff ==========

    /**
     * Get the locality cutoff for receptor desolvation in PAIRWISE mode.
     * Only receptor atoms within this distance of any ligand atom have their
     * Born radii updated with ligand screening. Distant atoms keep reference
     * Born radii (precomputed without ligand). This reduces the per-frame
     * receptor desolvation cost from O(N_rec^2) to O(|A| * N_rec).
     *
     * Returns NO_LOCALITY_CUTOFF (-1.0) if all receptor atoms are updated.
     */
    /**
     * @deprecated Now an alias for getCutoffDistance(). Retained for API
     * compatibility. See setReceptorLocalityCutoff() for details.
     */
    double getReceptorLocalityCutoff() const { return cutoffDistance; }

    /**
     * Set the locality cutoff for receptor desolvation (nm).
     * Set to NO_LOCALITY_CUTOFF (-1.0) to update all receptor atoms (default).
     * A value of ~2.0-2.5 nm typically gives < 4 kJ/mol error.
     *
     * Note: This only affects receptor desolvation (Step 7). Ligand Born radii
     * and the cross-term always use all receptor atoms.
     */
    /**
     * @deprecated Now a passthrough to setCutoffDistance(). Two separate
     * cutoff APIs were historically supported (per-pair distance vs
     * tile-skip optimization), but the internal implementation now uses a
     * single unified cutoff to match OpenMM's GBSAOBCForce convention.
     */
    void setReceptorLocalityCutoff(double distance) { setCutoffDistance(distance); }

    // ========== Receptor Mode ==========

    /**
     * Get the receptor mode (NONE, GRID, or PAIRWISE).
     */
    ReceptorMode getReceptorMode() const { return receptorMode; }

    /**
     * Set the receptor mode. Default is NONE.
     */
    void setReceptorMode(ReceptorMode mode) { receptorMode = mode; }

    // ========== Hessian Precision ==========

    /**
     * Get the storage precision used by computeHessian().
     * Default DOUBLE; see HessianPrecision enum for trade-offs.
     */
    HessianPrecision getHessianPrecision() const { return hessianPrecision; }

    /**
     * Set the storage precision used by computeHessian(). Mostly for speed
     * gains on platforms without hardware atomicAdd(double*, double) (pre-
     * sm_60 Maxwell/Kepler), where the software CAS fallback dominates the
     * Hessian wall time. For PAIRWISE this is storage-only: internal compute
     * stays in double, only the final dim3N*dim3N hessian buffer is float-
     * typed (hardware atomicAdd(float*, float)). For GRID it is full
     * float-compute. Default DOUBLE preserves the legacy semantics.
     */
    void setHessianPrecision(HessianPrecision precision) { hessianPrecision = precision; }

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

    // ========== Cross term and mirror term (GRID mode add-ons) ==========
    //
    // GRID mode on its own reproduces only the ligand side of the complex:
    // ligand Born radii descreened by the receptor, plus ligand self and
    // ligand-ligand GB. These two add-ons supply the remaining pose-dependent
    // blocks that PAIRWISE mode computes by direct summation, so a GRID-mode
    // force with both enabled targets the same energy as PAIRWISE at a cost
    // that does not scale with receptor size.
    //
    // Both require the receptor to be described via setNumReceptorAtoms /
    // setReceptorAtomParameters / setReceptorPositions, even in GRID mode.

    /**
     * Get the cross-term evaluation mode. Default is CROSS_NONE.
     */
    CrossMode getCrossMode() const { return crossMode; }

    /**
     * Set the cross-term evaluation mode.
     *
     * CROSS_RADIUS_GRID needs a cross field covering the ligand's accessible
     * region; one is generated from the receptor at the desolvation grid's
     * geometry if none is supplied.
     */
    void setCrossMode(CrossMode mode) { crossMode = mode; }

    /**
     * @deprecated Use setCrossMode(). true selects CROSS_EXACT.
     */
    bool getComputeCrossTermGrid() const { return crossMode != CROSS_NONE; }

    /**
     * @deprecated Use setCrossMode(). true selects CROSS_EXACT.
     */
    void setComputeCrossTermGrid(bool enable) {
        crossMode = enable ? CROSS_EXACT : CROSS_NONE;
    }

    /**
     * Get the mirror (receptor desolvation) mode. Default is MIRROR_NONE.
     */
    MirrorMode getMirrorMode() const { return mirrorMode; }

    /**
     * Set the mirror mode. MIRROR_LINEAR_GRID generates a mirror field from
     * the receptor at the desolvation grid's geometry if none is supplied.
     */
    void setMirrorMode(MirrorMode mode) { mirrorMode = mode; }

    /**
     * Get the near-shell cutoff (nm) used by CROSS_RADIUS_GRID. Receptor
     * atoms within this distance of a ligand atom are evaluated pairwise,
     * with their Born radii re-solved for the pose; the rest come from the
     * field at apo radii. It must reach past where the ligand perturbs the
     * receptor, not merely past the switch, or the mixed radii are
     * inconsistent and accuracy drops.
     */
    double getNearShellCutoff() const { return nearShellCutoff; }

    /**
     * Set the near-shell cutoff (nm). Must be at least the switch-off
     * radius. Default 0.6 nm.
     */
    void setNearShellCutoff(double distance);

    /**
     * Radii (nm) of the smootherstep switch that splits each field into a
     * gridded far part and a pairwise near part. The switch removes the
     * near-contact cusps that a lattice cannot represent, so the near shell
     * carries them exactly instead. Defaults are 0.15 and 0.35 nm.
     */
    void setFieldSwitchRadii(double switchOn, double switchOff);
    double getFieldSwitchOn() const { return fieldSwitchOn; }
    double getFieldSwitchOff() const { return fieldSwitchOff; }

    /**
     * Interpolation method used to read the cross and mirror fields:
     * TRILINEAR (0) or TRICUBIC_BSPLINE (1). Independent of the desolvation
     * grid's method because these fields carry sharper features; trilinear
     * on a 0.04 nm lattice leaves several kJ/mol of error in the cross term
     * and has a discontinuous gradient. Default is TRICUBIC_BSPLINE.
     *
     * A field built for one method cannot be read with the other: the
     * B-spline form stores prefiltered coefficients, not node values.
     */
    int getFieldInterpolationMethod() const { return fieldInterpolationMethod; }
    void setFieldInterpolationMethod(int method);

    /**
     * Whether the CROSS_RADIUS_GRID near shell re-solves the receptor Born
     * radii from the ligand-induced descreening, instead of holding them at
     * their apo values like the far field does.
     *
     * On (default) this is what removes the frozen-apo error the far field
     * cannot avoid. It is also the only feedback in the term that strengthens
     * as the ligand couples in, so it is the first thing to disable when
     * diagnosing behaviour that appears only at strong coupling.
     */
    bool getCrossPerturbReceptorRadii() const { return crossPerturbReceptorRadii; }
    void setCrossPerturbReceptorRadii(bool enable) { crossPerturbReceptorRadii = enable; }

    /**
     * Global scale applied to the mirror term, absorbing the systematic
     * under-count of the linear response on deeply buried receptor atoms.
     * Fit once per receptor against PAIRWISE over a pose set. Default 1.0.
     */
    double getMirrorScale() const { return mirrorScale; }
    void setMirrorScale(double scale) { mirrorScale = scale; }

    /**
     * Padding (nm) beyond the grid box within which receptor atoms are kept
     * for the runtime near lists. Must be at least the near-shell cutoff.
     * Default 1.0 nm.
     */
    double getPocketPadding() const { return pocketPadding; }
    void setPocketPadding(double padding);

    /**
     * Range (nm) over which receptor atoms contribute to the mirror field.
     * Truncating it biases the term low by a pose-dependent amount that a
     * global mirror scale only partly absorbs, so it is deliberately wider
     * than the near-shell cutoff; the cost is offline only. Default 1.6 nm.
     */
    double getMirrorFieldCutoff() const { return mirrorFieldCutoff; }
    void setMirrorFieldCutoff(double cutoff);

    /**
     * Cross-term far field, one slice per probe Born radius. Generated from
     * the receptor at initialization when CROSS_RADIUS_GRID is selected and
     * none has been set; retrieve it afterwards to save it and skip the
     * rebuild next time.
     */
    void setCrossField(std::shared_ptr<SolvationFieldGrid> field);
    std::shared_ptr<SolvationFieldGrid> getCrossField() const { return crossField; }
    void loadCrossField(const std::string& filename);

    /**
     * Probe Born radii the cross field is sliced at (nm, ascending). Left
     * empty by default, in which case they are log-spaced across the range
     * the ligand template can reach: from the smallest offset radius to the
     * largest OBC2 ceiling. Four to six slices saturate the achievable
     * accuracy; more do not help because the residual is the frozen apo
     * receptor radii beyond the near shell, not the radius axis.
     */
    void setCrossFieldRadii(const std::vector<double>& radii);
    const std::vector<double>& getCrossFieldRadii() const { return crossFieldRadii; }

    /** Slice count used when getCrossFieldRadii() is empty. Default 6. */
    int getNumCrossFieldSlices() const { return numCrossFieldSlices; }
    void setNumCrossFieldSlices(int n);

    /**
     * Mirror linear-response field, one slice per distinct ligand descreener
     * scaled radius. Generated at initialization when MIRROR_LINEAR_GRID is
     * selected and none has been set.
     */
    void setMirrorField(std::shared_ptr<SolvationFieldGrid> field);
    std::shared_ptr<SolvationFieldGrid> getMirrorField() const { return mirrorField; }
    void loadMirrorField(const std::string& filename);

    /**
     * Set precomputed apo receptor OBC2 Born radii (nm), length
     * numReceptorAtoms. Optional: they are computed from the receptor
     * parameters at initialization when not supplied.
     */
    void setReceptorBornRadiiBaseline(const std::vector<double>& radii);
    const std::vector<double>& getReceptorBornRadiiBaseline() const {
        return receptorBornRadiiBaseline;
    }

    /**
     * @deprecated The cross term no longer bins ligand atoms by radius; it
     * uses the per-atom Born radii the GRID path already computes. Retained
     * so existing setup code keeps working. The values are ignored.
     */
    void setCrossTermBinValues(const std::vector<double>& binValues);
    const std::vector<double>& getCrossTermBinValues() const {
        return crossTermBinValues;
    }

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

    // ========== Alchemical Scaling ==========

    /**
     * Get the global scaling factor applied to all energy/force contributions.
     * Multiplies all per-group scaling factors. Default is 1.0.
     */
    double getGlobalScalingFactor() const { return globalScalingFactor; }

    /**
     * Set the global scaling factor.
     * Total scale for a group = globalScalingFactor * groupScalingFactor[group].
     */
    void setGlobalScalingFactor(double factor) { globalScalingFactor = factor; }

    /**
     * Get the alchemical scaling factor for a specific particle group.
     */
    double getGroupScalingFactor(int groupIndex) const;

    /**
     * Set the alchemical scaling factor for a specific particle group.
     * Default is 1.0 for each group.
     *
     * @param groupIndex  Index of the particle group
     * @param factor      Scaling factor (0.0 = fully decoupled, 1.0 = fully coupled)
     */
    void setGroupScalingFactor(int groupIndex, double factor);

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
     * Get energies for all particle groups in a single call.
     * Only valid after calling getState() on the Context.
     *
     * @return vector of energies (kJ/mol), one per group
     */
    std::vector<double> getParticleGroupEnergies() const;

    /**
     * Get the ligand self-solvation GB energy for this group.
     *
     * Semantics: this is the Still self+pair GB energy over ligand atoms
     * computed with the Born radii that were actually used to produce
     * getGroupEnergy() -- i.e., WITH receptor descreening applied when
     * receptorMode is PAIRWISE or GRID, and without it in NONE mode.
     * Does not include the SA term (SA lives in getGroupEnergy() only).
     *
     * To recover the ligand-only GB energy (Born radii from ligand HCT
     * alone), subtract getGroupReceptorContribution():
     *   E_lig_only_GB = getGroupLigandSelfEnergy() - getGroupReceptorContribution()
     */
    double getGroupLigandSelfEnergy(int groupIndex) const;

    /**
     * Get the receptor descreening contribution to the ligand GB energy.
     * Equals (E_lig_with_receptor - E_lig_alone) for the GB Still energy
     * computed with the two respective Born radii. Populated by Reference;
     * the CUDA path currently returns 0.0 (TODO to populate).
     */
    double getGroupReceptorContribution(int groupIndex) const;

    /**
     * Get the receptor desolvation energy: the change in receptor GB energy
     * caused by the ligand screening it. Computed exactly in PAIRWISE mode,
     * and from the mirror field in GRID mode when MIRROR_LINEAR_GRID is set.
     * Zero otherwise.
     */
    double getGroupReceptorDesolvation(int groupIndex) const;

    /**
     * Get the cross-term energy (receptor-ligand GB pairs).
     * This is the solvent screening contribution to receptor-ligand electrostatics.
     * This is NOT the direct Coulomb - it's the implicit solvent correction.
     * Populated in PAIRWISE mode, and in GRID mode when getCrossMode() is not
     * CROSS_NONE.
     */
    double getGroupCrossTermEnergy(int groupIndex) const;

    /**
     * Get the Born radii for atoms in a particle group.
     * Requires setDownloadBornRadii(true) to have been called before the
     * last force evaluation (otherwise returns an empty vector).
     */
    std::vector<double> getGroupBornRadii(int groupIndex) const;

    /**
     * Enable (opt-in) downloading of Born radii from the device on every
     * force evaluation so that getGroupBornRadii() returns valid data.
     * Disabled by default because the GPU↔host sync is expensive in a
     * tight MD loop. Turn on only for diagnostic rescoring.
     */
    void setDownloadBornRadii(bool enabled) { downloadBornRadiiEnabled = enabled; }
    bool getDownloadBornRadii() const { return downloadBornRadiiEnabled; }

    /**
     * Get per-atom GB energies for a particle group.
     */
    std::vector<double> getGroupAtomEnergies(int groupIndex) const;

    /**
     * Get the ligand surface area energy for a particle group.
     * Computed from ligand Born radii (which include receptor descreening).
     * Uses ACE formula: SA_i = surfaceTension * 4π * (R_i + probe)² * (R_i / R_born_i)^6
     *
     * This is the ligand's contribution to the SA term. When a receptor is present,
     * the ligand Born radii are smaller (more buried) so the ligand SA is reduced.
     */
    double getGroupLigandSurfaceArea(int groupIndex) const;

    /**
     * Get per-atom ligand surface area energies.
     * Each element is the ACE contribution from that ligand atom.
     */
    std::vector<double> getGroupAtomSurfaceAreas(int groupIndex) const;

    /**
     * Get the receptor Born radii (PAIRWISE mode only).
     * These include the ligand's contribution to receptor HCT.
     */
    std::vector<double> getReceptorBornRadii(int groupIndex) const;

    /**
     * Get the receptor surface area energy change due to ligand (PAIRWISE mode only).
     * This is computed as: SA(receptor with ligand) - SA(receptor in vacuum)
     * where SA(vacuum) is estimated using intrinsic radii as Born radii.
     *
     * Note: This is an approximation. The "vacuum" reference uses R_born = R_intrinsic,
     * which is exact for isolated atoms but may differ slightly from the true vacuum
     * Born radii of the receptor in complex conformations.
     */
    double getGroupReceptorSurfaceAreaChange(int groupIndex) const;

    /**
     * Get the unscaled (no per-group alchemical scaling) total GBSA energies
     * for all particle groups. Only global scaling is applied.
     * This avoids the need for a separate force evaluation with scaling set to 1.0.
     *
     * @param context The Context to get energies from (must have been evaluated)
     * @return Vector of unscaled energies, one per particle group
     */
    std::vector<double> getParticleGroupUnscaledEnergies(OpenMM::Context& context) const;

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

    // Opt-in diagnostic: download Born radii from device per force eval.
    bool downloadBornRadiiEnabled;

    // Cutoff
    double cutoffDistance;

    // Receptor locality cutoff
    double receptorLocalityCutoff;

    // Receptor mode
    ReceptorMode receptorMode;
    HessianPrecision hessianPrecision = HESSIAN_DOUBLE;

    // Grid mode configuration
    std::shared_ptr<DesolvationGrid> desolvationGrid;
    int interpolationMethod;

    // Cross-term and mirror-term configuration (GRID mode add-ons)
    CrossMode crossMode;
    MirrorMode mirrorMode;
    double nearShellCutoff;
    double fieldSwitchOn;
    double fieldSwitchOff;
    double mirrorScale;
    bool crossPerturbReceptorRadii;
    double pocketPadding;
    double mirrorFieldCutoff;
    int fieldInterpolationMethod;
    std::shared_ptr<SolvationFieldGrid> crossField;
    std::shared_ptr<SolvationFieldGrid> mirrorField;
    std::vector<double> crossFieldRadii;
    int numCrossFieldSlices;
    std::vector<double> crossTermBinValues;        // deprecated, unused
    std::vector<double> receptorBornRadiiBaseline; // [numReceptorAtoms], nm

    // Pairwise mode configuration
    int numReceptorAtoms;
    std::vector<double> receptorCharges;
    std::vector<double> receptorRadii;
    std::vector<double> receptorScaleFactors;
    std::vector<double> receptorPositions;

    // Alchemical scaling
    double globalScalingFactor;
    std::vector<double> groupScalingFactors;

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
    mutable std::vector<std::vector<double>> groupReceptorBornRadii;  // PAIRWISE mode only

    friend class IsolatedGBSAForceImpl;
};

}  // namespace GridForcePlugin

#endif /*OPENMM_ISOLATEDGBSAFORCE_H_*/
