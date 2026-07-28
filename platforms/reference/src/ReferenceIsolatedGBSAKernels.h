#ifndef REFERENCE_ISOLATED_GBSA_KERNELS_H_
#define REFERENCE_ISOLATED_GBSA_KERNELS_H_

/* -------------------------------------------------------------------------- *
 *                              OpenMMGridForce                               *
 * -------------------------------------------------------------------------- *
 * Reference platform implementation of IsolatedGBSAForce kernel.            *
 * Computes GB implicit solvation with HCT/OBC-II Born radii for isolated    *
 * particle groups on CPU. Supports NONE, GRID, and PAIRWISE receptor modes. *
 * -------------------------------------------------------------------------- */

#include "IsolatedGBSAForceKernels.h"
#include "IsolatedGBSAForce.h"
#include "DesolvationGrid.h"
#include "SolvationFieldGrid.h"
#include "internal/SolvationFieldBuilder.h"
#include "openmm/Platform.h"
#include <vector>
#include <memory>
#include <functional>

namespace GridForcePlugin {

class ReferenceCalcIsolatedGBSAForceKernel : public CalcIsolatedGBSAForceKernel {
public:
    ReferenceCalcIsolatedGBSAForceKernel(std::string name, const OpenMM::Platform& platform)
        : CalcIsolatedGBSAForceKernel(name, platform),
          numAtoms(0), numParticleGroups(0),
          gbMethod(IsolatedGBSAForce::OBC_II),
          receptorMode(IsolatedGBSAForce::NONE),
          prefactor(0.0), includeSurfaceArea(false), surfaceTension(0.0),
          cutoffDistance(-1.0), globalScalingFactor(1.0),
          interpolationMethod(0),
          crossMode(IsolatedGBSAForce::CROSS_NONE),
          mirrorMode(IsolatedGBSAForce::MIRROR_NONE),
          nearShellCutoff(SolvationFields::DEFAULT_NEAR_CUTOFF),
          nearTaperOn(SolvationFields::DEFAULT_NEAR_CUTOFF
                      - SolvationFields::DEFAULT_NEAR_TAPER_WIDTH),
          fieldSwitchOn(SolvationFields::DEFAULT_SWITCH_ON),
          fieldSwitchOff(SolvationFields::DEFAULT_SWITCH_OFF),
          mirrorScale(1.0), crossPerturbReceptorRadii(true),
          mirrorFieldCutoff(SolvationFields::DEFAULT_MIRROR_BUILD_CUTOFF),
          fieldInterpolationMethod(1),
          numReceptorAtoms(0),
          receptorReferenceEnergy(0.0) {}

    void initialize(const OpenMM::System& system, const IsolatedGBSAForce& force) override;
    double execute(OpenMM::ContextImpl& context, bool includeForces, bool includeEnergy) override;
    void updateParametersInContext(OpenMM::ContextImpl& context, const IsolatedGBSAForce& force) override;
    std::vector<double> computeHessian(OpenMM::ContextImpl& context) override;

    // Per-group energy accessors
    double getGroupEnergy(int groupIndex) const override;
    double getGroupLigandSelfEnergy(int groupIndex) const override;
    double getGroupReceptorContribution(int groupIndex) const override;
    double getGroupReceptorDesolvation(int groupIndex) const override;
    double getGroupCrossTermEnergy(int groupIndex) const override;
    std::vector<double> getGroupBornRadii(int groupIndex) const override;
    std::vector<double> getGroupAtomEnergies(int groupIndex) const override;
    std::vector<double> getReceptorBornRadii(int groupIndex) const override;
    std::vector<double> getParticleGroupUnscaledEnergies() const override;

protected:
    // Compute one particle group's contribution (forces into the context array,
    // all per-group buffers indexed by g). Groups are disjoint atom sets, so
    // distinct groups never write the same force entry — safe to run concurrently.
    void computeGroup(int g, std::vector<OpenMM::Vec3>& posData,
                      std::vector<OpenMM::Vec3>& forceData,
                      bool includeForces, bool includeEnergy);

    // Run all groups. Serial here; the CPU platform overrides to parallelize.
    virtual void runGroups(OpenMM::ContextImpl& context,
                           std::vector<OpenMM::Vec3>& posData,
                           std::vector<OpenMM::Vec3>& forceData,
                           bool includeForces, bool includeEnergy);

    // Run an index range. Serial here; the CPU platform overrides to parallelize.
    virtual void parallelFor(OpenMM::ContextImpl& context, int count,
                             const std::function<void(int)>& body);

    // Configuration
    int numAtoms;
    int numParticleGroups;
    IsolatedGBSAForce::GBMethod gbMethod;
    IsolatedGBSAForce::ReceptorMode receptorMode;
    double prefactor;  // -COULOMB_CONSTANT * (1/solute - 1/solvent)
    bool includeSurfaceArea;
    double surfaceTension;
    double cutoffDistance;
    double receptorLocalityCutoff;
    double globalScalingFactor;
    std::vector<double> groupScalingFactors;

    // Template atom parameters
    std::vector<double> charges;
    std::vector<double> radii;
    std::vector<double> scaleFactors;

    // Particle groups: groupParticleIndices[g][i] = system particle index
    std::vector<std::vector<int>> groupParticleIndices;

    // GRID mode data
    int interpolationMethod;
    std::shared_ptr<DesolvationGrid> desolvationGrid;

    // GRID mode add-ons: receptor-ligand cross term and receptor desolvation
    IsolatedGBSAForce::CrossMode crossMode;
    IsolatedGBSAForce::MirrorMode mirrorMode;
    double nearShellCutoff;
    double nearTaperOn;
    double fieldSwitchOn;
    double fieldSwitchOff;
    double mirrorScale;
    bool crossPerturbReceptorRadii;
    double mirrorFieldCutoff;
    int fieldInterpolationMethod;
    std::shared_ptr<SolvationFieldGrid> crossField;
    std::shared_ptr<SolvationFieldGrid> mirrorField;
    std::vector<double> receptorApoHCT;         // [numReceptorAtoms]
    std::vector<double> receptorBornRadiiApo;   // [numReceptorAtoms]
    std::vector<double> receptorMirrorWeights;  // [numReceptorAtoms]
    std::vector<int> pocketAtoms;               // indices into the receptor arrays
    SolvationFields::PocketCellList pocketCells;
    std::vector<int> atomMirrorSlice;           // [numAtoms] mirror slice per atom

    /**
     * Load the receptor, derive its apo Born radii and mirror weights, build
     * the pocket cell list, and generate any field the selected modes need
     * and the force did not supply.
     */
    void initializeGridReceptorTerms(const IsolatedGBSAForce& force);

    /** True when GRID mode has at least one receptor add-on enabled. */
    bool usesGridReceptorTerms() const {
        return receptorMode == IsolatedGBSAForce::GRID &&
               (crossMode != IsolatedGBSAForce::CROSS_NONE ||
                mirrorMode != IsolatedGBSAForce::MIRROR_NONE);
    }

    /**
     * Receptor-ligand GB cross term in GRID mode.
     *
     * CROSS_EXACT sums over every receptor atom at apo radii.
     * CROSS_RADIUS_GRID reads the far field from the radius-sliced grid and
     * evaluates a near shell pairwise, re-solving the receptor Born radii
     * there from the ligand-induced HCT. That perturbation depends on ligand
     * positions, so it carries its own chain rule back onto the ligand,
     * applied here rather than through the caller's.
     *
     * Adds the energy to crossEnergy, the explicit position forces to
     * forceData, and the dE/dR_i coupling to dE_dR so the caller's
     * Born-radius chain rule picks that part up.
     */
    void addGridCrossTerm(int g, const std::vector<OpenMM::Vec3>& posData,
                          std::vector<OpenMM::Vec3>& forceData,
                          const std::vector<double>& bornRadii, double scale,
                          bool includeForces, double& crossEnergy,
                          std::vector<double>& dE_dR) const;

    /**
     * Receptor desolvation in GRID mode from the linear-response field. Adds
     * the energy to mirrorEnergy and its forces to forceData. Independent of
     * the ligand Born radii, so it feeds no chain rule.
     */
    void addGridMirrorTerm(int g, const std::vector<OpenMM::Vec3>& posData,
                           std::vector<OpenMM::Vec3>& forceData, double scale,
                           bool includeForces, double& mirrorEnergy) const;

    // PAIRWISE mode data
    int numReceptorAtoms;
    std::vector<double> receptorCharges;
    std::vector<double> receptorRadii;
    std::vector<double> receptorScaleFactors;
    std::vector<double> receptorPositions;  // [x0,y0,z0,x1,y1,z1,...]
    std::vector<double> receptorSelfHCT;    // receptor-receptor HCT per receptor atom
    std::vector<double> receptorBornRadiiRef; // receptor Born radii without ligand
    double receptorReferenceEnergy;         // receptor GB energy without ligand

    // Per-group results from last execute()
    mutable std::vector<double> groupEnergies_;
    mutable std::vector<double> groupLigandSelfEnergies_;
    mutable std::vector<double> groupReceptorContributions_;
    mutable std::vector<double> groupReceptorDesolvations_;
    mutable std::vector<double> groupCrossTermEnergies_;
    mutable std::vector<std::vector<double>> groupBornRadii_;
    mutable std::vector<std::vector<double>> groupAtomEnergies_;
    mutable std::vector<std::vector<double>> groupReceptorBornRadii_;

    // Internal helpers
    void computeBornRadii(const std::vector<double>& hctTotal, std::vector<double>& bornRadii) const;
    double computeGBEnergy(int groupIndex,
                           const std::vector<OpenMM::Vec3>& positions,
                           const std::vector<double>& bornRadii,
                           std::vector<double>& dE_dR) const;
    double computeSurfaceAreaEnergy(const std::vector<double>& bornRadii,
                                    std::vector<double>& dE_dR) const;

    // Adds the PAIRWISE-only receptor-desolvation and receptor-ligand
    // cross-term second-derivative contributions into the per-group ligand
    // Hessian block Hloc (local 3N x 3N, unscaled).
    void addPairwiseHessianContributions(
            int groupIndex,
            const std::vector<OpenMM::Vec3>& posData,
            const std::vector<double>& born,        // ligand full Born radii
            const std::vector<double>& hctReceptor,
            const std::vector<double>& hctLigand,
            const std::vector<double>& hctTotal,
            const std::vector<double>& dRdPsi,      // ligand dR/dPsi
            const std::vector<double>& J,           // ligand Jacobian [N x 3N]
            std::vector<double>& Hloc) const;
};

}  // namespace GridForcePlugin

#endif /* REFERENCE_ISOLATED_GBSA_KERNELS_H_ */
