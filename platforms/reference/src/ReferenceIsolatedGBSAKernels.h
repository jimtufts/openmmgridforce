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
#include "openmm/Platform.h"
#include <vector>
#include <memory>

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
          interpolationMethod(0), numReceptorAtoms(0),
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
