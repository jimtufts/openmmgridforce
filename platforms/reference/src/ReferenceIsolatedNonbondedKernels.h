#ifndef REFERENCE_ISOLATED_NONBONDED_KERNELS_H_
#define REFERENCE_ISOLATED_NONBONDED_KERNELS_H_

/* -------------------------------------------------------------------------- *
 *                              OpenMMGridForce                               *
 * -------------------------------------------------------------------------- *
 * Reference platform implementation of IsolatedNonbondedForce kernel.        *
 * Computes pairwise Coulomb + LJ interactions within isolated particle       *
 * groups on CPU.                                                             *
 * -------------------------------------------------------------------------- */

#include "IsolatedNonbondedForceKernels.h"
#include "openmm/Platform.h"
#include <vector>
#include <utility>

namespace GridForcePlugin {

class ReferenceCalcIsolatedNonbondedForceKernel : public CalcIsolatedNonbondedForceKernel {
public:
    ReferenceCalcIsolatedNonbondedForceKernel(std::string name, const OpenMM::Platform& platform)
        : CalcIsolatedNonbondedForceKernel(name, platform),
          numAtoms(0), numParticleGroups(0), globalScalingFactor(1.0) {}

    void initialize(const OpenMM::System& system, const IsolatedNonbondedForce& force) override;
    double execute(OpenMM::ContextImpl& context, bool includeForces, bool includeEnergy) override;
    void copyParametersToContext(OpenMM::ContextImpl& context, const IsolatedNonbondedForce& force) override;
    std::vector<double> computeHessian(OpenMM::ContextImpl& context, int groupIndex) override;
    double getGroupEnergy(int groupIndex) const override;

protected:
    // Compute one particle group's contribution (forces into the context array,
    // energy into groupEnergies[g]). Groups are disjoint atom sets, so distinct
    // groups never write the same force entry — safe to run concurrently.
    void computeGroup(int g, std::vector<OpenMM::Vec3>& posData,
                      std::vector<OpenMM::Vec3>& forceData,
                      bool includeForces, bool includeEnergy);

    // Run all groups. Serial here; the CPU platform overrides to parallelize.
    virtual void runGroups(OpenMM::ContextImpl& context,
                           std::vector<OpenMM::Vec3>& posData,
                           std::vector<OpenMM::Vec3>& forceData,
                           bool includeForces, bool includeEnergy);

    int numAtoms;
    int numParticleGroups;
    double globalScalingFactor;
    std::vector<double> groupScalingFactors;

    // Template atom parameters
    std::vector<double> charges;
    std::vector<double> sigmas;
    std::vector<double> epsilons;

    // Exclusions: pairs of template atom indices with no interaction
    std::vector<std::pair<int, int>> exclusions;

    // Exceptions: pairs with custom parameters (1-4 interactions)
    struct ExceptionInfo {
        int atom1, atom2;
        double chargeProd, sigma, epsilon;
    };
    std::vector<ExceptionInfo> exceptions;

    // Particle groups: groupParticleIndices[g][i] = system particle index
    std::vector<std::vector<int>> groupParticleIndices;

    // Per-group energy results from last execute()
    mutable std::vector<double> groupEnergies;

    // Helper: check if pair is excluded
    bool isExcluded(int i, int j) const;

    // Helper: find exception for pair, returns index or -1
    int findException(int i, int j) const;
};

}  // namespace GridForcePlugin

#endif /* REFERENCE_ISOLATED_NONBONDED_KERNELS_H_ */
