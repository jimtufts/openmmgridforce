#ifndef REFERENCE_ISOLATED_BONDED_KERNELS_H_
#define REFERENCE_ISOLATED_BONDED_KERNELS_H_

/* -------------------------------------------------------------------------- *
 *                              OpenMMGridForce                               *
 * -------------------------------------------------------------------------- *
 * Reference platform implementation of IsolatedBondedForce kernel.           *
 * Computes harmonic bonds, harmonic angles, and periodic torsions within     *
 * isolated particle groups on CPU.                                           *
 * -------------------------------------------------------------------------- */

#include "IsolatedBondedForceKernels.h"
#include "openmm/Platform.h"
#include <vector>

namespace GridForcePlugin {

class ReferenceCalcIsolatedBondedForceKernel : public CalcIsolatedBondedForceKernel {
public:
    ReferenceCalcIsolatedBondedForceKernel(std::string name, const OpenMM::Platform& platform)
        : CalcIsolatedBondedForceKernel(name, platform),
          numAtoms(0), numParticleGroups(0), globalScalingFactor(1.0) {}

    void initialize(const OpenMM::System& system, const IsolatedBondedForce& force) override;
    double execute(OpenMM::ContextImpl& context, bool includeForces, bool includeEnergy) override;
    void copyParametersToContext(OpenMM::ContextImpl& context, const IsolatedBondedForce& force) override;
    double getGroupEnergy(int groupIndex) const override;
    std::vector<double> computeHessian(OpenMM::ContextImpl& context, int groupIndex) override;
    std::vector<double> computeInternalForceConstants(OpenMM::ContextImpl& context, int groupIndex) override;

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

    // Bond parameters
    struct BondInfo {
        int atom1, atom2;
        double length, k;
    };
    std::vector<BondInfo> bonds;

    // Angle parameters
    struct AngleInfo {
        int atom1, atom2, atom3;
        double angle, k;
    };
    std::vector<AngleInfo> angles;

    // Torsion parameters
    struct TorsionInfo {
        int atom1, atom2, atom3, atom4;
        int periodicity;
        double phase, k;
    };
    std::vector<TorsionInfo> torsions;

    // Particle groups: groupParticleIndices[g][i] = system particle index
    std::vector<std::vector<int>> groupParticleIndices;

    // Per-group energy results from last execute()
    mutable std::vector<double> groupEnergies;
};

}  // namespace GridForcePlugin

#endif /* REFERENCE_ISOLATED_BONDED_KERNELS_H_ */
