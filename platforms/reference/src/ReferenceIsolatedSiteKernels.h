#ifndef OPENMM_REFERENCE_ISOLATEDSITEFORCE_KERNELS_H_
#define OPENMM_REFERENCE_ISOLATEDSITEFORCE_KERNELS_H_

/* -------------------------------------------------------------------------- *
 *                              OpenMMGridForce                               *
 * -------------------------------------------------------------------------- */

#include "IsolatedSiteForceKernels.h"
#include "openmm/Platform.h"
#include <vector>

namespace GridForcePlugin {

class ReferenceCalcIsolatedSiteForceKernel : public CalcIsolatedSiteForceKernel {
public:
    ReferenceCalcIsolatedSiteForceKernel(std::string name,
                                         const OpenMM::Platform& platform)
        : CalcIsolatedSiteForceKernel(name, platform),
          numAtoms(0), numParticleGroups(0),
          centerX(0), centerY(0), centerZ(0),
          maxRadius(0), forceConstant(0),
          globalScalingFactor(1.0) {}

    void initialize(const OpenMM::System& system,
                   const IsolatedSiteForce& force) override;
    double execute(OpenMM::ContextImpl& context,
                  bool includeForces, bool includeEnergy) override;
    void copyParametersToContext(OpenMM::ContextImpl& context,
                                const IsolatedSiteForce& force) override;
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

    // Site parameters
    double centerX, centerY, centerZ;
    double maxRadius;
    double forceConstant;

    // Scaling
    double globalScalingFactor;
    std::vector<double> groupScalingFactors;

    // Per-atom masses (template)
    std::vector<double> masses;
    double totalMass;

    // Particle groups: which system atoms map to template indices
    std::vector<std::vector<int>> groupParticleIndices;

    // Per-group energy results from last execute()
    mutable std::vector<double> groupEnergies;
};

}  // namespace GridForcePlugin

#endif /*OPENMM_REFERENCE_ISOLATEDSITEFORCE_KERNELS_H_*/
