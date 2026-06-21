#ifndef OPENMM_REFERENCE_MULTIGROUPHMC_KERNELS_H_
#define OPENMM_REFERENCE_MULTIGROUPHMC_KERNELS_H_

/* -------------------------------------------------------------------------- *
 *                              OpenMMGridForce                               *
 * -------------------------------------------------------------------------- *
 * Reference (CPU) implementation of MultiGroupHMCIntegrator kernel.         *
 * -------------------------------------------------------------------------- */

#include "MultiGroupHMCKernels.h"
#include "openmm/Platform.h"
#include "openmm/Vec3.h"
#include <vector>
#include <random>
#include <functional>

namespace GridForcePlugin {

class ReferenceIntegrateMultiGroupHMCStepKernel : public IntegrateMultiGroupHMCStepKernel {
public:
    ReferenceIntegrateMultiGroupHMCStepKernel(std::string name,
                                               const OpenMM::Platform& platform)
        : IntegrateMultiGroupHMCStepKernel(name, platform),
          numGroups(0), atomsPerGroup(0), numParticles(0) {}

    void initialize(const OpenMM::System& system,
                   const MultiGroupHMCIntegrator& integrator) override;

    void execute(OpenMM::ContextImpl& context,
                const MultiGroupHMCIntegrator& integrator,
                bool forcesAreValid) override;

    double computeKineticEnergy(OpenMM::ContextImpl& context,
                                const MultiGroupHMCIntegrator& integrator) override;

    std::vector<int> getAcceptedFlags() const override { return lastAccepted; }
    std::vector<double> getDeltaH() const override { return lastDeltaH; }
    std::vector<int> getAcceptCounts() const override { return acceptCounts; }
    std::vector<int> getTrialCounts() const override { return trialCounts; }
    std::vector<int> getStabilityRejectCounts() const override { return stabilityRejectCounts; }
    void resetCounters() override;

    // Rigid-body Monte Carlo pre-step (random rotation about COM + translation).
    void executeMC(OpenMM::ContextImpl& context,
                   const MultiGroupHMCIntegrator& integrator) override;
    int getMCAttempted() const override { return mcAttemptedTotal; }
    int getMCAccepted() const override { return mcAcceptedTotal; }
    std::vector<int> getLastMCAccepted() const override { return lastMCAcceptedPerGroup; }
    void resetMCCounters() override;

    // Riemannian metric (Reference platform — identity only, non-identity throws)
    std::vector<double> getGroupMetricConditionNumbers() const override {
        return std::vector<double>(numGroups, 1.0);
    }

private:
    int numGroups;
    int atomsPerGroup;
    int numParticles;

    // Per-particle masses (from system)
    std::vector<double> masses;

    // Position backup for reject
    std::vector<OpenMM::Vec3> positionsBackup;

    // Last step results
    std::vector<int> lastAccepted;
    std::vector<double> lastDeltaH;

    // Cumulative statistics
    std::vector<int> acceptCounts;
    std::vector<int> trialCounts;
    std::vector<int> stabilityRejectCounts;

    // Rigid-body MC statistics
    int mcAttemptedTotal = 0;
    int mcAcceptedTotal = 0;
    std::vector<int> lastMCAcceptedPerGroup;

    // RNG
    std::mt19937 rng;
    std::normal_distribution<double> normalDist;
    std::uniform_real_distribution<double> uniformDist;

    // Per-group energy extraction functions, populated during initialize().
    // Each entry is a callable that returns per-group energies [K] from one Force.
    // After calcForcesAndEnergy(), we call all of these and sum to get per-group PE.
    std::vector<std::function<std::vector<double>()>> groupEnergyExtractors;

    // Sum per-group PE from all registered forces (call after calcForcesAndEnergy)
    void computeGroupPE(std::vector<double>& groupPE) const;

    // Compute per-group KE from current velocities
    void computeGroupKE(OpenMM::ContextImpl& context,
                        std::vector<double>& groupKE) const;

    // RESPA velocity Verlet trajectory
    void respaTrajectory(OpenMM::ContextImpl& context,
                         const MultiGroupHMCIntegrator& integrator);
};

}  // namespace GridForcePlugin

#endif /*OPENMM_REFERENCE_MULTIGROUPHMC_KERNELS_H_*/
