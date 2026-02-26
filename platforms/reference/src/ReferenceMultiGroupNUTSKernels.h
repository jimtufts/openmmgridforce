#ifndef OPENMM_REFERENCE_MULTIGROUPNUTS_KERNELS_H_
#define OPENMM_REFERENCE_MULTIGROUPNUTS_KERNELS_H_

/* -------------------------------------------------------------------------- *
 *                              OpenMMGridForce                               *
 * -------------------------------------------------------------------------- *
 * Reference (CPU) implementation of MultiGroupNUTSIntegrator kernel.         *
 * -------------------------------------------------------------------------- */

#include "MultiGroupNUTSKernels.h"
#include "openmm/Platform.h"
#include "openmm/Vec3.h"
#include <vector>
#include <random>
#include <functional>

namespace GridForcePlugin {

class ReferenceIntegrateMultiGroupNUTSStepKernel : public IntegrateMultiGroupNUTSStepKernel {
public:
    ReferenceIntegrateMultiGroupNUTSStepKernel(std::string name,
                                                const OpenMM::Platform& platform)
        : IntegrateMultiGroupNUTSStepKernel(name, platform),
          numGroups(0), atomsPerGroup(0), numParticles(0) {}

    void initialize(const OpenMM::System& system,
                   const MultiGroupNUTSIntegrator& integrator) override;

    void execute(OpenMM::ContextImpl& context,
                const MultiGroupNUTSIntegrator& integrator,
                bool forcesAreValid) override;

    double computeKineticEnergy(OpenMM::ContextImpl& context,
                                const MultiGroupNUTSIntegrator& integrator) override;

    std::vector<int> getTreeDepths() const override { return lastTreeDepths; }
    std::vector<int> getDivergentFlags() const override { return lastDivergent; }
    std::vector<int> getAcceptedFlags() const override { return lastAccepted; }
    std::vector<int> getAcceptCounts() const override { return acceptCounts; }
    std::vector<int> getTrialCounts() const override { return trialCounts; }
    std::vector<int> getDivergenceCounts() const override { return divergenceCounts; }
    std::vector<long long> getCumulativeTreeDepths() const override { return cumulativeTreeDepths; }
    void resetCounters() override;

    // MC stubs (Reference platform — MC not yet implemented)
    void executeMC(OpenMM::ContextImpl& context,
                   const MultiGroupNUTSIntegrator& integrator) override {}
    int getMCAttempted() const override { return 0; }
    int getMCAccepted() const override { return 0; }
    std::vector<int> getLastMCAccepted() const override { return std::vector<int>(numGroups, 0); }
    void resetMCCounters() override {}

private:
    int numGroups;
    int atomsPerGroup;
    int numParticles;

    std::vector<double> masses;

    // Position backup for divergent groups
    std::vector<OpenMM::Vec3> positionsBackup;

    // Tree endpoint buffers
    std::vector<OpenMM::Vec3> xminus, xplus;
    std::vector<OpenMM::Vec3> vminus, vplus;
    std::vector<OpenMM::Vec3> candidatePos;
    std::vector<OpenMM::Vec3> subtreeCandidatePos;

    // Last step results
    std::vector<int> lastAccepted;
    std::vector<int> lastTreeDepths;
    std::vector<int> lastDivergent;

    // Cumulative statistics
    std::vector<int> acceptCounts;
    std::vector<int> trialCounts;
    std::vector<int> divergenceCounts;
    std::vector<long long> cumulativeTreeDepths;

    // RNG
    std::mt19937 rng;
    std::normal_distribution<double> normalDist;
    std::uniform_real_distribution<double> uniformDist;
    std::exponential_distribution<double> exponentialDist;

    std::vector<std::function<std::vector<double>()>> groupEnergyExtractors;

    void computeGroupPE(std::vector<double>& groupPE) const;
    void computeGroupKE(OpenMM::ContextImpl& context, std::vector<double>& groupKE) const;

    void leapfrogStep(OpenMM::ContextImpl& context,
                      const MultiGroupNUTSIntegrator& integrator,
                      const std::vector<int>& active,
                      const std::vector<double>& signedDt,
                      bool includeEnergy = false);
};

}  // namespace GridForcePlugin

#endif /*OPENMM_REFERENCE_MULTIGROUPNUTS_KERNELS_H_*/
