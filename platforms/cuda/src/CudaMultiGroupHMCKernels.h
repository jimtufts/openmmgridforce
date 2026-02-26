#ifndef OPENMM_CUDA_MULTIGROUPHMC_KERNELS_H_
#define OPENMM_CUDA_MULTIGROUPHMC_KERNELS_H_

/* -------------------------------------------------------------------------- *
 *                              OpenMMGridForce                               *
 * -------------------------------------------------------------------------- *
 * CUDA implementation of MultiGroupHMCIntegrator kernel.                    *
 * -------------------------------------------------------------------------- */

#include "MultiGroupHMCKernels.h"
#include "openmm/Platform.h"
#include "openmm/cuda/CudaContext.h"
#include "openmm/cuda/CudaArray.h"
#include <vector>
#include <random>
#include <functional>

namespace GridForcePlugin {

class CudaIntegrateMultiGroupHMCStepKernel : public IntegrateMultiGroupHMCStepKernel {
public:
    CudaIntegrateMultiGroupHMCStepKernel(std::string name,
                                          const OpenMM::Platform& platform,
                                          OpenMM::CudaContext& cu)
        : IntegrateMultiGroupHMCStepKernel(name, platform),
          cu(cu), hasInitializedKernel(false),
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

    // MC interface
    void executeMC(OpenMM::ContextImpl& context,
                   const MultiGroupHMCIntegrator& integrator) override;
    int getMCAttempted() const override { return mcAttemptedTotal; }
    int getMCAccepted() const override { return mcAcceptedTotal; }
    std::vector<int> getLastMCAccepted() const override { return lastMCAcceptedPerGroup; }
    void resetMCCounters() override;

private:
    OpenMM::CudaContext& cu;
    bool hasInitializedKernel;
    int numGroups;
    int atomsPerGroup;
    int numParticles;

    // GPU buffers
    OpenMM::CudaArray positionsBackup;
    OpenMM::CudaArray slowForcesBackup;
    OpenMM::CudaArray groupKEBuffer;
    OpenMM::CudaArray groupStepSizesBuffer;
    OpenMM::CudaArray groupKTBuffer;
    OpenMM::CudaArray acceptedBuffer;

    // Host-side caches
    std::vector<double> groupKEHost;
    std::vector<double> groupStepSizesHost;
    std::vector<double> groupKTHost;
    std::vector<int> acceptedHost;

    // Per-group PE extraction (host-side lambdas, same pattern as Reference)
    std::vector<std::function<std::vector<double>()>> groupEnergyExtractors;

    // Compiled CUDA kernels
    CUfunction backupPositionsKernel;
    CUfunction drawMBVelocitiesFullKernel;
    CUfunction drawMBVelocitiesPartialKernel;
    CUfunction computeGroupKEKernel;
    CUfunction velocityKickKernel;
    CUfunction positionDriftKernel;
    CUfunction copyForcesKernel;
    CUfunction restoreRejectedKernel;

    // Last step results
    std::vector<int> lastAccepted;
    std::vector<double> lastDeltaH;

    // Cumulative statistics
    std::vector<int> acceptCounts;
    std::vector<int> trialCounts;
    std::vector<int> stabilityRejectCounts;

    // Host RNG for Metropolis
    std::mt19937 rng;
    std::uniform_real_distribution<double> uniformDist;
    std::normal_distribution<double> normalDist;

    // MC GPU buffers
    OpenMM::CudaArray groupCOMBuffer;       // 4 doubles per group (sum_mx, sum_my, sum_mz, sum_m)
    OpenMM::CudaArray mcEnabledBuffer;      // int per group
    OpenMM::CudaArray mcRotationBuffer;     // 9 doubles per group (3x3 row-major)
    OpenMM::CudaArray mcTranslationBuffer;  // 3 doubles per group
    OpenMM::CudaArray mcCOMBuffer;          // 3 doubles per group (finalized COM)

    // MC host-side vectors
    std::vector<double> groupCOMHost;        // 4 * numGroups
    std::vector<int> mcEnabledHost;
    std::vector<double> mcRotationHost;      // 9 * numGroups
    std::vector<double> mcTranslationHost;   // 3 * numGroups
    std::vector<double> mcCOMHost;           // 3 * numGroups

    // MC CUDA kernels
    CUfunction mcComputeCOMKernel;
    CUfunction mcApplyRigidBodyMoveKernel;

    // MC counters
    int mcAttemptedTotal;
    int mcAcceptedTotal;
    std::vector<int> lastMCAcceptedPerGroup;

    void computeGroupPE(std::vector<double>& groupPE) const;
    void computeGroupKE(OpenMM::ContextImpl& context, std::vector<double>& groupKE);
    void respaTrajectory(OpenMM::ContextImpl& context,
                         const MultiGroupHMCIntegrator& integrator);
    void launchKick(CUdeviceptr forcePtr, double scaleFactor);
    void launchDrift();
};

}  // namespace GridForcePlugin

#endif /*OPENMM_CUDA_MULTIGROUPHMC_KERNELS_H_*/
