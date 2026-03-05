#ifndef OPENMM_CUDA_MULTIGROUPNUTS_KERNELS_H_
#define OPENMM_CUDA_MULTIGROUPNUTS_KERNELS_H_

/* -------------------------------------------------------------------------- *
 *                              OpenMMGridForce                               *
 * -------------------------------------------------------------------------- *
 * CUDA implementation of MultiGroupNUTSIntegrator kernel.                    *
 * -------------------------------------------------------------------------- */

#include "MultiGroupNUTSKernels.h"
#include "internal/GridForceImpl.h"
#include "internal/IsolatedBondedForceImpl.h"
#include "internal/IsolatedNonbondedForceImpl.h"
#include "openmm/Platform.h"
#include "openmm/cuda/CudaContext.h"
#include "openmm/cuda/CudaArray.h"
#include <vector>
#include <random>
#include <functional>

namespace GridForcePlugin {

class CudaIntegrateMultiGroupNUTSStepKernel : public IntegrateMultiGroupNUTSStepKernel {
public:
    CudaIntegrateMultiGroupNUTSStepKernel(std::string name,
                                           const OpenMM::Platform& platform,
                                           OpenMM::CudaContext& cu)
        : IntegrateMultiGroupNUTSStepKernel(name, platform),
          cu(cu), hasInitializedKernel(false),
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

    // Riemannian metric interface
    void assembleMetric(OpenMM::ContextImpl& context,
                       const MultiGroupNUTSIntegrator& integrator) override;
    std::vector<double> getGroupMetricConditionNumbers() const override;

    // MC interface
    void executeMC(OpenMM::ContextImpl& context,
                   const MultiGroupNUTSIntegrator& integrator) override;
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

    // GPU buffers (same as HMC)
    OpenMM::CudaArray positionsBackup;
    OpenMM::CudaArray slowForcesBackup;
    OpenMM::CudaArray groupKEBuffer;
    OpenMM::CudaArray groupStepSizesBuffer;
    OpenMM::CudaArray groupKTBuffer;

    // NUTS-specific GPU buffers
    OpenMM::CudaArray xminusBuffer;      // real4, tree backward endpoint positions
    OpenMM::CudaArray xplusBuffer;       // real4, tree forward endpoint positions
    OpenMM::CudaArray vminusBuffer;      // mixed4, tree backward endpoint velocities
    OpenMM::CudaArray vplusBuffer;       // mixed4, tree forward endpoint velocities
    OpenMM::CudaArray candidateBuffer;          // real4, main candidate positions
    OpenMM::CudaArray subtreeCandidateBuffer;   // real4, subtree candidate (bug fix: separate from main)
    OpenMM::CudaArray activeBuffer;      // int[K], per-group active mask
    OpenMM::CudaArray directionBuffer;   // int[K], per-group direction (+1/-1)
    OpenMM::CudaArray copyFlagBuffer;    // int[K], flag for candidate copy
    OpenMM::CudaArray uturnDotBuffer;    // double[2*K], U-turn dot products
    OpenMM::CudaArray divergentBuffer;   // int[K], per-group divergence flags

    // GPU-side tree state buffers (Phase 2: zero-sync inner loop)
    OpenMM::CudaArray loguBuffer;                // double[K], log slice variable
    OpenMM::CudaArray H0Buffer;                  // double[K], initial Hamiltonian
    OpenMM::CudaArray peInitBuffer;              // double[K], initial PE
    OpenMM::CudaArray totalGroupPEBuffer;        // double[K], consolidated PE from all forces
    OpenMM::CudaArray subtreeNValidBuffer;       // int[K], valid points in current subtree
    OpenMM::CudaArray nValidBuffer;              // int[K], total valid points
    OpenMM::CudaArray subtreeCandidatePEBuffer;  // double[K], PE of subtree candidate
    OpenMM::CudaArray candidatePEBuffer;         // double[K], PE of main candidate
    OpenMM::CudaArray subtreeHasCandidateBuffer; // int[K], subtree produced a candidate
    OpenMM::CudaArray anyActiveBuffer;           // int[1], reduction flag for depth loop exit

    // Force energy buffer pointers (stored on GPU for gather kernel)
    OpenMM::CudaArray forceEnergyPtrsBuffer;     // unsigned long long[numForces]
    std::vector<unsigned long long> forceEnergyPtrsHost;
    int numEnergyForces = 0;

    // Host-side caches
    std::vector<double> groupKEHost;
    std::vector<double> groupStepSizesHost;
    std::vector<double> groupKTHost;
    std::vector<int> activeHost;
    std::vector<int> directionHost;
    std::vector<int> copyFlagHost;
    std::vector<double> uturnDotHost;
    std::vector<int> divergentHost;

    // Per-group PE extraction (host-side lambdas, same as HMC)
    std::vector<std::function<std::vector<double>()>> groupEnergyExtractors;

    // CUDA kernel handles (shared with HMC pattern)
    CUfunction backupPositionsKernel;
    CUfunction drawMBVelocitiesFullKernel;
    CUfunction drawMBVelocitiesPartialKernel;
    CUfunction computeGroupKEKernel;
    CUfunction copyForcesKernel;

    // NUTS-specific CUDA kernel handles
    CUfunction restoreEndpointKernel;
    CUfunction saveEndpointKernel;
    CUfunction velocityKickKernel;
    CUfunction positionDriftKernel;
    CUfunction copyCandidatePosKernel;
    CUfunction computeUTurnDotKernel;
    CUfunction setFromCandidateKernel;
    CUfunction restoreDivergentKernel;
    CUfunction initializeTreeKernel;

    // Phase 2: GPU-side tree building kernel handles
    CUfunction gatherForceEnergiesKernel;
    CUfunction leapfrogDecisionKernel;
    CUfunction checkUTurnAndDeactivateKernel;
    CUfunction combineCandidatesKernel;
    CUfunction setDirectionKernel;

    // Last step results
    std::vector<int> lastAccepted;
    std::vector<int> lastTreeDepths;
    std::vector<int> lastDivergent;

    // Cumulative statistics
    std::vector<int> acceptCounts;
    std::vector<int> trialCounts;
    std::vector<int> divergenceCounts;
    std::vector<long long> cumulativeTreeDepths;

    // Host RNG
    std::mt19937 rng;
    std::uniform_real_distribution<double> uniformDist;
    std::exponential_distribution<double> exponentialDist;
    std::normal_distribution<double> normalDist;

    // MC GPU buffers
    OpenMM::CudaArray mcGroupCOMBuffer;
    OpenMM::CudaArray mcEnabledBuffer;
    OpenMM::CudaArray mcRotationBuffer;
    OpenMM::CudaArray mcTranslationBuffer;
    OpenMM::CudaArray mcCOMBuffer;
    OpenMM::CudaArray mcAcceptedBuffer;

    // MC host-side vectors
    std::vector<double> mcGroupCOMHost;
    std::vector<int> mcEnabledHost;
    std::vector<double> mcRotationHost;
    std::vector<double> mcTranslationHost;
    std::vector<double> mcCOMHost;
    std::vector<int> mcAcceptedHost;

    // MC CUDA kernels
    CUfunction mcComputeCOMKernel;
    CUfunction mcApplyRigidBodyMoveKernel;
    CUfunction mcRestoreRejectedKernel;

    // MC counters
    int mcAttemptedTotal;
    int mcAcceptedTotal;
    std::vector<int> lastMCAcceptedPerGroup;

    // Riemannian metric buffers
    OpenMM::CudaArray metricBuffer;          // float[totalAtoms * 6] — G
    OpenMM::CudaArray metricInvBuffer;       // float[totalAtoms * 6] — G^{-1}
    OpenMM::CudaArray choleskyBuffer;        // float[totalAtoms * 6] — Cholesky(G^{-1})
    OpenMM::CudaArray logDetBuffer;          // double[numGroups]
    OpenMM::CudaArray conditionBuffer;       // float[numGroups]
    OpenMM::CudaArray combinedHessianBuffer; // float[totalAtoms * 6] — accumulated Hessian

    // Metric CUDA kernel handles
    CUfunction assembleMetricKernel;
    CUfunction rmVelocityKickKernel;
    CUfunction rmComputeGroupKEKernel;
    CUfunction rmDrawMBVelocitiesFullKernel;
    CUfunction rmDrawMBVelocitiesPartialKernel;
    CUfunction setIdentityMetricKernel;
    CUfunction accumulateHessianKernel;
    CUfunction accumulateHessianWeightedKernel;

    // Metric state
    bool metricInitialized;
    std::vector<float> conditionNumbersHost;

    // GridForceImpl pointers for Hessian computation (all grid forces)
    std::vector<GridForceImpl*> gridForceImpls;

    // IsolatedBondedForceImpl pointer for bonded Hessian
    IsolatedBondedForceImpl* bondedForceImpl = nullptr;

    // IsolatedNonbondedForceImpl pointer for LJ+Coulomb Hessian
    IsolatedNonbondedForceImpl* nonbondedForceImpl = nullptr;

    // External Hessian buffer (uploaded from host, e.g. OBC solvation from JAX)
    OpenMM::CudaArray externalHessianBuffer;  // float[totalAtoms * 6]

    // Helper methods
    void computeGroupPE(std::vector<double>& groupPE) const;
    void computeGroupKE(OpenMM::ContextImpl& context, std::vector<double>& groupKE);
    void launchKick(CUdeviceptr forcePtr, double scaleFactor);
    void launchDrift();
    void nutsLeapfrogStep(OpenMM::ContextImpl& context,
                          const MultiGroupNUTSIntegrator& integrator,
                          CUdeviceptr forcePtr,
                          bool includeEnergy = false);
    void respaLeapfrogStep(OpenMM::ContextImpl& context,
                           const MultiGroupNUTSIntegrator& integrator,
                           bool includeEnergy = false);
};

}  // namespace GridForcePlugin

#endif /*OPENMM_CUDA_MULTIGROUPNUTS_KERNELS_H_*/
