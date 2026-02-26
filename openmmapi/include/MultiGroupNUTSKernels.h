#ifndef OPENMM_MULTIGROUPNUTS_KERNELS_H_
#define OPENMM_MULTIGROUPNUTS_KERNELS_H_

/* -------------------------------------------------------------------------- *
 *                              OpenMMGridForce                               *
 * -------------------------------------------------------------------------- *
 * Kernel interface for MultiGroupNUTSIntegrator.                            *
 * Each platform (CUDA, Reference) provides a concrete implementation.       *
 * -------------------------------------------------------------------------- */

#include "MultiGroupNUTSIntegrator.h"
#include "openmm/KernelImpl.h"
#include "openmm/Platform.h"
#include "openmm/System.h"
#include <string>
#include <vector>

namespace GridForcePlugin {

/**
 * This kernel is invoked by MultiGroupNUTSIntegrator to perform one NUTS trial
 * with tree-doubling, per-group U-turn detection, and adaptive trajectory length.
 */
class IntegrateMultiGroupNUTSStepKernel : public OpenMM::KernelImpl {
public:
    static std::string Name() {
        return "IntegrateMultiGroupNUTSStep";
    }

    IntegrateMultiGroupNUTSStepKernel(std::string name, const OpenMM::Platform& platform)
        : OpenMM::KernelImpl(name, platform) {}

    /**
     * Initialize the kernel with system and integrator parameters.
     */
    virtual void initialize(const OpenMM::System& system,
                           const MultiGroupNUTSIntegrator& integrator) = 0;

    /**
     * Execute one complete NUTS trial with tree-doubling.
     */
    virtual void execute(OpenMM::ContextImpl& context,
                        const MultiGroupNUTSIntegrator& integrator,
                        bool forcesAreValid) = 0;

    /**
     * Compute the total kinetic energy across all groups.
     */
    virtual double computeKineticEnergy(OpenMM::ContextImpl& context,
                                        const MultiGroupNUTSIntegrator& integrator) = 0;

    /**
     * Get per-group tree depths from the last execute() call.
     */
    virtual std::vector<int> getTreeDepths() const = 0;

    /**
     * Get per-group divergence flags from the last execute() call.
     * @return vector of length numGroups (1=divergent, 0=ok)
     */
    virtual std::vector<int> getDivergentFlags() const = 0;

    /**
     * Get per-group accept flags from the last execute() call.
     * @return vector of length numGroups (1=accepted, 0=rejected/divergent)
     */
    virtual std::vector<int> getAcceptedFlags() const = 0;

    virtual std::vector<int> getAcceptCounts() const = 0;
    virtual std::vector<int> getTrialCounts() const = 0;
    virtual std::vector<int> getDivergenceCounts() const = 0;
    virtual std::vector<long long> getCumulativeTreeDepths() const = 0;
    virtual void resetCounters() = 0;

    // ========== External MC ==========
    virtual void executeMC(OpenMM::ContextImpl& context,
                           const MultiGroupNUTSIntegrator& integrator) = 0;
    virtual int getMCAttempted() const = 0;
    virtual int getMCAccepted() const = 0;
    virtual std::vector<int> getLastMCAccepted() const = 0;
    virtual void resetMCCounters() = 0;
};

}  // namespace GridForcePlugin

#endif /*OPENMM_MULTIGROUPNUTS_KERNELS_H_*/
