#ifndef OPENMM_MULTIGROUPHMC_KERNELS_H_
#define OPENMM_MULTIGROUPHMC_KERNELS_H_

/* -------------------------------------------------------------------------- *
 *                              OpenMMGridForce                               *
 * -------------------------------------------------------------------------- *
 * Kernel interface for MultiGroupHMCIntegrator.                             *
 * Each platform (CUDA, Reference) provides a concrete implementation.       *
 * -------------------------------------------------------------------------- */

#include "MultiGroupHMCIntegrator.h"
#include "openmm/KernelImpl.h"
#include "openmm/Platform.h"
#include "openmm/System.h"
#include <string>
#include <vector>

namespace GridForcePlugin {

/**
 * This kernel is invoked by MultiGroupHMCIntegrator to perform one HMC trial
 * with RESPA multi-timestep integration and per-group accept/reject.
 */
class IntegrateMultiGroupHMCStepKernel : public OpenMM::KernelImpl {
public:
    static std::string Name() {
        return "IntegrateMultiGroupHMCStep";
    }

    IntegrateMultiGroupHMCStepKernel(std::string name, const OpenMM::Platform& platform)
        : OpenMM::KernelImpl(name, platform) {}

    /**
     * Initialize the kernel with system and integrator parameters.
     * Allocates buffers for position backup, per-group energies, etc.
     *
     * @param system     the System this kernel will be applied to
     * @param integrator the MultiGroupHMCIntegrator this kernel is used for
     */
    virtual void initialize(const OpenMM::System& system,
                           const MultiGroupHMCIntegrator& integrator) = 0;

    /**
     * Execute one complete HMC trial:
     *   1. Backup positions
     *   2. Draw MB velocities per group
     *   3. Compute initial KE + PE per group
     *   4. RESPA NVE trajectory
     *   5. Compute final KE + PE per group
     *   6. Metropolis accept/reject per group (with stability guard)
     *   7. Restore positions for rejected groups
     *
     * @param context        the context in which to execute
     * @param integrator     the integrator (provides temperatures, step sizes, etc.)
     * @param forcesAreValid false if cached forces are stale and must be recomputed
     */
    virtual void execute(OpenMM::ContextImpl& context,
                        const MultiGroupHMCIntegrator& integrator,
                        bool forcesAreValid) = 0;

    /**
     * Compute the total kinetic energy across all groups.
     */
    virtual double computeKineticEnergy(OpenMM::ContextImpl& context,
                                        const MultiGroupHMCIntegrator& integrator) = 0;

    /**
     * Get the accept/reject flags from the last execute() call.
     * @return vector of length numGroups (1=accepted, 0=rejected)
     */
    virtual std::vector<int> getAcceptedFlags() const = 0;

    /**
     * Get per-group deltaH from the last execute() call.
     * @return vector of length numGroups (kJ/mol)
     */
    virtual std::vector<double> getDeltaH() const = 0;

    /**
     * Get per-group accept counts (cumulative).
     */
    virtual std::vector<int> getAcceptCounts() const = 0;

    /**
     * Get per-group trial counts (cumulative).
     */
    virtual std::vector<int> getTrialCounts() const = 0;

    /**
     * Get per-group stability-guard rejection counts (cumulative).
     */
    virtual std::vector<int> getStabilityRejectCounts() const = 0;

    /**
     * Reset all acceptance/trial/stability counters.
     */
    virtual void resetCounters() = 0;

    // ========== External MC ==========

    /**
     * Execute rigid-body MC trials for eligible groups as a pre-step.
     * Called from execute() when numMCTrials > 0.
     */
    virtual void executeMC(OpenMM::ContextImpl& context,
                           const MultiGroupHMCIntegrator& integrator) = 0;

    /**
     * Get cumulative MC attempted count.
     */
    virtual int getMCAttempted() const = 0;

    /**
     * Get cumulative MC accepted count.
     */
    virtual int getMCAccepted() const = 0;

    /**
     * Get per-group MC accepted counts from the last step.
     */
    virtual std::vector<int> getLastMCAccepted() const = 0;

    /**
     * Reset MC counters.
     */
    virtual void resetMCCounters() = 0;
};

}  // namespace GridForcePlugin

#endif /*OPENMM_MULTIGROUPHMC_KERNELS_H_*/
