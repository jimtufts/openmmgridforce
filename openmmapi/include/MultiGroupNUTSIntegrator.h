#ifndef OPENMM_MULTIGROUPNUTSINTEGRATOR_H_
#define OPENMM_MULTIGROUPNUTSINTEGRATOR_H_

/* -------------------------------------------------------------------------- *
 *                              OpenMMGridForce                               *
 * -------------------------------------------------------------------------- *
 * MultiGroupNUTSIntegrator: GPU-native multi-group NUTS integrator with     *
 * RESPA multi-timestep integration and per-group temperatures/timesteps.    *
 *                                                                           *
 * Implements the No-U-Turn Sampler (Hoffman & Gelman, 2014) adapted for    *
 * simultaneous multi-group execution on GPU. Each "group" is an            *
 * independent set of atoms (ligand replica) that gets its own temperature, *
 * timestep, adaptive trajectory length, and tree depth.                     *
 *                                                                           *
 * One call to step(1) performs one complete NUTS trial:                     *
 *   1. Draw Maxwell-Boltzmann velocities per group at group temperature    *
 *   2. Sample slice variable u per group                                    *
 *   3. Build NUTS tree via iterative doubling:                              *
 *      a. Choose random direction per active group                          *
 *      b. Take 2^j leapfrog steps (masked for terminated groups)           *
 *      c. Update candidate via uniform selection from valid points          *
 *      d. Check U-turn and divergence per group                             *
 *   4. Set positions to selected candidates                                 *
 *   5. For divergent groups, restore to initial positions                   *
 * -------------------------------------------------------------------------- */

#include <string>
#include <vector>
#include <utility>

#include "internal/windowsExportGridForce.h"
#include "openmm/Integrator.h"
#include "openmm/Kernel.h"
#include "openmm/State.h"

namespace GridForcePlugin {

class OPENMM_EXPORT_GRIDFORCE MultiGroupNUTSIntegrator : public OpenMM::Integrator {
public:
    /**
     * Momentum refreshment modes (same as HMC).
     */
    enum MomentumRefreshMode {
        /** Draw fresh Maxwell-Boltzmann velocities each trial. */
        FULL = 0,
        /** Partial refresh: v_new = cos(theta)*v_old + sin(theta)*v_random. */
        PARTIAL = 1
    };

    /**
     * Metric tensor type for Riemannian NUTS.
     */
    enum MetricType {
        /** Standard diagonal mass matrix (current behavior). */
        METRIC_IDENTITY = 0,
        /** SoftAbs-regularized Hessian (Betancourt 2013). */
        METRIC_SOFTABS  = 1,
        /** Blended: (1-beta)*M + beta*softabs(H). */
        METRIC_BLENDED  = 2
    };

    /**
     * When to recompute the metric tensor.
     */
    enum MetricUpdateMode {
        /** No metric computation (IDENTITY only). */
        METRIC_UPDATE_NONE             = 0,
        /** Recompute G once per NUTS trial (explicit leapfrog with fixed G). */
        METRIC_UPDATE_EVERY_TRAJECTORY = 1
    };

    /**
     * Create a MultiGroupNUTSIntegrator.
     *
     * @param numGroups     number of independent particle groups (replicas)
     * @param atomsPerGroup number of atoms in each group (all groups same size)
     * @param stepSize      leapfrog timestep in picoseconds (used for all groups)
     */
    MultiGroupNUTSIntegrator(int numGroups, int atomsPerGroup, double stepSize);

    // ========== Group Configuration ==========

    int getNumGroups() const { return numGroups; }
    int getAtomsPerGroup() const { return atomsPerGroup; }

    // ========== Per-Group Temperatures ==========

    void setGroupTemperature(int group, double temperature);
    double getGroupTemperature(int group) const;
    void setAllGroupTemperatures(const std::vector<double>& temperatures);
    std::vector<double> getAllGroupTemperatures() const { return groupTemperatures; }

    // ========== RESPA Force Group Schedule ==========

    /**
     * Set the RESPA force group schedule.
     * Each entry is (forceGroupIndex, substeps). See MultiGroupHMCIntegrator
     * for details. If not set, defaults to single-level Verlet.
     */
    void setForceGroupSchedule(const std::vector<std::pair<int,int> >& schedule);
    const std::vector<std::pair<int,int> >& getForceGroupSchedule() const { return forceGroupSchedule; }

    // ========== NUTS Tree Depth ==========

    /**
     * Set the maximum tree depth. The maximum number of leapfrog steps per
     * NUTS trial is 2^maxTreeDepth - 1. Default is 10 (up to 1023 steps).
     */
    void setMaxTreeDepth(int depth);
    int getMaxTreeDepth() const { return maxTreeDepth; }

    // ========== Per-Group Timestep ==========

    void setGroupStepSize(int group, double stepSize);
    double getGroupStepSize(int group) const;
    void setAllGroupStepSizes(const std::vector<double>& stepSizes);
    std::vector<double> getAllGroupStepSizes() const { return groupStepSizes; }

    // ========== Momentum Refreshment ==========

    void setMomentumRefreshMode(MomentumRefreshMode mode) { momentumRefreshMode = mode; }
    MomentumRefreshMode getMomentumRefreshMode() const { return momentumRefreshMode; }
    void setPartialRefreshAngle(double theta) { partialRefreshAngle = theta; }
    double getPartialRefreshAngle() const { return partialRefreshAngle; }

    // ========== External MC Configuration ==========

    /**
     * Set the number of rigid-body MC trials per step.
     * MC trials run as a pre-step before the NUTS trajectory.
     * Set to 0 to disable MC (default).
     */
    void setNumMCTrials(int trials);
    int getNumMCTrials() const { return numMCTrials; }

    /**
     * Set the Gaussian translation step size for MC moves (in nm).
     */
    void setMCStepSize(double stepSize);
    double getMCStepSize() const { return mcStepSize; }

    /**
     * Enable/disable MC moves for a specific group.
     */
    void setGroupMCEnabled(int group, bool enabled);
    bool getGroupMCEnabled(int group) const;

    /**
     * Batch set MC eligibility for all groups.
     */
    void setAllGroupMCEnabled(const std::vector<int>& enabled);
    std::vector<int> getAllGroupMCEnabled() const { return groupMCEnabled; }

    // ========== MC Accept/Reject Results ==========

    int getMCAttempted() const { return mcAttempted; }
    int getMCAccepted() const { return mcAccepted; }
    std::vector<int> getAllGroupMCAccepted() const { return lastMCAccepted; }
    void resetMCCounts();

    // ========== Stability Guard ==========

    /**
     * Set the stability/divergence threshold. NUTS trajectories where BOTH
     * |deltaPE|/kT > threshold AND |deltaH|/kT > threshold are marked as
     * divergent. Divergent groups keep their initial position.
     * Default: 200.0 (matching AlGDock NUTS).
     */
    void setStabilityThreshold(double threshold) { stabilityThreshold = threshold; }
    double getStabilityThreshold() const { return stabilityThreshold; }

    // ========== Tree Depth Diagnostics ==========

    /**
     * Get the tree depth reached by a specific group in the last trial.
     * Tree depth j means 2^j leapfrog steps were taken.
     */
    int getGroupTreeDepth(int group) const;

    /**
     * Get tree depths for all groups from the last trial.
     */
    std::vector<int> getAllGroupTreeDepths() const;

    /**
     * Get mean tree depth for a group (cumulative average over all trials).
     */
    double getGroupMeanTreeDepth(int group) const;

    // ========== Divergence Diagnostics ==========

    /**
     * Get whether a specific group diverged in the last trial.
     * Divergent = trajectory hit the stability threshold.
     */
    bool getGroupDivergent(int group) const;

    /**
     * Get divergence flags for all groups from the last trial.
     * @return vector of flags (1=divergent, 0=ok)
     */
    std::vector<int> getAllGroupDivergent() const;

    // ========== Accept/Reject Results ==========

    /**
     * Get whether a group was accepted in the last trial.
     * Accepted = non-divergent (NUTS always accepts unless divergent).
     */
    bool getGroupAccepted(int group) const;
    std::vector<int> getAllGroupAccepted() const;
    double getGroupAcceptanceRate(int group) const;
    int getGroupAcceptCount(int group) const;
    int getGroupTrialCount(int group) const;
    int getGroupDivergenceCount(int group) const;
    std::vector<int> getAllGroupDivergenceCounts() const;
    void resetAcceptanceCounts();

    // ========== Riemannian Metric Configuration ==========

    /**
     * Set the metric tensor type. Default: METRIC_IDENTITY (standard HMC).
     * METRIC_SOFTABS uses SoftAbs-regularized Hessian as position-dependent mass.
     * METRIC_BLENDED interpolates between mass matrix and SoftAbs Hessian.
     */
    void setMetricType(MetricType type) { metricType = type; }
    MetricType getMetricType() const { return metricType; }

    /**
     * Set the metric update mode. Default: METRIC_UPDATE_NONE.
     * METRIC_UPDATE_EVERY_TRAJECTORY recomputes G once per NUTS trial.
     */
    void setMetricUpdateMode(MetricUpdateMode mode) { metricUpdateMode = mode; }
    MetricUpdateMode getMetricUpdateMode() const { return metricUpdateMode; }

    /**
     * Set the SoftAbs sharpness parameter alpha. Default: 1e6.
     * Larger alpha → softabs(lambda) approaches |lambda| (sharper).
     * Smaller alpha → softabs(lambda) approaches 1/alpha (more uniform).
     */
    void setSoftAbsAlpha(double alpha);
    double getSoftAbsAlpha() const { return softAbsAlpha; }

    /**
     * Set the metric blend factor beta in [0, 1]. Default: 1.0.
     * G = (1-beta)*m*I + beta*softabs(H).
     * beta=0 recovers standard HMC; beta=1 gives full Riemannian.
     * Only used when MetricType is METRIC_BLENDED.
     */
    void setMetricBlendFactor(double beta);
    double getMetricBlendFactor() const { return metricBlendFactor; }

    /**
     * Set the weight of the grid force Hessian in the metric. Default: 0.0.
     * H_combined = H_bonded + w * H_grid.
     * w=0 uses bonded-only preconditioning (grid forces confine normally).
     * w=1 includes full grid Hessian (neutralizes grid confinement).
     */
    void setGridHessianWeight(double w) { gridHessianWeight = w; }
    double getGridHessianWeight() const { return gridHessianWeight; }

    /**
     * Get per-group metric condition numbers from the last metric assembly.
     * @return vector of condition numbers (max_eig/min_eig), one per group
     */
    std::vector<double> getGroupMetricConditionNumbers() const;

    // ========== External Diagonal Hessian ==========

    /**
     * Set an external diagonal Hessian to be accumulated into the metric.
     * This allows injecting Hessian contributions from forces not directly
     * discoverable by the CUDA kernel (e.g., OBC solvation computed via JAX).
     *
     * The buffer layout is 6 floats per atom (Hxx, Hyy, Hzz, Hxy, Hxz, Hyz),
     * with atoms ordered as [group0_atom0, group0_atom1, ..., groupK_atomN].
     * Total size must be 6 * numGroups * atomsPerGroup.
     *
     * @param hessian flat vector of diagonal Hessian blocks
     */
    void setExternalDiagonalHessian(const std::vector<float>& hessian);

    /**
     * Check whether an external Hessian has been set.
     */
    bool hasExternalHessian() const { return !externalHessian.empty(); }

    /**
     * Clear the external Hessian (disable external accumulation).
     */
    void clearExternalHessian() { externalHessian.clear(); }

    /**
     * Get the external diagonal Hessian buffer.
     */
    const std::vector<float>& getExternalDiagonalHessian() const { return externalHessian; }

    // ========== GPU Tree Building Toggle ==========

    /**
     * Enable/disable GPU-side tree building. When enabled (default),
     * NUTS tree decisions (divergence, slice, candidate selection) are
     * computed on the GPU with zero CPU-GPU syncs per leapfrog step.
     * When disabled, decisions are computed on the host (slower but
     * useful for validation).
     */
    void setGpuTreeBuilding(bool enabled) { gpuTreeBuilding = enabled; }
    bool getGpuTreeBuilding() const { return gpuTreeBuilding; }

    // ========== Random Number Seed ==========

    int getRandomNumberSeed() const { return randomNumberSeed; }
    void setRandomNumberSeed(int seed) { randomNumberSeed = seed; }

    // ========== Integrator Interface ==========

    /**
     * Advance the simulation by taking a series of NUTS trials.
     * Each step is one complete NUTS trial with adaptive trajectory length.
     */
    void step(int steps);

protected:
    void initialize(OpenMM::ContextImpl& context);
    void cleanup();
    void stateChanged(OpenMM::State::DataType changed);
    std::vector<std::string> getKernelNames();
    double computeKineticEnergy();

private:
    int numGroups;
    int atomsPerGroup;

    std::vector<double> groupTemperatures;
    std::vector<double> groupStepSizes;
    std::vector<std::pair<int,int> > forceGroupSchedule;

    int maxTreeDepth;

    MomentumRefreshMode momentumRefreshMode;
    double partialRefreshAngle;
    double stabilityThreshold;
    bool gpuTreeBuilding;

    // MC configuration
    int numMCTrials;
    double mcStepSize;
    std::vector<int> groupMCEnabled;
    int mcAttempted;
    int mcAccepted;
    std::vector<int> lastMCAccepted;

    int randomNumberSeed;

    // Riemannian metric configuration
    MetricType metricType;
    MetricUpdateMode metricUpdateMode;
    double softAbsAlpha;
    double metricBlendFactor;
    double gridHessianWeight;

    // External diagonal Hessian (e.g., OBC solvation from JAX)
    std::vector<float> externalHessian;

    // Acceptance statistics (host-side)
    std::vector<int> acceptCounts;
    std::vector<int> trialCounts;
    std::vector<int> divergenceCounts;

    // Tree depth statistics
    std::vector<long long> cumulativeTreeDepths;

    // Last trial results (host-side)
    mutable std::vector<int> lastAccepted;
    mutable std::vector<int> lastTreeDepths;
    mutable std::vector<int> lastDivergent;

    // Kernel
    OpenMM::Kernel kernel;
    bool forcesAreValid;
};

}  // namespace GridForcePlugin

#endif /*OPENMM_MULTIGROUPNUTSINTEGRATOR_H_*/
