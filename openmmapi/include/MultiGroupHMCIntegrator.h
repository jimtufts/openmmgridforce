#ifndef OPENMM_MULTIGROUPHMCINTEGRATOR_H_
#define OPENMM_MULTIGROUPHMCINTEGRATOR_H_

/* -------------------------------------------------------------------------- *
 *                              OpenMMGridForce                               *
 * -------------------------------------------------------------------------- *
 * MultiGroupHMCIntegrator: GPU-native multi-group HMC integrator with       *
 * RESPA multi-timestep integration and per-group temperatures/timesteps.    *
 *                                                                           *
 * Designed for parallel tempering with grid-based binding PMF calculations. *
 * Each "group" is an independent set of atoms (ligand replica) that gets    *
 * its own temperature, timestep, and accept/reject decision.                *
 *                                                                           *
 * One call to step(1) performs one complete HMC trial:                      *
 *   1. Backup positions                                                     *
 *   2. Draw Maxwell-Boltzmann velocities per group at group temperature     *
 *   3. Compute initial KE and PE per group                                  *
 *   4. Run RESPA NVE trajectory (numOuterSteps outer steps)                 *
 *   5. Compute final KE and PE per group                                    *
 *   6. Metropolis accept/reject per group independently                     *
 *   7. Restore positions for rejected groups                                *
 * -------------------------------------------------------------------------- */

#include <string>
#include <vector>
#include <utility>

#include "internal/windowsExportGridForce.h"
#include "openmm/Integrator.h"
#include "openmm/Kernel.h"
#include "openmm/State.h"

namespace GridForcePlugin {

class OPENMM_EXPORT_GRIDFORCE MultiGroupHMCIntegrator : public OpenMM::Integrator {
public:
    /**
     * Momentum refreshment modes for HMC.
     */
    enum MomentumRefreshMode {
        /** Draw fresh Maxwell-Boltzmann velocities each trial. */
        FULL = 0,
        /** Partial refresh: v_new = cos(theta)*v_old + sin(theta)*v_random. */
        PARTIAL = 1
    };

    /**
     * Metric tensor type for Riemannian HMC.
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
        /** Recompute G once per HMC trial (explicit leapfrog with fixed G). */
        METRIC_UPDATE_EVERY_TRAJECTORY = 1
    };

    /**
     * Create a MultiGroupHMCIntegrator.
     *
     * @param numGroups     number of independent particle groups (replicas)
     * @param atomsPerGroup number of atoms in each group (all groups same size)
     * @param stepSize      initial outer timestep in picoseconds (used for all groups)
     */
    MultiGroupHMCIntegrator(int numGroups, int atomsPerGroup, double stepSize);

    // ========== Group Configuration ==========

    /**
     * Get the number of particle groups.
     */
    int getNumGroups() const { return numGroups; }

    /**
     * Get the number of atoms per group.
     */
    int getAtomsPerGroup() const { return atomsPerGroup; }

    // ========== Per-Group Temperatures ==========

    /**
     * Set the temperature for a specific group.
     * @param group       group index (0 to numGroups-1)
     * @param temperature temperature in Kelvin
     */
    void setGroupTemperature(int group, double temperature);

    /**
     * Get the temperature for a specific group.
     * @param group  group index
     * @return temperature in Kelvin
     */
    double getGroupTemperature(int group) const;

    /**
     * Set temperatures for all groups at once.
     * @param temperatures vector of length numGroups, in Kelvin
     */
    void setAllGroupTemperatures(const std::vector<double>& temperatures);

    /**
     * Get temperatures for all groups.
     * @return vector of temperatures in Kelvin
     */
    std::vector<double> getAllGroupTemperatures() const { return groupTemperatures; }

    // ========== RESPA Force Group Schedule ==========

    /**
     * Set the RESPA force group schedule.
     *
     * Each entry is (forceGroupIndex, substeps). Force group 0 with substeps=1
     * is the slow group (evaluated once per outer step). Force group 1 with
     * substeps=4 is the fast group (evaluated 4x per outer step).
     *
     * If not set, defaults to single-level Verlet (all forces every step).
     *
     * @param schedule vector of (forceGroupIndex, substeps) pairs
     */
    void setForceGroupSchedule(const std::vector<std::pair<int,int> >& schedule);

    /**
     * Get the current RESPA force group schedule.
     */
    const std::vector<std::pair<int,int> >& getForceGroupSchedule() const { return forceGroupSchedule; }

    // ========== HMC Trajectory Length ==========

    /**
     * Set the number of outer RESPA steps per HMC trial for all groups.
     * Total trajectory length for group k = groupStepSize[k] * numOuterSteps.
     * This is the legacy single-value API. For per-group control (matching
     * AlGDock reference where each state adapts independently), use
     * setGroupStepsPerTrial() or setAllGroupStepsPerTrial().
     * @param steps number of outer steps
     */
    void setNumOuterSteps(int steps);

    /**
     * Get the maximum number of outer RESPA steps across all groups.
     */
    int getNumOuterSteps() const { return numOuterSteps; }

    /**
     * Set per-group outer steps. Groups with fewer steps have their dt
     * effectively zeroed for the remaining outer steps.
     * @param group group index
     * @param steps number of outer steps for this group
     */
    void setGroupStepsPerTrial(int group, int steps);

    /**
     * Get outer steps for a specific group.
     */
    int getGroupStepsPerTrial(int group) const;

    /**
     * Set outer steps for all groups at once. Updates numOuterSteps to
     * the maximum across groups.
     * @param steps vector of length numGroups
     */
    void setAllGroupStepsPerTrial(const std::vector<int>& steps);

    /**
     * Get outer steps for all groups.
     */
    std::vector<int> getAllGroupStepsPerTrial() const { return groupStepsPerTrial; }

    // ========== Per-Group Timestep ==========

    /**
     * Set the outer timestep for a specific group. The RESPA inner timestep
     * for this group is groupStepSize / respaRatio.
     *
     * @param group    group index
     * @param stepSize timestep in picoseconds
     */
    void setGroupStepSize(int group, double stepSize);

    /**
     * Get the outer timestep for a specific group.
     * @param group  group index
     * @return timestep in picoseconds
     */
    double getGroupStepSize(int group) const;

    /**
     * Set timesteps for all groups at once.
     * @param stepSizes vector of length numGroups, in picoseconds
     */
    void setAllGroupStepSizes(const std::vector<double>& stepSizes);

    /**
     * Get timesteps for all groups.
     * @return vector of timesteps in picoseconds
     */
    std::vector<double> getAllGroupStepSizes() const { return groupStepSizes; }

    // ========== Momentum Refreshment ==========

    /**
     * Set the momentum refreshment mode.
     * FULL: draw fresh MB velocities each trial (default).
     * PARTIAL: v_new = cos(theta)*v_old + sin(theta)*v_random.
     */
    void setMomentumRefreshMode(MomentumRefreshMode mode) { momentumRefreshMode = mode; }
    MomentumRefreshMode getMomentumRefreshMode() const { return momentumRefreshMode; }

    /**
     * Set the partial refresh angle (radians). Only used when mode is PARTIAL.
     * theta = pi/2 is equivalent to FULL refresh.
     * @param theta angle in radians
     */
    void setPartialRefreshAngle(double theta) { partialRefreshAngle = theta; }
    double getPartialRefreshAngle() const { return partialRefreshAngle; }

    // ========== Stability Guard ==========

    /**
     * Set the stability guard threshold. HMC moves where BOTH
     * |deltaPE|/kT > threshold AND |deltaH|/kT > threshold are rejected.
     * This catches numerically unstable trajectories at large timesteps.
     *
     * @param threshold  dimensionless threshold (default 250.0, matching AlGDock)
     */
    void setStabilityThreshold(double threshold) { stabilityThreshold = threshold; }
    double getStabilityThreshold() const { return stabilityThreshold; }

    // ========== Accept/Reject Results ==========

    /**
     * Get whether a specific group was accepted in the last step.
     * @param group  group index
     * @return true if accepted
     */
    bool getGroupAccepted(int group) const;

    /**
     * Get accept/reject flags for all groups from the last step.
     * @return vector of flags (1=accepted, 0=rejected)
     */
    std::vector<int> getAllGroupAccepted() const;

    /**
     * Get the cumulative acceptance rate for a group.
     */
    double getGroupAcceptanceRate(int group) const;

    /**
     * Get the cumulative accept count for a group.
     */
    int getGroupAcceptCount(int group) const;

    /**
     * Get the cumulative trial count for a group.
     */
    int getGroupTrialCount(int group) const;

    /**
     * Get the cumulative stability-guard rejection count for a group.
     */
    int getGroupStabilityRejectCount(int group) const;

    /**
     * Get cumulative stability-guard rejection counts for all groups.
     * @return vector of length numGroups with per-group counts.
     */
    std::vector<int> getAllGroupStabilityRejectCounts() const;

    /**
     * Reset all acceptance/trial/stability counters to zero.
     */
    void resetAcceptanceCounts();

    // ========== External MC Configuration ==========

    /**
     * Set the number of rigid-body MC trials per step.
     * MC trials run as a pre-step before the HMC trajectory.
     * Set to 0 to disable MC (default).
     *
     * @param trials number of MC trials (0 = disabled)
     */
    void setNumMCTrials(int trials);
    int getNumMCTrials() const { return numMCTrials; }

    /**
     * Set the Gaussian translation step size for MC moves (in nm).
     * @param stepSize  sigma for Gaussian translation (default 0.025 nm)
     */
    void setMCStepSize(double stepSize);
    double getMCStepSize() const { return mcStepSize; }

    /**
     * Enable/disable MC moves for a specific group.
     * Only enabled groups receive MC trial moves.
     * @param group   group index
     * @param enabled true to enable MC for this group
     */
    void setGroupMCEnabled(int group, bool enabled);
    bool getGroupMCEnabled(int group) const;

    /**
     * Batch set MC eligibility for all groups.
     * @param enabled vector of length numGroups (1=enabled, 0=disabled)
     */
    void setAllGroupMCEnabled(const std::vector<int>& enabled);
    std::vector<int> getAllGroupMCEnabled() const { return groupMCEnabled; }

    // ========== MC Accept/Reject Results ==========

    /**
     * Get cumulative MC attempted count (across all groups and trials).
     */
    int getMCAttempted() const { return mcAttempted; }

    /**
     * Get cumulative MC accepted count (across all groups and trials).
     */
    int getMCAccepted() const { return mcAccepted; }

    /**
     * Get per-group MC acceptance from the last step.
     * Each element is the number of accepted trials for that group in the last step.
     */
    std::vector<int> getAllGroupMCAccepted() const { return lastMCAccepted; }

    /**
     * Reset cumulative MC counters to zero.
     */
    void resetMCCounts();

    // ========== Diagnostics ==========

    /**
     * Get deltaH for a specific group from the last step.
     * @param group  group index
     * @return deltaH in kJ/mol
     */
    double getGroupDeltaH(int group) const;

    /**
     * Get deltaH for all groups from the last step.
     * @return vector of deltaH values in kJ/mol
     */
    std::vector<double> getAllGroupDeltaH() const;

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
     * METRIC_UPDATE_EVERY_TRAJECTORY recomputes G once per HMC trial.
     */
    void setMetricUpdateMode(MetricUpdateMode mode) { metricUpdateMode = mode; }
    MetricUpdateMode getMetricUpdateMode() const { return metricUpdateMode; }

    /**
     * Set the SoftAbs sharpness parameter alpha. Default: 1e6.
     * Larger alpha -> softabs(lambda) approaches |lambda| (sharper).
     * Smaller alpha -> softabs(lambda) approaches 1/alpha (more uniform).
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

    // ========== Random Number Seed ==========

    /**
     * Get the random number seed.
     */
    int getRandomNumberSeed() const { return randomNumberSeed; }

    /**
     * Set the random number seed. If 0 (default), a unique seed is chosen
     * when the Context is created.
     */
    void setRandomNumberSeed(int seed) { randomNumberSeed = seed; }

    // ========== Integrator Interface ==========

    /**
     * Advance the simulation by taking a series of HMC trials.
     * Each step is one complete HMC trial (draw momenta, RESPA trajectory,
     * accept/reject per group).
     *
     * @param steps number of HMC trials to perform
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

    // Per-group temperatures (Kelvin)
    std::vector<double> groupTemperatures;

    // Per-group outer timesteps (ps)
    std::vector<double> groupStepSizes;

    // RESPA schedule: (forceGroupIndex, substeps) pairs
    std::vector<std::pair<int,int> > forceGroupSchedule;

    // HMC trajectory length (number of outer steps).
    // numOuterSteps is the maximum across groups; groupStepsPerTrial
    // contains the per-group values (matching AlGDock reference).
    int numOuterSteps;
    std::vector<int> groupStepsPerTrial;

    // Momentum refresh
    MomentumRefreshMode momentumRefreshMode;
    double partialRefreshAngle;

    // Stability guard threshold (dimensionless)
    double stabilityThreshold;

    // MC configuration
    int numMCTrials;                  // number of MC trials per step (0=disabled)
    double mcStepSize;                // Gaussian translation sigma in nm
    std::vector<int> groupMCEnabled;  // per-group MC eligibility (1=on, 0=off)
    int mcAttempted;                  // cumulative MC attempted count
    int mcAccepted;                   // cumulative MC accepted count
    std::vector<int> lastMCAccepted;  // per-group accepted from last step

    // Riemannian metric configuration
    MetricType metricType;
    MetricUpdateMode metricUpdateMode;
    double softAbsAlpha;
    double metricBlendFactor;
    double gridHessianWeight;

    // External diagonal Hessian (e.g., OBC solvation from JAX)
    std::vector<float> externalHessian;

    // Random seed
    int randomNumberSeed;

    // Acceptance statistics (host-side)
    std::vector<int> acceptCounts;
    std::vector<int> trialCounts;
    std::vector<int> stabilityRejectCounts;

    // Last step results (host-side)
    mutable std::vector<int> lastAccepted;
    mutable std::vector<double> lastDeltaH;

    // Kernel
    OpenMM::Kernel kernel;
    bool forcesAreValid;
};

}  // namespace GridForcePlugin

#endif /*OPENMM_MULTIGROUPHMCINTEGRATOR_H_*/
