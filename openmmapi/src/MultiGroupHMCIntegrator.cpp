/* -------------------------------------------------------------------------- *
 *                              OpenMMGridForce                               *
 * -------------------------------------------------------------------------- */

#include "MultiGroupHMCIntegrator.h"
#include "MultiGroupHMCKernels.h"
#include "openmm/Context.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/ContextImpl.h"
#include <cmath>
#include <string>

using namespace GridForcePlugin;
using namespace OpenMM;
using namespace std;

MultiGroupHMCIntegrator::MultiGroupHMCIntegrator(int numGroups, int atomsPerGroup, double stepSize)
    : numGroups(numGroups), atomsPerGroup(atomsPerGroup),
      numOuterSteps(25),
      momentumRefreshMode(FULL), partialRefreshAngle(M_PI / 2.0),
      stabilityThreshold(250.0),
      metricType(METRIC_IDENTITY),
      metricUpdateMode(METRIC_UPDATE_NONE),
      softAbsAlpha(1e6),
      metricBlendFactor(1.0),
      gridHessianWeight(0.0),
      randomNumberSeed(0),
      forcesAreValid(false) {

    if (numGroups < 1)
        throw OpenMMException("MultiGroupHMCIntegrator: numGroups must be >= 1");
    if (atomsPerGroup < 1)
        throw OpenMMException("MultiGroupHMCIntegrator: atomsPerGroup must be >= 1");

    setStepSize(stepSize);

    // Initialize per-group arrays with defaults
    groupTemperatures.resize(numGroups, 300.0);   // 300 K default
    groupStepSizes.resize(numGroups, stepSize);    // uniform initial dt

    acceptCounts.resize(numGroups, 0);
    trialCounts.resize(numGroups, 0);
    stabilityRejectCounts.resize(numGroups, 0);
    lastAccepted.resize(numGroups, 0);
    lastDeltaH.resize(numGroups, 0.0);

    // MC defaults (disabled)
    numMCTrials = 0;
    mcStepSize = 0.025;  // 0.025 nm default
    groupMCEnabled.resize(numGroups, 0);
    mcAttempted = 0;
    mcAccepted = 0;
    lastMCAccepted.resize(numGroups, 0);
}

// ========== Per-Group Temperatures ==========

void MultiGroupHMCIntegrator::setGroupTemperature(int group, double temperature) {
    if (group < 0 || group >= numGroups)
        throw OpenMMException("MultiGroupHMCIntegrator: group index out of range");
    if (temperature <= 0)
        throw OpenMMException("MultiGroupHMCIntegrator: temperature must be positive");
    groupTemperatures[group] = temperature;
}

double MultiGroupHMCIntegrator::getGroupTemperature(int group) const {
    if (group < 0 || group >= numGroups)
        throw OpenMMException("MultiGroupHMCIntegrator: group index out of range");
    return groupTemperatures[group];
}

void MultiGroupHMCIntegrator::setAllGroupTemperatures(const vector<double>& temperatures) {
    if (static_cast<int>(temperatures.size()) != numGroups)
        throw OpenMMException("MultiGroupHMCIntegrator: temperatures size must match numGroups");
    for (int i = 0; i < numGroups; i++)
        if (temperatures[i] <= 0)
            throw OpenMMException("MultiGroupHMCIntegrator: all temperatures must be positive");
    groupTemperatures = temperatures;
}

// ========== RESPA Schedule ==========

void MultiGroupHMCIntegrator::setForceGroupSchedule(const vector<pair<int,int> >& schedule) {
    for (size_t i = 0; i < schedule.size(); i++) {
        if (schedule[i].first < 0 || schedule[i].first > 31)
            throw OpenMMException("MultiGroupHMCIntegrator: force group index must be 0-31");
        if (schedule[i].second < 1)
            throw OpenMMException("MultiGroupHMCIntegrator: substeps must be >= 1");
    }
    forceGroupSchedule = schedule;
}

// ========== HMC Trajectory Length ==========

void MultiGroupHMCIntegrator::setNumOuterSteps(int steps) {
    if (steps < 1)
        throw OpenMMException("MultiGroupHMCIntegrator: numOuterSteps must be >= 1");
    numOuterSteps = steps;
}

// ========== Per-Group Timestep ==========

void MultiGroupHMCIntegrator::setGroupStepSize(int group, double stepSize) {
    if (group < 0 || group >= numGroups)
        throw OpenMMException("MultiGroupHMCIntegrator: group index out of range");
    if (stepSize <= 0)
        throw OpenMMException("MultiGroupHMCIntegrator: stepSize must be positive");
    groupStepSizes[group] = stepSize;
}

double MultiGroupHMCIntegrator::getGroupStepSize(int group) const {
    if (group < 0 || group >= numGroups)
        throw OpenMMException("MultiGroupHMCIntegrator: group index out of range");
    return groupStepSizes[group];
}

void MultiGroupHMCIntegrator::setAllGroupStepSizes(const vector<double>& stepSizes) {
    if (static_cast<int>(stepSizes.size()) != numGroups)
        throw OpenMMException("MultiGroupHMCIntegrator: stepSizes size must match numGroups");
    for (int i = 0; i < numGroups; i++)
        if (stepSizes[i] <= 0)
            throw OpenMMException("MultiGroupHMCIntegrator: all stepSizes must be positive");
    groupStepSizes = stepSizes;
}

// ========== Accept/Reject Results ==========

bool MultiGroupHMCIntegrator::getGroupAccepted(int group) const {
    if (group < 0 || group >= numGroups)
        throw OpenMMException("MultiGroupHMCIntegrator: group index out of range");
    return lastAccepted[group] != 0;
}

vector<int> MultiGroupHMCIntegrator::getAllGroupAccepted() const {
    return lastAccepted;
}

double MultiGroupHMCIntegrator::getGroupAcceptanceRate(int group) const {
    if (group < 0 || group >= numGroups)
        throw OpenMMException("MultiGroupHMCIntegrator: group index out of range");
    if (trialCounts[group] == 0) return 0.0;
    return static_cast<double>(acceptCounts[group]) / trialCounts[group];
}

int MultiGroupHMCIntegrator::getGroupAcceptCount(int group) const {
    if (group < 0 || group >= numGroups)
        throw OpenMMException("MultiGroupHMCIntegrator: group index out of range");
    return acceptCounts[group];
}

int MultiGroupHMCIntegrator::getGroupTrialCount(int group) const {
    if (group < 0 || group >= numGroups)
        throw OpenMMException("MultiGroupHMCIntegrator: group index out of range");
    return trialCounts[group];
}

int MultiGroupHMCIntegrator::getGroupStabilityRejectCount(int group) const {
    if (group < 0 || group >= numGroups)
        throw OpenMMException("MultiGroupHMCIntegrator: group index out of range");
    return stabilityRejectCounts[group];
}

std::vector<int> MultiGroupHMCIntegrator::getAllGroupStabilityRejectCounts() const {
    return stabilityRejectCounts;
}

void MultiGroupHMCIntegrator::resetAcceptanceCounts() {
    fill(acceptCounts.begin(), acceptCounts.end(), 0);
    fill(trialCounts.begin(), trialCounts.end(), 0);
    fill(stabilityRejectCounts.begin(), stabilityRejectCounts.end(), 0);
    if (context != NULL) {
        kernel.getAs<IntegrateMultiGroupHMCStepKernel>().resetCounters();
    }
}

// ========== External MC ==========

void MultiGroupHMCIntegrator::setNumMCTrials(int trials) {
    if (trials < 0)
        throw OpenMMException("MultiGroupHMCIntegrator: numMCTrials must be >= 0");
    numMCTrials = trials;
}

void MultiGroupHMCIntegrator::setMCStepSize(double stepSize) {
    if (stepSize <= 0)
        throw OpenMMException("MultiGroupHMCIntegrator: mcStepSize must be positive");
    mcStepSize = stepSize;
}

void MultiGroupHMCIntegrator::setGroupMCEnabled(int group, bool enabled) {
    if (group < 0 || group >= numGroups)
        throw OpenMMException("MultiGroupHMCIntegrator: group index out of range");
    groupMCEnabled[group] = enabled ? 1 : 0;
}

bool MultiGroupHMCIntegrator::getGroupMCEnabled(int group) const {
    if (group < 0 || group >= numGroups)
        throw OpenMMException("MultiGroupHMCIntegrator: group index out of range");
    return groupMCEnabled[group] != 0;
}

void MultiGroupHMCIntegrator::setAllGroupMCEnabled(const vector<int>& enabled) {
    if (static_cast<int>(enabled.size()) != numGroups)
        throw OpenMMException("MultiGroupHMCIntegrator: enabled size must match numGroups");
    groupMCEnabled = enabled;
}

void MultiGroupHMCIntegrator::resetMCCounts() {
    mcAttempted = 0;
    mcAccepted = 0;
    fill(lastMCAccepted.begin(), lastMCAccepted.end(), 0);
    if (context != NULL) {
        kernel.getAs<IntegrateMultiGroupHMCStepKernel>().resetMCCounters();
    }
}

// ========== Riemannian Metric ==========

void MultiGroupHMCIntegrator::setSoftAbsAlpha(double alpha) {
    if (alpha <= 0)
        throw OpenMMException("MultiGroupHMCIntegrator: softAbsAlpha must be positive");
    softAbsAlpha = alpha;
}

void MultiGroupHMCIntegrator::setMetricBlendFactor(double beta) {
    if (beta < 0.0 || beta > 1.0)
        throw OpenMMException("MultiGroupHMCIntegrator: metricBlendFactor must be in [0, 1]");
    metricBlendFactor = beta;
}

vector<double> MultiGroupHMCIntegrator::getGroupMetricConditionNumbers() const {
    if (context == NULL)
        return vector<double>(numGroups, 1.0);
    return kernel.getAs<IntegrateMultiGroupHMCStepKernel>().getGroupMetricConditionNumbers();
}

// ========== External Diagonal Hessian ==========

void MultiGroupHMCIntegrator::setExternalDiagonalHessian(const vector<float>& hessian) {
    int expected = 6 * numGroups * atomsPerGroup;
    if (static_cast<int>(hessian.size()) != expected)
        throw OpenMMException("MultiGroupHMCIntegrator: externalHessian size must be 6*numGroups*atomsPerGroup ("
                              + to_string(expected) + "), got " + to_string(hessian.size()));
    externalHessian = hessian;
}

// ========== Diagnostics ==========

double MultiGroupHMCIntegrator::getGroupDeltaH(int group) const {
    if (group < 0 || group >= numGroups)
        throw OpenMMException("MultiGroupHMCIntegrator: group index out of range");
    return lastDeltaH[group];
}

vector<double> MultiGroupHMCIntegrator::getAllGroupDeltaH() const {
    return lastDeltaH;
}

// ========== Integrator Interface ==========

void MultiGroupHMCIntegrator::initialize(ContextImpl& contextRef) {
    if (owner != NULL && &contextRef.getOwner() != owner)
        throw OpenMMException("This Integrator is already bound to a context");

    // Verify the system has enough particles
    int totalAtoms = numGroups * atomsPerGroup;
    if (contextRef.getSystem().getNumParticles() < totalAtoms)
        throw OpenMMException("MultiGroupHMCIntegrator: System has fewer particles than "
                             "numGroups * atomsPerGroup");

    context = &contextRef;
    owner = &contextRef.getOwner();
    kernel = context->getPlatform().createKernel(IntegrateMultiGroupHMCStepKernel::Name(), contextRef);
    kernel.getAs<IntegrateMultiGroupHMCStepKernel>().initialize(contextRef.getSystem(), *this);
}

void MultiGroupHMCIntegrator::cleanup() {
    kernel = Kernel();
}

void MultiGroupHMCIntegrator::stateChanged(State::DataType changed) {
    forcesAreValid = false;
}

vector<string> MultiGroupHMCIntegrator::getKernelNames() {
    vector<string> names;
    names.push_back(IntegrateMultiGroupHMCStepKernel::Name());
    return names;
}

double MultiGroupHMCIntegrator::computeKineticEnergy() {
    return kernel.getAs<IntegrateMultiGroupHMCStepKernel>().computeKineticEnergy(*context, *this);
}

void MultiGroupHMCIntegrator::step(int steps) {
    if (context == NULL)
        throw OpenMMException("This Integrator is not bound to a context!");
    for (int i = 0; i < steps; ++i) {
        kernel.getAs<IntegrateMultiGroupHMCStepKernel>().execute(*context, *this, forcesAreValid);
        forcesAreValid = true;

        // Read back results from kernel
        lastAccepted = kernel.getAs<IntegrateMultiGroupHMCStepKernel>().getAcceptedFlags();
        lastDeltaH = kernel.getAs<IntegrateMultiGroupHMCStepKernel>().getDeltaH();

        // Update cumulative statistics from kernel
        vector<int> kernelAccept = kernel.getAs<IntegrateMultiGroupHMCStepKernel>().getAcceptCounts();
        vector<int> kernelTrials = kernel.getAs<IntegrateMultiGroupHMCStepKernel>().getTrialCounts();
        vector<int> kernelStabReject = kernel.getAs<IntegrateMultiGroupHMCStepKernel>().getStabilityRejectCounts();
        for (int k = 0; k < numGroups; k++) {
            acceptCounts[k] = kernelAccept[k];
            trialCounts[k] = kernelTrials[k];
            stabilityRejectCounts[k] = kernelStabReject[k];
        }

        // Read back MC statistics
        mcAttempted = kernel.getAs<IntegrateMultiGroupHMCStepKernel>().getMCAttempted();
        mcAccepted = kernel.getAs<IntegrateMultiGroupHMCStepKernel>().getMCAccepted();
        lastMCAccepted = kernel.getAs<IntegrateMultiGroupHMCStepKernel>().getLastMCAccepted();

        // If any group was rejected, forces may be stale (positions restored)
        for (int k = 0; k < numGroups; k++) {
            if (lastAccepted[k] == 0) {
                forcesAreValid = false;
                break;
            }
        }
    }
}
