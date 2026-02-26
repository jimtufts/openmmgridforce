/* -------------------------------------------------------------------------- *
 *                              OpenMMGridForce                               *
 * -------------------------------------------------------------------------- */

#include "MultiGroupNUTSIntegrator.h"
#include "MultiGroupNUTSKernels.h"
#include "openmm/Context.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/ContextImpl.h"
#include <cmath>
#include <string>

using namespace GridForcePlugin;
using namespace OpenMM;
using namespace std;

MultiGroupNUTSIntegrator::MultiGroupNUTSIntegrator(int numGroups, int atomsPerGroup, double stepSize)
    : numGroups(numGroups), atomsPerGroup(atomsPerGroup),
      maxTreeDepth(10),
      momentumRefreshMode(FULL), partialRefreshAngle(M_PI / 2.0),
      stabilityThreshold(200.0),
      gpuTreeBuilding(true),
      randomNumberSeed(0),
      forcesAreValid(false) {

    if (numGroups < 1)
        throw OpenMMException("MultiGroupNUTSIntegrator: numGroups must be >= 1");
    if (atomsPerGroup < 1)
        throw OpenMMException("MultiGroupNUTSIntegrator: atomsPerGroup must be >= 1");

    setStepSize(stepSize);

    groupTemperatures.resize(numGroups, 300.0);
    groupStepSizes.resize(numGroups, stepSize);

    acceptCounts.resize(numGroups, 0);
    trialCounts.resize(numGroups, 0);
    divergenceCounts.resize(numGroups, 0);
    cumulativeTreeDepths.resize(numGroups, 0);
    lastAccepted.resize(numGroups, 0);
    lastTreeDepths.resize(numGroups, 0);
    lastDivergent.resize(numGroups, 0);

    // MC defaults (disabled)
    numMCTrials = 0;
    mcStepSize = 0.025;
    groupMCEnabled.resize(numGroups, 0);
    mcAttempted = 0;
    mcAccepted = 0;
    lastMCAccepted.resize(numGroups, 0);
}

// ========== Per-Group Temperatures ==========

void MultiGroupNUTSIntegrator::setGroupTemperature(int group, double temperature) {
    if (group < 0 || group >= numGroups)
        throw OpenMMException("MultiGroupNUTSIntegrator: group index out of range");
    if (temperature <= 0)
        throw OpenMMException("MultiGroupNUTSIntegrator: temperature must be positive");
    groupTemperatures[group] = temperature;
}

double MultiGroupNUTSIntegrator::getGroupTemperature(int group) const {
    if (group < 0 || group >= numGroups)
        throw OpenMMException("MultiGroupNUTSIntegrator: group index out of range");
    return groupTemperatures[group];
}

void MultiGroupNUTSIntegrator::setAllGroupTemperatures(const vector<double>& temperatures) {
    if (static_cast<int>(temperatures.size()) != numGroups)
        throw OpenMMException("MultiGroupNUTSIntegrator: temperatures size must match numGroups");
    for (int i = 0; i < numGroups; i++)
        if (temperatures[i] <= 0)
            throw OpenMMException("MultiGroupNUTSIntegrator: all temperatures must be positive");
    groupTemperatures = temperatures;
}

// ========== RESPA Schedule ==========

void MultiGroupNUTSIntegrator::setForceGroupSchedule(const vector<pair<int,int> >& schedule) {
    for (size_t i = 0; i < schedule.size(); i++) {
        if (schedule[i].first < 0 || schedule[i].first > 31)
            throw OpenMMException("MultiGroupNUTSIntegrator: force group index must be 0-31");
        if (schedule[i].second < 1)
            throw OpenMMException("MultiGroupNUTSIntegrator: substeps must be >= 1");
    }
    forceGroupSchedule = schedule;
}

// ========== NUTS Tree Depth ==========

void MultiGroupNUTSIntegrator::setMaxTreeDepth(int depth) {
    if (depth < 1 || depth > 20)
        throw OpenMMException("MultiGroupNUTSIntegrator: maxTreeDepth must be 1-20");
    maxTreeDepth = depth;
}

// ========== Per-Group Timestep ==========

void MultiGroupNUTSIntegrator::setGroupStepSize(int group, double stepSize) {
    if (group < 0 || group >= numGroups)
        throw OpenMMException("MultiGroupNUTSIntegrator: group index out of range");
    if (stepSize <= 0)
        throw OpenMMException("MultiGroupNUTSIntegrator: stepSize must be positive");
    groupStepSizes[group] = stepSize;
}

double MultiGroupNUTSIntegrator::getGroupStepSize(int group) const {
    if (group < 0 || group >= numGroups)
        throw OpenMMException("MultiGroupNUTSIntegrator: group index out of range");
    return groupStepSizes[group];
}

void MultiGroupNUTSIntegrator::setAllGroupStepSizes(const vector<double>& stepSizes) {
    if (static_cast<int>(stepSizes.size()) != numGroups)
        throw OpenMMException("MultiGroupNUTSIntegrator: stepSizes size must match numGroups");
    for (int i = 0; i < numGroups; i++)
        if (stepSizes[i] <= 0)
            throw OpenMMException("MultiGroupNUTSIntegrator: all stepSizes must be positive");
    groupStepSizes = stepSizes;
}

// ========== Tree Depth Diagnostics ==========

int MultiGroupNUTSIntegrator::getGroupTreeDepth(int group) const {
    if (group < 0 || group >= numGroups)
        throw OpenMMException("MultiGroupNUTSIntegrator: group index out of range");
    return lastTreeDepths[group];
}

vector<int> MultiGroupNUTSIntegrator::getAllGroupTreeDepths() const {
    return lastTreeDepths;
}

double MultiGroupNUTSIntegrator::getGroupMeanTreeDepth(int group) const {
    if (group < 0 || group >= numGroups)
        throw OpenMMException("MultiGroupNUTSIntegrator: group index out of range");
    if (trialCounts[group] == 0) return 0.0;
    return static_cast<double>(cumulativeTreeDepths[group]) / trialCounts[group];
}

// ========== Divergence Diagnostics ==========

bool MultiGroupNUTSIntegrator::getGroupDivergent(int group) const {
    if (group < 0 || group >= numGroups)
        throw OpenMMException("MultiGroupNUTSIntegrator: group index out of range");
    return lastDivergent[group] != 0;
}

vector<int> MultiGroupNUTSIntegrator::getAllGroupDivergent() const {
    return lastDivergent;
}

// ========== Accept/Reject Results ==========

bool MultiGroupNUTSIntegrator::getGroupAccepted(int group) const {
    if (group < 0 || group >= numGroups)
        throw OpenMMException("MultiGroupNUTSIntegrator: group index out of range");
    return lastAccepted[group] != 0;
}

vector<int> MultiGroupNUTSIntegrator::getAllGroupAccepted() const {
    return lastAccepted;
}

double MultiGroupNUTSIntegrator::getGroupAcceptanceRate(int group) const {
    if (group < 0 || group >= numGroups)
        throw OpenMMException("MultiGroupNUTSIntegrator: group index out of range");
    if (trialCounts[group] == 0) return 0.0;
    return static_cast<double>(acceptCounts[group]) / trialCounts[group];
}

int MultiGroupNUTSIntegrator::getGroupAcceptCount(int group) const {
    if (group < 0 || group >= numGroups)
        throw OpenMMException("MultiGroupNUTSIntegrator: group index out of range");
    return acceptCounts[group];
}

int MultiGroupNUTSIntegrator::getGroupTrialCount(int group) const {
    if (group < 0 || group >= numGroups)
        throw OpenMMException("MultiGroupNUTSIntegrator: group index out of range");
    return trialCounts[group];
}

int MultiGroupNUTSIntegrator::getGroupDivergenceCount(int group) const {
    if (group < 0 || group >= numGroups)
        throw OpenMMException("MultiGroupNUTSIntegrator: group index out of range");
    return divergenceCounts[group];
}

vector<int> MultiGroupNUTSIntegrator::getAllGroupDivergenceCounts() const {
    return divergenceCounts;
}

void MultiGroupNUTSIntegrator::resetAcceptanceCounts() {
    fill(acceptCounts.begin(), acceptCounts.end(), 0);
    fill(trialCounts.begin(), trialCounts.end(), 0);
    fill(divergenceCounts.begin(), divergenceCounts.end(), 0);
    fill(cumulativeTreeDepths.begin(), cumulativeTreeDepths.end(), 0LL);
    if (context != NULL) {
        kernel.getAs<IntegrateMultiGroupNUTSStepKernel>().resetCounters();
    }
}

// ========== External MC ==========

void MultiGroupNUTSIntegrator::setNumMCTrials(int trials) {
    if (trials < 0)
        throw OpenMMException("MultiGroupNUTSIntegrator: numMCTrials must be >= 0");
    numMCTrials = trials;
}

void MultiGroupNUTSIntegrator::setMCStepSize(double stepSize) {
    if (stepSize <= 0)
        throw OpenMMException("MultiGroupNUTSIntegrator: mcStepSize must be positive");
    mcStepSize = stepSize;
}

void MultiGroupNUTSIntegrator::setGroupMCEnabled(int group, bool enabled) {
    if (group < 0 || group >= numGroups)
        throw OpenMMException("MultiGroupNUTSIntegrator: group index out of range");
    groupMCEnabled[group] = enabled ? 1 : 0;
}

bool MultiGroupNUTSIntegrator::getGroupMCEnabled(int group) const {
    if (group < 0 || group >= numGroups)
        throw OpenMMException("MultiGroupNUTSIntegrator: group index out of range");
    return groupMCEnabled[group] != 0;
}

void MultiGroupNUTSIntegrator::setAllGroupMCEnabled(const vector<int>& enabled) {
    if (static_cast<int>(enabled.size()) != numGroups)
        throw OpenMMException("MultiGroupNUTSIntegrator: enabled size must match numGroups");
    groupMCEnabled = enabled;
}

void MultiGroupNUTSIntegrator::resetMCCounts() {
    mcAttempted = 0;
    mcAccepted = 0;
    fill(lastMCAccepted.begin(), lastMCAccepted.end(), 0);
    if (context != NULL) {
        kernel.getAs<IntegrateMultiGroupNUTSStepKernel>().resetMCCounters();
    }
}

// ========== Integrator Interface ==========

void MultiGroupNUTSIntegrator::initialize(ContextImpl& contextRef) {
    if (owner != NULL && &contextRef.getOwner() != owner)
        throw OpenMMException("This Integrator is already bound to a context");

    int totalAtoms = numGroups * atomsPerGroup;
    if (contextRef.getSystem().getNumParticles() < totalAtoms)
        throw OpenMMException("MultiGroupNUTSIntegrator: System has fewer particles than "
                             "numGroups * atomsPerGroup");

    context = &contextRef;
    owner = &contextRef.getOwner();
    kernel = context->getPlatform().createKernel(IntegrateMultiGroupNUTSStepKernel::Name(), contextRef);
    kernel.getAs<IntegrateMultiGroupNUTSStepKernel>().initialize(contextRef.getSystem(), *this);
}

void MultiGroupNUTSIntegrator::cleanup() {
    kernel = Kernel();
}

void MultiGroupNUTSIntegrator::stateChanged(State::DataType changed) {
    forcesAreValid = false;
}

vector<string> MultiGroupNUTSIntegrator::getKernelNames() {
    vector<string> names;
    names.push_back(IntegrateMultiGroupNUTSStepKernel::Name());
    return names;
}

double MultiGroupNUTSIntegrator::computeKineticEnergy() {
    return kernel.getAs<IntegrateMultiGroupNUTSStepKernel>().computeKineticEnergy(*context, *this);
}

void MultiGroupNUTSIntegrator::step(int steps) {
    if (context == NULL)
        throw OpenMMException("This Integrator is not bound to a context!");
    for (int i = 0; i < steps; ++i) {
        kernel.getAs<IntegrateMultiGroupNUTSStepKernel>().execute(*context, *this, forcesAreValid);
        forcesAreValid = false;  // NUTS always changes positions

        // Read back results from kernel
        lastAccepted = kernel.getAs<IntegrateMultiGroupNUTSStepKernel>().getAcceptedFlags();
        lastTreeDepths = kernel.getAs<IntegrateMultiGroupNUTSStepKernel>().getTreeDepths();
        lastDivergent = kernel.getAs<IntegrateMultiGroupNUTSStepKernel>().getDivergentFlags();

        // Update cumulative statistics from kernel
        vector<int> kernelAccept = kernel.getAs<IntegrateMultiGroupNUTSStepKernel>().getAcceptCounts();
        vector<int> kernelTrials = kernel.getAs<IntegrateMultiGroupNUTSStepKernel>().getTrialCounts();
        vector<int> kernelDivergence = kernel.getAs<IntegrateMultiGroupNUTSStepKernel>().getDivergenceCounts();
        vector<long long> kernelTreeDepths = kernel.getAs<IntegrateMultiGroupNUTSStepKernel>().getCumulativeTreeDepths();
        for (int k = 0; k < numGroups; k++) {
            acceptCounts[k] = kernelAccept[k];
            trialCounts[k] = kernelTrials[k];
            divergenceCounts[k] = kernelDivergence[k];
            cumulativeTreeDepths[k] = kernelTreeDepths[k];
        }

        // Read back MC statistics
        mcAttempted = kernel.getAs<IntegrateMultiGroupNUTSStepKernel>().getMCAttempted();
        mcAccepted = kernel.getAs<IntegrateMultiGroupNUTSStepKernel>().getMCAccepted();
        lastMCAccepted = kernel.getAs<IntegrateMultiGroupNUTSStepKernel>().getLastMCAccepted();
    }
}
