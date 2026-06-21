/* -------------------------------------------------------------------------- *
 *                              OpenMMGridForce                               *
 * -------------------------------------------------------------------------- *
 * Reference (CPU) implementation of MultiGroupHMCIntegrator kernel.         *
 * -------------------------------------------------------------------------- */

#include "ReferenceMultiGroupHMCKernels.h"
#include "ReferenceGridInterpolation.h"
#include "GridForce.h"
#include "IsolatedBondedForce.h"
#include "IsolatedNonbondedForce.h"
#include "IsolatedGBSAForce.h"
#include "IsolatedSiteForce.h"
#include "GBSAGridForce.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/OpenMMException.h"
#include "openmm/reference/ReferencePlatform.h"
#include "openmm/reference/SimTKOpenMMRealType.h"
#include <cmath>
#include <algorithm>

using namespace OpenMM;
using namespace std;

namespace GridForcePlugin {

static vector<Vec3>& refExtractVelocities(ContextImpl& context) {
    ReferencePlatform::PlatformData* data =
        reinterpret_cast<ReferencePlatform::PlatformData*>(context.getPlatformData());
    return *((vector<Vec3>*)data->velocities);
}

// Uniform random rotation matrix (row-major 3x3) via Shoemake's quaternion method.
static void generateRandomQuaternionRotation(
        mt19937& rng, uniform_real_distribution<double>& uDist, double* R) {
    double u0 = uDist(rng), u1 = uDist(rng), u2 = uDist(rng);
    double s0 = sqrt(1.0 - u0), s1 = sqrt(u0);
    double t1 = 2.0 * M_PI * u1, t2 = 2.0 * M_PI * u2;
    double q0 = s0 * sin(t1), q1 = s0 * cos(t1);
    double q2 = s1 * sin(t2), q3 = s1 * cos(t2);
    R[0] = q0*q0 + q1*q1 - q2*q2 - q3*q3;
    R[1] = 2*(q1*q2 - q0*q3);
    R[2] = 2*(q1*q3 + q0*q2);
    R[3] = 2*(q1*q2 + q0*q3);
    R[4] = q0*q0 - q1*q1 + q2*q2 - q3*q3;
    R[5] = 2*(q2*q3 - q0*q1);
    R[6] = 2*(q1*q3 - q0*q2);
    R[7] = 2*(q2*q3 + q0*q1);
    R[8] = q0*q0 - q1*q1 - q2*q2 + q3*q3;
}

void ReferenceIntegrateMultiGroupHMCStepKernel::initialize(
        const System& system, const MultiGroupHMCIntegrator& integrator) {

    numGroups = integrator.getNumGroups();
    atomsPerGroup = integrator.getAtomsPerGroup();
    numParticles = system.getNumParticles();

    masses.resize(numParticles);
    for (int i = 0; i < numParticles; i++)
        masses[i] = system.getParticleMass(i);

    positionsBackup.resize(numParticles);
    lastAccepted.resize(numGroups, 0);
    lastDeltaH.resize(numGroups, 0.0);
    acceptCounts.resize(numGroups, 0);
    trialCounts.resize(numGroups, 0);
    stabilityRejectCounts.resize(numGroups, 0);
    lastMCAcceptedPerGroup.resize(numGroups, 0);
    mcAttemptedTotal = 0;
    mcAcceptedTotal = 0;

    unsigned int seed = (unsigned int)integrator.getRandomNumberSeed();
    if (seed == 0) {
        random_device rd;
        seed = rd();
    }
    groupRng.resize(numGroups);
    groupNormal.assign(numGroups, normal_distribution<double>(0.0, 1.0));
    groupUniform.assign(numGroups, uniform_real_distribution<double>(0.0, 1.0));
    for (int k = 0; k < numGroups; k++) {
        seed_seq seq{seed, (unsigned int)(k + 1)};
        groupRng[k].seed(seq);
    }

    // Scan system forces and register per-group energy extractors.
    // Each plugin force type has getParticleGroupEnergies() that returns
    // cached per-group energies from the last calcForcesAndEnergy() call.
    int K = numGroups;
    groupEnergyExtractors.clear();
    for (int i = 0; i < system.getNumForces(); i++) {
        const Force& force = system.getForce(i);

        // Try each plugin force type
        if (auto* f = dynamic_cast<const GridForce*>(&force)) {
            if (f->getNumParticleGroups() > 0) {
                // GridForce::getParticleGroupEnergies requires a Context, but the
                // m_groupEnergies are populated by the ForceImpl after each
                // calcForcesAndEnergy call. We use the kernel-level accessor instead.
                // Since we can't call Force methods that need Context from here,
                // we'll use the ForceImpl approach in computeGroupPE instead.
            }
        }
        // The extractors will be wired up in the first execute() call instead,
        // where we have access to the ContextImpl and its ForceImpls.
    }
}

void ReferenceIntegrateMultiGroupHMCStepKernel::forEachGroup(
        ContextImpl& context, const std::function<void(int)>& body) {
    for (int k = 0; k < numGroups; k++)
        body(k);
}

void ReferenceIntegrateMultiGroupHMCStepKernel::computeGroupPE(
        vector<double>& groupPE) const {
    // Sum per-group energies from all registered extractors
    fill(groupPE.begin(), groupPE.end(), 0.0);
    for (auto& extractor : groupEnergyExtractors) {
        vector<double> energies = extractor();
        for (int k = 0; k < numGroups && k < (int)energies.size(); k++)
            groupPE[k] += energies[k];
    }
}

void ReferenceIntegrateMultiGroupHMCStepKernel::execute(
        ContextImpl& context, const MultiGroupHMCIntegrator& integrator,
        bool forcesAreValid) {

    vector<Vec3>& posData = refExtractPositions(context);
    vector<Vec3>& velData = refExtractVelocities(context);
    int K = numGroups;

    // On first call, wire up the group energy extractors using ForceImpls.
    // After calcForcesAndEnergy(), each ForceImpl populates its Force object's
    // m_groupEnergies cache. We read those via getParticleGroupEnergies().
    if (groupEnergyExtractors.empty()) {
        const System& system = context.getSystem();
        for (int i = 0; i < system.getNumForces(); i++) {
            const Force& force = system.getForce(i);

            if (auto* f = dynamic_cast<const IsolatedBondedForce*>(&force)) {
                if (f->getNumParticleGroups() > 0)
                    groupEnergyExtractors.push_back([f]() { return f->getParticleGroupEnergies(); });
            }
            else if (auto* f = dynamic_cast<const IsolatedNonbondedForce*>(&force)) {
                if (f->getNumParticleGroups() > 0)
                    groupEnergyExtractors.push_back([f]() { return f->getParticleGroupEnergies(); });
            }
            else if (auto* f = dynamic_cast<const IsolatedGBSAForce*>(&force)) {
                if (f->getNumParticleGroups() > 0)
                    groupEnergyExtractors.push_back([f]() { return f->getParticleGroupEnergies(); });
            }
            else if (auto* f = dynamic_cast<const IsolatedSiteForce*>(&force)) {
                if (f->getNumParticleGroups() > 0)
                    groupEnergyExtractors.push_back([f]() { return f->getParticleGroupEnergies(); });
            }
            else if (auto* f = dynamic_cast<const GBSAGridForce*>(&force)) {
                if (f->getNumParticleGroups() > 0) {
                    groupEnergyExtractors.push_back([f, K]() {
                        vector<double> energies(K);
                        for (int k = 0; k < K; k++)
                            energies[k] = f->getGroupEnergy(k);
                        return energies;
                    });
                }
            }
            else if (auto* f = dynamic_cast<const GridForce*>(&force)) {
                // GridForce::getParticleGroupEnergies requires Context.
                // The kernel populates per-group energies during execute().
                // After calcForcesAndEnergy, we can read them from the kernel
                // via the ForceImpl. Use a different approach: read from the
                // Force object's cached energies.
                // GridForce stores per-group energies in the kernel, accessible
                // via GridForceImpl::getParticleGroupEnergies(). We need the
                // context to call it. Store a reference to ContextImpl.
                if (f->getNumParticleGroups() > 0) {
                    // GridForce needs Context for getParticleGroupEnergies.
                    // We capture a pointer to the ContextImpl's owner (Context)
                    // and call the Force method that routes through the Impl.
                    Context* ctx = &context.getOwner();
                    const GridForce* gf = f;
                    groupEnergyExtractors.push_back([gf, ctx]() {
                        return gf->getParticleGroupEnergies(*ctx);
                    });
                }
            }
        }
    }

    // Check metric type — Reference platform only supports IDENTITY
    if (integrator.getMetricType() != MultiGroupHMCIntegrator::METRIC_IDENTITY)
        throw OpenMMException("ReferenceMultiGroupHMCKernel: Riemannian metric (non-IDENTITY) "
                              "is only supported on the CUDA platform.");

    // Rigid-body MC pre-step. May change positions, invalidating forces.
    if (integrator.getNumMCTrials() > 0)
        executeMC(context, integrator);

    // ===== 1. Backup positions =====
    for (int i = 0; i < numParticles; i++)
        positionsBackup[i] = posData[i];

    // ===== 2. Draw Maxwell-Boltzmann velocities per group =====
    forEachGroup(context, [&](int k) {
        double T = integrator.getGroupTemperature(k);
        double kT = BOLTZ * T;
        int baseAtom = k * atomsPerGroup;
        auto& rngk = groupRng[k];
        auto& nd = groupNormal[k];

        if (integrator.getMomentumRefreshMode() == MultiGroupHMCIntegrator::FULL) {
            for (int a = 0; a < atomsPerGroup; a++) {
                int idx = baseAtom + a;
                double m = masses[idx];
                if (m <= 0.0) continue;
                double sigma = sqrt(kT / m);
                velData[idx] = Vec3(sigma * nd(rngk),
                                    sigma * nd(rngk),
                                    sigma * nd(rngk));
            }
        } else {
            double theta = integrator.getPartialRefreshAngle();
            double cosTheta = cos(theta);
            double sinTheta = sin(theta);
            for (int a = 0; a < atomsPerGroup; a++) {
                int idx = baseAtom + a;
                double m = masses[idx];
                if (m <= 0.0) continue;
                double sigma = sqrt(kT / m);
                Vec3 vRand(sigma * nd(rngk),
                           sigma * nd(rngk),
                           sigma * nd(rngk));
                velData[idx] = velData[idx] * cosTheta + vRand * sinTheta;
            }
        }
    });

    // ===== 3. Compute initial KE and PE per group =====
    vector<double> keOld(K);
    computeGroupKE(context, keOld);

    // Evaluate all forces to get initial PE
    int allGroupsMask = 0xFFFFFFFF;
    context.calcForcesAndEnergy(true, true, allGroupsMask);

    vector<double> peOld(K, 0.0);
    computeGroupPE(peOld);

    // ===== 4. RESPA NVE trajectory =====
    respaTrajectory(context, integrator);

    // ===== 5. Compute final KE and PE per group =====
    vector<double> keNew(K);
    computeGroupKE(context, keNew);

    context.calcForcesAndEnergy(true, true, allGroupsMask);

    vector<double> peNew(K, 0.0);
    computeGroupPE(peNew);

    // ===== 6. Metropolis accept/reject per group =====
    double stabilityThreshold = integrator.getStabilityThreshold();

    forEachGroup(context, [&](int k) {
        double T = integrator.getGroupTemperature(k);
        double kT = BOLTZ * T;

        double deltaPE = peNew[k] - peOld[k];
        double deltaKE = keNew[k] - keOld[k];
        double deltaH = deltaPE + deltaKE;

        lastDeltaH[k] = deltaH;
        trialCounts[k]++;

        // Stability guard
        bool stable = (fabs(deltaPE) / kT < stabilityThreshold) ||
                      (fabs(deltaH) / kT < stabilityThreshold);
        if (!stable) {
            lastAccepted[k] = 0;
            stabilityRejectCounts[k]++;
            return;
        }

        // Standard Metropolis
        bool accept = (deltaH <= 0.0) || (groupUniform[k](groupRng[k]) < exp(-deltaH / kT));
        lastAccepted[k] = accept ? 1 : 0;
        if (accept)
            acceptCounts[k]++;
    });

    // ===== 7. Restore positions for rejected groups =====
    forEachGroup(context, [&](int k) {
        if (lastAccepted[k] == 0) {
            int baseAtom = k * atomsPerGroup;
            for (int a = 0; a < atomsPerGroup; a++) {
                int idx = baseAtom + a;
                posData[idx] = positionsBackup[idx];
                velData[idx] = Vec3(0, 0, 0);
            }
        }
    });
}

void ReferenceIntegrateMultiGroupHMCStepKernel::respaTrajectory(
        ContextImpl& context, const MultiGroupHMCIntegrator& integrator) {

    vector<Vec3>& posData = refExtractPositions(context);
    vector<Vec3>& velData = refExtractVelocities(context);

    int K = numGroups;
    int numOuterSteps = integrator.getNumOuterSteps();
    const vector<pair<int,int> >& schedule = integrator.getForceGroupSchedule();

    if (schedule.empty()) {
        // Simple Verlet (no RESPA) with per-group dt
        int allGroupsMask = 0xFFFFFFFF;

        for (int step = 0; step < numOuterSteps; step++) {
            // Half-kick
            {
                vector<Vec3>& forceData = refExtractForces(context);
                forEachGroup(context, [&](int k) {
                    double halfDt = 0.5 * integrator.getGroupStepSize(k);
                    int base = k * atomsPerGroup;
                    for (int a = 0; a < atomsPerGroup; a++) {
                        int idx = base + a;
                        if (masses[idx] > 0)
                            velData[idx] += forceData[idx] * (halfDt / masses[idx]);
                    }
                });
            }

            // Drift
            forEachGroup(context, [&](int k) {
                double dt = integrator.getGroupStepSize(k);
                int base = k * atomsPerGroup;
                for (int a = 0; a < atomsPerGroup; a++)
                    posData[base + a] += velData[base + a] * dt;
            });

            // Forces
            context.calcForcesAndEnergy(true, false, allGroupsMask);

            // Half-kick
            {
                vector<Vec3>& forceData = refExtractForces(context);
                forEachGroup(context, [&](int k) {
                    double halfDt = 0.5 * integrator.getGroupStepSize(k);
                    int base = k * atomsPerGroup;
                    for (int a = 0; a < atomsPerGroup; a++) {
                        int idx = base + a;
                        if (masses[idx] > 0)
                            velData[idx] += forceData[idx] * (halfDt / masses[idx]);
                    }
                });
            }
        }
        return;
    }

    // 2-level RESPA
    if (schedule.size() != 2)
        throw OpenMMException("MultiGroupHMCIntegrator: RESPA schedule must have exactly 2 entries");

    int slowIdx = (schedule[0].second <= schedule[1].second) ? 0 : 1;
    int fastIdx = 1 - slowIdx;
    int slowForceGroup = schedule[slowIdx].first;
    int fastForceGroup = schedule[fastIdx].first;
    int innerStepsPerOuter = schedule[fastIdx].second;
    if (innerStepsPerOuter < 1) innerStepsPerOuter = 1;

    int slowMask = (1 << slowForceGroup);
    int fastMask = (1 << fastForceGroup);

    // We need separate slow and fast force arrays since the Reference platform
    // overwrites a single force buffer on each calcForcesAndEnergy call.
    vector<Vec3> slowForces(numParticles, Vec3(0, 0, 0));

    // Compute initial slow forces
    context.calcForcesAndEnergy(true, false, slowMask);
    {
        vector<Vec3>& f = refExtractForces(context);
        for (int i = 0; i < numParticles; i++)
            slowForces[i] = f[i];
    }

    // Compute initial fast forces (left in the context force buffer)
    context.calcForcesAndEnergy(true, false, fastMask);

    for (int outer = 0; outer < numOuterSteps; outer++) {
        // Slow half-kick (outer dt)
        forEachGroup(context, [&](int k) {
            double halfOuterDt = 0.5 * integrator.getGroupStepSize(k);
            int base = k * atomsPerGroup;
            for (int a = 0; a < atomsPerGroup; a++) {
                int idx = base + a;
                if (masses[idx] > 0)
                    velData[idx] += slowForces[idx] * (halfOuterDt / masses[idx]);
            }
        });

        // Inner loop
        for (int inner = 0; inner < innerStepsPerOuter; inner++) {
            // Fast half-kick (inner dt)
            {
                vector<Vec3>& fastForces = refExtractForces(context);
                forEachGroup(context, [&](int k) {
                    double halfInnerDt = 0.5 * integrator.getGroupStepSize(k) / innerStepsPerOuter;
                    int base = k * atomsPerGroup;
                    for (int a = 0; a < atomsPerGroup; a++) {
                        int idx = base + a;
                        if (masses[idx] > 0)
                            velData[idx] += fastForces[idx] * (halfInnerDt / masses[idx]);
                    }
                });
            }

            // Drift (inner dt)
            forEachGroup(context, [&](int k) {
                double innerDt = integrator.getGroupStepSize(k) / innerStepsPerOuter;
                int base = k * atomsPerGroup;
                for (int a = 0; a < atomsPerGroup; a++)
                    posData[base + a] += velData[base + a] * innerDt;
            });

            // Recompute fast forces
            context.calcForcesAndEnergy(true, false, fastMask);

            // Fast half-kick (inner dt)
            {
                vector<Vec3>& fastForces = refExtractForces(context);
                forEachGroup(context, [&](int k) {
                    double halfInnerDt = 0.5 * integrator.getGroupStepSize(k) / innerStepsPerOuter;
                    int base = k * atomsPerGroup;
                    for (int a = 0; a < atomsPerGroup; a++) {
                        int idx = base + a;
                        if (masses[idx] > 0)
                            velData[idx] += fastForces[idx] * (halfInnerDt / masses[idx]);
                    }
                });
            }
        }

        // Recompute slow forces
        context.calcForcesAndEnergy(true, false, slowMask);
        {
            vector<Vec3>& f = refExtractForces(context);
            for (int i = 0; i < numParticles; i++)
                slowForces[i] = f[i];
        }

        // Slow half-kick (outer dt)
        forEachGroup(context, [&](int k) {
            double halfOuterDt = 0.5 * integrator.getGroupStepSize(k);
            int base = k * atomsPerGroup;
            for (int a = 0; a < atomsPerGroup; a++) {
                int idx = base + a;
                if (masses[idx] > 0)
                    velData[idx] += slowForces[idx] * (halfOuterDt / masses[idx]);
            }
        });
    }
}

void ReferenceIntegrateMultiGroupHMCStepKernel::computeGroupKE(
        ContextImpl& context, vector<double>& groupKE) const {

    vector<Vec3>& velData = refExtractVelocities(context);
    groupKE.resize(numGroups);

    for (int k = 0; k < numGroups; k++) {
        double ke = 0.0;
        int base = k * atomsPerGroup;
        for (int a = 0; a < atomsPerGroup; a++) {
            int idx = base + a;
            double m = masses[idx];
            if (m <= 0.0) continue;
            const Vec3& v = velData[idx];
            ke += 0.5 * m * (v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
        }
        groupKE[k] = ke;
    }
}

double ReferenceIntegrateMultiGroupHMCStepKernel::computeKineticEnergy(
        ContextImpl& context, const MultiGroupHMCIntegrator& integrator) {

    vector<double> groupKE;
    computeGroupKE(context, groupKE);
    double total = 0.0;
    for (int k = 0; k < numGroups; k++)
        total += groupKE[k];
    return total;
}

void ReferenceIntegrateMultiGroupHMCStepKernel::resetCounters() {
    fill(acceptCounts.begin(), acceptCounts.end(), 0);
    fill(trialCounts.begin(), trialCounts.end(), 0);
    fill(stabilityRejectCounts.begin(), stabilityRejectCounts.end(), 0);
}

void ReferenceIntegrateMultiGroupHMCStepKernel::resetMCCounters() {
    mcAttemptedTotal = 0;
    mcAcceptedTotal = 0;
    fill(lastMCAcceptedPerGroup.begin(), lastMCAcceptedPerGroup.end(), 0);
}

// Rigid-body Monte Carlo pre-step. Each trial proposes, per eligible group, a
// random rotation about the group center of mass (even trials) plus a Gaussian
// translation, then accepts or rejects per group with the Metropolis criterion
// on that group's potential energy. Mirrors the CUDA executeMC.
void ReferenceIntegrateMultiGroupHMCStepKernel::executeMC(
        ContextImpl& context, const MultiGroupHMCIntegrator& integrator) {

    int K = numGroups;
    int numTrials = integrator.getNumMCTrials();
    double mcStep = integrator.getMCStepSize();

    fill(lastMCAcceptedPerGroup.begin(), lastMCAcceptedPerGroup.end(), 0);

    vector<int> mcEnabled = integrator.getAllGroupMCEnabled();
    bool anyEligible = false;
    for (int k = 0; k < K; k++)
        if (mcEnabled[k]) { anyEligible = true; break; }
    if (!anyEligible) return;

    vector<Vec3>& posData = refExtractPositions(context);

    // Baseline per-group PE.
    int allGroupsMask = 0xFFFFFFFF;
    context.calcForcesAndEnergy(true, true, allGroupsMask);
    vector<double> peBaseline(K, 0.0);
    computeGroupPE(peBaseline);

    vector<double> kT(K);
    for (int k = 0; k < K; k++)
        kT[k] = BOLTZ * integrator.getGroupTemperature(k);

    vector<Vec3> backup(numParticles);

    for (int trial = 0; trial < numTrials; trial++) {
        // Backup, compute the group COM, and propose+apply the rigid-body move.
        // Each group uses only its own RNG stream and atoms, so the proposal is
        // independent of thread count.
        forEachGroup(context, [&](int k) {
            if (!mcEnabled[k]) return;
            int base = k * atomsPerGroup;
            for (int a = 0; a < atomsPerGroup; a++)
                backup[base + a] = posData[base + a];

            double totalMass = 0.0;
            Vec3 acc(0, 0, 0);
            for (int a = 0; a < atomsPerGroup; a++) {
                int idx = base + a;
                double m = masses[idx];
                acc += posData[idx] * m;
                totalMass += m;
            }
            Vec3 com = (totalMass > 0.0) ? acc * (1.0 / totalMass) : Vec3(0, 0, 0);

            double R[9];
            if (trial % 2 == 0)
                generateRandomQuaternionRotation(groupRng[k], groupUniform[k], R);
            else {
                for (int e = 0; e < 9; e++) R[e] = 0.0;
                R[0] = R[4] = R[8] = 1.0;   // identity (translation only)
            }
            Vec3 t(groupNormal[k](groupRng[k]) * mcStep,
                   groupNormal[k](groupRng[k]) * mcStep,
                   groupNormal[k](groupRng[k]) * mcStep);
            for (int a = 0; a < atomsPerGroup; a++) {
                int idx = base + a;
                Vec3 rel = posData[idx] - com;
                Vec3 rot(R[0]*rel[0] + R[1]*rel[1] + R[2]*rel[2],
                         R[3]*rel[0] + R[4]*rel[1] + R[5]*rel[2],
                         R[6]*rel[0] + R[7]*rel[1] + R[8]*rel[2]);
                posData[idx] = rot + com + t;
            }
        });

        // Recompute per-group PE and apply Metropolis per eligible group.
        context.calcForcesAndEnergy(true, true, allGroupsMask);
        vector<double> peTrial(K, 0.0);
        computeGroupPE(peTrial);

        forEachGroup(context, [&](int k) {
            if (!mcEnabled[k]) return;
            double dE = peTrial[k] - peBaseline[k];
            bool accept = (dE <= 0.0) || (groupUniform[k](groupRng[k]) < exp(-dE / kT[k]));
            if (accept) {
                lastMCAcceptedPerGroup[k]++;
                peBaseline[k] = peTrial[k];
            } else {
                int base = k * atomsPerGroup;
                for (int a = 0; a < atomsPerGroup; a++)
                    posData[base + a] = backup[base + a];
            }
        });
    }

    // Deterministic serial reduction of the global MC counters.
    int eligible = 0;
    for (int k = 0; k < K; k++)
        if (mcEnabled[k]) eligible++;
    mcAttemptedTotal += eligible * numTrials;
    for (int k = 0; k < K; k++)
        mcAcceptedTotal += lastMCAcceptedPerGroup[k];
}

}  // namespace GridForcePlugin
