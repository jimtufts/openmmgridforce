/* -------------------------------------------------------------------------- *
 *                              OpenMMGridForce                               *
 * -------------------------------------------------------------------------- *
 * Reference (CPU) implementation of MultiGroupNUTSIntegrator kernel.         *
 * -------------------------------------------------------------------------- */

#include "ReferenceMultiGroupNUTSKernels.h"
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
#include <cmath>
#include <algorithm>

using namespace OpenMM;
using namespace std;

// Use BOLTZ macro from OpenMM's SimTKOpenMMRealType.h = RGAS/KILO = kJ/(mol·K)

namespace GridForcePlugin {

static vector<Vec3>& refExtractVelocitiesNUTS(ContextImpl& context) {
    ReferencePlatform::PlatformData* data =
        reinterpret_cast<ReferencePlatform::PlatformData*>(context.getPlatformData());
    return *((vector<Vec3>*)data->velocities);
}

// Uniform random rotation matrix (row-major 3x3) via Shoemake's quaternion method.
static void generateRandomQuaternionRotationNUTS(
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

void ReferenceIntegrateMultiGroupNUTSStepKernel::initialize(
        const System& system, const MultiGroupNUTSIntegrator& integrator) {

    numGroups = integrator.getNumGroups();
    atomsPerGroup = integrator.getAtomsPerGroup();
    numParticles = system.getNumParticles();

    masses.resize(numParticles);
    for (int i = 0; i < numParticles; i++)
        masses[i] = system.getParticleMass(i);

    int totalGroupAtoms = numGroups * atomsPerGroup;
    positionsBackup.resize(numParticles);
    xminus.resize(totalGroupAtoms);
    xplus.resize(totalGroupAtoms);
    vminus.resize(totalGroupAtoms);
    vplus.resize(totalGroupAtoms);
    candidatePos.resize(totalGroupAtoms);
    subtreeCandidatePos.resize(totalGroupAtoms);

    lastAccepted.resize(numGroups, 0);
    lastTreeDepths.resize(numGroups, 0);
    lastDivergent.resize(numGroups, 0);
    acceptCounts.resize(numGroups, 0);
    trialCounts.resize(numGroups, 0);
    divergenceCounts.resize(numGroups, 0);
    cumulativeTreeDepths.resize(numGroups, 0LL);
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
    groupExponential.assign(numGroups, exponential_distribution<double>(1.0));
    for (int k = 0; k < numGroups; k++) {
        seed_seq seq{seed, (unsigned int)(k + 1)};
        groupRng[k].seed(seq);
    }
}

void ReferenceIntegrateMultiGroupNUTSStepKernel::forEachGroup(
        ContextImpl& context, const std::function<void(int)>& body) {
    for (int k = 0; k < numGroups; k++)
        body(k);
}

void ReferenceIntegrateMultiGroupNUTSStepKernel::computeGroupPE(
        vector<double>& groupPE) const {
    fill(groupPE.begin(), groupPE.end(), 0.0);
    for (auto& extractor : groupEnergyExtractors) {
        vector<double> energies = extractor();
        for (int k = 0; k < numGroups && k < (int)energies.size(); k++)
            groupPE[k] += energies[k];
    }
}

void ReferenceIntegrateMultiGroupNUTSStepKernel::computeGroupKE(
        ContextImpl& context, vector<double>& groupKE) const {

    vector<Vec3>& velData = refExtractVelocitiesNUTS(context);
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

void ReferenceIntegrateMultiGroupNUTSStepKernel::leapfrogStep(
        ContextImpl& context, const MultiGroupNUTSIntegrator& integrator,
        const vector<int>& active, const vector<double>& signedDt,
        bool includeEnergy) {

    vector<Vec3>& posData = refExtractPositions(context);
    vector<Vec3>& velData = refExtractVelocitiesNUTS(context);
    int allGroupsMask = 0xFFFFFFFF;

    // Half-kick
    {
        vector<Vec3>& forceData = refExtractForces(context);
        forEachGroup(context, [&](int k) {
            if (!active[k]) return;
            double halfDt = 0.5 * signedDt[k];
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
        if (!active[k]) return;
        double dt = signedDt[k];
        int base = k * atomsPerGroup;
        for (int a = 0; a < atomsPerGroup; a++)
            posData[base + a] += velData[base + a] * dt;
    });

    // Forces (include energy when requested so callers can read cached PE)
    context.calcForcesAndEnergy(true, includeEnergy, allGroupsMask);

    // Half-kick
    {
        vector<Vec3>& forceData = refExtractForces(context);
        forEachGroup(context, [&](int k) {
            if (!active[k]) return;
            double halfDt = 0.5 * signedDt[k];
            int base = k * atomsPerGroup;
            for (int a = 0; a < atomsPerGroup; a++) {
                int idx = base + a;
                if (masses[idx] > 0)
                    velData[idx] += forceData[idx] * (halfDt / masses[idx]);
            }
        });
    }
}

void ReferenceIntegrateMultiGroupNUTSStepKernel::execute(
        ContextImpl& context, const MultiGroupNUTSIntegrator& integrator,
        bool forcesAreValid) {

    vector<Vec3>& posData = refExtractPositions(context);
    vector<Vec3>& velData = refExtractVelocitiesNUTS(context);
    int K = numGroups;
    int totalGroupAtoms = K * atomsPerGroup;
    int allGroupsMask = 0xFFFFFFFF;
    double stabilityThreshold = integrator.getStabilityThreshold();
    int maxTreeDepth = integrator.getMaxTreeDepth();

    // Wire up energy extractors on first call (same as HMC)
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
                if (f->getNumParticleGroups() > 0) {
                    Context* ctx = &context.getOwner();
                    const GridForce* gf = f;
                    groupEnergyExtractors.push_back([gf, ctx]() {
                        return gf->getParticleGroupEnergies(*ctx);
                    });
                }
            }
        }
    }

    // Rigid-body MC pre-step. May change positions, invalidating forces.
    if (integrator.getNumMCTrials() > 0)
        executeMC(context, integrator);

    // Per-group kT and signed dt
    vector<double> kT(K);
    for (int k = 0; k < K; k++)
        kT[k] = BOLTZ * integrator.getGroupTemperature(k);

    // ===== 1. Backup positions =====
    for (int i = 0; i < numParticles; i++)
        positionsBackup[i] = posData[i];

    // ===== 2. Draw MB velocities =====
    forEachGroup(context, [&](int k) {
        int base = k * atomsPerGroup;
        auto& rngk = groupRng[k];
        auto& nd = groupNormal[k];
        if (integrator.getMomentumRefreshMode() == MultiGroupNUTSIntegrator::FULL) {
            for (int a = 0; a < atomsPerGroup; a++) {
                int idx = base + a;
                double m = masses[idx];
                if (m <= 0.0) continue;
                double sigma = sqrt(kT[k] / m);
                velData[idx] = Vec3(sigma * nd(rngk),
                                    sigma * nd(rngk),
                                    sigma * nd(rngk));
            }
        } else {
            double theta = integrator.getPartialRefreshAngle();
            double cosTheta = cos(theta);
            double sinTheta = sin(theta);
            for (int a = 0; a < atomsPerGroup; a++) {
                int idx = base + a;
                double m = masses[idx];
                if (m <= 0.0) continue;
                double sigma = sqrt(kT[k] / m);
                Vec3 vRand(sigma * nd(rngk),
                           sigma * nd(rngk),
                           sigma * nd(rngk));
                velData[idx] = velData[idx] * cosTheta + vRand * sinTheta;
            }
        }
    });

    // ===== 3. Compute initial H_0 =====
    vector<double> keInit(K);
    computeGroupKE(context, keInit);

    if (!forcesAreValid)
        context.calcForcesAndEnergy(true, true, allGroupsMask);

    vector<double> peInit(K, 0.0);
    computeGroupPE(peInit);

    vector<double> H0(K);
    vector<double> logu(K);
    forEachGroup(context, [&](int k) {
        H0[k] = peInit[k] + keInit[k];
        logu[k] = -H0[k] / kT[k] - groupExponential[k](groupRng[k]);
    });

    // ===== 4. Initialize tree =====
    for (int i = 0; i < totalGroupAtoms; i++) {
        xminus[i] = posData[i];
        xplus[i] = posData[i];
        candidatePos[i] = posData[i];
        vminus[i] = velData[i];
        vplus[i] = velData[i];
    }

    vector<int> nValid(K, 1);
    vector<double> candidatePE(K);
    for (int k = 0; k < K; k++)
        candidatePE[k] = peInit[k];

    vector<int> active(K, 1);
    vector<int> divergent(K, 0);
    fill(lastTreeDepths.begin(), lastTreeDepths.end(), 0);

    // ===== 5. Tree doubling =====
    for (int depth = 0; depth < maxTreeDepth; depth++) {
        bool anyActive = false;
        for (int k = 0; k < K; k++)
            if (active[k]) { anyActive = true; break; }
        if (!anyActive) break;

        // Choose random direction per active group
        vector<int> direction(K);
        forEachGroup(context, [&](int k) {
            direction[k] = active[k] ? ((groupUniform[k](groupRng[k]) < 0.5) ? -1 : 1) : 1;
        });

        // Restore from appropriate endpoint
        forEachGroup(context, [&](int k) {
            if (!active[k]) return;
            int base = k * atomsPerGroup;
            for (int a = 0; a < atomsPerGroup; a++) {
                int idx = base + a;
                if (direction[k] > 0) {
                    posData[idx] = xplus[idx];
                    velData[idx] = vplus[idx];
                } else {
                    posData[idx] = xminus[idx];
                    velData[idx] = vminus[idx];
                }
            }
        });

        // Signed dt
        vector<double> signedDt(K);
        for (int k = 0; k < K; k++)
            signedDt[k] = direction[k] * fabs(integrator.getGroupStepSize(k));

        // Compute forces before stepping
        context.calcForcesAndEnergy(true, false, allGroupsMask);

        // Take 2^depth leapfrog steps
        int numSteps = 1 << depth;
        vector<int> subtreeNValid(K, 0);
        vector<double> subtreeCandidatePE(K, 0.0);
        // char, not vector<bool>: distinct elements must be independently
        // writable from different threads (vector<bool> packs bits into shared words).
        vector<char> subtreeHasCandidate(K, 0);

        for (int step = 0; step < numSteps; step++) {
            // includeEnergy=true so forces cache per-group energies
            // inside the leapfrog step, eliminating the redundant force eval
            leapfrogStep(context, integrator, active, signedDt, true);

            vector<double> peStep(K, 0.0);
            computeGroupPE(peStep);
            vector<double> keStep(K);
            computeGroupKE(context, keStep);

            forEachGroup(context, [&](int k) {
                if (!active[k]) return;

                double Hk = peStep[k] + keStep[k];
                double logPk = -Hk / kT[k];

                double deltaPE = peStep[k] - peInit[k];
                double deltaH = Hk - H0[k];
                bool isDivergent = (fabs(deltaPE) / kT[k] > stabilityThreshold) &&
                                   (fabs(deltaH) / kT[k] > stabilityThreshold);

                if (isDivergent) {
                    active[k] = 0;
                    divergent[k] = 1;
                    return;
                }

                if (logPk > logu[k]) {
                    subtreeNValid[k]++;
                    if (groupUniform[k](groupRng[k]) < 1.0 / subtreeNValid[k]) {
                        subtreeCandidatePE[k] = peStep[k];
                        subtreeHasCandidate[k] = true;
                        // Save to subtree candidate (not main candidate)
                        int base = k * atomsPerGroup;
                        for (int a = 0; a < atomsPerGroup; a++)
                            subtreeCandidatePos[base + a] = posData[base + a];
                    }
                }
            });
        }

        // Save to endpoint
        forEachGroup(context, [&](int k) {
            if (!active[k]) return;
            int base = k * atomsPerGroup;
            for (int a = 0; a < atomsPerGroup; a++) {
                int idx = base + a;
                if (direction[k] > 0) {
                    xplus[idx] = posData[idx];
                    vplus[idx] = velData[idx];
                } else {
                    xminus[idx] = posData[idx];
                    vminus[idx] = velData[idx];
                }
            }
        });

        // Combine subtree candidate with main candidate
        forEachGroup(context, [&](int k) {
            if (!active[k]) return;
            if (!subtreeHasCandidate[k]) return;

            int totalN = nValid[k] + subtreeNValid[k];
            double acceptProb = (double)subtreeNValid[k] / totalN;
            if (groupUniform[k](groupRng[k]) < acceptProb) {
                candidatePE[k] = subtreeCandidatePE[k];
                // Copy subtree candidate to main candidate
                int base = k * atomsPerGroup;
                for (int a = 0; a < atomsPerGroup; a++)
                    candidatePos[base + a] = subtreeCandidatePos[base + a];
            }
            nValid[k] = totalN;
        });

        // U-turn check
        forEachGroup(context, [&](int k) {
            if (!active[k]) return;
            int base = k * atomsPerGroup;

            double dot1 = 0.0, dot2 = 0.0;
            for (int a = 0; a < atomsPerGroup; a++) {
                int idx = base + a;
                Vec3 dx = xplus[idx] - xminus[idx];
                dot1 += dx[0]*vminus[idx][0] + dx[1]*vminus[idx][1] + dx[2]*vminus[idx][2];
                dot2 += dx[0]*vplus[idx][0] + dx[1]*vplus[idx][1] + dx[2]*vplus[idx][2];
            }

            if (dot1 < 0.0 || dot2 < 0.0)
                active[k] = 0;
        });

        // Record tree depths
        for (int k = 0; k < K; k++) {
            if (lastTreeDepths[k] < depth + 1 && (active[k] || depth == 0))
                lastTreeDepths[k] = depth + 1;
            if (!active[k] && lastTreeDepths[k] == 0)
                lastTreeDepths[k] = depth + 1;
        }
    }

    // Final tree depths
    for (int k = 0; k < K; k++) {
        if (active[k])
            lastTreeDepths[k] = maxTreeDepth;
        if (lastTreeDepths[k] == 0)
            lastTreeDepths[k] = 1;
    }

    // ===== 6. Set from candidates, restore divergent =====
    forEachGroup(context, [&](int k) {
        int base = k * atomsPerGroup;
        if (divergent[k]) {
            for (int a = 0; a < atomsPerGroup; a++) {
                posData[base + a] = positionsBackup[base + a];
                velData[base + a] = Vec3(0, 0, 0);
            }
        } else {
            for (int a = 0; a < atomsPerGroup; a++) {
                posData[base + a] = candidatePos[base + a];
                velData[base + a] = Vec3(0, 0, 0);
            }
        }
    });

    // ===== 7. Update statistics =====
    for (int k = 0; k < K; k++) {
        trialCounts[k]++;
        cumulativeTreeDepths[k] += lastTreeDepths[k];

        if (divergent[k]) {
            lastAccepted[k] = 0;
            lastDivergent[k] = 1;
            divergenceCounts[k]++;
        } else {
            lastAccepted[k] = 1;
            lastDivergent[k] = 0;
            acceptCounts[k]++;
        }
    }
}

double ReferenceIntegrateMultiGroupNUTSStepKernel::computeKineticEnergy(
        ContextImpl& context, const MultiGroupNUTSIntegrator& integrator) {

    vector<double> groupKE;
    computeGroupKE(context, groupKE);
    double total = 0.0;
    for (int k = 0; k < numGroups; k++)
        total += groupKE[k];
    return total;
}

void ReferenceIntegrateMultiGroupNUTSStepKernel::resetCounters() {
    fill(acceptCounts.begin(), acceptCounts.end(), 0);
    fill(trialCounts.begin(), trialCounts.end(), 0);
    fill(divergenceCounts.begin(), divergenceCounts.end(), 0);
    fill(cumulativeTreeDepths.begin(), cumulativeTreeDepths.end(), 0LL);
}

void ReferenceIntegrateMultiGroupNUTSStepKernel::resetMCCounters() {
    mcAttemptedTotal = 0;
    mcAcceptedTotal = 0;
    fill(lastMCAcceptedPerGroup.begin(), lastMCAcceptedPerGroup.end(), 0);
}

// Rigid-body Monte Carlo pre-step. Each trial proposes, per eligible group, a
// random rotation about the group center of mass (even trials) plus a Gaussian
// translation, then accepts or rejects per group with the Metropolis criterion
// on that group's potential energy. Mirrors the CUDA executeMC.
void ReferenceIntegrateMultiGroupNUTSStepKernel::executeMC(
        ContextImpl& context, const MultiGroupNUTSIntegrator& integrator) {

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
                generateRandomQuaternionRotationNUTS(groupRng[k], groupUniform[k], R);
            else {
                for (int e = 0; e < 9; e++) R[e] = 0.0;
                R[0] = R[4] = R[8] = 1.0;
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
