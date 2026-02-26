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

    int seed = integrator.getRandomNumberSeed();
    if (seed == 0) {
        random_device rd;
        seed = rd();
    }
    rng.seed(seed);
    normalDist = normal_distribution<double>(0.0, 1.0);
    uniformDist = uniform_real_distribution<double>(0.0, 1.0);
    exponentialDist = exponential_distribution<double>(1.0);
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
    vector<Vec3>& forceData = refExtractForces(context);
    for (int k = 0; k < numGroups; k++) {
        if (!active[k]) continue;
        double halfDt = 0.5 * signedDt[k];
        int base = k * atomsPerGroup;
        for (int a = 0; a < atomsPerGroup; a++) {
            int idx = base + a;
            if (masses[idx] > 0)
                velData[idx] += forceData[idx] * (halfDt / masses[idx]);
        }
    }

    // Drift
    for (int k = 0; k < numGroups; k++) {
        if (!active[k]) continue;
        double dt = signedDt[k];
        int base = k * atomsPerGroup;
        for (int a = 0; a < atomsPerGroup; a++)
            posData[base + a] += velData[base + a] * dt;
    }

    // Forces (include energy when requested so callers can read cached PE)
    context.calcForcesAndEnergy(true, includeEnergy, allGroupsMask);

    // Half-kick
    forceData = refExtractForces(context);
    for (int k = 0; k < numGroups; k++) {
        if (!active[k]) continue;
        double halfDt = 0.5 * signedDt[k];
        int base = k * atomsPerGroup;
        for (int a = 0; a < atomsPerGroup; a++) {
            int idx = base + a;
            if (masses[idx] > 0)
                velData[idx] += forceData[idx] * (halfDt / masses[idx]);
        }
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

    // Per-group kT and signed dt
    vector<double> kT(K);
    for (int k = 0; k < K; k++)
        kT[k] = BOLTZ * integrator.getGroupTemperature(k);

    // ===== 1. Backup positions =====
    for (int i = 0; i < numParticles; i++)
        positionsBackup[i] = posData[i];

    // ===== 2. Draw MB velocities =====
    for (int k = 0; k < K; k++) {
        int base = k * atomsPerGroup;
        if (integrator.getMomentumRefreshMode() == MultiGroupNUTSIntegrator::FULL) {
            for (int a = 0; a < atomsPerGroup; a++) {
                int idx = base + a;
                double m = masses[idx];
                if (m <= 0.0) continue;
                double sigma = sqrt(kT[k] / m);
                velData[idx] = Vec3(sigma * normalDist(rng),
                                    sigma * normalDist(rng),
                                    sigma * normalDist(rng));
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
                Vec3 vRand(sigma * normalDist(rng),
                           sigma * normalDist(rng),
                           sigma * normalDist(rng));
                velData[idx] = velData[idx] * cosTheta + vRand * sinTheta;
            }
        }
    }

    // ===== 3. Compute initial H_0 =====
    vector<double> keInit(K);
    computeGroupKE(context, keInit);

    if (!forcesAreValid)
        context.calcForcesAndEnergy(true, true, allGroupsMask);

    vector<double> peInit(K, 0.0);
    computeGroupPE(peInit);

    vector<double> H0(K);
    vector<double> logu(K);
    for (int k = 0; k < K; k++) {
        H0[k] = peInit[k] + keInit[k];
        logu[k] = -H0[k] / kT[k] - exponentialDist(rng);
    }

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
        for (int k = 0; k < K; k++)
            direction[k] = active[k] ? ((uniformDist(rng) < 0.5) ? -1 : 1) : 1;

        // Restore from appropriate endpoint
        for (int k = 0; k < K; k++) {
            if (!active[k]) continue;
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
        }

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
        vector<bool> subtreeHasCandidate(K, false);

        for (int step = 0; step < numSteps; step++) {
            // includeEnergy=true so forces cache per-group energies
            // inside the leapfrog step, eliminating the redundant force eval
            leapfrogStep(context, integrator, active, signedDt, true);

            vector<double> peStep(K, 0.0);
            computeGroupPE(peStep);
            vector<double> keStep(K);
            computeGroupKE(context, keStep);

            for (int k = 0; k < K; k++) {
                if (!active[k]) continue;

                double Hk = peStep[k] + keStep[k];
                double logPk = -Hk / kT[k];

                double deltaPE = peStep[k] - peInit[k];
                double deltaH = Hk - H0[k];
                bool isDivergent = (fabs(deltaPE) / kT[k] > stabilityThreshold) &&
                                   (fabs(deltaH) / kT[k] > stabilityThreshold);

                if (isDivergent) {
                    active[k] = 0;
                    divergent[k] = 1;
                    continue;
                }

                if (logPk > logu[k]) {
                    subtreeNValid[k]++;
                    if (uniformDist(rng) < 1.0 / subtreeNValid[k]) {
                        subtreeCandidatePE[k] = peStep[k];
                        subtreeHasCandidate[k] = true;
                        // Save to subtree candidate (not main candidate)
                        int base = k * atomsPerGroup;
                        for (int a = 0; a < atomsPerGroup; a++)
                            subtreeCandidatePos[base + a] = posData[base + a];
                    }
                }
            }
        }

        // Save to endpoint
        for (int k = 0; k < K; k++) {
            if (!active[k]) continue;
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
        }

        // Combine subtree candidate with main candidate
        for (int k = 0; k < K; k++) {
            if (!active[k]) continue;
            if (!subtreeHasCandidate[k]) continue;

            int totalN = nValid[k] + subtreeNValid[k];
            double acceptProb = (double)subtreeNValid[k] / totalN;
            if (uniformDist(rng) < acceptProb) {
                candidatePE[k] = subtreeCandidatePE[k];
                // Copy subtree candidate to main candidate
                int base = k * atomsPerGroup;
                for (int a = 0; a < atomsPerGroup; a++)
                    candidatePos[base + a] = subtreeCandidatePos[base + a];
            }
            nValid[k] = totalN;
        }

        // U-turn check
        for (int k = 0; k < K; k++) {
            if (!active[k]) continue;
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
        }

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
    for (int k = 0; k < K; k++) {
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
    }

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

}  // namespace GridForcePlugin
