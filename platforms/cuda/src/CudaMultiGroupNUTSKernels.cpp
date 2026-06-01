/* -------------------------------------------------------------------------- *
 *                              OpenMMGridForce                               *
 * -------------------------------------------------------------------------- *
 * CUDA implementation of MultiGroupNUTSIntegrator kernel.                    *
 *                                                                            *
 * Implements the No-U-Turn Sampler (Hoffman & Gelman, 2014) adapted for     *
 * simultaneous multi-group execution on GPU. Each group independently        *
 * builds a NUTS tree with adaptive trajectory length.                        *
 * -------------------------------------------------------------------------- */

#include "CudaMultiGroupNUTSKernels.h"
#include "CudaGridForceKernelSources.h"
#include "GridForce.h"
#include "IsolatedBondedForce.h"
#include "IsolatedNonbondedForce.h"
#include "IsolatedGBSAForce.h"
#include "IsolatedSiteForce.h"
#include "GBSAGridForce.h"
#include "internal/GridForceImpl.h"
#include "internal/IsolatedBondedForceImpl.h"
#include "internal/IsolatedNonbondedForceImpl.h"
#include "internal/IsolatedGBSAForceImpl.h"
#include "internal/IsolatedSiteForceImpl.h"
#include "internal/GBSAGridForceImpl.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/OpenMMException.h"
#include "openmm/cuda/CudaContext.h"
#include "openmm/cuda/CudaArray.h"
#include "openmm/cuda/CudaIntegrationUtilities.h"
#include <cmath>
#include <algorithm>

using namespace OpenMM;
using namespace std;

// kJ/(mol·K)
// kJ/(mol·K) — same as OpenMM's BOLTZ macro from SimTKOpenMMRealType.h
// but that header isn't included in the CUDA platform.
static const double BOLTZ = 0.008314462618;

namespace GridForcePlugin {

void CudaIntegrateMultiGroupNUTSStepKernel::initialize(
        const System& system, const MultiGroupNUTSIntegrator& integrator) {

    cu.initializeContexts();

    numGroups = integrator.getNumGroups();
    atomsPerGroup = integrator.getAtomsPerGroup();
    numParticles = system.getNumParticles();

    // Resize host-side vectors
    groupKEHost.resize(numGroups, 0.0);
    groupStepSizesHost.resize(numGroups, 0.0);
    groupKTHost.resize(numGroups, 0.0);
    activeHost.resize(numGroups, 1);
    directionHost.resize(numGroups, 1);
    copyFlagHost.resize(numGroups, 0);
    uturnDotHost.resize(2 * numGroups, 0.0);
    divergentHost.resize(numGroups, 0);

    lastAccepted.resize(numGroups, 0);
    lastTreeDepths.resize(numGroups, 0);
    lastDivergent.resize(numGroups, 0);
    acceptCounts.resize(numGroups, 0);
    trialCounts.resize(numGroups, 0);
    divergenceCounts.resize(numGroups, 0);
    cumulativeTreeDepths.resize(numGroups, 0LL);

    // Seed host RNG
    rng.seed((unsigned int)integrator.getRandomNumberSeed());

    // MC counters
    mcAttemptedTotal = 0;
    mcAcceptedTotal = 0;
    lastMCAcceptedPerGroup.resize(numGroups, 0);

    // Metric state
    metricInitialized = false;
    gridForceImpls.clear();
    conditionNumbersHost.resize(numGroups, 1.0f);

    // MC host vectors
    mcGroupCOMHost.resize(4 * numGroups, 0.0);
    mcEnabledHost.resize(numGroups, 0);
    mcRotationHost.resize(9 * numGroups, 0.0);
    mcTranslationHost.resize(3 * numGroups, 0.0);
    mcCOMHost.resize(3 * numGroups, 0.0);
    mcAcceptedHost.resize(numGroups, 0);

    hasInitializedKernel = false;
}

void CudaIntegrateMultiGroupNUTSStepKernel::computeGroupPE(
        vector<double>& groupPE) const {
    fill(groupPE.begin(), groupPE.end(), 0.0);
    for (auto& extractor : groupEnergyExtractors) {
        vector<double> energies = extractor();
        for (int k = 0; k < numGroups && k < (int)energies.size(); k++)
            groupPE[k] += energies[k];
    }
}

void CudaIntegrateMultiGroupNUTSStepKernel::computeGroupKE(
        ContextImpl& context, vector<double>& groupKE) {

    cu.clearBuffer(groupKEBuffer);

    CUdeviceptr velmPtr = cu.getVelm().getDevicePointer();
    CUdeviceptr kePtr = groupKEBuffer.getDevicePointer();

    int totalAtoms = numGroups * atomsPerGroup;
    int blockSize = 128;
    int numBlocks = min((totalAtoms + blockSize - 1) / blockSize,
                        cu.getNumThreadBlocks());

    if (metricInitialized && metricBuffer.isInitialized()) {
        // Metric-aware KE: 0.5 * v^T * G * v
        CUdeviceptr mPtr = metricBuffer.getDevicePointer();
        void* args[] = {
            &velmPtr, &mPtr, &kePtr,
            &atomsPerGroup, &numGroups
        };
        cu.executeKernel(rmComputeGroupKEKernel, args, numBlocks * blockSize, blockSize);
    } else {
        // Standard KE: 0.5 * v^2 / invMass
        void* args[] = {
            &velmPtr, &kePtr,
            &atomsPerGroup, &numGroups
        };
        cu.executeKernel(computeGroupKEKernel, args, numBlocks * blockSize, blockSize);
    }

    groupKEBuffer.download(groupKEHost);
    groupKE = groupKEHost;
}

void CudaIntegrateMultiGroupNUTSStepKernel::launchKick(
        CUdeviceptr forcePtr, double scaleFactor) {

    CUdeviceptr velmPtr = cu.getVelm().getDevicePointer();
    CUdeviceptr dtPtr = groupStepSizesBuffer.getDevicePointer();
    CUdeviceptr activePtr = activeBuffer.getDevicePointer();
    int paddedNumAtoms = cu.getPaddedNumAtoms();

    int totalAtoms = numGroups * atomsPerGroup;
    int blockSize = 128;
    int numBlocks = min((totalAtoms + blockSize - 1) / blockSize,
                        cu.getNumThreadBlocks());

    float scaleF = (float)scaleFactor;
    double scaleD = scaleFactor;
    void* scalePtr;
    if (cu.getUseDoublePrecision() || cu.getUseMixedPrecision())
        scalePtr = &scaleD;
    else
        scalePtr = &scaleF;

    if (metricInitialized && metricInvBuffer.isInitialized()) {
        // Metric-aware kick: v += scale * dt * G^{-1} * F
        CUdeviceptr miPtr = metricInvBuffer.getDevicePointer();
        void* args[] = {
            &velmPtr, &forcePtr, &miPtr, &dtPtr, &activePtr,
            &atomsPerGroup, &numGroups,
            scalePtr,
            &paddedNumAtoms
        };
        cu.executeKernel(rmVelocityKickKernel, args, numBlocks * blockSize, blockSize);
    } else {
        // Standard kick: v += scale * dt * F * invMass
        void* args[] = {
            &velmPtr, &forcePtr, &dtPtr, &activePtr,
            &atomsPerGroup, &numGroups,
            scalePtr,
            &paddedNumAtoms
        };
        cu.executeKernel(velocityKickKernel, args, numBlocks * blockSize, blockSize);
    }
}

void CudaIntegrateMultiGroupNUTSStepKernel::launchDrift() {

    CUdeviceptr posqPtr = cu.getPosq().getDevicePointer();
    CUdeviceptr velmPtr = cu.getVelm().getDevicePointer();
    CUdeviceptr dtPtr = groupStepSizesBuffer.getDevicePointer();
    CUdeviceptr activePtr = activeBuffer.getDevicePointer();

    int totalAtoms = numGroups * atomsPerGroup;
    int blockSize = 128;
    int numBlocks = min((totalAtoms + blockSize - 1) / blockSize,
                        cu.getNumThreadBlocks());

    void* args[] = {
        &posqPtr, &velmPtr, &dtPtr, &activePtr,
        &atomsPerGroup, &numGroups
    };
    cu.executeKernel(positionDriftKernel, args, numBlocks * blockSize, blockSize);
}

void CudaIntegrateMultiGroupNUTSStepKernel::nutsLeapfrogStep(
        ContextImpl& context, const MultiGroupNUTSIntegrator& integrator,
        CUdeviceptr forcePtr, bool includeEnergy) {

    // Simple Verlet: half-kick, drift, force, half-kick
    int allGroupsMask = 0xFFFFFFFF;

    launchKick(forcePtr, 0.5);
    launchDrift();
    context.calcForcesAndEnergy(true, includeEnergy, allGroupsMask);
    launchKick(forcePtr, 0.5);
}

void CudaIntegrateMultiGroupNUTSStepKernel::respaLeapfrogStep(
        ContextImpl& context, const MultiGroupNUTSIntegrator& integrator,
        bool includeEnergy) {

    int K = numGroups;
    const vector<pair<int,int> >& schedule = integrator.getForceGroupSchedule();
    CUdeviceptr forcePtr = cu.getLongForceBuffer().getDevicePointer();
    int paddedNumAtoms = cu.getPaddedNumAtoms();
    int numGroupAtoms = K * atomsPerGroup;

    if (schedule.empty()) {
        nutsLeapfrogStep(context, integrator, forcePtr, includeEnergy);
        return;
    }

    // 2-level RESPA
    if (schedule.size() != 2)
        throw OpenMMException("MultiGroupNUTSIntegrator: RESPA schedule must have exactly 2 entries");

    int slowIdx = (schedule[0].second <= schedule[1].second) ? 0 : 1;
    int fastIdx = 1 - slowIdx;
    int slowForceGroup = schedule[slowIdx].first;
    int fastForceGroup = schedule[fastIdx].first;
    int innerStepsPerOuter = schedule[fastIdx].second;
    if (innerStepsPerOuter < 1) innerStepsPerOuter = 1;

    int slowMask = (1 << slowForceGroup);
    int fastMask = (1 << fastForceGroup);

    CUdeviceptr slowForcePtr = slowForcesBackup.getDevicePointer();

    // Compute slow forces and save
    context.calcForcesAndEnergy(true, false, slowMask);
    {
        int blockSize = 128;
        int numBlocks = min((numGroupAtoms + blockSize - 1) / blockSize,
                            cu.getNumThreadBlocks());
        void* args[] = { &forcePtr, &slowForcePtr, &numGroupAtoms, &paddedNumAtoms };
        cu.executeKernel(copyForcesKernel, args, numBlocks * blockSize, blockSize);
    }

    // Compute fast forces
    context.calcForcesAndEnergy(true, false, fastMask);

    // Save current outer dt, prepare inner dt
    vector<double> outerDtHost = groupStepSizesHost;  // already set with signed dt
    vector<double> innerDtHost(K);
    for (int k = 0; k < K; k++)
        innerDtHost[k] = outerDtHost[k] / innerStepsPerOuter;

    // Slow half-kick (outer dt)
    groupStepSizesBuffer.upload(outerDtHost);
    launchKick(slowForcePtr, 0.5);

    // Inner loop: include energy only on the last fast-force eval
    for (int inner = 0; inner < innerStepsPerOuter; inner++) {
        groupStepSizesBuffer.upload(innerDtHost);
        launchKick(forcePtr, 0.5);
        launchDrift();
        bool lastInner = (inner == innerStepsPerOuter - 1);
        context.calcForcesAndEnergy(true, includeEnergy && lastInner, fastMask);
        launchKick(forcePtr, 0.5);
    }

    // Recompute slow forces (include energy so both force groups have cached PE)
    context.calcForcesAndEnergy(true, includeEnergy, slowMask);
    {
        int blockSize = 128;
        int numBlocks = min((numGroupAtoms + blockSize - 1) / blockSize,
                            cu.getNumThreadBlocks());
        void* args[] = { &forcePtr, &slowForcePtr, &numGroupAtoms, &paddedNumAtoms };
        cu.executeKernel(copyForcesKernel, args, numBlocks * blockSize, blockSize);
    }

    // Slow half-kick (outer dt)
    groupStepSizesBuffer.upload(outerDtHost);
    launchKick(slowForcePtr, 0.5);

    // Restore outer dt
    groupStepSizesBuffer.upload(outerDtHost);
}

double CudaIntegrateMultiGroupNUTSStepKernel::computeKineticEnergy(
        ContextImpl& context, const MultiGroupNUTSIntegrator& integrator) {

    if (!hasInitializedKernel)
        return 0.0;
    vector<double> groupKE;
    computeGroupKE(context, groupKE);
    double total = 0.0;
    for (int k = 0; k < numGroups; k++)
        total += groupKE[k];
    return total;
}

void CudaIntegrateMultiGroupNUTSStepKernel::resetCounters() {
    fill(acceptCounts.begin(), acceptCounts.end(), 0);
    fill(trialCounts.begin(), trialCounts.end(), 0);
    fill(divergenceCounts.begin(), divergenceCounts.end(), 0);
    fill(cumulativeTreeDepths.begin(), cumulativeTreeDepths.end(), 0LL);
}

void CudaIntegrateMultiGroupNUTSStepKernel::resetMCCounters() {
    mcAttemptedTotal = 0;
    mcAcceptedTotal = 0;
    fill(lastMCAcceptedPerGroup.begin(), lastMCAcceptedPerGroup.end(), 0);
}

// ========== Riemannian Metric ==========

void CudaIntegrateMultiGroupNUTSStepKernel::assembleMetric(
        ContextImpl& context, const MultiGroupNUTSIntegrator& integrator) {

    int K = numGroups;
    int totalAtoms = K * atomsPerGroup;
    int numElements = 6 * totalAtoms;
    int blockSize = 128;

    // Lazy initialization of metric buffers
    if (!metricInitialized) {
        metricBuffer.initialize<float>(cu, numElements, "nutsMetric");
        metricInvBuffer.initialize<float>(cu, numElements, "nutsMetricInv");
        choleskyBuffer.initialize<float>(cu, numElements, "nutsCholesky");
        logDetBuffer.initialize<double>(cu, K, "nutsLogDet");
        conditionBuffer.initialize<float>(cu, K, "nutsCondition");
        combinedHessianBuffer.initialize<float>(cu, numElements, "nutsCombinedHessian");

        // Discover ALL GridForceImpls for Hessian computation
        gridForceImpls.clear();
        for (ForceImpl* impl : context.getForceImpls()) {
            if (auto* p = dynamic_cast<GridForceImpl*>(impl)) {
                gridForceImpls.push_back(p);
            }
        }

        // Discover IsolatedBondedForceImpl for bonded Hessian
        bondedForceImpl = nullptr;
        for (ForceImpl* impl : context.getForceImpls()) {
            if (auto* p = dynamic_cast<IsolatedBondedForceImpl*>(impl)) {
                bondedForceImpl = p;
                break;
            }
        }

        // Discover IsolatedNonbondedForceImpl for LJ+Coulomb Hessian
        nonbondedForceImpl = nullptr;
        for (ForceImpl* impl : context.getForceImpls()) {
            if (auto* p = dynamic_cast<IsolatedNonbondedForceImpl*>(impl)) {
                nonbondedForceImpl = p;
                break;
            }
        }

        metricInitialized = true;
    }

    auto metricType = integrator.getMetricType();

    if (metricType == MultiGroupNUTSIntegrator::METRIC_IDENTITY) {
        // Set identity metric from mass matrix
        CUdeviceptr velmPtr = cu.getVelm().getDevicePointer();
        CUdeviceptr mPtr = metricBuffer.getDevicePointer();
        CUdeviceptr miPtr = metricInvBuffer.getDevicePointer();
        CUdeviceptr chPtr = choleskyBuffer.getDevicePointer();

        int numBlocks = min((totalAtoms + blockSize - 1) / blockSize,
                            cu.getNumThreadBlocks());
        void* args[] = { &velmPtr, &mPtr, &miPtr, &chPtr, &totalAtoms };
        cu.executeKernel(setIdentityMetricKernel, args, numBlocks * blockSize, blockSize);
        return;
    }

    // Accumulate Hessians from all forces into combined buffer

    // Zero the combined Hessian buffer
    cu.clearBuffer(combinedHessianBuffer);

    int numBlocksAccum = min((numElements + blockSize - 1) / blockSize,
                              cu.getNumThreadBlocks());

    int numHessiansAccumulated = 0;

    // Grid force Hessians (weighted by gridHessianWeight)
    float gridWeight = (float)integrator.getGridHessianWeight();
    if (gridWeight > 0.0f) {
        for (GridForceImpl* impl : gridForceImpls) {
            void* hessianPtr = impl->getHessianDevicePointer();
            if (hessianPtr == nullptr)
                continue;

            impl->computeHessianGPU();
            hessianPtr = impl->getHessianDevicePointer();

            CUdeviceptr destPtr = combinedHessianBuffer.getDevicePointer();
            CUdeviceptr srcPtr = (CUdeviceptr)hessianPtr;
            void* accumArgs[] = { &destPtr, &srcPtr, &gridWeight, &numElements };
            cu.executeKernel(accumulateHessianWeightedKernel, accumArgs,
                             numBlocksAccum * blockSize, blockSize);
            numHessiansAccumulated++;
        }
    }

    // Bonded Hessian (always at full strength, not alpha-scaled)
    if (bondedForceImpl != nullptr) {
        bondedForceImpl->computeDiagonalHessianGPU();
        void* bHessPtr = bondedForceImpl->getDiagonalHessianDevicePointer();

        if (bHessPtr != nullptr) {
            CUdeviceptr destPtr = combinedHessianBuffer.getDevicePointer();
            CUdeviceptr srcPtr = (CUdeviceptr)bHessPtr;
            void* accumArgs[] = { &destPtr, &srcPtr, &numElements };
            cu.executeKernel(accumulateHessianKernel, accumArgs,
                             numBlocksAccum * blockSize, blockSize);
            numHessiansAccumulated++;
        }
    }

    // Nonbonded Hessian (LJ + Coulomb, always at full strength)
    if (nonbondedForceImpl != nullptr) {
        nonbondedForceImpl->computeDiagonalHessianGPU();
        void* nbHessPtr = nonbondedForceImpl->getDiagonalHessianDevicePointer();

        if (nbHessPtr != nullptr) {
            CUdeviceptr destPtr = combinedHessianBuffer.getDevicePointer();
            CUdeviceptr srcPtr = (CUdeviceptr)nbHessPtr;
            void* accumArgs[] = { &destPtr, &srcPtr, &numElements };
            cu.executeKernel(accumulateHessianKernel, accumArgs,
                             numBlocksAccum * blockSize, blockSize);
            numHessiansAccumulated++;
        }
    }

    // External Hessian (e.g., OBC solvation computed via JAX on host)
    if (integrator.hasExternalHessian()) {
        const auto& extHess = integrator.getExternalDiagonalHessian();
        if (!externalHessianBuffer.isInitialized()) {
            externalHessianBuffer.initialize<float>(cu, numElements, "nutsExternalHessian");
        }
        externalHessianBuffer.upload(extHess);

        CUdeviceptr destPtr = combinedHessianBuffer.getDevicePointer();
        CUdeviceptr srcPtr = externalHessianBuffer.getDevicePointer();
        void* accumArgs[] = { &destPtr, &srcPtr, &numElements };
        cu.executeKernel(accumulateHessianKernel, accumArgs,
                         numBlocksAccum * blockSize, blockSize);
        numHessiansAccumulated++;
    }

    if (numHessiansAccumulated == 0)
        throw OpenMMException("MultiGroupNUTSIntegrator: No Hessian source found for metric computation");

    // Zero logDet and condition buffers before atomic accumulation
    cu.clearBuffer(logDetBuffer);
    cu.clearBuffer(conditionBuffer);

    // Launch metric assembly kernel with combined Hessian
    CUdeviceptr hPtr = combinedHessianBuffer.getDevicePointer();
    CUdeviceptr velmPtr = cu.getVelm().getDevicePointer();
    CUdeviceptr mPtr = metricBuffer.getDevicePointer();
    CUdeviceptr miPtr = metricInvBuffer.getDevicePointer();
    CUdeviceptr chPtr = choleskyBuffer.getDevicePointer();
    CUdeviceptr ldPtr = logDetBuffer.getDevicePointer();
    CUdeviceptr condPtr = conditionBuffer.getDevicePointer();

    int metricTypeInt = (int)metricType;
    float alpha = (float)integrator.getSoftAbsAlpha();
    float beta = (float)integrator.getMetricBlendFactor();

    int numBlocks = min((totalAtoms + blockSize - 1) / blockSize,
                        cu.getNumThreadBlocks());
    void* args[] = {
        &hPtr, &velmPtr,
        &mPtr, &miPtr, &chPtr, &ldPtr, &condPtr,
        &metricTypeInt, &alpha, &beta,
        &atomsPerGroup, &numGroups
    };
    cu.executeKernel(assembleMetricKernel, args, numBlocks * blockSize, blockSize);

    // Download condition numbers for diagnostics
    conditionBuffer.download(conditionNumbersHost);
}

vector<double> CudaIntegrateMultiGroupNUTSStepKernel::getGroupMetricConditionNumbers() const {
    vector<double> result(numGroups, 1.0);
    for (int k = 0; k < numGroups; k++)
        result[k] = (double)conditionNumbersHost[k];
    return result;
}

// Helper: random quaternion -> rotation matrix (same as HMC version)
static void nutsGenerateRandomQuaternionRotation(std::mt19937& rng,
    std::uniform_real_distribution<double>& uDist, double* R) {
    double u0 = uDist(rng);
    double u1 = uDist(rng);
    double u2 = uDist(rng);
    double s0 = sqrt(1.0 - u0);
    double s1 = sqrt(u0);
    double t1 = 2.0 * M_PI * u1;
    double t2 = 2.0 * M_PI * u2;
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

void CudaIntegrateMultiGroupNUTSStepKernel::executeMC(
        ContextImpl& context, const MultiGroupNUTSIntegrator& integrator) {

    int K = numGroups;
    int numTrials = integrator.getNumMCTrials();
    double mcStep = integrator.getMCStepSize();

    fill(lastMCAcceptedPerGroup.begin(), lastMCAcceptedPerGroup.end(), 0);

    vector<int> mcEnabled = integrator.getAllGroupMCEnabled();
    mcEnabledHost = mcEnabled;
    mcEnabledBuffer.upload(mcEnabledHost);

    bool anyEligible = false;
    for (int k = 0; k < K; k++)
        if (mcEnabled[k]) { anyEligible = true; break; }
    if (!anyEligible) return;

    int numGroupAtoms = K * atomsPerGroup;
    int blockSize = 128;

    // Ensure energy extractors are wired
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

    int allGroupsMask = 0xFFFFFFFF;
    context.calcForcesAndEnergy(true, true, allGroupsMask);
    vector<double> peBaseline(K, 0.0);
    computeGroupPE(peBaseline);

    vector<double> kT(K);
    for (int k = 0; k < K; k++)
        kT[k] = BOLTZ * integrator.getGroupTemperature(k);

    for (int trial = 0; trial < numTrials; trial++) {
        // 1. Backup positions
        {
            CUdeviceptr posqPtr = cu.getPosq().getDevicePointer();
            CUdeviceptr backupPtr = positionsBackup.getDevicePointer();
            int numBlocks = min((numGroupAtoms + blockSize - 1) / blockSize,
                                cu.getNumThreadBlocks());
            void* bkArgs[] = { &posqPtr, &backupPtr, &numGroupAtoms };
            cu.executeKernel(backupPositionsKernel, bkArgs, numBlocks * blockSize, blockSize);
        }

        // 2. Compute COM
        {
            cu.clearBuffer(mcGroupCOMBuffer);
            CUdeviceptr posqPtr = cu.getPosq().getDevicePointer();
            CUdeviceptr velmPtr = cu.getVelm().getDevicePointer();
            CUdeviceptr comPtr = mcGroupCOMBuffer.getDevicePointer();
            int numBlocks = min((numGroupAtoms + blockSize - 1) / blockSize,
                                cu.getNumThreadBlocks());
            void* comArgs[] = { &posqPtr, &velmPtr, &comPtr, &atomsPerGroup, &numGroups };
            cu.executeKernel(mcComputeCOMKernel, comArgs, numBlocks * blockSize, blockSize);

            mcGroupCOMBuffer.download(mcGroupCOMHost);
            for (int k = 0; k < K; k++) {
                double totalMass = mcGroupCOMHost[k * 4 + 3];
                if (totalMass > 0) {
                    mcCOMHost[k * 3 + 0] = mcGroupCOMHost[k * 4 + 0] / totalMass;
                    mcCOMHost[k * 3 + 1] = mcGroupCOMHost[k * 4 + 1] / totalMass;
                    mcCOMHost[k * 3 + 2] = mcGroupCOMHost[k * 4 + 2] / totalMass;
                }
            }
            mcCOMBuffer.upload(mcCOMHost);
        }

        // 3. Generate rotation + translation on host
        for (int k = 0; k < K; k++) {
            if (!mcEnabled[k]) {
                fill(mcRotationHost.begin() + k * 9, mcRotationHost.begin() + k * 9 + 9, 0.0);
                mcRotationHost[k * 9 + 0] = 1.0;
                mcRotationHost[k * 9 + 4] = 1.0;
                mcRotationHost[k * 9 + 8] = 1.0;
                fill(mcTranslationHost.begin() + k * 3, mcTranslationHost.begin() + k * 3 + 3, 0.0);
                continue;
            }
            if (trial % 2 == 0)
                nutsGenerateRandomQuaternionRotation(rng, uniformDist, &mcRotationHost[k * 9]);
            else {
                fill(mcRotationHost.begin() + k * 9, mcRotationHost.begin() + k * 9 + 9, 0.0);
                mcRotationHost[k * 9 + 0] = 1.0;
                mcRotationHost[k * 9 + 4] = 1.0;
                mcRotationHost[k * 9 + 8] = 1.0;
            }
            mcTranslationHost[k * 3 + 0] = normalDist(rng) * mcStep;
            mcTranslationHost[k * 3 + 1] = normalDist(rng) * mcStep;
            mcTranslationHost[k * 3 + 2] = normalDist(rng) * mcStep;
        }

        // 4. Upload and apply move
        mcRotationBuffer.upload(mcRotationHost);
        mcTranslationBuffer.upload(mcTranslationHost);
        {
            CUdeviceptr posqPtr = cu.getPosq().getDevicePointer();
            CUdeviceptr comPtr = mcCOMBuffer.getDevicePointer();
            CUdeviceptr rotPtr = mcRotationBuffer.getDevicePointer();
            CUdeviceptr transPtr = mcTranslationBuffer.getDevicePointer();
            CUdeviceptr enabledPtr = mcEnabledBuffer.getDevicePointer();
            int numBlocks = min((numGroupAtoms + blockSize - 1) / blockSize,
                                cu.getNumThreadBlocks());
            void* moveArgs[] = {
                &posqPtr, &comPtr, &rotPtr, &transPtr, &enabledPtr,
                &atomsPerGroup, &numGroups
            };
            cu.executeKernel(mcApplyRigidBodyMoveKernel, moveArgs, numBlocks * blockSize, blockSize);
        }

        // 5. Recompute PE
        context.calcForcesAndEnergy(true, true, allGroupsMask);
        vector<double> peTrial(K, 0.0);
        computeGroupPE(peTrial);

        // 6. Metropolis accept/reject
        for (int k = 0; k < K; k++)
            mcAcceptedHost[k] = mcEnabled[k] ? 0 : 1;

        for (int k = 0; k < K; k++) {
            if (!mcEnabled[k]) continue;
            mcAttemptedTotal++;
            double dE = peTrial[k] - peBaseline[k];
            bool accept = (dE <= 0.0) || (uniformDist(rng) < exp(-dE / kT[k]));
            if (accept) {
                mcAcceptedTotal++;
                mcAcceptedHost[k] = 1;
                lastMCAcceptedPerGroup[k]++;
                peBaseline[k] = peTrial[k];
            }
        }

        // 7. Restore rejected
        mcAcceptedBuffer.upload(mcAcceptedHost);
        {
            CUdeviceptr posqPtr = cu.getPosq().getDevicePointer();
            CUdeviceptr velmPtr = cu.getVelm().getDevicePointer();
            CUdeviceptr backupPtr = positionsBackup.getDevicePointer();
            CUdeviceptr accPtr = mcAcceptedBuffer.getDevicePointer();
            int numBlocks = min((numGroupAtoms + blockSize - 1) / blockSize,
                                cu.getNumThreadBlocks());
            void* restoreArgs[] = {
                &posqPtr, &velmPtr, &backupPtr, &accPtr,
                &atomsPerGroup, &numGroups
            };
            cu.executeKernel(mcRestoreRejectedKernel, restoreArgs, numBlocks * blockSize, blockSize);
        }
    }
}

// ========== Main NUTS execute ==========

void CudaIntegrateMultiGroupNUTSStepKernel::execute(
        ContextImpl& context, const MultiGroupNUTSIntegrator& integrator,
        bool forcesAreValid) {

    int K = numGroups;
    int numGroupAtoms = K * atomsPerGroup;
    int paddedNumAtoms = cu.getPaddedNumAtoms();
    int blockSize = 128;

    // ===== Deferred GPU initialization =====
    if (!hasInitializedKernel) {
        cu.setAsCurrent();

        cu.getIntegrationUtilities().initRandomNumberGenerator(
            (unsigned int)integrator.getRandomNumberSeed());

        positionsBackup.initialize(cu, paddedNumAtoms, cu.getPosq().getElementSize(),
                                   "nutsPosBackup");
        slowForcesBackup.initialize<long long>(cu, 3 * paddedNumAtoms, "nutsSlowForces");
        groupKEBuffer.initialize<double>(cu, K, "nutsGroupKE");
        groupStepSizesBuffer.initialize<double>(cu, K, "nutsGroupStepSizes");
        groupKTBuffer.initialize<double>(cu, K, "nutsGroupKT");

        // NUTS-specific buffers
        xminusBuffer.initialize(cu, paddedNumAtoms, cu.getPosq().getElementSize(), "nutsXminus");
        xplusBuffer.initialize(cu, paddedNumAtoms, cu.getPosq().getElementSize(), "nutsXplus");
        vminusBuffer.initialize(cu, paddedNumAtoms, cu.getVelm().getElementSize(), "nutsVminus");
        vplusBuffer.initialize(cu, paddedNumAtoms, cu.getVelm().getElementSize(), "nutsVplus");
        candidateBuffer.initialize(cu, paddedNumAtoms, cu.getPosq().getElementSize(), "nutsCandidate");
        subtreeCandidateBuffer.initialize(cu, paddedNumAtoms, cu.getPosq().getElementSize(), "nutsSubtreeCandidate");
        activeBuffer.initialize<int>(cu, K, "nutsActive");
        directionBuffer.initialize<int>(cu, K, "nutsDirection");
        copyFlagBuffer.initialize<int>(cu, K, "nutsCopyFlag");
        uturnDotBuffer.initialize<double>(cu, 2 * K, "nutsUturnDot");
        divergentBuffer.initialize<int>(cu, K, "nutsDivergent");

        CUmodule module = cu.createModule(
            CudaGridForceKernelSources::commonHeaders +
            CudaGridForceKernelSources::multiGroupNUTSKernel);
        backupPositionsKernel = cu.getKernel(module, "nutsBackupPositions");
        drawMBVelocitiesFullKernel = cu.getKernel(module, "nutsDrawMBVelocitiesFull");
        drawMBVelocitiesPartialKernel = cu.getKernel(module, "nutsDrawMBVelocitiesPartial");
        computeGroupKEKernel = cu.getKernel(module, "nutsComputeGroupKE");
        velocityKickKernel = cu.getKernel(module, "nutsVelocityKick");
        positionDriftKernel = cu.getKernel(module, "nutsPositionDrift");
        copyForcesKernel = cu.getKernel(module, "nutsCopyForces");
        restoreEndpointKernel = cu.getKernel(module, "nutsRestoreEndpoint");
        saveEndpointKernel = cu.getKernel(module, "nutsSaveEndpoint");
        copyCandidatePosKernel = cu.getKernel(module, "nutsCopyCandidatePos");
        computeUTurnDotKernel = cu.getKernel(module, "nutsComputeUTurnDot");
        setFromCandidateKernel = cu.getKernel(module, "nutsSetFromCandidate");
        restoreDivergentKernel = cu.getKernel(module, "nutsRestoreDivergent");
        initializeTreeKernel = cu.getKernel(module, "nutsInitializeTree");

        // Phase 2: GPU-side tree building kernels
        gatherForceEnergiesKernel = cu.getKernel(module, "nutsGatherForceEnergies");
        leapfrogDecisionKernel = cu.getKernel(module, "nutsLeapfrogDecision");
        checkUTurnAndDeactivateKernel = cu.getKernel(module, "nutsCheckUTurnAndDeactivate");
        combineCandidatesKernel = cu.getKernel(module, "nutsCombineCandidates");
        setDirectionKernel = cu.getKernel(module, "nutsSetDirection");

        // Phase 2: GPU-side tree state buffers
        loguBuffer.initialize<double>(cu, K, "nutsLogu");
        H0Buffer.initialize<double>(cu, K, "nutsH0");
        peInitBuffer.initialize<double>(cu, K, "nutsPeInit");
        totalGroupPEBuffer.initialize<double>(cu, K, "nutsTotalGroupPE");
        subtreeNValidBuffer.initialize<int>(cu, K, "nutsSubtreeNValid");
        nValidBuffer.initialize<int>(cu, K, "nutsNValid");
        subtreeCandidatePEBuffer.initialize<double>(cu, K, "nutsSubtreeCandidatePE");
        candidatePEBuffer.initialize<double>(cu, K, "nutsCandidatePE");
        subtreeHasCandidateBuffer.initialize<int>(cu, K, "nutsSubtreeHasCandidate");
        anyActiveBuffer.initialize<int>(cu, 1, "nutsAnyActive");

        // MC kernels (shared with HMC — same kernel names)
        mcComputeCOMKernel = cu.getKernel(module, "hmcComputeGroupCOM");
        mcApplyRigidBodyMoveKernel = cu.getKernel(module, "hmcApplyRigidBodyMove");
        mcRestoreRejectedKernel = cu.getKernel(module, "hmcRestoreRejected");

        // MC GPU buffers
        mcGroupCOMBuffer.initialize<double>(cu, 4 * K, "nutsMCGroupCOM");
        mcEnabledBuffer.initialize<int>(cu, K, "nutsMCEnabled");
        mcRotationBuffer.initialize<double>(cu, 9 * K, "nutsMCRotation");
        mcTranslationBuffer.initialize<double>(cu, 3 * K, "nutsMCTranslation");
        mcCOMBuffer.initialize<double>(cu, 3 * K, "nutsMCCOM");
        mcAcceptedBuffer.initialize<int>(cu, K, "nutsMCAccepted");

        // Metric kernels (always loaded; buffers allocated on first use)
        assembleMetricKernel = cu.getKernel(module, "assembleMetricTensor");
        rmVelocityKickKernel = cu.getKernel(module, "rmVelocityKick");
        rmComputeGroupKEKernel = cu.getKernel(module, "rmComputeGroupKE");
        rmDrawMBVelocitiesFullKernel = cu.getKernel(module, "rmDrawMBVelocitiesFull");
        rmDrawMBVelocitiesPartialKernel = cu.getKernel(module, "rmDrawMBVelocitiesPartial");
        setIdentityMetricKernel = cu.getKernel(module, "setIdentityMetric");
        accumulateHessianKernel = cu.getKernel(module, "accumulateHessian");
        accumulateHessianWeightedKernel = cu.getKernel(module, "accumulateHessianWeighted");

        hasInitializedKernel = true;
    }

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

        // Collect GPU energy buffer pointers from ForceImpls
        forceEnergyPtrsHost.clear();
        for (ForceImpl* impl : context.getForceImpls()) {
            void* ptr = nullptr;
            if (auto* p = dynamic_cast<IsolatedBondedForceImpl*>(impl))
                ptr = p->getGroupEnergyDevicePointer();
            else if (auto* p = dynamic_cast<IsolatedNonbondedForceImpl*>(impl))
                ptr = p->getGroupEnergyDevicePointer();
            else if (auto* p = dynamic_cast<IsolatedGBSAForceImpl*>(impl))
                ptr = p->getGroupEnergyDevicePointer();
            else if (auto* p = dynamic_cast<IsolatedSiteForceImpl*>(impl))
                ptr = p->getGroupEnergyDevicePointer();
            else if (auto* p = dynamic_cast<GridForceImpl*>(impl))
                ptr = p->getGroupEnergyDevicePointer();
            else if (auto* p = dynamic_cast<GBSAGridForceImpl*>(impl))
                ptr = p->getGroupEnergyDevicePointer();

            if (ptr)
                forceEnergyPtrsHost.push_back((unsigned long long)ptr);
        }
        numEnergyForces = (int)forceEnergyPtrsHost.size();
        if (numEnergyForces > 0) {
            forceEnergyPtrsBuffer.initialize<unsigned long long>(
                cu, numEnergyForces, "nutsForceEnergyPtrs");
            forceEnergyPtrsBuffer.upload(forceEnergyPtrsHost);
        }
    }

    // Execute MC pre-step if configured
    if (integrator.getNumMCTrials() > 0) {
        executeMC(context, integrator);
        forcesAreValid = false;
    }

    // Upload per-group kT and base step sizes
    for (int k = 0; k < K; k++) {
        groupKTHost[k] = BOLTZ * integrator.getGroupTemperature(k);
        groupStepSizesHost[k] = integrator.getGroupStepSize(k);
    }
    groupKTBuffer.upload(groupKTHost);

    int allGroupsMask = 0xFFFFFFFF;
    double stabilityThreshold = integrator.getStabilityThreshold();
    int maxTreeDepth = integrator.getMaxTreeDepth();

    // ===== 1. Backup positions =====
    {
        CUdeviceptr posqPtr = cu.getPosq().getDevicePointer();
        CUdeviceptr backupPtr = positionsBackup.getDevicePointer();
        int numBlocks = min((numGroupAtoms + blockSize - 1) / blockSize,
                            cu.getNumThreadBlocks());
        void* args[] = { &posqPtr, &backupPtr, &numGroupAtoms };
        cu.executeKernel(backupPositionsKernel, args, numBlocks * blockSize, blockSize);
    }

    // ===== 1b. Assemble metric tensor (if EVERY_TRAJECTORY mode) =====
    bool useMetric = (integrator.getMetricType() != MultiGroupNUTSIntegrator::METRIC_IDENTITY &&
                      integrator.getMetricUpdateMode() == MultiGroupNUTSIntegrator::METRIC_UPDATE_EVERY_TRAJECTORY);
    if (useMetric) {
        // Need forces computed at current positions for Hessian
        if (!forcesAreValid) {
            context.calcForcesAndEnergy(true, true, allGroupsMask);
            forcesAreValid = true;
        }
        assembleMetric(context, integrator);
    }

    // ===== 2. Draw MB velocities =====
    {
        int numRandoms = numGroupAtoms;
        int randomIndex = cu.getIntegrationUtilities().prepareRandomNumbers(numRandoms);

        CUdeviceptr velmPtr = cu.getVelm().getDevicePointer();
        CUdeviceptr randomPtr = cu.getIntegrationUtilities().getRandom().getDevicePointer();
        CUdeviceptr ktPtr = groupKTBuffer.getDevicePointer();
        unsigned int randIdx = (unsigned int)randomIndex;

        int numBlocks = min((numGroupAtoms + blockSize - 1) / blockSize,
                            cu.getNumThreadBlocks());

        if (useMetric && choleskyBuffer.isInitialized()) {
            // Metric-aware draw: v ~ N(0, kT * G^{-1}) via Cholesky
            CUdeviceptr chPtr = choleskyBuffer.getDevicePointer();
            if (integrator.getMomentumRefreshMode() == MultiGroupNUTSIntegrator::FULL) {
                void* args[] = {
                    &velmPtr, &randomPtr, &randIdx,
                    &chPtr, &ktPtr, &atomsPerGroup, &numGroups
                };
                cu.executeKernel(rmDrawMBVelocitiesFullKernel, args,
                                 numBlocks * blockSize, blockSize);
            } else {
                double theta = integrator.getPartialRefreshAngle();
                float cosThetaF = (float)cos(theta);
                float sinThetaF = (float)sin(theta);
                double cosThetaD = cos(theta);
                double sinThetaD = sin(theta);
                void* cosThetaPtr;
                void* sinThetaPtr;
                if (cu.getUseDoublePrecision() || cu.getUseMixedPrecision()) {
                    cosThetaPtr = &cosThetaD;
                    sinThetaPtr = &sinThetaD;
                } else {
                    cosThetaPtr = &cosThetaF;
                    sinThetaPtr = &sinThetaF;
                }
                void* args[] = {
                    &velmPtr, &randomPtr, &randIdx,
                    &chPtr, &ktPtr, &atomsPerGroup, &numGroups,
                    cosThetaPtr, sinThetaPtr
                };
                cu.executeKernel(rmDrawMBVelocitiesPartialKernel, args,
                                 numBlocks * blockSize, blockSize);
            }
        } else {
            // Standard draw: v ~ N(0, kT / mass)
            if (integrator.getMomentumRefreshMode() == MultiGroupNUTSIntegrator::FULL) {
                void* args[] = {
                    &velmPtr, &randomPtr, &randIdx,
                    &ktPtr, &atomsPerGroup, &numGroups
                };
                cu.executeKernel(drawMBVelocitiesFullKernel, args,
                                 numBlocks * blockSize, blockSize);
            } else {
                double theta = integrator.getPartialRefreshAngle();
                float cosThetaF = (float)cos(theta);
                float sinThetaF = (float)sin(theta);
                double cosThetaD = cos(theta);
                double sinThetaD = sin(theta);
                void* cosThetaPtr;
                void* sinThetaPtr;
                if (cu.getUseDoublePrecision() || cu.getUseMixedPrecision()) {
                    cosThetaPtr = &cosThetaD;
                    sinThetaPtr = &sinThetaD;
                } else {
                    cosThetaPtr = &cosThetaF;
                    sinThetaPtr = &sinThetaF;
                }
                void* args[] = {
                    &velmPtr, &randomPtr, &randIdx,
                    &ktPtr, &atomsPerGroup, &numGroups,
                    cosThetaPtr, sinThetaPtr
                };
                cu.executeKernel(drawMBVelocitiesPartialKernel, args,
                                 numBlocks * blockSize, blockSize);
            }
        }
    }

    // ===== 3. Compute initial H_0 per group =====
    vector<double> keInit;
    computeGroupKE(context, keInit);

    if (!forcesAreValid)
        context.calcForcesAndEnergy(true, true, allGroupsMask);

    vector<double> peInit(K, 0.0);
    computeGroupPE(peInit);

    // H_0[k] = PE[k] + KE[k], in units of kT for slice sampling
    // logu[k] = -H_0[k]/kT[k] - Exp(1)  (log of slice variable)
    vector<double> H0(K);
    vector<double> logu(K);
    for (int k = 0; k < K; k++) {
        H0[k] = peInit[k] + keInit[k];
        logu[k] = -H0[k] / groupKTHost[k] - exponentialDist(rng);
    }

    // ===== 4. Initialize tree =====
    {
        CUdeviceptr posqPtr = cu.getPosq().getDevicePointer();
        CUdeviceptr velmPtr = cu.getVelm().getDevicePointer();
        CUdeviceptr xmPtr = xminusBuffer.getDevicePointer();
        CUdeviceptr xpPtr = xplusBuffer.getDevicePointer();
        CUdeviceptr candPtr = candidateBuffer.getDevicePointer();
        CUdeviceptr vmPtr = vminusBuffer.getDevicePointer();
        CUdeviceptr vpPtr = vplusBuffer.getDevicePointer();

        int numBlocks = min((numGroupAtoms + blockSize - 1) / blockSize,
                            cu.getNumThreadBlocks());
        void* args[] = {
            &posqPtr, &velmPtr,
            &xmPtr, &xpPtr, &candPtr, &vmPtr, &vpPtr,
            &numGroupAtoms
        };
        cu.executeKernel(initializeTreeKernel, args, numBlocks * blockSize, blockSize);
    }

    // Per-group tree state — upload to GPU
    fill(activeHost.begin(), activeHost.end(), 1);
    fill(divergentHost.begin(), divergentHost.end(), 0);
    fill(lastTreeDepths.begin(), lastTreeDepths.end(), 0);

    // Upload GPU-side tree state
    loguBuffer.upload(logu);
    H0Buffer.upload(H0);
    peInitBuffer.upload(peInit);
    {
        vector<int> nValidInit(K, 1);
        nValidBuffer.upload(nValidInit);
    }
    candidatePEBuffer.upload(peInit);  // initial candidate PE = peInit
    activeBuffer.upload(activeHost);
    cu.clearBuffer(divergentBuffer);

    // Toggle skip-download on all forces (only when GPU tree building is enabled)
    auto toggleSkipDownload = [&context](bool skip) {
        for (ForceImpl* impl : context.getForceImpls()) {
            if (auto* p = dynamic_cast<IsolatedBondedForceImpl*>(impl))
                p->setSkipGroupEnergyDownload(skip);
            else if (auto* p = dynamic_cast<IsolatedNonbondedForceImpl*>(impl))
                p->setSkipGroupEnergyDownload(skip);
            else if (auto* p = dynamic_cast<IsolatedGBSAForceImpl*>(impl))
                p->setSkipGroupEnergyDownload(skip);
            else if (auto* p = dynamic_cast<IsolatedSiteForceImpl*>(impl))
                p->setSkipGroupEnergyDownload(skip);
            else if (auto* p = dynamic_cast<GridForceImpl*>(impl))
                p->setSkipGroupEnergyDownload(skip);
            else if (auto* p = dynamic_cast<GBSAGridForceImpl*>(impl))
                p->setSkipGroupEnergyDownload(skip);
        }
    };

    bool useGpuTree = integrator.getGpuTreeBuilding() && numEnergyForces > 0;

    if (useGpuTree)
        toggleSkipDownload(true);

    // ===== 5. Tree doubling loop =====
    // Host-side subtree state (used only in host path)
    vector<int> subtreeNValid(K, 0);
    vector<int> nValid(K, 1);
    vector<double> candidatePE(K);
    for (int k = 0; k < K; k++) candidatePE[k] = peInit[k];

    for (int depth = 0; depth < maxTreeDepth; depth++) {
        // Check if any groups still active (from host-side active state)
        bool anyActive = false;
        for (int k = 0; k < K; k++)
            if (activeHost[k]) { anyActive = true; break; }
        if (!anyActive) break;

        // Choose random direction per active group (host-side, per-depth)
        for (int k = 0; k < K; k++) {
            if (activeHost[k])
                directionHost[k] = (uniformDist(rng) < 0.5) ? -1 : 1;
            else
                directionHost[k] = 1;
        }
        directionBuffer.upload(directionHost);

        // Set up signed dt for this depth's direction
        for (int k = 0; k < K; k++)
            groupStepSizesHost[k] = directionHost[k] * fabs(integrator.getGroupStepSize(k));
        groupStepSizesBuffer.upload(groupStepSizesHost);

        // Restore positions/velocities from appropriate tree endpoint
        {
            CUdeviceptr posqPtr = cu.getPosq().getDevicePointer();
            CUdeviceptr velmPtr = cu.getVelm().getDevicePointer();
            CUdeviceptr xmPtr = xminusBuffer.getDevicePointer();
            CUdeviceptr xpPtr = xplusBuffer.getDevicePointer();
            CUdeviceptr vmPtr = vminusBuffer.getDevicePointer();
            CUdeviceptr vpPtr = vplusBuffer.getDevicePointer();
            CUdeviceptr actPtr = activeBuffer.getDevicePointer();
            CUdeviceptr dirPtr = directionBuffer.getDevicePointer();

            int numBlocks = min((numGroupAtoms + blockSize - 1) / blockSize,
                                cu.getNumThreadBlocks());
            void* args[] = {
                &posqPtr, &velmPtr,
                &xmPtr, &xpPtr, &vmPtr, &vpPtr,
                &actPtr, &dirPtr,
                &atomsPerGroup, &numGroups
            };
            cu.executeKernel(restoreEndpointKernel, args, numBlocks * blockSize, blockSize);
        }

        // Compute forces before stepping (needed for first leapfrog kick)
        context.calcForcesAndEnergy(true, false, allGroupsMask);

        int numStepsThisDepth = 1 << depth;

        if (useGpuTree) {
            // ===== GPU PATH: ZERO CPU-GPU SYNCS IN INNER LOOP =====
            // Clear subtree state on GPU
            cu.clearBuffer(subtreeNValidBuffer);
            cu.clearBuffer(subtreeHasCandidateBuffer);

            // Generate RNG seed for this depth level
            unsigned int depthRngSeed = (unsigned int)rng();

            for (int step = 0; step < numStepsThisDepth; step++) {
                // Leapfrog step (forces+energy computed on GPU, downloads skipped)
                respaLeapfrogStep(context, integrator, true);

                // Compute KE on GPU (no download)
                cu.clearBuffer(groupKEBuffer);
                {
                    CUdeviceptr velmPtr = cu.getVelm().getDevicePointer();
                    CUdeviceptr kePtr = groupKEBuffer.getDevicePointer();
                    int totalAtoms = K * atomsPerGroup;
                    int numBlocks = min((totalAtoms + blockSize - 1) / blockSize,
                                        cu.getNumThreadBlocks());
                    if (useMetric && metricBuffer.isInitialized()) {
                        CUdeviceptr mPtr = metricBuffer.getDevicePointer();
                        void* args[] = { &velmPtr, &mPtr, &kePtr, &atomsPerGroup, &numGroups };
                        cu.executeKernel(rmComputeGroupKEKernel, args,
                                         numBlocks * blockSize, blockSize);
                    } else {
                        void* args[] = { &velmPtr, &kePtr, &atomsPerGroup, &numGroups };
                        cu.executeKernel(computeGroupKEKernel, args,
                                         numBlocks * blockSize, blockSize);
                    }
                }

                // Gather PE from all force energy buffers on GPU
                {
                    CUdeviceptr pePtr = totalGroupPEBuffer.getDevicePointer();
                    CUdeviceptr fePtrsPtr = forceEnergyPtrsBuffer.getDevicePointer();
                    int numBlocks = min((K + blockSize - 1) / blockSize,
                                        cu.getNumThreadBlocks());
                    void* args[] = { &pePtr, &fePtrsPtr, &numEnergyForces, &numGroups };
                    cu.executeKernel(gatherForceEnergiesKernel, args,
                                     numBlocks * blockSize, blockSize);
                }

                // Fused decision: divergence + slice + candidate selection (all GPU)
                {
                    CUdeviceptr pePtr = totalGroupPEBuffer.getDevicePointer();
                    CUdeviceptr kePtr = groupKEBuffer.getDevicePointer();
                    CUdeviceptr loguPtr = loguBuffer.getDevicePointer();
                    CUdeviceptr h0Ptr = H0Buffer.getDevicePointer();
                    CUdeviceptr peInitPtr = peInitBuffer.getDevicePointer();
                    CUdeviceptr ktPtr = groupKTBuffer.getDevicePointer();
                    CUdeviceptr actPtr = activeBuffer.getDevicePointer();
                    CUdeviceptr divPtr = divergentBuffer.getDevicePointer();
                    CUdeviceptr snvPtr = subtreeNValidBuffer.getDevicePointer();
                    CUdeviceptr scpePtr = subtreeCandidatePEBuffer.getDevicePointer();
                    CUdeviceptr shcPtr = subtreeHasCandidateBuffer.getDevicePointer();
                    CUdeviceptr cfPtr = copyFlagBuffer.getDevicePointer();

                    int numBlocks = min((K + blockSize - 1) / blockSize,
                                        cu.getNumThreadBlocks());
                    void* args[] = {
                        &pePtr, &kePtr, &loguPtr, &h0Ptr, &peInitPtr, &ktPtr,
                        &actPtr, &divPtr, &snvPtr, &scpePtr, &shcPtr, &cfPtr,
                        &stabilityThreshold, &numGroups, &depthRngSeed, &step
                    };
                    cu.executeKernel(leapfrogDecisionKernel, args,
                                     numBlocks * blockSize, blockSize);
                }

                // Copy positions to subtree candidate buffer (reads GPU copyFlag)
                {
                    CUdeviceptr posqPtr = cu.getPosq().getDevicePointer();
                    CUdeviceptr candPtr = subtreeCandidateBuffer.getDevicePointer();
                    CUdeviceptr flagPtr = copyFlagBuffer.getDevicePointer();

                    int numBlocks = min((numGroupAtoms + blockSize - 1) / blockSize,
                                        cu.getNumThreadBlocks());
                    void* args[] = {
                        &posqPtr, &candPtr, &flagPtr,
                        &atomsPerGroup, &numGroups
                    };
                    cu.executeKernel(copyCandidatePosKernel, args,
                                     numBlocks * blockSize, blockSize);
                }
            }

            // Save current positions/velocities to appropriate endpoint
            {
                CUdeviceptr posqPtr = cu.getPosq().getDevicePointer();
                CUdeviceptr velmPtr = cu.getVelm().getDevicePointer();
                CUdeviceptr xmPtr = xminusBuffer.getDevicePointer();
                CUdeviceptr xpPtr = xplusBuffer.getDevicePointer();
                CUdeviceptr vmPtr = vminusBuffer.getDevicePointer();
                CUdeviceptr vpPtr = vplusBuffer.getDevicePointer();
                CUdeviceptr actPtr = activeBuffer.getDevicePointer();
                CUdeviceptr dirPtr = directionBuffer.getDevicePointer();

                int numBlocks = min((numGroupAtoms + blockSize - 1) / blockSize,
                                    cu.getNumThreadBlocks());
                void* args[] = {
                    &posqPtr, &velmPtr,
                    &xmPtr, &xpPtr, &vmPtr, &vpPtr,
                    &actPtr, &dirPtr,
                    &atomsPerGroup, &numGroups
                };
                cu.executeKernel(saveEndpointKernel, args, numBlocks * blockSize, blockSize);
            }

            // Combine subtree candidate with main candidate (GPU-side Metropolis)
            {
                unsigned int combineRngSeed = (unsigned int)rng();

                CUdeviceptr snvPtr = subtreeNValidBuffer.getDevicePointer();
                CUdeviceptr nvPtr = nValidBuffer.getDevicePointer();
                CUdeviceptr shcPtr = subtreeHasCandidateBuffer.getDevicePointer();
                CUdeviceptr scpePtr = subtreeCandidatePEBuffer.getDevicePointer();
                CUdeviceptr cpePtr = candidatePEBuffer.getDevicePointer();
                CUdeviceptr cfPtr = copyFlagBuffer.getDevicePointer();
                CUdeviceptr actPtr = activeBuffer.getDevicePointer();

                int numBlocks = min((K + blockSize - 1) / blockSize,
                                    cu.getNumThreadBlocks());
                void* args[] = {
                    &snvPtr, &nvPtr, &shcPtr, &scpePtr, &cpePtr, &cfPtr, &actPtr,
                    &numGroups, &combineRngSeed
                };
                cu.executeKernel(combineCandidatesKernel, args,
                                 numBlocks * blockSize, blockSize);
            }

            // Copy accepted subtree candidates to main candidate buffer
            {
                CUdeviceptr srcPtr = subtreeCandidateBuffer.getDevicePointer();
                CUdeviceptr dstPtr = candidateBuffer.getDevicePointer();
                CUdeviceptr flagPtr = copyFlagBuffer.getDevicePointer();

                int numBlocks = min((numGroupAtoms + blockSize - 1) / blockSize,
                                    cu.getNumThreadBlocks());
                void* args[] = {
                    &srcPtr, &dstPtr, &flagPtr,
                    &atomsPerGroup, &numGroups
                };
                cu.executeKernel(copyCandidatePosKernel, args,
                                 numBlocks * blockSize, blockSize);
            }

        } else {
            // ===== HOST PATH: per-step CPU-GPU sync (for validation) =====
            fill(subtreeNValid.begin(), subtreeNValid.end(), 0);
            vector<int> subtreeHasCandidate(K, 0);
            vector<double> subtreeCandidatePE(K, 0.0);

            for (int step = 0; step < numStepsThisDepth; step++) {
                respaLeapfrogStep(context, integrator, true);

                // Download KE and PE to host
                vector<double> stepKE;
                computeGroupKE(context, stepKE);
                vector<double> stepPE(K, 0.0);
                computeGroupPE(stepPE);

                // Per-group decision logic on host
                fill(copyFlagHost.begin(), copyFlagHost.end(), 0);
                for (int k = 0; k < K; k++) {
                    if (!activeHost[k]) continue;

                    double Hk = stepPE[k] + stepKE[k];
                    double logPk = -Hk / groupKTHost[k];

                    // Divergence check
                    double deltaPE = stepPE[k] - peInit[k];
                    double deltaH = Hk - H0[k];
                    if (fabs(deltaPE) / groupKTHost[k] > stabilityThreshold &&
                        fabs(deltaH) / groupKTHost[k] > stabilityThreshold) {
                        activeHost[k] = 0;
                        divergentHost[k] = 1;
                        continue;
                    }

                    // Slice check + candidate selection
                    if (logPk > logu[k]) {
                        subtreeNValid[k]++;
                        double u = uniformDist(rng);
                        if (u < 1.0 / subtreeNValid[k]) {
                            subtreeCandidatePE[k] = stepPE[k];
                            subtreeHasCandidate[k] = 1;
                            copyFlagHost[k] = 1;
                        }
                    }
                }

                // Upload active and divergent state
                activeBuffer.upload(activeHost);
                divergentBuffer.upload(divergentHost);

                // Copy positions to subtree candidate buffer
                copyFlagBuffer.upload(copyFlagHost);
                {
                    CUdeviceptr posqPtr = cu.getPosq().getDevicePointer();
                    CUdeviceptr candPtr = subtreeCandidateBuffer.getDevicePointer();
                    CUdeviceptr flagPtr = copyFlagBuffer.getDevicePointer();

                    int numBlocks = min((numGroupAtoms + blockSize - 1) / blockSize,
                                        cu.getNumThreadBlocks());
                    void* args[] = {
                        &posqPtr, &candPtr, &flagPtr,
                        &atomsPerGroup, &numGroups
                    };
                    cu.executeKernel(copyCandidatePosKernel, args,
                                     numBlocks * blockSize, blockSize);
                }
            }

            // Save endpoint
            {
                CUdeviceptr posqPtr = cu.getPosq().getDevicePointer();
                CUdeviceptr velmPtr = cu.getVelm().getDevicePointer();
                CUdeviceptr xmPtr = xminusBuffer.getDevicePointer();
                CUdeviceptr xpPtr = xplusBuffer.getDevicePointer();
                CUdeviceptr vmPtr = vminusBuffer.getDevicePointer();
                CUdeviceptr vpPtr = vplusBuffer.getDevicePointer();
                CUdeviceptr actPtr = activeBuffer.getDevicePointer();
                CUdeviceptr dirPtr = directionBuffer.getDevicePointer();

                int numBlocks = min((numGroupAtoms + blockSize - 1) / blockSize,
                                    cu.getNumThreadBlocks());
                void* args[] = {
                    &posqPtr, &velmPtr,
                    &xmPtr, &xpPtr, &vmPtr, &vpPtr,
                    &actPtr, &dirPtr,
                    &atomsPerGroup, &numGroups
                };
                cu.executeKernel(saveEndpointKernel, args, numBlocks * blockSize, blockSize);
            }

            // Combine subtree candidate with main candidate (host-side)
            fill(copyFlagHost.begin(), copyFlagHost.end(), 0);
            for (int k = 0; k < K; k++) {
                if (!activeHost[k] || !subtreeHasCandidate[k]) continue;
                int totalN = nValid[k] + subtreeNValid[k];
                double acceptProb = (double)subtreeNValid[k] / totalN;
                if (uniformDist(rng) < acceptProb) {
                    copyFlagHost[k] = 1;
                    candidatePE[k] = subtreeCandidatePE[k];
                }
                nValid[k] = totalN;
            }
            copyFlagBuffer.upload(copyFlagHost);
            {
                CUdeviceptr srcPtr = subtreeCandidateBuffer.getDevicePointer();
                CUdeviceptr dstPtr = candidateBuffer.getDevicePointer();
                CUdeviceptr flagPtr = copyFlagBuffer.getDevicePointer();

                int numBlocks = min((numGroupAtoms + blockSize - 1) / blockSize,
                                    cu.getNumThreadBlocks());
                void* args[] = {
                    &srcPtr, &dstPtr, &flagPtr,
                    &atomsPerGroup, &numGroups
                };
                cu.executeKernel(copyCandidatePosKernel, args,
                                 numBlocks * blockSize, blockSize);
            }
        }

        // U-turn check (GPU compute + GPU deactivation) — shared by both paths
        {
            CUdeviceptr xmPtr = xminusBuffer.getDevicePointer();
            CUdeviceptr xpPtr = xplusBuffer.getDevicePointer();
            CUdeviceptr vmPtr = vminusBuffer.getDevicePointer();
            CUdeviceptr vpPtr = vplusBuffer.getDevicePointer();
            CUdeviceptr actPtr = activeBuffer.getDevicePointer();
            CUdeviceptr dotPtr = uturnDotBuffer.getDevicePointer();

            int utBlockSize = min(32, atomsPerGroup);
            int reductionSize = 1;
            while (reductionSize < utBlockSize) reductionSize <<= 1;
            utBlockSize = max(reductionSize, 32);

            int sharedMem = 2 * utBlockSize * sizeof(double);
            void* args[] = {
                &xmPtr, &xpPtr, &vmPtr, &vpPtr,
                &actPtr, &dotPtr,
                &atomsPerGroup, &numGroups
            };
            cu.executeKernel(computeUTurnDotKernel, args,
                             K * utBlockSize, utBlockSize, sharedMem);
        }

        if (useGpuTree) {
            // GPU-side U-turn deactivation + anyActive reduction
            cu.clearBuffer(anyActiveBuffer);

            CUdeviceptr dotPtr = uturnDotBuffer.getDevicePointer();
            CUdeviceptr actPtr = activeBuffer.getDevicePointer();
            CUdeviceptr anyActPtr = anyActiveBuffer.getDevicePointer();

            int numBlocks = min((K + blockSize - 1) / blockSize,
                                cu.getNumThreadBlocks());
            void* args[] = { &dotPtr, &actPtr, &anyActPtr, &numGroups };
            cu.executeKernel(checkUTurnAndDeactivateKernel, args,
                             numBlocks * blockSize, blockSize);

            // SINGLE SYNC per depth: download active for tree depth tracking
            activeBuffer.download(activeHost);
        } else {
            // Host-side U-turn check
            uturnDotBuffer.download(uturnDotHost);
            for (int k = 0; k < K; k++) {
                if (!activeHost[k]) continue;
                if (uturnDotHost[2*k] < 0 || uturnDotHost[2*k+1] < 0)
                    activeHost[k] = 0;
            }
            activeBuffer.upload(activeHost);
        }

        // Record tree depth for groups that are still active or just terminated
        for (int k = 0; k < K; k++) {
            if (lastTreeDepths[k] < depth + 1 && (activeHost[k] || depth == 0))
                lastTreeDepths[k] = depth + 1;
            if (!activeHost[k] && lastTreeDepths[k] == 0)
                lastTreeDepths[k] = depth + 1;
        }
    }

    // Reset skip-download flag if we enabled it
    if (useGpuTree)
        toggleSkipDownload(false);

    // Download divergent flags for statistics
    if (useGpuTree)
        divergentBuffer.download(divergentHost);

    // Record final tree depths for groups still active at max depth
    for (int k = 0; k < K; k++) {
        if (activeHost[k])
            lastTreeDepths[k] = maxTreeDepth;
        if (lastTreeDepths[k] == 0)
            lastTreeDepths[k] = 1;  // minimum depth
    }

    // ===== 6. Set positions from candidates, zero velocities =====
    {
        CUdeviceptr posqPtr = cu.getPosq().getDevicePointer();
        CUdeviceptr velmPtr = cu.getVelm().getDevicePointer();
        CUdeviceptr candPtr = candidateBuffer.getDevicePointer();

        int numBlocks = min((numGroupAtoms + blockSize - 1) / blockSize,
                            cu.getNumThreadBlocks());
        void* args[] = {
            &posqPtr, &velmPtr, &candPtr, &numGroupAtoms
        };
        cu.executeKernel(setFromCandidateKernel, args, numBlocks * blockSize, blockSize);
    }

    // ===== 7. Restore divergent groups to backup positions =====
    // (divergentBuffer already on GPU from leapfrogDecision kernel)
    {
        CUdeviceptr posqPtr = cu.getPosq().getDevicePointer();
        CUdeviceptr velmPtr = cu.getVelm().getDevicePointer();
        CUdeviceptr backupPtr = positionsBackup.getDevicePointer();
        CUdeviceptr divPtr = divergentBuffer.getDevicePointer();

        int numBlocks = min((numGroupAtoms + blockSize - 1) / blockSize,
                            cu.getNumThreadBlocks());
        void* args[] = {
            &posqPtr, &velmPtr, &backupPtr, &divPtr,
            &atomsPerGroup, &numGroups
        };
        cu.executeKernel(restoreDivergentKernel, args, numBlocks * blockSize, blockSize);
    }

    // ===== 8. Update statistics =====
    for (int k = 0; k < K; k++) {
        trialCounts[k]++;
        cumulativeTreeDepths[k] += lastTreeDepths[k];

        if (divergentHost[k]) {
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

}  // namespace GridForcePlugin
