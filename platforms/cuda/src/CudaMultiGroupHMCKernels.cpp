/* -------------------------------------------------------------------------- *
 *                              OpenMMGridForce                               *
 * -------------------------------------------------------------------------- *
 * CUDA implementation of MultiGroupHMCIntegrator kernel.                    *
 * -------------------------------------------------------------------------- */

#include "CudaMultiGroupHMCKernels.h"
#include "CudaGridForceKernelSources.h"
#include "GridForce.h"
#include "IsolatedBondedForce.h"
#include "IsolatedNonbondedForce.h"
#include "IsolatedGBSAForce.h"
#include "IsolatedSiteForce.h"
#include "GBSAGridForce.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/OpenMMException.h"
#include "openmm/cuda/CudaContext.h"
#include "openmm/cuda/CudaArray.h"
#include "openmm/cuda/CudaIntegrationUtilities.h"
#include <cmath>
#include <algorithm>

using namespace OpenMM;
using namespace std;

// kJ/(mol·K) — hardcoded because SimTKOpenMMRealType.h (which defines the BOLTZ
// macro) is not available in the CUDA platform headers.  Value matches RGAS/KILO.
static const double BOLTZ = 0.008314462618;

namespace GridForcePlugin {

void CudaIntegrateMultiGroupHMCStepKernel::initialize(
        const System& system, const MultiGroupHMCIntegrator& integrator) {

    // Critical: finalize CUDA context setup (all OpenMM integrator kernels must call this)
    cu.initializeContexts();

    numGroups = integrator.getNumGroups();
    atomsPerGroup = integrator.getAtomsPerGroup();
    numParticles = system.getNumParticles();

    // Resize host-side vectors
    groupKEHost.resize(numGroups, 0.0);
    groupStepSizesHost.resize(numGroups, 0.0);
    groupKTHost.resize(numGroups, 0.0);
    acceptedHost.resize(numGroups, 0);

    lastAccepted.resize(numGroups, 0);
    lastDeltaH.resize(numGroups, 0.0);
    acceptCounts.resize(numGroups, 0);
    trialCounts.resize(numGroups, 0);
    stabilityRejectCounts.resize(numGroups, 0);

    // Seed host RNG
    rng.seed((unsigned int)integrator.getRandomNumberSeed());

    // MC counters
    mcAttemptedTotal = 0;
    mcAcceptedTotal = 0;
    lastMCAcceptedPerGroup.resize(numGroups, 0);

    // Metric state
    metricInitialized = false;
    conditionNumbersHost.resize(numGroups, 1.0f);
    gridForceImpls.clear();
    bondedForceImpl = nullptr;

    // MC host vectors
    groupCOMHost.resize(4 * numGroups, 0.0);
    mcEnabledHost.resize(numGroups, 0);
    mcRotationHost.resize(9 * numGroups, 0.0);
    mcTranslationHost.resize(3 * numGroups, 0.0);
    mcCOMHost.resize(3 * numGroups, 0.0);

    hasInitializedKernel = false;
}

void CudaIntegrateMultiGroupHMCStepKernel::computeGroupPE(
        vector<double>& groupPE) const {
    fill(groupPE.begin(), groupPE.end(), 0.0);
    for (auto& extractor : groupEnergyExtractors) {
        vector<double> energies = extractor();
        for (int k = 0; k < numGroups && k < (int)energies.size(); k++)
            groupPE[k] += energies[k];
    }
}

void CudaIntegrateMultiGroupHMCStepKernel::computeGroupKE(
        ContextImpl& context, vector<double>& groupKE) {

    // Zero the buffer
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

void CudaIntegrateMultiGroupHMCStepKernel::launchKick(
        CUdeviceptr forcePtr, double scaleFactor) {

    CUdeviceptr velmPtr = cu.getVelm().getDevicePointer();
    CUdeviceptr dtPtr = groupStepSizesBuffer.getDevicePointer();
    int paddedNumAtoms = cu.getPaddedNumAtoms();

    int totalAtoms = numGroups * atomsPerGroup;
    int blockSize = 128;
    int numBlocks = min((totalAtoms + blockSize - 1) / blockSize,
                        cu.getNumThreadBlocks());

    // scale is passed as mixed type
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
        CUdeviceptr activePtr = activeBuffer.getDevicePointer();
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
            &velmPtr, &forcePtr, &dtPtr,
            &atomsPerGroup, &numGroups,
            scalePtr,
            &paddedNumAtoms
        };
        cu.executeKernel(velocityKickKernel, args, numBlocks * blockSize, blockSize);
    }
}

void CudaIntegrateMultiGroupHMCStepKernel::launchDrift() {

    CUdeviceptr posqPtr = cu.getPosq().getDevicePointer();
    CUdeviceptr velmPtr = cu.getVelm().getDevicePointer();
    CUdeviceptr dtPtr = groupStepSizesBuffer.getDevicePointer();

    int totalAtoms = numGroups * atomsPerGroup;
    int blockSize = 128;
    int numBlocks = min((totalAtoms + blockSize - 1) / blockSize,
                        cu.getNumThreadBlocks());

    void* args[] = {
        &posqPtr, &velmPtr, &dtPtr,
        &atomsPerGroup, &numGroups
    };
    cu.executeKernel(positionDriftKernel, args, numBlocks * blockSize, blockSize);
}

void CudaIntegrateMultiGroupHMCStepKernel::execute(
        ContextImpl& context, const MultiGroupHMCIntegrator& integrator,
        bool forcesAreValid) {

    int K = numGroups;

    // Deferred GPU initialization (first execute call)
    if (!hasInitializedKernel) {
        cu.setAsCurrent();
        int paddedNumAtoms = cu.getPaddedNumAtoms();

        cu.getIntegrationUtilities().initRandomNumberGenerator(
            (unsigned int)integrator.getRandomNumberSeed());

        positionsBackup.initialize(cu, paddedNumAtoms, cu.getPosq().getElementSize(),
                                   "hmcPosBackup");
        slowForcesBackup.initialize<long long>(cu, 3 * paddedNumAtoms, "hmcSlowForces");
        groupKEBuffer.initialize<double>(cu, K, "hmcGroupKE");
        groupStepSizesBuffer.initialize<double>(cu, K, "hmcGroupStepSizes");
        groupKTBuffer.initialize<double>(cu, K, "hmcGroupKT");
        acceptedBuffer.initialize<int>(cu, K, "hmcAccepted");

        CUmodule module = cu.createModule(CudaGridForceKernelSources::gridForceKernel);
        backupPositionsKernel = cu.getKernel(module, "hmcBackupPositions");
        drawMBVelocitiesFullKernel = cu.getKernel(module, "hmcDrawMBVelocitiesFull");
        drawMBVelocitiesPartialKernel = cu.getKernel(module, "hmcDrawMBVelocitiesPartial");
        computeGroupKEKernel = cu.getKernel(module, "hmcComputeGroupKE");
        velocityKickKernel = cu.getKernel(module, "hmcVelocityKick");
        positionDriftKernel = cu.getKernel(module, "hmcPositionDrift");
        copyForcesKernel = cu.getKernel(module, "hmcCopyForces");
        restoreRejectedKernel = cu.getKernel(module, "hmcRestoreRejected");

        // Metric kernels (always loaded; buffers allocated on first use in assembleMetric)
        assembleMetricKernel = cu.getKernel(module, "assembleMetricTensor");
        rmVelocityKickKernel = cu.getKernel(module, "rmVelocityKick");
        rmComputeGroupKEKernel = cu.getKernel(module, "rmComputeGroupKE");
        rmDrawMBVelocitiesFullKernel = cu.getKernel(module, "rmDrawMBVelocitiesFull");
        rmDrawMBVelocitiesPartialKernel = cu.getKernel(module, "rmDrawMBVelocitiesPartial");
        setIdentityMetricKernel = cu.getKernel(module, "setIdentityMetric");
        accumulateHessianKernel = cu.getKernel(module, "accumulateHessian");
        accumulateHessianWeightedKernel = cu.getKernel(module, "accumulateHessianWeighted");

        // Active buffer: all-ones for HMC (rmVelocityKick requires active mask)
        activeBuffer.initialize<int>(cu, K, "hmcActive");
        vector<int> allActive(K, 1);
        activeBuffer.upload(allActive);

        // MC kernels
        mcComputeCOMKernel = cu.getKernel(module, "hmcComputeGroupCOM");
        mcApplyRigidBodyMoveKernel = cu.getKernel(module, "hmcApplyRigidBodyMove");

        // MC GPU buffers
        groupCOMBuffer.initialize<double>(cu, 4 * K, "hmcGroupCOM");
        mcEnabledBuffer.initialize<int>(cu, K, "hmcMCEnabled");
        mcRotationBuffer.initialize<double>(cu, 9 * K, "hmcMCRotation");
        mcTranslationBuffer.initialize<double>(cu, 3 * K, "hmcMCTranslation");
        mcCOMBuffer.initialize<double>(cu, 3 * K, "hmcMCCOM");

        hasInitializedKernel = true;
    }

    // On first call, wire up group energy extractors (same as Reference).
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

    // Execute MC pre-step if configured
    if (integrator.getNumMCTrials() > 0) {
        executeMC(context, integrator);
        // MC may have changed positions, so forces are stale
        forcesAreValid = false;
    }

    // Upload per-group kT and step sizes
    for (int k = 0; k < K; k++) {
        groupKTHost[k] = BOLTZ * integrator.getGroupTemperature(k);
        groupStepSizesHost[k] = integrator.getGroupStepSize(k);
    }
    groupKTBuffer.upload(groupKTHost);
    groupStepSizesBuffer.upload(groupStepSizesHost);

    int numGroupAtoms = K * atomsPerGroup;
    int paddedNumAtoms = cu.getPaddedNumAtoms();
    int blockSize = 128;
    int allGroupsMask = 0xFFFFFFFF;

    // ===== 1a. Assemble metric tensor (if Riemannian) =====
    bool useMetric = (integrator.getMetricType() != MultiGroupHMCIntegrator::METRIC_IDENTITY &&
                      integrator.getMetricUpdateMode() == MultiGroupHMCIntegrator::METRIC_UPDATE_EVERY_TRAJECTORY);
    if (useMetric) {
        // Need forces computed at current positions for Hessian
        if (!forcesAreValid) {
            context.calcForcesAndEnergy(true, true, allGroupsMask);
            forcesAreValid = true;
        }
        assembleMetric(context, integrator);
    }

    // ===== 1. Backup positions =====
    {
        CUdeviceptr posqPtr = cu.getPosq().getDevicePointer();
        CUdeviceptr backupPtr = positionsBackup.getDevicePointer();
        int numBlocks = min((numGroupAtoms + blockSize - 1) / blockSize,
                            cu.getNumThreadBlocks());
        void* args[] = { &posqPtr, &backupPtr, &numGroupAtoms };
        cu.executeKernel(backupPositionsKernel, args, numBlocks * blockSize, blockSize);
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
            if (integrator.getMomentumRefreshMode() == MultiGroupHMCIntegrator::FULL) {
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
            if (integrator.getMomentumRefreshMode() == MultiGroupHMCIntegrator::FULL) {
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

    // ===== 3. Compute KE_old =====
    vector<double> keOld;
    computeGroupKE(context, keOld);

    // ===== 4. Compute PE_old =====
    context.calcForcesAndEnergy(true, true, allGroupsMask);

    vector<double> peOld(K, 0.0);
    computeGroupPE(peOld);

    // ===== 5. RESPA trajectory =====
    respaTrajectory(context, integrator);

    // ===== 6. Compute KE_new =====
    vector<double> keNew;
    computeGroupKE(context, keNew);

    // ===== 7. Compute PE_new =====
    context.calcForcesAndEnergy(true, true, allGroupsMask);

    vector<double> peNew(K, 0.0);
    computeGroupPE(peNew);

    // ===== 8. Metropolis accept/reject per group (host) =====
    double stabilityThreshold = integrator.getStabilityThreshold();

    for (int k = 0; k < K; k++) {
        double T = integrator.getGroupTemperature(k);
        double kT = BOLTZ * T;

        double deltaPE = peNew[k] - peOld[k];
        double deltaKE = keNew[k] - keOld[k];
        double deltaH = deltaPE + deltaKE;

        lastDeltaH[k] = deltaH;
        trialCounts[k]++;

        bool stable = (fabs(deltaPE) / kT < stabilityThreshold) ||
                      (fabs(deltaH) / kT < stabilityThreshold);
        if (!stable) {
            lastAccepted[k] = 0;
            acceptedHost[k] = 0;
            stabilityRejectCounts[k]++;
            continue;
        }

        bool accept = (deltaH <= 0.0) || (uniformDist(rng) < exp(-deltaH / kT));
        lastAccepted[k] = accept ? 1 : 0;
        acceptedHost[k] = lastAccepted[k];
        if (accept)
            acceptCounts[k]++;
    }

    // ===== 9. Restore rejected positions =====
    {
        acceptedBuffer.upload(acceptedHost);

        CUdeviceptr posqPtr = cu.getPosq().getDevicePointer();
        CUdeviceptr velmPtr = cu.getVelm().getDevicePointer();
        CUdeviceptr backupPtr = positionsBackup.getDevicePointer();
        CUdeviceptr accPtr = acceptedBuffer.getDevicePointer();

        int numBlocks = min((numGroupAtoms + blockSize - 1) / blockSize,
                            cu.getNumThreadBlocks());
        void* args[] = {
            &posqPtr, &velmPtr, &backupPtr, &accPtr,
            &atomsPerGroup, &numGroups
        };
        cu.executeKernel(restoreRejectedKernel, args, numBlocks * blockSize, blockSize);
    }
}

// ========== Riemannian Metric ==========

void CudaIntegrateMultiGroupHMCStepKernel::assembleMetric(
        ContextImpl& context, const MultiGroupHMCIntegrator& integrator) {

    int K = numGroups;
    int totalAtoms = K * atomsPerGroup;
    int numElements = 6 * totalAtoms;
    int blockSize = 128;

    // Lazy initialization of metric buffers
    if (!metricInitialized) {
        metricBuffer.initialize<float>(cu, numElements, "hmcMetric");
        metricInvBuffer.initialize<float>(cu, numElements, "hmcMetricInv");
        choleskyBuffer.initialize<float>(cu, numElements, "hmcCholesky");
        logDetBuffer.initialize<double>(cu, K, "hmcLogDet");
        conditionBuffer.initialize<float>(cu, K, "hmcCondition");
        combinedHessianBuffer.initialize<float>(cu, numElements, "hmcCombinedHessian");

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

    if (metricType == MultiGroupHMCIntegrator::METRIC_IDENTITY) {
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
            externalHessianBuffer.initialize<float>(cu, numElements, "hmcExternalHessian");
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
        throw OpenMMException("MultiGroupHMCIntegrator: No Hessian source found for metric computation");

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

vector<double> CudaIntegrateMultiGroupHMCStepKernel::getGroupMetricConditionNumbers() const {
    vector<double> result(numGroups, 1.0);
    for (int k = 0; k < numGroups; k++)
        result[k] = (double)conditionNumbersHost[k];
    return result;
}

static void generateRandomQuaternionRotation(std::mt19937& rng, std::uniform_real_distribution<double>& uDist,
                                             double* R) {
    // Uniform quaternion on SO(3) using Shoemake's method
    double u0 = uDist(rng);
    double u1 = uDist(rng);
    double u2 = uDist(rng);

    double s0 = sqrt(1.0 - u0);
    double s1 = sqrt(u0);
    double t1 = 2.0 * M_PI * u1;
    double t2 = 2.0 * M_PI * u2;

    double q0 = s0 * sin(t1);
    double q1 = s0 * cos(t1);
    double q2 = s1 * sin(t2);
    double q3 = s1 * cos(t2);

    // Quaternion → rotation matrix (row-major)
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

void CudaIntegrateMultiGroupHMCStepKernel::executeMC(
        ContextImpl& context, const MultiGroupHMCIntegrator& integrator) {

    int K = numGroups;
    int numTrials = integrator.getNumMCTrials();
    double mcStep = integrator.getMCStepSize();

    // Reset per-step MC accepted counts
    fill(lastMCAcceptedPerGroup.begin(), lastMCAcceptedPerGroup.end(), 0);

    // Upload MC eligibility flags
    vector<int> mcEnabled = integrator.getAllGroupMCEnabled();
    mcEnabledHost = mcEnabled;
    mcEnabledBuffer.upload(mcEnabledHost);

    // Check if any groups are eligible
    bool anyEligible = false;
    for (int k = 0; k < K; k++) {
        if (mcEnabled[k]) { anyEligible = true; break; }
    }
    if (!anyEligible) return;

    int numGroupAtoms = K * atomsPerGroup;
    int blockSize = 128;

    // Ensure energy extractors are wired (may not be if MC runs before first HMC)
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

    // Get baseline PE (compute forces + energy for all groups)
    int allGroupsMask = 0xFFFFFFFF;
    context.calcForcesAndEnergy(true, true, allGroupsMask);
    vector<double> peBaseline(K, 0.0);
    computeGroupPE(peBaseline);

    // Get per-group kT for Metropolis
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
            void* args[] = { &posqPtr, &backupPtr, &numGroupAtoms };
            cu.executeKernel(backupPositionsKernel, args, numBlocks * blockSize, blockSize);
        }

        // 2. Compute COM per group
        {
            // Zero the COM accumulation buffer
            cu.clearBuffer(groupCOMBuffer);

            CUdeviceptr posqPtr = cu.getPosq().getDevicePointer();
            CUdeviceptr velmPtr = cu.getVelm().getDevicePointer();
            CUdeviceptr comPtr = groupCOMBuffer.getDevicePointer();

            int numBlocks = min((numGroupAtoms + blockSize - 1) / blockSize,
                                cu.getNumThreadBlocks());
            void* args[] = { &posqPtr, &velmPtr, &comPtr, &atomsPerGroup, &numGroups };
            cu.executeKernel(mcComputeCOMKernel, args, numBlocks * blockSize, blockSize);

            // Download and finalize COM (divide by total mass)
            groupCOMBuffer.download(groupCOMHost);
            for (int k = 0; k < K; k++) {
                double totalMass = groupCOMHost[k * 4 + 3];
                if (totalMass > 0) {
                    mcCOMHost[k * 3 + 0] = groupCOMHost[k * 4 + 0] / totalMass;
                    mcCOMHost[k * 3 + 1] = groupCOMHost[k * 4 + 1] / totalMass;
                    mcCOMHost[k * 3 + 2] = groupCOMHost[k * 4 + 2] / totalMass;
                }
            }
            mcCOMBuffer.upload(mcCOMHost);
        }

        // 3. Generate rotation + translation on host
        for (int k = 0; k < K; k++) {
            if (!mcEnabled[k]) {
                // Identity rotation, zero translation for non-eligible groups
                fill(mcRotationHost.begin() + k * 9, mcRotationHost.begin() + k * 9 + 9, 0.0);
                mcRotationHost[k * 9 + 0] = 1.0;  // R = I
                mcRotationHost[k * 9 + 4] = 1.0;
                mcRotationHost[k * 9 + 8] = 1.0;
                fill(mcTranslationHost.begin() + k * 3, mcTranslationHost.begin() + k * 3 + 3, 0.0);
                continue;
            }

            if (trial % 2 == 0) {
                // Even trial: random rotation + translation
                generateRandomQuaternionRotation(rng, uniformDist, &mcRotationHost[k * 9]);
            } else {
                // Odd trial: identity rotation (translation only)
                fill(mcRotationHost.begin() + k * 9, mcRotationHost.begin() + k * 9 + 9, 0.0);
                mcRotationHost[k * 9 + 0] = 1.0;
                mcRotationHost[k * 9 + 4] = 1.0;
                mcRotationHost[k * 9 + 8] = 1.0;
            }

            // Gaussian translation
            mcTranslationHost[k * 3 + 0] = normalDist(rng) * mcStep;
            mcTranslationHost[k * 3 + 1] = normalDist(rng) * mcStep;
            mcTranslationHost[k * 3 + 2] = normalDist(rng) * mcStep;
        }

        // 4. Upload and apply rigid-body move
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
            void* args[] = {
                &posqPtr, &comPtr, &rotPtr, &transPtr, &enabledPtr,
                &atomsPerGroup, &numGroups
            };
            cu.executeKernel(mcApplyRigidBodyMoveKernel, args, numBlocks * blockSize, blockSize);
        }

        // 5. Recompute PE
        context.calcForcesAndEnergy(true, true, allGroupsMask);
        vector<double> peTrial(K, 0.0);
        computeGroupPE(peTrial);

        // 6. Metropolis accept/reject per eligible group
        vector<int> mcAccepted(K, 0);
        for (int k = 0; k < K; k++) {
            if (!mcEnabled[k]) continue;

            mcAttemptedTotal++;
            double dE = peTrial[k] - peBaseline[k];

            bool accept = (dE <= 0.0) || (uniformDist(rng) < exp(-dE / kT[k]));
            if (accept) {
                mcAcceptedTotal++;
                mcAccepted[k] = 1;
                lastMCAcceptedPerGroup[k]++;
                peBaseline[k] = peTrial[k];
            }
        }

        // 7. Restore rejected eligible groups
        // Build accepted flags: non-eligible groups are "accepted" (don't restore)
        for (int k = 0; k < K; k++) {
            if (!mcEnabled[k])
                acceptedHost[k] = 1;  // keep non-eligible positions
            else
                acceptedHost[k] = mcAccepted[k];
        }
        acceptedBuffer.upload(acceptedHost);

        {
            CUdeviceptr posqPtr = cu.getPosq().getDevicePointer();
            CUdeviceptr velmPtr = cu.getVelm().getDevicePointer();
            CUdeviceptr backupPtr = positionsBackup.getDevicePointer();
            CUdeviceptr accPtr = acceptedBuffer.getDevicePointer();

            int numBlocks = min((numGroupAtoms + blockSize - 1) / blockSize,
                                cu.getNumThreadBlocks());
            void* args[] = {
                &posqPtr, &velmPtr, &backupPtr, &accPtr,
                &atomsPerGroup, &numGroups
            };
            cu.executeKernel(restoreRejectedKernel, args, numBlocks * blockSize, blockSize);
        }

        // If any MC group was rejected, update baseline PE for those groups
        // (positions were restored so PE is back to baseline — no update needed)
    }
}

void CudaIntegrateMultiGroupHMCStepKernel::resetMCCounters() {
    mcAttemptedTotal = 0;
    mcAcceptedTotal = 0;
    fill(lastMCAcceptedPerGroup.begin(), lastMCAcceptedPerGroup.end(), 0);
}

void CudaIntegrateMultiGroupHMCStepKernel::respaTrajectory(
        ContextImpl& context, const MultiGroupHMCIntegrator& integrator) {

    int K = numGroups;
    int numOuterSteps = integrator.getNumOuterSteps();
    const vector<pair<int,int> >& schedule = integrator.getForceGroupSchedule();

    CUdeviceptr forcePtr = cu.getLongForceBuffer().getDevicePointer();
    int paddedNumAtoms = cu.getPaddedNumAtoms();
    int numGroupAtoms = K * atomsPerGroup;

    if (schedule.empty()) {
        // Simple Verlet with per-group dt
        int allGroupsMask = 0xFFFFFFFF;

        for (int step = 0; step < numOuterSteps; step++) {
            // Half-kick
            launchKick(forcePtr, 0.5);

            // Drift
            launchDrift();

            // Recompute forces
            context.calcForcesAndEnergy(true, false, allGroupsMask);

            // Half-kick
            launchKick(forcePtr, 0.5);
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

    CUdeviceptr slowForcePtr = slowForcesBackup.getDevicePointer();

    // Compute initial slow forces and save them
    context.calcForcesAndEnergy(true, false, slowMask);
    {
        int blockSize = 128;
        int numBlocks = min((numGroupAtoms + blockSize - 1) / blockSize,
                            cu.getNumThreadBlocks());
        void* args[] = { &forcePtr, &slowForcePtr, &numGroupAtoms, &paddedNumAtoms };
        cu.executeKernel(copyForcesKernel, args, numBlocks * blockSize, blockSize);
    }

    // Compute initial fast forces (left in force buffer)
    context.calcForcesAndEnergy(true, false, fastMask);

    // Prepare inner dt buffer: groupStepSizes / innerStepsPerOuter
    vector<double> innerDtHost(K);
    for (int k = 0; k < K; k++)
        innerDtHost[k] = groupStepSizesHost[k] / innerStepsPerOuter;

    // We need two dt buffers: outer (already in groupStepSizesBuffer) and inner
    // To avoid extra allocation, we re-upload as needed.

    for (int outer = 0; outer < numOuterSteps; outer++) {
        // Slow half-kick (outer dt, from saved slow forces)
        groupStepSizesBuffer.upload(groupStepSizesHost);
        launchKick(slowForcePtr, 0.5);

        // Inner loop
        for (int inner = 0; inner < innerStepsPerOuter; inner++) {
            // Fast half-kick (inner dt)
            groupStepSizesBuffer.upload(innerDtHost);
            launchKick(forcePtr, 0.5);

            // Drift (inner dt)
            launchDrift();

            // Recompute fast forces
            context.calcForcesAndEnergy(true, false, fastMask);

            // Fast half-kick (inner dt)
            launchKick(forcePtr, 0.5);
        }

        // Recompute slow forces and save
        context.calcForcesAndEnergy(true, false, slowMask);
        {
            int blockSize = 128;
            int numBlocks = min((numGroupAtoms + blockSize - 1) / blockSize,
                                cu.getNumThreadBlocks());
            void* args[] = { &forcePtr, &slowForcePtr, &numGroupAtoms, &paddedNumAtoms };
            cu.executeKernel(copyForcesKernel, args, numBlocks * blockSize, blockSize);
        }

        // Slow half-kick (outer dt, from saved slow forces)
        groupStepSizesBuffer.upload(groupStepSizesHost);
        launchKick(slowForcePtr, 0.5);
    }

    // Restore outer dt in buffer for any subsequent use
    groupStepSizesBuffer.upload(groupStepSizesHost);
}

double CudaIntegrateMultiGroupHMCStepKernel::computeKineticEnergy(
        ContextImpl& context, const MultiGroupHMCIntegrator& integrator) {

    if (!hasInitializedKernel)
        return 0.0;
    vector<double> groupKE;
    computeGroupKE(context, groupKE);
    double total = 0.0;
    for (int k = 0; k < numGroups; k++)
        total += groupKE[k];
    return total;
}

void CudaIntegrateMultiGroupHMCStepKernel::resetCounters() {
    fill(acceptCounts.begin(), acceptCounts.end(), 0);
    fill(trialCounts.begin(), trialCounts.end(), 0);
    fill(stabilityRejectCounts.begin(), stabilityRejectCounts.end(), 0);
}

}  // namespace GridForcePlugin
