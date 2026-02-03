/* -------------------------------------------------------------------------- *
 *                              OpenMMGridForce                               *
 * -------------------------------------------------------------------------- */

#include "CudaIsolatedGBSAKernels.h"
#include "CudaGridForceKernelSources.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/cuda/CudaBondedUtilities.h"
#include "openmm/cuda/CudaForceInfo.h"
#include "openmm/OpenMMException.h"
#include <cmath>
#include <iostream>

using namespace GridForcePlugin;
using namespace OpenMM;
using namespace std;

// Coulomb constant in kJ*nm/mol/e^2
static const float ONE_4PI_EPS0 = 138.935456f;

CudaCalcIsolatedGBSAForceKernel::CudaCalcIsolatedGBSAForceKernel(string name, const Platform& platform,
                                                                   CudaContext& cu)
    : CalcIsolatedGBSAForceKernel(name, platform), cu(cu), hasInitializedKernel(false),
      numAtoms(0), numParticleGroups(0),
      gbMethod(IsolatedGBSAForce::OBC_II),
      receptorMode(IsolatedGBSAForce::NONE),
      prefactor(0), includeSurfaceArea(false), surfaceTension(0), cutoffDistance(-1.0f),
      originX(0), originY(0), originZ(0), gridSpacing(0), probeRadius(0),
      numBins(0), interpolationMethod(0), hasHctDerivatives(false),
      numReceptorAtoms(0), receptorReferenceEnergyValue(0.0f),
      computeReceptorHCTGridKernel(nullptr), computeReceptorHCTPairwiseKernel(nullptr),
      computeLigandHCTKernel(nullptr),
      computeBornRadiiHCTKernel(nullptr), computeBornRadiiOBCKernel(nullptr),
      computeGBEnergyKernel(nullptr), computeSAEnergyKernel(nullptr),
      accumulateBornRadiiDerivativesKernel(nullptr), accumulateSADerivativesKernel(nullptr),
      computeHCTChainRuleForcesKernel(nullptr),
      computeReceptorHCTGradientForceKernel(nullptr),
      computeReceptorHCTPairwiseChainRuleKernel(nullptr),
      computeReceptorSelfHCTKernel(nullptr),
      computeReceptorBornRadiiReferenceKernel(nullptr),
      computeReceptorReferenceEnergyKernel(nullptr),
      computeLigandToReceptorHCTKernel(nullptr),
      computeReceptorBornRadiiWithLigandKernel(nullptr),
      computeReceptorGBEnergyKernel(nullptr),
      computeReceptorDeDRSimpleKernel(nullptr),
      computeCrossTermGBEnergyKernel(nullptr),
      computeReceptorDesolvationForcesKernel(nullptr),
      computeReceptorDesolvationForcesOptimizedKernel(nullptr),
      computeCrossTermChainRuleForcesKernel(nullptr),
      computeHessianKernel(nullptr) {
}

CudaCalcIsolatedGBSAForceKernel::~CudaCalcIsolatedGBSAForceKernel() {
}

void CudaCalcIsolatedGBSAForceKernel::initialize(const System& system, const IsolatedGBSAForce& force) {
    cu.setAsCurrent();
    numAtoms = force.getNumAtoms();
    numParticleGroups = force.getNumParticleGroups();

    if (numAtoms == 0) {
        throw OpenMMException("IsolatedGBSAForce: no atoms defined");
    }

    // Store configuration
    gbMethod = force.getGBMethod();
    receptorMode = force.getReceptorMode();
    cutoffDistance = static_cast<float>(force.getCutoffDistance());

    // Compute GB prefactor: -138.935456 * (1/ε_solute - 1/ε_solvent)
    double soluteDielectric = force.getSoluteDielectric();
    double solventDielectric = force.getSolventDielectric();
    prefactor = static_cast<float>(-ONE_4PI_EPS0 * (1.0/soluteDielectric - 1.0/solventDielectric));

    includeSurfaceArea = force.getIncludeSurfaceArea();
    surfaceTension = static_cast<float>(force.getSurfaceTension());
    interpolationMethod = force.getInterpolationMethod();

    // Upload ligand atom parameters
    vector<float> chargesVec(numAtoms), radiiVec(numAtoms), scalesVec(numAtoms);
    for (int i = 0; i < numAtoms; i++) {
        double q, r, s;
        force.getAtomParameters(i, q, r, s);
        chargesVec[i] = static_cast<float>(q);
        radiiVec[i] = static_cast<float>(r);
        scalesVec[i] = static_cast<float>(s);
    }
    charges.initialize<float>(cu, numAtoms, "isolatedGbsaCharges");
    charges.upload(chargesVec);
    radii.initialize<float>(cu, numAtoms, "isolatedGbsaRadii");
    radii.upload(radiiVec);
    scaleFactors.initialize<float>(cu, numAtoms, "isolatedGbsaScaleFactors");
    scaleFactors.upload(scalesVec);

    // Initialize receptor mode specific data
    if (receptorMode == IsolatedGBSAForce::GRID) {
        auto grid = force.getDesolvationGrid();
        if (!grid) {
            throw OpenMMException("IsolatedGBSAForce: GRID mode requires a desolvation grid");
        }

        // Store grid parameters
        double ox, oy, oz;
        grid->getOrigin(ox, oy, oz);
        originX = static_cast<float>(ox);
        originY = static_cast<float>(oy);
        originZ = static_cast<float>(oz);
        gridSpacing = static_cast<float>(grid->getSpacing());
        probeRadius = static_cast<float>(grid->getProbeRadius());
        numBins = grid->getNumBins();

        // Upload grid dimensions
        int nx, ny, nz;
        grid->getCounts(nx, ny, nz);
        vector<int> counts = {nx, ny, nz};
        gridCounts.initialize<int>(cu, 3, "isolatedGbsaGridCounts");
        gridCounts.upload(counts);

        // Upload grid data
        int numPoints = nx * ny * nz;
        const auto& hctData = grid->getHctProbe();
        hasHctDerivatives = grid->hasDerivatives();

        if (hasHctDerivatives) {
            int numDerivs = grid->getNumDerivsPerPoint();
            vector<float> hctValues(hctData.begin(), hctData.begin() + numPoints);
            gridHctProbe.initialize<float>(cu, numPoints, "isolatedGbsaGridHctProbe");
            gridHctProbe.upload(hctValues);
            gridHctDerivatives.initialize<float>(cu, hctData.size(), "isolatedGbsaGridHctDerivatives");
            gridHctDerivatives.upload(hctData);
        } else {
            gridHctProbe.initialize<float>(cu, numPoints, "isolatedGbsaGridHctProbe");
            gridHctProbe.upload(hctData);
        }

        int corrSize = numBins * numPoints;
        gridCorrectionN.initialize<float>(cu, corrSize, "isolatedGbsaGridCorrectionN");
        gridCorrectionN.upload(grid->getCorrectionN());
        gridCorrectionA.initialize<float>(cu, corrSize, "isolatedGbsaGridCorrectionA");
        gridCorrectionA.upload(grid->getCorrectionA());
        gridCorrectionB.initialize<float>(cu, corrSize, "isolatedGbsaGridCorrectionB");
        gridCorrectionB.upload(grid->getCorrectionB());

        const auto& thresholds = grid->getRThresholds();
        vector<float> thresholdsFloat(thresholds.begin(), thresholds.end());
        rThresholds.initialize<float>(cu, numBins, "isolatedGbsaRThresholds");
        rThresholds.upload(thresholdsFloat);

    } else if (receptorMode == IsolatedGBSAForce::PAIRWISE) {
        numReceptorAtoms = force.getNumReceptorAtoms();
        if (numReceptorAtoms == 0) {
            throw OpenMMException("IsolatedGBSAForce: PAIRWISE mode requires receptor atoms");
        }

        const auto& recPos = force.getReceptorPositions();
        if (recPos.size() != static_cast<size_t>(numReceptorAtoms * 3)) {
            throw OpenMMException("IsolatedGBSAForce: receptor positions size mismatch");
        }

        // Upload receptor positions as float3
        vector<float3> positionsF(numReceptorAtoms);
        for (int i = 0; i < numReceptorAtoms; i++) {
            positionsF[i] = make_float3(
                static_cast<float>(recPos[i*3]),
                static_cast<float>(recPos[i*3 + 1]),
                static_cast<float>(recPos[i*3 + 2])
            );
        }
        receptorPositions.initialize<float3>(cu, numReceptorAtoms, "isolatedGbsaReceptorPositions");
        receptorPositions.upload(positionsF);

        // Upload receptor parameters
        vector<float> recRadii(numReceptorAtoms), recScales(numReceptorAtoms), recCharges(numReceptorAtoms);
        for (int i = 0; i < numReceptorAtoms; i++) {
            double q, r, s;
            force.getReceptorAtomParameters(i, q, r, s);
            recCharges[i] = static_cast<float>(q);
            recRadii[i] = static_cast<float>(r);
            recScales[i] = static_cast<float>(s);
        }
        receptorRadii.initialize<float>(cu, numReceptorAtoms, "isolatedGbsaReceptorRadii");
        receptorRadii.upload(recRadii);
        receptorScaleFactors.initialize<float>(cu, numReceptorAtoms, "isolatedGbsaReceptorScaleFactors");
        receptorScaleFactors.upload(recScales);
        receptorCharges.initialize<float>(cu, numReceptorAtoms, "isolatedGbsaReceptorCharges");
        receptorCharges.upload(recCharges);

        // Allocate buffers for receptor desolvation computation
        receptorSelfHCT.initialize<float>(cu, numReceptorAtoms, "isolatedGbsaReceptorSelfHCT");
        receptorBornRadiiRef.initialize<float>(cu, numReceptorAtoms, "isolatedGbsaReceptorBornRadiiRef");
        receptorReferenceEnergy.initialize<float>(cu, 1, "isolatedGbsaReceptorReferenceEnergy");
        receptorBornRadii.initialize<float>(cu, numReceptorAtoms, "isolatedGbsaReceptorBornRadii");
        receptorEnergy.initialize<float>(cu, 1, "isolatedGbsaReceptorEnergy");
    }

    // Process particle groups
    int totalParticles = 0;
    if (numParticleGroups > 0) {
        vector<int> allIndices;
        vector<int> groupStarts(numParticleGroups + 1);
        groupStarts[0] = 0;

        for (int g = 0; g < numParticleGroups; g++) {
            string name;
            vector<int> indices;
            force.getParticleGroup(g, name, indices);
            for (int idx : indices) {
                allIndices.push_back(idx);
            }
            groupStarts[g + 1] = static_cast<int>(allIndices.size());
        }

        totalParticles = static_cast<int>(allIndices.size());
        particleIndices.initialize<int>(cu, totalParticles, "isolatedGbsaParticleIndices");
        particleIndices.upload(allIndices);
        groupStartIndex.initialize<int>(cu, numParticleGroups + 1, "isolatedGbsaGroupStart");
        groupStartIndex.upload(groupStarts);
    } else {
        // Single group with all atoms from particles list
        const auto& particles = force.getParticles();
        if (!particles.empty()) {
            totalParticles = static_cast<int>(particles.size());
            particleIndices.initialize<int>(cu, totalParticles, "isolatedGbsaParticleIndices");
            particleIndices.upload(particles);
        } else {
            // All atoms
            totalParticles = numAtoms;
            vector<int> allIndices(numAtoms);
            for (int i = 0; i < numAtoms; i++) allIndices[i] = i;
            particleIndices.initialize<int>(cu, totalParticles, "isolatedGbsaParticleIndices");
            particleIndices.upload(allIndices);
        }

        numParticleGroups = 1;
        vector<int> groupStarts = {0, totalParticles};
        groupStartIndex.initialize<int>(cu, 2, "isolatedGbsaGroupStart");
        groupStartIndex.upload(groupStarts);
    }

    // Allocate per-group energy buffers
    groupEnergies.initialize<float>(cu, numParticleGroups, "isolatedGbsaGroupEnergies");
    groupLigandSelfEnergies.initialize<float>(cu, numParticleGroups, "isolatedGbsaGroupLigandSelfEnergies");
    groupReceptorContributions.initialize<float>(cu, numParticleGroups, "isolatedGbsaGroupReceptorContributions");
    groupReceptorDesolvations.initialize<float>(cu, numParticleGroups, "isolatedGbsaGroupReceptorDesolvations");
    groupCrossTermEnergies.initialize<float>(cu, numParticleGroups, "isolatedGbsaGroupCrossTermEnergies");

    groupEnergiesHost.resize(numParticleGroups);
    groupLigandSelfEnergiesHost.resize(numParticleGroups);
    groupReceptorContributionsHost.resize(numParticleGroups);
    groupReceptorDesolvationsHost.resize(numParticleGroups);
    groupCrossTermEnergiesHost.resize(numParticleGroups);
    groupBornRadiiHost.resize(numParticleGroups);
    groupAtomEnergiesHost.resize(numParticleGroups);

    // Allocate ligand-to-receptor HCT buffer for PAIRWISE mode
    if (receptorMode == IsolatedGBSAForce::PAIRWISE && numReceptorAtoms > 0) {
        ligandToReceptorHCT.initialize<float>(cu, numReceptorAtoms * numParticleGroups, "isolatedGbsaLigandToReceptorHCT");
    }

    // Allocate intermediate result buffers
    if (totalParticles > 0) {
        hctReceptor.initialize<float>(cu, totalParticles, "isolatedGbsaHctReceptor");
        hctLigand.initialize<float>(cu, totalParticles, "isolatedGbsaHctLigand");
        bornRadii.initialize<float>(cu, totalParticles, "isolatedGbsaBornRadii");
        dE_dR.initialize<float>(cu, totalParticles, "isolatedGbsaDEdR");
        atomEnergies.initialize<float>(cu, totalParticles, "isolatedGbsaAtomEnergies");
    }

    // Compile CUDA kernels (all kernels are in gridForceKernel)
    CUmodule module = cu.createModule(CudaGridForceKernelSources::gridForceKernel);
    computeLigandHCTKernel = cu.getKernel(module, "computeIsolatedLigandHCT");
    computeBornRadiiHCTKernel = cu.getKernel(module, "computeBornRadiiHCT");
    computeBornRadiiOBCKernel = cu.getKernel(module, "computeBornRadiiOBC");
    computeGBEnergyKernel = cu.getKernel(module, "computeIsolatedGBEnergy");
    computeSAEnergyKernel = cu.getKernel(module, "computeIsolatedSAEnergy");
    accumulateBornRadiiDerivativesKernel = cu.getKernel(module, "accumulateIsolatedBornRadiiDerivatives");
    accumulateSADerivativesKernel = cu.getKernel(module, "accumulateIsolatedSADerivatives");
    computeHCTChainRuleForcesKernel = cu.getKernel(module, "computeIsolatedHCTChainRuleForces");

    if (receptorMode == IsolatedGBSAForce::GRID) {
        computeReceptorHCTGridKernel = cu.getKernel(module, "computeIsolatedReceptorHCTGrid");
        computeReceptorHCTGradientForceKernel = cu.getKernel(module, "computeIsolatedReceptorHCTGradientForce");
    } else if (receptorMode == IsolatedGBSAForce::PAIRWISE) {
        computeReceptorHCTPairwiseKernel = cu.getKernel(module, "computeIsolatedReceptorHCTPairwise");
        computeReceptorHCTPairwiseChainRuleKernel = cu.getKernel(module, "computeIsolatedReceptorHCTPairwiseChainRule");

        // PAIRWISE mode: receptor desolvation kernels (both old and tiled versions)
        computeReceptorSelfHCTKernel = cu.getKernel(module, "computeReceptorSelfHCT");
        computeReceptorSelfHCTTiledKernel = cu.getKernel(module, "computeReceptorSelfHCTTiled");
        convertHCTToFloatKernel = cu.getKernel(module, "convertHCTToFloat");
        computeReceptorBornRadiiReferenceKernel = cu.getKernel(module, "computeReceptorBornRadiiReference");
        computeReceptorReferenceEnergyKernel = cu.getKernel(module, "computeReceptorReferenceEnergy");
        computeReceptorGBEnergyTiledKernel = cu.getKernel(module, "computeReceptorGBEnergyTiled");
        computeLigandToReceptorHCTKernel = cu.getKernel(module, "computeLigandToReceptorHCT");
        computeReceptorBornRadiiWithLigandKernel = cu.getKernel(module, "computeReceptorBornRadiiWithLigand");
        computeReceptorGBEnergyKernel = cu.getKernel(module, "computeReceptorGBEnergy");
        computeReceptorDeDRSimpleKernel = cu.getKernel(module, "computeReceptorDeDRSimple");
        computeCrossTermGBEnergyKernel = cu.getKernel(module, "computeCrossTermGBEnergy");
        computeReceptorDesolvationForcesKernel = cu.getKernel(module, "computeReceptorDesolvationForces");
        computeReceptorDesolvationForcesOptimizedKernel = cu.getKernel(module, "computeReceptorDesolvationForcesOptimized");
        computeCrossTermChainRuleForcesKernel = cu.getKernel(module, "computeCrossTermChainRuleForces");

        // Allocate fixed-point buffer for tiled HCT computation
        receptorSelfHCTFixed.initialize<unsigned long long>(cu, numReceptorAtoms, "receptorSelfHCTFixed");
        // Allocate buffer for pre-computed receptor dE/dR_born
        receptorDeDR.initialize<float>(cu, numReceptorAtoms, "receptorDeDR");

        // Compute receptor self-HCT using TILED kernel for O(N²) efficiency
        const int TILE_SIZE = 32;
        int numBlocks = (numReceptorAtoms + TILE_SIZE - 1) / TILE_SIZE;
        int numTiles = numBlocks * (numBlocks + 1) / 2;

        // Use enough warps to cover all tiles efficiently
        int recBlockSize = 256;  // 8 warps per block
        int totalWarps = (numTiles + 7) / 8 * 8;  // Round up to multiple of warps per block
        int recNumBlocks = (totalWarps * TILE_SIZE + recBlockSize - 1) / recBlockSize;
        // Ensure at least as many thread blocks as needed
        recNumBlocks = max(recNumBlocks, (numTiles + 7) / 8);

        CUdeviceptr receptorPosPtr = receptorPositions.getDevicePointer();
        CUdeviceptr receptorRadiiPtr = receptorRadii.getDevicePointer();
        CUdeviceptr receptorScalesPtr = receptorScaleFactors.getDevicePointer();
        CUdeviceptr receptorChargesPtr = receptorCharges.getDevicePointer();
        CUdeviceptr receptorSelfHCTPtr = receptorSelfHCT.getDevicePointer();
        CUdeviceptr receptorSelfHCTFixedPtr = receptorSelfHCTFixed.getDevicePointer();
        CUdeviceptr receptorBornRadiiRefPtr = receptorBornRadiiRef.getDevicePointer();
        CUdeviceptr receptorRefEnergyPtr = receptorReferenceEnergy.getDevicePointer();

        // Step 1: Clear fixed-point buffer and compute receptor-receptor self HCT (tiled)
        vector<unsigned long long> zerosFixed(numReceptorAtoms, 0);
        receptorSelfHCTFixed.upload(zerosFixed);

        void* selfHctTiledArgs[] = {
            &receptorPosPtr, &receptorRadiiPtr, &receptorScalesPtr,
            &numReceptorAtoms, &cutoffDistance, &receptorSelfHCTFixedPtr, &numTiles
        };
        cu.executeKernel(computeReceptorSelfHCTTiledKernel, selfHctTiledArgs, recNumBlocks * recBlockSize, recBlockSize);

        // Step 1b: Convert fixed-point HCT to float
        int convertBlocks = (numReceptorAtoms + recBlockSize - 1) / recBlockSize;
        void* convertArgs[] = {
            &receptorSelfHCTFixedPtr, &receptorSelfHCTPtr, &numReceptorAtoms
        };
        cu.executeKernel(convertHCTToFloatKernel, convertArgs, convertBlocks * recBlockSize, recBlockSize);

        // Step 2: Compute receptor Born radii without ligand
        void* bornRefArgs[] = {
            &receptorRadiiPtr, &receptorSelfHCTPtr, &numReceptorAtoms, &receptorBornRadiiRefPtr
        };
        cu.executeKernel(computeReceptorBornRadiiReferenceKernel, bornRefArgs, convertBlocks * recBlockSize, recBlockSize);

        // Step 3: Compute receptor reference energy using TILED kernel
        vector<float> zeroEnergy(1, 0.0f);
        receptorReferenceEnergy.upload(zeroEnergy);

        void* refEnergyTiledArgs[] = {
            &receptorPosPtr, &receptorChargesPtr, &receptorBornRadiiRefPtr,
            &numReceptorAtoms, &prefactor, &receptorRefEnergyPtr, &numTiles
        };
        cu.executeKernel(computeReceptorGBEnergyTiledKernel, refEnergyTiledArgs, recNumBlocks * recBlockSize, recBlockSize);

        // Download and cache reference energy
        vector<float> refEnergy(1);
        receptorReferenceEnergy.download(refEnergy);
        receptorReferenceEnergyValue = refEnergy[0];
    }

    hasInitializedKernel = true;
}

double CudaCalcIsolatedGBSAForceKernel::execute(ContextImpl& context,
                                                 bool includeForces,
                                                 bool includeEnergy) {
    if (!hasInitializedKernel) {
        throw OpenMMException("IsolatedGBSAForce kernel not initialized");
    }

    int totalParticles = particleIndices.getSize();
    if (totalParticles == 0) return 0.0;

    int paddedNumAtoms = cu.getPaddedNumAtoms();

    // Clear energy buffers
    vector<float> zeros(numParticleGroups, 0.0f);
    groupEnergies.upload(zeros);
    groupLigandSelfEnergies.upload(zeros);
    groupReceptorContributions.upload(zeros);
    groupReceptorDesolvations.upload(zeros);
    groupCrossTermEnergies.upload(zeros);

    // Clear intermediate buffers
    vector<float> zerosParticles(totalParticles, 0.0f);
    hctReceptor.upload(zerosParticles);
    hctLigand.upload(zerosParticles);

    // Get device pointers
    CUdeviceptr posqPtr = cu.getPosq().getDevicePointer();
    CUdeviceptr forcePtr = cu.getLongForceBuffer().getDevicePointer();
    CUdeviceptr particleIndicesPtr = particleIndices.getDevicePointer();
    CUdeviceptr radiiPtr = radii.getDevicePointer();
    CUdeviceptr scaleFactorsPtr = scaleFactors.getDevicePointer();
    CUdeviceptr chargesPtr = charges.getDevicePointer();
    CUdeviceptr groupStartPtr = groupStartIndex.getDevicePointer();
    CUdeviceptr hctReceptorPtr = hctReceptor.getDevicePointer();
    CUdeviceptr hctLigandPtr = hctLigand.getDevicePointer();
    CUdeviceptr bornRadiiPtr = bornRadii.getDevicePointer();
    CUdeviceptr groupEnergiesPtr = groupEnergies.getDevicePointer();
    CUdeviceptr groupLigandEnergiesPtr = groupLigandSelfEnergies.getDevicePointer();

    int blockSize = 256;
    int numBlocks = (totalParticles + blockSize - 1) / blockSize;

    // Step 1: Compute receptor HCT (if receptor mode is enabled)
    if (receptorMode == IsolatedGBSAForce::GRID) {
        CUdeviceptr gridCountsPtr = gridCounts.getDevicePointer();
        CUdeviceptr gridHctProbePtr = gridHctProbe.getDevicePointer();
        CUdeviceptr gridHctDerivativesPtr = hasHctDerivatives ? gridHctDerivatives.getDevicePointer() : 0;
        CUdeviceptr gridCorrectionNPtr = gridCorrectionN.getDevicePointer();
        CUdeviceptr gridCorrectionAPtr = gridCorrectionA.getDevicePointer();
        CUdeviceptr gridCorrectionBPtr = gridCorrectionB.getDevicePointer();
        CUdeviceptr rThresholdsPtr = rThresholds.getDevicePointer();

        void* receptorArgs[] = {
            &posqPtr, &particleIndicesPtr, &radiiPtr,
            &gridCountsPtr, &gridHctProbePtr, &gridHctDerivativesPtr,
            &gridCorrectionNPtr, &gridCorrectionAPtr, &gridCorrectionBPtr,
            &rThresholdsPtr, &groupStartPtr, &numParticleGroups,
            &originX, &originY, &originZ, &gridSpacing, &probeRadius,
            &numBins, &totalParticles, &numAtoms, &interpolationMethod, &hctReceptorPtr
        };
        cu.executeKernel(computeReceptorHCTGridKernel, receptorArgs, numBlocks * blockSize, blockSize);

    } else if (receptorMode == IsolatedGBSAForce::PAIRWISE) {
        CUdeviceptr receptorPosPtr = receptorPositions.getDevicePointer();
        CUdeviceptr receptorRadiiPtr = receptorRadii.getDevicePointer();
        CUdeviceptr receptorScalesPtr = receptorScaleFactors.getDevicePointer();

        void* receptorArgs[] = {
            &posqPtr, &particleIndicesPtr, &radiiPtr,
            &receptorPosPtr, &receptorRadiiPtr, &receptorScalesPtr,
            &numReceptorAtoms, &groupStartPtr, &numParticleGroups,
            &totalParticles, &numAtoms, &cutoffDistance, &hctReceptorPtr
        };
        cu.executeKernel(computeReceptorHCTPairwiseKernel, receptorArgs, numBlocks * blockSize, blockSize);
    }

    // Step 2: Compute ligand-ligand HCT
    void* ligandArgs[] = {
        &posqPtr, &particleIndicesPtr, &radiiPtr, &scaleFactorsPtr,
        &groupStartPtr, &numParticleGroups, &numAtoms, &cutoffDistance, &hctLigandPtr
    };
    cu.executeKernel(computeLigandHCTKernel, ligandArgs, numBlocks * blockSize, blockSize);

    // Step 3: Compute Born radii
    int gbMethodInt = static_cast<int>(gbMethod);
    if (gbMethod == IsolatedGBSAForce::HCT) {
        void* bornArgs[] = {
            &radiiPtr, &hctReceptorPtr, &hctLigandPtr,
            &totalParticles, &numAtoms, &bornRadiiPtr
        };
        cu.executeKernel(computeBornRadiiHCTKernel, bornArgs, numBlocks * blockSize, blockSize);
    } else {
        void* bornArgs[] = {
            &radiiPtr, &hctReceptorPtr, &hctLigandPtr,
            &totalParticles, &numAtoms, &bornRadiiPtr
        };
        cu.executeKernel(computeBornRadiiOBCKernel, bornArgs, numBlocks * blockSize, blockSize);
    }

    // Step 4: Compute GB energy and forces
    void* energyArgs[] = {
        &posqPtr, &particleIndicesPtr, &chargesPtr, &bornRadiiPtr,
        &groupStartPtr, &numParticleGroups, &numAtoms, &prefactor,
        &forcePtr, &groupEnergiesPtr, &groupLigandEnergiesPtr, &paddedNumAtoms
    };
    cu.executeKernel(computeGBEnergyKernel, energyArgs, numBlocks * blockSize, blockSize);

    // Step 4b: PAIRWISE mode - receptor desolvation and cross-term energy
    if (receptorMode == IsolatedGBSAForce::PAIRWISE) {
        // Download ligand energies first so we can add to them
        groupEnergies.download(groupEnergiesHost);

        CUdeviceptr receptorPosPtr = receptorPositions.getDevicePointer();
        CUdeviceptr receptorRadiiPtr = receptorRadii.getDevicePointer();
        CUdeviceptr receptorScalesPtr = receptorScaleFactors.getDevicePointer();
        CUdeviceptr receptorChargesPtr = receptorCharges.getDevicePointer();
        CUdeviceptr receptorSelfHCTPtr = receptorSelfHCT.getDevicePointer();
        CUdeviceptr ligandToReceptorHCTPtr = ligandToReceptorHCT.getDevicePointer();
        CUdeviceptr receptorBornRadiiPtr = receptorBornRadii.getDevicePointer();
        CUdeviceptr receptorEnergyPtr = receptorEnergy.getDevicePointer();
        CUdeviceptr groupDesolvPtr = groupReceptorDesolvations.getDevicePointer();
        CUdeviceptr groupCrossTermPtr = groupCrossTermEnergies.getDevicePointer();

        // Simple O(N) kernel launch parameters
        int recBlockSize = 256;
        int recNumBlocksSimple = (numReceptorAtoms + recBlockSize - 1) / recBlockSize;

        // Tiled O(N²) kernel launch parameters
        const int TILE_SIZE = 32;
        int numBlksTiled = (numReceptorAtoms + TILE_SIZE - 1) / TILE_SIZE;
        int numTiles = numBlksTiled * (numBlksTiled + 1) / 2;
        int totalWarps = (numTiles + 7) / 8 * 8;  // Round up to multiple of warps per block
        int recNumBlocksTiled = max((totalWarps * TILE_SIZE + recBlockSize - 1) / recBlockSize, (numTiles + 7) / 8);

        // 4b.1: Compute ligand→receptor HCT (how each ligand group screens each receptor atom)
        int ligRecWorkItems = numParticleGroups * numReceptorAtoms;
        int ligRecBlocks = (ligRecWorkItems + blockSize - 1) / blockSize;
        void* ligToRecHctArgs[] = {
            &posqPtr, &particleIndicesPtr, &radiiPtr, &scaleFactorsPtr,
            &receptorPosPtr, &receptorRadiiPtr, &groupStartPtr,
            &numParticleGroups, &numReceptorAtoms, &numAtoms, &cutoffDistance, &ligandToReceptorHCTPtr
        };
        cu.executeKernel(computeLigandToReceptorHCTKernel, ligToRecHctArgs, ligRecBlocks * blockSize, blockSize);

        // For each group, compute receptor desolvation and cross-term
        for (int g = 0; g < numParticleGroups; g++) {
            // 4b.2: Compute receptor Born radii with this ligand group's screening
            void* recBornArgs[] = {
                &receptorRadiiPtr, &receptorSelfHCTPtr, &ligandToReceptorHCTPtr,
                &numReceptorAtoms, &g, &receptorBornRadiiPtr
            };
            cu.executeKernel(computeReceptorBornRadiiWithLigandKernel, recBornArgs, recNumBlocksSimple * recBlockSize, recBlockSize);

            // 4b.3: Compute receptor GB energy with ligand present (tiled O(N²) kernel)
            vector<float> zeroEnergy(1, 0.0f);
            receptorEnergy.upload(zeroEnergy);

            void* recEnergyArgs[] = {
                &receptorPosPtr, &receptorChargesPtr, &receptorBornRadiiPtr,
                &numReceptorAtoms, &prefactor, &receptorEnergyPtr, &numTiles
            };
            cu.executeKernel(computeReceptorGBEnergyTiledKernel, recEnergyArgs, recNumBlocksTiled * recBlockSize, recBlockSize);

            // Download and compute desolvation = energy_with_ligand - reference
            vector<float> recEnergy(1);
            receptorEnergy.download(recEnergy);
            float desolvation = recEnergy[0] - receptorReferenceEnergyValue;

            // Store desolvation for this group
            groupReceptorDesolvationsHost[g] = desolvation;

            // Add desolvation to total group energy
            groupEnergiesHost[g] += desolvation;
        }

        // Upload desolvation values to device
        groupReceptorDesolvations.upload(groupReceptorDesolvationsHost);

        // 4b.4: Compute cross-term energy (receptor-ligand GB pairs)
        // Need receptor Born radii for each group - recompute or use last one
        // For now, compute cross-term for all groups at once
        void* crossTermArgs[] = {
            &posqPtr, &particleIndicesPtr, &chargesPtr, &bornRadiiPtr,
            &receptorPosPtr, &receptorChargesPtr, &receptorBornRadiiPtr,
            &groupStartPtr, &numParticleGroups, &numReceptorAtoms, &numAtoms, &prefactor,
            &groupCrossTermPtr, &forcePtr, &paddedNumAtoms
        };
        cu.executeKernel(computeCrossTermGBEnergyKernel, crossTermArgs, numBlocks * blockSize, blockSize);

        // Download cross-term energies and add to total
        groupCrossTermEnergies.download(groupCrossTermEnergiesHost);
        for (int g = 0; g < numParticleGroups; g++) {
            groupEnergiesHost[g] += groupCrossTermEnergiesHost[g];
        }

        // Re-upload updated group energies
        groupEnergies.upload(groupEnergiesHost);
    }

    // Step 5: Optional surface area term
    if (includeSurfaceArea) {
        float probe = (receptorMode == IsolatedGBSAForce::GRID) ? probeRadius : 0.14f;
        void* saArgs[] = {
            &radiiPtr, &bornRadiiPtr, &groupStartPtr,
            &numParticleGroups, &numAtoms,
            &surfaceTension, &probe, &groupEnergiesPtr
        };
        cu.executeKernel(computeSAEnergyKernel, saArgs, numBlocks * blockSize, blockSize);
    }

    // Step 6: Chain rule forces through Born radii
    if (includeForces) {
        CUdeviceptr dE_dRPtr = dE_dR.getDevicePointer();

        // Accumulate dE/dR_born
        void* bornDerivArgs[] = {
            &posqPtr, &particleIndicesPtr, &chargesPtr, &bornRadiiPtr,
            &groupStartPtr, &numParticleGroups, &numAtoms, &prefactor, &dE_dRPtr
        };
        cu.executeKernel(accumulateBornRadiiDerivativesKernel, bornDerivArgs, numBlocks * blockSize, blockSize);

        if (includeSurfaceArea) {
            float probe = (receptorMode == IsolatedGBSAForce::GRID) ? probeRadius : 0.14f;
            void* saDerivArgs[] = {
                &radiiPtr, &bornRadiiPtr, &groupStartPtr,
                &numParticleGroups, &numAtoms, &surfaceTension, &probe, &dE_dRPtr
            };
            cu.executeKernel(accumulateSADerivativesKernel, saDerivArgs, numBlocks * blockSize, blockSize);
        }

        // Ligand-ligand HCT chain rule forces
        void* hctChainArgs[] = {
            &posqPtr, &particleIndicesPtr, &radiiPtr, &scaleFactorsPtr,
            &bornRadiiPtr, &hctReceptorPtr, &hctLigandPtr, &dE_dRPtr,
            &groupStartPtr, &numParticleGroups, &numAtoms, &cutoffDistance,
            &forcePtr, &paddedNumAtoms
        };
        cu.executeKernel(computeHCTChainRuleForcesKernel, hctChainArgs, numBlocks * blockSize, blockSize);

        // Receptor contribution forces
        if (receptorMode == IsolatedGBSAForce::GRID) {
            CUdeviceptr gridCountsPtr = gridCounts.getDevicePointer();
            CUdeviceptr gridHctProbePtr = gridHctProbe.getDevicePointer();
            CUdeviceptr gridHctDerivativesPtr = hasHctDerivatives ? gridHctDerivatives.getDevicePointer() : 0;
            CUdeviceptr gridCorrectionNPtr = gridCorrectionN.getDevicePointer();
            CUdeviceptr gridCorrectionAPtr = gridCorrectionA.getDevicePointer();
            CUdeviceptr gridCorrectionBPtr = gridCorrectionB.getDevicePointer();
            CUdeviceptr rThresholdsPtr = rThresholds.getDevicePointer();

            void* receptorGradArgs[] = {
                &posqPtr, &particleIndicesPtr, &radiiPtr,
                &bornRadiiPtr, &hctReceptorPtr, &hctLigandPtr, &dE_dRPtr,
                &gridCountsPtr, &gridHctProbePtr, &gridHctDerivativesPtr,
                &gridCorrectionNPtr, &gridCorrectionAPtr, &gridCorrectionBPtr,
                &rThresholdsPtr, &groupStartPtr, &numParticleGroups,
                &originX, &originY, &originZ, &gridSpacing, &probeRadius,
                &numBins, &totalParticles, &numAtoms, &interpolationMethod,
                &forcePtr, &paddedNumAtoms
            };
            cu.executeKernel(computeReceptorHCTGradientForceKernel, receptorGradArgs, numBlocks * blockSize, blockSize);

        } else if (receptorMode == IsolatedGBSAForce::PAIRWISE) {
            CUdeviceptr receptorPosPtr = receptorPositions.getDevicePointer();
            CUdeviceptr receptorRadiiPtr = receptorRadii.getDevicePointer();
            CUdeviceptr receptorScalesPtr = receptorScaleFactors.getDevicePointer();
            CUdeviceptr receptorChargesPtr = receptorCharges.getDevicePointer();
            CUdeviceptr receptorSelfHCTPtr = receptorSelfHCT.getDevicePointer();
            CUdeviceptr ligandToReceptorHCTPtr = ligandToReceptorHCT.getDevicePointer();
            CUdeviceptr receptorBornRadiiPtr = receptorBornRadii.getDevicePointer();

            // Launch parameters for simple O(N) kernels
            int recBlockSize = 256;
            int recNumBlocksSimple = (numReceptorAtoms + recBlockSize - 1) / recBlockSize;

            // Chain rule for receptor→ligand HCT (how receptor screens ligand Born radii)
            void* receptorChainArgs[] = {
                &posqPtr, &particleIndicesPtr, &radiiPtr,
                &bornRadiiPtr, &hctReceptorPtr, &hctLigandPtr, &dE_dRPtr,
                &receptorPosPtr, &receptorRadiiPtr, &receptorScalesPtr,
                &numReceptorAtoms, &groupStartPtr, &numParticleGroups,
                &totalParticles, &numAtoms, &cutoffDistance, &forcePtr, &paddedNumAtoms
            };
            cu.executeKernel(computeReceptorHCTPairwiseChainRuleKernel, receptorChainArgs, numBlocks * blockSize, blockSize);

            // Pre-compute dE/dR_born for all receptor atoms (O(N²) but done once, not per ligand atom)
            CUdeviceptr receptorDeDRPtr = receptorDeDR.getDevicePointer();
            void* deDRArgs[] = {
                &receptorPosPtr, &receptorChargesPtr, &receptorBornRadiiPtr,
                &numReceptorAtoms, &prefactor, &receptorDeDRPtr
            };
            cu.executeKernel(computeReceptorDeDRSimpleKernel, deDRArgs, recNumBlocksSimple * recBlockSize, recBlockSize);

            // Receptor desolvation forces using pre-computed dE/dR (now O(N_lig × N_rec) instead of O(N_lig × N_rec²))
            void* recDesolvForceArgs[] = {
                &posqPtr, &particleIndicesPtr, &radiiPtr, &scaleFactorsPtr,
                &receptorPosPtr, &receptorRadiiPtr,
                &receptorSelfHCTPtr, &ligandToReceptorHCTPtr, &receptorBornRadiiPtr, &receptorDeDRPtr,
                &groupStartPtr, &numParticleGroups, &numReceptorAtoms, &numAtoms,
                &cutoffDistance, &forcePtr, &paddedNumAtoms
            };
            cu.executeKernel(computeReceptorDesolvationForcesOptimizedKernel, recDesolvForceArgs, numBlocks * blockSize, blockSize);

            // Cross-term chain rule forces (dE_cross/dR_born through HCT)
            void* crossChainArgs[] = {
                &posqPtr, &particleIndicesPtr, &radiiPtr, &scaleFactorsPtr, &chargesPtr,
                &bornRadiiPtr, &hctReceptorPtr, &hctLigandPtr,
                &receptorPosPtr, &receptorRadiiPtr, &receptorScalesPtr, &receptorChargesPtr,
                &receptorSelfHCTPtr, &ligandToReceptorHCTPtr, &receptorBornRadiiPtr,
                &groupStartPtr, &numParticleGroups, &numReceptorAtoms, &numAtoms,
                &prefactor, &cutoffDistance, &forcePtr, &paddedNumAtoms
            };
            cu.executeKernel(computeCrossTermChainRuleForcesKernel, crossChainArgs, numBlocks * blockSize, blockSize);
        }
    }

    // Download group energies
    groupEnergies.download(groupEnergiesHost);
    groupLigandSelfEnergies.download(groupLigandSelfEnergiesHost);

    // For PAIRWISE mode, desolvation and cross-term were already added to groupEnergiesHost
    // and downloaded above. Just download them here for accessors.
    if (receptorMode == IsolatedGBSAForce::PAIRWISE) {
        groupReceptorDesolvations.download(groupReceptorDesolvationsHost);
        groupCrossTermEnergies.download(groupCrossTermEnergiesHost);
    }

    // Sum total energy
    double totalEnergy = 0.0;
    for (int g = 0; g < numParticleGroups; g++) {
        totalEnergy += groupEnergiesHost[g];
    }

    return totalEnergy;
}

void CudaCalcIsolatedGBSAForceKernel::updateParametersInContext(ContextImpl& context,
                                                                 const IsolatedGBSAForce& force) {
    // Re-upload atom parameters
    vector<float> chargesVec(numAtoms), radiiVec(numAtoms), scalesVec(numAtoms);
    for (int i = 0; i < numAtoms; i++) {
        double q, r, s;
        force.getAtomParameters(i, q, r, s);
        chargesVec[i] = static_cast<float>(q);
        radiiVec[i] = static_cast<float>(r);
        scalesVec[i] = static_cast<float>(s);
    }
    charges.upload(chargesVec);
    radii.upload(radiiVec);
    scaleFactors.upload(scalesVec);

    // Update solvent parameters
    double soluteDielectric = force.getSoluteDielectric();
    double solventDielectric = force.getSolventDielectric();
    prefactor = static_cast<float>(-ONE_4PI_EPS0 * (1.0/soluteDielectric - 1.0/solventDielectric));

    includeSurfaceArea = force.getIncludeSurfaceArea();
    surfaceTension = static_cast<float>(force.getSurfaceTension());
    cutoffDistance = static_cast<float>(force.getCutoffDistance());
}

double CudaCalcIsolatedGBSAForceKernel::getGroupEnergy(int groupIndex) const {
    if (groupIndex < 0 || groupIndex >= static_cast<int>(groupEnergiesHost.size())) {
        throw OpenMMException("IsolatedGBSAForce: invalid group index");
    }
    return groupEnergiesHost[groupIndex];
}

double CudaCalcIsolatedGBSAForceKernel::getGroupLigandSelfEnergy(int groupIndex) const {
    if (groupIndex < 0 || groupIndex >= static_cast<int>(groupLigandSelfEnergiesHost.size())) {
        throw OpenMMException("IsolatedGBSAForce: invalid group index");
    }
    return groupLigandSelfEnergiesHost[groupIndex];
}

double CudaCalcIsolatedGBSAForceKernel::getGroupReceptorContribution(int groupIndex) const {
    if (groupIndex < 0 || groupIndex >= static_cast<int>(groupReceptorContributionsHost.size())) {
        throw OpenMMException("IsolatedGBSAForce: invalid group index");
    }
    return groupReceptorContributionsHost[groupIndex];
}

double CudaCalcIsolatedGBSAForceKernel::getGroupReceptorDesolvation(int groupIndex) const {
    if (groupIndex < 0 || groupIndex >= static_cast<int>(groupReceptorDesolvationsHost.size())) {
        throw OpenMMException("IsolatedGBSAForce: invalid group index");
    }
    return groupReceptorDesolvationsHost[groupIndex];
}

double CudaCalcIsolatedGBSAForceKernel::getGroupCrossTermEnergy(int groupIndex) const {
    if (groupIndex < 0 || groupIndex >= static_cast<int>(groupCrossTermEnergiesHost.size())) {
        throw OpenMMException("IsolatedGBSAForce: invalid group index");
    }
    return groupCrossTermEnergiesHost[groupIndex];
}

vector<double> CudaCalcIsolatedGBSAForceKernel::getGroupBornRadii(int groupIndex) const {
    if (groupIndex < 0 || groupIndex >= static_cast<int>(groupBornRadiiHost.size())) {
        throw OpenMMException("IsolatedGBSAForce: invalid group index");
    }

    // Download Born radii if not cached
    if (groupBornRadiiHost[groupIndex].empty()) {
        int startIdx = 0;
        int endIdx = 0;
        vector<int> groupStarts(numParticleGroups + 1);
        groupStartIndex.download(groupStarts);
        startIdx = groupStarts[groupIndex];
        endIdx = groupStarts[groupIndex + 1];

        vector<float> allBornRadii(bornRadii.getSize());
        bornRadii.download(allBornRadii);

        groupBornRadiiHost[groupIndex].assign(allBornRadii.begin() + startIdx,
                                               allBornRadii.begin() + endIdx);
    }

    vector<double> result(groupBornRadiiHost[groupIndex].begin(),
                          groupBornRadiiHost[groupIndex].end());
    return result;
}

vector<double> CudaCalcIsolatedGBSAForceKernel::getGroupAtomEnergies(int groupIndex) const {
    if (groupIndex < 0 || groupIndex >= static_cast<int>(groupAtomEnergiesHost.size())) {
        throw OpenMMException("IsolatedGBSAForce: invalid group index");
    }

    // For now, return empty - per-atom energies would need separate kernel
    vector<double> result;
    return result;
}

vector<double> CudaCalcIsolatedGBSAForceKernel::computeHessian(ContextImpl& context) {
    // Hessian computation for GBSA is complex - implement later
    throw OpenMMException("IsolatedGBSAForce: Hessian computation not yet implemented");
}
