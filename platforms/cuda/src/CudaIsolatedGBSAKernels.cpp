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
    receptorLocalityCutoff = static_cast<float>(force.getReceptorLocalityCutoff());

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

        // Allocate constant receptor buffers (per-group buffers allocated after groups are known)
        receptorSelfHCT.initialize<float>(cu, numReceptorAtoms, "isolatedGbsaReceptorSelfHCT");
        receptorBornRadiiRef.initialize<float>(cu, numReceptorAtoms, "isolatedGbsaReceptorBornRadiiRef");
        receptorReferenceEnergy.initialize<float>(cu, 1, "isolatedGbsaReceptorReferenceEnergy");
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

    // Initialize alchemical scaling factors (done after particle group processing
    // so numParticleGroups is finalized)

    // Initialize alchemical scaling factors
    globalScalingFactor = static_cast<float>(force.getGlobalScalingFactor());
    {
        int nGroups = force.getNumParticleGroups();
        std::vector<float> groupScalings(numParticleGroups, 1.0f);
        for (int i = 0; i < nGroups; i++) {
            groupScalings[i] = static_cast<float>(force.getGroupScalingFactor(i));
        }
        groupScalingFactorsBuffer.initialize<float>(cu, numParticleGroups, "isolatedGbsaGroupScalingFactors");
        groupScalingFactorsBuffer.upload(groupScalings);
    }

    // Allocate per-group energy buffers
    groupEnergies.initialize<float>(cu, numParticleGroups, "isolatedGbsaGroupEnergies");
    groupLigandSelfEnergies.initialize<float>(cu, numParticleGroups, "isolatedGbsaGroupLigandSelfEnergies");
    groupReceptorContributions.initialize<float>(cu, numParticleGroups, "isolatedGbsaGroupReceptorContributions");
    groupReceptorDesolvations.initialize<float>(cu, numParticleGroups, "isolatedGbsaGroupReceptorDesolvations");
    groupCrossTermEnergies.initialize<float>(cu, numParticleGroups, "isolatedGbsaGroupCrossTermEnergies");
    groupUnscaledEnergies.initialize<float>(cu, numParticleGroups, "isolatedGbsaGroupUnscaledEnergies");

    groupEnergiesHost.resize(numParticleGroups);
    groupLigandSelfEnergiesHost.resize(numParticleGroups);
    groupReceptorContributionsHost.resize(numParticleGroups);
    groupReceptorDesolvationsHost.resize(numParticleGroups);
    groupCrossTermEnergiesHost.resize(numParticleGroups);
    groupUnscaledEnergiesHost.resize(numParticleGroups);
    groupBornRadiiHost.resize(numParticleGroups);
    groupAtomEnergiesHost.resize(numParticleGroups);

    // Allocate per-group receptor buffers for PAIRWISE mode (now that K is known)
    if (receptorMode == IsolatedGBSAForce::PAIRWISE && numReceptorAtoms > 0) {
        ligandToReceptorHCT.initialize<float>(cu, numReceptorAtoms * numParticleGroups, "isolatedGbsaLigandToReceptorHCT");
        receptorBornRadii.initialize<float>(cu, numReceptorAtoms * numParticleGroups, "isolatedGbsaReceptorBornRadii");
        receptorEnergy.initialize<float>(cu, numParticleGroups, "isolatedGbsaReceptorEnergy");
        receptorDeDR.initialize<float>(cu, numReceptorAtoms * numParticleGroups, "isolatedGbsaReceptorDeDR");

        // Fixed-point accumulators for tiled HCT kernel
        hctReceptorFixed.initialize<unsigned long long>(cu, totalParticles, "hctReceptorFixed");
        ligToRecHCTFixed.initialize<unsigned long long>(cu, numReceptorAtoms * numParticleGroups, "ligToRecHCTFixed");
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
        computeReceptorHCTPairwiseTiledKernel = cu.getKernel(module, "computeIsolatedReceptorHCTPairwiseTiled");
        computeFusedReceptorLigandHCTKernel = cu.getKernel(module, "computeFusedReceptorLigandHCT");
        computeReceptorLigandHCTParallelKernel = cu.getKernel(module, "computeReceptorLigandHCTParallel");
        computeReceptorLigandHCTTiledKernel = cu.getKernel(module, "computeReceptorLigandHCTTiled");
        convertTiledHCTToFloatKernel = cu.getKernel(module, "convertTiledHCTToFloat");
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

        // GPU-side accumulation kernels (eliminate host-device sync)
        accumulateDesolvationOnGPUKernel = cu.getKernel(module, "accumulateDesolvationOnGPU");
        accumulateDesolvationDeltaOnGPUKernel = cu.getKernel(module, "accumulateDesolvationDeltaOnGPU");
        accumulateCrossTermOnGPUKernel = cu.getKernel(module, "accumulateCrossTermOnGPU");

        // Allocate fixed-point buffer for tiled HCT computation
        receptorSelfHCTFixed.initialize<unsigned long long>(cu, numReceptorAtoms, "receptorSelfHCTFixed");

        // Locality cutoff: allocate active atom mask and load kernels
        if (receptorLocalityCutoff > 0.0f) {
            isActiveRecAtom.initialize<int>(cu, numParticleGroups * numReceptorAtoms, "isActiveRecAtom");
            computeActiveReceptorAtomsKernel = cu.getKernel(module, "computeActiveReceptorAtoms");
            computeReceptorEnergyDeltaKernel = cu.getKernel(module, "computeReceptorEnergyDelta");
            computeReceptorDeDRActiveKernel = cu.getKernel(module, "computeReceptorDeDRActive");

            // Per-receptor-atom HCT baseline for frozen inactive contributions
            localityMaskValid = false;
            localityMaskAge = 0;

            hctReceptorPerAtom.initialize<float>(cu, totalParticles * numReceptorAtoms, "hctReceptorPerAtom");
            hctReceptorBaselineSum.initialize<float>(cu, totalParticles, "hctReceptorBaselineSum");
            hasHctBaseline = false;
            computeReceptorHCTPerAtomKernel = cu.getKernel(module, "computeReceptorHCTPerAtom");
            computeBaselineHCTSumKernel = cu.getKernel(module, "computeBaselineHCTSum");
            reconstructReceptorHCTKernel = cu.getKernel(module, "reconstructReceptorHCT");
            reconstructReceptorHCTFastKernel = cu.getKernel(module, "reconstructReceptorHCTFast");

            // Build receptor cell list for spatial neighbor lookup
            // Cell size = locality cutoff. Each cell contains receptor atoms in that region.
            const auto& recPos = force.getReceptorPositions();
            float3 recMin = make_float3(1e30f, 1e30f, 1e30f);
            float3 recMax = make_float3(-1e30f, -1e30f, -1e30f);
            for (int i = 0; i < numReceptorAtoms; i++) {
                float x = static_cast<float>(recPos[i*3]);
                float y = static_cast<float>(recPos[i*3+1]);
                float z = static_cast<float>(recPos[i*3+2]);
                recMin.x = std::min(recMin.x, x); recMax.x = std::max(recMax.x, x);
                recMin.y = std::min(recMin.y, y); recMax.y = std::max(recMax.y, y);
                recMin.z = std::min(recMin.z, z); recMax.z = std::max(recMax.z, z);
            }
            // Pad by cutoff so ligand atoms near edges still find neighbors
            float pad = receptorLocalityCutoff;
            cellOriginX = recMin.x - pad;
            cellOriginY = recMin.y - pad;
            cellOriginZ = recMin.z - pad;
            cellSize = receptorLocalityCutoff;
            cellNx = std::max(1, (int)ceilf((recMax.x + pad - cellOriginX) / cellSize));
            cellNy = std::max(1, (int)ceilf((recMax.y + pad - cellOriginY) / cellSize));
            cellNz = std::max(1, (int)ceilf((recMax.z + pad - cellOriginZ) / cellSize));
            int numCells = cellNx * cellNy * cellNz;

            // Count atoms per cell
            std::vector<int> cellCount(numCells, 0);
            std::vector<int> atomCell(numReceptorAtoms);
            for (int i = 0; i < numReceptorAtoms; i++) {
                int cx = std::min(cellNx-1, std::max(0, (int)((recPos[i*3] - cellOriginX) / cellSize)));
                int cy = std::min(cellNy-1, std::max(0, (int)((recPos[i*3+1] - cellOriginY) / cellSize)));
                int cz = std::min(cellNz-1, std::max(0, (int)((recPos[i*3+2] - cellOriginZ) / cellSize)));
                int cell = cx * cellNy * cellNz + cy * cellNz + cz;
                atomCell[i] = cell;
                cellCount[cell]++;
            }
            // Build cell start array (prefix sum)
            std::vector<int> cellStartHost(numCells + 1, 0);
            for (int c = 0; c < numCells; c++)
                cellStartHost[c + 1] = cellStartHost[c] + cellCount[c];
            // Build sorted atom index
            std::vector<int> atomIndexHost(numReceptorAtoms);
            std::vector<int> cellFill(numCells, 0);
            for (int i = 0; i < numReceptorAtoms; i++) {
                int c = atomCell[i];
                atomIndexHost[cellStartHost[c] + cellFill[c]] = i;
                cellFill[c]++;
            }
            // Upload to GPU
            cellAtomIndex.initialize<int>(cu, numReceptorAtoms, "cellAtomIndex");
            cellAtomIndex.upload(atomIndexHost);
            cellStart.initialize<int>(cu, numCells + 1, "cellStart");
            cellStart.upload(cellStartHost);
            hasCellList = true;

            // Load cell-list-based HCT kernel
            computeReceptorHCTCellListKernel = cu.getKernel(module, "computeIsolatedReceptorHCTCellList");
            reconstructReceptorHCTCellListKernel = cu.getKernel(module, "reconstructReceptorHCTCellList");
        }

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

    // Clear energy and intermediate buffers (async GPU clears — no CPU-GPU sync)
    cu.clearBuffer(groupEnergies);
    cu.clearBuffer(groupLigandSelfEnergies);
    cu.clearBuffer(groupReceptorContributions);
    cu.clearBuffer(groupReceptorDesolvations);
    cu.clearBuffer(groupUnscaledEnergies);
    cu.clearBuffer(groupCrossTermEnergies);
    cu.clearBuffer(hctReceptor);
    cu.clearBuffer(hctLigand);

    // Get device pointers
    CUdeviceptr posqPtr = cu.getPosq().getDevicePointer();
    fusedHCTComputed_ = false;

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

    // Step 0: Compute per-group active receptor atom masks (if locality cutoff enabled)
    // Cached across HMC steps within a sweep — recompute periodically.
    // Within a 200-step HMC trajectory, ligand positions change ~0.01nm per step,
    // far less than the 2.0nm locality cutoff, so the mask is stable.
    if (receptorMode == IsolatedGBSAForce::PAIRWISE && receptorLocalityCutoff > 0.0f) {
        localityMaskAge++;
        if (!localityMaskValid || localityMaskAge >= 200) {
            CUdeviceptr receptorPosPtr0 = receptorPositions.getDevicePointer();
            CUdeviceptr isActiveRecAtomPtr0 = isActiveRecAtom.getDevicePointer();
            float localityCutoff2 = receptorLocalityCutoff * receptorLocalityCutoff;
            int recBlockSize0 = 256;
            int totalMaskWork = numParticleGroups * numReceptorAtoms;
            int recNumBlocks0 = (totalMaskWork + recBlockSize0 - 1) / recBlockSize0;
            CUdeviceptr groupScalingFactorsPtr0 = groupScalingFactorsBuffer.getDevicePointer();
            void* activeArgs[] = {
                &posqPtr, &particleIndicesPtr, &receptorPosPtr0,
                &numReceptorAtoms, &groupStartPtr, &numParticleGroups,
                &localityCutoff2, &isActiveRecAtomPtr0,
                &globalScalingFactor, &groupScalingFactorsPtr0
            };
            cu.executeKernel(computeActiveReceptorAtomsKernel, activeArgs, recNumBlocks0 * recBlockSize0, recBlockSize0);
            localityMaskValid = true;
            localityMaskAge = 0;
        }
    }

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

        bool useLocalityHCT = (receptorLocalityCutoff > 0.0f);

        if (useLocalityHCT && !hasHctBaseline) {
            // First execute: compute per-receptor-atom HCT baseline
            CUdeviceptr hctPerAtomPtr = hctReceptorPerAtom.getDevicePointer();
            int totalWork = totalParticles * numReceptorAtoms;
            int perAtomBlocks = (totalWork + blockSize - 1) / blockSize;
            void* perAtomArgs[] = {
                &posqPtr, &particleIndicesPtr, &radiiPtr,
                &receptorPosPtr, &receptorRadiiPtr, &receptorScalesPtr,
                &numReceptorAtoms, &groupStartPtr, &numParticleGroups,
                &totalParticles, &numAtoms, &hctPerAtomPtr
            };
            cu.executeKernel(computeReceptorHCTPerAtomKernel, perAtomArgs, perAtomBlocks * blockSize, blockSize);

            // Compute baseline sum per ligand atom
            CUdeviceptr baselineSumPtr = hctReceptorBaselineSum.getDevicePointer();
            int sumBlocks = (totalParticles + blockSize - 1) / blockSize;
            void* sumArgs[] = {
                &hctPerAtomPtr, &numReceptorAtoms, &totalParticles, &baselineSumPtr
            };
            cu.executeKernel(computeBaselineHCTSumKernel, sumArgs, sumBlocks * blockSize, blockSize);

            hasHctBaseline = true;

            // Full HCT for first frame using cell list (no active mask pruning)
            CUdeviceptr cellAtomIdxPtr = cellAtomIndex.getDevicePointer();
            CUdeviceptr cellStartPtr2 = cellStart.getDevicePointer();
            void* receptorArgs[] = {
                &posqPtr, &particleIndicesPtr, &radiiPtr,
                &receptorPosPtr, &receptorRadiiPtr, &receptorScalesPtr,
                &cellAtomIdxPtr, &cellStartPtr2,
                &numReceptorAtoms,
                &cellNx, &cellNy, &cellNz,
                &cellOriginX, &cellOriginY, &cellOriginZ, &cellSize,
                &groupStartPtr, &numParticleGroups,
                &totalParticles, &numAtoms, &cutoffDistance, &hctReceptorPtr
            };
            cu.executeKernel(computeReceptorHCTCellListKernel, receptorArgs, numBlocks * blockSize, blockSize);
        } else if (useLocalityHCT) {
            // Cell list reconstruction: baseline sum + fresh for active atoms in nearby cells
            CUdeviceptr isActiveRecAtomPtr = isActiveRecAtom.getDevicePointer();
            CUdeviceptr hctPerAtomPtr = hctReceptorPerAtom.getDevicePointer();
            CUdeviceptr baselineSumPtr = hctReceptorBaselineSum.getDevicePointer();
            CUdeviceptr cellAtomIdxPtr = cellAtomIndex.getDevicePointer();
            CUdeviceptr cellStartPtr2 = cellStart.getDevicePointer();
            // Effective cutoff for this call (uses base cutoff; scaling handled per-group in mask)
            float effCutoff = receptorLocalityCutoff;
            void* reconstructArgs[] = {
                &posqPtr, &particleIndicesPtr, &radiiPtr,
                &receptorPosPtr, &receptorRadiiPtr, &receptorScalesPtr,
                &hctPerAtomPtr, &baselineSumPtr, &isActiveRecAtomPtr,
                &cellAtomIdxPtr, &cellStartPtr2,
                &numReceptorAtoms,
                &cellNx, &cellNy, &cellNz,
                &cellOriginX, &cellOriginY, &cellOriginZ, &cellSize,
                &effCutoff,
                &groupStartPtr, &numParticleGroups,
                &totalParticles, &numAtoms, &cutoffDistance, &hctReceptorPtr
            };
            cu.executeKernel(reconstructReceptorHCTCellListKernel, reconstructArgs, numBlocks * blockSize, blockSize);
        } else {
            // No locality: rectangular tiled HCT (OpenMM-style, both directions)
            CUdeviceptr hctRecFixedPtr = hctReceptorFixed.getDevicePointer();
            CUdeviceptr ligToRecFixedPtr = ligToRecHCTFixed.getDevicePointer();

            // Clear fixed-point accumulators
            cu.clearBuffer(hctReceptorFixed);
            cu.clearBuffer(ligToRecHCTFixed);

            // Compute tile count
            int ligGroupSize = totalParticles / numParticleGroups;  // atoms per group
            int numLigBlocks = (ligGroupSize + 31) / 32;
            int numRecBlocks = (numReceptorAtoms + 31) / 32;
            int tilesPerGroup = numRecBlocks * numLigBlocks;
            int totalTiles = tilesPerGroup * numParticleGroups;

            // Launch: 1 warp per tile, 8 warps per block
            int tiledBlockSize = 256;
            int totalWarps = totalTiles;
            int tiledBlocks = (totalWarps * 32 + tiledBlockSize - 1) / tiledBlockSize;

            void* tiledArgs[] = {
                &posqPtr, &particleIndicesPtr, &radiiPtr, &scaleFactorsPtr,
                &receptorPosPtr, &receptorRadiiPtr, &receptorScalesPtr,
                &numReceptorAtoms, &groupStartPtr, &numParticleGroups,
                &numAtoms, &cutoffDistance,
                &hctRecFixedPtr, &ligToRecFixedPtr, &tilesPerGroup
            };
            cu.executeKernel(computeReceptorLigandHCTTiledKernel, tiledArgs, tiledBlocks * tiledBlockSize, tiledBlockSize);

            // Convert fixed-point to float
            int convBlocks = (totalParticles + blockSize - 1) / blockSize;
            void* convArgs1[] = { &hctRecFixedPtr, &hctReceptorPtr, &totalParticles };
            cu.executeKernel(convertTiledHCTToFloatKernel, convArgs1, convBlocks * blockSize, blockSize);

            int ligRecTotal = numReceptorAtoms * numParticleGroups;
            CUdeviceptr ligandToReceptorHCTPtr2 = ligandToReceptorHCT.getDevicePointer();
            int convBlocks2 = (ligRecTotal + blockSize - 1) / blockSize;
            void* convArgs2[] = { &ligToRecFixedPtr, &ligandToReceptorHCTPtr2, &ligRecTotal };
            cu.executeKernel(convertTiledHCTToFloatKernel, convArgs2, convBlocks2 * blockSize, blockSize);

            fusedHCTComputed_ = true;
        }
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

    // Get scaling factor device pointers
    CUdeviceptr groupScalingFactorsPtr = groupScalingFactorsBuffer.getDevicePointer();
    CUdeviceptr groupUnscaledEnergiesPtr = groupUnscaledEnergies.getDevicePointer();

    // Step 4: Compute GB energy and forces
    void* energyArgs[] = {
        &posqPtr, &particleIndicesPtr, &chargesPtr, &bornRadiiPtr,
        &groupStartPtr, &numParticleGroups, &numAtoms, &prefactor,
        &forcePtr, &groupEnergiesPtr, &groupLigandEnergiesPtr, &paddedNumAtoms,
        &globalScalingFactor, &groupScalingFactorsPtr, &groupUnscaledEnergiesPtr
    };
    cu.executeKernel(computeGBEnergyKernel, energyArgs, numBlocks * blockSize, blockSize);

    // Step 4b: PAIRWISE mode - receptor desolvation and cross-term energy
    // All accumulation done on GPU to avoid host-device sync points.
    if (receptorMode == IsolatedGBSAForce::PAIRWISE) {
        CUdeviceptr receptorPosPtr = receptorPositions.getDevicePointer();
        CUdeviceptr receptorRadiiPtr = receptorRadii.getDevicePointer();
        CUdeviceptr receptorScalesPtr = receptorScaleFactors.getDevicePointer();
        CUdeviceptr receptorChargesPtr = receptorCharges.getDevicePointer();
        CUdeviceptr receptorSelfHCTPtr = receptorSelfHCT.getDevicePointer();
        CUdeviceptr ligandToReceptorHCTPtr = ligandToReceptorHCT.getDevicePointer();
        CUdeviceptr receptorBornRadiiPtr = receptorBornRadii.getDevicePointer();
        CUdeviceptr receptorBornRadiiRefPtr = receptorBornRadiiRef.getDevicePointer();
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
        int totalWarps = (numTiles + 7) / 8 * 8;
        int recNumBlocksTiled = max((totalWarps * TILE_SIZE + recBlockSize - 1) / recBlockSize, (numTiles + 7) / 8);

        bool useLocality = (receptorLocalityCutoff > 0.0f);
        CUdeviceptr isActiveRecAtomPtr = useLocality ? isActiveRecAtom.getDevicePointer() : (CUdeviceptr)0;

        // 4b.1: Compute ligand→receptor HCT
        // Skip if fused kernel already computed this in Step 1
        if (!fusedHCTComputed_) {
            int ligRecWorkItems = numParticleGroups * numReceptorAtoms;
            int ligRecBlocks = (ligRecWorkItems + blockSize - 1) / blockSize;
            void* ligToRecHctArgs[] = {
                &posqPtr, &particleIndicesPtr, &radiiPtr, &scaleFactorsPtr,
                &receptorPosPtr, &receptorRadiiPtr, &groupStartPtr,
                &numParticleGroups, &numReceptorAtoms, &numAtoms, &cutoffDistance, &ligandToReceptorHCTPtr,
                &isActiveRecAtomPtr
            };
            cu.executeKernel(computeLigandToReceptorHCTKernel, ligToRecHctArgs, ligRecBlocks * blockSize, blockSize);
        }

        CUdeviceptr receptorDeDRPtr = receptorDeDR.getDevicePointer();

        // 4b.2: Receptor Born radii for ALL groups (single batched launch)
        int totalBornWork = numParticleGroups * numReceptorAtoms;
        int bornBlocks = (totalBornWork + recBlockSize - 1) / recBlockSize;
        CUdeviceptr groupScalingFactorsPtr = groupScalingFactorsBuffer.getDevicePointer();
        void* recBornArgs[] = {
            &receptorRadiiPtr, &receptorSelfHCTPtr, &ligandToReceptorHCTPtr,
            &numReceptorAtoms, &numParticleGroups, &receptorBornRadiiPtr,
            &receptorBornRadiiRefPtr, &isActiveRecAtomPtr,
            &globalScalingFactor, &groupScalingFactorsPtr
        };
        cu.executeKernel(computeReceptorBornRadiiWithLigandKernel, recBornArgs, bornBlocks * recBlockSize, recBlockSize);

        // 4b.3: Receptor desolvation energy for ALL groups
        cu.clearBuffer(receptorEnergy);

        if (!useLocality) {
            // Full O(N²) receptor energy — must still loop per group for tiled kernel
            for (int g = 0; g < numParticleGroups; g++) {
                // Point to this group's Born radii via offset pointer
                CUdeviceptr groupBornRadiiPtr = receptorBornRadiiPtr + g * numReceptorAtoms * sizeof(float);

                // Use slot 0 of receptorEnergy as scratch (cleared each iteration)
                cu.clearBuffer(receptorEnergy);

                void* recEnergyArgs[] = {
                    &receptorPosPtr, &receptorChargesPtr, &groupBornRadiiPtr,
                    &numReceptorAtoms, &prefactor, &receptorEnergyPtr, &numTiles
                };
                cu.executeKernel(computeReceptorGBEnergyTiledKernel, recEnergyArgs, recNumBlocksTiled * recBlockSize, recBlockSize);

                void* accumArgs[] = {
                    &receptorEnergyPtr, &receptorReferenceEnergyValue, &g,
                    &globalScalingFactor, &groupScalingFactorsPtr,
                    &groupEnergiesPtr, &groupDesolvPtr, &groupUnscaledEnergiesPtr
                };
                cu.executeKernel(accumulateDesolvationOnGPUKernel, accumArgs, 1, 1);
            }
        } else {
            // Delta energy for ALL groups (single batched launch)
            void* deltaEnergyArgs[] = {
                &receptorPosPtr, &receptorChargesPtr, &receptorBornRadiiRefPtr,
                &receptorBornRadiiPtr, &isActiveRecAtomPtr,
                &numReceptorAtoms, &numParticleGroups, &prefactor, &receptorEnergyPtr,
                &globalScalingFactor, &groupScalingFactorsPtr
            };
            cu.executeKernel(computeReceptorEnergyDeltaKernel, deltaEnergyArgs, bornBlocks * recBlockSize, recBlockSize);

            // Accumulate all groups' desolvation at once
            for (int g = 0; g < numParticleGroups; g++) {
                CUdeviceptr groupEnergyPtr = receptorEnergyPtr + g * sizeof(float);
                void* accumArgs[] = {
                    &groupEnergyPtr, &g,
                    &globalScalingFactor, &groupScalingFactorsPtr,
                    &groupEnergiesPtr, &groupDesolvPtr, &groupUnscaledEnergiesPtr
                };
                cu.executeKernel(accumulateDesolvationDeltaOnGPUKernel, accumArgs, 1, 1);
            }
        }

        // 4b.3b: Receptor dE/dR and desolvation forces for ALL groups (batched)
        if (includeForces) {
            if (useLocality) {
                // Batched dE/dR for all groups
                void* deDRArgs[] = {
                    &receptorPosPtr, &receptorChargesPtr, &receptorBornRadiiPtr,
                    &isActiveRecAtomPtr, &numReceptorAtoms, &numParticleGroups, &prefactor, &receptorDeDRPtr
                };
                cu.executeKernel(computeReceptorDeDRActiveKernel, deDRArgs, bornBlocks * recBlockSize, recBlockSize);
            } else {
                // Non-locality: dE/dR for all groups (batched via computeReceptorDeDRSimple per group)
                // TODO: batch this too — for now loop per group
                for (int g = 0; g < numParticleGroups; g++) {
                    CUdeviceptr groupBornRadiiPtr = receptorBornRadiiPtr + g * numReceptorAtoms * sizeof(float);
                    CUdeviceptr groupDeDRPtr = receptorDeDRPtr + g * numReceptorAtoms * sizeof(float);
                    void* deDRArgs[] = {
                        &receptorPosPtr, &receptorChargesPtr, &groupBornRadiiPtr,
                        &numReceptorAtoms, &prefactor, &groupDeDRPtr
                    };
                    cu.executeKernel(computeReceptorDeDRSimpleKernel, deDRArgs, recNumBlocksSimple * recBlockSize, recBlockSize);
                }
            }

            // Desolvation forces — all groups, reads per-group dE/dR and Born radii
            void* recDesolvForceArgs[] = {
                &posqPtr, &particleIndicesPtr, &radiiPtr, &scaleFactorsPtr,
                &receptorPosPtr, &receptorRadiiPtr,
                &receptorSelfHCTPtr, &ligandToReceptorHCTPtr, &receptorBornRadiiPtr, &receptorDeDRPtr,
                &groupStartPtr, &numParticleGroups, &numReceptorAtoms, &numAtoms,
                &cutoffDistance, &forcePtr, &paddedNumAtoms,
                &globalScalingFactor, &groupScalingFactorsPtr
            };
            cu.executeKernel(computeReceptorDesolvationForcesOptimizedKernel, recDesolvForceArgs, numBlocks * blockSize, blockSize);
        }

        // 4b.4: Cross-term energy (all on GPU)
        void* crossTermArgs[] = {
            &posqPtr, &particleIndicesPtr, &chargesPtr, &bornRadiiPtr,
            &receptorPosPtr, &receptorChargesPtr, &receptorBornRadiiPtr,
            &groupStartPtr, &numParticleGroups, &numReceptorAtoms, &numAtoms, &prefactor,
            &groupCrossTermPtr, &forcePtr, &paddedNumAtoms,
            &globalScalingFactor, &groupScalingFactorsPtr, &isActiveRecAtomPtr
        };
        cu.executeKernel(computeCrossTermGBEnergyKernel, crossTermArgs, numBlocks * blockSize, blockSize);

        // GPU-side: add cross-term to group energies
        void* crossAccumArgs[] = {
            &groupCrossTermPtr, &numParticleGroups,
            &globalScalingFactor, &groupScalingFactorsPtr,
            &groupEnergiesPtr, &groupUnscaledEnergiesPtr
        };
        cu.executeKernel(accumulateCrossTermOnGPUKernel, crossAccumArgs, 1, 1);
    }

    // Step 5: Optional surface area term
    if (includeSurfaceArea) {
        float probe = (receptorMode == IsolatedGBSAForce::GRID) ? probeRadius : 0.14f;
        void* saArgs[] = {
            &radiiPtr, &bornRadiiPtr, &groupStartPtr,
            &numParticleGroups, &numAtoms,
            &surfaceTension, &probe, &groupEnergiesPtr,
            &globalScalingFactor, &groupScalingFactorsPtr, &groupUnscaledEnergiesPtr
        };
        cu.executeKernel(computeSAEnergyKernel, saArgs, numBlocks * blockSize, blockSize);
    }

    // Step 6: Chain rule forces through Born radii
    if (includeForces) {
        CUdeviceptr dE_dRPtr = dE_dR.getDevicePointer();

        // Accumulate dE/dR_born (scaled by alchemical factors)
        void* bornDerivArgs[] = {
            &posqPtr, &particleIndicesPtr, &chargesPtr, &bornRadiiPtr,
            &groupStartPtr, &numParticleGroups, &numAtoms, &prefactor, &dE_dRPtr,
            &globalScalingFactor, &groupScalingFactorsPtr
        };
        cu.executeKernel(accumulateBornRadiiDerivativesKernel, bornDerivArgs, numBlocks * blockSize, blockSize);

        if (includeSurfaceArea) {
            float probe = (receptorMode == IsolatedGBSAForce::GRID) ? probeRadius : 0.14f;
            void* saDerivArgs[] = {
                &radiiPtr, &bornRadiiPtr, &groupStartPtr,
                &numParticleGroups, &numAtoms, &surfaceTension, &probe, &dE_dRPtr,
                &globalScalingFactor, &groupScalingFactorsPtr
            };
            cu.executeKernel(accumulateSADerivativesKernel, saDerivArgs, numBlocks * blockSize, blockSize);
        }

        // Ligand-ligand HCT chain rule forces (scaling propagates via dE_dR)
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

            // Chain rule for receptor→ligand HCT (how receptor screens ligand Born radii)
            void* receptorChainArgs[] = {
                &posqPtr, &particleIndicesPtr, &radiiPtr,
                &bornRadiiPtr, &hctReceptorPtr, &hctLigandPtr, &dE_dRPtr,
                &receptorPosPtr, &receptorRadiiPtr, &receptorScalesPtr,
                &numReceptorAtoms, &groupStartPtr, &numParticleGroups,
                &totalParticles, &numAtoms, &cutoffDistance, &forcePtr, &paddedNumAtoms
            };
            cu.executeKernel(computeReceptorHCTPairwiseChainRuleKernel, receptorChainArgs, numBlocks * blockSize, blockSize);

            // Receptor desolvation forces were computed in Step 4b.3b (batched)

            // Cross-term chain rule forces (all groups, per-group receptor Born radii)
            CUdeviceptr isActiveRecAtomPtr3 = (receptorLocalityCutoff > 0.0f) ? isActiveRecAtom.getDevicePointer() : (CUdeviceptr)0;
            void* crossChainArgs[] = {
                &posqPtr, &particleIndicesPtr, &radiiPtr, &scaleFactorsPtr, &chargesPtr,
                &bornRadiiPtr, &hctReceptorPtr, &hctLigandPtr,
                &receptorPosPtr, &receptorRadiiPtr, &receptorScalesPtr, &receptorChargesPtr,
                &receptorSelfHCTPtr, &ligandToReceptorHCTPtr, &receptorBornRadiiPtr,
                &groupStartPtr, &numParticleGroups, &numReceptorAtoms, &numAtoms,
                &prefactor, &cutoffDistance, &forcePtr, &paddedNumAtoms,
                &globalScalingFactor, &groupScalingFactorsPtr, &isActiveRecAtomPtr3
            };
            cu.executeKernel(computeCrossTermChainRuleForcesKernel, crossChainArgs, numBlocks * blockSize, blockSize);
        }
    }

    // Download group energies only when energy is needed to avoid sync barriers
    if (includeEnergy && !skipGroupEnergyDownload_) {
        groupEnergies.download(groupEnergiesHost);
        groupLigandSelfEnergies.download(groupLigandSelfEnergiesHost);
        groupUnscaledEnergies.download(groupUnscaledEnergiesHost);

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

    return 0.0;
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
    receptorLocalityCutoff = static_cast<float>(force.getReceptorLocalityCutoff());

    // Invalidate locality mask cache (scaling changed, may need new mask)
    localityMaskValid = false;

    // Update alchemical scaling factors
    globalScalingFactor = static_cast<float>(force.getGlobalScalingFactor());
    int nGroups = force.getNumParticleGroups();
    if (nGroups > 0) {
        std::vector<float> groupScalings(numParticleGroups, 1.0f);
        for (int i = 0; i < nGroups; i++) {
            groupScalings[i] = static_cast<float>(force.getGroupScalingFactor(i));
        }
        groupScalingFactorsBuffer.upload(groupScalings);
    }
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

vector<double> CudaCalcIsolatedGBSAForceKernel::getReceptorBornRadii(int groupIndex) const {
    if (receptorMode != IsolatedGBSAForce::PAIRWISE) {
        throw OpenMMException("IsolatedGBSAForce: receptor Born radii only available in PAIRWISE mode");
    }
    if (groupIndex < 0 || groupIndex >= numParticleGroups) {
        throw OpenMMException("IsolatedGBSAForce: invalid group index");
    }

    // Ensure cache is sized correctly
    if (groupReceptorBornRadiiHost.size() < static_cast<size_t>(numParticleGroups)) {
        groupReceptorBornRadiiHost.resize(numParticleGroups);
    }

    // Download receptor Born radii (always recompute - could cache based on group if needed)
    vector<float> recBornRadii(numReceptorAtoms);
    receptorBornRadii.download(recBornRadii);
    groupReceptorBornRadiiHost[groupIndex] = recBornRadii;

    vector<double> result(recBornRadii.begin(), recBornRadii.end());
    return result;
}

vector<double> CudaCalcIsolatedGBSAForceKernel::getParticleGroupUnscaledEnergies() const {
    vector<double> result(groupUnscaledEnergiesHost.begin(), groupUnscaledEnergiesHost.end());
    return result;
}

vector<double> CudaCalcIsolatedGBSAForceKernel::computeHessian(ContextImpl& context) {
    // Hessian computation for GBSA is complex - implement later
    throw OpenMMException("IsolatedGBSAForce: Hessian computation not yet implemented");
}
