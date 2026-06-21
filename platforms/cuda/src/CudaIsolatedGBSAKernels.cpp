/* -------------------------------------------------------------------------- *
 *                              OpenMMGridForce                               *
 * -------------------------------------------------------------------------- */

#include "CudaIsolatedGBSAKernels.h"
#include "CudaGridForceKernelSources.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/cuda/CudaBondedUtilities.h"
#include "openmm/cuda/CudaForceInfo.h"
#include "openmm/OpenMMException.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <utility>
#include <vector>

using namespace GridForcePlugin;
using namespace OpenMM;
using namespace std;

// Coulomb constant in kJ*nm/mol/e^2
static const double ONE_4PI_EPS0 = 138.935456;

// Energy buffers follow the context precision (mixed = double in mixed/double),
// so high-magnitude energies are not narrowed to fp32 on accumulation/readback.
static int mixedEnergyElementSize(OpenMM::CudaContext& cu) {
    return (cu.getUseDoublePrecision() || cu.getUseMixedPrecision()) ? sizeof(double) : sizeof(float);
}

static void initMixedEnergyBuffer(OpenMM::CudaContext& cu, OpenMM::CudaArray& buf, int n, const char* name) {
    buf.initialize(cu, n, mixedEnergyElementSize(cu), name);
}

static void downloadMixedEnergy(OpenMM::CudaContext& cu, OpenMM::CudaArray& buf, std::vector<double>& out) {
    if (cu.getUseDoublePrecision() || cu.getUseMixedPrecision()) {
        buf.download(out);
    } else {
        std::vector<float> tmp(out.size());
        buf.download(tmp);
        out.assign(tmp.begin(), tmp.end());
    }
}

// Geometry/parameter buffers that feed the HCT/Born/GB math follow the context
// precision: in double mode `real` is double, so receptor positions and the
// scalar parameter arrays must be uploaded as double4/double to avoid silently
// narrowing the reference geometry to single precision. In single/mixed mode
// `real` is float, so these remain float4/float and the numerics are unchanged.
static int realElementSize(OpenMM::CudaContext& cu) {
    return cu.getUseDoublePrecision() ? sizeof(double) : sizeof(float);
}

static int real4ElementSize(OpenMM::CudaContext& cu) {
    return cu.getUseDoublePrecision() ? sizeof(double4) : sizeof(float4);
}

// Upload an x/y/z position array as real4 (w=0) at context precision.
static void uploadRealPositions(OpenMM::CudaContext& cu, OpenMM::CudaArray& buf,
                                const std::vector<double>& xyz, int n, const char* name) {
    buf.initialize(cu, n, real4ElementSize(cu), name);
    if (cu.getUseDoublePrecision()) {
        std::vector<double4> p(n);
        for (int i = 0; i < n; i++)
            p[i] = make_double4(xyz[i*3], xyz[i*3+1], xyz[i*3+2], 0.0);
        buf.upload(p);
    } else {
        std::vector<float4> p(n);
        for (int i = 0; i < n; i++)
            p[i] = make_float4((float)xyz[i*3], (float)xyz[i*3+1], (float)xyz[i*3+2], 0.0f);
        buf.upload(p);
    }
}

// Allocate a compute buffer of `real` elements at context precision.
static void initRealBuffer(OpenMM::CudaContext& cu, OpenMM::CudaArray& buf, int n, const char* name) {
    buf.initialize(cu, n, realElementSize(cu), name);
}

// Upload a scalar parameter array as real at context precision.
static void uploadRealScalars(OpenMM::CudaContext& cu, OpenMM::CudaArray& buf,
                              const std::vector<double>& vals, const char* name) {
    int n = (int)vals.size();
    buf.initialize(cu, n, realElementSize(cu), name);
    if (cu.getUseDoublePrecision()) {
        buf.upload(vals);
    } else {
        std::vector<float> f(vals.begin(), vals.end());
        buf.upload(f);
    }
}

CudaCalcIsolatedGBSAForceKernel::CudaCalcIsolatedGBSAForceKernel(string name, const Platform& platform,
                                                                   CudaContext& cu)
    : CalcIsolatedGBSAForceKernel(name, platform), cu(cu), hasInitializedKernel(false),
      numAtoms(0), numParticleGroups(0),
      gbMethod(IsolatedGBSAForce::OBC_II),
      receptorMode(IsolatedGBSAForce::NONE),
      prefactor(0), includeSurfaceArea(false), surfaceTension(0), cutoffDistance(-1.0f),
      originX(0), originY(0), originZ(0), gridSpacing(0), probeRadius(0),
      numBins(0), interpolationMethod(0), hasHctDerivatives(false),
      useKDECorrections(false), hasBinnedKDEDerivatives(false),
      computeCrossTermGrid(false), crossTermNumBins(0),
      numReceptorAtoms(0), receptorReferenceEnergyValue(0.0f),
      computeReceptorHCTGridKernel(nullptr),
      generateCrossTermGridKernel(nullptr),
      computeCrossTermFromGridKernel(nullptr),
      computeReceptorHCTPairwiseKernel(nullptr),
      computeLigandHCTKernel(nullptr),
      computeBornRadiiHCTKernel(nullptr), computeBornRadiiOBCKernel(nullptr),
      computeGBEnergyKernel(nullptr), computeSAEnergyKernel(nullptr),
      computeReceptorDeltaSAKernel(nullptr),
      accumulateBornRadiiDerivativesKernel(nullptr), accumulateSADerivativesKernel(nullptr),
      computeHCTChainRuleForcesKernel(nullptr),
      computeReceptorHCTGradientForceKernel(nullptr),
      computeReceptorHCTPairwiseChainRuleKernel(nullptr),
      computeHessianKernel(nullptr),
      prepareHessianIntermediatesKernel(nullptr),
      computeHCTJacobianPairwiseKernel(nullptr),
      computeReceptorPairwiseHessianKernel(nullptr),
      computeBornCouplingMatrixKernel(nullptr),
      assembleGBSAHessianKernel(nullptr),
      prepareHessianIntermediatesDoubleKernel(nullptr),
      computeHCTJacobianPairwiseDoubleKernel(nullptr),
      computeReceptorPairwiseHessianDoubleKernel(nullptr),
      computeBornCouplingMatrixDoubleKernel(nullptr),
      assembleGBSAHessianDoubleKernel(nullptr),
      convertTiledHCTToDoubleKernel(nullptr),
      computeBornRadiiOBCDoubleKernel(nullptr),
      computeHctReceptorPairwiseDoubleKernel(nullptr),
      computeHctLigandPairwiseDoubleKernel(nullptr) {
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
    prefactor = -ONE_4PI_EPS0 * (1.0/soluteDielectric - 1.0/solventDielectric);

    includeSurfaceArea = force.getIncludeSurfaceArea();
    surfaceTension = static_cast<float>(force.getSurfaceTension());
    interpolationMethod = force.getInterpolationMethod();

    // Upload ligand atom parameters at context precision (real)
    vector<double> chargesVec(numAtoms), radiiVec(numAtoms), scalesVec(numAtoms);
    for (int i = 0; i < numAtoms; i++) {
        double q, r, s;
        force.getAtomParameters(i, q, r, s);
        chargesVec[i] = q;
        radiiVec[i] = r;
        scalesVec[i] = s;
    }
    uploadRealScalars(cu, charges, chargesVec, "isolatedGbsaCharges");
    uploadRealScalars(cu, radii, radiiVec, "isolatedGbsaRadii");
    uploadRealScalars(cu, scaleFactors, scalesVec, "isolatedGbsaScaleFactors");

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

        // Detect correction grid mode by size (same logic as GBSAGridForce)
        const auto& corrNData = grid->getCorrectionN();
        size_t expectedBinnedKDESize = static_cast<size_t>(numBins) * 27 * numPoints;
        size_t expectedKDESize = static_cast<size_t>(27) * numPoints;
        size_t expectedBinnedSize = static_cast<size_t>(numBins) * numPoints;

        if (corrNData.size() == expectedBinnedKDESize) {
            useKDECorrections = true;
            hasBinnedKDEDerivatives = true;
        } else if (corrNData.size() == expectedKDESize) {
            useKDECorrections = true;
            hasBinnedKDEDerivatives = false;
        } else {
            useKDECorrections = false;
            hasBinnedKDEDerivatives = false;
        }

        int corrSize = static_cast<int>(corrNData.size());
        gridCorrectionN.initialize<float>(cu, corrSize, "isolatedGbsaGridCorrectionN");
        gridCorrectionN.upload(corrNData);
        gridCorrectionA.initialize<float>(cu, corrSize, "isolatedGbsaGridCorrectionA");
        gridCorrectionA.upload(grid->getCorrectionA());
        gridCorrectionB.initialize<float>(cu, corrSize, "isolatedGbsaGridCorrectionB");
        gridCorrectionB.upload(grid->getCorrectionB());

        const auto& thresholds = grid->getRThresholds();
        vector<float> thresholdsFloat(thresholds.begin(), thresholds.end());
        rThresholds.initialize<float>(cu, numBins, "isolatedGbsaRThresholds");
        rThresholds.upload(thresholdsFloat);

        // ---- Cross-term pairwise augment (GRID mode) ----
        // Uses the existing computeCrossTermGBEnergy kernel with
        // ligand Born radii from the HCT grid (already computed)
        // and baseline receptor Born radii (frozen, precomputed once).
        computeCrossTermGrid = force.getComputeCrossTermGrid();
        if (computeCrossTermGrid) {
            int nRec = force.getNumReceptorAtoms();
            if (nRec == 0) {
                throw OpenMMException(
                    "IsolatedGBSAForce: computeCrossTermGrid requires "
                    "receptor atoms (set via setNumReceptorAtoms / "
                    "setReceptorAtomParameters / setReceptorPositions) "
                    "even in GRID mode.");
            }
            const auto& recPos = force.getReceptorPositions();
            if (recPos.size() != static_cast<size_t>(nRec * 3)) {
                throw OpenMMException(
                    "IsolatedGBSAForce: receptor positions size mismatch.");
            }
            const auto& recBornBaseline = force.getReceptorBornRadiiBaseline();
            if ((int)recBornBaseline.size() != nRec) {
                throw OpenMMException(
                    "IsolatedGBSAForce: receptorBornRadiiBaseline must "
                    "have length numReceptorAtoms.");
            }
            numReceptorAtoms = nRec;

            // Upload receptor positions (persistent) at context precision
            uploadRealPositions(cu, receptorPositions, recPos, nRec,
                                "isolatedGbsaReceptorPositions");

            // Upload receptor charges (persistent) at context precision
            vector<double> recChargesD(nRec);
            for (int j = 0; j < nRec; j++) {
                double q, r, s;
                force.getReceptorAtomParameters(j, q, r, s);
                recChargesD[j] = q;
            }
            uploadRealScalars(cu, receptorCharges, recChargesD,
                              "isolatedGbsaReceptorCharges");

            // Upload baseline receptor Born radii, replicated K times
            // (the kernel indexes as receptorBornRadii[group * nRec + j])
            int K = numParticleGroups;
            vector<double> recBornD(K * nRec);
            for (int g = 0; g < K; g++) {
                for (int j = 0; j < nRec; j++) {
                    recBornD[g * nRec + j] = recBornBaseline[j];
                }
            }
            uploadRealScalars(cu, receptorBornRadii, recBornD,
                              "isolatedGbsaReceptorBornRadii");
        }

    } else if (receptorMode == IsolatedGBSAForce::PAIRWISE) {
        numReceptorAtoms = force.getNumReceptorAtoms();
        if (numReceptorAtoms == 0) {
            throw OpenMMException("IsolatedGBSAForce: PAIRWISE mode requires receptor atoms");
        }

        const auto& recPos = force.getReceptorPositions();
        if (recPos.size() != static_cast<size_t>(numReceptorAtoms * 3)) {
            throw OpenMMException("IsolatedGBSAForce: receptor positions size mismatch");
        }

        // Upload receptor positions at context precision (real4)
        uploadRealPositions(cu, receptorPositions, recPos, numReceptorAtoms,
                            "isolatedGbsaReceptorPositions");

        // Local float copy of positions used only for tile bounding spheres
        // (tile-skipping geometry; precision here is irrelevant).
        vector<float3> positionsF(numReceptorAtoms);
        for (int i = 0; i < numReceptorAtoms; i++) {
            positionsF[i] = make_float3(
                static_cast<float>(recPos[i*3]),
                static_cast<float>(recPos[i*3 + 1]),
                static_cast<float>(recPos[i*3 + 2])
            );
        }

        // Upload receptor parameters at context precision
        vector<double> recRadiiD(numReceptorAtoms), recScalesD(numReceptorAtoms), recChargesD(numReceptorAtoms);
        for (int i = 0; i < numReceptorAtoms; i++) {
            double q, r, s;
            force.getReceptorAtomParameters(i, q, r, s);
            recChargesD[i] = q;
            recRadiiD[i] = r;
            recScalesD[i] = s;
        }
        uploadRealScalars(cu, receptorRadii, recRadiiD, "isolatedGbsaReceptorRadii");
        uploadRealScalars(cu, receptorScaleFactors, recScalesD, "isolatedGbsaReceptorScaleFactors");
        uploadRealScalars(cu, receptorCharges, recChargesD, "isolatedGbsaReceptorCharges");

        // Precompute receptor block bounding spheres for tile-skipping
        {
            int numRecBlocks = (numReceptorAtoms + 31) / 32;
            vector<float4> blockBounds(numRecBlocks);
            for (int b = 0; b < numRecBlocks; b++) {
                int start = b * 32;
                int end = min(start + 32, numReceptorAtoms);
                // Compute center
                float cx = 0, cy = 0, cz = 0;
                for (int i = start; i < end; i++) {
                    cx += positionsF[i].x;
                    cy += positionsF[i].y;
                    cz += positionsF[i].z;
                }
                float n = (float)(end - start);
                cx /= n; cy /= n; cz /= n;
                // Compute radius (max distance from center)
                float maxR2 = 0;
                for (int i = start; i < end; i++) {
                    float dx = positionsF[i].x - cx;
                    float dy = positionsF[i].y - cy;
                    float dz = positionsF[i].z - cz;
                    float r2 = dx*dx + dy*dy + dz*dz;
                    if (r2 > maxR2) maxR2 = r2;
                }
                blockBounds[b] = make_float4(cx, cy, cz, sqrtf(maxR2));
            }
            recBlockBounds.initialize<float4>(cu, numRecBlocks, "recBlockBounds");
            recBlockBounds.upload(blockBounds);
        }

        // Allocate constant receptor buffers (per-group buffers allocated after groups are known)
        initRealBuffer(cu, receptorSelfHCT, numReceptorAtoms, "isolatedGbsaReceptorSelfHCT");
        initRealBuffer(cu, receptorBornRadiiRef, numReceptorAtoms, "isolatedGbsaReceptorBornRadiiRef");
        initRealBuffer(cu, receptorReferenceEnergy, 1, "isolatedGbsaReceptorReferenceEnergy");
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
        groupScalingFactorsHostCopy.resize(numParticleGroups, 1.0f);
        for (int i = 0; i < nGroups; i++) {
            groupScalingFactorsHostCopy[i] = static_cast<float>(force.getGroupScalingFactor(i));
        }
        groupScalingFactorsBuffer.initialize<float>(cu, numParticleGroups, "isolatedGbsaGroupScalingFactors");
        groupScalingFactorsBuffer.upload(groupScalingFactorsHostCopy);
    }

    // Allocate per-group energy buffers (mixed precision: double in mixed/double)
    initMixedEnergyBuffer(cu, groupEnergies, numParticleGroups, "isolatedGbsaGroupEnergies");
    initMixedEnergyBuffer(cu, groupLigandSelfEnergies, numParticleGroups, "isolatedGbsaGroupLigandSelfEnergies");
    initMixedEnergyBuffer(cu, groupReceptorContributions, numParticleGroups, "isolatedGbsaGroupReceptorContributions");
    initMixedEnergyBuffer(cu, groupReceptorDesolvations, numParticleGroups, "isolatedGbsaGroupReceptorDesolvations");
    initMixedEnergyBuffer(cu, groupCrossTermEnergies, numParticleGroups, "isolatedGbsaGroupCrossTermEnergies");
    initMixedEnergyBuffer(cu, groupUnscaledEnergies, numParticleGroups, "isolatedGbsaGroupUnscaledEnergies");

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
        initRealBuffer(cu, ligandToReceptorHCT, numReceptorAtoms * numParticleGroups, "isolatedGbsaLigandToReceptorHCT");
        initRealBuffer(cu, receptorBornRadii, numReceptorAtoms * numParticleGroups, "isolatedGbsaReceptorBornRadii");
        initRealBuffer(cu, receptorEnergy, numParticleGroups, "isolatedGbsaReceptorEnergy");
        initRealBuffer(cu, receptorDeDR, numReceptorAtoms * numParticleGroups, "isolatedGbsaReceptorDeDR");
        initRealBuffer(cu, receptorBornForces, numReceptorAtoms * numParticleGroups, "isolatedGbsaReceptorBornForces");

        // Fixed-point accumulators for tiled HCT kernel
        hctReceptorFixed.initialize<unsigned long long>(cu, totalParticles, "hctReceptorFixed");
        ligToRecHCTFixed.initialize<unsigned long long>(cu, numReceptorAtoms * numParticleGroups, "ligToRecHCTFixed");

        // Tiled force kernel buffers
        dEdR_crossTerm.initialize<unsigned long long>(cu, totalParticles, "dEdR_crossTerm");
        initRealBuffer(cu, bornForceLig, totalParticles, "bornForceLig");

        // Tile-skip cache for locality cutoff
        if (receptorLocalityCutoff > 0.0f) {
            int numRecBlocks = (numReceptorAtoms + 31) / 32;
            initRealBuffer(cu, hctRecBlockCache, totalParticles * numRecBlocks, "hctRecBlockCache");
            initRealBuffer(cu, ligToRecHCTCache, numReceptorAtoms * numParticleGroups, "ligToRecHCTCache");
            initRealBuffer(cu, crossTermBlockCache, numParticleGroups * numRecBlocks, "crossTermBlockCache");
            hasTileCache = false;
            hasCrossTermCache = false;
        }
    }

    // Allocate intermediate result buffers at context precision (real)
    if (totalParticles > 0) {
        initRealBuffer(cu, hctReceptor, totalParticles, "isolatedGbsaHctReceptor");
        initRealBuffer(cu, hctLigand, totalParticles, "isolatedGbsaHctLigand");
        initRealBuffer(cu, bornRadii, totalParticles, "isolatedGbsaBornRadii");
        initRealBuffer(cu, dE_dR, totalParticles, "isolatedGbsaDEdR");
        initRealBuffer(cu, atomEnergies, totalParticles, "isolatedGbsaAtomEnergies");
    }

    CUmodule module = cu.createModule(
        CudaGridForceKernelSources::commonHeaders +
        CudaGridForceKernelSources::gbsaGridForceKernel +
        CudaGridForceKernelSources::gbsaGridGenerationKernel +
        CudaGridForceKernelSources::isolatedGBSAKernel);
    computeLigandHCTKernel = cu.getKernel(module, "computeIsolatedLigandHCT");
    generateCrossTermGridKernel = cu.getKernel(module, "generateCrossTermGrid");
    computeCrossTermFromGridKernel = cu.getKernel(module, "computeCrossTermFromGrid");
    computeCrossTermPairwiseKernel = cu.getKernel(module, "computeCrossTermGBEnergy");
    accumulateCrossTermBornDerivativesKernel = cu.getKernel(module, "accumulateCrossTermBornDerivatives");
    computeBornRadiiHCTKernel = cu.getKernel(module, "computeBornRadiiHCT");
    computeBornRadiiOBCKernel = cu.getKernel(module, "computeBornRadiiOBC");
    computeGBEnergyKernel = cu.getKernel(module, "computeIsolatedGBEnergy");
    computeSAEnergyKernel = cu.getKernel(module, "computeIsolatedSAEnergy");
    computeReceptorDeltaSAKernel = cu.getKernel(module, "computeReceptorDeltaSA");
    accumulateBornRadiiDerivativesKernel = cu.getKernel(module, "accumulateIsolatedBornRadiiDerivatives");
    accumulateSADerivativesKernel = cu.getKernel(module, "accumulateIsolatedSADerivatives");
    computeHCTChainRuleForcesKernel = cu.getKernel(module, "computeIsolatedHCTChainRuleForces");

    if (receptorMode == IsolatedGBSAForce::GRID) {
        computeReceptorHCTGridKernel = cu.getKernel(module, "computeIsolatedReceptorHCTGrid");
        computeReceptorHCTGradientForceKernel = cu.getKernel(module, "computeIsolatedReceptorHCTGradientForce");
    } else if (receptorMode == IsolatedGBSAForce::PAIRWISE) {
        computeReceptorHCTPairwiseKernel = cu.getKernel(module, "computeIsolatedReceptorHCTPairwise");
        computeReceptorHCTPairwiseTiledKernel = cu.getKernel(module, "computeIsolatedReceptorHCTPairwiseTiled");
        computeReceptorLigandHCTTiledKernel = cu.getKernel(module, "computeReceptorLigandHCTTiled");
        convertTiledHCTToFloatKernel = cu.getKernel(module, "convertTiledHCTToFloat");
        addDistantHCTFromCacheKernel = cu.getKernel(module, "addDistantHCTFromCache");
        restoreDistantLigToRecHCTKernel = cu.getKernel(module, "restoreDistantLigToRecHCT");
        addDistantCrossTermFromCacheKernel = cu.getKernel(module, "addDistantCrossTermFromCache");
        computeReceptorHCTPairwiseChainRuleKernel = cu.getKernel(module, "computeIsolatedReceptorHCTPairwiseChainRule");

        // PAIRWISE mode: receptor desolvation kernels
        computeReceptorSelfHCTTiledKernel = cu.getKernel(module, "computeReceptorSelfHCTTiled");
        convertHCTToFloatKernel = cu.getKernel(module, "convertHCTToFloat");
        computeReceptorBornRadiiReferenceKernel = cu.getKernel(module, "computeReceptorBornRadiiReference");
        computeReceptorGBEnergyTiledKernel = cu.getKernel(module, "computeReceptorGBEnergyTiled");
        computeReceptorGBEnergyAndDeDRTiledKernel = cu.getKernel(module, "computeReceptorGBEnergyAndDeDRTiled");
        computeReceptorGBEnergyAndDeDRSimpleKernel = cu.getKernel(module, "computeReceptorGBEnergyAndDeDRSimple");
        computeReceptorBornRadiiWithLigandKernel = cu.getKernel(module, "computeReceptorBornRadiiWithLigand");
        precomputeReceptorBornForcesKernel = cu.getKernel(module, "precomputeReceptorBornForces");
        accumulateCrossTermReceptorDeDRKernel = cu.getKernel(module, "accumulateCrossTermReceptorDeDR");
        computePairwiseGBForceTiledKernel = cu.getKernel(module, "computePairwiseGBForceTiled");
        reduceLigandBornForceKernel = cu.getKernel(module, "reduceLigandBornForce");
        computePairwiseChainRuleTiledKernel = cu.getKernel(module, "computePairwiseChainRuleTiled");

        // GPU-side accumulation kernels (eliminate host-device sync)
        accumulateDesolvationOnGPUKernel = cu.getKernel(module, "accumulateDesolvationOnGPU");
        accumulateCrossTermOnGPUKernel = cu.getKernel(module, "accumulateCrossTermOnGPU");

        // Allocate fixed-point buffer for tiled HCT computation
        receptorSelfHCTFixed.initialize<unsigned long long>(cu, numReceptorAtoms, "receptorSelfHCTFixed");

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

        // Step 3: Compute receptor reference energy using the FUSED tiled
        // kernel (same one used at runtime). Using the same accumulation
        // order here eliminates the float32 pair-sum discrepancy between
        // setup and runtime (~11 kJ/mol over ~87M pair terms). The dE/dR
        // output goes to the receptorDeDR scratch buffer (group 0 slice),
        // which is overwritten at runtime and so is safe to clobber.
        if (cu.getUseDoublePrecision()) {
            vector<double> zeroEnergy(1, 0.0);
            receptorReferenceEnergy.upload(zeroEnergy);
        } else {
            vector<float> zeroEnergy(1, 0.0f);
            receptorReferenceEnergy.upload(zeroEnergy);
        }
        CUdeviceptr scratchDeDRPtr = receptorDeDR.getDevicePointer();
        cu.clearBuffer(receptorDeDR);

        float prefactorFloatInit = (float)prefactor;
        void* prefactorArgInit = (cu.getUseDoublePrecision() ? (void*)&prefactor : (void*)&prefactorFloatInit);
        void* refEnergyTiledArgs[] = {
            &receptorPosPtr, &receptorChargesPtr, &receptorBornRadiiRefPtr,
            &numReceptorAtoms, prefactorArgInit, &receptorRefEnergyPtr,
            &scratchDeDRPtr, &numTiles
        };
        cu.executeKernel(computeReceptorGBEnergyAndDeDRTiledKernel,
                         refEnergyTiledArgs, recNumBlocks * recBlockSize, recBlockSize);

        // Download and cache reference energy
        if (cu.getUseDoublePrecision()) {
            vector<double> refEnergy(1);
            receptorReferenceEnergy.download(refEnergy);
            receptorReferenceEnergyValue = refEnergy[0];
        } else {
            vector<float> refEnergy(1);
            receptorReferenceEnergy.download(refEnergy);
            receptorReferenceEnergyValue = refEnergy[0];
        }
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

    // Precision-aware prefactor: kernels take `real prefactor`, so pass an
    // 8-byte double in double mode and a 4-byte float in single/mixed mode.
    float prefactorFloat = (float)prefactor;
    void* prefactorArg = (cu.getUseDoublePrecision() ? (void*)&prefactor : (void*)&prefactorFloat);

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


    // Step 0: (tile-skipping replaces active mask — no per-atom mask needed)
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
            &numBins, &totalParticles, &numAtoms, &interpolationMethod,
            &useKDECorrections, &hasBinnedKDEDerivatives, &hctReceptorPtr
        };
        cu.executeKernel(computeReceptorHCTGridKernel, receptorArgs, numBlocks * blockSize, blockSize);

    } else if (receptorMode == IsolatedGBSAForce::PAIRWISE) {
        CUdeviceptr receptorPosPtr = receptorPositions.getDevicePointer();
        CUdeviceptr receptorRadiiPtr = receptorRadii.getDevicePointer();
        CUdeviceptr receptorScalesPtr = receptorScaleFactors.getDevicePointer();

        CUdeviceptr hctRecFixedPtr = hctReceptorFixed.getDevicePointer();
        CUdeviceptr ligToRecFixedPtr = ligToRecHCTFixed.getDevicePointer();
        CUdeviceptr recBlockBoundsPtr = recBlockBounds.getDevicePointer();

        // Clear fixed-point accumulators
        cu.clearBuffer(hctReceptorFixed);
        cu.clearBuffer(ligToRecHCTFixed);

        // Tile dimensions
        int ligGroupSize = totalParticles / numParticleGroups;
        int numLigBlocks = (ligGroupSize + 31) / 32;
        int numRecBlocks = (numReceptorAtoms + 31) / 32;
        int tilesPerGroup = numRecBlocks * numLigBlocks;
        int totalTiles = tilesPerGroup * numParticleGroups;
        int tiledBlockSize = 256;
        int tiledBlocks = (totalTiles * 32 + tiledBlockSize - 1) / tiledBlockSize;

        bool useLocalityCache = (receptorLocalityCutoff > 0.0f && hctRecBlockCache.isInitialized());

        // Determine tile-skip and cache mode
        float localityCutoffVal = -1.0f;  // no skip by default
        CUdeviceptr hctCachePtr = (CUdeviceptr)0;

        if (useLocalityCache && !hasTileCache) {
            // First call: full computation, write cache
            localityCutoffVal = -1.0f;  // no tile-skip
            hctCachePtr = hctRecBlockCache.getDevicePointer();  // write cache
        } else if (useLocalityCache && hasTileCache) {
            // Subsequent calls: tile-skip + cached reconstruction
            localityCutoffVal = receptorLocalityCutoff;  // enable tile-skip
            hctCachePtr = (CUdeviceptr)0;  // don't overwrite cache
        }

        CUdeviceptr groupScalingFactorsPtr2 = groupScalingFactorsBuffer.getDevicePointer();
        void* tiledArgs[] = {
            &posqPtr, &particleIndicesPtr, &radiiPtr, &scaleFactorsPtr,
            &receptorPosPtr, &receptorRadiiPtr, &receptorScalesPtr,
            &numReceptorAtoms, &groupStartPtr, &numParticleGroups,
            &numAtoms, &cutoffDistance,
            &hctRecFixedPtr, &ligToRecFixedPtr, &tilesPerGroup,
            &recBlockBoundsPtr, &localityCutoffVal,
            &hctCachePtr, &numRecBlocks,
            &globalScalingFactor, &groupScalingFactorsPtr2
        };
        cu.executeKernel(computeReceptorLigandHCTTiledKernel, tiledArgs, tiledBlocks * tiledBlockSize, tiledBlockSize);

        // For tile-skip mode: add cached HCT for distant blocks (receptor→ligand direction)
        if (useLocalityCache && hasTileCache) {
            CUdeviceptr cachePtr = hctRecBlockCache.getDevicePointer();
            float locCut = receptorLocalityCutoff;
            void* distantArgs[] = {
                &cachePtr, &posqPtr, &particleIndicesPtr,
                &recBlockBoundsPtr, &locCut, &numRecBlocks,
                &totalParticles, &hctRecFixedPtr
            };
            int distBlocks = (totalParticles + blockSize - 1) / blockSize;
            cu.executeKernel(addDistantHCTFromCacheKernel, distantArgs, distBlocks * blockSize, blockSize);
        }

        // Convert fixed-point to float
        int convBlocks = (totalParticles + blockSize - 1) / blockSize;
        void* convArgs1[] = { &hctRecFixedPtr, &hctReceptorPtr, &totalParticles };
        cu.executeKernel(convertTiledHCTToFloatKernel, convArgs1, convBlocks * blockSize, blockSize);

        int ligRecTotal = numReceptorAtoms * numParticleGroups;
        CUdeviceptr ligandToReceptorHCTPtr2 = ligandToReceptorHCT.getDevicePointer();
        int convBlocks2 = (ligRecTotal + blockSize - 1) / blockSize;
        void* convArgs2[] = { &ligToRecFixedPtr, &ligandToReceptorHCTPtr2, &ligRecTotal };
        cu.executeKernel(convertTiledHCTToFloatKernel, convArgs2, convBlocks2 * blockSize, blockSize);

        // For tile-skip mode: restore cached lig→rec HCT for distant receptor atoms
        if (useLocalityCache && hasTileCache) {
            CUdeviceptr ligRecCachePtr = ligToRecHCTCache.getDevicePointer();
            float locCut = receptorLocalityCutoff;
            void* restoreArgs[] = {
                &ligRecCachePtr, &ligandToReceptorHCTPtr2,
                &posqPtr, &particleIndicesPtr, &recBlockBoundsPtr, &locCut,
                &groupStartPtr, &numParticleGroups, &numReceptorAtoms, &totalParticles
            };
            int restoreBlocks = (ligRecTotal + blockSize - 1) / blockSize;
            cu.executeKernel(restoreDistantLigToRecHCTKernel, restoreArgs, restoreBlocks * blockSize, blockSize);
        }

        // First call: save cache for lig→rec direction
        if (useLocalityCache && !hasTileCache) {
            // Device-to-device copy of ligandToReceptorHCT → ligToRecHCTCache
            CUdeviceptr srcPtr = ligandToReceptorHCT.getDevicePointer();
            CUdeviceptr dstPtr = ligToRecHCTCache.getDevicePointer();
            cuMemcpyDtoD(dstPtr, srcPtr, ligRecTotal * (size_t)realElementSize(cu));
            hasTileCache = true;
        }

        fusedHCTComputed_ = true;
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
        &groupStartPtr, &numParticleGroups, &numAtoms, prefactorArg,
        &forcePtr, &groupEnergiesPtr, &groupLigandEnergiesPtr, &paddedNumAtoms,
        &globalScalingFactor, &groupScalingFactorsPtr, &groupUnscaledEnergiesPtr
    };
    cu.executeKernel(computeGBEnergyKernel, energyArgs, numBlocks * blockSize, blockSize);

    // Step 4a: GRID mode augment — direct pairwise cross-term.
    // Uses the existing computeCrossTermGBEnergy kernel with ligand Born
    // radii from the HCT grid (just computed) and baseline receptor Born
    // radii (frozen, uploaded at init). This is the receptor-ligand GB
    // cross term that computeGBEnergyKernel (ligand-only) omits.
    if (receptorMode == IsolatedGBSAForce::GRID && computeCrossTermGrid) {
        CUdeviceptr receptorPosPtr = receptorPositions.getDevicePointer();
        CUdeviceptr receptorChargesPtr = receptorCharges.getDevicePointer();
        CUdeviceptr receptorBornRadiiPtr = receptorBornRadii.getDevicePointer();
        CUdeviceptr groupCrossPtr = groupCrossTermEnergies.getDevicePointer();
        CUdeviceptr isActivePtr = (CUdeviceptr)0;  // unused
        void* crossArgs[] = {
            &posqPtr, &particleIndicesPtr, &chargesPtr,
            &bornRadiiPtr,
            &receptorPosPtr, &receptorChargesPtr, &receptorBornRadiiPtr,
            &groupStartPtr, &numParticleGroups,
            &numReceptorAtoms, &numAtoms,
            prefactorArg,
            &groupCrossPtr,
            &forcePtr, &paddedNumAtoms,
            &globalScalingFactor, &groupScalingFactorsPtr,
            &isActivePtr,
        };
        cu.executeKernel(computeCrossTermPairwiseKernel, crossArgs,
                         numBlocks * blockSize, blockSize);
        // Also add cross-term to total group energies
        // (the kernel writes to crossTermEnergies only; we need it in
        // groupEnergies too for the OpenMM total energy return path)
    }

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

        CUdeviceptr isActiveRecAtomPtr = (CUdeviceptr)0;  // no longer used for active mask


        // 4b.1: Ligand→receptor HCT already computed by tiled kernel in Step 1
        // (fusedHCTComputed_ is always true now)

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


        // 4b.3: Fused receptor energy + dE/dR per group (single tiled O(N²) pass)
        cu.clearBuffer(receptorEnergy);
        if (includeForces) {
            cu.clearBuffer(receptorDeDR);  // clear ALL groups' dE/dR at once (async)
        }

        double refEnergyDouble = receptorReferenceEnergyValue;
        float refEnergyFloat = receptorReferenceEnergyValue;
        void* refEnergyArg = (cu.getUseDoublePrecision() ? (void*)&refEnergyDouble : (void*)&refEnergyFloat);
        for (int g = 0; g < numParticleGroups; g++) {
            CUdeviceptr groupBornRadiiPtr = receptorBornRadiiPtr + (size_t)g * numReceptorAtoms * realElementSize(cu);
            CUdeviceptr groupDeDRPtr = receptorDeDRPtr + (size_t)g * numReceptorAtoms * realElementSize(cu);
            cu.clearBuffer(receptorEnergy);

            if (includeForces) {
                // Fused receptor energy + dE/dR for this group.
                // Tiled O(N²/2) kernel — uses rotation pattern in both diagonal
                // and off-diagonal loops to guarantee 32 distinct shared-memory
                // lanes per step (no SIMT race).
                void* tiledArgs[] = {
                    &receptorPosPtr, &receptorChargesPtr, &groupBornRadiiPtr,
                    &numReceptorAtoms, prefactorArg, &receptorEnergyPtr, &groupDeDRPtr,
                    &numTiles
                };
                cu.executeKernel(computeReceptorGBEnergyAndDeDRTiledKernel,
                                 tiledArgs, recNumBlocksTiled * recBlockSize, recBlockSize);
            } else {
                // Energy only
                void* recEnergyArgs[] = {
                    &receptorPosPtr, &receptorChargesPtr, &groupBornRadiiPtr,
                    &numReceptorAtoms, prefactorArg, &receptorEnergyPtr, &numTiles
                };
                cu.executeKernel(computeReceptorGBEnergyTiledKernel, recEnergyArgs, recNumBlocksTiled * recBlockSize, recBlockSize);
            }

            void* accumArgs[] = {
                &receptorEnergyPtr, refEnergyArg, &g,
                &globalScalingFactor, &groupScalingFactorsPtr,
                &groupEnergiesPtr, &groupDesolvPtr, &groupUnscaledEnergiesPtr
            };
            cu.executeKernel(accumulateDesolvationOnGPUKernel, accumArgs, 1, 1);
        }

        // 4b.3a+: Accumulate cross-term contribution to receptorDeDR.
        // computeReceptorGBEnergyAndDeDRTiled only populates receptor self +
        // intra-receptor terms; the cross-term's dE/dR_born_rec (symmetric
        // with the dE/dR_born_lig computed in pass 1) also needs to be added
        // so bornForcesRec captures all physical paths.
        if (includeForces) {
            CUdeviceptr receptorPosPtrC = receptorPositions.getDevicePointer();
            CUdeviceptr ligandChargesPtrC = charges.getDevicePointer();
            CUdeviceptr ligandBornRadiiPtrC = bornRadii.getDevicePointer();
            CUdeviceptr receptorChargesPtrC = receptorCharges.getDevicePointer();
            int crossBlocks = (numParticleGroups * numReceptorAtoms
                               + blockSize - 1) / blockSize;
            void* crossDedrArgs[] = {
                &posqPtr, &particleIndicesPtr, &receptorPosPtrC,
                &ligandChargesPtrC, &ligandBornRadiiPtrC,
                &receptorChargesPtrC, &receptorBornRadiiPtr,
                &groupStartPtr, &numParticleGroups, &numReceptorAtoms,
                &numAtoms, prefactorArg, &receptorDeDRPtr
            };
            cu.executeKernel(accumulateCrossTermReceptorDeDRKernel,
                             crossDedrArgs, crossBlocks * blockSize, blockSize);
        }

        // 4b.3b: Precompute bornForces per receptor atom
        if (includeForces) {
            CUdeviceptr bornForcesRecPtr = receptorBornForces.getDevicePointer();
            int bfBlocks = (numParticleGroups * numReceptorAtoms + blockSize - 1) / blockSize;
            void* bfArgs[] = {
                &receptorRadiiPtr, &receptorSelfHCTPtr, &ligandToReceptorHCTPtr,
                &receptorBornRadiiPtr, &receptorDeDRPtr,
                &numReceptorAtoms, &numParticleGroups, &bornForcesRecPtr,
                &globalScalingFactor, &groupScalingFactorsPtr
            };
            cu.executeKernel(precomputeReceptorBornForcesKernel, bfArgs, bfBlocks * blockSize, blockSize);
        }


        // === TILED FORCE PASS 1: cross-term energy + direct forces + dE/dR_lig + desolv chain rule ===
        CUdeviceptr receptorChargesPtr2 = receptorCharges.getDevicePointer();
        CUdeviceptr bornForcesRecPtr2 = receptorBornForces.getDevicePointer();
        CUdeviceptr dEdRCrossTermPtr = dEdR_crossTerm.getDevicePointer();
        CUdeviceptr bornForceLigPtr = bornForceLig.getDevicePointer();

        cu.clearBuffer(dEdR_crossTerm);
        cu.clearBuffer(groupCrossTermEnergies);

        int numRecBlocks2 = (numReceptorAtoms + 31) / 32;
        int totalForceTiles = numParticleGroups * numRecBlocks2;
        int forceThreads = totalForceTiles * 32;
        int forceBlocks2 = (forceThreads + blockSize - 1) / blockSize;

        CUdeviceptr recBlockBoundsPtr2 = recBlockBounds.getDevicePointer();
        bool useForceTileSkip = (receptorLocalityCutoff > 0.0f && crossTermBlockCache.isInitialized());

        // Force tile-skip: first call = no skip + cache write, subsequent = skip + reconstruct
        float forceTileSkipCutoff = -1.0f;
        CUdeviceptr crossCachePtr = (CUdeviceptr)0;

        if (useForceTileSkip && !hasCrossTermCache) {
            forceTileSkipCutoff = -1.0f;  // no skip, compute all
            crossCachePtr = crossTermBlockCache.getDevicePointer();  // write cache
        } else if (useForceTileSkip && hasCrossTermCache) {
            forceTileSkipCutoff = receptorLocalityCutoff;  // enable tile-skip
            crossCachePtr = (CUdeviceptr)0;  // don't overwrite cache
        }

        void* tiledForceArgs[] = {
            &posqPtr, &particleIndicesPtr, &radiiPtr, &scaleFactorsPtr, &chargesPtr,
            &bornRadiiPtr,
            &receptorPosPtr, &receptorRadiiPtr, &receptorScalesPtr, &receptorChargesPtr2,
            &receptorBornRadiiPtr, &bornForcesRecPtr2,
            &groupStartPtr, &numParticleGroups, &numReceptorAtoms, &numAtoms,
            prefactorArg, &cutoffDistance, &forcePtr, &paddedNumAtoms,
            &groupCrossTermPtr, &dEdRCrossTermPtr,
            &globalScalingFactor, &groupScalingFactorsPtr, &numRecBlocks2,
            &recBlockBoundsPtr2, &forceTileSkipCutoff, &crossCachePtr
        };
        cu.executeKernel(computePairwiseGBForceTiledKernel, tiledForceArgs, forceBlocks2 * blockSize, blockSize);

        // For tile-skip mode: add cached cross-term energy for distant blocks
        if (useForceTileSkip && hasCrossTermCache) {
            float locCut = receptorLocalityCutoff;
            void* distCrossArgs[] = {
                &crossCachePtr, &posqPtr, &particleIndicesPtr,
                &recBlockBoundsPtr2, &locCut, &groupStartPtr,
                &numParticleGroups, &numRecBlocks2,
                &globalScalingFactor, &groupScalingFactorsPtr, &groupCrossTermPtr
            };
            // Need to pass the actual cache pointer for reading
            CUdeviceptr crossCacheReadPtr = crossTermBlockCache.getDevicePointer();
            distCrossArgs[0] = &crossCacheReadPtr;
            int totalCrossTiles = numParticleGroups * numRecBlocks2;
            int crossBlocks = (totalCrossTiles + blockSize - 1) / blockSize;
            cu.executeKernel(addDistantCrossTermFromCacheKernel, distCrossArgs, crossBlocks * blockSize, blockSize);
        }

        // First call: mark cross-term cache as built
        if (useForceTileSkip && !hasCrossTermCache) {
            hasCrossTermCache = true;
        }

        // Add cross-term to group energies
        void* crossAccumArgs[] = {
            &groupCrossTermPtr, &numParticleGroups,
            &globalScalingFactor, &groupScalingFactorsPtr,
            &groupEnergiesPtr, &groupUnscaledEnergiesPtr
        };
        cu.executeKernel(accumulateCrossTermOnGPUKernel, crossAccumArgs, 1, 1);


        // Accumulate self-GB + intra-ligand dE/dR_born_lig into dE_dR before
        // reducing so that the receptor→ligand chain rule (pass 2) sees the
        // TOTAL dE/dR_lig rather than just the cross-term portion.
        CUdeviceptr dE_dRPtr_for_reduce = dE_dR.getDevicePointer();
        void* bornDerivArgsEarly[] = {
            &posqPtr, &particleIndicesPtr, &chargesPtr, &bornRadiiPtr,
            &groupStartPtr, &numParticleGroups, &numAtoms, prefactorArg,
            &dE_dRPtr_for_reduce,
            &globalScalingFactor, &groupScalingFactorsPtr
        };
        cu.executeKernel(accumulateBornRadiiDerivativesKernel,
                         bornDerivArgsEarly, numBlocks * blockSize, blockSize);

        if (includeSurfaceArea) {
            float probe = (receptorMode == IsolatedGBSAForce::GRID) ? probeRadius : 0.14f;
            void* saDerivArgsEarly[] = {
                &radiiPtr, &bornRadiiPtr, &groupStartPtr,
                &numParticleGroups, &numAtoms, &surfaceTension, &probe,
                &dE_dRPtr_for_reduce,
                &globalScalingFactor, &groupScalingFactorsPtr
            };
            cu.executeKernel(accumulateSADerivativesKernel,
                             saDerivArgsEarly, numBlocks * blockSize, blockSize);
        }

        // === REDUCE: (cross-term dE/dR + self-GB dE/dR) → bornForceLig ===
        int reduceBlocks = (totalParticles + blockSize - 1) / blockSize;
        void* reduceArgs[] = {
            &dEdRCrossTermPtr, &dE_dRPtr_for_reduce,
            &bornRadiiPtr, &hctReceptorPtr, &hctLigandPtr,
            &radiiPtr, &groupStartPtr, &numParticleGroups, &totalParticles, &numAtoms,
            &globalScalingFactor, &groupScalingFactorsPtr, &bornForceLigPtr
        };
        cu.executeKernel(reduceLigandBornForceKernel, reduceArgs, reduceBlocks * blockSize, blockSize);


        // === TILED FORCE PASS 2: cross-term HCT chain rule ===
        // Pass 2 has no energy — tile-skip freely when locality cutoff is set
        float chainTileSkipCutoff = (receptorLocalityCutoff > 0.0f) ? receptorLocalityCutoff : -1.0f;
        CUdeviceptr groupScalingFactorsPtr3 = groupScalingFactorsBuffer.getDevicePointer();
        void* tiledChainArgs[] = {
            &posqPtr, &particleIndicesPtr, &radiiPtr, &scaleFactorsPtr,
            &receptorPosPtr, &receptorRadiiPtr, &receptorScalesPtr,
            &bornForceLigPtr,
            &groupStartPtr, &numParticleGroups, &numReceptorAtoms, &numAtoms,
            &cutoffDistance, &forcePtr, &paddedNumAtoms, &numRecBlocks2,
            &recBlockBoundsPtr2, &chainTileSkipCutoff,
            &globalScalingFactor, &groupScalingFactorsPtr3
        };
        cu.executeKernel(computePairwiseChainRuleTiledKernel, tiledChainArgs, forceBlocks2 * blockSize, blockSize);
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

        // Step 5b: Receptor ΔSA (PAIRWISE only).
        // When the ligand descreens receptor atoms, their Born radii shrink,
        // which changes the receptor's self-SA energy. Stock OpenMM
        // E_GBSA(R+L) − E_GBSA(R) includes this term automatically; our
        // computeIsolatedSAEnergy loops only over ligand atoms and misses
        // it. Here we add the per-receptor-atom SA delta using
        // (R_born_withL vs R_born_alone), summed into group desolvation.
        if (receptorMode == IsolatedGBSAForce::PAIRWISE && numReceptorAtoms > 0) {
            CUdeviceptr receptorRadiiPtr = receptorRadii.getDevicePointer();
            CUdeviceptr receptorBornRadiiPtr = receptorBornRadii.getDevicePointer();
            CUdeviceptr receptorBornRadiiRefPtr = receptorBornRadiiRef.getDevicePointer();
            CUdeviceptr groupDesolvPtr = groupReceptorDesolvations.getDevicePointer();
            float rProbe = 0.14f;
            void* rSaArgs[] = {
                &receptorRadiiPtr, &receptorBornRadiiPtr, &receptorBornRadiiRefPtr,
                &numReceptorAtoms, &numParticleGroups,
                &surfaceTension, &rProbe,
                &globalScalingFactor, &groupScalingFactorsPtr,
                &groupDesolvPtr, &groupEnergiesPtr, &groupUnscaledEnergiesPtr
            };
            int totalWork = numParticleGroups * numReceptorAtoms;
            int saBlocks = (totalWork + blockSize - 1) / blockSize;
            cu.executeKernel(computeReceptorDeltaSAKernel, rSaArgs,
                             saBlocks * blockSize, blockSize);
        }
    }


    // Step 6: Chain rule forces through Born radii
    if (includeForces) {
        CUdeviceptr dE_dRPtr = dE_dR.getDevicePointer();

        // Populate dE_dR with self-GB + intra-ligand + SA derivatives.
        // For PAIRWISE this was already done in Step 4b before
        // reduceLigandBornForce; skip here to avoid redundant work.
        // For NONE and GRID modes, nothing else populates dE_dR, so
        // computeHCTChainRuleForces below would use zero/stale data and
        // drop the ligand-OBC chain-rule force contribution entirely.
        if (receptorMode != IsolatedGBSAForce::PAIRWISE) {
            void* bornDerivArgs[] = {
                &posqPtr, &particleIndicesPtr, &chargesPtr, &bornRadiiPtr,
                &groupStartPtr, &numParticleGroups, &numAtoms, prefactorArg,
                &dE_dRPtr,
                &globalScalingFactor, &groupScalingFactorsPtr
            };
            cu.executeKernel(accumulateBornRadiiDerivativesKernel,
                             bornDerivArgs, numBlocks * blockSize, blockSize);
            if (includeSurfaceArea) {
                float probe = (receptorMode == IsolatedGBSAForce::GRID)
                              ? probeRadius : 0.14f;
                void* saDerivArgs[] = {
                    &radiiPtr, &bornRadiiPtr, &groupStartPtr,
                    &numParticleGroups, &numAtoms, &surfaceTension, &probe,
                    &dE_dRPtr,
                    &globalScalingFactor, &groupScalingFactorsPtr
                };
                cu.executeKernel(accumulateSADerivativesKernel,
                                 saDerivArgs, numBlocks * blockSize, blockSize);
            }
        }
        // Cross-term contribution to dE_dR (adds, doesn't overwrite).
        bool addCrossTermToDEdR =
            (receptorMode == IsolatedGBSAForce::PAIRWISE) ||
            (receptorMode == IsolatedGBSAForce::GRID && computeCrossTermGrid);
        if (addCrossTermToDEdR) {
            CUdeviceptr receptorPosPtr2 = receptorPositions.getDevicePointer();
            CUdeviceptr receptorChargesPtr2 = receptorCharges.getDevicePointer();
            CUdeviceptr receptorBornRadiiPtr2 = receptorBornRadii.getDevicePointer();
            void* crossDerivArgs[] = {
                &posqPtr, &particleIndicesPtr, &chargesPtr,
                &bornRadiiPtr,
                &receptorPosPtr2, &receptorChargesPtr2, &receptorBornRadiiPtr2,
                &groupStartPtr, &numParticleGroups,
                &numReceptorAtoms, &numAtoms,
                prefactorArg,
                &dE_dRPtr,
                &globalScalingFactor, &groupScalingFactorsPtr,
            };
            cu.executeKernel(accumulateCrossTermBornDerivativesKernel,
                             crossDerivArgs, numBlocks * blockSize, blockSize);
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
                &useKDECorrections, &hasBinnedKDEDerivatives,
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

            // Desolvation + cross-term forces handled by tiled kernels in Step 4b
        }
    }

    // Download group energies only when energy is needed to avoid sync barriers
    if (includeEnergy && !skipGroupEnergyDownload_) {
        downloadMixedEnergy(cu, groupEnergies, groupEnergiesHost);
        downloadMixedEnergy(cu, groupLigandSelfEnergies, groupLigandSelfEnergiesHost);
        downloadMixedEnergy(cu, groupUnscaledEnergies, groupUnscaledEnergiesHost);

        if (receptorMode == IsolatedGBSAForce::PAIRWISE) {
            downloadMixedEnergy(cu, groupReceptorDesolvations, groupReceptorDesolvationsHost);
            downloadMixedEnergy(cu, groupCrossTermEnergies, groupCrossTermEnergiesHost);
        } else if (receptorMode == IsolatedGBSAForce::GRID
                   && computeCrossTermGrid) {
            downloadMixedEnergy(cu, groupCrossTermEnergies, groupCrossTermEnergiesHost);
        }

        // Sum total energy. In GRID+crossTerm mode the cross-term
        // kernel writes to its own buffer (not groupEnergies), so add
        // it here.
        double totalEnergy = 0.0;
        for (int g = 0; g < numParticleGroups; g++) {
            totalEnergy += groupEnergiesHost[g];
            if (receptorMode == IsolatedGBSAForce::GRID
                && computeCrossTermGrid) {
                totalEnergy += groupCrossTermEnergiesHost[g];
            }
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
    prefactor = -ONE_4PI_EPS0 * (1.0/soluteDielectric - 1.0/solventDielectric);

    includeSurfaceArea = force.getIncludeSurfaceArea();
    surfaceTension = static_cast<float>(force.getSurfaceTension());
    cutoffDistance = static_cast<float>(force.getCutoffDistance());
    receptorLocalityCutoff = static_cast<float>(force.getReceptorLocalityCutoff());

    // Update alchemical scaling factors
    globalScalingFactor = static_cast<float>(force.getGlobalScalingFactor());
    int nGroups = force.getNumParticleGroups();
    if (nGroups > 0) {
        groupScalingFactorsHostCopy.resize(numParticleGroups, 1.0f);
        for (int i = 0; i < nGroups; i++) {
            groupScalingFactorsHostCopy[i] = static_cast<float>(force.getGroupScalingFactor(i));
        }
        groupScalingFactorsBuffer.upload(groupScalingFactorsHostCopy);
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
    if (groupIndex < 0 || groupIndex >= numParticleGroups) {
        throw OpenMMException("IsolatedGBSAForce: invalid group index");
    }
    if (bornRadii.getSize() == 0) {
        throw OpenMMException(
            "IsolatedGBSAForce::getGroupBornRadii: Born radii buffer is "
            "empty. Call Context.getState(getEnergy=True) at least once "
            "before querying.");
    }

    // Force a fresh download from device on every call. The earlier caching
    // scheme was unreliable (and automatic download in getState caused
    // expensive host↔device syncs every sweep), so this explicit diagnostic
    // path always pays the sync cost but gives correct values.
    vector<int> groupStarts(numParticleGroups + 1);
    groupStartIndex.download(groupStarts);
    int startIdx = groupStarts[groupIndex];
    int endIdx = groupStarts[groupIndex + 1];

    vector<float> allBornRadii(bornRadii.getSize());
    bornRadii.download(allBornRadii);

    vector<double> result(allBornRadii.begin() + startIdx,
                          allBornRadii.begin() + endIdx);
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
    cu.setAsCurrent();

    int totalParticles = numParticleGroups * numAtoms;
    int templateN = numAtoms;
    int dim3N = 3 * totalParticles;
    if (totalParticles == 0) {
        return std::vector<double>();
    }

    // Lazy: allocate Hessian buffers (double precision) and load the
    // double-storage kernels on first call. The legacy float kernels
    // are loaded too so a future setHessianFloatStorage(true) debug
    // toggle can fall back to them without re-creating the module.
    if (!hessianBuffersInitialized) {
        hessianHctReceptorDouble.initialize<double>(cu, totalParticles, "isolatedGbsaHessHctRecD");
        hessianHctLigandDouble.initialize<double>(cu, totalParticles, "isolatedGbsaHessHctLigD");
        hessianBornRadiiDouble.initialize<double>(cu, totalParticles, "isolatedGbsaHessBornD");
        hessianDRdPsi.initialize<double>(cu, totalParticles, "isolatedGbsaHessDRdPsi");
        hessianD2RdPsi2.initialize<double>(cu, totalParticles, "isolatedGbsaHessD2RdPsi2");
        hessianDEdHCT.initialize<double>(cu, totalParticles, "isolatedGbsaHessDEdHCT");
        hessianJacobian.initialize<double>(cu, totalParticles * dim3N, "isolatedGbsaHessJacobian");
        hessianRecD2Psi.initialize<double>(cu, totalParticles * 6, "isolatedGbsaHessRecD2Psi");
        hessianCouplingMatrix.initialize<double>(cu, totalParticles * totalParticles, "isolatedGbsaHessCoupling");
        hessianMatrix.initialize<double>(cu, dim3N * dim3N, "isolatedGbsaHessFull");

        std::vector<int> dummyStart(templateN + 1, 0);
        hessianDummyExclStart.initialize<int>(cu, templateN + 1, "isolatedGbsaHessDummyExclStart");
        hessianDummyExclStart.upload(dummyStart);
        hessianDummyExclAtoms.initialize<int>(cu, 1, "isolatedGbsaHessDummyExclAtoms");
        std::vector<int> dummyAtoms(1, 0);
        hessianDummyExclAtoms.upload(dummyAtoms);

        CUmodule module = cu.createModule(
            CudaGridForceKernelSources::commonHeaders +
            CudaGridForceKernelSources::gbsaGridForceKernel +
            CudaGridForceKernelSources::isolatedGBSAKernel +
            CudaGridForceKernelSources::gbsaHessianDoubleKernel);
        prepareHessianIntermediatesKernel        = cu.getKernel(module, "prepareHessianIntermediates");
        computeHCTJacobianPairwiseKernel         = cu.getKernel(module, "computeHCTJacobianPairwise");
        computeReceptorPairwiseHessianKernel     = cu.getKernel(module, "computeReceptorPairwiseHessian");
        computeBornCouplingMatrixKernel          = cu.getKernel(module, "computeBornCouplingMatrix");
        assembleGBSAHessianKernel                = cu.getKernel(module, "assembleGBSAHessian");
        prepareHessianIntermediatesDoubleKernel  = cu.getKernel(module, "prepareHessianIntermediatesDouble");
        computeHCTJacobianPairwiseDoubleKernel   = cu.getKernel(module, "computeHCTJacobianPairwiseDouble");
        computeReceptorPairwiseHessianDoubleKernel = cu.getKernel(module, "computeReceptorPairwiseHessianDouble");
        computeBornCouplingMatrixDoubleKernel    = cu.getKernel(module, "computeBornCouplingMatrixDouble");
        assembleGBSAHessianDoubleKernel          = cu.getKernel(module, "assembleGBSAHessianDouble");
        convertTiledHCTToDoubleKernel            = cu.getKernel(module, "convertTiledHCTToDouble");
        computeBornRadiiOBCDoubleKernel          = cu.getKernel(module, "computeBornRadiiOBCDouble");
        computeHctReceptorPairwiseDoubleKernel   = cu.getKernel(module, "computeHctReceptorPairwiseDouble");
        computeHctLigandPairwiseDoubleKernel     = cu.getKernel(module, "computeHctLigandPairwiseDouble");

        hessianBuffersInitialized = true;
        hessianNumAtomsCached = totalParticles;
    }

    // Sanity: bornRadii / dE_dR / hctReceptor / hctLigand must already be
    // populated by a prior execute() / getState(getForces=True). We don't
    // recompute them here.
    if (!bornRadii.isInitialized() || !dE_dR.isInitialized() ||
        !hctReceptor.isInitialized() || !hctLigand.isInitialized()) {
        throw OpenMMException(
            "IsolatedGBSAForce: computeHessian() requires a prior "
            "getState(getForces=True) to populate Born radii and dE/dR.");
    }

    int blockSize = 256;
    int numBlocksN = (totalParticles + blockSize - 1) / blockSize;

    CUdeviceptr posqPtr           = cu.getPosq().getDevicePointer();
    CUdeviceptr particleIndicesPtr = particleIndices.getDevicePointer();
    CUdeviceptr radiiPtr          = radii.getDevicePointer();
    CUdeviceptr scaleFactorsPtr   = scaleFactors.getDevicePointer();
    CUdeviceptr chargesPtr        = charges.getDevicePointer();
    CUdeviceptr bornRadiiPtr      = bornRadii.getDevicePointer();
    CUdeviceptr hctReceptorPtr    = hctReceptor.getDevicePointer();
    CUdeviceptr hctLigandPtr      = hctLigand.getDevicePointer();
    CUdeviceptr dE_dRPtr          = dE_dR.getDevicePointer();
    CUdeviceptr groupStartPtr     = groupStartIndex.getDevicePointer();
    CUdeviceptr dRdPsiPtr         = hessianDRdPsi.getDevicePointer();
    CUdeviceptr d2RdPsi2Ptr       = hessianD2RdPsi2.getDevicePointer();
    CUdeviceptr dE_dHCTPtr        = hessianDEdHCT.getDevicePointer();
    CUdeviceptr jacobianPtr       = hessianJacobian.getDevicePointer();
    CUdeviceptr recD2PsiPtr       = hessianRecD2Psi.getDevicePointer();
    CUdeviceptr couplingPtr       = hessianCouplingMatrix.getDevicePointer();
    CUdeviceptr hessianPtr        = hessianMatrix.getDevicePointer();
    CUdeviceptr exclStartPtr      = hessianDummyExclStart.getDevicePointer();
    CUdeviceptr exclAtomsPtr      = hessianDummyExclAtoms.getDevicePointer();
    CUdeviceptr receptorPosPtr    = (numReceptorAtoms > 0) ? receptorPositions.getDevicePointer() : 0;
    CUdeviceptr receptorRadiiPtr  = (numReceptorAtoms > 0) ? receptorRadii.getDevicePointer() : 0;
    CUdeviceptr receptorScalesPtr = (numReceptorAtoms > 0) ? receptorScaleFactors.getDevicePointer() : 0;
    CUdeviceptr hctReceptorDoublePtr = hessianHctReceptorDouble.getDevicePointer();
    CUdeviceptr hctLigandDoublePtr   = hessianHctLigandDouble.getDevicePointer();
    CUdeviceptr bornRadiiDoublePtr   = hessianBornRadiiDouble.getDevicePointer();

    // === Kernel 0a: receptor->ligand HCT in double (pair-loop, no tiling).
    // Slower than the production float-tiled kernel but only runs once per
    // Hessian eval and gives JAX-equivalent precision in the HCT sum. ===
    if (numReceptorAtoms > 0) {
        void* hctRecArgs[] = {
            &posqPtr, &particleIndicesPtr, &radiiPtr,
            &groupStartPtr, &numParticleGroups, &templateN,
            &receptorPosPtr, &receptorRadiiPtr, &receptorScalesPtr,
            &numReceptorAtoms, &totalParticles, &hctReceptorDoublePtr
        };
        cu.executeKernel(computeHctReceptorPairwiseDoubleKernel, hctRecArgs,
                         numBlocksN * blockSize, blockSize);
    } else {
        cu.clearBuffer(hessianHctReceptorDouble);
    }

    // === Kernel 0b: ligand-ligand HCT in double ===
    void* hctLigArgs[] = {
        &posqPtr, &particleIndicesPtr, &radiiPtr, &scaleFactorsPtr,
        &groupStartPtr, &numParticleGroups, &templateN,
        &totalParticles, &hctLigandDoublePtr
    };
    cu.executeKernel(computeHctLigandPairwiseDoubleKernel, hctLigArgs,
                     numBlocksN * blockSize, blockSize);

    // === Kernel 0c: Born radii in double from double hctReceptor + double hctLigand ===
    void* bornArgs[] = {
        &radiiPtr, &hctReceptorDoublePtr, &hctLigandDoublePtr,
        &totalParticles, &templateN, &bornRadiiDoublePtr
    };
    cu.executeKernel(computeBornRadiiOBCDoubleKernel, bornArgs,
                     numBlocksN * blockSize, blockSize);

    // === Kernel 1: prepareHessianIntermediatesDouble ===
    void* prepArgs[] = {
        &radiiPtr, &bornRadiiDoublePtr, &hctReceptorDoublePtr, &hctLigandDoublePtr,
        &dE_dRPtr, &totalParticles, &templateN,
        &dRdPsiPtr, &d2RdPsi2Ptr, &dE_dHCTPtr
    };
    cu.executeKernel(prepareHessianIntermediatesDoubleKernel, prepArgs,
                     numBlocksN * blockSize, blockSize);

    // === Kernel 2: computeHCTJacobianPairwiseDouble ===
    void* jacArgs[] = {
        &posqPtr, &particleIndicesPtr, &radiiPtr, &scaleFactorsPtr,
        &exclAtomsPtr, &exclStartPtr, &groupStartPtr,
        &numParticleGroups, &templateN,
        &receptorPosPtr, &receptorRadiiPtr, &receptorScalesPtr,
        &numReceptorAtoms, &totalParticles, &jacobianPtr
    };
    cu.executeKernel(computeHCTJacobianPairwiseDoubleKernel, jacArgs,
                     numBlocksN * blockSize, blockSize);

    // === Kernel 2b: computeReceptorPairwiseHessianDouble ===
    if (numReceptorAtoms > 0) {
        void* recHessArgs[] = {
            &posqPtr, &particleIndicesPtr, &radiiPtr,
            &groupStartPtr, &numParticleGroups, &templateN,
            &receptorPosPtr, &receptorRadiiPtr, &receptorScalesPtr,
            &numReceptorAtoms, &totalParticles, &recD2PsiPtr
        };
        cu.executeKernel(computeReceptorPairwiseHessianDoubleKernel, recHessArgs,
                         numBlocksN * blockSize, blockSize);
    } else {
        cu.clearBuffer(hessianRecD2Psi);
    }

    // === Kernel 3: computeBornCouplingMatrixDouble ===
    int includeSAInt = includeSurfaceArea ? 1 : 0;
    // The SA probe radius defaults to 0.14 nm in NONE/PAIRWISE modes;
    // GRID mode reuses the grid's probe radius. Mirrors the dispatch
    // in accumulateSADerivativesKernel (line 1097).
    float saProbeRadius = (receptorMode == IsolatedGBSAForce::GRID)
                          ? probeRadius : 0.14f;
    // The gbsaHessianDouble kernels take a `float prefactor` (cast to double
    // internally), so pass a float copy of the now-double prefactor member.
    float prefactorHessF = (float)prefactor;
    void* couplingArgs[] = {
        &posqPtr, &particleIndicesPtr, &chargesPtr, &radiiPtr,
        &bornRadiiDoublePtr, &dRdPsiPtr, &d2RdPsi2Ptr, &dE_dRPtr,
        &exclAtomsPtr, &exclStartPtr, &groupStartPtr,
        &numParticleGroups, &templateN, &prefactorHessF,
        &includeSAInt, &surfaceTension, &saProbeRadius,
        &totalParticles, &couplingPtr
    };
    cu.executeKernel(computeBornCouplingMatrixDoubleKernel, couplingArgs,
                     numBlocksN * blockSize, blockSize);

    // === Kernel 4: assembleGBSAHessianDouble ===
    int totalElements = dim3N * dim3N;
    int numBlocksH = (totalElements + blockSize - 1) / blockSize;
    void* assembleArgs[] = {
        &posqPtr, &particleIndicesPtr, &chargesPtr, &bornRadiiDoublePtr,
        &exclAtomsPtr, &exclStartPtr, &groupStartPtr,
        &numParticleGroups, &templateN, &prefactorHessF,
        &jacobianPtr, &couplingPtr, &dE_dHCTPtr, &scaleFactorsPtr, &radiiPtr,
        &dRdPsiPtr, &recD2PsiPtr,
        &totalParticles, &hessianPtr
    };
    cu.executeKernel(assembleGBSAHessianDoubleKernel, assembleArgs,
                     numBlocksH * blockSize, blockSize);

    hessianFullHost.resize(dim3N * dim3N);
    hessianMatrix.download(hessianFullHost);
    return hessianFullHost;
}
