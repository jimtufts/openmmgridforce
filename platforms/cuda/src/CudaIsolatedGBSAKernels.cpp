/* -------------------------------------------------------------------------- *
 *                              OpenMMGridForce                               *
 * -------------------------------------------------------------------------- */

#include "CudaIsolatedGBSAKernels.h"
#include "internal/SolvationFieldBuilder.h"
#include "GridForceTypes.h"
#include "BSplinePrefilter.h"
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

// cuBLAS error check (matches the style in CudaAdaptivePrefilter.cu).
#define CUBLAS_CHECK(call) do {                                          \
    cublasStatus_t st_ = (call);                                         \
    if (st_ != CUBLAS_STATUS_SUCCESS) {                                  \
        throw OpenMMException(std::string("cuBLAS error in ") + #call +  \
                              ": status=" + std::to_string((int)st_));   \
    }                                                                    \
} while (0)

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
      hessianPrecision(IsolatedGBSAForce::HESSIAN_DOUBLE),
      prefactor(0), includeSurfaceArea(false), surfaceTension(0), cutoffDistance(-1.0f),
      originX(0), originY(0), originZ(0), gridSpacing(0), probeRadius(0),
      numBins(0), interpolationMethod(0), hasHctDerivatives(false),
      useKDECorrections(false), hasBinnedKDEDerivatives(false),
      computeCrossTermGrid(false), crossTermNumBins(0), maxNeighbors(0),
      crossMode(IsolatedGBSAForce::CROSS_NONE),
      mirrorMode(IsolatedGBSAForce::MIRROR_NONE),
      numReceptorAtoms(0), receptorReferenceEnergyValue(0.0f),
      computeReceptorHCTGridKernel(nullptr),
      generateCrossTermGridKernel(nullptr),
      computeCrossTermFromGridKernel(nullptr),
      computeReceptorHCTPairwiseKernel(nullptr),
      computeLigandHCTKernel(nullptr),
      computeBornRadiiHCTKernel(nullptr), computeBornRadiiOBCKernel(nullptr),
      computeGBEnergyKernel(nullptr), computeSAEnergyKernel(nullptr),
      computeReceptorDeltaSAKernel(nullptr),
      accumulateReceptorSADerivativesKernel(nullptr),
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
      computeLigandGBBornDerivDoubleKernel(nullptr),
      prepareHessianIntermediatesDoubleKernel(nullptr),
      computeHCTJacobianPairwiseDoubleKernel(nullptr),
      computeReceptorPairwiseHessianDoubleKernel(nullptr),
      computeBornCouplingMatrixDoubleKernel(nullptr),
      assembleGBSAHessianDoubleKernel(nullptr),
      convertTiledHCTToDoubleKernel(nullptr),
      computeBornRadiiOBCDoubleKernel(nullptr),
      computeHctReceptorPairwiseDoubleKernel(nullptr),
      computeHctLigandPairwiseDoubleKernel(nullptr),
      pairwiseRecBornDoubleKernel(nullptr),
      pairwiseRecCouplingDoubleKernel(nullptr),
      pairwiseRecJacobianDoubleKernel(nullptr),
      pairwiseCrossBornDeriv1DoubleKernel(nullptr),
      pairwiseBornGradHessianDoubleKernel(nullptr),
      pairwiseOuterProductHessianDoubleKernel(nullptr) {
}

CudaCalcIsolatedGBSAForceKernel::~CudaCalcIsolatedGBSAForceKernel() {
    if (cublasInitialized && cublasHandle != nullptr) {
        cublasDestroy(cublasHandle);
        cublasHandle = nullptr;
        cublasInitialized = false;
    }
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
    hessianPrecision = force.getHessianPrecision();
    cutoffDistance = static_cast<float>(force.getCutoffDistance());
    // Unified cutoff: receptorLocalityCutoff is an internal alias for
    // cutoffDistance so callers can't inadvertently mismatch the two.
    // Matches OpenMM's single-cutoff convention.
    receptorLocalityCutoff = cutoffDistance;

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
        for (int d = 0; d < 3; d++)
            gridCountsHost[d] = counts[d];

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

        if (corrNData.size() == expectedBinnedKDESize) {
            useKDECorrections = true;
            hasBinnedKDEDerivatives = true;
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
        //
        crossMode = force.getCrossMode();
        mirrorMode = force.getMirrorMode();
        nearShellCutoff = force.getNearShellCutoff();
        fieldSwitchOn = force.getFieldSwitchOn();
        fieldSwitchOff = force.getFieldSwitchOff();
        mirrorScale = force.getMirrorScale();
        crossPerturbReceptorRadii = force.getCrossPerturbReceptorRadii();
        mirrorFieldCutoff = force.getMirrorFieldCutoff();
        pocketPadding = force.getPocketPadding();
        fieldInterpolationMethod = force.getFieldInterpolationMethod();
        // CROSS_EXACT reuses the direct pairwise kernel below; the field-based
        // modes are set up after the module is built (they need its kernels).
        computeCrossTermGrid = (crossMode == IsolatedGBSAForce::CROSS_EXACT);
        if (computeCrossTermGrid ||
            crossMode == IsolatedGBSAForce::CROSS_RADIUS_GRID ||
            mirrorMode != IsolatedGBSAForce::MIRROR_NONE) {
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
            // Derive the apo receptor Born radii when they were not supplied,
            // so GRID+CROSS_EXACT needs the same setup here as on Reference.
            vector<double> recBornBaseline = force.getReceptorBornRadiiBaseline();
            if ((int)recBornBaseline.size() != nRec) {
                if (!recBornBaseline.empty()) {
                    throw OpenMMException(
                        "IsolatedGBSAForce: receptorBornRadiiBaseline must "
                        "have length numReceptorAtoms.");
                }
                vector<double> recQ(nRec), recR(nRec), recS(nRec);
                for (int j = 0; j < nRec; j++)
                    force.getReceptorAtomParameters(j, recQ[j], recR[j], recS[j]);
                vector<double> apoHCT, apoWeights;
                SolvationFields::computeApoReceptor(
                    recPos, recQ, recR, recS,
                    force.getGBMethod() == IsolatedGBSAForce::OBC_II,
                    prefactor, force.getCutoffDistance(),
                    force.getIncludeSurfaceArea(), force.getSurfaceTension(),
                    nullptr, apoHCT, recBornBaseline, apoWeights);
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
        initRealBuffer(cu, receptorEnergyRef, 1, "isolatedGbsaReceptorEnergyRef");
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

        // Ligand-only pass buffers for getGroupReceptorContribution:
        // receptorContribution = groupLigandSelfEnergies(full radii)
        //                        - gbEnergyLigOnly(ligand-only radii).
        initRealBuffer(cu, bornRadiiLigOnly, totalParticles, "isolatedGbsaBornRadiiLigOnly");
        initRealBuffer(cu, hctZero, totalParticles, "isolatedGbsaHctZero");
        cu.clearBuffer(hctZero);  // stays zero (never written) => ligand-only Born radii
        initMixedEnergyBuffer(cu, groupLigOnlyEnergies, numParticleGroups, "isolatedGbsaGroupLigOnlyEnergies");
        initMixedEnergyBuffer(cu, scratchGroupEnergies, numParticleGroups, "isolatedGbsaScratchGroupEnergies");
        scratchForce.initialize<unsigned long long>(cu, 3 * cu.getPaddedNumAtoms(), "isolatedGbsaScratchForce");
        groupLigOnlyEnergiesHost.resize(numParticleGroups);

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
        CudaGridForceKernelSources::isolatedGBSAKernel +
        CudaGridForceKernelSources::isolatedGBSAFieldsKernel);
    computeLigandHCTKernel = cu.getKernel(module, "computeIsolatedLigandHCT");
    generateCrossTermGridKernel = cu.getKernel(module, "generateCrossTermGrid");
    generateCrossFieldSlicesKernel = cu.getKernel(module, "generateCrossFieldSlices");
    generateMirrorFieldSlicesKernel = cu.getKernel(module, "generateMirrorFieldSlices");
    accumulateReceptorNearHCTKernel = cu.getKernel(module, "accumulateReceptorNearHCT");
    computeCrossRadiusGridEnergyKernel = cu.getKernel(module, "computeCrossRadiusGridEnergy");
    applyCrossReceptorChainRuleKernel = cu.getKernel(module, "applyCrossReceptorChainRule");
    computeMirrorFromFieldKernel = cu.getKernel(module, "computeMirrorFromField");

    if (receptorMode == IsolatedGBSAForce::GRID &&
        (crossMode == IsolatedGBSAForce::CROSS_RADIUS_GRID ||
         mirrorMode != IsolatedGBSAForce::MIRROR_NONE))
        initializeGridReceptorTerms(force);
    computeCrossTermFromGridKernel = cu.getKernel(module, "computeCrossTermFromGrid");
    computeCrossTermPairwiseKernel = cu.getKernel(module, "computeCrossTermGBEnergy");
    accumulateCrossTermBornDerivativesKernel = cu.getKernel(module, "accumulateCrossTermBornDerivatives");
    computeBornRadiiHCTKernel = cu.getKernel(module, "computeBornRadiiHCT");
    computeBornRadiiOBCKernel = cu.getKernel(module, "computeBornRadiiOBC");
    computeGBEnergyKernel = cu.getKernel(module, "computeIsolatedGBEnergy");
    computeSAEnergyKernel = cu.getKernel(module, "computeIsolatedSAEnergy");
    computeReceptorDeltaSAKernel = cu.getKernel(module, "computeReceptorDeltaSA");
    accumulateReceptorSADerivativesKernel = cu.getKernel(module, "accumulateReceptorSADerivatives");
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
            &scratchDeDRPtr, &numTiles, &cutoffDistance
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

        // Strict cutoff on HCT — matches vanilla OpenMM GBSAOBCForce with
        // CutoffNonPeriodic. No cache: correct under REPX / MC / position
        // resets by construction (no stale state to invalidate).
        float localityCutoffVal = (receptorLocalityCutoff > 0.0f)
            ? receptorLocalityCutoff : -1.0f;
        CUdeviceptr hctCachePtr = (CUdeviceptr)0;

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

        // Convert fixed-point accumulators to float
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
        &globalScalingFactor, &groupScalingFactorsPtr, &groupUnscaledEnergiesPtr,
        &cutoffDistance
    };
    cu.executeKernel(computeGBEnergyKernel, energyArgs, numBlocks * blockSize, blockSize);

    // Step 4-lig: PAIRWISE receptor contribution = ligandSelf(full radii)
    //   - gbEnergyLigOnly(ligand-only radii). Reuse the Born-radii and GB-energy
    //   kernels with the receptor HCT excluded (hctZero), capturing only the
    //   ligand-self slot into groupLigOnlyEnergies; forces/total go to scratch.
    //   Host-side delta computed at readback. Mirrors the Reference platform.
    if (receptorMode == IsolatedGBSAForce::PAIRWISE && numReceptorAtoms > 0) {
        CUdeviceptr hctZeroPtr = hctZero.getDevicePointer();
        CUdeviceptr bornRadiiLigOnlyPtr = bornRadiiLigOnly.getDevicePointer();
        void* bornLigArgs[] = {
            &radiiPtr, &hctZeroPtr, &hctLigandPtr,
            &totalParticles, &numAtoms, &bornRadiiLigOnlyPtr
        };
        cu.executeKernel(gbMethod == IsolatedGBSAForce::HCT
                             ? computeBornRadiiHCTKernel : computeBornRadiiOBCKernel,
                         bornLigArgs, numBlocks * blockSize, blockSize);

        cu.clearBuffer(scratchForce);
        cu.clearBuffer(scratchGroupEnergies);
        cu.clearBuffer(groupLigOnlyEnergies);
        CUdeviceptr scratchForcePtr = scratchForce.getDevicePointer();
        CUdeviceptr scratchGEPtr = scratchGroupEnergies.getDevicePointer();
        CUdeviceptr groupLigOnlyPtr = groupLigOnlyEnergies.getDevicePointer();
        CUdeviceptr nullUnscaledPtr = 0;  // kernel guards on non-null; skip unscaled
        void* ligOnlyArgs[] = {
            &posqPtr, &particleIndicesPtr, &chargesPtr, &bornRadiiLigOnlyPtr,
            &groupStartPtr, &numParticleGroups, &numAtoms, prefactorArg,
            &scratchForcePtr, &scratchGEPtr, &groupLigOnlyPtr, &paddedNumAtoms,
            &globalScalingFactor, &groupScalingFactorsPtr, &nullUnscaledPtr,
            &cutoffDistance
        };
        cu.executeKernel(computeGBEnergyKernel, ligOnlyArgs, numBlocks * blockSize, blockSize);
    }

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
            &groupCrossPtr, &groupUnscaledEnergiesPtr,
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


    // Step 4c: GRID mode add-ons — radius-sliced cross field and mirror field.
    // Both run off one cell list over the pocket atoms. The cross term writes
    // into dE_dR, so when forces are on it must land after the Born-radius
    // derivatives are seeded and before the HCT chain rule consumes them;
    // when they are off it can run here.
    auto runGridAddOns = [&](CUdeviceptr dEdRTarget) {
        if (receptorMode != IsolatedGBSAForce::GRID)
            return;
        bool doCross = (crossMode == IsolatedGBSAForce::CROSS_RADIUS_GRID);
        bool doMirror = (mirrorMode != IsolatedGBSAForce::MIRROR_NONE);
        if (!doCross && !doMirror)
            return;

        int nx = gridCountsHost[0], ny = gridCountsHost[1], nz = gridCountsHost[2];
        int totalPoints = nx * ny * nz;
        float fox = originX, foy = originY, foz = originZ;
        float sp = gridSpacing;
        float son = (float) fieldSwitchOn, soff = (float) fieldSwitchOff;
        float cellOx = (float) cellOrigin[0], cellOy = (float) cellOrigin[1],
              cellOz = (float) cellOrigin[2];
        float cellSz = (float) cellSize;
        int ccx = cellCounts[0], ccy = cellCounts[1], ccz = cellCounts[2];
        float nearCut = (float) nearShellCutoff;
        float nearTaper = (float) max(fieldSwitchOff,
            nearShellCutoff - SolvationFields::DEFAULT_NEAR_TAPER_WIDTH);
        float fcutRt = cutoffDistance;
        int useOBC = (gbMethod == IsolatedGBSAForce::OBC_II) ? 1 : 0;
        int interp = fieldInterpolationMethod;

        CUdeviceptr pocketPosPtr = pocketPositions.getDevicePointer();
        CUdeviceptr pocketQPtr = pocketCharges.getDevicePointer();
        CUdeviceptr pocketRPtr = pocketRadii.getDevicePointer();
        CUdeviceptr pocketHCTPtr = pocketApoHCT.getDevicePointer();
        CUdeviceptr pocketBornPtr = pocketBornApo.getDevicePointer();
        CUdeviceptr pocketWPtr = pocketWeights.getDevicePointer();
        CUdeviceptr cellStartPtr = cellStartArr.getDevicePointer();
        CUdeviceptr cellAtomsPtr = cellAtomsArr.getDevicePointer();
        CUdeviceptr deltaPtr = recDeltaHCT.getDevicePointer();
        CUdeviceptr dRrecPtr = dCrossDRrec.getDevicePointer();

        if (doCross) {
            cu.clearBuffer(recDeltaHCT);
            cu.clearBuffer(dCrossDRrec);
            cu.clearBuffer(groupCrossTermEnergies);
            int perturbRec = crossPerturbReceptorRadii ? 1 : 0;
            CUdeviceptr nbrListPtr = neighborList.getDevicePointer();
            CUdeviceptr nbrCountPtr = neighborCount.getDevicePointer();
            CUdeviceptr nbrOverflowPtr = neighborOverflow.getDevicePointer();
            void* accArgs[] = {
                &posqPtr, &particleIndicesPtr, &radiiPtr, &scaleFactorsPtr,
                &groupStartPtr, &numParticleGroups, &numAtoms, &totalParticles,
                &pocketPosPtr, &pocketRPtr, &numPocket,
                &cellStartPtr, &cellAtomsPtr,
                &cellOx, &cellOy, &cellOz, &cellSz, &ccx, &ccy, &ccz,
                &nearCut, &nearTaper, &perturbRec, &deltaPtr,
                &nbrListPtr, &nbrCountPtr, &maxNeighbors, &nbrOverflowPtr
            };
            cu.executeKernel(accumulateReceptorNearHCTKernel, accArgs,
                             numBlocks * blockSize, blockSize);

            CUdeviceptr crossFieldPtr = crossFieldData.getDevicePointer();
            CUdeviceptr crossRadiiPtr = crossSliceRadii.getDevicePointer();
            CUdeviceptr groupCrossPtr = groupCrossTermEnergies.getDevicePointer();
            void* crossArgs[] = {
                &posqPtr, &particleIndicesPtr, &chargesPtr, &bornRadiiPtr,
                &groupStartPtr, &numParticleGroups, &numAtoms, &totalParticles,
                prefactorArg,
                &crossFieldPtr, &crossRadiiPtr, &numCrossSlices, &totalPoints,
                &fox, &foy, &foz, &nx, &ny, &nz, &sp, &interp,
                &pocketPosPtr, &pocketQPtr, &pocketRPtr, &pocketHCTPtr,
                &pocketBornPtr, &numPocket,
                &nbrListPtr, &nbrCountPtr, &maxNeighbors,
                &nearCut, &nearTaper, &son, &soff, &fcutRt, &useOBC,
                &deltaPtr, &dRrecPtr, &dEdRTarget,
                &forcePtr, &paddedNumAtoms, &groupCrossPtr,
                &groupUnscaledEnergiesPtr,
                &globalScalingFactor, &groupScalingFactorsPtr
            };
            cu.executeKernel(computeCrossRadiusGridEnergyKernel, crossArgs,
                             numBlocks * blockSize, blockSize);

            if (includeForces && crossPerturbReceptorRadii) {
                void* chainArgs[] = {
                    &posqPtr, &particleIndicesPtr, &radiiPtr, &scaleFactorsPtr,
                    &groupStartPtr, &numParticleGroups, &numAtoms, &totalParticles,
                    &pocketPosPtr, &pocketRPtr, &pocketHCTPtr, &numPocket,
                    &nbrListPtr, &nbrCountPtr, &maxNeighbors,
                    &nearCut, &nearTaper, &useOBC, &deltaPtr, &dRrecPtr,
                    &forcePtr, &paddedNumAtoms,
                    &globalScalingFactor, &groupScalingFactorsPtr
                };
                cu.executeKernel(applyCrossReceptorChainRuleKernel, chainArgs,
                                 numBlocks * blockSize, blockSize);
            }
        }

        if (doMirror) {
            cu.clearBuffer(groupMirrorEnergies);
            CUdeviceptr mCellStartPtr = mirrorCellStartArr.getDevicePointer();
            CUdeviceptr mCellAtomsPtr = mirrorCellAtomsArr.getDevicePointer();
            float mOx = (float) mirrorCellOrigin[0], mOy = (float) mirrorCellOrigin[1],
                  mOz = (float) mirrorCellOrigin[2];
            float mSz = (float) mirrorCellSize;
            int mcx = mirrorCellCounts[0], mcy = mirrorCellCounts[1],
                mcz = mirrorCellCounts[2];
            CUdeviceptr mirrorFieldPtr = mirrorFieldData.getDevicePointer();
            CUdeviceptr mirrorRadiiPtr = mirrorSliceRadii.getDevicePointer();
            CUdeviceptr slicePtr = atomMirrorSlice.getDevicePointer();
            CUdeviceptr groupMirrorPtr = groupMirrorEnergies.getDevicePointer();
            float mscale = (float) mirrorScale;
            void* mirrorArgs[] = {
                &posqPtr, &particleIndicesPtr, &slicePtr,
                &groupStartPtr, &numParticleGroups, &numAtoms, &totalParticles,
                &mirrorFieldPtr, &mirrorRadiiPtr, &totalPoints,
                &fox, &foy, &foz, &nx, &ny, &nz, &sp, &interp,
                &pocketPosPtr, &pocketRPtr, &pocketWPtr, &numPocket,
                &mCellStartPtr, &mCellAtomsPtr,
                &mOx, &mOy, &mOz, &mSz, &mcx, &mcy, &mcz,
                &son, &soff, &mscale, &fcutRt,
                &forcePtr, &paddedNumAtoms, &groupMirrorPtr,
                &groupUnscaledEnergiesPtr,
                &globalScalingFactor, &groupScalingFactorsPtr
            };
            cu.executeKernel(computeMirrorFromFieldKernel, mirrorArgs,
                             numBlocks * blockSize, blockSize);
        }
    };

    if (!includeForces)
        runGridAddOns(dE_dR.getDevicePointer());

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

        // 4b.3-pre: Fresh reference energy this step.
        // Compute receptor GB energy with receptor-alone Born radii (constant)
        // via the SAME kernel invocation used for the with-ligand groups so
        // the desolvation subtraction inside accumulateDesolvationOnGPU
        // cancels any per-launch rounding that the previous init-time cached
        // scalar had been leaking through (was ~64 kJ/mol systematic offset
        // on OBC_RL delta for highly charged systems).
        cu.clearBuffer(receptorEnergyRef);
        CUdeviceptr receptorEnergyRefPtr = receptorEnergyRef.getDevicePointer();
        void* refEnergyStepArgs[] = {
            &receptorPosPtr, &receptorChargesPtr, &receptorBornRadiiRefPtr,
            &numReceptorAtoms, prefactorArg, &receptorEnergyRefPtr, &numTiles,
            &cutoffDistance
        };
        cu.executeKernel(computeReceptorGBEnergyTiledKernel, refEnergyStepArgs,
                         recNumBlocksTiled * recBlockSize, recBlockSize);

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
                    &numTiles, &cutoffDistance
                };
                cu.executeKernel(computeReceptorGBEnergyAndDeDRTiledKernel,
                                 tiledArgs, recNumBlocksTiled * recBlockSize, recBlockSize);
            } else {
                // Energy only
                void* recEnergyArgs[] = {
                    &receptorPosPtr, &receptorChargesPtr, &groupBornRadiiPtr,
                    &numReceptorAtoms, prefactorArg, &receptorEnergyPtr, &numTiles,
                    &cutoffDistance
                };
                cu.executeKernel(computeReceptorGBEnergyTiledKernel, recEnergyArgs, recNumBlocksTiled * recBlockSize, recBlockSize);
            }

            void* accumArgs[] = {
                &receptorEnergyPtr, &receptorEnergyRefPtr, &g,
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
                &numAtoms, prefactorArg, &receptorDeDRPtr, &cutoffDistance
            };
            cu.executeKernel(accumulateCrossTermReceptorDeDRKernel,
                             crossDedrArgs, crossBlocks * blockSize, blockSize);
        }

        // 4b.3a++: Accumulate receptor SA contribution to receptorDeDR.
        // dE_SA_rec/dR_born_rec is required so bornForcesRec captures the
        // chain-rule path from receptor ΔSA back to Cartesian force on the
        // ligand. Without this the SA energy is included but its derivative
        // is dropped, breaking F = -grad E for the SA term.
        if (includeForces && includeSurfaceArea) {
            float rProbe = 0.14f;
            int totalRecWork = numParticleGroups * numReceptorAtoms;
            int saDerivBlocks = (totalRecWork + blockSize - 1) / blockSize;
            void* recSaDerivArgs[] = {
                &receptorRadiiPtr, &receptorBornRadiiPtr,
                &numReceptorAtoms, &numParticleGroups,
                &surfaceTension, &rProbe,
                &receptorDeDRPtr,
                &globalScalingFactor, &groupScalingFactorsPtr
            };
            cu.executeKernel(accumulateReceptorSADerivativesKernel,
                             recSaDerivArgs, saDerivBlocks * blockSize, blockSize);
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

        // Strict cutoff on cross-term GB energy — matches vanilla OpenMM
        // GBSAOBCForce with CutoffNonPeriodic. No cache: correct under REPX
        // / MC by construction.
        float forceTileSkipCutoff = (receptorLocalityCutoff > 0.0f)
            ? receptorLocalityCutoff : -1.0f;
        CUdeviceptr crossCachePtr = (CUdeviceptr)0;

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
            &globalScalingFactor, &groupScalingFactorsPtr,
            &cutoffDistance
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
                &globalScalingFactor, &groupScalingFactorsPtr,
                &cutoffDistance
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
            float crossTermCutoff = (receptorLocalityCutoff > 0.0f)
                ? receptorLocalityCutoff : -1.0f;
            void* crossDerivArgs[] = {
                &posqPtr, &particleIndicesPtr, &chargesPtr,
                &bornRadiiPtr,
                &receptorPosPtr2, &receptorChargesPtr2, &receptorBornRadiiPtr2,
                &groupStartPtr, &numParticleGroups,
                &numReceptorAtoms, &numAtoms,
                prefactorArg,
                &dE_dRPtr,
                &globalScalingFactor, &groupScalingFactorsPtr,
                &crossTermCutoff,
            };
            cu.executeKernel(accumulateCrossTermBornDerivativesKernel,
                             crossDerivArgs, numBlocks * blockSize, blockSize);
        }

        if (includeForces)
            runGridAddOns(dE_dRPtr);

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
            // Receptor contribution = ligand GB with receptor descreening minus
            // ligand GB with ligand-only radii (both *scale). Matches Reference.
            if (numReceptorAtoms > 0) {
                downloadMixedEnergy(cu, groupLigOnlyEnergies, groupLigOnlyEnergiesHost);
                for (int g = 0; g < numParticleGroups; g++)
                    groupReceptorContributionsHost[g] =
                        groupLigandSelfEnergiesHost[g] - groupLigOnlyEnergiesHost[g];
            }
        } else if (receptorMode == IsolatedGBSAForce::GRID) {
            if (crossMode == IsolatedGBSAForce::CROSS_RADIUS_GRID) {
                vector<int> flag(1, 0);
                neighborOverflow.download(flag);
                if (flag[0] > maxNeighbors)
                    throw OpenMMException(
                        "IsolatedGBSAForce: near-shell neighbour list overflowed ("
                        + to_string(flag[0]) + " > " + to_string(maxNeighbors) +
                        "); reduce nearShellCutoff or report this receptor");
            }
            if (crossMode != IsolatedGBSAForce::CROSS_NONE)
                downloadMixedEnergy(cu, groupCrossTermEnergies, groupCrossTermEnergiesHost);
            if (mirrorMode != IsolatedGBSAForce::MIRROR_NONE) {
                downloadMixedEnergy(cu, groupMirrorEnergies, groupMirrorEnergiesHost);
                for (int g = 0; g < numParticleGroups; g++)
                    groupReceptorDesolvationsHost[g] = groupMirrorEnergiesHost[g];
            }
        }

        // The cross and mirror kernels write their own buffers, so fold them
        // into the per-group energy here: getParticleGroupEnergies() must
        // report the group's full contribution, as the Reference platform
        // does, or a sampler that builds its Hamiltonian from it integrates
        // forces it never accounts for.
        if (receptorMode == IsolatedGBSAForce::GRID) {
            for (int g = 0; g < numParticleGroups; g++) {
                if (crossMode != IsolatedGBSAForce::CROSS_NONE)
                    groupEnergiesHost[g] += groupCrossTermEnergiesHost[g];
                if (mirrorMode != IsolatedGBSAForce::MIRROR_NONE)
                    groupEnergiesHost[g] += groupMirrorEnergiesHost[g];
            }
        }

        double totalEnergy = 0.0;
        for (int g = 0; g < numParticleGroups; g++)
            totalEnergy += groupEnergiesHost[g];

        return totalEnergy;
    }

    return 0.0;
}

void CudaCalcIsolatedGBSAForceKernel::updateParametersInContext(ContextImpl& context,
                                                                 const IsolatedGBSAForce& force) {
    // Re-upload atom parameters. Route through uploadRealScalars so the
    // vector element size matches the CudaArray element size at this
    // context's precision (float in single, double in mixed/double).
    vector<double> chargesVec(numAtoms), radiiVec(numAtoms), scalesVec(numAtoms);
    for (int i = 0; i < numAtoms; i++) {
        double q, r, s;
        force.getAtomParameters(i, q, r, s);
        chargesVec[i] = q;
        radiiVec[i] = r;
        scalesVec[i] = s;
    }
    // Buffers were sized+typed at init time; here we just re-upload with
    // the element type matching the context precision.
    if (cu.getUseDoublePrecision()) {
        charges.upload(chargesVec);
        radii.upload(radiiVec);
        scaleFactors.upload(scalesVec);
    } else {
        std::vector<float> cf(chargesVec.begin(), chargesVec.end());
        std::vector<float> rf(radiiVec.begin(), radiiVec.end());
        std::vector<float> sf(scalesVec.begin(), scalesVec.end());
        charges.upload(cf);
        radii.upload(rf);
        scaleFactors.upload(sf);
    }

    // Update solvent parameters
    double soluteDielectric = force.getSoluteDielectric();
    double solventDielectric = force.getSolventDielectric();
    prefactor = -ONE_4PI_EPS0 * (1.0/soluteDielectric - 1.0/solventDielectric);

    includeSurfaceArea = force.getIncludeSurfaceArea();
    surfaceTension = static_cast<float>(force.getSurfaceTension());
    cutoffDistance = static_cast<float>(force.getCutoffDistance());
    // Unified cutoff: receptorLocalityCutoff is an internal alias for
    // cutoffDistance so callers can't inadvertently mismatch the two.
    // Matches OpenMM's single-cutoff convention.
    receptorLocalityCutoff = cutoffDistance;

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

    // Download the full K*numReceptorAtoms buffer, then slice out the requested group.
    size_t total = static_cast<size_t>(numReceptorAtoms) * numParticleGroups;
    vector<float> allBorn(total);
    receptorBornRadii.download(allBorn);
    size_t offset = static_cast<size_t>(groupIndex) * numReceptorAtoms;
    vector<float> recBornRadii(allBorn.begin() + offset,
                                allBorn.begin() + offset + numReceptorAtoms);
    groupReceptorBornRadiiHost[groupIndex] = recBornRadii;

    vector<double> result(recBornRadii.begin(), recBornRadii.end());
    return result;
}

vector<double> CudaCalcIsolatedGBSAForceKernel::getParticleGroupUnscaledEnergies() const {
    vector<double> result(groupUnscaledEnergiesHost.begin(), groupUnscaledEnergiesHost.end());
    return result;
}


// ===========================================================================
// GRID-mode receptor add-ons: field setup and generation
// ===========================================================================

shared_ptr<SolvationFieldGrid> CudaCalcIsolatedGBSAForceKernel::generateCrossFieldOnDevice(
        const IsolatedGBSAForce& force, const vector<double>& sliceR,
        const vector<double>& bornApo) {

    auto grid = force.getDesolvationGrid();
    int nx, ny, nz;
    grid->getCounts(nx, ny, nz);
    double ox, oy, oz;
    grid->getOrigin(ox, oy, oz);
    int totalPoints = nx * ny * nz;
    int nSlices = (int) sliceR.size();

    auto field = make_shared<SolvationFieldGrid>(
        nx, ny, nz, grid->getSpacing(), nSlices,
        SolvationFieldGrid::CROSS_GB, fieldInterpolationMethod);
    field->setOrigin(ox, oy, oz);
    field->setSwitchRadii(fieldSwitchOn, fieldSwitchOff);
    field->setSliceParameters(sliceR);

    CudaArray out, radiiBuf, bornBuf;
    out.initialize<float>(cu, (size_t) nSlices * totalPoints, "isolatedGbsaCrossFieldGen");
    vector<float> sliceF(sliceR.begin(), sliceR.end());
    radiiBuf.initialize<float>(cu, nSlices, "isolatedGbsaCrossSliceGen");
    radiiBuf.upload(sliceF);
    uploadRealScalars(cu, bornBuf, bornApo, "isolatedGbsaRecBornApoGen");

    CUdeviceptr outPtr = out.getDevicePointer();
    CUdeviceptr posPtr = receptorPositions.getDevicePointer();
    CUdeviceptr qPtr = receptorCharges.getDevicePointer();
    CUdeviceptr bornPtr = bornBuf.getDevicePointer();
    CUdeviceptr radiiPtr = radiiBuf.getDevicePointer();
    float fox = (float) ox, foy = (float) oy, foz = (float) oz;
    float sp = (float) grid->getSpacing();
    float son = (float) fieldSwitchOn, soff = (float) fieldSwitchOff;
    float fcut = cutoffDistance;
    void* args[] = {&outPtr, &posPtr, &qPtr, &bornPtr, &numReceptorAtoms,
                    &radiiPtr, &nSlices, &fox, &foy, &foz,
                    &nx, &ny, &nz, &sp, &son, &soff, &fcut, &totalPoints};
    int blockSize = 128;
    int blocks = min((totalPoints + blockSize - 1) / blockSize, 4096);
    cu.executeKernel(generateCrossFieldSlicesKernel, args, blocks * blockSize, blockSize);

    vector<float> data((size_t) nSlices * totalPoints);
    out.download(data);
    if (fieldInterpolationMethod == InterpolationMethod::TRICUBIC_BSPLINE) {
        for (int k = 0; k < nSlices; k++) {
            vector<float> slice(data.begin() + (size_t) k * totalPoints,
                                data.begin() + (size_t) (k + 1) * totalPoints);
            bsplinePrefilter3D(slice, nx, ny, nz);
            copy(slice.begin(), slice.end(),
                 data.begin() + (size_t) k * totalPoints);
        }
    }
    field->setData(std::move(data));
    return field;
}

shared_ptr<SolvationFieldGrid> CudaCalcIsolatedGBSAForceKernel::generateMirrorFieldOnDevice(
        const IsolatedGBSAForce& force, const vector<double>& sliceR,
        const vector<double>& weights) {

    auto grid = force.getDesolvationGrid();
    int nx, ny, nz;
    grid->getCounts(nx, ny, nz);
    double ox, oy, oz;
    grid->getOrigin(ox, oy, oz);
    int totalPoints = nx * ny * nz;
    int nSlices = (int) sliceR.size();
    int counts[3] = {nx, ny, nz};
    double origin[3] = {ox, oy, oz};

    // The mirror field reaches further than the runtime near lists, so it
    // gets its own, wider atom selection.
    const vector<double>& recPos = force.getReceptorPositions();
    double mirrorRange = (cutoffDistance > 0.0f)
                         ? min(mirrorFieldCutoff, (double) cutoffDistance)
                         : mirrorFieldCutoff;
    vector<int> sel = SolvationFields::selectPocketAtoms(
        recPos, origin, grid->getSpacing(), counts, mirrorRange);
    int nSel = (int) sel.size();

    auto field = make_shared<SolvationFieldGrid>(
        nx, ny, nz, grid->getSpacing(), nSlices,
        SolvationFieldGrid::MIRROR, fieldInterpolationMethod);
    field->setOrigin(ox, oy, oz);
    field->setSwitchRadii(fieldSwitchOn, fieldSwitchOff);
    field->setSliceParameters(sliceR);

    vector<double> selPos(3 * nSel), selRad(nSel), selW(nSel);
    for (int a = 0; a < nSel; a++) {
        int j = sel[a];
        for (int d = 0; d < 3; d++) selPos[a * 3 + d] = recPos[j * 3 + d];
        double q, r, sc;
        force.getReceptorAtomParameters(j, q, r, sc);
        selRad[a] = r;
        selW[a] = weights[j];
    }

    CudaArray out, posBuf, radBuf, wBuf, radiiBuf;
    out.initialize<float>(cu, (size_t) nSlices * totalPoints, "isolatedGbsaMirrorFieldGen");
    uploadRealPositions(cu, posBuf, selPos, nSel, "isolatedGbsaMirrorGenPos");
    uploadRealScalars(cu, radBuf, selRad, "isolatedGbsaMirrorGenRad");
    uploadRealScalars(cu, wBuf, selW, "isolatedGbsaMirrorGenW");
    vector<float> sliceF(sliceR.begin(), sliceR.end());
    radiiBuf.initialize<float>(cu, nSlices, "isolatedGbsaMirrorSliceGen");
    radiiBuf.upload(sliceF);

    CUdeviceptr outPtr = out.getDevicePointer();
    CUdeviceptr posPtr = posBuf.getDevicePointer();
    CUdeviceptr radPtr = radBuf.getDevicePointer();
    CUdeviceptr wPtr = wBuf.getDevicePointer();
    CUdeviceptr radiiPtr = radiiBuf.getDevicePointer();
    float fox = (float) ox, foy = (float) oy, foz = (float) oz;
    float sp = (float) grid->getSpacing();
    float son = (float) fieldSwitchOn, soff = (float) fieldSwitchOff;
    float bcut = (float) mirrorRange;
    void* args[] = {&outPtr, &posPtr, &radPtr, &wPtr, &nSel,
                    &radiiPtr, &nSlices, &fox, &foy, &foz,
                    &nx, &ny, &nz, &sp, &son, &soff, &bcut, &totalPoints};
    int blockSize = 128;
    int blocks = min((totalPoints + blockSize - 1) / blockSize, 4096);
    cu.executeKernel(generateMirrorFieldSlicesKernel, args, blocks * blockSize, blockSize);

    vector<float> data((size_t) nSlices * totalPoints);
    out.download(data);
    if (fieldInterpolationMethod == InterpolationMethod::TRICUBIC_BSPLINE) {
        for (int k = 0; k < nSlices; k++) {
            vector<float> slice(data.begin() + (size_t) k * totalPoints,
                                data.begin() + (size_t) (k + 1) * totalPoints);
            bsplinePrefilter3D(slice, nx, ny, nz);
            copy(slice.begin(), slice.end(),
                 data.begin() + (size_t) k * totalPoints);
        }
    }
    field->setData(std::move(data));
    return field;
}

void CudaCalcIsolatedGBSAForceKernel::initializeGridReceptorTerms(
        const IsolatedGBSAForce& force) {

    auto grid = force.getDesolvationGrid();
    int nx, ny, nz;
    grid->getCounts(nx, ny, nz);
    double ox, oy, oz;
    grid->getOrigin(ox, oy, oz);
    int counts[3] = {nx, ny, nz};
    double origin[3] = {ox, oy, oz};
    int totalPoints = nx * ny * nz;
    double spacing = grid->getSpacing();

    const vector<double>& recPos = force.getReceptorPositions();
    int nRec = force.getNumReceptorAtoms();
    vector<double> recQ(nRec), recR(nRec), recS(nRec);
    for (int j = 0; j < nRec; j++)
        force.getReceptorAtomParameters(j, recQ[j], recR[j], recS[j]);

    const vector<double>& supplied = force.getReceptorBornRadiiBaseline();
    const vector<double>* suppliedPtr =
        ((int) supplied.size() == nRec) ? &supplied : nullptr;
    vector<double> apoHCT, apoBorn, weights;
    SolvationFields::computeApoReceptor(
        recPos, recQ, recR, recS, gbMethod == IsolatedGBSAForce::OBC_II,
        prefactor, cutoffDistance, includeSurfaceArea, surfaceTension,
        suppliedPtr, apoHCT, apoBorn, weights);

    // Pocket set and cell list, sized by the widest near shell in use.
    vector<int> pocket = SolvationFields::selectPocketAtoms(
        recPos, origin, spacing, counts, pocketPadding);
    numPocket = (int) pocket.size();
    if (numPocket == 0)
        throw OpenMMException("IsolatedGBSAForce: no receptor atoms near the grid box");

    double cellCutoff = (crossMode == IsolatedGBSAForce::CROSS_RADIUS_GRID)
                        ? nearShellCutoff : fieldSwitchOff;
    SolvationFields::PocketCellList cl;
    SolvationFields::buildPocketCellList(
        recPos, pocket, cellCutoff / SolvationFields::CELLS_PER_CUTOFF, cl);
    for (int d = 0; d < 3; d++) {
        cellOrigin[d] = cl.origin[d];
        cellCounts[d] = cl.counts[d];
    }
    cellSize = cl.cellSize;

    // Pocket-local arrays: the per-group scratch then scales with the pocket,
    // not the protein.
    vector<int> recToPocket(nRec, -1);
    for (int a = 0; a < numPocket; a++)
        recToPocket[pocket[a]] = a;
    vector<double> pPos(3 * numPocket), pQ(numPocket), pR(numPocket),
                   pHCT(numPocket), pBorn(numPocket), pW(numPocket);
    for (int a = 0; a < numPocket; a++) {
        int j = pocket[a];
        for (int d = 0; d < 3; d++) pPos[a * 3 + d] = recPos[j * 3 + d];
        pQ[a] = recQ[j];
        pR[a] = recR[j];
        pHCT[a] = apoHCT[j];
        pBorn[a] = apoBorn[j];
        pW[a] = weights[j];
    }
    vector<int> cellAtomsLocal(cl.atoms.size());
    for (size_t a = 0; a < cl.atoms.size(); a++)
        cellAtomsLocal[a] = recToPocket[cl.atoms[a]];

    uploadRealPositions(cu, pocketPositions, pPos, numPocket, "isolatedGbsaPocketPos");
    uploadRealScalars(cu, pocketCharges, pQ, "isolatedGbsaPocketQ");
    uploadRealScalars(cu, pocketRadii, pR, "isolatedGbsaPocketR");
    uploadRealScalars(cu, pocketApoHCT, pHCT, "isolatedGbsaPocketApoHCT");
    uploadRealScalars(cu, pocketBornApo, pBorn, "isolatedGbsaPocketBornApo");
    uploadRealScalars(cu, pocketWeights, pW, "isolatedGbsaPocketW");
    cellStartArr.initialize<int>(cu, (int) cl.cellStart.size(), "isolatedGbsaCellStart");
    cellStartArr.upload(cl.cellStart);
    cellAtomsArr.initialize<int>(cu, max((int) cellAtomsLocal.size(), 1), "isolatedGbsaCellAtoms");
    if (!cellAtomsLocal.empty())
        cellAtomsArr.upload(cellAtomsLocal);

    if (mirrorMode != IsolatedGBSAForce::MIRROR_NONE) {
        SolvationFields::PocketCellList mcl;
        SolvationFields::buildPocketCellList(
            recPos, pocket, fieldSwitchOff / SolvationFields::CELLS_PER_CUTOFF, mcl);
        for (int d = 0; d < 3; d++) {
            mirrorCellOrigin[d] = mcl.origin[d];
            mirrorCellCounts[d] = mcl.counts[d];
        }
        mirrorCellSize = mcl.cellSize;
        vector<int> mLocal(mcl.atoms.size());
        for (size_t a = 0; a < mcl.atoms.size(); a++)
            mLocal[a] = recToPocket[mcl.atoms[a]];
        mirrorCellStartArr.initialize<int>(cu, (int) mcl.cellStart.size(),
                                           "isolatedGbsaMirrorCellStart");
        mirrorCellStartArr.upload(mcl.cellStart);
        mirrorCellAtomsArr.initialize<int>(cu, max((int) mLocal.size(), 1),
                                           "isolatedGbsaMirrorCellAtoms");
        if (!mLocal.empty())
            mirrorCellAtomsArr.upload(mLocal);
    }

    if (crossMode == IsolatedGBSAForce::CROSS_RADIUS_GRID) {
        // Capacity from the densest pocket atom's own neighbourhood, which
        // bounds what any ligand atom in the box can see, plus headroom.
        double cut2 = nearShellCutoff * nearShellCutoff;
        int densest = 0;
        for (int a = 0; a < numPocket; a++) {
            int ja = pocket[a];
            int cnt = 0;
            for (int b = 0; b < numPocket; b++) {
                int jb = pocket[b];
                double dx = recPos[ja * 3] - recPos[jb * 3];
                double dy = recPos[ja * 3 + 1] - recPos[jb * 3 + 1];
                double dz = recPos[ja * 3 + 2] - recPos[jb * 3 + 2];
                if (dx * dx + dy * dy + dz * dz <= cut2)
                    cnt++;
            }
            densest = max(densest, cnt);
        }
        maxNeighbors = min(max((int) (densest * 1.3) + 16, 32), numPocket);
        int totalLig = numParticleGroups * numAtoms;
        neighborList.initialize<int>(cu, (size_t) totalLig * maxNeighbors,
                                    "isolatedGbsaNeighborList");
        neighborCount.initialize<int>(cu, totalLig, "isolatedGbsaNeighborCount");
        neighborOverflow.initialize<int>(cu, 1, "isolatedGbsaNeighborOverflow");
        vector<int> zero(1, 0);
        neighborOverflow.upload(zero);
    }

    initRealBuffer(cu, recDeltaHCT, numParticleGroups * numPocket, "isolatedGbsaRecDeltaHCT");
    initRealBuffer(cu, dCrossDRrec, numParticleGroups * numPocket, "isolatedGbsaDCrossDRrec");
    initMixedEnergyBuffer(cu, groupMirrorEnergies, numParticleGroups, "isolatedGbsaGroupMirror");
    groupMirrorEnergiesHost.resize(numParticleGroups, 0.0);

    if (crossMode == IsolatedGBSAForce::CROSS_RADIUS_GRID) {
        auto field = force.getCrossField();
        if (!field) {
            vector<double> sliceR = force.getCrossFieldRadii();
            if (sliceR.empty()) {
                vector<double> ligRadii(numAtoms);
                for (int i = 0; i < numAtoms; i++) {
                    double q, r, sc;
                    force.getAtomParameters(i, q, r, sc);
                    ligRadii[i] = r;
                }
                sliceR = SolvationFields::defaultCrossFieldRadii(
                    ligRadii, force.getNumCrossFieldSlices());
            }
            field = generateCrossFieldOnDevice(force, sliceR, apoBorn);
            const_cast<IsolatedGBSAForce&>(force).setCrossField(field);
        }
        if (field->getNumPoints() != totalPoints)
            throw OpenMMException("IsolatedGBSAForce: cross field geometry does not "
                                  "match the desolvation grid");
        numCrossSlices = field->getNumSlices();
        if (numCrossSlices > 16)
            throw OpenMMException("IsolatedGBSAForce: at most 16 cross field slices "
                                  "are supported on CUDA");
        crossFieldData.initialize<float>(cu, (int) field->getData().size(),
                                         "isolatedGbsaCrossField");
        crossFieldData.upload(field->getData());
        vector<float> sr(field->getSliceParameters().begin(),
                         field->getSliceParameters().end());
        crossSliceRadii.initialize<float>(cu, numCrossSlices, "isolatedGbsaCrossSliceR");
        crossSliceRadii.upload(sr);
        fieldInterpolationMethod = field->getInterpolationMethod();
    }

    if (mirrorMode != IsolatedGBSAForce::MIRROR_NONE) {
        vector<double> ligRadii(numAtoms), ligScales(numAtoms);
        for (int i = 0; i < numAtoms; i++) {
            double q, r, sc;
            force.getAtomParameters(i, q, r, sc);
            ligRadii[i] = r;
            ligScales[i] = sc;
        }
        vector<double> sliceR = SolvationFields::distinctScaledRadii(ligRadii, ligScales);
        vector<int> slot(numAtoms);
        for (int i = 0; i < numAtoms; i++) {
            double sv = (ligRadii[i] - IsolatedGBSAForce::DIELECTRIC_OFFSET) * ligScales[i];
            slot[i] = SolvationFields::sliceForScaledRadius(sliceR, sv);
        }
        auto field = force.getMirrorField();
        if (!field) {
            field = generateMirrorFieldOnDevice(force, sliceR, weights);
            const_cast<IsolatedGBSAForce&>(force).setMirrorField(field);
        }
        if (field->getNumSlices() != (int) sliceR.size())
            throw OpenMMException("IsolatedGBSAForce: the mirror field slice count does "
                                  "not match the ligand template's descreener radii");
        numMirrorSlices = field->getNumSlices();
        if (numMirrorSlices > 16)
            throw OpenMMException("IsolatedGBSAForce: at most 16 mirror field slices "
                                  "are supported on CUDA");
        mirrorFieldData.initialize<float>(cu, (int) field->getData().size(),
                                          "isolatedGbsaMirrorField");
        mirrorFieldData.upload(field->getData());
        vector<float> sr(field->getSliceParameters().begin(),
                         field->getSliceParameters().end());
        mirrorSliceRadii.initialize<float>(cu, numMirrorSlices, "isolatedGbsaMirrorSliceR");
        mirrorSliceRadii.upload(sr);
        atomMirrorSlice.initialize<int>(cu, numAtoms, "isolatedGbsaAtomMirrorSlice");
        atomMirrorSlice.upload(slot);
        fieldInterpolationMethod = field->getInterpolationMethod();
    }
}

vector<double> CudaCalcIsolatedGBSAForceKernel::computeHessian(ContextImpl& context) {
    // The GRID-mode receptor add-ons contribute second derivatives that the
    // Hessian kernels do not carry. A loud rejection beats a matrix that is
    // silently inconsistent with the forces.
    if (crossMode != IsolatedGBSAForce::CROSS_NONE ||
        mirrorMode != IsolatedGBSAForce::MIRROR_NONE) {
        throw OpenMMException(
            "IsolatedGBSAForce: the Hessian does not include the GRID-mode "
            "cross or mirror terms; disable them to compute it.");
    }
    cu.setAsCurrent();

    int totalParticles = numParticleGroups * numAtoms;
    int templateN = numAtoms;
    int dim3N = 3 * totalParticles;
    if (totalParticles == 0) {
        return std::vector<double>();
    }

    // Precision toggle. GRID uses a full float-compute kernel set. PAIRWISE
    // uses a storage-only downgrade: compute stays double, only the final
    // dim3N*dim3N hessian buffer accumulates as float (hardware atomicAdd
    // vs software-CAS double on pre-sm_60).
    bool useFloatBufPair = false;
    if (hessianPrecision == IsolatedGBSAForce::HESSIAN_FLOAT) {
        if (receptorMode == IsolatedGBSAForce::GRID) {
            return computeHessianGridFloat(context);
        }
        if (receptorMode == IsolatedGBSAForce::PAIRWISE) {
            useFloatBufPair = true;
        } else {
            throw OpenMMException(
                "IsolatedGBSAForce: HESSIAN_FLOAT for receptorMode == NONE is "
                "not implemented; use HESSIAN_DOUBLE.");
        }
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
        initRealBuffer(cu, hessianLigGBdEdR, totalParticles, "isolatedGbsaHessLigGBdEdR");
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
        computeHCTJacobianGridKernel             = cu.getKernel(module, "computeHCTJacobian");
        computeReceptorGridHessianKernel         = cu.getKernel(module, "computeReceptorGridHessian");
        computeBornCouplingMatrixKernel          = cu.getKernel(module, "computeBornCouplingMatrix");
        assembleGBSAHessianKernel                = cu.getKernel(module, "assembleGBSAHessian");
        computeLigandGBBornDerivDoubleKernel     = cu.getKernel(module, "computeLigandGBBornDerivDouble");
        prepareHessianIntermediatesDoubleKernel  = cu.getKernel(module, "prepareHessianIntermediatesDouble");
        computeHCTJacobianPairwiseDoubleKernel   = cu.getKernel(module, "computeHCTJacobianPairwiseDouble");
        computeReceptorPairwiseHessianDoubleKernel = cu.getKernel(module, "computeReceptorPairwiseHessianDouble");
        computeBornCouplingMatrixDoubleKernel    = cu.getKernel(module, "computeBornCouplingMatrixDouble");
        assembleGBSAHessianDoubleKernel          = cu.getKernel(module, "assembleGBSAHessianDouble");
        convertTiledHCTToDoubleKernel            = cu.getKernel(module, "convertTiledHCTToDouble");
        computeBornRadiiOBCDoubleKernel          = cu.getKernel(module, "computeBornRadiiOBCDouble");
        computeHctReceptorPairwiseDoubleKernel   = cu.getKernel(module, "computeHctReceptorPairwiseDouble");
        computeHctLigandPairwiseDoubleKernel     = cu.getKernel(module, "computeHctLigandPairwiseDouble");
        computeHctReceptorGridDoubleKernel       = cu.getKernel(module, "computeHctReceptorGridDouble");
        computeHCTJacobianGridDoubleKernel       = cu.getKernel(module, "computeHCTJacobianGridDouble");
        computeReceptorGridHessianDoubleKernel   = cu.getKernel(module, "computeReceptorGridHessianDouble");
        pairwiseRecBornDoubleKernel              = cu.getKernel(module, "pairwiseRecBornDouble");
        pairwiseRecCouplingDoubleKernel          = cu.getKernel(module, "pairwiseRecCouplingDouble");
        pairwiseRecJacobianDoubleKernel          = cu.getKernel(module, "pairwiseRecJacobianDouble");
        pairwiseCrossBornDeriv1DoubleKernel      = cu.getKernel(module, "pairwiseCrossBornDeriv1Double");
        pairwiseBornGradHessianDoubleKernel      = cu.getKernel(module, "pairwiseBornGradHessianDouble");
        pairwiseOuterProductHessianDoubleKernel  = cu.getKernel(module, "pairwiseOuterProductHessianDouble");
        scatterDesolvationHessianGemmKernel      = cu.getKernel(module, "scatterDesolvationHessianGemm");
        addReceptorDiagToMRDoubleKernel          = cu.getKernel(module, "addReceptorDiagToMRDouble");
        pairwiseComputePerPairScalarsDoubleKernel = cu.getKernel(module, "pairwiseComputePerPairScalarsDouble");

        hessianBuffersInitialized = true;
        hessianNumAtomsCached = totalParticles;
    }

    // Lazy: float-storage variant of the four hessian-writing kernels +
    // a float-typed dim3N*dim3N buffer. Same source as the double path,
    // recompiled with -DHBUF_T=float; kernel names are unchanged.
    if (useFloatBufPair && !hessianFloatBufModuleLoaded) {
        std::map<std::string, std::string> floatDefines;
        floatDefines["HBUF_T"] = "float";
        CUmodule floatModule = cu.createModule(
            CudaGridForceKernelSources::commonHeaders +
            CudaGridForceKernelSources::gbsaGridForceKernel +
            CudaGridForceKernelSources::isolatedGBSAKernel +
            CudaGridForceKernelSources::gbsaHessianDoubleKernel,
            floatDefines);
        assembleGBSAHessianDoubleFloatBufKernel =
            cu.getKernel(floatModule, "assembleGBSAHessianDouble");
        pairwiseCrossBornDeriv1DoubleFloatBufKernel =
            cu.getKernel(floatModule, "pairwiseCrossBornDeriv1Double");
        pairwiseBornGradHessianDoubleFloatBufKernel =
            cu.getKernel(floatModule, "pairwiseBornGradHessianDouble");
        pairwiseOuterProductHessianDoubleFloatBufKernel =
            cu.getKernel(floatModule, "pairwiseOuterProductHessianDouble");
        scatterDesolvationHessianGemmFloatBufKernel =
            cu.getKernel(floatModule, "scatterDesolvationHessianGemm");
        hessianFloatBufModuleLoaded = true;
    }
    if (useFloatBufPair && !hessianFloatBufBuffersInitialized) {
        hessianMatrixFloatBuf.initialize<float>(cu, dim3N * dim3N,
                                                  "isolatedGbsaHessFullFB");
        hessianFloatBufBuffersInitialized = true;
    }

    // PAIRWISE receptor-desolvation + cross-term Hessian working buffers.
    // n3 here is the per-group local dimension (3 * numAtoms); each group
    // is isolated/block-diagonal so the receptor Jacobian uses local rows.
    bool pairwiseHessian = (receptorMode == IsolatedGBSAForce::PAIRWISE)
                           && (numReceptorAtoms > 0);
    int n3local = 3 * numAtoms;
    if (pairwiseHessian && !hessianPairwiseBuffersInitialized) {
        int KNr = numParticleGroups * numReceptorAtoms;
        hessianRecBorn.initialize<double>(cu, KNr, "isolatedGbsaHessRecBorn");
        hessianRecDRdPsi.initialize<double>(cu, KNr, "isolatedGbsaHessRecDRdPsi");
        hessianRecD2RdPsi2.initialize<double>(cu, KNr, "isolatedGbsaHessRecD2RdPsi2");
        hessianRecDeDR.initialize<double>(cu, KNr, "isolatedGbsaHessRecDeDR");
        hessianRecJacobian.initialize<double>(cu, (size_t)KNr * n3local, "isolatedGbsaHessRecJac");
        hessianRecCoupling.initialize<double>(cu, (size_t)KNr * numReceptorAtoms, "isolatedGbsaHessRecMR");
        hessianCrossDRL.initialize<double>(cu, totalParticles, "isolatedGbsaHessCrossDRL");
        hessianCrossDRR.initialize<double>(cu, KNr, "isolatedGbsaHessCrossDRR");
        hessianPairwiseBuffersInitialized = true;
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
    CUdeviceptr hessianPtr        = useFloatBufPair
                                       ? hessianMatrixFloatBuf.getDevicePointer()
                                       : hessianMatrix.getDevicePointer();
    CUdeviceptr exclStartPtr      = hessianDummyExclStart.getDevicePointer();
    CUdeviceptr exclAtomsPtr      = hessianDummyExclAtoms.getDevicePointer();
    CUdeviceptr receptorPosPtr    = (numReceptorAtoms > 0) ? receptorPositions.getDevicePointer() : 0;
    CUdeviceptr receptorRadiiPtr  = (numReceptorAtoms > 0) ? receptorRadii.getDevicePointer() : 0;
    CUdeviceptr receptorScalesPtr = (numReceptorAtoms > 0) ? receptorScaleFactors.getDevicePointer() : 0;
    CUdeviceptr hctReceptorDoublePtr = hessianHctReceptorDouble.getDevicePointer();
    CUdeviceptr hctLigandDoublePtr   = hessianHctLigandDouble.getDevicePointer();
    CUdeviceptr bornRadiiDoublePtr   = hessianBornRadiiDouble.getDevicePointer();

    bool useGridReceptorHess = (receptorMode == IsolatedGBSAForce::GRID);
    CUdeviceptr gridCountsPtr        = 0;
    CUdeviceptr gridHctProbePtr      = 0;
    CUdeviceptr gridHctDerivPtr      = 0;
    CUdeviceptr gridCorrNPtr         = 0;
    CUdeviceptr gridCorrAPtr         = 0;
    CUdeviceptr gridCorrBPtr         = 0;
    CUdeviceptr rThresholdsGridPtr   = 0;
    if (useGridReceptorHess) {
        gridCountsPtr      = gridCounts.getDevicePointer();
        gridHctProbePtr    = gridHctProbe.getDevicePointer();
        gridHctDerivPtr    = hasHctDerivatives ? gridHctDerivatives.getDevicePointer() : 0;
        gridCorrNPtr       = gridCorrectionN.getDevicePointer();
        gridCorrAPtr       = gridCorrectionA.getDevicePointer();
        gridCorrBPtr       = gridCorrectionB.getDevicePointer();
        rThresholdsGridPtr = rThresholds.getDevicePointer();
    }

    // === Kernel 0a: receptor->ligand HCT in double.
    // GRID mode: grid interp into hctReceptorDouble.
    // PAIRWISE mode: pair-loop sum.
    if (useGridReceptorHess) {
        void* hctRecArgs[] = {
            &posqPtr, &particleIndicesPtr, &radiiPtr,
            &gridCountsPtr, &gridHctProbePtr, &gridHctDerivPtr,
            &gridCorrNPtr, &gridCorrAPtr, &gridCorrBPtr,
            &rThresholdsGridPtr, &groupStartPtr, &numParticleGroups,
            &originX, &originY, &originZ, &gridSpacing, &probeRadius,
            &numBins, &interpolationMethod,
            &useKDECorrections, &hasBinnedKDEDerivatives,
            &totalParticles, &templateN, &hctReceptorDoublePtr
        };
        cu.executeKernel(computeHctReceptorGridDoubleKernel, hctRecArgs,
                         numBlocksN * blockSize, blockSize);
    } else if (numReceptorAtoms > 0) {
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

    // In PAIRWISE mode the production dE/dR conflates the ligand-GB and
    // cross-term Born derivatives. The Hessian core (prepare + coupling +
    // assemble) needs the ligand-GB-only part; the receptor-desolvation and
    // cross-term R-kernels add the cross part separately. Recompute it here so
    // those contributions are not double-counted in the curvature and the
    // receptor self-spatial terms.
    if (pairwiseHessian) {
        float pfLig = (float)prefactor;
        CUdeviceptr ligGBdEdRPtr = hessianLigGBdEdR.getDevicePointer();
        void* ligDEdRArgs[] = {
            &posqPtr, &particleIndicesPtr, &chargesPtr, &bornRadiiDoublePtr,
            &groupStartPtr, &numParticleGroups, &templateN, &pfLig,
            &totalParticles, &ligGBdEdRPtr
        };
        cu.executeKernel(computeLigandGBBornDerivDoubleKernel, ligDEdRArgs,
                         numBlocksN * blockSize, blockSize);
        dE_dRPtr = ligGBdEdRPtr;
    }

    // === Kernel 1: prepareHessianIntermediatesDouble ===
    void* prepArgs[] = {
        &radiiPtr, &bornRadiiDoublePtr, &hctReceptorDoublePtr, &hctLigandDoublePtr,
        &dE_dRPtr, &totalParticles, &templateN,
        &dRdPsiPtr, &d2RdPsi2Ptr, &dE_dHCTPtr
    };
    cu.executeKernel(prepareHessianIntermediatesDoubleKernel, prepArgs,
                     numBlocksN * blockSize, blockSize);

    // === Kernel 2: dPsi/dx Jacobian.
    if (useGridReceptorHess) {
        void* jacArgs[] = {
            &posqPtr, &particleIndicesPtr, &radiiPtr, &scaleFactorsPtr,
            &exclAtomsPtr, &exclStartPtr, &groupStartPtr,
            &numParticleGroups, &templateN,
            &gridCountsPtr, &gridHctProbePtr, &gridHctDerivPtr,
            &gridCorrNPtr, &gridCorrAPtr, &gridCorrBPtr,
            &rThresholdsGridPtr,
            &originX, &originY, &originZ, &gridSpacing, &probeRadius,
            &numBins, &interpolationMethod,
            &useKDECorrections, &hasBinnedKDEDerivatives,
            &totalParticles, &jacobianPtr
        };
        cu.executeKernel(computeHCTJacobianGridDoubleKernel, jacArgs,
                         numBlocksN * blockSize, blockSize);
    } else {
        void* jacArgs[] = {
            &posqPtr, &particleIndicesPtr, &radiiPtr, &scaleFactorsPtr,
            &exclAtomsPtr, &exclStartPtr, &groupStartPtr,
            &numParticleGroups, &templateN,
            &receptorPosPtr, &receptorRadiiPtr, &receptorScalesPtr,
            &numReceptorAtoms, &totalParticles, &jacobianPtr
        };
        cu.executeKernel(computeHCTJacobianPairwiseDoubleKernel, jacArgs,
                         numBlocksN * blockSize, blockSize);
    }

    // === Kernel 2b: d2Psi/dx2 receptor Hessian.
    if (useGridReceptorHess) {
        void* recHessArgs[] = {
            &posqPtr, &particleIndicesPtr, &radiiPtr,
            &groupStartPtr, &numParticleGroups, &templateN,
            &gridCountsPtr, &gridHctProbePtr, &gridHctDerivPtr,
            &gridCorrNPtr, &gridCorrAPtr, &gridCorrBPtr,
            &rThresholdsGridPtr,
            &originX, &originY, &originZ, &gridSpacing, &probeRadius,
            &numBins, &interpolationMethod,
            &useKDECorrections, &hasBinnedKDEDerivatives,
            &totalParticles, &recD2PsiPtr
        };
        cu.executeKernel(computeReceptorGridHessianDoubleKernel, recHessArgs,
                         numBlocksN * blockSize, blockSize);
    } else if (numReceptorAtoms > 0) {
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
    if (useFloatBufPair) {
        cu.clearBuffer(hessianMatrixFloatBuf);
    }
    cu.executeKernel(useFloatBufPair ? assembleGBSAHessianDoubleFloatBufKernel
                                       : assembleGBSAHessianDoubleKernel,
                     assembleArgs, numBlocksH * blockSize, blockSize);

    // === PAIRWISE: add receptor-desolvation + cross-term Hessian. ===
    // The assemble kernel above wrote only the ligand-ligand ("NONE core")
    // Hessian. These kernels atomicAdd the receptor-desolvation
    // (JR^T MR JR + receptor self-spatial + receptor OBC curvature) and the
    // cross-term contributions (explicit-r, Born-gradient, Born-Born
    // couplings, single-Born curvature) on top. Mirrors
    // addPairwiseHessianContributions() in the Reference (validated to
    // ~1e-15 vs JAX autodiff). NONE mode skips this block entirely.
    if (pairwiseHessian) {
        int isHCTInt = (gbMethod == IsolatedGBSAForce::HCT) ? 1 : 0;
        int KNr = numParticleGroups * numReceptorAtoms;
        int numBlocksRec = (KNr + blockSize - 1) / blockSize;

        CUdeviceptr receptorChargesPtr = receptorCharges.getDevicePointer();
        CUdeviceptr receptorSelfHCTPtr = receptorSelfHCT.getDevicePointer();
        CUdeviceptr recBornPtr     = hessianRecBorn.getDevicePointer();
        CUdeviceptr recDRdPsiPtr   = hessianRecDRdPsi.getDevicePointer();
        CUdeviceptr recD2RdPsi2Ptr = hessianRecD2RdPsi2.getDevicePointer();
        CUdeviceptr recDeDRPtr     = hessianRecDeDR.getDevicePointer();
        CUdeviceptr recJacPtr      = hessianRecJacobian.getDevicePointer();
        CUdeviceptr recMRPtr       = hessianRecCoupling.getDevicePointer();
        CUdeviceptr crossDRLPtr    = hessianCrossDRL.getDevicePointer();
        CUdeviceptr crossDRRPtr    = hessianCrossDRR.getDevicePointer();

        // R1: receptor Born radii + transform derivatives.
        void* recBornArgs[] = {
            &posqPtr, &particleIndicesPtr, &radiiPtr, &scaleFactorsPtr,
            &groupStartPtr, &numParticleGroups, &templateN,
            &receptorPosPtr, &receptorRadiiPtr, &receptorSelfHCTPtr,
            &numReceptorAtoms, &cutoffDistance, &isHCTInt,
            &recBornPtr, &recDRdPsiPtr, &recD2RdPsi2Ptr
        };
        cu.executeKernel(pairwiseRecBornDoubleKernel, recBornArgs,
                         numBlocksRec * blockSize, blockSize);

        // R2: receptor dE/dR + coupling matrix MR.
        int hessIncludeSAInt = includeSurfaceArea ? 1 : 0;
        float hessProbeRadius = 0.14f;
        void* recCoupArgs[] = {
            &receptorPosPtr, &receptorChargesPtr, &receptorRadiiPtr,
            &recBornPtr, &recDRdPsiPtr, &recD2RdPsi2Ptr,
            &numParticleGroups, &numReceptorAtoms, &prefactorHessF,
            &hessIncludeSAInt, &surfaceTension, &hessProbeRadius,
            &recDeDRPtr, &recMRPtr
        };
        cu.executeKernel(pairwiseRecCouplingDoubleKernel, recCoupArgs,
                         numBlocksRec * blockSize, blockSize);

        // R3: receptor Jacobian JR.
        void* recJacArgs[] = {
            &posqPtr, &particleIndicesPtr, &radiiPtr, &scaleFactorsPtr,
            &groupStartPtr, &numParticleGroups, &templateN,
            &receptorPosPtr, &receptorRadiiPtr,
            &numReceptorAtoms, &cutoffDistance, &n3local, &recJacPtr
        };
        cu.executeKernel(pairwiseRecJacobianDoubleKernel, recJacArgs,
                         numBlocksRec * blockSize, blockSize);

        // R4: cross-term Born first derivatives + explicit-r cross Hessian.
        // Zero the dCrossDRR accumulator (atomicAdd target) first.
        cu.clearBuffer(hessianCrossDRR);
        void* crossD1Args[] = {
            &posqPtr, &particleIndicesPtr, &chargesPtr, &bornRadiiDoublePtr,
            &groupStartPtr, &numParticleGroups, &templateN,
            &receptorPosPtr, &receptorChargesPtr, &recBornPtr,
            &numReceptorAtoms, &prefactorHessF, &totalParticles,
            &crossDRLPtr, &crossDRRPtr, &hessianPtr
        };
        cu.executeKernel(useFloatBufPair ? pairwiseCrossBornDeriv1DoubleFloatBufKernel
                                           : pairwiseCrossBornDeriv1DoubleKernel,
                         crossD1Args, numBlocksN * blockSize, blockSize);

        // R5: Born-gradient spatial Hessians (desolv self + cross ligand/rec).
        void* bornGradArgs[] = {
            &posqPtr, &particleIndicesPtr, &radiiPtr, &scaleFactorsPtr,
            &dRdPsiPtr, &groupStartPtr, &numParticleGroups, &templateN,
            &receptorPosPtr, &receptorRadiiPtr, &receptorScalesPtr,
            &recDeDRPtr, &recDRdPsiPtr, &crossDRLPtr, &crossDRRPtr,
            &numReceptorAtoms, &totalParticles, &hessianPtr
        };
        cu.executeKernel(useFloatBufPair ? pairwiseBornGradHessianDoubleFloatBufKernel
                                           : pairwiseBornGradHessianDoubleKernel,
                         bornGradArgs, numBlocksN * blockSize, blockSize);

        // R5.5: desolvation block H_des_local = J_R^T M_R J_R per group, via
        // a cuBLAS dgemm cascade. This used to live inside the inner double
        // loop of pairwiseOuterProductHessianDouble (R6 below) and dominated
        // that kernel's runtime (99% of 66s on EA1-sized systems). Splitting
        // it out lets cuBLAS handle the GEMM at peak throughput.
        if (!cublasInitialized) {
            CUBLAS_CHECK(cublasCreate(&cublasHandle));
            CUBLAS_CHECK(cublasSetStream(cublasHandle, cu.getCurrentStream()));
            cublasInitialized = true;
        }
        if (!hessianGemmScratchInitialized ||
            hessianGemmCachedNr != numReceptorAtoms ||
            hessianGemmCachedN3 != n3local) {
            hessianGemmScratchX.initialize<double>(
                cu, (size_t)numReceptorAtoms * n3local,
                "isolatedGbsaHessGemmScratchX");
            hessianGemmScratchH.initialize<double>(
                cu, (size_t)n3local * n3local,
                "isolatedGbsaHessGemmScratchH");
            hessianGemmCachedNr = numReceptorAtoms;
            hessianGemmCachedN3 = n3local;
            hessianGemmScratchInitialized = true;
        }
        // Stage A: per-receptor diagonal weight buffer
        int KxNr = numParticleGroups * numReceptorAtoms;
        if (!hessianWjInitialized || hessianWjCachedKxNr != KxNr) {
            hessianWj.initialize<double>(cu, (size_t)KxNr, "isolatedGbsaHessWj");
            hessianWjCachedKxNr = KxNr;
            hessianWjInitialized = true;
        }
        // Stage B: per-pair scalar cache (cRi, cRiRj, gri, grj, ir*dx,
        // ir*dy, ir*dz packed as 7 sub-arrays of size K*templateN*Nr each).
        int KxNxNr = numParticleGroups * templateN * numReceptorAtoms;
        if (!hessianPairScalarsInitialized ||
            hessianPairScalarsCachedKxNxNr != KxNxNr) {
            hessianPairScalars.initialize<double>(
                cu, (size_t)7 * KxNxNr, "isolatedGbsaHessPairScalars");
            hessianPairScalarsCachedKxNxNr = KxNxNr;
            hessianPairScalarsInitialized = true;
        }
        // Stage A+B.1: ONE kernel computes both the W_j accumulator and the
        // per-(g,iL,j) scalar cache. Each per-pair quantity (cRi, cRj,
        // cRiRj, gri, grj, geom) is computed ONCE here instead of being
        // recomputed 5000+ times per (iL, j) pair across surviving Hessian-
        // entry threads in the consumer.
        cu.clearBuffer(hessianWj);
        CUdeviceptr WjPtr = hessianWj.getDevicePointer();
        CUdeviceptr pairBasePtr = hessianPairScalars.getDevicePointer();
        CUdeviceptr cRiPtr   = pairBasePtr + (CUdeviceptr)(0 * KxNxNr * sizeof(double));
        CUdeviceptr cRiRjPtr = pairBasePtr + (CUdeviceptr)(1 * KxNxNr * sizeof(double));
        CUdeviceptr griPtr   = pairBasePtr + (CUdeviceptr)(2 * KxNxNr * sizeof(double));
        CUdeviceptr grjPtr   = pairBasePtr + (CUdeviceptr)(3 * KxNxNr * sizeof(double));
        CUdeviceptr irDxPtr  = pairBasePtr + (CUdeviceptr)(4 * KxNxNr * sizeof(double));
        CUdeviceptr irDyPtr  = pairBasePtr + (CUdeviceptr)(5 * KxNxNr * sizeof(double));
        CUdeviceptr irDzPtr  = pairBasePtr + (CUdeviceptr)(6 * KxNxNr * sizeof(double));
        {
            int per_pair = templateN * numReceptorAtoms;
            int totalAccumThreads = numParticleGroups * per_pair;
            int numBlocksAccum = (totalAccumThreads + blockSize - 1) / blockSize;
            void* scalarsArgs[] = {
                &posqPtr, &particleIndicesPtr, &chargesPtr, &bornRadiiDoublePtr,
                &dRdPsiPtr,
                &groupStartPtr, &numParticleGroups, &templateN,
                &receptorPosPtr, &receptorChargesPtr,
                &recBornPtr, &recDRdPsiPtr,
                &numReceptorAtoms, &prefactorHessF,
                &cRiPtr, &cRiRjPtr, &griPtr, &grjPtr,
                &irDxPtr, &irDyPtr, &irDzPtr,
                &WjPtr
            };
            cu.executeKernel(pairwiseComputePerPairScalarsDoubleKernel, scalarsArgs,
                             numBlocksAccum * blockSize, blockSize);
        }
        // Stage A.2: M_R[g, j, j] += W_j[g, j] + dCrossDRR(g,j) * d2R^R(g,j)/dPsi^2.
        // After this, the JR^T M_R JR cuBLAS dgemm below picks up BOTH the
        // original desolvation outer product AND the receptor-Born curvature
        // (the cRj contribution previously inside the R6 inner loop AND the
        // dCrossDRR loop) — all in one pass, no per-Hessian-entry loops.
        {
            int numBlocksDiag = (KxNr + blockSize - 1) / blockSize;
            void* diagArgs[] = {
                &WjPtr, &crossDRRPtr, &recD2RdPsi2Ptr,
                &recMRPtr, &numParticleGroups, &numReceptorAtoms
            };
            cu.executeKernel(addReceptorDiagToMRDoubleKernel, diagArgs,
                             numBlocksDiag * blockSize, blockSize);
        }
        {
            int Nr = numReceptorAtoms;
            int n3 = n3local;
            double alpha = 1.0, beta = 0.0;
            double* JR_base = (double*)recJacPtr;       // [K * Nr * n3]
            double* MR_base = (double*)recMRPtr;        // [K * Nr * Nr] (Stage A diagonal added in place)
            double* X_dev   = (double*)hessianGemmScratchX.getDevicePointer();
            double* H_dev   = (double*)hessianGemmScratchH.getDevicePointer();
            CUfunction scatterKernel = useFloatBufPair
                ? scatterDesolvationHessianGemmFloatBufKernel
                : scatterDesolvationHessianGemmKernel;
            int scatterBlocks = (n3 * n3 + blockSize - 1) / blockSize;
            // Per-group: JR_r row-major (Nr, n3); MR_r row-major (Nr, Nr) symmetric.
            // Want H_r = J_r^T M_r J_r (row-major (n3, n3), symmetric).
            // cuBLAS is column-major. Treat row-major (Nr, n3) data as
            // column-major (n3, Nr) = J_r^T.
            //   Step 1: X_c (n3, Nr) = J_c (n3, Nr) @ M_c (Nr, Nr)
            //           [== (M_r @ J_r)_c since H_r symmetric]
            //   Step 2: H_c (n3, n3) = X_c (n3, Nr) @ J_c (n3, Nr)^T
            //           [op_T on the second operand]
            for (int g = 0; g < numParticleGroups; g++) {
                double* JR_g = JR_base + (size_t)g * Nr * n3;
                double* MR_g = MR_base + (size_t)g * Nr * Nr;
                // Step 1
                CUBLAS_CHECK(cublasDgemm(
                    cublasHandle,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    n3, Nr, Nr,
                    &alpha,
                    JR_g, n3,
                    MR_g, Nr,
                    &beta,
                    X_dev, n3));
                // Step 2
                CUBLAS_CHECK(cublasDgemm(
                    cublasHandle,
                    CUBLAS_OP_N, CUBLAS_OP_T,
                    n3, n3, Nr,
                    &alpha,
                    X_dev, n3,
                    JR_g, n3,
                    &beta,
                    H_dev, n3));
                // Scatter H_dev into the block-diagonal slot of hessianMatrix.
                int gs = g * numAtoms;            // group start atom
                int gs3 = 3 * gs;
                void* scatterArgs[] = {
                    &H_dev, &hessianPtr, &n3, &dim3N, &gs3
                };
                cu.executeKernel(scatterKernel, scatterArgs,
                                 scatterBlocks * blockSize, blockSize);
            }
        }

        // R6: outer-product Hessian (cross couplings + curvature ONLY; the
        // J_R^T M_R J_R desolvation block was moved to R5.5 above. The
        // per-pair scalars cRi/cRiRj/gri/grj and ir*dxyz are now read from
        // the Stage B buffers — no per-thread exp/sqrt/div in the inner loop.).
        void* outerArgs[] = {
            &posqPtr, &particleIndicesPtr, &chargesPtr,
            &bornRadiiDoublePtr, &dRdPsiPtr, &d2RdPsi2Ptr, &jacobianPtr,
            &groupStartPtr, &numParticleGroups, &templateN,
            &receptorPosPtr, &receptorChargesPtr,
            &recBornPtr, &recDRdPsiPtr, &recD2RdPsi2Ptr,
            &crossDRLPtr, &crossDRRPtr, &recJacPtr, &recMRPtr,
            &numReceptorAtoms, &n3local, &prefactorHessF,
            &totalParticles,
            &cRiPtr, &cRiRjPtr, &griPtr, &grjPtr,
            &irDxPtr, &irDyPtr, &irDzPtr,
            &hessianPtr
        };
        // The kernel now launches threads per (group, lrow, lcol) within each
        // group's block-diagonal upper-tri only — numParticleGroups * n3 * n3
        // threads, vs the legacy dim3N * dim3N (228 M with 99.99% early-
        // returning). Drops kernel-launch overhead by ~4 orders of magnitude.
        int numOuterThreads = numParticleGroups * n3local * n3local;
        int numBlocksOuter = (numOuterThreads + blockSize - 1) / blockSize;
        cu.executeKernel(useFloatBufPair ? pairwiseOuterProductHessianDoubleFloatBufKernel
                                           : pairwiseOuterProductHessianDoubleKernel,
                         outerArgs, numBlocksOuter * blockSize, blockSize);
    }

    if (useFloatBufPair) {
        hessianFloatBufHost.resize(dim3N * dim3N);
        hessianMatrixFloatBuf.download(hessianFloatBufHost);
        hessianFullHost.assign(hessianFloatBufHost.begin(),
                               hessianFloatBufHost.end());
        return hessianFullHost;
    }
    hessianFullHost.resize(dim3N * dim3N);
    hessianMatrix.download(hessianFullHost);
    return hessianFullHost;
}

// Float-precision Hessian for receptorMode == GRID. Mirrors
// GBSAGridForce::computeHessian: prepare -> HCT Jacobian (grid) ->
// receptor grid Hessian -> Born coupling -> assemble. The float kernels
// and grid-state device pointers (gridCounts, gridHctProbe, ...) are
// shared with the energy/force GRID path.
vector<double> CudaCalcIsolatedGBSAForceKernel::computeHessianGridFloat(ContextImpl& context) {
    cu.setAsCurrent();

    int totalParticles = numParticleGroups * numAtoms;
    int templateN = numAtoms;
    int dim3N = 3 * totalParticles;
    if (totalParticles == 0) {
        return std::vector<double>();
    }

    if (!bornRadii.isInitialized() || !dE_dR.isInitialized() ||
        !hctReceptor.isInitialized() || !hctLigand.isInitialized()) {
        throw OpenMMException(
            "IsolatedGBSAForce: computeHessian() requires a prior "
            "getState(getForces=True) to populate Born radii and dE/dR.");
    }

    if (!hessianGridFloatBuffersInitialized) {
        hessianDRdPsiF.initialize<float>(cu, totalParticles, "isolatedGbsaHessDRdPsiF");
        hessianD2RdPsi2F.initialize<float>(cu, totalParticles, "isolatedGbsaHessD2RdPsi2F");
        hessianDEdHCTF.initialize<float>(cu, totalParticles, "isolatedGbsaHessDEdHCTF");
        hessianJacobianF.initialize<float>(cu, (size_t)totalParticles * dim3N,
                                            "isolatedGbsaHessJacobianF");
        hessianGridHCTHessian.initialize<float>(cu, totalParticles * 6,
                                                 "isolatedGbsaHessGridHCTHessian");
        hessianCouplingMatrixF.initialize<float>(cu, (size_t)totalParticles * totalParticles,
                                                  "isolatedGbsaHessCouplingF");
        initMixedEnergyBuffer(cu, hessianMatrixF, dim3N * dim3N, "isolatedGbsaHessMatrixF");
        hessianGridFloatBuffersInitialized = true;
    }

    int blockSize = 256;
    int numBlocksN = (totalParticles + blockSize - 1) / blockSize;

    CUdeviceptr posqPtr             = cu.getPosq().getDevicePointer();
    CUdeviceptr particleIndicesPtr  = particleIndices.getDevicePointer();
    CUdeviceptr radiiPtr            = radii.getDevicePointer();
    CUdeviceptr scaleFactorsPtr     = scaleFactors.getDevicePointer();
    CUdeviceptr chargesPtr          = charges.getDevicePointer();
    CUdeviceptr bornRadiiPtr        = bornRadii.getDevicePointer();
    CUdeviceptr hctReceptorPtr      = hctReceptor.getDevicePointer();
    CUdeviceptr hctLigandPtr        = hctLigand.getDevicePointer();
    CUdeviceptr dE_dRPtr            = dE_dR.getDevicePointer();
    CUdeviceptr groupStartPtr       = groupStartIndex.getDevicePointer();
    CUdeviceptr exclStartPtr        = hessianDummyExclStart.getDevicePointer();
    CUdeviceptr exclAtomsPtr        = hessianDummyExclAtoms.getDevicePointer();
    CUdeviceptr gridCountsPtr       = gridCounts.getDevicePointer();
    CUdeviceptr gridHctProbePtr     = gridHctProbe.getDevicePointer();
    CUdeviceptr gridHctDerivPtr     = hasHctDerivatives ? gridHctDerivatives.getDevicePointer() : 0;
    CUdeviceptr gridCorrNPtr        = gridCorrectionN.getDevicePointer();
    CUdeviceptr gridCorrAPtr        = gridCorrectionA.getDevicePointer();
    CUdeviceptr gridCorrBPtr        = gridCorrectionB.getDevicePointer();
    CUdeviceptr rThresholdsPtr      = rThresholds.getDevicePointer();
    CUdeviceptr dRdPsiPtr           = hessianDRdPsiF.getDevicePointer();
    CUdeviceptr d2RdPsi2Ptr         = hessianD2RdPsi2F.getDevicePointer();
    CUdeviceptr dE_dHCTPtr          = hessianDEdHCTF.getDevicePointer();
    CUdeviceptr jacobianPtr         = hessianJacobianF.getDevicePointer();
    CUdeviceptr gridHCTHessianPtr   = hessianGridHCTHessian.getDevicePointer();
    CUdeviceptr couplingPtr         = hessianCouplingMatrixF.getDevicePointer();
    CUdeviceptr hessianPtr          = hessianMatrixF.getDevicePointer();

    void* prepArgs[] = {
        &radiiPtr, &bornRadiiPtr, &hctReceptorPtr, &hctLigandPtr,
        &dE_dRPtr, &totalParticles, &templateN,
        &dRdPsiPtr, &d2RdPsi2Ptr, &dE_dHCTPtr
    };
    cu.executeKernel(prepareHessianIntermediatesKernel, prepArgs,
                     numBlocksN * blockSize, blockSize);

    void* jacArgs[] = {
        &posqPtr, &particleIndicesPtr, &radiiPtr, &scaleFactorsPtr,
        &exclAtomsPtr, &exclStartPtr, &groupStartPtr,
        &numParticleGroups, &templateN,
        &gridCountsPtr, &gridHctProbePtr, &gridHctDerivPtr,
        &gridCorrNPtr, &gridCorrAPtr, &gridCorrBPtr, &rThresholdsPtr,
        &originX, &originY, &originZ, &gridSpacing, &probeRadius,
        &numBins, &interpolationMethod,
        &useKDECorrections, &hasBinnedKDEDerivatives,
        &totalParticles, &jacobianPtr
    };
    cu.executeKernel(computeHCTJacobianGridKernel, jacArgs,
                     numBlocksN * blockSize, blockSize);

    void* gridHessArgs[] = {
        &posqPtr, &particleIndicesPtr, &radiiPtr,
        &groupStartPtr, &numParticleGroups, &templateN,
        &gridCountsPtr, &gridHctProbePtr, &gridHctDerivPtr,
        &gridCorrNPtr, &gridCorrAPtr, &gridCorrBPtr, &rThresholdsPtr,
        &originX, &originY, &originZ, &gridSpacing, &probeRadius,
        &numBins, &interpolationMethod,
        &useKDECorrections, &hasBinnedKDEDerivatives,
        &totalParticles, &gridHCTHessianPtr
    };
    cu.executeKernel(computeReceptorGridHessianKernel, gridHessArgs,
                     numBlocksN * blockSize, blockSize);

    int includeSAInt = includeSurfaceArea ? 1 : 0;
    float prefactorF = (float)prefactor;
    void* couplingArgs[] = {
        &posqPtr, &particleIndicesPtr, &chargesPtr, &radiiPtr,
        &bornRadiiPtr, &dRdPsiPtr, &d2RdPsi2Ptr, &dE_dRPtr,
        &exclAtomsPtr, &exclStartPtr, &groupStartPtr,
        &numParticleGroups, &templateN, &prefactorF,
        &includeSAInt, &surfaceTension, &probeRadius,
        &totalParticles, &couplingPtr
    };
    cu.executeKernel(computeBornCouplingMatrixKernel, couplingArgs,
                     numBlocksN * blockSize, blockSize);

    int totalElements = dim3N * dim3N;
    int numBlocksH = (totalElements + blockSize - 1) / blockSize;
    void* assembleArgs[] = {
        &posqPtr, &particleIndicesPtr, &chargesPtr, &bornRadiiPtr,
        &exclAtomsPtr, &exclStartPtr, &groupStartPtr,
        &numParticleGroups, &templateN, &prefactorF,
        &jacobianPtr, &couplingPtr, &dE_dHCTPtr, &scaleFactorsPtr, &radiiPtr,
        &dRdPsiPtr, &gridHCTHessianPtr,
        &totalParticles, &hessianPtr
    };
    cu.executeKernel(assembleGBSAHessianKernel, assembleArgs,
                     numBlocksH * blockSize, blockSize);

    hessianFullHost.resize(dim3N * dim3N);
    downloadMixedEnergy(cu, hessianMatrixF, hessianFullHost);
    return hessianFullHost;
}
