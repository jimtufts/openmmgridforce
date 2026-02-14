/* -------------------------------------------------------------------------- *
 *                              OpenMMGridForce                               *
 * -------------------------------------------------------------------------- */

#include "CudaGBSAGridForceKernels.h"
#include "CudaGridForceKernelSources.h"
#include "BSplinePrefilter.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/cuda/CudaBondedUtilities.h"
#include "openmm/cuda/CudaForceInfo.h"
#include "openmm/OpenMMException.h"
#include <cmath>
#include <iostream>
#include <fstream>

using namespace GridForcePlugin;
using namespace OpenMM;
using namespace std;

// Coulomb constant in kJ*nm/mol/e^2
static const float ONE_4PI_EPS0 = 138.935456f;

CudaCalcGBSAGridForceKernel::CudaCalcGBSAGridForceKernel(string name, const Platform& platform,
                                                           CudaContext& cu)
    : CalcGBSAGridForceKernel(name, platform), cu(cu), hasInitializedKernel(false),
      numAtoms(0), numParticleGroups(0), originX(0), originY(0), originZ(0),
      gridSpacing(0), probeRadius(0), numBins(0), prefactor(0),
      includeSurfaceArea(false), surfaceTension(0), globalScalingFactor(1.0f), interpolationMethod(0),
      hasHctDerivatives(false), useKDECorrections(false), hasBinnedKDEDerivatives(false),
      computeReceptorHCTKernel(nullptr), computeLigandHCTKernel(nullptr),
      computeBornRadiiKernel(nullptr), computeGBEnergyKernel(nullptr),
      computeSAEnergyKernel(nullptr),
      accumulateSADerivativesKernel(nullptr),
      accumulateBornRadiiDerivativesKernel(nullptr),
      computeHCTChainRuleForcesKernel(nullptr),
      computeReceptorHCTGradientForceKernel(nullptr),
      generateLigandHCTGridKernel(nullptr),
      generateLigandHCTGridWithCorrectionsKernel(nullptr),
      generateLigandHCTGridWithDerivativesKernel(nullptr),
      generateBinnedGridsWithKDEKernel(nullptr),
      generateBinnedGridsWithKDEDerivativesKernel(nullptr),
      generationModule(nullptr),
      hessianNumAtoms(0),
      hessianBuffersInitialized(false),
      prepareHessianIntermediatesKernel(nullptr),
      computeHCTJacobianKernel(nullptr),
      computeBornCouplingMatrixKernel(nullptr),
      assembleGBSAHessianKernel(nullptr) {
}

CudaCalcGBSAGridForceKernel::~CudaCalcGBSAGridForceKernel() {
}

void CudaCalcGBSAGridForceKernel::initialize(const System& system, const GBSAGridForce& force) {
    cu.setAsCurrent();
    numAtoms = force.getNumAtoms();
    numParticleGroups = force.getNumParticleGroups();

    if (numAtoms == 0) {
        throw OpenMMException("GBSAGridForce: no atoms defined");
    }

    auto grid = force.getDesolvationGrid();

    // Auto-generate grid if enabled and no grid is set
    if (force.getAutoGenerateGrid() && !grid) {
        // Validate generation parameters
        int nx, ny, nz;
        force.getGridCounts(nx, ny, nz);
        if (nx <= 0 || ny <= 0 || nz <= 0) {
            throw OpenMMException("GBSAGridForce: grid counts must be set for auto-generation");
        }

        const auto& recPos = force.getReceptorPositions();
        const auto& recRadii = force.getReceptorRadii();
        const auto& recScales = force.getReceptorScaleFactors();
        int numRecAtoms = static_cast<int>(recRadii.size());

        if (recPos.size() != static_cast<size_t>(numRecAtoms * 3)) {
            throw OpenMMException("GBSAGridForce: receptor positions size must be 3 * numReceptorAtoms");
        }
        if (recScales.size() != static_cast<size_t>(numRecAtoms)) {
            throw OpenMMException("GBSAGridForce: receptor scale factors size must match numReceptorAtoms");
        }

        double ox, oy, oz;
        force.getGridOrigin(ox, oy, oz);
        double origin[3] = {ox, oy, oz};
        int counts[3] = {nx, ny, nz};

        // Extract KDE parameters before grid generation
        kdeThreshold = static_cast<float>(force.getKDEThreshold());
        kdeBandwidth = static_cast<float>(force.getKDEBandwidth());
        kdeEpsilonB = static_cast<float>(force.getKDEEpsilonB());

        // Generate grid on GPU
        vector<float> hctProbe, corrN, corrA, corrB, derivatives;
        generateGrid(recPos, recRadii, recScales, numRecAtoms,
                     force.getProbeRadius(), force.getRThresholds(),
                     origin, counts, force.getGridSpacing(),
                     force.getComputeGridDerivatives(),
                     hctProbe, corrN, corrA, corrB, derivatives);

        // Create DesolvationGrid from generated data
        const auto& thresholds = force.getRThresholds();
        bool hasDerivs = force.getComputeGridDerivatives() && !derivatives.empty();
        grid = make_shared<DesolvationGrid>(nx, ny, nz,
                                            force.getGridSpacing(),
                                            force.getProbeRadius(),
                                            vector<double>(thresholds.begin(), thresholds.end()),
                                            hasDerivs);
        grid->setOrigin(ox, oy, oz);

        // Apply B-spline prefilter at generation time if requested
        int bsplineOrder = force.getBSplinePrefilterOrder();
        if (bsplineOrder > 0 && !hasDerivs) {
            bsplinePrefilter3DByOrder(hctProbe, nx, ny, nz, bsplineOrder);
            // Prefilter each bin of each correction grid independently
            int numBinsLocal = static_cast<int>(force.getRThresholds().size());
            if (numBinsLocal > 0 && corrN.size() == static_cast<size_t>(numBinsLocal) * nx * ny * nz) {
                int numPts = nx * ny * nz;
                for (int bin = 0; bin < numBinsLocal; bin++) {
                    // Extract bin slice, prefilter, copy back
                    vector<float> binSlice(corrN.begin() + bin * numPts,
                                           corrN.begin() + (bin + 1) * numPts);
                    bsplinePrefilter3DByOrder(binSlice, nx, ny, nz, bsplineOrder);
                    copy(binSlice.begin(), binSlice.end(), corrN.begin() + bin * numPts);

                    binSlice.assign(corrA.begin() + bin * numPts,
                                    corrA.begin() + (bin + 1) * numPts);
                    bsplinePrefilter3DByOrder(binSlice, nx, ny, nz, bsplineOrder);
                    copy(binSlice.begin(), binSlice.end(), corrA.begin() + bin * numPts);

                    binSlice.assign(corrB.begin() + bin * numPts,
                                    corrB.begin() + (bin + 1) * numPts);
                    bsplinePrefilter3DByOrder(binSlice, nx, ny, nz, bsplineOrder);
                    copy(binSlice.begin(), binSlice.end(), corrB.begin() + bin * numPts);
                }
            }
        }

        // Set grid data (after prefiltering so prefiltered coefficients are uploaded)
        if (hasDerivs) {
            // When derivatives are present, setHctProbe expects the full 27*n_points array
            grid->setHctProbe(derivatives);
        } else {
            grid->setHctProbe(hctProbe);
        }
        grid->setCorrectionN(corrN);
        grid->setCorrectionA(corrA);
        grid->setCorrectionB(corrB);

        // Store grid on force (const_cast needed since force is const)
        const_cast<GBSAGridForce&>(force).setDesolvationGrid(grid);
    }

    if (!grid) {
        throw OpenMMException("GBSAGridForce: no desolvation grid set");
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

    // Compute GB prefactor: -138.935456 * (1/ε_solute - 1/ε_solvent)
    double soluteDielectric = force.getSoluteDielectric();
    double solventDielectric = force.getSolventDielectric();
    prefactor = static_cast<float>(-ONE_4PI_EPS0 * (1.0/soluteDielectric - 1.0/solventDielectric));

    includeSurfaceArea = force.getIncludeSurfaceArea();
    surfaceTension = static_cast<float>(force.getSurfaceTension());
    interpolationMethod = force.getInterpolationMethod();

    // KDE smoothing parameters for grid generation
    kdeThreshold = static_cast<float>(force.getKDEThreshold());
    kdeBandwidth = static_cast<float>(force.getKDEBandwidth());
    kdeEpsilonB = static_cast<float>(force.getKDEEpsilonB());

    // Upload grid dimensions
    int nx, ny, nz;
    grid->getCounts(nx, ny, nz);
    vector<int> counts = {nx, ny, nz};
    gridCounts.initialize<int>(cu, 3, "gbsaGridCounts");
    gridCounts.upload(counts);

    // Upload grid data
    int numPoints = nx * ny * nz;

    // Handle HCT grid with optional derivatives
    const auto& hctData = grid->getHctProbe();
    hasHctDerivatives = grid->hasDerivatives();

    if (hasHctDerivatives) {
        // Derivatives are stored in derivative-major layout: [deriv_idx * n_points + point_idx]
        // The full RASPA3 array has 27 derivatives per point
        int numDerivs = grid->getNumDerivsPerPoint();  // 27

        // Validate data size
        size_t expectedSize = static_cast<size_t>(numDerivs) * numPoints;
        if (hctData.size() != expectedSize) {
            throw OpenMMException("GBSAGridForce: HCT derivative data size mismatch. Expected " +
                std::to_string(expectedSize) + " but got " + std::to_string(hctData.size()));
        }

        // Upload function values (first n_points of RASPA3 array) to gridHctProbe
        // This is used for trilinear fallback when derivatives aren't available
        vector<float> hctValues(hctData.begin(), hctData.begin() + numPoints);
        gridHctProbe.initialize<float>(cu, numPoints, "gbsaGridHctProbe");
        gridHctProbe.upload(hctValues);

        // Upload full derivative array to gridHctDerivatives
        gridHctDerivatives.initialize<float>(cu, hctData.size(), "gbsaGridHctDerivatives");
        gridHctDerivatives.upload(hctData);
    } else {
        // No derivatives - just upload function values
        if (hctData.size() != static_cast<size_t>(numPoints)) {
            throw OpenMMException("GBSAGridForce: HCT probe data size mismatch. Expected " +
                std::to_string(numPoints) + " but got " + std::to_string(hctData.size()));
        }
        gridHctProbe.initialize<float>(cu, numPoints, "gbsaGridHctProbe");
        gridHctProbe.upload(hctData);
    }

    // Upload correction grids - detect mode by size
    // Pure KDE mode: [27 * numPoints] - not binned, deprecated
    // Binned mode: [numBins * numPoints] for trilinear lookup by radius bin
    // Binned+KDE derivatives mode: [numBins * 27 * numPoints] for tricubic/triquintic per bin
    const auto& corrNData = grid->getCorrectionN();
    size_t expectedKDESize = static_cast<size_t>(27) * numPoints;
    size_t expectedBinnedSize = static_cast<size_t>(numBins) * numPoints;
    size_t expectedBinnedKDESize = static_cast<size_t>(numBins) * 27 * numPoints;

    if (corrNData.size() == expectedBinnedKDESize) {
        // Binned+KDE derivatives mode - correction grids have 27 derivatives per bin
        // Layout: [bin * 27 * numPoints + deriv * numPoints + point]
        useKDECorrections = true;  // Use high-order interpolation for corrections
        hasBinnedKDEDerivatives = true;  // Binned format with derivatives
        gridCorrectionN.initialize<float>(cu, corrNData.size(), "gbsaGridCorrectionN");
        gridCorrectionN.upload(corrNData);

        gridCorrectionA.initialize<float>(cu, grid->getCorrectionA().size(), "gbsaGridCorrectionA");
        gridCorrectionA.upload(grid->getCorrectionA());

        gridCorrectionB.initialize<float>(cu, grid->getCorrectionB().size(), "gbsaGridCorrectionB");
        gridCorrectionB.upload(grid->getCorrectionB());
    } else if (corrNData.size() == expectedKDESize) {
        // Pure KDE mode (deprecated) - correction grids have 27 derivatives, no bins
        useKDECorrections = true;
        hasBinnedKDEDerivatives = false;
        gridCorrectionN.initialize<float>(cu, corrNData.size(), "gbsaGridCorrectionN");
        gridCorrectionN.upload(corrNData);

        gridCorrectionA.initialize<float>(cu, grid->getCorrectionA().size(), "gbsaGridCorrectionA");
        gridCorrectionA.upload(grid->getCorrectionA());

        gridCorrectionB.initialize<float>(cu, grid->getCorrectionB().size(), "gbsaGridCorrectionB");
        gridCorrectionB.upload(grid->getCorrectionB());
    } else if (corrNData.size() == expectedBinnedSize || numBins == 0) {
        // Binned mode - correction grids indexed by radius bin (trilinear only)
        useKDECorrections = false;
        hasBinnedKDEDerivatives = false;
        int corrSize = numBins * numPoints;
        if (corrSize == 0) corrSize = numPoints;  // Handle single-bin case
        gridCorrectionN.initialize<float>(cu, corrSize, "gbsaGridCorrectionN");
        gridCorrectionN.upload(corrNData);

        gridCorrectionA.initialize<float>(cu, corrSize, "gbsaGridCorrectionA");
        gridCorrectionA.upload(grid->getCorrectionA());

        gridCorrectionB.initialize<float>(cu, corrSize, "gbsaGridCorrectionB");
        gridCorrectionB.upload(grid->getCorrectionB());
    } else {
        throw OpenMMException("GBSAGridForce: correction grid size mismatch. Expected " +
            std::to_string(expectedBinnedKDESize) + " (binned+KDE) or " +
            std::to_string(expectedKDESize) + " (KDE) or " +
            std::to_string(expectedBinnedSize) + " (binned) but got " +
            std::to_string(corrNData.size()));
    }

    // Upload R thresholds
    const auto& thresholds = grid->getRThresholds();
    vector<float> thresholdsFloat(thresholds.begin(), thresholds.end());
    rThresholds.initialize<float>(cu, numBins, "gbsaRThresholds");
    rThresholds.upload(thresholdsFloat);

    // Upload atom parameters
    vector<float> chargesVec(numAtoms), radiiVec(numAtoms), scalesVec(numAtoms);
    for (int i = 0; i < numAtoms; i++) {
        double q, r, s;
        force.getAtomParameters(i, q, r, s);
        chargesVec[i] = static_cast<float>(q);
        radiiVec[i] = static_cast<float>(r);
        scalesVec[i] = static_cast<float>(s);
    }
    charges.initialize<float>(cu, numAtoms, "gbsaCharges");
    charges.upload(chargesVec);
    radii.initialize<float>(cu, numAtoms, "gbsaRadii");
    radii.upload(radiiVec);
    scaleFactors.initialize<float>(cu, numAtoms, "gbsaScaleFactors");
    scaleFactors.upload(scalesVec);

    // Process exclusions into CSR format
    int numExclusions = force.getNumExclusions();
    vector<vector<int>> exclusionLists(numAtoms);
    for (int i = 0; i < numExclusions; i++) {
        int a1, a2;
        force.getExclusionParticles(i, a1, a2);
        exclusionLists[a1].push_back(a2);
        exclusionLists[a2].push_back(a1);
    }

    vector<int> exclusionAtomsVec;
    vector<int> exclusionStartVec(numAtoms + 1);
    exclusionStartVec[0] = 0;
    for (int i = 0; i < numAtoms; i++) {
        for (int j : exclusionLists[i]) {
            exclusionAtomsVec.push_back(j);
        }
        exclusionStartVec[i + 1] = static_cast<int>(exclusionAtomsVec.size());
    }

    if (exclusionAtomsVec.empty()) {
        exclusionAtomsVec.push_back(0);  // Dummy to avoid empty buffer
    }
    exclusionAtoms.initialize<int>(cu, exclusionAtomsVec.size(), "gbsaExclusionAtoms");
    exclusionAtoms.upload(exclusionAtomsVec);
    exclusionStartIndex.initialize<int>(cu, numAtoms + 1, "gbsaExclusionStart");
    exclusionStartIndex.upload(exclusionStartVec);

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
        particleIndices.initialize<int>(cu, totalParticles, "gbsaParticleIndices");
        particleIndices.upload(allIndices);
        groupStartIndex.initialize<int>(cu, numParticleGroups + 1, "gbsaGroupStart");
        groupStartIndex.upload(groupStarts);
        groupEnergies.initialize<float>(cu, numParticleGroups, "gbsaGroupEnergies");
        groupLigandEnergies.initialize<float>(cu, numParticleGroups, "gbsaGroupLigandEnergies");

        groupEnergiesHost.resize(numParticleGroups);
        groupLigandEnergiesHost.resize(numParticleGroups);
        groupBornRadiiHost.resize(numParticleGroups);
    } else {
        // Legacy mode: use particles from force or all atoms
        const auto& particles = force.getParticles();
        if (!particles.empty()) {
            totalParticles = static_cast<int>(particles.size());
            particleIndices.initialize<int>(cu, totalParticles, "gbsaParticleIndices");
            particleIndices.upload(particles);
        } else {
            // All atoms
            totalParticles = numAtoms;
            vector<int> allIndices(numAtoms);
            for (int i = 0; i < numAtoms; i++) allIndices[i] = i;
            particleIndices.initialize<int>(cu, totalParticles, "gbsaParticleIndices");
            particleIndices.upload(allIndices);
        }

        // Single group containing all particles
        numParticleGroups = 1;
        vector<int> groupStarts = {0, totalParticles};
        groupStartIndex.initialize<int>(cu, 2, "gbsaGroupStart");
        groupStartIndex.upload(groupStarts);
        groupEnergies.initialize<float>(cu, 1, "gbsaGroupEnergies");
        groupLigandEnergies.initialize<float>(cu, 1, "gbsaGroupLigandEnergies");
        groupEnergiesHost.resize(1);
        groupLigandEnergiesHost.resize(1);
        groupBornRadiiHost.resize(1);
    }

    // Initialize alchemical scaling
    globalScalingFactor = static_cast<float>(force.getGlobalScalingFactor());
    {
        int nGroups = force.getNumParticleGroups();
        vector<float> groupScalings(numParticleGroups, 1.0f);
        for (int g = 0; g < nGroups; g++) {
            groupScalings[g] = static_cast<float>(force.getGroupScalingFactor(g));
        }
        groupScalingFactorsBuffer.initialize<float>(cu, numParticleGroups, "gbsaGroupScalingFactors");
        groupScalingFactorsBuffer.upload(groupScalings);
    }

    // Allocate intermediate result buffers
    if (totalParticles > 0) {
        hctReceptor.initialize<float>(cu, totalParticles, "gbsaHctReceptor");
        hctLigand.initialize<float>(cu, totalParticles, "gbsaHctLigand");
        bornRadii.initialize<float>(cu, totalParticles, "gbsaBornRadii");
        dE_dR.initialize<float>(cu, totalParticles, "gbsaDEdR");
        dE_dHCT.initialize<float>(cu, totalParticles, "gbsaDEdHCT");
    }

    // Compile CUDA kernels
    CUmodule module = cu.createModule(CudaGridForceKernelSources::gridForceKernel);
    computeReceptorHCTKernel = cu.getKernel(module, "computeReceptorHCT");
    computeLigandHCTKernel = cu.getKernel(module, "computeLigandHCT");
    computeBornRadiiKernel = cu.getKernel(module, "computeBornRadii");
    computeGBEnergyKernel = cu.getKernel(module, "computeGBEnergy");
    computeSAEnergyKernel = cu.getKernel(module, "computeSAEnergy");
    accumulateSADerivativesKernel = cu.getKernel(module, "accumulateSADerivatives");
    accumulateBornRadiiDerivativesKernel = cu.getKernel(module, "accumulateBornRadiiDerivatives");
    computeHCTChainRuleForcesKernel = cu.getKernel(module, "computeHCTChainRuleForces");
    computeReceptorHCTGradientForceKernel = cu.getKernel(module, "computeReceptorHCTGradientForce");

    // Analytical Hessian kernels
    prepareHessianIntermediatesKernel = cu.getKernel(module, "prepareHessianIntermediates");
    computeHCTJacobianKernel = cu.getKernel(module, "computeHCTJacobian");
    computeReceptorGridHessianKernel = cu.getKernel(module, "computeReceptorGridHessian");
    computeBornCouplingMatrixKernel = cu.getKernel(module, "computeBornCouplingMatrix");
    assembleGBSAHessianKernel = cu.getKernel(module, "assembleGBSAHessian");

    hasInitializedKernel = true;
}

double CudaCalcGBSAGridForceKernel::execute(ContextImpl& context,
                                            bool includeForces,
                                            bool includeEnergy) {
    if (!hasInitializedKernel) {
        throw OpenMMException("GBSAGridForce kernel not initialized");
    }

    int totalParticles = particleIndices.getSize();
    if (totalParticles == 0) return 0.0;

    int paddedNumAtoms = cu.getPaddedNumAtoms();

    // Clear group energies (async GPU clear to avoid pipeline stalls during integration)
    if (includeEnergy) {
        cu.clearBuffer(groupEnergies);
        cu.clearBuffer(groupLigandEnergies);
    }

    // Get device pointers
    CUdeviceptr posqPtr = cu.getPosq().getDevicePointer();
    CUdeviceptr forcePtr = cu.getLongForceBuffer().getDevicePointer();
    CUdeviceptr particleIndicesPtr = particleIndices.getDevicePointer();
    CUdeviceptr radiiPtr = radii.getDevicePointer();
    CUdeviceptr scaleFactorsPtr = scaleFactors.getDevicePointer();
    CUdeviceptr chargesPtr = charges.getDevicePointer();
    CUdeviceptr gridCountsPtr = gridCounts.getDevicePointer();
    CUdeviceptr gridHctProbePtr = gridHctProbe.getDevicePointer();
    CUdeviceptr gridHctDerivativesPtr = hasHctDerivatives ? gridHctDerivatives.getDevicePointer() : 0;
    CUdeviceptr gridCorrectionNPtr = gridCorrectionN.getDevicePointer();
    CUdeviceptr gridCorrectionAPtr = gridCorrectionA.getDevicePointer();
    CUdeviceptr gridCorrectionBPtr = gridCorrectionB.getDevicePointer();
    CUdeviceptr rThresholdsPtr = rThresholds.getDevicePointer();
    CUdeviceptr exclusionAtomsPtr = exclusionAtoms.getDevicePointer();
    CUdeviceptr exclusionStartPtr = exclusionStartIndex.getDevicePointer();
    CUdeviceptr groupStartPtr = groupStartIndex.getDevicePointer();
    CUdeviceptr hctReceptorPtr = hctReceptor.getDevicePointer();
    CUdeviceptr hctLigandPtr = hctLigand.getDevicePointer();
    CUdeviceptr bornRadiiPtr = bornRadii.getDevicePointer();
    CUdeviceptr groupEnergiesPtr = groupEnergies.getDevicePointer();

    int blockSize = 256;
    int numBlocks = (totalParticles + blockSize - 1) / blockSize;

    // Step 1: Compute receptor HCT via grid interpolation
    void* receptorArgs[] = {
        &posqPtr, &particleIndicesPtr, &radiiPtr,
        &gridCountsPtr, &gridHctProbePtr, &gridHctDerivativesPtr,
        &gridCorrectionNPtr, &gridCorrectionAPtr, &gridCorrectionBPtr,
        &rThresholdsPtr, &groupStartPtr, &numParticleGroups,
        &originX, &originY, &originZ, &gridSpacing, &probeRadius,
        &numBins, &totalParticles, &numAtoms, &interpolationMethod,
        &useKDECorrections, &hasBinnedKDEDerivatives, &hctReceptorPtr
    };
    cu.executeKernel(computeReceptorHCTKernel, receptorArgs, numBlocks * blockSize, blockSize);

    // Step 2: Compute ligand-ligand HCT
    void* ligandArgs[] = {
        &posqPtr, &particleIndicesPtr, &radiiPtr, &scaleFactorsPtr,
        &exclusionAtomsPtr, &exclusionStartPtr, &groupStartPtr,
        &numParticleGroups, &numAtoms, &hctLigandPtr
    };
    cu.executeKernel(computeLigandHCTKernel, ligandArgs, numBlocks * blockSize, blockSize);

    // Step 3: Compute Born radii
    void* bornArgs[] = {
        &radiiPtr, &hctReceptorPtr, &hctLigandPtr,
        &totalParticles, &numAtoms, &bornRadiiPtr
    };
    cu.executeKernel(computeBornRadiiKernel, bornArgs, numBlocks * blockSize, blockSize);

    // Step 4: Compute GB energy and forces
    CUdeviceptr groupScalingFactorsPtr = groupScalingFactorsBuffer.getDevicePointer();
    void* energyArgs[] = {
        &posqPtr, &particleIndicesPtr, &chargesPtr, &bornRadiiPtr,
        &exclusionAtomsPtr, &exclusionStartPtr, &groupStartPtr,
        &numParticleGroups, &numAtoms, &prefactor,
        &forcePtr, &groupEnergiesPtr, &paddedNumAtoms,
        &globalScalingFactor, &groupScalingFactorsPtr
    };
    cu.executeKernel(computeGBEnergyKernel, energyArgs, numBlocks * blockSize, blockSize);

    // Step 5: Optional surface area term
    if (includeSurfaceArea) {
        void* saArgs[] = {
            &radiiPtr, &bornRadiiPtr, &groupStartPtr,
            &numParticleGroups, &numAtoms,
            &surfaceTension, &probeRadius, &groupEnergiesPtr,
            &globalScalingFactor, &groupScalingFactorsPtr
        };
        cu.executeKernel(computeSAEnergyKernel, saArgs, numBlocks * blockSize, blockSize);
    }

    // Step 6: Chain rule forces through Born radii
    if (includeForces) {
        CUdeviceptr dE_dRPtr = dE_dR.getDevicePointer();
        CUdeviceptr dE_dHCTPtr = dE_dHCT.getDevicePointer();

        // Step 6a: Accumulate dE/dR_born for each atom (GB energy contribution)
        void* bornDerivArgs[] = {
            &posqPtr, &particleIndicesPtr, &chargesPtr, &bornRadiiPtr,
            &exclusionAtomsPtr, &exclusionStartPtr, &groupStartPtr,
            &numParticleGroups, &numAtoms, &prefactor, &dE_dRPtr,
            &globalScalingFactor, &groupScalingFactorsPtr
        };
        cu.executeKernel(accumulateBornRadiiDerivativesKernel, bornDerivArgs, numBlocks * blockSize, blockSize);

        // Step 6a2: Add surface area contribution to dE/dR_born if enabled
        if (includeSurfaceArea) {
            void* saDerivArgs[] = {
                &radiiPtr, &bornRadiiPtr, &groupStartPtr,
                &numParticleGroups, &numAtoms, &surfaceTension, &probeRadius, &dE_dRPtr,
                &globalScalingFactor, &groupScalingFactorsPtr
            };
            cu.executeKernel(accumulateSADerivativesKernel, saDerivArgs, numBlocks * blockSize, blockSize);
        }

        // Step 6b: Compute ligand-ligand HCT chain rule forces
        void* hctChainArgs[] = {
            &posqPtr, &particleIndicesPtr, &radiiPtr, &scaleFactorsPtr,
            &bornRadiiPtr, &hctReceptorPtr, &hctLigandPtr, &dE_dRPtr,
            &exclusionAtomsPtr, &exclusionStartPtr, &groupStartPtr,
            &numParticleGroups, &numAtoms, &forcePtr, &paddedNumAtoms
        };
        cu.executeKernel(computeHCTChainRuleForcesKernel, hctChainArgs, numBlocks * blockSize, blockSize);

        // Step 6c: Compute receptor grid interpolation chain rule forces
        void* receptorGradArgs[] = {
            &posqPtr, &particleIndicesPtr, &radiiPtr,
            &bornRadiiPtr, &hctReceptorPtr, &hctLigandPtr, &dE_dRPtr,
            &gridCountsPtr, &gridHctProbePtr, &gridHctDerivativesPtr,
            &gridCorrectionNPtr, &gridCorrectionAPtr, &gridCorrectionBPtr,
            &rThresholdsPtr, &groupStartPtr, &numParticleGroups,
            &originX, &originY, &originZ, &gridSpacing, &probeRadius,
            &numBins, &totalParticles, &numAtoms, &interpolationMethod,
            &useKDECorrections, &hasBinnedKDEDerivatives, &forcePtr, &paddedNumAtoms
        };
        cu.executeKernel(computeReceptorHCTGradientForceKernel, receptorGradArgs, numBlocks * blockSize, blockSize);
    }

    // Download group energies only when energy is needed to avoid sync barriers
    if (includeEnergy) {
        groupEnergies.download(groupEnergiesHost);
        groupLigandEnergies.download(groupLigandEnergiesHost);

        // Copy ligand energies for separate reporting
        for (int g = 0; g < numParticleGroups; g++) {
            groupLigandEnergiesHost[g] = groupEnergiesHost[g];
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

void CudaCalcGBSAGridForceKernel::updateParametersInContext(ContextImpl& context,
                                                            const GBSAGridForce& force) {
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
    interpolationMethod = force.getInterpolationMethod();

    // Update alchemical scaling
    globalScalingFactor = static_cast<float>(force.getGlobalScalingFactor());
    int nGroups = force.getNumParticleGroups();
    if (nGroups > 0) {
        vector<float> groupScalings(numParticleGroups);
        for (int g = 0; g < nGroups; g++) {
            groupScalings[g] = static_cast<float>(force.getGroupScalingFactor(g));
        }
        groupScalingFactorsBuffer.upload(groupScalings);
    }
}

double CudaCalcGBSAGridForceKernel::getGroupEnergy(int groupIndex) const {
    if (groupIndex < 0 || groupIndex >= static_cast<int>(groupEnergiesHost.size())) {
        throw OpenMMException("GBSAGridForce: invalid group index");
    }
    return groupLigandEnergiesHost[groupIndex];
}

double CudaCalcGBSAGridForceKernel::getGroupLigandDesolvationEnergy(int groupIndex) const {
    if (groupIndex < 0 || groupIndex >= static_cast<int>(groupLigandEnergiesHost.size())) {
        throw OpenMMException("GBSAGridForce: invalid group index");
    }
    return groupLigandEnergiesHost[groupIndex];
}

vector<double> CudaCalcGBSAGridForceKernel::getGroupBornRadii(int groupIndex) const {
    if (groupIndex < 0 || groupIndex >= static_cast<int>(groupBornRadiiHost.size())) {
        throw OpenMMException("GBSAGridForce: invalid group index");
    }

    // Download Born radii if not cached
    if (groupBornRadiiHost[groupIndex].empty()) {
        vector<int> groupStarts(numParticleGroups + 1);
        groupStartIndex.download(groupStarts);
        int startIdx = groupStarts[groupIndex];
        int endIdx = groupStarts[groupIndex + 1];

        vector<float> allBornRadii(bornRadii.getSize());
        bornRadii.download(allBornRadii);

        groupBornRadiiHost[groupIndex].assign(allBornRadii.begin() + startIdx,
                                               allBornRadii.begin() + endIdx);
    }

    vector<double> result(groupBornRadiiHost[groupIndex].begin(),
                          groupBornRadiiHost[groupIndex].end());
    return result;
}

void CudaCalcGBSAGridForceKernel::computeHessian(ContextImpl& context) {
    cu.setAsCurrent();

    int totalParticles = particleIndices.getSize();
    if (totalParticles == 0) return;

    int dim3N = 3 * totalParticles;

    // Allocate host-side Hessian storage
    lastFullHessian.assign(dim3N * dim3N, 0.0);
    lastHessianBlocks.assign(6 * totalParticles, 0.0);
    hessianNumAtoms = totalParticles;

    // Allocate GPU buffers for analytical Hessian (once)
    if (!hessianBuffersInitialized) {
        hessianDRdPsi.initialize<float>(cu, totalParticles, "hessianDRdPsi");
        hessianD2RdPsi2.initialize<float>(cu, totalParticles, "hessianD2RdPsi2");
        hessianJacobian.initialize<float>(cu, totalParticles * dim3N, "hessianJacobian");
        hessianCouplingMatrix.initialize<float>(cu, totalParticles * totalParticles, "hessianCouplingM");
        hessianGridHCTHessian.initialize<float>(cu, totalParticles * 6, "hessianGridHCTHessian");
        hessianMatrix.initialize<float>(cu, dim3N * dim3N, "hessianMatrix");
        hessianBuffersInitialized = true;
    }

    // Ensure GBSA pipeline has been run (bornRadii, dE_dR populated)
    // The caller should have called getState(getForces=True) first, which runs execute().
    // Run the dE/dR accumulation explicitly to ensure it's fresh.
    int blockSize = 256;
    int numBlocks = (totalParticles + blockSize - 1) / blockSize;

    CUdeviceptr posqPtr = cu.getPosq().getDevicePointer();
    CUdeviceptr particleIndicesPtr = particleIndices.getDevicePointer();
    CUdeviceptr radiiPtr = radii.getDevicePointer();
    CUdeviceptr scaleFactorsPtr = scaleFactors.getDevicePointer();
    CUdeviceptr chargesPtr = charges.getDevicePointer();
    CUdeviceptr bornRadiiPtr = bornRadii.getDevicePointer();
    CUdeviceptr hctReceptorPtr = hctReceptor.getDevicePointer();
    CUdeviceptr hctLigandPtr = hctLigand.getDevicePointer();
    CUdeviceptr dE_dRPtr = dE_dR.getDevicePointer();
    CUdeviceptr exclusionAtomsPtr = exclusionAtoms.getDevicePointer();
    CUdeviceptr exclusionStartPtr = exclusionStartIndex.getDevicePointer();
    CUdeviceptr groupStartPtr = groupStartIndex.getDevicePointer();
    CUdeviceptr gridCountsPtr = gridCounts.getDevicePointer();
    CUdeviceptr gridHctProbePtr = gridHctProbe.getDevicePointer();
    CUdeviceptr gridHctDerivativesPtr = hasHctDerivatives ? gridHctDerivatives.getDevicePointer() : 0;
    CUdeviceptr gridCorrectionNPtr = gridCorrectionN.getDevicePointer();
    CUdeviceptr gridCorrectionAPtr = gridCorrectionA.getDevicePointer();
    CUdeviceptr gridCorrectionBPtr = gridCorrectionB.getDevicePointer();
    CUdeviceptr rThresholdsPtr = rThresholds.getDevicePointer();
    CUdeviceptr dRdPsiPtr = hessianDRdPsi.getDevicePointer();
    CUdeviceptr d2RdPsi2Ptr = hessianD2RdPsi2.getDevicePointer();
    CUdeviceptr jacobianPtr = hessianJacobian.getDevicePointer();
    CUdeviceptr couplingPtr = hessianCouplingMatrix.getDevicePointer();
    CUdeviceptr hessianPtr = hessianMatrix.getDevicePointer();

    // Kernel 1: Prepare OBC intermediates (dR/dΨ, d²R/dΨ², dE/dHCT)
    CUdeviceptr dE_dHCTPtr = dE_dHCT.getDevicePointer();
    void* prepArgs[] = {
        &radiiPtr, &bornRadiiPtr, &hctReceptorPtr, &hctLigandPtr,
        &dE_dRPtr, &totalParticles, &numAtoms, &dRdPsiPtr, &d2RdPsi2Ptr,
        &dE_dHCTPtr
    };
    cu.executeKernel(prepareHessianIntermediatesKernel, prepArgs, numBlocks * blockSize, blockSize);

    // Kernel 2: Compute HCT Jacobian J[N x 3N]
    void* jacArgs[] = {
        &posqPtr, &particleIndicesPtr, &radiiPtr, &scaleFactorsPtr,
        &exclusionAtomsPtr, &exclusionStartPtr, &groupStartPtr,
        &numParticleGroups, &numAtoms,
        &gridCountsPtr, &gridHctProbePtr, &gridHctDerivativesPtr,
        &gridCorrectionNPtr, &gridCorrectionAPtr, &gridCorrectionBPtr,
        &rThresholdsPtr,
        &originX, &originY, &originZ, &gridSpacing, &probeRadius,
        &numBins, &interpolationMethod,
        &useKDECorrections, &hasBinnedKDEDerivatives,
        &totalParticles, &jacobianPtr
    };
    cu.executeKernel(computeHCTJacobianKernel, jacArgs, numBlocks * blockSize, blockSize);

    // Kernel 2b: Compute receptor grid HCT Hessian d²Ψ_grid/(dx^α dx^β)
    CUdeviceptr gridHCTHessianPtr = hessianGridHCTHessian.getDevicePointer();
    void* gridHessArgs[] = {
        &posqPtr, &particleIndicesPtr, &radiiPtr,
        &groupStartPtr, &numParticleGroups, &numAtoms,
        &gridCountsPtr, &gridHctProbePtr, &gridHctDerivativesPtr,
        &gridCorrectionNPtr, &gridCorrectionAPtr, &gridCorrectionBPtr,
        &rThresholdsPtr,
        &originX, &originY, &originZ, &gridSpacing, &probeRadius,
        &numBins, &interpolationMethod,
        &useKDECorrections, &hasBinnedKDEDerivatives,
        &totalParticles, &gridHCTHessianPtr
    };
    cu.executeKernel(computeReceptorGridHessianKernel, gridHessArgs, numBlocks * blockSize, blockSize);

    // Kernel 3: Compute Born coupling matrix M[N x N]
    void* couplingArgs[] = {
        &posqPtr, &particleIndicesPtr, &chargesPtr, &radiiPtr,
        &bornRadiiPtr, &dRdPsiPtr, &d2RdPsi2Ptr, &dE_dRPtr,
        &exclusionAtomsPtr, &exclusionStartPtr, &groupStartPtr,
        &numParticleGroups, &numAtoms, &prefactor,
        &includeSurfaceArea, &surfaceTension, &probeRadius,
        &totalParticles, &couplingPtr
    };
    cu.executeKernel(computeBornCouplingMatrixKernel, couplingArgs, numBlocks * blockSize, blockSize);

    // Kernel 4: Assemble full Hessian H = H_direct + H_cross + J^T·M·J + H_hct2
    int totalElements = dim3N * dim3N;
    int numBlocksH = (totalElements + blockSize - 1) / blockSize;
    void* assembleArgs[] = {
        &posqPtr, &particleIndicesPtr, &chargesPtr, &bornRadiiPtr,
        &exclusionAtomsPtr, &exclusionStartPtr, &groupStartPtr,
        &numParticleGroups, &numAtoms, &prefactor,
        &jacobianPtr, &couplingPtr, &dE_dHCTPtr, &scaleFactorsPtr, &radiiPtr,
        &dRdPsiPtr, &gridHCTHessianPtr,
        &totalParticles, &hessianPtr
    };
    cu.executeKernel(assembleGBSAHessianKernel, assembleArgs, numBlocksH * blockSize, blockSize);

    // Download Hessian from GPU
    vector<float> hessianFloat(dim3N * dim3N);
    hessianMatrix.download(hessianFloat);

    // Convert to double
    for (int i = 0; i < dim3N * dim3N; i++) {
        lastFullHessian[i] = static_cast<double>(hessianFloat[i]);
    }

    // Extract diagonal blocks: [dxx, dyy, dzz, dxy, dxz, dyz] per atom
    for (int i = 0; i < totalParticles; i++) {
        int base = 3 * i;
        lastHessianBlocks[6 * i + 0] = lastFullHessian[(base + 0) * dim3N + (base + 0)];
        lastHessianBlocks[6 * i + 1] = lastFullHessian[(base + 1) * dim3N + (base + 1)];
        lastHessianBlocks[6 * i + 2] = lastFullHessian[(base + 2) * dim3N + (base + 2)];
        lastHessianBlocks[6 * i + 3] = lastFullHessian[(base + 0) * dim3N + (base + 1)];
        lastHessianBlocks[6 * i + 4] = lastFullHessian[(base + 0) * dim3N + (base + 2)];
        lastHessianBlocks[6 * i + 5] = lastFullHessian[(base + 1) * dim3N + (base + 2)];
    }
}

vector<double> CudaCalcGBSAGridForceKernel::getHessianBlocks() const {
    if (hessianNumAtoms == 0) {
        throw OpenMMException("GBSAGridForce: computeHessian() must be called first");
    }
    return lastHessianBlocks;
}

vector<double> CudaCalcGBSAGridForceKernel::getFullHessian() const {
    if (hessianNumAtoms == 0) {
        throw OpenMMException("GBSAGridForce: computeHessian() must be called first");
    }
    return lastFullHessian;
}

void CudaCalcGBSAGridForceKernel::generateGrid(
    const vector<double>& receptorPositions,
    const vector<double>& receptorRadii,
    const vector<double>& receptorScales,
    int numReceptorAtoms,
    double probeRadiusIn,
    const vector<double>& rThresholdsIn,
    const double* origin,
    const int* counts,
    double spacing,
    bool computeDerivatives,
    vector<float>& outHctProbe,
    vector<float>& outCorrectionN,
    vector<float>& outCorrectionA,
    vector<float>& outCorrectionB,
    vector<float>& outDerivatives
) {
    cu.setAsCurrent();

    // Validate input
    if (receptorPositions.size() != static_cast<size_t>(numReceptorAtoms * 3)) {
        throw OpenMMException("GBSAGridForce: receptorPositions size must be 3 * numReceptorAtoms");
    }

    int nx = counts[0];
    int ny = counts[1];
    int nz = counts[2];
    int totalGridPoints = nx * ny * nz;
    int numBinsIn = static_cast<int>(rThresholdsIn.size());

    // Load kernel module if not already done
    if (generationModule == nullptr) {
        generationModule = cu.createModule(CudaGridForceKernelSources::gridForceKernel);
        generateLigandHCTGridKernel = cu.getKernel(generationModule, "generateLigandHCTGrid");
        generateLigandHCTGridWithCorrectionsKernel = cu.getKernel(generationModule, "generateLigandHCTGridWithCorrections");
        generateLigandHCTGridWithDerivativesKernel = cu.getKernel(generationModule, "generateLigandHCTGridWithDerivatives");
        generateBinnedGridsWithKDEKernel = cu.getKernel(generationModule, "generateBinnedGridsWithKDE");
        generateBinnedGridsWithKDEDerivativesKernel = cu.getKernel(generationModule, "generateBinnedGridsWithKDEDerivatives");
        // New 4-grid generation kernels (all in same module)
        generateDesolvationGrids4Kernel = cu.getKernel(generationModule, "generateDesolvationGrids4");
        generateHCTProbeGridKernel = cu.getKernel(generationModule, "generateHCTProbeGrid");
    }

    // Convert to float arrays
    vector<float3> positionsF(numReceptorAtoms);
    vector<float> radiiF(numReceptorAtoms);
    vector<float> scalesF(numReceptorAtoms);

    for (int i = 0; i < numReceptorAtoms; i++) {
        positionsF[i] = make_float3(
            static_cast<float>(receptorPositions[i*3]),
            static_cast<float>(receptorPositions[i*3 + 1]),
            static_cast<float>(receptorPositions[i*3 + 2])
        );
        radiiF[i] = static_cast<float>(receptorRadii[i]);
        scalesF[i] = static_cast<float>(receptorScales[i]);
    }

    vector<float> thresholdsF(numBinsIn);
    for (int i = 0; i < numBinsIn; i++) {
        thresholdsF[i] = static_cast<float>(rThresholdsIn[i]);
    }

    float originXf = static_cast<float>(origin[0]);
    float originYf = static_cast<float>(origin[1]);
    float originZf = static_cast<float>(origin[2]);
    float spacingF = static_cast<float>(spacing);
    float probeRadiusF = static_cast<float>(probeRadiusIn);

    // Allocate GPU memory for receptor data
    CudaArray d_positions, d_radii, d_scales, d_thresholds, d_counts;

    d_positions.initialize<float3>(cu, numReceptorAtoms, "genRecPositions");
    d_radii.initialize<float>(cu, numReceptorAtoms, "genRecRadii");
    d_scales.initialize<float>(cu, numReceptorAtoms, "genRecScales");
    d_thresholds.initialize<float>(cu, numBinsIn, "genThresholds");
    d_counts.initialize<int>(cu, 3, "genCounts");

    d_positions.upload(positionsF);
    d_radii.upload(radiiF);
    d_scales.upload(scalesF);
    d_thresholds.upload(thresholdsF);
    vector<int> countsVec = {nx, ny, nz};
    d_counts.upload(countsVec);

    int blockSize = 256;
    int numBlocks = (totalGridPoints + blockSize - 1) / blockSize;

    if (computeDerivatives) {
        // Use generateBinnedGridsWithKDEDerivatives kernel: generates HCT grid with 27 derivatives
        // plus binned correction grids with 27 derivatives per bin.
        // Output layouts:
        //   gridHctDerivatives: [27 * totalGridPoints] in RASPA3 order
        //   gridCorrectionN/A/B: [numBins * 27 * totalGridPoints] - bin-major, then deriv, then point
        CudaArray d_hctDerivs, d_corrN, d_corrA, d_corrB;
        d_hctDerivs.initialize<float>(cu, 27 * totalGridPoints, "genHctDerivs");
        d_corrN.initialize<float>(cu, numBinsIn * 27 * totalGridPoints, "genCorrN");
        d_corrA.initialize<float>(cu, numBinsIn * 27 * totalGridPoints, "genCorrA");
        d_corrB.initialize<float>(cu, numBinsIn * 27 * totalGridPoints, "genCorrB");

        void* args[] = {
            &d_hctDerivs.getDevicePointer(),
            &d_corrN.getDevicePointer(),
            &d_corrA.getDevicePointer(),
            &d_corrB.getDevicePointer(),
            &d_positions.getDevicePointer(),
            &d_radii.getDevicePointer(),
            &d_scales.getDevicePointer(),
            &numReceptorAtoms,
            &probeRadiusF,
            &d_thresholds.getDevicePointer(),
            &numBinsIn,
            &kdeBandwidth,
            &kdeEpsilonB,
            &originXf, &originYf, &originZf,
            &d_counts.getDevicePointer(),
            &spacingF,
            &totalGridPoints
        };
        // Use cuLaunchKernel directly to bypass OpenMM's numThreadBlocks cap,
        // which limits grid dimensions and causes incomplete grid generation
        // at fine spacings (e.g. 256K+ grid points).
        CUresult result = cuLaunchKernel(generateBinnedGridsWithKDEDerivativesKernel,
            numBlocks, 1, 1, blockSize, 1, 1, 0, cu.getCurrentStream(), args, NULL);
        if (result != CUDA_SUCCESS)
            throw OpenMMException("Error launching generateBinnedGridsWithKDEDerivatives kernel");

        // Download HCT derivatives (27 per point, RASPA3 layout)
        outDerivatives.resize(27 * totalGridPoints);
        d_hctDerivs.download(outDerivatives);

        // Extract HCT values from derivative array (first plane is the value)
        outHctProbe.resize(totalGridPoints);
        for (int i = 0; i < totalGridPoints; i++) {
            outHctProbe[i] = outDerivatives[i];
        }

        // Download binned correction grids with derivatives
        // Layout: [bin * 27 * totalGridPoints + deriv * totalGridPoints + point]
        outCorrectionN.resize(numBinsIn * 27 * totalGridPoints);
        outCorrectionA.resize(numBinsIn * 27 * totalGridPoints);
        outCorrectionB.resize(numBinsIn * 27 * totalGridPoints);
        d_corrN.download(outCorrectionN);
        d_corrA.download(outCorrectionA);
        d_corrB.download(outCorrectionB);

        return;  // Binned+KDE derivatives path complete
    }

    // Non-derivative path: use binned corrections with KDE smoothing
    CudaArray d_hctProbe, d_corrN, d_corrA, d_corrB;
    d_hctProbe.initialize<float>(cu, totalGridPoints, "genHctProbe");
    d_corrN.initialize<float>(cu, numBinsIn * totalGridPoints, "genCorrN");
    d_corrA.initialize<float>(cu, numBinsIn * totalGridPoints, "genCorrA");
    d_corrB.initialize<float>(cu, numBinsIn * totalGridPoints, "genCorrB");

    {
        // Generate values and corrections with KDE smoothing within bins
        void* args[] = {
            &d_hctProbe.getDevicePointer(),
            &d_corrN.getDevicePointer(),
            &d_corrA.getDevicePointer(),
            &d_corrB.getDevicePointer(),
            &d_positions.getDevicePointer(),
            &d_radii.getDevicePointer(),
            &d_scales.getDevicePointer(),
            &numReceptorAtoms,
            &probeRadiusF,
            &d_thresholds.getDevicePointer(),
            &numBinsIn,
            &kdeBandwidth,  // KDE bandwidth for smooth transitions
            &originXf, &originYf, &originZf,
            &d_counts.getDevicePointer(),
            &spacingF,
            &totalGridPoints
        };
        // Use cuLaunchKernel directly to bypass OpenMM's numThreadBlocks cap,
        // which limits grid dimensions and causes incomplete grid generation
        // at fine spacings (e.g. 256K+ grid points).
        CUresult result = cuLaunchKernel(generateBinnedGridsWithKDEKernel,
            numBlocks, 1, 1, blockSize, 1, 1, 0, cu.getCurrentStream(), args, NULL);
        if (result != CUDA_SUCCESS)
            throw OpenMMException("Error launching generateBinnedGridsWithKDE kernel");

        outHctProbe.resize(totalGridPoints);
        d_hctProbe.download(outHctProbe);
    }

    // Download correction grids
    outCorrectionN.resize(numBinsIn * totalGridPoints);
    outCorrectionA.resize(numBinsIn * totalGridPoints);
    outCorrectionB.resize(numBinsIn * totalGridPoints);

    d_corrN.download(outCorrectionN);
    d_corrA.download(outCorrectionA);
    d_corrB.download(outCorrectionB);
}
