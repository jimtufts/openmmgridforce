/* -------------------------------------------------------------------------- *
 *                              OpenMMGridForce                               *
 * -------------------------------------------------------------------------- */

#include "CudaGBSAGridForceKernels.h"
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

CudaCalcGBSAGridForceKernel::CudaCalcGBSAGridForceKernel(string name, const Platform& platform,
                                                           CudaContext& cu)
    : CalcGBSAGridForceKernel(name, platform), cu(cu), hasInitializedKernel(false),
      numAtoms(0), numParticleGroups(0), originX(0), originY(0), originZ(0),
      gridSpacing(0), probeRadius(0), numBins(0), prefactor(0),
      includeSurfaceArea(false), surfaceTension(0), interpolationMethod(0),
      includeReceptorDesolvation(false), receptorDesolvProbeRadius(0),
      hasHctDerivatives(false), hasReceptorDesolvDerivatives(false),
      computeReceptorHCTKernel(nullptr), computeLigandHCTKernel(nullptr),
      computeBornRadiiKernel(nullptr), computeGBEnergyKernel(nullptr),
      computeSAEnergyKernel(nullptr),
      accumulateSADerivativesKernel(nullptr),
      accumulateBornRadiiDerivativesKernel(nullptr),
      computeHCTChainRuleForcesKernel(nullptr),
      computeReceptorHCTGradientForceKernel(nullptr),
      computeReceptorDesolvationKernel(nullptr),
      generateLigandHCTGridKernel(nullptr),
      generateLigandHCTGridWithCorrectionsKernel(nullptr),
      generateLigandHCTGridWithDerivativesKernel(nullptr),
      generationModule(nullptr) {
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

        // Set grid data
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

    int corrSize = numBins * numPoints;
    gridCorrectionN.initialize<float>(cu, corrSize, "gbsaGridCorrectionN");
    gridCorrectionN.upload(grid->getCorrectionN());

    gridCorrectionA.initialize<float>(cu, corrSize, "gbsaGridCorrectionA");
    gridCorrectionA.upload(grid->getCorrectionA());

    gridCorrectionB.initialize<float>(cu, corrSize, "gbsaGridCorrectionB");
    gridCorrectionB.upload(grid->getCorrectionB());

    // Upload R thresholds
    const auto& thresholds = grid->getRThresholds();
    vector<float> thresholdsFloat(thresholds.begin(), thresholds.end());
    rThresholds.initialize<float>(cu, numBins, "gbsaRThresholds");
    rThresholds.upload(thresholdsFloat);

    // Check for receptor desolvation data
    includeReceptorDesolvation = force.getIncludeReceptorDesolvation() && grid->hasReceptorDesolvation();
    if (includeReceptorDesolvation) {
        receptorDesolvProbeRadius = grid->getReceptorDesolvProbeRadius();
        hasReceptorDesolvDerivatives = grid->hasReceptorDesolvDerivatives();

        // Upload receptor desolvation energy grid
        const auto& recDesolvData = grid->getReceptorDesolvEnergy();
        gridReceptorDesolv.initialize<float>(cu, numPoints, "gbsaReceptorDesolv");
        gridReceptorDesolv.upload(recDesolvData);

        // Upload derivatives if available
        if (hasReceptorDesolvDerivatives) {
            const auto& recDesolvDerivs = grid->getReceptorDesolvDerivatives();
            gridReceptorDesolvDerivs.initialize<float>(cu, recDesolvDerivs.size(), "gbsaReceptorDesolvDerivs");
            gridReceptorDesolvDerivs.upload(recDesolvDerivs);
        }
    }

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
        groupReceptorEnergies.initialize<float>(cu, numParticleGroups, "gbsaGroupReceptorEnergies");

        groupEnergiesHost.resize(numParticleGroups);
        groupLigandEnergiesHost.resize(numParticleGroups);
        groupReceptorEnergiesHost.resize(numParticleGroups);
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
        groupReceptorEnergies.initialize<float>(cu, 1, "gbsaGroupReceptorEnergies");
        groupEnergiesHost.resize(1);
        groupLigandEnergiesHost.resize(1);
        groupReceptorEnergiesHost.resize(1);
        groupBornRadiiHost.resize(1);
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
    computeReceptorDesolvationKernel = cu.getKernel(module, "computeReceptorDesolvation");

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

    // Clear group energies
    vector<float> zeros(numParticleGroups, 0.0f);
    groupEnergies.upload(zeros);
    groupLigandEnergies.upload(zeros);
    groupReceptorEnergies.upload(zeros);

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
        &numBins, &totalParticles, &numAtoms, &interpolationMethod, &hctReceptorPtr
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
    void* energyArgs[] = {
        &posqPtr, &particleIndicesPtr, &chargesPtr, &bornRadiiPtr,
        &exclusionAtomsPtr, &exclusionStartPtr, &groupStartPtr,
        &numParticleGroups, &numAtoms, &prefactor,
        &forcePtr, &groupEnergiesPtr, &paddedNumAtoms
    };
    cu.executeKernel(computeGBEnergyKernel, energyArgs, numBlocks * blockSize, blockSize);

    // Step 5: Optional surface area term
    if (includeSurfaceArea) {
        void* saArgs[] = {
            &radiiPtr, &bornRadiiPtr, &groupStartPtr,
            &numParticleGroups, &numAtoms,
            &surfaceTension, &probeRadius, &groupEnergiesPtr
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
            &numParticleGroups, &numAtoms, &prefactor, &dE_dRPtr
        };
        cu.executeKernel(accumulateBornRadiiDerivativesKernel, bornDerivArgs, numBlocks * blockSize, blockSize);

        // Step 6a2: Add surface area contribution to dE/dR_born if enabled
        if (includeSurfaceArea) {
            void* saDerivArgs[] = {
                &radiiPtr, &bornRadiiPtr, &groupStartPtr,
                &numParticleGroups, &numAtoms, &surfaceTension, &probeRadius, &dE_dRPtr
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
            &numBins, &totalParticles, &numAtoms, &interpolationMethod, &forcePtr, &paddedNumAtoms
        };
        cu.executeKernel(computeReceptorHCTGradientForceKernel, receptorGradArgs, numBlocks * blockSize, blockSize);
    }

    // Step 7: Receptor desolvation energy (if enabled)
    if (includeReceptorDesolvation) {
        CUdeviceptr gridReceptorDesolvPtr = gridReceptorDesolv.getDevicePointer();
        CUdeviceptr gridReceptorDesolvDerivsPtr = hasReceptorDesolvDerivatives ?
            gridReceptorDesolvDerivs.getDevicePointer() : 0;
        CUdeviceptr groupReceptorEnergiesPtr = groupReceptorEnergies.getDevicePointer();
        int hasDerivativesInt = hasReceptorDesolvDerivatives ? 1 : 0;

        void* recDesolvArgs[] = {
            &posqPtr, &particleIndicesPtr, &radiiPtr, &scaleFactorsPtr,
            &gridCountsPtr, &gridReceptorDesolvPtr, &gridReceptorDesolvDerivsPtr,
            &groupStartPtr, &numParticleGroups,
            &originX, &originY, &originZ, &gridSpacing, &receptorDesolvProbeRadius,
            &totalParticles, &numAtoms, &interpolationMethod, &hasDerivativesInt,
            &groupReceptorEnergiesPtr, &forcePtr, &paddedNumAtoms
        };
        cu.executeKernel(computeReceptorDesolvationKernel, recDesolvArgs, numBlocks * blockSize, blockSize);
    }

    // Download group energies
    groupEnergies.download(groupEnergiesHost);  // Ligand desolvation
    groupLigandEnergies.download(groupLigandEnergiesHost);
    if (includeReceptorDesolvation) {
        groupReceptorEnergies.download(groupReceptorEnergiesHost);
    }

    // Copy ligand energies for separate reporting
    for (int g = 0; g < numParticleGroups; g++) {
        groupLigandEnergiesHost[g] = groupEnergiesHost[g];
    }

    // Sum total energy (ligand + receptor)
    double totalEnergy = 0.0;
    for (int g = 0; g < numParticleGroups; g++) {
        totalEnergy += groupEnergiesHost[g];
        if (includeReceptorDesolvation) {
            totalEnergy += groupReceptorEnergiesHost[g];
        }
    }

    return totalEnergy;
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
}

double CudaCalcGBSAGridForceKernel::getGroupEnergy(int groupIndex) const {
    if (groupIndex < 0 || groupIndex >= static_cast<int>(groupEnergiesHost.size())) {
        throw OpenMMException("GBSAGridForce: invalid group index");
    }
    // Total energy = ligand + receptor
    double total = groupLigandEnergiesHost[groupIndex];
    if (includeReceptorDesolvation && groupIndex < static_cast<int>(groupReceptorEnergiesHost.size())) {
        total += groupReceptorEnergiesHost[groupIndex];
    }
    return total;
}

double CudaCalcGBSAGridForceKernel::getGroupLigandDesolvationEnergy(int groupIndex) const {
    if (groupIndex < 0 || groupIndex >= static_cast<int>(groupLigandEnergiesHost.size())) {
        throw OpenMMException("GBSAGridForce: invalid group index");
    }
    return groupLigandEnergiesHost[groupIndex];
}

double CudaCalcGBSAGridForceKernel::getGroupReceptorDesolvationEnergy(int groupIndex) const {
    if (!includeReceptorDesolvation) {
        return 0.0;
    }
    if (groupIndex < 0 || groupIndex >= static_cast<int>(groupReceptorEnergiesHost.size())) {
        throw OpenMMException("GBSAGridForce: invalid group index");
    }
    return groupReceptorEnergiesHost[groupIndex];
}

vector<double> CudaCalcGBSAGridForceKernel::getGroupBornRadii(int groupIndex) const {
    if (groupIndex < 0 || groupIndex >= static_cast<int>(groupBornRadiiHost.size())) {
        throw OpenMMException("GBSAGridForce: invalid group index");
    }
    vector<double> result(groupBornRadiiHost[groupIndex].begin(),
                          groupBornRadiiHost[groupIndex].end());
    return result;
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

    // Allocate GPU memory
    CudaArray d_positions, d_radii, d_scales, d_thresholds, d_counts;
    CudaArray d_hctProbe, d_corrN, d_corrA, d_corrB, d_derivatives;

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

    // Allocate output arrays
    d_hctProbe.initialize<float>(cu, totalGridPoints, "genHctProbe");
    d_corrN.initialize<float>(cu, numBinsIn * totalGridPoints, "genCorrN");
    d_corrA.initialize<float>(cu, numBinsIn * totalGridPoints, "genCorrA");
    d_corrB.initialize<float>(cu, numBinsIn * totalGridPoints, "genCorrB");

    int blockSize = 256;
    int numBlocks = (totalGridPoints + blockSize - 1) / blockSize;

    if (computeDerivatives) {
        // Generate with 27 derivatives per point
        d_derivatives.initialize<float>(cu, 27 * totalGridPoints, "genDerivatives");

        void* args[] = {
            &d_derivatives.getDevicePointer(),
            &d_positions.getDevicePointer(),
            &d_radii.getDevicePointer(),
            &d_scales.getDevicePointer(),
            &numReceptorAtoms,
            &probeRadiusF,
            &originXf, &originYf, &originZf,
            &d_counts.getDevicePointer(),
            &spacingF,
            &totalGridPoints
        };
        cu.executeKernel(generateLigandHCTGridWithDerivativesKernel, args, numBlocks * blockSize, blockSize);

        // Download derivatives (includes value at index 0)
        outDerivatives.resize(27 * totalGridPoints);
        d_derivatives.download(outDerivatives);

        // Extract HCT values from derivative array (index 0 for each point)
        outHctProbe.resize(totalGridPoints);
        for (int i = 0; i < totalGridPoints; i++) {
            outHctProbe[i] = outDerivatives[i];  // First derivative plane is the value
        }

        // Also compute correction grids (always needed, use separate kernel)
        void* corrArgs[] = {
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
            &originXf, &originYf, &originZf,
            &d_counts.getDevicePointer(),
            &spacingF,
            &totalGridPoints
        };
        cu.executeKernel(generateLigandHCTGridWithCorrectionsKernel, corrArgs, numBlocks * blockSize, blockSize);
    } else {
        // Generate values and corrections only
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
            &originXf, &originYf, &originZf,
            &d_counts.getDevicePointer(),
            &spacingF,
            &totalGridPoints
        };
        cu.executeKernel(generateLigandHCTGridWithCorrectionsKernel, args, numBlocks * blockSize, blockSize);

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
