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
      includeSurfaceArea(false), surfaceTension(0),
      computeReceptorHCTKernel(nullptr), computeLigandHCTKernel(nullptr),
      computeBornRadiiKernel(nullptr), computeGBEnergyKernel(nullptr),
      computeSAEnergyKernel(nullptr),
      accumulateSADerivativesKernel(nullptr),
      accumulateBornRadiiDerivativesKernel(nullptr),
      computeHCTChainRuleForcesKernel(nullptr),
      computeReceptorHCTGradientForceKernel(nullptr) {
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

    // Upload grid dimensions
    int nx, ny, nz;
    grid->getCounts(nx, ny, nz);
    vector<int> counts = {nx, ny, nz};
    gridCounts.initialize<int>(cu, 3, "gbsaGridCounts");
    gridCounts.upload(counts);

    // Upload grid data
    int numPoints = nx * ny * nz;
    gridHctProbe.initialize<float>(cu, numPoints, "gbsaGridHctProbe");
    gridHctProbe.upload(grid->getHctProbe());

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

        groupEnergiesHost.resize(numParticleGroups);
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
        groupEnergiesHost.resize(1);
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

    // Get device pointers
    CUdeviceptr posqPtr = cu.getPosq().getDevicePointer();
    CUdeviceptr forcePtr = cu.getLongForceBuffer().getDevicePointer();
    CUdeviceptr particleIndicesPtr = particleIndices.getDevicePointer();
    CUdeviceptr radiiPtr = radii.getDevicePointer();
    CUdeviceptr scaleFactorsPtr = scaleFactors.getDevicePointer();
    CUdeviceptr chargesPtr = charges.getDevicePointer();
    CUdeviceptr gridCountsPtr = gridCounts.getDevicePointer();
    CUdeviceptr gridHctProbePtr = gridHctProbe.getDevicePointer();
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
        &gridCountsPtr, &gridHctProbePtr,
        &gridCorrectionNPtr, &gridCorrectionAPtr, &gridCorrectionBPtr,
        &rThresholdsPtr, &groupStartPtr, &numParticleGroups,
        &originX, &originY, &originZ, &gridSpacing, &probeRadius,
        &numBins, &totalParticles, &numAtoms, &hctReceptorPtr
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
            &gridCountsPtr, &gridHctProbePtr,
            &gridCorrectionNPtr, &gridCorrectionAPtr, &gridCorrectionBPtr,
            &rThresholdsPtr, &groupStartPtr, &numParticleGroups,
            &originX, &originY, &originZ, &gridSpacing, &probeRadius,
            &numBins, &totalParticles, &numAtoms, &forcePtr, &paddedNumAtoms
        };
        cu.executeKernel(computeReceptorHCTGradientForceKernel, receptorGradArgs, numBlocks * blockSize, blockSize);
    }

    // Download group energies
    groupEnergies.download(groupEnergiesHost);

    // Sum total energy
    double totalEnergy = 0.0;
    for (int g = 0; g < numParticleGroups; g++) {
        totalEnergy += groupEnergiesHost[g];
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
}

double CudaCalcGBSAGridForceKernel::getGroupEnergy(int groupIndex) const {
    if (groupIndex < 0 || groupIndex >= static_cast<int>(groupEnergiesHost.size())) {
        throw OpenMMException("GBSAGridForce: invalid group index");
    }
    return groupEnergiesHost[groupIndex];
}

vector<double> CudaCalcGBSAGridForceKernel::getGroupBornRadii(int groupIndex) const {
    if (groupIndex < 0 || groupIndex >= static_cast<int>(groupBornRadiiHost.size())) {
        throw OpenMMException("GBSAGridForce: invalid group index");
    }
    vector<double> result(groupBornRadiiHost[groupIndex].begin(),
                          groupBornRadiiHost[groupIndex].end());
    return result;
}
