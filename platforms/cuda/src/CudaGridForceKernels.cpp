/* -------------------------------------------------------------------------- *
 *                              OpenMMGridForce                               *
 * -------------------------------------------------------------------------- */

#define DEBUG_GRIDFORCE 0

#include "CudaGridForceKernels.h"
#include "CudaGridForceKernelSources.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/cuda/CudaContext.h"
#include "openmm/NonbondedForce.h"
#include "openmm/HarmonicBondForce.h"
#include "openmm/HarmonicAngleForce.h"
#include "openmm/PeriodicTorsionForce.h"
#include "IsolatedNonbondedForce.h"
#include "BSplinePrefilter.h"
#include "CudaAdaptivePrefilter.h"
#include <cuda_runtime.h>
#include <map>
#include <iostream>
#include <iomanip>
#include <string>
#include <cmath>
#include <algorithm>
#include <chrono>

using namespace GridForcePlugin;
using namespace OpenMM;
using namespace std;

// Grid cache for sharing GPU memory across multiple GridForce instances WITHIN same CudaContext
// Key: (CudaContext*, grid_hash), Value: (weak_ptr<CudaArray>, context_id)
// Using weak_ptr allows automatic cleanup when all kernel instances are destroyed
// Context ID helps detect stale entries from destroyed contexts
static std::map<std::pair<void*, size_t>, std::weak_ptr<CudaArray>> gridCache;
static std::map<std::pair<void*, size_t>, std::weak_ptr<CudaArray>> derivativeCache;
static size_t nextContextId = 0;
static std::map<void*, size_t> contextIds;

// Clear GPU-side caches to free CUDA memory
void clearCudaGridCaches() {
    gridCache.clear();
    derivativeCache.clear();
    contextIds.clear();
    nextContextId = 0;
}

// Helper to compute grid hash for caching
static size_t computeGridHash(const vector<int>& counts, const vector<double>& spacing, const vector<double>& vals) {
    size_t hash = 0;
    // Hash dimensions
    hash ^= std::hash<int>{}(counts[0]) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
    hash ^= std::hash<int>{}(counts[1]) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
    hash ^= std::hash<int>{}(counts[2]) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
    // Hash spacing
    hash ^= std::hash<double>{}(spacing[0]) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
    hash ^= std::hash<double>{}(spacing[1]) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
    hash ^= std::hash<double>{}(spacing[2]) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
    // Hash first 100 and last 100 values (sufficient to distinguish different grids)
    size_t n = std::min(vals.size(), (size_t)100);
    for (size_t i = 0; i < n; i++) {
        hash ^= std::hash<double>{}(vals[i]) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
    }
    if (vals.size() > 100) {
        for (size_t i = vals.size() - 100; i < vals.size(); i++) {
            hash ^= std::hash<double>{}(vals[i]) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        }
    }
    return hash;
}

CudaCalcGridForceKernel::~CudaCalcGridForceKernel() {
}

// Initialize analysis-related members in constructor if needed
// Note: analysisBuffersInitialized should be false by default

void CudaCalcGridForceKernel::initialize(const System& system, const GridForce& force) {
    cu.setAsCurrent();
    // Get grid parameters
    vector<int> counts_local;
    vector<double> spacing_local;
    vector<double> vals;
    vector<double> scaling_factors;
    force.getGridParameters(counts_local, spacing_local, vals, scaling_factors);
    double inv_power = force.getInvPower();
    double outOfBounds_k = force.getOutOfBoundsRestraint();
    double grid_cap = force.getGridCap();
    int interp_method = force.getInterpolationMethod();

    numAtoms = system.getNumParticles();

    // Store counts and spacing as member variables for generateGrid
    counts = counts_local;
    spacing = spacing_local;

    // Ensure CachedGridData exists for proper GPU cache keying
    // This is needed because loadFromFile() is called before m_systemPtr is set,
    // so CachedGridData might not have been created during file loading.
    if (!force.getCachedGridData() && !vals.empty()) {
        // Get grid origin
        double ox, oy, oz;
        force.getGridOrigin(ox, oy, oz);

        // Get derivatives if available (needed for tiled mode with tricubic/triquintic)
        const vector<double>& derivs = force.getDerivatives();

        // Create CachedGridData with current grid values and derivatives
        auto cachedGridData = std::make_shared<CachedGridData>(
            vals,
            derivs,  // Pass actual derivatives from the force
            counts_local,
            spacing_local,
            ox, oy, oz
        );

        // Set it on the GridForce object
        const_cast<GridForce&>(force).setCachedGridData(cachedGridData);
    }
//     std::cout << "CudaCalcGridForceKernel::initialize() received counts: ["
//               << counts[0] << ", " << counts[1] << ", " << counts[2] << "] = "
//               << (counts[0] * counts[1] * counts[2]) << " points" << std::endl;

    // Store grid cap, invPower, invPowerMode, arcsinhScale, and runtimeCap BEFORE generateGrid() is called
    gridCap = (float)grid_cap;
    invPower = (float)inv_power;
    invPowerMode = static_cast<int>(force.getInvPowerMode());
    arcsinhScale = (float)force.getArcsinhScale();
    globalScalingFactor = (float)force.getGlobalScalingFactor();
    runtimeCap = (float)force.getRuntimeCap();
    evaluateInVSpace = force.getEvaluateInVSpace() ? 1 : 0;
    interpolationMethod = interp_method;

    // Store ligand atoms and derivative computation flag
    ligandAtoms = force.getLigandAtoms();
    computeDerivatives = force.getComputeDerivatives();
    useDoubleStorage = force.getUseDoubleStorage();

    // Get filtered particle list (empty = all particles)
    particles = force.getParticles();
    if (!particles.empty()) {
        // Only process filtered particles
        numAtoms = particles.size();
    }

    // Auto-calculate scaling factors if enabled and not already provided
#if DEBUG_GRIDFORCE
    std::cout << "[DEBUG] Auto-calculate check: autoCalc=" << force.getAutoCalculateScalingFactors()
              << ", scaling_factors.empty()=" << scaling_factors.empty()
              << ", scaling_factors.size()=" << scaling_factors.size() << std::endl;
#endif
    if (force.getAutoCalculateScalingFactors() && scaling_factors.empty()) {
        std::string scalingProperty = force.getScalingProperty();
        if (scalingProperty.empty()) {
            throw OpenMMException("GridForce: Auto-calculate scaling factors enabled but no scaling property specified");
        }

        // Validate scaling property
        if (scalingProperty != "charge" && scalingProperty != "ljr" && scalingProperty != "lja") {
            throw OpenMMException("GridForce: Invalid scaling property '" + scalingProperty + "'. Must be 'charge', 'ljr', or 'lja'");
        }

        // Find NonbondedForce or IsolatedNonbondedForce in the system
        const NonbondedForce* nonbondedForce = nullptr;
        const IsolatedNonbondedForce* isolatedNbForce = nullptr;
        for (int i = 0; i < system.getNumForces(); i++) {
            if (dynamic_cast<const NonbondedForce*>(&system.getForce(i)) != nullptr) {
                nonbondedForce = dynamic_cast<const NonbondedForce*>(&system.getForce(i));
                break;
            }
            if (dynamic_cast<const IsolatedNonbondedForce*>(&system.getForce(i)) != nullptr) {
                isolatedNbForce = dynamic_cast<const IsolatedNonbondedForce*>(&system.getForce(i));
                // Keep looking for NonbondedForce (prefer it if both exist)
            }
        }

        if (nonbondedForce == nullptr && isolatedNbForce == nullptr) {
            throw OpenMMException("GridForce: Auto-calculate scaling factors requires a NonbondedForce or IsolatedNonbondedForce in the system");
        }

        // Extract scaling factors based on property
        int numTemplateAtoms = isolatedNbForce ? isolatedNbForce->getNumAtoms() : numAtoms;
        scaling_factors.resize(numAtoms);
        for (int i = 0; i < numAtoms; i++) {
            double charge, sigma, epsilon;
            if (nonbondedForce != nullptr) {
                nonbondedForce->getParticleParameters(i, charge, sigma, epsilon);
            } else {
                isolatedNbForce->getAtomParameters(i % numTemplateAtoms, charge, sigma, epsilon);
            }

#if DEBUG_GRIDFORCE
            if (i < 3) {
                std::cout << "[AUTO-CALC] Particle " << i << ": charge=" << charge
                          << ", sigma=" << sigma << ", epsilon=" << epsilon << std::endl;
            }
#endif

            if (scalingProperty == "charge") {
                // For electrostatic grids: use charge directly
                scaling_factors[i] = charge;
            } else if (scalingProperty == "ljr") {
                // For LJ repulsive: sqrt(epsilon) * Rmin^6
                // where Rmin = 2^(1/6) * sigma (AMBER convention)
                double rmin = std::pow(2.0, 1.0/6.0) * sigma;
                scaling_factors[i] = std::sqrt(epsilon) * std::pow(rmin, 6.0);
            } else if (scalingProperty == "lja") {
                // For LJ attractive: sqrt(epsilon) * Rmin^3
                // where Rmin = 2^(1/6) * sigma (AMBER convention)
                double rmin = std::pow(2.0, 1.0/6.0) * sigma;
                scaling_factors[i] = std::sqrt(epsilon) * std::pow(rmin, 3.0);
            }

#if DEBUG_GRIDFORCE
            if (i < 3) {
                std::cout << "[AUTO-CALC] scalingFactors[" << i << "] = " << scaling_factors[i]
                          << " (property=" << scalingProperty << ")" << std::endl;
            }
#endif
        }

        // Copy calculated scaling factors back to GridForce object
        const_cast<GridForce&>(force).setScalingFactors(scaling_factors);

        // IMPORTANT: If using particle groups, update their scaling factors too
        // Particle groups may have been initialized with default 1.0 values,
        // but we want to use the auto-calculated values instead
        if (force.getNumParticleGroups() > 0) {
            for (int i = 0; i < force.getNumParticleGroups(); i++) {
                ParticleGroup& group = const_cast<ParticleGroup&>(force.getParticleGroup(i));
                // Update the group's scaling factors with auto-calculated values
                group.scalingFactors.clear();
                for (int particleIdx : group.particleIndices) {
                    if (particleIdx < scaling_factors.size()) {
                        group.scalingFactors.push_back(scaling_factors[particleIdx]);
                    } else {
                        group.scalingFactors.push_back(0.0);
                    }
                }
            }
        }
    }

    // Handle particle groups for multi-ligand workflows
    numParticleGroups = force.getNumParticleGroups();
    if (numParticleGroups > 0) {
        // Initialize scaling_factors if not already done (needed for particle group mapping)
        if (scaling_factors.empty()) {
            scaling_factors.resize(numAtoms, 0.0);
        }

        // Create particle-to-group mapping for per-group energy tracking
        vector<int> particleToGroupMapHost(numAtoms, -1);  // -1 = no group

        // Calculate totalGroupParticles first (needed for buffer allocation)
        totalGroupParticles = 0;
        for (int i = 0; i < numParticleGroups; i++) {
            const ParticleGroup& group = force.getParticleGroup(i);
            totalGroupParticles += group.particleIndices.size();
        }

        // Populate from particle groups
        for (int i = 0; i < numParticleGroups; i++) {
            const ParticleGroup& group = force.getParticleGroup(i);
            for (size_t j = 0; j < group.particleIndices.size(); j++) {
                int particleIdx = group.particleIndices[j];
                if (particleIdx < numAtoms) {
                    // If group has explicit scaling factors, use them
                    // Otherwise keep the auto-calculated ones from above (or zeros)
                    if (!group.scalingFactors.empty() && j < group.scalingFactors.size()) {
                        scaling_factors[particleIdx] = group.scalingFactors[j];
                    }
                    particleToGroupMapHost[particleIdx] = i;  // Map particle to group index
                }
            }
        }

        // Upload particle-to-group mapping to GPU
        particleToGroupMap.initialize<int>(cu, numAtoms, "particleToGroupMap");
        particleToGroupMap.upload(particleToGroupMapHost);

        // Initialize per-group energy buffers
        groupEnergyBuffer.initialize<float>(cu, numParticleGroups, "groupEnergyBuffer");
        groupUnscaledEnergyBuffer.initialize<float>(cu, numParticleGroups, "groupUnscaledEnergyBuffer");

        // Initialize per-group scaling factors buffer
        std::vector<float> groupScalings(numParticleGroups, 1.0f);
        for (int i = 0; i < numParticleGroups; i++) {
            groupScalings[i] = (float)force.getParticleGroup(i).groupScalingFactor;
        }
        groupScalingFactorsBuffer.initialize<float>(cu, numParticleGroups, "groupScalingFactors");
        groupScalingFactorsBuffer.upload(groupScalings);

        // Initialize per-group runtime caps buffer
        std::vector<float> groupCaps(numParticleGroups, 0.0f);
        for (int i = 0; i < numParticleGroups; i++) {
            groupCaps[i] = (float)force.getParticleGroup(i).groupRuntimeCap;
        }
        groupRuntimeCapsBuffer.initialize<float>(cu, numParticleGroups, "groupRuntimeCaps");
        groupRuntimeCapsBuffer.upload(groupCaps);

        // Initialize per-atom energy buffer (for debugging/analysis)
        if (totalGroupParticles > 0) {
            atomEnergyBuffer.initialize<float>(cu, totalGroupParticles, "atomEnergyBuffer");
            lastAtomEnergies.resize(totalGroupParticles, 0.0f);
            // Initialize per-atom raw (pre-cap) energy buffer
            atomRawEnergyBuffer.initialize<float>(cu, totalGroupParticles, "atomRawEnergyBuffer");
            lastAtomRawEnergies.resize(totalGroupParticles, 0.0f);
            // Initialize per-atom out-of-bounds buffer
            outOfBoundsBuffer.initialize<int>(cu, totalGroupParticles, "outOfBoundsBuffer");
            lastOutOfBoundsFlags.resize(totalGroupParticles, 0);
        }
    } else {
        numParticleGroups = 0;
        totalGroupParticles = 0;
    }

    // Auto-generate grid if enabled and grid values are empty
    if (force.getAutoGenerateGrid() && vals.empty()) {
        std::string gridType = force.getGridType();

        // Validate grid type
        if (gridType != "charge" && gridType != "ljr" && gridType != "lja") {
            throw OpenMMException("GridForce: Invalid grid type '" + gridType + "'. Must be 'charge', 'ljr', or 'lja'");
        }

        // Ensure grid counts and spacing are set
        if (counts.size() != 3 || spacing.size() != 3) {
            throw OpenMMException("GridForce: Grid counts and spacing must be set before auto-generation");
        }

        // Find NonbondedForce
        const NonbondedForce* nonbondedForce = nullptr;
        for (int i = 0; i < system.getNumForces(); i++) {
            if (dynamic_cast<const NonbondedForce*>(&system.getForce(i)) != nullptr) {
                nonbondedForce = dynamic_cast<const NonbondedForce*>(&system.getForce(i));
                break;
            }
        }

        if (nonbondedForce == nullptr) {
            throw OpenMMException("GridForce: Auto-grid generation requires a NonbondedForce in the system");
        }

        // Get receptor atoms and positions
        std::vector<int> receptorAtoms = force.getReceptorAtoms();
        std::vector<int> ligandAtoms = force.getLigandAtoms();
        const std::vector<Vec3>& receptorPositions = force.getReceptorPositions();

        // If receptorAtoms not specified, use all atoms except ligandAtoms
        if (receptorAtoms.empty()) {
            for (int i = 0; i < system.getNumParticles(); i++) {
                bool isLigand = std::find(ligandAtoms.begin(), ligandAtoms.end(), i) != ligandAtoms.end();
                if (!isLigand) {
                    receptorAtoms.push_back(i);
                }
            }
        }

        // Validate receptor positions
        if (receptorPositions.empty()) {
            throw OpenMMException("GridForce: Receptor positions must be set for auto-grid generation");
        }

        if (receptorPositions.size() < receptorAtoms.size()) {
            throw OpenMMException("GridForce: Not enough receptor positions provided");
        }

        // Get grid origin
        double ox, oy, oz;
        force.getGridOrigin(ox, oy, oz);

        // Check if tiled output is requested
        std::string tiledOutputFile = force.getTiledOutputFile();
        if (!tiledOutputFile.empty()) {
            // Generate directly to tiled file - avoids holding full grid in memory
            int tiledTileSize = force.getTiledOutputTileSize();
            generateGridToTiledFile(system, nonbondedForce, gridType, receptorAtoms, receptorPositions,
                                    ox, oy, oz, tiledOutputFile, computeDerivatives, tiledTileSize);

            std::cout << "GridForce: Tiled grid saved to " << tiledOutputFile << std::endl;

            // Note: The CUDA tile generation kernel already applies V -> sign(V) * |V|^(1/n)
            // when invPower != 0.0, so tile values already contain transformed values.
            // No CPU-side transform needed for the tiled path.

            // Apply arcsinh transform to tiled file (streaming, tile-by-tile)
            // Must be done BEFORE prefiltering to compress dynamic range first
            if (arcsinhScale > 0.0f) {
                TiledGridData tiledForArcsinh;
                tiledForArcsinh.openForReading(tiledOutputFile);
                tiledForArcsinh.applyArcsinhTransform((double)arcsinhScale);
            }

            // Apply B-spline prefilter to the tiled file (streaming, tile-by-tile)
            int bsplineOrder = force.getBSplinePrefilterOrder();
            if (bsplineOrder > 0) {
                TiledGridData tiledForPrefilter;
                tiledForPrefilter.openForReading(tiledOutputFile);
                tiledForPrefilter.applyBSplinePrefilter(bsplineOrder);
            }

            // Automatically set up tiled input for evaluation if tiled mode is requested
            // This allows generate-then-evaluate in one run
            if (force.getTiledMode()) {
                std::cout << "  Setting up tiled input from generated file for evaluation" << std::endl;
                // Set the tiled input file to the generated output file
                const_cast<GridForce&>(force).setTiledInputFile(tiledOutputFile);
                // Skip normal grid generation - tiled file is ready for use
                // Continue to tiled mode initialization below
            } else {
                std::cout << "  Use setTiledInputFile() with setTiledMode(true) to evaluate" << std::endl;
                // Generation-only mode: initialize minimal state so getState() doesn't crash
                // but return 0 energy since we're not evaluating
                g_counts.initialize<int>(cu, 3, "gridCounts");
                g_spacing.initialize<float>(cu, 3, "gridSpacing");
                g_scaling_factors.initialize<float>(cu, 1, "scalingFactors");
                vector<int> countsVec = {counts[0], counts[1], counts[2]};
                vector<float> spacingVec = {(float)spacing[0], (float)spacing[1], (float)spacing[2]};
                vector<float> scalingVec = {0.0f};
                g_counts.upload(countsVec);
                g_spacing.upload(spacingVec);
                g_scaling_factors.upload(scalingVec);
                numAtoms = 0;  // Causes execute() to return 0.0 early
                hasInitializedKernel = true;
                return;
            }
        } else {
            // Normal (non-tiled) grid generation
            // Generate grid and derivatives (if enabled)
            std::vector<double> derivatives;
            generateGrid(system, nonbondedForce, gridType, receptorAtoms, receptorPositions,
                         ox, oy, oz, vals, derivatives);

            // Note: The CUDA generateGridKernel already applies V -> sign(V) * |V|^(1/n)
            // when invPower != 0.0, so vals already contain transformed values at this point.
            // No CPU-side transform needed for the non-tiled path.

            // Apply arcsinh transform to compress dynamic range before prefiltering.
            // This prevents Gibbs-like ringing in the prefilter for steep potentials (e.g., LJR).
            // The inverse sinh transform is applied in the CUDA evaluation kernel.
            // Note: arcsinh and stored inv_power can be stacked. Arcsinh is applied after
            // inv_power during generation and undone first during evaluation.
            if (arcsinhScale > 0.0f) {
                for (size_t i = 0; i < vals.size(); i++) {
                    vals[i] = std::asinh(vals[i] / (double)arcsinhScale);
                }
            }

            // Apply Gaussian blur in (possibly arcsinh-compressed) space to smooth
            // cap boundary discontinuities before prefiltering.
            double blurSigma = force.getGaussianBlurSigma();
            if (blurSigma > 0.0) {
                gaussianBlur3D(vals, counts[0], counts[1], counts[2], blurSigma);
            }

            // Apply B-spline prefilter so B-spline interpolation is interpolating
            // (passes through original function values at grid nodes).
            // Applied at generation time so prefiltered coefficients are stored in the file.
            int bsplineOrder = force.getBSplinePrefilterOrder();
            if (bsplineOrder > 0) {
                double cReg = force.getAdaptiveRegularization();
                if (cReg > 0.0 && bsplineOrder == 3) {
                    // Adaptive regularized prefilter (GPU-accelerated PCG solver)
                    CudaAdaptivePrefilter prefilter(
                        counts[0], counts[1], counts[2], cReg,
                        force.getRegularizationThreshold(),
                        force.getPrefilterPCGTolerance(),
                        force.getPrefilterMaxIterations());
                    prefilter.apply(vals);
                    std::cout << "  Adaptive prefilter: " << prefilter.getLastIterationCount()
                              << " PCG iterations, residual " << prefilter.getLastResidual() << std::endl;
                } else {
                    bsplinePrefilter3DByOrder(vals, counts[0], counts[1], counts[2], bsplineOrder);
                }
            }

            // Create GridData object with auto-generated values so saveToFile() uses new format
            // Use move semantics to avoid copying large derivative arrays (can be 45+ GB)
            std::shared_ptr<GridData> gridData = std::make_shared<GridData>(
                counts[0], counts[1], counts[2],
                spacing[0], spacing[1], spacing[2]
            );
            gridData->setOrigin(ox, oy, oz);  // Set the grid origin
            gridData->setValues(vals);  // vals is smaller, copy is fine
            if (!derivatives.empty()) {
                gridData->setDerivatives(std::move(derivatives));  // Move to avoid 45GB copy
            }

            // Copy generated values back to GridForce object so saveToFile() and getGridParameters() work
            // Note: We deliberately skip setDerivatives() on GridForce to avoid another 45GB copy
            // The derivatives are accessible through GridData which GridForce holds via setGridData()
            const_cast<GridForce&>(force).setGridValues(vals);

            // Set invPowerMode on the generated grid
            if (invPower != 0.0f) {
                gridData->setInvPower(invPower);
                InvPowerMode mode = static_cast<InvPowerMode>(invPowerMode);
                gridData->setInvPowerMode(mode);
                const_cast<GridForce&>(force).setInvPowerMode(mode, invPower);
            } else {
                gridData->setInvPower(0.0);
                gridData->setInvPowerMode(InvPowerMode::NONE);
                const_cast<GridForce&>(force).setInvPowerMode(InvPowerMode::NONE, 0.0);
            }

            const_cast<GridForce&>(force).setGridData(gridData);
        }
    }

    if (spacing.size() != 3 || counts.size() != 3) {
        throw OpenMMException("GridForce: Grid dimensions must be 3D");
    }

    // Skip vals validation when using tiled input file (vals will be empty, loaded on demand)
    std::string tiledInputFile = force.getTiledInputFile();
    bool usingTiledInput = !tiledInputFile.empty();

    if (!usingTiledInput && vals.size() != (size_t)(counts[0] * counts[1] * counts[2])) {
        throw OpenMMException("GridForce: Number of grid values doesn't match grid dimensions");
    }

    // Skip scaling factor validation during auto-grid generation (no ligand atoms in system yet)
    // Only validate when using grids with actual ligand particles
    if (!force.getAutoGenerateGrid()) {
        // Check if we have the right number of scaling factors
        if (scaling_factors.size() > numAtoms) {
            throw OpenMMException("GridForce: Too many scaling factors provided");
        }
        // If we have fewer, verify the missing ones are virtual sites or dummy particles (mass=0)
        if (scaling_factors.size() < numAtoms) {
            for (int i = scaling_factors.size(); i < numAtoms; i++) {
                double mass = system.getParticleMass(i);
                if (mass != 0.0 && !system.isVirtualSite(i)) {
                    throw OpenMMException("GridForce: Missing scaling factor for particle " +
                                        std::to_string(i) + " (mass=" + std::to_string(mass) + ")");
                }
            }
            // Pad with zeros for verified dummy/virtual particles
            while (scaling_factors.size() < numAtoms) {
                scaling_factors.push_back(0.0);
            }
        }
    } else {
        // Auto-grid generation: pad scaling factors with zeros (no ligand atoms to scale)
        while (scaling_factors.size() < numAtoms) {
            scaling_factors.push_back(0.0);
        }
    }

    // For non-auto-generated grids, apply arcsinh transform + prefilter before upload.
    // (For auto-generated grids, this was already done in the generate block above.)
    // Skip if values were loaded from file (already transformed during original generation).
    if (!force.getAutoGenerateGrid() && !force.getValuesPreTransformed() && !vals.empty()) {
        if (arcsinhScale > 0.0f) {
            for (size_t i = 0; i < vals.size(); i++) {
                vals[i] = std::asinh(vals[i] / (double)arcsinhScale);
            }
        }
        double blurSigma = force.getGaussianBlurSigma();
        if (blurSigma > 0.0) {
            gaussianBlur3D(vals, counts[0], counts[1], counts[2], blurSigma);
        }
        int bsplineOrder = force.getBSplinePrefilterOrder();
        if (bsplineOrder > 0) {
            double cReg = force.getAdaptiveRegularization();
            if (cReg > 0.0 && bsplineOrder == 3) {
                CudaAdaptivePrefilter prefilter(
                    counts[0], counts[1], counts[2], cReg,
                    force.getRegularizationThreshold(),
                    force.getPrefilterPCGTolerance(),
                    force.getPrefilterMaxIterations());
                prefilter.apply(vals);
                std::cout << "  Adaptive prefilter: " << prefilter.getLastIterationCount()
                          << " PCG iterations, residual " << prefilter.getLastResidual() << std::endl;
            } else {
                bsplinePrefilter3DByOrder(vals, counts[0], counts[1], counts[2], bsplineOrder);
            }
        }
    }

    // Initialize arrays
    g_counts.initialize<int>(cu, 3, "gridCounts");
    g_spacing.initialize<float>(cu, 3, "gridSpacing");
    g_scaling_factors.initialize<float>(cu, scaling_factors.size(), "scalingFactors");

    // Copy counts, spacing, and scaling factors to device
    vector<int> countsVec = {counts[0], counts[1], counts[2]};
    vector<float> spacingVec = {(float)spacing[0], (float)spacing[1], (float)spacing[2]};
    vector<float> scalingFloat(scaling_factors.begin(), scaling_factors.end());

    g_counts.upload(countsVec);
    g_spacing.upload(spacingVec);
    g_scaling_factors.upload(scalingFloat);

    // Check tiled mode early - if enabled, we skip GPU caching of full grid
    // (TileManager will handle streaming tiles to GPU on demand)
    bool willUseTiledMode = force.getTiledMode();

    // Auto-enable tiled mode if derivatives won't fit in GPU memory
    // This prevents Hermite/triquintic evaluation from silently returning zeros
    // Check both force.hasDerivatives() and GridData derivatives (auto-generated grids
    // store derivatives on GridData, not directly on force)
    bool hasDerivs = force.hasDerivatives();
    if (!hasDerivs) {
        auto gd = force.getGridData();
        if (gd && gd->hasDerivatives())
            hasDerivs = true;
        if (!hasDerivs) {
            auto cd = force.getCachedGridData();
            if (cd && cd->hasDerivatives())
                hasDerivs = true;
        }
    }
    if (!willUseTiledMode && hasDerivs) {
        size_t freeMem, totalMem;
        cuMemGetInfo(&freeMem, &totalMem);
        size_t derivativesBytes = (size_t)27 * counts[0] * counts[1] * counts[2] * sizeof(float);
        if (derivativesBytes > (size_t)(freeMem * 0.8)) {
            willUseTiledMode = true;
            std::cout << "GridForce: Auto-enabling tiled mode (derivatives need "
                      << (derivativesBytes / (1024.0 * 1024 * 1024)) << " GB, only "
                      << (freeMem / (1024.0 * 1024 * 1024)) << " GB available)" << std::endl;
        }
    }

    // Grid values: check cache first to enable sharing across multiple GridForce instances
    // Priority: TiledInput (no caching needed) > GridData > CachedGridData > vals.data()
    std::shared_ptr<GridData> sharedGridData = force.getGridData();
    std::shared_ptr<CachedGridData> cachedGridData = force.getCachedGridData();
    size_t gridHash = 0;  // Default hash (used for tiled mode)

    if (usingTiledInput) {
        // Tiled input mode: tiles loaded on demand, no full grid caching
        gridHash = std::hash<std::string>{}(tiledInputFile);
    } else if (sharedGridData) {
        // Use GridData pointer as hash for O(1) cache lookup
        gridHash = reinterpret_cast<size_t>(sharedGridData.get());
    } else if (cachedGridData) {
        // Use CachedGridData pointer as hash (for file-loaded grids)
        gridHash = reinterpret_cast<size_t>(cachedGridData.get());
    } else if (!vals.empty()) {
        // Use values vector data pointer as hash (fallback for old code paths)
        gridHash = reinterpret_cast<size_t>(vals.data());
    } else {
        throw OpenMMException("GridForce: No grid values, GridData, or tiled input file provided");
    }

    // Include interpolation method in hash so B-spline (prefiltered) and trilinear
    // (raw) forces sharing the same GridData get separate GPU cache entries
    gridHash ^= std::hash<int>{}(interpolationMethod) + 0x9e3779b9;

    // Only upload full grid to GPU if NOT using tiled mode
    // (Tiled mode streams tiles on demand via TileManager)
    if (!willUseTiledMode) {
        auto cacheKey = std::make_pair((void*)&cu, gridHash);
        auto it = gridCache.find(cacheKey);

        if (it != gridCache.end()) {
            // Try to lock weak_ptr to get shared_ptr
            g_vals_shared = it->second.lock();
            if (g_vals_shared) {
                // Successfully reused cached grid
                vals.clear();
                vals.shrink_to_fit();
            } else {
                // Cached entry expired - remove it and create new one
                gridCache.erase(it);
                vector<float> valsFloat(vals.begin(), vals.end());
                g_vals_shared = std::make_shared<CudaArray>();
                g_vals_shared->initialize<float>(cu, vals.size(), "gridValues");
                g_vals_shared->upload(valsFloat);
                gridCache[cacheKey] = g_vals_shared;
                vals.clear();
                vals.shrink_to_fit();
            }
        } else {
            // First instance with this grid - allocate and cache it
            // First, clean up any expired entries to prevent unbounded cache growth
            for (auto it = gridCache.begin(); it != gridCache.end(); ) {
                if (it->second.expired()) {
                    it = gridCache.erase(it);
                } else {
                    ++it;
                }
            }

            vector<float> valsFloat(vals.begin(), vals.end());
            g_vals_shared = std::make_shared<CudaArray>();
            g_vals_shared->initialize<float>(cu, vals.size(), "gridValues");
            g_vals_shared->upload(valsFloat);
            gridCache[cacheKey] = g_vals_shared;
            vals.clear();
            vals.shrink_to_fit();
        }
    } // end if (!willUseTiledMode)

    // Upload particle indices if filtering is enabled
    if (!particles.empty()) {
        particleIndices.initialize<int>(cu, particles.size(), "particleIndices");
        particleIndices.upload(particles);
    }

    // Upload derivatives if they exist (for tricubic/triquintic interpolation)
    // Skip if using tiled mode - TileManager handles derivative streaming
    // Skip if auto-generating grid (generation context doesn't need derivatives for evaluation)
    // Also skip if derivatives are too large for GPU memory (will need tiled mode for evaluation)
    bool isAutoGenContext = force.getAutoGenerateGrid();
    if (!willUseTiledMode && !isAutoGenContext && force.hasDerivatives()) {
        // Check available GPU memory before attempting to cache derivatives
        size_t freeMem, totalMem;
        cuMemGetInfo(&freeMem, &totalMem);

        vector<double> derivatives_vec = force.getDerivatives();
        const bool useDoubleStorage = force.getUseDoubleStorage();
        const size_t storageElemSize = useDoubleStorage ? sizeof(double) : sizeof(float);
        size_t derivativesBytes = derivatives_vec.size() * storageElemSize;

        // Upload derivatives as double, or down-convert the host buffer to float.
        auto uploadDerivatives = [&](std::shared_ptr<CudaArray>& arr) {
            if (useDoubleStorage) {
                arr->initialize<double>(cu, derivatives_vec.size(), "gridDerivatives");
                // Bypass OpenMM's uploadSubArray which has int overflow on huge arrays.
                CUdeviceptr devPtr = arr->getDevicePointer();
                CUresult uploadResult = cuMemcpyHtoD(devPtr, derivatives_vec.data(),
                    (size_t)derivatives_vec.size() * sizeof(double));
                if (uploadResult != CUDA_SUCCESS)
                    throw OpenMMException("Error uploading large derivative array");
            } else {
                vector<float> derivativesFloat(derivatives_vec.begin(), derivatives_vec.end());
                arr->initialize<float>(cu, derivatives_vec.size(), "gridDerivatives");
                CUdeviceptr devPtr = arr->getDevicePointer();
                CUresult uploadResult = cuMemcpyHtoD(devPtr, derivativesFloat.data(),
                    (size_t)derivativesFloat.size() * sizeof(float));
                if (uploadResult != CUDA_SUCCESS)
                    throw OpenMMException("Error uploading large derivative array");
            }
        };

        // Skip caching if derivatives would use more than 80% of available GPU memory
        if (derivativesBytes > (size_t)(freeMem * 0.8)) {
            std::cout << "GridForce: Derivatives too large for GPU cache ("
                      << (derivativesBytes / (1024.0 * 1024 * 1024)) << " GB needed, "
                      << (freeMem / (1024.0 * 1024 * 1024)) << " GB available). "
                      << "Use tiled mode for evaluation." << std::endl;
            // Clear derivatives_vec to skip caching but still allow save to file
            derivatives_vec.clear();
        }

        // Skip if derivatives vector is empty to avoid CUDA allocation errors
        if (!derivatives_vec.empty()) {
            // Check cache for derivatives (same hash as grid values since they're from same source)
            auto derivCacheKey = std::make_pair((void*)&cu,
                gridHash ^ (useDoubleStorage ? (size_t)0x9E3779B97F4A7C15ULL : (size_t)0));
            auto derivIt = derivativeCache.find(derivCacheKey);

            if (derivIt != derivativeCache.end()) {
                // Try to lock weak_ptr
                g_derivatives_shared = derivIt->second.lock();
                if (g_derivatives_shared) {
                    // Successfully reused cached derivatives
                    derivatives_vec.clear();
                    derivatives_vec.shrink_to_fit();
                } else {
                    // Cached entry expired - remove and create new
                    derivativeCache.erase(derivIt);
                    g_derivatives_shared = std::make_shared<CudaArray>();
                    cu.setAsCurrent();
                    try {
                        uploadDerivatives(g_derivatives_shared);
                        derivativeCache[derivCacheKey] = g_derivatives_shared;
                        derivatives_vec.clear();
                        derivatives_vec.shrink_to_fit();
                    } catch (const std::exception& e) {
                        std::cerr << "ERROR caching derivatives: " << e.what() << std::endl;
                        throw;
                    }
                }
            } else {
                // First instance - allocate and cache derivatives
                // First, clean up any expired entries to prevent unbounded cache growth
                for (auto it = derivativeCache.begin(); it != derivativeCache.end(); ) {
                    if (it->second.expired()) {
                        it = derivativeCache.erase(it);
                    } else {
                        ++it;
                    }
                }

                g_derivatives_shared = std::make_shared<CudaArray>();
                cu.setAsCurrent();
                try {
                    uploadDerivatives(g_derivatives_shared);
                    derivativeCache[derivCacheKey] = g_derivatives_shared;
                    derivatives_vec.clear();
                    derivatives_vec.shrink_to_fit();
                } catch (const std::exception& e) {
                    std::cerr << "ERROR caching derivatives: " << e.what() << std::endl;
                    throw;
                }
            }
        }
    }

    // Store out-of-bounds restraint and interpolation method (gridCap and invPower already set above)
    outOfBoundsRestraint = (float)outOfBounds_k;
    interpolationMethod = interp_method;

    // Store grid origin
    double ox, oy, oz;
    force.getGridOrigin(ox, oy, oz);
    originX = (float)ox;
    originY = (float)oy;
    originZ = (float)oz;

    // Compute effective bounds in grid-local coordinates
    if (force.hasEffectiveBounds()) {
        double ebMinX, ebMinY, ebMinZ, ebMaxX, ebMaxY, ebMaxZ;
        force.getEffectiveBounds(ebMinX, ebMinY, ebMinZ, ebMaxX, ebMaxY, ebMaxZ);
        // Convert from absolute coordinates to grid-local coordinates
        effectiveMinX = (float)(ebMinX - ox);
        effectiveMinY = (float)(ebMinY - oy);
        effectiveMinZ = (float)(ebMinZ - oz);
        effectiveMaxX = (float)(ebMaxX - ox);
        effectiveMaxY = (float)(ebMaxY - oy);
        effectiveMaxZ = (float)(ebMaxZ - oz);
    } else {
        // Default: full grid extent
        effectiveMinX = 0.0f;
        effectiveMinY = 0.0f;
        effectiveMinZ = 0.0f;
        effectiveMaxX = (float)(spacing_local[0] * (counts_local[0] - 1));
        effectiveMaxY = (float)(spacing_local[1] * (counts_local[1] - 1));
        effectiveMaxZ = (float)(spacing_local[2] * (counts_local[2] - 1));
    }

    // Handle particle groups for multi-ligand workflows
    // Flatten all groups into single arrays for efficient single-kernel-launch execution
    // (numParticleGroups already declared above when generating scaling factors)
    if (numParticleGroups > 0) {
        std::vector<int> flattenedParticleIndices;
        std::vector<float> flattenedScalingFactors;

        for (int i = 0; i < numParticleGroups; i++) {
            const ParticleGroup& group = force.getParticleGroup(i);

#if DEBUG_GRIDFORCE
            std::cout << "[DEBUG] Processing group " << i << ": name=" << group.name
                      << ", particleIndices.size()=" << group.particleIndices.size()
                      << ", scalingFactors.size()=" << group.scalingFactors.size() << std::endl;
            if (!group.scalingFactors.empty() && group.scalingFactors.size() <= 5) {
                std::cout << "[DEBUG] Group scaling factors: ";
                for (auto sf : group.scalingFactors) {
                    std::cout << sf << " ";
                }
                std::cout << std::endl;
            }
#endif

            // Append this group's data to flattened arrays
            flattenedParticleIndices.insert(flattenedParticleIndices.end(),
                                           group.particleIndices.begin(),
                                           group.particleIndices.end());

            // If group has explicit scaling factors, use them
            // Otherwise use auto-calculated values from scaling_factors array
            if (!group.scalingFactors.empty()) {
                flattenedScalingFactors.insert(flattenedScalingFactors.end(),
                                              group.scalingFactors.begin(),
                                              group.scalingFactors.end());
            } else {
                // Use auto-calculated scaling factors for this group's particles
                for (int particleIdx : group.particleIndices) {
                    if (particleIdx < scaling_factors.size()) {
                        flattenedScalingFactors.push_back(scaling_factors[particleIdx]);
                    } else {
                        flattenedScalingFactors.push_back(0.0f);
                    }
                }
            }
        }

        totalGroupParticles = flattenedParticleIndices.size();

#if DEBUG_GRIDFORCE
        std::cout << "[DEBUG] Flattened scaling factors for particle groups:" << std::endl;
        std::cout << "  Total particles: " << flattenedScalingFactors.size() << std::endl;
        if (flattenedScalingFactors.size() > 0) {
            std::cout << "  First 5: ";
            for (size_t i = 0; i < std::min((size_t)5, flattenedScalingFactors.size()); i++) {
                std::cout << flattenedScalingFactors[i] << " ";
            }
            std::cout << std::endl;
        }
#endif

        // Upload flattened arrays to GPU
        allGroupParticleIndices.initialize<int>(cu, flattenedParticleIndices.size(), "allGroupParticleIndices");
        allGroupParticleIndices.upload(flattenedParticleIndices);

        allGroupScalingFactors.initialize<float>(cu, flattenedScalingFactors.size(), "allGroupScalingFactors");
        allGroupScalingFactors.upload(flattenedScalingFactors);
    } else {
        totalGroupParticles = 0;
    }

    // Compile kernel from .cu file
    map<string, string> defines;
#if DEBUG_GRIDFORCE
    defines["DEBUG_GRIDFORCE"] = "1";
#endif
    // Read grid derivatives as double. The tiled kernels use their own float buffers.
    if (force.getUseDoubleStorage() && !willUseTiledMode)
        defines["GRID_STORAGE_TYPE"] = "double";
    CUmodule module = cu.createModule(
        CudaGridForceKernelSources::commonHeaders +
        CudaGridForceKernelSources::gridForceKernel +
        CudaGridForceKernelSources::gridForceTiledKernel +
        CudaGridForceKernelSources::gridHessianKernel +
        CudaGridForceKernelSources::gridHessianTiledKernel +
        CudaGridForceKernelSources::gridHessianAnalysisKernel +
        CudaGridForceKernelSources::tileCoverageCheckKernel, defines);
    kernelModule = module;  // Store for deferred kernel extraction (coverage check)
    kernel = cu.getKernel(module, "computeGridForce");
    addGroupEnergiesKernel = cu.getKernel(module, "addGroupEnergiesToTotal");

    // Initialize Hessian kernel (for normal modes analysis)
    // Available for bspline (method 1), triquintic (method 3), and quintic bspline (method 4)
    if (interpolationMethod == 1 || interpolationMethod == 3 || interpolationMethod == 4) {
        hessianKernel = cu.getKernel(module, "computeGridHessian");
        // Allocate Hessian buffer: 6 components per atom
        int hessianNumAtoms = (totalGroupParticles > 0) ? totalGroupParticles : numAtoms;
        if (hessianNumAtoms > 0) {
            hessianBuffer.initialize<float>(cu, 6 * hessianNumAtoms, "hessianBuffer");
        }

        // Initialize analysis kernels for eigendecomposition and metrics
        analysisKernel = cu.getKernel(module, "analyzeHessianKernel");
        sumEntropyKernel = cu.getKernel(module, "sumEntropyKernel");

        // Third derivative kernel (quintic B-spline only)
        if (interpolationMethod == 4) {
            thirdDerivKernel = cu.getKernel(module, "computeGridThirdDerivatives");
            if (hessianNumAtoms > 0) {
                thirdDerivBuffer.initialize<float>(cu, 10 * hessianNumAtoms, "thirdDerivBuffer");
            }
        }
    }
    analysisBuffersInitialized = false;

    // Initialize tiled mode if enabled (or auto-enabled due to memory constraints)
    tiledMode = willUseTiledMode;
    // Note: tiledInputFile already declared earlier in this function

    if (tiledMode) {
        // Compile tiled kernel (uses same module since all kernels are combined)
        tiledKernel = cu.getKernel(module, "computeGridForceTiled");

        // Compile tiled Hessian kernel if Hessian is supported
        if (interpolationMethod == 1 || interpolationMethod == 3 || interpolationMethod == 4) {
            tiledHessianKernel = cu.getKernel(module, "computeGridHessianTiled");
        }

        // Initialize TileManager
        TileConfig tileConfig;
        tileConfig.tileSize = force.getTileSize();
        tileConfig.memoryBudget = (size_t)force.getMemoryBudgetMB() * 1024 * 1024;

        tileManager.reset(new TileManager(cu, tileConfig.memoryBudget));

        // Check if we have a tiled input file (file-backed mode)
        if (!tiledInputFile.empty()) {
            // File-backed mode: load tiles on-demand from the tiled file
            // This is memory-efficient for very large grids
            tileManager->initFromTiledFile(tiledInputFile, tileConfig);
        } else {
            // Memory-backed mode: copy grid values to host storage for tile extraction
            // This requires the grid to fit in host memory
            std::shared_ptr<GridData> gridData = force.getGridData();
            std::shared_ptr<CachedGridData> cachedGridData = force.getCachedGridData();

            if (gridData) {
                hostGridValues.resize(counts[0] * counts[1] * counts[2]);
                const auto& srcVals = gridData->getValues();
                for (size_t i = 0; i < hostGridValues.size() && i < srcVals.size(); i++) {
                    hostGridValues[i] = (float)srcVals[i];
                }
                if (gridData->hasDerivatives()) {
                    const auto& srcDerivs = gridData->getDerivatives();
                    hostGridDerivatives.resize(srcDerivs.size());
                    for (size_t i = 0; i < hostGridDerivatives.size(); i++) {
                        hostGridDerivatives[i] = (float)srcDerivs[i];
                    }
                }
            } else if (cachedGridData) {
                hostGridValues.resize(counts[0] * counts[1] * counts[2]);
                const auto& srcVals = cachedGridData->getOriginalValues();
                for (size_t i = 0; i < hostGridValues.size() && i < srcVals.size(); i++) {
                    hostGridValues[i] = (float)srcVals[i];
                }
                if (cachedGridData->hasDerivatives()) {
                    const auto& srcDerivs = cachedGridData->getOriginalDerivatives();
                    hostGridDerivatives.resize(srcDerivs.size());
                    for (size_t i = 0; i < hostGridDerivatives.size(); i++) {
                        hostGridDerivatives[i] = (float)srcDerivs[i];
                    }
                }
            } else {
                throw OpenMMException("GridForce: Tiled mode requires GridData, CachedGridData, or a tiled input file");
            }

            tileManager->initFromGridData(
                hostGridValues.data(),
                hostGridDerivatives.empty() ? nullptr : hostGridDerivatives.data(),
                counts[0], counts[1], counts[2],
                (float)spacing[0], (float)spacing[1], (float)spacing[2],
                originX, originY, originZ,
                tileConfig
            );
        }
    }

    hasInitializedKernel = true;

#if DEBUG_GRIDFORCE
    std::cout << "[GRIDFORCE DEBUG] ========== INITIALIZATION ==========" << std::endl;
    std::cout << "[CONFIG] invPower=" << invPower << ", invPowerMode=" << invPowerMode
              << " (0=NONE, 1=RUNTIME, 2=STORED)" << std::endl;
    std::cout << "[CONFIG] interpolationMethod=" << interpolationMethod
              << " (0=trilinear, 1=bspline, 2=tricubic, 3=triquintic)" << std::endl;
    std::cout << "[CONFIG] Grid counts: [" << counts[0] << ", " << counts[1] << ", " << counts[2] << "]" << std::endl;
    std::cout << "[CONFIG] Grid spacing: [" << spacing[0] << ", " << spacing[1] << ", " << spacing[2] << "]" << std::endl;
    std::cout << "[CONFIG] Grid origin: [" << originX << ", " << originY << ", " << originZ << "]" << std::endl;
    std::cout << "[CONFIG] numAtoms=" << numAtoms << std::endl;
    std::cout << "[CONFIG] numParticleGroups=" << numParticleGroups << std::endl;
    std::cout << "[CONFIG] totalGroupParticles=" << totalGroupParticles << std::endl;
    std::cout << "[CONFIG] scalingProperty=" << force.getScalingProperty() << std::endl;
    std::cout << "[CONFIG] autoCalculateScalingFactors=" << force.getAutoCalculateScalingFactors() << std::endl;
    // Print first few scaling factors
    std::cout << "[CONFIG] scaling_factors.size()=" << scaling_factors.size() << std::endl;
    if (scaling_factors.size() > 0) {
        std::cout << "[CONFIG] scalingFactors[0]=" << scaling_factors[0] << std::endl;
        if (scaling_factors.size() > 1) {
            std::cout << "[CONFIG] scalingFactors[1]=" << scaling_factors[1] << std::endl;
        }
        if (scaling_factors.size() > 2) {
            std::cout << "[CONFIG] scalingFactors[2]=" << scaling_factors[2] << std::endl;
        }
    }
#endif
}

double CudaCalcGridForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
#if DEBUG_GRIDFORCE
    // Diagnostic: Count kernel calls per instance
    static std::map<void*, int> callCounts;
    callCounts[this]++;
    printf("[FORCE DEBUG] execute() called: this=%p, call #%d, includeForces=%d, includeEnergy=%d\n",
           (void*)this, callCounts[this], includeForces, includeEnergy);
#endif

    if (!hasInitializedKernel) {
        throw OpenMMException("CudaCalcGridForceKernel: Kernel not initialized before execution");
    }

    if (numAtoms == 0) {
        return 0.0;
    }
    // if (scalingCopy.size() > 0) {
    //     std::cout << "  scaling[0] = " << std::scientific << std::setprecision(6)
    //               << scalingCopy[0] << std::endl;
    // }

    // Set common kernel arguments
    int paddedNumAtoms = cu.getPaddedNumAtoms();
    CUdeviceptr posqPtr = cu.getPosq().getDevicePointer();
    CUdeviceptr forcePtr = cu.getLongForceBuffer().getDevicePointer();
    CUdeviceptr countsPtr = g_counts.getDevicePointer();
    CUdeviceptr spacingPtr = g_spacing.getDevicePointer();
    // valsPtr and derivsPtr are only used in non-tiled mode, defer initialization
    CUdeviceptr valsPtr = 0;
    CUdeviceptr derivsPtr = 0;
    if (!tiledMode) {
        valsPtr = (g_vals_shared != nullptr) ? g_vals_shared->getDevicePointer() : g_vals.getDevicePointer();
        derivsPtr = (g_derivatives_shared != nullptr) ? g_derivatives_shared->getDevicePointer() :
                    (g_derivatives.isInitialized() ? g_derivatives.getDevicePointer() : 0);
    }
    CUdeviceptr energyPtr = cu.getEnergyBuffer().getDevicePointer();

    // Execute kernel - single launch for either particle groups or legacy mode
    CUdeviceptr scalingPtr;
    CUdeviceptr particleIndicesPtr;
    int kernelNumAtoms;

    if (totalGroupParticles > 0) {
        // Multi-ligand mode: single kernel launch with all groups' particles flattened
        scalingPtr = allGroupScalingFactors.getDevicePointer();
        particleIndicesPtr = allGroupParticleIndices.getDevicePointer();
        kernelNumAtoms = totalGroupParticles;
    } else {
        // Legacy mode: single execution with global scaling factors and particle filter
        scalingPtr = g_scaling_factors.getDevicePointer();
        particleIndicesPtr = (particleIndices.isInitialized()) ? particleIndices.getDevicePointer() : 0;
        kernelNumAtoms = numAtoms;
    }

    // Clear group energy buffer if using particle groups
    CUdeviceptr particleToGroupMapPtr = 0;
    CUdeviceptr groupEnergyBufferPtr = 0;
    CUdeviceptr groupUnscaledEnergyBufferPtr = 0;
    CUdeviceptr groupScalingFactorsPtr = 0;
    CUdeviceptr atomEnergyBufferPtr = 0;
    CUdeviceptr atomRawEnergyBufferPtr = 0;
    CUdeviceptr outOfBoundsBufferPtr = 0;
    CUdeviceptr groupRuntimeCapsPtr = 0;
    if (numParticleGroups > 0 && groupEnergyBuffer.isInitialized()) {
        // Zero group energy buffers only when energy is needed (async GPU clear
        // instead of synchronous host upload to avoid pipeline stalls during integration)
        if (includeEnergy) {
            cu.clearBuffer(groupEnergyBuffer);
            if (groupUnscaledEnergyBuffer.isInitialized())
                cu.clearBuffer(groupUnscaledEnergyBuffer);
        }
        // Always set device pointers — needed for force computation kernels
        particleToGroupMapPtr = particleToGroupMap.getDevicePointer();
        groupEnergyBufferPtr = groupEnergyBuffer.getDevicePointer();
        if (groupUnscaledEnergyBuffer.isInitialized())
            groupUnscaledEnergyBufferPtr = groupUnscaledEnergyBuffer.getDevicePointer();
        if (groupScalingFactorsBuffer.isInitialized())
            groupScalingFactorsPtr = groupScalingFactorsBuffer.getDevicePointer();
        if (groupRuntimeCapsBuffer.isInitialized())
            groupRuntimeCapsPtr = groupRuntimeCapsBuffer.getDevicePointer();

        // Set up per-atom energy/out-of-bounds buffers
        if (atomEnergyBuffer.isInitialized()) {
            if (includeEnergy)
                cu.clearBuffer(atomEnergyBuffer);
            atomEnergyBufferPtr = atomEnergyBuffer.getDevicePointer();
        }
        if (atomRawEnergyBuffer.isInitialized()) {
            if (includeEnergy)
                cu.clearBuffer(atomRawEnergyBuffer);
            atomRawEnergyBufferPtr = atomRawEnergyBuffer.getDevicePointer();
        }
        if (outOfBoundsBuffer.isInitialized()) {
            if (includeEnergy)
                cu.clearBuffer(outOfBoundsBuffer);
            outOfBoundsBufferPtr = outOfBoundsBuffer.getDevicePointer();
        }
    }

    void* args[] = {
        &posqPtr,
        &forcePtr,
        &countsPtr,
        &spacingPtr,
        &valsPtr,
        &scalingPtr,
        &invPower,
        &invPowerMode,
        &interpolationMethod,
        &outOfBoundsRestraint,
        &originX,
        &originY,
        &originZ,
        &derivsPtr,
        &energyPtr,
        &kernelNumAtoms,
        &paddedNumAtoms,
        &particleIndicesPtr,
        &particleToGroupMapPtr,
        &groupEnergyBufferPtr,
        &groupUnscaledEnergyBufferPtr,
        &atomEnergyBufferPtr,
        &outOfBoundsBufferPtr,
        &numParticleGroups,
        &arcsinhScale,
        &globalScalingFactor,
        &groupScalingFactorsPtr,
        &runtimeCap,
        &groupRuntimeCapsPtr,
        &atomRawEnergyBufferPtr,
        &evaluateInVSpace,
        &effectiveMinX, &effectiveMinY, &effectiveMinZ,
        &effectiveMaxX, &effectiveMaxY, &effectiveMaxZ
    };

#if DEBUG_GRIDFORCE
    std::cout << "[GRIDFORCE DEBUG] ========== EXECUTE ==========" << std::endl;
    std::cout << "[EXEC] Launching kernel with numAtoms=" << kernelNumAtoms << std::endl;
    std::cout << "[EXEC] invPower=" << invPower << ", invPowerMode=" << invPowerMode << std::endl;
    printf("[FORCE DEBUG] Launching computeGridForce kernel (this=%p)\n", (void*)this);
#endif

    if (tiledMode && tileManager) {
        // Tiled execution path with GPU-side tile coverage check.
        // Instead of downloading all positions to CPU every step to determine
        // required tiles, we launch a tiny kernel that checks if all particles
        // are within currently-loaded tiles. Only when a particle escapes do we
        // fall back to the full CPU-side prepareTiles() path.

        bool needsCpuRetile = false;

        if (!tileManager->isCoverageCheckInitialized()) {
            // First call: initialize coverage check buffers and kernel
            tileManager->initCoverageCheck(kernelModule);
            needsCpuRetile = true;  // Must do initial tile loading
        } else {
            // GPU-side coverage check: one tiny kernel + one 4-byte readback
            int coverageNumParticles = (totalGroupParticles > 0) ? totalGroupParticles : kernelNumAtoms;
            CUdeviceptr coverageIndicesPtr = (totalGroupParticles > 0)
                ? allGroupParticleIndices.getDevicePointer() : 0;

            bool allCovered = tileManager->checkCoverageOnGPU(
                posqPtr, coverageIndicesPtr, coverageNumParticles, paddedNumAtoms);

            if (!allCovered) {
                needsCpuRetile = true;
            }
        }

        if (needsCpuRetile) {
            // CPU fallback: download positions and run prepareTiles()
            int totalParticles = cu.getNumAtoms();
            std::vector<float4> posqHost(totalParticles);
            cu.getPosq().download(posqHost);

            std::vector<float> positions;
            if (totalGroupParticles > 0) {
                std::vector<int> groupIndicesHost(totalGroupParticles);
                allGroupParticleIndices.download(groupIndicesHost);
                positions.reserve(totalGroupParticles * 3);
                for (int i = 0; i < totalGroupParticles; i++) {
                    int idx = groupIndicesHost[i];
                    positions.push_back(posqHost[idx].x);
                    positions.push_back(posqHost[idx].y);
                    positions.push_back(posqHost[idx].z);
                }
            } else if (!particles.empty()) {
                positions.reserve(particles.size() * 3);
                for (int idx : particles) {
                    positions.push_back(posqHost[idx].x);
                    positions.push_back(posqHost[idx].y);
                    positions.push_back(posqHost[idx].z);
                }
            } else {
                positions.reserve(numAtoms * 3);
                for (int i = 0; i < numAtoms; i++) {
                    positions.push_back(posqHost[i].x);
                    positions.push_back(posqHost[i].y);
                    positions.push_back(posqHost[i].z);
                }
            }

            if (!tileManager->prepareTiles(positions)) {
                throw OpenMMException("GridForce: Failed to prepare tiles for force computation");
            }

            // Update GPU occupancy bitmap to reflect newly loaded tiles
            tileManager->updateOccupancyBitmap();
        }

        // Get tile lookup table (cast away const for CudaArray::getDevicePointer which is non-const)
        TileLookupTable& lookup = const_cast<TileLookupTable&>(tileManager->getLookupTable());

        CUdeviceptr tileOffsetsPtr = lookup.tileOffsets.getDevicePointer();
        CUdeviceptr tileValuePtrsPtr = lookup.tileValuePtrs.getDevicePointer();
        CUdeviceptr tileDerivPtrsPtr = lookup.tileDerivPtrs.isInitialized() ? lookup.tileDerivPtrs.getDevicePointer() : 0;
        int numTiles = lookup.numLoadedTiles;

        const TileConfig& tileConfig = tileManager->getConfig();
        int tileSizeParam = tileConfig.tileSize;
        int tileOverlapParam = tileConfig.overlap;

        void* tiledArgs[] = {
            &posqPtr,
            &forcePtr,
            &countsPtr,
            &spacingPtr,
            &scalingPtr,
            &invPower,
            &invPowerMode,
            &interpolationMethod,
            &outOfBoundsRestraint,
            &originX,
            &originY,
            &originZ,
            &energyPtr,
            &kernelNumAtoms,
            &paddedNumAtoms,
            &particleIndicesPtr,
            &particleToGroupMapPtr,
            &groupEnergyBufferPtr,
            &groupUnscaledEnergyBufferPtr,
            &atomEnergyBufferPtr,
            &outOfBoundsBufferPtr,
            &numParticleGroups,
            &arcsinhScale,
            &globalScalingFactor,
            &groupScalingFactorsPtr,
            &runtimeCap,
            &groupRuntimeCapsPtr,
            &atomRawEnergyBufferPtr,
            &evaluateInVSpace,
            &tileOffsetsPtr,
            &tileValuePtrsPtr,
            &tileDerivPtrsPtr,
            &numTiles,
            &tileSizeParam,
            &tileOverlapParam,
            &effectiveMinX, &effectiveMinY, &effectiveMinZ,
            &effectiveMaxX, &effectiveMaxY, &effectiveMaxZ
        };

        cu.executeKernel(tiledKernel, tiledArgs, kernelNumAtoms, 256);
    } else {
        // Standard (non-tiled) execution path
        // Use larger block size to avoid OpenMM's thread block limit
        cu.executeKernel(kernel, args, kernelNumAtoms, 256);
    }

#if DEBUG_GRIDFORCE
    cudaDeviceSynchronize();  // Ensure kernel printf output is flushed
#endif

    // When particle groups are used and energy is requested, download group energies
    // and add them to the main energyBuffer. Skipped during force-only integration
    // steps to avoid GPU pipeline stalls from synchronous downloads.
    if (includeEnergy && numParticleGroups > 0 && groupEnergyBuffer.isInitialized()) {
        if (!skipGroupEnergyDownload_) {
            // Download and save group energies (buffer will be zeroed on next execute)
            lastGroupEnergies.resize(numParticleGroups);
            groupEnergyBuffer.download(lastGroupEnergies);

            // Download unscaled group energies
            if (groupUnscaledEnergyBuffer.isInitialized()) {
                lastGroupUnscaledEnergies.resize(numParticleGroups);
                groupUnscaledEnergyBuffer.download(lastGroupUnscaledEnergies);
            }

            // Download per-atom energies if buffer is initialized
            if (atomEnergyBuffer.isInitialized()) {
                lastAtomEnergies.resize(totalGroupParticles);
                atomEnergyBuffer.download(lastAtomEnergies);
            }

            // Download per-atom raw (pre-cap) energies if buffer is initialized
            if (atomRawEnergyBuffer.isInitialized()) {
                lastAtomRawEnergies.resize(totalGroupParticles);
                atomRawEnergyBuffer.download(lastAtomRawEnergies);
            }

            // Download per-atom out-of-bounds flags if buffer is initialized
            if (outOfBoundsBuffer.isInitialized()) {
                lastOutOfBoundsFlags.resize(totalGroupParticles);
                outOfBoundsBuffer.download(lastOutOfBoundsFlags);
            }
        }

        // Use a kernel to sum group energies and add to main energy buffer
        void* sumArgs[] = {
            &energyPtr,
            &groupEnergyBufferPtr,
            &numParticleGroups
        };
        cu.executeKernel(addGroupEnergiesKernel, sumArgs, 1);
    }

    // Debug: Read force buffer AFTER kernel execution
    // std::vector<long long> forcesAfter(num_force_components);
    // cuMemcpyDtoH(forcesAfter.data(), forcePtr, num_force_components * sizeof(long long));

    // // Convert fixed-point to float for atom 0
    // double fx_after = forcesAfter[0] / (double)0x100000000;
    // double fy_after = forcesAfter[paddedNumAtoms] / (double)0x100000000;
    // double fz_after = forcesAfter[2*paddedNumAtoms] / (double)0x100000000;
    // std::cout << "  Force buffer AFTER:  atom0 = (" << fx_after << ", "
    //           << fy_after << ", " << fz_after << ")" << std::endl;

    // // Show the delta
    // double delta_fx = fx_after - fx_before;
    // double delta_fy = fy_after - fy_before;
    // double delta_fz = fz_after - fz_before;
    // std::cout << "  Delta (this kernel):        = (" << delta_fx << ", "
    //           << delta_fy << ", " << delta_fz << ")" << std::endl;

    // Energy is accumulated in the energy buffer by the kernel
    // OpenMM Context will read from the energy buffer
    return 0.0;
}

void CudaCalcGridForceKernel::copyParametersToContext(ContextImpl& contextImpl, const GridForce& force) {
    // For updateParametersInContext, update inv_power, invPowerMode, and interpolation method
    // Grid values, scaling factors, and tiles don't change - they stay cached
    double inv_power = force.getInvPower();
    int inv_power_mode = static_cast<int>(force.getInvPowerMode());
    int interp_method = force.getInterpolationMethod();

    // Update transformation parameters and interpolation method
    invPower = (float)inv_power;
    invPowerMode = inv_power_mode;
    arcsinhScale = (float)force.getArcsinhScale();
    globalScalingFactor = (float)force.getGlobalScalingFactor();
    runtimeCap = (float)force.getRuntimeCap();
    evaluateInVSpace = force.getEvaluateInVSpace() ? 1 : 0;
    interpolationMethod = interp_method;

    // Update per-group scaling factors if groups exist
    if (numParticleGroups > 0 && groupScalingFactorsBuffer.isInitialized()) {
        std::vector<float> groupScalings(numParticleGroups);
        for (int i = 0; i < numParticleGroups; i++) {
            groupScalings[i] = (float)force.getParticleGroupScalingFactor(i);
        }
        groupScalingFactorsBuffer.upload(groupScalings);
    }

    // Update per-group runtime caps if groups exist
    if (numParticleGroups > 0 && groupRuntimeCapsBuffer.isInitialized()) {
        std::vector<float> groupCaps(numParticleGroups);
        for (int i = 0; i < numParticleGroups; i++) {
            groupCaps[i] = (float)force.getParticleGroupRuntimeCap(i);
        }
        groupRuntimeCapsBuffer.upload(groupCaps);
    }
}

vector<double> CudaCalcGridForceKernel::getParticleGroupEnergies() {
    vector<double> groupEnergies;

    // Return the saved group energies from the last execute()
    // Don't download from GPU buffer because it may have been zeroed
    if (numParticleGroups > 0 && !lastGroupEnergies.empty()) {
        groupEnergies.resize(numParticleGroups);
        for (int i = 0; i < numParticleGroups; i++) {
            groupEnergies[i] = (double)lastGroupEnergies[i];
        }
    }

    return groupEnergies;
}

vector<double> CudaCalcGridForceKernel::getParticleGroupUnscaledEnergies() {
    vector<double> groupEnergies;

    if (numParticleGroups > 0 && !lastGroupUnscaledEnergies.empty()) {
        groupEnergies.resize(numParticleGroups);
        for (int i = 0; i < numParticleGroups; i++) {
            groupEnergies[i] = (double)lastGroupUnscaledEnergies[i];
        }
    }

    return groupEnergies;
}

vector<double> CudaCalcGridForceKernel::getParticleAtomEnergies() {
    vector<double> atomEnergies;

    // Return the saved per-atom energies from the last execute()
    if (numParticleGroups > 0 && !lastAtomEnergies.empty()) {
        atomEnergies.resize(lastAtomEnergies.size());
        for (size_t i = 0; i < lastAtomEnergies.size(); i++) {
            atomEnergies[i] = (double)lastAtomEnergies[i];
        }
    }

    return atomEnergies;
}

vector<float> CudaCalcGridForceKernel::getParticleGroupAtomRawEnergies() {
    if (numParticleGroups > 0 && !lastAtomRawEnergies.empty()) {
        return lastAtomRawEnergies;
    }
    return vector<float>();
}

vector<int> CudaCalcGridForceKernel::getParticleOutOfBoundsFlags() {
    vector<int> outOfBoundsFlags;

    // Return the saved per-atom out-of-bounds flags from the last execute()
    if (numParticleGroups > 0 && !lastOutOfBoundsFlags.empty()) {
        outOfBoundsFlags = lastOutOfBoundsFlags;
    }

    return outOfBoundsFlags;
}

void CudaCalcGridForceKernel::computeHessian() {
    // Check if Hessian computation is supported for this interpolation method
    if (interpolationMethod != 1 && interpolationMethod != 3 && interpolationMethod != 4) {
        throw OpenMMException("Hessian computation only supported for bspline (method 1), triquintic (method 3), and quintic bspline (method 4) interpolation");
    }

    if (!hessianBuffer.isInitialized()) {
        throw OpenMMException("Hessian buffer not initialized - initialize kernel first");
    }

    cu.setAsCurrent();

    // Get pointers to GPU arrays
    CUdeviceptr posqPtr = cu.getPosq().getDevicePointer();
    CUdeviceptr hessianPtr = hessianBuffer.getDevicePointer();
    CUdeviceptr countsPtr = g_counts.getDevicePointer();
    CUdeviceptr spacingPtr = g_spacing.getDevicePointer();

    // Determine which scaling factors and particle indices to use
    CUdeviceptr scalingPtr;
    CUdeviceptr particleIndicesPtr;
    int kernelNumAtoms;

    if (totalGroupParticles > 0) {
        // Multi-ligand mode
        scalingPtr = allGroupScalingFactors.getDevicePointer();
        particleIndicesPtr = allGroupParticleIndices.getDevicePointer();
        kernelNumAtoms = totalGroupParticles;
    } else {
        // Legacy mode
        scalingPtr = g_scaling_factors.getDevicePointer();
        particleIndicesPtr = (particleIndices.isInitialized()) ? particleIndices.getDevicePointer() : 0;
        kernelNumAtoms = numAtoms;
    }

    if (tiledMode && tileManager) {
        // Tiled execution path: determine required tiles and launch tiled Hessian kernel

        // Get particle positions from GPU
        int totalParticles = cu.getNumAtoms();
        std::vector<float4> posqHost(totalParticles);
        cu.getPosq().download(posqHost);

        // Extract positions for tile determination
        std::vector<float> positions;
        if (totalGroupParticles > 0) {
            // Multi-ligand mode: use flattened group particle indices
            std::vector<int> groupIndicesHost(totalGroupParticles);
            allGroupParticleIndices.download(groupIndicesHost);
            positions.reserve(totalGroupParticles * 3);
            for (int i = 0; i < totalGroupParticles; i++) {
                int idx = groupIndicesHost[i];
                positions.push_back(posqHost[idx].x);
                positions.push_back(posqHost[idx].y);
                positions.push_back(posqHost[idx].z);
            }
        } else if (!particles.empty()) {
            // Filtered particles mode
            positions.reserve(particles.size() * 3);
            for (int idx : particles) {
                positions.push_back(posqHost[idx].x);
                positions.push_back(posqHost[idx].y);
                positions.push_back(posqHost[idx].z);
            }
        } else {
            // All particles mode
            positions.reserve(numAtoms * 3);
            for (int i = 0; i < numAtoms; i++) {
                positions.push_back(posqHost[i].x);
                positions.push_back(posqHost[i].y);
                positions.push_back(posqHost[i].z);
            }
        }

        // Prepare tiles for Hessian computation
        if (!tileManager->prepareTiles(positions)) {
            throw OpenMMException("GridForce: Failed to prepare tiles for Hessian computation");
        }

        // Get tile lookup table
        TileLookupTable& lookup = const_cast<TileLookupTable&>(tileManager->getLookupTable());

        // Get tile parameters
        CUdeviceptr tileOffsetsPtr = lookup.tileOffsets.getDevicePointer();
        CUdeviceptr tileValuePtrsPtr = lookup.tileValuePtrs.getDevicePointer();
        CUdeviceptr tileDerivPtrsPtr = lookup.tileDerivPtrs.isInitialized() ? lookup.tileDerivPtrs.getDevicePointer() : 0;
        int numTiles = lookup.numLoadedTiles;

        const TileConfig& tileConfig = tileManager->getConfig();
        int tileSizeParam = tileConfig.tileSize;
        int tileOverlapParam = tileConfig.overlap;

        // Get per-group scaling pointers for tiled Hessian
        CUdeviceptr tiledHessGroupScalingPtr = 0;
        CUdeviceptr tiledHessParticleToGroupPtr = 0;
        int tiledHessNumGroups = numParticleGroups;
        if (numParticleGroups > 0 && groupScalingFactorsBuffer.isInitialized() && particleToGroupMap.isInitialized()) {
            tiledHessGroupScalingPtr = groupScalingFactorsBuffer.getDevicePointer();
            tiledHessParticleToGroupPtr = particleToGroupMap.getDevicePointer();
        }

        // Get runtime cap pointers for tiled Hessian
        CUdeviceptr tiledHessGroupRuntimeCapsPtr = 0;
        if (numParticleGroups > 0 && groupRuntimeCapsBuffer.isInitialized() && particleToGroupMap.isInitialized()) {
            tiledHessGroupRuntimeCapsPtr = groupRuntimeCapsBuffer.getDevicePointer();
        }

        // Launch tiled Hessian kernel (with per-group scaling and runtime cap)
        void* tiledArgs[] = {
            &posqPtr,
            &hessianPtr,
            &countsPtr,
            &spacingPtr,
            &scalingPtr,
            &invPower,
            &invPowerMode,
            &interpolationMethod,
            &originX,
            &originY,
            &originZ,
            &kernelNumAtoms,
            &particleIndicesPtr,
            &tileOffsetsPtr,
            &tileValuePtrsPtr,
            &tileDerivPtrsPtr,
            &numTiles,
            &tileSizeParam,
            &tileOverlapParam,
            &arcsinhScale,
            &tiledHessGroupScalingPtr,
            &tiledHessParticleToGroupPtr,
            &tiledHessNumGroups,
            &runtimeCap,
            &tiledHessGroupRuntimeCapsPtr,
            &effectiveMinX, &effectiveMinY, &effectiveMinZ,
            &effectiveMaxX, &effectiveMaxY, &effectiveMaxZ
        };

        // Use larger block size to avoid OpenMM's thread block limit
        cu.executeKernel(tiledHessianKernel, tiledArgs, kernelNumAtoms, 256);
    } else {
        // Standard (non-tiled) execution path
        CUdeviceptr valsPtr = (g_vals_shared != nullptr) ? g_vals_shared->getDevicePointer() : g_vals.getDevicePointer();
        CUdeviceptr derivsPtr = (g_derivatives_shared != nullptr) ? g_derivatives_shared->getDevicePointer() :
                                (g_derivatives.isInitialized() ? g_derivatives.getDevicePointer() : 0);

        // Get per-group scaling pointers for Hessian (must match energy kernel scaling)
        CUdeviceptr hessGroupScalingPtr = 0;
        CUdeviceptr hessParticleToGroupPtr = 0;
        int hessNumGroups = numParticleGroups;
        if (numParticleGroups > 0 && groupScalingFactorsBuffer.isInitialized() && particleToGroupMap.isInitialized()) {
            hessGroupScalingPtr = groupScalingFactorsBuffer.getDevicePointer();
            hessParticleToGroupPtr = particleToGroupMap.getDevicePointer();
        }

        // Get runtime cap pointers for Hessian
        CUdeviceptr hessGroupRuntimeCapsPtr = 0;
        if (numParticleGroups > 0 && groupRuntimeCapsBuffer.isInitialized() && particleToGroupMap.isInitialized()) {
            hessGroupRuntimeCapsPtr = groupRuntimeCapsBuffer.getDevicePointer();
        }

        // Launch Hessian kernel (with invPower, arcsinh chain rule, per-group scaling, and runtime cap)
        void* args[] = {
            &posqPtr,
            &hessianPtr,
            &countsPtr,
            &spacingPtr,
            &valsPtr,
            &scalingPtr,
            &invPower,
            &invPowerMode,
            &interpolationMethod,
            &originX,
            &originY,
            &originZ,
            &derivsPtr,
            &kernelNumAtoms,
            &particleIndicesPtr,
            &arcsinhScale,
            &hessGroupScalingPtr,
            &hessParticleToGroupPtr,
            &hessNumGroups,
            &runtimeCap,
            &hessGroupRuntimeCapsPtr,
            &effectiveMinX, &effectiveMinY, &effectiveMinZ,
            &effectiveMaxX, &effectiveMaxY, &effectiveMaxZ
        };

        // Use larger block size to avoid OpenMM's thread block limit
        cu.executeKernel(hessianKernel, args, kernelNumAtoms, 256);
    }

    // Download results
    lastHessianBlocks.resize(6 * kernelNumAtoms);
    hessianBuffer.download(lastHessianBlocks);
}

vector<double> CudaCalcGridForceKernel::getHessianBlocks() {
    vector<double> hessianBlocks;

    if (!lastHessianBlocks.empty()) {
        hessianBlocks.resize(lastHessianBlocks.size());
        for (size_t i = 0; i < lastHessianBlocks.size(); i++) {
            hessianBlocks[i] = (double)lastHessianBlocks[i];
        }
    }

    return hessianBlocks;
}

void CudaCalcGridForceKernel::computeThirdDerivatives() {
    if (interpolationMethod != 4) {
        throw OpenMMException("Third derivative computation only supported for quintic B-spline (method 4) interpolation");
    }

    if (!thirdDerivBuffer.isInitialized()) {
        throw OpenMMException("Third derivative buffer not initialized - ensure quintic B-spline interpolation is being used");
    }

    cu.setAsCurrent();

    CUdeviceptr posqPtr = cu.getPosq().getDevicePointer();
    CUdeviceptr thirdDerivPtr = thirdDerivBuffer.getDevicePointer();
    CUdeviceptr countsPtr = g_counts.getDevicePointer();
    CUdeviceptr spacingPtr = g_spacing.getDevicePointer();

    CUdeviceptr scalingPtr;
    CUdeviceptr particleIndicesPtr;
    int kernelNumAtoms;

    if (totalGroupParticles > 0) {
        scalingPtr = allGroupScalingFactors.getDevicePointer();
        particleIndicesPtr = allGroupParticleIndices.getDevicePointer();
        kernelNumAtoms = totalGroupParticles;
    } else {
        scalingPtr = g_scaling_factors.getDevicePointer();
        particleIndicesPtr = (particleIndices.isInitialized()) ? particleIndices.getDevicePointer() : 0;
        kernelNumAtoms = numAtoms;
    }

    // Standard (non-tiled) path — tiled third derivatives not yet implemented
    CUdeviceptr valsPtr = (g_vals_shared != nullptr) ? g_vals_shared->getDevicePointer() : g_vals.getDevicePointer();
    CUdeviceptr derivsPtr = (g_derivatives_shared != nullptr) ? g_derivatives_shared->getDevicePointer() :
                            (g_derivatives.isInitialized() ? g_derivatives.getDevicePointer() : 0);

    void* args[] = {
        &posqPtr,
        &thirdDerivPtr,
        &countsPtr,
        &spacingPtr,
        &valsPtr,
        &scalingPtr,
        &invPower,
        &invPowerMode,
        &interpolationMethod,
        &originX,
        &originY,
        &originZ,
        &derivsPtr,
        &kernelNumAtoms,
        &particleIndicesPtr,
        &arcsinhScale
    };

    cu.executeKernel(thirdDerivKernel, args, kernelNumAtoms, 256);

    lastThirdDerivBlocks.resize(10 * kernelNumAtoms);
    thirdDerivBuffer.download(lastThirdDerivBlocks);
}

vector<double> CudaCalcGridForceKernel::getThirdDerivativeBlocks() {
    vector<double> result;
    if (!lastThirdDerivBlocks.empty()) {
        result.resize(lastThirdDerivBlocks.size());
        for (size_t i = 0; i < lastThirdDerivBlocks.size(); i++) {
            result[i] = (double)lastThirdDerivBlocks[i];
        }
    }
    return result;
}

void CudaCalcGridForceKernel::analyzeHessian(float temperature) {
    // Check if Hessian computation is supported for this interpolation method
    if (interpolationMethod != 1 && interpolationMethod != 3) {
        throw OpenMMException("Hessian analysis only supported for bspline (method 1) and triquintic (method 3) interpolation");
    }

    if (lastHessianBlocks.empty()) {
        throw OpenMMException("Must call computeHessian() before analyzeHessian()");
    }

    cu.setAsCurrent();

    // Determine number of atoms
    int kernelNumAtoms = (totalGroupParticles > 0) ? totalGroupParticles : numAtoms;

    // Initialize analysis buffers if needed
    if (!analysisBuffersInitialized && kernelNumAtoms > 0) {
        eigenvaluesBuffer.initialize<float>(cu, 3 * kernelNumAtoms, "eigenvalues");
        eigenvectorsBuffer.initialize<float>(cu, 9 * kernelNumAtoms, "eigenvectors");
        meanCurvatureBuffer.initialize<float>(cu, kernelNumAtoms, "meanCurvature");
        totalCurvatureBuffer.initialize<float>(cu, kernelNumAtoms, "totalCurvature");
        gaussianCurvatureBuffer.initialize<float>(cu, kernelNumAtoms, "gaussianCurvature");
        fracAnisotropyBuffer.initialize<float>(cu, kernelNumAtoms, "fracAnisotropy");
        entropyBuffer.initialize<float>(cu, kernelNumAtoms, "entropy");
        minEigenvalueBuffer.initialize<float>(cu, kernelNumAtoms, "minEigenvalue");
        numNegativeBuffer.initialize<int>(cu, kernelNumAtoms, "numNegative");
        totalEntropyBuffer.initialize<float>(cu, 1, "totalEntropy");
        analysisBuffersInitialized = true;
    }

    if (kernelNumAtoms == 0) {
        return;
    }

    // Calculate kT in kJ/mol
    float kB = 0.008314462618f;  // kJ/(mol·K)
    float kT = kB * temperature;

    // Get device pointers
    CUdeviceptr hessianPtr = hessianBuffer.getDevicePointer();
    CUdeviceptr eigenvaluesPtr = eigenvaluesBuffer.getDevicePointer();
    CUdeviceptr eigenvectorsPtr = eigenvectorsBuffer.getDevicePointer();
    CUdeviceptr meanCurvaturePtr = meanCurvatureBuffer.getDevicePointer();
    CUdeviceptr totalCurvaturePtr = totalCurvatureBuffer.getDevicePointer();
    CUdeviceptr gaussianCurvaturePtr = gaussianCurvatureBuffer.getDevicePointer();
    CUdeviceptr fracAnisotropyPtr = fracAnisotropyBuffer.getDevicePointer();
    CUdeviceptr entropyPtr = entropyBuffer.getDevicePointer();
    CUdeviceptr minEigenvaluePtr = minEigenvalueBuffer.getDevicePointer();
    CUdeviceptr numNegativePtr = numNegativeBuffer.getDevicePointer();
    CUdeviceptr totalEntropyPtr = totalEntropyBuffer.getDevicePointer();

    // Launch analysis kernel
    void* analysisArgs[] = {
        &hessianPtr,
        &eigenvaluesPtr,
        &eigenvectorsPtr,
        &meanCurvaturePtr,
        &totalCurvaturePtr,
        &gaussianCurvaturePtr,
        &fracAnisotropyPtr,
        &entropyPtr,
        &minEigenvaluePtr,
        &numNegativePtr,
        &kT,
        &kernelNumAtoms
    };

    // Use larger block size to avoid OpenMM's thread block limit
    cu.executeKernel(analysisKernel, analysisArgs, kernelNumAtoms, 256);

    // Clear total entropy buffer and sum
    vector<float> zeroEntropy(1, 0.0f);
    totalEntropyBuffer.upload(zeroEntropy);

    int blockSize = 256;
    int numBlocks = (kernelNumAtoms + blockSize - 1) / blockSize;

    void* sumArgs[] = {
        &entropyPtr,
        &totalEntropyPtr,
        &kernelNumAtoms
    };

    // Launch with shared memory for reduction
    CUresult result = cuLaunchKernel(
        sumEntropyKernel,
        numBlocks, 1, 1,
        blockSize, 1, 1,
        blockSize * sizeof(float),
        cu.getCurrentStream(),
        sumArgs,
        NULL
    );

    if (result != CUDA_SUCCESS) {
        throw OpenMMException("Error launching sumEntropyKernel");
    }

    // Download results
    lastEigenvalues.resize(3 * kernelNumAtoms);
    lastEigenvectors.resize(9 * kernelNumAtoms);
    lastMeanCurvature.resize(kernelNumAtoms);
    lastTotalCurvature.resize(kernelNumAtoms);
    lastGaussianCurvature.resize(kernelNumAtoms);
    lastFracAnisotropy.resize(kernelNumAtoms);
    lastEntropy.resize(kernelNumAtoms);
    lastMinEigenvalue.resize(kernelNumAtoms);
    lastNumNegative.resize(kernelNumAtoms);

    eigenvaluesBuffer.download(lastEigenvalues);
    eigenvectorsBuffer.download(lastEigenvectors);
    meanCurvatureBuffer.download(lastMeanCurvature);
    totalCurvatureBuffer.download(lastTotalCurvature);
    gaussianCurvatureBuffer.download(lastGaussianCurvature);
    fracAnisotropyBuffer.download(lastFracAnisotropy);
    entropyBuffer.download(lastEntropy);
    minEigenvalueBuffer.download(lastMinEigenvalue);
    numNegativeBuffer.download(lastNumNegative);

    vector<float> totalEntropyHost(1);
    totalEntropyBuffer.download(totalEntropyHost);
    lastTotalEntropy = totalEntropyHost[0];
}

vector<double> CudaCalcGridForceKernel::getEigenvalues() {
    vector<double> result;
    if (!lastEigenvalues.empty()) {
        result.resize(lastEigenvalues.size());
        for (size_t i = 0; i < lastEigenvalues.size(); i++) {
            result[i] = (double)lastEigenvalues[i];
        }
    }
    return result;
}

vector<double> CudaCalcGridForceKernel::getEigenvectors() {
    vector<double> result;
    if (!lastEigenvectors.empty()) {
        result.resize(lastEigenvectors.size());
        for (size_t i = 0; i < lastEigenvectors.size(); i++) {
            result[i] = (double)lastEigenvectors[i];
        }
    }
    return result;
}

vector<double> CudaCalcGridForceKernel::getMeanCurvature() {
    vector<double> result;
    if (!lastMeanCurvature.empty()) {
        result.resize(lastMeanCurvature.size());
        for (size_t i = 0; i < lastMeanCurvature.size(); i++) {
            result[i] = (double)lastMeanCurvature[i];
        }
    }
    return result;
}

vector<double> CudaCalcGridForceKernel::getTotalCurvature() {
    vector<double> result;
    if (!lastTotalCurvature.empty()) {
        result.resize(lastTotalCurvature.size());
        for (size_t i = 0; i < lastTotalCurvature.size(); i++) {
            result[i] = (double)lastTotalCurvature[i];
        }
    }
    return result;
}

vector<double> CudaCalcGridForceKernel::getGaussianCurvature() {
    vector<double> result;
    if (!lastGaussianCurvature.empty()) {
        result.resize(lastGaussianCurvature.size());
        for (size_t i = 0; i < lastGaussianCurvature.size(); i++) {
            result[i] = (double)lastGaussianCurvature[i];
        }
    }
    return result;
}

vector<double> CudaCalcGridForceKernel::getFracAnisotropy() {
    vector<double> result;
    if (!lastFracAnisotropy.empty()) {
        result.resize(lastFracAnisotropy.size());
        for (size_t i = 0; i < lastFracAnisotropy.size(); i++) {
            result[i] = (double)lastFracAnisotropy[i];
        }
    }
    return result;
}

vector<double> CudaCalcGridForceKernel::getEntropy() {
    vector<double> result;
    if (!lastEntropy.empty()) {
        result.resize(lastEntropy.size());
        for (size_t i = 0; i < lastEntropy.size(); i++) {
            result[i] = (double)lastEntropy[i];
        }
    }
    return result;
}

vector<double> CudaCalcGridForceKernel::getMinEigenvalue() {
    vector<double> result;
    if (!lastMinEigenvalue.empty()) {
        result.resize(lastMinEigenvalue.size());
        for (size_t i = 0; i < lastMinEigenvalue.size(); i++) {
            result[i] = (double)lastMinEigenvalue[i];
        }
    }
    return result;
}

vector<int> CudaCalcGridForceKernel::getNumNegative() {
    return lastNumNegative;
}

double CudaCalcGridForceKernel::getTotalEntropy() {
    return (double)lastTotalEntropy;
}

void CudaCalcGridForceKernel::generateGrid(
    const System& system,
    const NonbondedForce* nonbondedForce,
    const std::string& gridType,
    const std::vector<int>& receptorAtoms,
    const std::vector<Vec3>& receptorPositions,
    double originX, double originY, double originZ,
    std::vector<double>& vals,
    std::vector<double>& derivatives) {

    cu.setAsCurrent();

    // Total grid points
    // Use size_t to avoid any potential overflow issues
    size_t c0_size = static_cast<size_t>(counts[0]);
    size_t c1_size = static_cast<size_t>(counts[1]);
    size_t c2_size = static_cast<size_t>(counts[2]);
//     std::cout << "DEBUG size_t calculation:" << std::endl;
//     std::cout << "  c0_size = " << c0_size << std::endl;
//     std::cout << "  c1_size = " << c1_size << std::endl;
//     std::cout << "  c2_size = " << c2_size << std::endl;
    size_t product01 = c0_size * c1_size;
//     std::cout << "  c0_size * c1_size = " << product01 << std::endl;
    size_t totalPoints_size = product01 * c2_size;
//     std::cout << "  totalPoints_size = " << totalPoints_size << std::endl;
    int totalPoints = static_cast<int>(totalPoints_size);
//     std::cout << "  totalPoints (int) = " << totalPoints << std::endl;
    vals.resize(totalPoints, 0.0);

    // Extract receptor atom parameters
    std::vector<float3> positions(receptorAtoms.size());
    std::vector<float> charges(receptorAtoms.size());
    std::vector<float> sigmas(receptorAtoms.size());
    std::vector<float> epsilons(receptorAtoms.size());

    for (size_t i = 0; i < receptorAtoms.size(); i++) {
        double q, sig, eps;
        nonbondedForce->getParticleParameters(receptorAtoms[i], q, sig, eps);
        positions[i] = make_float3(receptorPositions[i][0], receptorPositions[i][1], receptorPositions[i][2]);
        charges[i] = q;
        sigmas[i] = sig;
        epsilons[i] = eps;
    }

    // Map grid type to integer
    int gridTypeInt = 0;
    if (gridType == "charge") gridTypeInt = 0;
    else if (gridType == "ljr") gridTypeInt = 1;
    else if (gridType == "lja") gridTypeInt = 2;

    // Allocate GPU memory
    CudaArray receptorPos, receptorCharges, receptorSigmas, receptorEpsilons, gridVals;
    receptorPos.initialize<float3>(cu, receptorAtoms.size(), "receptorPositions");
    receptorCharges.initialize<float>(cu, receptorAtoms.size(), "receptorCharges");
    receptorSigmas.initialize<float>(cu, receptorAtoms.size(), "receptorSigmas");
    receptorEpsilons.initialize<float>(cu, receptorAtoms.size(), "receptorEpsilons");
    gridVals.initialize<float>(cu, totalPoints, "gridValues");

    // Upload data to GPU
    receptorPos.upload(positions);
    receptorCharges.upload(charges);
    receptorSigmas.upload(sigmas);
    receptorEpsilons.upload(epsilons);

#if DEBUG_GRIDFORCE
    std::cout << "[GRID GEN] First 3 receptor positions BEFORE upload:" << std::endl;
    for (size_t i = 0; i < std::min((size_t)3, positions.size()); i++) {
        std::cout << "  " << i << ": (" << positions[i].x << ", " << positions[i].y << ", " << positions[i].z << ") nm" << std::endl;
    }
#endif

    // Prepare grid counts and spacing for GPU
    vector<int> gridCountsVec = {counts[0], counts[1], counts[2]};
    vector<float> gridSpacingVec = {(float)spacing[0], (float)spacing[1], (float)spacing[2]};

    CudaArray d_gridCounts, d_gridSpacing;
    d_gridCounts.initialize<int>(cu, 3, "gridCounts");
    d_gridSpacing.initialize<float>(cu, 3, "gridSpacing");
    d_gridCounts.upload(gridCountsVec);
    d_gridSpacing.upload(gridSpacingVec);

    // Generate the derivatives in double when double storage is enabled.
    map<string, string> genDefines;
    if (useDoubleStorage)
        genDefines["GRID_STORAGE_TYPE"] = "double";
    CUmodule module = cu.createModule(
        CudaGridForceKernelSources::commonHeaders +
        CudaGridForceKernelSources::gridGenerationKernel, genDefines);

    // Convert origin to float
    float originXf = (float)originX;
    float originYf = (float)originY;
    float originZf = (float)originZ;
    int numReceptorAtoms = receptorAtoms.size();

    int blockSize = 256;
    CUresult result;

    if (computeDerivatives) {
        // Determine chunk size based on available GPU memory
        // Query free GPU memory
        size_t freeMem, totalMem;
        cuMemGetInfo(&freeMem, &totalMem);

        // Use at most 50% of free memory for the chunk buffer, leave headroom for other allocations
        // (receptor arrays, grid spacing, counts, and other OpenMM context data)
        size_t maxChunkBytes = (size_t)(freeMem * 0.5);

        // Each grid point requires 27 derivative values (double when double storage is on)
        size_t bytesPerPoint = 27 * (useDoubleStorage ? sizeof(double) : sizeof(float));
        int maxPointsPerChunk = maxChunkBytes / bytesPerPoint;

        // Cap chunk size at 50 million points (~5.4 GB) for reasonable kernel times
        maxPointsPerChunk = std::min(maxPointsPerChunk, 50000000);

        // Determine if we need chunking
        bool needsChunking = (totalPoints > maxPointsPerChunk);
        int numChunks = needsChunking ? ((totalPoints + maxPointsPerChunk - 1) / maxPointsPerChunk) : 1;
        int pointsPerChunk = needsChunking ? maxPointsPerChunk : totalPoints;

        if (needsChunking) {
            std::cout << "GridForce: Using chunked generation for " << totalPoints << " points ("
                      << numChunks << " chunks of ~" << pointsPerChunk << " points each)" << std::endl;
            std::cout << "  Available GPU memory: " << (freeMem / (1024.0 * 1024 * 1024)) << " GB" << std::endl;
        }

        // Pre-allocate output arrays (use size_t to avoid overflow for large grids)
        derivatives.resize((size_t)27 * totalPoints);

        // Get analytical derivative kernel
        CUfunction analyticalKernel = cu.getKernel(module, "generateGridWithAnalyticalDerivatives");

        // Map gridType for analytical kernel: 0=charge, 1=ljr, 2=lja
        int analyticalGridType = gridTypeInt;
        float gridCapF = gridCap;
        float invPowerF = invPower;
        int invPowerModeInt = invPowerMode;  // 0=NONE, 1=RUNTIME, 2=STORED

        // Process chunks
        for (int chunk = 0; chunk < numChunks; chunk++) {
            int chunkOffset = chunk * pointsPerChunk;
            int chunkSize = std::min(pointsPerChunk, totalPoints - chunkOffset);

            if (needsChunking) {
                std::cout << "  Processing chunk " << (chunk + 1) << "/" << numChunks
                          << " (points " << chunkOffset << " to " << (chunkOffset + chunkSize - 1) << ")" << std::endl;
            }

            // Allocate this chunk's buffer in the storage precision.
            CudaArray gridDataGPU;
            if (useDoubleStorage)
                gridDataGPU.initialize<double>(cu, 27 * chunkSize, "gridDataChunk");
            else
                gridDataGPU.initialize<float>(cu, 27 * chunkSize, "gridDataChunk");

            int gridSizeKernel = (chunkSize + blockSize - 1) / blockSize;

            void* analyticalArgs[] = {
                &gridDataGPU.getDevicePointer(),
                &receptorPos.getDevicePointer(),
                &receptorCharges.getDevicePointer(),
                &receptorSigmas.getDevicePointer(),
                &receptorEpsilons.getDevicePointer(),
                &numReceptorAtoms,
                &analyticalGridType,
                &gridCapF,
                &invPowerF,
                &invPowerModeInt,
                &originXf,
                &originYf,
                &originZf,
                &d_gridCounts.getDevicePointer(),
                &d_gridSpacing.getDevicePointer(),
                &totalPoints,
                &chunkSize,
                &chunkOffset
            };

            // Launch analytical derivative kernel for this chunk
            result = cuLaunchKernel(analyticalKernel,
                gridSizeKernel, 1, 1,
                blockSize, 1, 1,
                0,
                cu.getCurrentStream(),
                analyticalArgs,
                NULL);

            if (result != CUDA_SUCCESS) {
                throw OpenMMException("Error launching analytical derivative kernel for chunk " + std::to_string(chunk));
            }

            // Synchronize
            cuStreamSynchronize(cu.getCurrentStream());

            // Download chunk data (in the storage precision) and scatter into the
            // final double derivative array.
            vector<double> chunkDataDouble;
            vector<float> chunkDataFloat;
            if (useDoubleStorage) {
                chunkDataDouble.resize(27 * chunkSize);
                gridDataGPU.download(chunkDataDouble);
            } else {
                chunkDataFloat.resize(27 * chunkSize);
                gridDataGPU.download(chunkDataFloat);
            }

            // Copy chunk data to final arrays
            // Layout in chunk: [deriv_idx * chunkSize + localIdx]
            // Layout in final: [deriv_idx * totalPoints + globalIdx]
            // Use size_t to avoid integer overflow for large grids
            for (int derivIdx = 0; derivIdx < 27; derivIdx++) {
                for (int localIdx = 0; localIdx < chunkSize; localIdx++) {
                    size_t globalIdx = (size_t)chunkOffset + localIdx;
                    size_t finalIdx = (size_t)derivIdx * totalPoints + globalIdx;
                    size_t chunkIdx = (size_t)derivIdx * chunkSize + localIdx;
                    derivatives[finalIdx] = useDoubleStorage ? chunkDataDouble[chunkIdx]
                                                             : (double)chunkDataFloat[chunkIdx];
                }
            }
        }

        // Extract energy values from derivatives array (stored at derivIdx=0)
        for (size_t i = 0; i < (size_t)totalPoints; i++) {
            vals[i] = derivatives[i];
        }

        // Diagnostic: check for problematic values in grid energies
        int nan_count = 0, inf_count = 0;
        double min_val = vals[0], max_val = vals[0];
        for (size_t i = 0; i < (size_t)totalPoints; i++) {
            if (std::isnan(vals[i])) nan_count++;
            if (std::isinf(vals[i])) inf_count++;
            if (std::isfinite(vals[i])) {
                min_val = std::min(min_val, vals[i]);
                max_val = std::max(max_val, vals[i]);
            }
        }
//         std::cout << "  Grid energy statistics: min=" << min_val << ", max=" << max_val
//                   << ", NaN=" << nan_count << ", Inf=" << inf_count << std::endl;

//         std::cout << "Generated grid with " << totalPoints << " points and "
//                   << derivatives.size() << " derivative values using RASPA3 analytical method" << std::endl;

    } else {
        // Use original finite difference method (no derivatives)
        // Determine chunk size based on available GPU memory
        size_t freeMem, totalMem;
        cuMemGetInfo(&freeMem, &totalMem);

        // Use at most 50% of free memory for the chunk buffer
        size_t maxChunkBytes = (size_t)(freeMem * 0.5);

        // Each grid point requires 1 float
        int maxPointsPerChunk = maxChunkBytes / sizeof(float);

        // Cap chunk size at 500 million points for reasonable kernel times
        maxPointsPerChunk = std::min(maxPointsPerChunk, 500000000);

        bool needsChunking = (totalPoints > maxPointsPerChunk);
        int numChunks = needsChunking ? ((totalPoints + maxPointsPerChunk - 1) / maxPointsPerChunk) : 1;
        int pointsPerChunk = needsChunking ? maxPointsPerChunk : totalPoints;

        if (needsChunking) {
            std::cout << "GridForce: Using chunked generation for " << totalPoints << " points ("
                      << numChunks << " chunks of ~" << pointsPerChunk << " points each)" << std::endl;
        }

        CUfunction kernel = cu.getKernel(module, "generateGridKernel");
        float gridCapF = gridCap;

#if DEBUG_GRIDFORCE
        std::cout << "[GRID GEN] Using finite difference method" << std::endl;
        std::cout << "[GRID GEN] gridType=" << gridTypeInt << " (0=charge, 1=ljr, 2=lja)" << std::endl;
        std::cout << "[GRID GEN] gridCap=" << gridCapF << " kJ/mol" << std::endl;
        std::cout << "[GRID GEN] invPower=" << invPower << std::endl;
        std::cout << "[GRID GEN] numReceptorAtoms=" << numReceptorAtoms << std::endl;

        // Download and print first few receptor parameters to verify
        std::vector<float> charges_host(std::min(3, numReceptorAtoms));
        std::vector<float> sigmas_host(std::min(3, numReceptorAtoms));
        std::vector<float> epsilons_host(std::min(3, numReceptorAtoms));
        cuMemcpyDtoH(charges_host.data(), receptorCharges.getDevicePointer(), charges_host.size() * sizeof(float));
        cuMemcpyDtoH(sigmas_host.data(), receptorSigmas.getDevicePointer(), sigmas_host.size() * sizeof(float));
        cuMemcpyDtoH(epsilons_host.data(), receptorEpsilons.getDevicePointer(), epsilons_host.size() * sizeof(float));

        std::cout << "[GRID GEN] First 3 receptor atoms on GPU:" << std::endl;
        for (size_t i = 0; i < charges_host.size(); i++) {
            std::cout << "  " << i << ": q=" << charges_host[i]
                      << ", sigma=" << sigmas_host[i]
                      << ", eps=" << epsilons_host[i] << std::endl;
        }
#endif

        // Process chunks
        for (int chunk = 0; chunk < numChunks; chunk++) {
            int chunkOffset = chunk * pointsPerChunk;
            int chunkSize = std::min(pointsPerChunk, totalPoints - chunkOffset);

            if (needsChunking) {
                std::cout << "  Processing chunk " << (chunk + 1) << "/" << numChunks
                          << " (points " << chunkOffset << " to " << (chunkOffset + chunkSize - 1) << ")" << std::endl;
            }

            // Allocate GPU buffer for this chunk
            CudaArray chunkVals;
            chunkVals.initialize<float>(cu, chunkSize, "gridValuesChunk");

            int gridSizeKernel = (chunkSize + blockSize - 1) / blockSize;

            void* args[] = {
                &chunkVals.getDevicePointer(),
                &receptorPos.getDevicePointer(),
                &receptorCharges.getDevicePointer(),
                &receptorSigmas.getDevicePointer(),
                &receptorEpsilons.getDevicePointer(),
                &numReceptorAtoms,
                &gridTypeInt,
                &gridCapF,
                &invPower,
                &originXf,
                &originYf,
                &originZf,
                &d_gridCounts.getDevicePointer(),
                &d_gridSpacing.getDevicePointer(),
                &totalPoints,
                &chunkSize,
                &chunkOffset
            };

            // Launch kernel for this chunk
            result = cuLaunchKernel(kernel,
                gridSizeKernel, 1, 1,
                blockSize, 1, 1,
                0,
                cu.getCurrentStream(),
                args,
                NULL);

            if (result != CUDA_SUCCESS) {
                throw OpenMMException("Error launching grid generation kernel for chunk " + std::to_string(chunk));
            }

            // Synchronize
            cuStreamSynchronize(cu.getCurrentStream());

            // Download chunk results
            vector<float> chunkValsFloat(chunkSize);
            chunkVals.download(chunkValsFloat);

            // Copy to output array
            for (int i = 0; i < chunkSize; i++) {
                vals[chunkOffset + i] = chunkValsFloat[i];
            }
        }
    }
}

void CudaCalcGridForceKernel::generateGridToTiledFile(
    const System& system,
    const NonbondedForce* nonbondedForce,
    const std::string& gridType,
    const std::vector<int>& receptorAtoms,
    const std::vector<Vec3>& receptorPositions,
    double originX, double originY, double originZ,
    const std::string& outputFilename,
    bool computeDerivatives,
    int tileSize) {

    cu.setAsCurrent();

    std::cout << "GridForce: Generating tiled grid to " << outputFilename << std::endl;
    std::cout << "  Grid size: " << counts[0] << "x" << counts[1] << "x" << counts[2] << std::endl;
    std::cout << "  Tile size: " << tileSize << std::endl;
    std::cout << "  Derivatives: " << (computeDerivatives ? "yes" : "no") << std::endl;

    // Create tiled grid file
    TiledGridData tiledGrid(counts[0], counts[1], counts[2],
                            spacing[0], spacing[1], spacing[2], tileSize);
    tiledGrid.setOrigin(originX, originY, originZ);
    tiledGrid.setInvPower(invPower);
    tiledGrid.setInvPowerMode(static_cast<InvPowerMode>(invPowerMode));
    tiledGrid.beginWriting(outputFilename, computeDerivatives);

    int numTilesX = tiledGrid.getNumTilesX();
    int numTilesY = tiledGrid.getNumTilesY();
    int numTilesZ = tiledGrid.getNumTilesZ();
    int totalTiles = tiledGrid.getTotalNumTiles();

    std::cout << "  Tiles: " << numTilesX << "x" << numTilesY << "x" << numTilesZ
              << " = " << totalTiles << " total" << std::endl;

    // Map grid type string to integer
    int gridTypeInt = 0;
    if (gridType == "charge") gridTypeInt = 0;
    else if (gridType == "ljr") gridTypeInt = 1;
    else if (gridType == "lja") gridTypeInt = 2;

    // Upload receptor data to GPU
    int numReceptorAtoms = receptorAtoms.size();
    CudaArray receptorPos, receptorCharges, receptorSigmas, receptorEpsilons;

    std::vector<float3> posVec(numReceptorAtoms);
    std::vector<float> chargesVec(numReceptorAtoms);
    std::vector<float> sigmasVec(numReceptorAtoms);
    std::vector<float> epsilonsVec(numReceptorAtoms);

    for (int i = 0; i < numReceptorAtoms; i++) {
        int atomIndex = receptorAtoms[i];
        posVec[i] = make_float3((float)receptorPositions[i][0],
                                 (float)receptorPositions[i][1],
                                 (float)receptorPositions[i][2]);

        double charge, sigma, epsilon;
        nonbondedForce->getParticleParameters(atomIndex, charge, sigma, epsilon);
        chargesVec[i] = (float)charge;
        sigmasVec[i] = (float)sigma;
        epsilonsVec[i] = (float)epsilon;
    }

    receptorPos.initialize<float3>(cu, numReceptorAtoms, "receptorPositions");
    receptorCharges.initialize<float>(cu, numReceptorAtoms, "receptorCharges");
    receptorSigmas.initialize<float>(cu, numReceptorAtoms, "receptorSigmas");
    receptorEpsilons.initialize<float>(cu, numReceptorAtoms, "receptorEpsilons");

    receptorPos.upload(posVec);
    receptorCharges.upload(chargesVec);
    receptorSigmas.upload(sigmasVec);
    receptorEpsilons.upload(epsilonsVec);

    // Get kernel module
    CUmodule module = cu.createModule(
        CudaGridForceKernelSources::commonHeaders +
        CudaGridForceKernelSources::gridGenerationKernel);

    float originXf = (float)originX;
    float originYf = (float)originY;
    float originZf = (float)originZ;
    float spacingXf = (float)spacing[0];
    float spacingYf = (float)spacing[1];
    float spacingZf = (float)spacing[2];
    float gridCapF = gridCap;
    float invPowerF = invPower;
    int invPowerModeInt = invPowerMode;

    int blockSize = 256;
    CUresult result;

    // Process tiles
    int tilesProcessed = 0;
    auto startTime = std::chrono::high_resolution_clock::now();

    for (int tx = 0; tx < numTilesX; tx++) {
        for (int ty = 0; ty < numTilesY; ty++) {
            for (int tz = 0; tz < numTilesZ; tz++) {
                // Get tile bounds
                int sizeX, sizeY, sizeZ;
                tiledGrid.getTileActualSize(tx, ty, tz, sizeX, sizeY, sizeZ);
                int tilePoints = sizeX * sizeY * sizeZ;

                auto range = tiledGrid.getTileGridRange(tx, ty, tz);
                int startX = range[0], startY = range[1], startZ = range[2];

                if (computeDerivatives) {
                    // Allocate GPU buffer for tile (27 values per point)
                    CudaArray tileDataGPU;
                    tileDataGPU.initialize<float>(cu, 27 * tilePoints, "tileData");

                    CUfunction tileKernel = cu.getKernel(module, "generateTileWithAnalyticalDerivatives");

                    int gridSizeKernel = (tilePoints + blockSize - 1) / blockSize;

                    void* args[] = {
                        &tileDataGPU.getDevicePointer(),
                        &receptorPos.getDevicePointer(),
                        &receptorCharges.getDevicePointer(),
                        &receptorSigmas.getDevicePointer(),
                        &receptorEpsilons.getDevicePointer(),
                        &numReceptorAtoms,
                        &gridTypeInt,
                        &gridCapF,
                        &invPowerF,
                        &invPowerModeInt,
                        &originXf,
                        &originYf,
                        &originZf,
                        &spacingXf,
                        &spacingYf,
                        &spacingZf,
                        &startX,
                        &startY,
                        &startZ,
                        &sizeX,
                        &sizeY,
                        &sizeZ
                    };

                    result = cuLaunchKernel(tileKernel,
                        gridSizeKernel, 1, 1,
                        blockSize, 1, 1,
                        0,
                        cu.getCurrentStream(),
                        args,
                        NULL);

                    if (result != CUDA_SUCCESS) {
                        throw OpenMMException("Error launching tile generation kernel");
                    }

                    cuStreamSynchronize(cu.getCurrentStream());

                    // Download tile data
                    std::vector<float> tileDataFloat(27 * tilePoints);
                    tileDataGPU.download(tileDataFloat);

                    // Extract values and derivatives
                    std::vector<float> values(tilePoints);
                    std::vector<float> derivatives(27 * tilePoints);

                    for (int i = 0; i < tilePoints; i++) {
                        values[i] = tileDataFloat[0 * tilePoints + i];  // First derivative index is energy
                    }
                    for (int d = 0; d < 27; d++) {
                        for (int i = 0; i < tilePoints; i++) {
                            derivatives[d * tilePoints + i] = tileDataFloat[d * tilePoints + i];
                        }
                    }

                    // Write tile to file
                    tiledGrid.writeTile(tx, ty, tz, values, derivatives);

                } else {
                    // Values only
                    CudaArray tileValsGPU;
                    tileValsGPU.initialize<float>(cu, tilePoints, "tileValues");

                    CUfunction tileKernel = cu.getKernel(module, "generateTileKernel");

                    int gridSizeKernel = (tilePoints + blockSize - 1) / blockSize;

                    void* args[] = {
                        &tileValsGPU.getDevicePointer(),
                        &receptorPos.getDevicePointer(),
                        &receptorCharges.getDevicePointer(),
                        &receptorSigmas.getDevicePointer(),
                        &receptorEpsilons.getDevicePointer(),
                        &numReceptorAtoms,
                        &gridTypeInt,
                        &gridCapF,
                        &invPowerF,
                        &originXf,
                        &originYf,
                        &originZf,
                        &spacingXf,
                        &spacingYf,
                        &spacingZf,
                        &startX,
                        &startY,
                        &startZ,
                        &sizeX,
                        &sizeY,
                        &sizeZ
                    };

                    result = cuLaunchKernel(tileKernel,
                        gridSizeKernel, 1, 1,
                        blockSize, 1, 1,
                        0,
                        cu.getCurrentStream(),
                        args,
                        NULL);

                    if (result != CUDA_SUCCESS) {
                        throw OpenMMException("Error launching tile generation kernel");
                    }

                    cuStreamSynchronize(cu.getCurrentStream());

                    // Download and write
                    std::vector<float> values(tilePoints);
                    tileValsGPU.download(values);

                    tiledGrid.writeTile(tx, ty, tz, values);
                }

                tilesProcessed++;

                // Progress update every 100 tiles
                if (tilesProcessed % 100 == 0 || tilesProcessed == totalTiles) {
                    auto now = std::chrono::high_resolution_clock::now();
                    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - startTime).count();
                    double tilesPerSec = tilesProcessed / (elapsed > 0 ? elapsed : 1.0);
                    int remaining = totalTiles - tilesProcessed;
                    double etaSec = remaining / tilesPerSec;
                    std::cout << "  Progress: " << tilesProcessed << "/" << totalTiles
                              << " tiles (" << (100 * tilesProcessed / totalTiles) << "%)"
                              << ", ETA: " << (int)etaSec << "s" << std::endl;
                }
            }
        }
    }

    tiledGrid.finishWriting();

    auto endTime = std::chrono::high_resolution_clock::now();
    auto totalSec = std::chrono::duration_cast<std::chrono::seconds>(endTime - startTime).count();
    std::cout << "  Complete! Generated " << totalTiles << " tiles in " << totalSec << "s" << std::endl;
}
// ============================================================================
// CudaCalcBondedHessianKernel implementation
// ============================================================================

CudaCalcBondedHessianKernel::~CudaCalcBondedHessianKernel() {
}

void CudaCalcBondedHessianKernel::initialize(const System& system) {
    cu.setAsCurrent();
    numAtoms = system.getNumParticles();

    // Extract HarmonicBondForce parameters
    for (int i = 0; i < system.getNumForces(); i++) {
        const HarmonicBondForce* bondForce = dynamic_cast<const HarmonicBondForce*>(&system.getForce(i));
        if (bondForce != nullptr) {
            numBonds = bondForce->getNumBonds();
            if (numBonds > 0) {
                vector<int> h_bondAtoms(numBonds * 2);
                vector<float> h_bondParams(numBonds * 2);

                for (int j = 0; j < numBonds; j++) {
                    int atom1, atom2;
                    double length, k;
                    bondForce->getBondParameters(j, atom1, atom2, length, k);
                    h_bondAtoms[j * 2] = atom1;
                    h_bondAtoms[j * 2 + 1] = atom2;
                    h_bondParams[j * 2] = (float)k;
                    h_bondParams[j * 2 + 1] = (float)length;
                }

                bondAtoms.initialize<int>(cu, numBonds * 2, "bondedHessian_bondAtoms");
                bondParams.initialize<float>(cu, numBonds * 2, "bondedHessian_bondParams");
                bondAtoms.upload(h_bondAtoms);
                bondParams.upload(h_bondParams);
            }
            break;
        }
    }

    // Extract HarmonicAngleForce parameters
    for (int i = 0; i < system.getNumForces(); i++) {
        const HarmonicAngleForce* angleForce = dynamic_cast<const HarmonicAngleForce*>(&system.getForce(i));
        if (angleForce != nullptr) {
            numAngles = angleForce->getNumAngles();
            if (numAngles > 0) {
                vector<int> h_angleAtoms(numAngles * 3);
                vector<float> h_angleParams(numAngles * 2);

                for (int j = 0; j < numAngles; j++) {
                    int atom1, atom2, atom3;
                    double angle, k;
                    angleForce->getAngleParameters(j, atom1, atom2, atom3, angle, k);
                    h_angleAtoms[j * 3] = atom1;
                    h_angleAtoms[j * 3 + 1] = atom2;
                    h_angleAtoms[j * 3 + 2] = atom3;
                    h_angleParams[j * 2] = (float)k;
                    h_angleParams[j * 2 + 1] = (float)angle;
                }

                angleAtoms.initialize<int>(cu, numAngles * 3, "bondedHessian_angleAtoms");
                angleParams.initialize<float>(cu, numAngles * 2, "bondedHessian_angleParams");
                angleAtoms.upload(h_angleAtoms);
                angleParams.upload(h_angleParams);
            }
            break;
        }
    }

    // Extract PeriodicTorsionForce parameters
    for (int i = 0; i < system.getNumForces(); i++) {
        const PeriodicTorsionForce* torsionForce = dynamic_cast<const PeriodicTorsionForce*>(&system.getForce(i));
        if (torsionForce != nullptr) {
            numTorsions = torsionForce->getNumTorsions();
            if (numTorsions > 0) {
                vector<int> h_torsionAtoms(numTorsions * 4);
                vector<float> h_torsionParams(numTorsions * 3);

                for (int j = 0; j < numTorsions; j++) {
                    int atom1, atom2, atom3, atom4, periodicity;
                    double phase, k;
                    torsionForce->getTorsionParameters(j, atom1, atom2, atom3, atom4, periodicity, phase, k);
                    h_torsionAtoms[j * 4] = atom1;
                    h_torsionAtoms[j * 4 + 1] = atom2;
                    h_torsionAtoms[j * 4 + 2] = atom3;
                    h_torsionAtoms[j * 4 + 3] = atom4;
                    h_torsionParams[j * 3] = (float)periodicity;  // n
                    h_torsionParams[j * 3 + 1] = (float)k;
                    h_torsionParams[j * 3 + 2] = (float)phase;
                }

                torsionAtoms.initialize<int>(cu, numTorsions * 4, "bondedHessian_torsionAtoms");
                torsionParams.initialize<float>(cu, numTorsions * 3, "bondedHessian_torsionParams");
                torsionAtoms.upload(h_torsionAtoms);
                torsionParams.upload(h_torsionParams);
            }
            break;
        }
    }

    // Allocate Hessian buffer (3N x 3N)
    int hessianSize = 3 * numAtoms;
    hessianBuffer.initialize<float>(cu, hessianSize * hessianSize, "bondedHessian_hessian");

    // Load CUDA kernels — no per-ligand defines so NVRTC cache hits across systems
    CUmodule module = cu.createModule(
        CudaGridForceKernelSources::commonHeaders +
        CudaGridForceKernelSources::bondedHessianKernel);
    bondHessianKernel = cu.getKernel(module, "computeBondHessians");
    angleHessianKernel = cu.getKernel(module, "computeAngleHessians");
    torsionHessianKernel = cu.getKernel(module, "computeTorsionHessians");
    initHessianKernel = cu.getKernel(module, "initializeHessian");

    hasInitializedKernel = true;
}

std::vector<double> CudaCalcBondedHessianKernel::computeHessian(ContextImpl& context) {
    if (!hasInitializedKernel) {
        throw OpenMMException("CudaCalcBondedHessianKernel: must call initialize() first");
    }

    cu.setAsCurrent();

    int hessianSize = 3 * numAtoms;
    int totalElements = hessianSize * hessianSize;

    // Zero out Hessian buffer
    {
        CUdeviceptr hessianPtr = hessianBuffer.getDevicePointer();
        void* args[] = {&hessianPtr, &totalElements};
        int blockSize = 256;
        int numBlocks = (totalElements + blockSize - 1) / blockSize;
        cu.executeKernel(initHessianKernel, args, numBlocks * blockSize, blockSize);
    }

    // Get positions
    CUdeviceptr posqPtr = cu.getPosq().getDevicePointer();
    CUdeviceptr hessianPtr = hessianBuffer.getDevicePointer();

    // Compute bond Hessians
    if (numBonds > 0) {
        CUdeviceptr bondAtomsPtr = bondAtoms.getDevicePointer();
        CUdeviceptr bondParamsPtr = bondParams.getDevicePointer();

        void* args[] = {
            &posqPtr,
            &bondAtomsPtr,
            &bondParamsPtr,
            &hessianPtr,
            &numBonds,
            &numAtoms
        };

        int blockSize = 128;
        int numBlocks = (numBonds + blockSize - 1) / blockSize;
        cu.executeKernel(bondHessianKernel, args, numBlocks * blockSize, blockSize);
    }

    // Compute angle Hessians
    if (numAngles > 0) {
        CUdeviceptr angleAtomsPtr = angleAtoms.getDevicePointer();
        CUdeviceptr angleParamsPtr = angleParams.getDevicePointer();

        void* args[] = {
            &posqPtr,
            &angleAtomsPtr,
            &angleParamsPtr,
            &hessianPtr,
            &numAngles,
            &numAtoms
        };

        int blockSize = 128;
        int numBlocks = (numAngles + blockSize - 1) / blockSize;
        cu.executeKernel(angleHessianKernel, args, numBlocks * blockSize, blockSize);
    }

    // Compute torsion Hessians
    if (numTorsions > 0) {
        CUdeviceptr torsionAtomsPtr = torsionAtoms.getDevicePointer();
        CUdeviceptr torsionParamsPtr = torsionParams.getDevicePointer();

        void* args[] = {
            &posqPtr,
            &torsionAtomsPtr,
            &torsionParamsPtr,
            &hessianPtr,
            &numTorsions,
            &numAtoms
        };

        int blockSize = 128;
        int numBlocks = (numTorsions + blockSize - 1) / blockSize;
        cu.executeKernel(torsionHessianKernel, args, numBlocks * blockSize, blockSize);
    }

    // Download Hessian and convert to double
    vector<float> h_hessian(totalElements);
    hessianBuffer.download(h_hessian);

    vector<double> result(totalElements);
    for (int i = 0; i < totalElements; i++) {
        result[i] = (double)h_hessian[i];
    }

    return result;
}
