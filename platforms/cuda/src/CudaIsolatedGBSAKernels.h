/* -------------------------------------------------------------------------- *
 *                              OpenMMGridForce                               *
 * -------------------------------------------------------------------------- */

#ifndef CUDA_ISOLATEDGBSAFORCE_KERNELS_H_
#define CUDA_ISOLATEDGBSAFORCE_KERNELS_H_

#include "IsolatedGBSAForceKernels.h"
#include "openmm/cuda/CudaContext.h"
#include "openmm/cuda/CudaArray.h"
#include <cuda.h>
#include <vector>

namespace GridForcePlugin {

/**
 * CUDA implementation of IsolatedGBSAForce kernel.
 */
class CudaCalcIsolatedGBSAForceKernel : public CalcIsolatedGBSAForceKernel {
public:
    CudaCalcIsolatedGBSAForceKernel(std::string name, const OpenMM::Platform& platform,
                                     OpenMM::CudaContext& cu);
    ~CudaCalcIsolatedGBSAForceKernel();

    void initialize(const OpenMM::System& system, const IsolatedGBSAForce& force) override;
    double execute(OpenMM::ContextImpl& context, bool includeForces, bool includeEnergy) override;
    void updateParametersInContext(OpenMM::ContextImpl& context, const IsolatedGBSAForce& force) override;

    // Per-group energy accessors
    double getGroupEnergy(int groupIndex) const override;
    double getGroupLigandSelfEnergy(int groupIndex) const override;
    double getGroupReceptorContribution(int groupIndex) const override;
    double getGroupReceptorDesolvation(int groupIndex) const override;
    double getGroupCrossTermEnergy(int groupIndex) const override;
    std::vector<double> getGroupBornRadii(int groupIndex) const override;
    std::vector<double> getGroupAtomEnergies(int groupIndex) const override;
    std::vector<double> getReceptorBornRadii(int groupIndex) const override;
    std::vector<double> getParticleGroupUnscaledEnergies() const override;

    // Hessian computation
    std::vector<double> computeHessian(OpenMM::ContextImpl& context) override;

private:
    OpenMM::CudaContext& cu;
    bool hasInitializedKernel;
    int numAtoms;
    int numParticleGroups;

    // GB parameters
    IsolatedGBSAForce::GBMethod gbMethod;
    IsolatedGBSAForce::ReceptorMode receptorMode;
    float prefactor;  // -138.935456 * (1/solute - 1/solvent)
    bool includeSurfaceArea;
    float surfaceTension;
    float cutoffDistance;  // -1.0 for no cutoff
    float receptorLocalityCutoff;  // -1.0 for no locality cutoff (update all receptor atoms)

    // Grid mode parameters (when receptorMode == GRID)
    float originX, originY, originZ;
    float gridSpacing;
    float probeRadius;
    int numBins;
    int interpolationMethod;
    bool hasHctDerivatives;

    // Pairwise mode parameters (when receptorMode == PAIRWISE)
    int numReceptorAtoms;
    float receptorReferenceEnergyValue;  // Cached receptor energy without ligand

    // Device arrays - grid data (for GRID mode)
    OpenMM::CudaArray gridCounts;
    OpenMM::CudaArray gridHctProbe;
    OpenMM::CudaArray gridHctDerivatives;
    OpenMM::CudaArray gridCorrectionN;
    OpenMM::CudaArray gridCorrectionA;
    OpenMM::CudaArray gridCorrectionB;
    OpenMM::CudaArray rThresholds;

    // Device arrays - receptor (for PAIRWISE mode)
    OpenMM::CudaArray receptorPositions;  // float3 array
    OpenMM::CudaArray receptorRadii;
    OpenMM::CudaArray receptorScaleFactors;
    OpenMM::CudaArray receptorCharges;
    OpenMM::CudaArray recBlockBounds;     // float4 array: (cx, cy, cz, radius) per 32-atom block

    // Device arrays - receptor desolvation (for PAIRWISE mode)
    OpenMM::CudaArray receptorSelfHCT;          // [N_rec] - receptor-receptor HCT (constant, float)
    OpenMM::CudaArray receptorSelfHCTFixed;     // [N_rec] - receptor-receptor HCT (fixed-point for tiled kernel)
    OpenMM::CudaArray receptorBornRadiiRef;     // [N_rec] - receptor Born radii without ligand
    OpenMM::CudaArray receptorReferenceEnergy;  // [1] - scalar reference energy
    OpenMM::CudaArray ligandToReceptorHCT;      // [N_rec * numGroups] - per-group ligand screening
    OpenMM::CudaArray receptorBornRadii;        // [K * N_rec] - per-group Born radii with ligand
    OpenMM::CudaArray receptorEnergy;           // [K] - per-group receptor energy working buffer
    OpenMM::CudaArray receptorDeDR;             // [K * N_rec] - per-group dE/dR_born for receptor atoms
    OpenMM::CudaArray receptorBornForces;       // [K * N_rec] - precomputed bornForces per receptor per group
    OpenMM::CudaArray dEdR_crossTerm;           // [totalParticles] - fixed-point dE_cross/dR_born_lig
    OpenMM::CudaArray bornForceLig;             // [totalParticles] - precomputed bornForce for ligand atoms
    OpenMM::CudaArray isActiveRecAtom;           // [K * N_rec] - per-group int mask: 1=active, 0=inactive
    bool fusedHCTComputed_;                       // true if fused kernel already computed ligandToReceptorHCT

    // Fixed-point accumulators for tiled HCT kernel
    OpenMM::CudaArray hctReceptorFixed;             // [totalParticles] - fixed-point receptor→ligand HCT
    OpenMM::CudaArray ligToRecHCTFixed;             // [K * N_rec] - fixed-point ligand→receptor HCT

    // Tile-skip cache (locality cutoff optimization)
    OpenMM::CudaArray hctRecBlockCache;             // [totalParticles * numRecBlocks] - per-block rec→lig HCT
    OpenMM::CudaArray ligToRecHCTCache;             // [K * N_rec] - cached lig→rec HCT
    bool hasTileCache;                              // true after first call builds cache
    bool localityMaskValid;                      // true if cached mask is still valid
    int localityMaskAge;                         // number of execute() calls since last mask recompute

    // Baseline HCT: per-receptor-atom contribution to each ligand atom (locality optimization)
    OpenMM::CudaArray hctReceptorPerAtom;       // [totalParticles * N_rec] - cached per-receptor contributions
    OpenMM::CudaArray hctReceptorBaselineSum;   // [totalParticles] - sum of all baseline values per ligand atom
    bool hasHctBaseline;                         // true after first execute computes baseline

    // Receptor cell list (spatial hash for fast neighbor lookup)
    OpenMM::CudaArray cellAtomIndex;            // [N_rec] - receptor atom indices sorted by cell
    OpenMM::CudaArray cellStart;                // [numCells+1] - start index for each cell in cellAtomIndex
    int cellNx, cellNy, cellNz;                 // cell grid dimensions
    float cellOriginX, cellOriginY, cellOriginZ; // cell grid origin
    float cellSize;                              // cell size (= locality cutoff)
    bool hasCellList;                            // true after cell list is built

    // Device arrays - ligand atom parameters
    OpenMM::CudaArray charges;
    OpenMM::CudaArray radii;
    OpenMM::CudaArray scaleFactors;

    // Device arrays - particle groups
    OpenMM::CudaArray particleIndices;
    OpenMM::CudaArray groupStartIndex;

    // Device arrays - intermediate results
    OpenMM::CudaArray hctReceptor;   // HCT from receptor (grid or pairwise)
    OpenMM::CudaArray hctLigand;     // HCT from ligand-ligand pairwise
    OpenMM::CudaArray bornRadii;     // Computed Born radii
    OpenMM::CudaArray dE_dR;         // dE/dR_born for chain rule

    // Alchemical scaling
    float globalScalingFactor;
    OpenMM::CudaArray groupScalingFactorsBuffer;  // Per-group scaling factors [numGroups]
    std::vector<float> groupScalingFactorsHostCopy;  // Host-side copy for CPU loops

    // Device arrays - per-group energies
    OpenMM::CudaArray groupEnergies;              // Total energy
    OpenMM::CudaArray groupLigandSelfEnergies;    // Ligand-ligand GB only
    OpenMM::CudaArray groupReceptorContributions; // Receptor effect on ligand
    OpenMM::CudaArray groupReceptorDesolvations;  // Receptor desolvation (PAIRWISE only)
    OpenMM::CudaArray groupCrossTermEnergies;     // Cross-term energy (PAIRWISE only)
    OpenMM::CudaArray groupUnscaledEnergies;      // Unscaled total (no per-group scaling)

    // Device arrays - per-atom energies
    OpenMM::CudaArray atomEnergies;

    // Hessian computation
    OpenMM::CudaArray hessianBuffer;

    // CUDA kernels
    CUfunction computeReceptorHCTGridKernel;      // Grid interpolation
    CUfunction computeReceptorHCTPairwiseKernel;  // Pairwise receptor-ligand (naive)
    CUfunction computeReceptorHCTPairwiseTiledKernel;  // Pairwise receptor-ligand (tiled, fast)
    CUfunction computeLigandHCTKernel;            // Ligand-ligand pairwise
    CUfunction computeBornRadiiHCTKernel;         // Raw HCT method
    CUfunction computeBornRadiiOBCKernel;         // OBC-II method
    CUfunction computeGBEnergyKernel;             // GB energy with forces
    CUfunction computeSAEnergyKernel;             // Surface area energy
    CUfunction accumulateBornRadiiDerivativesKernel;
    CUfunction accumulateSADerivativesKernel;
    CUfunction computeHCTChainRuleForcesKernel;
    CUfunction computeReceptorHCTGradientForceKernel;
    CUfunction computeReceptorHCTPairwiseChainRuleKernel;

    // PAIRWISE mode kernels - tiled versions for O(N²) efficiency
    CUfunction computeReceptorSelfHCTKernel;           // Init: receptor-receptor HCT (old, slow)
    CUfunction computeReceptorSelfHCTTiledKernel;      // Init: receptor-receptor HCT (tiled, fast)
    CUfunction convertHCTToFloatKernel;                // Convert fixed-point HCT to float
    CUfunction computeReceptorBornRadiiReferenceKernel; // Init: receptor Born radii (no ligand)
    CUfunction computeReceptorReferenceEnergyKernel;   // Init: receptor energy (no ligand)
    CUfunction computeReceptorGBEnergyTiledKernel;     // Init/Runtime: receptor energy (tiled, fast)
    CUfunction computeReceptorGBEnergyAndDeDRTiledKernel; // Fused energy + dE/dR (tiled)
    CUfunction computeLigandToReceptorHCTKernel;       // Runtime: ligand screens receptor
    CUfunction computeReceptorBornRadiiWithLigandKernel; // Runtime: receptor Born radii (with ligand)
    CUfunction computeReceptorGBEnergyKernel;          // Runtime: receptor energy (old, slow)
    CUfunction computeReceptorDeDRSimpleKernel;        // Runtime: compute dE/dR_born for receptors
    CUfunction computeCrossTermGBEnergyKernel;         // Runtime: receptor-ligand GB pairs
    CUfunction computeReceptorDesolvationForcesKernel; // Runtime: forces from receptor desolv (old, slow)
    CUfunction computeReceptorDesolvationForcesOptimizedKernel; // Runtime: forces with pre-computed bornForces
    CUfunction precomputeReceptorBornForcesKernel;    // Runtime: precompute bornForces per receptor
    CUfunction computeFusedReceptorForcesKernel;     // Runtime: fused desolv + cross-term forces (legacy)
    CUfunction computePairwiseGBForceTiledKernel;   // Runtime: tiled pass 1 (cross-term + desolv + dEdR)
    CUfunction reduceLigandBornForceKernel;          // Runtime: dEdR → bornForceLig
    CUfunction computePairwiseChainRuleTiledKernel;  // Runtime: tiled pass 2 (chain rule)
    CUfunction computeCrossTermChainRuleForcesKernel;  // Runtime: forces from cross-term chain rule

    // Locality cutoff kernels
    CUfunction computeActiveReceptorAtomsKernel;     // Runtime: build active atom mask
    CUfunction computeReceptorEnergyDeltaKernel;     // Runtime: O(|A|*N) delta energy
    CUfunction computeReceptorDeDRActiveKernel;      // Runtime: dE/dR for active atoms only
    CUfunction computeReceptorHCTPerAtomKernel;      // Init: per-receptor HCT contributions
    CUfunction computeBaselineHCTSumKernel;          // Init: sum baseline per ligand atom
    CUfunction reconstructReceptorHCTKernel;         // Runtime: reconstruct HCT (old, O(N_rec))
    CUfunction reconstructReceptorHCTFastKernel;     // Runtime: reconstruct HCT (fast, O(|A|))
    CUfunction computeReceptorHCTCellListKernel;     // Runtime: receptor HCT via cell list
    CUfunction reconstructReceptorHCTCellListKernel; // Runtime: reconstruct HCT via cell list
    CUfunction computeFusedReceptorLigandHCTKernel;  // Runtime: fused bidirectional HCT via warp shuffle
    CUfunction computeReceptorLigandHCTParallelKernel; // Runtime: receptor-threaded bidirectional HCT
    CUfunction computeReceptorLigandHCTTiledKernel;   // Runtime: rectangular tiled bidirectional HCT
    CUfunction convertTiledHCTToFloatKernel;           // Convert fixed-point HCT to float
    CUfunction addDistantHCTFromCacheKernel;          // Reconstruct distant HCT from cache
    CUfunction restoreDistantLigToRecHCTKernel;       // Restore cached lig→rec HCT for distant atoms

    // GPU-side accumulation (eliminate host-device sync)
    CUfunction accumulateDesolvationOnGPUKernel;
    CUfunction accumulateDesolvationDeltaOnGPUKernel;
    CUfunction accumulateCrossTermOnGPUKernel;

    CUfunction computeHessianKernel;

    // Host-side results cache
    mutable std::vector<float> groupEnergiesHost;
    mutable std::vector<float> groupLigandSelfEnergiesHost;
    mutable std::vector<float> groupReceptorContributionsHost;
    mutable std::vector<float> groupReceptorDesolvationsHost;
    mutable std::vector<float> groupCrossTermEnergiesHost;
    mutable std::vector<float> groupUnscaledEnergiesHost;
    mutable std::vector<std::vector<float>> groupBornRadiiHost;
    mutable std::vector<std::vector<float>> groupAtomEnergiesHost;
    mutable std::vector<std::vector<float>> groupReceptorBornRadiiHost;  // PAIRWISE mode only

    bool skipGroupEnergyDownload_ = false;

    // Profiling
    bool profilingEnabled_ = true;
    int profilingCallCount_ = 0;
    static constexpr int PROFILE_CALLS = 3;
public:
    void setSkipGroupEnergyDownload(bool skip) override { skipGroupEnergyDownload_ = skip; }
    void enableProfiling(bool enable) { profilingEnabled_ = enable; }
    void* getGroupEnergyDevicePointer() override {
        return groupEnergies.isInitialized()
            ? (void*)groupEnergies.getDevicePointer() : nullptr;
    }
};

} // namespace GridForcePlugin

#endif // CUDA_ISOLATEDGBSAFORCE_KERNELS_H_
