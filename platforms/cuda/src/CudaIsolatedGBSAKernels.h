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
    bool useKDECorrections;
    bool hasBinnedKDEDerivatives;

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

    // Cross-term scalar-field grid (GRID mode augment)
    bool computeCrossTermGrid;
    int crossTermNumBins;                  // = numAtoms (one bin per template atom)
    OpenMM::CudaArray crossTermGrid;       // [numBins * totalGridPoints] float
    OpenMM::CudaArray crossTermBinRLig;    // [numBins] float

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
    bool fusedHCTComputed_;                       // true if fused kernel already computed ligandToReceptorHCT

    // Fixed-point accumulators for tiled HCT kernel
    OpenMM::CudaArray hctReceptorFixed;             // [totalParticles] - fixed-point receptor→ligand HCT
    OpenMM::CudaArray ligToRecHCTFixed;             // [K * N_rec] - fixed-point ligand→receptor HCT

    // Tile-skip cache (locality cutoff optimization)
    OpenMM::CudaArray hctRecBlockCache;             // [totalParticles * numRecBlocks] - per-block rec→lig HCT
    OpenMM::CudaArray ligToRecHCTCache;             // [K * N_rec] - cached lig→rec HCT
    OpenMM::CudaArray crossTermBlockCache;          // [K * numRecBlocks] - per-tile cross-term energy
    bool hasTileCache;                              // true after first call builds cache
    bool hasCrossTermCache;                         // true after first call builds cross-term cache

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
    CUfunction generateCrossTermGridKernel;       // GRID mode augment: build cross-term scalar field
    CUfunction computeCrossTermFromGridKernel;    // GRID mode augment: runtime eval (phase 3)
    CUfunction computeCrossTermPairwiseKernel;    // GRID mode augment: direct pair-sum cross term
    CUfunction accumulateCrossTermBornDerivativesKernel; // GRID mode: dE_cross/dR_born chain rule
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
    CUfunction computeReceptorSelfHCTTiledKernel;      // Init: receptor-receptor HCT (tiled, fast)
    CUfunction convertHCTToFloatKernel;                // Convert fixed-point HCT to float
    CUfunction computeReceptorBornRadiiReferenceKernel; // Init: receptor Born radii (no ligand)
    CUfunction computeReceptorGBEnergyTiledKernel;     // Init/Runtime: receptor energy (tiled, fast)
    CUfunction computeReceptorGBEnergyAndDeDRTiledKernel; // Fused energy + dE/dR (tiled)
    CUfunction computeReceptorBornRadiiWithLigandKernel; // Runtime: receptor Born radii (with ligand)
    CUfunction precomputeReceptorBornForcesKernel;    // Runtime: precompute bornForces per receptor
    CUfunction computeFusedReceptorForcesKernel;     // Runtime: fused desolv + cross-term forces (legacy)
    CUfunction computePairwiseGBForceTiledKernel;   // Runtime: tiled pass 1 (cross-term + desolv + dEdR)
    CUfunction reduceLigandBornForceKernel;          // Runtime: dEdR → bornForceLig
    CUfunction computePairwiseChainRuleTiledKernel;  // Runtime: tiled pass 2 (chain rule)
    CUfunction computeReceptorLigandHCTTiledKernel;   // Runtime: rectangular tiled bidirectional HCT
    CUfunction convertTiledHCTToFloatKernel;           // Convert fixed-point HCT to float
    CUfunction addDistantHCTFromCacheKernel;          // Reconstruct distant HCT from cache
    CUfunction restoreDistantLigToRecHCTKernel;       // Restore cached lig→rec HCT for distant atoms
    CUfunction addDistantCrossTermFromCacheKernel;    // Reconstruct distant cross-term energy

    // GPU-side accumulation (eliminate host-device sync)
    CUfunction accumulateDesolvationOnGPUKernel;
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

public:
    void setSkipGroupEnergyDownload(bool skip) override { skipGroupEnergyDownload_ = skip; }
    void* getGroupEnergyDevicePointer() override {
        return groupEnergies.isInitialized()
            ? (void*)groupEnergies.getDevicePointer() : nullptr;
    }
};

} // namespace GridForcePlugin

#endif // CUDA_ISOLATEDGBSAFORCE_KERNELS_H_
