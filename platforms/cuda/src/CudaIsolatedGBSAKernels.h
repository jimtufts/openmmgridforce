/* -------------------------------------------------------------------------- *
 *                              OpenMMGridForce                               *
 * -------------------------------------------------------------------------- */

#ifndef CUDA_ISOLATEDGBSAFORCE_KERNELS_H_
#define CUDA_ISOLATEDGBSAFORCE_KERNELS_H_

#include "IsolatedGBSAForceKernels.h"
#include "SolvationFieldGrid.h"
#include "openmm/cuda/CudaContext.h"
#include "openmm/cuda/CudaArray.h"
#include <cuda.h>
#include <cublas_v2.h>
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
    std::vector<double> computeHessianGridFloat(OpenMM::ContextImpl& context);
    std::vector<double> computeHessianPairwiseFloatBuf(OpenMM::ContextImpl& context);

public:

private:
    OpenMM::CudaContext& cu;
    bool hasInitializedKernel;
    int numAtoms;
    int numParticleGroups;

    // GB parameters
    IsolatedGBSAForce::GBMethod gbMethod;
    IsolatedGBSAForce::ReceptorMode receptorMode;
    IsolatedGBSAForce::HessianPrecision hessianPrecision;
    double prefactor;  // -138.935456 * (1/solute - 1/solvent)
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
    IsolatedGBSAForce::CrossMode crossMode;
    IsolatedGBSAForce::MirrorMode mirrorMode;
    int crossTermNumBins;                  // = numAtoms (one bin per template atom)
    OpenMM::CudaArray crossTermGrid;       // [numBins * totalGridPoints] float
    OpenMM::CudaArray crossTermBinRLig;    // [numBins] float

    // GRID-mode receptor add-ons: radius-sliced cross field, mirror field,
    // and the pocket cell list their near shells run off.
    double nearShellCutoff, fieldSwitchOn, fieldSwitchOff, mirrorScale;
    double mirrorFieldCutoff, pocketPadding;
    int fieldInterpolationMethod;
    int numCrossSlices, numMirrorSlices, numPocket;
    int gridCountsHost[3];
    OpenMM::CudaArray crossFieldData;      // [numCrossSlices * totalGridPoints] float
    OpenMM::CudaArray crossSliceRadii;     // [numCrossSlices] float
    OpenMM::CudaArray mirrorFieldData;     // [numMirrorSlices * totalGridPoints] float
    OpenMM::CudaArray mirrorSliceRadii;    // [numMirrorSlices] float
    OpenMM::CudaArray atomMirrorSlice;     // [numAtoms] int
    OpenMM::CudaArray pocketPositions;     // [numPocket] real4
    OpenMM::CudaArray pocketCharges;       // [numPocket] real
    OpenMM::CudaArray pocketRadii;         // [numPocket] real
    OpenMM::CudaArray pocketApoHCT;        // [numPocket] real
    OpenMM::CudaArray pocketBornApo;       // [numPocket] real
    OpenMM::CudaArray pocketWeights;       // [numPocket] real
    OpenMM::CudaArray cellStartArr;        // [nCells + 1] int
    OpenMM::CudaArray cellAtomsArr;        // [numPocket] int, pocket-local indices
    // The mirror near shell reaches only to switchOff, so it gets its own,
    // finer cell list; sharing the cross term's would scan ~30x more atoms.
    OpenMM::CudaArray mirrorCellStartArr;
    OpenMM::CudaArray mirrorCellAtomsArr;
    double mirrorCellOrigin[3];
    double mirrorCellSize;
    int mirrorCellCounts[3];
    OpenMM::CudaArray recDeltaHCT;         // [numGroups * numPocket] real
    OpenMM::CudaArray dCrossDRrec;         // [numGroups * numPocket] real
    OpenMM::CudaArray groupMirrorEnergies; // [numGroups] mixed
    double cellOrigin[3];
    double cellSize;
    int cellCounts[3];
    std::vector<double> groupMirrorEnergiesHost;

    /** Set up the add-on fields, pocket arrays and cell list. */
    void initializeGridReceptorTerms(const IsolatedGBSAForce& force);
    /** Generate a field on the device and hand it back to the force. */
    std::shared_ptr<SolvationFieldGrid> generateCrossFieldOnDevice(
            const IsolatedGBSAForce& force, const std::vector<double>& sliceR,
            const std::vector<double>& bornApo);
    std::shared_ptr<SolvationFieldGrid> generateMirrorFieldOnDevice(
            const IsolatedGBSAForce& force, const std::vector<double>& sliceR,
            const std::vector<double>& weights);

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
    OpenMM::CudaArray receptorReferenceEnergy;  // [1] - scalar reference energy (init-time; kept for compat)
    OpenMM::CudaArray receptorEnergyRef;        // [1] - per-step recomputed reference energy
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
    // PAIRWISE ligand-only pass (for getGroupReceptorContribution): Born radii
    // computed with receptor descreening excluded, plus a zero HCT buffer.
    OpenMM::CudaArray bornRadiiLigOnly;
    OpenMM::CudaArray hctZero;       // always-zero HCT (stand-in for hctReceptor)

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
    // PAIRWISE ligand-only pass outputs (receptorContribution = ligandSelf - ligOnly)
    OpenMM::CudaArray groupLigOnlyEnergies;       // gbEnergyLigOnly*scale (ligand-only radii)
    OpenMM::CudaArray scratchGroupEnergies;       // discard total from the ligand-only pass
    OpenMM::CudaArray scratchForce;               // discard forces from the ligand-only pass

    // Device arrays - per-atom energies
    OpenMM::CudaArray atomEnergies;

    // Hessian computation
    // Mirrors the 5-pass analytical chain in CudaGBSAGridForceKernels.cpp:
    //   1. prepareHessianIntermediates fills hessianDRdPsi / hessianD2RdPsi2
    //      (mode-agnostic OBC-II radius transforms) and reuses dE_dHCT.
    //   2. computeHCTJacobianPairwise fills hessianJacobian = dPsi/dx.
    //   3. computeReceptorPairwiseHessian fills hessianRecD2Psi = d2Psi/dx^2
    //      (6 unique upper-triangle components per atom: xx, yy, zz, xy, xz, yz).
    //   4. computeBornCouplingMatrix fills hessianCouplingMatrix = d2U/dRi dRj.
    //   5. assembleGBSAHessian writes the full 3N x 3N matrix into hessianMatrix.
    // hessianBuffersInitialized gates one-time allocation across calls.
    OpenMM::CudaArray hessianBuffer;             // legacy placeholder; unused
    // Hessian path runs entirely in double. The upstream-shared force
    // buffers (hctReceptor, bornRadii, hctLigand, dE_dR) are float, so
    // for the Hessian we recompute hctReceptor (from the fixed-point
    // accumulator hctReceptorFixed, where the sum is exact to ~2^-32)
    // and Born radii in double. dE_dR stays float — its float
    // precision contributes negligibly to Hessian element accuracy
    // (the J^T M J error analysis showed the dominant noise was in
    // the float hctReceptor sum at Mpro scale).
    OpenMM::CudaArray hessianHctReceptorDouble;  // [N] double — receptor->ligand HCT in double
    OpenMM::CudaArray hessianHctLigandDouble;    // [N] double — ligand-ligand HCT in double
    OpenMM::CudaArray hessianBornRadiiDouble;    // [N] double — OBC2 from above
    OpenMM::CudaArray hessianDRdPsi;             // [N] double
    OpenMM::CudaArray hessianD2RdPsi2;           // [N] double
    OpenMM::CudaArray hessianDEdHCT;             // [N] double
    OpenMM::CudaArray hessianLigGBdEdR;          // [N] real — ligand-GB-only dE/dR (no cross)
    OpenMM::CudaArray hessianJacobian;           // [N * 3N] double
    OpenMM::CudaArray hessianRecD2Psi;           // [N * 6] double
    OpenMM::CudaArray hessianCouplingMatrix;     // [N * N] double
    OpenMM::CudaArray hessianMatrix;             // [3N * 3N] double
    // PAIRWISE receptor-desolvation + cross-term Hessian working buffers.
    // Allocated lazily alongside the rest of the Hessian buffers, only
    // sized/used when numReceptorAtoms > 0 and receptorMode == PAIRWISE.
    OpenMM::CudaArray hessianRecBorn;            // [K*Nr] receptor Born radii
    OpenMM::CudaArray hessianRecDRdPsi;          // [K*Nr] dR^R/dPsi
    OpenMM::CudaArray hessianRecD2RdPsi2;        // [K*Nr] d2R^R/dPsi2
    OpenMM::CudaArray hessianRecDeDR;            // [K*Nr] dE_rec/dR^R
    OpenMM::CudaArray hessianRecJacobian;        // [K*Nr*n3] JR
    OpenMM::CudaArray hessianRecCoupling;        // [K*Nr*Nr] MR
    OpenMM::CudaArray hessianCrossDRL;           // [totalParticles] dE_cross/dR^L
    OpenMM::CudaArray hessianCrossDRR;           // [K*Nr] dE_cross/dR^R
    bool hessianPairwiseBuffersInitialized = false;
    // Zero-filled dummy exclusion buffers: IsolatedGBSAForce has no
    // exclusion API (all pairs contribute by design), but the shared
    // Hessian kernels (ported from GBSAGridForce) read exclusion lists.
    // We give them empty lists so the exclusion check is a no-op.
    OpenMM::CudaArray hessianDummyExclStart;     // [templateN + 1] int (all zeros)
    OpenMM::CudaArray hessianDummyExclAtoms;     // [1] int (never read)
    bool hessianBuffersInitialized = false;
    int hessianNumAtomsCached = 0;
    std::vector<double> hessianFullHost;         // download cache

    // Float intermediates for the GRID-mode Hessian pipeline.
    OpenMM::CudaArray hessianDRdPsiF;
    OpenMM::CudaArray hessianD2RdPsi2F;
    OpenMM::CudaArray hessianDEdHCTF;
    OpenMM::CudaArray hessianJacobianF;
    OpenMM::CudaArray hessianGridHCTHessian;     // d^2 Psi_grid / dx^2, [N * 6]
    OpenMM::CudaArray hessianCouplingMatrixF;
    OpenMM::CudaArray hessianMatrixF;
    bool hessianGridFloatBuffersInitialized = false;
    std::vector<float> hessianGridFloatHost;

    // PAIRWISE HESSIAN_FLOAT: storage-only downgrade. Same kernels as the
    // double path, recompiled from the same source with -DHBUF_T=float so
    // the final dim3N*dim3N hessian buffer accumulates via hardware
    // atomicAdd(float*, float). Compute stays in double. Primarily for
    // speed gains on platforms without hardware atomicAdd(double*, double)
    // (pre-sm_60 Maxwell etc.).
    OpenMM::CudaArray hessianMatrixFloatBuf;
    bool hessianFloatBufBuffersInitialized = false;
    bool hessianFloatBufModuleLoaded = false;
    std::vector<float> hessianFloatBufHost;
    CUfunction assembleGBSAHessianDoubleFloatBufKernel = nullptr;
    CUfunction pairwiseCrossBornDeriv1DoubleFloatBufKernel = nullptr;
    CUfunction pairwiseBornGradHessianDoubleFloatBufKernel = nullptr;
    CUfunction pairwiseOuterProductHessianDoubleFloatBufKernel = nullptr;

    // cuBLAS-accelerated J_R^T M_R J_R desolvation outer product.
    // pairwiseOuterProductHessianDouble used to do this loop on-thread; it
    // dominated the entire pairwise Hessian cost (99% of 66 sec on EA1).
    // Replaced with a host-side cuBLAS dgemm cascade per group + a small
    // scatter kernel. cublasHandle is lazy-init on the first PAIRWISE
    // computeHessian call and torn down in the destructor.
    cublasHandle_t cublasHandle = nullptr;
    bool cublasInitialized = false;
    OpenMM::CudaArray hessianGemmScratchX;       // [Nr * n3] reused across groups
    OpenMM::CudaArray hessianGemmScratchH;       // [n3 * n3] reused across groups
    bool hessianGemmScratchInitialized = false;
    int hessianGemmCachedNr = 0;
    int hessianGemmCachedN3 = 0;
    CUfunction scatterDesolvationHessianGemmKernel = nullptr;
    CUfunction scatterDesolvationHessianGemmFloatBufKernel = nullptr;
    // Stage A: precompute receptor-diagonal weights, fold into M_R so the
    // existing JR^T M_R JR dgemm absorbs the cRj outer product and the
    // dCrossDRR*d2R^R single-Born curvature contribution.
    OpenMM::CudaArray hessianWj;                 // [K * Nr] double
    bool hessianWjInitialized = false;
    int hessianWjCachedKxNr = 0;
    CUfunction pairwiseAccumWjDoubleKernel = nullptr;       // deprecated; superseded by per-pair scalars kernel below
    CUfunction addReceptorDiagToMRDoubleKernel = nullptr;
    // Stage B: per-pair cross-term scalars (cRi, cRiRj, gri, grj, ir*dxyz)
    // computed once per (g, iL, j) and read by the cross-term consumer in
    // pairwiseOuterProductHessianDouble. Replaces 5000+ redundant
    // recomputations per (iL, j) pair in the inner loop.
    OpenMM::CudaArray hessianPairScalars;        // [K * templateN * Nr * 7] packed (7 doubles per pair)
    int hessianPairScalarsCachedKxNxNr = 0;
    bool hessianPairScalarsInitialized = false;
    CUfunction pairwiseComputePerPairScalarsDoubleKernel = nullptr;

    // CUDA kernels
    CUfunction computeReceptorHCTGridKernel;      // Grid interpolation
    CUfunction generateCrossTermGridKernel;       // GRID mode augment: build cross-term scalar field
    CUfunction generateCrossFieldSlicesKernel;
    CUfunction generateMirrorFieldSlicesKernel;
    CUfunction accumulateReceptorNearHCTKernel;
    CUfunction computeCrossRadiusGridEnergyKernel;
    CUfunction applyCrossReceptorChainRuleKernel;
    CUfunction computeMirrorFromFieldKernel;
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
    CUfunction computeReceptorDeltaSAKernel;      // Receptor ΔSA (PAIRWISE only)
    CUfunction accumulateReceptorSADerivativesKernel; // Receptor SA -> receptorDeDR (PAIRWISE)
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
    CUfunction computeReceptorGBEnergyAndDeDRSimpleKernel; // Simple O(N²) fallback
    CUfunction computeReceptorBornRadiiWithLigandKernel; // Runtime: receptor Born radii (with ligand)
    CUfunction precomputeReceptorBornForcesKernel;    // Runtime: precompute bornForces per receptor
    CUfunction accumulateCrossTermReceptorDeDRKernel; // Runtime: add cross-term dE/dR_rec to receptorDeDR
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

    CUfunction computeHessianKernel;                        // legacy placeholder
    // Hessian kernel chain. Double-precision storage variants are loaded
    // lazily in computeHessian() and are the default.
    CUfunction prepareHessianIntermediatesKernel;           // float Hessian path
    CUfunction computeHCTJacobianPairwiseKernel;            // float Hessian path (PAIRWISE)
    CUfunction computeReceptorPairwiseHessianKernel;        // float Hessian path (PAIRWISE)
    CUfunction computeHCTJacobianGridKernel;                // float Hessian path (GRID)
    CUfunction computeReceptorGridHessianKernel;            // float Hessian path (GRID)
    CUfunction computeBornCouplingMatrixKernel;             // float Hessian path
    CUfunction assembleGBSAHessianKernel;                   // float Hessian path
    CUfunction computeLigandGBBornDerivDoubleKernel;        // ligand-GB-only dE/dR (double)
    CUfunction prepareHessianIntermediatesDoubleKernel;     // OBC-II transforms (double)
    CUfunction computeHCTJacobianPairwiseDoubleKernel;      // dPsi/dx (double)
    CUfunction computeReceptorPairwiseHessianDoubleKernel;  // d2Psi/dx2 (double)
    CUfunction computeBornCouplingMatrixDoubleKernel;       // d2U/dRi dRj (double)
    CUfunction assembleGBSAHessianDoubleKernel;             // final 3N x 3N (double)
    CUfunction convertTiledHCTToDoubleKernel;               // fixed-point -> double (unused)
    CUfunction computeBornRadiiOBCDoubleKernel;             // OBC2 transform (double)
    CUfunction computeHctReceptorPairwiseDoubleKernel;      // receptor->ligand HCT in double
    CUfunction computeHctLigandPairwiseDoubleKernel;
    // GRID mode double-storage variants of the receptor-descreening chain.
    CUfunction computeHctReceptorGridDoubleKernel;          // receptor->ligand HCT via grid (double)
    CUfunction computeHCTJacobianGridDoubleKernel;          // dPsi/dx via grid (double)
    CUfunction computeReceptorGridHessianDoubleKernel;      // d2Psi/dx2 via grid (double)
    // PAIRWISE receptor-desolvation + cross-term Hessian kernels (double).
    CUfunction pairwiseRecBornDoubleKernel;                 // recBorn + transform derivs
    CUfunction pairwiseRecCouplingDoubleKernel;             // recDeDR + MR
    CUfunction pairwiseRecJacobianDoubleKernel;             // JR = dPsiR/dx
    CUfunction pairwiseCrossBornDeriv1DoubleKernel;         // dCross_dR* + explicit-r cross
    CUfunction pairwiseBornGradHessianDoubleKernel;         // Born-gradient spatial Hessians
    CUfunction pairwiseOuterProductHessianDoubleKernel;        // ligand-ligand HCT in double

    // Host-side results cache
    mutable std::vector<double> groupEnergiesHost;
    mutable std::vector<double> groupLigandSelfEnergiesHost;
    mutable std::vector<double> groupReceptorContributionsHost;
    mutable std::vector<double> groupReceptorDesolvationsHost;
    mutable std::vector<double> groupCrossTermEnergiesHost;
    mutable std::vector<double> groupUnscaledEnergiesHost;
    mutable std::vector<double> groupLigOnlyEnergiesHost;
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
