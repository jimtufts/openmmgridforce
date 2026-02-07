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

    // Device arrays - receptor desolvation (for PAIRWISE mode)
    OpenMM::CudaArray receptorSelfHCT;          // [N_rec] - receptor-receptor HCT (constant, float)
    OpenMM::CudaArray receptorSelfHCTFixed;     // [N_rec] - receptor-receptor HCT (fixed-point for tiled kernel)
    OpenMM::CudaArray receptorBornRadiiRef;     // [N_rec] - receptor Born radii without ligand
    OpenMM::CudaArray receptorReferenceEnergy;  // [1] - scalar reference energy
    OpenMM::CudaArray ligandToReceptorHCT;      // [N_rec * numGroups] - per-group ligand screening
    OpenMM::CudaArray receptorBornRadii;        // [N_rec] - working buffer for Born radii with ligand
    OpenMM::CudaArray receptorEnergy;           // [1] - working buffer for receptor energy
    OpenMM::CudaArray receptorDeDR;             // [N_rec] - pre-computed dE/dR_born for receptor atoms

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

    // Device arrays - per-group energies
    OpenMM::CudaArray groupEnergies;              // Total energy
    OpenMM::CudaArray groupLigandSelfEnergies;    // Ligand-ligand GB only
    OpenMM::CudaArray groupReceptorContributions; // Receptor effect on ligand
    OpenMM::CudaArray groupReceptorDesolvations;  // Receptor desolvation (PAIRWISE only)
    OpenMM::CudaArray groupCrossTermEnergies;     // Cross-term energy (PAIRWISE only)

    // Device arrays - per-atom energies
    OpenMM::CudaArray atomEnergies;

    // Hessian computation
    OpenMM::CudaArray hessianBuffer;

    // CUDA kernels
    CUfunction computeReceptorHCTGridKernel;      // Grid interpolation
    CUfunction computeReceptorHCTPairwiseKernel;  // Pairwise receptor-ligand
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
    CUfunction computeLigandToReceptorHCTKernel;       // Runtime: ligand screens receptor
    CUfunction computeReceptorBornRadiiWithLigandKernel; // Runtime: receptor Born radii (with ligand)
    CUfunction computeReceptorGBEnergyKernel;          // Runtime: receptor energy (old, slow)
    CUfunction computeReceptorDeDRSimpleKernel;        // Runtime: compute dE/dR_born for receptors
    CUfunction computeCrossTermGBEnergyKernel;         // Runtime: receptor-ligand GB pairs
    CUfunction computeReceptorDesolvationForcesKernel; // Runtime: forces from receptor desolv (old, slow)
    CUfunction computeReceptorDesolvationForcesOptimizedKernel; // Runtime: forces with pre-computed dE/dR
    CUfunction computeCrossTermChainRuleForcesKernel;  // Runtime: forces from cross-term chain rule

    CUfunction computeHessianKernel;

    // Host-side results cache
    mutable std::vector<float> groupEnergiesHost;
    mutable std::vector<float> groupLigandSelfEnergiesHost;
    mutable std::vector<float> groupReceptorContributionsHost;
    mutable std::vector<float> groupReceptorDesolvationsHost;
    mutable std::vector<float> groupCrossTermEnergiesHost;
    mutable std::vector<std::vector<float>> groupBornRadiiHost;
    mutable std::vector<std::vector<float>> groupAtomEnergiesHost;
    mutable std::vector<std::vector<float>> groupReceptorBornRadiiHost;  // PAIRWISE mode only
};

} // namespace GridForcePlugin

#endif // CUDA_ISOLATEDGBSAFORCE_KERNELS_H_
