/* -------------------------------------------------------------------------- *
 *                              OpenMMGridForce                               *
 * -------------------------------------------------------------------------- */

#ifndef CUDA_GBSAGRIDFORCE_KERNELS_H_
#define CUDA_GBSAGRIDFORCE_KERNELS_H_

#include "GBSAGridForceKernels.h"
#include "openmm/cuda/CudaContext.h"
#include "openmm/cuda/CudaArray.h"
#include <cuda.h>
#include <vector>

namespace GridForcePlugin {

/**
 * CUDA implementation of GBSAGridForce kernel.
 */
class CudaCalcGBSAGridForceKernel : public CalcGBSAGridForceKernel {
public:
    CudaCalcGBSAGridForceKernel(std::string name, const OpenMM::Platform& platform,
                                 OpenMM::CudaContext& cu);
    ~CudaCalcGBSAGridForceKernel();

    void initialize(const OpenMM::System& system, const GBSAGridForce& force) override;
    double execute(OpenMM::ContextImpl& context, bool includeForces, bool includeEnergy) override;
    void updateParametersInContext(OpenMM::ContextImpl& context, const GBSAGridForce& force) override;
    double getGroupEnergy(int groupIndex) const override;
    double getGroupLigandDesolvationEnergy(int groupIndex) const override;
    std::vector<double> getGroupBornRadii(int groupIndex) const override;
    void computeHessian(OpenMM::ContextImpl& context) override;
    std::vector<double> getHessianBlocks() const override;
    std::vector<double> getFullHessian() const override;

    /**
     * Generate the desolvation grid on GPU.
     *
     * @param receptorPositions  Receptor atom positions [numReceptorAtoms * 3]
     * @param receptorRadii      Receptor intrinsic radii [numReceptorAtoms]
     * @param receptorScales     Receptor OBC scale factors [numReceptorAtoms]
     * @param numReceptorAtoms   Number of receptor atoms
     * @param probeRadius        Probe intrinsic radius (nm)
     * @param rThresholds        R thresholds for correction bins
     * @param origin             Grid origin [3]
     * @param counts             Grid dimensions [3]
     * @param spacing            Grid spacing (uniform, nm)
     * @param computeDerivatives Whether to compute derivatives (for tricubic/triquintic)
     * @param gridHctProbe       Output: HCT values
     * @param gridCorrectionN    Output: N correction values
     * @param gridCorrectionA    Output: A correction values
     * @param gridCorrectionB    Output: B correction values
     * @param gridDerivatives    Output: Derivatives (if computeDerivatives=true)
     */
    void generateGrid(
        const std::vector<double>& receptorPositions,
        const std::vector<double>& receptorRadii,
        const std::vector<double>& receptorScales,
        int numReceptorAtoms,
        double probeRadius,
        const std::vector<double>& rThresholds,
        const double* origin,
        const int* counts,
        double spacing,
        bool computeDerivatives,
        std::vector<float>& gridHctProbe,
        std::vector<float>& gridCorrectionN,
        std::vector<float>& gridCorrectionA,
        std::vector<float>& gridCorrectionB,
        std::vector<float>& gridDerivatives
    );

private:
    OpenMM::CudaContext& cu;
    bool hasInitializedKernel;
    int numAtoms;
    int numParticleGroups;

    // Grid parameters
    float originX, originY, originZ;
    float gridSpacing;
    float probeRadius;
    int numBins;

    // Solvent parameters
    float prefactor;  // -138.935456 * (1/solute - 1/solvent)
    bool includeSurfaceArea;
    float surfaceTension;

    // KDE correction parameters
    float kdeThreshold;   // Distance threshold for KDE correction (nm)
    float kdeBandwidth;   // KDE sigmoid bandwidth (nm)
    float kdeEpsilonB;    // Smoothing parameter for B grid 1/sqrt(r²+ε²)

    // Interpolation method (0=trilinear, 1=bspline, 2=tricubic, 3=triquintic)
    int interpolationMethod;

    // Correction grid format flags:
    // - useKDECorrections: True when using high-order interpolation for corrections
    // - hasBinnedKDEDerivatives: True when corrections are binned with 27 derivs per bin
    //   Layout: [bin * 27 * numPoints + deriv * numPoints + point]
    //   When false but useKDECorrections=true: pure KDE format [27 * numPoints] (deprecated)
    bool useKDECorrections;
    bool hasBinnedKDEDerivatives;  // New: binned corrections with derivatives

    // Device arrays - grid data
    OpenMM::CudaArray gridCounts;
    OpenMM::CudaArray gridHctProbe;
    OpenMM::CudaArray gridHctDerivatives;  // HCT derivatives for tricubic/triquintic
    bool hasHctDerivatives;                // Whether derivatives are available
    OpenMM::CudaArray gridCorrectionN;
    OpenMM::CudaArray gridCorrectionA;
    OpenMM::CudaArray gridCorrectionB;
    OpenMM::CudaArray rThresholds;

    // Device arrays - atom parameters
    OpenMM::CudaArray charges;
    OpenMM::CudaArray radii;
    OpenMM::CudaArray scaleFactors;

    // Device arrays - exclusions
    OpenMM::CudaArray exclusionAtoms;
    OpenMM::CudaArray exclusionStartIndex;

    // Device arrays - particle groups
    OpenMM::CudaArray particleIndices;
    OpenMM::CudaArray groupStartIndex;
    OpenMM::CudaArray groupEnergies;              // Total energy
    OpenMM::CudaArray groupLigandEnergies;        // Ligand desolvation

    // Device arrays - intermediate results
    OpenMM::CudaArray hctReceptor;   // HCT from grid interpolation
    OpenMM::CudaArray hctLigand;     // HCT from ligand-ligand pairwise
    OpenMM::CudaArray bornRadii;     // Computed Born radii
    OpenMM::CudaArray dE_dR;         // dE/dR_born for chain rule
    OpenMM::CudaArray dE_dHCT;       // dE/dHCT for chain rule

    // CUDA kernels - runtime evaluation
    CUfunction computeReceptorHCTKernel;
    CUfunction computeLigandHCTKernel;
    CUfunction computeBornRadiiKernel;
    CUfunction computeGBEnergyKernel;
    CUfunction computeSAEnergyKernel;
    CUfunction accumulateSADerivativesKernel;
    CUfunction accumulateBornRadiiDerivativesKernel;
    CUfunction computeHCTChainRuleForcesKernel;
    CUfunction computeReceptorHCTGradientForceKernel;

    // CUDA kernels - grid generation (legacy)
    CUfunction generateLigandHCTGridKernel;
    CUfunction generateLigandHCTGridWithCorrectionsKernel;
    CUfunction generateLigandHCTGridWithDerivativesKernel;
    CUfunction generateBinnedGridsWithKDEKernel;  // Binned grids with KDE smoothing
    CUfunction generateBinnedGridsWithKDEDerivativesKernel;  // Binned grids with KDE + 27 derivs per bin

    // CUDA kernels - 4-grid generation with analytical derivatives
    CUfunction generateDesolvationGrids4Kernel;
    CUfunction generateHCTProbeGridKernel;
    CUmodule generationModule;

    // Host-side results cache
    mutable std::vector<float> groupEnergiesHost;
    mutable std::vector<float> groupLigandEnergiesHost;
    mutable std::vector<std::vector<float>> groupBornRadiiHost;

    // Hessian support
    std::vector<double> lastHessianBlocks;   // Per-atom [6 * N]
    std::vector<double> lastFullHessian;     // Full [3N * 3N]
    int hessianNumAtoms;                     // N used in last computation

    // Analytical Hessian GPU buffers
    OpenMM::CudaArray hessianDRdPsi;         // dR_born/dΨ per atom [N]
    OpenMM::CudaArray hessianD2RdPsi2;       // d²R_born/dΨ² per atom [N]
    OpenMM::CudaArray hessianJacobian;       // HCT Jacobian J[N * 3N]
    OpenMM::CudaArray hessianCouplingMatrix;  // Born coupling M[N * N]
    OpenMM::CudaArray hessianGridHCTHessian;  // Receptor grid HCT Hessian [N * 6]
    OpenMM::CudaArray hessianMatrix;          // Full Hessian H[3N * 3N]
    bool hessianBuffersInitialized;

    // Analytical Hessian CUDA kernels
    CUfunction prepareHessianIntermediatesKernel;
    CUfunction computeHCTJacobianKernel;
    CUfunction computeReceptorGridHessianKernel;
    CUfunction computeBornCouplingMatrixKernel;
    CUfunction assembleGBSAHessianKernel;
};

} // namespace GridForcePlugin

#endif // CUDA_GBSAGRIDFORCE_KERNELS_H_
