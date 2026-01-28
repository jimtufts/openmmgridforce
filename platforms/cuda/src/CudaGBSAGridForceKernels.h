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
    std::vector<double> getGroupBornRadii(int groupIndex) const override;

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

    // Interpolation method (0=trilinear, 1=bspline, 2=tricubic, 3=triquintic)
    int interpolationMethod;

    // Device arrays - grid data
    OpenMM::CudaArray gridCounts;
    OpenMM::CudaArray gridHctProbe;
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
    OpenMM::CudaArray groupEnergies;

    // Device arrays - intermediate results
    OpenMM::CudaArray hctReceptor;   // HCT from grid interpolation
    OpenMM::CudaArray hctLigand;     // HCT from ligand-ligand pairwise
    OpenMM::CudaArray bornRadii;     // Computed Born radii
    OpenMM::CudaArray dE_dR;         // dE/dR_born for chain rule
    OpenMM::CudaArray dE_dHCT;       // dE/dHCT for chain rule

    // CUDA kernels
    CUfunction computeReceptorHCTKernel;
    CUfunction computeLigandHCTKernel;
    CUfunction computeBornRadiiKernel;
    CUfunction computeGBEnergyKernel;
    CUfunction computeSAEnergyKernel;
    CUfunction accumulateSADerivativesKernel;
    CUfunction accumulateBornRadiiDerivativesKernel;
    CUfunction computeHCTChainRuleForcesKernel;
    CUfunction computeReceptorHCTGradientForceKernel;

    // Host-side results cache
    mutable std::vector<float> groupEnergiesHost;
    mutable std::vector<std::vector<float>> groupBornRadiiHost;
};

} // namespace GridForcePlugin

#endif // CUDA_GBSAGRIDFORCE_KERNELS_H_
