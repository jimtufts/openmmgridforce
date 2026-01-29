/* -------------------------------------------------------------------------- *
 *                              OpenMMGridForce                               *
 * -------------------------------------------------------------------------- */

#ifndef CUDA_GBSAGRIDGENERATOR_H_
#define CUDA_GBSAGRIDGENERATOR_H_

#include "DesolvationGrid.h"
#include <cuda.h>
#include <vector>
#include <memory>

namespace GridForcePlugin {

/**
 * CUDA-accelerated generator for GBSA desolvation grids.
 *
 * Generates receptor desolvation energy grids that store the change in
 * receptor GB energy when a probe is placed at each grid point.
 */
class CudaGBSAGridGenerator {
public:
    CudaGBSAGridGenerator();
    ~CudaGBSAGridGenerator();

    /**
     * Generate receptor desolvation energy grid using CUDA.
     *
     * @param receptorPositions  Flat array [x0,y0,z0,x1,y1,z1,...] in nm
     * @param receptorCharges    Partial charges in e
     * @param receptorRadii      Intrinsic radii in nm
     * @param receptorScales     OBC scale factors
     * @param grid               DesolvationGrid with dimensions already set
     * @param probeRadius        Probe radius for S³ scaling (nm)
     * @param probeScale         Probe OBC scale factor
     * @param computeDerivatives Whether to compute derivatives for tricubic/triquintic
     * @param soluteDielectric   Solute dielectric (default 1.0)
     * @param solventDielectric  Solvent dielectric (default 78.5)
     */
    void generateReceptorDesolvationGrid(
        const std::vector<double>& receptorPositions,
        const std::vector<double>& receptorCharges,
        const std::vector<double>& receptorRadii,
        const std::vector<double>& receptorScales,
        std::shared_ptr<DesolvationGrid> grid,
        double probeRadius = 0.14,
        double probeScale = 0.85,
        bool computeDerivatives = false,
        double soluteDielectric = 1.0,
        double solventDielectric = 78.5
    );

    /**
     * Get the baseline receptor GB energy computed during generation.
     */
    double getBaselineReceptorEnergy() const { return baselineEnergy; }

private:
    bool initialized;
    CUcontext cudaContext;
    CUmodule cudaModule;

    // Kernels
    CUfunction computeReceptorReceptorHCTKernel;
    CUfunction computeBaselineReceptorEnergyKernel;
    CUfunction generateReceptorDesolvationGridKernel;
    CUfunction generateReceptorDesolvationGridWithDerivativesKernel;

    double baselineEnergy;

    void initialize();
    void ensureInitialized();
};

} // namespace GridForcePlugin

#endif // CUDA_GBSAGRIDGENERATOR_H_
