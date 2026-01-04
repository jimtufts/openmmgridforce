#ifndef CUDA_GRIDFORCE_KERNELS_H_
#define CUDA_GRIDFORCE_KERNELS_H_

#include "GridForceKernels.h"
#include "openmm/cuda/CudaContext.h"
#include "openmm/cuda/CudaArray.h"
#include "openmm/NonbondedForce.h"

namespace GridForcePlugin {

/**
 * This kernel is invoked by GridForce to calculate the forces acting on the system and the energy of the system.
 */
class CudaCalcGridForceKernel : public CalcGridForceKernel {
public:
    CudaCalcGridForceKernel(std::string name, const OpenMM::Platform& platform, OpenMM::CudaContext& cu) :
            CalcGridForceKernel(name, platform), hasInitializedKernel(false), cu(cu) {
    }
    ~CudaCalcGridForceKernel();
    /**
     * Initialize the kernel.
     *
     * @param system     the System this kernel will be applied to
     * @param force      the GridForce this kernel will be used for
     */
    void initialize(const OpenMM::System& system, const GridForce& force);
    /**
     * Execute the kernel to calculate the forces and/or energy.
     *
     * @param context        the context in which to execute this kernel
     * @param includeForces  true if forces should be calculated
     * @param includeEnergy  true if the energy should be calculated
     * @return the potential energy due to the force
     */
    double execute(OpenMM::ContextImpl& context, bool includeForces, bool includeEnergy);
    /**
     * Copy changed parameters over to a context.
     *
     * @param context    the context to copy parameters to
     * @param force      the GridForce to copy the parameters from
     */
    void copyParametersToContext(OpenMM::ContextImpl& context, const GridForce& force);
private:
    /**
     * Generate grid values from receptor atoms and NonbondedForce parameters.
     */
    void generateGrid(const OpenMM::System& system,
                     const OpenMM::NonbondedForce* nonbondedForce,
                     const std::string& gridType,
                     const std::vector<int>& receptorAtoms,
                     const std::vector<OpenMM::Vec3>& receptorPositions,
                     double originX, double originY, double originZ,
                     std::vector<double>& vals,
                     std::vector<double>& derivatives);

    bool hasInitializedKernel;
    int numAtoms;
    float invPower;
    int invPowerMode;         // 0=NONE, 1=RUNTIME, 2=STORED
    float gridCap;
    float outOfBoundsRestraint;
    int interpolationMethod;  // 0=trilinear, 1=cubic B-spline, 2=tricubic, 3=quintic Hermite
    float originX, originY, originZ;
    OpenMM::CudaContext& cu;
    OpenMM::CudaArray g_counts;
    OpenMM::CudaArray g_spacing;
    OpenMM::CudaArray g_vals;  // Only used if not sharing
    OpenMM::CudaArray g_scaling_factors;
    OpenMM::CudaArray g_derivatives;  // Only used if not sharing
    OpenMM::CudaArray particleIndices;  // Filtered particle indices (empty = all particles)
    std::shared_ptr<OpenMM::CudaArray> g_vals_shared;        // Shared grid values (when cached)
    std::shared_ptr<OpenMM::CudaArray> g_derivatives_shared; // Shared derivatives (when cached)
    CUfunction kernel;
    std::vector<int> counts;
    std::vector<double> spacing;
    std::vector<int> ligandAtoms;  // Particle indices for ligand atoms
    std::vector<int> particles;    // Filtered particles for evaluation (empty = all particles)
    bool computeDerivatives;       // Whether derivatives were computed
};

} // namespace GridForcePlugin

#endif /*CUDA_GRIDFORCE_KERNELS_H_*/
