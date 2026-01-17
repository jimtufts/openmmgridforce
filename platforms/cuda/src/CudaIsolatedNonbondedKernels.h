#ifndef CUDA_ISOLATEDNONBONDEDFORCE_KERNELS_H_
#define CUDA_ISOLATEDNONBONDEDFORCE_KERNELS_H_

#include "IsolatedNonbondedForceKernels.h"
#include "openmm/cuda/CudaContext.h"
#include "openmm/cuda/CudaArray.h"
#include <vector>

namespace GridForcePlugin {

/**
 * This kernel is invoked by IsolatedNonbondedForce to calculate the forces acting on the system and the energy of the system.
 */
class CudaCalcIsolatedNonbondedForceKernel : public CalcIsolatedNonbondedForceKernel {
public:
    CudaCalcIsolatedNonbondedForceKernel(std::string name, const OpenMM::Platform& platform, OpenMM::CudaContext& cu) :
            CalcIsolatedNonbondedForceKernel(name, platform), hasInitializedKernel(false), cu(cu) {
    }
    ~CudaCalcIsolatedNonbondedForceKernel();

    /**
     * Initialize the kernel.
     *
     * @param system  the System this kernel will be applied to
     * @param force   the IsolatedNonbondedForce this kernel will be used for
     */
    void initialize(const OpenMM::System& system, const IsolatedNonbondedForce& force);

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
     * @param context  the context to copy parameters to
     * @param force    the IsolatedNonbondedForce to copy the parameters from
     */
    void copyParametersToContext(OpenMM::ContextImpl& context, const IsolatedNonbondedForce& force);

    /**
     * Compute the Hessian (second derivatives) for the isolated nonbonded force.
     * This computes d²E/dr_i dr_j for all pairs of atoms.
     *
     * @param context  the context containing the current positions
     * @return the full Hessian matrix as a flattened vector (3N x 3N)
     */
    std::vector<double> computeHessian(OpenMM::ContextImpl& context) override;

private:
    bool hasInitializedKernel;
    int numAtoms;
    OpenMM::CudaContext& cu;
    CUfunction kernel;

    // GPU arrays
    OpenMM::CudaArray particleIndices;  // Which particles this force applies to
    OpenMM::CudaArray charges;          // Partial charges
    OpenMM::CudaArray sigmas;           // LJ sigma parameters
    OpenMM::CudaArray epsilons;         // LJ epsilon parameters
    OpenMM::CudaArray exclusions;       // Excluded atom pairs (int2)
    OpenMM::CudaArray exceptions;       // Exception atom pairs (int2)
    OpenMM::CudaArray exceptionParams;  // Exception parameters (float3: chargeProd, sigma, epsilon)

    std::vector<int> h_particleIndices;  // Host copy for updates

    // Hessian computation support
    CUfunction hessianKernel;             // Kernel for Hessian computation
    OpenMM::CudaArray hessianBuffer;      // Full Hessian matrix (3N x 3N)
};

} // namespace GridForcePlugin

#endif /*CUDA_ISOLATEDNONBONDEDFORCE_KERNELS_H_*/
