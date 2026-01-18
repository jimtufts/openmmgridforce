#ifndef CUDA_BONDED_HESSIAN_H_
#define CUDA_BONDED_HESSIAN_H_

#include "openmm/cuda/CudaContext.h"
#include "openmm/cuda/CudaArray.h"
#include "openmm/System.h"
#include "openmm/Context.h"
#include <vector>

namespace GridForcePlugin {

/**
 * CudaBondedHessian computes the Hessian (second derivative matrix) for bonded
 * interactions using CUDA GPU acceleration.
 *
 * This extracts parameters from HarmonicBondForce, HarmonicAngleForce, and
 * PeriodicTorsionForce, uploads them to GPU, and calls CUDA kernels.
 */
class CudaBondedHessian {
public:
    CudaBondedHessian();
    ~CudaBondedHessian();

    /**
     * Initialize by extracting bonded force parameters from the System.
     * The Context must be using the CUDA platform.
     *
     * @param system   the System containing bonded forces
     * @param context  the Context (must use CUDA platform)
     */
    void initialize(const OpenMM::System& system, OpenMM::Context& context);

    /**
     * Compute the full Hessian matrix for all bonded interactions on GPU.
     *
     * @param context  the Context containing current positions
     * @return flattened 3N x 3N Hessian matrix in row-major order
     */
    std::vector<double> computeHessian(OpenMM::Context& context);

    int getNumBonds() const { return numBonds; }
    int getNumAngles() const { return numAngles; }
    int getNumTorsions() const { return numTorsions; }

private:
    OpenMM::CudaContext* cu;
    bool initialized;
    int numAtoms;
    int numBonds;
    int numAngles;
    int numTorsions;

    // GPU arrays for parameters
    OpenMM::CudaArray bondAtoms;      // [numBonds * 2]
    OpenMM::CudaArray bondParams;     // [numBonds * 2]: k, r0
    OpenMM::CudaArray angleAtoms;     // [numAngles * 3]
    OpenMM::CudaArray angleParams;    // [numAngles * 2]: k, theta0
    OpenMM::CudaArray torsionAtoms;   // [numTorsions * 4]
    OpenMM::CudaArray torsionParams;  // [numTorsions * 3]: k, n, phi0

    // GPU array for Hessian output
    OpenMM::CudaArray hessianBuffer;

    // CUDA kernels
    CUfunction bondHessianKernel;
    CUfunction angleHessianKernel;
    CUfunction torsionHessianKernel;
    CUfunction initHessianKernel;
};

}  // namespace GridForcePlugin

#endif /*CUDA_BONDED_HESSIAN_H_*/
