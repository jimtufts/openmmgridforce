#ifndef CUDA_GRIDFORCE_LINEAR_SOLVER_KERNELS_H_
#define CUDA_GRIDFORCE_LINEAR_SOLVER_KERNELS_H_

#include "LinearSolverKernels.h"
#include "openmm/cuda/CudaContext.h"
#include "openmm/cuda/CudaArray.h"
#include <cusolverDn.h>

namespace GridForcePlugin {

class CudaCalcLinearSolverKernel : public CalcLinearSolverKernel {
public:
    CudaCalcLinearSolverKernel(std::string name, const OpenMM::Platform& platform,
                               OpenMM::CudaContext& cu)
        : CalcLinearSolverKernel(name, platform), cu(cu),
          handle(nullptr), maxN(0), workspaceBytes(0), initialized(false) {}

    ~CudaCalcLinearSolverKernel();

    void initialize() override;

    int solveLMCholesky(const std::vector<double>& H,
                        const std::vector<double>& b,
                        std::vector<double>& x, int n,
                        double lambdaMax,
                        double& lambdaUsedOut) override;

private:
    void ensureBuffers(int n);

    OpenMM::CudaContext& cu;
    cusolverDnHandle_t handle;
    OpenMM::CudaArray d_H;         // n*n double
    OpenMM::CudaArray d_H_orig;    // n*n double (undamped copy, for retries)
    OpenMM::CudaArray d_b;          // n double
    OpenMM::CudaArray d_workspace;  // cuSolver workspace
    OpenMM::CudaArray d_info;       // 1 int
    int maxN;
    size_t workspaceBytes;
    bool initialized;
};

}  // namespace GridForcePlugin

#endif  // CUDA_GRIDFORCE_LINEAR_SOLVER_KERNELS_H_
