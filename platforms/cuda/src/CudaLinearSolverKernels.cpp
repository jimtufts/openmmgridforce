/* -------------------------------------------------------------------------- *
 *                              OpenMMGridForce                               *
 * -------------------------------------------------------------------------- *
 * cuSOLVER-backed dense Cholesky solver used by NewtonMinimizer to compute   *
 * the Newton search direction (H + lambda*I) p = -g each outer iteration.    *
 * Ramps lambda in {0, 1, 10, ..., lambdaMax} until dpotrf succeeds.         *
 * -------------------------------------------------------------------------- */

#include "CudaLinearSolverKernels.h"
#include "openmm/OpenMMException.h"
#include <cmath>
#include <cstring>
#include <string>

using namespace GridForcePlugin;
using namespace OpenMM;

#define CUSOLVER_CHECK(call)                                                \
    do {                                                                    \
        cusolverStatus_t st_ = (call);                                      \
        if (st_ != CUSOLVER_STATUS_SUCCESS) {                               \
            throw OpenMMException(std::string("cuSOLVER error in ") +       \
                                  #call + " (status=" +                     \
                                  std::to_string((int)st_) + ")");          \
        }                                                                   \
    } while (0)

CudaCalcLinearSolverKernel::~CudaCalcLinearSolverKernel() {
    if (handle != nullptr) {
        cusolverDnDestroy(handle);
        handle = nullptr;
    }
}

void CudaCalcLinearSolverKernel::initialize() {
    cu.setAsCurrent();
    CUSOLVER_CHECK(cusolverDnCreate(&handle));
    initialized = true;
}

void CudaCalcLinearSolverKernel::ensureBuffers(int n) {
    if (n <= maxN && d_H.isInitialized()) return;
    maxN = n;
    if (d_H.isInitialized())        d_H.resize(n * n);
    else                            d_H.initialize<double>(cu, n * n, "linsolve_H");
    if (d_b.isInitialized())        d_b.resize(n);
    else                            d_b.initialize<double>(cu, n, "linsolve_b");
    if (!d_info.isInitialized())    d_info.initialize<int>(cu, 1, "linsolve_info");

    int lwork = 0;
    CUSOLVER_CHECK(cusolverDnDpotrf_bufferSize(
        handle, CUBLAS_FILL_MODE_LOWER, n,
        (double*)d_H.getDevicePointer(), n, &lwork));
    workspaceBytes = (size_t)lwork * sizeof(double);
    if (d_workspace.isInitialized()) d_workspace.resize(lwork);
    else                             d_workspace.initialize<double>(cu, lwork, "linsolve_ws");
}

int CudaCalcLinearSolverKernel::solveLMCholesky(const std::vector<double>& H,
                                                 const std::vector<double>& b,
                                                 std::vector<double>& x, int n,
                                                 double lambdaMax,
                                                 double& lambdaUsedOut) {
    if (!initialized) initialize();
    if ((int)H.size() != n * n || (int)b.size() != n)
        throw OpenMMException("CudaCalcLinearSolverKernel::solveLMCholesky: "
                              "H or b size mismatch");
    cu.setAsCurrent();
    ensureBuffers(n);

    // Scratch host copy with LM shift baked in.  Reallocated once per solve;
    // n is small (~ few hundred typically), so this is cheap.
    std::vector<double> H_shifted(H.size());
    double lambda = 0.0;
    lambdaUsedOut = 0.0;

    for (int attempt = 0; attempt < 20; attempt++) {
        std::memcpy(H_shifted.data(), H.data(), sizeof(double) * H.size());
        if (lambda > 0.0)
            for (int i = 0; i < n; i++)
                H_shifted[(size_t)i * n + i] += lambda;

        // Upload H and b (b copied fresh each attempt because dpotrs overwrites).
        d_H.upload(H_shifted);
        d_b.upload(b);

        CUSOLVER_CHECK(cusolverDnDpotrf(
            handle, CUBLAS_FILL_MODE_LOWER, n,
            (double*)d_H.getDevicePointer(), n,
            (double*)d_workspace.getDevicePointer(),
            (int)(workspaceBytes / sizeof(double)),
            (int*)d_info.getDevicePointer()));

        std::vector<int> info(1);
        d_info.download(info);
        if (info[0] == 0) {
            CUSOLVER_CHECK(cusolverDnDpotrs(
                handle, CUBLAS_FILL_MODE_LOWER, n, 1,
                (double*)d_H.getDevicePointer(), n,
                (double*)d_b.getDevicePointer(), n,
                (int*)d_info.getDevicePointer()));
            x.resize(n);
            d_b.download(x);
            lambdaUsedOut = lambda;
            return lambda == 0.0 ? 0 : 1;
        }

        lambda = (lambda == 0.0) ? 1.0 : lambda * 10.0;
        if (lambda > lambdaMax) break;
    }

    // Fallback: preconditioned steepest descent p_i = -b_i / max(|H_ii|, 1)
    x.assign(n, 0.0);
    for (int i = 0; i < n; i++) {
        double h = std::fabs(H[(size_t)i * n + i]);
        x[i] = -b[i] / std::max(h, 1.0);
    }
    lambdaUsedOut = lambdaMax;
    return 2;
}
