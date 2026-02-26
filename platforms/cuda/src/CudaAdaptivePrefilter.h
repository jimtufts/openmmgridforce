#ifndef CUDA_ADAPTIVE_PREFILTER_H_
#define CUDA_ADAPTIVE_PREFILTER_H_

#include <vector>
#include <cublas_v2.h>
#include <cuda_runtime.h>

namespace GridForcePlugin {

/**
 * GPU-accelerated adaptive regularized B-spline prefilter.
 *
 * Solves: (N^T N + sum_beta M_beta^T Lambda^2 M_beta) P = N^T Q
 * using Preconditioned Conjugate Gradient (PCG) with cuBLAS.
 *
 * N is the cubic B-spline collocation matrix (Kronecker-structured).
 * M_beta are second-derivative operators (xx, yy, zz).
 * Lambda is a per-point adaptive regularization field computed from
 * the gradient of the input data.
 *
 * When cReg=0 (or lambda=0 everywhere), reduces to exact interpolation
 * identical to BSplinePrefilter.h's Thomas algorithm.
 *
 * Based on: Lenz et al. (2023), "Customizable Adaptive Regularization
 * Techniques for B-Spline Modeling", arXiv:2301.01209
 */
class CudaAdaptivePrefilter {
public:
    /**
     * Construct the prefilter for a given grid size.
     *
     * @param nx, ny, nz  Grid dimensions
     * @param cReg        Regularization coefficient (0.0 = disabled, falls back to Thomas)
     * @param threshold   Gradient magnitude threshold below which lambda = 0
     * @param pcgTol      Relative residual tolerance for PCG convergence
     * @param maxIter     Maximum PCG iterations
     */
    CudaAdaptivePrefilter(int nx, int ny, int nz,
                          double cReg = 0.0, double threshold = 0.0,
                          double pcgTol = 1e-6, int maxIter = 200);

    ~CudaAdaptivePrefilter();

    // Non-copyable
    CudaAdaptivePrefilter(const CudaAdaptivePrefilter&) = delete;
    CudaAdaptivePrefilter& operator=(const CudaAdaptivePrefilter&) = delete;

    /**
     * Apply the adaptive prefilter in-place.
     * Drop-in replacement for bsplinePrefilter3DByOrder().
     *
     * Grid layout is row-major: vals[ix * ny*nz + iy * nz + iz]
     *
     * @param vals  Grid values (size nx*ny*nz), modified in-place to B-spline coefficients
     */
    void apply(std::vector<double>& vals);

    /** Number of PCG iterations used in the last call to apply(). Returns 0 for fast path. */
    int getLastIterationCount() const { return lastIterCount_; }

    /** Final relative residual ||r|| / ||b|| from the last PCG solve. */
    double getLastResidual() const { return lastResidual_; }

private:
    // Grid dimensions
    int nx_, ny_, nz_;
    long long totalN_;  // nx * ny * nz

    // Parameters
    double cReg_;
    double threshold_;
    double pcgTol_;
    int maxIter_;

    // cuBLAS handle
    cublasHandle_t cublasHandle_;

    // Device memory buffers (all double, size totalN_ each)
    // During setup: d_buf0_ = Q (input data), d_buf1_ = b (RHS = N^T Q)
    // During PCG:   d_buf0_ = temp (axis scratch), d_buf1_ = temp2 (axis scratch)
    double* d_P_;       // Solution
    double* d_R_;       // Residual
    double* d_Z_;       // Preconditioned residual
    double* d_D_;       // Search direction
    double* d_AD_;      // A * D result
    double* d_lambda_;  // Per-point regularization weights
    double* d_buf0_;    // Aliased: Q during setup, temp during PCG
    double* d_buf1_;    // Aliased: b during setup, temp2 during PCG

    // Precomputed Thomas algorithm modified diagonals (host + device)
    std::vector<double> thomasDiagX_, thomasDiagY_, thomasDiagZ_;
    double* d_thomasDiagX_;
    double* d_thomasDiagY_;
    double* d_thomasDiagZ_;

    // Convergence tracking
    int lastIterCount_;
    double lastResidual_;

    // Axis parameter computation
    struct AxisParams {
        int nLines;      // Number of independent 1D problems
        int lineLength;  // Length of each 1D problem
        int stride;      // Stride between consecutive elements in a line
        int dim1;        // Second dimension for 2D line indexing
        int lineStride0; // Stride for first line index
        int lineStride1; // Stride for second line index
    };
    AxisParams getAxisParams(int axis) const;

    // Memory management
    void allocateDevice();
    void freeDevice();

    // Precompute Thomas diagonals for a given line length
    static std::vector<double> precomputeThomasDiag(int N);

    // Fast path: separable Thomas solve (when cReg == 0)
    void applySeparableThomas(std::vector<double>& vals);

    // Lambda field computation
    void computeLambdaField();

    // RHS computation: b = N^T Q
    void computeRHS(double* d_b, const double* d_Q);

    // Matrix-vector product: out = A * in, where A = N^T N + regularization
    void applyA(double* d_out, const double* d_in);

    // Separable N^T N application (three pentadiagonal matvec passes)
    void applyNtN(double* d_out, const double* d_in);

    // Regularization term: out += sum_beta M_beta^T Lambda^2 M_beta * in
    void applyRegularization(double* d_out, const double* d_in);

    // Preconditioner: separable Thomas solves (approximates (N^T N)^{-1})
    void applyPreconditioner(double* d_out, const double* d_in);

    // Axis-wise kernel dispatchers
    void launchTridiagMatvec(double* d_out, const double* d_in, int axis);
    void launchPentadiagMatvec(double* d_out, const double* d_in, int axis);
    void launchThomasSolve(double* d_out, const double* d_in, int axis,
                           const double* d_thomasDiag);
    void launchSecondDeriv(double* d_out, const double* d_in, int axis);
    void launchSecondDerivTranspose(double* d_out, const double* d_in, int axis);
};

}  // namespace GridForcePlugin

#endif  // CUDA_ADAPTIVE_PREFILTER_H_
