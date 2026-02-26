/**
 * CUDA-native adaptive regularized B-spline prefilter.
 *
 * Solves: (N^T N + sum_beta M_beta^T Lambda^2 M_beta) P = N^T Q
 * via Preconditioned Conjugate Gradient with cuBLAS.
 *
 * See CudaAdaptivePrefilter.h for design overview.
 */

#include "CudaAdaptivePrefilter.h"
#include <iostream>
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <cassert>

namespace GridForcePlugin {

// ============================================================================
// Helper: check CUDA errors
// ============================================================================

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        throw std::runtime_error(std::string("CUDA error in ") + __FILE__ + ":" + \
            std::to_string(__LINE__) + ": " + cudaGetErrorString(err)); \
    } \
} while(0)

#define CUBLAS_CHECK(call) do { \
    cublasStatus_t status = (call); \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        throw std::runtime_error(std::string("cuBLAS error in ") + __FILE__ + ":" + \
            std::to_string(__LINE__) + ": status=" + std::to_string(status)); \
    } \
} while(0)

// ============================================================================
// Device helper: compute base offset from lineIdx and axis parameters
// ============================================================================

__device__ __forceinline__
int computeBase(int lineIdx, int dim1, int lineStride0, int lineStride1) {
    int i0 = lineIdx / dim1;
    int i1 = lineIdx - i0 * dim1;  // lineIdx % dim1 without expensive modulo
    return i0 * lineStride0 + i1 * lineStride1;
}

// ============================================================================
// Kernel 1: Batched tridiagonal matvec (N applied along one axis)
//
// Applies the cubic B-spline collocation matrix N to each independent 1D line.
// Interior stencil: (1/6, 4/6, 1/6)
// Boundary row 0:   (5/6, 1/6)
// Boundary row N-1: (1/6, 5/6)
// ============================================================================

__global__ void batchedTridiagMatvecKernel(
    double* __restrict__ out,
    const double* __restrict__ in,
    int nLines, int lineLength, int stride,
    int dim1, int lineStride0, int lineStride1)
{
    int lineIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (lineIdx >= nLines) return;

    int base = computeBase(lineIdx, dim1, lineStride0, lineStride1);
    int N = lineLength;

    const double c1 = 1.0 / 6.0;
    const double c4 = 4.0 / 6.0;
    const double c5 = 5.0 / 6.0;

    // Boundary row 0
    out[base] = c5 * in[base] + c1 * in[base + stride];

    // Interior rows
    for (int j = 1; j < N - 1; j++) {
        int pos = base + j * stride;
        out[pos] = c1 * in[pos - stride] + c4 * in[pos] + c1 * in[pos + stride];
    }

    // Boundary row N-1
    if (N > 1) {
        int last = base + (N - 1) * stride;
        out[last] = c1 * in[last - stride] + c5 * in[last];
    }
}

// ============================================================================
// Kernel 2: Batched pentadiagonal matvec (N^T N applied along one axis)
//
// Interior stencil: (1, 8, 18, 8, 1) / 36
// Boundary row 0:   (26, 9, 1) / 36
// Boundary row 1:   (9, 18, 8, 1) / 36
// Boundary row N-2: (1, 8, 18, 9) / 36
// Boundary row N-1: (1, 9, 26) / 36
// ============================================================================

__global__ void batchedPentadiagMatvecKernel(
    double* __restrict__ out,
    const double* __restrict__ in,
    int nLines, int lineLength, int stride,
    int dim1, int lineStride0, int lineStride1)
{
    int lineIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (lineIdx >= nLines) return;

    int base = computeBase(lineIdx, dim1, lineStride0, lineStride1);
    int N = lineLength;
    const double inv36 = 1.0 / 36.0;

    // Row 0: (26, 9, 1) / 36
    {
        double val = 26.0 * in[base] + 9.0 * in[base + stride];
        if (N > 2) val += 1.0 * in[base + 2 * stride];
        out[base] = inv36 * val;
    }

    // Row 1: (9, 18, 8, 1) / 36
    if (N > 2) {
        int p = base + stride;
        double val = 9.0 * in[p - stride] + 18.0 * in[p] + 8.0 * in[p + stride];
        if (N > 3) val += 1.0 * in[p + 2 * stride];
        out[p] = inv36 * val;
    }

    // Interior rows 2..N-3: (1, 8, 18, 8, 1) / 36
    for (int j = 2; j < N - 2; j++) {
        int p = base + j * stride;
        out[p] = inv36 * (1.0 * in[p - 2 * stride] + 8.0 * in[p - stride] +
                           18.0 * in[p] + 8.0 * in[p + stride] + 1.0 * in[p + 2 * stride]);
    }

    // Row N-2: (1, 8, 18, 9) / 36
    if (N > 3) {
        int p = base + (N - 2) * stride;
        double val = 9.0 * in[p + stride] + 18.0 * in[p] + 8.0 * in[p - stride];
        if (N > 3) val += 1.0 * in[p - 2 * stride];
        out[p] = inv36 * val;
    }

    // Row N-1: (1, 9, 26) / 36
    if (N > 1) {
        int p = base + (N - 1) * stride;
        double val = 26.0 * in[p] + 9.0 * in[p - stride];
        if (N > 2) val += 1.0 * in[p - 2 * stride];
        out[p] = inv36 * val;
    }
}

// ============================================================================
// Kernel 3: Batched Thomas solve (preconditioner)
//
// Solves: c[i-1] + 4*c[i] + c[i+1] = 6*f[i], boundary diag=5
// Uses precomputed modified diagonal d[] (data-independent, in global memory).
// ============================================================================

__global__ void batchedThomasSolveKernel(
    double* __restrict__ out,
    const double* __restrict__ rhs,
    const double* __restrict__ thomasDiag,  // precomputed modified diagonal, length lineLength
    int nLines, int lineLength, int stride,
    int dim1, int lineStride0, int lineStride1)
{
    int lineIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (lineIdx >= nLines) return;

    int base = computeBase(lineIdx, dim1, lineStride0, lineStride1);
    int N = lineLength;

    // Forward elimination: compute modified RHS
    // b'[0] = 6 * f[0]
    // b'[i] = 6 * f[i] - (1/d[i-1]) * b'[i-1]
    double b_prev = 6.0 * rhs[base];
    out[base] = b_prev;

    for (int j = 1; j < N; j++) {
        double m = 1.0 / thomasDiag[j - 1];
        b_prev = 6.0 * rhs[base + j * stride] - m * b_prev;
        out[base + j * stride] = b_prev;
    }

    // Back substitution
    out[base + (N - 1) * stride] /= thomasDiag[N - 1];
    for (int j = N - 2; j >= 0; j--) {
        int pos = base + j * stride;
        out[pos] = (out[pos] - out[pos + stride]) / thomasDiag[j];
    }
}

// ============================================================================
// Kernel 4: Lambda field computation
//
// lambda_j = cReg * max(|grad f_j| - threshold, 0)
// Gradient via central differences with clamped boundary indexing.
// ============================================================================

__global__ void computeLambdaFieldKernel(
    double* __restrict__ lambda,
    const double* __restrict__ Q,
    int nx, int ny, int nz,
    double cReg, double threshold)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = nx * ny * nz;
    if (idx >= total) return;

    int nyz = ny * nz;
    int ix = idx / nyz;
    int rem = idx - ix * nyz;
    int iy = rem / nz;
    int iz = rem - iy * nz;

    // Central differences with clamped boundaries
    int ixm = max(ix - 1, 0);
    int ixp = min(ix + 1, nx - 1);
    int iym = max(iy - 1, 0);
    int iyp = min(iy + 1, ny - 1);
    int izm = max(iz - 1, 0);
    int izp = min(iz + 1, nz - 1);

    double gx = 0.5 * (Q[ixp * nyz + iy * nz + iz] - Q[ixm * nyz + iy * nz + iz]);
    double gy = 0.5 * (Q[ix * nyz + iyp * nz + iz] - Q[ix * nyz + iym * nz + iz]);
    double gz = 0.5 * (Q[ix * nyz + iy * nz + izp] - Q[ix * nyz + iy * nz + izm]);

    double gradMag = sqrt(gx * gx + gy * gy + gz * gz);
    double lam = cReg * fmax(gradMag - threshold, 0.0);
    lambda[idx] = lam;
}

// ============================================================================
// Kernel 5a: Second derivative operator along one axis
//
// Stencil: [1, -2, 1], boundary: [-1, 1] at start, [1, -1] at end
// (zero-slope / natural boundary conditions)
// ============================================================================

__global__ void applySecondDerivKernel(
    double* __restrict__ out,
    const double* __restrict__ in,
    int nLines, int lineLength, int stride,
    int dim1, int lineStride0, int lineStride1)
{
    int lineIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (lineIdx >= nLines) return;

    int base = computeBase(lineIdx, dim1, lineStride0, lineStride1);
    int N = lineLength;

    // Boundary row 0: [-1, 1] (forward difference approximation)
    out[base] = -in[base] + in[base + stride];

    // Interior: [1, -2, 1]
    for (int j = 1; j < N - 1; j++) {
        int p = base + j * stride;
        out[p] = in[p - stride] - 2.0 * in[p] + in[p + stride];
    }

    // Boundary row N-1: [1, -1]
    if (N > 1) {
        int p = base + (N - 1) * stride;
        out[p] = in[p - stride] - in[p];
    }
}

// ============================================================================
// Kernel 5b: Transpose of second derivative operator
//
// Since boundary BCs are [-1,1] and [1,-1], the transpose has different
// boundary rows than the forward operator:
// Row 0:   [-1, 1, 0, ...]       (transpose of column 0 of forward operator)
// Row 1:   [1, -2, 1, 0, ...]    (same as interior)
// Row N-2: [0, ..., 1, -2, 1]    (same as interior)
// Row N-1: [0, ..., 1, -1]       (transpose of column N-1)
//
// Actually for the natural BC second difference operator:
// Forward:  D[0,:] = [-1, 1, 0, ...]
//           D[j,:] = [... 1, -2, 1, ...]  for 1 <= j <= N-2
//           D[N-1,:] = [..., 0, 1, -1]
//
// D^T[i,:] = column i of D. Since D is nearly symmetric (interior is symmetric),
// and boundary rows are transposes of boundary columns:
// D^T[0,:] = [-1, 1, 0, ...]      (column 0 of D: D[0,0]=-1, D[1,0]=1, rest 0)
// D^T[1,:] = [1, -2, 1, 0, ...]   (column 1 of D: D[0,1]=1, D[1,1]=-2, D[2,1]=1)
// D^T[j,:] = [... 1, -2, 1, ...]  for 2 <= j <= N-3
// D^T[N-2,:] = [..., 1, -2, 1]    (column N-2: D[N-3,N-2]=1, D[N-2,N-2]=-2, D[N-1,N-2]=1)
// D^T[N-1,:] = [..., 0, 1, -1]    (column N-1: D[N-2,N-1]=1, D[N-1,N-1]=-1)
//
// So D^T = D! The operator is symmetric with these BCs.
// We can reuse applySecondDerivKernel for the transpose.
// ============================================================================

// (No separate transpose kernel needed — D^T = D for these boundary conditions)

// ============================================================================
// Kernel 6: Elementwise multiply by lambda^2
// ============================================================================

__global__ void elementwiseLambdaSqMultiplyKernel(
    double* __restrict__ inout,
    const double* __restrict__ lambda,
    int total)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    double l = lambda[idx];
    inout[idx] *= l * l;
}

// ============================================================================
// Kernel 7: Elementwise accumulate (out += in)
// ============================================================================

__global__ void elementwiseAccumulateKernel(
    double* __restrict__ out,
    const double* __restrict__ in,
    int total)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    out[idx] += in[idx];
}

// ============================================================================
// Host class implementation
// ============================================================================

static constexpr int BLOCK_SIZE = 256;

CudaAdaptivePrefilter::CudaAdaptivePrefilter(int nx, int ny, int nz,
                                               double cReg, double threshold,
                                               double pcgTol, int maxIter)
    : nx_(nx), ny_(ny), nz_(nz),
      totalN_((long long)nx * ny * nz),
      cReg_(cReg), threshold_(threshold),
      pcgTol_(pcgTol), maxIter_(maxIter),
      d_P_(nullptr), d_R_(nullptr), d_Z_(nullptr), d_D_(nullptr),
      d_AD_(nullptr), d_lambda_(nullptr), d_buf0_(nullptr), d_buf1_(nullptr),
      d_thomasDiagX_(nullptr), d_thomasDiagY_(nullptr), d_thomasDiagZ_(nullptr),
      lastIterCount_(0), lastResidual_(0.0)
{
    CUBLAS_CHECK(cublasCreate(&cublasHandle_));

    // Precompute Thomas algorithm modified diagonals
    thomasDiagX_ = precomputeThomasDiag(nx_);
    thomasDiagY_ = precomputeThomasDiag(ny_);
    thomasDiagZ_ = precomputeThomasDiag(nz_);

    allocateDevice();
}

CudaAdaptivePrefilter::~CudaAdaptivePrefilter() {
    freeDevice();
    cublasDestroy(cublasHandle_);
}

void CudaAdaptivePrefilter::allocateDevice() {
    size_t bytes = totalN_ * sizeof(double);
    CUDA_CHECK(cudaMalloc(&d_P_, bytes));
    CUDA_CHECK(cudaMalloc(&d_R_, bytes));
    CUDA_CHECK(cudaMalloc(&d_Z_, bytes));
    CUDA_CHECK(cudaMalloc(&d_D_, bytes));
    CUDA_CHECK(cudaMalloc(&d_AD_, bytes));
    CUDA_CHECK(cudaMalloc(&d_lambda_, bytes));
    CUDA_CHECK(cudaMalloc(&d_buf0_, bytes));
    CUDA_CHECK(cudaMalloc(&d_buf1_, bytes));

    // Upload precomputed Thomas diagonals to device (persist for lifetime)
    CUDA_CHECK(cudaMalloc(&d_thomasDiagX_, nx_ * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_thomasDiagY_, ny_ * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_thomasDiagZ_, nz_ * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_thomasDiagX_, thomasDiagX_.data(), nx_ * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_thomasDiagY_, thomasDiagY_.data(), ny_ * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_thomasDiagZ_, thomasDiagZ_.data(), nz_ * sizeof(double), cudaMemcpyHostToDevice));
}

void CudaAdaptivePrefilter::freeDevice() {
    if (d_P_)      cudaFree(d_P_);
    if (d_R_)      cudaFree(d_R_);
    if (d_Z_)      cudaFree(d_Z_);
    if (d_D_)      cudaFree(d_D_);
    if (d_AD_)     cudaFree(d_AD_);
    if (d_lambda_) cudaFree(d_lambda_);
    if (d_buf0_)   cudaFree(d_buf0_);
    if (d_buf1_)   cudaFree(d_buf1_);
    if (d_thomasDiagX_) cudaFree(d_thomasDiagX_);
    if (d_thomasDiagY_) cudaFree(d_thomasDiagY_);
    if (d_thomasDiagZ_) cudaFree(d_thomasDiagZ_);
    d_P_ = d_R_ = d_Z_ = d_D_ = d_AD_ = d_lambda_ = d_buf0_ = d_buf1_ = nullptr;
    d_thomasDiagX_ = d_thomasDiagY_ = d_thomasDiagZ_ = nullptr;
}

std::vector<double> CudaAdaptivePrefilter::precomputeThomasDiag(int N) {
    // Thomas algorithm for tridiagonal system:
    //   diag = [5, 4, 4, ..., 4, 5], off-diag = 1
    // Forward elimination: d[i] = diag[i] - 1/d[i-1]
    std::vector<double> d(N);
    d[0] = 5.0;
    for (int i = 1; i < N; i++) {
        double diagVal = (i == N - 1) ? 5.0 : 4.0;
        d[i] = diagVal - 1.0 / d[i - 1];
    }
    return d;
}

CudaAdaptivePrefilter::AxisParams CudaAdaptivePrefilter::getAxisParams(int axis) const {
    AxisParams p;
    int nyz = ny_ * nz_;
    switch (axis) {
        case 0: // X-axis: ny*nz lines of length nx, stride=ny*nz
            p.nLines = ny_ * nz_;
            p.lineLength = nx_;
            p.stride = nyz;
            p.dim1 = nz_;
            p.lineStride0 = nz_;   // iy * nz
            p.lineStride1 = 1;     // iz * 1
            break;
        case 1: // Y-axis: nx*nz lines of length ny, stride=nz
            p.nLines = nx_ * nz_;
            p.lineLength = ny_;
            p.stride = nz_;
            p.dim1 = nz_;
            p.lineStride0 = nyz;   // ix * ny*nz
            p.lineStride1 = 1;     // iz * 1
            break;
        case 2: // Z-axis: nx*ny lines of length nz, stride=1
            p.nLines = nx_ * ny_;
            p.lineLength = nz_;
            p.stride = 1;
            p.dim1 = ny_;
            p.lineStride0 = nyz;   // ix * ny*nz
            p.lineStride1 = nz_;   // iy * nz
            break;
        default:
            throw std::runtime_error("Invalid axis");
    }
    return p;
}

// ============================================================================
// Kernel launch dispatchers
// ============================================================================

void CudaAdaptivePrefilter::launchTridiagMatvec(double* d_out, const double* d_in, int axis) {
    AxisParams p = getAxisParams(axis);
    int gridSize = (p.nLines + BLOCK_SIZE - 1) / BLOCK_SIZE;
    batchedTridiagMatvecKernel<<<gridSize, BLOCK_SIZE>>>(
        d_out, d_in, p.nLines, p.lineLength, p.stride,
        p.dim1, p.lineStride0, p.lineStride1);
    CUDA_CHECK(cudaGetLastError());
}

void CudaAdaptivePrefilter::launchPentadiagMatvec(double* d_out, const double* d_in, int axis) {
    AxisParams p = getAxisParams(axis);
    int gridSize = (p.nLines + BLOCK_SIZE - 1) / BLOCK_SIZE;
    batchedPentadiagMatvecKernel<<<gridSize, BLOCK_SIZE>>>(
        d_out, d_in, p.nLines, p.lineLength, p.stride,
        p.dim1, p.lineStride0, p.lineStride1);
    CUDA_CHECK(cudaGetLastError());
}

void CudaAdaptivePrefilter::launchThomasSolve(double* d_out, const double* d_in, int axis,
                                                const double* d_thomasDiag) {
    AxisParams p = getAxisParams(axis);
    int gridSize = (p.nLines + BLOCK_SIZE - 1) / BLOCK_SIZE;
    batchedThomasSolveKernel<<<gridSize, BLOCK_SIZE>>>(
        d_out, d_in, d_thomasDiag, p.nLines, p.lineLength, p.stride,
        p.dim1, p.lineStride0, p.lineStride1);
    CUDA_CHECK(cudaGetLastError());
}

void CudaAdaptivePrefilter::launchSecondDeriv(double* d_out, const double* d_in, int axis) {
    AxisParams p = getAxisParams(axis);
    int gridSize = (p.nLines + BLOCK_SIZE - 1) / BLOCK_SIZE;
    applySecondDerivKernel<<<gridSize, BLOCK_SIZE>>>(
        d_out, d_in, p.nLines, p.lineLength, p.stride,
        p.dim1, p.lineStride0, p.lineStride1);
    CUDA_CHECK(cudaGetLastError());
}

void CudaAdaptivePrefilter::launchSecondDerivTranspose(double* d_out, const double* d_in, int axis) {
    // D^T = D for natural boundary conditions (see kernel 5b comment)
    launchSecondDeriv(d_out, d_in, axis);
}

// ============================================================================
// High-level operators
// ============================================================================

void CudaAdaptivePrefilter::computeLambdaField() {
    // d_buf0_ contains Q at this point
    int gridSize = (totalN_ + BLOCK_SIZE - 1) / BLOCK_SIZE;
    computeLambdaFieldKernel<<<gridSize, BLOCK_SIZE>>>(
        d_lambda_, d_buf0_, nx_, ny_, nz_, cReg_, threshold_);
    CUDA_CHECK(cudaGetLastError());
}

void CudaAdaptivePrefilter::computeRHS(double* d_b, const double* d_Q) {
    // b = N^T Q
    // N^T = N (symmetric matrix), so this is just three tridiag matvec passes.
    // Apply along Z first, then Y, then X (Kronecker product order).
    // IMPORTANT: d_b may alias d_buf1_, so alternate between d_b and d_P_ to avoid
    // reading/writing the same buffer in the tridiag matvec kernel.
    launchTridiagMatvec(d_b, d_Q, 2);           // Z-pass: d_b = N_z * Q along z
    launchTridiagMatvec(d_P_, d_b, 1);          // Y-pass: d_P_ = N_y * d_b along y
    launchTridiagMatvec(d_b, d_P_, 0);          // X-pass: d_b = N_x * d_P_ along x
}

void CudaAdaptivePrefilter::applyNtN(double* d_out, const double* d_in) {
    // Apply separable N^T N = (N_x^T N_x) ⊗ (N_y^T N_y) ⊗ (N_z^T N_z)
    // Three pentadiagonal matvec passes, one per axis
    launchPentadiagMatvec(d_out, d_in, 2);      // Z-pass
    launchPentadiagMatvec(d_buf0_, d_out, 1);   // Y-pass (using d_buf0_ as temp)
    launchPentadiagMatvec(d_out, d_buf0_, 0);   // X-pass
}

void CudaAdaptivePrefilter::applyRegularization(double* d_out, const double* d_in) {
    // Compute sum_beta M_beta^T Lambda^2 M_beta * in for beta = {xx, yy, zz}
    // For each axis:
    //   1. temp = D2_axis * in          (second derivative along axis)
    //   2. temp *= lambda^2             (elementwise)
    //   3. out += D2_axis^T * temp      (transpose, accumulate)

    int elemGridSize = (totalN_ + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (int axis = 0; axis < 3; axis++) {
        // Step 1: d_buf0_ = D2 * in along this axis
        launchSecondDeriv(d_buf0_, d_in, axis);

        // Step 2: d_buf0_ *= lambda^2
        elementwiseLambdaSqMultiplyKernel<<<elemGridSize, BLOCK_SIZE>>>(
            d_buf0_, d_lambda_, totalN_);
        CUDA_CHECK(cudaGetLastError());

        // Step 3: d_buf1_ = D2^T * d_buf0_; out += d_buf1_
        launchSecondDerivTranspose(d_buf1_, d_buf0_, axis);
        elementwiseAccumulateKernel<<<elemGridSize, BLOCK_SIZE>>>(
            d_out, d_buf1_, totalN_);
        CUDA_CHECK(cudaGetLastError());
    }
}

void CudaAdaptivePrefilter::applyA(double* d_out, const double* d_in) {
    // out = (N^T N + regularization) * in
    applyNtN(d_out, d_in);
    if (cReg_ > 0.0) {
        applyRegularization(d_out, d_in);
    }
}

void CudaAdaptivePrefilter::applyPreconditioner(double* d_out, const double* d_in) {
    // Approximate (N^T N)^{-1} via separable Thomas solves: N^{-1} applied per axis
    // This is the same as the existing BSplinePrefilter.h Thomas algorithm.
    // Thomas diagonals are pre-uploaded to device in allocateDevice().

    // Three Thomas solve passes: Z, Y, X
    launchThomasSolve(d_out, d_in, 2, d_thomasDiagZ_);
    launchThomasSolve(d_buf0_, d_out, 1, d_thomasDiagY_);
    launchThomasSolve(d_out, d_buf0_, 0, d_thomasDiagX_);
}

// ============================================================================
// Fast path: separable Thomas solve (cReg == 0)
// ============================================================================

void CudaAdaptivePrefilter::applySeparableThomas(std::vector<double>& vals) {
    // Upload data to GPU
    CUDA_CHECK(cudaMemcpy(d_buf0_, vals.data(), totalN_ * sizeof(double), cudaMemcpyHostToDevice));

    // Three Thomas solve passes: X, Y, Z (matching BSplinePrefilter.h order).
    // Thomas diagonals are pre-uploaded to device in allocateDevice().
    launchThomasSolve(d_P_, d_buf0_, 0, d_thomasDiagX_);     // X-pass
    launchThomasSolve(d_buf0_, d_P_, 1, d_thomasDiagY_);     // Y-pass
    launchThomasSolve(d_P_, d_buf0_, 2, d_thomasDiagZ_);     // Z-pass

    // Download result
    CUDA_CHECK(cudaMemcpy(vals.data(), d_P_, totalN_ * sizeof(double), cudaMemcpyDeviceToHost));

    lastIterCount_ = 0;
    lastResidual_ = 0.0;
}

// ============================================================================
// Main entry point: PCG solver
// ============================================================================

void CudaAdaptivePrefilter::apply(std::vector<double>& vals) {
    if ((long long)vals.size() != totalN_) {
        throw std::runtime_error("CudaAdaptivePrefilter::apply: vals size mismatch");
    }

    // Fast path when no regularization
    if (cReg_ <= 0.0) {
        applySeparableThomas(vals);
        return;
    }

    // Upload Q to d_buf0_ (will be aliased as temp later)
    CUDA_CHECK(cudaMemcpy(d_buf0_, vals.data(), totalN_ * sizeof(double), cudaMemcpyHostToDevice));

    // Step 1: Compute lambda field from Q
    computeLambdaField();

    // Step 2: Compute RHS b = N^T Q (stored in d_buf1_)
    computeRHS(d_buf1_, d_buf0_);

    // Step 3: Initial guess: P = preconditioner applied to b (warm start)
    applyPreconditioner(d_P_, d_buf1_);

    // IMPORTANT: Save b (in d_buf1_) to d_R_ and compute ||b|| BEFORE calling applyA,
    // because applyA's regularization step uses d_buf1_ as a scratch buffer.
    // R = b (will become R = b - A*P after the axpy below)
    CUDA_CHECK(cudaMemcpy(d_R_, d_buf1_, totalN_ * sizeof(double), cudaMemcpyDeviceToDevice));

    // Compute ||b|| for relative convergence (must read d_buf1_ before applyA corrupts it)
    double bnorm;
    CUBLAS_CHECK(cublasDnrm2(cublasHandle_, totalN_, d_buf1_, 1, &bnorm));
    if (bnorm == 0.0) bnorm = 1.0;

    // Step 4: Initial residual R = b - A*P
    applyA(d_AD_, d_P_);
    // Note: d_buf0_ and d_buf1_ are now corrupted (used as temp by applyNtN/applyRegularization)

    // R -= A*P (R already contains b from the copy above)
    double neg_one = -1.0;
    CUBLAS_CHECK(cublasDaxpy(cublasHandle_, totalN_, &neg_one, d_AD_, 1, d_R_, 1));

    // Step 5: Z = preconditioner(R)
    applyPreconditioner(d_Z_, d_R_);

    // D = Z (copy)
    CUDA_CHECK(cudaMemcpy(d_D_, d_Z_, totalN_ * sizeof(double), cudaMemcpyDeviceToDevice));

    // rho = R^T Z
    double rho;
    CUBLAS_CHECK(cublasDdot(cublasHandle_, totalN_, d_R_, 1, d_Z_, 1, &rho));

    // Initial residual norm
    double rnorm;
    CUBLAS_CHECK(cublasDnrm2(cublasHandle_, totalN_, d_R_, 1, &rnorm));

    lastIterCount_ = 0;
    lastResidual_ = rnorm / bnorm;

    if (lastResidual_ < pcgTol_) {
        // Already converged (warm start was good enough)
        CUDA_CHECK(cudaMemcpy(vals.data(), d_P_, totalN_ * sizeof(double), cudaMemcpyDeviceToHost));
        return;
    }

    // PCG iteration loop
    for (int iter = 0; iter < maxIter_; iter++) {
        // AD = A * D
        applyA(d_AD_, d_D_);

        // alpha = rho / (D^T AD)
        double dAd;
        CUBLAS_CHECK(cublasDdot(cublasHandle_, totalN_, d_D_, 1, d_AD_, 1, &dAd));
        if (dAd == 0.0) break;  // Avoid division by zero
        double alpha = rho / dAd;

        // P += alpha * D
        CUBLAS_CHECK(cublasDaxpy(cublasHandle_, totalN_, &alpha, d_D_, 1, d_P_, 1));

        // R -= alpha * AD
        double neg_alpha = -alpha;
        CUBLAS_CHECK(cublasDaxpy(cublasHandle_, totalN_, &neg_alpha, d_AD_, 1, d_R_, 1));

        // Check convergence
        CUBLAS_CHECK(cublasDnrm2(cublasHandle_, totalN_, d_R_, 1, &rnorm));
        lastResidual_ = rnorm / bnorm;
        lastIterCount_ = iter + 1;

        if (lastResidual_ < pcgTol_) break;

        // Z = preconditioner(R)
        applyPreconditioner(d_Z_, d_R_);

        // rho_new = R^T Z
        double rho_new;
        CUBLAS_CHECK(cublasDdot(cublasHandle_, totalN_, d_R_, 1, d_Z_, 1, &rho_new));

        // beta = rho_new / rho
        if (rho == 0.0) break;
        double beta = rho_new / rho;

        // D = Z + beta * D
        CUBLAS_CHECK(cublasDscal(cublasHandle_, totalN_, &beta, d_D_, 1));
        double one = 1.0;
        CUBLAS_CHECK(cublasDaxpy(cublasHandle_, totalN_, &one, d_Z_, 1, d_D_, 1));

        rho = rho_new;
    }

    // Download result
    CUDA_CHECK(cudaMemcpy(vals.data(), d_P_, totalN_ * sizeof(double), cudaMemcpyDeviceToHost));
}

}  // namespace GridForcePlugin
