/* CUDA kernels and host class for smart darting.
 *
 * Two kernels:
 *   findNearestTargetKernel  — per replica, scans M targets sequentially
 *                              (single thread, deterministic).
 *   proposeDartKernel        — per replica, computes BAT' = BAT + (target[k]
 *                              - target[j]) on perturbable DoFs with angular
 *                              wrap.
 *
 * The host orchestrates the full dart attempt by chaining
 * cartesianToBAT → findNearest → (host-sample k) → proposeDart →
 * BATToCartesian → cartesianToBAT → findNearest (verify).
 */

#include "CudaSmartDartingPool.h"

#include <cuda_runtime.h>
#include "openmm/Context.h"
#include "openmm/cuda/CudaContext.h"
#include "openmm/cuda/CudaPlatform.h"
#include "openmm/internal/ContextImpl.h"
#include <stdexcept>
#include <string>

namespace GridForcePlugin {

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        throw std::runtime_error(std::string("CUDA error ") + __FILE__ + ":" \
            + std::to_string(__LINE__) + ": " + cudaGetErrorString(err)); \
    } \
} while (0)


// ============================================================================
// Device helpers
// ============================================================================

__device__ __forceinline__ float wrapPiF(float x) {
    const float TWO_PI = 6.28318530717958647692f;
    const float PI = 3.14159265358979323846f;
    float w = fmodf(x + PI, TWO_PI);
    if (w < 0) w += TWO_PI;
    return w - PI;
}

/**
 * Compute squared "perturbable BAT distance" between bat[] and target[],
 * over the dimensions where mask=1. Angular DoFs (ang_idx=1) use wrapped
 * differences; Cartesian DoFs use raw differences.
 */
__device__ float perturbBATDistSq(const float* bat, const float* target,
                                   const int* mask, const int* ang_idx,
                                   int dim) {
    float acc = 0.0f;
    for (int i = 0; i < dim; ++i) {
        if (mask[i] == 0) continue;
        float d = bat[i] - target[i];
        if (ang_idx[i]) d = wrapPiF(d);
        acc += d * d;
    }
    return acc;
}


// ============================================================================
// Kernels
// ============================================================================

__global__ void findNearestTargetKernel(
    const float* __restrict__ bat_all,    // [K * dim]
    const float* __restrict__ targets,    // [M * dim]
    const int*   __restrict__ mask,       // [dim]
    const int*   __restrict__ ang_idx,    // [dim]
    int*         __restrict__ j_out,      // [K]
    float*       __restrict__ d2_out,     // [K]
    int K, int M, int dim)
{
    int r = blockIdx.x;
    if (r >= K) return;
    if (threadIdx.x != 0) return;

    const float* bat = bat_all + r * dim;
    int   best_j = 0;
    float best_d2 = INFINITY;
    for (int t = 0; t < M; ++t) {
        const float* tgt = targets + t * dim;
        float d2 = perturbBATDistSq(bat, tgt, mask, ang_idx, dim);
        // Stable tie-break: keep lower index on equal distance.
        if (d2 < best_d2) {
            best_d2 = d2;
            best_j = t;
        }
    }
    j_out[r] = best_j;
    d2_out[r] = best_d2;
}


__global__ void proposeDartKernel(
    const float* __restrict__ bat_all,         // [K * dim]
    const float* __restrict__ targets,         // [M * dim]
    const int*   __restrict__ j_per_replica,   // [K]
    const int*   __restrict__ k_per_replica,   // [K]
    const int*   __restrict__ mask,            // [dim]
    const int*   __restrict__ ang_idx,         // [dim]
    float*       __restrict__ bat_proposed_all,// [K * dim]
    int K, int dim)
{
    int r = blockIdx.x;
    if (r >= K) return;
    if (threadIdx.x != 0) return;

    int j = j_per_replica[r];
    int k = k_per_replica[r];

    const float* bat = bat_all + r * dim;
    const float* tj = targets + j * dim;
    const float* tk = targets + k * dim;
    float* out = bat_proposed_all + r * dim;

    for (int i = 0; i < dim; ++i) {
        if (mask[i] == 0) {
            out[i] = bat[i];
            continue;
        }
        float delta = tk[i] - tj[i];
        if (ang_idx[i]) delta = wrapPiF(delta);
        float v = bat[i] + delta;
        if (ang_idx[i]) v = wrapPiF(v);
        out[i] = v;
    }
}


// ============================================================================
// Host class
// ============================================================================

CudaSmartDartingPool::CudaSmartDartingPool() {}

void CudaSmartDartingPool::initialize(
    const BATTopology& topo,
    OpenMM::Context& context,
    const std::vector<float>& targets_BAT_flat,
    const std::vector<float>& weights,
    float epsilon_sq) {
    n_atoms_ = topo.getNumAtoms();
    M_ = (int)weights.size();
    epsilon_sq_ = epsilon_sq;
    int dim = 3 * n_atoms_;
    if ((int)targets_BAT_flat.size() != M_ * dim)
        throw std::runtime_error("CudaSmartDartingPool: targets size mismatch");
    if (topo.getPerturbableMask().size() != (size_t)dim)
        throw std::runtime_error("CudaSmartDartingPool: mask size mismatch");
    weights_host_ = weights;

    auto* impl = *reinterpret_cast<OpenMM::ContextImpl**>(&context);
    auto* data = static_cast<OpenMM::CudaPlatform::PlatformData*>(
        impl->getPlatformData());
    if (!data || data->contexts.empty())
        throw std::runtime_error(
            "CudaSmartDartingPool: Context must use the CUDA platform");
    OpenMM::CudaContext* cu = data->contexts[0];
    cu_ = cu;
    cu->setAsCurrent();

    CUDA_CHECK(cudaMalloc(&d_targets_, sizeof(float) * M_ * dim));
    CUDA_CHECK(cudaMemcpy(d_targets_, targets_BAT_flat.data(),
                          sizeof(float) * M_ * dim, cudaMemcpyHostToDevice));

    // Mask
    CUDA_CHECK(cudaMalloc(&d_mask_, sizeof(int) * dim));
    CUDA_CHECK(cudaMemcpy(d_mask_, topo.getPerturbableMask().data(),
                          sizeof(int) * dim, cudaMemcpyHostToDevice));

    // Angular-DoF mask: indices 3..6 (Euler) and the torsion block.
    int n_t = n_atoms_ - 3;
    std::vector<int> ang(dim, 0);
    for (int i = 3; i < 6; ++i) ang[i] = 1;
    for (int i = 9 + 2 * n_t; i < 9 + 3 * n_t; ++i) ang[i] = 1;
    CUDA_CHECK(cudaMalloc(&d_ang_idx_, sizeof(int) * dim));
    CUDA_CHECK(cudaMemcpy(d_ang_idx_, ang.data(),
                          sizeof(int) * dim, cudaMemcpyHostToDevice));
}

CudaSmartDartingPool::~CudaSmartDartingPool() {
    if (cu_) static_cast<OpenMM::CudaContext*>(cu_)->setAsCurrent();
    if (d_targets_)  cudaFree(d_targets_);
    if (d_mask_)     cudaFree(d_mask_);
    if (d_ang_idx_)  cudaFree(d_ang_idx_);
}

void CudaSmartDartingPool::findNearest(
    const float* d_bat, int K,
    int* d_j_out, float* d_d2_out, cudaStream_t stream) {
    if (K <= 0) return;
    if (cu_) static_cast<OpenMM::CudaContext*>(cu_)->setAsCurrent();
    int dim = 3 * n_atoms_;
    findNearestTargetKernel<<<K, 1, 0, stream>>>(
        d_bat, d_targets_, d_mask_, d_ang_idx_, d_j_out, d_d2_out, K, M_, dim);
    CUDA_CHECK(cudaGetLastError());
}

void CudaSmartDartingPool::proposeDart(
    const float* d_bat, const int* d_j, const int* d_k,
    float* d_bat_proposed, int K, cudaStream_t stream) {
    if (K <= 0) return;
    if (cu_) static_cast<OpenMM::CudaContext*>(cu_)->setAsCurrent();
    int dim = 3 * n_atoms_;
    proposeDartKernel<<<K, 1, 0, stream>>>(
        d_bat, d_targets_, d_j, d_k, d_mask_, d_ang_idx_,
        d_bat_proposed, K, dim);
    CUDA_CHECK(cudaGetLastError());
}

// ----- Host-buffer convenience -----

std::vector<int> CudaSmartDartingPool::findNearestHost(
    const std::vector<float>& bat, int K) {
    int dim = 3 * n_atoms_;
    if (cu_) static_cast<OpenMM::CudaContext*>(cu_)->setAsCurrent();
    if ((int)bat.size() != K * dim)
        throw std::runtime_error("findNearestHost: bat size mismatch");
    float* d_bat = nullptr;
    int*   d_j   = nullptr;
    float* d_d2  = nullptr;
    CUDA_CHECK(cudaMalloc(&d_bat, sizeof(float) * bat.size()));
    CUDA_CHECK(cudaMalloc(&d_j, sizeof(int) * K));
    CUDA_CHECK(cudaMalloc(&d_d2, sizeof(float) * K));
    CUDA_CHECK(cudaMemcpy(d_bat, bat.data(), sizeof(float) * bat.size(),
                          cudaMemcpyHostToDevice));
    findNearest(d_bat, K, d_j, d_d2);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<int> out(K);
    CUDA_CHECK(cudaMemcpy(out.data(), d_j, sizeof(int) * K,
                          cudaMemcpyDeviceToHost));
    cudaFree(d_bat); cudaFree(d_j); cudaFree(d_d2);
    return out;
}

std::vector<float> CudaSmartDartingPool::findNearestDistsHost(
    const std::vector<float>& bat, int K) {
    int dim = 3 * n_atoms_;
    if (cu_) static_cast<OpenMM::CudaContext*>(cu_)->setAsCurrent();
    if ((int)bat.size() != K * dim)
        throw std::runtime_error("findNearestDistsHost: bat size mismatch");
    float* d_bat = nullptr;
    int*   d_j   = nullptr;
    float* d_d2  = nullptr;
    CUDA_CHECK(cudaMalloc(&d_bat, sizeof(float) * bat.size()));
    CUDA_CHECK(cudaMalloc(&d_j, sizeof(int) * K));
    CUDA_CHECK(cudaMalloc(&d_d2, sizeof(float) * K));
    CUDA_CHECK(cudaMemcpy(d_bat, bat.data(), sizeof(float) * bat.size(),
                          cudaMemcpyHostToDevice));
    findNearest(d_bat, K, d_j, d_d2);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> out(K);
    CUDA_CHECK(cudaMemcpy(out.data(), d_d2, sizeof(float) * K,
                          cudaMemcpyDeviceToHost));
    cudaFree(d_bat); cudaFree(d_j); cudaFree(d_d2);
    return out;
}

std::vector<float> CudaSmartDartingPool::proposeDartHost(
    const std::vector<float>& bat,
    const std::vector<int>& j_per,
    const std::vector<int>& k_per, int K) {
    int dim = 3 * n_atoms_;
    if (cu_) static_cast<OpenMM::CudaContext*>(cu_)->setAsCurrent();
    if ((int)bat.size() != K * dim)
        throw std::runtime_error("proposeDartHost: bat size mismatch");
    if ((int)j_per.size() != K || (int)k_per.size() != K)
        throw std::runtime_error("proposeDartHost: j/k size mismatch");
    float* d_bat = nullptr;
    float* d_out = nullptr;
    int*   d_j   = nullptr;
    int*   d_k   = nullptr;
    CUDA_CHECK(cudaMalloc(&d_bat, sizeof(float) * bat.size()));
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(float) * bat.size()));
    CUDA_CHECK(cudaMalloc(&d_j, sizeof(int) * K));
    CUDA_CHECK(cudaMalloc(&d_k, sizeof(int) * K));
    CUDA_CHECK(cudaMemcpy(d_bat, bat.data(), sizeof(float) * bat.size(),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_j, j_per.data(), sizeof(int) * K,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_k, k_per.data(), sizeof(int) * K,
                          cudaMemcpyHostToDevice));
    proposeDart(d_bat, d_j, d_k, d_out, K);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> out(bat.size());
    CUDA_CHECK(cudaMemcpy(out.data(), d_out, sizeof(float) * out.size(),
                          cudaMemcpyDeviceToHost));
    cudaFree(d_bat); cudaFree(d_out); cudaFree(d_j); cudaFree(d_k);
    return out;
}

}  // namespace GridForcePlugin
