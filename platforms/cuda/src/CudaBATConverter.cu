/* CUDA implementation of cartesian <-> BAT transforms.
 *
 * One block per replica, threadIdx.x = 0 only. The atom-by-atom loop is
 * sequential (each atom depends on parent/grandparent/great-grandparent),
 * so per-replica parallelism is forced; we instead parallelize across
 * replicas via blockIdx.x.
 *
 * For our typical N=58 ligands and K=100-200 replicas this gives plenty
 * of occupancy. For much larger ligands a tree-DFS warp-level kernel
 * would help, but is unnecessary for the BPMF use case.
 */

#include "CudaBATConverter.h"

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
// Device math helpers (deterministic, IEEE intrinsics)
// ============================================================================

__device__ __forceinline__ float3 vsub(float3 a, float3 b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
__device__ __forceinline__ float3 vadd(float3 a, float3 b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
__device__ __forceinline__ float3 vscale(float3 a, float s) {
    return make_float3(a.x * s, a.y * s, a.z * s);
}
__device__ __forceinline__ float vdot(float3 a, float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
__device__ __forceinline__ float3 vcross(float3 a, float3 b) {
    return make_float3(a.y * b.z - a.z * b.y,
                       a.z * b.x - a.x * b.z,
                       a.x * b.y - a.y * b.x);
}
__device__ __forceinline__ float vnorm(float3 a) {
    return sqrtf(a.x * a.x + a.y * a.y + a.z * a.z);
}
__device__ __forceinline__ float3 vnormalize(float3 a) {
    float n = vnorm(a);
    return make_float3(a.x / n, a.y / n, a.z / n);
}

__device__ __forceinline__ float bondAngleDev(float3 a, float3 b, float3 c) {
    float3 v1 = vsub(a, b);
    float3 v2 = vsub(c, b);
    float n1 = vnorm(v1), n2 = vnorm(v2);
    float cs = vdot(v1, v2) / (n1 * n2);
    cs = fmaxf(-1.0f, fminf(1.0f, cs));
    return acosf(cs);
}

/**
 * Dihedral matching MDAnalysis sign convention: the value returned by
 * `bat_coords._dihedral`. Note the negation at the end.
 */
__device__ __forceinline__ float dihedralDev(float3 p0, float3 p1,
                                              float3 p2, float3 p3) {
    float3 b1 = vsub(p1, p0);
    float3 b2 = vsub(p2, p1);
    float3 b3 = vsub(p3, p2);
    float3 n1 = vcross(b1, b2);
    float3 n2 = vcross(b2, b3);
    float n_b2 = vnorm(b2);
    float3 b2_hat = vscale(b2, 1.0f / n_b2);
    float3 m = vcross(n1, b2_hat);
    float y = vdot(m, n2);
    float x = vdot(n1, n2);
    return -atan2f(y, x);
}


// ============================================================================
// Forward kernel: Cartesian -> BAT
// ============================================================================

__device__ __forceinline__ float wrapPi(float x) {
    // Wrap to [-pi, pi]
    const float TWO_PI = 6.28318530717958647692f;
    const float PI = 3.14159265358979323846f;
    float w = fmodf(x + PI, TWO_PI);
    if (w < 0) w += TWO_PI;
    return w - PI;
}

__global__ void cartesianToBATKernel(
    const float* __restrict__ pos_all,   // [K * N * 3]
    float* __restrict__ bat_all,         // [K * 3 * N]
    const int* __restrict__ root,        // [3]
    const int* __restrict__ torsions,    // [4 * (N-3)]
    const int* __restrict__ primary,     // [N-3] or nullptr
    int N, int K)
{
    int r = blockIdx.x;
    if (r >= K) return;
    if (threadIdx.x != 0) return;

    const float3* pos = (const float3*)(pos_all + r * N * 3);
    float* bat = bat_all + r * 3 * N;

    int p0_idx = root[0];
    int p1_idx = root[1];
    int p2_idx = root[2];
    float3 p0 = pos[p0_idx];
    float3 p1 = pos[p1_idx];
    float3 p2 = pos[p2_idx];

    // Origin = position of root atom 0
    bat[0] = p0.x; bat[1] = p0.y; bat[2] = p0.z;

    // r01 = |p1 - p0|, r12 = |p1 - p2|, a012 angle at p1.
    float3 v01 = vsub(p1, p0);
    float3 v21 = vsub(p1, p2);
    float r01 = vnorm(v01);
    float r12 = vnorm(v21);
    float a012;
    {
        float cs = vdot(v01, v21) / (r01 * r12);
        cs = fmaxf(-1.0f, fminf(1.0f, cs));
        a012 = acosf(cs);
    }

    // External Euler-like (phi, theta, omega) — direction of v01 + rotation
    // around it that places p2 in the xz-plane after Rz alignment.
    float3 e = vscale(v01, 1.0f / r01);
    float phi = atan2f(e.y, e.x);
    float theta = acosf(fmaxf(-1.0f, fminf(1.0f, e.z)));
    float cp = cosf(phi), sp = sinf(phi);
    float ct = cosf(theta), st = sinf(theta);
    // Rz @ (p2 - p1)
    float3 d = vsub(p2, p1);
    float pos2x = cp*ct * d.x + ct*sp * d.y + (-st) * d.z;
    float pos2y = -sp * d.x + cp * d.y + 0.0f * d.z;
    // pos2z not needed for omega
    float omega = atan2f(pos2y, pos2x);

    bat[3] = phi;
    bat[4] = theta;
    bat[5] = omega;
    bat[6] = r01;
    bat[7] = r12;
    bat[8] = a012;

    int n_t = N - 3;
    int bond_off = 9;
    int angle_off = 9 + n_t;
    int torsion_off = 9 + 2 * n_t;

    for (int i = 0; i < n_t; ++i) {
        int a0 = torsions[4*i + 0];
        int a1 = torsions[4*i + 1];
        int a2 = torsions[4*i + 2];
        int a3 = torsions[4*i + 3];
        float3 q0 = pos[a0];
        float3 q1 = pos[a1];
        float3 q2 = pos[a2];
        float3 q3 = pos[a3];
        bat[bond_off + i]    = vnorm(vsub(q0, q1));
        bat[angle_off + i]   = bondAngleDev(q0, q1, q2);
        bat[torsion_off + i] = dihedralDev(q0, q1, q2, q3);
    }
    // Improper-torsion shift: secondary torsions sharing a (a1, a2)
    // central bond with an earlier "primary" torsion are stored as
    // offsets (T_i - T_primary). Primary torsions store the raw value.
    // Mirror MDAnalysis convention. Skip if no primary array supplied.
    if (primary != nullptr) {
        // Snapshot torsions because shift refers to the *raw* values.
        // n_t = N-3, fits comfortably in registers/local memory.
        for (int i = 0; i < n_t; ++i) {
            int p = primary[i];
            if (p == i) continue;             // primary, no shift
            float t_raw = bat[torsion_off + i];
            float t_p   = bat[torsion_off + p];  // primary's raw value (already in array)
            bat[torsion_off + i] = wrapPi(t_raw - t_p);
        }
    }
}


// ============================================================================
// Inverse kernel: BAT -> Cartesian
// ============================================================================

__global__ void BATToCartesianKernel(
    const float* __restrict__ bat_all,
    float* __restrict__ pos_all,
    const int* __restrict__ root,
    const int* __restrict__ torsions,
    const int* __restrict__ primary,
    int N, int K)
{
    int r = blockIdx.x;
    if (r >= K) return;
    if (threadIdx.x != 0) return;

    const float* bat = bat_all + r * 3 * N;
    float3* pos = (float3*)(pos_all + r * N * 3);

    float3 origin = make_float3(bat[0], bat[1], bat[2]);
    float phi = bat[3], theta = bat[4], omega = bat[5];
    float r01 = bat[6], r12 = bat[7], a012 = bat[8];

    int n_t = N - 3;
    int bond_off = 9;
    int angle_off = 9 + n_t;
    int torsion_off = 9 + 2 * n_t;

    // Reverse the improper-torsion shift before NeRF.
    // Each replica stores its torsion block in the input bat[] buffer; we
    // need a local working copy because we'll add primary's value to
    // secondaries and the primary value itself must be the unshifted one.
    // For N=58 -> n_t=55 floats; fits easily in registers/local memory.
    extern __shared__ float s_torsions[];  // size = blockDim.x * n_t per block
    // We use threadIdx.x = 0 only, so just take the first n_t slots.
    float* tor_local = s_torsions;
    for (int i = 0; i < n_t; ++i) tor_local[i] = bat[torsion_off + i];
    if (primary != nullptr) {
        for (int i = 0; i < n_t; ++i) {
            int p = primary[i];
            if (p == i) continue;
            // tor_local[p] is already the raw primary value (wasn't shifted).
            float v = tor_local[i] + tor_local[p];
            // Wrap to [-pi, pi] inline (no helper needed).
            const float TWO_PI = 6.28318530717958647692f;
            const float PI = 3.14159265358979323846f;
            float w = fmodf(v + PI, TWO_PI);
            if (w < 0) w += TWO_PI;
            tor_local[i] = w - PI;
        }
    }

    // Place root in canonical pose.
    float3 p0 = make_float3(0, 0, 0);
    float3 p1 = make_float3(0, 0, r01);
    float3 p2 = make_float3(r12 * sinf(a012), 0.0f,
                             r01 - r12 * cosf(a012));

    // Rotate p2 by omega around z.
    float co = cosf(omega), so = sinf(omega);
    p2 = make_float3(co * p2.x - so * p2.y,
                     so * p2.x + co * p2.y,
                     p2.z);

    // Re @ p1, Re @ p2
    float cp = cosf(phi), sp = sinf(phi);
    float ct = cosf(theta), st = sinf(theta);
    auto applyRe = [&](float3 v) {
        return make_float3(
            cp*ct * v.x + (-sp) * v.y + cp*st * v.z,
            ct*sp * v.x +  cp   * v.y + sp*st * v.z,
            -st  * v.x +   0.0f * v.y + ct    * v.z);
    };
    p1 = applyRe(p1);
    p2 = applyRe(p2);
    p0 = vadd(p0, origin);
    p1 = vadd(p1, origin);
    p2 = vadd(p2, origin);

    int p0_idx = root[0];
    int p1_idx = root[1];
    int p2_idx = root[2];
    pos[p0_idx] = p0;
    pos[p1_idx] = p1;
    pos[p2_idx] = p2;

    // NeRF placement of the remaining atoms.
    for (int i = 0; i < n_t; ++i) {
        int a0 = torsions[4*i + 0];
        int a1 = torsions[4*i + 1];
        int a2 = torsions[4*i + 2];
        int a3 = torsions[4*i + 3];
        float bond = bat[bond_off + i];
        float angle = bat[angle_off + i];
        float tor = tor_local[i];

        float3 q1 = pos[a1];
        float3 q2 = pos[a2];
        float3 q3 = pos[a3];
        float sn_a = sinf(angle), cs_a = cosf(angle);
        float sn_t = sinf(tor),   cs_t = cosf(tor);

        float3 v21 = vnormalize(vsub(q1, q2));
        float3 v32 = vnormalize(vsub(q2, q3));
        float3 vp = vcross(v32, v21);
        float cs = vdot(v21, v32);
        float sn = sqrtf(fmaxf(1.0f - cs * cs, 1e-24f));
        vp = vscale(vp, 1.0f / sn);
        float3 vu = vcross(vp, v21);

        // p1 + bond * (-cs_a * v21 + sn_a * (cs_t * vu + sn_t * vp))
        float3 contrib = vadd(vscale(v21, -cs_a),
                               vadd(vscale(vu, sn_a * cs_t),
                                    vscale(vp, sn_a * sn_t)));
        pos[a0] = vadd(q1, vscale(contrib, bond));
    }
}


// ============================================================================
// Host class
// ============================================================================

CudaBATConverter::CudaBATConverter() {}

void CudaBATConverter::initialize(const BATTopology& topo,
                                    OpenMM::Context& context, int maxK) {
    n_atoms_ = topo.getNumAtoms();
    max_K_ = maxK;
    if (n_atoms_ < 3)
        throw std::runtime_error("CudaBATConverter: topology has < 3 atoms");

    // Get OpenMM's CUDA context so our allocations and launches share it.
    auto* impl = *reinterpret_cast<OpenMM::ContextImpl**>(&context);
    auto* data = static_cast<OpenMM::CudaPlatform::PlatformData*>(
        impl->getPlatformData());
    if (!data || data->contexts.empty())
        throw std::runtime_error(
            "CudaBATConverter: Context must use the CUDA platform");
    OpenMM::CudaContext* cu = data->contexts[0];
    cu_ = cu;
    cu->setAsCurrent();

    int n_t = n_atoms_ - 3;
    int n_t4 = 4 * n_t;
    CUDA_CHECK(cudaMalloc(&d_root_, sizeof(int) * 3));
    CUDA_CHECK(cudaMalloc(&d_torsions_, sizeof(int) * n_t4));
    CUDA_CHECK(cudaMemcpy(d_root_, topo.getRoot().data(), sizeof(int) * 3,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_torsions_, topo.getTorsions().data(),
                          sizeof(int) * n_t4, cudaMemcpyHostToDevice));
    const std::vector<int>& prim = topo.getPrimaryTorsionIndices();
    if (!prim.empty()) {
        if ((int)prim.size() != n_t)
            throw std::runtime_error(
                "CudaBATConverter: primary indices wrong length");
        CUDA_CHECK(cudaMalloc(&d_primary_, sizeof(int) * n_t));
        CUDA_CHECK(cudaMemcpy(d_primary_, prim.data(), sizeof(int) * n_t,
                              cudaMemcpyHostToDevice));
        has_primary_ = true;
    }
}

CudaBATConverter::~CudaBATConverter() {
    if (cu_) static_cast<OpenMM::CudaContext*>(cu_)->setAsCurrent();
    if (d_root_) cudaFree(d_root_);
    if (d_torsions_) cudaFree(d_torsions_);
    if (d_primary_) cudaFree(d_primary_);
}

void CudaBATConverter::setMaxReplicas(int newMaxK) {
    if (newMaxK > max_K_) max_K_ = newMaxK;
}

void CudaBATConverter::cartesianToBAT(const float* d_pos, float* d_BAT,
                                       int K, cudaStream_t stream) {
    if (K <= 0) return;
    if (cu_) static_cast<OpenMM::CudaContext*>(cu_)->setAsCurrent();
    cartesianToBATKernel<<<K, 1, 0, stream>>>(
        d_pos, d_BAT, d_root_, d_torsions_, d_primary_, n_atoms_, K);
    CUDA_CHECK(cudaGetLastError());
}

void CudaBATConverter::BATToCartesian(const float* d_BAT, float* d_pos,
                                       int K, cudaStream_t stream) {
    if (K <= 0) return;
    if (cu_) static_cast<OpenMM::CudaContext*>(cu_)->setAsCurrent();
    int n_t = n_atoms_ - 3;
    size_t shm_bytes = sizeof(float) * n_t;  // tor_local[]
    BATToCartesianKernel<<<K, 1, shm_bytes, stream>>>(
        d_BAT, d_pos, d_root_, d_torsions_, d_primary_, n_atoms_, K);
    CUDA_CHECK(cudaGetLastError());
}

// ----- Host-buffer convenience overloads -----

std::vector<float> CudaBATConverter::cartesianToBATHost(
    const std::vector<float>& positions, int K) {
    int N = n_atoms_;
    if ((int)positions.size() != K * N * 3)
        throw std::runtime_error("cartesianToBATHost: positions size mismatch");
    if (cu_) static_cast<OpenMM::CudaContext*>(cu_)->setAsCurrent();
    float* d_pos = nullptr;
    float* d_bat = nullptr;
    CUDA_CHECK(cudaMalloc(&d_pos, sizeof(float) * positions.size()));
    CUDA_CHECK(cudaMalloc(&d_bat, sizeof(float) * K * 3 * N));
    CUDA_CHECK(cudaMemcpy(d_pos, positions.data(),
                          sizeof(float) * positions.size(),
                          cudaMemcpyHostToDevice));
    cartesianToBAT(d_pos, d_bat, K);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> out(K * 3 * N);
    CUDA_CHECK(cudaMemcpy(out.data(), d_bat, sizeof(float) * out.size(),
                          cudaMemcpyDeviceToHost));
    cudaFree(d_pos); cudaFree(d_bat);
    return out;
}

std::vector<float> CudaBATConverter::BATToCartesianHost(
    const std::vector<float>& bat, int K) {
    int N = n_atoms_;
    if ((int)bat.size() != K * 3 * N)
        throw std::runtime_error("BATToCartesianHost: bat size mismatch");
    if (cu_) static_cast<OpenMM::CudaContext*>(cu_)->setAsCurrent();
    float* d_pos = nullptr;
    float* d_bat = nullptr;
    CUDA_CHECK(cudaMalloc(&d_pos, sizeof(float) * K * N * 3));
    CUDA_CHECK(cudaMalloc(&d_bat, sizeof(float) * bat.size()));
    CUDA_CHECK(cudaMemcpy(d_bat, bat.data(), sizeof(float) * bat.size(),
                          cudaMemcpyHostToDevice));
    BATToCartesian(d_bat, d_pos, K);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> out(K * N * 3);
    CUDA_CHECK(cudaMemcpy(out.data(), d_pos, sizeof(float) * out.size(),
                          cudaMemcpyDeviceToHost));
    cudaFree(d_pos); cudaFree(d_bat);
    return out;
}

}  // namespace GridForcePlugin
