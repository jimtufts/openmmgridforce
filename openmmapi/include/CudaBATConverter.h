#ifndef OPENMM_CUDA_BAT_CONVERTER_H_
#define OPENMM_CUDA_BAT_CONVERTER_H_

/**
 * Cartesian <-> BAT converter (CUDA).
 *
 * Holds the topology arrays in device memory and provides forward / inverse
 * BAT transforms over K replicas in parallel. Forward and inverse are exact
 * round-trip in single precision (validated against bat_coords.py).
 *
 * Determinism guarantees:
 *   - Per-replica work is single-thread (one block per replica, threadIdx.x=0).
 *     The sequential atom loop has fixed order; no FP atomics, no reductions.
 *   - Transcendentals via IEEE-compliant intrinsics (sinf/cosf/atan2f/sqrtf),
 *     not the __sinf-style fast variants. Bit-deterministic per CUDA driver.
 *   - No reliance on warp ballots or shared atomic accumulation.
 *
 * Layout (matches BATTopology + bat_coords.py):
 *   bat[0:3]   root atom 0 origin
 *   bat[3:6]   external rotation (phi, theta, omega)
 *   bat[6:9]   root internals (r01, r12, a012)
 *   bat[9 : 9+(N-3)]               bond lengths
 *   bat[9+(N-3) : 9+2(N-3)]        bond angles
 *   bat[9+2(N-3) : 9+3(N-3)]       torsion angles
 */

#include "BATTopology.h"

#include <vector>

// Forward-declare the CUDA stream type so this header does not pull in
// cuda_runtime.h. Host-only translation units (e.g. the SWIG wrapper) can
// include this file safely.
typedef struct CUstream_st* cudaStream_t;

#include "openmm/Context.h"

namespace GridForcePlugin {

class CudaBATConverter {
public:
    /** Empty constructor; call `initialize` before any other method. */
    CudaBATConverter();
    ~CudaBATConverter();

    /**
     * Initialize device buffers using OpenMM's CUDA context.
     * Must be called once before any other method.
     */
    void initialize(const BATTopology& topo,
                    OpenMM::Context& context, int maxK = 256);

    CudaBATConverter(const CudaBATConverter&) = delete;
    CudaBATConverter& operator=(const CudaBATConverter&) = delete;

    int getNumAtoms() const { return n_atoms_; }
    int getNumTorsions() const { return n_atoms_ - 3; }

    /**
     * Forward: positions [K * N * 3] (float, replica-major) -> BAT [K * 3*N].
     * Both buffers are caller-allocated device memory.
     */
    void cartesianToBAT(const float* d_pos, float* d_BAT, int K,
                        cudaStream_t stream = 0);

    /** Inverse. */
    void BATToCartesian(const float* d_BAT, float* d_pos, int K,
                        cudaStream_t stream = 0);

    /** Resize internal scratch buffers (no-op unless newK > current capacity). */
    void setMaxReplicas(int newMaxK);

    /**
     * Host-buffer convenience overloads (validation/debug only).
     * Allocates device memory, runs the kernel, returns host buffer.
     * The shape is `K * N * 3` for positions and `K * 3 * N` for BAT.
     */
    std::vector<float> cartesianToBATHost(const std::vector<float>& positions,
                                          int K);
    std::vector<float> BATToCartesianHost(const std::vector<float>& bat,
                                          int K);

private:
    void* cu_ = nullptr;           // CudaContext*, opaque in the header
    int n_atoms_ = 0;
    int max_K_ = 0;
    bool has_primary_ = false;

    // Device topology: root[3], torsions[4*(N-3)] flat.
    int* d_root_ = nullptr;        // length 3
    int* d_torsions_ = nullptr;    // length 4 * (N - 3)
    int* d_primary_ = nullptr;     // length N-3 (or null if disabled)
};

}  // namespace GridForcePlugin

#endif  // OPENMM_CUDA_BAT_CONVERTER_H_
