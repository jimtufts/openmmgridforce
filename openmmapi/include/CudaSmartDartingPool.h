#ifndef OPENMM_CUDA_SMART_DARTING_POOL_H_
#define OPENMM_CUDA_SMART_DARTING_POOL_H_

/**
 * Smart-darting pool (CUDA).
 *
 * Holds the immutable per-pool data (target BAT representations, Boltzmann
 * weights, perturbable mask, epsilon^2 eligibility threshold) on the device,
 * and provides the kernels needed for each dart attempt:
 *
 *   - `findNearest`: for each replica, find the index of the nearest target
 *     in BAT space (perturbable subset only), with stable tie-breaking by
 *     lower index. Returns squared distance for eligibility check.
 *
 *   - `proposeDart`: compute BAT' = BAT + (target[k] - target[j]) on
 *     perturbable DoFs only, with angular wrap on torsions and externals.
 *     Bonds and angles (and any other masked-out DoFs) are kept fixed.
 *
 * The actual move pipeline (forward BAT → findNearest → host samples k →
 * proposeDart → inverse BAT → forward → findNearest verify → energy →
 * Metropolis) is host-orchestrated to keep the kernels simple and to let
 * the caller plug in any energy function.
 *
 * Determinism notes (mirroring CudaBATConverter):
 *   - Single thread per replica; no FP atomics; sequential argmin with
 *     index tie-break for reproducible nearest-target choice.
 *   - Single-precision FP, IEEE-compliant intrinsics.
 *   - The host samples target k from per-replica weights (numpy
 *     `np.random.choice` with a deterministic seed). The kernel never
 *     calls cuRAND — k is an input.
 */

#include "BATTopology.h"

#include <vector>

typedef struct CUstream_st* cudaStream_t;

#include "openmm/Context.h"

namespace GridForcePlugin {

class CudaSmartDartingPool {
public:
    /**
     * @param topo                  Topology (provides perturbable mask
     *                              and primary indices).
     * @param targets_BAT_flat      M * 3*N floats: BAT representation of
     *                              each pool member, in topology layout
     *                              (origin / Euler / root internals /
     *                              bonds / angles / torsions, with the
     *                              improper-shift already applied if the
     *                              topology has primary_torsion_indices).
     * @param weights               length M, sum to 1, Boltzmann weights at T_HIGH.
     * @param epsilon_sq            eligibility threshold (squared BAT
     *                              distance, perturbable DoFs only).
     */
    /** Empty constructor; call `initialize` before any other method. */
    CudaSmartDartingPool();
    ~CudaSmartDartingPool();

    /** Construct the pool device buffers using OpenMM's CUDA context. */
    void initialize(const BATTopology& topo,
                    OpenMM::Context& context,
                    const std::vector<float>& targets_BAT_flat,
                    const std::vector<float>& weights,
                    float epsilon_sq);

    CudaSmartDartingPool(const CudaSmartDartingPool&) = delete;
    CudaSmartDartingPool& operator=(const CudaSmartDartingPool&) = delete;

    int getNumTargets() const { return M_; }
    int getNumAtoms() const { return n_atoms_; }
    int getNumHeavy() const { return n_heavy_; }
    float getEpsilonSq() const { return epsilon_sq_; }
    const std::vector<float>& getWeightsHost() const { return weights_host_; }

    /**
     * Find nearest target per replica (perturbable BAT distance).
     *
     * @param d_bat   [K * 3*N]  current BAT vectors.
     * @param d_j_out [K]        nearest target index per replica.
     * @param d_d2_out [K]       squared distance to nearest target.
     */
    void findNearest(const float* d_bat, int K,
                     int* d_j_out, float* d_d2_out,
                     cudaStream_t stream = 0);

    /**
     * Propose BAT' = BAT + (target[k] - target[j]) on perturbable DoFs.
     * Angular DoFs are wrapped to [-pi, pi].
     */
    void proposeDart(const float* d_bat, const int* d_j, const int* d_k,
                     float* d_bat_proposed, int K,
                     cudaStream_t stream = 0);

    /**
     * Upload heavy-atom Cartesian targets for the CD-path closest-pose
     * metric (matches reference SmartDarting.py:303-304, extended=True).
     *
     * @param heavy_inds          length N_h, atom indices into the N-atom
     *                            ligand topology.
     * @param targets_heavy_flat  M * N_h * 3 floats, in (target, atom,
     *                            xyz) row-major order. Caller is
     *                            responsible for matching the per-target
     *                            ordering with the BAT targets passed
     *                            into initialize().
     */
    void setCartesianTargets(const std::vector<int>& heavy_inds,
                              const std::vector<float>& targets_heavy_flat);

    /**
     * Upload the full per-pool Cartesian poses (all atoms) for use by
     * proposeCartesianDart. Stored as M * N * 3 floats in (target, atom,
     * xyz) order.
     */
    void setCartesianFullTargets(const std::vector<float>& targets_full_flat);

    /**
     * Cartesian dart proposal: pos_out[a] = pos_in[a] + (target_k[a] -
     * target_j[a]) for every atom a. Lands the proposed pose at k's
     * geometry plus the current pose's drift from j, which preserves the
     * sampler's local thermal fluctuation while jumping basins. No BAT
     * round-trip, so no reconstruction error.
     *
     * @param d_pos_in    [K * N * 3]  current ligand positions.
     * @param d_j_per     [K]          source target indices (one per replica).
     * @param d_k_per     [K]          destination target indices (one per replica).
     * @param d_pos_out   [K * N * 3]  proposed positions.
     */
    void proposeCartesianDart(const float* d_pos_in,
                               const int* d_j_per, const int* d_k_per,
                               float* d_pos_out, int K,
                               cudaStream_t stream = 0);

    /**
     * Find nearest target per replica using Cartesian heavy-atom SSD.
     *
     * @param d_pos   [K * N * 3]  current ligand positions.
     * @param d_j_out [K]          nearest target index per replica.
     * @param d_d2_out [K]         squared SSD to nearest target.
     */
    void findNearestCartesian(const float* d_pos, int K,
                               int* d_j_out, float* d_d2_out,
                               cudaStream_t stream = 0);

    // ---- host-buffer convenience for validation ----
    std::vector<int> findNearestHost(const std::vector<float>& bat, int K);
    std::vector<float> findNearestDistsHost(const std::vector<float>& bat, int K);
    std::vector<float> proposeDartHost(const std::vector<float>& bat,
                                       const std::vector<int>& j_per,
                                       const std::vector<int>& k_per, int K);
    std::vector<int> findNearestCartesianHost(
        const std::vector<float>& pos, int K);
    std::vector<float> findNearestCartesianDistsHost(
        const std::vector<float>& pos, int K);
    std::vector<float> proposeCartesianDartHost(
        const std::vector<float>& pos,
        const std::vector<int>& j_per,
        const std::vector<int>& k_per, int K);

private:
    void* cu_ = nullptr;
    int n_atoms_ = 0;
    int M_ = 0;
    int n_heavy_ = 0;
    float epsilon_sq_ = 0.0f;
    std::vector<float> weights_host_;

    float* d_targets_ = nullptr;          // M * 3*N
    int*   d_mask_ = nullptr;             // 3*N (0/1)
    int*   d_ang_idx_ = nullptr;          // 3*N (1 if angular DoF: 3..6 + torsion block)
    int*   d_heavy_inds_ = nullptr;       // N_h (atom indices)
    float* d_targets_heavy_ = nullptr;    // M * N_h * 3
    float* d_targets_full_ = nullptr;     // M * N * 3 (for Cartesian dart proposal)
};

}  // namespace GridForcePlugin

#endif  // OPENMM_CUDA_SMART_DARTING_POOL_H_
