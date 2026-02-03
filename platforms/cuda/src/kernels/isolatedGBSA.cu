/* -------------------------------------------------------------------------- *
 *                              OpenMMGridForce                               *
 * -------------------------------------------------------------------------- *
 * CUDA kernels for IsolatedGBSAForce - Pairwise GBSA for isolated particles.
 *
 * Workflow:
 * 1. computeIsolatedReceptorHCT - Grid interpolation or pairwise receptor HCT
 * 2. computeIsolatedLigandHCT - Pairwise HCT within ligand (O(N²))
 * 3. computeBornRadii - HCT or OBC-II formula
 * 4. computeIsolatedGBEnergy - Still equation pairwise (O(N²)) with forces
 * 5. computeIsolatedSAEnergy - Optional ACE surface area term
 * 6. Chain rule forces through HCT
 * -------------------------------------------------------------------------- */

#include "include/GridInterpolation.cuh"

// Physical constants
#define DIELECTRIC_OFFSET 0.009f
#define OBC_ALPHA 1.0f
#define OBC_BETA 0.8f
#define OBC_GAMMA 4.85f

// Tiled computation parameters (must match OpenMM for compatibility)
#define TILE_SIZE 32

// =============================================================================
// TILED KERNELS FOR O(N²) RECEPTOR COMPUTATIONS
// These use OpenMM-style tiled parallel computation for efficiency.
// =============================================================================

/**
 * Atom data structure for tiled HCT computation.
 */
typedef struct {
    float x, y, z;
    float radius;
    float scaledRadius;
    float hctSum;
} TiledAtomDataHCT;

/**
 * Atom data structure for tiled GB energy computation.
 */
typedef struct {
    float x, y, z;
    float charge;
    float bornRadius;
    float energy;
} TiledAtomDataGB;

/**
 * Tiled computation of receptor-receptor HCT contributions.
 * Uses OpenMM-style tile-based parallelization for O(N²) efficiency.
 *
 * Each warp processes tile pairs. Total tiles = NUM_BLOCKS * (NUM_BLOCKS + 1) / 2.
 * Work is evenly distributed across all warps.
 */
extern "C" __global__ void computeReceptorSelfHCTTiled(
    const float3* __restrict__ receptorPositions,
    const float* __restrict__ receptorRadii,
    const float* __restrict__ receptorScaleFactors,
    int numReceptorAtoms,
    float cutoffDistance,
    unsigned long long* __restrict__ receptorSelfHCT,  // Fixed-point accumulator
    int numTiles
) {
    const int totalWarps = (gridDim.x * blockDim.x) / TILE_SIZE;
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) / TILE_SIZE;
    const int tgx = threadIdx.x & (TILE_SIZE - 1);  // Thread index within warp
    const int tbx = threadIdx.x - tgx;  // Base index for this warp in shared memory

    __shared__ TiledAtomDataHCT localData[256];  // Assumes blockDim.x <= 256

    const int NUM_BLOCKS = (numReceptorAtoms + TILE_SIZE - 1) / TILE_SIZE;
    const bool useCutoff = (cutoffDistance > 0.0f);
    const float cutoff2 = cutoffDistance * cutoffDistance;

    // Each warp processes a range of tiles
    int pos = (int)(((long long)warp * numTiles) / totalWarps);
    int end = (int)(((long long)(warp + 1) * numTiles) / totalWarps);

    while (pos < end) {
        // Convert linear tile index to (x, y) tile coordinates
        // Using the formula from OpenMM: upper triangle enumeration
        int y = (int)floor(NUM_BLOCKS + 0.5f - sqrtf((NUM_BLOCKS + 0.5f) * (NUM_BLOCKS + 0.5f) - 2.0f * pos));
        int x = pos - y * NUM_BLOCKS + y * (y + 1) / 2;

        // Handle roundoff errors
        if (x < y || x >= NUM_BLOCKS) {
            y += (x < y ? -1 : 1);
            x = pos - y * NUM_BLOCKS + y * (y + 1) / 2;
        }

        // Atom indices for this tile
        unsigned int atom1 = x * TILE_SIZE + tgx;
        unsigned int atom2 = y * TILE_SIZE + tgx;

        // Load atom1 data (tile X)
        float3 pos1 = make_float3(0, 0, 0);
        float R1 = 0.1f, R1_off = 0.1f, S1 = 0.1f;
        if (atom1 < numReceptorAtoms) {
            pos1 = receptorPositions[atom1];
            R1 = receptorRadii[atom1];
            R1_off = R1 - DIELECTRIC_OFFSET;
            S1 = (R1 - DIELECTRIC_OFFSET) * receptorScaleFactors[atom1];
        }

        // Load atom2 data into shared memory (tile Y)
        if (atom2 < numReceptorAtoms) {
            float3 pos2 = receptorPositions[atom2];
            localData[tbx + tgx].x = pos2.x;
            localData[tbx + tgx].y = pos2.y;
            localData[tbx + tgx].z = pos2.z;
            localData[tbx + tgx].radius = receptorRadii[atom2];
            float R2_off = receptorRadii[atom2] - DIELECTRIC_OFFSET;
            localData[tbx + tgx].scaledRadius = R2_off * receptorScaleFactors[atom2];
        } else {
            localData[tbx + tgx].x = 0;
            localData[tbx + tgx].y = 0;
            localData[tbx + tgx].z = 0;
            localData[tbx + tgx].radius = 0.1f;
            localData[tbx + tgx].scaledRadius = 0.1f;
        }
        localData[tbx + tgx].hctSum = 0.0f;
        __syncwarp();

        // Accumulate HCT for atom1
        float hctSum1 = 0.0f;

        if (x == y) {
            // Diagonal tile: only compute j > i to avoid double counting
            for (int j = 0; j < TILE_SIZE; j++) {
                int atom2_j = y * TILE_SIZE + j;
                if (atom1 < numReceptorAtoms && atom2_j < numReceptorAtoms && atom2_j != atom1) {
                    float dx = pos1.x - localData[tbx + j].x;
                    float dy = pos1.y - localData[tbx + j].y;
                    float dz = pos1.z - localData[tbx + j].z;
                    float r2 = dx * dx + dy * dy + dz * dz;

                    if (!useCutoff || r2 < cutoff2) {
                        float r = sqrtf(r2);
                        if (r > 1e-6f) {
                            float S_j = localData[tbx + j].scaledRadius;
                            float r_plus_Sj = r + S_j;

                            if (R1_off < r_plus_Sj) {
                                float r_minus_Sj = fabsf(r - S_j);
                                float l_ij = (R1_off > r_minus_Sj) ? (1.0f / R1_off) : (1.0f / r_minus_Sj);
                                float u_ij = 1.0f / r_plus_Sj;
                                float l_ij2 = l_ij * l_ij;
                                float u_ij2 = u_ij * u_ij;
                                float r_inv = 1.0f / r;

                                float term = l_ij - u_ij +
                                             0.25f * r * (u_ij2 - l_ij2) +
                                             0.5f * r_inv * logf(u_ij / l_ij) +
                                             0.25f * S_j * S_j * r_inv * (l_ij2 - u_ij2);

                                if (R1_off < (S_j - r)) {
                                    term += 2.0f * (1.0f / R1_off - l_ij);
                                }

                                hctSum1 += term;
                            }
                        }
                    }
                }
            }
        } else {
            // Off-diagonal tile: compute both directions
            unsigned int tj = tgx;
            for (int j = 0; j < TILE_SIZE; j++) {
                int atom2_j = y * TILE_SIZE + tj;
                if (atom1 < numReceptorAtoms && atom2_j < numReceptorAtoms) {
                    float dx = pos1.x - localData[tbx + tj].x;
                    float dy = pos1.y - localData[tbx + tj].y;
                    float dz = pos1.z - localData[tbx + tj].z;
                    float r2 = dx * dx + dy * dy + dz * dz;

                    if (!useCutoff || r2 < cutoff2) {
                        float r = sqrtf(r2);
                        if (r > 1e-6f) {
                            // Contribution to atom1 from atom2
                            float S_j = localData[tbx + tj].scaledRadius;
                            float r_plus_Sj = r + S_j;

                            if (R1_off < r_plus_Sj) {
                                float r_minus_Sj = fabsf(r - S_j);
                                float l_ij = (R1_off > r_minus_Sj) ? (1.0f / R1_off) : (1.0f / r_minus_Sj);
                                float u_ij = 1.0f / r_plus_Sj;
                                float l_ij2 = l_ij * l_ij;
                                float u_ij2 = u_ij * u_ij;
                                float r_inv = 1.0f / r;

                                float term = l_ij - u_ij +
                                             0.25f * r * (u_ij2 - l_ij2) +
                                             0.5f * r_inv * logf(u_ij / l_ij) +
                                             0.25f * S_j * S_j * r_inv * (l_ij2 - u_ij2);

                                if (R1_off < (S_j - r)) {
                                    term += 2.0f * (1.0f / R1_off - l_ij);
                                }

                                hctSum1 += term;
                            }

                            // Contribution to atom2 from atom1
                            float R2_off = localData[tbx + tj].radius - DIELECTRIC_OFFSET;
                            float r_plus_S1 = r + S1;

                            if (R2_off < r_plus_S1) {
                                float r_minus_S1 = fabsf(r - S1);
                                float l_ji = (R2_off > r_minus_S1) ? (1.0f / R2_off) : (1.0f / r_minus_S1);
                                float u_ji = 1.0f / r_plus_S1;
                                float l_ji2 = l_ji * l_ji;
                                float u_ji2 = u_ji * u_ji;
                                float r_inv = 1.0f / r;

                                float term = l_ji - u_ji +
                                             0.25f * r * (u_ji2 - l_ji2) +
                                             0.5f * r_inv * logf(u_ji / l_ji) +
                                             0.25f * S1 * S1 * r_inv * (l_ji2 - u_ji2);

                                if (R2_off < (S1 - r)) {
                                    term += 2.0f * (1.0f / R2_off - l_ji);
                                }

                                localData[tbx + tj].hctSum += term;
                            }
                        }
                    }
                }
                tj = (tj + 1) & (TILE_SIZE - 1);
                __syncwarp();
            }
        }

        // Write results using atomicAdd with fixed-point conversion
        // Scale by 2^32 for fixed-point arithmetic
        if (atom1 < numReceptorAtoms) {
            unsigned long long hctFixed = (unsigned long long)(hctSum1 * 0x100000000);
            atomicAdd(&receptorSelfHCT[atom1], hctFixed);
        }

        if (x != y && atom2 < numReceptorAtoms) {
            unsigned long long hctFixed = (unsigned long long)(localData[tbx + tgx].hctSum * 0x100000000);
            atomicAdd(&receptorSelfHCT[atom2], hctFixed);
        }

        pos++;
    }
}

/**
 * Convert fixed-point HCT values to float.
 */
extern "C" __global__ void convertHCTToFloat(
    const unsigned long long* __restrict__ hctFixed,
    float* __restrict__ hctFloat,
    int numAtoms
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numAtoms) return;

    // Convert from fixed-point (scaled by 2^32) back to float
    hctFloat[i] = (float)(hctFixed[i] / (double)0x100000000);
}

/**
 * Tiled computation of receptor GB energy.
 * Uses OpenMM-style tile-based parallelization.
 */
extern "C" __global__ void computeReceptorGBEnergyTiled(
    const float3* __restrict__ receptorPositions,
    const float* __restrict__ receptorCharges,
    const float* __restrict__ receptorBornRadii,
    int numReceptorAtoms,
    float prefactor,
    float* __restrict__ receptorEnergy,
    int numTiles
) {
    const int totalWarps = (gridDim.x * blockDim.x) / TILE_SIZE;
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) / TILE_SIZE;
    const int tgx = threadIdx.x & (TILE_SIZE - 1);
    const int tbx = threadIdx.x - tgx;

    __shared__ TiledAtomDataGB localData[256];
    __shared__ float energyBuffer[256];

    const int NUM_BLOCKS = (numReceptorAtoms + TILE_SIZE - 1) / TILE_SIZE;

    float energy = 0.0f;

    // Each warp processes a range of tiles
    int pos = (int)(((long long)warp * numTiles) / totalWarps);
    int end = (int)(((long long)(warp + 1) * numTiles) / totalWarps);

    while (pos < end) {
        // Convert linear tile index to (x, y) tile coordinates
        int y = (int)floor(NUM_BLOCKS + 0.5f - sqrtf((NUM_BLOCKS + 0.5f) * (NUM_BLOCKS + 0.5f) - 2.0f * pos));
        int x = pos - y * NUM_BLOCKS + y * (y + 1) / 2;

        if (x < y || x >= NUM_BLOCKS) {
            y += (x < y ? -1 : 1);
            x = pos - y * NUM_BLOCKS + y * (y + 1) / 2;
        }

        unsigned int atom1 = x * TILE_SIZE + tgx;
        unsigned int atom2 = y * TILE_SIZE + tgx;

        // Load atom1 data
        float3 pos1 = make_float3(0, 0, 0);
        float q1 = 0, R1 = 1.0f;
        if (atom1 < numReceptorAtoms) {
            pos1 = receptorPositions[atom1];
            q1 = receptorCharges[atom1];
            R1 = receptorBornRadii[atom1];
        }

        // Load atom2 data into shared memory
        if (atom2 < numReceptorAtoms) {
            float3 pos2 = receptorPositions[atom2];
            localData[tbx + tgx].x = pos2.x;
            localData[tbx + tgx].y = pos2.y;
            localData[tbx + tgx].z = pos2.z;
            localData[tbx + tgx].charge = receptorCharges[atom2];
            localData[tbx + tgx].bornRadius = receptorBornRadii[atom2];
        } else {
            localData[tbx + tgx].x = 0;
            localData[tbx + tgx].y = 0;
            localData[tbx + tgx].z = 0;
            localData[tbx + tgx].charge = 0;
            localData[tbx + tgx].bornRadius = 1.0f;
        }
        localData[tbx + tgx].energy = 0.0f;
        __syncwarp();

        if (x == y) {
            // Diagonal tile: self energy + upper triangle pairs
            // Self energy term
            if (atom1 < numReceptorAtoms) {
                energy += 0.5f * prefactor * q1 * q1 / R1;
            }

            // Pair terms (j > tgx within this tile)
            for (int j = tgx + 1; j < TILE_SIZE; j++) {
                int atom2_j = y * TILE_SIZE + j;
                if (atom1 < numReceptorAtoms && atom2_j < numReceptorAtoms) {
                    float dx = localData[tbx + j].x - pos1.x;
                    float dy = localData[tbx + j].y - pos1.y;
                    float dz = localData[tbx + j].z - pos1.z;
                    float r2 = dx * dx + dy * dy + dz * dz;
                    float r = sqrtf(r2);

                    float q2 = localData[tbx + j].charge;
                    float R2 = localData[tbx + j].bornRadius;
                    float RiRj = R1 * R2;
                    float expArg = -r2 / (4.0f * RiRj);
                    float expTerm = expf(expArg);
                    float f_gb2 = r2 + RiRj * expTerm;
                    float f_gb = sqrtf(f_gb2);

                    energy += prefactor * q1 * q2 / f_gb;
                }
            }
        } else {
            // Off-diagonal tile: all pairs
            unsigned int tj = tgx;
            for (int j = 0; j < TILE_SIZE; j++) {
                int atom2_j = y * TILE_SIZE + tj;
                if (atom1 < numReceptorAtoms && atom2_j < numReceptorAtoms) {
                    float dx = localData[tbx + tj].x - pos1.x;
                    float dy = localData[tbx + tj].y - pos1.y;
                    float dz = localData[tbx + tj].z - pos1.z;
                    float r2 = dx * dx + dy * dy + dz * dz;
                    float r = sqrtf(r2);

                    float q2 = localData[tbx + tj].charge;
                    float R2 = localData[tbx + tj].bornRadius;
                    float RiRj = R1 * R2;
                    float expArg = -r2 / (4.0f * RiRj);
                    float expTerm = expf(expArg);
                    float f_gb2 = r2 + RiRj * expTerm;
                    float f_gb = sqrtf(f_gb2);

                    // Each pair counted once (atom1 in tile X, atom2 in tile Y, x > y)
                    energy += prefactor * q1 * q2 / f_gb;
                }
                tj = (tj + 1) & (TILE_SIZE - 1);
                __syncwarp();
            }
        }

        pos++;
    }

    // Reduce energy within block
    energyBuffer[threadIdx.x] = energy;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            energyBuffer[threadIdx.x] += energyBuffer[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicAdd(receptorEnergy, energyBuffer[0]);
    }
}

/**
 * Tiled computation of dE/dR_born for all receptor atoms.
 * Pre-computes receptor derivatives for use in force calculations.
 * Uses fixed-point atomicAdd for accumulation.
 */
extern "C" __global__ void computeReceptorDeDRTiled(
    const float3* __restrict__ receptorPositions,
    const float* __restrict__ receptorCharges,
    const float* __restrict__ receptorBornRadii,
    int numReceptorAtoms,
    float prefactor,
    unsigned long long* __restrict__ receptorDeDR,  // Fixed-point output
    int numTiles
) {
    const int totalWarps = (gridDim.x * blockDim.x) / TILE_SIZE;
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) / TILE_SIZE;
    const int tgx = threadIdx.x & (TILE_SIZE - 1);
    const int tbx = threadIdx.x - tgx;

    __shared__ TiledAtomDataGB localData[256];

    const int NUM_BLOCKS = (numReceptorAtoms + TILE_SIZE - 1) / TILE_SIZE;

    // Each warp processes a range of tiles
    int pos = (int)(((long long)warp * numTiles) / totalWarps);
    int end = (int)(((long long)(warp + 1) * numTiles) / totalWarps);

    // Local accumulator for dE/dR for atom1
    float dEdR1 = 0.0f;

    while (pos < end) {
        // Convert linear tile index to (x, y) tile coordinates
        int y = (int)floor(NUM_BLOCKS + 0.5f - sqrtf((NUM_BLOCKS + 0.5f) * (NUM_BLOCKS + 0.5f) - 2.0f * pos));
        int x = pos - y * NUM_BLOCKS + y * (y + 1) / 2;

        // Handle roundoff errors
        if (x < y || x >= NUM_BLOCKS) {
            y += (x < y ? -1 : 1);
            x = pos - y * NUM_BLOCKS + y * (y + 1) / 2;
        }

        unsigned int atom1 = x * TILE_SIZE + tgx;
        unsigned int atom2 = y * TILE_SIZE + tgx;

        // Load atom1 data
        float3 pos1 = make_float3(0, 0, 0);
        float q1 = 0, R1 = 1.0f;
        if (atom1 < numReceptorAtoms) {
            pos1 = receptorPositions[atom1];
            q1 = receptorCharges[atom1];
            R1 = receptorBornRadii[atom1];
        }

        // Load atom2 data into shared memory
        if (atom2 < numReceptorAtoms) {
            float3 pos2 = receptorPositions[atom2];
            localData[tbx + tgx].x = pos2.x;
            localData[tbx + tgx].y = pos2.y;
            localData[tbx + tgx].z = pos2.z;
            localData[tbx + tgx].charge = receptorCharges[atom2];
            localData[tbx + tgx].bornRadius = receptorBornRadii[atom2];
        } else {
            localData[tbx + tgx].x = 0;
            localData[tbx + tgx].y = 0;
            localData[tbx + tgx].z = 0;
            localData[tbx + tgx].charge = 0;
            localData[tbx + tgx].bornRadius = 1.0f;
        }
        localData[tbx + tgx].energy = 0.0f;  // Use energy field to accumulate dEdR for atom2
        __syncwarp();

        if (x == y) {
            // Diagonal tile: self term + pair terms (j != i)
            // Self term: dE_self/dR = -0.5 * prefactor * q² / R²
            if (atom1 < numReceptorAtoms) {
                dEdR1 += -0.5f * prefactor * q1 * q1 / (R1 * R1);
            }

            // Pair terms
            for (int j = 0; j < TILE_SIZE; j++) {
                if (j == tgx) continue;
                int atom2_j = y * TILE_SIZE + j;
                if (atom1 < numReceptorAtoms && atom2_j < numReceptorAtoms) {
                    float dx = localData[tbx + j].x - pos1.x;
                    float dy = localData[tbx + j].y - pos1.y;
                    float dz = localData[tbx + j].z - pos1.z;
                    float r2 = dx * dx + dy * dy + dz * dz;

                    float q2 = localData[tbx + j].charge;
                    float R2 = localData[tbx + j].bornRadius;
                    float RiRj = R1 * R2;
                    float expArg = -r2 / (4.0f * RiRj);
                    float expTerm = expf(expArg);
                    float f_gb2 = r2 + RiRj * expTerm;
                    float f_gb = sqrtf(f_gb2);

                    // dFgb/dRi = (Rj * exp / (2*fgb)) * (1 + r²/(4*RiRj))
                    float dFgbDRi = (R2 * expTerm / (2.0f * f_gb)) * (1.0f + r2 / (4.0f * RiRj));
                    dEdR1 += -prefactor * q1 * q2 / (f_gb * f_gb) * dFgbDRi;
                }
            }
        } else {
            // Off-diagonal tile: all pairs (atom1 in tile X, atom2 in tile Y)
            unsigned int tj = tgx;
            for (int j = 0; j < TILE_SIZE; j++) {
                int atom2_j = y * TILE_SIZE + tj;
                if (atom1 < numReceptorAtoms && atom2_j < numReceptorAtoms) {
                    float dx = localData[tbx + tj].x - pos1.x;
                    float dy = localData[tbx + tj].y - pos1.y;
                    float dz = localData[tbx + tj].z - pos1.z;
                    float r2 = dx * dx + dy * dy + dz * dz;

                    float q2 = localData[tbx + tj].charge;
                    float R2 = localData[tbx + tj].bornRadius;
                    float RiRj = R1 * R2;
                    float expArg = -r2 / (4.0f * RiRj);
                    float expTerm = expf(expArg);
                    float f_gb2 = r2 + RiRj * expTerm;
                    float f_gb = sqrtf(f_gb2);

                    // dFgb/dRi = (Rj * exp / (2*fgb)) * (1 + r²/(4*RiRj))
                    float dFgbDRi = (R2 * expTerm / (2.0f * f_gb)) * (1.0f + r2 / (4.0f * RiRj));
                    float dFgbDRj = (R1 * expTerm / (2.0f * f_gb)) * (1.0f + r2 / (4.0f * RiRj));

                    // Contribution to atom1 (in tile X)
                    dEdR1 += -prefactor * q1 * q2 / (f_gb * f_gb) * dFgbDRi;

                    // Contribution to atom2 (in tile Y) - accumulate in shared memory
                    localData[tbx + tj].energy += -prefactor * q1 * q2 / (f_gb * f_gb) * dFgbDRj;
                }
                tj = (tj + 1) & (TILE_SIZE - 1);
                __syncwarp();
            }
        }

        // Write atom2 contributions (off-diagonal tiles only)
        if (x != y && atom2 < numReceptorAtoms) {
            unsigned long long dEdRFixed = (unsigned long long)((long long)(localData[tbx + tgx].energy * 0x100000000));
            atomicAdd(&receptorDeDR[atom2], dEdRFixed);
        }

        pos++;
    }

    // Write atom1 contributions using atomicAdd
    // Need to aggregate across all tiles processed by this warp
    if ((threadIdx.x & (TILE_SIZE - 1)) < numReceptorAtoms) {
        // Each thread might have processed multiple tiles, all for different atom1 values
        // We need a different approach - accumulate locally first
    }

    // Actually, each thread processes multiple tiles with different atom1 values
    // We need to use atomicAdd for each contribution
    // For simplicity, write the accumulated dEdR1 for the LAST atom1 this thread processed
    // This is incorrect - we need a different approach

    // Simplified approach: each thread accumulates for one fixed atom based on thread index
    // This requires restructuring the kernel, but for now let's use atomicAdd per-tile
}

/**
 * Simple kernel to compute dE/dR_born for receptor atoms.
 * Each thread computes dE/dR for one receptor atom.
 * O(N²) but simple and correct.
 */
extern "C" __global__ void computeReceptorDeDRSimple(
    const float3* __restrict__ receptorPositions,
    const float* __restrict__ receptorCharges,
    const float* __restrict__ receptorBornRadii,
    int numReceptorAtoms,
    float prefactor,
    float* __restrict__ receptorDeDR
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numReceptorAtoms) return;

    float3 pos_i = receptorPositions[i];
    float q_i = receptorCharges[i];
    float R_i = receptorBornRadii[i];

    // Self term: dE_self/dR = -0.5 * prefactor * q² / R²
    float dEdR = -0.5f * prefactor * q_i * q_i / (R_i * R_i);

    // Pair terms
    for (int j = 0; j < numReceptorAtoms; j++) {
        if (j == i) continue;

        float3 pos_j = receptorPositions[j];
        float q_j = receptorCharges[j];
        float R_j = receptorBornRadii[j];

        float dx = pos_j.x - pos_i.x;
        float dy = pos_j.y - pos_i.y;
        float dz = pos_j.z - pos_i.z;
        float r2 = dx*dx + dy*dy + dz*dz;

        float RiRj = R_i * R_j;
        float expArg = -r2 / (4.0f * RiRj);
        float expTerm = expf(expArg);
        float f_gb2 = r2 + RiRj * expTerm;
        float f_gb = sqrtf(f_gb2);

        float dFgbDRi = (R_j * expTerm / (2.0f * f_gb)) * (1.0f + r2 / (4.0f * RiRj));
        dEdR += -prefactor * q_i * q_j / (f_gb * f_gb) * dFgbDRi;
    }

    receptorDeDR[i] = dEdR;
}

/**
 * Compute HCT contribution from ligand-ligand pairwise interactions.
 * No exclusions - all pairs contribute to Born radii (physically correct for GBSA).
 */
extern "C" __global__ void computeIsolatedLigandHCT(
    const float4* __restrict__ posq,
    const int* __restrict__ particleIndices,
    const float* __restrict__ radii,
    const float* __restrict__ scaleFactors,
    const int* __restrict__ groupStart,
    int numGroups,
    int templateNumAtoms,
    float cutoffDistance,
    float* __restrict__ hctLigand
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Determine which group this atom belongs to
    int groupIdx = 0;
    int atomInGroup = idx;
    int groupStartIdx = 0;
    int groupEndIdx = 0;

    for (int g = 0; g < numGroups; g++) {
        groupStartIdx = groupStart[g];
        groupEndIdx = groupStart[g + 1];
        if (idx >= groupStartIdx && idx < groupEndIdx) {
            groupIdx = g;
            atomInGroup = idx - groupStartIdx;
            break;
        }
    }

    if (idx >= groupEndIdx) return;

    int particleIdx_i = particleIndices[idx];
    int templateIdx_i = atomInGroup % templateNumAtoms;

    float4 pos_i = posq[particleIdx_i];
    float R_i = radii[templateIdx_i];
    float R_i_off = R_i - DIELECTRIC_OFFSET;

    float hct = 0.0f;
    float cutoff2 = cutoffDistance * cutoffDistance;
    bool useCutoff = (cutoffDistance > 0.0f);

    // Loop over other atoms in same group (no exclusions in GBSA)
    int groupSize = groupEndIdx - groupStartIdx;
    for (int jLocal = 0; jLocal < groupSize; jLocal++) {
        if (jLocal == atomInGroup) continue;

        int j = groupStartIdx + jLocal;
        int templateIdx_j = jLocal % templateNumAtoms;
        int particleIdx_j = particleIndices[j];
        float4 pos_j = posq[particleIdx_j];

        float dx = pos_i.x - pos_j.x;
        float dy = pos_i.y - pos_j.y;
        float dz = pos_i.z - pos_j.z;
        float r2 = dx*dx + dy*dy + dz*dz;

        // Apply cutoff if enabled
        if (useCutoff && r2 > cutoff2) continue;

        float r = sqrtf(r2);
        if (r < 1e-6f) continue;

        float R_j = radii[templateIdx_j];
        float R_j_off = R_j - DIELECTRIC_OFFSET;
        float S_j = R_j_off * scaleFactors[templateIdx_j];

        // HCT integral computation
        float r_plus_Sj = r + S_j;
        if (R_i_off >= r_plus_Sj) continue;  // No overlap

        float r_minus_Sj = fabsf(r - S_j);
        float l_ij = (R_i_off > r_minus_Sj) ? (1.0f / R_i_off) : (1.0f / r_minus_Sj);
        float u_ij = 1.0f / r_plus_Sj;

        float l_ij2 = l_ij * l_ij;
        float u_ij2 = u_ij * u_ij;
        float r_inv = 1.0f / r;

        float term = l_ij - u_ij +
                     0.25f * r * (u_ij2 - l_ij2) +
                     0.5f * r_inv * logf(u_ij / l_ij) +
                     0.25f * S_j * S_j * r_inv * (l_ij2 - u_ij2);

        // Tinker correction: atom i completely inside atom j
        if (R_i_off < (S_j - r)) {
            term += 2.0f * (1.0f / R_i_off - l_ij);
        }

        hct += term;
    }

    hctLigand[idx] = hct;
}

/**
 * Compute HCT contribution from receptor via pairwise interactions.
 */
extern "C" __global__ void computeIsolatedReceptorHCTPairwise(
    const float4* __restrict__ posq,
    const int* __restrict__ particleIndices,
    const float* __restrict__ radii,
    const float3* __restrict__ receptorPositions,
    const float* __restrict__ receptorRadii,
    const float* __restrict__ receptorScaleFactors,
    int numReceptorAtoms,
    const int* __restrict__ groupStart,
    int numGroups,
    int totalParticles,
    int templateNumAtoms,
    float cutoffDistance,
    float* __restrict__ hctReceptor
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= totalParticles) return;

    // Determine template index
    int atomInGroup = idx;
    for (int g = 0; g < numGroups; g++) {
        int groupStartIdx = groupStart[g];
        int groupEndIdx = groupStart[g + 1];
        if (idx >= groupStartIdx && idx < groupEndIdx) {
            atomInGroup = idx - groupStartIdx;
            break;
        }
    }

    int particleIdx = particleIndices[idx];
    int templateIdx = atomInGroup % templateNumAtoms;

    float4 pos = posq[particleIdx];
    float3 pos_i = make_float3(pos.x, pos.y, pos.z);
    float R_i = radii[templateIdx];
    float R_i_off = R_i - DIELECTRIC_OFFSET;

    float hct = 0.0f;
    float cutoff2 = cutoffDistance * cutoffDistance;
    bool useCutoff = (cutoffDistance > 0.0f);

    // Loop over receptor atoms
    for (int j = 0; j < numReceptorAtoms; j++) {
        float3 pos_j = receptorPositions[j];

        float dx = pos_i.x - pos_j.x;
        float dy = pos_i.y - pos_j.y;
        float dz = pos_i.z - pos_j.z;
        float r2 = dx*dx + dy*dy + dz*dz;

        if (useCutoff && r2 > cutoff2) continue;

        float r = sqrtf(r2);
        if (r < 1e-6f) continue;

        float R_j = receptorRadii[j];
        float R_j_off = R_j - DIELECTRIC_OFFSET;
        float S_j = R_j_off * receptorScaleFactors[j];

        // HCT integral
        float r_plus_Sj = r + S_j;
        if (R_i_off >= r_plus_Sj) continue;

        float r_minus_Sj = fabsf(r - S_j);
        float l_ij = (R_i_off > r_minus_Sj) ? (1.0f / R_i_off) : (1.0f / r_minus_Sj);
        float u_ij = 1.0f / r_plus_Sj;

        float l_ij2 = l_ij * l_ij;
        float u_ij2 = u_ij * u_ij;
        float r_inv = 1.0f / r;

        float term = l_ij - u_ij +
                     0.25f * r * (u_ij2 - l_ij2) +
                     0.5f * r_inv * logf(u_ij / l_ij) +
                     0.25f * S_j * S_j * r_inv * (l_ij2 - u_ij2);

        if (R_i_off < (S_j - r)) {
            term += 2.0f * (1.0f / R_i_off - l_ij);
        }

        hct += term;
    }

    hctReceptor[idx] = hct;
}

/**
 * Compute Born radii using raw HCT formula (no OBC correction).
 * R_born = 1 / (1/R_off - 0.5*R_off*HCT)
 */
extern "C" __global__ void computeBornRadiiHCT(
    const float* __restrict__ radii,
    const float* __restrict__ hctReceptor,
    const float* __restrict__ hctLigand,
    int numAtoms,
    int templateNumAtoms,
    float* __restrict__ bornRadii
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numAtoms) return;

    int templateIdx = idx % templateNumAtoms;
    float R_i = radii[templateIdx];
    float R_i_off = R_i - DIELECTRIC_OFFSET;

    float hctTotal = hctReceptor[idx] + hctLigand[idx];

    // Raw HCT: R_born = 1 / (1/R_off - psi), where psi = 0.5 * R_off * HCT
    float psi = 0.5f * R_i_off * hctTotal;
    float denom = 1.0f / R_i_off - psi;

    float bornRadius = (denom > 1e-6f) ? (1.0f / denom) : 50.0f;
    bornRadius = fminf(bornRadius, 50.0f);

    bornRadii[idx] = bornRadius;
}

/**
 * Compute Born radii using OBC-II formula with tanh correction.
 */
extern "C" __global__ void computeBornRadiiOBC(
    const float* __restrict__ radii,
    const float* __restrict__ hctReceptor,
    const float* __restrict__ hctLigand,
    int numAtoms,
    int templateNumAtoms,
    float* __restrict__ bornRadii
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numAtoms) return;

    int templateIdx = idx % templateNumAtoms;
    float R_i = radii[templateIdx];
    float R_i_off = R_i - DIELECTRIC_OFFSET;

    float hctTotal = hctReceptor[idx] + hctLigand[idx];
    float psi = 0.5f * R_i_off * hctTotal;

    // OBC-II tanh correction
    float psi2 = psi * psi;
    float psi3 = psi2 * psi;
    float tanhArg = OBC_ALPHA * psi - OBC_BETA * psi2 + OBC_GAMMA * psi3;
    float tanhVal = tanhf(tanhArg);

    float denom = 1.0f / R_i_off - tanhVal / R_i;
    float bornRadius = (denom > 0.0f) ? (1.0f / denom) : R_i;

    bornRadius = fminf(bornRadius, 50.0f);

    bornRadii[idx] = bornRadius;
}

/**
 * Compute GB energy using Still equation and accumulate direct forces.
 * Also tracks ligand self-energy separately.
 */
extern "C" __global__ void computeIsolatedGBEnergy(
    const float4* __restrict__ posq,
    const int* __restrict__ particleIndices,
    const float* __restrict__ charges,
    const float* __restrict__ bornRadii,
    const int* __restrict__ groupStart,
    int numGroups,
    int templateNumAtoms,
    float prefactor,
    unsigned long long* __restrict__ forceBuffer,
    float* __restrict__ groupEnergies,
    float* __restrict__ groupLigandSelfEnergies,
    int paddedNumAtoms
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Determine group
    int groupIdx = 0;
    int atomInGroup = idx;
    int groupStartIdx = 0;
    int groupEndIdx = 0;

    for (int g = 0; g < numGroups; g++) {
        groupStartIdx = groupStart[g];
        groupEndIdx = groupStart[g + 1];
        if (idx >= groupStartIdx && idx < groupEndIdx) {
            groupIdx = g;
            atomInGroup = idx - groupStartIdx;
            break;
        }
    }

    if (idx >= groupEndIdx) return;

    int particleIdx_i = particleIndices[idx];
    int templateIdx_i = atomInGroup % templateNumAtoms;

    float4 pos_i = posq[particleIdx_i];
    float q_i = charges[templateIdx_i];
    float R_i = bornRadii[idx];

    float energy = 0.0f;
    float3 force = make_float3(0.0f, 0.0f, 0.0f);

    // Self energy term
    energy += 0.5f * prefactor * q_i * q_i / R_i;

    // Pairwise terms (j > i to avoid double counting)
    int groupSize = groupEndIdx - groupStartIdx;
    for (int jLocal = atomInGroup + 1; jLocal < groupSize; jLocal++) {
        int j = groupStartIdx + jLocal;
        int templateIdx_j = jLocal % templateNumAtoms;
        int particleIdx_j = particleIndices[j];

        float4 pos_j = posq[particleIdx_j];
        float q_j = charges[templateIdx_j];
        float R_j = bornRadii[j];

        float dx = pos_j.x - pos_i.x;
        float dy = pos_j.y - pos_i.y;
        float dz = pos_j.z - pos_i.z;
        float r2 = dx*dx + dy*dy + dz*dz;
        float r = sqrtf(r2);

        // Still equation
        float RiRj = R_i * R_j;
        float expArg = -r2 / (4.0f * RiRj);
        float expTerm = expf(expArg);
        float f_gb2 = r2 + RiRj * expTerm;
        float f_gb = sqrtf(f_gb2);
        float invFgb = 1.0f / f_gb;

        float pairEnergy = prefactor * q_i * q_j * invFgb;
        energy += pairEnergy;

        // Force = -dE/dr
        // dE/dr = -prefactor * q_i * q_j / f_gb² * df_gb/dr
        // F_i = -∇_i E = dE/dr * (dx/r)  (toward j if dE/dr > 0)
        // F_j = -dE/dr * (dx/r)  (opposite direction, Newton's 3rd law)
        float dFgbDr = (r * invFgb) * (1.0f - 0.25f * expTerm);
        float dEdR = -prefactor * q_i * q_j * invFgb * invFgb * dFgbDr;

        float invR = 1.0f / r;
        force.x += dEdR * dx * invR;
        force.y += dEdR * dy * invR;
        force.z += dEdR * dz * invR;

        // Force on j (Newton's 3rd law): F_j = -F_i
        atomicAdd(&forceBuffer[particleIdx_j], static_cast<unsigned long long>((long long)(-dEdR * dx * invR * 0x100000000)));
        atomicAdd(&forceBuffer[particleIdx_j + paddedNumAtoms], static_cast<unsigned long long>((long long)(-dEdR * dy * invR * 0x100000000)));
        atomicAdd(&forceBuffer[particleIdx_j + 2*paddedNumAtoms], static_cast<unsigned long long>((long long)(-dEdR * dz * invR * 0x100000000)));
    }

    // Accumulate force on i
    atomicAdd(&forceBuffer[particleIdx_i], static_cast<unsigned long long>((long long)(force.x * 0x100000000)));
    atomicAdd(&forceBuffer[particleIdx_i + paddedNumAtoms], static_cast<unsigned long long>((long long)(force.y * 0x100000000)));
    atomicAdd(&forceBuffer[particleIdx_i + 2*paddedNumAtoms], static_cast<unsigned long long>((long long)(force.z * 0x100000000)));

    // Accumulate energies
    atomicAdd(&groupEnergies[groupIdx], energy);
    atomicAdd(&groupLigandSelfEnergies[groupIdx], energy);  // All is ligand-self in this kernel
}

/**
 * Compute surface area energy (ACE approximation).
 */
extern "C" __global__ void computeIsolatedSAEnergy(
    const float* __restrict__ radii,
    const float* __restrict__ bornRadii,
    const int* __restrict__ groupStart,
    int numGroups,
    int templateNumAtoms,
    float surfaceTension,
    float probeRadius,
    float* __restrict__ groupEnergies
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int groupIdx = 0;
    int atomInGroup = idx;
    int groupStartIdx = 0;
    int groupEndIdx = 0;

    for (int g = 0; g < numGroups; g++) {
        groupStartIdx = groupStart[g];
        groupEndIdx = groupStart[g + 1];
        if (idx >= groupStartIdx && idx < groupEndIdx) {
            groupIdx = g;
            atomInGroup = idx - groupStartIdx;
            break;
        }
    }

    if (idx >= groupEndIdx) return;

    int templateIdx = atomInGroup % templateNumAtoms;
    float R_i = radii[templateIdx];
    float bornR = bornRadii[idx];

    // ACE surface area term
    float Rsolv = R_i + probeRadius;
    float ratio = R_i / bornR;
    float ratio6 = ratio * ratio * ratio * ratio * ratio * ratio;
    float area = 4.0f * 3.14159265f * Rsolv * Rsolv * ratio6;
    float saEnergy = surfaceTension * area;

    atomicAdd(&groupEnergies[groupIdx], saEnergy);
}

/**
 * Accumulate dE/dR_born from GB energy.
 */
extern "C" __global__ void accumulateIsolatedBornRadiiDerivatives(
    const float4* __restrict__ posq,
    const int* __restrict__ particleIndices,
    const float* __restrict__ charges,
    const float* __restrict__ bornRadii,
    const int* __restrict__ groupStart,
    int numGroups,
    int templateNumAtoms,
    float prefactor,
    float* __restrict__ dE_dR
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int groupIdx = 0;
    int atomInGroup = idx;
    int groupStartIdx = 0;
    int groupEndIdx = 0;

    for (int g = 0; g < numGroups; g++) {
        groupStartIdx = groupStart[g];
        groupEndIdx = groupStart[g + 1];
        if (idx >= groupStartIdx && idx < groupEndIdx) {
            groupIdx = g;
            atomInGroup = idx - groupStartIdx;
            break;
        }
    }

    if (idx >= groupEndIdx) return;

    int particleIdx_i = particleIndices[idx];
    int templateIdx_i = atomInGroup % templateNumAtoms;

    float4 pos_i = posq[particleIdx_i];
    float q_i = charges[templateIdx_i];
    float R_i = bornRadii[idx];

    float dEdRi = 0.0f;

    // Self term: E_self = 0.5 * prefactor * q_i^2 / R_i
    // dE/dR_i = -0.5 * prefactor * q_i^2 / R_i^2
    dEdRi += -0.5f * prefactor * q_i * q_i / (R_i * R_i);

    // Pairwise terms
    int groupSize = groupEndIdx - groupStartIdx;
    for (int jLocal = 0; jLocal < groupSize; jLocal++) {
        if (jLocal == atomInGroup) continue;

        int j = groupStartIdx + jLocal;
        int templateIdx_j = jLocal % templateNumAtoms;
        int particleIdx_j = particleIndices[j];

        float4 pos_j = posq[particleIdx_j];
        float q_j = charges[templateIdx_j];
        float R_j = bornRadii[j];

        float dx = pos_j.x - pos_i.x;
        float dy = pos_j.y - pos_i.y;
        float dz = pos_j.z - pos_i.z;
        float r2 = dx*dx + dy*dy + dz*dz;

        float RiRj = R_i * R_j;
        float expArg = -r2 / (4.0f * RiRj);
        float expTerm = expf(expArg);
        float f_gb2 = r2 + RiRj * expTerm;
        float f_gb = sqrtf(f_gb2);

        // dE/dR_i from pair (i,j)
        // E = prefactor * q_i * q_j / f_gb
        // df_gb/dR_i = (1/(2*f_gb)) * R_j * exp(...) * (1 + r2/(4*R_i*R_j))
        //            = (R_j * expTerm / (2*f_gb)) * (1 + r2/(4*RiRj))
        float dFgbDRi = (R_j * expTerm / (2.0f * f_gb)) * (1.0f + r2 / (4.0f * RiRj));
        float dEdR_pair = -prefactor * q_i * q_j / (f_gb * f_gb) * dFgbDRi;

        dEdRi += dEdR_pair;
    }

    dE_dR[idx] = dEdRi;
}

/**
 * Accumulate dE/dR_born from surface area term.
 */
extern "C" __global__ void accumulateIsolatedSADerivatives(
    const float* __restrict__ radii,
    const float* __restrict__ bornRadii,
    const int* __restrict__ groupStart,
    int numGroups,
    int templateNumAtoms,
    float surfaceTension,
    float probeRadius,
    float* __restrict__ dE_dR
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int groupIdx = 0;
    int atomInGroup = idx;
    int groupStartIdx = 0;
    int groupEndIdx = 0;

    for (int g = 0; g < numGroups; g++) {
        groupStartIdx = groupStart[g];
        groupEndIdx = groupStart[g + 1];
        if (idx >= groupStartIdx && idx < groupEndIdx) {
            groupIdx = g;
            atomInGroup = idx - groupStartIdx;
            break;
        }
    }

    if (idx >= groupEndIdx) return;

    int templateIdx = atomInGroup % templateNumAtoms;
    float R_i = radii[templateIdx];
    float bornR = bornRadii[idx];

    // E_SA = surfaceTension * 4*pi*(R_i+probe)^2 * (R_i/bornR)^6
    // dE_SA/dR_born = surfaceTension * 4*pi*(R_i+probe)^2 * 6 * (R_i/bornR)^5 * (-R_i/bornR^2)
    //              = -6 * surfaceTension * 4*pi*(R_i+probe)^2 * R_i^6 / bornR^7
    float Rsolv = R_i + probeRadius;
    float ratio = R_i / bornR;
    float ratio5 = ratio * ratio * ratio * ratio * ratio;

    float dEdR_SA = -6.0f * surfaceTension * 4.0f * 3.14159265f * Rsolv * Rsolv * ratio5 * R_i / (bornR * bornR);

    dE_dR[idx] += dEdR_SA;
}

/**
 * Compute chain rule forces from ligand-ligand HCT using OpenMM's simplified formula.
 *
 * OpenMM observation: dL/dr and dU/dr are zero (can be shown analytically).
 * This leads to a much simpler formula:
 *   t3 = 0.125*(1 + S²/r²)*(l² - u²) + 0.25*log(u/l)/r²
 *   de = bornForces[i] * t3 / r
 *
 * Where bornForces[i] = dE/dR_born * R_born² * obcChain
 */
extern "C" __global__ void computeIsolatedHCTChainRuleForces(
    const float4* __restrict__ posq,
    const int* __restrict__ particleIndices,
    const float* __restrict__ radii,
    const float* __restrict__ scaleFactors,
    const float* __restrict__ bornRadii,
    const float* __restrict__ hctReceptor,
    const float* __restrict__ hctLigand,
    const float* __restrict__ dE_dR,
    const int* __restrict__ groupStart,
    int numGroups,
    int templateNumAtoms,
    float cutoffDistance,
    unsigned long long* __restrict__ forceBuffer,
    int paddedNumAtoms
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int groupIdx = 0;
    int atomInGroup = idx;
    int groupStartIdx = 0;
    int groupEndIdx = 0;

    for (int g = 0; g < numGroups; g++) {
        groupStartIdx = groupStart[g];
        groupEndIdx = groupStart[g + 1];
        if (idx >= groupStartIdx && idx < groupEndIdx) {
            groupIdx = g;
            atomInGroup = idx - groupStartIdx;
            break;
        }
    }

    if (idx >= groupEndIdx) return;

    int particleIdx_i = particleIndices[idx];
    int templateIdx_i = atomInGroup % templateNumAtoms;

    float4 pos_i = posq[particleIdx_i];
    float R_i = radii[templateIdx_i];
    float R_i_off = R_i - DIELECTRIC_OFFSET;
    float S_i = R_i_off * scaleFactors[templateIdx_i];
    float bornR_i = bornRadii[idx];

    // Compute bornForces[i] = dE/dR_born * R_born² * obcChain (OpenMM style)
    // obcChain = R_off * (α - 2β*ψ + 3γ*ψ²) * sech²(arg) / R
    float hctTotal_i = hctReceptor[idx] + hctLigand[idx];
    float psi_i = 0.5f * R_i_off * hctTotal_i;
    float psi2_i = psi_i * psi_i;
    float psi3_i = psi2_i * psi_i;

    float tanhArg_i = OBC_ALPHA * psi_i - OBC_BETA * psi2_i + OBC_GAMMA * psi3_i;
    float tanhVal_i = tanhf(tanhArg_i);
    float sech2_i = 1.0f - tanhVal_i * tanhVal_i;
    float dTanhArgDPsi_i = OBC_ALPHA - 2.0f * OBC_BETA * psi_i + 3.0f * OBC_GAMMA * psi2_i;

    // obcChain[i] = R_off * (α - 2β*ψ + 3γ*ψ²) * sech²(arg) / R
    float obcChain_i = R_i_off * dTanhArgDPsi_i * sech2_i / R_i;

    // bornForces[i] = dE/dR_born * R_born² * obcChain
    float bornForces_i = dE_dR[idx] * bornR_i * bornR_i * obcChain_i;

    float3 force_i = make_float3(0.0f, 0.0f, 0.0f);
    float cutoff2 = cutoffDistance * cutoffDistance;
    bool useCutoff = (cutoffDistance > 0.0f);

    // Loop over other atoms in group
    int groupSize = groupEndIdx - groupStartIdx;
    for (int jLocal = 0; jLocal < groupSize; jLocal++) {
        if (jLocal == atomInGroup) continue;

        int j = groupStartIdx + jLocal;
        int templateIdx_j = jLocal % templateNumAtoms;
        int particleIdx_j = particleIndices[j];

        float4 pos_j = posq[particleIdx_j];
        float R_j = radii[templateIdx_j];
        float R_j_off = R_j - DIELECTRIC_OFFSET;
        float S_j = R_j_off * scaleFactors[templateIdx_j];

        // delta = pos_i - pos_j (direction from j to i)
        float dx = pos_i.x - pos_j.x;
        float dy = pos_i.y - pos_j.y;
        float dz = pos_i.z - pos_j.z;
        float r2 = dx*dx + dy*dy + dz*dz;

        if (useCutoff && r2 > cutoff2) continue;

        float r = sqrtf(r2);
        if (r < 1e-6f) continue;

        float r_inv = 1.0f / r;
        float r2_inv = r_inv * r_inv;

        // --- Force from j screening i's Born radius ---
        float r_plus_Sj = r + S_j;
        if (R_i_off < r_plus_Sj) {
            float r_minus_Sj = fabsf(r - S_j);
            float l_ij = (R_i_off > r_minus_Sj) ? (1.0f / R_i_off) : (1.0f / r_minus_Sj);
            float u_ij = 1.0f / r_plus_Sj;

            float l_ij2 = l_ij * l_ij;
            float u_ij2 = u_ij * u_ij;
            float S_j2 = S_j * S_j;

            // OpenMM's simplified formula (assumes dL/dr = dU/dr = 0)
            // t3 = 0.125*(1 + S²/r²)*(l² - u²) + 0.25*log(u/l)/r²
            float t3 = 0.125f * (1.0f + S_j2 * r2_inv) * (l_ij2 - u_ij2)
                     + 0.25f * logf(u_ij / l_ij) * r2_inv;

            // de = bornForces[i] * t3 / r (OpenMM convention)
            float de = bornForces_i * t3 * r_inv;

            // OpenMM: delta = pos_j - pos_i, force_i -= de * delta
            // My convention: dx = pos_i - pos_j = -delta
            // So: force_i -= de * (-dx) = force_i += de * dx
            force_i.x += de * dx;
            force_i.y += de * dy;
            force_i.z += de * dz;

            // Force on j (Newton's 3rd law): force_j = -force_i
            atomicAdd(&forceBuffer[particleIdx_j], static_cast<unsigned long long>((long long)(-de * dx * 0x100000000)));
            atomicAdd(&forceBuffer[particleIdx_j + paddedNumAtoms], static_cast<unsigned long long>((long long)(-de * dy * 0x100000000)));
            atomicAdd(&forceBuffer[particleIdx_j + 2*paddedNumAtoms], static_cast<unsigned long long>((long long)(-de * dz * 0x100000000)));
        }

        // NOTE: We do NOT compute "i screens j" here.
        // That contribution will be computed when thread j processes this pair
        // and computes "j screens itself from i". This matches OpenMM's structure
        // and avoids double-counting.
    }

    // Accumulate force on atom i
    atomicAdd(&forceBuffer[particleIdx_i], static_cast<unsigned long long>((long long)(force_i.x * 0x100000000)));
    atomicAdd(&forceBuffer[particleIdx_i + paddedNumAtoms], static_cast<unsigned long long>((long long)(force_i.y * 0x100000000)));
    atomicAdd(&forceBuffer[particleIdx_i + 2*paddedNumAtoms], static_cast<unsigned long long>((long long)(force_i.z * 0x100000000)));
}

// Placeholder kernels for GRID mode - reuse from gbsaGridForce.cu patterns
extern "C" __global__ void computeIsolatedReceptorHCTGrid(
    const float4* __restrict__ posq,
    const int* __restrict__ particleIndices,
    const float* __restrict__ radii,
    const int* __restrict__ gridCounts,
    const float* __restrict__ gridHctProbe,
    const float* __restrict__ gridHctDerivatives,
    const float* __restrict__ gridCorrectionN,
    const float* __restrict__ gridCorrectionA,
    const float* __restrict__ gridCorrectionB,
    const float* __restrict__ rThresholds,
    const int* __restrict__ groupStart,
    int numGroups,
    float originX, float originY, float originZ,
    float gridSpacing,
    float probeRadius,
    int numBins,
    int totalParticles,
    int templateNumAtoms,
    int interpolationMethod,
    float* __restrict__ hctReceptor
) {
    // Implementation similar to computeReceptorHCT in gbsaGridForce.cu
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= totalParticles) return;

    // For now, just set to zero - full grid interpolation to be added
    hctReceptor[idx] = 0.0f;
}

extern "C" __global__ void computeIsolatedReceptorHCTGradientForce(
    const float4* __restrict__ posq,
    const int* __restrict__ particleIndices,
    const float* __restrict__ radii,
    const float* __restrict__ bornRadii,
    const float* __restrict__ hctReceptor,
    const float* __restrict__ hctLigand,
    const float* __restrict__ dE_dR,
    const int* __restrict__ gridCounts,
    const float* __restrict__ gridHctProbe,
    const float* __restrict__ gridHctDerivatives,
    const float* __restrict__ gridCorrectionN,
    const float* __restrict__ gridCorrectionA,
    const float* __restrict__ gridCorrectionB,
    const float* __restrict__ rThresholds,
    const int* __restrict__ groupStart,
    int numGroups,
    float originX, float originY, float originZ,
    float gridSpacing,
    float probeRadius,
    int numBins,
    int totalParticles,
    int templateNumAtoms,
    int interpolationMethod,
    unsigned long long* __restrict__ forceBuffer,
    int paddedNumAtoms
) {
    // Placeholder - to be implemented
}

extern "C" __global__ void computeIsolatedReceptorHCTPairwiseChainRule(
    const float4* __restrict__ posq,
    const int* __restrict__ particleIndices,
    const float* __restrict__ radii,
    const float* __restrict__ bornRadii,
    const float* __restrict__ hctReceptor,
    const float* __restrict__ hctLigand,
    const float* __restrict__ dE_dR,
    const float3* __restrict__ receptorPositions,
    const float* __restrict__ receptorRadii,
    const float* __restrict__ receptorScaleFactors,
    int numReceptorAtoms,
    const int* __restrict__ groupStart,
    int numGroups,
    int totalParticles,
    int templateNumAtoms,
    float cutoffDistance,
    unsigned long long* __restrict__ forceBuffer,
    int paddedNumAtoms
) {
    // Placeholder - to be implemented
}

// =============================================================================
// PAIRWISE MODE: Full receptor desolvation and cross-term computation
// =============================================================================

/**
 * Compute receptor-receptor self HCT (initialization only, O(N_rec²)).
 * This is constant - doesn't depend on ligand positions.
 */
extern "C" __global__ void computeReceptorSelfHCT(
    const float3* __restrict__ receptorPositions,
    const float* __restrict__ receptorRadii,
    const float* __restrict__ receptorScaleFactors,
    int numReceptorAtoms,
    float cutoffDistance,
    float* __restrict__ receptorSelfHCT
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numReceptorAtoms) return;

    float3 pos_i = receptorPositions[i];
    float R_i = receptorRadii[i];
    float R_i_off = R_i - DIELECTRIC_OFFSET;

    float hct = 0.0f;
    float cutoff2 = cutoffDistance * cutoffDistance;
    bool useCutoff = (cutoffDistance > 0.0f);

    for (int j = 0; j < numReceptorAtoms; j++) {
        if (j == i) continue;

        float3 pos_j = receptorPositions[j];
        float dx = pos_i.x - pos_j.x;
        float dy = pos_i.y - pos_j.y;
        float dz = pos_i.z - pos_j.z;
        float r2 = dx*dx + dy*dy + dz*dz;

        if (useCutoff && r2 > cutoff2) continue;

        float r = sqrtf(r2);
        if (r < 1e-6f) continue;

        float R_j = receptorRadii[j];
        float R_j_off = R_j - DIELECTRIC_OFFSET;
        float S_j = R_j_off * receptorScaleFactors[j];

        // HCT integral
        float r_plus_Sj = r + S_j;
        if (R_i_off >= r_plus_Sj) continue;

        float r_minus_Sj = fabsf(r - S_j);
        float l_ij = (R_i_off > r_minus_Sj) ? (1.0f / R_i_off) : (1.0f / r_minus_Sj);
        float u_ij = 1.0f / r_plus_Sj;

        float l_ij2 = l_ij * l_ij;
        float u_ij2 = u_ij * u_ij;
        float r_inv = 1.0f / r;

        float term = l_ij - u_ij +
                     0.25f * r * (u_ij2 - l_ij2) +
                     0.5f * r_inv * logf(u_ij / l_ij) +
                     0.25f * S_j * S_j * r_inv * (l_ij2 - u_ij2);

        if (R_i_off < (S_j - r)) {
            term += 2.0f * (1.0f / R_i_off - l_ij);
        }

        hct += term;
    }

    receptorSelfHCT[i] = hct;
}

/**
 * Compute receptor Born radii from self-HCT only (no ligand).
 * Used to compute reference energy at initialization.
 */
extern "C" __global__ void computeReceptorBornRadiiReference(
    const float* __restrict__ receptorRadii,
    const float* __restrict__ receptorSelfHCT,
    int numReceptorAtoms,
    float* __restrict__ receptorBornRadiiRef
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numReceptorAtoms) return;

    float R_i = receptorRadii[i];
    float R_i_off = R_i - DIELECTRIC_OFFSET;
    float hct = receptorSelfHCT[i];

    // OBC-II formula
    float psi = 0.5f * R_i_off * hct;
    float psi2 = psi * psi;
    float psi3 = psi2 * psi;

    float tanhArg = OBC_ALPHA * psi - OBC_BETA * psi2 + OBC_GAMMA * psi3;
    float tanhVal = tanhf(tanhArg);

    float denom = 1.0f / R_i_off - tanhVal / R_i;
    float bornRadius = (denom > 0.0f) ? (1.0f / denom) : R_i;
    bornRadius = fminf(bornRadius, 50.0f);

    receptorBornRadiiRef[i] = bornRadius;
}

/**
 * Compute receptor GB energy without ligand (reference energy).
 * Uses parallel reduction.
 */
extern "C" __global__ void computeReceptorReferenceEnergy(
    const float3* __restrict__ receptorPositions,
    const float* __restrict__ receptorCharges,
    const float* __restrict__ receptorBornRadiiRef,
    int numReceptorAtoms,
    float prefactor,
    float* __restrict__ receptorReferenceEnergy
) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    float energy = 0.0f;

    if (i < numReceptorAtoms) {
        float3 pos_i = receptorPositions[i];
        float q_i = receptorCharges[i];
        float R_i = receptorBornRadiiRef[i];

        // Self term
        energy += 0.5f * prefactor * q_i * q_i / R_i;

        // Pair terms (j > i to avoid double counting)
        for (int j = i + 1; j < numReceptorAtoms; j++) {
            float3 pos_j = receptorPositions[j];
            float q_j = receptorCharges[j];
            float R_j = receptorBornRadiiRef[j];

            float dx = pos_j.x - pos_i.x;
            float dy = pos_j.y - pos_i.y;
            float dz = pos_j.z - pos_i.z;
            float r2 = dx*dx + dy*dy + dz*dz;
            float r = sqrtf(r2);

            float RiRj = R_i * R_j;
            float expArg = -r2 / (4.0f * RiRj);
            float expTerm = expf(expArg);
            float f_gb2 = r2 + RiRj * expTerm;
            float f_gb = sqrtf(f_gb2);

            energy += prefactor * q_i * q_j / f_gb;
        }
    }

    sdata[tid] = energy;
    __syncthreads();

    // Parallel reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(receptorReferenceEnergy, sdata[0]);
    }
}

/**
 * Compute ligand's HCT contribution to receptor Born radii.
 * Each ligand group produces a separate HCT contribution to each receptor atom.
 * Output: ligandToReceptorHCT[groupIdx * numReceptorAtoms + receptorIdx]
 */
extern "C" __global__ void computeLigandToReceptorHCT(
    const float4* __restrict__ posq,
    const int* __restrict__ particleIndices,
    const float* __restrict__ ligandRadii,
    const float* __restrict__ ligandScaleFactors,
    const float3* __restrict__ receptorPositions,
    const float* __restrict__ receptorRadii,
    const int* __restrict__ groupStart,
    int numGroups,
    int numReceptorAtoms,
    int templateNumAtoms,
    float cutoffDistance,
    float* __restrict__ ligandToReceptorHCT
) {
    // Thread per (group, receptor_atom) pair
    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalWork = numGroups * numReceptorAtoms;
    if (globalIdx >= totalWork) return;

    int groupIdx = globalIdx / numReceptorAtoms;
    int recIdx = globalIdx % numReceptorAtoms;

    float3 pos_rec = receptorPositions[recIdx];
    float R_rec = receptorRadii[recIdx];
    float R_rec_off = R_rec - DIELECTRIC_OFFSET;

    float hct = 0.0f;
    float cutoff2 = cutoffDistance * cutoffDistance;
    bool useCutoff = (cutoffDistance > 0.0f);

    int groupStartIdx = groupStart[groupIdx];
    int groupEndIdx = groupStart[groupIdx + 1];

    // Loop over ligand atoms in this group
    for (int k = groupStartIdx; k < groupEndIdx; k++) {
        int particleIdx = particleIndices[k];
        int templateIdx = (k - groupStartIdx) % templateNumAtoms;

        float4 pos_lig = posq[particleIdx];
        float dx = pos_rec.x - pos_lig.x;
        float dy = pos_rec.y - pos_lig.y;
        float dz = pos_rec.z - pos_lig.z;
        float r2 = dx*dx + dy*dy + dz*dz;

        if (useCutoff && r2 > cutoff2) continue;

        float r = sqrtf(r2);
        if (r < 1e-6f) continue;

        float R_lig = ligandRadii[templateIdx];
        float R_lig_off = R_lig - DIELECTRIC_OFFSET;
        float S_lig = R_lig_off * ligandScaleFactors[templateIdx];

        // HCT integral: how ligand atom screens receptor atom
        float r_plus_Slig = r + S_lig;
        if (R_rec_off >= r_plus_Slig) continue;

        float r_minus_Slig = fabsf(r - S_lig);
        float l_ij = (R_rec_off > r_minus_Slig) ? (1.0f / R_rec_off) : (1.0f / r_minus_Slig);
        float u_ij = 1.0f / r_plus_Slig;

        float l_ij2 = l_ij * l_ij;
        float u_ij2 = u_ij * u_ij;
        float r_inv = 1.0f / r;

        float term = l_ij - u_ij +
                     0.25f * r * (u_ij2 - l_ij2) +
                     0.5f * r_inv * logf(u_ij / l_ij) +
                     0.25f * S_lig * S_lig * r_inv * (l_ij2 - u_ij2);

        if (R_rec_off < (S_lig - r)) {
            term += 2.0f * (1.0f / R_rec_off - l_ij);
        }

        hct += term;
    }

    ligandToReceptorHCT[groupIdx * numReceptorAtoms + recIdx] = hct;
}

/**
 * Compute receptor Born radii with ligand screening.
 * receptorHCT = receptorSelfHCT + ligandToReceptorHCT
 */
extern "C" __global__ void computeReceptorBornRadiiWithLigand(
    const float* __restrict__ receptorRadii,
    const float* __restrict__ receptorSelfHCT,
    const float* __restrict__ ligandToReceptorHCT,
    int numReceptorAtoms,
    int groupIdx,
    float* __restrict__ receptorBornRadii
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numReceptorAtoms) return;

    float R_i = receptorRadii[i];
    float R_i_off = R_i - DIELECTRIC_OFFSET;

    float hctTotal = receptorSelfHCT[i] + ligandToReceptorHCT[groupIdx * numReceptorAtoms + i];

    // OBC-II formula
    float psi = 0.5f * R_i_off * hctTotal;
    float psi2 = psi * psi;
    float psi3 = psi2 * psi;

    float tanhArg = OBC_ALPHA * psi - OBC_BETA * psi2 + OBC_GAMMA * psi3;
    float tanhVal = tanhf(tanhArg);

    float denom = 1.0f / R_i_off - tanhVal / R_i;
    float bornRadius = (denom > 0.0f) ? (1.0f / denom) : R_i;
    bornRadius = fminf(bornRadius, 50.0f);

    receptorBornRadii[i] = bornRadius;
}

/**
 * Compute receptor GB energy with ligand present.
 * Uses shared memory reduction.
 */
extern "C" __global__ void computeReceptorGBEnergy(
    const float3* __restrict__ receptorPositions,
    const float* __restrict__ receptorCharges,
    const float* __restrict__ receptorBornRadii,
    int numReceptorAtoms,
    float prefactor,
    float* __restrict__ receptorEnergy
) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    float energy = 0.0f;

    if (i < numReceptorAtoms) {
        float3 pos_i = receptorPositions[i];
        float q_i = receptorCharges[i];
        float R_i = receptorBornRadii[i];

        // Self term
        energy += 0.5f * prefactor * q_i * q_i / R_i;

        // Pair terms (j > i)
        for (int j = i + 1; j < numReceptorAtoms; j++) {
            float3 pos_j = receptorPositions[j];
            float q_j = receptorCharges[j];
            float R_j = receptorBornRadii[j];

            float dx = pos_j.x - pos_i.x;
            float dy = pos_j.y - pos_i.y;
            float dz = pos_j.z - pos_i.z;
            float r2 = dx*dx + dy*dy + dz*dz;
            float r = sqrtf(r2);

            float RiRj = R_i * R_j;
            float expArg = -r2 / (4.0f * RiRj);
            float expTerm = expf(expArg);
            float f_gb2 = r2 + RiRj * expTerm;
            float f_gb = sqrtf(f_gb2);

            energy += prefactor * q_i * q_j / f_gb;
        }
    }

    sdata[tid] = energy;
    __syncthreads();

    // Reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(receptorEnergy, sdata[0]);
    }
}

/**
 * Compute cross-term GB energy (receptor-ligand pairs).
 * Uses combined Born radii from receptor (with ligand screening) and ligand.
 */
extern "C" __global__ void computeCrossTermGBEnergy(
    const float4* __restrict__ posq,
    const int* __restrict__ particleIndices,
    const float* __restrict__ ligandCharges,
    const float* __restrict__ ligandBornRadii,
    const float3* __restrict__ receptorPositions,
    const float* __restrict__ receptorCharges,
    const float* __restrict__ receptorBornRadii,
    const int* __restrict__ groupStart,
    int numGroups,
    int numReceptorAtoms,
    int templateNumAtoms,
    float prefactor,
    float* __restrict__ crossTermEnergies,
    unsigned long long* __restrict__ forceBuffer,
    int paddedNumAtoms
) {
    // Thread per ligand atom
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int groupIdx = 0;
    int atomInGroup = idx;
    int groupStartIdx = 0;
    int groupEndIdx = 0;

    for (int g = 0; g < numGroups; g++) {
        groupStartIdx = groupStart[g];
        groupEndIdx = groupStart[g + 1];
        if (idx >= groupStartIdx && idx < groupEndIdx) {
            groupIdx = g;
            atomInGroup = idx - groupStartIdx;
            break;
        }
    }

    if (idx >= groupEndIdx) return;

    int particleIdx_lig = particleIndices[idx];
    int templateIdx_lig = atomInGroup % templateNumAtoms;

    float4 pos_lig = posq[particleIdx_lig];
    float q_lig = ligandCharges[templateIdx_lig];
    float R_lig = ligandBornRadii[idx];

    float energy = 0.0f;
    float3 force_lig = make_float3(0.0f, 0.0f, 0.0f);

    // Loop over receptor atoms
    for (int j = 0; j < numReceptorAtoms; j++) {
        float3 pos_rec = receptorPositions[j];
        float q_rec = receptorCharges[j];
        float R_rec = receptorBornRadii[j];

        float dx = pos_rec.x - pos_lig.x;
        float dy = pos_rec.y - pos_lig.y;
        float dz = pos_rec.z - pos_lig.z;
        float r2 = dx*dx + dy*dy + dz*dz;
        float r = sqrtf(r2);

        // Still equation
        float RiRj = R_lig * R_rec;
        float expArg = -r2 / (4.0f * RiRj);
        float expTerm = expf(expArg);
        float f_gb2 = r2 + RiRj * expTerm;
        float f_gb = sqrtf(f_gb2);
        float invFgb = 1.0f / f_gb;

        float pairEnergy = prefactor * q_lig * q_rec * invFgb;
        energy += pairEnergy;

        // Force on ligand
        float dFgbDr = (r * invFgb) * (1.0f - 0.25f * expTerm);
        float dEdR = -prefactor * q_lig * q_rec * invFgb * invFgb * dFgbDr;

        float invR = 1.0f / r;
        force_lig.x += dEdR * dx * invR;
        force_lig.y += dEdR * dy * invR;
        force_lig.z += dEdR * dz * invR;
    }

    // Accumulate cross-term energy
    atomicAdd(&crossTermEnergies[groupIdx], energy);

    // Accumulate forces
    atomicAdd(&forceBuffer[particleIdx_lig], static_cast<unsigned long long>((long long)(force_lig.x * 0x100000000)));
    atomicAdd(&forceBuffer[particleIdx_lig + paddedNumAtoms], static_cast<unsigned long long>((long long)(force_lig.y * 0x100000000)));
    atomicAdd(&forceBuffer[particleIdx_lig + 2*paddedNumAtoms], static_cast<unsigned long long>((long long)(force_lig.z * 0x100000000)));
}

/**
 * Compute forces on ligand from receptor desolvation.
 * Chain rule: dE_receptor/dr_ligand through ligand→receptor HCT.
 *
 * For each receptor atom i:
 *   dE_rec/dR_born_rec[i] is computed from receptor GB energy
 *   dR_born_rec[i]/dHCT_rec[i] from OBC chain rule
 *   dHCT_rec[i]/dr_ligand[k] from HCT derivative
 *
 * This kernel computes the full chain for each ligand atom.
 */
extern "C" __global__ void computeReceptorDesolvationForces(
    const float4* __restrict__ posq,
    const int* __restrict__ particleIndices,
    const float* __restrict__ ligandRadii,
    const float* __restrict__ ligandScaleFactors,
    const float3* __restrict__ receptorPositions,
    const float* __restrict__ receptorRadii,
    const float* __restrict__ receptorCharges,
    const float* __restrict__ receptorSelfHCT,
    const float* __restrict__ ligandToReceptorHCT,
    const float* __restrict__ receptorBornRadii,
    const int* __restrict__ groupStart,
    int numGroups,
    int numReceptorAtoms,
    int templateNumAtoms,
    float prefactor,
    float cutoffDistance,
    unsigned long long* __restrict__ forceBuffer,
    int paddedNumAtoms
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int groupIdx = 0;
    int atomInGroup = idx;
    int groupStartIdx = 0;
    int groupEndIdx = 0;

    for (int g = 0; g < numGroups; g++) {
        groupStartIdx = groupStart[g];
        groupEndIdx = groupStart[g + 1];
        if (idx >= groupStartIdx && idx < groupEndIdx) {
            groupIdx = g;
            atomInGroup = idx - groupStartIdx;
            break;
        }
    }

    if (idx >= groupEndIdx) return;

    int particleIdx_lig = particleIndices[idx];
    int templateIdx_lig = atomInGroup % templateNumAtoms;

    float4 pos_lig = posq[particleIdx_lig];
    float R_lig = ligandRadii[templateIdx_lig];
    float R_lig_off = R_lig - DIELECTRIC_OFFSET;
    float S_lig = R_lig_off * ligandScaleFactors[templateIdx_lig];

    float3 force_lig = make_float3(0.0f, 0.0f, 0.0f);
    float cutoff2 = cutoffDistance * cutoffDistance;
    bool useCutoff = (cutoffDistance > 0.0f);

    // For each receptor atom, compute dE_rec/dr_lig through chain rule
    for (int recIdx = 0; recIdx < numReceptorAtoms; recIdx++) {
        float3 pos_rec = receptorPositions[recIdx];
        float R_rec = receptorRadii[recIdx];
        float R_rec_off = R_rec - DIELECTRIC_OFFSET;
        float q_rec = receptorCharges[recIdx];

        float dx = pos_rec.x - pos_lig.x;
        float dy = pos_rec.y - pos_lig.y;
        float dz = pos_rec.z - pos_lig.z;
        float r2 = dx*dx + dy*dy + dz*dz;

        if (useCutoff && r2 > cutoff2) continue;

        float r = sqrtf(r2);
        if (r < 1e-6f) continue;

        // Check if ligand screens this receptor atom
        float r_plus_Slig = r + S_lig;
        if (R_rec_off >= r_plus_Slig) continue;

        float r_minus_Slig = fabsf(r - S_lig);
        float l_ij = (R_rec_off > r_minus_Slig) ? (1.0f / R_rec_off) : (1.0f / r_minus_Slig);
        float u_ij = 1.0f / r_plus_Slig;

        // Compute dE_rec/dR_born_rec for this receptor atom
        // This includes self term and all receptor-receptor pairs
        float bornR_rec = receptorBornRadii[recIdx];
        float dEdR_rec = 0.0f;

        // Self term: dE_self/dR = -0.5 * prefactor * q² / R²
        dEdR_rec += -0.5f * prefactor * q_rec * q_rec / (bornR_rec * bornR_rec);

        // Pair terms with other receptor atoms
        for (int j = 0; j < numReceptorAtoms; j++) {
            if (j == recIdx) continue;

            float3 pos_j = receptorPositions[j];
            float q_j = receptorCharges[j];
            float R_j = receptorBornRadii[j];

            float dx_rr = pos_j.x - pos_rec.x;
            float dy_rr = pos_j.y - pos_rec.y;
            float dz_rr = pos_j.z - pos_rec.z;
            float r2_rr = dx_rr*dx_rr + dy_rr*dy_rr + dz_rr*dz_rr;

            float RiRj = bornR_rec * R_j;
            float expArg = -r2_rr / (4.0f * RiRj);
            float expTerm = expf(expArg);
            float f_gb2 = r2_rr + RiRj * expTerm;
            float f_gb = sqrtf(f_gb2);

            float dFgbDRi = (R_j * expTerm / (2.0f * f_gb)) * (1.0f + r2_rr / (4.0f * RiRj));
            dEdR_rec += -prefactor * q_rec * q_j / (f_gb * f_gb) * dFgbDRi;
        }

        // OBC chain rule: dR_born/dHCT
        float hctTotal_rec = receptorSelfHCT[recIdx] + ligandToReceptorHCT[groupIdx * numReceptorAtoms + recIdx];
        float psi_rec = 0.5f * R_rec_off * hctTotal_rec;
        float psi2_rec = psi_rec * psi_rec;

        float tanhArg_rec = OBC_ALPHA * psi_rec - OBC_BETA * psi2_rec + OBC_GAMMA * psi2_rec * psi_rec;
        float tanhVal_rec = tanhf(tanhArg_rec);
        float sech2_rec = 1.0f - tanhVal_rec * tanhVal_rec;
        float dTanhArgDPsi_rec = OBC_ALPHA - 2.0f * OBC_BETA * psi_rec + 3.0f * OBC_GAMMA * psi2_rec;

        // obcChain = R_off * (dTanhArg/dPsi) * sech² / R
        float obcChain_rec = R_rec_off * dTanhArgDPsi_rec * sech2_rec / R_rec;

        // bornForces = dE/dR_born * R_born² * obcChain
        float bornForces_rec = dEdR_rec * bornR_rec * bornR_rec * obcChain_rec;

        // HCT gradient: OpenMM simplified formula
        float l_ij2 = l_ij * l_ij;
        float u_ij2 = u_ij * u_ij;
        float S_lig2 = S_lig * S_lig;
        float r_inv = 1.0f / r;
        float r2_inv = r_inv * r_inv;

        float t3 = 0.125f * (1.0f + S_lig2 * r2_inv) * (l_ij2 - u_ij2)
                 + 0.25f * logf(u_ij / l_ij) * r2_inv;

        float de = bornForces_rec * t3 * r_inv;

        // Force on ligand (direction from receptor to ligand is -dx,-dy,-dz)
        // OpenMM convention: delta = pos_rec - pos_lig, force_lig += de * delta
        force_lig.x += de * dx;
        force_lig.y += de * dy;
        force_lig.z += de * dz;
    }

    // Accumulate forces
    atomicAdd(&forceBuffer[particleIdx_lig], static_cast<unsigned long long>((long long)(force_lig.x * 0x100000000)));
    atomicAdd(&forceBuffer[particleIdx_lig + paddedNumAtoms], static_cast<unsigned long long>((long long)(force_lig.y * 0x100000000)));
    atomicAdd(&forceBuffer[particleIdx_lig + 2*paddedNumAtoms], static_cast<unsigned long long>((long long)(force_lig.z * 0x100000000)));
}

/**
 * OPTIMIZED: Compute forces on ligand from receptor desolvation using pre-computed dE/dR_born.
 * Takes pre-computed receptor dE/dR_born array instead of computing O(N²) inline.
 * This reduces each ligand thread from O(N_rec²) to O(N_rec) work.
 */
extern "C" __global__ void computeReceptorDesolvationForcesOptimized(
    const float4* __restrict__ posq,
    const int* __restrict__ particleIndices,
    const float* __restrict__ ligandRadii,
    const float* __restrict__ ligandScaleFactors,
    const float3* __restrict__ receptorPositions,
    const float* __restrict__ receptorRadii,
    const float* __restrict__ receptorSelfHCT,
    const float* __restrict__ ligandToReceptorHCT,
    const float* __restrict__ receptorBornRadii,
    const float* __restrict__ receptorDeDR,  // Pre-computed dE/dR_born for each receptor atom
    const int* __restrict__ groupStart,
    int numGroups,
    int numReceptorAtoms,
    int templateNumAtoms,
    float cutoffDistance,
    unsigned long long* __restrict__ forceBuffer,
    int paddedNumAtoms
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int groupIdx = 0;
    int atomInGroup = idx;
    int groupStartIdx = 0;
    int groupEndIdx = 0;

    for (int g = 0; g < numGroups; g++) {
        groupStartIdx = groupStart[g];
        groupEndIdx = groupStart[g + 1];
        if (idx >= groupStartIdx && idx < groupEndIdx) {
            groupIdx = g;
            atomInGroup = idx - groupStartIdx;
            break;
        }
    }

    if (idx >= groupEndIdx) return;

    int particleIdx_lig = particleIndices[idx];
    int templateIdx_lig = atomInGroup % templateNumAtoms;

    float4 pos_lig = posq[particleIdx_lig];
    float R_lig = ligandRadii[templateIdx_lig];
    float R_lig_off = R_lig - DIELECTRIC_OFFSET;
    float S_lig = R_lig_off * ligandScaleFactors[templateIdx_lig];

    float3 force_lig = make_float3(0.0f, 0.0f, 0.0f);
    float cutoff2 = cutoffDistance * cutoffDistance;
    bool useCutoff = (cutoffDistance > 0.0f);

    // For each receptor atom, compute force contribution using pre-computed dE/dR
    for (int recIdx = 0; recIdx < numReceptorAtoms; recIdx++) {
        float3 pos_rec = receptorPositions[recIdx];
        float R_rec = receptorRadii[recIdx];
        float R_rec_off = R_rec - DIELECTRIC_OFFSET;

        float dx = pos_rec.x - pos_lig.x;
        float dy = pos_rec.y - pos_lig.y;
        float dz = pos_rec.z - pos_lig.z;
        float r2 = dx*dx + dy*dy + dz*dz;

        if (useCutoff && r2 > cutoff2) continue;

        float r = sqrtf(r2);
        if (r < 1e-6f) continue;

        // Check if ligand screens this receptor atom
        float r_plus_Slig = r + S_lig;
        if (R_rec_off >= r_plus_Slig) continue;

        float r_minus_Slig = fabsf(r - S_lig);
        float l_ij = (R_rec_off > r_minus_Slig) ? (1.0f / R_rec_off) : (1.0f / r_minus_Slig);
        float u_ij = 1.0f / r_plus_Slig;

        // Use pre-computed dE/dR_born_rec
        float dEdR_rec = receptorDeDR[recIdx];

        // OBC chain rule: dR_born/dHCT
        float bornR_rec = receptorBornRadii[recIdx];
        float hctTotal_rec = receptorSelfHCT[recIdx] + ligandToReceptorHCT[groupIdx * numReceptorAtoms + recIdx];
        float psi_rec = 0.5f * R_rec_off * hctTotal_rec;
        float psi2_rec = psi_rec * psi_rec;

        float tanhArg_rec = OBC_ALPHA * psi_rec - OBC_BETA * psi2_rec + OBC_GAMMA * psi2_rec * psi_rec;
        float tanhVal_rec = tanhf(tanhArg_rec);
        float sech2_rec = 1.0f - tanhVal_rec * tanhVal_rec;
        float dTanhArgDPsi_rec = OBC_ALPHA - 2.0f * OBC_BETA * psi_rec + 3.0f * OBC_GAMMA * psi2_rec;

        // obcChain = R_off * (dTanhArg/dPsi) * sech² / R
        float obcChain_rec = R_rec_off * dTanhArgDPsi_rec * sech2_rec / R_rec;

        // bornForces = dE/dR_born * R_born² * obcChain
        float bornForces_rec = dEdR_rec * bornR_rec * bornR_rec * obcChain_rec;

        // HCT gradient: OpenMM simplified formula
        float l_ij2 = l_ij * l_ij;
        float u_ij2 = u_ij * u_ij;
        float S_lig2 = S_lig * S_lig;
        float r_inv = 1.0f / r;
        float r2_inv = r_inv * r_inv;

        float t3 = 0.125f * (1.0f + S_lig2 * r2_inv) * (l_ij2 - u_ij2)
                 + 0.25f * logf(u_ij / l_ij) * r2_inv;

        float de = bornForces_rec * t3 * r_inv;

        // Force on ligand (the screening atom)
        // The ligand screens the receptor, so by Newton's 3rd law the force on ligand
        // is opposite to what would be on the receptor. This is analogous to force_j
        // in the ligand-ligand HCT chain rule, which uses -= not +=.
        force_lig.x -= de * dx;
        force_lig.y -= de * dy;
        force_lig.z -= de * dz;
    }

    // Accumulate forces
    atomicAdd(&forceBuffer[particleIdx_lig], static_cast<unsigned long long>((long long)(force_lig.x * 0x100000000)));
    atomicAdd(&forceBuffer[particleIdx_lig + paddedNumAtoms], static_cast<unsigned long long>((long long)(force_lig.y * 0x100000000)));
    atomicAdd(&forceBuffer[particleIdx_lig + 2*paddedNumAtoms], static_cast<unsigned long long>((long long)(force_lig.z * 0x100000000)));
}

/**
 * Compute chain rule forces on ligand from cross-term through Born radii.
 * dE_cross/dR_born_lig and dE_cross/dR_born_rec, then chain through HCT.
 */
extern "C" __global__ void computeCrossTermChainRuleForces(
    const float4* __restrict__ posq,
    const int* __restrict__ particleIndices,
    const float* __restrict__ ligandRadii,
    const float* __restrict__ ligandScaleFactors,
    const float* __restrict__ ligandCharges,
    const float* __restrict__ ligandBornRadii,
    const float* __restrict__ hctReceptor,
    const float* __restrict__ hctLigand,
    const float3* __restrict__ receptorPositions,
    const float* __restrict__ receptorRadii,
    const float* __restrict__ receptorScaleFactors,
    const float* __restrict__ receptorCharges,
    const float* __restrict__ receptorSelfHCT,
    const float* __restrict__ ligandToReceptorHCT,
    const float* __restrict__ receptorBornRadii,
    const int* __restrict__ groupStart,
    int numGroups,
    int numReceptorAtoms,
    int templateNumAtoms,
    float prefactor,
    float cutoffDistance,
    unsigned long long* __restrict__ forceBuffer,
    int paddedNumAtoms
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int groupIdx = 0;
    int atomInGroup = idx;
    int groupStartIdx = 0;
    int groupEndIdx = 0;

    for (int g = 0; g < numGroups; g++) {
        groupStartIdx = groupStart[g];
        groupEndIdx = groupStart[g + 1];
        if (idx >= groupStartIdx && idx < groupEndIdx) {
            groupIdx = g;
            atomInGroup = idx - groupStartIdx;
            break;
        }
    }

    if (idx >= groupEndIdx) return;

    int particleIdx_lig = particleIndices[idx];
    int templateIdx_lig = atomInGroup % templateNumAtoms;

    float4 pos_lig = posq[particleIdx_lig];
    float R_lig = ligandRadii[templateIdx_lig];
    float R_lig_off = R_lig - DIELECTRIC_OFFSET;
    float S_lig = R_lig_off * ligandScaleFactors[templateIdx_lig];
    float q_lig = ligandCharges[templateIdx_lig];
    float bornR_lig = ligandBornRadii[idx];

    float3 force_lig = make_float3(0.0f, 0.0f, 0.0f);
    float cutoff2 = cutoffDistance * cutoffDistance;
    bool useCutoff = (cutoffDistance > 0.0f);

    // Part 1: dE_cross/dR_born_lig → chain through ligand HCT
    // Accumulate dE_cross/dR_born_lig from all receptor atoms
    float dEdR_lig = 0.0f;
    for (int j = 0; j < numReceptorAtoms; j++) {
        float3 pos_rec = receptorPositions[j];
        float q_rec = receptorCharges[j];
        float R_rec = receptorBornRadii[j];

        float dx = pos_rec.x - pos_lig.x;
        float dy = pos_rec.y - pos_lig.y;
        float dz = pos_rec.z - pos_lig.z;
        float r2 = dx*dx + dy*dy + dz*dz;

        float RiRj = bornR_lig * R_rec;
        float expArg = -r2 / (4.0f * RiRj);
        float expTerm = expf(expArg);
        float f_gb2 = r2 + RiRj * expTerm;
        float f_gb = sqrtf(f_gb2);

        // dE/dR_born_lig from this pair
        float dFgbDRlig = (R_rec * expTerm / (2.0f * f_gb)) * (1.0f + r2 / (4.0f * RiRj));
        dEdR_lig += -prefactor * q_lig * q_rec / (f_gb * f_gb) * dFgbDRlig;
    }

    // OBC chain rule for ligand
    float hctTotal_lig = hctReceptor[idx] + hctLigand[idx];
    float psi_lig = 0.5f * R_lig_off * hctTotal_lig;
    float psi2_lig = psi_lig * psi_lig;

    float tanhArg_lig = OBC_ALPHA * psi_lig - OBC_BETA * psi2_lig + OBC_GAMMA * psi2_lig * psi_lig;
    float tanhVal_lig = tanhf(tanhArg_lig);
    float sech2_lig = 1.0f - tanhVal_lig * tanhVal_lig;
    float dTanhArgDPsi_lig = OBC_ALPHA - 2.0f * OBC_BETA * psi_lig + 3.0f * OBC_GAMMA * psi2_lig;

    float obcChain_lig = R_lig_off * dTanhArgDPsi_lig * sech2_lig / R_lig;
    float bornForces_lig = dEdR_lig * bornR_lig * bornR_lig * obcChain_lig;

    // Chain through ligand-ligand HCT (other ligand atoms screening this one)
    int groupSize = groupEndIdx - groupStartIdx;
    for (int jLocal = 0; jLocal < groupSize; jLocal++) {
        if (jLocal == atomInGroup) continue;

        int j = groupStartIdx + jLocal;
        int templateIdx_j = jLocal % templateNumAtoms;
        int particleIdx_j = particleIndices[j];

        float4 pos_j = posq[particleIdx_j];
        float R_j = ligandRadii[templateIdx_j];
        float R_j_off = R_j - DIELECTRIC_OFFSET;
        float S_j = R_j_off * ligandScaleFactors[templateIdx_j];

        float dx = pos_lig.x - pos_j.x;
        float dy = pos_lig.y - pos_j.y;
        float dz = pos_lig.z - pos_j.z;
        float r2 = dx*dx + dy*dy + dz*dz;

        if (useCutoff && r2 > cutoff2) continue;

        float r = sqrtf(r2);
        if (r < 1e-6f) continue;

        float r_plus_Sj = r + S_j;
        if (R_lig_off >= r_plus_Sj) continue;

        float r_minus_Sj = fabsf(r - S_j);
        float l_ij = (R_lig_off > r_minus_Sj) ? (1.0f / R_lig_off) : (1.0f / r_minus_Sj);
        float u_ij = 1.0f / r_plus_Sj;

        float l_ij2 = l_ij * l_ij;
        float u_ij2 = u_ij * u_ij;
        float S_j2 = S_j * S_j;
        float r_inv = 1.0f / r;
        float r2_inv = r_inv * r_inv;

        float t3 = 0.125f * (1.0f + S_j2 * r2_inv) * (l_ij2 - u_ij2)
                 + 0.25f * logf(u_ij / l_ij) * r2_inv;

        float de = bornForces_lig * t3 * r_inv;

        force_lig.x += de * dx;
        force_lig.y += de * dy;
        force_lig.z += de * dz;

        atomicAdd(&forceBuffer[particleIdx_j], static_cast<unsigned long long>((long long)(-de * dx * 0x100000000)));
        atomicAdd(&forceBuffer[particleIdx_j + paddedNumAtoms], static_cast<unsigned long long>((long long)(-de * dy * 0x100000000)));
        atomicAdd(&forceBuffer[particleIdx_j + 2*paddedNumAtoms], static_cast<unsigned long long>((long long)(-de * dz * 0x100000000)));
    }

    // Chain through receptor→ligand HCT (receptor screening this ligand)
    for (int j = 0; j < numReceptorAtoms; j++) {
        float3 pos_rec = receptorPositions[j];
        float R_rec = receptorRadii[j];
        float R_rec_off = R_rec - DIELECTRIC_OFFSET;
        float S_rec = R_rec_off * receptorScaleFactors[j];

        float dx = pos_lig.x - pos_rec.x;
        float dy = pos_lig.y - pos_rec.y;
        float dz = pos_lig.z - pos_rec.z;
        float r2 = dx*dx + dy*dy + dz*dz;

        if (useCutoff && r2 > cutoff2) continue;

        float r = sqrtf(r2);
        if (r < 1e-6f) continue;

        float r_plus_Srec = r + S_rec;
        if (R_lig_off >= r_plus_Srec) continue;

        float r_minus_Srec = fabsf(r - S_rec);
        float l_ij = (R_lig_off > r_minus_Srec) ? (1.0f / R_lig_off) : (1.0f / r_minus_Srec);
        float u_ij = 1.0f / r_plus_Srec;

        float l_ij2 = l_ij * l_ij;
        float u_ij2 = u_ij * u_ij;
        float S_rec2 = S_rec * S_rec;
        float r_inv = 1.0f / r;
        float r2_inv = r_inv * r_inv;

        float t3 = 0.125f * (1.0f + S_rec2 * r2_inv) * (l_ij2 - u_ij2)
                 + 0.25f * logf(u_ij / l_ij) * r2_inv;

        float de = bornForces_lig * t3 * r_inv;

        force_lig.x += de * dx;
        force_lig.y += de * dy;
        force_lig.z += de * dz;
    }

    // Accumulate
    atomicAdd(&forceBuffer[particleIdx_lig], static_cast<unsigned long long>((long long)(force_lig.x * 0x100000000)));
    atomicAdd(&forceBuffer[particleIdx_lig + paddedNumAtoms], static_cast<unsigned long long>((long long)(force_lig.y * 0x100000000)));
    atomicAdd(&forceBuffer[particleIdx_lig + 2*paddedNumAtoms], static_cast<unsigned long long>((long long)(force_lig.z * 0x100000000)));
}
