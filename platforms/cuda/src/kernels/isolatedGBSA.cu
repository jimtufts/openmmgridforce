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
 * Fused tiled computation of receptor GB energy AND dE/dR_born.
 * Single pass over all receptor-receptor pairs computes both quantities,
 * eliminating the separate O(N²) dE/dR kernel.
 */
extern "C" __global__ void computeReceptorGBEnergyAndDeDRTiled(
    const float3* __restrict__ receptorPositions,
    const float* __restrict__ receptorCharges,
    const float* __restrict__ receptorBornRadii,
    int numReceptorAtoms,
    float prefactor,
    float* __restrict__ receptorEnergy,    // [1] scalar output
    float* __restrict__ receptorDeDR,      // [numReceptorAtoms] per-atom output
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
    float myDeDR = 0.0f;  // dE/dR for atom1 (this thread's atom)

    int pos = (int)(((long long)warp * numTiles) / totalWarps);
    int end = (int)(((long long)(warp + 1) * numTiles) / totalWarps);

    // Track which atom this thread represents across tiles
    // (changes per tile, so we flush dE/dR when atom1 changes)
    int prevAtom1 = -1;

    while (pos < end) {
        int y = (int)floor(NUM_BLOCKS + 0.5f - sqrtf((NUM_BLOCKS + 0.5f) * (NUM_BLOCKS + 0.5f) - 2.0f * pos));
        int x = pos - y * NUM_BLOCKS + y * (y + 1) / 2;
        if (x < y || x >= NUM_BLOCKS) {
            y += (x < y ? -1 : 1);
            x = pos - y * NUM_BLOCKS + y * (y + 1) / 2;
        }

        unsigned int atom1 = x * TILE_SIZE + tgx;
        unsigned int atom2 = y * TILE_SIZE + tgx;

        // Flush dE/dR if atom1 changed
        if ((int)atom1 != prevAtom1 && prevAtom1 >= 0 && prevAtom1 < numReceptorAtoms) {
            atomicAdd(&receptorDeDR[prevAtom1], myDeDR);
            myDeDR = 0.0f;
        }
        prevAtom1 = atom1;

        // Load atom1
        float3 pos1 = make_float3(0, 0, 0);
        float q1 = 0, R1 = 1.0f;
        if (atom1 < numReceptorAtoms) {
            pos1 = receptorPositions[atom1];
            q1 = receptorCharges[atom1];
            R1 = receptorBornRadii[atom1];
        }

        // Load atom2 into shared memory
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
        localData[tbx + tgx].energy = 0.0f;  // used for atom2 dE/dR accumulation
        __syncwarp();

        if (x == y) {
            // Diagonal tile: self term + upper triangle
            if (atom1 < numReceptorAtoms) {
                energy += 0.5f * prefactor * q1 * q1 / R1;
                myDeDR += -0.5f * prefactor * q1 * q1 / (R1 * R1);
            }

            for (int j = tgx + 1; j < TILE_SIZE; j++) {
                int atom2_j = y * TILE_SIZE + j;
                if (atom1 < numReceptorAtoms && atom2_j < numReceptorAtoms) {
                    float dx = localData[tbx + j].x - pos1.x;
                    float dy = localData[tbx + j].y - pos1.y;
                    float dz = localData[tbx + j].z - pos1.z;
                    float r2 = dx*dx + dy*dy + dz*dz;

                    float q2 = localData[tbx + j].charge;
                    float R2 = localData[tbx + j].bornRadius;
                    float RiRj = R1 * R2;
                    float expArg = -r2 / (4.0f * RiRj);
                    float expTerm = expf(expArg);
                    float f_gb2 = r2 + RiRj * expTerm;
                    float f_gb = sqrtf(f_gb2);
                    float invFgb2 = 1.0f / f_gb2;

                    energy += prefactor * q1 * q2 / f_gb;

                    // dE/dR_born_i from pair (i,j)
                    float factor = -prefactor * q1 * q2 * invFgb2 / f_gb;
                    float dFgbDR1 = (R2 * expTerm / (2.0f * f_gb)) * (1.0f + r2 / (4.0f * RiRj));
                    float dFgbDR2 = (R1 * expTerm / (2.0f * f_gb)) * (1.0f + r2 / (4.0f * RiRj));
                    myDeDR += factor * dFgbDR1;
                    localData[tbx + j].energy += factor * dFgbDR2;
                }
            }

            // Write atom2 dE/dR contributions from shared memory
            __syncwarp();
            if (atom2 < numReceptorAtoms && localData[tbx + tgx].energy != 0.0f) {
                atomicAdd(&receptorDeDR[atom2], localData[tbx + tgx].energy);
            }

        } else {
            // Off-diagonal tile
            unsigned int tj = tgx;
            for (int j = 0; j < TILE_SIZE; j++) {
                int atom2_j = y * TILE_SIZE + tj;
                if (atom1 < numReceptorAtoms && atom2_j < numReceptorAtoms) {
                    float dx = localData[tbx + tj].x - pos1.x;
                    float dy = localData[tbx + tj].y - pos1.y;
                    float dz = localData[tbx + tj].z - pos1.z;
                    float r2 = dx*dx + dy*dy + dz*dz;

                    float q2 = localData[tbx + tj].charge;
                    float R2 = localData[tbx + tj].bornRadius;
                    float RiRj = R1 * R2;
                    float expArg = -r2 / (4.0f * RiRj);
                    float expTerm = expf(expArg);
                    float f_gb2 = r2 + RiRj * expTerm;
                    float f_gb = sqrtf(f_gb2);
                    float invFgb2 = 1.0f / f_gb2;

                    energy += prefactor * q1 * q2 / f_gb;

                    float factor = -prefactor * q1 * q2 * invFgb2 / f_gb;
                    float dFgbDR1 = (R2 * expTerm / (2.0f * f_gb)) * (1.0f + r2 / (4.0f * RiRj));
                    float dFgbDR2 = (R1 * expTerm / (2.0f * f_gb)) * (1.0f + r2 / (4.0f * RiRj));
                    myDeDR += factor * dFgbDR1;
                    localData[tbx + tj].energy += factor * dFgbDR2;
                }
                tj = (tj + 1) & (TILE_SIZE - 1);
                __syncwarp();
            }

            // Write atom2 dE/dR from shared memory
            if (atom2 < numReceptorAtoms && localData[tbx + tgx].energy != 0.0f) {
                atomicAdd(&receptorDeDR[atom2], localData[tbx + tgx].energy);
            }
        }

        pos++;
    }

    // Flush remaining dE/dR
    if (prevAtom1 >= 0 && prevAtom1 < numReceptorAtoms && myDeDR != 0.0f) {
        atomicAdd(&receptorDeDR[prevAtom1], myDeDR);
    }

    // Reduce energy within block
    energyBuffer[threadIdx.x] = energy;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            energyBuffer[threadIdx.x] += energyBuffer[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0)
        atomicAdd(receptorEnergy, energyBuffer[0]);
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

        float invR = rsqrtf(r2);
        float r = r2 * invR;
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
/**
 * Rectangular tiled HCT kernel (OpenMM-style).
 *
 * Processes receptor-ligand pairs in 32×32 tiles, matching OpenMM's tiling
 * pattern but for asymmetric receptor×ligand interactions.
 *
 * Each warp handles one (group, rec_block, lig_block) tile.
 * - Thread tgx holds receptor atom (rec_block*32 + tgx) in registers
 * - Ligand atoms (lig_block*32 + 0..31) loaded into shared memory
 * - Inner loop: 32 iterations, both HCT directions computed per pair
 * - Receptor-side HCT accumulated in registers, one atomicAdd per tile
 * - Ligand-side HCT accumulated in shared memory, one atomicAdd per tile
 *
 * Tiles per group: (N_rec/32) × ceil(N_lig/32)
 * Total tiles: tiles_per_group × K
 */
extern "C" __global__ void computeReceptorLigandHCTTiled(
    const float4* __restrict__ posq,
    const int* __restrict__ particleIndices,
    const float* __restrict__ ligandRadii,
    const float* __restrict__ ligandScaleFactors,
    const float3* __restrict__ receptorPositions,
    const float* __restrict__ receptorRadii,
    const float* __restrict__ receptorScaleFactors,
    int numReceptorAtoms,
    const int* __restrict__ groupStart,
    int numGroups,
    int templateNumAtoms,
    float cutoffDistance,
    unsigned long long* __restrict__ global_hctReceptor,  // fixed-point accumulator
    unsigned long long* __restrict__ global_ligToRecHCT, // fixed-point accumulator [K * N_rec]
    int numTilesPerGroup,
    const float4* __restrict__ recBlockBounds,        // [numRecBlocks] (cx, cy, cz, radius) or NULL
    float localityCutoff,                             // tile-skip cutoff (-1 = no skip)
    float* __restrict__ hctRecBlockCache,             // [totalParticles * numRecBlocks] or NULL
    int numRecBlocks,                                 // for cache indexing
    float globalScalingFactor,                        // alchemical scaling
    const float* __restrict__ groupScalingFactors     // per-group scaling
) {
    const int tgx = threadIdx.x & (TILE_SIZE - 1);
    const int tbx = threadIdx.x - tgx;
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) / TILE_SIZE;
    const int totalWarps = (gridDim.x * blockDim.x) / TILE_SIZE;

    const int NUM_REC_BLOCKS = (numReceptorAtoms + TILE_SIZE - 1) / TILE_SIZE;

    __shared__ float3 sLigPos[256];    // shared memory for ligand tile (8 warps × 32)
    __shared__ float sLigR_off[256];
    __shared__ float sLigS[256];
    __shared__ float sLigHCT[256];     // ligand-side HCT accumulator

    int totalTiles = numTilesPerGroup * numGroups;
    float cutoff2 = cutoffDistance * cutoffDistance;
    bool useCutoff = (cutoffDistance > 0.0f);

    // Grid-stride over tiles
    for (int tileIdx = warp; tileIdx < totalTiles; tileIdx += totalWarps) {
        // Decompose tile index into (group, rec_block, lig_block)
        int tilesInGroup = tileIdx / numGroups;  // wrong - should be tileIdx % numTilesPerGroup
        int groupIdx = tileIdx / numTilesPerGroup;
        int tileInGroup = tileIdx % numTilesPerGroup;

        // Skip zero-scaled groups
        float scale = globalScalingFactor * groupScalingFactors[groupIdx];
        if (scale < 0.05f) continue;

        int gs = groupStart[groupIdx];
        int ge = groupStart[groupIdx + 1];
        int groupSize = ge - gs;
        int numLigBlocks = (groupSize + TILE_SIZE - 1) / TILE_SIZE;

        int recBlock = tileInGroup / numLigBlocks;
        int ligBlock = tileInGroup % numLigBlocks;

        // Load receptor atom for this thread
        int recIdx = recBlock * TILE_SIZE + tgx;
        float3 recPos = make_float3(0, 0, 0);
        float recR_off = 0.1f;
        float recS = 0.1f;
        bool validRec = (recIdx < numReceptorAtoms);
        if (validRec) {
            recPos = receptorPositions[recIdx];
            float R = receptorRadii[recIdx];
            recR_off = R - DIELECTRIC_OFFSET;
            recS = recR_off * receptorScaleFactors[recIdx];
        }

        // Load ligand tile into shared memory
        int ligLocalIdx = ligBlock * TILE_SIZE + tgx;
        int ligGlobalIdx = gs + ligLocalIdx;
        bool validLig = (ligLocalIdx < groupSize);
        if (validLig) {
            int particleIdx = particleIndices[ligGlobalIdx];
            float4 p = posq[particleIdx];
            sLigPos[tbx + tgx] = make_float3(p.x, p.y, p.z);
            int templateIdx = ligLocalIdx % templateNumAtoms;
            float R = ligandRadii[templateIdx];
            sLigR_off[tbx + tgx] = R - DIELECTRIC_OFFSET;
            sLigS[tbx + tgx] = (R - DIELECTRIC_OFFSET) * ligandScaleFactors[templateIdx];
        } else {
            sLigPos[tbx + tgx] = make_float3(0, 0, 0);
            sLigR_off[tbx + tgx] = 0.1f;
            sLigS[tbx + tgx] = 0.1f;
        }
        sLigHCT[tbx + tgx] = 0.0f;
        __syncwarp();

        // Tile-skip: check if any ligand atom in this tile is close to this receptor block
        bool useTileSkip = (localityCutoff > 0.0f && recBlockBounds != NULL);
        if (useTileSkip) {
            float4 bounds = recBlockBounds[recBlock];
            float threshold = localityCutoff + bounds.w;
            float threshold2 = threshold * threshold;
            bool anyClose = false;
            int nInTile = min(TILE_SIZE, groupSize - ligBlock * TILE_SIZE);
            for (int i = 0; i < nInTile && !anyClose; i++) {
                float dx = sLigPos[tbx + i].x - bounds.x;
                float dy = sLigPos[tbx + i].y - bounds.y;
                float dz = sLigPos[tbx + i].z - bounds.z;
                if (dx*dx + dy*dy + dz*dz < threshold2) anyClose = true;
            }
            if (!anyClose) continue;  // skip this tile
        }

        // Accumulate receptor-side HCT in register
        float recHCT = 0.0f;

        // Process all 32 ligand atoms in the tile
        unsigned int tj = tgx;
        for (int j = 0; j < TILE_SIZE; j++) {
            int ligLocal = ligBlock * TILE_SIZE + tj;
            bool vLig = (ligLocal < groupSize);

            if (validRec && vLig) {
                float dx = recPos.x - sLigPos[tbx + tj].x;
                float dy = recPos.y - sLigPos[tbx + tj].y;
                float dz = recPos.z - sLigPos[tbx + tj].z;
                float r2 = dx*dx + dy*dy + dz*dz;

                if (!useCutoff || r2 < cutoff2) {
                    float invR = rsqrtf(r2);
                    float r = r2 * invR;

                    if (r > 1e-6f) {
                        float lS = sLigS[tbx + tj];
                        float lR_off = sLigR_off[tbx + tj];

                        // Ligand→Receptor: ligand screens receptor
                        float r_plus_Si = r + lS;
                        if (recR_off < r_plus_Si) {
                            float r_minus_Si = fabsf(r - lS);
                            float l = (recR_off > r_minus_Si) ? (1.0f/recR_off) : (1.0f/r_minus_Si);
                            float u = 1.0f / r_plus_Si;
                            float l2 = l*l, u2 = u*u;
                            float term = l - u + 0.25f*r*(u2-l2) + 0.5f*(1.0f/r)*logf(u/l) + 0.25f*lS*lS*(1.0f/r)*(l2-u2);
                            if (recR_off < (lS - r)) term += 2.0f*(1.0f/recR_off - l);
                            recHCT += term;
                        }

                        // Receptor→Ligand: receptor screens ligand
                        float r_plus_Sj = r + recS;
                        if (lR_off < r_plus_Sj) {
                            float r_minus_Sj = fabsf(r - recS);
                            float l = (lR_off > r_minus_Sj) ? (1.0f/lR_off) : (1.0f/r_minus_Sj);
                            float u = 1.0f / r_plus_Sj;
                            float l2 = l*l, u2 = u*u;
                            float term = l - u + 0.25f*r*(u2-l2) + 0.5f*(1.0f/r)*logf(u/l) + 0.25f*recS*recS*(1.0f/r)*(l2-u2);
                            if (lR_off < (recS - r)) term += 2.0f*(1.0f/lR_off - l);
                            sLigHCT[tbx + tj] += term;
                        }
                    }
                }
            }
            tj = (tj + 1) & (TILE_SIZE - 1);
            __syncwarp();
        }

        // Write results via fixed-point atomicAdd (one per atom per tile)
        if (validRec && recHCT != 0.0f) {
            // Ligand→receptor HCT for this receptor atom in this group
            atomicAdd(&global_ligToRecHCT[groupIdx * numReceptorAtoms + recIdx],
                      (unsigned long long)(long long)(recHCT * 0x100000000));
        }

        // Ligand-side: write accumulated HCT from shared memory
        float ligHCTVal = sLigHCT[tbx + tgx];
        if (validLig && ligHCTVal != 0.0f) {
            atomicAdd(&global_hctReceptor[ligGlobalIdx],
                      (unsigned long long)(long long)(ligHCTVal * 0x100000000));
        }

        // Write per-block cache for receptor→ligand direction
        if (hctRecBlockCache != NULL && validLig) {
            hctRecBlockCache[ligGlobalIdx * numRecBlocks + recBlock] = ligHCTVal;
        }
    }
}

/**
 * Add cached HCT values for distant receptor blocks (receptor→ligand direction).
 * After a tile-skipped tiled HCT, distant blocks contributed zero. This kernel
 * adds the cached per-block partial sums for blocks that were skipped.
 */
extern "C" __global__ void addDistantHCTFromCache(
    const float* __restrict__ hctRecBlockCache,  // [totalParticles * numRecBlocks]
    const float4* __restrict__ posq,
    const int* __restrict__ particleIndices,
    const float4* __restrict__ recBlockBounds,   // [numRecBlocks]
    float localityCutoff,
    int numRecBlocks,
    int totalParticles,
    unsigned long long* __restrict__ global_hctReceptor  // fixed-point accumulator to add to
) {
    int ligIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (ligIdx >= totalParticles) return;

    int particleIdx = particleIndices[ligIdx];
    float4 p = posq[particleIdx];

    float cachedSum = 0.0f;
    for (int b = 0; b < numRecBlocks; b++) {
        float4 bounds = recBlockBounds[b];
        float threshold = localityCutoff + bounds.w;
        float dx = p.x - bounds.x;
        float dy = p.y - bounds.y;
        float dz = p.z - bounds.z;
        if (dx*dx + dy*dy + dz*dz >= threshold * threshold) {
            // Block was distant (skipped by tiled kernel) → add cached value
            cachedSum += hctRecBlockCache[ligIdx * numRecBlocks + b];
        }
    }

    if (cachedSum != 0.0f) {
        atomicAdd(&global_hctReceptor[ligIdx],
                  (unsigned long long)(long long)(cachedSum * 0x100000000));
    }
}

/**
 * Restore cached ligand→receptor HCT for distant receptor atoms.
 * After tile-skipped HCT, distant receptor atoms got zero contribution.
 * This kernel replaces zero with the cached full value for distant atoms.
 */
extern "C" __global__ void restoreDistantLigToRecHCT(
    const float* __restrict__ ligToRecHCTCache,  // [K * N_rec] cached values
    float* __restrict__ ligandToReceptorHCT,     // [K * N_rec] current (post-conversion)
    const float4* __restrict__ posq,
    const int* __restrict__ particleIndices,
    const float4* __restrict__ recBlockBounds,
    float localityCutoff,
    const int* __restrict__ groupStart,
    int numGroups,
    int numReceptorAtoms,
    int totalParticles
) {
    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalWork = numGroups * numReceptorAtoms;
    if (globalIdx >= totalWork) return;

    int groupIdx = globalIdx / numReceptorAtoms;
    int recIdx = globalIdx % numReceptorAtoms;
    int recBlock = recIdx / TILE_SIZE;

    float4 bounds = recBlockBounds[recBlock];
    float threshold = localityCutoff + bounds.w;
    float threshold2 = threshold * threshold;

    // Check if ANY ligand atom in this group is near this block
    int gs = groupStart[groupIdx];
    int ge = groupStart[groupIdx + 1];
    bool anyClose = false;
    for (int li = gs; li < ge && !anyClose; li++) {
        int particleIdx = particleIndices[li];
        float4 p = posq[particleIdx];
        float dx = p.x - bounds.x;
        float dy = p.y - bounds.y;
        float dz = p.z - bounds.z;
        if (dx*dx + dy*dy + dz*dz < threshold2) anyClose = true;
    }

    if (!anyClose) {
        // Block was distant → restore cached value
        ligandToReceptorHCT[globalIdx] = ligToRecHCTCache[globalIdx];
    }
}

// Fixed-point to float conversion for tiled HCT results
extern "C" __global__ void convertTiledHCTToFloat(
    const unsigned long long* __restrict__ hctFixed,
    float* __restrict__ hctFloat,
    int numAtoms
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numAtoms) return;
    hctFloat[i] = (float)((long long)hctFixed[i] / (double)0x100000000);
}

/**
 * Receptor-threaded HCT kernel.
 *
 * Parallelizes over (group × receptor_atom) = K * N_rec threads.
 * Each thread loads one receptor atom and loops over N_lig ligand atoms
 * from shared memory. Both HCT directions computed simultaneously.
 *
 * Ligand data (47 atoms × 20 bytes = 940 bytes) fits in shared memory
 * and is broadcast to all threads in the warp.
 *
 * Output:
 *   hctReceptor[ligIdx] += receptor→ligand HCT (via warp reduction + atomicAdd)
 *   ligandToReceptorHCT[groupIdx * N_rec + recIdx] = ligand→receptor HCT (direct write)
 */
extern "C" __global__ void computeReceptorLigandHCTParallel(
    const float4* __restrict__ posq,
    const int* __restrict__ particleIndices,
    const float* __restrict__ ligandRadii,
    const float* __restrict__ ligandScaleFactors,
    const float3* __restrict__ receptorPositions,
    const float* __restrict__ receptorRadii,
    const float* __restrict__ receptorScaleFactors,
    int numReceptorAtoms,
    const int* __restrict__ groupStart,
    int numGroups,
    int templateNumAtoms,
    float cutoffDistance,
    float* __restrict__ hctReceptor,
    float* __restrict__ ligandToReceptorHCT
) {
    // Max ligand atoms we can handle in shared memory
    // 64 atoms × 20 bytes = 1280 bytes — well within shared memory limits
    const int MAX_LIG_ATOMS = 64;

    // Shared memory for ligand atoms (per warp would be ideal, but shared per block)
    __shared__ float3 sLigPos[MAX_LIG_ATOMS];
    __shared__ float sLigR_off[MAX_LIG_ATOMS];
    __shared__ float sLigS[MAX_LIG_ATOMS];
    __shared__ int sLigParticleIdx[MAX_LIG_ATOMS];
    __shared__ int sGroupIdx;
    __shared__ int sGroupStart;
    __shared__ int sGroupSize;

    int totalWork = numGroups * numReceptorAtoms;
    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (globalIdx >= totalWork) return;

    int groupIdx = globalIdx / numReceptorAtoms;
    int recIdx = globalIdx % numReceptorAtoms;

    // First thread in block loads the group info and ligand data into shared memory
    // All threads in a block may span multiple groups, so we need per-thread group info
    // However, for efficiency let's check if the whole block is in one group
    int groupIdxFirst = (blockIdx.x * blockDim.x) / numReceptorAtoms;
    int groupIdxLast = ((blockIdx.x + 1) * blockDim.x - 1) / numReceptorAtoms;
    bool singleGroup = (groupIdxFirst == groupIdxLast);

    // Load ligand data into shared memory (cooperative load)
    if (singleGroup && threadIdx.x == 0) {
        sGroupIdx = groupIdx;
        sGroupStart = groupStart[groupIdx];
        sGroupSize = groupStart[groupIdx + 1] - groupStart[groupIdx];
    }
    __syncthreads();

    int ligGroupStart, ligGroupSize;
    if (singleGroup) {
        ligGroupStart = sGroupStart;
        ligGroupSize = sGroupSize;
        // Cooperative load of ligand atoms
        for (int i = threadIdx.x; i < ligGroupSize && i < MAX_LIG_ATOMS; i += blockDim.x) {
            int ligGlobalIdx = ligGroupStart + i;
            int particleIdx = particleIndices[ligGlobalIdx];
            int templateIdx = (i % templateNumAtoms);
            float4 p = posq[particleIdx];
            sLigPos[i] = make_float3(p.x, p.y, p.z);
            float R = ligandRadii[templateIdx];
            sLigR_off[i] = R - DIELECTRIC_OFFSET;
            sLigS[i] = (R - DIELECTRIC_OFFSET) * ligandScaleFactors[templateIdx];
            sLigParticleIdx[i] = ligGlobalIdx;
        }
        __syncthreads();
    } else {
        // Block spans multiple groups — fall back to per-thread group lookup
        ligGroupStart = groupStart[groupIdx];
        ligGroupSize = groupStart[groupIdx + 1] - ligGroupStart;
    }

    // Load this thread's receptor atom
    float3 recPos = receptorPositions[recIdx];
    float recR = receptorRadii[recIdx];
    float recR_off = recR - DIELECTRIC_OFFSET;
    float recS = recR_off * receptorScaleFactors[recIdx];

    float cutoff2 = cutoffDistance * cutoffDistance;
    bool useCutoff = (cutoffDistance > 0.0f);

    // Accumulate ligand→receptor HCT (this receptor atom screened by all ligand atoms)
    float hctLigToRec = 0.0f;

    // Loop over ligand atoms
    int nLig = (ligGroupSize < MAX_LIG_ATOMS) ? ligGroupSize : MAX_LIG_ATOMS;
    for (int li = 0; li < nLig; li++) {
        float3 ligPos;
        float ligR_off, ligS;
        int ligGlobalIdx;

        if (singleGroup) {
            ligPos = sLigPos[li];
            ligR_off = sLigR_off[li];
            ligS = sLigS[li];
            ligGlobalIdx = sLigParticleIdx[li];
        } else {
            int idx = ligGroupStart + li;
            int particleIdx = particleIndices[idx];
            int templateIdx = li % templateNumAtoms;
            float4 p = posq[particleIdx];
            ligPos = make_float3(p.x, p.y, p.z);
            float R = ligandRadii[templateIdx];
            ligR_off = R - DIELECTRIC_OFFSET;
            ligS = (R - DIELECTRIC_OFFSET) * ligandScaleFactors[templateIdx];
            ligGlobalIdx = idx;
        }

        float dx = recPos.x - ligPos.x;
        float dy = recPos.y - ligPos.y;
        float dz = recPos.z - ligPos.z;
        float r2 = dx*dx + dy*dy + dz*dz;

        if (useCutoff && r2 > cutoff2) continue;

        float invR = rsqrtf(r2);
        float r = r2 * invR;
        if (r < 1e-6f) continue;

        // --- Ligand→Receptor HCT (ligand screens this receptor atom) ---
        float r_plus_Si = r + ligS;
        if (recR_off < r_plus_Si) {
            float r_minus_Si = fabsf(r - ligS);
            float l = (recR_off > r_minus_Si) ? (1.0f / recR_off) : (1.0f / r_minus_Si);
            float u = 1.0f / r_plus_Si;
            float l2 = l*l, u2 = u*u;
            float r_inv = 1.0f / r;
            float term = l - u + 0.25f*r*(u2-l2) + 0.5f*r_inv*logf(u/l) + 0.25f*ligS*ligS*r_inv*(l2-u2);
            if (recR_off < (ligS - r)) term += 2.0f*(1.0f/recR_off - l);
            hctLigToRec += term;
        }

        // --- Receptor→Ligand HCT (this receptor screens ligand atom) ---
        float r_plus_Sj = r + recS;
        if (ligR_off < r_plus_Sj) {
            float r_minus_Sj = fabsf(r - recS);
            float l = (ligR_off > r_minus_Sj) ? (1.0f / ligR_off) : (1.0f / r_minus_Sj);
            float u = 1.0f / r_plus_Sj;
            float l2 = l*l, u2 = u*u;
            float r_inv = 1.0f / r;
            float term = l - u + 0.25f*r*(u2-l2) + 0.5f*r_inv*logf(u/l) + 0.25f*recS*recS*r_inv*(l2-u2);
            if (ligR_off < (recS - r)) term += 2.0f*(1.0f/ligR_off - l);

            // Accumulate into ligand atom's HCT via atomicAdd
            atomicAdd(&hctReceptor[ligGlobalIdx], term);
        }
    }

    // Direct write: ligand→receptor HCT for this receptor atom
    ligandToReceptorHCT[groupIdx * numReceptorAtoms + recIdx] = hctLigToRec;
}

/**
 * Fused receptor↔ligand HCT kernel using warp shuffle.
 *
 * Computes BOTH receptor→ligand HCT (for ligand Born radii) AND
 * ligand→receptor HCT (for receptor desolvation) in a single kernel,
 * processing each receptor-ligand pair only once.
 *
 * Architecture:
 * - Each warp processes TILE_SIZE receptor atoms at a time
 * - Each thread in the warp holds one ligand atom's data in registers
 * - Receptor data rotates through the warp via __shfl_sync
 * - Both HCT sums are accumulated: ligand-side in registers, receptor-side
 *   via atomicAdd to global memory
 *
 * Output:
 *   hctReceptor[ligIdx] += receptor→ligand HCT
 *   ligandToReceptorHCT[groupIdx * numReceptorAtoms + recIdx] += ligand→receptor HCT
 */
extern "C" __global__ void computeFusedReceptorLigandHCT(
    const float4* __restrict__ posq,
    const int* __restrict__ particleIndices,
    const float* __restrict__ ligandRadii,
    const float* __restrict__ ligandScaleFactors,
    const float3* __restrict__ receptorPositions,
    const float* __restrict__ receptorRadii,
    const float* __restrict__ receptorScaleFactors,
    int numReceptorAtoms,
    const int* __restrict__ groupStart,
    int numGroups,
    int totalParticles,
    int templateNumAtoms,
    float cutoffDistance,
    float* __restrict__ hctReceptor,
    float* __restrict__ ligandToReceptorHCT,
    const int* __restrict__ isActiveRecAtom,
    int hasBaseline
) {
    const int tgx = threadIdx.x & (TILE_SIZE - 1);
    const int warpIdx = (blockIdx.x * blockDim.x + threadIdx.x) / TILE_SIZE;
    const int totalWarps = (gridDim.x * blockDim.x) / TILE_SIZE;

    // Each warp handles a batch of ligand atoms
    // Distribute ligand atoms across warps via grid-stride
    for (int ligBase = warpIdx * TILE_SIZE; ligBase < totalParticles; ligBase += totalWarps * TILE_SIZE) {
        int ligIdx = ligBase + tgx;
        bool validLig = (ligIdx < totalParticles);

        // Load ligand atom data into registers
        float4 ligPos = make_float4(0, 0, 0, 0);
        float ligR_off = 0.1f;
        float ligS = 0.1f;
        int myGroupIdx = 0;
        int templateIdx = 0;

        if (validLig) {
            int particleIdx = particleIndices[ligIdx];
            ligPos = posq[particleIdx];

            int atomInGroup = ligIdx;
            for (int g = 0; g < numGroups; g++) {
                int gs = groupStart[g];
                int ge = groupStart[g + 1];
                if (ligIdx >= gs && ligIdx < ge) {
                    atomInGroup = ligIdx - gs;
                    myGroupIdx = g;
                    break;
                }
            }
            templateIdx = atomInGroup % templateNumAtoms;
            float R_i = ligandRadii[templateIdx];
            ligR_off = R_i - DIELECTRIC_OFFSET;
            ligS = ligR_off * ligandScaleFactors[templateIdx];
        }

        float hctLigAccum = 0.0f;  // receptor→ligand HCT for this ligand atom

        float cutoff2 = cutoffDistance * cutoffDistance;
        bool useCutoff = (cutoffDistance > 0.0f);

        // Iterate over receptor atoms in tiles of TILE_SIZE
        int numRecTiles = (numReceptorAtoms + TILE_SIZE - 1) / TILE_SIZE;
        for (int recTile = 0; recTile < numRecTiles; recTile++) {
            int recIdx = recTile * TILE_SIZE + tgx;

            // Load receptor atom data into registers (one per thread)
            float3 recPos = make_float3(0, 0, 0);
            float recR_off = 0.1f;
            float recS = 0.1f;
            bool validRec = (recIdx < numReceptorAtoms);

            if (validRec) {
                recPos = receptorPositions[recIdx];
                float R_j = receptorRadii[recIdx];
                recR_off = R_j - DIELECTRIC_OFFSET;
                recS = recR_off * receptorScaleFactors[recIdx];
            }

            // Rotate receptor data through the warp via shuffle.
            // Each rotation presents one receptor atom to all 32 ligand threads.
            for (int rot = 0; rot < TILE_SIZE; rot++) {
                float3 rPos;
                rPos.x = __shfl_sync(0xFFFFFFFF, recPos.x, rot);
                rPos.y = __shfl_sync(0xFFFFFFFF, recPos.y, rot);
                rPos.z = __shfl_sync(0xFFFFFFFF, recPos.z, rot);
                float rR_off = __shfl_sync(0xFFFFFFFF, recR_off, rot);
                float rS = __shfl_sync(0xFFFFFFFF, recS, rot);
                int rValid = __shfl_sync(0xFFFFFFFF, (int)validRec, rot);
                int rIdx = recTile * TILE_SIZE + rot;

                if (!validLig || !rValid) continue;

                if (hasBaseline && isActiveRecAtom != 0 &&
                    !isActiveRecAtom[myGroupIdx * numReceptorAtoms + rIdx])
                    continue;

                float dx = ligPos.x - rPos.x;
                float dy = ligPos.y - rPos.y;
                float dz = ligPos.z - rPos.z;
                float r2 = dx*dx + dy*dy + dz*dz;

                if (useCutoff && r2 > cutoff2) continue;

                float r = sqrtf(r2);
                if (r < 1e-6f) continue;

                // Receptor→Ligand HCT (receptor j screens ligand i)
                float r_plus_Sj = r + rS;
                if (ligR_off < r_plus_Sj) {
                    float r_minus_Sj = fabsf(r - rS);
                    float l_ij = (ligR_off > r_minus_Sj) ? (1.0f / ligR_off) : (1.0f / r_minus_Sj);
                    float u_ij = 1.0f / r_plus_Sj;
                    float l2 = l_ij * l_ij;
                    float u2 = u_ij * u_ij;
                    float r_inv = 1.0f / r;
                    float term = l_ij - u_ij + 0.25f*r*(u2-l2) + 0.5f*r_inv*logf(u_ij/l_ij) + 0.25f*rS*rS*r_inv*(l2-u2);
                    if (ligR_off < (rS - r)) term += 2.0f * (1.0f/ligR_off - l_ij);
                    hctLigAccum += term;
                }

                // Ligand→Receptor HCT (ligand i screens receptor j)
                // Each ligand thread computes its contribution to receptor rIdx.
                // Sum across the warp and write once via lane 0.
                float recTerm = 0.0f;
                float r_plus_Si = r + ligS;
                if (rR_off < r_plus_Si) {
                    float r_minus_Si = fabsf(r - ligS);
                    float l_ji = (rR_off > r_minus_Si) ? (1.0f / rR_off) : (1.0f / r_minus_Si);
                    float u_ji = 1.0f / r_plus_Si;
                    float l2 = l_ji * l_ji;
                    float u2 = u_ji * u_ji;
                    float r_inv = 1.0f / r;
                    recTerm = l_ji - u_ji + 0.25f*r*(u2-l2) + 0.5f*r_inv*logf(u_ji/l_ji) + 0.25f*ligS*ligS*r_inv*(l2-u2);
                    if (rR_off < (ligS - r)) recTerm += 2.0f * (1.0f/rR_off - l_ji);
                }

                // Each thread atomicAdds its ligand→receptor contribution to its own
                // group's slot. Can't warp-reduce because threads may be in different groups.
                if (recTerm != 0.0f) {
                    atomicAdd(&ligandToReceptorHCT[myGroupIdx * numReceptorAtoms + rIdx], recTerm);
                }
            }
        }

        // Write receptor→ligand HCT
        if (validLig) {
            hctReceptor[ligIdx] = hctLigAccum;
        }
    }
}

/**
 * Receptor→ligand HCT via cell list.
 *
 * Each ligand thread looks up which cells are within the cutoff and only
 * iterates receptor atoms in those cells. Cost: O(atoms_in_neighborhood)
 * instead of O(N_rec).
 */
extern "C" __global__ void computeIsolatedReceptorHCTCellList(
    const float4* __restrict__ posq,
    const int* __restrict__ particleIndices,
    const float* __restrict__ radii,
    const float3* __restrict__ receptorPositions,
    const float* __restrict__ receptorRadii,
    const float* __restrict__ receptorScaleFactors,
    const int* __restrict__ cellAtomIdx,
    const int* __restrict__ cellStartArr,
    int numReceptorAtoms,
    int cellNx, int cellNy, int cellNz,
    float cellOriginX, float cellOriginY, float cellOriginZ,
    float cellSz,
    const int* __restrict__ groupStart,
    int numGroups,
    int totalParticles,
    int templateNumAtoms,
    float cutoffDistance,
    float* __restrict__ hctReceptor
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= totalParticles) return;

    int atomInGroup = idx;
    for (int g = 0; g < numGroups; g++) {
        int gs = groupStart[g];
        int ge = groupStart[g + 1];
        if (idx >= gs && idx < ge) {
            atomInGroup = idx - gs;
            break;
        }
    }
    int templateIdx = atomInGroup % templateNumAtoms;

    int particleIdx = particleIndices[idx];
    float4 pos = posq[particleIdx];
    float R_i = radii[templateIdx];
    float R_i_off = R_i - DIELECTRIC_OFFSET;

    float hct = 0.0f;
    float cutoff2 = cutoffDistance * cutoffDistance;
    bool useCutoff = (cutoffDistance > 0.0f);

    // Determine which cell this ligand atom is in
    int cx0 = (int)((pos.x - cellOriginX) / cellSz);
    int cy0 = (int)((pos.y - cellOriginY) / cellSz);
    int cz0 = (int)((pos.z - cellOriginZ) / cellSz);

    // Iterate over 27 neighboring cells (3x3x3)
    for (int dcx = -1; dcx <= 1; dcx++) {
        int cx = cx0 + dcx;
        if (cx < 0 || cx >= cellNx) continue;
        for (int dcy = -1; dcy <= 1; dcy++) {
            int cy = cy0 + dcy;
            if (cy < 0 || cy >= cellNy) continue;
            for (int dcz = -1; dcz <= 1; dcz++) {
                int cz = cz0 + dcz;
                if (cz < 0 || cz >= cellNz) continue;

                int cell = cx * cellNy * cellNz + cy * cellNz + cz;
                int start = cellStartArr[cell];
                int end = cellStartArr[cell + 1];

                for (int k = start; k < end; k++) {
                    int j = cellAtomIdx[k];

                    float3 pos_rec = receptorPositions[j];
                    float dx = pos.x - pos_rec.x;
                    float dy = pos.y - pos_rec.y;
                    float dz = pos.z - pos_rec.z;
                    float r2 = dx*dx + dy*dy + dz*dz;

                    if (useCutoff && r2 > cutoff2) continue;

                    float r = sqrtf(r2);
                    if (r < 1e-6f) continue;

                    float R_j = receptorRadii[j];
                    float R_j_off = R_j - DIELECTRIC_OFFSET;
                    float S_j = R_j_off * receptorScaleFactors[j];

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
            }
        }
    }

    hctReceptor[idx] = hct;
}

/**
 * Fast HCT reconstruction via cell list + baseline sum.
 *
 * Starts with precomputed baseline sum, then for active atoms in nearby
 * cells: subtract baseline, add fresh computation.
 * Cost: O(atoms_in_neighborhood) instead of O(N_rec).
 */
extern "C" __global__ void reconstructReceptorHCTCellList(
    const float4* __restrict__ posq,
    const int* __restrict__ particleIndices,
    const float* __restrict__ radii,
    const float3* __restrict__ receptorPositions,
    const float* __restrict__ receptorRadii,
    const float* __restrict__ receptorScaleFactors,
    const float* __restrict__ hctPerAtomBaseline,
    const float* __restrict__ baselineSum,
    const int* __restrict__ isActiveRecAtom,
    const int* __restrict__ cellAtomIdx,
    const int* __restrict__ cellStartArr,
    int numReceptorAtoms,
    int cellNx, int cellNy, int cellNz,
    float cellOriginX, float cellOriginY, float cellOriginZ,
    float cellSz,
    float effectiveCutoff,
    const int* __restrict__ groupStart,
    int numGroups,
    int totalParticles,
    int templateNumAtoms,
    float cutoffDistance,
    float* __restrict__ hctReceptor
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= totalParticles) return;

    int atomInGroup = idx;
    int myGroupIdx = 0;
    for (int g = 0; g < numGroups; g++) {
        int gs = groupStart[g];
        int ge = groupStart[g + 1];
        if (idx >= gs && idx < ge) {
            atomInGroup = idx - gs;
            myGroupIdx = g;
            break;
        }
    }
    int templateIdx = atomInGroup % templateNumAtoms;
    const int* activeMask = isActiveRecAtom + myGroupIdx * numReceptorAtoms;
    const float* myBaseline = hctPerAtomBaseline + idx * numReceptorAtoms;

    int particleIdx = particleIndices[idx];
    float4 pos = posq[particleIdx];
    float R_i = radii[templateIdx];
    float R_i_off = R_i - DIELECTRIC_OFFSET;

    // Start with precomputed baseline sum
    float hct = baselineSum[idx];

    float cutoff2 = cutoffDistance * cutoffDistance;
    bool useCutoff = (cutoffDistance > 0.0f);
    float effCutoff2 = effectiveCutoff * effectiveCutoff;

    // Determine cell range based on effective cutoff
    int cx0 = (int)((pos.x - cellOriginX) / cellSz);
    int cy0 = (int)((pos.y - cellOriginY) / cellSz);
    int cz0 = (int)((pos.z - cellOriginZ) / cellSz);

    // Number of cells to search in each direction (based on effective cutoff)
    int cellRange = (int)ceilf(effectiveCutoff / cellSz);

    for (int dcx = -cellRange; dcx <= cellRange; dcx++) {
        int cx = cx0 + dcx;
        if (cx < 0 || cx >= cellNx) continue;
        for (int dcy = -cellRange; dcy <= cellRange; dcy++) {
            int cy = cy0 + dcy;
            if (cy < 0 || cy >= cellNy) continue;
            for (int dcz = -cellRange; dcz <= cellRange; dcz++) {
                int cz = cz0 + dcz;
                if (cz < 0 || cz >= cellNz) continue;

                int cell = cx * cellNy * cellNz + cy * cellNz + cz;
                int start = cellStartArr[cell];
                int end = cellStartArr[cell + 1];

                for (int k = start; k < end; k++) {
                    int j = cellAtomIdx[k];

                    if (!activeMask[j]) continue;

                    // Subtract baseline for this atom
                    hct -= myBaseline[j];

                    // Compute fresh
                    float3 pos_rec = receptorPositions[j];
                    float dx = pos.x - pos_rec.x;
                    float dy = pos.y - pos_rec.y;
                    float dz = pos.z - pos_rec.z;
                    float r2 = dx*dx + dy*dy + dz*dz;

                    if (useCutoff && r2 > cutoff2) continue;

                    float r = sqrtf(r2);
                    if (r < 1e-6f) continue;

                    float R_j = receptorRadii[j];
                    float R_j_off = R_j - DIELECTRIC_OFFSET;
                    float S_j = R_j_off * receptorScaleFactors[j];

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
            }
        }
    }

    hctReceptor[idx] = hct;
}

/**
 * Tiled receptor→ligand HCT kernel.
 *
 * Each thread handles one ligand atom. Receptor atoms are loaded in
 * TILE_SIZE=32 tiles into shared memory, shared across the warp.
 * This improves arithmetic intensity from ~5 to ~80 FLOP/byte.
 *
 * For per-group locality: uses isActiveRecAtom[myGroupIdx * numReceptorAtoms + j]
 * to skip inactive receptor atoms (they use the frozen HCT baseline via
 * reconstructReceptorHCT instead).
 *
 * When hasBaseline=false (first call), computes the full HCT for all receptor
 * atoms. When hasBaseline=true, only computes for active atoms (inactive
 * contributions come from the cached baseline in reconstructReceptorHCT).
 */
extern "C" __global__ void computeIsolatedReceptorHCTPairwiseTiled(
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
    float* __restrict__ hctReceptor,
    const int* __restrict__ isActiveRecAtom,
    int hasBaseline
) {
    // Shared memory for receptor tile data
    __shared__ float3 tile_pos[TILE_SIZE * 8];  // 8 warps per block max
    __shared__ float tile_scaledR[TILE_SIZE * 8];

    const int tgx = threadIdx.x & (TILE_SIZE - 1);  // Thread index within warp
    const int warpInBlock = threadIdx.x / TILE_SIZE;
    const int tileBase = warpInBlock * TILE_SIZE;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= totalParticles) return;

    // Determine group and template index
    int atomInGroup = idx;
    int myGroupIdx = 0;
    for (int g = 0; g < numGroups; g++) {
        int gs = groupStart[g];
        int ge = groupStart[g + 1];
        if (idx >= gs && idx < ge) {
            atomInGroup = idx - gs;
            myGroupIdx = g;
            break;
        }
    }
    int templateIdx = atomInGroup % templateNumAtoms;

    int particleIdx = particleIndices[idx];
    float4 pos = posq[particleIdx];
    float R_i = radii[templateIdx];
    float R_i_off = R_i - DIELECTRIC_OFFSET;

    float hct = 0.0f;
    float cutoff2 = cutoffDistance * cutoffDistance;
    bool useCutoff = (cutoffDistance > 0.0f);

    // Per-group active mask offset (null if no locality)
    const int* activeMask = (isActiveRecAtom != 0) ?
        isActiveRecAtom + myGroupIdx * numReceptorAtoms : 0;

    // Iterate over receptor tiles
    int numRecTiles = (numReceptorAtoms + TILE_SIZE - 1) / TILE_SIZE;
    for (int tile = 0; tile < numRecTiles; tile++) {
        int recBase = tile * TILE_SIZE;

        // Cooperative load: each thread in the warp loads one receptor atom
        int loadIdx = recBase + tgx;
        if (loadIdx < numReceptorAtoms) {
            float3 rp = receptorPositions[loadIdx];
            tile_pos[tileBase + tgx] = rp;
            float R_j = receptorRadii[loadIdx];
            float R_j_off = R_j - DIELECTRIC_OFFSET;
            tile_scaledR[tileBase + tgx] = R_j_off * receptorScaleFactors[loadIdx];
        } else {
            tile_pos[tileBase + tgx] = make_float3(0, 0, 0);
            tile_scaledR[tileBase + tgx] = 0.1f;
        }
        __syncwarp();

        // Each thread processes all TILE_SIZE receptor atoms from shared memory
        for (int t = 0; t < TILE_SIZE; t++) {
            int j = recBase + t;
            if (j >= numReceptorAtoms) break;

            // Skip inactive atoms when baseline is available
            if (hasBaseline && activeMask != 0 && !activeMask[j]) continue;

            float3 pos_rec = tile_pos[tileBase + t];
            float dx = pos.x - pos_rec.x;
            float dy = pos.y - pos_rec.y;
            float dz = pos.z - pos_rec.z;
            float r2 = dx*dx + dy*dy + dz*dz;

            if (useCutoff && r2 > cutoff2) continue;

            float r = sqrtf(r2);
            if (r < 1e-6f) continue;

            float S_j = tile_scaledR[tileBase + t];
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
        __syncwarp();
    }

    hctReceptor[idx] = hct;
}

/**
 * Compute HCT contribution from receptor via pairwise interactions (naive version).
 * If isActiveRecAtom is non-null, only active receptor atoms contribute.
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
    float* __restrict__ hctReceptor,
    const int* __restrict__ isActiveRecAtom
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= totalParticles) return;

    // Determine template index and group
    int atomInGroup = idx;
    int myGroupIdx = 0;
    for (int g = 0; g < numGroups; g++) {
        int groupStartIdx = groupStart[g];
        int groupEndIdx = groupStart[g + 1];
        if (idx >= groupStartIdx && idx < groupEndIdx) {
            atomInGroup = idx - groupStartIdx;
            myGroupIdx = g;
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

    // Loop over receptor atoms (skip inactive for this group if locality mask provided)
    for (int j = 0; j < numReceptorAtoms; j++) {
        if (isActiveRecAtom != 0 && !isActiveRecAtom[myGroupIdx * numReceptorAtoms + j]) continue;

        float3 pos_j = receptorPositions[j];

        float dx = pos_i.x - pos_j.x;
        float dy = pos_i.y - pos_j.y;
        float dz = pos_i.z - pos_j.z;
        float r2 = dx*dx + dy*dy + dz*dz;

        if (useCutoff && r2 > cutoff2) continue;

        float invR = rsqrtf(r2);
        float r = r2 * invR;
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
    int paddedNumAtoms,
    float globalScalingFactor,
    const float* __restrict__ groupScalingFactors,
    float* __restrict__ groupUnscaledEnergies
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

    // Compute alchemical scaling for this group
    float scale = globalScalingFactor * groupScalingFactors[groupIdx];

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

        // Force = -dE/dr scaled by alchemical factor
        float dFgbDr = (r * invFgb) * (1.0f - 0.25f * expTerm);
        float dEdR = -prefactor * q_i * q_j * invFgb * invFgb * dFgbDr * scale;

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

    // Accumulate scaled energies
    atomicAdd(&groupEnergies[groupIdx], energy * scale);
    atomicAdd(&groupLigandSelfEnergies[groupIdx], energy * scale);
    // Accumulate unscaled energies (no per-group alchemical scaling)
    if (groupUnscaledEnergies != 0) {
        float unscaledScale = globalScalingFactor;  // only global, no group scaling
        atomicAdd(&groupUnscaledEnergies[groupIdx], energy * unscaledScale);
    }
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
    float* __restrict__ groupEnergies,
    float globalScalingFactor,
    const float* __restrict__ groupScalingFactors,
    float* __restrict__ groupUnscaledEnergies
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

    // ACE surface area term with alchemical scaling
    float scale = globalScalingFactor * groupScalingFactors[groupIdx];
    float Rsolv = R_i + probeRadius;
    float ratio = R_i / bornR;
    float ratio6 = ratio * ratio * ratio * ratio * ratio * ratio;
    float area = 4.0f * 3.14159265f * Rsolv * Rsolv * ratio6;
    float saEnergy = surfaceTension * area * scale;

    atomicAdd(&groupEnergies[groupIdx], saEnergy);
    // Accumulate unscaled SA energy (no per-group alchemical scaling)
    if (groupUnscaledEnergies != 0) {
        float saEnergyUnscaled = surfaceTension * area * globalScalingFactor;
        atomicAdd(&groupUnscaledEnergies[groupIdx], saEnergyUnscaled);
    }
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
    float* __restrict__ dE_dR,
    float globalScalingFactor,
    const float* __restrict__ groupScalingFactors
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

    // Alchemical scaling for this group
    float scale = globalScalingFactor * groupScalingFactors[groupIdx];

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

    dE_dR[idx] = dEdRi * scale;
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
    float* __restrict__ dE_dR,
    float globalScalingFactor,
    const float* __restrict__ groupScalingFactors
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

    float scale = globalScalingFactor * groupScalingFactors[groupIdx];
    float dEdR_SA = -6.0f * surfaceTension * 4.0f * 3.14159265f * Rsolv * Rsolv * ratio5 * R_i / (bornR * bornR);

    dE_dR[idx] += dEdR_SA * scale;
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

        float invR = rsqrtf(r2);
        float r = r2 * invR;
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
    bool useKDECorrections,
    bool hasBinnedKDEDerivatives,
    float* __restrict__ hctReceptor
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= totalParticles) return;

    // Determine which group this atom belongs to and its position within the group
    int atomInGroup = idx;
    for (int g = 0; g < numGroups; g++) {
        int groupStartIdx = groupStart[g];
        int groupEndIdx = groupStart[g + 1];
        if (idx >= groupStartIdx && idx < groupEndIdx) {
            atomInGroup = idx - groupStartIdx;
            break;
        }
    }

    // Get particle index and template atom index
    int particleIdx = particleIndices[idx];
    int templateIdx = atomInGroup % templateNumAtoms;

    // Get position
    float4 pos = posq[particleIdx];
    float3 position = make_float3(pos.x, pos.y, pos.z);

    // Get radius and compute offset radius
    float R_i = radii[templateIdx];
    float R_i_off = R_i - DIELECTRIC_OFFSET;
    float R_probe_off = probeRadius - DIELECTRIC_OFFSET;

    // Grid dimensions
    int nx = gridCounts[0];
    int ny = gridCounts[1];
    int nz = gridCounts[2];
    int numPoints = nx * ny * nz;

    // Find appropriate bin for this radius
    int binIdx = numBins - 1;
    for (int b = 0; b < numBins; b++) {
        if (rThresholds[b] >= R_i_off) {
            binIdx = b;
            break;
        }
    }
    int binOffset = binIdx * numPoints;

    // Use the shared interpolation helper (defined in gbsaGridForce.cu,
    // available because all .cu files are compiled into one module)
    GBSAInterpolationResult result = interpolateGBSAGrids(
        position, R_i_off, R_probe_off,
        gridCounts, gridSpacing,
        originX, originY, originZ,
        gridHctProbe, gridHctDerivatives,
        gridCorrectionN, gridCorrectionA, gridCorrectionB,
        binOffset, interpolationMethod, false,
        useKDECorrections, hasBinnedKDEDerivatives
    );

    hctReceptor[idx] = result.isInside ? result.hct : 0.0f;
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
    bool useKDECorrections,
    bool hasBinnedKDEDerivatives,
    unsigned long long* __restrict__ forceBuffer,
    int paddedNumAtoms
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= totalParticles) return;

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
    float3 position = make_float3(pos.x, pos.y, pos.z);

    float R_i = radii[templateIdx];
    float R_i_off = R_i - DIELECTRIC_OFFSET;
    float R_probe_off = probeRadius - DIELECTRIC_OFFSET;

    int numPoints = gridCounts[0] * gridCounts[1] * gridCounts[2];
    int binIdx = numBins - 1;
    for (int b = 0; b < numBins; b++) {
        if (rThresholds[b] >= R_i_off) {
            binIdx = b;
            break;
        }
    }
    int binOffset = binIdx * numPoints;

    GBSAInterpolationResult result = interpolateGBSAGrids(
        position, R_i_off, R_probe_off,
        gridCounts, gridSpacing,
        originX, originY, originZ,
        gridHctProbe, gridHctDerivatives,
        gridCorrectionN, gridCorrectionA, gridCorrectionB,
        binOffset, interpolationMethod, true,
        useKDECorrections, hasBinnedKDEDerivatives
    );

    if (!result.isInside) return;

    float hctTotal = hctReceptor[idx] + hctLigand[idx];
    float psi = 0.5f * R_i_off * hctTotal;
    float psi2 = psi * psi;
    float psi3 = psi2 * psi;
    float tanhArg = OBC_ALPHA * psi - OBC_BETA * psi2 + OBC_GAMMA * psi3;
    float tanhVal = tanhf(tanhArg);
    float sech2 = 1.0f - tanhVal * tanhVal;
    float dTanhArgDPsi = OBC_ALPHA - 2.0f * OBC_BETA * psi + 3.0f * OBC_GAMMA * psi2;
    float obcChain = R_i_off * dTanhArgDPsi * sech2 / R_i;

    float bornR = bornRadii[idx];
    float bornForces = dE_dR[idx] * bornR * bornR * obcChain;

    float fx = -0.5f * bornForces * result.gradient.x;
    float fy = -0.5f * bornForces * result.gradient.y;
    float fz = -0.5f * bornForces * result.gradient.z;

    atomicAdd(&forceBuffer[particleIdx],
              static_cast<unsigned long long>((long long)(fx * 0x100000000)));
    atomicAdd(&forceBuffer[particleIdx + paddedNumAtoms],
              static_cast<unsigned long long>((long long)(fy * 0x100000000)));
    atomicAdd(&forceBuffer[particleIdx + 2*paddedNumAtoms],
              static_cast<unsigned long long>((long long)(fz * 0x100000000)));
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

        float invR = rsqrtf(r2);
        float r = r2 * invR;
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
 *
 * If isActiveRecAtom is non-null, only active receptor atoms are computed.
 * Inactive atoms get ligandToReceptorHCT = 0.
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
    float* __restrict__ ligandToReceptorHCT,
    const int* __restrict__ isActiveRecAtom
) {
    // Grid-stride loop: each thread handles multiple (group, receptor_atom) pairs.
    int totalWork = numGroups * numReceptorAtoms;
    for (int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
         globalIdx < totalWork;
         globalIdx += gridDim.x * blockDim.x) {

        int groupIdx = globalIdx / numReceptorAtoms;
        int recIdx = globalIdx % numReceptorAtoms;

        // Skip inactive receptor atoms for this group if locality mask is provided
        if (isActiveRecAtom != 0 && !isActiveRecAtom[groupIdx * numReceptorAtoms + recIdx]) {
            ligandToReceptorHCT[groupIdx * numReceptorAtoms + recIdx] = 0.0f;
            continue;
        }

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
}

/**
 * Compute receptor Born radii with ligand screening.
 * receptorHCT = receptorSelfHCT + ligandToReceptorHCT
 *
 * Batched: processes all groups in a single launch.
 * If isActiveRecAtom is non-null, inactive atoms copy from receptorBornRadiiRef.
 * Output: receptorBornRadii[groupIdx * numReceptorAtoms + i]
 */
extern "C" __global__ void computeReceptorBornRadiiWithLigand(
    const float* __restrict__ receptorRadii,
    const float* __restrict__ receptorSelfHCT,
    const float* __restrict__ ligandToReceptorHCT,
    int numReceptorAtoms,
    int numGroups,
    float* __restrict__ receptorBornRadii,
    const float* __restrict__ receptorBornRadiiRef,
    const int* __restrict__ isActiveRecAtom,
    float globalScalingFactor,
    const float* __restrict__ groupScalingFactors
) {
    int totalWork = numGroups * numReceptorAtoms;
    for (int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
         globalIdx < totalWork;
         globalIdx += gridDim.x * blockDim.x) {

        int groupIdx = globalIdx / numReceptorAtoms;
        int i = globalIdx % numReceptorAtoms;

        // Zero-scaled groups: all receptor atoms keep reference Born radii
        float scale = globalScalingFactor * groupScalingFactors[groupIdx];
        if (scale < 0.05f) {
            receptorBornRadii[groupIdx * numReceptorAtoms + i] = receptorBornRadiiRef[i];
            continue;
        }

        // Inactive atoms keep reference Born radii
        if (isActiveRecAtom != 0 && !isActiveRecAtom[groupIdx * numReceptorAtoms + i]) {
            receptorBornRadii[groupIdx * numReceptorAtoms + i] = receptorBornRadiiRef[i];
            continue;
        }

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

        receptorBornRadii[groupIdx * numReceptorAtoms + i] = bornRadius;
    }
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
    int paddedNumAtoms,
    float globalScalingFactor,
    const float* __restrict__ groupScalingFactors,
    const int* __restrict__ isActiveRecAtom
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

    // Alchemical scaling for this group
    float scale = globalScalingFactor * groupScalingFactors[groupIdx];

    int particleIdx_lig = particleIndices[idx];
    int templateIdx_lig = atomInGroup % templateNumAtoms;

    float4 pos_lig = posq[particleIdx_lig];
    float q_lig = ligandCharges[templateIdx_lig];
    float R_lig = ligandBornRadii[idx];

    float energy = 0.0f;
    float3 force_lig = make_float3(0.0f, 0.0f, 0.0f);

    // Loop over all receptor atoms (cross-term is Coulomb-like — not pruned)
    // isActiveRecAtom parameter kept in signature for interface consistency but ignored
    // Minimum distance floor to prevent singularity when ligand overlaps receptor
    const float MIN_CROSS_R2 = 0.01f;  // 0.1 nm = 1 Angstrom
    for (int j = 0; j < numReceptorAtoms; j++) {
        float3 pos_rec = receptorPositions[j];
        float q_rec = receptorCharges[j];
        float R_rec = receptorBornRadii[groupIdx * numReceptorAtoms + j];

        float dx = pos_rec.x - pos_lig.x;
        float dy = pos_rec.y - pos_lig.y;
        float dz = pos_rec.z - pos_lig.z;
        float r2 = dx*dx + dy*dy + dz*dz;
        if (r2 < MIN_CROSS_R2) continue;
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

        // Force on ligand (scaled by alchemical factor)
        float dFgbDr = (r * invFgb) * (1.0f - 0.25f * expTerm);
        float dEdR = -prefactor * q_lig * q_rec * invFgb * invFgb * dFgbDr * scale;

        float invR = 1.0f / r;
        force_lig.x += dEdR * dx * invR;
        force_lig.y += dEdR * dy * invR;
        force_lig.z += dEdR * dz * invR;
    }

    // Accumulate scaled cross-term energy
    atomicAdd(&crossTermEnergies[groupIdx], energy * scale);

    // Accumulate forces
    atomicAdd(&forceBuffer[particleIdx_lig], static_cast<unsigned long long>((long long)(force_lig.x * 0x100000000)));
    atomicAdd(&forceBuffer[particleIdx_lig + paddedNumAtoms], static_cast<unsigned long long>((long long)(force_lig.y * 0x100000000)));
    atomicAdd(&forceBuffer[particleIdx_lig + 2*paddedNumAtoms], static_cast<unsigned long long>((long long)(force_lig.z * 0x100000000)));
}

/**
 * Accumulate dE_cross/dR_born_lig into the Born radii chain-rule buffer.
 *
 * For each ligand atom i, computes:
 *   dE_cross/dR_i = prefactor * q_i * Σ_j q_j * dGpol_dalpha2 * R_rec_j
 * where dGpol_dalpha2 = -0.5 * Gpol * exp(-D) * (1+D) / f_gb²
 * and D = r²/(4*R_i*R_j).
 *
 * This is added to the existing dE_dR buffer so the HCT chain-rule
 * kernel propagates both self-GB and cross-term derivatives.
 */
extern "C" __global__ void accumulateCrossTermBornDerivatives(
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
    float* __restrict__ dE_dR,
    float globalScalingFactor,
    const float* __restrict__ groupScalingFactors
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

    float scale = globalScalingFactor * groupScalingFactors[groupIdx];

    int particleIdx = particleIndices[idx];
    int templateIdx = atomInGroup % templateNumAtoms;

    float4 pos_lig = posq[particleIdx];
    float q_lig = ligandCharges[templateIdx];
    float R_lig = ligandBornRadii[idx];

    float dEdR_accum = 0.0f;
    const float MIN_CROSS_R2 = 0.01f;

    for (int j = 0; j < numReceptorAtoms; j++) {
        float3 pos_rec = receptorPositions[j];
        float q_rec = receptorCharges[j];
        float R_rec = receptorBornRadii[groupIdx * numReceptorAtoms + j];

        float dx = pos_rec.x - pos_lig.x;
        float dy = pos_rec.y - pos_lig.y;
        float dz = pos_rec.z - pos_lig.z;
        float r2 = dx*dx + dy*dy + dz*dz;
        if (r2 < MIN_CROSS_R2) continue;

        float RiRj = R_lig * R_rec;
        float D = r2 / (4.0f * RiRj);
        float expTerm = expf(-D);
        float f_gb2 = r2 + RiRj * expTerm;
        float invFgb2 = 1.0f / f_gb2;
        float invFgb = rsqrtf(f_gb2);

        // dGpol/d(alpha2_ij) where alpha2_ij = R_i * R_j
        // = -0.5 * Gpol * exp(-D) * (1+D) / f_gb²
        float Gpol = prefactor * q_lig * q_rec * invFgb;
        float dGpol_dalpha2 = -0.5f * Gpol * expTerm * (1.0f + D) * invFgb2;

        // dE/dR_i = dGpol/d(alpha2) * d(alpha2)/dR_i = dGpol_dalpha2 * R_rec
        dEdR_accum += dGpol_dalpha2 * R_rec;
    }

    // Add to the chain-rule buffer (scaled alchemically).
    // The existing accumulateBornRadiiDerivatives already wrote the
    // self-GB contribution; we add the cross-term on top.
    dE_dR[idx] += dEdR_accum * scale;
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

        float invR = rsqrtf(r2);
        float r = r2 * invR;
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
/**
 * Precompute bornForces for each receptor atom per group.
 * bornForces = dE/dR_born * R_born^2 * obcChain
 *
 * This extracts the expensive OBC chain rule (tanh + sech²) computation
 * from the per-pair force loop, where it was redundantly computed for
 * every ligand-receptor pair even though it only depends on the receptor atom.
 *
 * Output: bornForcesRec[groupIdx * numReceptorAtoms + recIdx]
 */
extern "C" __global__ void precomputeReceptorBornForces(
    const float* __restrict__ receptorRadii,
    const float* __restrict__ receptorSelfHCT,
    const float* __restrict__ ligandToReceptorHCT,
    const float* __restrict__ receptorBornRadii,
    const float* __restrict__ receptorDeDR,
    int numReceptorAtoms,
    int numGroups,
    float* __restrict__ bornForcesRec,
    float globalScalingFactor,
    const float* __restrict__ groupScalingFactors
) {
    int totalWork = numGroups * numReceptorAtoms;
    for (int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
         globalIdx < totalWork;
         globalIdx += gridDim.x * blockDim.x) {

        int groupIdx = globalIdx / numReceptorAtoms;
        int recIdx = globalIdx % numReceptorAtoms;
        int gOffset = groupIdx * numReceptorAtoms + recIdx;

        // Skip zero-scaled groups (low alpha)
        float scale = globalScalingFactor * groupScalingFactors[groupIdx];
        if (scale < 0.05f) {
            bornForcesRec[gOffset] = 0.0f;
            continue;
        }

        float R_rec = receptorRadii[recIdx];
        float R_rec_off = R_rec - DIELECTRIC_OFFSET;
        float bornR_rec = receptorBornRadii[gOffset];
        float dEdR_rec = receptorDeDR[gOffset];

        float hctTotal = receptorSelfHCT[recIdx] + ligandToReceptorHCT[gOffset];
        float psi = 0.5f * R_rec_off * hctTotal;
        float psi2 = psi * psi;

        float tanhArg = OBC_ALPHA * psi - OBC_BETA * psi2 + OBC_GAMMA * psi2 * psi;
        float tanhVal = tanhf(tanhArg);
        float sech2 = 1.0f - tanhVal * tanhVal;
        float dTanhArgDPsi = OBC_ALPHA - 2.0f * OBC_BETA * psi + 3.0f * OBC_GAMMA * psi2;

        float obcChain = R_rec_off * dTanhArgDPsi * sech2 / R_rec;
        bornForcesRec[gOffset] = dEdR_rec * bornR_rec * bornR_rec * obcChain;
    }
}

/**
 * Fused receptor force kernel: desolvation + cross-term chain rule in TWO passes.
 *
 * Pass 1: iterate N_rec once per ligand atom, computing:
 *   - Desolvation force (HCT gradient × precomputed bornForcesRec)
 *   - dE_cross/dR_born_lig accumulation (Still equation derivative)
 *
 * Between passes: compute bornForces_lig from dEdR_lig (tanh, once per ligand atom)
 *
 * Pass 2: iterate N_rec again, computing:
 *   - Cross-term chain rule force (HCT gradient × bornForces_lig)
 *   - Ligand-ligand chain rule (N_lig iterations, tiny)
 *
 * Replaces: computeReceptorDesolvationForcesOptimized + computeCrossTermChainRuleForces
 */
extern "C" __global__ void computeFusedReceptorForces(
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
    const float* __restrict__ receptorBornRadii,    // [K * N_rec]
    const float* __restrict__ bornForcesRec,        // [K * N_rec] precomputed
    const int* __restrict__ groupStart,
    int numGroups,
    int numReceptorAtoms,
    int templateNumAtoms,
    float prefactor,
    float cutoffDistance,
    unsigned long long* __restrict__ forceBuffer,
    int paddedNumAtoms,
    float globalScalingFactor,
    const float* __restrict__ groupScalingFactors
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

    float scale = globalScalingFactor * groupScalingFactors[groupIdx];
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
    const float MIN_R2 = 0.01f;

    // ===== PASS 1: Desolvation forces + dE_cross/dR_born_lig =====
    float dEdR_lig = 0.0f;

    for (int recIdx = 0; recIdx < numReceptorAtoms; recIdx++) {
        float3 pos_rec = receptorPositions[recIdx];
        float dx = pos_rec.x - pos_lig.x;
        float dy = pos_rec.y - pos_lig.y;
        float dz = pos_rec.z - pos_lig.z;
        float r2 = dx*dx + dy*dy + dz*dz;

        if (useCutoff && r2 > cutoff2) continue;
        if (r2 < MIN_R2) continue;

        float invR = rsqrtf(r2);
        float r = r2 * invR;

        float R_rec_off = receptorRadii[recIdx] - DIELECTRIC_OFFSET;

        // --- Desolvation force: ligand screens receptor ---
        float r_plus_Slig = r + S_lig;
        if (R_rec_off < r_plus_Slig) {
            float r_minus_Slig = fabsf(r - S_lig);
            float l = (R_rec_off > r_minus_Slig) ? (1.0f/R_rec_off) : (1.0f/r_minus_Slig);
            float u = 1.0f / r_plus_Slig;
            float l2 = l*l, u2 = u*u;
            float r2_inv = invR * invR;

            float t3 = 0.125f * (1.0f + S_lig*S_lig*r2_inv) * (l2 - u2)
                     + 0.25f * logf(u/l) * r2_inv;

            float de = bornForcesRec[groupIdx * numReceptorAtoms + recIdx] * t3 * invR * scale;
            force_lig.x -= de * dx;
            force_lig.y -= de * dy;
            force_lig.z -= de * dz;
        }

        // --- Cross-term: dE_cross/dR_born_lig accumulation ---
        float R_rec_born = receptorBornRadii[groupIdx * numReceptorAtoms + recIdx];
        float q_rec = receptorCharges[recIdx];
        float RiRj = bornR_lig * R_rec_born;
        float expArg = -r2 / (4.0f * RiRj);
        float expTerm = expf(expArg);
        float f_gb2 = r2 + RiRj * expTerm;
        float f_gb = sqrtf(f_gb2);
        float dFgbDRlig = (R_rec_born * expTerm / (2.0f * f_gb)) * (1.0f + r2 / (4.0f * RiRj));
        dEdR_lig += -prefactor * q_lig * q_rec / (f_gb * f_gb) * dFgbDRlig;
    }

    // ===== Between passes: compute bornForces_lig =====
    float hctTotal_lig = hctReceptor[idx] + hctLigand[idx];
    float psi = 0.5f * R_lig_off * hctTotal_lig;
    float psi2 = psi * psi;
    float tanhArg = OBC_ALPHA * psi - OBC_BETA * psi2 + OBC_GAMMA * psi2 * psi;
    float tanhVal = tanhf(tanhArg);
    float sech2 = 1.0f - tanhVal * tanhVal;
    float dTanhDPsi = OBC_ALPHA - 2.0f * OBC_BETA * psi + 3.0f * OBC_GAMMA * psi2;
    float obcChain = R_lig_off * dTanhDPsi * sech2 / R_lig;
    float bornForces_lig = dEdR_lig * bornR_lig * bornR_lig * obcChain * scale;

    // ===== PASS 2: Cross-term chain rule forces =====

    // Part A: receptor→ligand HCT gradient (receptor screens this ligand)
    for (int recIdx = 0; recIdx < numReceptorAtoms; recIdx++) {
        float3 pos_rec = receptorPositions[recIdx];
        float R_rec = receptorRadii[recIdx];
        float R_rec_off = R_rec - DIELECTRIC_OFFSET;
        float S_rec = R_rec_off * receptorScaleFactors[recIdx];

        float dx = pos_lig.x - pos_rec.x;
        float dy = pos_lig.y - pos_rec.y;
        float dz = pos_lig.z - pos_rec.z;
        float r2 = dx*dx + dy*dy + dz*dz;

        if (useCutoff && r2 > cutoff2) continue;

        float invR = rsqrtf(r2);
        float r = r2 * invR;
        if (r < 1e-6f) continue;

        float r_plus_Srec = r + S_rec;
        if (R_lig_off >= r_plus_Srec) continue;

        float r_minus_Srec = fabsf(r - S_rec);
        float l = (R_lig_off > r_minus_Srec) ? (1.0f/R_lig_off) : (1.0f/r_minus_Srec);
        float u = 1.0f / r_plus_Srec;
        float l2 = l*l, u2 = u*u;
        float r2_inv = invR * invR;

        float t3 = 0.125f * (1.0f + S_rec*S_rec*r2_inv) * (l2 - u2)
                 + 0.25f * logf(u/l) * r2_inv;

        float de = bornForces_lig * t3 * invR;
        force_lig.x += de * dx;
        force_lig.y += de * dy;
        force_lig.z += de * dz;
    }

    // Part B: ligand-ligand HCT gradient (other ligand atoms screening this one)
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
        float invR = rsqrtf(r2);
        float r = r2 * invR;
        if (r < 1e-6f) continue;

        float r_plus_Sj = r + S_j;
        if (R_lig_off >= r_plus_Sj) continue;

        float r_minus_Sj = fabsf(r - S_j);
        float l = (R_lig_off > r_minus_Sj) ? (1.0f/R_lig_off) : (1.0f/r_minus_Sj);
        float u = 1.0f / r_plus_Sj;
        float l2 = l*l, u2 = u*u;
        float r2_inv = invR * invR;

        float t3 = 0.125f * (1.0f + S_j*S_j*r2_inv) * (l2 - u2)
                 + 0.25f * logf(u/l) * r2_inv;

        float de = bornForces_lig * t3 * invR;
        force_lig.x += de * dx;
        force_lig.y += de * dy;
        force_lig.z += de * dz;

        // Newton's 3rd law on the screening atom
        atomicAdd(&forceBuffer[particleIdx_j], static_cast<unsigned long long>((long long)(-de * dx * 0x100000000)));
        atomicAdd(&forceBuffer[particleIdx_j + paddedNumAtoms], static_cast<unsigned long long>((long long)(-de * dy * 0x100000000)));
        atomicAdd(&forceBuffer[particleIdx_j + 2*paddedNumAtoms], static_cast<unsigned long long>((long long)(-de * dz * 0x100000000)));
    }

    // Write accumulated force
    atomicAdd(&forceBuffer[particleIdx_lig], static_cast<unsigned long long>((long long)(force_lig.x * 0x100000000)));
    atomicAdd(&forceBuffer[particleIdx_lig + paddedNumAtoms], static_cast<unsigned long long>((long long)(force_lig.y * 0x100000000)));
    atomicAdd(&forceBuffer[particleIdx_lig + 2*paddedNumAtoms], static_cast<unsigned long long>((long long)(force_lig.z * 0x100000000)));
}

/**
 * Compute forces on ligand from receptor desolvation (legacy, kept for fallback).
 * Uses pre-computed bornForces per receptor atom (no OBC chain rule in inner loop).
 */
extern "C" __global__ void computeReceptorDesolvationForcesOptimized(
    const float4* __restrict__ posq,
    const int* __restrict__ particleIndices,
    const float* __restrict__ ligandRadii,
    const float* __restrict__ ligandScaleFactors,
    const float3* __restrict__ receptorPositions,
    const float* __restrict__ receptorRadii,
    const float* __restrict__ bornForcesRec,  // Pre-computed: dE/dR * R²_born * obcChain
    const int* __restrict__ groupStart,
    int numGroups,
    int numReceptorAtoms,
    int templateNumAtoms,
    float cutoffDistance,
    unsigned long long* __restrict__ forceBuffer,
    int paddedNumAtoms,
    float globalScalingFactor,
    const float* __restrict__ groupScalingFactors
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

    // Alchemical scaling for this group
    float scale = globalScalingFactor * groupScalingFactors[groupIdx];

    int particleIdx_lig = particleIndices[idx];
    int templateIdx_lig = atomInGroup % templateNumAtoms;

    float4 pos_lig = posq[particleIdx_lig];
    float R_lig = ligandRadii[templateIdx_lig];
    float R_lig_off = R_lig - DIELECTRIC_OFFSET;
    float S_lig = R_lig_off * ligandScaleFactors[templateIdx_lig];

    float3 force_lig = make_float3(0.0f, 0.0f, 0.0f);
    float cutoff2 = cutoffDistance * cutoffDistance;
    bool useCutoff = (cutoffDistance > 0.0f);

    // Desolvation force: each ligand atom loops over receptor atoms
    for (int recIdx = 0; recIdx < numReceptorAtoms; recIdx++) {
        float3 pos_rec = receptorPositions[recIdx];
        float R_rec = receptorRadii[recIdx];
        float R_rec_off = R_rec - DIELECTRIC_OFFSET;

        float dx = pos_rec.x - pos_lig.x;
        float dy = pos_rec.y - pos_lig.y;
        float dz = pos_rec.z - pos_lig.z;
        float r2 = dx*dx + dy*dy + dz*dz;

        if (useCutoff && r2 > cutoff2) continue;

        float invR = rsqrtf(r2);
        float r = r2 * invR;
        if (r < 1e-6f) continue;

        float r_plus_Slig = r + S_lig;
        if (R_rec_off >= r_plus_Slig) continue;

        float r_minus_Slig = fabsf(r - S_lig);
        float l_ij = (R_rec_off > r_minus_Slig) ? (1.0f / R_rec_off) : (1.0f / r_minus_Slig);
        float u_ij = 1.0f / r_plus_Slig;

        float bf = bornForcesRec[groupIdx * numReceptorAtoms + recIdx];

        float l2 = l_ij * l_ij;
        float u2 = u_ij * u_ij;
        float r2_inv = invR * invR;

        float t3 = 0.125f * (1.0f + S_lig * S_lig * r2_inv) * (l2 - u2)
                 + 0.25f * logf(u_ij / l_ij) * r2_inv;

        float de = bf * t3 * invR * scale;

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
    int paddedNumAtoms,
    float globalScalingFactor,
    const float* __restrict__ groupScalingFactors,
    const int* __restrict__ isActiveRecAtom
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

    // Alchemical scaling for this group
    float scale = globalScalingFactor * groupScalingFactors[groupIdx];

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
    // Accumulate dE_cross/dR_born_lig from all receptor atoms (not pruned)
    const float MIN_CROSS_R2_CHAIN = 0.01f;  // match cross-term distance floor
    float dEdR_lig = 0.0f;
    for (int j = 0; j < numReceptorAtoms; j++) {
        float3 pos_rec = receptorPositions[j];
        float q_rec = receptorCharges[j];
        float R_rec = receptorBornRadii[groupIdx * numReceptorAtoms + j];

        float dx = pos_rec.x - pos_lig.x;
        float dy = pos_rec.y - pos_lig.y;
        float dz = pos_rec.z - pos_lig.z;
        float r2 = dx*dx + dy*dy + dz*dz;
        if (r2 < MIN_CROSS_R2_CHAIN) continue;

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
    float bornForces_lig = dEdR_lig * bornR_lig * bornR_lig * obcChain_lig * scale;

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

        float invR = rsqrtf(r2);
        float r = r2 * invR;
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

    // Chain through receptor→ligand HCT (receptor screening this ligand, not pruned)
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

        float invR = rsqrtf(r2);
        float r = r2 * invR;
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

// =============================================================================
// GPU-SIDE ACCUMULATION KERNELS (eliminate host-device sync)
// =============================================================================

/**
 * Accumulate desolvation energy into group energy buffers on GPU.
 * Single-thread kernel: computes desolvation = receptorEnergy[0] - referenceEnergy,
 * applies scaling, and adds to groupEnergies/groupDesolvations/groupUnscaledEnergies.
 */
extern "C" __global__ void accumulateDesolvationOnGPU(
    const float* __restrict__ receptorEnergy,
    float referenceEnergy,
    int groupIdx,
    float globalScalingFactor,
    const float* __restrict__ groupScalingFactors,
    float* __restrict__ groupEnergies,
    float* __restrict__ groupReceptorDesolvations,
    float* __restrict__ groupUnscaledEnergies
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    float desolvation = receptorEnergy[0] - referenceEnergy;
    float scale = globalScalingFactor * groupScalingFactors[groupIdx];

    groupReceptorDesolvations[groupIdx] = desolvation * scale;
    groupEnergies[groupIdx] += desolvation * scale;
    if (groupUnscaledEnergies != 0) {
        groupUnscaledEnergies[groupIdx] += desolvation * globalScalingFactor;
    }
}

/**
 * Accumulate desolvation delta energy into group energy buffers on GPU.
 * For locality cutoff: receptorEnergy[0] already contains the delta.
 */
extern "C" __global__ void accumulateDesolvationDeltaOnGPU(
    const float* __restrict__ receptorEnergyDelta,
    int groupIdx,
    float globalScalingFactor,
    const float* __restrict__ groupScalingFactors,
    float* __restrict__ groupEnergies,
    float* __restrict__ groupReceptorDesolvations,
    float* __restrict__ groupUnscaledEnergies
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    float desolvation = receptorEnergyDelta[0];
    float scale = globalScalingFactor * groupScalingFactors[groupIdx];

    groupReceptorDesolvations[groupIdx] = desolvation * scale;
    groupEnergies[groupIdx] += desolvation * scale;
    if (groupUnscaledEnergies != 0) {
        groupUnscaledEnergies[groupIdx] += desolvation * globalScalingFactor;
    }
}

/**
 * Accumulate cross-term energies into group energy buffers on GPU.
 * Single-thread kernel, handles all groups.
 */
extern "C" __global__ void accumulateCrossTermOnGPU(
    const float* __restrict__ crossTermEnergies,
    int numGroups,
    float globalScalingFactor,
    const float* __restrict__ groupScalingFactors,
    float* __restrict__ groupEnergies,
    float* __restrict__ groupUnscaledEnergies
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    for (int g = 0; g < numGroups; g++) {
        groupEnergies[g] += crossTermEnergies[g];
        if (groupUnscaledEnergies != 0) {
            float groupScale = groupScalingFactors[g];
            if (groupScale != 0.0f) {
                groupUnscaledEnergies[g] += crossTermEnergies[g] / groupScale;
            }
        }
    }
}

// =============================================================================
// RECEPTOR LOCALITY CUTOFF KERNELS
// =============================================================================

/**
 * Compute per-receptor-atom HCT contributions to each ligand atom.
 * Stores hctPerAtom[ligIdx * numReceptorAtoms + recIdx] for baseline caching.
 * Called once at first execute to build the baseline.
 */
extern "C" __global__ void computeReceptorHCTPerAtom(
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
    float* __restrict__ hctPerAtom
) {
    // Grid-stride loop over (ligand_atom, receptor_atom) pairs
    int totalWork = totalParticles * numReceptorAtoms;
    for (int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
         globalIdx < totalWork;
         globalIdx += gridDim.x * blockDim.x) {

        int ligIdx = globalIdx / numReceptorAtoms;
        int recIdx = globalIdx % numReceptorAtoms;

        if (ligIdx >= totalParticles) continue;

        // Determine template index for this ligand atom
        int atomInGroup = ligIdx;
        for (int g = 0; g < numGroups; g++) {
            int gs = groupStart[g];
            int ge = groupStart[g + 1];
            if (ligIdx >= gs && ligIdx < ge) {
                atomInGroup = ligIdx - gs;
                break;
            }
        }
        int templateIdx = atomInGroup % templateNumAtoms;

        int particleIdx = particleIndices[ligIdx];
        float4 pos = posq[particleIdx];
        float R_i = radii[templateIdx];
        float R_i_off = R_i - DIELECTRIC_OFFSET;

        float3 pos_rec = receptorPositions[recIdx];
        float dx = pos.x - pos_rec.x;
        float dy = pos.y - pos_rec.y;
        float dz = pos.z - pos_rec.z;
        float r2 = dx*dx + dy*dy + dz*dz;
        float r = sqrtf(r2);

        float term = 0.0f;
        if (r > 1e-6f) {
            float R_j = receptorRadii[recIdx];
            float R_j_off = R_j - DIELECTRIC_OFFSET;
            float S_j = R_j_off * receptorScaleFactors[recIdx];

            float r_plus_Sj = r + S_j;
            if (R_i_off < r_plus_Sj) {
                float r_minus_Sj = fabsf(r - S_j);
                float l_ij = (R_i_off > r_minus_Sj) ? (1.0f / R_i_off) : (1.0f / r_minus_Sj);
                float u_ij = 1.0f / r_plus_Sj;
                float l_ij2 = l_ij * l_ij;
                float u_ij2 = u_ij * u_ij;
                float r_inv = 1.0f / r;

                term = l_ij - u_ij +
                       0.25f * r * (u_ij2 - l_ij2) +
                       0.5f * r_inv * logf(u_ij / l_ij) +
                       0.25f * S_j * S_j * r_inv * (l_ij2 - u_ij2);

                if (R_i_off < (S_j - r)) {
                    term += 2.0f * (1.0f / R_i_off - l_ij);
                }
            }
        }

        hctPerAtom[ligIdx * numReceptorAtoms + recIdx] = term;
    }
}

/**
 * Reconstruct receptor HCT on ligand using baseline + active update.
 * For each ligand atom: sum frozen baseline from inactive atoms + fresh from active atoms.
 * Active atoms use freshly computed values; inactive atoms use cached baseline.
 */
extern "C" __global__ void reconstructReceptorHCT(
    const float4* __restrict__ posq,
    const int* __restrict__ particleIndices,
    const float* __restrict__ radii,
    const float3* __restrict__ receptorPositions,
    const float* __restrict__ receptorRadii,
    const float* __restrict__ receptorScaleFactors,
    const float* __restrict__ hctPerAtomBaseline,
    const int* __restrict__ isActiveRecAtom,
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

    int atomInGroup = idx;
    int myGroupIdx = 0;
    for (int g = 0; g < numGroups; g++) {
        int gs = groupStart[g];
        int ge = groupStart[g + 1];
        if (idx >= gs && idx < ge) {
            atomInGroup = idx - gs;
            myGroupIdx = g;
            break;
        }
    }
    int templateIdx = atomInGroup % templateNumAtoms;
    const int* activeMask = isActiveRecAtom + myGroupIdx * numReceptorAtoms;

    int particleIdx = particleIndices[idx];
    float4 pos = posq[particleIdx];
    float R_i = radii[templateIdx];
    float R_i_off = R_i - DIELECTRIC_OFFSET;

    float hct = 0.0f;
    float cutoff2 = cutoffDistance * cutoffDistance;
    bool useCutoff = (cutoffDistance > 0.0f);

    for (int j = 0; j < numReceptorAtoms; j++) {
        if (!activeMask[j]) {
            // Inactive: use frozen baseline
            hct += hctPerAtomBaseline[idx * numReceptorAtoms + j];
        } else {
            // Active: compute fresh
            float3 pos_rec = receptorPositions[j];
            float dx = pos.x - pos_rec.x;
            float dy = pos.y - pos_rec.y;
            float dz = pos.z - pos_rec.z;
            float r2 = dx*dx + dy*dy + dz*dz;

            if (useCutoff && r2 > cutoff2) continue;

            float r = sqrtf(r2);
            if (r < 1e-6f) continue;

            float R_j = receptorRadii[j];
            float R_j_off = R_j - DIELECTRIC_OFFSET;
            float S_j = R_j_off * receptorScaleFactors[j];

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
    }

    hctReceptor[idx] = hct;
}

/**
 * Compute baseline HCT sum for each ligand atom.
 * Sums hctPerAtomBaseline[idx * numReceptorAtoms + j] for all j.
 * Called once after the per-atom baseline is computed.
 */
extern "C" __global__ void computeBaselineHCTSum(
    const float* __restrict__ hctPerAtomBaseline,
    int numReceptorAtoms,
    int totalParticles,
    float* __restrict__ baselineSum
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= totalParticles) return;

    float sum = 0.0f;
    const float* row = hctPerAtomBaseline + idx * numReceptorAtoms;
    for (int j = 0; j < numReceptorAtoms; j++) {
        sum += row[j];
    }
    baselineSum[idx] = sum;
}

/**
 * Fast receptor HCT reconstruction using precomputed baseline sum.
 *
 * Instead of iterating ALL N_rec atoms (reading baseline for inactive,
 * computing fresh for active), starts with the precomputed baseline sum
 * and only iterates active atoms to compute the delta:
 *
 *   hct = baselineSum[idx] + sum_active(fresh_j - baseline_j)
 *
 * Cost: O(|A|) per thread instead of O(N_rec).
 * Uses tiled shared memory for active receptor atoms.
 */
extern "C" __global__ void reconstructReceptorHCTFast(
    const float4* __restrict__ posq,
    const int* __restrict__ particleIndices,
    const float* __restrict__ radii,
    const float3* __restrict__ receptorPositions,
    const float* __restrict__ receptorRadii,
    const float* __restrict__ receptorScaleFactors,
    const float* __restrict__ hctPerAtomBaseline,
    const float* __restrict__ baselineSum,
    const int* __restrict__ isActiveRecAtom,
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

    int atomInGroup = idx;
    int myGroupIdx = 0;
    for (int g = 0; g < numGroups; g++) {
        int gs = groupStart[g];
        int ge = groupStart[g + 1];
        if (idx >= gs && idx < ge) {
            atomInGroup = idx - gs;
            myGroupIdx = g;
            break;
        }
    }
    int templateIdx = atomInGroup % templateNumAtoms;
    const int* activeMask = isActiveRecAtom + myGroupIdx * numReceptorAtoms;
    const float* myBaseline = hctPerAtomBaseline + idx * numReceptorAtoms;

    int particleIdx = particleIndices[idx];
    float4 pos = posq[particleIdx];
    float R_i = radii[templateIdx];
    float R_i_off = R_i - DIELECTRIC_OFFSET;

    // Start with precomputed baseline sum (all receptor contributions at reference position)
    float hct = baselineSum[idx];

    float cutoff2 = cutoffDistance * cutoffDistance;
    bool useCutoff = (cutoffDistance > 0.0f);

    // Only iterate active atoms: compute (fresh - baseline) delta
    for (int j = 0; j < numReceptorAtoms; j++) {
        if (!activeMask[j]) continue;

        // Subtract baseline contribution for this atom
        hct -= myBaseline[j];

        // Add fresh computation
        float3 pos_rec = receptorPositions[j];
        float dx = pos.x - pos_rec.x;
        float dy = pos.y - pos_rec.y;
        float dz = pos.z - pos_rec.z;
        float r2 = dx*dx + dy*dy + dz*dz;

        if (useCutoff && r2 > cutoff2) continue;

        float invR = rsqrtf(r2);
        float r = r2 * invR;
        if (r < 1e-6f) continue;

        float R_j = receptorRadii[j];
        float R_j_off = R_j - DIELECTRIC_OFFSET;
        float S_j = R_j_off * receptorScaleFactors[j];

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
 * Determine which receptor atoms are within locality cutoff of any ligand atom.
 * Sets isActiveRecAtom[j] = 1 if receptor atom j is within cutoff of any ligand atom
 * in any group, 0 otherwise.
 */
/**
 * Compute per-group active receptor atom masks with scale-dependent cutoff.
 *
 * The cutoff shrinks with sqrt(groupScale): at full scaling, use the base
 * cutoff. At low scaling, fewer atoms are active — the rest fall back to
 * the precomputed HCT baseline. At zero scaling, ALL atoms use baseline
 * (no fresh HCT computation needed).
 *
 * This is safe because the HCT integral is distance-independent beyond ~3A,
 * so the baseline stays accurate as the ligand moves. The error from using
 * baseline values is proportional to the scaling factor, making it negligible
 * at low alpha.
 */
extern "C" __global__ void computeActiveReceptorAtoms(
    const float4* __restrict__ posq,
    const int* __restrict__ particleIndices,
    const float3* __restrict__ receptorPositions,
    int numReceptorAtoms,
    const int* __restrict__ groupStart,
    int numGroups,
    float baseCutoff2,
    int* __restrict__ isActiveRecAtom,
    float globalScalingFactor,
    const float* __restrict__ groupScalingFactors
) {
    int totalWork = numGroups * numReceptorAtoms;
    for (int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
         globalIdx < totalWork;
         globalIdx += gridDim.x * blockDim.x) {

        int g = globalIdx / numReceptorAtoms;
        int j = globalIdx % numReceptorAtoms;

        // Scale cutoff: low-alpha groups use smaller cutoff,
        // falling back to cached HCT baseline for more atoms
        float scale = globalScalingFactor * groupScalingFactors[g];
        float effectiveCutoff2 = baseCutoff2 * sqrtf(fmaxf(scale, 0.0f));

        float3 pos_rec = receptorPositions[j];
        int active = 0;

        int gs = groupStart[g];
        int ge = groupStart[g + 1];
        for (int k = gs; k < ge && !active; k++) {
            int particleIdx = particleIndices[k];
            float4 pos_lig = posq[particleIdx];
            float dx = pos_rec.x - pos_lig.x;
            float dy = pos_rec.y - pos_lig.y;
            float dz = pos_rec.z - pos_lig.z;
            float r2 = dx*dx + dy*dy + dz*dz;
            if (r2 < effectiveCutoff2) {
                active = 1;
            }
        }

        isActiveRecAtom[g * numReceptorAtoms + j] = active;
    }
}

/**
 * Compute receptor energy DELTA for all groups (batched).
 * Each thread handles one (group, receptor_atom) pair.
 * Output: energyDelta[groupIdx] via atomicAdd.
 * Born radii: receptorBornRadii[groupIdx * numReceptorAtoms + i]
 */
extern "C" __global__ void computeReceptorEnergyDelta(
    const float3* __restrict__ receptorPositions,
    const float* __restrict__ receptorCharges,
    const float* __restrict__ receptorBornRadiiRef,
    const float* __restrict__ receptorBornRadii,
    const int* __restrict__ isActiveRecAtom,
    int numReceptorAtoms,
    int numGroups,
    float prefactor,
    float* __restrict__ energyDelta,
    float globalScalingFactor,
    const float* __restrict__ groupScalingFactors
) {
    int totalWork = numGroups * numReceptorAtoms;
    for (int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
         globalIdx < totalWork;
         globalIdx += gridDim.x * blockDim.x) {

        int groupIdx = globalIdx / numReceptorAtoms;
        int a = globalIdx % numReceptorAtoms;

        // Skip zero-scaled groups entirely (no desolvation contribution)
        float scale = globalScalingFactor * groupScalingFactors[groupIdx];
        if (scale < 0.05f) continue;

        const int* activeMask = isActiveRecAtom + groupIdx * numReceptorAtoms;
        const float* bornRadiiG = receptorBornRadii + groupIdx * numReceptorAtoms;

        if (!activeMask[a]) continue;

        float3 pos_a = receptorPositions[a];
        float q_a = receptorCharges[a];
        float R_a_new = bornRadiiG[a];
        float R_a_ref = receptorBornRadiiRef[a];

        float delta = 0.0f;

        // Self-term delta
        delta += 0.5f * prefactor * q_a * q_a * (1.0f / R_a_new - 1.0f / R_a_ref);

        // Pair-term deltas
        for (int j = 0; j < numReceptorAtoms; j++) {
            if (j == a) continue;
            if (activeMask[j] && j < a) continue;

            float3 pos_j = receptorPositions[j];
            float q_j = receptorCharges[j];

            float dx = pos_a.x - pos_j.x;
            float dy = pos_a.y - pos_j.y;
            float dz = pos_a.z - pos_j.z;
            float r2 = dx*dx + dy*dy + dz*dz;

            float R_j_new = bornRadiiG[j];
            float D_new = R_a_new * R_j_new;
            float exp_new = expf(-r2 / (4.0f * D_new));
            float f_new = sqrtf(r2 + D_new * exp_new);
            float E_new = prefactor * q_a * q_j / f_new;

            float R_j_ref = receptorBornRadiiRef[j];
            float D_ref = R_a_ref * R_j_ref;
            float exp_ref = expf(-r2 / (4.0f * D_ref));
            float f_ref = sqrtf(r2 + D_ref * exp_ref);
            float E_ref = prefactor * q_a * q_j / f_ref;

            delta += E_new - E_ref;
        }

        atomicAdd(&energyDelta[groupIdx], delta);
    }
}

// Legacy single-group shared-memory reduction version (kept for non-locality path)
extern "C" __global__ void computeReceptorEnergyDeltaLegacy(
    const float3* __restrict__ receptorPositions,
    const float* __restrict__ receptorCharges,
    const float* __restrict__ receptorBornRadiiRef,
    const float* __restrict__ receptorBornRadii,
    const int* __restrict__ isActiveRecAtom,
    int numReceptorAtoms,
    int groupIdx,
    float prefactor,
    float* __restrict__ energyDelta
) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int a = blockIdx.x * blockDim.x + threadIdx.x;
    const int* activeMask = isActiveRecAtom + groupIdx * numReceptorAtoms;
    const float* bornRadiiG = receptorBornRadii + groupIdx * numReceptorAtoms;

    float delta = 0.0f;
    if (a < numReceptorAtoms && activeMask[a]) {
        float3 pos_a = receptorPositions[a];
        float q_a = receptorCharges[a];
        float R_a_new = bornRadiiG[a];
        float R_a_ref = receptorBornRadiiRef[a];
        delta += 0.5f * prefactor * q_a * q_a * (1.0f / R_a_new - 1.0f / R_a_ref);
        for (int j = 0; j < numReceptorAtoms; j++) {
            if (j == a) continue;
            if (activeMask[j] && j < a) continue;
            float3 pos_j = receptorPositions[j];
            float q_j = receptorCharges[j];
            float dx = pos_a.x - pos_j.x; float dy = pos_a.y - pos_j.y; float dz = pos_a.z - pos_j.z;
            float r2 = dx*dx + dy*dy + dz*dz;
            float R_j_new = bornRadiiG[j]; float D_new = R_a_new * R_j_new;
            float exp_new = expf(-r2 / (4.0f * D_new)); float f_new = sqrtf(r2 + D_new * exp_new);
            float R_j_ref = receptorBornRadiiRef[j]; float D_ref = R_a_ref * R_j_ref;
            float exp_ref = expf(-r2 / (4.0f * D_ref)); float f_ref = sqrtf(r2 + D_ref * exp_ref);
            delta += prefactor * q_a * q_j * (1.0f/f_new - 1.0f/f_ref);
        }
    }
    sdata[tid] = delta; __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s]; __syncthreads();
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(energyDelta, sdata[0]);
    }
}

/**
 * Compute dE/dR_born for active receptor atoms only.
 * Each active atom iterates all N_rec atoms for its dE/dR.
 * Inactive atoms get dE/dR = 0 (they don't contribute to desolvation forces).
 */
/**
 * Compute dE/dR_born for all groups (batched).
 * Output: receptorDeDR[groupIdx * numReceptorAtoms + i]
 * Born radii: receptorBornRadii[groupIdx * numReceptorAtoms + i]
 */
extern "C" __global__ void computeReceptorDeDRActive(
    const float3* __restrict__ receptorPositions,
    const float* __restrict__ receptorCharges,
    const float* __restrict__ receptorBornRadii,
    const int* __restrict__ isActiveRecAtom,
    int numReceptorAtoms,
    int numGroups,
    float prefactor,
    float* __restrict__ receptorDeDR
) {
    int totalWork = numGroups * numReceptorAtoms;
    for (int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
         globalIdx < totalWork;
         globalIdx += gridDim.x * blockDim.x) {

        int groupIdx = globalIdx / numReceptorAtoms;
        int i = globalIdx % numReceptorAtoms;

        if (!isActiveRecAtom[groupIdx * numReceptorAtoms + i]) {
            receptorDeDR[groupIdx * numReceptorAtoms + i] = 0.0f;
            continue;
        }

        const float* bornRadiiG = receptorBornRadii + groupIdx * numReceptorAtoms;

        float3 pos_i = receptorPositions[i];
        float q_i = receptorCharges[i];
        float R_i = bornRadiiG[i];

        float dEdR = -0.5f * prefactor * q_i * q_i / (R_i * R_i);

        for (int j = 0; j < numReceptorAtoms; j++) {
            if (j == i) continue;

            float3 pos_j = receptorPositions[j];
            float q_j = receptorCharges[j];
            float R_j = bornRadiiG[j];

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

        receptorDeDR[groupIdx * numReceptorAtoms + i] = dEdR;
    }
}

// =============================================================================
// TILED FORCE KERNELS (matching OpenMM architecture)
// =============================================================================

/**
 * Tiled pass 1: Cross-term energy + direct forces + dE/dR_born_lig + desolvation chain rule.
 *
 * Each warp processes one receptor block (32 atoms) against ALL ligand atoms
 * for one group. Ligand data loaded into shared memory once per warp.
 *
 * Fuses: computeCrossTermGBEnergy + computeFusedReceptorForces pass 1
 * into a SINGLE pass of N_rec per group.
 *
 * Output:
 *   forceBuffer: cross-term direct forces + desolvation chain rule forces on ligand
 *   crossTermEnergies[groupIdx]: cross-term energy per group
 *   dEdR_crossTerm[ligGlobalIdx]: accumulated dE_cross/dR_born_lig (fixed-point)
 */
extern "C" __global__ void computePairwiseGBForceTiled(
    const float4* __restrict__ posq,
    const int* __restrict__ particleIndices,
    const float* __restrict__ ligandRadii,
    const float* __restrict__ ligandScaleFactors,
    const float* __restrict__ ligandCharges,
    const float* __restrict__ ligandBornRadii,
    const float3* __restrict__ receptorPositions,
    const float* __restrict__ receptorRadii,
    const float* __restrict__ receptorScaleFactors,
    const float* __restrict__ receptorCharges,
    const float* __restrict__ receptorBornRadii,      // [K * N_rec]
    const float* __restrict__ bornForcesRec,          // [K * N_rec] precomputed
    const int* __restrict__ groupStart,
    int numGroups,
    int numReceptorAtoms,
    int templateNumAtoms,
    float prefactor,
    float cutoffDistance,
    unsigned long long* __restrict__ forceBuffer,
    int paddedNumAtoms,
    float* __restrict__ crossTermEnergies,            // [numGroups]
    unsigned long long* __restrict__ dEdR_crossTerm,  // [totalParticles] fixed-point
    float globalScalingFactor,
    const float* __restrict__ groupScalingFactors,
    int numRecBlocks,
    const float4* __restrict__ recBlockBounds,        // [numRecBlocks] (cx, cy, cz, radius)
    float localityCutoff,                             // tile-skip cutoff (-1 = no skip)
    float* __restrict__ crossTermBlockCache           // [numGroups * numRecBlocks] or NULL
) {
    // Max ligand atoms in shared memory
    const int MAX_LIG = 64;

    // Shared memory: ligand data for this warp's group
    __shared__ float3 sLigPos[MAX_LIG * 8];        // 8 warps per block
    __shared__ float sLigCharge[MAX_LIG * 8];
    __shared__ float sLigBornR[MAX_LIG * 8];
    __shared__ float sLigR_off[MAX_LIG * 8];
    __shared__ float sLigS[MAX_LIG * 8];

    const int tgx = threadIdx.x & (TILE_SIZE - 1);
    const int warpInBlock = threadIdx.x / TILE_SIZE;
    const int sBase = warpInBlock * MAX_LIG;

    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) / TILE_SIZE;
    const int totalWarps = (gridDim.x * blockDim.x) / TILE_SIZE;

    int totalTiles = numGroups * numRecBlocks;
    float cutoff2 = cutoffDistance * cutoffDistance;
    bool useCutoff = (cutoffDistance > 0.0f);
    bool useTileSkip = (localityCutoff > 0.0f);
    const float MIN_R2 = 0.01f;

    for (int tileIdx = warp; tileIdx < totalTiles; tileIdx += totalWarps) {
        int groupIdx = tileIdx / numRecBlocks;
        int recBlock = tileIdx % numRecBlocks;

        float scale = globalScalingFactor * groupScalingFactors[groupIdx];
        if (scale < 0.05f) continue;  // skip zero-scaled groups

        int gs = groupStart[groupIdx];
        int ge = groupStart[groupIdx + 1];
        int groupSize = ge - gs;
        int nLig = (groupSize < MAX_LIG) ? groupSize : MAX_LIG;

        // Cooperative load: ligand atoms into shared memory
        for (int i = tgx; i < nLig; i += TILE_SIZE) {
            int ligGlobal = gs + i;
            int particleIdx = particleIndices[ligGlobal];
            int templateIdx = i % templateNumAtoms;
            float4 p = posq[particleIdx];
            sLigPos[sBase + i] = make_float3(p.x, p.y, p.z);
            sLigCharge[sBase + i] = ligandCharges[templateIdx];
            sLigBornR[sBase + i] = ligandBornRadii[ligGlobal];
            float R = ligandRadii[templateIdx];
            sLigR_off[sBase + i] = R - DIELECTRIC_OFFSET;
            sLigS[sBase + i] = (R - DIELECTRIC_OFFSET) * ligandScaleFactors[templateIdx];
        }
        __syncwarp();

        // Tile-skip: check if any ligand atom is close enough to this receptor block
        if (useTileSkip) {
            float4 bounds = recBlockBounds[recBlock];  // (cx, cy, cz, radius)
            float threshold = localityCutoff + bounds.w;
            float threshold2 = threshold * threshold;
            bool anyClose = false;
            for (int i = 0; i < nLig && !anyClose; i++) {
                float dx = sLigPos[sBase + i].x - bounds.x;
                float dy = sLigPos[sBase + i].y - bounds.y;
                float dz = sLigPos[sBase + i].z - bounds.z;
                if (dx*dx + dy*dy + dz*dz < threshold2) anyClose = true;
            }
            if (!anyClose) continue;  // skip this tile entirely
        }

        // Load receptor atom for this thread
        int recIdx = recBlock * TILE_SIZE + tgx;
        bool validRec = (recIdx < numReceptorAtoms);

        float3 recPos = make_float3(0, 0, 0);
        float recR = 0.1f, recR_off = 0.1f, recS = 0.1f;
        float recQ = 0.0f, recBornR = 1.0f, recBF = 0.0f;

        if (validRec) {
            recPos = receptorPositions[recIdx];
            recR = receptorRadii[recIdx];
            recR_off = recR - DIELECTRIC_OFFSET;
            recS = recR_off * receptorScaleFactors[recIdx];
            recQ = receptorCharges[recIdx];
            recBornR = receptorBornRadii[groupIdx * numReceptorAtoms + recIdx];
            recBF = bornForcesRec[groupIdx * numReceptorAtoms + recIdx];
        }

        // Accumulate per-receptor-atom contributions
        float3 recForceOnLig = make_float3(0, 0, 0);
        float crossEnergy = 0.0f;

        // Iterate over all ligand atoms from shared memory
        for (int li = 0; li < nLig; li++) {
            float3 lPos = sLigPos[sBase + li];
            float lQ = sLigCharge[sBase + li];
            float lBornR = sLigBornR[sBase + li];
            float lR_off = sLigR_off[sBase + li];
            float lS = sLigS[sBase + li];

            float dx = recPos.x - lPos.x;
            float dy = recPos.y - lPos.y;
            float dz = recPos.z - lPos.z;
            float r2 = dx*dx + dy*dy + dz*dz;

            if (useCutoff && r2 > cutoff2) continue;
            if (r2 < MIN_R2) continue;

            float invR = rsqrtf(r2);
            float r = r2 * invR;

            // === Cross-term Still equation: energy + direct force ===
            float RiRj = lBornR * recBornR;
            float expArg = -r2 / (4.0f * RiRj);
            float expTerm = expf(expArg);
            float f_gb2 = r2 + RiRj * expTerm;
            float f_gb = sqrtf(f_gb2);
            float invFgb = 1.0f / f_gb;

            float pairEnergy = prefactor * lQ * recQ * invFgb;
            crossEnergy += pairEnergy;

            // Direct force on ligand
            float dFgbDr = (r * invFgb) * (1.0f - 0.25f * expTerm);
            float dEdR_direct = -prefactor * lQ * recQ * invFgb * invFgb * dFgbDr * scale;
            // Force direction: rec → lig = (dx, dy, dz), force on lig
            float fx = dEdR_direct * dx * invR;
            float fy = dEdR_direct * dy * invR;
            float fz = dEdR_direct * dz * invR;

            // === dE_cross/dR_born_lig accumulation ===
            float dFgbDRlig = (recBornR * expTerm / (2.0f * f_gb)) * (1.0f + r2 / (4.0f * RiRj));
            float dEdR_lig = -prefactor * lQ * recQ * invFgb * invFgb * dFgbDRlig;

            // === Desolvation chain rule: ligand screens receptor ===
            float r_plus_Slig = r + lS;
            if (recR_off < r_plus_Slig) {
                float r_minus_Slig = fabsf(r - lS);
                float l = (recR_off > r_minus_Slig) ? (1.0f/recR_off) : (1.0f/r_minus_Slig);
                float u = 1.0f / r_plus_Slig;
                float l2 = l*l, u2 = u*u;
                float r2_inv = invR * invR;

                float t3 = 0.125f * (1.0f + lS*lS*r2_inv) * (l2 - u2)
                         + 0.25f * logf(u/l) * r2_inv;

                float de_desolv = recBF * t3 * invR * scale;
                fx -= de_desolv * dx;
                fy -= de_desolv * dy;
                fz -= de_desolv * dz;
            }

            // Write forces on ligand atom via atomicAdd
            int ligGlobal = gs + li;
            int ligParticle = particleIndices[ligGlobal];
            atomicAdd(&forceBuffer[ligParticle], static_cast<unsigned long long>((long long)(fx * 0x100000000)));
            atomicAdd(&forceBuffer[ligParticle + paddedNumAtoms], static_cast<unsigned long long>((long long)(fy * 0x100000000)));
            atomicAdd(&forceBuffer[ligParticle + 2*paddedNumAtoms], static_cast<unsigned long long>((long long)(fz * 0x100000000)));

            // Accumulate dE/dR_born_lig via fixed-point atomicAdd
            atomicAdd(&dEdR_crossTerm[ligGlobal], static_cast<unsigned long long>((long long)(dEdR_lig * 0x100000000)));
        }

        // Write cross-term energy (per-group, scaled)
        if (validRec) {
            atomicAdd(&crossTermEnergies[groupIdx], crossEnergy * scale);
        }

        // Cache per-tile cross-term energy (warp reduction) for tile-skip reconstruction
        if (crossTermBlockCache != NULL) {
            // Warp-reduce crossEnergy * scale across 32 threads
            float tileEnergy = crossEnergy * scale;
            for (int offset = TILE_SIZE/2; offset > 0; offset >>= 1)
                tileEnergy += __shfl_down_sync(0xFFFFFFFF, tileEnergy, offset);
            if (tgx == 0)
                crossTermBlockCache[groupIdx * numRecBlocks + recBlock] = tileEnergy;
        }

        __syncwarp();
    }
}

/**
 * Add cached cross-term energy for distant receptor blocks (skipped by tile-skip).
 * Parallelized: one thread per (group, recBlock) tile. Each thread checks nearness
 * and adds cached energy if the tile was distant.
 */
extern "C" __global__ void addDistantCrossTermFromCache(
    const float* __restrict__ crossTermBlockCache,  // [numGroups * numRecBlocks]
    const float4* __restrict__ posq,
    const int* __restrict__ particleIndices,
    const float4* __restrict__ recBlockBounds,
    float localityCutoff,
    const int* __restrict__ groupStart,
    int numGroups,
    int numRecBlocks,
    float globalScalingFactor,
    const float* __restrict__ groupScalingFactors,
    float* __restrict__ crossTermEnergies            // [numGroups] — add to existing
) {
    int tileIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalTiles = numGroups * numRecBlocks;
    if (tileIdx >= totalTiles) return;

    int groupIdx = tileIdx / numRecBlocks;
    int b = tileIdx % numRecBlocks;

    float scale = globalScalingFactor * groupScalingFactors[groupIdx];
    if (scale < 0.05f) return;

    float4 bounds = recBlockBounds[b];
    float threshold = localityCutoff + bounds.w;
    float threshold2 = threshold * threshold;

    int gs = groupStart[groupIdx];
    int ge = groupStart[groupIdx + 1];
    bool anyClose = false;
    for (int li = gs; li < ge && !anyClose; li++) {
        int particleIdx = particleIndices[li];
        float4 p = posq[particleIdx];
        float dx = p.x - bounds.x;
        float dy = p.y - bounds.y;
        float dz = p.z - bounds.z;
        if (dx*dx + dy*dy + dz*dz < threshold2) anyClose = true;
    }

    if (!anyClose) {
        float cached = crossTermBlockCache[groupIdx * numRecBlocks + b];
        if (cached != 0.0f) {
            atomicAdd(&crossTermEnergies[groupIdx], cached);
        }
    }
}

/**
 * Reduce dE/dR_born_lig to bornForce_lig for each ligand atom.
 * bornForce_lig = dE/dR * R_born^2 * obcChain * scale
 */
extern "C" __global__ void reduceLigandBornForce(
    const unsigned long long* __restrict__ dEdR_crossTerm,  // fixed-point
    const float* __restrict__ ligandBornRadii,
    const float* __restrict__ hctReceptor,
    const float* __restrict__ hctLigand,
    const float* __restrict__ ligandRadii,
    const int* __restrict__ groupStart,
    int numGroups,
    int totalParticles,
    int templateNumAtoms,
    float globalScalingFactor,
    const float* __restrict__ groupScalingFactors,
    float* __restrict__ bornForceLig   // output [totalParticles]
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= totalParticles) return;

    // Determine group and template
    int atomInGroup = idx;
    int groupIdx = 0;
    for (int g = 0; g < numGroups; g++) {
        int gs = groupStart[g];
        int ge = groupStart[g + 1];
        if (idx >= gs && idx < ge) {
            atomInGroup = idx - gs;
            groupIdx = g;
            break;
        }
    }
    int templateIdx = atomInGroup % templateNumAtoms;

    float scale = globalScalingFactor * groupScalingFactors[groupIdx];

    // Convert fixed-point dE/dR to float
    float dEdR_lig = (float)((long long)dEdR_crossTerm[idx] / (double)0x100000000);

    // OBC chain rule
    float R = ligandRadii[templateIdx];
    float R_off = R - DIELECTRIC_OFFSET;
    float bornR = ligandBornRadii[idx];
    float hctTotal = hctReceptor[idx] + hctLigand[idx];
    float psi = 0.5f * R_off * hctTotal;
    float psi2 = psi * psi;
    float tanhArg = OBC_ALPHA * psi - OBC_BETA * psi2 + OBC_GAMMA * psi2 * psi;
    float tanhVal = tanhf(tanhArg);
    float sech2 = 1.0f - tanhVal * tanhVal;
    float dTanhDPsi = OBC_ALPHA - 2.0f * OBC_BETA * psi + 3.0f * OBC_GAMMA * psi2;
    float obcChain = R_off * dTanhDPsi * sech2 / R;

    bornForceLig[idx] = dEdR_lig * bornR * bornR * obcChain * scale;
}

/**
 * Tiled pass 2: Cross-term HCT chain rule forces.
 * Uses precomputed bornForceLig from reduceLigandBornForce.
 *
 * For each (receptor, ligand) pair where receptor screens ligand:
 *   force_lig += bornForceLig * t3(r, S_rec, R_lig_off) * (dx/r)
 *
 * Same tile structure as pass 1.
 */
extern "C" __global__ void computePairwiseChainRuleTiled(
    const float4* __restrict__ posq,
    const int* __restrict__ particleIndices,
    const float* __restrict__ ligandRadii,
    const float* __restrict__ ligandScaleFactors,
    const float3* __restrict__ receptorPositions,
    const float* __restrict__ receptorRadii,
    const float* __restrict__ receptorScaleFactors,
    const float* __restrict__ bornForceLig,
    const int* __restrict__ groupStart,
    int numGroups,
    int numReceptorAtoms,
    int templateNumAtoms,
    float cutoffDistance,
    unsigned long long* __restrict__ forceBuffer,
    int paddedNumAtoms,
    int numRecBlocks,
    const float4* __restrict__ recBlockBounds,        // [numRecBlocks] (cx, cy, cz, radius)
    float localityCutoff,                             // tile-skip cutoff (-1 = no skip)
    float globalScalingFactor,
    const float* __restrict__ groupScalingFactors
) {
    const int MAX_LIG = 64;

    __shared__ float3 sLigPos[MAX_LIG * 8];
    __shared__ float sLigR_off[MAX_LIG * 8];
    __shared__ float sLigBF[MAX_LIG * 8];  // bornForceLig per ligand

    const int tgx = threadIdx.x & (TILE_SIZE - 1);
    const int warpInBlock = threadIdx.x / TILE_SIZE;
    const int sBase = warpInBlock * MAX_LIG;

    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) / TILE_SIZE;
    const int totalWarps = (gridDim.x * blockDim.x) / TILE_SIZE;

    int totalTiles = numGroups * numRecBlocks;
    float cutoff2 = cutoffDistance * cutoffDistance;
    bool useCutoff = (cutoffDistance > 0.0f);
    bool useTileSkip = (localityCutoff > 0.0f);

    for (int tileIdx = warp; tileIdx < totalTiles; tileIdx += totalWarps) {
        int groupIdx = tileIdx / numRecBlocks;
        int recBlock = tileIdx % numRecBlocks;

        // Skip zero-scaled groups
        float scale = globalScalingFactor * groupScalingFactors[groupIdx];
        if (scale < 0.05f) continue;

        int gs = groupStart[groupIdx];
        int ge = groupStart[groupIdx + 1];
        int groupSize = ge - gs;
        int nLig = (groupSize < MAX_LIG) ? groupSize : MAX_LIG;

        // Load ligand data into shared memory
        for (int i = tgx; i < nLig; i += TILE_SIZE) {
            int ligGlobal = gs + i;
            int particleIdx = particleIndices[ligGlobal];
            int templateIdx = i % templateNumAtoms;
            float4 p = posq[particleIdx];
            sLigPos[sBase + i] = make_float3(p.x, p.y, p.z);
            float R = ligandRadii[templateIdx];
            sLigR_off[sBase + i] = R - DIELECTRIC_OFFSET;
            sLigBF[sBase + i] = bornForceLig[ligGlobal];
        }
        __syncwarp();

        // Tile-skip: check if any ligand atom is close enough to this receptor block
        if (useTileSkip) {
            float4 bounds = recBlockBounds[recBlock];
            float threshold = localityCutoff + bounds.w;
            float threshold2 = threshold * threshold;
            bool anyClose = false;
            for (int i = 0; i < nLig && !anyClose; i++) {
                float dx = sLigPos[sBase + i].x - bounds.x;
                float dy = sLigPos[sBase + i].y - bounds.y;
                float dz = sLigPos[sBase + i].z - bounds.z;
                if (dx*dx + dy*dy + dz*dz < threshold2) anyClose = true;
            }
            if (!anyClose) continue;
        }

        // Load receptor atom
        int recIdx = recBlock * TILE_SIZE + tgx;
        bool validRec = (recIdx < numReceptorAtoms);
        float3 recPos = make_float3(0, 0, 0);
        float recR_off = 0.1f, recS = 0.1f;
        if (validRec) {
            recPos = receptorPositions[recIdx];
            float R = receptorRadii[recIdx];
            recR_off = R - DIELECTRIC_OFFSET;
            recS = recR_off * receptorScaleFactors[recIdx];
        }

        // Iterate ligand atoms: receptor screens ligand → force on ligand
        for (int li = 0; li < nLig; li++) {
            float3 lPos = sLigPos[sBase + li];
            float lR_off = sLigR_off[sBase + li];
            float lBF = sLigBF[sBase + li];

            float dx = lPos.x - recPos.x;
            float dy = lPos.y - recPos.y;
            float dz = lPos.z - recPos.z;
            float r2 = dx*dx + dy*dy + dz*dz;

            if (useCutoff && r2 > cutoff2) continue;

            float invR = rsqrtf(r2);
            float r = r2 * invR;
            if (r < 1e-6f) continue;

            float r_plus_Srec = r + recS;
            if (!validRec || lR_off >= r_plus_Srec) continue;

            float r_minus_Srec = fabsf(r - recS);
            float l = (lR_off > r_minus_Srec) ? (1.0f/lR_off) : (1.0f/r_minus_Srec);
            float u = 1.0f / r_plus_Srec;
            float l2 = l*l, u2 = u*u;
            float r2_inv = invR * invR;

            float t3 = 0.125f * (1.0f + recS*recS*r2_inv) * (l2 - u2)
                     + 0.25f * logf(u/l) * r2_inv;

            float de = lBF * t3 * invR;

            int ligGlobal = gs + li;
            int ligParticle = particleIndices[ligGlobal];
            atomicAdd(&forceBuffer[ligParticle], static_cast<unsigned long long>((long long)(de * dx * 0x100000000)));
            atomicAdd(&forceBuffer[ligParticle + paddedNumAtoms], static_cast<unsigned long long>((long long)(de * dy * 0x100000000)));
            atomicAdd(&forceBuffer[ligParticle + 2*paddedNumAtoms], static_cast<unsigned long long>((long long)(de * dz * 0x100000000)));
        }

        __syncwarp();
    }
}

/**
 * Cross-term GBSA grid runtime evaluation (GRID mode augment).
 *
 * For each ligand atom, looks up the precomputed scalar field G_b(r) where
 * b is the per-template-atom bin (b = template atom index). Adds
 *     U_cross_i = prefactor * q_i * G_b(r_i)
 * to the per-group energy and
 *     F_cross_i = -prefactor * q_i * grad G_b(r_i)
 * to the per-group force buffer. Uses local trilinear interpolation with
 * analytic gradient so no external interpolation helper is needed.
 *
 * Bin-major grid layout: crossTermGrid[b * totalGridPoints + gridIdx].
 *
 * One thread per ligand atom. Receptor data is NOT touched at runtime —
 * all receptor dependence is already baked into the scalar field.
 */
extern "C" __global__ void computeCrossTermFromGrid(
    const float4* __restrict__ posq,
    const int* __restrict__ particleIndices,
    const float* __restrict__ charges,
    const int* __restrict__ groupStart,
    int numGroups,
    int templateNumAtoms,
    float prefactor,
    const float* __restrict__ crossTermGrid,
    int totalGridPoints,
    float originX, float originY, float originZ,
    const int* __restrict__ gridCounts,
    float gridSpacing,
    unsigned long long* __restrict__ forceBuffer,
    float* __restrict__ groupEnergies,
    float* __restrict__ groupCrossTermEnergies,
    int paddedNumAtoms,
    float globalScalingFactor,
    const float* __restrict__ groupScalingFactors,
    float* __restrict__ groupUnscaledEnergies
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Find which group this atom belongs to (same pattern as the rest of
    // the ligand-side kernels in this file).
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

    // Bin = template atom index (per-atom binning: each ligand template
    // atom has its own slice of the cross-term grid).
    int binIdx = atomInGroup % templateNumAtoms;
    const float* G = crossTermGrid + binIdx * totalGridPoints;

    float scale = globalScalingFactor * groupScalingFactors[groupIdx];

    int particleIdx = particleIndices[idx];
    float4 pos = posq[particleIdx];
    float q_i = charges[binIdx];

    // --- Trilinear interpolation + analytic gradient on G[binIdx] ---
    int nx = gridCounts[0];
    int ny = gridCounts[1];
    int nz = gridCounts[2];
    int nyz = ny * nz;

    float fx = (pos.x - originX) / gridSpacing;
    float fy = (pos.y - originY) / gridSpacing;
    float fz = (pos.z - originZ) / gridSpacing;

    // If the atom is outside the grid, skip (silent no-op; the grid
    // extent should cover the binding site margin). Caller must ensure
    // the grid spans all sampled ligand positions.
    if (fx < 0.0f || fy < 0.0f || fz < 0.0f ||
        fx >= (float)(nx - 1) ||
        fy >= (float)(ny - 1) ||
        fz >= (float)(nz - 1)) {
        return;
    }


    int ix = (int)fx;
    int iy = (int)fy;
    int iz = (int)fz;
    float tx = fx - (float)ix;
    float ty = fy - (float)iy;
    float tz = fz - (float)iz;

    // 8 cube corners: G(ix+a, iy+b, iz+c), a,b,c in {0,1}
    int base = ix * nyz + iy * nz + iz;
    float c000 = G[base];
    float c001 = G[base + 1];
    float c010 = G[base + nz];
    float c011 = G[base + nz + 1];
    float c100 = G[base + nyz];
    float c101 = G[base + nyz + 1];
    float c110 = G[base + nyz + nz];
    float c111 = G[base + nyz + nz + 1];

    // Interpolated value: trilinear blend
    float w000 = (1.0f - tx) * (1.0f - ty) * (1.0f - tz);
    float w001 = (1.0f - tx) * (1.0f - ty) * tz;
    float w010 = (1.0f - tx) * ty         * (1.0f - tz);
    float w011 = (1.0f - tx) * ty         * tz;
    float w100 = tx          * (1.0f - ty) * (1.0f - tz);
    float w101 = tx          * (1.0f - ty) * tz;
    float w110 = tx          * ty         * (1.0f - tz);
    float w111 = tx          * ty         * tz;

    float Gval = w000 * c000 + w001 * c001 + w010 * c010 + w011 * c011
               + w100 * c100 + w101 * c101 + w110 * c110 + w111 * c111;

    // Analytic gradient of trilinear interpolant (dG/dfx, etc. are in
    // units of grid cells, convert to /nm by dividing by gridSpacing).
    float dG_dfx = (1.0f - ty) * (1.0f - tz) * (c100 - c000)
                 + (1.0f - ty) * tz         * (c101 - c001)
                 + ty         * (1.0f - tz) * (c110 - c010)
                 + ty         * tz         * (c111 - c011);
    float dG_dfy = (1.0f - tx) * (1.0f - tz) * (c010 - c000)
                 + (1.0f - tx) * tz         * (c011 - c001)
                 + tx         * (1.0f - tz) * (c110 - c100)
                 + tx         * tz         * (c111 - c101);
    float dG_dfz = (1.0f - tx) * (1.0f - ty) * (c001 - c000)
                 + (1.0f - tx) * ty         * (c011 - c010)
                 + tx         * (1.0f - ty) * (c101 - c100)
                 + tx         * ty         * (c111 - c110);
    float invSpacing = 1.0f / gridSpacing;
    float dG_dx = dG_dfx * invSpacing;
    float dG_dy = dG_dfy * invSpacing;
    float dG_dz = dG_dfz * invSpacing;

    // Energy contribution (unscaled vs scaled handled like the rest of
    // the group energy accumulators in this file).
    float e_i = prefactor * q_i * Gval;


    // Force = -dE/dr = -prefactor * q_i * grad G, scaled alchemically
    float fxx = -prefactor * q_i * dG_dx * scale;
    float fyy = -prefactor * q_i * dG_dy * scale;
    float fzz = -prefactor * q_i * dG_dz * scale;

    atomicAdd(&forceBuffer[particleIdx],
              static_cast<unsigned long long>(
                  (long long)(fxx * 0x100000000)));
    atomicAdd(&forceBuffer[particleIdx + paddedNumAtoms],
              static_cast<unsigned long long>(
                  (long long)(fyy * 0x100000000)));
    atomicAdd(&forceBuffer[particleIdx + 2 * paddedNumAtoms],
              static_cast<unsigned long long>(
                  (long long)(fzz * 0x100000000)));

    atomicAdd(&groupEnergies[groupIdx], e_i * scale);
    if (groupCrossTermEnergies != 0) {
        atomicAdd(&groupCrossTermEnergies[groupIdx], e_i * scale);
    }
    if (groupUnscaledEnergies != 0) {
        atomicAdd(&groupUnscaledEnergies[groupIdx],
                  e_i * globalScalingFactor);
    }
}
