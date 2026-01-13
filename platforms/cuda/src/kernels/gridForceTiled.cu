/**
 * CUDA implementation of tiled grid force calculation.
 * Kernel for computing forces using tile-based grid storage.
 */

#include "include/InterpolationBasis.cuh"

// Tile configuration constants (must match TileConfig in TileManager.h)
#define TILE_SIZE 64
#define TILE_OVERLAP 4
#define TILE_WITH_OVERLAP (TILE_SIZE + 2 * TILE_OVERLAP)  // 72

/**
 * Find which tile contains a grid position.
 * Returns tile index or -1 if not found.
 */
__device__ int findTileForPosition(
    int gridX, int gridY, int gridZ,
    const int* __restrict__ tileOffsets,  // x,y,z for each tile
    int numTiles
) {
    // Linear search for now - works for small number of tiles
    // TODO: Use spatial hash map for many tiles
    for (int t = 0; t < numTiles; t++) {
        int tileStartX = tileOffsets[t * 3 + 0];
        int tileStartY = tileOffsets[t * 3 + 1];
        int tileStartZ = tileOffsets[t * 3 + 2];

        // Check if grid position falls within this tile's core region
        // (The tile stores TILE_WITH_OVERLAP points, but its "ownership" is TILE_SIZE)
        if (gridX >= tileStartX && gridX < tileStartX + TILE_SIZE &&
            gridY >= tileStartY && gridY < tileStartY + TILE_SIZE &&
            gridZ >= tileStartZ && gridZ < tileStartZ + TILE_SIZE) {
            return t;
        }
    }
    return -1;
}

/**
 * Trilinear interpolation using tile-local data.
 */
__device__ void trilinearInterpolateTiled(
    const float* __restrict__ tileValues,
    int localX, int localY, int localZ,  // Position in tile coordinates (including overlap offset)
    float fx, float fy, float fz,        // Fractional position within cell
    float* energy,
    float* dEdx, float* dEdy, float* dEdz,
    float spacingX, float spacingY, float spacingZ
) {
    // Tile dimensions
    const int tileNY = TILE_WITH_OVERLAP;
    const int tileNZ = TILE_WITH_OVERLAP;
    const int tileNYZ = tileNY * tileNZ;

    // Get 8 corner values from tile
    float v000 = tileValues[localX * tileNYZ + localY * tileNZ + localZ];
    float v001 = tileValues[localX * tileNYZ + localY * tileNZ + (localZ + 1)];
    float v010 = tileValues[localX * tileNYZ + (localY + 1) * tileNZ + localZ];
    float v011 = tileValues[localX * tileNYZ + (localY + 1) * tileNZ + (localZ + 1)];
    float v100 = tileValues[(localX + 1) * tileNYZ + localY * tileNZ + localZ];
    float v101 = tileValues[(localX + 1) * tileNYZ + localY * tileNZ + (localZ + 1)];
    float v110 = tileValues[(localX + 1) * tileNYZ + (localY + 1) * tileNZ + localZ];
    float v111 = tileValues[(localX + 1) * tileNYZ + (localY + 1) * tileNZ + (localZ + 1)];

    // Trilinear interpolation
    float fx1 = 1.0f - fx;
    float fy1 = 1.0f - fy;
    float fz1 = 1.0f - fz;

    *energy = fx1 * (fy1 * (fz1 * v000 + fz * v001) + fy * (fz1 * v010 + fz * v011)) +
              fx  * (fy1 * (fz1 * v100 + fz * v101) + fy * (fz1 * v110 + fz * v111));

    // Compute gradients
    *dEdx = (fy1 * (fz1 * (v100 - v000) + fz * (v101 - v001)) +
             fy  * (fz1 * (v110 - v010) + fz * (v111 - v011))) / spacingX;
    *dEdy = (fx1 * (fz1 * (v010 - v000) + fz * (v011 - v001)) +
             fx  * (fz1 * (v110 - v100) + fz * (v111 - v101))) / spacingY;
    *dEdz = (fx1 * (fy1 * (v001 - v000) + fy * (v011 - v010)) +
             fx  * (fy1 * (v101 - v100) + fy * (v111 - v110))) / spacingZ;
}

/**
 * Main tiled grid force kernel.
 * Computes forces using tile-based grid storage.
 */
extern "C" __global__ void computeGridForceTiled(
    const float4* __restrict__ posq,
    unsigned long long* __restrict__ forceBuffers,
    const int* __restrict__ gridCounts,        // Full grid dimensions
    const float* __restrict__ gridSpacing,
    const float* __restrict__ scalingFactors,
    const float invPower,
    const int invPowerMode,
    const int interpolationMethod,
    const float outOfBoundsK,
    const float originX,
    const float originY,
    const float originZ,
    float* __restrict__ energyBuffer,
    const int numAtoms,
    const int paddedNumAtoms,
    const int* __restrict__ particleIndices,
    const int* __restrict__ particleToGroupMap,
    float* __restrict__ groupEnergyBuffer,
    float* __restrict__ atomEnergyBuffer,
    const int numGroups,
    // Tile-specific parameters
    const int* __restrict__ tileOffsets,           // Grid offsets for each tile (x,y,z,x,y,z,...)
    const unsigned long long* __restrict__ tileValuePtrs,   // Device pointers to tile values
    const unsigned long long* __restrict__ tileDerivPtrs,   // Device pointers to tile derivatives
    const int numTiles
) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= numAtoms) return;

    const unsigned int particleIndex = (particleIndices != nullptr) ? particleIndices[index] : index;

    float4 posOrig = posq[particleIndex];
    float scalingFactor = scalingFactors[particleIndex];

    // Transform position to grid coordinates
    float3 pos;
    pos.x = posOrig.x - originX;
    pos.y = posOrig.y - originY;
    pos.z = posOrig.z - originZ;

    float3 atomForce = make_float3(0.0f, 0.0f, 0.0f);
    float threadEnergy = 0.0f;

    // Calculate grid boundaries
    float3 gridCorner;
    gridCorner.x = gridSpacing[0] * (gridCounts[0] - 1);
    gridCorner.y = gridSpacing[1] * (gridCounts[1] - 1);
    gridCorner.z = gridSpacing[2] * (gridCounts[2] - 1);

    bool isInside = (pos.x >= 0.0f && pos.x <= gridCorner.x &&
                     pos.y >= 0.0f && pos.y <= gridCorner.y &&
                     pos.z >= 0.0f && pos.z <= gridCorner.z);

    if (isInside && scalingFactor != 0.0f) {
        // Calculate grid indices
        int ix = min(max((int)(pos.x / gridSpacing[0]), 0), gridCounts[0] - 2);
        int iy = min(max((int)(pos.y / gridSpacing[1]), 0), gridCounts[1] - 2);
        int iz = min(max((int)(pos.z / gridSpacing[2]), 0), gridCounts[2] - 2);

        // Calculate fractional position
        float fx = (pos.x / gridSpacing[0]) - ix;
        float fy = (pos.y / gridSpacing[1]) - iy;
        float fz = (pos.z / gridSpacing[2]) - iz;
        fx = min(max(fx, 0.0f), 1.0f);
        fy = min(max(fy, 0.0f), 1.0f);
        fz = min(max(fz, 0.0f), 1.0f);

        // Find which tile contains this grid position
        int tileIdx = findTileForPosition(ix, iy, iz, tileOffsets, numTiles);

        if (tileIdx >= 0) {
            // Get tile data pointer
            const float* tileValues = (const float*)tileValuePtrs[tileIdx];

            // Convert global grid coordinates to tile-local coordinates
            // Add TILE_OVERLAP to account for the overlap region at the start
            int tileStartX = tileOffsets[tileIdx * 3 + 0];
            int tileStartY = tileOffsets[tileIdx * 3 + 1];
            int tileStartZ = tileOffsets[tileIdx * 3 + 2];

            int localX = (ix - tileStartX) + TILE_OVERLAP;
            int localY = (iy - tileStartY) + TILE_OVERLAP;
            int localZ = (iz - tileStartZ) + TILE_OVERLAP;

            float interpolated = 0.0f;
            float dx = 0.0f, dy = 0.0f, dz = 0.0f;

            if (interpolationMethod == 0) {
                // Trilinear interpolation
                trilinearInterpolateTiled(
                    tileValues, localX, localY, localZ,
                    fx, fy, fz,
                    &interpolated, &dx, &dy, &dz,
                    gridSpacing[0], gridSpacing[1], gridSpacing[2]
                );
            }
            // TODO: Add tricubic and triquintic tiled interpolation

            // Apply inv_power transformation if RUNTIME mode
            if (invPowerMode == 1 && invPower != 0.0f) {
                float p = 1.0f / invPower;
                if (interpolated > 0.0f) {
                    float U = interpolated;
                    float V = powf(U, p);
                    float dVdU = p * powf(U, p - 1.0f);
                    interpolated = V;
                    dx *= dVdU;
                    dy *= dVdU;
                    dz *= dVdU;
                } else if (interpolated < 0.0f) {
                    float U = -interpolated;
                    float V = -powf(U, p);
                    float dVdU = p * powf(U, p - 1.0f);
                    interpolated = V;
                    dx *= dVdU;
                    dy *= dVdU;
                    dz *= dVdU;
                }
            }

            // Apply scaling factor and compute energy/force
            threadEnergy = scalingFactor * interpolated;
            atomForce.x = -scalingFactor * dx;
            atomForce.y = -scalingFactor * dy;
            atomForce.z = -scalingFactor * dz;
        }
    } else if (!isInside && outOfBoundsK > 0.0f) {
        // Out-of-bounds restraint (same as non-tiled kernel)
        float restraintEnergy = 0.0f;
        if (pos.x < 0.0f) {
            restraintEnergy += outOfBoundsK * pos.x * pos.x;
            atomForce.x = 2.0f * outOfBoundsK * pos.x;
        } else if (pos.x > gridCorner.x) {
            float dx = pos.x - gridCorner.x;
            restraintEnergy += outOfBoundsK * dx * dx;
            atomForce.x = -2.0f * outOfBoundsK * dx;
        }
        if (pos.y < 0.0f) {
            restraintEnergy += outOfBoundsK * pos.y * pos.y;
            atomForce.y = 2.0f * outOfBoundsK * pos.y;
        } else if (pos.y > gridCorner.y) {
            float dy = pos.y - gridCorner.y;
            restraintEnergy += outOfBoundsK * dy * dy;
            atomForce.y = -2.0f * outOfBoundsK * dy;
        }
        if (pos.z < 0.0f) {
            restraintEnergy += outOfBoundsK * pos.z * pos.z;
            atomForce.z = 2.0f * outOfBoundsK * pos.z;
        } else if (pos.z > gridCorner.z) {
            float dz = pos.z - gridCorner.z;
            restraintEnergy += outOfBoundsK * dz * dz;
            atomForce.z = -2.0f * outOfBoundsK * dz;
        }
        threadEnergy = restraintEnergy;
    }

    // Store per-atom energy if buffer provided
    if (atomEnergyBuffer != nullptr) {
        atomEnergyBuffer[index] = threadEnergy;
    }

    // Accumulate energy to group buffer if using particle groups
    if (particleToGroupMap != nullptr && groupEnergyBuffer != nullptr) {
        int groupIdx = particleToGroupMap[index];
        if (groupIdx >= 0 && groupIdx < numGroups) {
            atomicAdd(&groupEnergyBuffer[groupIdx], threadEnergy);
        }
    }

    // Accumulate total energy
    atomicAdd(energyBuffer, threadEnergy);

    // Convert force to fixed point and accumulate
    atomicAdd(&forceBuffers[particleIndex], static_cast<unsigned long long>((long long)(atomForce.x * 0x100000000)));
    atomicAdd(&forceBuffers[particleIndex + paddedNumAtoms], static_cast<unsigned long long>((long long)(atomForce.y * 0x100000000)));
    atomicAdd(&forceBuffers[particleIndex + 2 * paddedNumAtoms], static_cast<unsigned long long>((long long)(atomForce.z * 0x100000000)));
}
