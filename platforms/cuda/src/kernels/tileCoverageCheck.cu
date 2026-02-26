/**
 * GPU-side tile coverage check kernel.
 * Checks if all particles are within currently-loaded tiles by looking up
 * a compact occupancy bitmap. Sets a flag if any particle needs a tile
 * that isn't loaded, triggering CPU-side retiling.
 */

extern "C" __global__ void checkTileCoverage(
    const real4* __restrict__ posq,
    const int* __restrict__ particleIndices,
    int numParticles,
    int paddedNumAtoms,
    float originX, float originY, float originZ,
    float invSpacingX, float invSpacingY, float invSpacingZ,
    int tileSize,
    const unsigned char* __restrict__ tileOccupancy,
    int numTilesX, int numTilesY, int numTilesZ,
    int* __restrict__ needsRetile
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticles) return;

    int atomIdx = (particleIndices != NULL) ? particleIndices[idx] : idx;
    real4 pos = posq[atomIdx];

    // Position -> grid index -> tile index
    float gx = (pos.x - originX) * invSpacingX;
    float gy = (pos.y - originY) * invSpacingY;
    float gz = (pos.z - originZ) * invSpacingZ;

    int tx = (int)(gx / tileSize);
    int ty = (int)(gy / tileSize);
    int tz = (int)(gz / tileSize);

    // Clamp to valid tile range
    tx = max(0, min(tx, numTilesX - 1));
    ty = max(0, min(ty, numTilesY - 1));
    tz = max(0, min(tz, numTilesZ - 1));

    int tileIdx = tx * numTilesY * numTilesZ + ty * numTilesZ + tz;
    if (tileOccupancy[tileIdx] == 0) {
        atomicMax(needsRetile, 1);
    }
}
