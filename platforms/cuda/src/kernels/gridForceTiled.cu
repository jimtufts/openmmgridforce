/**
 * CUDA implementation of tiled grid force calculation.
 * Kernel for computing forces using tile-based grid storage.
 */

#include "include/InterpolationBasis.cuh"
#include "include/TricubicCoefficients.cuh"
#include "include/TriquinticCoefficients.cuh"
#include "include/InvPowerChainRule.cuh"

/**
 * Find which tile contains a grid position.
 * Returns tile index or -1 if not found.
 */
__device__ int findTileForPosition(
    int gridX, int gridY, int gridZ,
    const int* __restrict__ tileOffsets,  // x,y,z for each tile
    int numTiles,
    int tileSize  // Core tile size (excluding overlap)
) {
    // Linear search for now - works for small number of tiles
    // TODO: Use spatial hash map for many tiles
    for (int t = 0; t < numTiles; t++) {
        int tileStartX = tileOffsets[t * 3 + 0];
        int tileStartY = tileOffsets[t * 3 + 1];
        int tileStartZ = tileOffsets[t * 3 + 2];

        // Check if grid position falls within this tile's core region
        // (The tile stores tileSize + 2*overlap points, but its "ownership" is tileSize)
        if (gridX >= tileStartX && gridX < tileStartX + tileSize &&
            gridY >= tileStartY && gridY < tileStartY + tileSize &&
            gridZ >= tileStartZ && gridZ < tileStartZ + tileSize) {
            return t;
        }
    }
    return -1;
}

/**
 * Helper to compute tile-local linear index.
 */
__device__ __forceinline__ int tileIndex(int lx, int ly, int lz, int tileWithOverlap) {
    return lx * tileWithOverlap * tileWithOverlap + ly * tileWithOverlap + lz;
}

/**
 * Helper to apply inv_power forward transform: val -> sign(val) * |val|^(1/n)
 */
__device__ __forceinline__ float invPowerForwardTransform(float val, float invN) {
    if (fabsf(val) >= 1e-10f) {
        return (val >= 0.0f ? 1.0f : -1.0f) * powf(fabsf(val), invN);
    }
    return 0.0f;
}

/**
 * Trilinear interpolation using tile-local data.
 * For RUNTIME inv_power mode, transforms grid values before interpolation
 * and returns interpolated value in transformed space.
 */
__device__ void trilinearInterpolateTiled(
    const float* __restrict__ tileValues,
    int localX, int localY, int localZ,  // Position in tile coordinates (including overlap offset)
    float fx, float fy, float fz,        // Fractional position within cell
    float* energy,
    float* dEdx, float* dEdy, float* dEdz,
    float spacingX, float spacingY, float spacingZ,
    int tileWithOverlap,  // Total tile dimension including overlap
    float invPower,       // inv_power value
    int invPowerMode      // 0=NONE, 1=RUNTIME, 2=STORED
) {
    // Get 8 corner values from tile
    float v000 = tileValues[tileIndex(localX, localY, localZ, tileWithOverlap)];
    float v001 = tileValues[tileIndex(localX, localY, localZ + 1, tileWithOverlap)];
    float v010 = tileValues[tileIndex(localX, localY + 1, localZ, tileWithOverlap)];
    float v011 = tileValues[tileIndex(localX, localY + 1, localZ + 1, tileWithOverlap)];
    float v100 = tileValues[tileIndex(localX + 1, localY, localZ, tileWithOverlap)];
    float v101 = tileValues[tileIndex(localX + 1, localY, localZ + 1, tileWithOverlap)];
    float v110 = tileValues[tileIndex(localX + 1, localY + 1, localZ, tileWithOverlap)];
    float v111 = tileValues[tileIndex(localX + 1, localY + 1, localZ + 1, tileWithOverlap)];

    // RUNTIME mode: transform corner values BEFORE interpolation
    // val -> sign(val) * |val|^(1/n)
    if (invPowerMode == 1 && invPower != 0.0f) {
        float invN = 1.0f / invPower;
        v000 = invPowerForwardTransform(v000, invN);
        v001 = invPowerForwardTransform(v001, invN);
        v010 = invPowerForwardTransform(v010, invN);
        v011 = invPowerForwardTransform(v011, invN);
        v100 = invPowerForwardTransform(v100, invN);
        v101 = invPowerForwardTransform(v101, invN);
        v110 = invPowerForwardTransform(v110, invN);
        v111 = invPowerForwardTransform(v111, invN);
    }

    // Trilinear interpolation
    float fx1 = 1.0f - fx;
    float fy1 = 1.0f - fy;
    float fz1 = 1.0f - fz;

    *energy = fx1 * (fy1 * (fz1 * v000 + fz * v001) + fy * (fz1 * v010 + fz * v011)) +
              fx  * (fy1 * (fz1 * v100 + fz * v101) + fy * (fz1 * v110 + fz * v111));

    // Compute gradients in transformed space (don't divide by spacing yet for RUNTIME mode)
    if (invPowerMode == 1 && invPower != 0.0f) {
        // For RUNTIME mode, compute gradients WITHOUT dividing by spacing
        // (chain rule will be applied later, then divide by spacing)
        *dEdx = (fy1 * (fz1 * (v100 - v000) + fz * (v101 - v001)) +
                 fy  * (fz1 * (v110 - v010) + fz * (v111 - v011)));
        *dEdy = (fx1 * (fz1 * (v010 - v000) + fz * (v011 - v001)) +
                 fx  * (fz1 * (v110 - v100) + fz * (v111 - v101)));
        *dEdz = (fx1 * (fy1 * (v001 - v000) + fy * (v011 - v010)) +
                 fx  * (fy1 * (v101 - v100) + fy * (v111 - v110)));
    } else {
        // For NONE/STORED mode, divide by spacing immediately
        *dEdx = (fy1 * (fz1 * (v100 - v000) + fz * (v101 - v001)) +
                 fy  * (fz1 * (v110 - v010) + fz * (v111 - v011))) / spacingX;
        *dEdy = (fx1 * (fz1 * (v010 - v000) + fz * (v011 - v001)) +
                 fx  * (fz1 * (v110 - v100) + fz * (v111 - v101))) / spacingY;
        *dEdz = (fx1 * (fy1 * (v001 - v000) + fy * (v011 - v010)) +
                 fx  * (fy1 * (v101 - v100) + fy * (v111 - v110))) / spacingZ;
    }
}

/**
 * B-spline interpolation using tile-local data (4x4x4 stencil).
 */
__device__ void bsplineInterpolateTiled(
    const float* __restrict__ tileValues,
    int localX, int localY, int localZ,  // Position in tile coordinates (center of stencil)
    float fx, float fy, float fz,        // Fractional position within cell
    float* energy,
    float* dEdx, float* dEdy, float* dEdz,
    float spacingX, float spacingY, float spacingZ,
    int tileWithOverlap,
    float invPower,
    int invPowerMode
) {
    // Precompute basis functions
    float bx[4] = {bspline_basis0(fx), bspline_basis1(fx), bspline_basis2(fx), bspline_basis3(fx)};
    float by[4] = {bspline_basis0(fy), bspline_basis1(fy), bspline_basis2(fy), bspline_basis3(fy)};
    float bz[4] = {bspline_basis0(fz), bspline_basis1(fz), bspline_basis2(fz), bspline_basis3(fz)};

    float dbx[4] = {bspline_deriv0(fx), bspline_deriv1(fx), bspline_deriv2(fx), bspline_deriv3(fx)};
    float dby[4] = {bspline_deriv0(fy), bspline_deriv1(fy), bspline_deriv2(fy), bspline_deriv3(fy)};
    float dbz[4] = {bspline_deriv0(fz), bspline_deriv1(fz), bspline_deriv2(fz), bspline_deriv3(fz)};

    float interpolated = 0.0f;
    float dvdx = 0.0f, dvdy = 0.0f, dvdz = 0.0f;

    // B-spline uses a 4x4x4 stencil centered at (localX-1, localY-1, localZ-1) to (localX+2, localY+2, localZ+2)
    for (int i = 0; i < 4; i++) {
        int lx = localX - 1 + i;
        // Clamp to tile bounds
        lx = min(max(lx, 0), tileWithOverlap - 1);

        for (int j = 0; j < 4; j++) {
            int ly = localY - 1 + j;
            ly = min(max(ly, 0), tileWithOverlap - 1);

            for (int k = 0; k < 4; k++) {
                int lz = localZ - 1 + k;
                lz = min(max(lz, 0), tileWithOverlap - 1);

                float val = tileValues[tileIndex(lx, ly, lz, tileWithOverlap)];

                // Apply RUNTIME inv_power transformation before interpolation
                if (invPowerMode == 1 && invPower != 0.0f) {
                    float invN = 1.0f / invPower;
                    if (fabsf(val) >= 1e-10f) {
                        val = (val >= 0.0f ? 1.0f : -1.0f) * powf(fabsf(val), invN);
                    } else {
                        val = 0.0f;
                    }
                }

                float weight = bx[i] * by[j] * bz[k];
                interpolated += weight * val;
                dvdx += dbx[i] * by[j] * bz[k] * val;
                dvdy += bx[i] * dby[j] * bz[k] * val;
                dvdz += bx[i] * by[j] * dbz[k] * val;
            }
        }
    }

    *energy = interpolated;
    *dEdx = dvdx / spacingX;
    *dEdy = dvdy / spacingY;
    *dEdz = dvdz / spacingZ;
}

/**
 * Tricubic (Lekien-Marsden) interpolation using tile-local data.
 * Requires precomputed derivatives stored in tileDerivatives.
 */
__device__ void tricubicInterpolateTiled(
    const float* __restrict__ tileValues,
    const float* __restrict__ tileDerivatives,
    int localX, int localY, int localZ,
    float fx, float fy, float fz,
    float* energy,
    float* dEdx, float* dEdy, float* dEdz,
    float spacingX, float spacingY, float spacingZ,
    int tileWithOverlap,
    float invPower,
    int invPowerMode
) {
    int tilePoints = tileWithOverlap * tileWithOverlap * tileWithOverlap;

    // 8 corners of the cell
    int corners[8][3] = {
        {localX, localY, localZ}, {localX+1, localY, localZ},
        {localX, localY+1, localZ}, {localX+1, localY+1, localZ},
        {localX, localY, localZ+1}, {localX+1, localY, localZ+1},
        {localX, localY+1, localZ+1}, {localX+1, localY+1, localZ+1}
    };

    // Storage: X[deriv*8 + corner] - DERIVATIVE-MAJOR (matches RASPA3/Lekien-Marsden)
    // Derivatives: 0=f, 1=fx, 2=fy, 3=fz, 4=fxy, 5=fxz, 6=fyz, 7=fxyz
    float X[64];

    // Map tricubic derivative order to gridDerivatives storage order
    // Tricubic needs: 0=f, 1=fx, 2=fy, 3=fz, 4=fxy, 5=fxz, 6=fyz, 7=fxyz
    // gridDerivatives (RASPA3 order): 0=f, 1=dx, 2=dy, 3=dz, 4=dxx, 5=dxy, 6=dxz, 7=dyy, 8=dyz, 9=dzz, ..., 13=dxyz
    const int derivMap[8] = {0, 1, 2, 3, 5, 6, 8, 13};

    if (invPowerMode == 1 && invPower != 0.0f) {
        // RUNTIME mode: transform all 27 derivatives per corner, then extract needed 8
        float p = 1.0f / invPower;
        for (int c = 0; c < 8; c++) {
            int point_idx = tileIndex(corners[c][0], corners[c][1], corners[c][2], tileWithOverlap);
            float U_derivs[27], V_derivs[27];
            for (int d = 0; d < 27; d++) {
                U_derivs[d] = tileDerivatives[d * tilePoints + point_idx];
            }
            applyInvPowerChainRule(U_derivs, p, V_derivs);
            for (int d = 0; d < 8; d++) {
                X[d*8 + c] = V_derivs[derivMap[d]];
            }
        }
    } else {
        // STORED or NONE mode: load directly
        for (int d = 0; d < 8; d++) {
            for (int c = 0; c < 8; c++) {
                int point_idx = tileIndex(corners[c][0], corners[c][1], corners[c][2], tileWithOverlap);
                X[d*8 + c] = tileDerivatives[derivMap[d] * tilePoints + point_idx];
            }
        }
    }

    // Multiply X by TRICUBIC_COEFFICIENTS matrix to get polynomial coefficients
    float a[64];
    for (int i = 0; i < 64; i++) {
        a[i] = 0.0f;
        for (int j = 0; j < 64; j++) {
            a[i] += TRICUBIC_COEFFICIENTS[i][j] * X[j];
        }
    }

    // Evaluate tricubic polynomial at (fx, fy, fz)
    float interpolated = 0.0f;
    float dx = 0.0f, dy = 0.0f, dz = 0.0f;

    for (int k = 0; k < 4; k++) {
        float fz_pow_k = (k == 0) ? 1.0f : (k == 1) ? fz : (k == 2) ? fz*fz : fz*fz*fz;
        float fz_pow_k_deriv = (k == 0) ? 0.0f : (k == 1) ? 1.0f : (k == 2) ? 2.0f*fz : 3.0f*fz*fz;

        for (int j = 0; j < 4; j++) {
            float fy_pow_j = (j == 0) ? 1.0f : (j == 1) ? fy : (j == 2) ? fy*fy : fy*fy*fy;
            float fy_pow_j_deriv = (j == 0) ? 0.0f : (j == 1) ? 1.0f : (j == 2) ? 2.0f*fy : 3.0f*fy*fy;

            for (int i = 0; i < 4; i++) {
                float fx_pow_i = (i == 0) ? 1.0f : (i == 1) ? fx : (i == 2) ? fx*fx : fx*fx*fx;
                float fx_pow_i_deriv = (i == 0) ? 0.0f : (i == 1) ? 1.0f : (i == 2) ? 2.0f*fx : 3.0f*fx*fx;

                float coeff = a[i + 4*j + 16*k];

                interpolated += coeff * fx_pow_i * fy_pow_j * fz_pow_k;
                dx += coeff * fx_pow_i_deriv * fy_pow_j * fz_pow_k;
                dy += coeff * fx_pow_i * fy_pow_j_deriv * fz_pow_k;
                dz += coeff * fx_pow_i * fy_pow_j * fz_pow_k_deriv;
            }
        }
    }

    *energy = interpolated;
    *dEdx = dx / spacingX;
    *dEdy = dy / spacingY;
    *dEdz = dz / spacingZ;
}

/**
 * Triquintic Hermite interpolation using tile-local data.
 * Requires precomputed derivatives stored in tileDerivatives (27 per point).
 */
__device__ void triquinticInterpolateTiled(
    const float* __restrict__ tileValues,
    const float* __restrict__ tileDerivatives,
    int localX, int localY, int localZ,
    float fx, float fy, float fz,
    float* energy,
    float* dEdx, float* dEdy, float* dEdz,
    float spacingX, float spacingY, float spacingZ,
    int tileWithOverlap,
    float invPower,
    int invPowerMode
) {
    int tilePoints = tileWithOverlap * tileWithOverlap * tileWithOverlap;

    // 8 corners of the cell
    int corners[8][3] = {
        {localX, localY, localZ}, {localX+1, localY, localZ},
        {localX, localY+1, localZ}, {localX+1, localY+1, localZ},
        {localX, localY, localZ+1}, {localX+1, localY, localZ+1},
        {localX, localY+1, localZ+1}, {localX+1, localY+1, localZ+1}
    };

    // Gather derivatives in DERIVATIVE-MAJOR layout: X[deriv_idx * 8 + corner_idx]
    float X[216];
    if (invPowerMode == 1 && invPower != 0.0f) {
        // RUNTIME mode: transform all 27 derivatives per corner
        float p = 1.0f / invPower;
        for (int c = 0; c < 8; c++) {
            int point_idx = tileIndex(corners[c][0], corners[c][1], corners[c][2], tileWithOverlap);
            float U_derivs[27], V_derivs[27];
            for (int d = 0; d < 27; d++) {
                U_derivs[d] = tileDerivatives[d * tilePoints + point_idx];
            }
            applyInvPowerChainRule(U_derivs, p, V_derivs);
            for (int d = 0; d < 27; d++) {
                X[d * 8 + c] = V_derivs[d];
            }
        }
    } else {
        // STORED or NONE mode: load directly
        for (int d = 0; d < 27; d++) {
            for (int c = 0; c < 8; c++) {
                int point_idx = tileIndex(corners[c][0], corners[c][1], corners[c][2], tileWithOverlap);
                X[d * 8 + c] = tileDerivatives[d * tilePoints + point_idx];
            }
        }
    }

    // Compute polynomial coefficients: a = 0.125 * TRIQUINTIC_COEFFICIENTS * X
    float a[216];
    const float scale = 0.125f;
    for (int i = 0; i < 216; i++) {
        a[i] = 0.0f;
        for (int j = 0; j < 216; j++) {
            a[i] += TRIQUINTIC_COEFFICIENTS[i][j] * X[j];
        }
        a[i] *= scale;
    }

    // Precompute powers of local coordinates
    float sx_pow[6], sy_pow[6], sz_pow[6];
    sx_pow[0] = sy_pow[0] = sz_pow[0] = 1.0f;
    for (int p = 1; p < 6; p++) {
        sx_pow[p] = sx_pow[p-1] * fx;
        sy_pow[p] = sy_pow[p-1] * fy;
        sz_pow[p] = sz_pow[p-1] * fz;
    }

    // Evaluate polynomial: sum over i,j,k of a[i+6j+36k] * fx^i * fy^j * fz^k
    float value = 0.0f;
    float dvalue_dx = 0.0f, dvalue_dy = 0.0f, dvalue_dz = 0.0f;

    for (int k = 0; k < 6; k++) {
        for (int j = 0; j < 6; j++) {
            for (int i = 0; i < 6; i++) {
                int coeff_idx = i + 6*j + 36*k;
                float coeff = a[coeff_idx];
                value += coeff * sx_pow[i] * sy_pow[j] * sz_pow[k];
                if (i > 0) dvalue_dx += coeff * i * sx_pow[i-1] * sy_pow[j] * sz_pow[k];
                if (j > 0) dvalue_dy += coeff * j * sx_pow[i] * sy_pow[j-1] * sz_pow[k];
                if (k > 0) dvalue_dz += coeff * k * sx_pow[i] * sy_pow[j] * sz_pow[k-1];
            }
        }
    }

    *energy = value;
    *dEdx = dvalue_dx / spacingX;
    *dEdy = dvalue_dy / spacingY;
    *dEdz = dvalue_dz / spacingZ;
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
    int* __restrict__ outOfBoundsBuffer,           // Per-atom out-of-bounds flags (null = don't store)
    const int numGroups,
    // Tile-specific parameters
    const int* __restrict__ tileOffsets,           // Grid offsets for each tile (x,y,z,x,y,z,...)
    const unsigned long long* __restrict__ tileValuePtrs,   // Device pointers to tile values
    const unsigned long long* __restrict__ tileDerivPtrs,   // Device pointers to tile derivatives
    const int numTiles,
    const int tileSize,                             // Core tile size (excluding overlap)
    const int tileOverlap                           // Overlap for interpolation stencil
) {
    // Compute tile dimensions
    const int tileWithOverlap = tileSize + 2 * tileOverlap;
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
        int tileIdx = findTileForPosition(ix, iy, iz, tileOffsets, numTiles, tileSize);

        if (tileIdx >= 0) {
            // Get tile data pointers
            const float* tileValues = (const float*)tileValuePtrs[tileIdx];
            const float* tileDerivatives = (const float*)tileDerivPtrs[tileIdx];

            // Convert global grid coordinates to tile-local coordinates
            // Add tileOverlap to account for the overlap region at the start
            int tileStartX = tileOffsets[tileIdx * 3 + 0];
            int tileStartY = tileOffsets[tileIdx * 3 + 1];
            int tileStartZ = tileOffsets[tileIdx * 3 + 2];

            int localX = (ix - tileStartX) + tileOverlap;
            int localY = (iy - tileStartY) + tileOverlap;
            int localZ = (iz - tileStartZ) + tileOverlap;

            float interpolated = 0.0f;
            float dx = 0.0f, dy = 0.0f, dz = 0.0f;

            if (interpolationMethod == 0) {
                // Trilinear interpolation (now handles inv_power internally for pre-transform)
                trilinearInterpolateTiled(
                    tileValues, localX, localY, localZ,
                    fx, fy, fz,
                    &interpolated, &dx, &dy, &dz,
                    gridSpacing[0], gridSpacing[1], gridSpacing[2],
                    tileWithOverlap,
                    invPower, invPowerMode
                );

                // Back-transform from transformed space if RUNTIME or STORED mode
                // val^(1/n) -> val^(1/n)^n = val
                if ((invPowerMode == 1 || invPowerMode == 2) && invPower != 0.0f) {
                    float sign = (interpolated >= 0.0f) ? 1.0f : -1.0f;
                    float absVal = fabsf(interpolated);
                    if (absVal > 1e-10f) {
                        // Back-convert: v -> sign(v)*|v|^n
                        float powerFactor = invPower * powf(absVal, invPower - 1.0f);
                        interpolated = sign * powf(absVal, invPower);
                        // Apply chain rule to gradients, then divide by spacing
                        dx = dx * powerFactor / gridSpacing[0];
                        dy = dy * powerFactor / gridSpacing[1];
                        dz = dz * powerFactor / gridSpacing[2];
                    }
                }
            } else if (interpolationMethod == 1) {
                // B-spline interpolation (handles inv_power pre-transform internally)
                bsplineInterpolateTiled(
                    tileValues, localX, localY, localZ,
                    fx, fy, fz,
                    &interpolated, &dx, &dy, &dz,
                    gridSpacing[0], gridSpacing[1], gridSpacing[2],
                    tileWithOverlap,
                    invPower, invPowerMode
                );

                // Back-transform from transformed space if RUNTIME or STORED mode
                // (bspline already divided by spacing, so just apply power factor to gradients)
                if ((invPowerMode == 1 || invPowerMode == 2) && invPower != 0.0f) {
                    float sign = (interpolated >= 0.0f) ? 1.0f : -1.0f;
                    float absVal = fabsf(interpolated);
                    if (absVal > 1e-10f) {
                        float powerFactor = invPower * powf(absVal, invPower - 1.0f);
                        interpolated = sign * powf(absVal, invPower);
                        dx *= powerFactor;
                        dy *= powerFactor;
                        dz *= powerFactor;
                    }
                }
            } else if (interpolationMethod == 2 && tileDerivatives != nullptr) {
                // Tricubic interpolation (handles inv_power pre-transform internally)
                tricubicInterpolateTiled(
                    tileValues, tileDerivatives, localX, localY, localZ,
                    fx, fy, fz,
                    &interpolated, &dx, &dy, &dz,
                    gridSpacing[0], gridSpacing[1], gridSpacing[2],
                    tileWithOverlap,
                    invPower, invPowerMode
                );

                // Back-transform from transformed space if RUNTIME or STORED mode
                if ((invPowerMode == 1 || invPowerMode == 2) && invPower != 0.0f) {
                    float sign = (interpolated >= 0.0f) ? 1.0f : -1.0f;
                    float absVal = fabsf(interpolated);
                    if (absVal > 1e-10f) {
                        float powerFactor = invPower * powf(absVal, invPower - 1.0f);
                        interpolated = sign * powf(absVal, invPower);
                        dx *= powerFactor;
                        dy *= powerFactor;
                        dz *= powerFactor;
                    }
                }
            } else if (interpolationMethod == 3 && tileDerivatives != nullptr) {
                // Triquintic interpolation (handles inv_power pre-transform internally)
                triquinticInterpolateTiled(
                    tileValues, tileDerivatives, localX, localY, localZ,
                    fx, fy, fz,
                    &interpolated, &dx, &dy, &dz,
                    gridSpacing[0], gridSpacing[1], gridSpacing[2],
                    tileWithOverlap,
                    invPower, invPowerMode
                );

                // Back-transform from transformed space if RUNTIME or STORED mode
                if ((invPowerMode == 1 || invPowerMode == 2) && invPower != 0.0f) {
                    float sign = (interpolated >= 0.0f) ? 1.0f : -1.0f;
                    float absVal = fabsf(interpolated);
                    if (absVal > 1e-10f) {
                        float powerFactor = invPower * powf(absVal, invPower - 1.0f);
                        interpolated = sign * powf(absVal, invPower);
                        dx *= powerFactor;
                        dy *= powerFactor;
                        dz *= powerFactor;
                    }
                }
            } else {
                // Fallback to trilinear if derivatives not available for tricubic/triquintic
                trilinearInterpolateTiled(
                    tileValues, localX, localY, localZ,
                    fx, fy, fz,
                    &interpolated, &dx, &dy, &dz,
                    gridSpacing[0], gridSpacing[1], gridSpacing[2],
                    tileWithOverlap,
                    invPower, invPowerMode
                );

                // Back-transform from transformed space if RUNTIME or STORED mode
                if ((invPowerMode == 1 || invPowerMode == 2) && invPower != 0.0f) {
                    float sign = (interpolated >= 0.0f) ? 1.0f : -1.0f;
                    float absVal = fabsf(interpolated);
                    if (absVal > 1e-10f) {
                        float powerFactor = invPower * powf(absVal, invPower - 1.0f);
                        interpolated = sign * powf(absVal, invPower);
                        dx = dx * powerFactor / gridSpacing[0];
                        dy = dy * powerFactor / gridSpacing[1];
                        dz = dz * powerFactor / gridSpacing[2];
                    }
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

    // Store per-atom out-of-bounds flag if buffer provided
    if (outOfBoundsBuffer != nullptr) {
        outOfBoundsBuffer[index] = isInside ? 0 : 1;
    }

    // Accumulate energy - EITHER to group OR to total, not both
    // (This matches the non-tiled kernel behavior)
    if (particleToGroupMap != nullptr && groupEnergyBuffer != nullptr) {
        int groupIdx = particleToGroupMap[index];
        if (groupIdx >= 0 && groupIdx < numGroups) {
            // Particle in a group - only add to group energy
            atomicAdd(&groupEnergyBuffer[groupIdx], threadEnergy);
        } else {
            // Particle not in any group - add to total
            atomicAdd(energyBuffer, threadEnergy);
        }
    } else {
        // No group tracking - add to total
        atomicAdd(energyBuffer, threadEnergy);
    }

    // Convert force to fixed point and accumulate
    atomicAdd(&forceBuffers[particleIndex], static_cast<unsigned long long>((long long)(atomForce.x * 0x100000000)));
    atomicAdd(&forceBuffers[particleIndex + paddedNumAtoms], static_cast<unsigned long long>((long long)(atomForce.y * 0x100000000)));
    atomicAdd(&forceBuffers[particleIndex + 2 * paddedNumAtoms], static_cast<unsigned long long>((long long)(atomForce.z * 0x100000000)));
}
