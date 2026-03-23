/**
 * CUDA implementation of tiled grid Hessian (second derivative) calculation.
 *
 * This kernel computes the 3x3 Hessian block for each atom from the grid potential
 * using tile-based grid storage. Same as gridHessian.cu but uses TileManager for
 * streaming grid data.
 */

#include "include/InterpolationBasis.cuh"
#include "include/TriquinticCoefficients.cuh"
#include "include/InvPowerChainRule.cuh"

/**
 * Apply runtime tanh cap chain rule to Hessian.
 * f(v) = C*tanh(v/C), f'(v) = sech²(v/C), f''(v) = -2*tanh(v/C)*sech²(v/C)/C
 * d^2f/dxi dxj = f'(v)*d^2v/dxi dxj + f''(v)*dv/dxi*dv/dxj
 */
__device__ inline void applyRuntimeCapHessianChainRuleTiled(
    float v, float dvdx, float dvdy, float dvdz,
    float& d2xx, float& d2yy, float& d2zz,
    float& d2xy, float& d2xz, float& d2yz,
    float cap
) {
    float t = tanhf(v / cap);
    float sech2 = 1.0f - t * t;
    float fPrime = sech2;
    float fDoublePrime = -2.0f * t * sech2 / cap;

    float new_d2xx = fPrime * d2xx + fDoublePrime * dvdx * dvdx;
    float new_d2yy = fPrime * d2yy + fDoublePrime * dvdy * dvdy;
    float new_d2zz = fPrime * d2zz + fDoublePrime * dvdz * dvdz;
    float new_d2xy = fPrime * d2xy + fDoublePrime * dvdx * dvdy;
    float new_d2xz = fPrime * d2xz + fDoublePrime * dvdx * dvdz;
    float new_d2yz = fPrime * d2yz + fDoublePrime * dvdy * dvdz;

    d2xx = new_d2xx; d2yy = new_d2yy; d2zz = new_d2zz;
    d2xy = new_d2xy; d2xz = new_d2xz; d2yz = new_d2yz;
}

/**
 * Find which tile contains a grid position.
 * Returns tile index or -1 if not found.
 */
__device__ int findTileForPositionHessian(
    int gridX, int gridY, int gridZ,
    const int* __restrict__ tileOffsets,  // x,y,z for each tile
    int numTiles,
    int tileSize  // Core tile size (excluding overlap)
) {
    // Linear search for now - works for small number of tiles
    for (int t = 0; t < numTiles; t++) {
        int tileStartX = tileOffsets[t * 3 + 0];
        int tileStartY = tileOffsets[t * 3 + 1];
        int tileStartZ = tileOffsets[t * 3 + 2];

        // Check if grid position falls within this tile's core region
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
__device__ __forceinline__ int tileIndexHessian(int lx, int ly, int lz, int tileWithOverlap) {
    return lx * tileWithOverlap * tileWithOverlap + ly * tileWithOverlap + lz;
}

/**
 * Compute Hessian blocks for all atoms using tiled grid interpolation.
 *
 * @param posq              Atom positions (x, y, z, charge)
 * @param hessianBuffer     Output: 6 floats per atom [dxx, dyy, dzz, dxy, dxz, dyz]
 * @param gridCounts        Grid dimensions [nx, ny, nz]
 * @param gridSpacing       Grid spacing [dx, dy, dz]
 * @param scalingFactors    Per-atom scaling factors
 * @param invPower          Inverse power parameter (e.g., -6.0)
 * @param invPowerMode      0=NONE, 1=RUNTIME, 2=STORED
 * @param interpolationMethod  1=bspline, 3=triquintic (others not supported)
 * @param originX/Y/Z       Grid origin coordinates
 * @param numAtoms          Number of atoms to process
 * @param particleIndices   Optional particle index filtering
 * @param tileOffsets       Grid offsets for each tile (x,y,z,x,y,z,...)
 * @param tileValuePtrs     Device pointers to tile values
 * @param tileDerivPtrs     Device pointers to tile derivatives
 * @param numTiles          Number of loaded tiles
 * @param tileSize          Core tile size (excluding overlap)
 * @param tileOverlap       Overlap for interpolation stencil
 */
extern "C" __global__ void computeGridHessianTiled(
    const float4* __restrict__ posq,
    float* __restrict__ hessianBuffer,  // 6 components per atom
    const int* __restrict__ gridCounts,
    const float* __restrict__ gridSpacing,
    const float* __restrict__ scalingFactors,
    const float invPower,
    const int invPowerMode,  // 0=NONE, 1=RUNTIME, 2=STORED
    const int interpolationMethod,
    const float originX,
    const float originY,
    const float originZ,
    const int numAtoms,
    const int* __restrict__ particleIndices,
    // Tile-specific parameters
    const int* __restrict__ tileOffsets,
    const unsigned long long* __restrict__ tileValuePtrs,
    const unsigned long long* __restrict__ tileDerivPtrs,
    const int numTiles,
    const int tileSize,
    const int tileOverlap,
    const float arcsinhScale,
    const float* __restrict__ groupScalingFactors,  // Per-group alchemical scaling (null = no per-group scaling)
    const int* __restrict__ particleToGroupMap,      // Maps particle index to group index (null = no mapping)
    const int numGroups,                             // Number of particle groups
    const float runtimeCap,                          // Global runtime cap (0=disabled)
    const float* __restrict__ groupRuntimeCaps,      // Per-group runtime caps (null = use global, 0 = use global)
    const float effectiveMinX, const float effectiveMinY, const float effectiveMinZ,
    const float effectiveMaxX, const float effectiveMaxY, const float effectiveMaxZ)
{
    const int tileWithOverlap = tileSize + 2 * tileOverlap;
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= numAtoms)
        return;

    // Get actual particle index
    const unsigned int particleIndex = (particleIndices != 0) ? particleIndices[index] : index;

    // Load position and scaling factor (with per-group alchemical scaling)
    float4 posOrig = posq[particleIndex];
    float groupScale = 1.0f;
    if (groupScalingFactors != 0 && particleToGroupMap != 0) {
        int groupIdx = particleToGroupMap[particleIndex];
        if (groupIdx >= 0 && groupIdx < numGroups) {
            groupScale = groupScalingFactors[groupIdx];
        }
    }
    float scalingFactor = groupScale * scalingFactors[particleIndex];

    // Resolve effective runtime cap: per-group if available, else global
    float effectiveCap = runtimeCap;
    if (groupRuntimeCaps != 0 && particleToGroupMap != 0) {
        int gIdx = particleToGroupMap[particleIndex];
        if (gIdx >= 0 && gIdx < numGroups && groupRuntimeCaps[gIdx] > 0.0f) {
            effectiveCap = groupRuntimeCaps[gIdx];
        }
    }

    // Transform to grid coordinates
    float3 pos;
    pos.x = posOrig.x - originX;
    pos.y = posOrig.y - originY;
    pos.z = posOrig.z - originZ;

    // Initialize Hessian components to zero
    float d2xx = 0.0f, d2yy = 0.0f, d2zz = 0.0f;
    float d2xy = 0.0f, d2xz = 0.0f, d2yz = 0.0f;

    // Check against effective evaluation bounds
    bool isInside = (pos.x >= effectiveMinX && pos.x <= effectiveMaxX &&
                     pos.y >= effectiveMinY && pos.y <= effectiveMaxY &&
                     pos.z >= effectiveMinZ && pos.z <= effectiveMaxZ);

    if (isInside && scalingFactor != 0.0f) {
        // Grid indices
        int ix = min(max((int)(pos.x / gridSpacing[0]), 0), gridCounts[0] - 2);
        int iy = min(max((int)(pos.y / gridSpacing[1]), 0), gridCounts[1] - 2);
        int iz = min(max((int)(pos.z / gridSpacing[2]), 0), gridCounts[2] - 2);

        // Fractional position [0, 1]
        float fx = (pos.x / gridSpacing[0]) - ix;
        float fy = (pos.y / gridSpacing[1]) - iy;
        float fz = (pos.z / gridSpacing[2]) - iz;

        fx = min(max(fx, 0.0f), 1.0f);
        fy = min(max(fy, 0.0f), 1.0f);
        fz = min(max(fz, 0.0f), 1.0f);

        // Find which tile contains this grid position
        int tileIdx = findTileForPositionHessian(ix, iy, iz, tileOffsets, numTiles, tileSize);

        if (tileIdx >= 0) {
            // Get tile data pointers
            const float* tileValues = (const float*)tileValuePtrs[tileIdx];
            const float* tileDerivatives = (const float*)tileDerivPtrs[tileIdx];

            // Convert global grid coordinates to tile-local coordinates
            int tileStartX = tileOffsets[tileIdx * 3 + 0];
            int tileStartY = tileOffsets[tileIdx * 3 + 1];
            int tileStartZ = tileOffsets[tileIdx * 3 + 2];

            int localX = (ix - tileStartX) + tileOverlap;
            int localY = (iy - tileStartY) + tileOverlap;
            int localZ = (iz - tileStartZ) + tileOverlap;

            int tilePoints = tileWithOverlap * tileWithOverlap * tileWithOverlap;

            float interpolated = 0.0f;
            float dx = 0.0f, dy = 0.0f, dz = 0.0f;

            if (interpolationMethod == 3 && tileDerivatives != 0) {
                // TRIQUINTIC HERMITE - Analytical second derivatives using tile data

                // 8 corners of the cell in tile-local coordinates
                int corners[8][3] = {
                    {localX, localY, localZ}, {localX+1, localY, localZ},
                    {localX, localY+1, localZ}, {localX+1, localY+1, localZ},
                    {localX, localY, localZ+1}, {localX+1, localY, localZ+1},
                    {localX, localY+1, localZ+1}, {localX+1, localY+1, localZ+1}
                };

                // Gather derivatives in DERIVATIVE-MAJOR layout
                float X[216];
                if (invPowerMode == 1) {
                    // RUNTIME mode: transform all 27 derivatives per corner
                    float p = 1.0f / invPower;
                    for (int c = 0; c < 8; c++) {
                        int point_idx = tileIndexHessian(corners[c][0], corners[c][1], corners[c][2], tileWithOverlap);
                        float G_derivs[27], S_derivs[27];
                        for (int d = 0; d < 27; d++) {
                            G_derivs[d] = tileDerivatives[d * tilePoints + point_idx];
                        }
                        applyInvPowerChainRule(G_derivs, p, S_derivs);
                        for (int d = 0; d < 27; d++) {
                            X[d * 8 + c] = S_derivs[d];
                        }
                    }
                } else {
                    // STORED or NONE mode: load directly
                    for (int d = 0; d < 27; d++) {
                        for (int c = 0; c < 8; c++) {
                            int point_idx = tileIndexHessian(corners[c][0], corners[c][1], corners[c][2], tileWithOverlap);
                            X[d * 8 + c] = tileDerivatives[d * tilePoints + point_idx];
                        }
                    }
                }

                // Compute polynomial coefficients
                float a[216];
                const float scale = 0.125f;
                for (int i = 0; i < 216; i++) {
                    a[i] = 0.0f;
                    for (int j = 0; j < 216; j++) {
                        a[i] += TRIQUINTIC_COEFFICIENTS[i][j] * X[j];
                    }
                    a[i] *= scale;
                }

                // Precompute powers
                float sx_pow[6], sy_pow[6], sz_pow[6];
                sx_pow[0] = sy_pow[0] = sz_pow[0] = 1.0f;
                for (int p = 1; p < 6; p++) {
                    sx_pow[p] = sx_pow[p-1] * fx;
                    sy_pow[p] = sy_pow[p-1] * fy;
                    sz_pow[p] = sz_pow[p-1] * fz;
                }

                // Evaluate polynomial value, first derivatives, and second derivatives
                for (int k = 0; k < 6; k++) {
                    for (int j = 0; j < 6; j++) {
                        for (int i = 0; i < 6; i++) {
                            int coeff_idx = i + 6*j + 36*k;
                            float coeff = a[coeff_idx];
                            float term = sx_pow[i] * sy_pow[j] * sz_pow[k];

                            // Interpolated value
                            interpolated += coeff * term;

                            // First derivatives
                            if (i >= 1) dx += coeff * i * sx_pow[i-1] * sy_pow[j] * sz_pow[k];
                            if (j >= 1) dy += coeff * j * sx_pow[i] * sy_pow[j-1] * sz_pow[k];
                            if (k >= 1) dz += coeff * k * sx_pow[i] * sy_pow[j] * sz_pow[k-1];

                            // Second derivatives
                            if (i >= 2) d2xx += coeff * (i * (i-1)) * sx_pow[i-2] * sy_pow[j] * sz_pow[k];
                            if (j >= 2) d2yy += coeff * (j * (j-1)) * sx_pow[i] * sy_pow[j-2] * sz_pow[k];
                            if (k >= 2) d2zz += coeff * (k * (k-1)) * sx_pow[i] * sy_pow[j] * sz_pow[k-2];
                            if (i >= 1 && j >= 1) d2xy += coeff * (i * j) * sx_pow[i-1] * sy_pow[j-1] * sz_pow[k];
                            if (i >= 1 && k >= 1) d2xz += coeff * (i * k) * sx_pow[i-1] * sy_pow[j] * sz_pow[k-1];
                            if (j >= 1 && k >= 1) d2yz += coeff * (j * k) * sx_pow[i] * sy_pow[j-1] * sz_pow[k-1];
                        }
                    }
                }

                // Back-convert from smoothed space to actual potential for RUNTIME mode
                if (invPowerMode == 1 && fabsf(invPower) > 1e-10f) {
                    float absU = fabsf(interpolated);
                    if (absU > 1e-10f) {
                        float n = invPower;
                        float absU_nm1 = powf(absU, n - 1.0f);
                        float absU_nm2 = powf(absU, n - 2.0f);
                        float f2_1 = n * (n - 1.0f) * absU_nm2;
                        float f2_2 = n * absU_nm1;

                        float new_d2xx = f2_1 * dx * dx + f2_2 * d2xx;
                        float new_d2yy = f2_1 * dy * dy + f2_2 * d2yy;
                        float new_d2zz = f2_1 * dz * dz + f2_2 * d2zz;
                        float new_d2xy = f2_1 * dx * dy + f2_2 * d2xy;
                        float new_d2xz = f2_1 * dx * dz + f2_2 * d2xz;
                        float new_d2yz = f2_1 * dy * dz + f2_2 * d2yz;

                        d2xx = new_d2xx;
                        d2yy = new_d2yy;
                        d2zz = new_d2zz;
                        d2xy = new_d2xy;
                        d2xz = new_d2xz;
                        d2yz = new_d2yz;
                    }
                }

                // Arcsinh chain rule: V = scale*sinh(g), d²V/dxi dxj = scale*[sinh(g)*dg_i*dg_j + cosh(g)*d²g_ij]
                if (arcsinhScale > 0.0f) {
                    float g = interpolated;
                    float sinhG = sinhf(g);
                    float coshG = coshf(g);

                    float new_d2xx = arcsinhScale * (sinhG * dx * dx + coshG * d2xx);
                    float new_d2yy = arcsinhScale * (sinhG * dy * dy + coshG * d2yy);
                    float new_d2zz = arcsinhScale * (sinhG * dz * dz + coshG * d2zz);
                    float new_d2xy = arcsinhScale * (sinhG * dx * dy + coshG * d2xy);
                    float new_d2xz = arcsinhScale * (sinhG * dx * dz + coshG * d2xz);
                    float new_d2yz = arcsinhScale * (sinhG * dy * dz + coshG * d2yz);

                    dx = arcsinhScale * coshG * dx;
                    dy = arcsinhScale * coshG * dy;
                    dz = arcsinhScale * coshG * dz;

                    d2xx = new_d2xx; d2yy = new_d2yy; d2zz = new_d2zz;
                    d2xy = new_d2xy; d2xz = new_d2xz; d2yz = new_d2yz;
                }

                // Convert from unit cell to physical coordinates
                float inv_dx = 1.0f / gridSpacing[0];
                float inv_dy = 1.0f / gridSpacing[1];
                float inv_dz = 1.0f / gridSpacing[2];
                float inv_dx2 = inv_dx * inv_dx;
                float inv_dy2 = inv_dy * inv_dy;
                float inv_dz2 = inv_dz * inv_dz;
                float inv_dxdy = inv_dx * inv_dy;
                float inv_dxdz = inv_dx * inv_dz;
                float inv_dydz = inv_dy * inv_dz;

                d2xx *= inv_dx2;
                d2yy *= inv_dy2;
                d2zz *= inv_dz2;
                d2xy *= inv_dxdy;
                d2xz *= inv_dxdz;
                d2yz *= inv_dydz;

            } else if (interpolationMethod == 1) {
                // CUBIC B-SPLINE - Analytical second derivatives using tile data

                // Precompute basis functions and derivatives
                float bx[4] = {bspline_basis0(fx), bspline_basis1(fx), bspline_basis2(fx), bspline_basis3(fx)};
                float by[4] = {bspline_basis0(fy), bspline_basis1(fy), bspline_basis2(fy), bspline_basis3(fy)};
                float bz[4] = {bspline_basis0(fz), bspline_basis1(fz), bspline_basis2(fz), bspline_basis3(fz)};

                float dbx[4] = {bspline_deriv0(fx), bspline_deriv1(fx), bspline_deriv2(fx), bspline_deriv3(fx)};
                float dby[4] = {bspline_deriv0(fy), bspline_deriv1(fy), bspline_deriv2(fy), bspline_deriv3(fy)};
                float dbz[4] = {bspline_deriv0(fz), bspline_deriv1(fz), bspline_deriv2(fz), bspline_deriv3(fz)};

                float d2bx[4] = {bspline_deriv2_0(fx), bspline_deriv2_1(fx), bspline_deriv2_2(fx), bspline_deriv2_3(fx)};
                float d2by[4] = {bspline_deriv2_0(fy), bspline_deriv2_1(fy), bspline_deriv2_2(fy), bspline_deriv2_3(fy)};
                float d2bz[4] = {bspline_deriv2_0(fz), bspline_deriv2_1(fz), bspline_deriv2_2(fz), bspline_deriv2_3(fz)};

                for (int i = 0; i < 4; i++) {
                    int lx = localX - 1 + i;
                    lx = min(max(lx, 0), tileWithOverlap - 1);

                    for (int j = 0; j < 4; j++) {
                        int ly = localY - 1 + j;
                        ly = min(max(ly, 0), tileWithOverlap - 1);

                        for (int k = 0; k < 4; k++) {
                            int lz = localZ - 1 + k;
                            lz = min(max(lz, 0), tileWithOverlap - 1);

                            float val = tileValues[tileIndexHessian(lx, ly, lz, tileWithOverlap)];

                            // Apply RUNTIME inv_power transformation before interpolation
                            if (invPowerMode == 1) {
                                float invN = 1.0f / invPower;
                                if (fabsf(val) >= 1e-10f) {
                                    val = (val >= 0.0f ? 1.0f : -1.0f) * powf(fabsf(val), invN);
                                } else {
                                    val = 0.0f;
                                }
                            }

                            // Interpolated value
                            interpolated += bx[i] * by[j] * bz[k] * val;

                            // First derivatives
                            dx += dbx[i] * by[j] * bz[k] * val;
                            dy += bx[i] * dby[j] * bz[k] * val;
                            dz += bx[i] * by[j] * dbz[k] * val;

                            // Second derivatives
                            d2xx += d2bx[i] * by[j] * bz[k] * val;
                            d2yy += bx[i] * d2by[j] * bz[k] * val;
                            d2zz += bx[i] * by[j] * d2bz[k] * val;
                            d2xy += dbx[i] * dby[j] * bz[k] * val;
                            d2xz += dbx[i] * by[j] * dbz[k] * val;
                            d2yz += bx[i] * dby[j] * dbz[k] * val;
                        }
                    }
                }

                // Undo transforms in reverse order: arcsinh first, then inv_power.
                // (Matches gridHessian.cu and gridForce.cu ordering)
                if (arcsinhScale > 0.0f) {
                    float g = interpolated;
                    float sinhG = sinhf(g);
                    float coshG = coshf(g);

                    float new_d2xx = arcsinhScale * (sinhG * dx * dx + coshG * d2xx);
                    float new_d2yy = arcsinhScale * (sinhG * dy * dy + coshG * d2yy);
                    float new_d2zz = arcsinhScale * (sinhG * dz * dz + coshG * d2zz);
                    float new_d2xy = arcsinhScale * (sinhG * dx * dy + coshG * d2xy);
                    float new_d2xz = arcsinhScale * (sinhG * dx * dz + coshG * d2xz);
                    float new_d2yz = arcsinhScale * (sinhG * dy * dz + coshG * d2yz);

                    interpolated = arcsinhScale * sinhG;
                    dx = arcsinhScale * coshG * dx;
                    dy = arcsinhScale * coshG * dy;
                    dz = arcsinhScale * coshG * dz;

                    d2xx = new_d2xx; d2yy = new_d2yy; d2zz = new_d2zz;
                    d2xy = new_d2xy; d2xz = new_d2xz; d2yz = new_d2yz;
                }

                if ((invPowerMode == 1 || invPowerMode == 2) && fabsf(invPower) > 1e-10f) {
                    float absU = fabsf(interpolated);
                    if (absU > 1e-10f) {
                        float n = invPower;
                        float absU_nm1 = powf(absU, n - 1.0f);
                        float absU_nm2 = powf(absU, n - 2.0f);
                        float f2_1 = n * (n - 1.0f) * absU_nm2;
                        float f2_2 = n * absU_nm1;

                        float new_d2xx = f2_1 * dx * dx + f2_2 * d2xx;
                        float new_d2yy = f2_1 * dy * dy + f2_2 * d2yy;
                        float new_d2zz = f2_1 * dz * dz + f2_2 * d2zz;
                        float new_d2xy = f2_1 * dx * dy + f2_2 * d2xy;
                        float new_d2xz = f2_1 * dx * dz + f2_2 * d2xz;
                        float new_d2yz = f2_1 * dy * dz + f2_2 * d2yz;

                        dx *= f2_2;
                        dy *= f2_2;
                        dz *= f2_2;

                        float sign = (interpolated >= 0.0f) ? 1.0f : -1.0f;
                        interpolated = sign * powf(absU, n);

                        d2xx = new_d2xx; d2yy = new_d2yy; d2zz = new_d2zz;
                        d2xy = new_d2xy; d2xz = new_d2xz; d2yz = new_d2yz;
                    }
                }

                // Apply runtime tanh cap chain rule (must match gridForce.cu cap order)
                if (effectiveCap > 0.0f) {
                    applyRuntimeCapHessianChainRuleTiled(interpolated, dx, dy, dz,
                        d2xx, d2yy, d2zz, d2xy, d2xz, d2yz, effectiveCap);
                    float t = tanhf(interpolated / effectiveCap);
                    float gradFactor = 1.0f - t * t;
                    interpolated = effectiveCap * t;
                    dx *= gradFactor;
                    dy *= gradFactor;
                    dz *= gradFactor;
                }

                // Convert to physical coordinates
                float inv_dx = 1.0f / gridSpacing[0];
                float inv_dy = 1.0f / gridSpacing[1];
                float inv_dz = 1.0f / gridSpacing[2];
                float inv_dx2 = inv_dx * inv_dx;
                float inv_dy2 = inv_dy * inv_dy;
                float inv_dz2 = inv_dz * inv_dz;
                float inv_dxdy = inv_dx * inv_dy;
                float inv_dxdz = inv_dx * inv_dz;
                float inv_dydz = inv_dy * inv_dz;

                d2xx *= inv_dx2;
                d2yy *= inv_dy2;
                d2zz *= inv_dz2;
                d2xy *= inv_dxdy;
                d2xz *= inv_dxdz;
                d2yz *= inv_dydz;

            } else if (interpolationMethod == 4) {
                // QUINTIC B-SPLINE - Analytical second derivatives from 6-point stencil

                float bx[6] = {qbspline_basis0(fx), qbspline_basis1(fx), qbspline_basis2(fx),
                               qbspline_basis3(fx), qbspline_basis4(fx), qbspline_basis5(fx)};
                float by[6] = {qbspline_basis0(fy), qbspline_basis1(fy), qbspline_basis2(fy),
                               qbspline_basis3(fy), qbspline_basis4(fy), qbspline_basis5(fy)};
                float bz[6] = {qbspline_basis0(fz), qbspline_basis1(fz), qbspline_basis2(fz),
                               qbspline_basis3(fz), qbspline_basis4(fz), qbspline_basis5(fz)};

                float dbx[6] = {qbspline_deriv0(fx), qbspline_deriv1(fx), qbspline_deriv2(fx),
                                qbspline_deriv3(fx), qbspline_deriv4(fx), qbspline_deriv5(fx)};
                float dby[6] = {qbspline_deriv0(fy), qbspline_deriv1(fy), qbspline_deriv2(fy),
                                qbspline_deriv3(fy), qbspline_deriv4(fy), qbspline_deriv5(fy)};
                float dbz[6] = {qbspline_deriv0(fz), qbspline_deriv1(fz), qbspline_deriv2(fz),
                                qbspline_deriv3(fz), qbspline_deriv4(fz), qbspline_deriv5(fz)};

                float d2bx[6] = {qbspline_deriv2_0(fx), qbspline_deriv2_1(fx), qbspline_deriv2_2(fx),
                                 qbspline_deriv2_3(fx), qbspline_deriv2_4(fx), qbspline_deriv2_5(fx)};
                float d2by[6] = {qbspline_deriv2_0(fy), qbspline_deriv2_1(fy), qbspline_deriv2_2(fy),
                                 qbspline_deriv2_3(fy), qbspline_deriv2_4(fy), qbspline_deriv2_5(fy)};
                float d2bz[6] = {qbspline_deriv2_0(fz), qbspline_deriv2_1(fz), qbspline_deriv2_2(fz),
                                 qbspline_deriv2_3(fz), qbspline_deriv2_4(fz), qbspline_deriv2_5(fz)};

                for (int i = 0; i < 6; i++) {
                    int lx = localX - 2 + i;
                    lx = min(max(lx, 0), tileWithOverlap - 1);

                    for (int j = 0; j < 6; j++) {
                        int ly = localY - 2 + j;
                        ly = min(max(ly, 0), tileWithOverlap - 1);

                        for (int k = 0; k < 6; k++) {
                            int lz = localZ - 2 + k;
                            lz = min(max(lz, 0), tileWithOverlap - 1);

                            float val = tileValues[tileIndexHessian(lx, ly, lz, tileWithOverlap)];

                            if (invPowerMode == 1) {
                                float invN = 1.0f / invPower;
                                if (fabsf(val) >= 1e-10f) {
                                    val = (val >= 0.0f ? 1.0f : -1.0f) * powf(fabsf(val), invN);
                                } else {
                                    val = 0.0f;
                                }
                            }

                            interpolated += bx[i] * by[j] * bz[k] * val;
                            dx += dbx[i] * by[j] * bz[k] * val;
                            dy += bx[i] * dby[j] * bz[k] * val;
                            dz += bx[i] * by[j] * dbz[k] * val;
                            d2xx += d2bx[i] * by[j] * bz[k] * val;
                            d2yy += bx[i] * d2by[j] * bz[k] * val;
                            d2zz += bx[i] * by[j] * d2bz[k] * val;
                            d2xy += dbx[i] * dby[j] * bz[k] * val;
                            d2xz += dbx[i] * by[j] * dbz[k] * val;
                            d2yz += bx[i] * dby[j] * dbz[k] * val;
                        }
                    }
                }

                // Undo transforms in reverse order: arcsinh first, then inv_power.
                if (arcsinhScale > 0.0f) {
                    float g = interpolated;
                    float sinhG = sinhf(g);
                    float coshG = coshf(g);

                    float new_d2xx = arcsinhScale * (sinhG * dx * dx + coshG * d2xx);
                    float new_d2yy = arcsinhScale * (sinhG * dy * dy + coshG * d2yy);
                    float new_d2zz = arcsinhScale * (sinhG * dz * dz + coshG * d2zz);
                    float new_d2xy = arcsinhScale * (sinhG * dx * dy + coshG * d2xy);
                    float new_d2xz = arcsinhScale * (sinhG * dx * dz + coshG * d2xz);
                    float new_d2yz = arcsinhScale * (sinhG * dy * dz + coshG * d2yz);

                    interpolated = arcsinhScale * sinhG;
                    dx = arcsinhScale * coshG * dx;
                    dy = arcsinhScale * coshG * dy;
                    dz = arcsinhScale * coshG * dz;

                    d2xx = new_d2xx; d2yy = new_d2yy; d2zz = new_d2zz;
                    d2xy = new_d2xy; d2xz = new_d2xz; d2yz = new_d2yz;
                }

                if ((invPowerMode == 1 || invPowerMode == 2) && fabsf(invPower) > 1e-10f) {
                    float absU = fabsf(interpolated);
                    if (absU > 1e-10f) {
                        float n = invPower;
                        float absU_nm1 = powf(absU, n - 1.0f);
                        float absU_nm2 = powf(absU, n - 2.0f);
                        float f2_1 = n * (n - 1.0f) * absU_nm2;
                        float f2_2 = n * absU_nm1;

                        float new_d2xx = f2_1 * dx * dx + f2_2 * d2xx;
                        float new_d2yy = f2_1 * dy * dy + f2_2 * d2yy;
                        float new_d2zz = f2_1 * dz * dz + f2_2 * d2zz;
                        float new_d2xy = f2_1 * dx * dy + f2_2 * d2xy;
                        float new_d2xz = f2_1 * dx * dz + f2_2 * d2xz;
                        float new_d2yz = f2_1 * dy * dz + f2_2 * d2yz;

                        // Update value and first derivatives to post-inv_power
                        float sign = (interpolated >= 0.0f) ? 1.0f : -1.0f;
                        float pf = n * absU_nm1;
                        interpolated = sign * powf(absU, n);
                        dx *= pf;
                        dy *= pf;
                        dz *= pf;

                        d2xx = new_d2xx; d2yy = new_d2yy; d2zz = new_d2zz;
                        d2xy = new_d2xy; d2xz = new_d2xz; d2yz = new_d2yz;
                    }
                }

                // Apply runtime tanh cap chain rule (must match gridForce.cu cap order)
                if (effectiveCap > 0.0f) {
                    applyRuntimeCapHessianChainRuleTiled(interpolated, dx, dy, dz,
                        d2xx, d2yy, d2zz, d2xy, d2xz, d2yz, effectiveCap);
                    float t = tanhf(interpolated / effectiveCap);
                    float gradFactor = 1.0f - t * t;
                    interpolated = effectiveCap * t;
                    dx *= gradFactor;
                    dy *= gradFactor;
                    dz *= gradFactor;
                }

                // Convert to physical coordinates
                float inv_dx = 1.0f / gridSpacing[0];
                float inv_dy = 1.0f / gridSpacing[1];
                float inv_dz = 1.0f / gridSpacing[2];

                d2xx *= inv_dx * inv_dx;
                d2yy *= inv_dy * inv_dy;
                d2zz *= inv_dz * inv_dz;
                d2xy *= inv_dx * inv_dy;
                d2xz *= inv_dx * inv_dz;
                d2yz *= inv_dy * inv_dz;
            }
            // For unsupported methods (trilinear, tricubic), Hessian remains zero

            // Apply scaling factor to Hessian
            d2xx *= scalingFactor;
            d2yy *= scalingFactor;
            d2zz *= scalingFactor;
            d2xy *= scalingFactor;
            d2xz *= scalingFactor;
            d2yz *= scalingFactor;
        }
    }

    // Store Hessian components (6 per atom)
    int offset = index * 6;
    hessianBuffer[offset + 0] = d2xx;
    hessianBuffer[offset + 1] = d2yy;
    hessianBuffer[offset + 2] = d2zz;
    hessianBuffer[offset + 3] = d2xy;
    hessianBuffer[offset + 4] = d2xz;
    hessianBuffer[offset + 5] = d2yz;
}
