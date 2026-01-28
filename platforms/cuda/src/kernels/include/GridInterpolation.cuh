/**
 * Shared grid interpolation library for OpenMM GridForce plugins.
 *
 * This header provides reusable interpolation functions that can be used by
 * multiple Force implementations (GridForce, GBSAGridForce, etc.)
 *
 * Supported methods:
 *   0 - Trilinear (8 points, C0 continuity)
 *   1 - Cubic B-spline (64 points, C2 continuity)
 *   2 - Tricubic Lekien-Marsden (8 corners + derivatives, C1 continuity)
 *   3 - Triquintic Hermite (8 corners + 27 derivatives each, C4 continuity)
 *
 * Usage:
 *   #include "GridInterpolation.cuh"
 *
 *   InterpolationResult result = interpolateGrid(
 *       gridValues, gridDerivatives, gridCounts, gridSpacing,
 *       originX, originY, originZ, position, method, true);
 */

#ifndef OPENMM_GRID_INTERPOLATION_CUH_
#define OPENMM_GRID_INTERPOLATION_CUH_

#include "InterpolationBasis.cuh"
#include "TricubicCoefficients.cuh"
#include "TriquinticCoefficients.cuh"

/**
 * Result of grid interpolation containing value and gradient.
 */
struct InterpolationResult {
    float value;      // Interpolated value
    float3 gradient;  // Gradient - either physical (per nm) or raw (per unit cell), depending on divideBySpacing flag
    bool isInside;    // Whether the query point was inside the grid
};

/**
 * Check if a position is inside the grid bounds.
 */
__device__ inline bool isInsideGrid(
    float3 position,
    const int* __restrict__ gridCounts,
    const float* __restrict__ gridSpacing,
    float originX, float originY, float originZ)
{
    float relX = position.x - originX;
    float relY = position.y - originY;
    float relZ = position.z - originZ;

    float extentX = gridSpacing[0] * (gridCounts[0] - 1);
    float extentY = gridSpacing[1] * (gridCounts[1] - 1);
    float extentZ = gridSpacing[2] * (gridCounts[2] - 1);

    return (relX >= 0.0f && relX <= extentX &&
            relY >= 0.0f && relY <= extentY &&
            relZ >= 0.0f && relZ <= extentZ);
}

/**
 * Compute grid cell indices and fractional coordinates.
 * Returns false if position is outside grid.
 */
__device__ inline bool computeGridCell(
    float3 position,
    const int* __restrict__ gridCounts,
    const float* __restrict__ gridSpacing,
    float originX, float originY, float originZ,
    int& ix, int& iy, int& iz,
    float& fx, float& fy, float& fz)
{
    // Transform to grid-relative coordinates
    float3 pos;
    pos.x = position.x - originX;
    pos.y = position.y - originY;
    pos.z = position.z - originZ;

    // Check bounds
    float3 gridCorner;
    gridCorner.x = gridSpacing[0] * (gridCounts[0] - 1);
    gridCorner.y = gridSpacing[1] * (gridCounts[1] - 1);
    gridCorner.z = gridSpacing[2] * (gridCounts[2] - 1);

    bool isInside = (pos.x >= 0.0f && pos.x <= gridCorner.x &&
                     pos.y >= 0.0f && pos.y <= gridCorner.y &&
                     pos.z >= 0.0f && pos.z <= gridCorner.z);

    if (!isInside) return false;

    // Calculate grid indices
    ix = min(max((int)(pos.x / gridSpacing[0]), 0), gridCounts[0] - 2);
    iy = min(max((int)(pos.y / gridSpacing[1]), 0), gridCounts[1] - 2);
    iz = min(max((int)(pos.z / gridSpacing[2]), 0), gridCounts[2] - 2);

    // Calculate fractional position within the cell [0, 1]
    fx = (pos.x / gridSpacing[0]) - ix;
    fy = (pos.y / gridSpacing[1]) - iy;
    fz = (pos.z / gridSpacing[2]) - iz;

    fx = min(max(fx, 0.0f), 1.0f);
    fy = min(max(fy, 0.0f), 1.0f);
    fz = min(max(fz, 0.0f), 1.0f);

    return true;
}

/**
 * Trilinear interpolation (method 0).
 * Uses 8 corner points (2x2x2) for simple linear interpolation.
 * Provides C0 continuity (continuous values, discontinuous first derivatives).
 *
 * @param divideBySpacing If true, gradient is in physical units (per nm).
 *                        If false, gradient is in unit cell coords (for chain rule application).
 */
__device__ inline InterpolationResult trilinearInterpolate(
    const float* __restrict__ gridValues,
    const int* __restrict__ gridCounts,
    const float* __restrict__ gridSpacing,
    float originX, float originY, float originZ,
    float3 position,
    bool computeGradient = true,
    bool divideBySpacing = true)
{
    InterpolationResult result;
    result.value = 0.0f;
    result.gradient = make_float3(0.0f, 0.0f, 0.0f);

    int ix, iy, iz;
    float fx, fy, fz;
    result.isInside = computeGridCell(position, gridCounts, gridSpacing,
                                       originX, originY, originZ,
                                       ix, iy, iz, fx, fy, fz);

    if (!result.isInside) return result;

    float ox = 1.0f - fx;
    float oy = 1.0f - fy;
    float oz = 1.0f - fz;

    int nyz = gridCounts[1] * gridCounts[2];
    int baseIndex = ix * nyz + iy * gridCounts[2] + iz;
    int ip = baseIndex + nyz;
    int imp = baseIndex + gridCounts[2];
    int ipp = ip + gridCounts[2];

    // Get 8 corner values
    float vmmm = gridValues[baseIndex];
    float vmmp = gridValues[baseIndex + 1];
    float vmpm = gridValues[imp];
    float vmpp = gridValues[imp + 1];
    float vpmm = gridValues[ip];
    float vpmp = gridValues[ip + 1];
    float vppm = gridValues[ipp];
    float vppp = gridValues[ipp + 1];

    // Trilinear interpolation
    float vmm = oz * vmmm + fz * vmmp;
    float vmp = oz * vmpm + fz * vmpp;
    float vpm = oz * vpmm + fz * vpmp;
    float vpp = oz * vppm + fz * vppp;

    float vm = oy * vmm + fy * vmp;
    float vp = oy * vpm + fy * vpp;

    result.value = ox * vm + fx * vp;

    if (computeGradient) {
        float dx = (vp - vm);
        float dy = (ox * (vmp - vmm) + fx * (vpp - vpm));
        float dz = (ox * (oy * (vmmp - vmmm) + fy * (vmpp - vmpm)) +
                    fx * (oy * (vpmp - vpmm) + fy * (vppp - vppm)));

        if (divideBySpacing) {
            result.gradient.x = dx / gridSpacing[0];
            result.gradient.y = dy / gridSpacing[1];
            result.gradient.z = dz / gridSpacing[2];
        } else {
            result.gradient.x = dx;
            result.gradient.y = dy;
            result.gradient.z = dz;
        }
    }

    return result;
}

/**
 * Cubic B-spline interpolation (method 1).
 * Uses 64 grid points (4x4x4 stencil) for smoother interpolation.
 * Provides C2 continuity (continuous second derivatives).
 *
 * @param divideBySpacing If true, gradient is in physical units (per nm).
 *                        If false, gradient is in unit cell coords (for chain rule application).
 */
__device__ inline InterpolationResult bsplineInterpolate(
    const float* __restrict__ gridValues,
    const int* __restrict__ gridCounts,
    const float* __restrict__ gridSpacing,
    float originX, float originY, float originZ,
    float3 position,
    bool computeGradient = true,
    bool divideBySpacing = true)
{
    InterpolationResult result;
    result.value = 0.0f;
    result.gradient = make_float3(0.0f, 0.0f, 0.0f);

    int ix, iy, iz;
    float fx, fy, fz;
    result.isInside = computeGridCell(position, gridCounts, gridSpacing,
                                       originX, originY, originZ,
                                       ix, iy, iz, fx, fy, fz);

    if (!result.isInside) return result;

    int nyz = gridCounts[1] * gridCounts[2];

    // Precompute B-spline basis functions
    float bx[4] = {bspline_basis0(fx), bspline_basis1(fx), bspline_basis2(fx), bspline_basis3(fx)};
    float by[4] = {bspline_basis0(fy), bspline_basis1(fy), bspline_basis2(fy), bspline_basis3(fy)};
    float bz[4] = {bspline_basis0(fz), bspline_basis1(fz), bspline_basis2(fz), bspline_basis3(fz)};

    float dbx[4], dby[4], dbz[4];
    if (computeGradient) {
        dbx[0] = bspline_deriv0(fx); dbx[1] = bspline_deriv1(fx);
        dbx[2] = bspline_deriv2(fx); dbx[3] = bspline_deriv3(fx);
        dby[0] = bspline_deriv0(fy); dby[1] = bspline_deriv1(fy);
        dby[2] = bspline_deriv2(fy); dby[3] = bspline_deriv3(fy);
        dbz[0] = bspline_deriv0(fz); dbz[1] = bspline_deriv1(fz);
        dbz[2] = bspline_deriv2(fz); dbz[3] = bspline_deriv3(fz);
    }

    float interpolated = 0.0f;
    float dvdx = 0.0f, dvdy = 0.0f, dvdz = 0.0f;

    for (int i = 0; i < 4; i++) {
        int gx = min(max(ix - 1 + i, 0), gridCounts[0] - 1);
        for (int j = 0; j < 4; j++) {
            int gy = min(max(iy - 1 + j, 0), gridCounts[1] - 1);
            for (int k = 0; k < 4; k++) {
                int gz = min(max(iz - 1 + k, 0), gridCounts[2] - 1);
                int gridIdx = gx * nyz + gy * gridCounts[2] + gz;
                float val = gridValues[gridIdx];

                float weight = bx[i] * by[j] * bz[k];
                interpolated += weight * val;

                if (computeGradient) {
                    dvdx += dbx[i] * by[j] * bz[k] * val;
                    dvdy += bx[i] * dby[j] * bz[k] * val;
                    dvdz += bx[i] * by[j] * dbz[k] * val;
                }
            }
        }
    }

    result.value = interpolated;

    if (computeGradient) {
        if (divideBySpacing) {
            result.gradient.x = dvdx / gridSpacing[0];
            result.gradient.y = dvdy / gridSpacing[1];
            result.gradient.z = dvdz / gridSpacing[2];
        } else {
            result.gradient.x = dvdx;
            result.gradient.y = dvdy;
            result.gradient.z = dvdz;
        }
    }

    return result;
}

/**
 * Tricubic Lekien-Marsden interpolation (method 2).
 * Uses 8 corner points with 8 derivatives each (64 coefficients).
 * Requires precomputed analytical derivatives in gridDerivatives.
 * Provides C1 continuity.
 *
 * Derivative storage (RASPA3 order): gridDerivatives[deriv_idx * totalPoints + point_idx]
 * where deriv_idx: 0=f, 1=dx, 2=dy, 3=dz, 4=dxx, 5=dxy, 6=dxz, 7=dyy, 8=dyz, 9=dzz, ..., 13=dxyz
 */
__device__ inline InterpolationResult tricubicInterpolate(
    const float* __restrict__ gridValues,
    const float* __restrict__ gridDerivatives,
    const int* __restrict__ gridCounts,
    const float* __restrict__ gridSpacing,
    float originX, float originY, float originZ,
    float3 position,
    bool computeGradient = true,
    bool divideBySpacing = true)
{
    InterpolationResult result;
    result.value = 0.0f;
    result.gradient = make_float3(0.0f, 0.0f, 0.0f);

    if (gridDerivatives == nullptr) {
        result.isInside = false;
        return result;
    }

    int ix, iy, iz;
    float fx, fy, fz;
    result.isInside = computeGridCell(position, gridCounts, gridSpacing,
                                       originX, originY, originZ,
                                       ix, iy, iz, fx, fy, fz);

    if (!result.isInside) return result;

    int nyz = gridCounts[1] * gridCounts[2];
    int totalPoints = gridCounts[0] * gridCounts[1] * gridCounts[2];

    // Get 8 corner indices
    int corners[8][3] = {
        {ix, iy, iz}, {ix+1, iy, iz}, {ix, iy+1, iz}, {ix+1, iy+1, iz},
        {ix, iy, iz+1}, {ix+1, iy, iz+1}, {ix, iy+1, iz+1}, {ix+1, iy+1, iz+1}
    };

    // Map tricubic derivative order to gridDerivatives storage order
    // Tricubic needs: 0=f, 1=fx, 2=fy, 3=fz, 4=fxy, 5=fxz, 6=fyz, 7=fxyz
    // gridDerivatives (RASPA3): 0=f, 1=dx, 2=dy, 3=dz, 4=dxx, 5=dxy, 6=dxz, 7=dyy, 8=dyz, 9=dzz, ..., 13=dxyz
    const int derivMap[8] = {0, 1, 2, 3, 5, 6, 8, 13};

    // Storage: X[deriv*8 + corner] - DERIVATIVE-MAJOR (matches RASPA3/Lekien-Marsden)
    float X[64];
    for (int d = 0; d < 8; d++) {
        for (int c = 0; c < 8; c++) {
            int point_idx = corners[c][0] * nyz + corners[c][1] * gridCounts[2] + corners[c][2];
            X[d*8 + c] = gridDerivatives[derivMap[d] * totalPoints + point_idx];
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

    // Evaluate tricubic polynomial P(x,y,z) = sum_{i,j,k=0}^3 a_{ijk} * x^i * y^j * z^k
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
                if (computeGradient) {
                    dx += coeff * fx_pow_i_deriv * fy_pow_j * fz_pow_k;
                    dy += coeff * fx_pow_i * fy_pow_j_deriv * fz_pow_k;
                    dz += coeff * fx_pow_i * fy_pow_j * fz_pow_k_deriv;
                }
            }
        }
    }

    result.value = interpolated;

    if (computeGradient) {
        if (divideBySpacing) {
            result.gradient.x = dx / gridSpacing[0];
            result.gradient.y = dy / gridSpacing[1];
            result.gradient.z = dz / gridSpacing[2];
        } else {
            result.gradient.x = dx;
            result.gradient.y = dy;
            result.gradient.z = dz;
        }
    }

    return result;
}

/**
 * Triquintic Hermite interpolation (method 3).
 * Uses 8 corner points with 27 derivatives each (216 coefficients).
 * Requires precomputed analytical derivatives in gridDerivatives.
 * Provides C4 continuity (smoothest option).
 *
 * Derivative storage (RASPA3 order): gridDerivatives[deriv_idx * totalPoints + point_idx]
 */
__device__ inline InterpolationResult triquinticInterpolate(
    const float* __restrict__ gridValues,
    const float* __restrict__ gridDerivatives,
    const int* __restrict__ gridCounts,
    const float* __restrict__ gridSpacing,
    float originX, float originY, float originZ,
    float3 position,
    bool computeGradient = true,
    bool divideBySpacing = true)
{
    InterpolationResult result;
    result.value = 0.0f;
    result.gradient = make_float3(0.0f, 0.0f, 0.0f);

    if (gridDerivatives == nullptr) {
        result.isInside = false;
        return result;
    }

    int ix, iy, iz;
    float fx, fy, fz;
    result.isInside = computeGridCell(position, gridCounts, gridSpacing,
                                       originX, originY, originZ,
                                       ix, iy, iz, fx, fy, fz);

    if (!result.isInside) return result;

    int nyz = gridCounts[1] * gridCounts[2];
    int totalPoints = gridCounts[0] * gridCounts[1] * gridCounts[2];

    int corners[8][3] = {
        {ix, iy, iz}, {ix+1, iy, iz}, {ix, iy+1, iz}, {ix+1, iy+1, iz},
        {ix, iy, iz+1}, {ix+1, iy, iz+1}, {ix, iy+1, iz+1}, {ix+1, iy+1, iz+1}
    };

    // Gather derivatives in DERIVATIVE-MAJOR layout: X[deriv_idx * 8 + corner_idx]
    float X[216];
    for (int d = 0; d < 27; d++) {
        for (int c = 0; c < 8; c++) {
            int point_idx = corners[c][0] * nyz + corners[c][1] * gridCounts[2] + corners[c][2];
            X[d * 8 + c] = gridDerivatives[d * totalPoints + point_idx];
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
                if (computeGradient) {
                    if (i > 0) dvalue_dx += coeff * i * sx_pow[i-1] * sy_pow[j] * sz_pow[k];
                    if (j > 0) dvalue_dy += coeff * j * sx_pow[i] * sy_pow[j-1] * sz_pow[k];
                    if (k > 0) dvalue_dz += coeff * k * sx_pow[i] * sy_pow[j] * sz_pow[k-1];
                }
            }
        }
    }

    result.value = value;

    if (computeGradient) {
        if (divideBySpacing) {
            result.gradient.x = dvalue_dx / gridSpacing[0];
            result.gradient.y = dvalue_dy / gridSpacing[1];
            result.gradient.z = dvalue_dz / gridSpacing[2];
        } else {
            result.gradient.x = dvalue_dx;
            result.gradient.y = dvalue_dy;
            result.gradient.z = dvalue_dz;
        }
    }

    return result;
}

/**
 * Generic grid interpolation dispatcher.
 *
 * @param gridValues      Grid data array [nx * ny * nz]
 * @param gridDerivatives Derivative array (required for methods 2,3; can be nullptr for 0,1)
 * @param gridCounts      Grid dimensions {nx, ny, nz}
 * @param gridSpacing     Grid spacing {dx, dy, dz} in nm
 * @param originX/Y/Z     Grid origin coordinates in nm
 * @param position        Query position in absolute coordinates (nm)
 * @param method          0=trilinear, 1=bspline, 2=tricubic, 3=triquintic
 * @param computeGradient Whether to compute the gradient
 * @param divideBySpacing If true, gradient is in physical units (per nm).
 *                        If false, gradient is in unit cell coords (for chain rule application).
 * @return InterpolationResult with value and gradient
 */
__device__ inline InterpolationResult interpolateGrid(
    const float* __restrict__ gridValues,
    const float* __restrict__ gridDerivatives,
    const int* __restrict__ gridCounts,
    const float* __restrict__ gridSpacing,
    float originX, float originY, float originZ,
    float3 position,
    int method,
    bool computeGradient = true,
    bool divideBySpacing = true)
{
    switch (method) {
        case 1:
            return bsplineInterpolate(gridValues, gridCounts, gridSpacing,
                                      originX, originY, originZ, position, computeGradient, divideBySpacing);
        case 2:
            return tricubicInterpolate(gridValues, gridDerivatives, gridCounts, gridSpacing,
                                       originX, originY, originZ, position, computeGradient, divideBySpacing);
        case 3:
            return triquinticInterpolate(gridValues, gridDerivatives, gridCounts, gridSpacing,
                                         originX, originY, originZ, position, computeGradient, divideBySpacing);
        default:
            return trilinearInterpolate(gridValues, gridCounts, gridSpacing,
                                        originX, originY, originZ, position, computeGradient, divideBySpacing);
    }
}

/**
 * Interpolate multiple grids at the same position (values only, no gradients).
 * Useful for GBSA where we need to interpolate HCT, N, A, B grids at once.
 * All grids must have the same dimensions and spacing.
 *
 * @param gridValues      Array of grid pointers
 * @param numGrids        Number of grids to interpolate
 * @param gridCounts      Grid dimensions {nx, ny, nz}
 * @param gridSpacing     Grid spacing {dx, dy, dz} in nm
 * @param originX/Y/Z     Grid origin coordinates in nm
 * @param position        Query position in absolute coordinates (nm)
 * @param method          Interpolation method: 0=trilinear, 1=bspline
 * @param results         Output array of interpolated values (length numGrids)
 * @return true if position is inside grid bounds
 */
__device__ inline bool interpolateMultipleGrids(
    const float* const* gridValues,
    int numGrids,
    const int* __restrict__ gridCounts,
    const float* __restrict__ gridSpacing,
    float originX, float originY, float originZ,
    float3 position,
    int method,
    float* results)
{
    int ix, iy, iz;
    float fx, fy, fz;
    bool isInside = computeGridCell(position, gridCounts, gridSpacing,
                                     originX, originY, originZ,
                                     ix, iy, iz, fx, fy, fz);

    if (!isInside) {
        for (int g = 0; g < numGrids; g++) {
            results[g] = 0.0f;
        }
        return false;
    }

    int nyz = gridCounts[1] * gridCounts[2];

    // Initialize results
    for (int g = 0; g < numGrids; g++) {
        results[g] = 0.0f;
    }

    if (method == 1) {
        // B-spline interpolation (4x4x4)
        float bx[4] = {bspline_basis0(fx), bspline_basis1(fx), bspline_basis2(fx), bspline_basis3(fx)};
        float by[4] = {bspline_basis0(fy), bspline_basis1(fy), bspline_basis2(fy), bspline_basis3(fy)};
        float bz[4] = {bspline_basis0(fz), bspline_basis1(fz), bspline_basis2(fz), bspline_basis3(fz)};

        for (int i = 0; i < 4; i++) {
            int gx = min(max(ix - 1 + i, 0), gridCounts[0] - 1);
            for (int j = 0; j < 4; j++) {
                int gy = min(max(iy - 1 + j, 0), gridCounts[1] - 1);
                for (int k = 0; k < 4; k++) {
                    int gz = min(max(iz - 1 + k, 0), gridCounts[2] - 1);
                    int gridIdx = gx * nyz + gy * gridCounts[2] + gz;
                    float weight = bx[i] * by[j] * bz[k];

                    for (int g = 0; g < numGrids; g++) {
                        results[g] += weight * gridValues[g][gridIdx];
                    }
                }
            }
        }
    } else {
        // Trilinear interpolation (2x2x2)
        float ox = 1.0f - fx;
        float oy = 1.0f - fy;
        float oz = 1.0f - fz;

        float weights[8] = {
            ox * oy * oz, ox * oy * fz, ox * fy * oz, ox * fy * fz,
            fx * oy * oz, fx * oy * fz, fx * fy * oz, fx * fy * fz
        };

        int baseIndex = ix * nyz + iy * gridCounts[2] + iz;
        int indices[8] = {
            baseIndex, baseIndex + 1,
            baseIndex + gridCounts[2], baseIndex + gridCounts[2] + 1,
            baseIndex + nyz, baseIndex + nyz + 1,
            baseIndex + nyz + gridCounts[2], baseIndex + nyz + gridCounts[2] + 1
        };

        for (int c = 0; c < 8; c++) {
            for (int g = 0; g < numGrids; g++) {
                results[g] += weights[c] * gridValues[g][indices[c]];
            }
        }
    }

    return true;
}

#endif // OPENMM_GRID_INTERPOLATION_CUH_
