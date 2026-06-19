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
 *   3 - Triquintic Hermite (8 corners + 27 derivatives each, C2 continuity)
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

    if (gridDerivatives == 0) {
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
    TricubicAccum X[64];
    for (int d = 0; d < 8; d++) {
        for (int c = 0; c < 8; c++) {
            int point_idx = corners[c][0] * nyz + corners[c][1] * gridCounts[2] + corners[c][2];
            X[d*8 + c] = gridDerivatives[derivMap[d] * totalPoints + point_idx];
        }
    }

    // Assemble + evaluate via the shared double-precision helpers (the fp32
    // 64-solve loses cross-cell C1 continuity and perturbs L-BFGS).
    TricubicAccum a[64];
    tricubicAssemble(X, a);
    TricubicAccum interpolated, dx, dy, dz;
    tricubicEvalVG(a, fx, fy, fz, &interpolated, &dx, &dy, &dz);

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
 * Provides C2 continuity (smoothest option).
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

    if (gridDerivatives == 0) {
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

    // Gather the 216 stored derivatives (DERIVATIVE-MAJOR: X[deriv*8 + corner]),
    // then assemble + evaluate via the shared double-precision helpers (the fp32
    // 216-solve loses cross-cell C2 and breaks L-BFGS / fakes non-PD Hessians).
    TriquinticAccum X[216];
    for (int d = 0; d < 27; d++) {
        for (int c = 0; c < 8; c++) {
            int point_idx = corners[c][0] * nyz + corners[c][1] * gridCounts[2] + corners[c][2];
            X[d * 8 + c] = gridDerivatives[d * totalPoints + point_idx];
        }
    }

    TriquinticAccum a[216];
    triquinticAssemble(X, a);

    TriquinticAccum value, dvalue_dx, dvalue_dy, dvalue_dz;
    triquinticEvalVG(a, fx, fy, fz, &value, &dvalue_dx, &dvalue_dy, &dvalue_dz);

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
 * Quintic B-spline interpolation (method 4).
 * Uses 6x6x6 = 216 grid points with quintic (degree 5) B-spline basis functions.
 * Requires prefiltered grid values (quintic B-spline prefilter, pentadiagonal solver).
 * No analytical derivatives needed - uses only grid values.
 * Provides C3 continuity when combined with quintic prefilter.
 */
__device__ inline InterpolationResult quinticBsplineInterpolate(
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

    // Precompute quintic B-spline basis functions (6 per axis)
    float bx[6] = {qbspline_basis0(fx), qbspline_basis1(fx), qbspline_basis2(fx),
                   qbspline_basis3(fx), qbspline_basis4(fx), qbspline_basis5(fx)};
    float by[6] = {qbspline_basis0(fy), qbspline_basis1(fy), qbspline_basis2(fy),
                   qbspline_basis3(fy), qbspline_basis4(fy), qbspline_basis5(fy)};
    float bz[6] = {qbspline_basis0(fz), qbspline_basis1(fz), qbspline_basis2(fz),
                   qbspline_basis3(fz), qbspline_basis4(fz), qbspline_basis5(fz)};

    float dbx[6], dby[6], dbz[6];
    if (computeGradient) {
        dbx[0] = qbspline_deriv0(fx); dbx[1] = qbspline_deriv1(fx);
        dbx[2] = qbspline_deriv2(fx); dbx[3] = qbspline_deriv3(fx);
        dbx[4] = qbspline_deriv4(fx); dbx[5] = qbspline_deriv5(fx);
        dby[0] = qbspline_deriv0(fy); dby[1] = qbspline_deriv1(fy);
        dby[2] = qbspline_deriv2(fy); dby[3] = qbspline_deriv3(fy);
        dby[4] = qbspline_deriv4(fy); dby[5] = qbspline_deriv5(fy);
        dbz[0] = qbspline_deriv0(fz); dbz[1] = qbspline_deriv1(fz);
        dbz[2] = qbspline_deriv2(fz); dbz[3] = qbspline_deriv3(fz);
        dbz[4] = qbspline_deriv4(fz); dbz[5] = qbspline_deriv5(fz);
    }

    float interpolated = 0.0f;
    float dvdx = 0.0f, dvdy = 0.0f, dvdz = 0.0f;

    // 6x6x6 stencil: offsets -2..+3 from cell corner
    for (int i = 0; i < 6; i++) {
        int gx = min(max(ix - 2 + i, 0), gridCounts[0] - 1);
        for (int j = 0; j < 6; j++) {
            int gy = min(max(iy - 2 + j, 0), gridCounts[1] - 1);
            for (int k = 0; k < 6; k++) {
                int gz = min(max(iz - 2 + k, 0), gridCounts[2] - 1);
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
 * Generic grid interpolation dispatcher.
 *
 * @param gridValues      Grid data array [nx * ny * nz]
 * @param gridDerivatives Derivative array (required for methods 2,3; can be 0 for 0,1,4)
 * @param gridCounts      Grid dimensions {nx, ny, nz}
 * @param gridSpacing     Grid spacing {dx, dy, dz} in nm
 * @param originX/Y/Z     Grid origin coordinates in nm
 * @param position        Query position in absolute coordinates (nm)
 * @param method          0=trilinear, 1=bspline, 2=tricubic, 3=triquintic Hermite, 4=quintic bspline
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
        case 4:
            return quinticBsplineInterpolate(gridValues, gridCounts, gridSpacing,
                                             originX, originY, originZ, position, computeGradient, divideBySpacing);
        default:
            return trilinearInterpolate(gridValues, gridCounts, gridSpacing,
                                        originX, originY, originZ, position, computeGradient, divideBySpacing);
    }
}

// Maximum number of grids for multi-grid interpolation (static allocation)
#define MULTI_GRID_MAX 8

/**
 * Result of multi-grid interpolation containing values and gradients.
 * Uses static allocation for CUDA efficiency.
 */
struct MultiGridResult {
    float values[MULTI_GRID_MAX];      // Interpolated values
    float3 gradients[MULTI_GRID_MAX];  // Gradients in real space (per nm)
    int numGrids;                       // Actual number of grids used
    bool isInside;                      // Whether query point was inside grid
};

/**
 * Trilinear interpolation of multiple grids with gradients.
 * Computes grid cell once and applies to all grids efficiently.
 *
 * @param gridValues      Array of grid pointers (up to MULTI_GRID_MAX)
 * @param numGrids        Number of grids to interpolate
 * @param gridCounts      Grid dimensions {nx, ny, nz}
 * @param gridSpacing     Grid spacing {dx, dy, dz} in nm
 * @param originX/Y/Z     Grid origin coordinates in nm
 * @param position        Query position in absolute coordinates (nm)
 * @return MultiGridResult with values and gradients for all grids
 */
__device__ inline MultiGridResult trilinearInterpolateMultipleWithGradients(
    const float* const* gridValues,
    int numGrids,
    const int* __restrict__ gridCounts,
    const float* __restrict__ gridSpacing,
    float originX, float originY, float originZ,
    float3 position)
{
    MultiGridResult result;
    result.numGrids = numGrids;

    // Initialize to zero
    for (int g = 0; g < numGrids; g++) {
        result.values[g] = 0.0f;
        result.gradients[g] = make_float3(0.0f, 0.0f, 0.0f);
    }

    int ix, iy, iz;
    float fx, fy, fz;
    result.isInside = computeGridCell(position, gridCounts, gridSpacing,
                                       originX, originY, originZ,
                                       ix, iy, iz, fx, fy, fz);

    if (!result.isInside) {
        return result;
    }

    // Precompute complementary fractions
    float ox = 1.0f - fx;
    float oy = 1.0f - fy;
    float oz = 1.0f - fz;

    // Grid indexing
    int nyz = gridCounts[1] * gridCounts[2];
    int nz = gridCounts[2];

    // Corner indices: c[i][j][k] where i,j,k ∈ {0,1}
    int baseIndex = ix * nyz + iy * nz + iz;
    int c000 = baseIndex;
    int c001 = baseIndex + 1;
    int c010 = baseIndex + nz;
    int c011 = baseIndex + nz + 1;
    int c100 = baseIndex + nyz;
    int c101 = baseIndex + nyz + 1;
    int c110 = baseIndex + nyz + nz;
    int c111 = baseIndex + nyz + nz + 1;

    // Inverse spacing for gradient conversion
    float invSpacingX = 1.0f / gridSpacing[0];
    float invSpacingY = 1.0f / gridSpacing[1];
    float invSpacingZ = 1.0f / gridSpacing[2];

    // Process each grid
    for (int g = 0; g < numGrids; g++) {
        const float* grid = gridValues[g];

        // Load corner values
        float v000 = grid[c000];
        float v001 = grid[c001];
        float v010 = grid[c010];
        float v011 = grid[c011];
        float v100 = grid[c100];
        float v101 = grid[c101];
        float v110 = grid[c110];
        float v111 = grid[c111];

        // Trilinear interpolation for value
        // v = ox*oy*oz*v000 + ox*oy*fz*v001 + ox*fy*oz*v010 + ox*fy*fz*v011
        //   + fx*oy*oz*v100 + fx*oy*fz*v101 + fx*fy*oz*v110 + fx*fy*fz*v111
        float vmm = oz * v000 + fz * v001;  // v at (0, 0, z)
        float vmp = oz * v010 + fz * v011;  // v at (0, 1, z)
        float vpm = oz * v100 + fz * v101;  // v at (1, 0, z)
        float vpp = oz * v110 + fz * v111;  // v at (1, 1, z)

        float vm = oy * vmm + fy * vmp;     // v at (0, y, z)
        float vp = oy * vpm + fy * vpp;     // v at (1, y, z)

        result.values[g] = ox * vm + fx * vp;

        // Analytical gradient in fractional coordinates
        // dv/dfx = vp - vm
        float dv_dfx = vp - vm;

        // dv/dfy = ox*(vmp - vmm) + fx*(vpp - vpm)
        float dv_dfy = ox * (vmp - vmm) + fx * (vpp - vpm);

        // dv/dfz = ox*(oy*(v001-v000) + fy*(v011-v010)) + fx*(oy*(v101-v100) + fy*(v111-v110))
        float dv_dfz = ox * (oy * (v001 - v000) + fy * (v011 - v010)) +
                       fx * (oy * (v101 - v100) + fy * (v111 - v110));

        // Convert to real-space gradients
        result.gradients[g].x = dv_dfx * invSpacingX;
        result.gradients[g].y = dv_dfy * invSpacingY;
        result.gradients[g].z = dv_dfz * invSpacingZ;
    }

    return result;
}

/**
 * B-spline interpolation of multiple grids with gradients.
 * Uses 4x4x4 stencil for C2 continuous interpolation.
 *
 * @param gridValues      Array of grid pointers (up to MULTI_GRID_MAX)
 * @param numGrids        Number of grids to interpolate
 * @param gridCounts      Grid dimensions {nx, ny, nz}
 * @param gridSpacing     Grid spacing {dx, dy, dz} in nm
 * @param originX/Y/Z     Grid origin coordinates in nm
 * @param position        Query position in absolute coordinates (nm)
 * @return MultiGridResult with values and gradients for all grids
 */
__device__ inline MultiGridResult bsplineInterpolateMultipleWithGradients(
    const float* const* gridValues,
    int numGrids,
    const int* __restrict__ gridCounts,
    const float* __restrict__ gridSpacing,
    float originX, float originY, float originZ,
    float3 position)
{
    MultiGridResult result;
    result.numGrids = numGrids;

    // Initialize to zero
    for (int g = 0; g < numGrids; g++) {
        result.values[g] = 0.0f;
        result.gradients[g] = make_float3(0.0f, 0.0f, 0.0f);
    }

    int ix, iy, iz;
    float fx, fy, fz;
    result.isInside = computeGridCell(position, gridCounts, gridSpacing,
                                       originX, originY, originZ,
                                       ix, iy, iz, fx, fy, fz);

    if (!result.isInside) {
        return result;
    }

    int nyz = gridCounts[1] * gridCounts[2];
    int nz = gridCounts[2];

    // Precompute B-spline basis functions and derivatives
    float bx[4] = {bspline_basis0(fx), bspline_basis1(fx), bspline_basis2(fx), bspline_basis3(fx)};
    float by[4] = {bspline_basis0(fy), bspline_basis1(fy), bspline_basis2(fy), bspline_basis3(fy)};
    float bz[4] = {bspline_basis0(fz), bspline_basis1(fz), bspline_basis2(fz), bspline_basis3(fz)};

    float dbx[4] = {bspline_deriv0(fx), bspline_deriv1(fx), bspline_deriv2(fx), bspline_deriv3(fx)};
    float dby[4] = {bspline_deriv0(fy), bspline_deriv1(fy), bspline_deriv2(fy), bspline_deriv3(fy)};
    float dbz[4] = {bspline_deriv0(fz), bspline_deriv1(fz), bspline_deriv2(fz), bspline_deriv3(fz)};

    // Inverse spacing for gradient conversion
    float invSpacingX = 1.0f / gridSpacing[0];
    float invSpacingY = 1.0f / gridSpacing[1];
    float invSpacingZ = 1.0f / gridSpacing[2];

    // Loop over 4x4x4 stencil
    for (int i = 0; i < 4; i++) {
        int gx = min(max(ix - 1 + i, 0), gridCounts[0] - 1);
        for (int j = 0; j < 4; j++) {
            int gy = min(max(iy - 1 + j, 0), gridCounts[1] - 1);
            for (int k = 0; k < 4; k++) {
                int gz = min(max(iz - 1 + k, 0), gridCounts[2] - 1);
                int gridIdx = gx * nyz + gy * nz + gz;

                float weight = bx[i] * by[j] * bz[k];
                float dwdx = dbx[i] * by[j] * bz[k];
                float dwdy = bx[i] * dby[j] * bz[k];
                float dwdz = bx[i] * by[j] * dbz[k];

                for (int g = 0; g < numGrids; g++) {
                    float val = gridValues[g][gridIdx];
                    result.values[g] += weight * val;
                    result.gradients[g].x += dwdx * val;
                    result.gradients[g].y += dwdy * val;
                    result.gradients[g].z += dwdz * val;
                }
            }
        }
    }

    // Convert gradients to real space
    for (int g = 0; g < numGrids; g++) {
        result.gradients[g].x *= invSpacingX;
        result.gradients[g].y *= invSpacingY;
        result.gradients[g].z *= invSpacingZ;
    }

    return result;
}

/**
 * Quintic B-spline interpolation of multiple grids with gradients.
 * Uses 6x6x6 stencil for C3 continuous interpolation.
 *
 * @param gridValues      Array of grid pointers (up to MULTI_GRID_MAX)
 * @param numGrids        Number of grids to interpolate
 * @param gridCounts      Grid dimensions {nx, ny, nz}
 * @param gridSpacing     Grid spacing {dx, dy, dz} in nm
 * @param originX/Y/Z     Grid origin coordinates in nm
 * @param position        Query position in absolute coordinates (nm)
 * @return MultiGridResult with values and gradients for all grids
 */
__device__ inline MultiGridResult quinticBsplineInterpolateMultipleWithGradients(
    const float* const* gridValues,
    int numGrids,
    const int* __restrict__ gridCounts,
    const float* __restrict__ gridSpacing,
    float originX, float originY, float originZ,
    float3 position)
{
    MultiGridResult result;
    result.numGrids = numGrids;

    // Initialize to zero
    for (int g = 0; g < numGrids; g++) {
        result.values[g] = 0.0f;
        result.gradients[g] = make_float3(0.0f, 0.0f, 0.0f);
    }

    int ix, iy, iz;
    float fx, fy, fz;
    result.isInside = computeGridCell(position, gridCounts, gridSpacing,
                                       originX, originY, originZ,
                                       ix, iy, iz, fx, fy, fz);

    if (!result.isInside) {
        return result;
    }

    int nyz = gridCounts[1] * gridCounts[2];
    int nz = gridCounts[2];

    // Precompute quintic B-spline basis functions and derivatives
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

    // Inverse spacing for gradient conversion
    float invSpacingX = 1.0f / gridSpacing[0];
    float invSpacingY = 1.0f / gridSpacing[1];
    float invSpacingZ = 1.0f / gridSpacing[2];

    // Loop over 6x6x6 stencil: offsets -2..+3 from cell corner
    for (int i = 0; i < 6; i++) {
        int gx = min(max(ix - 2 + i, 0), gridCounts[0] - 1);
        for (int j = 0; j < 6; j++) {
            int gy = min(max(iy - 2 + j, 0), gridCounts[1] - 1);
            for (int k = 0; k < 6; k++) {
                int gz = min(max(iz - 2 + k, 0), gridCounts[2] - 1);
                int gridIdx = gx * nyz + gy * nz + gz;

                float weight = bx[i] * by[j] * bz[k];
                float dwdx = dbx[i] * by[j] * bz[k];
                float dwdy = bx[i] * dby[j] * bz[k];
                float dwdz = bx[i] * by[j] * dbz[k];

                for (int g = 0; g < numGrids; g++) {
                    float val = gridValues[g][gridIdx];
                    result.values[g] += weight * val;
                    result.gradients[g].x += dwdx * val;
                    result.gradients[g].y += dwdy * val;
                    result.gradients[g].z += dwdz * val;
                }
            }
        }
    }

    // Convert gradients to real space
    for (int g = 0; g < numGrids; g++) {
        result.gradients[g].x *= invSpacingX;
        result.gradients[g].y *= invSpacingY;
        result.gradients[g].z *= invSpacingZ;
    }

    return result;
}

/**
 * Generic multi-grid interpolation with gradients.
 * Dispatches to appropriate method implementation.
 *
 * @param gridValues      Array of grid pointers (up to MULTI_GRID_MAX)
 * @param numGrids        Number of grids to interpolate
 * @param gridCounts      Grid dimensions {nx, ny, nz}
 * @param gridSpacing     Grid spacing {dx, dy, dz} in nm
 * @param originX/Y/Z     Grid origin coordinates in nm
 * @param position        Query position in absolute coordinates (nm)
 * @param method          0=trilinear, 1=bspline, 2=tricubic, 3=triquintic, 4=quintic bspline
 * @return MultiGridResult with values and gradients for all grids
 *
 * Note: Methods 2 and 3 (tricubic/triquintic) require derivative grids
 * and are not yet implemented for multi-grid case.
 */
__device__ inline MultiGridResult interpolateMultipleGridsWithGradients(
    const float* const* gridValues,
    int numGrids,
    const int* __restrict__ gridCounts,
    const float* __restrict__ gridSpacing,
    float originX, float originY, float originZ,
    float3 position,
    int method)
{
    switch (method) {
        case 1:
            return bsplineInterpolateMultipleWithGradients(
                gridValues, numGrids, gridCounts, gridSpacing,
                originX, originY, originZ, position);
        case 4:
            return quinticBsplineInterpolateMultipleWithGradients(
                gridValues, numGrids, gridCounts, gridSpacing,
                originX, originY, originZ, position);
        case 2:
        case 3:
            // Tricubic and triquintic multi-grid not yet implemented
            // Fall through to trilinear for now
        default:
            return trilinearInterpolateMultipleWithGradients(
                gridValues, numGrids, gridCounts, gridSpacing,
                originX, originY, originZ, position);
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
