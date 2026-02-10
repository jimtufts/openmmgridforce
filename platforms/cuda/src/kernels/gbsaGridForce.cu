/* -------------------------------------------------------------------------- *
 *                              OpenMMGridForce                               *
 * -------------------------------------------------------------------------- *
 * CUDA kernels for GBSAGridForce - Grid-based Generalized Born solvation.
 *
 * Workflow:
 * 1. computeReceptorHCT - Interpolate grid, apply binned correction
 * 2. computeLigandHCT - Pairwise HCT within ligand (O(N²))
 * 3. computeBornRadii - OBC-II correction formula
 * 4. computeGBEnergy - Still equation pairwise (O(N²)) with forces
 * 5. computeSAEnergy - Optional ACE surface area term
 * -------------------------------------------------------------------------- */

#include "include/GridInterpolation.cuh"
#include "include/HCTChainRule.cuh"

// Physical constants
#define DIELECTRIC_OFFSET 0.009f
#define OBC_ALPHA 1.0f
#define OBC_BETA 0.8f
#define OBC_GAMMA 4.85f

/**
 * Result of GBSA grid interpolation.
 */
struct GBSAInterpolationResult {
    float hct;           // Corrected HCT value
    float3 gradient;     // Gradient of corrected HCT w.r.t. position (real space)
    bool isInside;       // Whether position is inside grid
};

/**
 * Result of GBSA grid interpolation with analytical Hessian (second derivatives).
 */
struct GBSAHessianResult {
    float hct;           // Corrected HCT value
    float3 gradient;     // Gradient of corrected HCT w.r.t. position (real space)
    float hessian[6];    // Second derivatives: [xx, yy, zz, xy, xz, yz]
    bool isInside;       // Whether position is inside grid
};

/**
 * Interpolate GBSA grids with correction formula.
 * Supports trilinear (method=0), B-spline (method=1), tricubic (method=2),
 * and triquintic (method=3).
 *
 * Correction grids (N, A, B) can be either:
 * - Binned: [numBins * numPoints], use binOffset for radius-dependent lookup, trilinear only
 * - KDE: [27 * numPoints], smooth KDE-weighted values with derivatives, same method as HCT
 *
 * When useKDECorrections=true, ALL grids use the same interpolation method.
 * This is critical because N, A, B contribute 94% of discontinuity.
 *
 * @param position       Query position
 * @param R_i_off        Offset radius of querying atom
 * @param R_probe_off    Probe offset radius used in grid generation
 * @param gridCounts     Grid dimensions [nx, ny, nz]
 * @param gridSpacing    Uniform grid spacing (nm)
 * @param origin         Grid origin (x, y, z)
 * @param gridHctProbe   HCT values at probe radius (also stores f derivative for tricubic/triquintic)
 * @param gridHctDerivatives  HCT derivatives for tricubic (8) or triquintic (27), RASPA3 layout
 *                            Shape: (n_derivs, total_points), nullptr for trilinear/bspline
 * @param gridCorrectionN, A, B  Correction grids: [numBins * numPoints] or [27 * numPoints] for KDE
 * @param binOffset      Offset into correction grids for selected bin (ignored if useKDECorrections=true)
 * @param method         Interpolation method (0=trilinear, 1=bspline, 2=tricubic, 3=triquintic)
 * @param computeGradient Whether to compute gradient
 * @param useKDECorrections If true, correction grids are KDE format with 27 derivatives each
 */
__device__ inline GBSAInterpolationResult interpolateGBSAGrids(
    float3 position,
    float R_i_off,
    float R_probe_off,
    const int* __restrict__ gridCounts,
    float gridSpacing,
    float originX, float originY, float originZ,
    const float* __restrict__ gridHctProbe,
    const float* __restrict__ gridHctDerivatives,
    const float* __restrict__ gridCorrectionN,
    const float* __restrict__ gridCorrectionA,
    const float* __restrict__ gridCorrectionB,
    int binOffset,
    int method,
    bool computeGradient,
    bool useKDECorrections = false,
    bool hasBinnedKDEDerivatives = false)
{
    GBSAInterpolationResult result;
    result.hct = 0.0f;
    result.gradient = make_float3(0.0f, 0.0f, 0.0f);

    // Use shared library for grid cell computation
    float gridSpacingArr[3] = {gridSpacing, gridSpacing, gridSpacing};
    int ix, iy, iz;
    float fx, fy, fz;
    result.isInside = computeGridCell(position, gridCounts, gridSpacingArr,
                                       originX, originY, originZ,
                                       ix, iy, iz, fx, fy, fz);

    if (!result.isInside) {
        return result;
    }

    int ny = gridCounts[1];
    int nz = gridCounts[2];
    int nyz = ny * nz;

    // Corner indices
    int c000 = ix * nyz + iy * nz + iz;
    int c001 = c000 + 1;
    int c010 = c000 + nz;
    int c011 = c010 + 1;
    int c100 = c000 + nyz;
    int c101 = c100 + 1;
    int c110 = c100 + nz;
    int c111 = c110 + 1;

    float invSpacing = 1.0f / gridSpacing;
    float ox = 1.0f - fx;
    float oy = 1.0f - fy;
    float oz = 1.0f - fz;

    if (method == 1 || method == 4) {
        // B-spline interpolation using shared library
        // method 1 = cubic B-spline (4x4x4), method 4 = quintic B-spline (6x6x6)
        // Compute correction offset based on format
        int corrOffset;
        if (hasBinnedKDEDerivatives) {
            int numPoints = gridCounts[0] * gridCounts[1] * gridCounts[2];
            int binIdx = binOffset / numPoints;
            corrOffset = binIdx * 27 * numPoints;  // Values are at deriv=0
        } else if (useKDECorrections) {
            corrOffset = 0;
        } else {
            corrOffset = binOffset;
        }
        const float* grids[4] = {gridHctProbe,
                                  gridCorrectionN + corrOffset,
                                  gridCorrectionA + corrOffset,
                                  gridCorrectionB + corrOffset};

        MultiGridResult mgResult = (method == 4)
            ? quinticBsplineInterpolateMultipleWithGradients(
                grids, 4, gridCounts, gridSpacingArr,
                originX, originY, originZ, position)
            : bsplineInterpolateMultipleWithGradients(
                grids, 4, gridCounts, gridSpacingArr,
                originX, originY, originZ, position);

        float hct = mgResult.values[0];
        float N = mgResult.values[1];
        float A = mgResult.values[2];
        float B = mgResult.values[3];

        // Always apply correction - N, A, B all go to zero together
        // in regions far from receptor, so correction naturally vanishes
        {
            float invRi = 1.0f / R_i_off;
            float invRp = 1.0f / R_probe_off;
            float delta = invRi - invRp;
            float sigma = invRi + invRp;
            float logTerm = logf(R_i_off / R_probe_off);

            result.hct = hct + delta * (N - 0.25f * A * sigma) + B * logTerm;

            if (computeGradient) {
                float dCorr_dN = delta;
                float dCorr_dA = -0.25f * delta * sigma;
                float dCorr_dB = logTerm;

                float3 dCorr;
                dCorr.x = dCorr_dN * mgResult.gradients[1].x +
                    dCorr_dA * mgResult.gradients[2].x + dCorr_dB * mgResult.gradients[3].x;
                dCorr.y = dCorr_dN * mgResult.gradients[1].y +
                    dCorr_dA * mgResult.gradients[2].y + dCorr_dB * mgResult.gradients[3].y;
                dCorr.z = dCorr_dN * mgResult.gradients[1].z +
                    dCorr_dA * mgResult.gradients[2].z + dCorr_dB * mgResult.gradients[3].z;

                result.gradient.x = mgResult.gradients[0].x + dCorr.x;
                result.gradient.y = mgResult.gradients[0].y + dCorr.y;
                result.gradient.z = mgResult.gradients[0].z + dCorr.z;
            }
        }
    } else if (method == 2 || method == 3) {
        // Tricubic (2) or triquintic (3) interpolation for HCT
        // Correction grids use same method when KDE mode, trilinear when binned mode

        if (gridHctDerivatives == nullptr) {
            // No derivatives provided, fall back to trilinear
            // (will be handled by the else branch below)
            method = 0;
        } else {
            // Use higher-order interpolation for HCT grid
            InterpolationResult hctResult;
            if (method == 2) {
                hctResult = tricubicInterpolate(
                    gridHctProbe, gridHctDerivatives, gridCounts, gridSpacingArr,
                    originX, originY, originZ, position, computeGradient, true);
            } else {
                hctResult = triquinticInterpolate(
                    gridHctProbe, gridHctDerivatives, gridCounts, gridSpacingArr,
                    originX, originY, originZ, position, computeGradient, true);
            }

            float hct = hctResult.value;
            float N, A, B;
            float3 nGrad = make_float3(0.0f, 0.0f, 0.0f);
            float3 aGrad = make_float3(0.0f, 0.0f, 0.0f);
            float3 bGrad = make_float3(0.0f, 0.0f, 0.0f);

            if (useKDECorrections) {
                // KDE mode: use same interpolation method for correction grids
                // KDE grids have 27 derivatives in RASPA3 layout
                // Binned+KDE derivatives layout: [bin * 27 * numPoints + deriv * numPoints + point]
                // Compute bin offset for binned+KDE derivatives format
                int numPoints = gridCounts[0] * gridCounts[1] * gridCounts[2];
                int binIdx = (binOffset > 0) ? (binOffset / numPoints) : 0;
                int binDerivOffset = binIdx * 27 * numPoints;

                // Get pointers to this bin's data
                const float* nGridBin = gridCorrectionN + binDerivOffset;
                const float* aGridBin = gridCorrectionA + binDerivOffset;
                const float* bGridBin = gridCorrectionB + binDerivOffset;

                InterpolationResult nResult, aResult, bResult;
                if (method == 2) {
                    nResult = tricubicInterpolate(
                        nGridBin, nGridBin, gridCounts, gridSpacingArr,
                        originX, originY, originZ, position, computeGradient, true);
                    aResult = tricubicInterpolate(
                        aGridBin, aGridBin, gridCounts, gridSpacingArr,
                        originX, originY, originZ, position, computeGradient, true);
                    bResult = tricubicInterpolate(
                        bGridBin, bGridBin, gridCounts, gridSpacingArr,
                        originX, originY, originZ, position, computeGradient, true);
                } else {
                    nResult = triquinticInterpolate(
                        nGridBin, nGridBin, gridCounts, gridSpacingArr,
                        originX, originY, originZ, position, computeGradient, true);
                    aResult = triquinticInterpolate(
                        aGridBin, aGridBin, gridCounts, gridSpacingArr,
                        originX, originY, originZ, position, computeGradient, true);
                    bResult = triquinticInterpolate(
                        bGridBin, bGridBin, gridCounts, gridSpacingArr,
                        originX, originY, originZ, position, computeGradient, true);
                }
                N = nResult.value;
                A = aResult.value;
                B = bResult.value;
                nGrad = nResult.gradient;
                aGrad = aResult.gradient;
                bGrad = bResult.gradient;
            } else {
                // Binned mode: use trilinear for correction grids N, A, B
                float N000 = gridCorrectionN[binOffset + c000];
                float N001 = gridCorrectionN[binOffset + c001];
                float N010 = gridCorrectionN[binOffset + c010];
                float N011 = gridCorrectionN[binOffset + c011];
                float N100 = gridCorrectionN[binOffset + c100];
                float N101 = gridCorrectionN[binOffset + c101];
                float N110 = gridCorrectionN[binOffset + c110];
                float N111 = gridCorrectionN[binOffset + c111];

                float Nmm = oz * N000 + fz * N001;
                float Nmp = oz * N010 + fz * N011;
                float Npm = oz * N100 + fz * N101;
                float Npp = oz * N110 + fz * N111;
                float Nm = oy * Nmm + fy * Nmp;
                float Np = oy * Npm + fy * Npp;
                N = ox * Nm + fx * Np;

                // Compute trilinear gradients for N
                if (computeGradient) {
                    nGrad.x = (Np - Nm) * invSpacing;
                    nGrad.y = (ox * (Nmp - Nmm) + fx * (Npp - Npm)) * invSpacing;
                    nGrad.z = (ox * (oy * (N001 - N000) + fy * (N011 - N010)) +
                               fx * (oy * (N101 - N100) + fy * (N111 - N110))) * invSpacing;
                }

                // Load A, B values
                float A000 = gridCorrectionA[binOffset + c000];
                float A001 = gridCorrectionA[binOffset + c001];
                float A010 = gridCorrectionA[binOffset + c010];
                float A011 = gridCorrectionA[binOffset + c011];
                float A100 = gridCorrectionA[binOffset + c100];
                float A101 = gridCorrectionA[binOffset + c101];
                float A110 = gridCorrectionA[binOffset + c110];
                float A111 = gridCorrectionA[binOffset + c111];

                float Amm = oz * A000 + fz * A001;
                float Amp = oz * A010 + fz * A011;
                float Apm = oz * A100 + fz * A101;
                float App = oz * A110 + fz * A111;
                float Am = oy * Amm + fy * Amp;
                float Ap = oy * Apm + fy * App;
                A = ox * Am + fx * Ap;

                float B000 = gridCorrectionB[binOffset + c000];
                float B001 = gridCorrectionB[binOffset + c001];
                float B010 = gridCorrectionB[binOffset + c010];
                float B011 = gridCorrectionB[binOffset + c011];
                float B100 = gridCorrectionB[binOffset + c100];
                float B101 = gridCorrectionB[binOffset + c101];
                float B110 = gridCorrectionB[binOffset + c110];
                float B111 = gridCorrectionB[binOffset + c111];

                float Bmm = oz * B000 + fz * B001;
                float Bmp = oz * B010 + fz * B011;
                float Bpm = oz * B100 + fz * B101;
                float Bpp = oz * B110 + fz * B111;
                float Bm = oy * Bmm + fy * Bmp;
                float Bp = oy * Bpm + fy * Bpp;
                B = ox * Bm + fx * Bp;

                if (computeGradient) {
                    aGrad.x = (Ap - Am) * invSpacing;
                    aGrad.y = (ox * (Amp - Amm) + fx * (App - Apm)) * invSpacing;
                    aGrad.z = (ox * (oy * (A001 - A000) + fy * (A011 - A010)) +
                               fx * (oy * (A101 - A100) + fy * (A111 - A110))) * invSpacing;

                    bGrad.x = (Bp - Bm) * invSpacing;
                    bGrad.y = (ox * (Bmp - Bmm) + fx * (Bpp - Bpm)) * invSpacing;
                    bGrad.z = (ox * (oy * (B001 - B000) + fy * (B011 - B010)) +
                               fx * (oy * (B101 - B100) + fy * (B111 - B110))) * invSpacing;
                }
            }

            // Always apply correction - N, A, B all go to zero together
            {
                float invRi = 1.0f / R_i_off;
                float invRp = 1.0f / R_probe_off;
                float delta = invRi - invRp;
                float sigma = invRi + invRp;
                float logTerm = logf(R_i_off / R_probe_off);

                result.hct = hct + delta * (N - 0.25f * A * sigma) + B * logTerm;

                if (computeGradient) {
                    float dCorr_dN = delta;
                    float dCorr_dA = -0.25f * delta * sigma;
                    float dCorr_dB = logTerm;

                    float3 dCorr;
                    dCorr.x = dCorr_dN * nGrad.x + dCorr_dA * aGrad.x + dCorr_dB * bGrad.x;
                    dCorr.y = dCorr_dN * nGrad.y + dCorr_dA * aGrad.y + dCorr_dB * bGrad.y;
                    dCorr.z = dCorr_dN * nGrad.z + dCorr_dA * aGrad.z + dCorr_dB * bGrad.z;

                    result.gradient.x = hctResult.gradient.x + dCorr.x;
                    result.gradient.y = hctResult.gradient.y + dCorr.y;
                    result.gradient.z = hctResult.gradient.z + dCorr.z;
                }
            }

            return result;
        }
    }

    if (method == 0) {
        // Trilinear interpolation (default, optimized version)
        // Compute correction offset based on format:
        // - Pure KDE [27*numPoints]: offset=0
        // - Binned [numBins*numPoints]: offset=binOffset
        // - Binned+KDE [numBins*27*numPoints]: offset=binIdx*27*numPoints (values at deriv=0)
        int corrOffset;
        if (hasBinnedKDEDerivatives) {
            // Binned+KDE: binOffset was computed as binIdx*numPoints, need binIdx*27*numPoints
            int numPoints = gridCounts[0] * gridCounts[1] * gridCounts[2];
            int binIdx = binOffset / numPoints;
            corrOffset = binIdx * 27 * numPoints;  // Values are at deriv=0
        } else if (useKDECorrections) {
            // Pure KDE: no binning
            corrOffset = 0;
        } else {
            // Standard binned: use binOffset directly
            corrOffset = binOffset;
        }

        // Load HCT probe values
        float h000 = gridHctProbe[c000];
        float h001 = gridHctProbe[c001];
        float h010 = gridHctProbe[c010];
        float h011 = gridHctProbe[c011];
        float h100 = gridHctProbe[c100];
        float h101 = gridHctProbe[c101];
        float h110 = gridHctProbe[c110];
        float h111 = gridHctProbe[c111];

        // Trilinear interpolation for HCT
        float vmm = oz * h000 + fz * h001;
        float vmp = oz * h010 + fz * h011;
        float vpm = oz * h100 + fz * h101;
        float vpp = oz * h110 + fz * h111;
        float vm = oy * vmm + fy * vmp;
        float vp = oy * vpm + fy * vpp;
        float hct = ox * vm + fx * vp;

        // Load and interpolate N
        float N000 = gridCorrectionN[corrOffset + c000];
        float N001 = gridCorrectionN[corrOffset + c001];
        float N010 = gridCorrectionN[corrOffset + c010];
        float N011 = gridCorrectionN[corrOffset + c011];
        float N100 = gridCorrectionN[corrOffset + c100];
        float N101 = gridCorrectionN[corrOffset + c101];
        float N110 = gridCorrectionN[corrOffset + c110];
        float N111 = gridCorrectionN[corrOffset + c111];

        float Nmm = oz * N000 + fz * N001;
        float Nmp = oz * N010 + fz * N011;
        float Npm = oz * N100 + fz * N101;
        float Npp = oz * N110 + fz * N111;
        float Nm = oy * Nmm + fy * Nmp;
        float Np = oy * Npm + fy * Npp;
        float N = ox * Nm + fx * Np;

        // Always apply correction - N, A, B all go to zero together
        {
            // Load and interpolate A, B
            float A000 = gridCorrectionA[corrOffset + c000];
            float A001 = gridCorrectionA[corrOffset + c001];
            float A010 = gridCorrectionA[corrOffset + c010];
            float A011 = gridCorrectionA[corrOffset + c011];
            float A100 = gridCorrectionA[corrOffset + c100];
            float A101 = gridCorrectionA[corrOffset + c101];
            float A110 = gridCorrectionA[corrOffset + c110];
            float A111 = gridCorrectionA[corrOffset + c111];

            float Amm = oz * A000 + fz * A001;
            float Amp = oz * A010 + fz * A011;
            float Apm = oz * A100 + fz * A101;
            float App = oz * A110 + fz * A111;
            float Am = oy * Amm + fy * Amp;
            float Ap = oy * Apm + fy * App;
            float A = ox * Am + fx * Ap;

            float B000 = gridCorrectionB[corrOffset + c000];
            float B001 = gridCorrectionB[corrOffset + c001];
            float B010 = gridCorrectionB[corrOffset + c010];
            float B011 = gridCorrectionB[corrOffset + c011];
            float B100 = gridCorrectionB[corrOffset + c100];
            float B101 = gridCorrectionB[corrOffset + c101];
            float B110 = gridCorrectionB[corrOffset + c110];
            float B111 = gridCorrectionB[corrOffset + c111];

            float Bmm = oz * B000 + fz * B001;
            float Bmp = oz * B010 + fz * B011;
            float Bpm = oz * B100 + fz * B101;
            float Bpp = oz * B110 + fz * B111;
            float Bm = oy * Bmm + fy * Bmp;
            float Bp = oy * Bpm + fy * Bpp;
            float B = ox * Bm + fx * Bp;

            // Apply correction
            float invRi = 1.0f / R_i_off;
            float invRp = 1.0f / R_probe_off;
            float delta = invRi - invRp;
            float sigma = invRi + invRp;
            float logTerm = logf(R_i_off / R_probe_off);

            result.hct = hct + delta * (N - 0.25f * A * sigma) + B * logTerm;

            if (computeGradient) {
                float dCorr_dN = delta;
                float dCorr_dA = -0.25f * delta * sigma;
                float dCorr_dB = logTerm;

                // Analytical trilinear gradients
                float dh_dfx = vp - vm;
                float dh_dfy = ox * (vmp - vmm) + fx * (vpp - vpm);
                float dh_dfz = ox * (oy * (h001 - h000) + fy * (h011 - h010)) +
                               fx * (oy * (h101 - h100) + fy * (h111 - h110));

                float dN_dfx = Np - Nm;
                float dN_dfy = ox * (Nmp - Nmm) + fx * (Npp - Npm);
                float dN_dfz = ox * (oy * (N001 - N000) + fy * (N011 - N010)) +
                               fx * (oy * (N101 - N100) + fy * (N111 - N110));

                float dA_dfx = Ap - Am;
                float dA_dfy = ox * (Amp - Amm) + fx * (App - Apm);
                float dA_dfz = ox * (oy * (A001 - A000) + fy * (A011 - A010)) +
                               fx * (oy * (A101 - A100) + fy * (A111 - A110));

                float dB_dfx = Bp - Bm;
                float dB_dfy = ox * (Bmp - Bmm) + fx * (Bpp - Bpm);
                float dB_dfz = ox * (oy * (B001 - B000) + fy * (B011 - B010)) +
                               fx * (oy * (B101 - B100) + fy * (B111 - B110));

                result.gradient.x = (dh_dfx + dCorr_dN * dN_dfx + dCorr_dA * dA_dfx + dCorr_dB * dB_dfx) * invSpacing;
                result.gradient.y = (dh_dfy + dCorr_dN * dN_dfy + dCorr_dA * dA_dfy + dCorr_dB * dB_dfy) * invSpacing;
                result.gradient.z = (dh_dfz + dCorr_dN * dN_dfz + dCorr_dA * dA_dfz + dCorr_dB * dB_dfz) * invSpacing;
            }
        }
    }

    return result;
}

/**
 * Interpolate GBSA grids with correction formula and compute analytical Hessian.
 * Returns corrected HCT value, gradient, AND second derivatives.
 *
 * The correction formula is LINEAR in grid values:
 *   Psi = hct + delta*(N - 0.25*A*sigma) + B*logTerm
 * so the combined Hessian is a linear combination of per-grid Hessians.
 *
 * For B-spline methods, all 4 grids share the same stencil, enabling a single
 * combined evaluation with 10 accumulators (value + 3 gradient + 6 Hessian).
 * For Hermite with KDE, derivatives are combined before polynomial evaluation.
 * For Hermite with binned corrections, HCT uses polynomial Hessian and
 * correction grids use trilinear (pure d2 = 0, mixed d2 = analytical).
 */
__device__ inline GBSAHessianResult interpolateGBSAGridsWithHessian(
    float3 position,
    float R_i_off,
    float R_probe_off,
    const int* __restrict__ gridCounts,
    float gridSpacing,
    float originX, float originY, float originZ,
    const float* __restrict__ gridHctProbe,
    const float* __restrict__ gridHctDerivatives,
    const float* __restrict__ gridCorrectionN,
    const float* __restrict__ gridCorrectionA,
    const float* __restrict__ gridCorrectionB,
    int binOffset,
    int method,
    bool useKDECorrections,
    bool hasBinnedKDEDerivatives)
{
    GBSAHessianResult result;
    result.hct = 0.0f;
    result.gradient = make_float3(0.0f, 0.0f, 0.0f);
    for (int i = 0; i < 6; i++) result.hessian[i] = 0.0f;

    float gridSpacingArr[3] = {gridSpacing, gridSpacing, gridSpacing};
    int ix, iy, iz;
    float fx, fy, fz;
    result.isInside = computeGridCell(position, gridCounts, gridSpacingArr,
                                       originX, originY, originZ,
                                       ix, iy, iz, fx, fy, fz);
    if (!result.isInside) return result;

    int ny = gridCounts[1];
    int nz = gridCounts[2];
    int nyz = ny * nz;

    float invSpacing = 1.0f / gridSpacing;
    float invSpacing2 = invSpacing * invSpacing;
    float ox = 1.0f - fx;
    float oy = 1.0f - fy;
    float oz = 1.0f - fz;

    // Corner indices (for trilinear correction grids and trilinear fallback)
    int c000 = ix * nyz + iy * nz + iz;
    int c001 = c000 + 1;
    int c010 = c000 + nz;
    int c011 = c010 + 1;
    int c100 = c000 + nyz;
    int c101 = c100 + 1;
    int c110 = c100 + nz;
    int c111 = c110 + 1;

    // Correction coefficients (Psi = hct + delta*(N - 0.25*A*sigma) + B*logTerm)
    float invRi = 1.0f / R_i_off;
    float invRp = 1.0f / R_probe_off;
    float delta = invRi - invRp;
    float sigma = invRi + invRp;
    float logTerm = logf(R_i_off / R_probe_off);
    float dCorr_dN = delta;
    float dCorr_dA = -0.25f * delta * sigma;
    float dCorr_dB = logTerm;

    if (method == 1 || method == 4) {
        // =============================================================
        // B-SPLINE: All 4 grids share the same stencil.
        // Form combined value per stencil point for single accumulation.
        // =============================================================
        int corrOffset;
        if (hasBinnedKDEDerivatives) {
            int numPoints = gridCounts[0] * nyz;
            int binIdx = binOffset / numPoints;
            corrOffset = binIdx * 27 * numPoints;
        } else if (useKDECorrections) {
            corrOffset = 0;
        } else {
            corrOffset = binOffset;
        }

        float val = 0, gx = 0, gy = 0, gz = 0;
        float hxx = 0, hyy = 0, hzz = 0, hxy = 0, hxz = 0, hyz = 0;

        if (method == 1) {
            // Cubic B-spline (4x4x4 stencil)
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
                int gxi = min(max(ix - 1 + i, 0), gridCounts[0] - 1);
                for (int j = 0; j < 4; j++) {
                    int gyi = min(max(iy - 1 + j, 0), gridCounts[1] - 1);
                    for (int k = 0; k < 4; k++) {
                        int gzi = min(max(iz - 1 + k, 0), gridCounts[2] - 1);
                        int gridIdx = gxi * nyz + gyi * nz + gzi;
                        float combined = gridHctProbe[gridIdx]
                            + dCorr_dN * gridCorrectionN[corrOffset + gridIdx]
                            + dCorr_dA * gridCorrectionA[corrOffset + gridIdx]
                            + dCorr_dB * gridCorrectionB[corrOffset + gridIdx];
                        val += bx[i] * by[j] * bz[k] * combined;
                        gx  += dbx[i] * by[j] * bz[k] * combined;
                        gy  += bx[i] * dby[j] * bz[k] * combined;
                        gz  += bx[i] * by[j] * dbz[k] * combined;
                        hxx += d2bx[i] * by[j] * bz[k] * combined;
                        hyy += bx[i] * d2by[j] * bz[k] * combined;
                        hzz += bx[i] * by[j] * d2bz[k] * combined;
                        hxy += dbx[i] * dby[j] * bz[k] * combined;
                        hxz += dbx[i] * by[j] * dbz[k] * combined;
                        hyz += bx[i] * dby[j] * dbz[k] * combined;
                    }
                }
            }
        } else {
            // Quintic B-spline (6x6x6 stencil)
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
                int gxi = min(max(ix - 2 + i, 0), gridCounts[0] - 1);
                for (int j = 0; j < 6; j++) {
                    int gyi = min(max(iy - 2 + j, 0), gridCounts[1] - 1);
                    for (int k = 0; k < 6; k++) {
                        int gzi = min(max(iz - 2 + k, 0), gridCounts[2] - 1);
                        int gridIdx = gxi * nyz + gyi * nz + gzi;
                        float combined = gridHctProbe[gridIdx]
                            + dCorr_dN * gridCorrectionN[corrOffset + gridIdx]
                            + dCorr_dA * gridCorrectionA[corrOffset + gridIdx]
                            + dCorr_dB * gridCorrectionB[corrOffset + gridIdx];
                        val += bx[i] * by[j] * bz[k] * combined;
                        gx  += dbx[i] * by[j] * bz[k] * combined;
                        gy  += bx[i] * dby[j] * bz[k] * combined;
                        gz  += bx[i] * by[j] * dbz[k] * combined;
                        hxx += d2bx[i] * by[j] * bz[k] * combined;
                        hyy += bx[i] * d2by[j] * bz[k] * combined;
                        hzz += bx[i] * by[j] * d2bz[k] * combined;
                        hxy += dbx[i] * dby[j] * bz[k] * combined;
                        hxz += dbx[i] * by[j] * dbz[k] * combined;
                        hyz += bx[i] * dby[j] * dbz[k] * combined;
                    }
                }
            }
        }

        result.hct = val;
        result.gradient = make_float3(gx * invSpacing, gy * invSpacing, gz * invSpacing);
        result.hessian[0] = hxx * invSpacing2;
        result.hessian[1] = hyy * invSpacing2;
        result.hessian[2] = hzz * invSpacing2;
        result.hessian[3] = hxy * invSpacing2;
        result.hessian[4] = hxz * invSpacing2;
        result.hessian[5] = hyz * invSpacing2;

    } else if (method == 2 || method == 3) {
        // =============================================================
        // HERMITE: Polynomial evaluation with analytical Hessian.
        // =============================================================
        if (gridHctDerivatives == nullptr) {
            method = 0;  // Fall through to trilinear below
        } else {
            int totalPoints = gridCounts[0] * nyz;
            int corners[8][3] = {
                {ix, iy, iz}, {ix+1, iy, iz}, {ix, iy+1, iz}, {ix+1, iy+1, iz},
                {ix, iy, iz+1}, {ix+1, iy, iz+1}, {ix, iy+1, iz+1}, {ix+1, iy+1, iz+1}
            };

            if (useKDECorrections) {
                // KDE mode: combine all 4 grids' derivatives before polynomial evaluation.
                // Since correction is linear, combined polynomial = sum of individual polynomials.
                int numPoints = totalPoints;
                int binIdx = (binOffset > 0) ? (binOffset / numPoints) : 0;
                int binDerivOffset = hasBinnedKDEDerivatives ? binIdx * 27 * numPoints : 0;
                const float* nGrid = gridCorrectionN + binDerivOffset;
                const float* aGrid = gridCorrectionA + binDerivOffset;
                const float* bGrid = gridCorrectionB + binDerivOffset;

                if (method == 3) {
                    // Triquintic Hermite: evaluate each grid separately to avoid
                    // float32 accumulation error in 216-element matrix multiply
                    // when combining grids with different magnitudes.
                    float hct_val = 0, hct_dx = 0, hct_dy = 0, hct_dz = 0;
                    float hct_d2xx = 0, hct_d2yy = 0, hct_d2zz = 0;
                    float hct_d2xy = 0, hct_d2xz = 0, hct_d2yz = 0;
                    float corr_val = 0, corr_dx = 0, corr_dy = 0, corr_dz = 0;
                    float corr_d2xx = 0, corr_d2yy = 0, corr_d2zz = 0;
                    float corr_d2xy = 0, corr_d2xz = 0, corr_d2yz = 0;

                    float sx_pow[6], sy_pow[6], sz_pow[6];
                    sx_pow[0] = sy_pow[0] = sz_pow[0] = 1.0f;
                    for (int p = 1; p < 6; p++) {
                        sx_pow[p] = sx_pow[p-1] * fx;
                        sy_pow[p] = sy_pow[p-1] * fy;
                        sz_pow[p] = sz_pow[p-1] * fz;
                    }

                    // Evaluate each grid (HCT, N, A, B) separately
                    const float* gridPtrs[4] = {gridHctDerivatives, nGrid, aGrid, bGrid};
                    float coeffs[4] = {1.0f, dCorr_dN, dCorr_dA, dCorr_dB};

                    for (int g = 0; g < 4; g++) {
                        float X[216];
                        for (int d = 0; d < 27; d++) {
                            for (int c = 0; c < 8; c++) {
                                int pidx = corners[c][0]*nyz + corners[c][1]*nz + corners[c][2];
                                X[d*8 + c] = gridPtrs[g][d * numPoints + pidx];
                            }
                        }
                        float a[216];
                        const float scale = 0.125f;
                        for (int i = 0; i < 216; i++) {
                            a[i] = 0.0f;
                            for (int j = 0; j < 216; j++)
                                a[i] += TRIQUINTIC_COEFFICIENTS[i][j] * X[j];
                            a[i] *= scale;
                        }

                        float gv = 0, gdx = 0, gdy = 0, gdz = 0;
                        float gd2xx = 0, gd2yy = 0, gd2zz = 0, gd2xy = 0, gd2xz = 0, gd2yz = 0;
                        for (int kk = 0; kk < 6; kk++) {
                            for (int jj = 0; jj < 6; jj++) {
                                for (int ii = 0; ii < 6; ii++) {
                                    float coeff = a[ii + 6*jj + 36*kk];
                                    gv += coeff * sx_pow[ii] * sy_pow[jj] * sz_pow[kk];
                                    if (ii >= 1) gdx += coeff * ii * sx_pow[ii-1] * sy_pow[jj] * sz_pow[kk];
                                    if (jj >= 1) gdy += coeff * jj * sx_pow[ii] * sy_pow[jj-1] * sz_pow[kk];
                                    if (kk >= 1) gdz += coeff * kk * sx_pow[ii] * sy_pow[jj] * sz_pow[kk-1];
                                    if (ii >= 2) gd2xx += coeff * (ii*(ii-1)) * sx_pow[ii-2] * sy_pow[jj] * sz_pow[kk];
                                    if (jj >= 2) gd2yy += coeff * (jj*(jj-1)) * sx_pow[ii] * sy_pow[jj-2] * sz_pow[kk];
                                    if (kk >= 2) gd2zz += coeff * (kk*(kk-1)) * sx_pow[ii] * sy_pow[jj] * sz_pow[kk-2];
                                    if (ii >= 1 && jj >= 1) gd2xy += coeff * (ii*jj) * sx_pow[ii-1] * sy_pow[jj-1] * sz_pow[kk];
                                    if (ii >= 1 && kk >= 1) gd2xz += coeff * (ii*kk) * sx_pow[ii-1] * sy_pow[jj] * sz_pow[kk-1];
                                    if (jj >= 1 && kk >= 1) gd2yz += coeff * (jj*kk) * sx_pow[ii] * sy_pow[jj-1] * sz_pow[kk-1];
                                }
                            }
                        }

                        float w = coeffs[g];
                        if (g == 0) {
                            hct_val = gv; hct_dx = gdx; hct_dy = gdy; hct_dz = gdz;
                            hct_d2xx = gd2xx; hct_d2yy = gd2yy; hct_d2zz = gd2zz;
                            hct_d2xy = gd2xy; hct_d2xz = gd2xz; hct_d2yz = gd2yz;
                        } else {
                            corr_val += w * gv; corr_dx += w * gdx; corr_dy += w * gdy; corr_dz += w * gdz;
                            corr_d2xx += w * gd2xx; corr_d2yy += w * gd2yy; corr_d2zz += w * gd2zz;
                            corr_d2xy += w * gd2xy; corr_d2xz += w * gd2xz; corr_d2yz += w * gd2yz;
                        }
                    }

                    float pval = hct_val + corr_val;
                    float pdx = hct_dx + corr_dx;
                    float pdy = hct_dy + corr_dy;
                    float pdz = hct_dz + corr_dz;
                    float pd2xx = hct_d2xx + corr_d2xx;
                    float pd2yy = hct_d2yy + corr_d2yy;
                    float pd2zz = hct_d2zz + corr_d2zz;
                    float pd2xy = hct_d2xy + corr_d2xy;
                    float pd2xz = hct_d2xz + corr_d2xz;
                    float pd2yz = hct_d2yz + corr_d2yz;

                    result.hct = pval;
                    result.gradient = make_float3(pdx * invSpacing, pdy * invSpacing, pdz * invSpacing);
                    result.hessian[0] = pd2xx * invSpacing2;
                    result.hessian[1] = pd2yy * invSpacing2;
                    result.hessian[2] = pd2zz * invSpacing2;
                    result.hessian[3] = pd2xy * invSpacing2;
                    result.hessian[4] = pd2xz * invSpacing2;
                    result.hessian[5] = pd2yz * invSpacing2;
                } else {
                    // Tricubic Hermite: evaluate each grid separately
                    const int derivMap[8] = {0, 1, 2, 3, 5, 6, 8, 13};
                    float hct_val = 0, hct_dx = 0, hct_dy = 0, hct_dz = 0;
                    float hct_d2xx = 0, hct_d2yy = 0, hct_d2zz = 0;
                    float hct_d2xy = 0, hct_d2xz = 0, hct_d2yz = 0;
                    float corr_val = 0, corr_dx = 0, corr_dy = 0, corr_dz = 0;
                    float corr_d2xx = 0, corr_d2yy = 0, corr_d2zz = 0;
                    float corr_d2xy = 0, corr_d2xz = 0, corr_d2yz = 0;

                    float sx_pow[4], sy_pow[4], sz_pow[4];
                    sx_pow[0] = sy_pow[0] = sz_pow[0] = 1.0f;
                    for (int p = 1; p < 4; p++) {
                        sx_pow[p] = sx_pow[p-1] * fx;
                        sy_pow[p] = sy_pow[p-1] * fy;
                        sz_pow[p] = sz_pow[p-1] * fz;
                    }

                    const float* gridPtrs[4] = {gridHctDerivatives, nGrid, aGrid, bGrid};
                    float wt[4] = {1.0f, dCorr_dN, dCorr_dA, dCorr_dB};

                    for (int g = 0; g < 4; g++) {
                        float X[64];
                        for (int d = 0; d < 8; d++) {
                            for (int c = 0; c < 8; c++) {
                                int pidx = corners[c][0]*nyz + corners[c][1]*nz + corners[c][2];
                                X[d*8 + c] = gridPtrs[g][derivMap[d] * numPoints + pidx];
                            }
                        }
                        float a[64];
                        for (int i = 0; i < 64; i++) {
                            a[i] = 0.0f;
                            for (int j = 0; j < 64; j++)
                                a[i] += TRICUBIC_COEFFICIENTS[i][j] * X[j];
                        }

                        float gv = 0, gdx = 0, gdy = 0, gdz = 0;
                        float gd2xx = 0, gd2yy = 0, gd2zz = 0, gd2xy = 0, gd2xz = 0, gd2yz = 0;
                        for (int kk = 0; kk < 4; kk++) {
                            for (int jj = 0; jj < 4; jj++) {
                                for (int ii = 0; ii < 4; ii++) {
                                    float coeff = a[ii + 4*jj + 16*kk];
                                    gv += coeff * sx_pow[ii] * sy_pow[jj] * sz_pow[kk];
                                    if (ii >= 1) gdx += coeff * ii * sx_pow[ii-1] * sy_pow[jj] * sz_pow[kk];
                                    if (jj >= 1) gdy += coeff * jj * sx_pow[ii] * sy_pow[jj-1] * sz_pow[kk];
                                    if (kk >= 1) gdz += coeff * kk * sx_pow[ii] * sy_pow[jj] * sz_pow[kk-1];
                                    if (ii >= 2) gd2xx += coeff * (ii*(ii-1)) * sx_pow[ii-2] * sy_pow[jj] * sz_pow[kk];
                                    if (jj >= 2) gd2yy += coeff * (jj*(jj-1)) * sx_pow[ii] * sy_pow[jj-2] * sz_pow[kk];
                                    if (kk >= 2) gd2zz += coeff * (kk*(kk-1)) * sx_pow[ii] * sy_pow[jj] * sz_pow[kk-2];
                                    if (ii >= 1 && jj >= 1) gd2xy += coeff * (ii*jj) * sx_pow[ii-1] * sy_pow[jj-1] * sz_pow[kk];
                                    if (ii >= 1 && kk >= 1) gd2xz += coeff * (ii*kk) * sx_pow[ii-1] * sy_pow[jj] * sz_pow[kk-1];
                                    if (jj >= 1 && kk >= 1) gd2yz += coeff * (jj*kk) * sx_pow[ii] * sy_pow[jj-1] * sz_pow[kk-1];
                                }
                            }
                        }

                        float w = wt[g];
                        if (g == 0) {
                            hct_val = gv; hct_dx = gdx; hct_dy = gdy; hct_dz = gdz;
                            hct_d2xx = gd2xx; hct_d2yy = gd2yy; hct_d2zz = gd2zz;
                            hct_d2xy = gd2xy; hct_d2xz = gd2xz; hct_d2yz = gd2yz;
                        } else {
                            corr_val += w * gv; corr_dx += w * gdx; corr_dy += w * gdy; corr_dz += w * gdz;
                            corr_d2xx += w * gd2xx; corr_d2yy += w * gd2yy; corr_d2zz += w * gd2zz;
                            corr_d2xy += w * gd2xy; corr_d2xz += w * gd2xz; corr_d2yz += w * gd2yz;
                        }
                    }

                    float pval = hct_val + corr_val;
                    float pdx = hct_dx + corr_dx;
                    float pdy = hct_dy + corr_dy;
                    float pdz = hct_dz + corr_dz;
                    float pd2xx = hct_d2xx + corr_d2xx;
                    float pd2yy = hct_d2yy + corr_d2yy;
                    float pd2zz = hct_d2zz + corr_d2zz;
                    float pd2xy = hct_d2xy + corr_d2xy;
                    float pd2xz = hct_d2xz + corr_d2xz;
                    float pd2yz = hct_d2yz + corr_d2yz;

                    result.hct = pval;
                    result.gradient = make_float3(pdx * invSpacing, pdy * invSpacing, pdz * invSpacing);
                    result.hessian[0] = pd2xx * invSpacing2;
                    result.hessian[1] = pd2yy * invSpacing2;
                    result.hessian[2] = pd2zz * invSpacing2;
                    result.hessian[3] = pd2xy * invSpacing2;
                    result.hessian[4] = pd2xz * invSpacing2;
                    result.hessian[5] = pd2yz * invSpacing2;
                }
            } else {
                // Binned mode: HCT uses Hermite polynomial, correction grids use trilinear.
                float hct_val = 0, hct_dx = 0, hct_dy = 0, hct_dz = 0;
                float hct_d2xx = 0, hct_d2yy = 0, hct_d2zz = 0;
                float hct_d2xy = 0, hct_d2xz = 0, hct_d2yz = 0;

                if (method == 3) {
                    // Triquintic HCT Hessian
                    float X[216];
                    for (int d = 0; d < 27; d++) {
                        for (int c = 0; c < 8; c++) {
                            int pidx = corners[c][0]*nyz + corners[c][1]*nz + corners[c][2];
                            X[d*8+c] = gridHctDerivatives[d * totalPoints + pidx];
                        }
                    }
                    float a[216];
                    for (int i = 0; i < 216; i++) {
                        a[i] = 0.0f;
                        for (int j = 0; j < 216; j++) a[i] += TRIQUINTIC_COEFFICIENTS[i][j] * X[j];
                        a[i] *= 0.125f;
                    }
                    float sx_pow[6], sy_pow[6], sz_pow[6];
                    sx_pow[0] = sy_pow[0] = sz_pow[0] = 1.0f;
                    for (int p = 1; p < 6; p++) {
                        sx_pow[p] = sx_pow[p-1] * fx; sy_pow[p] = sy_pow[p-1] * fy; sz_pow[p] = sz_pow[p-1] * fz;
                    }
                    for (int kk = 0; kk < 6; kk++) {
                        for (int jj = 0; jj < 6; jj++) {
                            for (int ii = 0; ii < 6; ii++) {
                                float coeff = a[ii + 6*jj + 36*kk];
                                hct_val += coeff * sx_pow[ii] * sy_pow[jj] * sz_pow[kk];
                                if (ii >= 1) hct_dx += coeff * ii * sx_pow[ii-1] * sy_pow[jj] * sz_pow[kk];
                                if (jj >= 1) hct_dy += coeff * jj * sx_pow[ii] * sy_pow[jj-1] * sz_pow[kk];
                                if (kk >= 1) hct_dz += coeff * kk * sx_pow[ii] * sy_pow[jj] * sz_pow[kk-1];
                                if (ii >= 2) hct_d2xx += coeff * (ii*(ii-1)) * sx_pow[ii-2] * sy_pow[jj] * sz_pow[kk];
                                if (jj >= 2) hct_d2yy += coeff * (jj*(jj-1)) * sx_pow[ii] * sy_pow[jj-2] * sz_pow[kk];
                                if (kk >= 2) hct_d2zz += coeff * (kk*(kk-1)) * sx_pow[ii] * sy_pow[jj] * sz_pow[kk-2];
                                if (ii >= 1 && jj >= 1) hct_d2xy += coeff * (ii*jj) * sx_pow[ii-1] * sy_pow[jj-1] * sz_pow[kk];
                                if (ii >= 1 && kk >= 1) hct_d2xz += coeff * (ii*kk) * sx_pow[ii-1] * sy_pow[jj] * sz_pow[kk-1];
                                if (jj >= 1 && kk >= 1) hct_d2yz += coeff * (jj*kk) * sx_pow[ii] * sy_pow[jj-1] * sz_pow[kk-1];
                            }
                        }
                    }
                } else {
                    // Tricubic HCT Hessian
                    const int derivMap[8] = {0, 1, 2, 3, 5, 6, 8, 13};
                    float X[64];
                    for (int d = 0; d < 8; d++) {
                        for (int c = 0; c < 8; c++) {
                            int pidx = corners[c][0]*nyz + corners[c][1]*nz + corners[c][2];
                            X[d*8+c] = gridHctDerivatives[derivMap[d] * totalPoints + pidx];
                        }
                    }
                    float a[64];
                    for (int i = 0; i < 64; i++) {
                        a[i] = 0.0f;
                        for (int j = 0; j < 64; j++) a[i] += TRICUBIC_COEFFICIENTS[i][j] * X[j];
                    }
                    float sx_pow[4], sy_pow[4], sz_pow[4];
                    sx_pow[0] = sy_pow[0] = sz_pow[0] = 1.0f;
                    for (int p = 1; p < 4; p++) {
                        sx_pow[p] = sx_pow[p-1] * fx; sy_pow[p] = sy_pow[p-1] * fy; sz_pow[p] = sz_pow[p-1] * fz;
                    }
                    for (int kk = 0; kk < 4; kk++) {
                        for (int jj = 0; jj < 4; jj++) {
                            for (int ii = 0; ii < 4; ii++) {
                                float coeff = a[ii + 4*jj + 16*kk];
                                hct_val += coeff * sx_pow[ii] * sy_pow[jj] * sz_pow[kk];
                                if (ii >= 1) hct_dx += coeff * ii * sx_pow[ii-1] * sy_pow[jj] * sz_pow[kk];
                                if (jj >= 1) hct_dy += coeff * jj * sx_pow[ii] * sy_pow[jj-1] * sz_pow[kk];
                                if (kk >= 1) hct_dz += coeff * kk * sx_pow[ii] * sy_pow[jj] * sz_pow[kk-1];
                                if (ii >= 2) hct_d2xx += coeff * (ii*(ii-1)) * sx_pow[ii-2] * sy_pow[jj] * sz_pow[kk];
                                if (jj >= 2) hct_d2yy += coeff * (jj*(jj-1)) * sx_pow[ii] * sy_pow[jj-2] * sz_pow[kk];
                                if (kk >= 2) hct_d2zz += coeff * (kk*(kk-1)) * sx_pow[ii] * sy_pow[jj] * sz_pow[kk-2];
                                if (ii >= 1 && jj >= 1) hct_d2xy += coeff * (ii*jj) * sx_pow[ii-1] * sy_pow[jj-1] * sz_pow[kk];
                                if (ii >= 1 && kk >= 1) hct_d2xz += coeff * (ii*kk) * sx_pow[ii-1] * sy_pow[jj] * sz_pow[kk-1];
                                if (jj >= 1 && kk >= 1) hct_d2yz += coeff * (jj*kk) * sx_pow[ii] * sy_pow[jj-1] * sz_pow[kk-1];
                            }
                        }
                    }
                }

                // Trilinear correction: combine N, A, B into one field at 8 corners
                float corr000 = dCorr_dN * gridCorrectionN[binOffset + c000]
                              + dCorr_dA * gridCorrectionA[binOffset + c000]
                              + dCorr_dB * gridCorrectionB[binOffset + c000];
                float corr001 = dCorr_dN * gridCorrectionN[binOffset + c001]
                              + dCorr_dA * gridCorrectionA[binOffset + c001]
                              + dCorr_dB * gridCorrectionB[binOffset + c001];
                float corr010 = dCorr_dN * gridCorrectionN[binOffset + c010]
                              + dCorr_dA * gridCorrectionA[binOffset + c010]
                              + dCorr_dB * gridCorrectionB[binOffset + c010];
                float corr011 = dCorr_dN * gridCorrectionN[binOffset + c011]
                              + dCorr_dA * gridCorrectionA[binOffset + c011]
                              + dCorr_dB * gridCorrectionB[binOffset + c011];
                float corr100 = dCorr_dN * gridCorrectionN[binOffset + c100]
                              + dCorr_dA * gridCorrectionA[binOffset + c100]
                              + dCorr_dB * gridCorrectionB[binOffset + c100];
                float corr101 = dCorr_dN * gridCorrectionN[binOffset + c101]
                              + dCorr_dA * gridCorrectionA[binOffset + c101]
                              + dCorr_dB * gridCorrectionB[binOffset + c101];
                float corr110 = dCorr_dN * gridCorrectionN[binOffset + c110]
                              + dCorr_dA * gridCorrectionA[binOffset + c110]
                              + dCorr_dB * gridCorrectionB[binOffset + c110];
                float corr111 = dCorr_dN * gridCorrectionN[binOffset + c111]
                              + dCorr_dA * gridCorrectionA[binOffset + c111]
                              + dCorr_dB * gridCorrectionB[binOffset + c111];

                // Trilinear interpolation of combined correction
                float cmm = oz * corr000 + fz * corr001;
                float cmp = oz * corr010 + fz * corr011;
                float cpm = oz * corr100 + fz * corr101;
                float cpp = oz * corr110 + fz * corr111;
                float cm = oy * cmm + fy * cmp;
                float cp = oy * cpm + fy * cpp;
                float corrVal = ox * cm + fx * cp;

                // Correction gradient (unit cell coords)
                float corr_dfx = cp - cm;
                float corr_dfy = ox * (cmp - cmm) + fx * (cpp - cpm);
                float corr_dfz = ox * (oy * (corr001 - corr000) + fy * (corr011 - corr010))
                               + fx * (oy * (corr101 - corr100) + fy * (corr111 - corr110));

                // Correction mixed Hessian (trilinear pure d2 = 0)
                float corr_d2xy = oz * (corr000 - corr010 - corr100 + corr110)
                                + fz * (corr001 - corr011 - corr101 + corr111);
                float corr_d2xz = oy * (corr000 - corr001 - corr100 + corr101)
                                + fy * (corr010 - corr011 - corr110 + corr111);
                float corr_d2yz = ox * (corr000 - corr001 - corr010 + corr011)
                                + fx * (corr100 - corr101 - corr110 + corr111);

                // Combine HCT polynomial Hessian + trilinear correction Hessian
                result.hct = hct_val + corrVal;
                result.gradient.x = (hct_dx + corr_dfx) * invSpacing;
                result.gradient.y = (hct_dy + corr_dfy) * invSpacing;
                result.gradient.z = (hct_dz + corr_dfz) * invSpacing;
                result.hessian[0] = hct_d2xx * invSpacing2;  // pure d2 from HCT only (trilinear pure = 0)
                result.hessian[1] = hct_d2yy * invSpacing2;
                result.hessian[2] = hct_d2zz * invSpacing2;
                result.hessian[3] = (hct_d2xy + corr_d2xy) * invSpacing2;
                result.hessian[4] = (hct_d2xz + corr_d2xz) * invSpacing2;
                result.hessian[5] = (hct_d2yz + corr_d2yz) * invSpacing2;
            }

            return result;
        }
    }

    if (method == 0) {
        // =============================================================
        // TRILINEAR: Pure d2 = 0, only mixed d2 are nonzero.
        // =============================================================
        int corrOffset;
        if (hasBinnedKDEDerivatives) {
            int numPoints = gridCounts[0] * nyz;
            int binIdx = binOffset / numPoints;
            corrOffset = binIdx * 27 * numPoints;
        } else if (useKDECorrections) {
            corrOffset = 0;
        } else {
            corrOffset = binOffset;
        }

        // Combine all 4 grids at 8 corners
        float v000 = gridHctProbe[c000] + dCorr_dN * gridCorrectionN[corrOffset + c000]
                   + dCorr_dA * gridCorrectionA[corrOffset + c000] + dCorr_dB * gridCorrectionB[corrOffset + c000];
        float v001 = gridHctProbe[c001] + dCorr_dN * gridCorrectionN[corrOffset + c001]
                   + dCorr_dA * gridCorrectionA[corrOffset + c001] + dCorr_dB * gridCorrectionB[corrOffset + c001];
        float v010 = gridHctProbe[c010] + dCorr_dN * gridCorrectionN[corrOffset + c010]
                   + dCorr_dA * gridCorrectionA[corrOffset + c010] + dCorr_dB * gridCorrectionB[corrOffset + c010];
        float v011 = gridHctProbe[c011] + dCorr_dN * gridCorrectionN[corrOffset + c011]
                   + dCorr_dA * gridCorrectionA[corrOffset + c011] + dCorr_dB * gridCorrectionB[corrOffset + c011];
        float v100 = gridHctProbe[c100] + dCorr_dN * gridCorrectionN[corrOffset + c100]
                   + dCorr_dA * gridCorrectionA[corrOffset + c100] + dCorr_dB * gridCorrectionB[corrOffset + c100];
        float v101 = gridHctProbe[c101] + dCorr_dN * gridCorrectionN[corrOffset + c101]
                   + dCorr_dA * gridCorrectionA[corrOffset + c101] + dCorr_dB * gridCorrectionB[corrOffset + c101];
        float v110 = gridHctProbe[c110] + dCorr_dN * gridCorrectionN[corrOffset + c110]
                   + dCorr_dA * gridCorrectionA[corrOffset + c110] + dCorr_dB * gridCorrectionB[corrOffset + c110];
        float v111 = gridHctProbe[c111] + dCorr_dN * gridCorrectionN[corrOffset + c111]
                   + dCorr_dA * gridCorrectionA[corrOffset + c111] + dCorr_dB * gridCorrectionB[corrOffset + c111];

        // Trilinear interpolation
        float vmm = oz * v000 + fz * v001;
        float vmp = oz * v010 + fz * v011;
        float vpm = oz * v100 + fz * v101;
        float vpp = oz * v110 + fz * v111;
        float vm = oy * vmm + fy * vmp;
        float vp = oy * vpm + fy * vpp;
        result.hct = ox * vm + fx * vp;

        // Gradient
        result.gradient.x = (vp - vm) * invSpacing;
        result.gradient.y = (ox * (vmp - vmm) + fx * (vpp - vpm)) * invSpacing;
        result.gradient.z = (ox * (oy * (v001 - v000) + fy * (v011 - v010))
                           + fx * (oy * (v101 - v100) + fy * (v111 - v110))) * invSpacing;

        // Mixed Hessian (pure d2 = 0 for trilinear)
        result.hessian[0] = 0.0f;
        result.hessian[1] = 0.0f;
        result.hessian[2] = 0.0f;
        result.hessian[3] = (oz * (v000 - v010 - v100 + v110) + fz * (v001 - v011 - v101 + v111)) * invSpacing2;
        result.hessian[4] = (oy * (v000 - v001 - v100 + v101) + fy * (v010 - v011 - v110 + v111)) * invSpacing2;
        result.hessian[5] = (ox * (v000 - v001 - v010 + v011) + fx * (v100 - v101 - v110 + v111)) * invSpacing2;
    }

    return result;
}

/**
 * Compute HCT contribution from receptor via grid interpolation.
 * Applies correction for exact results at any ligand radius.
 * Supports multiple interpolation methods via the method parameter.
 *
 * When useKDECorrections=true, correction grids are KDE format [27*numPoints]
 * and use the same interpolation method as HCT. Otherwise uses binned format.
 */
extern "C" __global__ void computeReceptorHCT(
    const float4* __restrict__ posq,           // Positions (xyz) and charges (w)
    const int* __restrict__ particleIndices,   // Which particles to process
    const float* __restrict__ radii,           // Intrinsic radii (template)
    const int* __restrict__ gridCounts,        // [nx, ny, nz]
    const float* __restrict__ gridHctProbe,    // HCT values at probe radius
    const float* __restrict__ gridHctDerivatives, // HCT derivatives for tricubic/triquintic (or nullptr)
    const float* __restrict__ gridCorrectionN, // Correction N [nBins * nPoints] or [27 * nPoints] for KDE
    const float* __restrict__ gridCorrectionA, // Correction A
    const float* __restrict__ gridCorrectionB, // Correction B
    const float* __restrict__ rThresholds,     // Bin thresholds (ignored for KDE mode)
    const int* __restrict__ groupStart,        // Group start indices
    int numGroups,
    float originX, float originY, float originZ,
    float gridSpacing,
    float probeRadius,
    int numBins,
    int totalParticles,
    int templateNumAtoms,
    int interpolationMethod,                   // 0=trilinear, 1=bspline, 2=tricubic, 3=triquintic
    bool useKDECorrections,                    // True if correction grids have 27 derivatives
    bool hasBinnedKDEDerivatives,              // True if corrections are binned+KDE [numBins*27*nPoints]
    float* __restrict__ hctReceptor            // Output: HCT from receptor
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
    int templateIdx = atomInGroup % templateNumAtoms;  // For multi-ligand, wrap around

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

    // Use the interpolation helper function
    GBSAInterpolationResult result = interpolateGBSAGrids(
        position, R_i_off, R_probe_off,
        gridCounts, gridSpacing,
        originX, originY, originZ,
        gridHctProbe, gridHctDerivatives,
        gridCorrectionN, gridCorrectionA, gridCorrectionB,
        binOffset, interpolationMethod, false,  // computeGradient=false
        useKDECorrections, hasBinnedKDEDerivatives
    );

    hctReceptor[idx] = result.isInside ? result.hct : 0.0f;
}

/**
 * Compute HCT contribution from ligand-ligand pairwise interactions.
 * Uses standard HCT integral formula.
 */
extern "C" __global__ void computeLigandHCT(
    const float4* __restrict__ posq,
    const int* __restrict__ particleIndices,
    const float* __restrict__ radii,
    const float* __restrict__ scaleFactors,
    const int* __restrict__ exclusionAtoms,
    const int* __restrict__ exclusionStart,
    const int* __restrict__ groupStart,
    int numGroups,
    int templateNumAtoms,
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

    // Get exclusion range for this atom
    int exclStart = exclusionStart[templateIdx_i];
    int exclEnd = exclusionStart[templateIdx_i + 1];

    float hct = 0.0f;

    // Loop over other atoms in same group
    for (int jLocal = 0; jLocal < (groupEndIdx - groupStartIdx); jLocal++) {
        if (jLocal == atomInGroup) continue;

        int j = groupStartIdx + jLocal;
        int templateIdx_j = jLocal % templateNumAtoms;

        // Check exclusions
        bool excluded = false;
        for (int e = exclStart; e < exclEnd; e++) {
            if (exclusionAtoms[e] == templateIdx_j) {
                excluded = true;
                break;
            }
        }
        if (excluded) continue;

        int particleIdx_j = particleIndices[j];
        float4 pos_j = posq[particleIdx_j];

        float dx = pos_i.x - pos_j.x;
        float dy = pos_i.y - pos_j.y;
        float dz = pos_i.z - pos_j.z;
        float r = sqrtf(dx*dx + dy*dy + dz*dz);

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
 * Compute Born radii from total HCT using OBC-II formula.
 */
extern "C" __global__ void computeBornRadii(
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

    // Clamp to reasonable range (only upper bound)
    bornRadius = fminf(bornRadius, 50.0f);  // Max 50 nm

    bornRadii[idx] = bornRadius;
}

/**
 * Compute GB energy using Still equation and accumulate forces.
 * E_gb = prefactor * sum_ij (q_i * q_j / f_gb)
 * where f_gb = sqrt(r_ij^2 + R_i*R_j*exp(-r_ij^2/(4*R_i*R_j)))
 */
extern "C" __global__ void computeGBEnergy(
    const float4* __restrict__ posq,
    const int* __restrict__ particleIndices,
    const float* __restrict__ charges,
    const float* __restrict__ bornRadii,
    const int* __restrict__ exclusionAtoms,
    const int* __restrict__ exclusionStart,
    const int* __restrict__ groupStart,
    int numGroups,
    int templateNumAtoms,
    float prefactor,
    unsigned long long* __restrict__ forceBuffer,
    float* __restrict__ groupEnergies,
    int paddedNumAtoms,
    float globalScalingFactor,
    const float* __restrict__ groupScalingFactors
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

    int exclStart = exclusionStart[templateIdx_i];
    int exclEnd = exclusionStart[templateIdx_i + 1];

    float energy = 0.0f;
    float3 force = make_float3(0.0f, 0.0f, 0.0f);

    // Self energy term (0.5 factor for GB formula)
    energy += 0.5f * prefactor * q_i * q_i / R_i;

    // Pairwise terms (only count j > i to avoid double counting)
    int groupSize = groupEndIdx - groupStartIdx;
    for (int jLocal = atomInGroup + 1; jLocal < groupSize; jLocal++) {
        int j = groupStartIdx + jLocal;
        int templateIdx_j = jLocal % templateNumAtoms;

        // Check exclusions
        bool excluded = false;
        for (int e = exclStart; e < exclEnd; e++) {
            if (exclusionAtoms[e] == templateIdx_j) {
                excluded = true;
                break;
            }
        }
        if (excluded) continue;

        int particleIdx_j = particleIndices[j];
        float4 pos_j = posq[particleIdx_j];
        float q_j = charges[templateIdx_j];
        float R_j = bornRadii[j];

        float dx = pos_j.x - pos_i.x;
        float dy = pos_j.y - pos_i.y;
        float dz = pos_j.z - pos_i.z;
        float r2 = dx*dx + dy*dy + dz*dz;
        float r = sqrtf(r2);

        // Still equation: f_gb = sqrt(r^2 + R_i*R_j*exp(-r^2/(4*R_i*R_j)))
        float RiRj = R_i * R_j;
        float expArg = -r2 / (4.0f * RiRj);
        float expTerm = expf(expArg);
        float f_gb2 = r2 + RiRj * expTerm;
        float f_gb = sqrtf(f_gb2);
        float invFgb = 1.0f / f_gb;

        // Energy (each pair counted once with j > i loop)
        float pairEnergy = prefactor * q_i * q_j * invFgb;
        energy += pairEnergy;

        // Force = -dE/dr
        // dE/dr = prefactor * q_i * q_j * (-1/f_gb^2) * df_gb/dr
        // df_gb/dr = (1/f_gb) * (r - 0.25*RiRj*exp(-r^2/(4*RiRj))*(2r/(4*RiRj)))
        //          = (1/f_gb) * (r + 0.5*r*exp(-r^2/(4*RiRj))/RiRj * (-r^2/(4*RiRj) derivative? no...
        // Let me redo: f_gb^2 = r^2 + RiRj*exp(-r^2/(4*RiRj))
        // d(f_gb^2)/dr = 2r + RiRj * exp(...) * (-2r/(4*RiRj)) = 2r - 0.5*r*exp(...) = 2r*(1 - 0.25*exp(...))
        // df_gb/dr = (1/(2*f_gb)) * 2r * (1 - 0.25*exp(-r^2/(4*RiRj)))
        //          = r/f_gb * (1 - 0.25*expTerm)

        float dFgbDr = (r * invFgb) * (1.0f - 0.25f * expTerm);
        float dEdR = -prefactor * q_i * q_j * invFgb * invFgb * dFgbDr;

        // Force on i from j: F_i = -dE/dx_i = dEdR * dx/r (dx = pos_j - pos_i)
        float invR = 1.0f / r;
        force.x += dEdR * dx * invR;
        force.y += dEdR * dy * invR;
        force.z += dEdR * dz * invR;

        // Accumulate force on j (Newton's 3rd law): F_j = -F_i
        atomicAdd(&forceBuffer[particleIdx_j], static_cast<unsigned long long>((long long)(-dEdR * dx * invR * 0x100000000)));
        atomicAdd(&forceBuffer[particleIdx_j + paddedNumAtoms], static_cast<unsigned long long>((long long)(-dEdR * dy * invR * 0x100000000)));
        atomicAdd(&forceBuffer[particleIdx_j + 2*paddedNumAtoms], static_cast<unsigned long long>((long long)(-dEdR * dz * invR * 0x100000000)));
    }

    // Accumulate force on i
    atomicAdd(&forceBuffer[particleIdx_i], static_cast<unsigned long long>((long long)(force.x * 0x100000000)));
    atomicAdd(&forceBuffer[particleIdx_i + paddedNumAtoms], static_cast<unsigned long long>((long long)(force.y * 0x100000000)));
    atomicAdd(&forceBuffer[particleIdx_i + 2*paddedNumAtoms], static_cast<unsigned long long>((long long)(force.z * 0x100000000)));

    // Accumulate group energy
    atomicAdd(&groupEnergies[groupIdx], energy);
}

/**
 * Compute surface area energy (ACE approximation).
 * E_sa = surfaceTension * sum_i (4*pi*(R_i + probe)^2 * (R_i/bornRadius)^6)
 */
extern "C" __global__ void computeSAEnergy(
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

    int templateIdx = atomInGroup % templateNumAtoms;
    float R_i = radii[templateIdx];
    float B_i = bornRadii[idx];

    // ACE formula
    float Rprobe = R_i + probeRadius;
    float ratio = R_i / B_i;
    float ratio6 = ratio * ratio * ratio * ratio * ratio * ratio;
    float area = 4.0f * 3.14159265359f * Rprobe * Rprobe;
    float saEnergy = surfaceTension * area * ratio6;

    atomicAdd(&groupEnergies[groupIdx], saEnergy);
}

/**
 * Accumulate surface area derivatives into dE_dR.
 * dE_sa/dR_born = -6 * surfaceTension * 4*pi*(R_i + probe)^2 * R_i^6 / R_born^7
 *               = -6 * E_sa / R_born
 * Must be called after accumulateBornRadiiDerivatives to add to existing dE_dR.
 */
extern "C" __global__ void accumulateSADerivatives(
    const float* __restrict__ radii,
    const float* __restrict__ bornRadii,
    const int* __restrict__ groupStart,
    int numGroups,
    int templateNumAtoms,
    float surfaceTension,
    float probeRadius,
    float* __restrict__ dE_dR  // In/Out: accumulate dE_sa/dR_born
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Determine group
    int atomInGroup = idx;
    int groupEndIdx = 0;

    for (int g = 0; g < numGroups; g++) {
        int groupStartIdx = groupStart[g];
        groupEndIdx = groupStart[g + 1];
        if (idx >= groupStartIdx && idx < groupEndIdx) {
            atomInGroup = idx - groupStartIdx;
            break;
        }
    }

    if (idx >= groupEndIdx) return;

    int templateIdx = atomInGroup % templateNumAtoms;
    float R_i = radii[templateIdx];
    float R_born = bornRadii[idx];

    // ACE formula: E_sa = surfaceTension * 4*pi*(R_i + probe)^2 * (R_i/R_born)^6
    // Derivative: dE_sa/dR_born = -6 * E_sa / R_born
    float Rprobe = R_i + probeRadius;
    float ratio = R_i / R_born;
    float ratio6 = ratio * ratio * ratio * ratio * ratio * ratio;
    float area = 4.0f * 3.14159265359f * Rprobe * Rprobe;
    float E_sa = surfaceTension * area * ratio6;

    // Add SA derivative to dE_dR
    dE_dR[idx] += -6.0f * E_sa / R_born;
}

/**
 * Compute forces from receptor HCT grid interpolation gradient.
 * Supports multiple interpolation methods via the interpolationMethod parameter.
 *
 * When useKDECorrections=true, correction grids are KDE format [27*numPoints]
 * and use the same interpolation method as HCT for smooth force continuity.
 */
extern "C" __global__ void computeReceptorHCTGradientForce(
    const float4* __restrict__ posq,
    const int* __restrict__ particleIndices,
    const float* __restrict__ radii,
    const float* __restrict__ bornRadii,
    const float* __restrict__ hctReceptor,
    const float* __restrict__ hctLigand,
    const float* __restrict__ dE_dR,            // dE/dR_born from accumulation
    const int* __restrict__ gridCounts,
    const float* __restrict__ gridHctProbe,
    const float* __restrict__ gridHctDerivatives, // HCT derivatives for tricubic/triquintic (or nullptr)
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
    int interpolationMethod,                    // 0=trilinear, 1=bspline, 2=tricubic, 3=triquintic
    bool useKDECorrections,                     // True if correction grids have 27 derivatives
    bool hasBinnedKDEDerivatives,               // True if corrections are binned+KDE [numBins*27*nPoints]
    unsigned long long* __restrict__ forceBuffer,
    int paddedNumAtoms
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= totalParticles) return;

    // Determine group and atom in group
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
    float R_born_i = bornRadii[idx];

    // Compute dE_dHCT from dE_dR using OBC-II chain rule
    float hctTotal_i = hctReceptor[idx] + hctLigand[idx];
    float psi_i = 0.5f * R_i_off * hctTotal_i;

    float psi2 = psi_i * psi_i;
    float psi3 = psi2 * psi_i;
    float tanh_arg = OBC_ALPHA * psi_i - OBC_BETA * psi2 + OBC_GAMMA * psi3;
    float t = tanhf(tanh_arg);
    float sech2 = 1.0f - t * t;
    float darg_dpsi = OBC_ALPHA - 2.0f * OBC_BETA * psi_i + 3.0f * OBC_GAMMA * psi2;

    // dR_born/dpsi = R_born^2 * sech^2 * darg/dpsi / R_i
    float dR_dpsi_i = R_born_i * R_born_i * sech2 * darg_dpsi / R_i;

    // dE/dHCT = dE/dR * dR/dpsi * dpsi/dHCT = dE/dR * dR/dpsi * 0.5 * R_off
    float dEdHCT = dE_dR[idx] * dR_dpsi_i * 0.5f * R_i_off;

    int nx = gridCounts[0];
    int ny = gridCounts[1];
    int nz = gridCounts[2];
    int numPoints = nx * ny * nz;

    // Find bin for this atom
    int binIdx = numBins - 1;
    for (int b = 0; b < numBins; b++) {
        if (rThresholds[b] >= R_i_off) {
            binIdx = b;
            break;
        }
    }
    int binOffset = binIdx * numPoints;

    // Use the interpolation helper function with gradient computation
    GBSAInterpolationResult result = interpolateGBSAGrids(
        position, R_i_off, R_probe_off,
        gridCounts, gridSpacing,
        originX, originY, originZ,
        gridHctProbe, gridHctDerivatives,
        gridCorrectionN, gridCorrectionA, gridCorrectionB,
        binOffset, interpolationMethod, true,  // computeGradient=true
        useKDECorrections, hasBinnedKDEDerivatives
    );

    // Force = -dE/dHCT * gradient (gradient is zero if outside grid)
    float3 force;
    force.x = -dEdHCT * result.gradient.x;
    force.y = -dEdHCT * result.gradient.y;
    force.z = -dEdHCT * result.gradient.z;

    // Accumulate force using fixed-point arithmetic
    atomicAdd(&forceBuffer[particleIdx], static_cast<unsigned long long>((long long)(force.x * 0x100000000)));
    atomicAdd(&forceBuffer[particleIdx + paddedNumAtoms], static_cast<unsigned long long>((long long)(force.y * 0x100000000)));
    atomicAdd(&forceBuffer[particleIdx + 2*paddedNumAtoms], static_cast<unsigned long long>((long long)(force.z * 0x100000000)));
}

/**
 * Accumulate dE/dR_born for each atom.
 * This must be called after computeGBEnergy to collect derivatives.
 */
extern "C" __global__ void accumulateBornRadiiDerivatives(
    const float4* __restrict__ posq,
    const int* __restrict__ particleIndices,
    const float* __restrict__ charges,
    const float* __restrict__ bornRadii,
    const int* __restrict__ exclusionAtoms,
    const int* __restrict__ exclusionStart,
    const int* __restrict__ groupStart,
    int numGroups,
    int templateNumAtoms,
    float prefactor,
    float* __restrict__ dE_dR  // Output: dE/dR_born for each atom
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

    int exclStart = exclusionStart[templateIdx_i];
    int exclEnd = exclusionStart[templateIdx_i + 1];

    // Self energy derivative: dE_self/dR_i = -0.5 * prefactor * q_i^2 / R_i^2
    float dE_dRi = -0.5f * prefactor * q_i * q_i / (R_i * R_i);

    // Pairwise contributions
    int groupSize = groupEndIdx - groupStartIdx;
    for (int jLocal = 0; jLocal < groupSize; jLocal++) {
        if (jLocal == atomInGroup) continue;

        int j = groupStartIdx + jLocal;
        int templateIdx_j = jLocal % templateNumAtoms;

        // Check exclusions
        bool excluded = false;
        for (int e = exclStart; e < exclEnd; e++) {
            if (exclusionAtoms[e] == templateIdx_j) {
                excluded = true;
                break;
            }
        }
        if (excluded) continue;

        int particleIdx_j = particleIndices[j];
        float4 pos_j = posq[particleIdx_j];
        float q_j = charges[templateIdx_j];
        float R_j = bornRadii[j];

        float dx = pos_j.x - pos_i.x;
        float dy = pos_j.y - pos_i.y;
        float dz = pos_j.z - pos_i.z;
        float r2 = dx*dx + dy*dy + dz*dz;

        // f_gb computation
        float RiRj = R_i * R_j;
        float expArg = -r2 / (4.0f * RiRj);
        float expTerm = expf(expArg);
        float fgb2 = r2 + RiRj * expTerm;
        float fgb = sqrtf(fgb2);

        // d(fgb^2)/dR_i = R_j * expTerm * (1 + r2/(4*R_i^2*R_j))
        //               = R_j * expTerm + 0.25 * expTerm * r2 / R_i
        float dfgb2_dRi = R_j * expTerm + 0.25f * expTerm * r2 / R_i;
        float dfgb_dRi = dfgb2_dRi / (2.0f * fgb);

        // dE_pair/dR_i = -prefactor * q_i * q_j * dfgb_dRi / fgb^2
        float dE_pair_dRi = -prefactor * q_i * q_j * dfgb_dRi / fgb2;

        dE_dRi += dE_pair_dRi;
    }

    dE_dR[idx] = dE_dRi;
}


/**
 * Compute forces from HCT chain rule through Born radii.
 * Uses OpenMM's bornForces + t3 formula with Newton's 3rd law,
 * matching the validated IsolatedGBSA approach.
 * Must be called after accumulateBornRadiiDerivatives.
 */
extern "C" __global__ void computeHCTChainRuleForces(
    const float4* __restrict__ posq,
    const int* __restrict__ particleIndices,
    const float* __restrict__ radii,
    const float* __restrict__ scaleFactors,
    const float* __restrict__ bornRadii,
    const float* __restrict__ hctReceptor,
    const float* __restrict__ hctLigand,
    const float* __restrict__ dE_dR,
    const int* __restrict__ exclusionAtoms,
    const int* __restrict__ exclusionStart,
    const int* __restrict__ groupStart,
    int numGroups,
    int templateNumAtoms,
    unsigned long long* __restrict__ forceBuffer,
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
    float R_i = radii[templateIdx_i];
    float R_i_off = R_i - DIELECTRIC_OFFSET;
    float S_i = R_i_off * scaleFactors[templateIdx_i];
    float R_born_i = bornRadii[idx];

    // Compute bornForces[i] = dE/dR_born * R_born^2 * obcChain (OpenMM convention)
    float hctTotal_i = hctReceptor[idx] + hctLigand[idx];
    float psi_i = 0.5f * R_i_off * hctTotal_i;
    float psi2_i = psi_i * psi_i;

    float tanhArg_i = OBC_ALPHA * psi_i - OBC_BETA * psi2_i + OBC_GAMMA * psi2_i * psi_i;
    float tanhVal_i = tanhf(tanhArg_i);
    float sech2_i = 1.0f - tanhVal_i * tanhVal_i;
    float dTanhArgDPsi_i = OBC_ALPHA - 2.0f * OBC_BETA * psi_i + 3.0f * OBC_GAMMA * psi2_i;

    // obcChain = R_off * (α - 2β*ψ + 3γ*ψ²) * sech²(arg) / R
    float obcChain_i = R_i_off * dTanhArgDPsi_i * sech2_i / R_i;

    // bornForces = dE/dR_born * R_born² * obcChain
    float bornForces_i = dE_dR[idx] * R_born_i * R_born_i * obcChain_i;

    int exclStart = exclusionStart[templateIdx_i];
    int exclEnd = exclusionStart[templateIdx_i + 1];

    float3 force_i = make_float3(0.0f, 0.0f, 0.0f);

    // Loop over other atoms in group
    int groupSize = groupEndIdx - groupStartIdx;
    for (int jLocal = 0; jLocal < groupSize; jLocal++) {
        if (jLocal == atomInGroup) continue;

        int j = groupStartIdx + jLocal;
        int templateIdx_j = jLocal % templateNumAtoms;

        // Check exclusions
        bool excluded = false;
        for (int e = exclStart; e < exclEnd; e++) {
            if (exclusionAtoms[e] == templateIdx_j) {
                excluded = true;
                break;
            }
        }
        if (excluded) continue;

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

            // OpenMM's t3 formula
            float t3 = 0.125f * (1.0f + S_j2 * r2_inv) * (l_ij2 - u_ij2)
                     + 0.25f * logf(u_ij / l_ij) * r2_inv;

            // de = bornForces[i] * t3 / r
            float de = bornForces_i * t3 * r_inv;

            // force_i += de * dx (OpenMM convention: dx = pos_i - pos_j)
            force_i.x += de * dx;
            force_i.y += de * dy;
            force_i.z += de * dz;

            // Force on j (Newton's 3rd law): force_j = -force_i contribution
            atomicAdd(&forceBuffer[particleIdx_j], static_cast<unsigned long long>((long long)(-de * dx * 0x100000000)));
            atomicAdd(&forceBuffer[particleIdx_j + paddedNumAtoms], static_cast<unsigned long long>((long long)(-de * dy * 0x100000000)));
            atomicAdd(&forceBuffer[particleIdx_j + 2*paddedNumAtoms], static_cast<unsigned long long>((long long)(-de * dz * 0x100000000)));
        }

        // NOTE: "i screens j" is handled when thread j processes this pair.
    }

    // Accumulate force on atom i
    atomicAdd(&forceBuffer[particleIdx_i], static_cast<unsigned long long>((long long)(force_i.x * 0x100000000)));
    atomicAdd(&forceBuffer[particleIdx_i + paddedNumAtoms], static_cast<unsigned long long>((long long)(force_i.y * 0x100000000)));
    atomicAdd(&forceBuffer[particleIdx_i + 2*paddedNumAtoms], static_cast<unsigned long long>((long long)(force_i.z * 0x100000000)));
}

// ==========================================================================
// Analytical Hessian kernels
// ==========================================================================

/**
 * Kernel 1: Prepare per-atom OBC-II intermediates for analytical Hessian.
 *
 * Computes dR_born/dΨ and d²R_born/dΨ² for each atom, where Ψ is the
 * total HCT integral (receptor + ligand). These are needed by the Born
 * coupling matrix kernel.
 *
 * Formula (validated against JAX autodiff in derive_gbsa_hessian.py):
 *   ψ_s = 0.5 * R_off * Ψ
 *   arg = α*ψ_s - β*ψ_s² + γ*ψ_s³
 *   t = tanh(arg), sech² = 1 - t²
 *   R_born = 1/(1/R_off - t/R_i)
 *   dR/dΨ = R_born² * sech² * darg/dψ_s * (R_off/2) / R_i
 *   d²R/dΨ² = product rule on A*B*C*D where A=R², B=sech², C=darg/dψ_s, D=R_off/(2R_i)
 */
extern "C" __global__ void prepareHessianIntermediates(
    const float* __restrict__ radii,
    const float* __restrict__ bornRadii,
    const float* __restrict__ hctReceptor,
    const float* __restrict__ hctLigand,
    const float* __restrict__ dE_dR_in,
    int numAtoms,
    int templateNumAtoms,
    float* __restrict__ dR_dPsi_out,
    float* __restrict__ d2R_dPsi2_out,
    float* __restrict__ dE_dHCT_out
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numAtoms) return;

    int templateIdx = idx % templateNumAtoms;
    float R_i = radii[templateIdx];
    float R_i_off = R_i - DIELECTRIC_OFFSET;
    float R_born = bornRadii[idx];

    float hctTotal = hctReceptor[idx] + hctLigand[idx];
    float psi_s = 0.5f * R_i_off * hctTotal;

    float psi_s2 = psi_s * psi_s;
    float tanh_arg = OBC_ALPHA * psi_s - OBC_BETA * psi_s2 + OBC_GAMMA * psi_s2 * psi_s;
    float t = tanhf(tanh_arg);
    float sech2 = 1.0f - t * t;

    float darg_dpsi_s = OBC_ALPHA - 2.0f * OBC_BETA * psi_s + 3.0f * OBC_GAMMA * psi_s2;
    float d2arg_dpsi_s2 = -2.0f * OBC_BETA + 6.0f * OBC_GAMMA * psi_s;

    // dpsi_s/dΨ = 0.5 * R_off
    float dpsi_s_dPsi = 0.5f * R_i_off;

    // D = dpsi_s_dPsi / R_i  (constant factor)
    float D = dpsi_s_dPsi / R_i;

    // dR/dΨ = R_born² * sech² * darg/dψ_s * D
    float dR_dPsi = R_born * R_born * sech2 * darg_dpsi_s * D;

    // d²R/dΨ² via product rule: d(A*B*C)/dΨ * D
    // where A = R_born², B = sech², C = darg/dψ_s
    float A = R_born * R_born;
    float B = sech2;
    float C = darg_dpsi_s;

    // dA/dΨ = 2*R_born*dR/dΨ
    float dA_dPsi = 2.0f * R_born * dR_dPsi;

    // dB/dΨ = -2*sech²*tanh(arg) * darg/dΨ
    // darg/dΨ = darg/dψ_s * dψ_s/dΨ
    float darg_dPsi = darg_dpsi_s * dpsi_s_dPsi;
    float dB_dPsi = -2.0f * sech2 * t * darg_dPsi;

    // dC/dΨ = d²arg/dψ_s² * dψ_s/dΨ
    float dC_dPsi = d2arg_dpsi_s2 * dpsi_s_dPsi;

    float d2R_dPsi2 = (dA_dPsi * B * C + A * dB_dPsi * C + A * B * dC_dPsi) * D;

    dR_dPsi_out[idx] = dR_dPsi;
    d2R_dPsi2_out[idx] = d2R_dPsi2;

    // dE/dHCT = dE/dR * dR/dΨ (derivative of energy w.r.t. raw HCT integral)
    dE_dHCT_out[idx] = dE_dR_in[idx] * dR_dPsi;
}

/**
 * Kernel 2: Compute HCT Jacobian matrix J[N x 3N].
 *
 * J[k, 3i+α] = dΨ_k / dx_i^α  (how atom k's total HCT changes when atom i moves)
 *
 * Two contributions:
 * - Receptor grid: J[k, 3k+α] += grid gradient (only self-dependence)
 * - Pairwise ligand: J[k, 3j+α] = dI_kj/dr * (x_j - x_k)^α / r_kj
 *                     J[k, 3k+α] += Σ_j dI_kj/dr * (x_k - x_j)^α / r_kj
 *
 * One thread per atom k fills its entire row J[k, :].
 */
extern "C" __global__ void computeHCTJacobian(
    const float4* __restrict__ posq,
    const int* __restrict__ particleIndices,
    const float* __restrict__ radii,
    const float* __restrict__ scaleFactors,
    const int* __restrict__ exclusionAtoms,
    const int* __restrict__ exclusionStart,
    const int* __restrict__ groupStart,
    int numGroups,
    int templateNumAtoms,
    const int* __restrict__ gridCounts,
    const float* __restrict__ gridHctProbe,
    const float* __restrict__ gridHctDerivatives,
    const float* __restrict__ gridCorrectionN,
    const float* __restrict__ gridCorrectionA,
    const float* __restrict__ gridCorrectionB,
    const float* __restrict__ rThresholdsBuf,
    float originX, float originY, float originZ,
    float gridSpacing,
    float probeRadiusVal,
    int numBins,
    int interpolationMethod,
    bool useKDECorrections,
    bool hasBinnedKDEDerivatives,
    int totalParticles,
    float* __restrict__ jacobian
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int atomInGroup = idx;
    int groupStartIdx = 0;
    int groupEndIdx = 0;

    for (int g = 0; g < numGroups; g++) {
        groupStartIdx = groupStart[g];
        groupEndIdx = groupStart[g + 1];
        if (idx >= groupStartIdx && idx < groupEndIdx) {
            atomInGroup = idx - groupStartIdx;
            break;
        }
    }
    if (idx >= groupEndIdx) return;

    int dim3N = 3 * totalParticles;

    // Zero this row
    for (int c = 0; c < dim3N; c++) {
        jacobian[idx * dim3N + c] = 0.0f;
    }

    int particleIdx_k = particleIndices[idx];
    int templateIdx_k = atomInGroup % templateNumAtoms;
    float4 pos_k = posq[particleIdx_k];
    float R_k = radii[templateIdx_k];
    float R_k_off = R_k - DIELECTRIC_OFFSET;
    float R_probe_off = probeRadiusVal - DIELECTRIC_OFFSET;

    // --- Receptor grid gradient ---
    int numPoints = gridCounts[0] * gridCounts[1] * gridCounts[2];
    int binIdx = numBins - 1;
    for (int b = 0; b < numBins; b++) {
        if (rThresholdsBuf[b] >= R_k_off) { binIdx = b; break; }
    }
    int binOffset = binIdx * numPoints;

    float3 position = make_float3(pos_k.x, pos_k.y, pos_k.z);
    GBSAInterpolationResult gridResult = interpolateGBSAGrids(
        position, R_k_off, R_probe_off,
        gridCounts, gridSpacing, originX, originY, originZ,
        gridHctProbe, gridHctDerivatives,
        gridCorrectionN, gridCorrectionA, gridCorrectionB,
        binOffset, interpolationMethod, true,
        useKDECorrections, hasBinnedKDEDerivatives);

    if (gridResult.isInside) {
        jacobian[idx * dim3N + 3 * idx + 0] += gridResult.gradient.x;
        jacobian[idx * dim3N + 3 * idx + 1] += gridResult.gradient.y;
        jacobian[idx * dim3N + 3 * idx + 2] += gridResult.gradient.z;
    }

    // --- Pairwise ligand HCT derivatives ---
    int exclStart = exclusionStart[templateIdx_k];
    int exclEnd = exclusionStart[templateIdx_k + 1];
    int groupSize = groupEndIdx - groupStartIdx;

    for (int jLocal = 0; jLocal < groupSize; jLocal++) {
        if (jLocal == atomInGroup) continue;

        int j = groupStartIdx + jLocal;
        int templateIdx_j = jLocal % templateNumAtoms;

        bool excluded = false;
        for (int e = exclStart; e < exclEnd; e++) {
            if (exclusionAtoms[e] == templateIdx_j) { excluded = true; break; }
        }
        if (excluded) continue;

        int particleIdx_j = particleIndices[j];
        float4 pos_j = posq[particleIdx_j];
        float R_j = radii[templateIdx_j];
        float R_j_off = R_j - DIELECTRIC_OFFSET;
        float S_j = R_j_off * scaleFactors[templateIdx_j];

        float dx = pos_k.x - pos_j.x;
        float dy = pos_k.y - pos_j.y;
        float dz = pos_k.z - pos_j.z;
        float r2 = dx*dx + dy*dy + dz*dz;
        float r = sqrtf(r2);
        if (r < 1e-6f) continue;

        float r_plus_Sj = r + S_j;
        if (R_k_off >= r_plus_Sj) continue;

        float r_minus_Sj = fabsf(r - S_j);

        // Use the validated t3 formula (same as force kernel) for dI/dr.
        // dI/dr = -2*t3, where t3 = 0.125*(1+S²/r²)*(l²-u²) + 0.25*ln(u/l)/r²
        // This is simpler and avoids case-by-case derivative bugs.
        float l_val = (R_k_off > r_minus_Sj) ? (1.0f / R_k_off) : (1.0f / r_minus_Sj);
        float u_val = 1.0f / r_plus_Sj;
        float l2 = l_val * l_val;
        float u2 = u_val * u_val;
        float S2 = S_j * S_j;
        float invr2 = 1.0f / (r * r);
        float t3 = 0.125f * (1.0f + S2 * invr2) * (l2 - u2)
                  + 0.25f * logf(u_val / l_val) * invr2;
        float dHCT_dr = -2.0f * t3;

        float invr = 1.0f / r;

        // J[k, 3k+α] += dI_kj/dr * (x_k - x_j)^α / r
        jacobian[idx * dim3N + 3 * idx + 0] += dHCT_dr * dx * invr;
        jacobian[idx * dim3N + 3 * idx + 1] += dHCT_dr * dy * invr;
        jacobian[idx * dim3N + 3 * idx + 2] += dHCT_dr * dz * invr;

        // J[k, 3j+α] += dI_kj/dr * (x_j - x_k)^α / r
        jacobian[idx * dim3N + 3 * j + 0] += -dHCT_dr * dx * invr;
        jacobian[idx * dim3N + 3 * j + 1] += -dHCT_dr * dy * invr;
        jacobian[idx * dim3N + 3 * j + 2] += -dHCT_dr * dz * invr;
    }
}

/**
 * Kernel 2b: Compute receptor grid HCT Hessian for each atom.
 *
 * Uses central FD of the grid interpolation gradient to get d²Ψ_grid/(dx^α dx^β).
 * Output: 6 unique elements per atom [xx, yy, zz, xy, xz, yz].
 * These are added to H_hct2 diagonal blocks in the assembly kernel.
 *
 * One thread per atom k.
 */
extern "C" __global__ void computeReceptorGridHessian(
    const float4* __restrict__ posq,
    const int* __restrict__ particleIndices,
    const float* __restrict__ radii,
    const int* __restrict__ groupStart,
    int numGroups,
    int templateNumAtoms,
    const int* __restrict__ gridCounts,
    const float* __restrict__ gridHctProbe,
    const float* __restrict__ gridHctDerivatives,
    const float* __restrict__ gridCorrectionN,
    const float* __restrict__ gridCorrectionA,
    const float* __restrict__ gridCorrectionB,
    const float* __restrict__ rThresholdsBuf,
    float originX, float originY, float originZ,
    float gridSpacing,
    float probeRadiusVal,
    int numBins,
    int interpolationMethod,
    bool useKDECorrections,
    bool hasBinnedKDEDerivatives,
    int totalParticles,
    float* __restrict__ gridHessianOut
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int atomInGroup = idx;
    int groupEndIdx = 0;
    for (int g = 0; g < numGroups; g++) {
        int gs = groupStart[g];
        groupEndIdx = groupStart[g + 1];
        if (idx >= gs && idx < groupEndIdx) {
            atomInGroup = idx - gs;
            break;
        }
    }
    if (idx >= groupEndIdx) return;

    int templateIdx = atomInGroup % templateNumAtoms;
    int particleIdx = particleIndices[idx];
    float4 pk = posq[particleIdx];
    float R_k = radii[templateIdx];
    float R_k_off = R_k - DIELECTRIC_OFFSET;
    float R_probe_off = probeRadiusVal - DIELECTRIC_OFFSET;

    int numPoints = gridCounts[0] * gridCounts[1] * gridCounts[2];
    int binIdx = numBins - 1;
    for (int b = 0; b < numBins; b++) {
        if (rThresholdsBuf[b] >= R_k_off) { binIdx = b; break; }
    }
    int binOffset = binIdx * numPoints;

    // Analytical Hessian: single interpolation with second derivatives
    GBSAHessianResult hres = interpolateGBSAGridsWithHessian(
        make_float3(pk.x, pk.y, pk.z), R_k_off, R_probe_off,
        gridCounts, gridSpacing, originX, originY, originZ,
        gridHctProbe, gridHctDerivatives,
        gridCorrectionN, gridCorrectionA, gridCorrectionB,
        binOffset, interpolationMethod,
        useKDECorrections, hasBinnedKDEDerivatives);

    for (int i = 0; i < 6; i++)
        gridHessianOut[idx * 6 + i] = hres.isInside ? hres.hessian[i] : 0.0f;
}

/**
 * Kernel 3: Compute Born coupling matrix M[N x N].
 *
 * M[k,l] captures how energy couples through Born radii:
 *   M[k,l] = d²E/(dR_k dR_l) * dR_k/dΨ_k * dR_l/dΨ_l   (for k != l)
 *   M[k,k] = d²E/dR_k² * (dR_k/dΨ_k)² + dE/dR_k * d²R_k/dΨ_k²
 *
 * Still equation second derivatives (validated in derive_gbsa_hessian.py):
 *   α = exp(-r²/(4 R_k R_l))
 *   f² = r² + R_k R_l α
 *   d²E/dR_k² and d²E/(dR_k dR_l) from chain rule on C/f_gb
 *
 * One thread per atom k computes row M[k, :].
 */
extern "C" __global__ void computeBornCouplingMatrix(
    const float4* __restrict__ posq,
    const int* __restrict__ particleIndices,
    const float* __restrict__ charges,
    const float* __restrict__ intrinsicRadii,
    const float* __restrict__ bornRadii,
    const float* __restrict__ dR_dPsi,
    const float* __restrict__ d2R_dPsi2,
    const float* __restrict__ dE_dR,
    const int* __restrict__ exclusionAtoms,
    const int* __restrict__ exclusionStart,
    const int* __restrict__ groupStart,
    int numGroups,
    int templateNumAtoms,
    float prefactor,
    bool includeSA,
    float surfaceTension,
    float probeRadiusVal,
    int totalParticles,
    float* __restrict__ couplingMatrix
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int atomInGroup = idx;
    int groupStartIdx = 0;
    int groupEndIdx = 0;
    for (int g = 0; g < numGroups; g++) {
        groupStartIdx = groupStart[g];
        groupEndIdx = groupStart[g + 1];
        if (idx >= groupStartIdx && idx < groupEndIdx) {
            atomInGroup = idx - groupStartIdx;
            break;
        }
    }
    if (idx >= groupEndIdx) return;

    int templateIdx_k = atomInGroup % templateNumAtoms;
    int particleIdx_k = particleIndices[idx];
    float4 pos_k = posq[particleIdx_k];
    float q_k = charges[templateIdx_k];
    float R_k = bornRadii[idx];
    float R_k_intr = intrinsicRadii[templateIdx_k];
    float dRk = dR_dPsi[idx];
    float d2Rk = d2R_dPsi2[idx];

    int exclStart = exclusionStart[templateIdx_k];
    int exclEnd = exclusionStart[templateIdx_k + 1];
    int groupSize = groupEndIdx - groupStartIdx;

    // Zero row
    for (int c = 0; c < totalParticles; c++) {
        couplingMatrix[idx * totalParticles + c] = 0.0f;
    }

    // Diagonal: self energy second derivative
    // E_self = 0.5 * pf * qk² / Rk
    // d²E_self/dRk² = pf * qk² / Rk³
    float d2E_self = prefactor * q_k * q_k / (R_k * R_k * R_k);
    float diag = d2E_self * dRk * dRk;

    // Diagonal: SA second derivative if enabled
    // E_SA = σ * 4π * (R_intr + probe)² * (R_intr/R_born)^6
    // d²E_SA/dR² = 42 * E_SA / R_born²
    if (includeSA) {
        float Rprobe = R_k_intr + probeRadiusVal;
        float ratio = R_k_intr / R_k;
        float ratio6 = ratio * ratio * ratio * ratio * ratio * ratio;
        float E_SA = surfaceTension * 4.0f * 3.14159265359f * Rprobe * Rprobe * ratio6;
        float d2E_SA = 42.0f * E_SA / (R_k * R_k);
        diag += d2E_SA * dRk * dRk;
    }

    // Diagonal: OBC curvature term
    diag += dE_dR[idx] * d2Rk;

    // Pairwise contributions to diagonal and off-diagonal
    for (int lLocal = 0; lLocal < groupSize; lLocal++) {
        if (lLocal == atomInGroup) continue;

        int l = groupStartIdx + lLocal;
        int templateIdx_l = lLocal % templateNumAtoms;

        bool excluded = false;
        for (int e = exclStart; e < exclEnd; e++) {
            if (exclusionAtoms[e] == templateIdx_l) { excluded = true; break; }
        }
        if (excluded) continue;

        int particleIdx_l = particleIndices[l];
        float4 pos_l = posq[particleIdx_l];
        float q_l = charges[templateIdx_l];
        float R_l = bornRadii[l];
        float dRl = dR_dPsi[l];

        float dx = pos_l.x - pos_k.x;
        float dy = pos_l.y - pos_k.y;
        float dz = pos_l.z - pos_k.z;
        float r2 = dx*dx + dy*dy + dz*dz;

        // Still equation intermediates
        float RkRl = R_k * R_l;
        float expArg = -r2 / (4.0f * RkRl);
        float alpha = expf(expArg);
        float f2 = r2 + RkRl * alpha;
        float f = sqrtf(f2);
        float C = prefactor * q_k * q_l;

        // df²/dRk = α*(Rl + 0.25*r²/Rk)
        float df2_dRk = alpha * (R_l + 0.25f * r2 / R_k);
        float df_dRk = df2_dRk / (2.0f * f);

        // d²f²/dRk²
        float dalpha_dRk = alpha * r2 / (4.0f * R_k * R_k * R_l);
        float d2f2_dRk2 = dalpha_dRk * (R_l + 0.25f * r2 / R_k) + alpha * (-0.25f * r2 / (R_k * R_k));
        float d2f_dRk2 = d2f2_dRk2 / (2.0f * f) - df2_dRk * df2_dRk / (4.0f * f * f * f);

        // d²E/dRk² for this pair
        float d2E_dRk2 = C * (2.0f * df_dRk * df_dRk / (f * f * f) - d2f_dRk2 / (f * f));
        diag += d2E_dRk2 * dRk * dRk;

        // Off-diagonal: d²E/(dRk dRl)
        float df2_dRl = alpha * (R_k + 0.25f * r2 / R_l);
        float df_dRl = df2_dRl / (2.0f * f);
        float dalpha_dRl = alpha * r2 / (4.0f * R_k * R_l * R_l);
        float d2f2_dRkRl = dalpha_dRl * (R_l + 0.25f * r2 / R_k) + alpha;
        float d2f_dRkRl = d2f2_dRkRl / (2.0f * f) - df2_dRk * df2_dRl / (4.0f * f * f * f);
        float d2E_dRkRl = C * (2.0f * df_dRk * df_dRl / (f * f * f) - d2f_dRkRl / (f * f));

        couplingMatrix[idx * totalParticles + l] = d2E_dRkRl * dRk * dRl;
    }

    couplingMatrix[idx * totalParticles + idx] = diag;
}

/**
 * Kernel 4: Assemble the full 3N x 3N GBSA Hessian.
 *
 * H = H_direct + H_cross + J^T·M·J + H_hct2
 *
 * H_direct: radial Hessian from Still equation distance dependence (frozen Born radii)
 * H_cross:  mixed d²E/(dr·dR_k) coupling distance and Born radius changes
 * J^T·M·J:  Born radius coupling through HCT Jacobian
 * H_hct2:   Σ_k dE/dΨ_k · d²Ψ_k/dx²  (HCT second derivatives)
 *
 * One thread per upper-triangle element of H[3N x 3N].
 */
extern "C" __global__ void assembleGBSAHessian(
    const float4* __restrict__ posq,
    const int* __restrict__ particleIndices,
    const float* __restrict__ charges,
    const float* __restrict__ bornRadii,
    const int* __restrict__ exclusionAtoms,
    const int* __restrict__ exclusionStart,
    const int* __restrict__ groupStart,
    int numGroups,
    int templateNumAtoms,
    float prefactor,
    const float* __restrict__ jacobian,
    const float* __restrict__ couplingMatrix,
    const float* __restrict__ dE_dHCT,
    const float* __restrict__ scaleFactors,
    const float* __restrict__ intrinsicRadii,
    const float* __restrict__ dR_dPsi,
    const float* __restrict__ gridHCTHessian,
    int totalParticles,
    float* __restrict__ hessian
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int dim3N = 3 * totalParticles;
    if (tid >= dim3N * dim3N) return;

    int row = tid / dim3N;
    int col = tid % dim3N;
    if (col < row) return;  // upper-triangle threads write both (r,c) and (c,r)

    int atom_i = row / 3, alpha = row % 3;
    int atom_j = col / 3, beta  = col % 3;

    // Find group for atom_i
    int atomInGroup_i = atom_i, groupStart_i = 0, groupEnd_i = 0;
    for (int g = 0; g < numGroups; g++) {
        groupStart_i = groupStart[g];
        groupEnd_i = groupStart[g + 1];
        if (atom_i >= groupStart_i && atom_i < groupEnd_i) {
            atomInGroup_i = atom_i - groupStart_i; break;
        }
    }
    int templateIdx_i = atomInGroup_i % templateNumAtoms;
    int exclStart_i = exclusionStart[templateIdx_i];
    int exclEnd_i = exclusionStart[templateIdx_i + 1];

    float H_val = 0.0f;

    // --- H_direct + H_cross + H_hct2 (computed pair-by-pair) ---
    if (atom_i == atom_j) {
        // Diagonal block: accumulate from all pairs involving atom_i
        int particleIdx_i = particleIndices[atom_i];
        float4 pos_i = posq[particleIdx_i];
        float q_i = charges[templateIdx_i];
        float R_i = bornRadii[atom_i];
        int groupSize = groupEnd_i - groupStart_i;

        float Ri_off = intrinsicRadii[templateIdx_i] - DIELECTRIC_OFFSET;
        float Si = Ri_off * scaleFactors[templateIdx_i];

        for (int lLocal = 0; lLocal < groupSize; lLocal++) {
            if (lLocal == atomInGroup_i) continue;
            int l = groupStart_i + lLocal;
            int tl = lLocal % templateNumAtoms;
            bool excl = false;
            for (int e = exclStart_i; e < exclEnd_i; e++)
                if (exclusionAtoms[e] == tl) { excl = true; break; }
            if (excl) continue;

            float4 pos_l = posq[particleIndices[l]];
            float q_l = charges[tl];
            float R_l = bornRadii[l];
            float dx = pos_l.x - pos_i.x, dy = pos_l.y - pos_i.y, dz = pos_l.z - pos_i.z;
            float r2 = dx*dx+dy*dy+dz*dz, r = sqrtf(r2);
            if (r < 1e-10f) continue;
            float RiRl = R_i * R_l;
            float et = expf(-r2/(4.0f*RiRl));
            float f2 = r2 + RiRl*et, f = sqrtf(f2);
            float C = prefactor * q_i * q_l;

            // H_direct: Still equation distance derivatives (frozen Born radii)
            float df2_dr = 2.0f*r*(1.0f-0.25f*et);
            float df_dr = df2_dr/(2.0f*f);
            float dEdr = -C*df_dr/(f*f);
            float d2f2_dr2 = 2.0f - 0.5f*et + 0.25f*r2*et/RiRl;
            float d2f_dr2 = d2f2_dr2/(2.0f*f) - df2_dr*df2_dr/(4.0f*f*f*f);
            float d2Edr2 = C*(2.0f*df_dr*df_dr/(f*f*f) - d2f_dr2/(f*f));

            // H_hct2: HCT second derivative contribution
            float Rl_off = intrinsicRadii[tl] - DIELECTRIC_OFFSET;
            float Sl = Rl_off * scaleFactors[tl];

            if (Ri_off < r + Sl) {
                float I_il[7];
                computeHCT_rDerivs(r, Sl, Ri_off, I_il);
                dEdr += dE_dHCT[atom_i] * I_il[1];
                d2Edr2 += dE_dHCT[atom_i] * I_il[2];
            }
            if (Rl_off < r + Si) {
                float I_li[7];
                computeHCT_rDerivs(r, Si, Rl_off, I_li);
                dEdr += dE_dHCT[l] * I_li[1];
                d2Edr2 += dE_dHCT[l] * I_li[2];
            }

            // H_cross: d²E/(dr dR_k) * dR_k/dΨ_k cross terms
            // For this pair, need d²E/(dr dR_i) and d²E/(dr dR_l)
            float dalpha_dRi = et * r2 / (4.0f * R_i * R_i * R_l);
            float d2f2_dr_dRi = 2.0f * r * (-0.25f * dalpha_dRi);
            float df2_dRi = et * (R_l + 0.25f * r2 / R_i);
            float df_dRi = df2_dRi / (2.0f * f);
            float d2f_dr_dRi = d2f2_dr_dRi / (2.0f * f) - df2_dr * df2_dRi / (4.0f * f * f * f);
            float d2E_dr_dRi = C * (2.0f * df_dr * df_dRi / (f*f*f) - d2f_dr_dRi / (f*f));
            float g_i_val = d2E_dr_dRi * dR_dPsi[atom_i];

            float dalpha_dRl = et * r2 / (4.0f * R_i * R_l * R_l);
            float d2f2_dr_dRl = 2.0f * r * (-0.25f * dalpha_dRl);
            float df2_dRl = et * (R_i + 0.25f * r2 / R_l);
            float df_dRl = df2_dRl / (2.0f * f);
            float d2f_dr_dRl = d2f2_dr_dRl / (2.0f * f) - df2_dr * df2_dRl / (4.0f * f * f * f);
            float d2E_dr_dRl = C * (2.0f * df_dr * df_dRl / (f*f*f) - d2f_dr_dRl / (f*f));
            float g_l_val = d2E_dr_dRl * dR_dPsi[l];

            float D[3] = {dx, dy, dz};
            float ir = 1.0f/r, ir2 = ir*ir;

            // H_direct + H_hct2 radial Hessian (diagonal block = sum of -H_offdiag)
            H_val += (d2Edr2 - dEdr*ir)*D[alpha]*D[beta]*ir2
                   + ((alpha==beta) ? dEdr*ir : 0.0f);

            // H_cross for diagonal block (atom_i == atom_j == a == b):
            // dr/dx_i^alpha = -D[alpha]/r, J[k, 3i+beta] from Jacobian
            // Contribution: Σ_k g_k * (dr/dx_i^alpha * J[k, 3i+beta] + J[k, 3i+alpha] * dr/dx_i^beta)
            float dr_dxi_a = -D[alpha] * ir;
            float dr_dxi_b = -D[beta] * ir;
            H_val += g_i_val * (dr_dxi_a * jacobian[atom_i * dim3N + 3*atom_i + beta]
                              + jacobian[atom_i * dim3N + 3*atom_i + alpha] * dr_dxi_b);
            H_val += g_l_val * (dr_dxi_a * jacobian[l * dim3N + 3*atom_i + beta]
                              + jacobian[l * dim3N + 3*atom_i + alpha] * dr_dxi_b);

            // Also cross contribution from pair (i,l) where a=b=i uses dr_l/dx_i = -D/r:
            // (already handled above since dr/dx_i and J[k, 3*atom_i] are the same)
        }

        // Grid H_hct2: receptor grid second derivatives for diagonal block
        // d²Ψ_grid_k/(dx^α dx^β) stored as [xx, yy, zz, xy, xz, yz]
        {
            int hess_idx;
            if (alpha == beta)
                hess_idx = alpha;  // 0=xx, 1=yy, 2=zz
            else {
                int mn = alpha < beta ? alpha : beta;
                int mx = alpha > beta ? alpha : beta;
                hess_idx = (mn == 0 && mx == 1) ? 3 : (mn == 0 ? 4 : 5);  // 3=xy, 4=xz, 5=yz
            }
            H_val += dE_dHCT[atom_i] * gridHCTHessian[atom_i * 6 + hess_idx];
        }

    } else if (atom_j >= groupStart_i && atom_j < groupEnd_i) {
        // Off-diagonal, same group
        int atomInGroup_j = atom_j - groupStart_i;
        int templateIdx_j = atomInGroup_j % templateNumAtoms;
        int groupSize = groupEnd_i - groupStart_i;

        // --- H_direct + H_hct2: only from the specific pair (atom_i, atom_j) ---
        bool excl_ij = false;
        for (int e = exclStart_i; e < exclEnd_i; e++)
            if (exclusionAtoms[e] == templateIdx_j) { excl_ij = true; break; }

        if (!excl_ij) {
            float4 pos_i = posq[particleIndices[atom_i]];
            float4 pos_j = posq[particleIndices[atom_j]];
            float q_i = charges[templateIdx_i], q_j = charges[templateIdx_j];
            float R_i = bornRadii[atom_i], R_j = bornRadii[atom_j];
            float dx = pos_j.x-pos_i.x, dy = pos_j.y-pos_i.y, dz = pos_j.z-pos_i.z;
            float r2 = dx*dx+dy*dy+dz*dz, r = sqrtf(r2);
            if (r >= 1e-10f) {
                float RiRj = R_i*R_j;
                float et = expf(-r2/(4.0f*RiRj));
                float f2 = r2+RiRj*et, f = sqrtf(f2);
                float C = prefactor*q_i*q_j;
                float df2_dr = 2.0f*r*(1.0f-0.25f*et);
                float df_dr = df2_dr/(2.0f*f);
                float dEdr = -C*df_dr/(f*f);
                float d2f2_dr2 = 2.0f-0.5f*et+0.25f*r2*et/RiRj;
                float d2f_dr2 = d2f2_dr2/(2.0f*f)-df2_dr*df2_dr/(4.0f*f*f*f);
                float d2Edr2 = C*(2.0f*df_dr*df_dr/(f*f*f)-d2f_dr2/(f*f));

                // H_hct2
                float Ri_off = intrinsicRadii[templateIdx_i] - DIELECTRIC_OFFSET;
                float Rj_off = intrinsicRadii[templateIdx_j] - DIELECTRIC_OFFSET;
                float Si = Ri_off * scaleFactors[templateIdx_i];
                float Sj = Rj_off * scaleFactors[templateIdx_j];
                if (Ri_off < r + Sj) {
                    float I_ij[7];
                    computeHCT_rDerivs(r, Sj, Ri_off, I_ij);
                    dEdr += dE_dHCT[atom_i] * I_ij[1];
                    d2Edr2 += dE_dHCT[atom_i] * I_ij[2];
                }
                if (Rj_off < r + Si) {
                    float I_ji[7];
                    computeHCT_rDerivs(r, Si, Rj_off, I_ji);
                    dEdr += dE_dHCT[atom_j] * I_ji[1];
                    d2Edr2 += dE_dHCT[atom_j] * I_ji[2];
                }

                float ir = 1.0f/r, ir2 = ir*ir;
                float D_ij[3] = {dx, dy, dz};
                H_val += -(d2Edr2-dEdr*ir)*D_ij[alpha]*D_ij[beta]*ir2
                       - ((alpha==beta) ? dEdr*ir : 0.0f);
            }
        }

        // --- H_cross: contributions from ALL pairs involving atom_i or atom_j ---
        // H_cross = Term1 + Term1^T  where
        // Term1[row, col] = Σ_{l: pair(atom_i,l)} (-D[α]/r) * (g_i*J[i,col] + g_l*J[l,col])
        // Term1^T[row, col] = Σ_{m: pair(atom_j,m)} (-D'[β]/r') * (g_j*J[j,row] + g_m*J[m,row])

        float4 pos_ii = posq[particleIndices[atom_i]];
        float4 pos_jj = posq[particleIndices[atom_j]];

        // Part A: all pairs involving atom_i → contributes dr/dx_i * v[col]
        for (int lLocal = 0; lLocal < groupSize; lLocal++) {
            if (lLocal == atomInGroup_i) continue;
            int l = groupStart_i + lLocal;
            int tl = lLocal % templateNumAtoms;
            bool excl_l = false;
            for (int e = exclStart_i; e < exclEnd_i; e++)
                if (exclusionAtoms[e] == tl) { excl_l = true; break; }
            if (excl_l) continue;

            float4 pos_l = posq[particleIndices[l]];
            float R_ii = bornRadii[atom_i], R_l = bornRadii[l];
            float dx_ = pos_l.x-pos_ii.x, dy_ = pos_l.y-pos_ii.y, dz_ = pos_l.z-pos_ii.z;
            float r2_ = dx_*dx_+dy_*dy_+dz_*dz_, r_ = sqrtf(r2_);
            if (r_ < 1e-10f) continue;

            float RiRl = R_ii * R_l;
            float et_ = expf(-r2_/(4.0f*RiRl));
            float f_ = sqrtf(r2_+RiRl*et_);
            float C_ = prefactor*charges[templateIdx_i]*charges[tl];
            float df2_ = 2.0f*r_*(1.0f-0.25f*et_);
            float df_ = df2_/(2.0f*f_);

            float da_i = et_*r2_/(4.0f*R_ii*R_ii*R_l);
            float d2f2ri = 2.0f*r_*(-0.25f*da_i);
            float df2ri = et_*(R_l+0.25f*r2_/R_ii);
            float dfri = df2ri/(2.0f*f_);
            float d2fri = d2f2ri/(2.0f*f_) - df2_*df2ri/(4.0f*f_*f_*f_);
            float g_i_ = C_*(2.0f*df_*dfri/(f_*f_*f_) - d2fri/(f_*f_)) * dR_dPsi[atom_i];

            float da_l = et_*r2_/(4.0f*R_ii*R_l*R_l);
            float d2f2rl = 2.0f*r_*(-0.25f*da_l);
            float df2rl = et_*(R_ii+0.25f*r2_/R_l);
            float dfrl = df2rl/(2.0f*f_);
            float d2frl = d2f2rl/(2.0f*f_) - df2_*df2rl/(4.0f*f_*f_*f_);
            float g_l_ = C_*(2.0f*df_*dfrl/(f_*f_*f_) - d2frl/(f_*f_)) * dR_dPsi[l];

            float D_[3] = {dx_, dy_, dz_};
            float dr_a = -D_[alpha] / r_;
            float v_col = g_i_ * jacobian[atom_i * dim3N + col] + g_l_ * jacobian[l * dim3N + col];
            H_val += dr_a * v_col;
        }

        // Part B: all pairs involving atom_j → contributes v[row] * dr/dx_j
        int exclStart_j = exclusionStart[templateIdx_j];
        int exclEnd_j = exclusionStart[templateIdx_j + 1];
        for (int mLocal = 0; mLocal < groupSize; mLocal++) {
            if (mLocal == atomInGroup_j) continue;
            int m = groupStart_i + mLocal;
            int tm = mLocal % templateNumAtoms;
            bool excl_m = false;
            for (int e = exclStart_j; e < exclEnd_j; e++)
                if (exclusionAtoms[e] == tm) { excl_m = true; break; }
            if (excl_m) continue;

            float4 pos_m = posq[particleIndices[m]];
            float R_jj = bornRadii[atom_j], R_m = bornRadii[m];
            float dx_ = pos_m.x-pos_jj.x, dy_ = pos_m.y-pos_jj.y, dz_ = pos_m.z-pos_jj.z;
            float r2_ = dx_*dx_+dy_*dy_+dz_*dz_, r_ = sqrtf(r2_);
            if (r_ < 1e-10f) continue;

            float RjRm = R_jj * R_m;
            float et_ = expf(-r2_/(4.0f*RjRm));
            float f_ = sqrtf(r2_+RjRm*et_);
            float C_ = prefactor*charges[templateIdx_j]*charges[tm];
            float df2_ = 2.0f*r_*(1.0f-0.25f*et_);
            float df_ = df2_/(2.0f*f_);

            float da_j = et_*r2_/(4.0f*R_jj*R_jj*R_m);
            float d2f2rj = 2.0f*r_*(-0.25f*da_j);
            float df2rj = et_*(R_m+0.25f*r2_/R_jj);
            float dfrj = df2rj/(2.0f*f_);
            float d2frj = d2f2rj/(2.0f*f_) - df2_*df2rj/(4.0f*f_*f_*f_);
            float g_j_ = C_*(2.0f*df_*dfrj/(f_*f_*f_) - d2frj/(f_*f_)) * dR_dPsi[atom_j];

            float da_m = et_*r2_/(4.0f*R_jj*R_m*R_m);
            float d2f2rm = 2.0f*r_*(-0.25f*da_m);
            float df2rm = et_*(R_jj+0.25f*r2_/R_m);
            float dfrm = df2rm/(2.0f*f_);
            float d2frm = d2f2rm/(2.0f*f_) - df2_*df2rm/(4.0f*f_*f_*f_);
            float g_m_ = C_*(2.0f*df_*dfrm/(f_*f_*f_) - d2frm/(f_*f_)) * dR_dPsi[m];

            float D_[3] = {dx_, dy_, dz_};
            float dr_b = -D_[beta] / r_;
            float v_row = g_j_ * jacobian[atom_j * dim3N + row] + g_m_ * jacobian[m * dim3N + row];
            H_val += v_row * dr_b;
        }
    }

    // --- J^T · M · J ---
    for (int k = 0; k < totalParticles; k++) {
        float Jk = jacobian[k * dim3N + row];
        if (fabsf(Jk) < 1e-15f) continue;
        for (int l = 0; l < totalParticles; l++) {
            float Mkl = couplingMatrix[k * totalParticles + l];
            if (fabsf(Mkl) < 1e-15f) continue;
            H_val += Jk * Mkl * jacobian[l * dim3N + col];
        }
    }

write_result:
    hessian[row * dim3N + col] = H_val;
    if (col > row) hessian[col * dim3N + row] = H_val;
}
