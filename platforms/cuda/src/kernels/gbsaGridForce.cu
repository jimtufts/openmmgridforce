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

        // Apply correction
        if (N > 0.5f) {
            float invRi = 1.0f / R_i_off;
            float invRp = 1.0f / R_probe_off;
            float delta = invRi - invRp;
            float sigma = invRi + invRp;
            float logTerm = logf(R_i_off / R_probe_off);

            result.hct = hct + delta * (N - 0.25f * A * sigma) + B * logTerm;

            if (computeGradient) {
                // Combine gradients using correction formula coefficients
                float dCorr_dN = delta;
                float dCorr_dA = -0.25f * delta * sigma;
                float dCorr_dB = logTerm;

                result.gradient.x = mgResult.gradients[0].x +
                    dCorr_dN * mgResult.gradients[1].x +
                    dCorr_dA * mgResult.gradients[2].x +
                    dCorr_dB * mgResult.gradients[3].x;
                result.gradient.y = mgResult.gradients[0].y +
                    dCorr_dN * mgResult.gradients[1].y +
                    dCorr_dA * mgResult.gradients[2].y +
                    dCorr_dB * mgResult.gradients[3].y;
                result.gradient.z = mgResult.gradients[0].z +
                    dCorr_dN * mgResult.gradients[1].z +
                    dCorr_dA * mgResult.gradients[2].z +
                    dCorr_dB * mgResult.gradients[3].z;
            }
        } else {
            result.hct = hct;
            if (computeGradient) {
                result.gradient = mgResult.gradients[0];
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

            if (N > 0.5f) {
                // Apply correction
                float invRi = 1.0f / R_i_off;
                float invRp = 1.0f / R_probe_off;
                float delta = invRi - invRp;
                float sigma = invRi + invRp;
                float logTerm = logf(R_i_off / R_probe_off);

                result.hct = hct + delta * (N - 0.25f * A * sigma) + B * logTerm;

                if (computeGradient) {
                    // Chain rule coefficients for correction formula
                    float dCorr_dN = delta;
                    float dCorr_dA = -0.25f * delta * sigma;
                    float dCorr_dB = logTerm;

                    // Combine HCT gradient with correction gradients
                    result.gradient.x = hctResult.gradient.x +
                        dCorr_dN * nGrad.x + dCorr_dA * aGrad.x + dCorr_dB * bGrad.x;
                    result.gradient.y = hctResult.gradient.y +
                        dCorr_dN * nGrad.y + dCorr_dA * aGrad.y + dCorr_dB * bGrad.y;
                    result.gradient.z = hctResult.gradient.z +
                        dCorr_dN * nGrad.z + dCorr_dA * aGrad.z + dCorr_dB * bGrad.z;
                }
            } else {
                result.hct = hct;
                if (computeGradient) {
                    result.gradient = hctResult.gradient;
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

        if (N > 0.5f) {
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
                // Gradient coefficients
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
        } else {
            result.hct = hct;
            if (computeGradient) {
                float dh_dfx = vp - vm;
                float dh_dfy = ox * (vmp - vmm) + fx * (vpp - vpm);
                float dh_dfz = ox * (oy * (h001 - h000) + fy * (h011 - h010)) +
                               fx * (oy * (h101 - h100) + fy * (h111 - h110));
                result.gradient.x = dh_dfx * invSpacing;
                result.gradient.y = dh_dfy * invSpacing;
                result.gradient.z = dh_dfz * invSpacing;
            }
        }
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

        // Force on i from j
        float invR = 1.0f / r;
        force.x -= dEdR * dx * invR;
        force.y -= dEdR * dy * invR;
        force.z -= dEdR * dz * invR;

        // Accumulate force on j (Newton's 3rd law)
        // Using atomic add for thread safety
        atomicAdd(&forceBuffer[particleIdx_j], static_cast<unsigned long long>((long long)(dEdR * dx * invR * 0x100000000)));
        atomicAdd(&forceBuffer[particleIdx_j + paddedNumAtoms], static_cast<unsigned long long>((long long)(dEdR * dy * invR * 0x100000000)));
        atomicAdd(&forceBuffer[particleIdx_j + 2*paddedNumAtoms], static_cast<unsigned long long>((long long)(dEdR * dz * invR * 0x100000000)));
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

        // Count each pair once (j>i contribution doubled)
        dE_dRi += 0.5f * dE_pair_dRi;  // Half because we count both i<j and i>j
    }

    dE_dR[idx] = dE_dRi;
}


/**
 * Compute forces from HCT chain rule through Born radii.
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
    float R_born_i = bornRadii[idx];

    // Compute dR_born/dpsi using OBC-II derivative
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

    // dE/dHCT_i = dE/dR_i * dR_i/dpsi_i * dpsi_i/dHCT_i
    // where dpsi/dHCT = 0.5 * R_off
    float dE_dHCT_i = dE_dR[idx] * dR_dpsi_i * 0.5f * R_i_off;

    int exclStart = exclusionStart[templateIdx_i];
    int exclEnd = exclusionStart[templateIdx_i + 1];

    float3 force = make_float3(0.0f, 0.0f, 0.0f);

    // Force from ligand-ligand HCT derivatives
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

        float dx = pos_i.x - pos_j.x;
        float dy = pos_i.y - pos_j.y;
        float dz = pos_i.z - pos_j.z;
        float r2 = dx*dx + dy*dy + dz*dz;
        float r = sqrtf(r2);

        if (r < 1e-6f) continue;

        // Compute dHCT_i/dr for contribution of atom j to HCT of atom i
        float r_plus_Sj = r + S_j;
        if (R_i_off >= r_plus_Sj) continue;

        float r_minus_Sj = fabsf(r - S_j);

        // Determine case and compute derivative
        float dHCT_dr;

        if (R_i_off > r_minus_Sj) {
            // Case 1: l = 1/R_i (constant in r)
            float u = 1.0f / r_plus_Sj;
            float l = 1.0f / R_i_off;
            float u2 = u * u;
            float l2 = l * l;
            float S2 = S_j * S_j;
            float invr = 1.0f / r;
            float invr2 = invr * invr;

            float du_dr = -u2;
            float ln_u_l = logf(u / l);

            dHCT_dr = -du_dr  // = u²
                    + 0.25f * (u2 - l2)
                    + 0.5f * r * u * du_dr  // = -0.5 * r * u³
                    - 0.5f * invr2 * ln_u_l
                    + 0.5f * invr * (-1.0f) * du_dr  // = 0.5/r * u²
                    - 0.25f * S2 * invr2 * (l2 - u2)
                    + 0.25f * S2 * invr * 2.0f * u * (-du_dr);  // = 0.5 * S²/r * u³
        } else {
            // Case 2: l = 1/(r-S) for r > S
            float rms = r - S_j;
            float l = 1.0f / rms;
            float u = 1.0f / r_plus_Sj;
            float l2 = l * l;
            float u2 = u * u;
            float S2 = S_j * S_j;
            float invr = 1.0f / r;
            float invr2 = invr * invr;

            float dl_dr = -l2;
            float du_dr = -u2;
            float ln_u_l = logf(u / l);

            dHCT_dr = dl_dr - du_dr
                    + 0.25f * (u2 - l2)
                    + 0.5f * r * (u * du_dr - l * dl_dr)
                    - 0.5f * invr2 * ln_u_l
                    + 0.5f * invr * ((1.0f/u) * du_dr - (1.0f/l) * dl_dr)
                    - 0.25f * S2 * invr2 * (l2 - u2)
                    + 0.25f * S2 * invr * (2.0f * l * dl_dr - 2.0f * u * du_dr);
        }

        // Force contribution: F_i += -dE_dHCT_i * dHCT_i/dr * dr/dx_i
        float invr = 1.0f / r;
        float dE_dr_i = dE_dHCT_i * dHCT_dr;

        // This force acts on i due to HCT_i depending on position of j
        force.x -= dE_dr_i * dx * invr;
        force.y -= dE_dr_i * dy * invr;
        force.z -= dE_dr_i * dz * invr;

        // Also need force on i from HCT_j depending on position of i
        // Compute dE_dHCT_j
        float R_born_j = bornRadii[j];
        float hctTotal_j = hctReceptor[j] + hctLigand[j];
        float psi_j = 0.5f * R_j_off * hctTotal_j;
        float psi2_j = psi_j * psi_j;
        float psi3_j = psi2_j * psi_j;
        float tanh_arg_j = OBC_ALPHA * psi_j - OBC_BETA * psi2_j + OBC_GAMMA * psi3_j;
        float t_j = tanhf(tanh_arg_j);
        float sech2_j = 1.0f - t_j * t_j;
        float darg_dpsi_j = OBC_ALPHA - 2.0f * OBC_BETA * psi_j + 3.0f * OBC_GAMMA * psi2_j;
        float dR_dpsi_j = R_born_j * R_born_j * sech2_j * darg_dpsi_j / R_j;
        float dE_dHCT_j = dE_dR[j] * dR_dpsi_j * 0.5f * R_j_off;

        // Compute dHCT_j/dr for contribution of atom i to HCT of atom j
        float S_i = R_i_off * scaleFactors[templateIdx_i];
        float r_plus_Si = r + S_i;
        if (R_j_off < r_plus_Si) {
            float r_minus_Si = fabsf(r - S_i);
            float dHCT_j_dr;

            if (R_j_off > r_minus_Si) {
                // Case 1 for j
                float u = 1.0f / r_plus_Si;
                float l = 1.0f / R_j_off;
                float u2 = u * u;
                float l2 = l * l;
                float S2 = S_i * S_i;
                float invr_loc = 1.0f / r;
                float invr2 = invr_loc * invr_loc;

                float du_dr = -u2;
                float ln_u_l = logf(u / l);

                dHCT_j_dr = -du_dr
                          + 0.25f * (u2 - l2)
                          + 0.5f * r * u * du_dr
                          - 0.5f * invr2 * ln_u_l
                          + 0.5f * invr_loc * (-1.0f) * du_dr
                          - 0.25f * S2 * invr2 * (l2 - u2)
                          + 0.25f * S2 * invr_loc * 2.0f * u * (-du_dr);
            } else {
                // Case 2 for j
                float rms = r - S_i;
                float l = 1.0f / rms;
                float u = 1.0f / r_plus_Si;
                float l2 = l * l;
                float u2 = u * u;
                float S2 = S_i * S_i;
                float invr_loc = 1.0f / r;
                float invr2 = invr_loc * invr_loc;

                float dl_dr = -l2;
                float du_dr = -u2;
                float ln_u_l = logf(u / l);

                dHCT_j_dr = dl_dr - du_dr
                          + 0.25f * (u2 - l2)
                          + 0.5f * r * (u * du_dr - l * dl_dr)
                          - 0.5f * invr2 * ln_u_l
                          + 0.5f * invr_loc * ((1.0f/u) * du_dr - (1.0f/l) * dl_dr)
                          - 0.25f * S2 * invr2 * (l2 - u2)
                          + 0.25f * S2 * invr_loc * (2.0f * l * dl_dr - 2.0f * u * du_dr);
            }

            // Force on i from HCT_j
            float dE_dr_j = dE_dHCT_j * dHCT_j_dr;
            force.x += dE_dr_j * dx * invr;
            force.y += dE_dr_j * dy * invr;
            force.z += dE_dr_j * dz * invr;
        }
    }

    // Accumulate force
    atomicAdd(&forceBuffer[particleIdx_i], static_cast<unsigned long long>((long long)(force.x * 0x100000000)));
    atomicAdd(&forceBuffer[particleIdx_i + paddedNumAtoms], static_cast<unsigned long long>((long long)(force.y * 0x100000000)));
    atomicAdd(&forceBuffer[particleIdx_i + 2*paddedNumAtoms], static_cast<unsigned long long>((long long)(force.z * 0x100000000)));
}
