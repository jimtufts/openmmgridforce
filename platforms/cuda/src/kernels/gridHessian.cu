/**
 * CUDA implementation of grid Hessian (second derivative) and third derivative calculation.
 *
 * computeGridHessian: 3x3 Hessian block for each atom from the grid potential.
 *   Output: 6 unique components per atom: dxx, dyy, dzz, dxy, dxz, dyz
 *   Supports methods 1 (cubic B-spline), 3 (triquintic Hermite), 4 (quintic B-spline).
 *
 * computeGridThirdDerivatives: 10 unique third derivative components per atom.
 *   Output: d3xxx, d3yyy, d3zzz, d3xxy, d3xxz, d3xyy, d3xzz, d3yyz, d3yzz, d3xyz
 *   Only supports method 4 (quintic B-spline, C4 continuity).
 *
 * Both support inv_power and arcsinh chain rule transformations.
 */

#include "include/InterpolationBasis.cuh"
#include "include/TriquinticCoefficients.cuh"
#include "include/InvPowerChainRule.cuh"

/**
 * Apply inv_power chain rule to convert stored derivatives to actual derivatives.
 *
 * For V = sign(U) * |U|^p:
 *   dV/dx = p * |U|^(p-1) * dU/dx
 *   d²V/dx² = p*(p-1) * |U|^(p-2) * (dU/dx)² + p * |U|^(p-1) * d²U/dx²
 *   d²V/dxdy = p*(p-1) * |U|^(p-2) * dU/dx * dU/dy + p * |U|^(p-1) * d²U/dxdy
 *
 * @param U          Stored grid value
 * @param dUdx/y/z   Stored first derivatives
 * @param d2Uxx/yy/zz/xy/xz/yz  Stored second derivatives (modified in place)
 * @param p          Power parameter (1/invPower, e.g., -6 for inv_power=-6)
 */
__device__ inline void applyHessianChainRule(
    float U,
    float dUdx, float dUdy, float dUdz,
    float& d2xx, float& d2yy, float& d2zz,
    float& d2xy, float& d2xz, float& d2yz,
    float p
) {
    float absU = fabsf(U);
    if (absU < 1e-10f) absU = 1e-10f;  // Clamp to avoid divide by zero

    // Precompute powers
    float absU_pm1 = powf(absU, p - 1.0f);
    float absU_pm2 = powf(absU, p - 2.0f);

    // Chain rule coefficients
    float f2_1 = p * (p - 1.0f) * absU_pm2;  // For (dU/dx)² terms
    float f2_2 = p * absU_pm1;                // For d²U/dx² terms

    // Apply chain rule to each Hessian component
    float new_d2xx = f2_1 * dUdx * dUdx + f2_2 * d2xx;
    float new_d2yy = f2_1 * dUdy * dUdy + f2_2 * d2yy;
    float new_d2zz = f2_1 * dUdz * dUdz + f2_2 * d2zz;
    float new_d2xy = f2_1 * dUdx * dUdy + f2_2 * d2xy;
    float new_d2xz = f2_1 * dUdx * dUdz + f2_2 * d2xz;
    float new_d2yz = f2_1 * dUdy * dUdz + f2_2 * d2yz;

    d2xx = new_d2xx;
    d2yy = new_d2yy;
    d2zz = new_d2zz;
    d2xy = new_d2xy;
    d2xz = new_d2xz;
    d2yz = new_d2yz;
}

/**
 * Compute Hessian blocks for all atoms using grid interpolation.
 *
 * @param posq              Atom positions (x, y, z, charge)
 * @param hessianBuffer     Output: 6 floats per atom [dxx, dyy, dzz, dxy, dxz, dyz]
 * @param gridCounts        Grid dimensions [nx, ny, nz]
 * @param gridSpacing       Grid spacing [dx, dy, dz]
 * @param gridValues        Grid potential values
 * @param scalingFactors    Per-atom scaling factors
 * @param invPower          Inverse power parameter (e.g., -6.0)
 * @param invPowerMode      0=NONE, 1=RUNTIME, 2=STORED
 * @param interpolationMethod  1=bspline, 3=triquintic (others not supported)
 * @param originX/Y/Z       Grid origin coordinates
 * @param gridDerivatives   Precomputed derivatives (required for triquintic)
 * @param numAtoms          Number of atoms to process
 * @param particleIndices   Optional particle index filtering
 */
extern "C" __global__ void computeGridHessian(
    const float4* __restrict__ posq,
    float* __restrict__ hessianBuffer,  // 6 components per atom
    const int* __restrict__ gridCounts,
    const float* __restrict__ gridSpacing,
    const float* __restrict__ gridValues,
    const float* __restrict__ scalingFactors,
    const float invPower,
    const int invPowerMode,  // 0=NONE, 1=RUNTIME, 2=STORED
    const int interpolationMethod,
    const float originX,
    const float originY,
    const float originZ,
    const float* __restrict__ gridDerivatives,
    const int numAtoms,
    const int* __restrict__ particleIndices,
    const float arcsinhScale)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= numAtoms)
        return;

    // Get actual particle index
    const unsigned int particleIndex = (particleIndices != nullptr) ? particleIndices[index] : index;

    // Load position and scaling factor
    float4 posOrig = posq[particleIndex];
    float scalingFactor = scalingFactors[particleIndex];

    // Transform to grid coordinates
    float3 pos;
    pos.x = posOrig.x - originX;
    pos.y = posOrig.y - originY;
    pos.z = posOrig.z - originZ;

    // Initialize interpolated value, first derivatives, and Hessian components to zero
    float interpolated = 0.0f;
    float dx = 0.0f, dy = 0.0f, dz = 0.0f;
    float d2xx = 0.0f, d2yy = 0.0f, d2zz = 0.0f;
    float d2xy = 0.0f, d2xz = 0.0f, d2yz = 0.0f;

    // Grid boundaries
    float3 gridCorner;
    gridCorner.x = gridSpacing[0] * (gridCounts[0] - 1);
    gridCorner.y = gridSpacing[1] * (gridCounts[1] - 1);
    gridCorner.z = gridSpacing[2] * (gridCounts[2] - 1);

    bool isInside = (pos.x >= 0.0f && pos.x <= gridCorner.x &&
                     pos.y >= 0.0f && pos.y <= gridCorner.y &&
                     pos.z >= 0.0f && pos.z <= gridCorner.z);

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

        int nyz = gridCounts[1] * gridCounts[2];

        if (interpolationMethod == 3 && gridDerivatives != nullptr) {
            // TRIQUINTIC HERMITE - Analytical second derivatives
            //
            // For RUNTIME mode: transform corners from actual potential (G) to smoothed space (S),
            // interpolate in S space, then back-convert to G space.
            // S = |G|^(1/n) where n = invPower (e.g., -6)
            // After interpolation: G = |S|^n

            int totalPoints = gridCounts[0] * gridCounts[1] * gridCounts[2];
            int corners[8][3] = {
                {ix, iy, iz}, {ix+1, iy, iz}, {ix, iy+1, iz}, {ix+1, iy+1, iz},
                {ix, iy, iz+1}, {ix+1, iy, iz+1}, {ix, iy+1, iz+1}, {ix+1, iy+1, iz+1}
            };

            // Gather derivatives - transform at corners for RUNTIME mode
            float X[216];
            if (invPowerMode == 1) {
                // RUNTIME mode: transform corners from G space to S space BEFORE interpolation
                // p = 1/invPower transforms G → S = |G|^p
                float p = 1.0f / invPower;
                for (int c = 0; c < 8; c++) {
                    int point_idx = corners[c][0] * nyz + corners[c][1] * gridCounts[2] + corners[c][2];
                    float G_derivs[27], S_derivs[27];
                    for (int d = 0; d < 27; d++) {
                        G_derivs[d] = gridDerivatives[d * totalPoints + point_idx];
                    }
                    applyInvPowerChainRule(G_derivs, p, S_derivs);
                    for (int d = 0; d < 27; d++) {
                        X[d * 8 + c] = S_derivs[d];
                    }
                }
            } else {
                // STORED or NONE mode: load directly without transformation
                for (int d = 0; d < 27; d++) {
                    for (int c = 0; c < 8; c++) {
                        int point_idx = corners[c][0] * nyz + corners[c][1] * gridCounts[2] + corners[c][2];
                        X[d * 8 + c] = gridDerivatives[d * totalPoints + point_idx];
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
            // P(x,y,z) = sum_{i,j,k} a[i+6j+36k] * x^i * y^j * z^k

            for (int k = 0; k < 6; k++) {
                for (int j = 0; j < 6; j++) {
                    for (int i = 0; i < 6; i++) {
                        int coeff_idx = i + 6*j + 36*k;
                        float coeff = a[coeff_idx];
                        float term = sx_pow[i] * sy_pow[j] * sz_pow[k];

                        // Interpolated value (needed for chain rule)
                        interpolated += coeff * term;

                        // dV/dx: need i >= 1
                        if (i >= 1) {
                            dx += coeff * i * sx_pow[i-1] * sy_pow[j] * sz_pow[k];
                        }

                        // dV/dy: need j >= 1
                        if (j >= 1) {
                            dy += coeff * j * sx_pow[i] * sy_pow[j-1] * sz_pow[k];
                        }

                        // dV/dz: need k >= 1
                        if (k >= 1) {
                            dz += coeff * k * sx_pow[i] * sy_pow[j] * sz_pow[k-1];
                        }

                        // d²V/dx²: need i >= 2
                        if (i >= 2) {
                            d2xx += coeff * (i * (i-1)) * sx_pow[i-2] * sy_pow[j] * sz_pow[k];
                        }

                        // d²V/dy²: need j >= 2
                        if (j >= 2) {
                            d2yy += coeff * (j * (j-1)) * sx_pow[i] * sy_pow[j-2] * sz_pow[k];
                        }

                        // d²V/dz²: need k >= 2
                        if (k >= 2) {
                            d2zz += coeff * (k * (k-1)) * sx_pow[i] * sy_pow[j] * sz_pow[k-2];
                        }

                        // d²V/dxdy: need i >= 1 and j >= 1
                        if (i >= 1 && j >= 1) {
                            d2xy += coeff * (i * j) * sx_pow[i-1] * sy_pow[j-1] * sz_pow[k];
                        }

                        // d²V/dxdz: need i >= 1 and k >= 1
                        if (i >= 1 && k >= 1) {
                            d2xz += coeff * (i * k) * sx_pow[i-1] * sy_pow[j] * sz_pow[k-1];
                        }

                        // d²V/dydz: need j >= 1 and k >= 1
                        if (j >= 1 && k >= 1) {
                            d2yz += coeff * (j * k) * sx_pow[i] * sy_pow[j-1] * sz_pow[k-1];
                        }
                    }
                }
            }

            // Undo transforms in reverse order of application during generation.
            // Generation order: inv_power → arcsinh → blur → prefilter.
            // Evaluation undo order: arcsinh first, then inv_power.

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

                interpolated = arcsinhScale * sinhG;
                dx = arcsinhScale * coshG * dx;
                dy = arcsinhScale * coshG * dy;
                dz = arcsinhScale * coshG * dz;

                d2xx = new_d2xx; d2yy = new_d2yy; d2zz = new_d2zz;
                d2xy = new_d2xy; d2xz = new_d2xz; d2yz = new_d2yz;
            }

            // Back-convert from smoothed space (S) to actual potential (G)
            // S = V^(1/n), G = sign(S)*|S|^n
            //   dG/dx = n*|S|^(n-1) * dS/dx
            //   d²G/dx² = n*(n-1)*|S|^(n-2)*(dS/dx)² + n*|S|^(n-1)*d²S/dx²
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

                    d2xx = new_d2xx;
                    d2yy = new_d2yy;
                    d2zz = new_d2zz;
                    d2xy = new_d2xy;
                    d2xz = new_d2xz;
                    d2yz = new_d2yz;
                }
            }

            // NOW convert from unit cell to physical coordinates
            float inv_dx = 1.0f / gridSpacing[0];
            float inv_dy = 1.0f / gridSpacing[1];
            float inv_dz = 1.0f / gridSpacing[2];
            float inv_dx2 = inv_dx * inv_dx;
            float inv_dy2 = inv_dy * inv_dy;
            float inv_dz2 = inv_dz * inv_dz;
            float inv_dxdy = inv_dx * inv_dy;
            float inv_dxdz = inv_dx * inv_dz;
            float inv_dydz = inv_dy * inv_dz;

            // First derivatives
            dx *= inv_dx;
            dy *= inv_dy;
            dz *= inv_dz;

            // Second derivatives
            d2xx *= inv_dx2;
            d2yy *= inv_dy2;
            d2zz *= inv_dz2;
            d2xy *= inv_dxdy;
            d2xz *= inv_dxdz;
            d2yz *= inv_dydz;

        } else if (interpolationMethod == 1) {
            // CUBIC B-SPLINE - Analytical second derivatives

            // Precompute basis functions and second derivatives
            float bx[4] = {bspline_basis0(fx), bspline_basis1(fx), bspline_basis2(fx), bspline_basis3(fx)};
            float by[4] = {bspline_basis0(fy), bspline_basis1(fy), bspline_basis2(fy), bspline_basis3(fy)};
            float bz[4] = {bspline_basis0(fz), bspline_basis1(fz), bspline_basis2(fz), bspline_basis3(fz)};

            float dbx[4] = {bspline_deriv0(fx), bspline_deriv1(fx), bspline_deriv2(fx), bspline_deriv3(fx)};
            float dby[4] = {bspline_deriv0(fy), bspline_deriv1(fy), bspline_deriv2(fy), bspline_deriv3(fy)};
            float dbz[4] = {bspline_deriv0(fz), bspline_deriv1(fz), bspline_deriv2(fz), bspline_deriv3(fz)};

            // Second derivatives of B-spline basis functions
            float d2bx[4] = {bspline_deriv2_0(fx), bspline_deriv2_1(fx), bspline_deriv2_2(fx), bspline_deriv2_3(fx)};
            float d2by[4] = {bspline_deriv2_0(fy), bspline_deriv2_1(fy), bspline_deriv2_2(fy), bspline_deriv2_3(fy)};
            float d2bz[4] = {bspline_deriv2_0(fz), bspline_deriv2_1(fz), bspline_deriv2_2(fz), bspline_deriv2_3(fz)};

            for (int i = 0; i < 4; i++) {
                int gx = min(max(ix - 1 + i, 0), gridCounts[0] - 1);
                for (int j = 0; j < 4; j++) {
                    int gy = min(max(iy - 1 + j, 0), gridCounts[1] - 1);
                    for (int k = 0; k < 4; k++) {
                        int gz = min(max(iz - 1 + k, 0), gridCounts[2] - 1);
                        int gridIdx = gx * nyz + gy * gridCounts[2] + gz;
                        float val = gridValues[gridIdx];

                        // Apply RUNTIME inv_power transformation before interpolation
                        // (matching force kernel behavior for consistency)
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

                    d2xx = new_d2xx;
                    d2yy = new_d2yy;
                    d2zz = new_d2zz;
                    d2xy = new_d2xy;
                    d2xz = new_d2xz;
                    d2yz = new_d2yz;
                }
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

            // First derivatives
            dx *= inv_dx;
            dy *= inv_dy;
            dz *= inv_dz;

            // Second derivatives
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
                int gx = min(max(ix - 2 + i, 0), gridCounts[0] - 1);
                for (int j = 0; j < 6; j++) {
                    int gy = min(max(iy - 2 + j, 0), gridCounts[1] - 1);
                    for (int k = 0; k < 6; k++) {
                        int gz = min(max(iz - 2 + k, 0), gridCounts[2] - 1);
                        int gridIdx = gx * nyz + gy * gridCounts[2] + gz;
                        float val = gridValues[gridIdx];

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
                applyHessianChainRule(interpolated, dx, dy, dz,
                                     d2xx, d2yy, d2zz, d2xy, d2xz, d2yz, invPower);
            }

            // Convert to physical coordinates
            float inv_dx = 1.0f / gridSpacing[0];
            float inv_dy = 1.0f / gridSpacing[1];
            float inv_dz = 1.0f / gridSpacing[2];

            dx *= inv_dx;
            dy *= inv_dy;
            dz *= inv_dz;

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

    // Store Hessian components (6 per atom)
    int offset = index * 6;
    hessianBuffer[offset + 0] = d2xx;
    hessianBuffer[offset + 1] = d2yy;
    hessianBuffer[offset + 2] = d2zz;
    hessianBuffer[offset + 3] = d2xy;
    hessianBuffer[offset + 4] = d2xz;
    hessianBuffer[offset + 5] = d2yz;
}


/**
 * Apply inv_power chain rule to third derivatives.
 *
 * For V = sign(U)*|U|^p, the third derivative d³V/dxi dxj dxk uses Faà di Bruno:
 *   f3_1 * product_of_first_derivs  +  f3_2 * (first * second_deriv combinations)  +  f3_3 * third_deriv
 *
 * All input derivatives (dU*, d2U*, d3*) must be in the ORIGINAL (pre-transform) space.
 * The output d3* values are overwritten with the transformed derivatives.
 */
__device__ inline void applyThirdDerivChainRule(
    float U,
    float dUdx, float dUdy, float dUdz,
    float d2Uxx, float d2Uyy, float d2Uzz,
    float d2Uxy, float d2Uxz, float d2Uyz,
    float& d3xxx, float& d3yyy, float& d3zzz,
    float& d3xxy, float& d3xxz, float& d3xyy,
    float& d3xzz, float& d3yyz, float& d3yzz,
    float& d3xyz,
    float p
) {
    float absU = fabsf(U);
    if (absU < 1e-10f) absU = 1e-10f;

    float absU_pm1 = powf(absU, p - 1.0f);
    float absU_pm2 = powf(absU, p - 2.0f);
    float absU_pm3 = powf(absU, p - 3.0f);

    float f3_1 = p * (p - 1.0f) * (p - 2.0f) * absU_pm3;
    float f3_2 = p * (p - 1.0f) * absU_pm2;
    float f3_3 = p * absU_pm1;

    d3xxx = f3_1*dUdx*dUdx*dUdx + 3.0f*f3_2*dUdx*d2Uxx                            + f3_3*d3xxx;
    d3yyy = f3_1*dUdy*dUdy*dUdy + 3.0f*f3_2*dUdy*d2Uyy                            + f3_3*d3yyy;
    d3zzz = f3_1*dUdz*dUdz*dUdz + 3.0f*f3_2*dUdz*d2Uzz                            + f3_3*d3zzz;
    d3xxy = f3_1*dUdx*dUdx*dUdy + f3_2*(2.0f*dUdx*d2Uxy + dUdy*d2Uxx)             + f3_3*d3xxy;
    d3xxz = f3_1*dUdx*dUdx*dUdz + f3_2*(2.0f*dUdx*d2Uxz + dUdz*d2Uxx)             + f3_3*d3xxz;
    d3xyy = f3_1*dUdx*dUdy*dUdy + f3_2*(dUdx*d2Uyy + 2.0f*dUdy*d2Uxy)             + f3_3*d3xyy;
    d3xzz = f3_1*dUdx*dUdz*dUdz + f3_2*(dUdx*d2Uzz + 2.0f*dUdz*d2Uxz)             + f3_3*d3xzz;
    d3yyz = f3_1*dUdy*dUdy*dUdz + f3_2*(2.0f*dUdy*d2Uyz + dUdz*d2Uyy)             + f3_3*d3yyz;
    d3yzz = f3_1*dUdy*dUdz*dUdz + f3_2*(dUdy*d2Uzz + 2.0f*dUdz*d2Uyz)             + f3_3*d3yzz;
    d3xyz = f3_1*dUdx*dUdy*dUdz + f3_2*(dUdx*d2Uyz + dUdy*d2Uxz + dUdz*d2Uxy)     + f3_3*d3xyz;
}


/**
 * Compute third derivative blocks for all atoms using quintic B-spline grid interpolation.
 *
 * Only supports interpolation method 4 (quintic B-spline, C4 continuity).
 * Output: 10 unique components per atom:
 *   [d3xxx, d3yyy, d3zzz, d3xxy, d3xxz, d3xyy, d3xzz, d3yyz, d3yzz, d3xyz]
 *
 * Parameters match computeGridHessian exactly, except output buffer has 10 components per atom.
 */
extern "C" __global__ void computeGridThirdDerivatives(
    const float4* __restrict__ posq,
    float* __restrict__ thirdDerivBuffer,  // 10 components per atom
    const int* __restrict__ gridCounts,
    const float* __restrict__ gridSpacing,
    const float* __restrict__ gridValues,
    const float* __restrict__ scalingFactors,
    const float invPower,
    const int invPowerMode,
    const int interpolationMethod,
    const float originX,
    const float originY,
    const float originZ,
    const float* __restrict__ gridDerivatives,
    const int numAtoms,
    const int* __restrict__ particleIndices,
    const float arcsinhScale)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= numAtoms)
        return;

    const unsigned int particleIndex = (particleIndices != nullptr) ? particleIndices[index] : index;

    float4 posOrig = posq[particleIndex];
    float scalingFactor = scalingFactors[particleIndex];

    float3 pos;
    pos.x = posOrig.x - originX;
    pos.y = posOrig.y - originY;
    pos.z = posOrig.z - originZ;

    // All derivatives initialized to zero
    float interpolated = 0.0f;
    float dx = 0.0f, dy = 0.0f, dz = 0.0f;
    float d2xx = 0.0f, d2yy = 0.0f, d2zz = 0.0f;
    float d2xy = 0.0f, d2xz = 0.0f, d2yz = 0.0f;
    float d3xxx = 0.0f, d3yyy = 0.0f, d3zzz = 0.0f;
    float d3xxy = 0.0f, d3xxz = 0.0f, d3xyy = 0.0f;
    float d3xzz = 0.0f, d3yyz = 0.0f, d3yzz = 0.0f;
    float d3xyz = 0.0f;

    float3 gridCorner;
    gridCorner.x = gridSpacing[0] * (gridCounts[0] - 1);
    gridCorner.y = gridSpacing[1] * (gridCounts[1] - 1);
    gridCorner.z = gridSpacing[2] * (gridCounts[2] - 1);

    bool isInside = (pos.x >= 0.0f && pos.x <= gridCorner.x &&
                     pos.y >= 0.0f && pos.y <= gridCorner.y &&
                     pos.z >= 0.0f && pos.z <= gridCorner.z);

    if (isInside && scalingFactor != 0.0f && interpolationMethod == 4) {
        int ix = min(max((int)(pos.x / gridSpacing[0]), 0), gridCounts[0] - 2);
        int iy = min(max((int)(pos.y / gridSpacing[1]), 0), gridCounts[1] - 2);
        int iz = min(max((int)(pos.z / gridSpacing[2]), 0), gridCounts[2] - 2);

        float fx = (pos.x / gridSpacing[0]) - ix;
        float fy = (pos.y / gridSpacing[1]) - iy;
        float fz = (pos.z / gridSpacing[2]) - iz;

        fx = min(max(fx, 0.0f), 1.0f);
        fy = min(max(fy, 0.0f), 1.0f);
        fz = min(max(fz, 0.0f), 1.0f);

        int nyz = gridCounts[1] * gridCounts[2];

        // Basis functions: value, 1st, 2nd, and 3rd derivatives
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

        float d3bx[6] = {qbspline_deriv3_0(fx), qbspline_deriv3_1(fx), qbspline_deriv3_2(fx),
                         qbspline_deriv3_3(fx), qbspline_deriv3_4(fx), qbspline_deriv3_5(fx)};
        float d3by[6] = {qbspline_deriv3_0(fy), qbspline_deriv3_1(fy), qbspline_deriv3_2(fy),
                         qbspline_deriv3_3(fy), qbspline_deriv3_4(fy), qbspline_deriv3_5(fy)};
        float d3bz[6] = {qbspline_deriv3_0(fz), qbspline_deriv3_1(fz), qbspline_deriv3_2(fz),
                         qbspline_deriv3_3(fz), qbspline_deriv3_4(fz), qbspline_deriv3_5(fz)};

        for (int i = 0; i < 6; i++) {
            int gx = min(max(ix - 2 + i, 0), gridCounts[0] - 1);
            for (int j = 0; j < 6; j++) {
                int gy = min(max(iy - 2 + j, 0), gridCounts[1] - 1);
                for (int k = 0; k < 6; k++) {
                    int gz = min(max(iz - 2 + k, 0), gridCounts[2] - 1);
                    int gridIdx = gx * nyz + gy * gridCounts[2] + gz;
                    float val = gridValues[gridIdx];

                    if (invPowerMode == 1) {
                        float invN = 1.0f / invPower;
                        if (fabsf(val) >= 1e-10f) {
                            val = (val >= 0.0f ? 1.0f : -1.0f) * powf(fabsf(val), invN);
                        } else {
                            val = 0.0f;
                        }
                    }

                    interpolated += bx[i] * by[j] * bz[k] * val;
                    dx  += dbx[i] *  by[j] *  bz[k] * val;
                    dy  +=  bx[i] * dby[j] *  bz[k] * val;
                    dz  +=  bx[i] *  by[j] * dbz[k] * val;

                    d2xx += d2bx[i] *   by[j] *   bz[k] * val;
                    d2yy +=   bx[i] * d2by[j] *   bz[k] * val;
                    d2zz +=   bx[i] *   by[j] * d2bz[k] * val;
                    d2xy +=  dbx[i] *  dby[j] *   bz[k] * val;
                    d2xz +=  dbx[i] *   by[j] *  dbz[k] * val;
                    d2yz +=   bx[i] *  dby[j] *  dbz[k] * val;

                    d3xxx += d3bx[i] *   by[j] *   bz[k] * val;
                    d3yyy +=   bx[i] * d3by[j] *   bz[k] * val;
                    d3zzz +=   bx[i] *   by[j] * d3bz[k] * val;
                    d3xxy += d2bx[i] *  dby[j] *   bz[k] * val;
                    d3xxz += d2bx[i] *   by[j] *  dbz[k] * val;
                    d3xyy +=  dbx[i] * d2by[j] *   bz[k] * val;
                    d3xzz +=  dbx[i] *   by[j] * d2bz[k] * val;
                    d3yyz +=   bx[i] * d2by[j] *  dbz[k] * val;
                    d3yzz +=   bx[i] *  dby[j] * d2bz[k] * val;
                    d3xyz +=  dbx[i] *  dby[j] *  dbz[k] * val;
                }
            }
        }

        // Chain rule transformations operate on the raw interpolated derivatives.
        // Each transformation is applied to all derivative levels simultaneously,
        // using the PRE-transform values at each level.

        // Undo transforms in reverse order: arcsinh first, then inv_power.

        // 1. Arcsinh: V = scale * sinh(g)
        if (arcsinhScale > 0.0f) {
            float g = interpolated;
            float sinhG = sinhf(g);
            float coshG = coshf(g);
            float s = arcsinhScale;

            // Third derivatives (Faà di Bruno for sinh(g)):
            //   d³V/dxi dxj dxk = s * [cosh(g)*gi*gj*gk + sinh(g)*(gi*gjk + gj*gik + gk*gij) + cosh(g)*gijk]
            float new_d3xxx = s*(coshG*dx*dx*dx + 3.0f*sinhG*dx*d2xx + coshG*d3xxx);
            float new_d3yyy = s*(coshG*dy*dy*dy + 3.0f*sinhG*dy*d2yy + coshG*d3yyy);
            float new_d3zzz = s*(coshG*dz*dz*dz + 3.0f*sinhG*dz*d2zz + coshG*d3zzz);
            float new_d3xxy = s*(coshG*dx*dx*dy + sinhG*(2.0f*dx*d2xy + dy*d2xx) + coshG*d3xxy);
            float new_d3xxz = s*(coshG*dx*dx*dz + sinhG*(2.0f*dx*d2xz + dz*d2xx) + coshG*d3xxz);
            float new_d3xyy = s*(coshG*dx*dy*dy + sinhG*(dx*d2yy + 2.0f*dy*d2xy) + coshG*d3xyy);
            float new_d3xzz = s*(coshG*dx*dz*dz + sinhG*(dx*d2zz + 2.0f*dz*d2xz) + coshG*d3xzz);
            float new_d3yyz = s*(coshG*dy*dy*dz + sinhG*(2.0f*dy*d2yz + dz*d2yy) + coshG*d3yyz);
            float new_d3yzz = s*(coshG*dy*dz*dz + sinhG*(dy*d2zz + 2.0f*dz*d2yz) + coshG*d3yzz);
            float new_d3xyz = s*(coshG*dx*dy*dz + sinhG*(dx*d2yz + dy*d2xz + dz*d2xy) + coshG*d3xyz);

            // Second derivatives
            float new_d2xx = s*(sinhG*dx*dx + coshG*d2xx);
            float new_d2yy = s*(sinhG*dy*dy + coshG*d2yy);
            float new_d2zz = s*(sinhG*dz*dz + coshG*d2zz);
            float new_d2xy = s*(sinhG*dx*dy + coshG*d2xy);
            float new_d2xz = s*(sinhG*dx*dz + coshG*d2xz);
            float new_d2yz = s*(sinhG*dy*dz + coshG*d2yz);

            // First derivatives and value
            interpolated = s*sinhG;
            dx = s*coshG*dx; dy = s*coshG*dy; dz = s*coshG*dz;

            d2xx = new_d2xx; d2yy = new_d2yy; d2zz = new_d2zz;
            d2xy = new_d2xy; d2xz = new_d2xz; d2yz = new_d2yz;
            d3xxx = new_d3xxx; d3yyy = new_d3yyy; d3zzz = new_d3zzz;
            d3xxy = new_d3xxy; d3xxz = new_d3xxz; d3xyy = new_d3xyy;
            d3xzz = new_d3xzz; d3yyz = new_d3yyz; d3yzz = new_d3yzz;
            d3xyz = new_d3xyz;
        }

        // 2. InvPower: V = sign(U)|U|^p, convert from smoothed to actual space
        if ((invPowerMode == 1 || invPowerMode == 2) && fabsf(invPower) > 1e-10f) {
            float p = invPower;
            float absU = fabsf(interpolated);
            if (absU < 1e-10f) absU = 1e-10f;
            float absU_pm1 = powf(absU, p - 1.0f);
            float absU_pm2 = powf(absU, p - 2.0f);

            // Third derivatives (uses original 1st, 2nd, 3rd)
            applyThirdDerivChainRule(interpolated, dx, dy, dz,
                                    d2xx, d2yy, d2zz, d2xy, d2xz, d2yz,
                                    d3xxx, d3yyy, d3zzz, d3xxy, d3xxz,
                                    d3xyy, d3xzz, d3yyz, d3yzz, d3xyz, p);

            // Second derivatives (uses original 1st, overwrites 2nd)
            float f2_1 = p * (p - 1.0f) * absU_pm2;
            float f2_2 = p * absU_pm1;
            float new_d2xx = f2_1*dx*dx + f2_2*d2xx;
            float new_d2yy = f2_1*dy*dy + f2_2*d2yy;
            float new_d2zz = f2_1*dz*dz + f2_2*d2zz;
            float new_d2xy = f2_1*dx*dy + f2_2*d2xy;
            float new_d2xz = f2_1*dx*dz + f2_2*d2xz;
            float new_d2yz = f2_1*dy*dz + f2_2*d2yz;

            // First derivatives
            float f1 = p * absU_pm1;
            dx = f1 * dx;
            dy = f1 * dy;
            dz = f1 * dz;

            d2xx = new_d2xx; d2yy = new_d2yy; d2zz = new_d2zz;
            d2xy = new_d2xy; d2xz = new_d2xz; d2yz = new_d2yz;
        }

        // 3. Convert from grid-cell coordinates to physical coordinates
        float inv_dx = 1.0f / gridSpacing[0];
        float inv_dy = 1.0f / gridSpacing[1];
        float inv_dz = 1.0f / gridSpacing[2];

        d3xxx *= inv_dx * inv_dx * inv_dx;
        d3yyy *= inv_dy * inv_dy * inv_dy;
        d3zzz *= inv_dz * inv_dz * inv_dz;
        d3xxy *= inv_dx * inv_dx * inv_dy;
        d3xxz *= inv_dx * inv_dx * inv_dz;
        d3xyy *= inv_dx * inv_dy * inv_dy;
        d3xzz *= inv_dx * inv_dz * inv_dz;
        d3yyz *= inv_dy * inv_dy * inv_dz;
        d3yzz *= inv_dy * inv_dz * inv_dz;
        d3xyz *= inv_dx * inv_dy * inv_dz;

        // Apply per-atom scaling factor
        d3xxx *= scalingFactor; d3yyy *= scalingFactor; d3zzz *= scalingFactor;
        d3xxy *= scalingFactor; d3xxz *= scalingFactor; d3xyy *= scalingFactor;
        d3xzz *= scalingFactor; d3yyz *= scalingFactor; d3yzz *= scalingFactor;
        d3xyz *= scalingFactor;
    }

    // Store third derivative components (10 per atom)
    int offset = index * 10;
    thirdDerivBuffer[offset + 0] = d3xxx;
    thirdDerivBuffer[offset + 1] = d3yyy;
    thirdDerivBuffer[offset + 2] = d3zzz;
    thirdDerivBuffer[offset + 3] = d3xxy;
    thirdDerivBuffer[offset + 4] = d3xxz;
    thirdDerivBuffer[offset + 5] = d3xyy;
    thirdDerivBuffer[offset + 6] = d3xzz;
    thirdDerivBuffer[offset + 7] = d3yyz;
    thirdDerivBuffer[offset + 8] = d3yzz;
    thirdDerivBuffer[offset + 9] = d3xyz;
}
