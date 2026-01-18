/**
 * CUDA implementation of grid Hessian (second derivative) calculation.
 *
 * This kernel computes the 3x3 Hessian block for each atom from the grid potential.
 * The Hessian is the matrix of second partial derivatives:
 *   H = [d²V/dx², d²V/dxdy, d²V/dxdz]
 *       [d²V/dydx, d²V/dy², d²V/dydz]
 *       [d²V/dzdx, d²V/dzdy, d²V/dz²]
 *
 * Since mixed partials are equal (d²V/dxdy = d²V/dydx), we store only 6 unique components
 * per atom: dxx, dyy, dzz, dxy, dxz, dyz
 *
 * Supports inv_power transformation: when enabled, applies chain rule to convert
 * stored (transformed) derivatives to actual derivatives.
 *
 * Supported interpolation methods:
 *   - Triquintic (method 3): Analytical second derivatives from 5th order polynomial
 *   - B-spline (method 1): Analytical second derivatives from cubic B-spline basis
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
    const int* __restrict__ particleIndices)
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

            // Back-convert from smoothed space (S) to actual potential (G) for RUNTIME mode
            // After interpolation we have S, dS/dx, d²S/dx² in smoothed space
            // Convert to G = sign(S)*|S|^n where n = invPower (e.g., -6)
            //   dG/dx = n*|S|^(n-1) * dS/dx
            //   d²G/dx² = n*(n-1)*|S|^(n-2)*(dS/dx)² + n*|S|^(n-1)*d²S/dx²
            if (invPowerMode == 1 && fabsf(invPower) > 1e-10f) {
                float absU = fabsf(interpolated);
                if (absU > 1e-10f) {
                    float n = invPower;  // The power to convert U back to V

                    // Precompute powers
                    float absU_nm1 = powf(absU, n - 1.0f);
                    float absU_nm2 = powf(absU, n - 2.0f);

                    // Chain rule coefficients
                    float f2_1 = n * (n - 1.0f) * absU_nm2;  // For (dU/dx)² terms
                    float f2_2 = n * absU_nm1;               // For d²U/dx² terms

                    // Convert second derivatives (Hessian) using unit cell gradients
                    float new_d2xx = f2_1 * dx * dx + f2_2 * d2xx;
                    float new_d2yy = f2_1 * dy * dy + f2_2 * d2yy;
                    float new_d2zz = f2_1 * dz * dz + f2_2 * d2zz;
                    float new_d2xy = f2_1 * dx * dy + f2_2 * d2xy;
                    float new_d2xz = f2_1 * dx * dz + f2_2 * d2xz;
                    float new_d2yz = f2_1 * dy * dz + f2_2 * d2yz;

                    // Note: first derivatives are NOT used in output, but update for completeness
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

            // Back-convert from transformed space BEFORE converting to physical coords
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
