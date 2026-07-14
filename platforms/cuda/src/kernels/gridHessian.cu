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
 * Both support inv_power, arcsinh, and runtime cap chain rule transformations.
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
template<typename T>
__device__ inline void applyHessianChainRule(
    T U,
    T dUdx, T dUdy, T dUdz,
    T& d2xx, T& d2yy, T& d2zz,
    T& d2xy, T& d2xz, T& d2yz,
    T p
) {
    T absU = fabs(U);
    if (absU < 1e-10f) absU = 1e-10f;  // Clamp to avoid divide by zero

    // Precompute powers
    T absU_pm1 = pow(absU, p - 1.0f);
    T absU_pm2 = pow(absU, p - 2.0f);

    // Chain rule coefficients
    T f2_1 = p * (p - 1.0f) * absU_pm2;  // For (dU/dx)² terms
    T f2_2 = p * absU_pm1;                // For d²U/dx² terms

    // Apply chain rule to each Hessian component
    T new_d2xx = f2_1 * dUdx * dUdx + f2_2 * d2xx;
    T new_d2yy = f2_1 * dUdy * dUdy + f2_2 * d2yy;
    T new_d2zz = f2_1 * dUdz * dUdz + f2_2 * d2zz;
    T new_d2xy = f2_1 * dUdx * dUdy + f2_2 * d2xy;
    T new_d2xz = f2_1 * dUdx * dUdz + f2_2 * d2xz;
    T new_d2yz = f2_1 * dUdy * dUdz + f2_2 * d2yz;

    d2xx = new_d2xx;
    d2yy = new_d2yy;
    d2zz = new_d2zz;
    d2xy = new_d2xy;
    d2xz = new_d2xz;
    d2yz = new_d2yz;
}

/**
 * Apply runtime tanh cap chain rule to Hessian.
 *
 * The cap function is f(v) = C * tanh(v/C), applied after inv_power and arcsinh.
 *
 * First derivative: f'(v) = sech²(v/C) = 1 - tanh²(v/C)
 * Second derivative: f''(v) = -2 * tanh(v/C) * sech²(v/C) / C
 *
 * For the Hessian:
 *   d^2f/dxi dxj = f'(v) * d^2v/dxi dxj + f''(v) * dv/dxi * dv/dxj
 *
 * @param v         Post-transform value (after arcsinh + inv_power, before cap)
 * @param dvdx/y/z  Post-transform first derivatives (before cap)
 * @param d2xx/...   Second derivatives (modified in place)
 * @param cap       Runtime cap value C (must be > 0)
 */
template<typename T>
__device__ inline void applyRuntimeCapHessianChainRule(
    T v,
    T dvdx, T dvdy, T dvdz,
    T& d2xx, T& d2yy, T& d2zz,
    T& d2xy, T& d2xz, T& d2yz,
    T cap
) {
    T t = tanh(v / cap);
    T sech2 = 1.0f - t * t;
    T fPrime = sech2;
    T fDoublePrime = -2.0f * t * sech2 / cap;

    T new_d2xx = fPrime * d2xx + fDoublePrime * dvdx * dvdx;
    T new_d2yy = fPrime * d2yy + fDoublePrime * dvdy * dvdy;
    T new_d2zz = fPrime * d2zz + fDoublePrime * dvdz * dvdz;
    T new_d2xy = fPrime * d2xy + fDoublePrime * dvdx * dvdy;
    T new_d2xz = fPrime * d2xz + fDoublePrime * dvdx * dvdz;
    T new_d2yz = fPrime * d2yz + fDoublePrime * dvdy * dvdz;

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
    const real4* __restrict__ posq,
    mixed* __restrict__ hessianBuffer,  // 6 components per atom
    const int* __restrict__ gridCounts,
    const float* __restrict__ gridSpacing,
    const GRID_VALUES_TYPE* __restrict__ gridValues,
    const float* __restrict__ scalingFactors,
    const float invPower,
    const int invPowerMode,  // 0=NONE, 1=RUNTIME, 2=STORED
    const int interpolationMethod,
    const float originX,
    const float originY,
    const float originZ,
    const GRID_STORAGE_TYPE* __restrict__ gridDerivatives,
    const int numAtoms,
    const int* __restrict__ particleIndices,
    const float arcsinhScale,
    const float* __restrict__ groupScalingFactors,  // Per-group alchemical scaling (null = no per-group scaling)
    const int* __restrict__ particleToGroupMap,      // Maps particle index to group index (null = no mapping)
    const int numGroups,                             // Number of particle groups
    const float runtimeCap,                          // Global runtime cap (0=disabled)
    const float* __restrict__ groupRuntimeCaps,      // Per-group runtime caps (null = use global, 0 = use global)
    const float effectiveMinX, const float effectiveMinY, const float effectiveMinZ,
    const float effectiveMaxX, const float effectiveMaxY, const float effectiveMaxZ)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= numAtoms)
        return;

    // Get actual particle index
    const unsigned int particleIndex = (particleIndices != 0) ? particleIndices[index] : index;

    // Load position and scaling factor (with per-group alchemical scaling)
    real4 posOrig = posq[particleIndex];
    float groupScale = 1.0f;
    if (groupScalingFactors != 0 && particleToGroupMap != 0) {
        int groupIdx = particleToGroupMap[particleIndex];
        if (groupIdx >= 0 && groupIdx < numGroups) {
            groupScale = groupScalingFactors[groupIdx];
        }
    }
    float scalingFactor = groupScale * scalingFactors[particleIndex];

    // Resolve effective runtime cap: per-group if available, else global
    real effectiveCap = runtimeCap;
    if (groupRuntimeCaps != 0 && particleToGroupMap != 0) {
        int gIdx = particleToGroupMap[particleIndex];
        if (gIdx >= 0 && gIdx < numGroups && groupRuntimeCaps[gIdx] > 0.0f) {
            effectiveCap = groupRuntimeCaps[gIdx];
        }
    }

    // Transform to grid coordinates
    real3 pos;
    pos.x = posOrig.x - originX;
    pos.y = posOrig.y - originY;
    pos.z = posOrig.z - originZ;

    // Accumulators stay real for the downstream chain-rule helpers (the triquintic
    // assembly and eval temporaries are double).
    real interpolated = 0.0f;
    real dx = 0.0f, dy = 0.0f, dz = 0.0f;
    real d2xx = 0.0f, d2yy = 0.0f, d2zz = 0.0f;
    real d2xy = 0.0f, d2xz = 0.0f, d2yz = 0.0f;

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
        real fx = (pos.x / gridSpacing[0]) - ix;
        real fy = (pos.y / gridSpacing[1]) - iy;
        real fz = (pos.z / gridSpacing[2]) - iz;

        fx = min(max(fx, 0.0f), 1.0f);
        fy = min(max(fy, 0.0f), 1.0f);
        fz = min(max(fz, 0.0f), 1.0f);

        int nyz = gridCounts[1] * gridCounts[2];

        if (interpolationMethod == 3 && gridDerivatives != 0) {
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
            double X[216];
            if (invPowerMode == 1) {
                // RUNTIME mode: transform corners from G space to S space BEFORE interpolation
                // p = 1/invPower transforms G → S = |G|^p
                real p = 1.0f / invPower;
                for (int c = 0; c < 8; c++) {
                    int point_idx = corners[c][0] * nyz + corners[c][1] * gridCounts[2] + corners[c][2];
                    real G_derivs[27], S_derivs[27];
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

            // Assemble + evaluate (value, gradient, Hessian) via the shared
            // double-precision helpers; truncate into the real accumulators that
            // the downstream chain-rule helpers expect (consistency preserved --
            // both cells run identical double->real ops).
            TriquinticAccum a[216];
            triquinticAssemble(X, a);
            TriquinticAccum v, gx, gy, gz, hxx, hyy, hzz, hxy, hxz, hyz;
            triquinticEvalVGH(a, fx, fy, fz, &v, &gx, &gy, &gz,
                              &hxx, &hyy, &hzz, &hxy, &hxz, &hyz);
            interpolated = v; dx = gx; dy = gy; dz = gz;
            d2xx = hxx; d2yy = hyy; d2zz = hzz; d2xy = hxy; d2xz = hxz; d2yz = hyz;

            // Undo transforms in reverse order of application during generation.
            // Generation order: inv_power → arcsinh → blur → prefilter.
            // Evaluation undo order: arcsinh first, then inv_power.

            // Arcsinh chain rule: V = scale*sinh(g), d²V/dxi dxj = scale*[sinh(g)*dg_i*dg_j + cosh(g)*d²g_ij]
            if (arcsinhScale > 0.0f) {
                real g = interpolated;
                real sinhG = sinh(g);
                real coshG = coshf(g);

                real new_d2xx = arcsinhScale * (sinhG * dx * dx + coshG * d2xx);
                real new_d2yy = arcsinhScale * (sinhG * dy * dy + coshG * d2yy);
                real new_d2zz = arcsinhScale * (sinhG * dz * dz + coshG * d2zz);
                real new_d2xy = arcsinhScale * (sinhG * dx * dy + coshG * d2xy);
                real new_d2xz = arcsinhScale * (sinhG * dx * dz + coshG * d2xz);
                real new_d2yz = arcsinhScale * (sinhG * dy * dz + coshG * d2yz);

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
            if ((invPowerMode == 1 || invPowerMode == 2) && fabs(invPower) > 1e-10f) {
                real absU = fabs(interpolated);
                if (absU > 1e-10f) {
                    real n = invPower;

                    real absU_nm1 = pow(absU, n - 1.0f);
                    real absU_nm2 = pow(absU, n - 2.0f);

                    real f2_1 = n * (n - 1.0f) * absU_nm2;
                    real f2_2 = n * absU_nm1;

                    real new_d2xx = f2_1 * dx * dx + f2_2 * d2xx;
                    real new_d2yy = f2_1 * dy * dy + f2_2 * d2yy;
                    real new_d2zz = f2_1 * dz * dz + f2_2 * d2zz;
                    real new_d2xy = f2_1 * dx * dy + f2_2 * d2xy;
                    real new_d2xz = f2_1 * dx * dz + f2_2 * d2xz;
                    real new_d2yz = f2_1 * dy * dz + f2_2 * d2yz;

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
            real inv_dx = 1.0f / gridSpacing[0];
            real inv_dy = 1.0f / gridSpacing[1];
            real inv_dz = 1.0f / gridSpacing[2];
            real inv_dx2 = inv_dx * inv_dx;
            real inv_dy2 = inv_dy * inv_dy;
            real inv_dz2 = inv_dz * inv_dz;
            real inv_dxdy = inv_dx * inv_dy;
            real inv_dxdz = inv_dx * inv_dz;
            real inv_dydz = inv_dy * inv_dz;

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
            real bx[4] = {bspline_basis0(fx), bspline_basis1(fx), bspline_basis2(fx), bspline_basis3(fx)};
            real by[4] = {bspline_basis0(fy), bspline_basis1(fy), bspline_basis2(fy), bspline_basis3(fy)};
            real bz[4] = {bspline_basis0(fz), bspline_basis1(fz), bspline_basis2(fz), bspline_basis3(fz)};

            real dbx[4] = {bspline_deriv0(fx), bspline_deriv1(fx), bspline_deriv2(fx), bspline_deriv3(fx)};
            real dby[4] = {bspline_deriv0(fy), bspline_deriv1(fy), bspline_deriv2(fy), bspline_deriv3(fy)};
            real dbz[4] = {bspline_deriv0(fz), bspline_deriv1(fz), bspline_deriv2(fz), bspline_deriv3(fz)};

            // Second derivatives of B-spline basis functions
            real d2bx[4] = {bspline_deriv2_0(fx), bspline_deriv2_1(fx), bspline_deriv2_2(fx), bspline_deriv2_3(fx)};
            real d2by[4] = {bspline_deriv2_0(fy), bspline_deriv2_1(fy), bspline_deriv2_2(fy), bspline_deriv2_3(fy)};
            real d2bz[4] = {bspline_deriv2_0(fz), bspline_deriv2_1(fz), bspline_deriv2_2(fz), bspline_deriv2_3(fz)};

            for (int i = 0; i < 4; i++) {
                int gx = min(max(ix - 1 + i, 0), gridCounts[0] - 1);
                for (int j = 0; j < 4; j++) {
                    int gy = min(max(iy - 1 + j, 0), gridCounts[1] - 1);
                    for (int k = 0; k < 4; k++) {
                        int gz = min(max(iz - 1 + k, 0), gridCounts[2] - 1);
                        int gridIdx = gx * nyz + gy * gridCounts[2] + gz;
                        real val = gridValues[gridIdx];

                        // Apply RUNTIME inv_power transformation before interpolation
                        // (matching force kernel behavior for consistency)
                        if (invPowerMode == 1) {
                            real invN = 1.0f / invPower;
                            if (fabs(val) >= 1e-10f) {
                                val = (val >= 0.0f ? 1.0f : -1.0f) * pow(fabs(val), invN);
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
                real g = interpolated;
                real sinhG = sinh(g);
                real coshG = coshf(g);

                real new_d2xx = arcsinhScale * (sinhG * dx * dx + coshG * d2xx);
                real new_d2yy = arcsinhScale * (sinhG * dy * dy + coshG * d2yy);
                real new_d2zz = arcsinhScale * (sinhG * dz * dz + coshG * d2zz);
                real new_d2xy = arcsinhScale * (sinhG * dx * dy + coshG * d2xy);
                real new_d2xz = arcsinhScale * (sinhG * dx * dz + coshG * d2xz);
                real new_d2yz = arcsinhScale * (sinhG * dy * dz + coshG * d2yz);

                interpolated = arcsinhScale * sinhG;
                dx = arcsinhScale * coshG * dx;
                dy = arcsinhScale * coshG * dy;
                dz = arcsinhScale * coshG * dz;

                d2xx = new_d2xx; d2yy = new_d2yy; d2zz = new_d2zz;
                d2xy = new_d2xy; d2xz = new_d2xz; d2yz = new_d2yz;
            }

            if ((invPowerMode == 1 || invPowerMode == 2) && fabs(invPower) > 1e-10f) {
                real absU = fabs(interpolated);
                if (absU > 1e-10f) {
                    real n = invPower;
                    real absU_nm1 = pow(absU, n - 1.0f);
                    real absU_nm2 = pow(absU, n - 2.0f);
                    real f2_1 = n * (n - 1.0f) * absU_nm2;
                    real f2_2 = n * absU_nm1;

                    real new_d2xx = f2_1 * dx * dx + f2_2 * d2xx;
                    real new_d2yy = f2_1 * dy * dy + f2_2 * d2yy;
                    real new_d2zz = f2_1 * dz * dz + f2_2 * d2zz;
                    real new_d2xy = f2_1 * dx * dy + f2_2 * d2xy;
                    real new_d2xz = f2_1 * dx * dz + f2_2 * d2xz;
                    real new_d2yz = f2_1 * dy * dz + f2_2 * d2yz;

                    dx *= f2_2;
                    dy *= f2_2;
                    dz *= f2_2;

                    // Update interpolated to post-inv_power value
                    real sign = (interpolated >= 0.0f) ? 1.0f : -1.0f;
                    interpolated = sign * pow(absU, n);

                    d2xx = new_d2xx;
                    d2yy = new_d2yy;
                    d2zz = new_d2zz;
                    d2xy = new_d2xy;
                    d2xz = new_d2xz;
                    d2yz = new_d2yz;
                }
            }

            // Apply runtime tanh cap chain rule (must match gridForce.cu cap order)
            if (effectiveCap > 0.0f) {
                applyRuntimeCapHessianChainRule(interpolated, dx, dy, dz,
                    d2xx, d2yy, d2zz, d2xy, d2xz, d2yz, effectiveCap);
                real t = tanh(interpolated / effectiveCap);
                real gradFactor = 1.0f - t * t;
                interpolated = effectiveCap * t;
                dx *= gradFactor;
                dy *= gradFactor;
                dz *= gradFactor;
            }

            // Convert to physical coordinates
            real inv_dx = 1.0f / gridSpacing[0];
            real inv_dy = 1.0f / gridSpacing[1];
            real inv_dz = 1.0f / gridSpacing[2];
            real inv_dx2 = inv_dx * inv_dx;
            real inv_dy2 = inv_dy * inv_dy;
            real inv_dz2 = inv_dz * inv_dz;
            real inv_dxdy = inv_dx * inv_dy;
            real inv_dxdz = inv_dx * inv_dz;
            real inv_dydz = inv_dy * inv_dz;

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

            real bx[6] = {qbspline_basis0(fx), qbspline_basis1(fx), qbspline_basis2(fx),
                           qbspline_basis3(fx), qbspline_basis4(fx), qbspline_basis5(fx)};
            real by[6] = {qbspline_basis0(fy), qbspline_basis1(fy), qbspline_basis2(fy),
                           qbspline_basis3(fy), qbspline_basis4(fy), qbspline_basis5(fy)};
            real bz[6] = {qbspline_basis0(fz), qbspline_basis1(fz), qbspline_basis2(fz),
                           qbspline_basis3(fz), qbspline_basis4(fz), qbspline_basis5(fz)};

            real dbx[6] = {qbspline_deriv0(fx), qbspline_deriv1(fx), qbspline_deriv2(fx),
                            qbspline_deriv3(fx), qbspline_deriv4(fx), qbspline_deriv5(fx)};
            real dby[6] = {qbspline_deriv0(fy), qbspline_deriv1(fy), qbspline_deriv2(fy),
                            qbspline_deriv3(fy), qbspline_deriv4(fy), qbspline_deriv5(fy)};
            real dbz[6] = {qbspline_deriv0(fz), qbspline_deriv1(fz), qbspline_deriv2(fz),
                            qbspline_deriv3(fz), qbspline_deriv4(fz), qbspline_deriv5(fz)};

            real d2bx[6] = {qbspline_deriv2_0(fx), qbspline_deriv2_1(fx), qbspline_deriv2_2(fx),
                             qbspline_deriv2_3(fx), qbspline_deriv2_4(fx), qbspline_deriv2_5(fx)};
            real d2by[6] = {qbspline_deriv2_0(fy), qbspline_deriv2_1(fy), qbspline_deriv2_2(fy),
                             qbspline_deriv2_3(fy), qbspline_deriv2_4(fy), qbspline_deriv2_5(fy)};
            real d2bz[6] = {qbspline_deriv2_0(fz), qbspline_deriv2_1(fz), qbspline_deriv2_2(fz),
                             qbspline_deriv2_3(fz), qbspline_deriv2_4(fz), qbspline_deriv2_5(fz)};

            for (int i = 0; i < 6; i++) {
                int gx = min(max(ix - 2 + i, 0), gridCounts[0] - 1);
                for (int j = 0; j < 6; j++) {
                    int gy = min(max(iy - 2 + j, 0), gridCounts[1] - 1);
                    for (int k = 0; k < 6; k++) {
                        int gz = min(max(iz - 2 + k, 0), gridCounts[2] - 1);
                        int gridIdx = gx * nyz + gy * gridCounts[2] + gz;
                        real val = gridValues[gridIdx];

                        if (invPowerMode == 1) {
                            real invN = 1.0f / invPower;
                            if (fabs(val) >= 1e-10f) {
                                val = (val >= 0.0f ? 1.0f : -1.0f) * pow(fabs(val), invN);
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
                real g = interpolated;
                real sinhG = sinh(g);
                real coshG = coshf(g);

                real new_d2xx = arcsinhScale * (sinhG * dx * dx + coshG * d2xx);
                real new_d2yy = arcsinhScale * (sinhG * dy * dy + coshG * d2yy);
                real new_d2zz = arcsinhScale * (sinhG * dz * dz + coshG * d2zz);
                real new_d2xy = arcsinhScale * (sinhG * dx * dy + coshG * d2xy);
                real new_d2xz = arcsinhScale * (sinhG * dx * dz + coshG * d2xz);
                real new_d2yz = arcsinhScale * (sinhG * dy * dz + coshG * d2yz);

                interpolated = arcsinhScale * sinhG;
                dx = arcsinhScale * coshG * dx;
                dy = arcsinhScale * coshG * dy;
                dz = arcsinhScale * coshG * dz;

                d2xx = new_d2xx; d2yy = new_d2yy; d2zz = new_d2zz;
                d2xy = new_d2xy; d2xz = new_d2xz; d2yz = new_d2yz;
            }

            if ((invPowerMode == 1 || invPowerMode == 2) && fabs(invPower) > 1e-10f) {
                applyHessianChainRule(interpolated, dx, dy, dz,
                                     d2xx, d2yy, d2zz, d2xy, d2xz, d2yz, (real)invPower);
                // Update value and first derivatives to post-inv_power
                real absU = fabs(interpolated);
                if (absU > 1e-10f) {
                    real sign = (interpolated >= 0.0f) ? 1.0f : -1.0f;
                    real pf = invPower * pow(absU, (real)invPower - (real)1.0f);
                    interpolated = sign * pow(absU, (real)invPower);
                    dx *= pf;
                    dy *= pf;
                    dz *= pf;
                }
            }

            // Apply runtime tanh cap chain rule (must match gridForce.cu cap order)
            if (effectiveCap > 0.0f) {
                applyRuntimeCapHessianChainRule(interpolated, dx, dy, dz,
                    d2xx, d2yy, d2zz, d2xy, d2xz, d2yz, effectiveCap);
                // Update first derivatives for any downstream use
                real t = tanh(interpolated / effectiveCap);
                real gradFactor = 1.0f - t * t;
                interpolated = effectiveCap * t;
                dx *= gradFactor;
                dy *= gradFactor;
                dz *= gradFactor;
            }

            // Convert to physical coordinates
            real inv_dx = 1.0f / gridSpacing[0];
            real inv_dy = 1.0f / gridSpacing[1];
            real inv_dz = 1.0f / gridSpacing[2];

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
template<typename T>
__device__ inline void applyThirdDerivChainRule(
    T U,
    T dUdx, T dUdy, T dUdz,
    T d2Uxx, T d2Uyy, T d2Uzz,
    T d2Uxy, T d2Uxz, T d2Uyz,
    T& d3xxx, T& d3yyy, T& d3zzz,
    T& d3xxy, T& d3xxz, T& d3xyy,
    T& d3xzz, T& d3yyz, T& d3yzz,
    T& d3xyz,
    T p
) {
    T absU = fabs(U);
    if (absU < 1e-10f) absU = 1e-10f;

    T absU_pm1 = pow(absU, p - 1.0f);
    T absU_pm2 = pow(absU, p - 2.0f);
    T absU_pm3 = pow(absU, p - 3.0f);

    T f3_1 = p * (p - 1.0f) * (p - 2.0f) * absU_pm3;
    T f3_2 = p * (p - 1.0f) * absU_pm2;
    T f3_3 = p * absU_pm1;

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
    const real4* __restrict__ posq,
    mixed* __restrict__ thirdDerivBuffer,  // 10 components per atom
    const int* __restrict__ gridCounts,
    const float* __restrict__ gridSpacing,
    const GRID_VALUES_TYPE* __restrict__ gridValues,
    const float* __restrict__ scalingFactors,
    const float invPower,
    const int invPowerMode,
    const int interpolationMethod,
    const float originX,
    const float originY,
    const float originZ,
    const GRID_STORAGE_TYPE* __restrict__ gridDerivatives,
    const int numAtoms,
    const int* __restrict__ particleIndices,
    const float arcsinhScale)
{
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= numAtoms)
        return;

    const unsigned int particleIndex = (particleIndices != 0) ? particleIndices[index] : index;

    real4 posOrig = posq[particleIndex];
    float scalingFactor = scalingFactors[particleIndex];

    real3 pos;
    pos.x = posOrig.x - originX;
    pos.y = posOrig.y - originY;
    pos.z = posOrig.z - originZ;

    // All derivatives initialized to zero
    real interpolated = 0.0f;
    real dx = 0.0f, dy = 0.0f, dz = 0.0f;
    real d2xx = 0.0f, d2yy = 0.0f, d2zz = 0.0f;
    real d2xy = 0.0f, d2xz = 0.0f, d2yz = 0.0f;
    real d3xxx = 0.0f, d3yyy = 0.0f, d3zzz = 0.0f;
    real d3xxy = 0.0f, d3xxz = 0.0f, d3xyy = 0.0f;
    real d3xzz = 0.0f, d3yyz = 0.0f, d3yzz = 0.0f;
    real d3xyz = 0.0f;

    real3 gridCorner;
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

        real fx = (pos.x / gridSpacing[0]) - ix;
        real fy = (pos.y / gridSpacing[1]) - iy;
        real fz = (pos.z / gridSpacing[2]) - iz;

        fx = min(max(fx, 0.0f), 1.0f);
        fy = min(max(fy, 0.0f), 1.0f);
        fz = min(max(fz, 0.0f), 1.0f);

        int nyz = gridCounts[1] * gridCounts[2];

        // Basis functions: value, 1st, 2nd, and 3rd derivatives
        real bx[6] = {qbspline_basis0(fx), qbspline_basis1(fx), qbspline_basis2(fx),
                       qbspline_basis3(fx), qbspline_basis4(fx), qbspline_basis5(fx)};
        real by[6] = {qbspline_basis0(fy), qbspline_basis1(fy), qbspline_basis2(fy),
                       qbspline_basis3(fy), qbspline_basis4(fy), qbspline_basis5(fy)};
        real bz[6] = {qbspline_basis0(fz), qbspline_basis1(fz), qbspline_basis2(fz),
                       qbspline_basis3(fz), qbspline_basis4(fz), qbspline_basis5(fz)};

        real dbx[6] = {qbspline_deriv0(fx), qbspline_deriv1(fx), qbspline_deriv2(fx),
                        qbspline_deriv3(fx), qbspline_deriv4(fx), qbspline_deriv5(fx)};
        real dby[6] = {qbspline_deriv0(fy), qbspline_deriv1(fy), qbspline_deriv2(fy),
                        qbspline_deriv3(fy), qbspline_deriv4(fy), qbspline_deriv5(fy)};
        real dbz[6] = {qbspline_deriv0(fz), qbspline_deriv1(fz), qbspline_deriv2(fz),
                        qbspline_deriv3(fz), qbspline_deriv4(fz), qbspline_deriv5(fz)};

        real d2bx[6] = {qbspline_deriv2_0(fx), qbspline_deriv2_1(fx), qbspline_deriv2_2(fx),
                         qbspline_deriv2_3(fx), qbspline_deriv2_4(fx), qbspline_deriv2_5(fx)};
        real d2by[6] = {qbspline_deriv2_0(fy), qbspline_deriv2_1(fy), qbspline_deriv2_2(fy),
                         qbspline_deriv2_3(fy), qbspline_deriv2_4(fy), qbspline_deriv2_5(fy)};
        real d2bz[6] = {qbspline_deriv2_0(fz), qbspline_deriv2_1(fz), qbspline_deriv2_2(fz),
                         qbspline_deriv2_3(fz), qbspline_deriv2_4(fz), qbspline_deriv2_5(fz)};

        real d3bx[6] = {qbspline_deriv3_0(fx), qbspline_deriv3_1(fx), qbspline_deriv3_2(fx),
                         qbspline_deriv3_3(fx), qbspline_deriv3_4(fx), qbspline_deriv3_5(fx)};
        real d3by[6] = {qbspline_deriv3_0(fy), qbspline_deriv3_1(fy), qbspline_deriv3_2(fy),
                         qbspline_deriv3_3(fy), qbspline_deriv3_4(fy), qbspline_deriv3_5(fy)};
        real d3bz[6] = {qbspline_deriv3_0(fz), qbspline_deriv3_1(fz), qbspline_deriv3_2(fz),
                         qbspline_deriv3_3(fz), qbspline_deriv3_4(fz), qbspline_deriv3_5(fz)};

        for (int i = 0; i < 6; i++) {
            int gx = min(max(ix - 2 + i, 0), gridCounts[0] - 1);
            for (int j = 0; j < 6; j++) {
                int gy = min(max(iy - 2 + j, 0), gridCounts[1] - 1);
                for (int k = 0; k < 6; k++) {
                    int gz = min(max(iz - 2 + k, 0), gridCounts[2] - 1);
                    int gridIdx = gx * nyz + gy * gridCounts[2] + gz;
                    real val = gridValues[gridIdx];

                    if (invPowerMode == 1) {
                        real invN = 1.0f / invPower;
                        if (fabs(val) >= 1e-10f) {
                            val = (val >= 0.0f ? 1.0f : -1.0f) * pow(fabs(val), invN);
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
            real g = interpolated;
            real sinhG = sinh(g);
            real coshG = coshf(g);
            real s = arcsinhScale;

            // Third derivatives (Faà di Bruno for sinh(g)):
            //   d³V/dxi dxj dxk = s * [cosh(g)*gi*gj*gk + sinh(g)*(gi*gjk + gj*gik + gk*gij) + cosh(g)*gijk]
            real new_d3xxx = s*(coshG*dx*dx*dx + 3.0f*sinhG*dx*d2xx + coshG*d3xxx);
            real new_d3yyy = s*(coshG*dy*dy*dy + 3.0f*sinhG*dy*d2yy + coshG*d3yyy);
            real new_d3zzz = s*(coshG*dz*dz*dz + 3.0f*sinhG*dz*d2zz + coshG*d3zzz);
            real new_d3xxy = s*(coshG*dx*dx*dy + sinhG*(2.0f*dx*d2xy + dy*d2xx) + coshG*d3xxy);
            real new_d3xxz = s*(coshG*dx*dx*dz + sinhG*(2.0f*dx*d2xz + dz*d2xx) + coshG*d3xxz);
            real new_d3xyy = s*(coshG*dx*dy*dy + sinhG*(dx*d2yy + 2.0f*dy*d2xy) + coshG*d3xyy);
            real new_d3xzz = s*(coshG*dx*dz*dz + sinhG*(dx*d2zz + 2.0f*dz*d2xz) + coshG*d3xzz);
            real new_d3yyz = s*(coshG*dy*dy*dz + sinhG*(2.0f*dy*d2yz + dz*d2yy) + coshG*d3yyz);
            real new_d3yzz = s*(coshG*dy*dz*dz + sinhG*(dy*d2zz + 2.0f*dz*d2yz) + coshG*d3yzz);
            real new_d3xyz = s*(coshG*dx*dy*dz + sinhG*(dx*d2yz + dy*d2xz + dz*d2xy) + coshG*d3xyz);

            // Second derivatives
            real new_d2xx = s*(sinhG*dx*dx + coshG*d2xx);
            real new_d2yy = s*(sinhG*dy*dy + coshG*d2yy);
            real new_d2zz = s*(sinhG*dz*dz + coshG*d2zz);
            real new_d2xy = s*(sinhG*dx*dy + coshG*d2xy);
            real new_d2xz = s*(sinhG*dx*dz + coshG*d2xz);
            real new_d2yz = s*(sinhG*dy*dz + coshG*d2yz);

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
        if ((invPowerMode == 1 || invPowerMode == 2) && fabs(invPower) > 1e-10f) {
            real p = invPower;
            real absU = fabs(interpolated);
            if (absU < 1e-10f) absU = 1e-10f;
            real absU_pm1 = pow(absU, p - 1.0f);
            real absU_pm2 = pow(absU, p - 2.0f);

            // Third derivatives (uses original 1st, 2nd, 3rd)
            applyThirdDerivChainRule(interpolated, dx, dy, dz,
                                    d2xx, d2yy, d2zz, d2xy, d2xz, d2yz,
                                    d3xxx, d3yyy, d3zzz, d3xxy, d3xxz,
                                    d3xyy, d3xzz, d3yyz, d3yzz, d3xyz, p);

            // Second derivatives (uses original 1st, overwrites 2nd)
            real f2_1 = p * (p - 1.0f) * absU_pm2;
            real f2_2 = p * absU_pm1;
            real new_d2xx = f2_1*dx*dx + f2_2*d2xx;
            real new_d2yy = f2_1*dy*dy + f2_2*d2yy;
            real new_d2zz = f2_1*dz*dz + f2_2*d2zz;
            real new_d2xy = f2_1*dx*dy + f2_2*d2xy;
            real new_d2xz = f2_1*dx*dz + f2_2*d2xz;
            real new_d2yz = f2_1*dy*dz + f2_2*d2yz;

            // First derivatives
            real f1 = p * absU_pm1;
            dx = f1 * dx;
            dy = f1 * dy;
            dz = f1 * dz;

            d2xx = new_d2xx; d2yy = new_d2yy; d2zz = new_d2zz;
            d2xy = new_d2xy; d2xz = new_d2xz; d2yz = new_d2yz;
        }

        // 3. Convert from grid-cell coordinates to physical coordinates
        real inv_dx = 1.0f / gridSpacing[0];
        real inv_dy = 1.0f / gridSpacing[1];
        real inv_dz = 1.0f / gridSpacing[2];

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
