/**
 * CUDA implementation of grid force calculation.
 * Main kernel for computing forces on ligand atoms from interpolated grid values.
 *
 * This kernel supports multiple interpolation methods and inv_power transformations.
 * For the base case (no inv_power), it uses the shared GridInterpolation library.
 * For inv_power modes, it uses specialized inline code for correct chain rule handling.
 */

#define DEBUG_GRIDFORCE 0

#include "include/GridInterpolation.cuh"
#include "include/HermiteBasis.cuh"
#include "include/InvPowerChainRule.cuh"

extern "C" __global__ void computeGridForce(
    const real4* __restrict__ posq,
    unsigned long long* __restrict__ forceBuffers,
    const int* __restrict__ gridCounts,
    const float* __restrict__ gridSpacing,
    const float* __restrict__ gridValues,
    const float* __restrict__ scalingFactors,
    const float invPower,
    const int invPowerMode,  // 0=NONE, 1=RUNTIME, 2=STORED
    const int interpolationMethod,  // 0=trilinear, 1=B-spline, 2=tricubic, 3=triquintic
    const float outOfBoundsK,
    const float originX,
    const float originY,
    const float originZ,
    const GRID_STORAGE_TYPE* __restrict__ gridDerivatives,  // For triquintic: 27 derivatives per point
    mixed* __restrict__ energyBuffer,
    const int numAtoms,
    const int paddedNumAtoms,
    const int* __restrict__ particleIndices,  // Filtered particle indices (null = all particles)
    const int* __restrict__ particleToGroupMap,  // Map particle index to group index (null = no groups)
    mixed* __restrict__ groupEnergyBuffer,  // Per-group energy buffer (null = no groups)
    mixed* __restrict__ groupUnscaledEnergyBuffer,  // Per-group unscaled energy (no group scaling, null = don't store)
    mixed* __restrict__ atomEnergyBuffer,   // Per-atom energy buffer (null = don't store)
    int* __restrict__ outOfBoundsBuffer,    // Per-atom out-of-bounds flags (null = don't store)
    const int numGroups,  // Number of particle groups
    const float arcsinhScale,  // 0.0=disabled, >0.0=apply sinh inverse after interpolation
    const float globalScalingFactor,  // Multiplies all per-particle scaling factors (for alchemical scaling)
    const float* __restrict__ groupScalingFactors,  // Per-group alchemical scaling factors (null = no per-group scaling)
    const float runtimeCap,   // Global runtime cap (0=disabled)
    const float* __restrict__ groupRuntimeCaps,    // Per-group runtime caps (null = use global, 0 = use global)
    mixed* __restrict__ atomRawEnergyBuffer,         // Per-atom raw (pre-cap) energy storage (null = don't store)
    const int evaluateInVSpace,   // Per-corner ^n + V-space cap + no post-interp back-transform
    const float effectiveMinX, const float effectiveMinY, const float effectiveMinZ,  // Effective evaluation bounds (grid-local coords)
    const float effectiveMaxX, const float effectiveMaxY, const float effectiveMaxZ) {

    // Get thread index
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= numAtoms)
        return;

    // Get actual particle index (use filtering if enabled)
    const unsigned int particleIndex = (particleIndices != 0) ? particleIndices[index] : index;

    // Load atom position and scaling factor (with global and per-group alchemical scaling)
    real4 posOrig = posq[particleIndex];
    float groupScale = 1.0f;
    if (groupScalingFactors != 0 && particleToGroupMap != 0) {
        int groupIdx = particleToGroupMap[particleIndex];
        if (groupIdx >= 0 && groupIdx < numGroups) {
            groupScale = groupScalingFactors[groupIdx];
        }
    }
    float scalingFactor = globalScalingFactor * groupScale * scalingFactors[particleIndex];
    float unscaledScaling = globalScalingFactor * scalingFactors[particleIndex];  // No group scaling

    // Resolve effective runtime cap: per-group if available, else global
    float effectiveCap = runtimeCap;
    if (groupRuntimeCaps != 0 && particleToGroupMap != 0) {
        int gIdx = particleToGroupMap[particleIndex];
        if (gIdx >= 0 && gIdx < numGroups && groupRuntimeCaps[gIdx] > 0.0f) {
            effectiveCap = groupRuntimeCaps[gIdx];
        }
    }

    // Transform position to grid coordinates (relative to origin)
    real3 pos;
    pos.x = posOrig.x - originX;
    pos.y = posOrig.y - originY;
    pos.z = posOrig.z - originZ;

    // Initialize force to zero (real/mixed so double mode keeps full precision)
    real3 atomForce = make_real3((real)0, (real)0, (real)0);
    mixed threadEnergy = (mixed)0;
    mixed threadUnscaledEnergy = (mixed)0;

    // Check if the atom is inside the effective evaluation bounds.
    // Effective bounds default to the full grid extent but can be set smaller
    // via setEffectiveBounds() to clip evaluation at a tighter region (e.g.,
    // when the ELE grid is larger than the LJr grid).
    bool isInside = (pos.x >= effectiveMinX && pos.x <= effectiveMaxX &&
                    pos.y >= effectiveMinY && pos.y <= effectiveMaxY &&
                    pos.z >= effectiveMinZ && pos.z <= effectiveMaxZ);

    // Enter interpolation if scaled OR unscaled energy is needed
    bool needUnscaled = (groupUnscaledEnergyBuffer != 0 && unscaledScaling != 0.0f);
    if (isInside && (scalingFactor != 0.0f || needUnscaled)) {
        // =====================================================================
        // Fast path: Use shared GridInterpolation library when no inv_power transformation
        // =====================================================================
        if (invPowerMode == 0 && interpolationMethod == 3 && gridDerivatives != 0 &&
            arcsinhScale == 0.0f && effectiveCap <= 0.0f) {
            // Full real-precision triquintic (no arcsinh/cap): position, eval, and force stay real.
            real3 rpos = make_real3(posOrig.x, posOrig.y, posOrig.z);
            real rval, rgx, rgy, rgz;
            if (triquinticInterpolateReal(gridDerivatives, gridCounts, gridSpacing,
                    (real)originX, (real)originY, (real)originZ, rpos, &rval, &rgx, &rgy, &rgz)) {
                if (atomRawEnergyBuffer != 0)
                    atomRawEnergyBuffer[index] = unscaledScaling * rval;
                threadEnergy = scalingFactor * rval;
                threadUnscaledEnergy = unscaledScaling * rval;
                atomForce.x = -scalingFactor * rgx;
                atomForce.y = -scalingFactor * rgy;
                atomForce.z = -scalingFactor * rgz;
            }
        }
        else if (invPowerMode == 0) {
            // No inv_power transformation - use shared library directly
            real3 absPosition = make_real3(posOrig.x, posOrig.y, posOrig.z);

            InterpolationResult result = interpolateGrid(
                gridValues, gridDerivatives, gridCounts, gridSpacing,
                originX, originY, originZ, absPosition,
                interpolationMethod, true, true);

            if (result.isInside) {
                real val = result.value;
                real gx = result.gradient.x;
                real gy = result.gradient.y;
                real gz = result.gradient.z;

                if (arcsinhScale > 0.0f) {
                    // Arcsinh inverse: V = scale * sinh(g), dV/dr = scale * cosh(g) * dg/dr
                    real sinhG = sinh(val);
                    real coshG = cosh(val);
                    val = arcsinhScale * sinhG;
                    real chainFactor = arcsinhScale * coshG;
                    gx *= chainFactor;
                    gy *= chainFactor;
                    gz *= chainFactor;
                }

                // Store raw (pre-cap) energy for u_kln recomputation
                if (atomRawEnergyBuffer != 0) {
                    atomRawEnergyBuffer[index] = unscaledScaling * val;
                }

                if (effectiveCap > 0.0f) {
                    // Tanh cap: f(v) = C * tanh(v/C), bounded by ±C
                    // Gradient factor: sech²(v/C) = 1 - tanh²(v/C)
                    real t = tanh(val / effectiveCap);
                    real gradFactor = 1.0f - t * t;
                    val = effectiveCap * t;
                    gx *= gradFactor;
                    gy *= gradFactor;
                    gz *= gradFactor;
                }

                threadEnergy = scalingFactor * val;
                threadUnscaledEnergy = unscaledScaling * val;
                atomForce.x = -scalingFactor * gx;
                atomForce.y = -scalingFactor * gy;
                atomForce.z = -scalingFactor * gz;
            }
            // Fall through to force buffer accumulation below
        }
        else {
        // =====================================================================
        // Slow path: inv_power transformation requires specialized code
        // =====================================================================

        // Calculate grid indices
        int ix = min(max((int)(pos.x / gridSpacing[0]), 0), gridCounts[0] - 2);
        int iy = min(max((int)(pos.y / gridSpacing[1]), 0), gridCounts[1] - 2);
        int iz = min(max((int)(pos.z / gridSpacing[2]), 0), gridCounts[2] - 2);

        // Calculate fractional position within the cell
        real fx = (pos.x / gridSpacing[0]) - ix;
        real fy = (pos.y / gridSpacing[1]) - iy;
        real fz = (pos.z / gridSpacing[2]) - iz;

        fx = min(max(fx, 0.0f), 1.0f);
        fy = min(max(fy, 0.0f), 1.0f);
        fz = min(max(fz, 0.0f), 1.0f);

#if DEBUG_GRIDFORCE
        if (index == 0) {
            printf("[KERNEL atom=0] ========== INV_POWER INTERP ==========\n");
            printf("[KERNEL atom=0] invPower=%.6f, invPowerMode=%d\n", invPower, invPowerMode);
            printf("[KERNEL atom=0] scalingFactor=%.6f\n", scalingFactor);
            printf("[KERNEL atom=0] Position: (%.6f, %.6f, %.6f)\n", pos.x, pos.y, pos.z);
            printf("[KERNEL atom=0] Grid indices: ix=%d, iy=%d, iz=%d\n", ix, iy, iz);
            printf("[KERNEL atom=0] Fractional: fx=%.6f, fy=%.6f, fz=%.6f\n", fx, fy, fz);
        }
#endif

        // Declare variables for interpolation
        real interpolated = 0.0f;
        real dx, dy, dz;
        int nyz = gridCounts[1] * gridCounts[2];

        if (interpolationMethod == 1) {
            // CUBIC B-SPLINE INTERPOLATION (4x4x4 = 64 points)
            // Precompute basis functions
            real bx[4] = {bspline_basis0(fx), bspline_basis1(fx), bspline_basis2(fx), bspline_basis3(fx)};
            real by[4] = {bspline_basis0(fy), bspline_basis1(fy), bspline_basis2(fy), bspline_basis3(fy)};
            real bz[4] = {bspline_basis0(fz), bspline_basis1(fz), bspline_basis2(fz), bspline_basis3(fz)};

            real dbx[4] = {bspline_deriv0(fx), bspline_deriv1(fx), bspline_deriv2(fx), bspline_deriv3(fx)};
            real dby[4] = {bspline_deriv0(fy), bspline_deriv1(fy), bspline_deriv2(fy), bspline_deriv3(fy)};
            real dbz[4] = {bspline_deriv0(fz), bspline_deriv1(fz), bspline_deriv2(fz), bspline_deriv3(fz)};

            real dvdx = 0.0f, dvdy = 0.0f, dvdz = 0.0f;

            // Tri-linear B-spline interpolation
            for (int i = 0; i < 4; i++) {
                int gx = min(max(ix - 1 + i, 0), gridCounts[0] - 1);
                for (int j = 0; j < 4; j++) {
                    int gy = min(max(iy - 1 + j, 0), gridCounts[1] - 1);
                    for (int k = 0; k < 4; k++) {
                        int gz = min(max(iz - 1 + k, 0), gridCounts[2] - 1);
                        int gridIdx = gx * nyz + gy * gridCounts[2] + gz;
                        real val = gridValues[gridIdx];

                        // Apply RUNTIME inv_power transformation before interpolation
                        if (invPowerMode == 1) {  // RUNTIME mode
                            real invN = 1.0f / invPower;
                            if (fabs(val) >= 1e-10f) {
                                val = (val >= 0.0f ? 1.0f : -1.0f) * pow(fabs(val), invN);
                            } else {
                                val = 0.0f;
                            }
                        }

                        real weight = bx[i] * by[j] * bz[k];
                        interpolated += weight * val;
                        dvdx += dbx[i] * by[j] * bz[k] * val;
                        dvdy += bx[i] * dby[j] * bz[k] * val;
                        dvdz += bx[i] * by[j] * dbz[k] * val;
                    }
                }
            }

            // Don't divide by spacing here - let the common code at the end handle it
            // This ensures chain rule is applied to unit cell gradients for RUNTIME inv_power mode
            dx = dvdx;
            dy = dvdy;
            dz = dvdz;

        } else if (interpolationMethod == 4) {
            // QUINTIC B-SPLINE INTERPOLATION (6x6x6 = 216 points)
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

            real dvdx = 0.0f, dvdy = 0.0f, dvdz = 0.0f;

            for (int i = 0; i < 6; i++) {
                int gx = min(max(ix - 2 + i, 0), gridCounts[0] - 1);
                for (int j = 0; j < 6; j++) {
                    int gy = min(max(iy - 2 + j, 0), gridCounts[1] - 1);
                    for (int k = 0; k < 6; k++) {
                        int gz = min(max(iz - 2 + k, 0), gridCounts[2] - 1);
                        int gridIdx = gx * nyz + gy * gridCounts[2] + gz;
                        real val = gridValues[gridIdx];

                        // Apply RUNTIME inv_power transformation before interpolation
                        if (invPowerMode == 1) {
                            real invN = 1.0f / invPower;
                            if (fabs(val) >= 1e-10f) {
                                val = (val >= 0.0f ? 1.0f : -1.0f) * pow(fabs(val), invN);
                            } else {
                                val = 0.0f;
                            }
                        }

                        real weight = bx[i] * by[j] * bz[k];
                        interpolated += weight * val;
                        dvdx += dbx[i] * by[j] * bz[k] * val;
                        dvdy += bx[i] * dby[j] * bz[k] * val;
                        dvdz += bx[i] * by[j] * dbz[k] * val;
                    }
                }
            }

            dx = dvdx;
            dy = dvdy;
            dz = dvdz;

        } else if (interpolationMethod == 2 && gridDerivatives != 0) {
            // LEKIEN-MARSDEN TRICUBIC INTERPOLATION (as used in RASPA3)
            // Uses 64x64 transformation matrix to compute polynomial coefficients
            // Requires precomputed analytical derivatives
            // Debug output removed - feature now working correctly
            // if (index == 0) {
            //     if (gridDerivatives != 0) {
            //         printf("TRICUBIC (Lekien-Marsden with ANALYTICAL derivatives) BRANCH EXECUTED for atom 0\n");
            //     } else {
            //         printf("TRICUBIC (Lekien-Marsden with finite differences) BRANCH EXECUTED for atom 0\n");
            //     }
            // }

            // Get 8 corner indices
            int corners[8][3] = {
                {ix, iy, iz}, {ix+1, iy, iz}, {ix, iy+1, iz}, {ix+1, iy+1, iz},
                {ix, iy, iz+1}, {ix+1, iy, iz+1}, {ix, iy+1, iz+1}, {ix+1, iy+1, iz+1}
            };

            // Storage: X[deriv*8 + corner] - DERIVATIVE-MAJOR (matches RASPA3/Lekien-Marsden)
            // Derivatives: 0=f, 1=fx, 2=fy, 3=fz, 4=fxy, 5=fxz, 6=fyz, 7=fxyz
            TricubicAccum X[64];

            // Load analytical derivatives from precomputed grid
            int totalPoints = gridCounts[0] * gridCounts[1] * gridCounts[2];

            // Map tricubic derivative order to gridDerivatives storage order
            // Tricubic needs: 0=f, 1=fx, 2=fy, 3=fz, 4=fxy, 5=fxz, 6=fyz, 7=fxyz
            // gridDerivatives (RASPA3 order): 0=f, 1=dx, 2=dy, 3=dz, 4=dxx, 5=dxy, 6=dxz, 7=dyy, 8=dyz, 9=dzz, ..., 13=dxyz
            const int derivMap[8] = {0, 1, 2, 3, 5, 6, 8, 13};

            if (invPowerMode == 1) {
                // RUNTIME mode: transform all 27 derivatives per corner, then extract needed 8
                real p = 1.0f / invPower;
                for (int c = 0; c < 8; c++) {
                    int point_idx = corners[c][0] * nyz + corners[c][1] * gridCounts[2] + corners[c][2];
                    real U_derivs[27], V_derivs[27];
                    for (int d = 0; d < 27; d++) {
                        U_derivs[d] = gridDerivatives[d * totalPoints + point_idx];
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
                        int point_idx = corners[c][0] * nyz + corners[c][1] * gridCounts[2] + corners[c][2];
                        X[d*8 + c] = gridDerivatives[derivMap[d] * totalPoints + point_idx];
                    }
                }
            }

            // Assemble + evaluate via the shared double-precision helpers.
            TricubicAccum a[64];
            tricubicAssemble(X, a);
            TricubicAccum tv, tgx, tgy, tgz;
            tricubicEvalVG(a, fx, fy, fz, &tv, &tgx, &tgy, &tgz);
            interpolated = tv;
            dx = tgx;
            dy = tgy;
            dz = tgz;

            // Don't divide by spacing here - let the common code at the end handle it
            // This ensures chain rule is applied to unit cell gradients for RUNTIME inv_power mode

        } else if (interpolationMethod == 3 && gridDerivatives != 0) {
            // TRIQUINTIC HERMITE INTERPOLATION (requires precomputed derivatives)
            // Gather 216 derivative values (27 derivatives × 8 corners)
            int totalPoints = gridCounts[0] * gridCounts[1] * gridCounts[2];
            int corners[8][3] = {
                {ix, iy, iz}, {ix+1, iy, iz}, {ix, iy+1, iz}, {ix+1, iy+1, iz},
                {ix, iy, iz+1}, {ix+1, iy, iz+1}, {ix, iy+1, iz+1}, {ix+1, iy+1, iz+1}
            };

            // Gather derivatives in DERIVATIVE-MAJOR layout: X[deriv_idx * 8 + corner_idx]
            // This matches RASPA3's layout expected by TRIQUINTIC_COEFFICIENTS matrix
            // Assemble and evaluate in double (fp32 solve is C2-discontinuous across cells).
            double X[216];
            if (invPowerMode == 1) {
                // RUNTIME mode: transform all 27 derivatives per corner
                real p = 1.0f / invPower;
                for (int c = 0; c < 8; c++) {
                    int point_idx = corners[c][0] * nyz + corners[c][1] * gridCounts[2] + corners[c][2];
                    real U_derivs[27], V_derivs[27];
                    for (int d = 0; d < 27; d++) {
                        U_derivs[d] = gridDerivatives[d * totalPoints + point_idx];
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
                        int point_idx = corners[c][0] * nyz + corners[c][1] * gridCounts[2] + corners[c][2];
                        X[d * 8 + c] = gridDerivatives[d * totalPoints + point_idx];
                    }
                }
            }

            // Assemble + evaluate via the shared double-precision helpers.
            TriquinticAccum a[216];
            triquinticAssemble(X, a);
            TriquinticAccum value, dvalue_dx, dvalue_dy, dvalue_dz;
            triquinticEvalVG(a, fx, fy, fz, &value, &dvalue_dx, &dvalue_dy, &dvalue_dz);

            interpolated = value;
            // Don't divide by spacing here - let the common code at the end handle it
            // This ensures chain rule is applied to unit cell gradients for RUNTIME inv_power mode
            dx = dvalue_dx;
            dy = dvalue_dy;
            dz = dvalue_dz;

        } else if ((interpolationMethod == 2 || interpolationMethod == 3) && gridDerivatives == 0) {
            // Tricubic/Triquintic requested but derivatives not available - return NaN
            // This prevents silent fallback to trilinear which would give incorrect results
            interpolated = nanf("");
            dx = dy = dz = nanf("");
            if (index == 0) {
                printf("ERROR: Interpolation method %d requires derivatives but gridDerivatives is null\n", interpolationMethod);
            }
        } else {
            // TRILINEAR INTERPOLATION (default for method 0, 2x2x2 = 8 points)
            real ox = 1.0f - fx;
            real oy = 1.0f - fy;
            real oz = 1.0f - fz;

            int baseIndex = ix * nyz + iy * gridCounts[2] + iz;
            int ip = baseIndex + nyz;           // ix+1
            int imp = baseIndex + gridCounts[2]; // iy+1
            int ipp = ip + gridCounts[2];       // ix+1, iy+1

            // Get grid values
            real vmmm = gridValues[baseIndex];
            real vmmp = gridValues[baseIndex + 1];
            real vmpm = gridValues[imp];
            real vmpp = gridValues[imp + 1];
            real vpmm = gridValues[ip];
            real vpmp = gridValues[ip + 1];
            real vppm = gridValues[ipp];
            real vppp = gridValues[ipp + 1];

#if DEBUG_GRIDFORCE
            if (index == 0) {
                printf("[KERNEL atom=0] Corner values BEFORE transform:\n");
                printf("  v000=%.6e, v001=%.6e, v010=%.6e, v011=%.6e\n", vmmm, vmmp, vmpm, vmpp);
                printf("  v100=%.6e, v101=%.6e, v110=%.6e, v111=%.6e\n", vpmm, vpmp, vppm, vppp);
            }
#endif

            // RUNTIME mode: Transform grid values BEFORE interpolation
            // This makes interpolation smoother for steep potentials (e.g., LJ)
            if (invPowerMode == 1) {  // 1 = RUNTIME
                real invN = 1.0f / invPower;
                // Transform each grid value: G -> sign(G) * |G|^(1/n)
                if (fabs(vmmm) >= 1e-10f) vmmm = (vmmm >= 0.0f ? 1.0f : -1.0f) * pow(fabs(vmmm), invN);
                else vmmm = 0.0f;
                if (fabs(vmmp) >= 1e-10f) vmmp = (vmmp >= 0.0f ? 1.0f : -1.0f) * pow(fabs(vmmp), invN);
                else vmmp = 0.0f;
                if (fabs(vmpm) >= 1e-10f) vmpm = (vmpm >= 0.0f ? 1.0f : -1.0f) * pow(fabs(vmpm), invN);
                else vmpm = 0.0f;
                if (fabs(vmpp) >= 1e-10f) vmpp = (vmpp >= 0.0f ? 1.0f : -1.0f) * pow(fabs(vmpp), invN);
                else vmpp = 0.0f;
                if (fabs(vpmm) >= 1e-10f) vpmm = (vpmm >= 0.0f ? 1.0f : -1.0f) * pow(fabs(vpmm), invN);
                else vpmm = 0.0f;
                if (fabs(vpmp) >= 1e-10f) vpmp = (vpmp >= 0.0f ? 1.0f : -1.0f) * pow(fabs(vpmp), invN);
                else vpmp = 0.0f;
                if (fabs(vppm) >= 1e-10f) vppm = (vppm >= 0.0f ? 1.0f : -1.0f) * pow(fabs(vppm), invN);
                else vppm = 0.0f;
                if (fabs(vppp) >= 1e-10f) vppp = (vppp >= 0.0f ? 1.0f : -1.0f) * pow(fabs(vppp), invN);
                else vppp = 0.0f;
            }

            // Per-corner back-transform v^(1/n) -> V before cap (STORED mode only).
            // Cap operates in V-space, linear interp in V-space, no post-interp
            // back-transform. Matches trilinear_grid.c semantic.
            if (evaluateInVSpace && invPowerMode == 2) {
                vmmm = (vmmm >= 0.0f ? 1.0f : -1.0f) * pow(fabs(vmmm), (real)invPower);
                vmmp = (vmmp >= 0.0f ? 1.0f : -1.0f) * pow(fabs(vmmp), (real)invPower);
                vmpm = (vmpm >= 0.0f ? 1.0f : -1.0f) * pow(fabs(vmpm), (real)invPower);
                vmpp = (vmpp >= 0.0f ? 1.0f : -1.0f) * pow(fabs(vmpp), (real)invPower);
                vpmm = (vpmm >= 0.0f ? 1.0f : -1.0f) * pow(fabs(vpmm), (real)invPower);
                vpmp = (vpmp >= 0.0f ? 1.0f : -1.0f) * pow(fabs(vpmp), (real)invPower);
                vppm = (vppm >= 0.0f ? 1.0f : -1.0f) * pow(fabs(vppm), (real)invPower);
                vppp = (vppp >= 0.0f ? 1.0f : -1.0f) * pow(fabs(vppp), (real)invPower);
            }

            // Apply runtime tanh cap PER CORNER before interpolation.
            // This preserves gradients near clash boundaries (matching AlGDock's
            // pre-capped grid approach) instead of capping the interpolated value
            // which kills the gradient when the interpolated value >> cap.
            if (effectiveCap > 0.0f) {
                vmmm = effectiveCap * tanh(vmmm / effectiveCap);
                vmmp = effectiveCap * tanh(vmmp / effectiveCap);
                vmpm = effectiveCap * tanh(vmpm / effectiveCap);
                vmpp = effectiveCap * tanh(vmpp / effectiveCap);
                vpmm = effectiveCap * tanh(vpmm / effectiveCap);
                vpmp = effectiveCap * tanh(vpmp / effectiveCap);
                vppm = effectiveCap * tanh(vppm / effectiveCap);
                vppp = effectiveCap * tanh(vppp / effectiveCap);
            }

            // Perform trilinear interpolation (in transformed/capped space)
            real vmm = oz * vmmm + fz * vmmp;
            real vmp = oz * vmpm + fz * vmpp;
            real vpm = oz * vpmm + fz * vpmp;
            real vpp = oz * vppm + fz * vppp;

            real vm = oy * vmm + fy * vmp;
            real vp = oy * vpm + fy * vpp;

            interpolated = ox * vm + fx * vp;

            // Calculate forces (gradients in capped space — natural from capped corners)
            // NOTE: Do NOT divide by spacing yet - must apply chain rule first!
            dx = (vp - vm);
            dy = (ox * (vmp - vmm) + fx * (vpp - vpm));
            dz = (ox * (oy * (vmmp - vmmm) + fy * (vmpp - vmpm)) +
                   fx * (oy * (vpmp - vpmm) + fy * (vppp - vppm)));
        }

        // Undo arcsinh for stacked STORED + arcsinh mode.
        // Grid stores arcsinh(V^(1/n) / scale); undo arcsinh first → V^(1/n) space.
        if (arcsinhScale > 0.0f && invPowerMode == 2 && !evaluateInVSpace) {
            real sinhG = sinh(interpolated);
            real coshG = cosh(interpolated);
            interpolated = arcsinhScale * sinhG;
            real chainFactor = arcsinhScale * coshG;
            dx *= chainFactor;
            dy *= chainFactor;
            dz *= chainFactor;
        }

        // Back-convert from transformed space to get final energy
        // Both RUNTIME and STORED modes need this: val^(1/n) -> (val^(1/n))^n = val
        // Skipped when evaluateInVSpace is set (back-transform already done per-corner).
        if ((invPowerMode == 1 || invPowerMode == 2) && !evaluateInVSpace) {
            real sign = (interpolated >= 0.0f) ? 1.0f : -1.0f;
            real absVal = fabs(interpolated);
            if (absVal > 1e-10f) {
                // Back-convert: val^(1/n) -> val^n recovers original energy
                // Chain rule: if E = sign(v)*|v|^n, then dE/dx = n*|v|^(n-1) * dv/dx
                // The sign cancels: d/dx[sign(v)*|v|^n] = sign(v) * n*|v|^(n-1) * sign(v) * d|v|/dx = n*|v|^(n-1) * dv/dx
                real powerFactor = invPower * pow(absVal, (real)invPower - (real)1.0f);
                interpolated = sign * pow(absVal, (real)invPower);
                // Apply chain rule to gradients BEFORE dividing by spacing
                dx *= powerFactor;
                dy *= powerFactor;
                dz *= powerFactor;
            }
        }

        // Store raw (pre-cap) energy for u_kln recomputation
        if (atomRawEnergyBuffer != 0) {
            atomRawEnergyBuffer[index] = unscaledScaling * interpolated;
        }

        // (tanh cap applied per-corner before interpolation)

        // Now convert gradients to forces by dividing by spacing
        dx /= gridSpacing[0];
        dy /= gridSpacing[1];
        dz /= gridSpacing[2];

        threadEnergy = scalingFactor * interpolated;
        threadUnscaledEnergy = unscaledScaling * interpolated;

        atomForce.x = -scalingFactor * dx;
        atomForce.y = -scalingFactor * dy;
        atomForce.z = -scalingFactor * dz;

        // Debug: Print if forces are abnormally large
        // if (index == 0) {
        //     real force_mag = sqrtf(atomForce.x*atomForce.x + atomForce.y*atomForce.y + atomForce.z*atomForce.z);
        //     if (force_mag > 1e4f) {
        //         printf("[FORCE DEBUG] atom=0: mag=%.6e kJ/mol/nm\n", force_mag);
        //         printf("  Energy: %.6e, Force: (%.6e, %.6e, %.6e)\n",
        //                threadEnergy, atomForce.x, atomForce.y, atomForce.z);
        //     }
        // }
        } // End of invPowerMode != 0 branch
    }
    else {
        // Apply harmonic restraint outside effective bounds (if enabled)
        // NOTE: This restraint is NOT scaled by scalingFactor - it applies uniformly
        // to all particles to keep them within the evaluation boundaries
        real3 dev = make_real3((real)0, (real)0, (real)0);

        if (pos.x < effectiveMinX)
            dev.x = pos.x - effectiveMinX;
        else if (pos.x > effectiveMaxX)
            dev.x = pos.x - effectiveMaxX;

        if (pos.y < effectiveMinY)
            dev.y = pos.y - effectiveMinY;
        else if (pos.y > effectiveMaxY)
            dev.y = pos.y - effectiveMaxY;

        if (pos.z < effectiveMinZ)
            dev.z = pos.z - effectiveMinZ;
        else if (pos.z > effectiveMaxZ)
            dev.z = pos.z - effectiveMaxZ;

        threadEnergy = 0.5f * outOfBoundsK * (dev.x * dev.x + dev.y * dev.y + dev.z * dev.z);
        threadUnscaledEnergy = threadEnergy;  // OOB restraint is not group-scaled
        atomForce.x = -outOfBoundsK * dev.x;  // Don't scale the out-of-bounds restraint!
        atomForce.y = -outOfBoundsK * dev.y;  // Don't scale the out-of-bounds restraint!
        atomForce.z = -outOfBoundsK * dev.z;  // Don't scale the out-of-bounds restraint!
    }

    // Store forces using atomicAdd with unsigned long long
    // IMPORTANT: Must cast to signed long long first to preserve sign bit!
    unsigned long long fx_fixed = (unsigned long long)((long long)(atomForce.x * 0x100000000));
    unsigned long long fy_fixed = (unsigned long long)((long long)(atomForce.y * 0x100000000));
    unsigned long long fz_fixed = (unsigned long long)((long long)(atomForce.z * 0x100000000));

    // if (index == 0) {
    //     printf("  atomicAdd: indices=(%d, %d, %d) | fixed=(%llu, %llu, %llu)\n",
    //            index, index + paddedNumAtoms, index + 2*paddedNumAtoms,
    //            fx_fixed, fy_fixed, fz_fixed);
    // }

    atomicAdd(&forceBuffers[particleIndex], fx_fixed);
    atomicAdd(&forceBuffers[particleIndex + paddedNumAtoms], fy_fixed);
    atomicAdd(&forceBuffers[particleIndex + 2 * paddedNumAtoms], fz_fixed);

    // Store per-atom energy if buffer provided (for debugging/analysis)
    if (atomEnergyBuffer != 0) {
        atomEnergyBuffer[index] = threadEnergy;
    }

    // Store per-atom out-of-bounds flag if buffer provided
    if (outOfBoundsBuffer != 0) {
        outOfBoundsBuffer[index] = isInside ? 0 : 1;
    }

    // Accumulate energy - EITHER to group OR to total, not both
    if (particleToGroupMap != 0 && groupEnergyBuffer != 0) {
        int groupIndex = particleToGroupMap[particleIndex];
        if (groupIndex >= 0 && groupIndex < numGroups) {
            // Particle in a group - only add to group energy
            atomicAdd(&groupEnergyBuffer[groupIndex], threadEnergy);
            // Also track unscaled energy (without group scaling factor)
            if (groupUnscaledEnergyBuffer != 0) {
                atomicAdd(&groupUnscaledEnergyBuffer[groupIndex], threadUnscaledEnergy);
            }
        } else {
            // Particle not in any group - add to total
            atomicAdd(&energyBuffer[0], (mixed)threadEnergy);
        }
    } else {
        // No group tracking - add to total
        atomicAdd(&energyBuffer[0], (mixed)threadEnergy);
    }
}

/**
 * Add group energies to main energy buffer.
 * Simple kernel to sum per-group energies and add to total.
 */
extern "C" __global__ void addGroupEnergiesToTotal(
    mixed* __restrict__ energyBuffer,
    const mixed* __restrict__ groupEnergyBuffer,
    const int numGroups)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        mixed total = (mixed)0;
        for (int i = 0; i < numGroups; i++) {
            total += groupEnergyBuffer[i];
        }
        atomicAdd(&energyBuffer[0], total);
    }
}
