/**
 * CUDA implementation of grid force calculation.
 * Main kernel for computing forces on ligand atoms from interpolated grid values.
 */

#include "include/InterpolationBasis.cuh"
#include "include/HermiteBasis.cuh"
#include "include/TricubicCoefficients.cuh"
#include "include/TriquinticCoefficients.cuh"

extern "C" __global__ void computeGridForce(
    const float4* __restrict__ posq,
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
    const float* __restrict__ gridDerivatives,  // For triquintic: 27 derivatives per point
    float* __restrict__ energyBuffer,
    const int numAtoms,
    const int paddedNumAtoms,
    const int* __restrict__ particleIndices,  // Filtered particle indices (null = all particles)
    const int* __restrict__ particleToGroupMap,  // Map particle index to group index (null = no groups)
    float* __restrict__ groupEnergyBuffer,  // Per-group energy buffer (null = no groups)
    const int numGroups) {  // Number of particle groups

    // Get thread index
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= numAtoms)
        return;

    // Get actual particle index (use filtering if enabled)
    const unsigned int particleIndex = (particleIndices != nullptr) ? particleIndices[index] : index;

    // Load atom position and scaling factor
    float4 posOrig = posq[particleIndex];
    float scalingFactor = scalingFactors[particleIndex];

    // Transform position to grid coordinates (relative to origin)
    float3 pos;
    pos.x = posOrig.x - originX;
    pos.y = posOrig.y - originY;
    pos.z = posOrig.z - originZ;

    // Initialize force to zero
    float3 atomForce = make_float3(0.0f, 0.0f, 0.0f);
    float threadEnergy = 0.0f;

    // Calculate grid boundaries
    float3 gridCorner;
    gridCorner.x = gridSpacing[0] * (gridCounts[0] - 1);
    gridCorner.y = gridSpacing[1] * (gridCounts[1] - 1);
    gridCorner.z = gridSpacing[2] * (gridCounts[2] - 1);

    // Check if the atom is inside the grid
    bool isInside = (pos.x >= 0.0f && pos.x <= gridCorner.x &&
                    pos.y >= 0.0f && pos.y <= gridCorner.y &&
                    pos.z >= 0.0f && pos.z <= gridCorner.z);

    if (isInside && scalingFactor != 0.0f) {
        // Calculate grid indices
        int ix = min(max((int)(pos.x / gridSpacing[0]), 0), gridCounts[0] - 2);
        int iy = min(max((int)(pos.y / gridSpacing[1]), 0), gridCounts[1] - 2);
        int iz = min(max((int)(pos.z / gridSpacing[2]), 0), gridCounts[2] - 2);

        // Calculate fractional position within the cell
        float fx = (pos.x / gridSpacing[0]) - ix;
        float fy = (pos.y / gridSpacing[1]) - iy;
        float fz = (pos.z / gridSpacing[2]) - iz;

        fx = min(max(fx, 0.0f), 1.0f);
        fy = min(max(fy, 0.0f), 1.0f);
        fz = min(max(fz, 0.0f), 1.0f);

        // Declare variables for interpolation
        float interpolated = 0.0f;
        float dx, dy, dz;
        int nyz = gridCounts[1] * gridCounts[2];

        if (interpolationMethod == 1) {
            // CUBIC B-SPLINE INTERPOLATION (4x4x4 = 64 points)
            // Precompute basis functions
            float bx[4] = {bspline_basis0(fx), bspline_basis1(fx), bspline_basis2(fx), bspline_basis3(fx)};
            float by[4] = {bspline_basis0(fy), bspline_basis1(fy), bspline_basis2(fy), bspline_basis3(fy)};
            float bz[4] = {bspline_basis0(fz), bspline_basis1(fz), bspline_basis2(fz), bspline_basis3(fz)};

            float dbx[4] = {bspline_deriv0(fx), bspline_deriv1(fx), bspline_deriv2(fx), bspline_deriv3(fx)};
            float dby[4] = {bspline_deriv0(fy), bspline_deriv1(fy), bspline_deriv2(fy), bspline_deriv3(fy)};
            float dbz[4] = {bspline_deriv0(fz), bspline_deriv1(fz), bspline_deriv2(fz), bspline_deriv3(fz)};

            float dvdx = 0.0f, dvdy = 0.0f, dvdz = 0.0f;

            // Tri-linear B-spline interpolation
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
                        dvdx += dbx[i] * by[j] * bz[k] * val;
                        dvdy += bx[i] * dby[j] * bz[k] * val;
                        dvdz += bx[i] * by[j] * dbz[k] * val;
                    }
                }
            }

            dx = dvdx / gridSpacing[0];
            dy = dvdy / gridSpacing[1];
            dz = dvdz / gridSpacing[2];

        } else if (interpolationMethod == 2) {
            // LEKIEN-MARSDEN TRICUBIC INTERPOLATION (as used in RASPA3)
            // Uses 64x64 transformation matrix to compute polynomial coefficients
            // Can use either analytical derivatives (if available) or finite differences
            // Debug output removed - feature now working correctly
            // if (index == 0) {
            //     if (gridDerivatives != nullptr) {
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
            float X[64];

            if (gridDerivatives != nullptr) {
                // USE ANALYTICAL DERIVATIVES (high precision, eliminates finite difference errors)
                int totalPoints = gridCounts[0] * gridCounts[1] * gridCounts[2];

                // Map tricubic derivative order to gridDerivatives storage order
                // Tricubic needs: 0=f, 1=fx, 2=fy, 3=fz, 4=fxy, 5=fxz, 6=fyz, 7=fxyz
                // gridDerivatives: 0=f, 1=dx, 2=dy, 3=dz, 4=dxx, 5=dyy, 6=dzz, 7=dxy, 8=dxz, 9=dyz, ..., 13=dxyz
                const int derivMap[8] = {0, 1, 2, 3, 7, 8, 9, 13};

                for (int d = 0; d < 8; d++) {
                    for (int c = 0; c < 8; c++) {
                        int point_idx = corners[c][0] * nyz + corners[c][1] * gridCounts[2] + corners[c][2];
                        X[d*8 + c] = gridDerivatives[derivMap[d] * totalPoints + point_idx];
                        // Debug output removed - feature now working correctly
                        // if (index == 0 && d == 0 && c == 0) {
                        //     printf("DEBUG: X[0] = gridDerivatives[%d] = %f (deriv=%d, corner=%d, point_idx=%d)\n",
                        //            derivMap[d] * totalPoints + point_idx, X[d*8 + c], d, c, point_idx);
                        // }
                    }
                }
            } else {
                // USE FINITE DIFFERENCES (fallback when analytical derivatives not available)
                int im = ix * nyz + iy * gridCounts[2] + iz;
                int imp = im + gridCounts[2];
                int ip = im + nyz;
                int ipp = ip + gridCounts[2];

                // Get grid corner values
                float f[8];
                f[0] = gridValues[im];      // (0,0,0)
                f[1] = gridValues[ip];      // (1,0,0)
                f[2] = gridValues[imp];     // (0,1,0)
                f[3] = gridValues[ipp];     // (1,1,0)
                f[4] = gridValues[im + 1];  // (0,0,1)
                f[5] = gridValues[ip + 1];  // (1,0,1)
                f[6] = gridValues[imp + 1]; // (0,1,1)
                f[7] = gridValues[ipp + 1]; // (1,1,1)

                // Helper macros for grid indexing
                #define GRID(dx, dy, dz) gridValues[((ix+(dx))*nyz + (iy+(dy))*gridCounts[2] + (iz+(dz)))]

                // For each corner, compute function value and all derivatives using finite differences
                // Store in derivative-major layout: X[deriv*8 + corner]
                for (int c = 0; c < 8; c++) {
                    int cx = (c & 1);  // x offset (0 or 1)
                    int cy = (c >> 1) & 1;  // y offset (0 or 1)
                    int cz = (c >> 2) & 1;  // z offset (0 or 1)

                    // Function value
                    X[0*8 + c] = f[c];

                    // fx - x derivative (centered finite difference)
                    if (ix + cx > 0 && ix + cx < gridCounts[0] - 1) {
                        X[1*8 + c] = (GRID(cx+1, cy, cz) - GRID(cx-1, cy, cz)) / (2.0f * gridSpacing[0]);
                    } else {
                        X[1*8 + c] = 0.0f;
                    }

                    // fy - y derivative
                    if (iy + cy > 0 && iy + cy < gridCounts[1] - 1) {
                        X[2*8 + c] = (GRID(cx, cy+1, cz) - GRID(cx, cy-1, cz)) / (2.0f * gridSpacing[1]);
                    } else {
                        X[2*8 + c] = 0.0f;
                    }

                    // fz - z derivative
                    if (iz + cz > 0 && iz + cz < gridCounts[2] - 1) {
                        X[3*8 + c] = (GRID(cx, cy, cz+1) - GRID(cx, cy, cz-1)) / (2.0f * gridSpacing[2]);
                    } else {
                        X[3*8 + c] = 0.0f;
                    }

                    // fxy - xy cross derivative
                    if (ix + cx > 0 && ix + cx < gridCounts[0] - 1 && iy + cy > 0 && iy + cy < gridCounts[1] - 1) {
                        X[4*8 + c] = (GRID(cx+1, cy+1, cz) - GRID(cx+1, cy-1, cz) - GRID(cx-1, cy+1, cz) + GRID(cx-1, cy-1, cz)) / (4.0f * gridSpacing[0] * gridSpacing[1]);
                    } else {
                        X[4*8 + c] = 0.0f;
                    }

                    // fxz - xz cross derivative
                    if (ix + cx > 0 && ix + cx < gridCounts[0] - 1 && iz + cz > 0 && iz + cz < gridCounts[2] - 1) {
                        X[5*8 + c] = (GRID(cx+1, cy, cz+1) - GRID(cx+1, cy, cz-1) - GRID(cx-1, cy, cz+1) + GRID(cx-1, cy, cz-1)) / (4.0f * gridSpacing[0] * gridSpacing[2]);
                    } else {
                        X[5*8 + c] = 0.0f;
                    }

                    // fyz - yz cross derivative
                    if (iy + cy > 0 && iy + cy < gridCounts[1] - 1 && iz + cz > 0 && iz + cz < gridCounts[2] - 1) {
                        X[6*8 + c] = (GRID(cx, cy+1, cz+1) - GRID(cx, cy+1, cz-1) - GRID(cx, cy-1, cz+1) + GRID(cx, cy-1, cz-1)) / (4.0f * gridSpacing[1] * gridSpacing[2]);
                    } else {
                        X[6*8 + c] = 0.0f;
                    }

                    // fxyz - xyz mixed derivative
                    if (ix + cx > 0 && ix + cx < gridCounts[0] - 1 && iy + cy > 0 && iy + cy < gridCounts[1] - 1 && iz + cz > 0 && iz + cz < gridCounts[2] - 1) {
                        X[7*8 + c] = (GRID(cx+1, cy+1, cz+1) - GRID(cx+1, cy+1, cz-1) - GRID(cx+1, cy-1, cz+1) + GRID(cx+1, cy-1, cz-1)
                                    - GRID(cx-1, cy+1, cz+1) + GRID(cx-1, cy+1, cz-1) + GRID(cx-1, cy-1, cz+1) - GRID(cx-1, cy-1, cz-1))
                                    / (8.0f * gridSpacing[0] * gridSpacing[1] * gridSpacing[2]);
                    } else {
                        X[7*8 + c] = 0.0f;
                    }
                }

                #undef GRID
            }

            // Multiply X by TRICUBIC_COEFFICIENTS matrix to get polynomial coefficients
            float a[64];
            for (int i = 0; i < 64; i++) {
                a[i] = 0.0f;
                for (int j = 0; j < 64; j++) {
                    a[i] += TRICUBIC_COEFFICIENTS[i][j] * X[j];
                }
            }

            // Evaluate tricubic polynomial at (fx, fy, fz) to get interpolated value and derivatives
            // P(x,y,z) = sum_{i,j,k=0}^3 a_{ijk} * x^i * y^j * z^k
            // where a are arranged as a[i + 4*j + 16*k]

            interpolated = 0.0f;
            dx = 0.0f;
            dy = 0.0f;
            dz = 0.0f;

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

            // Convert derivatives from unit cell coordinates to physical coordinates
            dx /= gridSpacing[0];
            dy /= gridSpacing[1];
            dz /= gridSpacing[2];

        } else if (interpolationMethod == 3 && gridDerivatives != nullptr) {
            // TRIQUINTIC HERMITE INTERPOLATION (requires precomputed derivatives)
            // Gather 216 derivative values (27 derivatives × 8 corners)
            int totalPoints = gridCounts[0] * gridCounts[1] * gridCounts[2];
            int corners[8][3] = {
                {ix, iy, iz}, {ix+1, iy, iz}, {ix, iy+1, iz}, {ix+1, iy+1, iz},
                {ix, iy, iz+1}, {ix+1, iy, iz+1}, {ix, iy+1, iz+1}, {ix+1, iy+1, iz+1}
            };

            // Gather derivatives in DERIVATIVE-MAJOR layout: X[deriv_idx * 8 + corner_idx]
            // This matches RASPA3's layout expected by TRIQUINTIC_COEFFICIENTS matrix
            // Derivatives are stored as RAW values (not logarithmic)
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
                        if (i > 0) dvalue_dx += coeff * i * sx_pow[i-1] * sy_pow[j] * sz_pow[k];
                        if (j > 0) dvalue_dy += coeff * j * sx_pow[i] * sy_pow[j-1] * sz_pow[k];
                        if (k > 0) dvalue_dz += coeff * k * sx_pow[i] * sy_pow[j] * sz_pow[k-1];
                    }
                }
            }

            interpolated = value;
            // Convert gradients from cell-local [0,1] to physical coordinates
            dx = dvalue_dx / gridSpacing[0];
            dy = dvalue_dy / gridSpacing[1];
            dz = dvalue_dz / gridSpacing[2];

            // DEBUG: Only print if there are problematic values (NaN, Inf, or coefficients with all-zero derivatives)
            bool hasProblems = isnan(value) || isinf(value) || isnan(a[0]) || isinf(a[0]) ||
                               (X[0] > 1e6f) ||  // Very large corner value
                               (fabsf(X[8]) < 1e-10f && fabsf(X[16]) < 1e-10f && fabsf(X[24]) < 1e-10f && X[0] > 1000.0f); // Zero derivs with high energy
            if (hasProblems && index < 5) {  // Only print for first 5 atoms to avoid spam
                printf("WARNING: TRIQUINTIC ISSUE (atom %d):\n", index);
                printf("  corner0: f=%f, dx=%f, dy=%f, dz=%f\n", X[0], X[8], X[16], X[24]);
                printf("  coeffs: a[0]=%f, a[1]=%f, a[6]=%f, a[36]=%f\n", a[0], a[1], a[6], a[36]);
                printf("  interp: fx=%f, fy=%f, fz=%f, value=%f\n", fx, fy, fz, value);
                printf("  derivs: dx=%f, dy=%f, dz=%f\n", dvalue_dx, dvalue_dy, dvalue_dz);
            }

        } else {
            // TRILINEAR INTERPOLATION (default, 2x2x2 = 8 points)
            float ox = 1.0f - fx;
            float oy = 1.0f - fy;
            float oz = 1.0f - fz;

            int baseIndex = ix * nyz + iy * gridCounts[2] + iz;
            int ip = baseIndex + nyz;           // ix+1
            int imp = baseIndex + gridCounts[2]; // iy+1
            int ipp = ip + gridCounts[2];       // ix+1, iy+1

            // Get grid values
            float vmmm = gridValues[baseIndex];
            float vmmp = gridValues[baseIndex + 1];
            float vmpm = gridValues[imp];
            float vmpp = gridValues[imp + 1];
            float vpmm = gridValues[ip];
            float vpmp = gridValues[ip + 1];
            float vppm = gridValues[ipp];
            float vppp = gridValues[ipp + 1];

            // RUNTIME mode: Transform grid values BEFORE interpolation
            // This makes interpolation smoother for steep potentials (e.g., LJ)
            if (invPowerMode == 1) {  // 1 = RUNTIME
                float invN = 1.0f / invPower;
                // Transform each grid value: G -> sign(G) * |G|^(1/n)
                if (fabsf(vmmm) >= 1e-10f) vmmm = (vmmm >= 0.0f ? 1.0f : -1.0f) * powf(fabsf(vmmm), invN);
                else vmmm = 0.0f;
                if (fabsf(vmmp) >= 1e-10f) vmmp = (vmmp >= 0.0f ? 1.0f : -1.0f) * powf(fabsf(vmmp), invN);
                else vmmp = 0.0f;
                if (fabsf(vmpm) >= 1e-10f) vmpm = (vmpm >= 0.0f ? 1.0f : -1.0f) * powf(fabsf(vmpm), invN);
                else vmpm = 0.0f;
                if (fabsf(vmpp) >= 1e-10f) vmpp = (vmpp >= 0.0f ? 1.0f : -1.0f) * powf(fabsf(vmpp), invN);
                else vmpp = 0.0f;
                if (fabsf(vpmm) >= 1e-10f) vpmm = (vpmm >= 0.0f ? 1.0f : -1.0f) * powf(fabsf(vpmm), invN);
                else vpmm = 0.0f;
                if (fabsf(vpmp) >= 1e-10f) vpmp = (vpmp >= 0.0f ? 1.0f : -1.0f) * powf(fabsf(vpmp), invN);
                else vpmp = 0.0f;
                if (fabsf(vppm) >= 1e-10f) vppm = (vppm >= 0.0f ? 1.0f : -1.0f) * powf(fabsf(vppm), invN);
                else vppm = 0.0f;
                if (fabsf(vppp) >= 1e-10f) vppp = (vppp >= 0.0f ? 1.0f : -1.0f) * powf(fabsf(vppp), invN);
                else vppp = 0.0f;
            }

            // Perform trilinear interpolation (in transformed space if RUNTIME mode)
            float vmm = oz * vmmm + fz * vmmp;
            float vmp = oz * vmpm + fz * vmpp;
            float vpm = oz * vpmm + fz * vpmp;
            float vpp = oz * vppm + fz * vppp;

            float vm = oy * vmm + fy * vmp;
            float vp = oy * vpm + fy * vpp;

            interpolated = ox * vm + fx * vp;

            // Calculate forces (gradients in transformed space)
            dx = (vp - vm) / gridSpacing[0];
            dy = (ox * (vmp - vmm) + fx * (vpp - vpm)) / gridSpacing[1];
            dz = (ox * (oy * (vmmp - vmmm) + fy * (vmpp - vmpm)) +
                   fx * (oy * (vpmp - vpmm) + fy * (vppp - vppm))) / gridSpacing[2];
        }

        // Back-convert from transformed space to get final energy
        // Both RUNTIME and STORED modes need this: val^(1/n) -> (val^(1/n))^n = val
        if (invPowerMode == 1 || invPowerMode == 2) {
            float sign = (interpolated >= 0.0f) ? 1.0f : -1.0f;
            float absVal = fabsf(interpolated);
            if (absVal > 1e-10f) {
                // Back-convert: val^(1/n) -> val^n recovers original energy
                // Derivative chain rule: d/dx[val^n] = n * val^(n-1) * d(val)/dx
                float powerFactor = invPower * powf(absVal, invPower - 1.0f);
                interpolated = sign * powf(absVal, invPower);
                dx *= powerFactor;
                dy *= powerFactor;
                dz *= powerFactor;
            }
        }

        threadEnergy = scalingFactor * interpolated;

        atomForce.x = -scalingFactor * dx;
        atomForce.y = -scalingFactor * dy;
        atomForce.z = -scalingFactor * dz;
    }
    else {
        // Apply harmonic restraint outside grid (if enabled)
        // NOTE: This restraint is NOT scaled by scalingFactor - it applies uniformly
        // to all particles to keep them within the grid boundaries
        float3 dev = make_float3(0.0f, 0.0f, 0.0f);

        if (pos.x < 0.0f)
            dev.x = pos.x;
        else if (pos.x > gridCorner.x)
            dev.x = pos.x - gridCorner.x;

        if (pos.y < 0.0f)
            dev.y = pos.y;
        else if (pos.y > gridCorner.y)
            dev.y = pos.y - gridCorner.y;

        if (pos.z < 0.0f)
            dev.z = pos.z;
        else if (pos.z > gridCorner.z)
            dev.z = pos.z - gridCorner.z;

        threadEnergy = 0.5f * outOfBoundsK * (dev.x * dev.x + dev.y * dev.y + dev.z * dev.z);
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

    // Accumulate energy (total and per-group if enabled)
    atomicAdd(&energyBuffer[0], threadEnergy);

    // Per-group energy tracking
    if (particleToGroupMap != nullptr && groupEnergyBuffer != nullptr) {
        int groupIndex = particleToGroupMap[particleIndex];
        if (groupIndex >= 0 && groupIndex < numGroups) {
            atomicAdd(&groupEnergyBuffer[groupIndex], threadEnergy);
        }
    }
}
