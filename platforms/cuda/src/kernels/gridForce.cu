/**
 * CUDA implementation of grid force calculation.
 * Main kernel for computing forces on ligand atoms from interpolated grid values.
 */

#include "include/InterpolationBasis.cuh"
#include "include/TriquinticCoefficients.cuh"

extern "C" __global__ void computeGridForce(
    const float4* __restrict__ posq,
    unsigned long long* __restrict__ forceBuffers,
    const int* __restrict__ gridCounts,
    const float* __restrict__ gridSpacing,
    const float* __restrict__ gridValues,
    const float* __restrict__ scalingFactors,
    const float invPower,
    const int interpolationMethod,  // 0=trilinear, 1=B-spline, 2=tricubic, 3=triquintic
    const float outOfBoundsK,
    const float originX,
    const float originY,
    const float originZ,
    const float* __restrict__ gridDerivatives,  // For triquintic: 27 derivatives per point
    float* __restrict__ energyBuffer,
    const int numAtoms,
    const int paddedNumAtoms) {

    // Get thread index
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= numAtoms)
        return;

    // Load atom position and scaling factor
    float4 posOrig = posq[index];
    float scalingFactor = scalingFactors[index];

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

            // Perform trilinear interpolation
            float vmm = oz * vmmm + fz * vmmp;
            float vmp = oz * vmpm + fz * vmpp;
            float vpm = oz * vpmm + fz * vpmp;
            float vpp = oz * vppm + fz * vppp;

            float vm = oy * vmm + fy * vmp;
            float vp = oy * vpm + fy * vpp;

            interpolated = ox * vm + fx * vp;

            // Calculate forces (gradients)
            dx = (vp - vm) / gridSpacing[0];
            dy = (ox * (vmp - vmm) + fx * (vpp - vpm)) / gridSpacing[1];
            dz = (ox * (oy * (vmmp - vmmm) + fy * (vmpp - vmpm)) +
                   fx * (oy * (vpmp - vpmm) + fy * (vppp - vppm))) / gridSpacing[2];
        }

        // Apply inverse power transformation if specified
        if (invPower > 0.0f) {
            float powerFactor = invPower * powf(interpolated, invPower - 1.0f);
            interpolated = powf(interpolated, invPower);
            dx *= powerFactor;
            dy *= powerFactor;
            dz *= powerFactor;
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

    atomicAdd(&forceBuffers[index], fx_fixed);
    atomicAdd(&forceBuffers[index + paddedNumAtoms], fy_fixed);
    atomicAdd(&forceBuffers[index + 2 * paddedNumAtoms], fz_fixed);

    // Accumulate energy
    atomicAdd(&energyBuffer[0], threadEnergy);
}
