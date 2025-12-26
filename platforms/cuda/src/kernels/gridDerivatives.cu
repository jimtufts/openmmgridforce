/**
 * Grid derivative computation kernel for OpenMM GridForce plugin.
 * Computes derivatives using finite differences from grid energy values.
 * 
 * NOTE: This finite difference approach will be replaced with analytical
 * derivatives computed during grid generation (RASPA3 method).
 */

extern "C" __global__ void computeDerivativesKernel(
    float* __restrict__ derivatives,      // Output: 27 * totalGridPoints
    const float* __restrict__ gridValues, // Input: capped grid values
    const int* __restrict__ gridCounts,
    const float* __restrict__ gridSpacing,
    const int totalGridPoints) {

    const unsigned int gridIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (gridIdx >= totalGridPoints)
        return;

    // Convert linear index to 3D coordinates
    const int nyz = gridCounts[1] * gridCounts[2];
    const int nx = gridCounts[0];
    const int ny = gridCounts[1];
    const int nz = gridCounts[2];

    const int ix = gridIdx / nyz;
    const int remainder = gridIdx % nyz;
    const int iy = remainder / nz;
    const int iz = remainder % nz;

    const float dx = gridSpacing[0];
    const float dy = gridSpacing[1];
    const float dz = gridSpacing[2];

    const float f = gridValues[gridIdx];

    // Helper macro to safely get grid value with clamped indices
    #define GET_VAL(i, j, k) gridValues[min(max((i), 0), nx-1) * nyz + min(max((j), 0), ny-1) * nz + min(max((k), 0), nz-1)]

    // First derivatives in CELL-FRACTIONAL coordinates (not physical)
    // For triquintic: s ∈ [0,1] within cell, so df/ds = (f[i+1] - f[i-1]) / 2
    float dx_f, dy_f, dz_f;
    if (ix == 0) {
        dx_f = (GET_VAL(ix+1, iy, iz) - f);
    } else if (ix == nx-1) {
        dx_f = (f - GET_VAL(ix-1, iy, iz));
    } else {
        dx_f = (GET_VAL(ix+1, iy, iz) - GET_VAL(ix-1, iy, iz)) / 2.0f;
    }

    if (iy == 0) {
        dy_f = (GET_VAL(ix, iy+1, iz) - f);
    } else if (iy == ny-1) {
        dy_f = (f - GET_VAL(ix, iy-1, iz));
    } else {
        dy_f = (GET_VAL(ix, iy+1, iz) - GET_VAL(ix, iy-1, iz)) / 2.0f;
    }

    if (iz == 0) {
        dz_f = (GET_VAL(ix, iy, iz+1) - f);
    } else if (iz == nz-1) {
        dz_f = (f - GET_VAL(ix, iy, iz-1));
    } else {
        dz_f = (GET_VAL(ix, iy, iz+1) - GET_VAL(ix, iy, iz-1)) / 2.0f;
    }

    // Second derivatives in CELL-FRACTIONAL coordinates
    float dxx_f, dyy_f, dzz_f, dxy_f, dxz_f, dyz_f;

    // dxx: use one-sided at boundaries
    if (ix == 0 && nx >= 3) {
        dxx_f = (GET_VAL(ix, iy, iz) - 2.0f*GET_VAL(ix+1, iy, iz) + GET_VAL(ix+2, iy, iz));
    } else if (ix == nx-1 && nx >= 3) {
        dxx_f = (GET_VAL(ix, iy, iz) - 2.0f*GET_VAL(ix-1, iy, iz) + GET_VAL(ix-2, iy, iz));
    } else if (nx >= 3) {
        dxx_f = (GET_VAL(ix+1, iy, iz) - 2.0f*f + GET_VAL(ix-1, iy, iz));
    } else {
        dxx_f = 0.0f;
    }

    // dyy: use one-sided at boundaries
    if (iy == 0 && ny >= 3) {
        dyy_f = (GET_VAL(ix, iy, iz) - 2.0f*GET_VAL(ix, iy+1, iz) + GET_VAL(ix, iy+2, iz));
    } else if (iy == ny-1 && ny >= 3) {
        dyy_f = (GET_VAL(ix, iy, iz) - 2.0f*GET_VAL(ix, iy-1, iz) + GET_VAL(ix, iy-2, iz));
    } else if (ny >= 3) {
        dyy_f = (GET_VAL(ix, iy+1, iz) - 2.0f*f + GET_VAL(ix, iy-1, iz));
    } else {
        dyy_f = 0.0f;
    }

    // dzz: use one-sided at boundaries
    if (iz == 0 && nz >= 3) {
        dzz_f = (GET_VAL(ix, iy, iz) - 2.0f*GET_VAL(ix, iy, iz+1) + GET_VAL(ix, iy, iz+2));
    } else if (iz == nz-1 && nz >= 3) {
        dzz_f = (GET_VAL(ix, iy, iz) - 2.0f*GET_VAL(ix, iy, iz-1) + GET_VAL(ix, iy, iz-2));
    } else if (nz >= 3) {
        dzz_f = (GET_VAL(ix, iy, iz+1) - 2.0f*f + GET_VAL(ix, iy, iz-1));
    } else {
        dzz_f = 0.0f;
    }

    // Mixed second derivatives: use centered if available, one-sided at boundaries
    if (ix == 0 || ix == nx-1 || iy == 0 || iy == ny-1) {
        // Use simple one-sided approximation for dxy at boundaries
        float dx_yp = (ix == nx-1) ? (GET_VAL(ix, iy+1, iz) - GET_VAL(ix-1, iy+1, iz)) :
                                      (GET_VAL(ix+1, iy+1, iz) - GET_VAL(ix, iy+1, iz));
        float dx_ym = (ix == nx-1) ? (GET_VAL(ix, iy-1, iz) - GET_VAL(ix-1, iy-1, iz)) :
                                      (GET_VAL(ix+1, iy-1, iz) - GET_VAL(ix, iy-1, iz));
        dxy_f = (iy == ny-1) ? (dx_ym) : (iy == 0) ? (dx_yp) : (dx_yp - dx_ym) / 2.0f;
    } else {
        dxy_f = (GET_VAL(ix+1, iy+1, iz) - GET_VAL(ix+1, iy-1, iz) -
                 GET_VAL(ix-1, iy+1, iz) + GET_VAL(ix-1, iy-1, iz)) / 4.0f;
    }

    if (ix == 0 || ix == nx-1 || iz == 0 || iz == nz-1) {
        // Use simple one-sided approximation for dxz at boundaries
        float dx_zp = (ix == nx-1) ? (GET_VAL(ix, iy, iz+1) - GET_VAL(ix-1, iy, iz+1)) :
                                      (GET_VAL(ix+1, iy, iz+1) - GET_VAL(ix, iy, iz+1));
        float dx_zm = (ix == nx-1) ? (GET_VAL(ix, iy, iz-1) - GET_VAL(ix-1, iy, iz-1)) :
                                      (GET_VAL(ix+1, iy, iz-1) - GET_VAL(ix, iy, iz-1));
        dxz_f = (iz == nz-1) ? (dx_zm) : (iz == 0) ? (dx_zp) : (dx_zp - dx_zm) / 2.0f;
    } else {
        dxz_f = (GET_VAL(ix+1, iy, iz+1) - GET_VAL(ix+1, iy, iz-1) -
                 GET_VAL(ix-1, iy, iz+1) + GET_VAL(ix-1, iy, iz-1)) / 4.0f;
    }

    if (iy == 0 || iy == ny-1 || iz == 0 || iz == nz-1) {
        // Use simple one-sided approximation for dyz at boundaries
        float dy_zp = (iy == ny-1) ? (GET_VAL(ix, iy, iz+1) - GET_VAL(ix, iy-1, iz+1)) :
                                      (GET_VAL(ix, iy+1, iz+1) - GET_VAL(ix, iy, iz+1));
        float dy_zm = (iy == ny-1) ? (GET_VAL(ix, iy, iz-1) - GET_VAL(ix, iy-1, iz-1)) :
                                      (GET_VAL(ix, iy+1, iz-1) - GET_VAL(ix, iy, iz-1));
        dyz_f = (iz == nz-1) ? (dy_zm) : (iz == 0) ? (dy_zp) : (dy_zp - dy_zm) / 2.0f;
    } else {
        dyz_f = (GET_VAL(ix, iy+1, iz+1) - GET_VAL(ix, iy+1, iz-1) -
                 GET_VAL(ix, iy-1, iz+1) + GET_VAL(ix, iy-1, iz-1)) / 4.0f;
    }

    // Higher order derivatives
    // Use helper macro with bounds checking
    #define GET_VAL(i, j, k) gridValues[min(max((i), 0), nx-1) * nyz + min(max((j), 0), ny-1) * nz + min(max((k), 0), nz-1)]

    float dxxy_f = 0.0f, dxxz_f = 0.0f, dxyy_f = 0.0f, dxyz_f = 0.0f;
    float dxzz_f = 0.0f, dyyz_f = 0.0f, dyzz_f = 0.0f;
    float dxxyy_f = 0.0f, dxxzz_f = 0.0f, dyyzz_f = 0.0f;
    float dxxyz_f = 0.0f, dxyyz_f = 0.0f, dxyzz_f = 0.0f;
    float dxxyyz_f = 0.0f, dxxyzz_f = 0.0f, dxyyzz_f = 0.0f, dxxyyzz_f = 0.0f;

    // Check if we have enough neighbors for higher-order derivatives
    // Need at least nx,ny,nz >= 3 to compute these derivatives
    const bool can_compute = (nx >= 3 && ny >= 3 && nz >= 3);

    if (can_compute) {
        // Third derivatives (7 unique)
        // dxxy = d/dy(dxx) = d/dy(f[i+1] - 2*f[i] + f[i-1])
        dxxy_f = ((GET_VAL(ix+1, iy+1, iz) - 2.0f*GET_VAL(ix, iy+1, iz) + GET_VAL(ix-1, iy+1, iz)) -
                  (GET_VAL(ix+1, iy-1, iz) - 2.0f*GET_VAL(ix, iy-1, iz) + GET_VAL(ix-1, iy-1, iz))) / 2.0f;

        // dxxz = d/dz(dxx)
        dxxz_f = ((GET_VAL(ix+1, iy, iz+1) - 2.0f*GET_VAL(ix, iy, iz+1) + GET_VAL(ix-1, iy, iz+1)) -
                  (GET_VAL(ix+1, iy, iz-1) - 2.0f*GET_VAL(ix, iy, iz-1) + GET_VAL(ix-1, iy, iz-1))) / 2.0f;

        // dxyy = d/dx(dyy)
        dxyy_f = ((GET_VAL(ix+1, iy+1, iz) - 2.0f*GET_VAL(ix+1, iy, iz) + GET_VAL(ix+1, iy-1, iz)) -
                  (GET_VAL(ix-1, iy+1, iz) - 2.0f*GET_VAL(ix-1, iy, iz) + GET_VAL(ix-1, iy-1, iz))) / 2.0f;

        // dyyz = d/dz(dyy)
        dyyz_f = ((GET_VAL(ix, iy+1, iz+1) - 2.0f*GET_VAL(ix, iy, iz+1) + GET_VAL(ix, iy-1, iz+1)) -
                  (GET_VAL(ix, iy+1, iz-1) - 2.0f*GET_VAL(ix, iy, iz-1) + GET_VAL(ix, iy-1, iz-1))) / 2.0f;

        // dxzz = d/dx(dzz)
        dxzz_f = ((GET_VAL(ix+1, iy, iz+1) - 2.0f*GET_VAL(ix+1, iy, iz) + GET_VAL(ix+1, iy, iz-1)) -
                  (GET_VAL(ix-1, iy, iz+1) - 2.0f*GET_VAL(ix-1, iy, iz) + GET_VAL(ix-1, iy, iz-1))) / 2.0f;

        // dyzz = d/dy(dzz)
        dyzz_f = ((GET_VAL(ix, iy+1, iz+1) - 2.0f*GET_VAL(ix, iy+1, iz) + GET_VAL(ix, iy+1, iz-1)) -
                  (GET_VAL(ix, iy-1, iz+1) - 2.0f*GET_VAL(ix, iy-1, iz) + GET_VAL(ix, iy-1, iz-1))) / 2.0f;

        // dxyz = d/dz(dxy) - mixed partial derivative
        dxyz_f = ((GET_VAL(ix+1, iy+1, iz+1) - GET_VAL(ix+1, iy-1, iz+1) -
                   GET_VAL(ix-1, iy+1, iz+1) + GET_VAL(ix-1, iy-1, iz+1)) -
                  (GET_VAL(ix+1, iy+1, iz-1) - GET_VAL(ix+1, iy-1, iz-1) -
                   GET_VAL(ix-1, iy+1, iz-1) + GET_VAL(ix-1, iy-1, iz-1))) / 8.0f;

        // Fourth derivatives (6 unique)
        // dxxyy = d²/dy²(dxx) = d/dy(dxxy)
        dxxyy_f = ((GET_VAL(ix+1, iy+1, iz) - 2.0f*GET_VAL(ix, iy+1, iz) + GET_VAL(ix-1, iy+1, iz)) -
                   2.0f*(GET_VAL(ix+1, iy, iz) - 2.0f*f + GET_VAL(ix-1, iy, iz)) +
                   (GET_VAL(ix+1, iy-1, iz) - 2.0f*GET_VAL(ix, iy-1, iz) + GET_VAL(ix-1, iy-1, iz)));

        // dxxzz = d²/dz²(dxx)
        dxxzz_f = ((GET_VAL(ix+1, iy, iz+1) - 2.0f*GET_VAL(ix, iy, iz+1) + GET_VAL(ix-1, iy, iz+1)) -
                   2.0f*(GET_VAL(ix+1, iy, iz) - 2.0f*f + GET_VAL(ix-1, iy, iz)) +
                   (GET_VAL(ix+1, iy, iz-1) - 2.0f*GET_VAL(ix, iy, iz-1) + GET_VAL(ix-1, iy, iz-1)));

        // dyyzz = d²/dz²(dyy)
        dyyzz_f = ((GET_VAL(ix, iy+1, iz+1) - 2.0f*GET_VAL(ix, iy, iz+1) + GET_VAL(ix, iy-1, iz+1)) -
                   2.0f*(GET_VAL(ix, iy+1, iz) - 2.0f*f + GET_VAL(ix, iy-1, iz)) +
                   (GET_VAL(ix, iy+1, iz-1) - 2.0f*GET_VAL(ix, iy, iz-1) + GET_VAL(ix, iy-1, iz-1)));

        // dxxyz = d/dy(dxxz) - or equivalently d/dz(dxxy)
        dxxyz_f = (((GET_VAL(ix+1, iy+1, iz+1) - 2.0f*GET_VAL(ix, iy+1, iz+1) + GET_VAL(ix-1, iy+1, iz+1)) -
                    (GET_VAL(ix+1, iy+1, iz-1) - 2.0f*GET_VAL(ix, iy+1, iz-1) + GET_VAL(ix-1, iy+1, iz-1))) -
                   ((GET_VAL(ix+1, iy-1, iz+1) - 2.0f*GET_VAL(ix, iy-1, iz+1) + GET_VAL(ix-1, iy-1, iz+1)) -
                    (GET_VAL(ix+1, iy-1, iz-1) - 2.0f*GET_VAL(ix, iy-1, iz-1) + GET_VAL(ix-1, iy-1, iz-1)))) / 4.0f;

        // dxyyz = d/dz(dxyy) - or equivalently d/dy(dxyz)
        dxyyz_f = (((GET_VAL(ix+1, iy+1, iz+1) - 2.0f*GET_VAL(ix+1, iy, iz+1) + GET_VAL(ix+1, iy-1, iz+1)) -
                    (GET_VAL(ix-1, iy+1, iz+1) - 2.0f*GET_VAL(ix-1, iy, iz+1) + GET_VAL(ix-1, iy-1, iz+1))) -
                   ((GET_VAL(ix+1, iy+1, iz-1) - 2.0f*GET_VAL(ix+1, iy, iz-1) + GET_VAL(ix+1, iy-1, iz-1)) -
                    (GET_VAL(ix-1, iy+1, iz-1) - 2.0f*GET_VAL(ix-1, iy, iz-1) + GET_VAL(ix-1, iy-1, iz-1)))) / 4.0f;

        // dxyzz = d/dx(dyzz) - or equivalently d/dz(dxyz)
        dxyzz_f = (((GET_VAL(ix+1, iy+1, iz+1) - 2.0f*GET_VAL(ix+1, iy+1, iz) + GET_VAL(ix+1, iy+1, iz-1)) -
                    (GET_VAL(ix+1, iy-1, iz+1) - 2.0f*GET_VAL(ix+1, iy-1, iz) + GET_VAL(ix+1, iy-1, iz-1))) -
                   ((GET_VAL(ix-1, iy+1, iz+1) - 2.0f*GET_VAL(ix-1, iy+1, iz) + GET_VAL(ix-1, iy+1, iz-1)) -
                    (GET_VAL(ix-1, iy-1, iz+1) - 2.0f*GET_VAL(ix-1, iy-1, iz) + GET_VAL(ix-1, iy-1, iz-1)))) / 4.0f;

        // Fifth derivatives (3 unique)
        // dxxyyz = d/dz(dxxyy) - or equivalently d/dy(dxxyz)
        dxxyyz_f = (((GET_VAL(ix+1, iy+1, iz+1) - 2.0f*GET_VAL(ix, iy+1, iz+1) + GET_VAL(ix-1, iy+1, iz+1)) -
                     2.0f*(GET_VAL(ix+1, iy, iz+1) - 2.0f*GET_VAL(ix, iy, iz+1) + GET_VAL(ix-1, iy, iz+1)) +
                     (GET_VAL(ix+1, iy-1, iz+1) - 2.0f*GET_VAL(ix, iy-1, iz+1) + GET_VAL(ix-1, iy-1, iz+1))) -
                    ((GET_VAL(ix+1, iy+1, iz-1) - 2.0f*GET_VAL(ix, iy+1, iz-1) + GET_VAL(ix-1, iy+1, iz-1)) -
                     2.0f*(GET_VAL(ix+1, iy, iz-1) - 2.0f*GET_VAL(ix, iy, iz-1) + GET_VAL(ix-1, iy, iz-1)) +
                     (GET_VAL(ix+1, iy-1, iz-1) - 2.0f*GET_VAL(ix, iy-1, iz-1) + GET_VAL(ix-1, iy-1, iz-1)))) / 2.0f;

        // dxxyzz = d/dy(dxxzz) - or equivalently d/dz(dxxyz)
        dxxyzz_f = (((GET_VAL(ix+1, iy+1, iz+1) - 2.0f*GET_VAL(ix, iy+1, iz+1) + GET_VAL(ix-1, iy+1, iz+1)) -
                     2.0f*(GET_VAL(ix+1, iy+1, iz) - 2.0f*GET_VAL(ix, iy+1, iz) + GET_VAL(ix-1, iy+1, iz)) +
                     (GET_VAL(ix+1, iy+1, iz-1) - 2.0f*GET_VAL(ix, iy+1, iz-1) + GET_VAL(ix-1, iy+1, iz-1))) -
                    ((GET_VAL(ix+1, iy-1, iz+1) - 2.0f*GET_VAL(ix, iy-1, iz+1) + GET_VAL(ix-1, iy-1, iz+1)) -
                     2.0f*(GET_VAL(ix+1, iy-1, iz) - 2.0f*GET_VAL(ix, iy-1, iz) + GET_VAL(ix-1, iy-1, iz)) +
                     (GET_VAL(ix+1, iy-1, iz-1) - 2.0f*GET_VAL(ix, iy-1, iz-1) + GET_VAL(ix-1, iy-1, iz-1)))) / 2.0f;

        // dxyyzz = d/dx(dyyzz) - or equivalently d/dy(dxyzz) - or equivalently d/dz(dxyyz)
        dxyyzz_f = (((GET_VAL(ix+1, iy+1, iz+1) - 2.0f*GET_VAL(ix+1, iy, iz+1) + GET_VAL(ix+1, iy-1, iz+1)) -
                     2.0f*(GET_VAL(ix+1, iy+1, iz) - 2.0f*GET_VAL(ix+1, iy, iz) + GET_VAL(ix+1, iy-1, iz)) +
                     (GET_VAL(ix+1, iy+1, iz-1) - 2.0f*GET_VAL(ix+1, iy, iz-1) + GET_VAL(ix+1, iy-1, iz-1))) -
                    ((GET_VAL(ix-1, iy+1, iz+1) - 2.0f*GET_VAL(ix-1, iy, iz+1) + GET_VAL(ix-1, iy-1, iz+1)) -
                     2.0f*(GET_VAL(ix-1, iy+1, iz) - 2.0f*GET_VAL(ix-1, iy, iz) + GET_VAL(ix-1, iy-1, iz)) +
                     (GET_VAL(ix-1, iy+1, iz-1) - 2.0f*GET_VAL(ix-1, iy, iz-1) + GET_VAL(ix-1, iy-1, iz-1)))) / 2.0f;

        // Sixth derivative (1 unique)
        // dxxyyzz = d/dz(dxxyy) - or d/dy(dxxyzz) - or d/dx(dxyyzz)
        dxxyyzz_f = ((GET_VAL(ix+1, iy+1, iz+1) - 2.0f*GET_VAL(ix, iy+1, iz+1) + GET_VAL(ix-1, iy+1, iz+1)) -
                     2.0f*(GET_VAL(ix+1, iy, iz+1) - 2.0f*GET_VAL(ix, iy, iz+1) + GET_VAL(ix-1, iy, iz+1)) +
                     (GET_VAL(ix+1, iy-1, iz+1) - 2.0f*GET_VAL(ix, iy-1, iz+1) + GET_VAL(ix-1, iy-1, iz+1))) -
                    2.0f*((GET_VAL(ix+1, iy+1, iz) - 2.0f*GET_VAL(ix, iy+1, iz) + GET_VAL(ix-1, iy+1, iz)) -
                          2.0f*(GET_VAL(ix+1, iy, iz) - 2.0f*f + GET_VAL(ix-1, iy, iz)) +
                          (GET_VAL(ix+1, iy-1, iz) - 2.0f*GET_VAL(ix, iy-1, iz) + GET_VAL(ix-1, iy-1, iz))) +
                    ((GET_VAL(ix+1, iy+1, iz-1) - 2.0f*GET_VAL(ix, iy+1, iz-1) + GET_VAL(ix-1, iy+1, iz-1)) -
                     2.0f*(GET_VAL(ix+1, iy, iz-1) - 2.0f*GET_VAL(ix, iy, iz-1) + GET_VAL(ix-1, iy, iz-1)) +
                     (GET_VAL(ix+1, iy-1, iz-1) - 2.0f*GET_VAL(ix, iy-1, iz-1) + GET_VAL(ix-1, iy-1, iz-1)));
    }

    #undef GET_VAL

    // Convert to LOGARITHMIC derivatives for efficient power transformations
    // L1 = G'/G, L2 = G''/G, etc.
    // This allows: H=G^n => H'=n*H*L1, H''=H*(n*L2 + n*(n-1)*L1²)
    const float epsilon = 1e-10f;  // Avoid division by zero
    const float f_safe = (fabsf(f) < epsilon) ? copysignf(epsilon, f) : f;

    // First logarithmic derivatives: L1_x = dx_f / f, etc.
    float L1_x = dx_f / f_safe;
    float L1_y = dy_f / f_safe;
    float L1_z = dz_f / f_safe;

    // Second logarithmic derivatives: L2_xx = dxx_f / f, etc.
    float L2_xx = dxx_f / f_safe;
    float L2_yy = dyy_f / f_safe;
    float L2_zz = dzz_f / f_safe;
    float L2_xy = dxy_f / f_safe;
    float L2_xz = dxz_f / f_safe;
    float L2_yz = dyz_f / f_safe;

    // Third logarithmic derivatives: L3_xxy = dxxy_f / f, etc.
    float L3_xxy = dxxy_f / f_safe;
    float L3_xxz = dxxz_f / f_safe;
    float L3_xyy = dxyy_f / f_safe;
    float L3_xyz = dxyz_f / f_safe;
    float L3_yzz = dyzz_f / f_safe;
    float L3_xzz = dxzz_f / f_safe;
    float L3_yyz = dyyz_f / f_safe;

    // Fourth logarithmic derivatives: L4_xxyy = dxxyy_f / f, etc.
    float L4_xxyy = dxxyy_f / f_safe;
    float L4_xxzz = dxxzz_f / f_safe;
    float L4_yyzz = dyyzz_f / f_safe;
    float L4_xxyz = dxxyz_f / f_safe;
    float L4_xyyz = dxyyz_f / f_safe;
    float L4_xyzz = dxyzz_f / f_safe;

    // Fifth logarithmic derivatives: L5_xxyyz = dxxyyz_f / f, etc.
    float L5_xxyyz = dxxyyz_f / f_safe;
    float L5_xxyzz = dxxyzz_f / f_safe;
    float L5_xyyzz = dxyyzz_f / f_safe;

    // Sixth logarithmic derivative: L6_xxyyzz = dxxyyzz_f / f
    float L6_xxyyzz = dxxyyzz_f / f_safe;

    // Store in layout [deriv_idx][x][y][z]
    // Total: 27 derivatives for triquintic Hermite interpolation
    // Layout matches RASPA3: f, dx, dy, dz, dxx, dxy, dxz, dyy, dyz, dzz, dxxy, dxxz, dxyy, dxyz, dyyz, dxzz, dyzz,
    //                        dxxyy, dxxzz, dyyzz, dxxyz, dxyyz, dxyzz, dxxyyz, dxxyzz, dxyyzz, dxxyyzz
    const int offset = gridIdx;
    derivatives[0 * totalGridPoints + offset] = f;          // Function value
    derivatives[1 * totalGridPoints + offset] = L1_x;       // 1st derivatives
    derivatives[2 * totalGridPoints + offset] = L1_y;
    derivatives[3 * totalGridPoints + offset] = L1_z;
    derivatives[4 * totalGridPoints + offset] = L2_xx;      // 2nd derivatives
    derivatives[5 * totalGridPoints + offset] = L2_yy;
    derivatives[6 * totalGridPoints + offset] = L2_zz;
    derivatives[7 * totalGridPoints + offset] = L2_xy;
    derivatives[8 * totalGridPoints + offset] = L2_xz;
    derivatives[9 * totalGridPoints + offset] = L2_yz;
    derivatives[10 * totalGridPoints + offset] = L3_xxy;    // 3rd derivatives
    derivatives[11 * totalGridPoints + offset] = L3_xxz;
    derivatives[12 * totalGridPoints + offset] = L3_xyy;
    derivatives[13 * totalGridPoints + offset] = L3_xyz;
    derivatives[14 * totalGridPoints + offset] = L3_yyz;
    derivatives[15 * totalGridPoints + offset] = L3_xzz;
    derivatives[16 * totalGridPoints + offset] = L3_yzz;
    derivatives[17 * totalGridPoints + offset] = L4_xxyy;   // 4th derivatives
    derivatives[18 * totalGridPoints + offset] = L4_xxzz;
    derivatives[19 * totalGridPoints + offset] = L4_yyzz;
    derivatives[20 * totalGridPoints + offset] = L4_xxyz;
    derivatives[21 * totalGridPoints + offset] = L4_xyyz;
    derivatives[22 * totalGridPoints + offset] = L4_xyzz;
    derivatives[23 * totalGridPoints + offset] = L5_xxyyz;  // 5th derivatives
    derivatives[24 * totalGridPoints + offset] = L5_xxyzz;
    derivatives[25 * totalGridPoints + offset] = L5_xyyzz;
    derivatives[26 * totalGridPoints + offset] = L6_xxyyzz; // 6th derivative
}

/**
 * Generate grid values on GPU.
 * Each thread calculates one grid point by summing contributions from all receptor atoms.
 */
