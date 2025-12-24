// Cubic B-spline basis functions
inline DEVICE float bspline_basis0(float t) { return (1.0f - t) * (1.0f - t) * (1.0f - t) / 6.0f; }
inline DEVICE float bspline_basis1(float t) { return (3.0f * t * t * t - 6.0f * t * t + 4.0f) / 6.0f; }
inline DEVICE float bspline_basis2(float t) { return (-3.0f * t * t * t + 3.0f * t * t + 3.0f * t + 1.0f) / 6.0f; }
inline DEVICE float bspline_basis3(float t) { return t * t * t / 6.0f; }

// Derivatives of cubic B-spline basis functions
inline DEVICE float bspline_deriv0(float t) { return -(1.0f - t) * (1.0f - t) / 2.0f; }
inline DEVICE float bspline_deriv1(float t) { return (3.0f * t * t - 4.0f * t) / 2.0f; }
inline DEVICE float bspline_deriv2(float t) { return (-3.0f * t * t + 2.0f * t + 1.0f) / 2.0f; }
inline DEVICE float bspline_deriv3(float t) { return t * t / 2.0f; }

/**
 * Calculate grid forces and energy.
 */
KERNEL void computeGridForce(GLOBAL const real4* RESTRICT posq,
                              GLOBAL mm_long* RESTRICT forceBuffers,
                              GLOBAL const int* RESTRICT gridCounts,
                              GLOBAL const float* RESTRICT gridSpacing,
                              GLOBAL const float* RESTRICT gridValues,
                              GLOBAL const float* RESTRICT scalingFactors,
                              const float invPower,
                              const int interpolationMethod,
                              const float outOfBoundsK,
                              GLOBAL mixed* RESTRICT energyBuffer) {
    // Get thread index
    const unsigned int index = GLOBAL_ID;
    if (index >= NUM_ATOMS)
        return;

    // Load atom position and scaling factor
    real4 pos = posq[index];
    float scalingFactor = scalingFactors[index];

    // Initialize force to zero
    real3 atomForce = make_real3(0, 0, 0);
    mixed threadEnergy = 0;

    // Skip if scaling factor is zero
    if (scalingFactor == 0.0f) {
        ATOMIC_ADD(&forceBuffers[index], (mm_long) (atomForce.x * 0x100000000));
        ATOMIC_ADD(&forceBuffers[index + PADDED_NUM_ATOMS], (mm_long) (atomForce.y * 0x100000000));
        ATOMIC_ADD(&forceBuffers[index + 2 * PADDED_NUM_ATOMS], (mm_long) (atomForce.z * 0x100000000));
        return;
    }

    // Calculate grid boundaries
    real3 gridCorner;
    gridCorner.x = gridSpacing[0] * (gridCounts[0] - 1);
    gridCorner.y = gridSpacing[1] * (gridCounts[1] - 1);
    gridCorner.z = gridSpacing[2] * (gridCounts[2] - 1);

    // Check if the atom is inside the grid
    bool isInside = (pos.x >= 0.0f && pos.x <= gridCorner.x &&
                    pos.y >= 0.0f && pos.y <= gridCorner.y &&
                    pos.z >= 0.0f && pos.z <= gridCorner.z);

    if (isInside) {
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
            float bx[4];
            bx[0] = bspline_basis0(fx);
            bx[1] = bspline_basis1(fx);
            bx[2] = bspline_basis2(fx);
            bx[3] = bspline_basis3(fx);

            float by[4];
            by[0] = bspline_basis0(fy);
            by[1] = bspline_basis1(fy);
            by[2] = bspline_basis2(fy);
            by[3] = bspline_basis3(fy);

            float bz[4];
            bz[0] = bspline_basis0(fz);
            bz[1] = bspline_basis1(fz);
            bz[2] = bspline_basis2(fz);
            bz[3] = bspline_basis3(fz);

            float dbx[4];
            dbx[0] = bspline_deriv0(fx);
            dbx[1] = bspline_deriv1(fx);
            dbx[2] = bspline_deriv2(fx);
            dbx[3] = bspline_deriv3(fx);

            float dby[4];
            dby[0] = bspline_deriv0(fy);
            dby[1] = bspline_deriv1(fy);
            dby[2] = bspline_deriv2(fy);
            dby[3] = bspline_deriv3(fy);

            float dbz[4];
            dbz[0] = bspline_deriv0(fz);
            dbz[1] = bspline_deriv1(fz);
            dbz[2] = bspline_deriv2(fz);
            dbz[3] = bspline_deriv3(fz);

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
            float powerFactor = invPower * POW(interpolated, invPower - 1.0f);
            interpolated = POW(interpolated, invPower);
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
        real3 dev = make_real3(0, 0, 0);

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
        atomForce.x = -scalingFactor * outOfBoundsK * dev.x;
        atomForce.y = -scalingFactor * outOfBoundsK * dev.y;
        atomForce.z = -scalingFactor * outOfBoundsK * dev.z;
    }

    // Store forces - use explicit atomicAdd for CUDA, += operator for OpenCL
#ifdef __CUDACC__
    atomicAdd((unsigned long long*)&forceBuffers[index], (unsigned long long)(atomForce.x * 0x100000000));
    atomicAdd((unsigned long long*)&forceBuffers[index + PADDED_NUM_ATOMS], (unsigned long long)(atomForce.y * 0x100000000));
    atomicAdd((unsigned long long*)&forceBuffers[index + 2 * PADDED_NUM_ATOMS], (unsigned long long)(atomForce.z * 0x100000000));
#else
    forceBuffers[index] += (mm_ulong) (atomForce.x * 0x100000000);
    forceBuffers[index + PADDED_NUM_ATOMS] += (mm_ulong) (atomForce.y * 0x100000000);
    forceBuffers[index + 2 * PADDED_NUM_ATOMS] += (mm_ulong) (atomForce.z * 0x100000000);
#endif

    // Accumulate energy
    energyBuffer[0] += threadEnergy;
}
