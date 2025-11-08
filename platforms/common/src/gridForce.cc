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

        // Calculate one minus fractional position
        float ox = 1.0f - fx;
        float oy = 1.0f - fy;
        float oz = 1.0f - fz;

        // Get grid points
        int nyz = gridCounts[1] * gridCounts[2];
        int baseIndex = ix * nyz + iy * gridCounts[2] + iz;
        int ip = baseIndex + nyz;           // ix+1
        int imp = baseIndex + gridCounts[2]; // iy+1
        int ipp = ip + gridCounts[2];       // ix+1, iy+1

        // Get grid values safely
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

        // Get interpolated value (still on transformed scale if inv_power was used)
        float interpolated = ox * vm + fx * vp;

        // Apply inverse power transformation if specified
        // This reverses the grid transformation: (G^(1/n))^n = G
        if (invPower > 0.0f) {
            interpolated = POW(interpolated, invPower);
        }

        threadEnergy = scalingFactor * interpolated;

        // Calculate forces (gradients)
        float dx = (vp - vm) / gridSpacing[0];
        float dy = (ox * (vmp - vmm) + fx * (vpp - vpm)) / gridSpacing[1];
        float dz = (ox * (oy * (vmmp - vmmm) + fy * (vmpp - vmpm)) +
                   fx * (oy * (vpmp - vpmm) + fy * (vppp - vppm))) / gridSpacing[2];

        // Apply chain rule if inv_power is set
        // d/dx(f^n) = n * f^(n-1) * df/dx
        if (invPower > 0.0f) {
            float baseInterpolated = ox * vm + fx * vp;  // Value before power transform
            float powerFactor = invPower * POW(baseInterpolated, invPower - 1.0f);
            dx *= powerFactor;
            dy *= powerFactor;
            dz *= powerFactor;
        }

        atomForce.x = -scalingFactor * dx;
        atomForce.y = -scalingFactor * dy;
        atomForce.z = -scalingFactor * dz;
    }
    else {
        // Apply harmonic restraint outside grid
        const float kval = 10000.0f; // kJ/mol nm^2
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

        threadEnergy = 0.5f * kval * (dev.x * dev.x + dev.y * dev.y + dev.z * dev.z);
        atomForce.x = -scalingFactor * kval * dev.x;
        atomForce.y = -scalingFactor * kval * dev.y;
        atomForce.z = -scalingFactor * kval * dev.z;
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
