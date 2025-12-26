/**
 * Grid generation kernels for OpenMM GridForce plugin.
 * Generates energy grids from receptor atoms using direct LJ/electrostatic calculations.
 */

extern "C" __global__ void generateGridKernel(
    float* __restrict__ gridValues,
    const float3* __restrict__ receptorPositions,
    const float* __restrict__ receptorCharges,
    const float* __restrict__ receptorSigmas,
    const float* __restrict__ receptorEpsilons,
    const int numReceptorAtoms,
    const int gridType,  // 0=charge, 1=ljr, 2=lja
    const float gridCap,  // Capping threshold (kJ/mol)
    const float invPower,  // Inverse power transformation
    const float originX,
    const float originY,
    const float originZ,
    const int* __restrict__ gridCounts,
    const float* __restrict__ gridSpacing,
    const int totalGridPoints) {

    // Get grid point index
    const unsigned int gridIdx = blockIdx.x * blockDim.x + threadIdx.x;

    if (gridIdx >= totalGridPoints)
        return;

    // Convert linear index to 3D grid coordinates
    const int nyz = gridCounts[1] * gridCounts[2];
    const int i = gridIdx / nyz;
    const int remainder = gridIdx % nyz;
    const int j = remainder / gridCounts[2];
    const int k = remainder % gridCounts[2];

    // Calculate grid point position (in nm)
    const float gx = originX + i * gridSpacing[0];
    const float gy = originY + j * gridSpacing[1];
    const float gz = originZ + k * gridSpacing[2];

    // Physics constants
    const float COULOMB_CONST = 138.935456f;  // kJ·nm/(mol·e²)
    const float U_MAX = gridCap;              // Configurable capping threshold

    // Calculate contribution from each receptor atom
    float gridValue = 0.0f;
    for (int atomIdx = 0; atomIdx < numReceptorAtoms; atomIdx++) {
        // Get atom position
        float3 atomPos = receptorPositions[atomIdx];

        // Calculate distance
        const float dx = gx - atomPos.x;
        const float dy = gy - atomPos.y;
        const float dz = gz - atomPos.z;
        const float r2 = dx*dx + dy*dy + dz*dz;
        float r = sqrtf(r2);

        // Avoid singularities at very small distances
        if (r < 1e-6f) {
            r = 1e-6f;
        }

        // Calculate contribution based on grid type
        if (gridType == 0) {
            // Electrostatic potential: k * q / r
            gridValue += COULOMB_CONST * receptorCharges[atomIdx] / r;
        } else if (gridType == 1) {
            // LJ repulsive: sqrt(epsilon) * Rmin^6 / r^12
            // where Rmin = 2^(1/6) * sigma (AMBER convention)
            const float rmin = powf(2.0f, 1.0f/6.0f) * receptorSigmas[atomIdx];
            const float r6 = rmin * rmin * rmin * rmin * rmin * rmin;
            const float r12 = r2 * r2 * r2 * r2 * r2 * r2;
            gridValue += sqrtf(receptorEpsilons[atomIdx]) * r6 / r12;
        } else if (gridType == 2) {
            // LJ attractive: -2 * sqrt(epsilon) * Rmin^3 / r^6
            // where Rmin = 2^(1/6) * sigma (AMBER convention)
            const float rmin = powf(2.0f, 1.0f/6.0f) * receptorSigmas[atomIdx];
            const float r3 = rmin * rmin * rmin;
            const float r6 = r2 * r2 * r2;
            gridValue += -2.0f * sqrtf(receptorEpsilons[atomIdx]) * r3 / r6;
        }
    }

    // Apply capping to avoid extreme values
    gridValue = U_MAX * tanhf(gridValue / U_MAX);

    // Apply inverse power transformation if specified
    // Grid should store G^(1/n), kernel will apply ^n to recover G
    if (invPower > 0.0f) {
        gridValue = powf(gridValue, 1.0f / invPower);
    }

    // Store result
    gridValues[gridIdx] = gridValue;
}

