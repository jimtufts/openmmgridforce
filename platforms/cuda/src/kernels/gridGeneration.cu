/**
 * Grid generation kernels for OpenMM GridForce plugin.
 * Generates energy grids from receptor atoms using direct LJ/electrostatic calculations.
 */

#include "include/TanhChainRule.cuh"
#include "include/InvPowerChainRule.cuh"

/**
 * Generate grid with analytical derivatives using RASPA3 method.
 *
 * For each grid point, loops over all atoms and accumulates energy and all 27 derivatives
 * using analytical LJ radial derivative formulas combined with tensor conversion.
 *
 * Output format: gridData[gridIdx * 27 + i] where i = 0..26:
 *   [0]      = U (energy)
 *   [1-3]    = ∂U/∂x, ∂U/∂y, ∂U/∂z
 *   [4-9]    = 6 second derivatives
 *   [10-16]  = 7 third derivatives
 *   [17-22]  = 6 fourth derivatives
 *   [23-25]  = 3 fifth derivatives
 *   [26]     = 1 sixth derivative
 */
extern "C" __global__ void generateGridWithAnalyticalDerivatives(
    float* __restrict__ gridData,           // Output: 27 values per grid point
    const float3* __restrict__ receptorPositions,
    const float* __restrict__ receptorCharges,
    const float* __restrict__ receptorSigmas,
    const float* __restrict__ receptorEpsilons,
    const int numReceptorAtoms,
    const int gridType,  // 0=charge, 1=ljr, 2=lja
    const float gridCap,  // Capping threshold U_max (kJ/mol)
    const float invPower,  // Inverse power transformation exponent (0 = disabled)
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
    const float gridPos[3] = {
        originX + i * gridSpacing[0],
        originY + j * gridSpacing[1],
        originZ + k * gridSpacing[2]
    };

    // Initialize accumulator for 27 Cartesian derivatives
    float cartesian_derivs[27];
    for (int idx = 0; idx < 27; idx++) {
        cartesian_derivs[idx] = 0.0f;
    }

    // Loop over all receptor atoms and accumulate contributions
    for (int atomIdx = 0; atomIdx < numReceptorAtoms; atomIdx++) {
        // Get atom position
        float3 atomPos = receptorPositions[atomIdx];

        // Calculate displacement vector: dr = grid_point - atom_position
        float dr[3] = {
            gridPos[0] - atomPos.x,
            gridPos[1] - atomPos.y,
            gridPos[2] - atomPos.z
        };

        // Calculate r²
        float r2 = dr[0]*dr[0] + dr[1]*dr[1] + dr[2]*dr[2];

        // Avoid singularities at very small distances
        // Use r_min = 0.02 nm (0.2 Å) to prevent overflow in r^-12 terms
        const float r2_min = 0.0004f;  // (0.02 nm)^2
        if (r2 < r2_min) {
            r2 = r2_min;
        }

        // Compute 7 radial derivatives based on grid type
        float radial_derivs[7];

        if (gridType == 0) {
            // Charge grid: U = k * q / r
            float charge = receptorCharges[atomIdx];
            computeCoulombRadialDerivatives(r2, charge, radial_derivs);
        } else if (gridType == 1) {
            // LJ repulsion: U = sqrt(epsilon) * Rmin^6 / r^12
            float epsilon = receptorEpsilons[atomIdx];
            float sigma = receptorSigmas[atomIdx];
            computeGeometricLJRepulsionRadialDerivatives(r2, epsilon, sigma, radial_derivs);
        } else if (gridType == 2) {
            // LJ attraction: U = -2 * sqrt(epsilon) * Rmin^3 / r^6
            float epsilon = receptorEpsilons[atomIdx];
            float sigma = receptorSigmas[atomIdx];
            computeGeometricLJAttractionRadialDerivatives(r2, epsilon, sigma, radial_derivs);
        }

        // Convert radial derivatives to Cartesian and accumulate (uncapped)
        accumulateCartesianDerivatives(dr, radial_derivs, cartesian_derivs);
    }

    // Apply exact tanh capping using Faà di Bruno formula
    // This properly computes all 27 derivatives of V = U_max * tanh(U/U_max)
    float capped_derivs[27];
    applyTanhChainRule(cartesian_derivs, gridCap, capped_derivs);

    // Copy capped values back
    for (int i = 0; i < 27; i++) {
        cartesian_derivs[i] = capped_derivs[i];
    }

    // Apply inverse power transformation if specified
    // Transform U -> V = U^p where p = 1/invPower
    // Uses exact chain rule formulas for all 27 derivatives (see InvPowerChainRule.cuh)
    if (invPower != 0.0f) {
        float p = 1.0f / invPower;
        float transformed_derivs[27];
        applyInvPowerChainRule(cartesian_derivs, p, transformed_derivs);

        // Copy transformed values back
        for (int i = 0; i < 27; i++) {
            cartesian_derivs[i] = transformed_derivs[i];
        }
    }

    // Convert from PHYSICAL coordinates (nm) to CELL-FRACTIONAL coordinates
    // Triquintic Hermite needs derivatives where s ∈ [0,1] within each cell
    // dU/dx_physical = dU/ds * (ds/dx_physical) = dU/ds * (1/gridSpacing)
    // So: dU/ds = dU/dx_physical * gridSpacing
    const float dx = gridSpacing[0], dy = gridSpacing[1], dz = gridSpacing[2];

    // Scale first derivatives: ∂U/∂s = ∂U/∂x * Δx
    cartesian_derivs[1] *= dx;  // ∂U/∂x
    cartesian_derivs[2] *= dy;  // ∂U/∂y
    cartesian_derivs[3] *= dz;  // ∂U/∂z

    // Scale second derivatives: ∂²U/∂s² = ∂²U/∂x² * Δx²
    cartesian_derivs[4] *= dx * dx;    // ∂²U/∂x²
    cartesian_derivs[5] *= dx * dy;    // ∂²U/∂x∂y
    cartesian_derivs[6] *= dx * dz;    // ∂²U/∂x∂z
    cartesian_derivs[7] *= dy * dy;    // ∂²U/∂y²
    cartesian_derivs[8] *= dy * dz;    // ∂²U/∂y∂z
    cartesian_derivs[9] *= dz * dz;    // ∂²U/∂z²

    // Scale third derivatives
    cartesian_derivs[10] *= dx * dx * dy;  // ∂³U/∂x²∂y
    cartesian_derivs[11] *= dx * dx * dz;  // ∂³U/∂x²∂z
    cartesian_derivs[12] *= dx * dy * dy;  // ∂³U/∂x∂y²
    cartesian_derivs[13] *= dx * dy * dz;  // ∂³U/∂x∂y∂z
    cartesian_derivs[14] *= dy * dy * dz;  // ∂³U/∂y²∂z
    cartesian_derivs[15] *= dx * dz * dz;  // ∂³U/∂x∂z²
    cartesian_derivs[16] *= dy * dz * dz;  // ∂³U/∂y∂z²

    // Scale fourth derivatives
    cartesian_derivs[17] *= dx * dx * dy * dy;  // ∂⁴U/∂x²∂y²
    cartesian_derivs[18] *= dx * dx * dz * dz;  // ∂⁴U/∂x²∂z²
    cartesian_derivs[19] *= dy * dy * dz * dz;  // ∂⁴U/∂y²∂z²
    cartesian_derivs[20] *= dx * dx * dy * dz;  // ∂⁴U/∂x²∂y∂z
    cartesian_derivs[21] *= dx * dy * dy * dz;  // ∂⁴U/∂x∂y²∂z
    cartesian_derivs[22] *= dx * dy * dz * dz;  // ∂⁴U/∂x∂y∂z²

    // Scale fifth derivatives
    cartesian_derivs[23] *= dx * dx * dy * dy * dz;  // ∂⁵U/∂x²∂y²∂z
    cartesian_derivs[24] *= dx * dx * dy * dz * dz;  // ∂⁵U/∂x²∂y∂z²
    cartesian_derivs[25] *= dx * dy * dy * dz * dz;  // ∂⁵U/∂x∂y²∂z²

    // Scale sixth derivative
    cartesian_derivs[26] *= dx * dx * dy * dy * dz * dz;  // ∂⁶U/∂x²∂y²∂z²

    // Store in layout [deriv_idx * totalGridPoints + gridIdx]
    // Keep RASPA3 order for compatibility with TRIQUINTIC_COEFFICIENTS matrix
    // Order: f, dx, dy, dz, dxx, dxy, dxz, dyy, dyz, dzz, dxxy, ...
    //
    // IMPORTANT: RASPA3 triquintic uses RAW derivatives, not logarithmic!
    // Store derivatives as-is for standard triquintic interpolation.
    for (int idx = 0; idx < 27; idx++) {
        gridData[idx * totalGridPoints + gridIdx] = cartesian_derivs[idx];
    }
}

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

#if DEBUG_GRIDFORCE
    // Debug for test grid point [100, 130, 110]
    bool isTestPoint = (i == 100 && j == 130 && k == 110);
    int topContributors[5] = {-1, -1, -1, -1, -1};
    float topContribValues[5] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    // Print first receptor position for this grid point
    if (isTestPoint && numReceptorAtoms > 0) {
        float3 pos0 = receptorPositions[0];
        printf("[GRIDGEN] First receptor position on GPU: (%.6f, %.6f, %.6f)\n", pos0.x, pos0.y, pos0.z);
    }
#endif

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
        float contrib = 0.0f;
        if (gridType == 0) {
            // Electrostatic potential: k * q / r
            contrib = COULOMB_CONST * receptorCharges[atomIdx] / r;
            gridValue += contrib;
        } else if (gridType == 1) {
            // LJ repulsive: sqrt(epsilon) * Rmin^6 / r^12
            // where Rmin = 2^(1/6) * sigma (AMBER convention)
            const float rmin = powf(2.0f, 1.0f/6.0f) * receptorSigmas[atomIdx];
            const float r6 = rmin * rmin * rmin * rmin * rmin * rmin;
            const float r12 = r2 * r2 * r2 * r2 * r2 * r2;
            contrib = sqrtf(receptorEpsilons[atomIdx]) * r6 / r12;
            gridValue += contrib;
        } else if (gridType == 2) {
            // LJ attractive: -2 * sqrt(epsilon) * Rmin^3 / r^6
            // where Rmin = 2^(1/6) * sigma (AMBER convention)
            const float rmin = powf(2.0f, 1.0f/6.0f) * receptorSigmas[atomIdx];
            const float r3 = rmin * rmin * rmin;
            const float r6 = r2 * r2 * r2;
            contrib = -2.0f * sqrtf(receptorEpsilons[atomIdx]) * r3 / r6;
            gridValue += contrib;
        }

#if DEBUG_GRIDFORCE
        // Track top contributors for test point
        if (isTestPoint) {
            // Print first 5 atoms and top contributors
            if (atomIdx < 5) {
                float sigma = receptorSigmas[atomIdx];
                float eps = receptorEpsilons[atomIdx];
                float rmin = powf(2.0f, 1.0f/6.0f) * sigma;
                printf("[GRIDGEN]   Atom %d: sigma=%.6e, eps=%.6e, rmin=%.6e, r=%.6e, contrib=%.6e\n",
                       atomIdx, sigma, eps, rmin, r, contrib);
            }

            // Track absolute top contributors
            for (int i = 0; i < 5; i++) {
                if (topContributors[i] == -1 || fabsf(contrib) > fabsf(topContribValues[i])) {
                    // Shift down
                    for (int j = 4; j > i; j--) {
                        topContributors[j] = topContributors[j-1];
                        topContribValues[j] = topContribValues[j-1];
                    }
                    topContributors[i] = atomIdx;
                    topContribValues[i] = contrib;
                    break;
                }
            }
        }
#endif
    }

#if DEBUG_GRIDFORCE
    if (isTestPoint) {
        printf("[GRIDGEN] Test point [%d, %d, %d] at (%.6f, %.6f, %.6f)\n", i, j, k, gx, gy, gz);
        printf("[GRIDGEN]   gridType=%d, gridValue BEFORE capping: %.6e\n", gridType, gridValue);
        printf("[GRIDGEN]   Top 5 contributors:\n");
        for (int idx = 0; idx < 5; idx++) {
            if (topContributors[idx] >= 0) {
                printf("[GRIDGEN]     Atom %d: contrib=%.6e\n", topContributors[idx], topContribValues[idx]);
            }
        }
    }
#endif

    // Apply capping to avoid extreme values
    float gridValueBeforeCap = gridValue;
    gridValue = U_MAX * tanhf(gridValue / U_MAX);

#if DEBUG_GRIDFORCE
    if (isTestPoint) {
        printf("[GRIDGEN]   U_MAX=%.6f, gridValue AFTER capping: %.6e\n", U_MAX, gridValue);
    }
#endif

    // Apply inverse power transformation if specified
    // Grid should store G^(1/n), kernel will apply ^n to recover G
    // Handle negative values: sign(G) * |G|^(1/n)
    if (invPower != 0.0f) {
        float sign = (gridValue >= 0.0f) ? 1.0f : -1.0f;
        gridValue = sign * powf(fabsf(gridValue), 1.0f / invPower);
#if DEBUG_GRIDFORCE
        if (isTestPoint) {
            printf("[GRIDGEN]   invPower=%.6f, gridValue AFTER inv_power: %.6e\n", invPower, gridValue);
        }
#endif
    }

#if DEBUG_GRIDFORCE
    if (isTestPoint) {
        printf("[GRIDGEN]   FINAL gridValue stored: %.6e\n", gridValue);
    }
#endif

    // Store result
    gridValues[gridIdx] = gridValue;
}

