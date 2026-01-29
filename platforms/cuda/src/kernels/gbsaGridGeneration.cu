/**
 * CUDA kernels for GBSA grid generation.
 *
 * Generates receptor desolvation energy grids for use with GBSAGridForce.
 * The grid stores the change in receptor GB energy when a probe atom is placed
 * at each grid point, enabling efficient computation of receptor desolvation
 * during ligand evaluation.
 */

// OBC-II parameters
#define OBC_ALPHA 1.0f
#define OBC_BETA 0.8f
#define OBC_GAMMA 4.85f
#define DIELECTRIC_OFFSET 0.009f

/**
 * Compute HCT integral contribution from atom j to atom i.
 * This is the standard Hawkins-Cramer-Truhlar formula.
 *
 * @param r         Distance between atoms i and j
 * @param R_i_off   Offset radius of atom i (R_i - 0.009)
 * @param S_j       Scaled radius of atom j (R_j_off * scale_j)
 * @return          HCT contribution
 */
__device__ float computeHCTContribution(float r, float R_i_off, float S_j) {
    if (r < 1e-6f) return 0.0f;

    float r_plus_Sj = r + S_j;
    if (R_i_off >= r_plus_Sj) return 0.0f;  // No overlap

    float r_minus_Sj = fabsf(r - S_j);
    float l_ij = (R_i_off > r_minus_Sj) ? (1.0f / R_i_off) : (1.0f / r_minus_Sj);
    float u_ij = 1.0f / r_plus_Sj;

    float l_ij2 = l_ij * l_ij;
    float u_ij2 = u_ij * u_ij;
    float r_inv = 1.0f / r;

    float term = l_ij - u_ij +
                 0.25f * r * (u_ij2 - l_ij2) +
                 0.5f * r_inv * logf(u_ij / l_ij) +
                 0.25f * S_j * S_j * r_inv * (l_ij2 - u_ij2);

    // Tinker correction: atom i completely inside atom j
    if (R_i_off < (S_j - r)) {
        term += 2.0f * (1.0f / R_i_off - l_ij);
    }

    return term;
}

/**
 * Compute Born radius from HCT using OBC-II formula.
 */
__device__ float computeBornRadius(float R_i, float hctTotal) {
    float R_i_off = R_i - DIELECTRIC_OFFSET;
    float psi = 0.5f * R_i_off * hctTotal;

    float psi2 = psi * psi;
    float psi3 = psi2 * psi;
    float tanhArg = OBC_ALPHA * psi - OBC_BETA * psi2 + OBC_GAMMA * psi3;
    float tanhVal = tanhf(tanhArg);

    float denom = 1.0f / R_i_off - tanhVal / R_i;
    float bornRadius = (denom > 0.0f) ? (1.0f / denom) : R_i;

    // Clamp to reasonable range
    bornRadius = fminf(bornRadius, 50.0f);

    return bornRadius;
}

/**
 * Compute receptor-receptor HCT integrals.
 * This precomputes the baseline HCT for each receptor atom.
 *
 * Each thread computes the HCT for one receptor atom by summing
 * contributions from all other receptor atoms.
 */
extern "C" __global__ void computeReceptorReceptorHCT(
    const float3* __restrict__ receptorPositions,
    const float* __restrict__ receptorRadii,
    const float* __restrict__ receptorScales,
    int numReceptorAtoms,
    float* __restrict__ hctValues
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numReceptorAtoms) return;

    float3 pos_i = receptorPositions[i];
    float R_i = receptorRadii[i];
    float R_i_off = R_i - DIELECTRIC_OFFSET;

    float hct = 0.0f;

    for (int j = 0; j < numReceptorAtoms; j++) {
        if (j == i) continue;

        float3 pos_j = receptorPositions[j];
        float dx = pos_i.x - pos_j.x;
        float dy = pos_i.y - pos_j.y;
        float dz = pos_i.z - pos_j.z;
        float r = sqrtf(dx*dx + dy*dy + dz*dz);

        float R_j = receptorRadii[j];
        float R_j_off = R_j - DIELECTRIC_OFFSET;
        float S_j = R_j_off * receptorScales[j];

        hct += computeHCTContribution(r, R_i_off, S_j);
    }

    hctValues[i] = hct;
}

/**
 * Compute baseline receptor GB energy (sum of self terms).
 * E_baseline = prefactor * sum_i (q_i^2 / R_born_i) / 2
 *
 * This is a reduction kernel - each block computes partial sum,
 * final reduction done on CPU or with another kernel.
 */
extern "C" __global__ void computeBaselineReceptorEnergy(
    const float* __restrict__ receptorCharges,
    const float* __restrict__ receptorRadii,
    const float* __restrict__ hctValues,
    int numReceptorAtoms,
    float prefactor,
    float* __restrict__ partialSums
) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    float energy = 0.0f;
    if (i < numReceptorAtoms) {
        float q_i = receptorCharges[i];
        float R_i = receptorRadii[i];
        float bornRadius = computeBornRadius(R_i, hctValues[i]);

        // Self energy term (0.5 factor in GB formula)
        energy = 0.5f * prefactor * q_i * q_i / bornRadius;
    }

    sdata[tid] = energy;
    __syncthreads();

    // Block reduction
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        partialSums[blockIdx.x] = sdata[0];
    }
}

/**
 * Generate receptor desolvation energy grid.
 *
 * For each grid point, computes the change in receptor GB energy when
 * a probe atom is placed at that point:
 *   ΔE = E_with_probe - E_baseline
 *
 * The probe affects receptor Born radii through its HCT contribution.
 * The grid stores energy values computed with a specific probe radius;
 * at runtime, S³ scaling is applied for different ligand atom sizes.
 *
 * @param gridData               Output grid [numPoints] or [27*numPoints] with derivatives
 * @param receptorPositions      Receptor atom positions [numReceptorAtoms]
 * @param receptorCharges        Receptor charges [numReceptorAtoms]
 * @param receptorRadii          Receptor intrinsic radii [numReceptorAtoms]
 * @param receptorScales         Receptor OBC scale factors [numReceptorAtoms]
 * @param baselineHCT            Precomputed receptor-receptor HCT [numReceptorAtoms]
 * @param baselineBornRadii      Precomputed baseline Born radii [numReceptorAtoms]
 * @param baselineEnergy         Precomputed baseline GB energy (scalar)
 * @param numReceptorAtoms       Number of receptor atoms
 * @param probeRadius            Probe intrinsic radius (nm)
 * @param probeScale             Probe OBC scale factor
 * @param prefactor              GB prefactor: -138.935456 * (1/eps_in - 1/eps_out)
 * @param originX/Y/Z            Grid origin
 * @param gridCounts             Grid dimensions [nx, ny, nz]
 * @param gridSpacing            Grid spacing [dx, dy, dz]
 * @param totalGridPoints        Total number of grid points
 * @param computeDerivatives     0=values only, 1=compute 27 derivatives per point
 */
extern "C" __global__ void generateReceptorDesolvationGrid(
    float* __restrict__ gridData,
    const float3* __restrict__ receptorPositions,
    const float* __restrict__ receptorCharges,
    const float* __restrict__ receptorRadii,
    const float* __restrict__ receptorScales,
    const float* __restrict__ baselineHCT,
    const float* __restrict__ baselineBornRadii,
    float baselineEnergy,
    int numReceptorAtoms,
    float probeRadius,
    float probeScale,
    float prefactor,
    float originX, float originY, float originZ,
    const int* __restrict__ gridCounts,
    const float* __restrict__ gridSpacing,
    int totalGridPoints,
    int computeDerivatives
) {
    int gridIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (gridIdx >= totalGridPoints) return;

    // Convert linear index to 3D coordinates
    int nx = gridCounts[0];
    int ny = gridCounts[1];
    int nz = gridCounts[2];
    int nyz = ny * nz;

    int ix = gridIdx / nyz;
    int remainder = gridIdx % nyz;
    int iy = remainder / nz;
    int iz = remainder % nz;

    // Grid point position
    float gx = originX + ix * gridSpacing[0];
    float gy = originY + iy * gridSpacing[1];
    float gz = originZ + iz * gridSpacing[2];

    // Probe parameters
    float probeRadiusOff = probeRadius - DIELECTRIC_OFFSET;
    float S_probe = probeRadiusOff * probeScale;

    // Compute new receptor energy with probe at this grid point
    float newEnergy = 0.0f;

    for (int i = 0; i < numReceptorAtoms; i++) {
        float3 pos_i = receptorPositions[i];
        float q_i = receptorCharges[i];
        float R_i = receptorRadii[i];
        float R_i_off = R_i - DIELECTRIC_OFFSET;

        // Distance from receptor atom to probe (grid point)
        float dx = pos_i.x - gx;
        float dy = pos_i.y - gy;
        float dz = pos_i.z - gz;
        float r = sqrtf(dx*dx + dy*dy + dz*dz);

        // HCT contribution from probe to receptor atom i
        float hctFromProbe = computeHCTContribution(r, R_i_off, S_probe);

        // New total HCT = baseline + probe contribution
        float newHCT = baselineHCT[i] + hctFromProbe;

        // New Born radius
        float newBornRadius = computeBornRadius(R_i, newHCT);

        // Self energy contribution (0.5 factor in GB formula)
        newEnergy += 0.5f * prefactor * q_i * q_i / newBornRadius;
    }

    // Delta E = new energy - baseline energy
    float deltaE = newEnergy - baselineEnergy;

    // Store result
    if (computeDerivatives) {
        // For now, just store value at index 0, derivatives would go at indices 1-26
        // Full derivative computation would require finite differences or analytical formulas
        gridData[gridIdx * 27] = deltaE;
        // Zero out derivative slots (to be implemented with numerical or analytical derivs)
        for (int d = 1; d < 27; d++) {
            gridData[gridIdx * 27 + d] = 0.0f;
        }
    } else {
        gridData[gridIdx] = deltaE;
    }
}

/**
 * Generate receptor desolvation grid with numerical derivatives.
 *
 * Computes first derivatives using central differences.
 * For triquintic interpolation, we only need first derivatives since
 * the receptor desolvation energy varies smoothly in space.
 */
extern "C" __global__ void generateReceptorDesolvationGridWithDerivatives(
    float* __restrict__ gridData,
    const float3* __restrict__ receptorPositions,
    const float* __restrict__ receptorCharges,
    const float* __restrict__ receptorRadii,
    const float* __restrict__ receptorScales,
    const float* __restrict__ baselineHCT,
    float baselineEnergy,
    int numReceptorAtoms,
    float probeRadius,
    float probeScale,
    float prefactor,
    float originX, float originY, float originZ,
    const int* __restrict__ gridCounts,
    const float* __restrict__ gridSpacing,
    int totalGridPoints
) {
    int gridIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (gridIdx >= totalGridPoints) return;

    // Convert linear index to 3D coordinates
    int nx = gridCounts[0];
    int ny = gridCounts[1];
    int nz = gridCounts[2];
    int nyz = ny * nz;

    int ix = gridIdx / nyz;
    int remainder = gridIdx % nyz;
    int iy = remainder / nz;
    int iz = remainder % nz;

    float dx = gridSpacing[0];
    float dy = gridSpacing[1];
    float dz = gridSpacing[2];

    // Grid point position
    float gx = originX + ix * dx;
    float gy = originY + iy * dy;
    float gz = originZ + iz * dz;

    // Probe parameters
    float probeRadiusOff = probeRadius - DIELECTRIC_OFFSET;
    float S_probe = probeRadiusOff * probeScale;

    // Helper lambda to compute energy at a position
    auto computeEnergyAtPoint = [&](float px, float py, float pz) -> float {
        float energy = 0.0f;
        for (int i = 0; i < numReceptorAtoms; i++) {
            float3 pos_i = receptorPositions[i];
            float q_i = receptorCharges[i];
            float R_i = receptorRadii[i];
            float R_i_off = R_i - DIELECTRIC_OFFSET;

            float ddx = pos_i.x - px;
            float ddy = pos_i.y - py;
            float ddz = pos_i.z - pz;
            float r = sqrtf(ddx*ddx + ddy*ddy + ddz*ddz);

            float hctFromProbe = computeHCTContribution(r, R_i_off, S_probe);
            float newHCT = baselineHCT[i] + hctFromProbe;
            float newBornRadius = computeBornRadius(R_i, newHCT);
            energy += 0.5f * prefactor * q_i * q_i / newBornRadius;
        }
        return energy - baselineEnergy;
    };

    // Compute value and derivatives using central differences
    float E0 = computeEnergyAtPoint(gx, gy, gz);

    // First derivatives
    float h = 0.001f;  // Small step for numerical differentiation (1 pm)
    float dE_dx = (computeEnergyAtPoint(gx + h, gy, gz) - computeEnergyAtPoint(gx - h, gy, gz)) / (2.0f * h);
    float dE_dy = (computeEnergyAtPoint(gx, gy + h, gz) - computeEnergyAtPoint(gx, gy - h, gz)) / (2.0f * h);
    float dE_dz = (computeEnergyAtPoint(gx, gy, gz + h) - computeEnergyAtPoint(gx, gy, gz - h)) / (2.0f * h);

    // Store in RASPA3 layout: [deriv_idx * numPoints + gridIdx]
    // Index 0 = value, 1-3 = first derivatives, 4-26 = higher derivatives (set to 0)
    gridData[0 * totalGridPoints + gridIdx] = E0;
    gridData[1 * totalGridPoints + gridIdx] = dE_dx * dx;  // Scale to cell-fractional
    gridData[2 * totalGridPoints + gridIdx] = dE_dy * dy;
    gridData[3 * totalGridPoints + gridIdx] = dE_dz * dz;

    // Higher derivatives set to zero (tricubic only needs up to first derivs)
    for (int d = 4; d < 27; d++) {
        gridData[d * totalGridPoints + gridIdx] = 0.0f;
    }
}
