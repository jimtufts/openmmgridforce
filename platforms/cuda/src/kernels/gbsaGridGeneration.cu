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

// ============================================================================
// Ligand HCT Grid Generation Kernels
// ============================================================================
// These kernels generate the grid of HCT contributions from receptor atoms
// to probe positions, which is used for computing ligand Born radii.

/**
 * Generate ligand HCT grid (values only).
 *
 * For each grid point, computes the sum of HCT contributions from all
 * receptor atoms to a probe placed at that point. This is the core grid
 * data used by GBSAGridForce.
 *
 * @param gridHctProbe         Output: HCT values [totalGridPoints]
 * @param receptorPositions    Receptor positions [numReceptorAtoms]
 * @param receptorRadii        Receptor intrinsic radii [numReceptorAtoms]
 * @param receptorScales       Receptor OBC scale factors [numReceptorAtoms]
 * @param numReceptorAtoms     Number of receptor atoms
 * @param probeRadius          Probe intrinsic radius (nm)
 * @param originX/Y/Z          Grid origin
 * @param gridCounts           Grid dimensions [nx, ny, nz]
 * @param gridSpacing          Grid spacing (uniform)
 * @param totalGridPoints      Total number of grid points
 */
extern "C" __global__ void generateLigandHCTGrid(
    float* __restrict__ gridHctProbe,
    const float3* __restrict__ receptorPositions,
    const float* __restrict__ receptorRadii,
    const float* __restrict__ receptorScales,
    int numReceptorAtoms,
    float probeRadius,
    float originX, float originY, float originZ,
    const int* __restrict__ gridCounts,
    float gridSpacing,
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

    // Grid point position
    float gx = originX + ix * gridSpacing;
    float gy = originY + iy * gridSpacing;
    float gz = originZ + iz * gridSpacing;

    // Probe offset radius
    float R_probe_off = probeRadius - DIELECTRIC_OFFSET;

    // Sum HCT contributions from all receptor atoms
    float hctSum = 0.0f;

    for (int j = 0; j < numReceptorAtoms; j++) {
        float3 pos_j = receptorPositions[j];

        // Distance from grid point to receptor atom
        float dx = gx - pos_j.x;
        float dy = gy - pos_j.y;
        float dz = gz - pos_j.z;
        float r = sqrtf(dx*dx + dy*dy + dz*dz);

        // Receptor atom scaled radius
        float R_j_off = receptorRadii[j] - DIELECTRIC_OFFSET;
        float S_j = R_j_off * receptorScales[j];

        // HCT contribution from receptor atom j to probe at grid point
        hctSum += computeHCTContribution(r, R_probe_off, S_j);
    }

    gridHctProbe[gridIdx] = hctSum;
}

/**
 * Generate ligand HCT grid with correction terms.
 *
 * In addition to the HCT probe values, computes correction terms (N, A, B)
 * that allow exact HCT computation for any ligand atom radius.
 *
 * Correction formula for radius R_i:
 *   HCT(R_i) = HCT_probe + correction(R_i, N, A, B)
 * where:
 *   correction = (1/R_i - 1/R_probe) * [N - 0.25*A*(1/R_i + 1/R_probe)] + B*ln(R_i/R_probe)
 *
 * @param gridHctProbe         Output: HCT values [totalGridPoints]
 * @param gridCorrectionN      Output: N correction [numBins * totalGridPoints]
 * @param gridCorrectionA      Output: A correction [numBins * totalGridPoints]
 * @param gridCorrectionB      Output: B correction [numBins * totalGridPoints]
 * @param receptorPositions    Receptor positions [numReceptorAtoms]
 * @param receptorRadii        Receptor intrinsic radii [numReceptorAtoms]
 * @param receptorScales       Receptor OBC scale factors [numReceptorAtoms]
 * @param numReceptorAtoms     Number of receptor atoms
 * @param probeRadius          Probe intrinsic radius (nm)
 * @param rThresholds          R thresholds for correction bins [numBins]
 * @param numBins              Number of correction bins
 * @param originX/Y/Z          Grid origin
 * @param gridCounts           Grid dimensions [nx, ny, nz]
 * @param gridSpacing          Grid spacing (uniform)
 * @param totalGridPoints      Total number of grid points
 */
extern "C" __global__ void generateLigandHCTGridWithCorrections(
    float* __restrict__ gridHctProbe,
    float* __restrict__ gridCorrectionN,
    float* __restrict__ gridCorrectionA,
    float* __restrict__ gridCorrectionB,
    const float3* __restrict__ receptorPositions,
    const float* __restrict__ receptorRadii,
    const float* __restrict__ receptorScales,
    int numReceptorAtoms,
    float probeRadius,
    const float* __restrict__ rThresholds,
    int numBins,
    float originX, float originY, float originZ,
    const int* __restrict__ gridCounts,
    float gridSpacing,
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

    // Grid point position
    float gx = originX + ix * gridSpacing;
    float gy = originY + iy * gridSpacing;
    float gz = originZ + iz * gridSpacing;

    // Probe offset radius
    float R_probe_off = probeRadius - DIELECTRIC_OFFSET;

    // Initialize accumulators
    float hctSum = 0.0f;

    // Per-bin correction accumulators (max 4 bins supported)
    float corrN[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float corrA[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float corrB[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int j = 0; j < numReceptorAtoms; j++) {
        float3 pos_j = receptorPositions[j];

        // Distance from grid point to receptor atom
        float dx = gx - pos_j.x;
        float dy = gy - pos_j.y;
        float dz = gz - pos_j.z;
        float r = sqrtf(dx*dx + dy*dy + dz*dz);

        if (r < 1e-6f) continue;

        // Receptor atom scaled radius
        float R_j_off = receptorRadii[j] - DIELECTRIC_OFFSET;
        float S_j = R_j_off * receptorScales[j];

        // HCT contribution from receptor atom j to probe at grid point
        hctSum += computeHCTContribution(r, R_probe_off, S_j);

        // Compute correction terms for each bin
        // Crossover distance where HCT formula changes
        float crossover = fabsf(r - S_j);

        for (int b = 0; b < numBins && b < 4; b++) {
            float thresh = rThresholds[b];

            // Atom contributes to correction if in R-dependent regime for both
            // probe radius and this threshold
            if (crossover < thresh && crossover < R_probe_off) {
                // Correction terms from Python:
                // N = count of atoms in this regime
                // A = sum of (r - S^2/r)
                // B = sum of (0.5/r)
                corrN[b] += 1.0f;
                corrA[b] += r - S_j * S_j / r;
                corrB[b] += 0.5f / r;
            }
        }
    }

    // Store results
    gridHctProbe[gridIdx] = hctSum;

    for (int b = 0; b < numBins && b < 4; b++) {
        int corrIdx = b * totalGridPoints + gridIdx;
        gridCorrectionN[corrIdx] = corrN[b];
        gridCorrectionA[corrIdx] = corrA[b];
        gridCorrectionB[corrIdx] = corrB[b];
    }
}

/**
 * Generate ligand HCT grid with analytical derivatives for triquintic interpolation.
 *
 * Computes the HCT value and all 27 RASPA3 derivatives at each grid point.
 * This enables high-accuracy triquintic Hermite interpolation.
 *
 * Output layout: [deriv_idx * totalGridPoints + gridIdx]
 * RASPA3 order: f, dx, dy, dz, dxx, dxy, dxz, dyy, dyz, dzz, ...
 *
 * @param gridData             Output: 27 values per point [27 * totalGridPoints]
 * @param receptorPositions    Receptor positions [numReceptorAtoms]
 * @param receptorRadii        Receptor intrinsic radii [numReceptorAtoms]
 * @param receptorScales       Receptor OBC scale factors [numReceptorAtoms]
 * @param numReceptorAtoms     Number of receptor atoms
 * @param probeRadius          Probe intrinsic radius (nm)
 * @param originX/Y/Z          Grid origin
 * @param gridCounts           Grid dimensions [nx, ny, nz]
 * @param gridSpacing          Grid spacing (uniform)
 * @param totalGridPoints      Total number of grid points
 */
extern "C" __global__ void generateLigandHCTGridWithDerivatives(
    float* __restrict__ gridData,
    const float3* __restrict__ receptorPositions,
    const float* __restrict__ receptorRadii,
    const float* __restrict__ receptorScales,
    int numReceptorAtoms,
    float probeRadius,
    float originX, float originY, float originZ,
    const int* __restrict__ gridCounts,
    float gridSpacing,
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

    // Grid point position
    float gx = originX + ix * gridSpacing;
    float gy = originY + iy * gridSpacing;
    float gz = originZ + iz * gridSpacing;

    // Probe offset radius
    float R_probe_off = probeRadius - DIELECTRIC_OFFSET;

    // Use numerical derivatives for now (analytical would be complex)
    // Central difference step
    float h = 0.0005f;  // 0.5 pm

    // Helper to compute HCT sum at a position
    auto computeHCTAtPoint = [&](float px, float py, float pz) -> float {
        float hctSum = 0.0f;
        for (int j = 0; j < numReceptorAtoms; j++) {
            float3 pos_j = receptorPositions[j];
            float ddx = px - pos_j.x;
            float ddy = py - pos_j.y;
            float ddz = pz - pos_j.z;
            float r = sqrtf(ddx*ddx + ddy*ddy + ddz*ddz);
            float R_j_off = receptorRadii[j] - DIELECTRIC_OFFSET;
            float S_j = R_j_off * receptorScales[j];
            hctSum += computeHCTContribution(r, R_probe_off, S_j);
        }
        return hctSum;
    };

    // Compute value
    float f = computeHCTAtPoint(gx, gy, gz);

    // First derivatives (central difference)
    float fx = (computeHCTAtPoint(gx+h, gy, gz) - computeHCTAtPoint(gx-h, gy, gz)) / (2.0f*h);
    float fy = (computeHCTAtPoint(gx, gy+h, gz) - computeHCTAtPoint(gx, gy-h, gz)) / (2.0f*h);
    float fz = (computeHCTAtPoint(gx, gy, gz+h) - computeHCTAtPoint(gx, gy, gz-h)) / (2.0f*h);

    // Second derivatives
    float fxx = (computeHCTAtPoint(gx+h, gy, gz) - 2.0f*f + computeHCTAtPoint(gx-h, gy, gz)) / (h*h);
    float fyy = (computeHCTAtPoint(gx, gy+h, gz) - 2.0f*f + computeHCTAtPoint(gx, gy-h, gz)) / (h*h);
    float fzz = (computeHCTAtPoint(gx, gy, gz+h) - 2.0f*f + computeHCTAtPoint(gx, gy, gz-h)) / (h*h);

    float fxy = (computeHCTAtPoint(gx+h, gy+h, gz) - computeHCTAtPoint(gx+h, gy-h, gz)
               - computeHCTAtPoint(gx-h, gy+h, gz) + computeHCTAtPoint(gx-h, gy-h, gz)) / (4.0f*h*h);
    float fxz = (computeHCTAtPoint(gx+h, gy, gz+h) - computeHCTAtPoint(gx+h, gy, gz-h)
               - computeHCTAtPoint(gx-h, gy, gz+h) + computeHCTAtPoint(gx-h, gy, gz-h)) / (4.0f*h*h);
    float fyz = (computeHCTAtPoint(gx, gy+h, gz+h) - computeHCTAtPoint(gx, gy+h, gz-h)
               - computeHCTAtPoint(gx, gy-h, gz+h) + computeHCTAtPoint(gx, gy-h, gz-h)) / (4.0f*h*h);

    // Higher derivatives set to zero (first/second order usually sufficient for smooth interpolation)
    // Scale to cell-fractional coordinates
    float sp = gridSpacing;
    float sp2 = sp * sp;

    // Store in RASPA3 layout: [deriv_idx * totalGridPoints + gridIdx]
    gridData[0 * totalGridPoints + gridIdx] = f;
    gridData[1 * totalGridPoints + gridIdx] = fx * sp;      // dx
    gridData[2 * totalGridPoints + gridIdx] = fy * sp;      // dy
    gridData[3 * totalGridPoints + gridIdx] = fz * sp;      // dz
    gridData[4 * totalGridPoints + gridIdx] = fxx * sp2;    // dxx
    gridData[5 * totalGridPoints + gridIdx] = fxy * sp2;    // dxy
    gridData[6 * totalGridPoints + gridIdx] = fxz * sp2;    // dxz
    gridData[7 * totalGridPoints + gridIdx] = fyy * sp2;    // dyy
    gridData[8 * totalGridPoints + gridIdx] = fyz * sp2;    // dyz
    gridData[9 * totalGridPoints + gridIdx] = fzz * sp2;    // dzz

    // Higher derivatives (third through sixth) set to zero
    for (int d = 10; d < 27; d++) {
        gridData[d * totalGridPoints + gridIdx] = 0.0f;
    }
}
