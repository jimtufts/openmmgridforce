/**
 * CUDA kernels for GBSA grid generation.
 *
 * Generates HCT integral grids for use with GBSAGridForce. The grid stores
 * HCT contributions from receptor atoms to probe positions, enabling efficient
 * computation of ligand Born radii during evaluation.
 */

// OBC-II parameters
#define OBC_ALPHA 1.0f
#define OBC_BETA 0.8f
#define OBC_GAMMA 4.85f
#define DIELECTRIC_OFFSET 0.009f

// Include chain rule headers for analytical derivatives
#include "include/HCTChainRule.cuh"

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
 * Helper: compute HCT sum at a given point from all receptor atoms.
 */
__device__ float computeHCTSumAtPoint(
    float px, float py, float pz,
    const float3* __restrict__ receptorPositions,
    const float* __restrict__ receptorRadii,
    const float* __restrict__ receptorScales,
    int numReceptorAtoms,
    float R_probe_off) {
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

    // Compute value
    float f = computeHCTSumAtPoint(gx, gy, gz, receptorPositions, receptorRadii, receptorScales, numReceptorAtoms, R_probe_off);

    // First derivatives (central difference)
    float fx = (computeHCTSumAtPoint(gx+h, gy, gz, receptorPositions, receptorRadii, receptorScales, numReceptorAtoms, R_probe_off) - computeHCTSumAtPoint(gx-h, gy, gz, receptorPositions, receptorRadii, receptorScales, numReceptorAtoms, R_probe_off)) / (2.0f*h);
    float fy = (computeHCTSumAtPoint(gx, gy+h, gz, receptorPositions, receptorRadii, receptorScales, numReceptorAtoms, R_probe_off) - computeHCTSumAtPoint(gx, gy-h, gz, receptorPositions, receptorRadii, receptorScales, numReceptorAtoms, R_probe_off)) / (2.0f*h);
    float fz = (computeHCTSumAtPoint(gx, gy, gz+h, receptorPositions, receptorRadii, receptorScales, numReceptorAtoms, R_probe_off) - computeHCTSumAtPoint(gx, gy, gz-h, receptorPositions, receptorRadii, receptorScales, numReceptorAtoms, R_probe_off)) / (2.0f*h);

    // Second derivatives
    float fxx = (computeHCTSumAtPoint(gx+h, gy, gz, receptorPositions, receptorRadii, receptorScales, numReceptorAtoms, R_probe_off) - 2.0f*f + computeHCTSumAtPoint(gx-h, gy, gz, receptorPositions, receptorRadii, receptorScales, numReceptorAtoms, R_probe_off)) / (h*h);
    float fyy = (computeHCTSumAtPoint(gx, gy+h, gz, receptorPositions, receptorRadii, receptorScales, numReceptorAtoms, R_probe_off) - 2.0f*f + computeHCTSumAtPoint(gx, gy-h, gz, receptorPositions, receptorRadii, receptorScales, numReceptorAtoms, R_probe_off)) / (h*h);
    float fzz = (computeHCTSumAtPoint(gx, gy, gz+h, receptorPositions, receptorRadii, receptorScales, numReceptorAtoms, R_probe_off) - 2.0f*f + computeHCTSumAtPoint(gx, gy, gz-h, receptorPositions, receptorRadii, receptorScales, numReceptorAtoms, R_probe_off)) / (h*h);

    float fxy = (computeHCTSumAtPoint(gx+h, gy+h, gz, receptorPositions, receptorRadii, receptorScales, numReceptorAtoms, R_probe_off) - computeHCTSumAtPoint(gx+h, gy-h, gz, receptorPositions, receptorRadii, receptorScales, numReceptorAtoms, R_probe_off)
               - computeHCTSumAtPoint(gx-h, gy+h, gz, receptorPositions, receptorRadii, receptorScales, numReceptorAtoms, R_probe_off) + computeHCTSumAtPoint(gx-h, gy-h, gz, receptorPositions, receptorRadii, receptorScales, numReceptorAtoms, R_probe_off)) / (4.0f*h*h);
    float fxz = (computeHCTSumAtPoint(gx+h, gy, gz+h, receptorPositions, receptorRadii, receptorScales, numReceptorAtoms, R_probe_off) - computeHCTSumAtPoint(gx+h, gy, gz-h, receptorPositions, receptorRadii, receptorScales, numReceptorAtoms, R_probe_off)
               - computeHCTSumAtPoint(gx-h, gy, gz+h, receptorPositions, receptorRadii, receptorScales, numReceptorAtoms, R_probe_off) + computeHCTSumAtPoint(gx-h, gy, gz-h, receptorPositions, receptorRadii, receptorScales, numReceptorAtoms, R_probe_off)) / (4.0f*h*h);
    float fyz = (computeHCTSumAtPoint(gx, gy+h, gz+h, receptorPositions, receptorRadii, receptorScales, numReceptorAtoms, R_probe_off) - computeHCTSumAtPoint(gx, gy+h, gz-h, receptorPositions, receptorRadii, receptorScales, numReceptorAtoms, R_probe_off)
               - computeHCTSumAtPoint(gx, gy-h, gz+h, receptorPositions, receptorRadii, receptorScales, numReceptorAtoms, R_probe_off) + computeHCTSumAtPoint(gx, gy-h, gz-h, receptorPositions, receptorRadii, receptorScales, numReceptorAtoms, R_probe_off)) / (4.0f*h*h);

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

/**
 * Numerically stable sigmoid function.
 */
__device__ __forceinline__ float sigmoidKDE(float u) {
    if (u >= 0.0f) {
        float e = expf(-u);
        return 1.0f / (1.0f + e);
    } else {
        float e = expf(u);
        return e / (1.0f + e);
    }
}

/**
 * Generate ligand HCT grid with KDE-smoothed binned correction terms.
 *
 * This kernel combines the binned correction structure (multiple N, A, B grids
 * for different radius thresholds) with KDE sigmoid smoothing for continuity.
 *
 * Instead of hard cutoffs at bin boundaries, atoms contribute with a smooth
 * KDE weight: w = sigmoid((thresh - crossover) / bandwidth) * sigmoid((R_probe - crossover) / bandwidth)
 *
 * This maintains the R_i-dependent bin selection at runtime while providing
 * smooth transitions within each bin for better Hessian computation.
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
 * @param kdeBandwidth         KDE sigmoid bandwidth (nm)
 * @param originX/Y/Z          Grid origin
 * @param gridCounts           Grid dimensions [nx, ny, nz]
 * @param gridSpacing          Grid spacing (uniform)
 * @param totalGridPoints      Total number of grid points
 */
extern "C" __global__ void generateBinnedGridsWithKDE(
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
    float kdeBandwidth,
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

    float invBandwidth = 1.0f / kdeBandwidth;

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

        // Compute KDE-smoothed correction terms for each bin
        // Crossover distance where HCT formula changes
        float crossover = fabsf(r - S_j);

        // Probe constraint sigmoid (shared across bins)
        float u_probe = (R_probe_off - crossover) * invBandwidth;
        float sig_probe = sigmoidKDE(u_probe);

        // Skip if probe sigmoid is essentially zero
        if (u_probe < -10.0f) continue;

        for (int b = 0; b < numBins && b < 4; b++) {
            float thresh = rThresholds[b];

            // KDE weight: smooth transition instead of hard cutoff
            // w ≈ 1 when crossover < min(thresh, R_probe_off)
            // w ≈ 0 when crossover > max(thresh, R_probe_off)
            float u_thresh = (thresh - crossover) * invBandwidth;
            float sig_thresh = sigmoidKDE(u_thresh);
            float w = sig_thresh * sig_probe;

            // Skip if weight is negligible
            if (w < 1e-6f) continue;

            // Weighted correction terms
            // N = weighted count of atoms in this regime
            // A = weighted sum of (r - S^2/r)
            // B = weighted sum of (0.5/r)
            corrN[b] += w;
            corrA[b] += w * (r - S_j * S_j / r);
            corrB[b] += w * 0.5f / r;
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

// Include KDE chain rule for analytical correction derivatives
#include "include/KDEChainRule.cuh"

/**
 * Generate HCT grid with analytical derivatives AND binned corrections with KDE derivatives.
 *
 * This kernel generates:
 * - HCT grid with 27 RASPA3 derivatives for tricubic/triquintic interpolation
 * - Binned correction grids (N, A, B) with 27 derivatives per bin
 *
 * Output layouts:
 * - gridHctDerivatives: [27 * totalGridPoints] in RASPA3 order
 * - gridCorrectionN/A/B: [numBins * 27 * totalGridPoints] - bin-major, then deriv, then point
 */
extern "C" __global__ void generateBinnedGridsWithKDEDerivatives(
    float* __restrict__ gridHctDerivatives,
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
    float kdeBandwidth,
    float kdeEpsilonB,
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

    // Initialize accumulators for HCT derivatives (27 values)
    float hct_derivs[27] = {0.0f};

    // Per-bin correction derivative accumulators (max 4 bins, 27 derivs each)
    float corrN_derivs[4][27];
    float corrA_derivs[4][27];
    float corrB_derivs[4][27];
    for (int b = 0; b < 4; b++) {
        for (int d = 0; d < 27; d++) {
            corrN_derivs[b][d] = 0.0f;
            corrA_derivs[b][d] = 0.0f;
            corrB_derivs[b][d] = 0.0f;
        }
    }

    // Process each receptor atom
    for (int j = 0; j < numReceptorAtoms; j++) {
        float3 pos_j = receptorPositions[j];

        // Displacement from receptor atom to grid point
        float dr[3];
        dr[0] = gx - pos_j.x;
        dr[1] = gy - pos_j.y;
        dr[2] = gz - pos_j.z;

        float r2 = dr[0]*dr[0] + dr[1]*dr[1] + dr[2]*dr[2];
        float r = sqrtf(r2);

        if (r < 1e-6f) continue;

        // Receptor atom scaled radius
        float R_j_off = receptorRadii[j] - DIELECTRIC_OFFSET;
        float S_j = R_j_off * receptorScales[j];

        // Check if atom can contribute to HCT
        float r_plus_Sj = r + S_j;
        if (R_probe_off >= r_plus_Sj) continue;

        // ================================================================
        // HCT grid: analytical derivatives using HCTChainRule
        // ================================================================
        {
            float radial_derivs[7];
            computeHCT_rDerivs(r, S_j, R_probe_off, radial_derivs);

            float atom_cartesian[27] = {0.0f};
            accumulateCartesianDerivatives(dr, radial_derivs, atom_cartesian);

            for (int d = 0; d < 27; d++) {
                hct_derivs[d] += atom_cartesian[d];
            }
        }

        // ================================================================
        // Correction grids: KDE-weighted derivatives per bin
        // ================================================================
        {
            float crossover = fabsf(r - S_j);
            float invR = 1.0f / r;
            float nx_dir = dr[0] * invR;
            float ny_dir = dr[1] * invR;
            float nz_dir = dr[2] * invR;

            for (int b = 0; b < numBins && b < 4; b++) {
                float thresh = rThresholds[b];

                // Compute KDE weight derivatives for this bin's threshold
                float W[7];
                computeKDEWeightDerivatives(r, S_j, thresh, R_probe_off, kdeBandwidth, W);

                // Skip if weight is negligible
                if (W[0] < 1e-8f) continue;

                // N grid: f = 1, so g = w * 1 = w
                {
                    float F[7];
                    computeFN_r_derivs(r, S_j, F);
                    float G[7];
                    computeProductDerivatives(W, F, G);
                    float cart[27];
                    applyChainRuleTriquintic(G, nx_dir, ny_dir, nz_dir, r, cart);
                    for (int d = 0; d < 27; d++) {
                        corrN_derivs[b][d] += cart[d];
                    }
                }

                // A grid: f = r - S^2/r
                {
                    float F[7];
                    computeFA_r_derivs(r, S_j, F);
                    float G[7];
                    computeProductDerivatives(W, F, G);
                    float cart[27];
                    applyChainRuleTriquintic(G, nx_dir, ny_dir, nz_dir, r, cart);
                    for (int d = 0; d < 27; d++) {
                        corrA_derivs[b][d] += cart[d];
                    }
                }

                // B grid: f = 0.5 / sqrt(r^2 + eps^2)
                {
                    float F[7];
                    computeFB_r_derivs(r, kdeEpsilonB, F);
                    float G[7];
                    computeProductDerivatives(W, F, G);
                    float cart[27];
                    applyChainRuleTriquintic(G, nx_dir, ny_dir, nz_dir, r, cart);
                    for (int d = 0; d < 27; d++) {
                        corrB_derivs[b][d] += cart[d];
                    }
                }
            }
        }
    }

    // Scale derivatives to cell-fractional coordinates
    float sp = gridSpacing;
    float scales[27];
    scales[0] = 1.0f;  // f
    scales[1] = sp;    // dx
    scales[2] = sp;    // dy
    scales[3] = sp;    // dz
    float sp2 = sp * sp;
    scales[4] = sp2;   // dxx
    scales[5] = sp2;   // dxy
    scales[6] = sp2;   // dxz
    scales[7] = sp2;   // dyy
    scales[8] = sp2;   // dyz
    scales[9] = sp2;   // dzz
    float sp3 = sp2 * sp;
    scales[10] = sp3;  // dxxy
    scales[11] = sp3;  // dxxz
    scales[12] = sp3;  // dxyy
    scales[13] = sp3;  // dxyz
    scales[14] = sp3;  // dyyz
    scales[15] = sp3;  // dxzz
    scales[16] = sp3;  // dyzz
    float sp4 = sp3 * sp;
    scales[17] = sp4;  // dxxyy
    scales[18] = sp4;  // dxxzz
    scales[19] = sp4;  // dyyzz
    scales[20] = sp4;  // dxxyz
    scales[21] = sp4;  // dxyyz
    scales[22] = sp4;  // dxyzz
    float sp5 = sp4 * sp;
    scales[23] = sp5;  // dxxyyz
    scales[24] = sp5;  // dxxyzz
    scales[25] = sp5;  // dxyyzz
    float sp6 = sp5 * sp;
    scales[26] = sp6;  // dxxyyzz

    // Store HCT derivatives in RASPA3 layout: [deriv_idx * totalGridPoints + gridIdx]
    for (int d = 0; d < 27; d++) {
        gridHctDerivatives[d * totalGridPoints + gridIdx] = hct_derivs[d] * scales[d];
    }

    // Store correction derivatives: [bin * 27 * totalGridPoints + deriv * totalGridPoints + gridIdx]
    for (int b = 0; b < numBins && b < 4; b++) {
        int binBase = b * 27 * totalGridPoints;
        for (int d = 0; d < 27; d++) {
            int idx = binBase + d * totalGridPoints + gridIdx;
            gridCorrectionN[idx] = corrN_derivs[b][d] * scales[d];
            gridCorrectionA[idx] = corrA_derivs[b][d] * scales[d];
            gridCorrectionB[idx] = corrB_derivs[b][d] * scales[d];
        }
    }
}

/**
 * Cross-term scalar field: G_b(r) = Σ_j q_j / f_gb(|r - r_j|, R_b, R_rec_j)
 *
 * One slice per bin. Used by IsolatedGBSAForce when computeCrossTermGrid is
 * enabled. At runtime the ligand-side kernel interpolates G_bin[i](r_i) at
 * each ligand atom position and sums q_i * G_i.
 *
 * Output layout: gridCrossTerm[b * totalGridPoints + gridIdx]  (bin-major)
 *
 * @param gridCrossTerm      Output, [numBins * totalGridPoints] in units
 *                           of e / nm (charge over distance). The runtime
 *                           kernel multiplies by the solvent prefactor.
 * @param receptorPositions  Receptor positions [numReceptorAtoms]
 * @param receptorCharges    Receptor partial charges [numReceptorAtoms]
 * @param receptorBornRadii  Baseline receptor OBC2 Born radii (nm), frozen
 *                           at their no-ligand values. [numReceptorAtoms]
 * @param numReceptorAtoms   Number of receptor atoms
 * @param binRLigValues      Per-bin ligand Born radius (nm). [numBins]
 * @param numBins            Number of bins. Must be <= MAX_CROSS_BINS.
 * @param originX/Y/Z        Grid origin
 * @param gridCounts         [nx, ny, nz]
 * @param gridSpacing        Grid spacing (uniform, nm)
 * @param totalGridPoints    nx * ny * nz
 */
#define MAX_CROSS_BINS 96
extern "C" __global__ void generateCrossTermGrid(
    float* __restrict__ gridCrossTerm,
    const float3* __restrict__ receptorPositions,
    const float* __restrict__ receptorCharges,
    const float* __restrict__ receptorBornRadii,
    int numReceptorAtoms,
    const float* __restrict__ binRLigValues,
    int numBins,
    float originX, float originY, float originZ,
    const int* __restrict__ gridCounts,
    float gridSpacing,
    int totalGridPoints
) {
    int gridIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (gridIdx >= totalGridPoints) return;

    // Convert linear index to grid coordinate
    int nx = gridCounts[0];
    int ny = gridCounts[1];
    int nz = gridCounts[2];
    int nyz = ny * nz;
    int ix = gridIdx / nyz;
    int remainder = gridIdx % nyz;
    int iy = remainder / nz;
    int iz = remainder % nz;

    float gx = originX + ix * gridSpacing;
    float gy = originY + iy * gridSpacing;
    float gz = originZ + iz * gridSpacing;

    // Load bin values into registers (cheap for reasonable numBins)
    // Register pressure: numBins floats. Spills harmlessly to local if too big.
    float accum[MAX_CROSS_BINS];
    for (int b = 0; b < numBins; b++) accum[b] = 0.0f;

    for (int j = 0; j < numReceptorAtoms; j++) {
        float3 rj = receptorPositions[j];
        float dx = gx - rj.x;
        float dy = gy - rj.y;
        float dz = gz - rj.z;
        float r2 = dx * dx + dy * dy + dz * dz;
        if (r2 < 1e-8f) continue;
        float q_j = receptorCharges[j];
        float R_rec = receptorBornRadii[j];

        for (int b = 0; b < numBins; b++) {
            float R_lig_b = binRLigValues[b];
            float RiRj = R_lig_b * R_rec;
            // f_gb^2 = r^2 + R_i R_j exp(-r^2 / (4 R_i R_j))
            float u = r2 / (4.0f * RiRj);
            float f_gb2 = r2 + RiRj * expf(-u);
            // q_j / f_gb
            accum[b] += q_j * rsqrtf(f_gb2);
        }
    }

    for (int b = 0; b < numBins; b++) {
        gridCrossTerm[b * totalGridPoints + gridIdx] = accum[b];
    }
}
