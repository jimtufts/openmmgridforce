/**
 * CUDA implementation of per-atom Hessian analysis for normal modes.
 *
 * This kernel computes eigendecomposition, curvature metrics, and entropy
 * estimates from the 3x3 Hessian blocks computed by gridHessian.cu.
 *
 * Uses Cardano's method for analytical eigenvalue computation of 3x3 symmetric
 * matrices, which is exact and avoids iterative methods.
 */

// PI constant (NVRTC doesn't provide math_constants.h)
#define M_PI_F 3.14159265358979323846f

// ============================================================
// 3x3 Symmetric Eigendecomposition (Cardano's Method)
// ============================================================

/**
 * Compute eigenvalues of a 3x3 symmetric matrix using Cardano's formula.
 *
 * For symmetric matrices, all eigenvalues are real and this method is
 * numerically stable. Returns eigenvalues sorted in ascending order.
 *
 * @param dxx, dyy, dzz  Diagonal elements of Hessian
 * @param dxy, dxz, dyz  Off-diagonal elements of Hessian
 * @param lambda         Output: 3 eigenvalues, sorted ascending
 */
__device__ void eigenvalues_3x3_symmetric(
    real dxx, real dyy, real dzz,
    real dxy, real dxz, real dyz,
    real* __restrict__ lambda
) {
    // Construct invariants
    real trace = dxx + dyy + dzz;
    real q = trace / 3.0f;

    // Shift matrix: A = H - qI (makes trace zero)
    real a00 = dxx - q;
    real a11 = dyy - q;
    real a22 = dzz - q;

    // p² = (1/6) * ||A||²_F  (Frobenius norm of shifted matrix)
    real p2 = (a00*a00 + a11*a11 + a22*a22 +
                2.0f*(dxy*dxy + dxz*dxz + dyz*dyz)) / 6.0f;
    real p = sqrt(p2);

    if (p < 1e-10f) {
        // Matrix is already diagonal (or zero)
        lambda[0] = lambda[1] = lambda[2] = q;
        return;
    }

    // B = (1/p) * A
    real inv_p = 1.0f / p;
    real b00 = a00 * inv_p;
    real b11 = a11 * inv_p;
    real b22 = a22 * inv_p;
    real b01 = dxy * inv_p;
    real b02 = dxz * inv_p;
    real b12 = dyz * inv_p;

    // r = det(B) / 2
    real detB = b00 * (b11*b22 - b12*b12)
               - b01 * (b01*b22 - b12*b02)
               + b02 * (b01*b12 - b11*b02);
    real r = detB * 0.5f;

    // Clamp r to [-1, 1] for numerical stability
    r = fmin((real)1.0, fmax((real)-1.0, r));

    // phi = arccos(r) / 3
    real phi = acos(r) / 3.0f;

    // Eigenvalues of shifted matrix (sorted descending from cosine formula)
    real eig0 = 2.0f * p * cos(phi);
    real eig1 = 2.0f * p * cos(phi - 2.0f * (real)3.14159265358979323846 / 3.0f);
    real eig2 = 2.0f * p * cos(phi + 2.0f * (real)3.14159265358979323846 / 3.0f);

    // Shift back and sort ascending
    // Note: eig0 >= eig1 >= eig2 from the cosine formula
    lambda[0] = eig2 + q;  // smallest
    lambda[1] = eig1 + q;  // middle
    lambda[2] = eig0 + q;  // largest
}


/**
 * Compute eigenvector for a given eigenvalue using cross-product method.
 *
 * For a 3x3 symmetric matrix H and eigenvalue λ, finds the null space of (H - λI)
 * using cross products of rows, which is robust for non-degenerate cases.
 *
 * @param dxx, dyy, dzz, dxy, dxz, dyz  Hessian components
 * @param lambda                         Eigenvalue to compute eigenvector for
 * @param v                             Output: normalized eigenvector
 */
__device__ void eigenvector_for_eigenvalue(
    real dxx, real dyy, real dzz,
    real dxy, real dxz, real dyz,
    real lambda,
    real* __restrict__ v
) {
    // Compute (H - λI) and find null space via cross product of rows
    real a00 = dxx - lambda;
    real a11 = dyy - lambda;
    real a22 = dzz - lambda;

    // Row 0: (a00, dxy, dxz)
    // Row 1: (dxy, a11, dyz)
    // Row 2: (dxz, dyz, a22)

    // Try cross product of row 0 and row 1
    real v0 = dxy * dyz - a11 * dxz;
    real v1 = dxz * dxy - a00 * dyz;
    real v2 = a00 * a11 - dxy * dxy;

    real norm = sqrt(v0*v0 + v1*v1 + v2*v2);

    if (norm < 1e-10f) {
        // Degenerate case: try row 0 × row 2
        v0 = dxy * a22 - dyz * dxz;
        v1 = dxz * dxz - a00 * a22;
        v2 = a00 * dyz - dxz * dxy;
        norm = sqrt(v0*v0 + v1*v1 + v2*v2);
    }

    if (norm < 1e-10f) {
        // Still degenerate: try row 1 × row 2
        v0 = a11 * a22 - dyz * dyz;
        v1 = dyz * dxz - dxy * a22;
        v2 = dxy * dyz - a11 * dxz;
        norm = sqrt(v0*v0 + v1*v1 + v2*v2);
    }

    if (norm < 1e-10f) {
        // Fallback for highly degenerate case (isotropic Hessian)
        v[0] = 1.0f; v[1] = 0.0f; v[2] = 0.0f;
        return;
    }

    real inv_norm = 1.0f / norm;
    v[0] = v0 * inv_norm;
    v[1] = v1 * inv_norm;
    v[2] = v2 * inv_norm;
}


// ============================================================
// Analysis Kernel: Eigenvalues, Curvature, Anisotropy, Entropy
// ============================================================

/**
 * Analyze Hessian blocks to compute per-atom metrics.
 *
 * For each atom, computes:
 * - Eigenvalues (sorted ascending)
 * - Eigenvectors (optional)
 * - Mean, total, and Gaussian curvature
 * - Fractional anisotropy (0=isotropic, 1=linear)
 * - Harmonic entropy estimate
 * - Number of negative eigenvalues (saddle point indicator)
 *
 * @param hessianBlocks      Input: [6 * numAtoms] packed blocks
 * @param eigenvalues        Output: [3 * numAtoms]
 * @param eigenvectors       Output: [9 * numAtoms] (optional, can be null)
 * @param meanCurvature      Output: [numAtoms]
 * @param totalCurvature     Output: [numAtoms]
 * @param gaussianCurvature  Output: [numAtoms]
 * @param fracAnisotropy     Output: [numAtoms]
 * @param entropy            Output: [numAtoms]
 * @param minEigenvalue      Output: [numAtoms]
 * @param numNegative        Output: [numAtoms]
 * @param kT                 k_B * T in energy units (kJ/mol)
 * @param numAtoms           Number of atoms to process
 */
extern "C" __global__ void analyzeHessianKernel(
    const mixed* __restrict__ hessianBlocks,
    mixed* __restrict__ eigenvalues,
    mixed* __restrict__ eigenvectors,
    mixed* __restrict__ meanCurvature,
    mixed* __restrict__ totalCurvature,
    mixed* __restrict__ gaussianCurvature,
    mixed* __restrict__ fracAnisotropy,
    mixed* __restrict__ entropy,
    mixed* __restrict__ minEigenvalue,
    int* __restrict__ numNegative,
    float kT,
    int numAtoms
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numAtoms) return;

    // Load Hessian block
    int base = 6 * idx;
    real dxx = hessianBlocks[base + 0];
    real dyy = hessianBlocks[base + 1];
    real dzz = hessianBlocks[base + 2];
    real dxy = hessianBlocks[base + 3];
    real dxz = hessianBlocks[base + 4];
    real dyz = hessianBlocks[base + 5];

    // Compute eigenvalues
    real lambda[3];
    eigenvalues_3x3_symmetric(dxx, dyy, dzz, dxy, dxz, dyz, lambda);

    // Store eigenvalues
    eigenvalues[3*idx + 0] = lambda[0];
    eigenvalues[3*idx + 1] = lambda[1];
    eigenvalues[3*idx + 2] = lambda[2];

    // Compute eigenvectors if requested
    if (eigenvectors != 0) {
        real v[3];
        for (int i = 0; i < 3; i++) {
            eigenvector_for_eigenvalue(dxx, dyy, dzz, dxy, dxz, dyz, lambda[i], v);
            eigenvectors[9*idx + 3*i + 0] = v[0];
            eigenvectors[9*idx + 3*i + 1] = v[1];
            eigenvectors[9*idx + 3*i + 2] = v[2];
        }
    }

    // ---- Curvature metrics ----

    real mean = (lambda[0] + lambda[1] + lambda[2]) / 3.0f;
    real total = lambda[0] + lambda[1] + lambda[2];
    real gauss = lambda[0] * lambda[1] * lambda[2];

    meanCurvature[idx] = mean;
    totalCurvature[idx] = total;
    gaussianCurvature[idx] = gauss;
    minEigenvalue[idx] = lambda[0];

    // Count negative eigenvalues
    int nNeg = (lambda[0] < 0.0f) + (lambda[1] < 0.0f) + (lambda[2] < 0.0f);
    numNegative[idx] = nNeg;

    // ---- Fractional anisotropy ----
    // FA = sqrt[ sum(lambda_i - mean)^2 / (2 * sum(lambda_i^2)) ]

    real diff0 = lambda[0] - mean;
    real diff1 = lambda[1] - mean;
    real diff2 = lambda[2] - mean;
    real numerator = diff0*diff0 + diff1*diff1 + diff2*diff2;
    real denominator = lambda[0]*lambda[0] + lambda[1]*lambda[1] + lambda[2]*lambda[2];

    real fa = 0.0f;
    if (denominator > 1e-20f) {
        fa = sqrt(numerator / (2.0f * denominator));
    }
    fracAnisotropy[idx] = fa;

    // ---- Entropy (harmonic approximation) ----
    // S = (k_B/2) * sum_i [1 + ln(2π k_B T / λ_i)]
    // Only valid if all eigenvalues positive

    if (lambda[0] > 1e-10f && lambda[1] > 1e-10f && lambda[2] > 1e-10f) {
        real two_pi_kT = 2.0f * M_PI_F * kT;
        real S = 0.0f;
        S += 0.5f * (1.0f + log(two_pi_kT / lambda[0]));
        S += 0.5f * (1.0f + log(two_pi_kT / lambda[1]));
        S += 0.5f * (1.0f + log(two_pi_kT / lambda[2]));
        entropy[idx] = S;
    } else {
        entropy[idx] = nanf("");  // Undefined at saddle points
    }
}


// ============================================================
// Reduction Kernel: Sum entropy across atoms
// ============================================================

/**
 * Sum per-atom entropy values to get total configurational entropy.
 * Skips NaN values (atoms at saddle points).
 *
 * @param entropy       Input: [numAtoms] per-atom entropy
 * @param totalEntropy  Output: [1] sum of valid entropies
 * @param numAtoms      Number of atoms
 */
extern "C" __global__ void sumEntropyKernel(
    const mixed* __restrict__ entropy,
    mixed* __restrict__ totalEntropy,
    int numAtoms
) {
    extern __shared__ mixed sdata[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load and accumulate (skip NaN values)
    real val = 0.0f;
    if (idx < numAtoms) {
        real e = entropy[idx];
        if (!isnan(e)) {
            val = e;
        }
    }
    sdata[tid] = val;
    __syncthreads();

    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write block result
    if (tid == 0) {
        atomicAdd(totalEntropy, sdata[0]);
    }
}


// ============================================================
// Count Valid Atoms Kernel: Count atoms with positive eigenvalues
// ============================================================

/**
 * Count atoms where all eigenvalues are positive (valid for entropy).
 *
 * @param numNegative  Input: [numAtoms] count of negative eigenvalues per atom
 * @param validCount   Output: [1] number of atoms with all positive eigenvalues
 * @param numAtoms     Number of atoms
 */
extern "C" __global__ void countValidAtomsKernel(
    const int* __restrict__ numNegative,
    int* __restrict__ validCount,
    int numAtoms
) {
    extern __shared__ int sdata_int[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Count valid atoms in this thread
    int count = 0;
    if (idx < numAtoms) {
        if (numNegative[idx] == 0) {
            count = 1;
        }
    }
    sdata_int[tid] = count;
    __syncthreads();

    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata_int[tid] += sdata_int[tid + s];
        }
        __syncthreads();
    }

    // Write block result
    if (tid == 0) {
        atomicAdd(validCount, sdata_int[0]);
    }
}
