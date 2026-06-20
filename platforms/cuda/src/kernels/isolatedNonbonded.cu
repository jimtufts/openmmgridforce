/**
 * Compute isolated nonbonded interactions for a subset of particles.
 * This kernel computes all pairwise Coulomb + LJ interactions within
 * a specified set of particles, with no interaction outside that set.
 */

// Fixed-point scale for Hessian accumulation. 2^30 resolves soft eigenvalues while
// leaving ample headroom for ~10^6 kJ/mol/nm^2 entries (must match HESSIAN_SCALE_INV host-side).
#define HESSIAN_SCALE 0x40000000

// Helper function to decode linear pair index to (i,j) indices
__device__ void decodePairIndex(int pairIdx, int* i, int* j, int numAtoms) {
    // Convert linear pair index to (i,j) where i < j
    // Pairs are ordered row-major: (0,1), (0,2), ..., (0,n-1), (1,2), (1,3), ..., (1,n-1), ...
    // Row i contains (numAtoms - i - 1) pairs
    // Using inverse formula for upper triangular indexing
    float discriminant = (2.0f * numAtoms - 1.0f) * (2.0f * numAtoms - 1.0f) - 8.0f * pairIdx;
    *i = (int)floor((2.0f * numAtoms - 1.0f - sqrtf(discriminant)) / 2.0f);
    *j = pairIdx - (*i) * (2 * numAtoms - (*i) - 1) / 2 + (*i) + 1;
}

extern "C" __global__ void computeIsolatedNonbonded(
    const real4* __restrict__ posq,             // All positions in Context
    unsigned long long* __restrict__ forceBuffers,  // Force output buffers
    unsigned long long* __restrict__ fixedPointEnergy,  // Fixed-point energy accumulator
    const int* __restrict__ groupParticleIndices, // Particle indices per group [numGroups * numAtoms]
    const float* __restrict__ charges,           // Partial charges [numAtoms] (template)
    const float* __restrict__ sigmas,            // LJ sigma [numAtoms] (template)
    const float* __restrict__ epsilons,          // LJ epsilon [numAtoms] (template)
    const int2* __restrict__ exclusions,        // Excluded pairs [numExclusions] (template)
    const int2* __restrict__ exceptions,        // Exception pairs [numExceptions] (template)
    const float3* __restrict__ exceptionParams, // Exception parameters (chargeProd, sigma, epsilon) [numExceptions]
    mixed* __restrict__ groupEnergies,          // Per-group energy accumulation [numGroups]
    const float* __restrict__ groupScalingFactors, // Per-group scaling [numGroups]
    const float globalScalingFactor,            // Global scaling factor
    const int numAtoms,
    const int numPairs,
    const int numGroups,
    const int paddedNumAtoms,
    const int numExclusions,
    const int numExceptions,
    const bool includeEnergy) {

    // Coulomb constant in kJ*nm/(mol*e^2)
    // Note: Using local constant to avoid macro conflicts when kernels are concatenated
    const real COULOMB_CONST = 138.935456f;

    // Grid-stride loop: each thread handles multiple (group, pair) combinations.
    // OpenMM's executeKernel caps the grid size at numThreadBlocks (= SMs * blocksPerSM),
    // so we must loop to cover all work items when totalWork exceeds the grid capacity.
    int totalWork = numGroups * numPairs;
    for (int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
         globalIdx < totalWork;
         globalIdx += gridDim.x * blockDim.x) {

        int groupIdx = globalIdx / numPairs;
        int pairIdx = globalIdx % numPairs;

        // Get scaling factor for this group
        float scale = globalScalingFactor * groupScalingFactors[groupIdx];
        if (scale == 0.0f) continue;

        // Pointer to this group's particle indices
        const int* particleIndices = groupParticleIndices + groupIdx * numAtoms;

        // Decode pair index to atom indices within the template
        int i, j;
        decodePairIndex(pairIdx, &i, &j, numAtoms);

        // Check if this pair is excluded (runtime numExclusions; loop is no-op when 0)
        bool excluded = false;
        for (int k = 0; k < numExclusions; k++) {
            int2 excl = exclusions[k];
            if ((excl.x == i && excl.y == j) || (excl.x == j && excl.y == i)) {
                excluded = true;
                break;
            }
        }
        if (excluded) continue;

        // Check if this pair is an exception (1-4 interaction with custom parameters)
        bool isException = false;
        real qq, sigma, epsilon;
        for (int k = 0; k < numExceptions; k++) {
            int2 exc = exceptions[k];
            if ((exc.x == i && exc.y == j) || (exc.x == j && exc.y == i)) {
                isException = true;
                float3 params = exceptionParams[k];
                qq = params.x;          // chargeProd
                sigma = params.y;       // sigma
                epsilon = params.z;     // epsilon
                break;
            }
        }

        // If not an exception, use standard combining rules
        if (!isException) {
            qq = charges[i] * charges[j];
            sigma = (sigmas[i] + sigmas[j]) * 0.5f;  // Arithmetic mean
            epsilon = SQRT(epsilons[i] * epsilons[j]);  // Geometric mean
        }

        // Get actual particle indices in the System for this group
        int particleI = particleIndices[i];
        int particleJ = particleIndices[j];

        // Load positions
        real4 posqI = posq[particleI];
        real4 posqJ = posq[particleJ];

        // Compute distance
        real dx = posqI.x - posqJ.x;
        real dy = posqI.y - posqJ.y;
        real dz = posqI.z - posqJ.z;
        real r2 = dx*dx + dy*dy + dz*dz;
        real invR = RSQRT(r2);
        real r = r2 * invR;

        // Coulomb interaction
        real coulombEnergy = COULOMB_CONST * qq * invR;

        // Lennard-Jones interaction
        real sig_r = sigma * invR;
        real sig_r2 = sig_r * sig_r;
        real sig_r6 = sig_r2 * sig_r2 * sig_r2;
        real sig_r12 = sig_r6 * sig_r6;
        real ljEnergy = 4.0f * epsilon * (sig_r12 - sig_r6);

        // Total energy (scaled)
        real pairEnergy = (coulombEnergy + ljEnergy) * scale;

        // Compute force: F = -dE/dr (scaled)
        real coulombForce = coulombEnergy * invR * scale;
        real ljForce = 4.0f * epsilon * (12.0f * sig_r12 - 6.0f * sig_r6) * invR * scale;
        real forceMagnitude = coulombForce + ljForce;

        // Force components: F_vec = forceMagnitude * (r_vec/|r|)
        real fx = forceMagnitude * dx * invR;
        real fy = forceMagnitude * dy * invR;
        real fz = forceMagnitude * dz * invR;

        // Accumulate forces (Newton's third law: equal and opposite)
        atomicAdd(&forceBuffers[particleI], static_cast<unsigned long long>((long long)(fx * 0x100000000)));
        atomicAdd(&forceBuffers[particleI + paddedNumAtoms], static_cast<unsigned long long>((long long)(fy * 0x100000000)));
        atomicAdd(&forceBuffers[particleI + 2*paddedNumAtoms], static_cast<unsigned long long>((long long)(fz * 0x100000000)));

        atomicAdd(&forceBuffers[particleJ], static_cast<unsigned long long>((long long)(-fx * 0x100000000)));
        atomicAdd(&forceBuffers[particleJ + paddedNumAtoms], static_cast<unsigned long long>((long long)(-fy * 0x100000000)));
        atomicAdd(&forceBuffers[particleJ + 2*paddedNumAtoms], static_cast<unsigned long long>((long long)(-fz * 0x100000000)));

        // Accumulate energy (fixed-point for pre-sm_60 GPU compatibility)
        if (includeEnergy) {
            atomicAdd(fixedPointEnergy, static_cast<unsigned long long>((long long)((double)pairEnergy * 0x100000000)));
            atomicAdd(&groupEnergies[groupIdx], pairEnergy);
        }
    }
}

/**
 * Compute Hessian (second derivatives) for isolated nonbonded interactions.
 *
 * For a pairwise potential E(r), the Cartesian Hessian has structure:
 *   H_αβ^{ii} = (d²E/dr² - (1/r)dE/dr) * r_α*r_β/r² + (1/r)dE/dr * δ_αβ
 *   H_αβ^{ij} = -H_αβ^{ii}
 *   H_αβ^{jj} = H_αβ^{ii}
 *
 * Output is stored as 3x3 blocks: hessianBlocks[atomI*numAtoms + atomJ] contains
 * the 9 elements of the 3x3 block for atoms (i,j) in row-major order.
 */
extern "C" __global__ void computeIsolatedNonbondedHessians(
    const real4* __restrict__ posq,             // All positions in Context
    const int* __restrict__ particleIndices,    // Which particles this force applies to [numAtoms]
    const float* __restrict__ charges,           // Partial charges [numAtoms]
    const float* __restrict__ sigmas,            // LJ sigma [numAtoms]
    const float* __restrict__ epsilons,          // LJ epsilon [numAtoms]
    const int2* __restrict__ exclusions,        // Excluded pairs [numExclusions]
    const int2* __restrict__ exceptions,        // Exception pairs [numExceptions]
    const float3* __restrict__ exceptionParams, // Exception parameters (chargeProd, sigma, epsilon) [numExceptions]
    unsigned long long* __restrict__ hessianBlocks, // Output: fixed-point 3x3 blocks [numAtoms*numAtoms*9]
    const int numAtoms,
    const int numPairs,
    const int paddedNumAtoms,
    const int numExclusions,
    const int numExceptions) {

    // Coulomb constant in kJ*nm/(mol*e^2)
    const real COULOMB_CONST = 138.935456f;

    // Each thread handles one pair
    int pairIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pairIdx >= numPairs) return;

    // Decode pair index to atom indices within this ligand
    int i, j;
    decodePairIndex(pairIdx, &i, &j, numAtoms);

    // Check if this pair is excluded (runtime numExclusions; loop is no-op when 0)
    bool excluded = false;
    for (int k = 0; k < numExclusions; k++) {
        int2 excl = exclusions[k];
        if ((excl.x == i && excl.y == j) || (excl.x == j && excl.y == i)) {
            excluded = true;
            break;
        }
    }
    if (excluded) return;

    // Check if this pair is an exception (1-4 interaction with custom parameters)
    bool isException = false;
    real qq, sigma, epsilon;
    for (int k = 0; k < numExceptions; k++) {
        int2 exc = exceptions[k];
        if ((exc.x == i && exc.y == j) || (exc.x == j && exc.y == i)) {
            isException = true;
            float3 params = exceptionParams[k];
            qq = params.x;          // chargeProd
            sigma = params.y;       // sigma
            epsilon = params.z;     // epsilon
            break;
        }
    }

    // If not an exception, use standard combining rules
    if (!isException) {
        qq = charges[i] * charges[j];
        sigma = (sigmas[i] + sigmas[j]) * 0.5f;  // Arithmetic mean
        epsilon = SQRT(epsilons[i] * epsilons[j]);  // Geometric mean
    }

    // Get actual particle indices in the System
    int particleI = particleIndices[i];
    int particleJ = particleIndices[j];

    real4 posqI = posq[particleI];
    real4 posqJ = posq[particleJ];

    // Compute distance vector (pointing from j to i)
    real dx = posqI.x - posqJ.x;
    real dy = posqI.y - posqJ.y;
    real dz = posqI.z - posqJ.z;
    real r2 = dx*dx + dy*dy + dz*dz;
    real invR2 = 1.0f / r2;
    real invR = SQRT(invR2);
    real r = r2 * invR;

    // Compute dE/dr and d²E/dr² for LJ
    // E_LJ = 4ε[(σ/r)¹² - (σ/r)⁶]
    // dE/dr = 4ε[-12σ¹²/r¹³ + 6σ⁶/r⁷]
    // d²E/dr² = 4ε[156σ¹²/r¹⁴ - 42σ⁶/r⁸]
    real sig_r = sigma * invR;
    real sig_r2 = sig_r * sig_r;
    real sig_r6 = sig_r2 * sig_r2 * sig_r2;
    real sig_r12 = sig_r6 * sig_r6;

    real dE_dr_LJ = 4.0f * epsilon * (-12.0f * sig_r12 + 6.0f * sig_r6) * invR;
    real d2E_dr2_LJ = 4.0f * epsilon * (156.0f * sig_r12 - 42.0f * sig_r6) * invR2;

    // Compute dE/dr and d²E/dr² for Coulomb
    // E_c = k*qq/r
    // dE/dr = -k*qq/r²
    // d²E/dr² = 2k*qq/r³
    real dE_dr_C = -COULOMB_CONST * qq * invR2;
    real d2E_dr2_C = 2.0f * COULOMB_CONST * qq * invR2 * invR;

    // Total derivatives
    real dE_dr = dE_dr_LJ + dE_dr_C;
    real d2E_dr2 = d2E_dr2_LJ + d2E_dr2_C;

    // Hessian formula coefficients:
    // H_αβ = term1 * r_α*r_β/r² + term2 * δ_αβ
    // where term1 = d²E/dr² - (1/r)dE/dr
    //       term2 = (1/r)dE/dr
    real term1 = d2E_dr2 - dE_dr * invR;
    real term2 = dE_dr * invR;

    // Normalized direction components
    real nx = dx * invR;
    real ny = dy * invR;
    real nz = dz * invR;

    // Compute 3x3 Hessian block for atom pair (i,i) contribution from this pair
    // H^{ii}_αβ = term1 * n_α*n_β + term2 * δ_αβ
    real Hxx = term1 * nx * nx + term2;
    real Hyy = term1 * ny * ny + term2;
    real Hzz = term1 * nz * nz + term2;
    real Hxy = term1 * nx * ny;
    real Hxz = term1 * nx * nz;
    real Hyz = term1 * ny * nz;

    // Accumulate to diagonal blocks H[i,i] and H[j,j]
    // Block index for (i,i): i*numAtoms + i
    // Block index for (j,j): j*numAtoms + j
    int blockII = (i * numAtoms + i) * 9;
    int blockJJ = (j * numAtoms + j) * 9;

    // H[i,i] += H_pair (symmetric 3x3) - using fixed-point for determinism
    atomicAdd(&hessianBlocks[blockII + 0], static_cast<unsigned long long>((long long)(Hxx * HESSIAN_SCALE)));
    atomicAdd(&hessianBlocks[blockII + 1], static_cast<unsigned long long>((long long)(Hxy * HESSIAN_SCALE)));
    atomicAdd(&hessianBlocks[blockII + 2], static_cast<unsigned long long>((long long)(Hxz * HESSIAN_SCALE)));
    atomicAdd(&hessianBlocks[blockII + 3], static_cast<unsigned long long>((long long)(Hxy * HESSIAN_SCALE)));
    atomicAdd(&hessianBlocks[blockII + 4], static_cast<unsigned long long>((long long)(Hyy * HESSIAN_SCALE)));
    atomicAdd(&hessianBlocks[blockII + 5], static_cast<unsigned long long>((long long)(Hyz * HESSIAN_SCALE)));
    atomicAdd(&hessianBlocks[blockII + 6], static_cast<unsigned long long>((long long)(Hxz * HESSIAN_SCALE)));
    atomicAdd(&hessianBlocks[blockII + 7], static_cast<unsigned long long>((long long)(Hyz * HESSIAN_SCALE)));
    atomicAdd(&hessianBlocks[blockII + 8], static_cast<unsigned long long>((long long)(Hzz * HESSIAN_SCALE)));

    // H[j,j] += H_pair (same as H[i,i] contribution)
    atomicAdd(&hessianBlocks[blockJJ + 0], static_cast<unsigned long long>((long long)(Hxx * HESSIAN_SCALE)));
    atomicAdd(&hessianBlocks[blockJJ + 1], static_cast<unsigned long long>((long long)(Hxy * HESSIAN_SCALE)));
    atomicAdd(&hessianBlocks[blockJJ + 2], static_cast<unsigned long long>((long long)(Hxz * HESSIAN_SCALE)));
    atomicAdd(&hessianBlocks[blockJJ + 3], static_cast<unsigned long long>((long long)(Hxy * HESSIAN_SCALE)));
    atomicAdd(&hessianBlocks[blockJJ + 4], static_cast<unsigned long long>((long long)(Hyy * HESSIAN_SCALE)));
    atomicAdd(&hessianBlocks[blockJJ + 5], static_cast<unsigned long long>((long long)(Hyz * HESSIAN_SCALE)));
    atomicAdd(&hessianBlocks[blockJJ + 6], static_cast<unsigned long long>((long long)(Hxz * HESSIAN_SCALE)));
    atomicAdd(&hessianBlocks[blockJJ + 7], static_cast<unsigned long long>((long long)(Hyz * HESSIAN_SCALE)));
    atomicAdd(&hessianBlocks[blockJJ + 8], static_cast<unsigned long long>((long long)(Hzz * HESSIAN_SCALE)));

    // Off-diagonal blocks: H[i,j] = H[j,i] = -H_pair
    int blockIJ = (i * numAtoms + j) * 9;
    int blockJI = (j * numAtoms + i) * 9;

    atomicAdd(&hessianBlocks[blockIJ + 0], static_cast<unsigned long long>((long long)(-Hxx * HESSIAN_SCALE)));
    atomicAdd(&hessianBlocks[blockIJ + 1], static_cast<unsigned long long>((long long)(-Hxy * HESSIAN_SCALE)));
    atomicAdd(&hessianBlocks[blockIJ + 2], static_cast<unsigned long long>((long long)(-Hxz * HESSIAN_SCALE)));
    atomicAdd(&hessianBlocks[blockIJ + 3], static_cast<unsigned long long>((long long)(-Hxy * HESSIAN_SCALE)));
    atomicAdd(&hessianBlocks[blockIJ + 4], static_cast<unsigned long long>((long long)(-Hyy * HESSIAN_SCALE)));
    atomicAdd(&hessianBlocks[blockIJ + 5], static_cast<unsigned long long>((long long)(-Hyz * HESSIAN_SCALE)));
    atomicAdd(&hessianBlocks[blockIJ + 6], static_cast<unsigned long long>((long long)(-Hxz * HESSIAN_SCALE)));
    atomicAdd(&hessianBlocks[blockIJ + 7], static_cast<unsigned long long>((long long)(-Hyz * HESSIAN_SCALE)));
    atomicAdd(&hessianBlocks[blockIJ + 8], static_cast<unsigned long long>((long long)(-Hzz * HESSIAN_SCALE)));

    atomicAdd(&hessianBlocks[blockJI + 0], static_cast<unsigned long long>((long long)(-Hxx * HESSIAN_SCALE)));
    atomicAdd(&hessianBlocks[blockJI + 1], static_cast<unsigned long long>((long long)(-Hxy * HESSIAN_SCALE)));
    atomicAdd(&hessianBlocks[blockJI + 2], static_cast<unsigned long long>((long long)(-Hxz * HESSIAN_SCALE)));
    atomicAdd(&hessianBlocks[blockJI + 3], static_cast<unsigned long long>((long long)(-Hxy * HESSIAN_SCALE)));
    atomicAdd(&hessianBlocks[blockJI + 4], static_cast<unsigned long long>((long long)(-Hyy * HESSIAN_SCALE)));
    atomicAdd(&hessianBlocks[blockJI + 5], static_cast<unsigned long long>((long long)(-Hyz * HESSIAN_SCALE)));
    atomicAdd(&hessianBlocks[blockJI + 6], static_cast<unsigned long long>((long long)(-Hxz * HESSIAN_SCALE)));
    atomicAdd(&hessianBlocks[blockJI + 7], static_cast<unsigned long long>((long long)(-Hyz * HESSIAN_SCALE)));
    atomicAdd(&hessianBlocks[blockJI + 8], static_cast<unsigned long long>((long long)(-Hzz * HESSIAN_SCALE)));
}


// ==================== Diagonal Hessian for Riemannian Metric ====================
// Computes only the diagonal 3x3 blocks (H[i,i] and H[j,j]) for all groups.
// Same math as computeIsolatedNonbondedHessians but skips off-diagonal blocks
// and operates over all particle groups simultaneously.
//
// Output layout: diagHessian[groupIdx * numAtoms * 6 + atomIdx * 6 + {0..5}]
//   = [Hxx, Hyy, Hzz, Hxy, Hxz, Hyz] per atom per group.

__device__ void addNBDiagBlock(float* diagHessian, int groupIdx, int localAtomIdx,
                               int numAtoms, float Hxx, float Hyy, float Hzz,
                               float Hxy, float Hxz, float Hyz) {
    int base = (groupIdx * numAtoms + localAtomIdx) * 6;
    atomicAdd(&diagHessian[base + 0], Hxx);
    atomicAdd(&diagHessian[base + 1], Hyy);
    atomicAdd(&diagHessian[base + 2], Hzz);
    atomicAdd(&diagHessian[base + 3], Hxy);
    atomicAdd(&diagHessian[base + 4], Hxz);
    atomicAdd(&diagHessian[base + 5], Hyz);
}

extern "C" __global__ void computeIsolatedNonbondedDiagHessian(
    const real4* __restrict__ posq,
    const int* __restrict__ groupParticleIndices,
    const float* __restrict__ charges,
    const float* __restrict__ sigmas,
    const float* __restrict__ epsilons,
    const int2* __restrict__ exclusions,
    const int2* __restrict__ exceptions,
    const float3* __restrict__ exceptionParams,
    float* __restrict__ diagHessian,
    const int numAtoms,
    const int numPairs,
    const int numGroups,
    const int numExclusions,
    const int numExceptions) {

    const real COULOMB_CONST = 138.935456f;

    int totalWork = numGroups * numPairs;
    for (int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
         globalIdx < totalWork;
         globalIdx += gridDim.x * blockDim.x) {

        int groupIdx = globalIdx / numPairs;
        int pairIdx = globalIdx % numPairs;

        // Decode pair index to local atom indices
        int i, j;
        decodePairIndex(pairIdx, &i, &j, numAtoms);

        // Check exclusions (runtime numExclusions; loop is no-op when 0)
        bool excluded = false;
        for (int k = 0; k < numExclusions; k++) {
            int2 excl = exclusions[k];
            if ((excl.x == i && excl.y == j) || (excl.x == j && excl.y == i)) {
                excluded = true;
                break;
            }
        }
        if (excluded) continue;

        // Check exceptions
        bool isException = false;
        real qq, sigma, epsilon;
        for (int k = 0; k < numExceptions; k++) {
            int2 exc = exceptions[k];
            if ((exc.x == i && exc.y == j) || (exc.x == j && exc.y == i)) {
                isException = true;
                float3 params = exceptionParams[k];
                qq = params.x;
                sigma = params.y;
                epsilon = params.z;
                break;
            }
        }
        if (!isException) {
            qq = charges[i] * charges[j];
            sigma = (sigmas[i] + sigmas[j]) * 0.5f;
            epsilon = SQRT(epsilons[i] * epsilons[j]);
        }

        // Get actual particle positions for this group
        int particleI = groupParticleIndices[groupIdx * numAtoms + i];
        int particleJ = groupParticleIndices[groupIdx * numAtoms + j];

        real4 posqI = posq[particleI];
        real4 posqJ = posq[particleJ];

        real dx = (posqI.x - posqJ.x);
        real dy = (posqI.y - posqJ.y);
        real dz = (posqI.z - posqJ.z);
        real r2 = dx*dx + dy*dy + dz*dz;
        real invR2 = 1.0f / fmax(r2, (real)1e-20);
        real invR = sqrt(invR2);
        real r = r2 * invR;

        // LJ derivatives
        real sig_r = sigma * invR;
        real sig_r2 = sig_r * sig_r;
        real sig_r6 = sig_r2 * sig_r2 * sig_r2;
        real sig_r12 = sig_r6 * sig_r6;

        real dE_dr_LJ = 4.0f * epsilon * (-12.0f * sig_r12 + 6.0f * sig_r6) * invR;
        real d2E_dr2_LJ = 4.0f * epsilon * (156.0f * sig_r12 - 42.0f * sig_r6) * invR2;

        // Coulomb derivatives
        real dE_dr_C = -COULOMB_CONST * qq * invR2;
        real d2E_dr2_C = 2.0f * COULOMB_CONST * qq * invR2 * invR;

        real dE_dr = dE_dr_LJ + dE_dr_C;
        real d2E_dr2 = d2E_dr2_LJ + d2E_dr2_C;

        // Hessian: H_ab = term1 * n_a*n_b + term2 * delta_ab
        real term1 = d2E_dr2 - dE_dr * invR;
        real term2 = dE_dr * invR;

        real nx = dx * invR;
        real ny = dy * invR;
        real nz = dz * invR;

        real Hxx = term1 * nx * nx + term2;
        real Hyy = term1 * ny * ny + term2;
        real Hzz = term1 * nz * nz + term2;
        real Hxy = term1 * nx * ny;
        real Hxz = term1 * nx * nz;
        real Hyz = term1 * ny * nz;

        // Both atoms get the same diagonal block (H[i,i] = H[j,j] for pairwise)
        addNBDiagBlock(diagHessian, groupIdx, i, numAtoms,
                       Hxx, Hyy, Hzz, Hxy, Hxz, Hyz);
        addNBDiagBlock(diagHessian, groupIdx, j, numAtoms,
                       Hxx, Hyy, Hzz, Hxy, Hxz, Hyz);
    }
}
