/**
 * Compute isolated nonbonded interactions for a subset of particles.
 * This kernel computes all pairwise Coulomb + LJ interactions within
 * a specified set of particles, with no interaction outside that set.
 */

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
    mixed* __restrict__ energyBuffer,           // Energy accumulator
    const int* __restrict__ particleIndices,    // Which particles this force applies to [numAtoms]
    const real* __restrict__ charges,           // Partial charges [numAtoms]
    const real* __restrict__ sigmas,            // LJ sigma [numAtoms]
    const real* __restrict__ epsilons,          // LJ epsilon [numAtoms]
    const int2* __restrict__ exclusions,        // Excluded pairs [numExclusions]
    const int2* __restrict__ exceptions,        // Exception pairs [numExceptions]
    const float3* __restrict__ exceptionParams, // Exception parameters (chargeProd, sigma, epsilon) [numExceptions]
    const int numAtoms,
    const int numPairs,
    const int paddedNumAtoms,
    const bool includeEnergy) {

    const real ONE_4PI_EPS0 = 138.935456f;  // kJ*nm/(mol*e^2)

    // Each thread handles one pair
    int pairIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pairIdx >= numPairs) return;

    // Decode pair index to atom indices within this ligand
    int i, j;
    decodePairIndex(pairIdx, &i, &j, numAtoms);

    // Check if this pair is excluded
    bool excluded = false;
#if NUM_EXCLUSIONS > 0
    for (int k = 0; k < NUM_EXCLUSIONS; k++) {
        int2 excl = exclusions[k];
        if ((excl.x == i && excl.y == j) || (excl.x == j && excl.y == i)) {
            excluded = true;
            break;
        }
    }
#endif
    if (excluded) return;

    // Check if this pair is an exception (1-4 interaction with custom parameters)
    bool isException = false;
    real qq, sigma, epsilon;
#if NUM_EXCEPTIONS > 0
    for (int k = 0; k < NUM_EXCEPTIONS; k++) {
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
#endif

    // If not an exception, use standard combining rules
    if (!isException) {
        qq = charges[i] * charges[j];
        sigma = (sigmas[i] + sigmas[j]) * 0.5f;  // Arithmetic mean
        epsilon = SQRT(epsilons[i] * epsilons[j]);  // Geometric mean
    }

    // Get actual particle indices in the System
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
    real coulombEnergy = ONE_4PI_EPS0 * qq * invR;

    // Lennard-Jones interaction
    real sig_r = sigma * invR;
    real sig_r2 = sig_r * sig_r;
    real sig_r6 = sig_r2 * sig_r2 * sig_r2;
    real sig_r12 = sig_r6 * sig_r6;
    real ljEnergy = 4.0f * epsilon * (sig_r12 - sig_r6);

    // Total energy
    real pairEnergy = coulombEnergy + ljEnergy;

    // Compute force magnitude: dE/dr
    real coulombForce = coulombEnergy * invR;
    real ljForce = 4.0f * epsilon * (12.0f * sig_r12 - 6.0f * sig_r6) * invR;
    real forceMag = (coulombForce + ljForce) * invR;  // multiply by 1/r for direction

    // Force components
    real fx = forceMag * dx;
    real fy = forceMag * dy;
    real fz = forceMag * dz;

    // Accumulate forces (Newton's third law: equal and opposite)
    // Use OpenMM's force buffer format (fixed-point atomicAdd)
    atomicAdd(&forceBuffers[particleI], static_cast<unsigned long long>((long long)(fx * 0x100000000)));
    atomicAdd(&forceBuffers[particleI + paddedNumAtoms], static_cast<unsigned long long>((long long)(fy * 0x100000000)));
    atomicAdd(&forceBuffers[particleI + 2*paddedNumAtoms], static_cast<unsigned long long>((long long)(fz * 0x100000000)));

    atomicAdd(&forceBuffers[particleJ], static_cast<unsigned long long>((long long)(-fx * 0x100000000)));
    atomicAdd(&forceBuffers[particleJ + paddedNumAtoms], static_cast<unsigned long long>((long long)(-fy * 0x100000000)));
    atomicAdd(&forceBuffers[particleJ + 2*paddedNumAtoms], static_cast<unsigned long long>((long long)(-fz * 0x100000000)));

    // Accumulate energy
    if (includeEnergy) {
        atomicAdd(energyBuffer, (mixed)pairEnergy);
    }
}
