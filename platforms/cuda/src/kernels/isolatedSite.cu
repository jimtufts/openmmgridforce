/**
 * Compute isolated site restraint for multiple particle groups.
 *
 * Flat-bottom sphere: E = 0.5 * k * max(0, r_com - maxR)^2
 * where r_com = |COM - center| with mass-weighted COM.
 *
 * Each thread handles one (group, atom) pair using grid-stride loops.
 * Phase 1: Compute per-group COM via atomicAdd into shared memory.
 * Phase 2: Compute energy and distribute force to each atom.
 *
 * Design: One block per group. Threads within the block cooperate on
 * COM reduction, then each thread applies force to its atom(s).
 * This avoids the blocks/threads cap bug since we use grid-stride over
 * groups (not fixed thread assignment).
 *
 * IMPORTANT: Uses grid-stride loop pattern to handle any number of groups,
 * even when numGroups > gridDim.x. Uses atomicAdd for energy accumulation
 * to avoid buffer out-of-bounds.
 */

extern "C" __global__ void computeIsolatedSiteRestraint(
    const real4* __restrict__ posq,
    unsigned long long* __restrict__ forceBuffers,
    unsigned long long* __restrict__ fixedPointEnergy,
    const int* __restrict__ groupParticleIndices,
    const float* __restrict__ atomMasses,
    float* __restrict__ groupEnergies,
    const float* __restrict__ groupScalingFactors,
    const float globalScalingFactor,
    const float centerX,
    const float centerY,
    const float centerZ,
    const float maxRadius,
    const float forceConstant,
    const float totalMass,
    const int numAtoms,
    const int numGroups,
    const int paddedNumAtoms,
    const bool includeEnergy) {

    // Grid-stride loop over all (group, atom) pairs
    int totalWork = numGroups * numAtoms;
    for (int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
         globalIdx < totalWork;
         globalIdx += gridDim.x * blockDim.x) {

        int groupIdx = globalIdx / numAtoms;
        int atomIdx = globalIdx % numAtoms;

        float scale = globalScalingFactor * groupScalingFactors[groupIdx];
        if (scale == 0.0f) continue;

        // Get system particle index for this (group, atom)
        int particleIdx = groupParticleIndices[groupIdx * numAtoms + atomIdx];
        real4 pos = posq[particleIdx];
        float mass = atomMasses[atomIdx];

        // Mass-weighted position contribution
        float wx = mass * (float)pos.x;
        float wy = mass * (float)pos.y;
        float wz = mass * (float)pos.z;

        // Compute COM: sum mass-weighted positions over all atoms in group.
        // Since we're in a grid-stride loop, different threads may handle
        // different groups. We need a serial loop per group for COM.
        // For small numAtoms this is efficient.
        float comX = 0, comY = 0, comZ = 0;
        for (int a = 0; a < numAtoms; a++) {
            int p = groupParticleIndices[groupIdx * numAtoms + a];
            real4 apos = posq[p];
            float am = atomMasses[a];
            comX += am * (float)apos.x;
            comY += am * (float)apos.y;
            comZ += am * (float)apos.z;
        }
        float invTotalMass = 1.0f / totalMass;
        comX *= invTotalMass;
        comY *= invTotalMass;
        comZ *= invTotalMass;

        // Distance from site center
        float dx = comX - centerX;
        float dy = comY - centerY;
        float dz = comZ - centerZ;
        float r = sqrtf(dx * dx + dy * dy + dz * dz);

        // Flat-bottom: only restrain outside maxRadius
        float deltaR = r - maxRadius;
        if (deltaR <= 0.0f) continue;

        // Energy (only first atom in group accumulates to avoid double-counting)
        if (atomIdx == 0) {
            float energy = 0.5f * forceConstant * deltaR * deltaR * scale;
            if (includeEnergy) {
                atomicAdd(&groupEnergies[groupIdx], energy);
                atomicAdd(fixedPointEnergy, static_cast<unsigned long long>((long long)((double)energy * 0x100000000)));
            }
        }

        // Force on this atom: F_i = -dE/dr * (COM - center) / r * m_i / M
        if (r > 1.0e-12f) {
            float dEdR = forceConstant * deltaR * scale;
            float prefactor = -dEdR / (r * totalMass);
            float w = prefactor * mass;
            real fx = (real)(w * dx);
            real fy = (real)(w * dy);
            real fz = (real)(w * dz);

            // Fixed-point force accumulation (matches OpenMM convention)
            atomicAdd(&forceBuffers[particleIdx],
                      static_cast<unsigned long long>((long long)(fx * 0x100000000)));
            atomicAdd(&forceBuffers[particleIdx + paddedNumAtoms],
                      static_cast<unsigned long long>((long long)(fy * 0x100000000)));
            atomicAdd(&forceBuffers[particleIdx + 2 * paddedNumAtoms],
                      static_cast<unsigned long long>((long long)(fz * 0x100000000)));
        }
    }
}
