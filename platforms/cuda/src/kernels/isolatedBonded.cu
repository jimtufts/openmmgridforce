/**
 * Compute isolated bonded interactions (bonds, angles, torsions) for
 * multiple particle groups. Each group is an isolated ligand replica.
 *
 * Three separate kernels: bonds, angles, torsions.
 * Each thread handles one (group, term) combination.
 * Forces accumulated atomically via fixed-point arithmetic.
 * Per-group energies accumulated atomically into groupEnergies buffer.
 *
 * Bonded math adapted from OpenMM Reference platform:
 *   ReferenceHarmonicBondIxn.cpp
 *   ReferenceAngleBondIxn.cpp
 *   ReferenceProperDihedralBond.cpp
 */

// ==================== Harmonic Bonds ====================
// E = 0.5 * k * (r - r0)^2

extern "C" __global__ void computeIsolatedBonds(
    const real4* __restrict__ posq,
    unsigned long long* __restrict__ forceBuffers,
    mixed* __restrict__ energyBuffer,
    const int* __restrict__ groupParticleIndices,
    const int2* __restrict__ bondAtoms,
    const float2* __restrict__ bondParams,
    float* __restrict__ groupEnergies,
    const float* __restrict__ groupScalingFactors,
    const float globalScalingFactor,
    const int numAtoms,
    const int numBonds,
    const int numGroups,
    const int paddedNumAtoms,
    const bool includeEnergy) {

    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalWork = numGroups * numBonds;
    if (globalIdx >= totalWork) return;

    int groupIdx = globalIdx / numBonds;
    int bondIdx = globalIdx % numBonds;

    float scale = globalScalingFactor * groupScalingFactors[groupIdx];
    if (scale == 0.0f) return;

    // Get template atom indices and map to system particles
    int2 atoms = bondAtoms[bondIdx];
    int particleI = groupParticleIndices[groupIdx * numAtoms + atoms.x];
    int particleJ = groupParticleIndices[groupIdx * numAtoms + atoms.y];

    // Load positions
    real4 posI = posq[particleI];
    real4 posJ = posq[particleJ];

    real dx = posI.x - posJ.x;
    real dy = posI.y - posJ.y;
    real dz = posI.z - posJ.z;
    real r2 = dx * dx + dy * dy + dz * dz;
    real r = SQRT(r2);

    // Bond parameters: x = length (r0), y = k
    float2 params = bondParams[bondIdx];
    real r0 = params.x;
    real k = params.y;

    real deltaR = r - r0;
    real energy = (real)0.5 * k * deltaR * deltaR * scale;

    // Accumulate energy
    if (includeEnergy) {
        atomicAdd(&groupEnergies[groupIdx], (float)energy);
        energyBuffer[blockIdx.x * blockDim.x + threadIdx.x] += energy;
    }

    // Compute and accumulate forces
    if (r > (real)1.0e-12) {
        real dEdR = k * deltaR * scale / r;
        real fx = -dEdR * dx;
        real fy = -dEdR * dy;
        real fz = -dEdR * dz;

        // Fixed-point force accumulation
        atomicAdd(&forceBuffers[particleI], static_cast<unsigned long long>((long long)(fx * 0x100000000)));
        atomicAdd(&forceBuffers[particleI + paddedNumAtoms], static_cast<unsigned long long>((long long)(fy * 0x100000000)));
        atomicAdd(&forceBuffers[particleI + 2 * paddedNumAtoms], static_cast<unsigned long long>((long long)(fz * 0x100000000)));
        atomicAdd(&forceBuffers[particleJ], static_cast<unsigned long long>((long long)(-fx * 0x100000000)));
        atomicAdd(&forceBuffers[particleJ + paddedNumAtoms], static_cast<unsigned long long>((long long)(-fy * 0x100000000)));
        atomicAdd(&forceBuffers[particleJ + 2 * paddedNumAtoms], static_cast<unsigned long long>((long long)(-fz * 0x100000000)));
    }
}

// ==================== Harmonic Angles ====================
// E = 0.5 * k * (theta - theta0)^2

extern "C" __global__ void computeIsolatedAngles(
    const real4* __restrict__ posq,
    unsigned long long* __restrict__ forceBuffers,
    mixed* __restrict__ energyBuffer,
    const int* __restrict__ groupParticleIndices,
    const int4* __restrict__ angleAtoms,
    const float2* __restrict__ angleParams,
    float* __restrict__ groupEnergies,
    const float* __restrict__ groupScalingFactors,
    const float globalScalingFactor,
    const int numAtoms,
    const int numAngles,
    const int numGroups,
    const int paddedNumAtoms,
    const bool includeEnergy) {

    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalWork = numGroups * numAngles;
    if (globalIdx >= totalWork) return;

    int groupIdx = globalIdx / numAngles;
    int angleIdx = globalIdx % numAngles;

    float scale = globalScalingFactor * groupScalingFactors[groupIdx];
    if (scale == 0.0f) return;

    int4 atoms = angleAtoms[angleIdx];
    int pA = groupParticleIndices[groupIdx * numAtoms + atoms.x];
    int pB = groupParticleIndices[groupIdx * numAtoms + atoms.y];  // central
    int pC = groupParticleIndices[groupIdx * numAtoms + atoms.z];

    real4 posA = posq[pA];
    real4 posB = posq[pB];
    real4 posC = posq[pC];

    // Vectors matching OpenMM: v0 = B-A, v1 = B-C
    real v0x = posB.x - posA.x;
    real v0y = posB.y - posA.y;
    real v0z = posB.z - posA.z;
    real rBA2 = v0x * v0x + v0y * v0y + v0z * v0z;

    real v1x = posB.x - posC.x;
    real v1y = posB.y - posC.y;
    real v1z = posB.z - posC.z;
    real rBC2 = v1x * v1x + v1y * v1y + v1z * v1z;

    // Cross product p = v0 x v1 = (B-A) x (B-C)
    real px = v0y * v1z - v0z * v1y;
    real py = v0z * v1x - v0x * v1z;
    real pz = v0x * v1y - v0y * v1x;
    real rp = SQRT(px * px + py * py + pz * pz);
    if (rp < (real)1.0e-06) rp = (real)1.0e-06;

    real dot = v0x * v1x + v0y * v1y + v0z * v1z;
    real cosine = dot * RSQRT(rBA2 * rBC2);
    if (cosine > (real)1.0) cosine = (real)1.0;
    if (cosine < (real)-1.0) cosine = (real)-1.0;

    real theta = acos(cosine);

    // Angle parameters: x = theta0, y = k
    float2 params = angleParams[angleIdx];
    real theta0 = params.x;
    real k = params.y;

    real deltaTheta = theta - theta0;
    real energy = (real)0.5 * k * deltaTheta * deltaTheta * scale;
    real dEdTheta = k * deltaTheta * scale;

    if (includeEnergy) {
        atomicAdd(&groupEnergies[groupIdx], (float)energy);
        energyBuffer[blockIdx.x * blockDim.x + threadIdx.x] += energy;
    }

    // Force decomposition from ReferenceAngleBondIxn
    real termA =  dEdTheta / (rBA2 * rp);
    real termC = -dEdTheta / (rBC2 * rp);

    // crossA = v0 x p, crossC = v1 x p
    real crossAx = v0y * pz - v0z * py;
    real crossAy = v0z * px - v0x * pz;
    real crossAz = v0x * py - v0y * px;

    real crossCx = v1y * pz - v1z * py;
    real crossCy = v1z * px - v1x * pz;
    real crossCz = v1x * py - v1y * px;

    real fAx = termA * crossAx;
    real fAy = termA * crossAy;
    real fAz = termA * crossAz;

    real fCx = termC * crossCx;
    real fCy = termC * crossCy;
    real fCz = termC * crossCz;

    // Central atom B = -(fA + fC)
    real fBx = -(fAx + fCx);
    real fBy = -(fAy + fCy);
    real fBz = -(fAz + fCz);

    // Accumulate forces
    atomicAdd(&forceBuffers[pA], static_cast<unsigned long long>((long long)(fAx * 0x100000000)));
    atomicAdd(&forceBuffers[pA + paddedNumAtoms], static_cast<unsigned long long>((long long)(fAy * 0x100000000)));
    atomicAdd(&forceBuffers[pA + 2 * paddedNumAtoms], static_cast<unsigned long long>((long long)(fAz * 0x100000000)));
    atomicAdd(&forceBuffers[pB], static_cast<unsigned long long>((long long)(fBx * 0x100000000)));
    atomicAdd(&forceBuffers[pB + paddedNumAtoms], static_cast<unsigned long long>((long long)(fBy * 0x100000000)));
    atomicAdd(&forceBuffers[pB + 2 * paddedNumAtoms], static_cast<unsigned long long>((long long)(fBz * 0x100000000)));
    atomicAdd(&forceBuffers[pC], static_cast<unsigned long long>((long long)(fCx * 0x100000000)));
    atomicAdd(&forceBuffers[pC + paddedNumAtoms], static_cast<unsigned long long>((long long)(fCy * 0x100000000)));
    atomicAdd(&forceBuffers[pC + 2 * paddedNumAtoms], static_cast<unsigned long long>((long long)(fCz * 0x100000000)));
}

// ==================== Periodic Torsions ====================
// E = k * (1 + cos(n*phi - phase))

extern "C" __global__ void computeIsolatedTorsions(
    const real4* __restrict__ posq,
    unsigned long long* __restrict__ forceBuffers,
    mixed* __restrict__ energyBuffer,
    const int* __restrict__ groupParticleIndices,
    const int4* __restrict__ torsionAtoms,
    const float4* __restrict__ torsionParams,
    float* __restrict__ groupEnergies,
    const float* __restrict__ groupScalingFactors,
    const float globalScalingFactor,
    const int numAtoms,
    const int numTorsions,
    const int numGroups,
    const int paddedNumAtoms,
    const bool includeEnergy) {

    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalWork = numGroups * numTorsions;
    if (globalIdx >= totalWork) return;

    int groupIdx = globalIdx / numTorsions;
    int torsionIdx = globalIdx % numTorsions;

    float scale = globalScalingFactor * groupScalingFactors[groupIdx];
    if (scale == 0.0f) return;

    int4 atoms = torsionAtoms[torsionIdx];
    int pA = groupParticleIndices[groupIdx * numAtoms + atoms.x];
    int pB = groupParticleIndices[groupIdx * numAtoms + atoms.y];
    int pC = groupParticleIndices[groupIdx * numAtoms + atoms.z];
    int pD = groupParticleIndices[groupIdx * numAtoms + atoms.w];

    real4 posA = posq[pA];
    real4 posB = posq[pB];
    real4 posC = posq[pC];
    real4 posD = posq[pD];

    // Vectors matching OpenMM: v1 = A-B, v2 = C-B, v3 = C-D
    real v1x = posA.x - posB.x;
    real v1y = posA.y - posB.y;
    real v1z = posA.z - posB.z;

    real v2x = posC.x - posB.x;
    real v2y = posC.y - posB.y;
    real v2z = posC.z - posB.z;

    real v3x = posC.x - posD.x;
    real v3y = posC.y - posD.y;
    real v3z = posC.z - posD.z;

    // Cross products: cp1 = v1 x v2, cp2 = v2 x v3
    real cp1x = v1y * v2z - v1z * v2y;
    real cp1y = v1z * v2x - v1x * v2z;
    real cp1z = v1x * v2y - v1y * v2x;

    real cp2x = v2y * v3z - v2z * v3y;
    real cp2y = v2z * v3x - v2x * v3z;
    real cp2z = v2x * v3y - v2y * v3x;

    real normCross1 = cp1x * cp1x + cp1y * cp1y + cp1z * cp1z;
    real normCross2 = cp2x * cp2x + cp2y * cp2y + cp2z * cp2z;
    real normV2_2 = v2x * v2x + v2y * v2y + v2z * v2z;
    real normV2 = SQRT(normV2_2);

    if (normCross1 < (real)1.0e-12 || normCross2 < (real)1.0e-12 || normV2 < (real)1.0e-12)
        return;

    // Dihedral angle
    real dotCross = cp1x * cp2x + cp1y * cp2y + cp1z * cp2z;
    real cosPhi = dotCross * RSQRT(normCross1 * normCross2);
    if (cosPhi > (real)1.0) cosPhi = (real)1.0;
    if (cosPhi < (real)-1.0) cosPhi = (real)-1.0;

    // Sign from v1 . cp2 (matches OpenMM: deltaR[0] . crossProduct[1])
    real signPhi = v1x * cp2x + v1y * cp2y + v1z * cp2z;
    real phi = acos(cosPhi);
    if (signPhi < (real)0.0) phi = -phi;

    // Torsion parameters: x = periodicity, y = phase, z = k
    float4 params = torsionParams[torsionIdx];
    int n = (int)params.x;
    real phase = params.y;
    real kT = params.z;

    real deltaAngle = n * phi - phase;
    real energy = kT * ((real)1.0 + cos(deltaAngle)) * scale;
    real dEdAngle = -kT * n * sin(deltaAngle) * scale;

    if (includeEnergy) {
        atomicAdd(&groupEnergies[groupIdx], (float)energy);
        energyBuffer[blockIdx.x * blockDim.x + threadIdx.x] += energy;
    }

    // Force computation (from ReferenceProperDihedralBond)
    real forceFactors0 = (-dEdAngle * normV2) / normCross1;
    real forceFactors3 = (dEdAngle * normV2) / normCross2;

    real dotV1_V2 = v1x * v2x + v1y * v2y + v1z * v2z;
    real dotV3_V2 = v3x * v2x + v3y * v2y + v3z * v2z;

    real forceFactors1 = dotV1_V2 / normV2_2;
    real forceFactors2 = dotV3_V2 / normV2_2;

    real fAx = forceFactors0 * cp1x;
    real fAy = forceFactors0 * cp1y;
    real fAz = forceFactors0 * cp1z;

    real fDx = forceFactors3 * cp2x;
    real fDy = forceFactors3 * cp2y;
    real fDz = forceFactors3 * cp2z;

    real sBx = forceFactors1 * fAx - forceFactors2 * fDx;
    real sBy = forceFactors1 * fAy - forceFactors2 * fDy;
    real sBz = forceFactors1 * fAz - forceFactors2 * fDz;

    real fBx = fAx - sBx;
    real fBy = fAy - sBy;
    real fBz = fAz - sBz;

    real fCx = fDx + sBx;
    real fCy = fDy + sBy;
    real fCz = fDz + sBz;

    // Accumulate forces (A gets +fA, B gets -fB, C gets -fC, D gets +fD)
    atomicAdd(&forceBuffers[pA], static_cast<unsigned long long>((long long)(fAx * 0x100000000)));
    atomicAdd(&forceBuffers[pA + paddedNumAtoms], static_cast<unsigned long long>((long long)(fAy * 0x100000000)));
    atomicAdd(&forceBuffers[pA + 2 * paddedNumAtoms], static_cast<unsigned long long>((long long)(fAz * 0x100000000)));
    atomicAdd(&forceBuffers[pB], static_cast<unsigned long long>((long long)(-fBx * 0x100000000)));
    atomicAdd(&forceBuffers[pB + paddedNumAtoms], static_cast<unsigned long long>((long long)(-fBy * 0x100000000)));
    atomicAdd(&forceBuffers[pB + 2 * paddedNumAtoms], static_cast<unsigned long long>((long long)(-fBz * 0x100000000)));
    atomicAdd(&forceBuffers[pC], static_cast<unsigned long long>((long long)(-fCx * 0x100000000)));
    atomicAdd(&forceBuffers[pC + paddedNumAtoms], static_cast<unsigned long long>((long long)(-fCy * 0x100000000)));
    atomicAdd(&forceBuffers[pC + 2 * paddedNumAtoms], static_cast<unsigned long long>((long long)(-fCz * 0x100000000)));
    atomicAdd(&forceBuffers[pD], static_cast<unsigned long long>((long long)(fDx * 0x100000000)));
    atomicAdd(&forceBuffers[pD + paddedNumAtoms], static_cast<unsigned long long>((long long)(fDy * 0x100000000)));
    atomicAdd(&forceBuffers[pD + 2 * paddedNumAtoms], static_cast<unsigned long long>((long long)(fDz * 0x100000000)));
}
