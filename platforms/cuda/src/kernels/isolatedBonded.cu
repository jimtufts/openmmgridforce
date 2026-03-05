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

    int totalWork = numGroups * numBonds;
    for (int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
         globalIdx < totalWork;
         globalIdx += gridDim.x * blockDim.x) {

        int groupIdx = globalIdx / numBonds;
        int bondIdx = globalIdx % numBonds;

        float scale = globalScalingFactor * groupScalingFactors[groupIdx];
        if (scale == 0.0f) continue;

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
            atomicAdd(energyBuffer, (mixed)energy);
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

    int totalWork = numGroups * numAngles;
    for (int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
         globalIdx < totalWork;
         globalIdx += gridDim.x * blockDim.x) {

        int groupIdx = globalIdx / numAngles;
        int angleIdx = globalIdx % numAngles;

        float scale = globalScalingFactor * groupScalingFactors[groupIdx];
        if (scale == 0.0f) continue;

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
            atomicAdd(energyBuffer, (mixed)energy);
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

    int totalWork = numGroups * numTorsions;
    for (int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
         globalIdx < totalWork;
         globalIdx += gridDim.x * blockDim.x) {

        int groupIdx = globalIdx / numTorsions;
        int torsionIdx = globalIdx % numTorsions;

        float scale = globalScalingFactor * groupScalingFactors[groupIdx];
        if (scale == 0.0f) continue;

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
            continue;

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
            atomicAdd(energyBuffer, (mixed)energy);
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
}


// ====================================================================
// Diagonal Hessian kernels for Riemannian metric
// ====================================================================
//
// These compute the diagonal 3x3 blocks of the bonded Hessian for ALL
// groups. Output is 6 floats per atom [xx,yy,zz,xy,xz,yz] matching
// the GridForce Hessian format. Used by assembleMetricTensor.
//
// Each thread handles one (group, bonded_term) pair, computing the
// diagonal block contributions and atomically adding to the output.

// Helper: atomically add a 3x3 symmetric diagonal block to the output buffer
__device__ __forceinline__ void addDiagBlock(float* diagHessian, int groupIdx,
                                              int localAtomIdx, int numAtoms,
                                              float Hxx, float Hyy, float Hzz,
                                              float Hxy, float Hxz, float Hyz) {
    int base = (groupIdx * numAtoms + localAtomIdx) * 6;
    atomicAdd(&diagHessian[base + 0], Hxx);
    atomicAdd(&diagHessian[base + 1], Hyy);
    atomicAdd(&diagHessian[base + 2], Hzz);
    atomicAdd(&diagHessian[base + 3], Hxy);
    atomicAdd(&diagHessian[base + 4], Hxz);
    atomicAdd(&diagHessian[base + 5], Hyz);
}


// ==================== Bond Diagonal Hessian ====================
// d²E/dx_i² for E = 0.5 * k * (r - r0)²
// Hii = k*(1-r0/r)*I + k*r0/r³ * (rij ⊗ rij)
// Both atoms get the same diagonal block.

extern "C" __global__ void computeIsolatedBondDiagHessian(
    const real4* __restrict__ posq,
    const int* __restrict__ groupParticleIndices,
    const int2* __restrict__ bondAtoms,
    const float2* __restrict__ bondParams,
    float* __restrict__ diagHessian,
    const int numAtoms,
    const int numBonds,
    const int numGroups) {

    int totalWork = numGroups * numBonds;
    for (int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
         globalIdx < totalWork;
         globalIdx += gridDim.x * blockDim.x) {

        int groupIdx = globalIdx / numBonds;
        int bondIdx = globalIdx % numBonds;

        int2 atoms = bondAtoms[bondIdx];
        int pI = groupParticleIndices[groupIdx * numAtoms + atoms.x];
        int pJ = groupParticleIndices[groupIdx * numAtoms + atoms.y];

        real4 posI = posq[pI];
        real4 posJ = posq[pJ];

        float dx = (float)(posJ.x - posI.x);
        float dy = (float)(posJ.y - posI.y);
        float dz = (float)(posJ.z - posI.z);
        float r2 = dx*dx + dy*dy + dz*dz;
        float r = sqrtf(fmaxf(r2, 1e-20f));
        float invR = 1.0f / r;

        float2 params = bondParams[bondIdx];
        float r0 = params.x;
        float k = params.y;

        float factor1 = k * (1.0f - r0 * invR);
        float factor2 = k * r0 * invR * invR * invR;

        float Hxx = factor1 + factor2 * dx * dx;
        float Hyy = factor1 + factor2 * dy * dy;
        float Hzz = factor1 + factor2 * dz * dz;
        float Hxy = factor2 * dx * dy;
        float Hxz = factor2 * dx * dz;
        float Hyz = factor2 * dy * dz;

        // Both atoms get the same diagonal block
        addDiagBlock(diagHessian, groupIdx, atoms.x, numAtoms,
                     Hxx, Hyy, Hzz, Hxy, Hxz, Hyz);
        addDiagBlock(diagHessian, groupIdx, atoms.y, numAtoms,
                     Hxx, Hyy, Hzz, Hxy, Hxz, Hyz);
    }
}


// ==================== Angle Diagonal Hessian ====================
// d²E/dx_a² for E = 0.5 * k * (θ - θ₀)²
// Computes diagonal 3x3 blocks for atoms 1, 2 (central), 3.
// Adapted from BondedHessianAnalytical::computeAngleHessian.

extern "C" __global__ void computeIsolatedAngleDiagHessian(
    const real4* __restrict__ posq,
    const int* __restrict__ groupParticleIndices,
    const int4* __restrict__ angleAtoms,
    const float2* __restrict__ angleParams,
    float* __restrict__ diagHessian,
    const int numAtoms,
    const int numAngles,
    const int numGroups) {

    int totalWork = numGroups * numAngles;
    for (int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
         globalIdx < totalWork;
         globalIdx += gridDim.x * blockDim.x) {

        int groupIdx = globalIdx / numAngles;
        int angleIdx = globalIdx % numAngles;

        int4 atoms = angleAtoms[angleIdx];
        int pA = groupParticleIndices[groupIdx * numAtoms + atoms.x];
        int pB = groupParticleIndices[groupIdx * numAtoms + atoms.y]; // central
        int pC = groupParticleIndices[groupIdx * numAtoms + atoms.z];

        real4 rA = posq[pA];
        real4 rB = posq[pB];
        real4 rC = posq[pC];

        // Bond vectors: b1 = r1 - r2, b3 = r3 - r2
        float b1x = (float)(rA.x - rB.x), b1y = (float)(rA.y - rB.y), b1z = (float)(rA.z - rB.z);
        float b3x = (float)(rC.x - rB.x), b3y = (float)(rC.y - rB.y), b3z = (float)(rC.z - rB.z);

        float L1 = sqrtf(b1x*b1x + b1y*b1y + b1z*b1z);
        float L3 = sqrtf(b3x*b3x + b3y*b3y + b3z*b3z);
        if (L1 < 1e-10f || L3 < 1e-10f) continue;

        float invL1 = 1.0f / L1, invL3 = 1.0f / L3;
        float invL1_sq = invL1 * invL1, invL3_sq = invL3 * invL3;

        // Unit vectors
        float e1x = b1x * invL1, e1y = b1y * invL1, e1z = b1z * invL1;
        float e3x = b3x * invL3, e3y = b3y * invL3, e3z = b3z * invL3;

        float cos_theta = e1x*e3x + e1y*e3y + e1z*e3z;
        cos_theta = fmaxf(-0.9999999f, fminf(0.9999999f, cos_theta));
        float theta = acosf(cos_theta);
        float sin_theta = sinf(theta);
        if (fabsf(sin_theta) < 1e-10f) continue;

        float inv_sin = 1.0f / sin_theta;
        float cot_theta = cos_theta * inv_sin;

        float2 params = angleParams[angleIdx];
        float theta0 = params.x;
        float k = params.y;

        float dtheta = theta - theta0;
        float dE_dtheta = k * dtheta;
        float d2E_dtheta2 = k;

        // v vectors: v1 = e3 - cos*e1, v3 = e1 - cos*e3
        float v1x = e3x - cos_theta*e1x, v1y = e3y - cos_theta*e1y, v1z = e3z - cos_theta*e1z;
        float v3x = e1x - cos_theta*e3x, v3y = e1y - cos_theta*e3y, v3z = e1z - cos_theta*e3z;

        // Gradients: g1 = -inv_sin/L1 * v1, g3 = -inv_sin/L3 * v3, g2 = -(g1+g3)
        float g1x = -inv_sin * invL1 * v1x, g1y = -inv_sin * invL1 * v1y, g1z = -inv_sin * invL1 * v1z;
        float g3x = -inv_sin * invL3 * v3x, g3y = -inv_sin * invL3 * v3y, g3z = -inv_sin * invL3 * v3z;
        float g2x = -(g1x + g3x), g2y = -(g1y + g3y), g2z = -(g1z + g3z);

        // Projection matrices
        // P1[a][b] = delta_ab - e1a*e1b, P3[a][b] = delta_ab - e3a*e3b
        float cot_over_sin = cot_theta * inv_sin;

        // Compute diagonal blocks of H_theta
        // Block (0,0): d²θ/dr1²
        float Ht00[6]; // [xx,yy,zz,xy,xz,yz]
        {
            float c = inv_sin * invL1_sq;
            // H00[a][b] = c * (e1a*v1b + v1a*e1b + cos*P1ab - cot/sin * v1a*v1b)
            Ht00[0] = c * (e1x*v1x + v1x*e1x + cos_theta*(1.0f - e1x*e1x) - cot_over_sin*v1x*v1x);
            Ht00[1] = c * (e1y*v1y + v1y*e1y + cos_theta*(1.0f - e1y*e1y) - cot_over_sin*v1y*v1y);
            Ht00[2] = c * (e1z*v1z + v1z*e1z + cos_theta*(1.0f - e1z*e1z) - cot_over_sin*v1z*v1z);
            Ht00[3] = c * (e1x*v1y + v1x*e1y + cos_theta*(-e1x*e1y) - cot_over_sin*v1x*v1y);
            Ht00[4] = c * (e1x*v1z + v1x*e1z + cos_theta*(-e1x*e1z) - cot_over_sin*v1x*v1z);
            Ht00[5] = c * (e1y*v1z + v1y*e1z + cos_theta*(-e1y*e1z) - cot_over_sin*v1y*v1z);
        }

        // Block (2,2): d²θ/dr3²
        float Ht22[6];
        {
            float c = inv_sin * invL3_sq;
            Ht22[0] = c * (e3x*v3x + v3x*e3x + cos_theta*(1.0f - e3x*e3x) - cot_over_sin*v3x*v3x);
            Ht22[1] = c * (e3y*v3y + v3y*e3y + cos_theta*(1.0f - e3y*e3y) - cot_over_sin*v3y*v3y);
            Ht22[2] = c * (e3z*v3z + v3z*e3z + cos_theta*(1.0f - e3z*e3z) - cot_over_sin*v3z*v3z);
            Ht22[3] = c * (e3x*v3y + v3x*e3y + cos_theta*(-e3x*e3y) - cot_over_sin*v3x*v3y);
            Ht22[4] = c * (e3x*v3z + v3x*e3z + cos_theta*(-e3x*e3z) - cot_over_sin*v3x*v3z);
            Ht22[5] = c * (e3y*v3z + v3y*e3z + cos_theta*(-e3y*e3z) - cot_over_sin*v3y*v3z);
        }

        // Block (0,2): d²θ/dr1 dr3 (needed to compute block (1,1) via translational invariance)
        float Ht02[9]; // full 3x3 [00,01,02,10,11,12,20,21,22]
        {
            float c = -inv_sin * invL1 * invL3;
            // P3ab = delta_ab - e3a*e3b
            // Ht02[a][b] = c * (P3ab - e1a*v3b + cot/sin * v1a*v3b)
            Ht02[0] = c * ((1.0f - e3x*e3x) - e1x*v3x + cot_over_sin*v1x*v3x);
            Ht02[1] = c * ((-e3x*e3y) - e1x*v3y + cot_over_sin*v1x*v3y);
            Ht02[2] = c * ((-e3x*e3z) - e1x*v3z + cot_over_sin*v1x*v3z);
            Ht02[3] = c * ((-e3y*e3x) - e1y*v3x + cot_over_sin*v1y*v3x);
            Ht02[4] = c * ((1.0f - e3y*e3y) - e1y*v3y + cot_over_sin*v1y*v3y);
            Ht02[5] = c * ((-e3y*e3z) - e1y*v3z + cot_over_sin*v1y*v3z);
            Ht02[6] = c * ((-e3z*e3x) - e1z*v3x + cot_over_sin*v1z*v3x);
            Ht02[7] = c * ((-e3z*e3y) - e1z*v3y + cot_over_sin*v1z*v3y);
            Ht02[8] = c * ((1.0f - e3z*e3z) - e1z*v3z + cot_over_sin*v1z*v3z);
        }

        // Block (0,1) = -H00 - H02, Block (2,1) = -H22 - H02^T
        // Block (1,1) = -H01 - H21 = H00 + H02 + H22 + H02^T (translational invariance)
        // We only need the 6 symmetric components of block (1,1):
        // Ht11[xx] = Ht00[xx] + Ht02[00] + Ht22[xx] + Ht02[00]
        // Wait, more carefully:
        // H01[a][b] = -Ht00[a,b] - Ht02[a,b]
        // H21[a][b] = -Ht22[a,b] - Ht02[b,a]  (transpose of H02)
        // H11[a][b] = -H01[a,b] - H21[a,b] = Ht00[a,b] + Ht02[a,b] + Ht22[a,b] + Ht02[b,a]
        float Ht11[6];
        Ht11[0] = Ht00[0] + Ht02[0] + Ht22[0] + Ht02[0]; // xx: Ht02[0,0] + Ht02[0,0]
        Ht11[1] = Ht00[1] + Ht02[4] + Ht22[1] + Ht02[4]; // yy: Ht02[1,1] + Ht02[1,1]
        Ht11[2] = Ht00[2] + Ht02[8] + Ht22[2] + Ht02[8]; // zz: Ht02[2,2] + Ht02[2,2]
        Ht11[3] = Ht00[3] + Ht02[1] + Ht22[3] + Ht02[3]; // xy: Ht02[0,1] + Ht02[1,0]
        Ht11[4] = Ht00[4] + Ht02[2] + Ht22[4] + Ht02[6]; // xz: Ht02[0,2] + Ht02[2,0]
        Ht11[5] = Ht00[5] + Ht02[5] + Ht22[5] + Ht02[7]; // yz: Ht02[1,2] + Ht02[2,1]

        // Full angle Hessian diagonal blocks: H = d2E_dtheta2 * g⊗g + dE_dtheta * Ht
        // Atom 1 (outer)
        addDiagBlock(diagHessian, groupIdx, atoms.x, numAtoms,
            d2E_dtheta2*g1x*g1x + dE_dtheta*Ht00[0],
            d2E_dtheta2*g1y*g1y + dE_dtheta*Ht00[1],
            d2E_dtheta2*g1z*g1z + dE_dtheta*Ht00[2],
            d2E_dtheta2*g1x*g1y + dE_dtheta*Ht00[3],
            d2E_dtheta2*g1x*g1z + dE_dtheta*Ht00[4],
            d2E_dtheta2*g1y*g1z + dE_dtheta*Ht00[5]);

        // Atom 2 (central)
        addDiagBlock(diagHessian, groupIdx, atoms.y, numAtoms,
            d2E_dtheta2*g2x*g2x + dE_dtheta*Ht11[0],
            d2E_dtheta2*g2y*g2y + dE_dtheta*Ht11[1],
            d2E_dtheta2*g2z*g2z + dE_dtheta*Ht11[2],
            d2E_dtheta2*g2x*g2y + dE_dtheta*Ht11[3],
            d2E_dtheta2*g2x*g2z + dE_dtheta*Ht11[4],
            d2E_dtheta2*g2y*g2z + dE_dtheta*Ht11[5]);

        // Atom 3 (outer)
        addDiagBlock(diagHessian, groupIdx, atoms.z, numAtoms,
            d2E_dtheta2*g3x*g3x + dE_dtheta*Ht22[0],
            d2E_dtheta2*g3y*g3y + dE_dtheta*Ht22[1],
            d2E_dtheta2*g3z*g3z + dE_dtheta*Ht22[2],
            d2E_dtheta2*g3x*g3y + dE_dtheta*Ht22[3],
            d2E_dtheta2*g3x*g3z + dE_dtheta*Ht22[4],
            d2E_dtheta2*g3y*g3z + dE_dtheta*Ht22[5]);
    }
}


// ==================== Torsion Diagonal Hessian ====================
// d²E/dx_a² for E = k * (1 + cos(n*φ - phase))
// Computes diagonal 3x3 blocks for atoms 1, 2, 3, 4.
// Uses Blondel-Karplus formulation adapted from BondedHessianAnalytical.

extern "C" __global__ void computeIsolatedTorsionDiagHessian(
    const real4* __restrict__ posq,
    const int* __restrict__ groupParticleIndices,
    const int4* __restrict__ torsionAtoms,
    const float4* __restrict__ torsionParams,
    float* __restrict__ diagHessian,
    const int numAtoms,
    const int numTorsions,
    const int numGroups) {

    int totalWork = numGroups * numTorsions;
    for (int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
         globalIdx < totalWork;
         globalIdx += gridDim.x * blockDim.x) {

        int groupIdx = globalIdx / numTorsions;
        int torsionIdx = globalIdx % numTorsions;

        int4 at = torsionAtoms[torsionIdx];
        int pA = groupParticleIndices[groupIdx * numAtoms + at.x];
        int pB = groupParticleIndices[groupIdx * numAtoms + at.y];
        int pC = groupParticleIndices[groupIdx * numAtoms + at.z];
        int pD = groupParticleIndices[groupIdx * numAtoms + at.w];

        real4 p1 = posq[pA], p2 = posq[pB], p3 = posq[pC], p4 = posq[pD];

        // Bond vectors
        float b1x = (float)(p2.x-p1.x), b1y = (float)(p2.y-p1.y), b1z = (float)(p2.z-p1.z);
        float b2x = (float)(p3.x-p2.x), b2y = (float)(p3.y-p2.y), b2z = (float)(p3.z-p2.z);
        float b3x = (float)(p4.x-p3.x), b3y = (float)(p4.y-p3.y), b3z = (float)(p4.z-p3.z);

        // Normal vectors: m = b1 × b2, n = b2 × b3
        float mx = b1y*b2z - b1z*b2y, my = b1z*b2x - b1x*b2z, mz = b1x*b2y - b1y*b2x;
        float nx = b2y*b3z - b2z*b3y, ny = b2z*b3x - b2x*b3z, nz = b2x*b3y - b2y*b3x;

        float m_sq = mx*mx + my*my + mz*mz;
        float n_sq = nx*nx + ny*ny + nz*nz;
        float b2_sq = b2x*b2x + b2y*b2y + b2z*b2z;
        if (m_sq < 1e-20f || n_sq < 1e-20f || b2_sq < 1e-20f) continue;

        float b2_norm = sqrtf(b2_sq);

        // Dihedral angle
        float m_hat[3] = {mx/sqrtf(m_sq), my/sqrtf(m_sq), mz/sqrtf(m_sq)};
        float n_hat[3] = {nx/sqrtf(n_sq), ny/sqrtf(n_sq), nz/sqrtf(n_sq)};
        float b2_hat[3] = {b2x/b2_norm, b2y/b2_norm, b2z/b2_norm};

        float cos_phi = m_hat[0]*n_hat[0] + m_hat[1]*n_hat[1] + m_hat[2]*n_hat[2];
        cos_phi = fmaxf(-1.0f, fminf(1.0f, cos_phi));
        float mcb2[3] = {m_hat[1]*b2_hat[2]-m_hat[2]*b2_hat[1],
                          m_hat[2]*b2_hat[0]-m_hat[0]*b2_hat[2],
                          m_hat[0]*b2_hat[1]-m_hat[1]*b2_hat[0]};
        float sin_phi = mcb2[0]*n_hat[0] + mcb2[1]*n_hat[1] + mcb2[2]*n_hat[2];
        float phi = atan2f(sin_phi, cos_phi);

        float4 params = torsionParams[torsionIdx];
        int n = (int)params.x;
        float phase = params.y;
        float kT = params.z;

        float dE_dphi = -kT * n * sinf(n * phi - phase);
        float d2E_dphi2 = -kT * n * n * cosf(n * phi - phase);

        // Blondel-Karplus gradients
        float G1[3] = {b2_norm/m_sq * mx, b2_norm/m_sq * my, b2_norm/m_sq * mz};
        float G4[3] = {-b2_norm/n_sq * nx, -b2_norm/n_sq * ny, -b2_norm/n_sq * nz};

        float b1b2 = b1x*b2x + b1y*b2y + b1z*b2z;
        float b3b2 = b3x*b2x + b3y*b2y + b3z*b2z;
        float alpha_c = b1b2 / b2_sq;
        float beta_c = b3b2 / b2_sq;
        float c1 = -(1.0f + alpha_c), c4 = beta_c;
        float d1 = alpha_c, d4 = -(1.0f + beta_c);

        float G2[3], G3[3];
        for (int d = 0; d < 3; d++) {
            G2[d] = c1 * G1[d] + c4 * G4[d];
            G3[d] = d1 * G1[d] + d4 * G4[d];
        }

        // Compute diagonal blocks of H_phi using dG/dp
        // For each atom a, we need dGa/dpa (3x3 matrix) for the diagonal block
        // H_phi diagonal: H_phi[3a+i, 3a+j] = dGa[i]/dpa[j]

        // Skew matrices for cross product derivatives
        // skew(b)[i][j]: b × e_j gives column j of skew
        // dm/dp: dm/dp1 = skew(b2), dm/dp2 = -skew(b2)-skew(b1), dm/dp3 = skew(b1), dm/dp4 = 0
        // dn/dp: dn/dp1 = 0, dn/dp2 = skew(b3), dn/dp3 = -skew(b3)-skew(b2), dn/dp4 = skew(b2)
        float b1_sk[3][3] = {{0, -b1z, b1y}, {b1z, 0, -b1x}, {-b1y, b1x, 0}};
        float b2_sk[3][3] = {{0, -b2z, b2y}, {b2z, 0, -b2x}, {-b2y, b2x, 0}};
        float b3_sk[3][3] = {{0, -b3z, b3y}, {b3z, 0, -b3x}, {-b3y, b3x, 0}};

        // dm/dp for each atom: dm_dp[atom][i][j]
        float dm_dp[4][3][3], dn_dp[4][3][3];
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                dm_dp[0][i][j] = b2_sk[i][j];
                dm_dp[1][i][j] = -b2_sk[i][j] - b1_sk[i][j];
                dm_dp[2][i][j] = b1_sk[i][j];
                dm_dp[3][i][j] = 0;
                dn_dp[0][i][j] = 0;
                dn_dp[1][i][j] = b3_sk[i][j];
                dn_dp[2][i][j] = -b3_sk[i][j] - b2_sk[i][j];
                dn_dp[3][i][j] = b2_sk[i][j];
            }
        }

        // d|b2|/dp for each atom
        float db2n_dp[4][3];
        for (int d = 0; d < 3; d++) {
            db2n_dp[0][d] = 0;
            db2n_dp[1][d] = -b2_hat[d];
            db2n_dp[2][d] = b2_hat[d];
            db2n_dp[3][d] = 0;
        }

        // Dot product derivatives
        float db1b2_dp[4][3], db3b2_dp[4][3], db2sq_dp[4][3];
        for (int d = 0; d < 3; d++) {
            float b1v[3] = {b1x, b1y, b1z};
            float b2v[3] = {b2x, b2y, b2z};
            float b3v[3] = {b3x, b3y, b3z};
            db1b2_dp[0][d] = -b2v[d];
            db1b2_dp[1][d] = b2v[d] - b1v[d];
            db1b2_dp[2][d] = b1v[d];
            db1b2_dp[3][d] = 0;
            db3b2_dp[0][d] = 0;
            db3b2_dp[1][d] = -b3v[d];
            db3b2_dp[2][d] = b3v[d] - b2v[d];
            db3b2_dp[3][d] = b2v[d];
            db2sq_dp[0][d] = 0;
            db2sq_dp[1][d] = -2.0f * b2v[d];
            db2sq_dp[2][d] = 2.0f * b2v[d];
            db2sq_dp[3][d] = 0;
        }

        // Outer products for dG1/dp and dG4/dp
        float mm[3][3], nn[3][3];
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++) {
                mm[i][j] = mx * (i==0?1:0) * mx * (j==0?1:0);  // wrong, fix below
                nn[i][j] = nx * (i==0?1:0) * nx * (j==0?1:0);
            }
        // Actually need proper outer products
        float mv[3] = {mx, my, mz};
        float nv[3] = {nx, ny, nz};
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++) {
                mm[i][j] = mv[i] * mv[j];
                nn[i][j] = nv[i] * nv[j];
            }

        // For each atom, compute diagonal block of H_phi
        // H_phi[3a+i, 3a+j] is the (i,j) component of dGa/dpa
        // where Ga = G1 for a=0, G2 for a=1, G3 for a=2, G4 for a=3

        float allG[4][3] = {{G1[0],G1[1],G1[2]}, {G2[0],G2[1],G2[2]},
                             {G3[0],G3[1],G3[2]}, {G4[0],G4[1],G4[2]}};
        int atomIdx[4] = {at.x, at.y, at.z, at.w};

        for (int a = 0; a < 4; a++) {
            // Compute dGa/dpa (diagonal block of H_phi)
            // dG1/dp: dG1/dpj[i][k] = m[i] * db2n/dpj[k] / m_sq +
            //         b2_norm/m_sq * sum_c (delta_ic - 2*mm[i][c]/m_sq) * dm/dpj[c][k]
            // dG4/dp: dG4/dpj[i][k] = -n[i] * db2n/dpj[k] / n_sq +
            //         -b2_norm/n_sq * sum_c (delta_ic - 2*nn[i][c]/n_sq) * dn/dpj[c][k]

            float dG1_dpa[3][3], dG4_dpa[3][3];
            for (int i = 0; i < 3; i++) {
                for (int k = 0; k < 3; k++) {
                    float t1 = mv[i] * db2n_dp[a][k] / m_sq;
                    float t2 = 0;
                    for (int c = 0; c < 3; c++) {
                        float Imc = ((i == c) ? 1.0f : 0.0f) - 2.0f * mm[i][c] / m_sq;
                        t2 += (b2_norm / m_sq) * Imc * dm_dp[a][c][k];
                    }
                    dG1_dpa[i][k] = t1 + t2;

                    float t1n = -nv[i] * db2n_dp[a][k] / n_sq;
                    float t2n = 0;
                    for (int c = 0; c < 3; c++) {
                        float Inc = ((i == c) ? 1.0f : 0.0f) - 2.0f * nn[i][c] / n_sq;
                        t2n += -(b2_norm / n_sq) * Inc * dn_dp[a][c][k];
                    }
                    dG4_dpa[i][k] = t1n + t2n;
                }
            }

            // Coefficient derivatives for this atom
            float dc1_dpa[3], dc4_dpa[3], dd1_dpa[3], dd4_dpa[3];
            for (int d = 0; d < 3; d++) {
                float inv_b2sq = 1.0f / b2_sq;
                float inv_b2sq2 = inv_b2sq * inv_b2sq;
                dc1_dpa[d] = -db1b2_dp[a][d] * inv_b2sq + b1b2 * db2sq_dp[a][d] * inv_b2sq2;
                dc4_dpa[d] = db3b2_dp[a][d] * inv_b2sq - b3b2 * db2sq_dp[a][d] * inv_b2sq2;
                dd1_dpa[d] = db1b2_dp[a][d] * inv_b2sq - b1b2 * db2sq_dp[a][d] * inv_b2sq2;
                dd4_dpa[d] = -db3b2_dp[a][d] * inv_b2sq + b3b2 * db2sq_dp[a][d] * inv_b2sq2;
            }

            // Compute dGa/dpa based on which atom this is
            float Hphi_diag[3][3]; // diagonal block of H_phi for this atom
            if (a == 0) {
                // dG1/dp1
                for (int i = 0; i < 3; i++)
                    for (int k = 0; k < 3; k++)
                        Hphi_diag[i][k] = dG1_dpa[i][k];
            } else if (a == 3) {
                // dG4/dp4
                for (int i = 0; i < 3; i++)
                    for (int k = 0; k < 3; k++)
                        Hphi_diag[i][k] = dG4_dpa[i][k];
            } else if (a == 1) {
                // dG2/dp2: G2 = c1*G1 + c4*G4
                // dG2/dp2 = G1 ⊗ dc1/dp2 + c1*dG1/dp2 + G4 ⊗ dc4/dp2 + c4*dG4/dp2
                for (int i = 0; i < 3; i++)
                    for (int k = 0; k < 3; k++)
                        Hphi_diag[i][k] = G1[i]*dc1_dpa[k] + c1*dG1_dpa[i][k]
                                         + G4[i]*dc4_dpa[k] + c4*dG4_dpa[i][k];
            } else { // a == 2
                // dG3/dp3: G3 = d1*G1 + d4*G4
                for (int i = 0; i < 3; i++)
                    for (int k = 0; k < 3; k++)
                        Hphi_diag[i][k] = G1[i]*dd1_dpa[k] + d1*dG1_dpa[i][k]
                                         + G4[i]*dd4_dpa[k] + d4*dG4_dpa[i][k];
            }

            // Symmetrize (H_phi may not be perfectly symmetric due to float precision)
            for (int i = 0; i < 3; i++)
                for (int k = i+1; k < 3; k++) {
                    float avg = 0.5f * (Hphi_diag[i][k] + Hphi_diag[k][i]);
                    Hphi_diag[i][k] = avg;
                    Hphi_diag[k][i] = avg;
                }

            // Full Hessian diagonal: H[3a+i][3a+j] = d2E_dphi2 * Ga[i]*Ga[j] + dE_dphi * Hphi_diag[i][j]
            float Ga[3] = {allG[a][0], allG[a][1], allG[a][2]};
            float Hxx = d2E_dphi2 * Ga[0]*Ga[0] + dE_dphi * Hphi_diag[0][0];
            float Hyy = d2E_dphi2 * Ga[1]*Ga[1] + dE_dphi * Hphi_diag[1][1];
            float Hzz = d2E_dphi2 * Ga[2]*Ga[2] + dE_dphi * Hphi_diag[2][2];
            float Hxy = d2E_dphi2 * Ga[0]*Ga[1] + dE_dphi * Hphi_diag[0][1];
            float Hxz = d2E_dphi2 * Ga[0]*Ga[2] + dE_dphi * Hphi_diag[0][2];
            float Hyz = d2E_dphi2 * Ga[1]*Ga[2] + dE_dphi * Hphi_diag[1][2];

            addDiagBlock(diagHessian, groupIdx, atomIdx[a], numAtoms,
                         Hxx, Hyy, Hzz, Hxy, Hxz, Hyz);
        }
    }
}
