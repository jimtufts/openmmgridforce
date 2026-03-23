/**
 * CUDA kernels for MultiGroupHMCIntegrator.
 *
 * Group k owns atoms [k*ATOMS_PER_GROUP .. (k+1)*ATOMS_PER_GROUP - 1].
 * All kernels use grid-stride loops.
 */

extern "C" __global__ void hmcBackupPositions(
    const real4* __restrict__ posq,
    real4* __restrict__ posBackup,
    const int numGroupAtoms) {

    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < numGroupAtoms;
         i += gridDim.x * blockDim.x) {
        posBackup[i] = posq[i];
    }
}

extern "C" __global__ void hmcDrawMBVelocitiesFull(
    mixed4* __restrict__ velm,
    const float4* __restrict__ random,
    const unsigned int randomIndex,
    const double* __restrict__ groupKT,
    const int atomsPerGroup,
    const int numGroups) {

    int totalAtoms = numGroups * atomsPerGroup;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < totalAtoms;
         i += gridDim.x * blockDim.x) {

        mixed4 v = velm[i];
        mixed invMass = v.w;
        if (invMass == 0) continue;

        int groupIdx = i / atomsPerGroup;
        mixed kT = (mixed)groupKT[groupIdx];
        mixed sigma = SQRT(kT * invMass);

        float4 r = random[randomIndex + i];
        v.x = sigma * (mixed)r.x;
        v.y = sigma * (mixed)r.y;
        v.z = sigma * (mixed)r.z;
        velm[i] = v;
    }
}

extern "C" __global__ void hmcDrawMBVelocitiesPartial(
    mixed4* __restrict__ velm,
    const float4* __restrict__ random,
    const unsigned int randomIndex,
    const double* __restrict__ groupKT,
    const int atomsPerGroup,
    const int numGroups,
    const mixed cosTheta,
    const mixed sinTheta) {

    int totalAtoms = numGroups * atomsPerGroup;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < totalAtoms;
         i += gridDim.x * blockDim.x) {

        mixed4 v = velm[i];
        mixed invMass = v.w;
        if (invMass == 0) continue;

        int groupIdx = i / atomsPerGroup;
        mixed kT = (mixed)groupKT[groupIdx];
        mixed sigma = SQRT(kT * invMass);

        float4 r = random[randomIndex + i];
        v.x = cosTheta * v.x + sinTheta * sigma * (mixed)r.x;
        v.y = cosTheta * v.y + sinTheta * sigma * (mixed)r.y;
        v.z = cosTheta * v.z + sinTheta * sigma * (mixed)r.z;
        velm[i] = v;
    }
}

extern "C" __global__ void hmcComputeGroupKE(
    const mixed4* __restrict__ velm,
    unsigned long long* __restrict__ groupKE,
    const int atomsPerGroup,
    const int numGroups) {

    int totalAtoms = numGroups * atomsPerGroup;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < totalAtoms;
         i += gridDim.x * blockDim.x) {

        mixed4 v = velm[i];
        mixed invMass = v.w;
        if (invMass == 0) continue;

        int groupIdx = i / atomsPerGroup;
        double ke = 0.5 * ((double)(v.x * v.x) + (double)(v.y * v.y) + (double)(v.z * v.z))
                    / (double)invMass;
        atomicAdd(&groupKE[groupIdx], static_cast<unsigned long long>((long long)(ke * 0x100000000)));
    }
}

/**
 * Velocity half-kick: v += scale * dt[group] * F / m
 *
 * forceBuffer is the long long fixed-point buffer.
 * scale is 0.5 for a half-kick.
 * groupDt contains the per-group timestep (already divided by innerSteps if needed).
 */
extern "C" __global__ void hmcVelocityKick(
    mixed4* __restrict__ velm,
    const long long* __restrict__ forceBuffer,
    const double* __restrict__ groupDt,
    const int atomsPerGroup,
    const int numGroups,
    const mixed scale,
    const int paddedNumAtoms) {

    int totalAtoms = numGroups * atomsPerGroup;
    const double forceScale = 1.0 / (double)0x100000000;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < totalAtoms;
         i += gridDim.x * blockDim.x) {

        mixed4 v = velm[i];
        mixed invMass = v.w;
        if (invMass == 0) continue;

        int groupIdx = i / atomsPerGroup;
        mixed dt = scale * (mixed)groupDt[groupIdx];

        mixed fx = (mixed)((double)forceBuffer[i] * forceScale);
        mixed fy = (mixed)((double)forceBuffer[i + paddedNumAtoms] * forceScale);
        mixed fz = (mixed)((double)forceBuffer[i + 2 * paddedNumAtoms] * forceScale);

        v.x += dt * fx * invMass;
        v.y += dt * fy * invMass;
        v.z += dt * fz * invMass;
        velm[i] = v;
    }
}

extern "C" __global__ void hmcPositionDrift(
    real4* __restrict__ posq,
    const mixed4* __restrict__ velm,
    const double* __restrict__ groupDt,
    const int atomsPerGroup,
    const int numGroups) {

    int totalAtoms = numGroups * atomsPerGroup;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < totalAtoms;
         i += gridDim.x * blockDim.x) {

        int groupIdx = i / atomsPerGroup;
        mixed dt = (mixed)groupDt[groupIdx];

        mixed4 v = velm[i];
        real4 pos = posq[i];
        pos.x += (real)(dt * v.x);
        pos.y += (real)(dt * v.y);
        pos.z += (real)(dt * v.z);
        posq[i] = pos;
    }
}

/**
 * Copy force buffer for group atoms (used to save slow forces in RESPA).
 * Force layout: [Fx[0..pad-1], Fy[0..pad-1], Fz[0..pad-1]]
 */
extern "C" __global__ void hmcCopyForces(
    const long long* __restrict__ src,
    long long* __restrict__ dst,
    const int numGroupAtoms,
    const int paddedNumAtoms) {

    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < numGroupAtoms;
         i += gridDim.x * blockDim.x) {
        dst[i] = src[i];
        dst[i + paddedNumAtoms] = src[i + paddedNumAtoms];
        dst[i + 2 * paddedNumAtoms] = src[i + 2 * paddedNumAtoms];
    }
}

/**
 * Compute mass-weighted center of mass per group.
 * COM[k] = sum(mass_i * pos_i) / sum(mass_i) for atoms in group k.
 * Uses atomicAdd for parallel reduction.
 *
 * groupCOM layout: [cx0, cy0, cz0, mass0, cx1, cy1, cz1, mass1, ...]
 *   - 4 doubles per group: (sum_mx, sum_my, sum_mz, sum_m)
 *   - Caller must zero the buffer before launch and divide by mass after.
 */
extern "C" __global__ void hmcComputeGroupCOM(
    const real4* __restrict__ posq,
    const mixed4* __restrict__ velm,
    unsigned long long* __restrict__ groupCOM,
    const int atomsPerGroup,
    const int numGroups) {

    int totalAtoms = numGroups * atomsPerGroup;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < totalAtoms;
         i += gridDim.x * blockDim.x) {

        mixed4 v = velm[i];
        mixed invMass = v.w;
        if (invMass == 0) continue;

        double mass = 1.0 / (double)invMass;
        real4 pos = posq[i];
        int groupIdx = i / atomsPerGroup;

        atomicAdd(&groupCOM[groupIdx * 4 + 0], static_cast<unsigned long long>((long long)(mass * (double)pos.x * 0x100000000)));
        atomicAdd(&groupCOM[groupIdx * 4 + 1], static_cast<unsigned long long>((long long)(mass * (double)pos.y * 0x100000000)));
        atomicAdd(&groupCOM[groupIdx * 4 + 2], static_cast<unsigned long long>((long long)(mass * (double)pos.z * 0x100000000)));
        atomicAdd(&groupCOM[groupIdx * 4 + 3], static_cast<unsigned long long>((long long)(mass * 0x100000000)));
    }
}

/**
 * Apply rigid-body move (rotation about COM + translation) for MC-enabled groups.
 *
 * For each atom in an enabled group:
 *   pos -= COM[group]
 *   pos = R[group] * pos
 *   pos += COM[group] + translation[group]
 *
 * rotationMat: 9 doubles per group (row-major 3x3)
 * translation: 3 doubles per group
 * groupCOM: 3 doubles per group (already divided by total mass)
 */
extern "C" __global__ void hmcApplyRigidBodyMove(
    real4* __restrict__ posq,
    const double* __restrict__ groupCOM,
    const double* __restrict__ rotationMat,
    const double* __restrict__ translation,
    const int* __restrict__ mcEnabled,
    const int atomsPerGroup,
    const int numGroups) {

    int totalAtoms = numGroups * atomsPerGroup;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < totalAtoms;
         i += gridDim.x * blockDim.x) {

        int groupIdx = i / atomsPerGroup;
        if (!mcEnabled[groupIdx]) continue;

        real4 pos = posq[i];

        // Subtract COM
        double dx = (double)pos.x - groupCOM[groupIdx * 3 + 0];
        double dy = (double)pos.y - groupCOM[groupIdx * 3 + 1];
        double dz = (double)pos.z - groupCOM[groupIdx * 3 + 2];

        // Apply rotation (row-major 3x3)
        const double* R = &rotationMat[groupIdx * 9];
        double rx = R[0]*dx + R[1]*dy + R[2]*dz;
        double ry = R[3]*dx + R[4]*dy + R[5]*dz;
        double rz = R[6]*dx + R[7]*dy + R[8]*dz;

        // Add COM + translation
        pos.x = (real)(rx + groupCOM[groupIdx * 3 + 0] + translation[groupIdx * 3 + 0]);
        pos.y = (real)(ry + groupCOM[groupIdx * 3 + 1] + translation[groupIdx * 3 + 1]);
        pos.z = (real)(rz + groupCOM[groupIdx * 3 + 2] + translation[groupIdx * 3 + 2]);
        posq[i] = pos;
    }
}

/**
 * Restore positions and zero velocities for rejected groups.
 */
extern "C" __global__ void hmcRestoreRejected(
    real4* __restrict__ posq,
    mixed4* __restrict__ velm,
    const real4* __restrict__ posBackup,
    const int* __restrict__ accepted,
    const int atomsPerGroup,
    const int numGroups) {

    int totalAtoms = numGroups * atomsPerGroup;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < totalAtoms;
         i += gridDim.x * blockDim.x) {

        int groupIdx = i / atomsPerGroup;
        if (accepted[groupIdx]) continue;

        posq[i] = posBackup[i];
        mixed4 v = velm[i];
        v.x = 0;
        v.y = 0;
        v.z = 0;
        velm[i] = v;
    }
}
