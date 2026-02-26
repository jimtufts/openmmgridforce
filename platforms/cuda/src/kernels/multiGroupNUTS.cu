/**
 * CUDA kernels for MultiGroupNUTSIntegrator.
 *
 * Group k owns atoms [k*ATOMS_PER_GROUP .. (k+1)*ATOMS_PER_GROUP - 1].
 * All kernels use grid-stride loops.
 *
 * NUTS-specific kernels add an active[k] mask to skip terminated groups
 * during leapfrog integration, plus tree endpoint management and U-turn
 * dot product computation.
 */

/**
 * Restore positions and velocities from the appropriate tree endpoint
 * based on direction. This sets up the starting point before taking
 * 2^j leapfrog steps in the chosen direction.
 *
 * direction[k] = +1: restore from xplus/vplus (extend forward)
 * direction[k] = -1: restore from xminus/vminus (extend backward)
 * inactive groups are skipped.
 */
extern "C" __global__ void nutsRestoreEndpoint(
    real4* __restrict__ posq,
    mixed4* __restrict__ velm,
    const real4* __restrict__ xminus,
    const real4* __restrict__ xplus,
    const mixed4* __restrict__ vminus,
    const mixed4* __restrict__ vplus,
    const int* __restrict__ active,
    const int* __restrict__ direction,
    const int atomsPerGroup,
    const int numGroups) {

    int totalAtoms = numGroups * atomsPerGroup;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < totalAtoms;
         i += gridDim.x * blockDim.x) {

        int groupIdx = i / atomsPerGroup;
        if (!active[groupIdx]) continue;

        if (direction[groupIdx] > 0) {
            posq[i] = xplus[i];
            velm[i] = vplus[i];
        } else {
            posq[i] = xminus[i];
            velm[i] = vminus[i];
        }
    }
}

/**
 * Save current positions and velocities to the appropriate tree endpoint
 * after completing 2^j leapfrog steps.
 *
 * direction[k] = +1: save to xplus/vplus (extended forward)
 * direction[k] = -1: save to xminus/vminus (extended backward)
 */
extern "C" __global__ void nutsSaveEndpoint(
    const real4* __restrict__ posq,
    const mixed4* __restrict__ velm,
    real4* __restrict__ xminus,
    real4* __restrict__ xplus,
    mixed4* __restrict__ vminus,
    mixed4* __restrict__ vplus,
    const int* __restrict__ active,
    const int* __restrict__ direction,
    const int atomsPerGroup,
    const int numGroups) {

    int totalAtoms = numGroups * atomsPerGroup;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < totalAtoms;
         i += gridDim.x * blockDim.x) {

        int groupIdx = i / atomsPerGroup;
        if (!active[groupIdx]) continue;

        if (direction[groupIdx] > 0) {
            xplus[i] = posq[i];
            vplus[i] = velm[i];
        } else {
            xminus[i] = posq[i];
            vminus[i] = velm[i];
        }
    }
}

/**
 * Velocity kick with active-group masking.
 * v += scale * dt[group] * F / m
 * Inactive groups are skipped (frozen).
 * Note: dt[group] can be negative for backward stepping.
 */
extern "C" __global__ void nutsVelocityKick(
    mixed4* __restrict__ velm,
    const long long* __restrict__ forceBuffer,
    const double* __restrict__ groupDt,
    const int* __restrict__ active,
    const int atomsPerGroup,
    const int numGroups,
    const mixed scale,
    const int paddedNumAtoms) {

    int totalAtoms = numGroups * atomsPerGroup;
    const double forceScale = 1.0 / (double)0x100000000;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < totalAtoms;
         i += gridDim.x * blockDim.x) {

        int groupIdx = i / atomsPerGroup;
        if (!active[groupIdx]) continue;

        mixed4 v = velm[i];
        mixed invMass = v.w;
        if (invMass == 0) continue;

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

/**
 * Position drift with active-group masking.
 * x += dt[group] * v
 * Note: dt[group] can be negative for backward stepping.
 */
extern "C" __global__ void nutsPositionDrift(
    real4* __restrict__ posq,
    const mixed4* __restrict__ velm,
    const double* __restrict__ groupDt,
    const int* __restrict__ active,
    const int atomsPerGroup,
    const int numGroups) {

    int totalAtoms = numGroups * atomsPerGroup;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < totalAtoms;
         i += gridDim.x * blockDim.x) {

        int groupIdx = i / atomsPerGroup;
        if (!active[groupIdx]) continue;

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
 * Conditionally copy current positions to the candidate buffer.
 * For each group where copyFlag[k] != 0, copy positions.
 */
extern "C" __global__ void nutsCopyCandidatePos(
    const real4* __restrict__ posq,
    real4* __restrict__ candidatePos,
    const int* __restrict__ copyFlag,
    const int atomsPerGroup,
    const int numGroups) {

    int totalAtoms = numGroups * atomsPerGroup;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < totalAtoms;
         i += gridDim.x * blockDim.x) {

        int groupIdx = i / atomsPerGroup;
        if (!copyFlag[groupIdx]) continue;
        candidatePos[i] = posq[i];
    }
}

/**
 * Compute U-turn dot products per group.
 *
 * For each active group k, compute:
 *   dot1 = sum_atoms (xplus - xminus) . vminus   (using position xyz and velocity xyz)
 *   dot2 = sum_atoms (xplus - xminus) . vplus
 *
 * Output: uturnDot[2*k] = dot1, uturnDot[2*k+1] = dot2
 *
 * Uses one block per group with shared memory reduction.
 * Since atomsPerGroup is typically small (~20), this is efficient.
 */
extern "C" __global__ void nutsComputeUTurnDot(
    const real4* __restrict__ xminus,
    const real4* __restrict__ xplus,
    const mixed4* __restrict__ vminus,
    const mixed4* __restrict__ vplus,
    const int* __restrict__ active,
    double* __restrict__ uturnDot,
    const int atomsPerGroup,
    const int numGroups) {

    // One block per group
    int groupIdx = blockIdx.x;
    if (groupIdx >= numGroups) return;
    if (!active[groupIdx]) {
        // Inactive: set positive values so U-turn is not flagged
        if (threadIdx.x == 0) {
            uturnDot[2 * groupIdx] = 1.0;
            uturnDot[2 * groupIdx + 1] = 1.0;
        }
        return;
    }

    extern __shared__ double nutsUturnShared[];
    double* sDot1 = nutsUturnShared;
    double* sDot2 = nutsUturnShared + blockDim.x;

    int baseAtom = groupIdx * atomsPerGroup;
    double localDot1 = 0.0;
    double localDot2 = 0.0;

    // Each thread handles multiple atoms if atomsPerGroup > blockDim.x
    for (int a = threadIdx.x; a < atomsPerGroup; a += blockDim.x) {
        int idx = baseAtom + a;
        real4 xm = xminus[idx];
        real4 xp = xplus[idx];
        mixed4 vm = vminus[idx];
        mixed4 vp = vplus[idx];

        double dx = (double)xp.x - (double)xm.x;
        double dy = (double)xp.y - (double)xm.y;
        double dz = (double)xp.z - (double)xm.z;

        localDot1 += dx * (double)vm.x + dy * (double)vm.y + dz * (double)vm.z;
        localDot2 += dx * (double)vp.x + dy * (double)vp.y + dz * (double)vp.z;
    }

    sDot1[threadIdx.x] = localDot1;
    sDot2[threadIdx.x] = localDot2;
    __syncthreads();

    // Reduce within block
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sDot1[threadIdx.x] += sDot1[threadIdx.x + s];
            sDot2[threadIdx.x] += sDot2[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        uturnDot[2 * groupIdx] = sDot1[0];
        uturnDot[2 * groupIdx + 1] = sDot2[0];
    }
}

/**
 * Set positions from candidate buffer and zero velocities for all groups.
 * Used at the end of NUTS trial to finalize positions.
 */
extern "C" __global__ void nutsSetFromCandidate(
    real4* __restrict__ posq,
    mixed4* __restrict__ velm,
    const real4* __restrict__ candidatePos,
    const int numGroupAtoms) {

    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < numGroupAtoms;
         i += gridDim.x * blockDim.x) {

        posq[i] = candidatePos[i];
        mixed4 v = velm[i];
        v.x = 0;
        v.y = 0;
        v.z = 0;
        velm[i] = v;
    }
}

/**
 * Restore positions from backup for divergent groups.
 * Non-divergent groups keep their candidate positions.
 */
extern "C" __global__ void nutsRestoreDivergent(
    real4* __restrict__ posq,
    mixed4* __restrict__ velm,
    const real4* __restrict__ posBackup,
    const int* __restrict__ divergent,
    const int atomsPerGroup,
    const int numGroups) {

    int totalAtoms = numGroups * atomsPerGroup;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < totalAtoms;
         i += gridDim.x * blockDim.x) {

        int groupIdx = i / atomsPerGroup;
        if (!divergent[groupIdx]) continue;

        posq[i] = posBackup[i];
        mixed4 v = velm[i];
        v.x = 0;
        v.y = 0;
        v.z = 0;
        velm[i] = v;
    }
}

/**
 * Compute per-group kinetic energy (same as HMC version).
 */
extern "C" __global__ void nutsComputeGroupKE(
    const mixed4* __restrict__ velm,
    double* __restrict__ groupKE,
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
        atomicAdd(&groupKE[groupIdx], ke);
    }
}

/**
 * Backup positions (same as HMC version).
 */
extern "C" __global__ void nutsBackupPositions(
    const real4* __restrict__ posq,
    real4* __restrict__ posBackup,
    const int numGroupAtoms) {

    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < numGroupAtoms;
         i += gridDim.x * blockDim.x) {
        posBackup[i] = posq[i];
    }
}

/**
 * Draw fresh Maxwell-Boltzmann velocities per group (same as HMC version).
 */
extern "C" __global__ void nutsDrawMBVelocitiesFull(
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

/**
 * Partial momentum refresh (same as HMC version).
 */
extern "C" __global__ void nutsDrawMBVelocitiesPartial(
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

/**
 * Copy force buffer (same as HMC, for RESPA slow forces).
 */
extern "C" __global__ void nutsCopyForces(
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

// ============================================================
// GPU-side NUTS tree-building kernels (Phase 2)
// These kernels move per-step decision logic to the GPU,
// eliminating CPU-GPU sync points in the inner loop.
// ============================================================

/**
 * Simple counter-based hash RNG for GPU-side uniform random generation.
 * Returns a uniform float in [0, 1) deterministically for a given (seed, k, step).
 * Uses Murmur3-style finalization. Not cryptographic, but sufficient for
 * NUTS candidate selection and direction sampling.
 */
__device__ __forceinline__ float nutsUniformRng(unsigned int seed, unsigned int k, unsigned int step) {
    unsigned int h = seed ^ (k * 2654435761u) ^ (step * 2246822519u);
    h = (h ^ (h >> 16)) * 0x45d9f3bu;
    h = (h ^ (h >> 16)) * 0x45d9f3bu;
    h = h ^ (h >> 16);
    return (float)(h & 0x7FFFFFu) / (float)0x800000u;
}

/**
 * Sum per-group energies from all force energy buffers into a single double[K] buffer.
 * Each force stores group energies as float[numGroups] on the GPU.
 * forceEnergyPtrs[f] is the device address (as unsigned long long) of force f's buffer.
 * One thread per group.
 */
extern "C" __global__ void nutsGatherForceEnergies(
    double* __restrict__ totalGroupPE,
    const unsigned long long* __restrict__ forceEnergyPtrs,
    int numForces,
    int numGroups) {

    for (int k = blockIdx.x * blockDim.x + threadIdx.x;
         k < numGroups;
         k += gridDim.x * blockDim.x) {
        double sum = 0.0;
        for (int f = 0; f < numForces; f++) {
            const float* buf = (const float*)forceEnergyPtrs[f];
            sum += (double)buf[k];
        }
        totalGroupPE[k] = sum;
    }
}

/**
 * Fused per-step decision kernel for NUTS inner loop.
 * Replaces the host-side loop that checks divergence, slice validity,
 * and performs 1/n candidate selection.
 *
 * One thread per group. Writes: active, divergent, subtreeNValid,
 * copyFlag, subtreeCandidatePE, subtreeHasCandidate.
 *
 * copyFlag is set to 0 for all non-selected groups (including inactive),
 * ensuring the subsequent copyCandidatePos kernel only copies selected groups.
 */
extern "C" __global__ void nutsLeapfrogDecision(
    const double* __restrict__ totalGroupPE,
    const double* __restrict__ groupKE,
    const double* __restrict__ logu,
    const double* __restrict__ H0,
    const double* __restrict__ peInit,
    const double* __restrict__ groupKT,
    int* __restrict__ active,
    int* __restrict__ divergent,
    int* __restrict__ subtreeNValid,
    double* __restrict__ subtreeCandidatePE,
    int* __restrict__ subtreeHasCandidate,
    int* __restrict__ copyFlag,
    double stabilityThreshold,
    int numGroups,
    unsigned int rngSeed,
    int stepIndex) {

    for (int k = blockIdx.x * blockDim.x + threadIdx.x;
         k < numGroups;
         k += gridDim.x * blockDim.x) {

        if (!active[k]) {
            copyFlag[k] = 0;
            continue;
        }

        double pe = totalGroupPE[k];
        double ke = groupKE[k];
        double kT = groupKT[k];
        double Hk = pe + ke;
        double logPk = -Hk / kT;

        // Divergence check
        double deltaPE = pe - peInit[k];
        double deltaH = Hk - H0[k];
        if (fabs(deltaPE) / kT > stabilityThreshold &&
            fabs(deltaH) / kT > stabilityThreshold) {
            active[k] = 0;
            divergent[k] = 1;
            copyFlag[k] = 0;
            continue;
        }

        // Slice validity check
        if (logPk > logu[k]) {
            int nv = subtreeNValid[k] + 1;
            subtreeNValid[k] = nv;

            // 1/n uniform candidate selection
            float u = nutsUniformRng(rngSeed, k, stepIndex);
            if (u < 1.0f / (float)nv) {
                subtreeCandidatePE[k] = pe;
                subtreeHasCandidate[k] = 1;
                copyFlag[k] = 1;
            } else {
                copyFlag[k] = 0;
            }
        } else {
            copyFlag[k] = 0;
        }
    }
}

/**
 * Fused U-turn check + active deactivation + anyActive reduction.
 * Reads U-turn dot products (2 per group from nutsComputeUTurnDot),
 * deactivates groups where either dot < 0, and atomically ORs into
 * anyActive[0] if any group remains active.
 *
 * anyActive[0] must be cleared to 0 before this kernel is launched.
 * One thread per group.
 */
extern "C" __global__ void nutsCheckUTurnAndDeactivate(
    const double* __restrict__ uturnDot,
    int* __restrict__ active,
    int* __restrict__ anyActive,
    int numGroups) {

    for (int k = blockIdx.x * blockDim.x + threadIdx.x;
         k < numGroups;
         k += gridDim.x * blockDim.x) {

        if (!active[k]) continue;

        double dot1 = uturnDot[2 * k];
        double dot2 = uturnDot[2 * k + 1];

        if (dot1 < 0.0 || dot2 < 0.0) {
            active[k] = 0;
        }

        if (active[k]) {
            atomicOr(anyActive, 1);
        }
    }
}

/**
 * Metropolis-within-tree: combine subtree candidate with main candidate.
 * For each active group with a subtree candidate, accept with probability
 * subtreeNValid / (nValid + subtreeNValid). If accepted, set copyFlag=1
 * so the subsequent copyCandidatePos kernel copies subtreeCandidate -> candidate.
 *
 * Also updates nValid = nValid + subtreeNValid for all active groups with candidates.
 * One thread per group.
 */
extern "C" __global__ void nutsCombineCandidates(
    const int* __restrict__ subtreeNValid,
    int* __restrict__ nValid,
    const int* __restrict__ subtreeHasCandidate,
    const double* __restrict__ subtreeCandidatePE,
    double* __restrict__ candidatePE,
    int* __restrict__ copyFlag,
    const int* __restrict__ active,
    int numGroups,
    unsigned int rngSeed) {

    for (int k = blockIdx.x * blockDim.x + threadIdx.x;
         k < numGroups;
         k += gridDim.x * blockDim.x) {

        copyFlag[k] = 0;

        if (!active[k]) continue;
        if (!subtreeHasCandidate[k]) continue;

        int totalN = nValid[k] + subtreeNValid[k];
        float acceptProb = (float)subtreeNValid[k] / (float)totalN;

        float u = nutsUniformRng(rngSeed, k, 0u);
        if (u < acceptProb) {
            candidatePE[k] = subtreeCandidatePE[k];
            copyFlag[k] = 1;
        }
        nValid[k] = totalN;
    }
}

/**
 * Set random direction (+1 or -1) per group for tree doubling.
 * Uses hash-based RNG. One thread per group.
 */
extern "C" __global__ void nutsSetDirection(
    int* __restrict__ direction,
    const int* __restrict__ active,
    int numGroups,
    unsigned int rngSeed) {

    for (int k = blockIdx.x * blockDim.x + threadIdx.x;
         k < numGroups;
         k += gridDim.x * blockDim.x) {

        if (active[k]) {
            float u = nutsUniformRng(rngSeed, k, 0u);
            direction[k] = (u < 0.5f) ? -1 : 1;
        } else {
            direction[k] = 1;  // doesn't matter for inactive
        }
    }
}

// ============================================================
// End of Phase 2 kernels
// ============================================================

/**
 * Initialize tree endpoints from current positions/velocities.
 * Sets xminus=xplus=candidate=posq, vminus=vplus=velm.
 */
extern "C" __global__ void nutsInitializeTree(
    const real4* __restrict__ posq,
    const mixed4* __restrict__ velm,
    real4* __restrict__ xminus,
    real4* __restrict__ xplus,
    real4* __restrict__ candidatePos,
    mixed4* __restrict__ vminus,
    mixed4* __restrict__ vplus,
    const int numGroupAtoms) {

    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < numGroupAtoms;
         i += gridDim.x * blockDim.x) {
        real4 pos = posq[i];
        mixed4 vel = velm[i];
        xminus[i] = pos;
        xplus[i] = pos;
        candidatePos[i] = pos;
        vminus[i] = vel;
        vplus[i] = vel;
    }
}
