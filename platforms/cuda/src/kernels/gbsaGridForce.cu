/* -------------------------------------------------------------------------- *
 *                              OpenMMGridForce                               *
 * -------------------------------------------------------------------------- *
 * CUDA kernels for GBSAGridForce - Grid-based Generalized Born solvation.
 *
 * Workflow:
 * 1. computeReceptorHCT - Interpolate grid, apply binned correction
 * 2. computeLigandHCT - Pairwise HCT within ligand (O(N²))
 * 3. computeBornRadii - OBC-II correction formula
 * 4. computeGBEnergy - Still equation pairwise (O(N²)) with forces
 * 5. computeSAEnergy - Optional ACE surface area term
 * -------------------------------------------------------------------------- */

#include "include/GridInterpolation.cuh"

// Physical constants
#define DIELECTRIC_OFFSET 0.009f
#define OBC_ALPHA 1.0f
#define OBC_BETA 0.8f
#define OBC_GAMMA 4.85f

/**
 * Compute HCT contribution from receptor via grid interpolation.
 * Applies binned correction for exact results at any ligand radius.
 */
extern "C" __global__ void computeReceptorHCT(
    const float4* __restrict__ posq,           // Positions (xyz) and charges (w)
    const int* __restrict__ particleIndices,   // Which particles to process
    const float* __restrict__ radii,           // Intrinsic radii (template)
    const int* __restrict__ gridCounts,        // [nx, ny, nz]
    const float* __restrict__ gridHctProbe,    // HCT values at probe radius
    const float* __restrict__ gridCorrectionN, // Correction N [nBins * nPoints]
    const float* __restrict__ gridCorrectionA, // Correction A
    const float* __restrict__ gridCorrectionB, // Correction B
    const float* __restrict__ rThresholds,     // Bin thresholds
    const int* __restrict__ groupStart,        // Group start indices
    int numGroups,
    float originX, float originY, float originZ,
    float gridSpacing,
    float probeRadius,
    int numBins,
    int totalParticles,
    int templateNumAtoms,
    float* __restrict__ hctReceptor            // Output: HCT from receptor
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= totalParticles) return;

    // Determine which group this atom belongs to and its position within the group
    int atomInGroup = idx;
    for (int g = 0; g < numGroups; g++) {
        int groupStartIdx = groupStart[g];
        int groupEndIdx = groupStart[g + 1];
        if (idx >= groupStartIdx && idx < groupEndIdx) {
            atomInGroup = idx - groupStartIdx;
            break;
        }
    }

    // Get particle index and template atom index
    int particleIdx = particleIndices[idx];
    int templateIdx = atomInGroup % templateNumAtoms;  // For multi-ligand, wrap around

    // Get position
    float4 pos = posq[particleIdx];
    float3 position = make_float3(pos.x, pos.y, pos.z);

    // Get radius and compute offset radius
    float R_i = radii[templateIdx];
    float R_i_off = R_i - DIELECTRIC_OFFSET;
    float R_probe_off = probeRadius - DIELECTRIC_OFFSET;

    // Grid dimensions
    int nx = gridCounts[0];
    int ny = gridCounts[1];
    int nz = gridCounts[2];
    int numPoints = nx * ny * nz;

    // Convert to grid coordinates
    float gx = (position.x - originX) / gridSpacing;
    float gy = (position.y - originY) / gridSpacing;
    float gz = (position.z - originZ) / gridSpacing;

    // Check bounds
    if (gx < 0 || gx >= nx - 1 || gy < 0 || gy >= ny - 1 || gz < 0 || gz >= nz - 1) {
        hctReceptor[idx] = 0.0f;
        return;
    }

    // Integer indices
    int i0 = (int)gx;
    int j0 = (int)gy;
    int k0 = (int)gz;

    // Fractional parts
    float fx = gx - i0;
    float fy = gy - j0;
    float fz = gz - k0;

    // Trilinear interpolation weights
    float w000 = (1-fx) * (1-fy) * (1-fz);
    float w001 = (1-fx) * (1-fy) * fz;
    float w010 = (1-fx) * fy * (1-fz);
    float w011 = (1-fx) * fy * fz;
    float w100 = fx * (1-fy) * (1-fz);
    float w101 = fx * (1-fy) * fz;
    float w110 = fx * fy * (1-fz);
    float w111 = fx * fy * fz;

    // Corner indices in flat array
    int nyz = ny * nz;
    int c000 = i0 * nyz + j0 * nz + k0;
    int c001 = c000 + 1;
    int c010 = c000 + nz;
    int c011 = c010 + 1;
    int c100 = c000 + nyz;
    int c101 = c100 + 1;
    int c110 = c100 + nz;
    int c111 = c110 + 1;

    // Interpolate HCT probe value
    float hct = w000 * gridHctProbe[c000] + w001 * gridHctProbe[c001] +
                w010 * gridHctProbe[c010] + w011 * gridHctProbe[c011] +
                w100 * gridHctProbe[c100] + w101 * gridHctProbe[c101] +
                w110 * gridHctProbe[c110] + w111 * gridHctProbe[c111];

    // Find appropriate bin for this radius
    int binIdx = numBins - 1;
    for (int b = 0; b < numBins; b++) {
        if (rThresholds[b] >= R_i_off) {
            binIdx = b;
            break;
        }
    }

    // Interpolate correction terms
    int binOffset = binIdx * numPoints;
    float N = w000 * gridCorrectionN[binOffset + c000] + w001 * gridCorrectionN[binOffset + c001] +
              w010 * gridCorrectionN[binOffset + c010] + w011 * gridCorrectionN[binOffset + c011] +
              w100 * gridCorrectionN[binOffset + c100] + w101 * gridCorrectionN[binOffset + c101] +
              w110 * gridCorrectionN[binOffset + c110] + w111 * gridCorrectionN[binOffset + c111];

    // Apply correction if N > 0.5 (threshold due to interpolation)
    if (N > 0.5f) {
        float A = w000 * gridCorrectionA[binOffset + c000] + w001 * gridCorrectionA[binOffset + c001] +
                  w010 * gridCorrectionA[binOffset + c010] + w011 * gridCorrectionA[binOffset + c011] +
                  w100 * gridCorrectionA[binOffset + c100] + w101 * gridCorrectionA[binOffset + c101] +
                  w110 * gridCorrectionA[binOffset + c110] + w111 * gridCorrectionA[binOffset + c111];

        float B = w000 * gridCorrectionB[binOffset + c000] + w001 * gridCorrectionB[binOffset + c001] +
                  w010 * gridCorrectionB[binOffset + c010] + w011 * gridCorrectionB[binOffset + c011] +
                  w100 * gridCorrectionB[binOffset + c100] + w101 * gridCorrectionB[binOffset + c101] +
                  w110 * gridCorrectionB[binOffset + c110] + w111 * gridCorrectionB[binOffset + c111];

        // Correction formula: dI = delta * (N - 0.25*A*sigma) + B * ln(R_i/R_probe)
        float invRi = 1.0f / R_i_off;
        float invRp = 1.0f / R_probe_off;
        float delta = invRi - invRp;
        float sigma = invRi + invRp;
        float correction = delta * (N - 0.25f * A * sigma) + B * logf(R_i_off / R_probe_off);
        hct += correction;
    }

    hctReceptor[idx] = hct;
}

/**
 * Compute HCT contribution from ligand-ligand pairwise interactions.
 * Uses standard HCT integral formula.
 */
extern "C" __global__ void computeLigandHCT(
    const float4* __restrict__ posq,
    const int* __restrict__ particleIndices,
    const float* __restrict__ radii,
    const float* __restrict__ scaleFactors,
    const int* __restrict__ exclusionAtoms,
    const int* __restrict__ exclusionStart,
    const int* __restrict__ groupStart,
    int numGroups,
    int templateNumAtoms,
    float* __restrict__ hctLigand
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Determine which group this atom belongs to
    int groupIdx = 0;
    int atomInGroup = idx;
    int groupStartIdx = 0;
    int groupEndIdx = 0;

    for (int g = 0; g < numGroups; g++) {
        groupStartIdx = groupStart[g];
        groupEndIdx = groupStart[g + 1];
        if (idx >= groupStartIdx && idx < groupEndIdx) {
            groupIdx = g;
            atomInGroup = idx - groupStartIdx;
            break;
        }
    }

    if (idx >= groupEndIdx) return;

    int particleIdx_i = particleIndices[idx];
    int templateIdx_i = atomInGroup % templateNumAtoms;

    float4 pos_i = posq[particleIdx_i];
    float R_i = radii[templateIdx_i];
    float R_i_off = R_i - DIELECTRIC_OFFSET;

    // Get exclusion range for this atom
    int exclStart = exclusionStart[templateIdx_i];
    int exclEnd = exclusionStart[templateIdx_i + 1];

    float hct = 0.0f;

    // Loop over other atoms in same group
    for (int jLocal = 0; jLocal < (groupEndIdx - groupStartIdx); jLocal++) {
        if (jLocal == atomInGroup) continue;

        int j = groupStartIdx + jLocal;
        int templateIdx_j = jLocal % templateNumAtoms;

        // Check exclusions
        bool excluded = false;
        for (int e = exclStart; e < exclEnd; e++) {
            if (exclusionAtoms[e] == templateIdx_j) {
                excluded = true;
                break;
            }
        }
        if (excluded) continue;

        int particleIdx_j = particleIndices[j];
        float4 pos_j = posq[particleIdx_j];

        float dx = pos_i.x - pos_j.x;
        float dy = pos_i.y - pos_j.y;
        float dz = pos_i.z - pos_j.z;
        float r = sqrtf(dx*dx + dy*dy + dz*dz);

        if (r < 1e-6f) continue;

        float R_j = radii[templateIdx_j];
        float R_j_off = R_j - DIELECTRIC_OFFSET;
        float S_j = R_j_off * scaleFactors[templateIdx_j];

        // HCT integral computation
        float r_plus_Sj = r + S_j;
        if (R_i_off >= r_plus_Sj) continue;  // No overlap

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

        hct += term;
    }

    hctLigand[idx] = hct;
}

/**
 * Compute Born radii from total HCT using OBC-II formula.
 */
extern "C" __global__ void computeBornRadii(
    const float* __restrict__ radii,
    const float* __restrict__ hctReceptor,
    const float* __restrict__ hctLigand,
    int numAtoms,
    int templateNumAtoms,
    float* __restrict__ bornRadii
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numAtoms) return;

    int templateIdx = idx % templateNumAtoms;
    float R_i = radii[templateIdx];
    float R_i_off = R_i - DIELECTRIC_OFFSET;

    float hctTotal = hctReceptor[idx] + hctLigand[idx];
    float psi = 0.5f * R_i_off * hctTotal;

    // OBC-II tanh correction
    float psi2 = psi * psi;
    float psi3 = psi2 * psi;
    float tanhArg = OBC_ALPHA * psi - OBC_BETA * psi2 + OBC_GAMMA * psi3;
    float tanhVal = tanhf(tanhArg);

    float denom = 1.0f / R_i_off - tanhVal / R_i;
    float bornRadius = (denom > 0.0f) ? (1.0f / denom) : R_i;

    // Clamp to reasonable range (only upper bound)
    bornRadius = fminf(bornRadius, 50.0f);  // Max 50 nm

    bornRadii[idx] = bornRadius;
}

/**
 * Compute GB energy using Still equation and accumulate forces.
 * E_gb = prefactor * sum_ij (q_i * q_j / f_gb)
 * where f_gb = sqrt(r_ij^2 + R_i*R_j*exp(-r_ij^2/(4*R_i*R_j)))
 */
extern "C" __global__ void computeGBEnergy(
    const float4* __restrict__ posq,
    const int* __restrict__ particleIndices,
    const float* __restrict__ charges,
    const float* __restrict__ bornRadii,
    const int* __restrict__ exclusionAtoms,
    const int* __restrict__ exclusionStart,
    const int* __restrict__ groupStart,
    int numGroups,
    int templateNumAtoms,
    float prefactor,
    unsigned long long* __restrict__ forceBuffer,
    float* __restrict__ groupEnergies,
    int paddedNumAtoms
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Determine group
    int groupIdx = 0;
    int atomInGroup = idx;
    int groupStartIdx = 0;
    int groupEndIdx = 0;

    for (int g = 0; g < numGroups; g++) {
        groupStartIdx = groupStart[g];
        groupEndIdx = groupStart[g + 1];
        if (idx >= groupStartIdx && idx < groupEndIdx) {
            groupIdx = g;
            atomInGroup = idx - groupStartIdx;
            break;
        }
    }

    if (idx >= groupEndIdx) return;

    int particleIdx_i = particleIndices[idx];
    int templateIdx_i = atomInGroup % templateNumAtoms;

    float4 pos_i = posq[particleIdx_i];
    float q_i = charges[templateIdx_i];
    float R_i = bornRadii[idx];

    int exclStart = exclusionStart[templateIdx_i];
    int exclEnd = exclusionStart[templateIdx_i + 1];

    float energy = 0.0f;
    float3 force = make_float3(0.0f, 0.0f, 0.0f);

    // Self energy term (0.5 factor for GB formula)
    energy += 0.5f * prefactor * q_i * q_i / R_i;

    // Pairwise terms (only count j > i to avoid double counting)
    int groupSize = groupEndIdx - groupStartIdx;
    for (int jLocal = atomInGroup + 1; jLocal < groupSize; jLocal++) {
        int j = groupStartIdx + jLocal;
        int templateIdx_j = jLocal % templateNumAtoms;

        // Check exclusions
        bool excluded = false;
        for (int e = exclStart; e < exclEnd; e++) {
            if (exclusionAtoms[e] == templateIdx_j) {
                excluded = true;
                break;
            }
        }
        if (excluded) continue;

        int particleIdx_j = particleIndices[j];
        float4 pos_j = posq[particleIdx_j];
        float q_j = charges[templateIdx_j];
        float R_j = bornRadii[j];

        float dx = pos_j.x - pos_i.x;
        float dy = pos_j.y - pos_i.y;
        float dz = pos_j.z - pos_i.z;
        float r2 = dx*dx + dy*dy + dz*dz;
        float r = sqrtf(r2);

        // Still equation: f_gb = sqrt(r^2 + R_i*R_j*exp(-r^2/(4*R_i*R_j)))
        float RiRj = R_i * R_j;
        float expArg = -r2 / (4.0f * RiRj);
        float expTerm = expf(expArg);
        float f_gb2 = r2 + RiRj * expTerm;
        float f_gb = sqrtf(f_gb2);
        float invFgb = 1.0f / f_gb;

        // Energy (each pair counted once with j > i loop)
        float pairEnergy = prefactor * q_i * q_j * invFgb;
        energy += pairEnergy;

        // Force = -dE/dr
        // dE/dr = prefactor * q_i * q_j * (-1/f_gb^2) * df_gb/dr
        // df_gb/dr = (1/f_gb) * (r - 0.25*RiRj*exp(-r^2/(4*RiRj))*(2r/(4*RiRj)))
        //          = (1/f_gb) * (r + 0.5*r*exp(-r^2/(4*RiRj))/RiRj * (-r^2/(4*RiRj) derivative? no...
        // Let me redo: f_gb^2 = r^2 + RiRj*exp(-r^2/(4*RiRj))
        // d(f_gb^2)/dr = 2r + RiRj * exp(...) * (-2r/(4*RiRj)) = 2r - 0.5*r*exp(...) = 2r*(1 - 0.25*exp(...))
        // df_gb/dr = (1/(2*f_gb)) * 2r * (1 - 0.25*exp(-r^2/(4*RiRj)))
        //          = r/f_gb * (1 - 0.25*expTerm)

        float dFgbDr = (r * invFgb) * (1.0f - 0.25f * expTerm);
        float dEdR = -prefactor * q_i * q_j * invFgb * invFgb * dFgbDr;

        // Force on i from j
        float invR = 1.0f / r;
        force.x -= dEdR * dx * invR;
        force.y -= dEdR * dy * invR;
        force.z -= dEdR * dz * invR;

        // Accumulate force on j (Newton's 3rd law)
        // Using atomic add for thread safety
        atomicAdd(&forceBuffer[particleIdx_j], static_cast<unsigned long long>((long long)(dEdR * dx * invR * 0x100000000)));
        atomicAdd(&forceBuffer[particleIdx_j + paddedNumAtoms], static_cast<unsigned long long>((long long)(dEdR * dy * invR * 0x100000000)));
        atomicAdd(&forceBuffer[particleIdx_j + 2*paddedNumAtoms], static_cast<unsigned long long>((long long)(dEdR * dz * invR * 0x100000000)));
    }

    // Accumulate force on i
    atomicAdd(&forceBuffer[particleIdx_i], static_cast<unsigned long long>((long long)(force.x * 0x100000000)));
    atomicAdd(&forceBuffer[particleIdx_i + paddedNumAtoms], static_cast<unsigned long long>((long long)(force.y * 0x100000000)));
    atomicAdd(&forceBuffer[particleIdx_i + 2*paddedNumAtoms], static_cast<unsigned long long>((long long)(force.z * 0x100000000)));

    // Accumulate group energy
    atomicAdd(&groupEnergies[groupIdx], energy);
}

/**
 * Compute surface area energy (ACE approximation).
 * E_sa = surfaceTension * sum_i (4*pi*(R_i + probe)^2 * (R_i/bornRadius)^6)
 */
extern "C" __global__ void computeSAEnergy(
    const float* __restrict__ radii,
    const float* __restrict__ bornRadii,
    const int* __restrict__ groupStart,
    int numGroups,
    int templateNumAtoms,
    float surfaceTension,
    float probeRadius,
    float* __restrict__ groupEnergies
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Determine group
    int groupIdx = 0;
    int atomInGroup = idx;
    int groupStartIdx = 0;
    int groupEndIdx = 0;

    for (int g = 0; g < numGroups; g++) {
        groupStartIdx = groupStart[g];
        groupEndIdx = groupStart[g + 1];
        if (idx >= groupStartIdx && idx < groupEndIdx) {
            groupIdx = g;
            atomInGroup = idx - groupStartIdx;
            break;
        }
    }

    if (idx >= groupEndIdx) return;

    int templateIdx = atomInGroup % templateNumAtoms;
    float R_i = radii[templateIdx];
    float B_i = bornRadii[idx];

    // ACE formula
    float Rprobe = R_i + probeRadius;
    float ratio = R_i / B_i;
    float ratio6 = ratio * ratio * ratio * ratio * ratio * ratio;
    float area = 4.0f * 3.14159265359f * Rprobe * Rprobe;
    float saEnergy = surfaceTension * area * ratio6;

    atomicAdd(&groupEnergies[groupIdx], saEnergy);
}

/**
 * Accumulate surface area derivatives into dE_dR.
 * dE_sa/dR_born = -6 * surfaceTension * 4*pi*(R_i + probe)^2 * R_i^6 / R_born^7
 *               = -6 * E_sa / R_born
 * Must be called after accumulateBornRadiiDerivatives to add to existing dE_dR.
 */
extern "C" __global__ void accumulateSADerivatives(
    const float* __restrict__ radii,
    const float* __restrict__ bornRadii,
    const int* __restrict__ groupStart,
    int numGroups,
    int templateNumAtoms,
    float surfaceTension,
    float probeRadius,
    float* __restrict__ dE_dR  // In/Out: accumulate dE_sa/dR_born
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Determine group
    int atomInGroup = idx;
    int groupEndIdx = 0;

    for (int g = 0; g < numGroups; g++) {
        int groupStartIdx = groupStart[g];
        groupEndIdx = groupStart[g + 1];
        if (idx >= groupStartIdx && idx < groupEndIdx) {
            atomInGroup = idx - groupStartIdx;
            break;
        }
    }

    if (idx >= groupEndIdx) return;

    int templateIdx = atomInGroup % templateNumAtoms;
    float R_i = radii[templateIdx];
    float R_born = bornRadii[idx];

    // ACE formula: E_sa = surfaceTension * 4*pi*(R_i + probe)^2 * (R_i/R_born)^6
    // Derivative: dE_sa/dR_born = -6 * E_sa / R_born
    float Rprobe = R_i + probeRadius;
    float ratio = R_i / R_born;
    float ratio6 = ratio * ratio * ratio * ratio * ratio * ratio;
    float area = 4.0f * 3.14159265359f * Rprobe * Rprobe;
    float E_sa = surfaceTension * area * ratio6;

    // Add SA derivative to dE_dR
    dE_dR[idx] += -6.0f * E_sa / R_born;
}

/**
 * Compute forces from receptor HCT grid interpolation gradient.
 * Uses analytical trilinear gradient for efficiency and accuracy.
 */
extern "C" __global__ void computeReceptorHCTGradientForce(
    const float4* __restrict__ posq,
    const int* __restrict__ particleIndices,
    const float* __restrict__ radii,
    const float* __restrict__ bornRadii,
    const float* __restrict__ hctReceptor,
    const float* __restrict__ hctLigand,
    const float* __restrict__ dE_dR,            // dE/dR_born from accumulation
    const int* __restrict__ gridCounts,
    const float* __restrict__ gridHctProbe,
    const float* __restrict__ gridCorrectionN,
    const float* __restrict__ gridCorrectionA,
    const float* __restrict__ gridCorrectionB,
    const float* __restrict__ rThresholds,
    const int* __restrict__ groupStart,
    int numGroups,
    float originX, float originY, float originZ,
    float gridSpacing,
    float probeRadius,
    int numBins,
    int totalParticles,
    int templateNumAtoms,
    unsigned long long* __restrict__ forceBuffer,
    int paddedNumAtoms
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= totalParticles) return;

    // Determine group and atom in group
    int atomInGroup = idx;
    for (int g = 0; g < numGroups; g++) {
        int groupStartIdx = groupStart[g];
        int groupEndIdx = groupStart[g + 1];
        if (idx >= groupStartIdx && idx < groupEndIdx) {
            atomInGroup = idx - groupStartIdx;
            break;
        }
    }

    int particleIdx = particleIndices[idx];
    int templateIdx = atomInGroup % templateNumAtoms;

    float4 pos = posq[particleIdx];
    float3 position = make_float3(pos.x, pos.y, pos.z);

    float R_i = radii[templateIdx];
    float R_i_off = R_i - DIELECTRIC_OFFSET;
    float R_probe_off = probeRadius - DIELECTRIC_OFFSET;
    float R_born_i = bornRadii[idx];

    // Compute dE_dHCT from dE_dR using OBC-II chain rule
    float hctTotal_i = hctReceptor[idx] + hctLigand[idx];
    float psi_i = 0.5f * R_i_off * hctTotal_i;

    float psi2 = psi_i * psi_i;
    float psi3 = psi2 * psi_i;
    float tanh_arg = OBC_ALPHA * psi_i - OBC_BETA * psi2 + OBC_GAMMA * psi3;
    float t = tanhf(tanh_arg);
    float sech2 = 1.0f - t * t;
    float darg_dpsi = OBC_ALPHA - 2.0f * OBC_BETA * psi_i + 3.0f * OBC_GAMMA * psi2;

    // dR_born/dpsi = R_born^2 * sech^2 * darg/dpsi / R_i
    float dR_dpsi_i = R_born_i * R_born_i * sech2 * darg_dpsi / R_i;

    // dE/dHCT = dE/dR * dR/dpsi * dpsi/dHCT = dE/dR * dR/dpsi * 0.5 * R_off
    float dEdHCT = dE_dR[idx] * dR_dpsi_i * 0.5f * R_i_off;

    int nx = gridCounts[0];
    int ny = gridCounts[1];
    int nz = gridCounts[2];
    int numPoints = nx * ny * nz;
    int nyz = ny * nz;

    // Find bin for this atom
    int binIdx = numBins - 1;
    for (int b = 0; b < numBins; b++) {
        if (rThresholds[b] >= R_i_off) {
            binIdx = b;
            break;
        }
    }
    int binOffset = binIdx * numPoints;

    // Compute grid coordinates
    float invSpacing = 1.0f / gridSpacing;
    float gx = (position.x - originX) * invSpacing;
    float gy = (position.y - originY) * invSpacing;
    float gz = (position.z - originZ) * invSpacing;

    float3 gradient = make_float3(0.0f, 0.0f, 0.0f);

    // Check bounds
    if (gx >= 0 && gx < nx - 1 && gy >= 0 && gy < ny - 1 && gz >= 0 && gz < nz - 1) {
        int i0 = (int)gx;
        int j0 = (int)gy;
        int k0 = (int)gz;
        float fx = gx - i0;
        float fy = gy - j0;
        float fz = gz - k0;

        // Corner indices
        int c000 = i0 * nyz + j0 * nz + k0;
        int c001 = c000 + 1;
        int c010 = c000 + nz;
        int c011 = c010 + 1;
        int c100 = c000 + nyz;
        int c101 = c100 + 1;
        int c110 = c100 + nz;
        int c111 = c110 + 1;

        // Load corner values for hctProbe
        float h000 = gridHctProbe[c000];
        float h001 = gridHctProbe[c001];
        float h010 = gridHctProbe[c010];
        float h011 = gridHctProbe[c011];
        float h100 = gridHctProbe[c100];
        float h101 = gridHctProbe[c101];
        float h110 = gridHctProbe[c110];
        float h111 = gridHctProbe[c111];

        // Trilinear interpolation weights
        float w000 = (1.0f - fx) * (1.0f - fy) * (1.0f - fz);
        float w001 = (1.0f - fx) * (1.0f - fy) * fz;
        float w010 = (1.0f - fx) * fy * (1.0f - fz);
        float w011 = (1.0f - fx) * fy * fz;
        float w100 = fx * (1.0f - fy) * (1.0f - fz);
        float w101 = fx * (1.0f - fy) * fz;
        float w110 = fx * fy * (1.0f - fz);
        float w111 = fx * fy * fz;

        // Load correction values
        float N000 = gridCorrectionN[binOffset + c000];
        float N001 = gridCorrectionN[binOffset + c001];
        float N010 = gridCorrectionN[binOffset + c010];
        float N011 = gridCorrectionN[binOffset + c011];
        float N100 = gridCorrectionN[binOffset + c100];
        float N101 = gridCorrectionN[binOffset + c101];
        float N110 = gridCorrectionN[binOffset + c110];
        float N111 = gridCorrectionN[binOffset + c111];

        // Interpolate N to check threshold
        float N_interp = w000*N000 + w001*N001 + w010*N010 + w011*N011 +
                         w100*N100 + w101*N101 + w110*N110 + w111*N111;

        // Analytical trilinear gradient for hctProbe
        // d/dfx = bilinear interp of (h1jk - h0jk) in yz plane
        float dh_dfx = (1.0f - fy) * (1.0f - fz) * (h100 - h000) +
                       (1.0f - fy) * fz * (h101 - h001) +
                       fy * (1.0f - fz) * (h110 - h010) +
                       fy * fz * (h111 - h011);

        float dh_dfy = (1.0f - fx) * (1.0f - fz) * (h010 - h000) +
                       (1.0f - fx) * fz * (h011 - h001) +
                       fx * (1.0f - fz) * (h110 - h100) +
                       fx * fz * (h111 - h101);

        float dh_dfz = (1.0f - fx) * (1.0f - fy) * (h001 - h000) +
                       (1.0f - fx) * fy * (h011 - h010) +
                       fx * (1.0f - fy) * (h101 - h100) +
                       fx * fy * (h111 - h110);

        // Gradient in real space: d/dx = d/dfx * dfx/dx = d/dfx / spacing
        gradient.x = dh_dfx * invSpacing;
        gradient.y = dh_dfy * invSpacing;
        gradient.z = dh_dfz * invSpacing;

        // Add correction gradient if N > 0.5
        if (N_interp > 0.5f) {
            // Load A and B correction values
            float A000 = gridCorrectionA[binOffset + c000];
            float A001 = gridCorrectionA[binOffset + c001];
            float A010 = gridCorrectionA[binOffset + c010];
            float A011 = gridCorrectionA[binOffset + c011];
            float A100 = gridCorrectionA[binOffset + c100];
            float A101 = gridCorrectionA[binOffset + c101];
            float A110 = gridCorrectionA[binOffset + c110];
            float A111 = gridCorrectionA[binOffset + c111];

            float B000 = gridCorrectionB[binOffset + c000];
            float B001 = gridCorrectionB[binOffset + c001];
            float B010 = gridCorrectionB[binOffset + c010];
            float B011 = gridCorrectionB[binOffset + c011];
            float B100 = gridCorrectionB[binOffset + c100];
            float B101 = gridCorrectionB[binOffset + c101];
            float B110 = gridCorrectionB[binOffset + c110];
            float B111 = gridCorrectionB[binOffset + c111];

            // Analytical gradients for N, A, B
            float dN_dfx = (1.0f - fy) * (1.0f - fz) * (N100 - N000) +
                           (1.0f - fy) * fz * (N101 - N001) +
                           fy * (1.0f - fz) * (N110 - N010) +
                           fy * fz * (N111 - N011);

            float dN_dfy = (1.0f - fx) * (1.0f - fz) * (N010 - N000) +
                           (1.0f - fx) * fz * (N011 - N001) +
                           fx * (1.0f - fz) * (N110 - N100) +
                           fx * fz * (N111 - N101);

            float dN_dfz = (1.0f - fx) * (1.0f - fy) * (N001 - N000) +
                           (1.0f - fx) * fy * (N011 - N010) +
                           fx * (1.0f - fy) * (N101 - N100) +
                           fx * fy * (N111 - N110);

            float dA_dfx = (1.0f - fy) * (1.0f - fz) * (A100 - A000) +
                           (1.0f - fy) * fz * (A101 - A001) +
                           fy * (1.0f - fz) * (A110 - A010) +
                           fy * fz * (A111 - A011);

            float dA_dfy = (1.0f - fx) * (1.0f - fz) * (A010 - A000) +
                           (1.0f - fx) * fz * (A011 - A001) +
                           fx * (1.0f - fz) * (A110 - A100) +
                           fx * fz * (A111 - A101);

            float dA_dfz = (1.0f - fx) * (1.0f - fy) * (A001 - A000) +
                           (1.0f - fx) * fy * (A011 - A010) +
                           fx * (1.0f - fy) * (A101 - A100) +
                           fx * fy * (A111 - A110);

            float dB_dfx = (1.0f - fy) * (1.0f - fz) * (B100 - B000) +
                           (1.0f - fy) * fz * (B101 - B001) +
                           fy * (1.0f - fz) * (B110 - B010) +
                           fy * fz * (B111 - B011);

            float dB_dfy = (1.0f - fx) * (1.0f - fz) * (B010 - B000) +
                           (1.0f - fx) * fz * (B011 - B001) +
                           fx * (1.0f - fz) * (B110 - B100) +
                           fx * fz * (B111 - B101);

            float dB_dfz = (1.0f - fx) * (1.0f - fy) * (B001 - B000) +
                           (1.0f - fx) * fy * (B011 - B010) +
                           fx * (1.0f - fy) * (B101 - B100) +
                           fx * fy * (B111 - B110);

            // Correction formula: hct += delta * (N - 0.25 * A * sigma) + B * log(R_i_off / R_probe_off)
            // d(correction)/dN = delta
            // d(correction)/dA = -0.25 * delta * sigma
            // d(correction)/dB = log(R_i_off / R_probe_off)
            float invRi = 1.0f / R_i_off;
            float invRp = 1.0f / R_probe_off;
            float delta = invRi - invRp;
            float sigma = invRi + invRp;
            float logTerm = logf(R_i_off / R_probe_off);

            float dCorr_dN = delta;
            float dCorr_dA = -0.25f * delta * sigma;
            float dCorr_dB = logTerm;

            // Total correction gradient
            gradient.x += (dCorr_dN * dN_dfx + dCorr_dA * dA_dfx + dCorr_dB * dB_dfx) * invSpacing;
            gradient.y += (dCorr_dN * dN_dfy + dCorr_dA * dA_dfy + dCorr_dB * dB_dfy) * invSpacing;
            gradient.z += (dCorr_dN * dN_dfz + dCorr_dA * dA_dfz + dCorr_dB * dB_dfz) * invSpacing;
        }
    }

    // Force = -dE/dHCT * gradient
    float3 force;
    force.x = -dEdHCT * gradient.x;
    force.y = -dEdHCT * gradient.y;
    force.z = -dEdHCT * gradient.z;

    // Accumulate force using fixed-point arithmetic
    atomicAdd(&forceBuffer[particleIdx], static_cast<unsigned long long>((long long)(force.x * 0x100000000)));
    atomicAdd(&forceBuffer[particleIdx + paddedNumAtoms], static_cast<unsigned long long>((long long)(force.y * 0x100000000)));
    atomicAdd(&forceBuffer[particleIdx + 2*paddedNumAtoms], static_cast<unsigned long long>((long long)(force.z * 0x100000000)));
}

/**
 * Accumulate dE/dR_born for each atom.
 * This must be called after computeGBEnergy to collect derivatives.
 */
extern "C" __global__ void accumulateBornRadiiDerivatives(
    const float4* __restrict__ posq,
    const int* __restrict__ particleIndices,
    const float* __restrict__ charges,
    const float* __restrict__ bornRadii,
    const int* __restrict__ exclusionAtoms,
    const int* __restrict__ exclusionStart,
    const int* __restrict__ groupStart,
    int numGroups,
    int templateNumAtoms,
    float prefactor,
    float* __restrict__ dE_dR  // Output: dE/dR_born for each atom
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Determine group
    int groupIdx = 0;
    int atomInGroup = idx;
    int groupStartIdx = 0;
    int groupEndIdx = 0;

    for (int g = 0; g < numGroups; g++) {
        groupStartIdx = groupStart[g];
        groupEndIdx = groupStart[g + 1];
        if (idx >= groupStartIdx && idx < groupEndIdx) {
            groupIdx = g;
            atomInGroup = idx - groupStartIdx;
            break;
        }
    }

    if (idx >= groupEndIdx) return;

    int particleIdx_i = particleIndices[idx];
    int templateIdx_i = atomInGroup % templateNumAtoms;

    float4 pos_i = posq[particleIdx_i];
    float q_i = charges[templateIdx_i];
    float R_i = bornRadii[idx];

    int exclStart = exclusionStart[templateIdx_i];
    int exclEnd = exclusionStart[templateIdx_i + 1];

    // Self energy derivative: dE_self/dR_i = -0.5 * prefactor * q_i^2 / R_i^2
    float dE_dRi = -0.5f * prefactor * q_i * q_i / (R_i * R_i);

    // Pairwise contributions
    int groupSize = groupEndIdx - groupStartIdx;
    for (int jLocal = 0; jLocal < groupSize; jLocal++) {
        if (jLocal == atomInGroup) continue;

        int j = groupStartIdx + jLocal;
        int templateIdx_j = jLocal % templateNumAtoms;

        // Check exclusions
        bool excluded = false;
        for (int e = exclStart; e < exclEnd; e++) {
            if (exclusionAtoms[e] == templateIdx_j) {
                excluded = true;
                break;
            }
        }
        if (excluded) continue;

        int particleIdx_j = particleIndices[j];
        float4 pos_j = posq[particleIdx_j];
        float q_j = charges[templateIdx_j];
        float R_j = bornRadii[j];

        float dx = pos_j.x - pos_i.x;
        float dy = pos_j.y - pos_i.y;
        float dz = pos_j.z - pos_i.z;
        float r2 = dx*dx + dy*dy + dz*dz;

        // f_gb computation
        float RiRj = R_i * R_j;
        float expArg = -r2 / (4.0f * RiRj);
        float expTerm = expf(expArg);
        float fgb2 = r2 + RiRj * expTerm;
        float fgb = sqrtf(fgb2);

        // d(fgb^2)/dR_i = R_j * expTerm * (1 + r2/(4*R_i^2*R_j))
        //               = R_j * expTerm + 0.25 * expTerm * r2 / R_i
        float dfgb2_dRi = R_j * expTerm + 0.25f * expTerm * r2 / R_i;
        float dfgb_dRi = dfgb2_dRi / (2.0f * fgb);

        // dE_pair/dR_i = -prefactor * q_i * q_j * dfgb_dRi / fgb^2
        float dE_pair_dRi = -prefactor * q_i * q_j * dfgb_dRi / fgb2;

        // Count each pair once (j>i contribution doubled)
        dE_dRi += 0.5f * dE_pair_dRi;  // Half because we count both i<j and i>j
    }

    dE_dR[idx] = dE_dRi;
}


/**
 * Compute forces from HCT chain rule through Born radii.
 * Must be called after accumulateBornRadiiDerivatives.
 */
extern "C" __global__ void computeHCTChainRuleForces(
    const float4* __restrict__ posq,
    const int* __restrict__ particleIndices,
    const float* __restrict__ radii,
    const float* __restrict__ scaleFactors,
    const float* __restrict__ bornRadii,
    const float* __restrict__ hctReceptor,
    const float* __restrict__ hctLigand,
    const float* __restrict__ dE_dR,
    const int* __restrict__ exclusionAtoms,
    const int* __restrict__ exclusionStart,
    const int* __restrict__ groupStart,
    int numGroups,
    int templateNumAtoms,
    unsigned long long* __restrict__ forceBuffer,
    int paddedNumAtoms
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Determine group
    int groupIdx = 0;
    int atomInGroup = idx;
    int groupStartIdx = 0;
    int groupEndIdx = 0;

    for (int g = 0; g < numGroups; g++) {
        groupStartIdx = groupStart[g];
        groupEndIdx = groupStart[g + 1];
        if (idx >= groupStartIdx && idx < groupEndIdx) {
            groupIdx = g;
            atomInGroup = idx - groupStartIdx;
            break;
        }
    }

    if (idx >= groupEndIdx) return;

    int particleIdx_i = particleIndices[idx];
    int templateIdx_i = atomInGroup % templateNumAtoms;

    float4 pos_i = posq[particleIdx_i];
    float R_i = radii[templateIdx_i];
    float R_i_off = R_i - DIELECTRIC_OFFSET;
    float R_born_i = bornRadii[idx];

    // Compute dR_born/dpsi using OBC-II derivative
    float hctTotal_i = hctReceptor[idx] + hctLigand[idx];
    float psi_i = 0.5f * R_i_off * hctTotal_i;

    float psi2 = psi_i * psi_i;
    float psi3 = psi2 * psi_i;
    float tanh_arg = OBC_ALPHA * psi_i - OBC_BETA * psi2 + OBC_GAMMA * psi3;
    float t = tanhf(tanh_arg);
    float sech2 = 1.0f - t * t;
    float darg_dpsi = OBC_ALPHA - 2.0f * OBC_BETA * psi_i + 3.0f * OBC_GAMMA * psi2;

    // dR_born/dpsi = R_born^2 * sech^2 * darg/dpsi / R_i
    float dR_dpsi_i = R_born_i * R_born_i * sech2 * darg_dpsi / R_i;

    // dE/dHCT_i = dE/dR_i * dR_i/dpsi_i * dpsi_i/dHCT_i
    // where dpsi/dHCT = 0.5 * R_off
    float dE_dHCT_i = dE_dR[idx] * dR_dpsi_i * 0.5f * R_i_off;

    int exclStart = exclusionStart[templateIdx_i];
    int exclEnd = exclusionStart[templateIdx_i + 1];

    float3 force = make_float3(0.0f, 0.0f, 0.0f);

    // Force from ligand-ligand HCT derivatives
    int groupSize = groupEndIdx - groupStartIdx;
    for (int jLocal = 0; jLocal < groupSize; jLocal++) {
        if (jLocal == atomInGroup) continue;

        int j = groupStartIdx + jLocal;
        int templateIdx_j = jLocal % templateNumAtoms;

        // Check exclusions
        bool excluded = false;
        for (int e = exclStart; e < exclEnd; e++) {
            if (exclusionAtoms[e] == templateIdx_j) {
                excluded = true;
                break;
            }
        }
        if (excluded) continue;

        int particleIdx_j = particleIndices[j];
        float4 pos_j = posq[particleIdx_j];
        float R_j = radii[templateIdx_j];
        float R_j_off = R_j - DIELECTRIC_OFFSET;
        float S_j = R_j_off * scaleFactors[templateIdx_j];

        float dx = pos_i.x - pos_j.x;
        float dy = pos_i.y - pos_j.y;
        float dz = pos_i.z - pos_j.z;
        float r2 = dx*dx + dy*dy + dz*dz;
        float r = sqrtf(r2);

        if (r < 1e-6f) continue;

        // Compute dHCT_i/dr for contribution of atom j to HCT of atom i
        float r_plus_Sj = r + S_j;
        if (R_i_off >= r_plus_Sj) continue;

        float r_minus_Sj = fabsf(r - S_j);

        // Determine case and compute derivative
        float dHCT_dr;

        if (R_i_off > r_minus_Sj) {
            // Case 1: l = 1/R_i (constant in r)
            float u = 1.0f / r_plus_Sj;
            float l = 1.0f / R_i_off;
            float u2 = u * u;
            float l2 = l * l;
            float S2 = S_j * S_j;
            float invr = 1.0f / r;
            float invr2 = invr * invr;

            float du_dr = -u2;
            float ln_u_l = logf(u / l);

            dHCT_dr = -du_dr  // = u²
                    + 0.25f * (u2 - l2)
                    + 0.5f * r * u * du_dr  // = -0.5 * r * u³
                    - 0.5f * invr2 * ln_u_l
                    + 0.5f * invr * (-1.0f) * du_dr  // = 0.5/r * u²
                    - 0.25f * S2 * invr2 * (l2 - u2)
                    + 0.25f * S2 * invr * 2.0f * u * (-du_dr);  // = 0.5 * S²/r * u³
        } else {
            // Case 2: l = 1/(r-S) for r > S
            float rms = r - S_j;
            float l = 1.0f / rms;
            float u = 1.0f / r_plus_Sj;
            float l2 = l * l;
            float u2 = u * u;
            float S2 = S_j * S_j;
            float invr = 1.0f / r;
            float invr2 = invr * invr;

            float dl_dr = -l2;
            float du_dr = -u2;
            float ln_u_l = logf(u / l);

            dHCT_dr = dl_dr - du_dr
                    + 0.25f * (u2 - l2)
                    + 0.5f * r * (u * du_dr - l * dl_dr)
                    - 0.5f * invr2 * ln_u_l
                    + 0.5f * invr * ((1.0f/u) * du_dr - (1.0f/l) * dl_dr)
                    - 0.25f * S2 * invr2 * (l2 - u2)
                    + 0.25f * S2 * invr * (2.0f * l * dl_dr - 2.0f * u * du_dr);
        }

        // Force contribution: F_i += -dE_dHCT_i * dHCT_i/dr * dr/dx_i
        float invr = 1.0f / r;
        float dE_dr_i = dE_dHCT_i * dHCT_dr;

        // This force acts on i due to HCT_i depending on position of j
        force.x -= dE_dr_i * dx * invr;
        force.y -= dE_dr_i * dy * invr;
        force.z -= dE_dr_i * dz * invr;

        // Also need force on i from HCT_j depending on position of i
        // Compute dE_dHCT_j
        float R_born_j = bornRadii[j];
        float hctTotal_j = hctReceptor[j] + hctLigand[j];
        float psi_j = 0.5f * R_j_off * hctTotal_j;
        float psi2_j = psi_j * psi_j;
        float psi3_j = psi2_j * psi_j;
        float tanh_arg_j = OBC_ALPHA * psi_j - OBC_BETA * psi2_j + OBC_GAMMA * psi3_j;
        float t_j = tanhf(tanh_arg_j);
        float sech2_j = 1.0f - t_j * t_j;
        float darg_dpsi_j = OBC_ALPHA - 2.0f * OBC_BETA * psi_j + 3.0f * OBC_GAMMA * psi2_j;
        float dR_dpsi_j = R_born_j * R_born_j * sech2_j * darg_dpsi_j / R_j;
        float dE_dHCT_j = dE_dR[j] * dR_dpsi_j * 0.5f * R_j_off;

        // Compute dHCT_j/dr for contribution of atom i to HCT of atom j
        float S_i = R_i_off * scaleFactors[templateIdx_i];
        float r_plus_Si = r + S_i;
        if (R_j_off < r_plus_Si) {
            float r_minus_Si = fabsf(r - S_i);
            float dHCT_j_dr;

            if (R_j_off > r_minus_Si) {
                // Case 1 for j
                float u = 1.0f / r_plus_Si;
                float l = 1.0f / R_j_off;
                float u2 = u * u;
                float l2 = l * l;
                float S2 = S_i * S_i;
                float invr_loc = 1.0f / r;
                float invr2 = invr_loc * invr_loc;

                float du_dr = -u2;
                float ln_u_l = logf(u / l);

                dHCT_j_dr = -du_dr
                          + 0.25f * (u2 - l2)
                          + 0.5f * r * u * du_dr
                          - 0.5f * invr2 * ln_u_l
                          + 0.5f * invr_loc * (-1.0f) * du_dr
                          - 0.25f * S2 * invr2 * (l2 - u2)
                          + 0.25f * S2 * invr_loc * 2.0f * u * (-du_dr);
            } else {
                // Case 2 for j
                float rms = r - S_i;
                float l = 1.0f / rms;
                float u = 1.0f / r_plus_Si;
                float l2 = l * l;
                float u2 = u * u;
                float S2 = S_i * S_i;
                float invr_loc = 1.0f / r;
                float invr2 = invr_loc * invr_loc;

                float dl_dr = -l2;
                float du_dr = -u2;
                float ln_u_l = logf(u / l);

                dHCT_j_dr = dl_dr - du_dr
                          + 0.25f * (u2 - l2)
                          + 0.5f * r * (u * du_dr - l * dl_dr)
                          - 0.5f * invr2 * ln_u_l
                          + 0.5f * invr_loc * ((1.0f/u) * du_dr - (1.0f/l) * dl_dr)
                          - 0.25f * S2 * invr2 * (l2 - u2)
                          + 0.25f * S2 * invr_loc * (2.0f * l * dl_dr - 2.0f * u * du_dr);
            }

            // Force on i from HCT_j
            float dE_dr_j = dE_dHCT_j * dHCT_j_dr;
            force.x += dE_dr_j * dx * invr;
            force.y += dE_dr_j * dy * invr;
            force.z += dE_dr_j * dz * invr;
        }
    }

    // Accumulate force
    atomicAdd(&forceBuffer[particleIdx_i], static_cast<unsigned long long>((long long)(force.x * 0x100000000)));
    atomicAdd(&forceBuffer[particleIdx_i + paddedNumAtoms], static_cast<unsigned long long>((long long)(force.y * 0x100000000)));
    atomicAdd(&forceBuffer[particleIdx_i + 2*paddedNumAtoms], static_cast<unsigned long long>((long long)(force.z * 0x100000000)));
}
