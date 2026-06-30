// Double-precision storage variants of the Hessian-chain kernels for
// IsolatedGBSAForce in PAIRWISE mode.
//
// The float-precision storage path (prepareHessianIntermediates,
// computeHCTJacobianPairwise, computeReceptorPairwiseHessian,
// computeBornCouplingMatrix, assembleGBSAHessian) accumulates each
// kernel's internal sums in double but truncates to float for the
// inter-kernel buffer. At Mpro scale (N_rec=9364) the J^T M J inner
// product over ~58^2 products of float-precision intermediates leaves
// a ~1% relative floor on near-cancelled Hessian elements. These
// variants do every step in double and store doubles between kernels.
//
// Upstream-shared scalar/position buffers (dE_dR, radii, charges,
// scaleFactors, posq, receptor*) are bound at context precision (real),
// matching the host uploads. Their values are cast to double on read.
//
// Constants OBC_ALPHA / OBC_BETA / OBC_GAMMA / DIELECTRIC_OFFSET are
// #defined in gbsaGridForce.cu; the file-encoder bundles all .cu files
// into one TU and gbsaGridForce.cu (G) precedes gbsaHessianDouble.cu (H).

// atomicAdd(double*, double) fallback for pre-sm_60 GPUs lives in
// kernels/include/AtomicAddDouble.cuh, included via commonHeaders.

// Final-hessian-buffer storage type. Default double; host compiles this
// source a second time with -DHBUF_T=float to get a float-storage
// module (same kernel names, picked by which module getKernel() targets).
// Compute stays in double throughout; only the dim3N*dim3N hessian
// buffer atomicAdd is narrowed. Intended for speed on platforms without
// hardware atomicAdd(double*, double) (e.g. pre-sm_60 Maxwell).
#ifndef HBUF_T
#define HBUF_T double
#endif


// Compute first and second r-derivatives of the HCT integrand in double.
// Mirrors the case dispatch in HCTChainRule.cuh::computeHCT_rDerivs but
// uses the algebraically simplified closed form
//   dI/dr  = (1/4)*(1 + S^2/r^2)*(u^2 - l^2) + ln(l/u)/(2 r^2)
//   d2I/dr2 = -(S^2/(2 r^3))*(u^2 - l^2)
//             + (1/2)*(1 + S^2/r^2) * (l^3*sign(r-S) - u^3) case2
//                                       (-u^3)               case1
//             - ln(l/u)/r^3
//             + ( -sign(r-S)*(l-u)/(2 r^2) case2 with r>S
//                 -(l-u)/(2 r^2)           case2 with r<=S (Tinker is a separate add)
//                  u/(2 r^2)               case1 )
// with l = 1/R_probe (case1, when R_probe > |r-S|) or 1/|r-S| (case2,
// when R_probe <= |r-S|), and u = 1/(r+S).
//
// Tinker correction (atom i completely inside atom j, i.e. R_probe <
// (S - r)) adds +2*(1/R_probe - 1/(S - r)) to the HCT integral. The
// chain-rule derivatives become
//   dI_tinker/dr  = -2 / (S - r)^2
//   d2I_tinker/dr2 = -4 / (S - r)^3
// We include them here so the Mpro Hessian agrees with the JAX
// reference (which has the same correction in hct_term).
__device__ __forceinline__ void computeHCT_r12_double(
    double r, double S, double R_probe, double& I1, double& I2)
{
    double r_minus_S = r - S;
    double abs_r_minus_S = fabs(r_minus_S);
    double sgn_r_minus_S = (r_minus_S >= 0.0) ? 1.0 : -1.0;
    double u = 1.0 / (r + S);
    double l;
    bool case1 = (R_probe > abs_r_minus_S);
    if (case1) {
        l = 1.0 / R_probe;
    } else {
        l = 1.0 / abs_r_minus_S;
    }

    double l2 = l * l;
    double u2 = u * u;
    double l3 = l2 * l;
    double u3 = u2 * u;
    double r2 = r * r;
    double r3 = r2 * r;
    double S2 = S * S;
    double one_plus_S2_over_r2 = 1.0 + S2 / r2;
    double log_l_over_u = log(l / u);

    I1 = 0.25 * one_plus_S2_over_r2 * (u2 - l2) + log_l_over_u / (2.0 * r2);

    double term1 = -(S2 / (2.0 * r3)) * (u2 - l2);
    double term3 = -log_l_over_u / r3;
    if (case1) {
        I2 = term1 + 0.5 * one_plus_S2_over_r2 * (-u3) + term3 + u / (2.0 * r2);
    } else {
        // For case2 with r<S the chain-rule signs flip because dl/dr =
        // d(1/|r-S|)/dr = -sgn(r-S)*l^2. The l^3 term picks up sgn,
        // the (l-u) term in d2I picks up sgn on the l part.
        double signed_l3 = sgn_r_minus_S * l3;
        double signed_l = sgn_r_minus_S * l;
        I2 = term1 + 0.5 * one_plus_S2_over_r2 * (signed_l3 - u3) + term3
           - (signed_l - u) / (2.0 * r2);
    }

    // Tinker correction kicks in when R_probe < (S - r), i.e. atom is
    // completely inside the other.  case1 means R_probe > |r-S| so it
    // can't coincide with Tinker (Tinker needs R_probe < S - r and r<S,
    // both implying |r-S| = S-r > R_probe, contradiction with case1).
    // So Tinker only fires in case2-with-r<S.
    if (R_probe < (S - r)) {
        double S_minus_r = S - r;
        double inv_Smr = 1.0 / S_minus_r;
        double inv_Smr2 = inv_Smr * inv_Smr;
        I1 += -2.0 * inv_Smr2;
        I2 += -4.0 * inv_Smr2 * inv_Smr;
    }
}


// Convert the fixed-point receptor->ligand HCT accumulator to double.
// The existing convertTiledHCTToFloat downcasts to float, losing the
// ~2^-32 absolute precision the fixed-point sum delivers. Kept here
// for completeness, but Born radii in the Hessian path are now seeded
// by computeHctReceptorPairwiseDouble instead — the fixed-point
// accumulator only holds the SUM of float-precision pair terms, so
// converting it doesn't recover JAX-grade double precision.
extern "C" __global__ void convertTiledHCTToDouble(
    const unsigned long long* __restrict__ hctFixed,
    double* __restrict__ hctDouble,
    int numAtoms
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numAtoms) return;
    hctDouble[i] = (double)((long long)hctFixed[i]) / (double)0x100000000ULL;
}


// Receptor->ligand HCT integral, computed in double from scratch (no
// tiling, one thread per ligand atom, loop over all receptor atoms).
// Mirrors jax_gbsa_reference.hct_term so the Mpro Hessian sees the
// same Born radii JAX autodiffs against. Slower than the production
// tiled kernel but only runs once per Hessian eval.
extern "C" __global__ void computeHctReceptorPairwiseDouble(
    const real4* __restrict__ posq,
    const int* __restrict__ particleIndices,
    const real* __restrict__ ligandRadii,
    const int* __restrict__ groupStart,
    int numGroups,
    int templateNumAtoms,
    const real4* __restrict__ receptorPositions,
    const real* __restrict__ receptorRadii,
    const real* __restrict__ receptorScaleFactors,
    int numReceptorAtoms,
    int totalParticles,
    double* __restrict__ hctReceptorOut
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= totalParticles) return;

    int atomInGroup = idx;
    int groupEnd = 0;
    for (int g = 0; g < numGroups; g++) {
        int gs = groupStart[g];
        groupEnd = groupStart[g + 1];
        if (idx >= gs && idx < groupEnd) {
            atomInGroup = idx - gs;
            break;
        }
    }
    if (idx >= groupEnd) {
        hctReceptorOut[idx] = 0.0;
        return;
    }

    int templateIdx = atomInGroup % templateNumAtoms;
    int particleIdx = particleIndices[idx];
    real4 pos_f = posq[particleIdx];
    double pk_x = (double)pos_f.x;
    double pk_y = (double)pos_f.y;
    double pk_z = (double)pos_f.z;
    double R_k_off = (double)ligandRadii[templateIdx] - (double)DIELECTRIC_OFFSET;

    double hct = 0.0;
    for (int rj = 0; rj < numReceptorAtoms; rj++) {
        real4 pj = receptorPositions[rj];
        double dx = pk_x - (double)pj.x;
        double dy = pk_y - (double)pj.y;
        double dz = pk_z - (double)pj.z;
        double r2 = dx*dx + dy*dy + dz*dz;
        double r = sqrt(r2);
        if (r < 1e-10) continue;

        double R_rj_off = (double)receptorRadii[rj] - (double)DIELECTRIC_OFFSET;
        double S = R_rj_off * (double)receptorScaleFactors[rj];
        double r_plus_S = r + S;
        if (R_k_off >= r_plus_S) continue;

        double r_minus_S = fabs(r - S);
        double l = (R_k_off > r_minus_S) ? (1.0 / R_k_off) : (1.0 / r_minus_S);
        double u = 1.0 / r_plus_S;
        double l2 = l * l, u2 = u * u;
        double term = (l - u)
                    + 0.25 * r * (u2 - l2)
                    + 0.5 * log(u / l) / r
                    + 0.25 * S * S * (l2 - u2) / r;
        if (R_k_off < (S - r)) term += 2.0 * (1.0 / R_k_off - l);
        hct += term;
    }
    hctReceptorOut[idx] = hct;
}


// Ligand-ligand HCT integral in double. Same formula as
// computeIsolatedLigandHCT but every step is double-precision so the
// Hessian's hctTotal = hctReceptor + hctLigand is consistent.
extern "C" __global__ void computeHctLigandPairwiseDouble(
    const real4* __restrict__ posq,
    const int* __restrict__ particleIndices,
    const real* __restrict__ ligandRadii,
    const real* __restrict__ ligandScaleFactors,
    const int* __restrict__ groupStart,
    int numGroups,
    int templateNumAtoms,
    int totalParticles,
    double* __restrict__ hctLigandOut
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= totalParticles) return;

    int atomInGroup = idx;
    int groupStartIdx = 0;
    int groupEnd = 0;
    for (int g = 0; g < numGroups; g++) {
        groupStartIdx = groupStart[g];
        groupEnd = groupStart[g + 1];
        if (idx >= groupStartIdx && idx < groupEnd) {
            atomInGroup = idx - groupStartIdx;
            break;
        }
    }
    if (idx >= groupEnd) {
        hctLigandOut[idx] = 0.0;
        return;
    }

    int templateIdx_i = atomInGroup % templateNumAtoms;
    int particleIdx_i = particleIndices[idx];
    real4 pos_f_i = posq[particleIdx_i];
    double pi_x = (double)pos_f_i.x;
    double pi_y = (double)pos_f_i.y;
    double pi_z = (double)pos_f_i.z;
    double R_i_off = (double)ligandRadii[templateIdx_i] - (double)DIELECTRIC_OFFSET;

    double hct = 0.0;
    int groupSize = groupEnd - groupStartIdx;
    for (int jLocal = 0; jLocal < groupSize; jLocal++) {
        if (jLocal == atomInGroup) continue;
        int j = groupStartIdx + jLocal;
        int templateIdx_j = jLocal % templateNumAtoms;
        int particleIdx_j = particleIndices[j];
        real4 pos_f_j = posq[particleIdx_j];
        double dx = pi_x - (double)pos_f_j.x;
        double dy = pi_y - (double)pos_f_j.y;
        double dz = pi_z - (double)pos_f_j.z;
        double r2 = dx*dx + dy*dy + dz*dz;
        double r = sqrt(r2);
        if (r < 1e-10) continue;

        double R_j_off = (double)ligandRadii[templateIdx_j] - (double)DIELECTRIC_OFFSET;
        double S = R_j_off * (double)ligandScaleFactors[templateIdx_j];
        double r_plus_S = r + S;
        if (R_i_off >= r_plus_S) continue;

        double r_minus_S = fabs(r - S);
        double l = (R_i_off > r_minus_S) ? (1.0 / R_i_off) : (1.0 / r_minus_S);
        double u = 1.0 / r_plus_S;
        double l2 = l * l, u2 = u * u;
        double term = (l - u)
                    + 0.25 * r * (u2 - l2)
                    + 0.5 * log(u / l) / r
                    + 0.25 * S * S * (l2 - u2) / r;
        if (R_i_off < (S - r)) term += 2.0 * (1.0 / R_i_off - l);
        hct += term;
    }
    hctLigandOut[idx] = hct;
}


// OBC2 Born radii in double from double hctReceptor + double hctLigand.
// Both upstream HCT inputs come from the double pair-loop kernels
// (computeHctReceptorPairwiseDouble + computeHctLigandPairwiseDouble).
extern "C" __global__ void computeBornRadiiOBCDouble(
    const real* __restrict__ radii,
    const double* __restrict__ hctReceptorD,
    const double* __restrict__ hctLigand,
    int numAtoms,
    int templateNumAtoms,
    double* __restrict__ bornRadiiOut
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numAtoms) return;

    int templateIdx = idx % templateNumAtoms;
    double R_i = (double)radii[templateIdx];
    double R_i_off = R_i - (double)DIELECTRIC_OFFSET;

    double hctTotal = hctReceptorD[idx] + hctLigand[idx];
    double psi = 0.5 * R_i_off * hctTotal;
    double psi2 = psi * psi;
    double tanhArg = (double)OBC_ALPHA * psi
                   - (double)OBC_BETA * psi2
                   + (double)OBC_GAMMA * psi2 * psi;
    double tanhVal = tanh(tanhArg);

    double denom = 1.0 / R_i_off - tanhVal / R_i;
    double R_born = (denom > 0.0) ? (1.0 / denom) : R_i;
    if (R_born > 50.0) R_born = 50.0;

    bornRadiiOut[idx] = R_born;
}


// Ligand-only GB energy derivative dE_ligGB/dR_k (no cross/desolv), in double.
// The production dE/dR conflates the ligand-GB and cross-term Born derivatives
// in PAIRWISE mode; the Hessian core needs the ligand-GB-only part so the
// receptor-desolvation/cross R-kernels (which add the cross part separately)
// do not double-count it. GB has no exclusions (all intra-group pairs).
extern "C" __global__ void computeLigandGBBornDerivDouble(
    const real4* __restrict__ posq,
    const int* __restrict__ particleIndices,
    const real* __restrict__ charges,
    const double* __restrict__ bornRadii,
    const int* __restrict__ groupStart,
    int numGroups,
    int templateNumAtoms,
    float prefactor,
    int totalParticles,
    real* __restrict__ dE_dR_out)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= totalParticles) return;
    int gs = 0, ge = 0;
    for (int g = 0; g < numGroups; g++) {
        gs = groupStart[g]; ge = groupStart[g + 1];
        if (idx >= gs && idx < ge) break;
    }
    if (idx >= ge) return;
    int atomInGroup = idx - gs;
    int tk = atomInGroup % templateNumAtoms;
    double pf = (double)prefactor;
    double q_k = (double)charges[tk];
    double R_k = bornRadii[idx];
    real4 pk = posq[particleIndices[idx]];
    double g = -0.5 * pf * q_k * q_k / (R_k * R_k);   // self term
    for (int lLocal = 0; lLocal < ge - gs; lLocal++) {
        if (lLocal == atomInGroup) continue;
        int l = gs + lLocal;
        int tl = lLocal % templateNumAtoms;
        double q_l = (double)charges[tl];
        double R_l = bornRadii[l];
        real4 pl = posq[particleIndices[l]];
        double dx = (double)pk.x - (double)pl.x;
        double dy = (double)pk.y - (double)pl.y;
        double dz = (double)pk.z - (double)pl.z;
        double r2 = dx*dx + dy*dy + dz*dz;
        if (r2 < 1e-20) continue;
        double RkRl = R_k * R_l;
        double et = exp(-r2 / (4.0 * RkRl));
        double f2 = r2 + RkRl * et;
        double f = sqrt(f2);
        double df2_dRk = et * (R_l + 0.25 * r2 / R_k);
        double df_dRk = df2_dRk / (2.0 * f);
        g += -pf * q_k * q_l / f2 * df_dRk;
    }
    dE_dR_out[idx] = (real)g;
}


// Per-atom OBC2 transform: psi_s -> Born radius and chain-rule factors
// dR/dPsi, d2R/dPsi2, dE/dHCT. Reads double bornRadii + double
// hctReceptor (recomputed for the Hessian path from the fixed-point
// buffer) plus hctLigand cast to double. dE_dR_in is read at context
// precision (real) and cast to double.
extern "C" __global__ void prepareHessianIntermediatesDouble(
    const real* __restrict__ radii,
    const double* __restrict__ bornRadii,
    const double* __restrict__ hctReceptor,
    const double* __restrict__ hctLigand,
    const real* __restrict__ dE_dR_in,
    int numAtoms,
    int templateNumAtoms,
    double* __restrict__ dR_dPsi_out,
    double* __restrict__ d2R_dPsi2_out,
    double* __restrict__ dE_dHCT_out
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numAtoms) return;

    int templateIdx = idx % templateNumAtoms;
    double R_i = (double)radii[templateIdx];
    double R_i_off = R_i - (double)DIELECTRIC_OFFSET;
    double R_born = bornRadii[idx];

    double hctTotal = hctReceptor[idx] + hctLigand[idx];
    double psi_s = 0.5 * R_i_off * hctTotal;
    double psi_s2 = psi_s * psi_s;
    double tanh_arg = (double)OBC_ALPHA * psi_s
                    - (double)OBC_BETA * psi_s2
                    + (double)OBC_GAMMA * psi_s2 * psi_s;
    double t = tanh(tanh_arg);
    double sech2 = 1.0 - t * t;

    double darg_dpsi_s = (double)OBC_ALPHA
                       - 2.0 * (double)OBC_BETA * psi_s
                       + 3.0 * (double)OBC_GAMMA * psi_s2;
    double d2arg_dpsi_s2 = -2.0 * (double)OBC_BETA
                         + 6.0 * (double)OBC_GAMMA * psi_s;

    double dpsi_s_dPsi = 0.5 * R_i_off;
    double D_const = dpsi_s_dPsi / R_i;

    double dR_dPsi = R_born * R_born * sech2 * darg_dpsi_s * D_const;

    double A_const = R_born * R_born;
    double B_const = sech2;
    double C_const = darg_dpsi_s;

    double dA_dPsi = 2.0 * R_born * dR_dPsi;
    double darg_dPsi = darg_dpsi_s * dpsi_s_dPsi;
    double dB_dPsi = -2.0 * sech2 * t * darg_dPsi;
    double dC_dPsi = d2arg_dpsi_s2 * dpsi_s_dPsi;

    double d2R_dPsi2 = (dA_dPsi * B_const * C_const
                       + A_const * dB_dPsi * C_const
                       + A_const * B_const * dC_dPsi) * D_const;

    dR_dPsi_out[idx]     = dR_dPsi;
    d2R_dPsi2_out[idx]   = d2R_dPsi2;
    dE_dHCT_out[idx]     = (double)dE_dR_in[idx] * dR_dPsi;
}


// Mirrors computeHCTJacobianPairwise in isolatedGBSA.cu but writes a
// double-precision jacobian. Internal accumulators were already double
// in the float version; here we also do the per-pair math in double.
extern "C" __global__ void computeHCTJacobianPairwiseDouble(
    const real4* __restrict__ posq,
    const int* __restrict__ particleIndices,
    const real* __restrict__ radii,
    const real* __restrict__ scaleFactors,
    const int* __restrict__ exclusionAtoms,
    const int* __restrict__ exclusionStart,
    const int* __restrict__ groupStart,
    int numGroups,
    int templateNumAtoms,
    const real4* __restrict__ receptorPositions,
    const real* __restrict__ receptorRadii,
    const real* __restrict__ receptorScaleFactors,
    int numReceptorAtoms,
    int totalParticles,
    double* __restrict__ jacobian
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int atomInGroup = idx;
    int groupStartIdx = 0;
    int groupEndIdx = 0;
    for (int g = 0; g < numGroups; g++) {
        groupStartIdx = groupStart[g];
        groupEndIdx = groupStart[g + 1];
        if (idx >= groupStartIdx && idx < groupEndIdx) {
            atomInGroup = idx - groupStartIdx;
            break;
        }
    }
    if (idx >= groupEndIdx) return;

    int dim3N = 3 * totalParticles;
    for (int c = 0; c < dim3N; c++) {
        jacobian[idx * dim3N + c] = 0.0;
    }

    int particleIdx_k = particleIndices[idx];
    int templateIdx_k = atomInGroup % templateNumAtoms;
    real4 pos_k_f = posq[particleIdx_k];
    double pk_x = (double)pos_k_f.x;
    double pk_y = (double)pos_k_f.y;
    double pk_z = (double)pos_k_f.z;
    double R_k = (double)radii[templateIdx_k];
    double R_k_off = R_k - (double)DIELECTRIC_OFFSET;

    double jx_self = 0.0, jy_self = 0.0, jz_self = 0.0;

    // Receptor pairwise sum (only self-diagonal block of J accumulates;
    // receptor atoms are frozen so off-diagonal entries are zero).
    for (int rj = 0; rj < numReceptorAtoms; rj++) {
        real4 pj = receptorPositions[rj];
        double dx = pk_x - (double)pj.x;
        double dy = pk_y - (double)pj.y;
        double dz = pk_z - (double)pj.z;
        double r2 = dx*dx + dy*dy + dz*dz;
        double r = sqrt(r2);
        if (r < 1e-9) continue;

        double R_rj = (double)receptorRadii[rj];
        double R_rj_off = R_rj - (double)DIELECTRIC_OFFSET;
        double S_rj = R_rj_off * (double)receptorScaleFactors[rj];
        if (R_k_off >= r + S_rj) continue;

        double I1, I2;
        computeHCT_r12_double(r, S_rj, R_k_off, I1, I2);
        double invr = 1.0 / r;
        jx_self += I1 * dx * invr;
        jy_self += I1 * dy * invr;
        jz_self += I1 * dz * invr;
    }

    // Ligand-ligand pairwise contribution (self + cross).
    int exclStart = exclusionStart[templateIdx_k];
    int exclEnd = exclusionStart[templateIdx_k + 1];
    int groupSize = groupEndIdx - groupStartIdx;
    for (int jLocal = 0; jLocal < groupSize; jLocal++) {
        if (jLocal == atomInGroup) continue;
        int j = groupStartIdx + jLocal;
        int templateIdx_j = jLocal % templateNumAtoms;

        bool excluded = false;
        for (int e = exclStart; e < exclEnd; e++) {
            if (exclusionAtoms[e] == templateIdx_j) { excluded = true; break; }
        }
        if (excluded) continue;

        int particleIdx_j = particleIndices[j];
        real4 pos_j_f = posq[particleIdx_j];
        double pj_x = (double)pos_j_f.x;
        double pj_y = (double)pos_j_f.y;
        double pj_z = (double)pos_j_f.z;
        double R_j = (double)radii[templateIdx_j];
        double R_j_off = R_j - (double)DIELECTRIC_OFFSET;
        double S_j = R_j_off * (double)scaleFactors[templateIdx_j];

        double dx = pk_x - pj_x;
        double dy = pk_y - pj_y;
        double dz = pk_z - pj_z;
        double r2 = dx*dx + dy*dy + dz*dz;
        double r = sqrt(r2);
        if (r < 1e-9) continue;
        if (R_k_off >= r + S_j) continue;

        double I1, I2;
        computeHCT_r12_double(r, S_j, R_k_off, I1, I2);
        double invr = 1.0 / r;
        jx_self += I1 * dx * invr;
        jy_self += I1 * dy * invr;
        jz_self += I1 * dz * invr;
        // Off-diagonal entries (single contribution per j).
        jacobian[idx * dim3N + 3 * j + 0] = -I1 * dx * invr;
        jacobian[idx * dim3N + 3 * j + 1] = -I1 * dy * invr;
        jacobian[idx * dim3N + 3 * j + 2] = -I1 * dz * invr;
    }

    jacobian[idx * dim3N + 3 * idx + 0] = jx_self;
    jacobian[idx * dim3N + 3 * idx + 1] = jy_self;
    jacobian[idx * dim3N + 3 * idx + 2] = jz_self;
}


// Mirrors computeReceptorPairwiseHessian but stores double output.
extern "C" __global__ void computeReceptorPairwiseHessianDouble(
    const real4* __restrict__ posq,
    const int* __restrict__ particleIndices,
    const real* __restrict__ radii,
    const int* __restrict__ groupStart,
    int numGroups,
    int templateNumAtoms,
    const real4* __restrict__ receptorPositions,
    const real* __restrict__ receptorRadii,
    const real* __restrict__ receptorScaleFactors,
    int numReceptorAtoms,
    int totalParticles,
    double* __restrict__ hessianOut       // [N * 6]: xx, yy, zz, xy, xz, yz
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int atomInGroup = idx;
    int groupEndIdx = 0;
    for (int g = 0; g < numGroups; g++) {
        int gs = groupStart[g];
        groupEndIdx = groupStart[g + 1];
        if (idx >= gs && idx < groupEndIdx) {
            atomInGroup = idx - gs;
            break;
        }
    }
    if (idx >= groupEndIdx) return;

    int templateIdx = atomInGroup % templateNumAtoms;
    int particleIdx = particleIndices[idx];
    real4 pk_f = posq[particleIdx];
    double pk_x = (double)pk_f.x;
    double pk_y = (double)pk_f.y;
    double pk_z = (double)pk_f.z;
    double R_k = (double)radii[templateIdx];
    double R_k_off = R_k - (double)DIELECTRIC_OFFSET;

    double Hxx = 0.0, Hyy = 0.0, Hzz = 0.0;
    double Hxy = 0.0, Hxz = 0.0, Hyz = 0.0;

    for (int rj = 0; rj < numReceptorAtoms; rj++) {
        real4 pj = receptorPositions[rj];
        double dx = pk_x - (double)pj.x;
        double dy = pk_y - (double)pj.y;
        double dz = pk_z - (double)pj.z;
        double r2 = dx*dx + dy*dy + dz*dz;
        double r = sqrt(r2);
        if (r < 1e-9) continue;

        double R_rj = (double)receptorRadii[rj];
        double R_rj_off = R_rj - (double)DIELECTRIC_OFFSET;
        double S_rj = R_rj_off * (double)receptorScaleFactors[rj];
        if (R_k_off >= r + S_rj) continue;

        double I1, I2;
        computeHCT_r12_double(r, S_rj, R_k_off, I1, I2);

        double invr = 1.0 / r;
        double rhx = dx * invr;
        double rhy = dy * invr;
        double rhz = dz * invr;

        // d2I/dx_a dx_b = (dI/dr)*(d_ab - rh_a*rh_b)/r + (d2I/dr2)*rh_a*rh_b
        //               = A*d_ab + B*rh_a*rh_b   with  A = (dI/dr)/r, B = (d2I/dr2) - A
        double A = I1 * invr;
        double B = I2 - A;

        Hxx += A + B * rhx * rhx;
        Hyy += A + B * rhy * rhy;
        Hzz += A + B * rhz * rhz;
        Hxy += B * rhx * rhy;
        Hxz += B * rhx * rhz;
        Hyz += B * rhy * rhz;
    }

    hessianOut[idx * 6 + 0] = Hxx;
    hessianOut[idx * 6 + 1] = Hyy;
    hessianOut[idx * 6 + 2] = Hzz;
    hessianOut[idx * 6 + 3] = Hxy;
    hessianOut[idx * 6 + 4] = Hxz;
    hessianOut[idx * 6 + 5] = Hyz;
}


// Mirrors computeBornCouplingMatrix; reads float bornRadii but does all
// Still-pair math in double and stores double M.
extern "C" __global__ void computeBornCouplingMatrixDouble(
    const real4* __restrict__ posq,
    const int* __restrict__ particleIndices,
    const real* __restrict__ charges,
    const real* __restrict__ intrinsicRadii,
    const double* __restrict__ bornRadii,
    const double* __restrict__ dR_dPsi,
    const double* __restrict__ d2R_dPsi2,
    const real* __restrict__ dE_dR,
    const int* __restrict__ exclusionAtoms,
    const int* __restrict__ exclusionStart,
    const int* __restrict__ groupStart,
    int numGroups,
    int templateNumAtoms,
    float prefactor,
    int includeSA,
    float surfaceTension,
    float probeRadiusVal,
    int totalParticles,
    double* __restrict__ couplingMatrix
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int atomInGroup = idx;
    int groupStartIdx = 0;
    int groupEndIdx = 0;
    for (int g = 0; g < numGroups; g++) {
        groupStartIdx = groupStart[g];
        groupEndIdx = groupStart[g + 1];
        if (idx >= groupStartIdx && idx < groupEndIdx) {
            atomInGroup = idx - groupStartIdx;
            break;
        }
    }
    if (idx >= groupEndIdx) return;

    int templateIdx_k = atomInGroup % templateNumAtoms;
    int particleIdx_k = particleIndices[idx];
    real4 pos_k_f = posq[particleIdx_k];
    double pk_x = (double)pos_k_f.x;
    double pk_y = (double)pos_k_f.y;
    double pk_z = (double)pos_k_f.z;
    double q_k = (double)charges[templateIdx_k];
    double R_k = bornRadii[idx];
    double R_k_intr = (double)intrinsicRadii[templateIdx_k];
    double dRk = dR_dPsi[idx];
    double d2Rk = d2R_dPsi2[idx];
    double pf = (double)prefactor;

    int exclStart = exclusionStart[templateIdx_k];
    int exclEnd = exclusionStart[templateIdx_k + 1];
    int groupSize = groupEndIdx - groupStartIdx;

    for (int c = 0; c < totalParticles; c++) {
        couplingMatrix[idx * totalParticles + c] = 0.0;
    }

    double d2E_self = pf * q_k * q_k / (R_k * R_k * R_k);
    double diag = d2E_self * dRk * dRk;

    if (includeSA) {
        double Rprobe = R_k_intr + (double)probeRadiusVal;
        double ratio = R_k_intr / R_k;
        double ratio2 = ratio * ratio;
        double ratio6 = ratio2 * ratio2 * ratio2;
        double E_SA = (double)surfaceTension * 4.0 * 3.14159265358979323846
                    * Rprobe * Rprobe * ratio6;
        double d2E_SA = 42.0 * E_SA / (R_k * R_k);
        diag += d2E_SA * dRk * dRk;
    }

    // OBC curvature: dE/dR * d2R/dPsi2.
    diag += (double)dE_dR[idx] * d2Rk;

    for (int lLocal = 0; lLocal < groupSize; lLocal++) {
        if (lLocal == atomInGroup) continue;
        int l = groupStartIdx + lLocal;
        int templateIdx_l = lLocal % templateNumAtoms;

        bool excluded = false;
        for (int e = exclStart; e < exclEnd; e++) {
            if (exclusionAtoms[e] == templateIdx_l) { excluded = true; break; }
        }
        if (excluded) continue;

        int particleIdx_l = particleIndices[l];
        real4 pos_l_f = posq[particleIdx_l];
        double pl_x = (double)pos_l_f.x;
        double pl_y = (double)pos_l_f.y;
        double pl_z = (double)pos_l_f.z;
        double q_l = (double)charges[templateIdx_l];
        double R_l = bornRadii[l];
        double dRl = dR_dPsi[l];

        double dx = pl_x - pk_x;
        double dy = pl_y - pk_y;
        double dz = pl_z - pk_z;
        double r2 = dx*dx + dy*dy + dz*dz;

        double RkRl = R_k * R_l;
        double expArg = -r2 / (4.0 * RkRl);
        double alpha_ = exp(expArg);
        double f2 = r2 + RkRl * alpha_;
        double f = sqrt(f2);
        double C = pf * q_k * q_l;

        double df2_dRk = alpha_ * (R_l + 0.25 * r2 / R_k);
        double df_dRk = df2_dRk / (2.0 * f);

        double dalpha_dRk = alpha_ * r2 / (4.0 * R_k * R_k * R_l);
        double d2f2_dRk2 = dalpha_dRk * (R_l + 0.25 * r2 / R_k)
                         + alpha_ * (-0.25 * r2 / (R_k * R_k));
        double d2f_dRk2 = d2f2_dRk2 / (2.0 * f)
                        - df2_dRk * df2_dRk / (4.0 * f * f * f);

        double d2E_dRk2 = C * (2.0 * df_dRk * df_dRk / (f * f * f)
                             - d2f_dRk2 / (f * f));
        diag += d2E_dRk2 * dRk * dRk;

        double df2_dRl = alpha_ * (R_k + 0.25 * r2 / R_l);
        double df_dRl = df2_dRl / (2.0 * f);
        double dalpha_dRl = alpha_ * r2 / (4.0 * R_k * R_l * R_l);
        double d2f2_dRkRl = dalpha_dRl * (R_l + 0.25 * r2 / R_k) + alpha_;
        double d2f_dRkRl = d2f2_dRkRl / (2.0 * f)
                         - df2_dRk * df2_dRl / (4.0 * f * f * f);
        double d2E_dRkRl = C * (2.0 * df_dRk * df_dRl / (f * f * f)
                              - d2f_dRkRl / (f * f));

        couplingMatrix[idx * totalParticles + l] = d2E_dRkRl * dRk * dRl;
    }

    couplingMatrix[idx * totalParticles + idx] = diag;
}


// Mirrors assembleGBSAHessian; reads double inter-kernel buffers,
// accumulates in double, writes double output. Final Hessian buffer
// is returned to the host as doubles already; computeHessian only
// needed to cast to double.
extern "C" __global__ void assembleGBSAHessianDouble(
    const real4* __restrict__ posq,
    const int* __restrict__ particleIndices,
    const real* __restrict__ charges,
    const double* __restrict__ bornRadii,
    const int* __restrict__ exclusionAtoms,
    const int* __restrict__ exclusionStart,
    const int* __restrict__ groupStart,
    int numGroups,
    int templateNumAtoms,
    float prefactor,
    const double* __restrict__ jacobian,
    const double* __restrict__ couplingMatrix,
    const double* __restrict__ dE_dHCT,
    const real* __restrict__ scaleFactors,
    const real* __restrict__ intrinsicRadii,
    const double* __restrict__ dR_dPsi,
    const double* __restrict__ gridHCTHessian,
    int totalParticles,
    HBUF_T* __restrict__ hessian
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int dim3N = 3 * totalParticles;
    if (tid >= dim3N * dim3N) return;

    int row = tid / dim3N;
    int col = tid % dim3N;
    if (col < row) return;

    int atom_i = row / 3, alpha_ = row % 3;
    int atom_j = col / 3, beta  = col % 3;

    int atomInGroup_i = atom_i, groupStart_i = 0, groupEnd_i = 0;
    for (int g = 0; g < numGroups; g++) {
        groupStart_i = groupStart[g];
        groupEnd_i = groupStart[g + 1];
        if (atom_i >= groupStart_i && atom_i < groupEnd_i) {
            atomInGroup_i = atom_i - groupStart_i; break;
        }
    }
    int templateIdx_i = atomInGroup_i % templateNumAtoms;
    int exclStart_i = exclusionStart[templateIdx_i];
    int exclEnd_i = exclusionStart[templateIdx_i + 1];

    double pf = (double)prefactor;
    double H_val = 0.0;

    if (atom_i == atom_j) {
        int particleIdx_i = particleIndices[atom_i];
        real4 pos_i_f = posq[particleIdx_i];
        double pix = (double)pos_i_f.x;
        double piy = (double)pos_i_f.y;
        double piz = (double)pos_i_f.z;
        double q_i = (double)charges[templateIdx_i];
        double R_i = bornRadii[atom_i];
        int groupSize = groupEnd_i - groupStart_i;

        double Ri_off = (double)intrinsicRadii[templateIdx_i] - (double)DIELECTRIC_OFFSET;
        double Si = Ri_off * (double)scaleFactors[templateIdx_i];

        for (int lLocal = 0; lLocal < groupSize; lLocal++) {
            if (lLocal == atomInGroup_i) continue;
            int l = groupStart_i + lLocal;
            int tl = lLocal % templateNumAtoms;
            bool excl = false;
            for (int e = exclStart_i; e < exclEnd_i; e++)
                if (exclusionAtoms[e] == tl) { excl = true; break; }
            if (excl) continue;

            real4 pos_l_f = posq[particleIndices[l]];
            double q_l = (double)charges[tl];
            double R_l = bornRadii[l];
            double dx = (double)pos_l_f.x - pix;
            double dy = (double)pos_l_f.y - piy;
            double dz = (double)pos_l_f.z - piz;
            double r2 = dx*dx + dy*dy + dz*dz;
            double r = sqrt(r2);
            if (r < 1e-10) continue;
            double RiRl = R_i * R_l;
            double et = exp(-r2 / (4.0 * RiRl));
            double f2 = r2 + RiRl * et;
            double f = sqrt(f2);
            double C = pf * q_i * q_l;

            double df2_dr = 2.0 * r * (1.0 - 0.25 * et);
            double df_dr = df2_dr / (2.0 * f);
            double dEdr = -C * df_dr / (f * f);
            double d2f2_dr2 = 2.0 - 0.5 * et + 0.25 * r2 * et / RiRl;
            double d2f_dr2 = d2f2_dr2 / (2.0 * f) - df2_dr * df2_dr / (4.0 * f * f * f);
            double d2Edr2 = C * (2.0 * df_dr * df_dr / (f * f * f) - d2f_dr2 / (f * f));

            double Rl_off = (double)intrinsicRadii[tl] - (double)DIELECTRIC_OFFSET;
            double Sl = Rl_off * (double)scaleFactors[tl];

            if (Ri_off < r + Sl) {
                double I1, I2;
                computeHCT_r12_double(r, Sl, Ri_off, I1, I2);
                dEdr   += dE_dHCT[atom_i] * I1;
                d2Edr2 += dE_dHCT[atom_i] * I2;
            }
            if (Rl_off < r + Si) {
                double I1, I2;
                computeHCT_r12_double(r, Si, Rl_off, I1, I2);
                dEdr   += dE_dHCT[l] * I1;
                d2Edr2 += dE_dHCT[l] * I2;
            }

            double dalpha_dRi = et * r2 / (4.0 * R_i * R_i * R_l);
            double d2f2_dr_dRi = 2.0 * r * (-0.25 * dalpha_dRi);
            double df2_dRi = et * (R_l + 0.25 * r2 / R_i);
            double df_dRi = df2_dRi / (2.0 * f);
            double d2f_dr_dRi = d2f2_dr_dRi / (2.0 * f)
                              - df2_dr * df2_dRi / (4.0 * f * f * f);
            double d2E_dr_dRi = C * (2.0 * df_dr * df_dRi / (f*f*f) - d2f_dr_dRi / (f*f));
            double g_i_val = d2E_dr_dRi * dR_dPsi[atom_i];

            double dalpha_dRl = et * r2 / (4.0 * R_i * R_l * R_l);
            double d2f2_dr_dRl = 2.0 * r * (-0.25 * dalpha_dRl);
            double df2_dRl = et * (R_i + 0.25 * r2 / R_l);
            double df_dRl = df2_dRl / (2.0 * f);
            double d2f_dr_dRl = d2f2_dr_dRl / (2.0 * f)
                              - df2_dr * df2_dRl / (4.0 * f * f * f);
            double d2E_dr_dRl = C * (2.0 * df_dr * df_dRl / (f*f*f) - d2f_dr_dRl / (f*f));
            double g_l_val = d2E_dr_dRl * dR_dPsi[l];

            double D[3] = {dx, dy, dz};
            double ir = 1.0 / r, ir2 = ir * ir;

            H_val += (d2Edr2 - dEdr * ir) * D[alpha_] * D[beta] * ir2
                   + ((alpha_ == beta) ? dEdr * ir : 0.0);

            double dr_dxi_a = -D[alpha_] * ir;
            double dr_dxi_b = -D[beta]  * ir;
            H_val += g_i_val * (dr_dxi_a * jacobian[atom_i * dim3N + 3*atom_i + beta]
                              + jacobian[atom_i * dim3N + 3*atom_i + alpha_] * dr_dxi_b);
            H_val += g_l_val * (dr_dxi_a * jacobian[l * dim3N + 3*atom_i + beta]
                              + jacobian[l * dim3N + 3*atom_i + alpha_] * dr_dxi_b);
        }

        {
            int hess_idx;
            if (alpha_ == beta) {
                hess_idx = alpha_;
            } else {
                int mn = alpha_ < beta ? alpha_ : beta;
                int mx = alpha_ > beta ? alpha_ : beta;
                hess_idx = (mn == 0 && mx == 1) ? 3 : (mn == 0 ? 4 : 5);
            }
            H_val += dE_dHCT[atom_i] * gridHCTHessian[atom_i * 6 + hess_idx];
        }

    } else if (atom_j >= groupStart_i && atom_j < groupEnd_i) {
        int atomInGroup_j = atom_j - groupStart_i;
        int templateIdx_j = atomInGroup_j % templateNumAtoms;
        int groupSize = groupEnd_i - groupStart_i;

        bool excl_ij = false;
        for (int e = exclStart_i; e < exclEnd_i; e++)
            if (exclusionAtoms[e] == templateIdx_j) { excl_ij = true; break; }

        if (!excl_ij) {
            real4 pos_i_f = posq[particleIndices[atom_i]];
            real4 pos_j_f = posq[particleIndices[atom_j]];
            double q_i = (double)charges[templateIdx_i];
            double q_j = (double)charges[templateIdx_j];
            double R_i = bornRadii[atom_i];
            double R_j = bornRadii[atom_j];
            double dx = (double)pos_j_f.x - (double)pos_i_f.x;
            double dy = (double)pos_j_f.y - (double)pos_i_f.y;
            double dz = (double)pos_j_f.z - (double)pos_i_f.z;
            double r2 = dx*dx + dy*dy + dz*dz;
            double r = sqrt(r2);
            if (r >= 1e-10) {
                double RiRj = R_i * R_j;
                double et = exp(-r2 / (4.0 * RiRj));
                double f2 = r2 + RiRj * et;
                double f = sqrt(f2);
                double C = pf * q_i * q_j;
                double df2_dr = 2.0 * r * (1.0 - 0.25 * et);
                double df_dr = df2_dr / (2.0 * f);
                double dEdr = -C * df_dr / (f * f);
                double d2f2_dr2 = 2.0 - 0.5 * et + 0.25 * r2 * et / RiRj;
                double d2f_dr2 = d2f2_dr2 / (2.0 * f) - df2_dr * df2_dr / (4.0 * f * f * f);
                double d2Edr2 = C * (2.0 * df_dr * df_dr / (f * f * f) - d2f_dr2 / (f * f));

                double Ri_off = (double)intrinsicRadii[templateIdx_i] - (double)DIELECTRIC_OFFSET;
                double Rj_off = (double)intrinsicRadii[templateIdx_j] - (double)DIELECTRIC_OFFSET;
                double Si = Ri_off * (double)scaleFactors[templateIdx_i];
                double Sj = Rj_off * (double)scaleFactors[templateIdx_j];
                if (Ri_off < r + Sj) {
                    double I1, I2;
                    computeHCT_r12_double(r, Sj, Ri_off, I1, I2);
                    dEdr   += dE_dHCT[atom_i] * I1;
                    d2Edr2 += dE_dHCT[atom_i] * I2;
                }
                if (Rj_off < r + Si) {
                    double I1, I2;
                    computeHCT_r12_double(r, Si, Rj_off, I1, I2);
                    dEdr   += dE_dHCT[atom_j] * I1;
                    d2Edr2 += dE_dHCT[atom_j] * I2;
                }

                double ir = 1.0 / r, ir2 = ir * ir;
                double D_ij[3] = {dx, dy, dz};
                H_val += -(d2Edr2 - dEdr * ir) * D_ij[alpha_] * D_ij[beta] * ir2
                       - ((alpha_ == beta) ? dEdr * ir : 0.0);
            }
        }

        real4 pos_ii_f = posq[particleIndices[atom_i]];
        real4 pos_jj_f = posq[particleIndices[atom_j]];
        double pii_x = (double)pos_ii_f.x;
        double pii_y = (double)pos_ii_f.y;
        double pii_z = (double)pos_ii_f.z;
        double pjj_x = (double)pos_jj_f.x;
        double pjj_y = (double)pos_jj_f.y;
        double pjj_z = (double)pos_jj_f.z;

        // Part A: pairs involving atom_i.
        for (int lLocal = 0; lLocal < groupSize; lLocal++) {
            if (lLocal == atomInGroup_i) continue;
            int l = groupStart_i + lLocal;
            int tl = lLocal % templateNumAtoms;
            bool excl_l = false;
            for (int e = exclStart_i; e < exclEnd_i; e++)
                if (exclusionAtoms[e] == tl) { excl_l = true; break; }
            if (excl_l) continue;

            real4 pos_l_f = posq[particleIndices[l]];
            double R_ii = bornRadii[atom_i], R_l = bornRadii[l];
            double dx_ = (double)pos_l_f.x - pii_x;
            double dy_ = (double)pos_l_f.y - pii_y;
            double dz_ = (double)pos_l_f.z - pii_z;
            double r2_ = dx_*dx_ + dy_*dy_ + dz_*dz_;
            double r_ = sqrt(r2_);
            if (r_ < 1e-10) continue;

            double RiRl = R_ii * R_l;
            double et_ = exp(-r2_ / (4.0 * RiRl));
            double f_ = sqrt(r2_ + RiRl * et_);
            double C_ = pf * (double)charges[templateIdx_i] * (double)charges[tl];
            double df2_ = 2.0 * r_ * (1.0 - 0.25 * et_);
            double df_ = df2_ / (2.0 * f_);

            double da_i = et_ * r2_ / (4.0 * R_ii * R_ii * R_l);
            double d2f2ri = 2.0 * r_ * (-0.25 * da_i);
            double df2ri = et_ * (R_l + 0.25 * r2_ / R_ii);
            double dfri = df2ri / (2.0 * f_);
            double d2fri = d2f2ri / (2.0 * f_) - df2_ * df2ri / (4.0 * f_ * f_ * f_);
            double g_i_ = C_ * (2.0 * df_ * dfri / (f_*f_*f_) - d2fri / (f_*f_)) * dR_dPsi[atom_i];

            double da_l = et_ * r2_ / (4.0 * R_ii * R_l * R_l);
            double d2f2rl = 2.0 * r_ * (-0.25 * da_l);
            double df2rl = et_ * (R_ii + 0.25 * r2_ / R_l);
            double dfrl = df2rl / (2.0 * f_);
            double d2frl = d2f2rl / (2.0 * f_) - df2_ * df2rl / (4.0 * f_ * f_ * f_);
            double g_l_ = C_ * (2.0 * df_ * dfrl / (f_*f_*f_) - d2frl / (f_*f_)) * dR_dPsi[l];

            double D_[3] = {dx_, dy_, dz_};
            double dr_a = -D_[alpha_] / r_;
            double v_col = g_i_ * jacobian[atom_i * dim3N + col]
                         + g_l_ * jacobian[l * dim3N + col];
            H_val += dr_a * v_col;
        }

        // Part B: pairs involving atom_j.
        int exclStart_j = exclusionStart[templateIdx_j];
        int exclEnd_j = exclusionStart[templateIdx_j + 1];
        for (int mLocal = 0; mLocal < groupSize; mLocal++) {
            if (mLocal == atomInGroup_j) continue;
            int m = groupStart_i + mLocal;
            int tm = mLocal % templateNumAtoms;
            bool excl_m = false;
            for (int e = exclStart_j; e < exclEnd_j; e++)
                if (exclusionAtoms[e] == tm) { excl_m = true; break; }
            if (excl_m) continue;

            real4 pos_m_f = posq[particleIndices[m]];
            double R_jj = bornRadii[atom_j], R_m = bornRadii[m];
            double dx_ = (double)pos_m_f.x - pjj_x;
            double dy_ = (double)pos_m_f.y - pjj_y;
            double dz_ = (double)pos_m_f.z - pjj_z;
            double r2_ = dx_*dx_ + dy_*dy_ + dz_*dz_;
            double r_ = sqrt(r2_);
            if (r_ < 1e-10) continue;

            double RjRm = R_jj * R_m;
            double et_ = exp(-r2_ / (4.0 * RjRm));
            double f_ = sqrt(r2_ + RjRm * et_);
            double C_ = pf * (double)charges[templateIdx_j] * (double)charges[tm];
            double df2_ = 2.0 * r_ * (1.0 - 0.25 * et_);
            double df_ = df2_ / (2.0 * f_);

            double da_j = et_ * r2_ / (4.0 * R_jj * R_jj * R_m);
            double d2f2rj = 2.0 * r_ * (-0.25 * da_j);
            double df2rj = et_ * (R_m + 0.25 * r2_ / R_jj);
            double dfrj = df2rj / (2.0 * f_);
            double d2frj = d2f2rj / (2.0 * f_) - df2_ * df2rj / (4.0 * f_ * f_ * f_);
            double g_j_ = C_ * (2.0 * df_ * dfrj / (f_*f_*f_) - d2frj / (f_*f_)) * dR_dPsi[atom_j];

            double da_m = et_ * r2_ / (4.0 * R_jj * R_m * R_m);
            double d2f2rm = 2.0 * r_ * (-0.25 * da_m);
            double df2rm = et_ * (R_jj + 0.25 * r2_ / R_m);
            double dfrm = df2rm / (2.0 * f_);
            double d2frm = d2f2rm / (2.0 * f_) - df2_ * df2rm / (4.0 * f_ * f_ * f_);
            double g_m_ = C_ * (2.0 * df_ * dfrm / (f_*f_*f_) - d2frm / (f_*f_)) * dR_dPsi[m];

            double D_[3] = {dx_, dy_, dz_};
            double dr_b = -D_[beta] / r_;
            double v_row = g_j_ * jacobian[atom_j * dim3N + row]
                         + g_m_ * jacobian[m * dim3N + row];
            H_val += v_row * dr_b;
        }
    }

    // J^T M J
    for (int k = 0; k < totalParticles; k++) {
        double Jk = jacobian[k * dim3N + row];
        if (fabs(Jk) < 1e-18) continue;
        for (int l = 0; l < totalParticles; l++) {
            double Mkl = couplingMatrix[k * totalParticles + l];
            if (fabs(Mkl) < 1e-18) continue;
            H_val += Jk * Mkl * jacobian[l * dim3N + col];
        }
    }

    hessian[row * dim3N + col] = H_val;
    if (col > row) hessian[col * dim3N + row] = H_val;
}


// ==================================================================
// PAIRWISE receptor-desolvation + cross-term Hessian contributions.
//
// Receptor atoms are frozen; only ligand coordinates are Hessian
// variables. The receptor Born radii R^R_j depend on ligand positions
// through PsiR_j(x) = sum_i HCT(r_{ji}; rec_off_j, lig_off_i, lig_s_i).
//
// These kernels mirror addPairwiseHessianContributions(...) in
// platforms/reference/src/ReferenceIsolatedGBSAKernels.cpp, which was
// validated against JAX autodiff to ~1e-15. Each kernel/term below is
// annotated with the Reference line range it ports.
//
// Per-group layout (one block of receptor work per particle group g):
//   recBorn / recDRdPsi / recD2RdPsi2 / recDeDR / dCrossDRR : [K*Nr]
//   JR : [K * Nr * n3]   (n3 = 3*groupSize, group-local rows)
//   MR : [K * Nr * Nr]
//   dCrossDRL : [totalParticles]  (one entry per ligand atom)
// Hessian rows/cols are global (3*globalAtom + a); per-group blocks are
// block-diagonal, so JR/MR rows index group-local atoms and are mapped
// to global atom (groupStart[g] + localAtom) at assembly.
//
// HCT-integral first/second r-derivatives reuse computeHCT_r12_double
// above (= computeHCTTermDerivative / computeHCTTermSecondDerivative).
// ==================================================================

// Receptor OBC/HCT Born transform: R, dR/dPsi, d2R/dPsi2 from the total
// screening sum. Mirrors bornTransformDerivs(...) in the Reference.
__device__ __forceinline__ void recBornTransformDouble(
    double R_intrinsic, double R_off, double tot, bool isHCT,
    double& R_born, double& dRdPsi, double& d2RdPsi2)
{
    if (R_off <= 0.0) {
        // Match the Reference: still produce a Born radius but zero derivs.
        if (isHCT) {
            double inner = 1.0/R_off - 0.5*R_off*tot;
            R_born = (inner > 0.0) ? 1.0/inner : 500.0;
        } else {
            double psi = 0.5*R_off*tot;
            double tv = tanh((double)OBC_ALPHA*psi - (double)OBC_BETA*psi*psi
                             + (double)OBC_GAMMA*psi*psi*psi);
            double inner = 1.0/R_off - tv/R_intrinsic;
            R_born = (inner > 0.0) ? 1.0/inner : 500.0;
        }
        dRdPsi = 0.0; d2RdPsi2 = 0.0; return;
    }
    if (isHCT) {
        double inner = 1.0/R_off - 0.5*R_off*tot;
        R_born = (inner > 0.0) ? 1.0/inner : 500.0;
        double a = 0.5 * R_off;
        dRdPsi = a * R_born * R_born;
        d2RdPsi2 = 2.0 * R_born * dRdPsi * a;
    } else {
        double psi_s = 0.5 * R_off * tot;
        double psi_s2 = psi_s * psi_s;
        double arg = (double)OBC_ALPHA*psi_s - (double)OBC_BETA*psi_s2
                   + (double)OBC_GAMMA*psi_s2*psi_s;
        double t = tanh(arg);
        double sech2 = 1.0 - t*t;
        double inner = 1.0/R_off - t/R_intrinsic;
        R_born = (inner > 0.0) ? 1.0/inner : 500.0;
        double darg = (double)OBC_ALPHA - 2.0*(double)OBC_BETA*psi_s
                    + 3.0*(double)OBC_GAMMA*psi_s2;
        double d2arg = -2.0*(double)OBC_BETA + 6.0*(double)OBC_GAMMA*psi_s;
        double dpsi = 0.5 * R_off;
        double Dc = dpsi / R_intrinsic;
        dRdPsi = R_born * R_born * sech2 * darg * Dc;
        double A = R_born*R_born, B = sech2, C = darg;
        double dA = 2.0 * R_born * dRdPsi;
        double dargP = darg * dpsi;
        double dB = -2.0 * sech2 * t * dargP;
        double dC = d2arg * dpsi;
        d2RdPsi2 = (dA*B*C + A*dB*C + A*B*dC) * Dc;
    }
}


// HCT integral value (matches computeHCTTerm / the double HCT kernels).
__device__ __forceinline__ double computeHCTValDouble(
    double r, double R_i_off, double S)
{
    double r_plus_S = r + S;
    if (R_i_off >= r_plus_S) return 0.0;
    double r_minus_S = fabs(r - S);
    double l = (R_i_off > r_minus_S) ? (1.0/R_i_off) : (1.0/r_minus_S);
    double u = 1.0/r_plus_S;
    double l2 = l*l, u2 = u*u;
    double term = (l - u) + 0.25*r*(u2 - l2)
                + 0.5*log(u/l)/r + 0.25*S*S*(l2 - u2)/r;
    if (R_i_off < (S - r)) term += 2.0*(1.0/R_i_off - l);
    return term;
}


// Kernel R1: per receptor atom j (per group) compute ligToRec (ligand
// screening of receptor j), the receptor Born radius and its transform
// derivatives. Ports Reference lines 1607-1640.
extern "C" __global__ void pairwiseRecBornDouble(
    const real4* __restrict__ posq,
    const int* __restrict__ particleIndices,
    const real* __restrict__ ligandRadii,
    const real* __restrict__ ligandScaleFactors,
    const int* __restrict__ groupStart,
    int numGroups,
    int templateNumAtoms,
    const real4* __restrict__ receptorPositions,
    const real* __restrict__ receptorRadii,
    const real* __restrict__ receptorSelfHCT,
    int numReceptorAtoms,
    float cutoffDistance,
    int isHCT,
    double* __restrict__ recBorn,
    double* __restrict__ recDRdPsi,
    double* __restrict__ recD2RdPsi2)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int Nr = numReceptorAtoms;
    if (tid >= numGroups * Nr) return;
    int g = tid / Nr;
    int j = tid % Nr;

    int gs = groupStart[g];
    int ge = groupStart[g + 1];
    int groupSize = ge - gs;

    real4 pj = receptorPositions[j];
    double Rj_off = (double)receptorRadii[j] - (double)DIELECTRIC_OFFSET;

    double tot = (double)receptorSelfHCT[j];
    for (int iL = 0; iL < groupSize; iL++) {
        int gAtom = gs + iL;
        int tIdx = iL % templateNumAtoms;
        real4 pi = posq[particleIndices[gAtom]];
        double dx = (double)pj.x - (double)pi.x;
        double dy = (double)pj.y - (double)pi.y;
        double dz = (double)pj.z - (double)pi.z;
        double r = sqrt(dx*dx + dy*dy + dz*dz);
        if (r < 1e-10) continue;
        if (cutoffDistance > 0.0f && r > (double)cutoffDistance) continue;
        double Ri_off = (double)ligandRadii[tIdx] - (double)DIELECTRIC_OFFSET;
        double S = Ri_off * (double)ligandScaleFactors[tIdx];
        // ligToRec uses HCT(r; Rj_off probe, lig offset, lig scale)
        tot += computeHCTValDouble(r, Rj_off, S);
    }

    double R, dR, d2R;
    recBornTransformDouble((double)receptorRadii[j], Rj_off, tot, isHCT != 0,
                           R, dR, d2R);
    recBorn[tid] = R;
    recDRdPsi[tid] = dR;
    recD2RdPsi2[tid] = d2R;
}


// Kernel R2: per receptor atom j compute recDeDR[j] = dE_rec/dR^R_j and
// the receptor coupling matrix row MR[j][*]. The diagonal MR[j][j] folds
// in the receptor self curvature, the Still-pair d2E/dRj2 contributions,
// and the receptor OBC curvature dE_rec/dR^R_j * d2R^R_j/dPsi2.
// Ports Reference lines 1645-1684.
extern "C" __global__ void pairwiseRecCouplingDouble(
    const real4* __restrict__ receptorPositions,
    const real* __restrict__ receptorCharges,
    const double* __restrict__ recBorn,
    const double* __restrict__ recDRdPsi,
    const double* __restrict__ recD2RdPsi2,
    int numGroups,
    int numReceptorAtoms,
    float prefactor,
    double* __restrict__ recDeDR,
    double* __restrict__ MR)        // [K * Nr * Nr]
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int Nr = numReceptorAtoms;
    if (tid >= numGroups * Nr) return;
    int g = tid / Nr;
    int j = tid % Nr;
    size_t mrBase = (size_t)g * Nr * Nr;
    double pf = (double)prefactor;

    double Rj = recBorn[tid];
    double dRj = recDRdPsi[tid];
    double qj = (double)receptorCharges[j];
    real4 pj = receptorPositions[j];

    double deDR = -0.5 * pf * qj*qj / (Rj*Rj);
    double diag = pf * qj*qj / (Rj*Rj*Rj) * dRj * dRj;   // self curvature

    for (int m = 0; m < Nr; m++) {
        MR[mrBase + (size_t)j*Nr + m] = 0.0;
    }

    for (int m = 0; m < Nr; m++) {
        if (m == j) continue;
        real4 pm = receptorPositions[m];
        double dx = (double)pj.x - (double)pm.x;
        double dy = (double)pj.y - (double)pm.y;
        double dz = (double)pj.z - (double)pm.z;
        double r2 = dx*dx + dy*dy + dz*dz;
        double Rm = recBorn[(size_t)g*Nr + m];
        double dRm = recDRdPsi[(size_t)g*Nr + m];
        double RaRb = Rj*Rm;
        double et = exp(-r2/(4.0*RaRb));
        double f2 = r2 + RaRb*et;
        double f = sqrt(f2);
        double C = pf * qj * (double)receptorCharges[m];
        double df2_dRj = et*(Rm + 0.25*r2/Rj);
        double df_dRj = df2_dRj/(2.0*f);
        deDR += -pf * qj * (double)receptorCharges[m] / f2 * df_dRj;
        // d2E/dRj2
        double dalpha_dRj = et*r2/(4.0*Rj*Rj*Rm);
        double d2f2_dRj2 = dalpha_dRj*(Rm+0.25*r2/Rj) + et*(-0.25*r2/(Rj*Rj));
        double d2f_dRj2 = d2f2_dRj2/(2.0*f) - df2_dRj*df2_dRj/(4.0*f*f*f);
        double d2E_dRj2 = C*(2.0*df_dRj*df_dRj/(f*f*f) - d2f_dRj2/(f*f));
        diag += d2E_dRj2 * dRj * dRj;
        // d2E/dRj dRm
        double df2_dRm = et*(Rj+0.25*r2/Rm);
        double df_dRm = df2_dRm/(2.0*f);
        double dalpha_dRm = et*r2/(4.0*Rj*Rm*Rm);
        double d2f2_dRjRm = dalpha_dRm*(Rm+0.25*r2/Rj) + et;
        double d2f_dRjRm = d2f2_dRjRm/(2.0*f) - df2_dRj*df2_dRm/(4.0*f*f*f);
        double d2E_dRjRm = C*(2.0*df_dRj*df_dRm/(f*f*f) - d2f_dRjRm/(f*f));
        MR[mrBase + (size_t)j*Nr + m] = d2E_dRjRm * dRj * dRm;
    }

    // receptor OBC curvature folded into MR diagonal.
    diag += deDR * recD2RdPsi2[tid];
    MR[mrBase + (size_t)j*Nr + j] = diag;
    recDeDR[tid] = deDR;
}


// Kernel R3: receptor Jacobian JR[j][3*i+a] = dPsiR_j/dx_{i,a}.
// PsiR_j = sum_i HCT(r_{ji}); receptor frozen, so only ligand atom i's
// own block is nonzero. dPsiR_j/dx_{i,a} = I1(r_{ji}) * (lig-rec)_a / r.
// Ports Reference lines 1690-1710.
extern "C" __global__ void pairwiseRecJacobianDouble(
    const real4* __restrict__ posq,
    const int* __restrict__ particleIndices,
    const real* __restrict__ ligandRadii,
    const real* __restrict__ ligandScaleFactors,
    const int* __restrict__ groupStart,
    int numGroups,
    int templateNumAtoms,
    const real4* __restrict__ receptorPositions,
    const real* __restrict__ receptorRadii,
    int numReceptorAtoms,
    float cutoffDistance,
    int n3,                          // 3 * groupSize
    double* __restrict__ JR)         // [K * Nr * n3]
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int Nr = numReceptorAtoms;
    if (tid >= numGroups * Nr) return;
    int g = tid / Nr;
    int j = tid % Nr;

    int gs = groupStart[g];
    int ge = groupStart[g + 1];
    int groupSize = ge - gs;
    size_t jrBase = (size_t)g * Nr * n3 + (size_t)j * n3;
    for (int c = 0; c < n3; c++) JR[jrBase + c] = 0.0;

    real4 pj = receptorPositions[j];
    double Rj_off = (double)receptorRadii[j] - (double)DIELECTRIC_OFFSET;

    for (int iL = 0; iL < groupSize; iL++) {
        int gAtom = gs + iL;
        int tIdx = iL % templateNumAtoms;
        real4 pi = posq[particleIndices[gAtom]];
        double dx = (double)pi.x - (double)pj.x;   // lig - rec
        double dy = (double)pi.y - (double)pj.y;
        double dz = (double)pi.z - (double)pj.z;
        double r2 = dx*dx + dy*dy + dz*dz, r = sqrt(r2);
        if (r < 1e-10) continue;
        if (cutoffDistance > 0.0f && r > (double)cutoffDistance) continue;
        double Ri_off = (double)ligandRadii[tIdx] - (double)DIELECTRIC_OFFSET;
        double Si = Ri_off * (double)ligandScaleFactors[tIdx];
        if (Rj_off >= r + Si) continue;
        double I1, I2;
        computeHCT_r12_double(r, Si, Rj_off, I1, I2);
        double invr = 1.0/r;
        JR[jrBase + 3*iL + 0] = I1*dx*invr;
        JR[jrBase + 3*iL + 1] = I1*dy*invr;
        JR[jrBase + 3*iL + 2] = I1*dz*invr;
    }
}


// Kernel R4: cross-term first Born derivatives dCross_dRL[i] (per ligand
// atom) and dCross_dRR[j] (per receptor atom, per group), plus the
// explicit-r cross spatial Hessian on each ligand atom's diagonal block.
// Ports Reference lines 1769-1832 + 1807-1813. dCrossDRR is accumulated
// per receptor atom (each thread owns one ligand atom i and atomicAdds
// into the receptor-indexed dCrossDRR[g*Nr+j]); dCrossDRL[i] is owned by
// the thread so no atomics needed for it.
extern "C" __global__ void pairwiseCrossBornDeriv1Double(
    const real4* __restrict__ posq,
    const int* __restrict__ particleIndices,
    const real* __restrict__ charges,
    const double* __restrict__ bornRadii,        // ligand Born radii (double)
    const int* __restrict__ groupStart,
    int numGroups,
    int templateNumAtoms,
    const real4* __restrict__ receptorPositions,
    const real* __restrict__ receptorCharges,
    const double* __restrict__ recBorn,
    int numReceptorAtoms,
    float prefactor,
    int totalParticles,
    double* __restrict__ dCrossDRL,              // [totalParticles]
    double* __restrict__ dCrossDRR,              // [K * Nr]  (atomicAdd)
    HBUF_T* __restrict__ hessian)                // [dim3N * dim3N] (atomicAdd)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= totalParticles) return;
    int Nr = numReceptorAtoms;
    int dim3N = 3 * totalParticles;
    const double MIN_CROSS_R2 = 0.01;

    int g = 0, gs = 0, ge = 0;
    for (int gg = 0; gg < numGroups; gg++) {
        gs = groupStart[gg]; ge = groupStart[gg + 1];
        if (idx >= gs && idx < ge) { g = gg; break; }
    }
    if (idx >= ge) { dCrossDRL[idx] = 0.0; return; }

    int iL = idx - gs;
    int tIdx = iL % templateNumAtoms;
    double pf = (double)prefactor;
    real4 pi = posq[particleIndices[idx]];
    double q_i = (double)charges[tIdx];
    double Ri = bornRadii[idx];

    double dCRL = 0.0;
    double Hd[3][3] = {{0,0,0},{0,0,0},{0,0,0}};

    for (int j = 0; j < Nr; j++) {
        real4 pj = receptorPositions[j];
        double dx = (double)pi.x - (double)pj.x;   // lig - rec
        double dy = (double)pi.y - (double)pj.y;
        double dz = (double)pi.z - (double)pj.z;
        double r2 = dx*dx + dy*dy + dz*dz;
        if (r2 < MIN_CROSS_R2) continue;
        double r = sqrt(r2);
        double Rj = recBorn[(size_t)g*Nr + j];
        double RaRb = Ri*Rj;
        double et = exp(-r2/(4.0*RaRb));
        double f2 = r2 + RaRb*et;
        double f = sqrt(f2);
        double C = pf * q_i * (double)receptorCharges[j];

        // explicit r derivatives
        double df2_dr = 2.0*r*(1.0 - 0.25*et);
        double df_dr = df2_dr/(2.0*f);
        double dEdr = -C*df_dr/(f*f);
        double d2f2_dr2 = 2.0 - 0.5*et + 0.25*r2*et/(Ri*Rj);
        double d2f_dr2 = d2f2_dr2/(2.0*f) - df2_dr*df2_dr/(4.0*f*f*f);
        double d2Edr2 = C*(2.0*df_dr*df_dr/(f*f*f) - d2f_dr2/(f*f));

        // Born first derivatives
        double df2_dRi = et*(Rj + 0.25*r2/Ri);
        double df_dRi = df2_dRi/(2.0*f);
        double dE_dRi = -C*df_dRi/(f*f);
        double df2_dRj = et*(Ri + 0.25*r2/Rj);
        double df_dRj = df2_dRj/(2.0*f);
        double dE_dRj = -C*df_dRj/(f*f);
        dCRL += dE_dRi;
        atomicAdd(&dCrossDRR[(size_t)g*Nr + j], dE_dRj);

        // explicit spatial Hessian on ligand atom i (diagonal block)
        double ir = 1.0/r, ir2 = ir*ir;
        double D[3] = {dx, dy, dz};
        for (int a=0;a<3;a++) for (int b=0;b<3;b++) {
            Hd[a][b] += (d2Edr2 - dEdr*ir)*D[a]*D[b]*ir2
                      + ((a==b)?dEdr*ir:0.0);
        }
    }

    dCrossDRL[idx] = dCRL;
    // scatter the ligand-atom-i diagonal explicit-r block into hessian.
    for (int a=0;a<3;a++) for (int b=0;b<3;b++) {
        int row = 3*idx + a, col = 3*idx + b;
        atomicAdd(&hessian[(size_t)row*dim3N + col], (HBUF_T)Hd[a][b]);
    }
}


// Kernel R5: desolvation self spatial Hessian + cross-term Born-gradient
// spatial Hessians, all routed through per-ligand-atom HCT spatial
// Hessians. One thread per ligand atom i.
//   (a) receptor desolvation self: w = recDeDR[j]*recDRdPsi[j] over the
//       receptor->ligand HCT spatial Hessian (i diagonal block).
//       Ports Reference lines 1734-1760.
//   (b) cross ligand Born gradient: w = dCross_dRL[i]*dRdPsi[i] over the
//       ligand-ligand HCT (ii/il/li/ll blocks) and the receptor->ligand
//       HCT (i diagonal). Ports Reference lines 1843-1902.
//   (c) cross receptor Born gradient: w = dCross_dRR[j]*recDRdPsi[j] over
//       the receptor->ligand HCT spatial Hessian (i diagonal block).
//       Ports Reference lines 1903-1929.
extern "C" __global__ void pairwiseBornGradHessianDouble(
    const real4* __restrict__ posq,
    const int* __restrict__ particleIndices,
    const real* __restrict__ ligandRadii,
    const real* __restrict__ ligandScaleFactors,
    const double* __restrict__ dRdPsi,           // ligand dR/dPsi [totalParticles]
    const int* __restrict__ groupStart,
    int numGroups,
    int templateNumAtoms,
    const real4* __restrict__ receptorPositions,
    const real* __restrict__ receptorRadii,
    const real* __restrict__ receptorScaleFactors,
    const double* __restrict__ recDeDR,          // [K*Nr]
    const double* __restrict__ recDRdPsi,        // [K*Nr]
    const double* __restrict__ dCrossDRL,        // [totalParticles]
    const double* __restrict__ dCrossDRR,        // [K*Nr]
    int numReceptorAtoms,
    int totalParticles,
    HBUF_T* __restrict__ hessian)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= totalParticles) return;
    int Nr = numReceptorAtoms;
    int dim3N = 3 * totalParticles;

    int g = 0, gs = 0, ge = 0;
    for (int gg = 0; gg < numGroups; gg++) {
        gs = groupStart[gg]; ge = groupStart[gg + 1];
        if (idx >= gs && idx < ge) { g = gg; break; }
    }
    if (idx >= ge) return;
    int groupSize = ge - gs;

    int iL = idx - gs;
    int tIdx = iL % templateNumAtoms;
    real4 pi = posq[particleIndices[idx]];
    double Ri_off = (double)ligandRadii[tIdx] - (double)DIELECTRIC_OFFSET;
    double Si = Ri_off * (double)ligandScaleFactors[tIdx];

    double wL = dCrossDRL[idx] * dRdPsi[idx];   // cross ligand Born-gradient weight

    // ---- (b1) cross ligand Born gradient: ligand-ligand HCT pairs ----
    // d2(Psi^L_i)/dxdx via HCT(r_il; Ri_off, Rl_off, s_l): blocks ii,il,li,ll.
    for (int lL = 0; lL < groupSize; lL++) {
        if (lL == iL) continue;
        int lAtom = gs + lL;
        int tl = lL % templateNumAtoms;
        real4 pl = posq[particleIndices[lAtom]];
        double dx = (double)pi.x - (double)pl.x;
        double dy = (double)pi.y - (double)pl.y;
        double dz = (double)pi.z - (double)pl.z;
        double r2 = dx*dx + dy*dy + dz*dz, r = sqrt(r2);
        if (r < 1e-10) continue;
        double Rl_off = (double)ligandRadii[tl] - (double)DIELECTRIC_OFFSET;
        double Sl = Rl_off * (double)ligandScaleFactors[tl];
        if (Ri_off >= r + Sl) continue;
        double I1, I2;
        computeHCT_r12_double(r, Sl, Ri_off, I1, I2);
        double invr=1.0/r, rhx=dx*invr, rhy=dy*invr, rhz=dz*invr;
        double A = I1*invr, Bc = I2 - A;
        double blk[3][3] = {
            {A+Bc*rhx*rhx, Bc*rhx*rhy, Bc*rhx*rhz},
            {Bc*rhx*rhy, A+Bc*rhy*rhy, Bc*rhy*rhz},
            {Bc*rhx*rhz, Bc*rhy*rhz, A+Bc*rhz*rhz}};
        for (int a=0;a<3;a++) for (int b=0;b<3;b++) {
            double hv = wL*blk[a][b];
            atomicAdd(&hessian[(size_t)(3*idx+a)*dim3N + 3*idx+b],   (HBUF_T) hv);
            atomicAdd(&hessian[(size_t)(3*idx+a)*dim3N + 3*lAtom+b], (HBUF_T)-hv);
            atomicAdd(&hessian[(size_t)(3*lAtom+a)*dim3N + 3*idx+b], (HBUF_T)-hv);
            atomicAdd(&hessian[(size_t)(3*lAtom+a)*dim3N + 3*lAtom+b],(HBUF_T)hv);
        }
    }

    // ---- receptor->ligand HCT spatial Hessian (i diagonal block) ----
    // Three weights share the same geometry; accumulate combined diagonal.
    double Hd[3][3] = {{0,0,0},{0,0,0},{0,0,0}};
    for (int j = 0; j < Nr; j++) {
        real4 pj = receptorPositions[j];
        double dx = (double)pi.x - (double)pj.x;
        double dy = (double)pi.y - (double)pj.y;
        double dz = (double)pi.z - (double)pj.z;
        double r2 = dx*dx + dy*dy + dz*dz, r = sqrt(r2);
        if (r < 1e-10) continue;
        double Rj_off = (double)receptorRadii[j] - (double)DIELECTRIC_OFFSET;

        // (a) + (c): self-spatial PsiR_j Hessian uses HCT(r; Rj_off probe,
        //     Ri_off, lig scale Si) -> screening of receptor j by ligand i.
        if (Rj_off < r + Si) {
            double I1, I2;
            computeHCT_r12_double(r, Si, Rj_off, I1, I2);
            double invr=1.0/r, rhx=dx*invr, rhy=dy*invr, rhz=dz*invr;
            double A = I1*invr, Bc = I2 - A;
            // (a) desolvation self weight + (c) cross receptor-Born weight.
            double wRec = recDeDR[(size_t)g*Nr + j] * recDRdPsi[(size_t)g*Nr + j]
                        + dCrossDRR[(size_t)g*Nr + j] * recDRdPsi[(size_t)g*Nr + j];
            Hd[0][0] += wRec*(A+Bc*rhx*rhx); Hd[1][1] += wRec*(A+Bc*rhy*rhy); Hd[2][2] += wRec*(A+Bc*rhz*rhz);
            Hd[0][1] += wRec*Bc*rhx*rhy; Hd[0][2] += wRec*Bc*rhx*rhz; Hd[1][2] += wRec*Bc*rhy*rhz;
            Hd[1][0] += wRec*Bc*rhx*rhy; Hd[2][0] += wRec*Bc*rhx*rhz; Hd[2][1] += wRec*Bc*rhy*rhz;
        }

        // (b2) cross ligand Born gradient receptor->ligand HCT term:
        //      weight wL, uses HCT(r; Ri_off probe, Rrj_off, rec scale).
        double Rrj_off = (double)receptorRadii[j] - (double)DIELECTRIC_OFFSET;
        double Srj = Rrj_off * (double)receptorScaleFactors[j];
        if (Ri_off < r + Srj) {
            double I1, I2;
            computeHCT_r12_double(r, Srj, Ri_off, I1, I2);
            double invr=1.0/r, rhx=dx*invr, rhy=dy*invr, rhz=dz*invr;
            double A = I1*invr, Bc = I2 - A;
            Hd[0][0] += wL*(A+Bc*rhx*rhx); Hd[1][1] += wL*(A+Bc*rhy*rhy); Hd[2][2] += wL*(A+Bc*rhz*rhz);
            Hd[0][1] += wL*Bc*rhx*rhy; Hd[0][2] += wL*Bc*rhx*rhz; Hd[1][2] += wL*Bc*rhy*rhz;
            Hd[1][0] += wL*Bc*rhx*rhy; Hd[2][0] += wL*Bc*rhx*rhz; Hd[2][1] += wL*Bc*rhy*rhz;
        }
    }
    for (int a=0;a<3;a++) for (int b=0;b<3;b++)
        atomicAdd(&hessian[(size_t)(3*idx+a)*dim3N + 3*idx+b], (HBUF_T)Hd[a][b]);
}


// Kernel R6: assemble the outer-product Hessian contributions over global
// (row, col) entries:
//   - desolvation JR^T MR JR  (Reference lines 1717-1733)
//   - cross-term Born-Born + mixed r-Born couplings via J and JR outer
//     products plus the dr/dx coupling  (Reference lines 1936-2013)
//   - cross-term single-Born curvature dCross_dRL[i]*d2R^L/dPsi2 (via J)
//     and dCross_dRR[j]*d2R^R/dPsi2 (via JR)  (Reference lines 2020-2047)
// One thread per upper-triangular (row, col) entry of the global Hessian.
// row/col are mapped to a group-local atom range; cross-group entries are
// zero (block diagonal) and skipped.
extern "C" __global__ void pairwiseOuterProductHessianDouble(
    const real4* __restrict__ posq,
    const int* __restrict__ particleIndices,
    const real* __restrict__ charges,
    const double* __restrict__ bornRadii,        // ligand Born radii (double)
    const double* __restrict__ dRdPsi,           // ligand dR/dPsi [totalParticles]
    const double* __restrict__ d2RdPsi2,         // ligand d2R/dPsi2 [totalParticles]
    const double* __restrict__ ligJacobian,      // [totalParticles * dim3N] (J)
    const int* __restrict__ groupStart,
    int numGroups,
    int templateNumAtoms,
    const real4* __restrict__ receptorPositions,
    const real* __restrict__ receptorCharges,
    const double* __restrict__ recBorn,
    const double* __restrict__ recDRdPsi,
    const double* __restrict__ recD2RdPsi2,
    const double* __restrict__ dCrossDRL,        // [totalParticles]
    const double* __restrict__ dCrossDRR,        // [K*Nr]
    const double* __restrict__ JR,               // [K * Nr * n3]
    const double* __restrict__ MR,               // [K * Nr * Nr]
    int numReceptorAtoms,
    int n3,                                       // 3 * groupSize
    float prefactor,
    int totalParticles,
    HBUF_T* __restrict__ hessian)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int dim3N = 3 * totalParticles;
    if (tid >= dim3N * dim3N) return;
    int row = tid / dim3N;
    int col = tid % dim3N;
    if (col < row) return;
    int Nr = numReceptorAtoms;
    double pf = (double)prefactor;
    const double MIN_CROSS_R2 = 0.01;

    int atom_row = row / 3;
    int atom_col = col / 3;

    // Find the group of atom_row; require atom_col in the same group.
    int g = 0, gs = 0, ge = 0; bool found = false;
    for (int gg = 0; gg < numGroups; gg++) {
        gs = groupStart[gg]; ge = groupStart[gg + 1];
        if (atom_row >= gs && atom_row < ge) { g = gg; found = true; break; }
    }
    if (!found) return;
    if (atom_col < gs || atom_col >= ge) return;   // block diagonal only

    // group-local row/col within [0, n3)
    int lrow = row - 3*gs;
    int lcol = col - 3*gs;
    size_t jrBase = (size_t)g * Nr * n3;
    size_t mrBase = (size_t)g * Nr * Nr;

    double v = 0.0;

    // ---- desolvation JR^T MR JR ----
    for (int j = 0; j < Nr; j++) {
        double Jjr = JR[jrBase + (size_t)j*n3 + lrow];
        if (fabs(Jjr) < 1e-18) continue;
        for (int m = 0; m < Nr; m++) {
            double Mjm = MR[mrBase + (size_t)j*Nr + m];
            if (fabs(Mjm) < 1e-18) continue;
            v += Jjr * Mjm * JR[jrBase + (size_t)m*n3 + lcol];
        }
    }

    // ---- cross-term Born-Born + mixed r-Born couplings ----
    // For each (ligand i in group, receptor j): outer products of J[i]
    // (ligand Jacobian row for global atom gs+i) and JR[j], plus dr/dx
    // (nonzero only when the atom owning the row/col is ligand i).
    int groupSize = ge - gs;
    for (int iL = 0; iL < groupSize; iL++) {
        int gAtom = gs + iL;
        int tIdx = iL % templateNumAtoms;
        real4 pi = posq[particleIndices[gAtom]];
        double q_i = (double)charges[tIdx];
        double Ri = bornRadii[gAtom];
        double dRi_dPsi = dRdPsi[gAtom];

        // Precompute J[i] entries for this row/col (global jacobian rows).
        double JLi_row = ligJacobian[(size_t)gAtom*dim3N + row];
        double JLi_col = ligJacobian[(size_t)gAtom*dim3N + col];

        for (int j = 0; j < Nr; j++) {
            real4 pj = receptorPositions[j];
            double dx = (double)pi.x - (double)pj.x;   // lig - rec
            double dy = (double)pi.y - (double)pj.y;
            double dz = (double)pi.z - (double)pj.z;
            double r2 = dx*dx + dy*dy + dz*dz;
            if (r2 < MIN_CROSS_R2) continue;
            double r = sqrt(r2);
            double Rj = recBorn[(size_t)g*Nr + j];
            double dRj_dPsi = recDRdPsi[(size_t)g*Nr + j];
            double RaRb = Ri*Rj;
            double et = exp(-r2/(4.0*RaRb));
            double f2 = r2 + RaRb*et;
            double f = sqrt(f2);
            double C = pf * q_i * (double)receptorCharges[j];

            double df2_dRi = et*(Rj + 0.25*r2/Ri), df_dRi = df2_dRi/(2.0*f);
            double df2_dRj = et*(Ri + 0.25*r2/Rj), df_dRj = df2_dRj/(2.0*f);
            double df2_dr = 2.0*r*(1.0-0.25*et), df_dr = df2_dr/(2.0*f);

            double dalpha_dRi = et*r2/(4.0*Ri*Ri*Rj);
            double d2f2_dRi2 = dalpha_dRi*(Rj+0.25*r2/Ri) + et*(-0.25*r2/(Ri*Ri));
            double d2f_dRi2 = d2f2_dRi2/(2.0*f) - df2_dRi*df2_dRi/(4.0*f*f*f);
            double d2E_dRi2 = C*(2.0*df_dRi*df_dRi/(f*f*f) - d2f_dRi2/(f*f));
            double dalpha_dRj = et*r2/(4.0*Ri*Rj*Rj);
            double d2f2_dRj2 = dalpha_dRj*(Ri+0.25*r2/Rj) + et*(-0.25*r2/(Rj*Rj));
            double d2f_dRj2 = d2f2_dRj2/(2.0*f) - df2_dRj*df2_dRj/(4.0*f*f*f);
            double d2E_dRj2 = C*(2.0*df_dRj*df_dRj/(f*f*f) - d2f_dRj2/(f*f));
            double d2f2_dRiRj = dalpha_dRj*(Rj+0.25*r2/Ri) + et;
            double d2f_dRiRj = d2f2_dRiRj/(2.0*f) - df2_dRi*df2_dRj/(4.0*f*f*f);
            double d2E_dRiRj = C*(2.0*df_dRi*df_dRj/(f*f*f) - d2f_dRiRj/(f*f));

            double d2f2_dr_dRi = 2.0*r*(-0.25*dalpha_dRi);
            double d2f_dr_dRi = d2f2_dr_dRi/(2.0*f) - df2_dr*df2_dRi/(4.0*f*f*f);
            double d2E_dr_dRi = C*(2.0*df_dr*df_dRi/(f*f*f) - d2f_dr_dRi/(f*f));
            double d2f2_dr_dRj = 2.0*r*(-0.25*dalpha_dRj);
            double d2f_dr_dRj = d2f2_dr_dRj/(2.0*f) - df2_dr*df2_dRj/(4.0*f*f*f);
            double d2E_dr_dRj = C*(2.0*df_dr*df_dRj/(f*f*f) - d2f_dr_dRj/(f*f));

            double ir = 1.0/r;
            double D[3] = {dx, dy, dz};
            double cRi = d2E_dRi2 * dRi_dPsi*dRi_dPsi;
            double cRj = d2E_dRj2 * dRj_dPsi*dRj_dPsi;
            double cRiRj = d2E_dRiRj * dRi_dPsi*dRj_dPsi;
            double gri = d2E_dr_dRi * dRi_dPsi;
            double grj = d2E_dr_dRj * dRj_dPsi;

            double JRj_row = JR[jrBase + (size_t)j*n3 + lrow];
            double JRj_col = JR[jrBase + (size_t)j*n3 + lcol];

            // dr/dx is nonzero only for ligand atom i's rows/cols.
            double dr_row = (atom_row == gAtom) ? D[row%3]*ir : 0.0;
            double dr_col = (atom_col == gAtom) ? D[col%3]*ir : 0.0;

            v += cRi*JLi_row*JLi_col
               + cRj*JRj_row*JRj_col
               + cRiRj*(JLi_row*JRj_col + JRj_row*JLi_col)
               + gri*(dr_row*JLi_col + JLi_row*dr_col)
               + grj*(dr_row*JRj_col + JRj_row*dr_col);
        }
    }

    // ---- cross-term single-Born curvature ----
    // dCross_dRL[i] * d2R^L_i/dPsi2 routed through J[i] outer J[i].
    for (int iL = 0; iL < groupSize; iL++) {
        int gAtom = gs + iL;
        double w = dCrossDRL[gAtom] * d2RdPsi2[gAtom];
        if (fabs(w) < 1e-300) continue;
        double Jr = ligJacobian[(size_t)gAtom*dim3N + row];
        if (fabs(Jr) < 1e-300) continue;
        v += w * Jr * ligJacobian[(size_t)gAtom*dim3N + col];
    }
    // dCross_dRR[j] * d2R^R_j/dPsi2 routed through JR[j] outer JR[j].
    for (int j = 0; j < Nr; j++) {
        double w = dCrossDRR[(size_t)g*Nr + j] * recD2RdPsi2[(size_t)g*Nr + j];
        if (fabs(w) < 1e-300) continue;
        double Jr = JR[jrBase + (size_t)j*n3 + lrow];
        if (fabs(Jr) < 1e-300) continue;
        v += w * Jr * JR[jrBase + (size_t)j*n3 + lcol];
    }

    atomicAdd(&hessian[(size_t)row*dim3N + col], (HBUF_T)v);
    if (col > row) atomicAdd(&hessian[(size_t)col*dim3N + row], (HBUF_T)v);
}
