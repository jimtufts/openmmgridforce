/**
 * CUDA implementation of analytical Hessians for bonded and nonbonded interactions.
 *
 * This kernel computes exact analytical second derivatives for:
 *   - Harmonic bond: E = 0.5 * k * (r - r0)^2
 *   - Harmonic angle: E = 0.5 * k * (theta - theta0)^2
 *   - Periodic torsion: E = k * (1 + cos(n*phi - phi0))
 *   - Nonbonded pairs: E = 4*eps*((sigma/r)^12 - (sigma/r)^6) + q1*q2/(4*pi*eps0*r)
 *
 * The torsion implementation uses the Blondel-Karplus formulation which
 * avoids singularities at linear configurations.
 *
 * Reference: Blondel & Karplus (1996) J. Comput. Chem. 17, 1132-1141
 */

#define M_PI_F 3.14159265358979323846f
#define ONE_4PI_EPS0 138.935456f  // kJ*nm/(mol*e^2)

// Fixed-point scale for deterministic atomicAdd (same as IsolatedNonbondedForce)
#define HESSIAN_SCALE 0x40000000  // 2^30 (soft-mode resolution margin)

// ============================================================
// Helper functions (real = float in single/mixed, double in double)
// ============================================================

__device__ inline real3 cross(real3 a, real3 b) {
    return make_real3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

__device__ inline real dot(real3 a, real3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ inline real length_sq(real3 v) {
    return dot(v, v);
}

__device__ inline real length(real3 v) {
    return sqrt(length_sq(v));
}

__device__ inline real3 sub(real3 a, real3 b) {
    return make_real3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ inline real3 add(real3 a, real3 b) {
    return make_real3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ inline real3 scale(real3 v, real s) {
    return make_real3(v.x * s, v.y * s, v.z * s);
}

// ============================================================
// Bond Hessian Kernel
// ============================================================

/**
 * Compute analytical Hessian for harmonic bond: E = 0.5 * k * (r - r0)^2
 *
 * The Hessian has the form:
 *   d²E/dr1² = k * [r̂⊗r̂ + (1 - r0/r) * (I - r̂⊗r̂)]
 *   d²E/dr2² = d²E/dr1²
 *   d²E/dr1dr2 = -d²E/dr1²
 *
 * @param p1, p2   Atom positions
 * @param k        Force constant (kJ/mol/nm^2)
 * @param r0       Equilibrium distance (nm)
 * @param hess     Output: 6x6 Hessian matrix (row-major)
 */
__device__ void computeBondHessian(
    real3 p1, real3 p2,
    real k, real r0,
    real* hess  // 36 entries
) {
    // Initialize to zero
    for (int i = 0; i < 36; i++) hess[i] = 0.0f;

    real3 r_vec = sub(p2, p1);
    real r = length(r_vec);

    if (r < 1e-10f) return;

    real3 r_hat = scale(r_vec, 1.0f / r);

    // Outer product r_hat ⊗ r_hat
    real rr[9];
    real rv[3] = {r_hat.x, r_hat.y, r_hat.z};
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            rr[i*3+j] = rv[i] * rv[j];
        }
    }

    // d²E/dr1² = k * [rr + (1 - r0/r) * (I - rr)]
    real factor = 1.0f - r0 / r;
    real block[9];
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            real I_ij = (i == j) ? 1.0f : 0.0f;
            block[i*3+j] = k * (rr[i*3+j] + factor * (I_ij - rr[i*3+j]));
        }
    }

    // Fill 6x6 Hessian
    // H[0:3, 0:3] = block (d²E/dr1²)
    // H[3:6, 3:6] = block (d²E/dr2²)
    // H[0:3, 3:6] = -block (d²E/dr1dr2)
    // H[3:6, 0:3] = -block (d²E/dr2dr1)
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            hess[i*6 + j] = block[i*3+j];           // top-left
            hess[(i+3)*6 + (j+3)] = block[i*3+j];   // bottom-right
            hess[i*6 + (j+3)] = -block[i*3+j];      // top-right
            hess[(i+3)*6 + j] = -block[i*3+j];      // bottom-left
        }
    }
}

/**
 * Kernel to compute bond Hessians for all bonds in the system.
 */
extern "C" __global__ void computeBondHessians(
    const real4* __restrict__ posq,
    const int* __restrict__ bondAtoms,      // [numBonds * 2]: atom indices
    const float* __restrict__ bondParams,   // [numBonds * 2]: k, r0
    unsigned long long* __restrict__ globalHessian,
    int numBonds,
    int numAtoms
) {
    int bondIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (bondIdx >= numBonds) return;

    int i1 = bondAtoms[bondIdx * 2 + 0];
    int i2 = bondAtoms[bondIdx * 2 + 1];

    real4 pos1 = posq[i1];
    real4 pos2 = posq[i2];

    real3 p1 = make_real3(pos1.x, pos1.y, pos1.z);
    real3 p2 = make_real3(pos2.x, pos2.y, pos2.z);

    real k = bondParams[bondIdx * 2 + 0];
    real r0 = bondParams[bondIdx * 2 + 1];

    real localHess[36];
    computeBondHessian(p1, p2, k, r0, localHess);

    // Accumulate into global Hessian using fixed-point for determinism
    int atomIndices[2] = {i1, i2};
    int stride = numAtoms * 3;

    for (int localI = 0; localI < 2; localI++) {
        for (int localJ = 0; localJ < 2; localJ++) {
            int globalI = atomIndices[localI];
            int globalJ = atomIndices[localJ];

            for (int di = 0; di < 3; di++) {
                for (int dj = 0; dj < 3; dj++) {
                    int globalRow = globalI * 3 + di;
                    int globalCol = globalJ * 3 + dj;
                    int localRow = localI * 3 + di;
                    int localCol = localJ * 3 + dj;

                    real val = localHess[localRow * 6 + localCol];
                    atomicAdd(&globalHessian[globalRow * stride + globalCol],
                              static_cast<unsigned long long>(static_cast<long long>(val * HESSIAN_SCALE)));
                }
            }
        }
    }
}

// ============================================================
// Angle Hessian Kernel
// ============================================================

/**
 * Compute analytical Hessian for harmonic angle: E = 0.5 * k * (theta - theta0)^2
 *
 * Uses fully analytical formulas derived from:
 *   theta = acos((r21 · r23) / (|r21| |r23|))
 *
 * The Hessian has the form:
 *   d²E/dri drj = k * (dθ/dri)(dθ/drj) + k*(θ - θ0) * d²θ/dri drj
 *
 * @param p1, p2, p3  Atom positions (p2 is central atom)
 * @param k           Force constant (kJ/mol/rad^2)
 * @param theta0      Equilibrium angle (radians)
 * @param hess        Output: 9x9 Hessian matrix (row-major)
 */
__device__ void computeAngleHessian(
    real3 p1, real3 p2, real3 p3,
    real k, real theta0,
    real* hess  // 81 entries
) {
    // Initialize to zero
    for (int i = 0; i < 81; i++) hess[i] = 0.0f;

    // Vectors from central atom
    real3 r21 = sub(p1, p2);
    real3 r23 = sub(p3, p2);

    real L1 = length(r21);
    real L3 = length(r23);

    if (L1 < 1e-10f || L3 < 1e-10f) return;

    real invL1 = 1.0f / L1;
    real invL3 = 1.0f / L3;
    real invL1_sq = invL1 * invL1;
    real invL3_sq = invL3 * invL3;

    real3 e1 = scale(r21, invL1);  // unit vector along r21
    real3 e3 = scale(r23, invL3);  // unit vector along r23

    real cos_theta = dot(e1, e3);
    cos_theta = fmin((real)0.9999999f, fmax((real)-0.9999999f, cos_theta));
    real theta = acos(cos_theta);
    real sin_theta = sin(theta);

    if (fabs(sin_theta) < 1e-10f) return;

    real inv_sin = 1.0f / sin_theta;
    real cot_theta = cos_theta * inv_sin;

    // Energy derivatives
    real dtheta = theta - theta0;
    real dE_dtheta = k * dtheta;
    real d2E_dtheta2 = k;

    // ============================================
    // Gradient of theta: dθ/dr_i
    // ============================================
    // dθ/dr1 = -1/(L1 sin θ) * (e3 - cos θ * e1)
    // dθ/dr3 = -1/(L3 sin θ) * (e1 - cos θ * e3)
    // dθ/dr2 = -(dθ/dr1 + dθ/dr3)

    real3 v1 = sub(e3, scale(e1, cos_theta));  // e3 - cos θ * e1
    real3 v3 = sub(e1, scale(e3, cos_theta));  // e1 - cos θ * e3

    real3 g1 = scale(v1, -inv_sin * invL1);
    real3 g3 = scale(v3, -inv_sin * invL3);
    real3 g2 = scale(add(g1, g3), -1.0f);

    real grad[9];
    grad[0] = g1.x; grad[1] = g1.y; grad[2] = g1.z;
    grad[3] = g2.x; grad[4] = g2.y; grad[5] = g2.z;
    grad[6] = g3.x; grad[7] = g3.y; grad[8] = g3.z;

    // ============================================
    // Analytical Hessian of theta: d²θ/dr_i dr_j
    // ============================================
    // We need to differentiate the gradient expressions.
    //
    // Key derivatives needed:
    //   de1/dr1 = (I - e1⊗e1)/L1,  de1/dr2 = -(I - e1⊗e1)/L1,  de1/dr3 = 0
    //   de3/dr3 = (I - e3⊗e3)/L3,  de3/dr2 = -(I - e3⊗e3)/L3,  de3/dr1 = 0
    //   dL1/dr1 = e1,  dL1/dr2 = -e1,  dL1/dr3 = 0
    //   dL3/dr3 = e3,  dL3/dr2 = -e3,  dL3/dr1 = 0
    //   d(cos θ)/dr1 = (e3 - cos θ * e1)/L1 = v1/L1
    //   d(cos θ)/dr3 = (e1 - cos θ * e3)/L3 = v3/L3
    //   d(sin θ)/dr = -cos θ / sin θ * d(cos θ)/dr = -cot θ * d(cos θ)/dr
    //
    // Let P1 = (I - e1⊗e1), P3 = (I - e3⊗e3) (projection matrices)

    // Store unit vector components for outer products
    real e1v[3] = {e1.x, e1.y, e1.z};
    real e3v[3] = {e3.x, e3.y, e3.z};
    real v1v[3] = {v1.x, v1.y, v1.z};
    real v3v[3] = {v3.x, v3.y, v3.z};

    // Projection matrices P1 = I - e1⊗e1, P3 = I - e3⊗e3
    real P1[9], P3[9];
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            real delta = (i == j) ? 1.0f : 0.0f;
            P1[i*3+j] = delta - e1v[i] * e1v[j];
            P3[i*3+j] = delta - e3v[i] * e3v[j];
        }
    }

    // Initialize H_theta (9x9 matrix for d²θ/dr)
    real H_theta[81];
    for (int i = 0; i < 81; i++) H_theta[i] = 0.0f;

    // ----------------------------------------
    // d²θ/dr1 dr1
    // ----------------------------------------
    // g1 = -inv_sin * invL1 * (e3 - cos θ * e1)
    // dg1/dr1 involves: d(invL1)/dr1, d(inv_sin)/dr1, d(e1)/dr1, d(cos θ)/dr1
    //
    // d(invL1)/dr1 = -invL1² * e1
    // d(inv_sin)/dr1 = -inv_sin * cot θ * d(cos θ)/dr1 = -inv_sin * cot θ * v1/L1
    // d(e1)/dr1 = P1/L1
    // d(cos θ * e1)/dr1 = d(cos θ)/dr1 ⊗ e1 + cos θ * d(e1)/dr1
    //                   = (v1/L1) ⊗ e1 + cos θ * P1/L1
    //
    // Full derivative (factor out -inv_sin):
    // dg1/dr1 = -inv_sin * invL1 * [-(e3 - cos θ * e1) ⊗ (e1/L1) - cot θ * (e3 - cos θ * e1) ⊗ (v1/L1) + P3/L1*0 - (v1/L1 ⊗ e1 + cos θ * P1/L1)]
    //         = -inv_sin * invL1 * [-v1 ⊗ e1 * invL1 - cot θ * v1 ⊗ v1 * invL1 - v1 ⊗ e1 * invL1 - cos θ * P1 * invL1]
    //         = inv_sin * invL1² * [2 * v1 ⊗ e1 + cot θ * v1 ⊗ v1 + cos θ * P1]

    for (int a = 0; a < 3; a++) {
        for (int b = 0; b < 3; b++) {
            real val = inv_sin * invL1_sq * (
                2.0f * v1v[a] * e1v[b] +
                cot_theta * v1v[a] * v1v[b] +
                cos_theta * P1[a*3+b]
            );
            H_theta[(0+a)*9 + (0+b)] = val;
        }
    }

    // ----------------------------------------
    // d²θ/dr3 dr3
    // ----------------------------------------
    // By symmetry with dr1 dr1:
    for (int a = 0; a < 3; a++) {
        for (int b = 0; b < 3; b++) {
            real val = inv_sin * invL3_sq * (
                2.0f * v3v[a] * e3v[b] +
                cot_theta * v3v[a] * v3v[b] +
                cos_theta * P3[a*3+b]
            );
            H_theta[(6+a)*9 + (6+b)] = val;
        }
    }

    // ----------------------------------------
    // d²θ/dr1 dr3
    // ----------------------------------------
    // g1 = -inv_sin * invL1 * v1
    // dg1/dr3 involves only: d(inv_sin)/dr3, since e1, L1 don't depend on r3
    //   but v1 = e3 - cos θ * e1, so dv1/dr3 = de3/dr3 - d(cos θ)/dr3 * e1 = P3/L3 - v3/L3 ⊗ e1
    //
    // dg1/dr3 = -inv_sin * invL1 * [dv1/dr3 + v1 ⊗ d(-inv_sin)/dr3 / (-inv_sin)]
    //         = -inv_sin * invL1 * [P3/L3 - v3 ⊗ e1 / L3 - cot θ * v1 ⊗ v3 / L3]
    //         = -inv_sin * invL1 * invL3 * [P3 - v3 ⊗ e1 - cot θ * v1 ⊗ v3]

    for (int a = 0; a < 3; a++) {
        for (int b = 0; b < 3; b++) {
            real val = -inv_sin * invL1 * invL3 * (
                P3[a*3+b] -
                v3v[a] * e1v[b] -
                cot_theta * v1v[a] * v3v[b]
            );
            H_theta[(0+a)*9 + (6+b)] = val;
            H_theta[(6+b)*9 + (0+a)] = val;  // symmetric
        }
    }

    // ----------------------------------------
    // d²θ/dr1 dr2 and d²θ/dr3 dr2
    // ----------------------------------------
    // g2 = -(g1 + g3), so dg2/dr_i = -(dg1/dr_i + dg3/dr_i)
    // Also, dg1/dr2 and dg3/dr2 need to be computed.
    //
    // For g1 = -inv_sin * invL1 * v1:
    // dg1/dr2 involves: d(invL1)/dr2 = invL1² * e1, d(inv_sin)/dr2, de1/dr2 = -P1/L1, dv1/dr2
    //   d(cos θ)/dr2 = -v1/L1 - v3/L3
    //   dv1/dr2 = de3/dr2 - d(cos θ)/dr2 * e1 - cos θ * de1/dr2
    //           = -P3/L3 + (v1/L1 + v3/L3) ⊗ e1 + cos θ * P1/L1
    //
    // This is getting complex. For dr2 terms, use the relation:
    //   d²θ/dr2 dr_j = -d²θ/dr1 dr_j - d²θ/dr3 dr_j  (due to g2 = -(g1 + g3))
    //   d²θ/dr_i dr2 = -d²θ/dr_i dr1 - d²θ/dr_i dr3

    // Fill in dr1 dr2 and dr2 dr1
    for (int a = 0; a < 3; a++) {
        for (int b = 0; b < 3; b++) {
            real d2_r1_r1 = H_theta[(0+a)*9 + (0+b)];
            real d2_r1_r3 = H_theta[(0+a)*9 + (6+b)];
            // d²θ/dr1 dr2 = -d²θ/dr1 dr1 - d²θ/dr1 dr3
            real val = -d2_r1_r1 - d2_r1_r3;
            H_theta[(0+a)*9 + (3+b)] = val;
            H_theta[(3+b)*9 + (0+a)] = val;  // symmetric
        }
    }

    // Fill in dr3 dr2 and dr2 dr3
    for (int a = 0; a < 3; a++) {
        for (int b = 0; b < 3; b++) {
            real d2_r3_r3 = H_theta[(6+a)*9 + (6+b)];
            real d2_r3_r1 = H_theta[(6+a)*9 + (0+b)];
            // d²θ/dr3 dr2 = -d²θ/dr3 dr3 - d²θ/dr3 dr1
            real val = -d2_r3_r3 - d2_r3_r1;
            H_theta[(6+a)*9 + (3+b)] = val;
            H_theta[(3+b)*9 + (6+a)] = val;  // symmetric
        }
    }

    // Fill in dr2 dr2
    // d²θ/dr2 dr2 = -d²θ/dr1 dr2 - d²θ/dr3 dr2
    for (int a = 0; a < 3; a++) {
        for (int b = 0; b < 3; b++) {
            real d2_r1_r2 = H_theta[(0+a)*9 + (3+b)];
            real d2_r3_r2 = H_theta[(6+a)*9 + (3+b)];
            H_theta[(3+a)*9 + (3+b)] = -d2_r1_r2 - d2_r3_r2;
        }
    }

    // ============================================
    // Full Hessian: H = d²E/dθ² * grad ⊗ grad + dE/dθ * H_theta
    // ============================================
    for (int i = 0; i < 9; i++) {
        for (int j = 0; j < 9; j++) {
            hess[i * 9 + j] = d2E_dtheta2 * grad[i] * grad[j] + dE_dtheta * H_theta[i * 9 + j];
        }
    }
}

/**
 * Kernel to compute angle Hessians for all angles in the system.
 */
extern "C" __global__ void computeAngleHessians(
    const real4* __restrict__ posq,
    const int* __restrict__ angleAtoms,      // [numAngles * 3]: atom indices
    const float* __restrict__ angleParams,   // [numAngles * 2]: k, theta0
    unsigned long long* __restrict__ globalHessian,
    int numAngles,
    int numAtoms
) {
    int angleIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (angleIdx >= numAngles) return;

    int i1 = angleAtoms[angleIdx * 3 + 0];
    int i2 = angleAtoms[angleIdx * 3 + 1];  // central atom
    int i3 = angleAtoms[angleIdx * 3 + 2];

    real4 pos1 = posq[i1];
    real4 pos2 = posq[i2];
    real4 pos3 = posq[i3];

    real3 p1 = make_real3(pos1.x, pos1.y, pos1.z);
    real3 p2 = make_real3(pos2.x, pos2.y, pos2.z);
    real3 p3 = make_real3(pos3.x, pos3.y, pos3.z);

    real k = angleParams[angleIdx * 2 + 0];
    real theta0 = angleParams[angleIdx * 2 + 1];

    real localHess[81];
    computeAngleHessian(p1, p2, p3, k, theta0, localHess);

    // Accumulate into global Hessian using fixed-point for determinism
    int atomIndices[3] = {i1, i2, i3};
    int stride = numAtoms * 3;

    for (int localI = 0; localI < 3; localI++) {
        for (int localJ = 0; localJ < 3; localJ++) {
            int globalI = atomIndices[localI];
            int globalJ = atomIndices[localJ];

            for (int di = 0; di < 3; di++) {
                for (int dj = 0; dj < 3; dj++) {
                    int globalRow = globalI * 3 + di;
                    int globalCol = globalJ * 3 + dj;
                    int localRow = localI * 3 + di;
                    int localCol = localJ * 3 + dj;

                    real val = localHess[localRow * 9 + localCol];
                    atomicAdd(&globalHessian[globalRow * stride + globalCol],
                              static_cast<unsigned long long>(static_cast<long long>(val * HESSIAN_SCALE)));
                }
            }
        }
    }
}

// ============================================================
// Torsion Hessian Kernel
// ============================================================

/**
 * Compute dihedral angle and gradient using Blondel-Karplus formulation.
 *
 * @param p1, p2, p3, p4  Atom positions
 * @param phi             Output: dihedral angle in radians
 * @param grad            Output: gradient (4 x 3 array, row-major)
 */
__device__ void computeDihedralAndGradient(
    real3 p1, real3 p2, real3 p3, real3 p4,
    real& phi,
    real* grad  // 12 entries: grad[0-2]=dr1, grad[3-5]=dr2, etc.
) {
    // Bond vectors
    real3 b1 = sub(p2, p1);
    real3 b2 = sub(p3, p2);
    real3 b3 = sub(p4, p3);

    // Normal vectors to planes
    real3 m = cross(b1, b2);
    real3 n = cross(b2, b3);

    real m_sq = length_sq(m);
    real n_sq = length_sq(n);
    real b2_sq = length_sq(b2);
    real b2_norm = sqrt(b2_sq);

    // Degenerate case check
    if (m_sq < 1e-20f || n_sq < 1e-20f || b2_sq < 1e-20f) {
        phi = 0.0f;
        for (int i = 0; i < 12; i++) grad[i] = 0.0f;
        return;
    }

    // Normalized vectors
    real m_norm = sqrt(m_sq);
    real n_norm = sqrt(n_sq);
    real3 m_hat = scale(m, 1.0f / m_norm);
    real3 n_hat = scale(n, 1.0f / n_norm);
    real3 b2_hat = scale(b2, 1.0f / b2_norm);

    // Dihedral angle
    real cos_phi = dot(m_hat, n_hat);
    real3 m_cross_b2 = cross(m_hat, b2_hat);
    real sin_phi = dot(m_cross_b2, n_hat);
    phi = atan2(sin_phi, cos_phi);

    // Gradient of phi with respect to atom positions
    // dphi/dr1 = (|b2| / |m|^2) * m
    // dphi/dr4 = -(|b2| / |n|^2) * n
    real3 dphi_dr1 = scale(m, b2_norm / m_sq);
    real3 dphi_dr4 = scale(n, -b2_norm / n_sq);

    // Projection factors
    real b1_dot_b2 = dot(b1, b2);
    real b3_dot_b2 = dot(b3, b2);
    real alpha = b1_dot_b2 / b2_sq;
    real beta = b3_dot_b2 / b2_sq;

    // Corrected coefficients for middle atoms
    real c1 = -(1.0f + alpha);
    real c4 = beta;
    real d1 = alpha;
    real d4 = -(1.0f + beta);

    // dphi/dr2 = c1 * dphi/dr1 + c4 * dphi/dr4
    // dphi/dr3 = d1 * dphi/dr1 + d4 * dphi/dr4
    real3 dphi_dr2 = add(scale(dphi_dr1, c1), scale(dphi_dr4, c4));
    real3 dphi_dr3 = add(scale(dphi_dr1, d1), scale(dphi_dr4, d4));

    // Store gradient
    grad[0] = dphi_dr1.x; grad[1] = dphi_dr1.y; grad[2] = dphi_dr1.z;
    grad[3] = dphi_dr2.x; grad[4] = dphi_dr2.y; grad[5] = dphi_dr2.z;
    grad[6] = dphi_dr3.x; grad[7] = dphi_dr3.y; grad[8] = dphi_dr3.z;
    grad[9] = dphi_dr4.x; grad[10] = dphi_dr4.y; grad[11] = dphi_dr4.z;
}

/**
 * Compute analytical Hessian of the dihedral angle d^2phi/dri drj.
 *
 * Uses the Blondel-Karplus formulation with analytically derived second derivatives.
 *
 * @param p1, p2, p3, p4  Atom positions
 * @param hess            Output: 12x12 Hessian matrix (row-major, symmetric)
 */
__device__ void computeDihedralHessian(
    real3 p1, real3 p2, real3 p3, real3 p4,
    real* hess  // 144 entries
) {
    // Initialize to zero
    for (int i = 0; i < 144; i++) hess[i] = 0.0f;

    // Bond vectors
    real3 b1 = sub(p2, p1);
    real3 b2 = sub(p3, p2);
    real3 b3 = sub(p4, p3);

    // Normal vectors
    real3 m = cross(b1, b2);
    real3 n = cross(b2, b3);

    real m_sq = length_sq(m);
    real n_sq = length_sq(n);
    real b2_sq = length_sq(b2);
    real b2_norm = sqrt(b2_sq);

    if (m_sq < 1e-20f || n_sq < 1e-20f || b2_sq < 1e-20f) {
        return;
    }

    // Precompute outer products (stored as 3x3 arrays, row-major)
    real mm[9], nn[9];
    real mv[3] = {m.x, m.y, m.z};
    real nv[3] = {n.x, n.y, n.z};
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            mm[i*3+j] = mv[i] * mv[j];
            nn[i*3+j] = nv[i] * nv[j];
        }
    }

    // Skew-symmetric matrices for cross products
    // [v]x = [[0, -vz, vy], [vz, 0, -vx], [-vy, vx, 0]]
    real b1_x[9] = {0, -b1.z, b1.y, b1.z, 0, -b1.x, -b1.y, b1.x, 0};
    real b2_x[9] = {0, -b2.z, b2.y, b2.z, 0, -b2.x, -b2.y, b2.x, 0};
    real b3_x[9] = {0, -b3.z, b3.y, b3.z, 0, -b3.x, -b3.y, b3.x, 0};

    // Derivatives of m = b1 x b2 with respect to positions (3x3 matrices)
    // dm/dp1 = [b2]x
    // dm/dp2 = -[b2]x - [b1]x
    // dm/dp3 = [b1]x
    // dm/dp4 = 0
    real dm_dp[4][9];
    for (int i = 0; i < 9; i++) {
        dm_dp[0][i] = b2_x[i];
        dm_dp[1][i] = -b2_x[i] - b1_x[i];
        dm_dp[2][i] = b1_x[i];
        dm_dp[3][i] = 0.0f;
    }

    // Derivatives of n = b2 x b3 with respect to positions
    // dn/dp1 = 0
    // dn/dp2 = [b3]x
    // dn/dp3 = -[b3]x - [b2]x
    // dn/dp4 = [b2]x
    real dn_dp[4][9];
    for (int i = 0; i < 9; i++) {
        dn_dp[0][i] = 0.0f;
        dn_dp[1][i] = b3_x[i];
        dn_dp[2][i] = -b3_x[i] - b2_x[i];
        dn_dp[3][i] = b2_x[i];
    }

    // Gradient components
    real3 G1 = scale(m, b2_norm / m_sq);
    real3 G4 = scale(n, -b2_norm / n_sq);

    // Coefficients
    real b1_dot_b2 = dot(b1, b2);
    real b3_dot_b2 = dot(b3, b2);
    real alpha = b1_dot_b2 / b2_sq;
    real beta = b3_dot_b2 / b2_sq;
    real c1 = -(1.0f + alpha);
    real c4 = beta;
    real d1 = alpha;
    real d4 = -(1.0f + beta);

    // Derivative of |b2| with respect to positions
    real db2_norm_dp[4][3];
    real b2v[3] = {b2.x, b2.y, b2.z};
    for (int i = 0; i < 3; i++) {
        db2_norm_dp[0][i] = 0.0f;
        db2_norm_dp[1][i] = -b2v[i] / b2_norm;
        db2_norm_dp[2][i] = b2v[i] / b2_norm;
        db2_norm_dp[3][i] = 0.0f;
    }

    // Derivatives of b1.b2, b3.b2, b2^2
    real b1v[3] = {b1.x, b1.y, b1.z};
    real b3v[3] = {b3.x, b3.y, b3.z};

    real db1_dot_b2_dp[4][3], db3_dot_b2_dp[4][3], db2_sq_dp[4][3];
    for (int i = 0; i < 3; i++) {
        db1_dot_b2_dp[0][i] = -b2v[i];
        db1_dot_b2_dp[1][i] = b2v[i] - b1v[i];
        db1_dot_b2_dp[2][i] = b1v[i];
        db1_dot_b2_dp[3][i] = 0.0f;

        db3_dot_b2_dp[0][i] = 0.0f;
        db3_dot_b2_dp[1][i] = -b3v[i];
        db3_dot_b2_dp[2][i] = b3v[i] - b2v[i];
        db3_dot_b2_dp[3][i] = b2v[i];

        db2_sq_dp[0][i] = 0.0f;
        db2_sq_dp[1][i] = -2.0f * b2v[i];
        db2_sq_dp[2][i] = 2.0f * b2v[i];
        db2_sq_dp[3][i] = 0.0f;
    }

    // Compute dG1/dpj and dG4/dpj (3x3 matrices for each j)
    real dG1_dp[4][9], dG4_dp[4][9];

    for (int j = 0; j < 4; j++) {
        // dG1/dpj = outer(m, db2_norm/dpj) / m_sq + (b2_norm/m_sq) * (I - 2*mm/m_sq) @ dm/dpj
        // dG4/dpj = -outer(n, db2_norm/dpj) / n_sq - (b2_norm/n_sq) * (I - 2*nn/n_sq) @ dn/dpj

        for (int a = 0; a < 3; a++) {
            for (int b = 0; b < 3; b++) {
                int idx = a * 3 + b;

                // term1 for G1: m[a] * db2_norm[j][b] / m_sq
                real term1_G1 = mv[a] * db2_norm_dp[j][b] / m_sq;

                // term2 for G1: (b2_norm/m_sq) * ((I - 2*mm/m_sq) @ dm_dp[j])_ab
                // (I - 2*mm/m_sq)_ac * dm_dp[j]_cb
                real sum_G1 = 0.0f;
                for (int c = 0; c < 3; c++) {
                    real factor = (a == c ? 1.0f : 0.0f) - 2.0f * mm[a*3+c] / m_sq;
                    sum_G1 += factor * dm_dp[j][c*3+b];
                }
                real term2_G1 = (b2_norm / m_sq) * sum_G1;

                dG1_dp[j][idx] = term1_G1 + term2_G1;

                // term1 for G4: -n[a] * db2_norm[j][b] / n_sq
                real term1_G4 = -nv[a] * db2_norm_dp[j][b] / n_sq;

                // term2 for G4: -(b2_norm/n_sq) * ((I - 2*nn/n_sq) @ dn_dp[j])_ab
                real sum_G4 = 0.0f;
                for (int c = 0; c < 3; c++) {
                    real factor = (a == c ? 1.0f : 0.0f) - 2.0f * nn[a*3+c] / n_sq;
                    sum_G4 += factor * dn_dp[j][c*3+b];
                }
                real term2_G4 = -(b2_norm / n_sq) * sum_G4;

                dG4_dp[j][idx] = term1_G4 + term2_G4;
            }
        }
    }

    // Coefficient derivatives
    real dc1_dp[4][3], dc4_dp[4][3], dd1_dp[4][3], dd4_dp[4][3];

    for (int j = 0; j < 4; j++) {
        for (int i = 0; i < 3; i++) {
            // c1 = -(1 + alpha) = -(1 + b1.b2/b2^2)
            dc1_dp[j][i] = -db1_dot_b2_dp[j][i] / b2_sq + b1_dot_b2 * db2_sq_dp[j][i] / (b2_sq * b2_sq);

            // c4 = beta = b3.b2/b2^2
            dc4_dp[j][i] = db3_dot_b2_dp[j][i] / b2_sq - b3_dot_b2 * db2_sq_dp[j][i] / (b2_sq * b2_sq);

            // d1 = alpha = b1.b2/b2^2
            dd1_dp[j][i] = db1_dot_b2_dp[j][i] / b2_sq - b1_dot_b2 * db2_sq_dp[j][i] / (b2_sq * b2_sq);

            // d4 = -(1 + beta) = -(1 + b3.b2/b2^2)
            dd4_dp[j][i] = -db3_dot_b2_dp[j][i] / b2_sq + b3_dot_b2 * db2_sq_dp[j][i] / (b2_sq * b2_sq);
        }
    }

    // Compute dG2/dpj and dG3/dpj
    // G2 = c1*G1 + c4*G4
    // dG2/dpj = outer(G1, dc1/dpj) + c1*dG1/dpj + outer(G4, dc4/dpj) + c4*dG4/dpj
    real G1v[3] = {G1.x, G1.y, G1.z};
    real G4v[3] = {G4.x, G4.y, G4.z};

    real dG2_dp[4][9], dG3_dp[4][9];

    for (int j = 0; j < 4; j++) {
        for (int a = 0; a < 3; a++) {
            for (int b = 0; b < 3; b++) {
                int idx = a * 3 + b;

                // dG2/dpj
                dG2_dp[j][idx] = G1v[a] * dc1_dp[j][b] + c1 * dG1_dp[j][idx]
                               + G4v[a] * dc4_dp[j][b] + c4 * dG4_dp[j][idx];

                // dG3/dpj
                dG3_dp[j][idx] = G1v[a] * dd1_dp[j][b] + d1 * dG1_dp[j][idx]
                               + G4v[a] * dd4_dp[j][b] + d4 * dG4_dp[j][idx];
            }
        }
    }

    // Assemble full 12x12 Hessian
    // H[3*i:3*i+3, 3*j:3*j+3] = dGi/dpj
    for (int i = 0; i < 4; i++) {  // gradient block (atom i)
        for (int j = 0; j < 4; j++) {  // derivative with respect to atom j
            real* dGi_dpj;
            if (i == 0) dGi_dpj = dG1_dp[j];
            else if (i == 1) dGi_dpj = dG2_dp[j];
            else if (i == 2) dGi_dpj = dG3_dp[j];
            else dGi_dpj = dG4_dp[j];

            for (int a = 0; a < 3; a++) {
                for (int b = 0; b < 3; b++) {
                    hess[(3*i + a) * 12 + (3*j + b)] = dGi_dpj[a * 3 + b];
                }
            }
        }
    }

    // Symmetrize
    for (int i = 0; i < 12; i++) {
        for (int j = i + 1; j < 12; j++) {
            real avg = 0.5f * (hess[i * 12 + j] + hess[j * 12 + i]);
            hess[i * 12 + j] = avg;
            hess[j * 12 + i] = avg;
        }
    }
}

/**
 * Compute torsion energy Hessian for a single torsion.
 *
 * For E = k * (1 + cos(n*phi - phi0)):
 *   d^2E/dri drj = d^2E/dphi^2 * (dphi/dri)(dphi/drj) + dE/dphi * d^2phi/dri drj
 *
 * @param p1, p2, p3, p4  Atom positions
 * @param k               Force constant (kJ/mol)
 * @param n               Periodicity
 * @param phi0            Phase offset (radians)
 * @param hess            Output: 12x12 Hessian matrix (kJ/mol/nm^2)
 */
__device__ void computeTorsionHessian(
    real3 p1, real3 p2, real3 p3, real3 p4,
    real k, int n, real phi0,
    real* hess  // 144 entries
) {
    // Get dihedral angle and gradient
    real phi;
    real grad[12];
    computeDihedralAndGradient(p1, p2, p3, p4, phi, grad);

    // Get dihedral Hessian
    real H_phi[144];
    computeDihedralHessian(p1, p2, p3, p4, H_phi);

    // Energy derivatives
    real arg = n * phi - phi0;
    real dE_dphi = -k * n * sin(arg);
    real d2E_dphi2 = -k * n * n * cos(arg);

    // Full Hessian via chain rule
    // H = d2E_dphi2 * outer(grad, grad) + dE_dphi * H_phi
    for (int i = 0; i < 12; i++) {
        for (int j = 0; j < 12; j++) {
            hess[i * 12 + j] = d2E_dphi2 * grad[i] * grad[j] + dE_dphi * H_phi[i * 12 + j];
        }
    }
}

/**
 * Kernel to compute torsion Hessians for all torsions in the system.
 *
 * Each thread computes one torsion's contribution to the full system Hessian.
 * The output is accumulated into the global Hessian using atomic operations.
 *
 * @param posq           Atom positions (float4: x, y, z, charge)
 * @param torsionParams  Torsion parameters: [i1, i2, i3, i4, n, k, phi0] per torsion
 * @param globalHessian  Output: Full N*3 x N*3 Hessian (accumulated atomically)
 * @param numTorsions    Number of torsions
 * @param numAtoms       Total number of atoms
 */
extern "C" __global__ void computeTorsionHessians(
    const real4* __restrict__ posq,
    const int* __restrict__ torsionAtoms,      // [numTorsions * 4]: atom indices
    const float* __restrict__ torsionParams,   // [numTorsions * 3]: n, k, phi0
    unsigned long long* __restrict__ globalHessian,  // [numAtoms * 3 * numAtoms * 3]
    int numTorsions,
    int numAtoms
) {
    int torsionIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (torsionIdx >= numTorsions) return;

    // Load atom indices
    int i1 = torsionAtoms[torsionIdx * 4 + 0];
    int i2 = torsionAtoms[torsionIdx * 4 + 1];
    int i3 = torsionAtoms[torsionIdx * 4 + 2];
    int i4 = torsionAtoms[torsionIdx * 4 + 3];

    // Load positions
    real4 pos1 = posq[i1];
    real4 pos2 = posq[i2];
    real4 pos3 = posq[i3];
    real4 pos4 = posq[i4];

    real3 p1 = make_real3(pos1.x, pos1.y, pos1.z);
    real3 p2 = make_real3(pos2.x, pos2.y, pos2.z);
    real3 p3 = make_real3(pos3.x, pos3.y, pos3.z);
    real3 p4 = make_real3(pos4.x, pos4.y, pos4.z);

    // Load parameters
    int n = (int)torsionParams[torsionIdx * 3 + 0];
    real k = torsionParams[torsionIdx * 3 + 1];
    real phi0 = torsionParams[torsionIdx * 3 + 2];

    // Compute torsion Hessian
    real localHess[144];
    computeTorsionHessian(p1, p2, p3, p4, k, n, phi0, localHess);

    // Map local indices (0-3) to global atom indices
    int atomIndices[4] = {i1, i2, i3, i4};

    // Accumulate into global Hessian using fixed-point for determinism
    int stride = numAtoms * 3;
    for (int localI = 0; localI < 4; localI++) {
        for (int localJ = 0; localJ < 4; localJ++) {
            int globalI = atomIndices[localI];
            int globalJ = atomIndices[localJ];

            for (int di = 0; di < 3; di++) {
                for (int dj = 0; dj < 3; dj++) {
                    int globalRow = globalI * 3 + di;
                    int globalCol = globalJ * 3 + dj;
                    int localRow = localI * 3 + di;
                    int localCol = localJ * 3 + dj;

                    real val = localHess[localRow * 12 + localCol];
                    atomicAdd(&globalHessian[globalRow * stride + globalCol],
                              static_cast<unsigned long long>(static_cast<long long>(val * HESSIAN_SCALE)));
                }
            }
        }
    }
}

/**
 * Alternative kernel that outputs per-torsion Hessian blocks instead of
 * accumulating into a global matrix. This is useful for:
 *   1. Debugging/validation
 *   2. Systems where atomic adds are expensive
 *   3. Sparse Hessian storage
 *
 * @param posq             Atom positions
 * @param torsionAtoms     Torsion atom indices
 * @param torsionParams    Torsion parameters
 * @param torsionHessians  Output: [numTorsions * 144] Hessian blocks
 * @param numTorsions      Number of torsions
 */
/**
 * Debug kernel that outputs ONLY the dihedral angle Hessian (H_phi) for inspection.
 */
extern "C" __global__ void computeDihedralHessianBlocks(
    const real4* __restrict__ posq,
    const int* __restrict__ torsionAtoms,
    float* __restrict__ dihedralHessians,  // [numTorsions * 144]
    int numTorsions
) {
    int torsionIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (torsionIdx >= numTorsions) return;

    int i1 = torsionAtoms[torsionIdx * 4 + 0];
    int i2 = torsionAtoms[torsionIdx * 4 + 1];
    int i3 = torsionAtoms[torsionIdx * 4 + 2];
    int i4 = torsionAtoms[torsionIdx * 4 + 3];

    real4 pos1 = posq[i1];
    real4 pos2 = posq[i2];
    real4 pos3 = posq[i3];
    real4 pos4 = posq[i4];

    real3 p1 = make_real3(pos1.x, pos1.y, pos1.z);
    real3 p2 = make_real3(pos2.x, pos2.y, pos2.z);
    real3 p3 = make_real3(pos3.x, pos3.y, pos3.z);
    real3 p4 = make_real3(pos4.x, pos4.y, pos4.z);

    // Output H_phi directly
    real localHess[144];
    computeDihedralHessian(p1, p2, p3, p4, localHess);
    for (int i = 0; i < 144; i++)
        dihedralHessians[torsionIdx * 144 + i] = (float)localHess[i];
}

extern "C" __global__ void computeTorsionHessianBlocks(
    const real4* __restrict__ posq,
    const int* __restrict__ torsionAtoms,
    const float* __restrict__ torsionParams,
    float* __restrict__ torsionHessians,  // [numTorsions * 144]
    int numTorsions
) {
    int torsionIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (torsionIdx >= numTorsions) return;

    // Load atom indices
    int i1 = torsionAtoms[torsionIdx * 4 + 0];
    int i2 = torsionAtoms[torsionIdx * 4 + 1];
    int i3 = torsionAtoms[torsionIdx * 4 + 2];
    int i4 = torsionAtoms[torsionIdx * 4 + 3];

    // Load positions
    real4 pos1 = posq[i1];
    real4 pos2 = posq[i2];
    real4 pos3 = posq[i3];
    real4 pos4 = posq[i4];

    real3 p1 = make_real3(pos1.x, pos1.y, pos1.z);
    real3 p2 = make_real3(pos2.x, pos2.y, pos2.z);
    real3 p3 = make_real3(pos3.x, pos3.y, pos3.z);
    real3 p4 = make_real3(pos4.x, pos4.y, pos4.z);

    // Load parameters
    int n = (int)torsionParams[torsionIdx * 3 + 0];
    real k = torsionParams[torsionIdx * 3 + 1];
    real phi0 = torsionParams[torsionIdx * 3 + 2];

    // Compute torsion Hessian
    real localHess[144];
    computeTorsionHessian(p1, p2, p3, p4, k, n, phi0, localHess);
    for (int i = 0; i < 144; i++)
        torsionHessians[torsionIdx * 144 + i] = (float)localHess[i];
}

// ============================================================
// Nonbonded Pair Hessian Kernel
// ============================================================

/**
 * Compute analytical Hessian for nonbonded pair interaction:
 *   E = 4*eps*((sigma/r)^12 - (sigma/r)^6) + q1*q2*ONE_4PI_EPS0/r
 *
 * For a radial potential V(r):
 *   dV/dri = dV/dr * r_hat  (for atom 1, negative for atom 2)
 *   d²V/dri drj = d²V/dr² * r_hat ⊗ r_hat + (dV/dr)/r * (I - r_hat ⊗ r_hat)
 *
 * @param p1, p2      Atom positions
 * @param q1, q2      Atomic charges (elementary charges)
 * @param sigma       LJ sigma parameter (nm)
 * @param epsilon     LJ epsilon parameter (kJ/mol)
 * @param hess        Output: 6x6 Hessian matrix (row-major)
 */
__device__ void computeNonbondedPairHessian(
    real3 p1, real3 p2,
    real q1, real q2,
    real sigma, real epsilon,
    real* hess  // 36 entries
) {
    // Initialize to zero
    for (int i = 0; i < 36; i++) hess[i] = 0.0f;

    real3 r_vec = sub(p2, p1);
    real r2 = length_sq(r_vec);
    real r = sqrt(r2);

    if (r < 1e-10f) return;

    real3 r_hat = scale(r_vec, 1.0f / r);

    // LJ potential: V_LJ = 4*eps*((sigma/r)^12 - (sigma/r)^6)
    // Let x = sigma/r, then V_LJ = 4*eps*(x^12 - x^6)
    real x = sigma / r;
    real x2 = x * x;
    real x6 = x2 * x2 * x2;
    real x12 = x6 * x6;

    // dV_LJ/dr = 4*eps*(-12*sigma^12/r^13 + 6*sigma^6/r^7)
    //          = 4*eps*(6*x^6/r - 12*x^12/r)
    //          = (24*eps/r)*(x^6 - 2*x^12)
    real dV_LJ_dr = (24.0f * epsilon / r) * (x6 - 2.0f * x12);

    // d²V_LJ/dr² = 4*eps*(12*13*sigma^12/r^14 - 6*7*sigma^6/r^8)
    //            = 4*eps*(156*x^12/r^2 - 42*x^6/r^2)
    //            = (24*eps/r^2)*(13*x^12 - 7*x^6/2) ... let me redo this
    // Actually: d/dr[(24*eps/r)*(x^6 - 2*x^12)]
    //   = 24*eps*[(-1/r^2)*(x^6 - 2*x^12) + (1/r)*(6*x^5*(-sigma/r^2) - 12*x^11*(-sigma/r^2))]
    //   = 24*eps*[(-1/r^2)*(x^6 - 2*x^12) + (sigma/r^3)*(12*x^11 - 6*x^5)]
    //   = 24*eps/r^2 * [-(x^6 - 2*x^12) + (12*x^12 - 6*x^6)]
    //   = 24*eps/r^2 * [-x^6 + 2*x^12 + 12*x^12 - 6*x^6]
    //   = 24*eps/r^2 * [14*x^12 - 7*x^6]
    //   = (24*eps/r^2) * 7 * (2*x^12 - x^6)
    real d2V_LJ_dr2 = (24.0f * epsilon / r2) * 7.0f * (2.0f * x12 - x6);

    // Coulomb potential: V_C = q1*q2*ONE_4PI_EPS0/r
    // dV_C/dr = -q1*q2*ONE_4PI_EPS0/r^2
    // d²V_C/dr² = 2*q1*q2*ONE_4PI_EPS0/r^3
    real qq = q1 * q2 * ONE_4PI_EPS0;
    real dV_C_dr = -qq / r2;
    real d2V_C_dr2 = 2.0f * qq / (r2 * r);

    // Total derivatives
    real dV_dr = dV_LJ_dr + dV_C_dr;
    real d2V_dr2 = d2V_LJ_dr2 + d2V_C_dr2;

    // Outer product r_hat ⊗ r_hat
    real rr[9];
    real rv[3] = {r_hat.x, r_hat.y, r_hat.z};
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            rr[i*3+j] = rv[i] * rv[j];
        }
    }

    // d²V/dr1² = d²V/dr² * rr + (dV/dr)/r * (I - rr)
    real dV_dr_over_r = dV_dr / r;
    real block[9];
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            real I_ij = (i == j) ? 1.0f : 0.0f;
            block[i*3+j] = d2V_dr2 * rr[i*3+j] + dV_dr_over_r * (I_ij - rr[i*3+j]);
        }
    }

    // Fill 6x6 Hessian (same structure as bond Hessian)
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            hess[i*6 + j] = block[i*3+j];           // d²V/dr1²
            hess[(i+3)*6 + (j+3)] = block[i*3+j];   // d²V/dr2²
            hess[i*6 + (j+3)] = -block[i*3+j];      // d²V/dr1dr2
            hess[(i+3)*6 + j] = -block[i*3+j];      // d²V/dr2dr1
        }
    }
}

/**
 * Kernel to compute nonbonded pair Hessians for all pairs in the system.
 *
 * This handles 1-4 interactions and other exception pairs from NonbondedForce.
 */
extern "C" __global__ void computeNonbondedPairHessians(
    const real4* __restrict__ posq,           // positions and charges
    const int* __restrict__ pairAtoms,         // [numPairs * 2]: atom indices
    const float* __restrict__ pairParams,      // [numPairs * 2]: sigma, epsilon
    unsigned long long* __restrict__ globalHessian,
    int numPairs,
    int numAtoms
) {
    int pairIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pairIdx >= numPairs) return;

    int i1 = pairAtoms[pairIdx * 2 + 0];
    int i2 = pairAtoms[pairIdx * 2 + 1];

    real4 pos1 = posq[i1];
    real4 pos2 = posq[i2];

    real3 p1 = make_real3(pos1.x, pos1.y, pos1.z);
    real3 p2 = make_real3(pos2.x, pos2.y, pos2.z);
    real q1 = pos1.w;  // charge stored in w component
    real q2 = pos2.w;

    real sigma = pairParams[pairIdx * 2 + 0];
    real epsilon = pairParams[pairIdx * 2 + 1];

    real localHess[36];
    computeNonbondedPairHessian(p1, p2, q1, q2, sigma, epsilon, localHess);

    // Accumulate into global Hessian using fixed-point for determinism
    int atomIndices[2] = {i1, i2};
    int stride = numAtoms * 3;

    for (int localI = 0; localI < 2; localI++) {
        for (int localJ = 0; localJ < 2; localJ++) {
            int globalI = atomIndices[localI];
            int globalJ = atomIndices[localJ];

            for (int di = 0; di < 3; di++) {
                for (int dj = 0; dj < 3; dj++) {
                    int globalRow = globalI * 3 + di;
                    int globalCol = globalJ * 3 + dj;
                    int localRow = localI * 3 + di;
                    int localCol = localJ * 3 + dj;

                    real val = localHess[localRow * 6 + localCol];
                    atomicAdd(&globalHessian[globalRow * stride + globalCol],
                              static_cast<unsigned long long>(static_cast<long long>(val * HESSIAN_SCALE)));
                }
            }
        }
    }
}

// Note: computeIsolatedNonbondedHessians kernel is defined in isolatedNonbonded.cu
// to include both LJ and Coulomb interactions

// ============================================================
// Utility Kernels
// ============================================================

/**
 * Initialize the global Hessian matrix to zero.
 * Uses unsigned long long for fixed-point deterministic accumulation.
 */
extern "C" __global__ void initializeHessian(
    unsigned long long* __restrict__ hessian,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        hessian[idx] = 0ULL;
    }
}
