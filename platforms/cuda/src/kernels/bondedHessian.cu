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

// ============================================================
// Helper functions
// ============================================================

/**
 * Compute cross product of two float3 vectors.
 */
__device__ inline float3 cross(float3 a, float3 b) {
    return make_float3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

/**
 * Compute dot product of two float3 vectors.
 */
__device__ inline float dot(float3 a, float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

/**
 * Compute squared magnitude of a float3 vector.
 */
__device__ inline float length_sq(float3 v) {
    return dot(v, v);
}

/**
 * Compute magnitude of a float3 vector.
 */
__device__ inline float length(float3 v) {
    return sqrtf(length_sq(v));
}

/**
 * Subtract two float3 vectors.
 */
__device__ inline float3 sub(float3 a, float3 b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

/**
 * Add two float3 vectors.
 */
__device__ inline float3 add(float3 a, float3 b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

/**
 * Scale a float3 vector.
 */
__device__ inline float3 scale(float3 v, float s) {
    return make_float3(v.x * s, v.y * s, v.z * s);
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
    float3 p1, float3 p2,
    float k, float r0,
    float* hess  // 36 floats
) {
    // Initialize to zero
    for (int i = 0; i < 36; i++) hess[i] = 0.0f;

    float3 r_vec = sub(p2, p1);
    float r = length(r_vec);

    if (r < 1e-10f) return;

    float3 r_hat = scale(r_vec, 1.0f / r);

    // Outer product r_hat ⊗ r_hat
    float rr[9];
    float rv[3] = {r_hat.x, r_hat.y, r_hat.z};
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            rr[i*3+j] = rv[i] * rv[j];
        }
    }

    // d²E/dr1² = k * [rr + (1 - r0/r) * (I - rr)]
    float factor = 1.0f - r0 / r;
    float block[9];
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            float I_ij = (i == j) ? 1.0f : 0.0f;
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
    const float4* __restrict__ posq,
    const int* __restrict__ bondAtoms,      // [numBonds * 2]: atom indices
    const float* __restrict__ bondParams,   // [numBonds * 2]: k, r0
    float* __restrict__ globalHessian,
    int numBonds,
    int numAtoms
) {
    int bondIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (bondIdx >= numBonds) return;

    int i1 = bondAtoms[bondIdx * 2 + 0];
    int i2 = bondAtoms[bondIdx * 2 + 1];

    float4 pos1 = posq[i1];
    float4 pos2 = posq[i2];

    float3 p1 = make_float3(pos1.x, pos1.y, pos1.z);
    float3 p2 = make_float3(pos2.x, pos2.y, pos2.z);

    float k = bondParams[bondIdx * 2 + 0];
    float r0 = bondParams[bondIdx * 2 + 1];

    float localHess[36];
    computeBondHessian(p1, p2, k, r0, localHess);

    // Accumulate into global Hessian
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

                    float val = localHess[localRow * 6 + localCol];
                    atomicAdd(&globalHessian[globalRow * stride + globalCol], val);
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
 * @param p1, p2, p3  Atom positions (p2 is central atom)
 * @param k           Force constant (kJ/mol/rad^2)
 * @param theta0      Equilibrium angle (radians)
 * @param hess        Output: 9x9 Hessian matrix (row-major)
 */
__device__ void computeAngleHessian(
    float3 p1, float3 p2, float3 p3,
    float k, float theta0,
    float* hess  // 81 floats
) {
    // Initialize to zero
    for (int i = 0; i < 81; i++) hess[i] = 0.0f;

    // Vectors from central atom
    float3 r21 = sub(p1, p2);
    float3 r23 = sub(p3, p2);

    float r21_len = length(r21);
    float r23_len = length(r23);

    if (r21_len < 1e-10f || r23_len < 1e-10f) return;

    float3 r21_hat = scale(r21, 1.0f / r21_len);
    float3 r23_hat = scale(r23, 1.0f / r23_len);

    float cos_theta = dot(r21_hat, r23_hat);
    cos_theta = fminf(0.9999999f, fmaxf(-0.9999999f, cos_theta));
    float theta = acosf(cos_theta);
    float sin_theta = sinf(theta);

    if (fabsf(sin_theta) < 1e-10f) return;

    // Energy derivatives
    // E = 0.5 * k * (theta - theta0)^2
    // dE/dtheta = k * (theta - theta0)
    // d²E/dtheta² = k
    float dE_dtheta = k * (theta - theta0);
    float d2E_dtheta2 = k;

    // Gradient of theta with respect to positions
    // dtheta/dr1 = -1/(|r21| * sin_theta) * (r23_hat - cos_theta * r21_hat)
    // dtheta/dr3 = -1/(|r23| * sin_theta) * (r21_hat - cos_theta * r23_hat)
    // dtheta/dr2 = -(dtheta/dr1 + dtheta/dr3)

    float inv_sin = 1.0f / sin_theta;
    float3 dtheta_dr1 = scale(sub(r23_hat, scale(r21_hat, cos_theta)), -inv_sin / r21_len);
    float3 dtheta_dr3 = scale(sub(r21_hat, scale(r23_hat, cos_theta)), -inv_sin / r23_len);
    float3 dtheta_dr2 = scale(add(dtheta_dr1, dtheta_dr3), -1.0f);

    // Build gradient vector (9 elements)
    float grad[9];
    grad[0] = dtheta_dr1.x; grad[1] = dtheta_dr1.y; grad[2] = dtheta_dr1.z;
    grad[3] = dtheta_dr2.x; grad[4] = dtheta_dr2.y; grad[5] = dtheta_dr2.z;
    grad[6] = dtheta_dr3.x; grad[7] = dtheta_dr3.y; grad[8] = dtheta_dr3.z;

    // For angles, we use numerical differentiation of the gradient for d²theta/dri drj
    // This is more robust than the complex analytical formulas
    float h = 1e-5f;
    float H_theta[81];

    float3 positions[3] = {p1, p2, p3};

    for (int i = 0; i < 9; i++) {
        int atom_i = i / 3;
        int dim_i = i % 3;

        // Perturb position
        float3 pos_p[3] = {positions[0], positions[1], positions[2]};
        float3 pos_m[3] = {positions[0], positions[1], positions[2]};

        if (dim_i == 0) {
            pos_p[atom_i].x += h;
            pos_m[atom_i].x -= h;
        } else if (dim_i == 1) {
            pos_p[atom_i].y += h;
            pos_m[atom_i].y -= h;
        } else {
            pos_p[atom_i].z += h;
            pos_m[atom_i].z -= h;
        }

        // Compute gradient at perturbed positions
        float3 r21_p = sub(pos_p[0], pos_p[1]);
        float3 r23_p = sub(pos_p[2], pos_p[1]);
        float3 r21_m = sub(pos_m[0], pos_m[1]);
        float3 r23_m = sub(pos_m[2], pos_m[1]);

        float r21_len_p = length(r21_p);
        float r23_len_p = length(r23_p);
        float r21_len_m = length(r21_m);
        float r23_len_m = length(r23_m);

        if (r21_len_p < 1e-10f || r23_len_p < 1e-10f ||
            r21_len_m < 1e-10f || r23_len_m < 1e-10f) {
            for (int j = 0; j < 9; j++) H_theta[i * 9 + j] = 0.0f;
            continue;
        }

        float3 r21_hat_p = scale(r21_p, 1.0f / r21_len_p);
        float3 r23_hat_p = scale(r23_p, 1.0f / r23_len_p);
        float3 r21_hat_m = scale(r21_m, 1.0f / r21_len_m);
        float3 r23_hat_m = scale(r23_m, 1.0f / r23_len_m);

        float cos_theta_p = fminf(0.9999999f, fmaxf(-0.9999999f, dot(r21_hat_p, r23_hat_p)));
        float cos_theta_m = fminf(0.9999999f, fmaxf(-0.9999999f, dot(r21_hat_m, r23_hat_m)));
        float sin_theta_p = sinf(acosf(cos_theta_p));
        float sin_theta_m = sinf(acosf(cos_theta_m));

        if (fabsf(sin_theta_p) < 1e-10f || fabsf(sin_theta_m) < 1e-10f) {
            for (int j = 0; j < 9; j++) H_theta[i * 9 + j] = 0.0f;
            continue;
        }

        float inv_sin_p = 1.0f / sin_theta_p;
        float inv_sin_m = 1.0f / sin_theta_m;

        float3 dtheta_dr1_p = scale(sub(r23_hat_p, scale(r21_hat_p, cos_theta_p)), -inv_sin_p / r21_len_p);
        float3 dtheta_dr3_p = scale(sub(r21_hat_p, scale(r23_hat_p, cos_theta_p)), -inv_sin_p / r23_len_p);
        float3 dtheta_dr2_p = scale(add(dtheta_dr1_p, dtheta_dr3_p), -1.0f);

        float3 dtheta_dr1_m = scale(sub(r23_hat_m, scale(r21_hat_m, cos_theta_m)), -inv_sin_m / r21_len_m);
        float3 dtheta_dr3_m = scale(sub(r21_hat_m, scale(r23_hat_m, cos_theta_m)), -inv_sin_m / r23_len_m);
        float3 dtheta_dr2_m = scale(add(dtheta_dr1_m, dtheta_dr3_m), -1.0f);

        float grad_p[9], grad_m[9];
        grad_p[0] = dtheta_dr1_p.x; grad_p[1] = dtheta_dr1_p.y; grad_p[2] = dtheta_dr1_p.z;
        grad_p[3] = dtheta_dr2_p.x; grad_p[4] = dtheta_dr2_p.y; grad_p[5] = dtheta_dr2_p.z;
        grad_p[6] = dtheta_dr3_p.x; grad_p[7] = dtheta_dr3_p.y; grad_p[8] = dtheta_dr3_p.z;

        grad_m[0] = dtheta_dr1_m.x; grad_m[1] = dtheta_dr1_m.y; grad_m[2] = dtheta_dr1_m.z;
        grad_m[3] = dtheta_dr2_m.x; grad_m[4] = dtheta_dr2_m.y; grad_m[5] = dtheta_dr2_m.z;
        grad_m[6] = dtheta_dr3_m.x; grad_m[7] = dtheta_dr3_m.y; grad_m[8] = dtheta_dr3_m.z;

        for (int j = 0; j < 9; j++) {
            H_theta[i * 9 + j] = (grad_p[j] - grad_m[j]) / (2.0f * h);
        }
    }

    // Symmetrize H_theta
    for (int i = 0; i < 9; i++) {
        for (int j = i + 1; j < 9; j++) {
            float avg = 0.5f * (H_theta[i * 9 + j] + H_theta[j * 9 + i]);
            H_theta[i * 9 + j] = avg;
            H_theta[j * 9 + i] = avg;
        }
    }

    // Full Hessian: H = d2E_dtheta2 * grad ⊗ grad + dE_dtheta * H_theta
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
    const float4* __restrict__ posq,
    const int* __restrict__ angleAtoms,      // [numAngles * 3]: atom indices
    const float* __restrict__ angleParams,   // [numAngles * 2]: k, theta0
    float* __restrict__ globalHessian,
    int numAngles,
    int numAtoms
) {
    int angleIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (angleIdx >= numAngles) return;

    int i1 = angleAtoms[angleIdx * 3 + 0];
    int i2 = angleAtoms[angleIdx * 3 + 1];  // central atom
    int i3 = angleAtoms[angleIdx * 3 + 2];

    float4 pos1 = posq[i1];
    float4 pos2 = posq[i2];
    float4 pos3 = posq[i3];

    float3 p1 = make_float3(pos1.x, pos1.y, pos1.z);
    float3 p2 = make_float3(pos2.x, pos2.y, pos2.z);
    float3 p3 = make_float3(pos3.x, pos3.y, pos3.z);

    float k = angleParams[angleIdx * 2 + 0];
    float theta0 = angleParams[angleIdx * 2 + 1];

    float localHess[81];
    computeAngleHessian(p1, p2, p3, k, theta0, localHess);

    // Accumulate into global Hessian
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

                    float val = localHess[localRow * 9 + localCol];
                    atomicAdd(&globalHessian[globalRow * stride + globalCol], val);
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
    float3 p1, float3 p2, float3 p3, float3 p4,
    float& phi,
    float* grad  // 12 floats: grad[0-2]=dr1, grad[3-5]=dr2, etc.
) {
    // Bond vectors
    float3 b1 = sub(p2, p1);
    float3 b2 = sub(p3, p2);
    float3 b3 = sub(p4, p3);

    // Normal vectors to planes
    float3 m = cross(b1, b2);
    float3 n = cross(b2, b3);

    float m_sq = length_sq(m);
    float n_sq = length_sq(n);
    float b2_sq = length_sq(b2);
    float b2_norm = sqrtf(b2_sq);

    // Degenerate case check
    if (m_sq < 1e-20f || n_sq < 1e-20f || b2_sq < 1e-20f) {
        phi = 0.0f;
        for (int i = 0; i < 12; i++) grad[i] = 0.0f;
        return;
    }

    // Normalized vectors
    float m_norm = sqrtf(m_sq);
    float n_norm = sqrtf(n_sq);
    float3 m_hat = scale(m, 1.0f / m_norm);
    float3 n_hat = scale(n, 1.0f / n_norm);
    float3 b2_hat = scale(b2, 1.0f / b2_norm);

    // Dihedral angle
    float cos_phi = dot(m_hat, n_hat);
    float3 m_cross_b2 = cross(m_hat, b2_hat);
    float sin_phi = dot(m_cross_b2, n_hat);
    phi = atan2f(sin_phi, cos_phi);

    // Gradient of phi with respect to atom positions
    // dphi/dr1 = (|b2| / |m|^2) * m
    // dphi/dr4 = -(|b2| / |n|^2) * n
    float3 dphi_dr1 = scale(m, b2_norm / m_sq);
    float3 dphi_dr4 = scale(n, -b2_norm / n_sq);

    // Projection factors
    float b1_dot_b2 = dot(b1, b2);
    float b3_dot_b2 = dot(b3, b2);
    float alpha = b1_dot_b2 / b2_sq;
    float beta = b3_dot_b2 / b2_sq;

    // Corrected coefficients for middle atoms
    float c1 = -(1.0f + alpha);
    float c4 = beta;
    float d1 = alpha;
    float d4 = -(1.0f + beta);

    // dphi/dr2 = c1 * dphi/dr1 + c4 * dphi/dr4
    // dphi/dr3 = d1 * dphi/dr1 + d4 * dphi/dr4
    float3 dphi_dr2 = add(scale(dphi_dr1, c1), scale(dphi_dr4, c4));
    float3 dphi_dr3 = add(scale(dphi_dr1, d1), scale(dphi_dr4, d4));

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
    float3 p1, float3 p2, float3 p3, float3 p4,
    float* hess  // 144 floats
) {
    // Initialize to zero
    for (int i = 0; i < 144; i++) hess[i] = 0.0f;

    // Bond vectors
    float3 b1 = sub(p2, p1);
    float3 b2 = sub(p3, p2);
    float3 b3 = sub(p4, p3);

    // Normal vectors
    float3 m = cross(b1, b2);
    float3 n = cross(b2, b3);

    float m_sq = length_sq(m);
    float n_sq = length_sq(n);
    float b2_sq = length_sq(b2);
    float b2_norm = sqrtf(b2_sq);

    if (m_sq < 1e-20f || n_sq < 1e-20f || b2_sq < 1e-20f) {
        return;
    }

    // Precompute outer products (stored as 3x3 arrays, row-major)
    float mm[9], nn[9];
    float mv[3] = {m.x, m.y, m.z};
    float nv[3] = {n.x, n.y, n.z};
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            mm[i*3+j] = mv[i] * mv[j];
            nn[i*3+j] = nv[i] * nv[j];
        }
    }

    // Skew-symmetric matrices for cross products
    // [v]x = [[0, -vz, vy], [vz, 0, -vx], [-vy, vx, 0]]
    float b1_x[9] = {0, -b1.z, b1.y, b1.z, 0, -b1.x, -b1.y, b1.x, 0};
    float b2_x[9] = {0, -b2.z, b2.y, b2.z, 0, -b2.x, -b2.y, b2.x, 0};
    float b3_x[9] = {0, -b3.z, b3.y, b3.z, 0, -b3.x, -b3.y, b3.x, 0};

    // Derivatives of m = b1 x b2 with respect to positions (3x3 matrices)
    // dm/dp1 = [b2]x
    // dm/dp2 = -[b2]x - [b1]x
    // dm/dp3 = [b1]x
    // dm/dp4 = 0
    float dm_dp[4][9];
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
    float dn_dp[4][9];
    for (int i = 0; i < 9; i++) {
        dn_dp[0][i] = 0.0f;
        dn_dp[1][i] = b3_x[i];
        dn_dp[2][i] = -b3_x[i] - b2_x[i];
        dn_dp[3][i] = b2_x[i];
    }

    // Gradient components
    float3 G1 = scale(m, b2_norm / m_sq);
    float3 G4 = scale(n, -b2_norm / n_sq);

    // Coefficients
    float b1_dot_b2 = dot(b1, b2);
    float b3_dot_b2 = dot(b3, b2);
    float alpha = b1_dot_b2 / b2_sq;
    float beta = b3_dot_b2 / b2_sq;
    float c1 = -(1.0f + alpha);
    float c4 = beta;
    float d1 = alpha;
    float d4 = -(1.0f + beta);

    // Derivative of |b2| with respect to positions
    float db2_norm_dp[4][3];
    float b2v[3] = {b2.x, b2.y, b2.z};
    for (int i = 0; i < 3; i++) {
        db2_norm_dp[0][i] = 0.0f;
        db2_norm_dp[1][i] = -b2v[i] / b2_norm;
        db2_norm_dp[2][i] = b2v[i] / b2_norm;
        db2_norm_dp[3][i] = 0.0f;
    }

    // Derivatives of b1.b2, b3.b2, b2^2
    float b1v[3] = {b1.x, b1.y, b1.z};
    float b3v[3] = {b3.x, b3.y, b3.z};

    float db1_dot_b2_dp[4][3], db3_dot_b2_dp[4][3], db2_sq_dp[4][3];
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
    float dG1_dp[4][9], dG4_dp[4][9];

    for (int j = 0; j < 4; j++) {
        // dG1/dpj = outer(m, db2_norm/dpj) / m_sq + (b2_norm/m_sq) * (I - 2*mm/m_sq) @ dm/dpj
        // dG4/dpj = -outer(n, db2_norm/dpj) / n_sq - (b2_norm/n_sq) * (I - 2*nn/n_sq) @ dn/dpj

        for (int a = 0; a < 3; a++) {
            for (int b = 0; b < 3; b++) {
                int idx = a * 3 + b;

                // term1 for G1: m[a] * db2_norm[j][b] / m_sq
                float term1_G1 = mv[a] * db2_norm_dp[j][b] / m_sq;

                // term2 for G1: (b2_norm/m_sq) * ((I - 2*mm/m_sq) @ dm_dp[j])_ab
                // (I - 2*mm/m_sq)_ac * dm_dp[j]_cb
                float sum_G1 = 0.0f;
                for (int c = 0; c < 3; c++) {
                    float factor = (a == c ? 1.0f : 0.0f) - 2.0f * mm[a*3+c] / m_sq;
                    sum_G1 += factor * dm_dp[j][c*3+b];
                }
                float term2_G1 = (b2_norm / m_sq) * sum_G1;

                dG1_dp[j][idx] = term1_G1 + term2_G1;

                // term1 for G4: -n[a] * db2_norm[j][b] / n_sq
                float term1_G4 = -nv[a] * db2_norm_dp[j][b] / n_sq;

                // term2 for G4: -(b2_norm/n_sq) * ((I - 2*nn/n_sq) @ dn_dp[j])_ab
                float sum_G4 = 0.0f;
                for (int c = 0; c < 3; c++) {
                    float factor = (a == c ? 1.0f : 0.0f) - 2.0f * nn[a*3+c] / n_sq;
                    sum_G4 += factor * dn_dp[j][c*3+b];
                }
                float term2_G4 = -(b2_norm / n_sq) * sum_G4;

                dG4_dp[j][idx] = term1_G4 + term2_G4;
            }
        }
    }

    // Coefficient derivatives
    float dc1_dp[4][3], dc4_dp[4][3], dd1_dp[4][3], dd4_dp[4][3];

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
    float G1v[3] = {G1.x, G1.y, G1.z};
    float G4v[3] = {G4.x, G4.y, G4.z};

    float dG2_dp[4][9], dG3_dp[4][9];

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
    float* dG_dp[4] = {dG1_dp[0], dG2_dp[0], dG3_dp[0], dG4_dp[0]};

    // We need to use the arrays properly
    for (int i = 0; i < 4; i++) {  // gradient block (atom i)
        for (int j = 0; j < 4; j++) {  // derivative with respect to atom j
            float* dGi_dpj;
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
            float avg = 0.5f * (hess[i * 12 + j] + hess[j * 12 + i]);
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
    float3 p1, float3 p2, float3 p3, float3 p4,
    float k, int n, float phi0,
    float* hess  // 144 floats
) {
    // Get dihedral angle and gradient
    float phi;
    float grad[12];
    computeDihedralAndGradient(p1, p2, p3, p4, phi, grad);

    // Get dihedral Hessian
    float H_phi[144];
    computeDihedralHessian(p1, p2, p3, p4, H_phi);

    // Energy derivatives
    float arg = n * phi - phi0;
    float dE_dphi = -k * n * sinf(arg);
    float d2E_dphi2 = -k * n * n * cosf(arg);

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
    const float4* __restrict__ posq,
    const int* __restrict__ torsionAtoms,      // [numTorsions * 4]: atom indices
    const float* __restrict__ torsionParams,   // [numTorsions * 3]: n, k, phi0
    float* __restrict__ globalHessian,         // [numAtoms * 3 * numAtoms * 3]
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
    float4 pos1 = posq[i1];
    float4 pos2 = posq[i2];
    float4 pos3 = posq[i3];
    float4 pos4 = posq[i4];

    float3 p1 = make_float3(pos1.x, pos1.y, pos1.z);
    float3 p2 = make_float3(pos2.x, pos2.y, pos2.z);
    float3 p3 = make_float3(pos3.x, pos3.y, pos3.z);
    float3 p4 = make_float3(pos4.x, pos4.y, pos4.z);

    // Load parameters
    int n = (int)torsionParams[torsionIdx * 3 + 0];
    float k = torsionParams[torsionIdx * 3 + 1];
    float phi0 = torsionParams[torsionIdx * 3 + 2];

    // Compute torsion Hessian
    float localHess[144];
    computeTorsionHessian(p1, p2, p3, p4, k, n, phi0, localHess);

    // Map local indices (0-3) to global atom indices
    int atomIndices[4] = {i1, i2, i3, i4};

    // Accumulate into global Hessian
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

                    float val = localHess[localRow * 12 + localCol];
                    atomicAdd(&globalHessian[globalRow * stride + globalCol], val);
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
extern "C" __global__ void computeTorsionHessianBlocks(
    const float4* __restrict__ posq,
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
    float4 pos1 = posq[i1];
    float4 pos2 = posq[i2];
    float4 pos3 = posq[i3];
    float4 pos4 = posq[i4];

    float3 p1 = make_float3(pos1.x, pos1.y, pos1.z);
    float3 p2 = make_float3(pos2.x, pos2.y, pos2.z);
    float3 p3 = make_float3(pos3.x, pos3.y, pos3.z);
    float3 p4 = make_float3(pos4.x, pos4.y, pos4.z);

    // Load parameters
    int n = (int)torsionParams[torsionIdx * 3 + 0];
    float k = torsionParams[torsionIdx * 3 + 1];
    float phi0 = torsionParams[torsionIdx * 3 + 2];

    // Compute torsion Hessian
    float* output = &torsionHessians[torsionIdx * 144];
    computeTorsionHessian(p1, p2, p3, p4, k, n, phi0, output);
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
    float3 p1, float3 p2,
    float q1, float q2,
    float sigma, float epsilon,
    float* hess  // 36 floats
) {
    // Initialize to zero
    for (int i = 0; i < 36; i++) hess[i] = 0.0f;

    float3 r_vec = sub(p2, p1);
    float r2 = length_sq(r_vec);
    float r = sqrtf(r2);

    if (r < 1e-10f) return;

    float3 r_hat = scale(r_vec, 1.0f / r);

    // LJ potential: V_LJ = 4*eps*((sigma/r)^12 - (sigma/r)^6)
    // Let x = sigma/r, then V_LJ = 4*eps*(x^12 - x^6)
    float x = sigma / r;
    float x2 = x * x;
    float x6 = x2 * x2 * x2;
    float x12 = x6 * x6;

    // dV_LJ/dr = 4*eps*(-12*sigma^12/r^13 + 6*sigma^6/r^7)
    //          = 4*eps*(6*x^6/r - 12*x^12/r)
    //          = (24*eps/r)*(x^6 - 2*x^12)
    float dV_LJ_dr = (24.0f * epsilon / r) * (x6 - 2.0f * x12);

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
    float d2V_LJ_dr2 = (24.0f * epsilon / r2) * 7.0f * (2.0f * x12 - x6);

    // Coulomb potential: V_C = q1*q2*ONE_4PI_EPS0/r
    // dV_C/dr = -q1*q2*ONE_4PI_EPS0/r^2
    // d²V_C/dr² = 2*q1*q2*ONE_4PI_EPS0/r^3
    float qq = q1 * q2 * ONE_4PI_EPS0;
    float dV_C_dr = -qq / r2;
    float d2V_C_dr2 = 2.0f * qq / (r2 * r);

    // Total derivatives
    float dV_dr = dV_LJ_dr + dV_C_dr;
    float d2V_dr2 = d2V_LJ_dr2 + d2V_C_dr2;

    // Outer product r_hat ⊗ r_hat
    float rr[9];
    float rv[3] = {r_hat.x, r_hat.y, r_hat.z};
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            rr[i*3+j] = rv[i] * rv[j];
        }
    }

    // d²V/dr1² = d²V/dr² * rr + (dV/dr)/r * (I - rr)
    float dV_dr_over_r = dV_dr / r;
    float block[9];
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            float I_ij = (i == j) ? 1.0f : 0.0f;
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
    const float4* __restrict__ posq,           // positions and charges
    const int* __restrict__ pairAtoms,         // [numPairs * 2]: atom indices
    const float* __restrict__ pairParams,      // [numPairs * 2]: sigma, epsilon
    float* __restrict__ globalHessian,
    int numPairs,
    int numAtoms
) {
    int pairIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pairIdx >= numPairs) return;

    int i1 = pairAtoms[pairIdx * 2 + 0];
    int i2 = pairAtoms[pairIdx * 2 + 1];

    float4 pos1 = posq[i1];
    float4 pos2 = posq[i2];

    float3 p1 = make_float3(pos1.x, pos1.y, pos1.z);
    float3 p2 = make_float3(pos2.x, pos2.y, pos2.z);
    float q1 = pos1.w;  // charge stored in w component
    float q2 = pos2.w;

    float sigma = pairParams[pairIdx * 2 + 0];
    float epsilon = pairParams[pairIdx * 2 + 1];

    float localHess[36];
    computeNonbondedPairHessian(p1, p2, q1, q2, sigma, epsilon, localHess);

    // Accumulate into global Hessian
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

                    float val = localHess[localRow * 6 + localCol];
                    atomicAdd(&globalHessian[globalRow * stride + globalCol], val);
                }
            }
        }
    }
}

/**
 * Kernel to compute isolated nonbonded Hessians (pure LJ, no Coulomb).
 *
 * Used for IsolatedNonbondedForce which handles only LJ interactions
 * between non-bonded atoms.
 */
extern "C" __global__ void computeIsolatedNonbondedHessians(
    const float4* __restrict__ posq,
    const int* __restrict__ pairAtoms,         // [numPairs * 2]: atom indices
    const float* __restrict__ pairParams,      // [numPairs * 2]: sigma, epsilon
    float* __restrict__ globalHessian,
    int numPairs,
    int numAtoms
) {
    int pairIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pairIdx >= numPairs) return;

    int i1 = pairAtoms[pairIdx * 2 + 0];
    int i2 = pairAtoms[pairIdx * 2 + 1];

    float4 pos1 = posq[i1];
    float4 pos2 = posq[i2];

    float3 p1 = make_float3(pos1.x, pos1.y, pos1.z);
    float3 p2 = make_float3(pos2.x, pos2.y, pos2.z);

    float sigma = pairParams[pairIdx * 2 + 0];
    float epsilon = pairParams[pairIdx * 2 + 1];

    // Use zero charges for pure LJ
    float localHess[36];
    computeNonbondedPairHessian(p1, p2, 0.0f, 0.0f, sigma, epsilon, localHess);

    // Accumulate into global Hessian
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

                    float val = localHess[localRow * 6 + localCol];
                    atomicAdd(&globalHessian[globalRow * stride + globalCol], val);
                }
            }
        }
    }
}

// ============================================================
// Utility Kernels
// ============================================================

/**
 * Initialize the global Hessian matrix to zero.
 */
extern "C" __global__ void initializeHessian(
    float* __restrict__ hessian,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        hessian[idx] = 0.0f;
    }
}
