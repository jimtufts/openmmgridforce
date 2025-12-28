#ifndef OPENMM_GRIDFORCE_LJ_ANALYTICAL_DERIVATIVES_H_
#define OPENMM_GRIDFORCE_LJ_ANALYTICAL_DERIVATIVES_H_

/**
 * Analytical radial derivatives for Lennard-Jones potential.
 * Based on RASPA3 implementation (MIT license).
 * 
 * These functions compute radial derivatives (dU/dr, d²U/dr², etc.) for LJ potentials.
 * To generate grid derivatives, these radial derivatives must be combined with
 * tensor conversion formulas using the displacement vector components.
 */

/**
 * Analytical radial derivatives for Lennard-Jones potential.
 * Based on RASPA3 implementation (MIT license).
 * Computes derivatives up to 6th order with respect to radial distance r.
 * Returns: derivs[0]=U, derivs[1]=dU/dr, derivs[2]=d2U/dr2, ..., derivs[6]=d6U/dr6
 */
__device__ inline void computeLJRadialDerivatives(
    float r2,
    float epsilon,
    float sigma,
    float shift,
    float* derivs
) {
    float arg1 = 4.0f * epsilon;
    float arg2 = sigma * sigma;
    float temp3 = (arg2 / r2) * (arg2 / r2) * (arg2 / r2);

    float r4 = r2 * r2;
    float r6 = r4 * r2;
    float r8 = r6 * r2;
    float r10 = r8 * r2;
    float r12 = r10 * r2;

    derivs[0] = arg1 * (temp3 * temp3 - temp3) - shift;
    derivs[1] = -6.0f * arg1 * (2.0f * temp3 * temp3 - temp3) / r2;
    derivs[2] = 24.0f * arg1 * (7.0f * temp3 * temp3 - 2.0f * temp3) / r4;
    derivs[3] = -96.0f * arg1 * (28.0f * temp3 * temp3 - 5.0f * temp3) / r6;
    derivs[4] = 1152.0f * arg1 * (42.0f * temp3 * temp3 - 5.0f * temp3) / r8;
    derivs[5] = -80640.0f * arg1 * (12.0f * temp3 * temp3 - temp3) / r10;
    derivs[6] = 645120.0f * arg1 * (33.0f * temp3 * temp3 - 2.0f * temp3) / r12;
}

__device__ inline void computeLJRepulsionRadialDerivatives(
    float r2,
    float epsilon,
    float sigma,
    float cutoff2,
    float* derivs
) {
    float arg1 = 4.0f * epsilon;
    float arg2 = sigma * sigma;
    float temp3 = (arg2 / r2) * (arg2 / r2) * (arg2 / r2);
    float temp3_rc = (arg2 / cutoff2) * (arg2 / cutoff2) * (arg2 / cutoff2);

    float r4 = r2 * r2;
    float r6 = r4 * r2;
    float r8 = r6 * r2;
    float r10 = r8 * r2;
    float r12 = r10 * r2;

    derivs[0] = arg1 * (temp3 * temp3 - temp3_rc * temp3_rc);
    derivs[1] = -12.0f * arg1 * temp3 * temp3 / r2;
    derivs[2] = 168.0f * arg1 * temp3 * temp3 / r4;
    derivs[3] = -2688.0f * arg1 * temp3 * temp3 / r6;
    derivs[4] = 48384.0f * arg1 * temp3 * temp3 / r8;
    derivs[5] = -967680.0f * arg1 * temp3 * temp3 / r10;
    derivs[6] = 21288960.0f * arg1 * temp3 * temp3 / r12;
}

__device__ inline void computeLJAttractionRadialDerivatives(
    float r2,
    float epsilon,
    float sigma,
    float cutoff2,
    float* derivs
) {
    float arg1 = 4.0f * epsilon;
    float arg2 = sigma * sigma;
    float temp3 = (arg2 / r2) * (arg2 / r2) * (arg2 / r2);
    float temp3_rc = (arg2 / cutoff2) * (arg2 / cutoff2) * (arg2 / cutoff2);

    float r4 = r2 * r2;
    float r6 = r4 * r2;
    float r8 = r6 * r2;
    float r10 = r8 * r2;
    float r12 = r10 * r2;

    derivs[0] = arg1 * (temp3 - temp3_rc);
    derivs[1] = -6.0f * arg1 * temp3 / r2;
    derivs[2] = 48.0f * arg1 * temp3 / r4;
    derivs[3] = -480.0f * arg1 * temp3 / r6;
    derivs[4] = 5760.0f * arg1 * temp3 / r8;
    derivs[5] = -80640.0f * arg1 * temp3 / r10;
    derivs[6] = 1290240.0f * arg1 * temp3 / r12;
}

/**
 * Analytical radial derivatives for electrostatic (Coulomb) potential.
 * U(r) = k * q / r  where k = 138.935456 kJ·nm/(mol·e²)
 *
 * Computes derivatives up to 6th order with respect to radial distance r.
 */
__device__ inline void computeCoulombRadialDerivatives(
    float r2,
    float charge,
    float* derivs
) {
    const float COULOMB_CONST = 138.935456f;  // kJ·nm/(mol·e²)

    // Compute r and powers needed
    float r = sqrtf(r2);
    float r2_val = r * r;     // r²
    float r3 = r2_val * r;    // r³
    float r4 = r2_val * r2_val; // r⁴
    float r5 = r4 * r;        // r⁵
    float r6 = r3 * r3;       // r⁶
    float r7 = r6 * r;        // r⁷

    float K = COULOMB_CONST * charge;

    // Derivatives of U = K / r
    derivs[0] = K / r;              // U
    derivs[1] = -K / r2_val;        // dU/dr
    derivs[2] = 2.0f * K / r3;      // d²U/dr²
    derivs[3] = -6.0f * K / r4;     // d³U/dr³
    derivs[4] = 24.0f * K / r5;     // d⁴U/dr⁴
    derivs[5] = -120.0f * K / r6;   // d⁵U/dr⁵
    derivs[6] = 720.0f * K / r7;    // d⁶U/dr⁶
}

/**
 * Analytical radial derivatives for geometric mean LJ repulsive potential.
 * This matches the form used in the original grid generation:
 * U(r) = sqrt(epsilon) * Rmin^6 / r^12  where Rmin = 2^(1/6) * sigma
 *
 * Computes derivatives up to 6th order with respect to radial distance r.
 */
__device__ inline void computeGeometricLJRepulsionRadialDerivatives(
    float r2,
    float epsilon,
    float sigma,
    float* derivs
) {
    // Compute r and powers needed
    float r = sqrtf(r2);
    float r3 = r2 * r;
    float r6 = r3 * r3;
    float r7 = r6 * r;
    float r8 = r6 * r2;
    float r9 = r8 * r;
    float r10 = r8 * r2;
    float r11 = r10 * r;
    float r12 = r6 * r6;
    float r13 = r12 * r;
    float r14 = r12 * r2;
    float r15 = r14 * r;
    float r16 = r14 * r2;
    float r17 = r16 * r;
    float r18 = r16 * r2;

    // Compute Rmin and K = sqrt(epsilon) * Rmin^6
    float rmin = powf(2.0f, 1.0f/6.0f) * sigma;
    float rmin3 = rmin * rmin * rmin;
    float rmin6 = rmin3 * rmin3;
    float K = sqrtf(epsilon) * rmin6;

    // Derivatives of U = K / r^12
    derivs[0] = K / r12;                    // U
    derivs[1] = -12.0f * K / r13;           // dU/dr
    derivs[2] = 156.0f * K / r14;           // d²U/dr²
    derivs[3] = -2184.0f * K / r15;         // d³U/dr³
    derivs[4] = 32760.0f * K / r16;         // d⁴U/dr⁴
    derivs[5] = -524160.0f * K / r17;       // d⁵U/dr⁵
    derivs[6] = 8910720.0f * K / r18;       // d⁶U/dr⁶
}

/**
 * Analytical radial derivatives for geometric mean LJ attractive potential.
 * This matches the form used in the original grid generation:
 * U(r) = -2 * sqrt(epsilon) * Rmin^3 / r^6  where Rmin = 2^(1/6) * sigma
 *
 * Computes derivatives up to 6th order with respect to radial distance r.
 */
__device__ inline void computeGeometricLJAttractionRadialDerivatives(
    float r2,
    float epsilon,
    float sigma,
    float* derivs
) {
    // Compute r and powers needed
    float r = sqrtf(r2);
    float r3 = r2 * r;
    float r6 = r3 * r3;
    float r7 = r6 * r;
    float r8 = r6 * r2;
    float r9 = r8 * r;
    float r10 = r8 * r2;
    float r11 = r10 * r;
    float r12 = r6 * r6;

    // Compute Rmin and K = -2 * sqrt(epsilon) * Rmin^3
    float rmin = powf(2.0f, 1.0f/6.0f) * sigma;
    float rmin3 = rmin * rmin * rmin;
    float K = -2.0f * sqrtf(epsilon) * rmin3;

    // Derivatives of U = K / r^6
    derivs[0] = K / r6;                     // U
    derivs[1] = -6.0f * K / r7;             // dU/dr
    derivs[2] = 42.0f * K / r8;             // d²U/dr²
    derivs[3] = -336.0f * K / r9;           // d³U/dr³
    derivs[4] = 3024.0f * K / r10;          // d⁴U/dr⁴
    derivs[5] = -30240.0f * K / r11;        // d⁵U/dr⁵
    derivs[6] = 332640.0f * K / r12;        // d⁶U/dr⁶
}

/**
 * Apply tanh capping to energy and its derivatives (up to 6th order).
 *
 * V_capped = U_max * tanh(U_raw / U_max)
 *
 * Uses Faà di Bruno's formula for chain rule of composite functions.
 * When U_raw/U_max > 20, returns flat potential (all derivatives = 0).
 *
 * Input/Output: derivs[7] = [U, dU/dr, d²U/dr², ..., d⁶U/dr⁶]
 */
__device__ inline void applyCappingToDerivatives(float* derivs, float U_max) {
    float u = derivs[0] / U_max;  // Scaled energy

    // Saturation regime: sech² underflows, return flat potential
    if (u > 20.0f) {
        derivs[0] = U_max;
        for (int i = 1; i < 7; i++) {
            derivs[i] = 0.0f;
        }
        return;
    }

    float t = tanhf(u);
    float s2 = 1.0f - t * t;  // sech²(u)

    // Scaled raw derivatives: u_n = (1/U_max)^n * d^n(U_raw)/dr^n
    float u1 = derivs[1] / U_max;
    float u2 = derivs[2] / U_max;
    float u3 = derivs[3] / U_max;
    float u4 = derivs[4] / U_max;
    float u5 = derivs[5] / U_max;
    float u6 = derivs[6] / U_max;

    // Derivatives of tanh(u) w.r.t. u (pre-multiplied by sech²)
    float dt1 = s2;
    float dt2 = -2.0f * s2 * t;
    float dt3 = 2.0f * s2 * (3.0f * t*t - 1.0f);
    float dt4 = -8.0f * s2 * t * (3.0f * t*t - 2.0f);
    float dt5 = 8.0f * s2 * (15.0f * t*t*t*t - 15.0f * t*t + 2.0f);
    float dt6 = -16.0f * s2 * t * (45.0f * t*t*t*t - 60.0f * t*t + 17.0f);

    // Apply Faà di Bruno's formula: V = U_max * tanh(u(r))
    derivs[0] = U_max * t;

    derivs[1] = U_max * dt1 * u1;

    derivs[2] = U_max * (dt1 * u2 + dt2 * u1*u1);

    derivs[3] = U_max * (dt1 * u3 + 3.0f * dt2 * u1 * u2 + dt3 * u1*u1*u1);

    derivs[4] = U_max * (dt1 * u4 + 4.0f * dt2 * u1 * u3 + 3.0f * dt2 * u2*u2
                       + 6.0f * dt3 * u1*u1 * u2 + dt4 * u1*u1*u1*u1);

    derivs[5] = U_max * (dt1 * u5 + 5.0f * dt2 * u1 * u4 + 10.0f * dt2 * u2 * u3
                       + 10.0f * dt3 * u1*u1 * u3 + 15.0f * dt3 * u1 * u2*u2
                       + 10.0f * dt4 * u1*u1*u1 * u2 + dt5 * u1*u1*u1*u1*u1);

    derivs[6] = U_max * (dt1 * u6 + 6.0f * dt2 * u1 * u5 + 15.0f * dt2 * u2 * u4
                       + 10.0f * dt2 * u3*u3 + 15.0f * dt3 * u1*u1 * u4
                       + 60.0f * dt3 * u1 * u2 * u3 + 15.0f * dt3 * u2*u2*u2
                       + 20.0f * dt4 * u1*u1*u1 * u3 + 45.0f * dt4 * u1*u1 * u2*u2
                       + 15.0f * dt5 * u1*u1*u1*u1 * u2 + dt6 * u1*u1*u1*u1*u1*u1);
}

/**
 * Accumulate one atom's contribution to the 27 Cartesian derivatives using RASPA3 tensor formulas.
 *
 * This implements the exact tensor chain rule formulas from RASPA3's framework_molecule_grid.cpp
 * to convert radial LJ derivatives to Cartesian spatial derivatives.
 *
 * Input:
 *   dr[3]: displacement vector (grid_point - atom_position) components [dx, dy, dz]
 *   radial_derivs[7]: radial derivatives [U, dU/dr, d²U/dr², d³U/dr³, d⁴U/dr⁴, d⁵U/dr⁵, d⁶U/dr⁶]
 *
 * Output (accumulated):
 *   cartesian_derivs[27]: the 27 Cartesian derivatives in RASPA3 storage order
 *     [0]      = U
 *     [1-3]    = ∂U/∂x, ∂U/∂y, ∂U/∂z
 *     [4-9]    = ∂²U/∂x², ∂²U/∂x∂y, ∂²U/∂x∂z, ∂²U/∂y², ∂²U/∂y∂z, ∂²U/∂z²
 *     [10-16]  = 7 third derivatives (∂³U/∂x²∂y, ∂³U/∂x²∂z, ∂³U/∂x∂y², ∂³U/∂x∂y∂z, ∂³U/∂y²∂z, ∂³U/∂x∂z², ∂³U/∂y∂z²)
 *     [17-22]  = 6 fourth derivatives
 *     [23-25]  = 3 fifth derivatives
 *     [26]     = 1 sixth derivative (∂⁶U/∂x²∂y²∂z²)
 */
__device__ inline void accumulateCartesianDerivatives(
    const float dr[3],
    const float radial_derivs[7],
    float cartesian_derivs[27]
) {
    // Aliases for readability
    const float dx = dr[0], dy = dr[1], dz = dr[2];
    const float U     = radial_derivs[0];
    const float dU    = radial_derivs[1];
    const float d2U   = radial_derivs[2];
    const float d3U   = radial_derivs[3];
    const float d4U   = radial_derivs[4];
    const float d5U   = radial_derivs[5];
    const float d6U   = radial_derivs[6];

    // Compute r and inverse powers for Cartesian conversion
    float r2 = dx*dx + dy*dy + dz*dz;
    float r = sqrtf(r2);
    float invr = 1.0f / r;

    // Index 0: Energy
    cartesian_derivs[0] += U;

    // Indices 1-3: First derivatives
    // Formula: ∂U/∂x_i = (dU/dr) * (dr[i] / r)
    cartesian_derivs[1] += (dx * invr) * dU;  // ∂U/∂x
    cartesian_derivs[2] += (dy * invr) * dU;  // ∂U/∂y
    cartesian_derivs[3] += (dz * invr) * dU;  // ∂U/∂z

    // Indices 4-9: Second derivatives (6 unique)
    // Formula: ∂²U/∂x_i∂x_j = (d²U/dr²) * dr[i] * dr[j] + δ_ij * (dU/dr)
    cartesian_derivs[4] += d2U * dx * dx + dU;      // ∂²U/∂x²
    cartesian_derivs[5] += d2U * dx * dy;           // ∂²U/∂x∂y
    cartesian_derivs[6] += d2U * dx * dz;           // ∂²U/∂x∂z
    cartesian_derivs[7] += d2U * dy * dy + dU;      // ∂²U/∂y²
    cartesian_derivs[8] += d2U * dy * dz;           // ∂²U/∂y∂z
    cartesian_derivs[9] += d2U * dz * dz + dU;      // ∂²U/∂z²

    // Indices 10-16: Third derivatives (7 unique)
    // Formula: ∂³U/∂x_i∂x_j∂x_k = (d³U/dr³) * dr[i] * dr[j] * dr[k]
    //                            + δ_jk * (d²U/dr²) * dr[i]
    //                            + δ_ik * (d²U/dr²) * dr[j]
    //                            + δ_ij * (d²U/dr²) * dr[k]
    cartesian_derivs[10] += d3U * dx * dx * dy + d2U * dy;              // ∂³U/∂x²∂y (i=0,j=0,k=1: δ_ij=1)
    cartesian_derivs[11] += d3U * dx * dx * dz + d2U * dz;              // ∂³U/∂x²∂z (i=0,j=0,k=2: δ_ij=1)
    cartesian_derivs[12] += d3U * dx * dy * dy + d2U * dx;              // ∂³U/∂x∂y² (i=0,j=1,k=1: δ_jk=1)
    cartesian_derivs[13] += d3U * dx * dy * dz;                         // ∂³U/∂x∂y∂z (i=0,j=1,k=2: no deltas)
    cartesian_derivs[14] += d3U * dy * dy * dz + d2U * dz;              // ∂³U/∂y²∂z (i=1,j=1,k=2: δ_ij=1)
    cartesian_derivs[15] += d3U * dx * dz * dz + d2U * dx;              // ∂³U/∂x∂z² (i=0,j=2,k=2: δ_jk=1)
    cartesian_derivs[16] += d3U * dy * dz * dz + d2U * dy;              // ∂³U/∂y∂z² (i=1,j=2,k=2: δ_jk=1)

    // Indices 17-22: Fourth derivatives (6 unique)
    // Formula: ∂⁴U/∂x_i∂x_j∂x_k∂x_l = (d⁴U/dr⁴) * dr[i] * dr[j] * dr[k] * dr[l]
    //                                + δ_ij * (d³U/dr³) * dr[k] * dr[l]
    //                                + δ_ik * (d³U/dr³) * dr[j] * dr[l]
    //                                + δ_il * (d³U/dr³) * dr[j] * dr[k]
    //                                + δ_jk * (d³U/dr³) * dr[i] * dr[l]
    //                                + δ_jl * (d³U/dr³) * dr[i] * dr[k]
    //                                + δ_kl * (d³U/dr³) * dr[i] * dr[j]
    //                                + δ_ij*δ_kl * (d²U/dr²)
    //                                + δ_il*δ_jk * (d²U/dr²)
    //                                + δ_ik*δ_jl * (d²U/dr²)
    cartesian_derivs[17] += d4U * dx * dx * dy * dy + d3U * dy * dy + d3U * dx * dx + d2U;  // ∂⁴U/∂x²∂y² (0,0,1,1)
    cartesian_derivs[18] += d4U * dx * dx * dz * dz + d3U * dz * dz + d3U * dx * dx + d2U;  // ∂⁴U/∂x²∂z² (0,0,2,2)
    cartesian_derivs[19] += d4U * dy * dy * dz * dz + d3U * dz * dz + d3U * dy * dy + d2U;  // ∂⁴U/∂y²∂z² (1,1,2,2)
    cartesian_derivs[20] += d4U * dx * dx * dy * dz + d3U * dy * dz;                        // ∂⁴U/∂x²∂y∂z (0,0,1,2)
    cartesian_derivs[21] += d4U * dx * dy * dy * dz + d3U * dx * dz;                        // ∂⁴U/∂x∂y²∂z (0,1,1,2)
    cartesian_derivs[22] += d4U * dx * dy * dz * dz + d3U * dx * dy;                        // ∂⁴U/∂x∂y∂z² (0,1,2,2)

    // Indices 23-25: Fifth derivatives (3 unique)
    // Formula follows similar pattern with 5th order main term + 4th order single-delta terms + 3rd order double-delta terms
    cartesian_derivs[23] += d5U * dx * dx * dy * dy * dz + d4U * dy * dy * dz + d4U * dx * dx * dz + d3U * dz;  // ∂⁵U/∂x²∂y²∂z (0,0,1,1,2)
    cartesian_derivs[24] += d5U * dx * dx * dy * dz * dz + d4U * dy * dz * dz + d4U * dx * dx * dy + d3U * dy;  // ∂⁵U/∂x²∂y∂z² (0,0,1,2,2)
    cartesian_derivs[25] += d5U * dx * dy * dy * dz * dz + d4U * dx * dz * dz + d4U * dx * dy * dy + d3U * dx;  // ∂⁵U/∂x∂y²∂z² (0,1,1,2,2)

    // Index 26: Sixth derivative (1 unique)
    // Formula: ∂⁶U/∂x²∂y²∂z² with full tensor expansion
    cartesian_derivs[26] += d6U * dx * dx * dy * dy * dz * dz
                          + d5U * dy * dy * dz * dz
                          + d5U * dx * dx * dz * dz
                          + d5U * dx * dx * dy * dy
                          + d4U * dz * dz
                          + d4U * dy * dy
                          + d4U * dx * dx
                          + d3U;  // ∂⁶U/∂x²∂y²∂z² (0,0,1,1,2,2)
}

#endif  // OPENMM_GRIDFORCE_LJ_ANALYTICAL_DERIVATIVES_H_
