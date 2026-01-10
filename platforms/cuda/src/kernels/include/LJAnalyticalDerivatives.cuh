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
    float temp3 = (arg2 / r2) * (arg2 / r2) * (arg2 / r2);  // (σ/r)^6

    // Need r for correct odd-power denominators
    float r = sqrtf(r2);
    float r3 = r2 * r;
    float r4 = r2 * r2;
    float r5 = r4 * r;
    float r6 = r4 * r2;

    // Coefficients derived from d^n/dr^n of r^(-12) and r^(-6) terms:
    // d^n/dr^n(r^(-m)) = (-1)^n * m*(m+1)*...*(m+n-1) * r^(-m-n)
    derivs[0] = arg1 * (temp3 * temp3 - temp3) - shift;
    derivs[1] = arg1 * (-12.0f * temp3 * temp3 + 6.0f * temp3) / r;
    derivs[2] = arg1 * (156.0f * temp3 * temp3 - 42.0f * temp3) / r2;
    derivs[3] = arg1 * (-2184.0f * temp3 * temp3 + 336.0f * temp3) / r3;
    derivs[4] = arg1 * (32760.0f * temp3 * temp3 - 3024.0f * temp3) / r4;
    derivs[5] = arg1 * (-524160.0f * temp3 * temp3 + 30240.0f * temp3) / r5;
    derivs[6] = arg1 * (8910720.0f * temp3 * temp3 - 332640.0f * temp3) / r6;
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

    // Need r for correct odd-power denominators
    float r = sqrtf(r2);
    float r3 = r2 * r;
    float r4 = r2 * r2;
    float r5 = r4 * r;
    float r6 = r4 * r2;

    // U = 4ε(σ/r)¹² with shift at cutoff
    // Coefficients: d^n/dr^n(r^(-12)) = (-1)^n × 12×13×...×(12+n-1) × r^(-12-n)
    derivs[0] = arg1 * (temp3 * temp3 - temp3_rc * temp3_rc);
    derivs[1] = -12.0f * arg1 * temp3 * temp3 / r;
    derivs[2] = 156.0f * arg1 * temp3 * temp3 / r2;
    derivs[3] = -2184.0f * arg1 * temp3 * temp3 / r3;
    derivs[4] = 32760.0f * arg1 * temp3 * temp3 / r4;
    derivs[5] = -524160.0f * arg1 * temp3 * temp3 / r5;
    derivs[6] = 8910720.0f * arg1 * temp3 * temp3 / r6;
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

    // Need r for correct odd-power denominators
    float r = sqrtf(r2);
    float r3 = r2 * r;
    float r4 = r2 * r2;
    float r5 = r4 * r;
    float r6 = r4 * r2;

    // U = 4ε(σ/r)⁶ with shift at cutoff (note: positive, not standard -4ε form)
    // Coefficients: d^n/dr^n(r^(-6)) = (-1)^n × 6×7×...×(6+n-1) × r^(-6-n)
    derivs[0] = arg1 * (temp3 - temp3_rc);
    derivs[1] = -6.0f * arg1 * temp3 / r;
    derivs[2] = 42.0f * arg1 * temp3 / r2;
    derivs[3] = -336.0f * arg1 * temp3 / r3;
    derivs[4] = 3024.0f * arg1 * temp3 / r4;
    derivs[5] = -30240.0f * arg1 * temp3 / r5;
    derivs[6] = 332640.0f * arg1 * temp3 / r6;
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
 * Accumulate one atom's contribution to the 27 Cartesian derivatives using tensor formulas.
 *
 * This implements the exact tensor chain rule formulas to convert radial derivatives
 * to Cartesian spatial derivatives for radially symmetric functions U(r).
 *
 * For a radially symmetric function U(r) where r = |x|, the key insight is that
 * derivatives are expressed in terms of direction cosines n_i = x_i/r and
 * auxiliary coefficients involving radial derivatives and inverse powers of r.
 *
 * Input:
 *   dr[3]: displacement vector (grid_point - atom_position) components [dx, dy, dz]
 *   radial_derivs[7]: radial derivatives [U, dU/dr, d²U/dr², d³U/dr³, d⁴U/dr⁴, d⁵U/dr⁵, d⁶U/dr⁶]
 *
 * Output (accumulated):
 *   cartesian_derivs[27]: the 27 Cartesian derivatives
 *     [0]      = U
 *     [1-3]    = ∂U/∂x, ∂U/∂y, ∂U/∂z
 *     [4-9]    = ∂²U/∂x², ∂²U/∂x∂y, ∂²U/∂x∂z, ∂²U/∂y², ∂²U/∂y∂z, ∂²U/∂z²
 *     [10-16]  = 7 third derivatives
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
    float invr2 = invr * invr;
    float invr3 = invr2 * invr;
    float invr4 = invr2 * invr2;
    float invr5 = invr4 * invr;

    // Direction cosines
    float nx = dx * invr;
    float ny = dy * invr;
    float nz = dz * invr;
    float nx2 = nx * nx;
    float ny2 = ny * ny;
    float nz2 = nz * nz;

    // Auxiliary coefficients for different orders
    // These combine radial derivatives with 1/r factors for proper tensor conversion
    // Order 2: ∂²U/∂xi∂xj = A2*ni*nj + (dU/r)*δij
    float A2 = d2U - dU * invr;

    // Order 3: ∂³U/∂xi∂xj∂xk = A3*ni*nj*nk + B3*(sum of delta-n terms)
    float A3 = d3U - 3.0f * d2U * invr + 3.0f * dU * invr2;
    float B3 = d2U * invr - dU * invr2;

    // Order 4: ∂⁴U/∂xi∂xj∂xk∂xl = A4*ni*nj*nk*nl + B4*(delta-nn terms) + C4*(double-delta terms)
    float A4 = d4U - 6.0f * d3U * invr + 15.0f * d2U * invr2 - 15.0f * dU * invr3;
    float B4 = d3U * invr - 3.0f * d2U * invr2 + 3.0f * dU * invr3;
    float C4 = d2U * invr2 - dU * invr3;

    // Order 5
    float A5 = d5U - 10.0f * d4U * invr + 45.0f * d3U * invr2 - 105.0f * d2U * invr3 + 105.0f * dU * invr4;
    float B5 = d4U * invr - 6.0f * d3U * invr2 + 15.0f * d2U * invr3 - 15.0f * dU * invr4;
    float C5 = d3U * invr2 - 3.0f * d2U * invr3 + 3.0f * dU * invr4;

    // Order 6
    float A6 = d6U - 15.0f * d5U * invr + 105.0f * d4U * invr2 - 420.0f * d3U * invr3 + 945.0f * d2U * invr4 - 945.0f * dU * invr5;
    float B6 = d5U * invr - 10.0f * d4U * invr2 + 45.0f * d3U * invr3 - 105.0f * d2U * invr4 + 105.0f * dU * invr5;
    float C6 = d4U * invr2 - 6.0f * d3U * invr3 + 15.0f * d2U * invr4 - 15.0f * dU * invr5;
    float D6 = d3U * invr3 - 3.0f * d2U * invr4 + 3.0f * dU * invr5;

    // Index 0: Energy
    cartesian_derivs[0] += U;

    // Indices 1-3: First derivatives
    // Formula: ∂U/∂xi = dU * ni
    cartesian_derivs[1] += dU * nx;  // ∂U/∂x
    cartesian_derivs[2] += dU * ny;  // ∂U/∂y
    cartesian_derivs[3] += dU * nz;  // ∂U/∂z

    // Indices 4-9: Second derivatives (6 unique)
    // Formula: ∂²U/∂xi∂xj = A2*ni*nj + (dU/r)*δij
    cartesian_derivs[4] += A2 * nx2 + dU * invr;  // ∂²U/∂x²
    cartesian_derivs[5] += A2 * nx * ny;          // ∂²U/∂x∂y
    cartesian_derivs[6] += A2 * nx * nz;          // ∂²U/∂x∂z
    cartesian_derivs[7] += A2 * ny2 + dU * invr;  // ∂²U/∂y²
    cartesian_derivs[8] += A2 * ny * nz;          // ∂²U/∂y∂z
    cartesian_derivs[9] += A2 * nz2 + dU * invr;  // ∂²U/∂z²

    // Indices 10-16: Third derivatives (7 unique)
    // Formula: ∂³U/∂xi∂xj∂xk = A3*ni*nj*nk + B3*(δij*nk + δik*nj + δjk*ni)
    cartesian_derivs[10] += A3 * nx2 * ny + B3 * ny;       // ∂³U/∂x²∂y (δij=1)
    cartesian_derivs[11] += A3 * nx2 * nz + B3 * nz;       // ∂³U/∂x²∂z (δij=1)
    cartesian_derivs[12] += A3 * nx * ny2 + B3 * nx;       // ∂³U/∂x∂y² (δjk=1)
    cartesian_derivs[13] += A3 * nx * ny * nz;             // ∂³U/∂x∂y∂z (no deltas)
    cartesian_derivs[14] += A3 * ny2 * nz + B3 * nz;       // ∂³U/∂y²∂z (δij=1)
    cartesian_derivs[15] += A3 * nx * nz2 + B3 * nx;       // ∂³U/∂x∂z² (δjk=1)
    cartesian_derivs[16] += A3 * ny * nz2 + B3 * ny;       // ∂³U/∂y∂z² (δjk=1)

    // Indices 17-22: Fourth derivatives (6 unique)
    // Formula: ∂⁴U/∂xi∂xj∂xk∂xl = A4*ni*nj*nk*nl + B4*(delta-nn sums) + C4*(double-delta)
    cartesian_derivs[17] += A4 * nx2 * ny2 + B4 * (nx2 + ny2) + C4;  // ∂⁴U/∂x²∂y²
    cartesian_derivs[18] += A4 * nx2 * nz2 + B4 * (nx2 + nz2) + C4;  // ∂⁴U/∂x²∂z²
    cartesian_derivs[19] += A4 * ny2 * nz2 + B4 * (ny2 + nz2) + C4;  // ∂⁴U/∂y²∂z²
    cartesian_derivs[20] += A4 * nx2 * ny * nz + B4 * ny * nz;       // ∂⁴U/∂x²∂y∂z
    cartesian_derivs[21] += A4 * nx * ny2 * nz + B4 * nx * nz;       // ∂⁴U/∂x∂y²∂z
    cartesian_derivs[22] += A4 * nx * ny * nz2 + B4 * nx * ny;       // ∂⁴U/∂x∂y∂z²

    // Indices 23-25: Fifth derivatives (3 unique)
    cartesian_derivs[23] += A5 * nx2 * ny2 * nz + B5 * ((nx2 + ny2) * nz) + C5 * nz;  // ∂⁵U/∂x²∂y²∂z
    cartesian_derivs[24] += A5 * nx2 * ny * nz2 + B5 * (ny * nz2 + nx2 * ny) + C5 * ny;  // ∂⁵U/∂x²∂y∂z²
    cartesian_derivs[25] += A5 * nx * ny2 * nz2 + B5 * (nx * nz2 + nx * ny2) + C5 * nx;  // ∂⁵U/∂x∂y²∂z²

    // Index 26: Sixth derivative (1 unique)
    cartesian_derivs[26] += A6 * nx2 * ny2 * nz2 + B6 * (nx2 * ny2 + nx2 * nz2 + ny2 * nz2) + C6 * (nx2 + ny2 + nz2) + D6;  // ∂⁶U/∂x²∂y²∂z²
}

#endif  // OPENMM_GRIDFORCE_LJ_ANALYTICAL_DERIVATIVES_H_
