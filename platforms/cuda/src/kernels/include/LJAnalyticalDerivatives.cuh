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
 * Convert radial derivatives to Cartesian spatial derivatives (1st and 2nd order).
 * Input: radial_derivs[7] containing U, dU/dr, d2U/dr2, ..., d6U/dr6
 * Output: spatial_derivs[10] containing U, dU/dx, dU/dy, dU/dz, d2U/dx2, d2U/dxdy, ...
 * Uses chain rule: dU/dx = (dU/dr)(x/r), d2U/dx2 = (d2U/dr2)(x/r)^2 + (dU/dr)(1/r - x^2/r^3)
 */

#endif  // OPENMM_GRIDFORCE_LJ_ANALYTICAL_DERIVATIVES_H_
