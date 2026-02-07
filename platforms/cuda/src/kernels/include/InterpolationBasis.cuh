#ifndef OPENMM_GRIDFORCE_INTERPOLATION_BASIS_H_
#define OPENMM_GRIDFORCE_INTERPOLATION_BASIS_H_

/**
 * Basis functions for grid interpolation.
 * Includes cubic and quintic B-spline basis functions and their derivatives.
 */

// Cubic B-spline basis functions
__device__ inline float bspline_basis0(float t) { return (1.0f - t) * (1.0f - t) * (1.0f - t) / 6.0f; }
__device__ inline float bspline_basis1(float t) { return (3.0f * t * t * t - 6.0f * t * t + 4.0f) / 6.0f; }
__device__ inline float bspline_basis2(float t) { return (-3.0f * t * t * t + 3.0f * t * t + 3.0f * t + 1.0f) / 6.0f; }
__device__ inline float bspline_basis3(float t) { return t * t * t / 6.0f; }

// Derivatives of cubic B-spline basis functions
__device__ inline float bspline_deriv0(float t) { return -(1.0f - t) * (1.0f - t) / 2.0f; }
__device__ inline float bspline_deriv1(float t) { return (3.0f * t * t - 4.0f * t) / 2.0f; }
__device__ inline float bspline_deriv2(float t) { return (-3.0f * t * t + 2.0f * t + 1.0f) / 2.0f; }
__device__ inline float bspline_deriv3(float t) { return t * t / 2.0f; }

// Second derivatives of cubic B-spline basis functions
// Used for Hessian computation in normal modes analysis
// Derived from: B0(t) = (1-t)³/6, B1(t) = (3t³-6t²+4)/6, etc.
__device__ inline float bspline_deriv2_0(float t) { return 1.0f - t; }
__device__ inline float bspline_deriv2_1(float t) { return 3.0f * t - 2.0f; }
__device__ inline float bspline_deriv2_2(float t) { return -3.0f * t + 1.0f; }
__device__ inline float bspline_deriv2_3(float t) { return t; }

// =============================================================================
// Quintic B-spline basis functions (degree 5, 6-point stencil)
// =============================================================================
// For parameter t in [0,1], the 6 basis functions correspond to grid points
// at offsets -2, -1, 0, +1, +2, +3 relative to the cell containing the point.
//
// Formulas derived from the normalized uniform quintic B-spline:
//   w0(t) = (1-t)^5 / 120
//   w1(t) = [(2-t)^5 - 6(1-t)^5] / 120
//   w2(t) = [(3-t)^5 - 6(2-t)^5 + 15(1-t)^5] / 120
//   w3(t) = [(2+t)^5 - 6(1+t)^5 + 15*t^5] / 120
//   w4(t) = [(1+t)^5 - 6*t^5] / 120
//   w5(t) = t^5 / 120
//
// Partition of unity: sum = 1 for all t.
// At t=0: weights = [1, 26, 66, 26, 1, 0] / 120
// At t=1: weights = [0, 1, 26, 66, 26, 1] / 120

__device__ inline float qbspline_basis0(float t) {
    float s = 1.0f - t;
    return s*s*s*s*s / 120.0f;
}
__device__ inline float qbspline_basis1(float t) {
    float a = 2.0f - t; float s = 1.0f - t;
    return (a*a*a*a*a - 6.0f*s*s*s*s*s) / 120.0f;
}
__device__ inline float qbspline_basis2(float t) {
    float a = 3.0f - t; float b = 2.0f - t; float s = 1.0f - t;
    return (a*a*a*a*a - 6.0f*b*b*b*b*b + 15.0f*s*s*s*s*s) / 120.0f;
}
__device__ inline float qbspline_basis3(float t) {
    float a = 2.0f + t; float b = 1.0f + t;
    return (a*a*a*a*a - 6.0f*b*b*b*b*b + 15.0f*t*t*t*t*t) / 120.0f;
}
__device__ inline float qbspline_basis4(float t) {
    float a = 1.0f + t;
    return (a*a*a*a*a - 6.0f*t*t*t*t*t) / 120.0f;
}
__device__ inline float qbspline_basis5(float t) {
    return t*t*t*t*t / 120.0f;
}

// Derivatives of quintic B-spline basis functions
// dw/dt = (d/dt of the power expressions) / 120
//   w0'(t) = -5(1-t)^4 / 120 = -(1-t)^4 / 24
//   w1'(t) = [-5(2-t)^4 + 30(1-t)^4] / 120
//   w2'(t) = [-5(3-t)^4 + 30(2-t)^4 - 75(1-t)^4] / 120
//   w3'(t) = [5(2+t)^4 - 30(1+t)^4 + 75*t^4] / 120
//   w4'(t) = [5(1+t)^4 - 30*t^4] / 120
//   w5'(t) = 5*t^4 / 120 = t^4 / 24

__device__ inline float qbspline_deriv0(float t) {
    float s = 1.0f - t;
    return -5.0f*s*s*s*s / 120.0f;
}
__device__ inline float qbspline_deriv1(float t) {
    float a = 2.0f - t; float s = 1.0f - t;
    return (-5.0f*a*a*a*a + 30.0f*s*s*s*s) / 120.0f;
}
__device__ inline float qbspline_deriv2(float t) {
    float a = 3.0f - t; float b = 2.0f - t; float s = 1.0f - t;
    return (-5.0f*a*a*a*a + 30.0f*b*b*b*b - 75.0f*s*s*s*s) / 120.0f;
}
__device__ inline float qbspline_deriv3(float t) {
    float a = 2.0f + t; float b = 1.0f + t;
    return (5.0f*a*a*a*a - 30.0f*b*b*b*b + 75.0f*t*t*t*t) / 120.0f;
}
__device__ inline float qbspline_deriv4(float t) {
    float a = 1.0f + t;
    return (5.0f*a*a*a*a - 30.0f*t*t*t*t) / 120.0f;
}
__device__ inline float qbspline_deriv5(float t) {
    return 5.0f*t*t*t*t / 120.0f;
}

#endif  // OPENMM_GRIDFORCE_INTERPOLATION_BASIS_H_
