#ifndef OPENMM_GRIDFORCE_INTERPOLATION_BASIS_H_
#define OPENMM_GRIDFORCE_INTERPOLATION_BASIS_H_

/**
 * Basis functions for grid interpolation.
 * Includes cubic B-spline basis functions and their derivatives.
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

#endif  // OPENMM_GRIDFORCE_INTERPOLATION_BASIS_H_
