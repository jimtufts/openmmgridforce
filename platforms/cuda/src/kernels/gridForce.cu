/**
 * CUDA implementation of grid force calculation.
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

/**
 * RASPA3 Triquintic Hermite interpolation transformation matrix (216x216).
 * Converts 216 derivative values (27 derivatives × 8 corners) to polynomial coefficients.
 * Usage: a[i] = 0.125 * sum_j(TRIQUINTIC_COEFFICIENTS[i][j] * X[j])
 *
 * Adapted from RASPA3 molecular simulation code (MIT license)
 * https://github.com/iRASPA/RASPA3
 */
__device__ const float TRIQUINTIC_COEFFICIENTS[216][216] = {
    {8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {-80, 80,  0, 0, 0, 0, 0, 0, -48, -32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0,   -12, 4, 0, 0, 0, 0, 0, 0,   0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0,   0,   0, 0, 0, 0, 0, 0, 0,   0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0,   0,   0, 0, 0, 0, 0, 0, 0,   0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0,   0,   0, 0, 0, 0, 0, 0, 0,   0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0,   0,   0, 0, 0, 0, 0, 0, 0,   0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0,   0,   0, 0, 0, 0, 0, 0, 0,   0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {120, -120, 0,  0, 0, 0, 0, 0, 64, 56, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0,   12,   -8, 0, 0, 0, 0, 0, 0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0,   0,    0,  0, 0, 0, 0, 0, 0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0,   0,    0,  0, 0, 0, 0, 0, 0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0,   0,    0,  0, 0, 0, 0, 0, 0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0,   0,    0,  0, 0, 0, 0, 0, 0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0,   0,    0,  0, 0, 0, 0, 0, 0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {-48, 48, 0, 0, 0, 0, 0, 0, -24, -24, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0,   -4, 4, 0, 0, 0, 0, 0, 0,   0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0,   0,  0, 0, 0, 0, 0, 0, 0,   0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0,   0,  0, 0, 0, 0, 0, 0, 0,   0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0,   0,  0, 0, 0, 0, 0, 0, 0,   0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0,   0,  0, 0, 0, 0, 0, 0, 0,   0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0,   0,  0, 0, 0, 0, 0, 0, 0,   0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0,   0,   0, 0, 0, 0, 0, -80, 80, 0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, -48, -32, 0, 0, 0, 0, 0, 0,   0,  0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0,   0,   0, 0, 0, 0, 0, 0,   0,  -12, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0,   0,   0, 0, 0, 0, 0, 0,   0,  0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0,   0,   0, 0, 0, 0, 0, 0,   0,  0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0,   0,   0, 0, 0, 0, 0, 0,   0,  0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0,   0,   0, 0, 0, 0, 0, 0,   0,  0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0, 0, 0, 0, 0, 120, -120, 0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 64, 56, 0, 0, 0, 0, 0, 0,   0,    0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0, 0, 0, 0, 0, 0,   0,    12, -8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0, 0, 0, 0, 0, 0,   0,    0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0, 0, 0, 0, 0, 0,   0,    0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0, 0, 0, 0, 0, 0,   0,    0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0, 0, 0, 0, 0, 0,   0,    0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0,   0,   0, 0, 0, 0, 0, -48, 48, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, -24, -24, 0, 0, 0, 0, 0, 0,   0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0,   0,   0, 0, 0, 0, 0, 0,   0,  -4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0,   0,   0, 0, 0, 0, 0, 0,   0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0,   0,   0, 0, 0, 0, 0, 0,   0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0,   0,   0, 0, 0, 0, 0, 0,   0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0,   0,   0, 0, 0, 0, 0, 0,   0,  0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0,   0,   0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,   0,  0, 0, 0, 0,
     0, 0, 0, 0,   0,   0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -40, 40, 0, 0, 0, 0,
     0, 0, 0, 0,   0,   0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,   0,  0, 0, 0, 0,
     0, 0, 0, -24, -16, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,   0,  0, 0, 0, 0,
     0, 0, 0, 0,   0,   0, 0, 0, 0, 0, 0, 0, -6, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,   0,  0, 0, 0, 0,
     0, 0, 0, 0,   0,   0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,   0,  0, 0, 0, 0,
     0, 0, 0, 0,   0,   0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,   0,  0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,   0, 0, 0,  0,  0, 0, 0, 0,  0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 60, -60, 0, 0, 0,  0,  0, 0, 0, 0,  0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,   0, 0, 32, 28, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,   0, 0, 0,  0,  0, 0, 6, -4, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,   0, 0, 0,  0,  0, 0, 0, 0,  0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,   0, 0, 0,  0,  0, 0, 0, 0,  0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0,   0,   0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,   0,  0, 0, 0, 0,
     0, 0, 0, 0,   0,   0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -24, 24, 0, 0, 0, 0,
     0, 0, 0, 0,   0,   0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,   0,  0, 0, 0, 0,
     0, 0, 0, -12, -12, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,   0,  0, 0, 0, 0,
     0, 0, 0, 0,   0,   0, 0, 0, 0, 0, 0, 0, -2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,   0,  0, 0, 0, 0,
     0, 0, 0, 0,   0,   0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,   0,  0, 0, 0, 0,
     0, 0, 0, 0,   0,   0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,   0,  0, 0, 0},
    {-80, 0, 80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -48, 0, -32, 0, 0, 0, 0, 0, 0, 0,   0, 0, 0, 0, 0,
     0,   0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,   0, 0,   0, 0, 0, 0, 0, 0, -12, 0, 4, 0, 0, 0,
     0,   0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,   0, 0,   0, 0, 0, 0, 0, 0, 0,   0, 0, 0, 0, 0,
     0,   0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,   0, 0,   0, 0, 0, 0, 0, 0, 0,   0, 0, 0, 0, 0,
     0,   0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,   0, 0,   0, 0, 0, 0, 0, 0, 0,   0, 0, 0, 0, 0,
     0,   0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,   0, 0,   0, 0, 0, 0, 0, 0, 0,   0, 0, 0, 0, 0,
     0,   0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,   0, 0,   0, 0, 0, 0, 0, 0, 0,   0, 0, 0, 0},
    {0, 0, 0, 0,   0, 0, 0, 0, -80, 0,   80, 0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0,   0, 0, 0, 0, 0,   -48, 0,  -32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0,   0, 0, 0, 0, 0,   0,   0,  0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, -12, 0, 4, 0, 0, 0,   0,   0,  0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0,   0, 0, 0, 0, 0,   0,   0,  0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0,   0, 0, 0, 0, 0,   0,   0,  0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0,   0, 0, 0, 0, 0,   0,   0,  0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0,   0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0,   0, 0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, -40, 0, 40, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0,   0, 0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0,   0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, -24, 0, -16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0,   0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0,   0, 0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0,   0, 0,  0, 0, 0, 0, 0, 0, 0, 0, -6, 0, 2, 0, 0, 0, 0,   0, 0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0,   0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0,   0, 0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0,   0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0,   0, 0,   0, 0, 0, 0, 0, 0, 0, 0, 0},
    {800, -800, -800, 800, 0, 0, 0, 0, 480, 320,  -480, -320, 0, 0, 0, 0, 480, -480, 320, -320, 0, 0, 0, 0,
     0,   0,    0,    0,   0, 0, 0, 0, 120, -40,  -120, 40,   0, 0, 0, 0, 288, 192,  192, 128,  0, 0, 0, 0,
     0,   0,    0,    0,   0, 0, 0, 0, 120, -120, -40,  40,   0, 0, 0, 0, 0,   0,    0,   0,    0, 0, 0, 0,
     0,   0,    0,    0,   0, 0, 0, 0, 72,  -24,  48,   -16,  0, 0, 0, 0, 0,   0,    0,   0,    0, 0, 0, 0,
     72,  48,   -24,  -16, 0, 0, 0, 0, 0,   0,    0,    0,    0, 0, 0, 0, 0,   0,    0,   0,    0, 0, 0, 0,
     0,   0,    0,    0,   0, 0, 0, 0, 0,   0,    0,    0,    0, 0, 0, 0, 18,  -6,   -6,  2,    0, 0, 0, 0,
     0,   0,    0,    0,   0, 0, 0, 0, 0,   0,    0,    0,    0, 0, 0, 0, 0,   0,    0,   0,    0, 0, 0, 0,
     0,   0,    0,    0,   0, 0, 0, 0, 0,   0,    0,    0,    0, 0, 0, 0, 0,   0,    0,   0,    0, 0, 0, 0,
     0,   0,    0,    0,   0, 0, 0, 0, 0,   0,    0,    0,    0, 0, 0, 0, 0,   0,    0,   0,    0, 0, 0, 0},
    {-1200, 1200, 1200, -1200, 0, 0, 0, 0, -640, -560, 640, 560, 0, 0, 0, 0, -720, 720,  -480, 480,  0, 0, 0, 0,
     0,     0,    0,    0,     0, 0, 0, 0, -120, 80,   120, -80, 0, 0, 0, 0, -384, -336, -256, -224, 0, 0, 0, 0,
     0,     0,    0,    0,     0, 0, 0, 0, -180, 180,  60,  -60, 0, 0, 0, 0, 0,    0,    0,    0,    0, 0, 0, 0,
     0,     0,    0,    0,     0, 0, 0, 0, -72,  48,   -48, 32,  0, 0, 0, 0, 0,    0,    0,    0,    0, 0, 0, 0,
     -96,   -84,  32,   28,    0, 0, 0, 0, 0,    0,    0,   0,   0, 0, 0, 0, 0,    0,    0,    0,    0, 0, 0, 0,
     0,     0,    0,    0,     0, 0, 0, 0, 0,    0,    0,   0,   0, 0, 0, 0, -18,  12,   6,    -4,   0, 0, 0, 0,
     0,     0,    0,    0,     0, 0, 0, 0, 0,    0,    0,   0,   0, 0, 0, 0, 0,    0,    0,    0,    0, 0, 0, 0,
     0,     0,    0,    0,     0, 0, 0, 0, 0,    0,    0,   0,   0, 0, 0, 0, 0,    0,    0,    0,    0, 0, 0, 0,
     0,     0,    0,    0,     0, 0, 0, 0, 0,    0,    0,   0,   0, 0, 0, 0, 0,    0,    0,    0,    0, 0, 0, 0},
    {480, -480, -480, 480, 0, 0, 0, 0, 240, 240, -240, -240, 0, 0, 0, 0, 288, -288, 192, -192, 0, 0, 0, 0,
     0,   0,    0,    0,   0, 0, 0, 0, 40,  -40, -40,  40,   0, 0, 0, 0, 144, 144,  96,  96,   0, 0, 0, 0,
     0,   0,    0,    0,   0, 0, 0, 0, 72,  -72, -24,  24,   0, 0, 0, 0, 0,   0,    0,   0,    0, 0, 0, 0,
     0,   0,    0,    0,   0, 0, 0, 0, 24,  -24, 16,   -16,  0, 0, 0, 0, 0,   0,    0,   0,    0, 0, 0, 0,
     36,  36,   -12,  -12, 0, 0, 0, 0, 0,   0,   0,    0,    0, 0, 0, 0, 0,   0,    0,   0,    0, 0, 0, 0,
     0,   0,    0,    0,   0, 0, 0, 0, 0,   0,   0,    0,    0, 0, 0, 0, 6,   -6,   -2,  2,    0, 0, 0, 0,
     0,   0,    0,    0,   0, 0, 0, 0, 0,   0,   0,    0,    0, 0, 0, 0, 0,   0,    0,   0,    0, 0, 0, 0,
     0,   0,    0,    0,   0, 0, 0, 0, 0,   0,   0,    0,    0, 0, 0, 0, 0,   0,    0,   0,    0, 0, 0, 0,
     0,   0,    0,    0,   0, 0, 0, 0, 0,   0,   0,    0,    0, 0, 0, 0, 0,   0,    0,   0,    0, 0, 0, 0},
    {120, 0, -120, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 64, 0, 56, 0, 0, 0, 0, 0, 0, 0,  0, 0,  0, 0, 0,
     0,   0, 0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0,  0, 0, 0, 0, 0, 0, 12, 0, -8, 0, 0, 0,
     0,   0, 0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0,  0, 0, 0, 0, 0, 0, 0,  0, 0,  0, 0, 0,
     0,   0, 0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0,  0, 0, 0, 0, 0, 0, 0,  0, 0,  0, 0, 0,
     0,   0, 0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0,  0, 0, 0, 0, 0, 0, 0,  0, 0,  0, 0, 0,
     0,   0, 0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0,  0, 0, 0, 0, 0, 0, 0,  0, 0,  0, 0, 0,
     0,   0, 0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0,  0, 0, 0, 0, 0, 0, 0,  0, 0,  0, 0},
    {0, 0, 0, 0,  0, 0,  0, 0, 120, 0,  -120, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0,  0, 0,  0, 0, 0,   64, 0,    56, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0,  0, 0,  0, 0, 0,   0,  0,    0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 12, 0, -8, 0, 0, 0,   0,  0,    0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0,  0, 0,  0, 0, 0,   0,  0,    0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0,  0, 0,  0, 0, 0,   0,  0,    0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0,  0, 0,  0, 0, 0,   0,  0,    0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 60, 0, -60, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0,  0, 0,   0,
     0, 0, 0, 0, 0, 0, 0, 0, 32, 0, 28, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0,  0, 0,   0,
     0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0, -4, 0, 0,  0, 0,   0,
     0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0,  0, 0,   0,
     0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0,  0, 0,   0},
    {-1200, 1200, 1200, -1200, 0, 0, 0, 0, -720, -480, 720, 480, 0, 0, 0, 0, -640, 640,  -560, 560,  0, 0, 0, 0,
     0,     0,    0,    0,     0, 0, 0, 0, -180, 60,   180, -60, 0, 0, 0, 0, -384, -256, -336, -224, 0, 0, 0, 0,
     0,     0,    0,    0,     0, 0, 0, 0, -120, 120,  80,  -80, 0, 0, 0, 0, 0,    0,    0,    0,    0, 0, 0, 0,
     0,     0,    0,    0,     0, 0, 0, 0, -96,  32,   -84, 28,  0, 0, 0, 0, 0,    0,    0,    0,    0, 0, 0, 0,
     -72,   -48,  48,   32,    0, 0, 0, 0, 0,    0,    0,   0,   0, 0, 0, 0, 0,    0,    0,    0,    0, 0, 0, 0,
     0,     0,    0,    0,     0, 0, 0, 0, 0,    0,    0,   0,   0, 0, 0, 0, -18,  6,    12,   -4,   0, 0, 0, 0,
     0,     0,    0,    0,     0, 0, 0, 0, 0,    0,    0,   0,   0, 0, 0, 0, 0,    0,    0,    0,    0, 0, 0, 0,
     0,     0,    0,    0,     0, 0, 0, 0, 0,    0,    0,   0,   0, 0, 0, 0, 0,    0,    0,    0,    0, 0, 0, 0,
     0,     0,    0,    0,     0, 0, 0, 0, 0,    0,    0,   0,   0, 0, 0, 0, 0,    0,    0,    0,    0, 0, 0, 0},
    {1800, -1800, -1800, 1800, 0, 0, 0, 0, 960, 840,  -960, -840, 0, 0, 0, 0, 960, -960, 840, -840, 0, 0, 0, 0,
     0,    0,     0,     0,    0, 0, 0, 0, 180, -120, -180, 120,  0, 0, 0, 0, 512, 448,  448, 392,  0, 0, 0, 0,
     0,    0,     0,     0,    0, 0, 0, 0, 180, -180, -120, 120,  0, 0, 0, 0, 0,   0,    0,   0,    0, 0, 0, 0,
     0,    0,     0,     0,    0, 0, 0, 0, 96,  -64,  84,   -56,  0, 0, 0, 0, 0,   0,    0,   0,    0, 0, 0, 0,
     96,   84,    -64,   -56,  0, 0, 0, 0, 0,   0,    0,    0,    0, 0, 0, 0, 0,   0,    0,   0,    0, 0, 0, 0,
     0,    0,     0,     0,    0, 0, 0, 0, 0,   0,    0,    0,    0, 0, 0, 0, 18,  -12,  -12, 8,    0, 0, 0, 0,
     0,    0,     0,     0,    0, 0, 0, 0, 0,   0,    0,    0,    0, 0, 0, 0, 0,   0,    0,   0,    0, 0, 0, 0,
     0,    0,     0,     0,    0, 0, 0, 0, 0,   0,    0,    0,    0, 0, 0, 0, 0,   0,    0,   0,    0, 0, 0, 0,
     0,    0,     0,     0,    0, 0, 0, 0, 0,   0,    0,    0,    0, 0, 0, 0, 0,   0,    0,   0,    0, 0, 0, 0},
    {-720, 720, 720, -720, 0, 0, 0, 0, -360, -360, 360, 360, 0, 0, 0, 0, -384, 384,  -336, 336,  0, 0, 0, 0,
     0,    0,   0,   0,    0, 0, 0, 0, -60,  60,   60,  -60, 0, 0, 0, 0, -192, -192, -168, -168, 0, 0, 0, 0,
     0,    0,   0,   0,    0, 0, 0, 0, -72,  72,   48,  -48, 0, 0, 0, 0, 0,    0,    0,    0,    0, 0, 0, 0,
     0,    0,   0,   0,    0, 0, 0, 0, -32,  32,   -28, 28,  0, 0, 0, 0, 0,    0,    0,    0,    0, 0, 0, 0,
     -36,  -36, 24,  24,   0, 0, 0, 0, 0,    0,    0,   0,   0, 0, 0, 0, 0,    0,    0,    0,    0, 0, 0, 0,
     0,    0,   0,   0,    0, 0, 0, 0, 0,    0,    0,   0,   0, 0, 0, 0, -6,   6,    4,    -4,   0, 0, 0, 0,
     0,    0,   0,   0,    0, 0, 0, 0, 0,    0,    0,   0,   0, 0, 0, 0, 0,    0,    0,    0,    0, 0, 0, 0,
     0,    0,   0,   0,    0, 0, 0, 0, 0,    0,    0,   0,   0, 0, 0, 0, 0,    0,    0,    0,    0, 0, 0, 0,
     0,    0,   0,   0,    0, 0, 0, 0, 0,    0,    0,   0,   0, 0, 0, 0, 0,    0,    0,    0,    0, 0, 0, 0},
    {-48, 0, 48, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -24, 0, -24, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0,
     0,   0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,   0, 0,   0, 0, 0, 0, 0, 0, -4, 0, 4, 0, 0, 0,
     0,   0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,   0, 0,   0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0,
     0,   0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,   0, 0,   0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0,
     0,   0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,   0, 0,   0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0}
};

/**
 * Compute all 27 derivatives at each grid point using finite differences.
 * Operates on the capped grid values to avoid singularities.
 * Output: derivatives[27][nx][ny][nz] where derivatives are in physical coordinates.
 */
extern "C" __global__ void computeDerivativesKernel(
    float* __restrict__ derivatives,      // Output: 27 * totalGridPoints
    const float* __restrict__ gridValues, // Input: capped grid values
    const int* __restrict__ gridCounts,
    const float* __restrict__ gridSpacing,
    const int totalGridPoints) {

    const unsigned int gridIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (gridIdx >= totalGridPoints)
        return;

    // Convert linear index to 3D coordinates
    const int nyz = gridCounts[1] * gridCounts[2];
    const int nx = gridCounts[0];
    const int ny = gridCounts[1];
    const int nz = gridCounts[2];

    const int ix = gridIdx / nyz;
    const int remainder = gridIdx % nyz;
    const int iy = remainder / nz;
    const int iz = remainder % nz;

    const float dx = gridSpacing[0];
    const float dy = gridSpacing[1];
    const float dz = gridSpacing[2];

    const float f = gridValues[gridIdx];

    // Helper macro to safely get grid value with clamped indices
    #define GET_VAL(i, j, k) gridValues[min(max((i), 0), nx-1) * nyz + min(max((j), 0), ny-1) * nz + min(max((k), 0), nz-1)]

    // First derivatives (one-sided at boundaries, centered otherwise)
    float dx_f, dy_f, dz_f;
    if (ix == 0) {
        dx_f = (GET_VAL(ix+1, iy, iz) - f) / dx;
    } else if (ix == nx-1) {
        dx_f = (f - GET_VAL(ix-1, iy, iz)) / dx;
    } else {
        dx_f = (GET_VAL(ix+1, iy, iz) - GET_VAL(ix-1, iy, iz)) / (2.0f * dx);
    }

    if (iy == 0) {
        dy_f = (GET_VAL(ix, iy+1, iz) - f) / dy;
    } else if (iy == ny-1) {
        dy_f = (f - GET_VAL(ix, iy-1, iz)) / dy;
    } else {
        dy_f = (GET_VAL(ix, iy+1, iz) - GET_VAL(ix, iy-1, iz)) / (2.0f * dy);
    }

    if (iz == 0) {
        dz_f = (GET_VAL(ix, iy, iz+1) - f) / dz;
    } else if (iz == nz-1) {
        dz_f = (f - GET_VAL(ix, iy, iz-1)) / dz;
    } else {
        dz_f = (GET_VAL(ix, iy, iz+1) - GET_VAL(ix, iy, iz-1)) / (2.0f * dz);
    }

    // Second derivatives
    float dxx_f, dyy_f, dzz_f, dxy_f, dxz_f, dyz_f;
    if (ix == 0 || ix == nx-1) {
        dxx_f = 0.0f;
        dxy_f = 0.0f;
        dxz_f = 0.0f;
    } else {
        dxx_f = (GET_VAL(ix+1, iy, iz) - 2.0f*f + GET_VAL(ix-1, iy, iz)) / (dx*dx);
        dxy_f = (GET_VAL(ix+1, iy+1, iz) - GET_VAL(ix+1, iy-1, iz) -
                 GET_VAL(ix-1, iy+1, iz) + GET_VAL(ix-1, iy-1, iz)) / (4.0f * dx * dy);
        dxz_f = (GET_VAL(ix+1, iy, iz+1) - GET_VAL(ix+1, iy, iz-1) -
                 GET_VAL(ix-1, iy, iz+1) + GET_VAL(ix-1, iy, iz-1)) / (4.0f * dx * dz);
    }

    if (iy == 0 || iy == ny-1) {
        dyy_f = 0.0f;
        dyz_f = 0.0f;
        if (ix != 0 && ix != nx-1) dxy_f = 0.0f;
    } else {
        dyy_f = (GET_VAL(ix, iy+1, iz) - 2.0f*f + GET_VAL(ix, iy-1, iz)) / (dy*dy);
        dyz_f = (GET_VAL(ix, iy+1, iz+1) - GET_VAL(ix, iy+1, iz-1) -
                 GET_VAL(ix, iy-1, iz+1) + GET_VAL(ix, iy-1, iz-1)) / (4.0f * dy * dz);
    }

    if (iz == 0 || iz == nz-1) {
        dzz_f = 0.0f;
        if (ix != 0 && ix != nx-1) dxz_f = 0.0f;
        if (iy != 0 && iy != ny-1) dyz_f = 0.0f;
    } else {
        dzz_f = (GET_VAL(ix, iy, iz+1) - 2.0f*f + GET_VAL(ix, iy, iz-1)) / (dz*dz);
    }

    #undef GET_VAL

    // Higher order derivatives (computed only in interior)
    float dxxy_f = 0.0f, dxxz_f = 0.0f, dxyy_f = 0.0f, dxyz_f = 0.0f;
    float dxzz_f = 0.0f, dyyz_f = 0.0f, dyzz_f = 0.0f;
    float dxxyy_f = 0.0f, dxxzz_f = 0.0f, dyyzz_f = 0.0f;
    float dxxyz_f = 0.0f, dxyyz_f = 0.0f, dxyzz_f = 0.0f;
    float dxxyyz_f = 0.0f, dxxyzz_f = 0.0f, dxyyzz_f = 0.0f, dxxyyzz_f = 0.0f;

    // Store in layout [deriv_idx][x][y][z]
    const int offset = gridIdx;
    derivatives[0 * totalGridPoints + offset] = f;
    derivatives[1 * totalGridPoints + offset] = dx_f;
    derivatives[2 * totalGridPoints + offset] = dy_f;
    derivatives[3 * totalGridPoints + offset] = dz_f;
    derivatives[4 * totalGridPoints + offset] = dxx_f;
    derivatives[5 * totalGridPoints + offset] = dxy_f;
    derivatives[6 * totalGridPoints + offset] = dxz_f;
    derivatives[7 * totalGridPoints + offset] = dyy_f;
    derivatives[8 * totalGridPoints + offset] = dyz_f;
    derivatives[9 * totalGridPoints + offset] = dzz_f;
    derivatives[10 * totalGridPoints + offset] = dxxy_f;
    derivatives[11 * totalGridPoints + offset] = dxxz_f;
    derivatives[12 * totalGridPoints + offset] = dxyy_f;
    derivatives[13 * totalGridPoints + offset] = dxyz_f;
    derivatives[14 * totalGridPoints + offset] = dxzz_f;
    derivatives[15 * totalGridPoints + offset] = dyyz_f;
    derivatives[16 * totalGridPoints + offset] = dyzz_f;
    derivatives[17 * totalGridPoints + offset] = dxxyy_f;
    derivatives[18 * totalGridPoints + offset] = dxxzz_f;
    derivatives[19 * totalGridPoints + offset] = dyyzz_f;
    derivatives[20 * totalGridPoints + offset] = dxxyz_f;
    derivatives[21 * totalGridPoints + offset] = dxyyz_f;
    derivatives[22 * totalGridPoints + offset] = dxyzz_f;
    derivatives[23 * totalGridPoints + offset] = dxxyyz_f;
    derivatives[24 * totalGridPoints + offset] = dxxyzz_f;
    derivatives[25 * totalGridPoints + offset] = dxyyzz_f;
    derivatives[26 * totalGridPoints + offset] = dxxyyzz_f;
}

/**
 * Generate grid values on GPU.
 * Each thread calculates one grid point by summing contributions from all receptor atoms.
 */
extern "C" __global__ void generateGridKernel(
    float* __restrict__ gridValues,
    const float3* __restrict__ receptorPositions,
    const float* __restrict__ receptorCharges,
    const float* __restrict__ receptorSigmas,
    const float* __restrict__ receptorEpsilons,
    const int numReceptorAtoms,
    const int gridType,  // 0=charge, 1=ljr, 2=lja
    const float gridCap,  // Capping threshold (kJ/mol)
    const float invPower,  // Inverse power transformation
    const float originX,
    const float originY,
    const float originZ,
    const int* __restrict__ gridCounts,
    const float* __restrict__ gridSpacing,
    const int totalGridPoints) {

    // Get grid point index
    const unsigned int gridIdx = blockIdx.x * blockDim.x + threadIdx.x;

    if (gridIdx >= totalGridPoints)
        return;

    // Convert linear index to 3D grid coordinates
    const int nyz = gridCounts[1] * gridCounts[2];
    const int i = gridIdx / nyz;
    const int remainder = gridIdx % nyz;
    const int j = remainder / gridCounts[2];
    const int k = remainder % gridCounts[2];

    // Calculate grid point position (in nm)
    const float gx = originX + i * gridSpacing[0];
    const float gy = originY + j * gridSpacing[1];
    const float gz = originZ + k * gridSpacing[2];

    // Physics constants
    const float COULOMB_CONST = 138.935456f;  // kJ·nm/(mol·e²)
    const float U_MAX = gridCap;              // Configurable capping threshold

    // Calculate contribution from each receptor atom
    float gridValue = 0.0f;
    for (int atomIdx = 0; atomIdx < numReceptorAtoms; atomIdx++) {
        // Get atom position
        float3 atomPos = receptorPositions[atomIdx];

        // Calculate distance
        const float dx = gx - atomPos.x;
        const float dy = gy - atomPos.y;
        const float dz = gz - atomPos.z;
        const float r2 = dx*dx + dy*dy + dz*dz;
        float r = sqrtf(r2);

        // Avoid singularities at very small distances
        if (r < 1e-6f) {
            r = 1e-6f;
        }

        // Calculate contribution based on grid type
        if (gridType == 0) {
            // Electrostatic potential: k * q / r
            gridValue += COULOMB_CONST * receptorCharges[atomIdx] / r;
        } else if (gridType == 1) {
            // LJ repulsive: sqrt(epsilon) * Rmin^6 / r^12
            // where Rmin = 2^(1/6) * sigma (AMBER convention)
            const float rmin = powf(2.0f, 1.0f/6.0f) * receptorSigmas[atomIdx];
            const float r6 = rmin * rmin * rmin * rmin * rmin * rmin;
            const float r12 = r2 * r2 * r2 * r2 * r2 * r2;
            gridValue += sqrtf(receptorEpsilons[atomIdx]) * r6 / r12;
        } else if (gridType == 2) {
            // LJ attractive: -2 * sqrt(epsilon) * Rmin^3 / r^6
            // where Rmin = 2^(1/6) * sigma (AMBER convention)
            const float rmin = powf(2.0f, 1.0f/6.0f) * receptorSigmas[atomIdx];
            const float r3 = rmin * rmin * rmin;
            const float r6 = r2 * r2 * r2;
            gridValue += -2.0f * sqrtf(receptorEpsilons[atomIdx]) * r3 / r6;
        }
    }

    // Apply capping to avoid extreme values
    gridValue = U_MAX * tanhf(gridValue / U_MAX);

    // Apply inverse power transformation if specified
    // Grid should store G^(1/n), kernel will apply ^n to recover G
    if (invPower > 0.0f) {
        gridValue = powf(gridValue, 1.0f / invPower);
    }

    // Store result
    gridValues[gridIdx] = gridValue;
}

extern "C" __global__ void computeGridForce(
    const float4* __restrict__ posq,
    unsigned long long* __restrict__ forceBuffers,
    const int* __restrict__ gridCounts,
    const float* __restrict__ gridSpacing,
    const float* __restrict__ gridValues,
    const float* __restrict__ scalingFactors,
    const float invPower,
    const int interpolationMethod,  // 0=trilinear, 1=B-spline, 2=tricubic, 3=triquintic
    const float outOfBoundsK,
    const float originX,
    const float originY,
    const float originZ,
    const float* __restrict__ gridDerivatives,  // For triquintic: 27 derivatives per point
    float* __restrict__ energyBuffer,
    const int numAtoms,
    const int paddedNumAtoms) {

    // Get thread index
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= numAtoms)
        return;

    // Load atom position and scaling factor
    float4 posOrig = posq[index];
    float scalingFactor = scalingFactors[index];

    // Transform position to grid coordinates (relative to origin)
    float3 pos;
    pos.x = posOrig.x - originX;
    pos.y = posOrig.y - originY;
    pos.z = posOrig.z - originZ;

    // Initialize force to zero
    float3 atomForce = make_float3(0.0f, 0.0f, 0.0f);
    float threadEnergy = 0.0f;

    // Calculate grid boundaries
    float3 gridCorner;
    gridCorner.x = gridSpacing[0] * (gridCounts[0] - 1);
    gridCorner.y = gridSpacing[1] * (gridCounts[1] - 1);
    gridCorner.z = gridSpacing[2] * (gridCounts[2] - 1);

    // Check if the atom is inside the grid
    bool isInside = (pos.x >= 0.0f && pos.x <= gridCorner.x &&
                    pos.y >= 0.0f && pos.y <= gridCorner.y &&
                    pos.z >= 0.0f && pos.z <= gridCorner.z);

    if (isInside && scalingFactor != 0.0f) {
        // Calculate grid indices
        int ix = min(max((int)(pos.x / gridSpacing[0]), 0), gridCounts[0] - 2);
        int iy = min(max((int)(pos.y / gridSpacing[1]), 0), gridCounts[1] - 2);
        int iz = min(max((int)(pos.z / gridSpacing[2]), 0), gridCounts[2] - 2);

        // Calculate fractional position within the cell
        float fx = (pos.x / gridSpacing[0]) - ix;
        float fy = (pos.y / gridSpacing[1]) - iy;
        float fz = (pos.z / gridSpacing[2]) - iz;

        fx = min(max(fx, 0.0f), 1.0f);
        fy = min(max(fy, 0.0f), 1.0f);
        fz = min(max(fz, 0.0f), 1.0f);

        // Declare variables for interpolation
        float interpolated = 0.0f;
        float dx, dy, dz;
        int nyz = gridCounts[1] * gridCounts[2];

        if (interpolationMethod == 1) {
            // CUBIC B-SPLINE INTERPOLATION (4x4x4 = 64 points)
            // Precompute basis functions
            float bx[4] = {bspline_basis0(fx), bspline_basis1(fx), bspline_basis2(fx), bspline_basis3(fx)};
            float by[4] = {bspline_basis0(fy), bspline_basis1(fy), bspline_basis2(fy), bspline_basis3(fy)};
            float bz[4] = {bspline_basis0(fz), bspline_basis1(fz), bspline_basis2(fz), bspline_basis3(fz)};

            float dbx[4] = {bspline_deriv0(fx), bspline_deriv1(fx), bspline_deriv2(fx), bspline_deriv3(fx)};
            float dby[4] = {bspline_deriv0(fy), bspline_deriv1(fy), bspline_deriv2(fy), bspline_deriv3(fy)};
            float dbz[4] = {bspline_deriv0(fz), bspline_deriv1(fz), bspline_deriv2(fz), bspline_deriv3(fz)};

            float dvdx = 0.0f, dvdy = 0.0f, dvdz = 0.0f;

            // Tri-linear B-spline interpolation
            for (int i = 0; i < 4; i++) {
                int gx = min(max(ix - 1 + i, 0), gridCounts[0] - 1);
                for (int j = 0; j < 4; j++) {
                    int gy = min(max(iy - 1 + j, 0), gridCounts[1] - 1);
                    for (int k = 0; k < 4; k++) {
                        int gz = min(max(iz - 1 + k, 0), gridCounts[2] - 1);
                        int gridIdx = gx * nyz + gy * gridCounts[2] + gz;
                        float val = gridValues[gridIdx];
                        float weight = bx[i] * by[j] * bz[k];
                        interpolated += weight * val;
                        dvdx += dbx[i] * by[j] * bz[k] * val;
                        dvdy += bx[i] * dby[j] * bz[k] * val;
                        dvdz += bx[i] * by[j] * dbz[k] * val;
                    }
                }
            }

            dx = dvdx / gridSpacing[0];
            dy = dvdy / gridSpacing[1];
            dz = dvdz / gridSpacing[2];

        } else if (interpolationMethod == 3 && gridDerivatives != nullptr) {
            // TRIQUINTIC HERMITE INTERPOLATION (requires precomputed derivatives)
            // NOTE: Works best with moderate grid caps (<100k kJ/mol, <5% of grid at cap).
            // At very high caps (>500k), capping artifacts cause polynomial oscillations.
            // Gather 216 derivative values (27 derivatives × 8 corners)
            int totalPoints = gridCounts[0] * gridCounts[1] * gridCounts[2];
            int corners[8][3] = {
                {ix, iy, iz}, {ix+1, iy, iz}, {ix, iy+1, iz}, {ix+1, iy+1, iz},
                {ix, iy, iz+1}, {ix+1, iy, iz+1}, {ix, iy+1, iz+1}, {ix+1, iy+1, iz+1}
            };

            // Gather derivatives in layout: X[deriv_idx * 8 + corner_idx]
            float X[216];
            for (int d = 0; d < 27; d++) {
                for (int c = 0; c < 8; c++) {
                    int point_idx = corners[c][0] * nyz + corners[c][1] * gridCounts[2] + corners[c][2];
                    X[d * 8 + c] = gridDerivatives[d * totalPoints + point_idx];
                }
            }

            // Compute polynomial coefficients: a = 0.125 * TRIQUINTIC_COEFFICIENTS * X
            float a[216];
            const float scale = 0.125f;
            for (int i = 0; i < 216; i++) {
                a[i] = 0.0f;
                for (int j = 0; j < 216; j++) {
                    a[i] += TRIQUINTIC_COEFFICIENTS[i][j] * X[j];
                }
                a[i] *= scale;
            }

            // Precompute powers of local coordinates
            float sx_pow[6], sy_pow[6], sz_pow[6];
            sx_pow[0] = sy_pow[0] = sz_pow[0] = 1.0f;
            for (int p = 1; p < 6; p++) {
                sx_pow[p] = sx_pow[p-1] * fx;
                sy_pow[p] = sy_pow[p-1] * fy;
                sz_pow[p] = sz_pow[p-1] * fz;
            }

            // Evaluate polynomial: sum over i,j,k of a[i+6j+36k] * fx^i * fy^j * fz^k
            float value = 0.0f;
            float dvalue_dx = 0.0f, dvalue_dy = 0.0f, dvalue_dz = 0.0f;

            for (int k = 0; k < 6; k++) {
                for (int j = 0; j < 6; j++) {
                    for (int i = 0; i < 6; i++) {
                        int coeff_idx = i + 6*j + 36*k;
                        float coeff = a[coeff_idx];
                        value += coeff * sx_pow[i] * sy_pow[j] * sz_pow[k];
                        if (i > 0) dvalue_dx += coeff * i * sx_pow[i-1] * sy_pow[j] * sz_pow[k];
                        if (j > 0) dvalue_dy += coeff * j * sx_pow[i] * sy_pow[j-1] * sz_pow[k];
                        if (k > 0) dvalue_dz += coeff * k * sx_pow[i] * sy_pow[j] * sz_pow[k-1];
                    }
                }
            }

            interpolated = value;
            // Convert gradients from cell-local [0,1] to physical coordinates
            dx = dvalue_dx / gridSpacing[0];
            dy = dvalue_dy / gridSpacing[1];
            dz = dvalue_dz / gridSpacing[2];

        } else {
            // TRILINEAR INTERPOLATION (default, 2x2x2 = 8 points)
            float ox = 1.0f - fx;
            float oy = 1.0f - fy;
            float oz = 1.0f - fz;

            int baseIndex = ix * nyz + iy * gridCounts[2] + iz;
            int ip = baseIndex + nyz;           // ix+1
            int imp = baseIndex + gridCounts[2]; // iy+1
            int ipp = ip + gridCounts[2];       // ix+1, iy+1

            // Get grid values
            float vmmm = gridValues[baseIndex];
            float vmmp = gridValues[baseIndex + 1];
            float vmpm = gridValues[imp];
            float vmpp = gridValues[imp + 1];
            float vpmm = gridValues[ip];
            float vpmp = gridValues[ip + 1];
            float vppm = gridValues[ipp];
            float vppp = gridValues[ipp + 1];

            // Perform trilinear interpolation
            float vmm = oz * vmmm + fz * vmmp;
            float vmp = oz * vmpm + fz * vmpp;
            float vpm = oz * vpmm + fz * vpmp;
            float vpp = oz * vppm + fz * vppp;

            float vm = oy * vmm + fy * vmp;
            float vp = oy * vpm + fy * vpp;

            interpolated = ox * vm + fx * vp;

            // Calculate forces (gradients)
            dx = (vp - vm) / gridSpacing[0];
            dy = (ox * (vmp - vmm) + fx * (vpp - vpm)) / gridSpacing[1];
            dz = (ox * (oy * (vmmp - vmmm) + fy * (vmpp - vmpm)) +
                   fx * (oy * (vpmp - vpmm) + fy * (vppp - vppm))) / gridSpacing[2];
        }

        // Apply inverse power transformation if specified
        if (invPower > 0.0f) {
            float powerFactor = invPower * powf(interpolated, invPower - 1.0f);
            interpolated = powf(interpolated, invPower);
            dx *= powerFactor;
            dy *= powerFactor;
            dz *= powerFactor;
        }

        threadEnergy = scalingFactor * interpolated;

        atomForce.x = -scalingFactor * dx;
        atomForce.y = -scalingFactor * dy;
        atomForce.z = -scalingFactor * dz;

        // Debug: print force calculation for atom 0 only (critical for comparison)
        // if (index == 0) {
        //     printf("  A0: force=(%.6f, %.6f, %.6f) | gradients=(%.6f, %.6f, %.6f) | scale=%.9f\n",
        //            atomForce.x, atomForce.y, atomForce.z, dx, dy, dz, scalingFactor);
        // }
    }
    else {
        // Apply harmonic restraint outside grid (if enabled)
        // NOTE: This restraint is NOT scaled by scalingFactor - it applies uniformly
        // to all particles to keep them within the grid boundaries
        float3 dev = make_float3(0.0f, 0.0f, 0.0f);

        if (pos.x < 0.0f)
            dev.x = pos.x;
        else if (pos.x > gridCorner.x)
            dev.x = pos.x - gridCorner.x;

        if (pos.y < 0.0f)
            dev.y = pos.y;
        else if (pos.y > gridCorner.y)
            dev.y = pos.y - gridCorner.y;

        if (pos.z < 0.0f)
            dev.z = pos.z;
        else if (pos.z > gridCorner.z)
            dev.z = pos.z - gridCorner.z;

        threadEnergy = 0.5f * outOfBoundsK * (dev.x * dev.x + dev.y * dev.y + dev.z * dev.z);
        atomForce.x = -outOfBoundsK * dev.x;  // Don't scale the out-of-bounds restraint!
        atomForce.y = -outOfBoundsK * dev.y;  // Don't scale the out-of-bounds restraint!
        atomForce.z = -outOfBoundsK * dev.z;  // Don't scale the out-of-bounds restraint!
    }

    // Store forces using atomicAdd with unsigned long long
    // IMPORTANT: Must cast to signed long long first to preserve sign bit!
    unsigned long long fx_fixed = (unsigned long long)((long long)(atomForce.x * 0x100000000));
    unsigned long long fy_fixed = (unsigned long long)((long long)(atomForce.y * 0x100000000));
    unsigned long long fz_fixed = (unsigned long long)((long long)(atomForce.z * 0x100000000));

    // if (index == 0) {
    //     printf("  atomicAdd: indices=(%d, %d, %d) | fixed=(%llu, %llu, %llu)\n",
    //            index, index + paddedNumAtoms, index + 2*paddedNumAtoms,
    //            fx_fixed, fy_fixed, fz_fixed);
    // }

    atomicAdd(&forceBuffers[index], fx_fixed);
    atomicAdd(&forceBuffers[index + paddedNumAtoms], fy_fixed);
    atomicAdd(&forceBuffers[index + 2 * paddedNumAtoms], fz_fixed);

    // Accumulate energy
    atomicAdd(&energyBuffer[0], threadEnergy);
}
