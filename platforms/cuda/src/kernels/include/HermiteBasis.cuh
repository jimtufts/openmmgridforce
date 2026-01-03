/**
 * Cubic Hermite basis functions for tricubic interpolation.
 * These interpolate exactly through points while maintaining C1 continuity.
 */

#ifndef HERMITE_BASIS_CUH
#define HERMITE_BASIS_CUH

// Cubic Hermite basis functions
// h00, h01 are for function values; h10, h11 are for derivatives
__device__ inline float hermite_h00(float t) { return (1.0f + 2.0f*t) * (1.0f - t) * (1.0f - t); }  // Interpolates f(0)
__device__ inline float hermite_h10(float t) { return t * (1.0f - t) * (1.0f - t); }              // Scales f'(0)
__device__ inline float hermite_h01(float t) { return t * t * (3.0f - 2.0f*t); }                  // Interpolates f(1)
__device__ inline float hermite_h11(float t) { return t * t * (t - 1.0f); }                      // Scales f'(1)

// Derivatives of Hermite basis functions (for computing forces)
__device__ inline float hermite_dh00(float t) { return 6.0f*t*t - 6.0f*t; }
__device__ inline float hermite_dh10(float t) { return 3.0f*t*t - 4.0f*t + 1.0f; }
__device__ inline float hermite_dh01(float t) { return -6.0f*t*t + 6.0f*t; }
__device__ inline float hermite_dh11(float t) { return 3.0f*t*t - 2.0f*t; }

#endif // HERMITE_BASIS_CUH
