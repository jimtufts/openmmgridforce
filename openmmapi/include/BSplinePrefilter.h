#ifndef OPENMM_BSPLINE_PREFILTER_H_
#define OPENMM_BSPLINE_PREFILTER_H_

#include <vector>
#include <algorithm>
#include <stdexcept>

/**
 * B-spline prefilters for converting approximating B-splines to interpolating ones.
 *
 * === Cubic (degree 3) ===
 *
 * At grid node n (t=0), the cubic B-spline evaluates to:
 *     S(n) = (1/6)*c[n-1] + (4/6)*c[n] + (1/6)*c[n+1]
 *
 * Interpolation condition S(n) = f(n) gives the tridiagonal system:
 *     c[i-1] + 4*c[i] + c[i+1] = 6*f[i]
 *
 * Boundary conditions match CUDA kernel index clamping min(max(idx, 0), N-1):
 *     Row 0:   5*c[0]   + c[1]            = 6*f[0]
 *     Row N-1: c[N-2]   + 5*c[N-1]        = 6*f[N-1]
 *
 * === Quintic (degree 5) ===
 *
 * At grid node n (t=0), the quintic B-spline evaluates to:
 *     S(n) = (1/120)*c[n-2] + (26/120)*c[n-1] + (66/120)*c[n]
 *          + (26/120)*c[n+1] + (1/120)*c[n+2]
 *
 * Interpolation condition gives the pentadiagonal system:
 *     c[i-2] + 26*c[i-1] + 66*c[i] + 26*c[i+1] + c[i+2] = 120*f[i]
 *
 * Boundary conditions from clamping:
 *     Row 0:   93*c[0]  + 26*c[1] + c[2]                   = 120*f[0]
 *     Row 1:   27*c[0]  + 66*c[1] + 26*c[2] + c[3]         = 120*f[1]
 *     Row N-2: c[N-4] + 26*c[N-3] + 66*c[N-2] + 27*c[N-1]  = 120*f[N-2]
 *     Row N-1: c[N-3] + 26*c[N-2] + 93*c[N-1]               = 120*f[N-1]
 *
 * Both prefilters are separable in 3D (tensor product property).
 *
 * Header-only implementation -- no additional .cpp or CMake changes needed.
 */

namespace GridForcePlugin {

// ============================================================================
// Cubic B-spline prefilter (tridiagonal, Thomas algorithm)
// ============================================================================

/**
 * Apply cubic B-spline prefilter to a 1D array in-place.
 * Solves the tridiagonal system using the Thomas algorithm. O(N) time.
 *
 * @param data   Pointer to the 1D data array (overwritten in-place with coefficients)
 * @param N      Number of elements
 * @param stride Stride between consecutive elements (default 1)
 */
template <typename T>
void bsplinePrefilter1D(T* data, int N, int stride = 1) {
    if (N < 2) return;

    // Record input range for post-solve clamping (prevents Gibbs ringing overshoot)
    T valMin = data[0];
    T valMax = data[0];
    for (int i = 1; i < N; i++) {
        T v = data[i * stride];
        if (v < valMin) valMin = v;
        if (v > valMax) valMax = v;
    }

    std::vector<T> d(N);
    std::vector<T> b(N);

    // Main diagonal [5, 4, 4, ..., 4, 5], off-diagonals all 1
    d[0] = static_cast<T>(5);
    b[0] = static_cast<T>(6) * data[0 * stride];
    for (int i = 1; i < N - 1; i++) {
        d[i] = static_cast<T>(4);
        b[i] = static_cast<T>(6) * data[i * stride];
    }
    d[N - 1] = static_cast<T>(5);
    b[N - 1] = static_cast<T>(6) * data[(N - 1) * stride];

    // Forward elimination
    for (int i = 1; i < N; i++) {
        T m = static_cast<T>(1) / d[i - 1];
        d[i] -= m;
        b[i] -= m * b[i - 1];
    }

    // Back substitution
    data[(N - 1) * stride] = b[N - 1] / d[N - 1];
    for (int i = N - 2; i >= 0; i--) {
        data[i * stride] = (b[i] - data[(i + 1) * stride]) / d[i];
    }

    // Clamp coefficients to input range to prevent overshoot
    for (int i = 0; i < N; i++) {
        T& c = data[i * stride];
        if (c < valMin) c = valMin;
        if (c > valMax) c = valMax;
    }
}

/**
 * Apply cubic B-spline prefilter separably to a 3D grid in-place.
 * Grid layout is row-major: values[ix * ny*nz + iy * nz + iz]
 */
template <typename T>
void bsplinePrefilter3D(std::vector<T>& values, int nx, int ny, int nz) {
    if (values.size() != static_cast<size_t>(nx) * ny * nz) return;
    int nyz = ny * nz;

    for (int iy = 0; iy < ny; iy++)
        for (int iz = 0; iz < nz; iz++)
            bsplinePrefilter1D(&values[iy * nz + iz], nx, nyz);

    for (int ix = 0; ix < nx; ix++)
        for (int iz = 0; iz < nz; iz++)
            bsplinePrefilter1D(&values[ix * nyz + iz], ny, nz);

    for (int ix = 0; ix < nx; ix++)
        for (int iy = 0; iy < ny; iy++)
            bsplinePrefilter1D(&values[ix * nyz + iy * nz], nz, 1);
}

// ============================================================================
// Quintic B-spline prefilter (pentadiagonal)
// ============================================================================

/**
 * Apply quintic B-spline prefilter to a 1D array in-place.
 *
 * Solves the pentadiagonal system with bandwidth 2 using banded LU
 * decomposition without pivoting.
 *
 * System (interior rows):
 *   1*c[i-2] + 26*c[i-1] + 66*c[i] + 26*c[i+1] + 1*c[i+2] = 120*f[i]
 *
 * Boundary rows (from index clamping):
 *   Row 0:   93*c[0]  + 26*c[1] + 1*c[2]                   = 120*f[0]
 *   Row 1:   27*c[0]  + 66*c[1] + 26*c[2] + 1*c[3]         = 120*f[1]
 *   Row N-2: 1*c[N-4] + 26*c[N-3] + 66*c[N-2] + 27*c[N-1]  = 120*f[N-2]
 *   Row N-1: 1*c[N-3] + 26*c[N-2] + 93*c[N-1]               = 120*f[N-1]
 *
 * @param data   Pointer to the 1D data array (overwritten in-place)
 * @param N      Number of elements (must be >= 3)
 * @param stride Stride between consecutive elements (default 1)
 */
template <typename T>
void quinticBsplinePrefilter1D(T* data, int N, int stride = 1) {
    if (N < 3) return;

    // Record input range for post-solve clamping (prevents Gibbs ringing overshoot)
    T valMin = data[0];
    T valMax = data[0];
    for (int i = 1; i < N; i++) {
        T v = data[i * stride];
        if (v < valMin) valMin = v;
        if (v > valMax) valMax = v;
    }

    // Build the banded system: 5 diagonals stored as arrays
    // e[i] = sub-sub-diagonal (band -2): A[i][i-2]
    // a[i] = sub-diagonal (band -1): A[i][i-1]
    // d[i] = main diagonal: A[i][i]
    // u[i] = super-diagonal (band +1): A[i][i+1]
    // v[i] = super-super-diagonal (band +2): A[i][i+2]
    std::vector<T> e(N, static_cast<T>(0));  // e[0], e[1] unused
    std::vector<T> a(N, static_cast<T>(0));  // a[0] unused
    std::vector<T> d(N);
    std::vector<T> u(N, static_cast<T>(0));  // u[N-1] unused
    std::vector<T> v(N, static_cast<T>(0));  // v[N-2], v[N-1] unused
    std::vector<T> b(N);

    // Initialize with clamping boundary conditions
    // Row 0: 93*c[0] + 26*c[1] + 1*c[2] = 120*f[0]
    d[0] = static_cast<T>(93);
    u[0] = static_cast<T>(26);
    v[0] = static_cast<T>(1);
    b[0] = static_cast<T>(120) * data[0 * stride];

    // Row 1: 27*c[0] + 66*c[1] + 26*c[2] + 1*c[3] = 120*f[1]
    a[1] = static_cast<T>(27);
    d[1] = static_cast<T>(66);
    u[1] = static_cast<T>(26);
    if (N > 3) v[1] = static_cast<T>(1);
    b[1] = static_cast<T>(120) * data[1 * stride];

    // Interior rows: 1*c[i-2] + 26*c[i-1] + 66*c[i] + 26*c[i+1] + 1*c[i+2]
    for (int i = 2; i < N - 2; i++) {
        e[i] = static_cast<T>(1);
        a[i] = static_cast<T>(26);
        d[i] = static_cast<T>(66);
        u[i] = static_cast<T>(26);
        v[i] = static_cast<T>(1);
        b[i] = static_cast<T>(120) * data[i * stride];
    }

    // Row N-2: 1*c[N-4] + 26*c[N-3] + 66*c[N-2] + 27*c[N-1]
    if (N > 3) e[N - 2] = static_cast<T>(1);
    a[N - 2] = static_cast<T>(26);
    d[N - 2] = static_cast<T>(66);
    u[N - 2] = static_cast<T>(27);
    b[N - 2] = static_cast<T>(120) * data[(N - 2) * stride];

    // Row N-1: 1*c[N-3] + 26*c[N-2] + 93*c[N-1]
    e[N - 1] = static_cast<T>(1);
    a[N - 1] = static_cast<T>(26);
    d[N - 1] = static_cast<T>(93);
    b[N - 1] = static_cast<T>(120) * data[(N - 1) * stride];

    // Forward elimination (banded LU, no pivoting)
    // For each row i, eliminate the sub-diagonal (a[i]) and sub-sub-diagonal (e[i])
    for (int i = 1; i < N; i++) {
        // Eliminate sub-diagonal a[i] using row i-1
        if (a[i] != static_cast<T>(0)) {
            T m = a[i] / d[i - 1];
            a[i] = static_cast<T>(0);
            d[i] -= m * u[i - 1];
            if (i < N - 1) u[i] -= m * v[i - 1];
            b[i] -= m * b[i - 1];

            // The sub-sub-diagonal from row i-1's elimination may have
            // filled into a[i]'s position, but we handle e[i] below
        }

        // Eliminate sub-sub-diagonal e[i] using row i-2
        if (i >= 2 && e[i] != static_cast<T>(0)) {
            T m = e[i] / d[i - 2];
            e[i] = static_cast<T>(0);
            a[i] -= m * u[i - 2];  // This may create fill-in at a[i]
            d[i] -= m * v[i - 2];
            b[i] -= m * b[i - 2];

            // Now eliminate the new a[i] fill-in using row i-1
            if (a[i] != static_cast<T>(0)) {
                T m2 = a[i] / d[i - 1];
                a[i] = static_cast<T>(0);
                d[i] -= m2 * u[i - 1];
                if (i < N - 1) u[i] -= m2 * v[i - 1];
                b[i] -= m2 * b[i - 1];
            }
        }
    }

    // Back substitution
    data[(N - 1) * stride] = b[N - 1] / d[N - 1];
    if (N >= 2) {
        data[(N - 2) * stride] = (b[N - 2] - u[N - 2] * data[(N - 1) * stride]) / d[N - 2];
    }
    for (int i = N - 3; i >= 0; i--) {
        data[i * stride] = (b[i] - u[i] * data[(i + 1) * stride]
                                  - v[i] * data[(i + 2) * stride]) / d[i];
    }

    // Clamp coefficients to input range to prevent overshoot
    for (int i = 0; i < N; i++) {
        T& c = data[i * stride];
        if (c < valMin) c = valMin;
        if (c > valMax) c = valMax;
    }
}

/**
 * Apply quintic B-spline prefilter separably to a 3D grid in-place.
 * Grid layout is row-major: values[ix * ny*nz + iy * nz + iz]
 */
template <typename T>
void quinticBsplinePrefilter3D(std::vector<T>& values, int nx, int ny, int nz) {
    if (values.size() != static_cast<size_t>(nx) * ny * nz) return;
    int nyz = ny * nz;

    for (int iy = 0; iy < ny; iy++)
        for (int iz = 0; iz < nz; iz++)
            quinticBsplinePrefilter1D(&values[iy * nz + iz], nx, nyz);

    for (int ix = 0; ix < nx; ix++)
        for (int iz = 0; iz < nz; iz++)
            quinticBsplinePrefilter1D(&values[ix * nyz + iz], ny, nz);

    for (int ix = 0; ix < nx; ix++)
        for (int iy = 0; iy < ny; iy++)
            quinticBsplinePrefilter1D(&values[ix * nyz + iy * nz], nz, 1);
}

// ============================================================================
// Dispatch function: apply prefilter based on B-spline order
// ============================================================================

/**
 * Apply B-spline prefilter to a 1D array based on order.
 * @param order  B-spline degree: 3 (cubic) or 5 (quintic)
 */
template <typename T>
void bsplinePrefilter1DByOrder(T* data, int N, int order, int stride = 1) {
    if (order == 3)
        bsplinePrefilter1D(data, N, stride);
    else if (order == 5)
        quinticBsplinePrefilter1D(data, N, stride);
}

/**
 * Apply B-spline prefilter to a 3D grid based on order.
 * @param order  B-spline degree: 3 (cubic) or 5 (quintic)
 */
template <typename T>
void bsplinePrefilter3DByOrder(std::vector<T>& values, int nx, int ny, int nz, int order) {
    if (order == 3)
        bsplinePrefilter3D(values, nx, ny, nz);
    else if (order == 5)
        quinticBsplinePrefilter3D(values, nx, ny, nz);
}

} // namespace GridForcePlugin

#endif // OPENMM_BSPLINE_PREFILTER_H_
