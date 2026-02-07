#ifndef OPENMM_GRIDFORCE_TYPES_H_
#define OPENMM_GRIDFORCE_TYPES_H_

namespace GridForcePlugin {

/**
 * Inverse power transformation mode.
 * Controls how and when the inv_power transformation is applied to grid values.
 */
enum class InvPowerMode {
    /**
     * No transformation applied.
     * Grid values are used as-is, and no power transformation occurs during evaluation.
     */
    NONE = 0,

    /**
     * Transform grid values at initialization/runtime.
     * Grid values are transformed G -> G^(1/n) once after loading, before evaluation.
     * The evaluation kernel then applies ^n to recover original values.
     * Only valid for grids WITHOUT analytical derivatives.
     */
    RUNTIME = 1,

    /**
     * Grid values already have transformation stored.
     * Grid values are already G^(1/n) (from generation or prior transformation).
     * The evaluation kernel applies ^n to recover original values.
     * Compatible with analytical derivatives.
     */
    STORED = 2
};

/**
 * Interpolation method identifiers.
 * These can be passed to setInterpolationMethod() / getInterpolationMethod().
 * Defined as plain integer constants so existing code using bare ints still works.
 */
namespace InterpolationMethod {
    constexpr int TRILINEAR           = 0;  ///< Trilinear interpolation (requires only grid values)
    constexpr int TRICUBIC_BSPLINE    = 1;  ///< Tricubic B-spline (requires prefiltered coefficients for interpolating mode)
    constexpr int TRICUBIC_HERMITE    = 2;  ///< Tricubic Hermite (requires stored analytical derivatives)
    constexpr int TRIQUINTIC_HERMITE  = 3;  ///< Triquintic Hermite (requires stored analytical derivatives)
    constexpr int TRIQUINTIC_BSPLINE  = 4;  ///< Triquintic B-spline (requires prefiltered coefficients for interpolating mode)
}

}  // namespace GridForcePlugin

#endif /*OPENMM_GRIDFORCE_TYPES_H_*/
