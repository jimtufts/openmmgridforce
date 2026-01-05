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

}  // namespace GridForcePlugin

#endif /*OPENMM_GRIDFORCE_TYPES_H_*/
