#ifndef REFERENCE_GRID_INTERPOLATION_H_
#define REFERENCE_GRID_INTERPOLATION_H_

/* -------------------------------------------------------------------------- *
 *                              OpenMMGridForce                               *
 * -------------------------------------------------------------------------- *
 * Shared interpolation utilities and position/force extraction helpers       *
 * for Reference platform kernel implementations.                            *
 * -------------------------------------------------------------------------- */

#include "openmm/Vec3.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/reference/ReferencePlatform.h"
#include "openmm/reference/SimTKOpenMMRealType.h"
#include "internal/HCTKernels.h"
#include <vector>
#include <cmath>

namespace GridForcePlugin {

// Position and force extraction from Reference platform context
inline std::vector<OpenMM::Vec3>& refExtractPositions(OpenMM::ContextImpl& context) {
    OpenMM::ReferencePlatform::PlatformData* data =
        reinterpret_cast<OpenMM::ReferencePlatform::PlatformData*>(context.getPlatformData());
    return *((std::vector<OpenMM::Vec3>*)data->positions);
}

inline std::vector<OpenMM::Vec3>& refExtractForces(OpenMM::ContextImpl& context) {
    OpenMM::ReferencePlatform::PlatformData* data =
        reinterpret_cast<OpenMM::ReferencePlatform::PlatformData*>(context.getPlatformData());
    return *((std::vector<OpenMM::Vec3>*)data->forces);
}

// ==================== HCT integral computation ====================
// computeHCTTerm and its r-derivatives live in the shared header so grid
// generation and pairwise evaluation use the same closed form.

// Coulomb constant in kJ*nm/(mol*e^2) — use OpenMM's exact definition
static constexpr double COULOMB_CONSTANT = ONE_4PI_EPS0;

}  // namespace GridForcePlugin

#endif /* REFERENCE_GRID_INTERPOLATION_H_ */
