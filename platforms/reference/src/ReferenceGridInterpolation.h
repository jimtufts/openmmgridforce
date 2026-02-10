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

// Dielectric offset constant for GBSA Born radii (nm)
static constexpr double DIELECTRIC_OFFSET = 0.009;

// Coulomb constant in kJ*nm/(mol*e^2)
static constexpr double COULOMB_CONSTANT = 138.935456;

/**
 * Compute the HCT integral contribution from atom j to atom i.
 *
 * @param r           Distance between atoms i and j (nm)
 * @param R_i_off     Offset radius of atom i: radius_i - DIELECTRIC_OFFSET (nm)
 * @param R_j_off     Offset radius of atom j: radius_j - DIELECTRIC_OFFSET (nm)
 * @param scaleFactor_j  HCT scale factor for atom j
 * @return HCT integral contribution
 */
inline double computeHCTTerm(double r, double R_i_off, double R_j_off, double scaleFactor_j) {
    double S_j = R_j_off * scaleFactor_j;

    // Check if atom j is too far to contribute
    if (R_i_off >= r + S_j)
        return 0.0;

    double r_minus_Sj = std::abs(r - S_j);
    double r_plus_Sj = r + S_j;

    double l_ij = (R_i_off > r_minus_Sj) ? (1.0 / R_i_off) : (1.0 / r_minus_Sj);
    double u_ij = 1.0 / r_plus_Sj;

    double l_ij2 = l_ij * l_ij;
    double u_ij2 = u_ij * u_ij;
    double r_inv = 1.0 / r;

    double term = l_ij - u_ij
                + 0.25 * r * (u_ij2 - l_ij2)
                + 0.5 * r_inv * std::log(u_ij / l_ij)
                + 0.25 * S_j * S_j * r_inv * (l_ij2 - u_ij2);

    // Tinker correction: atom i fully inside atom j
    if (R_i_off < (S_j - r)) {
        term += 2.0 * (1.0 / R_i_off - l_ij);
    }

    return term;
}

/**
 * Compute derivative of HCT integral w.r.t. distance r.
 * Used for chain rule force computation in GBSA.
 *
 * @param r           Distance between atoms i and j (nm)
 * @param R_i_off     Offset radius of atom i (nm)
 * @param R_j_off     Offset radius of atom j (nm)
 * @param scaleFactor_j  HCT scale factor for atom j
 * @return dHCT/dr
 */
inline double computeHCTTermDerivative(double r, double R_i_off, double R_j_off, double scaleFactor_j) {
    double S_j = R_j_off * scaleFactor_j;

    if (R_i_off >= r + S_j)
        return 0.0;

    double r_minus_Sj = std::abs(r - S_j);
    double r_plus_Sj = r + S_j;

    double l_ij = (R_i_off > r_minus_Sj) ? (1.0 / R_i_off) : (1.0 / r_minus_Sj);
    double u_ij = 1.0 / r_plus_Sj;

    double l_ij2 = l_ij * l_ij;
    double u_ij2 = u_ij * u_ij;
    double r_inv = 1.0 / r;
    double r_inv2 = r_inv * r_inv;

    // dl/dr and du/dr
    double dl_dr, du_dr;
    du_dr = -u_ij * u_ij;  // d/dr (1/(r+S)) = -1/(r+S)^2

    if (R_i_off > r_minus_Sj) {
        dl_dr = 0.0;  // l = 1/R_i_off, independent of r
    } else {
        // l = 1/|r - S_j|
        if (r > S_j)
            dl_dr = -l_ij * l_ij;   // d/dr (1/(r-S)) = -1/(r-S)^2
        else
            dl_dr = l_ij * l_ij;    // d/dr (1/(S-r)) = 1/(S-r)^2
    }

    double dterm_dr = dl_dr - du_dr
                    + 0.25 * (u_ij2 - l_ij2)
                    + 0.25 * r * (2.0 * u_ij * du_dr - 2.0 * l_ij * dl_dr)
                    - 0.5 * r_inv2 * std::log(u_ij / l_ij)
                    + 0.5 * r_inv * (du_dr / u_ij - dl_dr / l_ij)
                    - 0.25 * S_j * S_j * r_inv2 * (l_ij2 - u_ij2)
                    + 0.25 * S_j * S_j * r_inv * (2.0 * l_ij * dl_dr - 2.0 * u_ij * du_dr);

    return dterm_dr;
}

}  // namespace GridForcePlugin

#endif /* REFERENCE_GRID_INTERPOLATION_H_ */
