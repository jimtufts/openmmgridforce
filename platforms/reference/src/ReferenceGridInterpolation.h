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

// Coulomb constant in kJ*nm/(mol*e^2) — use OpenMM's exact definition
static constexpr double COULOMB_CONSTANT = ONE_4PI_EPS0;

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

/**
 * Compute second derivative of HCT integral w.r.t. distance r.
 * Used for analytical Hessian computation in GBSA. Differentiates the
 * same closed form as computeHCTTermDerivative once more in r.
 *
 * @param r           Distance between atoms i and j (nm)
 * @param R_i_off     Offset radius of atom i (nm)
 * @param R_j_off     Offset radius of atom j (nm)
 * @param scaleFactor_j  HCT scale factor for atom j
 * @return d2HCT/dr2
 */
inline double computeHCTTermSecondDerivative(double r, double R_i_off, double R_j_off, double scaleFactor_j) {
    double S_j = R_j_off * scaleFactor_j;

    if (R_i_off >= r + S_j)
        return 0.0;

    double r_minus_Sj = std::abs(r - S_j);
    double r_plus_Sj = r + S_j;

    bool l_fixed = (R_i_off > r_minus_Sj);
    double l_ij = l_fixed ? (1.0 / R_i_off) : (1.0 / r_minus_Sj);
    double u_ij = 1.0 / r_plus_Sj;

    double l2 = l_ij * l_ij;
    double u2 = u_ij * u_ij;
    double l3 = l2 * l_ij;
    double u3 = u2 * u_ij;
    double r_inv = 1.0 / r;
    double r_inv2 = r_inv * r_inv;
    double r_inv3 = r_inv2 * r_inv;

    // First and second r-derivatives of l and u.
    double du = -u2;          // d/dr (1/(r+S))
    double d2u = 2.0 * u3;    // d2/dr2 (1/(r+S))
    double dl, d2l;
    if (l_fixed) {
        dl = 0.0;
        d2l = 0.0;
    } else if (r > S_j) {
        dl = -l2;            // d/dr (1/(r-S))
        d2l = 2.0 * l3;
    } else {
        dl = l2;             // d/dr (1/(S-r))
        d2l = 2.0 * l3;
    }

    double S2 = S_j * S_j;
    double log_ul = std::log(u_ij / l_ij);

    // Derivatives of the ratio terms du/u and dl/l (appear in the log term).
    double duu = du / u_ij;
    double dll = (l_fixed) ? 0.0 : (dl / l_ij);
    double d_duu = d2u / u_ij - duu * duu;   // d/dr (du/u)
    double d_dll = l_fixed ? 0.0 : (d2l / l_ij - dll * dll);

    // d2(term)/dr2 = derivative of dterm_dr (see computeHCTTermDerivative).
    double d2 = d2l - d2u
              + 0.25 * (2.0 * u_ij * du - 2.0 * l_ij * dl)
              // d/dr of 0.25*r*(u^2-l^2)' :
              + 0.25 * (2.0 * u_ij * du - 2.0 * l_ij * dl)
              + 0.25 * r * (2.0 * du * du + 2.0 * u_ij * d2u
                            - 2.0 * dl * dl - 2.0 * l_ij * d2l)
              // d/dr of -0.5*r^-2*log(u/l) :
              + 1.0 * r_inv3 * log_ul
              - 0.5 * r_inv2 * (duu - dll)
              // d/dr of 0.5*r^-1*(du/u - dl/l) :
              - 0.5 * r_inv2 * (duu - dll)
              + 0.5 * r_inv * (d_duu - d_dll)
              // d/dr of -0.25*S^2*r^-2*(l^2-u^2) :
              + 0.5 * S2 * r_inv3 * (l2 - u2)
              - 0.25 * S2 * r_inv2 * (2.0 * l_ij * dl - 2.0 * u_ij * du)
              // d/dr of 0.25*S^2*r^-1*(2 l dl - 2 u du) :
              - 0.25 * S2 * r_inv2 * (2.0 * l_ij * dl - 2.0 * u_ij * du)
              + 0.25 * S2 * r_inv * (2.0 * dl * dl + 2.0 * l_ij * d2l
                                     - 2.0 * du * du - 2.0 * u_ij * d2u);

    // Tinker correction: atom i fully inside atom j.
    if (R_i_off < (S_j - r)) {
        d2 += -2.0 * d2l;
    }

    return d2;
}

}  // namespace GridForcePlugin

#endif /* REFERENCE_GRID_INTERPOLATION_H_ */
