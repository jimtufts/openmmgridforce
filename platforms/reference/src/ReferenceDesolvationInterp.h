#ifndef REFERENCE_DESOLVATION_INTERP_H_
#define REFERENCE_DESOLVATION_INTERP_H_

/* -------------------------------------------------------------------------- *
 *                              OpenMMGridForce                               *
 * -------------------------------------------------------------------------- *
 * Shared free function that interpolates the DesolvationGrid HCT probe +     *
 * N/A/B corrections for any interpolation method (0=trilinear, 1=cubic       *
 * B-spline, 2=tricubic Hermite, 3=triquintic Hermite). Used by both the      *
 * GBSAGridForce Reference kernel and the IsolatedGBSAForce GRID-mode         *
 * Reference kernel so LocalEnergyMinimizer (which spawns a Reference helper  *
 * context for line search) works for any interp method.                     *
 * -------------------------------------------------------------------------- */

namespace GridForcePlugin {

class DesolvationGrid;

/**
 * Interpolate the receptor HCT contribution at position (x,y,z) for a ligand
 * atom with offset radius R_i_off.  Handles all four interpolation methods
 * against the DesolvationGrid layout used by both GBSAGridForce and
 * IsolatedGBSAForce GRID-mode receptor descreening.
 *
 * @param grid            DesolvationGrid to sample.
 * @param x,y,z           Query position in nm.
 * @param R_i_off         Offset Born radius of the query atom: R_i - offset.
 * @param interpMethod    0=trilinear, 1=cubic B-spline, 2=tricubic Hermite,
 *                        3=triquintic Hermite. Methods 2/3 require the grid
 *                        to carry derivatives (hasDerivatives()); they fall
 *                        through to trilinear when it does not.
 * @param probeRadius     Grid probe radius (nm) — needed for the N/A/B
 *                        correction formula.
 * @param computeGrad     If true, populate gradX/Y/Z with dhct/d{x,y,z}.
 * @param gradX/Y/Z       Output gradient (nm^-1 units multiplied by the
 *                        stored HCT dimensions).
 * @param hess            Optional 6-element Voigt Hessian buffer
 *                        [xx,yy,zz,xy,xz,yz]; trilinear-only, mixed-partials
 *                        only. Pass nullptr to skip.
 * @param outOfBounds     Optional flag set when (x,y,z) is outside the grid
 *                        cell range. Pass nullptr to skip.
 * @return                Interpolated HCT contribution.
 */
double interpolateDesolvationGridHCT(
        const DesolvationGrid* grid,
        double x, double y, double z,
        double R_i_off,
        int interpMethod,
        double probeRadius,
        bool computeGrad,
        double& gradX, double& gradY, double& gradZ,
        double* hess,
        bool* outOfBounds);

}  // namespace GridForcePlugin

#endif  // REFERENCE_DESOLVATION_INTERP_H_
