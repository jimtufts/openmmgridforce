/* -------------------------------------------------------------------------- *
 *                              OpenMMGridForce                               *
 * -------------------------------------------------------------------------- *
 * Offline precompute for the grid-based receptor terms of IsolatedGBSAForce
 * GRID mode: apo receptor Born radii, mirror linear-response weights, pocket
 * selection, and CPU generation of the COULOMB and MIRROR fields.
 *
 * Everything here is receptor-only and pose-independent, so it runs once when
 * the kernel initializes (or is loaded from file). The CUDA platform reuses
 * the apo/pocket helpers and replaces the field builders with kernels.
 *
 * Units follow the plugin: nm, kJ/mol, elementary charge. HCT integrals use
 * the plugin convention where psi = 0.5 * R_off * hct.
 * -------------------------------------------------------------------------- */

#ifndef OPENMM_SOLVATIONFIELDBUILDER_H_
#define OPENMM_SOLVATIONFIELDBUILDER_H_

#include "SolvationFieldGrid.h"
#include "internal/windowsExportGridForce.h"
#include <cmath>
#include <functional>
#include <memory>
#include <vector>

namespace GridForcePlugin {

namespace SolvationFields {

/** Default near/far split radii and near-shell cutoff, nm. */
static constexpr double DEFAULT_SWITCH_ON = 0.15;
static constexpr double DEFAULT_SWITCH_OFF = 0.35;
static constexpr double DEFAULT_NEAR_CUTOFF = 0.80;

/** Default padding beyond the grid box for the pocket set, nm. */
static constexpr double DEFAULT_POCKET_PADDING = 1.2;

/** Range over which receptor atoms contribute to the mirror field, nm. */
static constexpr double DEFAULT_MIRROR_BUILD_CUTOFF = 1.6;

/**
 * Smootherstep switch: 0 for r <= rOn, 1 for r >= rOff, C2 in between.
 * The generated far field carries a factor S; the runtime near shell carries
 * the complementary (1 - S), so their sum reproduces the untruncated term.
 */
double switchValue(double r, double rOn, double rOff);

/**
 * dS/dr for switchValue().
 */
double switchDerivative(double r, double rOn, double rOff);

/**
 * Receptor quantities at the apo (no-ligand) state.
 *
 * mirrorWeights[j] = (dE_receptor/dR_j) * (dR_j/dhct_j), evaluated at apo, so
 * that dE_mirror = sum_j mirrorWeights[j] * (ligand-induced HCT on atom j).
 * dE_receptor/dR_j includes the receptor self term, all receptor-receptor GB
 * pair terms, and the receptor ACE surface-area term when includeSurfaceArea
 * is set -- the same three contributions PAIRWISE mode reports as
 * getGroupReceptorDesolvation().
 *
 * @param positions      Receptor positions, [3*N], nm
 * @param charges        Receptor charges, [N]
 * @param radii          Receptor intrinsic radii, [N], nm
 * @param scaleFactors   Receptor HCT scale factors, [N]
 * @param useOBC         OBC-II Born transform if true, plain HCT otherwise
 * @param prefactor      -COULOMB_CONSTANT * (1/eps_in - 1/eps_out)
 * @param cutoffDistance Pair cutoff in nm, or <= 0 for none
 * @param includeSurfaceArea  Fold the receptor ACE response into the weights
 * @param surfaceTension kJ/mol/nm^2
 * @param suppliedBornRadii  Use these apo Born radii instead of deriving them
 *                       from the receptor parameters; pass nullptr to derive
 * @param hct            Output apo HCT integral per atom, [N], plugin
 *                       convention (psi = 0.5 * R_off * hct)
 * @param bornRadii      Output apo Born radii, [N], nm
 * @param mirrorWeights  Output weights, [N]
 * @param parallelFor    Runs body(i) for i in [0,count); may run concurrently
 */
void computeApoReceptor(const std::vector<double>& positions,
                        const std::vector<double>& charges,
                        const std::vector<double>& radii,
                        const std::vector<double>& scaleFactors,
                        bool useOBC, double prefactor, double cutoffDistance,
                        bool includeSurfaceArea, double surfaceTension,
                        const std::vector<double>* suppliedBornRadii,
                        std::vector<double>& hct,
                        std::vector<double>& bornRadii,
                        std::vector<double>& mirrorWeights,
                        const std::function<void(int, const std::function<void(int)>&)>&
                            parallelFor = nullptr);

/**
 * Receptor atoms within `padding` of the axis-aligned grid box. These are the
 * only atoms the runtime near lists need to visit.
 */
std::vector<int> selectPocketAtoms(const std::vector<double>& positions,
                                   const double origin[3], double spacing,
                                   const int counts[3], double padding);

/**
 * Uniform bucket list over the pocket atoms, built once because the receptor
 * is rigid. A near-shell loop that scanned every pocket atom would cost as
 * much as the exact sum it replaces, so the near terms query this instead.
 */
struct PocketCellList {
    double origin[3];
    double cellSize;
    int counts[3];
    std::vector<int> cellStart;    // [nCells + 1] prefix offsets into atoms
    std::vector<int> atoms;        // receptor atom indices, bucketed by cell
};

/**
 * Cells per near-shell cutoff. At one cell per cutoff a 3x3x3 block spans
 * 6.4x the volume of the sphere it stands in for, so most of what the near
 * loops touch is rejected by distance. Subdividing tightens that to about 3x
 * for a modest number of extra (contiguous, well-cached) cell reads.
 */
static constexpr int CELLS_PER_CUTOFF = 2;

/**
 * @param cellSize  Edge length (nm); pass cutoff / CELLS_PER_CUTOFF
 */
void buildPocketCellList(const std::vector<double>& positions,
                         const std::vector<int>& atomIndices,
                         double cellSize, PocketCellList& out);

/**
 * Invoke body(receptorAtomIndex) for every pocket atom in the cells that can
 * hold a point within `cutoff` of (x,y,z). Callers still apply their own
 * distance test; this only bounds which cells are worth visiting.
 */
template <typename Body>
inline void forEachNearPocketAtom(const PocketCellList& cl,
                                  double x, double y, double z, double cutoff,
                                  Body body) {
    if (cl.atoms.empty())
        return;
    double inv = 1.0 / cl.cellSize;
    const double p0[3] = {x, y, z};
    int lo[3], hi[3];
    for (int d = 0; d < 3; d++) {
        lo[d] = static_cast<int>(std::floor((p0[d] - cutoff - cl.origin[d]) * inv));
        hi[d] = static_cast<int>(std::floor((p0[d] + cutoff - cl.origin[d]) * inv));
        if (lo[d] < 0) lo[d] = 0;
        if (hi[d] > cl.counts[d] - 1) hi[d] = cl.counts[d] - 1;
    }
    for (int ix = lo[0]; ix <= hi[0]; ix++)
        for (int iy = lo[1]; iy <= hi[1]; iy++) {
            int base = (ix * cl.counts[1] + iy) * cl.counts[2];
            for (int iz = lo[2]; iz <= hi[2]; iz++) {
                int cell = base + iz;
                for (int p = cl.cellStart[cell]; p < cl.cellStart[cell + 1]; p++)
                    body(cl.atoms[p]);
            }
        }
}

/**
 * Distinct descreener scaled radii s = (radius - offset) * scaleFactor of a
 * ligand template, sorted ascending and deduplicated. One MIRROR slice per
 * value reproduces the template exactly, with no binning error.
 *
 * @param tolerance  Values closer than this (nm) collapse to one slice
 */
std::vector<double> distinctScaledRadii(const std::vector<double>& radii,
                                        const std::vector<double>& scaleFactors,
                                        double tolerance = 1e-6);

/** Slice index in `sliceValues` for a given scaled radius. */
int sliceForScaledRadius(const std::vector<double>& sliceValues, double s);

/**
 * Probe Born radii to slice the cross field at: log-spaced between the
 * smallest offset radius in the template (an isolated atom) and the largest
 * OBC2 ceiling R_off * rho / OFFSET (a fully buried one). Both bounds are
 * template properties, so no pose sampling is needed to place the slices.
 */
std::vector<double> defaultCrossFieldRadii(const std::vector<double>& radii,
                                           int numSlices);

/**
 * Build Phi_k(x) = sum_j q_j * S(|x-r_j|) / f_GB(|x-r_j|, R_k, R_apo_j), one
 * slice per entry of sliceValues. Uses every atom passed in: the field decays
 * like 1/r, too slowly to truncate at the pocket boundary, and the build is
 * one-time.
 *
 * @param bornRadiiApo  Apo receptor Born radii, [N], nm
 * @param interpMethod  TRILINEAR or TRICUBIC_BSPLINE; the latter prefilters
 *                      the node values into B-spline coefficients in place
 */
std::shared_ptr<SolvationFieldGrid> buildCrossField(
        const std::vector<double>& positions, const std::vector<double>& charges,
        const std::vector<double>& bornRadiiApo,
        const std::vector<int>& atomIndices,
        const std::vector<double>& sliceValues,
        const double origin[3], double spacing, const int counts[3],
        double switchOn, double switchOff, int interpMethod,
        const std::function<void(int, const std::function<void(int)>&)>&
            parallelFor = nullptr);

/**
 * Build Psi_b(x) = sum_j w_j * H(|x-r_j|, R_j_off, s_b) * S(|x-r_j|), one
 * slice per entry of sliceValues.
 *
 * @param buildCutoff  Skip receptor atoms farther than this from a grid point
 */
std::shared_ptr<SolvationFieldGrid> buildMirrorField(
        const std::vector<double>& positions, const std::vector<double>& radii,
        const std::vector<double>& mirrorWeights,
        const std::vector<int>& atomIndices,
        const std::vector<double>& sliceValues,
        const double origin[3], double spacing, const int counts[3],
        double switchOn, double switchOff, double buildCutoff, int interpMethod,
        const std::function<void(int, const std::function<void(int)>&)>&
            parallelFor = nullptr);

/**
 * Sample a field slice at (x,y,z), optionally with its gradient. Returns 0 and
 * a zero gradient when the position falls outside the interpolable region.
 *
 * @param interpMethod  Must match grid.getInterpolationMethod()
 */
double interpolateField(const SolvationFieldGrid& grid, int slice,
                        double x, double y, double z, int interpMethod,
                        bool computeGrad,
                        double& gradX, double& gradY, double& gradZ);

}  // namespace SolvationFields

}  // namespace GridForcePlugin

#endif  // OPENMM_SOLVATIONFIELDBUILDER_H_
