/* -------------------------------------------------------------------------- *
 *                              OpenMMGridForce                               *
 * -------------------------------------------------------------------------- */

#include "internal/SolvationFieldBuilder.h"
#include "internal/HCTKernels.h"
#include "BSplinePrefilter.h"
#include "GridForceTypes.h"
#include "openmm/OpenMMException.h"

#include <algorithm>
#include <cmath>

using namespace GridForcePlugin;
using namespace OpenMM;
using namespace std;

namespace {

// OBC-II parameters, matching IsolatedGBSAForce.
constexpr double OBC_ALPHA = 1.0;
constexpr double OBC_BETA = 0.8;
constexpr double OBC_GAMMA = 4.85;

// Probe radius of the receptor ACE surface-area term, nm.
constexpr double SA_PROBE_RADIUS = 0.14;

double bornFromHCT(double radius, double hct, bool useOBC) {
    double R_off = radius - DIELECTRIC_OFFSET;
    if (R_off <= 0.0)
        return 500.0;
    double inner;
    if (!useOBC) {
        inner = 1.0 / R_off - 0.5 * R_off * hct;
    } else {
        double psi = 0.5 * R_off * hct;
        double t = tanh(OBC_ALPHA * psi - OBC_BETA * psi * psi
                        + OBC_GAMMA * psi * psi * psi);
        inner = 1.0 / R_off - t / radius;
    }
    return (inner > 0.0) ? 1.0 / inner : 500.0;
}

// dR_born/dhct in the plugin convention (psi = 0.5 * R_off * hct).
double dBornDHCT(double radius, double hct, double born, bool useOBC) {
    double R_off = radius - DIELECTRIC_OFFSET;
    if (R_off <= 0.0)
        return 0.0;
    if (!useOBC)
        return 0.5 * R_off * born * born;
    double psi = 0.5 * R_off * hct;
    double t = tanh(OBC_ALPHA * psi - OBC_BETA * psi * psi
                    + OBC_GAMMA * psi * psi * psi);
    double sech2 = 1.0 - t * t;
    double dtanh_dpsi = sech2 * (OBC_ALPHA - 2.0 * OBC_BETA * psi
                                 + 3.0 * OBC_GAMMA * psi * psi);
    return born * born * dtanh_dpsi * 0.5 * R_off / radius;
}

// Cubic B-spline basis weights and their derivatives for a fractional offset.
void cubicWeights(double t, double w[4], double dw[4]) {
    double t2 = t * t, t3 = t2 * t;
    w[0] = (1.0 - 3.0 * t + 3.0 * t2 - t3) / 6.0;
    w[1] = (4.0 - 6.0 * t2 + 3.0 * t3) / 6.0;
    w[2] = (1.0 + 3.0 * t + 3.0 * t2 - 3.0 * t3) / 6.0;
    w[3] = t3 / 6.0;
    dw[0] = (-3.0 + 6.0 * t - 3.0 * t2) / 6.0;
    dw[1] = (-12.0 * t + 9.0 * t2) / 6.0;
    dw[2] = (3.0 + 6.0 * t - 9.0 * t2) / 6.0;
    dw[3] = 3.0 * t2 / 6.0;
}

void runFor(const function<void(int, const function<void(int)>&)>& parallelFor,
            int count, const function<void(int)>& body) {
    if (parallelFor) {
        parallelFor(count, body);
    } else {
        for (int i = 0; i < count; i++)
            body(i);
    }
}

}  // namespace

namespace GridForcePlugin {
namespace SolvationFields {

double switchValue(double r, double rOn, double rOff) {
    if (rOff <= rOn)
        return 1.0;
    if (r <= rOn)
        return 0.0;
    if (r >= rOff)
        return 1.0;
    double t = (r - rOn) / (rOff - rOn);
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

double switchDerivative(double r, double rOn, double rOff) {
    if (rOff <= rOn || r <= rOn || r >= rOff)
        return 0.0;
    double inv = 1.0 / (rOff - rOn);
    double t = (r - rOn) * inv;
    return 30.0 * t * t * (t - 1.0) * (t - 1.0) * inv;
}

void computeApoReceptor(const vector<double>& positions,
                        const vector<double>& charges,
                        const vector<double>& radii,
                        const vector<double>& scaleFactors,
                        bool useOBC, double prefactor, double cutoffDistance,
                        bool includeSurfaceArea, double surfaceTension,
                        const vector<double>* suppliedBornRadii,
                        vector<double>& hct, vector<double>& bornRadii,
                        vector<double>& mirrorWeights,
                        const function<void(int, const function<void(int)>&)>&
                            parallelFor) {

    int n = static_cast<int>(radii.size());
    if (static_cast<int>(positions.size()) != 3 * n)
        throw OpenMMException("computeApoReceptor: positions size mismatch");

    hct.assign(n, 0.0);
    bornRadii.assign(n, 0.0);
    mirrorWeights.assign(n, 0.0);
    if (n == 0)
        return;

    bool useCutoff = (cutoffDistance > 0.0);
    double cutoff2 = cutoffDistance * cutoffDistance;

    // Apo HCT integral and Born radius per receptor atom.
    runFor(parallelFor, n, [&](int j) {
        double Rj_off = radii[j] - DIELECTRIC_OFFSET;
        double sum = 0.0;
        for (int k = 0; k < n; k++) {
            if (k == j)
                continue;
            double dx = positions[j * 3] - positions[k * 3];
            double dy = positions[j * 3 + 1] - positions[k * 3 + 1];
            double dz = positions[j * 3 + 2] - positions[k * 3 + 2];
            double r2 = dx * dx + dy * dy + dz * dz;
            if (useCutoff && r2 > cutoff2)
                continue;
            double Rk_off = radii[k] - DIELECTRIC_OFFSET;
            sum += computeHCTTerm(sqrt(r2), Rj_off, Rk_off, scaleFactors[k]);
        }
        hct[j] = sum;
        bornRadii[j] = bornFromHCT(radii[j], sum, useOBC);
    });

    if (suppliedBornRadii != nullptr) {
        if (static_cast<int>(suppliedBornRadii->size()) != n)
            throw OpenMMException("computeApoReceptor: suppliedBornRadii size mismatch");
        bornRadii = *suppliedBornRadii;
    }

    // dE_receptor/dR_j at apo: self + receptor-receptor GB pairs + ACE.
    runFor(parallelFor, n, [&](int j) {
        double Rj = bornRadii[j];
        double dEdR = -0.5 * prefactor * charges[j] * charges[j] / (Rj * Rj);

        for (int k = 0; k < n; k++) {
            if (k == j)
                continue;
            double dx = positions[j * 3] - positions[k * 3];
            double dy = positions[j * 3 + 1] - positions[k * 3 + 1];
            double dz = positions[j * 3 + 2] - positions[k * 3 + 2];
            double r2 = dx * dx + dy * dy + dz * dz;
            double D = Rj * bornRadii[k];
            double alpha = r2 / (4.0 * D);
            double expAlpha = exp(-alpha);
            double fgb2 = r2 + D * expAlpha;
            double fgb = sqrt(fgb2);
            double dfgb_dRj = bornRadii[k] * expAlpha * (1.0 + alpha) / (2.0 * fgb);
            dEdR += -prefactor * charges[j] * charges[k] / fgb2 * dfgb_dRj;
        }

        if (includeSurfaceArea) {
            double Rsolv = radii[j] + SA_PROBE_RADIUS;
            double rho6 = pow(radii[j], 6.0);
            dEdR += -6.0 * surfaceTension * 4.0 * M_PI * Rsolv * Rsolv * rho6
                    / pow(Rj, 7.0);
        }

        mirrorWeights[j] = dEdR * dBornDHCT(radii[j], hct[j], Rj, useOBC);
    });
}

vector<int> selectPocketAtoms(const vector<double>& positions,
                              const double origin[3], double spacing,
                              const int counts[3], double padding) {
    double lo[3], hi[3];
    for (int d = 0; d < 3; d++) {
        lo[d] = origin[d] - padding;
        hi[d] = origin[d] + (counts[d] - 1) * spacing + padding;
    }
    int n = static_cast<int>(positions.size() / 3);
    vector<int> pocket;
    pocket.reserve(n / 4 + 1);
    for (int j = 0; j < n; j++) {
        bool inside = true;
        for (int d = 0; d < 3 && inside; d++) {
            double p = positions[j * 3 + d];
            inside = (p >= lo[d] && p <= hi[d]);
        }
        if (inside)
            pocket.push_back(j);
    }
    return pocket;
}

vector<double> distinctScaledRadii(const vector<double>& radii,
                                   const vector<double>& scaleFactors,
                                   double tolerance) {
    vector<double> values;
    values.reserve(radii.size());
    for (size_t i = 0; i < radii.size(); i++)
        values.push_back((radii[i] - DIELECTRIC_OFFSET) * scaleFactors[i]);
    sort(values.begin(), values.end());

    vector<double> distinct;
    for (double v : values) {
        if (distinct.empty() || v - distinct.back() > tolerance)
            distinct.push_back(v);
    }
    return distinct;
}

int sliceForScaledRadius(const vector<double>& sliceValues, double s) {
    int best = 0;
    double bestDiff = 1e30;
    for (size_t b = 0; b < sliceValues.size(); b++) {
        double diff = fabs(sliceValues[b] - s);
        if (diff < bestDiff) {
            bestDiff = diff;
            best = static_cast<int>(b);
        }
    }
    return best;
}

vector<double> defaultCrossFieldRadii(const vector<double>& radii, int numSlices) {
    if (numSlices < 2)
        throw OpenMMException("defaultCrossFieldRadii: need at least two slices");
    if (radii.empty())
        throw OpenMMException("defaultCrossFieldRadii: no template radii");

    double lo = 1e30, hi = 0.0;
    for (double rho : radii) {
        double R_off = rho - DIELECTRIC_OFFSET;
        if (R_off <= 0.0)
            continue;
        lo = min(lo, R_off);
        // OBC2 saturates at 1/(1/R_off - 1/rho); a fully buried atom reaches it.
        hi = max(hi, R_off * rho / DIELECTRIC_OFFSET);
    }
    if (!(lo < hi))
        throw OpenMMException("defaultCrossFieldRadii: degenerate radius range");

    lo *= 0.98;
    hi *= 1.02;
    vector<double> out(numSlices);
    double step = (log(hi) - log(lo)) / (numSlices - 1);
    for (int k = 0; k < numSlices; k++)
        out[k] = exp(log(lo) + k * step);
    return out;
}

shared_ptr<SolvationFieldGrid> buildCrossField(
        const vector<double>& positions, const vector<double>& charges,
        const vector<double>& bornRadiiApo, const vector<int>& atomIndices,
        const vector<double>& sliceValues,
        const double origin[3], double spacing, const int counts[3],
        double switchOn, double switchOff, int interpMethod,
        const function<void(int, const function<void(int)>&)>& parallelFor) {

    if (interpMethod != InterpolationMethod::TRILINEAR &&
        interpMethod != InterpolationMethod::TRICUBIC_BSPLINE)
        throw OpenMMException("buildCrossField: interpolationMethod must be "
                              "0 (trilinear) or 1 (tricubic B-spline)");
    if (sliceValues.size() < 2)
        throw OpenMMException("buildCrossField: need at least two radius slices");

    int numSlices = static_cast<int>(sliceValues.size());
    auto grid = make_shared<SolvationFieldGrid>(
        counts[0], counts[1], counts[2], spacing, numSlices,
        SolvationFieldGrid::CROSS_GB, interpMethod);
    grid->setOrigin(origin[0], origin[1], origin[2]);
    grid->setSwitchRadii(switchOn, switchOff);
    grid->setSliceParameters(sliceValues);

    int nx = counts[0], ny = counts[1], nz = counts[2];
    int nyz = ny * nz;
    int numPoints = nx * ny * nz;
    vector<float> data(static_cast<size_t>(numSlices) * numPoints, 0.0f);

    runFor(parallelFor, nx, [&](int ix) {
        double px = origin[0] + ix * spacing;
        vector<double> accum(numSlices);
        for (int iy = 0; iy < ny; iy++) {
            double py = origin[1] + iy * spacing;
            for (int iz = 0; iz < nz; iz++) {
                double pz = origin[2] + iz * spacing;
                fill(accum.begin(), accum.end(), 0.0);

                for (int idx : atomIndices) {
                    double dx = px - positions[idx * 3];
                    double dy = py - positions[idx * 3 + 1];
                    double dz = pz - positions[idx * 3 + 2];
                    double r2 = dx * dx + dy * dy + dz * dz;
                    if (r2 < 1e-20)
                        continue;
                    double r = sqrt(r2);
                    double sw = switchValue(r, switchOn, switchOff);
                    if (sw == 0.0)
                        continue;
                    double qs = charges[idx] * sw;
                    double Rj = bornRadiiApo[idx];
                    for (int k = 0; k < numSlices; k++) {
                        double D = sliceValues[k] * Rj;
                        accum[k] += qs / sqrt(r2 + D * exp(-r2 / (4.0 * D)));
                    }
                }

                int point = ix * nyz + iy * nz + iz;
                for (int k = 0; k < numSlices; k++)
                    data[static_cast<size_t>(k) * numPoints + point] =
                        static_cast<float>(accum[k]);
            }
        }
    });

    if (interpMethod == InterpolationMethod::TRICUBIC_BSPLINE) {
        for (int k = 0; k < numSlices; k++) {
            vector<float> slice(data.begin() + static_cast<size_t>(k) * numPoints,
                                data.begin() + static_cast<size_t>(k + 1) * numPoints);
            bsplinePrefilter3D(slice, nx, ny, nz);
            copy(slice.begin(), slice.end(),
                 data.begin() + static_cast<size_t>(k) * numPoints);
        }
    }

    grid->setData(std::move(data));
    return grid;
}

shared_ptr<SolvationFieldGrid> buildMirrorField(
        const vector<double>& positions, const vector<double>& radii,
        const vector<double>& mirrorWeights, const vector<int>& atomIndices,
        const vector<double>& sliceValues,
        const double origin[3], double spacing, const int counts[3],
        double switchOn, double switchOff, double buildCutoff, int interpMethod,
        const function<void(int, const function<void(int)>&)>& parallelFor) {

    if (interpMethod != InterpolationMethod::TRILINEAR &&
        interpMethod != InterpolationMethod::TRICUBIC_BSPLINE)
        throw OpenMMException("buildMirrorField: interpolationMethod must be "
                              "0 (trilinear) or 1 (tricubic B-spline)");
    if (sliceValues.empty())
        throw OpenMMException("buildMirrorField: at least one slice required");

    int numSlices = static_cast<int>(sliceValues.size());
    auto grid = make_shared<SolvationFieldGrid>(
        counts[0], counts[1], counts[2], spacing, numSlices,
        SolvationFieldGrid::MIRROR, interpMethod);
    grid->setOrigin(origin[0], origin[1], origin[2]);
    grid->setSwitchRadii(switchOn, switchOff);
    grid->setSliceParameters(sliceValues);

    int nx = counts[0], ny = counts[1], nz = counts[2];
    int nyz = ny * nz;
    int numPoints = nx * ny * nz;
    double cutoff2 = buildCutoff * buildCutoff;
    vector<float> data(static_cast<size_t>(numSlices) * numPoints, 0.0f);

    runFor(parallelFor, nx, [&](int ix) {
        double px = origin[0] + ix * spacing;
        vector<double> accum(numSlices);
        for (int iy = 0; iy < ny; iy++) {
            double py = origin[1] + iy * spacing;
            for (int iz = 0; iz < nz; iz++) {
                double pz = origin[2] + iz * spacing;
                fill(accum.begin(), accum.end(), 0.0);

                for (int idx : atomIndices) {
                    double w = mirrorWeights[idx];
                    if (w == 0.0)
                        continue;
                    double dx = px - positions[idx * 3];
                    double dy = py - positions[idx * 3 + 1];
                    double dz = pz - positions[idx * 3 + 2];
                    double r2 = dx * dx + dy * dy + dz * dz;
                    if (r2 > cutoff2)
                        continue;
                    double r = sqrt(r2);
                    if (r < 1e-10)
                        continue;
                    double s = switchValue(r, switchOn, switchOff);
                    if (s == 0.0)
                        continue;
                    double Rj_off = radii[idx] - DIELECTRIC_OFFSET;
                    double ws = w * s;
                    for (int b = 0; b < numSlices; b++) {
                        // The HCT closed form depends on the descreener only
                        // through its scaled radius, so pass s_b directly.
                        accum[b] += ws * computeHCTTerm(r, Rj_off, sliceValues[b], 1.0);
                    }
                }

                int point = ix * nyz + iy * nz + iz;
                for (int b = 0; b < numSlices; b++)
                    data[static_cast<size_t>(b) * numPoints + point] =
                        static_cast<float>(accum[b]);
            }
        }
    });

    if (interpMethod == InterpolationMethod::TRICUBIC_BSPLINE) {
        for (int b = 0; b < numSlices; b++) {
            vector<float> slice(data.begin() + static_cast<size_t>(b) * numPoints,
                                data.begin() + static_cast<size_t>(b + 1) * numPoints);
            bsplinePrefilter3D(slice, nx, ny, nz);
            copy(slice.begin(), slice.end(),
                 data.begin() + static_cast<size_t>(b) * numPoints);
        }
    }

    grid->setData(std::move(data));
    return grid;
}

double interpolateField(const SolvationFieldGrid& grid, int slice,
                        double x, double y, double z, int interpMethod,
                        bool computeGrad,
                        double& gradX, double& gradY, double& gradZ) {

    gradX = gradY = gradZ = 0.0;

    int nx, ny, nz;
    grid.getCounts(nx, ny, nz);
    double ox, oy, oz;
    grid.getOrigin(ox, oy, oz);
    double spacing = grid.getSpacing();
    double invSpacing = 1.0 / spacing;
    int nyz = ny * nz;

    const vector<float>& data = grid.getData();
    const float* g = data.data() + grid.getSliceOffset(slice);

    double fx = (x - ox) * invSpacing;
    double fy = (y - oy) * invSpacing;
    double fz = (z - oz) * invSpacing;

    int ix = static_cast<int>(floor(fx));
    int iy = static_cast<int>(floor(fy));
    int iz = static_cast<int>(floor(fz));

    if (ix < 0 || ix >= nx - 1 || iy < 0 || iy >= ny - 1 || iz < 0 || iz >= nz - 1)
        return 0.0;

    double tx = fx - ix, ty = fy - iy, tz = fz - iz;

    if (interpMethod == InterpolationMethod::TRICUBIC_BSPLINE) {
        double wx[4], wy[4], wz[4], dwx[4], dwy[4], dwz[4];
        cubicWeights(tx, wx, dwx);
        cubicWeights(ty, wy, dwy);
        cubicWeights(tz, wz, dwz);

        double value = 0.0, dvx = 0.0, dvy = 0.0, dvz = 0.0;
        for (int a = 0; a < 4; a++) {
            int cx = min(max(ix - 1 + a, 0), nx - 1);
            for (int b = 0; b < 4; b++) {
                int cy = min(max(iy - 1 + b, 0), ny - 1);
                for (int c = 0; c < 4; c++) {
                    int cz = min(max(iz - 1 + c, 0), nz - 1);
                    double v = g[cx * nyz + cy * nz + cz];
                    value += wx[a] * wy[b] * wz[c] * v;
                    if (computeGrad) {
                        dvx += dwx[a] * wy[b] * wz[c] * v;
                        dvy += wx[a] * dwy[b] * wz[c] * v;
                        dvz += wx[a] * wy[b] * dwz[c] * v;
                    }
                }
            }
        }
        if (computeGrad) {
            gradX = dvx * invSpacing;
            gradY = dvy * invSpacing;
            gradZ = dvz * invSpacing;
        }
        return value;
    }

    // Trilinear
    int base = ix * nyz + iy * nz + iz;
    double c000 = g[base];
    double c001 = g[base + 1];
    double c010 = g[base + nz];
    double c011 = g[base + nz + 1];
    double c100 = g[base + nyz];
    double c101 = g[base + nyz + 1];
    double c110 = g[base + nyz + nz];
    double c111 = g[base + nyz + nz + 1];

    double mx = 1.0 - tx, my = 1.0 - ty, mz = 1.0 - tz;
    double value = mx * my * mz * c000 + mx * my * tz * c001
                 + mx * ty * mz * c010 + mx * ty * tz * c011
                 + tx * my * mz * c100 + tx * my * tz * c101
                 + tx * ty * mz * c110 + tx * ty * tz * c111;

    if (computeGrad) {
        gradX = invSpacing * (my * mz * (c100 - c000) + my * tz * (c101 - c001)
                            + ty * mz * (c110 - c010) + ty * tz * (c111 - c011));
        gradY = invSpacing * (mx * mz * (c010 - c000) + mx * tz * (c011 - c001)
                            + tx * mz * (c110 - c100) + tx * tz * (c111 - c101));
        gradZ = invSpacing * (mx * my * (c001 - c000) + mx * ty * (c011 - c010)
                            + tx * my * (c101 - c100) + tx * ty * (c111 - c110));
    }
    return value;
}

void buildPocketCellList(const vector<double>& positions,
                         const vector<int>& atomIndices,
                         double cellSize, PocketCellList& out) {
    out.cellSize = (cellSize > 0.0) ? cellSize : 1.0;
    out.atoms.clear();
    out.cellStart.clear();
    for (int d = 0; d < 3; d++) {
        out.origin[d] = 0.0;
        out.counts[d] = 1;
    }
    if (atomIndices.empty()) {
        out.cellStart.assign(2, 0);
        return;
    }

    double lo[3], hi[3];
    for (int d = 0; d < 3; d++) {
        lo[d] = 1e30;
        hi[d] = -1e30;
    }
    for (int idx : atomIndices) {
        for (int d = 0; d < 3; d++) {
            double p = positions[idx * 3 + d];
            lo[d] = min(lo[d], p);
            hi[d] = max(hi[d], p);
        }
    }

    int nCells = 1;
    for (int d = 0; d < 3; d++) {
        out.origin[d] = lo[d];
        out.counts[d] = max(1, static_cast<int>((hi[d] - lo[d]) / out.cellSize) + 1);
        nCells *= out.counts[d];
    }

    // Counting sort of pocket atoms into cells.
    vector<int> counts(nCells, 0);
    vector<int> cellOf(atomIndices.size());
    double inv = 1.0 / out.cellSize;
    for (size_t a = 0; a < atomIndices.size(); a++) {
        int idx = atomIndices[a];
        int c[3];
        for (int d = 0; d < 3; d++) {
            c[d] = static_cast<int>((positions[idx * 3 + d] - out.origin[d]) * inv);
            c[d] = min(max(c[d], 0), out.counts[d] - 1);
        }
        int cell = (c[0] * out.counts[1] + c[1]) * out.counts[2] + c[2];
        cellOf[a] = cell;
        counts[cell]++;
    }

    out.cellStart.assign(nCells + 1, 0);
    for (int c = 0; c < nCells; c++)
        out.cellStart[c + 1] = out.cellStart[c] + counts[c];

    vector<int> cursor(out.cellStart.begin(), out.cellStart.end() - 1);
    out.atoms.resize(atomIndices.size());
    for (size_t a = 0; a < atomIndices.size(); a++)
        out.atoms[cursor[cellOf[a]]++] = atomIndices[a];
}

}  // namespace SolvationFields
}  // namespace GridForcePlugin
