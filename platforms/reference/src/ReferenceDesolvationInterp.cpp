#include "ReferenceDesolvationInterp.h"
#include "ReferenceGridInterpolation.h"
#include "TricubicMatrix.h"
#include "TriquinticMatrix.h"
#include "DesolvationGrid.h"

#include <algorithm>
#include <cmath>

namespace GridForcePlugin {

namespace {

// Cubic B-spline basis functions and first derivatives on [0,1] (4-point
// stencil). Duplicated from ReferenceGBSAGridForceKernels.cpp's anonymous
// namespace so both callers share one implementation.
inline double bsplineBasis(int i, double t) {
    switch (i) {
        case 0: return (1.0 - t)*(1.0 - t)*(1.0 - t) / 6.0;
        case 1: return (3.0*t*t*t - 6.0*t*t + 4.0) / 6.0;
        case 2: return (-3.0*t*t*t + 3.0*t*t + 3.0*t + 1.0) / 6.0;
        default: return t*t*t / 6.0;
    }
}
inline double bsplineDeriv(int i, double t) {
    switch (i) {
        case 0: return -(1.0 - t)*(1.0 - t) / 2.0;
        case 1: return (3.0*t*t - 4.0*t) / 2.0;
        case 2: return (-3.0*t*t + 2.0*t + 1.0) / 2.0;
        default: return t*t / 2.0;
    }
}

}  // namespace

double interpolateDesolvationGridHCT(
        const DesolvationGrid* desolvationGrid,
        double x, double y, double z,
        double R_i_off,
        int interpolationMethod,
        double probeRadius,
        bool computeGrad,
        double& gradX, double& gradY, double& gradZ,
        double* hess,
        bool* outOfBounds) {

    gradX = gradY = gradZ = 0.0;
    if (hess) for (int c = 0; c < 6; c++) hess[c] = 0.0;
    if (outOfBounds) *outOfBounds = false;

    if (!desolvationGrid) return 0.0;

    double ox, oy, oz;
    desolvationGrid->getOrigin(ox, oy, oz);
    double spacing = desolvationGrid->getSpacing();
    int nx, ny, nz;
    desolvationGrid->getCounts(nx, ny, nz);
    int numBins = desolvationGrid->getNumBins();

    double rx = (x - ox) / spacing;
    double ry = (y - oy) / spacing;
    double rz = (z - oz) / spacing;
    int ix = (int)std::floor(rx);
    int iy = (int)std::floor(ry);
    int iz = (int)std::floor(rz);

    if (ix < 0 || ix >= nx - 1 || iy < 0 || iy >= ny - 1 || iz < 0 || iz >= nz - 1) {
        if (outOfBounds) *outOfBounds = true;
        return 0.0;
    }

    double fx = rx - ix;
    double fy = ry - iy;
    double fz = rz - iz;
    double ofx = 1.0 - fx;
    double ofy = 1.0 - fy;
    double ofz = 1.0 - fz;

    int nyz = ny * nz;

    int c000 = ix * nyz + iy * nz + iz;
    int c001 = c000 + 1;
    int c010 = c000 + nz;
    int c011 = c010 + 1;
    int c100 = c000 + nyz;
    int c101 = c100 + 1;
    int c110 = c100 + nz;
    int c111 = c110 + 1;

    double R_probe_off = probeRadius - DIELECTRIC_OFFSET;
    int binIdx = 0;
    if (numBins > 1) {
        const auto& rThresh = desolvationGrid->getRThresholds();
        for (int b = 0; b < numBins - 1; b++) {
            if (R_i_off > rThresh[b]) binIdx = b + 1;
        }
    }
    int numPoints = nx * ny * nz;
    int binOffset = binIdx * numPoints;

    const auto& hctProbeData = desolvationGrid->getHctProbe();
    const auto& corrN = desolvationGrid->getCorrectionN();
    const auto& corrA = desolvationGrid->getCorrectionA();
    const auto& corrB = desolvationGrid->getCorrectionB();

    // Cubic B-spline (method 1): interpolate the COMBINED field hct_probe + the
    // radius-weighted corrections over a 4x4x4 stencil. B-spline approximates (it
    // does not pass through the samples), so it smooths the step-like correction
    // grids to C2 - continuous forces through the receptor-surface shell.
    if (interpolationMethod == 1) {
        double invRiB = 1.0 / R_i_off, invRpB = 1.0 / R_probe_off;
        double deltaB = invRiB - invRpB, sigmaB = invRiB + invRpB;
        double dN = deltaB, dA = -0.25 * deltaB * sigmaB, dB = std::log(R_i_off / R_probe_off);
        double bxv[4], byv[4], bzv[4], dbxv[4], dbyv[4], dbzv[4];
        for (int t = 0; t < 4; t++) {
            bxv[t] = bsplineBasis(t, fx); byv[t] = bsplineBasis(t, fy); bzv[t] = bsplineBasis(t, fz);
            dbxv[t] = bsplineDeriv(t, fx); dbyv[t] = bsplineDeriv(t, fy); dbzv[t] = bsplineDeriv(t, fz);
        }
        double val = 0.0, gx = 0.0, gy = 0.0, gz = 0.0;
        for (int i = 0; i < 4; i++) {
            int gxi = std::min(std::max(ix - 1 + i, 0), nx - 1);
            for (int j = 0; j < 4; j++) {
                int gyi = std::min(std::max(iy - 1 + j, 0), ny - 1);
                for (int k = 0; k < 4; k++) {
                    int gzi = std::min(std::max(iz - 1 + k, 0), nz - 1);
                    int gi = gxi * nyz + gyi * nz + gzi;
                    double combined = hctProbeData[gi]
                        + dN * corrN[binOffset + gi]
                        + dA * corrA[binOffset + gi]
                        + dB * corrB[binOffset + gi];
                    val += bxv[i] * byv[j] * bzv[k] * combined;
                    gx  += dbxv[i] * byv[j] * bzv[k] * combined;
                    gy  += bxv[i] * dbyv[j] * bzv[k] * combined;
                    gz  += bxv[i] * byv[j] * dbzv[k] * combined;
                }
            }
        }
        if (computeGrad) {
            double invSpacing = 1.0 / spacing;
            gradX = gx * invSpacing; gradY = gy * invSpacing; gradZ = gz * invSpacing;
        }
        return val;
    }

    double h000 = hctProbeData[c000], h001 = hctProbeData[c001];
    double h010 = hctProbeData[c010], h011 = hctProbeData[c011];
    double h100 = hctProbeData[c100], h101 = hctProbeData[c101];
    double h110 = hctProbeData[c110], h111 = hctProbeData[c111];

    double vmm = ofz * h000 + fz * h001;
    double vmp = ofz * h010 + fz * h011;
    double vpm = ofz * h100 + fz * h101;
    double vpp = ofz * h110 + fz * h111;
    double vm = ofy * vmm + fy * vmp;
    double vp = ofy * vpm + fy * vpp;
    double hct = ofx * vm + fx * vp;

    // Higher-order Hermite override for the HCT probe field: tricubic (method 2,
    // C1, 8 derivatives/corner) or triquintic (method 3, C2, 27 derivatives/corner).
    // Corner order matches the CUDA kernel:
    // {(ix,iy,iz),(ix+1,iy,iz),(ix,iy+1,iz),(ix+1,iy+1,iz),
    //  (ix,iy,iz+1),(ix+1,iy,iz+1),(ix,iy+1,iz+1),(ix+1,iy+1,iz+1)}.
    bool useTricubic = (interpolationMethod == 2) && desolvationGrid->hasDerivatives();
    bool useTriquintic = (interpolationMethod == 3) && desolvationGrid->hasDerivatives();
    double hctTri = 0.0, dhTri_dfx = 0.0, dhTri_dfy = 0.0, dhTri_dfz = 0.0;
    if (useTricubic) {
        const int corners[8] = {c000, c100, c010, c110, c001, c101, c011, c111};
        const int derivMap[8] = {0, 1, 2, 3, 5, 6, 8, 13};
        double X[64];
        for (int d = 0; d < 8; d++)
            for (int c = 0; c < 8; c++)
                X[d * 8 + c] = hctProbeData[(size_t)derivMap[d] * numPoints + corners[c]];
        double a[64];
        tricubicAssemble(X, a);
        tricubicEvalVG(a, fx, fy, fz, &hctTri, &dhTri_dfx, &dhTri_dfy, &dhTri_dfz);
        hct = hctTri;
    } else if (useTriquintic) {
        const int corners[8] = {c000, c100, c010, c110, c001, c101, c011, c111};
        double X[216];
        for (int d = 0; d < 27; d++)
            for (int c = 0; c < 8; c++)
                X[d * 8 + c] = hctProbeData[(size_t)d * numPoints + corners[c]];

        double a[216];
        for (int i = 0; i < 216; i++) {
            double acc = 0.0;
            for (int j = 0; j < 216; j++)
                acc += TRIQUINTIC_COEFFICIENTS[i][j] * X[j];
            a[i] = 0.125 * acc;
        }

        double sxp[6], syp[6], szp[6];
        sxp[0] = syp[0] = szp[0] = 1.0;
        for (int p = 1; p < 6; p++) {
            sxp[p] = sxp[p-1] * fx;
            syp[p] = syp[p-1] * fy;
            szp[p] = szp[p-1] * fz;
        }
        for (int k = 0; k < 6; k++)
        for (int j = 0; j < 6; j++)
        for (int i = 0; i < 6; i++) {
            double coeff = a[i + 6*j + 36*k];
            hctTri += coeff * sxp[i] * syp[j] * szp[k];
            if (i > 0) dhTri_dfx += coeff * i * sxp[i-1] * syp[j] * szp[k];
            if (j > 0) dhTri_dfy += coeff * j * sxp[i] * syp[j-1] * szp[k];
            if (k > 0) dhTri_dfz += coeff * k * sxp[i] * syp[j] * szp[k-1];
        }
        hct = hctTri;
    }

    // Trilinear interpolation of the N, A, B correction grids (binned mode)
    double N000 = corrN[binOffset + c000], N001 = corrN[binOffset + c001];
    double N010 = corrN[binOffset + c010], N011 = corrN[binOffset + c011];
    double N100 = corrN[binOffset + c100], N101 = corrN[binOffset + c101];
    double N110 = corrN[binOffset + c110], N111 = corrN[binOffset + c111];

    double Nmm = ofz * N000 + fz * N001;
    double Nmp = ofz * N010 + fz * N011;
    double Npm = ofz * N100 + fz * N101;
    double Npp = ofz * N110 + fz * N111;
    double Nm = ofy * Nmm + fy * Nmp;
    double Np = ofy * Npm + fy * Npp;
    double N_val = ofx * Nm + fx * Np;

    double A000 = corrA[binOffset + c000], A001 = corrA[binOffset + c001];
    double A010 = corrA[binOffset + c010], A011 = corrA[binOffset + c011];
    double A100 = corrA[binOffset + c100], A101 = corrA[binOffset + c101];
    double A110 = corrA[binOffset + c110], A111 = corrA[binOffset + c111];

    double Amm = ofz * A000 + fz * A001;
    double Amp = ofz * A010 + fz * A011;
    double Apm = ofz * A100 + fz * A101;
    double App = ofz * A110 + fz * A111;
    double Am = ofy * Amm + fy * Amp;
    double Ap = ofy * Apm + fy * App;
    double A_val = ofx * Am + fx * Ap;

    double B000 = corrB[binOffset + c000], B001 = corrB[binOffset + c001];
    double B010 = corrB[binOffset + c010], B011 = corrB[binOffset + c011];
    double B100 = corrB[binOffset + c100], B101 = corrB[binOffset + c101];
    double B110 = corrB[binOffset + c110], B111 = corrB[binOffset + c111];

    double Bmm = ofz * B000 + fz * B001;
    double Bmp = ofz * B010 + fz * B011;
    double Bpm = ofz * B100 + fz * B101;
    double Bpp = ofz * B110 + fz * B111;
    double Bm = ofy * Bmm + fy * Bmp;
    double Bp = ofy * Bpm + fy * Bpp;
    double B_val = ofx * Bm + fx * Bp;

    double invRi = 1.0 / R_i_off;
    double invRp = 1.0 / R_probe_off;
    double delta = invRi - invRp;
    double sigma = invRi + invRp;
    double logTerm = std::log(R_i_off / R_probe_off);

    double result = hct + delta * (N_val - 0.25 * A_val * sigma) + B_val * logTerm;

    if (computeGrad) {
        double invSpacing = 1.0 / spacing;
        double dCorr_dN = delta;
        double dCorr_dA = -0.25 * delta * sigma;
        double dCorr_dB = logTerm;

        double dh_dfx, dh_dfy, dh_dfz;
        if (useTricubic || useTriquintic) {
            dh_dfx = dhTri_dfx;
            dh_dfy = dhTri_dfy;
            dh_dfz = dhTri_dfz;
        } else {
            dh_dfx = vp - vm;
            dh_dfy = ofx * (vmp - vmm) + fx * (vpp - vpm);
            dh_dfz = ofx * (ofy * (h001 - h000) + fy * (h011 - h010)) +
                      fx * (ofy * (h101 - h100) + fy * (h111 - h110));
        }

        double dN_dfx = Np - Nm;
        double dN_dfy = ofx * (Nmp - Nmm) + fx * (Npp - Npm);
        double dN_dfz = ofx * (ofy * (N001 - N000) + fy * (N011 - N010)) +
                         fx * (ofy * (N101 - N100) + fy * (N111 - N110));

        double dA_dfx = Ap - Am;
        double dA_dfy = ofx * (Amp - Amm) + fx * (App - Apm);
        double dA_dfz = ofx * (ofy * (A001 - A000) + fy * (A011 - A010)) +
                         fx * (ofy * (A101 - A100) + fy * (A111 - A110));

        double dB_dfx = Bp - Bm;
        double dB_dfy = ofx * (Bmp - Bmm) + fx * (Bpp - Bpm);
        double dB_dfz = ofx * (ofy * (B001 - B000) + fy * (B011 - B010)) +
                         fx * (ofy * (B101 - B100) + fy * (B111 - B110));

        gradX = (dh_dfx + dCorr_dN * dN_dfx + dCorr_dA * dA_dfx + dCorr_dB * dB_dfx) * invSpacing;
        gradY = (dh_dfy + dCorr_dN * dN_dfy + dCorr_dA * dA_dfy + dCorr_dB * dB_dfy) * invSpacing;
        gradZ = (dh_dfz + dCorr_dN * dN_dfz + dCorr_dA * dA_dfz + dCorr_dB * dB_dfz) * invSpacing;
    }

    if (hess) {
        double dN = delta, dA = -0.25 * delta * sigma, dB = logTerm;
        double v000 = h000 + dN * N000 + dA * A000 + dB * B000;
        double v001 = h001 + dN * N001 + dA * A001 + dB * B001;
        double v010 = h010 + dN * N010 + dA * A010 + dB * B010;
        double v011 = h011 + dN * N011 + dA * A011 + dB * B011;
        double v100 = h100 + dN * N100 + dA * A100 + dB * B100;
        double v101 = h101 + dN * N101 + dA * A101 + dB * B101;
        double v110 = h110 + dN * N110 + dA * A110 + dB * B110;
        double v111 = h111 + dN * N111 + dA * A111 + dB * B111;
        double invSpacing2 = 1.0 / (spacing * spacing);
        hess[0] = 0.0;
        hess[1] = 0.0;
        hess[2] = 0.0;
        hess[3] = (ofz * (v000 - v010 - v100 + v110)
                   + fz * (v001 - v011 - v101 + v111)) * invSpacing2;
        hess[4] = (ofy * (v000 - v001 - v100 + v101)
                   + fy * (v010 - v011 - v110 + v111)) * invSpacing2;
        hess[5] = (ofx * (v000 - v001 - v010 + v011)
                   + fx * (v100 - v101 - v110 + v111)) * invSpacing2;
    }

    return result;
}

}  // namespace GridForcePlugin
