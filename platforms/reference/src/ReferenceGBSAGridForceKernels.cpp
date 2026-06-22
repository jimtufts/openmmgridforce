/* -------------------------------------------------------------------------- *
 *                              OpenMMGridForce                               *
 * -------------------------------------------------------------------------- *
 * Reference platform implementation of GBSAGridForce kernel.                *
 * Computes grid-based GB implicit solvation with OBC-II Born radii for      *
 * isolated particle groups on CPU. Uses DesolvationGrid for receptor HCT.   *
 * -------------------------------------------------------------------------- */

#include "ReferenceGBSAGridForceKernels.h"
#include "ReferenceGridInterpolation.h"
#include "GBSAGridForce.h"

#include "openmm/OpenMMException.h"
#include "openmm/internal/ContextImpl.h"

#include <cmath>
#include <algorithm>
#include <vector>
#include <string>
#include <memory>
#include <iostream>

using namespace OpenMM;
using namespace std;

namespace GridForcePlugin {

// OBC-II parameters
static const double OBC_ALPHA = 1.0;
static const double OBC_BETA = 0.8;
static const double OBC_GAMMA = 4.85;

// ==================== isExcluded ====================

bool ReferenceCalcGBSAGridForceKernel::isExcluded(int i, int j) const {
    if (i >= (int)exclusionSets.size()) return false;
    return exclusionSets[i].count(j) > 0;
}

// ==================== interpolateReceptorHCT ====================

double ReferenceCalcGBSAGridForceKernel::interpolateReceptorHCT(
        double x, double y, double z,
        double R_i_off, bool computeGrad,
        double& gradX, double& gradY, double& gradZ,
        double* hess, bool* outOfBounds) const {

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

    // Compute grid cell
    double rx = (x - ox) / spacing;
    double ry = (y - oy) / spacing;
    double rz = (z - oz) / spacing;
    int ix = (int)floor(rx);
    int iy = (int)floor(ry);
    int iz = (int)floor(rz);

    // Bounds check
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

    // Corner indices
    int c000 = ix * nyz + iy * nz + iz;
    int c001 = c000 + 1;
    int c010 = c000 + nz;
    int c011 = c010 + 1;
    int c100 = c000 + nyz;
    int c101 = c100 + 1;
    int c110 = c100 + nz;
    int c111 = c110 + 1;

    // Determine bin for this radius
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

    // Trilinear interpolation for HCT probe
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

    // Trilinear interpolation for N, A, B correction grids
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

    // Apply correction formula
    double invRi = 1.0 / R_i_off;
    double invRp = 1.0 / R_probe_off;
    double delta = invRi - invRp;
    double sigma = invRi + invRp;
    double logTerm = log(R_i_off / R_probe_off);

    double result = hct + delta * (N_val - 0.25 * A_val * sigma) + B_val * logTerm;

    if (computeGrad) {
        double invSpacing = 1.0 / spacing;
        double dCorr_dN = delta;
        double dCorr_dA = -0.25 * delta * sigma;
        double dCorr_dB = logTerm;

        // Trilinear gradients for HCT
        double dh_dfx = vp - vm;
        double dh_dfy = ofx * (vmp - vmm) + fx * (vpp - vpm);
        double dh_dfz = ofx * (ofy * (h001 - h000) + fy * (h011 - h010)) +
                         fx * (ofy * (h101 - h100) + fy * (h111 - h110));

        // Trilinear gradients for N
        double dN_dfx = Np - Nm;
        double dN_dfy = ofx * (Nmp - Nmm) + fx * (Npp - Npm);
        double dN_dfz = ofx * (ofy * (N001 - N000) + fy * (N011 - N010)) +
                         fx * (ofy * (N101 - N100) + fy * (N111 - N110));

        // Trilinear gradients for A
        double dA_dfx = Ap - Am;
        double dA_dfy = ofx * (Amp - Amm) + fx * (App - Apm);
        double dA_dfz = ofx * (ofy * (A001 - A000) + fy * (A011 - A010)) +
                         fx * (ofy * (A101 - A100) + fy * (A111 - A110));

        // Trilinear gradients for B
        double dB_dfx = Bp - Bm;
        double dB_dfy = ofx * (Bmp - Bmm) + fx * (Bpp - Bpm);
        double dB_dfz = ofx * (ofy * (B001 - B000) + fy * (B011 - B010)) +
                         fx * (ofy * (B101 - B100) + fy * (B111 - B110));

        gradX = (dh_dfx + dCorr_dN * dN_dfx + dCorr_dA * dA_dfx + dCorr_dB * dB_dfx) * invSpacing;
        gradY = (dh_dfy + dCorr_dN * dN_dfy + dCorr_dA * dA_dfy + dCorr_dB * dB_dfy) * invSpacing;
        gradZ = (dh_dfz + dCorr_dN * dN_dfz + dCorr_dA * dA_dfz + dCorr_dB * dB_dfz) * invSpacing;
    }

    // Second derivatives of the interpolated HCT (trilinear). Pure d2 (xx,yy,zz)
    // vanish for trilinear; only the mixed terms survive. The combined field at
    // each corner is hct + delta*N - 0.25*delta*sigma*A + logTerm*B.
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
        hess[0] = 0.0;  // xx
        hess[1] = 0.0;  // yy
        hess[2] = 0.0;  // zz
        hess[3] = (ofz * (v000 - v010 - v100 + v110)
                   + fz * (v001 - v011 - v101 + v111)) * invSpacing2;  // xy
        hess[4] = (ofy * (v000 - v001 - v100 + v101)
                   + fy * (v010 - v011 - v110 + v111)) * invSpacing2;  // xz
        hess[5] = (ofx * (v000 - v001 - v010 + v011)
                   + fx * (v100 - v101 - v110 + v111)) * invSpacing2;  // yz
    }

    return result;
}

// ==================== initialize ====================

void ReferenceCalcGBSAGridForceKernel::initialize(
        const System& system, const GBSAGridForce& force) {

    numAtoms = force.getNumAtoms();
    if (numAtoms == 0)
        throw OpenMMException("GBSAGridForce: no atoms defined");

    // Process particle groups
    int nGroups = force.getNumParticleGroups();
    if (nGroups > 0) {
        numParticleGroups = nGroups;
        groupParticleIndices.resize(nGroups);
        for (int g = 0; g < nGroups; g++) {
            string name;
            vector<int> indices;
            force.getParticleGroup(g, name, indices);
            if ((int)indices.size() != numAtoms)
                throw OpenMMException("GBSAGridForce: particle group has wrong number of indices");
            groupParticleIndices[g] = indices;
        }
    } else {
        numParticleGroups = 1;
        groupParticleIndices.resize(1);
        groupParticleIndices[0] = force.getParticles();
        if ((int)groupParticleIndices[0].size() != numAtoms)
            throw OpenMMException("GBSAGridForce: must set particles or add particle groups");
    }

    // Alchemical scaling
    globalScalingFactor = force.getGlobalScalingFactor();
    groupScalingFactors.resize(numParticleGroups, 1.0);
    for (int g = 0; g < nGroups; g++) {
        groupScalingFactors[g] = force.getGroupScalingFactor(g);
    }

    // Atom parameters
    charges.resize(numAtoms);
    radii.resize(numAtoms);
    scaleFactors.resize(numAtoms);
    for (int i = 0; i < numAtoms; i++) {
        force.getAtomParameters(i, charges[i], radii[i], scaleFactors[i]);
    }

    // Exclusions: build per-atom sets
    exclusionSets.resize(numAtoms);
    int numExcl = force.getNumExclusions();
    for (int e = 0; e < numExcl; e++) {
        int a1, a2;
        force.getExclusionParticles(e, a1, a2);
        exclusionSets[a1].insert(a2);
        exclusionSets[a2].insert(a1);
    }

    // Solvent parameters
    double soluteDielectric = force.getSoluteDielectric();
    double solventDielectric = force.getSolventDielectric();
    prefactor = -COULOMB_CONSTANT * (1.0 / soluteDielectric - 1.0 / solventDielectric);

    includeSurfaceArea = force.getIncludeSurfaceArea();
    surfaceTension = force.getSurfaceTension();
    interpolationMethod = force.getInterpolationMethod();

    // Grid: use a supplied grid, else generate one from the receptor parameters
    // (lazily in the first execute(), where a Context/thread pool is available).
    desolvationGrid = force.getDesolvationGrid();
    if (desolvationGrid) {
        probeRadius = desolvationGrid->getProbeRadius();
    } else if (force.getAutoGenerateGrid()) {
        autoGenerateGrid_ = true;
        genReceptorPositions_ = force.getReceptorPositions();
        genReceptorRadii_ = force.getReceptorRadii();
        genReceptorScales_ = force.getReceptorScaleFactors();
        force.getGridCounts(genCounts_[0], genCounts_[1], genCounts_[2]);
        force.getGridOrigin(genOrigin_[0], genOrigin_[1], genOrigin_[2]);
        genSpacing_ = force.getGridSpacing();
        genRThresholds_ = force.getRThresholds();
        probeRadius = force.getProbeRadius();
        if (genReceptorPositions_.empty() || genReceptorRadii_.empty() ||
            genReceptorScales_.empty())
            throw OpenMMException("GBSAGridForce: auto-generation requires receptor "
                                  "positions, radii, and scale factors");
        if (genCounts_[0] <= 0 || genCounts_[1] <= 0 || genCounts_[2] <= 0 || genSpacing_ <= 0.0)
            throw OpenMMException("GBSAGridForce: auto-generation requires grid counts and spacing");
        if (genRThresholds_.empty())
            throw OpenMMException("GBSAGridForce: auto-generation requires at least one R threshold");
    } else {
        throw OpenMMException("GBSAGridForce: desolvation grid must be set, or enable "
                              "setAutoGenerateGrid with receptor parameters");
    }

    // Initialize per-group result storage
    groupEnergies_.resize(numParticleGroups, 0.0);
    groupLigandEnergies_.resize(numParticleGroups, 0.0);
    groupBornRadii_.resize(numParticleGroups);
}

// ==================== execute ====================

void ReferenceCalcGBSAGridForceKernel::computeGroup(
        int g, vector<Vec3>& posData, vector<Vec3>& forceData,
        bool includeForces, bool includeEnergy) {

        double scale = globalScalingFactor * groupScalingFactors[g];
        if (scale == 0.0) {
            groupBornRadii_[g].assign(numAtoms, 0.0);
            return;
        }

        const vector<int>& particles = groupParticleIndices[g];

        // Step 1: Receptor HCT via grid interpolation
        vector<double> hctReceptor(numAtoms, 0.0);
        vector<double> hctRecGradX(numAtoms, 0.0);
        vector<double> hctRecGradY(numAtoms, 0.0);
        vector<double> hctRecGradZ(numAtoms, 0.0);

        for (int i = 0; i < numAtoms; i++) {
            int pi = particles[i];
            double R_i_off = radii[i] - DIELECTRIC_OFFSET;
            double gx, gy, gz;
            bool oob = false;
            hctReceptor[i] = interpolateReceptorHCT(
                posData[pi][0], posData[pi][1], posData[pi][2],
                R_i_off, includeForces, gx, gy, gz, nullptr, &oob);
            hctRecGradX[i] = gx;
            hctRecGradY[i] = gy;
            hctRecGradZ[i] = gz;
            outOfBoundsFlags_[(size_t)g * numAtoms + i] = oob ? 1 : 0;
        }

        // Step 2: Ligand-ligand HCT (with exclusions)
        vector<double> hctLigand(numAtoms, 0.0);
        for (int i = 0; i < numAtoms; i++) {
            int pi = particles[i];
            double R_i_off = radii[i] - DIELECTRIC_OFFSET;

            for (int j = 0; j < numAtoms; j++) {
                if (i == j) continue;
                if (isExcluded(i, j)) continue;

                int pj = particles[j];
                double dx = posData[pi][0] - posData[pj][0];
                double dy = posData[pi][1] - posData[pj][1];
                double dz = posData[pi][2] - posData[pj][2];
                double r = sqrt(dx * dx + dy * dy + dz * dz);
                if (r < 1e-10) continue;

                double R_j_off = radii[j] - DIELECTRIC_OFFSET;
                hctLigand[i] += computeHCTTerm(r, R_i_off, R_j_off, scaleFactors[j]);
            }
        }

        // Step 3: Born radii (OBC-II)
        vector<double> hctTotal(numAtoms);
        vector<double> bornRadii(numAtoms);
        for (int i = 0; i < numAtoms; i++) {
            hctTotal[i] = hctReceptor[i] + hctLigand[i];
            double R_off = radii[i] - DIELECTRIC_OFFSET;
            double psi = 0.5 * R_off * hctTotal[i];
            double psi2 = psi * psi;
            double psi3 = psi2 * psi;
            double tanh_arg = OBC_ALPHA * psi - OBC_BETA * psi2 + OBC_GAMMA * psi3;
            double tanh_val = tanh(tanh_arg);
            double inner = 1.0 / R_off - tanh_val / radii[i];
            if (inner > 0.0)
                bornRadii[i] = 1.0 / inner;
            else
                bornRadii[i] = 500.0;
        }

        groupBornRadii_[g] = bornRadii;

        // Step 4: GB energy (Still equation) + dE/dR accumulation
        double gbEnergy = 0.0;
        vector<double> dE_dR(numAtoms, 0.0);

        // Self-energy terms
        for (int i = 0; i < numAtoms; i++) {
            double E_self = 0.5 * prefactor * charges[i] * charges[i] / bornRadii[i];
            gbEnergy += E_self;
            dE_dR[i] += -0.5 * prefactor * charges[i] * charges[i] / (bornRadii[i] * bornRadii[i]);
        }

        // Pairwise terms
        for (int i = 0; i < numAtoms; i++) {
            int pi = particles[i];
            for (int j = i + 1; j < numAtoms; j++) {
                if (isExcluded(i, j)) continue;
                int pj = particles[j];

                double dx = posData[pi][0] - posData[pj][0];
                double dy = posData[pi][1] - posData[pj][1];
                double dz = posData[pi][2] - posData[pj][2];
                double r2 = dx * dx + dy * dy + dz * dz;

                double D = bornRadii[i] * bornRadii[j];
                double alpha_val = r2 / (4.0 * D);
                double exp_alpha = exp(-alpha_val);
                double f_gb2 = r2 + D * exp_alpha;
                double f_gb = sqrt(f_gb2);

                double qq = charges[i] * charges[j];
                double E_pair = prefactor * qq / f_gb;
                gbEnergy += E_pair;

                // dE/dR_born for chain rule
                double df_dRi = bornRadii[j] * exp_alpha * (1.0 + alpha_val) / (2.0 * f_gb);
                double df_dRj = bornRadii[i] * exp_alpha * (1.0 + alpha_val) / (2.0 * f_gb);
                dE_dR[i] += -prefactor * qq / f_gb2 * df_dRi;
                dE_dR[j] += -prefactor * qq / f_gb2 * df_dRj;

                // Direct GB forces
                if (includeForces) {
                    double r = sqrt(r2);
                    if (r > 1e-10) {
                        double dE_dr = -prefactor * qq * r * (4.0 - exp_alpha)
                                       / (4.0 * f_gb * f_gb2) * scale;
                        double invR = 1.0 / r;
                        double fx = dE_dr * dx * invR;
                        double fy = dE_dr * dy * invR;
                        double fz = dE_dr * dz * invR;
                        forceData[pi][0] -= fx;
                        forceData[pi][1] -= fy;
                        forceData[pi][2] -= fz;
                        forceData[pj][0] += fx;
                        forceData[pj][1] += fy;
                        forceData[pj][2] += fz;
                    }
                }
            }
        }

        gbEnergy *= scale;
        groupEnergies_[g] += gbEnergy;
        groupLigandEnergies_[g] += gbEnergy;

        // Step 5: Surface area (optional)
        double saEnergy = 0.0;
        vector<double> dE_dR_sa(numAtoms, 0.0);
        if (includeSurfaceArea) {
            double probe = probeRadius;
            for (int i = 0; i < numAtoms; i++) {
                double R_probe_i = radii[i] + probe;
                double ratio = radii[i] / bornRadii[i];
                double ratio6 = ratio * ratio * ratio;
                ratio6 = ratio6 * ratio6;
                double E_sa = surfaceTension * 4.0 * M_PI * R_probe_i * R_probe_i * ratio6;
                saEnergy += E_sa;

                // dE_sa/dR_born = surfaceTension * 4pi * R_probe² * 6 * R^6 / R_born^7 * (-1)
                dE_dR_sa[i] = -6.0 * surfaceTension * 4.0 * M_PI * R_probe_i * R_probe_i
                              * ratio6 / bornRadii[i];
            }
            saEnergy *= scale;
            groupEnergies_[g] += saEnergy;
            groupLigandEnergies_[g] += saEnergy;
        }

        // Step 6: Chain rule forces
        if (includeForces) {
            // Scale dE/dR by alchemical factor
            for (int i = 0; i < numAtoms; i++) {
                dE_dR[i] *= scale;
                if (includeSurfaceArea)
                    dE_dR[i] += dE_dR_sa[i] * scale;
            }

            // Compute dR_born/dHCT (OBC-II chain rule)
            vector<double> dR_dHCT(numAtoms, 0.0);
            for (int i = 0; i < numAtoms; i++) {
                double R_off = radii[i] - DIELECTRIC_OFFSET;
                if (R_off <= 0.0) continue;
                double psi = 0.5 * R_off * hctTotal[i];
                double psi2 = psi * psi;
                double tanh_arg = OBC_ALPHA * psi - OBC_BETA * psi2 + OBC_GAMMA * psi * psi2;
                double tanh_val = tanh(tanh_arg);
                double sech2 = 1.0 - tanh_val * tanh_val;
                double dpsi_dhct = 0.5 * R_off;
                double dtanh_dpsi = sech2 * (OBC_ALPHA - 2.0 * OBC_BETA * psi
                                              + 3.0 * OBC_GAMMA * psi2);
                dR_dHCT[i] = bornRadii[i] * bornRadii[i]
                             * dtanh_dpsi * dpsi_dhct / radii[i];
            }

            // Combined chain rule factor
            vector<double> chainFactor(numAtoms);
            for (int i = 0; i < numAtoms; i++)
                chainFactor[i] = dE_dR[i] * dR_dHCT[i];

            // 6a: Chain rule forces through ligand-ligand HCT
            for (int i = 0; i < numAtoms; i++) {
                int pi = particles[i];
                double R_i_off = radii[i] - DIELECTRIC_OFFSET;

                for (int j = 0; j < numAtoms; j++) {
                    if (i == j) continue;
                    if (isExcluded(i, j)) continue;
                    int pj = particles[j];

                    double dx = posData[pi][0] - posData[pj][0];
                    double dy = posData[pi][1] - posData[pj][1];
                    double dz = posData[pi][2] - posData[pj][2];
                    double r2 = dx * dx + dy * dy + dz * dz;
                    double r = sqrt(r2);
                    if (r < 1e-10) continue;

                    double R_j_off = radii[j] - DIELECTRIC_OFFSET;
                    double dHCT_dr = computeHCTTermDerivative(r, R_i_off, R_j_off, scaleFactors[j]);

                    double forceMag = -chainFactor[i] * dHCT_dr;
                    double invR = 1.0 / r;
                    // Force on atom i
                    forceData[pi][0] += forceMag * dx * invR;
                    forceData[pi][1] += forceMag * dy * invR;
                    forceData[pi][2] += forceMag * dz * invR;
                    // Newton's 3rd law: reaction force on atom j
                    forceData[pj][0] -= forceMag * dx * invR;
                    forceData[pj][1] -= forceMag * dy * invR;
                    forceData[pj][2] -= forceMag * dz * invR;
                }
            }

            // 6b: Chain rule forces through receptor grid HCT
            for (int i = 0; i < numAtoms; i++) {
                int pi = particles[i];
                double forceMag = -chainFactor[i];
                forceData[pi][0] += forceMag * hctRecGradX[i];
                forceData[pi][1] += forceMag * hctRecGradY[i];
                forceData[pi][2] += forceMag * hctRecGradZ[i];
            }
        }
}

void ReferenceCalcGBSAGridForceKernel::runGroups(
        ContextImpl& context, vector<Vec3>& posData, vector<Vec3>& forceData,
        bool includeForces, bool includeEnergy) {
    for (int g = 0; g < numParticleGroups; g++)
        computeGroup(g, posData, forceData, includeForces, includeEnergy);
}

double ReferenceCalcGBSAGridForceKernel::execute(
        ContextImpl& context, bool includeForces, bool includeEnergy) {

    // Build the desolvation grid on first use (auto-generation path).
    if (autoGenerateGrid_ && !desolvationGrid)
        generateDesolvationGrid(context);

    vector<Vec3>& posData = refExtractPositions(context);
    vector<Vec3>& forceData = refExtractForces(context);

    fill(groupEnergies_.begin(), groupEnergies_.end(), 0.0);
    fill(groupLigandEnergies_.begin(), groupLigandEnergies_.end(), 0.0);
    outOfBoundsFlags_.assign((size_t)numParticleGroups * numAtoms, 0);

    runGroups(context, posData, forceData, includeForces, includeEnergy);

    // One-time warning if any ligand atom fell outside the desolvation grid: those
    // atoms received zero receptor screening, silently biasing the result.
    if (!warnedOutOfBounds_) {
        int nOut = 0;
        for (int f : outOfBoundsFlags_) nOut += f;
        if (nOut > 0) {
            std::cerr << "WARNING: GBSAGridForce: " << nOut << " ligand atom(s) fell "
                      << "outside the desolvation grid and received zero receptor "
                      << "screening. Enlarge the grid to cover all ligand positions."
                      << std::endl;
            warnedOutOfBounds_ = true;
        }
    }

    // Deterministic, group-ordered reduction (matches serial Reference exactly).
    double totalEnergy = 0.0;
    for (int g = 0; g < numParticleGroups; g++)
        totalEnergy += groupEnergies_[g];
    return totalEnergy;
}

// ==================== generateDesolvationGrid ====================
//
// Builds the receptor desolvation grid (hct_probe + per-bin N/A/B corrections)
// from the stored receptor parameters. Mirrors python/desolvation_grid_generator.py:
// for each grid point, sum the HCT descreening of a water probe over receptor
// atoms (hct_probe) and accumulate the radius-correction moments per bin. Grid
// points are independent, so parallelFor runs this serially on Reference and
// across the thread pool on the CPU platform.

void ReferenceCalcGBSAGridForceKernel::generateDesolvationGrid(ContextImpl& context) {
    const int nx = genCounts_[0], ny = genCounts_[1], nz = genCounts_[2];
    const int nyz = ny * nz;
    const int numPoints = nx * ny * nz;
    const int nbins = (int)genRThresholds_.size();
    const int nrec = (int)genReceptorRadii_.size();
    const double sp = genSpacing_;
    const double ox = genOrigin_[0], oy = genOrigin_[1], oz = genOrigin_[2];
    const double R_probe_off = probeRadius - DIELECTRIC_OFFSET;

    // Precompute receptor offset radii / scaled radii (matches the Python generator).
    vector<double> recOff(nrec), recScale(nrec), recS(nrec);
    for (int j = 0; j < nrec; j++) {
        recOff[j] = std::max(genReceptorRadii_[j] - DIELECTRIC_OFFSET, 1e-6);
        recScale[j] = genReceptorScales_[j];
        recS[j] = recOff[j] * recScale[j];
    }

    vector<float> hctProbe((size_t)numPoints, 0.0f);
    vector<float> corrN((size_t)nbins * numPoints, 0.0f);
    vector<float> corrA((size_t)nbins * numPoints, 0.0f);
    vector<float> corrB((size_t)nbins * numPoints, 0.0f);

    parallelFor(context, numPoints, [&](int idx) {
        int ix = idx / nyz;
        int rem = idx % nyz;
        int iy = rem / nz;
        int iz = rem % nz;
        double px = ox + ix * sp, py = oy + iy * sp, pz = oz + iz * sp;

        double hp = 0.0;
        vector<double> N(nbins, 0.0), A(nbins, 0.0), B(nbins, 0.0);
        for (int j = 0; j < nrec; j++) {
            double dx = px - genReceptorPositions_[3*j+0];
            double dy = py - genReceptorPositions_[3*j+1];
            double dz = pz - genReceptorPositions_[3*j+2];
            double r = sqrt(dx*dx + dy*dy + dz*dz);
            // Probe HCT descreening from receptor atom j (probe is the receiver).
            hp += computeHCTTerm(r, R_probe_off, recOff[j], recScale[j]);
            if (r <= 1e-6) continue;
            double cross = fabs(r - recS[j]);
            if (cross >= R_probe_off) continue;
            for (int b = 0; b < nbins; b++) {
                if (cross < genRThresholds_[b]) {
                    N[b] += 1.0;
                    A[b] += r - recS[j]*recS[j] / r;
                    B[b] += 0.5 / r;
                }
            }
        }
        hctProbe[idx] = (float)hp;
        for (int b = 0; b < nbins; b++) {
            corrN[(size_t)b*numPoints + idx] = (float)N[b];
            corrA[(size_t)b*numPoints + idx] = (float)A[b];
            corrB[(size_t)b*numPoints + idx] = (float)B[b];
        }
    });

    auto grid = std::make_shared<DesolvationGrid>(nx, ny, nz, sp, probeRadius, genRThresholds_);
    grid->setOrigin(ox, oy, oz);
    grid->setHctProbe(std::move(hctProbe));
    grid->setCorrectionN(std::move(corrN));
    grid->setCorrectionA(std::move(corrA));
    grid->setCorrectionB(std::move(corrB));
    desolvationGrid = grid;
}

// ==================== updateParametersInContext ====================

void ReferenceCalcGBSAGridForceKernel::updateParametersInContext(
        ContextImpl& context, const GBSAGridForce& force) {

    if (numAtoms != force.getNumAtoms())
        throw OpenMMException("Cannot update GBSAGridForce: number of atoms has changed");

    for (int i = 0; i < numAtoms; i++) {
        force.getAtomParameters(i, charges[i], radii[i], scaleFactors[i]);
    }

    double soluteDielectric = force.getSoluteDielectric();
    double solventDielectric = force.getSolventDielectric();
    prefactor = -COULOMB_CONSTANT * (1.0 / soluteDielectric - 1.0 / solventDielectric);

    includeSurfaceArea = force.getIncludeSurfaceArea();
    surfaceTension = force.getSurfaceTension();

    globalScalingFactor = force.getGlobalScalingFactor();
    int nGroups = force.getNumParticleGroups();
    for (int g = 0; g < nGroups; g++) {
        groupScalingFactors[g] = force.getGroupScalingFactor(g);
    }
}

// ==================== computeHessian ====================
//
// Analytical second derivatives of the grid-GB energy. Each particle group is
// isolated, so the Hessian is block-diagonal in the (3*totalParticles) matrix.
// Within a group the energy depends on coordinates directly (the explicit r in
// the Still pair term) and through the Born radii R_i = G(Psi_i), where
// Psi_i = Psi_i^grid(x_i) + sum_{j!=i} HCT(r_ij). The receptor screening enters
// only through the per-atom grid term Psi_i^grid (no receptor degrees of
// freedom), contributing its gradient to the Jacobian self-block and its
// second derivative to the diagonal atom block. The assembly mirrors the CUDA
// kernel: H = H_direct + H_cross + J^T M J + dE/dPsi * d2Psi.

namespace {

// Born transform derivatives dR/dPsi and d2R/dPsi2 for OBC-II Born radii, where
// Psi is the (unscaled) HCT sum. Returns zeros where the radius is capped.
struct GridBornDerivs { double dRdPsi, d2RdPsi2; };

inline GridBornDerivs gridBornDerivs(double R_intrinsic, double R_off,
                                     double hctTotal, double R_born, bool capped) {
    GridBornDerivs d{0.0, 0.0};
    if (capped || R_off <= 0.0) return d;
    double psi = 0.5 * R_off * hctTotal;
    double psi2 = psi * psi;
    double arg = OBC_ALPHA * psi - OBC_BETA * psi2 + OBC_GAMMA * psi2 * psi;
    double t = tanh(arg);
    double sech2 = 1.0 - t * t;
    double darg = OBC_ALPHA - 2.0 * OBC_BETA * psi + 3.0 * OBC_GAMMA * psi2;
    double d2arg = -2.0 * OBC_BETA + 6.0 * OBC_GAMMA * psi;
    double dpsi = 0.5 * R_off;          // dpsi/dPsi
    double Dc = dpsi / R_intrinsic;
    d.dRdPsi = R_born * R_born * sech2 * darg * Dc;
    double A = R_born * R_born, B = sech2, C = darg;
    double dA = 2.0 * R_born * d.dRdPsi;
    double dargP = darg * dpsi;
    double dB = -2.0 * sech2 * t * dargP;
    double dC = d2arg * dpsi;
    d.d2RdPsi2 = (dA * B * C + A * dB * C + A * B * dC) * Dc;
    return d;
}

struct GridStillPair { double f, f2, et; };
inline GridStillPair gridStillPair(double r2, double Ra, double Rb) {
    GridStillPair s;
    double RaRb = Ra * Rb;
    s.et = exp(-r2 / (4.0 * RaRb));
    s.f2 = r2 + RaRb * s.et;
    s.f = sqrt(s.f2);
    return s;
}

}  // namespace

void ReferenceCalcGBSAGridForceKernel::parallelFor(ContextImpl& context, int count, const std::function<void(int)>& body) {
    for (int i = 0; i < count; i++) body(i);
}

void ReferenceCalcGBSAGridForceKernel::computeHessian(ContextImpl& context) {
    vector<Vec3>& posData = refExtractPositions(context);

    int totalParticles = numParticleGroups * numAtoms;
    int dim3N = 3 * totalParticles;
    hessianBlocks_.assign(6 * totalParticles, 0.0);
    fullHessian_.assign(static_cast<size_t>(dim3N) * dim3N, 0.0);
    if (totalParticles == 0) return;

    const int hmap[3][3] = {{0, 3, 4}, {3, 1, 5}, {4, 5, 2}};  // (a,b)->block index

    parallelFor(context, numParticleGroups, [&](int g) {
        double scale = globalScalingFactor * groupScalingFactors[g];
        if (scale == 0.0) return;
        const vector<int>& particles = groupParticleIndices[g];
        int N = numAtoms;
        int n3 = 3 * N;

        // ---- Forward quantities the Hessian differentiates ----
        // Receptor grid HCT (value + gradient + second derivative) and the
        // ligand-ligand HCT sum (honoring exclusions, as in execute()).
        vector<double> hctReceptor(N, 0.0), hctLigand(N, 0.0);
        vector<double> gridGradX(N, 0.0), gridGradY(N, 0.0), gridGradZ(N, 0.0);
        vector<double> gridHess(static_cast<size_t>(N) * 6, 0.0);
        for (int i = 0; i < N; i++) {
            int pi = particles[i];
            double Ri_off = radii[i] - DIELECTRIC_OFFSET;
            double gx, gy, gz, hh[6];
            hctReceptor[i] = interpolateReceptorHCT(
                posData[pi][0], posData[pi][1], posData[pi][2],
                Ri_off, true, gx, gy, gz, hh);
            gridGradX[i] = gx; gridGradY[i] = gy; gridGradZ[i] = gz;
            for (int c = 0; c < 6; c++) gridHess[(size_t)i*6 + c] = hh[c];
            for (int j = 0; j < N; j++) {
                if (i == j || isExcluded(i, j)) continue;
                int pj = particles[j];
                double dx = posData[pi][0]-posData[pj][0];
                double dy = posData[pi][1]-posData[pj][1];
                double dz = posData[pi][2]-posData[pj][2];
                double r = sqrt(dx*dx+dy*dy+dz*dz);
                if (r < 1e-10) continue;
                double Rj_off = radii[j] - DIELECTRIC_OFFSET;
                hctLigand[i] += computeHCTTerm(r, Ri_off, Rj_off, scaleFactors[j]);
            }
        }

        vector<double> hctTotal(N), born(N);
        vector<char> capped(N, 0);
        for (int i = 0; i < N; i++) {
            hctTotal[i] = hctReceptor[i] + hctLigand[i];
            double R_off = radii[i] - DIELECTRIC_OFFSET;
            double psi = 0.5 * R_off * hctTotal[i];
            double psi2 = psi*psi;
            double t = tanh(OBC_ALPHA*psi - OBC_BETA*psi2 + OBC_GAMMA*psi2*psi);
            double inner = 1.0/R_off - t/radii[i];
            if (inner > 0.0) born[i] = 1.0/inner;
            else { born[i] = 500.0; capped[i] = 1; }
        }

        // dE/dR (GB self + ligand-ligand pairs) and surface area.
        vector<double> dE_dR(N, 0.0);
        for (int i = 0; i < N; i++)
            dE_dR[i] += -0.5 * prefactor * charges[i]*charges[i] / (born[i]*born[i]);
        for (int i = 0; i < N; i++) {
            int pi = particles[i];
            for (int j = i+1; j < N; j++) {
                if (isExcluded(i, j)) continue;
                int pj = particles[j];
                double dx = posData[pi][0]-posData[pj][0];
                double dy = posData[pi][1]-posData[pj][1];
                double dz = posData[pi][2]-posData[pj][2];
                double r2 = dx*dx+dy*dy+dz*dz;
                double D = born[i]*born[j];
                double al = r2/(4.0*D), et = exp(-al);
                double f2 = r2 + D*et, f = sqrt(f2);
                double qq = charges[i]*charges[j];
                double dfi = born[j]*et*(1.0+al)/(2.0*f);
                double dfj = born[i]*et*(1.0+al)/(2.0*f);
                dE_dR[i] += -prefactor*qq/f2*dfi;
                dE_dR[j] += -prefactor*qq/f2*dfj;
            }
        }
        if (includeSurfaceArea) {
            for (int i = 0; i < N; i++) {
                double Rp = radii[i] + probeRadius;
                double ratio = radii[i]/born[i];
                double ratio6 = ratio*ratio*ratio; ratio6 *= ratio6;
                dE_dR[i] += -6.0 * surfaceTension * 4.0 * M_PI * Rp*Rp * ratio6 / born[i];
            }
        }

        // Per-atom Born transform derivatives and dE/dPsi.
        vector<double> dRdPsi(N), d2RdPsi2(N), dE_dHCT(N);
        for (int i = 0; i < N; i++) {
            double R_off = radii[i] - DIELECTRIC_OFFSET;
            GridBornDerivs bd = gridBornDerivs(radii[i], R_off, hctTotal[i], born[i], capped[i]);
            dRdPsi[i] = bd.dRdPsi;
            d2RdPsi2[i] = bd.d2RdPsi2;
            dE_dHCT[i] = dE_dR[i] * bd.dRdPsi;
        }

        // ---- Jacobian J[k][3*i+a] = dPsi_k/dx_{i,a} ----
        // Ligand-ligand pairwise terms plus the receptor grid gradient (grid
        // depends only on x_k, so it enters only the self block).
        vector<double> J(static_cast<size_t>(N) * n3, 0.0);
        for (int k = 0; k < N; k++) {
            int pk = particles[k];
            double Rk_off = radii[k] - DIELECTRIC_OFFSET;
            double jsx = gridGradX[k], jsy = gridGradY[k], jsz = gridGradZ[k];
            for (int j = 0; j < N; j++) {
                if (j == k || isExcluded(k, j)) continue;
                int pj = particles[j];
                double dx = posData[pk][0]-posData[pj][0];
                double dy = posData[pk][1]-posData[pj][1];
                double dz = posData[pk][2]-posData[pj][2];
                double r = sqrt(dx*dx+dy*dy+dz*dz);
                if (r < 1e-10) continue;
                double Rj_off = radii[j] - DIELECTRIC_OFFSET;
                double I1 = computeHCTTermDerivative(r, Rk_off, Rj_off, scaleFactors[j]);
                double invr = 1.0/r;
                jsx += I1*dx*invr; jsy += I1*dy*invr; jsz += I1*dz*invr;
                J[(size_t)k*n3 + 3*j + 0] = -I1*dx*invr;
                J[(size_t)k*n3 + 3*j + 1] = -I1*dy*invr;
                J[(size_t)k*n3 + 3*j + 2] = -I1*dz*invr;
            }
            J[(size_t)k*n3 + 3*k + 0] = jsx;
            J[(size_t)k*n3 + 3*k + 1] = jsy;
            J[(size_t)k*n3 + 3*k + 2] = jsz;
        }

        // ---- Coupling matrix M[k][l] = d2E/dPsi_k dPsi_l ----
        vector<double> M(static_cast<size_t>(N) * N, 0.0);
        for (int k = 0; k < N; k++) {
            int pk = particles[k];
            double q_k = charges[k];
            double Rk = born[k], dRk = dRdPsi[k];
            double diag = prefactor * q_k*q_k / (Rk*Rk*Rk) * dRk*dRk;  // self
            if (includeSurfaceArea) {
                double Rp = radii[k] + probeRadius;
                double ratio = radii[k]/Rk;
                double ratio6 = ratio*ratio*ratio; ratio6 *= ratio6;
                double E_SA = surfaceTension * 4.0 * M_PI * Rp*Rp * ratio6;
                diag += 42.0 * E_SA / (Rk*Rk) * dRk*dRk;
            }
            diag += dE_dR[k] * d2RdPsi2[k];     // OBC curvature
            for (int l = 0; l < N; l++) {
                if (l == k || isExcluded(k, l)) continue;
                int pl = particles[l];
                double dx = posData[pl][0]-posData[pk][0];
                double dy = posData[pl][1]-posData[pk][1];
                double dz = posData[pl][2]-posData[pk][2];
                double r2 = dx*dx+dy*dy+dz*dz;
                double Rl = born[l], dRl = dRdPsi[l];
                GridStillPair sp = gridStillPair(r2, Rk, Rl);
                double f = sp.f, et = sp.et;
                double C = prefactor * q_k * charges[l];
                double df2_dRk = et*(Rl + 0.25*r2/Rk);
                double df_dRk = df2_dRk/(2.0*f);
                double dalpha_dRk = et*r2/(4.0*Rk*Rk*Rl);
                double d2f2_dRk2 = dalpha_dRk*(Rl + 0.25*r2/Rk) + et*(-0.25*r2/(Rk*Rk));
                double d2f_dRk2 = d2f2_dRk2/(2.0*f) - df2_dRk*df2_dRk/(4.0*f*f*f);
                double d2E_dRk2 = C*(2.0*df_dRk*df_dRk/(f*f*f) - d2f_dRk2/(f*f));
                diag += d2E_dRk2 * dRk * dRk;
                double df2_dRl = et*(Rk + 0.25*r2/Rl);
                double df_dRl = df2_dRl/(2.0*f);
                double dalpha_dRl = et*r2/(4.0*Rk*Rl*Rl);
                double d2f2_dRkRl = dalpha_dRl*(Rl + 0.25*r2/Rk) + et;
                double d2f_dRkRl = d2f2_dRkRl/(2.0*f) - df2_dRk*df2_dRl/(4.0*f*f*f);
                double d2E_dRkRl = C*(2.0*df_dRk*df_dRl/(f*f*f) - d2f_dRkRl/(f*f));
                M[(size_t)k*N + l] = d2E_dRkRl * dRk * dRl;
            }
            M[(size_t)k*N + k] = diag;
        }

        // ---- Assemble the group block ----
        vector<double> Hloc(static_cast<size_t>(n3) * n3, 0.0);
        for (int ai = 0; ai < N; ai++) {
            int pai = particles[ai];
            for (int aj = ai; aj < N; aj++) {
                int paj = particles[aj];
                for (int a = 0; a < 3; a++) {
                    int row = 3*ai + a;
                    int bStart = (aj == ai) ? a : 0;
                    for (int b = bStart; b < 3; b++) {
                        int col = 3*aj + b;
                        double Hval = 0.0;

                        if (ai == aj) {
                            double pix = posData[pai][0], piy = posData[pai][1], piz = posData[pai][2];
                            double q_i = charges[ai];
                            double Ri = born[ai];
                            double Ri_off = radii[ai] - DIELECTRIC_OFFSET;
                            double Si = Ri_off * scaleFactors[ai];
                            for (int l = 0; l < N; l++) {
                                if (l == ai || isExcluded(ai, l)) continue;
                                int pl = particles[l];
                                double dx = posData[pl][0]-pix, dy = posData[pl][1]-piy, dz = posData[pl][2]-piz;
                                double r2 = dx*dx+dy*dy+dz*dz, r = sqrt(r2);
                                if (r < 1e-10) continue;
                                double Rl = born[l];
                                GridStillPair sp = gridStillPair(r2, Ri, Rl);
                                double f = sp.f, et = sp.et;
                                double C = prefactor * q_i * charges[l];
                                double df2_dr = 2.0*r*(1.0 - 0.25*et);
                                double df_dr = df2_dr/(2.0*f);
                                double dEdr = -C*df_dr/(f*f);
                                double d2f2_dr2 = 2.0 - 0.5*et + 0.25*r2*et/(Ri*Rl);
                                double d2f_dr2 = d2f2_dr2/(2.0*f) - df2_dr*df2_dr/(4.0*f*f*f);
                                double d2Edr2 = C*(2.0*df_dr*df_dr/(f*f*f) - d2f_dr2/(f*f));
                                double Rl_off = radii[l] - DIELECTRIC_OFFSET;
                                double Sl = Rl_off * scaleFactors[l];
                                if (Ri_off < r + Sl) {
                                    double I1 = computeHCTTermDerivative(r, Ri_off, Rl_off, scaleFactors[l]);
                                    double I2 = computeHCTTermSecondDerivative(r, Ri_off, Rl_off, scaleFactors[l]);
                                    dEdr += dE_dHCT[ai]*I1; d2Edr2 += dE_dHCT[ai]*I2;
                                }
                                if (Rl_off < r + Si) {
                                    double I1 = computeHCTTermDerivative(r, Rl_off, Ri_off, scaleFactors[ai]);
                                    double I2 = computeHCTTermSecondDerivative(r, Rl_off, Ri_off, scaleFactors[ai]);
                                    dEdr += dE_dHCT[l]*I1; d2Edr2 += dE_dHCT[l]*I2;
                                }
                                double dalpha_dRi = et*r2/(4.0*Ri*Ri*Rl);
                                double d2f2_dr_dRi = 2.0*r*(-0.25*dalpha_dRi);
                                double df2_dRi = et*(Rl + 0.25*r2/Ri);
                                double d2f_dr_dRi = d2f2_dr_dRi/(2.0*f) - df2_dr*df2_dRi/(4.0*f*f*f);
                                double d2E_dr_dRi = C*(2.0*df_dr*(df2_dRi/(2.0*f))/(f*f*f) - d2f_dr_dRi/(f*f));
                                double g_i = d2E_dr_dRi * dRdPsi[ai];
                                double dalpha_dRl = et*r2/(4.0*Ri*Rl*Rl);
                                double d2f2_dr_dRl = 2.0*r*(-0.25*dalpha_dRl);
                                double df2_dRl = et*(Ri + 0.25*r2/Rl);
                                double d2f_dr_dRl = d2f2_dr_dRl/(2.0*f) - df2_dr*df2_dRl/(4.0*f*f*f);
                                double d2E_dr_dRl = C*(2.0*df_dr*(df2_dRl/(2.0*f))/(f*f*f) - d2f_dr_dRl/(f*f));
                                double g_l = d2E_dr_dRl * dRdPsi[l];
                                double Dd[3] = {dx, dy, dz};
                                double ir = 1.0/r, ir2 = ir*ir;
                                Hval += (d2Edr2 - dEdr*ir)*Dd[a]*Dd[b]*ir2
                                      + ((a==b) ? dEdr*ir : 0.0);
                                double dr_a = -Dd[a]*ir, dr_b = -Dd[b]*ir;
                                Hval += g_i*(dr_a*J[(size_t)ai*n3 + 3*ai + b] + J[(size_t)ai*n3 + 3*ai + a]*dr_b);
                                Hval += g_l*(dr_a*J[(size_t)l*n3 + 3*ai + b] + J[(size_t)l*n3 + 3*ai + a]*dr_b);
                            }
                            // Receptor grid self-Hessian: dE/dPsi_ai * d2Psi_grid/dx dx.
                            Hval += dE_dHCT[ai] * gridHess[(size_t)ai*6 + hmap[a][b]];
                        } else {
                            double pix = posData[pai][0], piy = posData[pai][1], piz = posData[pai][2];
                            double pjx = posData[paj][0], pjy = posData[paj][1], pjz = posData[paj][2];
                            double q_i = charges[ai], q_j = charges[aj];
                            double Ri = born[ai], Rj = born[aj];
                            double dx = pjx-pix, dy = pjy-piy, dz = pjz-piz;
                            double r2 = dx*dx+dy*dy+dz*dz, r = sqrt(r2);
                            if (r >= 1e-10 && !isExcluded(ai, aj)) {
                                GridStillPair sp = gridStillPair(r2, Ri, Rj);
                                double f = sp.f, et = sp.et;
                                double C = prefactor * q_i * q_j;
                                double df2_dr = 2.0*r*(1.0 - 0.25*et);
                                double df_dr = df2_dr/(2.0*f);
                                double dEdr = -C*df_dr/(f*f);
                                double d2f2_dr2 = 2.0 - 0.5*et + 0.25*r2*et/(Ri*Rj);
                                double d2f_dr2 = d2f2_dr2/(2.0*f) - df2_dr*df2_dr/(4.0*f*f*f);
                                double d2Edr2 = C*(2.0*df_dr*df_dr/(f*f*f) - d2f_dr2/(f*f));
                                double Ri_off = radii[ai]-DIELECTRIC_OFFSET, Rj_off = radii[aj]-DIELECTRIC_OFFSET;
                                double Si = Ri_off*scaleFactors[ai], Sj = Rj_off*scaleFactors[aj];
                                if (Ri_off < r + Sj) {
                                    double I1 = computeHCTTermDerivative(r, Ri_off, Rj_off, scaleFactors[aj]);
                                    double I2 = computeHCTTermSecondDerivative(r, Ri_off, Rj_off, scaleFactors[aj]);
                                    dEdr += dE_dHCT[ai]*I1; d2Edr2 += dE_dHCT[ai]*I2;
                                }
                                if (Rj_off < r + Si) {
                                    double I1 = computeHCTTermDerivative(r, Rj_off, Ri_off, scaleFactors[ai]);
                                    double I2 = computeHCTTermSecondDerivative(r, Rj_off, Ri_off, scaleFactors[ai]);
                                    dEdr += dE_dHCT[aj]*I1; d2Edr2 += dE_dHCT[aj]*I2;
                                }
                                double ir = 1.0/r, ir2 = ir*ir;
                                double Dd[3] = {dx, dy, dz};
                                Hval += -(d2Edr2 - dEdr*ir)*Dd[a]*Dd[b]*ir2
                                      - ((a==b) ? dEdr*ir : 0.0);
                            }
                            // g-terms: pairs involving ai (column), then aj (row).
                            for (int l = 0; l < N; l++) {
                                if (l == ai || isExcluded(ai, l)) continue;
                                int pl = particles[l];
                                double dx_=posData[pl][0]-pix, dy_=posData[pl][1]-piy, dz_=posData[pl][2]-piz;
                                double r2_=dx_*dx_+dy_*dy_+dz_*dz_, r_=sqrt(r2_);
                                if (r_ < 1e-10) continue;
                                double Rl = born[l];
                                GridStillPair sp = gridStillPair(r2_, Ri, Rl);
                                double f=sp.f, et=sp.et; double C=prefactor*q_i*charges[l];
                                double df2 = 2.0*r_*(1.0-0.25*et); double df = df2/(2.0*f);
                                double da_i = et*r2_/(4.0*Ri*Ri*Rl);
                                double d2f2ri = 2.0*r_*(-0.25*da_i);
                                double df2ri = et*(Rl+0.25*r2_/Ri);
                                double d2fri = d2f2ri/(2.0*f) - df2*df2ri/(4.0*f*f*f);
                                double g_i = C*(2.0*df*(df2ri/(2.0*f))/(f*f*f) - d2fri/(f*f))*dRdPsi[ai];
                                double da_l = et*r2_/(4.0*Ri*Rl*Rl);
                                double d2f2rl = 2.0*r_*(-0.25*da_l);
                                double df2rl = et*(Ri+0.25*r2_/Rl);
                                double d2frl = d2f2rl/(2.0*f) - df2*df2rl/(4.0*f*f*f);
                                double g_l = C*(2.0*df*(df2rl/(2.0*f))/(f*f*f) - d2frl/(f*f))*dRdPsi[l];
                                double Dd_[3]={dx_,dy_,dz_};
                                double dr_a = -Dd_[a]/r_;
                                double v_col = g_i*J[(size_t)ai*n3+col] + g_l*J[(size_t)l*n3+col];
                                Hval += dr_a*v_col;
                            }
                            for (int m = 0; m < N; m++) {
                                if (m == aj || isExcluded(aj, m)) continue;
                                int pm = particles[m];
                                double dx_=posData[pm][0]-pjx, dy_=posData[pm][1]-pjy, dz_=posData[pm][2]-pjz;
                                double r2_=dx_*dx_+dy_*dy_+dz_*dz_, r_=sqrt(r2_);
                                if (r_ < 1e-10) continue;
                                double Rm = born[m];
                                GridStillPair sp = gridStillPair(r2_, Rj, Rm);
                                double f=sp.f, et=sp.et; double C=prefactor*q_j*charges[m];
                                double df2 = 2.0*r_*(1.0-0.25*et); double df = df2/(2.0*f);
                                double da_j = et*r2_/(4.0*Rj*Rj*Rm);
                                double d2f2rj = 2.0*r_*(-0.25*da_j);
                                double df2rj = et*(Rm+0.25*r2_/Rj);
                                double d2frj = d2f2rj/(2.0*f) - df2*df2rj/(4.0*f*f*f);
                                double g_j = C*(2.0*df*(df2rj/(2.0*f))/(f*f*f) - d2frj/(f*f))*dRdPsi[aj];
                                double da_m = et*r2_/(4.0*Rj*Rm*Rm);
                                double d2f2rm = 2.0*r_*(-0.25*da_m);
                                double df2rm = et*(Rj+0.25*r2_/Rm);
                                double d2frm = d2f2rm/(2.0*f) - df2*df2rm/(4.0*f*f*f);
                                double g_m = C*(2.0*df*(df2rm/(2.0*f))/(f*f*f) - d2frm/(f*f))*dRdPsi[m];
                                double Dd_[3]={dx_,dy_,dz_};
                                double dr_b = -Dd_[b]/r_;
                                double v_row = g_j*J[(size_t)aj*n3+row] + g_m*J[(size_t)m*n3+row];
                                Hval += v_row*dr_b;
                            }
                        }

                        // J^T M J (Born coupling across all atoms).
                        for (int kk = 0; kk < N; kk++) {
                            double Jk = J[(size_t)kk*n3 + row];
                            if (fabs(Jk) < 1e-18) continue;
                            for (int ll = 0; ll < N; ll++) {
                                double Mkl = M[(size_t)kk*N + ll];
                                if (fabs(Mkl) < 1e-18) continue;
                                Hval += Jk * Mkl * J[(size_t)ll*n3 + col];
                            }
                        }

                        Hloc[(size_t)row*n3 + col] = Hval;
                        if (col != row) Hloc[(size_t)col*n3 + row] = Hval;
                    }
                }
            }
        }

        // Scale and scatter the group block into the global (slot-indexed) matrix.
        for (int r = 0; r < n3; r++) {
            int slotRow = 3*(g*N + r/3) + (r%3);
            for (int c = 0; c < n3; c++) {
                int slotCol = 3*(g*N + c/3) + (c%3);
                fullHessian_[(size_t)slotRow*dim3N + slotCol] = scale * Hloc[(size_t)r*n3 + c];
            }
        }
    });

    // Extract per-atom diagonal blocks [dxx, dyy, dzz, dxy, dxz, dyz].
    for (int i = 0; i < totalParticles; i++) {
        int base = 3*i;
        hessianBlocks_[6*i + 0] = fullHessian_[(size_t)(base+0)*dim3N + (base+0)];
        hessianBlocks_[6*i + 1] = fullHessian_[(size_t)(base+1)*dim3N + (base+1)];
        hessianBlocks_[6*i + 2] = fullHessian_[(size_t)(base+2)*dim3N + (base+2)];
        hessianBlocks_[6*i + 3] = fullHessian_[(size_t)(base+0)*dim3N + (base+1)];
        hessianBlocks_[6*i + 4] = fullHessian_[(size_t)(base+0)*dim3N + (base+2)];
        hessianBlocks_[6*i + 5] = fullHessian_[(size_t)(base+1)*dim3N + (base+2)];
    }
}

// ==================== accessors ====================

double ReferenceCalcGBSAGridForceKernel::getGroupEnergy(int groupIndex) const {
    if (groupIndex < 0 || groupIndex >= numParticleGroups)
        throw OpenMMException("GBSAGridForce: group index out of range");
    return groupEnergies_[groupIndex];
}

double ReferenceCalcGBSAGridForceKernel::getGroupLigandDesolvationEnergy(int groupIndex) const {
    if (groupIndex < 0 || groupIndex >= numParticleGroups)
        throw OpenMMException("GBSAGridForce: group index out of range");
    return groupLigandEnergies_[groupIndex];
}

vector<double> ReferenceCalcGBSAGridForceKernel::getGroupBornRadii(int groupIndex) const {
    if (groupIndex < 0 || groupIndex >= numParticleGroups)
        throw OpenMMException("GBSAGridForce: group index out of range");
    return groupBornRadii_[groupIndex];
}

vector<double> ReferenceCalcGBSAGridForceKernel::getHessianBlocks() const {
    return hessianBlocks_;
}

vector<double> ReferenceCalcGBSAGridForceKernel::getFullHessian() const {
    return fullHessian_;
}

}  // namespace GridForcePlugin
