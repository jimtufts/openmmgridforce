/* -------------------------------------------------------------------------- *
 *                              OpenMMGridForce                               *
 * -------------------------------------------------------------------------- *
 * Reference platform implementation of IsolatedGBSAForce kernel.            *
 * Computes GB implicit solvation with HCT/OBC-II Born radii for isolated    *
 * particle groups on CPU using double precision throughout.                  *
 * -------------------------------------------------------------------------- */

#include "ReferenceIsolatedGBSAKernels.h"
#include "ReferenceGridInterpolation.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/ContextImpl.h"

#include <cmath>
#include <algorithm>

using namespace OpenMM;
using namespace std;

namespace GridForcePlugin {

// OBC-II parameters
static constexpr double OBC_ALPHA = 1.0;
static constexpr double OBC_BETA = 0.8;
static constexpr double OBC_GAMMA = 4.85;

// ==================== Helper: Born radii from HCT ====================

void ReferenceCalcIsolatedGBSAForceKernel::computeBornRadii(
        const vector<double>& hctTotal,
        vector<double>& bornRadii) const {

    bornRadii.resize(numAtoms);
    for (int i = 0; i < numAtoms; i++) {
        double R_off = radii[i] - DIELECTRIC_OFFSET;
        if (R_off <= 0.0) {
            bornRadii[i] = radii[i];
            continue;
        }

        if (gbMethod == IsolatedGBSAForce::HCT) {
            double inner = 1.0 / R_off - 0.5 * R_off * hctTotal[i];
            if (inner > 0.0)
                bornRadii[i] = 1.0 / inner;
            else
                bornRadii[i] = 500.0;  // very large Born radius (atom fully buried)
        } else {
            // OBC-II: psi = 0.5 * R_off * hctTotal (standard OBC-II formula)
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
    }
}

// ==================== Helper: GB energy + dE/dR accumulation ====================

double ReferenceCalcIsolatedGBSAForceKernel::computeGBEnergy(
        int groupIndex,
        const vector<Vec3>& positions,
        const vector<double>& bornRadii,
        vector<double>& dE_dR) const {

    const vector<int>& particles = groupParticleIndices[groupIndex];
    double energy = 0.0;

    // Self-energy terms
    for (int i = 0; i < numAtoms; i++) {
        double E_self = 0.5 * prefactor * charges[i] * charges[i] / bornRadii[i];
        energy += E_self;
        dE_dR[i] += -0.5 * prefactor * charges[i] * charges[i] / (bornRadii[i] * bornRadii[i]);
    }

    // Pairwise terms
    for (int i = 0; i < numAtoms; i++) {
        int pi = particles[i];
        for (int j = i + 1; j < numAtoms; j++) {
            int pj = particles[j];

            double dx = positions[pi][0] - positions[pj][0];
            double dy = positions[pi][1] - positions[pj][1];
            double dz = positions[pi][2] - positions[pj][2];
            double r2 = dx * dx + dy * dy + dz * dz;

            double D = bornRadii[i] * bornRadii[j];
            double alpha = r2 / (4.0 * D);
            double exp_alpha = exp(-alpha);
            double f_gb2 = r2 + D * exp_alpha;
            double f_gb = sqrt(f_gb2);

            double qq = charges[i] * charges[j];
            double E_pair = prefactor * qq / f_gb;
            energy += E_pair;

            // dE/dR_born_i from this pair
            double df_dRi = bornRadii[j] * exp_alpha * (1.0 + alpha) / (2.0 * f_gb);
            double df_dRj = bornRadii[i] * exp_alpha * (1.0 + alpha) / (2.0 * f_gb);

            double dE_df = -prefactor * qq / f_gb2;
            dE_dR[i] += dE_df * df_dRi;
            dE_dR[j] += dE_df * df_dRj;
        }
    }

    return energy;
}

// ==================== Helper: Surface area energy ====================

double ReferenceCalcIsolatedGBSAForceKernel::computeSurfaceAreaEnergy(
        const vector<double>& bornRadii,
        vector<double>& dE_dR) const {

    double probe = 0.14;  // default probe radius in nm
    if (receptorMode == IsolatedGBSAForce::GRID && desolvationGrid)
        probe = desolvationGrid->getProbeRadius();

    double energy = 0.0;
    for (int i = 0; i < numAtoms; i++) {
        double R_i = radii[i];
        double R_born = bornRadii[i];
        double ratio = R_i / R_born;
        double ratio6 = ratio * ratio * ratio * ratio * ratio * ratio;
        double Rp = R_i + probe;
        double E_sa = surfaceTension * 4.0 * M_PI * Rp * Rp * ratio6;
        energy += E_sa;

        // dE_sa/dR_born = surfaceTension * 4*pi * Rp^2 * (-6) * R_i^6 / R_born^7
        dE_dR[i] += -6.0 * E_sa / R_born;
    }
    return energy;
}

// ==================== initialize ====================

void ReferenceCalcIsolatedGBSAForceKernel::initialize(
        const System& system, const IsolatedGBSAForce& force) {

    numAtoms = force.getNumAtoms();
    if (numAtoms == 0)
        throw OpenMMException("IsolatedGBSAForce: no atoms defined");

    // Store configuration
    gbMethod = force.getGBMethod();
    receptorMode = force.getReceptorMode();
    cutoffDistance = force.getCutoffDistance();
    receptorLocalityCutoff = force.getReceptorLocalityCutoff();
    includeSurfaceArea = force.getIncludeSurfaceArea();
    surfaceTension = force.getSurfaceTension();
    interpolationMethod = force.getInterpolationMethod();

    // Compute GB prefactor
    double soluteDielectric = force.getSoluteDielectric();
    double solventDielectric = force.getSolventDielectric();
    prefactor = -COULOMB_CONSTANT * (1.0 / soluteDielectric - 1.0 / solventDielectric);

    // Extract atom parameters
    charges.resize(numAtoms);
    radii.resize(numAtoms);
    scaleFactors.resize(numAtoms);
    for (int i = 0; i < numAtoms; i++) {
        force.getAtomParameters(i, charges[i], radii[i], scaleFactors[i]);
    }

    // Process particle groups
    int nGroups = force.getNumParticleGroups();
    if (nGroups > 0) {
        numParticleGroups = nGroups;
        groupParticleIndices.resize(nGroups);
        for (int g = 0; g < nGroups; g++) {
            string name;
            vector<int> indices;
            force.getParticleGroup(g, name, indices);
            groupParticleIndices[g] = indices;
        }
    } else {
        numParticleGroups = 1;
        groupParticleIndices.resize(1);
        const auto& particles = force.getParticles();
        if (!particles.empty()) {
            groupParticleIndices[0] = particles;
        } else {
            groupParticleIndices[0].resize(numAtoms);
            for (int i = 0; i < numAtoms; i++)
                groupParticleIndices[0][i] = i;
        }
    }

    // Alchemical scaling
    globalScalingFactor = force.getGlobalScalingFactor();
    groupScalingFactors.resize(numParticleGroups, 1.0);
    for (int g = 0; g < nGroups; g++) {
        groupScalingFactors[g] = force.getGroupScalingFactor(g);
    }

    // GRID mode setup
    if (receptorMode == IsolatedGBSAForce::GRID) {
        desolvationGrid = force.getDesolvationGrid();
        if (!desolvationGrid)
            throw OpenMMException("IsolatedGBSAForce: GRID mode requires a desolvation grid");
    }

    // PAIRWISE mode setup
    if (receptorMode == IsolatedGBSAForce::PAIRWISE) {
        numReceptorAtoms = force.getNumReceptorAtoms();
        if (numReceptorAtoms == 0)
            throw OpenMMException("IsolatedGBSAForce: PAIRWISE mode requires receptor atoms");

        receptorPositions = force.getReceptorPositions();
        receptorCharges.resize(numReceptorAtoms);
        receptorRadii.resize(numReceptorAtoms);
        receptorScaleFactors.resize(numReceptorAtoms);
        for (int i = 0; i < numReceptorAtoms; i++) {
            force.getReceptorAtomParameters(i, receptorCharges[i],
                                             receptorRadii[i], receptorScaleFactors[i]);
        }

        // Compute receptor self-HCT (receptor-receptor, constant for all groups)
        receptorSelfHCT.resize(numReceptorAtoms, 0.0);
        for (int i = 0; i < numReceptorAtoms; i++) {
            double R_i_off = receptorRadii[i] - DIELECTRIC_OFFSET;
            for (int j = 0; j < numReceptorAtoms; j++) {
                if (i == j) continue;
                double dx = receptorPositions[i * 3] - receptorPositions[j * 3];
                double dy = receptorPositions[i * 3 + 1] - receptorPositions[j * 3 + 1];
                double dz = receptorPositions[i * 3 + 2] - receptorPositions[j * 3 + 2];
                double r = sqrt(dx * dx + dy * dy + dz * dz);

                if (cutoffDistance > 0.0 && r > cutoffDistance) continue;

                double R_j_off = receptorRadii[j] - DIELECTRIC_OFFSET;
                receptorSelfHCT[i] += computeHCTTerm(r, R_i_off, R_j_off, receptorScaleFactors[j]);
            }
        }

        // Compute receptor Born radii without ligand
        receptorBornRadiiRef.resize(numReceptorAtoms);
        for (int i = 0; i < numReceptorAtoms; i++) {
            double R_off = receptorRadii[i] - DIELECTRIC_OFFSET;
            if (gbMethod == IsolatedGBSAForce::HCT) {
                double inner = 1.0 / R_off - 0.5 * R_off * receptorSelfHCT[i];
                receptorBornRadiiRef[i] = (inner > 0.0) ? 1.0 / inner : 500.0;
            } else {
                double psi = 0.5 * R_off * receptorSelfHCT[i];
                double tanh_val = tanh(OBC_ALPHA * psi - OBC_BETA * psi * psi
                                       + OBC_GAMMA * psi * psi * psi);
                double inner = 1.0 / R_off - tanh_val / receptorRadii[i];
                receptorBornRadiiRef[i] = (inner > 0.0) ? 1.0 / inner : 500.0;
            }
        }

        // Compute receptor reference energy (without ligand)
        receptorReferenceEnergy = 0.0;
        for (int i = 0; i < numReceptorAtoms; i++) {
            // Self term
            receptorReferenceEnergy += 0.5 * prefactor * receptorCharges[i] * receptorCharges[i]
                                       / receptorBornRadiiRef[i];
            // Pair terms
            for (int j = i + 1; j < numReceptorAtoms; j++) {
                double dx = receptorPositions[i * 3] - receptorPositions[j * 3];
                double dy = receptorPositions[i * 3 + 1] - receptorPositions[j * 3 + 1];
                double dz = receptorPositions[i * 3 + 2] - receptorPositions[j * 3 + 2];
                double r2 = dx * dx + dy * dy + dz * dz;

                double D = receptorBornRadiiRef[i] * receptorBornRadiiRef[j];
                double exp_alpha = exp(-r2 / (4.0 * D));
                double f_gb = sqrt(r2 + D * exp_alpha);
                receptorReferenceEnergy += prefactor * receptorCharges[i]
                                           * receptorCharges[j] / f_gb;
            }
        }
    }

    // Allocate per-group result storage
    groupEnergies_.resize(numParticleGroups, 0.0);
    groupLigandSelfEnergies_.resize(numParticleGroups, 0.0);
    groupReceptorContributions_.resize(numParticleGroups, 0.0);
    groupReceptorDesolvations_.resize(numParticleGroups, 0.0);
    groupCrossTermEnergies_.resize(numParticleGroups, 0.0);
    groupBornRadii_.resize(numParticleGroups);
    groupAtomEnergies_.resize(numParticleGroups);
    groupReceptorBornRadii_.resize(numParticleGroups);
}

// ==================== execute ====================

void ReferenceCalcIsolatedGBSAForceKernel::computeGroup(
        int g, vector<Vec3>& posData, vector<Vec3>& forceData,
        bool includeForces, bool includeEnergy) {

        double scale = globalScalingFactor * groupScalingFactors[g];
        if (scale == 0.0) return;

        const vector<int>& particles = groupParticleIndices[g];

        // Build active receptor atom set (for locality optimization)
        double localityCutoff2 = 0.0;
        bool useLocality = (receptorLocalityCutoff > 0.0 && receptorMode == IsolatedGBSAForce::PAIRWISE);
        vector<bool> isActiveRecAtom(numReceptorAtoms, !useLocality);
        if (useLocality) {
            localityCutoff2 = receptorLocalityCutoff * receptorLocalityCutoff;
            for (int j = 0; j < numReceptorAtoms; j++) {
                for (int i = 0; i < numAtoms; i++) {
                    int pi = particles[i];
                    double dx = receptorPositions[j * 3] - posData[pi][0];
                    double dy = receptorPositions[j * 3 + 1] - posData[pi][1];
                    double dz = receptorPositions[j * 3 + 2] - posData[pi][2];
                    double r2 = dx * dx + dy * dy + dz * dz;
                    if (r2 < localityCutoff2) {
                        isActiveRecAtom[j] = true;
                        break;
                    }
                }
            }
        }

        // ---- Step 1: Receptor HCT for each ligand atom ----
        vector<double> hctReceptor(numAtoms, 0.0);

        if (receptorMode == IsolatedGBSAForce::GRID) {
            // Grid-based receptor HCT
            double ox, oy, oz;
            desolvationGrid->getOrigin(ox, oy, oz);
            double spacing = desolvationGrid->getSpacing();
            double probeRadius = desolvationGrid->getProbeRadius();
            int nx, ny, nz;
            desolvationGrid->getCounts(nx, ny, nz);
            int numBins = desolvationGrid->getNumBins();

            const auto& hctProbeData = desolvationGrid->getHctProbe();
            const auto& corrN = desolvationGrid->getCorrectionN();
            const auto& corrA = desolvationGrid->getCorrectionA();
            const auto& corrB = desolvationGrid->getCorrectionB();
            const auto& rThresholds = desolvationGrid->getRThresholds();

            int nyz = ny * nz;
            int numPoints = nx * ny * nz;

            for (int i = 0; i < numAtoms; i++) {
                int pi = particles[i];
                double x = posData[pi][0];
                double y = posData[pi][1];
                double z = posData[pi][2];

                // Fractional grid coordinates
                double fx = (x - ox) / spacing;
                double fy = (y - oy) / spacing;
                double fz = (z - oz) / spacing;

                int ix = (int)floor(fx);
                int iy = (int)floor(fy);
                int iz = (int)floor(fz);

                // Bounds check
                if (ix < 0 || ix >= nx - 1 || iy < 0 || iy >= ny - 1 ||
                    iz < 0 || iz >= nz - 1) {
                    hctReceptor[i] = 0.0;
                    continue;
                }

                double wx = fx - ix;
                double wy = fy - iy;
                double wz = fz - iz;

                // Trilinear interpolation of HCT_probe
                double hctProbe = 0.0;
                for (int di = 0; di < 2; di++) {
                    double wi = (di == 0) ? (1.0 - wx) : wx;
                    for (int dj = 0; dj < 2; dj++) {
                        double wj = (dj == 0) ? (1.0 - wy) : wy;
                        for (int dk = 0; dk < 2; dk++) {
                            double wk = (dk == 0) ? (1.0 - wz) : wz;
                            int idx = (ix + di) * nyz + (iy + dj) * nz + (iz + dk);
                            hctProbe += wi * wj * wk * hctProbeData[idx];
                        }
                    }
                }

                // Determine radius bin for this atom
                double R_i_off = radii[i] - DIELECTRIC_OFFSET;
                int bin = numBins - 1;
                for (int b = 0; b < numBins; b++) {
                    if (R_i_off <= rThresholds[b]) {
                        bin = b;
                        break;
                    }
                }

                // Trilinear interpolation of correction terms
                double interpN = 0.0, interpA = 0.0, interpB = 0.0;
                for (int di = 0; di < 2; di++) {
                    double wi = (di == 0) ? (1.0 - wx) : wx;
                    for (int dj = 0; dj < 2; dj++) {
                        double wj = (dj == 0) ? (1.0 - wy) : wy;
                        for (int dk = 0; dk < 2; dk++) {
                            double wk = (dk == 0) ? (1.0 - wz) : wz;
                            int idx = (ix + di) * nyz + (iy + dj) * nz + (iz + dk);
                            int offset = bin * numPoints + idx;
                            interpN += wi * wj * wk * corrN[offset];
                            interpA += wi * wj * wk * corrA[offset];
                            interpB += wi * wj * wk * corrB[offset];
                        }
                    }
                }

                // Apply radius correction
                double R_probe_off = probeRadius - DIELECTRIC_OFFSET;
                double invRi = (R_i_off > 0.0) ? 1.0 / R_i_off : 0.0;
                double invRp = (R_probe_off > 0.0) ? 1.0 / R_probe_off : 0.0;
                double correction = (invRi - invRp)
                                    * (interpN - 0.25 * interpA * (invRi + invRp))
                                    + interpB * log(R_i_off / R_probe_off);

                hctReceptor[i] = hctProbe + correction;
            }

        } else if (receptorMode == IsolatedGBSAForce::PAIRWISE) {
            // Pairwise receptor HCT (skip inactive receptor atoms if locality enabled)
            for (int i = 0; i < numAtoms; i++) {
                int pi = particles[i];
                double R_i_off = radii[i] - DIELECTRIC_OFFSET;
                for (int j = 0; j < numReceptorAtoms; j++) {
                    if (useLocality && !isActiveRecAtom[j]) continue;
                    double dx = posData[pi][0] - receptorPositions[j * 3];
                    double dy = posData[pi][1] - receptorPositions[j * 3 + 1];
                    double dz = posData[pi][2] - receptorPositions[j * 3 + 2];
                    double r = sqrt(dx * dx + dy * dy + dz * dz);

                    if (cutoffDistance > 0.0 && r > cutoffDistance) continue;

                    double R_j_off = receptorRadii[j] - DIELECTRIC_OFFSET;
                    hctReceptor[i] += computeHCTTerm(r, R_i_off, R_j_off, receptorScaleFactors[j]);
                }
            }
        }

        // ---- Step 2: Ligand-ligand HCT (all pairs, no exclusions) ----
        vector<double> hctLigand(numAtoms, 0.0);
        for (int i = 0; i < numAtoms; i++) {
            int pi = particles[i];
            double R_i_off = radii[i] - DIELECTRIC_OFFSET;
            for (int j = 0; j < numAtoms; j++) {
                if (i == j) continue;
                int pj = particles[j];
                double dx = posData[pi][0] - posData[pj][0];
                double dy = posData[pi][1] - posData[pj][1];
                double dz = posData[pi][2] - posData[pj][2];
                double r = sqrt(dx * dx + dy * dy + dz * dz);

                if (cutoffDistance > 0.0 && r > cutoffDistance) continue;

                double R_j_off = radii[j] - DIELECTRIC_OFFSET;
                hctLigand[i] += computeHCTTerm(r, R_i_off, R_j_off, scaleFactors[j]);
            }
        }

        // ---- Step 3: Born radii ----
        // Full Born radii (receptor + ligand HCT)
        vector<double> hctTotal(numAtoms);
        for (int i = 0; i < numAtoms; i++)
            hctTotal[i] = hctReceptor[i] + hctLigand[i];

        vector<double> bornRadiiFull;
        computeBornRadii(hctTotal, bornRadiiFull);

        // Ligand-only Born radii (for energy decomposition)
        vector<double> bornRadiiLigOnly;
        computeBornRadii(hctLigand, bornRadiiLigOnly);

        // Store Born radii for accessor
        groupBornRadii_[g] = bornRadiiFull;

        // ---- Step 4: GB energy ----
        vector<double> dE_dR_full(numAtoms, 0.0);
        double gbEnergyFull = computeGBEnergy(g, posData, bornRadiiFull, dE_dR_full);

        vector<double> dE_dR_ligOnly(numAtoms, 0.0);
        double gbEnergyLigOnly = computeGBEnergy(g, posData, bornRadiiLigOnly, dE_dR_ligOnly);

        groupEnergies_[g] = gbEnergyFull * scale;
        groupLigandSelfEnergies_[g] = gbEnergyLigOnly * scale;
        groupReceptorContributions_[g] = (gbEnergyFull - gbEnergyLigOnly) * scale;

        // ---- Step 5: Surface area ----
        if (includeSurfaceArea) {
            vector<double> dE_dR_sa(numAtoms, 0.0);
            double saEnergy = computeSurfaceAreaEnergy(bornRadiiFull, dE_dR_sa);
            groupEnergies_[g] += saEnergy * scale;

            // Add SA derivatives to full dE/dR
            for (int i = 0; i < numAtoms; i++)
                dE_dR_full[i] += dE_dR_sa[i];
        }

        // ---- Step 6: Forces (chain rule through Born radii) ----
        if (includeForces) {
            // Scale dE/dR by alchemical factor
            for (int i = 0; i < numAtoms; i++)
                dE_dR_full[i] *= scale;

            // Compute dR_born/dHCT for each atom
            vector<double> dR_dHCT(numAtoms, 0.0);
            for (int i = 0; i < numAtoms; i++) {
                double R_off = radii[i] - DIELECTRIC_OFFSET;
                if (R_off <= 0.0) continue;

                if (gbMethod == IsolatedGBSAForce::HCT) {
                    // R_born = 1/(1/R_off - 0.5*R_off*hct)
                    // dR/dhct = 0.5 * R_off * R_born^2
                    dR_dHCT[i] = 0.5 * R_off * bornRadiiFull[i] * bornRadiiFull[i];
                } else {
                    // OBC-II: psi = 0.5 * R_off * hctTotal
                    double psi = 0.5 * R_off * hctTotal[i];
                    double psi2 = psi * psi;
                    double psi3 = psi2 * psi;
                    double tanh_arg = OBC_ALPHA * psi - OBC_BETA * psi2 + OBC_GAMMA * psi3;
                    double tanh_val = tanh(tanh_arg);
                    double sech2 = 1.0 - tanh_val * tanh_val;
                    double dpsi_dhct = 0.5 * R_off;
                    double dtanh_dpsi = sech2 * (OBC_ALPHA - 2.0 * OBC_BETA * psi
                                                  + 3.0 * OBC_GAMMA * psi2);
                    dR_dHCT[i] = bornRadiiFull[i] * bornRadiiFull[i]
                                 * dtanh_dpsi * dpsi_dhct / radii[i];
                }
            }

            // Combined chain rule factor: dE/dpos = dE/dR * dR/dHCT * dHCT/dpos
            vector<double> chainFactor(numAtoms);
            for (int i = 0; i < numAtoms; i++)
                chainFactor[i] = dE_dR_full[i] * dR_dHCT[i];

            // 6a: Direct GB forces (from r-dependence of f_gb in pair energy)
            for (int i = 0; i < numAtoms; i++) {
                int pi = particles[i];
                for (int j = i + 1; j < numAtoms; j++) {
                    int pj = particles[j];

                    double dx = posData[pi][0] - posData[pj][0];
                    double dy = posData[pi][1] - posData[pj][1];
                    double dz = posData[pi][2] - posData[pj][2];
                    double r2 = dx * dx + dy * dy + dz * dz;
                    double r = sqrt(r2);
                    if (r < 1e-10) continue;

                    double D = bornRadiiFull[i] * bornRadiiFull[j];
                    double alpha_val = r2 / (4.0 * D);
                    double exp_alpha = exp(-alpha_val);
                    double f_gb2 = r2 + D * exp_alpha;
                    double f_gb = sqrt(f_gb2);

                    double qq = charges[i] * charges[j];

                    // dE/dr = -prefactor * qq / f_gb^2 * df_gb/dr
                    // df_gb/dr = r * (2 - 0.5*exp(-alpha)) / (2*f_gb)
                    //          = r * (4 - exp(-alpha)) / (4*f_gb)
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

            // 6b: Chain rule forces through ligand-ligand HCT
            for (int i = 0; i < numAtoms; i++) {
                int pi = particles[i];
                double R_i_off = radii[i] - DIELECTRIC_OFFSET;

                for (int j = 0; j < numAtoms; j++) {
                    if (i == j) continue;
                    int pj = particles[j];

                    double dx = posData[pi][0] - posData[pj][0];
                    double dy = posData[pi][1] - posData[pj][1];
                    double dz = posData[pi][2] - posData[pj][2];
                    double r2 = dx * dx + dy * dy + dz * dz;
                    double r = sqrt(r2);
                    if (r < 1e-10) continue;

                    if (cutoffDistance > 0.0 && r > cutoffDistance) continue;

                    double R_j_off = radii[j] - DIELECTRIC_OFFSET;
                    double dHCT_dr = computeHCTTermDerivative(r, R_i_off, R_j_off, scaleFactors[j]);

                    // Force on atom i from chain rule:
                    // F_i += -chainFactor[i] * dHCT_i/dr_ij * (r_vec/r)
                    double forceMag = -chainFactor[i] * dHCT_dr;
                    double invR = 1.0 / r;
                    forceData[pi][0] += forceMag * dx * invR;
                    forceData[pi][1] += forceMag * dy * invR;
                    forceData[pi][2] += forceMag * dz * invR;
                    // Newton's 3rd law: reaction force on screening atom j
                    forceData[pj][0] -= forceMag * dx * invR;
                    forceData[pj][1] -= forceMag * dy * invR;
                    forceData[pj][2] -= forceMag * dz * invR;
                }
            }

            // 6c: Chain rule forces through receptor HCT (PAIRWISE mode)
            if (receptorMode == IsolatedGBSAForce::PAIRWISE) {
                for (int i = 0; i < numAtoms; i++) {
                    int pi = particles[i];
                    double R_i_off = radii[i] - DIELECTRIC_OFFSET;

                    for (int j = 0; j < numReceptorAtoms; j++) {
                        double dx = posData[pi][0] - receptorPositions[j * 3];
                        double dy = posData[pi][1] - receptorPositions[j * 3 + 1];
                        double dz = posData[pi][2] - receptorPositions[j * 3 + 2];
                        double r2 = dx * dx + dy * dy + dz * dz;
                        double r = sqrt(r2);
                        if (r < 1e-10) continue;

                        if (cutoffDistance > 0.0 && r > cutoffDistance) continue;

                        double R_j_off = receptorRadii[j] - DIELECTRIC_OFFSET;
                        double dHCT_dr = computeHCTTermDerivative(r, R_i_off, R_j_off,
                                                                   receptorScaleFactors[j]);

                        double forceMag = -chainFactor[i] * dHCT_dr;
                        double invR = 1.0 / r;
                        // Force only on ligand atom (receptor is fixed)
                        forceData[pi][0] += forceMag * dx * invR;
                        forceData[pi][1] += forceMag * dy * invR;
                        forceData[pi][2] += forceMag * dz * invR;
                    }
                }
            }

            // 6d: Chain rule forces through GRID receptor HCT
            if (receptorMode == IsolatedGBSAForce::GRID) {
                double ox, oy, oz;
                desolvationGrid->getOrigin(ox, oy, oz);
                double spacing = desolvationGrid->getSpacing();
                double probeRadius = desolvationGrid->getProbeRadius();
                int nx, ny, nz;
                desolvationGrid->getCounts(nx, ny, nz);
                int numBins = desolvationGrid->getNumBins();

                const auto& hctProbeData = desolvationGrid->getHctProbe();
                const auto& corrN = desolvationGrid->getCorrectionN();
                const auto& corrA = desolvationGrid->getCorrectionA();
                const auto& corrB = desolvationGrid->getCorrectionB();
                const auto& rThresh = desolvationGrid->getRThresholds();

                int nyz = ny * nz;
                int numPoints = nx * ny * nz;
                double invSpacing = 1.0 / spacing;

                for (int i = 0; i < numAtoms; i++) {
                    int pi = particles[i];
                    double x = posData[pi][0];
                    double y = posData[pi][1];
                    double z = posData[pi][2];

                    double fx_grid = (x - ox) * invSpacing;
                    double fy_grid = (y - oy) * invSpacing;
                    double fz_grid = (z - oz) * invSpacing;

                    int ix = (int)floor(fx_grid);
                    int iy = (int)floor(fy_grid);
                    int iz = (int)floor(fz_grid);

                    if (ix < 0 || ix >= nx - 1 || iy < 0 || iy >= ny - 1 ||
                        iz < 0 || iz >= nz - 1)
                        continue;

                    double wx = fx_grid - ix;
                    double wy = fy_grid - iy;
                    double wz = fz_grid - iz;

                    // Determine radius bin
                    double R_i_off = radii[i] - DIELECTRIC_OFFSET;
                    int bin = numBins - 1;
                    for (int b = 0; b < numBins; b++) {
                        if (R_i_off <= rThresh[b]) { bin = b; break; }
                    }

                    // Compute gradient of interpolated HCT by differentiating trilinear
                    // dHCT/dx = sum over grid neighbors of (dw/dx) * value
                    // For trilinear, dw_i/dx = +/-1/spacing * wy * wz, etc.
                    double gradX = 0.0, gradY = 0.0, gradZ = 0.0;

                    for (int di = 0; di < 2; di++) {
                        double dwi_dx = (di == 0) ? -invSpacing : invSpacing;
                        double wi = (di == 0) ? (1.0 - wx) : wx;
                        for (int dj = 0; dj < 2; dj++) {
                            double dwj_dy = (dj == 0) ? -invSpacing : invSpacing;
                            double wj = (dj == 0) ? (1.0 - wy) : wy;
                            for (int dk = 0; dk < 2; dk++) {
                                double dwk_dz = (dk == 0) ? -invSpacing : invSpacing;
                                double wk = (dk == 0) ? (1.0 - wz) : wz;

                                int idx = (ix + di) * nyz + (iy + dj) * nz + (iz + dk);
                                int offset = bin * numPoints + idx;

                                // Total HCT at this grid point = hctProbe + correction
                                double R_probe_off = probeRadius - DIELECTRIC_OFFSET;
                                double invRi = (R_i_off > 0.0) ? 1.0 / R_i_off : 0.0;
                                double invRp = (R_probe_off > 0.0) ? 1.0 / R_probe_off : 0.0;

                                double valHct = hctProbeData[idx];
                                double valN = corrN[offset];
                                double valA = corrA[offset];
                                double valB = corrB[offset];

                                double val = valHct + (invRi - invRp)
                                             * (valN - 0.25 * valA * (invRi + invRp))
                                             + valB * log(R_i_off / R_probe_off);

                                gradX += dwi_dx * wj * wk * val;
                                gradY += wi * dwj_dy * wk * val;
                                gradZ += wi * wj * dwk_dz * val;
                            }
                        }
                    }

                    // Apply chain rule force
                    forceData[pi][0] -= chainFactor[i] * gradX;
                    forceData[pi][1] -= chainFactor[i] * gradY;
                    forceData[pi][2] -= chainFactor[i] * gradZ;
                }
            }
        }

        // ---- Step 7: PAIRWISE receptor desolvation + cross-term ----
        if (receptorMode == IsolatedGBSAForce::PAIRWISE) {
            // Active atom mask was already computed at the top of this group loop

            // 7a: Compute ligand→receptor HCT screening (only active atoms)
            vector<double> ligandToRecHCT(numReceptorAtoms, 0.0);
            for (int j = 0; j < numReceptorAtoms; j++) {
                if (!isActiveRecAtom[j]) continue;
                double R_j_off = receptorRadii[j] - DIELECTRIC_OFFSET;
                for (int i = 0; i < numAtoms; i++) {
                    int pi = particles[i];
                    double dx = receptorPositions[j * 3] - posData[pi][0];
                    double dy = receptorPositions[j * 3 + 1] - posData[pi][1];
                    double dz = receptorPositions[j * 3 + 2] - posData[pi][2];
                    double r = sqrt(dx * dx + dy * dy + dz * dz);

                    if (cutoffDistance > 0.0 && r > cutoffDistance) continue;

                    double R_i_off = radii[i] - DIELECTRIC_OFFSET;
                    ligandToRecHCT[j] += computeHCTTerm(r, R_j_off, R_i_off, scaleFactors[i]);
                }
            }

            // 7b: Receptor Born radii with ligand screening
            // Active atoms: recompute with ligand HCT. Inactive: keep reference.
            vector<double> recBornRadiiWithLig(numReceptorAtoms);
            for (int j = 0; j < numReceptorAtoms; j++) {
                if (!isActiveRecAtom[j]) {
                    recBornRadiiWithLig[j] = receptorBornRadiiRef[j];
                    continue;
                }
                double totalHCT = receptorSelfHCT[j] + ligandToRecHCT[j];
                double R_off = receptorRadii[j] - DIELECTRIC_OFFSET;
                if (gbMethod == IsolatedGBSAForce::HCT) {
                    double inner = 1.0 / R_off - 0.5 * R_off * totalHCT;
                    recBornRadiiWithLig[j] = (inner > 0.0) ? 1.0 / inner : 500.0;
                } else {
                    double psi = 0.5 * R_off * totalHCT;
                    double tanh_val = tanh(OBC_ALPHA * psi - OBC_BETA * psi * psi
                                           + OBC_GAMMA * psi * psi * psi);
                    double inner = 1.0 / R_off - tanh_val / receptorRadii[j];
                    recBornRadiiWithLig[j] = (inner > 0.0) ? 1.0 / inner : 500.0;
                }
            }

            groupReceptorBornRadii_[g] = recBornRadiiWithLig;

            // 7c: Receptor desolvation energy
            double desolvation = 0.0;
            if (!useLocality) {
                // Full O(N_rec^2) computation
                double recEnergyWithLig = 0.0;
                for (int i = 0; i < numReceptorAtoms; i++) {
                    recEnergyWithLig += 0.5 * prefactor * receptorCharges[i] * receptorCharges[i]
                                        / recBornRadiiWithLig[i];
                    for (int j = i + 1; j < numReceptorAtoms; j++) {
                        double dx = receptorPositions[i * 3] - receptorPositions[j * 3];
                        double dy = receptorPositions[i * 3 + 1] - receptorPositions[j * 3 + 1];
                        double dz = receptorPositions[i * 3 + 2] - receptorPositions[j * 3 + 2];
                        double r2 = dx * dx + dy * dy + dz * dz;
                        double D = recBornRadiiWithLig[i] * recBornRadiiWithLig[j];
                        double exp_alpha = exp(-r2 / (4.0 * D));
                        double f_gb = sqrt(r2 + D * exp_alpha);
                        recEnergyWithLig += prefactor * receptorCharges[i]
                                            * receptorCharges[j] / f_gb;
                    }
                }
                desolvation = recEnergyWithLig - receptorReferenceEnergy;
            } else {
                // Delta approach: O(|A| * N_rec) where A = active atoms
                // Only active atoms have changed Born radii, so we compute
                // the energy difference from pairs involving at least one active atom.
                for (int a = 0; a < numReceptorAtoms; a++) {
                    if (!isActiveRecAtom[a]) continue;

                    // Self-term delta
                    desolvation += 0.5 * prefactor * receptorCharges[a] * receptorCharges[a]
                                   * (1.0 / recBornRadiiWithLig[a] - 1.0 / receptorBornRadiiRef[a]);

                    // Pair-term deltas: iterate all j != a
                    for (int j = 0; j < numReceptorAtoms; j++) {
                        if (j == a) continue;
                        // Avoid double-counting when both a and j are active
                        if (isActiveRecAtom[j] && j < a) continue;

                        double dx = receptorPositions[a * 3] - receptorPositions[j * 3];
                        double dy = receptorPositions[a * 3 + 1] - receptorPositions[j * 3 + 1];
                        double dz = receptorPositions[a * 3 + 2] - receptorPositions[j * 3 + 2];
                        double r2 = dx * dx + dy * dy + dz * dz;

                        // Energy with new (mixed) Born radii
                        double D_new = recBornRadiiWithLig[a] * recBornRadiiWithLig[j];
                        double exp_new = exp(-r2 / (4.0 * D_new));
                        double f_new = sqrt(r2 + D_new * exp_new);
                        double E_new = prefactor * receptorCharges[a] * receptorCharges[j] / f_new;

                        // Energy with reference Born radii
                        double D_ref = receptorBornRadiiRef[a] * receptorBornRadiiRef[j];
                        double exp_ref = exp(-r2 / (4.0 * D_ref));
                        double f_ref = sqrt(r2 + D_ref * exp_ref);
                        double E_ref = prefactor * receptorCharges[a] * receptorCharges[j] / f_ref;

                        desolvation += E_new - E_ref;
                    }
                }
            }
            desolvation *= scale;
            groupReceptorDesolvations_[g] = desolvation;
            groupEnergies_[g] += desolvation;

            // 7d: Cross-term energy (receptor-ligand GB pairs, ALL receptor atoms)
            // Distance floor prevents singularity when ligand overlaps receptor
            static constexpr double MIN_CROSS_R2 = 0.01;  // 0.1 nm = 1 Angstrom
            double crossTermEnergy = 0.0;
            // Cross-term dependence on the two Born radii (both vary with ligand
            // position through the HCT sums). Accumulated here and applied via
            // the HCT chain rule below; the explicit r-dependence is inline.
            vector<double> dCross_dRi(numAtoms, 0.0);
            vector<double> dCross_dRrecj(numReceptorAtoms, 0.0);
            for (int i = 0; i < numAtoms; i++) {
                int pi = particles[i];
                for (int j = 0; j < numReceptorAtoms; j++) {
                    double dx = posData[pi][0] - receptorPositions[j * 3];
                    double dy = posData[pi][1] - receptorPositions[j * 3 + 1];
                    double dz = posData[pi][2] - receptorPositions[j * 3 + 2];
                    double r2 = dx * dx + dy * dy + dz * dz;
                    if (r2 < MIN_CROSS_R2) continue;

                    double D = bornRadiiFull[i] * recBornRadiiWithLig[j];
                    double exp_alpha = exp(-r2 / (4.0 * D));
                    double f_gb = sqrt(r2 + D * exp_alpha);

                    double E_cross = prefactor * charges[i] * receptorCharges[j] / f_gb;
                    crossTermEnergy += E_cross;

                    // Cross-term forces on ligand atoms
                    if (includeForces) {
                        double r = sqrt(r2);
                        if (r < 1e-10) continue;
                        double f_gb2 = r2 + D * exp_alpha;
                        double alpha_val = r2 / (4.0 * D);
                        double dE_dr = -prefactor * charges[i] * receptorCharges[j]
                                       * r * (4.0 - exp(-alpha_val))
                                       / (4.0 * f_gb * f_gb2) * scale;

                        double invR = 1.0 / r;
                        forceData[pi][0] -= dE_dr * dx * invR;
                        forceData[pi][1] -= dE_dr * dy * invR;
                        forceData[pi][2] -= dE_dr * dz * invR;

                        // Born-radius derivatives of the cross term (unscaled;
                        // scale is applied where these feed the chain rule).
                        double dEcross = -prefactor * charges[i] * receptorCharges[j] / f_gb2;
                        double dfdR_common = exp_alpha * (1.0 + alpha_val) / (2.0 * f_gb);
                        dCross_dRi[i]    += dEcross * recBornRadiiWithLig[j] * dfdR_common;
                        dCross_dRrecj[j] += dEcross * bornRadiiFull[i]       * dfdR_common;
                    }
                }
            }

            crossTermEnergy *= scale;
            groupCrossTermEnergies_[g] = crossTermEnergy;
            groupEnergies_[g] += crossTermEnergy;

            // 7e: Receptor desolvation forces on ligand atoms
            if (includeForces) {
                // Compute receptor dE/dR_born for active atoms only.
                // Only active atoms have nonzero ligandToRecHCT, so only they
                // contribute to desolvation forces through the chain rule.
                // We need dE/dR for active atoms, which requires O(|A|*N_rec) work.
                vector<double> recDeDR(numReceptorAtoms, 0.0);
                for (int a = 0; a < numReceptorAtoms; a++) {
                    if (!isActiveRecAtom[a]) continue;
                    // Self term
                    recDeDR[a] += -0.5 * prefactor * receptorCharges[a] * receptorCharges[a]
                                  / (recBornRadiiWithLig[a] * recBornRadiiWithLig[a]);
                    // Pair terms with all other receptor atoms
                    for (int j = 0; j < numReceptorAtoms; j++) {
                        if (j == a) continue;
                        double dx = receptorPositions[a * 3] - receptorPositions[j * 3];
                        double dy = receptorPositions[a * 3 + 1] - receptorPositions[j * 3 + 1];
                        double dz = receptorPositions[a * 3 + 2] - receptorPositions[j * 3 + 2];
                        double r2 = dx * dx + dy * dy + dz * dz;
                        double D = recBornRadiiWithLig[a] * recBornRadiiWithLig[j];
                        double alpha_val = r2 / (4.0 * D);
                        double exp_alpha = exp(-alpha_val);
                        double f_gb2 = r2 + D * exp_alpha;
                        double f_gb = sqrt(f_gb2);
                        double qq = receptorCharges[a] * receptorCharges[j];

                        double df_dRa = recBornRadiiWithLig[j] * exp_alpha
                                        * (1.0 + alpha_val) / (2.0 * f_gb);
                        recDeDR[a] += -prefactor * qq / f_gb2 * df_dRa;
                    }
                }

                // Fold the cross-term's receptor-Born dependence into the
                // receptor energy derivative so it propagates through the same
                // ligand->receptor HCT chain rule below.
                for (int a = 0; a < numReceptorAtoms; a++)
                    if (isActiveRecAtom[a])
                        recDeDR[a] += dCross_dRrecj[a];

                // Compute receptor dR_born/dHCT (only for active atoms)
                vector<double> recDRdHCT(numReceptorAtoms, 0.0);
                for (int j = 0; j < numReceptorAtoms; j++) {
                    if (!isActiveRecAtom[j]) continue;
                    double R_off = receptorRadii[j] - DIELECTRIC_OFFSET;
                    if (R_off <= 0.0) continue;
                    double totalHCT = receptorSelfHCT[j] + ligandToRecHCT[j];
                    if (gbMethod == IsolatedGBSAForce::HCT) {
                        recDRdHCT[j] = 0.5 * R_off * recBornRadiiWithLig[j] * recBornRadiiWithLig[j];
                    } else {
                        double psi = 0.5 * R_off * totalHCT;
                        double tanh_arg = OBC_ALPHA * psi - OBC_BETA * psi * psi
                                          + OBC_GAMMA * psi * psi * psi;
                        double tanh_val = tanh(tanh_arg);
                        double sech2 = 1.0 - tanh_val * tanh_val;
                        double dtanh_dpsi = sech2 * (OBC_ALPHA - 2.0 * OBC_BETA * psi
                                                      + 3.0 * OBC_GAMMA * psi * psi);
                        recDRdHCT[j] = recBornRadiiWithLig[j] * recBornRadiiWithLig[j]
                                       * dtanh_dpsi * 0.5 * R_off / receptorRadii[j];
                    }
                }

                // Chain rule: force on ligand atom from receptor desolvation
                // Only iterate active receptor atoms (inactive have zero ligandToRecHCT)
                for (int i = 0; i < numAtoms; i++) {
                    int pi = particles[i];
                    double R_i_off = radii[i] - DIELECTRIC_OFFSET;

                    for (int j = 0; j < numReceptorAtoms; j++) {
                        if (!isActiveRecAtom[j]) continue;
                        double dx = receptorPositions[j * 3] - posData[pi][0];
                        double dy = receptorPositions[j * 3 + 1] - posData[pi][1];
                        double dz = receptorPositions[j * 3 + 2] - posData[pi][2];
                        double r2 = dx * dx + dy * dy + dz * dz;
                        double r = sqrt(r2);
                        if (r < 1e-10) continue;

                        if (cutoffDistance > 0.0 && r > cutoffDistance) continue;

                        double R_j_off = receptorRadii[j] - DIELECTRIC_OFFSET;
                        double dHCT_dr = computeHCTTermDerivative(r, R_j_off, R_i_off,
                                                                   scaleFactors[i]);

                        double recChainFactor = recDeDR[j] * recDRdHCT[j];
                        double forceMag = -scale * recChainFactor * dHCT_dr;
                        double invR = 1.0 / r;
                        forceData[pi][0] -= forceMag * dx * invR;
                        forceData[pi][1] -= forceMag * dy * invR;
                        forceData[pi][2] -= forceMag * dz * invR;
                    }
                }

                // Cross-term dependence on the ligand Born radii: propagate
                // dE_cross/dR_lig through the ligand and receptor HCT sums, the
                // same chain rule Step 6 uses for the ligand-ligand GB energy.
                vector<double> crossDRdHCT(numAtoms, 0.0);
                for (int i = 0; i < numAtoms; i++) {
                    double R_off = radii[i] - DIELECTRIC_OFFSET;
                    if (R_off <= 0.0) continue;
                    if (gbMethod == IsolatedGBSAForce::HCT) {
                        crossDRdHCT[i] = 0.5 * R_off * bornRadiiFull[i] * bornRadiiFull[i];
                    } else {
                        double psi = 0.5 * R_off * hctTotal[i];
                        double tanh_arg = OBC_ALPHA * psi - OBC_BETA * psi * psi
                                          + OBC_GAMMA * psi * psi * psi;
                        double tanh_val = tanh(tanh_arg);
                        double sech2 = 1.0 - tanh_val * tanh_val;
                        double dtanh_dpsi = sech2 * (OBC_ALPHA - 2.0 * OBC_BETA * psi
                                                      + 3.0 * OBC_GAMMA * psi * psi);
                        crossDRdHCT[i] = bornRadiiFull[i] * bornRadiiFull[i]
                                         * dtanh_dpsi * 0.5 * R_off / radii[i];
                    }
                }

                vector<double> crossChainFactor(numAtoms);
                for (int i = 0; i < numAtoms; i++)
                    crossChainFactor[i] = scale * dCross_dRi[i] * crossDRdHCT[i];

                // Through ligand-ligand HCT
                for (int i = 0; i < numAtoms; i++) {
                    int pi = particles[i];
                    double R_i_off = radii[i] - DIELECTRIC_OFFSET;
                    for (int j = 0; j < numAtoms; j++) {
                        if (i == j) continue;
                        int pj = particles[j];
                        double dx = posData[pi][0] - posData[pj][0];
                        double dy = posData[pi][1] - posData[pj][1];
                        double dz = posData[pi][2] - posData[pj][2];
                        double r = sqrt(dx * dx + dy * dy + dz * dz);
                        if (r < 1e-10) continue;
                        if (cutoffDistance > 0.0 && r > cutoffDistance) continue;
                        double R_j_off = radii[j] - DIELECTRIC_OFFSET;
                        double dHCT_dr = computeHCTTermDerivative(r, R_i_off, R_j_off,
                                                                   scaleFactors[j]);
                        double forceMag = -crossChainFactor[i] * dHCT_dr;
                        double invR = 1.0 / r;
                        forceData[pi][0] += forceMag * dx * invR;
                        forceData[pi][1] += forceMag * dy * invR;
                        forceData[pi][2] += forceMag * dz * invR;
                        forceData[pj][0] -= forceMag * dx * invR;
                        forceData[pj][1] -= forceMag * dy * invR;
                        forceData[pj][2] -= forceMag * dz * invR;
                    }
                }

                // Through receptor->ligand HCT (receptor fixed; force on ligand)
                for (int i = 0; i < numAtoms; i++) {
                    int pi = particles[i];
                    double R_i_off = radii[i] - DIELECTRIC_OFFSET;
                    for (int j = 0; j < numReceptorAtoms; j++) {
                        double dx = posData[pi][0] - receptorPositions[j * 3];
                        double dy = posData[pi][1] - receptorPositions[j * 3 + 1];
                        double dz = posData[pi][2] - receptorPositions[j * 3 + 2];
                        double r = sqrt(dx * dx + dy * dy + dz * dz);
                        if (r < 1e-10) continue;
                        if (cutoffDistance > 0.0 && r > cutoffDistance) continue;
                        double R_j_off = receptorRadii[j] - DIELECTRIC_OFFSET;
                        double dHCT_dr = computeHCTTermDerivative(r, R_i_off, R_j_off,
                                                                   receptorScaleFactors[j]);
                        double forceMag = -crossChainFactor[i] * dHCT_dr;
                        double invR = 1.0 / r;
                        forceData[pi][0] += forceMag * dx * invR;
                        forceData[pi][1] += forceMag * dy * invR;
                        forceData[pi][2] += forceMag * dz * invR;
                    }
                }
            }
        }
}

void ReferenceCalcIsolatedGBSAForceKernel::runGroups(
        ContextImpl& context, vector<Vec3>& posData, vector<Vec3>& forceData,
        bool includeForces, bool includeEnergy) {
    for (int g = 0; g < numParticleGroups; g++)
        computeGroup(g, posData, forceData, includeForces, includeEnergy);
}

double ReferenceCalcIsolatedGBSAForceKernel::execute(
        ContextImpl& context, bool includeForces, bool includeEnergy) {

    vector<Vec3>& posData = refExtractPositions(context);
    vector<Vec3>& forceData = refExtractForces(context);

    // Clear per-group results
    fill(groupEnergies_.begin(), groupEnergies_.end(), 0.0);
    fill(groupLigandSelfEnergies_.begin(), groupLigandSelfEnergies_.end(), 0.0);
    fill(groupReceptorContributions_.begin(), groupReceptorContributions_.end(), 0.0);
    fill(groupReceptorDesolvations_.begin(), groupReceptorDesolvations_.end(), 0.0);
    fill(groupCrossTermEnergies_.begin(), groupCrossTermEnergies_.end(), 0.0);

    runGroups(context, posData, forceData, includeForces, includeEnergy);

    // Deterministic, group-ordered reduction (matches serial Reference exactly).
    // groupEnergies_[g] holds the group's full contribution (GB + SA + PAIRWISE
    // desolvation + cross-term), which is exactly what was added to totalEnergy.
    double totalEnergy = 0.0;
    for (int g = 0; g < numParticleGroups; g++)
        totalEnergy += groupEnergies_[g];
    return totalEnergy;
}

// ==================== updateParametersInContext ====================

void ReferenceCalcIsolatedGBSAForceKernel::updateParametersInContext(
        ContextImpl& context, const IsolatedGBSAForce& force) {

    // Update atom parameters
    for (int i = 0; i < numAtoms; i++) {
        force.getAtomParameters(i, charges[i], radii[i], scaleFactors[i]);
    }

    // Update solvent parameters
    double soluteDielectric = force.getSoluteDielectric();
    double solventDielectric = force.getSolventDielectric();
    prefactor = -COULOMB_CONSTANT * (1.0 / soluteDielectric - 1.0 / solventDielectric);

    includeSurfaceArea = force.getIncludeSurfaceArea();
    surfaceTension = force.getSurfaceTension();
    cutoffDistance = force.getCutoffDistance();
    receptorLocalityCutoff = force.getReceptorLocalityCutoff();

    // Update alchemical scaling
    globalScalingFactor = force.getGlobalScalingFactor();
    int nGroups = force.getNumParticleGroups();
    for (int g = 0; g < nGroups; g++) {
        groupScalingFactors[g] = force.getGroupScalingFactor(g);
    }
}

// ==================== Hessian ====================

namespace {

// OBC-II / HCT Born-radius transform and its first/second derivatives w.r.t.
// the (unscaled) HCT sum Psi. Returns R_born, dR/dPsi, d2R/dPsi2.
// For OBC-II: psi_s = 0.5*R_off*Psi, tanh transform; matches execute()/CUDA.
struct BornDerivs { double R, dRdPsi, d2RdPsi2; };

inline BornDerivs bornTransformDerivs(double R_intrinsic, double R_off,
                                      double hctTotal, double R_born,
                                      bool isHCT) {
    BornDerivs d;
    d.R = R_born;
    if (R_off <= 0.0) { d.dRdPsi = 0.0; d.d2RdPsi2 = 0.0; return d; }
    if (isHCT) {
        // R = 1/(1/R_off - 0.5*R_off*Psi)  => dR/dPsi = 0.5*R_off*R^2,
        // d2R/dPsi2 = 2*R*(dR/dPsi)*0.5*R_off = R_off*R^2*dR/dPsi... compute directly.
        double a = 0.5 * R_off;
        d.dRdPsi = a * R_born * R_born;
        d.d2RdPsi2 = 2.0 * R_born * d.dRdPsi * a;   // = 2 a^2 R^3
    } else {
        double psi_s = 0.5 * R_off * hctTotal;
        double psi_s2 = psi_s * psi_s;
        double arg = OBC_ALPHA * psi_s - OBC_BETA * psi_s2 + OBC_GAMMA * psi_s2 * psi_s;
        double t = tanh(arg);
        double sech2 = 1.0 - t * t;
        double darg = OBC_ALPHA - 2.0 * OBC_BETA * psi_s + 3.0 * OBC_GAMMA * psi_s2;
        double d2arg = -2.0 * OBC_BETA + 6.0 * OBC_GAMMA * psi_s;
        double dpsi = 0.5 * R_off;          // dpsi_s/dPsi
        double Dc = dpsi / R_intrinsic;
        d.dRdPsi = R_born * R_born * sech2 * darg * Dc;
        double A = R_born * R_born, B = sech2, C = darg;
        double dA = 2.0 * R_born * d.dRdPsi;
        double dargP = darg * dpsi;
        double dB = -2.0 * sech2 * t * dargP;
        double dC = d2arg * dpsi;
        d.d2RdPsi2 = (dA * B * C + A * dB * C + A * B * dC) * Dc;
    }
    return d;
}

// Still-pair f_gb and the energy second-derivatives used by the coupling
// matrix and assembly: d2E/dRa2, d2E/dRaRb, d2E/dr2, d2E/dr dRa.
struct StillPair {
    double f, f2, et, RaRb;
};
inline StillPair stillPair(double r2, double Ra, double Rb) {
    StillPair s;
    s.RaRb = Ra * Rb;
    s.et = exp(-r2 / (4.0 * s.RaRb));
    s.f2 = r2 + s.RaRb * s.et;
    s.f = sqrt(s.f2);
    return s;
}

}  // namespace

void ReferenceCalcIsolatedGBSAForceKernel::parallelFor(ContextImpl& context, int count, const std::function<void(int)>& body) {
    for (int i = 0; i < count; i++) body(i);
}

vector<double> ReferenceCalcIsolatedGBSAForceKernel::computeHessian(ContextImpl& context) {
    vector<Vec3>& posData = refExtractPositions(context);

    int totalParticles = numParticleGroups * numAtoms;
    int dim3N = 3 * totalParticles;
    vector<double> H(static_cast<size_t>(dim3N) * dim3N, 0.0);
    if (totalParticles == 0)
        return H;

    bool isHCT = (gbMethod == IsolatedGBSAForce::HCT);
    bool pairwise = (receptorMode == IsolatedGBSAForce::PAIRWISE);
    if (receptorMode == IsolatedGBSAForce::GRID)
        throw OpenMMException("IsolatedGBSAForce: GRID-mode Hessian not implemented");

    // Each group is isolated -> block diagonal in the global (3*totalParticles)
    // matrix. We compute the per-group block in local indexing then scatter.
    parallelFor(context, numParticleGroups, [&](int g) {
        double scale = globalScalingFactor * groupScalingFactors[g];
        if (scale == 0.0) return;
        const vector<int>& particles = groupParticleIndices[g];
        int N = numAtoms;
        int n3 = 3 * N;

        // ---- Recompute the forward quantities the Hessian differentiates ----
        // Ligand-ligand HCT and receptor->ligand HCT (frozen receptor).
        vector<double> hctReceptor(N, 0.0), hctLigand(N, 0.0);
        for (int i = 0; i < N; i++) {
            int pi = particles[i];
            double Ri_off = radii[i] - DIELECTRIC_OFFSET;
            if (pairwise) {
                for (int j = 0; j < numReceptorAtoms; j++) {
                    double dx = posData[pi][0] - receptorPositions[j*3];
                    double dy = posData[pi][1] - receptorPositions[j*3+1];
                    double dz = posData[pi][2] - receptorPositions[j*3+2];
                    double r = sqrt(dx*dx+dy*dy+dz*dz);
                    if (cutoffDistance > 0.0 && r > cutoffDistance) continue;
                    double Rj_off = receptorRadii[j] - DIELECTRIC_OFFSET;
                    hctReceptor[i] += computeHCTTerm(r, Ri_off, Rj_off, receptorScaleFactors[j]);
                }
            }
            for (int j = 0; j < N; j++) {
                if (i == j) continue;
                int pj = particles[j];
                double dx = posData[pi][0] - posData[pj][0];
                double dy = posData[pi][1] - posData[pj][1];
                double dz = posData[pi][2] - posData[pj][2];
                double r = sqrt(dx*dx+dy*dy+dz*dz);
                if (cutoffDistance > 0.0 && r > cutoffDistance) continue;
                double Rj_off = radii[j] - DIELECTRIC_OFFSET;
                hctLigand[i] += computeHCTTerm(r, Ri_off, Rj_off, scaleFactors[j]);
            }
        }
        vector<double> hctTotal(N);
        for (int i = 0; i < N; i++) hctTotal[i] = hctReceptor[i] + hctLigand[i];
        vector<double> born;
        computeBornRadii(hctTotal, born);

        // dE/dR (ligand GB self + ligand-ligand pairs) and SA, in double.
        vector<double> dE_dR(N, 0.0);
        (void)computeGBEnergy(g, posData, born, dE_dR);
        if (includeSurfaceArea) {
            vector<double> dE_sa(N, 0.0);
            (void)computeSurfaceAreaEnergy(born, dE_sa);
            for (int i = 0; i < N; i++) dE_dR[i] += dE_sa[i];
        }

        // Per-atom Born transform derivatives.
        vector<double> dRdPsi(N), d2RdPsi2(N), dE_dHCT(N);
        for (int i = 0; i < N; i++) {
            double R_off = radii[i] - DIELECTRIC_OFFSET;
            BornDerivs bd = bornTransformDerivs(radii[i], R_off, hctTotal[i], born[i], isHCT);
            dRdPsi[i] = bd.dRdPsi;
            d2RdPsi2[i] = bd.d2RdPsi2;
            dE_dHCT[i] = dE_dR[i] * bd.dRdPsi;   // dE/dPsi for the ligand
        }

        // ---- Jacobian J[k][3*i+a] = dPsi_k/dx_{i,a} (ligand atoms only) ----
        // Psi_k = sum_{j!=k} HCT(r_kj) over ligand + sum_recv HCT(r_k,rec).
        // Receptor frozen: only the self (k) row's diagonal block gets the
        // receptor contribution; ligand-ligand gives self + off-diagonal.
        vector<double> J(static_cast<size_t>(N) * n3, 0.0);
        for (int k = 0; k < N; k++) {
            int pk = particles[k];
            double Rk_off = radii[k] - DIELECTRIC_OFFSET;
            double jsx = 0.0, jsy = 0.0, jsz = 0.0;
            if (pairwise) {
                for (int rj = 0; rj < numReceptorAtoms; rj++) {
                    double dx = posData[pk][0] - receptorPositions[rj*3];
                    double dy = posData[pk][1] - receptorPositions[rj*3+1];
                    double dz = posData[pk][2] - receptorPositions[rj*3+2];
                    double r2 = dx*dx+dy*dy+dz*dz;
                    double r = sqrt(r2);
                    if (r < 1e-10) continue;
                    if (cutoffDistance > 0.0 && r > cutoffDistance) continue;
                    double Rrj_off = receptorRadii[rj] - DIELECTRIC_OFFSET;
                    double I1 = computeHCTTermDerivative(r, Rk_off, Rrj_off, receptorScaleFactors[rj]);
                    double invr = 1.0 / r;
                    jsx += I1 * dx * invr; jsy += I1 * dy * invr; jsz += I1 * dz * invr;
                }
            }
            for (int j = 0; j < N; j++) {
                if (j == k) continue;
                int pj = particles[j];
                double dx = posData[pk][0] - posData[pj][0];
                double dy = posData[pk][1] - posData[pj][1];
                double dz = posData[pk][2] - posData[pj][2];
                double r2 = dx*dx+dy*dy+dz*dz;
                double r = sqrt(r2);
                if (r < 1e-10) continue;
                if (cutoffDistance > 0.0 && r > cutoffDistance) continue;
                double Rj_off = radii[j] - DIELECTRIC_OFFSET;
                double I1 = computeHCTTermDerivative(r, Rk_off, Rj_off, scaleFactors[j]);
                double invr = 1.0 / r;
                jsx += I1 * dx * invr; jsy += I1 * dy * invr; jsz += I1 * dz * invr;
                J[(size_t)k*n3 + 3*j + 0] = -I1 * dx * invr;
                J[(size_t)k*n3 + 3*j + 1] = -I1 * dy * invr;
                J[(size_t)k*n3 + 3*j + 2] = -I1 * dz * invr;
            }
            J[(size_t)k*n3 + 3*k + 0] = jsx;
            J[(size_t)k*n3 + 3*k + 1] = jsy;
            J[(size_t)k*n3 + 3*k + 2] = jsz;
        }

        // ---- Coupling matrix M[k][l] = d2E/dPsi_k dPsi_l (ligand) ----
        // Diagonal: self GB curvature + SA curvature + OBC curvature + sum of
        // pair d2E/dRk2; off-diagonal: pair d2E/dRk dRl, all times dR/dPsi.
        vector<double> M(static_cast<size_t>(N) * N, 0.0);
        double probe = 0.14;
        if (receptorMode == IsolatedGBSAForce::GRID && desolvationGrid)
            probe = desolvationGrid->getProbeRadius();
        for (int k = 0; k < N; k++) {
            int pk = particles[k];
            double q_k = charges[k];
            double Rk = born[k], dRk = dRdPsi[k];
            double diag = prefactor * q_k * q_k / (Rk*Rk*Rk) * dRk * dRk;  // self
            if (includeSurfaceArea) {
                double Rp = radii[k] + probe;
                double ratio = radii[k] / Rk;
                double ratio6 = ratio*ratio*ratio*ratio*ratio*ratio;
                double E_SA = surfaceTension * 4.0 * M_PI * Rp * Rp * ratio6;
                double d2E_SA = 42.0 * E_SA / (Rk*Rk);
                diag += d2E_SA * dRk * dRk;
            }
            diag += dE_dR[k] * d2RdPsi2[k];     // OBC curvature
            for (int l = 0; l < N; l++) {
                if (l == k) continue;
                int pl = particles[l];
                double dx = posData[pl][0] - posData[pk][0];
                double dy = posData[pl][1] - posData[pk][1];
                double dz = posData[pl][2] - posData[pk][2];
                double r2 = dx*dx+dy*dy+dz*dz;
                double Rl = born[l], dRl = dRdPsi[l];
                StillPair sp = stillPair(r2, Rk, Rl);
                double f = sp.f, f2 = sp.f2, et = sp.et;
                double C = prefactor * q_k * charges[l];
                double df2_dRk = et * (Rl + 0.25*r2/Rk);
                double df_dRk = df2_dRk / (2.0*f);
                double dalpha_dRk = et * r2 / (4.0*Rk*Rk*Rl);
                double d2f2_dRk2 = dalpha_dRk * (Rl + 0.25*r2/Rk) + et*(-0.25*r2/(Rk*Rk));
                double d2f_dRk2 = d2f2_dRk2/(2.0*f) - df2_dRk*df2_dRk/(4.0*f*f*f);
                double d2E_dRk2 = C*(2.0*df_dRk*df_dRk/(f*f*f) - d2f_dRk2/(f*f));
                diag += d2E_dRk2 * dRk * dRk;

                double df2_dRl = et * (Rk + 0.25*r2/Rl);
                double df_dRl = df2_dRl / (2.0*f);
                double dalpha_dRl = et * r2 / (4.0*Rk*Rl*Rl);
                double d2f2_dRkRl = dalpha_dRl*(Rl + 0.25*r2/Rk) + et;
                double d2f_dRkRl = d2f2_dRkRl/(2.0*f) - df2_dRk*df2_dRl/(4.0*f*f*f);
                double d2E_dRkRl = C*(2.0*df_dRk*df_dRl/(f*f*f) - d2f_dRkRl/(f*f));
                M[(size_t)k*N + l] = d2E_dRkRl * dRk * dRl;
            }
            M[(size_t)k*N + k] = diag;
        }

        // ---- Assemble ligand block (mirrors assembleGBSAHessianDouble) ----
        // Hloc holds the unscaled (scale=1) group Hessian; scaled at scatter.
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
                            // diagonal atom block: sum over partner pairs l
                            double pix = posData[pai][0], piy = posData[pai][1], piz = posData[pai][2];
                            double q_i = charges[ai];
                            double Ri = born[ai];
                            double Ri_off = radii[ai] - DIELECTRIC_OFFSET;
                            double Si = Ri_off * scaleFactors[ai];
                            for (int l = 0; l < N; l++) {
                                if (l == ai) continue;
                                int pl = particles[l];
                                double dx = posData[pl][0]-pix, dy = posData[pl][1]-piy, dz = posData[pl][2]-piz;
                                double r2 = dx*dx+dy*dy+dz*dz, r = sqrt(r2);
                                if (r < 1e-10) continue;
                                double Rl = born[l];
                                StillPair sp = stillPair(r2, Ri, Rl);
                                double f = sp.f, f2 = sp.f2, et = sp.et;
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
                                // mixed r-R "g" terms
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

                                double D[3] = {dx, dy, dz};
                                double ir = 1.0/r, ir2 = ir*ir;
                                Hval += (d2Edr2 - dEdr*ir)*D[a]*D[b]*ir2
                                      + ((a==b) ? dEdr*ir : 0.0);
                                double dr_a = -D[a]*ir, dr_b = -D[b]*ir;
                                Hval += g_i*(dr_a*J[(size_t)ai*n3 + 3*ai + b] + J[(size_t)ai*n3 + 3*ai + a]*dr_b);
                                Hval += g_l*(dr_a*J[(size_t)l*n3 + 3*ai + b] + J[(size_t)l*n3 + 3*ai + a]*dr_b);
                            }
                            // receptor self-Hessian: dE/dHCT_ai * d2(hctReceptor_ai)/dx dx
                            if (pairwise) {
                                double Hxx=0,Hyy=0,Hzz=0,Hxy=0,Hxz=0,Hyz=0;
                                double Ri_off2 = radii[ai] - DIELECTRIC_OFFSET;
                                for (int rj = 0; rj < numReceptorAtoms; rj++) {
                                    double dx = pix-receptorPositions[rj*3];
                                    double dy = piy-receptorPositions[rj*3+1];
                                    double dz = piz-receptorPositions[rj*3+2];
                                    double r2 = dx*dx+dy*dy+dz*dz, r = sqrt(r2);
                                    if (r < 1e-10) continue;
                                    double Rrj_off = receptorRadii[rj] - DIELECTRIC_OFFSET;
                                    double Srj = Rrj_off * receptorScaleFactors[rj];
                                    if (Ri_off2 >= r + Srj) continue;
                                    double I1 = computeHCTTermDerivative(r, Ri_off2, Rrj_off, receptorScaleFactors[rj]);
                                    double I2 = computeHCTTermSecondDerivative(r, Ri_off2, Rrj_off, receptorScaleFactors[rj]);
                                    double invr=1.0/r, rhx=dx*invr, rhy=dy*invr, rhz=dz*invr;
                                    double A = I1*invr, Bc = I2 - A;
                                    Hxx += A + Bc*rhx*rhx; Hyy += A + Bc*rhy*rhy; Hzz += A + Bc*rhz*rhz;
                                    Hxy += Bc*rhx*rhy; Hxz += Bc*rhx*rhz; Hyz += Bc*rhy*rhz;
                                }
                                double Hr[3][3] = {{Hxx,Hxy,Hxz},{Hxy,Hyy,Hyz},{Hxz,Hyz,Hzz}};
                                Hval += dE_dHCT[ai] * Hr[a][b];
                            }
                        } else {
                            // off-diagonal atom block (ai != aj, same group)
                            double pix = posData[pai][0], piy = posData[pai][1], piz = posData[pai][2];
                            double pjx = posData[paj][0], pjy = posData[paj][1], pjz = posData[paj][2];
                            double q_i = charges[ai], q_j = charges[aj];
                            double Ri = born[ai], Rj = born[aj];
                            double dx = pjx-pix, dy = pjy-piy, dz = pjz-piz;
                            double r2 = dx*dx+dy*dy+dz*dz, r = sqrt(r2);
                            if (r >= 1e-10) {
                                StillPair sp = stillPair(r2, Ri, Rj);
                                double f = sp.f, f2 = sp.f2, et = sp.et;
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
                                double D[3] = {dx, dy, dz};
                                Hval += -(d2Edr2 - dEdr*ir)*D[a]*D[b]*ir2
                                      - ((a==b) ? dEdr*ir : 0.0);
                            }
                            // g-terms: part A (pairs involving ai), part B (involving aj)
                            for (int l = 0; l < N; l++) {
                                if (l == ai) continue;
                                int pl = particles[l];
                                double dx_=posData[pl][0]-pix, dy_=posData[pl][1]-piy, dz_=posData[pl][2]-piz;
                                double r2_=dx_*dx_+dy_*dy_+dz_*dz_, r_=sqrt(r2_);
                                if (r_ < 1e-10) continue;
                                double Rl = born[l];
                                StillPair sp = stillPair(r2_, Ri, Rl);
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
                                double D_[3]={dx_,dy_,dz_};
                                double dr_a = -D_[a]/r_;
                                double v_col = g_i*J[(size_t)ai*n3+col] + g_l*J[(size_t)l*n3+col];
                                Hval += dr_a*v_col;
                            }
                            for (int m = 0; m < N; m++) {
                                if (m == aj) continue;
                                int pm = particles[m];
                                double dx_=posData[pm][0]-pjx, dy_=posData[pm][1]-pjy, dz_=posData[pm][2]-pjz;
                                double r2_=dx_*dx_+dy_*dy_+dz_*dz_, r_=sqrt(r2_);
                                if (r_ < 1e-10) continue;
                                double Rm = born[m];
                                StillPair sp = stillPair(r2_, Rj, Rm);
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
                                double D_[3]={dx_,dy_,dz_};
                                double dr_b = -D_[b]/r_;
                                double v_row = g_j*J[(size_t)aj*n3+row] + g_m*J[(size_t)m*n3+row];
                                Hval += v_row*dr_b;
                            }
                        }

                        // J^T M J (ligand Born coupling)
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

        // ---- PAIRWISE: receptor desolvation + cross-term Hessian ----
        if (pairwise) {
            addPairwiseHessianContributions(g, posData, born, hctReceptor, hctLigand,
                                            hctTotal, dRdPsi, J, Hloc);
        }

        // Scale and scatter the group block into the global Hessian.
        for (int r = 0; r < n3; r++) {
            int gi = particles[r/3];
            int grow = 3*gi + (r%3);
            for (int c = 0; c < n3; c++) {
                int gj = particles[c/3];
                int gcol = 3*gj + (c%3);
                H[(size_t)grow*dim3N + gcol] = scale * Hloc[(size_t)r*n3 + c];
            }
        }
    });

    return H;
}

// ==================== PAIRWISE desolvation + cross-term Hessian ===========
//
// Receptor positions are frozen; only ligand coordinates are variables.
//
//   Desolv(x) = E_rec( R^R(x) )                 (receptor GB energy with
//               ligand screening), R^R_j = G(recSelf_j + PsiR_j(x)),
//               PsiR_j(x) = sum_i HCT(r_{ji}; rec_off_j, lig_off_i, lig_s_i)
//
//   E_cross(x)= sum_{i,j} pf q_i rq_j / f(r_{ij}, R^L_i(x), R^R_j(x))
//
// Both R^L (ligand) and R^R (receptor) Born radii depend on x. d2R/dx dx =
// dR/dPsi * d2Psi/dx dx + d2R/dPsi2 * dPsi/dx dPsi/dx. The Desolv part has no
// explicit-r term (E_rec depends on x only through R^R); the cross part has an
// explicit r_{ij} dependence plus R^L and R^R dependence.
void ReferenceCalcIsolatedGBSAForceKernel::addPairwiseHessianContributions(
        int groupIndex,
        const vector<Vec3>& posData,
        const vector<double>& born,
        const vector<double>& hctReceptor,
        const vector<double>& hctLigand,
        const vector<double>& hctTotal,
        const vector<double>& dRdPsi,
        const vector<double>& J,
        vector<double>& Hloc) const {

    bool isHCT = (gbMethod == IsolatedGBSAForce::HCT);
    const vector<int>& particles = groupParticleIndices[groupIndex];
    int N = numAtoms;
    int n3 = 3 * N;
    int Nr = numReceptorAtoms;
    static constexpr double MIN_CROSS_R2 = 0.01;

    // ---- receptor screening sums PsiR_j and receptor Born radii ----
    vector<double> ligToRec(Nr, 0.0);
    for (int j = 0; j < Nr; j++) {
        double Rj_off = receptorRadii[j] - DIELECTRIC_OFFSET;
        for (int i = 0; i < N; i++) {
            int pi = particles[i];
            double dx = receptorPositions[j*3]-posData[pi][0];
            double dy = receptorPositions[j*3+1]-posData[pi][1];
            double dz = receptorPositions[j*3+2]-posData[pi][2];
            double r = sqrt(dx*dx+dy*dy+dz*dz);
            if (cutoffDistance > 0.0 && r > cutoffDistance) continue;
            double Ri_off = radii[i] - DIELECTRIC_OFFSET;
            ligToRec[j] += computeHCTTerm(r, Rj_off, Ri_off, scaleFactors[i]);
        }
    }
    vector<double> recBorn(Nr), recDRdPsi(Nr), recD2RdPsi2(Nr);
    for (int j = 0; j < Nr; j++) {
        double Rj_off = receptorRadii[j] - DIELECTRIC_OFFSET;
        double tot = receptorSelfHCT[j] + ligToRec[j];
        double R;
        if (isHCT) {
            double inner = 1.0/Rj_off - 0.5*Rj_off*tot;
            R = (inner > 0.0) ? 1.0/inner : 500.0;
        } else {
            double psi = 0.5*Rj_off*tot;
            double tv = tanh(OBC_ALPHA*psi - OBC_BETA*psi*psi + OBC_GAMMA*psi*psi*psi);
            double inner = 1.0/Rj_off - tv/receptorRadii[j];
            R = (inner > 0.0) ? 1.0/inner : 500.0;
        }
        recBorn[j] = R;
        BornDerivs bd = bornTransformDerivs(receptorRadii[j], Rj_off, tot, R, isHCT);
        recDRdPsi[j] = bd.dRdPsi;
        recD2RdPsi2[j] = bd.d2RdPsi2;
    }

    // ---- receptor energy derivatives dE_rec/dR^R_j and the receptor coupling
    //      matrix MR[j][m] = d2E_rec/dPsiR_j dPsiR_m (receptor Still pairs).
    //      Also dE_rec/dHCT_j = dE_rec/dR^R_j * recDRdPsi[j].
    vector<double> recDeDR(Nr, 0.0);
    vector<double> MR(static_cast<size_t>(Nr) * Nr, 0.0);
    for (int j = 0; j < Nr; j++) {
        double Rj = recBorn[j], dRj = recDRdPsi[j];
        double qj = receptorCharges[j];
        recDeDR[j] += -0.5 * prefactor * qj*qj / (Rj*Rj);
        double diag = prefactor * qj*qj / (Rj*Rj*Rj) * dRj * dRj;     // self curvature
        for (int m = 0; m < Nr; m++) {
            if (m == j) continue;
            double dx = receptorPositions[j*3]-receptorPositions[m*3];
            double dy = receptorPositions[j*3+1]-receptorPositions[m*3+1];
            double dz = receptorPositions[j*3+2]-receptorPositions[m*3+2];
            double r2 = dx*dx+dy*dy+dz*dz;
            double Rm = recBorn[m], dRm = recDRdPsi[m];
            StillPair sp = stillPair(r2, Rj, Rm);
            double f = sp.f, et = sp.et;
            double C = prefactor * qj * receptorCharges[m];
            double df2_dRj = et*(Rm + 0.25*r2/Rj);
            double df_dRj = df2_dRj/(2.0*f);
            recDeDR[j] += -prefactor * qj * receptorCharges[m] / sp.f2 * df_dRj;
            // d2E/dRj2
            double dalpha_dRj = et*r2/(4.0*Rj*Rj*Rm);
            double d2f2_dRj2 = dalpha_dRj*(Rm+0.25*r2/Rj) + et*(-0.25*r2/(Rj*Rj));
            double d2f_dRj2 = d2f2_dRj2/(2.0*f) - df2_dRj*df2_dRj/(4.0*f*f*f);
            double d2E_dRj2 = C*(2.0*df_dRj*df_dRj/(f*f*f) - d2f_dRj2/(f*f));
            diag += d2E_dRj2 * dRj * dRj;
            // d2E/dRj dRm
            double df2_dRm = et*(Rj+0.25*r2/Rm);
            double df_dRm = df2_dRm/(2.0*f);
            double dalpha_dRm = et*r2/(4.0*Rj*Rm*Rm);
            double d2f2_dRjRm = dalpha_dRm*(Rm+0.25*r2/Rj) + et;
            double d2f_dRjRm = d2f2_dRjRm/(2.0*f) - df2_dRj*df2_dRm/(4.0*f*f*f);
            double d2E_dRjRm = C*(2.0*df_dRj*df_dRm/(f*f*f) - d2f_dRjRm/(f*f));
            MR[(size_t)j*Nr + m] = d2E_dRjRm * dRj * dRm;
        }
        // receptor OBC curvature: dE_rec/dR^R_j * d2R^R_j/dPsi2 (recDeDR[j] is
        // fully accumulated for atom j at this point).
        diag += recDeDR[j] * recD2RdPsi2[j];
        MR[(size_t)j*Nr + j] = diag;
    }

    // ---- receptor Jacobian JR[j][3*i+a] = dPsiR_j/dx_{i,a} ----
    // PsiR_j = sum_i HCT(r_{ji}); r_{ji}=rec_j-lig_i, frozen rec.
    // dPsiR_j/dx_{i,a} = I1(r_{ji}) * d r_{ji}/dx_{i,a} = I1 * (-(rec-lig)/r)
    //                  = I1 * (lig-rec)/r.
    vector<double> JR(static_cast<size_t>(Nr) * n3, 0.0);
    for (int j = 0; j < Nr; j++) {
        double Rj_off = receptorRadii[j] - DIELECTRIC_OFFSET;
        for (int i = 0; i < N; i++) {
            int pi = particles[i];
            double dx = posData[pi][0]-receptorPositions[j*3];   // lig - rec
            double dy = posData[pi][1]-receptorPositions[j*3+1];
            double dz = posData[pi][2]-receptorPositions[j*3+2];
            double r2 = dx*dx+dy*dy+dz*dz, r = sqrt(r2);
            if (r < 1e-10) continue;
            if (cutoffDistance > 0.0 && r > cutoffDistance) continue;
            double Ri_off = radii[i] - DIELECTRIC_OFFSET;
            double Si = Ri_off * scaleFactors[i];
            if (Rj_off >= r + Si) continue;
            double I1 = computeHCTTermDerivative(r, Rj_off, Ri_off, scaleFactors[i]);
            double invr = 1.0/r;
            JR[(size_t)j*n3 + 3*i+0] = I1*dx*invr;
            JR[(size_t)j*n3 + 3*i+1] = I1*dy*invr;
            JR[(size_t)j*n3 + 3*i+2] = I1*dz*invr;
        }
    }

    // ===== Desolvation Hessian = JR^T MR JR + dE_rec/dHCT_j * d2PsiR_j/dxdx =====
    // The self term: for each (j,i) pair add the receptor-style spatial Hessian
    // of HCT(r_{ji}) into ligand atom i's diagonal block, weighted by
    // dE_rec/dHCT_j = recDeDR[j]*recDRdPsi[j].
    {
        // JR^T MR JR (full N x N atom blocks)
        for (int row = 0; row < n3; row++) {
            for (int col = row; col < n3; col++) {
                double v = 0.0;
                for (int j = 0; j < Nr; j++) {
                    double Jjr = JR[(size_t)j*n3 + row];
                    if (fabs(Jjr) < 1e-18) continue;
                    for (int m = 0; m < Nr; m++) {
                        double Mjm = MR[(size_t)j*Nr + m];
                        if (fabs(Mjm) < 1e-18) continue;
                        v += Jjr * Mjm * JR[(size_t)m*n3 + col];
                    }
                }
                Hloc[(size_t)row*n3 + col] += v;
                if (col != row) Hloc[(size_t)col*n3 + row] += v;
            }
        }
        // self spatial Hessian of PsiR_j w.r.t. ligand atom i
        for (int i = 0; i < N; i++) {
            int pi = particles[i];
            double Ri_off = radii[i] - DIELECTRIC_OFFSET;
            double Si = Ri_off * scaleFactors[i];
            double Hxx=0,Hyy=0,Hzz=0,Hxy=0,Hxz=0,Hyz=0;
            for (int j = 0; j < Nr; j++) {
                double dx = posData[pi][0]-receptorPositions[j*3];
                double dy = posData[pi][1]-receptorPositions[j*3+1];
                double dz = posData[pi][2]-receptorPositions[j*3+2];
                double r2 = dx*dx+dy*dy+dz*dz, r = sqrt(r2);
                if (r < 1e-10) continue;
                double Rj_off = receptorRadii[j] - DIELECTRIC_OFFSET;
                if (Rj_off >= r + Si) continue;
                // HCT(r; Rj_off, Ri_off, s_i): I1,I2 w.r.t. r
                double I1 = computeHCTTermDerivative(r, Rj_off, Ri_off, scaleFactors[i]);
                double I2 = computeHCTTermSecondDerivative(r, Rj_off, Ri_off, scaleFactors[i]);
                double w = recDeDR[j] * recDRdPsi[j];   // dE_rec/dHCT_j
                double invr=1.0/r, rhx=dx*invr, rhy=dy*invr, rhz=dz*invr;
                double A = I1*invr, Bc = I2 - A;
                Hxx += w*(A + Bc*rhx*rhx); Hyy += w*(A + Bc*rhy*rhy); Hzz += w*(A + Bc*rhz*rhz);
                Hxy += w*Bc*rhx*rhy; Hxz += w*Bc*rhx*rhz; Hyz += w*Bc*rhy*rhz;
            }
            double Hr[3][3] = {{Hxx,Hxy,Hxz},{Hxy,Hyy,Hyz},{Hxz,Hyz,Hzz}};
            for (int a=0;a<3;a++) for (int b=0;b<3;b++)
                Hloc[(size_t)(3*i+a)*n3 + 3*i+b] += Hr[a][b];
        }
    }

    // ===== Cross-term Hessian =====
    // E_cross = sum_{i,j} C_ij / f(r_ij, R^L_i, R^R_j), C_ij = pf q_i rq_j.
    // Variables: r_ij (lig i moves), R^L_i(x), R^R_j(x).
    // First derivatives of cross energy w.r.t. the Born radii, accumulated so
    // they feed the J^T-style Born chain (gradient terms dE/dR * d2R/dxdx and
    // the d2E/dRdR couplings) below.
    vector<double> dCross_dRL(N, 0.0);   // dE_cross/dR^L_i
    vector<double> dCross_dRR(Nr, 0.0);  // dE_cross/dR^R_j

    // Accumulate the explicit-r and mixed r-R contributions atom-block by block.
    for (int i = 0; i < N; i++) {
        int pi = particles[i];
        double q_i = charges[i];
        double Ri = born[i];
        for (int j = 0; j < Nr; j++) {
            double dx = posData[pi][0]-receptorPositions[j*3];   // lig - rec
            double dy = posData[pi][1]-receptorPositions[j*3+1];
            double dz = posData[pi][2]-receptorPositions[j*3+2];
            double r2 = dx*dx+dy*dy+dz*dz;
            if (r2 < MIN_CROSS_R2) continue;
            double r = sqrt(r2);
            double Rj = recBorn[j];
            StillPair sp = stillPair(r2, Ri, Rj);
            double f = sp.f, f2 = sp.f2, et = sp.et;
            double C = prefactor * q_i * receptorCharges[j];

            // explicit r derivatives
            double df2_dr = 2.0*r*(1.0 - 0.25*et);
            double df_dr = df2_dr/(2.0*f);
            double dEdr = -C*df_dr/(f*f);
            double d2f2_dr2 = 2.0 - 0.5*et + 0.25*r2*et/(Ri*Rj);
            double d2f_dr2 = d2f2_dr2/(2.0*f) - df2_dr*df2_dr/(4.0*f*f*f);
            double d2Edr2 = C*(2.0*df_dr*df_dr/(f*f*f) - d2f_dr2/(f*f));

            // Born first derivatives
            double df2_dRi = et*(Rj + 0.25*r2/Ri);
            double df_dRi = df2_dRi/(2.0*f);
            double dE_dRi = -C*df_dRi/(f*f);
            double df2_dRj = et*(Ri + 0.25*r2/Rj);
            double df_dRj = df2_dRj/(2.0*f);
            double dE_dRj = -C*df_dRj/(f*f);
            dCross_dRL[i] += dE_dRi;
            dCross_dRR[j] += dE_dRj;

            // explicit spatial Hessian on ligand atom i (diagonal block)
            double ir = 1.0/r, ir2 = ir*ir;
            double D[3] = {dx, dy, dz};
            for (int a=0;a<3;a++) for (int b=0;b<3;b++) {
                double hv = (d2Edr2 - dEdr*ir)*D[a]*D[b]*ir2 + ((a==b)?dEdr*ir:0.0);
                Hloc[(size_t)(3*i+a)*n3 + 3*i+b] += hv;
            }

            // mixed explicit-r / R^L_i and r / R^R_j terms.
            // d2E/dr dRi:
            double dalpha_dRi = et*r2/(4.0*Ri*Ri*Rj);
            double d2f2_dr_dRi = 2.0*r*(-0.25*dalpha_dRi);
            double d2f_dr_dRi = d2f2_dr_dRi/(2.0*f) - df2_dr*df2_dRi/(4.0*f*f*f);
            double d2E_dr_dRi = C*(2.0*df_dr*df_dRi/(f*f*f) - d2f_dr_dRi/(f*f));
            double dalpha_dRj = et*r2/(4.0*Ri*Rj*Rj);
            double d2f2_dr_dRj = 2.0*r*(-0.25*dalpha_dRj);
            double d2f_dr_dRj = d2f2_dr_dRj/(2.0*f) - df2_dr*df2_dRj/(4.0*f*f*f);
            double d2E_dr_dRj = C*(2.0*df_dr*df_dRj/(f*f*f) - d2f_dr_dRj/(f*f));

            // The mixed r-R coupling and Born-Born coupling of the cross term
            // are assembled in the dedicated JT M J block below (using the
            // d2E_dr_dRi / d2E_dr_dRj values computed here is unnecessary there
            // since they are recomputed; kept local for clarity).
            (void)d2E_dr_dRi; (void)d2E_dr_dRj;
        }
    }

    // Cross-term Born gradient terms: dE_cross/dR * d2R/dxdx, expressed through
    // the spatial Hessian of the corresponding Psi sums (ligand and receptor)
    // and the dR/dPsi curvature. These mirror the ligand assembly's
    // dE/dHCT * d2HCT/dxdx and the d2R/dPsi2 diagonal, but with the cross-term
    // energy gradient as the weight.
    {
        // ligand Born gradient: weight per ligand atom = dCross_dRL[i].
        // Build dE_cross/dHCT_i (ligand) and route through ligand J spatial
        // Hessian (ligand-ligand HCT and receptor->ligand HCT) plus dRdPsi2.
        vector<double> dCross_dHCT_L(N);
        for (int i = 0; i < N; i++) dCross_dHCT_L[i] = dCross_dRL[i] * dRdPsi[i];

        // d2(Psi^L_i)/dxdx spatial Hessian weighted by dE_cross/dHCT_i:
        //   - ligand-ligand HCT pairs (i with each ligand l): both i and l blocks
        //   - receptor->ligand HCT (i with each receptor): only i diagonal block
        for (int i = 0; i < N; i++) {
            int pi = particles[i];
            double Ri_off = radii[i] - DIELECTRIC_OFFSET;
            double w = dCross_dHCT_L[i];
            // ligand-ligand
            for (int l = 0; l < N; l++) {
                if (l == i) continue;
                int pl = particles[l];
                double dx = posData[pi][0]-posData[pl][0];
                double dy = posData[pi][1]-posData[pl][1];
                double dz = posData[pi][2]-posData[pl][2];
                double r2 = dx*dx+dy*dy+dz*dz, r = sqrt(r2);
                if (r < 1e-10) continue;
                double Rl_off = radii[l] - DIELECTRIC_OFFSET;
                double Sl = Rl_off * scaleFactors[l];
                if (Ri_off >= r + Sl) continue;
                double I1 = computeHCTTermDerivative(r, Ri_off, Rl_off, scaleFactors[l]);
                double I2 = computeHCTTermSecondDerivative(r, Ri_off, Rl_off, scaleFactors[l]);
                double invr=1.0/r, rhx=dx*invr, rhy=dy*invr, rhz=dz*invr;
                double A = I1*invr, Bc = I2 - A;
                double blk[3][3] = {
                    {A+Bc*rhx*rhx, Bc*rhx*rhy, Bc*rhx*rhz},
                    {Bc*rhx*rhy, A+Bc*rhy*rhy, Bc*rhy*rhz},
                    {Bc*rhx*rhz, Bc*rhy*rhz, A+Bc*rhz*rhz}};
                for (int a=0;a<3;a++) for (int b=0;b<3;b++) {
                    double hv = w*blk[a][b];
                    Hloc[(size_t)(3*i+a)*n3 + 3*i+b] += hv;       // ii
                    Hloc[(size_t)(3*i+a)*n3 + 3*l+b] -= hv;       // il
                    Hloc[(size_t)(3*l+a)*n3 + 3*i+b] -= hv;       // li
                    Hloc[(size_t)(3*l+a)*n3 + 3*l+b] += hv;       // ll
                }
            }
            // receptor->ligand (receptor frozen): only ii block
            for (int rj = 0; rj < numReceptorAtoms; rj++) {
                double dx = posData[pi][0]-receptorPositions[rj*3];
                double dy = posData[pi][1]-receptorPositions[rj*3+1];
                double dz = posData[pi][2]-receptorPositions[rj*3+2];
                double r2 = dx*dx+dy*dy+dz*dz, r = sqrt(r2);
                if (r < 1e-10) continue;
                double Rrj_off = receptorRadii[rj] - DIELECTRIC_OFFSET;
                double Srj = Rrj_off * receptorScaleFactors[rj];
                if (Ri_off >= r + Srj) continue;
                double I1 = computeHCTTermDerivative(r, Ri_off, Rrj_off, receptorScaleFactors[rj]);
                double I2 = computeHCTTermSecondDerivative(r, Ri_off, Rrj_off, receptorScaleFactors[rj]);
                double invr=1.0/r, rhx=dx*invr, rhy=dy*invr, rhz=dz*invr;
                double A = I1*invr, Bc = I2 - A;
                double blk[3][3] = {
                    {A+Bc*rhx*rhx, Bc*rhx*rhy, Bc*rhx*rhz},
                    {Bc*rhx*rhy, A+Bc*rhy*rhy, Bc*rhy*rhz},
                    {Bc*rhx*rhz, Bc*rhy*rhz, A+Bc*rhz*rhz}};
                for (int a=0;a<3;a++) for (int b=0;b<3;b++)
                    Hloc[(size_t)(3*i+a)*n3 + 3*i+b] += w*blk[a][b];
            }
        }
        // receptor Born gradient: weight = dCross_dRR[j], route through JR self
        // spatial Hessian (already same geometry as desolvation self term).
        for (int i = 0; i < N; i++) {
            int pi = particles[i];
            double Ri_off = radii[i] - DIELECTRIC_OFFSET;
            double Si = Ri_off * scaleFactors[i];
            double Hxx=0,Hyy=0,Hzz=0,Hxy=0,Hxz=0,Hyz=0;
            for (int j = 0; j < Nr; j++) {
                double dx = posData[pi][0]-receptorPositions[j*3];
                double dy = posData[pi][1]-receptorPositions[j*3+1];
                double dz = posData[pi][2]-receptorPositions[j*3+2];
                double r2 = dx*dx+dy*dy+dz*dz, r = sqrt(r2);
                if (r < 1e-10) continue;
                double Rj_off = receptorRadii[j] - DIELECTRIC_OFFSET;
                if (Rj_off >= r + Si) continue;
                double I1 = computeHCTTermDerivative(r, Rj_off, Ri_off, scaleFactors[i]);
                double I2 = computeHCTTermSecondDerivative(r, Rj_off, Ri_off, scaleFactors[i]);
                double w = dCross_dRR[j] * recDRdPsi[j];   // dE_cross/dHCT^R_j
                double invr=1.0/r, rhx=dx*invr, rhy=dy*invr, rhz=dz*invr;
                double A = I1*invr, Bc = I2 - A;
                Hxx += w*(A+Bc*rhx*rhx); Hyy += w*(A+Bc*rhy*rhy); Hzz += w*(A+Bc*rhz*rhz);
                Hxy += w*Bc*rhx*rhy; Hxz += w*Bc*rhx*rhz; Hyz += w*Bc*rhy*rhz;
            }
            double Hr[3][3] = {{Hxx,Hxy,Hxz},{Hxy,Hyy,Hyz},{Hxz,Hyz,Hzz}};
            for (int a=0;a<3;a++) for (int b=0;b<3;b++)
                Hloc[(size_t)(3*i+a)*n3 + 3*i+b] += Hr[a][b];
        }
    }

    // Cross-term Born-Born and mixed r-Born couplings (JT M J style across the
    // ligand R^L and receptor R^R potentials). Build the combined coupling
    // contributions of the cross energy and route through the ligand J and
    // receptor JR Jacobians.
    {
        // d2E_cross/dR_a dR_b second derivatives per (i,j) pair, plus the
        // mixed explicit r-R terms folded as g-vectors. Assemble directly.
        for (int i = 0; i < N; i++) {
            int pi = particles[i];
            double q_i = charges[i];
            double Ri = born[i];
            for (int j = 0; j < Nr; j++) {
                double dx = posData[pi][0]-receptorPositions[j*3];
                double dy = posData[pi][1]-receptorPositions[j*3+1];
                double dz = posData[pi][2]-receptorPositions[j*3+2];
                double r2 = dx*dx+dy*dy+dz*dz;
                if (r2 < MIN_CROSS_R2) continue;
                double r = sqrt(r2);
                double Rj = recBorn[j];
                StillPair sp = stillPair(r2, Ri, Rj);
                double f = sp.f, et = sp.et;
                double C = prefactor * q_i * receptorCharges[j];
                double df2_dRi = et*(Rj + 0.25*r2/Ri), df_dRi = df2_dRi/(2.0*f);
                double df2_dRj = et*(Ri + 0.25*r2/Rj), df_dRj = df2_dRj/(2.0*f);
                double df2_dr = 2.0*r*(1.0-0.25*et), df_dr = df2_dr/(2.0*f);

                // second derivatives of E wrt (Ri,Rj)
                double dalpha_dRi = et*r2/(4.0*Ri*Ri*Rj);
                double d2f2_dRi2 = dalpha_dRi*(Rj+0.25*r2/Ri) + et*(-0.25*r2/(Ri*Ri));
                double d2f_dRi2 = d2f2_dRi2/(2.0*f) - df2_dRi*df2_dRi/(4.0*f*f*f);
                double d2E_dRi2 = C*(2.0*df_dRi*df_dRi/(f*f*f) - d2f_dRi2/(f*f));
                double dalpha_dRj = et*r2/(4.0*Ri*Rj*Rj);
                double d2f2_dRj2 = dalpha_dRj*(Ri+0.25*r2/Rj) + et*(-0.25*r2/(Rj*Rj));
                double d2f_dRj2 = d2f2_dRj2/(2.0*f) - df2_dRj*df2_dRj/(4.0*f*f*f);
                double d2E_dRj2 = C*(2.0*df_dRj*df_dRj/(f*f*f) - d2f_dRj2/(f*f));
                double d2f2_dRiRj = dalpha_dRj*(Rj+0.25*r2/Ri) + et;
                double d2f_dRiRj = d2f2_dRiRj/(2.0*f) - df2_dRi*df2_dRj/(4.0*f*f*f);
                double d2E_dRiRj = C*(2.0*df_dRi*df_dRj/(f*f*f) - d2f_dRiRj/(f*f));

                // mixed r-R
                double d2f2_dr_dRi = 2.0*r*(-0.25*dalpha_dRi);
                double d2f_dr_dRi = d2f2_dr_dRi/(2.0*f) - df2_dr*df2_dRi/(4.0*f*f*f);
                double d2E_dr_dRi = C*(2.0*df_dr*df_dRi/(f*f*f) - d2f_dr_dRi/(f*f));
                double d2f2_dr_dRj = 2.0*r*(-0.25*dalpha_dRj);
                double d2f_dr_dRj = d2f2_dr_dRj/(2.0*f) - df2_dr*df2_dRj/(4.0*f*f*f);
                double d2E_dr_dRj = C*(2.0*df_dr*df_dRj/(f*f*f) - d2f_dr_dRj/(f*f));

                // dR^L_i/dx = dRdPsi[i]*J[i][.]; dR^R_j/dx = recDRdPsi[j]*JR[j][.]
                // dr/dx_{i,a} = D[a]/r at ligand atom i (receptor frozen).
                double ir = 1.0/r;
                double D[3] = {dx, dy, dz};
                double cRi = d2E_dRi2 * dRdPsi[i]*dRdPsi[i];
                double cRj = d2E_dRj2 * recDRdPsi[j]*recDRdPsi[j];
                double cRiRj = d2E_dRiRj * dRdPsi[i]*recDRdPsi[j];
                double gri = d2E_dr_dRi * dRdPsi[i];
                double grj = d2E_dr_dRj * recDRdPsi[j];

                for (int row = 0; row < n3; row++) {
                    double JLi_row = J[(size_t)i*n3 + row];
                    double JRj_row = JR[(size_t)j*n3 + row];
                    // dr/dx contributes only at ligand atom i rows
                    double dr_row = 0.0;
                    if (row/3 == i) dr_row = D[row%3]*ir;
                    if (fabs(JLi_row) < 1e-300 && fabs(JRj_row) < 1e-300 && dr_row == 0.0)
                        continue;
                    for (int col = row; col < n3; col++) {
                        double JLi_col = J[(size_t)i*n3 + col];
                        double JRj_col = JR[(size_t)j*n3 + col];
                        double dr_col = 0.0;
                        if (col/3 == i) dr_col = D[col%3]*ir;
                        double v = cRi*JLi_row*JLi_col
                                 + cRj*JRj_row*JRj_col
                                 + cRiRj*(JLi_row*JRj_col + JRj_row*JLi_col)
                                 + gri*(dr_row*JLi_col + JLi_row*dr_col)
                                 + grj*(dr_row*JRj_col + JRj_row*dr_col);
                        Hloc[(size_t)row*n3 + col] += v;
                        if (col != row) Hloc[(size_t)col*n3 + row] += v;
                    }
                }
            }
        }
    }

    // Single-Born curvature of the cross term: dE_cross/dR * d2R/dPsi2, routed
    // through J (ligand) and JR (receptor). The Born-Born outer products above
    // carry only the d2E/dR2 (dR/dPsi)^2 part; this adds the (dE/dR)(d2R/dPsi2)
    // curvature for each Born radius (the term the desolvation path folds into
    // its MR diagonal, and the core into M).
    for (int i = 0; i < N; i++) {
        double Ri_off = radii[i] - DIELECTRIC_OFFSET;
        BornDerivs bd = bornTransformDerivs(radii[i], Ri_off, hctTotal[i], born[i], isHCT);
        double w = dCross_dRL[i] * bd.d2RdPsi2;
        if (fabs(w) < 1e-300) continue;
        for (int row = 0; row < n3; row++) {
            double Jr = J[(size_t)i*n3 + row];
            if (fabs(Jr) < 1e-300) continue;
            for (int col = row; col < n3; col++) {
                double v = w * Jr * J[(size_t)i*n3 + col];
                Hloc[(size_t)row*n3 + col] += v;
                if (col != row) Hloc[(size_t)col*n3 + row] += v;
            }
        }
    }
    for (int j = 0; j < Nr; j++) {
        double w = dCross_dRR[j] * recD2RdPsi2[j];
        if (fabs(w) < 1e-300) continue;
        for (int row = 0; row < n3; row++) {
            double Jr = JR[(size_t)j*n3 + row];
            if (fabs(Jr) < 1e-300) continue;
            for (int col = row; col < n3; col++) {
                double v = w * Jr * JR[(size_t)j*n3 + col];
                Hloc[(size_t)row*n3 + col] += v;
                if (col != row) Hloc[(size_t)col*n3 + row] += v;
            }
        }
    }
}

// ==================== Per-group energy accessors ====================

double ReferenceCalcIsolatedGBSAForceKernel::getGroupEnergy(int groupIndex) const {
    if (groupIndex < 0 || groupIndex >= numParticleGroups)
        throw OpenMMException("IsolatedGBSAForce: invalid group index");
    return groupEnergies_[groupIndex];
}

double ReferenceCalcIsolatedGBSAForceKernel::getGroupLigandSelfEnergy(int groupIndex) const {
    if (groupIndex < 0 || groupIndex >= numParticleGroups)
        throw OpenMMException("IsolatedGBSAForce: invalid group index");
    return groupLigandSelfEnergies_[groupIndex];
}

double ReferenceCalcIsolatedGBSAForceKernel::getGroupReceptorContribution(int groupIndex) const {
    if (groupIndex < 0 || groupIndex >= numParticleGroups)
        throw OpenMMException("IsolatedGBSAForce: invalid group index");
    return groupReceptorContributions_[groupIndex];
}

double ReferenceCalcIsolatedGBSAForceKernel::getGroupReceptorDesolvation(int groupIndex) const {
    if (groupIndex < 0 || groupIndex >= numParticleGroups)
        throw OpenMMException("IsolatedGBSAForce: invalid group index");
    return groupReceptorDesolvations_[groupIndex];
}

double ReferenceCalcIsolatedGBSAForceKernel::getGroupCrossTermEnergy(int groupIndex) const {
    if (groupIndex < 0 || groupIndex >= numParticleGroups)
        throw OpenMMException("IsolatedGBSAForce: invalid group index");
    return groupCrossTermEnergies_[groupIndex];
}

vector<double> ReferenceCalcIsolatedGBSAForceKernel::getGroupBornRadii(int groupIndex) const {
    if (groupIndex < 0 || groupIndex >= numParticleGroups)
        throw OpenMMException("IsolatedGBSAForce: invalid group index");
    return groupBornRadii_[groupIndex];
}

vector<double> ReferenceCalcIsolatedGBSAForceKernel::getGroupAtomEnergies(int groupIndex) const {
    if (groupIndex < 0 || groupIndex >= numParticleGroups)
        throw OpenMMException("IsolatedGBSAForce: invalid group index");
    return groupAtomEnergies_[groupIndex];
}

vector<double> ReferenceCalcIsolatedGBSAForceKernel::getReceptorBornRadii(int groupIndex) const {
    if (receptorMode != IsolatedGBSAForce::PAIRWISE)
        throw OpenMMException("IsolatedGBSAForce: receptor Born radii only available in PAIRWISE mode");
    if (groupIndex < 0 || groupIndex >= numParticleGroups)
        throw OpenMMException("IsolatedGBSAForce: invalid group index");
    return groupReceptorBornRadii_[groupIndex];
}

vector<double> ReferenceCalcIsolatedGBSAForceKernel::getParticleGroupUnscaledEnergies() const {
    // Reference platform: return empty vector (not implemented)
    // The CUDA platform is the primary target for this optimization
    return vector<double>(numParticleGroups, 0.0);
}

}  // namespace GridForcePlugin
