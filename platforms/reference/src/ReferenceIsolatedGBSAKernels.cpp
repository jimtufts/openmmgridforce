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

double ReferenceCalcIsolatedGBSAForceKernel::execute(
        ContextImpl& context, bool includeForces, bool includeEnergy) {

    vector<Vec3>& posData = refExtractPositions(context);
    vector<Vec3>& forceData = refExtractForces(context);

    double totalEnergy = 0.0;

    // Clear per-group results
    fill(groupEnergies_.begin(), groupEnergies_.end(), 0.0);
    fill(groupLigandSelfEnergies_.begin(), groupLigandSelfEnergies_.end(), 0.0);
    fill(groupReceptorContributions_.begin(), groupReceptorContributions_.end(), 0.0);
    fill(groupReceptorDesolvations_.begin(), groupReceptorDesolvations_.end(), 0.0);
    fill(groupCrossTermEnergies_.begin(), groupCrossTermEnergies_.end(), 0.0);

    for (int g = 0; g < numParticleGroups; g++) {
        double scale = globalScalingFactor * groupScalingFactors[g];
        if (scale == 0.0) continue;

        const vector<int>& particles = groupParticleIndices[g];

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
            // Pairwise receptor HCT
            for (int i = 0; i < numAtoms; i++) {
                int pi = particles[i];
                double R_i_off = radii[i] - DIELECTRIC_OFFSET;
                for (int j = 0; j < numReceptorAtoms; j++) {
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

        totalEnergy += groupEnergies_[g];

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
            // 7a: Compute ligand→receptor HCT screening
            vector<double> ligandToRecHCT(numReceptorAtoms, 0.0);
            for (int j = 0; j < numReceptorAtoms; j++) {
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
            vector<double> recBornRadiiWithLig(numReceptorAtoms);
            for (int j = 0; j < numReceptorAtoms; j++) {
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

            // 7c: Receptor GB energy with ligand present
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

            double desolvation = (recEnergyWithLig - receptorReferenceEnergy) * scale;
            groupReceptorDesolvations_[g] = desolvation;
            groupEnergies_[g] += desolvation;
            totalEnergy += desolvation;

            // 7d: Cross-term energy (receptor-ligand GB pairs)
            double crossTermEnergy = 0.0;
            for (int i = 0; i < numAtoms; i++) {
                int pi = particles[i];
                for (int j = 0; j < numReceptorAtoms; j++) {
                    double dx = posData[pi][0] - receptorPositions[j * 3];
                    double dy = posData[pi][1] - receptorPositions[j * 3 + 1];
                    double dz = posData[pi][2] - receptorPositions[j * 3 + 2];
                    double r2 = dx * dx + dy * dy + dz * dz;

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
                    }
                }
            }

            crossTermEnergy *= scale;
            groupCrossTermEnergies_[g] = crossTermEnergy;
            groupEnergies_[g] += crossTermEnergy;
            totalEnergy += crossTermEnergy;

            // 7e: Receptor desolvation forces on ligand atoms
            if (includeForces) {
                // Compute receptor dE/dR_born (for chain rule through receptor Born radii)
                vector<double> recDeDR(numReceptorAtoms, 0.0);
                for (int i = 0; i < numReceptorAtoms; i++) {
                    // Self term
                    recDeDR[i] += -0.5 * prefactor * receptorCharges[i] * receptorCharges[i]
                                  / (recBornRadiiWithLig[i] * recBornRadiiWithLig[i]);
                    // Pair terms with other receptor atoms
                    for (int j = 0; j < numReceptorAtoms; j++) {
                        if (i == j) continue;
                        double dx = receptorPositions[i * 3] - receptorPositions[j * 3];
                        double dy = receptorPositions[i * 3 + 1] - receptorPositions[j * 3 + 1];
                        double dz = receptorPositions[i * 3 + 2] - receptorPositions[j * 3 + 2];
                        double r2 = dx * dx + dy * dy + dz * dz;
                        double D = recBornRadiiWithLig[i] * recBornRadiiWithLig[j];
                        double alpha_val = r2 / (4.0 * D);
                        double exp_alpha = exp(-alpha_val);
                        double f_gb2 = r2 + D * exp_alpha;
                        double f_gb = sqrt(f_gb2);
                        double qq = receptorCharges[i] * receptorCharges[j];

                        double df_dRi = recBornRadiiWithLig[j] * exp_alpha
                                        * (1.0 + alpha_val) / (2.0 * f_gb);
                        recDeDR[i] += -prefactor * qq / f_gb2 * df_dRi;
                    }
                }

                // Compute receptor dR_born/dHCT
                vector<double> recDRdHCT(numReceptorAtoms, 0.0);
                for (int j = 0; j < numReceptorAtoms; j++) {
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
                // F_ligand += -scale * sum_j (recDeDR[j] * recDRdHCT[j] * dHCT_j/dpos_ligand)
                for (int i = 0; i < numAtoms; i++) {
                    int pi = particles[i];
                    double R_i_off = radii[i] - DIELECTRIC_OFFSET;

                    for (int j = 0; j < numReceptorAtoms; j++) {
                        double dx = receptorPositions[j * 3] - posData[pi][0];
                        double dy = receptorPositions[j * 3 + 1] - posData[pi][1];
                        double dz = receptorPositions[j * 3 + 2] - posData[pi][2];
                        double r2 = dx * dx + dy * dy + dz * dz;
                        double r = sqrt(r2);
                        if (r < 1e-10) continue;

                        if (cutoffDistance > 0.0 && r > cutoffDistance) continue;

                        double R_j_off = receptorRadii[j] - DIELECTRIC_OFFSET;
                        // dHCT_j/dr where r is distance from receptor j to ligand i
                        double dHCT_dr = computeHCTTermDerivative(r, R_j_off, R_i_off,
                                                                   scaleFactors[i]);

                        // Force on ligand atom: -dE/dr along receptor→ligand direction
                        // The distance vector is rec→lig, so force on ligand is
                        // along -rec→lig direction
                        double recChainFactor = recDeDR[j] * recDRdHCT[j];
                        double forceMag = -scale * recChainFactor * dHCT_dr;
                        double invR = 1.0 / r;
                        // dx,dy,dz point from ligand to receptor, so negate for force on ligand
                        forceData[pi][0] -= forceMag * dx * invR;
                        forceData[pi][1] -= forceMag * dy * invR;
                        forceData[pi][2] -= forceMag * dz * invR;
                    }
                }
            }
        }
    }

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

    // Update alchemical scaling
    globalScalingFactor = force.getGlobalScalingFactor();
    int nGroups = force.getNumParticleGroups();
    for (int g = 0; g < nGroups; g++) {
        groupScalingFactors[g] = force.getGroupScalingFactor(g);
    }
}

// ==================== Hessian ====================

vector<double> ReferenceCalcIsolatedGBSAForceKernel::computeHessian(ContextImpl& context) {
    throw OpenMMException("IsolatedGBSAForce: Hessian computation not yet implemented");
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
