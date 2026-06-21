/* -------------------------------------------------------------------------- *
 *                              OpenMMGridForce                               *
 * -------------------------------------------------------------------------- *
 * Reference platform implementation of IsolatedBondedForce kernel.           *
 * Computes harmonic bonds, harmonic angles, and periodic torsions within     *
 * isolated particle groups on CPU.                                           *
 *                                                                            *
 * Bond/angle/torsion math adapted from OpenMM Reference platform:            *
 *   ReferenceHarmonicBondIxn.cpp                                             *
 *   ReferenceAngleBondIxn.cpp                                                *
 *   ReferenceProperDihedralBond.cpp                                          *
 * -------------------------------------------------------------------------- */

#include "ReferenceIsolatedBondedKernels.h"
#include "ReferenceGridInterpolation.h"
#include "IsolatedBondedForce.h"
#include "internal/BondedHessianAnalytical.h"

#include "openmm/OpenMMException.h"
#include "openmm/internal/ContextImpl.h"

#include <cmath>
#include <algorithm>

using namespace OpenMM;
using namespace std;

namespace GridForcePlugin {

// ==================== initialize ====================

void ReferenceCalcIsolatedBondedForceKernel::initialize(
        const System& system, const IsolatedBondedForce& force) {

    numAtoms = force.getNumAtoms();
    if (numAtoms == 0)
        throw OpenMMException("IsolatedBondedForce: Must set number of atoms before initialization");

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
                throw OpenMMException("IsolatedBondedForce: particle group " + name + " has wrong number of indices");
            groupParticleIndices[g] = indices;
        }
    } else {
        throw OpenMMException("IsolatedBondedForce: Must add at least one particle group");
    }

    // Alchemical scaling
    globalScalingFactor = force.getGlobalScalingFactor();
    groupScalingFactors.resize(numParticleGroups, 1.0);
    for (int g = 0; g < nGroups; g++) {
        groupScalingFactors[g] = force.getGroupScalingFactor(g);
    }

    // Extract bond parameters
    int numBonds = force.getNumBonds();
    bonds.resize(numBonds);
    for (int i = 0; i < numBonds; i++) {
        force.getBondParameters(i, bonds[i].atom1, bonds[i].atom2,
                               bonds[i].length, bonds[i].k);
    }

    // Extract angle parameters
    int numAngles = force.getNumAngles();
    angles.resize(numAngles);
    for (int i = 0; i < numAngles; i++) {
        force.getAngleParameters(i, angles[i].atom1, angles[i].atom2, angles[i].atom3,
                                angles[i].angle, angles[i].k);
    }

    // Extract torsion parameters
    int numTorsions = force.getNumTorsions();
    torsions.resize(numTorsions);
    for (int i = 0; i < numTorsions; i++) {
        force.getTorsionParameters(i, torsions[i].atom1, torsions[i].atom2,
                                  torsions[i].atom3, torsions[i].atom4,
                                  torsions[i].periodicity, torsions[i].phase,
                                  torsions[i].k);
    }

    // Initialize per-group energy storage
    groupEnergies.resize(numParticleGroups, 0.0);
}

// ==================== execute ====================

void ReferenceCalcIsolatedBondedForceKernel::computeGroup(
        int g, vector<Vec3>& posData, vector<Vec3>& forceData,
        bool includeForces, bool includeEnergy) {

    double scale = globalScalingFactor * groupScalingFactors[g];
    if (scale == 0.0) return;

    const vector<int>& particles = groupParticleIndices[g];
    double groupEnergy = 0.0;

    {
        // ========== Harmonic Bonds ==========
        // E = 0.5 * k * (r - r0)^2
        // Adapted from ReferenceHarmonicBondIxn::calculateBondIxn

        for (int b = 0; b < (int)bonds.size(); b++) {
            int pI = particles[bonds[b].atom1];
            int pJ = particles[bonds[b].atom2];

            double dx = posData[pI][0] - posData[pJ][0];
            double dy = posData[pI][1] - posData[pJ][1];
            double dz = posData[pI][2] - posData[pJ][2];
            double r = sqrt(dx * dx + dy * dy + dz * dz);

            double deltaIdeal = r - bonds[b].length;
            double energy = 0.5 * bonds[b].k * deltaIdeal * deltaIdeal * scale;

            if (includeEnergy)
                groupEnergy += energy;

            if (includeForces && r > 0.0) {
                double dEdR = bonds[b].k * deltaIdeal * scale / r;
                forceData[pI][0] -= dEdR * dx;
                forceData[pI][1] -= dEdR * dy;
                forceData[pI][2] -= dEdR * dz;
                forceData[pJ][0] += dEdR * dx;
                forceData[pJ][1] += dEdR * dy;
                forceData[pJ][2] += dEdR * dz;
            }
        }

        // ========== Harmonic Angles ==========
        // E = 0.5 * k * (theta - theta0)^2
        // Adapted from ReferenceAngleBondIxn::calculateBondIxn
        // Vector convention matches OpenMM exactly:
        //   deltaR[0] = B - A, deltaR[1] = B - C

        for (int a = 0; a < (int)angles.size(); a++) {
            int pA = particles[angles[a].atom1];
            int pB = particles[angles[a].atom2];  // central atom
            int pC = particles[angles[a].atom3];

            // Vectors matching OpenMM: v0 = B-A, v1 = B-C
            double v0x = posData[pB][0] - posData[pA][0];
            double v0y = posData[pB][1] - posData[pA][1];
            double v0z = posData[pB][2] - posData[pA][2];
            double rBA2 = v0x * v0x + v0y * v0y + v0z * v0z;

            double v1x = posData[pB][0] - posData[pC][0];
            double v1y = posData[pB][1] - posData[pC][1];
            double v1z = posData[pB][2] - posData[pC][2];
            double rBC2 = v1x * v1x + v1y * v1y + v1z * v1z;

            // Cross product p = v0 x v1 = (B-A) x (B-C)
            double px = v0y * v1z - v0z * v1y;
            double py = v0z * v1x - v0x * v1z;
            double pz = v0x * v1y - v0y * v1x;
            double rp = sqrt(px * px + py * py + pz * pz);
            if (rp < 1.0e-06) rp = 1.0e-06;

            double dot = v0x * v1x + v0y * v1y + v0z * v1z;
            double cosine = dot / sqrt(rBA2 * rBC2);
            if (cosine > 1.0) cosine = 1.0;
            if (cosine < -1.0) cosine = -1.0;

            double theta = acos(cosine);
            double deltaTheta = theta - angles[a].angle;
            double energy = 0.5 * angles[a].k * deltaTheta * deltaTheta * scale;
            double dEdTheta = angles[a].k * deltaTheta * scale;

            if (includeEnergy)
                groupEnergy += energy;

            if (includeForces) {
                // Force decomposition from ReferenceAngleBondIxn
                double termA =  dEdTheta / (rBA2 * rp);
                double termC = -dEdTheta / (rBC2 * rp);

                // crossA = v0 x p, crossC = v1 x p
                double crossAx = v0y * pz - v0z * py;
                double crossAy = v0z * px - v0x * pz;
                double crossAz = v0x * py - v0y * px;

                double crossCx = v1y * pz - v1z * py;
                double crossCy = v1z * px - v1x * pz;
                double crossCz = v1x * py - v1y * px;

                double fAx = termA * crossAx;
                double fAy = termA * crossAy;
                double fAz = termA * crossAz;

                double fCx = termC * crossCx;
                double fCy = termC * crossCy;
                double fCz = termC * crossCz;

                // A, C get their forces; B gets negative sum
                forceData[pA][0] += fAx;
                forceData[pA][1] += fAy;
                forceData[pA][2] += fAz;
                forceData[pC][0] += fCx;
                forceData[pC][1] += fCy;
                forceData[pC][2] += fCz;
                forceData[pB][0] -= (fAx + fCx);
                forceData[pB][1] -= (fAy + fCy);
                forceData[pB][2] -= (fAz + fCz);
            }
        }

        // ========== Periodic Torsions ==========
        // E = k * (1 + cos(n*phi - phase))
        // Adapted from ReferenceProperDihedralBond::calculateBondIxn
        // Vector convention matches OpenMM exactly:
        //   deltaR[0] = A - B, deltaR[1] = C - B, deltaR[2] = C - D

        for (int t = 0; t < (int)torsions.size(); t++) {
            int pA = particles[torsions[t].atom1];
            int pB = particles[torsions[t].atom2];
            int pC = particles[torsions[t].atom3];
            int pD = particles[torsions[t].atom4];

            // Vectors matching OpenMM: v1 = A-B, v2 = C-B, v3 = C-D
            double v1x = posData[pA][0] - posData[pB][0];
            double v1y = posData[pA][1] - posData[pB][1];
            double v1z = posData[pA][2] - posData[pB][2];

            double v2x = posData[pC][0] - posData[pB][0];
            double v2y = posData[pC][1] - posData[pB][1];
            double v2z = posData[pC][2] - posData[pB][2];

            double v3x = posData[pC][0] - posData[pD][0];
            double v3y = posData[pC][1] - posData[pD][1];
            double v3z = posData[pC][2] - posData[pD][2];

            // Cross products: cp1 = v1 x v2, cp2 = v2 x v3
            double cp1x = v1y * v2z - v1z * v2y;
            double cp1y = v1z * v2x - v1x * v2z;
            double cp1z = v1x * v2y - v1y * v2x;

            double cp2x = v2y * v3z - v2z * v3y;
            double cp2y = v2z * v3x - v2x * v3z;
            double cp2z = v2x * v3y - v2y * v3x;

            double normCross1 = cp1x * cp1x + cp1y * cp1y + cp1z * cp1z;
            double normCross2 = cp2x * cp2x + cp2y * cp2y + cp2z * cp2z;
            double normV2_2 = v2x * v2x + v2y * v2y + v2z * v2z;
            double normV2 = sqrt(normV2_2);

            if (normCross1 < 1.0e-12 || normCross2 < 1.0e-12 || normV2 < 1.0e-12)
                continue;

            // Dihedral angle: angle between cross product vectors, with sign
            double dotCross = cp1x * cp2x + cp1y * cp2y + cp1z * cp2z;
            double cosPhi = dotCross / sqrt(normCross1 * normCross2);
            if (cosPhi > 1.0) cosPhi = 1.0;
            if (cosPhi < -1.0) cosPhi = -1.0;

            // Sign from v1 . cp2 (matches OpenMM: deltaR[0] . crossProduct[1])
            double signPhi = v1x * cp2x + v1y * cp2y + v1z * cp2z;
            double phi = acos(cosPhi);
            if (signPhi < 0.0) phi = -phi;

            int n = torsions[t].periodicity;
            double phaseT = torsions[t].phase;
            double kT = torsions[t].k;

            double deltaAngle = n * phi - phaseT;
            double energy = kT * (1.0 + cos(deltaAngle)) * scale;
            double dEdAngle = -kT * n * sin(deltaAngle) * scale;

            if (includeEnergy)
                groupEnergy += energy;

            if (includeForces) {
                // Force computation from ReferenceProperDihedralBond
                double forceFactors0 = (-dEdAngle * normV2) / normCross1;
                double forceFactors3 = (dEdAngle * normV2) / normCross2;

                double dotV1_V2 = v1x * v2x + v1y * v2y + v1z * v2z;
                double dotV3_V2 = v3x * v2x + v3y * v2y + v3z * v2z;

                double forceFactors1 = dotV1_V2 / normV2_2;
                double forceFactors2 = dotV3_V2 / normV2_2;

                double fAx = forceFactors0 * cp1x;
                double fAy = forceFactors0 * cp1y;
                double fAz = forceFactors0 * cp1z;

                double fDx = forceFactors3 * cp2x;
                double fDy = forceFactors3 * cp2y;
                double fDz = forceFactors3 * cp2z;

                double sBx = forceFactors1 * fAx - forceFactors2 * fDx;
                double sBy = forceFactors1 * fAy - forceFactors2 * fDy;
                double sBz = forceFactors1 * fAz - forceFactors2 * fDz;

                double fBx = fAx - sBx;
                double fBy = fAy - sBy;
                double fBz = fAz - sBz;

                double fCx = fDx + sBx;
                double fCy = fDy + sBy;
                double fCz = fDz + sBz;

                forceData[pA][0] += fAx;
                forceData[pA][1] += fAy;
                forceData[pA][2] += fAz;
                forceData[pB][0] -= fBx;
                forceData[pB][1] -= fBy;
                forceData[pB][2] -= fBz;
                forceData[pC][0] -= fCx;
                forceData[pC][1] -= fCy;
                forceData[pC][2] -= fCz;
                forceData[pD][0] += fDx;
                forceData[pD][1] += fDy;
                forceData[pD][2] += fDz;
            }
        }
    }

    if (includeEnergy)
        groupEnergies[g] = groupEnergy;
}

void ReferenceCalcIsolatedBondedForceKernel::runGroups(
        ContextImpl& context, vector<Vec3>& posData, vector<Vec3>& forceData,
        bool includeForces, bool includeEnergy) {
    for (int g = 0; g < numParticleGroups; g++)
        computeGroup(g, posData, forceData, includeForces, includeEnergy);
}

double ReferenceCalcIsolatedBondedForceKernel::execute(
        ContextImpl& context, bool includeForces, bool includeEnergy) {

    vector<Vec3>& posData = refExtractPositions(context);
    vector<Vec3>& forceData = refExtractForces(context);

    fill(groupEnergies.begin(), groupEnergies.end(), 0.0);

    runGroups(context, posData, forceData, includeForces, includeEnergy);

    // Deterministic, group-ordered reduction (matches serial Reference exactly).
    double totalEnergy = 0.0;
    if (includeEnergy)
        for (int g = 0; g < numParticleGroups; g++)
            totalEnergy += groupEnergies[g];
    return totalEnergy;
}

// ==================== getGroupEnergy ====================

double ReferenceCalcIsolatedBondedForceKernel::getGroupEnergy(int groupIndex) const {
    if (groupIndex < 0 || groupIndex >= numParticleGroups)
        throw OpenMMException("IsolatedBondedForce: group index out of range");
    return groupEnergies[groupIndex];
}

// ==================== copyParametersToContext ====================

void ReferenceCalcIsolatedBondedForceKernel::copyParametersToContext(
        ContextImpl& context, const IsolatedBondedForce& force) {

    if (numAtoms != force.getNumAtoms())
        throw OpenMMException("Cannot update IsolatedBondedForce: number of atoms has changed");

    // Update bond parameters
    int numBonds = force.getNumBonds();
    bonds.resize(numBonds);
    for (int i = 0; i < numBonds; i++) {
        force.getBondParameters(i, bonds[i].atom1, bonds[i].atom2,
                               bonds[i].length, bonds[i].k);
    }

    // Update angle parameters
    int numAngles = force.getNumAngles();
    angles.resize(numAngles);
    for (int i = 0; i < numAngles; i++) {
        force.getAngleParameters(i, angles[i].atom1, angles[i].atom2, angles[i].atom3,
                                angles[i].angle, angles[i].k);
    }

    // Update torsion parameters
    int numTorsions = force.getNumTorsions();
    torsions.resize(numTorsions);
    for (int i = 0; i < numTorsions; i++) {
        force.getTorsionParameters(i, torsions[i].atom1, torsions[i].atom2,
                                  torsions[i].atom3, torsions[i].atom4,
                                  torsions[i].periodicity, torsions[i].phase,
                                  torsions[i].k);
    }

    // Update scaling factors
    globalScalingFactor = force.getGlobalScalingFactor();
    int nGroups = force.getNumParticleGroups();
    for (int g = 0; g < nGroups; g++) {
        groupScalingFactors[g] = force.getGroupScalingFactor(g);
    }
}

// ==================== computeHessian ====================

vector<double> ReferenceCalcIsolatedBondedForceKernel::computeHessian(
        ContextImpl& context, int groupIndex) {

    if (groupIndex < 0 || groupIndex >= numParticleGroups)
        throw OpenMMException("IsolatedBondedForce: group index out of range for computeHessian");

    vector<Vec3>& posData = refExtractPositions(context);
    const vector<int>& particles = groupParticleIndices[groupIndex];

    int N3 = 3 * numAtoms;
    vector<double> H(N3 * N3, 0.0);

    // Bond Hessians
    for (int b = 0; b < (int)bonds.size(); b++) {
        int i = bonds[b].atom1;
        int j = bonds[b].atom2;
        double Hii[9], Hij[9];
        BondedHessianAnalytical::computeBondHessianBlock(
            posData[particles[i]], posData[particles[j]],
            bonds[b].k, bonds[b].length, Hii, Hij);
        BondedHessianAnalytical::addBlock(H, N3, i, i, Hii);
        BondedHessianAnalytical::addBlock(H, N3, j, j, Hii);
        BondedHessianAnalytical::addBlock(H, N3, i, j, Hij);
        BondedHessianAnalytical::addBlock(H, N3, j, i, Hij);
    }

    // Angle Hessians
    for (int a = 0; a < (int)angles.size(); a++) {
        int i = angles[a].atom1;
        int j = angles[a].atom2;
        int k = angles[a].atom3;
        double Ha[9][9];
        BondedHessianAnalytical::computeAngleHessian(
            posData[particles[i]], posData[particles[j]], posData[particles[k]],
            angles[a].k, angles[a].angle, Ha);
        int atoms[3] = {i, j, k};
        for (int ai = 0; ai < 3; ai++) {
            for (int aj = 0; aj < 3; aj++) {
                for (int di = 0; di < 3; di++) {
                    for (int dj = 0; dj < 3; dj++) {
                        H[(3*atoms[ai] + di) * N3 + (3*atoms[aj] + dj)] += Ha[3*ai + di][3*aj + dj];
                    }
                }
            }
        }
    }

    // Torsion Hessians
    for (int t = 0; t < (int)torsions.size(); t++) {
        int i = torsions[t].atom1;
        int j = torsions[t].atom2;
        int k = torsions[t].atom3;
        int l = torsions[t].atom4;
        double Ht[12][12];
        BondedHessianAnalytical::computeTorsionHessian(
            posData[particles[i]], posData[particles[j]],
            posData[particles[k]], posData[particles[l]],
            torsions[t].periodicity, torsions[t].phase, torsions[t].k, Ht);
        int atoms[4] = {i, j, k, l};
        for (int ai = 0; ai < 4; ai++) {
            for (int aj = 0; aj < 4; aj++) {
                for (int di = 0; di < 3; di++) {
                    for (int dj = 0; dj < 3; dj++) {
                        H[(3*atoms[ai] + di) * N3 + (3*atoms[aj] + dj)] += Ht[3*ai + di][3*aj + dj];
                    }
                }
            }
        }
    }

    // Symmetrize
    for (int i = 0; i < N3; i++) {
        for (int j = i + 1; j < N3; j++) {
            double avg = 0.5 * (H[i * N3 + j] + H[j * N3 + i]);
            H[i * N3 + j] = avg;
            H[j * N3 + i] = avg;
        }
    }

    return H;
}

// ==================== computeInternalForceConstants ====================

vector<double> ReferenceCalcIsolatedBondedForceKernel::computeInternalForceConstants(
        ContextImpl& context, int groupIndex) {

    if (groupIndex < 0 || groupIndex >= numParticleGroups)
        throw OpenMMException("IsolatedBondedForce: group index out of range for computeInternalForceConstants");

    vector<Vec3>& posData = refExtractPositions(context);
    const vector<int>& particles = groupParticleIndices[groupIndex];

    int total = (int)bonds.size() + (int)angles.size() + (int)torsions.size();
    vector<double> constants(total);
    int idx = 0;

    // Bonds: d2E/dr2 = k
    for (int b = 0; b < (int)bonds.size(); b++) {
        constants[idx++] = bonds[b].k;
    }

    // Angles: d2E/dtheta2 = k
    for (int a = 0; a < (int)angles.size(); a++) {
        constants[idx++] = angles[a].k;
    }

    // Torsions: d2E/dphi2 = -k * n^2 * cos(n*phi - phi0)
    for (int t = 0; t < (int)torsions.size(); t++) {
        int pI = particles[torsions[t].atom1];
        int pJ = particles[torsions[t].atom2];
        int pK = particles[torsions[t].atom3];
        int pL = particles[torsions[t].atom4];
        double phi = BondedHessianAnalytical::computeDihedralAngle(
            posData[pI], posData[pJ], posData[pK], posData[pL]);
        int n = torsions[t].periodicity;
        constants[idx++] = -torsions[t].k * n * n * cos(n * phi - torsions[t].phase);
    }

    return constants;
}

}  // namespace GridForcePlugin
