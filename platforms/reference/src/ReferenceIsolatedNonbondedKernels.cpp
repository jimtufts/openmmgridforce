/* -------------------------------------------------------------------------- *
 *                              OpenMMGridForce                               *
 * -------------------------------------------------------------------------- *
 * Reference platform implementation of IsolatedNonbondedForce kernel.        *
 * Computes pairwise Coulomb + LJ for isolated ligand particle groups on CPU. *
 * -------------------------------------------------------------------------- */

#include "ReferenceIsolatedNonbondedKernels.h"
#include "ReferenceGridInterpolation.h"
#include "IsolatedNonbondedForce.h"

#include "openmm/OpenMMException.h"
#include "openmm/internal/ContextImpl.h"

#include <cmath>
#include <algorithm>

using namespace OpenMM;
using namespace std;

namespace GridForcePlugin {

// Coulomb constant in kJ*nm/(mol*e^2)
static const double COULOMB_CONST = 138.935456;

// ==================== Helper methods ====================

bool ReferenceCalcIsolatedNonbondedForceKernel::isExcluded(int i, int j) const {
    for (const auto& excl : exclusions) {
        if ((excl.first == i && excl.second == j) ||
            (excl.first == j && excl.second == i))
            return true;
    }
    return false;
}

int ReferenceCalcIsolatedNonbondedForceKernel::findException(int i, int j) const {
    for (int k = 0; k < (int)exceptions.size(); k++) {
        if ((exceptions[k].atom1 == i && exceptions[k].atom2 == j) ||
            (exceptions[k].atom1 == j && exceptions[k].atom2 == i))
            return k;
    }
    return -1;
}

// ==================== initialize ====================

void ReferenceCalcIsolatedNonbondedForceKernel::initialize(
        const System& system, const IsolatedNonbondedForce& force) {

    numAtoms = force.getNumAtoms();
    if (numAtoms == 0)
        throw OpenMMException("IsolatedNonbondedForce: Must set number of atoms before initialization");

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
                throw OpenMMException("IsolatedNonbondedForce: particle group " + name + " has wrong number of indices");
            groupParticleIndices[g] = indices;
        }
    } else {
        // Single-group mode: use setParticles() as implicit single group
        numParticleGroups = 1;
        groupParticleIndices.resize(1);
        groupParticleIndices[0] = force.getParticles();
        if ((int)groupParticleIndices[0].size() != numAtoms)
            throw OpenMMException("IsolatedNonbondedForce: Must call setParticles() with numAtoms indices or add particle groups");
    }

    // Alchemical scaling
    globalScalingFactor = force.getGlobalScalingFactor();
    groupScalingFactors.resize(numParticleGroups, 1.0);
    for (int g = 0; g < nGroups; g++) {
        groupScalingFactors[g] = force.getGroupScalingFactor(g);
    }

    // Extract atom parameters
    charges.resize(numAtoms);
    sigmas.resize(numAtoms);
    epsilons.resize(numAtoms);
    for (int i = 0; i < numAtoms; i++) {
        force.getAtomParameters(i, charges[i], sigmas[i], epsilons[i]);
    }

    // Extract exclusions
    int numExcl = force.getNumExclusions();
    exclusions.resize(numExcl);
    for (int i = 0; i < numExcl; i++) {
        int a1, a2;
        force.getExclusion(i, a1, a2);
        exclusions[i] = make_pair(a1, a2);
    }

    // Extract exceptions
    int numExcep = force.getNumExceptions();
    exceptions.resize(numExcep);
    for (int i = 0; i < numExcep; i++) {
        force.getExceptionParameters(i, exceptions[i].atom1, exceptions[i].atom2,
                                     exceptions[i].chargeProd, exceptions[i].sigma,
                                     exceptions[i].epsilon);
    }

    // Initialize per-group energy storage
    groupEnergies.resize(numParticleGroups, 0.0);
}

// ==================== execute ====================

double ReferenceCalcIsolatedNonbondedForceKernel::execute(
        ContextImpl& context, bool includeForces, bool includeEnergy) {

    vector<Vec3>& posData = refExtractPositions(context);
    vector<Vec3>& forceData = refExtractForces(context);

    double totalEnergy = 0.0;
    fill(groupEnergies.begin(), groupEnergies.end(), 0.0);

    for (int g = 0; g < numParticleGroups; g++) {
        double scale = globalScalingFactor * groupScalingFactors[g];
        if (scale == 0.0) continue;

        const vector<int>& particles = groupParticleIndices[g];

        for (int i = 0; i < numAtoms; i++) {
            for (int j = i + 1; j < numAtoms; j++) {
                // Check exclusion
                if (isExcluded(i, j)) continue;

                // Get parameters
                double qq, sig, eps;
                int excepIdx = findException(i, j);
                if (excepIdx >= 0) {
                    qq = exceptions[excepIdx].chargeProd;
                    sig = exceptions[excepIdx].sigma;
                    eps = exceptions[excepIdx].epsilon;
                } else {
                    qq = charges[i] * charges[j];
                    sig = (sigmas[i] + sigmas[j]) * 0.5;
                    eps = sqrt(epsilons[i] * epsilons[j]);
                }

                // Get system particle indices
                int particleI = particles[i];
                int particleJ = particles[j];

                // Compute distance
                double dx = posData[particleI][0] - posData[particleJ][0];
                double dy = posData[particleI][1] - posData[particleJ][1];
                double dz = posData[particleI][2] - posData[particleJ][2];
                double r2 = dx * dx + dy * dy + dz * dz;
                double r = sqrt(r2);
                double invR = 1.0 / r;
                double invR2 = invR * invR;

                // Coulomb energy: E_c = k * qq / r
                double coulombEnergy = COULOMB_CONST * qq * invR;

                // LJ energy: E_lj = 4*eps*((sig/r)^12 - (sig/r)^6)
                double sig_r = sig * invR;
                double sig_r2 = sig_r * sig_r;
                double sig_r6 = sig_r2 * sig_r2 * sig_r2;
                double sig_r12 = sig_r6 * sig_r6;
                double ljEnergy = 4.0 * eps * (sig_r12 - sig_r6);

                double pairEnergy = (coulombEnergy + ljEnergy) * scale;

                if (includeEnergy) {
                    totalEnergy += pairEnergy;
                    groupEnergies[g] += pairEnergy;
                }

                if (includeForces) {
                    // Force magnitude: F = -dE/dr * scale, but we want force vector
                    // dE_c/dr = -k*qq/r^2, dE_lj/dr = 4*eps*(-12*sig^12/r^13 + 6*sig^6/r^7)
                    // F_magnitude = (k*qq/r^2 + 4*eps*(12*sig_r12 - 6*sig_r6)/r) * scale
                    double coulombForce = coulombEnergy * invR;  // k*qq/r^2
                    double ljForce = 4.0 * eps * (12.0 * sig_r12 - 6.0 * sig_r6) * invR;
                    double forceMagnitude = (coulombForce + ljForce) * scale;

                    double fx = forceMagnitude * dx * invR;
                    double fy = forceMagnitude * dy * invR;
                    double fz = forceMagnitude * dz * invR;

                    // Newton's 3rd law
                    forceData[particleI][0] += fx;
                    forceData[particleI][1] += fy;
                    forceData[particleI][2] += fz;
                    forceData[particleJ][0] -= fx;
                    forceData[particleJ][1] -= fy;
                    forceData[particleJ][2] -= fz;
                }
            }
        }
    }

    return totalEnergy;
}

// ==================== computeHessian ====================

vector<double> ReferenceCalcIsolatedNonbondedForceKernel::computeHessian(ContextImpl& context) {
    vector<Vec3>& posData = refExtractPositions(context);

    int hessianSize = 3 * numAtoms;
    vector<double> hessian(hessianSize * hessianSize, 0.0);

    // Use first group's particle indices for Hessian (matches CUDA behavior)
    const vector<int>& particles = groupParticleIndices[0];

    for (int i = 0; i < numAtoms; i++) {
        for (int j = i + 1; j < numAtoms; j++) {
            if (isExcluded(i, j)) continue;

            // Get parameters
            double qq, sig, eps;
            int excepIdx = findException(i, j);
            if (excepIdx >= 0) {
                qq = exceptions[excepIdx].chargeProd;
                sig = exceptions[excepIdx].sigma;
                eps = exceptions[excepIdx].epsilon;
            } else {
                qq = charges[i] * charges[j];
                sig = (sigmas[i] + sigmas[j]) * 0.5;
                eps = sqrt(epsilons[i] * epsilons[j]);
            }

            int particleI = particles[i];
            int particleJ = particles[j];

            double dx = posData[particleI][0] - posData[particleJ][0];
            double dy = posData[particleI][1] - posData[particleJ][1];
            double dz = posData[particleI][2] - posData[particleJ][2];
            double r2 = dx * dx + dy * dy + dz * dz;
            double invR2 = 1.0 / r2;
            double invR = sqrt(invR2);
            double r = r2 * invR;

            // LJ derivatives
            double sig_r = sig * invR;
            double sig_r2 = sig_r * sig_r;
            double sig_r6 = sig_r2 * sig_r2 * sig_r2;
            double sig_r12 = sig_r6 * sig_r6;

            double dE_dr_LJ = 4.0 * eps * (-12.0 * sig_r12 + 6.0 * sig_r6) * invR;
            double d2E_dr2_LJ = 4.0 * eps * (156.0 * sig_r12 - 42.0 * sig_r6) * invR2;

            // Coulomb derivatives
            double dE_dr_C = -COULOMB_CONST * qq * invR2;
            double d2E_dr2_C = 2.0 * COULOMB_CONST * qq * invR2 * invR;

            double dE_dr = dE_dr_LJ + dE_dr_C;
            double d2E_dr2 = d2E_dr2_LJ + d2E_dr2_C;

            // Hessian block coefficients:
            // H_ab = term1 * n_a*n_b + term2 * delta_ab
            double term1 = d2E_dr2 - dE_dr * invR;
            double term2 = dE_dr * invR;

            double nx = dx * invR;
            double ny = dy * invR;
            double nz = dz * invR;

            // 3x3 Hessian block elements
            double Hxx = term1 * nx * nx + term2;
            double Hyy = term1 * ny * ny + term2;
            double Hzz = term1 * nz * nz + term2;
            double Hxy = term1 * nx * ny;
            double Hxz = term1 * nx * nz;
            double Hyz = term1 * ny * nz;

            // H_block[3x3] = {{Hxx, Hxy, Hxz}, {Hxy, Hyy, Hyz}, {Hxz, Hyz, Hzz}}
            double block[3][3] = {
                {Hxx, Hxy, Hxz},
                {Hxy, Hyy, Hyz},
                {Hxz, Hyz, Hzz}
            };

            // Accumulate into full Hessian matrix
            for (int di = 0; di < 3; di++) {
                for (int dj = 0; dj < 3; dj++) {
                    int ri = 3 * i + di;
                    int rj = 3 * i + dj;
                    int ci = 3 * j + di;
                    int cj = 3 * j + dj;

                    // H[i,i] += block
                    hessian[ri * hessianSize + rj] += block[di][dj];
                    // H[j,j] += block
                    hessian[ci * hessianSize + cj] += block[di][dj];
                    // H[i,j] -= block
                    hessian[ri * hessianSize + cj] -= block[di][dj];
                    // H[j,i] -= block
                    hessian[ci * hessianSize + rj] -= block[di][dj];
                }
            }
        }
    }

    return hessian;
}

// ==================== getGroupEnergy ====================

double ReferenceCalcIsolatedNonbondedForceKernel::getGroupEnergy(int groupIndex) const {
    if (groupIndex < 0 || groupIndex >= numParticleGroups)
        throw OpenMMException("IsolatedNonbondedForce: group index out of range");
    return groupEnergies[groupIndex];
}

// ==================== copyParametersToContext ====================

void ReferenceCalcIsolatedNonbondedForceKernel::copyParametersToContext(
        ContextImpl& context, const IsolatedNonbondedForce& force) {

    if (numAtoms != force.getNumAtoms())
        throw OpenMMException("Cannot update IsolatedNonbondedForce: number of atoms has changed");

    // Update atom parameters
    for (int i = 0; i < numAtoms; i++) {
        force.getAtomParameters(i, charges[i], sigmas[i], epsilons[i]);
    }

    // Update scaling factors
    globalScalingFactor = force.getGlobalScalingFactor();
    int nGroups = force.getNumParticleGroups();
    for (int g = 0; g < nGroups; g++) {
        groupScalingFactors[g] = force.getGroupScalingFactor(g);
    }
}

}  // namespace GridForcePlugin
