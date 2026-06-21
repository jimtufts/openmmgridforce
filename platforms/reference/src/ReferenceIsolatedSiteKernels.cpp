/* -------------------------------------------------------------------------- *
 *                              OpenMMGridForce                               *
 * -------------------------------------------------------------------------- */

#include "ReferenceIsolatedSiteKernels.h"
#include "ReferenceGridInterpolation.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/OpenMMException.h"
#include "openmm/reference/ReferencePlatform.h"
#include <cmath>

using namespace OpenMM;
using namespace std;

namespace GridForcePlugin {

void ReferenceCalcIsolatedSiteForceKernel::initialize(
        const System& system, const IsolatedSiteForce& force) {

    numAtoms = force.getNumAtoms();
    numParticleGroups = force.getNumParticleGroups();

    // Site parameters
    force.getSiteCenter(centerX, centerY, centerZ);
    maxRadius = force.getMaxRadius();
    forceConstant = force.getForceConstant();
    globalScalingFactor = force.getGlobalScalingFactor();

    // Masses
    const vector<double>& m = force.getAtomMasses();
    masses = m;
    totalMass = 0.0;
    for (int i = 0; i < numAtoms; i++)
        totalMass += masses[i];

    // Particle group mappings
    groupParticleIndices.resize(numParticleGroups);
    groupScalingFactors.resize(numParticleGroups);
    for (int g = 0; g < numParticleGroups; g++) {
        string name;
        vector<int> indices;
        force.getParticleGroup(g, name, indices);
        groupParticleIndices[g] = indices;
        groupScalingFactors[g] = force.getGroupScalingFactor(g);
    }

    groupEnergies.resize(numParticleGroups, 0.0);
}

void ReferenceCalcIsolatedSiteForceKernel::computeGroup(
        int g, vector<Vec3>& posData, vector<Vec3>& forceData,
        bool includeForces, bool includeEnergy) {

    double scale = globalScalingFactor * groupScalingFactors[g];
    if (scale == 0.0) return;

    const vector<int>& particles = groupParticleIndices[g];

    // Compute mass-weighted COM
    double comX = 0, comY = 0, comZ = 0;
    for (int i = 0; i < numAtoms; i++) {
        int p = particles[i];
        comX += masses[i] * posData[p][0];
        comY += masses[i] * posData[p][1];
        comZ += masses[i] * posData[p][2];
    }
    comX /= totalMass;
    comY /= totalMass;
    comZ /= totalMass;

    // Distance from site center
    double dx = comX - centerX;
    double dy = comY - centerY;
    double dz = comZ - centerZ;
    double r = sqrt(dx * dx + dy * dy + dz * dz);

    // Flat-bottom: only restrain outside maxRadius
    double deltaR = r - maxRadius;
    if (deltaR <= 0.0) return;

    double energy = 0.5 * forceConstant * deltaR * deltaR * scale;

    if (includeEnergy) {
        groupEnergies[g] += energy;
    }

    if (includeForces && r > 1.0e-12) {
        // dE/dr = k * (r - maxR) * scale
        // F_i = -dE/dr * (COM - center) / r * m_i / M_total
        double dEdR = forceConstant * deltaR * scale;
        double prefactor = -dEdR / (r * totalMass);

        for (int i = 0; i < numAtoms; i++) {
            int p = particles[i];
            double w = prefactor * masses[i];
            forceData[p][0] += w * dx;
            forceData[p][1] += w * dy;
            forceData[p][2] += w * dz;
        }
    }
}

void ReferenceCalcIsolatedSiteForceKernel::runGroups(
        ContextImpl& context, vector<Vec3>& posData, vector<Vec3>& forceData,
        bool includeForces, bool includeEnergy) {
    for (int g = 0; g < numParticleGroups; g++)
        computeGroup(g, posData, forceData, includeForces, includeEnergy);
}

double ReferenceCalcIsolatedSiteForceKernel::execute(
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

void ReferenceCalcIsolatedSiteForceKernel::copyParametersToContext(
        ContextImpl& context, const IsolatedSiteForce& force) {

    force.getSiteCenter(centerX, centerY, centerZ);
    maxRadius = force.getMaxRadius();
    forceConstant = force.getForceConstant();
    globalScalingFactor = force.getGlobalScalingFactor();

    for (int g = 0; g < numParticleGroups; g++) {
        groupScalingFactors[g] = force.getGroupScalingFactor(g);
    }
}

double ReferenceCalcIsolatedSiteForceKernel::getGroupEnergy(int groupIndex) const {
    if (groupIndex < 0 || groupIndex >= numParticleGroups)
        throw OpenMMException("IsolatedSiteForce: group index out of range");
    return groupEnergies[groupIndex];
}

}  // namespace GridForcePlugin
