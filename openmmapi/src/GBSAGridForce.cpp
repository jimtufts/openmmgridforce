/* -------------------------------------------------------------------------- *
 *                              OpenMMGridForce                               *
 * -------------------------------------------------------------------------- */

#include "GBSAGridForce.h"
#include "internal/GBSAGridForceImpl.h"
#include "openmm/OpenMMException.h"

using namespace GridForcePlugin;
using namespace OpenMM;
using namespace std;

// Define static constexpr members
constexpr double GBSAGridForce::OBC_ALPHA;
constexpr double GBSAGridForce::OBC_BETA;
constexpr double GBSAGridForce::OBC_GAMMA;
constexpr double GBSAGridForce::DIELECTRIC_OFFSET;
constexpr double GBSAGridForce::DEFAULT_SOLUTE_DIELECTRIC;
constexpr double GBSAGridForce::DEFAULT_SOLVENT_DIELECTRIC;
constexpr double GBSAGridForce::DEFAULT_SA_SURFACE_TENSION;

GBSAGridForce::GBSAGridForce()
    : numAtoms(0),
      soluteDielectric(DEFAULT_SOLUTE_DIELECTRIC),
      solventDielectric(DEFAULT_SOLVENT_DIELECTRIC),
      includeSurfaceArea(false),
      surfaceTension(DEFAULT_SA_SURFACE_TENSION),
      interpolationMethod(0) {
}

void GBSAGridForce::setNumAtoms(int n) {
    if (n < 0) {
        throw OpenMMException("GBSAGridForce: numAtoms must be non-negative");
    }
    numAtoms = n;
    charges.resize(n, 0.0);
    radii.resize(n, 0.1);  // Default 1 Angstrom
    scaleFactors.resize(n, 0.8);  // Default OBC scale
}

void GBSAGridForce::setParticles(const vector<int>& p) {
    particles = p;
}

void GBSAGridForce::setAtomParameters(int index, double charge, double radius, double scaleFactor) {
    if (index < 0 || index >= numAtoms) {
        throw OpenMMException("GBSAGridForce: atom index out of range");
    }
    if (radius <= 0) {
        throw OpenMMException("GBSAGridForce: radius must be positive");
    }
    if (scaleFactor <= 0) {
        throw OpenMMException("GBSAGridForce: scaleFactor must be positive");
    }
    charges[index] = charge;
    radii[index] = radius;
    scaleFactors[index] = scaleFactor;
}

void GBSAGridForce::getAtomParameters(int index, double& charge, double& radius, double& scaleFactor) const {
    if (index < 0 || index >= numAtoms) {
        throw OpenMMException("GBSAGridForce: atom index out of range");
    }
    charge = charges[index];
    radius = radii[index];
    scaleFactor = scaleFactors[index];
}

void GBSAGridForce::setDesolvationGrid(shared_ptr<DesolvationGrid> grid) {
    desolvationGrid = grid;
}

void GBSAGridForce::loadDesolvationGrid(const string& filename) {
    desolvationGrid = DesolvationGrid::loadFromFile(filename);
}

void GBSAGridForce::addExclusion(int atom1, int atom2) {
    if (atom1 < 0 || atom1 >= numAtoms || atom2 < 0 || atom2 >= numAtoms) {
        throw OpenMMException("GBSAGridForce: exclusion atom index out of range");
    }
    if (atom1 == atom2) {
        throw OpenMMException("GBSAGridForce: cannot exclude an atom from itself");
    }
    exclusions.push_back(make_pair(min(atom1, atom2), max(atom1, atom2)));
}

void GBSAGridForce::getExclusionParticles(int index, int& atom1, int& atom2) const {
    if (index < 0 || index >= static_cast<int>(exclusions.size())) {
        throw OpenMMException("GBSAGridForce: exclusion index out of range");
    }
    atom1 = exclusions[index].first;
    atom2 = exclusions[index].second;
}

int GBSAGridForce::addParticleGroup(const string& name, const vector<int>& particleIndices) {
    ParticleGroupInfo group;
    group.name = name;
    group.particleIndices = particleIndices;
    particleGroups.push_back(group);
    return static_cast<int>(particleGroups.size()) - 1;
}

void GBSAGridForce::getParticleGroup(int index, string& name, vector<int>& particleIndices) const {
    if (index < 0 || index >= static_cast<int>(particleGroups.size())) {
        throw OpenMMException("GBSAGridForce: particle group index out of range");
    }
    name = particleGroups[index].name;
    particleIndices = particleGroups[index].particleIndices;
}

void GBSAGridForce::setSoluteDielectric(double dielectric) {
    if (dielectric <= 0) {
        throw OpenMMException("GBSAGridForce: dielectric must be positive");
    }
    soluteDielectric = dielectric;
}

void GBSAGridForce::setSolventDielectric(double dielectric) {
    if (dielectric <= 0) {
        throw OpenMMException("GBSAGridForce: dielectric must be positive");
    }
    solventDielectric = dielectric;
}

void GBSAGridForce::setIncludeSurfaceArea(bool include) {
    includeSurfaceArea = include;
}

void GBSAGridForce::setSurfaceTension(double tension) {
    surfaceTension = tension;
}

void GBSAGridForce::setInterpolationMethod(int method) {
    if (method < 0 || method > 3) {
        throw OpenMMException("GBSAGridForce: interpolationMethod must be 0 (trilinear), 1 (bspline), 2 (tricubic), or 3 (triquintic)");
    }
    interpolationMethod = method;
}

double GBSAGridForce::getGroupEnergy(int groupIndex) const {
    if (groupIndex < 0 || groupIndex >= static_cast<int>(groupEnergies.size())) {
        throw OpenMMException("GBSAGridForce: group energy not available (call getState first)");
    }
    return groupEnergies[groupIndex];
}

vector<double> GBSAGridForce::getGroupBornRadii(int groupIndex) const {
    if (groupIndex < 0 || groupIndex >= static_cast<int>(groupBornRadii.size())) {
        throw OpenMMException("GBSAGridForce: group Born radii not available (call getState first)");
    }
    return groupBornRadii[groupIndex];
}

ForceImpl* GBSAGridForce::createImpl() const {
    return new GBSAGridForceImpl(*this);
}
