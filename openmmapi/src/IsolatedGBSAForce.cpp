/* -------------------------------------------------------------------------- *
 *                              OpenMMGridForce                               *
 * -------------------------------------------------------------------------- */

#include "IsolatedGBSAForce.h"
#include "internal/IsolatedGBSAForceImpl.h"
#include "openmm/OpenMMException.h"

using namespace GridForcePlugin;
using namespace OpenMM;
using namespace std;

// Define static constexpr members
constexpr double IsolatedGBSAForce::OBC_ALPHA;
constexpr double IsolatedGBSAForce::OBC_BETA;
constexpr double IsolatedGBSAForce::OBC_GAMMA;
constexpr double IsolatedGBSAForce::DIELECTRIC_OFFSET;
constexpr double IsolatedGBSAForce::DEFAULT_SOLUTE_DIELECTRIC;
constexpr double IsolatedGBSAForce::DEFAULT_SOLVENT_DIELECTRIC;
constexpr double IsolatedGBSAForce::DEFAULT_SA_SURFACE_TENSION;
constexpr double IsolatedGBSAForce::NO_CUTOFF;

IsolatedGBSAForce::IsolatedGBSAForce()
    : numAtoms(0),
      gbMethod(OBC_II),
      soluteDielectric(DEFAULT_SOLUTE_DIELECTRIC),
      solventDielectric(DEFAULT_SOLVENT_DIELECTRIC),
      includeSurfaceArea(false),
      surfaceTension(DEFAULT_SA_SURFACE_TENSION),
      cutoffDistance(NO_CUTOFF),
      receptorMode(NONE),
      interpolationMethod(0),
      numReceptorAtoms(0) {
}

void IsolatedGBSAForce::setNumAtoms(int n) {
    if (n < 0) {
        throw OpenMMException("IsolatedGBSAForce: numAtoms must be non-negative");
    }
    numAtoms = n;
    charges.resize(n, 0.0);
    radii.resize(n, 0.1);  // Default 1 Angstrom
    scaleFactors.resize(n, 0.8);  // Default OBC scale
}

void IsolatedGBSAForce::setParticles(const vector<int>& p) {
    particles = p;
}

void IsolatedGBSAForce::setAtomParameters(int index, double charge, double radius, double scaleFactor) {
    if (index < 0 || index >= numAtoms) {
        throw OpenMMException("IsolatedGBSAForce: atom index out of range");
    }
    if (radius <= 0) {
        throw OpenMMException("IsolatedGBSAForce: radius must be positive");
    }
    if (scaleFactor <= 0) {
        throw OpenMMException("IsolatedGBSAForce: scaleFactor must be positive");
    }
    charges[index] = charge;
    radii[index] = radius;
    scaleFactors[index] = scaleFactor;
}

void IsolatedGBSAForce::getAtomParameters(int index, double& charge, double& radius, double& scaleFactor) const {
    if (index < 0 || index >= numAtoms) {
        throw OpenMMException("IsolatedGBSAForce: atom index out of range");
    }
    charge = charges[index];
    radius = radii[index];
    scaleFactor = scaleFactors[index];
}

void IsolatedGBSAForce::setSoluteDielectric(double dielectric) {
    if (dielectric <= 0) {
        throw OpenMMException("IsolatedGBSAForce: dielectric must be positive");
    }
    soluteDielectric = dielectric;
}

void IsolatedGBSAForce::setSolventDielectric(double dielectric) {
    if (dielectric <= 0) {
        throw OpenMMException("IsolatedGBSAForce: dielectric must be positive");
    }
    solventDielectric = dielectric;
}

void IsolatedGBSAForce::setDesolvationGrid(shared_ptr<DesolvationGrid> grid) {
    desolvationGrid = grid;
}

void IsolatedGBSAForce::loadDesolvationGrid(const string& filename) {
    desolvationGrid = DesolvationGrid::loadFromFile(filename);
}

void IsolatedGBSAForce::setInterpolationMethod(int method) {
    if (method < 0 || method > 3) {
        throw OpenMMException("IsolatedGBSAForce: interpolationMethod must be 0 (trilinear), 1 (bspline), 2 (tricubic), or 3 (triquintic)");
    }
    interpolationMethod = method;
}

void IsolatedGBSAForce::setNumReceptorAtoms(int n) {
    if (n < 0) {
        throw OpenMMException("IsolatedGBSAForce: numReceptorAtoms must be non-negative");
    }
    numReceptorAtoms = n;
    receptorCharges.resize(n, 0.0);
    receptorRadii.resize(n, 0.1);
    receptorScaleFactors.resize(n, 0.8);
}

void IsolatedGBSAForce::setReceptorAtomParameters(int index, double charge, double radius, double scaleFactor) {
    if (index < 0 || index >= numReceptorAtoms) {
        throw OpenMMException("IsolatedGBSAForce: receptor atom index out of range");
    }
    if (radius <= 0) {
        throw OpenMMException("IsolatedGBSAForce: receptor radius must be positive");
    }
    if (scaleFactor <= 0) {
        throw OpenMMException("IsolatedGBSAForce: receptor scaleFactor must be positive");
    }
    receptorCharges[index] = charge;
    receptorRadii[index] = radius;
    receptorScaleFactors[index] = scaleFactor;
}

void IsolatedGBSAForce::getReceptorAtomParameters(int index, double& charge, double& radius, double& scaleFactor) const {
    if (index < 0 || index >= numReceptorAtoms) {
        throw OpenMMException("IsolatedGBSAForce: receptor atom index out of range");
    }
    charge = receptorCharges[index];
    radius = receptorRadii[index];
    scaleFactor = receptorScaleFactors[index];
}

void IsolatedGBSAForce::setReceptorPositions(const vector<double>& positions) {
    if (positions.size() != static_cast<size_t>(numReceptorAtoms * 3)) {
        throw OpenMMException("IsolatedGBSAForce: receptor positions size must be 3 * numReceptorAtoms");
    }
    receptorPositions = positions;
}

int IsolatedGBSAForce::addParticleGroup(const string& name, const vector<int>& indices) {
    if (numAtoms > 0 && indices.size() != static_cast<size_t>(numAtoms)) {
        throw OpenMMException("IsolatedGBSAForce: particle group size must match template size");
    }
    ParticleGroupInfo group;
    group.name = name;
    group.indices = indices;
    particleGroups.push_back(group);
    return static_cast<int>(particleGroups.size()) - 1;
}

void IsolatedGBSAForce::getParticleGroup(int index, string& name, vector<int>& indices) const {
    if (index < 0 || index >= static_cast<int>(particleGroups.size())) {
        throw OpenMMException("IsolatedGBSAForce: particle group index out of range");
    }
    name = particleGroups[index].name;
    indices = particleGroups[index].indices;
}

double IsolatedGBSAForce::getGroupEnergy(int groupIndex) const {
    if (groupIndex < 0 || groupIndex >= static_cast<int>(groupEnergies.size())) {
        throw OpenMMException("IsolatedGBSAForce: group energy not available (call getState first)");
    }
    return groupEnergies[groupIndex];
}

double IsolatedGBSAForce::getGroupLigandSelfEnergy(int groupIndex) const {
    if (groupIndex < 0 || groupIndex >= static_cast<int>(groupLigandSelfEnergies.size())) {
        throw OpenMMException("IsolatedGBSAForce: ligand self energy not available (call getState first)");
    }
    return groupLigandSelfEnergies[groupIndex];
}

double IsolatedGBSAForce::getGroupReceptorContribution(int groupIndex) const {
    if (groupIndex < 0 || groupIndex >= static_cast<int>(groupReceptorContributions.size())) {
        throw OpenMMException("IsolatedGBSAForce: receptor contribution not available (call getState first)");
    }
    return groupReceptorContributions[groupIndex];
}

double IsolatedGBSAForce::getGroupReceptorDesolvation(int groupIndex) const {
    if (groupIndex < 0 || groupIndex >= static_cast<int>(groupReceptorDesolvations.size())) {
        throw OpenMMException("IsolatedGBSAForce: receptor desolvation not available (call getState first)");
    }
    return groupReceptorDesolvations[groupIndex];
}

double IsolatedGBSAForce::getGroupCrossTermEnergy(int groupIndex) const {
    if (groupIndex < 0 || groupIndex >= static_cast<int>(groupCrossTermEnergies.size())) {
        throw OpenMMException("IsolatedGBSAForce: cross-term energy not available (call getState first)");
    }
    return groupCrossTermEnergies[groupIndex];
}

vector<double> IsolatedGBSAForce::getGroupBornRadii(int groupIndex) const {
    if (groupIndex < 0 || groupIndex >= static_cast<int>(groupBornRadii.size())) {
        throw OpenMMException("IsolatedGBSAForce: group Born radii not available (call getState first)");
    }
    return groupBornRadii[groupIndex];
}

vector<double> IsolatedGBSAForce::getGroupAtomEnergies(int groupIndex) const {
    if (groupIndex < 0 || groupIndex >= static_cast<int>(groupAtomEnergies.size())) {
        throw OpenMMException("IsolatedGBSAForce: per-atom energies not available (call getState first)");
    }
    return groupAtomEnergies[groupIndex];
}

void IsolatedGBSAForce::updateParametersInContext(Context& context) {
    dynamic_cast<IsolatedGBSAForceImpl&>(getImplInContext(context)).updateParametersInContext(getContextImpl(context));
}

vector<double> IsolatedGBSAForce::computeHessian(Context& context) {
    return dynamic_cast<IsolatedGBSAForceImpl&>(getImplInContext(context)).computeHessian(getContextImpl(context));
}

ForceImpl* IsolatedGBSAForce::createImpl() const {
    return new IsolatedGBSAForceImpl(*this);
}
