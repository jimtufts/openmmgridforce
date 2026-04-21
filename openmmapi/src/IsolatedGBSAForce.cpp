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
constexpr double IsolatedGBSAForce::NO_LOCALITY_CUTOFF;

IsolatedGBSAForce::IsolatedGBSAForce()
    : numAtoms(0),
      gbMethod(OBC_II),
      soluteDielectric(DEFAULT_SOLUTE_DIELECTRIC),
      solventDielectric(DEFAULT_SOLVENT_DIELECTRIC),
      includeSurfaceArea(true),
      surfaceTension(DEFAULT_SA_SURFACE_TENSION),
      downloadBornRadiiEnabled(false),
      cutoffDistance(NO_CUTOFF),
      receptorLocalityCutoff(NO_LOCALITY_CUTOFF),
      receptorMode(NONE),
      interpolationMethod(0),
      computeCrossTermGrid(false),
      numReceptorAtoms(0),
      globalScalingFactor(1.0) {
}

void IsolatedGBSAForce::setCrossTermBinValues(const vector<double>& binValues) {
    if ((int)binValues.size() != numAtoms) {
        throw OpenMMException(
            "IsolatedGBSAForce: crossTermBinValues must have length numAtoms");
    }
    for (double v : binValues) {
        if (!(v > 0)) {
            throw OpenMMException(
                "IsolatedGBSAForce: cross-term bin values must be positive");
        }
    }
    crossTermBinValues = binValues;
}

void IsolatedGBSAForce::setReceptorBornRadiiBaseline(const vector<double>& r) {
    if ((int)r.size() != numReceptorAtoms) {
        throw OpenMMException(
            "IsolatedGBSAForce: receptorBornRadiiBaseline must have length "
            "numReceptorAtoms");
    }
    for (double v : r) {
        if (!(v > 0)) {
            throw OpenMMException(
                "IsolatedGBSAForce: baseline Born radii must be positive");
        }
    }
    receptorBornRadiiBaseline = r;
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

double IsolatedGBSAForce::getGroupScalingFactor(int groupIndex) const {
    if (groupIndex < 0 || groupIndex >= static_cast<int>(groupScalingFactors.size())) {
        throw OpenMMException("IsolatedGBSAForce: group index out of range");
    }
    return groupScalingFactors[groupIndex];
}

void IsolatedGBSAForce::setGroupScalingFactor(int groupIndex, double factor) {
    if (groupIndex < 0 || groupIndex >= static_cast<int>(groupScalingFactors.size())) {
        throw OpenMMException("IsolatedGBSAForce: group index out of range");
    }
    groupScalingFactors[groupIndex] = factor;
}

int IsolatedGBSAForce::addParticleGroup(const string& name, const vector<int>& indices) {
    if (numAtoms > 0 && indices.size() != static_cast<size_t>(numAtoms)) {
        throw OpenMMException("IsolatedGBSAForce: particle group size must match template size");
    }
    ParticleGroupInfo group;
    group.name = name;
    group.indices = indices;
    particleGroups.push_back(group);
    groupScalingFactors.push_back(1.0);
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

std::vector<double> IsolatedGBSAForce::getParticleGroupEnergies() const {
    return groupEnergies;
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

double IsolatedGBSAForce::getGroupLigandSurfaceArea(int groupIndex) const {
    if (groupIndex < 0 || groupIndex >= static_cast<int>(groupBornRadii.size())) {
        throw OpenMMException("IsolatedGBSAForce: group Born radii not available (call getState first)");
    }
    const vector<double>& bornRadii = groupBornRadii[groupIndex];

    // ACE formula: SA_i = surfaceTension * 4π * (R_i + probe)² * (R_i / R_born_i)^6
    const double probeRadius = 0.14;  // nm
    const double fourPi = 4.0 * 3.14159265358979323846;

    double totalSA = 0.0;
    for (int i = 0; i < numAtoms; i++) {
        double R_i = radii[i];
        double R_born = bornRadii[i];
        if (R_born > 0) {
            double r_ratio = R_i / R_born;
            double r_ratio_6 = r_ratio * r_ratio * r_ratio * r_ratio * r_ratio * r_ratio;
            double surface = fourPi * (R_i + probeRadius) * (R_i + probeRadius);
            totalSA += surfaceTension * surface * r_ratio_6;
        }
    }
    return totalSA;
}

vector<double> IsolatedGBSAForce::getGroupAtomSurfaceAreas(int groupIndex) const {
    if (groupIndex < 0 || groupIndex >= static_cast<int>(groupBornRadii.size())) {
        throw OpenMMException("IsolatedGBSAForce: group Born radii not available (call getState first)");
    }
    const vector<double>& bornRadii = groupBornRadii[groupIndex];

    const double probeRadius = 0.14;  // nm
    const double fourPi = 4.0 * 3.14159265358979323846;

    vector<double> atomSAs(numAtoms);
    for (int i = 0; i < numAtoms; i++) {
        double R_i = radii[i];
        double R_born = bornRadii[i];
        if (R_born > 0) {
            double r_ratio = R_i / R_born;
            double r_ratio_6 = r_ratio * r_ratio * r_ratio * r_ratio * r_ratio * r_ratio;
            double surface = fourPi * (R_i + probeRadius) * (R_i + probeRadius);
            atomSAs[i] = surfaceTension * surface * r_ratio_6;
        } else {
            atomSAs[i] = 0.0;
        }
    }
    return atomSAs;
}

vector<double> IsolatedGBSAForce::getReceptorBornRadii(int groupIndex) const {
    if (receptorMode != PAIRWISE) {
        throw OpenMMException("IsolatedGBSAForce: receptor Born radii only available in PAIRWISE mode");
    }
    if (groupIndex < 0 || groupIndex >= static_cast<int>(groupReceptorBornRadii.size())) {
        throw OpenMMException("IsolatedGBSAForce: receptor Born radii not available (call getState first)");
    }
    return groupReceptorBornRadii[groupIndex];
}

double IsolatedGBSAForce::getGroupReceptorSurfaceAreaChange(int groupIndex) const {
    if (receptorMode != PAIRWISE) {
        throw OpenMMException("IsolatedGBSAForce: receptor SA change only available in PAIRWISE mode");
    }
    if (groupIndex < 0 || groupIndex >= static_cast<int>(groupReceptorBornRadii.size())) {
        throw OpenMMException("IsolatedGBSAForce: receptor Born radii not available (call getState first)");
    }
    const vector<double>& recBornRadii = groupReceptorBornRadii[groupIndex];

    // ACE formula: SA_i = surfaceTension * 4π * (R_i + probe)² * (R_i / R_born_i)^6
    const double probeRadius = 0.14;  // nm
    const double fourPi = 4.0 * 3.14159265358979323846;

    // Compute SA with ligand-descreened Born radii
    double saWithLigand = 0.0;
    for (int i = 0; i < numReceptorAtoms; i++) {
        double R_i = receptorRadii[i];
        double R_born = recBornRadii[i];
        if (R_born > 0) {
            double r_ratio = R_i / R_born;
            double r_ratio_6 = r_ratio * r_ratio * r_ratio * r_ratio * r_ratio * r_ratio;
            double surface = fourPi * (R_i + probeRadius) * (R_i + probeRadius);
            saWithLigand += surfaceTension * surface * r_ratio_6;
        }
    }

    // Compute "vacuum" SA (approximation: Born radius = intrinsic radius when fully exposed)
    // In vacuum, HCT = 0 so Born radius would be the intrinsic radius
    double saVacuum = 0.0;
    for (int i = 0; i < numReceptorAtoms; i++) {
        double R_i = receptorRadii[i];
        // In vacuum, R_born = R_i, so r_ratio = 1, r_ratio^6 = 1
        double surface = fourPi * (R_i + probeRadius) * (R_i + probeRadius);
        saVacuum += surfaceTension * surface;
    }

    // Return the change (positive = increased SA when ligand present, which shouldn't happen
    // normally - ligand typically buries receptor surface, reducing SA)
    return saWithLigand - saVacuum;
}

vector<double> IsolatedGBSAForce::getParticleGroupUnscaledEnergies(Context& context) const {
    return dynamic_cast<IsolatedGBSAForceImpl&>(getImplInContext(context)).getParticleGroupUnscaledEnergies();
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
