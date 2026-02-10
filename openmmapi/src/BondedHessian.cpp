/* -------------------------------------------------------------------------- *
 *                              OpenMMGridForce                               *
 * -------------------------------------------------------------------------- *
 * Utility class for computing analytical Hessians of bonded forces.         *
 * Uses shared analytical formulas from BondedHessianAnalytical.h.           *
 * -------------------------------------------------------------------------- */

#include "BondedHessian.h"
#include "internal/BondedHessianAnalytical.h"
#include "openmm/HarmonicBondForce.h"
#include "openmm/HarmonicAngleForce.h"
#include "openmm/PeriodicTorsionForce.h"
#include "openmm/Platform.h"
#include "openmm/OpenMMException.h"
#include "openmm/State.h"
#include <cmath>
#include <algorithm>

using namespace GridForcePlugin;
using namespace GridForcePlugin::BondedHessianAnalytical;
using namespace OpenMM;
using namespace std;

// Internal implementation class
class BondedHessian::Impl {
public:
    int numAtoms;
    int numBonds;
    int numAngles;
    int numTorsions;

    Impl() : numAtoms(0), numBonds(0), numAngles(0), numTorsions(0) {}

    vector<int> bondAtoms;
    vector<double> bondLengths;
    vector<double> bondKs;

    vector<int> angleAtoms;
    vector<double> angleValues;
    vector<double> angleKs;

    vector<int> torsionAtoms;
    vector<int> torsionPeriodicities;
    vector<double> torsionPhases;
    vector<double> torsionKs;

    string platformName;
};

BondedHessian::BondedHessian() : impl(new Impl()), initialized(false) {
}

BondedHessian::~BondedHessian() {
    delete impl;
}

void BondedHessian::initialize(const System& system, Context& context) {
    impl->numAtoms = system.getNumParticles();
    impl->platformName = context.getPlatform().getName();

    // Extract HarmonicBondForce parameters
    for (int i = 0; i < system.getNumForces(); i++) {
        const HarmonicBondForce* bondForce = dynamic_cast<const HarmonicBondForce*>(&system.getForce(i));
        if (bondForce != nullptr) {
            impl->numBonds = bondForce->getNumBonds();
            impl->bondAtoms.resize(2 * impl->numBonds);
            impl->bondLengths.resize(impl->numBonds);
            impl->bondKs.resize(impl->numBonds);

            for (int j = 0; j < impl->numBonds; j++) {
                int atom1, atom2;
                double length, k;
                bondForce->getBondParameters(j, atom1, atom2, length, k);
                impl->bondAtoms[2*j] = atom1;
                impl->bondAtoms[2*j + 1] = atom2;
                impl->bondLengths[j] = length;
                impl->bondKs[j] = k;
            }
            break;
        }
    }

    // Extract HarmonicAngleForce parameters
    for (int i = 0; i < system.getNumForces(); i++) {
        const HarmonicAngleForce* angleForce = dynamic_cast<const HarmonicAngleForce*>(&system.getForce(i));
        if (angleForce != nullptr) {
            impl->numAngles = angleForce->getNumAngles();
            impl->angleAtoms.resize(3 * impl->numAngles);
            impl->angleValues.resize(impl->numAngles);
            impl->angleKs.resize(impl->numAngles);

            for (int j = 0; j < impl->numAngles; j++) {
                int atom1, atom2, atom3;
                double angle, k;
                angleForce->getAngleParameters(j, atom1, atom2, atom3, angle, k);
                impl->angleAtoms[3*j] = atom1;
                impl->angleAtoms[3*j + 1] = atom2;
                impl->angleAtoms[3*j + 2] = atom3;
                impl->angleValues[j] = angle;
                impl->angleKs[j] = k;
            }
            break;
        }
    }

    // Extract PeriodicTorsionForce parameters
    for (int i = 0; i < system.getNumForces(); i++) {
        const PeriodicTorsionForce* torsionForce = dynamic_cast<const PeriodicTorsionForce*>(&system.getForce(i));
        if (torsionForce != nullptr) {
            impl->numTorsions = torsionForce->getNumTorsions();
            impl->torsionAtoms.resize(4 * impl->numTorsions);
            impl->torsionPeriodicities.resize(impl->numTorsions);
            impl->torsionPhases.resize(impl->numTorsions);
            impl->torsionKs.resize(impl->numTorsions);

            for (int j = 0; j < impl->numTorsions; j++) {
                int atom1, atom2, atom3, atom4, periodicity;
                double phase, k;
                torsionForce->getTorsionParameters(j, atom1, atom2, atom3, atom4, periodicity, phase, k);
                impl->torsionAtoms[4*j] = atom1;
                impl->torsionAtoms[4*j + 1] = atom2;
                impl->torsionAtoms[4*j + 2] = atom3;
                impl->torsionAtoms[4*j + 3] = atom4;
                impl->torsionPeriodicities[j] = periodicity;
                impl->torsionPhases[j] = phase;
                impl->torsionKs[j] = k;
            }
            break;
        }
    }

    initialized = true;
}

std::vector<double> BondedHessian::computeHessian(Context& context) {
    if (!initialized) {
        throw OpenMMException("BondedHessian: must call initialize() before computeHessian()");
    }

    int N3 = 3 * impl->numAtoms;
    vector<double> H(N3 * N3, 0.0);

    // Get positions from context
    State state = context.getState(State::Positions);
    vector<Vec3> positions = state.getPositions();

    // Compute bond Hessians
    for (int b = 0; b < impl->numBonds; b++) {
        int i = impl->bondAtoms[2*b];
        int j = impl->bondAtoms[2*b + 1];
        double Hii[9], Hij[9];
        computeBondHessianBlock(positions[i], positions[j], impl->bondKs[b], impl->bondLengths[b], Hii, Hij);
        addBlock(H, N3, i, i, Hii);
        addBlock(H, N3, j, j, Hii);
        addBlock(H, N3, i, j, Hij);
        addBlock(H, N3, j, i, Hij);
    }

    // Compute angle Hessians
    for (int a = 0; a < impl->numAngles; a++) {
        int i = impl->angleAtoms[3*a];
        int j = impl->angleAtoms[3*a + 1];
        int k_idx = impl->angleAtoms[3*a + 2];
        double Ha[9][9];
        computeAngleHessian(positions[i], positions[j], positions[k_idx], impl->angleKs[a], impl->angleValues[a], Ha);
        int atoms[3] = {i, j, k_idx};
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

    // Compute torsion Hessians
    for (int t = 0; t < impl->numTorsions; t++) {
        int i = impl->torsionAtoms[4*t];
        int j = impl->torsionAtoms[4*t + 1];
        int k_idx = impl->torsionAtoms[4*t + 2];
        int l = impl->torsionAtoms[4*t + 3];
        double Ht[12][12];
        computeTorsionHessian(positions[i], positions[j], positions[k_idx], positions[l],
                               impl->torsionPeriodicities[t], impl->torsionPhases[t], impl->torsionKs[t], Ht);
        int atoms[4] = {i, j, k_idx, l};
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

int BondedHessian::getNumBonds() const {
    return impl ? impl->numBonds : 0;
}

int BondedHessian::getNumAngles() const {
    return impl ? impl->numAngles : 0;
}

int BondedHessian::getNumTorsions() const {
    return impl ? impl->numTorsions : 0;
}

std::vector<double> BondedHessian::computeInternalForceConstants(Context& context) {
    if (!initialized)
        throw OpenMMException("BondedHessian: must call initialize() before computeInternalForceConstants()");

    State state = context.getState(State::Positions);
    vector<Vec3> positions = state.getPositions();

    int total = impl->numBonds + impl->numAngles + impl->numTorsions;
    vector<double> constants(total);
    int idx = 0;

    // Bonds: d2E/dr2 = k (always k for harmonic bond)
    for (int b = 0; b < impl->numBonds; b++) {
        constants[idx++] = impl->bondKs[b];
    }

    // Angles: d2E/dtheta2 = k (always k for harmonic angle)
    for (int a = 0; a < impl->numAngles; a++) {
        constants[idx++] = impl->angleKs[a];
    }

    // Torsions: d2E/dphi2 = -k * n^2 * cos(n*phi - phi0)
    for (int t = 0; t < impl->numTorsions; t++) {
        int i = impl->torsionAtoms[4*t];
        int j = impl->torsionAtoms[4*t + 1];
        int k = impl->torsionAtoms[4*t + 2];
        int l = impl->torsionAtoms[4*t + 3];
        int n = impl->torsionPeriodicities[t];
        double phi0 = impl->torsionPhases[t];
        double kk = impl->torsionKs[t];

        double phi = computeDihedralAngle(positions[i], positions[j], positions[k], positions[l]);
        constants[idx++] = -kk * n * n * cos(n * phi - phi0);
    }

    return constants;
}

std::vector<int> BondedHessian::getInternalCoordinateAtomIndices() const {
    if (!initialized)
        throw OpenMMException("BondedHessian: must call initialize() before getInternalCoordinateAtomIndices()");

    vector<int> indices;
    indices.reserve(2 * impl->numBonds + 3 * impl->numAngles + 4 * impl->numTorsions);

    for (int b = 0; b < impl->numBonds; b++) {
        indices.push_back(impl->bondAtoms[2*b]);
        indices.push_back(impl->bondAtoms[2*b + 1]);
    }

    for (int a = 0; a < impl->numAngles; a++) {
        indices.push_back(impl->angleAtoms[3*a]);
        indices.push_back(impl->angleAtoms[3*a + 1]);
        indices.push_back(impl->angleAtoms[3*a + 2]);
    }

    for (int t = 0; t < impl->numTorsions; t++) {
        indices.push_back(impl->torsionAtoms[4*t]);
        indices.push_back(impl->torsionAtoms[4*t + 1]);
        indices.push_back(impl->torsionAtoms[4*t + 2]);
        indices.push_back(impl->torsionAtoms[4*t + 3]);
    }

    return indices;
}
