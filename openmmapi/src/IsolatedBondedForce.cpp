/* -------------------------------------------------------------------------- *
 *                              OpenMMGridForce                               *
 * -------------------------------------------------------------------------- */

#include "IsolatedBondedForce.h"
#include "internal/IsolatedBondedForceImpl.h"
#include "openmm/OpenMMException.h"

using namespace GridForcePlugin;
using namespace OpenMM;
using namespace std;

IsolatedBondedForce::IsolatedBondedForce() : m_numAtoms(0), m_globalScalingFactor(1.0) {
}

int IsolatedBondedForce::getNumAtoms() const {
    return m_numAtoms;
}

void IsolatedBondedForce::setNumAtoms(int numAtoms) {
    if (numAtoms < 0) {
        throw OpenMMException("IsolatedBondedForce: Number of atoms must be non-negative");
    }
    m_numAtoms = numAtoms;
}

// ========== Bonds ==========

int IsolatedBondedForce::addBond(int atom1, int atom2, double length, double k) {
    if (m_numAtoms > 0 && (atom1 < 0 || atom1 >= m_numAtoms || atom2 < 0 || atom2 >= m_numAtoms)) {
        throw OpenMMException("IsolatedBondedForce: Bond atom indices out of range");
    }
    BondInfo bond;
    bond.atom1 = atom1;
    bond.atom2 = atom2;
    bond.length = length;
    bond.k = k;
    m_bonds.push_back(bond);
    return static_cast<int>(m_bonds.size()) - 1;
}

int IsolatedBondedForce::getNumBonds() const {
    return static_cast<int>(m_bonds.size());
}

void IsolatedBondedForce::getBondParameters(int index, int& atom1, int& atom2,
                                             double& length, double& k) const {
    if (index < 0 || index >= static_cast<int>(m_bonds.size())) {
        throw OpenMMException("IsolatedBondedForce: Bond index out of range");
    }
    atom1 = m_bonds[index].atom1;
    atom2 = m_bonds[index].atom2;
    length = m_bonds[index].length;
    k = m_bonds[index].k;
}

void IsolatedBondedForce::setBondParameters(int index, int atom1, int atom2,
                                             double length, double k) {
    if (index < 0 || index >= static_cast<int>(m_bonds.size())) {
        throw OpenMMException("IsolatedBondedForce: Bond index out of range");
    }
    m_bonds[index].atom1 = atom1;
    m_bonds[index].atom2 = atom2;
    m_bonds[index].length = length;
    m_bonds[index].k = k;
}

// ========== Angles ==========

int IsolatedBondedForce::addAngle(int atom1, int atom2, int atom3, double angle, double k) {
    if (m_numAtoms > 0 && (atom1 < 0 || atom1 >= m_numAtoms ||
        atom2 < 0 || atom2 >= m_numAtoms || atom3 < 0 || atom3 >= m_numAtoms)) {
        throw OpenMMException("IsolatedBondedForce: Angle atom indices out of range");
    }
    AngleInfo info;
    info.atom1 = atom1;
    info.atom2 = atom2;
    info.atom3 = atom3;
    info.angle = angle;
    info.k = k;
    m_angles.push_back(info);
    return static_cast<int>(m_angles.size()) - 1;
}

int IsolatedBondedForce::getNumAngles() const {
    return static_cast<int>(m_angles.size());
}

void IsolatedBondedForce::getAngleParameters(int index, int& atom1, int& atom2, int& atom3,
                                              double& angle, double& k) const {
    if (index < 0 || index >= static_cast<int>(m_angles.size())) {
        throw OpenMMException("IsolatedBondedForce: Angle index out of range");
    }
    atom1 = m_angles[index].atom1;
    atom2 = m_angles[index].atom2;
    atom3 = m_angles[index].atom3;
    angle = m_angles[index].angle;
    k = m_angles[index].k;
}

void IsolatedBondedForce::setAngleParameters(int index, int atom1, int atom2, int atom3,
                                              double angle, double k) {
    if (index < 0 || index >= static_cast<int>(m_angles.size())) {
        throw OpenMMException("IsolatedBondedForce: Angle index out of range");
    }
    m_angles[index].atom1 = atom1;
    m_angles[index].atom2 = atom2;
    m_angles[index].atom3 = atom3;
    m_angles[index].angle = angle;
    m_angles[index].k = k;
}

// ========== Torsions ==========

int IsolatedBondedForce::addTorsion(int atom1, int atom2, int atom3, int atom4,
                                     int periodicity, double phase, double k) {
    if (m_numAtoms > 0 && (atom1 < 0 || atom1 >= m_numAtoms || atom2 < 0 || atom2 >= m_numAtoms ||
        atom3 < 0 || atom3 >= m_numAtoms || atom4 < 0 || atom4 >= m_numAtoms)) {
        throw OpenMMException("IsolatedBondedForce: Torsion atom indices out of range");
    }
    TorsionInfo info;
    info.atom1 = atom1;
    info.atom2 = atom2;
    info.atom3 = atom3;
    info.atom4 = atom4;
    info.periodicity = periodicity;
    info.phase = phase;
    info.k = k;
    m_torsions.push_back(info);
    return static_cast<int>(m_torsions.size()) - 1;
}

int IsolatedBondedForce::getNumTorsions() const {
    return static_cast<int>(m_torsions.size());
}

void IsolatedBondedForce::getTorsionParameters(int index, int& atom1, int& atom2,
                                                int& atom3, int& atom4,
                                                int& periodicity, double& phase, double& k) const {
    if (index < 0 || index >= static_cast<int>(m_torsions.size())) {
        throw OpenMMException("IsolatedBondedForce: Torsion index out of range");
    }
    atom1 = m_torsions[index].atom1;
    atom2 = m_torsions[index].atom2;
    atom3 = m_torsions[index].atom3;
    atom4 = m_torsions[index].atom4;
    periodicity = m_torsions[index].periodicity;
    phase = m_torsions[index].phase;
    k = m_torsions[index].k;
}

void IsolatedBondedForce::setTorsionParameters(int index, int atom1, int atom2,
                                                int atom3, int atom4,
                                                int periodicity, double phase, double k) {
    if (index < 0 || index >= static_cast<int>(m_torsions.size())) {
        throw OpenMMException("IsolatedBondedForce: Torsion index out of range");
    }
    m_torsions[index].atom1 = atom1;
    m_torsions[index].atom2 = atom2;
    m_torsions[index].atom3 = atom3;
    m_torsions[index].atom4 = atom4;
    m_torsions[index].periodicity = periodicity;
    m_torsions[index].phase = phase;
    m_torsions[index].k = k;
}

// ========== Scaling ==========

double IsolatedBondedForce::getGroupScalingFactor(int groupIndex) const {
    if (groupIndex < 0 || groupIndex >= static_cast<int>(m_groupScalingFactors.size())) {
        throw OpenMMException("IsolatedBondedForce: group index out of range");
    }
    return m_groupScalingFactors[groupIndex];
}

void IsolatedBondedForce::setGroupScalingFactor(int groupIndex, double factor) {
    if (groupIndex < 0 || groupIndex >= static_cast<int>(m_groupScalingFactors.size())) {
        throw OpenMMException("IsolatedBondedForce: group index out of range");
    }
    m_groupScalingFactors[groupIndex] = factor;
}

// ========== Particle Groups ==========

int IsolatedBondedForce::addParticleGroup(const string& name, const vector<int>& indices) {
    if (m_numAtoms > 0 && static_cast<int>(indices.size()) != m_numAtoms) {
        throw OpenMMException("IsolatedBondedForce: particle group size must match template size");
    }
    ParticleGroupInfo group;
    group.name = name;
    group.indices = indices;
    m_particleGroups.push_back(group);
    m_groupScalingFactors.push_back(1.0);
    return static_cast<int>(m_particleGroups.size()) - 1;
}

void IsolatedBondedForce::getParticleGroup(int index, string& name, vector<int>& indices) const {
    if (index < 0 || index >= static_cast<int>(m_particleGroups.size())) {
        throw OpenMMException("IsolatedBondedForce: particle group index out of range");
    }
    name = m_particleGroups[index].name;
    indices = m_particleGroups[index].indices;
}

// ========== Per-Group Energy ==========

double IsolatedBondedForce::getGroupEnergy(int groupIndex) const {
    if (groupIndex < 0 || groupIndex >= static_cast<int>(m_groupEnergies.size())) {
        throw OpenMMException("IsolatedBondedForce: group energy not available (call getState first)");
    }
    return m_groupEnergies[groupIndex];
}

// ========== Context Interface ==========

void IsolatedBondedForce::updateParametersInContext(Context& context) {
    dynamic_cast<IsolatedBondedForceImpl&>(getImplInContext(context)).updateParametersInContext(getContextImpl(context));
}

// ========== Hessian ==========

vector<double> IsolatedBondedForce::computeHessian(Context& context, int groupIndex) {
    return dynamic_cast<IsolatedBondedForceImpl&>(getImplInContext(context)).computeHessian(getContextImpl(context), groupIndex);
}

vector<double> IsolatedBondedForce::computeInternalForceConstants(Context& context, int groupIndex) {
    return dynamic_cast<IsolatedBondedForceImpl&>(getImplInContext(context)).computeInternalForceConstants(getContextImpl(context), groupIndex);
}

vector<int> IsolatedBondedForce::getInternalCoordinateAtomIndices() const {
    vector<int> indices;
    indices.reserve(2 * m_bonds.size() + 3 * m_angles.size() + 4 * m_torsions.size());

    for (int b = 0; b < (int)m_bonds.size(); b++) {
        indices.push_back(m_bonds[b].atom1);
        indices.push_back(m_bonds[b].atom2);
    }
    for (int a = 0; a < (int)m_angles.size(); a++) {
        indices.push_back(m_angles[a].atom1);
        indices.push_back(m_angles[a].atom2);
        indices.push_back(m_angles[a].atom3);
    }
    for (int t = 0; t < (int)m_torsions.size(); t++) {
        indices.push_back(m_torsions[t].atom1);
        indices.push_back(m_torsions[t].atom2);
        indices.push_back(m_torsions[t].atom3);
        indices.push_back(m_torsions[t].atom4);
    }
    return indices;
}

ForceImpl* IsolatedBondedForce::createImpl() const {
    return new IsolatedBondedForceImpl(*this);
}
