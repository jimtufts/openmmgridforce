/* -------------------------------------------------------------------------- *
 *                              OpenMMGridForce                               *
 * -------------------------------------------------------------------------- */

#include "IsolatedSiteForce.h"
#include "internal/IsolatedSiteForceImpl.h"
#include "openmm/OpenMMException.h"

using namespace GridForcePlugin;
using namespace OpenMM;
using namespace std;

IsolatedSiteForce::IsolatedSiteForce()
    : m_numAtoms(0), m_centerX(0), m_centerY(0), m_centerZ(0),
      m_maxRadius(0), m_forceConstant(10000.0),
      m_globalScalingFactor(1.0) {
}

int IsolatedSiteForce::getNumAtoms() const {
    return m_numAtoms;
}

void IsolatedSiteForce::setNumAtoms(int numAtoms) {
    if (numAtoms < 0)
        throw OpenMMException("IsolatedSiteForce: Number of atoms must be non-negative");
    m_numAtoms = numAtoms;
}

// ========== Site Parameters ==========

void IsolatedSiteForce::setSiteCenter(double x, double y, double z) {
    m_centerX = x;
    m_centerY = y;
    m_centerZ = z;
}

void IsolatedSiteForce::getSiteCenter(double& x, double& y, double& z) const {
    x = m_centerX;
    y = m_centerY;
    z = m_centerZ;
}

void IsolatedSiteForce::setMaxRadius(double maxR) {
    if (maxR < 0)
        throw OpenMMException("IsolatedSiteForce: maxRadius must be non-negative");
    m_maxRadius = maxR;
}

double IsolatedSiteForce::getMaxRadius() const {
    return m_maxRadius;
}

void IsolatedSiteForce::setForceConstant(double k) {
    m_forceConstant = k;
}

double IsolatedSiteForce::getForceConstant() const {
    return m_forceConstant;
}

// ========== Atom Masses ==========

void IsolatedSiteForce::setAtomMasses(const vector<double>& masses) {
    if (m_numAtoms > 0 && static_cast<int>(masses.size()) != m_numAtoms)
        throw OpenMMException("IsolatedSiteForce: masses size must match numAtoms");
    m_masses = masses;
}

const vector<double>& IsolatedSiteForce::getAtomMasses() const {
    return m_masses;
}

// ========== Scaling ==========

double IsolatedSiteForce::getGroupScalingFactor(int groupIndex) const {
    if (groupIndex < 0 || groupIndex >= static_cast<int>(m_groupScalingFactors.size()))
        throw OpenMMException("IsolatedSiteForce: group index out of range");
    return m_groupScalingFactors[groupIndex];
}

void IsolatedSiteForce::setGroupScalingFactor(int groupIndex, double factor) {
    if (groupIndex < 0 || groupIndex >= static_cast<int>(m_groupScalingFactors.size()))
        throw OpenMMException("IsolatedSiteForce: group index out of range");
    m_groupScalingFactors[groupIndex] = factor;
}

// ========== Particle Groups ==========

int IsolatedSiteForce::addParticleGroup(const string& name, const vector<int>& indices) {
    if (m_numAtoms > 0 && static_cast<int>(indices.size()) != m_numAtoms)
        throw OpenMMException("IsolatedSiteForce: particle group size must match template size");
    ParticleGroupInfo group;
    group.name = name;
    group.indices = indices;
    m_particleGroups.push_back(group);
    m_groupScalingFactors.push_back(1.0);
    return static_cast<int>(m_particleGroups.size()) - 1;
}

void IsolatedSiteForce::getParticleGroup(int index, string& name, vector<int>& indices) const {
    if (index < 0 || index >= static_cast<int>(m_particleGroups.size()))
        throw OpenMMException("IsolatedSiteForce: particle group index out of range");
    name = m_particleGroups[index].name;
    indices = m_particleGroups[index].indices;
}

// ========== Per-Group Energy ==========

double IsolatedSiteForce::getGroupEnergy(int groupIndex) const {
    if (groupIndex < 0 || groupIndex >= static_cast<int>(m_groupEnergies.size()))
        throw OpenMMException("IsolatedSiteForce: group energy not available (call getState first)");
    return m_groupEnergies[groupIndex];
}

vector<double> IsolatedSiteForce::getParticleGroupEnergies() const {
    return m_groupEnergies;
}

// ========== Context Interface ==========

void IsolatedSiteForce::updateParametersInContext(Context& context) {
    dynamic_cast<IsolatedSiteForceImpl&>(getImplInContext(context))
        .updateParametersInContext(getContextImpl(context));
}

ForceImpl* IsolatedSiteForce::createImpl() const {
    return new IsolatedSiteForceImpl(*this);
}
