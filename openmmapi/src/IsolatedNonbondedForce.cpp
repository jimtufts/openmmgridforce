/* -------------------------------------------------------------------------- *
 *                              OpenMMGridForce                               *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2008-2012 Stanford University and the Authors.      *
 * Authors:                                                                   *
 * Contributors:                                                              *
 *                                                                            *
 * Permission is hereby granted, free of charge, to any person obtaining a    *
 * copy of this software and associated documentation files (the "Software"), *
 * to deal in the Software without restriction, including without limitation  *
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,   *
 * and/or sell copies of the Software, and to permit persons to whom the      *
 * Software is furnished to do so, subject to the following conditions:       *
 *                                                                            *
 * The above copyright notice and this permission notice shall be included in *
 * all copies or substantial portions of the Software.                        *
 *                                                                            *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR *
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,   *
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL    *
 * THE AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,    *
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR      *
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE  *
 * USE OR OTHER DEALINGS IN THE SOFTWARE.                                     *
 * -------------------------------------------------------------------------- */

#include "IsolatedNonbondedForce.h"
#include "internal/IsolatedNonbondedForceImpl.h"
#include "openmm/OpenMMException.h"

using namespace GridForcePlugin;
using namespace OpenMM;
using namespace std;

IsolatedNonbondedForce::IsolatedNonbondedForce() : m_numAtoms(0), m_globalScalingFactor(1.0) {
}

int IsolatedNonbondedForce::getNumAtoms() const {
    return m_numAtoms;
}

void IsolatedNonbondedForce::setNumAtoms(int numAtoms) {
    if (numAtoms < 0) {
        throw OpenMMException("IsolatedNonbondedForce: Number of atoms must be non-negative");
    }
    m_numAtoms = numAtoms;
    m_charges.resize(numAtoms, 0.0);
    m_sigmas.resize(numAtoms, 0.0);
    m_epsilons.resize(numAtoms, 0.0);
}

void IsolatedNonbondedForce::setAtomParameters(int index, double charge, double sigma, double epsilon) {
    if (index < 0 || index >= m_numAtoms) {
        throw OpenMMException("IsolatedNonbondedForce: Atom index out of range");
    }
    m_charges[index] = charge;
    m_sigmas[index] = sigma;
    m_epsilons[index] = epsilon;
}

void IsolatedNonbondedForce::getAtomParameters(int index, double& charge, double& sigma, double& epsilon) const {
    if (index < 0 || index >= m_numAtoms) {
        throw OpenMMException("IsolatedNonbondedForce: Atom index out of range");
    }
    charge = m_charges[index];
    sigma = m_sigmas[index];
    epsilon = m_epsilons[index];
}

void IsolatedNonbondedForce::addExclusion(int atom1, int atom2) {
    if (atom1 < 0 || atom1 >= m_numAtoms || atom2 < 0 || atom2 >= m_numAtoms) {
        throw OpenMMException("IsolatedNonbondedForce: Exclusion atom indices out of range");
    }
    // Store exclusions in canonical order (smaller index first)
    if (atom1 > atom2) {
        swap(atom1, atom2);
    }
    m_exclusions.push_back(make_pair(atom1, atom2));
}

int IsolatedNonbondedForce::getNumExclusions() const {
    return m_exclusions.size();
}

void IsolatedNonbondedForce::getExclusion(int index, int& atom1, int& atom2) const {
    if (index < 0 || index >= (int)m_exclusions.size()) {
        throw OpenMMException("IsolatedNonbondedForce: Exclusion index out of range");
    }
    atom1 = m_exclusions[index].first;
    atom2 = m_exclusions[index].second;
}

int IsolatedNonbondedForce::addException(int atom1, int atom2, double chargeProd, double sigma, double epsilon) {
    if (atom1 < 0 || atom1 >= m_numAtoms || atom2 < 0 || atom2 >= m_numAtoms) {
        throw OpenMMException("IsolatedNonbondedForce: Exception atom indices out of range");
    }
    // Store exceptions in canonical order (smaller index first)
    if (atom1 > atom2) {
        swap(atom1, atom2);
    }
    m_exceptionAtom1.push_back(atom1);
    m_exceptionAtom2.push_back(atom2);
    m_exceptionChargeProd.push_back(chargeProd);
    m_exceptionSigma.push_back(sigma);
    m_exceptionEpsilon.push_back(epsilon);
    return m_exceptionAtom1.size() - 1;
}

int IsolatedNonbondedForce::getNumExceptions() const {
    return m_exceptionAtom1.size();
}

void IsolatedNonbondedForce::getExceptionParameters(int index, int& atom1, int& atom2, double& chargeProd,
                                                     double& sigma, double& epsilon) const {
    if (index < 0 || index >= (int)m_exceptionAtom1.size()) {
        throw OpenMMException("IsolatedNonbondedForce: Exception index out of range");
    }
    atom1 = m_exceptionAtom1[index];
    atom2 = m_exceptionAtom2[index];
    chargeProd = m_exceptionChargeProd[index];
    sigma = m_exceptionSigma[index];
    epsilon = m_exceptionEpsilon[index];
}

void IsolatedNonbondedForce::setParticles(const std::vector<int>& particles) {
    if ((int)particles.size() != m_numAtoms) {
        throw OpenMMException("IsolatedNonbondedForce: Number of particles must match numAtoms");
    }
    m_particles = particles;
}

const std::vector<int>& IsolatedNonbondedForce::getParticles() const {
    return m_particles;
}

double IsolatedNonbondedForce::getGroupScalingFactor(int groupIndex) const {
    if (groupIndex < 0 || groupIndex >= static_cast<int>(m_groupScalingFactors.size())) {
        throw OpenMMException("IsolatedNonbondedForce: group index out of range");
    }
    return m_groupScalingFactors[groupIndex];
}

void IsolatedNonbondedForce::setGroupScalingFactor(int groupIndex, double factor) {
    if (groupIndex < 0 || groupIndex >= static_cast<int>(m_groupScalingFactors.size())) {
        throw OpenMMException("IsolatedNonbondedForce: group index out of range");
    }
    m_groupScalingFactors[groupIndex] = factor;
}

int IsolatedNonbondedForce::addParticleGroup(const string& name, const vector<int>& indices) {
    if (m_numAtoms > 0 && static_cast<int>(indices.size()) != m_numAtoms) {
        throw OpenMMException("IsolatedNonbondedForce: particle group size must match template size");
    }
    ParticleGroupInfo group;
    group.name = name;
    group.indices = indices;
    m_particleGroups.push_back(group);
    m_groupScalingFactors.push_back(1.0);
    return static_cast<int>(m_particleGroups.size()) - 1;
}

void IsolatedNonbondedForce::getParticleGroup(int index, string& name, vector<int>& indices) const {
    if (index < 0 || index >= static_cast<int>(m_particleGroups.size())) {
        throw OpenMMException("IsolatedNonbondedForce: particle group index out of range");
    }
    name = m_particleGroups[index].name;
    indices = m_particleGroups[index].indices;
}

double IsolatedNonbondedForce::getGroupEnergy(int groupIndex) const {
    if (groupIndex < 0 || groupIndex >= static_cast<int>(m_groupEnergies.size())) {
        throw OpenMMException("IsolatedNonbondedForce: group energy not available (call getState first)");
    }
    return m_groupEnergies[groupIndex];
}

std::vector<double> IsolatedNonbondedForce::getParticleGroupEnergies() const {
    return m_groupEnergies;
}

void IsolatedNonbondedForce::updateParametersInContext(Context& context) {
    dynamic_cast<IsolatedNonbondedForceImpl&>(getImplInContext(context)).updateParametersInContext(getContextImpl(context));
}

std::vector<double> IsolatedNonbondedForce::computeHessian(Context& context) {
    return dynamic_cast<IsolatedNonbondedForceImpl&>(getImplInContext(context)).computeHessian(getContextImpl(context));
}

ForceImpl* IsolatedNonbondedForce::createImpl() const {
    return new IsolatedNonbondedForceImpl(*this);
}
