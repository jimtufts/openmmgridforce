#ifndef OPENMM_ISOLATEDBONDEDFORCE_H_
#define OPENMM_ISOLATEDBONDEDFORCE_H_

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

#include <string>
#include <vector>

#include "internal/windowsExportGridForce.h"
#include "openmm/Context.h"
#include "openmm/Force.h"

using namespace OpenMM;

namespace GridForcePlugin {

/**
 * IsolatedBondedForce computes bonded interactions (bonds, angles, torsions) for
 * multiple isolated ligands. Each ligand is completely isolated - bonded terms in
 * ligand i only involve atoms in ligand i, never atoms in ligand j.
 *
 * All ligands share the same template (number of atoms, bonded parameters), but
 * can have different positions. This enables efficient batched evaluation of
 * multiple ligand conformations with per-group energy reporting.
 *
 * Supported bonded interactions:
 * - Harmonic bonds: E = 0.5 * k * (r - r0)^2
 * - Harmonic angles: E = 0.5 * k * (theta - theta0)^2
 * - Periodic torsions: E = k * (1 + cos(n*phi - phase))
 */
class OPENMM_EXPORT_GRIDFORCE IsolatedBondedForce : public OpenMM::Force {
public:
    IsolatedBondedForce();

    // ========== Template Size ==========

    int getNumAtoms() const;
    void setNumAtoms(int numAtoms);

    // ========== Bonds: E = 0.5 * k * (r - r0)^2 ==========

    /**
     * Add a bond to the template.
     *
     * @param atom1   first atom index (0 to numAtoms-1)
     * @param atom2   second atom index (0 to numAtoms-1)
     * @param length  equilibrium bond length (nm)
     * @param k       force constant (kJ/mol/nm^2)
     * @return the index of the bond that was added
     */
    int addBond(int atom1, int atom2, double length, double k);

    int getNumBonds() const;

    void getBondParameters(int index, int& atom1, int& atom2, double& length, double& k) const;

    void setBondParameters(int index, int atom1, int atom2, double length, double k);

    // ========== Angles: E = 0.5 * k * (theta - theta0)^2 ==========

    /**
     * Add an angle to the template.
     *
     * @param atom1  first atom index
     * @param atom2  central atom index
     * @param atom3  third atom index
     * @param angle  equilibrium angle (radians)
     * @param k      force constant (kJ/mol/rad^2)
     * @return the index of the angle that was added
     */
    int addAngle(int atom1, int atom2, int atom3, double angle, double k);

    int getNumAngles() const;

    void getAngleParameters(int index, int& atom1, int& atom2, int& atom3,
                            double& angle, double& k) const;

    void setAngleParameters(int index, int atom1, int atom2, int atom3, double angle, double k);

    // ========== Torsions: E = k * (1 + cos(n*phi - phase)) ==========

    /**
     * Add a periodic torsion to the template.
     * Multiple torsion terms on the same four atoms are stored as separate entries.
     *
     * @param atom1        first atom index
     * @param atom2        second atom index
     * @param atom3        third atom index
     * @param atom4        fourth atom index
     * @param periodicity  periodicity of the torsion (integer >= 1)
     * @param phase        phase offset (radians)
     * @param k            force constant (kJ/mol)
     * @return the index of the torsion that was added
     */
    int addTorsion(int atom1, int atom2, int atom3, int atom4,
                   int periodicity, double phase, double k);

    int getNumTorsions() const;

    void getTorsionParameters(int index, int& atom1, int& atom2, int& atom3, int& atom4,
                              int& periodicity, double& phase, double& k) const;

    void setTorsionParameters(int index, int atom1, int atom2, int atom3, int atom4,
                              int periodicity, double phase, double k);

    // ========== Alchemical Scaling ==========

    double getGlobalScalingFactor() const { return m_globalScalingFactor; }
    void setGlobalScalingFactor(double factor) { m_globalScalingFactor = factor; }

    double getGroupScalingFactor(int groupIndex) const;
    void setGroupScalingFactor(int groupIndex, double factor);

    // ========== Particle Groups ==========

    /**
     * Add a particle group (replica). Each group contains numAtoms particle indices
     * from the System, representing one copy of the ligand template.
     *
     * @param name     a name for this group
     * @param indices  particle indices in the System (must have numAtoms elements)
     * @return the index of the group that was added
     */
    int addParticleGroup(const std::string& name, const std::vector<int>& indices);

    int getNumParticleGroups() const { return static_cast<int>(m_particleGroups.size()); }

    void getParticleGroup(int index, std::string& name, std::vector<int>& indices) const;

    // ========== Per-Group Energy ==========

    double getGroupEnergy(int groupIndex) const;

    /**
     * Get energies for all particle groups in a single call.
     * Only available after calling getState() with energy.
     *
     * @return vector of energies (kJ/mol), one per group
     */
    std::vector<double> getParticleGroupEnergies() const;

    // ========== Parameter Updates ==========

    void updateParametersInContext(OpenMM::Context& context);

    // ========== Hessian ==========

    /**
     * Compute the Hessian (second derivative matrix) for the bonded interactions
     * using a specific particle group's positions.
     *
     * @param context     the Context containing the current positions
     * @param groupIndex  which particle group's positions to use (default 0)
     * @return flattened 3N x 3N Hessian matrix in row-major order (N = numAtoms template)
     *         Units: kJ/(mol*nm^2)
     */
    std::vector<double> computeHessian(OpenMM::Context& context, int groupIndex = 0);

    /**
     * Compute scalar force constants (d^2E/dq^2) for each internal coordinate
     * at the current geometry of the specified particle group.
     * Returns a flat vector in order: [bonds, angles, torsions].
     *
     * For harmonic bonds: d^2E/dr^2 = k
     * For harmonic angles: d^2E/dtheta^2 = k
     * For periodic torsions: d^2E/dphi^2 = -k*n^2*cos(n*phi - phi0)
     *
     * @param context     the Context containing the current positions
     * @param groupIndex  which particle group's positions to use (default 0)
     * @return force constants vector (size = numBonds + numAngles + numTorsions)
     */
    std::vector<double> computeInternalForceConstants(OpenMM::Context& context, int groupIndex = 0);

    /**
     * Get atom indices (template indices) for each internal coordinate.
     * Returns a flat vector: [bond_0_i, bond_0_j, ..., angle_0_i, ..., torsion_0_i, ...]
     */
    std::vector<int> getInternalCoordinateAtomIndices() const;

    // ========== OpenMM Force Interface ==========

    bool usesPeriodicBoundaryConditions() const override { return false; }

protected:
    OpenMM::ForceImpl* createImpl() const override;

private:
    int m_numAtoms;

    // Bond parameters
    struct BondInfo {
        int atom1, atom2;
        double length, k;
    };
    std::vector<BondInfo> m_bonds;

    // Angle parameters
    struct AngleInfo {
        int atom1, atom2, atom3;
        double angle, k;
    };
    std::vector<AngleInfo> m_angles;

    // Torsion parameters
    struct TorsionInfo {
        int atom1, atom2, atom3, atom4;
        int periodicity;
        double phase, k;
    };
    std::vector<TorsionInfo> m_torsions;

    // Alchemical scaling
    double m_globalScalingFactor;
    std::vector<double> m_groupScalingFactors;

    // Particle groups
    struct ParticleGroupInfo {
        std::string name;
        std::vector<int> indices;
    };
    std::vector<ParticleGroupInfo> m_particleGroups;

    // Per-group energy cache
    mutable std::vector<double> m_groupEnergies;

    friend class IsolatedBondedForceImpl;
};

}  // namespace GridForcePlugin

#endif /*OPENMM_ISOLATEDBONDEDFORCE_H_*/
