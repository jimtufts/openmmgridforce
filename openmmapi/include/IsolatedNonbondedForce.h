#ifndef OPENMM_ISOLATEDNONBONDEDFORCE_H_
#define OPENMM_ISOLATEDNONBONDEDFORCE_H_

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
#include "openmm/Vec3.h"

using namespace OpenMM;

namespace GridForcePlugin {

/**
 * IsolatedNonbondedForce computes nonbonded interactions for multiple isolated ligands.
 * Each ligand is completely isolated - atoms in ligand i only interact with other atoms
 * in ligand i, never with atoms in ligand j. This enables efficient batched evaluation
 * of multiple ligand conformations on the GPU.
 *
 * All ligands share the same template (number of atoms, parameters, exclusions), but
 * can have different positions. This is ideal for:
 * - Molecular docking (evaluate many poses of the same ligand)
 * - Conformational sampling (evaluate many conformations)
 * - High-throughput screening with parameter sweeps
 *
 * The force uses a simple O(N²) all-pairs algorithm per ligand, which is efficient
 * for small molecules (~50 atoms) and eliminates the overhead of neighbor lists.
 */
class OPENMM_EXPORT_GRIDFORCE IsolatedNonbondedForce : public OpenMM::Force {
public:
    /**
     * Create an IsolatedNonbondedForce.
     */
    IsolatedNonbondedForce();

    /**
     * Get the number of atoms in the ligand template.
     */
    int getNumAtoms() const;

    /**
     * Set the number of atoms in the ligand template.
     * This must be called before setting atom parameters.
     *
     * @param numAtoms  number of atoms in each ligand
     */
    void setNumAtoms(int numAtoms);

    /**
     * Set which particle indices in the System this force applies to.
     * Must be called before adding force to the System.
     *
     * @param particles  vector of particle indices
     */
    void setParticles(const std::vector<int>& particles);

    /**
     * Get the particle indices this force applies to.
     */
    const std::vector<int>& getParticles() const;

    /**
     * Set nonbonded parameters for an atom in the ligand template.
     *
     * @param index    atom index (0 to numAtoms-1)
     * @param charge   partial charge in elementary charge units (e)
     * @param sigma    Lennard-Jones sigma parameter (nm)
     * @param epsilon  Lennard-Jones epsilon parameter (kJ/mol)
     */
    void setAtomParameters(int index, double charge, double sigma, double epsilon);

    /**
     * Get nonbonded parameters for an atom.
     *
     * @param index    atom index (0 to numAtoms-1)
     * @param charge   output: partial charge (e)
     * @param sigma    output: LJ sigma (nm)
     * @param epsilon  output: LJ epsilon (kJ/mol)
     */
    void getAtomParameters(int index, double& charge, double& sigma, double& epsilon) const;

    /**
     * Add an exclusion between two atoms.
     * Excluded pairs do not interact. This applies to all ligand instances.
     *
     * @param atom1  index of first atom
     * @param atom2  index of second atom
     */
    void addExclusion(int atom1, int atom2);

    /**
     * Get the number of exclusions.
     */
    int getNumExclusions() const;

    /**
     * Get an exclusion.
     *
     * @param index  exclusion index (0 to getNumExclusions()-1)
     * @param atom1  output: index of first atom
     * @param atom2  output: index of second atom
     */
    void getExclusion(int index, int& atom1, int& atom2) const;

    /**
     * Add an exception to the normal nonbonded interaction rules.
     * In many force fields, pairs of bonded atoms (1-2, 1-3, 1-4) have
     * modified interactions. Use addExclusion() for complete exclusions
     * (zero interaction). Use addException() for scaled interactions
     * (e.g., 1-4 interactions with custom parameters).
     *
     * @param atom1       index of first atom
     * @param atom2       index of second atom
     * @param chargeProd  product of partial charges (e^2)
     * @param sigma       Lennard-Jones sigma parameter (nm)
     * @param epsilon     Lennard-Jones epsilon parameter (kJ/mol)
     * @return the index of the exception that was added
     */
    int addException(int atom1, int atom2, double chargeProd, double sigma, double epsilon);

    /**
     * Get the number of exceptions.
     */
    int getNumExceptions() const;

    /**
     * Get the parameters for an exception.
     *
     * @param index       exception index (0 to getNumExceptions()-1)
     * @param atom1       output: index of first atom
     * @param atom2       output: index of second atom
     * @param chargeProd  output: product of partial charges (e^2)
     * @param sigma       output: LJ sigma (nm)
     * @param epsilon     output: LJ epsilon (kJ/mol)
     */
    void getExceptionParameters(int index, int& atom1, int& atom2, double& chargeProd,
                                 double& sigma, double& epsilon) const;

    /**
     * Update the parameters in a Context to match those stored in this Force object.
     * This method provides an efficient way to update certain parameters without
     * recreating the Context.
     *
     * @param context  the Context in which to update parameters
     */
    void updateParametersInContext(OpenMM::Context& context);

protected:
    OpenMM::ForceImpl* createImpl() const;

private:
    int m_numAtoms;
    std::vector<int> m_particles;
    std::vector<double> m_charges;
    std::vector<double> m_sigmas;
    std::vector<double> m_epsilons;
    std::vector<std::pair<int, int>> m_exclusions;

    // Exception parameters: atom indices and custom parameters
    std::vector<int> m_exceptionAtom1;
    std::vector<int> m_exceptionAtom2;
    std::vector<double> m_exceptionChargeProd;
    std::vector<double> m_exceptionSigma;
    std::vector<double> m_exceptionEpsilon;
};

}  // namespace GridForcePlugin

#endif /*OPENMM_ISOLATEDNONBONDEDFORCE_H_*/
