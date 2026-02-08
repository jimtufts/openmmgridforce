#ifndef OPENMM_BONDEDHESSIAN_H_
#define OPENMM_BONDEDHESSIAN_H_

/* -------------------------------------------------------------------------- *
 *                              OpenMMGridForce                               *
 * -------------------------------------------------------------------------- *
 * Utility class for computing analytical Hessians of bonded forces.         *
 * -------------------------------------------------------------------------- */

#include <vector>
#include "internal/windowsExportGridForce.h"
#include "openmm/Context.h"
#include "openmm/System.h"

namespace GridForcePlugin {

/**
 * BondedHessian computes the Hessian (second derivative matrix) for bonded
 * interactions in an OpenMM System. It extracts parameters from HarmonicBondForce,
 * HarmonicAngleForce, and PeriodicTorsionForce, then computes the full 3N x 3N
 * Hessian matrix analytically.
 *
 * Usage:
 *   BondedHessian hessianCalc;
 *   hessianCalc.initialize(system, context);
 *   std::vector<double> H = hessianCalc.computeHessian(context);
 *   // H is a flattened 3N x 3N matrix in row-major order
 */
class OPENMM_EXPORT_GRIDFORCE BondedHessian {
public:
    /**
     * Create a BondedHessian calculator.
     */
    BondedHessian();

    ~BondedHessian();

    /**
     * Initialize the calculator by extracting bonded force parameters from the System.
     * This must be called before computeHessian().
     *
     * @param system   the System containing bonded forces
     * @param context  the Context (used to get positions)
     */
    void initialize(const OpenMM::System& system, OpenMM::Context& context);

    /**
     * Compute the full Hessian matrix for all bonded interactions.
     * The Hessian includes contributions from:
     * - HarmonicBondForce (if present)
     * - HarmonicAngleForce (if present)
     * - PeriodicTorsionForce (if present)
     *
     * @param context  the Context containing current positions
     * @return flattened 3N x 3N Hessian matrix in row-major order
     */
    std::vector<double> computeHessian(OpenMM::Context& context);

    /**
     * Get the number of bonds found in the System.
     */
    int getNumBonds() const;

    /**
     * Get the number of angles found in the System.
     */
    int getNumAngles() const;

    /**
     * Get the number of torsions found in the System.
     */
    int getNumTorsions() const;

    /**
     * Compute scalar force constants (d²E/dq²) for each internal coordinate
     * at the current geometry. Returns a flat vector in order:
     *   [bonds (numBonds), angles (numAngles), torsions (numTorsions)]
     *
     * For harmonic bonds: d²E/dr² = k
     * For harmonic angles: d²E/dθ² = k
     * For periodic torsions: d²E/dφ² = -k*n²*cos(n*φ - φ0)
     *
     * Multiple torsion terms on the same atom quartet are kept separate.
     * Units: kJ/(mol·nm²) for bonds, kJ/(mol·rad²) for angles/torsions.
     */
    std::vector<double> computeInternalForceConstants(OpenMM::Context& context);

    /**
     * Get atom indices for each internal coordinate, for computing effective
     * masses via the Wilson B-matrix. Returns a flat vector:
     *   [bond_0_i, bond_0_j, ..., angle_0_i, angle_0_j, angle_0_k, ...,
     *    torsion_0_i, torsion_0_j, torsion_0_k, torsion_0_l, ...]
     *
     * Total size: 2*numBonds + 3*numAngles + 4*numTorsions.
     */
    std::vector<int> getInternalCoordinateAtomIndices() const;

private:
    class Impl;
    Impl* impl;
    bool initialized;
};

}  // namespace GridForcePlugin

#endif /*OPENMM_BONDEDHESSIAN_H_*/
