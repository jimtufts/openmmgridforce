#ifndef OPENMM_BAT_TOPOLOGY_H_
#define OPENMM_BAT_TOPOLOGY_H_

/* -------------------------------------------------------------------------- *
 *                              OpenMMGridForce                               *
 * -------------------------------------------------------------------------- */

#include <vector>
#include <string>

#include "internal/windowsExportGridForce.h"

namespace GridForcePlugin {

/**
 * BAT (Bond/Angle/Torsion) coordinate topology for smart-darting.
 *
 * This is a passive data container: it stores the tree structure that
 * `cartesianToBAT` and `BATToCartesian` kernels walk to compute internal
 * coordinates. The topology is shared across all replicas and never
 * mutated after construction.
 *
 * Layout (matching MDAnalysis BAT):
 *   - Three "root" atom indices (initial, second, third) form the
 *     coordinate system for the external 6 DoFs.
 *   - For each non-root atom (n_atoms - 3 of them), a 4-tuple
 *     (a0, a1, a2, a3) describes how that atom is placed by NeRF:
 *     a0 is the new atom, a1 is its parent, a2 grandparent,
 *     a3 great-grandparent. The torsion list ordering must be such
 *     that (a1, a2, a3) are already placed when a0 is reached.
 *
 * BAT vector layout (length 3*N):
 *   [0:3]   root atom 0 position (translation)
 *   [3:6]   external rotation (phi, theta, omega)
 *   [6:9]   root-triangle internals (r01, r12, a012)
 *   [9 : 9+(N-3)]               bond lengths (one per torsion entry)
 *   [9+(N-3) : 9+2*(N-3)]       bond angles
 *   [9+2*(N-3) : 9+3*(N-3)]     torsion angles
 *
 * Build the topology in Python via AlGDock/mwe/bat_coords.BATTopology
 * and pass the resulting integer arrays here. The Python prototype is
 * the validation oracle (see tests/test_bat_roundtrip.py).
 */
class OPENMM_EXPORT_GRIDFORCE BATTopology {
public:
    /**
     * Construct an empty topology. Use `setTopology` before passing
     * to any kernel.
     */
    BATTopology();

    /**
     * Define the BAT topology.
     *
     * @param nAtoms             total atom count N
     * @param root               (initial, second, third) atom indices,
     *                           length 3
     * @param torsions           flat (a0, a1, a2, a3) atom indices,
     *                           length 4 * (N - 3); `torsions[4*i + j]`
     *                           is the j-th index of the i-th torsion
     * @param perturbableMask    length 3*N booleans (stored as int 0/1);
     *                           true = this BAT DoF participates in
     *                           smart-darting jumps; false = held fixed
     *                           across darts. Typical mask: 6 externals
     *                           + all torsions = perturbable; bonds and
     *                           angles fixed.
     * @param primaryTorsionIdx  length N-3; for each torsion `i`, the
     *                           index of the *first* torsion sharing the
     *                           same central bond (a1, a2). For primary
     *                           torsions this equals i. Used to convert
     *                           secondary (improper) torsions into
     *                           relative offsets, matching MDAnalysis.
     *                           Pass an empty vector to disable the
     *                           offset transform (raw dihedrals).
     * @throws std::invalid_argument if sizes are inconsistent or the
     *         tree-traversal order is invalid (a placement references
     *         an unplaced atom).
     */
    void setTopology(int nAtoms,
                     const std::vector<int>& root,
                     const std::vector<int>& torsions,
                     const std::vector<int>& perturbableMask,
                     const std::vector<int>& primaryTorsionIdx = {});

    int getNumAtoms() const { return n_atoms_; }
    int getNumTorsions() const { return n_atoms_ > 3 ? n_atoms_ - 3 : 0; }

    /** Root atom indices (length 3). */
    const std::vector<int>& getRoot() const { return root_; }

    /** Flat torsion 4-tuples (length 4*(N-3)). */
    const std::vector<int>& getTorsions() const { return torsions_; }

    /** Per-DoF perturbable mask (length 3*N). 1 = perturbable. */
    const std::vector<int>& getPerturbableMask() const { return perturbable_; }

    /** Primary-torsion indices (length N-3). Empty if not provided. */
    const std::vector<int>& getPrimaryTorsionIndices() const { return primary_; }

    /**
     * Returns a 64-bit hash that uniquely identifies this topology.
     * Use this to validate that a host topology and a Python-side
     * topology built independently are bit-identical.
     */
    unsigned long long getHash() const;

    /**
     * Sanity-check the tree:
     *   - Root atoms are distinct and in [0, N).
     *   - Each torsion's a0 is unique (no atom placed twice).
     *   - For each torsion (a0, a1, a2, a3), all of a1, a2, a3 are
     *     either in the root or appear as a0 in an EARLIER torsion.
     *   - Mask length is 3*N.
     * @throws std::invalid_argument on any failure.
     */
    void validate() const;

private:
    int n_atoms_ = 0;
    std::vector<int> root_;        // length 3
    std::vector<int> torsions_;    // length 4 * (N - 3), flat
    std::vector<int> perturbable_; // length 3 * N
    std::vector<int> primary_;     // length N-3 (or empty for no shift)
};

}  // namespace GridForcePlugin

#endif  // OPENMM_BAT_TOPOLOGY_H_
