/* -------------------------------------------------------------------------- *
 *                              OpenMMGridForce                               *
 * -------------------------------------------------------------------------- */

#include "BATTopology.h"

#include <algorithm>
#include <set>
#include <sstream>
#include <stdexcept>

using namespace GridForcePlugin;

BATTopology::BATTopology() {}

void BATTopology::setTopology(int nAtoms,
                              const std::vector<int>& root,
                              const std::vector<int>& torsions,
                              const std::vector<int>& perturbableMask,
                              const std::vector<int>& primaryTorsionIdx) {
    if (nAtoms < 3)
        throw std::invalid_argument("BATTopology: need at least 3 atoms");
    if ((int)root.size() != 3)
        throw std::invalid_argument("BATTopology: root must have length 3");
    int expected_torsions = 4 * (nAtoms - 3);
    if ((int)torsions.size() != expected_torsions) {
        std::ostringstream s;
        s << "BATTopology: torsions has length " << torsions.size()
          << ", expected " << expected_torsions << " (= 4 * (N - 3))";
        throw std::invalid_argument(s.str());
    }
    if ((int)perturbableMask.size() != 3 * nAtoms) {
        std::ostringstream s;
        s << "BATTopology: perturbableMask has length "
          << perturbableMask.size() << ", expected " << 3 * nAtoms;
        throw std::invalid_argument(s.str());
    }
    if (!primaryTorsionIdx.empty()
        && (int)primaryTorsionIdx.size() != nAtoms - 3) {
        std::ostringstream s;
        s << "BATTopology: primaryTorsionIdx has length "
          << primaryTorsionIdx.size() << ", expected " << (nAtoms - 3);
        throw std::invalid_argument(s.str());
    }
    n_atoms_ = nAtoms;
    root_ = root;
    torsions_ = torsions;
    perturbable_ = perturbableMask;
    primary_ = primaryTorsionIdx;
    validate();
}

void BATTopology::validate() const {
    if (n_atoms_ < 3)
        throw std::invalid_argument("BATTopology: not initialized");
    // Root atoms distinct and in range
    std::set<int> root_set(root_.begin(), root_.end());
    if ((int)root_set.size() != 3)
        throw std::invalid_argument("BATTopology: root atoms not distinct");
    for (int a : root_)
        if (a < 0 || a >= n_atoms_) {
            std::ostringstream s;
            s << "BATTopology: root atom " << a << " out of range [0, "
              << n_atoms_ << ")";
            throw std::invalid_argument(s.str());
        }
    // Tree consistency: a0s unique and reference only earlier-placed atoms
    std::set<int> placed(root_.begin(), root_.end());
    int n_t = (int)torsions_.size() / 4;
    for (int i = 0; i < n_t; ++i) {
        int a0 = torsions_[4*i + 0];
        int a1 = torsions_[4*i + 1];
        int a2 = torsions_[4*i + 2];
        int a3 = torsions_[4*i + 3];
        for (int a : {a0, a1, a2, a3})
            if (a < 0 || a >= n_atoms_) {
                std::ostringstream s;
                s << "BATTopology: torsion " << i << " has atom " << a
                  << " out of range";
                throw std::invalid_argument(s.str());
            }
        if (placed.count(a0)) {
            std::ostringstream s;
            s << "BATTopology: torsion " << i << " places atom " << a0
              << " which was already placed";
            throw std::invalid_argument(s.str());
        }
        for (int a : {a1, a2, a3})
            if (!placed.count(a)) {
                std::ostringstream s;
                s << "BATTopology: torsion " << i << " references atom "
                  << a << " which has not been placed yet";
                throw std::invalid_argument(s.str());
            }
        placed.insert(a0);
    }
    if ((int)placed.size() != n_atoms_) {
        std::ostringstream s;
        s << "BATTopology: only " << placed.size() << " of " << n_atoms_
          << " atoms are placed by the topology";
        throw std::invalid_argument(s.str());
    }
}

unsigned long long BATTopology::getHash() const {
    // FNV-1a 64-bit over (n_atoms, root, torsions, perturbable).
    // Cheap, deterministic, collision-resistant enough for a sanity hash.
    const unsigned long long FNV_OFFSET = 14695981039346656037ULL;
    const unsigned long long FNV_PRIME  = 1099511628211ULL;
    unsigned long long h = FNV_OFFSET;
    auto absorb = [&](long long v) {
        unsigned long long u = (unsigned long long)v;
        for (int b = 0; b < 8; ++b) {
            unsigned char byte = (unsigned char)(u >> (8*b));
            h ^= byte;
            h *= FNV_PRIME;
        }
    };
    absorb(n_atoms_);
    for (int v : root_) absorb(v);
    for (int v : torsions_) absorb(v);
    for (int v : perturbable_) absorb(v);
    for (int v : primary_) absorb(v);
    return h;
}
