/* -------------------------------------------------------------------------- *
 *                              OpenMMGridForce                               *
 * -------------------------------------------------------------------------- *
 * Utility class for computing analytical Hessians of bonded forces.         *
 * -------------------------------------------------------------------------- */

#include "BondedHessian.h"
#include "openmm/HarmonicBondForce.h"
#include "openmm/HarmonicAngleForce.h"
#include "openmm/PeriodicTorsionForce.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/ContextImpl.h"
#include <cmath>

using namespace GridForcePlugin;
using namespace OpenMM;
using namespace std;

// Internal implementation class (platform-agnostic reference implementation)
class BondedHessian::Impl {
public:
    // Bond parameters: [atom1, atom2], length, k
    vector<int> bondAtoms;      // 2 * numBonds
    vector<double> bondLengths;
    vector<double> bondKs;

    // Angle parameters: [atom1, atom2, atom3], angle, k
    vector<int> angleAtoms;     // 3 * numAngles
    vector<double> angleValues;
    vector<double> angleKs;

    // Torsion parameters: [atom1, atom2, atom3, atom4], periodicity, phase, k
    vector<int> torsionAtoms;   // 4 * numTorsions
    vector<int> torsionPeriodicities;
    vector<double> torsionPhases;
    vector<double> torsionKs;
};

BondedHessian::BondedHessian() : impl(new Impl()), numAtoms(0), numBonds(0),
                                  numAngles(0), numTorsions(0), initialized(false) {
}

BondedHessian::~BondedHessian() {
    delete impl;
}

void BondedHessian::initialize(const System& system, Context& context) {
    numAtoms = system.getNumParticles();

    // Extract HarmonicBondForce parameters
    for (int i = 0; i < system.getNumForces(); i++) {
        const HarmonicBondForce* bondForce = dynamic_cast<const HarmonicBondForce*>(&system.getForce(i));
        if (bondForce != nullptr) {
            numBonds = bondForce->getNumBonds();
            impl->bondAtoms.resize(2 * numBonds);
            impl->bondLengths.resize(numBonds);
            impl->bondKs.resize(numBonds);

            for (int j = 0; j < numBonds; j++) {
                int atom1, atom2;
                double length, k;
                bondForce->getBondParameters(j, atom1, atom2, length, k);
                impl->bondAtoms[2*j] = atom1;
                impl->bondAtoms[2*j + 1] = atom2;
                impl->bondLengths[j] = length;
                impl->bondKs[j] = k;
            }
            break;  // Only one HarmonicBondForce expected
        }
    }

    // Extract HarmonicAngleForce parameters
    for (int i = 0; i < system.getNumForces(); i++) {
        const HarmonicAngleForce* angleForce = dynamic_cast<const HarmonicAngleForce*>(&system.getForce(i));
        if (angleForce != nullptr) {
            numAngles = angleForce->getNumAngles();
            impl->angleAtoms.resize(3 * numAngles);
            impl->angleValues.resize(numAngles);
            impl->angleKs.resize(numAngles);

            for (int j = 0; j < numAngles; j++) {
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
            numTorsions = torsionForce->getNumTorsions();
            impl->torsionAtoms.resize(4 * numTorsions);
            impl->torsionPeriodicities.resize(numTorsions);
            impl->torsionPhases.resize(numTorsions);
            impl->torsionKs.resize(numTorsions);

            for (int j = 0; j < numTorsions; j++) {
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

// Helper: add a 3x3 block to the Hessian matrix
static void addBlock(vector<double>& H, int N3, int i, int j, const double block[9]) {
    for (int di = 0; di < 3; di++) {
        for (int dj = 0; dj < 3; dj++) {
            H[(3*i + di) * N3 + (3*j + dj)] += block[di * 3 + dj];
        }
    }
}

// Compute bond Hessian block for atoms i,j
static void computeBondHessianBlock(const Vec3& ri, const Vec3& rj, double k, double r0,
                                     double Hii[9], double Hij[9]) {
    Vec3 rij = rj - ri;
    double r = sqrt(rij[0]*rij[0] + rij[1]*rij[1] + rij[2]*rij[2]);
    if (r < 1e-10) r = 1e-10;

    double invR = 1.0 / r;
    double invR2 = invR * invR;
    double dr = r - r0;

    // d²E/dri² = k * [(1 - r0/r) * I + r0/r³ * rij⊗rij]
    // where E = 0.5 * k * (r - r0)²
    double factor1 = k * (1.0 - r0 * invR);
    double factor2 = k * r0 * invR * invR2;

    for (int a = 0; a < 3; a++) {
        for (int b = 0; b < 3; b++) {
            double delta_ab = (a == b) ? 1.0 : 0.0;
            Hii[a*3 + b] = factor1 * delta_ab + factor2 * rij[a] * rij[b];
            Hij[a*3 + b] = -Hii[a*3 + b];  // Off-diagonal block is negative
        }
    }
}

// Compute angle Hessian (simplified - uses numerical differentiation for second derivatives)
static void computeAngleHessian(const Vec3& r1, const Vec3& r2, const Vec3& r3,
                                 double k, double theta0,
                                 double H[12][12]) {
    // Zero out
    for (int i = 0; i < 12; i++)
        for (int j = 0; j < 12; j++)
            H[i][j] = 0.0;

    const double h = 1e-5;
    Vec3 pos[3] = {r1, r2, r3};

    // Numerical second derivatives
    for (int i = 0; i < 9; i++) {  // 3 atoms * 3 coords
        for (int j = i; j < 9; j++) {
            int ai = i / 3, di = i % 3;
            int aj = j / 3, dj = j % 3;

            // Compute angle at displaced positions
            auto computeAngle = [](const Vec3& p1, const Vec3& p2, const Vec3& p3) {
                Vec3 v1 = p1 - p2;
                Vec3 v2 = p3 - p2;
                double dot = v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2];
                double len1 = sqrt(v1[0]*v1[0] + v1[1]*v1[1] + v1[2]*v1[2]);
                double len2 = sqrt(v2[0]*v2[0] + v2[1]*v2[1] + v2[2]*v2[2]);
                double cosTheta = dot / (len1 * len2);
                if (cosTheta > 1.0) cosTheta = 1.0;
                if (cosTheta < -1.0) cosTheta = -1.0;
                return acos(cosTheta);
            };

            auto computeEnergy = [&](const Vec3 p[3]) {
                double theta = computeAngle(p[0], p[1], p[2]);
                double dtheta = theta - theta0;
                return 0.5 * k * dtheta * dtheta;
            };

            // E(+h, +h)
            Vec3 pp[3] = {pos[0], pos[1], pos[2]};
            pp[ai][di] += h;
            pp[aj][dj] += h;
            double Epp = computeEnergy(pp);

            // E(+h, -h)
            Vec3 pm[3] = {pos[0], pos[1], pos[2]};
            pm[ai][di] += h;
            pm[aj][dj] -= h;
            double Epm = computeEnergy(pm);

            // E(-h, +h)
            Vec3 mp[3] = {pos[0], pos[1], pos[2]};
            mp[ai][di] -= h;
            mp[aj][dj] += h;
            double Emp = computeEnergy(mp);

            // E(-h, -h)
            Vec3 mm[3] = {pos[0], pos[1], pos[2]};
            mm[ai][di] -= h;
            mm[aj][dj] -= h;
            double Emm = computeEnergy(mm);

            double d2E = (Epp - Epm - Emp + Emm) / (4.0 * h * h);
            H[i][j] = d2E;
            H[j][i] = d2E;  // Symmetric
        }
    }
}

// Compute torsion Hessian using analytical Blondel-Karplus formulation
static void computeTorsionHessian(const Vec3& p1, const Vec3& p2, const Vec3& p3, const Vec3& p4,
                                   int n, double phi0, double k,
                                   double H[12][12]) {
    // Zero out
    for (int i = 0; i < 12; i++)
        for (int j = 0; j < 12; j++)
            H[i][j] = 0.0;

    // Bond vectors
    Vec3 b1 = {p2[0]-p1[0], p2[1]-p1[1], p2[2]-p1[2]};
    Vec3 b2 = {p3[0]-p2[0], p3[1]-p2[1], p3[2]-p2[2]};
    Vec3 b3 = {p4[0]-p3[0], p4[1]-p3[1], p4[2]-p3[2]};

    // Cross products
    Vec3 m = {b1[1]*b2[2] - b1[2]*b2[1], b1[2]*b2[0] - b1[0]*b2[2], b1[0]*b2[1] - b1[1]*b2[0]};
    Vec3 n_vec = {b2[1]*b3[2] - b2[2]*b3[1], b2[2]*b3[0] - b2[0]*b3[2], b2[0]*b3[1] - b2[1]*b3[0]};

    double m_sq = m[0]*m[0] + m[1]*m[1] + m[2]*m[2];
    double n_sq = n_vec[0]*n_vec[0] + n_vec[1]*n_vec[1] + n_vec[2]*n_vec[2];
    double b2_sq = b2[0]*b2[0] + b2[1]*b2[1] + b2[2]*b2[2];

    if (m_sq < 1e-20 || n_sq < 1e-20 || b2_sq < 1e-20) return;

    double b2_norm = sqrt(b2_sq);
    double m_norm = sqrt(m_sq);
    double n_norm = sqrt(n_sq);

    // Compute dihedral angle
    Vec3 m_hat = {m[0]/m_norm, m[1]/m_norm, m[2]/m_norm};
    Vec3 n_hat = {n_vec[0]/n_norm, n_vec[1]/n_norm, n_vec[2]/n_norm};
    Vec3 b2_hat = {b2[0]/b2_norm, b2[1]/b2_norm, b2[2]/b2_norm};

    double cos_phi = m_hat[0]*n_hat[0] + m_hat[1]*n_hat[1] + m_hat[2]*n_hat[2];
    Vec3 m_cross_b2 = {m_hat[1]*b2_hat[2] - m_hat[2]*b2_hat[1],
                       m_hat[2]*b2_hat[0] - m_hat[0]*b2_hat[2],
                       m_hat[0]*b2_hat[1] - m_hat[1]*b2_hat[0]};
    double sin_phi = m_cross_b2[0]*n_hat[0] + m_cross_b2[1]*n_hat[1] + m_cross_b2[2]*n_hat[2];
    double phi = atan2(sin_phi, cos_phi);

    // Energy derivatives
    // E = k * (1 + cos(n*phi - phi0))
    // dE/dphi = -k * n * sin(n*phi - phi0)
    // d²E/dphi² = -k * n² * cos(n*phi - phi0)
    double dE_dphi = -k * n * sin(n * phi - phi0);
    double d2E_dphi2 = -k * n * n * cos(n * phi - phi0);

    // Gradient of phi w.r.t. atom positions (Blondel-Karplus)
    Vec3 dphi_dr1 = {b2_norm / m_sq * m[0], b2_norm / m_sq * m[1], b2_norm / m_sq * m[2]};
    Vec3 dphi_dr4 = {-b2_norm / n_sq * n_vec[0], -b2_norm / n_sq * n_vec[1], -b2_norm / n_sq * n_vec[2]};

    // Projection factors
    double b1_dot_b2 = b1[0]*b2[0] + b1[1]*b2[1] + b1[2]*b2[2];
    double b3_dot_b2 = b3[0]*b2[0] + b3[1]*b2[1] + b3[2]*b2[2];
    double alpha = b1_dot_b2 / b2_sq;
    double beta = b3_dot_b2 / b2_sq;

    double c1 = -(1.0 + alpha);
    double c4 = beta;
    double d1 = alpha;
    double d4 = -(1.0 + beta);

    Vec3 dphi_dr2 = {c1 * dphi_dr1[0] + c4 * dphi_dr4[0],
                     c1 * dphi_dr1[1] + c4 * dphi_dr4[1],
                     c1 * dphi_dr1[2] + c4 * dphi_dr4[2]};
    Vec3 dphi_dr3 = {d1 * dphi_dr1[0] + d4 * dphi_dr4[0],
                     d1 * dphi_dr1[1] + d4 * dphi_dr4[1],
                     d1 * dphi_dr1[2] + d4 * dphi_dr4[2]};

    Vec3 dphi_dr[4] = {dphi_dr1, dphi_dr2, dphi_dr3, dphi_dr4};

    // Hessian = d²E/dphi² * (dphi/dri)(dphi/drj) + dE/dphi * d²phi/dri drj
    // For the outer product term (dominant contribution):
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            for (int di = 0; di < 3; di++) {
                for (int dj = 0; dj < 3; dj++) {
                    H[3*i + di][3*j + dj] += d2E_dphi2 * dphi_dr[i][di] * dphi_dr[j][dj];
                }
            }
        }
    }

    // Note: The second term (dE/dphi * d²phi/dri drj) is more complex and involves
    // derivatives of the skew matrices. For a full implementation, this would need
    // to be added. For now, we include only the dominant outer product term.
}

std::vector<double> BondedHessian::computeHessian(Context& context) {
    if (!initialized) {
        throw OpenMMException("BondedHessian: must call initialize() before computeHessian()");
    }

    int N3 = 3 * numAtoms;
    vector<double> H(N3 * N3, 0.0);

    // Get positions
    State state = context.getState(State::Positions);
    vector<Vec3> positions = state.getPositions();

    // Compute bond Hessians
    for (int b = 0; b < numBonds; b++) {
        int i = impl->bondAtoms[2*b];
        int j = impl->bondAtoms[2*b + 1];
        double k = impl->bondKs[b];
        double r0 = impl->bondLengths[b];

        double Hii[9], Hij[9];
        computeBondHessianBlock(positions[i], positions[j], k, r0, Hii, Hij);

        addBlock(H, N3, i, i, Hii);
        addBlock(H, N3, j, j, Hii);  // Same as Hii for bonds
        addBlock(H, N3, i, j, Hij);
        addBlock(H, N3, j, i, Hij);  // Symmetric
    }

    // Compute angle Hessians
    for (int a = 0; a < numAngles; a++) {
        int i = impl->angleAtoms[3*a];
        int j = impl->angleAtoms[3*a + 1];
        int k_idx = impl->angleAtoms[3*a + 2];
        double k = impl->angleKs[a];
        double theta0 = impl->angleValues[a];

        double Ha[12][12];
        computeAngleHessian(positions[i], positions[j], positions[k_idx], k, theta0, Ha);

        // Add to full Hessian
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
    for (int t = 0; t < numTorsions; t++) {
        int i = impl->torsionAtoms[4*t];
        int j = impl->torsionAtoms[4*t + 1];
        int k_idx = impl->torsionAtoms[4*t + 2];
        int l = impl->torsionAtoms[4*t + 3];
        int periodicity = impl->torsionPeriodicities[t];
        double phase = impl->torsionPhases[t];
        double k = impl->torsionKs[t];

        double Ht[12][12];
        computeTorsionHessian(positions[i], positions[j], positions[k_idx], positions[l],
                              periodicity, phase, k, Ht);

        // Add to full Hessian
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

    return H;
}
