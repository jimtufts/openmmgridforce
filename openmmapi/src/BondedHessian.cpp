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

// Compute analytical angle Hessian for E = 0.5 * k * (theta - theta0)^2
static void computeAngleHessian(const Vec3& r1, const Vec3& r2, const Vec3& r3,
                                 double k, double theta0,
                                 double H[12][12]) {
    // Zero out
    for (int i = 0; i < 12; i++)
        for (int j = 0; j < 12; j++)
            H[i][j] = 0.0;

    // Vectors from central atom
    Vec3 b1, b3;
    for (int d = 0; d < 3; d++) {
        b1[d] = r1[d] - r2[d];
        b3[d] = r3[d] - r2[d];
    }

    double L1 = sqrt(b1[0]*b1[0] + b1[1]*b1[1] + b1[2]*b1[2]);
    double L3 = sqrt(b3[0]*b3[0] + b3[1]*b3[1] + b3[2]*b3[2]);

    if (L1 < 1e-10 || L3 < 1e-10) return;

    double invL1 = 1.0 / L1;
    double invL3 = 1.0 / L3;
    double invL1_sq = invL1 * invL1;
    double invL3_sq = invL3 * invL3;

    // Unit vectors
    double e1[3], e3[3];
    for (int d = 0; d < 3; d++) {
        e1[d] = b1[d] * invL1;
        e3[d] = b3[d] * invL3;
    }

    double cos_theta = e1[0]*e3[0] + e1[1]*e3[1] + e1[2]*e3[2];
    if (cos_theta > 0.9999999) cos_theta = 0.9999999;
    if (cos_theta < -0.9999999) cos_theta = -0.9999999;
    double theta = acos(cos_theta);
    double sin_theta = sin(theta);

    if (fabs(sin_theta) < 1e-10) return;

    double inv_sin = 1.0 / sin_theta;
    double cot_theta = cos_theta * inv_sin;

    // Energy derivatives
    double dtheta = theta - theta0;
    double dE_dtheta = k * dtheta;
    double d2E_dtheta2 = k;

    // Gradient vectors: v1 = e3 - cos θ * e1, v3 = e1 - cos θ * e3
    double v1[3], v3[3];
    for (int d = 0; d < 3; d++) {
        v1[d] = e3[d] - cos_theta * e1[d];
        v3[d] = e1[d] - cos_theta * e3[d];
    }

    // Gradient of theta
    double g1[3], g3[3], g2[3];
    for (int d = 0; d < 3; d++) {
        g1[d] = -inv_sin * invL1 * v1[d];
        g3[d] = -inv_sin * invL3 * v3[d];
        g2[d] = -(g1[d] + g3[d]);
    }

    // Full gradient vector (9 elements)
    double grad[9];
    for (int d = 0; d < 3; d++) {
        grad[d] = g1[d];
        grad[3+d] = g2[d];
        grad[6+d] = g3[d];
    }

    // Projection matrices P1 = I - e1⊗e1, P3 = I - e3⊗e3
    double P1[9], P3[9];
    for (int a = 0; a < 3; a++) {
        for (int b = 0; b < 3; b++) {
            double delta = (a == b) ? 1.0 : 0.0;
            P1[a*3+b] = delta - e1[a] * e1[b];
            P3[a*3+b] = delta - e3[a] * e3[b];
        }
    }

    // Initialize H_theta (9x9 matrix for d²θ/dr)
    double H_theta[81];
    for (int i = 0; i < 81; i++) H_theta[i] = 0.0;

    // d²θ/dr1 dr1
    for (int a = 0; a < 3; a++) {
        for (int b = 0; b < 3; b++) {
            double val = inv_sin * invL1_sq * (
                2.0 * v1[a] * e1[b] +
                cot_theta * v1[a] * v1[b] +
                cos_theta * P1[a*3+b]
            );
            H_theta[(0+a)*9 + (0+b)] = val;
        }
    }

    // d²θ/dr3 dr3
    for (int a = 0; a < 3; a++) {
        for (int b = 0; b < 3; b++) {
            double val = inv_sin * invL3_sq * (
                2.0 * v3[a] * e3[b] +
                cot_theta * v3[a] * v3[b] +
                cos_theta * P3[a*3+b]
            );
            H_theta[(6+a)*9 + (6+b)] = val;
        }
    }

    // d²θ/dr1 dr3
    for (int a = 0; a < 3; a++) {
        for (int b = 0; b < 3; b++) {
            double val = -inv_sin * invL1 * invL3 * (
                P3[a*3+b] -
                v3[a] * e1[b] -
                cot_theta * v1[a] * v3[b]
            );
            H_theta[(0+a)*9 + (6+b)] = val;
            H_theta[(6+b)*9 + (0+a)] = val;
        }
    }

    // d²θ/dr1 dr2 and d²θ/dr2 dr1 (via g2 = -(g1 + g3))
    for (int a = 0; a < 3; a++) {
        for (int b = 0; b < 3; b++) {
            double d2_r1_r1 = H_theta[(0+a)*9 + (0+b)];
            double d2_r1_r3 = H_theta[(0+a)*9 + (6+b)];
            double val = -d2_r1_r1 - d2_r1_r3;
            H_theta[(0+a)*9 + (3+b)] = val;
            H_theta[(3+b)*9 + (0+a)] = val;
        }
    }

    // d²θ/dr3 dr2 and d²θ/dr2 dr3
    for (int a = 0; a < 3; a++) {
        for (int b = 0; b < 3; b++) {
            double d2_r3_r3 = H_theta[(6+a)*9 + (6+b)];
            double d2_r3_r1 = H_theta[(6+a)*9 + (0+b)];
            double val = -d2_r3_r3 - d2_r3_r1;
            H_theta[(6+a)*9 + (3+b)] = val;
            H_theta[(3+b)*9 + (6+a)] = val;
        }
    }

    // d²θ/dr2 dr2
    for (int a = 0; a < 3; a++) {
        for (int b = 0; b < 3; b++) {
            double d2_r1_r2 = H_theta[(0+a)*9 + (3+b)];
            double d2_r3_r2 = H_theta[(6+a)*9 + (3+b)];
            H_theta[(3+a)*9 + (3+b)] = -d2_r1_r2 - d2_r3_r2;
        }
    }

    // Full Hessian: H = d²E/dθ² * grad ⊗ grad + dE/dθ * H_theta
    for (int i = 0; i < 9; i++) {
        for (int j = 0; j < 9; j++) {
            H[i][j] = d2E_dtheta2 * grad[i] * grad[j] + dE_dtheta * H_theta[i * 9 + j];
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
