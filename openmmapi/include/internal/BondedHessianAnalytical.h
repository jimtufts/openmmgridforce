#ifndef BONDED_HESSIAN_ANALYTICAL_H_
#define BONDED_HESSIAN_ANALYTICAL_H_

/* -------------------------------------------------------------------------- *
 *                              OpenMMGridForce                               *
 * -------------------------------------------------------------------------- *
 * Analytical Hessian formulas for harmonic bonds, harmonic angles, and       *
 * periodic torsions. Shared between BondedHessian and                        *
 * IsolatedBondedForce kernel implementations.                                *
 * -------------------------------------------------------------------------- */

#include "openmm/Vec3.h"
#include <cmath>
#include <algorithm>
#include <vector>

namespace GridForcePlugin {
namespace BondedHessianAnalytical {

// Helper: add a 3x3 block to a flattened Hessian matrix
inline void addBlock(std::vector<double>& H, int N3, int i, int j, const double block[9]) {
    for (int di = 0; di < 3; di++) {
        for (int dj = 0; dj < 3; dj++) {
            H[(3*i + di) * N3 + (3*j + dj)] += block[di * 3 + dj];
        }
    }
}

// Compute bond Hessian block (two 3x3 blocks: Hii and Hij)
// E = 0.5 * k * (r - r0)^2
inline void computeBondHessianBlock(const OpenMM::Vec3& ri, const OpenMM::Vec3& rj,
                                     double k, double r0,
                                     double Hii[9], double Hij[9]) {
    OpenMM::Vec3 rij = rj - ri;
    double r = std::sqrt(rij[0]*rij[0] + rij[1]*rij[1] + rij[2]*rij[2]);
    if (r < 1e-10) r = 1e-10;

    double invR = 1.0 / r;
    double invR2 = invR * invR;
    double factor1 = k * (1.0 - r0 * invR);
    double factor2 = k * r0 * invR * invR2;

    for (int a = 0; a < 3; a++) {
        for (int b = 0; b < 3; b++) {
            double delta_ab = (a == b) ? 1.0 : 0.0;
            Hii[a*3 + b] = factor1 * delta_ab + factor2 * rij[a] * rij[b];
            Hij[a*3 + b] = -Hii[a*3 + b];
        }
    }
}

// Compute analytical angle Hessian (9x9 matrix for 3 atoms)
// E = 0.5 * k * (theta - theta0)^2
inline void computeAngleHessian(const OpenMM::Vec3& r1, const OpenMM::Vec3& r2, const OpenMM::Vec3& r3,
                                 double k, double theta0, double H[9][9]) {
    for (int i = 0; i < 9; i++)
        for (int j = 0; j < 9; j++)
            H[i][j] = 0.0;

    OpenMM::Vec3 b1, b3;
    for (int d = 0; d < 3; d++) {
        b1[d] = r1[d] - r2[d];
        b3[d] = r3[d] - r2[d];
    }

    double L1 = std::sqrt(b1[0]*b1[0] + b1[1]*b1[1] + b1[2]*b1[2]);
    double L3 = std::sqrt(b3[0]*b3[0] + b3[1]*b3[1] + b3[2]*b3[2]);
    if (L1 < 1e-10 || L3 < 1e-10) return;

    double invL1 = 1.0 / L1, invL3 = 1.0 / L3;
    double invL1_sq = invL1 * invL1, invL3_sq = invL3 * invL3;

    double e1[3], e3[3];
    for (int d = 0; d < 3; d++) {
        e1[d] = b1[d] * invL1;
        e3[d] = b3[d] * invL3;
    }

    double cos_theta = e1[0]*e3[0] + e1[1]*e3[1] + e1[2]*e3[2];
    cos_theta = std::max(-0.9999999, std::min(0.9999999, cos_theta));
    double theta = std::acos(cos_theta);
    double sin_theta = std::sin(theta);
    if (std::fabs(sin_theta) < 1e-10) return;

    double inv_sin = 1.0 / sin_theta;
    double cot_theta = cos_theta * inv_sin;
    double dtheta = theta - theta0;
    double dE_dtheta = k * dtheta;
    double d2E_dtheta2 = k;

    double v1[3], v3[3];
    for (int d = 0; d < 3; d++) {
        v1[d] = e3[d] - cos_theta * e1[d];
        v3[d] = e1[d] - cos_theta * e3[d];
    }

    double g1[3], g3[3], g2[3];
    for (int d = 0; d < 3; d++) {
        g1[d] = -inv_sin * invL1 * v1[d];
        g3[d] = -inv_sin * invL3 * v3[d];
        g2[d] = -(g1[d] + g3[d]);
    }

    double grad[9];
    for (int d = 0; d < 3; d++) {
        grad[d] = g1[d];
        grad[3+d] = g2[d];
        grad[6+d] = g3[d];
    }

    double P1[9], P3[9];
    for (int a = 0; a < 3; a++) {
        for (int b = 0; b < 3; b++) {
            double delta = (a == b) ? 1.0 : 0.0;
            P1[a*3+b] = delta - e1[a] * e1[b];
            P3[a*3+b] = delta - e3[a] * e3[b];
        }
    }

    double H_theta[81];
    for (int i = 0; i < 81; i++) H_theta[i] = 0.0;

    double cot_over_sin = cot_theta * inv_sin;

    // Block (0,0): d2theta/dr1 dr1
    for (int a = 0; a < 3; a++) {
        for (int b = 0; b < 3; b++) {
            H_theta[a*9 + b] = inv_sin * invL1_sq * (
                e1[a] * v1[b] + v1[a] * e1[b] + cos_theta * P1[a*3+b] - cot_over_sin * v1[a] * v1[b]
            );
        }
    }

    // Block (2,2): d2theta/dr3 dr3
    for (int a = 0; a < 3; a++) {
        for (int b = 0; b < 3; b++) {
            H_theta[(6+a)*9 + (6+b)] = inv_sin * invL3_sq * (
                e3[a] * v3[b] + v3[a] * e3[b] + cos_theta * P3[a*3+b] - cot_over_sin * v3[a] * v3[b]
            );
        }
    }

    // Block (0,2): d2theta/dr1 dr3
    for (int a = 0; a < 3; a++) {
        for (int b = 0; b < 3; b++) {
            double val = -inv_sin * invL1 * invL3 * (P3[a*3+b] - e1[a] * v3[b] + cot_over_sin * v1[a] * v3[b]);
            H_theta[a*9 + (6+b)] = val;
            H_theta[(6+b)*9 + a] = val;
        }
    }

    // Fill remaining blocks by translational invariance
    for (int a = 0; a < 3; a++) {
        for (int b = 0; b < 3; b++) {
            double val = -H_theta[a*9 + b] - H_theta[a*9 + (6+b)];
            H_theta[a*9 + (3+b)] = val;
            H_theta[(3+b)*9 + a] = val;
        }
    }

    for (int a = 0; a < 3; a++) {
        for (int b = 0; b < 3; b++) {
            double val = -H_theta[(6+a)*9 + (6+b)] - H_theta[(6+a)*9 + b];
            H_theta[(6+a)*9 + (3+b)] = val;
            H_theta[(3+b)*9 + (6+a)] = val;
        }
    }

    for (int a = 0; a < 3; a++) {
        for (int b = 0; b < 3; b++) {
            H_theta[(3+a)*9 + (3+b)] = -H_theta[a*9 + (3+b)] - H_theta[(6+a)*9 + (3+b)];
        }
    }

    for (int i = 0; i < 9; i++) {
        for (int j = 0; j < 9; j++) {
            H[i][j] = d2E_dtheta2 * grad[i] * grad[j] + dE_dtheta * H_theta[i * 9 + j];
        }
    }
}

// Helper: 3x3 skew-symmetric matrix from vector
inline void skew(const double v[3], double S[3][3]) {
    S[0][0] = 0;     S[0][1] = -v[2]; S[0][2] = v[1];
    S[1][0] = v[2];  S[1][1] = 0;     S[1][2] = -v[0];
    S[2][0] = -v[1]; S[2][1] = v[0];  S[2][2] = 0;
}

// Compute dihedral angle using Blondel-Karplus formulation
inline double computeDihedralAngle(const OpenMM::Vec3& p1, const OpenMM::Vec3& p2,
                                    const OpenMM::Vec3& p3, const OpenMM::Vec3& p4) {
    double b1[3] = {p2[0]-p1[0], p2[1]-p1[1], p2[2]-p1[2]};
    double b2[3] = {p3[0]-p2[0], p3[1]-p2[1], p3[2]-p2[2]};
    double b3[3] = {p4[0]-p3[0], p4[1]-p3[1], p4[2]-p3[2]};

    double m[3] = {b1[1]*b2[2]-b1[2]*b2[1], b1[2]*b2[0]-b1[0]*b2[2], b1[0]*b2[1]-b1[1]*b2[0]};
    double n[3] = {b2[1]*b3[2]-b2[2]*b3[1], b2[2]*b3[0]-b2[0]*b3[2], b2[0]*b3[1]-b2[1]*b3[0]};

    double m_norm = std::sqrt(m[0]*m[0] + m[1]*m[1] + m[2]*m[2]);
    double n_norm = std::sqrt(n[0]*n[0] + n[1]*n[1] + n[2]*n[2]);
    double b2_norm = std::sqrt(b2[0]*b2[0] + b2[1]*b2[1] + b2[2]*b2[2]);

    if (m_norm < 1e-10 || n_norm < 1e-10 || b2_norm < 1e-10)
        return 0.0;

    double m_hat[3] = {m[0]/m_norm, m[1]/m_norm, m[2]/m_norm};
    double n_hat[3] = {n[0]/n_norm, n[1]/n_norm, n[2]/n_norm};
    double b2_hat[3] = {b2[0]/b2_norm, b2[1]/b2_norm, b2[2]/b2_norm};

    double cos_phi = m_hat[0]*n_hat[0] + m_hat[1]*n_hat[1] + m_hat[2]*n_hat[2];
    double mcb2[3] = {m_hat[1]*b2_hat[2]-m_hat[2]*b2_hat[1],
                      m_hat[2]*b2_hat[0]-m_hat[0]*b2_hat[2],
                      m_hat[0]*b2_hat[1]-m_hat[1]*b2_hat[0]};
    double sin_phi = mcb2[0]*n_hat[0] + mcb2[1]*n_hat[1] + mcb2[2]*n_hat[2];
    return std::atan2(sin_phi, cos_phi);
}

// Compute torsion Hessian using Blondel-Karplus with full H_phi term (12x12 matrix for 4 atoms)
// E = k * (1 + cos(n*phi - phase))
inline void computeTorsionHessian(const OpenMM::Vec3& p1, const OpenMM::Vec3& p2,
                                   const OpenMM::Vec3& p3, const OpenMM::Vec3& p4,
                                   int n, double phi0, double k, double H[12][12]) {
    for (int i = 0; i < 12; i++)
        for (int j = 0; j < 12; j++)
            H[i][j] = 0.0;

    double b1[3] = {p2[0]-p1[0], p2[1]-p1[1], p2[2]-p1[2]};
    double b2[3] = {p3[0]-p2[0], p3[1]-p2[1], p3[2]-p2[2]};
    double b3[3] = {p4[0]-p3[0], p4[1]-p3[1], p4[2]-p3[2]};

    double m[3] = {b1[1]*b2[2] - b1[2]*b2[1], b1[2]*b2[0] - b1[0]*b2[2], b1[0]*b2[1] - b1[1]*b2[0]};
    double nv[3] = {b2[1]*b3[2] - b2[2]*b3[1], b2[2]*b3[0] - b2[0]*b3[2], b2[0]*b3[1] - b2[1]*b3[0]};

    double m_sq = m[0]*m[0] + m[1]*m[1] + m[2]*m[2];
    double n_sq = nv[0]*nv[0] + nv[1]*nv[1] + nv[2]*nv[2];
    double b2_sq = b2[0]*b2[0] + b2[1]*b2[1] + b2[2]*b2[2];
    if (m_sq < 1e-20 || n_sq < 1e-20 || b2_sq < 1e-20) return;

    double b2_norm = std::sqrt(b2_sq);
    double m_norm = std::sqrt(m_sq), n_norm = std::sqrt(n_sq);

    double m_hat[3] = {m[0]/m_norm, m[1]/m_norm, m[2]/m_norm};
    double n_hat[3] = {nv[0]/n_norm, nv[1]/n_norm, nv[2]/n_norm};
    double b2_hat[3] = {b2[0]/b2_norm, b2[1]/b2_norm, b2[2]/b2_norm};

    double cos_phi = m_hat[0]*n_hat[0] + m_hat[1]*n_hat[1] + m_hat[2]*n_hat[2];
    double mcb2[3] = {m_hat[1]*b2_hat[2] - m_hat[2]*b2_hat[1],
                      m_hat[2]*b2_hat[0] - m_hat[0]*b2_hat[2],
                      m_hat[0]*b2_hat[1] - m_hat[1]*b2_hat[0]};
    double sin_phi = mcb2[0]*n_hat[0] + mcb2[1]*n_hat[1] + mcb2[2]*n_hat[2];
    double phi = std::atan2(sin_phi, cos_phi);

    double dE_dphi = -k * n * std::sin(n * phi - phi0);
    double d2E_dphi2 = -k * n * n * std::cos(n * phi - phi0);

    // Gradients G1, G4 (Blondel-Karplus)
    double G1[3] = {b2_norm / m_sq * m[0], b2_norm / m_sq * m[1], b2_norm / m_sq * m[2]};
    double G4[3] = {-b2_norm / n_sq * nv[0], -b2_norm / n_sq * nv[1], -b2_norm / n_sq * nv[2]};

    double b1b2 = b1[0]*b2[0] + b1[1]*b2[1] + b1[2]*b2[2];
    double b3b2 = b3[0]*b2[0] + b3[1]*b2[1] + b3[2]*b2[2];
    double alpha = b1b2 / b2_sq, beta = b3b2 / b2_sq;
    double c1 = -(1.0 + alpha), c4 = beta, d1 = alpha, d4 = -(1.0 + beta);

    double G2[3] = {c1 * G1[0] + c4 * G4[0], c1 * G1[1] + c4 * G4[1], c1 * G1[2] + c4 * G4[2]};
    double G3[3] = {d1 * G1[0] + d4 * G4[0], d1 * G1[1] + d4 * G4[1], d1 * G1[2] + d4 * G4[2]};

    // Outer products m*m and n*n
    double mm[3][3], nn[3][3];
    for (int a = 0; a < 3; a++) {
        for (int b = 0; b < 3; b++) {
            mm[a][b] = m[a] * m[b];
            nn[a][b] = nv[a] * nv[b];
        }
    }

    // Skew matrices
    double b1_x[3][3], b2_x[3][3], b3_x[3][3];
    skew(b1, b1_x);
    skew(b2, b2_x);
    skew(b3, b3_x);

    // dm/dp and dn/dp for each atom (3x3 matrices)
    double dm_dp[4][3][3], dn_dp[4][3][3];
    for (int a = 0; a < 3; a++) {
        for (int b = 0; b < 3; b++) {
            dm_dp[0][a][b] = b2_x[a][b];
            dm_dp[1][a][b] = -b2_x[a][b] - b1_x[a][b];
            dm_dp[2][a][b] = b1_x[a][b];
            dm_dp[3][a][b] = 0;
            dn_dp[0][a][b] = 0;
            dn_dp[1][a][b] = b3_x[a][b];
            dn_dp[2][a][b] = -b3_x[a][b] - b2_x[a][b];
            dn_dp[3][a][b] = b2_x[a][b];
        }
    }

    // d|b2|/dp for each atom
    double db2_norm_dp[4][3];
    for (int d = 0; d < 3; d++) {
        db2_norm_dp[0][d] = 0;
        db2_norm_dp[1][d] = -b2[d] / b2_norm;
        db2_norm_dp[2][d] = b2[d] / b2_norm;
        db2_norm_dp[3][d] = 0;
    }

    // Dot product derivatives
    double db1_dot_b2_dp[4][3], db3_dot_b2_dp[4][3], db2_sq_dp[4][3];
    for (int d = 0; d < 3; d++) {
        db1_dot_b2_dp[0][d] = -b2[d];
        db1_dot_b2_dp[1][d] = b2[d] - b1[d];
        db1_dot_b2_dp[2][d] = b1[d];
        db1_dot_b2_dp[3][d] = 0;
        db3_dot_b2_dp[0][d] = 0;
        db3_dot_b2_dp[1][d] = -b3[d];
        db3_dot_b2_dp[2][d] = b3[d] - b2[d];
        db3_dot_b2_dp[3][d] = b2[d];
        db2_sq_dp[0][d] = 0;
        db2_sq_dp[1][d] = -2 * b2[d];
        db2_sq_dp[2][d] = 2 * b2[d];
        db2_sq_dp[3][d] = 0;
    }

    // Compute dG1/dp and dG4/dp for each atom
    double dG1_dp[4][3][3], dG4_dp[4][3][3];
    for (int j = 0; j < 4; j++) {
        for (int a = 0; a < 3; a++) {
            for (int b = 0; b < 3; b++) {
                double term1 = m[a] * db2_norm_dp[j][b] / m_sq;
                double term2 = 0;
                for (int c = 0; c < 3; c++) {
                    double Imc = ((a == c) ? 1.0 : 0.0) - 2.0 * mm[a][c] / m_sq;
                    term2 += (b2_norm / m_sq) * Imc * dm_dp[j][c][b];
                }
                dG1_dp[j][a][b] = term1 + term2;

                double term1_G4 = -nv[a] * db2_norm_dp[j][b] / n_sq;
                double term2_G4 = 0;
                for (int c = 0; c < 3; c++) {
                    double Inc = ((a == c) ? 1.0 : 0.0) - 2.0 * nn[a][c] / n_sq;
                    term2_G4 += -(b2_norm / n_sq) * Inc * dn_dp[j][c][b];
                }
                dG4_dp[j][a][b] = term1_G4 + term2_G4;
            }
        }
    }

    // Compute coefficient derivatives
    double dc1_dp[4][3], dc4_dp[4][3], dd1_dp[4][3], dd4_dp[4][3];
    for (int j = 0; j < 4; j++) {
        for (int d = 0; d < 3; d++) {
            dc1_dp[j][d] = -db1_dot_b2_dp[j][d] / b2_sq + b1b2 * db2_sq_dp[j][d] / (b2_sq * b2_sq);
            dc4_dp[j][d] = db3_dot_b2_dp[j][d] / b2_sq - b3b2 * db2_sq_dp[j][d] / (b2_sq * b2_sq);
            dd1_dp[j][d] = db1_dot_b2_dp[j][d] / b2_sq - b1b2 * db2_sq_dp[j][d] / (b2_sq * b2_sq);
            dd4_dp[j][d] = -db3_dot_b2_dp[j][d] / b2_sq + b3b2 * db2_sq_dp[j][d] / (b2_sq * b2_sq);
        }
    }

    // Compute dG2/dp and dG3/dp
    double dG2_dp[4][3][3], dG3_dp[4][3][3];
    for (int j = 0; j < 4; j++) {
        for (int a = 0; a < 3; a++) {
            for (int b = 0; b < 3; b++) {
                dG2_dp[j][a][b] = G1[a] * dc1_dp[j][b] + c1 * dG1_dp[j][a][b]
                                + G4[a] * dc4_dp[j][b] + c4 * dG4_dp[j][a][b];
                dG3_dp[j][a][b] = G1[a] * dd1_dp[j][b] + d1 * dG1_dp[j][a][b]
                                + G4[a] * dd4_dp[j][b] + d4 * dG4_dp[j][a][b];
            }
        }
    }

    // Assemble H_phi (12x12 matrix)
    double H_phi[12][12];
    for (int i = 0; i < 12; i++)
        for (int j = 0; j < 12; j++)
            H_phi[i][j] = 0.0;

    for (int j = 0; j < 4; j++) {
        for (int a = 0; a < 3; a++) {
            for (int b = 0; b < 3; b++) {
                H_phi[0 + a][3*j + b] = dG1_dp[j][a][b];
                H_phi[3 + a][3*j + b] = dG2_dp[j][a][b];
                H_phi[6 + a][3*j + b] = dG3_dp[j][a][b];
                H_phi[9 + a][3*j + b] = dG4_dp[j][a][b];
            }
        }
    }

    // Symmetrize H_phi
    for (int i = 0; i < 12; i++) {
        for (int j = i + 1; j < 12; j++) {
            double avg = 0.5 * (H_phi[i][j] + H_phi[j][i]);
            H_phi[i][j] = avg;
            H_phi[j][i] = avg;
        }
    }

    // Full Hessian: H = d2E/dphi2 * G*G + dE/dphi * H_phi
    double G[4][3] = {{G1[0], G1[1], G1[2]}, {G2[0], G2[1], G2[2]},
                      {G3[0], G3[1], G3[2]}, {G4[0], G4[1], G4[2]}};

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            for (int di = 0; di < 3; di++) {
                for (int dj = 0; dj < 3; dj++) {
                    H[3*i + di][3*j + dj] = d2E_dphi2 * G[i][di] * G[j][dj]
                                          + dE_dphi * H_phi[3*i + di][3*j + dj];
                }
            }
        }
    }
}

}  // namespace BondedHessianAnalytical
}  // namespace GridForcePlugin

#endif /*BONDED_HESSIAN_ANALYTICAL_H_*/
