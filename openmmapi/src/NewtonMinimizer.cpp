/* -------------------------------------------------------------------------- *
 *                              OpenMMGridForce                               *
 * -------------------------------------------------------------------------- *
 * Line-search Newton minimizer with Levenberg-Marquardt Cholesky inner       *
 * solve.  Structure:                                                         *
 *                                                                            *
 *   outer loop  (Ponder & Richards, JCC 8, 1016, 1987; TINKER tncg.f)        *
 *     |                                                                      *
 *     +-- assemble analytical Hessian from                                   *
 *     |     BondedHessian + IsolatedBondedForce                              *
 *     |     + IsolatedNonbondedForce + GridForce (diag blocks)               *
 *     |     + GBSAGridForce (full)                                           *
 *     |                                                                      *
 *     +-- newton_direction():  Levenberg 1944 / Marquardt 1963.  Try         *
 *     |     Cholesky on H; if not PD, ramp lambda in H + lambda*I until      *
 *     |     Cholesky succeeds, else fall back to preconditioned SD.          *
 *     |                                                                      *
 *     +-- search():  Ponder & Richards 1987 line search (TINKER search.f):   *
 *           parabolic extrapolation + cubic interpolation with a Wolfe-      *
 *           type projected-gradient test (|s.g_b/s.g_0| <= cappa).           *
 *                                                                            *
 * References                                                                 *
 *   Ponder, J. W. & Richards, F. M. J. Comput. Chem. 8, 1016-1024 (1987).    *
 *   Levenberg, K. Q. Appl. Math. 2, 164-168 (1944).                          *
 *   Marquardt, D. W. J. Soc. Ind. Appl. Math. 11, 431-441 (1963).            *
 *   Nocedal, J. & Wright, S. J. Numerical Optimization (Springer 2006) 3.4.  *
 *                                                                            *
 * Alternative TN-CG inner solver (Ponder & Richards 1987 tnsolve,            *
 * symmetric-scaled diag preconditioner, forcing eps = min(1/cycle, g_rms))  *
 * is compiled in and available at runtime with NEWTON_USE_TNCG=1.  It        *
 * converges to the same minima as LM-Cholesky (bench 45915) but is           *
 * slower for small n; it should overtake Cholesky for n >~ 10^3 because     *
 * it scales O(n^2 * iter_CG) vs O(n^3) for Cholesky.                        *
 *                                                                            *
 * Historical footnote: earlier benchmarks that showed TN-CG hitting          *
 * negative-curvature every iteration were run against an assembled H         *
 * that was silently missing the bonded contribution (BondedHessian only      *
 * reads stock OpenMM Harmonic* forces, not the plugin's IsolatedBonded-      *
 * Force).  Once that Hessian bug was fixed, TN-CG worked as designed.       *
 * -------------------------------------------------------------------------- */

#include "NewtonMinimizer.h"
#include "BondedHessian.h"
#include "GridForce.h"
#include "GBSAGridForce.h"
#include "IsolatedBondedForce.h"
#include "IsolatedGBSAForce.h"
#include "IsolatedNonbondedForce.h"
#include "openmm/HarmonicAngleForce.h"
#include "openmm/HarmonicBondForce.h"
#include "openmm/PeriodicTorsionForce.h"
#include "openmm/State.h"
#include "openmm/OpenMMException.h"
#include <cmath>
#include <algorithm>
#include <cstdio>
#include <cstdlib>

using namespace GridForcePlugin;
using namespace OpenMM;
using namespace std;

namespace {

// Evaluate energy + gradient at flattened Cartesian positions x.
static double evaluate(Context& ctx, int numAtoms, const vector<double>& x,
                       vector<double>& g_out) {
    vector<Vec3> pos(numAtoms);
    for (int a = 0; a < numAtoms; a++)
        pos[a] = Vec3(x[3*a], x[3*a+1], x[3*a+2]);
    ctx.setPositions(pos);
    State st = ctx.getState(State::Forces | State::Energy);
    g_out.assign(3*numAtoms, 0.0);
    vector<Vec3> frc = st.getForces();
    for (int a = 0; a < numAtoms; a++) {
        g_out[3*a]   = -frc[a][0];
        g_out[3*a+1] = -frc[a][1];
        g_out[3*a+2] = -frc[a][2];
    }
    return st.getPotentialEnergy();
}

// Levenberg-Marquardt Cholesky inner solver (Levenberg 1944, Marquardt
// 1963; see Nocedal & Wright, Numerical Optimization 3.4).  Newton's
// equation H p = -g requires PD H.  In practice the assembled Hessian
// is indefinite for many MM geometries (LJ well concavity, grid
// discretization artifacts, near-zero rigid-body modes), so we ramp
// lambda in H + lambda*I until Cholesky succeeds.  This always produces
// a descent direction.
//
// iterOut counts the number of Cholesky attempts (informational).
// termReasonOut: 0=succeeded first try, 1=needed damping, 2=fallback SD.
static bool cholesky_solve_symmetric(const vector<double>& A,
                                     const vector<double>& b,
                                     vector<double>& x, int n);

static void newton_direction(vector<double>& H, vector<double>& g, int n, int /*cycle*/,
                             vector<double>& p, int& iterOut, int& termReasonOut) {
    p.assign(n, 0.0);
    iterOut = 0;
    termReasonOut = 0;

    // Estimate diagonal magnitude for the LM ramp
    double h_scale = 0.0;
    for (int i = 0; i < n; i++) h_scale = std::max(h_scale, std::fabs(H[i*n + i]));
    if (h_scale < 1.0) h_scale = 1.0;

    // Try to solve H p = -g, ramping lambda if not PD
    vector<double> Hd(H);
    vector<double> rhs(n);
    for (int i = 0; i < n; i++) rhs[i] = -g[i];

    double lambda = 0.0;
    const double lambda_max = 1e4 * h_scale;
    for (int attempt = 0; attempt < 20; attempt++) {
        iterOut = attempt + 1;
        // reset diagonal
        for (int i = 0; i < n; i++) Hd[i*n + i] = H[i*n + i] + lambda;
        if (cholesky_solve_symmetric(Hd, rhs, p, n)) {
            termReasonOut = (attempt == 0) ? 0 : 1;
            return;
        }
        lambda = (lambda == 0.0) ? 1.0 : lambda * 10.0;
        if (lambda > lambda_max) break;
    }
    // Final fallback: preconditioned steepest descent
    termReasonOut = 2;
    for (int i = 0; i < n; i++) {
        double hii = std::fabs(H[i*n + i]);
        p[i] = -g[i] / std::max(hii, 1.0);
    }
}

static bool cholesky_solve_symmetric(const vector<double>& A,
                                     const vector<double>& b,
                                     vector<double>& x, int n) {
    // In-place Cholesky on lower triangle
    vector<double> L(n * n, 0.0);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= i; j++) {
            double sum = A[i*n + j];
            for (int k = 0; k < j; k++) sum -= L[i*n + k] * L[j*n + k];
            if (i == j) {
                if (sum <= 0.0) return false;
                L[i*n + j] = std::sqrt(sum);
            } else {
                L[i*n + j] = sum / L[j*n + j];
            }
        }
    }
    // Solve L y = b
    vector<double> y(n);
    for (int i = 0; i < n; i++) {
        double sum = b[i];
        for (int j = 0; j < i; j++) sum -= L[i*n + j] * y[j];
        y[i] = sum / L[i*n + i];
    }
    // Solve L^T x = y
    x.assign(n, 0.0);
    for (int i = n - 1; i >= 0; i--) {
        double sum = y[i];
        for (int j = i + 1; j < n; j++) sum -= L[j*n + i] * x[j];
        x[i] = sum / L[i*n + i];
    }
    return true;
}

// ALTERNATIVE inner solver, selected at runtime by env var
// NEWTON_USE_TNCG=1.  Faithful port of TINKER's tnsolve (Ponder &
// Richards 1987): symmetric-scaled preconditioned CG with diagonal
// preconditioner and forcing eps = min(1/cycle, g_rms).  Validated
// against LM-Cholesky in bench 45915 (grid mode, n=141): both converge
// to identical minima; LM-Cholesky is ~2x faster on this size, but
// TN-CG scales O(n^2 * iter_CG) vs Cholesky's O(n^3) so it should win
// for n >= ~10^3.  Kept live because it's the natural default for
// larger MM systems.
//
// Note: an earlier bench (before the IsolatedBondedForce Hessian fix)
// showed this solver hitting negative-curvature termination every
// iteration.  That was a Hessian-assembly bug, not a solver bug — with
// the correct H, TN-CG works as designed.
static void tnsolve(vector<double>& H, vector<double>& g, int n, int cycle,
                    vector<double>& p, int& iterOut, int& termReasonOut) {
    p.assign(n, 0.0);
    vector<double> m(n), r(n), s(n), d(n), q(n);

    // Symmetric scaling m_i = 1/sqrt(|H_ii|)
    for (int i = 0; i < n; i++) {
        double h = std::fabs(H[i*n + i]);
        m[i] = (h > 1e-14) ? 1.0 / std::sqrt(h) : 1.0;
    }
    for (int i = 0; i < n; i++) {
        g[i] *= m[i];
        for (int j = 0; j < n; j++)
            H[i*n + j] *= m[i] * m[j];
    }

    // Init: r = -g, s = M^-1 r, d = s
    double gg = 0.0;
    for (int i = 0; i < n; i++) {
        r[i] = -g[i];
        gg  += g[i] * g[i];
    }
    double g_norm = std::sqrt(gg);
    for (int i = 0; i < n; i++) {
        double h = std::fabs(H[i*n + i]);
        s[i] = r[i] / std::max(h, 1e-14);
    }
    double rs = 0.0;
    for (int i = 0; i < n; i++) {
        d[i] = s[i];
        rs  += r[i] * s[i];
    }

    // Forcing sequence.  TINKER's `min(1/cycle, g_rms)` is calibrated for
    // large systems (>1e3 atoms).  For small MM ligands (n~100-500 DOF) it
    // truncates PCG so aggressively (1-2 iters) that the returned direction
    // is essentially diagonal-preconditioned steepest descent, and full
    // Newton convergence is lost.  Use a tighter fixed eps.
    double g_rms = g_norm / std::sqrt((double)n);
    double eps   = std::min({1.0e-3, 1.0 / (double)cycle, g_rms});
    int maxCG    = std::max((int)std::lround(10.0 * std::sqrt((double)n)), n);

    int iter = 1;
    int termReason = 0;  // 0=maxCG, 1=neg-curve, 2=converged
    while (true) {
        // q = H*d
        for (int i = 0; i < n; i++) {
            double sum = 0.0;
            for (int j = 0; j < n; j++) sum += H[i*n + j] * d[j];
            q[i] = sum;
        }
        // Negative-curvature check
        double dq = 0.0;
        for (int i = 0; i < n; i++) dq += d[i] * q[i];
        if (dq <= 0.0) {
            if (iter == 1) {
                for (int i = 0; i < n; i++) p[i] = d[i];
            }
            termReason = 1;
            break;
        }
        // TN step: p += alpha d;  r -= alpha q
        double alpha = rs / dq;
        double rr = 0.0;
        for (int i = 0; i < n; i++) {
            p[i] += alpha * d[i];
            r[i] -= alpha * q[i];
            rr   += r[i] * r[i];
        }
        double r_norm = std::sqrt(rr);
        if (g_norm > 0.0 && r_norm / g_norm <= eps) { termReason = 2; break; }

        // Precondition, update d
        for (int i = 0; i < n; i++) {
            double h = std::fabs(H[i*n + i]);
            s[i] = r[i] / std::max(h, 1e-14);
        }
        double rs_new = 0.0;
        for (int i = 0; i < n; i++) rs_new += r[i] * s[i];
        double beta = rs_new / rs;
        rs = rs_new;
        for (int i = 0; i < n; i++) d[i] = s[i] + beta * d[i];

        if (iter >= maxCG) break;   // termReason stays 0
        iter++;
    }
    iterOut = iter;
    termReasonOut = termReason;

    // Untransform back to original coordinates
    for (int i = 0; i < n; i++) {
        p[i] *= m[i];
        g[i] /= m[i];
    }
}

// TINKER-style line search: parabolic extrapolation + cubic interpolation.
// Uses both function and gradient values (Wolfe condition |sg/sg0| <= cappa).
// x, f, g are updated in place to the accepted point (or best trial on
// failure). Returns 0 on success/rescale/research, 1 on WideAngle,
// 2 on IntplnErr, 3 on BadIntpln.
static int search(Context& ctx, int numAtoms, vector<double>& x, double& f,
                  vector<double>& g, const vector<double>& p, double f_move,
                  int& fgCalls, double stpmax, double cappa, double slpmax,
                  double angmax, int intmax, double stpmin) {
    int n = 3 * numAtoms;

    vector<double> s(p);
    vector<double> x_0(x);

    double s_norm2 = 0.0, g_norm2 = 0.0;
    for (int i = 0; i < n; i++) { s_norm2 += p[i]*p[i]; g_norm2 += g[i]*g[i]; }
    double s_norm = std::sqrt(s_norm2);
    double g_norm = std::sqrt(g_norm2);
    if (s_norm == 0.0) return 2;

    double f_0 = f;
    double sg_0 = 0.0;
    for (int i = 0; i < n; i++) {
        s[i] = p[i] / s_norm;
        sg_0 += s[i] * g[i];
    }

    // Angle vs -g (WideAngle = no descent)
    double cosang = -sg_0 / std::max(g_norm, 1e-30);
    cosang = std::min(1.0, std::max(-1.0, cosang));
    double angle = std::acos(cosang) * (180.0 / M_PI);
    if (angle > angmax) return 1;

    // Initial step
    double step = 2.0 * std::fabs(f_move / (sg_0 == 0.0 ? -1e-30 : sg_0));
    step = std::min(step, s_norm);
    if (step > stpmax) step = stpmax;
    if (step < stpmin) step = stpmin;

    bool haveReSearch = false;
    int retStatus = 0;

    // === Outer restart loop ===
    while (true) {
        bool restart = true;
        int intpln = 0;
        double f_b = f_0, sg_b = sg_0;
        double f_a = 0.0, sg_a = 0.0;
        double f_c = 0.0, sg_c = 0.0, cube = 0.0;
        bool goToCubic = false;

        // --- Parabolic extrapolation phase ---
        while (true) {
            f_a  = f_b;
            sg_a = sg_b;
            for (int i = 0; i < n; i++) x[i] += step * s[i];

            fgCalls++;
            f_b = evaluate(ctx, numAtoms, x, g);
            sg_b = 0.0;
            for (int i = 0; i < n; i++) sg_b += s[i] * g[i];

            if (!std::isfinite(f_b) || !std::isfinite(sg_b)) {
                for (int i = 0; i < n; i++) x[i] = x_0[i];
                fgCalls++;
                f = evaluate(ctx, numAtoms, x, g);
                return 2;
            }

            if (std::fabs(sg_b) >= slpmax * std::fabs(sg_a) && restart) {
                for (int i = 0; i < n; i++) x[i] = x_0[i];
                step /= 10.0;
                retStatus = 0;
                goto restart_outer;
            }
            restart = false;

            if (std::fabs(sg_b) <= cappa * std::fabs(sg_0) && f_b < f_a) {
                f = f_b;
                return retStatus;
            }

            if (sg_b * sg_a < 0.0 || f_b > f_a) {
                goToCubic = true;
                break;
            }

            step *= 2.0;
            if (sg_b > sg_a) {
                double parab = (f_a - f_b) / (sg_b - sg_a);
                if (parab > 2.0 * step) parab = 2.0 * step;
                if (parab < 0.5 * step) parab = 0.5 * step;
                step = parab;
            }
            if (step > stpmax) step = stpmax;
        }

        // --- Cubic interpolation phase ---
        if (goToCubic) {
            while (true) {
                intpln++;
                double sss = 3.0 * (f_b - f_a) / step - sg_a - sg_b;
                double ttt = sss * sss - sg_a * sg_b;
                if (ttt < 0.0) { f = f_b; return 2; }
                ttt = std::sqrt(ttt);
                cube = step * (sg_b + ttt + sss) / (sg_b - sg_a + 2.0*ttt);
                if (cube < 0.0 || cube > step) { f = f_b; return 2; }
                for (int i = 0; i < n; i++) x[i] -= cube * s[i];

                fgCalls++;
                f_c = evaluate(ctx, numAtoms, x, g);
                sg_c = 0.0;
                for (int i = 0; i < n; i++) sg_c += s[i] * g[i];

                if (std::fabs(sg_c) <= cappa * std::fabs(sg_0)) {
                    f = f_c;
                    return retStatus;
                }

                // Update brackets
                bool bracketed = false;
                if (f_c <= f_a || f_c <= f_b) {
                    double cubstp = std::min(std::fabs(cube),
                                             std::fabs(step - cube));
                    if (cubstp >= stpmin && intpln < intmax) {
                        if (sg_a * sg_b < 0.0) {
                            if (sg_a * sg_c < 0.0) {
                                f_b = f_c; sg_b = sg_c;
                                step = step - cube;
                            } else {
                                f_a = f_c; sg_a = sg_c;
                                step = cube;
                                for (int i = 0; i < n; i++) x[i] += cube*s[i];
                            }
                        } else {
                            if (sg_a * sg_c < 0.0 || f_a <= f_c) {
                                f_b = f_c; sg_b = sg_c;
                                step = step - cube;
                            } else {
                                f_a = f_c; sg_a = sg_c;
                                step = cube;
                                for (int i = 0; i < n; i++) x[i] += cube*s[i];
                            }
                        }
                        bracketed = true;
                    }
                }
                if (!bracketed) break;
            }
        }

        // Interpolation failed: pick best of {a, b, c}
        double f_1;
        double sg_1;
        if (f_a <= f_b && f_a <= f_c) {
            f_1 = f_a; sg_1 = sg_a;
            for (int i = 0; i < n; i++) x[i] += (cube - step) * s[i];
        } else if (f_b <= f_c) {
            f_1 = f_b; sg_1 = sg_b;
            for (int i = 0; i < n; i++) x[i] += cube * s[i];
        } else {
            f_1 = f_c; sg_1 = sg_c;
        }

        if (f_1 > f_0) {
            fgCalls++;
            f = evaluate(ctx, numAtoms, x, g);
            return 2;
        }
        f_0 = f_1;
        sg_0 = sg_1;
        if (sg_1 > 0.0) {
            for (int i = 0; i < n; i++) s[i] = -s[i];
            sg_0 = -sg_1;
        }
        step = std::max(cube, step - cube) / 10.0;
        if (step < stpmin) step = stpmin;

        if (haveReSearch) {
            fgCalls++;
            f = evaluate(ctx, numAtoms, x, g);
            return 3;
        }
        haveReSearch = true;
        retStatus = 0;

    restart_outer:
        continue;
    }
}

}  // anonymous namespace

NewtonMinimizer::NewtonMinimizer()
    : lastIterations(0), lastRMSForce(0.0), dampingFactor(0.01),
      useLineSearch(true), maxStep(0.05), innerSolver(LMCholesky),
      kBatchBlockDiagonal(false) {
}

NewtonMinimizer::~NewtonMinimizer() {
}

double NewtonMinimizer::computeRMS(const vector<double>& v) {
    double sum = 0.0;
    for (size_t i = 0; i < v.size(); i++) {
        sum += v[i] * v[i];
    }
    return sqrt(sum / v.size());
}

bool NewtonMinimizer::solveCholesky(const vector<double>& H, const vector<double>& b,
                                     vector<double>& x, int n) {
    // Cholesky decomposition: H = L * L^T
    vector<double> L(n * n, 0.0);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= i; j++) {
            double sum = H[i * n + j];
            for (int k = 0; k < j; k++) {
                sum -= L[i * n + k] * L[j * n + k];
            }
            if (i == j) {
                if (sum <= 0.0) return false;  // Not positive definite
                L[i * n + j] = sqrt(sum);
            } else {
                L[i * n + j] = sum / L[j * n + j];
            }
        }
    }

    // Solve L * y = b
    vector<double> y(n);
    for (int i = 0; i < n; i++) {
        double sum = b[i];
        for (int j = 0; j < i; j++) {
            sum -= L[i * n + j] * y[j];
        }
        y[i] = sum / L[i * n + i];
    }

    // Solve L^T * x = y
    x.resize(n);
    for (int i = n - 1; i >= 0; i--) {
        double sum = y[i];
        for (int j = i + 1; j < n; j++) {
            sum -= L[j * n + i] * x[j];
        }
        x[i] = sum / L[i * n + i];
    }

    return true;
}

void NewtonMinimizer::solveDamped(const vector<double>& H, const vector<double>& b,
                                   vector<double>& x, int n, double lambda) {
    // Add damping: H_damped = H + lambda * I
    vector<double> H_damped = H;
    for (int i = 0; i < n; i++) {
        H_damped[i * n + i] += lambda;
    }

    // Try Cholesky first
    if (!solveCholesky(H_damped, b, x, n)) {
        // If still not positive definite, use more aggressive damping
        double scale = 1.0;
        for (int attempt = 0; attempt < 10; attempt++) {
            scale *= 10.0;
            for (int i = 0; i < n; i++) {
                H_damped[i * n + i] = H[i * n + i] + lambda * scale;
            }
            if (solveCholesky(H_damped, b, x, n)) {
                return;
            }
        }
        // Fall back to steepest descent
        double bNorm = 0.0;
        for (int i = 0; i < n; i++) bNorm += b[i] * b[i];
        bNorm = sqrt(bNorm);
        double stepSize = 0.001 / (bNorm + 1e-10);
        x.resize(n);
        for (int i = 0; i < n; i++) {
            x[i] = stepSize * b[i];
        }
    }
}

bool NewtonMinimizer::minimizeBondedOnly(Context& context, double tolerance, int maxIterations) {
    // Discover bonded sources.  BondedHessian reads stock OpenMM forces;
    // IsolatedBondedForce holds the plugin's K-replica bonded terms.
    const System& sys = context.getSystem();
    vector<IsolatedBondedForce*> isoBondedForces;
    bool hasStockBonded = false;
    for (int i = 0; i < sys.getNumForces(); i++) {
        Force& force = const_cast<Force&>(sys.getForce(i));
        if (auto* ibf = dynamic_cast<IsolatedBondedForce*>(&force))
            isoBondedForces.push_back(ibf);
        if (dynamic_cast<const HarmonicBondForce*>(&force)   != nullptr ||
            dynamic_cast<const HarmonicAngleForce*>(&force)  != nullptr ||
            dynamic_cast<const PeriodicTorsionForce*>(&force) != nullptr)
            hasStockBonded = true;
    }
    BondedHessian hessianCalc;
    if (hasStockBonded) hessianCalc.initialize(sys, context);

    int numAtoms = sys.getNumParticles();
    int n = 3 * numAtoms;

    for (int iter = 0; iter < maxIterations; iter++) {
        lastIterations = iter + 1;

        // Get current state
        State state = context.getState(State::Positions | State::Forces | State::Energy);
        vector<Vec3> positions = state.getPositions();
        vector<Vec3> forces = state.getForces();
        double energy = state.getPotentialEnergy();

        // Convert forces to gradient (negative forces)
        vector<double> gradient(n);
        for (int i = 0; i < numAtoms; i++) {
            gradient[3*i]     = -forces[i][0];
            gradient[3*i + 1] = -forces[i][1];
            gradient[3*i + 2] = -forces[i][2];
        }

        // Check convergence
        lastRMSForce = computeRMS(gradient);
        if (lastRMSForce < tolerance) {
            return true;
        }

        // Compute Hessian (stock + isolated bonded)
        vector<double> H = hasStockBonded
            ? hessianCalc.computeHessian(context)
            : vector<double>(n * n, 0.0);
        for (IsolatedBondedForce* ibf : isoBondedForces) {
            vector<double> iH = ibf->computeHessian(context, 0);
            if (iH.size() == H.size())
                for (size_t i = 0; i < H.size(); i++) H[i] += iH[i];
        }

        // Solve for Newton step: H * dx = -gradient
        vector<double> dx;
        solveDamped(H, gradient, dx, n, dampingFactor);

        // Negate to get descent direction
        for (int i = 0; i < n; i++) {
            dx[i] = -dx[i];
        }

        // Line search (optional)
        double alpha = 1.0;
        if (useLineSearch) {
            // Backtracking line search
            double c = 0.0001;  // Armijo condition parameter
            double rho = 0.5;   // Step reduction factor

            double directionalDeriv = 0.0;
            for (int i = 0; i < n; i++) {
                directionalDeriv += gradient[i] * dx[i];
            }

            for (int ls = 0; ls < 20; ls++) {
                // Update positions
                vector<Vec3> newPos = positions;
                for (int i = 0; i < numAtoms; i++) {
                    newPos[i][0] += alpha * dx[3*i];
                    newPos[i][1] += alpha * dx[3*i + 1];
                    newPos[i][2] += alpha * dx[3*i + 2];
                }
                context.setPositions(newPos);

                // Check energy
                State newState = context.getState(State::Energy);
                double newEnergy = newState.getPotentialEnergy();

                if (newEnergy <= energy + c * alpha * directionalDeriv || alpha < 1e-10) {
                    break;
                }
                alpha *= rho;
            }
        } else {
            // Direct step (no line search)
            vector<Vec3> newPos = positions;
            for (int i = 0; i < numAtoms; i++) {
                newPos[i][0] += alpha * dx[3*i];
                newPos[i][1] += alpha * dx[3*i + 1];
                newPos[i][2] += alpha * dx[3*i + 2];
            }
            context.setPositions(newPos);
        }
    }

    return false;  // Did not converge
}

bool NewtonMinimizer::minimize(Context& context, double tolerance, int maxIterations) {
    const System& system = context.getSystem();
    int numAtoms = system.getNumParticles();
    int n = 3 * numAtoms;

    // Discover force types providing Hessians.  Systems built with either
    // (a) stock OpenMM HarmonicBond/Angle/Torsion, or (b) plugin
    // IsolatedBondedForce need different bonded-Hessian dispatch.
    vector<IsolatedBondedForce*> isoBondedForces;
    vector<GridForce*> gridForces;
    vector<IsolatedNonbondedForce*> isoNBForces;
    vector<IsolatedGBSAForce*> isoGBSAForces;
    vector<GBSAGridForce*> gbsaForces;
    bool hasStockBonded = false;
    for (int i = 0; i < system.getNumForces(); i++) {
        Force& force = const_cast<Force&>(system.getForce(i));
        if (auto* ibf = dynamic_cast<IsolatedBondedForce*>(&force)) isoBondedForces.push_back(ibf);
        if (auto* gf = dynamic_cast<GridForce*>(&force)) gridForces.push_back(gf);
        if (auto* inb = dynamic_cast<IsolatedNonbondedForce*>(&force)) isoNBForces.push_back(inb);
        if (auto* igbsa = dynamic_cast<IsolatedGBSAForce*>(&force)) isoGBSAForces.push_back(igbsa);
        if (auto* gbsa = dynamic_cast<GBSAGridForce*>(&force)) gbsaForces.push_back(gbsa);
        if (dynamic_cast<const HarmonicBondForce*>(&force)   != nullptr ||
            dynamic_cast<const HarmonicAngleForce*>(&force)  != nullptr ||
            dynamic_cast<const PeriodicTorsionForce*>(&force) != nullptr)
            hasStockBonded = true;
    }
    // BondedHessian only reads stock Harmonic*/PeriodicTorsion; skip its
    // (guaranteed-zero) call entirely when the System has none.
    BondedHessian bondedHessian;
    if (hasStockBonded) bondedHessian.initialize(system, context);

    // Discover K (number of particle groups) from the first isolated force that
    // has groups.  In K-replica systems, IsolatedBondedForce / IsolatedNonbonded
    // Force return a 3N x 3N block for a single group; we need to loop over the
    // K groups and place each block at the right position in the assembled H.
    int K = 1;
    for (auto* ibf : isoBondedForces) {
        int g = ibf->getNumParticleGroups();
        if (g > K) K = g;
    }
    for (auto* inb : isoNBForces) {
        int g = inb->getNumParticleGroups();
        if (g > K) K = g;
    }
    if (K < 1) K = 1;

    // Cache each group's System-particle-index list once per minimize().
    // groupIndices[g][a] is the System particle index for template atom a
    // in group g.  Used to scatter per-group Hessian blocks into H.
    vector<vector<int>> groupIndices(K);
    if (!isoBondedForces.empty()) {
        for (int g = 0; g < K; g++) {
            std::string name;
            isoBondedForces[0]->getParticleGroup(g, name, groupIndices[g]);
        }
    } else if (!isoNBForces.empty()) {
        for (int g = 0; g < K; g++) {
            std::string name;
            isoNBForces[0]->getParticleGroup(g, name, groupIndices[g]);
        }
    }

    // Line-search parameters (TINKER defaults).  stpmax is a total-L2 cap on
    // the trial step; TINKER's default is 5.0.  For a well-truncated Newton
    // direction the natural full step is ||p||_2, which may be O(nm)-scale
    // even for a small ligand, so the cap must be generous.
    const double stpmax = 5.0;
    const double stpmin = 1e-16;
    const double cappa  = 0.1;
    const double slpmax = 10000.0;
    const double angmax = 180.0;
    const int    intmax = 8;
    (void)maxStep;   // legacy field, no longer used for stpmax

    // Initial evaluation
    State st = context.getState(State::Positions | State::Forces | State::Energy);
    vector<Vec3> pos = st.getPositions();
    vector<Vec3> frc = st.getForces();
    double f = st.getPotentialEnergy();

    vector<double> x(n), g(n);
    for (int a = 0; a < numAtoms; a++) {
        x[3*a]   = pos[a][0];
        x[3*a+1] = pos[a][1];
        x[3*a+2] = pos[a][2];
        g[3*a]   = -frc[a][0];
        g[3*a+1] = -frc[a][1];
        g[3*a+2] = -frc[a][2];
    }
    double f_old = f;

    double g_norm2 = 0.0;
    for (double v : g) g_norm2 += v*v;
    double g_norm = std::sqrt(g_norm2);
    double f_move = 0.5 * stpmax * g_norm;   // TINKER bootstrap for first step

    int fgCalls = 1;

    lastRMSForce = g_norm / std::sqrt((double)n);
    if (lastRMSForce < tolerance) {
        lastIterations = 0;
        return true;
    }

    int nerr = 0;
    const int maxerr = 3;

    const bool VERBOSE = std::getenv("NEWTON_VERBOSE") != nullptr;
    if (VERBOSE) {
        std::fprintf(stderr, "[TNCG] init: n=%d f=%.4f g_rms=%.3e stpmax=%.4f f_move=%.3e\n",
                     n, f, lastRMSForce, stpmax, f_move);
        std::fflush(stderr);
    }

    const int nAtomsTpl = groupIndices.empty() || groupIndices[0].empty()
        ? 0 : (int)groupIndices[0].size();
    const bool blockDiag = kBatchBlockDiagonal && K > 1 && nAtomsTpl > 0;

    for (int cycle = 1; cycle <= maxIterations; cycle++) {
        lastIterations = cycle;

        vector<double> p(n, 0.0);
        int iterCG = 0, termReason = 0;
        double f_before = f;
        double p_norm = 0.0, pg_dot = 0.0;

        if (blockDiag) {
            // ---- K-batch block-diagonal fast path ----
            // Assemble each group's 3N x 3N sub-Hessian directly (no full-H
            // allocation) and solve K independent Newton systems.  Correct
            // only if the physical H has no cross-group coupling; caller
            // opts in via setKBatchBlockDiagonal(true).
            int n3t = 3 * nAtomsTpl;

            // If any force provides a full K*N x K*N Hessian (IsolatedGBSA,
            // GBSAGrid), pre-fetch once and cache row starts so we can
            // extract per-group diagonal blocks without re-computing.
            vector<double> isoGBSAFull, gbsaGridFull;
            for (IsolatedGBSAForce* igbsa : isoGBSAForces)
                isoGBSAFull = igbsa->computeHessian(context);
            for (GBSAGridForce* gbsa : gbsaForces) {
                gbsa->computeHessian(context);
                gbsaGridFull = gbsa->getFullHessian(context);
            }

            // GridForce provides per-atom 3x3 blocks (6 packed values), no
            // cross-atom coupling by design.  One call, K groups all read
            // from the same array.
            vector<double> gridBlocks;
            for (GridForce* gf : gridForces) {
                gf->computeHessian(context);
                gridBlocks = gf->getHessianBlocks(context);
            }

            for (int gr = 0; gr < K; gr++) {
                // Extract per-group gradient
                const auto& idx = groupIndices[gr];
                vector<double> gg(n3t);
                for (int a = 0; a < nAtomsTpl; a++) {
                    gg[3*a]   = g[3*idx[a]];
                    gg[3*a+1] = g[3*idx[a]+1];
                    gg[3*a+2] = g[3*idx[a]+2];
                }

                // Assemble per-group Hessian
                vector<double> Hg(n3t * n3t, 0.0);
                for (IsolatedBondedForce* ibf : isoBondedForces) {
                    vector<double> iH = ibf->computeHessian(context, gr);
                    for (size_t i = 0; i < iH.size() && i < Hg.size(); i++)
                        Hg[i] += iH[i];
                }
                for (IsolatedNonbondedForce* inb : isoNBForces) {
                    vector<double> iH = inb->computeHessian(context, gr);
                    for (size_t i = 0; i < iH.size() && i < Hg.size(); i++)
                        Hg[i] += iH[i];
                }
                // Extract diagonal block from full K*N x K*N sources
                auto extractDiagBlock = [&](const vector<double>& Hfull) {
                    if ((int)Hfull.size() != n * n) return;
                    for (int a = 0; a < nAtomsTpl; a++)
                        for (int b = 0; b < nAtomsTpl; b++) {
                            int rf = 3 * idx[a], cf = 3 * idx[b];
                            for (int di = 0; di < 3; di++)
                                for (int dj = 0; dj < 3; dj++)
                                    Hg[(3*a+di)*n3t + (3*b+dj)]
                                        += Hfull[(rf+di)*n + (cf+dj)];
                        }
                };
                extractDiagBlock(isoGBSAFull);
                extractDiagBlock(gbsaGridFull);

                // GridForce per-atom diagonal blocks for atoms in this group
                for (int a = 0; a < nAtomsTpl; a++) {
                    int atom = idx[a];
                    if (6*atom + 5 >= (int)gridBlocks.size()) continue;
                    double dxx = gridBlocks[6*atom + 0];
                    double dyy = gridBlocks[6*atom + 1];
                    double dzz = gridBlocks[6*atom + 2];
                    double dxy = gridBlocks[6*atom + 3];
                    double dxz = gridBlocks[6*atom + 4];
                    double dyz = gridBlocks[6*atom + 5];
                    int base = 3*a;
                    Hg[(base+0)*n3t + (base+0)] += dxx;
                    Hg[(base+1)*n3t + (base+1)] += dyy;
                    Hg[(base+2)*n3t + (base+2)] += dzz;
                    Hg[(base+0)*n3t + (base+1)] += dxy;
                    Hg[(base+1)*n3t + (base+0)] += dxy;
                    Hg[(base+0)*n3t + (base+2)] += dxz;
                    Hg[(base+2)*n3t + (base+0)] += dxz;
                    Hg[(base+1)*n3t + (base+2)] += dyz;
                    Hg[(base+2)*n3t + (base+1)] += dyz;
                }

                // Inner solve (LM-Cholesky always for the fast path -- TNCG
                // per-block is not worth the setup for a 3N x 3N system)
                vector<double> pg;
                int iCG = 0, iTR = 0;
                newton_direction(Hg, gg, n3t, cycle, pg, iCG, iTR);
                iterCG += iCG;
                if (iTR > termReason) termReason = iTR;

                // Descent-direction guard per group (steepest-descent fallback)
                double pgg = 0.0;
                for (int i = 0; i < n3t; i++) pgg += pg[i] * gg[i];
                if (pgg >= 0.0)
                    for (int i = 0; i < n3t; i++) pg[i] = -gg[i];

                // Scatter to full p
                for (int a = 0; a < nAtomsTpl; a++) {
                    p[3*idx[a]]   = pg[3*a];
                    p[3*idx[a]+1] = pg[3*a+1];
                    p[3*idx[a]+2] = pg[3*a+2];
                }
            }

            for (int i = 0; i < n; i++) {
                p_norm += p[i]*p[i];
                pg_dot += p[i]*g[i];
            }
            p_norm = std::sqrt(p_norm);

            goto lineSearchStep;
        }

        {  // ---- Full-Hessian path (default) ----

        // ---- Assemble full analytical Hessian for this outer step ----
        // The assembled H is (3*K*N) x (3*K*N) in System-particle ordering.
        vector<double> H = hasStockBonded
            ? bondedHessian.computeHessian(context)
            : vector<double>(n * n, 0.0);
        // Bonded and intra-group nonbonded Hessians come out per-group in
        // template-atom ordering (3N x 3N).  Scatter each into the full
        // matrix at the rows/cols given by that group's System particle
        // indices.
        for (IsolatedBondedForce* ibf : isoBondedForces) {
            int nAtomsTpl = groupIndices[0].empty() ? 0 : (int)groupIndices[0].size();
            for (int g = 0; g < K; g++) {
                vector<double> iH = ibf->computeHessian(context, g);
                int n3t = 3 * nAtomsTpl;
                const auto& idx = groupIndices[g];
                for (int a = 0; a < nAtomsTpl; a++) {
                    for (int b = 0; b < nAtomsTpl; b++) {
                        int rowFull = 3 * idx[a];
                        int colFull = 3 * idx[b];
                        for (int di = 0; di < 3; di++)
                            for (int dj = 0; dj < 3; dj++)
                                H[(rowFull + di) * n + (colFull + dj)]
                                    += iH[(3*a + di) * n3t + (3*b + dj)];
                    }
                }
            }
        }
        for (IsolatedNonbondedForce* inb : isoNBForces) {
            int nAtomsTpl = groupIndices[0].empty() ? 0 : (int)groupIndices[0].size();
            for (int g = 0; g < K; g++) {
                vector<double> iH = inb->computeHessian(context, g);
                int n3t = 3 * nAtomsTpl;
                const auto& idx = groupIndices[g];
                for (int a = 0; a < nAtomsTpl; a++) {
                    for (int b = 0; b < nAtomsTpl; b++) {
                        int rowFull = 3 * idx[a];
                        int colFull = 3 * idx[b];
                        for (int di = 0; di < 3; di++)
                            for (int dj = 0; dj < 3; dj++)
                                H[(rowFull + di) * n + (colFull + dj)]
                                    += iH[(3*a + di) * n3t + (3*b + dj)];
                    }
                }
            }
        }

        for (GridForce* gf : gridForces) {
            gf->computeHessian(context);
            vector<double> blocks = gf->getHessianBlocks(context);
            for (int i = 0; i < numAtoms; i++) {
                if (6*i + 5 < (int)blocks.size()) {
                    double dxx = blocks[6*i+0], dyy = blocks[6*i+1], dzz = blocks[6*i+2];
                    double dxy = blocks[6*i+3], dxz = blocks[6*i+4], dyz = blocks[6*i+5];
                    int b = 3*i;
                    H[(b+0)*n + (b+0)] += dxx;
                    H[(b+1)*n + (b+1)] += dyy;
                    H[(b+2)*n + (b+2)] += dzz;
                    H[(b+0)*n + (b+1)] += dxy;
                    H[(b+1)*n + (b+0)] += dxy;
                    H[(b+0)*n + (b+2)] += dxz;
                    H[(b+2)*n + (b+0)] += dxz;
                    H[(b+1)*n + (b+2)] += dyz;
                    H[(b+2)*n + (b+1)] += dyz;
                }
            }
        }
        // IsolatedGBSAForce returns a full 3(K*N) x 3(K*N) Hessian already in
        // System-particle ordering, so no per-group placement is needed.
        for (IsolatedGBSAForce* igbsa : isoGBSAForces) {
            vector<double> iH = igbsa->computeHessian(context);
            if (iH.size() == H.size())
                for (size_t i = 0; i < H.size(); i++) H[i] += iH[i];
        }
        for (GBSAGridForce* gbsa : gbsaForces) {
            gbsa->computeHessian(context);
            vector<double> gbsaH = gbsa->getFullHessian(context);
            if (gbsaH.size() == H.size())
                for (size_t i = 0; i < H.size(); i++) H[i] += gbsaH[i];
        }

        // Diagnostic: H_ii statistics + off-diag magnitude
        if (VERBOSE && cycle <= 3) {
            double hmin = H[0], hmax = H[0], hsum = 0.0;
            int nneg = 0;
            for (int i = 0; i < n; i++) {
                double h = H[i*n + i];
                if (h < hmin) hmin = h;
                if (h > hmax) hmax = h;
                if (h < 0.0) nneg++;
                hsum += h;
            }
            std::fprintf(stderr,
                "[TNCG] H_diag: min=%+.3e max=%+.3e mean=%+.3e n_neg=%d/%d\n",
                hmin, hmax, hsum/n, nneg, n);
            std::fflush(stderr);
        }

        // ---- Inner solve for Newton search direction p ----
        if (innerSolver == TNCG) {
            // Symmetric-scaled PCG mutates g into scaled form and back.
            vector<double> g_scaled = g;
            tnsolve(H, g_scaled, n, cycle, p, iterCG, termReason);
            g = g_scaled;
        } else {
            newton_direction(H, g, n, cycle, p, iterCG, termReason);
        }

        // Defensive: if p is not a descent direction (numerical blowup) fall
        // back to steepest descent
        {
            double pg = 0.0;
            for (int i = 0; i < n; i++) pg += p[i] * g[i];
            if (pg >= 0.0) {
                for (int i = 0; i < n; i++) p[i] = -g[i];
            }
        }

        p_norm = 0.0; pg_dot = 0.0;
        for (int i = 0; i < n; i++) { p_norm += p[i]*p[i]; pg_dot += p[i]*g[i]; }
        p_norm = std::sqrt(p_norm);

        }  // end full-H block

    lineSearchStep:
        // ---- Line search along p ----
        int lss = search(context, numAtoms, x, f, g, p, f_move, fgCalls,
                         stpmax, cappa, slpmax, angmax, intmax, stpmin);

        f_move = f_old - f;
        f_old  = f;

        g_norm2 = 0.0;
        for (double v : g) g_norm2 += v*v;
        g_norm = std::sqrt(g_norm2);
        lastRMSForce = g_norm / std::sqrt((double)n);

        if (VERBOSE && (cycle <= 20 || cycle % 25 == 0)) {
            const char* tr = (termReason == 0) ? "PD" :
                             (termReason == 1) ? "damped" : "sd-fallback";
            std::fprintf(stderr,
                "[TNCG] c=%d try=%d(%s) ||p||=%.3e p.g=%+.3e f: %.4f -> %.4f (dE=%+.3e) g_rms=%.3e lss=%d\n",
                cycle, iterCG, tr, p_norm, pg_dot, f_before, f, f-f_before, lastRMSForce, lss);
            std::fflush(stderr);
        }

        if (lastRMSForce < tolerance) return true;
        if (f_move == 0.0) return false;

        if (lss == 2 || lss == 3) {  // IntplnErr or BadIntpln
            nerr++;
            if (nerr >= maxerr) return false;
        } else {
            nerr = 0;
        }
    }

    return false;
}
