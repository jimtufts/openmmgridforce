#include "RFOMinimizer.h"
#include "BondedHessian.h"
#include "GridForce.h"
#include "GBSAGridForce.h"
#include "IsolatedNonbondedForce.h"
#include "openmm/State.h"
#include "openmm/OpenMMException.h"
#include <algorithm>
#include <cmath>
#include <limits>

using namespace GridForcePlugin;
using namespace OpenMM;
using namespace std;

RFOMinimizer::RFOMinimizer()
    : lastIterations(0), lastRMSForce(0.0), maxStep(0.05), useLineSearch(true) {}

RFOMinimizer::~RFOMinimizer() {}

void RFOMinimizer::jacobiEigen(vector<double>& H, int n,
                               vector<double>& eigvals,
                               vector<double>& eigvecs) {
    eigvecs.assign(n * n, 0.0);
    for (int i = 0; i < n; i++) eigvecs[i * n + i] = 1.0;

    const int maxSweeps = 100;
    const double tol = 1e-14;
    for (int sweep = 0; sweep < maxSweeps; sweep++) {
        double off = 0.0;
        for (int p = 0; p < n - 1; p++)
            for (int q = p + 1; q < n; q++)
                off += H[p * n + q] * H[p * n + q];
        if (off < tol) break;

        for (int p = 0; p < n - 1; p++) {
            for (int q = p + 1; q < n; q++) {
                double Hpq = H[p * n + q];
                if (fabs(Hpq) < 1e-16) continue;
                double Hpp = H[p * n + p];
                double Hqq = H[q * n + q];
                double theta = (Hqq - Hpp) / (2.0 * Hpq);
                double t = (theta >= 0.0)
                           ?  1.0 / (theta + sqrt(1.0 + theta * theta))
                           : -1.0 / (-theta + sqrt(1.0 + theta * theta));
                double c = 1.0 / sqrt(1.0 + t * t);
                double s = t * c;

                H[p * n + p] = Hpp - t * Hpq;
                H[q * n + q] = Hqq + t * Hpq;
                H[p * n + q] = 0.0;
                H[q * n + p] = 0.0;

                for (int i = 0; i < n; i++) {
                    if (i == p || i == q) continue;
                    double Hip = H[i * n + p];
                    double Hiq = H[i * n + q];
                    H[i * n + p] = c * Hip - s * Hiq;
                    H[p * n + i] = H[i * n + p];
                    H[i * n + q] = s * Hip + c * Hiq;
                    H[q * n + i] = H[i * n + q];
                }
                for (int i = 0; i < n; i++) {
                    double Vip = eigvecs[i * n + p];
                    double Viq = eigvecs[i * n + q];
                    eigvecs[i * n + p] = c * Vip - s * Viq;
                    eigvecs[i * n + q] = s * Vip + c * Viq;
                }
            }
        }
    }
    eigvals.resize(n);
    for (int i = 0; i < n; i++) eigvals[i] = H[i * n + i];
}

double RFOMinimizer::solveRFOShift(const vector<double>& eigvals,
                                    const vector<double>& g_proj) {
    // Find mu < min(eigvals) that solves f(mu) = Sum_i g_i^2 / (mu - lambda_i) = 1.
    // f is monotone decreasing on (-inf, min(lambda_i)), starting at +inf as mu
    // approaches min(lambda_i) from below and going to 0 as mu -> -inf.
    // Bisect on mu in (min(lambda_i) - large, min(lambda_i) - eps).
    int n = (int)eigvals.size();
    double lambdaMin = *min_element(eigvals.begin(), eigvals.end());
    double gTotal = 0.0;
    for (int i = 0; i < n; i++) gTotal += g_proj[i] * g_proj[i];
    if (gTotal < 1e-30) return lambdaMin - 1e-6;

    // Bracket: mu_hi just below lambdaMin, mu_lo far below.
    double eps = max(1e-12, 1e-6 * fabs(lambdaMin));
    double muHi = lambdaMin - eps;
    double muLo = lambdaMin - max(1.0, gTotal);
    // Expand muLo until f(muLo) < 1.
    for (int expand = 0; expand < 60; expand++) {
        double f = 0.0;
        for (int i = 0; i < n; i++) f += g_proj[i] * g_proj[i] / (muLo - eigvals[i]);
        if (f < 1.0) break;
        muLo -= (muHi - muLo);
    }
    // Bisect.
    for (int it = 0; it < 200; it++) {
        double mu = 0.5 * (muLo + muHi);
        double f = 0.0;
        for (int i = 0; i < n; i++) f += g_proj[i] * g_proj[i] / (mu - eigvals[i]);
        if (fabs(f - 1.0) < 1e-10) return mu;
        if (f > 1.0) muHi = mu;   // too close to lambdaMin
        else         muLo = mu;
    }
    return 0.5 * (muLo + muHi);
}

bool RFOMinimizer::minimize(Context& context, double tolerance, int maxIterations) {
    const System& system = context.getSystem();
    int numAtoms = system.getNumParticles();
    int n = 3 * numAtoms;

    BondedHessian bondedHessian;
    bondedHessian.initialize(system, context);

    vector<GridForce*> gridForces;
    vector<IsolatedNonbondedForce*> isoNBForces;
    vector<GBSAGridForce*> gbsaForces;
    for (int i = 0; i < system.getNumForces(); i++) {
        Force& force = const_cast<Force&>(system.getForce(i));
        if (auto gf = dynamic_cast<GridForce*>(&force)) gridForces.push_back(gf);
        if (auto inb = dynamic_cast<IsolatedNonbondedForce*>(&force)) isoNBForces.push_back(inb);
        if (auto gbsa = dynamic_cast<GBSAGridForce*>(&force)) gbsaForces.push_back(gbsa);
    }

    for (int iter = 0; iter < maxIterations; iter++) {
        lastIterations = iter + 1;

        State state = context.getState(State::Positions | State::Forces | State::Energy);
        vector<Vec3> positions = state.getPositions();
        vector<Vec3> forces = state.getForces();
        double energy = state.getPotentialEnergy();

        vector<double> gradient(n);
        for (int i = 0; i < numAtoms; i++) {
            gradient[3*i]     = -forces[i][0];
            gradient[3*i + 1] = -forces[i][1];
            gradient[3*i + 2] = -forces[i][2];
        }

        double sum = 0.0;
        for (double g : gradient) sum += g * g;
        lastRMSForce = sqrt(sum / gradient.size());
        if (lastRMSForce < tolerance) return true;

        vector<double> H = bondedHessian.computeHessian(context);
        for (GridForce* gf : gridForces) {
            gf->computeHessian(context);
            vector<double> blocks = gf->getHessianBlocks(context);
            for (int i = 0; i < numAtoms; i++) {
                if (6*i + 5 < (int)blocks.size()) {
                    double dxx = blocks[6*i + 0], dyy = blocks[6*i + 1], dzz = blocks[6*i + 2];
                    double dxy = blocks[6*i + 3], dxz = blocks[6*i + 4], dyz = blocks[6*i + 5];
                    int base = 3*i;
                    H[(base+0)*n + (base+0)] += dxx;
                    H[(base+1)*n + (base+1)] += dyy;
                    H[(base+2)*n + (base+2)] += dzz;
                    H[(base+0)*n + (base+1)] += dxy;
                    H[(base+1)*n + (base+0)] += dxy;
                    H[(base+0)*n + (base+2)] += dxz;
                    H[(base+2)*n + (base+0)] += dxz;
                    H[(base+1)*n + (base+2)] += dyz;
                    H[(base+2)*n + (base+1)] += dyz;
                }
            }
        }
        for (IsolatedNonbondedForce* inb : isoNBForces) {
            vector<double> nbH = inb->computeHessian(context);
            if (nbH.size() == H.size()) for (size_t i = 0; i < H.size(); i++) H[i] += nbH[i];
        }
        for (GBSAGridForce* gbsa : gbsaForces) {
            gbsa->computeHessian(context);
            vector<double> gbsaH = gbsa->getFullHessian(context);
            if (gbsaH.size() == H.size()) for (size_t i = 0; i < H.size(); i++) H[i] += gbsaH[i];
        }

        // Symmetrize H (Jacobi requires exact symmetry to full precision).
        for (int i = 0; i < n; i++)
            for (int j = i + 1; j < n; j++) {
                double sym = 0.5 * (H[i*n + j] + H[j*n + i]);
                H[i*n + j] = sym;
                H[j*n + i] = sym;
            }

        vector<double> eigvals, eigvecs;
        jacobiEigen(H, n, eigvals, eigvecs);

        // Project gradient onto eigenvectors.
        vector<double> g_proj(n, 0.0);
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                g_proj[i] += eigvecs[j * n + i] * gradient[j];

        double mu = solveRFOShift(eigvals, g_proj);

        // Step in eigenbasis: alpha_i = -g_i / (lambda_i - mu).
        vector<double> alpha(n, 0.0);
        for (int i = 0; i < n; i++) {
            double denom = eigvals[i] - mu;
            if (fabs(denom) < 1e-12) denom = (denom < 0 ? -1e-12 : 1e-12);
            alpha[i] = -g_proj[i] / denom;
        }

        vector<double> dx(n, 0.0);
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                dx[i] += eigvecs[i * n + j] * alpha[j];

        // Scale down if any component exceeds maxStep.
        double maxComp = 0.0;
        for (double v : dx) maxComp = max(maxComp, fabs(v));
        double stepScale = 1.0;
        if (maxComp > maxStep) stepScale = maxStep / maxComp;

        vector<Vec3> newPos = positions;
        for (int i = 0; i < numAtoms; i++) {
            newPos[i][0] += stepScale * dx[3*i];
            newPos[i][1] += stepScale * dx[3*i + 1];
            newPos[i][2] += stepScale * dx[3*i + 2];
        }
        context.setPositions(newPos);

        if (useLineSearch) {
            // Simple backtracking: halve until energy decreases or 5 attempts.
            State s2 = context.getState(State::Energy);
            double e2 = s2.getPotentialEnergy();
            double alpha_ls = stepScale;
            for (int bt = 0; bt < 5 && !(e2 < energy); bt++) {
                alpha_ls *= 0.5;
                for (int i = 0; i < numAtoms; i++) {
                    newPos[i][0] = positions[i][0] + alpha_ls * dx[3*i];
                    newPos[i][1] = positions[i][1] + alpha_ls * dx[3*i + 1];
                    newPos[i][2] = positions[i][2] + alpha_ls * dx[3*i + 2];
                }
                context.setPositions(newPos);
                s2 = context.getState(State::Energy);
                e2 = s2.getPotentialEnergy();
            }
        }
    }
    return false;
}
