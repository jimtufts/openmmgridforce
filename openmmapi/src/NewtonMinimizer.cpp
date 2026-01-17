/* -------------------------------------------------------------------------- *
 *                              OpenMMGridForce                               *
 * -------------------------------------------------------------------------- *
 * Newton-Raphson minimizer using analytical Hessians for fast convergence.  *
 * -------------------------------------------------------------------------- */

#include "NewtonMinimizer.h"
#include "BondedHessian.h"
#include "openmm/State.h"
#include "openmm/OpenMMException.h"
#include <cmath>
#include <algorithm>

using namespace GridForcePlugin;
using namespace OpenMM;
using namespace std;

NewtonMinimizer::NewtonMinimizer()
    : lastIterations(0), lastRMSForce(0.0), dampingFactor(0.01), useLineSearch(true) {
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
    // Initialize BondedHessian
    BondedHessian hessianCalc;
    hessianCalc.initialize(context.getSystem(), context);

    int numAtoms = context.getSystem().getNumParticles();
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

        // Compute Hessian
        vector<double> H = hessianCalc.computeHessian(context);

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
    // For now, just use bonded-only minimization
    // Full implementation would include nonbonded Hessian from IsolatedNonbondedForce
    return minimizeBondedOnly(context, tolerance, maxIterations);
}
