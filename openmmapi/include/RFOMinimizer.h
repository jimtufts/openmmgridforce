#ifndef OPENMM_RFOMINIMIZER_H_
#define OPENMM_RFOMINIMIZER_H_

/* -------------------------------------------------------------------------- *
 *                              OpenMMGridForce                               *
 * -------------------------------------------------------------------------- *
 * Partitioned Rational Function Optimization (P-RFO) minimizer on top of the *
 * plugin's analytical Hessian machinery.  Handles indefinite Hessians (saddle*
 * regions, small/negative eigenvalues) cleanly by solving the augmented      *
 * eigenvalue equation instead of adding uniform damping.                    *
 * -------------------------------------------------------------------------- */

#include <vector>
#include "internal/windowsExportGridForce.h"
#include "openmm/Context.h"

namespace GridForcePlugin {

/**
 * P-RFO minimizer.  Reuses NewtonMinimizer's Hessian assembly (BondedHessian
 * + GridForce + IsolatedNonbonded + GBSAGrid contributions) but replaces the
 * damped Cholesky step with the RFO shifted-Newton step:
 *
 *   dx = Sum_i [-g_i / (lambda_i - mu)] * v_i
 *
 * where (lambda_i, v_i) are the eigenpairs of the Hessian, g_i = v_i^T grad,
 * and mu is the smallest root of the secular equation
 *   1 = Sum_i g_i^2 / (mu - lambda_i)
 * (mu < min lambda_i for a downhill minimization step).  Near a minimum with
 * positive-definite H the step reduces to standard Newton; near a saddle it
 * follows the negative-curvature direction.
 */
class OPENMM_EXPORT_GRIDFORCE RFOMinimizer {
public:
    RFOMinimizer();
    ~RFOMinimizer();

    /**
     * Minimize the energy of the system in the Context.
     *
     * @param context        the Context containing the System to minimize
     * @param tolerance      RMS force tolerance for convergence (kJ/mol/nm)
     * @param maxIterations  maximum number of RFO iterations
     * @return true if converged, false if max iterations reached
     */
    bool minimize(OpenMM::Context& context, double tolerance = 1.0, int maxIterations = 100);

    int getNumIterations() const { return lastIterations; }
    double getFinalRMSForce() const { return lastRMSForce; }

    /**
     * Maximum step size in nm per Cartesian degree of freedom.  If the RFO
     * step exceeds this, it is scaled down uniformly.  Default 0.05 nm.
     */
    void setMaxStep(double s) { maxStep = s; }

    /**
     * Enable or disable backtracking line search along the RFO step.
     * Default: enabled.
     */
    void setLineSearch(bool enable) { useLineSearch = enable; }

private:
    int lastIterations;
    double lastRMSForce;
    double maxStep;
    bool useLineSearch;

    // Symmetric eigendecomposition via cyclic Jacobi (n = 3N).
    // On return, eigenvalues (unsorted) live in eigvals; eigvecs is
    // column-major with column i = eigenvector for eigvals[i].
    void jacobiEigen(std::vector<double>& H, int n,
                     std::vector<double>& eigvals,
                     std::vector<double>& eigvecs);

    // Solve the RFO secular equation for the shift mu, mu < min(eigvals).
    double solveRFOShift(const std::vector<double>& eigvals,
                         const std::vector<double>& g_proj);
};

}  // namespace GridForcePlugin

#endif  // OPENMM_RFOMINIMIZER_H_
