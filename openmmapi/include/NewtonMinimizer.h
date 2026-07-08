#ifndef OPENMM_NEWTONMINIMIZER_H_
#define OPENMM_NEWTONMINIMIZER_H_

/* -------------------------------------------------------------------------- *
 *                              OpenMMGridForce                               *
 * -------------------------------------------------------------------------- *
 * Newton-Raphson minimizer using analytical Hessians for fast convergence.  *
 * -------------------------------------------------------------------------- */

#include <vector>
#include "internal/windowsExportGridForce.h"
#include "openmm/Context.h"

namespace GridForcePlugin {

/**
 * NewtonMinimizer performs energy minimization using Newton-Raphson optimization
 * with analytical Hessians. This provides quadratic convergence near minima,
 * making it much faster than gradient-based methods for small molecules.
 *
 * The minimizer computes the Hessian of bonded forces (bonds, angles, torsions)
 * and optionally nonbonded forces, then solves H * dx = -g for the step direction.
 *
 * Features:
 * - Full Newton steps when Hessian is positive definite
 * - Levenberg-Marquardt damping for non-positive-definite regions
 * - Line search for robustness
 * - Support for bonded + nonbonded Hessians
 *
 * Usage:
 *   NewtonMinimizer minimizer;
 *   minimizer.minimize(context, tolerance, maxIterations);
 */
class OPENMM_EXPORT_GRIDFORCE NewtonMinimizer {
public:
    /**
     * Inner solver used to compute the Newton search direction each outer
     * iteration.
     *
     *  LMCholesky (default): dense Cholesky on H, with Levenberg-Marquardt
     *      diagonal shift (H + lambda*I) when H is not PD.  O(n^3) per
     *      outer iteration; the natural choice for small MM systems
     *      (n_dof <~ 10^3).
     *
     *  TNCG: TINKER-style symmetric-scaled preconditioned CG (Ponder &
     *      Richards 1987), forcing eps = min(1/cycle, g_rms).  Scales
     *      O(n^2 * iter_CG) per outer iteration and should overtake
     *      LM-Cholesky for larger systems.
     */
    enum InnerSolver {
        LMCholesky = 0,
        TNCG       = 1,
    };

    /**
     * Create a NewtonMinimizer.
     */
    NewtonMinimizer();

    ~NewtonMinimizer();

    /**
     * Minimize the energy of a System using Newton-Raphson optimization.
     *
     * @param context        the Context containing the System to minimize
     * @param tolerance      the energy tolerance for convergence (kJ/mol)
     * @param maxIterations  maximum number of Newton iterations
     * @return true if converged, false if max iterations reached
     */
    bool minimize(OpenMM::Context& context, double tolerance = 1.0, int maxIterations = 100);

    /**
     * Minimize using only bonded forces (faster, useful for initial relaxation).
     *
     * @param context        the Context containing the System to minimize
     * @param tolerance      the RMS force tolerance for convergence (kJ/mol/nm)
     * @param maxIterations  maximum number of Newton iterations
     * @return true if converged, false if max iterations reached
     */
    bool minimizeBondedOnly(OpenMM::Context& context, double tolerance = 10.0, int maxIterations = 50);

    /**
     * Get the number of iterations used in the last minimization.
     */
    int getNumIterations() const { return lastIterations; }

    /**
     * Get the final RMS force from the last minimization.
     */
    double getFinalRMSForce() const { return lastRMSForce; }

    /**
     * Set the Levenberg-Marquardt damping parameter.
     * Higher values make the algorithm more like gradient descent (more robust but slower).
     * Default is 0.01.
     *
     * @param lambda  the damping parameter
     */
    void setDamping(double lambda) { dampingFactor = lambda; }

    /**
     * Enable or disable line search (default: enabled).
     * Line search improves robustness but adds some overhead.
     *
     * @param enable  true to enable line search
     */
    void setLineSearch(bool enable) { useLineSearch = enable; }

    /**
     * Set the trust-region cap on maximum per-Cartesian-component step size
     * (nm).  Prevents huge Newton steps when the Hessian has soft directions
     * (small eigenvalues + numerical noise -> arbitrarily large H^-1 g).
     * Default is 0.05 nm to match RFOMinimizer.  Set to a large value (or 0)
     * to disable the cap.
     *
     * @param s  max per-component step (nm)
     */
    void setMaxStep(double s) { maxStep = s; }

    /**
     * Choose the inner solver used to compute the Newton search
     * direction.  Default is LMCholesky.  See the InnerSolver docstring
     * above for the tradeoff.
     */
    void setInnerSolver(InnerSolver s) { innerSolver = s; }
    InnerSolver getInnerSolver() const { return innerSolver; }

    /**
     * Enable the block-diagonal K-batch fast path for K > 1 replicas.
     *
     * When on: the assembled Hessian is treated as K independent 3N x 3N
     * blocks (one per particle group), and each block's Newton step is
     * solved independently.  Total cost is K*N^3 instead of (K*N)^3, and
     * the full K*N x K*N matrix is never allocated.  This is essential
     * for large K (e.g. K=88 harmonic-entropy pipelines) where the full
     * matrix would exhaust memory and Cholesky would take minutes.
     *
     * When off (default): the full 3(K*N) x 3(K*N) H is assembled and
     * solved as a single system.  Correct for any coupling pattern but
     * O(K^3) more expensive in both flops and memory.
     *
     * Caller must guarantee that the physical Hessian is block-diagonal
     * across particle groups (i.e. no forces couple replicas).  Standard
     * K-replica MM systems with IsolatedBondedForce +
     * IsolatedNonbondedForce + IsolatedGBSAForce (NONE/PAIRWISE) +
     * per-atom GridForce satisfy this; forces that share state across
     * replicas do not.
     */
    void setKBatchBlockDiagonal(bool enable) { kBatchBlockDiagonal = enable; }
    bool getKBatchBlockDiagonal() const { return kBatchBlockDiagonal; }

private:
    int lastIterations;
    double lastRMSForce;
    double dampingFactor;
    bool useLineSearch;
    double maxStep;
    InnerSolver innerSolver;
    bool kBatchBlockDiagonal;

    // Solve H * x = b using Cholesky decomposition (for positive definite H)
    // Returns false if H is not positive definite
    bool solveCholesky(const std::vector<double>& H, const std::vector<double>& b,
                       std::vector<double>& x, int n);

    // Solve with Levenberg-Marquardt damping: (H + lambda*I) * x = b
    void solveDamped(const std::vector<double>& H, const std::vector<double>& b,
                     std::vector<double>& x, int n, double lambda);

    // Compute RMS of a vector
    double computeRMS(const std::vector<double>& v);
};

}  // namespace GridForcePlugin

#endif /*OPENMM_NEWTONMINIMIZER_H_*/
