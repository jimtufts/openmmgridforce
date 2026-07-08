#ifndef OPENMM_GRIDFORCE_LINEAR_SOLVER_KERNELS_H_
#define OPENMM_GRIDFORCE_LINEAR_SOLVER_KERNELS_H_

/* -------------------------------------------------------------------------- *
 *                              OpenMMGridForce                               *
 * -------------------------------------------------------------------------- *
 * Platform-dispatchable dense-linear-solver kernel used by NewtonMinimizer.  *
 * Currently provides Cholesky with Levenberg-Marquardt diagonal shift.       *
 * -------------------------------------------------------------------------- */

#include "openmm/KernelImpl.h"
#include "internal/windowsExportGridForce.h"
#include <string>
#include <vector>

namespace GridForcePlugin {

class OPENMM_EXPORT_GRIDFORCE CalcLinearSolverKernel : public OpenMM::KernelImpl {
public:
    static std::string Name() { return "GridForceLinearSolver"; }

    CalcLinearSolverKernel(std::string name, const OpenMM::Platform& platform)
        : KernelImpl(name, platform) {}

    virtual void initialize() = 0;

    /**
     * Solve (H + lambda*I) x = b via Cholesky, ramping lambda in
     * {0, 1, 10, ..., lambdaMax} until factorization succeeds.  H is
     * dense row-major n x n on host input; on return x contains the
     * solution and lambdaUsedOut records the final shift (0 = H was PD).
     *
     * Returns:
     *   0 succeeded on first try (lambda = 0)
     *   1 succeeded after ramping
     *   2 fallback: preconditioned steepest descent (Cholesky never
     *     succeeded before lambdaMax)
     */
    virtual int solveLMCholesky(const std::vector<double>& H,
                                 const std::vector<double>& b,
                                 std::vector<double>& x, int n,
                                 double lambdaMax,
                                 double& lambdaUsedOut) = 0;
};

}  // namespace GridForcePlugin

#endif  // OPENMM_GRIDFORCE_LINEAR_SOLVER_KERNELS_H_
