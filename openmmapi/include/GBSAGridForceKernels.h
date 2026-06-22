/* -------------------------------------------------------------------------- *
 *                              OpenMMGridForce                               *
 * -------------------------------------------------------------------------- */

#ifndef OPENMM_GBSAGRIDFORCE_KERNELS_H_
#define OPENMM_GBSAGRIDFORCE_KERNELS_H_

#include "GBSAGridForce.h"
#include "openmm/KernelImpl.h"
#include "openmm/Platform.h"
#include "openmm/System.h"
#include <string>

namespace GridForcePlugin {

/**
 * This kernel computes the GBSAGridForce.
 */
class CalcGBSAGridForceKernel : public OpenMM::KernelImpl {
public:
    static std::string Name() {
        return "CalcGBSAGridForce";
    }

    CalcGBSAGridForceKernel(std::string name, const OpenMM::Platform& platform)
        : OpenMM::KernelImpl(name, platform) {
    }

    /**
     * Initialize the kernel.
     *
     * @param system  The System this kernel will be applied to
     * @param force   The GBSAGridForce this kernel computes
     */
    virtual void initialize(const OpenMM::System& system, const GBSAGridForce& force) = 0;

    /**
     * Execute the kernel to calculate forces and/or energy.
     *
     * @param context       The context in which to execute
     * @param includeForces True if forces should be calculated
     * @param includeEnergy True if energy should be calculated
     * @return The potential energy
     */
    virtual double execute(OpenMM::ContextImpl& context,
                          bool includeForces,
                          bool includeEnergy) = 0;

    /**
     * Update parameters in the context after they have changed.
     *
     * @param context The context in which to update
     * @param force   The GBSAGridForce with updated parameters
     */
    virtual void updateParametersInContext(OpenMM::ContextImpl& context,
                                           const GBSAGridForce& force) = 0;

    /**
     * Get the GB energy for a specific particle group.
     * Called after execute() to retrieve per-group energies.
     */
    virtual double getGroupEnergy(int groupIndex) const = 0;

    /**
     * Get the ligand desolvation energy for a particle group.
     * This is the GB + optional SA energy for the ligand only.
     */
    virtual double getGroupLigandDesolvationEnergy(int groupIndex) const = 0;

    /**
     * Get the Born radii for atoms in a particle group.
     */
    virtual std::vector<double> getGroupBornRadii(int groupIndex) const = 0;

    /**
     * Compute the Hessian (second derivatives) via numerical finite differences.
     * Must be called after execute() so that internal state is valid.
     */
    virtual void computeHessian(OpenMM::ContextImpl& context) = 0;

    /**
     * Get per-atom 3x3 Hessian diagonal blocks [6 * N]: dxx, dyy, dzz, dxy, dxz, dyz.
     * Only valid after computeHessian().
     */
    virtual std::vector<double> getHessianBlocks() const = 0;

    /**
     * Get the full 3N x 3N Hessian matrix (row-major).
     * Only valid after computeHessian().
     */
    virtual std::vector<double> getFullHessian() const = 0;

    virtual void setSkipGroupEnergyDownload(bool) {}
    virtual void* getGroupEnergyDevicePointer() { return nullptr; }

    /**
     * Per-(group,atom) flags from the last execute(): 1 if the ligand atom fell
     * outside the desolvation grid (zero receptor screening). Empty if untracked.
     */
    virtual std::vector<int> getParticleOutOfBoundsFlags() const { return {}; }
};

} // namespace GridForcePlugin

#endif // OPENMM_GBSAGRIDFORCE_KERNELS_H_
