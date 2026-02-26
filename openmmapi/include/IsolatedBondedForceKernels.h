#ifndef OPENMM_ISOLATEDBONDEDFORCE_KERNELS_H_
#define OPENMM_ISOLATEDBONDEDFORCE_KERNELS_H_

/* -------------------------------------------------------------------------- *
 *                              OpenMMGridForce                               *
 * -------------------------------------------------------------------------- */

#include <string>
#include <vector>

#include "IsolatedBondedForce.h"
#include "openmm/KernelImpl.h"
#include "openmm/Platform.h"
#include "openmm/System.h"

namespace GridForcePlugin {

/**
 * This kernel is invoked by IsolatedBondedForce to calculate bonded forces
 * and energies for isolated particle groups.
 */
class CalcIsolatedBondedForceKernel : public OpenMM::KernelImpl {
public:
    static std::string Name() {
        return "CalcIsolatedBondedForce";
    }

    CalcIsolatedBondedForceKernel(std::string name, const OpenMM::Platform& platform)
        : OpenMM::KernelImpl(name, platform) {
    }

    virtual void initialize(const OpenMM::System& system,
                           const IsolatedBondedForce& force) = 0;

    virtual double execute(OpenMM::ContextImpl& context, bool includeForces, bool includeEnergy) = 0;

    virtual void copyParametersToContext(OpenMM::ContextImpl& context,
                                        const IsolatedBondedForce& force) = 0;

    virtual double getGroupEnergy(int groupIndex) const = 0;

    /**
     * Compute the Hessian (second derivatives) for bonded interactions
     * using a specific particle group's positions.
     * Returns flattened 3N x 3N matrix (N = template atom count).
     */
    virtual std::vector<double> computeHessian(OpenMM::ContextImpl& context, int groupIndex) = 0;

    /**
     * Compute scalar force constants for each internal coordinate
     * using a specific particle group's positions.
     */
    virtual std::vector<double> computeInternalForceConstants(OpenMM::ContextImpl& context, int groupIndex) = 0;

    /**
     * Control whether group energy downloads are skipped after execute().
     * When true, energies are computed on GPU but not downloaded to host.
     * Used by NUTS integrator to avoid CPU-GPU sync points in the inner loop.
     */
    virtual void setSkipGroupEnergyDownload(bool) {}
    virtual void* getGroupEnergyDevicePointer() { return nullptr; }
};

}  // namespace GridForcePlugin

#endif /* OPENMM_ISOLATEDBONDEDFORCE_KERNELS_H_*/
