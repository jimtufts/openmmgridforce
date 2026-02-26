#ifndef OPENMM_ISOLATEDSITEFORCE_KERNELS_H_
#define OPENMM_ISOLATEDSITEFORCE_KERNELS_H_

/* -------------------------------------------------------------------------- *
 *                              OpenMMGridForce                               *
 * -------------------------------------------------------------------------- */

#include "IsolatedSiteForce.h"
#include "openmm/KernelImpl.h"
#include "openmm/Platform.h"
#include "openmm/System.h"
#include <string>
#include <vector>

namespace GridForcePlugin {

/**
 * Kernel interface for IsolatedSiteForce.
 *
 * Computes per-group flat-bottom sphere restraint from COM to site center.
 * Each platform (CUDA, Reference) provides a concrete implementation.
 */
class CalcIsolatedSiteForceKernel : public OpenMM::KernelImpl {
public:
    static std::string Name() {
        return "CalcIsolatedSiteForce";
    }

    CalcIsolatedSiteForceKernel(std::string name, const OpenMM::Platform& platform)
        : OpenMM::KernelImpl(name, platform) {}

    /**
     * Initialize the kernel with force parameters and particle groups.
     */
    virtual void initialize(const OpenMM::System& system,
                           const IsolatedSiteForce& force) = 0;

    /**
     * Execute the kernel: compute per-group COM, energy, and forces.
     * Returns total energy summed across all groups.
     */
    virtual double execute(OpenMM::ContextImpl& context,
                          bool includeForces,
                          bool includeEnergy) = 0;

    /**
     * Copy updated parameters (site center, radius, k, scaling) to the kernel.
     */
    virtual void copyParametersToContext(OpenMM::ContextImpl& context,
                                        const IsolatedSiteForce& force) = 0;

    /**
     * Get the energy of a specific particle group from the last execute().
     */
    virtual double getGroupEnergy(int groupIndex) const = 0;

    virtual void setSkipGroupEnergyDownload(bool) {}
    virtual void* getGroupEnergyDevicePointer() { return nullptr; }
};

}  // namespace GridForcePlugin

#endif /*OPENMM_ISOLATEDSITEFORCE_KERNELS_H_*/
