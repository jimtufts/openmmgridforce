#ifndef OPENMM_ISOLATEDSITEFORCE_IMPL_H_
#define OPENMM_ISOLATEDSITEFORCE_IMPL_H_

/* -------------------------------------------------------------------------- *
 *                              OpenMMGridForce                               *
 * -------------------------------------------------------------------------- */

#include "IsolatedSiteForce.h"
#include "IsolatedSiteForceKernels.h"
#include "openmm/Kernel.h"
#include "openmm/internal/ForceImpl.h"
#include <string>
#include <vector>

namespace GridForcePlugin {

class OPENMM_EXPORT_GRIDFORCE IsolatedSiteForceImpl : public OpenMM::ForceImpl {
public:
    IsolatedSiteForceImpl(const IsolatedSiteForce& owner);
    ~IsolatedSiteForceImpl();
    void initialize(OpenMM::ContextImpl& context);
    const IsolatedSiteForce& getOwner() const {
        return owner;
    }
    void updateContextState(OpenMM::ContextImpl& context, bool& forcesInvalid) {
    }
    double calcForcesAndEnergy(OpenMM::ContextImpl& context,
                               bool includeForces,
                               bool includeEnergy,
                               int groups);
    std::map<std::string, double> getDefaultParameters() {
        return std::map<std::string, double>();
    }
    std::vector<std::string> getKernelNames();

    void updateParametersInContext(OpenMM::ContextImpl& context);

    void setSkipGroupEnergyDownload(bool skip) {
        kernel.getAs<CalcIsolatedSiteForceKernel>().setSkipGroupEnergyDownload(skip);
    }
    void* getGroupEnergyDevicePointer() {
        return kernel.getAs<CalcIsolatedSiteForceKernel>().getGroupEnergyDevicePointer();
    }

private:
    const IsolatedSiteForce& owner;
    OpenMM::Kernel kernel;
};

}  // namespace GridForcePlugin

#endif /*OPENMM_ISOLATEDSITEFORCE_IMPL_H_*/
