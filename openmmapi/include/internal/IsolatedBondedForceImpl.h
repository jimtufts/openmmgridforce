#ifndef OPENMM_ISOLATEDBONDEDFORCE_IMPL_H_
#define OPENMM_ISOLATEDBONDEDFORCE_IMPL_H_

/* -------------------------------------------------------------------------- *
 *                              OpenMMGridForce                               *
 * -------------------------------------------------------------------------- */

#include "IsolatedBondedForce.h"
#include "openmm/Kernel.h"
#include "openmm/internal/ForceImpl.h"
#include <string>
#include <vector>

namespace GridForcePlugin {

class OPENMM_EXPORT_GRIDFORCE IsolatedBondedForceImpl : public OpenMM::ForceImpl {
public:
    IsolatedBondedForceImpl(const IsolatedBondedForce& owner);
    ~IsolatedBondedForceImpl();
    void initialize(OpenMM::ContextImpl& context);
    const IsolatedBondedForce& getOwner() const {
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

    std::vector<double> computeHessian(OpenMM::ContextImpl& context, int groupIndex);
    std::vector<double> computeInternalForceConstants(OpenMM::ContextImpl& context, int groupIndex);

private:
    const IsolatedBondedForce& owner;
    OpenMM::Kernel kernel;
};

}  // namespace GridForcePlugin

#endif /*OPENMM_ISOLATEDBONDEDFORCE_IMPL_H_*/
