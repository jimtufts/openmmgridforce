/* -------------------------------------------------------------------------- *
 *                              OpenMMGridForce                               *
 * -------------------------------------------------------------------------- */

#ifndef OPENMM_GBSAGRIDFORCE_IMPL_H_
#define OPENMM_GBSAGRIDFORCE_IMPL_H_

#include "GBSAGridForce.h"
#include "openmm/Kernel.h"
#include "openmm/internal/ForceImpl.h"
#include <string>
#include <vector>
#include <map>

namespace GridForcePlugin {

/**
 * Internal implementation of GBSAGridForce.
 */
class OPENMM_EXPORT_GRIDFORCE GBSAGridForceImpl : public OpenMM::ForceImpl {
public:
    GBSAGridForceImpl(const GBSAGridForce& owner);
    ~GBSAGridForceImpl();

    void initialize(OpenMM::ContextImpl& context);

    const GBSAGridForce& getOwner() const {
        return owner;
    }

    void updateContextState(OpenMM::ContextImpl& context, bool& forcesInvalid) {
        // This force doesn't update context state
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

    // Hessian support
    void computeHessian(OpenMM::ContextImpl& context);
    std::vector<double> getHessianBlocks();
    std::vector<double> getFullHessian();

private:
    const GBSAGridForce& owner;
    OpenMM::Kernel kernel;
};

} // namespace GridForcePlugin

#endif // OPENMM_GBSAGRIDFORCE_IMPL_H_
