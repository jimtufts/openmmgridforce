#ifndef OPENMM_ISOLATEDGBSAFORCE_IMPL_H_
#define OPENMM_ISOLATEDGBSAFORCE_IMPL_H_

/* -------------------------------------------------------------------------- *
 *                              OpenMMGridForce                               *
 * -------------------------------------------------------------------------- */

#include "IsolatedGBSAForce.h"
#include "openmm/Kernel.h"
#include "openmm/internal/ForceImpl.h"
#include <string>
#include <vector>

namespace GridForcePlugin {

/**
 * Internal implementation of IsolatedGBSAForce.
 */
class OPENMM_EXPORT_GRIDFORCE IsolatedGBSAForceImpl : public OpenMM::ForceImpl {
public:
    IsolatedGBSAForceImpl(const IsolatedGBSAForce& owner);
    ~IsolatedGBSAForceImpl();

    void initialize(OpenMM::ContextImpl& context);

    const IsolatedGBSAForce& getOwner() const {
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

    /**
     * Compute the Hessian (second derivatives) for the GBSA force.
     */
    std::vector<double> computeHessian(OpenMM::ContextImpl& context);

private:
    const IsolatedGBSAForce& owner;
    OpenMM::Kernel kernel;
};

}  // namespace GridForcePlugin

#endif /*OPENMM_ISOLATEDGBSAFORCE_IMPL_H_*/
