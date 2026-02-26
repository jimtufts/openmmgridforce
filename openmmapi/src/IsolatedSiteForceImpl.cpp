/* -------------------------------------------------------------------------- *
 *                              OpenMMGridForce                               *
 * -------------------------------------------------------------------------- */

#include "internal/IsolatedSiteForceImpl.h"
#include "IsolatedSiteForceKernels.h"
#include "openmm/Platform.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/OpenMMException.h"

using namespace OpenMM;
using namespace std;

namespace GridForcePlugin {

IsolatedSiteForceImpl::IsolatedSiteForceImpl(const IsolatedSiteForce& owner) : owner(owner) {
}

IsolatedSiteForceImpl::~IsolatedSiteForceImpl() {
}

void IsolatedSiteForceImpl::initialize(ContextImpl& context) {
    kernel = context.getPlatform().createKernel(CalcIsolatedSiteForceKernel::Name(), context);
    kernel.getAs<CalcIsolatedSiteForceKernel>().initialize(context.getSystem(), owner);
}

double IsolatedSiteForceImpl::calcForcesAndEnergy(ContextImpl& context, bool includeForces, bool includeEnergy, int groups) {
    if ((groups & (1 << owner.getForceGroup())) != 0) {
        double energy = kernel.getAs<CalcIsolatedSiteForceKernel>().execute(context, includeForces, includeEnergy);
        // Update per-group energies in the Force object
        int numGroups = owner.getNumParticleGroups();
        if (numGroups == 0) numGroups = 1;
        auto& mutableOwner = const_cast<IsolatedSiteForce&>(owner);
        mutableOwner.m_groupEnergies.resize(numGroups);
        for (int g = 0; g < numGroups; g++) {
            mutableOwner.m_groupEnergies[g] = kernel.getAs<CalcIsolatedSiteForceKernel>().getGroupEnergy(g);
        }
        return energy;
    }
    return 0.0;
}

std::vector<std::string> IsolatedSiteForceImpl::getKernelNames() {
    std::vector<std::string> names;
    names.push_back(CalcIsolatedSiteForceKernel::Name());
    return names;
}

void IsolatedSiteForceImpl::updateParametersInContext(ContextImpl& context) {
    kernel.getAs<CalcIsolatedSiteForceKernel>().copyParametersToContext(context, owner);
}

}  // namespace GridForcePlugin
