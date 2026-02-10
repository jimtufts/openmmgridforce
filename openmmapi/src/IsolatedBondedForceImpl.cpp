/* -------------------------------------------------------------------------- *
 *                              OpenMMGridForce                               *
 * -------------------------------------------------------------------------- */

#include "internal/IsolatedBondedForceImpl.h"
#include "IsolatedBondedForceKernels.h"
#include "openmm/Platform.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/OpenMMException.h"

using namespace OpenMM;
using namespace std;

namespace GridForcePlugin {

IsolatedBondedForceImpl::IsolatedBondedForceImpl(const IsolatedBondedForce& owner) : owner(owner) {
}

IsolatedBondedForceImpl::~IsolatedBondedForceImpl() {
}

void IsolatedBondedForceImpl::initialize(ContextImpl& context) {
    kernel = context.getPlatform().createKernel(CalcIsolatedBondedForceKernel::Name(), context);
    kernel.getAs<CalcIsolatedBondedForceKernel>().initialize(context.getSystem(), owner);
}

double IsolatedBondedForceImpl::calcForcesAndEnergy(ContextImpl& context, bool includeForces, bool includeEnergy, int groups) {
    if ((groups & (1 << owner.getForceGroup())) != 0) {
        double energy = kernel.getAs<CalcIsolatedBondedForceKernel>().execute(context, includeForces, includeEnergy);
        // Update per-group energies in the Force object
        int numGroups = owner.getNumParticleGroups();
        if (numGroups == 0) numGroups = 1;
        auto& mutableOwner = const_cast<IsolatedBondedForce&>(owner);
        mutableOwner.m_groupEnergies.resize(numGroups);
        for (int g = 0; g < numGroups; g++) {
            mutableOwner.m_groupEnergies[g] = kernel.getAs<CalcIsolatedBondedForceKernel>().getGroupEnergy(g);
        }
        return energy;
    }
    return 0.0;
}

std::vector<std::string> IsolatedBondedForceImpl::getKernelNames() {
    std::vector<std::string> names;
    names.push_back(CalcIsolatedBondedForceKernel::Name());
    return names;
}

void IsolatedBondedForceImpl::updateParametersInContext(ContextImpl& context) {
    kernel.getAs<CalcIsolatedBondedForceKernel>().copyParametersToContext(context, owner);
}

std::vector<double> IsolatedBondedForceImpl::computeHessian(ContextImpl& context, int groupIndex) {
    return kernel.getAs<CalcIsolatedBondedForceKernel>().computeHessian(context, groupIndex);
}

std::vector<double> IsolatedBondedForceImpl::computeInternalForceConstants(ContextImpl& context, int groupIndex) {
    return kernel.getAs<CalcIsolatedBondedForceKernel>().computeInternalForceConstants(context, groupIndex);
}

}  // namespace GridForcePlugin
