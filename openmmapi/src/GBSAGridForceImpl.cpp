/* -------------------------------------------------------------------------- *
 *                              OpenMMGridForce                               *
 * -------------------------------------------------------------------------- */

#include "internal/GBSAGridForceImpl.h"
#include "GBSAGridForceKernels.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/ContextImpl.h"

using namespace GridForcePlugin;
using namespace OpenMM;
using namespace std;

GBSAGridForceImpl::GBSAGridForceImpl(const GBSAGridForce& owner) : owner(owner) {
}

GBSAGridForceImpl::~GBSAGridForceImpl() {
}

void GBSAGridForceImpl::initialize(ContextImpl& context) {
    // Validate force configuration
    if (owner.getNumAtoms() == 0) {
        throw OpenMMException("GBSAGridForce: no atoms defined");
    }
    if (!owner.getDesolvationGrid()) {
        throw OpenMMException("GBSAGridForce: no desolvation grid set");
    }

    // Create kernel
    kernel = context.getPlatform().createKernel(CalcGBSAGridForceKernel::Name(), context);
    kernel.getAs<CalcGBSAGridForceKernel>().initialize(context.getSystem(), owner);
}

double GBSAGridForceImpl::calcForcesAndEnergy(ContextImpl& context,
                                               bool includeForces,
                                               bool includeEnergy,
                                               int groups) {
    if ((groups & (1 << owner.getForceGroup())) != 0) {
        double energy = kernel.getAs<CalcGBSAGridForceKernel>().execute(context, includeForces, includeEnergy);

        // Copy per-group energies from kernel to Force object (mutable member)
        int numGroups = owner.getNumParticleGroups();
        if (numGroups == 0) numGroups = 1;  // Default single group
        owner.groupEnergies.resize(numGroups);
        for (int g = 0; g < numGroups; g++) {
            owner.groupEnergies[g] = kernel.getAs<CalcGBSAGridForceKernel>().getGroupEnergy(g);
        }

        return energy;
    }
    return 0.0;
}

vector<string> GBSAGridForceImpl::getKernelNames() {
    vector<string> names;
    names.push_back(CalcGBSAGridForceKernel::Name());
    return names;
}

void GBSAGridForceImpl::updateParametersInContext(ContextImpl& context) {
    kernel.getAs<CalcGBSAGridForceKernel>().updateParametersInContext(context, owner);
}
