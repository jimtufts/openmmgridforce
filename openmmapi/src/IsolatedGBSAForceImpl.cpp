/* -------------------------------------------------------------------------- *
 *                              OpenMMGridForce                               *
 * -------------------------------------------------------------------------- */

#include "internal/IsolatedGBSAForceImpl.h"
#include "IsolatedGBSAForceKernels.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/ContextImpl.h"

using namespace GridForcePlugin;
using namespace OpenMM;
using namespace std;

IsolatedGBSAForceImpl::IsolatedGBSAForceImpl(const IsolatedGBSAForce& owner) : owner(owner) {
}

IsolatedGBSAForceImpl::~IsolatedGBSAForceImpl() {
}

void IsolatedGBSAForceImpl::initialize(ContextImpl& context) {
    // Validate force configuration
    if (owner.getNumAtoms() == 0) {
        throw OpenMMException("IsolatedGBSAForce: no atoms defined");
    }

    // Validate receptor mode configuration
    IsolatedGBSAForce::ReceptorMode mode = owner.getReceptorMode();
    if (mode == IsolatedGBSAForce::GRID && !owner.getDesolvationGrid()) {
        throw OpenMMException("IsolatedGBSAForce: GRID mode requires a desolvation grid");
    }
    if (mode == IsolatedGBSAForce::PAIRWISE) {
        if (owner.getNumReceptorAtoms() == 0) {
            throw OpenMMException("IsolatedGBSAForce: PAIRWISE mode requires receptor atoms");
        }
        if (owner.getReceptorPositions().size() != static_cast<size_t>(owner.getNumReceptorAtoms() * 3)) {
            throw OpenMMException("IsolatedGBSAForce: receptor positions not set for PAIRWISE mode");
        }
    }

    // Create kernel
    kernel = context.getPlatform().createKernel(CalcIsolatedGBSAForceKernel::Name(), context);
    kernel.getAs<CalcIsolatedGBSAForceKernel>().initialize(context.getSystem(), owner);
}

double IsolatedGBSAForceImpl::calcForcesAndEnergy(ContextImpl& context,
                                                   bool includeForces,
                                                   bool includeEnergy,
                                                   int groups) {
    if ((groups & (1 << owner.getForceGroup())) != 0) {
        double energy = kernel.getAs<CalcIsolatedGBSAForceKernel>().execute(context, includeForces, includeEnergy);

        // Copy per-group energies from kernel to Force object (mutable members)
        int numGroups = owner.getNumParticleGroups();
        if (numGroups == 0) numGroups = 1;  // Default single group

        owner.groupEnergies.resize(numGroups);
        owner.groupLigandSelfEnergies.resize(numGroups);
        owner.groupReceptorContributions.resize(numGroups);
        owner.groupReceptorDesolvations.resize(numGroups);
        owner.groupCrossTermEnergies.resize(numGroups);
        owner.groupBornRadii.resize(numGroups);
        owner.groupAtomEnergies.resize(numGroups);

        // Resize receptor Born radii for PAIRWISE mode
        if (owner.getReceptorMode() == IsolatedGBSAForce::PAIRWISE) {
            owner.groupReceptorBornRadii.resize(numGroups);
        }

        for (int g = 0; g < numGroups; g++) {
            owner.groupEnergies[g] = kernel.getAs<CalcIsolatedGBSAForceKernel>().getGroupEnergy(g);
            owner.groupLigandSelfEnergies[g] = kernel.getAs<CalcIsolatedGBSAForceKernel>().getGroupLigandSelfEnergy(g);
            owner.groupReceptorContributions[g] = kernel.getAs<CalcIsolatedGBSAForceKernel>().getGroupReceptorContribution(g);
            owner.groupReceptorDesolvations[g] = kernel.getAs<CalcIsolatedGBSAForceKernel>().getGroupReceptorDesolvation(g);
            owner.groupCrossTermEnergies[g] = kernel.getAs<CalcIsolatedGBSAForceKernel>().getGroupCrossTermEnergy(g);
            // Born radii and atom energies: skip expensive GPU downloads during force evaluation.
            // These are diagnostic arrays — download on-demand via getGroupBornRadii() / getReceptorBornRadii().
        }

        return energy;
    }
    return 0.0;
}

vector<string> IsolatedGBSAForceImpl::getKernelNames() {
    vector<string> names;
    names.push_back(CalcIsolatedGBSAForceKernel::Name());
    return names;
}

void IsolatedGBSAForceImpl::updateParametersInContext(ContextImpl& context) {
    kernel.getAs<CalcIsolatedGBSAForceKernel>().updateParametersInContext(context, owner);
}

vector<double> IsolatedGBSAForceImpl::getParticleGroupUnscaledEnergies() {
    return kernel.getAs<CalcIsolatedGBSAForceKernel>().getParticleGroupUnscaledEnergies();
}

vector<double> IsolatedGBSAForceImpl::computeHessian(ContextImpl& context) {
    return kernel.getAs<CalcIsolatedGBSAForceKernel>().computeHessian(context);
}
