/* -------------------------------------------------------------------------- *
 *                              OpenMMGridForce                               *
 * -------------------------------------------------------------------------- *
 * CPU platform implementation of GridForce kernel.                           *
 * -------------------------------------------------------------------------- */

#include "CpuGridForceKernels.h"
#include "openmm/cpu/CpuPlatform.h"
#include "openmm/internal/ThreadPool.h"

using namespace OpenMM;
using namespace std;

namespace GridForcePlugin {

void CpuCalcGridForceKernel::runAtoms(
        ContextImpl& context, vector<Vec3>& posData, vector<Vec3>& forceData,
        bool includeForces, bool includeEnergy) {

    int natom_lig = g_scaling_factors.size();
    g_atomEnergyContribution.resize(natom_lig);
    g_atomUnscaledContribution.resize(natom_lig);
    g_atomGroupIdx.resize(natom_lig);

    CpuPlatform::PlatformData& data = CpuPlatform::getPlatformData(context);
    ThreadPool& pool = data.threads;

    // Distribute the ligand atoms across worker threads. Each atom writes only
    // its own force entry and its own per-atom result slots, so concurrent atoms
    // never conflict; execute() then reduces the per-atom contributions in order.
    pool.execute([&](ThreadPool& p, int threadIndex) {
        int nThreads = p.getNumThreads();
        for (int ia = threadIndex; ia < natom_lig; ia += nThreads)
            computeAtom(ia, posData, forceData, includeForces, includeEnergy,
                        g_atomEnergyContribution[ia], g_atomUnscaledContribution[ia],
                        g_atomGroupIdx[ia]);
    });
    pool.waitForThreads();
}

}  // namespace GridForcePlugin
