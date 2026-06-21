/* -------------------------------------------------------------------------- *
 *                              OpenMMGridForce                               *
 * -------------------------------------------------------------------------- *
 * CPU platform implementation of GBSAGridForce kernel.                       *
 * -------------------------------------------------------------------------- */

#include "CpuGBSAGridForceKernels.h"
#include "openmm/cpu/CpuPlatform.h"
#include "openmm/internal/ThreadPool.h"

using namespace OpenMM;
using namespace std;

namespace GridForcePlugin {

void CpuCalcGBSAGridForceKernel::runGroups(
        ContextImpl& context, vector<Vec3>& posData, vector<Vec3>& forceData,
        bool includeForces, bool includeEnergy) {

    CpuPlatform::PlatformData& data = CpuPlatform::getPlatformData(context);
    ThreadPool& pool = data.threads;

    // Distribute the disjoint particle groups across worker threads. Each group
    // writes only its own atoms' forces and its own energy slot, so concurrent
    // groups never conflict; execute() then sums the per-group energies in order.
    pool.execute([&](ThreadPool& p, int threadIndex) {
        int nThreads = p.getNumThreads();
        for (int g = threadIndex; g < numParticleGroups; g += nThreads)
            computeGroup(g, posData, forceData, includeForces, includeEnergy);
    });
    pool.waitForThreads();
}

void CpuCalcGBSAGridForceKernel::parallelFor(
        ContextImpl& context, int count, const std::function<void(int)>& body) {

    CpuPlatform::PlatformData& data = CpuPlatform::getPlatformData(context);
    ThreadPool& pool = data.threads;

    pool.execute([&](ThreadPool& p, int threadIndex) {
        for (int i = threadIndex; i < count; i += p.getNumThreads())
            body(i);
    });
    pool.waitForThreads();
}

}  // namespace GridForcePlugin
