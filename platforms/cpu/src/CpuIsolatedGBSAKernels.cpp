/* -------------------------------------------------------------------------- *
 *                              OpenMMGridForce                               *
 * -------------------------------------------------------------------------- *
 * CPU platform implementation of IsolatedGBSAForce kernel.                   *
 * -------------------------------------------------------------------------- */

#include "CpuIsolatedGBSAKernels.h"
#include "openmm/cpu/CpuPlatform.h"
#include "openmm/internal/ThreadPool.h"

using namespace OpenMM;
using namespace std;

namespace GridForcePlugin {

void CpuCalcIsolatedGBSAForceKernel::runGroups(
        ContextImpl& context, vector<Vec3>& posData, vector<Vec3>& forceData,
        bool includeForces, bool includeEnergy) {

    CpuPlatform::PlatformData& data = CpuPlatform::getPlatformData(context);
    ThreadPool& pool = data.threads;

    // Distribute the disjoint particle groups across worker threads. Each group
    // writes only its own atoms' forces and its own per-group buffers, so
    // concurrent groups never conflict; execute() then sums the per-group
    // energies in order.
    pool.execute([&](ThreadPool& p, int threadIndex) {
        int nThreads = p.getNumThreads();
        for (int g = threadIndex; g < numParticleGroups; g += nThreads)
            computeGroup(g, posData, forceData, includeForces, includeEnergy);
    });
    pool.waitForThreads();
}

void CpuCalcIsolatedGBSAForceKernel::parallelFor(
        ContextImpl& context, int count, const std::function<void(int)>& body) {

    CpuPlatform::PlatformData& data = CpuPlatform::getPlatformData(context);
    ThreadPool& pool = data.threads;

    // Each index writes only its own disjoint region of the output, so the
    // strided indices never conflict.
    pool.execute([&](ThreadPool& p, int threadIndex) {
        int nThreads = p.getNumThreads();
        for (int i = threadIndex; i < count; i += nThreads)
            body(i);
    });
    pool.waitForThreads();
}

}  // namespace GridForcePlugin
