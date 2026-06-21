/* -------------------------------------------------------------------------- *
 *                              OpenMMGridForce                               *
 * -------------------------------------------------------------------------- *
 * CPU platform implementation of MultiGroupNUTSIntegrator kernel.            *
 * -------------------------------------------------------------------------- */

#include "CpuMultiGroupNUTSKernels.h"
#include "openmm/cpu/CpuPlatform.h"
#include "openmm/internal/ThreadPool.h"

using namespace OpenMM;
using namespace std;

namespace GridForcePlugin {

void CpuIntegrateMultiGroupNUTSStepKernel::forEachGroup(
        ContextImpl& context, const std::function<void(int)>& body) {

    ThreadPool& pool = CpuPlatform::getPlatformData(context).threads;

    // Each group uses only its own atoms, per-group buffers, and per-group RNG
    // stream, so distributing groups across threads yields identical results for
    // any thread count.
    pool.execute([&](ThreadPool& p, int threadIndex) {
        int nThreads = p.getNumThreads();
        for (int k = threadIndex; k < numGroups; k += nThreads)
            body(k);
    });
    pool.waitForThreads();
}

}  // namespace GridForcePlugin
