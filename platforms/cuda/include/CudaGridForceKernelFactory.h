#ifndef OPENMM_CUDAGRIDFORCEKERNELFACTORY_H_
#define OPENMM_CUDAGRIDFORCEKERNELFACTORY_H_

#include "openmm/KernelFactory.h"

namespace OpenMM {

/**
 * This KernelFactory creates kernels for the CUDA implementation of the GridForce plugin.
 */

class CudaGridForceKernelFactory : public KernelFactory {
public:
    KernelImpl* createKernelImpl(std::string name, const Platform& platform, ContextImpl& context) const;
};

} // namespace OpenMM

#endif /*OPENMM_CUDAGRIDFORCEKERNELFACTORY_H_*/
