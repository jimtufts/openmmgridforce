#ifndef OPENMM_CUDAGRIDFORCEKERNELSOURCES_H_
#define OPENMM_CUDAGRIDFORCEKERNELSOURCES_H_

#include <string>

namespace GridForcePlugin {

/**
 * This class is a central holding place for the source code of CUDA kernels.
 */

class CudaGridForceKernelSources {
public:
    static std::string gridForceKernel;
};

} // namespace GridForcePlugin

#endif /*OPENMM_CUDAGRIDFORCEKERNELSOURCES_H_*/
