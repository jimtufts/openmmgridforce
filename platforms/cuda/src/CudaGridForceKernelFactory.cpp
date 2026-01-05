#include <exception>
#include <iostream>

#include "CudaGridForceKernelFactory.h"
#include "CudaGridForceKernels.h"
#include "CudaIsolatedNonbondedKernels.h"
#include "GridForce.h"
#include "IsolatedNonbondedForce.h"
#include "openmm/cuda/CudaContext.h"
#include "openmm/internal/windowsExport.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/OpenMMException.h"

using namespace GridForcePlugin;
using namespace OpenMM;

extern "C" OPENMM_EXPORT void registerKernelFactories() {
    try {
        Platform& platform = Platform::getPlatformByName("CUDA");
        CudaGridForceKernelFactory* factory = new CudaGridForceKernelFactory();
        platform.registerKernelFactory(CalcGridForceKernel::Name(), factory);
        platform.registerKernelFactory(CalcIsolatedNonbondedForceKernel::Name(), factory);
    }
    catch (...) {
    }
}

extern "C" OPENMM_EXPORT void registerPlatforms() {
}

KernelImpl* CudaGridForceKernelFactory::createKernelImpl(std::string name, const Platform& platform, ContextImpl& context) const {
    CudaContext& cu = *static_cast<CudaPlatform::PlatformData*>(context.getPlatformData())->contexts[0];
    if (name == CalcGridForceKernel::Name())
        return new CudaCalcGridForceKernel(name, platform, cu);
    if (name == CalcIsolatedNonbondedForceKernel::Name())
        return new CudaCalcIsolatedNonbondedForceKernel(name, platform, cu);
    throw OpenMMException((std::string("Tried to create kernel with illegal kernel name '") + name + "'").c_str());
}
