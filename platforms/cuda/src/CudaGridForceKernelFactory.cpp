#include <exception>
#include <iostream>

#include "CudaGridForceKernelFactory.h"
#include "CudaGridForceKernels.h"
#include "GridForce.h"
#include "openmm/cuda/CudaContext.h"
#include "openmm/internal/windowsExport.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/OpenMMException.h"

using namespace GridForcePlugin;
using namespace OpenMM;

static void registerKernelsImpl(Platform& platform) {
    std::cout << "GridForceCUDA: Registering CUDA kernels" << std::endl;
    CudaGridForceKernelFactory* factory = new CudaGridForceKernelFactory();
    platform.registerKernelFactory(CalcGridForceKernel::Name(), factory);
}

extern "C" OPENMM_EXPORT void registerKernelFactories() {
    std::cout << "GridForceCUDA: registerKernelFactories called" << std::endl;
    try {
        Platform& platform = Platform::getPlatformByName("CUDA");
        registerKernelsImpl(platform);
    }
    catch (std::exception& ex) {
        std::cout << "GridForceCUDA: Unable to register CUDA kernels: " << ex.what() << std::endl;
    }
}

extern "C" OPENMM_EXPORT void registerGridForceCudaKernelFactories() {
    std::cout << "GridForceCUDA: registerGridForceCudaKernelFactories called" << std::endl;
    try {
        if (Platform::getNumPlatforms() > 0) {
            std::cout << "GridForceCUDA: CUDA platform already registered" << std::endl;
        }
        else {
            std::cout << "GridForceCUDA: Registering CUDA platform" << std::endl;
            Platform::registerPlatform(new CudaPlatform());
        }
        registerKernelFactories();
    }
    catch (std::exception& ex) {
        std::cout << "GridForceCUDA: Error registering plugin: " << ex.what() << std::endl;
    }
}

KernelImpl* CudaGridForceKernelFactory::createKernelImpl(std::string name, const Platform& platform, ContextImpl& context) const {
    std::cout << "GridForceCUDA: Creating kernel implementation for: " << name << std::endl;
    CudaContext& cu = *static_cast<CudaPlatform::PlatformData*>(context.getPlatformData())->contexts[0];
    if (name == CalcGridForceKernel::Name()) {
        std::cout << "GridForceCUDA: Creating CudaCalcGridForceKernel" << std::endl;
        return new CudaCalcGridForceKernel(name, platform, cu);
    }
    throw OpenMMException((std::string("Tried to create kernel with illegal kernel name '") + name + "'").c_str());
}
