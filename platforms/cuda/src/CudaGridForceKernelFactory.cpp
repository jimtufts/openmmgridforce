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

extern "C" OPENMM_EXPORT void registerGridForceCudaKernelFactories() {
    std::cout << "GridForceCUDA: registerGridForceCudaKernelFactories called" << std::endl;
    bool foundCuda = false;
    try {
        // Find first CUDA platform
        for (int i = 0; i < Platform::getNumPlatforms(); i++) {
            Platform& platform = Platform::getPlatform(i);
            if (platform.getName() == "CUDA") {
                std::cout << "GridForceCUDA: Found CUDA platform" << std::endl;
                foundCuda = true;
                try {
                    CudaGridForceKernelFactory* factory = new CudaGridForceKernelFactory();
                    platform.registerKernelFactory(CalcGridForceKernel::Name(), factory);
                    std::cout << "GridForceCUDA: Successfully registered kernel factory" << std::endl;
                }
                catch (std::exception& ex) {
                    std::cout << "GridForceCUDA: Error registering kernel factory: " << ex.what() << std::endl;
                    throw;
                }
                break;
            }
        }
        if (!foundCuda) {
            std::cout << "GridForceCUDA: No CUDA platform found" << std::endl;
        }
    }
    catch (std::exception& ex) {
        std::cout << "GridForceCUDA: Error during plugin registration: " << ex.what() << std::endl;
    }
}

// Remove other registration functions that might be causing duplicates
extern "C" OPENMM_EXPORT void registerPlatforms() {
    // Do nothing
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

KernelImpl* CudaGridForceKernelFactory::createKernelImpl(std::string name, const Platform& platform, ContextImpl& context) const {
    std::cout << "GridForceCUDA: Attempting to create kernel: " << name << std::endl;
    std::cout << "GridForceCUDA: Expected kernel name: " << CalcGridForceKernel::Name() << std::endl;

    CudaContext& cu = *static_cast<CudaPlatform::PlatformData*>(context.getPlatformData())->contexts[0];
    if (name == CalcGridForceKernel::Name())
        return new CudaCalcGridForceKernel(name, platform, cu);
    throw OpenMMException((std::string("Tried to create kernel with illegal kernel name '") + name + "'").c_str());
}
