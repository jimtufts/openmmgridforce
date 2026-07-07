#include <exception>
#include <iostream>

#include "CudaGridForceKernelFactory.h"
#include "CudaGridForceKernels.h"
#include "CudaIsolatedNonbondedKernels.h"
#include "CudaIsolatedBondedKernels.h"
#include "CudaIsolatedSiteKernels.h"
#include "CudaGBSAGridForceKernels.h"
#include "CudaIsolatedGBSAKernels.h"
#include "CudaMultiGroupHMCKernels.h"
#include "CudaMultiGroupNUTSKernels.h"
#include "MultiGroupHMCKernels.h"
#include "MultiGroupNUTSKernels.h"
#include "PluginCompatMinimizeKernel.h"
#include "openmm/kernels.h"
#include "GridForce.h"
#include "IsolatedNonbondedForce.h"
#include "IsolatedBondedForce.h"
#include "IsolatedSiteForce.h"
#include "GBSAGridForce.h"
#include "IsolatedGBSAForce.h"
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
        platform.registerKernelFactory(CalcBondedHessianKernel::Name(), factory);
        platform.registerKernelFactory(CalcGBSAGridForceKernel::Name(), factory);
        platform.registerKernelFactory(CalcIsolatedGBSAForceKernel::Name(), factory);
        platform.registerKernelFactory(CalcIsolatedBondedForceKernel::Name(), factory);
        platform.registerKernelFactory(CalcIsolatedSiteForceKernel::Name(), factory);
        platform.registerKernelFactory(IntegrateMultiGroupHMCStepKernel::Name(), factory);
        platform.registerKernelFactory(IntegrateMultiGroupNUTSStepKernel::Name(), factory);
        // Replace stock CUDA MinimizeKernel factory with our compat version
        // (byte-identical on sm >= 60; carries a software atomicAdd(double*)
        // fallback for pre-Pascal). openmm.LocalEnergyMinimizer.minimize()
        // automatically routes through this whenever the plugin is loaded.
        platform.registerKernelFactory(MinimizeKernel::Name(), factory);
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
    if (name == CalcBondedHessianKernel::Name())
        return new CudaCalcBondedHessianKernel(name, platform, cu);
    if (name == CalcGBSAGridForceKernel::Name())
        return new CudaCalcGBSAGridForceKernel(name, platform, cu);
    if (name == CalcIsolatedGBSAForceKernel::Name())
        return new CudaCalcIsolatedGBSAForceKernel(name, platform, cu);
    if (name == CalcIsolatedBondedForceKernel::Name())
        return new CudaCalcIsolatedBondedForceKernel(name, platform, cu);
    if (name == CalcIsolatedSiteForceKernel::Name())
        return new CudaCalcIsolatedSiteForceKernel(name, platform, cu);
    if (name == IntegrateMultiGroupHMCStepKernel::Name())
        return new CudaIntegrateMultiGroupHMCStepKernel(name, platform, cu);
    if (name == IntegrateMultiGroupNUTSStepKernel::Name())
        return new CudaIntegrateMultiGroupNUTSStepKernel(name, platform, cu);
    if (name == MinimizeKernel::Name())
        return new PluginCompatMinimizeKernel(name, platform, cu);
    throw OpenMMException((std::string("Tried to create kernel with illegal kernel name '") + name + "'").c_str());
}
