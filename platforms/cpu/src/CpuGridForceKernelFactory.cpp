/* -------------------------------------------------------------------------- *
 *                            OpenMMGridForce                                 *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2014 Stanford University and the Authors.           *
 * Authors:                                                                   *
 * Contributors:                                                              *
 *                                                                            *
 * Permission is hereby granted, free of charge, to any person obtaining a    *
 * copy of this software and associated documentation files (the "Software"), *
 * to deal in the Software without restriction, including without limitation  *
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,   *
 * and/or sell copies of the Software, and to permit persons to whom the      *
 * Software is furnished to do so, subject to the following conditions:       *
 *                                                                            *
 * The above copyright notice and this permission notice shall be included in *
 * all copies or substantial portions of the Software.                        *
 *                                                                            *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR *
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,   *
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL    *
 * THE AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,    *
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR      *
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE  *
 * USE OR OTHER DEALINGS IN THE SOFTWARE.                                     *
 * -------------------------------------------------------------------------- */

#include <iostream>

#include "CpuGridForceKernelFactory.h"
#include "CpuGridForceKernels.h"
#include "CpuIsolatedNonbondedKernels.h"
#include "CpuIsolatedBondedKernels.h"
#include "CpuIsolatedGBSAKernels.h"
#include "CpuIsolatedSiteKernels.h"
#include "CpuGBSAGridForceKernels.h"
#include "CpuMultiGroupHMCKernels.h"
#include "CpuMultiGroupNUTSKernels.h"
#include "ReferenceGridForceKernels.h"
#include "ReferenceIsolatedNonbondedKernels.h"
#include "ReferenceIsolatedBondedKernels.h"
#include "ReferenceIsolatedGBSAKernels.h"
#include "ReferenceIsolatedSiteKernels.h"
#include "ReferenceGBSAGridForceKernels.h"
#include "ReferenceMultiGroupHMCKernels.h"
#include "ReferenceMultiGroupNUTSKernels.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/reference/ReferencePlatform.h"

using namespace OpenMM;

namespace GridForcePlugin {

extern "C" OPENMM_EXPORT void registerPlatforms() {
}

extern "C" OPENMM_EXPORT void registerKernelFactories() {
    try {
        for (int i = 0; i < Platform::getNumPlatforms(); i++) {
            Platform& platform = Platform::getPlatform(i);
            if (platform.getName() == "CPU") {
                CpuGridForceKernelFactory* factory = new CpuGridForceKernelFactory();
                platform.registerKernelFactory(CalcGridForceKernel::Name(), factory);
                platform.registerKernelFactory(CalcBondedHessianKernel::Name(), factory);
                platform.registerKernelFactory(CalcIsolatedNonbondedForceKernel::Name(), factory);
                platform.registerKernelFactory(CalcIsolatedGBSAForceKernel::Name(), factory);
                platform.registerKernelFactory(CalcGBSAGridForceKernel::Name(), factory);
                platform.registerKernelFactory(CalcIsolatedBondedForceKernel::Name(), factory);
                platform.registerKernelFactory(CalcIsolatedSiteForceKernel::Name(), factory);
                platform.registerKernelFactory(IntegrateMultiGroupHMCStepKernel::Name(), factory);
                platform.registerKernelFactory(IntegrateMultiGroupNUTSStepKernel::Name(), factory);
            }
        }
    }
    catch (...) {
    }
}

KernelImpl* CpuGridForceKernelFactory::createKernelImpl(std::string name, const Platform& platform, ContextImpl& context) const {
    // Delegate all kernel types to the Reference implementation.
    // CpuPlatform::PlatformData inherits from ReferencePlatform::PlatformData,
    // so the Reference kernels work correctly on the CPU platform.
    if (name == CalcGridForceKernel::Name())
        return new CpuCalcGridForceKernel(name, platform);
    if (name == CalcBondedHessianKernel::Name())
        return new ReferenceCalcBondedHessianKernel(name, platform);
    if (name == CalcIsolatedNonbondedForceKernel::Name())
        return new CpuCalcIsolatedNonbondedForceKernel(name, platform);
    if (name == CalcIsolatedGBSAForceKernel::Name())
        return new CpuCalcIsolatedGBSAForceKernel(name, platform);
    if (name == CalcGBSAGridForceKernel::Name())
        return new CpuCalcGBSAGridForceKernel(name, platform);
    if (name == CalcIsolatedBondedForceKernel::Name())
        return new CpuCalcIsolatedBondedForceKernel(name, platform);
    if (name == CalcIsolatedSiteForceKernel::Name())
        return new CpuCalcIsolatedSiteForceKernel(name, platform);
    if (name == IntegrateMultiGroupHMCStepKernel::Name())
        return new CpuIntegrateMultiGroupHMCStepKernel(name, platform);
    if (name == IntegrateMultiGroupNUTSStepKernel::Name())
        return new CpuIntegrateMultiGroupNUTSStepKernel(name, platform);

    throw OpenMMException((std::string("Tried to create kernel with illegal kernel name '") + name + "'").c_str());
}

} // namespace GridForcePlugin
