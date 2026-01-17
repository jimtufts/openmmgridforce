/* -------------------------------------------------------------------------- *
 *                              OpenMMGridForce                               *
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

#include "internal/GridForceImpl.h"
#include "GridForceKernels.h"
#include "openmm/Platform.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/NonbondedForce.h"
#include "openmm/OpenMMException.h"

#include <cmath>
#include <vector>
#include <sstream>


using namespace OpenMM;
using namespace std;

namespace GridForcePlugin {

GridForceImpl::GridForceImpl(const GridForce &owner) : owner(owner) {
}

GridForceImpl::~GridForceImpl() {
}

void GridForceImpl::initialize(ContextImpl &context) {
    // Set the System pointer for per-System cache scoping
    // Cast away const - this is safe because we only use it as a key, never dereference it
    const_cast<GridForce&>(owner).setSystemPointer(&context.getSystem());

    kernel = context.getPlatform().createKernel(CalcGridForceKernel::Name(), context);
    kernel.getAs<CalcGridForceKernel>().initialize(context.getSystem(), owner);
}

double GridForceImpl::calcForcesAndEnergy(ContextImpl &context, bool includeForces, bool includeEnergy, int groups) {
    if ((groups & (1 << owner.getForceGroup())) != 0)
        return kernel.getAs<CalcGridForceKernel>().execute(context, includeForces, includeEnergy);
    return 0.0;
}

std::vector<std::string> GridForceImpl::getKernelNames() {
    std::vector<std::string> names;
    names.push_back(CalcGridForceKernel::Name());
    return names;
}

void GridForceImpl::updateParametersInContext(ContextImpl &context) {
    kernel.getAs<CalcGridForceKernel>().copyParametersToContext(context, owner);
}

std::vector<double> GridForceImpl::getParticleGroupEnergies() {
    return kernel.getAs<CalcGridForceKernel>().getParticleGroupEnergies();
}

std::vector<double> GridForceImpl::getParticleAtomEnergies() {
    return kernel.getAs<CalcGridForceKernel>().getParticleAtomEnergies();
}

std::vector<int> GridForceImpl::getParticleOutOfBoundsFlags() {
    return kernel.getAs<CalcGridForceKernel>().getParticleOutOfBoundsFlags();
}

void GridForceImpl::computeHessian() {
    kernel.getAs<CalcGridForceKernel>().computeHessian();
}

std::vector<double> GridForceImpl::getHessianBlocks() {
    return kernel.getAs<CalcGridForceKernel>().getHessianBlocks();
}

void GridForceImpl::analyzeHessian(float temperature) {
    kernel.getAs<CalcGridForceKernel>().analyzeHessian(temperature);
}

std::vector<double> GridForceImpl::getEigenvalues() {
    return kernel.getAs<CalcGridForceKernel>().getEigenvalues();
}

std::vector<double> GridForceImpl::getEigenvectors() {
    return kernel.getAs<CalcGridForceKernel>().getEigenvectors();
}

std::vector<double> GridForceImpl::getMeanCurvature() {
    return kernel.getAs<CalcGridForceKernel>().getMeanCurvature();
}

std::vector<double> GridForceImpl::getTotalCurvature() {
    return kernel.getAs<CalcGridForceKernel>().getTotalCurvature();
}

std::vector<double> GridForceImpl::getGaussianCurvature() {
    return kernel.getAs<CalcGridForceKernel>().getGaussianCurvature();
}

std::vector<double> GridForceImpl::getFracAnisotropy() {
    return kernel.getAs<CalcGridForceKernel>().getFracAnisotropy();
}

std::vector<double> GridForceImpl::getEntropy() {
    return kernel.getAs<CalcGridForceKernel>().getEntropy();
}

std::vector<double> GridForceImpl::getMinEigenvalue() {
    return kernel.getAs<CalcGridForceKernel>().getMinEigenvalue();
}

std::vector<int> GridForceImpl::getNumNegative() {
    return kernel.getAs<CalcGridForceKernel>().getNumNegative();
}

double GridForceImpl::getTotalEntropy() {
    return kernel.getAs<CalcGridForceKernel>().getTotalEntropy();
}

}  // namespace GridForcePlugin
