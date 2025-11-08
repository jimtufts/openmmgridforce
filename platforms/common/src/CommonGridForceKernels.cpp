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

#include "CommonGridForceKernels.h"
#include "CommonGridForceKernelSources.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/common/ComputeForceInfo.h"
#include "openmm/common/ContextSelector.h"
#include <map>
#include <iostream>
#include <cstring>
#include <algorithm>

using namespace GridForcePlugin;
using namespace OpenMM;
using namespace std;

class GridForceInfo : public ComputeForceInfo {
public:
    GridForceInfo(int numAtoms) : numAtoms(numAtoms) {
    }
    int getNumParticleGroups() {
        return numAtoms;
    }
    void getParticlesInGroup(int index, vector<int>& particles) {
        particles.clear();
        particles.push_back(index);
    }
    bool areGroupsIdentical(int group1, int group2) {
        return true;
    }
private:
    int numAtoms;
};

void CommonCalcGridForceKernel::initialize(const System& system, const GridForce& force) {
    ContextSelector selector(cc);

    // Get grid parameters
    vector<int> counts;
    vector<double> spacing;
    vector<double> vals;
    vector<double> scaling_factors;
    force.getGridParameters(counts, spacing, vals, scaling_factors);
    double inv_power = force.getInvPower();

    numAtoms = system.getNumParticles();

    if (spacing.size() != 3 || counts.size() != 3) {
        throw OpenMMException("GridForce: Grid dimensions must be 3D");
    }

    if (vals.size() != counts[0] * counts[1] * counts[2]) {
        throw OpenMMException("GridForce: Number of grid values doesn't match grid dimensions");
    }

    if (scaling_factors.size() != numAtoms) {
        throw OpenMMException("GridForce: Number of scaling factors must match number of atoms");
    }

    // Initialize arrays
    g_counts.initialize<int>(cc, 3, "gridCounts");
    g_spacing.initialize<float>(cc, 3, "gridSpacing");
    g_vals.initialize<float>(cc, vals.size(), "gridValues");
    g_scaling_factors.initialize<float>(cc, scaling_factors.size(), "scalingFactors");

    // Copy data to device
    vector<int> countsVec = {counts[0], counts[1], counts[2]};
    vector<float> spacingVec = {(float)spacing[0], (float)spacing[1], (float)spacing[2]};
    vector<float> valsFloat(vals.begin(), vals.end());
    vector<float> scalingFloat(scaling_factors.begin(), scaling_factors.end());

    g_counts.upload(countsVec);
    g_spacing.upload(spacingVec);
    g_vals.upload(valsFloat);
    g_scaling_factors.upload(scalingFloat);

    // Create kernel
    map<string, string> defines;
    defines["GRID_SIZE_X"] = cc.intToString(counts[0]);
    defines["GRID_SIZE_Y"] = cc.intToString(counts[1]);
    defines["GRID_SIZE_Z"] = cc.intToString(counts[2]);
    defines["NUM_ATOMS"] = cc.intToString(numAtoms);
    defines["PADDED_NUM_ATOMS"] = cc.intToString(cc.getPaddedNumAtoms());

    ComputeProgram program = cc.compileProgram(CommonGridForceKernelSources::gridForceSource, defines);
    computeKernel = program->createKernel("computeGridForce");

    // Set kernel arguments
    computeKernel->addArg(cc.getPosq());
    computeKernel->addArg(cc.getLongForceBuffer());
    computeKernel->addArg(g_counts);
    computeKernel->addArg(g_spacing);
    computeKernel->addArg(g_vals);
    computeKernel->addArg(g_scaling_factors);
    computeKernel->addArg((float)inv_power);
    computeKernel->addArg(cc.getEnergyBuffer());

    cc.addForce(new GridForceInfo(numAtoms));

    hasInitializedKernel = true;
}

double CommonCalcGridForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    if (!hasInitializedKernel) {
        throw OpenMMException("CommonCalcGridForceKernel: Kernel not initialized before execution");
    }

    if (numAtoms == 0) {
        return 0.0;
    }

    // Execute the kernel
    computeKernel->execute(numAtoms);

    return 0.0;
}

void CommonCalcGridForceKernel::copyParametersToContext(ContextImpl& contextImpl, const GridForce& force) {
    vector<int> counts;
    vector<double> spacing;
    vector<double> vals;
    vector<double> scaling_factors;
    force.getGridParameters(counts, spacing, vals, scaling_factors);
    double inv_power = force.getInvPower();

    if (numAtoms != contextImpl.getSystem().getNumParticles())
        throw OpenMMException("updateParametersInContext: The number of particles has changed");

    if (spacing.size() != 3 || counts.size() != 3)
        throw OpenMMException("GridForce: Grid dimensions must be 3D");

    if (vals.size() != counts[0] * counts[1] * counts[2])
        throw OpenMMException("GridForce: Number of grid values doesn't match grid dimensions");

    if (scaling_factors.size() != numAtoms)
        throw OpenMMException("GridForce: Number of scaling factors must match number of atoms");

    // Update arrays
    vector<int> countsVec = {counts[0], counts[1], counts[2]};
    vector<float> spacingVec = {(float)spacing[0], (float)spacing[1], (float)spacing[2]};
    vector<float> valsFloat(vals.begin(), vals.end());
    vector<float> scalingFloat(scaling_factors.begin(), scaling_factors.end());

    g_counts.upload(countsVec);
    g_spacing.upload(spacingVec);
    g_vals.upload(valsFloat);
    g_scaling_factors.upload(scalingFloat);

    // Update inv_power parameter
    computeKernel->setArg(6, (float)inv_power);
}
