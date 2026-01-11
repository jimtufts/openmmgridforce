#ifndef REFERENCE_GRIDFORCE_KERNELS_H_
#define REFERENCE_GRIDFORCE_KERNELS_H_

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



#include "GridForceKernels.h"
#include "openmm/Platform.h"
#include "openmm/Vec3.h"
#include <vector>

namespace OpenMM {
    class NonbondedForce;
}

namespace GridForcePlugin {

/**
 * This kernel is invoked by OpenMMGridForce to calculate the force 
 */
class ReferenceCalcGridForceKernel : public CalcGridForceKernel {
   public:
    ReferenceCalcGridForceKernel(std::string name,
                                 const OpenMM::Platform &platform) 
                                 : CalcGridForceKernel(name, platform) {
    }
    /**
     * Initialize the kernel.
     * 
     * @param system     the System this kernel will be applied to
     * @param force      the AlGDockNonbondForce this kernel will be used for
     */
    void initialize(const OpenMM::System &system, const GridForce &force);
    /**
     * Execute the kernel to calculate the forces and/or energy.
     *
     * @param context        the context in which to execute this kernel
     * @param includeForces  true if forces should be calculated
     * @param includeEnergy  true if the energy should be calculated
     * @return the potential energy due to the force
     */
    double execute(OpenMM::ContextImpl &context, bool includeForces, bool includeEnergy);
    /**
     * Copy changed parameters over to a context.
     *
     * @param context    the context to copy parameters to
     * @param force      the AlGDockNonbondForce to copy the parameters from
     */
    void copyParametersToContext(OpenMM::ContextImpl &context,
                                const GridForce &force);

    std::vector<double> getParticleGroupEnergies();
    std::vector<double> getParticleAtomEnergies();

   private:
    /**
     * Generate grid from receptor atoms and NonbondedForce parameters.
     *
     * @param system             the System containing the force
     * @param nonbondedForce     the NonbondedForce to extract parameters from
     * @param gridType           type of grid ("charge", "ljr", "lja")
     * @param receptorAtoms      indices of receptor atoms
     * @param receptorPositions  positions of receptor atoms (nm)
     * @param originX            grid origin x-coordinate (nm)
     * @param originY            grid origin y-coordinate (nm)
     * @param originZ            grid origin z-coordinate (nm)
     */
    void generateGrid(const OpenMM::System& system,
                     const OpenMM::NonbondedForce* nonbondedForce,
                     const std::string& gridType,
                     const std::vector<int>& receptorAtoms,
                     const std::vector<OpenMM::Vec3>& receptorPositions,
                     double originX, double originY, double originZ);

    std::vector<int> g_counts;
    std::vector<double> g_spacing;
    std::vector<double> g_vals;
    std::vector<double> g_scaling_factors;
    std::vector<int> g_ligand_atoms;    // Particle indices for ligand atoms (corresponds to scaling factors)
    double g_inv_power;
    double g_gridCap;
    double g_outOfBoundsRestraint;
    int g_interpolationMethod;  // 0=trilinear, 1=cubic B-spline, 2=tricubic, 3=quintic Hermite
    double g_origin_x, g_origin_y, g_origin_z;
    std::vector<double> g_derivatives;  // 27 derivatives per grid point for triquintic [27, nx, ny, nz]
    bool g_computeDerivatives;          // Whether to compute derivatives during grid generation

    /**
     * Compute all 27 derivatives at a grid point using finite differences.
     * Returns a vector of 27 values in the order specified in TRIQUINTIC_GRID_FORMAT.md.
     *
     * @param rawGrid       raw grid values [nx, ny, nz] before capping
     * @param ix, iy, iz    grid point indices
     * @param dx, dy, dz    grid spacing
     */
    std::vector<double> computeDerivativesAtPoint(
        const std::vector<double>& rawGrid,
        int ix, int iy, int iz,
        double dx, double dy, double dz) const;
};




}  // namespace GridForcePlugin

#endif /*REFERENCE_GRIDFORCE_KERNELS_H_*/
