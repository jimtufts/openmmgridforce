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
#include <functional>
#include <vector>
#include <map>

namespace OpenMM {
    class NonbondedForce;
    class ContextImpl;
}

namespace GridForcePlugin {
    class IsolatedNonbondedForce;
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
    std::vector<double> getParticleGroupUnscaledEnergies();
    std::vector<double> getParticleAtomEnergies();
    std::vector<double> getParticleGroupAtomRawEnergies();
    std::vector<int> getParticleOutOfBoundsFlags();

    // ---- Per-atom Hessian (block-diagonal 3x3 per atom) ----
    void computeHessian() override;
    std::vector<double> getHessianBlocks() override;
    void computeThirdDerivatives() override;
    std::vector<double> getThirdDerivativeBlocks() override;
    void analyzeHessian(float temperature) override;
    std::vector<double> getEigenvalues() override;
    std::vector<double> getEigenvectors() override;
    std::vector<double> getMeanCurvature() override;
    std::vector<double> getTotalCurvature() override;
    std::vector<double> getGaussianCurvature() override;
    std::vector<double> getFracAnisotropy() override;
    std::vector<double> getEntropy() override;
    std::vector<double> getMinEigenvalue() override;
    std::vector<int> getNumNegative() override;
    double getTotalEntropy() override;

   protected:
    /**
     * Compute one ligand atom's contribution: writes forceData[ia] and returns
     * this atom's energy contribution, unscaled per-group contribution, and the
     * group index it maps to. Per-atom force writes are disjoint, so distinct
     * atoms never write the same force entry; execute() reduces the returned
     * per-atom contributions serially in atom order.
     */
    void computeAtom(int ia, std::vector<OpenMM::Vec3>& posData,
                     std::vector<OpenMM::Vec3>& forceData,
                     bool includeForces, bool includeEnergy,
                     double& atomEnergy, double& atomUnscaled, int& groupIdx);

    // Fill the per-atom result arrays. Serial here; the CPU platform overrides
    // to distribute atoms across the thread pool.
    virtual void runAtoms(OpenMM::ContextImpl& context,
                          std::vector<OpenMM::Vec3>& posData,
                          std::vector<OpenMM::Vec3>& forceData,
                          bool includeForces, bool includeEnergy);

    // Per-atom contributions filled by runAtoms() and reduced in execute().
    std::vector<double> g_atomEnergyContribution;
    std::vector<double> g_atomUnscaledContribution;
    std::vector<int> g_atomGroupIdx;

    /**
     * Compute the per-atom 3x3 interpolant Hessian blocks for the current
     * positions. Shared by computeHessian() (and used to seed analyzeHessian()).
     * Output is 6 components per ligand atom: [xx, yy, zz, xy, xz, yz].
     */
    void computeHessianForPositions(const std::vector<OpenMM::Vec3>& posData);

    // Run body(i) for i in [0, count). Serial here; the CPU platform overrides
    // to distribute the iterations across the thread pool. Uses g_lastContext
    // for the pool, so it is only valid inside the Hessian path.
    virtual void parallelFor(int count, const std::function<void(int)>& body);

    // Cached pointer to the last context passed to execute(); positions are read
    // from it inside computeHessian()/computeThirdDerivatives().
    OpenMM::ContextImpl* g_lastContext = nullptr;

    // Hessian / analysis result storage (mutable cache, downloaded on demand).
    std::vector<double> g_hessianBlocks;       // 6 per atom
    std::vector<double> g_thirdDerivBlocks;    // 10 per atom
    std::vector<double> g_eigenvalues;         // 3 per atom
    std::vector<double> g_eigenvectors;        // 9 per atom
    std::vector<double> g_meanCurvature;
    std::vector<double> g_totalCurvature;
    std::vector<double> g_gaussianCurvature;
    std::vector<double> g_fracAnisotropy;
    std::vector<double> g_entropy;
    std::vector<double> g_minEigenvalue;
    std::vector<int> g_numNegative;
    double g_totalEntropy = 0.0;

    /**
     * Generate grid from receptor atoms and NonbondedForce/IsolatedNonbondedForce parameters.
     *
     * @param system                    the System containing the force
     * @param nonbondedForce            the NonbondedForce to extract parameters from (may be nullptr)
     * @param isolatedNonbondedForce    the IsolatedNonbondedForce to extract parameters from (may be nullptr)
     * @param gridType                  type of grid ("charge", "ljr", "lja")
     * @param receptorAtoms             indices of receptor atoms
     * @param receptorPositions         positions of receptor atoms (nm)
     * @param originX                   grid origin x-coordinate (nm)
     * @param originY                   grid origin y-coordinate (nm)
     * @param originZ                   grid origin z-coordinate (nm)
     */
    void generateGrid(const OpenMM::System& system,
                     const OpenMM::NonbondedForce* nonbondedForce,
                     const IsolatedNonbondedForce* isolatedNonbondedForce,
                     const std::string& gridType,
                     const std::vector<int>& receptorAtoms,
                     const std::vector<OpenMM::Vec3>& receptorPositions,
                     double originX, double originY, double originZ);

    // Auto-generation inputs captured in initialize(); the grid is built lazily in
    // the first execute(), where a Context (and thus the thread pool used by the
    // CPU platform's parallelFor) is available. System/Forces persist for the
    // Context lifetime, so storing the pointers is safe.
    bool genNeedsGrid_ = false;
    const OpenMM::System* genSystem_ = nullptr;
    const OpenMM::NonbondedForce* genNonbonded_ = nullptr;
    const IsolatedNonbondedForce* genIsolated_ = nullptr;
    std::string genGridType_;
    std::vector<int> genReceptorAtoms_;
    std::vector<OpenMM::Vec3> genReceptorPositions_;
    double genOrigin_[3] = {0.0, 0.0, 0.0};

    std::vector<int> g_counts;
    std::vector<double> g_spacing;
    std::vector<double> g_vals;
    std::vector<double> g_scaling_factors;
    double g_globalScalingFactor;       // Multiplies all per-particle scaling factors (default 1.0)
    std::vector<double> g_groupScalingFactors;  // Per-group alchemical scaling factors
    std::vector<double> g_groupRuntimeCaps;     // Per-group runtime caps (0 = use global)
    std::vector<std::vector<int>> g_groupParticleIndices;  // Per-group particle indices
    std::map<int, int> g_atomToGroup;   // Map from particle index to group index
    std::vector<double> g_groupEnergies;  // Per-group energies from last execute()
    std::vector<double> g_groupUnscaledEnergies;  // Per-group unscaled energies (no group scaling)
    std::vector<int> g_ligand_atoms;    // Particle indices for ligand atoms (corresponds to scaling factors)
    double g_inv_power;
    double g_gridCap;
    double g_runtimeCap;
    double g_outOfBoundsRestraint;
    double g_effectiveMinX, g_effectiveMinY, g_effectiveMinZ;  // Effective bounds (grid-local)
    double g_effectiveMaxX, g_effectiveMaxY, g_effectiveMaxZ;
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




/**
 * Reference implementation of CalcBondedHessianKernel.
 * Computes analytical Hessian of bonded forces on CPU.
 */
class ReferenceCalcBondedHessianKernel : public CalcBondedHessianKernel {
public:
    ReferenceCalcBondedHessianKernel(std::string name, const OpenMM::Platform& platform)
        : CalcBondedHessianKernel(name, platform), numAtoms(0), numBonds(0),
          numAngles(0), numTorsions(0) {
    }

    void initialize(const OpenMM::System& system);
    std::vector<double> computeHessian(OpenMM::ContextImpl& context);
    int getNumBonds() const { return numBonds; }
    int getNumAngles() const { return numAngles; }
    int getNumTorsions() const { return numTorsions; }

private:
    int numAtoms;
    int numBonds;
    int numAngles;
    int numTorsions;

    // Bond parameters
    std::vector<int> bondAtoms;      // [atom1, atom2] * numBonds
    std::vector<double> bondLengths;
    std::vector<double> bondKs;

    // Angle parameters
    std::vector<int> angleAtoms;     // [atom1, atom2, atom3] * numAngles
    std::vector<double> angleValues;
    std::vector<double> angleKs;

    // Torsion parameters
    std::vector<int> torsionAtoms;   // [atom1, atom2, atom3, atom4] * numTorsions
    std::vector<int> torsionPeriodicities;
    std::vector<double> torsionPhases;
    std::vector<double> torsionKs;
};

}  // namespace GridForcePlugin

#endif /*REFERENCE_GRIDFORCE_KERNELS_H_*/
