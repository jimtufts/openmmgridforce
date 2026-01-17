#ifndef OPENMM_GRIDFORCE_KERNELS_H_
#define OPENMM_GRIDFORCE_KERNELS_H_

/* -------------------------------------------------------------------------- *
 *                               OpenMMGridForce                              *
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

#include <string>

#include "GridForce.h"
#include "openmm/KernelImpl.h"
#include "openmm/OpenMMException.h"
#include "openmm/Platform.h"
#include "openmm/System.h"

namespace GridForcePlugin {


// Add the other ForceKernels
class CalcGridForceKernel : public OpenMM::KernelImpl {
   public:
    static std::string Name() {
        return "CalcGridForce";
    }

    CalcGridForceKernel(std::string name, const OpenMM::Platform &platform) 
        : OpenMM::KernelImpl(name, platform) {
    }
    /**
     * Initialize the kernel.
     * 
     * @param system     the System this kernel will be applied to
     * @param force      the GridForce this kernel will be used for
     */
    virtual void initialize(const OpenMM::System &system, 
                            const GridForce &force) = 0;
    /**
     * Execute the kernel to calculate the forces and/or energy.
     *
     * @param context        the context in which to execute this kernel
     * @param includeForces  true if forces should be calculated
     * @param includeEnergy  true if the energy should be calculated
     * @return the potential energy due to the force
     */
    virtual double execute(OpenMM::ContextImpl &context, bool includeForces, bool includeEnergy) = 0;
    /**
     * Copy changed parameters over to a context.
     *
     * @param context    the context to copy parameters to
     * @param force      the AlGDockForce to copy the parameters from
     */
    virtual void copyParametersToContext(OpenMM::ContextImpl &context,
                                        const GridForce &force) = 0;
    /**
     * Get per-particle-group energies.
     *
     * @return vector of energies, one per particle group (empty if no groups)
     */
    virtual std::vector<double> getParticleGroupEnergies() = 0;
    /**
     * Get per-atom energies for particles in groups.
     *
     * @return vector of energies, one per particle across all groups
     */
    virtual std::vector<double> getParticleAtomEnergies() = 0;
    /**
     * Get per-atom out-of-bounds flags for particles in groups.
     *
     * @return vector of flags (0=inside, 1=outside), one per particle across all groups
     */
    virtual std::vector<int> getParticleOutOfBoundsFlags() = 0;
    /**
     * Compute Hessian (second derivative) blocks for each atom from grid potential.
     * Must be called after execute(). Results stored internally and retrieved via getHessianBlocks().
     * Only supported for bspline (method 1) and triquintic (method 3) interpolation.
     * Default implementation throws - override in platform-specific implementations.
     */
    virtual void computeHessian() {
        throw OpenMM::OpenMMException("Hessian computation not supported on this platform");
    }
    /**
     * Get the Hessian blocks computed by computeHessian().
     *
     * @return vector of 6 components per atom: [dxx, dyy, dzz, dxy, dxz, dyz]
     *         Total size is 6 * numAtoms. Empty if computeHessian() not called.
     */
    virtual std::vector<double> getHessianBlocks() {
        return std::vector<double>();
    }
    /**
     * Analyze Hessian blocks to compute per-atom metrics: eigenvalues, curvature,
     * anisotropy, and entropy estimates. Must call computeHessian() first.
     * Default implementation throws - override in platform-specific implementations.
     *
     * @param temperature  Temperature in Kelvin for entropy calculation
     */
    virtual void analyzeHessian(float temperature) {
        throw OpenMM::OpenMMException("Hessian analysis not supported on this platform");
    }
    /**
     * Get the eigenvalues computed by analyzeHessian().
     * @return vector of 3 eigenvalues per atom [λ1, λ2, λ3], sorted ascending
     */
    virtual std::vector<double> getEigenvalues() {
        return std::vector<double>();
    }
    /**
     * Get the eigenvectors computed by analyzeHessian().
     * @return vector of 9 components per atom (3 eigenvectors × 3 components)
     */
    virtual std::vector<double> getEigenvectors() {
        return std::vector<double>();
    }
    /**
     * Get the mean curvature computed by analyzeHessian().
     * @return vector of mean curvature per atom: (λ1 + λ2 + λ3) / 3
     */
    virtual std::vector<double> getMeanCurvature() {
        return std::vector<double>();
    }
    /**
     * Get the total curvature computed by analyzeHessian().
     * @return vector of total curvature per atom: λ1 + λ2 + λ3
     */
    virtual std::vector<double> getTotalCurvature() {
        return std::vector<double>();
    }
    /**
     * Get the Gaussian curvature computed by analyzeHessian().
     * @return vector of Gaussian curvature per atom: λ1 * λ2 * λ3
     */
    virtual std::vector<double> getGaussianCurvature() {
        return std::vector<double>();
    }
    /**
     * Get the fractional anisotropy computed by analyzeHessian().
     * @return vector of FA per atom, range [0, 1] (0=isotropic, 1=linear)
     */
    virtual std::vector<double> getFracAnisotropy() {
        return std::vector<double>();
    }
    /**
     * Get the per-atom entropy computed by analyzeHessian().
     * @return vector of entropy per atom in kB units (NaN for saddle points)
     */
    virtual std::vector<double> getEntropy() {
        return std::vector<double>();
    }
    /**
     * Get the minimum eigenvalue computed by analyzeHessian().
     * @return vector of minimum eigenvalue per atom
     */
    virtual std::vector<double> getMinEigenvalue() {
        return std::vector<double>();
    }
    /**
     * Get the count of negative eigenvalues computed by analyzeHessian().
     * @return vector of negative eigenvalue count per atom (0-3)
     */
    virtual std::vector<int> getNumNegative() {
        return std::vector<int>();
    }
    /**
     * Get the total entropy summed over all atoms (excluding saddle points).
     * @return total entropy in kB units
     */
    virtual double getTotalEntropy() {
        return 0.0;
    }
};

/**
 * Kernel for computing Hessians of bonded forces (bonds, angles, torsions).
 * Platform-specific implementations compute analytical second derivatives.
 */
class CalcBondedHessianKernel : public OpenMM::KernelImpl {
public:
    static std::string Name() {
        return "CalcBondedHessian";
    }

    CalcBondedHessianKernel(std::string name, const OpenMM::Platform& platform)
        : OpenMM::KernelImpl(name, platform) {
    }

    /**
     * Initialize the kernel by extracting bonded force parameters from the System.
     *
     * @param system  the System containing bonded forces
     */
    virtual void initialize(const OpenMM::System& system) = 0;

    /**
     * Compute the full Hessian matrix for bonded interactions.
     *
     * @param context  the context containing current positions
     * @return flattened 3N x 3N Hessian matrix in row-major order
     */
    virtual std::vector<double> computeHessian(OpenMM::ContextImpl& context) = 0;

    /**
     * Get the number of bonds extracted from the System.
     */
    virtual int getNumBonds() const = 0;

    /**
     * Get the number of angles extracted from the System.
     */
    virtual int getNumAngles() const = 0;

    /**
     * Get the number of torsions extracted from the System.
     */
    virtual int getNumTorsions() const = 0;
};

}  // namespace GridForcePlugin

#endif /* OPENMM_GRIDFORCE_KERNELS_H_*/
