#ifndef OPENMM_GRIDFORCE_IMPL_H_
#define OPENMM_GRIDFORCE_IMPL_H_

/* -------------------------------------------------------------------------- *
 *                                OpenMMGridForce                             *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2008 Stanford University and the Authors.           *
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



#include "GridForce.h"
#include "openmm/Kernel.h"
#include "openmm/internal/ForceImpl.h"
#include <string>
#include <utility>
#include <vector>


namespace GridForcePlugin {

/**
 * This is the internal implementation of AlGDockGridForce.
 */

class OPENMM_EXPORT_GRIDFORCE GridForceImpl : public OpenMM::ForceImpl {
   public:
    GridForceImpl(const GridForce &owner);
    ~GridForceImpl();
    void initialize(OpenMM::ContextImpl &context);
    const GridForce &getOwner() const {
        return owner;
    }
    void updateContextState(OpenMM::ContextImpl &context, bool& forcesInvalid) {
        // This force field doesn't update the state directly.
    }
    double calcForcesAndEnergy(OpenMM::ContextImpl &context, 
                                bool includeForces, 
                                bool includeEnergy, 
                                int groups);
    std::map<std::string, double> getDefaultParameters() {
        return std::map<std::string, double>();  // This force field doesn't define any parameters.
    }
    std::vector<std::string> getKernelNames();

    void updateParametersInContext(OpenMM::ContextImpl &context);

    std::vector<double> getParticleGroupEnergies();

    std::vector<double> getParticleAtomEnergies();

    std::vector<int> getParticleOutOfBoundsFlags();

    /**
     * Compute Hessian (second derivative) blocks for each atom from grid potential.
     * Results stored in kernel and retrieved via getHessianBlocks().
     */
    void computeHessian();

    /**
     * Get the Hessian blocks computed by computeHessian().
     * @return vector of 6 components per atom: [dxx, dyy, dzz, dxy, dxz, dyz]
     */
    std::vector<double> getHessianBlocks();

    /**
     * Analyze Hessian blocks to compute per-atom metrics.
     * @param temperature  Temperature in Kelvin for entropy calculation
     */
    void analyzeHessian(float temperature);

    /**
     * Get eigenvalues computed by analyzeHessian().
     */
    std::vector<double> getEigenvalues();

    /**
     * Get eigenvectors computed by analyzeHessian().
     */
    std::vector<double> getEigenvectors();

    /**
     * Get mean curvature computed by analyzeHessian().
     */
    std::vector<double> getMeanCurvature();

    /**
     * Get total curvature computed by analyzeHessian().
     */
    std::vector<double> getTotalCurvature();

    /**
     * Get Gaussian curvature computed by analyzeHessian().
     */
    std::vector<double> getGaussianCurvature();

    /**
     * Get fractional anisotropy computed by analyzeHessian().
     */
    std::vector<double> getFracAnisotropy();

    /**
     * Get per-atom entropy computed by analyzeHessian().
     */
    std::vector<double> getEntropy();

    /**
     * Get minimum eigenvalue per atom computed by analyzeHessian().
     */
    std::vector<double> getMinEigenvalue();

    /**
     * Get count of negative eigenvalues per atom computed by analyzeHessian().
     */
    std::vector<int> getNumNegative();

    /**
     * Get total entropy computed by analyzeHessian().
     */
    double getTotalEntropy();

   private:
    const GridForce &owner;
    OpenMM::Kernel kernel;
};

}  // namespace GridForcePlugin

#endif /*OPENMM_GRIDFORCE_IMPL_H_*/
