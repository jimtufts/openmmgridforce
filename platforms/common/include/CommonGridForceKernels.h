#ifndef OPENMM_COMMONGRIDFORCEKERNELS_H_
#define OPENMM_COMMONGRIDFORCEKERNELS_H_

#include "GridForceKernels.h"
#include "openmm/common/ComputeContext.h"
#include "openmm/NonbondedForce.h"
#include <vector>

namespace GridForcePlugin {

/**
 * This kernel is implemented by CUDA and OpenCL platforms to compute grid forces.
 */
class CommonCalcGridForceKernel : public CalcGridForceKernel {
public:
    CommonCalcGridForceKernel(std::string name, const OpenMM::Platform& platform, OpenMM::ComputeContext& cc) : 
        CalcGridForceKernel(name, platform), hasInitializedKernel(false), cc(cc) {
    }
    /**
     * Initialize the kernel.
     * 
     * @param system     the System this kernel will be applied to
     * @param force      the GridForce this kernel will be used for
     */
    virtual void initialize(const OpenMM::System& system, const GridForce& force);
    /**
     * Execute the kernel to calculate the forces and/or energy.
     *
     * @param context        the context in which to execute this kernel
     * @param includeForces  true if forces should be calculated
     * @param includeEnergy  true if the energy should be calculated
     * @return the potential energy due to the force
     */
    virtual double execute(OpenMM::ContextImpl& context, bool includeForces, bool includeEnergy);
    /**
     * Copy changed parameters over to a context.
     *
     * @param context    the context to copy parameters to
     * @param force      the GridForce to copy the parameters from
     */
    virtual void copyParametersToContext(OpenMM::ContextImpl& context, const GridForce& force);

protected:
    /**
     * Generate grid values from receptor atoms and NonbondedForce parameters.
     */
    void generateGrid(const OpenMM::System& system,
                     const OpenMM::NonbondedForce* nonbondedForce,
                     const std::string& gridType,
                     const std::vector<int>& receptorAtoms,
                     const std::vector<OpenMM::Vec3>& receptorPositions,
                     double originX, double originY, double originZ,
                     std::vector<double>& vals);

    bool hasInitializedKernel;
    OpenMM::ComputeContext& cc;
    OpenMM::ComputeArray g_counts;
    OpenMM::ComputeArray g_spacing;
    OpenMM::ComputeArray g_vals;
    OpenMM::ComputeArray g_scaling_factors;
    OpenMM::ComputeKernel computeKernel;
    int numAtoms;
    std::vector<int> counts;
    std::vector<double> spacing;
};

} // namespace GridForcePlugin

#endif /*OPENMM_COMMONGRIDFORCEKERNELS_H_*/
