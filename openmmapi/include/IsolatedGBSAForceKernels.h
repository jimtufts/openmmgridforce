/* -------------------------------------------------------------------------- *
 *                              OpenMMGridForce                               *
 * -------------------------------------------------------------------------- */

#ifndef OPENMM_ISOLATEDGBSAFORCE_KERNELS_H_
#define OPENMM_ISOLATEDGBSAFORCE_KERNELS_H_

#include "IsolatedGBSAForce.h"
#include "openmm/KernelImpl.h"
#include "openmm/Platform.h"
#include "openmm/System.h"
#include <string>
#include <vector>

namespace GridForcePlugin {

/**
 * This kernel computes the IsolatedGBSAForce.
 */
class CalcIsolatedGBSAForceKernel : public OpenMM::KernelImpl {
public:
    static std::string Name() {
        return "CalcIsolatedGBSAForce";
    }

    CalcIsolatedGBSAForceKernel(std::string name, const OpenMM::Platform& platform)
        : OpenMM::KernelImpl(name, platform) {
    }

    /**
     * Initialize the kernel.
     *
     * @param system  The System this kernel will be applied to
     * @param force   The IsolatedGBSAForce this kernel computes
     */
    virtual void initialize(const OpenMM::System& system, const IsolatedGBSAForce& force) = 0;

    /**
     * Execute the kernel to calculate forces and/or energy.
     *
     * @param context       The context in which to execute
     * @param includeForces True if forces should be calculated
     * @param includeEnergy True if energy should be calculated
     * @return The potential energy
     */
    virtual double execute(OpenMM::ContextImpl& context,
                          bool includeForces,
                          bool includeEnergy) = 0;

    /**
     * Update parameters in the context after they have changed.
     *
     * @param context The context in which to update
     * @param force   The IsolatedGBSAForce with updated parameters
     */
    virtual void updateParametersInContext(OpenMM::ContextImpl& context,
                                           const IsolatedGBSAForce& force) = 0;

    /**
     * Get the total GB energy for a specific particle group.
     * Called after execute() to retrieve per-group energies.
     */
    virtual double getGroupEnergy(int groupIndex) const = 0;

    /**
     * Get the ligand self-solvation energy (ligand-ligand GB only).
     */
    virtual double getGroupLigandSelfEnergy(int groupIndex) const = 0;

    /**
     * Get the receptor contribution to ligand solvation.
     * (Change in ligand GB energy due to receptor screening)
     */
    virtual double getGroupReceptorContribution(int groupIndex) const = 0;

    /**
     * Get the receptor desolvation energy (PAIRWISE mode only).
     * (Change in receptor GB energy due to ligand screening)
     */
    virtual double getGroupReceptorDesolvation(int groupIndex) const = 0;

    /**
     * Get the cross-term energy (receptor-ligand GB pairs, PAIRWISE mode only).
     * This is the solvent screening contribution to receptor-ligand electrostatics.
     */
    virtual double getGroupCrossTermEnergy(int groupIndex) const = 0;

    /**
     * Get the Born radii for atoms in a particle group.
     */
    virtual std::vector<double> getGroupBornRadii(int groupIndex) const = 0;

    /**
     * Get per-atom GB energies for a particle group.
     */
    virtual std::vector<double> getGroupAtomEnergies(int groupIndex) const = 0;

    /**
     * Compute the Hessian (second derivatives) for the GBSA force.
     */
    virtual std::vector<double> computeHessian(OpenMM::ContextImpl& context) = 0;
};

} // namespace GridForcePlugin

#endif // OPENMM_ISOLATEDGBSAFORCE_KERNELS_H_
