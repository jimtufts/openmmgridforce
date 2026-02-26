#ifndef OPENMM_ISOLATEDSITEFORCE_H_
#define OPENMM_ISOLATEDSITEFORCE_H_

/* -------------------------------------------------------------------------- *
 *                              OpenMMGridForce                               *
 * -------------------------------------------------------------------------- *
 * IsolatedSiteForce: flat-bottom sphere restraint on per-group center of     *
 * mass. Keeps each ligand replica within a binding site sphere.              *
 *                                                                            *
 * E = 0.5 * k * max(0, r_com - maxR)^2                                      *
 *                                                                            *
 * where r_com = |COM_group - siteCenter| and COM is mass-weighted.           *
 * Forces are distributed to atoms proportional to mass:                      *
 *   F_i = -dE/dr * (COM - center) / r * m_i / M_total                       *
 * -------------------------------------------------------------------------- */

#include <string>
#include <vector>

#include "internal/windowsExportGridForce.h"
#include "openmm/Context.h"
#include "openmm/Force.h"

using namespace OpenMM;

namespace GridForcePlugin {

class OPENMM_EXPORT_GRIDFORCE IsolatedSiteForce : public OpenMM::Force {
public:
    IsolatedSiteForce();

    // ========== Template Size ==========

    int getNumAtoms() const;
    void setNumAtoms(int numAtoms);

    // ========== Site Parameters ==========

    /**
     * Set the center of the binding site sphere.
     * @param x, y, z  center coordinates (nm)
     */
    void setSiteCenter(double x, double y, double z);
    void getSiteCenter(double& x, double& y, double& z) const;

    /**
     * Set the flat-bottom radius. No restraint is applied when COM is
     * within this radius of the site center.
     * @param maxR  flat-bottom radius (nm)
     */
    void setMaxRadius(double maxR);
    double getMaxRadius() const;

    /**
     * Set the harmonic force constant for the restraint.
     * @param k  force constant (kJ/mol/nm^2)
     */
    void setForceConstant(double k);
    double getForceConstant() const;

    // ========== Atom Masses ==========

    /**
     * Set per-atom masses for COM computation.
     * Must have numAtoms elements. Units: daltons (amu).
     */
    void setAtomMasses(const std::vector<double>& masses);
    const std::vector<double>& getAtomMasses() const;

    // ========== Alchemical Scaling ==========

    double getGlobalScalingFactor() const { return m_globalScalingFactor; }
    void setGlobalScalingFactor(double factor) { m_globalScalingFactor = factor; }

    double getGroupScalingFactor(int groupIndex) const;
    void setGroupScalingFactor(int groupIndex, double factor);

    // ========== Particle Groups ==========

    /**
     * Add a particle group (replica). Each group contains numAtoms particle
     * indices from the System, representing one copy of the ligand.
     *
     * @param name     a name for this group
     * @param indices  particle indices in the System (must have numAtoms elements)
     * @return the index of the group that was added
     */
    int addParticleGroup(const std::string& name, const std::vector<int>& indices);

    int getNumParticleGroups() const { return static_cast<int>(m_particleGroups.size()); }

    void getParticleGroup(int index, std::string& name, std::vector<int>& indices) const;

    // ========== Per-Group Energy ==========

    double getGroupEnergy(int groupIndex) const;

    /**
     * Get energies for all particle groups in a single call.
     * Only available after calling getState() with energy.
     *
     * @return vector of energies (kJ/mol), one per group
     */
    std::vector<double> getParticleGroupEnergies() const;

    // ========== Parameter Updates ==========

    void updateParametersInContext(OpenMM::Context& context);

    // ========== OpenMM Force Interface ==========

    bool usesPeriodicBoundaryConditions() const override { return false; }

protected:
    OpenMM::ForceImpl* createImpl() const override;

private:
    int m_numAtoms;

    // Site geometry
    double m_centerX, m_centerY, m_centerZ;
    double m_maxRadius;
    double m_forceConstant;

    // Per-atom masses for COM
    std::vector<double> m_masses;

    // Alchemical scaling
    double m_globalScalingFactor;
    std::vector<double> m_groupScalingFactors;

    // Particle groups
    struct ParticleGroupInfo {
        std::string name;
        std::vector<int> indices;
    };
    std::vector<ParticleGroupInfo> m_particleGroups;

    // Per-group energy cache
    mutable std::vector<double> m_groupEnergies;

    friend class IsolatedSiteForceImpl;
};

}  // namespace GridForcePlugin

#endif /*OPENMM_ISOLATEDSITEFORCE_H_*/
