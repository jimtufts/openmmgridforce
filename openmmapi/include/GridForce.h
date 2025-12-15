#ifndef OPENMM_GRIDFORCE_H_
#define OPENMM_GRIDFORCE_H_

/* -------------------------------------------------------------------------- *
 *                              OpenMMGridForce                               *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2008-2012 Stanford University and the Authors.      *
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
#include <vector>

#include "internal/windowsExportGridForce.h"
#include "openmm/Context.h"
#include "openmm/Force.h"
#include "openmm/Vec3.h"

using namespace OpenMM;

namespace GridForcePlugin {

/**
 * This class implements the AlGDock Nonbond interaction.
 */

class OPENMM_EXPORT_GRIDFORCE GridForce : public OpenMM::Force {
   public:
    /**
     * Create a GridForce.
     * @param spacing       the grid space
     * @param vals          the value at each grid
     */
    GridForce();

    /**
     * Get the force field parameters for a Nonbond Energy term
     * 
     */
    void addGridCounts(int nx, int ny, int nz);
    void addGridSpacing(double dx, double dy, double dz);  // length unit is 'nm'

    void addGridValue(double val);
    void addScalingFactor(double val);
    void setScalingFactor(int index, double val);

    /**
     * Enable or disable automatic calculation of scaling factors from the System.
     * When enabled, scaling factors will be extracted from the NonbondedForce
     * in the System based on the scalingProperty setting. When disabled (default),
     * scaling factors must be added manually using addScalingFactor().
     *
     * @param enable  if true, auto-calculate scaling factors; if false, use manual values
     */
    void setAutoCalculateScalingFactors(bool enable);

    /**
     * Get whether automatic scaling factor calculation is enabled.
     *
     * @return  true if auto-calculation is enabled, false otherwise
     */
    bool getAutoCalculateScalingFactors() const;

    /**
     * Set the property to use for automatic scaling factor calculation.
     * This is only used when autoCalculateScalingFactors is enabled.
     *
     * Supported values:
     * - "charge": Use particle charges (for electrostatic grids)
     * - "ljr": Use sqrt(epsilon) * (2*sigma)^6 (for LJ repulsive grids)
     * - "lja": Use sqrt(epsilon) * (2*sigma)^3 (for LJ attractive grids)
     *
     * @param property  the scaling property to use
     */
    void setScalingProperty(const std::string& property);

    /**
     * Get the current scaling property setting.
     *
     * @return  the scaling property name
     */
    const std::string& getScalingProperty() const;

    /**
     * Set the inverse power parameter for grid value transformation.
     * When inv_power > 0, the interpolated grid value is raised to this power
     * before being multiplied by the scaling factor. This reverses the grid
     * transformation where grid values were stored as G^(1/inv_power).
     *
     * For example, if grids were transformed as G^(1/4), set inv_power=4.0
     * to compute: energy = scaling_factor * (interpolated)^4
     *
     * @param inv_power  exponent to apply to interpolated values (default: 0.0, meaning no transformation)
     */
    void setInvPower(double inv_power);

    /**
     * Get the current inverse power parameter.
     * @return  the inv_power value
     */
    double getInvPower() const;

    void getGridParameters(std::vector<int> &g_counts,
                           std::vector<double> &g_spacing,
                           std::vector<double> &g_vals,
                           std::vector<double> &g_scaling_factors) const;

    /**
     *
     */
    void updateParametersInContext(Context &context);

   protected:
    ForceImpl *createImpl() const;

   private:
    std::vector<int> m_counts;
    std::vector<double> m_spacing;  // the length unit is 'nm'
    std::vector<double> m_vals;
    std::vector<double> m_scaling_factors;
    double m_inv_power;
    bool m_autoCalculateScalingFactors;
    std::string m_scalingProperty;
};

}  // namespace GridForcePlugin

#endif /*OPENMM_GRIDFORCE_H_*/
