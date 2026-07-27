/* -------------------------------------------------------------------------- *
 *                              OpenMMGridForce                               *
 * -------------------------------------------------------------------------- *
 * SolvationFieldGrid: multi-slice scalar field sampled on a uniform lattice.
 *
 * Two fields use this container, both built once from a rigid receptor and
 * read at runtime with one lookup per ligand atom:
 *
 *   CROSS_GB One slice per probe Born radius R_k.
 *            Phi_k(x) = sum_j q_j * S(|x-r_j|) / f_GB(|x-r_j|, R_k, R_apo_j)
 *            Far part of the receptor-ligand GB cross term. Sliced in the
 *            ligand Born radius because 1/f_GB approaches 1/r only on the
 *            scale of sqrt(R_i R_j), which real ligand radii (up to ~2.3 nm
 *            at the OBC2 ceiling) push well past any usable near cutoff.
 *            Slices are log-spaced: R_i is strongly right-skewed.
 *
 *   MIRROR   One slice per ligand descreener bin b.
 *            Psi_b(x) = sum_j w_j * H(|x-r_j|, rho_off_j, s_b) * S(|x-r_j|)
 *            Linear response of the receptor GB energy to descreening by a
 *            ligand atom of bin b at x, where w_j = (dE/dR_j)(dR_j/dI_j) at
 *            the apo receptor. Same near/far split as CROSS_GB.
 *
 * Geometry (counts, spacing, origin) matches the DesolvationGrid the force
 * is configured with, so all grid reads for one ligand atom share indices.
 *
 * Storage layout, float32:
 *   m_data[slice * derivsPerPoint * numPoints + deriv * numPoints + point]
 *   point = ix * (ny*nz) + iy * nz + iz
 *   derivsPerPoint is 1, or NUM_DERIVATIVES for the Hermite methods (same
 *   27-value derivative order as DesolvationGrid).
 *
 * The field is built for one interpolation method and records which. For
 * TRICUBIC_BSPLINE the stored values are prefiltered coefficients, not node
 * values, so the reader must use the matching method.
 * -------------------------------------------------------------------------- */

#ifndef OPENMM_SOLVATIONFIELDGRID_H_
#define OPENMM_SOLVATIONFIELDGRID_H_

#include "internal/windowsExportGridForce.h"
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace GridForcePlugin {

class OPENMM_EXPORT_GRIDFORCE SolvationFieldGrid {
public:
    // File format constants
    static constexpr char MAGIC[8] = {'S', 'O', 'L', 'V', 'F', 'L', 'D', '\0'};
    static constexpr uint32_t VERSION = 1;
    static constexpr uint32_t HEADER_SIZE = 128;

    // Derivatives per point for Hermite interpolation, matching DesolvationGrid.
    static constexpr int NUM_DERIVATIVES = 27;

    /**
     * Which quantity the slices hold. Determines how the runtime kernel
     * combines the lookup with its near-shell correction.
     */
    enum FieldType {
        CROSS_GB = 0,  /**< Far part of the cross term, sliced in probe Born radius */
        MIRROR   = 1   /**< Far part of the receptor linear-response field */
    };

    SolvationFieldGrid();

    /**
     * @param nx, ny, nz     Lattice dimensions
     * @param spacing        Uniform spacing (nm)
     * @param numSlices      Number of slices
     * @param type           Which field this holds
     * @param interpMethod   InterpolationMethod the field is built for
     */
    SolvationFieldGrid(int nx, int ny, int nz, double spacing, int numSlices,
                       FieldType type, int interpMethod);

    static std::shared_ptr<SolvationFieldGrid> loadFromFile(const std::string& filename);
    void saveToFile(const std::string& filename) const;

    // ========== Geometry ==========

    int getNx() const { return m_counts[0]; }
    int getNy() const { return m_counts[1]; }
    int getNz() const { return m_counts[2]; }
    void getCounts(int& nx, int& ny, int& nz) const {
        nx = m_counts[0]; ny = m_counts[1]; nz = m_counts[2];
    }

    double getSpacing() const { return m_spacing; }

    void getOrigin(double& x, double& y, double& z) const {
        x = m_origin[0]; y = m_origin[1]; z = m_origin[2];
    }
    void setOrigin(double x, double y, double z) {
        m_origin[0] = x; m_origin[1] = y; m_origin[2] = z;
    }

    int getNumPoints() const { return m_numPoints; }

    // ========== Field parameters ==========

    FieldType getFieldType() const { return m_fieldType; }
    int getNumSlices() const { return m_numSlices; }

    /**
     * Per-slice parameter, ascending. For MIRROR this is the descreener
     * scaled radius s_b = S_b * (rho_b - offset) in nm; for CROSS_GB it is
     * the probe Born radius R_k the slice was built at. Both in nm.
     */
    const std::vector<double>& getSliceParameters() const { return m_sliceParameters; }
    void setSliceParameters(const std::vector<double>& params);

    /**
     * Radii (nm) of the switching function used to suppress the near field
     * during generation. S(r) is 0 below switchOn, 1 above switchOff, and a
     * smootherstep ramp between. The runtime kernel must apply the exactly
     * complementary near correction over pairs within switchOff.
     */
    double getSwitchOn() const { return m_switchOn; }
    double getSwitchOff() const { return m_switchOff; }
    void setSwitchRadii(double switchOn, double switchOff);

    // ========== Data ==========

    const std::vector<float>& getData() const { return m_data; }
    void setData(const std::vector<float>& data);
    void setData(std::vector<float>&& data);

    /**
     * Offset of a slice's value array within getData().
     */
    size_t getSliceOffset(int slice) const {
        return static_cast<size_t>(slice) * getDerivsPerPoint() * m_numPoints;
    }

    float getValue(int slice, int ix, int iy, int iz) const {
        return m_data[getSliceOffset(slice) + ix * m_nyz + iy * m_counts[2] + iz];
    }

    // ========== Interpolation ==========

    /** InterpolationMethod this field was built for. */
    int getInterpolationMethod() const { return m_interpMethod; }

    bool hasDerivatives() const { return getDerivsPerPoint() > 1; }

    int getDerivsPerPoint() const {
        return (m_interpMethod == 2 || m_interpMethod == 3) ? NUM_DERIVATIVES : 1;
    }

    size_t getMemoryBytes() const { return m_data.size() * sizeof(float); }

private:
    std::vector<int> m_counts;     // [nx, ny, nz]
    double m_spacing;              // nm
    std::vector<double> m_origin;  // [ox, oy, oz] nm

    FieldType m_fieldType;
    int m_numSlices;
    std::vector<double> m_sliceParameters;
    double m_switchOn;
    double m_switchOff;
    int m_interpMethod;

    int m_nyz;
    int m_numPoints;

    std::vector<float> m_data;
};

}  // namespace GridForcePlugin

#endif  // OPENMM_SOLVATIONFIELDGRID_H_
