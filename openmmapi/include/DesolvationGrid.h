/* -------------------------------------------------------------------------- *
 *                              OpenMMGridForce                               *
 * -------------------------------------------------------------------------- *
 * DesolvationGrid: Grid data container for grid-based GBSA solvation.
 *
 * Stores HCT integral values computed with a probe radius, along with
 * binned correction terms that enable exact computation for any ligand
 * atom radius at runtime.
 *
 * Grid storage per point (2-bin configuration, float32):
 *   Without derivatives:
 *     - hct_probe: HCT sum using probe radius
 *     - N_bin, A_bin, B_bin: correction terms per bin
 *     Total: 1 + 3*n_bins floats per point
 *
 *   With derivatives (for triquintic/tricubic interpolation):
 *     - hct_probe_derivs[27]: HCT + 26 spatial derivatives
 *     - N_bin_derivs[27], A_bin_derivs[27], B_bin_derivs[27] per bin
 *     Total: 27 * (1 + 3*n_bins) floats per point
 *
 * At runtime, the exact HCT for ligand radius R_i is:
 *   HCT(R_i) = HCT_probe + correction(R_i, N, A, B)
 *
 * where correction uses the analytical formula:
 *   dI = (1/R_i - 1/R_probe) * [N - 0.25*A*(1/R_i + 1/R_probe)]
 *      + B * ln(R_i/R_probe)
 *
 * Derivative storage order (27 values per point, same as GridForce):
 *   0=f, 1=fx, 2=fy, 3=fz, 4=fxx, 5=fxy, 6=fxz, 7=fyy, 8=fyz, 9=fzz, ...
 * -------------------------------------------------------------------------- */

#ifndef OPENMM_DESOLVATIONGRID_H_
#define OPENMM_DESOLVATIONGRID_H_

#include "internal/windowsExportGridForce.h"
#include <vector>
#include <string>
#include <memory>
#include <cstdint>

namespace GridForcePlugin {

/**
 * DesolvationGrid holds HCT grid values and correction terms for GBSA.
 *
 * The grid is designed to be shared across multiple GBSAGridForce instances
 * via shared_ptr for memory efficiency. Grid data is immutable after
 * construction to ensure thread-safe concurrent read access.
 */
class OPENMM_EXPORT_GRIDFORCE DesolvationGrid {
public:
    // File format constants
    static constexpr char MAGIC[8] = {'D', 'E', 'S', 'O', 'L', 'V', '\0', '\0'};
    static constexpr uint32_t VERSION = 3;  // v3 adds optional receptor desolvation
    static constexpr uint32_t MIN_SUPPORTED_VERSION = 2;  // Can read v2 files
    static constexpr uint32_t HEADER_SIZE = 128;

    // Number of derivatives per grid point for triquintic interpolation
    static constexpr int NUM_DERIVATIVES = 27;

    // Default physical parameters
    static constexpr double DEFAULT_PROBE_RADIUS = 0.14;  // nm (water probe)
    static constexpr double DIELECTRIC_OFFSET = 0.009;    // nm

    /**
     * Create an empty DesolvationGrid.
     */
    DesolvationGrid();

    /**
     * Create DesolvationGrid with specified dimensions.
     *
     * @param nx, ny, nz    Grid dimensions
     * @param spacing       Uniform grid spacing in nm
     * @param probeRadius   Probe radius used for HCT computation (nm)
     * @param rThresholds   Offset radius thresholds for correction bins (nm)
     * @param hasDerivatives If true, allocate space for 27 derivatives per point
     */
    DesolvationGrid(int nx, int ny, int nz, double spacing,
                    double probeRadius,
                    const std::vector<double>& rThresholds,
                    bool hasDerivatives = false);

    /**
     * Load grid from a binary file.
     *
     * @param filename  Path to .desolv grid file
     * @return shared_ptr to loaded grid
     */
    static std::shared_ptr<DesolvationGrid> loadFromFile(const std::string& filename);

    /**
     * Save grid to a binary file.
     *
     * @param filename  Path to output file
     */
    void saveToFile(const std::string& filename) const;

    // ========== Dimension accessors ==========

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

    // ========== Grid parameters ==========

    double getProbeRadius() const { return m_probeRadius; }
    int getNumBins() const { return static_cast<int>(m_rThresholds.size()); }
    const std::vector<double>& getRThresholds() const { return m_rThresholds; }

    /**
     * Get the bin index for a given offset radius.
     * Returns the first bin whose threshold >= radius, or last bin if none.
     */
    int getBinForRadius(double offsetRadius) const;

    // ========== Data accessors (const - grid is immutable) ==========

    /**
     * Get the HCT probe array (base HCT values computed with probe radius).
     * Layout: [iz + nz * (iy + ny * ix)] for point (ix, iy, iz)
     */
    const std::vector<float>& getHctProbe() const { return m_hctProbe; }

    /**
     * Get correction N array (count of atoms in R-dependent regime).
     * Layout: [bin * n_points + point_index]
     */
    const std::vector<float>& getCorrectionN() const { return m_correctionN; }

    /**
     * Get correction A array (A = r - S^2/r summed).
     * Layout: [bin * n_points + point_index]
     */
    const std::vector<float>& getCorrectionA() const { return m_correctionA; }

    /**
     * Get correction B array (B = 0.5/r summed).
     * Layout: [bin * n_points + point_index]
     */
    const std::vector<float>& getCorrectionB() const { return m_correctionB; }

    /**
     * Get HCT value at specific grid point.
     */
    float getHctProbeValue(int ix, int iy, int iz) const {
        return m_hctProbe[ix * m_nyz + iy * m_counts[2] + iz];
    }

    /**
     * Get correction values at specific grid point and bin.
     */
    void getCorrectionValues(int bin, int ix, int iy, int iz,
                            float& N, float& A, float& B) const {
        int pointIdx = ix * m_nyz + iy * m_counts[2] + iz;
        int offset = bin * m_numPoints + pointIdx;
        N = m_correctionN[offset];
        A = m_correctionA[offset];
        B = m_correctionB[offset];
    }

    // ========== Memory info ==========

    /**
     * Total memory used by grid data in bytes.
     */
    size_t getMemoryBytes() const;

    /**
     * Number of floats stored per grid point.
     */
    int getFloatsPerPoint() const {
        int base = 1 + 3 * getNumBins();
        return m_hasDerivatives ? base * NUM_DERIVATIVES : base;
    }

    // ========== Derivative support ==========

    /**
     * Check if the grid has precomputed derivatives for triquintic/tricubic interpolation.
     */
    bool hasDerivatives() const { return m_hasDerivatives; }

    /**
     * Enable derivative storage. Must be called before setting data.
     * When enabled, all data arrays store 27 values per grid point.
     */
    void setHasDerivatives(bool hasDerivs) { m_hasDerivatives = hasDerivs; }

    /**
     * Get number of derivatives per point (27 for triquintic, 1 for values only).
     */
    int getNumDerivsPerPoint() const { return m_hasDerivatives ? NUM_DERIVATIVES : 1; }

    // ========== Data setters (for construction) ==========

    void setHctProbe(const std::vector<float>& data);
    void setHctProbe(std::vector<float>&& data);

    void setCorrectionN(const std::vector<float>& data);
    void setCorrectionN(std::vector<float>&& data);

    void setCorrectionA(const std::vector<float>& data);
    void setCorrectionA(std::vector<float>&& data);

    void setCorrectionB(const std::vector<float>& data);
    void setCorrectionB(std::vector<float>&& data);

private:
    // Grid dimensions
    std::vector<int> m_counts;     // [nx, ny, nz]
    double m_spacing;              // Uniform spacing in nm
    std::vector<double> m_origin;  // [ox, oy, oz] in nm

    // Grid parameters
    double m_probeRadius;                // Probe radius used for HCT (nm)
    std::vector<double> m_rThresholds;   // Bin thresholds (offset radii, nm)

    // Cached values
    int m_nyz;         // ny * nz for index calculation
    int m_numPoints;   // nx * ny * nz

    // Derivative flag
    bool m_hasDerivatives;  // If true, arrays store 27 values per point

    // Grid data (float32 for memory efficiency)
    // Without derivatives: [n_points] or [n_bins * n_points]
    // With derivatives: [27 * n_points] or [27 * n_bins * n_points]
    // Layout with derivs: [deriv_idx * n_points + point_idx] (derivative-major)
    std::vector<float> m_hctProbe;     // [derivs * n_points]
    std::vector<float> m_correctionN;  // [derivs * n_bins * n_points]
    std::vector<float> m_correctionA;  // [derivs * n_bins * n_points]
    std::vector<float> m_correctionB;  // [derivs * n_bins * n_points]
};

} // namespace GridForcePlugin

#endif // OPENMM_DESOLVATIONGRID_H_
