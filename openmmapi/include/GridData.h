#ifndef OPENMM_GRIDDATA_H_
#define OPENMM_GRIDDATA_H_

#include "openmm/Vec3.h"
#include "GridForceTypes.h"
#include "internal/windowsExportGridForce.h"
#include <vector>
#include <string>
#include <memory>

namespace GridForcePlugin {

/**
 * GridData holds grid values, derivatives, and metadata.
 * Designed to be shared across multiple GridForce instances via shared_ptr
 * for memory efficiency with large grids.
 *
 * GridData is immutable after construction to ensure thread-safe concurrent
 * read access across multiple forces.
 */
class OPENMM_EXPORT_GRIDFORCE GridData {
public:
    /**
     * Create an empty GridData.
     */
    GridData();

    /**
     * Create GridData with specified dimensions and spacing.
     *
     * @param nx, ny, nz  Grid dimensions
     * @param dx, dy, dz  Grid spacing in nm
     */
    GridData(int nx, int ny, int nz, double dx, double dy, double dz);

    /**
     * Copy constructor (shallow copy - shares underlying data).
     */
    GridData(const GridData& other) = default;

    /**
     * Assignment operator (shallow copy - shares underlying data).
     */
    GridData& operator=(const GridData& other) = default;

    /**
     * Load grid data from a binary file.
     *
     * @param filename  Path to grid file (.grid or .bin format)
     * @return shared_ptr to the loaded GridData
     */
    static std::shared_ptr<GridData> loadFromFile(const std::string& filename);

    /**
     * Save grid data to a binary file.
     *
     * @param filename  Path to output file
     */
    void saveToFile(const std::string& filename) const;

    // Dimension accessors
    int getNx() const { return m_counts[0]; }
    int getNy() const { return m_counts[1]; }
    int getNz() const { return m_counts[2]; }
    void getCounts(int& nx, int& ny, int& nz) const {
        nx = m_counts[0]; ny = m_counts[1]; nz = m_counts[2];
    }

    // Spacing accessors
    double getDx() const { return m_spacing[0]; }
    double getDy() const { return m_spacing[1]; }
    double getDz() const { return m_spacing[2]; }
    void getSpacing(double& dx, double& dy, double& dz) const {
        dx = m_spacing[0]; dy = m_spacing[1]; dz = m_spacing[2];
    }

    // Origin accessors
    void getOrigin(double& x, double& y, double& z) const {
        x = m_origin[0]; y = m_origin[1]; z = m_origin[2];
    }
    void setOrigin(double x, double y, double z) {
        m_origin[0] = x; m_origin[1] = y; m_origin[2] = z;
    }

    // Data accessors (const only - GridData is immutable)
    const std::vector<double>& getValues() const { return *m_vals; }
    const std::vector<double>& getDerivatives() const { return *m_derivatives; }

    bool hasDerivatives() const { return m_derivatives && !m_derivatives->empty(); }

    /**
     * Get grid value at specific index (no bounds checking for performance).
     *
     * @param ix, iy, iz  Grid indices
     */
    double getValue(int ix, int iy, int iz) const {
        return (*m_vals)[ix * m_nyz + iy * m_counts[2] + iz];
    }

    /**
     * Get derivative at specific index.
     *
     * @param derivIndex  Derivative index (0-26 for triquintic)
     * @param ix, iy, iz  Grid indices
     */
    double getDerivative(int derivIndex, int ix, int iy, int iz) const {
        if (!m_derivatives) return 0.0;
        int totalPoints = m_counts[0] * m_counts[1] * m_counts[2];
        return (*m_derivatives)[derivIndex * totalPoints + ix * m_nyz + iy * m_counts[2] + iz];
    }

    // Metadata accessors
    const std::string& getGridType() const { return m_gridType; }
    void setGridType(const std::string& type) { m_gridType = type; }

    double getInvPower() const { return m_invPower; }
    void setInvPower(double invPower) { m_invPower = invPower; }

    InvPowerMode getInvPowerMode() const { return m_invPowerMode; }
    void setInvPowerMode(InvPowerMode mode) { m_invPowerMode = mode; }

    // Internal data access (for GridForce compatibility)
    std::shared_ptr<std::vector<double>> getValuesPtr() const { return m_vals; }
    std::shared_ptr<std::vector<double>> getDerivativesPtr() const { return m_derivatives; }

    /**
     * Set grid values (for construction/builder pattern).
     */
    void setValues(const std::vector<double>& vals);
    void setValues(std::shared_ptr<std::vector<double>> vals) { m_vals = vals; }

    /**
     * Set derivatives (for construction/builder pattern).
     */
    void setDerivatives(const std::vector<double>& derivs);
    void setDerivatives(std::vector<double>&& derivs);  // Move overload for large arrays
    void setDerivatives(std::shared_ptr<std::vector<double>> derivs) { m_derivatives = derivs; }

private:
    std::vector<int> m_counts;      // [nx, ny, nz]
    std::vector<double> m_spacing;  // [dx, dy, dz] in nm
    std::vector<double> m_origin;   // [ox, oy, oz] in nm

    std::shared_ptr<std::vector<double>> m_vals;        // Grid values (shared)
    std::shared_ptr<std::vector<double>> m_derivatives; // Derivatives (shared, optional)

    int m_nyz;  // Cached: ny * nz for index calculation

    std::string m_gridType;     // e.g., "charge", "ljr", "lja"
    double m_invPower;          // Inverse power for transformations
    InvPowerMode m_invPowerMode; // Transformation mode (NONE, RUNTIME, STORED)
};

} // namespace GridForcePlugin

#endif
