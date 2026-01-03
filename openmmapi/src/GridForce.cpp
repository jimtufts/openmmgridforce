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

#include "GridForce.h"
#include "internal/GridForceImpl.h"

#include "openmm/Force.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/AssertionUtilities.h"

#include <iostream>
#include <fstream>
#include <cstdint>
#include <cmath>

using namespace OpenMM;
using namespace std;

namespace GridForcePlugin {

GridForce::GridForce() : m_inv_power(0.0), m_invPowerMode(InvPowerMode::NONE), m_gridCap(41840.0), m_outOfBoundsRestraint(10000.0), m_interpolationMethod(0),
                         m_autoCalculateScalingFactors(false), m_scalingProperty(""),
                         m_autoGenerateGrid(false), m_gridType(""), m_gridOrigin({0.0, 0.0, 0.0}),
                         m_computeDerivatives(false) {
    //
}

void GridForce::addGridCounts(int nx, int ny, int nz) {
    m_counts.push_back(nx);
    m_counts.push_back(ny);
    m_counts.push_back(nz);
}

void GridForce::addGridSpacing(double dx, double dy, double dz) {
    // the length unit is 'nm'
    m_spacing.push_back(dx);
    m_spacing.push_back(dy);
    m_spacing.push_back(dz);
}

void GridForce::addGridValue(double val) {
    m_vals.push_back(val);
}

void GridForce::setGridValues(const std::vector<double>& vals) {
    m_vals = vals;
}

const std::vector<double>& GridForce::getGridValues() const {
    return m_vals;
}

void GridForce::addScalingFactor(double val) {
    m_scaling_factors.push_back(val);
}

void GridForce::setScalingFactor(int index, double val) {
    m_scaling_factors[index] = val;
}

void GridForce::setScalingFactors(const std::vector<double>& vals) {
    m_scaling_factors = vals;
}

void GridForce::setInvPowerMode(InvPowerMode mode, double inv_power) {
    // Validation
    if (mode != InvPowerMode::NONE && inv_power <= 0.0) {
        throw OpenMMException("GridForce: inv_power must be > 0 when mode != NONE");
    }
    if (mode == InvPowerMode::NONE && inv_power != 0.0) {
        throw OpenMMException("GridForce: inv_power must be 0 when mode == NONE");
    }

    // Check for conflicting mode changes after grid is loaded
    if (!m_vals.empty()) {
        if (m_invPowerMode == InvPowerMode::STORED && mode == InvPowerMode::RUNTIME) {
            throw OpenMMException(
                "GridForce: Cannot set RUNTIME mode on grid that already has STORED transformation. "
                "This would apply transformation twice!");
        }
        if (m_invPowerMode == InvPowerMode::RUNTIME && mode == InvPowerMode::STORED) {
            throw OpenMMException(
                "GridForce: Cannot set STORED mode on untransformed grid loaded with RUNTIME mode. "
                "Call applyInvPowerTransformation() first.");
        }
    }

    m_invPowerMode = mode;
    m_inv_power = inv_power;
}

InvPowerMode GridForce::getInvPowerMode() const {
    return m_invPowerMode;
}

void GridForce::applyInvPowerTransformation() {
    // Validation
    if (m_invPowerMode != InvPowerMode::RUNTIME) {
        throw OpenMMException(
            "GridForce: Can only call applyInvPowerTransformation() when mode == RUNTIME. "
            "Current mode: " + std::to_string(static_cast<int>(m_invPowerMode)));
    }

    if (m_inv_power <= 0.0) {
        throw OpenMMException("GridForce: inv_power must be > 0");
    }

    if (!m_derivatives.empty()) {
        throw OpenMMException(
            "GridForce: Cannot apply runtime transformation to grids with analytical derivatives. "
            "Use mode STORED with pre-transformed grids instead.");
    }

    if (m_vals.empty()) {
        throw OpenMMException("GridForce: No grid values to transform. Load grid first.");
    }

    // Apply transformation: G -> sign(G) * |G|^(1/inv_power)
    for (size_t i = 0; i < m_vals.size(); ++i) {
        if (m_vals[i] != 0.0) {
            double sign = (m_vals[i] >= 0.0) ? 1.0 : -1.0;
            m_vals[i] = sign * std::pow(std::abs(m_vals[i]), 1.0 / m_inv_power);
        }
    }

    // Update mode to STORED since grid is now transformed
    m_invPowerMode = InvPowerMode::STORED;
}

void GridForce::setInvPower(double inv_power) {
    // DEPRECATED: Use setInvPowerMode() instead for explicit control
    static bool warned = false;
    if (!warned) {
        std::cerr << "WARNING: GridForce::setInvPower() is deprecated. "
                  << "Use setInvPowerMode() instead for explicit control over transformation timing."
                  << std::endl;
        warned = true;
    }

    // If grid values are already loaded and inv_power is changing, transform the values
    // Uses sign-preserving power transformations to handle negative values correctly
    if (!m_vals.empty() && inv_power != m_inv_power) {
        if (m_inv_power == 0.0 && inv_power != 0.0) {
            // Transform from G to G^(1/n) (works for both positive and negative n)
            // Use sign-preserving: sign(G) * |G|^(1/n), preserve zeros
            for (size_t i = 0; i < m_vals.size(); ++i) {
                if (m_vals[i] == 0.0) {
                    m_vals[i] = 0.0;
                } else {
                    double sign = (m_vals[i] >= 0.0) ? 1.0 : -1.0;
                    m_vals[i] = sign * std::pow(std::abs(m_vals[i]), 1.0 / inv_power);
                }
            }
        } else if (m_inv_power != 0.0 && inv_power == 0.0) {
            // Transform from G^(1/m) back to G (works for both positive and negative m)
            // Use sign-preserving: sign(G) * |G|^m, preserve zeros
            for (size_t i = 0; i < m_vals.size(); ++i) {
                if (m_vals[i] == 0.0) {
                    m_vals[i] = 0.0;
                } else {
                    double sign = (m_vals[i] >= 0.0) ? 1.0 : -1.0;
                    m_vals[i] = sign * std::pow(std::abs(m_vals[i]), m_inv_power);
                }
            }
        } else if (m_inv_power != 0.0 && inv_power != 0.0) {
            // Transform from G^(1/m) to G^(1/n) (works for any combination of signs)
            // First: G^(1/m) -> G by raising to power m
            // Then: G -> G^(1/n) by raising to power 1/n
            // Combined: (G^(1/m))^(m/n) = G^(1/n)
            // Use sign-preserving: sign(G) * |G|^(m/n), preserve zeros
            double power_factor = m_inv_power / inv_power;
            for (size_t i = 0; i < m_vals.size(); ++i) {
                if (m_vals[i] == 0.0) {
                    m_vals[i] = 0.0;
                } else {
                    double sign = (m_vals[i] >= 0.0) ? 1.0 : -1.0;
                    m_vals[i] = sign * std::pow(std::abs(m_vals[i]), power_factor);
                }
            }
        }

        // Transform LOGARITHMIC derivatives using analytical formulas
        // Derivatives are stored as: [0]=f, [1-3]=L1 (G'/G), [4-9]=L2 (G''/G)
        // For H = G^p: L1_H = p*L1_G, L2_H = p*L2_G + p*(p-1)*L1_G²
        if (!m_derivatives.empty()) {
            double p; // power transformation factor
            if (m_inv_power == 0.0 && inv_power != 0.0) {
                p = 1.0 / inv_power;
            } else if (m_inv_power != 0.0 && inv_power == 0.0) {
                p = m_inv_power;
            } else {
                p = m_inv_power / inv_power;
            }

            size_t num_points = m_vals.size();

            for (size_t pt = 0; pt < num_points; ++pt) {
                // Update function value (already done above)
                m_derivatives[0 * num_points + pt] = m_vals[pt];

                // Get old L1 values before transformation
                double L1_x_old = m_derivatives[1 * num_points + pt];
                double L1_y_old = m_derivatives[2 * num_points + pt];
                double L1_z_old = m_derivatives[3 * num_points + pt];

                // Transform first logarithmic derivatives: L1_new = p * L1_old
                m_derivatives[1 * num_points + pt] = p * L1_x_old;
                m_derivatives[2 * num_points + pt] = p * L1_y_old;
                m_derivatives[3 * num_points + pt] = p * L1_z_old;

                // Transform second logarithmic derivatives: L2_new = p*L2_old + p*(p-1)*L1_old²
                // L2_xx
                m_derivatives[4 * num_points + pt] = p * m_derivatives[4 * num_points + pt] +
                                                      p * (p - 1) * L1_x_old * L1_x_old;
                // L2_yy
                m_derivatives[5 * num_points + pt] = p * m_derivatives[5 * num_points + pt] +
                                                      p * (p - 1) * L1_y_old * L1_y_old;
                // L2_zz
                m_derivatives[6 * num_points + pt] = p * m_derivatives[6 * num_points + pt] +
                                                      p * (p - 1) * L1_z_old * L1_z_old;
                // L2_xy
                m_derivatives[7 * num_points + pt] = p * m_derivatives[7 * num_points + pt] +
                                                      p * (p - 1) * L1_x_old * L1_y_old;
                // L2_xz
                m_derivatives[8 * num_points + pt] = p * m_derivatives[8 * num_points + pt] +
                                                      p * (p - 1) * L1_x_old * L1_z_old;
                // L2_yz
                m_derivatives[9 * num_points + pt] = p * m_derivatives[9 * num_points + pt] +
                                                      p * (p - 1) * L1_y_old * L1_z_old;
            }
        }
    }

    m_inv_power = inv_power;

    // Update mode for backward compatibility
    if (inv_power > 0.0) {
        m_invPowerMode = InvPowerMode::STORED;
    } else {
        m_invPowerMode = InvPowerMode::NONE;
    }
}

double GridForce::getInvPower() const {
    return m_inv_power;
}

void GridForce::setGridCap(double uMax) {
    m_gridCap = uMax;
}

double GridForce::getGridCap() const {
    return m_gridCap;
}

void GridForce::setOutOfBoundsRestraint(double k) {
    m_outOfBoundsRestraint = k;
}

double GridForce::getOutOfBoundsRestraint() const {
    return m_outOfBoundsRestraint;
}

void GridForce::setInterpolationMethod(int method) {
    if (method < 0 || method > 3) {
        throw OpenMMException("GridForce: Invalid interpolation method. Must be 0 (trilinear), 1 (cubic B-spline), 2 (tricubic), or 3 (quintic Hermite)");
    }
    m_interpolationMethod = method;
}

int GridForce::getInterpolationMethod() const {
    return m_interpolationMethod;
}

void GridForce::getGridParameters(std::vector<int> &g_counts,
                                  std::vector<double> &g_spacing,
                                  std::vector<double> &g_vals,
                                  std::vector<double> &g_scaling_factors) const {
    g_counts = m_counts;
    g_spacing = m_spacing;
    g_vals = m_vals;
    g_scaling_factors = m_scaling_factors;
}

ForceImpl *GridForce::createImpl() const {
    return new GridForceImpl(*this);
}

void GridForce::updateParametersInContext(Context &context) {
    dynamic_cast<GridForceImpl &>(getImplInContext(context)).updateParametersInContext(getContextImpl(context));
}

void GridForce::setAutoCalculateScalingFactors(bool enable) {
    m_autoCalculateScalingFactors = enable;
}

bool GridForce::getAutoCalculateScalingFactors() const {
    return m_autoCalculateScalingFactors;
}

void GridForce::setScalingProperty(const std::string& property) {
    // Don't validate here - validation happens in kernel initialization where exceptions are properly handled
    // This avoids SWIG exception translation issues
    m_scalingProperty = property;
}

const std::string& GridForce::getScalingProperty() const {
    return m_scalingProperty;
}

void GridForce::setAutoGenerateGrid(bool enable) {
    m_autoGenerateGrid = enable;
}

bool GridForce::getAutoGenerateGrid() const {
    return m_autoGenerateGrid;
}

void GridForce::setGridType(const std::string& type) {
    m_gridType = type;
}

const std::string& GridForce::getGridType() const {
    return m_gridType;
}

void GridForce::setGridOrigin(double x, double y, double z) {
    m_gridOrigin = {x, y, z};
}

void GridForce::getGridOrigin(double& x, double& y, double& z) const {
    if (m_gridOrigin.size() != 3) {
        x = y = z = 0.0;
    } else {
        x = m_gridOrigin[0];
        y = m_gridOrigin[1];
        z = m_gridOrigin[2];
    }
}

void GridForce::setReceptorAtoms(const std::vector<int>& atomIndices) {
    m_receptorAtoms = atomIndices;
}

const std::vector<int>& GridForce::getReceptorAtoms() const {
    return m_receptorAtoms;
}

void GridForce::setLigandAtoms(const std::vector<int>& atomIndices) {
    m_ligandAtoms = atomIndices;
}

const std::vector<int>& GridForce::getLigandAtoms() const {
    return m_ligandAtoms;
}

void GridForce::setReceptorPositions(const std::vector<Vec3>& positions) {
    m_receptorPositions = positions;
}

void GridForce::setReceptorPositionsFromArrays(const std::vector<double>& x,
                                                const std::vector<double>& y,
                                                const std::vector<double>& z) {
    if (x.size() != y.size() || y.size() != z.size()) {
        throw OpenMMException("GridForce: x, y, z arrays must have the same size");
    }

    m_receptorPositions.clear();
    m_receptorPositions.reserve(x.size());
    for (size_t i = 0; i < x.size(); i++) {
        m_receptorPositions.push_back(Vec3(x[i], y[i], z[i]));
    }
}

const std::vector<Vec3>& GridForce::getReceptorPositions() const {
    return m_receptorPositions;
}

void GridForce::setComputeDerivatives(bool compute) {
    m_computeDerivatives = compute;
}

bool GridForce::getComputeDerivatives() const {
    return m_computeDerivatives;
}

bool GridForce::hasDerivatives() const {
    return !m_derivatives.empty();
}

const std::vector<double>& GridForce::getDerivatives() const {
    return m_derivatives;
}

void GridForce::setDerivatives(const std::vector<double>& derivs) {
    m_derivatives = derivs;
}

void GridForce::loadFromFile(const std::string& filename) {
    // Binary file I/O implementation
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw OpenMMException("GridForce: Cannot open file '" + filename + "'");
    }

    // Read and validate magic number
    char magic[8];
    file.read(magic, 8);
    if (std::string(magic, 8) != std::string("OMGRID\0\0", 8)) {
        file.close();
        throw OpenMMException("GridForce: Invalid file format (bad magic number)");
    }

    // Read header
    uint32_t version, header_size;
    file.read(reinterpret_cast<char*>(&version), sizeof(uint32_t));
    file.read(reinterpret_cast<char*>(&header_size), sizeof(uint32_t));

    if (version < 1 || version > 3) {
        file.close();
        throw OpenMMException("GridForce: Unsupported file version " + std::to_string(version));
    }

    // Read grid counts
    int32_t nx, ny, nz;
    file.read(reinterpret_cast<char*>(&nx), sizeof(int32_t));
    file.read(reinterpret_cast<char*>(&ny), sizeof(int32_t));
    file.read(reinterpret_cast<char*>(&nz), sizeof(int32_t));

    // Read derivative count (Version 2) or skip padding (Version 1)
    uint32_t deriv_count = 0;
    if (version == 2) {
        file.read(reinterpret_cast<char*>(&deriv_count), sizeof(uint32_t));
    } else {
        file.seekg(4, std::ios::cur);  // Skip padding in Version 1
    }

    // Read grid spacing
    double dx, dy, dz;
    file.read(reinterpret_cast<char*>(&dx), sizeof(double));
    file.read(reinterpret_cast<char*>(&dy), sizeof(double));
    file.read(reinterpret_cast<char*>(&dz), sizeof(double));

    // Read data offset
    uint64_t data_offset;
    file.read(reinterpret_cast<char*>(&data_offset), sizeof(uint64_t));

    // Read metadata
    double origin_x, origin_y, origin_z;
    file.read(reinterpret_cast<char*>(&origin_x), sizeof(double));
    file.read(reinterpret_cast<char*>(&origin_y), sizeof(double));
    file.read(reinterpret_cast<char*>(&origin_z), sizeof(double));

    uint32_t grid_type_code, flags;
    file.read(reinterpret_cast<char*>(&grid_type_code), sizeof(uint32_t));
    file.read(reinterpret_cast<char*>(&flags), sizeof(uint32_t));

    double inv_power;
    file.read(reinterpret_cast<char*>(&inv_power), sizeof(double));

    // Read inv_power_mode (Version 3 only)
    uint32_t mode_value = 0;
    if (version >= 3) {
        file.read(reinterpret_cast<char*>(&mode_value), sizeof(uint32_t));
    }

    // Seek to data
    file.seekg(data_offset);

    // Read grid data
    size_t num_points = static_cast<size_t>(nx) * ny * nz;

    if (version == 2 && deriv_count > 0) {
        // Version 2: Read derivatives [deriv_count, nx, ny, nz]
        size_t total_values = deriv_count * num_points;
        m_derivatives.resize(total_values);
        file.read(reinterpret_cast<char*>(m_derivatives.data()), total_values * sizeof(double));

        // Extract function values (first derivative index = 0)
        std::vector<double> values(num_points);
        for (size_t i = 0; i < num_points; ++i) {
            values[i] = m_derivatives[i];  // f(x,y,z) is at offset i in [27, nx, ny, nz] layout
        }

        m_vals = values;
    } else {
        // Version 1: Read only function values
        std::vector<double> values(num_points);
        file.read(reinterpret_cast<char*>(values.data()), num_points * sizeof(double));
        m_vals = values;
        m_derivatives.clear();
    }

    file.close();

    // Set the grid parameters
    m_counts.clear();
    m_spacing.clear();

    addGridCounts(nx, ny, nz);
    addGridSpacing(dx, dy, dz);

    setGridOrigin(origin_x, origin_y, origin_z);
    // Restore inv_power directly without transformation (grid already has correct values)
    m_inv_power = inv_power;

    // Restore inv_power_mode
    if (version >= 3) {
        // Validate mode value
        if (mode_value > 2) {
            throw OpenMMException("GridForce: Invalid inv_power_mode value in file: " + std::to_string(mode_value));
        }
        m_invPowerMode = static_cast<InvPowerMode>(mode_value);

        // Additional validation
        if (m_invPowerMode != InvPowerMode::NONE && inv_power <= 0.0) {
            throw OpenMMException("GridForce: File has inv_power_mode enabled but invalid inv_power value");
        }
        if (m_invPowerMode == InvPowerMode::RUNTIME && !m_derivatives.empty()) {
            throw OpenMMException("GridForce: File has RUNTIME mode but also has derivatives (incompatible)");
        }
    } else {
        // V1/V2 files: default to NONE, user must set mode manually
        m_invPowerMode = InvPowerMode::NONE;
    }

    // Decode grid type
    if (grid_type_code == 1) m_gridType = "charge";
    else if (grid_type_code == 2) m_gridType = "ljr";
    else if (grid_type_code == 3) m_gridType = "lja";
    else m_gridType = "";
}

void GridForce::saveToFile(const std::string& filename) const {
    if (m_counts.size() != 3 || m_spacing.size() != 3) {
        throw OpenMMException("GridForce: Grid dimensions must be set before saving");
    }

    size_t expected_values = static_cast<size_t>(m_counts[0]) * m_counts[1] * m_counts[2];
    if (m_vals.size() != expected_values) {
        throw OpenMMException("GridForce: Number of grid values doesn't match dimensions");
    }

    bool hasDerivs = !m_derivatives.empty();
    uint32_t deriv_count = hasDerivs ? 27 : 0;

    if (hasDerivs && m_derivatives.size() != 27 * expected_values) {
        throw OpenMMException("GridForce: Number of derivative values doesn't match dimensions (expected 27 * nx * ny * nz)");
    }

    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw OpenMMException("GridForce: Cannot create file '" + filename + "'");
    }

    // Write magic number ("OMGRID" padded with nulls to 8 bytes)
    const char magic[8] = {'O', 'M', 'G', 'R', 'I', 'D', '\0', '\0'};
    file.write(magic, 8);

    // Write header (Version 3 format)
    uint32_t version = 3;
    uint32_t header_size = 128;  // Fixed size for V3
    file.write(reinterpret_cast<const char*>(&version), sizeof(uint32_t));
    file.write(reinterpret_cast<const char*>(&header_size), sizeof(uint32_t));

    // Write grid counts
    int32_t nx = m_counts[0], ny = m_counts[1], nz = m_counts[2];
    file.write(reinterpret_cast<const char*>(&nx), sizeof(int32_t));
    file.write(reinterpret_cast<const char*>(&ny), sizeof(int32_t));
    file.write(reinterpret_cast<const char*>(&nz), sizeof(int32_t));

    // Write deriv_count (0 if no derivatives)
    file.write(reinterpret_cast<const char*>(&deriv_count), sizeof(uint32_t));

    // Write grid spacing
    double dx = m_spacing[0], dy = m_spacing[1], dz = m_spacing[2];
    file.write(reinterpret_cast<const char*>(&dx), sizeof(double));
    file.write(reinterpret_cast<const char*>(&dy), sizeof(double));
    file.write(reinterpret_cast<const char*>(&dz), sizeof(double));

    // Write data offset (V3: 128-byte header)
    uint64_t data_offset = 128;
    file.write(reinterpret_cast<const char*>(&data_offset), sizeof(uint64_t));

    // Write metadata (origin)
    double origin_x = 0.0, origin_y = 0.0, origin_z = 0.0;
    if (m_gridOrigin.size() == 3) {
        origin_x = m_gridOrigin[0];
        origin_y = m_gridOrigin[1];
        origin_z = m_gridOrigin[2];
    }
    file.write(reinterpret_cast<const char*>(&origin_x), sizeof(double));
    file.write(reinterpret_cast<const char*>(&origin_y), sizeof(double));
    file.write(reinterpret_cast<const char*>(&origin_z), sizeof(double));

    // Write grid type
    uint32_t grid_type_code = 0;
    if (m_gridType == "charge") grid_type_code = 1;
    else if (m_gridType == "ljr") grid_type_code = 2;
    else if (m_gridType == "lja") grid_type_code = 3;
    file.write(reinterpret_cast<const char*>(&grid_type_code), sizeof(uint32_t));

    // Write flags
    uint32_t flags = 0;
    file.write(reinterpret_cast<const char*>(&flags), sizeof(uint32_t));

    // Write inv_power
    file.write(reinterpret_cast<const char*>(&m_inv_power), sizeof(double));

    // Write inv_power_mode (Version 3)
    uint32_t mode_value = static_cast<uint32_t>(m_invPowerMode);
    file.write(reinterpret_cast<const char*>(&mode_value), sizeof(uint32_t));

    // Pad to 128-byte header boundary
    // Current offset: 8 (magic) + 4 (version) + 4 (header_size) + 3*4 (nx,ny,nz) + 4 (deriv_count)
    //                 + 3*8 (dx,dy,dz) + 8 (data_offset) + 3*8 (origin) + 4 (grid_type) + 4 (flags)
    //                 + 8 (inv_power) + 4 (inv_power_mode) = 108 bytes
    // Need 128 - 108 = 20 bytes padding
    char reserved[20] = {0};
    file.write(reserved, 20);

    // Write grid data
    if (hasDerivs) {
        // Version 3 with derivatives: Write all derivatives [27, nx, ny, nz]
        file.write(reinterpret_cast<const char*>(m_derivatives.data()), m_derivatives.size() * sizeof(double));
    } else {
        // Version 3 without derivatives: Write only function values
        file.write(reinterpret_cast<const char*>(m_vals.data()), m_vals.size() * sizeof(double));
    }

    file.close();
}

}  // namespace GridForcePlugin
