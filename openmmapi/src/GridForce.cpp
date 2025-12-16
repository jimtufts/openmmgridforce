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

using namespace OpenMM;
using namespace std;

namespace GridForcePlugin {

GridForce::GridForce() : m_inv_power(0.0), m_autoCalculateScalingFactors(false), m_scalingProperty(""),
                         m_autoGenerateGrid(false), m_gridType(""), m_gridOrigin({0.0, 0.0, 0.0}) {
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

void GridForce::addScalingFactor(double val) {
    m_scaling_factors.push_back(val);
}

void GridForce::setScalingFactor(int index, double val) {
    m_scaling_factors[index] = val;
}

void GridForce::setInvPower(double inv_power) {
    m_inv_power = inv_power;
}

double GridForce::getInvPower() const {
    return m_inv_power;
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

    if (version != 1) {
        file.close();
        throw OpenMMException("GridForce: Unsupported file version " + std::to_string(version));
    }

    // Read grid counts
    int32_t nx, ny, nz;
    file.read(reinterpret_cast<char*>(&nx), sizeof(int32_t));
    file.read(reinterpret_cast<char*>(&ny), sizeof(int32_t));
    file.read(reinterpret_cast<char*>(&nz), sizeof(int32_t));

    // Skip padding
    file.seekg(4, std::ios::cur);

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

    // Seek to data
    file.seekg(data_offset);

    // Read grid values
    size_t num_values = static_cast<size_t>(nx) * ny * nz;
    std::vector<double> values(num_values);
    file.read(reinterpret_cast<char*>(values.data()), num_values * sizeof(double));

    file.close();

    // Set the grid parameters
    m_counts.clear();
    m_spacing.clear();
    m_vals.clear();

    addGridCounts(nx, ny, nz);
    addGridSpacing(dx, dy, dz);
    for (double val : values) {
        addGridValue(val);
    }

    setGridOrigin(origin_x, origin_y, origin_z);
    setInvPower(inv_power);

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

    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw OpenMMException("GridForce: Cannot create file '" + filename + "'");
    }

    // Write magic number ("OMGRID" padded with nulls to 8 bytes)
    const char magic[8] = {'O', 'M', 'G', 'R', 'I', 'D', '\0', '\0'};
    file.write(magic, 8);

    // Write header
    uint32_t version = 1;
    uint32_t header_size = 64;
    file.write(reinterpret_cast<const char*>(&version), sizeof(uint32_t));
    file.write(reinterpret_cast<const char*>(&header_size), sizeof(uint32_t));

    // Write grid counts
    int32_t nx = m_counts[0], ny = m_counts[1], nz = m_counts[2];
    file.write(reinterpret_cast<const char*>(&nx), sizeof(int32_t));
    file.write(reinterpret_cast<const char*>(&ny), sizeof(int32_t));
    file.write(reinterpret_cast<const char*>(&nz), sizeof(int32_t));

    // Write padding
    float padding = 0.0f;
    file.write(reinterpret_cast<const char*>(&padding), sizeof(float));

    // Write grid spacing
    double dx = m_spacing[0], dy = m_spacing[1], dz = m_spacing[2];
    file.write(reinterpret_cast<const char*>(&dx), sizeof(double));
    file.write(reinterpret_cast<const char*>(&dy), sizeof(double));
    file.write(reinterpret_cast<const char*>(&dz), sizeof(double));

    // Write data offset (224 bytes: 64 header + 160 metadata)
    uint64_t data_offset = 224;
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

    // Write description (120 bytes of zeros)
    char description[120] = {0};
    file.write(description, 120);

    // Write grid values
    file.write(reinterpret_cast<const char*>(m_vals.data()), m_vals.size() * sizeof(double));

    file.close();
}

}  // namespace GridForcePlugin
