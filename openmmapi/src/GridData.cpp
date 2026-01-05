/* -------------------------------------------------------------------------- *
 *                              OpenMMGridForce                               *
 * -------------------------------------------------------------------------- */

#include "GridData.h"
#include "openmm/OpenMMException.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <cstring>

using namespace GridForcePlugin;
using namespace OpenMM;
using namespace std;

GridData::GridData()
    : m_counts(3, 0), m_spacing(3, 0.0), m_origin(3, 0.0),
      m_vals(make_shared<vector<double>>()),
      m_derivatives(make_shared<vector<double>>()),
      m_nyz(0), m_invPower(0.0) {
}

GridData::GridData(int nx, int ny, int nz, double dx, double dy, double dz)
    : m_counts{nx, ny, nz}, m_spacing{dx, dy, dz}, m_origin(3, 0.0),
      m_vals(make_shared<vector<double>>()),
      m_derivatives(make_shared<vector<double>>()),
      m_nyz(ny * nz), m_invPower(0.0) {
}

void GridData::setValues(const vector<double>& vals) {
    if (!m_vals) {
        m_vals = make_shared<vector<double>>();
    }
    *m_vals = vals;
}

void GridData::setDerivatives(const vector<double>& derivs) {
    if (!m_derivatives) {
        m_derivatives = make_shared<vector<double>>();
    }
    *m_derivatives = derivs;
}

shared_ptr<GridData> GridData::loadFromFile(const string& filename) {
    auto gridData = make_shared<GridData>();

    ifstream file(filename.c_str(), ios::binary);
    if (!file.is_open()) {
        throw OpenMMException("GridData: Unable to open file: " + filename);
    }

    // Read magic number (8 bytes: "OMGRID\0\0")
    char magic[8];
    file.read(magic, 8);
    if (strncmp(magic, "OMGRID", 6) != 0) {
        throw OpenMMException("GridData: Invalid file format (bad magic number): " + filename);
    }

    // Read version and header size
    uint32_t version, header_size;
    file.read(reinterpret_cast<char*>(&version), sizeof(uint32_t));
    file.read(reinterpret_cast<char*>(&header_size), sizeof(uint32_t));

    if (version != 3) {
        throw OpenMMException("GridData: Unsupported file version: " + to_string(version));
    }
    if (header_size != 128) {
        throw OpenMMException("GridData: Invalid header size: " + to_string(header_size));
    }

    // Read grid dimensions
    int32_t nx, ny, nz;
    file.read(reinterpret_cast<char*>(&nx), sizeof(int32_t));
    file.read(reinterpret_cast<char*>(&ny), sizeof(int32_t));
    file.read(reinterpret_cast<char*>(&nz), sizeof(int32_t));
    gridData->m_counts[0] = nx;
    gridData->m_counts[1] = ny;
    gridData->m_counts[2] = nz;
    gridData->m_nyz = ny * nz;

    // Read deriv_count
    uint32_t deriv_count;
    file.read(reinterpret_cast<char*>(&deriv_count), sizeof(uint32_t));

    // Read grid spacing
    double dx, dy, dz;
    file.read(reinterpret_cast<char*>(&dx), sizeof(double));
    file.read(reinterpret_cast<char*>(&dy), sizeof(double));
    file.read(reinterpret_cast<char*>(&dz), sizeof(double));
    gridData->m_spacing[0] = dx;
    gridData->m_spacing[1] = dy;
    gridData->m_spacing[2] = dz;

    // Read data offset
    uint64_t data_offset;
    file.read(reinterpret_cast<char*>(&data_offset), sizeof(uint64_t));

    // Read origin
    double origin_x, origin_y, origin_z;
    file.read(reinterpret_cast<char*>(&origin_x), sizeof(double));
    file.read(reinterpret_cast<char*>(&origin_y), sizeof(double));
    file.read(reinterpret_cast<char*>(&origin_z), sizeof(double));
    gridData->m_origin[0] = origin_x;
    gridData->m_origin[1] = origin_y;
    gridData->m_origin[2] = origin_z;

    // Read grid_type
    uint32_t grid_type_code;
    file.read(reinterpret_cast<char*>(&grid_type_code), sizeof(uint32_t));

    // Read flags
    uint32_t flags;
    file.read(reinterpret_cast<char*>(&flags), sizeof(uint32_t));

    // Read inv_power
    double inv_power;
    file.read(reinterpret_cast<char*>(&inv_power), sizeof(double));
    gridData->m_invPower = inv_power;

    // Read inv_power_mode
    uint32_t mode_value;
    file.read(reinterpret_cast<char*>(&mode_value), sizeof(uint32_t));

    // Read reserved padding (20 bytes to reach 128-byte header)
    char reserved[20];
    file.read(reserved, 20);

    // Seek to data offset (should already be at position 128)
    file.seekg(data_offset, ios::beg);

    // Read grid values
    size_t numPoints = nx * ny * nz;
    gridData->m_vals = make_shared<vector<double>>(numPoints);
    file.read(reinterpret_cast<char*>(gridData->m_vals->data()), numPoints * sizeof(double));

    // Read scaling factors (for compatibility - empty for GridData)
    int numScalingFactors;
    file.read(reinterpret_cast<char*>(&numScalingFactors), sizeof(int));
    if (numScalingFactors > 0) {
        vector<double> scalingFactors(numScalingFactors);
        file.read(reinterpret_cast<char*>(scalingFactors.data()), numScalingFactors * sizeof(double));
        // Note: Scaling factors are per-particle, not grid property - ignored here
    }

    // Read origin again (version 3 compatibility - written after scaling factors)
    double origin[3];
    file.read(reinterpret_cast<char*>(origin), 3 * sizeof(double));
    // Origin already read from header, this is redundant but maintains compatibility

    // Check for optional derivatives block
    file.peek();
    if (!file.eof()) {
        char derivHeader[8];
        file.read(derivHeader, 8);
        if (strncmp(derivHeader, "DERIVS", 6) == 0) {
            int numDerivs = (static_cast<unsigned char>(derivHeader[6]) << 8) |
                           static_cast<unsigned char>(derivHeader[7]);
            size_t derivSize = numDerivs * numPoints;
            gridData->m_derivatives = make_shared<vector<double>>(derivSize);
            file.read(reinterpret_cast<char*>(gridData->m_derivatives->data()),
                     derivSize * sizeof(double));
        }
    }

    file.close();
    return gridData;
}

void GridData::saveToFile(const string& filename) const {
    ofstream file(filename.c_str(), ios::binary);
    if (!file.is_open()) {
        throw OpenMMException("GridData: Unable to create file: " + filename);
    }

    // Write magic number ("OMGRID" padded with nulls to 8 bytes)
    const char magic[8] = {'O', 'M', 'G', 'R', 'I', 'D', '\0', '\0'};
    file.write(magic, 8);

    // Write version and header size
    uint32_t version = 3;
    uint32_t header_size = 128;  // Fixed size for V3
    file.write(reinterpret_cast<const char*>(&version), sizeof(uint32_t));
    file.write(reinterpret_cast<const char*>(&header_size), sizeof(uint32_t));

    // Write grid counts
    int32_t nx = m_counts[0], ny = m_counts[1], nz = m_counts[2];
    file.write(reinterpret_cast<const char*>(&nx), sizeof(int32_t));
    file.write(reinterpret_cast<const char*>(&ny), sizeof(int32_t));
    file.write(reinterpret_cast<const char*>(&nz), sizeof(int32_t));

    // Write deriv_count (GridData doesn't store derivatives)
    uint32_t deriv_count = 0;
    file.write(reinterpret_cast<const char*>(&deriv_count), sizeof(uint32_t));

    // Write grid spacing
    double dx = m_spacing[0], dy = m_spacing[1], dz = m_spacing[2];
    file.write(reinterpret_cast<const char*>(&dx), sizeof(double));
    file.write(reinterpret_cast<const char*>(&dy), sizeof(double));
    file.write(reinterpret_cast<const char*>(&dz), sizeof(double));

    // Write data offset (V3: 128-byte header)
    uint64_t data_offset = 128;
    file.write(reinterpret_cast<const char*>(&data_offset), sizeof(uint64_t));

    // Write origin
    double origin_x = m_origin[0], origin_y = m_origin[1], origin_z = m_origin[2];
    file.write(reinterpret_cast<const char*>(&origin_x), sizeof(double));
    file.write(reinterpret_cast<const char*>(&origin_y), sizeof(double));
    file.write(reinterpret_cast<const char*>(&origin_z), sizeof(double));

    // Write grid type (GridData doesn't store type - write 0)
    uint32_t grid_type_code = 0;
    file.write(reinterpret_cast<const char*>(&grid_type_code), sizeof(uint32_t));

    // Write flags
    uint32_t flags = 0;
    file.write(reinterpret_cast<const char*>(&flags), sizeof(uint32_t));

    // Write inv_power (GridData doesn't store this - write 0.0)
    double inv_power = 0.0;
    file.write(reinterpret_cast<const char*>(&inv_power), sizeof(double));

    // Write inv_power_mode (GridData doesn't store this - write 0 = NONE)
    uint32_t mode_value = 0;
    file.write(reinterpret_cast<const char*>(&mode_value), sizeof(uint32_t));

    // Pad to 128-byte header boundary (20 bytes padding needed)
    char reserved[20] = {0};
    file.write(reserved, 20);

    // Write grid values
    size_t numPoints = m_counts[0] * m_counts[1] * m_counts[2];
    if (m_vals && m_vals->size() == numPoints) {
        file.write(reinterpret_cast<const char*>(m_vals->data()), numPoints * sizeof(double));
    } else {
        throw OpenMMException("GridData: Grid values size mismatch");
    }

    // Write scaling factors (empty for GridData - maintained for compatibility)
    int numScalingFactors = 0;
    file.write(reinterpret_cast<const char*>(&numScalingFactors), sizeof(int));

    // Write grid origin (version 3)
    double origin[3] = {m_origin[0], m_origin[1], m_origin[2]};
    file.write(reinterpret_cast<const char*>(origin), 3 * sizeof(double));

    // Write derivatives if present
    if (hasDerivatives()) {
        const char derivHeader[8] = {'D', 'E', 'R', 'I', 'V', 'S', 0, 27}; // 27 derivatives
        file.write(derivHeader, 8);
        file.write(reinterpret_cast<const char*>(m_derivatives->data()),
                  m_derivatives->size() * sizeof(double));
    }

    file.close();
}
