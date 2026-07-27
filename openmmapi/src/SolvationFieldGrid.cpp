/* -------------------------------------------------------------------------- *
 *                              OpenMMGridForce                               *
 * -------------------------------------------------------------------------- */

#include "SolvationFieldGrid.h"
#include "openmm/OpenMMException.h"
#include <cstring>
#include <fstream>

using namespace GridForcePlugin;
using namespace OpenMM;
using namespace std;

constexpr char SolvationFieldGrid::MAGIC[8];
constexpr uint32_t SolvationFieldGrid::VERSION;
constexpr uint32_t SolvationFieldGrid::HEADER_SIZE;
constexpr int SolvationFieldGrid::NUM_DERIVATIVES;

SolvationFieldGrid::SolvationFieldGrid()
    : m_counts(3, 0),
      m_spacing(0.0),
      m_origin(3, 0.0),
      m_fieldType(CROSS_GB),
      m_numSlices(0),
      m_switchOn(0.0),
      m_switchOff(0.0),
      m_interpMethod(0),
      m_nyz(0),
      m_numPoints(0) {
}

SolvationFieldGrid::SolvationFieldGrid(int nx, int ny, int nz, double spacing,
                                       int numSlices, FieldType type,
                                       int interpMethod)
    : m_counts{nx, ny, nz},
      m_spacing(spacing),
      m_origin(3, 0.0),
      m_fieldType(type),
      m_numSlices(numSlices),
      m_switchOn(0.0),
      m_switchOff(0.0),
      m_interpMethod(interpMethod),
      m_nyz(ny * nz),
      m_numPoints(nx * ny * nz) {

    if (nx <= 0 || ny <= 0 || nz <= 0)
        throw OpenMMException("SolvationFieldGrid: grid dimensions must be positive");
    if (spacing <= 0.0)
        throw OpenMMException("SolvationFieldGrid: grid spacing must be positive");
    if (numSlices <= 0)
        throw OpenMMException("SolvationFieldGrid: at least one slice required");
    if (interpMethod < 0 || interpMethod > 3)
        throw OpenMMException("SolvationFieldGrid: interpolationMethod must be 0..3");

    m_data.resize(static_cast<size_t>(numSlices) * getDerivsPerPoint() * m_numPoints, 0.0f);
}

void SolvationFieldGrid::setSliceParameters(const vector<double>& params) {
    if (!params.empty() && static_cast<int>(params.size()) != m_numSlices)
        throw OpenMMException("SolvationFieldGrid: sliceParameters must have length numSlices");
    m_sliceParameters = params;
}

void SolvationFieldGrid::setSwitchRadii(double switchOn, double switchOff) {
    if (switchOn < 0.0 || switchOff < switchOn)
        throw OpenMMException("SolvationFieldGrid: require 0 <= switchOn <= switchOff");
    m_switchOn = switchOn;
    m_switchOff = switchOff;
}

void SolvationFieldGrid::setData(const vector<float>& data) {
    if (data.size() != m_data.size())
        throw OpenMMException("SolvationFieldGrid: data size mismatch");
    m_data = data;
}

void SolvationFieldGrid::setData(vector<float>&& data) {
    if (data.size() != m_data.size())
        throw OpenMMException("SolvationFieldGrid: data size mismatch");
    m_data = std::move(data);
}

shared_ptr<SolvationFieldGrid> SolvationFieldGrid::loadFromFile(const string& filename) {
    ifstream file(filename.c_str(), ios::binary);
    if (!file.is_open())
        throw OpenMMException("SolvationFieldGrid: unable to open file: " + filename);

    char magic[8];
    file.read(magic, 8);
    if (strncmp(magic, MAGIC, 7) != 0)
        throw OpenMMException("SolvationFieldGrid: invalid file format (bad magic): " + filename);

    uint32_t version;
    file.read(reinterpret_cast<char*>(&version), sizeof(uint32_t));
    if (version != VERSION)
        throw OpenMMException("SolvationFieldGrid: unsupported version: " + to_string(version));

    uint32_t headerSize;
    file.read(reinterpret_cast<char*>(&headerSize), sizeof(uint32_t));
    if (headerSize != HEADER_SIZE)
        throw OpenMMException("SolvationFieldGrid: invalid header size: " + to_string(headerSize));

    int32_t nx, ny, nz;
    file.read(reinterpret_cast<char*>(&nx), sizeof(int32_t));
    file.read(reinterpret_cast<char*>(&ny), sizeof(int32_t));
    file.read(reinterpret_cast<char*>(&nz), sizeof(int32_t));

    double spacing, originX, originY, originZ, switchOn, switchOff;
    file.read(reinterpret_cast<char*>(&spacing), sizeof(double));
    file.read(reinterpret_cast<char*>(&originX), sizeof(double));
    file.read(reinterpret_cast<char*>(&originY), sizeof(double));
    file.read(reinterpret_cast<char*>(&originZ), sizeof(double));
    file.read(reinterpret_cast<char*>(&switchOn), sizeof(double));
    file.read(reinterpret_cast<char*>(&switchOff), sizeof(double));

    uint32_t fieldType, numSlices;
    int32_t interpMethod;
    file.read(reinterpret_cast<char*>(&fieldType), sizeof(uint32_t));
    file.read(reinterpret_cast<char*>(&numSlices), sizeof(uint32_t));
    file.read(reinterpret_cast<char*>(&interpMethod), sizeof(int32_t));

    uint64_t dataOffset;
    file.read(reinterpret_cast<char*>(&dataOffset), sizeof(uint64_t));

    auto grid = make_shared<SolvationFieldGrid>(
        nx, ny, nz, spacing, static_cast<int>(numSlices),
        static_cast<FieldType>(fieldType), interpMethod);
    grid->setOrigin(originX, originY, originZ);
    grid->setSwitchRadii(switchOn, switchOff);

    file.seekg(dataOffset, ios::beg);

    vector<double> sliceParams(numSlices);
    file.read(reinterpret_cast<char*>(sliceParams.data()), numSlices * sizeof(double));
    grid->setSliceParameters(sliceParams);

    file.read(reinterpret_cast<char*>(grid->m_data.data()),
              grid->m_data.size() * sizeof(float));

    if (!file.good())
        throw OpenMMException("SolvationFieldGrid: error reading file: " + filename);

    return grid;
}

void SolvationFieldGrid::saveToFile(const string& filename) const {
    ofstream file(filename.c_str(), ios::binary);
    if (!file.is_open())
        throw OpenMMException("SolvationFieldGrid: unable to create file: " + filename);

    file.write(MAGIC, 8);
    file.write(reinterpret_cast<const char*>(&VERSION), sizeof(uint32_t));
    file.write(reinterpret_cast<const char*>(&HEADER_SIZE), sizeof(uint32_t));

    int32_t nx = m_counts[0], ny = m_counts[1], nz = m_counts[2];
    file.write(reinterpret_cast<const char*>(&nx), sizeof(int32_t));
    file.write(reinterpret_cast<const char*>(&ny), sizeof(int32_t));
    file.write(reinterpret_cast<const char*>(&nz), sizeof(int32_t));

    file.write(reinterpret_cast<const char*>(&m_spacing), sizeof(double));
    file.write(reinterpret_cast<const char*>(&m_origin[0]), sizeof(double));
    file.write(reinterpret_cast<const char*>(&m_origin[1]), sizeof(double));
    file.write(reinterpret_cast<const char*>(&m_origin[2]), sizeof(double));
    file.write(reinterpret_cast<const char*>(&m_switchOn), sizeof(double));
    file.write(reinterpret_cast<const char*>(&m_switchOff), sizeof(double));

    uint32_t fieldType = static_cast<uint32_t>(m_fieldType);
    uint32_t numSlices = static_cast<uint32_t>(m_numSlices);
    int32_t interpMethod = m_interpMethod;
    file.write(reinterpret_cast<const char*>(&fieldType), sizeof(uint32_t));
    file.write(reinterpret_cast<const char*>(&numSlices), sizeof(uint32_t));
    file.write(reinterpret_cast<const char*>(&interpMethod), sizeof(int32_t));

    uint64_t dataOffset = HEADER_SIZE;
    file.write(reinterpret_cast<const char*>(&dataOffset), sizeof(uint64_t));

    size_t currentPos = file.tellp();
    if (currentPos < HEADER_SIZE) {
        vector<char> padding(HEADER_SIZE - currentPos, 0);
        file.write(padding.data(), padding.size());
    }

    // Slice parameters, then the field data.
    vector<double> sliceParams = m_sliceParameters;
    sliceParams.resize(m_numSlices, 0.0);
    file.write(reinterpret_cast<const char*>(sliceParams.data()),
               m_numSlices * sizeof(double));
    file.write(reinterpret_cast<const char*>(m_data.data()),
               m_data.size() * sizeof(float));

    if (!file.good())
        throw OpenMMException("SolvationFieldGrid: error writing file: " + filename);
}
