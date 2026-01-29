/* -------------------------------------------------------------------------- *
 *                              OpenMMGridForce                               *
 * -------------------------------------------------------------------------- */

#include "DesolvationGrid.h"
#include "openmm/OpenMMException.h"
#include <fstream>
#include <cstring>
#include <algorithm>

using namespace GridForcePlugin;
using namespace OpenMM;
using namespace std;

// Define static constexpr members
constexpr char DesolvationGrid::MAGIC[8];
constexpr uint32_t DesolvationGrid::VERSION;
constexpr uint32_t DesolvationGrid::MIN_SUPPORTED_VERSION;
constexpr uint32_t DesolvationGrid::HEADER_SIZE;
constexpr double DesolvationGrid::DEFAULT_PROBE_RADIUS;
constexpr double DesolvationGrid::DIELECTRIC_OFFSET;
constexpr int DesolvationGrid::NUM_DERIVATIVES;

DesolvationGrid::DesolvationGrid()
    : m_counts(3, 0),
      m_spacing(0.0),
      m_origin(3, 0.0),
      m_probeRadius(DEFAULT_PROBE_RADIUS),
      m_nyz(0),
      m_numPoints(0),
      m_hasDerivatives(false),
      m_hasReceptorDesolv(false),
      m_hasReceptorDesolvDerivs(false),
      m_receptorDesolvProbeRadius(0.0f) {
}

DesolvationGrid::DesolvationGrid(int nx, int ny, int nz, double spacing,
                                 double probeRadius,
                                 const vector<double>& rThresholds,
                                 bool hasDerivatives)
    : m_counts{nx, ny, nz},
      m_spacing(spacing),
      m_origin(3, 0.0),
      m_probeRadius(probeRadius),
      m_rThresholds(rThresholds),
      m_nyz(ny * nz),
      m_numPoints(nx * ny * nz),
      m_hasDerivatives(hasDerivatives),
      m_hasReceptorDesolv(false),
      m_hasReceptorDesolvDerivs(false),
      m_receptorDesolvProbeRadius(0.0f) {

    // Validate inputs
    if (nx <= 0 || ny <= 0 || nz <= 0) {
        throw OpenMMException("DesolvationGrid: Grid dimensions must be positive");
    }
    if (spacing <= 0) {
        throw OpenMMException("DesolvationGrid: Grid spacing must be positive");
    }
    if (rThresholds.empty()) {
        throw OpenMMException("DesolvationGrid: At least one R threshold required");
    }
    if (probeRadius <= 0) {
        throw OpenMMException("DesolvationGrid: Probe radius must be positive");
    }

    // Pre-allocate arrays
    int numBins = static_cast<int>(rThresholds.size());
    int derivsPerPoint = hasDerivatives ? NUM_DERIVATIVES : 1;
    m_hctProbe.resize(derivsPerPoint * m_numPoints, 0.0f);
    m_correctionN.resize(derivsPerPoint * numBins * m_numPoints, 0.0f);
    m_correctionA.resize(derivsPerPoint * numBins * m_numPoints, 0.0f);
    m_correctionB.resize(derivsPerPoint * numBins * m_numPoints, 0.0f);
}

int DesolvationGrid::getBinForRadius(double offsetRadius) const {
    for (size_t i = 0; i < m_rThresholds.size(); ++i) {
        if (m_rThresholds[i] >= offsetRadius) {
            return static_cast<int>(i);
        }
    }
    return static_cast<int>(m_rThresholds.size()) - 1;
}

size_t DesolvationGrid::getMemoryBytes() const {
    // Arrays are already sized correctly based on hasDerivatives flag
    size_t bytes = m_hctProbe.size() * sizeof(float) +
                   m_correctionN.size() * sizeof(float) +
                   m_correctionA.size() * sizeof(float) +
                   m_correctionB.size() * sizeof(float);

    // Add receptor desolvation if present
    if (m_hasReceptorDesolv) {
        bytes += m_receptorDesolvEnergy.size() * sizeof(float);
        if (m_hasReceptorDesolvDerivs) {
            bytes += m_receptorDesolvDerivs.size() * sizeof(float);
        }
    }

    return bytes;
}

void DesolvationGrid::setHctProbe(const vector<float>& data) {
    m_hctProbe = data;
}

void DesolvationGrid::setHctProbe(vector<float>&& data) {
    m_hctProbe = std::move(data);
}

void DesolvationGrid::setCorrectionN(const vector<float>& data) {
    m_correctionN = data;
}

void DesolvationGrid::setCorrectionN(vector<float>&& data) {
    m_correctionN = std::move(data);
}

void DesolvationGrid::setCorrectionA(const vector<float>& data) {
    m_correctionA = data;
}

void DesolvationGrid::setCorrectionA(vector<float>&& data) {
    m_correctionA = std::move(data);
}

void DesolvationGrid::setCorrectionB(const vector<float>& data) {
    m_correctionB = data;
}

void DesolvationGrid::setCorrectionB(vector<float>&& data) {
    m_correctionB = std::move(data);
}

void DesolvationGrid::setReceptorDesolvationData(const vector<float>& data, float probeRadius) {
    if (data.size() != static_cast<size_t>(m_numPoints)) {
        throw OpenMMException("DesolvationGrid: Receptor desolvation data size mismatch. Expected " +
            to_string(m_numPoints) + " but got " + to_string(data.size()));
    }
    m_receptorDesolvEnergy = data;
    m_receptorDesolvProbeRadius = probeRadius;
    m_hasReceptorDesolv = true;
}

void DesolvationGrid::setReceptorDesolvationData(vector<float>&& data, float probeRadius) {
    if (data.size() != static_cast<size_t>(m_numPoints)) {
        throw OpenMMException("DesolvationGrid: Receptor desolvation data size mismatch. Expected " +
            to_string(m_numPoints) + " but got " + to_string(data.size()));
    }
    m_receptorDesolvEnergy = std::move(data);
    m_receptorDesolvProbeRadius = probeRadius;
    m_hasReceptorDesolv = true;
}

void DesolvationGrid::setReceptorDesolvDerivatives(const vector<float>& derivs) {
    if (!m_hasReceptorDesolv) {
        throw OpenMMException("DesolvationGrid: Must set receptor desolvation data before derivatives");
    }
    size_t expectedSize = static_cast<size_t>(NUM_DERIVATIVES) * m_numPoints;
    if (derivs.size() != expectedSize) {
        throw OpenMMException("DesolvationGrid: Receptor desolvation derivatives size mismatch. Expected " +
            to_string(expectedSize) + " but got " + to_string(derivs.size()));
    }
    m_receptorDesolvDerivs = derivs;
    m_hasReceptorDesolvDerivs = true;
}

void DesolvationGrid::setReceptorDesolvDerivatives(vector<float>&& derivs) {
    if (!m_hasReceptorDesolv) {
        throw OpenMMException("DesolvationGrid: Must set receptor desolvation data before derivatives");
    }
    size_t expectedSize = static_cast<size_t>(NUM_DERIVATIVES) * m_numPoints;
    if (derivs.size() != expectedSize) {
        throw OpenMMException("DesolvationGrid: Receptor desolvation derivatives size mismatch. Expected " +
            to_string(expectedSize) + " but got " + to_string(derivs.size()));
    }
    m_receptorDesolvDerivs = std::move(derivs);
    m_hasReceptorDesolvDerivs = true;
}

shared_ptr<DesolvationGrid> DesolvationGrid::loadFromFile(const string& filename) {
    ifstream file(filename.c_str(), ios::binary);
    if (!file.is_open()) {
        throw OpenMMException("DesolvationGrid: Unable to open file: " + filename);
    }

    // Read and verify magic number
    char magic[8];
    file.read(magic, 8);
    if (strncmp(magic, MAGIC, 6) != 0) {
        throw OpenMMException("DesolvationGrid: Invalid file format (bad magic): " + filename);
    }

    // Read version (support v2 and v3)
    uint32_t version;
    file.read(reinterpret_cast<char*>(&version), sizeof(uint32_t));
    if (version < MIN_SUPPORTED_VERSION || version > VERSION) {
        throw OpenMMException("DesolvationGrid: Unsupported version: " + to_string(version) +
            " (supported: " + to_string(MIN_SUPPORTED_VERSION) + "-" + to_string(VERSION) + ")");
    }

    // Read header size
    uint32_t headerSize;
    file.read(reinterpret_cast<char*>(&headerSize), sizeof(uint32_t));
    if (headerSize != HEADER_SIZE) {
        throw OpenMMException("DesolvationGrid: Invalid header size: " + to_string(headerSize));
    }

    // Read grid dimensions
    int32_t nx, ny, nz;
    file.read(reinterpret_cast<char*>(&nx), sizeof(int32_t));
    file.read(reinterpret_cast<char*>(&ny), sizeof(int32_t));
    file.read(reinterpret_cast<char*>(&nz), sizeof(int32_t));

    // Read spacing
    double spacing;
    file.read(reinterpret_cast<char*>(&spacing), sizeof(double));

    // Read origin
    double originX, originY, originZ;
    file.read(reinterpret_cast<char*>(&originX), sizeof(double));
    file.read(reinterpret_cast<char*>(&originY), sizeof(double));
    file.read(reinterpret_cast<char*>(&originZ), sizeof(double));

    // Read probe radius
    double probeRadius;
    file.read(reinterpret_cast<char*>(&probeRadius), sizeof(double));

    // Read number of bins
    uint32_t numBins;
    file.read(reinterpret_cast<char*>(&numBins), sizeof(uint32_t));

    // Read hasDerivatives flag
    uint32_t hasDerivativesFlag;
    file.read(reinterpret_cast<char*>(&hasDerivativesFlag), sizeof(uint32_t));
    bool hasDerivatives = (hasDerivativesFlag != 0);

    // Read R thresholds
    vector<double> rThresholds(numBins);
    file.read(reinterpret_cast<char*>(rThresholds.data()), numBins * sizeof(double));

    // Read data offset
    uint64_t dataOffset;
    file.read(reinterpret_cast<char*>(&dataOffset), sizeof(uint64_t));

    // Skip padding to reach data offset
    file.seekg(dataOffset, ios::beg);

    // Create grid object
    auto grid = make_shared<DesolvationGrid>(nx, ny, nz, spacing, probeRadius, rThresholds, hasDerivatives);
    grid->setOrigin(originX, originY, originZ);

    int numPoints = nx * ny * nz;
    int derivsPerPoint = hasDerivatives ? NUM_DERIVATIVES : 1;

    // Read HCT probe array
    file.read(reinterpret_cast<char*>(grid->m_hctProbe.data()), derivsPerPoint * numPoints * sizeof(float));

    // Read correction arrays
    size_t corrSize = derivsPerPoint * numBins * numPoints;
    file.read(reinterpret_cast<char*>(grid->m_correctionN.data()), corrSize * sizeof(float));
    file.read(reinterpret_cast<char*>(grid->m_correctionA.data()), corrSize * sizeof(float));
    file.read(reinterpret_cast<char*>(grid->m_correctionB.data()), corrSize * sizeof(float));

    if (!file.good()) {
        throw OpenMMException("DesolvationGrid: Error reading file: " + filename);
    }

    // Read receptor desolvation data (v3+)
    if (version >= 3) {
        uint8_t hasReceptorDesolv;
        file.read(reinterpret_cast<char*>(&hasReceptorDesolv), sizeof(uint8_t));

        if (hasReceptorDesolv) {
            // Read probe radius for receptor desolvation
            float recDesolvProbeRadius;
            file.read(reinterpret_cast<char*>(&recDesolvProbeRadius), sizeof(float));

            // Read has derivatives flag
            uint8_t hasRecDesolvDerivs;
            file.read(reinterpret_cast<char*>(&hasRecDesolvDerivs), sizeof(uint8_t));

            // Read energy values
            grid->m_receptorDesolvEnergy.resize(numPoints);
            file.read(reinterpret_cast<char*>(grid->m_receptorDesolvEnergy.data()),
                      numPoints * sizeof(float));

            grid->m_receptorDesolvProbeRadius = recDesolvProbeRadius;
            grid->m_hasReceptorDesolv = true;

            // Read derivatives if present
            if (hasRecDesolvDerivs) {
                size_t derivSize = static_cast<size_t>(NUM_DERIVATIVES) * numPoints;
                grid->m_receptorDesolvDerivs.resize(derivSize);
                file.read(reinterpret_cast<char*>(grid->m_receptorDesolvDerivs.data()),
                          derivSize * sizeof(float));
                grid->m_hasReceptorDesolvDerivs = true;
            }

            if (!file.good()) {
                throw OpenMMException("DesolvationGrid: Error reading receptor desolvation data: " + filename);
            }
        }
    }

    file.close();
    return grid;
}

void DesolvationGrid::saveToFile(const string& filename) const {
    ofstream file(filename.c_str(), ios::binary);
    if (!file.is_open()) {
        throw OpenMMException("DesolvationGrid: Unable to create file: " + filename);
    }

    // Write magic number
    file.write(MAGIC, 8);

    // Write version
    file.write(reinterpret_cast<const char*>(&VERSION), sizeof(uint32_t));

    // Write header size
    file.write(reinterpret_cast<const char*>(&HEADER_SIZE), sizeof(uint32_t));

    // Write grid dimensions
    int32_t nx = m_counts[0], ny = m_counts[1], nz = m_counts[2];
    file.write(reinterpret_cast<const char*>(&nx), sizeof(int32_t));
    file.write(reinterpret_cast<const char*>(&ny), sizeof(int32_t));
    file.write(reinterpret_cast<const char*>(&nz), sizeof(int32_t));

    // Write spacing
    file.write(reinterpret_cast<const char*>(&m_spacing), sizeof(double));

    // Write origin
    file.write(reinterpret_cast<const char*>(&m_origin[0]), sizeof(double));
    file.write(reinterpret_cast<const char*>(&m_origin[1]), sizeof(double));
    file.write(reinterpret_cast<const char*>(&m_origin[2]), sizeof(double));

    // Write probe radius
    file.write(reinterpret_cast<const char*>(&m_probeRadius), sizeof(double));

    // Write number of bins
    uint32_t numBins = static_cast<uint32_t>(m_rThresholds.size());
    file.write(reinterpret_cast<const char*>(&numBins), sizeof(uint32_t));

    // Write hasDerivatives flag
    uint32_t hasDerivativesFlag = m_hasDerivatives ? 1 : 0;
    file.write(reinterpret_cast<const char*>(&hasDerivativesFlag), sizeof(uint32_t));

    // Write R thresholds (up to 8 bins supported in header)
    // For more bins, they're stored after header
    size_t thresholdBytes = numBins * sizeof(double);
    if (numBins <= 8) {
        file.write(reinterpret_cast<const char*>(m_rThresholds.data()), thresholdBytes);
        // Pad remaining threshold space
        char padding[64] = {0};
        size_t padBytes = 8 * sizeof(double) - thresholdBytes;
        file.write(padding, padBytes);
    } else {
        // Write first 8 thresholds in header
        file.write(reinterpret_cast<const char*>(m_rThresholds.data()), 8 * sizeof(double));
    }

    // Write data offset
    uint64_t dataOffset = HEADER_SIZE;
    file.write(reinterpret_cast<const char*>(&dataOffset), sizeof(uint64_t));

    // Calculate current position and add padding to reach header size
    size_t currentPos = file.tellp();
    if (currentPos < HEADER_SIZE) {
        size_t padBytes = HEADER_SIZE - currentPos;
        vector<char> padding(padBytes, 0);
        file.write(padding.data(), padBytes);
    }

    // Write grid data
    int numPoints = m_counts[0] * m_counts[1] * m_counts[2];
    int derivsPerPoint = m_hasDerivatives ? NUM_DERIVATIVES : 1;

    // Write HCT probe array
    file.write(reinterpret_cast<const char*>(m_hctProbe.data()), derivsPerPoint * numPoints * sizeof(float));

    // Write correction arrays
    size_t corrSize = derivsPerPoint * numBins * numPoints;
    file.write(reinterpret_cast<const char*>(m_correctionN.data()), corrSize * sizeof(float));
    file.write(reinterpret_cast<const char*>(m_correctionA.data()), corrSize * sizeof(float));
    file.write(reinterpret_cast<const char*>(m_correctionB.data()), corrSize * sizeof(float));

    // Write remaining thresholds if more than 8
    if (numBins > 8) {
        file.write(reinterpret_cast<const char*>(m_rThresholds.data() + 8),
                   (numBins - 8) * sizeof(double));
    }

    // Write receptor desolvation data (v3 feature)
    uint8_t hasReceptorDesolv = m_hasReceptorDesolv ? 1 : 0;
    file.write(reinterpret_cast<const char*>(&hasReceptorDesolv), sizeof(uint8_t));

    if (m_hasReceptorDesolv) {
        // Write probe radius
        file.write(reinterpret_cast<const char*>(&m_receptorDesolvProbeRadius), sizeof(float));

        // Write has derivatives flag
        uint8_t hasRecDesolvDerivs = m_hasReceptorDesolvDerivs ? 1 : 0;
        file.write(reinterpret_cast<const char*>(&hasRecDesolvDerivs), sizeof(uint8_t));

        // Write energy values
        file.write(reinterpret_cast<const char*>(m_receptorDesolvEnergy.data()),
                   numPoints * sizeof(float));

        // Write derivatives if present
        if (m_hasReceptorDesolvDerivs) {
            file.write(reinterpret_cast<const char*>(m_receptorDesolvDerivs.data()),
                       m_receptorDesolvDerivs.size() * sizeof(float));
        }
    }

    if (!file.good()) {
        throw OpenMMException("DesolvationGrid: Error writing file: " + filename);
    }

    file.close();
}
