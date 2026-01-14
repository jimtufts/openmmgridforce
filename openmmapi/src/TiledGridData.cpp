/* -------------------------------------------------------------------------- *
 *                              OpenMMGridForce                               *
 * -------------------------------------------------------------------------- */

#include "TiledGridData.h"
#include "openmm/OpenMMException.h"
#include <cstring>
#include <iostream>
#include <algorithm>

using namespace GridForcePlugin;
using namespace OpenMM;
using namespace std;

TiledGridData::TiledGridData()
    : m_counts(3, 0), m_spacing(3, 0.0), m_origin(3, 0.0),
      m_tileSize(DEFAULT_TILE_SIZE),
      m_numTilesX(0), m_numTilesY(0), m_numTilesZ(0),
      m_hasDerivatives(false),
      m_invPower(0.0), m_invPowerMode(InvPowerMode::NONE),
      m_isWriting(false), m_tileIndexOffset(0) {
}

TiledGridData::TiledGridData(int nx, int ny, int nz,
                             double dx, double dy, double dz,
                             int tileSize)
    : m_counts{nx, ny, nz}, m_spacing{dx, dy, dz}, m_origin(3, 0.0),
      m_tileSize(tileSize),
      m_hasDerivatives(false),
      m_invPower(0.0), m_invPowerMode(InvPowerMode::NONE),
      m_isWriting(false), m_tileIndexOffset(0) {
    computeTileCounts();
}

TiledGridData::~TiledGridData() {
    if (m_file.is_open()) {
        close();
    }
}

void TiledGridData::setOrigin(double ox, double oy, double oz) {
    m_origin = {ox, oy, oz};
}

void TiledGridData::computeTileCounts() {
    m_numTilesX = (m_counts[0] + m_tileSize - 1) / m_tileSize;
    m_numTilesY = (m_counts[1] + m_tileSize - 1) / m_tileSize;
    m_numTilesZ = (m_counts[2] + m_tileSize - 1) / m_tileSize;
}

vector<int> TiledGridData::getTileGridRange(int tileX, int tileY, int tileZ) const {
    int startX = tileX * m_tileSize;
    int startY = tileY * m_tileSize;
    int startZ = tileZ * m_tileSize;
    int endX = min(startX + m_tileSize, m_counts[0]);
    int endY = min(startY + m_tileSize, m_counts[1]);
    int endZ = min(startZ + m_tileSize, m_counts[2]);
    return {startX, startY, startZ, endX, endY, endZ};
}

void TiledGridData::getTileActualSize(int tileX, int tileY, int tileZ,
                                      int& sizeX, int& sizeY, int& sizeZ) const {
    auto range = getTileGridRange(tileX, tileY, tileZ);
    sizeX = range[3] - range[0];
    sizeY = range[4] - range[1];
    sizeZ = range[5] - range[2];
}

int TiledGridData::getTileLinearIndex(int tileX, int tileY, int tileZ) const {
    return tileX * m_numTilesY * m_numTilesZ + tileY * m_numTilesZ + tileZ;
}

// ========== Writing Implementation ==========

void TiledGridData::beginWriting(const string& filename, bool hasDerivatives) {
    if (m_file.is_open()) {
        throw OpenMMException("TiledGridData: File already open");
    }

    m_filename = filename;
    m_hasDerivatives = hasDerivatives;
    m_isWriting = true;

    // Open file for writing
    m_file.open(filename, ios::binary | ios::out | ios::trunc);
    if (!m_file.is_open()) {
        throw OpenMMException("TiledGridData: Unable to create file: " + filename);
    }

    // Initialize tile index (will be filled as tiles are written)
    int totalTiles = getTotalNumTiles();
    m_tileIndex.resize(totalTiles);
    for (int i = 0; i < totalTiles; i++) {
        m_tileIndex[i].fileOffset = -1;  // Mark as not yet written
        m_tileIndex[i].dataSize = 0;
    }

    // Write header (metadata will be finalized in finishWriting)
    writeHeader();
}

void TiledGridData::writeHeader() {
    // Seek to beginning
    m_file.seekp(0);

    // Magic number (8 bytes)
    m_file.write(TILED_GRID_MAGIC, 8);

    // Version (4 bytes)
    uint32_t version = TILED_GRID_VERSION;
    m_file.write(reinterpret_cast<char*>(&version), 4);

    // Header size (4 bytes)
    uint32_t headerSize = TILED_GRID_HEADER_SIZE;
    m_file.write(reinterpret_cast<char*>(&headerSize), 4);

    // Flags (4 bytes)
    uint32_t flags = 0;
    if (m_hasDerivatives) flags |= TILED_FLAG_HAS_DERIVATIVES;
    m_file.write(reinterpret_cast<char*>(&flags), 4);

    // Tile size (4 bytes)
    uint32_t tileSize = m_tileSize;
    m_file.write(reinterpret_cast<char*>(&tileSize), 4);

    // Reserved (40 bytes to reach 64-byte header)
    char reserved[40] = {0};
    m_file.write(reserved, 40);

    // Grid metadata
    // Counts (3 x uint32)
    for (int i = 0; i < 3; i++) {
        uint32_t count = m_counts[i];
        m_file.write(reinterpret_cast<char*>(&count), 4);
    }

    // Spacing (3 x double)
    for (int i = 0; i < 3; i++) {
        m_file.write(reinterpret_cast<char*>(&m_spacing[i]), 8);
    }

    // Origin (3 x double)
    for (int i = 0; i < 3; i++) {
        m_file.write(reinterpret_cast<char*>(&m_origin[i]), 8);
    }

    // invPower (double)
    m_file.write(reinterpret_cast<char*>(&m_invPower), 8);

    // invPowerMode (uint32)
    uint32_t modeVal = static_cast<uint32_t>(m_invPowerMode);
    m_file.write(reinterpret_cast<char*>(&modeVal), 4);

    // Number of tiles (uint32)
    uint32_t numTiles = getTotalNumTiles();
    m_file.write(reinterpret_cast<char*>(&numTiles), 4);

    // Tile index offset placeholder (int64) - will be updated in finishWriting
    m_tileIndexOffset = 0;
    m_file.write(reinterpret_cast<char*>(&m_tileIndexOffset), 8);
}

void TiledGridData::writeTile(int tileX, int tileY, int tileZ,
                              const vector<float>& values,
                              const vector<float>& derivatives) {
    if (!m_isWriting) {
        throw OpenMMException("TiledGridData: File not open for writing");
    }

    if (tileX < 0 || tileX >= m_numTilesX ||
        tileY < 0 || tileY >= m_numTilesY ||
        tileZ < 0 || tileZ >= m_numTilesZ) {
        throw OpenMMException("TiledGridData: Invalid tile coordinates");
    }

    // Get expected tile size
    int sizeX, sizeY, sizeZ;
    getTileActualSize(tileX, tileY, tileZ, sizeX, sizeY, sizeZ);
    int numPoints = sizeX * sizeY * sizeZ;

    if (values.size() != (size_t)numPoints) {
        throw OpenMMException("TiledGridData: Values size mismatch. Expected " +
                              to_string(numPoints) + ", got " + to_string(values.size()));
    }

    if (m_hasDerivatives) {
        size_t expectedDerivs = 27 * numPoints;
        if (derivatives.size() != expectedDerivs) {
            throw OpenMMException("TiledGridData: Derivatives size mismatch. Expected " +
                                  to_string(expectedDerivs) + ", got " + to_string(derivatives.size()));
        }
    }

    // Record file offset for this tile
    int tileIdx = getTileLinearIndex(tileX, tileY, tileZ);
    m_tileIndex[tileIdx].tileX = tileX;
    m_tileIndex[tileIdx].tileY = tileY;
    m_tileIndex[tileIdx].tileZ = tileZ;
    m_tileIndex[tileIdx].fileOffset = m_file.tellp();

    // Write tile header (tile dimensions for boundary tiles)
    uint16_t dims[3] = {(uint16_t)sizeX, (uint16_t)sizeY, (uint16_t)sizeZ};
    m_file.write(reinterpret_cast<char*>(dims), 6);

    // Write values
    m_file.write(reinterpret_cast<const char*>(values.data()), values.size() * sizeof(float));

    // Write derivatives if present
    if (m_hasDerivatives && !derivatives.empty()) {
        m_file.write(reinterpret_cast<const char*>(derivatives.data()), derivatives.size() * sizeof(float));
    }

    // Record data size
    m_tileIndex[tileIdx].dataSize = (int64_t)m_file.tellp() - m_tileIndex[tileIdx].fileOffset;
}

void TiledGridData::writeTileIndex() {
    // Record where tile index starts
    m_tileIndexOffset = m_file.tellp();

    // Write each tile index entry
    for (const auto& entry : m_tileIndex) {
        m_file.write(reinterpret_cast<const char*>(&entry.tileX), 4);
        m_file.write(reinterpret_cast<const char*>(&entry.tileY), 4);
        m_file.write(reinterpret_cast<const char*>(&entry.tileZ), 4);
        m_file.write(reinterpret_cast<const char*>(&entry.fileOffset), 8);
        m_file.write(reinterpret_cast<const char*>(&entry.dataSize), 8);
    }
}

void TiledGridData::finishWriting() {
    if (!m_isWriting) {
        throw OpenMMException("TiledGridData: Not in writing mode");
    }

    // Write tile index at end of file
    writeTileIndex();

    // Go back and update the tile index offset in header
    // Header offset for tileIndexOffset is: 64 (header) + 12 (counts) + 24 (spacing) + 24 (origin) + 8 (invPower) + 4 (mode) + 4 (numTiles) = 140
    m_file.seekp(140);
    m_file.write(reinterpret_cast<char*>(&m_tileIndexOffset), 8);

    m_file.close();
    m_isWriting = false;
}

// ========== Reading Implementation ==========

void TiledGridData::openForReading(const string& filename) {
    if (m_file.is_open()) {
        throw OpenMMException("TiledGridData: File already open");
    }

    m_filename = filename;
    m_isWriting = false;

    m_file.open(filename, ios::binary | ios::in);
    if (!m_file.is_open()) {
        throw OpenMMException("TiledGridData: Unable to open file: " + filename);
    }

    readHeader();
    readTileIndex();
}

void TiledGridData::readHeader() {
    m_file.seekg(0);

    // Read and verify magic
    char magic[8];
    m_file.read(magic, 8);
    if (strncmp(magic, TILED_GRID_MAGIC, 7) != 0) {
        throw OpenMMException("TiledGridData: Invalid file format (not a tiled grid file)");
    }

    // Version
    uint32_t version;
    m_file.read(reinterpret_cast<char*>(&version), 4);
    if (version != TILED_GRID_VERSION) {
        throw OpenMMException("TiledGridData: Unsupported file version: " + to_string(version));
    }

    // Header size
    uint32_t headerSize;
    m_file.read(reinterpret_cast<char*>(&headerSize), 4);

    // Flags
    uint32_t flags;
    m_file.read(reinterpret_cast<char*>(&flags), 4);
    m_hasDerivatives = (flags & TILED_FLAG_HAS_DERIVATIVES) != 0;

    // Tile size
    uint32_t tileSize;
    m_file.read(reinterpret_cast<char*>(&tileSize), 4);
    m_tileSize = tileSize;

    // Skip reserved bytes
    m_file.seekg(64);  // Jump to end of 64-byte header

    // Grid metadata
    m_counts.resize(3);
    for (int i = 0; i < 3; i++) {
        uint32_t count;
        m_file.read(reinterpret_cast<char*>(&count), 4);
        m_counts[i] = count;
    }

    m_spacing.resize(3);
    for (int i = 0; i < 3; i++) {
        m_file.read(reinterpret_cast<char*>(&m_spacing[i]), 8);
    }

    m_origin.resize(3);
    for (int i = 0; i < 3; i++) {
        m_file.read(reinterpret_cast<char*>(&m_origin[i]), 8);
    }

    m_file.read(reinterpret_cast<char*>(&m_invPower), 8);

    uint32_t modeVal;
    m_file.read(reinterpret_cast<char*>(&modeVal), 4);
    m_invPowerMode = static_cast<InvPowerMode>(modeVal);

    uint32_t numTiles;
    m_file.read(reinterpret_cast<char*>(&numTiles), 4);

    m_file.read(reinterpret_cast<char*>(&m_tileIndexOffset), 8);

    // Compute tile counts
    computeTileCounts();

    // Verify numTiles matches
    if ((int)numTiles != getTotalNumTiles()) {
        throw OpenMMException("TiledGridData: Tile count mismatch in file");
    }
}

void TiledGridData::readTileIndex() {
    m_file.seekg(m_tileIndexOffset);

    int totalTiles = getTotalNumTiles();
    m_tileIndex.resize(totalTiles);

    for (int i = 0; i < totalTiles; i++) {
        m_file.read(reinterpret_cast<char*>(&m_tileIndex[i].tileX), 4);
        m_file.read(reinterpret_cast<char*>(&m_tileIndex[i].tileY), 4);
        m_file.read(reinterpret_cast<char*>(&m_tileIndex[i].tileZ), 4);
        m_file.read(reinterpret_cast<char*>(&m_tileIndex[i].fileOffset), 8);
        m_file.read(reinterpret_cast<char*>(&m_tileIndex[i].dataSize), 8);
    }
}

void TiledGridData::readTile(int tileX, int tileY, int tileZ,
                             vector<float>& values,
                             vector<float>& derivatives) const {
    if (m_isWriting) {
        throw OpenMMException("TiledGridData: File is open for writing, not reading");
    }

    if (tileX < 0 || tileX >= m_numTilesX ||
        tileY < 0 || tileY >= m_numTilesY ||
        tileZ < 0 || tileZ >= m_numTilesZ) {
        throw OpenMMException("TiledGridData: Invalid tile coordinates");
    }

    int tileIdx = getTileLinearIndex(tileX, tileY, tileZ);
    const auto& entry = m_tileIndex[tileIdx];

    if (entry.fileOffset < 0) {
        throw OpenMMException("TiledGridData: Tile not present in file");
    }

    // Seek to tile
    m_file.seekg(entry.fileOffset);

    // Read tile dimensions
    uint16_t dims[3];
    m_file.read(reinterpret_cast<char*>(dims), 6);
    int sizeX = dims[0], sizeY = dims[1], sizeZ = dims[2];
    int numPoints = sizeX * sizeY * sizeZ;

    // Read values
    values.resize(numPoints);
    m_file.read(reinterpret_cast<char*>(values.data()), numPoints * sizeof(float));

    // Read derivatives if present
    if (m_hasDerivatives) {
        derivatives.resize(27 * numPoints);
        m_file.read(reinterpret_cast<char*>(derivatives.data()), 27 * numPoints * sizeof(float));
    } else {
        derivatives.clear();
    }
}

bool TiledGridData::hasTile(int tileX, int tileY, int tileZ) const {
    if (tileX < 0 || tileX >= m_numTilesX ||
        tileY < 0 || tileY >= m_numTilesY ||
        tileZ < 0 || tileZ >= m_numTilesZ) {
        return false;
    }
    int tileIdx = getTileLinearIndex(tileX, tileY, tileZ);
    return m_tileIndex[tileIdx].fileOffset >= 0;
}

void TiledGridData::close() {
    if (m_file.is_open()) {
        if (m_isWriting) {
            finishWriting();
        } else {
            m_file.close();
        }
    }
    m_tileIndex.clear();
}

// ========== Static Utilities ==========

bool TiledGridData::isTiledFormat(const string& filename) {
    ifstream file(filename, ios::binary);
    if (!file.is_open()) {
        return false;
    }

    char magic[8];
    file.read(magic, 8);
    file.close();

    return strncmp(magic, TILED_GRID_MAGIC, 7) == 0;
}
