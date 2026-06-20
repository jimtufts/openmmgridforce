/* -------------------------------------------------------------------------- *
 *                              OpenMMGridForce                               *
 * -------------------------------------------------------------------------- */

#include "TiledGridData.h"
#include "BSplinePrefilter.h"
#include "openmm/OpenMMException.h"
#include <cmath>
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
    if (m_doubleDerivatives) flags |= TILED_FLAG_DOUBLE_DERIVATIVES;
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
                              const vector<char>& derivBytes) {
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
        size_t expectedBytes = (size_t)27 * numPoints * derivativeElementSize();
        if (derivBytes.size() != expectedBytes) {
            throw OpenMMException("TiledGridData: Derivatives size mismatch. Expected " +
                                  to_string(expectedBytes) + " bytes, got " + to_string(derivBytes.size()));
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

    // Write derivative bytes verbatim (already in the file's element size)
    if (m_hasDerivatives && !derivBytes.empty()) {
        m_file.write(derivBytes.data(), derivBytes.size());
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
    m_doubleDerivatives = (flags & TILED_FLAG_DOUBLE_DERIVATIVES) != 0;

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
                             vector<char>& derivBytes) const {
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

    // Read derivative bytes verbatim in the file's element size
    if (m_hasDerivatives) {
        derivBytes.resize((size_t)27 * numPoints * derivativeElementSize());
        m_file.read(derivBytes.data(), derivBytes.size());
    } else {
        derivBytes.clear();
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

// ========== Arcsinh transform ==========

void TiledGridData::applyArcsinhTransform(double scale) {
    if (scale <= 0.0) {
        throw OpenMMException("TiledGridData::applyArcsinhTransform: scale must be > 0");
    }
    if (m_filename.empty() || m_tileIndex.empty()) {
        throw OpenMMException("TiledGridData: Must call openForReading() before applyArcsinhTransform()");
    }
    if (m_file.is_open()) {
        m_file.close();
    }

    // Reopen file for read-write
    m_file.open(m_filename, ios::binary | ios::in | ios::out);
    if (!m_file.is_open()) {
        throw OpenMMException("TiledGridData: Unable to reopen file for arcsinh transform: " + m_filename);
    }

    float scaleF = (float)scale;
    int totalTiles = m_numTilesX * m_numTilesY * m_numTilesZ;

    for (int txIdx = 0; txIdx < m_numTilesX; txIdx++) {
        for (int tyIdx = 0; tyIdx < m_numTilesY; tyIdx++) {
            for (int tzIdx = 0; tzIdx < m_numTilesZ; tzIdx++) {
                int tileSizeX, tileSizeY, tileSizeZ;
                getTileActualSize(txIdx, tyIdx, tzIdx, tileSizeX, tileSizeY, tileSizeZ);

                vector<float> tileVals; vector<char> tileDerivsUnused;
                readTile(txIdx, tyIdx, tzIdx, tileVals, tileDerivsUnused);

                // Apply arcsinh(value/scale) to each grid value
                for (size_t i = 0; i < tileVals.size(); i++) {
                    tileVals[i] = std::asinh(tileVals[i] / scaleF);
                }

                // Write back values only (skip 6-byte tile dims header)
                int tileIdx = getTileLinearIndex(txIdx, tyIdx, tzIdx);
                int64_t offset = m_tileIndex[tileIdx].fileOffset + 6;
                m_file.seekp(offset);
                m_file.write(reinterpret_cast<const char*>(tileVals.data()),
                             tileVals.size() * sizeof(float));
            }
        }
    }

    m_file.close();

    // Reopen in read-only mode
    m_file.open(m_filename, ios::binary | ios::in);
}

// ========== Stored inv_power transform ==========

void TiledGridData::applyInvPowerTransform(float invPower) {
    if (invPower == 0.0f) {
        throw OpenMMException("TiledGridData::applyInvPowerTransform: invPower must be != 0");
    }
    if (m_filename.empty() || m_tileIndex.empty()) {
        throw OpenMMException("TiledGridData: Must call openForReading() before applyInvPowerTransform()");
    }
    if (m_file.is_open()) {
        m_file.close();
    }

    // Reopen file for read-write
    m_file.open(m_filename, ios::binary | ios::in | ios::out);
    if (!m_file.is_open()) {
        throw OpenMMException("TiledGridData: Unable to reopen file for inv_power transform: " + m_filename);
    }

    float p = 1.0f / invPower;

    for (int txIdx = 0; txIdx < m_numTilesX; txIdx++) {
        for (int tyIdx = 0; tyIdx < m_numTilesY; tyIdx++) {
            for (int tzIdx = 0; tzIdx < m_numTilesZ; tzIdx++) {
                int tileSizeX, tileSizeY, tileSizeZ;
                getTileActualSize(txIdx, tyIdx, tzIdx, tileSizeX, tileSizeY, tileSizeZ);

                vector<float> tileVals; vector<char> tileDerivsUnused;
                readTile(txIdx, tyIdx, tzIdx, tileVals, tileDerivsUnused);

                // Apply sign(V) * |V|^(1/n) to each grid value
                for (size_t i = 0; i < tileVals.size(); i++) {
                    float v = tileVals[i];
                    if (v > 0.0f) {
                        float t = std::pow(v, p);
                        tileVals[i] = std::isfinite(t) ? t : 0.0f;
                    } else if (v < 0.0f) {
                        float t = std::pow(-v, p);
                        tileVals[i] = std::isfinite(t) ? -t : 0.0f;
                    }
                    // v == 0.0f stays 0.0f
                }

                // Write back values only (skip 6-byte tile dims header)
                int tileIdx = getTileLinearIndex(txIdx, tyIdx, tzIdx);
                int64_t offset = m_tileIndex[tileIdx].fileOffset + 6;
                m_file.seekp(offset);
                m_file.write(reinterpret_cast<const char*>(tileVals.data()),
                             tileVals.size() * sizeof(float));
            }
        }
    }

    m_file.close();

    // Reopen in read-only mode
    m_file.open(m_filename, ios::binary | ios::in);
}

// ========== B-spline prefilter ==========

void TiledGridData::applyBSplinePrefilter(int order) {
    // Requires metadata to be loaded (via openForReading or construction + openForReading).
    // Closes the read-only file handle and reopens in read-write mode.
    if (m_filename.empty() || m_tileIndex.empty()) {
        throw OpenMMException("TiledGridData: Must call openForReading() before applyBSplinePrefilter()");
    }
    if (m_file.is_open()) {
        m_file.close();
    }

    int nx = m_counts[0], ny = m_counts[1], nz = m_counts[2];

    // Reopen file for read-write (preserving content)
    m_file.open(m_filename, ios::binary | ios::in | ios::out);
    if (!m_file.is_open()) {
        throw OpenMMException("TiledGridData: Unable to reopen file for prefiltering: " + m_filename);
    }

    // Helper: overwrite just the values portion of a tile at its known file offset
    // (skips the 6-byte tile dims header, leaves derivatives untouched)
    auto overwriteTileValues = [&](int tileX, int tileY, int tileZ,
                                    const vector<float>& values) {
        int tileIdx = getTileLinearIndex(tileX, tileY, tileZ);
        int64_t offset = m_tileIndex[tileIdx].fileOffset + 6; // skip dims header
        m_file.seekp(offset);
        m_file.write(reinterpret_cast<const char*>(values.data()),
                     values.size() * sizeof(float));
    };

    // --- Phase 1+2: z-filter + y-filter per x-slab ---
    // For each tileX, read all tiles with that tileX into a buffer of
    // sizeX × ny × nz, apply z-filter and y-filter, write back.
    for (int txIdx = 0; txIdx < m_numTilesX; txIdx++) {
        int startX = txIdx * m_tileSize;
        int sizeX = min(m_tileSize, nx - startX);

        // Allocate buffer: sizeX × ny × nz (row-major like full grid)
        vector<float> buf(sizeX * ny * nz, 0.0f);

        // Gather from tiles
        for (int tyIdx = 0; tyIdx < m_numTilesY; tyIdx++) {
            for (int tzIdx = 0; tzIdx < m_numTilesZ; tzIdx++) {
                int startY = tyIdx * m_tileSize;
                int startZ = tzIdx * m_tileSize;
                int tileSizeY, tileSizeZ, tileSizeX_check;
                getTileActualSize(txIdx, tyIdx, tzIdx, tileSizeX_check, tileSizeY, tileSizeZ);

                vector<float> tileVals; vector<char> tileDerivsUnused;
                readTile(txIdx, tyIdx, tzIdx, tileVals, tileDerivsUnused);

                for (int lx = 0; lx < sizeX; lx++) {
                    for (int ly = 0; ly < tileSizeY; ly++) {
                        for (int lz = 0; lz < tileSizeZ; lz++) {
                            int gy = startY + ly;
                            int gz = startZ + lz;
                            buf[lx * ny * nz + gy * nz + gz] =
                                tileVals[lx * tileSizeY * tileSizeZ + ly * tileSizeZ + lz];
                        }
                    }
                }
            }
        }

        // Apply z-filter: for each (lx, gy), filter along gz
        for (int lx = 0; lx < sizeX; lx++) {
            for (int gy = 0; gy < ny; gy++) {
                bsplinePrefilter1DByOrder(&buf[lx * ny * nz + gy * nz], nz, order, 1);
            }
        }

        // Apply y-filter: for each (lx, gz), filter along gy
        for (int lx = 0; lx < sizeX; lx++) {
            for (int gz = 0; gz < nz; gz++) {
                bsplinePrefilter1DByOrder(&buf[lx * ny * nz + gz], ny, order, nz);
            }
        }

        // Scatter back to tiles and write
        for (int tyIdx = 0; tyIdx < m_numTilesY; tyIdx++) {
            for (int tzIdx = 0; tzIdx < m_numTilesZ; tzIdx++) {
                int startY = tyIdx * m_tileSize;
                int startZ = tzIdx * m_tileSize;
                int tileSizeX_check, tileSizeY, tileSizeZ;
                getTileActualSize(txIdx, tyIdx, tzIdx, tileSizeX_check, tileSizeY, tileSizeZ);
                int numPoints = sizeX * tileSizeY * tileSizeZ;

                vector<float> tileVals(numPoints);
                for (int lx = 0; lx < sizeX; lx++) {
                    for (int ly = 0; ly < tileSizeY; ly++) {
                        for (int lz = 0; lz < tileSizeZ; lz++) {
                            int gy = startY + ly;
                            int gz = startZ + lz;
                            tileVals[lx * tileSizeY * tileSizeZ + ly * tileSizeZ + lz] =
                                buf[lx * ny * nz + gy * nz + gz];
                        }
                    }
                }
                overwriteTileValues(txIdx, tyIdx, tzIdx, tileVals);
            }
        }
    }

    // --- Phase 3: x-filter per (tileY, tileZ) column ---
    // For each (tileY, tileZ), read all tiles along x into a buffer of
    // nx × sizeY × sizeZ, apply x-filter, write back.
    for (int tyIdx = 0; tyIdx < m_numTilesY; tyIdx++) {
        for (int tzIdx = 0; tzIdx < m_numTilesZ; tzIdx++) {
            int startY = tyIdx * m_tileSize;
            int startZ = tzIdx * m_tileSize;
            int colSizeY = min(m_tileSize, ny - startY);
            int colSizeZ = min(m_tileSize, nz - startZ);
            int colYZ = colSizeY * colSizeZ;

            // Buffer: nx × colSizeY × colSizeZ
            vector<float> buf(nx * colYZ, 0.0f);

            // Gather from tiles along x
            for (int txIdx = 0; txIdx < m_numTilesX; txIdx++) {
                int startX = txIdx * m_tileSize;
                int tileSizeX, tileSizeY_check, tileSizeZ_check;
                getTileActualSize(txIdx, tyIdx, tzIdx, tileSizeX, tileSizeY_check, tileSizeZ_check);

                vector<float> tileVals; vector<char> tileDerivsUnused;
                readTile(txIdx, tyIdx, tzIdx, tileVals, tileDerivsUnused);

                for (int lx = 0; lx < tileSizeX; lx++) {
                    int gx = startX + lx;
                    for (int ly = 0; ly < colSizeY; ly++) {
                        for (int lz = 0; lz < colSizeZ; lz++) {
                            buf[gx * colYZ + ly * colSizeZ + lz] =
                                tileVals[lx * colSizeY * colSizeZ + ly * colSizeZ + lz];
                        }
                    }
                }
            }

            // Apply x-filter: for each (ly, lz), filter along gx
            for (int ly = 0; ly < colSizeY; ly++) {
                for (int lz = 0; lz < colSizeZ; lz++) {
                    bsplinePrefilter1DByOrder(&buf[ly * colSizeZ + lz], nx, order, colYZ);
                }
            }

            // Scatter back to tiles and write
            for (int txIdx = 0; txIdx < m_numTilesX; txIdx++) {
                int startX = txIdx * m_tileSize;
                int tileSizeX, tileSizeY_check, tileSizeZ_check;
                getTileActualSize(txIdx, tyIdx, tzIdx, tileSizeX, tileSizeY_check, tileSizeZ_check);
                int numPoints = tileSizeX * colSizeY * colSizeZ;

                vector<float> tileVals(numPoints);
                for (int lx = 0; lx < tileSizeX; lx++) {
                    int gx = startX + lx;
                    for (int ly = 0; ly < colSizeY; ly++) {
                        for (int lz = 0; lz < colSizeZ; lz++) {
                            tileVals[lx * colSizeY * colSizeZ + ly * colSizeZ + lz] =
                                buf[gx * colYZ + ly * colSizeZ + lz];
                        }
                    }
                }
                overwriteTileValues(txIdx, tyIdx, tzIdx, tileVals);
            }
        }
    }

    m_file.close();
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
