/* -------------------------------------------------------------------------- *
 *                              OpenMMGridForce                               *
 * -------------------------------------------------------------------------- *
 * Tile-based grid storage format for large grids that don't fit in memory.
 *
 * File Format (version 1):
 *   [Header 64 bytes] [Grid Metadata] [Tile Index] [Tile 0] [Tile 1] ... [Tile N]
 *
 * Benefits:
 *   - Generate tile-by-tile without holding full grid in memory
 *   - Load tiles on demand during evaluation
 *   - Random access via tile index
 * -------------------------------------------------------------------------- */

#ifndef OPENMM_TILED_GRID_DATA_H_
#define OPENMM_TILED_GRID_DATA_H_

#include "GridForceTypes.h"
#include <string>
#include <vector>
#include <fstream>
#include <memory>
#include <cstdint>

namespace GridForcePlugin {

// File format constants
constexpr char TILED_GRID_MAGIC[8] = "OMGTILE";
constexpr uint32_t TILED_GRID_VERSION = 1;
constexpr uint32_t TILED_GRID_HEADER_SIZE = 64;

// Default tile size (32^3 = 32768 points per tile)
constexpr int DEFAULT_TILE_SIZE = 32;

// Flags for header
constexpr uint32_t TILED_FLAG_HAS_DERIVATIVES = 0x01;
constexpr uint32_t TILED_FLAG_COMPRESSED = 0x02;  // Future: per-tile compression

/**
 * Entry in the tile index - describes location of one tile in the file.
 */
struct TileIndexEntry {
    int32_t tileX, tileY, tileZ;  // Tile coordinates (in tile units, not grid points)
    int64_t fileOffset;           // Byte offset in file where tile data starts
    int64_t dataSize;             // Size of tile data in bytes (for validation/compression)
};

/**
 * TiledGridData - manages tile-based grid storage.
 *
 * Usage for generation:
 *   TiledGridData tiled(nx, ny, nz, dx, dy, dz, tileSize);
 *   tiled.beginWriting("output.grid");
 *   for each tile:
 *       std::vector<float> tileData = generateTile(tx, ty, tz);
 *       tiled.writeTile(tx, ty, tz, tileData, tileDerivatives);
 *   tiled.finishWriting();
 *
 * Usage for evaluation:
 *   TiledGridData tiled;
 *   tiled.openForReading("input.grid");
 *   auto data = tiled.readTile(tx, ty, tz);
 */
class TiledGridData {
public:
    TiledGridData();

    /**
     * Constructor for creating a new tiled grid.
     */
    TiledGridData(int nx, int ny, int nz,
                  double dx, double dy, double dz,
                  int tileSize = DEFAULT_TILE_SIZE);

    ~TiledGridData();

    // Grid metadata accessors
    const std::vector<int>& getCounts() const { return m_counts; }
    const std::vector<double>& getSpacing() const { return m_spacing; }
    const std::vector<double>& getOrigin() const { return m_origin; }
    int getTileSize() const { return m_tileSize; }
    bool hasDerivatives() const { return m_hasDerivatives; }

    void setOrigin(double ox, double oy, double oz);
    void setInvPower(double invPower) { m_invPower = invPower; }
    void setInvPowerMode(InvPowerMode mode) { m_invPowerMode = mode; }
    double getInvPower() const { return m_invPower; }
    InvPowerMode getInvPowerMode() const { return m_invPowerMode; }

    // Tile coordinate helpers
    int getNumTilesX() const { return m_numTilesX; }
    int getNumTilesY() const { return m_numTilesY; }
    int getNumTilesZ() const { return m_numTilesZ; }
    int getTotalNumTiles() const { return m_numTilesX * m_numTilesY * m_numTilesZ; }

    /**
     * Get the grid point range covered by a tile.
     * Returns [startX, startY, startZ, endX, endY, endZ] (end is exclusive)
     */
    std::vector<int> getTileGridRange(int tileX, int tileY, int tileZ) const;

    /**
     * Get the actual size of a tile (may be smaller at boundaries).
     */
    void getTileActualSize(int tileX, int tileY, int tileZ,
                           int& sizeX, int& sizeY, int& sizeZ) const;

    // ========== Writing API ==========

    /**
     * Begin writing a new tiled grid file.
     * @param filename Output file path
     * @param hasDerivatives Whether tiles will include derivative data
     */
    void beginWriting(const std::string& filename, bool hasDerivatives = true);

    /**
     * Write a single tile's data.
     * @param tileX, tileY, tileZ Tile coordinates
     * @param values Grid values for this tile (tileSize^3 floats, or actual size at boundaries)
     * @param derivatives Optional derivatives (27 * numPoints floats)
     *
     * Values layout: [z-fastest, then y, then x] within the tile
     * Derivatives layout: [deriv_idx * numPoints + point_idx]
     */
    void writeTile(int tileX, int tileY, int tileZ,
                   const std::vector<float>& values,
                   const std::vector<float>& derivatives = {});

    /**
     * Finish writing - writes tile index and closes file.
     */
    void finishWriting();

    // ========== Reading API ==========

    /**
     * Open a tiled grid file for reading.
     */
    void openForReading(const std::string& filename);

    /**
     * Read a single tile's data.
     * @param tileX, tileY, tileZ Tile coordinates
     * @param values Output: grid values
     * @param derivatives Output: derivatives (if file has them)
     */
    void readTile(int tileX, int tileY, int tileZ,
                  std::vector<float>& values,
                  std::vector<float>& derivatives) const;

    /**
     * Check if a tile exists in the file.
     */
    bool hasTile(int tileX, int tileY, int tileZ) const;

    /**
     * Close the file.
     */
    void close();

    /**
     * Check if file is open.
     */
    bool isOpen() const { return m_file.is_open(); }

    // ========== Static utilities ==========

    /**
     * Check if a file is in tiled format (vs monolithic).
     */
    static bool isTiledFormat(const std::string& filename);

private:
    // Grid dimensions
    std::vector<int> m_counts;      // [nx, ny, nz]
    std::vector<double> m_spacing;  // [dx, dy, dz]
    std::vector<double> m_origin;   // [ox, oy, oz]

    // Tile parameters
    int m_tileSize;
    int m_numTilesX, m_numTilesY, m_numTilesZ;
    bool m_hasDerivatives;

    // Transformation parameters
    double m_invPower;
    InvPowerMode m_invPowerMode;

    // File I/O
    mutable std::fstream m_file;
    std::string m_filename;
    bool m_isWriting;

    // Tile index (populated during write, or loaded during read)
    std::vector<TileIndexEntry> m_tileIndex;
    int64_t m_tileIndexOffset;  // File offset where tile index is stored

    // Helper methods
    void writeHeader();
    void readHeader();
    void writeTileIndex();
    void readTileIndex();
    int getTileLinearIndex(int tileX, int tileY, int tileZ) const;
    void computeTileCounts();
};

} // namespace GridForcePlugin

#endif // OPENMM_TILED_GRID_DATA_H_
