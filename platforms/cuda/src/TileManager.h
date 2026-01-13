#ifndef TILE_MANAGER_H_
#define TILE_MANAGER_H_

#include "openmm/cuda/CudaContext.h"
#include "openmm/cuda/CudaArray.h"
#include <vector>
#include <map>
#include <list>
#include <set>
#include <memory>
#include <cstdint>

namespace GridForcePlugin {

/**
 * Configuration for tile-based grid storage.
 */
struct TileConfig {
    static constexpr int DEFAULT_TILE_SIZE = 64;    // 64^3 points per tile (core region)
    static constexpr int DEFAULT_OVERLAP = 4;        // Overlap for interpolation stencil

    int tileSize;      // Core tile size (excluding overlap)
    int overlap;       // Overlap region for interpolation
    size_t memoryBudget;  // GPU memory budget in bytes

    TileConfig() : tileSize(DEFAULT_TILE_SIZE), overlap(DEFAULT_OVERLAP),
                   memoryBudget(2ULL * 1024 * 1024 * 1024) {}  // 2 GB default

    // Actual tile dimensions including overlap
    int getTileWithOverlap() const { return tileSize + 2 * overlap; }

    // Memory per tile (values only)
    size_t getTileValuesMemory() const {
        int s = getTileWithOverlap();
        return s * s * s * sizeof(float);
    }

    // Memory per tile (with 27 derivatives)
    size_t getTileDerivativesMemory() const {
        int s = getTileWithOverlap();
        return 27 * s * s * s * sizeof(float);
    }

    // Total memory per tile
    size_t getTileTotalMemory(bool hasDerivatives) const {
        return getTileValuesMemory() + (hasDerivatives ? getTileDerivativesMemory() : 0);
    }
};

/**
 * Tile identifier (coordinates in tile-space).
 */
struct TileID {
    int tx, ty, tz;

    TileID() : tx(0), ty(0), tz(0) {}
    TileID(int x, int y, int z) : tx(x), ty(y), tz(z) {}

    bool operator<(const TileID& other) const {
        if (tx != other.tx) return tx < other.tx;
        if (ty != other.ty) return ty < other.ty;
        return tz < other.tz;
    }

    bool operator==(const TileID& other) const {
        return tx == other.tx && ty == other.ty && tz == other.tz;
    }

    // Pack into single int for GPU hash map (assumes tile coords < 1024)
    int pack() const {
        return tx | (ty << 10) | (tz << 20);
    }
};

/**
 * GPU-resident tile data.
 */
struct GPUTile {
    TileID id;
    std::unique_ptr<OpenMM::CudaArray> values;       // Tile values with overlap
    std::unique_ptr<OpenMM::CudaArray> derivatives;  // Tile derivatives (optional)
    int3 gridOffset;         // Position in full grid coordinates (corner of core region)
    uint64_t lastAccess;     // Timestamp for LRU eviction

    size_t getMemoryUsage(bool hasDerivatives) const {
        size_t mem = values ? values->getSize() * values->getElementSize() : 0;
        if (hasDerivatives && derivatives) {
            mem += derivatives->getSize() * derivatives->getElementSize();
        }
        return mem;
    }
};

/**
 * Host-side tiled grid that wraps existing grid data and extracts tiles on demand.
 * Does not own the underlying data - just provides tile-based access.
 */
class TiledGrid {
public:
    TiledGrid();

    /**
     * Initialize from existing host-side grid data.
     *
     * @param values Pointer to grid values array (nx * ny * nz floats)
     * @param derivatives Pointer to derivatives array (27 * nx * ny * nz floats, or nullptr)
     * @param nx, ny, nz Grid dimensions
     * @param spacingX, spacingY, spacingZ Grid spacing
     * @param originX, originY, originZ Grid origin
     * @param config Tile configuration
     */
    void initFromGridData(const float* values, const float* derivatives,
                          int nx, int ny, int nz,
                          float spacingX, float spacingY, float spacingZ,
                          float originX, float originY, float originZ,
                          const TileConfig& config);

    /**
     * Get tile data for GPU upload (includes overlap region).
     * Extracts a tile from the host-side grid data.
     */
    void getTileData(const TileID& id,
                     std::vector<float>& tileValues,
                     std::vector<float>* tileDerivatives = nullptr) const;

    /**
     * Map world position to tile ID.
     */
    TileID positionToTile(float x, float y, float z) const;

    /**
     * Get all tiles needed for a set of positions.
     */
    std::set<TileID> getRequiredTiles(const std::vector<float>& positions) const;

    /**
     * Get grid offset for a tile (where tile starts in full grid coordinates).
     */
    int3 getTileGridOffset(const TileID& id) const;

    // Accessors
    int3 getFullGridSize() const { return make_int3(nx_, ny_, nz_); }
    int3 getTileCount() const { return make_int3(numTilesX_, numTilesY_, numTilesZ_); }
    float3 getSpacing() const { return make_float3(spacingX_, spacingY_, spacingZ_); }
    float3 getOrigin() const { return make_float3(originX_, originY_, originZ_); }
    bool hasDerivatives() const { return derivatives_ != nullptr; }
    const TileConfig& getConfig() const { return config_; }

private:
    const float* values_;
    const float* derivatives_;
    int nx_, ny_, nz_;
    float spacingX_, spacingY_, spacingZ_;
    float originX_, originY_, originZ_;
    int numTilesX_, numTilesY_, numTilesZ_;
    TileConfig config_;
    bool initialized_;
};

/**
 * GPU tile cache with LRU eviction.
 */
class TileCache {
public:
    TileCache(OpenMM::CudaContext& cu, size_t memoryBudget);
    ~TileCache();

    /**
     * Get a tile, loading it if necessary. May evict other tiles.
     */
    GPUTile* getTile(const TileID& id, const TiledGrid& hostGrid);

    /**
     * Check if a tile is loaded.
     */
    bool hasTile(const TileID& id) const;

    /**
     * Load multiple tiles, evicting as needed.
     */
    void loadTiles(const std::vector<TileID>& tiles, const TiledGrid& hostGrid);

    /**
     * Evict tiles to free memory.
     */
    void evictToFit(size_t requiredMemory);

    /**
     * Clear all loaded tiles.
     */
    void clear();

    // Statistics
    size_t getMemoryUsage() const { return currentMemory_; }
    size_t getMaxMemory() const { return maxMemory_; }
    size_t getTileCount() const { return loadedTiles_.size(); }
    size_t getCacheHits() const { return cacheHits_; }
    size_t getCacheMisses() const { return cacheMisses_; }
    size_t getEvictions() const { return evictions_; }
    float getHitRate() const {
        size_t total = cacheHits_ + cacheMisses_;
        return total > 0 ? float(cacheHits_) / total : 0.0f;
    }

private:
    void loadTile(const TileID& id, const TiledGrid& hostGrid);
    void evictOldest();
    void updateLRU(const TileID& id);

    OpenMM::CudaContext& cu_;
    size_t maxMemory_;
    size_t currentMemory_;
    bool hasDerivatives_;

    std::map<TileID, std::unique_ptr<GPUTile>> loadedTiles_;
    std::list<TileID> lruOrder_;  // Front = oldest, back = newest

    uint64_t accessCounter_;
    size_t cacheHits_;
    size_t cacheMisses_;
    size_t evictions_;
};

/**
 * Lookup table passed to CUDA kernel for tile-based interpolation.
 */
struct TileLookupTable {
    int numLoadedTiles;
    OpenMM::CudaArray tileOffsets;      // int array: grid offsets (x,y,z,x,y,z,...) for each tile
    OpenMM::CudaArray tileValuePtrs;    // unsigned long long array: device pointers to tile values
    OpenMM::CudaArray tileDerivPtrs;    // unsigned long long array: device pointers to tile derivatives
    OpenMM::CudaArray tileHashMap;      // Spatial hash map for tile lookup
    int hashMapSize;

    TileLookupTable() : numLoadedTiles(0), hashMapSize(0) {}
};

/**
 * Coordinates tile loading and provides lookup table for kernel.
 */
class TileManager {
public:
    TileManager(OpenMM::CudaContext& cu, size_t memoryBudget);
    ~TileManager();

    /**
     * Initialize from host-side grid data.
     */
    void initFromGridData(const float* values, const float* derivatives,
                          int nx, int ny, int nz,
                          float spacingX, float spacingY, float spacingZ,
                          float originX, float originY, float originZ,
                          const TileConfig& config);

    /**
     * Prepare tiles for force computation.
     * Determines which tiles are needed, loads them, and builds lookup table.
     *
     * @param positions Particle positions (x,y,z,x,y,z,...)
     * @return True if all required tiles were loaded successfully
     */
    bool prepareTiles(const std::vector<float>& positions);

    /**
     * Get the lookup table for kernel invocation.
     */
    const TileLookupTable& getLookupTable() const { return lookupTable_; }

    /**
     * Get tile configuration.
     */
    const TileConfig& getConfig() const { return hostGrid_.getConfig(); }

    /**
     * Get cache statistics.
     */
    size_t getCacheHits() const { return cache_.getCacheHits(); }
    size_t getCacheMisses() const { return cache_.getCacheMisses(); }
    size_t getEvictions() const { return cache_.getEvictions(); }
    float getHitRate() const { return cache_.getHitRate(); }

    /**
     * Clear all cached tiles.
     */
    void clearCache() { cache_.clear(); }

private:
    void buildLookupTable(const std::set<TileID>& tiles);

    OpenMM::CudaContext& cu_;
    TiledGrid hostGrid_;
    TileCache cache_;
    TileLookupTable lookupTable_;
    bool initialized_;
};

} // namespace GridForcePlugin

#endif // TILE_MANAGER_H_
