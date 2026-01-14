#include "TileManager.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

using namespace OpenMM;
using namespace GridForcePlugin;

// ============================================================================
// TiledGrid implementation
// ============================================================================

TiledGrid::TiledGrid()
    : values_(nullptr), derivatives_(nullptr),
      nx_(0), ny_(0), nz_(0),
      spacingX_(0), spacingY_(0), spacingZ_(0),
      originX_(0), originY_(0), originZ_(0),
      numTilesX_(0), numTilesY_(0), numTilesZ_(0),
      initialized_(false) {
}

void TiledGrid::initFromGridData(const float* values, const float* derivatives,
                                  int nx, int ny, int nz,
                                  float spacingX, float spacingY, float spacingZ,
                                  float originX, float originY, float originZ,
                                  const TileConfig& config) {
    values_ = values;
    derivatives_ = derivatives;
    nx_ = nx;
    ny_ = ny;
    nz_ = nz;
    spacingX_ = spacingX;
    spacingY_ = spacingY;
    spacingZ_ = spacingZ;
    originX_ = originX;
    originY_ = originY;
    originZ_ = originZ;
    config_ = config;

    // Calculate number of tiles in each dimension
    // Tiles cover the grid with overlap at boundaries
    numTilesX_ = (nx + config.tileSize - 1) / config.tileSize;
    numTilesY_ = (ny + config.tileSize - 1) / config.tileSize;
    numTilesZ_ = (nz + config.tileSize - 1) / config.tileSize;

    initialized_ = true;
}

TileID TiledGrid::positionToTile(float x, float y, float z) const {
    // Convert world position to grid index
    float gx = (x - originX_) / spacingX_;
    float gy = (y - originY_) / spacingY_;
    float gz = (z - originZ_) / spacingZ_;

    // Convert grid index to tile index
    int tx = std::max(0, std::min(numTilesX_ - 1, (int)(gx / config_.tileSize)));
    int ty = std::max(0, std::min(numTilesY_ - 1, (int)(gy / config_.tileSize)));
    int tz = std::max(0, std::min(numTilesZ_ - 1, (int)(gz / config_.tileSize)));

    return TileID(tx, ty, tz);
}

std::set<TileID> TiledGrid::getRequiredTiles(const std::vector<float>& positions) const {
    std::set<TileID> tiles;
    for (size_t i = 0; i + 2 < positions.size(); i += 3) {
        tiles.insert(positionToTile(positions[i], positions[i+1], positions[i+2]));
    }
    return tiles;
}

int3 TiledGrid::getTileGridOffset(const TileID& id) const {
    // The grid offset is where the tile's core region starts (excluding overlap)
    return make_int3(
        id.tx * config_.tileSize,
        id.ty * config_.tileSize,
        id.tz * config_.tileSize
    );
}

void TiledGrid::getTileData(const TileID& id,
                            std::vector<float>& tileValues,
                            std::vector<float>* tileDerivatives) const {
    if (!initialized_) {
        throw std::runtime_error("TiledGrid not initialized");
    }

    int tileWithOverlap = config_.getTileWithOverlap();
    int overlap = config_.overlap;

    // Calculate source region in full grid (with clamping at boundaries)
    int3 offset = getTileGridOffset(id);

    // Start position including overlap (may be negative at boundaries)
    int srcStartX = offset.x - overlap;
    int srcStartY = offset.y - overlap;
    int srcStartZ = offset.z - overlap;

    // Allocate tile arrays
    size_t tilePoints = tileWithOverlap * tileWithOverlap * tileWithOverlap;
    tileValues.resize(tilePoints);

    if (tileDerivatives && derivatives_) {
        tileDerivatives->resize(27 * tilePoints);
    }

    // Extract tile data with boundary handling (clamp to edge)
    for (int lx = 0; lx < tileWithOverlap; lx++) {
        int gx = std::max(0, std::min(nx_ - 1, srcStartX + lx));

        for (int ly = 0; ly < tileWithOverlap; ly++) {
            int gy = std::max(0, std::min(ny_ - 1, srcStartY + ly));

            for (int lz = 0; lz < tileWithOverlap; lz++) {
                int gz = std::max(0, std::min(nz_ - 1, srcStartZ + lz));

                // Linear indices
                int tileIdx = lx * tileWithOverlap * tileWithOverlap + ly * tileWithOverlap + lz;
                int gridIdx = gx * ny_ * nz_ + gy * nz_ + gz;

                // Copy value
                tileValues[tileIdx] = values_[gridIdx];

                // Copy derivatives if present
                if (tileDerivatives && derivatives_) {
                    for (int d = 0; d < 27; d++) {
                        (*tileDerivatives)[d * tilePoints + tileIdx] =
                            derivatives_[d * (nx_ * ny_ * nz_) + gridIdx];
                    }
                }
            }
        }
    }
}

// ============================================================================
// TileCache implementation
// ============================================================================

TileCache::TileCache(CudaContext& cu, size_t memoryBudget)
    : cu_(cu), maxMemory_(memoryBudget), currentMemory_(0), hasDerivatives_(false),
      accessCounter_(0), cacheHits_(0), cacheMisses_(0), evictions_(0) {
}

TileCache::~TileCache() {
    clear();
}

bool TileCache::hasTile(const TileID& id) const {
    return loadedTiles_.find(id) != loadedTiles_.end();
}

GPUTile* TileCache::getTile(const TileID& id, const TiledGrid& hostGrid) {
    auto it = loadedTiles_.find(id);
    if (it != loadedTiles_.end()) {
        cacheHits_++;
        updateLRU(id);
        it->second->lastAccess = ++accessCounter_;
        return it->second.get();
    }

    cacheMisses_++;
    loadTile(id, hostGrid);
    return loadedTiles_[id].get();
}

void TileCache::loadTiles(const std::vector<TileID>& tiles, const TiledGrid& hostGrid) {
    hasDerivatives_ = hostGrid.hasDerivatives();

    for (const TileID& id : tiles) {
        if (!hasTile(id)) {
            loadTile(id, hostGrid);
        }
    }
}

void TileCache::loadTile(const TileID& id, const TiledGrid& hostGrid) {
    const TileConfig& config = hostGrid.getConfig();
    hasDerivatives_ = hostGrid.hasDerivatives();
    size_t tileMemory = config.getTileTotalMemory(hasDerivatives_);

    // Evict tiles if needed
    while (currentMemory_ + tileMemory > maxMemory_ && !lruOrder_.empty()) {
        evictOldest();
    }

    // Get tile data from host
    std::vector<float> values;
    std::vector<float> derivatives;
    hostGrid.getTileData(id, values, hasDerivatives_ ? &derivatives : nullptr);

    // Create GPU tile
    std::unique_ptr<GPUTile> tile(new GPUTile());
    tile->id = id;
    tile->gridOffset = hostGrid.getTileGridOffset(id);
    tile->lastAccess = ++accessCounter_;

    // Upload values
    tile->values.reset(new CudaArray(cu_, values.size(), sizeof(float), "tileValues"));
    tile->values->upload(values);

    // Upload derivatives if present
    if (hasDerivatives_ && !derivatives.empty()) {
        tile->derivatives.reset(new CudaArray(cu_, derivatives.size(), sizeof(float), "tileDerivatives"));
        tile->derivatives->upload(derivatives);
    }

    currentMemory_ += tileMemory;
    lruOrder_.push_back(id);
    loadedTiles_[id] = std::move(tile);
}

void TileCache::evictOldest() {
    if (lruOrder_.empty()) return;

    TileID oldest = lruOrder_.front();
    lruOrder_.pop_front();

    auto it = loadedTiles_.find(oldest);
    if (it != loadedTiles_.end()) {
        const TileConfig config;  // Use default for memory calculation
        size_t tileMemory = it->second->getMemoryUsage(hasDerivatives_);
        currentMemory_ -= tileMemory;
        loadedTiles_.erase(it);
        evictions_++;
    }
}

void TileCache::evictToFit(size_t requiredMemory) {
    while (currentMemory_ + requiredMemory > maxMemory_ && !lruOrder_.empty()) {
        evictOldest();
    }
}

void TileCache::updateLRU(const TileID& id) {
    // Move id to back of LRU list (most recently used)
    lruOrder_.remove(id);
    lruOrder_.push_back(id);
}

void TileCache::clear() {
    loadedTiles_.clear();
    lruOrder_.clear();
    currentMemory_ = 0;
}

// ============================================================================
// TileManager implementation
// ============================================================================

TileManager::TileManager(CudaContext& cu, size_t memoryBudget)
    : cu_(cu), cache_(cu, memoryBudget), initialized_(false) {
}

TileManager::~TileManager() {
}

void TileManager::initFromGridData(const float* values, const float* derivatives,
                                    int nx, int ny, int nz,
                                    float spacingX, float spacingY, float spacingZ,
                                    float originX, float originY, float originZ,
                                    const TileConfig& config) {
    hostGrid_.initFromGridData(values, derivatives, nx, ny, nz,
                               spacingX, spacingY, spacingZ,
                               originX, originY, originZ, config);
    initialized_ = true;
}

bool TileManager::prepareTiles(const std::vector<float>& positions) {
    if (!initialized_) {
        throw std::runtime_error("TileManager not initialized");
    }

    // Determine required tiles
    std::set<TileID> requiredTiles = hostGrid_.getRequiredTiles(positions);

    // Load missing tiles
    std::vector<TileID> tilesToLoad;
    for (const TileID& id : requiredTiles) {
        if (!cache_.hasTile(id)) {
            tilesToLoad.push_back(id);
        }
    }

    if (!tilesToLoad.empty()) {
        cache_.loadTiles(tilesToLoad, hostGrid_);
    }

    // Build lookup table for kernel
    buildLookupTable(requiredTiles);

    return true;
}

void TileManager::buildLookupTable(const std::set<TileID>& tiles) {
    lookupTable_.numLoadedTiles = tiles.size();

    if (tiles.empty()) return;

    // Collect tile info
    std::vector<int> offsets;  // Packed as x, y, z, x, y, z, ...
    std::vector<unsigned long long> valuePtrs;  // CUdeviceptr is unsigned long long
    std::vector<unsigned long long> derivPtrs;

    for (const TileID& id : tiles) {
        GPUTile* tile = cache_.getTile(id, hostGrid_);

        offsets.push_back(tile->gridOffset.x);
        offsets.push_back(tile->gridOffset.y);
        offsets.push_back(tile->gridOffset.z);

        valuePtrs.push_back(tile->values->getDevicePointer());
        derivPtrs.push_back(tile->derivatives ? tile->derivatives->getDevicePointer() : 0);
    }

    // Upload to GPU
    // Check if arrays need to be (re)initialized
    // CudaArray::initialize can only be called once, so we need to check size and reinitialize if needed

    // Offsets: use int array with 3 ints per tile
    if (!lookupTable_.tileOffsets.isInitialized() || lookupTable_.tileOffsets.getSize() != offsets.size()) {
        // Need to create a new array - CudaArray doesn't support resize
        // The old array will be destroyed when we assign a new one, but CudaArray is not copyable
        // So we need to work around this by using a different approach
        if (lookupTable_.tileOffsets.isInitialized()) {
            // Resize by uploading to same buffer if size matches, else need workaround
            // For now, just upload if size matches; otherwise recreate the context arrays
        }
        if (!lookupTable_.tileOffsets.isInitialized()) {
            lookupTable_.tileOffsets.initialize<int>(cu_, offsets.size(), "tileOffsets");
        }
    }
    lookupTable_.tileOffsets.upload(offsets);

    // Pointers: device pointers (as unsigned long long)
    if (!lookupTable_.tileValuePtrs.isInitialized()) {
        lookupTable_.tileValuePtrs.initialize<unsigned long long>(cu_, valuePtrs.size(), "tileValuePtrs");
    }
    lookupTable_.tileValuePtrs.upload(valuePtrs);

    if (!lookupTable_.tileDerivPtrs.isInitialized()) {
        lookupTable_.tileDerivPtrs.initialize<unsigned long long>(cu_, derivPtrs.size(), "tileDerivPtrs");
    }
    lookupTable_.tileDerivPtrs.upload(derivPtrs);

    // Build spatial hash map for fast tile lookup in kernel
    // Use simple linear search for now (works for small number of tiles)
    // TODO: Implement proper hash map for many tiles
    lookupTable_.hashMapSize = tiles.size();
}
