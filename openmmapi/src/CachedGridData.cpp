#include "CachedGridData.h"
#include "openmm/OpenMMException.h"
#include <cmath>
#include <map>
#include <list>

using namespace OpenMM;
using namespace std;

namespace GridForcePlugin {

CachedGridData::CachedGridData(const std::vector<double>& original_values,
                               const std::vector<double>& original_derivs,
                               const std::vector<int>& counts,
                               const std::vector<double>& spacing,
                               double origin_x, double origin_y, double origin_z)
    : m_original_values(original_values),
      m_original_derivatives(original_derivs),
      m_counts(counts),
      m_spacing(spacing),
      m_origin({origin_x, origin_y, origin_z}),
      m_current_mode(InvPowerMode::NONE),
      m_current_inv_power(0.0),
      m_isTransformed(false) {

    // Lazy init: alias current pointers to originals (no copy)
    m_current_values = std::shared_ptr<std::vector<double>>(
        &m_original_values, [](std::vector<double>*){});  // no-op deleter
    m_current_derivatives = std::shared_ptr<std::vector<double>>(
        &m_original_derivatives, [](std::vector<double>*){});
}

std::shared_ptr<std::vector<double>> CachedGridData::getCurrentValues() const {
    return m_current_values;
}

std::shared_ptr<std::vector<double>> CachedGridData::getCurrentDerivatives() const {
    return m_current_derivatives;
}

const std::vector<double>& CachedGridData::getOriginalValues() const {
    return m_original_values;
}

const std::vector<double>& CachedGridData::getOriginalDerivatives() const {
    return m_original_derivatives;
}

void CachedGridData::getCurrentTransformation(InvPowerMode& mode, double& inv_power) const {
    mode = m_current_mode;
    inv_power = m_current_inv_power;
}

size_t CachedGridData::getMemorySize() const {
    size_t bytes = sizeof(double) * (m_original_values.size() + m_original_derivatives.size());
    if (m_isTransformed) {
        bytes += sizeof(double) * (m_current_values->size() + m_current_derivatives->size());
    }
    return bytes;
}

void CachedGridData::transformValues(std::vector<double>& values, double inv_power) const {
    // Apply transformation: G -> sign(G) * |G|^(1/inv_power)
    for (size_t i = 0; i < values.size(); ++i) {
        if (values[i] != 0.0) {
            double sign = (values[i] >= 0.0) ? 1.0 : -1.0;
            values[i] = sign * std::pow(std::abs(values[i]), 1.0 / inv_power);
        }
    }
}

void CachedGridData::applyTransformation(InvPowerMode mode, double inv_power, int interpMethod) {
    // Validation
    if (mode != InvPowerMode::NONE && inv_power == 0.0) {
        throw OpenMMException("CachedGridData: inv_power must be non-zero when mode != NONE");
    }

    if (mode == InvPowerMode::NONE && inv_power != 0.0) {
        throw OpenMMException("CachedGridData: inv_power must be 0 when mode == NONE");
    }

    // RUNTIME mode validation
    if (mode == InvPowerMode::RUNTIME) {
        if (hasDerivatives()) {
            throw OpenMMException(
                "CachedGridData: RUNTIME mode cannot be used with grids that have analytical derivatives. "
                "Use STORED mode with pre-transformed grids instead.");
        }

        // Only trilinear (0) and b-spline (1) support RUNTIME mode
        if (interpMethod != 0 && interpMethod != 1) {
            throw OpenMMException(
                "CachedGridData: RUNTIME mode only supports trilinear (0) and b-spline (1) interpolation. "
                "Tricubic (2) and triquintic (3) require STORED mode with pre-transformed grids.");
        }
    }

    // If no change needed, return early
    if (mode == m_current_mode && inv_power == m_current_inv_power) {
        return;
    }

    if (mode == InvPowerMode::RUNTIME) {
        // RUNTIME mode needs a real separate copy to transform in-place
        if (!m_isTransformed) {
            // Allocate separate copies (currently aliasing originals)
            m_current_values = std::make_shared<std::vector<double>>(m_original_values);
            if (!m_original_derivatives.empty()) {
                m_current_derivatives = std::make_shared<std::vector<double>>(m_original_derivatives);
            }
            m_isTransformed = true;
        } else {
            // Already have separate copies; revert to original data first
            *m_current_values = m_original_values;
            if (!m_original_derivatives.empty()) {
                *m_current_derivatives = m_original_derivatives;
            }
        }
        transformValues(*m_current_values, inv_power);
    } else {
        // NONE or STORED mode: current == original, switch back to aliasing
        if (m_isTransformed) {
            // Free the separate allocations and alias back to originals
            m_current_values = std::shared_ptr<std::vector<double>>(
                &m_original_values, [](std::vector<double>*){});
            m_current_derivatives = std::shared_ptr<std::vector<double>>(
                &m_original_derivatives, [](std::vector<double>*){});
            m_isTransformed = false;
        }
    }

    // Update state
    m_current_mode = mode;
    m_current_inv_power = inv_power;
}

// GridDataCache implementation

std::map<GridCacheKey, std::shared_ptr<CachedGridData>>& GridDataCache::getCache() {
    static std::map<GridCacheKey, std::shared_ptr<CachedGridData>> cache;
    return cache;
}

size_t& GridDataCache::getMaxMemoryRef() {
    static size_t maxMemory = 0;  // 0 = unlimited
    return maxMemory;
}

size_t& GridDataCache::getCurrentMemoryRef() {
    static size_t currentMemory = 0;
    return currentMemory;
}

std::list<GridCacheKey>& GridDataCache::getLRUList() {
    static std::list<GridCacheKey> lruList;
    return lruList;
}

std::map<GridCacheKey, std::list<GridCacheKey>::iterator>& GridDataCache::getLRUMap() {
    static std::map<GridCacheKey, std::list<GridCacheKey>::iterator> lruMap;
    return lruMap;
}

std::shared_ptr<CachedGridData> GridDataCache::get(const void* systemPtr,
                                                     const std::string& filename,
                                                     InvPowerMode mode,
                                                     double inv_power) {
    GridCacheKey key{systemPtr, filename, mode, inv_power};
    auto& cache = getCache();
    auto it = cache.find(key);
    if (it != cache.end()) {
        // Bump to front of LRU list (most recently used)
        auto& lruList = getLRUList();
        auto& lruMap = getLRUMap();
        auto lruIt = lruMap.find(key);
        if (lruIt != lruMap.end()) {
            lruList.erase(lruIt->second);
            lruList.push_front(key);
            lruIt->second = lruList.begin();
        }
        return it->second;
    }
    return nullptr;
}

void GridDataCache::put(const void* systemPtr,
                         const std::string& filename,
                         InvPowerMode mode,
                         double inv_power,
                         std::shared_ptr<CachedGridData> data) {
    GridCacheKey key{systemPtr, filename, mode, inv_power};
    auto& cache = getCache();
    auto& lruList = getLRUList();
    auto& lruMap = getLRUMap();
    auto& currentMemory = getCurrentMemoryRef();

    // If key already exists, remove old entry from tracking
    auto existingIt = cache.find(key);
    if (existingIt != cache.end()) {
        currentMemory -= existingIt->second->getMemorySize();
        auto lruIt = lruMap.find(key);
        if (lruIt != lruMap.end()) {
            lruList.erase(lruIt->second);
            lruMap.erase(lruIt);
        }
    }

    // Insert new entry
    cache[key] = data;
    currentMemory += data->getMemorySize();
    lruList.push_front(key);
    lruMap[key] = lruList.begin();

    // Evict if over memory cap
    evictIfNeeded();
}

void GridDataCache::evictIfNeeded() {
    size_t maxMem = getMaxMemoryRef();
    if (maxMem == 0) return;  // unlimited

    auto& cache = getCache();
    auto& lruList = getLRUList();
    auto& lruMap = getLRUMap();
    auto& currentMemory = getCurrentMemoryRef();

    // Evict from back of LRU list (least recently used)
    while (currentMemory > maxMem && !lruList.empty()) {
        GridCacheKey evictKey = lruList.back();
        lruList.pop_back();

        auto cacheIt = cache.find(evictKey);
        if (cacheIt != cache.end()) {
            currentMemory -= cacheIt->second->getMemorySize();
            cache.erase(cacheIt);
        }
        lruMap.erase(evictKey);
    }
}

void GridDataCache::clearSystem(const void* systemPtr) {
    auto& cache = getCache();
    auto& lruList = getLRUList();
    auto& lruMap = getLRUMap();
    auto& currentMemory = getCurrentMemoryRef();

    for (auto it = cache.begin(); it != cache.end(); ) {
        if (it->first.systemPtr == systemPtr) {
            currentMemory -= it->second->getMemorySize();
            auto lruIt = lruMap.find(it->first);
            if (lruIt != lruMap.end()) {
                lruList.erase(lruIt->second);
                lruMap.erase(lruIt);
            }
            it = cache.erase(it);
        } else {
            ++it;
        }
    }
}

void GridDataCache::clearAll() {
    getCache().clear();
    getLRUList().clear();
    getLRUMap().clear();
    getCurrentMemoryRef() = 0;
}

void GridDataCache::setMaxHostMemory(size_t bytes) {
    getMaxMemoryRef() = bytes;
    if (bytes > 0) {
        evictIfNeeded();
    }
}

size_t GridDataCache::getHostMemoryUsage() {
    return getCurrentMemoryRef();
}

size_t GridDataCache::getMaxHostMemory() {
    return getMaxMemoryRef();
}

}  // namespace GridForcePlugin
