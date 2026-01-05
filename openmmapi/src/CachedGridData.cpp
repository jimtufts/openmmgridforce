#include "CachedGridData.h"
#include "openmm/OpenMMException.h"
#include <cmath>
#include <map>

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
      m_current_inv_power(0.0) {

    // Initialize current data as copies of original (untransformed)
    m_current_values = std::make_shared<std::vector<double>>(original_values);
    m_current_derivatives = std::make_shared<std::vector<double>>(original_derivs);
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
    if (mode != InvPowerMode::NONE && inv_power <= 0.0) {
        throw OpenMMException("CachedGridData: inv_power must be > 0 when mode != NONE");
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

    // Step 1: Revert to original (untransformed) state
    *m_current_values = m_original_values;
    if (!m_original_derivatives.empty()) {
        *m_current_derivatives = m_original_derivatives;
    }

    // Step 2: Apply new transformation if needed
    if (mode == InvPowerMode::RUNTIME || mode == InvPowerMode::STORED) {
        // Note: For STORED mode, we assume the grid file already has G^(1/n).
        // For RUNTIME mode, we transform here.
        if (mode == InvPowerMode::RUNTIME) {
            transformValues(*m_current_values, inv_power);
        }
        // For STORED mode, the grid is already transformed, so we don't transform here.
        // The original values ARE the transformed values.
    }

    // Step 3: Update state
    m_current_mode = mode;
    m_current_inv_power = inv_power;
}

// GridDataCache implementation

std::map<GridCacheKey, std::shared_ptr<CachedGridData>>& GridDataCache::getCache() {
    static std::map<GridCacheKey, std::shared_ptr<CachedGridData>> cache;
    return cache;
}

std::shared_ptr<CachedGridData> GridDataCache::get(const void* systemPtr,
                                                     const std::string& filename,
                                                     InvPowerMode mode,
                                                     double inv_power) {
    GridCacheKey key{systemPtr, filename, mode, inv_power};
    auto& cache = getCache();
    auto it = cache.find(key);
    if (it != cache.end()) {
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
    getCache()[key] = data;
}

void GridDataCache::clearSystem(const void* systemPtr) {
    auto& cache = getCache();
    for (auto it = cache.begin(); it != cache.end(); ) {
        if (it->first.systemPtr == systemPtr) {
            it = cache.erase(it);
        } else {
            ++it;
        }
    }
}

void GridDataCache::clearAll() {
    getCache().clear();
}

}  // namespace GridForcePlugin
