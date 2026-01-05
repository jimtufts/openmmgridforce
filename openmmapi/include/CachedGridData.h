#ifndef OPENMM_CACHED_GRID_DATA_H_
#define OPENMM_CACHED_GRID_DATA_H_

#include <vector>
#include <memory>
#include <string>
#include <map>
#include "GridForceTypes.h"

namespace GridForcePlugin {

/**
 * Cached grid data with transformation state tracking.
 *
 * Stores grid values and metadata, tracking the current transformation state
 * to enable safe revert-and-reapply of inv_power transformations.
 *
 * Design:
 * - Stores original untransformed grid values
 * - Tracks current transformation state (mode, power)
 * - Provides safe transformation with automatic reversion
 * - Shared via shared_ptr for memory efficiency in multi-ligand workflows
 */
class CachedGridData {
public:
    /**
     * Construct cached grid data.
     *
     * @param original_values   Untransformed grid values (will be copied)
     * @param original_derivs   Untransformed derivatives (will be copied, empty if none)
     * @param counts           Grid dimensions [nx, ny, nz]
     * @param spacing          Grid spacing [dx, dy, dz]
     * @param origin_x         Grid origin x coordinate
     * @param origin_y         Grid origin y coordinate
     * @param origin_z         Grid origin z coordinate
     */
    CachedGridData(const std::vector<double>& original_values,
                   const std::vector<double>& original_derivs,
                   const std::vector<int>& counts,
                   const std::vector<double>& spacing,
                   double origin_x, double origin_y, double origin_z);

    /**
     * Get current grid values (transformed or original depending on state).
     *
     * @return shared_ptr to current values (may be transformed)
     */
    std::shared_ptr<std::vector<double>> getCurrentValues() const;

    /**
     * Get current derivatives (transformed or original depending on state).
     *
     * @return shared_ptr to current derivatives (empty if no derivatives)
     */
    std::shared_ptr<std::vector<double>> getCurrentDerivatives() const;

    /**
     * Get original untransformed values.
     *
     * @return const reference to original values
     */
    const std::vector<double>& getOriginalValues() const;

    /**
     * Get original untransformed derivatives.
     *
     * @return const reference to original derivatives
     */
    const std::vector<double>& getOriginalDerivatives() const;

    /**
     * Apply or change inv_power transformation.
     *
     * This method:
     * 1. Reverts any existing transformation (back to original)
     * 2. Applies new transformation if mode != NONE
     * 3. Updates current state
     *
     * RUNTIME mode requirements:
     * - Grid must NOT have derivatives
     * - Interpolation method must be trilinear (0) or b-spline (1)
     *
     * @param mode          Transformation mode
     * @param inv_power     Transformation exponent (must be > 0 if mode != NONE)
     * @param interpMethod  Interpolation method (for validation)
     * @throws OpenMMException if validation fails
     */
    void applyTransformation(InvPowerMode mode, double inv_power, int interpMethod);

    /**
     * Get current transformation state.
     *
     * @param mode       Output: current mode
     * @param inv_power  Output: current inv_power value
     */
    void getCurrentTransformation(InvPowerMode& mode, double& inv_power) const;

    /**
     * Get grid metadata.
     */
    const std::vector<int>& getCounts() const { return m_counts; }
    const std::vector<double>& getSpacing() const { return m_spacing; }
    void getOrigin(double& x, double& y, double& z) const {
        x = m_origin[0];
        y = m_origin[1];
        z = m_origin[2];
    }

    /**
     * Check if grid has derivatives.
     */
    bool hasDerivatives() const {
        return !m_original_derivatives.empty();
    }

private:
    // Original untransformed data (immutable)
    std::vector<double> m_original_values;
    std::vector<double> m_original_derivatives;

    // Current transformed data (mutated by applyTransformation)
    std::shared_ptr<std::vector<double>> m_current_values;
    std::shared_ptr<std::vector<double>> m_current_derivatives;

    // Grid metadata
    std::vector<int> m_counts;       // [nx, ny, nz]
    std::vector<double> m_spacing;   // [dx, dy, dz]
    std::vector<double> m_origin;    // [x, y, z]

    // Current transformation state
    InvPowerMode m_current_mode;
    double m_current_inv_power;

    // Helper: Apply inv_power transformation to values
    void transformValues(std::vector<double>& values, double inv_power) const;
};

/**
 * Cache key for per-System grid data sharing.
 */
struct GridCacheKey {
    const void* systemPtr;           // Pointer to System (for scoping)
    std::string filename;             // Grid file path
    InvPowerMode mode;                // Transformation mode
    double inv_power;                 // Transformation exponent

    bool operator<(const GridCacheKey& other) const {
        if (systemPtr != other.systemPtr) return systemPtr < other.systemPtr;
        if (filename != other.filename) return filename < other.filename;
        if (mode != other.mode) return mode < other.mode;
        return inv_power < other.inv_power;
    }
};

/**
 * Per-System grid data cache.
 *
 * Manages shared grid data to enable memory-efficient multi-ligand simulations.
 * Cache is scoped per-System and indexed by (filename, mode, inv_power).
 */
class GridDataCache {
public:
    /**
     * Get cached grid data, or return nullptr if not cached.
     *
     * @param systemPtr    Pointer to System (for cache scoping)
     * @param filename     Grid file path
     * @param mode         Desired transformation mode
     * @param inv_power    Desired transformation exponent
     * @return             Cached data or nullptr if not found
     */
    static std::shared_ptr<CachedGridData> get(const void* systemPtr,
                                                const std::string& filename,
                                                InvPowerMode mode,
                                                double inv_power);

    /**
     * Store grid data in cache.
     *
     * @param systemPtr    Pointer to System (for cache scoping)
     * @param filename     Grid file path
     * @param mode         Transformation mode
     * @param inv_power    Transformation exponent
     * @param data         Cached grid data to store
     */
    static void put(const void* systemPtr,
                    const std::string& filename,
                    InvPowerMode mode,
                    double inv_power,
                    std::shared_ptr<CachedGridData> data);

    /**
     * Clear all cached data for a specific System.
     * Called when System is destroyed.
     *
     * @param systemPtr    Pointer to System to clear
     */
    static void clearSystem(const void* systemPtr);

    /**
     * Clear entire cache (all Systems).
     * Useful for testing or long-running processes.
     */
    static void clearAll();

private:
    static std::map<GridCacheKey, std::shared_ptr<CachedGridData>>& getCache();
};

}  // namespace GridForcePlugin

#endif /*OPENMM_CACHED_GRID_DATA_H_*/
