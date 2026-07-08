/* -------------------------------------------------------------------------- *
 *                               OpenMMGridForce                              *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2014 Stanford University and the Authors.           *
 * Authors:                                                                   *
 * Contributors:                                                              *
 *                                                                            *
 * Permission is hereby granted, free of charge, to any person obtaining a    *
 * copy of this software and associated documentation files (the "Software"), *
 * to deal in the Software without restriction, including without limitation  *
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,   *
 * and/or sell copies of the Software, and to permit persons to whom the      *
 * Software is furnished to do so, subject to the following conditions:       *
 *                                                                            *
 * The above copyright notice and this permission notice shall be included in *
 * all copies or substantial portions of the Software.                        *
 *                                                                            *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR *
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,   *
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL    *
 * THE AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,    *
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR      *
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE  *
 * USE OR OTHER DEALINGS IN THE SOFTWARE.                                     *
 * -------------------------------------------------------------------------- */

#include "GridForce.h"
#include "GridForceKernels.h"
#include "internal/GridForceImpl.h"

#include "openmm/Force.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/AssertionUtilities.h"
#include "openmm/internal/ContextImpl.h"

#include <iostream>
#include <fstream>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <random>

using namespace OpenMM;
using namespace std;

namespace GridForcePlugin {

GridForce::GridForce() : m_inv_power(0.0), m_invPowerMode(InvPowerMode::NONE), m_gridCap(41840.0), m_runtimeCap(0.0), m_evaluateInVSpace(false), m_outOfBoundsRestraint(10000.0), m_hasEffectiveBounds(false), m_interpolationMethod(0),
                         m_bsplinePrefilterOrder(0), m_arcsinhScale(0.0), m_blurSigma(0.0),
                         m_adaptiveRegularization(0.0), m_regularizationThreshold(0.0),
                         m_prefilterPCGTolerance(1e-6), m_prefilterMaxIterations(200),
                         m_globalScalingFactor(1.0),
                         m_autoCalculateScalingFactors(false), m_scalingProperty(""),
                         m_autoGenerateGrid(false), m_gridType(""), m_gridOrigin({0.0, 0.0, 0.0}),
                         m_computeDerivatives(false), m_useDoubleStorage(false),
                         m_systemPtr(nullptr),
                         m_vals(std::make_shared<std::vector<double>>()),
                         m_derivatives(std::make_shared<std::vector<double>>()),
                         m_valuesPreTransformed(false),
                         m_tiledMode(false), m_tileSize(64), m_memoryBudgetMB(2048),
                         m_tiledOutputFile(""), m_tiledOutputTileSize(32) {
    //
}

GridForce::GridForce(std::shared_ptr<GridData> gridData)
    : m_inv_power(0.0), m_invPowerMode(InvPowerMode::NONE), m_gridCap(41840.0), m_runtimeCap(0.0),
      m_evaluateInVSpace(false),
      m_outOfBoundsRestraint(10000.0), m_interpolationMethod(0),
      m_bsplinePrefilterOrder(0), m_arcsinhScale(0.0), m_blurSigma(0.0),
      m_adaptiveRegularization(0.0), m_regularizationThreshold(0.0),
      m_prefilterPCGTolerance(1e-6), m_prefilterMaxIterations(200),
      m_globalScalingFactor(1.0),
      m_autoCalculateScalingFactors(false), m_scalingProperty(""),
      m_autoGenerateGrid(false), m_gridType(""), m_gridOrigin({0.0, 0.0, 0.0}),
      m_computeDerivatives(false), m_useDoubleStorage(false),
      m_systemPtr(nullptr),
      m_gridData(gridData),
      m_vals(std::make_shared<std::vector<double>>()),
      m_derivatives(std::make_shared<std::vector<double>>()),
      m_valuesPreTransformed(false),
      m_tiledMode(false), m_tileSize(64), m_memoryBudgetMB(2048),
                         m_tiledOutputFile(""), m_tiledOutputTileSize(32) {
    if (gridData) {
        // Extract grid data to populate legacy members
        m_counts.clear();
        m_counts.push_back(gridData->getNx());
        m_counts.push_back(gridData->getNy());
        m_counts.push_back(gridData->getNz());

        m_spacing.clear();
        m_spacing.push_back(gridData->getDx());
        m_spacing.push_back(gridData->getDy());
        m_spacing.push_back(gridData->getDz());

        double ox, oy, oz;
        gridData->getOrigin(ox, oy, oz);
        m_gridOrigin = {ox, oy, oz};

        // Share the underlying data via shared_ptr
        m_vals = gridData->getValuesPtr();
        m_derivatives = gridData->getDerivativesPtr();

        // Copy metadata
        m_gridType = gridData->getGridType();
        m_inv_power = gridData->getInvPower();

        // Determine inv_power mode based on metadata
        if (m_inv_power > 0.0) {
            m_invPowerMode = InvPowerMode::STORED;
        }
    }
}

void GridForce::setGridData(std::shared_ptr<GridData> gridData) {
    m_gridData = gridData;

    if (gridData) {
        // Extract grid data to update legacy members
        m_counts.clear();
        m_counts.push_back(gridData->getNx());
        m_counts.push_back(gridData->getNy());
        m_counts.push_back(gridData->getNz());

        m_spacing.clear();
        m_spacing.push_back(gridData->getDx());
        m_spacing.push_back(gridData->getDy());
        m_spacing.push_back(gridData->getDz());

        double ox, oy, oz;
        gridData->getOrigin(ox, oy, oz);
        m_gridOrigin = {ox, oy, oz};

        // Share the underlying data via shared_ptr
        m_vals = gridData->getValuesPtr();
        m_derivatives = gridData->getDerivativesPtr();

        // Copy metadata
        m_gridType = gridData->getGridType();
        m_inv_power = gridData->getInvPower();

        // Determine inv_power mode based on metadata
        if (m_inv_power > 0.0) {
            m_invPowerMode = InvPowerMode::STORED;
        }
    }
}

std::shared_ptr<GridData> GridForce::getGridData() const {
    return m_gridData;
}

std::shared_ptr<CachedGridData> GridForce::getCachedGridData() const {
    return m_cachedGridData;
}

void GridForce::setCachedGridData(std::shared_ptr<CachedGridData> cachedGridData) {
    m_cachedGridData = cachedGridData;
}

void GridForce::addGridCounts(int nx, int ny, int nz) {
    m_counts.push_back(nx);
    m_counts.push_back(ny);
    m_counts.push_back(nz);
}

void GridForce::addGridSpacing(double dx, double dy, double dz) {
    // the length unit is 'nm'
    m_spacing.push_back(dx);
    m_spacing.push_back(dy);
    m_spacing.push_back(dz);
}

void GridForce::addGridValue(double val) {
    m_vals->push_back(val);
}

void GridForce::setGridValues(const std::vector<double>& vals) {
    *m_vals = vals;
}

const std::vector<double>& GridForce::getGridValues() const {
    return *m_vals;
}

void GridForce::addScalingFactor(double val) {
    m_scaling_factors.push_back(val);
}

void GridForce::setScalingFactor(int index, double val) {
    if (index < 0 || index >= (int)m_scaling_factors.size()) {
        throw OpenMMException("GridForce::setScalingFactor: index " + std::to_string(index) +
            " out of range [0, " + std::to_string(m_scaling_factors.size()) +
            "). Use addScalingFactor() to add atoms first, or setAutoCalculateScalingFactors(true).");
    }
    m_scaling_factors[index] = val;
}

void GridForce::setScalingFactors(const std::vector<double>& vals) {
    m_scaling_factors = vals;
}

void GridForce::setGlobalScalingFactor(double factor) {
    m_globalScalingFactor = factor;
}

double GridForce::getGlobalScalingFactor() const {
    return m_globalScalingFactor;
}

void GridForce::setInvPowerMode(InvPowerMode mode, double inv_power) {
    // Validation
    if (mode != InvPowerMode::NONE && inv_power == 0.0) {
        throw OpenMMException("GridForce: inv_power must be non-zero when mode != NONE");
    }
    if (mode == InvPowerMode::NONE && inv_power != 0.0) {
        throw OpenMMException("GridForce: inv_power must be 0 when mode == NONE");
    }

    // Check for conflicting mode changes after grid is loaded
    if (m_vals && !m_vals->empty()) {
        if (m_invPowerMode == InvPowerMode::STORED && mode == InvPowerMode::RUNTIME) {
            throw OpenMMException(
                "GridForce: Cannot set RUNTIME mode on grid that already has STORED transformation. "
                "This would apply transformation twice!");
        }
        if (m_invPowerMode == InvPowerMode::RUNTIME && mode == InvPowerMode::STORED) {
            throw OpenMMException(
                "GridForce: Cannot set STORED mode on untransformed grid loaded with RUNTIME mode. "
                "Call applyInvPowerTransformation() first.");
        }
    }

    m_invPowerMode = mode;
    m_inv_power = inv_power;
}

InvPowerMode GridForce::getInvPowerMode() const {
    return m_invPowerMode;
}

void GridForce::applyInvPowerTransformation() {
    // Use CachedGridData if available
    if (m_cachedGridData) {
        // Let CachedGridData handle the transformation (includes validation)
        m_cachedGridData->applyTransformation(m_invPowerMode, m_inv_power, m_interpolationMethod);

        // Update our pointers to reflect the transformed data
        m_vals = m_cachedGridData->getCurrentValues();
        m_derivatives = m_cachedGridData->getCurrentDerivatives();

        // Update mode to match CachedGridData state
        InvPowerMode currentMode;
        double currentPower;
        m_cachedGridData->getCurrentTransformation(currentMode, currentPower);
        m_invPowerMode = currentMode;
        m_inv_power = currentPower;

        return;
    }

    // Fallback: Direct transformation (for grids loaded without System pointer)
    if (m_invPowerMode != InvPowerMode::RUNTIME) {
        throw OpenMMException(
            "GridForce: Can only call applyInvPowerTransformation() when mode == RUNTIME. "
            "Current mode: " + std::to_string(static_cast<int>(m_invPowerMode)));
    }

    if (m_inv_power == 0.0) {
        throw OpenMMException("GridForce: inv_power must be non-zero");
    }

    if (m_derivatives && !m_derivatives->empty()) {
        throw OpenMMException(
            "GridForce: Cannot apply runtime transformation to grids with analytical derivatives. "
            "Use mode STORED with pre-transformed grids instead.");
    }

    if (!m_vals || m_vals->empty()) {
        throw OpenMMException("GridForce: No grid values to transform. Load grid first.");
    }

    // Apply transformation: G -> sign(G) * |G|^(1/inv_power)
    for (size_t i = 0; i < m_vals->size(); ++i) {
        if ((*m_vals)[i] != 0.0) {
            double sign = ((*m_vals)[i] >= 0.0) ? 1.0 : -1.0;
            (*m_vals)[i] = sign * std::pow(std::abs((*m_vals)[i]), 1.0 / m_inv_power);
        }
    }

    // Update mode to STORED since grid is now transformed
    m_invPowerMode = InvPowerMode::STORED;
}

// setInvPower() REMOVED - Use setInvPowerMode() instead

double GridForce::getInvPower() const {
    return m_inv_power;
}

void GridForce::setSystemPointer(const void* systemPtr) {
    m_systemPtr = systemPtr;
}

const void* GridForce::getSystemPointer() const {
    return m_systemPtr;
}

void GridForce::setGridCap(double uMax) {
    m_gridCap = uMax;
}

double GridForce::getGridCap() const {
    return m_gridCap;
}

void GridForce::setRuntimeCap(double cap) {
    m_runtimeCap = cap;
}

double GridForce::getRuntimeCap() const {
    return m_runtimeCap;
}

void GridForce::setEvaluateInVSpace(bool enabled) {
    m_evaluateInVSpace = enabled;
}

bool GridForce::getEvaluateInVSpace() const {
    return m_evaluateInVSpace;
}

void GridForce::setOutOfBoundsRestraint(double k) {
    m_outOfBoundsRestraint = k;
}

double GridForce::getOutOfBoundsRestraint() const {
    return m_outOfBoundsRestraint;
}

void GridForce::setEffectiveBounds(double minX, double minY, double minZ,
                                    double maxX, double maxY, double maxZ) {
    m_effectiveBoundsMin = {minX, minY, minZ};
    m_effectiveBoundsMax = {maxX, maxY, maxZ};
    m_hasEffectiveBounds = true;
}

void GridForce::getEffectiveBounds(double& minX, double& minY, double& minZ,
                                    double& maxX, double& maxY, double& maxZ) const {
    if (m_hasEffectiveBounds) {
        minX = m_effectiveBoundsMin[0];
        minY = m_effectiveBoundsMin[1];
        minZ = m_effectiveBoundsMin[2];
        maxX = m_effectiveBoundsMax[0];
        maxY = m_effectiveBoundsMax[1];
        maxZ = m_effectiveBoundsMax[2];
    } else {
        // Return actual grid bounds
        minX = m_gridOrigin[0];
        minY = m_gridOrigin[1];
        minZ = m_gridOrigin[2];
        maxX = m_gridOrigin[0] + m_spacing[0] * (m_counts[0] - 1);
        maxY = m_gridOrigin[1] + m_spacing[1] * (m_counts[1] - 1);
        maxZ = m_gridOrigin[2] + m_spacing[2] * (m_counts[2] - 1);
    }
}

bool GridForce::hasEffectiveBounds() const {
    return m_hasEffectiveBounds;
}

void GridForce::clearEffectiveBounds() {
    m_hasEffectiveBounds = false;
    m_effectiveBoundsMin.clear();
    m_effectiveBoundsMax.clear();
}

void GridForce::setInterpolationMethod(int method) {
    if (method < 0 || method > 4) {
        throw OpenMMException("GridForce: Invalid interpolation method. Must be 0 (trilinear), 1 (cubic B-spline), 2 (tricubic Hermite), 3 (triquintic Hermite), or 4 (triquintic B-spline)");
    }
    m_interpolationMethod = method;
}

int GridForce::getInterpolationMethod() const {
    return m_interpolationMethod;
}

void GridForce::setBSplinePrefilterOrder(int order) {
    if (order != 0 && order != 3 && order != 5) {
        throw OpenMMException("GridForce: B-spline prefilter order must be 0 (none), 3 (cubic), or 5 (quintic)");
    }
    m_bsplinePrefilterOrder = order;
}

int GridForce::getBSplinePrefilterOrder() const {
    return m_bsplinePrefilterOrder;
}

void GridForce::setArcsinhScale(double scale) {
    if (scale < 0.0) {
        throw OpenMMException("GridForce: arcsinh scale must be >= 0.0 (0.0 = disabled)");
    }
    m_arcsinhScale = scale;
}

double GridForce::getArcsinhScale() const {
    return m_arcsinhScale;
}

void GridForce::setGaussianBlurSigma(double sigma) {
    if (sigma < 0.0) {
        throw OpenMMException("GridForce: Gaussian blur sigma must be >= 0.0 (0.0 = disabled)");
    }
    m_blurSigma = sigma;
}

double GridForce::getGaussianBlurSigma() const {
    return m_blurSigma;
}

void GridForce::setAdaptiveRegularization(double cReg) {
    if (cReg < 0.0) {
        throw OpenMMException("GridForce: adaptive regularization coefficient must be >= 0.0");
    }
    m_adaptiveRegularization = cReg;
}

double GridForce::getAdaptiveRegularization() const {
    return m_adaptiveRegularization;
}

void GridForce::setRegularizationThreshold(double threshold) {
    if (threshold < 0.0) {
        throw OpenMMException("GridForce: regularization threshold must be >= 0.0");
    }
    m_regularizationThreshold = threshold;
}

double GridForce::getRegularizationThreshold() const {
    return m_regularizationThreshold;
}

void GridForce::setPrefilterPCGTolerance(double tol) {
    if (tol <= 0.0) {
        throw OpenMMException("GridForce: PCG tolerance must be > 0.0");
    }
    m_prefilterPCGTolerance = tol;
}

double GridForce::getPrefilterPCGTolerance() const {
    return m_prefilterPCGTolerance;
}

void GridForce::setPrefilterMaxIterations(int maxIter) {
    if (maxIter < 1) {
        throw OpenMMException("GridForce: max iterations must be >= 1");
    }
    m_prefilterMaxIterations = maxIter;
}

int GridForce::getPrefilterMaxIterations() const {
    return m_prefilterMaxIterations;
}

void GridForce::setTiledMode(bool enable, int tileSize, int memoryBudgetMB) {
    if (tileSize < 8 || tileSize > 256) {
        throw OpenMMException("GridForce: Tile size must be between 8 and 256");
    }
    if (memoryBudgetMB < 64) {
        throw OpenMMException("GridForce: Memory budget must be at least 64 MB");
    }
    m_tiledMode = enable;
    m_tileSize = tileSize;
    m_memoryBudgetMB = memoryBudgetMB;
}

bool GridForce::getTiledMode() const {
    return m_tiledMode;
}

void GridForce::setTiledOutputFile(const std::string& filename, int tileSize) {
    if (tileSize < 8 || tileSize > 256) {
        throw OpenMMException("GridForce: Tile size must be between 8 and 256");
    }
    m_tiledOutputFile = filename;
    m_tiledOutputTileSize = tileSize;
}

void GridForce::setTiledInputFile(const std::string& filename) {
    m_tiledInputFile = filename;
    // Automatically enable tiled mode when using a tiled input file
    if (!filename.empty() && !m_tiledMode) {
        m_tiledMode = true;
    }
}

int GridForce::getTileSize() const {
    return m_tileSize;
}

int GridForce::getMemoryBudgetMB() const {
    return m_memoryBudgetMB;
}

void GridForce::getGridParameters(std::vector<int> &g_counts,
                                  std::vector<double> &g_spacing,
                                  std::vector<double> &g_vals,
                                  std::vector<double> &g_scaling_factors) const {
    g_counts = m_counts;
    g_spacing = m_spacing;
    g_vals = m_vals ? *m_vals : std::vector<double>();
    g_scaling_factors = m_scaling_factors;
}

ForceImpl *GridForce::createImpl() const {
    return new GridForceImpl(*this);
}

void GridForce::updateParametersInContext(Context &context) {
    dynamic_cast<GridForceImpl &>(getImplInContext(context)).updateParametersInContext(getContextImpl(context));
}

void GridForce::setAutoCalculateScalingFactors(bool enable) {
    m_autoCalculateScalingFactors = enable;
}

bool GridForce::getAutoCalculateScalingFactors() const {
    return m_autoCalculateScalingFactors;
}

void GridForce::setScalingProperty(const std::string& property) {
    // Don't validate here - validation happens in kernel initialization where exceptions are properly handled
    // This avoids SWIG exception translation issues
    m_scalingProperty = property;
}

const std::string& GridForce::getScalingProperty() const {
    return m_scalingProperty;
}

void GridForce::setAutoGenerateGrid(bool enable) {
    m_autoGenerateGrid = enable;
}

bool GridForce::getAutoGenerateGrid() const {
    return m_autoGenerateGrid;
}

void GridForce::setGridType(const std::string& type) {
    m_gridType = type;
}

const std::string& GridForce::getGridType() const {
    return m_gridType;
}

void GridForce::setGridOrigin(double x, double y, double z) {
    m_gridOrigin = {x, y, z};
}

void GridForce::getGridOrigin(double& x, double& y, double& z) const {
    if (m_gridOrigin.size() != 3) {
        x = y = z = 0.0;
    } else {
        x = m_gridOrigin[0];
        y = m_gridOrigin[1];
        z = m_gridOrigin[2];
    }
}

void GridForce::setParticles(const std::vector<int>& particles) {
    m_particles = particles;
}

const std::vector<int>& GridForce::getParticles() const {
    return m_particles;
}

void GridForce::clearGridData() {
    // NOTE: With shared_ptr implementation, grid data is shared across instances.
    // Clearing the shared data would affect all instances, so we make this a no-op.
    // The shared data will be automatically freed when the last instance is destroyed.
    // This preserves backward compatibility while enabling memory-efficient sharing.
}

void GridForce::setReceptorAtoms(const std::vector<int>& atomIndices) {
    m_receptorAtoms = atomIndices;
}

const std::vector<int>& GridForce::getReceptorAtoms() const {
    return m_receptorAtoms;
}

void GridForce::setLigandAtoms(const std::vector<int>& atomIndices) {
    m_ligandAtoms = atomIndices;
}

const std::vector<int>& GridForce::getLigandAtoms() const {
    return m_ligandAtoms;
}

void GridForce::setReceptorPositions(const std::vector<Vec3>& positions) {
    m_receptorPositions = positions;
}

void GridForce::setReceptorPositionsFromArrays(const std::vector<double>& x,
                                                const std::vector<double>& y,
                                                const std::vector<double>& z) {
    if (x.size() != y.size() || y.size() != z.size()) {
        throw OpenMMException("GridForce: x, y, z arrays must have the same size");
    }

    m_receptorPositions.clear();
    m_receptorPositions.reserve(x.size());
    for (size_t i = 0; i < x.size(); i++) {
        m_receptorPositions.push_back(Vec3(x[i], y[i], z[i]));
    }
}

const std::vector<Vec3>& GridForce::getReceptorPositions() const {
    return m_receptorPositions;
}

void GridForce::setComputeDerivatives(bool compute) {
    m_computeDerivatives = compute;
}

bool GridForce::getComputeDerivatives() const {
    return m_computeDerivatives;
}

void GridForce::setUseDoubleStorage(bool useDouble) {
    m_useDoubleStorage = useDouble;
}

bool GridForce::getUseDoubleStorage() const {
    return m_useDoubleStorage;
}

bool GridForce::hasDerivatives() const {
    return m_derivatives && !m_derivatives->empty();
}

const std::vector<double>& GridForce::getDerivatives() const {
    static const std::vector<double> empty;
    return m_derivatives ? *m_derivatives : empty;
}

void GridForce::setDerivatives(const std::vector<double>& derivs) {
    *m_derivatives = derivs;
}

void GridForce::loadFromFile(const std::string& filename) {
    // Try to get from per-System cache first
    if (m_systemPtr != nullptr) {
        auto cached = GridDataCache::get(m_systemPtr, filename, m_invPowerMode, m_inv_power);
        if (cached) {
            // Cache hit! Reuse the cached grid data
            m_cachedGridData = cached;

            // Extract metadata and current values
            m_counts.clear();
            const auto& counts = cached->getCounts();
            m_counts = counts;

            m_spacing.clear();
            const auto& spacing = cached->getSpacing();
            m_spacing = spacing;

            double ox, oy, oz;
            cached->getOrigin(ox, oy, oz);
            m_gridOrigin = {ox, oy, oz};

            // Share the current values and derivatives
            m_vals = cached->getCurrentValues();
            m_derivatives = cached->getCurrentDerivatives();

            return;
        }
    }

    // Binary file I/O implementation
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw OpenMMException("GridForce: Cannot open file '" + filename + "'");
    }

    // Read and validate magic number
    char magic[8];
    file.read(magic, 8);
    if (std::string(magic, 8) != std::string("OMGRID\0\0", 8)) {
        file.close();
        throw OpenMMException("GridForce: Invalid file format (bad magic number)");
    }

    // Read header
    uint32_t version, header_size;
    file.read(reinterpret_cast<char*>(&version), sizeof(uint32_t));
    file.read(reinterpret_cast<char*>(&header_size), sizeof(uint32_t));

    // Support V3 (legacy) and V4 (adds valuesPreTransformed flag in flags bit 0).
    if (version != 3 && version != 4) {
        file.close();
        throw OpenMMException(
            "GridForce: Only V3/V4 grid files are supported. "
            "Found version " + std::to_string(version) + ". "
            "Please regenerate your grid files using the current version of GridForce.");
    }

    // Read grid counts
    int32_t nx, ny, nz;
    file.read(reinterpret_cast<char*>(&nx), sizeof(int32_t));
    file.read(reinterpret_cast<char*>(&ny), sizeof(int32_t));
    file.read(reinterpret_cast<char*>(&nz), sizeof(int32_t));

    // Read derivative count (V3)
    uint32_t deriv_count = 0;
    file.read(reinterpret_cast<char*>(&deriv_count), sizeof(uint32_t));

    // Read grid spacing
    double dx, dy, dz;
    file.read(reinterpret_cast<char*>(&dx), sizeof(double));
    file.read(reinterpret_cast<char*>(&dy), sizeof(double));
    file.read(reinterpret_cast<char*>(&dz), sizeof(double));

    // Read data offset
    uint64_t data_offset;
    file.read(reinterpret_cast<char*>(&data_offset), sizeof(uint64_t));

    // Read metadata
    double origin_x, origin_y, origin_z;
    file.read(reinterpret_cast<char*>(&origin_x), sizeof(double));
    file.read(reinterpret_cast<char*>(&origin_y), sizeof(double));
    file.read(reinterpret_cast<char*>(&origin_z), sizeof(double));

    uint32_t grid_type_code, flags;
    file.read(reinterpret_cast<char*>(&grid_type_code), sizeof(uint32_t));
    file.read(reinterpret_cast<char*>(&flags), sizeof(uint32_t));

    double inv_power;
    file.read(reinterpret_cast<char*>(&inv_power), sizeof(double));

    // Read inv_power_mode (V3)
    uint32_t mode_value = 0;
    file.read(reinterpret_cast<char*>(&mode_value), sizeof(uint32_t));

    // V4: first 8 bytes of the reserved trailer store m_arcsinhScale so
    // the file carries the compression scale it was generated with.
    double saved_arcsinhScale = 0.0;
    if (version == 4)
        file.read(reinterpret_cast<char*>(&saved_arcsinhScale), sizeof(double));

    // Seek to data
    file.seekg(data_offset);

    // Read grid data
    size_t num_points = static_cast<size_t>(nx) * ny * nz;

    if (deriv_count > 0) {
        // V3 with derivatives: Read all derivatives [deriv_count, nx, ny, nz]
        size_t total_values = deriv_count * num_points;
        m_derivatives->resize(total_values);
        file.read(reinterpret_cast<char*>(m_derivatives->data()), total_values * sizeof(double));

        // Extract function values (first derivative index = 0)
        m_vals->resize(num_points);
        for (size_t i = 0; i < num_points; ++i) {
            (*m_vals)[i] = (*m_derivatives)[i];  // f(x,y,z) is at offset i in [27, nx, ny, nz] layout
        }
    } else {
        // V3 without derivatives in header: Read only function values
        m_vals->resize(num_points);
        file.read(reinterpret_cast<char*>(m_vals->data()), num_points * sizeof(double));
        m_derivatives->clear();

        // Skip scaling factors (for compatibility)
        int numScalingFactors;
        file.read(reinterpret_cast<char*>(&numScalingFactors), sizeof(int));
        if (numScalingFactors > 0) {
            vector<double> scalingFactors(numScalingFactors);
            file.read(reinterpret_cast<char*>(scalingFactors.data()), numScalingFactors * sizeof(double));
        }

        // Skip origin (written again for V3 compatibility)
        double origin[3];
        file.read(reinterpret_cast<char*>(origin), 3 * sizeof(double));

        // Check for optional DERIVS block (GridData format)
        file.peek();
        if (!file.eof()) {
            char derivHeader[8];
            file.read(derivHeader, 8);
            if (strncmp(derivHeader, "DERIVS", 6) == 0) {
                int numDerivs = (static_cast<unsigned char>(derivHeader[6]) << 8) |
                               static_cast<unsigned char>(derivHeader[7]);
                size_t derivSize = numDerivs * num_points;
                m_derivatives->resize(derivSize);
                file.read(reinterpret_cast<char*>(m_derivatives->data()),
                         derivSize * sizeof(double));
            }
        }
    }

    file.close();

    // Set the grid parameters
    m_counts.clear();
    m_spacing.clear();

    addGridCounts(nx, ny, nz);
    addGridSpacing(dx, dy, dz);

    setGridOrigin(origin_x, origin_y, origin_z);
    // Restore inv_power directly without transformation (grid already has correct values)
    m_inv_power = inv_power;

    // Restore inv_power_mode (V3)
    // Validate mode value
    if (mode_value > 2) {
        throw OpenMMException("GridForce: Invalid inv_power_mode value in file: " + std::to_string(mode_value));
    }
    m_invPowerMode = static_cast<InvPowerMode>(mode_value);

    // Additional validation
    if (m_invPowerMode != InvPowerMode::NONE && inv_power == 0.0) {
        throw OpenMMException("GridForce: File has inv_power_mode enabled but invalid inv_power value");
    }

    // Decode grid type
    if (grid_type_code == 1) m_gridType = "charge";
    else if (grid_type_code == 2) m_gridType = "ljr";
    else if (grid_type_code == 3) m_gridType = "lja";
    else m_gridType = "";

    // Create CachedGridData and populate the per-System cache
    if (m_systemPtr != nullptr) {
        m_cachedGridData = std::make_shared<CachedGridData>(
            *m_vals,                    // original values
            *m_derivatives,             // original derivatives (empty if none)
            m_counts,                   // grid dimensions
            m_spacing,                  // grid spacing
            origin_x, origin_y, origin_z  // origin
        );

        // If mode is STORED and inv_power > 0, the grid file already has transformed values
        // The CachedGridData stores these as "original", which is correct for STORED mode
        // For RUNTIME mode, applyInvPowerTransformation() will be called later

        // Store in cache
        GridDataCache::put(m_systemPtr, filename, m_invPowerMode, m_inv_power, m_cachedGridData);

        // Update our pointers to use the cached data
        m_vals = m_cachedGridData->getCurrentValues();
        m_derivatives = m_cachedGridData->getCurrentDerivatives();
    }

    // V4: restore the saved pre-transformed flag from flags bit 0.
    // V3 files predate the flag; assume pre-transformed=true (matches the
    // pre-V4 unconditional behavior, correct for auto-generated grids
    // which were the only supported save path).
    if (version == 4) {
        m_valuesPreTransformed = (flags & 0x1u) != 0;
        m_arcsinhScale = saved_arcsinhScale;
    } else {
        m_valuesPreTransformed = true;
    }
}

void GridForce::saveToFile(const std::string& filename) const {
    // If we have a GridData object, use its save method (new format)
    if (m_gridData) {
        m_gridData->saveToFile(filename);
        return;
    }

    // Otherwise, use legacy format for backward compatibility
    if (m_counts.size() != 3 || m_spacing.size() != 3) {
        throw OpenMMException("GridForce: Grid dimensions must be set before saving");
    }

    size_t expected_values = static_cast<size_t>(m_counts[0]) * m_counts[1] * m_counts[2];
    if (!m_vals || m_vals->size() != expected_values) {
        throw OpenMMException("GridForce: Number of grid values doesn't match dimensions");
    }

    bool hasDerivs = m_derivatives && !m_derivatives->empty();
    uint32_t deriv_count = hasDerivs ? 27 : 0;

    if (hasDerivs && m_derivatives->size() != 27 * expected_values) {
        throw OpenMMException("GridForce: Number of derivative values doesn't match dimensions (expected 27 * nx * ny * nz)");
    }

    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw OpenMMException("GridForce: Cannot create file '" + filename + "'");
    }

    // Write magic number ("OMGRID" padded with nulls to 8 bytes)
    const char magic[8] = {'O', 'M', 'G', 'R', 'I', 'D', '\0', '\0'};
    file.write(magic, 8);

    // Write header (Version 4 format).  V4 preserves m_valuesPreTransformed
    // through the flags field so save/load is self-consistent for both auto-
    // generated (values already transformed) and user-supplied (raw) grids.
    // V3 readers assumed values were always pre-transformed; V4 restores the
    // actual saved state.
    uint32_t version = 4;
    uint32_t header_size = 128;  // Fixed size for V3/V4
    file.write(reinterpret_cast<const char*>(&version), sizeof(uint32_t));
    file.write(reinterpret_cast<const char*>(&header_size), sizeof(uint32_t));

    // Write grid counts
    int32_t nx = m_counts[0], ny = m_counts[1], nz = m_counts[2];
    file.write(reinterpret_cast<const char*>(&nx), sizeof(int32_t));
    file.write(reinterpret_cast<const char*>(&ny), sizeof(int32_t));
    file.write(reinterpret_cast<const char*>(&nz), sizeof(int32_t));

    // Write deriv_count (0 if no derivatives)
    file.write(reinterpret_cast<const char*>(&deriv_count), sizeof(uint32_t));

    // Write grid spacing
    double dx = m_spacing[0], dy = m_spacing[1], dz = m_spacing[2];
    file.write(reinterpret_cast<const char*>(&dx), sizeof(double));
    file.write(reinterpret_cast<const char*>(&dy), sizeof(double));
    file.write(reinterpret_cast<const char*>(&dz), sizeof(double));

    // Write data offset (V3: 128-byte header)
    uint64_t data_offset = 128;
    file.write(reinterpret_cast<const char*>(&data_offset), sizeof(uint64_t));

    // Write metadata (origin)
    double origin_x = 0.0, origin_y = 0.0, origin_z = 0.0;
    if (m_gridOrigin.size() == 3) {
        origin_x = m_gridOrigin[0];
        origin_y = m_gridOrigin[1];
        origin_z = m_gridOrigin[2];
    }
    file.write(reinterpret_cast<const char*>(&origin_x), sizeof(double));
    file.write(reinterpret_cast<const char*>(&origin_y), sizeof(double));
    file.write(reinterpret_cast<const char*>(&origin_z), sizeof(double));

    // Write grid type
    uint32_t grid_type_code = 0;
    if (m_gridType == "charge") grid_type_code = 1;
    else if (m_gridType == "ljr") grid_type_code = 2;
    else if (m_gridType == "lja") grid_type_code = 3;
    file.write(reinterpret_cast<const char*>(&grid_type_code), sizeof(uint32_t));

    // Write flags.  Bit 0 = m_valuesPreTransformed (V4).
    uint32_t flags = 0;
    if (m_valuesPreTransformed) flags |= 0x1u;
    file.write(reinterpret_cast<const char*>(&flags), sizeof(uint32_t));

    // Write inv_power
    file.write(reinterpret_cast<const char*>(&m_inv_power), sizeof(double));

    // Write inv_power_mode (Version 3)
    uint32_t mode_value = static_cast<uint32_t>(m_invPowerMode);
    file.write(reinterpret_cast<const char*>(&mode_value), sizeof(uint32_t));

    // V4 uses the first 8 bytes of the trailing reserved region for
    // m_arcsinhScale so the file remains self-describing (the eval-time
    // sinh inverse-transform needs the same scale the values were
    // compressed with at generation).  Remaining 12 bytes stay reserved.
    double arcsinhScale = m_arcsinhScale;
    file.write(reinterpret_cast<const char*>(&arcsinhScale), sizeof(double));
    char reserved[12] = {0};
    file.write(reserved, 12);

    // Write grid data
    if (hasDerivs) {
        // Version 3 with derivatives: Write all derivatives [27, nx, ny, nz]
        file.write(reinterpret_cast<const char*>(m_derivatives->data()), m_derivatives->size() * sizeof(double));
    } else {
        // Version 3 without derivatives: Write only function values
        file.write(reinterpret_cast<const char*>(m_vals->data()), m_vals->size() * sizeof(double));
    }

    file.close();
}

// Particle Group Management

int GridForce::addParticleGroup(const std::string& name,
                                 const std::vector<int>& particleIndices,
                                 const std::vector<double>& scalingFactors) {
    // Check if group name already exists
    for (const auto& group : m_particleGroups) {
        if (group.name == name) {
            throw OpenMMException("Particle group '" + name + "' already exists");
        }
    }

    // Add the new group
    m_particleGroups.emplace_back(name, particleIndices, scalingFactors);
    return m_particleGroups.size() - 1;
}

int GridForce::getNumParticleGroups() const {
    return m_particleGroups.size();
}

const ParticleGroup& GridForce::getParticleGroup(int index) const {
    if (index < 0 || index >= (int)m_particleGroups.size()) {
        throw OpenMMException("Particle group index out of range");
    }
    return m_particleGroups[index];
}

const ParticleGroup* GridForce::getParticleGroupByName(const std::string& name) const {
    for (const auto& group : m_particleGroups) {
        if (group.name == name) {
            return &group;
        }
    }
    return nullptr;  // Group not found
}

void GridForce::removeParticleGroup(int index) {
    if (index < 0 || index >= (int)m_particleGroups.size()) {
        throw OpenMMException("Particle group index out of range");
    }
    m_particleGroups.erase(m_particleGroups.begin() + index);
}

void GridForce::clearParticleGroups() {
    m_particleGroups.clear();
}

void GridForce::setParticleGroupScalingFactor(int groupIndex, double factor) {
    if (groupIndex < 0 || groupIndex >= (int)m_particleGroups.size()) {
        throw OpenMMException("Particle group index out of range");
    }
    m_particleGroups[groupIndex].groupScalingFactor = factor;
}

double GridForce::getParticleGroupScalingFactor(int groupIndex) const {
    if (groupIndex < 0 || groupIndex >= (int)m_particleGroups.size()) {
        throw OpenMMException("Particle group index out of range");
    }
    return m_particleGroups[groupIndex].groupScalingFactor;
}

void GridForce::setParticleGroupPositions(Context& context, int groupIndex,
                                            const std::vector<Vec3>& positions) const {
    if (groupIndex < 0 || groupIndex >= (int)m_particleGroups.size()) {
        throw OpenMMException("Particle group index out of range");
    }
    const ParticleGroup& group = m_particleGroups[groupIndex];
    if (positions.size() != group.particleIndices.size()) {
        throw OpenMMException("Number of positions (" + std::to_string(positions.size()) +
                              ") doesn't match group size (" +
                              std::to_string(group.particleIndices.size()) + ")");
    }

    // Get current state to read all positions
    State state = context.getState(State::Positions);
    std::vector<Vec3> allPositions = state.getPositions();

    // Update only the positions for this group's particles
    for (size_t i = 0; i < group.particleIndices.size(); i++) {
        allPositions[group.particleIndices[i]] = positions[i];
    }

    context.setPositions(allPositions);
}

std::vector<Vec3> GridForce::getParticleGroupPositions(Context& context, int groupIndex) const {
    if (groupIndex < 0 || groupIndex >= (int)m_particleGroups.size()) {
        throw OpenMMException("Particle group index out of range");
    }
    const ParticleGroup& group = m_particleGroups[groupIndex];

    State state = context.getState(State::Positions);
    const std::vector<Vec3>& allPositions = state.getPositions();

    std::vector<Vec3> groupPositions(group.particleIndices.size());
    for (size_t i = 0; i < group.particleIndices.size(); i++) {
        groupPositions[i] = allPositions[group.particleIndices[i]];
    }

    return groupPositions;
}

void GridForce::swapParticleGroupPositions(Context& context, int group1, int group2) const {
    if (group1 < 0 || group1 >= (int)m_particleGroups.size()) {
        throw OpenMMException("Particle group index 1 out of range");
    }
    if (group2 < 0 || group2 >= (int)m_particleGroups.size()) {
        throw OpenMMException("Particle group index 2 out of range");
    }
    const ParticleGroup& g1 = m_particleGroups[group1];
    const ParticleGroup& g2 = m_particleGroups[group2];
    if (g1.particleIndices.size() != g2.particleIndices.size()) {
        throw OpenMMException("Cannot swap positions between groups of different sizes");
    }

    State state = context.getState(State::Positions);
    std::vector<Vec3> allPositions = state.getPositions();

    // Swap positions for corresponding particles
    for (size_t i = 0; i < g1.particleIndices.size(); i++) {
        std::swap(allPositions[g1.particleIndices[i]],
                  allPositions[g2.particleIndices[i]]);
    }

    context.setPositions(allPositions);
}

void GridForce::setParticleGroupVelocities(Context& context, int groupIndex,
                                            const std::vector<Vec3>& velocities) const {
    if (groupIndex < 0 || groupIndex >= (int)m_particleGroups.size()) {
        throw OpenMMException("Particle group index out of range");
    }
    const ParticleGroup& group = m_particleGroups[groupIndex];
    if (velocities.size() != group.particleIndices.size()) {
        throw OpenMMException("Number of velocities (" + std::to_string(velocities.size()) +
                              ") doesn't match group size (" +
                              std::to_string(group.particleIndices.size()) + ")");
    }

    State state = context.getState(State::Velocities);
    std::vector<Vec3> allVelocities = state.getVelocities();

    for (size_t i = 0; i < group.particleIndices.size(); i++) {
        allVelocities[group.particleIndices[i]] = velocities[i];
    }

    context.setVelocities(allVelocities);
}

std::vector<Vec3> GridForce::getParticleGroupVelocities(Context& context, int groupIndex) const {
    if (groupIndex < 0 || groupIndex >= (int)m_particleGroups.size()) {
        throw OpenMMException("Particle group index out of range");
    }
    const ParticleGroup& group = m_particleGroups[groupIndex];

    State state = context.getState(State::Velocities);
    const std::vector<Vec3>& allVelocities = state.getVelocities();

    std::vector<Vec3> groupVelocities(group.particleIndices.size());
    for (size_t i = 0; i < group.particleIndices.size(); i++) {
        groupVelocities[i] = allVelocities[group.particleIndices[i]];
    }

    return groupVelocities;
}

void GridForce::setParticleGroupPositionsFlat(Context& context, int groupIndex,
                                                const std::vector<double>& coords) const {
    if (groupIndex < 0 || groupIndex >= (int)m_particleGroups.size()) {
        throw OpenMMException("Particle group index out of range");
    }
    const ParticleGroup& group = m_particleGroups[groupIndex];
    if (coords.size() != 3 * group.particleIndices.size()) {
        throw OpenMMException("Flat coordinate array size (" + std::to_string(coords.size()) +
                              ") must be 3 * group size (" +
                              std::to_string(3 * group.particleIndices.size()) + ")");
    }
    std::vector<Vec3> positions(group.particleIndices.size());
    for (size_t i = 0; i < group.particleIndices.size(); i++) {
        positions[i] = Vec3(coords[3*i], coords[3*i+1], coords[3*i+2]);
    }
    setParticleGroupPositions(context, groupIndex, positions);
}

std::vector<double> GridForce::getParticleGroupPositionsFlat(Context& context, int groupIndex) const {
    std::vector<Vec3> positions = getParticleGroupPositions(context, groupIndex);
    std::vector<double> coords(3 * positions.size());
    for (size_t i = 0; i < positions.size(); i++) {
        coords[3*i]   = positions[i][0];
        coords[3*i+1] = positions[i][1];
        coords[3*i+2] = positions[i][2];
    }
    return coords;
}

void GridForce::setParticleGroupVelocitiesFlat(Context& context, int groupIndex,
                                                 const std::vector<double>& vels) const {
    if (groupIndex < 0 || groupIndex >= (int)m_particleGroups.size()) {
        throw OpenMMException("Particle group index out of range");
    }
    const ParticleGroup& group = m_particleGroups[groupIndex];
    if (vels.size() != 3 * group.particleIndices.size()) {
        throw OpenMMException("Flat velocity array size (" + std::to_string(vels.size()) +
                              ") must be 3 * group size (" +
                              std::to_string(3 * group.particleIndices.size()) + ")");
    }
    std::vector<Vec3> velocities(group.particleIndices.size());
    for (size_t i = 0; i < group.particleIndices.size(); i++) {
        velocities[i] = Vec3(vels[3*i], vels[3*i+1], vels[3*i+2]);
    }
    setParticleGroupVelocities(context, groupIndex, velocities);
}

std::vector<double> GridForce::getParticleGroupVelocitiesFlat(Context& context, int groupIndex) const {
    std::vector<Vec3> velocities = getParticleGroupVelocities(context, groupIndex);
    std::vector<double> vels(3 * velocities.size());
    for (size_t i = 0; i < velocities.size(); i++) {
        vels[3*i]   = velocities[i][0];
        vels[3*i+1] = velocities[i][1];
        vels[3*i+2] = velocities[i][2];
    }
    return vels;
}

vector<double> GridForce::getParticleGroupEnergies(Context& context) const {
    return dynamic_cast<GridForceImpl&>(getImplInContext(context)).getParticleGroupEnergies();
}

vector<double> GridForce::getParticleGroupUnscaledEnergies(Context& context) const {
    return dynamic_cast<GridForceImpl&>(getImplInContext(context)).getParticleGroupUnscaledEnergies();
}

vector<double> GridForce::getParticleAtomEnergies(Context& context) const {
    return dynamic_cast<GridForceImpl&>(getImplInContext(context)).getParticleAtomEnergies();
}

vector<int> GridForce::getParticleOutOfBoundsFlags(Context& context) const {
    return dynamic_cast<GridForceImpl&>(getImplInContext(context)).getParticleOutOfBoundsFlags();
}

// =========================================================================
// Batch HMC operations
// =========================================================================

void GridForce::drawAndSetGroupVelocities(Context& context,
                                           const vector<double>& temperatures,
                                           const vector<double>& masses,
                                           unsigned int seed) const {
    int numGroups = getNumParticleGroups();
    if ((int)temperatures.size() != numGroups)
        throw OpenMMException("drawAndSetGroupVelocities: temperatures size must equal numGroups");

    // Get current velocities (one GPU→CPU transfer)
    State state = context.getState(State::Velocities);
    vector<Vec3> velocities = state.getVelocities();

    // Set up random number generator
    std::mt19937 rng;
    if (seed == 0) {
        std::random_device rd;
        rng.seed(rd());
    } else {
        rng.seed(seed);
    }

    // kB in kJ/(mol·K) — OpenMM internal units
    const double kB = 8.3144621e-3;

    // Draw MB velocities for each group
    for (int k = 0; k < numGroups; k++) {
        const auto& group = getParticleGroup(k);
        const vector<int>& indices = group.particleIndices;
        double T = temperatures[k];

        for (size_t i = 0; i < indices.size(); i++) {
            int atomIdx = indices[i];
            // Use per-atom mass from the template (assumes all groups have same atom layout)
            double mass = (i < masses.size()) ? masses[i] : masses[0];
            if (mass <= 0.0) continue;
            double sigma = sqrt(kB * T / mass);  // sqrt(kB*T/m) in nm/ps
            std::normal_distribution<double> dist(0.0, sigma);
            velocities[atomIdx] = Vec3(dist(rng), dist(rng), dist(rng));
        }
    }

    // Set all velocities at once (one CPU→GPU transfer)
    context.setVelocities(velocities);
}

vector<double> GridForce::computeGroupKineticEnergies(Context& context,
                                                       const vector<double>& masses) const {
    int numGroups = getNumParticleGroups();
    vector<double> ke(numGroups, 0.0);

    // Get velocities (one GPU→CPU transfer)
    State state = context.getState(State::Velocities);
    vector<Vec3> velocities = state.getVelocities();

    for (int k = 0; k < numGroups; k++) {
        const auto& group = getParticleGroup(k);
        const vector<int>& indices = group.particleIndices;
        double groupKE = 0.0;
        for (size_t i = 0; i < indices.size(); i++) {
            int atomIdx = indices[i];
            double mass = (i < masses.size()) ? masses[i] : masses[0];
            if (mass <= 0.0) continue;
            Vec3 v = velocities[atomIdx];
            groupKE += 0.5 * mass * (v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
        }
        ke[k] = groupKE;
    }

    return ke;
}

vector<int> GridForce::acceptRejectGroups(Context& context,
                                           const vector<double>& positionsBackup,
                                           const vector<double>& pe_old,
                                           const vector<double>& pe_new,
                                           const vector<double>& ke_old,
                                           const vector<double>& ke_new,
                                           const vector<double>& temperatures,
                                           unsigned int seed) const {
    int numGroups = getNumParticleGroups();
    if ((int)pe_old.size() != numGroups || (int)pe_new.size() != numGroups ||
        (int)ke_old.size() != numGroups || (int)ke_new.size() != numGroups ||
        (int)temperatures.size() != numGroups)
        throw OpenMMException("acceptRejectGroups: all vectors must have length numGroups");

    const double kB = 8.3144621e-3;  // kJ/(mol·K)

    // Set up RNG
    std::mt19937 rng;
    if (seed == 0) {
        std::random_device rd;
        rng.seed(rd());
    } else {
        rng.seed(seed);
    }
    std::uniform_real_distribution<double> uniform(0.0, 1.0);

    // Evaluate accept/reject for each group
    vector<int> accepted(numGroups, 0);
    bool anyRejected = false;

    for (int k = 0; k < numGroups; k++) {
        double dE = (pe_new[k] + ke_new[k]) - (pe_old[k] + ke_old[k]);
        double RT = kB * temperatures[k];
        double dE_over_RT = dE / RT;

        if (dE_over_RT < 0.0 || (fabs(dE_over_RT) < 250.0 && uniform(rng) < exp(-dE_over_RT))) {
            accepted[k] = 1;  // Accept
        } else {
            accepted[k] = 0;  // Reject
            anyRejected = true;
        }
    }

    // If any groups were rejected, restore their positions from backup
    if (anyRejected) {
        State state = context.getState(State::Positions);
        vector<Vec3> positions = state.getPositions();

        for (int k = 0; k < numGroups; k++) {
            if (accepted[k] == 0) {
                const auto& group = getParticleGroup(k);
                const vector<int>& indices = group.particleIndices;
                for (size_t i = 0; i < indices.size(); i++) {
                    int atomIdx = indices[i];
                    // positionsBackup is flat [K * N * 3]
                    size_t offset = k * indices.size() * 3 + i * 3;
                    positions[atomIdx] = Vec3(positionsBackup[offset],
                                              positionsBackup[offset + 1],
                                              positionsBackup[offset + 2]);
                }
            }
        }

        context.setPositions(positions);
    }

    return accepted;
}

void GridForce::setAllParticleGroupScalingFactors(const vector<double>& factors) {
    int numGroups = getNumParticleGroups();
    if ((int)factors.size() != numGroups)
        throw OpenMMException("setAllParticleGroupScalingFactors: factors size must equal numGroups");
    for (int k = 0; k < numGroups; k++) {
        setParticleGroupScalingFactor(k, factors[k]);
    }
}

void GridForce::setParticleGroupRuntimeCap(int groupIndex, double cap) {
    if (groupIndex < 0 || groupIndex >= (int)m_particleGroups.size()) {
        throw OpenMMException("Particle group index out of range");
    }
    m_particleGroups[groupIndex].groupRuntimeCap = cap;
}

double GridForce::getParticleGroupRuntimeCap(int groupIndex) const {
    if (groupIndex < 0 || groupIndex >= (int)m_particleGroups.size()) {
        throw OpenMMException("Particle group index out of range");
    }
    return m_particleGroups[groupIndex].groupRuntimeCap;
}

void GridForce::setAllParticleGroupRuntimeCaps(const vector<double>& caps) {
    int numGroups = getNumParticleGroups();
    if ((int)caps.size() != numGroups)
        throw OpenMMException("setAllParticleGroupRuntimeCaps: caps size must equal numGroups");
    for (int k = 0; k < numGroups; k++) {
        setParticleGroupRuntimeCap(k, caps[k]);
    }
}

vector<double> GridForce::getAllParticleGroupRuntimeCaps() const {
    int numGroups = getNumParticleGroups();
    vector<double> caps(numGroups);
    for (int k = 0; k < numGroups; k++) {
        caps[k] = m_particleGroups[k].groupRuntimeCap;
    }
    return caps;
}

vector<double> GridForce::getParticleGroupAtomRawEnergies(Context& context) const {
    return dynamic_cast<GridForceImpl&>(getImplInContext(context)).getParticleGroupAtomRawEnergies();
}

void GridForce::computeHessian(Context& context) const {
    dynamic_cast<GridForceImpl&>(getImplInContext(context)).computeHessian();
}

vector<double> GridForce::getHessianBlocks(Context& context) const {
    return dynamic_cast<GridForceImpl&>(getImplInContext(context)).getHessianBlocks();
}

void GridForce::computeThirdDerivatives(Context& context) const {
    dynamic_cast<GridForceImpl&>(getImplInContext(context)).computeThirdDerivatives();
}

vector<double> GridForce::getThirdDerivativeBlocks(Context& context) const {
    return dynamic_cast<GridForceImpl&>(getImplInContext(context)).getThirdDerivativeBlocks();
}

HessianAnalysis GridForce::analyzeHessian(Context& context, float temperature) const {
    HessianAnalysis result;
    GridForceImpl& impl = dynamic_cast<GridForceImpl&>(getImplInContext(context));

    // Compute Hessian first if not already done
    impl.computeHessian();

    // Perform analysis
    impl.analyzeHessian(temperature);

    // Collect results
    result.eigenvalues = impl.getEigenvalues();
    result.eigenvectors = impl.getEigenvectors();
    result.meanCurvature = impl.getMeanCurvature();
    result.totalCurvature = impl.getTotalCurvature();
    result.gaussianCurvature = impl.getGaussianCurvature();
    result.fracAnisotropy = impl.getFracAnisotropy();
    result.entropy = impl.getEntropy();
    result.minEigenvalue = impl.getMinEigenvalue();
    result.numNegative = impl.getNumNegative();
    result.totalEntropy = impl.getTotalEntropy();

    return result;
}

}  // namespace GridForcePlugin
