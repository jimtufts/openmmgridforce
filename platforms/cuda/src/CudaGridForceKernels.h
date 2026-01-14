#ifndef CUDA_GRIDFORCE_KERNELS_H_
#define CUDA_GRIDFORCE_KERNELS_H_

#include "GridForceKernels.h"
#include "openmm/cuda/CudaContext.h"
#include "openmm/cuda/CudaArray.h"
#include "openmm/NonbondedForce.h"
#include "TileManager.h"
#include "TiledGridData.h"

namespace GridForcePlugin {

/**
 * This kernel is invoked by GridForce to calculate the forces acting on the system and the energy of the system.
 */
class CudaCalcGridForceKernel : public CalcGridForceKernel {
public:
    CudaCalcGridForceKernel(std::string name, const OpenMM::Platform& platform, OpenMM::CudaContext& cu) :
            CalcGridForceKernel(name, platform), hasInitializedKernel(false), cu(cu) {
    }
    ~CudaCalcGridForceKernel();
    /**
     * Initialize the kernel.
     *
     * @param system     the System this kernel will be applied to
     * @param force      the GridForce this kernel will be used for
     */
    void initialize(const OpenMM::System& system, const GridForce& force);
    /**
     * Execute the kernel to calculate the forces and/or energy.
     *
     * @param context        the context in which to execute this kernel
     * @param includeForces  true if forces should be calculated
     * @param includeEnergy  true if the energy should be calculated
     * @return the potential energy due to the force
     */
    double execute(OpenMM::ContextImpl& context, bool includeForces, bool includeEnergy);
    /**
     * Copy changed parameters over to a context.
     *
     * @param context    the context to copy parameters to
     * @param force      the GridForce to copy the parameters from
     */
    void copyParametersToContext(OpenMM::ContextImpl& context, const GridForce& force);
    /**
     * Get per-particle-group energies.
     *
     * @return vector of energies, one per particle group (empty if no groups)
     */
    std::vector<double> getParticleGroupEnergies();
    /**
     * Get per-atom energies for particles in groups.
     *
     * @return vector of energies, one per particle across all groups
     *         (in the same order as particles were added to groups)
     */
    std::vector<double> getParticleAtomEnergies();
private:
    /**
     * Generate grid values from receptor atoms and NonbondedForce parameters.
     */
    void generateGrid(const OpenMM::System& system,
                     const OpenMM::NonbondedForce* nonbondedForce,
                     const std::string& gridType,
                     const std::vector<int>& receptorAtoms,
                     const std::vector<OpenMM::Vec3>& receptorPositions,
                     double originX, double originY, double originZ,
                     std::vector<double>& vals,
                     std::vector<double>& derivatives);

    /**
     * Generate grid directly to a tiled file, tile-by-tile.
     * This avoids holding the full grid in memory - suitable for very large grids.
     */
    void generateGridToTiledFile(const OpenMM::System& system,
                                 const OpenMM::NonbondedForce* nonbondedForce,
                                 const std::string& gridType,
                                 const std::vector<int>& receptorAtoms,
                                 const std::vector<OpenMM::Vec3>& receptorPositions,
                                 double originX, double originY, double originZ,
                                 const std::string& outputFilename,
                                 bool computeDerivatives,
                                 int tileSize = 32);

    bool hasInitializedKernel;
    int numAtoms;
    float invPower;
    int invPowerMode;         // 0=NONE, 1=RUNTIME, 2=STORED
    float gridCap;
    float outOfBoundsRestraint;
    int interpolationMethod;  // 0=trilinear, 1=cubic B-spline, 2=tricubic, 3=quintic Hermite
    float originX, originY, originZ;
    OpenMM::CudaContext& cu;
    OpenMM::CudaArray g_counts;
    OpenMM::CudaArray g_spacing;
    OpenMM::CudaArray g_vals;  // Only used if not sharing
    OpenMM::CudaArray g_scaling_factors;
    OpenMM::CudaArray g_derivatives;  // Only used if not sharing
    OpenMM::CudaArray particleIndices;  // Filtered particle indices (empty = all particles)
    std::shared_ptr<OpenMM::CudaArray> g_vals_shared;        // Shared grid values (when cached)
    std::shared_ptr<OpenMM::CudaArray> g_derivatives_shared; // Shared derivatives (when cached)
    CUfunction kernel;
    CUfunction addGroupEnergiesKernel;  // Kernel to sum group energies into main buffer
    std::vector<int> counts;
    std::vector<double> spacing;
    std::vector<int> ligandAtoms;  // Particle indices for ligand atoms
    std::vector<int> particles;    // Filtered particles for evaluation (empty = all particles)
    bool computeDerivatives;       // Whether derivatives were computed

    // Particle group data for multi-ligand workflows (flattened for single kernel launch)
    OpenMM::CudaArray allGroupParticleIndices;  // Flattened particle indices from all groups
    OpenMM::CudaArray allGroupScalingFactors;   // Flattened scaling factors from all groups
    int totalGroupParticles;                     // Total particles across all groups

    // Per-group energy tracking
    OpenMM::CudaArray particleToGroupMap;       // Map from particle index to group index
    OpenMM::CudaArray groupEnergyBuffer;        // Per-group energy accumulation (gets zeroed each execute)
    std::vector<float> lastGroupEnergies;        // Persistent copy of last group energies
    int numParticleGroups;                       // Number of particle groups

    // Per-atom energy tracking (for debugging/analysis)
    OpenMM::CudaArray atomEnergyBuffer;         // Per-atom energy storage
    std::vector<float> lastAtomEnergies;         // Persistent copy of last atom energies

    // Tiled grid streaming support
    bool tiledMode;                              // Whether tiled streaming is enabled
    std::unique_ptr<TileManager> tileManager;   // Manages tile loading and caching
    CUfunction tiledKernel;                      // Kernel for tiled grid evaluation
    std::vector<float> hostGridValues;          // Host-side copy of grid values (for tiling)
    std::vector<float> hostGridDerivatives;     // Host-side copy of derivatives (for tiling)
};

// Clear GPU-side grid caches to free CUDA memory
// Call this between systems in batch workflows to prevent GPU memory exhaustion
void clearCudaGridCaches();

} // namespace GridForcePlugin

#endif /*CUDA_GRIDFORCE_KERNELS_H_*/
