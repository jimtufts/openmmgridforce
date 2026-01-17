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
    /**
     * Get per-atom out-of-bounds flags for particles in groups.
     *
     * @return vector of flags (0 = inside grid, 1 = outside grid), one per particle
     *         across all groups (in the same order as particles were added to groups)
     */
    std::vector<int> getParticleOutOfBoundsFlags();
    /**
     * Compute Hessian (second derivative) blocks for each atom from grid potential.
     * Must be called after execute() to get positions. Results stored internally.
     * Only supported for bspline (method 1) and triquintic (method 3) interpolation.
     */
    void computeHessian();
    /**
     * Get the Hessian blocks computed by computeHessian().
     *
     * @return vector of 6 components per atom: [dxx, dyy, dzz, dxy, dxz, dyz]
     *         Total size is 6 * numAtoms
     */
    std::vector<double> getHessianBlocks();
    /**
     * Analyze Hessian blocks to compute per-atom metrics: eigenvalues, curvature,
     * anisotropy, and entropy estimates. Must call computeHessian() first.
     *
     * @param temperature  Temperature in Kelvin for entropy calculation
     */
    void analyzeHessian(float temperature);
    /**
     * Get the eigenvalues computed by analyzeHessian().
     * @return vector of 3 eigenvalues per atom [λ1, λ2, λ3], sorted ascending
     */
    std::vector<double> getEigenvalues();
    /**
     * Get the eigenvectors computed by analyzeHessian().
     * @return vector of 9 components per atom (3 eigenvectors × 3 components)
     */
    std::vector<double> getEigenvectors();
    /**
     * Get the mean curvature computed by analyzeHessian().
     * @return vector of mean curvature per atom: (λ1 + λ2 + λ3) / 3
     */
    std::vector<double> getMeanCurvature();
    /**
     * Get the total curvature computed by analyzeHessian().
     * @return vector of total curvature per atom: λ1 + λ2 + λ3
     */
    std::vector<double> getTotalCurvature();
    /**
     * Get the Gaussian curvature computed by analyzeHessian().
     * @return vector of Gaussian curvature per atom: λ1 * λ2 * λ3
     */
    std::vector<double> getGaussianCurvature();
    /**
     * Get the fractional anisotropy computed by analyzeHessian().
     * @return vector of FA per atom, range [0, 1] (0=isotropic, 1=linear)
     */
    std::vector<double> getFracAnisotropy();
    /**
     * Get the per-atom entropy computed by analyzeHessian().
     * @return vector of entropy per atom in kB units (NaN for saddle points)
     */
    std::vector<double> getEntropy();
    /**
     * Get the minimum eigenvalue computed by analyzeHessian().
     * @return vector of minimum eigenvalue per atom
     */
    std::vector<double> getMinEigenvalue();
    /**
     * Get the count of negative eigenvalues computed by analyzeHessian().
     * @return vector of negative eigenvalue count per atom (0-3)
     */
    std::vector<int> getNumNegative();
    /**
     * Get the total entropy summed over all atoms (excluding saddle points).
     * @return total entropy in kB units
     */
    double getTotalEntropy();
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

    // Per-atom out-of-bounds tracking
    OpenMM::CudaArray outOfBoundsBuffer;        // Per-atom out-of-bounds flags (0=inside, 1=outside)
    std::vector<int> lastOutOfBoundsFlags;       // Persistent copy of last out-of-bounds flags

    // Tiled grid streaming support
    bool tiledMode;                              // Whether tiled streaming is enabled
    std::unique_ptr<TileManager> tileManager;   // Manages tile loading and caching
    CUfunction tiledKernel;                      // Kernel for tiled grid evaluation
    std::vector<float> hostGridValues;          // Host-side copy of grid values (for tiling)
    std::vector<float> hostGridDerivatives;     // Host-side copy of derivatives (for tiling)

    // Hessian (second derivative) computation support
    CUfunction hessianKernel;                    // Kernel for Hessian computation
    OpenMM::CudaArray hessianBuffer;            // Per-atom Hessian storage (6 floats per atom)
    std::vector<float> lastHessianBlocks;        // Persistent copy of last Hessian computation

    // Hessian analysis support (eigendecomposition, curvature, entropy)
    CUfunction analysisKernel;                   // Kernel for per-atom Hessian analysis
    CUfunction sumEntropyKernel;                 // Kernel for entropy reduction
    OpenMM::CudaArray eigenvaluesBuffer;         // [3 * numAtoms] eigenvalues
    OpenMM::CudaArray eigenvectorsBuffer;        // [9 * numAtoms] eigenvectors
    OpenMM::CudaArray meanCurvatureBuffer;       // [numAtoms]
    OpenMM::CudaArray totalCurvatureBuffer;      // [numAtoms]
    OpenMM::CudaArray gaussianCurvatureBuffer;   // [numAtoms]
    OpenMM::CudaArray fracAnisotropyBuffer;      // [numAtoms]
    OpenMM::CudaArray entropyBuffer;             // [numAtoms]
    OpenMM::CudaArray minEigenvalueBuffer;       // [numAtoms]
    OpenMM::CudaArray numNegativeBuffer;         // [numAtoms] int
    OpenMM::CudaArray totalEntropyBuffer;        // [1] scalar
    bool analysisBuffersInitialized;             // Whether analysis buffers are allocated
    std::vector<float> lastEigenvalues;
    std::vector<float> lastEigenvectors;
    std::vector<float> lastMeanCurvature;
    std::vector<float> lastTotalCurvature;
    std::vector<float> lastGaussianCurvature;
    std::vector<float> lastFracAnisotropy;
    std::vector<float> lastEntropy;
    std::vector<float> lastMinEigenvalue;
    std::vector<int> lastNumNegative;
    float lastTotalEntropy;
};

// Clear GPU-side grid caches to free CUDA memory
// Call this between systems in batch workflows to prevent GPU memory exhaustion
void clearCudaGridCaches();

/**
 * CUDA implementation of CalcBondedHessianKernel.
 * Computes analytical Hessian of bonded forces (bonds, angles, torsions) on GPU.
 */
class CudaCalcBondedHessianKernel : public CalcBondedHessianKernel {
public:
    CudaCalcBondedHessianKernel(std::string name, const OpenMM::Platform& platform, OpenMM::CudaContext& cu)
        : CalcBondedHessianKernel(name, platform), cu(cu), hasInitializedKernel(false),
          numAtoms(0), numBonds(0), numAngles(0), numTorsions(0) {
    }
    ~CudaCalcBondedHessianKernel();

    void initialize(const OpenMM::System& system);
    std::vector<double> computeHessian(OpenMM::ContextImpl& context);
    int getNumBonds() const { return numBonds; }
    int getNumAngles() const { return numAngles; }
    int getNumTorsions() const { return numTorsions; }

private:
    OpenMM::CudaContext& cu;
    bool hasInitializedKernel;
    int numAtoms;
    int numBonds;
    int numAngles;
    int numTorsions;

    // GPU arrays for bonded parameters
    OpenMM::CudaArray bondAtoms;      // [numBonds * 2]
    OpenMM::CudaArray bondParams;     // [numBonds * 2]: k, r0
    OpenMM::CudaArray angleAtoms;     // [numAngles * 3]
    OpenMM::CudaArray angleParams;    // [numAngles * 2]: k, theta0
    OpenMM::CudaArray torsionAtoms;   // [numTorsions * 4]
    OpenMM::CudaArray torsionParams;  // [numTorsions * 3]: n, k, phi0

    // GPU array for Hessian output
    OpenMM::CudaArray hessianBuffer;

    // CUDA kernels
    CUfunction bondHessianKernel;
    CUfunction angleHessianKernel;
    CUfunction torsionHessianKernel;
    CUfunction initHessianKernel;
};

} // namespace GridForcePlugin

#endif /*CUDA_GRIDFORCE_KERNELS_H_*/
