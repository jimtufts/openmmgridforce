#ifndef REFERENCE_GBSA_GRID_FORCE_KERNELS_H_
#define REFERENCE_GBSA_GRID_FORCE_KERNELS_H_

/* -------------------------------------------------------------------------- *
 *                              OpenMMGridForce                               *
 * -------------------------------------------------------------------------- *
 * Reference platform implementation of GBSAGridForce kernel.                *
 * Computes grid-based GB implicit solvation with OBC-II Born radii for      *
 * isolated particle groups on CPU. Uses DesolvationGrid for receptor HCT.   *
 * -------------------------------------------------------------------------- */

#include "GBSAGridForceKernels.h"
#include "GBSAGridForce.h"
#include "DesolvationGrid.h"
#include "openmm/Platform.h"
#include <vector>
#include <memory>
#include <set>
#include <functional>

namespace GridForcePlugin {

class ReferenceCalcGBSAGridForceKernel : public CalcGBSAGridForceKernel {
public:
    ReferenceCalcGBSAGridForceKernel(std::string name, const OpenMM::Platform& platform)
        : CalcGBSAGridForceKernel(name, platform),
          numAtoms(0), numParticleGroups(0),
          prefactor(0.0), includeSurfaceArea(false), surfaceTension(0.0),
          globalScalingFactor(1.0), interpolationMethod(0),
          probeRadius(0.0) {}

    void initialize(const OpenMM::System& system, const GBSAGridForce& force) override;
    double execute(OpenMM::ContextImpl& context, bool includeForces, bool includeEnergy) override;
    void updateParametersInContext(OpenMM::ContextImpl& context, const GBSAGridForce& force) override;

    // Per-group energy accessors
    double getGroupEnergy(int groupIndex) const override;
    double getGroupLigandDesolvationEnergy(int groupIndex) const override;
    std::vector<double> getGroupBornRadii(int groupIndex) const override;

    // Hessian
    void computeHessian(OpenMM::ContextImpl& context) override;
    std::vector<double> getHessianBlocks() const override;
    std::vector<double> getFullHessian() const override;

    // Per-(group,atom) out-of-bounds flags from the last execute(): 1 if the
    // ligand atom fell outside the desolvation grid (and so received zero
    // receptor screening). Layout [numParticleGroups * numAtoms].
    std::vector<int> getParticleOutOfBoundsFlags() const override { return outOfBoundsFlags_; }

protected:
    // Build the desolvation grid from the stored receptor parameters (auto-gen).
    // Uses parallelFor over grid points, so it runs serially on Reference and
    // multithreaded on the CPU platform.
    void generateDesolvationGrid(OpenMM::ContextImpl& context);
    // Compute one particle group's contribution (forces into the context array,
    // energy into the per-[g] buffers). Groups are disjoint atom sets, so distinct
    // groups never write the same force entry — safe to run concurrently.
    void computeGroup(int g, std::vector<OpenMM::Vec3>& posData,
                      std::vector<OpenMM::Vec3>& forceData,
                      bool includeForces, bool includeEnergy);

    // Run all groups. Serial here; the CPU platform overrides to parallelize.
    virtual void runGroups(OpenMM::ContextImpl& context,
                           std::vector<OpenMM::Vec3>& posData,
                           std::vector<OpenMM::Vec3>& forceData,
                           bool includeForces, bool includeEnergy);

    // Apply body(i) for i in [0, count). Serial here; the CPU platform overrides
    // to distribute the iterations across the platform thread pool.
    virtual void parallelFor(OpenMM::ContextImpl& context, int count,
                             const std::function<void(int)>& body);

    // Configuration
    int numAtoms;
    int numParticleGroups;
    double prefactor;  // -COULOMB_CONSTANT * (1/solute - 1/solvent)
    bool includeSurfaceArea;
    double surfaceTension;
    double globalScalingFactor;
    std::vector<double> groupScalingFactors;
    int interpolationMethod;
    double probeRadius;

    // Template atom parameters
    std::vector<double> charges;
    std::vector<double> radii;
    std::vector<double> scaleFactors;

    // Particle groups: groupParticleIndices[g][i] = system particle index
    std::vector<std::vector<int>> groupParticleIndices;

    // Exclusions: per-atom set of excluded template atom indices
    std::vector<std::set<int>> exclusionSets;

    // Grid data
    std::shared_ptr<DesolvationGrid> desolvationGrid;

    // Auto-generation inputs (captured in initialize() when no grid is supplied
    // and autoGenerateGrid is set; the grid is built lazily in the first
    // execute(), where a Context — and thus the thread pool — is available).
    bool autoGenerateGrid_ = false;
    std::vector<double> genReceptorPositions_;   // flattened [x0,y0,z0,...]
    std::vector<double> genReceptorRadii_;
    std::vector<double> genReceptorScales_;
    int genCounts_[3] = {0, 0, 0};
    double genOrigin_[3] = {0.0, 0.0, 0.0};
    double genSpacing_ = 0.0;
    std::vector<double> genRThresholds_;
    bool genComputeDerivatives_ = false;
    double genSmoothingSigma_ = 0.0;
    double genCullCutoff_ = 0.0;   // 0 = sum all receptor atoms (exact)
    int genBSplinePrefilterOrder_ = 0;  // 0 = no prefilter (approximating B-spline)
    bool genUseKDE_ = false;            // KDE-smoothed corrections (matches CUDA) vs exact binned
    double genKDEBandwidth_ = 0.04;

    // Per-(group,atom) out-of-bounds flags from the last execute().
    mutable std::vector<int> outOfBoundsFlags_;
    mutable bool warnedOutOfBounds_ = false;

    // Per-group results from last execute()
    mutable std::vector<double> groupEnergies_;
    mutable std::vector<double> groupLigandEnergies_;
    mutable std::vector<std::vector<double>> groupBornRadii_;

    // Hessian storage (over all particle groups, block-diagonal per group)
    mutable std::vector<double> hessianBlocks_;   // [6 * numParticleGroups * numAtoms]
    mutable std::vector<double> fullHessian_;      // [(3 * numParticleGroups * numAtoms)^2]

    // Internal helpers
    double interpolateReceptorHCT(double x, double y, double z,
                                   double R_i_off, bool computeGradient,
                                   double& gradX, double& gradY, double& gradZ,
                                   double* hess = nullptr,
                                   bool* outOfBounds = nullptr) const;
    bool isExcluded(int i, int j) const;
};

}  // namespace GridForcePlugin

#endif /* REFERENCE_GBSA_GRID_FORCE_KERNELS_H_ */
