#ifndef CUDA_ISOLATEDBONDEDFORCE_KERNELS_H_
#define CUDA_ISOLATEDBONDEDFORCE_KERNELS_H_

#include "IsolatedBondedForceKernels.h"
#include "openmm/cuda/CudaContext.h"
#include "openmm/cuda/CudaArray.h"
#include <vector>

namespace GridForcePlugin {

class CudaCalcIsolatedBondedForceKernel : public CalcIsolatedBondedForceKernel {
public:
    CudaCalcIsolatedBondedForceKernel(std::string name, const OpenMM::Platform& platform, OpenMM::CudaContext& cu) :
            CalcIsolatedBondedForceKernel(name, platform), hasInitializedKernel(false), cu(cu) {
    }
    ~CudaCalcIsolatedBondedForceKernel();

    void initialize(const OpenMM::System& system, const IsolatedBondedForce& force);
    double execute(OpenMM::ContextImpl& context, bool includeForces, bool includeEnergy);
    void copyParametersToContext(OpenMM::ContextImpl& context, const IsolatedBondedForce& force);
    double getGroupEnergy(int groupIndex) const override;
    std::vector<double> computeHessian(OpenMM::ContextImpl& context, int groupIndex) override;
    std::vector<double> computeInternalForceConstants(OpenMM::ContextImpl& context, int groupIndex) override;

private:
    bool hasInitializedKernel;
    int numAtoms;
    int numParticleGroups;
    int numBonds;
    int numAngles;
    int numTorsions;
    OpenMM::CudaContext& cu;
    CUfunction bondKernel;
    CUfunction angleKernel;
    CUfunction torsionKernel;

    // Bond parameters [numBonds]: (atom1, atom2) as int2, (length, k) as float2
    OpenMM::CudaArray bondAtoms;
    OpenMM::CudaArray bondParams;

    // Angle parameters [numAngles]: (atom1, atom2, atom3) as int4 (w unused), (angle, k) as float2
    OpenMM::CudaArray angleAtoms;
    OpenMM::CudaArray angleParams;

    // Torsion parameters [numTorsions]: (atom1, atom2, atom3, atom4) as int4, (periodicity, phase, k) as float4 (w unused)
    OpenMM::CudaArray torsionAtoms;
    OpenMM::CudaArray torsionParams;

    // Particle groups
    OpenMM::CudaArray groupParticleIndices;  // [numGroups * numAtoms]
    OpenMM::CudaArray groupEnergiesBuffer;   // [numGroups]
    OpenMM::CudaArray fixedPointEnergyBuffer; // [1] unsigned long long - fixed-point energy accumulator

    // Alchemical scaling
    float globalScalingFactor;
    OpenMM::CudaArray groupScalingFactorsBuffer;

    // Host-side cached results
    mutable std::vector<double> groupEnergiesHost;
    std::vector<int> h_particleIndices;

    // Host-side double-precision bonded parameters for Hessian computation
    struct BondInfoH { int atom1, atom2; double length, k; };
    struct AngleInfoH { int atom1, atom2, atom3; double angle, k; };
    struct TorsionInfoH { int atom1, atom2, atom3, atom4; int periodicity; double phase, k; };
    std::vector<BondInfoH> h_bonds;
    std::vector<AngleInfoH> h_angles;
    std::vector<TorsionInfoH> h_torsions;

    bool skipGroupEnergyDownload_ = false;

    // Diagonal Hessian for Riemannian metric
    OpenMM::CudaArray diagHessianBuffer;  // float[6 * numGroups * numAtoms]
    CUfunction bondDiagHessianKernel;
    CUfunction angleDiagHessianKernel;
    CUfunction torsionDiagHessianKernel;
    bool diagHessianInitialized = false;

public:
    void setSkipGroupEnergyDownload(bool skip) override { skipGroupEnergyDownload_ = skip; }
    void* getGroupEnergyDevicePointer() override {
        return groupEnergiesBuffer.isInitialized()
            ? (void*)groupEnergiesBuffer.getDevicePointer() : nullptr;
    }
    void computeDiagonalHessianGPU() override;
    void* getDiagonalHessianDevicePointer() override;
};

} // namespace GridForcePlugin

#endif /*CUDA_ISOLATEDBONDEDFORCE_KERNELS_H_*/
