#ifndef OPENMM_PLUGINCOMPATMINIMIZEKERNEL_H_
#define OPENMM_PLUGINCOMPATMINIMIZEKERNEL_H_

/* -------------------------------------------------------------------------- *
 *                              OpenMMGridForce                               *
 * -------------------------------------------------------------------------- *
 * Verbatim copy of OpenMM::CommonMinimizeKernel that swaps in the plugin-    *
 * local pluginCompatMinimize.cc NVRTC source blob, which carries a           *
 * pre-Pascal (sm < 60) software fallback for atomicAdd(double*). Registered  *
 * under OpenMM::MinimizeKernel::Name() so it REPLACES the stock CUDA         *
 * factory when the plugin loads; openmm.LocalEnergyMinimizer.minimize()      *
 * automatically routes through this on any Context that uses the CUDA       *
 * platform with the plugin loaded.                                          *
 *                                                                            *
 * On sm >= 60 the fallback code is #if-guarded off, so behavior is byte-    *
 * identical to OpenMM::CommonMinimizeKernel.                                *
 * -------------------------------------------------------------------------- */

#include "openmm/kernels.h"
#include "openmm/common/ComputeContext.h"

namespace GridForcePlugin {

class PluginCompatMinimizeKernel : public OpenMM::MinimizeKernel {
public:
    PluginCompatMinimizeKernel(std::string name, const OpenMM::Platform& platform,
                               OpenMM::ComputeContext& cc)
        : OpenMM::MinimizeKernel(name, platform), cc(cc), isSetup(false),
          cpuContext(NULL), cpuIntegrator(1) {}
    ~PluginCompatMinimizeKernel();
    void initialize(const OpenMM::System& system);
    void execute(OpenMM::ContextImpl& context, double tolerance, int maxIterations,
                 OpenMM::MinimizationReporter* reporter);

private:
    static const double minConstraintTol, kRestraintScale, prevMaxErrorInit,
                        kRestraintScaleUp, constraintTolScale;
    static const double fTol, wolfeParam, stepScaleDown, stepScaleUp, minStep, maxStep;
    static const int numVectors, maxLineSearchIterations;

    void setup(OpenMM::ContextImpl& context);
    void lbfgs(OpenMM::ContextImpl& context);
    void evaluateGpu(OpenMM::ContextImpl& context);
    double evaluateCpu(OpenMM::ContextImpl& context);
    bool report(OpenMM::ContextImpl& context, int iteration);
    void downloadReturnFlagStart();
    void downloadReturnValueStart();
    int downloadReturnFlagFinish();
    double downloadReturnValueFinish();
    double downloadReturnValueSync();
    double downloadGradNormSync();
    void runLineSearchKernels();

    OpenMM::ComputeContext& cc;

    int numParticles, numVariables, numConstraints;

    std::vector<OpenMM::Vec3> hostPositions;
    std::vector<double> hostX;
    std::vector<double> hostGrad;
    std::vector<OpenMM::mm_int2> hostConstraintIndices;
    std::vector<double> hostConstraintDistances;

    bool isSetup, mixedIsDouble;
    int elementSize, threadBlockSize;
    void* pinnedMemory;

    int forceGroups;
    double constraintTol;

    double tolerance;
    int maxIterations;
    OpenMM::MinimizationReporter* reporter;

    double kRestraint, energy;
    bool largeGrad;

    OpenMM::ComputeArray constraintIndices, constraintDistances;
    OpenMM::ComputeArray xInit, x, xPrev, grad, gradPrev, dir;
    OpenMM::ComputeArray alpha, scale, xDiff, gradDiff;
    OpenMM::ComputeArray returnFlag, returnValue, gradNorm, lineSearchData, lineSearchDataBackup;

    OpenMM::ComputeKernel recordInitialPosKernel;
    OpenMM::ComputeKernel restorePosKernel;
    OpenMM::ComputeKernel convertForcesKernel;
    OpenMM::ComputeKernel getConstraintEnergyForcesKernel;
    OpenMM::ComputeKernel getConstraintErrorKernel;
    OpenMM::ComputeKernel initializeDirKernel;
    OpenMM::ComputeKernel gradNormKernel;
    OpenMM::ComputeKernel getDiffKernel;
    OpenMM::ComputeKernel getScaleKernel;
    OpenMM::ComputeKernel reinitializeDirKernel;
    OpenMM::ComputeKernel updateDirAlphaKernel;
    OpenMM::ComputeKernel scaleDirKernel;
    OpenMM::ComputeKernel updateDirBetaKernel;
    OpenMM::ComputeKernel updateDirFinalKernel;
    OpenMM::ComputeKernel lineSearchSetupKernel;
    OpenMM::ComputeKernel lineSearchStepKernel;
    OpenMM::ComputeKernel lineSearchDotKernel;
    OpenMM::ComputeKernel lineSearchContinueKernel;

    OpenMM::ComputeEvent downloadStartEvent;
    OpenMM::ComputeEvent downloadFinishEvent;
    OpenMM::ComputeQueue downloadQueue;

    OpenMM::Context* cpuContext;
    OpenMM::VerletIntegrator cpuIntegrator;
};

} // namespace GridForcePlugin

#endif // OPENMM_PLUGINCOMPATMINIMIZEKERNEL_H_
