/* -------------------------------------------------------------------------- *
 *                              OpenMMGridForce                               *
 * -------------------------------------------------------------------------- */

#include "CudaGBSAGridGenerator.h"
#include "CudaGridForceKernelSources.h"
#include "openmm/OpenMMException.h"
#include <cuda_runtime.h>
#include <iostream>
#include <cstring>

using namespace GridForcePlugin;
using namespace OpenMM;
using namespace std;

// Constants matching the CUDA kernels
static const float ONE_4PI_EPS0 = 138.935456f;

CudaGBSAGridGenerator::CudaGBSAGridGenerator()
    : initialized(false), cudaContext(nullptr), cudaModule(nullptr),
      computeReceptorReceptorHCTKernel(nullptr),
      computeBaselineReceptorEnergyKernel(nullptr),
      generateReceptorDesolvationGridKernel(nullptr),
      generateReceptorDesolvationGridWithDerivativesKernel(nullptr),
      baselineEnergy(0.0) {
}

CudaGBSAGridGenerator::~CudaGBSAGridGenerator() {
    if (cudaModule) {
        cuModuleUnload(cudaModule);
    }
}

void CudaGBSAGridGenerator::initialize() {
    if (initialized) return;

    // Initialize CUDA
    CUresult result = cuInit(0);
    if (result != CUDA_SUCCESS) {
        throw OpenMMException("Failed to initialize CUDA");
    }

    // Get device and create context
    CUdevice device;
    result = cuDeviceGet(&device, 0);  // Use first GPU
    if (result != CUDA_SUCCESS) {
        throw OpenMMException("Failed to get CUDA device");
    }

    result = cuCtxCreate(&cudaContext, 0, device);
    if (result != CUDA_SUCCESS) {
        throw OpenMMException("Failed to create CUDA context");
    }

    // Load the kernel module from the embedded source
    const string& kernelSource = CudaGridForceKernelSources::gbsaGridGeneration;

    // Compile the kernel (using nvrtc would be better but let's use cuModuleLoadData for PTX)
    // For now, we need to compile at build time - the kernels are embedded as source strings
    // We'll use the same approach as OpenMM: compile with nvcc at build time

    // Actually, let's load from the pre-compiled module that's part of the plugin
    // The kernels are compiled as part of the CudaKernels target
    // We need to use JIT compilation here

    // Use nvrtc for runtime compilation
    #include <nvrtc.h>

    nvrtcProgram prog;
    nvrtcResult nvResult = nvrtcCreateProgram(&prog, kernelSource.c_str(), "gbsaGridGeneration.cu", 0, nullptr, nullptr);
    if (nvResult != NVRTC_SUCCESS) {
        throw OpenMMException("Failed to create NVRTC program");
    }

    // Compile
    const char* opts[] = {"--gpu-architecture=compute_70", "-default-device"};
    nvResult = nvrtcCompileProgram(prog, 2, opts);
    if (nvResult != NVRTC_SUCCESS) {
        size_t logSize;
        nvrtcGetProgramLogSize(prog, &logSize);
        vector<char> log(logSize);
        nvrtcGetProgramLog(prog, log.data());
        nvrtcDestroyProgram(&prog);
        throw OpenMMException(string("NVRTC compilation failed: ") + log.data());
    }

    // Get PTX
    size_t ptxSize;
    nvrtcGetPTXSize(prog, &ptxSize);
    vector<char> ptx(ptxSize);
    nvrtcGetPTX(prog, ptx.data());
    nvrtcDestroyProgram(&prog);

    // Load module
    result = cuModuleLoadData(&cudaModule, ptx.data());
    if (result != CUDA_SUCCESS) {
        throw OpenMMException("Failed to load CUDA module");
    }

    // Get kernel functions
    result = cuModuleGetFunction(&computeReceptorReceptorHCTKernel, cudaModule, "computeReceptorReceptorHCT");
    if (result != CUDA_SUCCESS) {
        throw OpenMMException("Failed to get computeReceptorReceptorHCT kernel");
    }

    result = cuModuleGetFunction(&computeBaselineReceptorEnergyKernel, cudaModule, "computeBaselineReceptorEnergy");
    if (result != CUDA_SUCCESS) {
        throw OpenMMException("Failed to get computeBaselineReceptorEnergy kernel");
    }

    result = cuModuleGetFunction(&generateReceptorDesolvationGridKernel, cudaModule, "generateReceptorDesolvationGrid");
    if (result != CUDA_SUCCESS) {
        throw OpenMMException("Failed to get generateReceptorDesolvationGrid kernel");
    }

    result = cuModuleGetFunction(&generateReceptorDesolvationGridWithDerivativesKernel, cudaModule, "generateReceptorDesolvationGridWithDerivatives");
    if (result != CUDA_SUCCESS) {
        throw OpenMMException("Failed to get generateReceptorDesolvationGridWithDerivatives kernel");
    }

    initialized = true;
}

void CudaGBSAGridGenerator::ensureInitialized() {
    if (!initialized) {
        initialize();
    }
}

void CudaGBSAGridGenerator::generateReceptorDesolvationGrid(
    const vector<double>& receptorPositions,
    const vector<double>& receptorCharges,
    const vector<double>& receptorRadii,
    const vector<double>& receptorScales,
    shared_ptr<DesolvationGrid> grid,
    double probeRadius,
    double probeScale,
    bool computeDerivatives,
    double soluteDielectric,
    double solventDielectric
) {
    ensureInitialized();

    int numReceptorAtoms = static_cast<int>(receptorCharges.size());
    if (receptorPositions.size() != numReceptorAtoms * 3) {
        throw OpenMMException("receptorPositions size must be 3 * numAtoms");
    }

    // Get grid parameters
    int nx, ny, nz;
    grid->getCounts(nx, ny, nz);
    double ox, oy, oz;
    grid->getOrigin(ox, oy, oz);
    float spacing = static_cast<float>(grid->getSpacing());

    int totalGridPoints = nx * ny * nz;

    // Compute GB prefactor
    float prefactor = static_cast<float>(-ONE_4PI_EPS0 * (1.0/soluteDielectric - 1.0/solventDielectric));

    // Convert to float arrays for GPU
    vector<float> positionsF(numReceptorAtoms * 3);
    vector<float> chargesF(numReceptorAtoms);
    vector<float> radiiF(numReceptorAtoms);
    vector<float> scalesF(numReceptorAtoms);

    for (int i = 0; i < numReceptorAtoms; i++) {
        positionsF[i*3] = static_cast<float>(receptorPositions[i*3]);
        positionsF[i*3+1] = static_cast<float>(receptorPositions[i*3+1]);
        positionsF[i*3+2] = static_cast<float>(receptorPositions[i*3+2]);
        chargesF[i] = static_cast<float>(receptorCharges[i]);
        radiiF[i] = static_cast<float>(receptorRadii[i]);
        scalesF[i] = static_cast<float>(receptorScales[i]);
    }

    // Allocate GPU memory
    CUdeviceptr d_positions, d_charges, d_radii, d_scales;
    CUdeviceptr d_baselineHCT, d_baselineBornRadii, d_partialSums, d_gridData;
    CUdeviceptr d_gridCounts, d_gridSpacing;

    size_t posSize = numReceptorAtoms * 3 * sizeof(float);
    size_t atomSize = numReceptorAtoms * sizeof(float);

    cuMemAlloc(&d_positions, posSize);
    cuMemAlloc(&d_charges, atomSize);
    cuMemAlloc(&d_radii, atomSize);
    cuMemAlloc(&d_scales, atomSize);
    cuMemAlloc(&d_baselineHCT, atomSize);
    cuMemAlloc(&d_baselineBornRadii, atomSize);

    // Grid counts and spacing
    int counts[3] = {nx, ny, nz};
    float spacingArr[3] = {spacing, spacing, spacing};
    cuMemAlloc(&d_gridCounts, 3 * sizeof(int));
    cuMemAlloc(&d_gridSpacing, 3 * sizeof(float));
    cuMemcpyHtoD(d_gridCounts, counts, 3 * sizeof(int));
    cuMemcpyHtoD(d_gridSpacing, spacingArr, 3 * sizeof(float));

    // Grid output
    size_t gridSize = computeDerivatives ? (27 * totalGridPoints * sizeof(float)) : (totalGridPoints * sizeof(float));
    cuMemAlloc(&d_gridData, gridSize);

    // Upload receptor data
    cuMemcpyHtoD(d_positions, positionsF.data(), posSize);
    cuMemcpyHtoD(d_charges, chargesF.data(), atomSize);
    cuMemcpyHtoD(d_radii, radiiF.data(), atomSize);
    cuMemcpyHtoD(d_scales, scalesF.data(), atomSize);

    int blockSize = 256;

    // Step 1: Compute receptor-receptor HCT
    {
        int numBlocks = (numReceptorAtoms + blockSize - 1) / blockSize;
        void* args[] = {
            &d_positions, &d_radii, &d_scales,
            &numReceptorAtoms, &d_baselineHCT
        };
        CUresult result = cuLaunchKernel(computeReceptorReceptorHCTKernel,
            numBlocks, 1, 1, blockSize, 1, 1, 0, nullptr, args, nullptr);
        if (result != CUDA_SUCCESS) {
            throw OpenMMException("Failed to launch computeReceptorReceptorHCT kernel");
        }
        cuCtxSynchronize();
    }

    // Step 2: Compute baseline Born radii and energy
    int energyBlocks = (numReceptorAtoms + blockSize - 1) / blockSize;
    cuMemAlloc(&d_partialSums, energyBlocks * sizeof(float));
    {
        size_t sharedMem = blockSize * sizeof(float);
        void* args[] = {
            &d_charges, &d_radii, &d_baselineHCT,
            &numReceptorAtoms, &prefactor, &d_partialSums
        };
        CUresult result = cuLaunchKernel(computeBaselineReceptorEnergyKernel,
            energyBlocks, 1, 1, blockSize, 1, 1, sharedMem, nullptr, args, nullptr);
        if (result != CUDA_SUCCESS) {
            throw OpenMMException("Failed to launch computeBaselineReceptorEnergy kernel");
        }
        cuCtxSynchronize();
    }

    // Download and sum partial energies
    vector<float> partialSums(energyBlocks);
    cuMemcpyDtoH(partialSums.data(), d_partialSums, energyBlocks * sizeof(float));
    baselineEnergy = 0.0;
    for (int i = 0; i < energyBlocks; i++) {
        baselineEnergy += partialSums[i];
    }
    float baselineEnergyF = static_cast<float>(baselineEnergy);

    // Also compute baseline Born radii for the grid generation kernel
    // (We need to download HCT and compute on CPU, or add another kernel)
    // For simplicity, the grid kernel recomputes Born radii internally

    // Step 3: Generate receptor desolvation grid
    {
        int gridBlocks = (totalGridPoints + blockSize - 1) / blockSize;
        float probeRadiusF = static_cast<float>(probeRadius);
        float probeScaleF = static_cast<float>(probeScale);
        float originXF = static_cast<float>(ox);
        float originYF = static_cast<float>(oy);
        float originZF = static_cast<float>(oz);

        if (computeDerivatives) {
            void* args[] = {
                &d_gridData, &d_positions, &d_charges, &d_radii, &d_scales,
                &d_baselineHCT, &baselineEnergyF, &numReceptorAtoms,
                &probeRadiusF, &probeScaleF, &prefactor,
                &originXF, &originYF, &originZF,
                &d_gridCounts, &d_gridSpacing, &totalGridPoints
            };
            CUresult result = cuLaunchKernel(generateReceptorDesolvationGridWithDerivativesKernel,
                gridBlocks, 1, 1, blockSize, 1, 1, 0, nullptr, args, nullptr);
            if (result != CUDA_SUCCESS) {
                throw OpenMMException("Failed to launch generateReceptorDesolvationGridWithDerivatives kernel");
            }
        } else {
            int computeDerivsInt = 0;
            // Need dummy baselineBornRadii - the kernel recomputes internally
            void* args[] = {
                &d_gridData, &d_positions, &d_charges, &d_radii, &d_scales,
                &d_baselineHCT, &d_baselineBornRadii, &baselineEnergyF, &numReceptorAtoms,
                &probeRadiusF, &probeScaleF, &prefactor,
                &originXF, &originYF, &originZF,
                &d_gridCounts, &d_gridSpacing, &totalGridPoints, &computeDerivsInt
            };
            CUresult result = cuLaunchKernel(generateReceptorDesolvationGridKernel,
                gridBlocks, 1, 1, blockSize, 1, 1, 0, nullptr, args, nullptr);
            if (result != CUDA_SUCCESS) {
                throw OpenMMException("Failed to launch generateReceptorDesolvationGrid kernel");
            }
        }
        cuCtxSynchronize();
    }

    // Download grid data
    if (computeDerivatives) {
        vector<float> gridDataHost(27 * totalGridPoints);
        cuMemcpyDtoH(gridDataHost.data(), d_gridData, gridSize);
        grid->setReceptorDesolvationData(
            vector<float>(gridDataHost.begin(), gridDataHost.begin() + totalGridPoints),
            static_cast<float>(probeRadius));
        grid->setReceptorDesolvDerivatives(gridDataHost);
    } else {
        vector<float> gridDataHost(totalGridPoints);
        cuMemcpyDtoH(gridDataHost.data(), d_gridData, gridSize);
        grid->setReceptorDesolvationData(gridDataHost, static_cast<float>(probeRadius));
    }

    // Free GPU memory
    cuMemFree(d_positions);
    cuMemFree(d_charges);
    cuMemFree(d_radii);
    cuMemFree(d_scales);
    cuMemFree(d_baselineHCT);
    cuMemFree(d_baselineBornRadii);
    cuMemFree(d_partialSums);
    cuMemFree(d_gridData);
    cuMemFree(d_gridCounts);
    cuMemFree(d_gridSpacing);
}
