#ifndef CUDA_GRIDFORCE_KERNELSOURCES_H_
#define CUDA_GRIDFORCE_KERNELSOURCES_H_

namespace GridForcePlugin {

class CudaGridForceKernelSources {
public:
static const char* gridForce;
};

const char* CudaGridForceKernelSources::gridForce = R"CUDA(
/**
 * Calculate grid forces and energy.
 */
extern "C" __global__ void computeGridForce(real4* __restrict__ posq,
                                          real4* __restrict__ force,
                                          const int* __restrict__ gridCounts,
                                          const float* __restrict__ gridSpacing,
                                          const float* __restrict__ gridValues,
                                          const float* __restrict__ scalingFactors,
                                          const int includeEnergy,
                                          mixed* __restrict__ energyBuffer) {
    // Initialize energy for this thread
    mixed threadEnergy = 0;

    // Get thread index
    const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= NUM_ATOMS)
        return;
        
    // Load atom position and scaling factor
    real4 pos = posq[tid];
    float scalingFactor = scalingFactors[tid];
    
    // Initialize force to zero
    real4 atomForce;
    atomForce.x = 0;
    atomForce.y = 0;
    atomForce.z = 0;
    atomForce.w = 0;
    
    // Skip if scaling factor is zero
    if (scalingFactor == 0.0f) {
        force[tid] = atomForce;
        return;
    }
    
    // Calculate grid boundaries
    float3 gridCorner;
    gridCorner.x = gridSpacing[0] * (gridCounts[0] - 1);
    gridCorner.y = gridSpacing[1] * (gridCounts[1] - 1);
    gridCorner.z = gridSpacing[2] * (gridCounts[2] - 1);
    
    // Check if the atom is inside the grid
    bool isInside = (pos.x >= 0.0f && pos.x <= gridCorner.x &&
                    pos.y >= 0.0f && pos.y <= gridCorner.y &&
                    pos.z >= 0.0f && pos.z <= gridCorner.z);
    
    if (isInside) {
        // Calculate grid indices
        int ix = min(max((int)(pos.x / gridSpacing[0]), 0), gridCounts[0] - 2);
        int iy = min(max((int)(pos.y / gridSpacing[1]), 0), gridCounts[1] - 2);
        int iz = min(max((int)(pos.z / gridSpacing[2]), 0), gridCounts[2] - 2);
        
        // Calculate fractional position within the cell
        float fx = (pos.x / gridSpacing[0]) - ix;
        float fy = (pos.y / gridSpacing[1]) - iy;
        float fz = (pos.z / gridSpacing[2]) - iz;
        
        fx = min(max(fx, 0.0f), 1.0f);
        fy = min(max(fy, 0.0f), 1.0f);
        fz = min(max(fz, 0.0f), 1.0f);
        
        // Calculate one minus fractional position
        float ox = 1.0f - fx;
        float oy = 1.0f - fy;
        float oz = 1.0f - fz;
        
        // Get grid points
        int nyz = gridCounts[1] * gridCounts[2];
        int baseIndex = ix * nyz + iy * gridCounts[2] + iz;
        int ip = baseIndex + nyz;           // ix+1
        int imp = baseIndex + gridCounts[2]; // iy+1
        int ipp = ip + gridCounts[2];       // ix+1, iy+1
        
        // Get grid values safely
        float vmmm = gridValues[baseIndex];
        float vmmp = gridValues[baseIndex + 1];
        float vmpm = gridValues[imp];
        float vmpp = gridValues[imp + 1];
        float vpmm = gridValues[ip];
        float vpmp = gridValues[ip + 1];
        float vppm = gridValues[ipp];
        float vppp = gridValues[ipp + 1];
        
        // Perform trilinear interpolation
        float vmm = oz * vmmm + fz * vmmp;
        float vmp = oz * vmpm + fz * vmpp;
        float vpm = oz * vpmm + fz * vpmp;
        float vpp = oz * vppm + fz * vppp;
        
        float vm = oy * vmm + fy * vmp;
        float vp = oy * vpm + fy * vpp;
        
        threadEnergy = scalingFactor * (ox * vm + fx * vp);
        
        // Calculate forces
        float dx = (vp - vm) / gridSpacing[0];
        float dy = (ox * (vmp - vmm) + fx * (vpp - vpm)) / gridSpacing[1];
        float dz = (ox * (oy * (vmmp - vmmm) + fy * (vmpp - vmpm)) +
                   fx * (oy * (vpmp - vpmm) + fy * (vppp - vppm))) / gridSpacing[2];
        
        atomForce.x = -scalingFactor * dx;
        atomForce.y = -scalingFactor * dy;
        atomForce.z = -scalingFactor * dz;
    }
    else {
        // Apply harmonic restraint outside grid
        const float kval = 10000.0f; // kJ/mol nm^2
        float3 dev;
        dev.x = 0.0f;
        dev.y = 0.0f;
        dev.z = 0.0f;
        
        if (pos.x < 0.0f)
            dev.x = pos.x;
        else if (pos.x > gridCorner.x)
            dev.x = pos.x - gridCorner.x;
            
        if (pos.y < 0.0f)
            dev.y = pos.y;
        else if (pos.y > gridCorner.y)
            dev.y = pos.y - gridCorner.y;
            
        if (pos.z < 0.0f)
            dev.z = pos.z;
        else if (pos.z > gridCorner.z)
            dev.z = pos.z - gridCorner.z;
            
        threadEnergy = 0.5f * kval * (dev.x * dev.x + dev.y * dev.y + dev.z * dev.z);
        atomForce.x = -scalingFactor * kval * dev.x;
        atomForce.y = -scalingFactor * kval * dev.y;
        atomForce.z = -scalingFactor * kval * dev.z;
    }
    
    // Store forces
    force[tid] = atomForce;
    
    // Carefully accumulate energy if requested
    if (includeEnergy && threadEnergy != 0) {
        atomicAdd(&energyBuffer[0], static_cast<mixed>(threadEnergy));
    }
}
)CUDA";

} // namespace GridForcePlugin

#endif /* CUDA_GRIDFORCE_KERNELSOURCES_H_ */
