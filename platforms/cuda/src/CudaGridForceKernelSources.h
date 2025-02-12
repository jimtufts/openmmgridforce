#ifndef CUDA_GRIDFORCE_KERNELSOURCES_H_
#define CUDA_GRIDFORCE_KERNELSOURCES_H_

namespace GridForcePlugin {

/**
 * This file defines the CUDA kernel for grid force calculations.
 */

class CudaGridForceKernelSources {
public:
static const char* gridForce;
};

const char* CudaGridForceKernelSources::gridForce = R"CUDA(
/**
 * Compute grid forces on atoms.
 */
extern "C" __global__ void computeGridForce(real4* __restrict__ posq,
                                          real4* __restrict__ force,
                                          const int* __restrict__ gridCounts,
                                          const float* __restrict__ gridSpacing,
                                          const float* __restrict__ gridValues,
                                          const float* __restrict__ scalingFactors,
                                          const bool includeEnergy,
                                          mixed* __restrict__ energyBuffer) {
    const int atom = blockIdx.x*blockDim.x + threadIdx.x;
    if (atom >= NUM_ATOMS)
        return;
        
    // Get atom position and check if scaling factor is non-zero
    real4 pos = posq[atom];
    float scalingFactor = scalingFactors[atom];
    if (scalingFactor == 0.0f)
        return;
        
    // Check if atom is inside grid boundaries
    real3 gridCorner = make_real3(gridSpacing[0]*(gridCounts[0]-1),
                                 gridSpacing[1]*(gridCounts[1]-1),
                                 gridSpacing[2]*(gridCounts[2]-1));
    bool isInside = true;
    if (pos.x < 0.0f || pos.x > gridCorner.x ||
        pos.y < 0.0f || pos.y > gridCorner.y ||
        pos.z < 0.0f || pos.z > gridCorner.z)
        isInside = false;
        
    real3 f = make_real3(0);
    mixed energy = 0;
    
    if (isInside) {
        // Calculate grid indices and fractions
        int3 index = make_int3(pos.x/gridSpacing[0],
                              pos.y/gridSpacing[1],
                              pos.z/gridSpacing[2]);
        real3 frac = make_real3(pos.x/gridSpacing[0] - index.x,
                               pos.y/gridSpacing[1] - index.y,
                               pos.z/gridSpacing[2] - index.z);
        real3 oneMinusFrac = make_real3(1.0f) - frac;
        
        // Get grid point indices
        int nyz = gridCounts[1]*gridCounts[2];
        int baseIndex = index.x*nyz + index.y*gridCounts[2] + index.z;
        int ip = baseIndex + nyz;           // ix+1
        int imp = baseIndex + gridCounts[2]; // iy+1
        int ipp = ip + gridCounts[2];       // ix+1, iy+1
        
        // Get grid values for trilinear interpolation
        float vmmm = gridValues[baseIndex];
        float vmmp = gridValues[baseIndex + 1];
        float vmpm = gridValues[imp];
        float vmpp = gridValues[imp + 1];
        float vpmm = gridValues[ip];
        float vpmp = gridValues[ip + 1];
        float vppm = gridValues[ipp];
        float vppp = gridValues[ipp + 1];
        
        // Perform trilinear interpolation
        float vmm = oneMinusFrac.z*vmmm + frac.z*vmmp;
        float vmp = oneMinusFrac.z*vmpm + frac.z*vmpp;
        float vpm = oneMinusFrac.z*vpmm + frac.z*vpmp;
        float vpp = oneMinusFrac.z*vppm + frac.z*vppp;
        
        float vm = oneMinusFrac.y*vmm + frac.y*vmp;
        float vp = oneMinusFrac.y*vpm + frac.y*vpp;
        
        energy = scalingFactor * (oneMinusFrac.x*vm + frac.x*vp);
        
        // Calculate forces
        float dx = (vp - vm)/gridSpacing[0];
        float dy = (oneMinusFrac.x*(vmp-vmm) + frac.x*(vpp-vpm))/gridSpacing[1];
        float dz = (oneMinusFrac.x*(oneMinusFrac.y*(vmmp-vmmm) + frac.y*(vmpp-vmpm)) +
                   frac.x*(oneMinusFrac.y*(vpmp-vpmm) + frac.y*(vppp-vppm)))/gridSpacing[2];
                   
        f = -scalingFactor*make_real3(dx, dy, dz);
    }
    else {
        // Apply harmonic restraint outside grid
        const float kval = 10000.0f; // kJ/mol nm^2
        real3 dev = make_real3(0);
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
            
        energy = 0.5f*kval*(dev.x*dev.x + dev.y*dev.y + dev.z*dev.z);
        f = -scalingFactor*kval*dev;
    }
    
    // Store forces and energy
    real4 f4 = force[atom];
    f4.x += f.x;
    f4.y += f.y;
    f4.z += f.z;
    force[atom] = f4;
    if (includeEnergy)
        atomicAdd(&energyBuffer[0], static_cast<mixed>(energy));
}
)CUDA";

} // namespace GridForcePlugin

#endif /* CUDA_GRIDFORCE_KERNELSOURCES_H_ */
