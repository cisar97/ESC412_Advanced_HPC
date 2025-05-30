#include "delta.h"

__global__ void delta(float *slab,int nGrid,float diRhoBar) {
    auto i = threadIdx.x + blockIdx.x * blockDim.x;
    auto jstart = threadIdx.y + blockIdx.y * blockDim.y;
    auto embed = 2 * (nGrid/2+1);
    auto jstride = blockDim.y * gridDim.y;
    if (i<nGrid) {
        for(auto j=jstart; j<nGrid; j+=jstride) {
            float &value = slab[i + j*embed];
            // Convert mass to delta and correct for the FFT
            value = (value * diRhoBar - 1) / (nGrid*nGrid*nGrid);
        }
    }
}


// This slab is in real space with padding so nGrid x nGrid
// becomes nGrid x 2 * (nGrid/2+1)
void compute_delta(float *gpu_slab,int nGrid,float diRhoBar,cudaStream_t stream) {
    // Thread blocks are 32x32 = 1024 threads
    // Calculate the number of thread blocks we need (rounding up)
    auto width = 32;
    auto blocks = (nGrid + width-1) / width;
    auto block_stride = 10;
    dim3 dimBlock(width,width,1);
    dim3 dimGrid(blocks,std::min(block_stride,blocks),1);
    delta<<<dimGrid,dimBlock,0,stream>>>(gpu_slab,nGrid,diRhoBar);
}
