// This slab is in real space with padding so nGrid x nGrid
// becomes nGrid x 2 * (nGrid/2+1)
void compute_delta(float *gpu_slab,int nGrid,float diRhoBar,cudaStream_t stream = 0);
