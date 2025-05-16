#include <blitz/array.h>
#include "cufft.h"
cufftHandle gpu_make_plan_2D(int nGrid);
void gpu_fft_2D_R2C(blitz::Array<float,2> &grid,void *slab,cufftHandle plan);
void *gpu_allocate_slab(size_t nGrid);

