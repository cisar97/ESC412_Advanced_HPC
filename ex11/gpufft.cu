#include "gpufft.h"

#define USE_PLAN_MANY 1

static void CUDA_Abort(cudaError_t rc, const char *fname, const char *file, int line) {
    fprintf(stderr,"%s error %d in %s(%d)\n%s\n", fname, rc, file, line, cudaGetErrorString(rc));
    exit(1);
}
static void CUDA_Abort(cufftResult rc, const char *fname, const char *file, int line) {
    fprintf(stderr,"%s error %d in %s(%d)\n", fname, rc, file, line);
    exit(1);
}
#define CUDA_CHECK(f,a) {auto rc = (f)a; if (rc!=0) CUDA_Abort(rc,#f,__FILE__,__LINE__);}

// Create a plan to do a 2D transform for the given grid (in-place)
cufftHandle gpu_make_plan_2D(int nGrid) {
    cufftHandle plan;
#if USE_PLAN_MANY
    int n[] = {nGrid,nGrid};       // 2D FFT of length NxN
    int inembed[] = {nGrid,2*(nGrid/2+1)};
    int onembed[] = {nGrid,nGrid/2+1};
    int howmany = 1;
    int odist = onembed[0] * onembed[1]; // Output distance is in "complex"
    int idist = 2*odist;   // Input distance is in "real"
    int istride = 1;       // Elements of each FFT are adjacent
    int ostride = 1;
    CUDA_CHECK(cufftPlanMany,(&plan,sizeof(n)/sizeof(n[0]), n,
                    inembed,istride,idist,
                    onembed,ostride,odist,
                    CUFFT_R2C,howmany));
#else
    CUDA_CHECK(cufftPlan2d,(&plan,nGrid,nGrid,CUFFT_R2C));
#endif
    return plan;
}

void gpu_fft_2D_R2C(blitz::Array<float,2> &grid,void *slab,cufftHandle plan) {
    auto data_size = sizeof(cufftComplex)*grid.rows()*(grid.cols()/2+1);
    CUDA_CHECK(cudaMemcpy,(slab, grid.dataFirst(), data_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cufftExecR2C,(plan,reinterpret_cast<cufftReal*>(slab),reinterpret_cast<cufftComplex*>(slab)));
    CUDA_CHECK(cudaMemcpy,(grid.dataFirst(), slab, data_size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize,());
}

void *gpu_allocate_slab(size_t nGrid) {
    void *cuda_slab;
    auto slab_size = sizeof(cufftComplex)*nGrid*(nGrid/2+1);
    CUDA_CHECK(cudaMalloc,((void**)&cuda_slab, slab_size));
    return cuda_slab;
}

