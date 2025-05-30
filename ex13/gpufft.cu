#include "gpufft.h"
#include "delta.h"
#include "cudacheck.h"

#define USE_PLAN_MANY 1

std::tuple<cufftHandle,size_t> gpu_make_plan_1D(int nGrid) {
    cufftHandle plan;
    int n[] = {nGrid};       // 2D FFT of length NxN
    int inembed[] = {nGrid};
    int onembed[] = {nGrid};
    int howmany = nGrid/2 + 1;	// There are this many pencils
    int odist = 1; 		// 1D pencils start next to each other
    int idist = 1;   		// 
    int istride = howmany;     	// Elements of each FFT are after each group
    int ostride = howmany;
    size_t workSize;
    CUDA_CHECK(cufftCreate,(&plan));
    CUDA_CHECK(cufftSetAutoAllocation,(plan,0));
    CUDA_CHECK(cufftMakePlanMany,(plan,sizeof(n)/sizeof(n[0]), n,
                    inembed,istride,idist,
                    onembed,ostride,odist,
                    CUFFT_C2C,howmany,&workSize));
    return {plan,workSize};
}

// Create a plan to do a 2D transform for the given grid (in-place)
std::tuple<cufftHandle,size_t> gpu_make_plan_2D(int nGrid) {
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
    size_t workSize;
    CUDA_CHECK(cufftCreate,(&plan));
    CUDA_CHECK(cufftSetAutoAllocation,(plan,0));
    CUDA_CHECK(cufftMakePlanMany,(plan,sizeof(n)/sizeof(n[0]), n,
                    inembed,istride,idist,
                    onembed,ostride,odist,
                    CUFFT_R2C,howmany,&workSize));
#else
    CUDA_CHECK(cufftPlan2d,(&plan,nGrid,nGrid,CUFFT_R2C));
#endif
    return {plan,workSize};
}

void gpu_fft_2D_R2C(blitz::Array<float,2> &grid,void *slab,cufftHandle plan,cudaStream_t stream,void *work,int nGrid,float diRhoBar) {
    auto fslab = static_cast<float*>(slab);
    auto data_size = sizeof(cufftComplex)*grid.rows()*(grid.cols()/2+1);
    CUDA_CHECK(cudaMemcpyAsync,(slab, grid.dataFirst(), data_size, cudaMemcpyHostToDevice,stream));
    compute_delta(fslab,nGrid,diRhoBar,stream);
    CUDA_CHECK(cufftSetStream,(plan,stream));
    CUDA_CHECK(cufftSetWorkArea,(plan,work));
    CUDA_CHECK(cufftExecR2C,(plan,reinterpret_cast<cufftReal*>(slab),reinterpret_cast<cufftComplex*>(slab)));
    CUDA_CHECK(cudaMemcpyAsync,(grid.dataFirst(), slab, data_size, cudaMemcpyDeviceToHost,stream));
}

void gpu_fft_1D_C2C(blitz::Array<std::complex<float>,2> &grid,void *slab,cufftHandle plan,cudaStream_t stream,void *work) {
    auto data_size = sizeof(cufftComplex)*grid.rows()*(grid.cols());
    CUDA_CHECK(cudaMemcpyAsync,(slab, grid.dataFirst(), data_size, cudaMemcpyHostToDevice,stream));
    CUDA_CHECK(cufftSetStream,(plan,stream));
    CUDA_CHECK(cufftSetWorkArea,(plan,work));
    CUDA_CHECK(cufftExecC2C,(plan,reinterpret_cast<cufftComplex*>(slab),reinterpret_cast<cufftComplex*>(slab),CUFFT_FORWARD));
    CUDA_CHECK(cudaMemcpyAsync,(grid.dataFirst(), slab, data_size, cudaMemcpyDeviceToHost,stream));
}

void *gpu_allocate_slab(size_t nGrid) {
    void *cuda_slab;
    auto slab_size = sizeof(cufftComplex)*nGrid*(nGrid/2+1);
    CUDA_CHECK(cudaMalloc,((void**)&cuda_slab, slab_size));
    return cuda_slab;
}

void *gpu_allocate(size_t n) {
    void *data;
    CUDA_CHECK(cudaMalloc,((void**)&data, n));
    return data;
}

