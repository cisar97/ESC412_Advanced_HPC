#include <blitz/array.h>
#include <fftw3.h>
#include <complex>
#include <cmath>
#include <iostream>
#include "cuda.h"
#include "cufft.h"
using namespace blitz;
using std::complex;

#define USE_GPUFFT 1
#if USE_GPUFFT
#include "gpufft.h"
#endif

void fill_array(Array<float,2> &data) {
    // Set the grid to the sum of two sine functions
    for (int i=0; i < data.rows(); i++) {
        for (int j=0; j < data.cols(); j++) {
            float x = (float)i / 25.0; // Period of 1/4 of the box in x
            float y = (float)j / 10.0; // Period of 1/10 of the box in y
            data(i,j) = sin(2.0 * M_PI * x) + sin(2.0 * M_PI * y);
        }
    }
}

// Verify the FFT (kdata) of data by performing a reverse transform and comparing
bool validate(Array<float,2> &data,Array<std::complex<float>, 2> kdata) {
    Array<float,2> rdata(data.extent());
    fftwf_plan plan = fftwf_plan_dft_c2r_2d(data.rows(), data.cols(),
        reinterpret_cast<fftwf_complex*>(kdata.data()), rdata.data(), FFTW_ESTIMATE);
    fftwf_execute(plan);
    fftwf_destroy_plan(plan);
    rdata /= data.size(); // Normalize for the FFT
    return all(abs(data - rdata) < 1e-5);
}

int main() {
    int n = 10000;

    // Out of place: rdata1 -> kdata1 without padding
    Array<float,2> rdata1(n,n);
    Array<std::complex<float>, 2> kdata1(n, n/2 + 1);
    fftwf_plan plan1  = fftwf_plan_dft_r2c_2d(n, n,
        rdata1.data(), reinterpret_cast<fftwf_complex*>(kdata1.data()), FFTW_ESTIMATE);
    fill_array(rdata1);
    fftwf_execute(plan1);
    fftwf_destroy_plan(plan1);
    std::cout << ">>> Out of place FFT " << (validate(rdata1,kdata1)?"match":"MISMATCH") << endl;

    // in-place: rdata2 -> kdata2 (overlays rdata2)
    Array<float,2> raw_data2(n,2*(n/2+1));
    Array<float,2> rdata2 = raw_data2(Range(0,n-1),Range(0,n-1));
    fftwf_plan plan2  = fftwf_plan_dft_r2c_2d(n, n,
        rdata2.data(), reinterpret_cast<fftwf_complex*>(rdata2.data()), FFTW_ESTIMATE);
    fill_array(rdata2);
    fftwf_execute(plan2);
    fftwf_destroy_plan(plan2);
    Array<std::complex<float>, 2> kdata2(reinterpret_cast<std::complex<float>*>(rdata2.data()),
        shape(n, n/2 + 1),neverDeleteData);
    // rdata2 is destroyed now, but it would have the same data as rdata1 so compare with rdata1
    std::cout << ">>> In-place FFT " << (validate(rdata1,kdata2)?"match":"MISMATCH") << endl;

    // GPU memory copy test
    Array<float,2> raw_data3(n,2*(n/2+1));
    Array<float,2> raw_data4(n,2*(n/2+1));
    Array<float,2> rdata3 = raw_data3(Range(0,n-1),Range(0,n-1));
    Array<float,2> rdata4 = raw_data4(Range(0,n-1),Range(0,n-1));
    fill_array(rdata3);

    // We transfer the entire slab, including the padding region
    size_t slab_size_in_bytes = raw_data3.size() * sizeof(float);
    void *cuda_slab;
    cudaMalloc((void**)&cuda_slab, slab_size_in_bytes);
    cudaMemcpy(cuda_slab,rdata3.data(),slab_size_in_bytes,cudaMemcpyHostToDevice);
    cudaMemcpy(rdata4.data(),cuda_slab,slab_size_in_bytes,cudaMemcpyDeviceToHost);
    std::cout << ">>> GPU Memory copy " << (all(rdata3==rdata4)?"match":"MISMATCH") << endl;
 
    // GPU in-place: we reuse rdata2 -> kdata2
    fill_array(rdata2);
#if USE_GPUFFT
    auto plan3 = gpu_make_plan_2D(n);
    gpu_fft_2D_R2C(rdata2,cuda_slab,plan3);
#else
    cufftHandle plan3;
    cufftPlan2d(&plan3,n,n,CUFFT_R2C);
    cudaMemcpy(cuda_slab, raw_data2.dataFirst(), slab_size_in_bytes, cudaMemcpyHostToDevice);
    cufftExecR2C(plan3,reinterpret_cast<cufftReal*>(cuda_slab),reinterpret_cast<cufftComplex*>(cuda_slab));
    cudaMemcpy(raw_data2.dataFirst(), cuda_slab, slab_size_in_bytes, cudaMemcpyDeviceToHost);
#endif
    std::cout << ">>> GPU In-place FFT " << (validate(rdata1,kdata2)?"match":"MISMATCH") << endl;

    return 0;
}
