// This uses features from C++17, so you may have to turn this on to compile
#include <iostream>
#include <fstream>
#include <cstdint>
#include <chrono>
#include <locale>
#include <new>
#include <complex>
#include <stdlib.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "blitz/array.h"
#include "fftw3.h"
#include "tipsy.h"
#include "aweights.h"
using namespace blitz;
using hrc = std::chrono::high_resolution_clock;
using duration = std::chrono::duration<double>;

// A separate version is created for each different "Order".
// This allows the compiler to optimize the process for each of the four orders
template<int Order=1>
void assign_mass(Array<float,3> &grid, Array<float,2> &R,Array<float,1> &M) {
    auto nGrid = grid.rows(); // total number of particles
    // C++ Lambda to apply the periodic wrap of the grid index
    auto wrap = [nGrid](int i) {
        if (i<0) i+=nGrid;
        else if (i>=nGrid) i-=nGrid;
        return i;
    };
    #pragma omp parallel for
    for(int pn=0; pn<R.rows(); ++pn) {
        float x = R(pn,0);
        float y = R(pn,1);
        float z = R(pn,2);
        float m = M(pn);
        AssignmentWeights<Order,float> Hx((x+0.5f)*nGrid),Hy((y+0.5f)*nGrid),Hz((z+0.5f)*nGrid);
        for(auto i=0; i<Order; ++i) {
            for(auto j=0; j<Order; ++j) {
                for(auto k=0; k<Order; ++k) {
                    #pragma omp atomic
                    grid(wrap(Hx.i+i),wrap(Hy.i+j),wrap(Hz.i+k)) += m * Hx.H[i]*Hy.H[j]*Hz.H[k];
                }
            }
        }
    }
}

int main(int argc, char *argv[]) {
    std::locale::global(std::locale("")); // e.g., LC_ALL=en_GB.UTF-8 or de_CH.UTF-8
    std::cerr.imbue(std::locale()); // e.g., LC_ALL=en_GB.UTF-8
    if (argc<=2) {
        std::cerr << "Usage: " << argv[0] << " tipsyfile.std grid-size[/[L]bin-count] [order]"
                  << std::endl;
        return 1;
    }

#ifdef _OPENMP
    fftwf_init_threads();
#endif

    // The second parameter will have the grid size, and optionally the number of bins to use, e.g.,
    // 100      Use a grid size of 100, and use 50 bins
    // 100/40   Grid size is 100 and use 40 bins
    // 100/L35  Grid siye is 100, and use 35 logarithmic bins
    int nGrid = atoi(argv[2]);
    const auto iNyquist = nGrid/2;
    int nBins = iNyquist;
    bool bLog = false;
    auto p = strchr(argv[2],'/');
    if (p) {
        *p++ = '\0';
        if (*p == 'L') {
            bLog = true;
            ++p;
        }
        nBins = atoi(p);
    }

    int iOrder = 1;
    if (argc>3) iOrder = atoi(argv[3]);

    // Read the position and masses from the file
    auto t0 = hrc::now();
    std::ifstream io(argv[1],std::ifstream::binary);
    if (!io) {
        std::cerr << "Unable to open tipsy file " << argv[1] << std::endl;
        return errno;
    }

    tipsy::header h;
    if (!io.read(reinterpret_cast<char*>(&h),sizeof(h))) {
        std::cerr << "error reading header" << std::endl;
        return errno;
    }
    io.close();

    // Load particle positions and masses
    std::uint64_t N = h.nDark;
    std::cerr << "Loading " << N << " particles" << std::endl;
    Array<float,2> r(N,3);
    Array<float,1> m(N);

    #pragma omp parallel
    {
#ifdef _OPENMP
        auto tid = omp_get_thread_num();            // We are thread 'tid'
        auto nthreads = omp_get_num_threads();      // of 'nthread threads'
#else
        // If OpenMP is not enabled/supported then we are thread 0 of 1 thread
        auto tid = 0;
        auto nthreads = 1;
#endif
        auto nper = (N + nthreads - 1) / nthreads;
        auto beg = nper * tid;
        auto end = nper * (tid + 1);
        if (beg>N) beg = N;
        if (end>N) end = N;
        std::ifstream io(argv[1],std::ifstream::binary);

        constexpr std::streamsize buffer_size = 8 * 1024 * 1024;
        char* buffer = new char[buffer_size];
        io.rdbuf()->pubsetbuf(buffer, buffer_size);
        io.seekg( sizeof(tipsy::header) + beg*sizeof(tipsy::dark));
        // Load the particles
        tipsy::dark d;
        for(int i=beg; i<end; ++i) {
            if (!io.read(reinterpret_cast<char*>(&d),sizeof(d))) {
                perror(argv[1]); abort();
            }
            r(i,0) = d.pos[0];
            r(i,1) = d.pos[1];
            r(i,2) = d.pos[2];
            m(i) = d.mass;
        }
        delete [] buffer;
    }

    duration dt = hrc::now() - t0;
    auto rate = (sizeof(tipsy::header) + N*sizeof(tipsy::dark)) / dt.count() / 1024 / 1024;
    std::cerr << "File reading took " << std::setw(9) << dt.count() << " seconds (" << rate <<" MB/s)." << std::endl;

    // Create Mass Assignment Grid
    t0 = hrc::now();

    auto k_nz = nGrid/2 + 1;
    auto n_floats = size_t(1) * nGrid * nGrid * 2*k_nz; // Careful. For odd nGrid 2*k_nz != nGrid + 2
    float *data = new (std:: align_val_t (64)) float [n_floats]; // 512-bit alignment
    Array<float,3> raw_grid(data,shape(nGrid,nGrid,2*k_nz),deleteDataWhenDone);
    Array<float,3> grid(raw_grid(Range(0,nGrid-1),Range(0,nGrid-1),Range(0,nGrid-1)));

    Array <std::complex<float>,3> kgrid(reinterpret_cast<std::complex<float>*>(data),shape(nGrid,nGrid,k_nz),neverDeleteData);

    // Assign the mass to the grid
    grid = 0;

    // This creates four different versions of "assign mass", one for each order
    std::cerr << "Assigning mass to the grid using order " << iOrder <<std::endl;
    switch(iOrder) {
    case 1:
        assign_mass<1>(grid,r,m);
        break;
    case 2:
        assign_mass<2>(grid,r,m);
        break;
    case 3:
        assign_mass<3>(grid,r,m);
        break;
    case 4:
        assign_mass<4>(grid,r,m);
        break;
    default:
        std::cerr << "Invalid order " << iOrder << " (must be 1, 2, 3 or 4)" << std::endl;
    }
    dt = hrc::now() - t0;
    std::cerr << "Mass assignment took " << std::setw(9) << dt.count() << " seconds." << std::endl;
    std::cerr << "Total mass assigned is " << std::setw(9) << blitz::sum(grid) << std::endl;

    // // Calculate projected density
    // t0 = hrc::now();
    // Array<float,2> projected(nGrid,nGrid);
    // thirdIndex ii;
    // projected = max(grid,ii);
    // dt = hrc::now() - t0;
    // std::cerr << "Density projection took " << std::setw(9) << dt.count() << " seconds." << std::endl;

    // // Write out the 2D map
    // std::ofstream of("density.dat",std::ios::binary);
    // of.write(reinterpret_cast<char*>(projected.data()),projected.size()*sizeof(float));

    // Calculate deleta
    float total_mass = blitz::sum(grid);
    float diRhoBar = ((1.0f*nGrid*nGrid*nGrid)/total_mass);
    grid = grid * diRhoBar - 1.0f;
    // Correct for FFT normalization
    grid /= (nGrid*nGrid*nGrid);

    // Calculate the FFT
    t0 = hrc::now();
#ifdef _OPENMP
    fftwf_plan_with_nthreads(omp_get_max_threads());
#endif
    auto plan = fftwf_plan_dft_r2c_3d(nGrid,nGrid,nGrid,
            grid.dataFirst(),
            reinterpret_cast<fftwf_complex*>(kgrid.dataFirst()),
            FFTW_ESTIMATE);
    fftwf_execute(plan);
    fftwf_destroy_plan(plan);
    dt = hrc::now() - t0;
    std::cerr << "FFT took " << std::setw(9) << dt.count() << " seconds." << std::endl;

    Array<double,1> ak(nBins);  // Sum of k values in each bin
    Array<double,1> pk(nBins);  // Sum of P(k) values in each bin
    Array<double,1> nk(nBins);  // Number of values in each bin

    ak = 0;
    pk = 0;
    nk = 0;

    std::cerr << "Using " << nBins << (bLog?" logarithmic":" linear") << " bins." << std::endl;

    AssignmentWindow W(nGrid,iOrder);
    for(auto ii=kgrid.begin(); ii!=kgrid.end(); ++ii) {
        auto pos = ii.position();
        auto bin = [iNyquist,nGrid](int k) {return k<=iNyquist ? k : k-nGrid;};
        auto kx = bin(pos[0]);
        auto ky = bin(pos[1]);
        auto kz = pos[2];
        float k = sqrt(kx*kx + ky*ky + kz*kz);
        *ii *= W[std::abs(kx)] * W[std::abs(ky)] * W[kz]; // Correction for mass assignment
        int i;
        if (bLog) i = int(log(k) / log(iNyquist) * nBins);
        else  i = int(k / iNyquist * nBins);
        if (i>0 && i < nBins) {
            ak(i) += k;
            pk(i) += std::norm(*ii);
            nk(i) += 1;
        }
    }
    ak = ak / nk;
    pk = pk / nk;

    for(auto i=1; i<nBins; ++i) {
        if (nk(i)) {
            printf("%.10g %.10g\n", ak(i), pk(i));
        }
    }





}
