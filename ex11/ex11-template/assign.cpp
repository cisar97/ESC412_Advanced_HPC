// This uses features from C++17, so you may have to turn this on to compile
#include <iostream>
#include <fstream>
#include <cstdint>
#include <chrono>
#include <locale>
#include <new>
#include <complex>
#include <algorithm>
#include <numeric>
#include <stdlib.h>
#include "mpi.h"
#ifdef _OPENMP
#include <omp.h>
#endif
#include "blitz/array.h"
#include "fftw3-mpi.h"
#include "tipsy.h"
#include "aweights.h"
using namespace blitz;
using hrc = std::chrono::high_resolution_clock;
using duration = std::chrono::duration<double>;

#define TRANSPOSED_ORDER 1 // Keep the transform in transposed order
#define MANUAL_TRANSPOSE 1 // Manually do the 2D, transpose, and final 1D FFTs

// Manual transpose requires that we use transposed order
static_assert(TRANSPOSED_ORDER==1 || MANUAL_TRANSPOSE==0);

#if TRANSPOSED_ORDER
// This lays out the k-space array in transposed order.
class TransposedArray : public blitz::GeneralArrayStorage<3> {
public:
    TransposedArray(int jstart=0)
        : blitz::GeneralArrayStorage<3>(noInitializeFlag()) {
        ordering_(0) = thirdRank;
        ordering_(1) = firstRank;
        ordering_(2) = secondRank;
        ascendingFlag_ = true;
        base_[0] = 0;
        base_[1] = jstart;
        base_[2] = 0;
        }
    };
#endif

// A separate version is created for each different "Order".
// This allows the compiler to optimize the process for each of the four orders
template<int Order=1>
void assign_mass(Array<float,3> &grid, Array<float,2> &R,Array<float,1> &M) {
    auto nGrid = grid.columns();
    // C++ Lambda to apply the periodic wrap of the grid index
    auto wrap = [nGrid](int i) {
        if (i<0) i+=nGrid;
        else if (i>=nGrid) i-=nGrid;
        return i;
    };

    #pragma omp parallel for
    for(int pn=R.lbound(0); pn<=R.ubound(0); ++pn) {
        float x = R(pn,0);
        float y = R(pn,1);
        float z = R(pn,2);
        float m = M(pn);
        AssignmentWeights<Order,float> Hx((x+0.5f)*nGrid),Hy((y+0.5f)*nGrid),Hz((z+0.5f)*nGrid);
        for(auto i=0; i<Order; ++i) {
            for(auto j=0; j<Order; ++j) {
                for(auto k=0; k<Order; ++k) {
                    // The slab index (first dimension) is calculated differently because we are
                    // allowed to index into the ghost region which may lie outside of the "grid"
                    #pragma omp atomic
                    grid(wrap(Hx.i) + i,wrap(Hy.i+j),wrap(Hz.i+k)) += m * Hx.H[i]*Hy.H[j]*Hz.H[k];
                }
            }
        }
    }
}

int main(int argc, char *argv[]) {
    int thread_support;
    MPI_Init_thread(&argc,&argv,MPI_THREAD_FUNNELED,&thread_support);
    if (thread_support < MPI_THREAD_FUNNELED) {
        cerr << "Insufficient MPI thread support -- Funneled required" << std::endl;
        return 1;
    }
    int irank, nrank; // Index of current MPI rank, and total number of MPI ranks.
    MPI_Comm_rank(MPI_COMM_WORLD,&irank);
    MPI_Comm_size(MPI_COMM_WORLD,&nrank);

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
    const auto k_nz = nGrid/2 + 1;
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

    auto wrap = [nGrid](int i) {
        if (i<0) i+=nGrid;
        else if (i>=nGrid) i-=nGrid ;
        return i;
    };

    // Read the position and masses from the file
    auto t0 = hrc::now();
    std::ifstream io(argv[1],std::ifstream::binary);
    if (!io) {
        std::cerr << "Unable to open tipsy file " << argv[1] << std::endl;
        return errno;
    }
    constexpr std::streamsize buffer_size = 8 * 1024 * 1024;
    char* buffer = new char[buffer_size];
    io.rdbuf()->pubsetbuf(buffer, buffer_size);

    tipsy::header h;
    if (!io.read(reinterpret_cast<char*>(&h),sizeof(h))) {
        std::cerr << "error reading header" << std::endl;
        return errno;
    }

    // Load particle positions and masses
    std::uint64_t N = h.nDark;
    if (irank==0) std::cerr << "Loading " << N << " particles" << std::endl;

    // Calculate the start and end particle for this MPI rank
    auto nper = (N + nrank - 1) / nrank;
    auto beg = nper * irank;
    auto end = nper * (irank + 1);
    if (beg>N) beg = N;
    if (end>N) end = N;

    // Create a single 2D array including both r and m, then make a subarray and slice
    Array<float,2> r_m(Range(beg,end-1),Range(0,3));        // x,y,z,m
    Array<float,2> r = r_m(Range(beg,end-1),Range(0,2));    // View into the array for r
    Array<float,1> m = r_m(Range(beg,end-1),3);             // View into the array for m

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
    io.close();
    delete [] buffer;

    duration dt = hrc::now() - t0;
    auto rate = (sizeof(tipsy::header) + N*sizeof(tipsy::dark)) / dt.count() / 1024 / 1024;
    if (irank==0) std::cerr << "File reading took " << std::setw(9) << dt.count() << " seconds (" << rate <<" MB/s)." << std::endl;

    // Calculate the starting slab, and the number of slabs.
    ptrdiff_t local_n0, local_0_start, complex_count;
#if TRANSPOSED_ORDER
    ptrdiff_t local_n1, local_1_start;
#if MANUAL_TRANSPOSE
    ptrdiff_t n_transpose[] = {nGrid,nGrid};
    complex_count = fftwf_mpi_local_size_many_transposed(2,n_transpose,k_nz,
        FFTW_MPI_DEFAULT_BLOCK,FFTW_MPI_DEFAULT_BLOCK,MPI_COMM_WORLD,
        &local_n0, &local_0_start, &local_n1, &local_1_start);
#else
    complex_count = fftwf_mpi_local_size_3d_transposed(nGrid, nGrid, k_nz, MPI_COMM_WORLD, &local_n0, &local_0_start, &local_n1, &local_1_start);
#endif
#else
    complex_count = fftwf_mpi_local_size_3d(nGrid,nGrid,k_nz,MPI_COMM_WORLD,&local_n0,&local_0_start);
#endif

    // Exchange information between ranks
    int start = local_0_start;
    Array<int,1> rank_start(nrank);
    MPI_Allgather(&start,1,MPI_INT,rank_start.data(),1,MPI_INT,MPI_COMM_WORLD);

    int count = local_n0;
    Array<int,1> rank_count(nrank);
    MPI_Allgather(&count,1,MPI_INT,rank_count.data(),1,MPI_INT,MPI_COMM_WORLD);

    // Create a slab to rank map
    Array<int,1> slab_to_rank(nGrid);
    slab_to_rank = -1; // These should all be replaced by a valid rank
    for(auto i=0; i<nrank; ++i) {
        for(auto j=0; j<rank_count(i); ++j) {
            slab_to_rank(rank_start(i) + j) = i;
        }
    }
    assert(!any(slab_to_rank<0)); // DEBUG: check that we filled in all of the entries

    // Sort the particles by the x dimension (slabs)
    t0 = hrc::now();
    struct record {
        float r[3];
        float m;
    };
    auto records = reinterpret_cast<record *>(r_m.data());
    auto compare = [&slab_to_rank,nGrid,iOrder](const record &a, const record &b) -> bool {
        auto wrap = [nGrid](int i) {
            if (i<0) i+=nGrid;
            else if (i>=nGrid) i-=nGrid;
            return i;
        };
        auto ai = slab_to_rank(wrap(AssignmentStart(iOrder,(a.r[0]+0.5f)*nGrid)));
        auto bi = slab_to_rank(wrap(AssignmentStart(iOrder,(b.r[0]+0.5f)*nGrid)));
        return ai < bi;
    };
    std::sort(records,records+end-beg,compare);
    dt = hrc::now() - t0;
    if (irank==0) std::cerr << "Sorting took " << std::setw(9) << dt.count() << " seconds." << std::endl;

    t0 = hrc::now();

    // We needs these for the upcoming MPI_Alltoallv where we send the particles to the correct rank
    Array<int,1> scounts(nrank);
    Array<int,1> soffset(nrank);
    Array<int,1> rcounts(nrank);
    Array<int,1> roffset(nrank);

    // Now count the number of particles that we need to send to each rank
    Array<float,1> x = r(Range::all(),0);
    scounts = 0;
    #pragma omp parallel for
    for(int pn=x.lbound(0); pn<=x.ubound(0); ++pn) {
        auto i = slab_to_rank(wrap(AssignmentStart(iOrder,(x(pn)+0.5f)*nGrid)));
        #pragma omp atomic
        ++scounts(i);
    }
    std::exclusive_scan(scounts.begin(),scounts.end(),soffset.begin(),0);

    // DEBUG: make sure we didn't mess it up
    for(auto i=0; i<nrank; ++i) {
        for(auto j=0; j<scounts(i); ++j) {
            int k = r.lbound(0) + soffset(i) + j;
            auto rank = slab_to_rank(wrap(AssignmentStart(iOrder,(r(k,0)+0.5f)*nGrid)));
            assert(rank==i);
        }
    }

    // Exchange these counts with the rank that needs to receive them
    MPI_Alltoall(scounts.data(), 1, MPI_INT, rcounts.data(), 1, MPI_INT, MPI_COMM_WORLD);
    std::exclusive_scan(rcounts.begin(),rcounts.end(),roffset.begin(),0);

    // Allocate the array to receive which may/will be different sizes from the sending arrays
    auto new_count = sum(rcounts);
    Array<float,2> r_m2(new_count,4);
    Array<float,2> r2 = r_m2(Range::all(),Range(0,2));
    Array<float,1> m2 = r_m2(Range::all(),3);

    // We send 4 floats per, so use an MPI type
    MPI_Datatype float4; // 1 block of 4 floats; stride does not matter (0)
    MPI_Type_vector(1, 4, 0, MPI_FLOAT, &float4);
    MPI_Type_commit(&float4);
    MPI_Alltoallv(r_m.data(), scounts.data(), soffset.data(), float4,
                  r_m2.data(), rcounts.data(), roffset.data(), float4,
                  MPI_COMM_WORLD);
    MPI_Type_free(&float4);

    // DEBUG: make sure we didn't mess it up
    for(int pn=r2.lbound(0); pn<=r2.ubound(0); ++pn) {
        auto i = wrap(AssignmentStart(iOrder,(r2(pn,0)+0.5f)*nGrid));
        assert(i >= local_0_start && i < local_0_start + local_n0 + iOrder);
        auto rank = slab_to_rank(i);
        assert(rank==irank);
    }

    dt = hrc::now() - t0;
    if (irank==0) std::cerr << "Particle exchange took " << std::setw(9) << dt.count() << " seconds." << std::endl;

    // Create Mass Assignment Grid

    // Calculate the size of the slabs in floats. This is the maximum of:
    // - what FFTW wants = 2 * complex_count
    // - what is needed for the actual real space slabs PLUS ghost region
    // The ghost region depends on the order of the mass assignment.
    auto slab_count = local_n0 + iOrder - 1;
    auto float_count = std::max(slab_count * nGrid * 2*k_nz,2*complex_count);

    // Create the local slabs
    float *slab_data = new (std:: align_val_t (64)) float [float_count]; // 512-bit alignment
    Array<float,3> raw_slab(slab_data,shape(slab_count,nGrid,2*k_nz),deleteDataWhenDone);
    raw_slab.reindexSelf(TinyVector<int,3>(local_0_start,0,0)); // Change the start index of the first dimension
    Array<float,3> slab(raw_slab(Range(local_0_start,local_0_start+local_n0-1),Range(0,nGrid-1),Range(0,nGrid-1)));
    Array<float,3> ghost(raw_slab(Range(local_0_start,local_0_start+slab_count-1),Range(0,nGrid-1),Range(0,nGrid-1)));
#if TRANSPOSED_ORDER
    Array <std::complex<float>,3> kslab(reinterpret_cast<std::complex<float>*>(slab_data),shape(nGrid,local_n1,k_nz),neverDeleteData,TransposedArray(local_1_start));
#else
    Array <std::complex<float>,3> kslab(reinterpret_cast<std::complex<float>*>(slab_data),shape(local_n0,nGrid,k_nz),neverDeleteData);
    kslab.reindexSelf(TinyVector<int,3>(local_0_start,0,0)); // Change the start index of the first dimension
#endif

    // Create the FFT plan. Generally this should be done BEFORE you set the data as "MEASURE" will use the buffer.
#ifdef _OPENMP
    fftwf_plan_with_nthreads(omp_get_max_threads());
#endif

#if MANUAL_TRANSPOSE && TRANSPOSED_ORDER
    // Transpose plan
    auto plan_transform = fftwf_mpi_plan_many_transpose(
                        nGrid,nGrid,2*k_nz,
                        FFTW_MPI_DEFAULT_BLOCK,FFTW_MPI_DEFAULT_BLOCK,
                        slab.dataFirst(),
                        slab.dataFirst(),
                        MPI_COMM_WORLD,0);

    auto plan_2d = fftwf_plan_dft_r2c_2d(nGrid, nGrid,
                        slab.dataFirst(),
                        reinterpret_cast<fftwf_complex*>(kslab.dataFirst()),
                        FFTW_ESTIMATE);

    constexpr int rank_1d = 1;
    int n_1d[rank_1d] = {nGrid};
    int stride_1d = nGrid/2 + 1;
    int dist_1d = 1;
    int howmany_1d = nGrid/2+1;
    auto plan_1d = fftwf_plan_many_dft(rank_1d, n_1d, howmany_1d,
                        reinterpret_cast<fftwf_complex*>(kslab.dataFirst()),n_1d,stride_1d,dist_1d,
                        reinterpret_cast<fftwf_complex*>(kslab.dataFirst()),n_1d,stride_1d,dist_1d,
                        FFTW_FORWARD,FFTW_ESTIMATE);
#else
    auto plan = fftwf_mpi_plan_dft_r2c_3d(nGrid,nGrid,nGrid,
            slab.dataFirst(),
            reinterpret_cast<fftwf_complex*>(kslab.dataFirst()),
            MPI_COMM_WORLD,
            FFTW_ESTIMATE
#if TRANSPOSED_ORDER
                        | FFTW_MPI_TRANSPOSED_OUT
#endif
            );
#endif

    t0 = hrc::now();

    // Assign the mass to the grid
    raw_slab = 0;

    // DEBUG: make sure we didn't mess it up
    for(int pn=r2.lbound(0); pn<=r2.ubound(0); ++pn) {
        auto i = wrap(AssignmentStart(iOrder,(r2(pn,0)+0.5f)*nGrid));
        assert(i >= local_0_start && i < local_0_start + local_n0 + iOrder);
        auto rank = slab_to_rank(i);
        assert(rank==irank);

        assert(i >= ghost.lbound(0) && i <= ghost.ubound(0));


    }

    // This creates four different versions of "assign mass", one for each order
    if (irank==0) std::cerr << "Assigning mass to the grid using order " << iOrder <<std::endl;
    switch(iOrder) {
    case 1:
        assign_mass<1>(ghost,r2,m2);
        break;
    case 2:
        assign_mass<2>(ghost,r2,m2);
        break;
    case 3:
        assign_mass<3>(ghost,r2,m2);
        break;
    case 4:
        assign_mass<4>(ghost,r2,m2);
        break;
    default:
        std::cerr << "Invalid order " << iOrder << " (must be 1, 2, 3 or 4)" << std::endl;
    }
    dt = hrc::now() - t0;
    if (irank==0) std::cerr << "Mass assignment took " << std::setw(9) << dt.count() << " seconds." << std::endl;


    int slab_size = nGrid * 2*k_nz;

    MPI_Comm comm1, comm2;
    MPI_Comm_split(MPI_COMM_WORLD, irank/2, 0, &comm1); // 0+1, 2+3, etc.
    MPI_Comm_split(MPI_COMM_WORLD, (irank+1) % nrank / 2, irank==0 ? nrank : irank, &comm2);
    if (irank&1) {
        MPI_Reduce(MPI_IN_PLACE,&ghost(int(local_0_start),0,0),slab_size*(iOrder-1),MPI_FLOAT,MPI_SUM,1,comm1);
        MPI_Reduce(&ghost(int(local_0_start+local_n0),0,0),nullptr,slab_size*(iOrder-1),MPI_FLOAT,MPI_SUM,1,comm2);
    }
    else {
        MPI_Reduce(&ghost(int(local_0_start+local_n0),0,0),nullptr,slab_size*(iOrder-1),MPI_FLOAT,MPI_SUM,1,comm1);
        MPI_Reduce(MPI_IN_PLACE,&ghost(int(local_0_start),0,0),slab_size*(iOrder-1),MPI_FLOAT,MPI_SUM,1,comm2);
    }
    MPI_Comm_free(&comm1);
    MPI_Comm_free(&comm2);

    // Calculate deleta
    float local_mass = blitz::sum(slab);
    float total_mass;
    MPI_Allreduce(&local_mass, &total_mass, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    float diRhoBar = ((1.0f*nGrid*nGrid*nGrid)/total_mass);
    slab = slab * diRhoBar - 1.0f;
    // Correct for FFT normalization
    slab /= (nGrid*nGrid*nGrid);

    // Calculate the FFT
    t0 = hrc::now();

#if MANUAL_TRANSPOSE && TRANSPOSED_ORDER
    // Do the 2D transforms
    if (irank==0) std::cerr << "2D transforms\n";
    for(auto i=0; i<local_n0; ++i) {
        fftwf_execute_dft_r2c(plan_2d,
                        reinterpret_cast<float*>(&slab(int(local_0_start+i),0,0)),
                        reinterpret_cast<fftwf_complex*>(&slab(int(local_0_start+i),0,0)));
    }
    // Do the transpose
    if (irank==0) std::cerr << "Global transpose\n";
    fftwf_execute_r2r(plan_transform,slab.dataFirst(),slab.dataFirst());

    // Do the 1D transforms
    if (irank==0) std::cerr << "1D transforms\n";
    for(auto i=0; i<local_n1; ++i) {
        fftwf_execute_dft(plan_1d,
                        reinterpret_cast<fftwf_complex*>(&kslab(0,int(local_1_start+i),0)),
                        reinterpret_cast<fftwf_complex*>(&kslab(0,int(local_1_start+i),0)));
    }
#else
    fftwf_execute(plan);
    fftwf_destroy_plan(plan);
#endif

    dt = hrc::now() - t0;
    if (irank==0) std::cerr << "FFT took " << std::setw(9) << dt.count() << " seconds." << std::endl;

    Array<double,1> ak(nBins);  // Sum of k values in each bin
    Array<double,1> pk(nBins);  // Sum of P(k) values in each bin
    Array<long,1> nk(nBins);  // Number of values in each bin

    ak = 0;
    pk = 0;
    nk = 0;

    if (irank==0) std::cerr << "Using " << nBins << (bLog?" logarithmic":" linear") << " bins." << std::endl;

    AssignmentWindow W(nGrid,iOrder);
    for(auto ii=kslab.begin(); ii!=kslab.end(); ++ii) {
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

    Array<double,1> sum_ak(nBins);
    Array<double,1> sum_pk(nBins);
    Array<long,1> sum_nk(nBins);

    MPI_Reduce(ak.data(), sum_ak.data(), nBins, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(pk.data(), sum_pk.data(), nBins, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(nk.data(), sum_nk.data(), nBins, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    if (irank==0)  {
        sum_ak = sum_ak / sum_nk;
        sum_pk = sum_pk / sum_nk;
        for(auto i=1; i<nBins; ++i) {
            if (sum_nk(i)) {
                printf("%.10g %.10g %ld\n", sum_ak(i), sum_pk(i),sum_nk(i));
            }
        }
    }

    MPI_Finalize();
}
