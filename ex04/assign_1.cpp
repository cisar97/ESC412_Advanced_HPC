// This uses features from C++17, so you may have to turn this on to compile
#include <iostream>
#include <fstream>
#include <cstdint>
#include <chrono>
#include <locale>
#include <new>
#include <stdlib.h>
#include "blitz/array.h"
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
    if (argc<=1) {
        std::cerr << "Usage: " << argv[0] << " tipsyfile.std [grid-size] [order]"
                  << std::endl;
        return 1;
    }

    int nGrid = 100;
    if (argc>2) nGrid = atoi(argv[2]);

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

    // Load particle positions and masses
    std::uint64_t N = h.nDark;
    std::cerr << "Loading " << std::fixed << N << " particles" << std::endl;
    std::cerr << "Loading " << N << " particles" << std::endl;
    Array<float,2> r(N,3);
    Array<float,1> m(N);

    // Load the particles
    tipsy::dark d;
    for(int i=0; i<N; ++i) {
        if (!io.read(reinterpret_cast<char*>(&d),sizeof(d))) {
            std::cerr << "error reading particle" << std::endl;
            return errno;
        }
        r(i,0) = d.pos[0];
        r(i,1) = d.pos[1];
        r(i,2) = d.pos[2];
        m(i) = d.mass;
    }

    duration dt = hrc::now() - t0;
    std::cerr << "File reading took " << std::setw(9) << dt.count() << " seconds." << std::endl;

    // Create Mass Assignment Grid
    t0 = hrc::now();

    // 1: Calculate the number of floats to allocate.
    // 1: Use size_t to avoid integer overflow for large grid sizes
    // 1: Allocate the storage ourselves
    auto n_floats = size_t(1) * nGrid * nGrid * nGrid;
    float *data = new (std:: align_val_t (64)) float [n_floats]; // 512-bit alignment
    Array<float,3> grid(data,shape(nGrid,nGrid,nGrid),deleteDataWhenDone);

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

    // Calculate projected density
    t0 = hrc::now();
    Array<float,2> projected(nGrid,nGrid);
    thirdIndex ii;
    projected = max(grid,ii);
    dt = hrc::now() - t0;
    std::cerr << "Density projection took " << std::setw(9) << dt.count() << " seconds." << std::endl;

    // Write out the 2D map
    std::ofstream of("density.dat",std::ios::binary);
    of.write(reinterpret_cast<char*>(projected.data()),projected.size()*sizeof(float));

}
