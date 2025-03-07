// This uses features from C++17, so you may have to turn this on to compile
#include <fstream>
#include <cstdint>
#include <stdlib.h>
#include "blitz/array.h"
#include "tipsy.h"
using namespace blitz;

int main(int argc, char *argv[]) {
    if (argc<=1) {
        std::cerr << "Usage: " << argv[0] << " tipsyfile.std [grid-size]"
                  << std::endl;
        return 1;
    }

    int nGrid = 100;
    if (argc>2) nGrid = atoi(argv[2]);

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
    // tipsy::swap(h); // Don't forget to write this function in tipsy.h

    // Load particle positions and masses
    std::uint64_t N = h.nDark;
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
        // tipsy::swap(d); // Don't forget to write this function in tipsy.h
        r(i,0) = d.pos[0];
        r(i,1) = d.pos[1];
        r(i,2) = d.pos[2];
        m(i) = d.mass;
    }

    // Create Mass Assignment Grid
    Array<float,3> grid(nGrid,nGrid,nGrid);

    grid = 0;
    for(int pn=0; pn<N; ++pn) {
        float x = r(pn,0);
        float y = r(pn,1);
        float z = r(pn,2);

        // Coordinates are from [-0.5,0.5)
        int i = (x + 0.5) * nGrid;
        int j = (y + 0.5) * nGrid;
        int k = (z + 0.5) * nGrid;

        // Deposit the mass onto grid(i,j,k)
        grid(i,j,k) += m(i);
    }

    // Calculate projected density
    Array<float,2> projected(nGrid,nGrid);
    thirdIndex ii;
    projected = max(grid,ii);

    // Write out the 2D map in binary
    // Read into Python with: np.fromfile(filename,dtype=np.single)
    std::ofstream of("density.dat",std::ios::binary);
    of.write(reinterpret_cast<char*>(projected.data()),projected.size()*sizeof(float));

}
