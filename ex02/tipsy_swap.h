#include "boost/endian/conversion.hpp"

namespace tipsy {
    using namespace boost::endian;

    // Header of a Tipsy file
    struct header {
        double dTime;
        std::uint32_t nBodies;
        std::uint32_t nDim;
        std::uint32_t nSph;
        std::uint32_t nDark;
        std::uint32_t nStar;
        std::uint32_t nPad;
    };

    inline void swap(header &hdr) {
        using boost::endian::big_to_native_inplace;
        // Swap all of the fields in "hdr"
        big_to_native_inplace(hdr.dTime);
        big_to_native_inplace(hdr.nBodies);
        big_to_native_inplace(hdr.nDim);
        big_to_native_inplace(hdr.nSph);
        big_to_native_inplace(hdr.nDark);
        big_to_native_inplace(hdr.nStar);
        big_to_native_inplace(hdr.nPad);
    }

    // Dark matter particle
    struct dark {
        float mass;
        float pos[3];
        float vel[3];
        float eps;
        float phi;
    };

    inline void swap(dark &d) {
        using boost::endian::big_to_native_inplace;
        big_to_native_inplace(d.mass);
        big_to_native_inplace(d.pos[0]);
        big_to_native_inplace(d.pos[1]);
        big_to_native_inplace(d.pos[2]);
        big_to_native_inplace(d.vel[0]);
        big_to_native_inplace(d.vel[1]);
        big_to_native_inplace(d.vel[2]);
        big_to_native_inplace(d.eps);
        big_to_native_inplace(d.phi);
    }
}
