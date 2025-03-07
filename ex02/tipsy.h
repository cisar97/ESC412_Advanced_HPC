#include "boost/endian/arithmetic.hpp"

namespace tipsy {
    using namespace boost::endian;

    // Header of a Tipsy file
    struct header {
        big_float64_at dTime;
        big_uint32_at  nBodies;
        big_uint32_at  nDim;
        big_uint32_at  nSph;
        big_uint32_at  nDark;
        big_uint32_at  nStar;
        big_uint32_at  nPad;
    };
    // inline void swap(header &hdr) {}

    // Dark matter particle
    struct dark {
        big_float32_at mass;
        big_float32_at pos[3];
        big_float32_at vel[3];
        big_float32_at eps;
        big_float32_at phi;
    };
    // inline void swap(dark &d) {}
}
