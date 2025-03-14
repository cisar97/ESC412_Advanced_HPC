#include <iostream>
#include <cstdarg>
#include "weights.h"

int verify(int i, float *W,int iExpected, int iOrder, ...) {
    std::va_list args;
    va_start(args, iOrder);
    int error = 0;
    if (i != iExpected) {
        std::cerr << "ERROR: O(" << iOrder << ") expected index " << iExpected << " but got " << i << std::endl;
        error = 1;
    }
    for(auto iw=0; iw<iOrder; ++iw) {
        float w_expected = va_arg(args, double);
        if (std::abs(W[iw]-w_expected)/w_expected > 5e-5) {
            std::cerr << "ERROR: O(" << iOrder << ")[" << iw << "] expected W " << w_expected << " but got " << W[iw] << std::endl;
            error = 1;
        }
    }
    return error;
}

int main() {
    int nBAD = 0;
    int i;
    float W[4];

    nBAD += verify(ngp_weights(10.0,W),W,10,1,1.0);
    nBAD += verify(ngp_weights(10.2,W),W,10,1,1.0);
    nBAD += verify(ngp_weights(10.5,W),W,10,1,1.0);
    nBAD += verify(ngp_weights(10.8,W),W,10,1,1.0);
    nBAD += verify(ngp_weights( 0.0,W),W, 0,1,1.0);

    nBAD += verify(cic_weights(10.5,W),W,10,2,1.0,0.0);
    nBAD += verify(cic_weights(10.7,W),W,10,2,0.8,0.2);
    nBAD += verify(cic_weights(11.0,W),W,10,2,0.5,0.5);
    nBAD += verify(cic_weights(11.2,W),W,10,2,0.3,0.7);
    nBAD += verify(cic_weights( 0.0,W),W,-1,2,0.5,0.5);

    nBAD += verify(tsc_weights(10.5,W),W, 9,3,0.125,0.75,0.125);
    nBAD += verify(tsc_weights(10.7,W),W, 9,3,0.045,0.71,0.245);
    nBAD += verify(tsc_weights(11.0,W),W,10,3,0.5,  0.5, 0.0);
    nBAD += verify(tsc_weights(11.2,W),W,10,3,0.32, 0.66,0.02);
    nBAD += verify(tsc_weights( 0.0,W),W,-1,3,0.5,  0.5, 0.0);

    nBAD += verify(pcs_weights(10.5,W),W, 9,4,1.000/6,4.000/6,1.000/6,0.0);
    nBAD += verify(pcs_weights(10.7,W),W, 9,4,0.512/6,3.784/6,1.696/6,0.008/6);
    nBAD += verify(pcs_weights(11.0,W),W, 9,4,0.125/6,2.875/6,2.875/6,0.125/6);
    nBAD += verify(pcs_weights(11.2,W),W, 9,4,0.027/6,2.089/6,3.541/6,0.343/6);
    nBAD += verify(pcs_weights( 0.0,W),W,-2,4,0.125/6,2.875/6,2.875/6,0.125/6);

    return nBAD;
}
