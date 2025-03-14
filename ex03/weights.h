#include <cmath>

inline int ngp_weights(float x, float *W) {
    int i = std::floor(x);
    W[0] = 1.0;
    return i;
}

inline int cic_weights(float x, float *W) {
    float i = std::floor(x-0.5);
    float s0 = x - (i+0.5f);
    W[0] = 1.0f - s0;
    W[1] = 1.0f - W[0];
    return i;
}

inline int tsc_weights(float x, float *W) {
    auto K0 = [](float h) { return 0.75 - h*h; };
    auto K1 = [](float h) { return 0.50 * h*h; };
    int i = std::floor(x-1.0);
    float h = x - i - 1.5;
    W[0] = K1(0.5 - h);
    W[1] = K0(h);
    W[2] = K1(0.5 + h);
    return i;
}

inline int pcs_weights(float x, float *W) {
    auto pow3 = [](float x) { return x*x*x; };
    auto K0   = [](float h) { return 1.0/6.0 * ( 4.0 - 6.0*h*h + 3.0*h*h*h); };
    auto K1   = [&pow3](float h) { return 1.0/6.0 * pow3(2.0 - h); };
    int i = std::floor(x-1.5);
    float h = x - (i+0.5);
    W[0] = K1(h);
    W[1] = K0(h-1);
    W[2] = K0(2-h);
    W[3] = K1(3-h);
    return i;
}

