#include "cudacheck.h"
#include <stdio.h>
void CUDA_Abort(cudaError_t rc, const char *fname, const char *file, int line) {
    fprintf(stderr,"%s error %d in %s(%d)\n%s\n", fname, rc, file, line, cudaGetErrorString(rc));
    exit(1);
}
void CUDA_Abort(cufftResult rc, const char *fname, const char *file, int line) {
    fprintf(stderr,"%s error %d in %s(%d)\n", fname, rc, file, line);
    exit(1);
}
#define CUDA_CHECK(f,a) {auto rc = (f)a; if (rc!=0) CUDA_Abort(rc,#f,__FILE__,__LINE__);}
