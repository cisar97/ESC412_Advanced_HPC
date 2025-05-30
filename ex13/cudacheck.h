#include "cufft.h"
#include "cuda_runtime_api.h"
void CUDA_Abort(cudaError_t rc, const char *fname, const char *file, int line);
void CUDA_Abort(cufftResult rc, const char *fname, const char *file, int line);
#define CUDA_CHECK(f,a) {auto rc = (f)a; if (rc!=0) CUDA_Abort(rc,#f,__FILE__,__LINE__);}
