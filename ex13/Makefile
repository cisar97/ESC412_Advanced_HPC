# module load daint-gpu cudatoolkit cray-fftw Boost

# You might need to specify the include paths for blitz and/or boost
BLITZ_INCLUDE=-I /store/uzh/uzh8/packages/include

CXX	= CC
OMP = -fopenmp
LDOMP = -fopenmp

CPPFLAGS = $(BLITZ_INCLUDE) $(OMP)  -O3 -std=c++17

all	: assign

assign	: assign.o gpufft.o delta.o cudacheck.o tipsy.h Makefile
	$(CXX) -o assign assign.o gpufft.o delta.o cudacheck.o $(LDOMP) $(LDFFTW) $(LDFLAGS) $(LDLIBS) -lcufft

assign.o	: assign.cpp tipsy.h aweights.h Makefile

gpufft.o	: gpufft.cu gpufft.h
	nvcc -c -o gpufft.o gpufft.cu $(BLITZ_INCLUDE) -O3 -std=c++17

delta.o		: delta.cu
	nvcc -c -o delta.o delta.cu -O3 -std=c++17

cudacheck.o		: cudacheck.cu
	nvcc -c -o cudacheck.o cudacheck.cu -O3 -std=c++17

clean:
	rm -f assign assign.o gpufft.o delta.o cudacheck.o
