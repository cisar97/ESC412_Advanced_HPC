# module load daint-gpu cudatoolkit cray-fftw Boost

# You might need to specify the include paths for blitz and/or boost
BLITZ_INCLUDE=-I /store/uzh/uzh8/packages/include

CXX	= CC
OMP = -fopenmp
LDOMP = -fopenmp

CPPFLAGS = $(BLITZ_INCLUDE) $(OMP)  -O3 -std=c++17

all	: assign fft11 fft11-gpu

assign	: assign.o gpufft.o tipsy.h Makefile
	$(CXX) -o assign assign.o gpufft.o $(LDOMP) $(LDFFTW) $(LDFLAGS) $(LDLIBS) -lcufft

assign.o	: assign.cpp tipsy.h aweights.h Makefile

fft11	: fft11.o
	$(CXX) -o fft11 fft11.o $(LDFFTW) $(LDFLAGS) $(LDLIBS)

fft11.o	: fft11.cpp

fft11-gpu	: fft11-gpu.o tipsy.h gpufft.o Makefile
	$(CXX) -o fft11-gpu fft11-gpu.o gpufft.o $(LDFFTW) $(LDFLAGS) $(LDLIBS) -lcufft

fft11-gpu.o	: fft11-gpu.cu
	nvcc -Xcompiler -fopenmp -c -o fft11-gpu.o fft11-gpu.cu $(BLITZ_INCLUDE) -O3 -std=c++17

gpufft.o	: gpufft.cu gpufft.h
	nvcc -c -o gpufft.o gpufft.cu $(BLITZ_INCLUDE) -O3 -std=c++17

clean:
	rm -f assign assign.o fft11 fft11.o fft11-gpu fft11-gpu.o gpufft.o
