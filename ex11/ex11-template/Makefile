# module load daint-gpu cudatoolkit cray-fftw Boost

# You might need to specify the include paths for blitz and/or boost
BLITZ_INCLUDE=-I /capstor/store/cscs/uzh/uzh8/blitz/include/
FFTW_INCLUDE=$(shell pkg-config --cflags fftw3f)
MPICH_INCLUDE=$(shell pkg-config --cflags mpich)
BOOST_INCLUDE=-I $(BOOST_ROOT)/include
CUDA_INCLUDE=-I $(CUDA_HOME)/include

LDFFTW=$(shell pkg-config --libs fftw3f) -lfftw3f_threads -lfftw3f_mpi
LDMPICH=$(shell pkg-config --libs mpich)

CXX	= g++
OMP = -fopenmp
LDOMP = -fopenmp

INCLUDES = $(BLITZ_INCLUDE) $(BOOST_INCLUDE) $(MPICH_INCLUDE) $(FFTW_INCLUDE) $(CUDA_INCLUDE)
CPPFLAGS = $(INCLUDES) $(OMP)  -O3 -std=c++17
LDLIBS = $(LDMPICH) $(LDFFTW)

all	: assign fft11 # fft11-gpu

assign	: assign.o tipsy.h Makefile
	$(CXX) -o assign assign.o $(LDOMP) $(LDFLAGS) $(LDLIBS)

assign.o	: assign.cpp tipsy.h aweights.h Makefile

fft11	: fft11.o
	$(CXX) -o fft11 fft11.o $(LDFFTW) $(LDFLAGS) $(LDLIBS)

fft11.o	: fft11.cpp

#fft11-gpu	: fft11-gpu.o tipsy.h Makefile
#	nvcc -Xcompiler -fopenmp -o fft11-gpu fft11-gpu.o $(INCLUDES) $(LDFFTW) $(LDFLAGS) $(LDLIBS) -lcufft

#fft11-gpu.o	: fft11-gpu.cu
#	nvcc -Xcompiler -fopenmp -c -o fft11-gpu.o fft11-gpu.cu $(INCLUDES) -O3 -std=c++17

clean:
	rm -f assign assign.o fft11 fft11.o fft11-gpu fft11-gpu.o
