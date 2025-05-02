CXX	= mpic++

# You might need to specify the include paths for blitz and/or boost
BLITZ_INCLUDE=-I /opt/homebrew/Cellar/blitz/1.0.2/include
BOOST_INCLUDE=-I /opt/homebrew/Cellar/boost/1.84.0_1/include
FFTW_INCLUDE=-I /opt/homebrew/Cellar/fftw/3.3.10_1/include

# Linux
# OMP = -fopenmp
# LDOMP = -fopenmp
# MacOS
OMP = -Xpreprocessor -fopenmp -I/opt/homebrew/Cellar/libomp/18.1.3/include
LDOMP = -L/opt/homebrew/Cellar/libomp/18.1.3/lib -lomp
LDFFTW=-L /opt/homebrew/Cellar/fftw/3.3.10_1/lib -lfftw3f -lfftw3f_omp -lfftw3f_mpi

CXXFLAGS = $(BLITZ_INCLUDE) $(BOOST_INCLUDE) $(FFTW_INCLUDE) $(OMP) -Wno-deprecated-declarations  -O3 -std=c++17

assign	: assign.o tipsy.h Makefile
	$(CXX) -o assign assign.o $(LDOMP) $(LDFFTW) $(LDFLAGS) $(LDLIBS)

assign.o	: assign.cpp tipsy.h aweights.h Makefile

clean:
	rm -f assign assign.o
