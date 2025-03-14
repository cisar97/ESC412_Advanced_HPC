CXX	= g++

# You might need to specify the include paths for blitz and/or boost
BLITZ_INCLUDE=-I /opt/homebrew/Cellar/blitz/1.0.2/include
BOOST_INCLUDE=-I /opt/homebrew/Cellar/boost/1.83.0/include

# Linux
# OMP = -fopenmp
# LDOMP = -fopenmp
# MacOS
OMP = -Xpreprocessor -fopenmp
LDOMP = -L/opt/homebrew/Cellar/libomp/18.1.0/lib -lomp

CXXFLAGS = $(BLITZ_INCLUDE) $(BOOST_INCLUDE) $(OMP) -Wno-deprecated-declarations  -O3 -std=c++17

all	: assign test_weights

assign	: assign.o tipsy.h Makefile
	$(CXX) -o assign assign.o $(LDOMP) $(LDFLAGS) $(LDLIBS)

assign.o	: assign.cpp tipsy.h aweights.h Makefile

test_weights: test_weights.cpp weights.h
	$(CXX) -std=c++11 -O3 -o test_weights test_weights.cpp

clean:
	rm -f assign assign.o test_weights
