CXX	= g++

# You might need to specify the include paths for blitz and/or boost
BLITZ_INCLUDE=-I /opt/homebrew/Cellar/blitz/1.0.2/include
BOOST_INCLUDE=-I /opt/homebrew/Cellar/boost/1.83.0/include

CXXFLAGS = $(BLITZ_INCLUDE) $(BOOST_INCLUDE) -Wno-deprecated-declarations  -O3 -std=c++17

assign	: assign.o tipsy.h
	$(CXX) -o assign assign.o $(LDFLAGS) $(LDLIBS)

assign.o	: assign.cpp tipsy.h

clean:
	rm -f assign assign.o
