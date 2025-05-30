/*  This file is part of PKDGRAV3 (http://www.pkdgrav.org/).
 *  Copyright (c) 2001-2024 Douglas Potter
 *
 *  PKDGRAV3 is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  PKDGRAV3 is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with PKDGRAV3.  If not, see <http://www.gnu.org/licenses/>.
 */
#ifndef AWEIGHTS_HPP
#define AWEIGHTS_HPP
#include <cmath>
#include <vector>

template<typename F=float>
int AssignmentStart(int order,F x) {
    return std::floor(x-F(0.5)*(order-1));
    return 0;
}

// This template object accepts an order (1 to 4) and a type for the weights (e.g., float)
// The constructor takes a single parameter which is the position on the ASSIGNMENT GRID.
// For example, for assignment onto a 100x100x100 grid, the x coordinate would be given
// as a floating point number between 0 and 100 (not including 100), e.g. 54.987
//
// It will calculate the index "i" of the first cell where to assign mass. Mass is assigned
// to i, i+1, i+2 and i+3 (depending on the order). Beyond order 1, the index i can be
// negative, and the last index (e.g. i+3) can be greater than the grid size. This needs to
// be handled outside, typically by applying periodic boundary conditions on the grid.
//
// The corresponding weight/fraction is calculated and stored in H[], so cell index "i" gets
// the fraction H[0], i+1 gets H[1] and so forth.
template<int Order,typename F=float>
class AssignmentWeights {
    template <int A, typename B> struct identity {};
    void weights(identity<1,F> d, F r) {		// NGP
        i = AssignmentStart(Order,r);
        H[0] = F(1.0);
    }
    void weights(identity<2,F> d, F r) {		// CIC
        F rr = r - F(0.5);
        i = AssignmentStart(Order,r);
        F h = rr - i;
        H[0] = F(1.0) - h;
        H[1] = h;
    }
    void weights(identity<3,F> d, F r) {		// TSC
        auto K0 = [](F h) {
            return F(0.75) - h*h;
        };
        auto K1 = [](F h) {
            return F(0.50) * h*h;
        };
        i = AssignmentStart(Order,r);
        F h = r - i - F(1.5);
        H[0] = K1(F(0.5) - h);
        H[1] = K0(h);
        H[2] = K1(F(0.5) + h);
    }
    void weights(identity<4,F> d, F r) {		// PCS
        auto pow3 = [](F x) {
            return x*x*x;
        };
        auto K0   = [](F h) {
            return F(1.0/6.0) * ( F(4.0) - F(6.0)*h*h + F(3.0)*h*h*h);
        };
        auto K1   = [&pow3](F h) {
            return F(1.0/6.0) * pow3(F(2.0) - h);
        };
        i = AssignmentStart(Order,r);
        F h = r - (i+F(0.5));
        H[0] = K1(h);
        H[1] = K0(h-1);
        H[2] = K0(2-h);
        H[3] = K1(3-h);
    }
public:
    F H[Order];
    int i;
    AssignmentWeights(F r) {
        weights(identity<Order,F>(),r);
    }
};

class AssignmentWindow : public std::vector<float> {
public:
    AssignmentWindow(int nGrid,int iAssignment) {
        reserve(nGrid);
        for( auto i=0; i<nGrid; ++i) {
            float win = M_PI * i / nGrid;
            if(win>0.1) win = win / sinf(win);
            else win=1.0 / (1.0-win*win/6.0*(1.0-win*win/20.0*(1.0-win*win/76.0)));
            push_back(powf(win,iAssignment));
        }
    }
};
#endif
