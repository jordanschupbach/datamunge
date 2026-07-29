#pragma once

// Borwein's quartic algorithm for 1/pi (Jonathan and Peter Borwein, 1987). Like
// the Gauss-Legendre AGM iteration but faster, it *quadruples* the number of
// correct digits every step -- quartic convergence -- so a handful of iterations
// suffice for thousands of digits (in exact arithmetic). Starting from
//
//   y0 = sqrt(2) - 1,   a0 = 6 - 4 sqrt(2),
//
// it iterates
//
//   y_{k+1} = (1 - (1 - y_k^4)^{1/4}) / (1 + (1 - y_k^4)^{1/4}),
//   a_{k+1} = a_k (1 + y_{k+1})^4 - 2^{2k+3} y_{k+1} (1 + y_{k+1} + y_{k+1}^2),
//
// and a_k -> 1/pi. Evaluated here in double precision, so the estimate saturates
// the ~15-16 significant digits a double holds after only two or three steps.

#include <cmath>
#include <vector>

namespace datamunge::algorithms {

// Estimate of pi after `iterations` Borwein quartic steps (0 returns the seed).
inline double borwein_pi(int iterations) {
    const double s = std::sqrt(2.0);
    double       y = s - 1.0;
    double       a = 6.0 - 4.0 * s;
    for (int k = 0; k < iterations; ++k) {
        const double r  = std::pow(1.0 - y * y * y * y, 0.25);
        const double yn = (1.0 - r) / (1.0 + r);
        const double p1 = (1.0 + yn) * (1.0 + yn);
        const double p4 = p1 * p1;
        a               = a * p4 - std::ldexp(1.0, 2 * k + 3) * yn * (1.0 + yn + yn * yn);
        y               = yn;
    }
    return 1.0 / a;
}

// The successive pi estimates a_0..a_iterations (index i = after i iterations),
// handy for showing how fast the error collapses.
inline std::vector<double> borwein_pi_sequence(int iterations) {
    const double        s = std::sqrt(2.0);
    double              y = s - 1.0;
    double              a = 6.0 - 4.0 * s;
    std::vector<double> est{1.0 / a};
    for (int k = 0; k < iterations; ++k) {
        const double r  = std::pow(1.0 - y * y * y * y, 0.25);
        const double yn = (1.0 - r) / (1.0 + r);
        const double p1 = (1.0 + yn) * (1.0 + yn);
        const double p4 = p1 * p1;
        a               = a * p4 - std::ldexp(1.0, 2 * k + 3) * yn * (1.0 + yn + yn * yn);
        y               = yn;
        est.push_back(1.0 / a);
    }
    return est;
}

} // namespace datamunge::algorithms
