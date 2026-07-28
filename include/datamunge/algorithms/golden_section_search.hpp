#pragma once

// Golden-section search: locate the maximum of a unimodal function on [a, b] by
// repeatedly shrinking the bracket using two interior probes placed at the golden
// ratio. Each iteration reuses one probe, so it needs only one new function
// evaluation and reduces the interval by the factor 1/phi ≈ 0.618 every step.

#include <cmath>

namespace datamunge::algorithms {

struct GoldenResult {
    double argmax{0};
    double value{0};
    int    iterations{0};
};

// Maximise `f` (assumed unimodal) over [a, b] to interval width `tol`.
template <typename F>
GoldenResult golden_section_search(F f, double a, double b, double tol = 1e-9, int max_iter = 500) {
    const double invphi = (std::sqrt(5.0) - 1.0) / 2.0; // 1/phi ≈ 0.618
    double       c = b - invphi * (b - a);
    double       d = a + invphi * (b - a);
    double       fc = f(c), fd = f(d);
    int          it = 0;
    while (b - a > tol && it < max_iter) {
        if (fc < fd) {           // maximum lies in [c, b]
            a = c;
            c = d; fc = fd;
            d = a + invphi * (b - a); fd = f(d);
        } else {                 // maximum lies in [a, d]
            b = d;
            d = c; fd = fc;
            c = b - invphi * (b - a); fc = f(c);
        }
        ++it;
    }
    const double x = 0.5 * (a + b);
    return {x, f(x), it};
}

} // namespace datamunge::algorithms
