#include <datamunge/algorithms/interpolation.hpp>

#include <algorithm>
#include <cstddef>
#include <vector>

namespace datamunge::algorithms {

double lagrange_interpolate(const std::vector<double>& xs, const std::vector<double>& ys, double x) {
    const std::size_t n   = xs.size();
    double            sum = 0.0;
    for (std::size_t j = 0; j < n; ++j) {
        double term = ys[j];
        for (std::size_t k = 0; k < n; ++k)
            if (k != j) term *= (x - xs[k]) / (xs[j] - xs[k]);
        sum += term;
    }
    return sum;
}

double neville_interpolate(const std::vector<double>& xs, const std::vector<double>& ys, double x) {
    const std::size_t   n = xs.size();
    std::vector<double> p(ys); // p[j] holds the running interpolant estimate ending at node j
    for (std::size_t i = 1; i < n; ++i)
        for (std::size_t j = n - 1; j >= i; --j) {
            p[j] = ((x - xs[j - i]) * p[j] - (x - xs[j]) * p[j - 1]) / (xs[j] - xs[j - i]);
            if (j == i) break; // avoid unsigned underflow
        }
    return n == 0 ? 0.0 : p[n - 1];
}


} // namespace datamunge::algorithms
