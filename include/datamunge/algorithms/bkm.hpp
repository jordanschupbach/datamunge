#pragma once

// BKM algorithm (Bajard, Kla, Muller, 1994): computes the elementary functions
// ln and exp using only additions, shifts, and a small precomputed table of
// logarithms ln(1 + 2^-k). Like CORDIC (which uses arctangents), BKM builds the
// result one bit at a time by conditionally multiplying by (1 + 2^-k) -- a shift
// and add -- and accumulating the matching table entry.

#include <cmath>
#include <vector>

namespace datamunge::algorithms {

namespace detail {

// Table of ln(1 + 2^-k), the "logarithms" BKM iterates over.
inline const std::vector<double>& bkm_table() {
    static const std::vector<double> t = [] {
        std::vector<double> v(53);
        for (int k = 0; k < 53; ++k) v[k] = std::log1p(std::ldexp(1.0, -k)); // ln(1 + 2^-k)
        return v;
    }();
    return t;
}

} // namespace detail

// L-mode: ln(v) for v in [1, ~4.768]. Multiply a running product by (1 + 2^-k)
// whenever it keeps the product <= v; the product converges to v and the summed
// table entries converge to ln(v).
inline double bkm_ln(double v) {
    const auto& table = detail::bkm_table();
    double      x = 1.0, y = 0.0;
    for (int k = 0; k < static_cast<int>(table.size()); ++k) {
        const double xn = x + std::ldexp(x, -k); // x * (1 + 2^-k) via a shift-add
        if (xn <= v) { x = xn; y += table[k]; }
    }
    return y;
}

// E-mode: exp(y) for y in [0, ~1.5622]. Add table[k] to a running sum whenever it
// stays <= y, and mirror it by multiplying a running product by (1 + 2^-k); the
// sum converges to y and the product to exp(y).
inline double bkm_exp(double y) {
    const auto& table = detail::bkm_table();
    double      x = 1.0, s = 0.0;
    for (int k = 0; k < static_cast<int>(table.size()); ++k) {
        if (s + table[k] <= y) { s += table[k]; x += std::ldexp(x, -k); } // shift-add product
    }
    return x;
}

} // namespace datamunge::algorithms
