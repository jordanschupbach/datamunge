#pragma once

// Birkhoff (lacunary) interpolation: find the polynomial that matches a set of
// prescribed *derivative values* at given nodes, where -- unlike Hermite
// interpolation -- the derivative orders specified at each node need not be a
// consecutive block starting at 0. For example one may fix p(x0), p''(x0),
// p(x1), p''(x1) while leaving the first derivatives free.
//
// With N constraints the interpolant is a degree-(N-1) polynomial
// p(x) = sum_k c_k x^k. Each constraint p^{(d)}(x_i) = v contributes one linear
// equation in the coefficients, since
//
//   d^d/dx^d [x^k] = k!/(k-d)! * x^{k-d}   (0 for k < d).
//
// Solving the resulting N x N system gives the coefficients -- provided the
// scheme is *poised* (the matrix is nonsingular); Birkhoff schemes, unlike
// Lagrange/Hermite, can be singular for some node/order patterns, in which case
// no unique interpolant exists and this returns an empty vector.

#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

namespace datamunge::algorithms {

// One interpolation condition: p^{(order)}(node) = value.
struct BirkhoffConstraint {
    double node{0};
    int    order{0};
    double value{0};
};

namespace detail {

// Coefficient of c_k in the `order`-th derivative of x^k evaluated at `node`:
// the falling factorial k*(k-1)*...*(k-order+1) times node^{k-order}.
inline double birkhoff_entry(int k, int order, double node) {
    if (k < order) return 0.0;
    double falling = 1.0;
    for (int t = 0; t < order; ++t) falling *= static_cast<double>(k - t);
    return falling * std::pow(node, static_cast<double>(k - order));
}

} // namespace detail

// Polynomial coefficients c_0..c_{N-1} (lowest degree first) satisfying every
// constraint, or an empty vector if the scheme is not poised (singular system).
inline std::vector<double> birkhoff_polynomial(const std::vector<BirkhoffConstraint>& cons) {
    const std::size_t N = cons.size();
    if (N == 0) return {};
    // Augmented matrix [A | b], A[i][k] = entry(k, order_i, node_i), b[i] = value_i.
    std::vector<std::vector<double>> M(N, std::vector<double>(N + 1, 0.0));
    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t k = 0; k < N; ++k)
            M[i][k] = detail::birkhoff_entry(static_cast<int>(k), cons[i].order, cons[i].node);
        M[i][N] = cons[i].value;
    }
    // Gaussian elimination with partial pivoting.
    for (std::size_t col = 0; col < N; ++col) {
        std::size_t piv = col;
        double      best = std::fabs(M[col][col]);
        for (std::size_t r = col + 1; r < N; ++r)
            if (std::fabs(M[r][col]) > best) { best = std::fabs(M[r][col]); piv = r; }
        if (best < 1e-14) return {}; // singular -> not poised
        std::swap(M[col], M[piv]);
        const double d = M[col][col];
        for (std::size_t r = 0; r < N; ++r) {
            if (r == col) continue;
            const double f = M[r][col] / d;
            if (f == 0.0) continue;
            for (std::size_t c = col; c <= N; ++c) M[r][c] -= f * M[col][c];
        }
    }
    std::vector<double> c(N);
    for (std::size_t i = 0; i < N; ++i) c[i] = M[i][N] / M[i][i];
    return c;
}

// Evaluate a coefficient vector (lowest degree first) at x by Horner's rule.
inline double birkhoff_eval(const std::vector<double>& coeffs, double x) {
    double acc = 0.0;
    for (std::size_t i = coeffs.size(); i-- > 0;) acc = acc * x + coeffs[i];
    return acc;
}

} // namespace datamunge::algorithms
