#pragma once

#include <datamunge/algebra/expression.hpp>
#include <datamunge/algebra/polynomial.hpp>

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace datamunge::algebra {

namespace detail {

[[nodiscard]] inline double simpson_estimate(const Expr& f, const std::string& var, double a, double b) {
    const double fa = f.evaluate({var}, {a});
    const double fb = f.evaluate({var}, {b});
    const double fm = f.evaluate({var}, {(a + b) / 2.0});
    return (b - a) / 6.0 * (fa + 4.0 * fm + fb);
}

// Classical adaptive Simpson's rule (McKeeman 1962): compares the whole-interval estimate
// against the sum of its two half-interval estimates; their difference gives a cheap
// Richardson-extrapolation error estimate (the true error is close to 1/15th of that
// difference), so the interval is only subdivided further where the two disagree by more than
// `tol`, concentrating work on the parts of [a, b] where f actually varies quickly.
[[nodiscard]] inline double adaptive_simpson_recurse(const Expr& f, const std::string& var, double a, double b,
                                                      double whole, double tol, int depth) {
    const double m = (a + b) / 2.0;
    const double left = simpson_estimate(f, var, a, m);
    const double right = simpson_estimate(f, var, m, b);
    const double delta = left + right - whole;
    if (depth <= 0 || std::fabs(delta) <= 15.0 * tol) return left + right + delta / 15.0;
    return adaptive_simpson_recurse(f, var, a, m, left, tol / 2.0, depth - 1) +
           adaptive_simpson_recurse(f, var, m, b, right, tol / 2.0, depth - 1);
}

} // namespace detail

/// @brief The definite integral of f (with every other name in f bound as usual by evaluate())
///        from a to b with respect to var, via adaptive Simpson's rule -- a numeric fallback/
///        complement to Expr::integrate() for cases outside that method's supported symbolic
///        forms (e.g. sin(x)*cos(x), or any other integrand it throws on), or simply to check
///        a symbolic result. Handles a > b by returning the negated integral over [b, a].
/// @param tolerance Target absolute error per the adaptive refinement's own error estimate; not
///        a hard guarantee for a pathological (e.g. discontinuous or highly oscillatory)
///        integrand.
/// @param max_depth Hard cap on recursive bisection depth, bounding worst-case cost.
[[nodiscard]] inline double definite_integral(const Expr& f, const std::string& var, double a, double b,
                                               double tolerance = 1e-9, int max_depth = 50) {
    if (a == b) return 0.0;
    if (a > b) return -definite_integral(f, var, b, a, tolerance, max_depth);
    const double whole = detail::simpson_estimate(f, var, a, b);
    return detail::adaptive_simpson_recurse(f, var, a, b, whole, tolerance, max_depth);
}

/// @brief The degree-`order` Taylor polynomial of f around var = center, built from `order`
///        repeated symbolic differentiations of f (each evaluated at center and divided by the
///        matching factorial) -- the returned polynomial q approximates f near center as
///        q.evaluate(x - center), NOT q.evaluate(x) directly (q's coefficients are for powers
///        of (x - center), matching the standard Taylor series form
///        f(x) ~= sum_k f^(k)(center)/k! * (x-center)^k).
/// @throws std::invalid_argument if order is negative, or std::invalid_argument propagated from
///         Expr::differentiate() if f contains a non-constant exponent.
[[nodiscard]] inline Polynomial taylor_series(const Expr& f, const std::string& var, double center, int order) {
    if (order < 0) throw std::invalid_argument("taylor_series: order must be nonnegative");

    std::vector<double> coeffs(static_cast<std::size_t>(order) + 1);
    Expr current = f;
    double factorial = 1.0;
    for (int k = 0; k <= order; ++k) {
        coeffs[static_cast<std::size_t>(k)] = current.evaluate({var}, {center}) / factorial;
        if (k < order) {
            current = current.differentiate(var).simplify();
            factorial *= static_cast<double>(k + 1);
        }
    }
    return Polynomial(std::move(coeffs));
}

} // namespace datamunge::algebra
