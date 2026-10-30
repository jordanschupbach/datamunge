/// @file elementary.cpp
/// @brief Kahan summation, Newton-Raphson division and roots, alpha max plus beta min.

#include <datamunge/algorithms/elementary.hpp>

#include <algorithm>
#include <cmath>
#include <numbers>
#include <stdexcept>
#include <vector>

namespace datamunge::algorithms {

double kahan_sum(const std::vector<double>& xs) {
    double sum = 0.0;
    double c   = 0.0; // running compensation for lost low-order bits
    for (double x : xs) {
        const double y = x - c;       // recover what was dropped last time
        const double t = sum + y;     // ... but this addition loses low bits of y
        c              = (t - sum) - y; // (t - sum) recovers the high part; subtracting y leaves the lost part
        sum            = t;
    }
    return sum;
}

double newton_raphson_division(double numerator, double denominator) {
    const double sign = denominator < 0.0 ? -1.0 : 1.0;
    double       d    = std::fabs(denominator);

    // Scale d into [0.5, 1) so a fixed linear seed converges: d = m * 2^e.
    int          e = 0;
    const double m = std::frexp(d, &e); // m in [0.5, 1)

    // Optimal linear initial guess for 1/m on [0.5, 1): x0 = 48/17 - 32/17 * m.
    double x = 48.0 / 17.0 - (32.0 / 17.0) * m;
    // Newton reciprocal iteration x <- x (2 - m x); doubles correct bits each step.
    for (int i = 0; i < 60; ++i) {
        const double xn = x * (2.0 - m * x);
        if (std::fabs(xn - x) <= 1e-17 * std::fabs(xn)) {
            x = xn;
            break;
        }
        x = xn;
    }
    const double recip = sign * std::ldexp(x, -e); // 1/denominator
    return numerator * recip;
}

double nth_root(double a, int n) {
    if (n <= 1) return a;
    if (a == 0.0) return 0.0;

    // Initial guess from the binary exponent: a = m * 2^e  =>  a^(1/n) ~ 2^(e/n).
    int e = 0;
    (void)std::frexp(a, &e);
    double x = std::ldexp(1.0, e / n);
    if (x <= 0.0) x = 1.0;

    // Newton on x^n - a = 0:  x <- ((n-1) x + a / x^(n-1)) / n.
    for (int i = 0; i < 100; ++i) {
        double pow_nm1 = 1.0;
        for (int k = 0; k < n - 1; ++k) pow_nm1 *= x; // x^(n-1)
        const double xn = ((n - 1) * x + a / pow_nm1) / n;
        if (std::fabs(xn - x) <= 1e-15 * std::fabs(xn)) {
            x = xn;
            break;
        }
        x = xn;
    }
    return x;
}

double alpha_max_beta_min(double x, double y) {
    static const AlphaBetaCoefficients c = alpha_max_beta_min_minimax();
    return alpha_max_beta_min(x, y, c);
}

double alpha_max_beta_min(double x, double y, const AlphaBetaCoefficients& c) {
    const double ax = std::fabs(x);
    const double ay = std::fabs(y);
    const double hi = ax > ay ? ax : ay;
    const double lo = ax > ay ? ay : ax;
    return c.alpha * hi + c.beta * lo;
}

AlphaBetaCoefficients alpha_max_beta_min_minimax() {
    // Ratio alpha cos(t) + beta sin(t) = A cos(t - phi) on [0, pi/4]: equal errors at both ends
    // put phi = pi/8, and A cos(pi/8) = 1 - e, A = 1 + e gives A = 2 / (1 + cos(pi/8)).
    const double c = std::cos(std::numbers::pi / 8), sn = std::sin(std::numbers::pi / 8);
    const double A = 2.0 / (1.0 + c);
    return {A * c, A * sn};
}

AlphaBetaCoefficients alpha_max_beta_min_no_underestimate() {
    return {1.0, std::tan(std::numbers::pi / 8)};
}

AlphaBetaCoefficients alpha_max_beta_min_least_squares() {
    // Normal equations from the integrals of cos^2, sin cos, sin^2, cos and sin over [0, pi/4].
    const double q = std::numbers::pi / 8, r = std::numbers::sqrt2 / 2;
    const double cc = q + 0.25, ss = q - 0.25, sc = 0.25, c1 = r, s1 = 1.0 - r;
    const double det = cc * ss - sc * sc;
    return {(c1 * ss - s1 * sc) / det, (s1 * cc - c1 * sc) / det};
}

AlphaBetaErrorRange alpha_max_beta_min_error(const AlphaBetaCoefficients& c) {
    const double q   = std::numbers::pi / 4;
    auto         f   = [&](double t) { return c.alpha * std::cos(t) + c.beta * std::sin(t); };
    double       lo  = std::min(f(0.0), f(q));
    double       hi  = std::max(f(0.0), f(q));
    const double phi = std::atan2(c.beta, c.alpha);
    if (phi > 0.0 && phi < q) {
        lo = std::min(lo, f(phi));
        hi = std::max(hi, f(phi));
    }
    return {lo, hi, std::max(std::fabs(lo - 1.0), std::fabs(hi - 1.0))};
}

std::vector<AlphaBetaCoefficients> alpha_max_beta_min_polygon(int lines) {
    if (lines < 1) throw std::invalid_argument("alpha_max_beta_min_polygon: lines must be at least 1");
    const double half = std::numbers::pi / (8.0 * lines);  // half the sector width
    const double A    = 2.0 / (1.0 + std::cos(half));
    std::vector<AlphaBetaCoefficients> out;
    for (int j = 0; j < lines; ++j) {
        const double phi = (2 * j + 1) * half;  // sector centre
        out.push_back({A * std::cos(phi), A * std::sin(phi)});
    }
    return out;
}

double alpha_max_beta_min(double x, double y, const std::vector<AlphaBetaCoefficients>& lines) {
    if (lines.empty()) throw std::invalid_argument("alpha_max_beta_min: need at least one line");
    double best = alpha_max_beta_min(x, y, lines.front());
    for (std::size_t j = 1; j < lines.size(); ++j) best = std::max(best, alpha_max_beta_min(x, y, lines[j]));
    return best;
}

} // namespace datamunge::algorithms
