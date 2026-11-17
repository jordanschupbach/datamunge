/// @file elementary.cpp
/// @brief Kahan summation, Newton-Raphson division and roots, alpha max plus beta min.

#include <datamunge/algorithms/elementary.hpp>

#include <algorithm>
#include <cmath>
#include <numbers>
#include <stdexcept>
#include <vector>

namespace datamunge::algorithms {

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
