#pragma once
/// @file noncentral_t_kernels.hpp
/// @brief Log-radius quadrature for the normal/chi-square representation of noncentral t.

#include "beta_kernels.hpp"

#include <array>
#include <limits>
#include <numbers>
#include <numeric>
#include <utility>

namespace datamunge::random::detail {

/// @brief Log standard normal density at the origin.
inline constexpr auto log_phi_zero = -0.91893853320467274178032973640562L;

/// @brief Computes the log normal CDF, preserving underflowed left tails.
/// @param z Standard normal coordinate.
/// @return Natural logarithm of the cumulative probability.
inline long double noncentral_t_log_normal(long double z) {
  const auto tail = std::erfc(std::abs(z) / std::numbers::sqrt2_v<long double>) / 2;
  if (z >= 0)
    return std::log1p(-tail);
  if (tail >= std::numeric_limits<long double>::min())
    return std::log(tail);
  const auto r = (1 / z) / z;
  return log_phi_zero - z * z / 2 - std::log(-z) +
         std::log1p(r * (-1 + r * (3 - 15 * r)));
}

/// @brief Evaluates a normal interval probability directly on the log scale.
/// @param lo Lower normal coordinate.
/// @param hi Upper normal coordinate, when rounded relative to lo.
/// @param width Unrounded distance between the endpoints.
/// @param log_width Logarithm of the distance, including underflowed widths.
/// @return Logarithm of the normal mass between the endpoints.
inline long double normal_interval_log(long double lo, long double hi, long double width, long double log_width) {
  if (width * (1 + std::abs(lo)) < .001L) {
    // Three-point Gaussian quadrature after factoring out the density
    // at the lower end. The distance and the small exponent stay distinct.
    constexpr auto shift = .387298334620741688517926539978L;
    /// @brief Normal density relative to its value at the lower endpoint.
    const auto density_ratio = [&](long double t) { return std::exp(-t * (lo + t / 2)); };
    const auto average = (5 * (density_ratio((.5L - shift) * width) +
                               density_ratio((.5L + shift) * width)) + 8 * density_ratio(width / 2)) / 18;
    return log_width + log_phi_zero - lo * lo / 2 + std::log(average);
  }
  const auto large = lo >= 0 ? noncentral_t_log_normal(-lo) : noncentral_t_log_normal(hi);
  const auto small = lo >= 0 ? noncentral_t_log_normal(-hi) : noncentral_t_log_normal(lo);
  if (small >= large) {
    // If the endpoints coincide in normal-tail arithmetic despite a
    // material density gradient, both are far into the same tail.
    const auto nearest = std::min(std::abs(lo), std::abs(hi));
    return log_width + log_phi_zero - nearest * nearest / 2;
  }
  return large + std::log(-std::expm1(small - large));
}

/// @brief Stable value of exp(x)-1-x.
/// @param x Exponential argument.
/// @return Nonnegative remainder in widened precision.
inline long double exponential_deviance(long double x) {
  return std::abs(x) < .001L ? x * x * (.5L + x * (1.L / 6 + x * (1.L / 24 + x * (1.L / 120 + x / 720))))
                           : std::expm1(x) - x;
}

/// @brief Log-radius integrand for one oriented noncentral t tail.
struct NoncentralTKernel {
  long double x;          ///< Positive cutoff after reflection.
  long double df;         ///< Positive degrees of freedom.
  long double delta;      ///< Signed normal displacement.
  long double log_x;      ///< Logarithm of the positive cutoff.
  long double log_norm;   ///< Log normalization of the log-radius density.
  bool lower;             ///< Whether to integrate lower-tail mass above the negative-numerator probability.

  /// @brief Returns the log integrand and its signed log-radius derivative.
  /// @param s Logarithm of sqrt(V / df).
  /// @return Log integrand and derivative of the log integrand.
  std::pair<long double, long double> evaluate(long double s) const {
    const auto log_width = log_x + s;
    const auto width = std::exp(log_width);
    const auto z = std::abs(s) < .5L ? std::fma(x, std::expm1(s), x - delta) : width - delta;
    const auto log_normal = lower ? normal_interval_log(-delta, z, width, log_width)
                                  : noncentral_t_log_normal(-z);
    if (!std::isfinite(log_normal)) {
      constexpr auto infinity = std::numeric_limits<long double>::infinity();
      return {-infinity, lower ? infinity : -infinity};
    }
    const auto log_slope = log_width + log_phi_zero - z * z / 2 - log_normal;
    const auto normal_slope = std::exp(log_slope);
    return {log_norm - (df / 2) * exponential_deviance(2 * s) + log_normal,
            -df * std::expm1(2 * s) + (lower ? normal_slope : -normal_slope)};
  }
};

/// @brief Integrates a smooth positive function using an embedded 7/15 Gauss–Kronrod rule.
/// @tparam Function Nonnegative integrand callable.
/// @param f Integrand on the interval.
/// @param a Left endpoint.
/// @param b Right endpoint.
/// @param depth Remaining adaptive refinements.
/// @return Quadrature estimate.
template <class Function>
long double nct_integrate(const Function& f, long double a, long double b, int depth) {
  /// @brief Nonnegative Kronrod abscissae, ordered from outside to center.
  constexpr std::array nodes{.991455371120812639206854697526329L, .949107912342758524526189684047851L,
      .864864423359769072789712788640926L, .741531185599394439863864773280788L,
      .586087235467691130294144838258730L, .405845151377397166906606412076961L,
      .207784955007898467600689403773245L, 0.L};
  /// @brief Kronrod rule weights.
  constexpr std::array kronrod{.022935322010529224963732008058970L, .063092092629978553290700663189204L,
      .104790010322250183839876322541518L, .140653259715525918745189590510238L,
      .169004726639267902826583426598550L, .190350578064785409913256402421014L,
      .204432940075298892414161999234649L, .209482141084727828012999174891714L};
  /// @brief Gauss rule weights at alternate Kronrod nodes.
  constexpr std::array gauss{.129484966168869693270611432679082L, .279705391489276667901467771423780L,
      .381830050505118944950369775488975L, .417959183673469387755102040816327L};
  const auto center = std::midpoint(a, b);
  if (a == b)
    return 0;
  const auto half = (b - a) / 2;
  const auto mid_value = f(center);
  auto fine = kronrod.back() * mid_value, coarse = gauss.back() * mid_value;
  for (std::size_t i = 0; i + 1 < nodes.size(); ++i) {
    const auto offset = half * nodes[i];
    const auto sum = f(center - offset) + f(center + offset);
    fine += kronrod[i] * sum;
    if (i % 2 != 0)
      coarse += gauss[i / 2] * sum;
  }
  if (depth == 0 || center == a || center == b || std::abs(fine - coarse) <= 2e-13L * (fine + 1e-18L))
    return half * fine;
  return nct_integrate(f, a, center, depth - 1) + nct_integrate(f, center, b, depth - 1);
}

/// @brief Bisects between a point satisfying a predicate and one violating it.
/// @tparam Predicate Callable deciding which half to keep.
/// @param inside Coordinate satisfying the predicate.
/// @param outside Bracketing coordinate on the opposite side.
/// @param contained Predicate that changes once within the bracket.
/// @return Coordinate near the transition.
template <class Predicate>
long double nct_boundary(long double inside, long double outside, const Predicate& contained) {
  for (int i = 0; i < 72; ++i) {
    const auto mid = std::midpoint(inside, outside);
    if (mid == inside || mid == outside)
      break;
    (contained(mid) ? inside : outside) = mid;
  }
  return std::midpoint(inside, outside);
}

/// @brief Integrates one noncentral t tail for a positive cutoff.
/// @param x Positive cutoff.
/// @param df Positive degrees of freedom.
/// @param delta Signed numerator displacement.
/// @param lower Whether to return the lower tail.
/// @return Selected direct tail probability.
inline long double noncentral_t_integrated_tail(long double x, long double df, long double delta, bool lower) {
  const auto base = lower ? std::exp(noncentral_t_log_normal(-delta)) : 0.L;
  if (base == 1)
    return 1;
  const auto a = df / 2;
  const auto log_norm = std::numbers::ln2_v<long double> +
      (a < 16 ? a * (std::log(a) - 1) - std::lgamma(a)
              : .5L * std::log(a / (2 * std::numbers::pi_v<long double>)) - stirling_remainder(a));
  const NoncentralTKernel kernel{x, df, delta, std::log(x), log_norm, lower};
  const auto step = std::min(1.L, 1 / std::sqrt(df));
  const auto left = std::asinh(-(1000 / df + 40 / std::sqrt(df)) / step);
  const auto right = std::asinh(std::log1p(std::sqrt(2000 / df)) / step);
  /// @brief Maps the bounded search coordinate onto the unrestricted log radius.
  const auto radius = [=](long double u) { return step * std::sinh(u); };
  const auto mode = nct_boundary(left, right, [&](long double u) { return kernel.evaluate(radius(u)).second >= 0; });
  const auto peak = kernel.evaluate(radius(mode)).first;
  const auto span = radius(right) - radius(left);
  if (!std::isfinite(peak) ||
      peak + std::log(span) < std::log(static_cast<long double>(std::numeric_limits<double>::denorm_min())) - 2)
    return base;

  /// @brief Scales the integrand to unit height at the mode.
  const auto scaled = [&](long double s) { return std::exp(kernel.evaluate(s).first - peak); };
  long double integral = 0;
  for (const auto limit : {left, right}) {
    auto inner = mode;
    for (const long double drop : {4, 16, 48}) {
      const auto outer = nct_boundary(inner, limit, [&](long double u) {
        return kernel.evaluate(radius(u)).first >= peak - drop;
      });
      const auto low = radius(inner), high = radius(outer);
      integral += nct_integrate(scaled, std::min(low, high), std::max(low, high), 12);
      inner = outer;
    }
  }
  return std::clamp(base + std::exp(peak + std::log(integral)), 0.L, 1.L);
}

} // namespace datamunge::random::detail
