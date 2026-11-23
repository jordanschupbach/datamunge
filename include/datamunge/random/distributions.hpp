#pragma once

/// @file distributions.hpp
/// @brief Numerical densities, cumulative probabilities and quantiles for probability laws.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace datamunge::random {

/// @brief Internal numerical kernels and probability parameter checks.
namespace detail {

inline constexpr double pi = 3.14159265358979323846;
inline constexpr double sqrt_two = 1.41421356237309504880;
inline constexpr double inv_sqrt_two_pi = 0.39894228040143267794;

inline void require_probability(double p, const char* fn) {
  if (!(p >= 0.0 && p <= 1.0)) {
    throw std::invalid_argument(std::string(fn) + ": probability must be in [0, 1]");
  }
}

inline void require_positive(double x, const char* fn, const char* name) {
  if (!(x > 0.0)) {
    throw std::invalid_argument(std::string(fn) + ": " + name + " must be > 0");
  }
}

inline void require_nonnegative(double x, const char* fn, const char* name) {
  if (!(x >= 0.0)) {
    throw std::invalid_argument(std::string(fn) + ": " + name + " must be >= 0");
  }
}

inline double normal_quantile_unit(double p) {
  require_probability(p, "normal_quantile");
  if (p == 0.0) return -std::numeric_limits<double>::infinity();
  if (p == 1.0) return std::numeric_limits<double>::infinity();

  static constexpr double a1 = -3.969683028665376e+01;
  static constexpr double a2 = 2.209460984245205e+02;
  static constexpr double a3 = -2.759285104469687e+02;
  static constexpr double a4 = 1.383577518672690e+02;
  static constexpr double a5 = -3.066479806614716e+01;
  static constexpr double a6 = 2.506628277459239e+00;

  static constexpr double b1 = -5.447609879822406e+01;
  static constexpr double b2 = 1.615858368580409e+02;
  static constexpr double b3 = -1.556989798598866e+02;
  static constexpr double b4 = 6.680131188771972e+01;
  static constexpr double b5 = -1.328068155288572e+01;

  static constexpr double c1 = -7.784894002430293e-03;
  static constexpr double c2 = -3.223964580411365e-01;
  static constexpr double c3 = -2.400758277161838e+00;
  static constexpr double c4 = -2.549732539343734e+00;
  static constexpr double c5 = 4.374664141464968e+00;
  static constexpr double c6 = 2.938163982698783e+00;

  static constexpr double d1 = 7.784695709041462e-03;
  static constexpr double d2 = 3.224671290700398e-01;
  static constexpr double d3 = 2.445134137142996e+00;
  static constexpr double d4 = 3.754408661907416e+00;

  static constexpr double plow = 0.02425;
  static constexpr double phigh = 1.0 - plow;

  if (p < plow) {
    const double q = std::sqrt(-2.0 * std::log(p));
    return (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6)
         / ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0);
  }
  if (p > phigh) {
    const double q = std::sqrt(-2.0 * std::log(1.0 - p));
    return -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6)
         / ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0);
  }

  const double q = p - 0.5;
  const double r = q * q;
  return (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q
       / (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0);
}

inline double poisson_pdf_nonneg(std::size_t k, double lambda) {
  return std::exp(-lambda + static_cast<double>(k) * std::log(lambda) - std::lgamma(static_cast<double>(k) + 1.0));
}

inline double log_beta(double a, double b) {
  return std::lgamma(a) + std::lgamma(b) - std::lgamma(a + b);
}

/// @brief Evaluates the incomplete-beta continued fraction on its lower-tail branch.
/// @tparam Real Floating-point working type.
/// @param a First positive beta shape.
/// @param b Second positive beta shape.
/// @param x Argument below (a+1)/(a+b+2).
/// @param max_iterations Iteration limit, default 200.
/// @return Continued fraction before multiplying by the beta prefactor over a.
template <typename Real>
inline Real incomplete_beta_cf(Real a, Real b, Real x, int max_iterations = 200) {
  const auto epsilon = 2 * std::numeric_limits<Real>::epsilon();
  const auto tiny = 16 * std::numeric_limits<Real>::min();

  const Real qab = a + b;
  const Real qap = a + 1.0;
  const Real qam = a - 1.0;

  Real c = 1.0;
  Real d = 1 - (qab / qap) * x;
  if (std::abs(d) < tiny) d = tiny;
  d = 1.0 / d;
  Real h = d;

  for (int m = 1; m <= max_iterations; ++m) {
    const Real m2 = 2.0 * m;

    Real aa = (m / (a + m2)) * ((b - m) / (qam + m2)) * x;
    d = 1.0 + aa * d; if (std::abs(d) < tiny) d = tiny;
    c = 1.0 + aa / c; if (std::abs(c) < tiny) c = tiny;
    d = 1.0 / d;
    h *= d * c;

    aa = -((a + m) / (a + m2)) * ((qab + m) / (qap + m2)) * x;
    d = 1.0 + aa * d; if (std::abs(d) < tiny) d = tiny;
    c = 1.0 + aa / c; if (std::abs(c) < tiny) c = tiny;
    d = 1.0 / d;
    const Real delta = d * c;
    h *= delta;

    if (std::abs(delta - 1.0) < epsilon) break;
  }
  return h;
}

// Regularized incomplete beta function I_x(a, b), used to derive the
// Student-t and F cumulative distribution functions from the incomplete
// beta relation (Abramowitz & Stegun 26.5.28 / 26.6.2).
inline double regularized_incomplete_beta(double a, double b, double x) {
  if (!(a > 0.0) || !(b > 0.0))
    throw std::invalid_argument("regularized_incomplete_beta: a and b must be > 0");
  if (x < 0.0 || x > 1.0)
    throw std::invalid_argument("regularized_incomplete_beta: x must be in [0, 1]");
  if (x == 0.0 || x == 1.0) return x;

  const double front = std::exp(a * std::log(x) + b * std::log1p(-x) - log_beta(a, b));
  if (x < (a + 1.0) / (a + b + 2.0))
    return front * incomplete_beta_cf(a, b, x) / a;
  return 1.0 - front * incomplete_beta_cf(b, a, 1.0 - x) / b;
}

// Bisection on a monotonically increasing CDF, expanding the bracket until
// it contains p. Used for quantile functions without a closed form.
template <typename CdfFn>
inline double invert_monotonic_cdf(double p, CdfFn cdf, double lo, double hi) {
  while (cdf(lo) > p) lo *= 2.0;
  while (cdf(hi) < p) hi *= 2.0;
  for (int iteration = 0; iteration < 200; ++iteration) {
    const double mid = 0.5 * (lo + hi);
    if (cdf(mid) < p) lo = mid; else hi = mid;
    if (hi - lo < 1e-12 * (std::abs(hi) + 1.0)) break;
  }
  return 0.5 * (lo + hi);
}

/// @brief Series expansion for the regularized lower incomplete gamma function.
/// @param a Positive shape.
/// @param x Nonnegative argument below a + 1.
/// @return Lower probability P(a,x).
inline double incomplete_gamma_series(double a, double x) {
  /// @brief Iteration ceiling with margin for large shape parameters.
  constexpr int    max_iterations = 10000;
  /// @brief Relative convergence threshold.
  constexpr double epsilon        = 3.0e-16;

  if (x <= 0.0) return 0.0;
  double term = 1.0 / a;
  double sum  = term;
  double ap   = a;
  for (int n = 0; n < max_iterations; ++n) {
    ap += 1.0;
    term *= x / ap;
    sum += term;
    if (std::abs(term) < std::abs(sum) * epsilon) break;
  }
  return sum * std::exp(-x + a * std::log(x) - std::lgamma(a));
}

/// @brief Continued-fraction expansion for the regularized upper incomplete gamma function.
/// @param a Positive shape.
/// @param x Positive argument, normally at least a + 1.
/// @return Upper probability Q(a,x).
inline double incomplete_gamma_cf(double a, double x) {
  /// @brief Iteration ceiling with margin for large shape parameters.
  constexpr int    max_iterations = 10000;
  /// @brief Relative convergence threshold.
  constexpr double epsilon        = 3.0e-16;
  /// @brief Floor for nonzero continued-fraction denominators.
  constexpr double tiny           = 1.0e-300;

  double b = x + 1.0 - a;
  double c = 1.0 / tiny;
  double d = 1.0 / b;
  double h = d;
  for (int i = 1; i <= max_iterations; ++i) {
    const double an = -static_cast<double>(i) * (static_cast<double>(i) - a);
    b += 2.0;
    d = an * d + b; if (std::abs(d) < tiny) d = tiny;
    c = b + an / c; if (std::abs(c) < tiny) c = tiny;
    d = 1.0 / d;
    const double delta = d * c;
    h *= delta;
    if (std::abs(delta - 1.0) < epsilon) break;
  }
  return std::exp(-x + a * std::log(x) - std::lgamma(a)) * h;
}

// Regularized lower incomplete gamma function P(a, x) = gamma(a, x) / Gamma(a), used to
// derive the chi-squared cumulative distribution function.
inline double regularized_lower_incomplete_gamma(double a, double x) {
  if (!(a > 0.0)) throw std::invalid_argument("regularized_lower_incomplete_gamma: a must be > 0");
  if (x < 0.0) throw std::invalid_argument("regularized_lower_incomplete_gamma: x must be >= 0");
  if (x == 0.0) return 0.0;
  if (x < a + 1.0) return incomplete_gamma_series(a, x);
  return 1.0 - incomplete_gamma_cf(a, x);
}

inline double log_choose(double n, double k) {
  return std::lgamma(n + 1.0) - std::lgamma(k + 1.0) - std::lgamma(n - k + 1.0);
}

} // namespace detail

inline double uniform_pdf(double x, double a = 0.0, double b = 1.0) {
  if (!(b > a)) throw std::invalid_argument("uniform_pdf: require b > a");
  return (x >= a && x <= b) ? 1.0 / (b - a) : 0.0;
}

inline double uniform_cdf(double x, double a = 0.0, double b = 1.0) {
  if (!(b > a)) throw std::invalid_argument("uniform_cdf: require b > a");
  if (x <= a) return 0.0;
  if (x >= b) return 1.0;
  return (x - a) / (b - a);
}

inline double uniform_quantile(double p, double a = 0.0, double b = 1.0) {
  detail::require_probability(p, "uniform_quantile");
  if (!(b > a)) throw std::invalid_argument("uniform_quantile: require b > a");
  return a + (b - a) * p;
}

inline double normal_pdf(double x, double mean = 0.0, double sd = 1.0) {
  detail::require_positive(sd, "normal_pdf", "sd");
  const double z = (x - mean) / sd;
  return detail::inv_sqrt_two_pi * std::exp(-0.5 * z * z) / sd;
}

inline double normal_cdf(double x, double mean = 0.0, double sd = 1.0) {
  detail::require_positive(sd, "normal_cdf", "sd");
  const double z = (x - mean) / (sd * detail::sqrt_two);
  return 0.5 * (1.0 + std::erf(z));
}

inline double normal_quantile(double p, double mean = 0.0, double sd = 1.0) {
  detail::require_positive(sd, "normal_quantile", "sd");
  return mean + sd * detail::normal_quantile_unit(p);
}

inline double lognormal_pdf(double x, double log_mean = 0.0, double log_sd = 1.0) {
  detail::require_positive(log_sd, "lognormal_pdf", "log_sd");
  if (x <= 0.0) return 0.0;
  const double z = (std::log(x) - log_mean) / log_sd;
  return detail::inv_sqrt_two_pi * std::exp(-0.5 * z * z) / (x * log_sd);
}

inline double lognormal_cdf(double x, double log_mean = 0.0, double log_sd = 1.0) {
  detail::require_positive(log_sd, "lognormal_cdf", "log_sd");
  if (x <= 0.0) return 0.0;
  return normal_cdf(std::log(x), log_mean, log_sd);
}

inline double lognormal_quantile(double p, double log_mean = 0.0, double log_sd = 1.0) {
  return std::exp(normal_quantile(p, log_mean, log_sd));
}

inline double exponential_pdf(double x, double rate = 1.0) {
  detail::require_positive(rate, "exponential_pdf", "rate");
  if (x < 0.0) return 0.0;
  return rate * std::exp(-rate * x);
}

inline double exponential_cdf(double x, double rate = 1.0) {
  detail::require_positive(rate, "exponential_cdf", "rate");
  if (x <= 0.0) return 0.0;
  return 1.0 - std::exp(-rate * x);
}

inline double exponential_quantile(double p, double rate = 1.0) {
  detail::require_probability(p, "exponential_quantile");
  detail::require_positive(rate, "exponential_quantile", "rate");
  if (p == 1.0) return std::numeric_limits<double>::infinity();
  return -std::log1p(-p) / rate;
}

inline double laplace_pdf(double x, double location = 0.0, double scale = 1.0) {
  detail::require_positive(scale, "laplace_pdf", "scale");
  return 0.5 / scale * std::exp(-std::abs(x - location) / scale);
}

inline double laplace_cdf(double x, double location = 0.0, double scale = 1.0) {
  detail::require_positive(scale, "laplace_cdf", "scale");
  if (x < location) return 0.5 * std::exp((x - location) / scale);
  return 1.0 - 0.5 * std::exp(-(x - location) / scale);
}

inline double laplace_quantile(double p, double location = 0.0, double scale = 1.0) {
  detail::require_probability(p, "laplace_quantile");
  detail::require_positive(scale, "laplace_quantile", "scale");
  if (p == 0.0) return -std::numeric_limits<double>::infinity();
  if (p == 1.0) return std::numeric_limits<double>::infinity();
  if (p < 0.5) return location + scale * std::log(2.0 * p);
  return location - scale * std::log(2.0 * (1.0 - p));
}

inline double logistic_pdf(double x, double location = 0.0, double scale = 1.0) {
  detail::require_positive(scale, "logistic_pdf", "scale");
  const double z = std::exp(-(x - location) / scale);
  const double d = 1.0 + z;
  return z / (scale * d * d);
}

inline double logistic_cdf(double x, double location = 0.0, double scale = 1.0) {
  detail::require_positive(scale, "logistic_cdf", "scale");
  return 1.0 / (1.0 + std::exp(-(x - location) / scale));
}

inline double logistic_quantile(double p, double location = 0.0, double scale = 1.0) {
  detail::require_probability(p, "logistic_quantile");
  detail::require_positive(scale, "logistic_quantile", "scale");
  if (p == 0.0) return -std::numeric_limits<double>::infinity();
  if (p == 1.0) return std::numeric_limits<double>::infinity();
  return location + scale * std::log(p / (1.0 - p));
}

inline double cauchy_pdf(double x, double location = 0.0, double scale = 1.0) {
  detail::require_positive(scale, "cauchy_pdf", "scale");
  const double z = (x - location) / scale;
  return 1.0 / (detail::pi * scale * (1.0 + z * z));
}

inline double cauchy_cdf(double x, double location = 0.0, double scale = 1.0) {
  detail::require_positive(scale, "cauchy_cdf", "scale");
  return 0.5 + std::atan((x - location) / scale) / detail::pi;
}

inline double cauchy_quantile(double p, double location = 0.0, double scale = 1.0) {
  detail::require_probability(p, "cauchy_quantile");
  detail::require_positive(scale, "cauchy_quantile", "scale");
  if (p == 0.0) return -std::numeric_limits<double>::infinity();
  if (p == 1.0) return std::numeric_limits<double>::infinity();
  return location + scale * std::tan(detail::pi * (p - 0.5));
}

inline double weibull_pdf(double x, double shape, double scale = 1.0) {
  detail::require_positive(shape, "weibull_pdf", "shape");
  detail::require_positive(scale, "weibull_pdf", "scale");
  if (x < 0.0) return 0.0;
  if (x == 0.0 && shape < 1.0) return std::numeric_limits<double>::infinity();
  const double xs = x / scale;
  return (shape / scale) * std::pow(xs, shape - 1.0) * std::exp(-std::pow(xs, shape));
}

inline double weibull_cdf(double x, double shape, double scale = 1.0) {
  detail::require_positive(shape, "weibull_cdf", "shape");
  detail::require_positive(scale, "weibull_cdf", "scale");
  if (x <= 0.0) return 0.0;
  return 1.0 - std::exp(-std::pow(x / scale, shape));
}

inline double weibull_quantile(double p, double shape, double scale = 1.0) {
  detail::require_probability(p, "weibull_quantile");
  detail::require_positive(shape, "weibull_quantile", "shape");
  detail::require_positive(scale, "weibull_quantile", "scale");
  if (p == 1.0) return std::numeric_limits<double>::infinity();
  return scale * std::pow(-std::log1p(-p), 1.0 / shape);
}

inline double rayleigh_pdf(double x, double scale = 1.0) {
  detail::require_positive(scale, "rayleigh_pdf", "scale");
  if (x < 0.0) return 0.0;
  return x / (scale * scale) * std::exp(-(x * x) / (2.0 * scale * scale));
}

inline double rayleigh_cdf(double x, double scale = 1.0) {
  detail::require_positive(scale, "rayleigh_cdf", "scale");
  if (x <= 0.0) return 0.0;
  return 1.0 - std::exp(-(x * x) / (2.0 * scale * scale));
}

inline double rayleigh_quantile(double p, double scale = 1.0) {
  detail::require_probability(p, "rayleigh_quantile");
  detail::require_positive(scale, "rayleigh_quantile", "scale");
  if (p == 1.0) return std::numeric_limits<double>::infinity();
  return scale * std::sqrt(-2.0 * std::log1p(-p));
}

inline double pareto_pdf(double x, double shape, double scale = 1.0) {
  detail::require_positive(shape, "pareto_pdf", "shape");
  detail::require_positive(scale, "pareto_pdf", "scale");
  if (x < scale) return 0.0;
  return shape * std::pow(scale, shape) / std::pow(x, shape + 1.0);
}

inline double pareto_cdf(double x, double shape, double scale = 1.0) {
  detail::require_positive(shape, "pareto_cdf", "shape");
  detail::require_positive(scale, "pareto_cdf", "scale");
  if (x < scale) return 0.0;
  return 1.0 - std::pow(scale / x, shape);
}

inline double pareto_quantile(double p, double shape, double scale = 1.0) {
  detail::require_probability(p, "pareto_quantile");
  detail::require_positive(shape, "pareto_quantile", "shape");
  detail::require_positive(scale, "pareto_quantile", "scale");
  if (p == 1.0) return std::numeric_limits<double>::infinity();
  return scale / std::pow(1.0 - p, 1.0 / shape);
}

inline double gumbel_pdf(double x, double location = 0.0, double scale = 1.0) {
  detail::require_positive(scale, "gumbel_pdf", "scale");
  const double z = (x - location) / scale;
  return (1.0 / scale) * std::exp(-(z + std::exp(-z)));
}

inline double gumbel_cdf(double x, double location = 0.0, double scale = 1.0) {
  detail::require_positive(scale, "gumbel_cdf", "scale");
  const double z = (x - location) / scale;
  return std::exp(-std::exp(-z));
}

inline double gumbel_quantile(double p, double location = 0.0, double scale = 1.0) {
  detail::require_probability(p, "gumbel_quantile");
  detail::require_positive(scale, "gumbel_quantile", "scale");
  if (p == 0.0) return -std::numeric_limits<double>::infinity();
  if (p == 1.0) return std::numeric_limits<double>::infinity();
  return location - scale * std::log(-std::log(p));
}

inline double bernoulli_pdf(int k, double p) {
  detail::require_probability(p, "bernoulli_pdf");
  if (k == 0) return 1.0 - p;
  if (k == 1) return p;
  return 0.0;
}

inline double bernoulli_cdf(int k, double p) {
  detail::require_probability(p, "bernoulli_cdf");
  if (k < 0) return 0.0;
  if (k == 0) return 1.0 - p;
  return 1.0;
}

inline int bernoulli_quantile(double q, double p) {
  detail::require_probability(q, "bernoulli_quantile");
  detail::require_probability(p, "bernoulli_quantile");
  return (q <= 1.0 - p) ? 0 : 1;
}

inline double geometric_pdf(std::size_t k, double p) {
  detail::require_probability(p, "geometric_pdf");
  if (p == 0.0) return 0.0;
  if (p == 1.0) return k == 0 ? 1.0 : 0.0;
  return p * std::pow(1.0 - p, static_cast<double>(k));
}

inline double geometric_cdf(std::size_t k, double p) {
  detail::require_probability(p, "geometric_cdf");
  if (p == 0.0) return 0.0;
  return 1.0 - std::pow(1.0 - p, static_cast<double>(k) + 1.0);
}

inline std::size_t geometric_quantile(double q, double p) {
  detail::require_probability(q, "geometric_quantile");
  detail::require_probability(p, "geometric_quantile");
  if (q == 0.0) return 0;
  if (q == 1.0) return std::numeric_limits<std::size_t>::max();
  if (p == 0.0) return std::numeric_limits<std::size_t>::max();
  if (p == 1.0) return 0;
  return static_cast<std::size_t>(
      std::ceil(std::log1p(-q) / std::log(1.0 - p) - 1.0));
}

inline double poisson_pdf(std::size_t k, double lambda) {
  detail::require_positive(lambda, "poisson_pdf", "lambda");
  return detail::poisson_pdf_nonneg(k, lambda);
}

inline double poisson_cdf(std::size_t k, double lambda) {
  detail::require_positive(lambda, "poisson_cdf", "lambda");
  double sum = 0.0;
  for (std::size_t i = 0; i <= k; ++i) {
    sum += detail::poisson_pdf_nonneg(i, lambda);
  }
  return std::min(1.0, sum);
}

inline std::size_t poisson_quantile(double q, double lambda) {
  detail::require_probability(q, "poisson_quantile");
  detail::require_positive(lambda, "poisson_quantile", "lambda");
  if (q == 0.0) return 0;
  if (q == 1.0) return std::numeric_limits<std::size_t>::max();
  double cumulative = 0.0;
  std::size_t k = 0;
  for (;; ++k) {
    cumulative += detail::poisson_pdf_nonneg(k, lambda);
    if (cumulative >= q || cumulative >= 1.0 - 1e-15) return k;
  }
}

inline double student_t_pdf(double x, double df) {
  detail::require_positive(df, "student_t_pdf", "df");
  const double half_df = 0.5 * df;
  const double log_norm = std::lgamma(half_df + 0.5) - std::lgamma(half_df)
                         - 0.5 * std::log(df * detail::pi);
  return std::exp(log_norm - (half_df + 0.5) * std::log1p(x * x / df));
}

inline double student_t_cdf(double x, double df) {
  detail::require_positive(df, "student_t_cdf", "df");
  const double xt = df / (df + x * x);
  const double tail = 0.5 * detail::regularized_incomplete_beta(0.5 * df, 0.5, xt);
  return (x >= 0.0) ? (1.0 - tail) : tail;
}

inline double student_t_quantile(double p, double df) {
  detail::require_probability(p, "student_t_quantile");
  detail::require_positive(df, "student_t_quantile", "df");
  if (p == 0.0) return -std::numeric_limits<double>::infinity();
  if (p == 1.0) return std::numeric_limits<double>::infinity();
  if (p == 0.5) return 0.0;
  return detail::invert_monotonic_cdf(
      p, [df](double x) { return student_t_cdf(x, df); }, -1.0, 1.0);
}

inline double f_pdf(double x, double df1, double df2) {
  detail::require_positive(df1, "f_pdf", "df1");
  detail::require_positive(df2, "f_pdf", "df2");
  if (x < 0.0) return 0.0;
  if (x == 0.0) return (df1 < 2.0) ? std::numeric_limits<double>::infinity() : 0.0;
  const double half1 = 0.5 * df1;
  const double half2 = 0.5 * df2;
  const double log_pdf = half1 * std::log(df1 / df2) + (half1 - 1.0) * std::log(x)
                        - (half1 + half2) * std::log1p(df1 * x / df2)
                        - detail::log_beta(half1, half2);
  return std::exp(log_pdf);
}

inline double f_cdf(double x, double df1, double df2) {
  detail::require_positive(df1, "f_cdf", "df1");
  detail::require_positive(df2, "f_cdf", "df2");
  if (x <= 0.0) return 0.0;
  const double xt = df1 * x / (df1 * x + df2);
  return detail::regularized_incomplete_beta(0.5 * df1, 0.5 * df2, xt);
}

inline double f_quantile(double p, double df1, double df2) {
  detail::require_probability(p, "f_quantile");
  detail::require_positive(df1, "f_quantile", "df1");
  detail::require_positive(df2, "f_quantile", "df2");
  if (p == 0.0) return 0.0;
  if (p == 1.0) return std::numeric_limits<double>::infinity();
  return detail::invert_monotonic_cdf(
      p, [df1, df2](double x) { return f_cdf(x, df1, df2); }, 1e-6, 1.0);
}

/// @brief Density of the Beta(\f$a,b\f$) distribution on \f$[0,1]\f$.
/// @param x  Point.
/// @param a  First shape (\f$>0\f$).
/// @param b  Second shape (\f$>0\f$).
/// @return \f$x^{a-1}(1-x)^{b-1}/B(a,b)\f$, 0 outside \f$[0,1]\f$.
/// @throws std::invalid_argument  If a shape is not positive.
///
/// @par Example
/// @code
/// std::print("{:.4f}\n", dm::random::beta_pdf(0.5, 2.0, 2.0));  // 1.5000
/// @endcode
inline double beta_pdf(double x, double a, double b) {
  detail::require_positive(a, "beta_pdf", "a");
  detail::require_positive(b, "beta_pdf", "b");
  if (x < 0.0 || x > 1.0) return 0.0;
  if (x == 0.0) return a < 1.0 ? std::numeric_limits<double>::infinity() : (a == 1.0 ? b : 0.0);
  if (x == 1.0) return b < 1.0 ? std::numeric_limits<double>::infinity() : (b == 1.0 ? a : 0.0);
  return std::exp((a - 1.0) * std::log(x) + (b - 1.0) * std::log1p(-x) - detail::log_beta(a, b));
}

/// @brief Distribution function of the Beta(\f$a,b\f$) distribution.
/// @param x  Point.
/// @param a  First shape (\f$>0\f$).
/// @param b  Second shape (\f$>0\f$).
/// @return The regularized incomplete beta function \f$I_x(a,b)\f$ (0 below 0, 1 above 1).
/// @throws std::invalid_argument  If a shape is not positive.
///
/// @par Example
/// @code
/// std::print("{:.4f}\n", dm::random::beta_cdf(0.9, 18.0, 2.0));  // 0.4203
/// @endcode
inline double beta_cdf(double x, double a, double b) {
  detail::require_positive(a, "beta_cdf", "a");
  detail::require_positive(b, "beta_cdf", "b");
  if (x <= 0.0) return 0.0;
  if (x >= 1.0) return 1.0;
  return detail::regularized_incomplete_beta(a, b, x);
}

/// @brief Quantile function of the Beta(\f$a,b\f$) distribution, by bisection.
/// @param p  Probability in \f$[0,1]\f$.
/// @param a  First shape (\f$>0\f$).
/// @param b  Second shape (\f$>0\f$).
/// @return The \f$x\f$ with \f$I_x(a,b)=p\f$.
/// @throws std::invalid_argument  If @p p is not a probability or a shape is not positive.
///
/// @par Example
/// @code
/// std::print("{:.4f}\n", dm::random::beta_quantile(0.5, 1.0, 1.0));  // 0.5000
/// @endcode
inline double beta_quantile(double p, double a, double b) {
  detail::require_probability(p, "beta_quantile");
  detail::require_positive(a, "beta_quantile", "a");
  detail::require_positive(b, "beta_quantile", "b");
  if (p == 0.0) return 0.0;
  if (p == 1.0) return 1.0;
  double lo = 0.0, hi = 1.0;
  for (int it = 0; it < 200 && hi - lo > 1e-15; ++it) {
    const double mid = 0.5 * (lo + hi);
    (detail::regularized_incomplete_beta(a, b, mid) < p ? lo : hi) = mid;
  }
  return 0.5 * (lo + hi);
}

inline double chi_squared_pdf(double x, double df) {
  detail::require_positive(df, "chi_squared_pdf", "df");
  if (x < 0.0) return 0.0;
  if (x == 0.0) return (df < 2.0) ? std::numeric_limits<double>::infinity() : (df == 2.0 ? 0.5 : 0.0);
  const double half_df = 0.5 * df;
  const double log_pdf = (half_df - 1.0) * std::log(x) - 0.5 * x - std::lgamma(half_df) - half_df * std::log(2.0);
  return std::exp(log_pdf);
}

inline double chi_squared_cdf(double x, double df) {
  detail::require_positive(df, "chi_squared_cdf", "df");
  if (x <= 0.0) return 0.0;
  return detail::regularized_lower_incomplete_gamma(0.5 * df, 0.5 * x);
}

inline double chi_squared_quantile(double p, double df) {
  detail::require_probability(p, "chi_squared_quantile");
  detail::require_positive(df, "chi_squared_quantile", "df");
  if (p == 0.0) return 0.0;
  if (p == 1.0) return std::numeric_limits<double>::infinity();
  return detail::invert_monotonic_cdf(
      p, [df](double x) { return chi_squared_cdf(x, df); }, 1e-6, std::max(df, 1.0));
}

namespace detail {

/// @brief Validates the positive finite parameters of a gamma distribution.
/// @param shape Gamma shape.
/// @param scale Gamma scale.
/// @throws std::invalid_argument Nonpositive or nonfinite shape or scale.
inline void check_gamma_parameters(double shape, double scale) {
  if (!(shape > 0 && scale > 0) || !std::isfinite(shape) || !std::isfinite(scale))
    throw std::invalid_argument("gamma distribution: shape and scale must be positive and finite");
}

/// @brief Evaluates the gamma saddle-point divergence d - log(1 + d).
/// @param d Scaled displacement from the gamma mean.
/// @return Nonnegative divergence, or infinity at the lower boundary.
inline long double gamma_divergence(long double d) {
  if (std::abs(d) < 1e-3L)
    return d * d * (.5L + d * (-1.0L / 3 + d * (.25L + d * (-.2L + d * (1.0L / 6 - d / 7)))));
  return d - std::log1p(d);
}

/// @brief Evaluates both gamma tails using series, continued fractions, or the leading uniform expansion.
/// @param scaled Positive scale-one variate, possibly infinite.
/// @param shape Positive finite gamma shape.
/// @param lower_tail Selects lower (true) or upper (false) probability.
/// @return Requested probability.
inline double gamma_probability(long double scaled, double shape, bool lower_tail) {
  if (scaled == 0 || std::isinf(scaled))
    return (scaled > 0) == lower_tail ? 1 : 0;
  if (shape >= 1e6) {
    const auto d = (scaled - shape) / shape;
    const auto deviance = gamma_divergence(d) * shape;
    const auto z = std::copysign(std::sqrt(deviance), d);
    const auto c0 = std::abs(d) < 1e-3L
        ? -1.0L / 3 + d * (1.0L / 12 - d * 23.0L / 540)
        : 1 / d - std::copysign(std::sqrt(shape / (2 * deviance)), d);
    const auto correction = std::exp(-deviance) * c0 / std::sqrt(2 * pi * static_cast<long double>(shape));
    const auto tail = lower_tail ? .5L * std::erfc(-z) - correction : .5L * std::erfc(z) + correction;
    return static_cast<double>(std::clamp(tail, 0.0L, 1.0L));
  }
  if (shape < 1 && scaled <= 1) {
    // DLMF 8.7.3. Form the upper tail without subtracting a lower probability
    // that can round to one when the shape is small.
    const long double a = shape;
    const auto log_gamma = a < 1e-5L
        ? a * (-0.5772156649015328606L + a * (.8224670334241132182L - a * .4006856343865314285L))
        : std::lgamma(1 + a);
    const auto exponent = a * std::log(scaled) - log_gamma;
    long double term = 1, sum = 0;
    for (int k = 1; k <= 32; ++k) {
      term *= -scaled / k;
      sum += term / (a + k);
    }
    const auto factor = a * sum;
    const auto tail = lower_tail ? std::exp(exponent) * (1 + factor)
                                 : -std::expm1(exponent) - std::exp(exponent) * factor;
    return static_cast<double>(std::clamp(tail, 0.0L, 1.0L));
  }
  if (scaled > std::numeric_limits<double>::max())
    return lower_tail ? 1 : 0;
  const auto x = static_cast<double>(scaled);
  if (x >= shape + 1 || shape < 1) {
    const auto upper = incomplete_gamma_cf(shape, x);
    return std::clamp(lower_tail ? 1 - upper : upper, 0.0, 1.0);
  }
  const auto lower = incomplete_gamma_series(shape, x);
  return std::clamp(lower_tail ? lower : 1 - lower, 0.0, 1.0);
}

} // namespace detail

/// @brief Evaluates the shape/scale gamma density, like R's dgamma().
/// @param x Point at which to evaluate the density.
/// @param shape Positive finite shape.
/// @param scale Positive finite scale, default 1.
/// @return Density, including the limiting value at zero.
/// @throws std::invalid_argument Invalid parameters or NaN observation.
/// @par Example
/// @code
/// std::print("{:.4f}\n", dm::random::gamma_pdf(2, 1, 2)); // 0.1839
/// @endcode
/// @par References
/// - https://stat.ethz.ch/R-manual/R-devel/library/stats/html/GammaDist.html
/// @see gamma_cdf()
inline double gamma_pdf(double x, double shape, double scale = 1.0) {
  detail::check_gamma_parameters(shape, scale);
  if (std::isnan(x))
    throw std::invalid_argument("gamma_pdf: x cannot be NaN");
  if (x < 0 || std::isinf(x))
    return 0;
  if (x == 0)
    return shape < 1 ? std::numeric_limits<double>::infinity() : (shape == 1 ? 1 / scale : 0);
  const long double a = shape;
  const auto scaled = static_cast<long double>(x) / scale;
  if (std::isinf(scaled))
    return 0;
  const auto logarithm = a >= 1e6
      ? .5L * std::log(a / (2 * detail::pi)) - std::log(static_cast<long double>(x)) -
            a * detail::gamma_divergence((scaled - a) / a) - 1 / (12 * a)
      : (a - 1) * (std::log(static_cast<long double>(x)) - std::log(static_cast<long double>(scale))) -
            scaled - std::lgamma(a) - std::log(static_cast<long double>(scale));
  return static_cast<double>(std::exp(logarithm));
}

/// @brief Evaluates the lower or upper gamma tail, like R's pgamma().
///
/// Uses the incomplete gamma series and continued fraction, with a uniform
/// asymptotic expansion for large shape. Computes a small upper tail directly.
/// @param x Query value, including signed infinities.
/// @param shape Positive finite shape.
/// @param scale Positive finite scale, default 1.
/// @param lower_tail True for P(X <= x), false for P(X > x), default true.
/// @return Probability in [0, 1].
/// @throws std::invalid_argument Invalid parameters or NaN query.
/// @par Example
/// @code
/// std::print("{:.4f}\n", dm::random::gamma_cdf(2, 1, 2, false)); // 0.3679
/// @endcode
/// @par References
/// - https://dlmf.nist.gov/8.7
/// - https://dlmf.nist.gov/8.12
/// @see gamma_quantile()
inline double gamma_cdf(double x, double shape, double scale = 1.0, bool lower_tail = true) {
  detail::check_gamma_parameters(shape, scale);
  if (std::isnan(x))
    throw std::invalid_argument("gamma_cdf: x cannot be NaN");
  if (x <= 0)
    return lower_tail ? 0 : 1;
  return detail::gamma_probability(static_cast<long double>(x) / scale, shape, lower_tail);
}

/// @brief Inverts the shape/scale gamma CDF, like R's qgamma().
///
/// Bisects on the log scale using the supplied tail, so upper-tail probabilities
/// need not be subtracted from one. Returns infinity above double range and zero
/// for a quantile strictly below the smallest positive double.
/// @param probability Tail probability in [0, 1].
/// @param shape Positive finite shape.
/// @param scale Positive finite scale, default 1.
/// @param lower_tail True for lower-tail probability, false for upper, default true.
/// @return Nonnegative quantile, with the natural zero and infinity endpoints.
/// @throws std::invalid_argument Invalid probability or parameters.
/// @par Example
/// @code
/// std::print("{:.4f}\n", dm::random::gamma_quantile(.5, 1, 2)); // 1.3863
/// @endcode
/// @par References
/// - https://stat.ethz.ch/R-manual/R-devel/library/stats/html/GammaDist.html
/// @see gamma_cdf()
inline double gamma_quantile(double probability, double shape, double scale = 1.0,
                             bool lower_tail = true) {
  detail::check_gamma_parameters(shape, scale);
  detail::require_probability(probability, "gamma_quantile");
  if (probability == 0 || probability == 1)
    return (probability > 0) == lower_tail ? std::numeric_limits<double>::infinity() : 0;
  if (probability > .5) {
    probability = 1 - probability;
    lower_tail = !lower_tail;
  }
  const long double scale_log = std::log(static_cast<long double>(scale));
  const auto minimum = std::numeric_limits<double>::denorm_min();
  const auto maximum = std::numeric_limits<double>::max();
  auto lo = std::log(static_cast<long double>(minimum));
  auto hi = std::log(static_cast<long double>(maximum));
  /// @brief Checks whether the CDF at a log-scale variate has reached the target.
  const auto reaches = [=](long double log_x) {
    const auto tail = detail::gamma_probability(std::exp(log_x - scale_log), shape, lower_tail);
    return lower_tail ? tail >= probability : tail <= probability;
  };
  if (!reaches(hi))
    return std::numeric_limits<double>::infinity();
  if (reaches(lo))
    return gamma_cdf(minimum, shape, scale, lower_tail) == probability ? minimum : 0;
  for (int i = 0; i < 80; ++i) {
    const auto mid = std::midpoint(lo, hi);
    if (reaches(mid))
      hi = mid;
    else
      lo = mid;
  }
  return static_cast<double>(std::min(std::exp(hi), static_cast<long double>(maximum)));
}

inline double binomial_pdf(std::size_t k, std::size_t n, double p) {
  detail::require_probability(p, "binomial_pdf");
  if (k > n) return 0.0;
  if (p == 0.0) return (k == 0) ? 1.0 : 0.0;
  if (p == 1.0) return (k == n) ? 1.0 : 0.0;
  const double log_pmf = detail::log_choose(static_cast<double>(n), static_cast<double>(k))
                        + static_cast<double>(k) * std::log(p)
                        + static_cast<double>(n - k) * std::log1p(-p);
  return std::exp(log_pmf);
}

inline double binomial_cdf(std::size_t k, std::size_t n, double p) {
  detail::require_probability(p, "binomial_cdf");
  if (k >= n) return 1.0;
  double sum = 0.0;
  for (std::size_t i = 0; i <= k; ++i) sum += binomial_pdf(i, n, p);
  return std::min(1.0, sum);
}

/// @brief Inverts the binomial CDF, like R's qbinom().
///
/// Returns the smallest count whose CDF reaches the requested probability. Uses binary
/// search with a beta tail, or a probability recurrence when the success probability
/// is too small for its complement to be represented accurately.
/// @param probability Cumulative probability in [0, 1].
/// @param n Number of Bernoulli trials, including zero.
/// @param p Probability of success per trial in [0, 1].
/// @return Count in the support. At probabilities zero and one, returns the support extremes.
/// @throws std::invalid_argument Invalid probability or success probability.
/// @par Example
/// @code
/// std::print("{}\n", dm::random::binomial_quantile(.5, 10, .3)); // 3
/// @endcode
/// @par References
/// - https://stat.ethz.ch/R-manual/R-devel/library/stats/html/Binomial.html
/// @see binomial_cdf()
inline std::size_t binomial_quantile(double probability, std::size_t n, double p) {
  detail::require_probability(probability, "binomial_quantile");
  detail::require_probability(p, "binomial_quantile");
  if (p == 0 || p == 1)
    return p == 0 ? 0 : n;
  if (probability == 0 || probability == 1)
    return probability == 0 ? 0 : n;
  if (1 - p == 1) {
    // In this regime the mean is at most about 1024 for 64-bit counts.
    // Accumulate the lower tail directly when 1 - p cannot encode the failure probability.
    const auto log_odds = std::log(static_cast<long double>(p)) - std::log1p(-static_cast<long double>(p));
    auto log_mass = static_cast<long double>(n) * std::log1p(-static_cast<long double>(p));
    long double cumulative = std::exp(log_mass);
    const auto mean = static_cast<long double>(n) * p;
    const auto upper_target = 1 - static_cast<long double>(probability);
    for (std::size_t k = 0; k < n; ++k) {
      if (probability <= .5 && static_cast<double>(cumulative) >= probability)
        return k;
      log_mass += std::log(static_cast<long double>(n - k)) -
                  std::log(static_cast<long double>(k) + 1) + log_odds;
      const auto next_mass = std::exp(log_mass);
      cumulative += next_mass;
      if (probability > .5 && k >= std::floor(mean)) {
        long double tail = next_mass, mass = next_mass;
        for (auto j = k + 1; j < n && tail <= upper_target; ++j) {
          mass *= ((n - j) * static_cast<long double>(p)) /
                  ((static_cast<long double>(j) + 1) * (1 - static_cast<long double>(p)));
          const auto increased_tail = tail + mass;
          if (increased_tail == tail)
            break;
          tail = increased_tail;
        }
        if (tail <= upper_target)
          return k;
      }
    }
    return n;
  }
  std::size_t lo = 0, hi = n;
  while (lo < hi) {
    const auto k = std::midpoint(lo, hi);
    const auto a = static_cast<double>(n - k);
    const auto b = static_cast<double>(k) + 1;
    // Evaluate the target tail directly, especially for quantiles near zero or one.
    const bool reaches = probability > .5
        ? detail::regularized_incomplete_beta(b, a, p) <= 1 - probability
        : detail::regularized_incomplete_beta(a, b, 1 - p) >= probability;
    if (reaches)
      hi = k;
    else
      lo = k + 1;
  }
  return lo;
}

// Hypergeometric distribution: drawing n items without replacement from a population of
// size population containing successes_in_population successes; pmf/cdf of the number of
// successes k in the sample. Used by Fisher's exact test.
inline double hypergeometric_pdf(std::size_t k, std::size_t population, std::size_t successes_in_population,
                                  std::size_t n) {
  if (successes_in_population > population || n > population) return 0.0;
  const std::size_t failures_in_population = population - successes_in_population;
  if (k > successes_in_population || (n - k) > failures_in_population) return 0.0;
  const double log_pmf = detail::log_choose(static_cast<double>(successes_in_population), static_cast<double>(k))
                        + detail::log_choose(static_cast<double>(failures_in_population), static_cast<double>(n - k))
                        - detail::log_choose(static_cast<double>(population), static_cast<double>(n));
  return std::exp(log_pmf);
}

inline double hypergeometric_cdf(std::size_t k, std::size_t population, std::size_t successes_in_population,
                                  std::size_t n) {
  const std::size_t lo = (n > population - successes_in_population) ? n - (population - successes_in_population) : 0;
  double sum = 0.0;
  for (std::size_t i = lo; i <= k; ++i) sum += hypergeometric_pdf(i, population, successes_in_population, n);
  return std::min(1.0, sum);
}

/// @brief Inverts the hypergeometric CDF, like R's qhyper().
///
/// Enumerates representable relative masses outward from the mode before accumulating
/// the selected tail. This avoids seeding the recurrence with an underflowed tail mass.
/// @param probability Cumulative probability in [0, 1].
/// @param population Population size.
/// @param successes_in_population Number of successes, at most population.
/// @param n Number drawn without replacement, at most population.
/// @return The smallest count reaching the requested CDF, with endpoints on the finite support.
/// @throws std::invalid_argument Invalid probability or population counts.
/// @par Example
/// @code
/// std::print("{}\n", dm::random::hypergeometric_quantile(.5, 20, 7, 5)); // 2
/// @endcode
/// @par References
/// - https://stat.ethz.ch/R-manual/R-devel/library/stats/html/Hypergeometric.html
/// @see hypergeometric_cdf()
inline std::size_t hypergeometric_quantile(double probability, std::size_t population,
                                           std::size_t successes_in_population, std::size_t n) {
  detail::require_probability(probability, "hypergeometric_quantile");
  if (successes_in_population > population || n > population)
    throw std::invalid_argument("hypergeometric_quantile: invalid population counts");
  const auto successes = successes_in_population;
  const auto failures = population - successes;
  const auto lower = n > failures ? n - failures : 0;
  const auto upper = std::min(n, successes);
  if (probability == 0 || lower == upper)
    return lower;
  if (probability == 1)
    return upper;

  const auto mode_rank = ((static_cast<long double>(n) + 1) *
      (static_cast<long double>(successes) + 1)) / (static_cast<long double>(population) + 2);
  const auto mode = mode_rank >= static_cast<long double>(upper) ? upper :
      std::max(lower, static_cast<std::size_t>(mode_rank));
  std::vector<long double> below, above;
  long double total = 1;
  long double mass = 1;
  for (auto k = mode; k > lower; --k) {
    mass *= (static_cast<long double>(k) * (failures - (n - k))) /
            ((static_cast<long double>(successes - k) + 1) * (static_cast<long double>(n - k) + 1));
    if (mass == 0)
      break;
    total += mass;
    below.push_back(mass);
  }
  mass = 1;
  for (auto k = mode; k < upper; ++k) {
    mass *= (static_cast<long double>(successes - k) * (n - k)) /
            ((static_cast<long double>(k) + 1) * (static_cast<long double>(failures - (n - k)) + 1));
    if (mass == 0)
      break;
    total += mass;
    above.push_back(mass);
  }

  // Select from the requested end of the support to avoid canceling a tiny tail.
  long double cumulative = 0;
  if (probability <= .5) {
    for (std::size_t i = below.size(); i > 0; --i) {
      cumulative += below[i - 1];
      if (static_cast<double>(cumulative / total) >= probability)
        return mode - i;
    }
    cumulative += 1;
    if (static_cast<double>(cumulative / total) >= probability)
      return mode;
    for (std::size_t i = 0; i < above.size(); ++i) {
      cumulative += above[i];
      if (static_cast<double>(cumulative / total) >= probability)
        return mode + i + 1;
    }
    return upper;
  }

  for (std::size_t i = above.size(); i > 0; --i) {
    cumulative += above[i - 1];
    if (static_cast<double>(1 - cumulative / total) < probability)
      return mode + i;
  }
  cumulative += 1;
  if (static_cast<double>(1 - cumulative / total) < probability)
    return mode;
  for (std::size_t i = 0; i < below.size(); ++i) {
    cumulative += below[i];
    if (static_cast<double>(1 - cumulative / total) < probability)
      return mode - i - 1;
  }
  return lower;
}

} // namespace datamunge::random
