#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>

namespace datamunge::random {

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

// Continued-fraction expansion (Numerical Recipes' betacf) used by
// regularized_incomplete_beta for the region x < (a+1)/(a+b+2).
inline double incomplete_beta_cf(double a, double b, double x) {
  constexpr int    max_iterations = 200;
  constexpr double epsilon        = 3.0e-16;
  constexpr double tiny           = 1.0e-300;

  const double qab = a + b;
  const double qap = a + 1.0;
  const double qam = a - 1.0;

  double c = 1.0;
  double d = 1.0 - qab * x / qap;
  if (std::abs(d) < tiny) d = tiny;
  d = 1.0 / d;
  double h = d;

  for (int m = 1; m <= max_iterations; ++m) {
    const double m2 = 2.0 * m;

    double aa = m * (b - m) * x / ((qam + m2) * (a + m2));
    d = 1.0 + aa * d; if (std::abs(d) < tiny) d = tiny;
    c = 1.0 + aa / c; if (std::abs(c) < tiny) c = tiny;
    d = 1.0 / d;
    h *= d * c;

    aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
    d = 1.0 + aa * d; if (std::abs(d) < tiny) d = tiny;
    c = 1.0 + aa / c; if (std::abs(c) < tiny) c = tiny;
    d = 1.0 / d;
    const double delta = d * c;
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

// Series expansion (Numerical Recipes' gser) for the regularized lower incomplete gamma
// function P(a, x), valid/fast-converging for x < a + 1.
inline double incomplete_gamma_series(double a, double x) {
  constexpr int    max_iterations = 200;
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

// Continued-fraction expansion (Numerical Recipes' gcf) for the regularized upper
// incomplete gamma function Q(a, x) = 1 - P(a, x), valid/fast-converging for x >= a + 1.
inline double incomplete_gamma_cf(double a, double x) {
  constexpr int    max_iterations = 200;
  constexpr double epsilon        = 3.0e-16;
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

} // namespace datamunge::random
