/// @file neville.cpp
/// @brief Checked original-order Neville producer with actual optional update capture.
#include <cmath>
#include <datamunge/algorithms/neville.hpp>
#include <limits>
#include <stdexcept>
namespace datamunge::algorithms {
/// @brief Private complete admission, finite arithmetic and prefix-study helpers.
namespace {
/// @brief Add count geometry without wrapping.
/// @param a First count.
/// @param b Second count.
/// @return Checked sum.
/// @throws std::length_error If sum overflows.
std::size_t add(std::size_t a, std::size_t b) {
  if (b > std::numeric_limits<std::size_t>::max() - a)
    throw std::length_error("Neville: count overflow");
  return a + b;
}
/// @brief Multiply count geometry without wrapping.
/// @param a First count.
/// @param b Second count.
/// @return Checked product.
/// @throws std::length_error If product overflows.
std::size_t mul(std::size_t a, std::size_t b) {
  if (b && a > std::numeric_limits<std::size_t>::max() / b)
    throw std::length_error("Neville: count overflow");
  return a * b;
}
/// @brief Execute admitted original descending-index recurrence, capturing only actual updates.
/// @param xs Original distinct finite nodes, arbitrary order.
/// @param ys Matching finite values.
/// @param x Finite query.
/// @param options Optional complete node/history policies.
/// @param out Optional actual capture sink.
/// @return Actual value and producing work.
/// @throws std::invalid_argument If complete input domain fails.
/// @throws std::length_error If complete geometry or policy is exceeded.
/// @throws std::overflow_error If an update is nonfinite after fallback.
NevilleAnalysis run(
    const std::vector<double>& xs,
    const std::vector<double>& ys,
    double                     x,
    NevilleOptions             options,
    NevilleTrace*              out) {
  if (xs.size() != ys.size() || !std::isfinite(x))
    throw std::invalid_argument("Neville: matched shape and finite query required");
  const auto n = xs.size();
  for (std::size_t j = 0; j < n; ++j) {
    if (!std::isfinite(xs[j]) || !std::isfinite(ys[j]))
      throw std::invalid_argument("Neville: finite samples required");
    for (std::size_t k = 0; k < j; ++k)
      if (xs[k] == xs[j])
        throw std::invalid_argument("Neville: duplicate nodes");
  }
  const auto updates = n < 2 ? 0 : (n % 2 ? mul(n, (n - 1) / 2) : mul(n / 2, n - 1));
  if (options.max_nodes && n > options.max_nodes)
    throw std::length_error("Neville: complete node policy");
  if (out) {
    out->captured_entries = add(1, add(mul(3, n), updates));
    if (options.max_trace_entries && out->captured_entries > options.max_trace_entries)
      throw std::length_error("Neville: complete capture policy");
    out->xs    = xs;
    out->ys    = ys;
    out->query = x;
    out->steps.reserve(updates);
  }
  NevilleAnalysis result;
  result.nodes          = n;
  std::vector<double> p = ys;
  for (std::size_t degree = 1; degree < n; ++degree)
    for (std::size_t j = n - 1;; --j) {
      const auto   first = j - degree;
      const double right = p[j], left = p[j - 1], a = x - xs[first], b = x - xs[j],
                   denominator = xs[j] - xs[first], r = a * right, l = b * left, numerator = r - l;
      double     value = numerator / denominator;
      const bool wide  = !std::isfinite(a) || !std::isfinite(b) || !std::isfinite(denominator) ||
                        !std::isfinite(r) || !std::isfinite(l) || !std::isfinite(numerator) ||
                        !std::isfinite(value);
      if (wide) {
        const long double ax = static_cast<long double>(x) - xs[first],
                          bx = static_cast<long double>(x) - xs[j],
                          dx = static_cast<long double>(xs[j]) - xs[first];
        value                = double((ax * right - bx * left) / dx);
        ++result.widened_updates;
      }
      if (!std::isfinite(value))
        throw std::overflow_error("Neville: nonfinite update");
      p[j] = value;
      ++result.updates;
      if (out)
        out->steps.push_back({degree, first, j, right, left, value, wide});
      if (j == degree)
        break;
    }
  result.value = n ? p.back() : 0;
  if (out)
    out->final_working = std::move(p);
  return result;
}
/// @brief Evaluate all original-order prefixes and compare with a stated analytic truth.
/// @param xs Complete distinct finite nodes, at least2.
/// @param ys Matching finite values.
/// @param query Common finite query.
/// @param truth Analytic generating-function comparison.
/// @return Owned actual prefix values, errors and successive changes.
NevilleStudy study(std::vector<double> xs, std::vector<double> ys, double query, double truth) {
  NevilleStudy s;
  s.xs    = std::move(xs);
  s.ys    = std::move(ys);
  s.query = query;
  s.truth = truth;
  for (std::size_t n = 2; n <= s.xs.size(); ++n) {
    auto value = neville_interpolate(
        {s.xs.begin(), s.xs.begin() + n}, {s.ys.begin(), s.ys.begin() + n}, query);
    s.degrees.push_back(n - 1);
    s.successive_changes.push_back(s.estimates.empty() ? 0 : std::abs(value - s.estimates.back()));
    s.estimates.push_back(value);
    s.absolute_errors.push_back(std::abs(value - truth));
  }
  return s;
}
} // namespace
double neville_interpolate(const std::vector<double>& x, const std::vector<double>& y, double q) {
  return run(x, y, q, {0, 0}, nullptr).value;
}
NevilleAnalysis neville_analyze(
    const std::vector<double>& x,
    const std::vector<double>& y,
    double                     q,
    NevilleOptions             o) {
  return run(x, y, q, o, nullptr);
}
NevilleTrace neville_trace(
    const std::vector<double>& x,
    const std::vector<double>& y,
    double                     q,
    NevilleOptions             o) {
  NevilleTrace t;
  t.analysis = run(x, y, q, o, &t);
  return t;
}
NevilleStudy neville_cosine_study() {
  std::vector<double> x, y;
  for (int i = 0; i <= 6; ++i) {
    x.push_back(.5 * i);
    y.push_back(std::cos(x.back()));
  }
  return study(std::move(x), std::move(y), 1.25, std::cos(1.25));
}
NevilleStudy neville_zero_sample_study() {
  return study({0, 1, 2}, {0, 0, 0}, .5, .375);
}
} // namespace datamunge::algorithms
