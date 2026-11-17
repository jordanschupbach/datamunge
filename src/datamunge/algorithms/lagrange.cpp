/// @file lagrange.cpp
/// @brief Checked direct original-order Lagrange products with optional actual factor/term records.
#include <cmath>
#include <datamunge/algorithms/lagrange.hpp>
#include <limits>
#include <stdexcept>
namespace datamunge::algorithms {
/// @brief Complete input, count and finite-intermediate helpers for the shared producer.
namespace lagrange_detail {
/// @brief Admit matched finite arrays, finite query and distinct abscissae before evaluation.
/// @param xs Borrowed complete nodes.
/// @param ys Borrowed complete matched values.
/// @param x Actual query.
/// @throws std::invalid_argument On shape/finite/duplicate violations.
void validate(const std::vector<double>& xs, const std::vector<double>& ys, double x) {
  if (xs.size() != ys.size() || !std::isfinite(x))
    throw std::invalid_argument("Lagrange shape/query");
  for (std::size_t j = 0; j < xs.size(); ++j) {
    if (!std::isfinite(xs[j]) || !std::isfinite(ys[j]))
      throw std::invalid_argument("Lagrange finite data");
    for (std::size_t k = 0; k < j; ++k)
      if (xs[j] == xs[k])
        throw std::invalid_argument("Lagrange duplicate node");
  }
}
/// @brief Increment actual selected operations without wrap.
/// @param count Actual mutable counter.
/// @throws std::length_error On size_t overflow.
void increment(std::size_t& count) {
  if (count == std::numeric_limits<std::size_t>::max())
    throw std::length_error("Lagrange work overflow");
  ++count;
}
/// @brief Require finite produced double arithmetic, preserving its value otherwise.
/// @param x Actual numeric intermediate/output.
/// @return The unchanged finite value.
/// @throws std::overflow_error On nonfinite numeric value.
double finite(double x) {
  if (!std::isfinite(x))
    throw std::overflow_error("Lagrange nonfinite intermediate");
  return x;
}
/// @brief Produce original-order ratio/products and sequential sums once.
/// @param xs Validated original finite distinct nodes.
/// @param ys Validated matched finite values.
/// @param x Validated finite query.
/// @param trace Optional preadmitted/preallocated actual capture.
/// @return Owned actual value and producing counters.
/// @throws std::length_error On actual count overflow.
/// @throws std::overflow_error On nonfinite ratio/product/sum.
LagrangeAnalysis
run(const std::vector<double>& xs, const std::vector<double>& ys, double x, LagrangeTrace* trace) {
  LagrangeAnalysis out;
  for (std::size_t j = 0; j < xs.size(); ++j) {
    double term = ys[j];
    for (std::size_t k = 0; k < xs.size(); ++k) {
      if (k == j)
        continue;
      const double numerator = x - xs[k], denominator = xs[j] - xs[k];
      const bool   wide  = !std::isfinite(numerator) || !std::isfinite(denominator);
      const double ratio = finite(
          wide ? static_cast<double>(
                     (static_cast<long double>(x) - xs[k]) /
                     (static_cast<long double>(xs[j]) - xs[k]))
               : numerator / denominator);
      increment(out.work.ratio_evaluations);
      const double before = term;
      term                = finite(term * ratio);
      increment(out.work.term_multiplications);
      if (trace)
        trace->factors.push_back({j, k, ratio, before, term, wide});
    }
    const auto before = out.value;
    out.value         = finite(out.value + term);
    increment(out.work.terms);
    increment(out.work.sum_additions);
    if (trace)
      trace->terms.push_back({j, term, before, out.value});
  }
  return out;
}
} // namespace lagrange_detail
double
lagrange_interpolate(const std::vector<double>& xs, const std::vector<double>& ys, double x) {
  lagrange_detail::validate(xs, ys, x);
  return lagrange_detail::run(xs, ys, x, nullptr).value;
}
LagrangeAnalysis
lagrange_analyze(const std::vector<double>& xs, const std::vector<double>& ys, double x) {
  lagrange_detail::validate(xs, ys, x);
  return lagrange_detail::run(xs, ys, x, nullptr);
}
LagrangeTrace lagrange_trace(
    const std::vector<double>& xs,
    const std::vector<double>& ys,
    double                     x,
    std::size_t                max_entries) {
  lagrange_detail::validate(xs, ys, x);
  const auto n = xs.size(), maximum = std::numeric_limits<std::size_t>::max();
  if (n > maximum - 2 || (n && n > maximum / (n + 2)))
    throw std::length_error("Lagrange capture count overflow");
  const auto count = n * (n + 2);
  if (max_entries && count > max_entries)
    throw std::length_error("Lagrange trace quota");
  LagrangeTrace out;
  out.xs               = xs;
  out.ys               = ys;
  out.query            = x;
  out.captured_entries = count;
  out.factors.reserve(n ? n * (n - 1) : 0);
  out.terms.reserve(n);
  const auto a = lagrange_detail::run(xs, ys, x, &out);
  out.value    = a.value;
  out.work     = a.work;
  return out;
}
} // namespace datamunge::algorithms
