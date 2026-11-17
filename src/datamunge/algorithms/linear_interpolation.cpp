/// @file linear_interpolation.cpp
/// @brief Complete checked knots, one shared interval/blend producer and original uniform sine study.
#include <algorithm>
#include <bit>
#include <cmath>
#include <datamunge/algorithms/linear_interpolation.hpp>
#include <limits>
#include <numbers>
#include <stdexcept>
namespace datamunge::algorithms {
/// @brief Private complete validation and actual interval-search/blend implementation.
namespace linear_detail {
/// @brief Validate complete knot/ordinate shape, finite values and strict ordering before any search.
/// @param xs Borrowed original abscissae.
/// @param ys Borrowed original ordinates.
/// @throws std::invalid_argument On complete shape/order/nonfinite violations.
void validate(const std::vector<double>& xs, const std::vector<double>& ys) {
  if (xs.size() < 2 || xs.size() != ys.size())
    throw std::invalid_argument("linear interpolation complete shape");
  for (std::size_t i = 0; i < xs.size(); ++i)
    if (!std::isfinite(xs[i]) || !std::isfinite(ys[i]) || (i && !(xs[i] > xs[i - 1])))
      throw std::invalid_argument("linear interpolation finite strictly ordered knots");
}
/// @brief Narrow actual widened arithmetic to finite double range.
/// @param value Actual widened result.
/// @return Rounded finite double.
/// @throws std::overflow_error On nonfinite/out-of-double-range result.
double narrow(long double value) {
  if (!std::isfinite(value) || std::abs(value) > std::numeric_limits<double>::max())
    throw std::overflow_error("linear interpolation output range");
  return static_cast<double>(value);
}
/// @brief Record one actual interval-selection comparison, excluding validation/blend decisions.
/// @param capture Optional preadmitted owned capture.
/// @param kind Actual first/penultimate/upper-bound role0/1/2.
/// @param index Actual original knot index.
/// @param knot Actual original finite abscissa.
/// @param predicate Actual role-specific comparison result.
void probe(
    LinearInterpolationTrace* capture,
    unsigned                  kind,
    std::size_t               index,
    double                    knot,
    bool                      predicate) {
  if (capture)
    capture->probes.push_back({index, knot, predicate, kind});
}
/// @brief Select an interval and produce one actual checked line value with optional search observation.
/// @param xs Once-validated complete abscissae.
/// @param ys Once-validated complete matched ordinates.
/// @param x Actual finite query checked here.
/// @param capture Optional complete preadmitted capture.
/// @return Actual selected endpoints/fraction/value and search/branch evidence.
/// @throws std::invalid_argument On nonfinite query.
/// @throws std::overflow_error On produced widened fraction/value range.
LinearInterpolationEvaluation evaluate(
    const std::vector<double>& xs,
    const std::vector<double>& ys,
    double                     x,
    LinearInterpolationTrace*  capture) {
  if (!std::isfinite(x))
    throw std::invalid_argument("linear interpolation finite query");
  LinearInterpolationEvaluation out;
  out.query         = x;
  std::size_t k     = 0;
  const bool  first = x <= xs.front();
  ++out.endpoint_tests;
  probe(capture, 0, 0, xs.front(), first);
  if (!first) {
    const auto penultimate = xs.size() - 2;
    const bool last        = x >= xs[penultimate];
    ++out.endpoint_tests;
    probe(capture, 1, penultimate, xs[penultimate], last);
    if (last)
      k = penultimate;
    else {
      std::size_t lo = 0, hi = xs.size();
      while (lo < hi) {
        const auto mid  = lo + (hi - lo) / 2;
        const bool less = x < xs[mid];
        ++out.binary_tests;
        probe(capture, 2, mid, xs[mid], less);
        if (less)
          hi = mid;
        else
          lo = mid + 1;
      }
      k = lo - 1;
    }
  }
  out.interval     = k;
  out.left_x       = xs[k];
  out.right_x      = xs[k + 1];
  out.left_y       = ys[k];
  out.right_y      = ys[k + 1];
  out.extrapolated = x < xs.front() || x > xs.back();
  if (x == xs[k]) {
    out.exact_node = true;
    out.value      = ys[k];
    return out;
  }
  if (x == xs[k + 1]) {
    out.exact_node = true;
    out.fraction   = 1;
    out.value      = ys[k + 1];
    return out;
  }
  const double dx = x - xs[k], width = xs[k + 1] - xs[k], t = dx / width, dy = ys[k + 1] - ys[k],
               scaled = t * dy, value = ys[k] + scaled;
  if (std::isfinite(dx) && std::isfinite(width) && std::isfinite(t) && std::isfinite(dy) &&
      std::isfinite(scaled) && std::isfinite(value)) {
    out.fraction = t;
    out.value    = value;
  } else {
    out.widened          = true;
    const long double lx = xs[k], width = static_cast<long double>(xs[k + 1]) - lx;
    out.fraction = (static_cast<long double>(x) - lx) / width;
    if (!std::isfinite(out.fraction))
      throw std::overflow_error("linear interpolation widened fraction range");
    out.value = narrow(
        static_cast<long double>(ys[k]) +
        out.fraction * (static_cast<long double>(ys[k + 1]) - ys[k]));
  }
  return out;
}
} // namespace linear_detail
LinearInterpolator::LinearInterpolator(
    const std::vector<double>& xs,
    const std::vector<double>& ys) {
  linear_detail::validate(xs, ys);
  xs_ = xs;
  ys_ = ys;
}
double LinearInterpolator::operator()(double x) const {
  return evaluate(x).value;
}
LinearInterpolationEvaluation LinearInterpolator::evaluate(double x) const {
  return linear_detail::evaluate(xs_, ys_, x, nullptr);
}
LinearInterpolationTrace LinearInterpolator::trace(double x, std::size_t max_entries) const {
  if (!std::isfinite(x))
    throw std::invalid_argument("linear interpolation finite query");
  const std::size_t n = xs_.size(), extra = 2 + std::bit_width(n),
                    max = std::numeric_limits<std::size_t>::max();
  if (n > (max - extra) / 2)
    throw std::length_error("linear interpolation capture geometry");
  if (max_entries && 2 * n + extra > max_entries)
    throw std::length_error("linear interpolation conservative capture quota");
  LinearInterpolationTrace out;
  out.xs = xs_;
  out.ys = ys_;
  out.probes.reserve(extra);
  out.evaluation       = linear_detail::evaluate(xs_, ys_, x, &out);
  out.captured_entries = 2 * n + out.probes.size();
  return out;
}
double linear_interpolate(const std::vector<double>& xs, const std::vector<double>& ys, double x) {
  linear_detail::validate(xs, ys);
  return linear_detail::evaluate(xs, ys, x, nullptr).value;
}
LinearInterpolationSineStudy linear_interpolation_sine_study(
    std::size_t nodes,
    std::size_t grid_intervals) {
  if (nodes < 2 || grid_intervals == 0)
    throw std::invalid_argument("linear sine node/grid counts");
  const auto max = std::vector<double>{}.max_size();
  if (nodes > max || grid_intervals >= max)
    throw std::length_error("linear sine complete geometry");
  LinearInterpolationSineStudy out;
  out.xs.reserve(nodes);
  out.ys.reserve(nodes);
  const double end = 2 * std::numbers::pi;
  for (std::size_t i = 0; i < nodes; ++i) {
    const double x = end * i / (nodes - 1);
    out.xs.push_back(x);
    out.ys.push_back(std::sin(x));
  }
  const LinearInterpolator f(out.xs, out.ys);
  out.queries.reserve(grid_intervals + 1);
  out.exact.reserve(grid_intervals + 1);
  out.interpolated.reserve(grid_intervals + 1);
  for (std::size_t i = 0; i <= grid_intervals; ++i) {
    const double x = end * i / grid_intervals, truth = std::sin(x), value = f(x);
    out.queries.push_back(x);
    out.exact.push_back(truth);
    out.interpolated.push_back(value);
    out.max_grid_error = std::max(out.max_grid_error, std::abs(value - truth));
  }
  out.spacing         = end / (nodes - 1);
  out.error_reference = out.spacing * out.spacing / 8;
  return out;
}
} // namespace datamunge::algorithms
