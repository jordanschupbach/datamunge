/// @file nearest_neighbor_interpolation.cpp
/// @brief Shared validated binary nearest-sample producer and native original fixture.
#include <algorithm>
#include <cmath>
#include <datamunge/algorithms/nearest_neighbor_interpolation.hpp>
#include <limits>
#include <stdexcept>
namespace datamunge::algorithms {
NearestNeighborInterpolant::NearestNeighborInterpolant(
    std::vector<double> xs,
    std::vector<double> ys,
    std::size_t         cap)
    : xs_(std::move(xs)), ys_(std::move(ys)) {
  if (xs_.size() != ys_.size())
    throw std::invalid_argument("nearest neighbor: node shapes differ");
  for (std::size_t i = 0; i < xs_.size(); ++i)
    if (!std::isfinite(xs_[i]) || !std::isfinite(ys_[i]) || (i && !(xs_[i] > xs_[i - 1])))
      throw std::invalid_argument("nearest neighbor: finite strictly ordered nodes required");
  if (cap && xs_.size() > cap)
    throw std::length_error("nearest neighbor: node cap exceeded");
}
double NearestNeighborInterpolant::run(double x, NearestNeighborTrace* out, std::size_t cap) const {
  const auto n = xs_.size();
  if (!n)
    return 0;
  if (out) {
    out->endpoint_tests = 1;
    out->route          = 1;
  }
  if (x <= xs_.front())
    return ys_.front();
  if (out) {
    out->endpoint_tests = 2;
    out->route          = 2;
    out->selected       = n - 1;
  }
  if (x >= xs_.back())
    return ys_.back();
  std::size_t lo = 0, hi = n - 1;
  while (lo < hi) {
    const auto mid   = lo + (hi - lo) / 2;
    const bool right = xs_[mid] < x;
    if (out) {
      if (out->captured_entries == std::numeric_limits<std::size_t>::max() ||
          (cap && out->captured_entries >= cap))
        throw std::length_error("nearest neighbor: search capture quota exceeded");
      ++out->captured_entries;
      out->steps.push_back({lo, hi, mid, xs_[mid], right});
    }
    if (right)
      lo = mid + 1;
    else
      hi = mid;
  }
  const long double left  = static_cast<long double>(x) - xs_[lo - 1],
                    right = static_cast<long double>(xs_[lo]) - x;
  const auto selected     = right < left ? lo : lo - 1;
  if (out) {
    out->route          = 3;
    out->left_distance  = left;
    out->right_distance = right;
    out->selected       = selected;
  }
  return ys_[selected];
}
double NearestNeighborInterpolant::evaluate(double x) const {
  if (!std::isfinite(x))
    throw std::invalid_argument("nearest neighbor: finite query required");
  return run(x, nullptr, 0);
}
NearestNeighborTrace NearestNeighborInterpolant::trace(double x, std::size_t cap) const {
  if (!std::isfinite(x))
    throw std::invalid_argument("nearest neighbor: finite query required");
  if (xs_.size() > (std::numeric_limits<std::size_t>::max() - 2) / 2)
    throw std::length_error("nearest neighbor: capture geometry overflow");
  const auto base = 2 * xs_.size() + 2;
  if (cap && base > cap)
    throw std::length_error("nearest neighbor: capture quota exceeded");
  NearestNeighborTrace t;
  t.captured_entries = base;
  t.query            = x;
  t.xs               = xs_;
  t.ys               = ys_;
  t.value            = run(x, &t, cap);
  return t;
}
double
nearest_neighbor_interpolate(const std::vector<double>& x, const std::vector<double>& y, double q) {
  return NearestNeighborInterpolant(x, y, 0).evaluate(q);
}
NearestNeighborDemo nearest_neighbor_demo() {
  NearestNeighborDemo d;
  for (int i = 0; i <= 8; ++i) {
    const double x = i * .8;
    d.xs.push_back(x);
    d.ys.push_back(std::sin(x) + .3 * x);
  }
  return d;
}
NearestNeighborStudy nearest_neighbor_demo_study(std::size_t nodes, std::size_t intervals) {
  if (nodes < 2 || !intervals)
    throw std::invalid_argument(
        "nearest_neighbor_demo_study: at least two nodes and one interval required");
  if (nodes > 1000000 || intervals > 1000000)
    throw std::length_error("nearest_neighbor_demo_study: count cap exceeded");
  NearestNeighborStudy s;
  for (std::size_t i = 0; i < nodes; ++i) {
    auto x = 6.4 * i / (nodes - 1);
    s.samples.xs.push_back(x);
    s.samples.ys.push_back(std::sin(x) + .3 * x);
  }
  NearestNeighborInterpolant model(s.samples.xs, s.samples.ys);
  s.error_reference = 1.3 * (6.4 / (nodes - 1)) / 2;
  for (std::size_t i = 0; i <= intervals; ++i) {
    auto x = 6.4 * i / intervals, truth = std::sin(x) + .3 * x, value = model.evaluate(x),
         error = std::abs(value - truth);
    s.queries.push_back(x);
    s.truth.push_back(truth);
    s.values.push_back(value);
    s.absolute_error.push_back(error);
    s.max_grid_error = std::max(s.max_grid_error, error);
  }
  return s;
}
} // namespace datamunge::algorithms
