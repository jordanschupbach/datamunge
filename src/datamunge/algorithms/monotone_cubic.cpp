/// @file monotone_cubic.cpp
/// @brief Canonical prepared cubic with sequential, sign-correct slope limiting.
#include <algorithm>
#include <cmath>
#include <datamunge/algorithms/monotone_cubic.hpp>
#include <stdexcept>
namespace datamunge::algorithms {
MonotoneCubic::MonotoneCubic(std::vector<double> xs, std::vector<double> ys, std::size_t cap)
    : xs_(std::move(xs)), ys_(std::move(ys)) {
  const auto n = xs_.size();
  if (n < 2 || ys_.size() != n)
    throw std::invalid_argument("MonotoneCubic: need equal arrays with at least two nodes");
  for (std::size_t i = 0; i < n; ++i)
    if (!std::isfinite(xs_[i]) || !std::isfinite(ys_[i]) || (i && !(xs_[i] > xs_[i - 1])))
      throw std::invalid_argument("MonotoneCubic: finite, strictly ordered nodes required");
  if (cap && n > cap)
    throw std::length_error("MonotoneCubic: node cap exceeded");
  std::vector<double> delta(n - 1);
  for (std::size_t i = 0; i + 1 < n; ++i) {
    auto h   = xs_[i + 1] - xs_[i];
    delta[i] = (ys_[i + 1] - ys_[i]) / h;
    if (!std::isfinite(h) || !std::isfinite(delta[i]))
      throw std::overflow_error("MonotoneCubic: unrepresentable width or secant");
  }
  tangents_.resize(n);
  tangents_[0]     = delta[0];
  tangents_[n - 1] = delta[n - 2];
  for (std::size_t i = 1; i + 1 < n; ++i)
    tangents_[i] = .5 * delta[i - 1] + .5 * delta[i];
  decisions_.reserve(n - 1);
  for (std::size_t i = 0; i + 1 < n; ++i) {
    MonotoneCubicDecision r;
    r.interval     = i;
    r.secant       = delta[i];
    r.left_before  = tangents_[i];
    r.right_before = tangents_[i + 1];
    if (delta[i] == 0) {
      r.flat       = true;
      tangents_[i] = tangents_[i + 1] = 0;
    } else {
      if (tangents_[i] != 0 && std::signbit(tangents_[i]) != std::signbit(delta[i])) {
        r.left_clamped = true;
        tangents_[i]   = 0;
      }
      if (tangents_[i + 1] != 0 && std::signbit(tangents_[i + 1]) != std::signbit(delta[i])) {
        r.right_clamped  = true;
        tangents_[i + 1] = 0;
      }
      const long double alpha = static_cast<long double>(tangents_[i]) / delta[i],
                        beta  = static_cast<long double>(tangents_[i + 1]) / delta[i];
      const auto norm         = std::hypot(alpha, beta);
      if (norm > 3) {
        r.scaled         = true;
        const auto scale = 3 / norm;
        tangents_[i]     = static_cast<double>(scale * tangents_[i]);
        tangents_[i + 1] = static_cast<double>(scale * tangents_[i + 1]);
      }
    }
    r.left_after  = tangents_[i];
    r.right_after = tangents_[i + 1];
    decisions_.push_back(r);
  }
}
double MonotoneCubic::evaluate(double x) const {
  if (!std::isfinite(x))
    throw std::invalid_argument("MonotoneCubic::evaluate: finite query required");
  if (x <= xs_.front())
    return ys_.front();
  if (x >= xs_.back())
    return ys_.back();
  const auto it    = std::lower_bound(xs_.begin(), xs_.end(), x);
  const auto upper = std::size_t(it - xs_.begin());
  if (*it == x)
    return ys_[upper];
  const auto        k = upper - 1;
  const long double h = static_cast<long double>(xs_[k + 1]) - xs_[k],
                    t = (static_cast<long double>(x) - xs_[k]) / h, t2 = t * t, t3 = t2 * t;
  const auto value = (2 * t3 - 3 * t2 + 1) * ys_[k] + (t3 - 2 * t2 + t) * h * tangents_[k] +
                     (-2 * t3 + 3 * t2) * ys_[k + 1] + (t3 - t2) * h * tangents_[k + 1];
  const auto result = static_cast<double>(value);
  if (!std::isfinite(result))
    throw std::overflow_error("MonotoneCubic: result overflow");
  return result;
}
double
monotone_cubic_interpolate(const std::vector<double>& x, const std::vector<double>& y, double q) {
  return MonotoneCubic(x, y, 0).evaluate(q);
}
} // namespace datamunge::algorithms
