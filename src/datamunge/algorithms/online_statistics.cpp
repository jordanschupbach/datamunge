/// @file online_statistics.cpp
/// @brief Atomic finite Welford/Chan moments, actual observations and original offset diagnostics.
#include <cmath>
#include <datamunge/algorithms/online_statistics.hpp>
#include <limits>
#include <random>
#include <set>
#include <stdexcept>
namespace datamunge::algorithms {
/// @brief Private checked scalar/count admission for complete atomic state commits.
namespace {
/// @brief Add accepted observation counts without wraparound.
/// @param a Existing count.
/// @param b Added count.
/// @return Checked combined count.
/// @throws std::length_error If addition overflows.
std::size_t add_count(std::size_t a, std::size_t b) {
  if (b > std::numeric_limits<std::size_t>::max() - a)
    throw std::length_error("online moments: count geometry");
  return a + b;
}
/// @brief Round one finite representable wide result to a checked finite double.
/// @param value Actual computed wide intermediate.
/// @return Rounded finite double, underflow to zero allowed.
/// @throws std::overflow_error If input/result cannot be finite double.
double finite_double(long double value) {
  double rounded = static_cast<double>(value);
  if (!std::isfinite(value) || !std::isfinite(rounded))
    throw std::overflow_error("online moments: finite-double state/diagnostic required");
  return rounded;
}
} // namespace
void OnlineVariance::add(double x) {
  (void)add_observed(x);
}
OnlineVarianceUpdate OnlineVariance::add_observed(double x) {
  if (!std::isfinite(x))
    throw std::invalid_argument("online variance: finite observation required");
  OnlineVarianceUpdate u;
  u.before     = state();
  u.input      = x;
  auto   n     = add_count(count_, 1);
  double delta = x - mean_, next_mean = mean_ + delta / static_cast<double>(n),
         delta2 = x - next_mean, increment = delta * delta2, next_m2 = m2_ + increment;
  u.widened = !std::isfinite(delta) || !std::isfinite(next_mean) || !std::isfinite(delta2) ||
              !std::isfinite(increment) || !std::isfinite(next_m2);
  if (u.widened) {
    u.delta        = static_cast<long double>(x) - mean_;
    auto wide_mean = static_cast<long double>(mean_) + u.delta / n;
    next_mean      = finite_double(wide_mean);
    u.delta_after  = static_cast<long double>(x) - next_mean;
    u.increment    = u.delta * u.delta_after;
    next_m2        = finite_double(static_cast<long double>(m2_) + u.increment);
  } else {
    u.delta       = delta;
    u.delta_after = delta2;
    u.increment   = increment;
  }
  if (next_m2 < 0)
    throw std::overflow_error("online variance: negative computed M2");
  u.after = {n, next_mean, next_m2};
  count_  = n;
  mean_   = next_mean;
  m2_     = next_m2;
  return u;
}
std::size_t OnlineVariance::count() const noexcept {
  return count_;
}
double OnlineVariance::mean() const noexcept {
  return mean_;
}
double OnlineVariance::variance() const noexcept {
  return count_ ? m2_ / static_cast<double>(count_) : 0;
}
double OnlineVariance::sample_variance() const noexcept {
  return count_ > 1 ? m2_ / static_cast<double>(count_ - 1) : 0;
}
double OnlineVariance::sample_stddev() const noexcept {
  return std::sqrt(sample_variance());
}
double OnlineVariance::sum_of_squares() const noexcept {
  return m2_;
}
OnlineVarianceState OnlineVariance::state() const noexcept {
  return {count_, mean_, m2_};
}
void OnlineVariance::merge(const OnlineVariance& other) {
  (void)merge_observed(other);
}
OnlineVarianceMerge OnlineVariance::merge_observed(const OnlineVariance& other) {
  OnlineVarianceMerge m;
  m.left  = state();
  m.right = other.state();
  if (!m.right.count) {
    m.after = m.left;
    return m;
  }
  if (!m.left.count) {
    *this   = other;
    m.after = state();
    return m;
  }
  auto   count = add_count(m.left.count, m.right.count);
  double na = static_cast<double>(m.left.count), nb = static_cast<double>(m.right.count),
         n = na + nb, delta = m.right.mean - m.left.mean, next_mean = m.left.mean + delta * nb / n,
         correction = delta * delta * na * nb / n, next_m2 = m.left.m2 + m.right.m2 + correction;
  m.widened = !std::isfinite(delta) || !std::isfinite(next_mean) || !std::isfinite(correction) ||
              !std::isfinite(next_m2);
  if (m.widened) {
    m.delta        = static_cast<long double>(m.right.mean) - m.left.mean;
    long double wa = m.left.count, wb = m.right.count, wn = wa + wb;
    m.correction = m.delta * m.delta * wa * wb / wn;
    next_mean    = finite_double(static_cast<long double>(m.left.mean) + m.delta * wb / wn);
    next_m2      = finite_double(static_cast<long double>(m.left.m2) + m.right.m2 + m.correction);
  } else {
    m.delta      = delta;
    m.correction = correction;
  }
  if (next_m2 < 0)
    throw std::overflow_error("online variance: negative merged M2");
  m.after = {count, next_mean, next_m2};
  count_  = count;
  mean_   = next_mean;
  m2_     = next_m2;
  return m;
}
void OnlineCovariance::add(double x, double y) {
  if (!std::isfinite(x) || !std::isfinite(y))
    throw std::invalid_argument("online covariance: finite complete pair required");
  auto count = add_count(count_, 1);
  auto vx = vx_, vy = vy_;
  vx.add(x);
  vy.add(y);
  double dx = x - mean_x_, next_x = mean_x_ + dx / static_cast<double>(count), dy = y - mean_y_,
         next_y = mean_y_ + dy / static_cast<double>(count), increment = dx * (y - next_y),
         next_c = c_ + increment;
  if (!std::isfinite(dx) || !std::isfinite(dy) || !std::isfinite(next_x) ||
      !std::isfinite(next_y) || !std::isfinite(increment) || !std::isfinite(next_c)) {
    long double wx = static_cast<long double>(x) - mean_x_,
                wy = static_cast<long double>(y) - mean_y_;
    next_x         = finite_double(static_cast<long double>(mean_x_) + wx / count);
    next_y         = finite_double(static_cast<long double>(mean_y_) + wy / count);
    next_c =
        finite_double(static_cast<long double>(c_) + wx * (static_cast<long double>(y) - next_y));
  }
  count_  = count;
  mean_x_ = next_x;
  mean_y_ = next_y;
  c_      = next_c;
  vx_     = vx;
  vy_     = vy;
}
std::size_t OnlineCovariance::count() const noexcept {
  return count_;
}
double OnlineCovariance::mean_x() const noexcept {
  return mean_x_;
}
double OnlineCovariance::mean_y() const noexcept {
  return mean_y_;
}
double OnlineCovariance::covariance() const noexcept {
  return count_ ? c_ / static_cast<double>(count_) : 0;
}
double OnlineCovariance::sample_covariance() const noexcept {
  return count_ > 1 ? c_ / static_cast<double>(count_ - 1) : 0;
}
double OnlineCovariance::correlation() const noexcept {
  if (vx_.sum_of_squares() <= 0 || vy_.sum_of_squares() <= 0)
    return 0;
  return static_cast<double>(
      static_cast<long double>(c_) / std::sqrt(static_cast<long double>(vx_.sum_of_squares())) /
      std::sqrt(static_cast<long double>(vy_.sum_of_squares())));
}
OnlineVarianceTrace online_variance_trace(
    const std::vector<double>& values,
    std::size_t                max_records) {
  for (double x : values)
    if (!std::isfinite(x))
      throw std::invalid_argument("online variance trace: complete finite inputs required");
  auto count = add_count(values.size(), 2);
  if (max_records && count > max_records)
    throw std::length_error("online variance trace: complete record quota");
  OnlineVarianceTrace t;
  t.captured_records = count;
  t.updates.reserve(values.size());
  OnlineVariance accumulator;
  t.initial = accumulator.state();
  for (double x : values)
    t.updates.push_back(accumulator.add_observed(x));
  t.final = accumulator.state();
  return t;
}
OnlineVarianceDemoData online_variance_demo_data() {
  std::mt19937_64                  rng(1);
  std::normal_distribution<double> normal(0, 1);
  OnlineVarianceDemoData           d;
  d.base.resize(10000);
  for (auto& x : d.base)
    x = normal(rng);
  d.x.resize(5000);
  d.y.resize(5000);
  for (std::size_t i = 0; i < 5000; ++i) {
    d.x[i] = normal(rng);
    d.y[i] = 3 * d.x[i] + .5 * normal(rng);
  }
  return d;
}
std::vector<VarianceOffsetRow> online_variance_offset_study(
    const std::vector<double>& base,
    const std::vector<int>&    exponents,
    std::size_t                max_values) {
  if (base.size() < 2)
    throw std::invalid_argument("online variance study: at least2 finite observations required");
  for (double x : base)
    if (!std::isfinite(x))
      throw std::invalid_argument("online variance study: complete finite base required");
  if (exponents.size() > std::numeric_limits<std::size_t>::max() / base.size() ||
      (max_values && exponents.size() * base.size() > max_values))
    throw std::length_error("online variance study: complete work geometry");
  for (int e : exponents) {
    double offset = std::pow(10., e);
    if (!std::isfinite(offset) || offset <= 0)
      throw std::invalid_argument("online variance study: finite positive offsets required");
    for (double x : base)
      if (!std::isfinite(x + offset))
        throw std::invalid_argument("online variance study: complete shifted inputs finite");
  }
  std::vector<VarianceOffsetRow> rows;
  rows.reserve(exponents.size());
  for (int e : exponents) {
    VarianceOffsetRow r;
    r.exponent = e;
    r.offset   = std::pow(10., e);
    r.count    = base.size();
    std::vector<double> data(base.size());
    std::set<double>    distinct;
    double              mean = 0, sum = 0, sum2 = 0;
    OnlineVariance      accumulator;
    for (std::size_t i = 0; i < data.size(); ++i) {
      data[i] = base[i] + r.offset;
      distinct.insert(data[i]);
      mean += data[i];
      sum += data[i];
      sum2 += data[i] * data[i];
      accumulator.add(data[i]);
    }
    r.distinct_values = distinct.size();
    mean /= data.size();
    double ss = 0;
    for (double x : data)
      ss += (x - mean) * (x - mean);
    r.two_pass         = ss / (data.size() - 1);
    double naive_mean  = sum / data.size();
    r.naive            = (sum2 - data.size() * naive_mean * naive_mean) / (data.size() - 1);
    r.welford          = accumulator.sample_variance();
    long double origin = data[0], center = 0;
    for (double x : data)
      center += static_cast<long double>(x) - origin;
    center /= data.size();
    long double wide_ss = 0;
    for (double x : data) {
      auto delta = (static_cast<long double>(x) - origin) - center;
      wide_ss += delta * delta;
    }
    r.wide_reference = wide_ss / (data.size() - 1);
    if (!std::isfinite(r.two_pass) || !std::isfinite(r.naive) || !std::isfinite(r.wide_reference))
      throw std::overflow_error("online variance study: finite diagnostics required");
    r.double_reference_spread = r.two_pass > 0;
    r.wide_reference_spread   = r.wide_reference > 0;
    if (r.double_reference_spread) {
      r.naive_relative_error =
          finite_double(std::abs(static_cast<long double>(r.naive) - r.two_pass) / r.two_pass);
      r.welford_relative_error =
          finite_double(std::abs(static_cast<long double>(r.welford) - r.two_pass) / r.two_pass);
    }
    if (r.wide_reference_spread)
      r.welford_wide_relative_error = finite_double(
          std::abs(static_cast<long double>(r.welford) - r.wide_reference) / r.wide_reference);
    rows.push_back(r);
  }
  return rows;
}
} // namespace datamunge::algorithms
