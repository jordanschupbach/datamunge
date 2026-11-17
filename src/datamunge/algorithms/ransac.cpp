/// @file ransac.cpp
/// @brief One checked sampled-line/TLS producer and actual hypothesis histories.
#include <algorithm>
#include <cmath>
#include <datamunge/algorithms/ransac.hpp>
#include <random>
#include <stdexcept>
#include <utility>
/// @brief Canonical robust line estimation implementation.
namespace datamunge::algorithms {
/// @brief Private widened line geometry and one shared sampled producer.
namespace {
/// @brief Narrow one actual geometric scalar without returning infinity/NaN.
/// @param value Actual widened arithmetic result.
/// @return Finite double.
double narrow(long double value) {
  const auto x = static_cast<double>(value);
  if (!std::isfinite(x))
    throw std::overflow_error("RANSAC: geometric range");
  return x;
}
/// @brief Fit the actual orthogonal line to one selected nonempty point subset.
/// @param x Complete x coordinates.
/// @param y Complete y coordinates.
/// @param ids Actual winning inlier ids.
/// @return Normalized finite coefficients; consensus is rescored by the caller.
RansacLine tls(
    const std::vector<double>&      x,
    const std::vector<double>&      y,
    const std::vector<std::size_t>& ids) {
  long double mx = 0, my = 0;
  for (auto i : ids) {
    mx += x[i];
    my += y[i];
  }
  mx /= ids.size();
  my /= ids.size();
  long double xx = 0, yy = 0, xy = 0;
  for (auto i : ids) {
    const long double dx = static_cast<long double>(x[i]) - mx,
                      dy = static_cast<long double>(y[i]) - my;
    xx += dx * dx;
    yy += dy * dy;
    xy += dx * dy;
  }
  const long double theta = .5L * std::atan2(2 * xy, xx - yy), a = -std::sin(theta),
                    b = std::cos(theta);
  RansacLine line;
  line.a = narrow(a);
  line.b = narrow(b);
  line.c = narrow(-(a * mx + b * my));
  return line;
}
/// @brief Evaluate one represented normalized-line perpendicular residual.
/// @param line Actual stored coefficients.
/// @param x Finite x coordinate.
/// @param y Finite y coordinate.
/// @return Finite absolute represented-line residual.
double distance(const RansacLine& line, double x, double y) {
  return narrow(
      std::fabs(
          static_cast<long double>(line.a) * x + static_cast<long double>(line.b) * y + line.c));
}
/// @brief Execute one actual index-sampling and TLS-refit stream.
/// @param x Complete finite x coordinates.
/// @param y Complete finite matching y coordinates.
/// @param threshold Nonnegative finite represented residual threshold.
/// @param iterations Attempted independent index pairs.
/// @param seed Native MT19937-64 seed.
/// @param o Execution/history policies.
/// @param capture Retain actual sampled trial records.
/// @return Owned sampled winner, rescored refit and selected work.
RansacRun solve(
    const std::vector<double>& x,
    const std::vector<double>& y,
    double                     threshold,
    std::size_t                iterations,
    std::uint64_t              seed,
    const RansacOptions&       o,
    bool                       capture) {
  const auto n = x.size();
  if (n != y.size() || !std::isfinite(threshold) || threshold < 0)
    throw std::invalid_argument("RANSAC: shape/threshold");
  for (std::size_t i = 0; i < n; ++i)
    if (!std::isfinite(x[i]) || !std::isfinite(y[i]))
      throw std::invalid_argument("RANSAC: nonfinite coordinate");
  if ((o.max_points && n > o.max_points) || (o.max_trials && iterations > o.max_trials) ||
      iterations > std::numeric_limits<std::size_t>::max() / 2)
    throw std::length_error("RANSAC: input/trial capacity");
  RansacRun out;
  if (n < 2)
    return out;
  std::mt19937_64                            rng(seed);
  std::uniform_int_distribution<std::size_t> pick(0, n - 1);
  std::size_t                                history = 0;
  /// @brief Count/admit one actual perpendicular residual test.
  const auto tick = [&]() {
    if (out.distance_tests == std::numeric_limits<std::size_t>::max() ||
        (o.max_distance_tests && out.distance_tests >= o.max_distance_tests))
      throw std::length_error("RANSAC: distance policy");
    ++out.distance_tests;
  };
  for (std::size_t trial = 0; trial < iterations; ++trial) {
    const auto i = pick(rng), j = pick(rng);
    out.draws += 2;
    RansacTrial record;
    record.iteration = trial;
    record.first     = i;
    record.second    = j;
    RansacLine line;
    if (i == j) {
      record.kind = RansacTrialKind::DuplicateIndex;
      ++out.skipped_duplicates;
    } else if (x[i] == x[j] && y[i] == y[j]) {
      record.kind = RansacTrialKind::CoincidentPoints;
      ++out.skipped_coincident;
    } else {
      record.kind          = RansacTrialKind::Scored;
      const long double dx = static_cast<long double>(x[j]) - x[i],
                        dy = static_cast<long double>(y[j]) - y[i], norm = std::hypot(dx, dy),
                        a = dy / norm, b = -dx / norm;
      line.a = narrow(a);
      line.b = narrow(b);
      line.c = narrow(-(a * x[i] + b * y[i]));
      line.inliers.assign(n, 0);
      for (std::size_t k = 0; k < n; ++k) {
        tick();
        if (distance(line, x[k], y[k]) <= threshold) {
          line.inliers[k] = 1;
          ++line.inlier_count;
        }
      }
      record.a        = line.a;
      record.b        = line.b;
      record.c        = line.c;
      record.count    = line.inlier_count;
      record.improved = line.inlier_count > out.hypothesis.inlier_count;
      if (record.improved) {
        out.hypothesis    = line;
        out.found         = true;
        out.winning_trial = trial;
      }
      if (capture)
        record.inliers = std::move(line.inliers);
    }
    if (capture) {
      const auto cost = record.inliers.size() + 1;
      if (cost > std::numeric_limits<std::size_t>::max() - history ||
          (o.max_trace_entries &&
           (history > o.max_trace_entries || cost > o.max_trace_entries - history)))
        throw std::length_error("RANSAC: history policy");
      history += cost;
      out.trials.push_back(std::move(record));
    }
  }
  out.hypothesis.trials = iterations;
  out.line.trials       = iterations;
  if (!out.found)
    return out;
  out.line = out.hypothesis;
  std::vector<std::size_t> ids;
  for (std::size_t i = 0; i < n; ++i)
    if (out.hypothesis.inliers[i])
      ids.push_back(i);
  if (ids.size() >= 2) {
    out.line        = tls(x, y, ids);
    out.refitted    = true;
    out.line.trials = iterations;
    out.line.inliers.assign(n, 0);
    for (std::size_t i = 0; i < n; ++i) {
      tick();
      if (distance(out.line, x[i], y[i]) <= threshold) {
        out.line.inliers[i] = 1;
        ++out.line.inlier_count;
      }
    }
  }
  return out;
}
/// @brief Convert one ideal good-attempt probability into a finite success ceiling.
/// @param good Good-model probability per independent attempt.
/// @param target Requested success probability.
/// @return Exact integer ceiling of the represented logarithmic model.
std::size_t attempts(long double good, double target) {
  if (!std::isfinite(target) || target < 0 || target > 1)
    throw std::invalid_argument("RANSAC probability: target");
  if (target == 0)
    return 0;
  if (good >= 1)
    return 1;
  if (good <= 0 || target == 1)
    throw std::domain_error("RANSAC probability: no finite bound");
  const long double n =
      std::ceil(std::log1p(-static_cast<long double>(target)) / std::log1p(-good));
  if (!std::isfinite(n) || n > std::numeric_limits<std::size_t>::max())
    throw std::length_error("RANSAC probability: count range");
  return static_cast<std::size_t>(n);
}
} // namespace
RansacRun ransac_line_run(
    const std::vector<double>& x,
    const std::vector<double>& y,
    double                     threshold,
    std::size_t                iterations,
    std::uint64_t              seed,
    const RansacOptions&       o) {
  return solve(x, y, threshold, iterations, seed, o, false);
}
RansacRun ransac_line_trace(
    const std::vector<double>& x,
    const std::vector<double>& y,
    double                     threshold,
    std::size_t                iterations,
    std::uint64_t              seed,
    const RansacOptions&       o) {
  return solve(x, y, threshold, iterations, seed, o, true);
}
RansacLine ransac_line(
    const std::vector<double>& x,
    const std::vector<double>& y,
    double                     threshold,
    std::size_t                iterations,
    std::uint64_t              seed) {
  return ransac_line_run(x, y, threshold, iterations, seed).line;
}
std::size_t ransac_required_iterations(double w, std::size_t sample, double p) {
  if (!std::isfinite(w) || w < 0 || w > 1 || !sample)
    throw std::invalid_argument("RANSAC probability: fraction/sample");
  return attempts(std::pow(static_cast<long double>(w), static_cast<long double>(sample)), p);
}
std::size_t ransac_pair_iterations(std::size_t inliers, std::size_t total, double p) {
  if (!total || inliers > total)
    throw std::invalid_argument("RANSAC probability: counts");
  const long double good = inliers < 2 ? 0
                                       : (static_cast<long double>(inliers) / total) *
                                             (static_cast<long double>(inliers - 1) / total);
  return attempts(good, p);
}
RansacPoints ransac_example_points(
    std::size_t   inliers,
    std::size_t   outliers,
    std::uint64_t seed,
    std::size_t   max_points) {
  if (outliers > std::numeric_limits<std::size_t>::max() - inliers ||
      (max_points && inliers + outliers > max_points))
    throw std::length_error("RANSAC fixture: allocation policy");
  RansacPoints                           p;
  std::mt19937_64                        rng(seed);
  std::normal_distribution<double>       noise(0, .05);
  std::uniform_real_distribution<double> ux(0, 10), uy(-20, 40);
  for (std::size_t i = 0; i < inliers; ++i) {
    const double x = ux(rng);
    p.x.push_back(x);
    p.y.push_back(2 * x + 1 + noise(rng));
    p.generated_inlier.push_back(1);
  }
  for (std::size_t i = 0; i < outliers; ++i) {
    p.x.push_back(ux(rng));
    p.y.push_back(uy(rng));
    p.generated_inlier.push_back(0);
  }
  return p;
}
} // namespace datamunge::algorithms
