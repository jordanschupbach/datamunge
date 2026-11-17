/// @file runge_kutta4.cpp
/// @brief One checked actual four-stage trajectory producer without report-side replay.
#include <cmath>
#include <datamunge/algorithms/runge_kutta4.hpp>
#include <limits>
#include <stdexcept>
#include <utility>
/// @brief Canonical classical RK4 implementation.
namespace datamunge::algorithms {
/// @brief Private complete geometry and stage execution.
namespace {
/// @brief Convert actual extended time to a finite scalar.
/// @param value Extended time arithmetic.
/// @return Finite double time.
/// @throws std::overflow_error For unrepresentable time.
double finite_time(long double value) {
  if (!std::isfinite(value) || std::fabs(value) > std::numeric_limits<double>::max())
    throw std::overflow_error("RK4: time range");
  return static_cast<double>(value);
}
/// @brief Execute validated actual four-stage calls and commit one intrinsic trajectory.
/// @param f Synchronous callback.
/// @param y Owned state.
/// @param t0 Initial time.
/// @param t1 Final time.
/// @param count Positive step count.
/// @param o Complete policies.
/// @param capture Retain stage states and actual derivatives.
/// @return Owned intrinsic trajectory and actual stage work.
RK4Run solve(
    const OdeSystem&    f,
    std::vector<double> y,
    double              t0,
    double              t1,
    int                 count,
    const RK4Options&   o,
    bool                capture) {
  if (!f || y.empty() || count <= 0 || !std::isfinite(t0) || !std::isfinite(t1))
    throw std::invalid_argument("RK4: callback/state/time/steps");
  for (double x : y)
    if (!std::isfinite(x))
      throw std::invalid_argument("RK4: nonfinite initial state");
  const auto n = static_cast<std::size_t>(count), d = y.size(),
             max = std::numeric_limits<std::size_t>::max();
  if (d == max || n == max || (o.max_steps && n > o.max_steps) ||
      (o.max_dimension && d > o.max_dimension) || (n + 1) > max / (d + 1) ||
      (o.max_solution_entries && (n + 1) * (d + 1) > o.max_solution_entries) || n > max / 4 ||
      n > max / d || d > (max - 1) / 9)
    throw std::length_error("RK4: intrinsic geometry policy");
  double h = (t1 - t0) / count;
  if (!std::isfinite(h))
    h = finite_time((static_cast<long double>(t1) - t0) / count);
  if (h == 0 && t0 != t1)
    throw std::underflow_error("RK4: vanished step");
  std::vector<double> grid(n + 1);
  grid[0] = t0;
  for (std::size_t i = 1; i <= n; ++i) {
    grid[i] = i == n ? t1 : t0 + static_cast<double>(i) * h;
    if (!std::isfinite(grid[i]))
      grid[i] = finite_time(static_cast<long double>(t0) + static_cast<long double>(i) * h);
    if (t0 != t1 && !((t1 > t0 && grid[i] > grid[i - 1]) || (t1 < t0 && grid[i] < grid[i - 1])))
      throw std::underflow_error("RK4: nonadvancing grid");
    finite_time(static_cast<long double>(grid[i - 1]) + .5L * h);
    finite_time(static_cast<long double>(grid[i - 1]) + h);
  }
  RK4Run out;
  out.step_size  = h;
  out.solution.t = grid;
  out.solution.y.reserve(n + 1);
  out.solution.y.push_back(y);
  std::size_t retained = 0;
  /// @brief Invoke and validate one actual derivative vector.
  /// @param time Actual stage time.
  /// @param state Actual stage input.
  /// @return Actual matching finite derivative vector.
  const auto evaluate = [&](double time, const std::vector<double>& state) {
    ++out.rhs_calls;
    auto value = f(time, state);
    if (value.size() != d)
      throw std::invalid_argument("RK4: derivative shape");
    for (double x : value)
      if (!std::isfinite(x))
        throw std::invalid_argument("RK4: nonfinite derivative");
    return value;
  };
  /// @brief Form one actual finite intermediate stage state.
  /// @param state Initial step state.
  /// @param factor Actual half/full nominal step.
  /// @param slope Actual previous stage derivative.
  /// @return Owned actual finite callback state.
  const auto stage =
      [&](const std::vector<double>& state, double factor, const std::vector<double>& slope) {
        auto result = state;
        for (std::size_t i = 0; i < d; ++i) {
          result[i] += factor * slope[i];
          if (!std::isfinite(result[i]))
            throw std::overflow_error("RK4: stage range");
        }
        return result;
      };
  for (std::size_t i = 0; i < n; ++i) {
    if (capture) {
      const auto cost = 9 * d + 1;
      if (cost > max - retained || (o.max_trace_entries && (retained > o.max_trace_entries ||
                                                            cost > o.max_trace_entries - retained)))
        throw std::length_error("RK4: history policy");
      retained += cost;
    }
    const double time = grid[i], mid = finite_time(static_cast<long double>(time) + .5L * h),
                 end = finite_time(static_cast<long double>(time) + h);
    auto k1          = evaluate(time, y);
    auto second      = stage(y, .5 * h, k1);
    auto k2          = evaluate(mid, second);
    auto third       = stage(y, .5 * h, k2);
    auto k3          = evaluate(mid, third);
    auto fourth      = stage(y, h, k3);
    auto k4          = evaluate(end, fourth);
    auto after       = y;
    for (std::size_t j = 0; j < d; ++j) {
      after[j] += (h / 6) * (k1[j] + 2 * k2[j] + 2 * k3[j] + k4[j]);
      if (!std::isfinite(after[j]))
        throw std::overflow_error("RK4: committed range");
      ++out.scalar_updates;
    }
    if (capture)
      out.steps.push_back(
          {i,
           time,
           mid,
           end,
           y,
           std::move(second),
           std::move(third),
           std::move(fourth),
           std::move(k1),
           std::move(k2),
           std::move(k3),
           std::move(k4),
           after});
    y = std::move(after);
    out.solution.y.push_back(y);
  }
  return out;
}
} // namespace
RK4Run runge_kutta4_run(
    const OdeSystem&    f,
    std::vector<double> y,
    double              t0,
    double              t1,
    int                 n,
    const RK4Options&   o) {
  return solve(f, std::move(y), t0, t1, n, o, false);
}
RK4Run runge_kutta4_trace(
    const OdeSystem&    f,
    std::vector<double> y,
    double              t0,
    double              t1,
    int                 n,
    const RK4Options&   o) {
  return solve(f, std::move(y), t0, t1, n, o, true);
}
OdeSolution runge_kutta4(const OdeSystem& f, std::vector<double> y, double t0, double t1, int n) {
  return solve(f, std::move(y), t0, t1, n, {}, false).solution;
}
} // namespace datamunge::algorithms
