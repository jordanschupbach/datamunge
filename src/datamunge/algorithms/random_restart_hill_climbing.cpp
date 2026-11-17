/// @file random_restart_hill_climbing.cpp
/// @brief Shared actual-site coordinate-search and original scalar-grid producers.
#include <algorithm>
#include <cmath>
#include <datamunge/algorithms/random_restart_hill_climbing.hpp>
#include <limits>
#include <utility>
/// @brief Canonical derivative-free search implementation.
namespace datamunge::algorithms {
/// @brief Private search state and producer.
namespace {
/// @brief Historical nonzero xorshift state with modulo64-bit transitions.
struct HcRng {
  std::uint64_t state; ///< Nonzero owned raw state.

  /// @brief Initialize the historical raw stream, mapping zero to a fixed nonzero seed.
  /// @param seed Caller-selected raw seed.
  explicit HcRng(std::uint64_t seed) : state(seed ? seed : 0x9E3779B97F4A7C15ULL) {}
  /// @brief Consume one raw transition and map its high53 bits to[0,1).
  /// @return Actual finite uniform-grid sample.
  double uniform() {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    return (state >> 11) * (1. / 9007199254740992.);
  }
};
/// @brief Run the canonical bounded coordinate search, optionally retaining actual proposals.
/// @param f Borrowed synchronous objective.
/// @param lo Finite lower bounds.
/// @param hi Matching finite upper bounds.
/// @param o Execution policy.
/// @param capture Retain actual initial/proposal records.
/// @return Owned best visited state and selected work.
HillClimbRun solve(
    const HillObjective&       f,
    const std::vector<double>& lo,
    const std::vector<double>& hi,
    const HillClimbOptions&    o,
    bool                       capture) {
  const auto d = lo.size();
  if (!f || !d || hi.size() != d || !std::isfinite(o.step_fraction) || o.step_fraction <= 0 ||
      !std::isfinite(o.minimum_step) || o.minimum_step <= 0)
    throw std::invalid_argument("hill climb: invalid domain/callback/step");
  if ((o.max_dimensions && d > o.max_dimensions) ||
      (o.max_restarts && o.restarts > o.max_restarts) ||
      d > (std::numeric_limits<std::size_t>::max() - 1) / 3)
    throw std::length_error("hill climb: input capacity");
  double initial_step = 0;
  for (std::size_t i = 0; i < d; ++i) {
    if (!std::isfinite(lo[i]) || !std::isfinite(hi[i]) || hi[i] < lo[i])
      throw std::invalid_argument("hill climb: bounds");
    const double width = hi[i] - lo[i];
    const double step =
        std::isfinite(width)
            ? o.step_fraction * width
            : static_cast<double>(o.step_fraction * (static_cast<long double>(hi[i]) - lo[i]));
    if (!std::isfinite(step))
      throw std::overflow_error("hill climb: initial step");
    initial_step = std::max(initial_step, step);
  }
  HillClimbRun out;
  HcRng        rng(o.seed);
  std::size_t  history = 0;
  /// @brief Admit a real three-vector state record before its callback/copy.
  const auto admit = [&]() {
    if (!capture)
      return;
    const auto cost = 3 * d + 1;
    if (cost > std::numeric_limits<std::size_t>::max() - history ||
        (o.max_trace_entries &&
         (history > o.max_trace_entries || cost > o.max_trace_entries - history)))
      throw std::length_error("hill climb: history policy");
    history += cost;
  };
  /// @brief Invoke and count one actual finite objective evaluation.
  /// @param x Actual admissible point.
  /// @return Actual finite callback value.
  const auto evaluate = [&](const std::vector<double>& x) {
    if (out.evaluations == std::numeric_limits<std::size_t>::max() ||
        (o.max_evaluations && out.evaluations >= o.max_evaluations))
      throw std::length_error("hill climb: evaluation policy");
    ++out.evaluations;
    const double v = f(x);
    if (!std::isfinite(v))
      throw std::overflow_error("hill climb: nonfinite objective");
    return v;
  };
  for (std::size_t restart = 0; restart < o.restarts; ++restart) {
    std::vector<double> x(d);
    for (std::size_t i = 0; i < d; ++i) {
      const double u = rng.uniform(), width = hi[i] - lo[i];
      x[i] = std::isfinite(width) ? lo[i] + width * u : std::lerp(lo[i], hi[i], u);
      x[i] = std::clamp(x[i], lo[i], hi[i]);
    }
    admit();
    double value = evaluate(x), step = initial_step;
    if (capture)
      out.steps.push_back({restart, d, step, true, true, true, x, x, x, value, value, value});
    while (step > o.minimum_step) {
      bool improved = false;
      for (std::size_t i = 0; i < d; ++i)
        for (double change : {step, -step}) {
          if (out.proposals == std::numeric_limits<std::size_t>::max())
            throw std::length_error("hill climb: proposal capacity");
          ++out.proposals;
          admit();
          auto candidate = x;
          candidate[i] += change;
          const bool   evaluated = candidate[i] >= lo[i] && candidate[i] <= hi[i];
          const auto   before    = capture ? x : std::vector<double>{};
          const double old       = value;
          double       proposed  = 0;
          bool         accepted  = false;
          if (evaluated) {
            proposed = evaluate(candidate);
            if (proposed > value) {
              x        = candidate;
              value    = proposed;
              accepted = true;
              improved = true;
              ++out.accepted;
            }
          }
          if (capture)
            out.steps.push_back(
                {restart,
                 i,
                 step,
                 false,
                 evaluated,
                 accepted,
                 before,
                 std::move(candidate),
                 x,
                 old,
                 proposed,
                 value});
        }
      if (!improved) {
        step *= .5;
        ++out.shrinks;
      }
    }
    if (!out.found || value > out.best.value) {
      out.best  = {x, value};
      out.found = true;
    }
  }
  return out;
}
} // namespace
HillClimbRun hill_climb_run(
    const HillObjective&       f,
    const std::vector<double>& lo,
    const std::vector<double>& hi,
    const HillClimbOptions&    o) {
  return solve(f, lo, hi, o, false);
}
HillClimbRun hill_climb_trace(
    const HillObjective&       f,
    const std::vector<double>& lo,
    const std::vector<double>& hi,
    const HillClimbOptions&    o) {
  return solve(f, lo, hi, o, true);
}
double hill_wave_objective(const std::vector<double>& x) {
  if (x.size() != 1 || !std::isfinite(x[0]))
    throw std::invalid_argument("hill wave: one finite coordinate required");
  const double scaled = 2.2 * x[0], value = std::sin(x[0]) + std::sin(scaled) + .3 * x[0];
  if (!std::isfinite(scaled) || !std::isfinite(value))
    throw std::overflow_error("hill wave: nonfinite value");
  return value;
}
HillGrid hill_objective_grid(
    const HillObjective& f,
    double               lo,
    double               hi,
    double               step,
    std::size_t          max_points) {
  if (!f || !std::isfinite(lo) || !std::isfinite(hi) || hi < lo || !std::isfinite(step) ||
      step <= 0)
    throw std::invalid_argument("hill grid: invalid inputs");
  HillGrid g;
  for (double x = lo; x <= hi;) {
    if (max_points && g.x.size() >= max_points)
      throw std::length_error("hill grid: allocation policy");
    const double value = f({x});
    if (!std::isfinite(value))
      throw std::overflow_error("hill grid: nonfinite objective");
    g.x.push_back(x);
    g.values.push_back(value);
    if (g.x.size() == 1 || value > g.maximum.value)
      g.maximum = {{x}, value};
    const double next = x + step;
    if (next > hi)
      break;
    if (!std::isfinite(next) || next <= x)
      throw std::overflow_error("hill grid: nonadvancing step");
    x = next;
  }
  return g;
}
} // namespace datamunge::algorithms
