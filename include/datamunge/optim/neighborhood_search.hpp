#pragma once

/// @file
/// @brief Typed, budgeted single-trajectory search over discrete neighborhoods.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

namespace datamunge::optim {

/// @brief Acceptance rules for discrete neighborhood search.
/// @par Example
/// @code
/// const auto rule = dm::optim::NeighborhoodAcceptance::annealing;
/// @endcode
/// @see neighborhood_search()
enum class NeighborhoodAcceptance {
  improving,  ///< Accept strictly improving moves.
  annealing,  ///< Metropolis acceptance with geometric cooling.
  threshold,  ///< Accept losses within a decreasing threshold of the current loss.
  record,     ///< Accept losses within a fixed tolerance of the best loss.
  deluge,     ///< Accept losses at or below a linearly decreasing water level.
  extended_deluge, ///< Accept improvements or losses under a level scheduled to a target.
  late        ///< Accept non-worsening moves or moves below the older loss in a ring buffer.
};

/// @brief Reason a discrete search stops.
/// @par Example
/// @code
/// const auto reason = dm::optim::NeighborhoodStop::budget;
/// @endcode
/// @see NeighborhoodResult
enum class NeighborhoodStop {
  budget,          ///< The objective-call budget, including the start, was exhausted.
  no_move          ///< No proposal, admissible move, or local improvement remained.
};

/// @brief Controls for the selected discrete acceptance rule.
/// @par Example
/// @code
/// dm::optim::NeighborhoodSearchOptions options;
/// options.acceptance = dm::optim::NeighborhoodAcceptance::late;
/// options.history_length = 20;
/// @endcode
/// @see neighborhood_search()
struct NeighborhoodSearchOptions {
  std::size_t evaluations{1000}; ///< Positive objective-call cap, including initialization.
  std::uint64_t seed{42}; ///< Seed of the local mt19937_64 used by proposals and acceptance.
  NeighborhoodAcceptance acceptance{NeighborhoodAcceptance::improving}; ///< Acceptance rule.
  double initial_temperature{1}; ///< Positive initial Metropolis temperature, in loss units.
  double cooling{0.99}; ///< Multiplicative temperature or threshold factor in (0, 1].
  double tolerance{0}; ///< Nonnegative initial threshold or constant record tolerance.
  double initial_level{0}; ///< Initial water level for classical deluge, in loss units.
  double target_level{0}; ///< Extended deluge target; must be finite and at most the start loss.
  double rain_speed{0.01}; ///< Nonnegative reduction of the water level per proposal.
  std::size_t history_length{16}; ///< Positive late-acceptance window in objective calls.
};

/// @brief Owned summary of a discrete neighborhood search.
/// @tparam State Type representing a complete feasible state.
/// @par Example
/// @code
/// const dm::optim::NeighborhoodResult<int> result{0, 0.0, 1, 0,
///     dm::optim::NeighborhoodStop::budget};
/// @endcode
/// @see neighborhood_search()
template <class State>
struct NeighborhoodResult {
  State best; ///< Earliest state having the lowest observed loss.
  double value; ///< Objective value already evaluated at best.
  std::size_t evaluations; ///< Actual score calls, including the start and duplicates.
  std::size_t accepted; ///< Number of accepted proposals.
  NeighborhoodStop stop; ///< Observed stop reason.
};

namespace detail {

/// @brief Validate the common budget and selected rule's controls.
/// @param options Search configuration.
/// @throws std::invalid_argument Invalid controls or unknown acceptance rule.
inline void validate_neighborhood(const NeighborhoodSearchOptions& options) {
  if (options.evaluations == 0) throw std::invalid_argument("Neighborhood budget must be positive");
  const bool valid_cooling = options.cooling > 0 && options.cooling <= 1;
  const bool valid_tolerance = std::isfinite(options.tolerance) && options.tolerance >= 0;
  bool valid = false;
  switch (options.acceptance) {
    case NeighborhoodAcceptance::improving:
      valid = true;
      break;
    case NeighborhoodAcceptance::annealing:
      valid = valid_cooling && std::isfinite(options.initial_temperature) &&
              options.initial_temperature > 0;
      break;
    case NeighborhoodAcceptance::threshold:
      valid = valid_cooling && valid_tolerance;
      break;
    case NeighborhoodAcceptance::record:
      valid = valid_tolerance;
      break;
    case NeighborhoodAcceptance::deluge:
      valid = std::isfinite(options.initial_level) && std::isfinite(options.rain_speed) &&
              options.rain_speed >= 0;
      break;
    case NeighborhoodAcceptance::extended_deluge:
      valid = std::isfinite(options.target_level);
      break;
    case NeighborhoodAcceptance::late:
      valid = options.history_length > 0;
      break;
  }
  if (!valid) throw std::invalid_argument("Invalid neighborhood acceptance controls");
}

} // namespace detail

/// @brief Minimize a discrete objective using typed feasible moves.
///
/// The initial state is scored once. Each proposed move is applied to a copy
/// of the current state and scored once, including duplicates. The optional
/// move may be empty to terminate. Improving-only search accepts strict
/// improvements. The other policies also support sideways or worsening moves.
/// Callbacks are responsible for feasibility and their exceptions propagate.
/// Seeded replay holds within a build for deterministic callbacks.
/// @tparam State Copyable complete state type.
/// @tparam Score Callable returning a finite loss for const State&.
/// @tparam Propose Callable returning std::optional<Move> from const State& and std::mt19937_64&.
/// @tparam Apply Callable applying const Move& to State& in place.
/// @param initial Valid starting state, taken by value.
/// @param score Objective to minimize, invoked for feasible const states.
/// @param propose Proposal generator with borrowed state and engine.
/// @param apply Move applicator responsible for preserving feasibility.
/// @param options Evaluation cap and acceptance controls (defaults to strict improvement).
/// @return Owned best state, observed loss and work counts.
/// @throws std::invalid_argument Invalid controls or extended deluge target above the starting loss.
/// @throws std::domain_error The score callback returns a nonfinite loss.
/// @par Example
/// @code
/// const auto result = dm::optim::neighborhood_search(
///     4, [](int x) { return x * x; },
///     [](int x, auto&) -> std::optional<int> { if (x > 0) return -1; return {}; },
///     [](int& x, int move) { x += move; });
/// std::print("{}\n", result.best); // 0
/// @endcode
/// @par References
/// - Kirkpatrick, Gelatt and Vecchi (1983). https://doi.org/10.1126/science.220.4598.671
/// - Dueck and Scheuer (1990). https://doi.org/10.1016/0021-9991(90)90201-B
/// - Dueck (1993). https://doi.org/10.1006/jcph.1993.1010
/// - Burke, Bykov, Newall and Petrović (2003). https://doi.org/10.2298/YJOR0302139B
/// - Burke and Bykov (2017). https://doi.org/10.1016/j.ejor.2016.07.012
/// @see NeighborhoodSearchOptions, NeighborhoodResult
template <class State, class Score, class Propose, class Apply>
NeighborhoodResult<State> neighborhood_search(
    State initial, Score&& score, Propose&& propose, Apply&& apply,
    const NeighborhoodSearchOptions& options = {}) {
  detail::validate_neighborhood(options);
  /// Score a feasible const state, requiring a finite objective value.
  const auto evaluate = [&](const State& state) {
    const double value = std::invoke(score, state);
    if (!std::isfinite(value)) throw std::domain_error("Neighborhood loss must be finite");
    return value;
  };

  double current_value = evaluate(initial);
  const double initial_value = current_value;
  if (options.acceptance == NeighborhoodAcceptance::extended_deluge &&
      options.target_level > initial_value) {
    throw std::invalid_argument("Extended deluge target must not exceed the initial loss");
  }
  NeighborhoodResult<State> result{initial, current_value, 1, 0, NeighborhoodStop::budget};
  State current = std::move(initial);
  std::mt19937_64 rng(options.seed);
  std::uniform_real_distribution<double> uniform;
  double temperature = options.initial_temperature;
  double threshold = options.tolerance;
  double level = options.initial_level;
  std::vector<double> history;
  if (options.acceptance == NeighborhoodAcceptance::late) {
    history.assign(std::min(options.history_length, options.evaluations - 1), current_value);
  }

  while (result.evaluations < options.evaluations) {
    const auto move = std::invoke(propose, std::as_const(current), rng);
    if (!move) {
      result.stop = NeighborhoodStop::no_move;
      break;
    }
    State candidate = current;
    std::invoke(apply, candidate, *move);
    const double value = evaluate(candidate);
    ++result.evaluations;
    bool accept = value < current_value;
    switch (options.acceptance) {
      case NeighborhoodAcceptance::improving:
        break;
      case NeighborhoodAcceptance::annealing:
        accept = accept || value == current_value ||
                 (temperature > 0 && uniform(rng) < std::exp((current_value - value) / temperature));
        temperature *= options.cooling;
        break;
      case NeighborhoodAcceptance::threshold:
        accept = accept || value - current_value <= threshold;
        threshold *= options.cooling;
        break;
      case NeighborhoodAcceptance::record:
        accept = accept || value - result.value <= options.tolerance;
        break;
      case NeighborhoodAcceptance::deluge:
        accept = value <= level;
        level -= options.rain_speed;
        break;
      case NeighborhoodAcceptance::extended_deluge: {
        const double fraction = static_cast<double>(result.evaluations - 1) /
                                static_cast<double>(options.evaluations - 1);
        accept = accept || value <= std::lerp(initial_value, options.target_level, fraction);
        break;
      }
      case NeighborhoodAcceptance::late: {
        auto& past = history[(result.evaluations - 2) % history.size()];
        accept = value <= current_value || value < past;
        past = accept ? value : current_value;
        break;
      }
    }
    if (value < result.value) {
      result.best = candidate;
      result.value = value;
    }
    if (accept) {
      current = std::move(candidate);
      current_value = value;
      ++result.accepted;
    }
  }
  return result;
}

} // namespace datamunge::optim
