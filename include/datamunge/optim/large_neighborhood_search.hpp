#pragma once

/// @file
/// @brief Randomized destroy-and-repair search with optional operator adaptation.

#include <datamunge/optim/neighborhood_search.hpp>

#include <limits>

namespace datamunge::optim {

/// @brief Controls for large neighborhood search and adaptive large neighborhood search.
/// @par Example
/// @code
/// dm::optim::LargeNeighborhoodOptions options;
/// options.destroy_operators = 2;
/// options.reaction = 0.2;
/// @endcode
/// @see large_neighborhood_search()
struct LargeNeighborhoodOptions {
  std::size_t evaluations{1000}; ///< Positive score-call cap including the start.
  std::uint64_t seed{42}; ///< Seed of the local engine.
  std::size_t destroy_operators{1}; ///< Positive number of indexed destroy policies.
  std::size_t repair_operators{1}; ///< Positive number of indexed repair policies.
  double temperature{}; ///< Nonnegative finite acceptance temperature; zero is greedy.
  double cooling{1}; ///< Temperature multiplier per attempted repair, in (0, 1].
  double reaction{}; ///< Smoothing weight of operator rewards in [0, 1]; zero gives LNS.
  std::size_t reaction_interval{32}; ///< Positive adaptation period, measured in attempts.
  std::size_t empty_attempt_limit{32}; ///< Maximum positive streak of unsuccessful constructions.
};

/// @brief Owned incumbent with search work and final operator weights.
/// @tparam State Type representing a complete feasible state.
/// @par Example
/// @code
/// const dm::optim::LargeNeighborhoodResult<int> result{
///     {0, 0.0, 1, 0, dm::optim::NeighborhoodStop::budget}, {1}, {1}, 0};
/// @endcode
/// @see large_neighborhood_search()
template <class State>
struct LargeNeighborhoodResult {
  NeighborhoodResult<State> search; ///< Scored incumbent, call count and termination status.
  std::vector<double> destroy_weights; ///< Final learned destroy-operator weights.
  std::vector<double> repair_weights; ///< Final learned repair-operator weights.
  std::size_t attempts; ///< Number of operator selection and construction attempts.
};

namespace detail {

/// @brief Temporary per-operator rewards accumulated inside an adaptation period.
struct OperatorRewards {
  std::vector<double> total; ///< Reward sum per operator.
  std::vector<std::size_t> uses; ///< Count per operator, including failed constructions.

  /// @brief Initialize statistics for equally weighted operators.
  /// @param count Positive operator count.
  explicit OperatorRewards(std::size_t count) : total(count), uses(count) {}

  /// @brief Smooth active operator weights toward their average reward and clear the period.
  /// @param weights Mutable positive weights.
  /// @param reaction Smoothing factor in [0, 1].
  void update(std::vector<double>& weights, double reaction) {
    for (std::size_t index = 0; index < uses.size(); ++index) {
      if (uses[index] != 0) {
        weights[index] = std::lerp(weights[index], total[index] / uses[index], reaction);
        total[index] = 0;
        uses[index] = 0;
      }
    }
  }
};

} // namespace detail

/// @brief Search feasible states by repeatedly destroying and repairing the incumbent.
///
/// Two distinct roulette selections choose indexed destroy and repair policies.
/// With reaction zero their weights stay uniform. Otherwise a period updates
/// the weights toward mean rewards: 8 for a new best, 4 for another improving
/// acceptance, 2 for a non-improving acceptance, and 1 otherwise. The same
/// reward applies to the pair. Optional empty partials or repairs do not score
/// a state; a run of empty attempts ends the search. A repair that returns a
/// feasible complete state uses one objective call even if duplicated. Use a
/// greedy repair and a single ruin policy for iterated greedy search.
/// Callback exceptions propagate. Seeded deterministic callbacks replay within
/// a build. Callbacks control feasibility and may use inexpensive construction
/// heuristics. Only calls to the supplied score made by the driver are counted.
/// @tparam State Copyable feasible state type.
/// @tparam Score Callable returning a finite loss on const State&.
/// @tparam Destroy Callable returning std::optional<Partial> for state, operator index and engine.
/// @tparam Repair Callable returning std::optional<State> for partial state, operator index and engine.
/// @param initial Feasible initial state, taken by value.
/// @param score Objective to minimize.
/// @param destroy Generator of a partial solution or empty optional.
/// @param repair Constructor of a feasible solution or empty optional, accepting an rvalue Partial.
/// @param options Budget, seed, construction limits and adaptation controls.
/// @return Best feasible state, work counts and final weights.
/// @throws std::invalid_argument Invalid options.
/// @throws std::domain_error Objective returns a nonfinite loss.
/// @throws std::overflow_error Attempt counter exceeds its representable range.
/// @par Example
/// @code
/// const auto result = dm::optim::large_neighborhood_search(
///     4, [](int x) { return x*x; },
///     [](int x, std::size_t, auto&) -> std::optional<int> { return x-1; },
///     [](int x, std::size_t, auto&) -> std::optional<int> { return std::max(0, x); },
///     {8});
/// std::print("{}\n", result.search.best); // 0
/// @endcode
/// @par References
/// - Røpke and Pisinger (2006). https://doi.org/10.1287/trsc.1050.0135
/// @see LargeNeighborhoodOptions, neighborhood_search()
template <class State, class Score, class Destroy, class Repair>
LargeNeighborhoodResult<State> large_neighborhood_search(
    State initial, Score&& score, Destroy&& destroy, Repair&& repair,
    const LargeNeighborhoodOptions& options = {}) {
  if (options.evaluations == 0 || options.destroy_operators == 0 ||
      options.repair_operators == 0 || options.empty_attempt_limit == 0 ||
      options.reaction_interval == 0 ||
      !(std::isfinite(options.temperature) && options.temperature >= 0 &&
        options.cooling > 0 && options.cooling <= 1 &&
        options.reaction >= 0 && options.reaction <= 1)) {
    throw std::invalid_argument("Invalid large neighborhood search controls");
  }
  /// Evaluate the borrowed objective and reject nonfinite output.
  const auto evaluate = [&](const State& x) {
    const double value = std::invoke(score, x);
    if (!std::isfinite(value)) throw std::domain_error("Large neighborhood loss must be finite");
    return value;
  };
  double current_value = evaluate(initial);
  LargeNeighborhoodResult<State> result{
      {initial, current_value, 1, 0, NeighborhoodStop::budget},
      std::vector<double>(options.destroy_operators, 1),
      std::vector<double>(options.repair_operators, 1), 0};
  detail::OperatorRewards destroy_stats(options.destroy_operators), repair_stats(options.repair_operators);
  std::discrete_distribution<std::size_t> choose_destroy(result.destroy_weights.begin(),
                                                        result.destroy_weights.end());
  std::discrete_distribution<std::size_t> choose_repair(result.repair_weights.begin(),
                                                       result.repair_weights.end());
  std::mt19937_64 rng(options.seed);
  std::uniform_real_distribution<double> uniform;
  std::size_t empty = 0, period = 0;
  double temperature = options.temperature;
  while (result.search.evaluations < options.evaluations && empty < options.empty_attempt_limit) {
    if (result.attempts == std::numeric_limits<std::size_t>::max()) {
      throw std::overflow_error("Large neighborhood attempt count exhausted");
    }
    ++result.attempts;
    const auto d = choose_destroy(rng);
    const auto r = choose_repair(rng);
    auto partial = std::invoke(destroy, std::as_const(initial), d, rng);
    std::optional<State> candidate;
    if (partial) candidate = std::invoke(repair, std::move(*partial), r, rng);
    double reward = 1;
    if (candidate) {
      empty = 0;
      const auto value = evaluate(*candidate);
      ++result.search.evaluations;
      const bool improved = value < current_value;
      if (improved || (temperature > 0 &&
                       uniform(rng) < std::exp((current_value - value) / temperature))) {
        initial = std::move(*candidate);
        current_value = value;
        reward = improved ? 4 : 2;
        ++result.search.accepted;
        if (value < result.search.value) {
          result.search.best = initial;
          result.search.value = value;
          reward = 8;
        }
      }
    } else {
      ++empty;
    }
    temperature *= options.cooling;
    if (options.reaction > 0) {
      destroy_stats.total[d] += reward;
      repair_stats.total[r] += reward;
      ++destroy_stats.uses[d];
      ++repair_stats.uses[r];
      if (++period == options.reaction_interval) {
        destroy_stats.update(result.destroy_weights, options.reaction);
        repair_stats.update(result.repair_weights, options.reaction);
        choose_destroy.param({result.destroy_weights.begin(), result.destroy_weights.end()});
        choose_repair.param({result.repair_weights.begin(), result.repair_weights.end()});
        period = 0;
      }
    }
  }
  if (result.search.evaluations < options.evaluations) result.search.stop = NeighborhoodStop::no_move;
  return result;
}

} // namespace datamunge::optim
