#pragma once

/// @file
/// @brief Budgeted path relinking between two complete discrete solutions.

#include <datamunge/optim/variable_neighborhood_search.hpp>

namespace datamunge::optim {

/// @brief Work and result of path relinking toward a guiding solution.
/// @tparam State Type of a feasible solution.
/// @par Example
/// @code
/// const dm::optim::PathRelinkResult<int> result{
///     {1, 1, 1, 0, dm::optim::NeighborhoodStop::budget}, false};
/// @endcode
/// @see path_relink()
template <class State>
struct PathRelinkResult {
  NeighborhoodResult<State> search; ///< Scored incumbent and path work counts.
  bool reached_guide; ///< Whether the selected path ended at distance zero.
};

namespace detail {

/// @brief Link two already constructed states under the caller's evaluation budget.
/// @tparam State Type representing a complete state.
/// @tparam Score Borrowed objective callable.
/// @tparam Distance Discrete distance to the guide.
/// @tparam Enumerate Forward-move generator.
/// @tparam Apply Typed feasible move applicator.
/// @param initial Path endpoint updated after each selected move; already scored.
/// @param guide Guiding state.
/// @param budget Shared incumbent, callback and evaluation cap.
/// @param distance Metric for distance-reducing moves.
/// @param enumerate Ordered move generator.
/// @param apply Feasible move applicator.
/// @param rng Borrowed random engine.
/// @return Whether the guide was reached.
template <class State, class Score, class Distance, class Enumerate, class Apply>
bool relink(State& initial, const State& guide, NeighborhoodBudget<State, Score>& budget,
            Distance& distance, Enumerate& enumerate, Apply& apply, std::mt19937_64& rng) {
  auto remaining = std::invoke(distance, std::as_const(initial), guide);
  while (remaining > 0 && budget.result.evaluations < budget.cap) {
    std::optional<State> chosen;
    auto chosen_distance = remaining;
    double chosen_value = std::numeric_limits<double>::infinity();
    for (const auto& move : std::invoke(enumerate, std::as_const(initial), guide, rng)) {
      if (budget.result.evaluations == budget.cap) break;
      State trial = initial;
      std::invoke(apply, trial, move);
      const auto next_distance = std::invoke(distance, std::as_const(trial), guide);
      if (next_distance >= remaining) continue;
      const auto value = budget.evaluate(trial);
      if (value < chosen_value) {
        chosen.emplace(std::move(trial));
        chosen_distance = next_distance;
        chosen_value = value;
      }
    }
    if (!chosen) break;
    initial = std::move(*chosen);
    remaining = chosen_distance;
    ++budget.result.accepted;
  }
  return remaining == 0;
}

} // namespace detail

/// @brief Greedily relink a feasible solution toward a guide using distance-reducing moves.
///
/// Both endpoints are feasible. The starting state is scored once. Each step
/// scores the supplied moves that strictly decrease the discrete distance to
/// the guide and selects the lowest-loss candidate, even when worse than the
/// current state. Ties follow enumeration order. The guide is scored only
/// when reached by an evaluated move. A zero distance must identify the guide.
/// Callback exceptions propagate. Replay holds within a build for deterministic
/// callbacks.
/// @tparam State Copyable feasible state.
/// @tparam Score Callable returning a finite loss for const State&.
/// @tparam Distance Callable returning a nonnegative integral distance between states.
/// @tparam Enumerate Callable returning a finite ordered range of moves for state, guide and engine.
/// @tparam Apply Callable applying a typed move to a copy of a state.
/// @param initial Starting endpoint, taken by value.
/// @param guide Borrowed guiding endpoint.
/// @param score Objective to minimize.
/// @param distance Nonnegative integer movement metric; zero identifies the guide.
/// @param enumerate Return candidate moves directed toward the guide.
/// @param apply Feasible move applicator.
/// @param evaluations Positive call budget, including the starting endpoint and repeats.
/// @param seed Local engine seed, default 42.
/// @return Best observed state, work counts and whether distance zero was reached.
/// @throws std::invalid_argument Zero budget.
/// @throws std::domain_error Objective result is nonfinite.
/// @par Example
/// @code
/// const auto result = dm::optim::path_relink(
///     3, 0, [](int x) { return x*x; },
///     [](int x, int y) { return std::size_t(std::abs(x-y)); },
///     [](int, int, auto&) { return std::vector{-1}; },
///     [](int& x, int m) { x += m; }, 10);
/// std::print("{}\n", result.reached_guide); // true
/// @endcode
/// @par References
/// - Resende and Ribeiro (2016). Optimization by GRASP.
///   https://doi.org/10.1007/978-1-4939-6530-4
/// @see neighborhood_search()
template <class State, class Score, class Distance, class Enumerate, class Apply>
PathRelinkResult<State> path_relink(State initial, const State& guide, Score&& score,
                                   Distance&& distance, Enumerate&& enumerate, Apply&& apply,
                                   std::size_t evaluations, std::uint64_t seed = 42) {
  if (evaluations == 0) throw std::invalid_argument("Path relinking requires a positive budget");
  detail::NeighborhoodBudget<State, std::remove_reference_t<Score>> budget{
      score, evaluations, {initial, std::numeric_limits<double>::infinity(), 0, 0, NeighborhoodStop::budget}};
  budget.evaluate(initial);
  std::mt19937_64 rng(seed);
  const bool reached_guide = detail::relink(initial, guide, budget, distance, enumerate, apply, rng);
  if (budget.result.evaluations < budget.cap) budget.result.stop = NeighborhoodStop::no_move;
  return {std::move(budget.result), reached_guide};
}

} // namespace datamunge::optim
