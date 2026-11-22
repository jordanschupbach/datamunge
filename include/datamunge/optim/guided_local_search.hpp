#pragma once

/// @file
/// @brief Budgeted guided local search with feature penalties.

#include <datamunge/optim/variable_neighborhood_search.hpp>

namespace datamunge::optim {

/// @brief Objective-call budget, seed and feature penalty strength.
/// @par Example
/// @code
/// const dm::optim::GuidedLocalOptions options{100, 7, 1.5};
/// @endcode
/// @see guided_local_search()
struct GuidedLocalOptions {
  std::size_t evaluations{1000}; ///< Positive call cap, including the start.
  std::uint64_t seed{42}; ///< Seed of the local engine provided to the move generator.
  double penalty_strength{1}; ///< Finite positive multiplier of the sum of active penalties.
};

/// @brief Best original objective value and final feature penalties.
/// @tparam State Copyable complete feasible state.
/// @par Example
/// @code
/// const dm::optim::GuidedLocalResult<int> result{{0, 0, 1, 0, dm::optim::NeighborhoodStop::budget}, {0}};
/// @endcode
/// @see guided_local_search()
template <class State>
struct GuidedLocalResult {
  NeighborhoodResult<State> search; ///< Best state under the original objective and work counts.
  std::vector<std::size_t> penalties; ///< Integer penalty count for each feature.
};

/// @brief Escape discrete local optima using penalties on costly active features.
///
/// Scan for first improvement in the augmented loss f + strength × sum of
/// active penalties. After a scan without improvement, increment the penalty
/// of every active feature tied for largest cost divided by one plus penalty.
/// Then rescan the same neighborhood. Empty neighborhoods or absent positive
/// utilities end the search. Each original objective callback counts once,
/// including repeated states. The returned incumbent minimizes the original
/// scored values. Callbacks own feasibility and propagate exceptions. Seeded
/// replay holds within a build for deterministic callbacks.
/// @tparam State Copyable complete state.
/// @tparam Score Callable returning finite loss for const State&.
/// @tparam Features Callable returning vector of unique zero-based feature indices for const State&.
/// @tparam Enumerate Callable returning finite ordered moves for state and engine.
/// @tparam Apply Callable applying a feasible typed move.
/// @param initial Feasible starting state, taken by value.
/// @param score Original loss to minimize.
/// @param costs Feature costs, each finite and nonnegative.
/// @param features Active features for a given state.
/// @param enumerate Ordered neighborhood generator.
/// @param apply Feasible move applicator.
/// @param options Budget, seed and penalty strength.
/// @return Incumbent and counts, plus final feature penalties.
/// @throws std::invalid_argument Invalid controls or duplicated/out-of-range feature indices.
/// @throws std::domain_error Objective returns a nonfinite value.
/// @par Example
/// @code
/// const auto result = dm::optim::guided_local_search(
///     1, [](int x) { return x*x; }, {1},
///     [](int x) { return x ? std::vector<std::size_t>{0} : std::vector<std::size_t>{}; },
///     [](int x, auto&) { return x ? std::vector{-1} : std::vector<int>{}; },
///     [](int& x, int m) { x += m; });
/// std::print("{}\n", result.search.best); // 0
/// @endcode
/// @par References
/// - Voudouris and Tsang (1999). https://doi.org/10.1016/S0377-2217(98)00099-X
/// @see neighborhood_search(), variable_neighborhood_search()
template <class State, class Score, class Features, class Enumerate, class Apply>
GuidedLocalResult<State> guided_local_search(
    State initial, Score&& score, const std::vector<double>& costs, Features&& features,
    Enumerate&& enumerate, Apply&& apply, const GuidedLocalOptions& options = {}) {
  if (options.evaluations == 0 || costs.empty() ||
      !(std::isfinite(options.penalty_strength) && options.penalty_strength > 0) ||
      !std::ranges::all_of(costs, [](double value) { return std::isfinite(value) && value >= 0; })) {
    throw std::invalid_argument("Invalid guided local search controls or feature costs");
  }
  /// Validate active features from one feasible state.
  const auto active = [&](const State& point) {
    auto indices = std::invoke(features, point);
    std::ranges::sort(indices);
    if ((!indices.empty() && indices.back() >= costs.size()) ||
        std::adjacent_find(indices.begin(), indices.end()) != indices.end()) {
      throw std::invalid_argument("Active features must contain unique valid indices");
    }
    return indices;
  };
  std::vector<std::size_t> penalties(costs.size());
  /// Compute the augmented loss for one set of active features.
  const auto augmented = [&](double value, const std::vector<std::size_t>& indices) {
    long double sum = 0;
    for (const auto index : indices) sum += penalties[index];
    return value + static_cast<long double>(options.penalty_strength) * sum;
  };
  /// Feature utility after its current penalty.
  const auto utility = [&](std::size_t index) {
    return costs[index] / (1 + static_cast<long double>(penalties[index]));
  };

  detail::NeighborhoodBudget<State, std::remove_reference_t<Score>> budget{
      score, options.evaluations,
      {initial, std::numeric_limits<double>::infinity(), 0, 0, NeighborhoodStop::budget}};
  auto features_here = active(initial);
  auto value = budget.evaluate(initial);
  std::mt19937_64 rng(options.seed);
  while (budget.result.evaluations < budget.cap) {
    const auto old_augmented = augmented(value, features_here);
    const auto before = budget.result.evaluations;
    bool accepted = false;
    for (const auto& move : std::invoke(enumerate, std::as_const(initial), rng)) {
      if (budget.result.evaluations == budget.cap) break;
      State trial = initial;
      std::invoke(apply, trial, move);
      auto trial_features = active(trial);
      const auto trial_value = budget.evaluate(trial);
      if (augmented(trial_value, trial_features) < old_augmented) {
        initial = std::move(trial);
        features_here = std::move(trial_features);
        value = trial_value;
        ++budget.result.accepted;
        accepted = true;
        break;
      }
    }
    if (accepted) continue;
    if (before == budget.result.evaluations) break;
    long double maximum = 0;
    for (const auto index : features_here) maximum = std::max(maximum, utility(index));
    if (maximum == 0) break;
    for (const auto index : features_here) {
      if (utility(index) == maximum) ++penalties[index];
    }
  }
  if (budget.result.evaluations < budget.cap) budget.result.stop = NeighborhoodStop::no_move;
  return {std::move(budget.result), std::move(penalties)};
}

} // namespace datamunge::optim
