#pragma once

/// @file
/// @brief Budgeted variable-neighborhood descent and iterated discrete local search.

#include <datamunge/optim/neighborhood_search.hpp>

#include <limits>
#include <numeric>
#include <type_traits>

namespace datamunge::optim {

/// @brief Choose descent, local search after shaking, or shaking alone.
/// @par Example
/// @code
/// const auto method = dm::optim::VariableNeighborhoodMethod::reduced;
/// @endcode
/// @see variable_neighborhood_search()
enum class VariableNeighborhoodMethod {
  descent,  ///< Deterministic first-improvement scans unless the callbacks or order are randomized.
  basic,    ///< Shake in a neighborhood, then descend across the local neighborhoods.
  reduced   ///< Evaluate the shake directly, without local descent.
};

/// @brief Neighborhood count, order and evaluation budget.
/// @par Example
/// @code
/// dm::optim::VariableNeighborhoodOptions options{128, 7, 3, true};
/// @endcode
/// @see variable_neighborhood_search(), iterated_local_search()
struct VariableNeighborhoodOptions {
  std::size_t evaluations{1000}; ///< Positive objective-call cap, including initialization.
  std::uint64_t seed{42}; ///< Seed of the local engine.
  std::size_t neighborhoods{1}; ///< Positive number of indexed neighborhoods.
  bool shuffle_order{false}; ///< Shuffle descent order once per descent.
};

namespace detail {

/// @brief Score and retain incumbents under one shared budget.
/// @tparam State Copyable feasible state.
/// @tparam Score Callable accepting a const state reference.
template <class State, class Score>
struct NeighborhoodBudget {
  Score& score; ///< Borrowed objective.
  std::size_t cap; ///< Maximum evaluations.
  NeighborhoodResult<State> result; ///< Owned incumbent and observed work.

  /// @brief Invoke the objective, count the evaluation and update the incumbent.
  /// @param point Feasible state and caller-checked remaining work.
  /// @return Finite objective value.
  /// @throws std::domain_error Objective result is nonfinite.
  double evaluate(const State& point) {
    const double value = std::invoke(score, point);
    if (!std::isfinite(value)) throw std::domain_error("Neighborhood loss must be finite");
    ++result.evaluations;
    if (value < result.value) {
      result.best = point;
      result.value = value;
    }
    return value;
  }
};

/// @brief Exhaust first-improvement scans, restarting at the first neighborhood after improvement.
/// @tparam State Copyable feasible state.
/// @tparam Score Objective callable.
/// @tparam Enumerate Callback yielding an ordered finite range of typed neighbors.
/// @tparam Apply Callback applying a move to a state.
/// @param current Starting state updated to the locally improved state.
/// @param current_value Score of current, updated in place.
/// @param budget Shared evaluator and budget.
/// @param enumerate Neighborhood enumerator.
/// @param apply Feasible move applicator.
/// @param order Neighborhood indices, optionally shuffled before descent.
/// @param rng Local random engine forwarded to the enumerator.
template <class State, class Score, class Enumerate, class Apply>
void descend(State& current, double& current_value, NeighborhoodBudget<State, Score>& budget,
             Enumerate& enumerate, Apply& apply, const std::vector<std::size_t>& order,
             std::mt19937_64& rng) {
  std::size_t k = 0;
  while (k < order.size() && budget.result.evaluations < budget.cap) {
    bool improved = false;
    for (const auto& move : std::invoke(enumerate, std::as_const(current), order[k], rng)) {
      if (budget.result.evaluations == budget.cap) break;
      State candidate = current;
      std::invoke(apply, candidate, move);
      const auto value = budget.evaluate(candidate);
      if (value < current_value) {
        current = std::move(candidate);
        current_value = value;
        ++budget.result.accepted;
        improved = true;
        break;
      }
    }
    k = improved ? 0 : k + 1;
  }
}

/// @brief Construct a fresh order for one local descent.
/// @param order Output containing one slot per neighborhood.
/// @param options Neighborhood order controls.
/// @param rng Borrowed local engine used only for randomized order.
inline void descent_order(std::vector<std::size_t>& order,
                          const VariableNeighborhoodOptions& options, std::mt19937_64& rng) {
  std::iota(order.begin(), order.end(), std::size_t{0});
  if (options.shuffle_order) std::shuffle(order.begin(), order.end(), rng);
}

} // namespace detail

/// @brief Run variable-neighborhood descent, basic VNS or reduced VNS.
///
/// Descent scans ordered neighborhoods until none improves the current state.
/// Basic VNS descends after each shake; reduced VNS scores only the shake.
/// Both shaking modes return to the first neighborhood on strict improvement
/// and start another pass after scanning the last index. If a full pass yields
/// no shake, the search stops. Every score call, including initialization,
/// rejected and duplicate trials, counts against one shared cap. The accepted
/// counter includes local descent steps and accepted outer moves.
/// Feasibility is the caller's responsibility and callback exceptions propagate.
/// Replay holds within a build for deterministic callbacks.
/// @tparam State Copyable complete state type.
/// @tparam Score Callable returning finite loss for const State&.
/// @tparam Enumerate Callable returning an ordered finite range of moves for state, index and engine.
/// @tparam Shake Callable returning an optional move for state, index and engine.
/// @tparam Apply Callable applying either move type to State&.
/// @param initial Feasible starting state, taken by value.
/// @param score Objective to minimize.
/// @param enumerate Local move range for the supplied neighborhood index.
/// @param shake Generator of a shaking move, unused in descent mode.
/// @param apply Move applicator responsible for preserving feasibility.
/// @param method Descent, basic or reduced mode.
/// @param options Budget, seed and neighborhood count and order.
/// @return Best state and work counts, with no_move for completed descent or no available shakes.
/// @throws std::invalid_argument Invalid mode, budget or neighborhood count.
/// @throws std::domain_error Objective returns a nonfinite value.
/// @par Example
/// @code
/// const auto result = dm::optim::variable_neighborhood_search(
///     4, [](int x) { return x*x; },
///     [](int x, std::size_t, auto&) { return x > 0 ? std::vector{-1} : std::vector<int>{}; },
///     [](int, std::size_t, auto&) -> std::optional<int> { return -1; },
///     [](int& x, int m) { x += m; },
///     dm::optim::VariableNeighborhoodMethod::descent);
/// std::print("{}\n", result.best); // 0
/// @endcode
/// @par References
/// - Hansen and Mladenović (2001). https://doi.org/10.1016/S0377-2217(00)00100-4
/// @see iterated_local_search(), neighborhood_search()
template <class State, class Score, class Enumerate, class Shake, class Apply>
NeighborhoodResult<State> variable_neighborhood_search(
    State initial, Score&& score, Enumerate&& enumerate, Shake&& shake, Apply&& apply,
    VariableNeighborhoodMethod method, const VariableNeighborhoodOptions& options = {}) {
  if (options.evaluations == 0 || options.neighborhoods == 0 ||
      (method != VariableNeighborhoodMethod::descent && method != VariableNeighborhoodMethod::basic &&
       method != VariableNeighborhoodMethod::reduced)) {
    throw std::invalid_argument("Invalid variable-neighborhood search controls");
  }
  detail::NeighborhoodBudget<State, std::remove_reference_t<Score>> budget{
      score, options.evaluations,
      {initial, std::numeric_limits<double>::infinity(), 0, 0, NeighborhoodStop::budget}};
  auto current_value = budget.evaluate(initial);
  std::mt19937_64 rng(options.seed);
  std::vector<std::size_t> order;
  if (method != VariableNeighborhoodMethod::reduced) order.resize(options.neighborhoods);

  if (method == VariableNeighborhoodMethod::descent) {
    detail::descent_order(order, options, rng);
    detail::descend(initial, current_value, budget, enumerate, apply, order, rng);
  } else {
    std::size_t k = 0;
    std::size_t consecutive_empty = 0;
    while (budget.result.evaluations < budget.cap && consecutive_empty < options.neighborhoods) {
      const auto move = std::invoke(shake, std::as_const(initial), k, rng);
      k = (k + 1) % options.neighborhoods;
      if (!move) {
        ++consecutive_empty;
        continue;
      }
      consecutive_empty = 0;
      State candidate = initial;
      std::invoke(apply, candidate, *move);
      auto value = budget.evaluate(candidate);
      if (method == VariableNeighborhoodMethod::basic && budget.result.evaluations < budget.cap) {
        detail::descent_order(order, options, rng);
        detail::descend(candidate, value, budget, enumerate, apply, order, rng);
      }
      if (value < current_value) {
        initial = std::move(candidate);
        current_value = value;
        ++budget.result.accepted;
        k = 0;
      }
    }
  }
  if (budget.result.evaluations < budget.cap) budget.result.stop = NeighborhoodStop::no_move;
  return std::move(budget.result);
}

/// @brief Alternate perturbation with first-improvement local descent.
///
/// Descends from the initial state, then repeatedly perturbs the current
/// local optimum and descends again. The outer search accepts strictly better
/// local optima. An empty perturbation ends the search. Inner and outer
/// score calls count against a shared cap and update one incumbent.
/// The accepted count includes inner improvement steps and outer improvements.
/// Callbacks determine feasibility. Their exceptions propagate, and
/// deterministic callbacks replay within the same build for a fixed seed.
/// @tparam State Copyable feasible state type.
/// @tparam Score Callable returning finite loss for const State&.
/// @tparam Enumerate Callable yielding an ordered finite range of moves for state, index and engine.
/// @tparam Perturb Callable returning an optional move for state and engine.
/// @tparam Apply Callable applying either move type to State&.
/// @param initial Feasible starting state, taken by value.
/// @param score Objective to minimize.
/// @param enumerate Local neighborhood generator.
/// @param perturb Generator of a feasible perturbation.
/// @param apply Move applicator responsible for preserving feasibility.
/// @param options Evaluation budget, seed and local descent order.
/// @return Best state and observed work counts.
/// @throws std::invalid_argument Zero budget or neighborhood count.
/// @throws std::domain_error Objective returns a nonfinite value.
/// @par Example
/// @code
/// const auto result = dm::optim::iterated_local_search(
///     4, [](int x) { return x*x; },
///     [](int x, std::size_t, auto&) { return x > 0 ? std::vector{-1} : std::vector<int>{}; },
///     [](int, auto&) -> std::optional<int> { return {}; },
///     [](int& x, int m) { x += m; });
/// std::print("{}\n", result.best); // 0
/// @endcode
/// @par References
/// - Lourenço, Martin and Stützle (2019), Iterated Local Search.
///   https://doi.org/10.1007/978-3-319-91086-4_5
/// @see variable_neighborhood_search(), neighborhood_search()
template <class State, class Score, class Enumerate, class Perturb, class Apply>
NeighborhoodResult<State> iterated_local_search(
    State initial, Score&& score, Enumerate&& enumerate, Perturb&& perturb, Apply&& apply,
    const VariableNeighborhoodOptions& options = {}) {
  if (options.evaluations == 0 || options.neighborhoods == 0) {
    throw std::invalid_argument("Local search budget and neighborhood count must be positive");
  }
  detail::NeighborhoodBudget<State, std::remove_reference_t<Score>> budget{
      score, options.evaluations,
      {initial, std::numeric_limits<double>::infinity(), 0, 0, NeighborhoodStop::budget}};
  auto current_value = budget.evaluate(initial);
  std::mt19937_64 rng(options.seed);
  std::vector<std::size_t> order(options.neighborhoods);
  detail::descent_order(order, options, rng);
  detail::descend(initial, current_value, budget, enumerate, apply, order, rng);
  while (budget.result.evaluations < budget.cap) {
    const auto move = std::invoke(perturb, std::as_const(initial), rng);
    if (!move) {
      budget.result.stop = NeighborhoodStop::no_move;
      break;
    }
    State candidate = initial;
    std::invoke(apply, candidate, *move);
    auto value = budget.evaluate(candidate);
    if (budget.result.evaluations < budget.cap) {
      detail::descent_order(order, options, rng);
      detail::descend(candidate, value, budget, enumerate, apply, order, rng);
    }
    if (value < current_value) {
      initial = std::move(candidate);
      current_value = value;
      ++budget.result.accepted;
    }
  }
  return std::move(budget.result);
}

} // namespace datamunge::optim
