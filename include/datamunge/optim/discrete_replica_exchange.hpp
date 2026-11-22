#pragma once

/// @file
/// @brief Parallel tempering over typed discrete states and moves.

#include <datamunge/optim/neighborhood_search.hpp>

namespace datamunge::optim {

/// @brief Evaluation cap, seed and fixed temperature ladder for discrete replica exchange.
/// @par Example
/// @code
/// const dm::optim::DiscreteReplicaOptions options{100, 42, {0.25, 1, 4}};
/// @endcode
/// @see discrete_replica_exchange()
struct DiscreteReplicaOptions {
  std::size_t evaluations{1000}; ///< Positive global cap, including replica initialization.
  std::uint64_t seed{42}; ///< Seed of the local mt19937_64.
  std::vector<double> temperatures{0.25, 1, 4}; ///< Positive, finite, strictly increasing ladder.
};

/// @brief Owned incumbent and replica-exchange work counts.
/// @tparam State Type representing a complete feasible state.
/// @par Example
/// @code
/// const dm::optim::DiscreteReplicaResult<int> result{
///     {0, 0.0, 1, 0, dm::optim::NeighborhoodStop::budget}, 0, 0};
/// @endcode
/// @see discrete_replica_exchange()
template <class State>
struct DiscreteReplicaResult {
  NeighborhoodResult<State> search; ///< Best trial and objective and local-move counts.
  std::size_t exchanges; ///< Attempted replica swaps.
  std::size_t accepted_exchanges; ///< Accepted replica swaps.
};

/// @brief Run Metropolis search at several temperatures and swap adjacent replicas.
///
/// Initialization scores one state per ladder position, starting at the cold end
/// and stopping if that uses the budget. Each sweep proposes once per replica.
/// An empty proposal skips that replica, and a sweep with no proposals stops the
/// search. After completed sweeps, adjacent replica pairs exchange states with
/// the parallel-tempering acceptance probability. For ladders longer than two,
/// the pair partition alternates between consecutive sweeps. A swap reuses
/// cached scores. Duplicate evaluations cost one each. Exceptions from callbacks
/// propagate. Seeded replay holds within a build for deterministic callbacks.
/// @tparam State Copyable complete state type.
/// @tparam Score Callable returning a finite loss for const State&.
/// @tparam Propose Callable returning std::optional<Move> from const State& and std::mt19937_64&.
/// @tparam Apply Callable applying const Move& to State& in place.
/// @param initial Feasible initial states, in order of increasing temperature.
/// @param score Objective to minimize.
/// @param propose Generator of feasible moves with a borrowed local engine.
/// @param apply Move applicator preserving feasibility.
/// @param options Evaluation cap, seed and ladder (default three replicas).
/// @return Best trial, call and acceptance counts, and exchange counts.
/// @throws std::invalid_argument Empty or mismatched states and ladder, invalid temperatures or zero budget.
/// @throws std::domain_error The callback returns a nonfinite loss.
/// @par Example
/// @code
/// const auto result = dm::optim::discrete_replica_exchange(
///     std::vector{1, 0, 2}, [](int x) { return x*x; },
///     [](int, auto&) -> std::optional<int> { return -1; },
///     [](int& x, int move) { x += move; });
/// std::print("{}\n", result.search.best); // 0
/// @endcode
/// @par References
/// - Hukushima and Nemoto (1996), Exchange Monte Carlo Method and Application
///   to Spin Glass Simulations. https://doi.org/10.1143/JPSJ.65.1604
/// @see neighborhood_search(), DiscreteReplicaOptions
template <class State, class Score, class Propose, class Apply>
DiscreteReplicaResult<State> discrete_replica_exchange(
    std::vector<State> initial, Score&& score, Propose&& propose, Apply&& apply,
    const DiscreteReplicaOptions& options = {}) {
  const auto& temperatures = options.temperatures;
  if (initial.size() < 2 || initial.size() != temperatures.size() || options.evaluations == 0) {
    throw std::invalid_argument("Replica exchange requires a matching ladder and positive budget");
  }
  double previous = 0;
  for (const double temperature : temperatures) {
    if (!(std::isfinite(temperature) && temperature > previous)) {
      throw std::invalid_argument("Replica temperatures must be positive and strictly increasing");
    }
    previous = temperature;
  }
  /// Require a finite loss from each evaluation.
  const auto evaluate = [&](const State& state) {
    const double value = std::invoke(score, state);
    if (!std::isfinite(value)) throw std::domain_error("Replica loss must be finite");
    return value;
  };

  DiscreteReplicaResult<State> result{
      {initial.front(), evaluate(initial.front()), 1, 0, NeighborhoodStop::budget}, 0, 0};
  const auto count = std::min(initial.size(), options.evaluations);
  std::vector<double> losses(count);
  losses[0] = result.search.value;
  /// Record an evaluated replica or trial, keeping the earliest best point.
  const auto observe = [&](const State& state, double value) {
    ++result.search.evaluations;
    if (value < result.search.value) {
      result.search.best = state;
      result.search.value = value;
    }
  };
  for (std::size_t i = 1; i < count; ++i) {
    losses[i] = evaluate(initial[i]);
    observe(initial[i], losses[i]);
  }

  std::mt19937_64 rng(options.seed);
  std::uniform_real_distribution<double> uniform;
  std::size_t first_pair = 0;
  while (result.search.evaluations < options.evaluations) {
    const auto before = result.search.evaluations;
    for (std::size_t i = 0; i < count && result.search.evaluations < options.evaluations; ++i) {
      const auto move = std::invoke(propose, std::as_const(initial[i]), rng);
      if (!move) continue;
      State candidate = initial[i];
      std::invoke(apply, candidate, *move);
      const double value = evaluate(candidate);
      observe(candidate, value);
      if (value <= losses[i] || uniform(rng) < std::exp((losses[i] - value) / temperatures[i])) {
        initial[i] = std::move(candidate);
        losses[i] = value;
        ++result.search.accepted;
      }
    }
    if (result.search.evaluations == before) {
      result.search.stop = NeighborhoodStop::no_move;
      break;
    }
    if (result.search.evaluations == options.evaluations) break;
    for (std::size_t i = first_pair; i + 1 < count; i += 2) {
      const auto j = i + 1;
      const long double inverse_gap =
          (1 - static_cast<long double>(temperatures[i]) / temperatures[j]) / temperatures[i];
      ++result.exchanges;
      if (losses[j] <= losses[i] ||
          uniform(rng) < std::exp(inverse_gap * (static_cast<long double>(losses[i]) - losses[j]))) {
        std::swap(initial[i], initial[j]);
        std::swap(losses[i], losses[j]);
        ++result.accepted_exchanges;
      }
    }
    if (count > 2) first_pair = 1 - first_pair;
  }
  return result;
}

} // namespace datamunge::optim
