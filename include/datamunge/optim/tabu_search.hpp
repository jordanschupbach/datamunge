#pragma once

/// @file
/// @brief Budgeted discrete tabu search with aspiration by best observed loss.

#include <datamunge/optim/neighborhood_search.hpp>

#include <deque>
#include <ranges>
#include <tuple>
#include <type_traits>
#include <unordered_map>

namespace datamunge::optim {

/// @brief Work cap, local random seed and tabu tenure.
/// @par Example
/// @code
/// const dm::optim::TabuOptions options{100, 7, 4};
/// @endcode
/// @see tabu_search()
struct TabuOptions {
  std::size_t evaluations{1000}; ///< Positive objective-call cap, including initialization.
  std::uint64_t seed{42}; ///< Seed of the candidate enumerator's local engine.
  std::size_t tenure{5}; ///< Keep the inverse key tabu for this many accepted moves.
};

namespace detail {

/// @brief Run a common tabu scan with a fixed or history-driven tenure strategy.
/// @tparam State Copyable feasible state.
/// @tparam Score Loss callback.
/// @tparam Enumerate Ordered neighborhood generator.
/// @tparam Apply Typed move applicator.
/// @tparam Keys Callback for forward and inverse tabu keys.
/// @tparam React Callback returning current tenure and an escape flag.
/// @param initial Feasible start, taken by value.
/// @param score Objective to minimize.
/// @param enumerate Ordered move generator.
/// @param apply Feasible move applicator.
/// @param keys Forward and inverse identifiers.
/// @param react Policy called once before each scan.
/// @param history_limit Maximum retained inverse keys.
/// @param options Budget and seed.
/// @return Global incumbent and objective-call counts.
template <class State, class Score, class Enumerate, class Apply, class Keys, class React>
NeighborhoodResult<State> tabu_loop(State initial, Score& score, Enumerate& enumerate,
                                    Apply& apply, Keys& keys, React& react, std::size_t history_limit,
                                    const TabuOptions& options) {
  if (options.evaluations == 0) throw std::invalid_argument("Tabu search budget must be positive");
  using Moves = std::invoke_result_t<Enumerate&, const State&, std::mt19937_64&>; ///< Enumerated range.
  using Move = std::ranges::range_value_t<Moves>; ///< Typed neighborhood move.
  using Key = std::tuple_element_t<1, std::invoke_result_t<Keys&, const State&, const Move&>>; ///< Tabu identifier.
  /// Evaluate a feasible state or propagate a numerical error.
  const auto evaluate = [&](const State& state) {
    const double loss = std::invoke(score, state);
    if (!std::isfinite(loss)) throw std::domain_error("Tabu search loss must be finite");
    return loss;
  };

  struct Trial {
    State state; ///< Proposed feasible state.
    double value; ///< Objective evaluated at state.
    Key inverse; ///< Key to remember if the trial is selected.
  };

  NeighborhoodResult<State> result{initial, evaluate(initial), 1, 0, NeighborhoodStop::budget};
  State current = std::move(initial);
  std::deque<Key> tabu;
  std::mt19937_64 rng(options.seed);
  while (result.evaluations < options.evaluations) {
    const auto [tenure, escape] = std::invoke(react, std::as_const(current), result.accepted);
    const auto timed_out = std::max(tabu.size(), tenure) - tenure;
    const auto first = tabu.begin() + static_cast<typename std::deque<Key>::difference_type>(timed_out);
    const auto record = result.value;
    std::size_t eligible = 0;
    std::optional<Trial> selected;
    for (const auto& move : std::invoke(enumerate, std::as_const(current), rng)) {
      if (result.evaluations == options.evaluations) break;
      auto [key, inverse] = std::invoke(keys, std::as_const(current), move);
      State candidate = current;
      std::invoke(apply, candidate, move);
      const auto value = evaluate(candidate);
      ++result.evaluations;
      if (value < result.value) {
        result.best = candidate;
        result.value = value;
      }
      if (!escape && value >= record && std::find(first, tabu.end(), key) != tabu.end()) continue;
      ++eligible;
      if (!selected || (escape
              ? std::uniform_int_distribution<std::size_t>(1, eligible)(rng) == 1
              : value < selected->value)) {
        selected.emplace(Trial{std::move(candidate), value, std::move(inverse)});
      }
    }
    if (!selected) {
      if (result.evaluations < options.evaluations) result.stop = NeighborhoodStop::no_move;
      break;
    }
    current = std::move(selected->state);
    if (history_limit != 0) {
      if (tabu.size() == history_limit) tabu.pop_front();
      tabu.push_back(std::move(selected->inverse));
    }
    ++result.accepted;
  }
  return result;
}

} // namespace detail

/// @brief Search typed neighborhoods with short-term inverse-move memory.
///
/// Scores the start followed by each enumerated neighbor, including tabu and
/// duplicate moves, through the budget limit. Accepts the lowest-loss admissible
/// candidate in callback order, even if it worsens the current state.
/// A tabu move is admissible if it strictly improves the best observed loss.
/// If the budget ends inside a scan, the selected move uses the scanned prefix.
/// Search stops without an admissible move. Callbacks define feasibility and
/// propagate any exceptions. Replay holds within a build for deterministic callbacks.
/// @tparam State Copyable type representing a complete feasible state.
/// @tparam Score Callable returning a finite loss for const State&.
/// @tparam Enumerate Callable returning a finite range of moves for const State& and std::mt19937_64&.
/// @tparam Apply Callable applying const Move& to State&.
/// @tparam Keys Callable returning a pair of forward and inverse keys, comparable by equality.
/// @param initial Feasible starting state, taken by value.
/// @param score Objective to minimize.
/// @param enumerate Ordered neighborhood generator with a borrowed local engine.
/// @param apply Move applicator responsible for preserving feasibility.
/// @param keys Map a state and a feasible move to its forward and inverse tabu keys.
/// @param options Budget, seed and tenure; zero tenure disables tabu memory.
/// @return Best observed state and observed work counts.
/// @throws std::invalid_argument Zero evaluation budget.
/// @throws std::domain_error Callback returns a nonfinite loss.
/// @par Example
/// @code
/// const auto result = dm::optim::tabu_search(
///     3, [](int x) { return x*x; },
///     [](int x, auto&) { return x > 0 ? std::vector{-1, 1} : std::vector<int>{}; },
///     [](int& x, int move) { x += move; },
///     [](int, int move) { return std::pair{move, -move}; });
/// std::print("{}\n", result.best); // 0
/// @endcode
/// @par References
/// - Glover (1989). Tabu Search, Part I. https://doi.org/10.1287/ijoc.1.3.190
/// @see neighborhood_search(), TabuOptions
template <class State, class Score, class Enumerate, class Apply, class Keys>
NeighborhoodResult<State> tabu_search(State initial, Score&& score, Enumerate&& enumerate,
                                      Apply&& apply, Keys&& keys, const TabuOptions& options = {}) {
  /// Keep a fixed tabu tenure with greedy admissible selection.
  const auto fixed = [&](const State&, std::size_t) { return std::pair{options.tenure, false}; };
  return detail::tabu_loop(std::move(initial), score, enumerate, apply, keys, fixed, options.tenure, options);
}

/// @brief Reactive tabu tenure bounds and recurrence settings.
/// @par Example
/// @code
/// dm::optim::ReactiveTabuOptions options;
/// options.maximum_tenure = 16;
/// @endcode
/// @see reactive_tabu_search()
struct ReactiveTabuOptions {
  TabuOptions tabu; ///< Objective budget, seed and initial tenure.
  std::size_t minimum_tenure{1}; ///< Lower bound on learned tabu tenure.
  std::size_t maximum_tenure{20}; ///< Upper bound and maximum tracked move history.
  std::size_t quiet_period{10}; ///< Accepted moves without a revisit before decreasing tenure.
  std::size_t repeats_before_escape{3}; ///< Positive revisit count at one state before diversification.
};

/// @brief Run reactive tabu search using explicit state recurrence and random diversification.
///
/// Each repeated state increases tenure by at least one, up to the maximum.
/// Each quiet period decreases tenure by one, down to the minimum. Excessive
/// repeats trigger a random walk whose length is the rounded moving average
/// of observed cycle lengths. During escape, scanned candidates are scored
/// normally and one is selected uniformly from the available prefix.
/// The historical best includes all scored trials. Move histories are bounded
/// by maximum tenure; state keys occupy O(number of visited states).
/// See tabu_search() for aspiration, budget and callback semantics.
/// @tparam State Copyable complete state type.
/// @tparam Score Finite-return objective callable.
/// @tparam Enumerate Ordered neighborhood generator.
/// @tparam Apply Feasible move applicator.
/// @tparam Keys Callable returning forward and inverse tabu keys.
/// @tparam StateKey Callable returning a hashable, equality-comparable state identifier.
/// @param initial Feasible start, taken by value.
/// @param score Objective to minimize.
/// @param enumerate Ordered move generator.
/// @param apply Feasible move applicator.
/// @param keys Map each move to a forward and inverse key.
/// @param state_key Stable identifier with equal values for states considered identical.
/// @param options Caps, recurrence parameters and initial tenure.
/// @return Best scored state and observed work counts.
/// @throws std::invalid_argument Invalid budget or reactive controls.
/// @throws std::domain_error Objective returns a nonfinite loss.
/// @par Example
/// @code
/// const auto result = dm::optim::reactive_tabu_search(
///     3, [](int x) { return x*x; },
///     [](int x, auto&) { return x > 0 ? std::vector{-1, 1} : std::vector<int>{}; },
///     [](int& x, int move) { x += move; },
///     [](int, int move) { return std::pair{move, -move}; },
///     [](int x) { return x; });
/// std::print("{}\n", result.best); // 0
/// @endcode
/// @par References
/// - Battiti and Tecchiolli (1994). https://doi.org/10.1287/ijoc.6.2.126
/// @see tabu_search()
template <class State, class Score, class Enumerate, class Apply, class Keys, class StateKey>
NeighborhoodResult<State> reactive_tabu_search(
    State initial, Score&& score, Enumerate&& enumerate, Apply&& apply, Keys&& keys, StateKey&& state_key,
    const ReactiveTabuOptions& options = {}) {
  if (options.minimum_tenure > options.tabu.tenure || options.tabu.tenure > options.maximum_tenure ||
      options.quiet_period == 0 || options.repeats_before_escape == 0) {
    throw std::invalid_argument("Invalid reactive tabu controls");
  }
  using Identifier = std::remove_cvref_t<std::invoke_result_t<StateKey&, const State&>>; ///< State identity.
  struct Visit {
    std::size_t last; ///< Previous accepted-move index.
    std::size_t repeats; ///< Repetitions since the last escape at this state.
  };
  std::unordered_map<Identifier, Visit> visits;
  std::size_t tenure = options.tabu.tenure, last_increase = 0, walk = 0;
  long double cycle_length = 1;
  /// Observe recurrence, learn a tabu tenure, and request an escape if needed.
  auto react = [&](const State& current, std::size_t turn) {
    const auto [iterator, inserted] = visits.try_emplace(std::invoke(state_key, current), Visit{turn, 0});
    auto& visit = iterator->second;
    if (!inserted) {
      cycle_length = std::lerp(cycle_length, static_cast<long double>(turn - visit.last), 0.1L);
      visit.last = turn;
      if (tenure < options.maximum_tenure) tenure += std::min(options.maximum_tenure - tenure,
                                                             std::max(std::size_t{1}, tenure / 10));
      last_increase = turn;
      if (++visit.repeats >= options.repeats_before_escape) {
        const auto length = std::min(std::round(cycle_length), static_cast<long double>(options.tabu.evaluations));
        walk = std::max(walk, static_cast<std::size_t>(length));
        visit.repeats = 0;
      }
    } else if (turn - last_increase >= options.quiet_period && tenure > options.minimum_tenure) {
      --tenure;
      last_increase = turn;
    }
    const bool escaping = walk != 0;
    if (escaping) --walk;
    return std::pair{tenure, escaping};
  };
  return detail::tabu_loop(std::move(initial), score, enumerate, apply, keys, react,
                           options.maximum_tenure, options.tabu);
}

} // namespace datamunge::optim
