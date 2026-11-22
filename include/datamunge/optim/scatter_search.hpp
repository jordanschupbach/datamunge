#pragma once

/// @file
/// @brief Stochastic diversification and quality-diversity reference-set relinking.

#include <datamunge/optim/path_relink.hpp>

namespace datamunge::optim {

/// @brief Reference-set sizes and work controls.
/// @par Example
/// @code
/// const dm::optim::ScatterSearchOptions options{200, 7, 8, 4, 6};
/// @endcode
/// @see scatter_search()
struct ScatterSearchOptions {
  std::size_t evaluations{1000}; ///< Positive cap counting diversification and relinking.
  std::uint64_t seed{42}; ///< Seed of the local engine.
  std::size_t reference_size{8}; ///< Positive total capacity of the reference set.
  std::size_t quality_size{4}; ///< Number of slots selected by loss, at least one.
  std::size_t diversification_batch{12}; ///< Positive random candidates generated per reference update.
};

namespace detail {

/// @brief Scored reference candidate.
/// @tparam State Copyable complete state type.
template <class State>
struct ScatterMember {
  State point; ///< Feasible complete solution.
  double value; ///< Observed original loss.
};

/// @brief Retain a quality subset followed by farthest-first additions.
/// @tparam State Complete discrete state type.
/// @tparam Distance Nonnegative distance callback.
/// @param pool Candidates owned by the current iteration.
/// @param distance Borrowed comparison metric; zero means two states are equivalent.
/// @param capacity Maximum retained count.
/// @param quality Number of places reserved for the lowest losses.
/// @return Updated quality and diversity reference set.
template <class State, class Distance>
std::vector<ScatterMember<State>> choose_references(std::vector<ScatterMember<State>> pool, Distance& distance,
                                                    std::size_t capacity, std::size_t quality) {
  std::stable_sort(pool.begin(), pool.end(), [](const auto& a, const auto& b) { return a.value < b.value; });
  std::vector<ScatterMember<State>> selected;
  std::vector<bool> available(pool.size(), true);
  while (selected.size() < capacity) {
    auto next = pool.size();
    std::size_t max_separation = 0;
    for (std::size_t i = 0; i < pool.size(); ++i) {
      if (!available[i]) continue;
      auto separation = std::numeric_limits<std::size_t>::max();
      for (const auto& member : selected) {
        separation = std::min(separation, static_cast<std::size_t>(std::invoke(
            distance, std::as_const(pool[i].point), std::as_const(member.point))));
      }
      if (separation == 0) continue;
      if (next == pool.size() || separation > max_separation) {
        next = i;
        max_separation = separation;
      }
      if (selected.size() < quality) break;
    }
    if (next == pool.size()) break;
    selected.push_back(pool[next]);
    available[next] = false;
  }
  return selected;
}

} // namespace detail

/// @brief Combine quality and diversity references with forward relinking.
///
/// Each round generates feasible random candidates and refreshes the reference
/// set. Quality slots keep the smallest losses. Remaining slots maximize
/// minimum distance to the references already selected. Distinct reference
/// pairs then relink from the worse member toward the better one. The best
/// trial on each path competes for the next set. Duplicate states may be
/// evaluated and count against the common cap. If an entire round consumes
/// no evaluations, the search stops. The accepted count records path steps.
/// Callbacks are responsible for feasibility and propagate exceptions.
/// Deterministic callbacks replay within a build using the same seed.
/// @tparam State Copyable complete feasible state.
/// @tparam Score Callable returning finite loss for const State&.
/// @tparam Diversify Callable returning optional<State> from a local engine.
/// @tparam Distance Callable returning a nonnegative distance representable as size_t.
/// @tparam Enumerate Callable returning ordered forward moves for state, guide and engine.
/// @tparam Apply Callable applying a feasible move in place.
/// @param initial Feasible starting solution.
/// @param score Objective to minimize.
/// @param diversify Stochastic generator of new feasible solutions.
/// @param distance Integer distance, zero meaning equivalent members.
/// @param enumerate Forward move generator.
/// @param apply Feasible move applicator.
/// @param options Global evaluation cap, seed and reference-set sizes.
/// @return Best scored state and work counts.
/// @throws std::invalid_argument Invalid budget or reference-set sizes.
/// @throws std::domain_error Objective result is nonfinite.
/// @par Example
/// @code
/// const auto result = dm::optim::scatter_search(
///     2, [](int x) { return x*x; },
///     [](auto&) -> std::optional<int> { return 0; },
///     [](int x, int y) { return std::size_t(std::abs(x-y)); },
///     [](int, int, auto&) { return std::vector{-1}; },
///     [](int& x, int move) { x += move; }, {20});
/// std::print("{}\n", result.best); // 0
/// @endcode
/// @par References
/// - Glover, Laguna and Martí (2005), Principles of scatter search.
///   https://www.sciencedirect.com/science/article/pii/S0377221704005430
/// @see path_relink(), grasp_path_relink()
template <class State, class Score, class Diversify, class Distance, class Enumerate, class Apply>
NeighborhoodResult<State> scatter_search(State initial, Score&& score, Diversify&& diversify,
    Distance&& distance, Enumerate&& enumerate, Apply&& apply, const ScatterSearchOptions& options = {}) {
  if (options.evaluations == 0 || options.reference_size == 0 || options.quality_size == 0 ||
      options.quality_size > options.reference_size || options.diversification_batch == 0) {
    throw std::invalid_argument("Invalid scatter search budget or reference-set controls");
  }

  using Member = detail::ScatterMember<State>; ///< Reference-set candidate type.
  std::optional<Member> path_best;
  bool capture = false;
  /// Observe any path-local incumbent while using the original objective.
  auto tracking_score = [&](const State& point) {
    const double value = std::invoke(score, point);
    if (capture && std::isfinite(value) && (!path_best || value < path_best->value)) {
      path_best.emplace(Member{point, value});
    }
    return value;
  };
  detail::NeighborhoodBudget<State, decltype(tracking_score)> budget{
      tracking_score, options.evaluations,
      {initial, std::numeric_limits<double>::infinity(), 0, 0, NeighborhoodStop::budget}};
  const auto first = budget.evaluate(initial);
  std::vector<Member> pool{{std::move(initial), first}};
  std::mt19937_64 rng(options.seed);
  while (budget.result.evaluations < budget.cap) {
    const auto before = budget.result.evaluations;
    for (std::size_t i = 0; i < options.diversification_batch && budget.result.evaluations < budget.cap; ++i) {
      auto point = std::invoke(diversify, rng);
      if (!point) break;
      const double value = budget.evaluate(*point);
      pool.push_back({std::move(*point), value});
    }
    const auto refs = detail::choose_references(std::move(pool), distance, options.reference_size,
                                                options.quality_size);
    pool = refs;
    capture = true;
    for (std::size_t i = 0; i < refs.size() && budget.result.evaluations < budget.cap; ++i) {
      for (std::size_t j = i + 1; j < refs.size() && budget.result.evaluations < budget.cap; ++j) {
        const bool i_better = refs[i].value <= refs[j].value;
        auto current = refs[i_better ? j : i].point;
        const auto& guide = refs[i_better ? i : j].point;
        path_best.reset();
        detail::relink(current, guide, budget, distance, enumerate, apply, rng);
        if (path_best) pool.push_back(std::move(*path_best));
      }
    }
    capture = false;
    if (budget.result.evaluations == before) {
      budget.result.stop = NeighborhoodStop::no_move;
      break;
    }
  }
  return std::move(budget.result);
}

} // namespace datamunge::optim
