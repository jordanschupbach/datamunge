#pragma once

/// @file
/// @brief Greedy randomized adaptive construction and budgeted multistart descent.

#include <datamunge/optim/path_relink.hpp>

namespace datamunge::optim {

/// @brief Select uniformly from a threshold-based restricted candidate list.
///
/// A move is eligible when its heuristic cost is at most
/// lerp(minimum heuristic, maximum heuristic, alpha). Costs can have either sign.
/// @tparam Move Copyable typed move.
/// @param candidates Ordered feasible moves paired with finite heuristic costs.
/// @param alpha Greediness control in [0, 1]; zero restricts selection to tied best moves.
/// @param rng Borrowed local engine.
/// @return Selected move, or an empty optional for an empty candidate list.
/// @throws std::invalid_argument Invalid alpha or nonfinite heuristic.
/// @par Example
/// @code
/// std::mt19937_64 rng(42);
/// const auto move = dm::optim::grasp_choose(std::vector{std::pair{2, 1.0}, std::pair{3, 2.0}}, 0, rng);
/// std::print("{}\n", *move); // 2
/// @endcode
/// @par References
/// - Resende and Ribeiro (2016). Optimization by GRASP. https://doi.org/10.1007/978-1-4939-6530-4
/// @see grasp_search()
template <class Move>
std::optional<Move> grasp_choose(const std::vector<std::pair<Move, double>>& candidates,
                                 double alpha, std::mt19937_64& rng) {
  if (!(alpha >= 0 && alpha <= 1)) throw std::invalid_argument("GRASP alpha must be in [0, 1]");
  if (candidates.empty()) return {};
  double best = candidates.front().second;
  double worst = best;
  for (const auto& candidate : candidates) {
    if (!std::isfinite(candidate.second)) throw std::invalid_argument("GRASP heuristic must be finite");
    best = std::min(best, candidate.second);
    worst = std::max(worst, candidate.second);
  }
  const auto threshold = std::lerp(best, worst, alpha);
  std::vector<std::size_t> restricted;
  for (std::size_t i = 0; i < candidates.size(); ++i) {
    if (candidates[i].second <= threshold) restricted.push_back(i);
  }
  const auto chosen = std::uniform_int_distribution<std::size_t>(0, restricted.size() - 1)(rng);
  return candidates[restricted[chosen]].first;
}

/// @brief Budget, construction randomness and adaptive alpha controls.
/// @par Example
/// @code
/// dm::optim::GraspOptions options;
/// options.alphas = {0.1, 0.3, 0.6};
/// @endcode
/// @see grasp_search()
struct GraspOptions {
  VariableNeighborhoodOptions local; ///< Global budget, seed and descent order.
  std::vector<double> alphas{0.3}; ///< Positive-size list of construction controls in [0, 1].
  std::size_t adaptation_interval{8}; ///< Number of completed starts between probability updates; zero disables.
};

/// @brief Owned results from randomized construction and local descent.
/// @tparam State Type representing a feasible state.
/// @par Example
/// @code
/// const dm::optim::GraspResult<int> result{{0, 0, 1, 0, dm::optim::NeighborhoodStop::budget}, {1}, 0};
/// @endcode
/// @see grasp_search()
template <class State>
struct GraspResult {
  NeighborhoodResult<State> search; ///< Best scored state and work counts.
  std::vector<double> alpha_probabilities; ///< Normalized final probabilities per alpha.
  std::size_t constructions; ///< Successful constructions scored.
};

namespace detail {

/// @brief Run a budgeted GRASP loop with a hook for post-descent intensification.
/// @tparam State Complete feasible state.
/// @tparam Score Original objective.
/// @tparam Construct Randomized greedy constructor.
/// @tparam Enumerate Local move enumerator.
/// @tparam Apply Feasible move applicator.
/// @tparam Intensify Optional operation combining the new descent with a guide.
/// @param initial Feasible starting state.
/// @param score Objective to minimize.
/// @param construct Randomized greedy constructor.
/// @param enumerate Local move enumerator.
/// @param apply Feasible move applicator.
/// @param intensify Operation after each construction and descent, borrowing the shared budget.
/// @param options GRASP budget, seed, alpha and adaptation controls.
/// @return Best scored state and work and alpha selection counts.
template <class State, class Score, class Construct, class Enumerate, class Apply, class Intensify>
GraspResult<State> grasp_loop(State initial, Score& score, Construct& construct, Enumerate& enumerate,
                              Apply& apply, Intensify& intensify, const GraspOptions& options) {
  const auto& local = options.local;
  if (local.evaluations == 0 || local.neighborhoods == 0 || options.alphas.empty() ||
      !std::ranges::all_of(options.alphas, [](double a) { return a >= 0 && a <= 1; })) {
    throw std::invalid_argument("Invalid GRASP evaluation cap or construction controls");
  }
  detail::NeighborhoodBudget<State, std::remove_reference_t<Score>> budget{
      score, local.evaluations, {initial, std::numeric_limits<double>::infinity(), 0, 0, NeighborhoodStop::budget}};
  std::mt19937_64 rng(local.seed);
  std::vector<std::size_t> order(local.neighborhoods);
  auto value = budget.evaluate(initial);
  detail::descent_order(order, local, rng);
  detail::descend(initial, value, budget, enumerate, apply, order, rng);

  std::vector<double> weights(options.alphas.size(), 1);
  std::vector<double> means(options.alphas.size());
  std::vector<std::size_t> uses(options.alphas.size());
  std::discrete_distribution<std::size_t> select(weights.begin(), weights.end());
  std::size_t starts = 0;
  while (budget.result.evaluations < budget.cap) {
    const auto alpha_index = select(rng);
    auto candidate = std::invoke(construct, options.alphas[alpha_index], rng);
    if (!candidate) {
      budget.result.stop = NeighborhoodStop::no_move;
      break;
    }
    ++starts;
    const State guide = budget.result.best;
    value = budget.evaluate(*candidate);
    detail::descent_order(order, local, rng);
    detail::descend(*candidate, value, budget, enumerate, apply, order, rng);
    intensify(*candidate, guide, budget, rng);
    if (weights.size() == 1 || options.adaptation_interval == 0) continue;

    const auto count = ++uses[alpha_index];
    means[alpha_index] = std::lerp(means[alpha_index], value, 1.0 / count);
    if (starts % options.adaptation_interval != 0) continue;
    double best = std::numeric_limits<double>::infinity();
    double worst = -best;
    for (std::size_t i = 0; i < means.size(); ++i) {
      if (uses[i] == 0) continue;
      best = std::min(best, means[i]);
      worst = std::max(worst, means[i]);
    }
    const double spread = worst / 2 - best / 2;
    if (spread > 0) {
      for (std::size_t i = 0; i < means.size(); ++i) {
        if (uses[i] == 0) continue;
        const double relative = (means[i] / 2 - best / 2) / spread;
        weights[i] = 1 / std::pow(1 + relative, 4);
      }
      select.param({weights.begin(), weights.end()});
    }
  }
  return {std::move(budget.result), select.probabilities(), starts};
}

} // namespace detail

/// @brief Run GRASP with optional reactive construction probabilities.
///
/// The initial feasible state is scored once and locally improved. Subsequent
/// iterations construct a new state using the chosen alpha and descend from it.
/// Multiple alpha values with a positive adaptation interval enable reactive
/// GRASP. Selection weights are inverse powers of losses above the best
/// historical mean across used choices, scaled by their observed mean spread.
/// Unused alphas retain unit weight so they remain available. The interval uses
/// completed starts. Duplicate states consume budget. An empty construction
/// terminates the search. Callback exceptions propagate. Deterministic
/// callbacks replay within a build for a fixed seed.
/// @tparam State Copyable feasible state type.
/// @tparam Score Callable returning finite loss for const State&.
/// @tparam Construct Callable returning optional<State> from alpha and a local engine.
/// @tparam Enumerate Callable returning finite ordered moves from state, index and engine.
/// @tparam Apply Callable applying a typed move while preserving feasibility.
/// @param initial Feasible starting state, taken by value.
/// @param score Objective to minimize.
/// @param construct Randomized greedy constructor; grasp_choose() can select partial moves.
/// @param enumerate Local neighborhood generator.
/// @param apply Feasible move applicator.
/// @param options Budget, seed, neighborhood order and reactive controls.
/// @return Observed incumbent, call counts and final construction probabilities.
/// @throws std::invalid_argument Invalid budget, neighborhood count or alphas.
/// @throws std::domain_error Objective result is nonfinite.
/// @par Example
/// @code
/// const auto result = dm::optim::grasp_search(
///     2, [](int x) { return x*x; },
///     [](double, auto&) -> std::optional<int> { return 0; },
///     [](int, std::size_t, auto&) { return std::vector<int>{}; },
///     [](int& x, int m) { x += m; }, {{4}});
/// std::print("{}\n", result.search.best); // 0
/// @endcode
/// @par References
/// - Resende and Ribeiro (2016). Optimization by GRASP. https://doi.org/10.1007/978-1-4939-6530-4
/// @see grasp_choose(), path_relink(), iterated_local_search()
template <class State, class Score, class Construct, class Enumerate, class Apply>
GraspResult<State> grasp_search(State initial, Score&& score, Construct&& construct,
                               Enumerate&& enumerate, Apply&& apply, const GraspOptions& options = {}) {
  /// Make no additional evaluations after ordinary descent.
  const auto no_intensification = [](State&, const State&, auto&, auto&) {};
  return detail::grasp_loop(std::move(initial), score, construct, enumerate, apply,
                           no_intensification, options);
}

/// @brief Run GRASP with forward path relinking into the best point from prior starts.
///
/// After each constructed state is locally optimized, a greedy path moves it
/// toward the previous incumbent. The same cap counts construction, descent,
/// and every scored path trial. Strictly distance-reducing moves are scored;
/// the descent endpoint is already scored and reused. The returned incumbent
/// can be on an intermediate path. Callbacks are responsible for feasibility
/// and their exceptions propagate. See grasp_search() for reactive alpha semantics.
/// @tparam State Copyable feasible state.
/// @tparam Score Objective accepting const State& and returning finite loss.
/// @tparam Construct Randomized greedy constructor returning optional<State>.
/// @tparam Enumerate Local move enumerator.
/// @tparam Apply Local move applicator.
/// @tparam Distance Nonnegative integral distance, with zero meaning equivalent.
/// @tparam Forward Generator of feasible toward-guide moves.
/// @tparam ApplyForward Path move applicator.
/// @param initial Feasible starting point.
/// @param score Objective to minimize.
/// @param construct Constructor forwarded alpha and engine.
/// @param enumerate Ordered local neighborhood generator.
/// @param apply Feasible local move applicator.
/// @param distance Distance to the guiding state.
/// @param forward Ordered path move generator given current state, guiding state and engine.
/// @param apply_forward Feasible path move applicator.
/// @param options Overall budget, seed and construction settings.
/// @return Best scored state and work counts.
/// @throws std::invalid_argument Invalid GRASP options.
/// @throws std::domain_error Objective returns nonfinite loss.
/// @par Example
/// @code
/// const auto result = dm::optim::grasp_path_relink(
///     0, [](int x) { return (x-1)*(x-1); },
///     [](double, auto&) -> std::optional<int> { return 2; },
///     [](int, std::size_t, auto&) { return std::vector<int>{}; },
///     [](int& x, int move) { x += move; },
///     [](int x, int y) { return std::size_t(std::abs(x-y)); },
///     [](int, int, auto&) { return std::vector{-1}; },
///     [](int& x, int move) { x += move; }, {{5}});
/// std::print("{}\n", result.search.best); // 1
/// @endcode
/// @par References
/// - Resende and Ribeiro (2016). Optimization by GRASP. https://doi.org/10.1007/978-1-4939-6530-4
/// @see grasp_search(), path_relink()
template <class State, class Score, class Construct, class Enumerate, class Apply,
          class Distance, class Forward, class ApplyForward>
GraspResult<State> grasp_path_relink(
    State initial, Score&& score, Construct&& construct, Enumerate&& enumerate, Apply&& apply,
    Distance&& distance, Forward&& forward, ApplyForward&& apply_forward, const GraspOptions& options = {}) {
  /// Relink under the existing GRASP budget, without scoring the start again.
  auto intensify = [&](State& candidate, const State& guide, auto& budget, auto& rng) {
    if (budget.result.evaluations < budget.cap) {
      detail::relink(candidate, guide, budget, distance, forward, apply_forward, rng);
    }
  };
  return detail::grasp_loop(std::move(initial), score, construct, enumerate, apply, intensify, options);
}

} // namespace datamunge::optim
