#pragma once

/// @file
/// @brief Typed MAP-Elites and novelty search with optional local competition.

#include <datamunge/optim/nsga2.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <deque>
#include <functional>
#include <limits>
#include <numeric>
#include <optional>
#include <random>
#include <type_traits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace datamunge::optim {

/// @brief Budget and behavior-cell count for typed MAP-Elites.
/// @par Example
/// @code
/// const dm::optim::MapElitesOptions options{100, 7, 12, 8};
/// @endcode
/// @see map_elites_search()
struct MapElitesOptions {
  std::size_t evaluations{1000}; ///< Positive budget, counting all scored candidates.
  std::uint64_t seed{42}; ///< Local seed for generation, parent choice and mutation.
  std::size_t cells{64}; ///< Number of behavior cells, at least one.
  std::size_t initial_samples{16}; ///< Positive count of initial random states.
};

/// @brief One occupied behavior cell and its earliest best candidate.
/// @tparam State Copyable complete state.
/// @par Example
/// @code
/// const dm::optim::MapElite<int> elite{2, 1, 0};
/// @endcode
/// @see MapElitesResult
template <class State>
struct MapElite {
  State state; ///< Retained candidate.
  double loss; ///< Finite objective loss at the candidate.
  std::size_t evaluation; ///< Zero-based evaluation index.
};

/// @brief Typed behavior grid and best observed cell.
/// @tparam State Copyable feasible state.
/// @par Example
/// @code
/// const dm::optim::MapElitesResult<int> run;
/// std::print("{}\n", run.evaluations); // 0
/// @endcode
/// @see map_elites_search()
template <class State>
struct MapElitesResult {
  std::vector<std::optional<MapElite<State>>> cells; ///< Owned elites in behavior-cell order.
  std::size_t evaluations{}; ///< Actual score calls.
  std::size_t best_cell{}; ///< Index of the cell holding the overall incumbent.
};

/// @brief Fill behavior cells by mutation of uniformly chosen occupied elites.
///
/// The generator supplies the initial population. Thereafter choose one
/// occupied cell uniformly, copy its elite and mutate the copy. A candidate
/// displaces an incumbent only on strict improvement in the same cell.
/// Every valid trial counts, including repeated states. The caller is
/// responsible for feasibility and behavioral discretization. Exceptions
/// propagate. Seeded replay holds for deterministic callbacks within a build.
/// @tparam Generate Callable returning a complete feasible state from a local engine.
/// @tparam Mutate Callable updating a state in place using the local engine.
/// @tparam Score Callable returning a finite loss for const State&.
/// @tparam Cell Callable returning a zero-based integer cell for const State&.
/// @param generate Feasible state generator.
/// @param mutate Mutation callback preserving feasibility.
/// @param score Quality to minimize within each cell.
/// @param cell Behavior callback mapping a state to a cell in the configured grid.
/// @param options Budget, seed and behavior grid limits.
/// @return Owned cells, overall incumbent cell and exact objective-call count.
/// @throws std::invalid_argument Invalid controls or out-of-range cell.
/// @throws std::domain_error Objective result is nonfinite.
/// @par Example
/// @code
/// const auto run = dm::optim::map_elites_search(
///     [](auto&) { return 1; }, [](int& x, auto&) { x = 0; },
///     [](int x) { return x*x; }, [](int x) { return std::size_t(x); },
///     {4, 7, 2, 1});
/// std::print("{}\n", run.best_cell); // 0
/// @endcode
/// @par References
/// - Mouret and Clune (2015), Illuminating search spaces by mapping elites.
///   https://arxiv.org/abs/1504.04909
/// @see novelty_search(), dm::rl::MapElites
template <class Generate, class Mutate, class Score, class Cell>
auto map_elites_search(Generate&& generate, Mutate&& mutate, Score&& score, Cell&& cell,
                       const MapElitesOptions& options = {}) {
  using State = std::remove_cvref_t<std::invoke_result_t<Generate&, std::mt19937_64&>>; ///< Candidate type.
  if (options.evaluations == 0 || options.cells == 0 || options.initial_samples == 0) {
    throw std::invalid_argument("Invalid MAP-Elites options");
  }

  MapElitesResult<State> result;
  result.cells.resize(options.cells);
  std::vector<std::size_t> occupied;
  double best = std::numeric_limits<double>::infinity();
  std::mt19937_64 rng(options.seed);
  while (result.evaluations < options.evaluations) {
    const bool initial = result.evaluations < options.initial_samples;
    State candidate = initial ? std::invoke(generate, rng)
        : result.cells[occupied[std::uniform_int_distribution<std::size_t>(0, occupied.size() - 1)(rng)]]
            ->state;
    if (!initial) std::invoke(mutate, candidate, rng);
    const auto index = static_cast<std::size_t>(std::invoke(cell, std::as_const(candidate)));
    if (index >= options.cells) throw std::invalid_argument("MAP-Elites behavior cell is out of range");
    const double loss = std::invoke(score, std::as_const(candidate));
    if (!std::isfinite(loss)) throw std::domain_error("MAP-Elites loss must be finite");
    auto& slot = result.cells[index];
    if (!slot) occupied.push_back(index);
    if (!slot || loss < slot->loss) slot.emplace(MapElite<State>{std::move(candidate), loss, result.evaluations});
    if (loss < best) {
      best = loss;
      result.best_cell = index;
    }
    ++result.evaluations;
  }
  return result;
}

/// @brief Budget, neighbor count and novelty archive admission settings.
/// @par Example
/// @code
/// dm::optim::NoveltyOptions options;
/// options.local_competition = true;
/// @endcode
/// @see novelty_search()
struct NoveltyOptions {
  std::size_t evaluations{1000}; ///< Positive call budget, including initialization and repeats.
  std::uint64_t seed{42}; ///< Local seed for mutation and archive admission.
  std::size_t population{24}; ///< Number of surviving parents, at least two.
  std::size_t neighbors{8}; ///< Positive number of nearest behavioral references.
  std::size_t archive_capacity{256}; ///< Maximum archive entries; zero disables the archive.
  double archive_probability{0.1}; ///< Independent admission probability per trial in [0,1].
  bool local_competition{false}; ///< Select by Pareto rank on novelty and fraction of local wins.
};

/// @brief Objective incumbent and observed archive size.
/// @tparam State Copyable feasible state.
/// @par Example
/// @code
/// const dm::optim::NoveltyResult<int> result{0, 0, 100, 20};
/// @endcode
/// @see novelty_search()
template <class State>
struct NoveltyResult {
  State best; ///< Earliest highest-quality scored state.
  double loss; ///< Original observed loss of best.
  std::size_t evaluations; ///< Actual quality evaluations.
  std::size_t archive_size; ///< Number of retained behavioral records.
};

namespace detail {

/// @brief Behavioral reference with the associated quality and unique evaluation index.
/// @tparam Behavior Copyable descriptor returned by the caller.
template <class Behavior>
struct BehavioralRecord {
  Behavior behavior; ///< Behavioral descriptor.
  double loss; ///< Scored quality.
  std::size_t id; ///< Unique objective evaluation index.
};

/// @brief One feasible candidate and metrics for selecting its descendants.
/// @tparam State Copyable candidate.
/// @tparam Behavior Copyable descriptor.
template <class State, class Behavior>
struct NovelCandidate {
  State state; ///< Complete feasible state.
  BehavioralRecord<Behavior> record; ///< Behavioral coordinates and score.
  double novelty{}; ///< Mean distance to nearest references.
  double competition{}; ///< Fraction of nearest neighbors beaten on quality.
  std::size_t rank{}; ///< Environmental rank using both criteria when enabled.
  double crowding{}; ///< Environmental diversity tie-break.
};

/// @brief Compute nearest-neighbor metrics and optional Pareto ranking.
/// @tparam State Copyable feasible state.
/// @tparam Behavior Copyable descriptor.
/// @tparam Distance Behavioral distance callable.
/// @param population Cohort to update.
/// @param archive Stored behavioral references, including possibly current candidates.
/// @param options Neighbor count and selection criteria.
/// @param distance Borrowed nonnegative finite distance callback.
template <class State, class Behavior, class Distance>
void novelty_metrics(std::vector<NovelCandidate<State, Behavior>>& population,
    const std::deque<BehavioralRecord<Behavior>>& archive, const NoveltyOptions& options, Distance& distance) {
  for (auto& point : population) {
    std::vector<std::pair<double, double>> nearest;
    /// Add a valid reference unless it is its own query.
    const auto add = [&](const BehavioralRecord<Behavior>& other) {
      if (point.record.id == other.id) return;
      const double delta = std::invoke(distance, std::as_const(point.record.behavior), other.behavior);
      if (!(delta >= 0 && std::isfinite(delta))) throw std::domain_error("Invalid behavioral distance");
      nearest.emplace_back(delta, other.loss);
    };
    for (const auto& other : population) add(other.record);
    for (const auto& other : archive) {
      if (!std::ranges::any_of(population, [&](const auto& active) { return active.record.id == other.id; })) {
        add(other);
      }
    }
    const auto n = std::min(options.neighbors, nearest.size());
    std::stable_sort(nearest.begin(), nearest.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
    long double total = 0;
    std::size_t wins = 0;
    for (std::size_t i = 0; i < n; ++i) {
      total += static_cast<long double>(nearest[i].first) / n;
      wins += point.record.loss < nearest[i].second;
    }
    point.novelty = static_cast<double>(total);
    point.competition = n == 0 ? 0 : static_cast<double>(wins) / n;
  }
  if (options.local_competition) {
    std::vector<ParetoPoint> metrics;
    double maximum = 1;
    for (const auto& point : population) maximum = std::max(maximum, point.novelty);
    for (const auto& point : population) metrics.push_back({{}, {-point.novelty / maximum, -point.competition}});
    const auto ranking = nsga2_rank(metrics);
    for (std::size_t i = 0; i < population.size(); ++i) {
      population[i].rank = ranking.ranks[i];
      population[i].crowding = ranking.crowding[i];
    }
  }
}

} // namespace detail

/// @brief Evolve toward behavioral novelty, optionally competing on local quality.
///
/// Novelty is the mean distance to the nearest active or archived behavioral
/// references, excluding a candidate's own record. The bounded archive admits
/// candidates randomly and evicts oldest first. Ordinary novelty selection
/// compares only novelty. Local competition uses nondominated sorting on
/// novelty and the fraction of neighbors beaten on quality. Both modes
/// retain a separate best observed quality. Mutations are applied to owned
/// copies. Every generated state is scored once and counted, including
/// duplicates or an incomplete last cohort. Callback exceptions propagate.
/// @tparam Generate Callable generating a complete feasible state from the local engine.
/// @tparam Mutate Callable mutating a state in place.
/// @tparam Score Callable scoring an immutable state.
/// @tparam Describe Callable returning a copyable behavior descriptor.
/// @tparam Distance Callable returning finite nonnegative distance between descriptors.
/// @param generate Feasible initial-state generator.
/// @param mutate Feasible typed mutation, borrowing the engine.
/// @param score Finite quality objective, minimized.
/// @param describe Map a state to an owned descriptor.
/// @param distance Behavioral dissimilarity, zero for identical behaviors.
/// @param options Work, seed, archive and competition settings.
/// @return Owned quality incumbent and observed work, including archive size.
/// @throws std::invalid_argument Invalid controls.
/// @throws std::domain_error Nonfinite loss or invalid behavioral distance.
/// @par Example
/// @code
/// const auto run = dm::optim::novelty_search(
///     [](auto&) { return 1; }, [](int& x, auto&) { x = 0; },
///     [](int x) { return x*x; }, [](int x) { return x; },
///     [](int a, int b) { return std::abs(a-b); }, {8, 7, 3});
/// std::print("{}\n", run.best); // 0
/// @endcode
/// @par References
/// - Lehman and Stanley (2011). https://doi.org/10.1162/EVCO_a_00025
/// - Lehman and Stanley (2011), Evolving a diversity of virtual creatures through novelty search and local competition.
/// @see map_elites_search(), NSGA2
template <class Generate, class Mutate, class Score, class Describe, class Distance>
auto novelty_search(Generate&& generate, Mutate&& mutate, Score&& score,
                    Describe&& describe, Distance&& distance, const NoveltyOptions& options = {}) {
  using State = std::remove_cvref_t<std::invoke_result_t<Generate&, std::mt19937_64&>>; ///< Candidate type.
  using Behavior = std::remove_cvref_t<std::invoke_result_t<Describe&, const State&>>; ///< Descriptor type.
  using Candidate = detail::NovelCandidate<State, Behavior>; ///< Scored cohort member.
  if (options.evaluations == 0 || options.population < 2 || options.neighbors == 0 ||
      !(options.archive_probability >= 0 && options.archive_probability <= 1)) {
    throw std::invalid_argument("Invalid novelty search controls");
  }

  std::mt19937_64 rng(options.seed);
  std::bernoulli_distribution admit(options.archive_probability);
  std::deque<detail::BehavioralRecord<Behavior>> archive;
  std::optional<State> best;
  double best_loss = std::numeric_limits<double>::infinity();
  std::size_t evaluations = 0;
  /// Score, describe and optionally admit one trial into the historical archive.
  auto evaluate = [&](State state) {
    const double value = std::invoke(score, std::as_const(state));
    if (!std::isfinite(value)) throw std::domain_error("Novelty search loss must be finite");
    auto descriptor = std::invoke(describe, std::as_const(state));
    if (value < best_loss) {
      best = state;
      best_loss = value;
    }
    detail::BehavioralRecord<Behavior> record{std::move(descriptor), value, evaluations++};
    if (options.archive_capacity != 0 && admit(rng)) {
      if (archive.size() == options.archive_capacity) archive.pop_front();
      archive.push_back(record);
    }
    return Candidate{std::move(state), std::move(record)};
  };
  /// Compare environment metrics by novelty or by Pareto rank then crowding.
  const auto preferred = [&](const Candidate& a, const Candidate& b) {
    if (!options.local_competition) return a.novelty > b.novelty;
    return a.rank != b.rank ? a.rank < b.rank : a.crowding > b.crowding;
  };

  const auto size = std::min(options.population, options.evaluations);
  std::vector<Candidate> population;
  for (std::size_t i = 0; i < size; ++i) population.push_back(evaluate(std::invoke(generate, rng)));
  while (evaluations < options.evaluations) {
    detail::novelty_metrics(population, archive, options, distance);
    std::uniform_int_distribution<std::size_t> parent(0, population.size() - 1);
    const auto count = std::min(size, options.evaluations - evaluations);
    for (std::size_t i = 0; i < count; ++i) {
      const auto a = parent(rng), b = parent(rng);
      State child = preferred(population[a], population[b]) ? population[a].state : population[b].state;
      std::invoke(mutate, child, rng);
      population.push_back(evaluate(std::move(child)));
    }
    if (evaluations == options.evaluations) break;
    detail::novelty_metrics(population, archive, options, distance);
    std::stable_sort(population.begin(), population.end(), preferred);
    population.erase(population.begin() + size, population.end());
  }

  return NoveltyResult<State>{std::move(*best), best_loss, evaluations, archive.size()};
}

} // namespace datamunge::optim
