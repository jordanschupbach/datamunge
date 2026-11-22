#pragma once

/// @file
/// @brief Generic synchronous resource-budgeted population-based training.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <numeric>
#include <optional>
#include <random>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace datamunge::optim {

namespace detail {

/// @brief Rank a same-fidelity population and pair good donors with worst recipients.
/// @tparam Index Integer type used for worker indices.
/// @param losses Finite validation losses in worker order.
/// @param replacements Number of workers to replace, at most half the population.
/// @return Donor indices in destination worker order.
/// @throws std::invalid_argument Empty/nonfinite losses or invalid replacement count.
template <class Index>
std::vector<Index> population_sources(std::span<const double> losses, std::size_t replacements) {
  if (losses.empty() || replacements > losses.size() / 2 ||
      !std::ranges::all_of(losses, [](double loss) { return std::isfinite(loss); })) {
    throw std::invalid_argument("Invalid population losses or replacement count");
  }
  std::vector<Index> order(losses.size());
  std::iota(order.begin(), order.end(), Index{0});
  auto sources = order;
  std::stable_sort(order.begin(), order.end(), [&](Index a, Index b) { return losses[a] < losses[b]; });
  for (std::size_t i = 0; i < replacements; ++i) sources[order[order.size() - replacements + i]] = order[i];
  return sources;
}

} // namespace detail

/// @brief Work granularity and exploit/explore controls.
/// @par Example
/// @code
/// const dm::optim::PopulationTrainingOptions options{512, 12, 4, 32, 1};
/// @endcode
/// @see population_based_training()
struct PopulationTrainingOptions {
  std::size_t resource_budget{1000}; ///< Positive sum of training resources across workers.
  std::uint64_t seed{42}; ///< Seed for hyperparameter exploration.
  std::size_t interval{4}; ///< Positive-resource tranche between exploit operations.
  std::size_t maximum_resource{64}; ///< Positive per-worker cumulative resource ceiling.
  std::size_t replacements{1}; ///< Number of workers replaced after a complete round, at most half.
};

/// @brief Owned best measured checkpoint and its comparable fidelity.
/// @tparam State Copyable independent worker checkpoint.
/// @par Example
/// @code
/// const dm::optim::TrainingObservation<int> observation{7, 0, 4, 1.0};
/// @endcode
/// @see PopulationTrainingResult
template <class State>
struct TrainingObservation {
  State checkpoint; ///< State copied at the time of scoring.
  std::size_t worker; ///< Original destination worker index.
  std::size_t resource; ///< Cumulative fidelity at evaluation.
  double loss; ///< Validation score to minimize.
};

/// @brief Trained workers, best checkpoint and actual work counts.
/// @tparam State Copyable independent worker checkpoint.
/// @par Example
/// @code
/// const dm::optim::PopulationTrainingResult<int> result;
/// std::print("{}\n", result.consumed); // 0
/// @endcode
/// @see population_based_training()
template <class State>
struct PopulationTrainingResult {
  std::vector<State> workers; ///< Final states in original worker order.
  std::optional<TrainingObservation<State>> best; ///< Lowest loss at the highest evaluated fidelity.
  std::size_t consumed{}; ///< Total additional training resources consumed.
  std::size_t evaluations{}; ///< Calls to advance and validate one worker.
  std::size_t replacements{}; ///< Actual donor assignments followed by exploration.
};

/// @brief Train workers, copy strong checkpoints, and explore new hyperparameters.
///
/// Each worker starts at resource zero. A round advances workers in index
/// order by equal tranches, with the final tranche clipped to the per-worker
/// ceiling. The scheduler does not dispatch a tranche exceeding the remaining
/// global budget. A partial final round has no donor assignments.
/// After a full round, copies the best checkpoints to the worst workers from
/// a snapshot of the completed states, then mutates the copies. Copying and
/// mutation use no advance calls. The result compares losses at the highest
/// measured fidelity. Callback exceptions propagate. Deterministic callbacks
/// replay within a build for a fixed seed.
/// @tparam State Independently copyable checkpoint including hyperparameters.
/// @tparam Advance Callable mutating one checkpoint and returning finite loss.
/// @tparam Explore Callable mutating a copied checkpoint using a local engine.
/// @param initial Feasible worker checkpoints, taken by value.
/// @param advance Training and validation callback taking state, previous and target cumulative resource.
/// @param explore Hyperparameter mutation callback taking state and engine.
/// @param options Additive cap, seed, train interval, replacement count and ceiling.
/// @return Final states, highest-fidelity best checkpoint and resource consumption.
/// @throws std::invalid_argument Empty population, invalid budget or replacement count.
/// @throws std::domain_error Advance callback returns a nonfinite loss.
/// @par Example
/// @code
/// const auto result = dm::optim::population_based_training(
///     std::vector{3, 1}, [](int& x, std::size_t, std::size_t) { return double(x); },
///     [](int&, auto&) {}, {8, 7, 2, 6, 1});
/// std::print("{}\n", result.best->loss); // 1
/// @endcode
/// @par References
/// - Jaderberg et al. (2017). Population Based Training of Neural Networks.
///   https://arxiv.org/abs/1711.09846
/// @see resource_search()
template <class State, class Advance, class Explore>
PopulationTrainingResult<State> population_based_training(
    std::vector<State> initial, Advance&& advance, Explore&& explore, const PopulationTrainingOptions& options = {}) {
  if (initial.empty() || options.resource_budget == 0 || options.interval == 0 ||
      options.maximum_resource == 0 || options.replacements > initial.size() / 2) {
    throw std::invalid_argument("Invalid population training options or initial population");
  }
  PopulationTrainingResult<State> result;
  result.workers = std::move(initial);
  std::vector<double> losses(result.workers.size());
  std::mt19937_64 rng(options.seed);
  std::size_t previous = 0;
  while (previous < options.maximum_resource) {
    const auto increments = std::min(options.interval, options.maximum_resource - previous);
    const auto next = previous + increments;
    const auto count = std::min(result.workers.size(), (options.resource_budget - result.consumed) / increments);
    for (std::size_t i = 0; i < count; ++i) {
      auto& worker = result.workers[i];
      const double loss = std::invoke(advance, worker, previous, next);
      if (!std::isfinite(loss)) throw std::domain_error("Population training loss must be finite");
      result.consumed += increments;
      ++result.evaluations;
      losses[i] = loss;
      if (!result.best || next > result.best->resource ||
          (next == result.best->resource && loss < result.best->loss)) {
        result.best.emplace(TrainingObservation<State>{worker, i, next, loss});
      }
    }
    if (count != result.workers.size() || next == options.maximum_resource ||
        options.resource_budget - result.consumed < std::min(options.interval, options.maximum_resource - next)) {
      break;
    }
    if (options.replacements != 0) {
      const auto sources = detail::population_sources<std::size_t>(losses, options.replacements);
      for (std::size_t i = 0; i < sources.size(); ++i) {
        if (i == sources[i]) continue;
        // Donors and recipients are disjoint at this replacement fraction.
        result.workers[i] = result.workers[sources[i]];
        std::invoke(explore, result.workers[i], rng);
        ++result.replacements;
      }
    }
    previous = next;
  }
  return result;
}

} // namespace datamunge::optim
