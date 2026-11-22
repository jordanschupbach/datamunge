#pragma once

/// @file
/// @brief Resource-aware trial promotion for successive halving, Hyperband and ASHA.

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <vector>

namespace datamunge::optim {

/// @brief Trial-start and promotion schedule.
/// @par Example
/// @code
/// const auto method = dm::optim::ResourceSearchMethod::asha;
/// @endcode
/// @see resource_search()
enum class ResourceSearchMethod {
  successive_halving, ///< One synchronous bracket with increasing fidelity.
  hyperband,          ///< Multiple synchronous brackets at different starting fidelities.
  asha                ///< Logical asynchronous workers with promotion on completed observations.
};

/// @brief Cumulative rung levels, trial limits and global resource cap.
/// @par Example
/// @code
/// dm::optim::ResourceSearchOptions options;
/// options.method = dm::optim::ResourceSearchMethod::asha;
/// options.resource_budget = 100;
/// @endcode
/// @see resource_search()
struct ResourceSearchOptions {
  ResourceSearchMethod method{ResourceSearchMethod::successive_halving}; ///< Scheduling policy.
  std::size_t resource_budget{100}; ///< Positive cap on total added resource across trials.
  std::size_t trials{16}; ///< Positive cap on fresh trials.
  std::size_t minimum_resource{1}; ///< Positive cumulative resource of the first rung.
  std::size_t maximum_resource{16}; ///< Maximum cumulative resource, at least the minimum.
  std::size_t reduction{2}; ///< Integral promotion factor, at least two.
  std::size_t workers{2}; ///< Positive number of logical ASHA workers.
  std::uint64_t seed{42}; ///< Seed of the local trial-seed generator.
  std::size_t trace_limit{0}; ///< Maximum number of completion events to retain; zero disables.
};

/// @brief Chronological completion of one trial tranche.
/// @par Example
/// @code
/// const dm::optim::ResourceObservation observation{0, 0, 1, 0, 2.0, 1.0};
/// @endcode
/// @see ResourceSearchResult
struct ResourceObservation {
  std::size_t trial; ///< Fresh trial identifier, starting from zero.
  std::size_t previous; ///< Cumulative fidelity before this tranche.
  std::size_t resource; ///< Cumulative fidelity after this tranche.
  std::size_t bracket; ///< Hyperband bracket index; zero for other methods.
  double loss; ///< Finite observed validation loss to minimize.
  double completion; ///< Simulated completion time for ASHA, or -1 for synchronous runs.
};

/// @brief Owned search result with highest-fidelity score, work and optional events.
/// @par Example
/// @code
/// const dm::optim::ResourceSearchResult result;
/// std::print("{}\n", result.consumed); // 0
/// @endcode
/// @see resource_search()
struct ResourceSearchResult {
  std::optional<ResourceObservation> best; ///< Lowest loss at the highest evaluated fidelity.
  std::size_t consumed{}; ///< Sum of the incremental resource used by completed tranches.
  std::size_t starts{}; ///< Number of fresh trials initialized.
  std::size_t evaluations{}; ///< Number of completed score callbacks.
  std::vector<ResourceObservation> trace; ///< At most trace_limit chronological completions.
};

/// @brief Initialize a fresh trial with its integer identifier and derived seed.
using StartTrial = std::function<void(std::size_t, std::uint64_t)>;

/// @brief Advance one trial from its previous cumulative fidelity to its target fidelity.
using AdvanceTrial = std::function<double(std::size_t, std::size_t, std::size_t)>;

/// @brief Simulated duration for an incremental amount of resource on a trial.
using TrialDuration = std::function<double(std::size_t, std::size_t)>;

/// @brief Allocate increasing fidelity under a shared additive resource budget.
///
/// Trial IDs are distinct and assigned in order. State lives with the caller
/// and a promotion continues the same trial, charging only the added resource.
/// Synchronous rungs keep the top floor(count/reduction) trials with a minimum
/// of one. Hyperband chooses starting counts from its bracket formula and
/// shares both caps across brackets. ASHA uses simulated completion times.
/// It reserves resource on dispatch and considers promotions from high rungs
/// first. Top floor(count/reduction) completed observations are eligible.
/// The scheduler completes every dispatched tranche and never exceeds either cap.
/// Callback exceptions propagate. Deterministic callbacks replay within a
/// build at the same seed. Best compares scores only at the highest fidelity.
/// @param start Borrowed callback initializing a trial before its first advancement.
/// @param advance Borrowed callback returning finite validation loss at the target fidelity.
/// @param options Fidelity ladder, budget, starts, method and trace controls.
/// @param duration Optional ASHA duration callback, default incremental resource as elapsed time.
/// @return Owned work totals and highest-fidelity best observation, if available.
/// @throws std::invalid_argument Empty required callback or invalid controls.
/// @throws std::domain_error Nonfinite loss or invalid simulated duration.
/// @par Example
/// @code
/// const auto run = dm::optim::resource_search([](std::size_t, std::uint64_t) {},
///     [](std::size_t trial, std::size_t, std::size_t) { return double(trial); });
/// std::print("{}\n", run.best->trial); // 0
/// @endcode
/// @par References
/// - Li et al. (2018), Hyperband. https://jmlr.org/papers/v18/16-558.html
/// - Li et al. (2020), A System for Massively Parallel Hyperparameter Tuning.
///   https://arxiv.org/abs/1810.05934
/// @see ResourceSearchOptions, ResourceObservation
ResourceSearchResult resource_search(const StartTrial& start, const AdvanceTrial& advance,
                                     const ResourceSearchOptions& options = {}, const TrialDuration& duration = {});

} // namespace datamunge::optim
