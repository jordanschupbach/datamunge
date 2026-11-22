#pragma once

/// @file
/// @brief Model-based and racing search across typed conditional parameter spaces.

#include <datamunge/optim/resource_search.hpp>
#include <datamunge/optim/simulation_search.hpp>

#include <variant>

namespace datamunge::optim {

/// @brief Kind of a parameter in a mixed Cartesian space.
/// @par Example
/// @code
/// const auto kind = dm::optim::ParameterKind::integer;
/// @endcode
/// @see MixedParameter
enum class ParameterKind {
  continuous, ///< Bounded double.
  integer,     ///< Bounded integral value.
  categorical  ///< Zero-based category from a finite count.
};

/// @brief Activate a parameter only at a specific earlier categorical value.
/// @par Example
/// @code
/// const dm::optim::ParameterCondition condition{0, 1};
/// @endcode
/// @see MixedParameter
struct ParameterCondition {
  std::size_t parent; ///< Earlier categorical parameter.
  std::size_t category; ///< Required category at parent.
};

/// @brief One real, integer or categorical parameter and an optional parent condition.
/// @par Example
/// @code
/// const dm::optim::MixedParameter p{dm::optim::ParameterKind::integer, 1, 10};
/// @endcode
/// @see mixed_search()
struct MixedParameter {
  ParameterKind kind{ParameterKind::continuous}; ///< Parameter representation.
  double lower{0}; ///< Inclusive lower numeric bound.
  double upper{1}; ///< Inclusive upper numeric bound.
  std::size_t categories{2}; ///< Positive number of categorical choices.
  std::optional<ParameterCondition> condition{}; ///< Earlier categorical activation, or unconditionally active.
};

/// @brief Typed assignment, with monostate for an inactive parameter.
using MixedValue = std::variant<std::monostate, double, std::int64_t, std::size_t>;

/// @brief Values in parameter order, with inactive slots preserved.
using MixedConfiguration = std::vector<MixedValue>;

/// @brief Objective on a complete internally validated configuration.
using MixedObjective = std::function<double(const MixedConfiguration&)>;

/// @brief Repeated-simulation objective with a supplied replication seed.
using MixedSimulation = std::function<double(const MixedConfiguration&, std::uint64_t)>;

/// @brief Model and acquisition strategy for mixed search.
/// @par Example
/// @code
/// const auto method = dm::optim::MixedSearchMethod::smac;
/// @endcode
/// @see mixed_search()
enum class MixedSearchMethod {
  tpe,               ///< Conditional Parzen estimators and elite-to-remainder density ratio.
  smac,              ///< Random-forest expected improvement over global and perturbed candidates.
  surrogate_evolution ///< Evolve a proposal pool on a repeatedly refitted random-forest surrogate.
};

/// @brief Work, initialization, models and surrogate-search controls.
/// @par Example
/// @code
/// dm::optim::MixedSearchOptions options;
/// options.method = dm::optim::MixedSearchMethod::tpe;
/// @endcode
/// @see mixed_search(), bohb_search()
struct MixedSearchOptions {
  std::size_t evaluations{100}; ///< Positive true objective-call count, including duplicate configurations.
  std::uint64_t seed{42}; ///< Local engine seed.
  MixedSearchMethod method{MixedSearchMethod::tpe}; ///< Model and acquisition policy.
  std::size_t startup{8}; ///< Minimum observations before model-driven proposals, at least two.
  std::size_t candidates{32}; ///< Positive acquisition pool size, at least two for surrogate evolution.
  double elite_fraction{0.25}; ///< Parzen elite fraction strictly between zero and one.
  double random_fraction{0.1}; ///< Probability of a uniform exploration proposal in [0,1].
  std::size_t trees{16}; ///< Regression forest tree count, at least two.
  std::size_t evolution_steps{4}; ///< Positive surrogate-only generations per objective opportunity.
};

/// @brief Owned configuration incumbent and work counters.
/// @par Example
/// @code
/// const dm::optim::MixedSearchResult result;
/// std::print("{}\n", result.evaluations); // 0
/// @endcode
/// @see mixed_search()
struct MixedSearchResult {
  MixedConfiguration point; ///< Earliest best truly scored configuration.
  double value{}; ///< Loss observed at point.
  std::size_t evaluations{}; ///< True objective evaluations.
  std::size_t model_evaluations{}; ///< Candidate model/acquisition rankings, excluding fitting operations.
};

/// @brief Minimize a finite loss in a mixed, conditionally active parameter space.
///
/// Real and integer features use normalized coordinates. Integer bounds are
/// exact integral doubles with magnitude at most 2^52. Categories range
/// from zero to categories minus one and may activate later parameters.
/// Inactive values use monostate. TPE uses bounded Gaussian mixtures and
/// categorical counts on active rows. The forest models inactive features
/// explicitly and scores candidates by expected improvement or through an
/// inner evolutionary search on predicted means. Scores used for these
/// inner choices never enter the true incumbent or work count. All actual
/// objective calls count, including repeated points. Callback exceptions
/// propagate. Seeded replay holds within a build for deterministic callbacks.
/// @param objective Borrowed callback invoked on a complete feasible configuration.
/// @param parameters Nonempty ordered mixed domain, parents preceding children.
/// @param options Actual budget and model proposal settings.
/// @return Best observed configuration and separate objective/acquisition counts.
/// @throws std::invalid_argument Invalid domain, empty objective or active options.
/// @throws std::domain_error Objective returns a nonfinite loss.
/// @throws std::overflow_error Model query count exceeds its representable range.
/// @par Example
/// @code
/// const auto result = dm::optim::mixed_search(
///     [](const auto& x) { return std::get<double>(x[0]); },
///     {{dm::optim::ParameterKind::continuous, 1, 1}}, {10});
/// std::print("{}\n", result.value); // 1
/// @endcode
/// @par References
/// - Bergstra et al. (2011). https://proceedings.neurips.cc/paper_files/paper/2011/hash/86e8f7ab32cfd12577bc2619bc635690-Abstract.html
/// - Hutter, Hoos and Leyton-Brown (2011). https://www.cs.ubc.ca/labs/algorithms/Projects/SMAC/papers/11-LION5-SMAC.pdf
/// @see bohb_search(), iterated_racing()
MixedSearchResult mixed_search(const MixedObjective& objective, const std::vector<MixedParameter>& parameters,
                               const MixedSearchOptions& options = {});

/// @brief Initialize a scheduled trial with a resolved configuration and its local seed.
using StartMixedTrial = std::function<void(std::size_t, const MixedConfiguration&, std::uint64_t)>;

/// @brief Owned Hyperband schedule and the typed configuration for each trial ID.
/// @par Example
/// @code
/// const dm::optim::BohbResult result;
/// std::print("{}\n", result.configurations.size()); // 0
/// @endcode
/// @see bohb_search()
struct BohbResult {
  ResourceSearchResult schedule; ///< Highest-fidelity score and consumed work.
  std::vector<MixedConfiguration> configurations; ///< Configuration per initialized trial.
  std::size_t model_starts{}; ///< Trials proposed by the conditional density model.
};

/// @brief Use an active-fidelity Parzen model to start Hyperband trials.
///
/// Trains the density model at the highest resource level with enough scored
/// configurations. Other starts use the mixed-domain prior. Consumed work is
/// measured by the Hyperband scheduler's incremental resource accounting.
/// Preserves state across promotions and uses scheduled randomness from the
/// resource options. Model budget and method fields are unused; TPE
/// settings govern initialization and configuration selection.
/// @param start Borrowed initializer receiving a typed configuration and trial seed.
/// @param advance Borrowed advancement callback with old and target cumulative resources.
/// @param parameters Nonempty ordered mixed parameter space.
/// @param resource Budget, seed, bracket geometry, and trace cap; Hyperband is selected.
/// @param model Model initialization, candidate pool and smoothing controls.
/// @return Resource search and all initialized configurations.
/// @throws std::invalid_argument Invalid domain, callbacks or active controls.
/// @throws std::domain_error Advancement returns a nonfinite loss.
/// @par Example
/// @code
/// const auto run = dm::optim::bohb_search([](std::size_t, const auto&, std::uint64_t) {},
///     [](std::size_t, std::size_t, std::size_t) { return 0.0; }, {dm::optim::MixedParameter{}});
/// std::print("{}\n", run.schedule.best->loss); // 0
/// @endcode
/// @par References
/// - Falkner, Klein and Hutter (2018). https://proceedings.mlr.press/v80/falkner18a.html
/// @see resource_search(), mixed_search()
BohbResult bohb_search(const StartMixedTrial& start, const AdvanceTrial& advance,
                       const std::vector<MixedParameter>& parameters,
                       ResourceSearchOptions resource = {}, const MixedSearchOptions& model = {});

/// @brief Shared simulation budget and sampling controls for iterated racing.
/// @par Example
/// @code
/// const dm::optim::IteratedRacingOptions options{200, 7, 8, 40};
/// @endcode
/// @see iterated_racing()
struct IteratedRacingOptions {
  std::size_t evaluations{1000}; ///< Positive global replication cap.
  std::uint64_t seed{42}; ///< Seed for configurations and stage replication streams.
  std::size_t candidates{12}; ///< Configurations per race, at least two.
  std::size_t stage_budget{100}; ///< Maximum simulation calls per race, positive.
  std::size_t pilot{4}; ///< Initial complete blocks for each race, at least two.
  double alpha{0.05}; ///< Friedman screening level in (0,0.5).
};

/// @brief Race fresh configurations and perturb previous winners under one global cap.
///
/// The first race uses uniform configurations. Following races carry the
/// previous winner at index zero and mix increasingly local perturbations
/// with uniform samples. Each stage uses paired Friedman screening and a
/// fresh replication seed stream. Its selected estimate determines the
/// next carryover and the final recommendation. Work counts actual
/// simulation replications, including partially completed races.
/// @param simulation Finite-return simulator taking a configuration and replication seed.
/// @param parameters Valid nonempty ordered mixed domain.
/// @param options Global budget, per-stage budget and racing controls.
/// @return Latest selected configuration, its sample mean and actual simulation count.
/// @throws std::invalid_argument Invalid objective, domain or options.
/// @throws std::domain_error A simulation result is nonfinite.
/// @par Example
/// @code
/// const auto result = dm::optim::iterated_racing(
///     [](const auto&, std::uint64_t) { return 0.0; }, {dm::optim::MixedParameter{}}, {32});
/// std::print("{}\n", result.evaluations); // 32
/// @endcode
/// @par References
/// - Birattari et al. (2002). https://www.metaheuristics.org/downloads/BirStuPaqVar2002_gecco.pdf
/// @see simulation_search(), mixed_search()
MixedSearchResult iterated_racing(const MixedSimulation& simulation, const std::vector<MixedParameter>& parameters,
                                  const IteratedRacingOptions& options = {});

} // namespace datamunge::optim
