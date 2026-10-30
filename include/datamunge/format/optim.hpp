#pragma once

/// @file optim.hpp
/// @brief std::format / std::print support for datamunge::optim.
/// Included by <datamunge/format.hpp>; requires C++20 <format>.
///
/// Every options struct prints as one line of its fields. Optimizer classes print their
/// configured options under the optimizer's own name, e.g. "LBFGS(max_iterations=500, ...)". The abstract
/// objective-function interfaces (ArbitraryFunction, DifferentiableFunction, ...) are
/// user-subclassed callbacks and are not formatted.
///
/// @par Example
/// @code
/// std::print("{}\n", dm::optim::LBFGSOptions{});
/// // LBFGSOptions(max_iterations=500, tolerance=1e-08, history_size=10, armijo_c1=1e-04, wolfe_c2=0.9, max_line_search_trials=50)
/// @endcode

#include <datamunge/format/text_formatter.hpp>
#include <datamunge/optim/optim.hpp>

#include <format>
#include <string>
#include <string_view>

/// @brief Prints optim::ACOROptions as one line of its fields.
template <>
struct std::formatter<datamunge::optim::ACOROptions> : datamunge::format_detail::text_formatter<datamunge::optim::ACOROptions> {
  /// @brief Every option field, in declaration order.
  /// @param x The value to print.
  /// @return "ACOROptions(archive_size=..., ...)".
  static std::string text(const datamunge::optim::ACOROptions& x) {
    return std::format("ACOROptions(archive_size={}, samples_per_iteration={}, locality={}, "
                       "convergence_speed={}, max_iterations={}, tolerance={}, seed={})",
                       x.archive_size, x.samples_per_iteration, x.locality, x.convergence_speed, x.max_iterations,
                       x.tolerance, x.seed);
  }
};

/// @brief Prints optim::AdaDeltaOptions as one line of its fields.
template <>
struct std::formatter<datamunge::optim::AdaDeltaOptions> : datamunge::format_detail::text_formatter<datamunge::optim::AdaDeltaOptions> {
  /// @brief Every option field, in declaration order.
  /// @param x The value to print.
  /// @return "AdaDeltaOptions(decay_rate=..., ...)".
  static std::string text(const datamunge::optim::AdaDeltaOptions& x) {
    return std::format("AdaDeltaOptions(decay_rate={}, epsilon={}, max_iterations={}, tolerance={})",
                       x.decay_rate, x.epsilon, x.max_iterations, x.tolerance);
  }
};

/// @brief Prints optim::AdaGradOptions as one line of its fields.
template <>
struct std::formatter<datamunge::optim::AdaGradOptions> : datamunge::format_detail::text_formatter<datamunge::optim::AdaGradOptions> {
  /// @brief Every option field, in declaration order.
  /// @param x The value to print.
  /// @return "AdaGradOptions(step_size=..., ...)".
  static std::string text(const datamunge::optim::AdaGradOptions& x) {
    return std::format("AdaGradOptions(step_size={}, epsilon={}, max_iterations={}, tolerance={})",
                       x.step_size, x.epsilon, x.max_iterations, x.tolerance);
  }
};

/// @brief Prints optim::AdamOptions as one line of its fields.
template <>
struct std::formatter<datamunge::optim::AdamOptions> : datamunge::format_detail::text_formatter<datamunge::optim::AdamOptions> {
  /// @brief Every option field, in declaration order.
  /// @param x The value to print.
  /// @return "AdamOptions(step_size=..., ...)".
  static std::string text(const datamunge::optim::AdamOptions& x) {
    return std::format("AdamOptions(step_size={}, beta1={}, beta2={}, epsilon={}, max_iterations={}, "
                       "tolerance={})",
                       x.step_size, x.beta1, x.beta2, x.epsilon, x.max_iterations, x.tolerance);
  }
};

/// @brief Prints optim::AMSGradOptions as one line of its fields.
template <>
struct std::formatter<datamunge::optim::AMSGradOptions> : datamunge::format_detail::text_formatter<datamunge::optim::AMSGradOptions> {
  /// @brief Every option field, in declaration order.
  /// @param x The value to print.
  /// @return "AMSGradOptions(step_size=..., ...)".
  static std::string text(const datamunge::optim::AMSGradOptions& x) {
    return std::format("AMSGradOptions(step_size={}, beta1={}, beta2={}, epsilon={}, max_iterations={}, "
                       "tolerance={})",
                       x.step_size, x.beta1, x.beta2, x.epsilon, x.max_iterations, x.tolerance);
  }
};

/// @brief Prints optim::ArtificialBeeColonyOptions as one line of its fields.
template <>
struct std::formatter<datamunge::optim::ArtificialBeeColonyOptions> : datamunge::format_detail::text_formatter<datamunge::optim::ArtificialBeeColonyOptions> {
  /// @brief Every option field, in declaration order.
  /// @param x The value to print.
  /// @return "ArtificialBeeColonyOptions(population_size=..., ...)".
  static std::string text(const datamunge::optim::ArtificialBeeColonyOptions& x) {
    return std::format("ArtificialBeeColonyOptions(population_size={}, abandonment_limit={}, "
                       "max_generations={}, tolerance={}, seed={})",
                       x.population_size, x.abandonment_limit, x.max_generations, x.tolerance, x.seed);
  }
};

/// @brief Prints optim::AugmentedLagrangianOptions as one line of its fields.
template <>
struct std::formatter<datamunge::optim::AugmentedLagrangianOptions> : datamunge::format_detail::text_formatter<datamunge::optim::AugmentedLagrangianOptions> {
  /// @brief Every option field, in declaration order.
  /// @param x The value to print.
  /// @return "AugmentedLagrangianOptions(step_size=..., ...)".
  static std::string text(const datamunge::optim::AugmentedLagrangianOptions& x) {
    return std::format("AugmentedLagrangianOptions(step_size={}, initial_penalty={}, max_outer_iterations={}, "
                       "inner_iterations={}, tolerance={})",
                       x.step_size, x.initial_penalty, x.max_outer_iterations, x.inner_iterations, x.tolerance);
  }
};

/// @brief Prints optim::BayesianOptimizationOptions as one line of its fields.
template <>
struct std::formatter<datamunge::optim::BayesianOptimizationOptions> : datamunge::format_detail::text_formatter<datamunge::optim::BayesianOptimizationOptions> {
  /// @brief Every option field, in declaration order.
  /// @param x The value to print.
  /// @return "BayesianOptimizationOptions(initial_samples=..., ...)".
  static std::string text(const datamunge::optim::BayesianOptimizationOptions& x) {
    return std::format("BayesianOptimizationOptions(initial_samples={}, max_iterations={}, seed={}, candidates={}, "
                       "refine_iterations={})",
                       x.initial_samples, x.max_iterations, x.seed, x.candidates, x.refine_iterations);
  }
};

/// @brief Prints optim::BayesianOptimizationResult as its best point and the evaluation count.
template <>
struct std::formatter<datamunge::optim::BayesianOptimizationResult> : datamunge::format_detail::text_formatter<datamunge::optim::BayesianOptimizationResult> {
  /// @brief Best value and point, and how many evaluations were made.
  /// @param x The value to print.
  /// @return "BayesianOptimizationResult(best_value=..., best_point=[...], evaluations=...)".
  static std::string text(const datamunge::optim::BayesianOptimizationResult& x) {
    return std::format("BayesianOptimizationResult(best_value={}, best_point={}, evaluations={})", x.best_value,
                       datamunge::format_detail::preview(x.best_point), x.values.size());
  }
};

/// @brief Prints optim::SurrogatePrediction as its mean and standard deviation.
template <>
struct std::formatter<datamunge::optim::SurrogatePrediction> : datamunge::format_detail::text_formatter<datamunge::optim::SurrogatePrediction> {
  /// @brief Both fields.
  /// @param x The value to print.
  /// @return "SurrogatePrediction(mean=..., sd=...)".
  static std::string text(const datamunge::optim::SurrogatePrediction& x) {
    return std::format("SurrogatePrediction(mean={}, sd={})", x.mean, x.sd);
  }
};

/// @brief Prints an optim::Acquisition as its enumerator name.
template <>
struct std::formatter<datamunge::optim::Acquisition> : datamunge::format_detail::text_formatter<datamunge::optim::Acquisition> {
  /// @brief The enumerator name.
  /// @param x The value to print.
  /// @return The enumerator name, or `Acquisition(<int>)` for an unknown value.
  static std::string text(const datamunge::optim::Acquisition& x) {
    switch (x) {
      case datamunge::optim::Acquisition::ExpectedImprovement: return "ExpectedImprovement";
      case datamunge::optim::Acquisition::ProbabilityOfImprovement: return "ProbabilityOfImprovement";
      case datamunge::optim::Acquisition::LowerConfidenceBound: return "LowerConfidenceBound";
    }
    return std::format("Acquisition({})", static_cast<int>(x));
  }
};

/// @brief Prints optim::GaussianProcessSurrogateOptions as one line of its fields.
template <>
struct std::formatter<datamunge::optim::GaussianProcessSurrogateOptions> : datamunge::format_detail::text_formatter<datamunge::optim::GaussianProcessSurrogateOptions> {
  /// @brief Every option field, in declaration order.
  /// @param x The value to print.
  /// @return "GaussianProcessSurrogateOptions(length_scale=..., ...)".
  static std::string text(const datamunge::optim::GaussianProcessSurrogateOptions& x) {
    return std::format("GaussianProcessSurrogateOptions(length_scale={}, signal_variance={}, noise={}, normalize_values={}, "
                       "fit_hyperparameters={}, fit_noise={}, min_length_scale={}, max_length_scale={}, acquisition={}, "
                       "exploration={}, kappa={})",
                       x.length_scale, x.signal_variance, x.noise, x.normalize_values, x.fit_hyperparameters, x.fit_noise,
                       x.min_length_scale, x.max_length_scale, x.acquisition, x.exploration, x.kappa);
  }
};

/// @brief Prints optim::CMAESOptions as one line of its fields.
template <>
struct std::formatter<datamunge::optim::CMAESOptions> : datamunge::format_detail::text_formatter<datamunge::optim::CMAESOptions> {
  /// @brief Every option field, in declaration order.
  /// @param x The value to print.
  /// @return "CMAESOptions(population_size=..., ...)".
  static std::string text(const datamunge::optim::CMAESOptions& x) {
    return std::format("CMAESOptions(population_size={}, initial_step_size={}, max_generations={}, "
                       "tolerance={}, seed={})",
                       x.population_size, x.initial_step_size, x.max_generations, x.tolerance, x.seed);
  }
};

/// @brief Prints optim::ConjugateGradientOptions as one line of its fields.
template <>
struct std::formatter<datamunge::optim::ConjugateGradientOptions> : datamunge::format_detail::text_formatter<datamunge::optim::ConjugateGradientOptions> {
  /// @brief Every option field, in declaration order.
  /// @param x The value to print.
  /// @return "ConjugateGradientOptions(max_iterations=..., ...)".
  static std::string text(const datamunge::optim::ConjugateGradientOptions& x) {
    return std::format("ConjugateGradientOptions(max_iterations={}, tolerance={}, armijo_c1={}, "
                       "backtracking_factor={}, max_line_search_trials={})",
                       x.max_iterations, x.tolerance, x.armijo_c1, x.backtracking_factor,
                       x.max_line_search_trials);
  }
};

/// @brief Prints optim::CoordinateDescentOptions as one line of its fields.
template <>
struct std::formatter<datamunge::optim::CoordinateDescentOptions> : datamunge::format_detail::text_formatter<datamunge::optim::CoordinateDescentOptions> {
  /// @brief Every option field, in declaration order.
  /// @param x The value to print.
  /// @return "CoordinateDescentOptions(step_size=..., ...)".
  static std::string text(const datamunge::optim::CoordinateDescentOptions& x) {
    return std::format("CoordinateDescentOptions(step_size={}, max_iterations={}, tolerance={}, armijo_c1={}, "
                       "backtracking_factor={}, max_line_search_trials={})",
                       x.step_size, x.max_iterations, x.tolerance, x.armijo_c1, x.backtracking_factor,
                       x.max_line_search_trials);
  }
};

/// @brief Prints optim::CrossEntropyMethodOptions as one line of its fields.
template <>
struct std::formatter<datamunge::optim::CrossEntropyMethodOptions> : datamunge::format_detail::text_formatter<datamunge::optim::CrossEntropyMethodOptions> {
  /// @brief Every option field, in declaration order.
  /// @param x The value to print.
  /// @return "CrossEntropyMethodOptions(population_size=..., ...)".
  static std::string text(const datamunge::optim::CrossEntropyMethodOptions& x) {
    return std::format("CrossEntropyMethodOptions(population_size={}, elite_ratio={}, initial_std_dev={}, "
                       "smoothing={}, max_iterations={}, tolerance={}, seed={})",
                       x.population_size, x.elite_ratio, x.initial_std_dev, x.smoothing, x.max_iterations,
                       x.tolerance, x.seed);
  }
};

/// @brief Prints optim::CuckooSearchOptions as one line of its fields.
template <>
struct std::formatter<datamunge::optim::CuckooSearchOptions> : datamunge::format_detail::text_formatter<datamunge::optim::CuckooSearchOptions> {
  /// @brief Every option field, in declaration order.
  /// @param x The value to print.
  /// @return "CuckooSearchOptions(population_size=..., ...)".
  static std::string text(const datamunge::optim::CuckooSearchOptions& x) {
    return std::format("CuckooSearchOptions(population_size={}, discovery_rate={}, levy_beta={}, step_scale={}, "
                       "max_iterations={}, tolerance={}, seed={})",
                       x.population_size, x.discovery_rate, x.levy_beta, x.step_scale, x.max_iterations,
                       x.tolerance, x.seed);
  }
};

/// @brief Prints optim::DEOptions as one line of its fields.
template <>
struct std::formatter<datamunge::optim::DEOptions> : datamunge::format_detail::text_formatter<datamunge::optim::DEOptions> {
  /// @brief Every option field, in declaration order.
  /// @param x The value to print.
  /// @return "DEOptions(population_size=..., ...)".
  static std::string text(const datamunge::optim::DEOptions& x) {
    return std::format("DEOptions(population_size={}, max_generations={}, differential_weight={}, "
                       "crossover_rate={}, mutation_strategy={}, crossover_strategy={}, tolerance={}, seed={})",
                       x.population_size, x.max_generations, x.differential_weight, x.crossover_rate,
                       x.mutation_strategy, x.crossover_strategy, x.tolerance, x.seed);
  }
};

/// @brief Prints optim::EstimationOfDistributionOptions as one line of its fields.
template <>
struct std::formatter<datamunge::optim::EstimationOfDistributionOptions> : datamunge::format_detail::text_formatter<datamunge::optim::EstimationOfDistributionOptions> {
  /// @brief Every option field, in declaration order.
  /// @param x The value to print.
  /// @return "EstimationOfDistributionOptions(population_size=..., ...)".
  static std::string text(const datamunge::optim::EstimationOfDistributionOptions& x) {
    return std::format("EstimationOfDistributionOptions(population_size={}, selection_ratio={}, "
                       "initial_std_dev={}, covariance_regularization={}, max_generations={}, tolerance={}, "
                       "seed={})",
                       x.population_size, x.selection_ratio, x.initial_std_dev, x.covariance_regularization,
                       x.max_generations, x.tolerance, x.seed);
  }
};

/// @brief Prints optim::EvolutionStrategyOptions as one line of its fields.
template <>
struct std::formatter<datamunge::optim::EvolutionStrategyOptions> : datamunge::format_detail::text_formatter<datamunge::optim::EvolutionStrategyOptions> {
  /// @brief Every option field, in declaration order.
  /// @param x The value to print.
  /// @return "EvolutionStrategyOptions(mu=..., ...)".
  static std::string text(const datamunge::optim::EvolutionStrategyOptions& x) {
    return std::format("EvolutionStrategyOptions(mu={}, offspring_size={}, strategy={}, initial_step_size={}, "
                       "max_generations={}, tolerance={}, seed={})",
                       x.mu, x.offspring_size, x.strategy, x.initial_step_size, x.max_generations, x.tolerance,
                       x.seed);
  }
};

/// @brief Prints optim::FireflyAlgorithmOptions as one line of its fields.
template <>
struct std::formatter<datamunge::optim::FireflyAlgorithmOptions> : datamunge::format_detail::text_formatter<datamunge::optim::FireflyAlgorithmOptions> {
  /// @brief Every option field, in declaration order.
  /// @param x The value to print.
  /// @return "FireflyAlgorithmOptions(population_size=..., ...)".
  static std::string text(const datamunge::optim::FireflyAlgorithmOptions& x) {
    return std::format("FireflyAlgorithmOptions(population_size={}, attractiveness_at_zero={}, "
                       "light_absorption={}, randomization_step={}, randomization_decay={}, max_iterations={}, "
                       "tolerance={}, seed={})",
                       x.population_size, x.attractiveness_at_zero, x.light_absorption, x.randomization_step,
                       x.randomization_decay, x.max_iterations, x.tolerance, x.seed);
  }
};

/// @brief Prints optim::FISTAOptions as one line of its fields.
template <>
struct std::formatter<datamunge::optim::FISTAOptions> : datamunge::format_detail::text_formatter<datamunge::optim::FISTAOptions> {
  /// @brief Every option field, in declaration order.
  /// @param x The value to print.
  /// @return "FISTAOptions(step_size=..., ...)".
  static std::string text(const datamunge::optim::FISTAOptions& x) {
    return std::format("FISTAOptions(step_size={}, max_iterations={}, tolerance={})",
                       x.step_size, x.max_iterations, x.tolerance);
  }
};

/// @brief Prints optim::GAOptions as one line of its fields.
template <>
struct std::formatter<datamunge::optim::GAOptions> : datamunge::format_detail::text_formatter<datamunge::optim::GAOptions> {
  /// @brief Every option field, in declaration order.
  /// @param x The value to print.
  /// @return "GAOptions(population_size=..., ...)".
  static std::string text(const datamunge::optim::GAOptions& x) {
    return std::format("GAOptions(population_size={}, max_generations={}, crossover_rate={}, mutation_rate={}, "
                       "mutation_std_dev={}, selection_strategy={}, tournament_size={}, crossover_strategy={}, "
                       "blend_alpha={}, elitism={}, elite_count={}, tolerance={}, seed={})",
                       x.population_size, x.max_generations, x.crossover_rate, x.mutation_rate,
                       x.mutation_std_dev, x.selection_strategy, x.tournament_size, x.crossover_strategy,
                       x.blend_alpha, x.elitism, x.elite_count, x.tolerance, x.seed);
  }
};

/// @brief Prints optim::GradientDescentOptions as one line of its fields.
template <>
struct std::formatter<datamunge::optim::GradientDescentOptions> : datamunge::format_detail::text_formatter<datamunge::optim::GradientDescentOptions> {
  /// @brief Every option field, in declaration order.
  /// @param x The value to print.
  /// @return "GradientDescentOptions(step_size=..., ...)".
  static std::string text(const datamunge::optim::GradientDescentOptions& x) {
    return std::format("GradientDescentOptions(step_size={}, momentum={}, max_iterations={}, tolerance={})",
                       x.step_size, x.momentum, x.max_iterations, x.tolerance);
  }
};

/// @brief Prints optim::GreyWolfOptimizerOptions as one line of its fields.
template <>
struct std::formatter<datamunge::optim::GreyWolfOptimizerOptions> : datamunge::format_detail::text_formatter<datamunge::optim::GreyWolfOptimizerOptions> {
  /// @brief Every option field, in declaration order.
  /// @param x The value to print.
  /// @return "GreyWolfOptimizerOptions(population_size=..., ...)".
  static std::string text(const datamunge::optim::GreyWolfOptimizerOptions& x) {
    return std::format("GreyWolfOptimizerOptions(population_size={}, max_iterations={}, tolerance={}, seed={})",
                       x.population_size, x.max_iterations, x.tolerance, x.seed);
  }
};

/// @brief Prints optim::HarmonySearchOptions as one line of its fields.
template <>
struct std::formatter<datamunge::optim::HarmonySearchOptions> : datamunge::format_detail::text_formatter<datamunge::optim::HarmonySearchOptions> {
  /// @brief Every option field, in declaration order.
  /// @param x The value to print.
  /// @return "HarmonySearchOptions(population_size=..., ...)".
  static std::string text(const datamunge::optim::HarmonySearchOptions& x) {
    return std::format("HarmonySearchOptions(population_size={}, memory_consideration_rate={}, "
                       "pitch_adjustment_rate={}, bandwidth_fraction={}, max_iterations={}, tolerance={}, "
                       "seed={})",
                       x.population_size, x.memory_consideration_rate, x.pitch_adjustment_rate,
                       x.bandwidth_fraction, x.max_iterations, x.tolerance, x.seed);
  }
};

/// @brief Prints optim::InteriorPointOptions as one line of its fields.
template <>
struct std::formatter<datamunge::optim::InteriorPointOptions> : datamunge::format_detail::text_formatter<datamunge::optim::InteriorPointOptions> {
  /// @brief Every option field, in declaration order.
  /// @param x The value to print.
  /// @return "InteriorPointOptions(step_size=..., ...)".
  static std::string text(const datamunge::optim::InteriorPointOptions& x) {
    return std::format("InteriorPointOptions(step_size={}, initial_barrier={}, barrier_decay={}, "
                       "max_outer_iterations={}, inner_iterations={})",
                       x.step_size, x.initial_barrier, x.barrier_decay, x.max_outer_iterations,
                       x.inner_iterations);
  }
};

/// @brief Prints optim::LBFGSOptions as one line of its fields.
template <>
struct std::formatter<datamunge::optim::LBFGSOptions> : datamunge::format_detail::text_formatter<datamunge::optim::LBFGSOptions> {
  /// @brief Every option field, in declaration order.
  /// @param x The value to print.
  /// @return "LBFGSOptions(max_iterations=..., ...)".
  static std::string text(const datamunge::optim::LBFGSOptions& x) {
    return std::format("LBFGSOptions(max_iterations={}, tolerance={}, history_size={}, armijo_c1={}, "
                       "wolfe_c2={}, max_line_search_trials={})",
                       x.max_iterations, x.tolerance, x.history_size, x.armijo_c1, x.wolfe_c2,
                       x.max_line_search_trials);
  }
};

/// @brief Prints optim::LevenbergMarquardtOptions as one line of its fields.
template <>
struct std::formatter<datamunge::optim::LevenbergMarquardtOptions> : datamunge::format_detail::text_formatter<datamunge::optim::LevenbergMarquardtOptions> {
  /// @brief Every option field, in declaration order.
  /// @param x The value to print.
  /// @return "LevenbergMarquardtOptions(initial_damping=..., ...)".
  static std::string text(const datamunge::optim::LevenbergMarquardtOptions& x) {
    return std::format("LevenbergMarquardtOptions(initial_damping={}, max_iterations={}, tolerance={})",
                       x.initial_damping, x.max_iterations, x.tolerance);
  }
};

/// @brief Prints optim::MOEADOptions as one line of its fields.
template <>
struct std::formatter<datamunge::optim::MOEADOptions> : datamunge::format_detail::text_formatter<datamunge::optim::MOEADOptions> {
  /// @brief Every option field, in declaration order.
  /// @param x The value to print.
  /// @return "MOEADOptions(population_size=..., ...)".
  static std::string text(const datamunge::optim::MOEADOptions& x) {
    return std::format("MOEADOptions(population_size={}, max_generations={}, neighborhood_size={}, "
                       "decomposition={}, crossover_rate={}, eta_crossover={}, mutation_rate={}, "
                       "eta_mutation={}, seed={})",
                       x.population_size, x.max_generations, x.neighborhood_size, x.decomposition,
                       x.crossover_rate, x.eta_crossover, x.mutation_rate, x.eta_mutation, x.seed);
  }
};

/// @brief Prints an optim::ParetoPoint as its coordinates and objective values.
template <>
struct std::formatter<datamunge::optim::ParetoPoint> : datamunge::format_detail::text_formatter<datamunge::optim::ParetoPoint> {
  /// @brief The decision vector and its objective vector.
  /// @param x The value to print.
  /// @return "ParetoPoint(coordinates=[...], objectives=[...])".
  static std::string text(const datamunge::optim::ParetoPoint& x) {
    return std::format("ParetoPoint(coordinates={}, objectives={})",
                       datamunge::format_detail::preview(x.coordinates),
                       datamunge::format_detail::preview(x.objectives));
  }
};

/// @brief Prints optim::NadamOptions as one line of its fields.
template <>
struct std::formatter<datamunge::optim::NadamOptions> : datamunge::format_detail::text_formatter<datamunge::optim::NadamOptions> {
  /// @brief Every option field, in declaration order.
  /// @param x The value to print.
  /// @return "NadamOptions(step_size=..., ...)".
  static std::string text(const datamunge::optim::NadamOptions& x) {
    return std::format("NadamOptions(step_size={}, beta1={}, beta2={}, epsilon={}, max_iterations={}, "
                       "tolerance={})",
                       x.step_size, x.beta1, x.beta2, x.epsilon, x.max_iterations, x.tolerance);
  }
};

/// @brief Prints optim::NelderMeadOptions as one line of its fields.
template <>
struct std::formatter<datamunge::optim::NelderMeadOptions> : datamunge::format_detail::text_formatter<datamunge::optim::NelderMeadOptions> {
  /// @brief Every option field, in declaration order.
  /// @param x The value to print.
  /// @return "NelderMeadOptions(initial_simplex_scale=..., ...)".
  static std::string text(const datamunge::optim::NelderMeadOptions& x) {
    return std::format("NelderMeadOptions(initial_simplex_scale={}, reflection={}, expansion={}, "
                       "contraction={}, shrink={}, max_iterations={}, tolerance={})",
                       x.initial_simplex_scale, x.reflection, x.expansion, x.contraction, x.shrink,
                       x.max_iterations, x.tolerance);
  }
};

/// @brief Prints optim::NesterovAcceleratedGradientOptions as one line of its fields.
template <>
struct std::formatter<datamunge::optim::NesterovAcceleratedGradientOptions> : datamunge::format_detail::text_formatter<datamunge::optim::NesterovAcceleratedGradientOptions> {
  /// @brief Every option field, in declaration order.
  /// @param x The value to print.
  /// @return "NesterovAcceleratedGradientOptions(step_size=..., ...)".
  static std::string text(const datamunge::optim::NesterovAcceleratedGradientOptions& x) {
    return std::format("NesterovAcceleratedGradientOptions(step_size={}, momentum={}, max_iterations={}, "
                       "tolerance={})",
                       x.step_size, x.momentum, x.max_iterations, x.tolerance);
  }
};

/// @brief Prints optim::NewtonOptions as one line of its fields.
template <>
struct std::formatter<datamunge::optim::NewtonOptions> : datamunge::format_detail::text_formatter<datamunge::optim::NewtonOptions> {
  /// @brief Every option field, in declaration order.
  /// @param x The value to print.
  /// @return "NewtonOptions(max_iterations=..., ...)".
  static std::string text(const datamunge::optim::NewtonOptions& x) {
    return std::format("NewtonOptions(max_iterations={}, tolerance={}, damping={})",
                       x.max_iterations, x.tolerance, x.damping);
  }
};

/// @brief Prints optim::NSGA2Options as one line of its fields.
template <>
struct std::formatter<datamunge::optim::NSGA2Options> : datamunge::format_detail::text_formatter<datamunge::optim::NSGA2Options> {
  /// @brief Every option field, in declaration order.
  /// @param x The value to print.
  /// @return "NSGA2Options(population_size=..., ...)".
  static std::string text(const datamunge::optim::NSGA2Options& x) {
    return std::format("NSGA2Options(population_size={}, max_generations={}, crossover_rate={}, "
                       "eta_crossover={}, mutation_rate={}, eta_mutation={}, seed={})",
                       x.population_size, x.max_generations, x.crossover_rate, x.eta_crossover, x.mutation_rate,
                       x.eta_mutation, x.seed);
  }
};

/// @brief Prints optim::ParallelTemperingOptions as one line of its fields.
template <>
struct std::formatter<datamunge::optim::ParallelTemperingOptions> : datamunge::format_detail::text_formatter<datamunge::optim::ParallelTemperingOptions> {
  /// @brief Every option field, in declaration order.
  /// @param x The value to print.
  /// @return "ParallelTemperingOptions(num_replicas=..., ...)".
  static std::string text(const datamunge::optim::ParallelTemperingOptions& x) {
    return std::format("ParallelTemperingOptions(num_replicas={}, initial_temperature={}, final_temperature={}, "
                       "step_std_dev={}, swap_interval={}, max_sweeps={}, seed={})",
                       x.num_replicas, x.initial_temperature, x.final_temperature, x.step_std_dev,
                       x.swap_interval, x.max_sweeps, x.seed);
  }
};

/// @brief Prints optim::ProximalGradientOptions as one line of its fields.
template <>
struct std::formatter<datamunge::optim::ProximalGradientOptions> : datamunge::format_detail::text_formatter<datamunge::optim::ProximalGradientOptions> {
  /// @brief Every option field, in declaration order.
  /// @param x The value to print.
  /// @return "ProximalGradientOptions(step_size=..., ...)".
  static std::string text(const datamunge::optim::ProximalGradientOptions& x) {
    return std::format("ProximalGradientOptions(step_size={}, max_iterations={}, tolerance={})",
                       x.step_size, x.max_iterations, x.tolerance);
  }
};

/// @brief Prints optim::PSOOptions as one line of its fields.
template <>
struct std::formatter<datamunge::optim::PSOOptions> : datamunge::format_detail::text_formatter<datamunge::optim::PSOOptions> {
  /// @brief Every option field, in declaration order.
  /// @param x The value to print.
  /// @return "PSOOptions(population_size=..., ...)".
  static std::string text(const datamunge::optim::PSOOptions& x) {
    return std::format("PSOOptions(population_size={}, max_iterations={}, inertia_weight={}, "
                       "cognitive_coefficient={}, social_coefficient={}, topology={}, ring_neighbors={}, "
                       "inertia_strategy={}, final_inertia_weight={}, tolerance={}, seed={})",
                       x.population_size, x.max_iterations, x.inertia_weight, x.cognitive_coefficient,
                       x.social_coefficient, x.topology, x.ring_neighbors, x.inertia_strategy,
                       x.final_inertia_weight, x.tolerance, x.seed);
  }
};

/// @brief Prints optim::RandomizedBlockCoordinateDescentOptions as one line of its fields.
template <>
struct std::formatter<datamunge::optim::RandomizedBlockCoordinateDescentOptions> : datamunge::format_detail::text_formatter<datamunge::optim::RandomizedBlockCoordinateDescentOptions> {
  /// @brief Every option field, in declaration order.
  /// @param x The value to print.
  /// @return "RandomizedBlockCoordinateDescentOptions(block_size=..., ...)".
  static std::string text(const datamunge::optim::RandomizedBlockCoordinateDescentOptions& x) {
    return std::format("RandomizedBlockCoordinateDescentOptions(block_size={}, step_size={}, max_iterations={}, "
                       "tolerance={}, armijo_c1={}, backtracking_factor={}, max_line_search_trials={}, seed={})",
                       x.block_size, x.step_size, x.max_iterations, x.tolerance, x.armijo_c1,
                       x.backtracking_factor, x.max_line_search_trials, x.seed);
  }
};

/// @brief Prints optim::RMSPropOptions as one line of its fields.
template <>
struct std::formatter<datamunge::optim::RMSPropOptions> : datamunge::format_detail::text_formatter<datamunge::optim::RMSPropOptions> {
  /// @brief Every option field, in declaration order.
  /// @param x The value to print.
  /// @return "RMSPropOptions(step_size=..., ...)".
  static std::string text(const datamunge::optim::RMSPropOptions& x) {
    return std::format("RMSPropOptions(step_size={}, decay_rate={}, epsilon={}, max_iterations={}, tolerance={})",
                       x.step_size, x.decay_rate, x.epsilon, x.max_iterations, x.tolerance);
  }
};

/// @brief Prints optim::SAGAOptions as one line of its fields.
template <>
struct std::formatter<datamunge::optim::SAGAOptions> : datamunge::format_detail::text_formatter<datamunge::optim::SAGAOptions> {
  /// @brief Every option field, in declaration order.
  /// @param x The value to print.
  /// @return "SAGAOptions(step_size=..., ...)".
  static std::string text(const datamunge::optim::SAGAOptions& x) {
    return std::format("SAGAOptions(step_size={}, max_iterations={}, tolerance={}, seed={})",
                       x.step_size, x.max_iterations, x.tolerance, x.seed);
  }
};

/// @brief Prints optim::SGDOptions as one line of its fields.
template <>
struct std::formatter<datamunge::optim::SGDOptions> : datamunge::format_detail::text_formatter<datamunge::optim::SGDOptions> {
  /// @brief Every option field, in declaration order.
  /// @param x The value to print.
  /// @return "SGDOptions(step_size=..., ...)".
  static std::string text(const datamunge::optim::SGDOptions& x) {
    return std::format("SGDOptions(step_size={}, max_epochs={}, batch_size={}, tolerance={}, shuffle={}, "
                       "seed={})",
                       x.step_size, x.max_epochs, x.batch_size, x.tolerance, x.shuffle, x.seed);
  }
};

/// @brief Prints optim::SimulatedAnnealingOptions as one line of its fields.
template <>
struct std::formatter<datamunge::optim::SimulatedAnnealingOptions> : datamunge::format_detail::text_formatter<datamunge::optim::SimulatedAnnealingOptions> {
  /// @brief Every option field, in declaration order.
  /// @param x The value to print.
  /// @return "SimulatedAnnealingOptions(initial_temperature=..., ...)".
  static std::string text(const datamunge::optim::SimulatedAnnealingOptions& x) {
    return std::format("SimulatedAnnealingOptions(initial_temperature={}, cooling_rate={}, max_iterations={}, "
                       "step_std_dev={}, seed={})",
                       x.initial_temperature, x.cooling_rate, x.max_iterations, x.step_std_dev, x.seed);
  }
};

/// @brief Prints optim::SPEA2Options as one line of its fields.
template <>
struct std::formatter<datamunge::optim::SPEA2Options> : datamunge::format_detail::text_formatter<datamunge::optim::SPEA2Options> {
  /// @brief Every option field, in declaration order.
  /// @param x The value to print.
  /// @return "SPEA2Options(population_size=..., ...)".
  static std::string text(const datamunge::optim::SPEA2Options& x) {
    return std::format("SPEA2Options(population_size={}, archive_size={}, max_generations={}, "
                       "crossover_rate={}, eta_crossover={}, mutation_rate={}, eta_mutation={}, seed={})",
                       x.population_size, x.archive_size, x.max_generations, x.crossover_rate, x.eta_crossover,
                       x.mutation_rate, x.eta_mutation, x.seed);
  }
};

/// @brief Prints optim::SQPOptions as one line of its fields.
template <>
struct std::formatter<datamunge::optim::SQPOptions> : datamunge::format_detail::text_formatter<datamunge::optim::SQPOptions> {
  /// @brief Every option field, in declaration order.
  /// @param x The value to print.
  /// @return "SQPOptions(step_size=..., ...)".
  static std::string text(const datamunge::optim::SQPOptions& x) {
    return std::format("SQPOptions(step_size={}, regularization={}, max_iterations={}, tolerance={})",
                       x.step_size, x.regularization, x.max_iterations, x.tolerance);
  }
};

/// @brief Prints optim::SVRGOptions as one line of its fields.
template <>
struct std::formatter<datamunge::optim::SVRGOptions> : datamunge::format_detail::text_formatter<datamunge::optim::SVRGOptions> {
  /// @brief Every option field, in declaration order.
  /// @param x The value to print.
  /// @return "SVRGOptions(step_size=..., ...)".
  static std::string text(const datamunge::optim::SVRGOptions& x) {
    return std::format("SVRGOptions(step_size={}, max_epochs={}, inner_iterations={}, tolerance={}, seed={})",
                       x.step_size, x.max_epochs, x.inner_iterations, x.tolerance, x.seed);
  }
};

/// @brief Prints optim::TrustRegionNewtonOptions as one line of its fields.
template <>
struct std::formatter<datamunge::optim::TrustRegionNewtonOptions> : datamunge::format_detail::text_formatter<datamunge::optim::TrustRegionNewtonOptions> {
  /// @brief Every option field, in declaration order.
  /// @param x The value to print.
  /// @return "TrustRegionNewtonOptions(initial_radius=..., ...)".
  static std::string text(const datamunge::optim::TrustRegionNewtonOptions& x) {
    return std::format("TrustRegionNewtonOptions(initial_radius={}, max_radius={}, max_iterations={}, "
                       "tolerance={})",
                       x.initial_radius, x.max_radius, x.max_iterations, x.tolerance);
  }
};

/// @brief Prints optim::WhaleOptimizationOptions as one line of its fields.
template <>
struct std::formatter<datamunge::optim::WhaleOptimizationOptions> : datamunge::format_detail::text_formatter<datamunge::optim::WhaleOptimizationOptions> {
  /// @brief Every option field, in declaration order.
  /// @param x The value to print.
  /// @return "WhaleOptimizationOptions(population_size=..., ...)".
  static std::string text(const datamunge::optim::WhaleOptimizationOptions& x) {
    return std::format("WhaleOptimizationOptions(population_size={}, spiral_constant={}, max_iterations={}, "
                       "tolerance={}, seed={})",
                       x.population_size, x.spiral_constant, x.max_iterations, x.tolerance, x.seed);
  }
};

namespace datamunge::format_detail {

/// @brief Renders an optimizer as its options' text with the options type name swapped for
///        the optimizer's, e.g. "LBFGSOptions(max_iterations=500, ...)" -> "LBFGS(max_iterations=500, ...)".
/// @param name The optimizer's type name.
/// @param options The optimizer's stored options (must be formattable).
/// @return "name(field=value, ...)".
template <typename Options>
std::string optim_with_options(std::string_view name, const Options& options) {
  std::string body = std::format("{}", options);
  const auto open = body.find('(');
  return std::string(name) + (open == std::string::npos ? std::string("()") : body.substr(open));
}

} // namespace datamunge::format_detail

/// @brief Prints an optim::ACOR as "ACOR(...)" with its configured options.
template <>
struct std::formatter<datamunge::optim::ACOR> : datamunge::format_detail::text_formatter<datamunge::optim::ACOR> {
  /// @brief The type name followed by every option field.
  /// @param x The value to print.
  /// @return "ACOR(field=..., ...)".
  static std::string text(const datamunge::optim::ACOR& x) {
    return datamunge::format_detail::optim_with_options("ACOR", x.options());
  }
};

/// @brief Prints an optim::AdaDelta as "AdaDelta(...)" with its configured options.
template <>
struct std::formatter<datamunge::optim::AdaDelta> : datamunge::format_detail::text_formatter<datamunge::optim::AdaDelta> {
  /// @brief The type name followed by every option field.
  /// @param x The value to print.
  /// @return "AdaDelta(field=..., ...)".
  static std::string text(const datamunge::optim::AdaDelta& x) {
    return datamunge::format_detail::optim_with_options("AdaDelta", x.options());
  }
};

/// @brief Prints an optim::AdaGrad as "AdaGrad(...)" with its configured options.
template <>
struct std::formatter<datamunge::optim::AdaGrad> : datamunge::format_detail::text_formatter<datamunge::optim::AdaGrad> {
  /// @brief The type name followed by every option field.
  /// @param x The value to print.
  /// @return "AdaGrad(field=..., ...)".
  static std::string text(const datamunge::optim::AdaGrad& x) {
    return datamunge::format_detail::optim_with_options("AdaGrad", x.options());
  }
};

/// @brief Prints an optim::Adam as "Adam(...)" with its configured options.
template <>
struct std::formatter<datamunge::optim::Adam> : datamunge::format_detail::text_formatter<datamunge::optim::Adam> {
  /// @brief The type name followed by every option field.
  /// @param x The value to print.
  /// @return "Adam(field=..., ...)".
  static std::string text(const datamunge::optim::Adam& x) {
    return datamunge::format_detail::optim_with_options("Adam", x.options());
  }
};

/// @brief Prints an optim::AMSGrad as "AMSGrad(...)" with its configured options.
template <>
struct std::formatter<datamunge::optim::AMSGrad> : datamunge::format_detail::text_formatter<datamunge::optim::AMSGrad> {
  /// @brief The type name followed by every option field.
  /// @param x The value to print.
  /// @return "AMSGrad(field=..., ...)".
  static std::string text(const datamunge::optim::AMSGrad& x) {
    return datamunge::format_detail::optim_with_options("AMSGrad", x.options());
  }
};

/// @brief Prints an optim::ArtificialBeeColony as "ArtificialBeeColony(...)" with its configured options.
template <>
struct std::formatter<datamunge::optim::ArtificialBeeColony> : datamunge::format_detail::text_formatter<datamunge::optim::ArtificialBeeColony> {
  /// @brief The type name followed by every option field.
  /// @param x The value to print.
  /// @return "ArtificialBeeColony(field=..., ...)".
  static std::string text(const datamunge::optim::ArtificialBeeColony& x) {
    return datamunge::format_detail::optim_with_options("ArtificialBeeColony", x.options());
  }
};

/// @brief Prints an optim::AugmentedLagrangian as "AugmentedLagrangian(...)" with its configured options.
template <>
struct std::formatter<datamunge::optim::AugmentedLagrangian> : datamunge::format_detail::text_formatter<datamunge::optim::AugmentedLagrangian> {
  /// @brief The type name followed by every option field.
  /// @param x The value to print.
  /// @return "AugmentedLagrangian(field=..., ...)".
  static std::string text(const datamunge::optim::AugmentedLagrangian& x) {
    return datamunge::format_detail::optim_with_options("AugmentedLagrangian", x.options());
  }
};

/// @brief Prints an optim::BayesianOptimization as "BayesianOptimization(...)" with its configured options.
template <>
struct std::formatter<datamunge::optim::BayesianOptimization> : datamunge::format_detail::text_formatter<datamunge::optim::BayesianOptimization> {
  /// @brief The type name followed by every option field.
  /// @param x The value to print.
  /// @return "BayesianOptimization(field=..., ...)".
  static std::string text(const datamunge::optim::BayesianOptimization& x) {
    return datamunge::format_detail::optim_with_options("BayesianOptimization", x.options());
  }
};

/// @brief Prints an optim::CMAES as "CMAES(...)" with its configured options.
template <>
struct std::formatter<datamunge::optim::CMAES> : datamunge::format_detail::text_formatter<datamunge::optim::CMAES> {
  /// @brief The type name followed by every option field.
  /// @param x The value to print.
  /// @return "CMAES(field=..., ...)".
  static std::string text(const datamunge::optim::CMAES& x) {
    return datamunge::format_detail::optim_with_options("CMAES", x.options());
  }
};

/// @brief Prints an optim::ConjugateGradient as "ConjugateGradient(...)" with its configured options.
template <>
struct std::formatter<datamunge::optim::ConjugateGradient> : datamunge::format_detail::text_formatter<datamunge::optim::ConjugateGradient> {
  /// @brief The type name followed by every option field.
  /// @param x The value to print.
  /// @return "ConjugateGradient(field=..., ...)".
  static std::string text(const datamunge::optim::ConjugateGradient& x) {
    return datamunge::format_detail::optim_with_options("ConjugateGradient", x.options());
  }
};

/// @brief Prints an optim::CoordinateDescent as "CoordinateDescent(...)" with its configured options.
template <>
struct std::formatter<datamunge::optim::CoordinateDescent> : datamunge::format_detail::text_formatter<datamunge::optim::CoordinateDescent> {
  /// @brief The type name followed by every option field.
  /// @param x The value to print.
  /// @return "CoordinateDescent(field=..., ...)".
  static std::string text(const datamunge::optim::CoordinateDescent& x) {
    return datamunge::format_detail::optim_with_options("CoordinateDescent", x.options());
  }
};

/// @brief Prints an optim::CrossEntropyMethod as "CrossEntropyMethod(...)" with its configured options.
template <>
struct std::formatter<datamunge::optim::CrossEntropyMethod> : datamunge::format_detail::text_formatter<datamunge::optim::CrossEntropyMethod> {
  /// @brief The type name followed by every option field.
  /// @param x The value to print.
  /// @return "CrossEntropyMethod(field=..., ...)".
  static std::string text(const datamunge::optim::CrossEntropyMethod& x) {
    return datamunge::format_detail::optim_with_options("CrossEntropyMethod", x.options());
  }
};

/// @brief Prints an optim::CuckooSearch as "CuckooSearch(...)" with its configured options.
template <>
struct std::formatter<datamunge::optim::CuckooSearch> : datamunge::format_detail::text_formatter<datamunge::optim::CuckooSearch> {
  /// @brief The type name followed by every option field.
  /// @param x The value to print.
  /// @return "CuckooSearch(field=..., ...)".
  static std::string text(const datamunge::optim::CuckooSearch& x) {
    return datamunge::format_detail::optim_with_options("CuckooSearch", x.options());
  }
};

/// @brief Prints an optim::DifferentialEvolution as "DifferentialEvolution(...)" with its configured options.
template <>
struct std::formatter<datamunge::optim::DifferentialEvolution> : datamunge::format_detail::text_formatter<datamunge::optim::DifferentialEvolution> {
  /// @brief The type name followed by every option field.
  /// @param x The value to print.
  /// @return "DifferentialEvolution(field=..., ...)".
  static std::string text(const datamunge::optim::DifferentialEvolution& x) {
    return datamunge::format_detail::optim_with_options("DifferentialEvolution", x.options());
  }
};

/// @brief Prints an optim::EstimationOfDistribution as "EstimationOfDistribution(...)" with its configured options.
template <>
struct std::formatter<datamunge::optim::EstimationOfDistribution> : datamunge::format_detail::text_formatter<datamunge::optim::EstimationOfDistribution> {
  /// @brief The type name followed by every option field.
  /// @param x The value to print.
  /// @return "EstimationOfDistribution(field=..., ...)".
  static std::string text(const datamunge::optim::EstimationOfDistribution& x) {
    return datamunge::format_detail::optim_with_options("EstimationOfDistribution", x.options());
  }
};

/// @brief Prints an optim::EvolutionStrategy as "EvolutionStrategy(...)" with its configured options.
template <>
struct std::formatter<datamunge::optim::EvolutionStrategy> : datamunge::format_detail::text_formatter<datamunge::optim::EvolutionStrategy> {
  /// @brief The type name followed by every option field.
  /// @param x The value to print.
  /// @return "EvolutionStrategy(field=..., ...)".
  static std::string text(const datamunge::optim::EvolutionStrategy& x) {
    return datamunge::format_detail::optim_with_options("EvolutionStrategy", x.options());
  }
};

/// @brief Prints an optim::FireflyAlgorithm as "FireflyAlgorithm(...)" with its configured options.
template <>
struct std::formatter<datamunge::optim::FireflyAlgorithm> : datamunge::format_detail::text_formatter<datamunge::optim::FireflyAlgorithm> {
  /// @brief The type name followed by every option field.
  /// @param x The value to print.
  /// @return "FireflyAlgorithm(field=..., ...)".
  static std::string text(const datamunge::optim::FireflyAlgorithm& x) {
    return datamunge::format_detail::optim_with_options("FireflyAlgorithm", x.options());
  }
};

/// @brief Prints an optim::FISTA as "FISTA(...)" with its configured options.
template <>
struct std::formatter<datamunge::optim::FISTA> : datamunge::format_detail::text_formatter<datamunge::optim::FISTA> {
  /// @brief The type name followed by every option field.
  /// @param x The value to print.
  /// @return "FISTA(field=..., ...)".
  static std::string text(const datamunge::optim::FISTA& x) {
    return datamunge::format_detail::optim_with_options("FISTA", x.options());
  }
};

/// @brief Prints an optim::GeneticAlgorithm as "GeneticAlgorithm(...)" with its configured options.
template <>
struct std::formatter<datamunge::optim::GeneticAlgorithm> : datamunge::format_detail::text_formatter<datamunge::optim::GeneticAlgorithm> {
  /// @brief The type name followed by every option field.
  /// @param x The value to print.
  /// @return "GeneticAlgorithm(field=..., ...)".
  static std::string text(const datamunge::optim::GeneticAlgorithm& x) {
    return datamunge::format_detail::optim_with_options("GeneticAlgorithm", x.options());
  }
};

/// @brief Prints an optim::GradientDescent as "GradientDescent(...)" with its configured options.
template <>
struct std::formatter<datamunge::optim::GradientDescent> : datamunge::format_detail::text_formatter<datamunge::optim::GradientDescent> {
  /// @brief The type name followed by every option field.
  /// @param x The value to print.
  /// @return "GradientDescent(field=..., ...)".
  static std::string text(const datamunge::optim::GradientDescent& x) {
    return datamunge::format_detail::optim_with_options("GradientDescent", x.options());
  }
};

/// @brief Prints an optim::GreyWolfOptimizer as "GreyWolfOptimizer(...)" with its configured options.
template <>
struct std::formatter<datamunge::optim::GreyWolfOptimizer> : datamunge::format_detail::text_formatter<datamunge::optim::GreyWolfOptimizer> {
  /// @brief The type name followed by every option field.
  /// @param x The value to print.
  /// @return "GreyWolfOptimizer(field=..., ...)".
  static std::string text(const datamunge::optim::GreyWolfOptimizer& x) {
    return datamunge::format_detail::optim_with_options("GreyWolfOptimizer", x.options());
  }
};

/// @brief Prints an optim::HarmonySearch as "HarmonySearch(...)" with its configured options.
template <>
struct std::formatter<datamunge::optim::HarmonySearch> : datamunge::format_detail::text_formatter<datamunge::optim::HarmonySearch> {
  /// @brief The type name followed by every option field.
  /// @param x The value to print.
  /// @return "HarmonySearch(field=..., ...)".
  static std::string text(const datamunge::optim::HarmonySearch& x) {
    return datamunge::format_detail::optim_with_options("HarmonySearch", x.options());
  }
};

/// @brief Prints an optim::InteriorPoint as "InteriorPoint(...)" with its configured options.
template <>
struct std::formatter<datamunge::optim::InteriorPoint> : datamunge::format_detail::text_formatter<datamunge::optim::InteriorPoint> {
  /// @brief The type name followed by every option field.
  /// @param x The value to print.
  /// @return "InteriorPoint(field=..., ...)".
  static std::string text(const datamunge::optim::InteriorPoint& x) {
    return datamunge::format_detail::optim_with_options("InteriorPoint", x.options());
  }
};

/// @brief Prints an optim::LBFGS as "LBFGS(...)" with its configured options.
template <>
struct std::formatter<datamunge::optim::LBFGS> : datamunge::format_detail::text_formatter<datamunge::optim::LBFGS> {
  /// @brief The type name followed by every option field.
  /// @param x The value to print.
  /// @return "LBFGS(field=..., ...)".
  static std::string text(const datamunge::optim::LBFGS& x) {
    return datamunge::format_detail::optim_with_options("LBFGS", x.options());
  }
};

/// @brief Prints an optim::LevenbergMarquardt as "LevenbergMarquardt(...)" with its configured options.
template <>
struct std::formatter<datamunge::optim::LevenbergMarquardt> : datamunge::format_detail::text_formatter<datamunge::optim::LevenbergMarquardt> {
  /// @brief The type name followed by every option field.
  /// @param x The value to print.
  /// @return "LevenbergMarquardt(field=..., ...)".
  static std::string text(const datamunge::optim::LevenbergMarquardt& x) {
    return datamunge::format_detail::optim_with_options("LevenbergMarquardt", x.options());
  }
};

/// @brief Prints an optim::MOEAD as "MOEAD(...)" with its configured options.
template <>
struct std::formatter<datamunge::optim::MOEAD> : datamunge::format_detail::text_formatter<datamunge::optim::MOEAD> {
  /// @brief The type name followed by every option field.
  /// @param x The value to print.
  /// @return "MOEAD(field=..., ...)".
  static std::string text(const datamunge::optim::MOEAD& x) {
    return datamunge::format_detail::optim_with_options("MOEAD", x.options());
  }
};

/// @brief Prints an optim::Nadam as "Nadam(...)" with its configured options.
template <>
struct std::formatter<datamunge::optim::Nadam> : datamunge::format_detail::text_formatter<datamunge::optim::Nadam> {
  /// @brief The type name followed by every option field.
  /// @param x The value to print.
  /// @return "Nadam(field=..., ...)".
  static std::string text(const datamunge::optim::Nadam& x) {
    return datamunge::format_detail::optim_with_options("Nadam", x.options());
  }
};

/// @brief Prints an optim::NelderMead as "NelderMead(...)" with its configured options.
template <>
struct std::formatter<datamunge::optim::NelderMead> : datamunge::format_detail::text_formatter<datamunge::optim::NelderMead> {
  /// @brief The type name followed by every option field.
  /// @param x The value to print.
  /// @return "NelderMead(field=..., ...)".
  static std::string text(const datamunge::optim::NelderMead& x) {
    return datamunge::format_detail::optim_with_options("NelderMead", x.options());
  }
};

/// @brief Prints an optim::NesterovAcceleratedGradient as "NesterovAcceleratedGradient(...)" with its configured options.
template <>
struct std::formatter<datamunge::optim::NesterovAcceleratedGradient> : datamunge::format_detail::text_formatter<datamunge::optim::NesterovAcceleratedGradient> {
  /// @brief The type name followed by every option field.
  /// @param x The value to print.
  /// @return "NesterovAcceleratedGradient(field=..., ...)".
  static std::string text(const datamunge::optim::NesterovAcceleratedGradient& x) {
    return datamunge::format_detail::optim_with_options("NesterovAcceleratedGradient", x.options());
  }
};

/// @brief Prints an optim::Newton as "Newton(...)" with its configured options.
template <>
struct std::formatter<datamunge::optim::Newton> : datamunge::format_detail::text_formatter<datamunge::optim::Newton> {
  /// @brief The type name followed by every option field.
  /// @param x The value to print.
  /// @return "Newton(field=..., ...)".
  static std::string text(const datamunge::optim::Newton& x) {
    return datamunge::format_detail::optim_with_options("Newton", x.options());
  }
};

/// @brief Prints an optim::NSGA2 as "NSGA2(...)" with its configured options.
template <>
struct std::formatter<datamunge::optim::NSGA2> : datamunge::format_detail::text_formatter<datamunge::optim::NSGA2> {
  /// @brief The type name followed by every option field.
  /// @param x The value to print.
  /// @return "NSGA2(field=..., ...)".
  static std::string text(const datamunge::optim::NSGA2& x) {
    return datamunge::format_detail::optim_with_options("NSGA2", x.options());
  }
};

/// @brief Prints an optim::ParallelTempering as "ParallelTempering(...)" with its configured options.
template <>
struct std::formatter<datamunge::optim::ParallelTempering> : datamunge::format_detail::text_formatter<datamunge::optim::ParallelTempering> {
  /// @brief The type name followed by every option field.
  /// @param x The value to print.
  /// @return "ParallelTempering(field=..., ...)".
  static std::string text(const datamunge::optim::ParallelTempering& x) {
    return datamunge::format_detail::optim_with_options("ParallelTempering", x.options());
  }
};

/// @brief Prints an optim::ProximalGradient as "ProximalGradient(...)" with its configured options.
template <>
struct std::formatter<datamunge::optim::ProximalGradient> : datamunge::format_detail::text_formatter<datamunge::optim::ProximalGradient> {
  /// @brief The type name followed by every option field.
  /// @param x The value to print.
  /// @return "ProximalGradient(field=..., ...)".
  static std::string text(const datamunge::optim::ProximalGradient& x) {
    return datamunge::format_detail::optim_with_options("ProximalGradient", x.options());
  }
};

/// @brief Prints an optim::PSO as "PSO(...)" with its configured options.
template <>
struct std::formatter<datamunge::optim::PSO> : datamunge::format_detail::text_formatter<datamunge::optim::PSO> {
  /// @brief The type name followed by every option field.
  /// @param x The value to print.
  /// @return "PSO(field=..., ...)".
  static std::string text(const datamunge::optim::PSO& x) {
    return datamunge::format_detail::optim_with_options("PSO", x.options());
  }
};

/// @brief Prints an optim::RandomizedBlockCoordinateDescent as "RandomizedBlockCoordinateDescent(...)" with its configured options.
template <>
struct std::formatter<datamunge::optim::RandomizedBlockCoordinateDescent> : datamunge::format_detail::text_formatter<datamunge::optim::RandomizedBlockCoordinateDescent> {
  /// @brief The type name followed by every option field.
  /// @param x The value to print.
  /// @return "RandomizedBlockCoordinateDescent(field=..., ...)".
  static std::string text(const datamunge::optim::RandomizedBlockCoordinateDescent& x) {
    return datamunge::format_detail::optim_with_options("RandomizedBlockCoordinateDescent", x.options());
  }
};

/// @brief Prints an optim::RBFGaussianProcessSurrogate with its kernel hyperparameters and
///        the number of training points it is fitted to.
template <>
struct std::formatter<datamunge::optim::RBFGaussianProcessSurrogate> : datamunge::format_detail::text_formatter<datamunge::optim::RBFGaussianProcessSurrogate> {
  /// @brief Kernel hyperparameters plus fitted state.
  /// @param x The value to print.
  /// @return "RBFGaussianProcessSurrogate(length_scale=..., noise=..., n=...)" or "..., unfitted)".
  static std::string text(const datamunge::optim::RBFGaussianProcessSurrogate& x) {
    if (x.points().empty())
      return std::format("RBFGaussianProcessSurrogate(length_scale={}, noise={}, unfitted)", x.length_scale(), x.noise());
    return std::format("RBFGaussianProcessSurrogate(length_scale={}, noise={}, n={}, dim={})", x.length_scale(),
                       x.noise(), x.points().size(), x.points().front().size());
  }
};

/// @brief Prints an optim::RMSProp as "RMSProp(...)" with its configured options.
template <>
struct std::formatter<datamunge::optim::RMSProp> : datamunge::format_detail::text_formatter<datamunge::optim::RMSProp> {
  /// @brief The type name followed by every option field.
  /// @param x The value to print.
  /// @return "RMSProp(field=..., ...)".
  static std::string text(const datamunge::optim::RMSProp& x) {
    return datamunge::format_detail::optim_with_options("RMSProp", x.options());
  }
};

/// @brief Prints an optim::SAGA as "SAGA(...)" with its configured options.
template <>
struct std::formatter<datamunge::optim::SAGA> : datamunge::format_detail::text_formatter<datamunge::optim::SAGA> {
  /// @brief The type name followed by every option field.
  /// @param x The value to print.
  /// @return "SAGA(field=..., ...)".
  static std::string text(const datamunge::optim::SAGA& x) {
    return datamunge::format_detail::optim_with_options("SAGA", x.options());
  }
};

/// @brief Prints an optim::SGD as "SGD(...)" with its configured options.
template <>
struct std::formatter<datamunge::optim::SGD> : datamunge::format_detail::text_formatter<datamunge::optim::SGD> {
  /// @brief The type name followed by every option field.
  /// @param x The value to print.
  /// @return "SGD(field=..., ...)".
  static std::string text(const datamunge::optim::SGD& x) {
    return datamunge::format_detail::optim_with_options("SGD", x.options());
  }
};

/// @brief Prints an optim::SimulatedAnnealing as "SimulatedAnnealing(...)" with its configured options.
template <>
struct std::formatter<datamunge::optim::SimulatedAnnealing> : datamunge::format_detail::text_formatter<datamunge::optim::SimulatedAnnealing> {
  /// @brief The type name followed by every option field.
  /// @param x The value to print.
  /// @return "SimulatedAnnealing(field=..., ...)".
  static std::string text(const datamunge::optim::SimulatedAnnealing& x) {
    return datamunge::format_detail::optim_with_options("SimulatedAnnealing", x.options());
  }
};

/// @brief Prints an optim::SPEA2 as "SPEA2(...)" with its configured options.
template <>
struct std::formatter<datamunge::optim::SPEA2> : datamunge::format_detail::text_formatter<datamunge::optim::SPEA2> {
  /// @brief The type name followed by every option field.
  /// @param x The value to print.
  /// @return "SPEA2(field=..., ...)".
  static std::string text(const datamunge::optim::SPEA2& x) {
    return datamunge::format_detail::optim_with_options("SPEA2", x.options());
  }
};

/// @brief Prints an optim::SQP as "SQP(...)" with its configured options.
template <>
struct std::formatter<datamunge::optim::SQP> : datamunge::format_detail::text_formatter<datamunge::optim::SQP> {
  /// @brief The type name followed by every option field.
  /// @param x The value to print.
  /// @return "SQP(field=..., ...)".
  static std::string text(const datamunge::optim::SQP& x) {
    return datamunge::format_detail::optim_with_options("SQP", x.options());
  }
};

/// @brief Prints an optim::SVRG as "SVRG(...)" with its configured options.
template <>
struct std::formatter<datamunge::optim::SVRG> : datamunge::format_detail::text_formatter<datamunge::optim::SVRG> {
  /// @brief The type name followed by every option field.
  /// @param x The value to print.
  /// @return "SVRG(field=..., ...)".
  static std::string text(const datamunge::optim::SVRG& x) {
    return datamunge::format_detail::optim_with_options("SVRG", x.options());
  }
};

/// @brief Prints an optim::TrustRegionNewton as "TrustRegionNewton(...)" with its configured options.
template <>
struct std::formatter<datamunge::optim::TrustRegionNewton> : datamunge::format_detail::text_formatter<datamunge::optim::TrustRegionNewton> {
  /// @brief The type name followed by every option field.
  /// @param x The value to print.
  /// @return "TrustRegionNewton(field=..., ...)".
  static std::string text(const datamunge::optim::TrustRegionNewton& x) {
    return datamunge::format_detail::optim_with_options("TrustRegionNewton", x.options());
  }
};

/// @brief Prints an optim::WhaleOptimization as "WhaleOptimization(...)" with its configured options.
template <>
struct std::formatter<datamunge::optim::WhaleOptimization> : datamunge::format_detail::text_formatter<datamunge::optim::WhaleOptimization> {
  /// @brief The type name followed by every option field.
  /// @param x The value to print.
  /// @return "WhaleOptimization(field=..., ...)".
  static std::string text(const datamunge::optim::WhaleOptimization& x) {
    return datamunge::format_detail::optim_with_options("WhaleOptimization", x.options());
  }
};
