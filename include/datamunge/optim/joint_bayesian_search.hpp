#pragma once

/// @file
/// @brief Bayesian minimization with jointly sampled latent acquisitions.

#include <datamunge/optim/rbf_gaussian_process_surrogate.hpp>

#include <cstdint>

namespace datamunge::optim {

/// @brief Acquisition policy based on correlated latent function draws.
/// @par Example
/// @code
/// const auto acquisition = dm::optim::JointAcquisition::noisy_improvement;
/// @endcode
/// @see joint_bayesian_search()
enum class JointAcquisition {
  thompson,         ///< Minimize a single posterior draw over candidate points.
  noisy_improvement ///< Maximize average improvement relative to sampled latent baseline minima.
};

/// @brief Budget, candidate-pool controls and Gaussian process configuration.
/// @par Example
/// @code
/// dm::optim::JointBayesianOptions options;
/// options.acquisition = dm::optim::JointAcquisition::noisy_improvement;
/// @endcode
/// @see joint_bayesian_search()
struct JointBayesianOptions {
  std::size_t evaluations{50}; ///< Positive total objective-call cap.
  std::uint64_t seed{42}; ///< Seed for candidate designs and joint posterior draws.
  std::size_t initial_samples{6}; ///< Positive number of independent initial samples.
  std::size_t candidates{32}; ///< Positive number of uniform candidates at each acquisition step.
  JointAcquisition acquisition{JointAcquisition::thompson}; ///< Candidate ranking rule.
  std::size_t posterior_draws{64}; ///< Positive Monte Carlo draws for noisy improvement.
  std::size_t replicate_every{0}; ///< Optional periodic replication of the most promising observed point.
  GaussianProcessSurrogateOptions gaussian; ///< Model settings; observation noise enables repeated locations.
};

/// @brief Bayesian observations and a noise-aware final recommendation.
/// @par Example
/// @code
/// const dm::optim::JointBayesianResult result;
/// std::print("{}\n", result.replicates); // 0
/// @endcode
/// @see joint_bayesian_search()
struct JointBayesianResult {
  BayesianOptimizationResult observed; ///< Chronological observations and raw best loss.
  std::vector<double> recommended_point; ///< Evaluated point minimizing the final latent posterior mean.
  double recommended_mean{}; ///< Final latent mean at the recommendation.
  std::size_t replicates{}; ///< Number of explicit repeated evaluations.
};

/// @brief Search a box using correlated Thompson or latent noisy expected improvement.
///
/// Fits an RBF Gaussian process to all observations, retaining repeated
/// locations individually. Thompson draws across the candidate pool from one
/// latent multivariate Gaussian per decision. Noisy improvement draws
/// jointly across observed locations and candidates, averaging positive
/// improvement relative to each draw's latent baseline minimum. An optional
/// periodic replicate evaluates the observed point of smallest posterior
/// mean. The final recommendation also uses posterior means, while the
/// observed result preserves the raw incumbent history. An acquisition score
/// is stored per model-driven evaluation; scheduled replicates store NaN.
/// The GP is fitted in normalized box coordinates. Callback exceptions
/// propagate. Seeded replay holds within a build for deterministic callbacks.
/// @param function Borrowed objective returning finite observed losses.
/// @param lower Finite nonempty lower bounds.
/// @param upper Finite upper bounds, at least as large as matching lower bounds.
/// @param options Budget, seed, surrogate and candidate selection settings.
/// @return Owned observations, actual replicate count and posterior recommendation.
/// @throws std::invalid_argument Invalid box, acquisition, budget or model controls.
/// @throws std::domain_error The objective returns nonfinite loss.
/// @throws std::runtime_error The model or sampled posterior covariance cannot be factored.
/// @par Example
/// @code
/// dm::optim::QuadraticObjective f({2}, {1});
/// dm::optim::JointBayesianOptions options;
/// options.evaluations = 3;
/// const auto result = dm::optim::joint_bayesian_search(f, {1}, {1}, options);
/// std::print("{}\n", result.observed.best_value); // 0
/// @endcode
/// @par References
/// - Russo et al. (2018). A Tutorial on Thompson Sampling. https://arxiv.org/abs/1707.02038
/// - Letham et al. (2019). Constrained Bayesian Optimization with Noisy Experiments.
///   https://arxiv.org/abs/1706.07094
/// @see RBFGaussianProcessSurrogate, BayesianOptimization
JointBayesianResult joint_bayesian_search(ArbitraryFunction& function, const std::vector<double>& lower,
                                           const std::vector<double>& upper,
                                           const JointBayesianOptions& options = {});

} // namespace datamunge::optim
