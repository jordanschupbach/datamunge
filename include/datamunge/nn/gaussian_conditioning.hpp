#pragma once

/// @file gaussian_conditioning.hpp
/// @brief Exact Gaussian conditioning with known covariance blocks.

#include <cmath>
#include <numbers>
#include <stdexcept>
#include <vector>

#include "datamunge/nn/autograd.hpp"
#include "datamunge/nn/linalg.hpp"
#include "datamunge/nn/ops.hpp"

namespace datamunge::nn {

/// @brief Fit state for a zero-mean Gaussian model, in Float64.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto fit = n::fit_gaussian_covariance(n::eye(2), n::zeros({2}));
/// const auto log_evidence = fit.log_evidence; // -log(2*pi)
/// @endcode
/// @see fit_gaussian_covariance(), condition_gaussian()
struct GaussianCovarianceFit {
  Tensor lower;          ///< Detached lower Cholesky factor of the observation covariance.
  Tensor whitened;       ///< Detached response after a forward triangular solve, shaped [N,1].
  double log_evidence{}; ///< Gaussian log density of the supplied observation vector.
};

/// @brief Joint conditional Gaussian moments.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto fit = n::fit_gaussian_covariance(n::eye(1), n::tensor({2.}));
/// const auto result = n::condition_gaussian(fit, n::full({1,1}, .5), n::eye(1));
/// const auto variance = result.covariance.item(); // 0.75
/// @endcode
/// @see condition_gaussian()
struct GaussianConditional {
  Tensor mean;       ///< Detached Float64 conditional mean [Q,1].
  Tensor covariance; ///< Detached Float64 joint conditional covariance [Q,Q].
};

/// @brief Factor the training covariance and evaluate a zero-mean Gaussian log density.
///
/// Work is O(N³) and storage is O(N²). Include training observation noise in
/// the covariance. For a nonzero prior mean, pass residuals about that mean.
/// The lower triangle defines the covariance for the Cholesky factorization.
/// @param covariance Nonempty finite CPU [N,N] covariance, positive definite.
/// @param observations Finite CPU [N] or [N,1] observation vector.
/// @return Detached Float64 factor and whitened response with their log evidence.
/// @throws std::invalid_argument For bad shapes, device or nonfinite inputs.
/// @throws std::runtime_error For failed factorization or overflow.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto fit = n::fit_gaussian_covariance(n::eye(1), n::tensor({2.}));
/// const auto whitened = fit.whitened.item(); // 2
/// @endcode
/// @par References
/// - Rasmussen and Williams, Gaussian Processes for Machine Learning, Chapter 2.
///   https://gaussianprocess.org/gpml/chapters/RW2.pdf
/// @see GaussianCovarianceFit, condition_gaussian()
inline GaussianCovarianceFit fit_gaussian_covariance(const Tensor& covariance,
                                                      const Tensor& observations) {
  if (!covariance.defined() || covariance.dim() != 2 || covariance.size(0) == 0 ||
      covariance.size(0) != covariance.size(1) || covariance.device().backend != compute::Backend::CPU ||
      !observations.defined() || observations.device().backend != compute::Backend::CPU ||
      (observations.shape() != Shape{covariance.size(0)} &&
       observations.shape() != Shape{covariance.size(0), 1}))
    throw std::invalid_argument("Gaussian conditioning requires a square covariance and one response");
  NoGradGuard guard;
  const auto matrix = covariance.to(DType::Float64);
  const auto values = observations.to(DType::Float64).reshape({covariance.size(0), 1});
  if (!all(is_finite(matrix)).item<bool>() || !all(is_finite(values)).item<bool>())
    throw std::invalid_argument("Gaussian conditioning inputs must be finite");
  const auto lower = cholesky(matrix);
  const auto whitened = triangular_solve(lower, values);
  const double evidence = -.5 * (sum(square(whitened)).item() +
      covariance.size(0) * std::log(2. * std::numbers::pi)) -
      sum(log(matrix_diag_part(lower))).item();
  if (!std::isfinite(evidence))
    throw std::runtime_error("Gaussian conditioning log evidence overflowed");
  return {lower, whitened, evidence};
}

/// @brief Evaluate exact Gaussian negative log evidence with its first covariance derivative.
///
/// The input covariance must be symmetric and positive definite. The forward
/// value uses a Float64 Cholesky solve. Reverse mode uses the symmetric
/// derivative 0.5 (K^-1 - a a^T), where a = K^-1 y.
/// This operation differentiates covariance parameters only and retains
/// first derivatives. Observations are treated as fixed.
/// @param covariance Finite CPU floating covariance [N,N].
/// @param observations Finite fixed observations [N] or [N,1].
/// @return Scalar Float64 negative log evidence.
/// @throws std::invalid_argument Invalid covariance dtype, shape or data.
/// @throws std::runtime_error Failed factorization or nonfinite gradient.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// auto variance = n::eye(1).requires_grad_(true);
/// auto loss = n::gaussian_covariance_nll(variance, n::zeros({1}));
/// auto derivative = n::grad(loss, {variance}).front().item(); // 0.5
/// @endcode
/// @par References
/// - Rasmussen and Williams, Gaussian Processes for Machine Learning, Chapter 5.
///   https://gaussianprocess.org/gpml/chapters/RW5.pdf
/// @see fit_gaussian_covariance(), condition_gaussian()
inline Tensor gaussian_covariance_nll(const Tensor& covariance, const Tensor& observations) {
  if (!covariance.defined() ||
      (covariance.dtype() != DType::Float32 && covariance.dtype() != DType::Float64))
    throw std::invalid_argument("Gaussian covariance gradient requires floating covariance");
  const auto fit = fit_gaussian_covariance(covariance, observations);
  const auto live = covariance.to(DType::Float64);
  const auto inverse_lower = triangular_solve(fit.lower, eye(live.size(0), DType::Float64));
  const auto alpha = matmul(inverse_lower, fit.whitened, true, false);
  const auto derivative = .5 * (matmul(inverse_lower, inverse_lower, true, false) -
                                matmul(alpha, alpha, false, true));
  if (!all(is_finite(derivative)).item<bool>())
    throw std::runtime_error("Gaussian covariance gradient overflowed");
  return -fit.log_evidence + sum((live - live.detach()) * derivative);
}

/// @brief Condition query values on the fitted Gaussian observations.
///
/// With L the stored factor and A = L^-1 cross, returns A^T whitened and
/// query_covariance - A^T A. The caller supplies coherent covariance blocks.
/// Include new-observation noise in the query covariance when predicting
/// responses. Work is O(N² Q + N Q²). Add the prior query mean separately
/// when conditioning on residuals about a nonzero mean.
/// @param fit Valid posterior state returned by fit_gaussian_covariance().
/// @param cross Finite CPU [N,Q] covariance between observations and queries, Q>0.
/// @param query_covariance Finite symmetric CPU [Q,Q] prior query covariance.
/// @return Detached Float64 joint query moments.
/// @throws std::invalid_argument For invalid shapes, devices or nonfinite inputs.
/// @throws std::runtime_error For nonfinite conditional moments.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto fit = n::fit_gaussian_covariance(n::eye(1), n::tensor({2.}));
/// const auto prediction = n::condition_gaussian(fit, n::full({1,1}, .5), n::eye(1));
/// const auto mean = prediction.mean.item(); // 1
/// @endcode
/// @par References
/// - Rasmussen and Williams, Gaussian Processes for Machine Learning, Chapter 2.
///   https://gaussianprocess.org/gpml/chapters/RW2.pdf
/// @see GaussianConditional, fit_gaussian_covariance()
inline GaussianConditional condition_gaussian(const GaussianCovarianceFit& fit, const Tensor& cross,
                                               const Tensor& query_covariance) {
  if (!fit.lower.defined() || !cross.defined() || cross.dim() != 2 ||
      cross.size(0) != fit.lower.size(0) || cross.size(1) == 0 ||
      cross.device().backend != compute::Backend::CPU || !query_covariance.defined() ||
      query_covariance.shape() != Shape{cross.size(1), cross.size(1)} ||
      query_covariance.device().backend != compute::Backend::CPU)
    throw std::invalid_argument("invalid Gaussian conditional query geometry");
  NoGradGuard guard;
  const auto cross64 = cross.to(DType::Float64);
  const auto prior64 = query_covariance.to(DType::Float64);
  if (!all(is_finite(cross64)).item<bool>() || !all(is_finite(prior64)).item<bool>())
    throw std::invalid_argument("Gaussian conditional covariance blocks must be finite");
  const auto projected = triangular_solve(fit.lower, cross64);
  const auto mean = matmul(projected, fit.whitened, true, false);
  const auto covariance = prior64 - matmul(projected, projected, true, false);
  if (!all(is_finite(mean)).item<bool>() || !all(is_finite(covariance)).item<bool>())
    throw std::runtime_error("Gaussian conditional moments overflowed");
  return {mean, covariance};
}

/// @brief Optimal whitened inducing posterior for Gaussian observations.
/// @see fit_sparse_variational_gaussian(), predict_sparse_variational_gaussian()
struct SparseVariationalGaussianFit {
  Tensor inducing_lower;   ///< Float64 lower Cholesky factor of the inducing covariance.
  Tensor precision_lower;  ///< Float64 lower Cholesky factor of whitened posterior precision.
  Tensor whitened_mean;    ///< Float64 posterior inducing mean [M,1].
  Tensor whitened_covariance; ///< Float64 posterior inducing covariance [M,M].
  double elbo{};          ///< Collapsed evidence lower bound in nats.
  double trace_penalty{}; ///< Residual prior variance on training inputs divided by observation variance.
};

/// @brief Fit the optimal sparse variational GP posterior for a zero-mean prior.
///
/// Supply a positive-definite inducing covariance with any required jitter
/// already applied. The training prior enters through its marginal variances.
/// Time is O(N M² + M³) and additional memory is O(N M + M²).
/// @param inducing_covariance Finite symmetric positive definite CPU [M,M] prior covariance.
/// @param cross Finite CPU [M,N] inducing-to-training covariance.
/// @param training_variance Finite nonnegative CPU [N] training prior marginal variances.
/// @param observations Finite CPU [N] or [N,1] training responses.
/// @param noise_variance Finite positive Gaussian observation variance.
/// @return Detached Float64 fit and its collapsed lower bound.
/// @throws std::invalid_argument For malformed or nonfinite inputs.
/// @throws std::runtime_error For failed Cholesky factorization or nonfinite outputs.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto fit = n::fit_sparse_variational_gaussian(
///     n::eye(1), n::eye(1), n::ones({1}), n::tensor({2.}), 1.);
/// const auto mean = fit.whitened_mean.item(); // 1
/// @endcode
/// @par References
/// - Titsias, Variational Learning of Inducing Variables in Sparse Gaussian Processes, AISTATS 2009.
///   https://proceedings.mlr.press/v5/titsias09a.html
/// @see SparseVariationalGaussianFit, predict_sparse_variational_gaussian()
inline SparseVariationalGaussianFit fit_sparse_variational_gaussian(
    const Tensor& inducing_covariance, const Tensor& cross, const Tensor& training_variance,
    const Tensor& observations, double noise_variance) {
  if (!inducing_covariance.defined() || inducing_covariance.dim() != 2 ||
      inducing_covariance.size(0) == 0 || inducing_covariance.size(0) != inducing_covariance.size(1) ||
      !cross.defined() || cross.dim() != 2 || cross.size(0) != inducing_covariance.size(0) ||
      cross.size(1) == 0 || !training_variance.defined() ||
      training_variance.shape() != Shape{cross.size(1)} || !observations.defined() ||
      (observations.shape() != Shape{cross.size(1)} &&
       observations.shape() != Shape{cross.size(1), 1}) ||
      !std::isfinite(noise_variance) || noise_variance <= 0.)
    throw std::invalid_argument("invalid sparse variational Gaussian training geometry or noise");
  NoGradGuard guard;
  for (const auto* input : {&inducing_covariance, &cross, &training_variance, &observations})
    if (input->device().backend != compute::Backend::CPU || !all(is_finite(*input)).item<bool>())
      throw std::invalid_argument("sparse variational Gaussian inputs must be finite CPU tensors");
  if (any(less(training_variance, 0.)).item<bool>())
    throw std::invalid_argument("training prior variances must be nonnegative");
  const auto inducing_lower = cholesky(inducing_covariance.to(DType::Float64));
  const auto a = triangular_solve(inducing_lower, cross.to(DType::Float64));
  const auto values = observations.to(DType::Float64).reshape({cross.size(1), 1});
  const auto identity = eye(cross.size(0), DType::Float64);
  const auto precision_lower = cholesky(identity + matmul(a, a, false, true) / noise_variance);
  const auto whitened_info = triangular_solve(precision_lower, matmul(a, values) / noise_variance);
  const auto inverse_lower = triangular_solve(precision_lower, identity);
  const auto mean = matmul(inverse_lower, whitened_info, true, false);
  const auto covariance = matmul(inverse_lower, inverse_lower, true, false);
  const double penalty = (sum(training_variance.to(DType::Float64)).item() -
                          sum(square(a)).item()) / noise_variance;
  const double quadratic = sum(square(values)).item() / noise_variance -
                           sum(square(whitened_info)).item();
  const double elbo = -sum(log(matrix_diag_part(precision_lower))).item() -
      .5 * (quadratic + penalty + cross.size(1) * std::log(2. * std::numbers::pi * noise_variance));
  if (!std::isfinite(elbo) || !all(is_finite(mean)).item<bool>() || !all(is_finite(covariance)).item<bool>())
    throw std::runtime_error("sparse variational Gaussian posterior overflowed");
  return {inducing_lower, precision_lower, mean, covariance, elbo, penalty};
}

/// @brief Compute joint latent query moments from an inducing posterior.
/// @param fit Valid fit returned by fit_sparse_variational_gaussian().
/// @param cross Finite CPU [M,Q] inducing-to-query covariance, Q>0.
/// @param query_covariance Finite symmetric CPU [Q,Q] prior covariance of query values.
/// @return Detached Float64 conditional mean and covariance.
/// @throws std::invalid_argument For bad shapes, devices or nonfinite inputs.
/// @throws std::runtime_error For nonfinite predictions.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto fit = n::fit_sparse_variational_gaussian(
///     n::eye(1), n::eye(1), n::ones({1}), n::tensor({2.}), 1.);
/// const auto prediction = n::predict_sparse_variational_gaussian(fit, n::eye(1), n::eye(1));
/// const auto variance = prediction.covariance.item(); // 0.5
/// @endcode
/// @see fit_sparse_variational_gaussian(), GaussianConditional
inline GaussianConditional predict_sparse_variational_gaussian(
    const SparseVariationalGaussianFit& fit, const Tensor& cross, const Tensor& query_covariance) {
  if (!fit.inducing_lower.defined() || !cross.defined() || cross.dim() != 2 ||
      cross.size(0) != fit.inducing_lower.size(0) || cross.size(1) == 0 ||
      cross.device().backend != compute::Backend::CPU || !query_covariance.defined() ||
      query_covariance.shape() != Shape{cross.size(1), cross.size(1)} ||
      query_covariance.device().backend != compute::Backend::CPU)
    throw std::invalid_argument("invalid sparse variational Gaussian query geometry");
  NoGradGuard guard;
  const auto cross64 = cross.to(DType::Float64);
  const auto prior = query_covariance.to(DType::Float64);
  if (!all(is_finite(cross64)).item<bool>() || !all(is_finite(prior)).item<bool>())
    throw std::invalid_argument("sparse variational Gaussian query blocks must be finite");
  const auto a = triangular_solve(fit.inducing_lower, cross64);
  const auto uncertainty = triangular_solve(fit.precision_lower, a);
  const auto mean = matmul(a, fit.whitened_mean, true, false);
  const auto covariance = prior - matmul(a, a, true, false) +
                          matmul(uncertainty, uncertainty, true, false);
  if (!all(is_finite(mean)).item<bool>() || !all(is_finite(covariance)).item<bool>())
    throw std::runtime_error("sparse variational Gaussian prediction overflowed");
  return {mean, covariance};
}

/// @brief Fixed scalar RBF layers and Gaussian likelihood for a two-layer deep GP.
/// @par Example
/// @code
/// auto options = dm::nn::DeepGaussianProcessOptions{};
/// options.observation_variance = .04;
/// @endcode
/// @see condition_deep_gaussian_process()
struct DeepGaussianProcessOptions {
  double hidden_length{.6};       ///< First-layer RBF length in input units.
  double hidden_variance{.09};    ///< First-layer RBF marginal variance.
  double output_length{.55};      ///< Second-layer RBF length in hidden units.
  double output_variance{1.};     ///< Second-layer RBF marginal variance.
  double observation_variance{.04}; ///< Independent Gaussian observation variance.
  double jitter{1e-8};             ///< First-layer diagonal stabilization variance.
};

/// @brief Marginal predictions and importance diagnostics for a two-layer deep GP.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto fit = n::condition_deep_gaussian_process(
///     n::tensor({0., 0.}), n::tensor({2.}), n::zeros({2, 2}));
/// const auto mean = fit.mean.item(); // 2 / 1.04
/// @endcode
/// @see condition_deep_gaussian_process()
struct DeepGaussianProcessPrediction {
  Tensor hidden;               ///< Joint hidden draws [S,N+Q], in input order.
  Tensor conditional_mean;     ///< Second-layer conditional mean per draw [S,Q].
  Tensor conditional_variance; ///< Second-layer latent variance per draw [S,Q].
  Tensor log_evidence;         ///< Training Gaussian log evidence per draw [S].
  Tensor weights;              ///< Normalized importance weights [S].
  Tensor mean;                 ///< Marginal latent mean [Q].
  Tensor variance;             ///< Marginal latent variance [Q].
  double effective_sample_size{}; ///< Inverse sum of squared normalized weights.
};

/// @brief Condition a two-layer scalar RBF Gaussian process using importance sampling.
///
/// Samples the hidden layer jointly over all inputs using an identity mean
/// and caller-supplied standard normals. Each sample gives an exact Gaussian
/// output conditional with a zero-mean prior. Normalized training likelihoods
/// weight the resulting mixture. Marginal variance includes within-draw and
/// between-draw uncertainty. Hyperparameters are fixed. Storage for the
/// mixture diagnostics is O(S (N+Q)), with an additional O((N+Q)²) for sampling.
/// @param inputs Finite CPU [N+Q] inputs ordered training first, then queries.
/// @param observations Finite CPU [N] training responses, N>0 and Q>0.
/// @param standard_normal Finite CPU [N+Q,S] base draws, S>0.
/// @param options Positive finite RBF lengths, variances, observation variance and jitter.
/// @return Detached Float64 latent moments and importance diagnostics.
/// @throws std::invalid_argument For malformed or nonfinite inputs or options.
/// @throws std::runtime_error For failed factorization or overflow.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto fit = n::condition_deep_gaussian_process(
///     n::tensor({0., 0.}), n::tensor({2.}), n::zeros({2, 2}));
/// const auto effective_draws = fit.effective_sample_size; // 2
/// @endcode
/// @par References
/// - Damianou and Lawrence, Deep Gaussian Processes, AISTATS 2013 (model family).
///   https://proceedings.mlr.press/v31/damianou13a.html
///   This implementation uses importance sampling for the hidden layer.
/// @see fit_gaussian_covariance(), DeepGaussianProcessPrediction
inline DeepGaussianProcessPrediction condition_deep_gaussian_process(
    const Tensor& inputs, const Tensor& observations, const Tensor& standard_normal,
    const DeepGaussianProcessOptions& options = {}) {
  if (!inputs.defined() || inputs.dim() != 1 || !observations.defined() ||
      observations.dim() != 1 || observations.size(0) == 0 || inputs.size(0) <= observations.size(0) ||
      !standard_normal.defined() || standard_normal.dim() != 2 ||
      standard_normal.size(0) != inputs.size(0) || standard_normal.size(1) == 0)
    throw std::invalid_argument("invalid two-layer Gaussian process geometry");
  for (double value : {options.hidden_length, options.hidden_variance, options.output_length,
                       options.output_variance, options.observation_variance, options.jitter})
    if (!std::isfinite(value) || value <= 0.)
      throw std::invalid_argument("two-layer Gaussian process options must be positive and finite");
  NoGradGuard guard;
  for (const auto* input : {&inputs, &observations, &standard_normal})
    if (input->device().backend != compute::Backend::CPU || !all(is_finite(*input)).item<bool>())
      throw std::invalid_argument("two-layer Gaussian process inputs must be finite CPU tensors");

  const auto kernel = [](const Tensor& a, const Tensor& b, double length, double variance) {
    return variance * exp(-.5 * square((expand_dims(a, 1) - expand_dims(b, 0)) / length));
  };
  const auto x = inputs.to(DType::Float64);
  const auto prior = kernel(x, x, options.hidden_length, options.hidden_variance) +
      options.jitter * eye(x.size(0), DType::Float64);
  const auto hidden = transpose(expand_dims(x, 1) +
      matmul(cholesky(prior), standard_normal.to(DType::Float64)), {1, 0});
  const auto n = observations.size(0), q = x.size(0) - n;
  const auto noise = options.observation_variance * eye(n, DType::Float64);
  std::vector<Tensor> means, variances;
  std::vector<double> evidence;
  for (std::int64_t draw = 0; draw < hidden.size(0); ++draw) {
    const auto train = slice(hidden, {draw, 0}, {1, n}).reshape({n});
    const auto query = slice(hidden, {draw, n}, {1, q}).reshape({q});
    const auto fit = fit_gaussian_covariance(
        kernel(train, train, options.output_length, options.output_variance) + noise, observations);
    const auto projected = triangular_solve(fit.lower,
        kernel(train, query, options.output_length, options.output_variance));
    means.push_back(matmul(projected, fit.whitened, true, false).reshape({q}));
    variances.push_back(options.output_variance - sum(square(projected), {0}));
    evidence.push_back(fit.log_evidence);
  }
  auto component_mean = stack(means), component_variance = stack(variances);
  auto log_evidence = from_vector(evidence, {hidden.size(0)});
  auto weights = softmax(log_evidence);
  const auto mass = expand_dims(weights, 1);
  auto mean = sum(mass * component_mean, {0});
  auto variance = sum(mass * (component_variance + square(component_mean - mean)), {0});
  if (!all(is_finite(mean)).item<bool>() || !all(is_finite(variance)).item<bool>())
    throw std::runtime_error("two-layer Gaussian process prediction overflowed");
  const double ess = 1. / sum(square(weights)).item();
  return {hidden, component_mean, component_variance, log_evidence, weights, mean, variance, ess};
}

}  // namespace datamunge::nn
