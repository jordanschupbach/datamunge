#pragma once

/// @file probabilistic_regression.hpp
/// @brief Gaussian regression heads and distribution-free prediction intervals.

#include <datamunge/nn/layers.hpp>
#include <datamunge/nn/ops.hpp>
#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <vector>

namespace datamunge::nn {

/// @brief Conditional Gaussian parameters from one prediction per input row.
struct GaussianRegressionPrediction {
  Tensor mean;     ///< Location, shape [rows, 1].
  Tensor variance; ///< Positive noise variance, shape [rows, 1].
};

/// @brief Read separate location and variance scores from a learned affine head.
/// @param parameters Caller-owned store initialized on first use.
/// @param name Prefix for the head's learned parameters.
/// @param features Floating matrix [rows, features].
/// @param variance_floor Positive additive floor for the softplus variance.
/// @return Differentiable location and variance values.
/// @throws std::invalid_argument Invalid dimensions or floor.
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore store(12);
/// auto prediction = n::GaussianRegressionHead(store, "gaussian", n::zeros({3, 4}));
/// @endcode
inline GaussianRegressionPrediction GaussianRegressionHead(
    ParameterStore& parameters, const std::string& name, const Tensor& features,
    double variance_floor = 1e-4) {
  if (!features.defined() || features.dim() != 2 || features.size(0) <= 0 ||
      features.size(1) <= 0 || !std::isfinite(variance_floor) || variance_floor <= 0 ||
      (features.dtype() != DType::Float32 && features.dtype() != DType::Float64) ||
      parameters.dtype() != features.dtype()) {
    throw std::invalid_argument("Gaussian regression head dimensions, dtype or variance floor");
  }
  if (features.dtype() == DType::Float32 &&
      (variance_floor < std::numeric_limits<float>::denorm_min() ||
       variance_floor > std::numeric_limits<float>::max())) {
    throw std::invalid_argument("Gaussian variance floor is outside Float32 range");
  }
  const auto head = dense(parameters, name, features, 2);
  return {slice_axis(head, 1, 0, 1), softplus(slice_axis(head, 1, 1, 2)) + variance_floor};
}

/// @brief Elementwise Gaussian log density, including the normalization term.
/// @param prediction Finite location and strictly positive finite variance, both shaped like observation.
/// @param observation Continuous observed values with nonempty shape.
/// @return Differentiable log density at each observation.
/// @throws std::invalid_argument Invalid or incompatible tensors.
/// @code
/// namespace n = dm::nn;
/// auto logp = n::GaussianRegressionLogDensity(
///     {n::zeros({2, 1}), n::ones({2, 1})}, n::zeros({2, 1}));
/// @endcode
inline Tensor GaussianRegressionLogDensity(
    const GaussianRegressionPrediction& prediction, const Tensor& observation) {
  const auto& [location, variance] = prediction;
  if (!location.defined() || !variance.defined() || !observation.defined() ||
      observation.numel() <= 0 || location.shape() != observation.shape() ||
      variance.shape() != observation.shape() || location.dtype() != observation.dtype() ||
      variance.dtype() != observation.dtype() ||
      (observation.dtype() != DType::Float32 && observation.dtype() != DType::Float64) ||
      !all(is_finite(location)).item<bool>() || !all(is_finite(observation)).item<bool>() ||
      !all(logical_and(is_finite(variance), greater(variance, 0))).item<bool>()) {
    throw std::invalid_argument("Gaussian regression likelihood requires compatible finite tensors and positive variance");
  }
  constexpr double log_sqrt_2pi = .91893853320467274178;
  return -log_sqrt_2pi - .5 * (log(variance) + square(observation - location) / variance);
}

/// @brief Mean Gaussian negative log-likelihood, including the normalization term.
/// @param prediction Conditional locations and variances with shape matching observation.
/// @param observation Continuous targets.
/// @return Differentiable scalar mean negative log-likelihood in nats.
/// @see GaussianRegressionLogDensity()
inline Tensor GaussianRegressionNll(
    const GaussianRegressionPrediction& prediction, const Tensor& observation) {
  return -mean(GaussianRegressionLogDensity(prediction, observation));
}

/// @brief Moments of an equal-weight mixture of Gaussian predictions.
struct GaussianEnsembleSummary {
  Tensor mean;    ///< Average member mean.
  Tensor within;  ///< Average member variance.
  Tensor between; ///< Population variance of member means.
  Tensor total;   ///< Predictive variance, the sum of within and between.
};

/// @brief Decompose the total variance across the leading member axis.
/// @param members Matching tensors [member, ...] with positive variances.
/// @return Differentiable mixture moments with the leading member axis removed.
/// @throws std::invalid_argument Invalid or empty member geometry.
/// @code
/// namespace n = dm::nn;
/// auto s = n::GaussianEnsembleMoments({n::zeros({2, 3}), n::ones({2, 3})});
/// @endcode
inline GaussianEnsembleSummary GaussianEnsembleMoments(const GaussianRegressionPrediction& members) {
  if (!members.mean.defined() || !members.variance.defined() ||
      members.mean.numel() <= 0 || members.mean.dim() < 2 ||
      members.mean.shape() != members.variance.shape() ||
      members.mean.dtype() != members.variance.dtype() ||
      (members.mean.dtype() != DType::Float32 && members.mean.dtype() != DType::Float64) ||
      !all(is_finite(members.mean)).item<bool>() ||
      !all(logical_and(is_finite(members.variance), greater(members.variance, 0))).item<bool>()) {
    throw std::invalid_argument("Gaussian ensemble moments require finite members and positive variances");
  }
  const auto center = mean(members.mean, {0});
  const auto within = mean(members.variance, {0});
  const auto between = mean(square(members.mean - center), {0});
  return {center, within, between, within + between};
}

/// @brief Combine member log densities without forming densities in probability space.
/// @param member_log_density Tensor [member, ...] of log densities.
/// @return Equal-weight mixture log density with the member axis removed.
/// @throws std::invalid_argument Invalid or empty member geometry.
/// @code
/// auto logp = dm::nn::GaussianEnsembleLogDensity(dm::nn::zeros({2, 3}));
/// @endcode
inline Tensor GaussianEnsembleLogDensity(const Tensor& member_log_density) {
  if (!member_log_density.defined() || member_log_density.dim() < 2 ||
      member_log_density.numel() <= 0 ||
      (member_log_density.dtype() != DType::Float32 && member_log_density.dtype() != DType::Float64)) {
    throw std::invalid_argument("Gaussian ensemble log densities require a nonempty member axis");
  }
  return logsumexp(member_log_density, {0}) - std::log(double(member_log_density.size(0)));
}

/// @brief A conditional Gaussian mixture with one row per input.
struct GaussianMixturePrediction {
  Tensor logits; ///< Unnormalized mixing weights, shape [rows, components].
  Tensor mean;   ///< Location of each component, with the same shape.
  Tensor scale;  ///< Positive component standard deviations, with the same shape.
};

/// @brief Convert an affine head's outputs into conditional mixture parameters.
/// @param output Floating matrix [rows, 3 * components], in logit/mean/scale order.
/// @param mean_offsets Floating vector [components] in the output dtype.
/// @param scale_floor Positive additive floor for the softplus scale.
/// @return Differentiable logits, shifted means and scales.
/// @throws std::invalid_argument Invalid shapes, dtype or scale floor.
/// @code
/// namespace n = dm::nn;
/// auto p = n::GaussianMixtureHead(n::zeros({3, 6}), n::zeros({2}));
/// @endcode
inline GaussianMixturePrediction GaussianMixtureHead(
    const Tensor& output, const Tensor& mean_offsets, double scale_floor = .02) {
  if (!output.defined() || !mean_offsets.defined() || output.dim() != 2 ||
      output.size(0) <= 0 || output.size(1) <= 0 || output.size(1) % 3 != 0 ||
      mean_offsets.shape() != Shape{output.size(1) / 3} || mean_offsets.dtype() != output.dtype() ||
      (output.dtype() != DType::Float32 && output.dtype() != DType::Float64) ||
      !std::isfinite(scale_floor) || scale_floor <= 0 ||
      (output.dtype() == DType::Float32 &&
       (scale_floor < std::numeric_limits<float>::denorm_min() ||
        scale_floor > std::numeric_limits<float>::max()))) {
    throw std::invalid_argument("Gaussian mixture head dimensions, dtype or scale floor");
  }
  const auto parts = split(output, 3, 1);
  return {parts[0], parts[1] + mean_offsets, softplus(parts[2]) + scale_floor};
}

/// @brief Mean negative log likelihood for a conditional Gaussian mixture.
/// @param mixture Gaussian component tensors [rows, components].
/// @param observation Observed values [rows, 1], in the same floating dtype.
/// @return Differentiable scalar NLL using stable log-sum-exp of component densities.
/// @throws std::invalid_argument Incompatible or nonfinite mixture parameters.
/// @code
/// namespace n = dm::nn;
/// auto nll = n::GaussianMixtureNll(n::GaussianMixtureHead(n::zeros({2, 6}), n::zeros({2})),
///                                    n::zeros({2, 1}));
/// @endcode
inline Tensor GaussianMixtureNll(
    const GaussianMixturePrediction& mixture, const Tensor& observation) {
  const auto& [logits, location, scale] = mixture;
  if (!logits.defined() || !location.defined() || !scale.defined() || !observation.defined() ||
      logits.dim() != 2 || logits.numel() <= 0 || location.shape() != logits.shape() ||
      scale.shape() != logits.shape() || observation.shape() != Shape{logits.size(0), 1} ||
      logits.dtype() != location.dtype() || logits.dtype() != scale.dtype() ||
      logits.dtype() != observation.dtype() ||
      (logits.dtype() != DType::Float32 && logits.dtype() != DType::Float64) ||
      !all(is_finite(logits)).item<bool>() || !all(is_finite(location)).item<bool>() ||
      !all(is_finite(observation)).item<bool>() ||
      !all(logical_and(is_finite(scale), greater(scale, 0))).item<bool>()) {
    throw std::invalid_argument("Gaussian mixture likelihood requires compatible finite parameters");
  }
  const auto residual = (observation - location) / scale;
  const auto component = -.91893853320467274178 - log(scale) - .5 * square(residual);
  return -mean(logsumexp(log_softmax(logits, 1) + component, {1}));
}

/// @brief Normal-inverse-gamma parameters for one continuous response per row.
struct EvidentialRegressionPrediction {
  Tensor mean;  ///< Conditional location, shape [rows, 1].
  Tensor nu;    ///< Positive multiplier of precision for the conditional mean.
  Tensor alpha; ///< Inverse-gamma shape, greater than one.
  Tensor beta;  ///< Positive inverse-gamma scale.
};

/// @brief Transform four affine outputs into normal-inverse-gamma parameters.
/// @param output Floating matrix [rows, 4] in mean, nu, alpha, beta order.
/// @return Differentiable parameters, with a small floor on each constrained output.
/// @throws std::invalid_argument Invalid shape or dtype.
/// @code
/// auto evidence = dm::nn::EvidentialRegressionHead(dm::nn::zeros({4, 4}));
/// @endcode
inline EvidentialRegressionPrediction EvidentialRegressionHead(const Tensor& output) {
  if (!output.defined() || output.dim() != 2 || output.size(0) <= 0 || output.size(1) != 4 ||
      (output.dtype() != DType::Float32 && output.dtype() != DType::Float64)) {
    throw std::invalid_argument("evidential regression requires a nonempty floating matrix with four columns");
  }
  const auto parts = split(output, 4, 1);
  return {parts[0], softplus(parts[1]) + .001, softplus(parts[2]) + 1.001, softplus(parts[3]) + .001};
}

/// @brief Average marginal Student-t NLL with an optional residual-weighted evidence penalty.
/// @param prediction Finite coefficient tensors shaped like the observation.
/// @param observation Observed values with shape matching every parameter tensor.
/// @param penalty Nonnegative multiplier for abs(residual) * (2 * nu + alpha).
/// @return Differentiable scalar objective.
/// @throws std::invalid_argument Incompatible or invalid parameters, observations or penalty.
/// @code
/// auto target = dm::nn::zeros({4, 1});
/// auto p = dm::nn::EvidentialRegressionHead(dm::nn::zeros({4, 4}));
/// auto loss = dm::nn::EvidentialRegressionLoss(p, target, .02);
/// @endcode
inline Tensor EvidentialRegressionLoss(
    const EvidentialRegressionPrediction& prediction, const Tensor& observation, double penalty = 0.) {
  const auto& [location, nu, alpha, beta] = prediction;
  if (!observation.defined() || observation.numel() <= 0 ||
      (observation.dtype() != DType::Float32 && observation.dtype() != DType::Float64) ||
      !std::isfinite(penalty) || penalty < 0) {
    throw std::invalid_argument("invalid evidential observation or penalty");
  }
  for (const auto* value : {&location, &nu, &alpha, &beta}) {
    if (!value->defined() || value->shape() != observation.shape() ||
        value->dtype() != observation.dtype() || !all(is_finite(*value)).item<bool>()) {
      throw std::invalid_argument("invalid evidential regression parameter");
    }
  }
  if (!all(is_finite(observation)).item<bool>() || !all(greater(nu, 0)).item<bool>() ||
      !all(greater(alpha, 1)).item<bool>() || !all(greater(beta, 0)).item<bool>()) {
    throw std::invalid_argument("evidential regression parameters must have positive scale and alpha above one");
  }
  const auto error = observation - location;
  const auto twice_scale = 2. * beta * (1. + nu);
  const auto alpha_half = alpha + .5;
  const auto nll = .5 * log(std::numbers::pi * twice_scale / nu) +
                   alpha_half * log1p(nu * square(error) / twice_scale) +
                   lgamma(alpha) - lgamma(alpha_half);
  return penalty == 0. ? mean(nll) : mean(nll + penalty * abs(error) * (2. * nu + alpha));
}

/// @brief Predictive variance components for the normal-inverse-gamma model.
struct EvidentialRegressionVariance {
  Tensor observation; ///< Expected observation variance beta / (alpha - 1).
  Tensor mean;        ///< Variance of the conditional mean, observation variance divided by nu.
};

/// @brief Decompose predictive variance under valid normal-inverse-gamma parameters.
/// @param prediction Valid prediction with nu > 0, beta > 0 and alpha > 1.
/// @return Differentiable expected observation variance and variance of the conditional mean.
inline EvidentialRegressionVariance VarianceComponents(const EvidentialRegressionPrediction& prediction) {
  const auto observation = prediction.beta / (prediction.alpha - 1.);
  return {observation, observation / prediction.nu};
}

/// @brief Calibrate a split-conformal radius from out-of-fit absolute errors.
/// The one-based rank is ceil((n + 1) * (1 - alpha)). An unattainable rank
/// yields an infinite radius. Exchangeability gives marginal coverage for
/// a fixed predictor when calibration and test observations are exchangeable.
/// @param prediction CPU calibration estimates, shaped like observation.
/// @param observation CPU calibration observations, finite and nonempty.
/// @param alpha Target miscoverage probability in (0, 1).
/// @return Nonnegative radius, possibly positive infinity.
/// @throws std::invalid_argument Invalid shapes, data or miscoverage.
/// @code
/// namespace n = dm::nn;
/// auto q = n::SplitConformalRadius(n::zeros({3}), n::tensor({1., 3., 2.}, {3}), .5);
/// @endcode
inline double SplitConformalRadius(
    const Tensor& prediction, const Tensor& observation, double alpha) {
  if (!prediction.defined() || !observation.defined() ||
      prediction.device().backend != compute::Backend::CPU ||
      observation.device().backend != compute::Backend::CPU ||
      prediction.shape() != observation.shape() || observation.numel() <= 0 ||
      !(alpha > 0.0 && alpha < 1.0)) {
    throw std::invalid_argument("calibration requires matching nonempty CPU tensors and alpha in (0, 1)");
  }
  const auto fitted = prediction.to_vector<double>();
  auto scores = observation.to_vector<double>();
  for (std::size_t i = 0; i < scores.size(); ++i) {
    scores[i] = std::abs(scores[i] - fitted[i]);
    if (!std::isfinite(scores[i]))
      throw std::invalid_argument("calibration scores must be finite");
  }
  const auto rank = std::size_t(std::ceil((double(scores.size()) + 1.) * (1. - alpha)));
  if (rank > scores.size())
    return std::numeric_limits<double>::infinity();
  const auto selected = scores.begin() + (rank - 1);
  std::nth_element(scores.begin(), selected, scores.end());
  return *selected;
}

}  // namespace datamunge::nn
