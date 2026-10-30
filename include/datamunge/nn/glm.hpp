// datamunge::nn port of tflow_examples src/common/ml/glm.h.
// Log-link generalized linear model negative log likelihoods (float32).
// Poisson: Var(Y)=mu. Gamma: fixed shape a, Var(Y)=mu^2/a, strictly positive Y.
// Negative binomial: fixed size r, Var(Y)=mu+mu^2/r, nonnegative integer Y.
// eta=log(mu). Gamma shape and NB size are supplied, not estimated. Response-
// only normalization terms use stop_gradient (as in TF, where Lgamma had no
// C++ gradient), retaining the exact reported NLL. NB uses softplus to evaluate
// log(1+exp(eta-log(r))) stably. Poisson/Gamma still require finite exp(eta).
#pragma once

/// @file glm.hpp
/// @brief Differentiable log-link likelihoods with fixed dispersion.

#include <cmath>
#include <datamunge/nn/autograd.hpp>
#include <datamunge/nn/ops.hpp>
#include <stdexcept>

namespace datamunge::nn::ml {
namespace nn = datamunge::nn;

/// @brief Log-link likelihood family; gamma shape and negative-binomial size are fixed.
enum class GlmFamily {
  kPoisson,         ///< Count mean equals variance.
  kGamma,           ///< Positive response, variance mu squared / shape.
  kNegativeBinomial ///< Count variance mu + mu squared / size.
};

/// @brief Compute elementwise log-link negative log likelihood with response constants.
///
/// Mean is exp(eta); gamma also uses the log link, unlike stats::GLM.
/// Response normalising terms are stop-gradient. Callers validate response support,
/// finite data and the family enum; unknown enum values follow the NB branch.
/// Poisson/gamma require finite exponentials; NB uses a stable softplus expression.
/// @param eta Floating log means, broadcast-compatible with y.
/// @param y Valid responses: gamma positive, count families nonnegative integers.
/// @param family Poisson, gamma or negative-binomial enum.
/// @param shape Positive finite gamma shape or NB size, default 1; checked for all families.
/// @return Broadcast per-observation losses, differentiable with respect to eta.
/// @throws std::invalid_argument On invalid shape or incompatible tensor operations.
///
/// @par Example
/// @code
/// namespace n=dm::nn; namespace m=n::ml;
/// auto eta=n::scalar(0,n::DType::Float64).requires_grad_();
/// const auto loss=m::GlmNll(eta,n::scalar(2,n::DType::Float64),m::GlmFamily::kPoisson);
/// const auto score=n::grad(loss,{eta})[0]; // -1
/// @endcode
///
/// @par References
/// - Stan Functions Reference,
/// https://mc-stan.org/docs/functions-reference/unbounded_discrete_distributions.html
/// @see dm::stats::GLM
inline nn::Tensor
GlmNll(const nn::Tensor& eta, const nn::Tensor& y, GlmFamily family, double shape = 1.0) {
  if (!std::isfinite(shape) || shape <= 0)
    throw std::invalid_argument("positive GLM shape required");
  if (family == GlmFamily::kPoisson)
    return nn::exp(eta) - y * eta + nn::stop_gradient(nn::lgamma(y + 1.0));
  if (family == GlmFamily::kGamma) {
    const auto constant = nn::stop_gradient(
        double(std::lgamma(shape) - shape * std::log(shape)) - double(shape - 1.0) * nn::log(y));
    return constant + double(shape) * (eta + y * nn::exp(-eta));
  }
  const auto log_ratio = eta - double(std::log(shape));
  const auto constant  = nn::stop_gradient(
      double(std::lgamma(shape)) + nn::lgamma(y + 1.0) - nn::lgamma(y + double(shape)));
  return constant + ((y + double(shape)) * nn::softplus(log_ratio) - y * log_ratio);
}

} // namespace datamunge::nn::ml
