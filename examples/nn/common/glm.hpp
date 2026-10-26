// datamunge::nn port of tflow_examples src/common/ml/glm.h.
// Log-link generalized linear model negative log likelihoods (float32).
// Poisson: Var(Y)=mu. Gamma: fixed shape a, Var(Y)=mu^2/a, strictly positive Y.
// Negative binomial: fixed size r, Var(Y)=mu+mu^2/r, nonnegative integer Y.
// eta=log(mu). Gamma shape and NB size are supplied, not estimated. Response-
// only normalization terms use stop_gradient (as in TF, where Lgamma had no
// C++ gradient), retaining the exact reported NLL. NB uses softplus to evaluate
// log(1+exp(eta-log(r))) stably. Poisson/Gamma still require finite exp(eta).
#pragma once

#include <cmath>
#include <datamunge/nn/nn.hpp>
#include <stdexcept>

namespace ml {
namespace nn = datamunge::nn;

enum class GlmFamily { kPoisson, kGamma, kNegativeBinomial };

inline nn::Tensor GlmNll(const nn::Tensor& eta, const nn::Tensor& y, GlmFamily family, float shape = 1.f) {
  if (!std::isfinite(shape) || shape <= 0) throw std::invalid_argument("positive GLM shape required");
  if (family == GlmFamily::kPoisson) return nn::exp(eta) - y * eta + nn::stop_gradient(nn::lgamma(y + 1.0));
  if (family == GlmFamily::kGamma) {
    const auto constant = nn::stop_gradient(double(std::lgamma(shape) - shape * std::log(shape)) -
                                            double(shape - 1.f) * nn::log(y));
    return constant + double(shape) * (eta + y * nn::exp(-eta));
  }
  const auto log_ratio = eta - double(std::log(shape));
  const auto constant =
      nn::stop_gradient(double(std::lgamma(shape)) + nn::lgamma(y + 1.0) - nn::lgamma(y + double(shape)));
  return constant + ((y + double(shape)) * nn::softplus(log_ratio) - y * log_ratio);
}

}  // namespace ml
