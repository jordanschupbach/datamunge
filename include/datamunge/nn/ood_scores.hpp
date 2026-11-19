#pragma once

/// @file ood_scores.hpp
/// @brief Confidence scores for out-of-distribution evaluation.

#include <cmath>
#include <stdexcept>

#include "datamunge/nn/autograd.hpp"
#include "datamunge/nn/ops.hpp"

namespace datamunge::nn {

/// @brief Scores and diagnostic tensors from one OOD scoring batch.
/// @par Example
/// @code
/// dm::nn::OodScores output;
/// @endcode
struct OodScores {
  Tensor scores; ///< [batch,3], maximum softmax, ODIN and negative energy.
  Tensor logits; ///< Original class logits.
  Tensor gradient; ///< Input derivative of tempered predicted-class log confidence.
  Tensor shifted; ///< Clipped ODIN inputs.
  Tensor tempered; ///< Log probabilities at shifted inputs after temperature scaling.
};

/// @brief Compute maximum softmax, ODIN and negative energy scores.
/// Larger scores indicate stronger support for the training distribution.
/// Negative energy uses temperature one, independent of the ODIN temperature.
/// @tparam Predict Differentiable row-wise batch classifier.
/// @param input Floating input batch, rank at least two.
/// @param predict Callable returning logits [batch,classes].
/// @param temperature Positive ODIN logit temperature.
/// @param step Nonnegative ODIN input displacement.
/// @param lower Finite lower clipping bound, default zero.
/// @param upper Finite upper clipping bound, default one.
/// @return Confidence scores and intermediate diagnostics.
/// @throws std::invalid_argument For invalid shape or scalar options.
/// @par Example
/// @code
/// auto result=dm::nn::ood_confidence_scores(dm::nn::ones({1,2}),
///     [](const auto& x){return x*2;},20,.015);
/// @endcode
/// @par References
/// - Liang et al. (2018), Enhancing the Reliability of Out-of-distribution Image Detection. https://arxiv.org/abs/1706.02690
/// - Liu et al. (2020), Energy-based Out-of-distribution Detection. https://arxiv.org/abs/2010.03759
template<class Predict>
OodScores ood_confidence_scores(const Tensor& input,Predict&& predict,double temperature,double step,
                                 double lower=0,double upper=1) {
  if (!input.defined() || input.dim()<2 || input.size(0)<1 || input.numel()<1 ||
      (input.dtype()!=DType::Float32 && input.dtype()!=DType::Float64) ||
      !std::isfinite(temperature) || temperature<=0 || !std::isfinite(step) || step<0 ||
      !std::isfinite(lower) || !std::isfinite(upper) || lower>upper)
    throw std::invalid_argument("Invalid OOD scoring inputs");
  EnableGradGuard recording;
  auto x=input.detach();
  x.requires_grad_(true);
  const auto logits=predict(x);
  if (!logits.defined() || logits.dim()!=2 || logits.size(0)!=x.size(0) || logits.size(1)<2)
    throw std::invalid_argument("Expected batched class logits");
  const auto labels=one_hot(argmax(logits),logits.size(1),1,0,logits.dtype());
  const auto gradient=grad(sum(labels*log_softmax(logits/temperature)),{x})[0];
  NoGradGuard guard;
  const auto shifted=clip(input+step*sign(gradient),lower,upper);
  const auto tempered=log_softmax(predict(shifted)/temperature);
  const auto wide=logits.to(DType::Float64);
  const auto scores=stack({max(softmax(wide),{1}),exp(max(tempered.to(DType::Float64),{1})),
                           logsumexp(wide,{1})},1);
  return {scores,logits.detach(),gradient,shifted,tempered};
}

}  // namespace datamunge::nn
