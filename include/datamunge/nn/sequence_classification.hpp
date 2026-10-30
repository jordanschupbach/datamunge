#pragma once
/// @file sequence_classification.hpp
/// @brief Tensor categorical sequence objectives and equal-observation scoring.
#include "datamunge/nn/autograd.hpp"
#include "datamunge/nn/ops.hpp"
#include "datamunge/nn/training.hpp"
namespace datamunge::nn::models {
namespace nn = datamunge::nn;
/// @brief Compute stable hard-label categorical cross-entropy with tensors.
///
/// First builds an exact one-hot target in logits dtype. No probabilities
/// are clipped and no optimizer,normalization or state mutation is hidden.
///
/// @param logits Floating[N,C] raw unbounded scores,N>0,C>=2.
/// @param labels Int32/64[N] bounded0..C-1 true class identities.
/// @return Differentiable scalar equal-observation mean negative log softmax.
/// @throws std::invalid_argument For invalid geometry,dtype or label range.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto x = n::tensor({1000., 999., 998., 0.}, {2, 2}).requires_grad_();
/// const auto labels = n::tensor({0, 1}).to(n::DType::Int32);
/// const auto loss = m::SequenceCrossEntropy(x, labels);
/// @endcode
/// @see ScoreSequenceClassifier()

inline nn::Tensor SequenceCrossEntropy(const nn::Tensor &logits, const nn::Tensor &labels) {
  if (logits.dim() != 2 || logits.size(0) < 1 || logits.size(1) < 2 ||
      labels.shape() != nn::Shape{logits.size(0)} ||
      (labels.dtype() != nn::DType::Int32 && labels.dtype() != nn::DType::Int64) ||
      !nn::all(nn::logical_and(nn::greater_equal(labels, 0.),
                               nn::less(labels, double(logits.size(1)))))
           .item<bool>())
    throw std::invalid_argument(
        "aligned categorical logits and bounded integer labels required");
  return -nn::mean(nn::sum(nn::one_hot(labels, logits.size(1), 1., 0., logits.dtype()) *
                               nn::log_softmax(logits, 1),
                           {1}));
}
/// @brief Own tensor-only equal-observation categorical classifier diagnostics.
///
/// All fields are detached tensors,so callers can plot without numeric host
/// loops. Confusion rows are true class,columns predicted class.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto s = m::ScoreSequenceClassifier(n::tensor({2., 1., 0., 3.}, {2, 2}),
///                                           n::tensor({0, 1}).to(n::DType::Int32));
/// @endcode
/// @see ScoreSequenceClassifier()
struct SequenceClassifierScore {
  nn::Tensor accuracy;  ///< Double scalar correct fraction,first class wins ties.
  nn::Tensor nll;       ///< Double scalar stable mean raw-logit cross-entropy.
  nn::Tensor confusion; ///< Int64[C,C] true-row/predicted-column counts.
  nn::Tensor predicted; ///< Int64[N] first-maximum class IDs.
};
/// @brief Score raw class logits with tensor accuracy,stable NLL and full confusion counts.
///
/// Each observation has equal weight. Stable Double log-softmax retains the
/// input floating values;argmax resolves ties toward the first class. No fit,
/// class reweighting,hidden threshold search or numeric host prediction loop.
///
/// @param logits Finite floating[N,C] raw class scores,N>0,C>=2.
/// @param labels Aligned bounded Int32/64[N] original true labels.
/// @return Detached Double accuracy/NLL,Int64 confusion and predicted IDs.
/// @throws std::invalid_argument For invalid logits,label contract or nonfinite score values.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto s = m::ScoreSequenceClassifier(n::tensor({2., 1., 0., 3.}, {2, 2}),
///                                           n::tensor({0, 1}).to(n::DType::Int32));
/// @endcode
/// @see SequenceCrossEntropy()

inline SequenceClassifierScore ScoreSequenceClassifier(const nn::Tensor &logits,
                                                       const nn::Tensor &labels) {
  nn::NoGradGuard guard;
  if (!nn::all(nn::is_finite(logits)).item<bool>())
    throw std::invalid_argument("finite classifier logits required");
  const auto nll = SequenceCrossEntropy(logits.to(nn::DType::Float64), labels),
             predicted = nn::argmax(logits, 1),
             truth = nn::one_hot(labels, logits.size(1), 1., 0., nn::DType::Float64),
             guess = nn::one_hot(predicted, logits.size(1), 1., 0., nn::DType::Float64);
  return {nn::mean(nn::equal(predicted, labels).to(nn::DType::Float64)), nll,
          nn::matmul(nn::transpose(truth, {1, 0}), guess).to(nn::DType::Int64), predicted};
}
} // namespace datamunge::nn::models
