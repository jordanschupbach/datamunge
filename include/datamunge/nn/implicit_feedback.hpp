/// @file
/// @brief Differentiable objectives for implicit-feedback recommendation.

#pragma once

#include <cmath>
#include <stdexcept>

#include "datamunge/nn/ops.hpp"

namespace datamunge::nn {

/// @brief Mean confidence-weighted squared error for implicit feedback.
///
/// A positive count has target one. Zero count has target zero.
/// Confidence is one plus alpha times the count. The mean includes
/// unobserved entries, and gradients propagate to the scores.
/// @param scores Nonempty floating-point [users, items] score matrix.
/// @param counts Finite nonnegative interaction counts of the same shape and device.
/// @param alpha Finite nonnegative multiplier, default 3.
/// @return Scalar weighted squared loss, without regularization.
/// @throws std::invalid_argument Undefined inputs, invalid shapes, dtype or multiplier.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto scores = n::zeros({1,2});
/// const auto counts = n::from_vector<float>({1,0}, {1,2});
/// std::print("{:.1f}\n", n::implicit_feedback_loss(scores,counts).item()); // 2.0
/// @endcode
/// @par References
/// - Hu, Koren and Volinsky (2008), Collaborative Filtering for Implicit Feedback Datasets.
///   https://doi.org/10.1109/ICDM.2008.22
/// @see bayesian_pairwise_ranking_loss()
inline Tensor implicit_feedback_loss(const Tensor& scores, const Tensor& counts, double alpha = 3.0) {
  if (!scores.defined() || !counts.defined() || scores.dim() != 2 || scores.numel() == 0 ||
      scores.shape() != counts.shape() || !is_floating(scores.dtype()) ||
      !(alpha >= 0.0 && std::isfinite(alpha))) {
    throw std::invalid_argument("implicit_feedback_loss: invalid scores, counts or multiplier");
  }
  const auto observations = counts.detach().to(scores.dtype());
  const auto preference = greater(observations, 0.0).to(scores.dtype());
  return mean((1.0 + alpha * observations) * square(scores - preference));
}

/// @brief Mean negative log probability that positives outrank sampled negatives.
///
/// Computes softplus of the negative score minus the positive score,
/// elementwise, then averages. The caller chooses the negative sampling
/// scheme and can add regularization separately.
/// @param positive Nonempty floating tensor of observed-item scores.
/// @param negative Same-shaped tensor of sampled unobserved-item scores,
/// with the same dtype and device.
/// @return Scalar ranking loss differentiable with respect to both inputs.
/// @throws std::invalid_argument Undefined, empty or incompatible scores.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto scores = n::zeros({2});
/// std::print("{:.3f}\n", n::bayesian_pairwise_ranking_loss(scores,scores).item()); // 0.693
/// @endcode
/// @par References
/// - Rendle et al. (2009), BPR: Bayesian Personalized Ranking from Implicit Feedback.
///   https://arxiv.org/abs/1205.2618
/// @see implicit_feedback_loss(), softplus()
inline Tensor bayesian_pairwise_ranking_loss(const Tensor& positive, const Tensor& negative) {
  if (!positive.defined() || !negative.defined() || positive.numel() == 0 ||
      positive.shape() != negative.shape() || positive.dtype() != negative.dtype() ||
      !is_floating(positive.dtype())) {
    throw std::invalid_argument("bayesian_pairwise_ranking_loss: incompatible scores");
  }
  return mean(softplus(negative - positive));
}

/// @brief Compute the full-catalog TOP1 next-item ranking loss.
///
/// For every row, compare each negative score to the positive score with
/// sigmoid(negative - positive). Add sigmoid(negative squared) to shrink
/// negatives towards zero. Average over rows and their negative items.
/// @param scores Floating-point logits [examples, items] with at least two items.
/// @param positive One-hot positive mask with the same shape and device.
/// @return Scalar loss differentiable with respect to scores.
/// @throws std::invalid_argument Undefined inputs or incompatible geometry or dtype.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto score = n::zeros({1,2});
/// const auto label = n::from_vector<float>({1,0}, {1,2});
/// std::print("{:.1f}\n", n::top1_ranking_loss(score,label).item()); // 1.0
/// @endcode
/// @par References
/// - Hidasi et al. (2016), Session-based Recommendations with Recurrent Neural Networks.
///   https://arxiv.org/abs/1511.06939
/// @see bayesian_pairwise_ranking_loss()
inline Tensor top1_ranking_loss(const Tensor& scores, const Tensor& positive) {
  if (!scores.defined() || !positive.defined() || scores.dim() != 2 ||
      scores.size(0) == 0 || scores.size(1) < 2 ||
      scores.shape() != positive.shape() || !is_floating(scores.dtype())) {
    throw std::invalid_argument("top1_ranking_loss: incompatible scores and mask");
  }
  const auto mask = positive.detach().to(scores.dtype());
  const auto chosen = sum(scores * mask, {1}, true);
  const auto terms = sigmoid(scores - chosen) + sigmoid(square(scores));
  return (double(scores.size(1)) / (scores.size(1)-1)) * mean((1.0-mask) * terms);
}

} // namespace datamunge::nn
