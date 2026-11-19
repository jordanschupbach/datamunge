/// @file
/// @brief Student-t clustering assignments and fixed-target refinement.
#pragma once

#include <datamunge/nn/ops.hpp>
#include <datamunge/nn/losses.hpp>
#include <cmath>

namespace datamunge::nn {

/// @brief Compute differentiable, row-normalized Student-t affinities with one degree of freedom.
///
/// @param latent Finite floating observations [N,D], with N and D positive.
/// @param centers Finite floating cluster centers [K,D], with K positive.
/// @return Assignment probabilities [N,K].
/// @throws std::invalid_argument For invalid shapes or dtypes.
/// @par Example
/// @code
/// const auto q = dm::nn::dec_assignments(dm::nn::zeros({4,2}), dm::nn::ones({3,2}));
/// @endcode
/// @par References
/// - Xie et al. (2016), https://proceedings.mlr.press/v48/xieb16.html
/// @see dec_targets(), dec_kl()
inline Tensor dec_assignments(const Tensor &latent, const Tensor &centers) {
  if (latent.dim() != 2 || centers.dim() != 2 || latent.shape()[0] < 1 ||
      centers.shape()[0] < 1 || latent.shape()[1] < 1 ||
      latent.shape()[1] != centers.shape()[1] || !is_floating(latent.dtype()) ||
      !is_floating(centers.dtype()))
    throw std::invalid_argument("dec_assignments: incompatible observations and centers");
  const auto delta = latent.reshape({latent.shape()[0], 1, latent.shape()[1]}) - centers;
  return softmax(-log1p(sum(square(delta), {2})), 1);
}

/// @brief Sharpen and frequency-correct DEC probabilities for a fixed training target.
///
/// Square each entry, divide by its original column mass, and normalize each
/// row to sum to one. The result is detached. Refresh it from the full
/// training set between refinement rounds.
///
/// @param assignments Strictly positive probability matrix [N,K] from dec_assignments().
/// @return Detached target distribution [N,K].
/// @throws std::invalid_argument For invalid shape or dtype.
/// @par Example
/// @code
/// const auto q = dm::nn::dec_assignments(dm::nn::zeros({4,2}), dm::nn::ones({3,2}));
/// const auto p = dm::nn::dec_targets(q);
/// @endcode
/// @par References
/// - Xie et al. (2016), https://proceedings.mlr.press/v48/xieb16.html
/// @see dec_assignments(), dec_kl()
inline Tensor dec_targets(const Tensor &assignments) {
  if (assignments.dim() != 2 || assignments.shape()[0] < 1 || assignments.shape()[1] < 1 ||
      !is_floating(assignments.dtype()))
    throw std::invalid_argument("dec_targets: expected a nonempty probability matrix");
  NoGradGuard guard;
  return softmax(2. * log(assignments) - log(sum(assignments, {0}, true)), 1);
}

/// @brief Compute mean KL divergence from a fixed target to current DEC assignments.
///
/// @param target Strictly positive target probability matrix [N,K].
/// @param assignments Strictly positive matching probability matrix [N,K].
/// @return Scalar KL divergence averaged across rows, with gradients through assignments.
/// @throws std::invalid_argument For incompatible shapes or dtypes.
/// @par Example
/// @code
/// const auto q = dm::nn::dec_assignments(dm::nn::zeros({4,2}), dm::nn::ones({3,2}));
/// const auto loss = dm::nn::dec_kl(dm::nn::dec_targets(q), q);
/// @endcode
/// @par References
/// - Xie et al. (2016), https://proceedings.mlr.press/v48/xieb16.html
/// @see dec_targets()
inline Tensor dec_kl(const Tensor &target, const Tensor &assignments) {
  if (target.dim() != 2 || target.shape() != assignments.shape() || target.shape()[0] < 1 ||
      target.shape()[1] < 1 || !is_floating(target.dtype()) || !is_floating(assignments.dtype()))
    throw std::invalid_argument("dec_kl: expected matching probability matrices");
  const auto p = target.detach();
  return mean(sum(p * (log(p) - log(assignments)), {1}));
}

/// @brief Give every occupied cluster unit total weight.
///
/// Cluster membership is fixed for a classification round. Empty clusters
/// have no sampled rows and never enter the reciprocal calculation.
///
/// @param labels Integral [N] pseudo-labels in [0,clusters), with N positive.
/// @param clusters Positive number of cluster IDs.
/// @return Detached Float32 weights [N], each reciprocal to its cluster size.
/// @throws std::invalid_argument For invalid labels or cluster count.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto w = n::cluster_balanced_weights(n::arange(0, 3, 1, n::DType::Int64), 4);
/// @endcode
/// @par References
/// - Caron et al. (2018), https://arxiv.org/abs/1807.05520
/// @see deepcluster_loss()
inline Tensor cluster_balanced_weights(const Tensor &labels, int clusters) {
  if (labels.dim() != 1 || labels.shape()[0] < 1 || !is_integral(labels.dtype()) ||
      clusters < 1 || min(labels).item<std::int64_t>() < 0 ||
      max(labels).item<std::int64_t>() >= clusters)
    throw std::invalid_argument("cluster_balanced_weights: invalid labels or cluster count");
  NoGradGuard guard;
  const auto counts = sum(one_hot(labels, clusters, 1, 0, DType::Int64), {0});
  return 1. / gather(counts, labels).to(DType::Float32);
}

/// @brief Score DeepCluster pseudo-labels by normalized weighted cross-entropy.
///
/// @param logits Floating class scores [N,K], with N and K positive.
/// @param labels Integral cluster IDs [N] in [0,K).
/// @param weights Nonnegative finite floating weights [N] with positive total mass.
/// @return Scalar loss with gradients through logits only.
/// @throws std::invalid_argument For invalid shapes or types, labels or weight mass.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto ids = n::arange(0, 3, 1, n::DType::Int64);
/// const auto loss = n::deepcluster_loss(n::zeros({3,4}), ids, n::cluster_balanced_weights(ids,4));
/// @endcode
/// @par References
/// - Caron et al. (2018), https://arxiv.org/abs/1807.05520
/// @see cluster_balanced_weights()
inline Tensor deepcluster_loss(const Tensor &logits, const Tensor &labels, const Tensor &weights) {
  if (logits.dim() != 2 || logits.shape()[0] < 1 || logits.shape()[1] < 1 ||
      !is_floating(logits.dtype()) || labels.shape() != Shape{logits.shape()[0]} ||
      weights.shape() != labels.shape() || !is_integral(labels.dtype()) || !is_floating(weights.dtype()) ||
      min(labels).item<std::int64_t>() < 0 || max(labels).item<std::int64_t>() >= logits.shape()[1] ||
      !all(is_finite(weights)).item<bool>() || min(weights).item() < 0.)
    throw std::invalid_argument("deepcluster_loss: invalid pseudo-label classification batch");
  const auto fixed = weights.detach();
  const auto mass = sum(fixed);
  if (!(mass.item() > 0.) || !std::isfinite(mass.item()))
    throw std::invalid_argument("deepcluster_loss: positive finite weight mass required");
  const auto nll = sparse_softmax_cross_entropy(logits, labels, Reduction::None);
  return sum(fixed * nll) / mass;
}

}  // namespace datamunge::nn
