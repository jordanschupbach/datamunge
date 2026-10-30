#pragma once

/// @file classical_inference.hpp
/// @brief Differentiable HMM evidence and tensor nearest-neighbour inference.

#include "datamunge/nn/ops.hpp"
#include <stdexcept>

namespace datamunge::nn::ml {

/// @brief Sum all hidden paths with a log-space tensor forward recursion.
///
/// Inputs are finite log potentials, not necessarily normalised probabilities.
/// Normalise learnable initial/transition/emission logits with log_softmax
/// before calling to obtain a generative likelihood. All floating inputs must
/// have the same dtype. Gradients of the summed result with respect to evidence
/// are smoothed state posteriors. No inputs are mutated; outputs own storage.
/// Work is O(B T K squared), with an O(B K squared) intermediate per step.
/// @param initial Log initial masses [K].
/// @param transition Log transition masses [K,K], predecessor rows.
/// @param evidence Log emission evidence [B,T,K], B,T,K positive.
/// @return Log partition/sequence likelihood [B], differentiable in all inputs.
/// @throws std::invalid_argument For incompatible shapes, dtypes or empty axes.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto p = n::log(n::tensor({.5, .5}, {2}));
/// const auto a = n::log(n::full({2, 2}, .5));
/// const auto e = n::log(n::full({1, 3, 2}, .25));
/// const auto likelihood = n::exp(n::ml::hmm_log_likelihood(p, a, e)); // 1/64
/// @endcode
///
/// @par References
/// - Rabiner (1989), A tutorial on hidden Markov models and selected applications
/// in speech recognition, https://doi.org/10.1109/5.18626
/// @see log_softmax(), grad()
inline Tensor hmm_log_likelihood(const Tensor& initial, const Tensor& transition,
                                 const Tensor& evidence) {
  if (!initial.defined() || !transition.defined() || !evidence.defined() ||
      initial.dim() != 1 || initial.size(0) < 1 ||
      transition.shape() != Shape{initial.size(0), initial.size(0)} ||
      evidence.dim() != 3 || evidence.size(0) < 1 || evidence.size(1) < 1 ||
      evidence.size(2) != initial.size(0) || !is_floating(initial.dtype()) ||
      initial.dtype() != transition.dtype() || initial.dtype() != evidence.dtype())
    throw std::invalid_argument("HMM requires matching floating [K], [K,K], [B,T,K]");
  auto alpha = initial + select(evidence, 1, 0);
  for (std::int64_t t = 1; t < evidence.size(1); ++t)
    alpha = logsumexp(expand_dims(alpha, 2) + transition, {1}) + select(evidence, 1, t);
  return logsumexp(alpha, {1});
}

/// @brief Fixed-reference nearest neighbours using direct tensor distances.
///
/// No parameters are trained. Constructor computes squared Euclidean distances
/// without cancellation at exact matches, then selects the nearest k indices.
/// Equal distances prefer the lower reference index. In weighted mode, exact
/// matches alone receive equal weight if any exist; otherwise weights are 1/d.
/// Caller supplies finite values; queries and references use the same floating
/// dtype. Work/storage include a [Q,N,D] difference tensor, so batch large queries.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const n::ml::TensorNeighbors near(n::tensor({0, 2}, {2, 1}),
///                                  n::tensor({.5}, {1, 1}), 2, true);
/// const auto prediction = near.regress(n::tensor({0, 10})); // 2.5
/// @endcode
///
/// @par References
/// - Cover and Hart (1967), Nearest neighbor pattern classification,
/// https://doi.org/10.1109/TIT.1967.1053964
/// @see top_k(), one_hot()
struct TensorNeighbors {
  Tensor indices; ///< Int64 selected reference rows [Q,k], increasing distance.
  Tensor squared_distances; ///< Floating selected squared distances [Q,k].
  Tensor weights; ///< Floating nonnegative [Q,k] weights, each row sums to one.

  /// @brief Compute neighbours and normalised weights without fitting a model.
  /// @param references Finite floating [N,D], positive N,D.
  /// @param queries Finite floating [Q,D], positive Q; matching dtype.
  /// @param k Number of neighbours, in [1,N].
  /// @param inverse_distance Use inverse Euclidean distance; default false.
  /// @throws std::invalid_argument For incompatible shapes/dtypes or invalid k.
  /// @par Example
  /// See the class example for weighted regression.
  /// @see regress(), classify()
  TensorNeighbors(const Tensor& references, const Tensor& queries, std::int64_t k,
                  bool inverse_distance = false) {
    if (!references.defined() || !queries.defined() || references.dim() != 2 ||
        queries.dim() != 2 || references.size(0) < 1 || references.size(1) < 1 ||
        queries.size(0) < 1 || queries.size(1) != references.size(1) ||
        !is_floating(references.dtype()) || references.dtype() != queries.dtype() ||
        k < 1 || k > references.size(0))
      throw std::invalid_argument("neighbours require matching [N,D], [Q,D] and k in [1,N]");
    const auto delta = expand_dims(queries, 1) - expand_dims(references, 0);
    const auto nearest = top_k(-sum(square(delta), {2}), k);
    indices = nearest.second;
    squared_distances = -nearest.first;
    weights = ones_like(squared_distances);
    if (inverse_distance) {
      const auto exact = equal(squared_distances, 0.0);
      const auto inverse = reciprocal(sqrt(maximum(squared_distances,
                                                   scalar_like(1e-20, squared_distances))));
      weights = where(any(exact, {1}, true), cast(exact, squared_distances.dtype()), inverse);
    }
    weights = weights / sum(weights, {1}, true);
  }

  /// @brief Average scalar target values with the stored neighbour weights.
  /// @param targets Floating [N] reference targets, same dtype as weights.
  /// @return Floating predictions [Q], differentiable in gathered targets.
  /// @throws std::invalid_argument For incompatible gather/broadcast shapes.
  /// @par Example
  /// See the class example, which predicts 2.5 at x=0.5.
  /// @see TensorNeighbors()
  Tensor regress(const Tensor& targets) const { return sum(gather(targets, indices) * weights, {1}); }

  /// @brief Vote over integer class IDs; equal votes choose the lowest ID.
  /// @param labels Integer reference IDs [N], all in [0,classes).
  /// @param classes Positive number of classes; label range is caller-checked.
  /// @return Int64 predicted class IDs [Q]; the discrete decision has no gradient.
  /// @throws std::invalid_argument For nonpositive classes or incompatible shapes.
  /// @par Example
  /// @code
  /// namespace n = dm::nn;
  /// const n::ml::TensorNeighbors near(n::tensor({0, 2}, {2, 1}),
  ///                                  n::tensor({1}, {1, 1}), 2);
  /// const auto predicted = near.classify(n::from_vector(std::vector<int64_t>{0, 1}), 2); // 0
  /// @endcode
  /// @see regress()
  Tensor classify(const Tensor& labels, std::int64_t classes) const {
    if (classes < 1)
      throw std::invalid_argument("positive class count required");
    const auto votes = one_hot(gather(labels, indices), classes, 1., 0., weights.dtype());
    return argmax(sum(votes * expand_dims(weights, -1), {1}), 1);
  }
};

} // namespace datamunge::nn::ml
