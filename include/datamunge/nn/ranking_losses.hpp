#pragma once

/// @file ranking_losses.hpp
/// @brief Query-grouped ranking objectives and NDCG diagnostics.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include "datamunge/nn/ops.hpp"

namespace datamunge::nn {

/// @brief Internal stable ordering and discounted-gain operations.
namespace ranking_detail {

/// @brief Check common ranking geometry.
/// @param scores Scores [N,D].
/// @param grades Relevance grades [N,D].
inline void validate(const Tensor& scores, const Tensor& grades) {
  if (!scores.defined() || !grades.defined() || scores.dim() != 2 ||
      scores.shape() != grades.shape() || scores.size(0) == 0 || scores.size(1) == 0)
    throw std::invalid_argument("ranking requires matching nonempty [queries,docs] tensors");
}

/// @brief Stable descending order, keeping the original order for ties.
/// @param values A finite query row.
/// @return Ordered document indices.
inline std::vector<std::size_t> order(std::span<const double> values) {
  std::vector<std::size_t> ids(values.size());
  if (!std::ranges::all_of(values, [](double value) {return std::isfinite(value);}))
    throw std::invalid_argument("ranking values must be finite");
  std::iota(ids.begin(), ids.end(), 0);
  std::stable_sort(ids.begin(), ids.end(), [&](std::size_t a, std::size_t b) {
    return values[a] > values[b];
  });
  return ids;
}

/// @brief Discounted cumulative gain of one ranked list.
/// @param gain Nonnegative gain for each document.
/// @param ranked Document indices in rank order.
/// @param discount Discount factor per rank.
/// @return Sum of discounted gains.
inline double dcg(std::span<const double> gain, std::span<const std::size_t> ranked,
                  std::span<const double> discount) {
  double value = 0.;
  for (std::size_t r = 0; r < ranked.size(); ++r)
    value += gain[ranked[r]] * discount[r];
  return value;
}

}  // namespace ranking_detail

/// @brief Compute full-list NDCG and pairwise logistic weights query by query.
///
/// Ties retain document order. An all-zero query contributes zero weights
/// and is excluded from the mean NDCG. LambdaRank weights use absolute
/// NDCG changes from swapping two documents in the current score order.
/// The weights are detached, as required by the LambdaRank update.
/// @param scores Finite score matrix [N,D].
/// @param grades Finite nonnegative relevance grades [N,D] with representable exponential gains.
/// @param lambda True for NDCG-swap weights, false for unit RankNet preferences.
/// @return Mean NDCG (zero if no positive queries) and weights [N,D,D].
/// @throws std::invalid_argument Invalid geometry, scores or grades.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// auto [ndcg, weights] = n::ranking_weights(n::tensor({{2.,1.}}),n::tensor({{1.,0.}}),true);
/// // NDCG is one.
/// @endcode
/// @par References
/// - Burges (2010), From RankNet to LambdaRank to LambdaMART.
///   https://www.microsoft.com/en-us/research/publication/from-ranknet-to-lambdarank-to-lambdamart-an-overview/
inline std::pair<double,Tensor> ranking_weights(const Tensor& scores, const Tensor& grades, bool lambda) {
  ranking_detail::validate(scores,grades);
  const auto n = scores.size(0), d = scores.size(1);
  const auto s = scores.to_vector<double>(), g = grades.to_vector<double>();
  std::vector<double> discount(d), weights;
  for (std::int64_t j = 0; j < d; ++j) discount[j] = 1./std::log2(j+2.);
  double total = 0.;
  std::int64_t used = 0;
  for (std::int64_t row = 0; row < n; ++row) {
    const std::span<const double> label(g.data()+row*d,d), score(s.data()+row*d,d);
    const auto ideal_order = ranking_detail::order(label), predicted_order = ranking_detail::order(score);
    std::vector<double> gain(d);
    for (std::int64_t j = 0; j < d; ++j) {
      gain[j] = std::exp2(label[j])-1.;
      if (label[j] < 0. || !std::isfinite(gain[j]))
        throw std::invalid_argument("invalid ranking relevance grade");
    }
    const auto ideal = ranking_detail::dcg(gain,ideal_order,discount);
    if (!std::isfinite(ideal)) throw std::invalid_argument("ranking gain sum overflow");
    if (ideal > 0.) {
      total += ranking_detail::dcg(gain,predicted_order,discount)/ideal;
      ++used;
    }
    std::vector<std::size_t> rank(d);
    for (std::int64_t j = 0; j < d; ++j) rank[predicted_order[j]] = j;
    for (std::int64_t i = 0; i < d; ++i)
      for (std::int64_t j = 0; j < d; ++j)
        weights.push_back(label[i] <= label[j] || ideal == 0. ? 0. : lambda
            ? ((gain[i]-gain[j])/ideal)*std::abs(discount[rank[i]]-discount[rank[j]]) : 1.);
  }
  return {used ? total/used : 0., from_vector(weights,{n,d,d}).to(scores.dtype()).to(scores.context())};
}

/// @brief Average the summed pairwise logistic loss over queries.
/// @param scores Score matrix [N,D].
/// @param weights Nonnegative preference weights [N,D,D], ordered preferred then other.
/// @return Differentiable scalar RankNet or LambdaRank objective with fixed weights.
/// @throws std::invalid_argument Invalid geometry.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// auto scores=n::zeros({1,2});
/// auto weights=n::ranking_weights(scores,n::tensor({{1.,0.}}),false).second;
/// auto loss=n::pairwise_ranking_loss(scores,weights); // log(2)
/// @endcode
/// @see ranking_weights()
inline Tensor pairwise_ranking_loss(const Tensor& scores, const Tensor& weights) {
  if (!scores.defined() || scores.dim() != 2 || scores.size(0) == 0 || scores.size(1) == 0 ||
      !weights.defined() || weights.shape() != Shape{scores.size(0),scores.size(1),scores.size(1)})
    throw std::invalid_argument("invalid pairwise ranking geometry");
  const auto difference = expand_dims(scores,2)-expand_dims(scores,1);
  return mean(sum(weights.detach()*softplus(-difference),{1,2}));
}

/// @brief Average the ListNet top-one cross entropy over queries.
/// @param scores Predicted logits [N,D].
/// @param grades Relevance logits [N,D] defining a softmax target.
/// @return Differentiable scalar cross entropy.
/// @throws std::invalid_argument Invalid geometry.
/// @par Example
/// @code
/// auto loss=dm::nn::listnet_loss(dm::nn::zeros({1,2}),dm::nn::ones({1,2})); // log(2)
/// @endcode
/// @par References
/// - Cao et al. (2007), Learning to Rank: From Pairwise Approach to Listwise Approach.
///   https://doi.org/10.1145/1273496.1273513
inline Tensor listnet_loss(const Tensor& scores, const Tensor& grades) {
  ranking_detail::validate(scores,grades);
  return -mean(sum(softmax(grades.detach())*log_softmax(scores),{1}));
}
}  // namespace datamunge::nn
