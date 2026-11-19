/// @file
/// @brief Top-k metrics for filtered recommendation rankings.

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <stdexcept>
#include <vector>

#include "datamunge/nn/tensor.hpp"

namespace datamunge::nn {

/// @brief Per-user and macro-averaged metrics from a filtered ranking.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto result = n::recommendation_metrics(n::from_vector<float>({2,1},{1,2}),
///     n::from_vector<int>({0,1},{1,2}), n::from_vector<int>({0,0},{1,2}), {2});
/// std::print("{:.1f}\n", result.averages.to_vector<double>()[2]); // 0.5
/// @endcode
/// @see recommendation_metrics()
struct RecommendationMetrics {
  Tensor ranked; ///< Int64 item IDs [users, max cutoff], padded with -1.
  Tensor per_user; ///< Float64 metrics [users, cutoffs, 3], NaN for excluded users.
  Tensor averages; ///< Float64 macro averages [cutoffs, 3], in the same metric order.
  std::int64_t eligible_users; ///< Number of users with at least one eligible positive item.
};

/// @brief Compute filtered top-k hit rate, graded NDCG, and reciprocal rank.
///
/// Exclude seen items from both the scored and ideal lists. Score ties use
/// ascending zero-based column IDs. A positive grade is relevant. NDCG uses
/// gain 2^grade - 1 and discount 1/log2(rank + 1), for one-based ranks.
/// Users with no eligible positive item do not enter any macro average.
/// The metric order is hit rate, NDCG, reciprocal rank.
/// @param scores Finite Float32 or Float64 CPU matrix [users, items] with positive extents.
/// @param relevance CPU matrix of integer grades in [0,30], same shape as scores.
/// @param seen CPU matrix of finite mask values, same shape as scores. Nonzero means seen.
/// @param cutoffs Nonempty list of positive cutoffs, preserved in input order. Cutoffs
/// larger than the eligible catalog use every eligible item.
/// @return Ranked IDs, metrics by user and cutoff, averages, and number of eligible users.
/// @throws std::invalid_argument Undefined inputs, mismatched shapes, non-CPU tensors,
/// invalid scores, grades, masks or cutoffs, or no eligible users.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto result = n::recommendation_metrics(n::from_vector<float>({2,1},{1,2}),
///     n::from_vector<int>({0,1},{1,2}), n::from_vector<int>({1,0},{1,2}), {1});
/// std::print("{:.1f}\n", result.averages.to_vector<double>()[0]); // 1.0
/// @endcode
/// @par References
/// - Järvelin and Kekäläinen (2002), Cumulated Gain-Based Evaluation of IR Techniques.
///   https://doi.org/10.1145/582415.582418
/// @see RecommendationMetrics, ranking_weights()
inline RecommendationMetrics recommendation_metrics(
    const Tensor& scores, const Tensor& relevance, const Tensor& seen,
    const std::vector<int>& cutoffs) {
  if (!scores.defined() || !relevance.defined() || !seen.defined() ||
      (scores.dtype() != DType::Float32 && scores.dtype() != DType::Float64) ||
      scores.shape().size() != 2 || scores.size(0) <= 0 || scores.size(1) <= 0 ||
      scores.shape() != relevance.shape() || scores.shape() != seen.shape() ||
      cutoffs.empty() || *std::min_element(cutoffs.begin(), cutoffs.end()) <= 0) {
    throw std::invalid_argument("recommendation_metrics: invalid shapes or cutoffs");
  }
  for (const auto* input : {&scores, &relevance, &seen}) {
    if (input->device().backend != compute::Backend::CPU) {
      throw std::invalid_argument("recommendation_metrics: expected CPU tensors");
    }
  }

  const auto users = scores.size(0), items = scores.size(1);
  const std::int64_t max_k = *std::max_element(cutoffs.begin(), cutoffs.end());
  const auto cutoff_count = static_cast<std::int64_t>(cutoffs.size());
  const auto score = scores.to_vector<double>();
  const auto grade = relevance.to_vector<double>();
  const auto mask = seen.to_vector<double>();
  auto ranked = full({users, max_k}, -1, DType::Int64);
  auto per_user = full({users, cutoff_count, 3}, std::numeric_limits<double>::quiet_NaN(), DType::Float64);
  auto averages = zeros({cutoff_count, 3}, DType::Float64);
  auto* ranked_ids = ranked.data_ptr<std::int64_t>();
  auto* metric = per_user.data_ptr<double>();
  auto* average = averages.data_ptr<double>();
  std::int64_t eligible_users = 0;

  for (std::int64_t user = 0; user < users; ++user) {
    const auto offset = user * items;
    std::vector<std::int64_t> order;
    std::vector<double> ideal;
    for (std::int64_t item = 0; item < items; ++item) {
      const auto i = offset + item;
      if (!std::isfinite(score[i]) || !std::isfinite(mask[i]) ||
          !(grade[i] >= 0 && grade[i] <= 30 && grade[i] == std::floor(grade[i]))) {
        throw std::invalid_argument("recommendation_metrics: invalid score, grade or mask");
      }
      if (mask[i] == 0) {
        order.push_back(item);
        ideal.push_back(grade[i]);
      }
    }
    const auto depth = std::min<std::int64_t>(max_k, order.size());
    std::partial_sort(order.begin(), order.begin() + depth, order.end(), [&](auto a, auto b) {
      return score[offset + a] == score[offset + b] ? a < b : score[offset + a] > score[offset + b];
    });
    std::copy_n(order.begin(), depth, ranked_ids + user * max_k);
    if (ideal.empty() || *std::max_element(ideal.begin(), ideal.end()) == 0) {
      continue;
    }
    ++eligible_users;
    std::partial_sort(ideal.begin(), ideal.begin() + depth, ideal.end(), std::greater<>());
    double dcg = 0.0, idcg = 0.0, reciprocal = 0.0;
    std::vector<std::array<double, 3>> prefix;
    prefix.reserve(depth);
    for (std::int64_t rank = 0; rank < depth; ++rank) {
      const double g = grade[offset + order[rank]];
      const double discount = std::log2(rank + 2.0);
      if (g > 0 && reciprocal == 0) {
        reciprocal = 1.0 / (rank + 1.0);
      }
      dcg += (std::exp2(g) - 1.0) / discount;
      idcg += (std::exp2(ideal[rank]) - 1.0) / discount;
      prefix.push_back({reciprocal > 0 ? 1.0 : 0.0, dcg / idcg, reciprocal});
    }
    for (std::int64_t k = 0; k < cutoff_count; ++k) {
      const auto& at_k = prefix[std::min<std::int64_t>(depth, cutoffs[k]) - 1];
      for (std::int64_t m = 0; m < 3; ++m) {
        metric[(user * cutoff_count + k) * 3 + m] = at_k[m];
        average[k * 3 + m] += at_k[m];
      }
    }
  }
  if (eligible_users == 0) {
    throw std::invalid_argument("recommendation_metrics: no eligible user has a positive grade");
  }
  for (std::int64_t i = 0; i < cutoff_count * 3; ++i) {
    average[i] /= eligible_users;
  }
  return {ranked, per_user, averages, eligible_users};
}

} // namespace datamunge::nn
