#pragma once

/// @file group_fairness.hpp
/// @brief Group-conditioned rates for binary classifier audits.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <optional>
#include <stdexcept>
#include <vector>

namespace datamunge::nn {

/// @brief Binary decision counts and rates for one group.
/// @par Example
/// @code
/// dm::nn::GroupFairnessRates group;
/// @endcode
struct GroupFairnessRates {
  std::size_t count=0; ///< Number of evaluated examples.
  std::size_t selected=0; ///< Number predicted positive.
  std::size_t positives=0; ///< Number labeled positive.
  std::size_t true_positives=0; ///< Positive labels predicted positive.
  std::size_t false_positives=0; ///< Negative labels predicted positive.
  std::optional<double> selection; ///< Predicted-positive fraction, undefined for empty groups.
  std::optional<double> tpr; ///< True-positive rate, undefined without positive labels.
  std::optional<double> fpr; ///< False-positive rate, undefined without negative labels.
};

/// @brief Compute selection, true-positive and false-positive rates by group.
/// @param scores Finite classifier scores.
/// @param truth Binary labels, one per score.
/// @param groups Integer group IDs in [0,group_count), one per score.
/// @param group_count Positive number of groups. Empty groups are retained.
/// @param threshold Finite inclusive threshold for predicting positive, default 0.5.
/// @return Counts and rates in group ID order.
/// @throws std::invalid_argument For mismatched vectors or invalid labels, IDs or scores.
/// @par Example
/// @code
/// auto rates=dm::nn::group_fairness_rates({.8,.3},{1,0},{0,1},2);
/// @endcode
/// @par References
/// - Hardt, Price and Srebro (2016), Equality of Opportunity in Supervised Learning.
///   https://arxiv.org/abs/1610.02413
/// @see group_fairness_gaps()
inline std::vector<GroupFairnessRates> group_fairness_rates(
    const std::vector<double>& scores,const std::vector<int>& truth,
    const std::vector<int>& groups,int group_count,double threshold=.5) {
  if (scores.size()!=truth.size() || scores.size()!=groups.size() ||
      group_count<1 || !std::isfinite(threshold))
    throw std::invalid_argument("Invalid group fairness dimensions or threshold");
  std::vector<GroupFairnessRates> result(group_count);
  for (std::size_t i=0;i<scores.size();++i) {
    if (!std::isfinite(scores[i]) || truth[i]<0 || truth[i]>1 ||
        groups[i]<0 || groups[i]>=group_count)
      throw std::invalid_argument("Invalid group fairness observation");
    auto& g=result[groups[i]];
    const bool positive=truth[i]==1, selected=scores[i]>=threshold;
    ++g.count;
    g.selected+=selected;
    g.positives+=positive;
    g.true_positives+=positive && selected;
    g.false_positives+=!positive && selected;
  }
  for (auto& g:result) {
    if (g.count) g.selection=double(g.selected)/g.count;
    if (g.positives) g.tpr=double(g.true_positives)/g.positives;
    if (g.count>g.positives) g.fpr=double(g.false_positives)/(g.count-g.positives);
  }
  return result;
}

/// @brief Pairwise absolute differences in binary classifier rates.
/// @par Example
/// @code
/// dm::nn::GroupFairnessGaps gaps;
/// @endcode
struct GroupFairnessGaps {
  std::optional<double> demographic_parity; ///< Absolute selection-rate gap.
  std::optional<double> tpr; ///< Absolute true-positive rate gap.
  std::optional<double> fpr; ///< Absolute false-positive rate gap.
  std::optional<double> equalized_odds; ///< Maximum of both conditional rate gaps, if both exist.
};

/// @brief Compare two groups, propagating undefined rates to their gaps.
/// @param first First group rates.
/// @param second Second group rates.
/// @return Pairwise demographic-parity and equalized-odds diagnostics.
/// @par Example
/// @code
/// auto r=dm::nn::group_fairness_rates({.8,.3},{1,0},{0,1},2);
/// auto gaps=dm::nn::group_fairness_gaps(r[0],r[1]);
/// @endcode
/// @see group_fairness_rates()
inline GroupFairnessGaps group_fairness_gaps(const GroupFairnessRates& first,
                                             const GroupFairnessRates& second) {
  /// Difference of available rates, preserving absent denominators.
  const auto gap=[](auto a,auto b)->std::optional<double> {
    if (!a || !b) return std::nullopt;
    return std::abs(*a-*b);
  };
  GroupFairnessGaps out{gap(first.selection,second.selection),gap(first.tpr,second.tpr),
                       gap(first.fpr,second.fpr),std::nullopt};
  if (out.tpr && out.fpr) out.equalized_odds=std::max(*out.tpr,*out.fpr);
  return out;
}

}  // namespace datamunge::nn
