/// @file
/// @brief Model privacy diagnostics based on held-out loss distributions.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iterator>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

#include "datamunge/nn/metrics.hpp"

namespace datamunge::nn {

/// @brief Evaluation of a threshold that identifies low-loss rows as members.
struct LossMembershipAudit {
  double threshold;           ///< Smallest calibration threshold maximizing balanced accuracy.
  std::size_t true_positives;  ///< Evaluation members at or below threshold.
  std::size_t false_positives; ///< Evaluation nonmembers at or below threshold.
  double auc;                 ///< Evaluation AUROC with lower loss indicating membership.
};

/// @brief Calibrate a loss threshold and measure membership leakage on separate rows.
/// @param calibration_members Losses on known training rows used for calibration.
/// @param calibration_nonmembers Losses on known nontraining rows used for calibration.
/// @param evaluation_members Losses on separate known training rows.
/// @param evaluation_nonmembers Losses on separate known nontraining rows.
/// @return Selected threshold, evaluation counts, and tie-aware AUROC.
/// @throws std::invalid_argument A loss span is empty or contains a nonfinite value.
inline LossMembershipAudit audit_loss_membership(std::span<const double> calibration_members,
    std::span<const double> calibration_nonmembers, std::span<const double> evaluation_members,
    std::span<const double> evaluation_nonmembers) {
  for (const auto losses : {calibration_members, calibration_nonmembers, evaluation_members,
                            evaluation_nonmembers}) {
    if (losses.empty() || !std::ranges::all_of(losses, [](double loss) { return std::isfinite(loss); })) {
      throw std::invalid_argument("audit_loss_membership: expected nonempty finite loss samples");
    }
  }
  const auto sorted = [](std::span<const double> losses) {
    std::vector<double> result(losses.begin(), losses.end());
    std::ranges::sort(result);
    return result;
  };
  const auto members = sorted(calibration_members), nonmembers = sorted(calibration_nonmembers);
  std::vector<double> candidates;
  std::ranges::merge(members, nonmembers, std::back_inserter(candidates));
  double threshold = -std::numeric_limits<double>::infinity();
  long double best = 0;
  for (const double candidate : candidates) {
    const auto true_positives = std::ranges::upper_bound(members, candidate) - members.begin();
    const auto false_positives = std::ranges::upper_bound(nonmembers, candidate) - nonmembers.begin();
    const auto score = static_cast<long double>(true_positives) * nonmembers.size() -
                       static_cast<long double>(false_positives) * members.size();
    if (score > best) {
      best = score;
      threshold = candidate;
    }
  }
  const auto hits = [threshold](std::span<const double> losses) {
    return std::size_t(std::ranges::count_if(losses, [threshold](double loss) { return loss <= threshold; }));
  };
  std::vector<int> truth;
  std::vector<double> scores;
  for (const auto losses : {evaluation_members, evaluation_nonmembers}) {
    for (const double loss : losses) scores.push_back(-loss);
  }
  truth.insert(truth.end(), evaluation_members.size(), 1);
  truth.insert(truth.end(), evaluation_nonmembers.size(), 0);
  return {threshold, hits(evaluation_members), hits(evaluation_nonmembers), roc_auc(truth, scores)};
}

} // namespace datamunge::nn
