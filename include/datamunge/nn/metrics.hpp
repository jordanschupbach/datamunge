#pragma once

/// @file metrics.hpp
/// @brief Classification metrics (tflow nn/metrics.h semantics).
///
/// Ratios with a zero denominator are 0; per-class IoU is NaN for a class
/// that never appears in truth or predictions.

#include <cstdint>
#include <vector>

#include "datamunge/nn/tensor.hpp"

namespace datamunge::nn {

struct ClassMetrics {
  double        precision{0}, recall{0}, f1{0}, iou{0};
  std::uint64_t support{0};
};

/// Counts of (truth, prediction) pairs.
class ConfusionMatrix {
 public:
  explicit ConfusionMatrix(std::size_t classes);
  void add(std::size_t truth, std::size_t prediction);
  /// Adds every pair from two integer tensors of equal size.
  void add(const Tensor& truth, const Tensor& prediction);
  [[nodiscard]] const std::vector<std::vector<std::uint64_t>>& cells() const noexcept { return cells_; }
  [[nodiscard]] std::size_t                                    classes() const noexcept { return cells_.size(); }
  [[nodiscard]] ClassMetrics                                   per_class(std::size_t c) const;
  [[nodiscard]] double                                         accuracy() const;
  /// Unweighted mean over classes (IoU over classes where it is defined).
  [[nodiscard]] ClassMetrics                                   macro() const;
  /// As an int64 tensor [classes, classes], truth rows.
  [[nodiscard]] Tensor                                         to_tensor() const;

 private:
  std::vector<std::vector<std::uint64_t>> cells_;
};

/// Fraction of rows where argmax(scores) equals the label. `scores` is
/// [N, C] (or already-predicted labels [N]); labels [N] are integers.
double accuracy(const Tensor& scores_or_predictions, const Tensor& labels);
/// Fraction of rows whose label has fewer than k strictly higher scores.
double top_k_accuracy(const Tensor& scores, const Tensor& labels, std::int64_t k);
/// Area under the ROC curve for binary labels (ties count half).
double roc_auc(const std::vector<int>& truth, const std::vector<double>& scores);
/// Area under the precision-recall curve (step interpolation, ties grouped).
double average_precision(const std::vector<int>& truth, const std::vector<double>& scores);

}  // namespace datamunge::nn
