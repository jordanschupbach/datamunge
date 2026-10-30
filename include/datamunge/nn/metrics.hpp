#pragma once

/// @file metrics.hpp
/// @brief Classification metrics (tflow nn/metrics.h semantics).
///
/// Ratios with a zero denominator are 0; per-class IoU is NaN for a class
/// that never appears in truth or predictions.

#include <cstdint>
#include <string>
#include <vector>

#include "datamunge/nn/tensor.hpp"

namespace datamunge::nn {

/// @brief Per-class or macro-averaged classification ratios plus ground-truth support.
struct ClassMetrics {
  double precision{
      0};           ///< True positives divided by predicted positives; zero if none are predicted.
  double recall{0}; ///< True positives divided by true support; zero for a class absent from truth.
  double f1{0};     ///< Harmonic precision/recall score; zero if both class counts are zero.
  double iou{0};    ///< Intersection over union; NaN if absent from both truth and predictions.
  std::uint64_t support{
      0}; ///< Ground-truth count for a class, or total support for a macro result.
};

/// @brief Value scale used by confusion-matrix SVG cells.
///
/// True and Predicted divide by row and column support respectively; All divides
/// by total support. Empty denominators produce zero. Counts remain unchanged.
/// @par Example
/// @code
/// dm::nn::ConfusionMatrixPlotOptions options;
/// options.normalization = dm::nn::ConfusionMatrixNormalization::True;
/// @endcode
/// @see ConfusionMatrixPlotOptions
enum class ConfusionMatrixNormalization {
  None,      ///< Raw observation counts.
  True,      ///< Fraction within each true-class row.
  Predicted, ///< Fraction within each predicted-class column.
  All        ///< Fraction of all observations.
};

/// @brief Presentation settings for an annotated confusion-matrix SVG.
///
/// True classes occupy rows, predicted classes columns. Labels default to zero-based
/// class indices; otherwise supply exactly one label per class. Text is XML-escaped.
/// The canvas fits its cells, title, labels and color scale, without a fixed blank
/// border. Sizes are SVG pixels; colors are six-digit #RRGGBB strings.
/// @par Example
/// @code
/// dm::nn::ConfusionMatrixPlotOptions options;
/// options.labels = {"negative", "positive"};
/// options.cell_size = 52;
/// @endcode
/// @see ConfusionMatrix::to_svg()
struct ConfusionMatrixPlotOptions {
  std::string title{"Confusion matrix"}; ///< Accessible heading, wrapped to the canvas.
  std::vector<std::string>     labels;   ///< Ordered class names; empty selects numeric indices.
  ConfusionMatrixNormalization normalization{
      ConfusionMatrixNormalization::None}; ///< Cell value scale.
  double      cell_size{48};               ///< Cell side in pixels, 24..160.
  double      font_size{12};               ///< Annotation/axis-label font size in pixels, 6..24.
  bool        show_values{true};           ///< Print counts or normalized percentages inside cells.
  std::string low_color{"#F1F5F9"};        ///< Zero-cell color.
  std::string high_color{"#2563EB"};       ///< Maximum count or unit normalized value color.
};

/// @brief Counts of true-class rows and predicted-class columns, with SVG rendering.
///
/// Accumulates observations without retaining their tensors. Const rendering does
/// not mutate counts or perform model inference. Concurrent mutation needs external
/// synchronization. Ratios and empty-support behavior follow the metric methods.
/// @par Example
/// @code
/// dm::nn::ConfusionMatrix matrix(2);
/// matrix.add(0, 0);
/// matrix.add(1, 0);
/// matrix.save_svg("confusion.svg", {.labels={"negative", "positive"}});
/// @endcode
/// @see ConfusionMatrixPlotOptions
class ConfusionMatrix {
 public:
  /// @brief Allocates a zero-filled square count matrix for a positive number of classes.
  /// @throws std::invalid_argument If classes is zero.
  explicit ConfusionMatrix(std::size_t classes);
  /// @brief Increments the cell at the true-class row and predicted-class column.
  /// @throws std::out_of_range If either class index exceeds the configured range.
  void add(std::size_t truth, std::size_t prediction);
  /// Adds every pair from two integer tensors of equal size.
  void add(const Tensor& truth, const Tensor& prediction);
  /// @brief Borrows the square count table, indexed by truth then prediction.
  [[nodiscard]] const std::vector<std::vector<std::uint64_t>>& cells() const noexcept {
    return cells_;
  }
  /// @brief Returns the configured number of classes.
  [[nodiscard]] std::size_t classes() const noexcept { return cells_.size(); }
  /// @brief Computes one class's precision, recall, F1, IoU, and true support.
  /// @throws std::out_of_range If c is outside the configured class range.
  [[nodiscard]] ClassMetrics per_class(std::size_t c) const;
  /// @brief Returns the fraction of observations on the diagonal.
  /// @throws std::logic_error If no observations have been added.
  [[nodiscard]] double accuracy() const;
  /// Unweighted mean over classes (IoU over classes where it is defined).
  [[nodiscard]] ClassMetrics macro() const;
  /// As an int64 tensor [classes, classes], truth rows.
  [[nodiscard]] Tensor to_tensor() const;

  /// @brief Own validated integer counts from an existing square tensor.
  ///
  /// Copies counts; retains no tensor storage or autograd graph. The input may be
  /// noncontiguous. Rows are true classes, columns predicted classes.
  /// @param counts Nonempty square [C,C] Int32/Int64 tensor with nonnegative counts.
  /// @return Independent confusion matrix containing exactly those counts.
  /// @throws std::invalid_argument For undefined, nonsquare, noninteger or negative input.
  /// @par Example
  /// @code
  /// const auto counts=dm::nn::tensor({3,1,2,4},{2,2}).to(dm::nn::DType::Int64);
  /// const auto matrix=dm::nn::ConfusionMatrix::from_counts(counts);
  /// const auto accuracy=matrix.accuracy(); // 0.7
  /// @endcode
  /// @see to_tensor(), add()
  [[nodiscard]] static ConfusionMatrix from_counts(const Tensor& counts);

  /// @brief Render counts or normalized values as a labeled, accessible SVG heatmap.
  ///
  /// True rows run top to bottom, predicted columns left to right. Values are raw
  /// integers or percentages. Normalized color limits are [0,1]; raw limits are
  /// [0,max(count,1)]. Empty matrices render zero cells without requiring accuracy().
  /// Large annotations shrink to fit their cells; tooltips retain exact counts.
  /// Output is deterministic and requires no display server or plotting backend.
  /// @param options Labels, normalization, cell/font size, colors and heading.
  /// @return Standalone UTF-8 SVG text; save it as a local figure, not report output.
  /// @throws std::invalid_argument For invalid enums, colors, labels or numeric settings.
  /// @par Example
  /// @code
  /// dm::nn::ConfusionMatrix matrix(2);
  /// matrix.add(1,0);
  /// const auto svg=matrix.to_svg({.normalization=dm::nn::ConfusionMatrixNormalization::True});
  /// @endcode
  /// @see save_svg()
  [[nodiscard]] std::string to_svg(const ConfusionMatrixPlotOptions& options = {}) const;

  /// @brief Save the annotated confusion-matrix SVG to a local file.
  ///
  /// Replaces an existing file. Parent directories must already exist. Counts and
  /// model state are unchanged. Same rendering settings and validation as to_svg().
  /// @param path Relative or absolute destination filename.
  /// @param options Labels, normalization, cell/font size, colors and heading.
  /// @throws std::invalid_argument If rendering settings are invalid.
  /// @throws std::runtime_error If opening or writing the file fails.
  /// @par Example
  /// @code
  /// dm::nn::ConfusionMatrix matrix(2);
  /// matrix.add(0,0);
  /// matrix.save_svg("confusion.svg");
  /// @endcode
  /// @see to_svg()
  void save_svg(const std::string& path, const ConfusionMatrixPlotOptions& options = {}) const;

 private:
  /// @brief Counts indexed as [true class][predicted class], mutated by add().
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

} // namespace datamunge::nn
