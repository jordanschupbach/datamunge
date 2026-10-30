/// @file metrics.cpp
/// @brief Confusion-matrix aggregation and tensor/ranking classification metric calculations.

#include "datamunge/nn/metrics.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

#include "datamunge/nn/autograd.hpp"
#include "datamunge/nn/ops.hpp"

namespace datamunge::nn {

/// @brief Zero-safe ratios, integer label validation, and shared score ranking.
namespace {

/// @brief Returns n/d, defining the result as zero for a zero denominator.
double ratio(double n, double d) { return d ? n / d : 0.0; }

/// @brief Copies Int32/Int64 tensor labels to a flat vector; rejects other dtypes with an operation name.
std::vector<std::int64_t> integer_values(const Tensor& t, const char* what) {
  if (!is_integral(t.dtype()))
    throw std::invalid_argument(std::string(what) + ": expected an int32 or int64 tensor, got " +
                                dtype_name(t.dtype()));
  return t.to_vector<std::int64_t>();
}

/// @brief One validated score/binary-label pair for ordered threshold sweeps.
struct Ranked {
  /// @brief Finite classifier score, with larger values ranking ahead of smaller ones.
  double score;
  /// @brief Binary ground-truth label, either zero or one.
  int    label;
};

/// @brief Validates nonempty paired binary labels/finite scores and stably sorts by descending score.
/// @throws std::invalid_argument For empty/different lengths, non-binary labels, or non-finite scores.
std::vector<Ranked> rank(const std::vector<int>& truth, const std::vector<double>& scores) {
  if (truth.empty() || truth.size() != scores.size()) throw std::invalid_argument("ranking: invalid inputs");
  std::vector<Ranked> ranked;
  for (std::size_t i = 0; i < truth.size(); ++i) {
    if ((truth[i] != 0 && truth[i] != 1) || !std::isfinite(scores[i]))
      throw std::invalid_argument("ranking: needs binary labels and finite scores");
    ranked.push_back({scores[i], truth[i]});
  }
  std::stable_sort(ranked.begin(), ranked.end(), [](const Ranked& a, const Ranked& b) { return a.score > b.score; });
  return ranked;
}

}  // namespace

// ---- ConfusionMatrix -----------------------------------------------------------------

ConfusionMatrix::ConfusionMatrix(std::size_t classes) : cells_(classes, std::vector<std::uint64_t>(classes)) {
  if (!classes) throw std::invalid_argument("ConfusionMatrix: need at least one class");
}

void ConfusionMatrix::add(std::size_t truth, std::size_t prediction) {
  if (truth >= cells_.size() || prediction >= cells_.size())
    throw std::out_of_range("ConfusionMatrix: class label out of range");
  ++cells_[truth][prediction];
}

void ConfusionMatrix::add(const Tensor& truth, const Tensor& prediction) {
  const auto t = integer_values(truth, "ConfusionMatrix::add");
  const auto p = integer_values(prediction, "ConfusionMatrix::add");
  if (t.size() != p.size()) throw std::invalid_argument("ConfusionMatrix::add: truth and prediction sizes differ");
  for (std::size_t i = 0; i < t.size(); ++i) {
    if (t[i] < 0 || p[i] < 0) throw std::out_of_range("ConfusionMatrix: class label out of range");
    add(static_cast<std::size_t>(t[i]), static_cast<std::size_t>(p[i]));
  }
}

ClassMetrics ConfusionMatrix::per_class(std::size_t c) const {
  if (c >= cells_.size()) throw std::out_of_range("ConfusionMatrix: class label out of range");
  std::uint64_t support = 0, predicted = 0;
  for (std::size_t i = 0; i < cells_.size(); ++i) {
    support += cells_[c][i];
    predicted += cells_[i][c];
  }
  const auto   tp       = cells_[c][c];
  const auto   combined = support + predicted;
  ClassMetrics m;
  m.precision = ratio(static_cast<double>(tp), static_cast<double>(predicted));
  m.recall    = ratio(static_cast<double>(tp), static_cast<double>(support));
  m.f1        = ratio(2.0 * static_cast<double>(tp), static_cast<double>(combined));
  m.iou       = combined > tp ? ratio(static_cast<double>(tp), static_cast<double>(combined - tp))
                              : std::numeric_limits<double>::quiet_NaN();
  m.support   = support;
  return m;
}

double ConfusionMatrix::accuracy() const {
  std::uint64_t correct = 0, count = 0;
  for (std::size_t i = 0; i < cells_.size(); ++i) {
    correct += cells_[i][i];
    count += std::accumulate(cells_[i].begin(), cells_[i].end(), std::uint64_t{0});
  }
  if (!count) throw std::logic_error("ConfusionMatrix: accuracy needs observations");
  return ratio(static_cast<double>(correct), static_cast<double>(count));
}

ClassMetrics ConfusionMatrix::macro() const {
  ClassMetrics result;
  std::size_t  present = 0;
  for (std::size_t i = 0; i < cells_.size(); ++i) {
    const auto m = per_class(i);
    result.precision += m.precision;
    result.recall += m.recall;
    result.f1 += m.f1;
    result.support += m.support;
    if (std::isfinite(m.iou)) {
      result.iou += m.iou;
      ++present;
    }
  }
  const double n = static_cast<double>(cells_.size());
  result.precision /= n;
  result.recall /= n;
  result.f1 /= n;
  result.iou = present ? result.iou / static_cast<double>(present) : std::numeric_limits<double>::quiet_NaN();
  return result;
}

Tensor ConfusionMatrix::to_tensor() const {
  const auto                n = static_cast<std::int64_t>(cells_.size());
  std::vector<std::int64_t> values;
  values.reserve(static_cast<std::size_t>(n * n));
  for (const auto& row : cells_)
    for (auto v : row) values.push_back(static_cast<std::int64_t>(v));
  return from_vector(values, {n, n});
}

// ---- tensor metrics ---------------------------------------------------------------------

double accuracy(const Tensor& scores_or_predictions, const Tensor& labels) {
  NoGradGuard guard;
  Tensor      predictions = scores_or_predictions.detach();
  if (predictions.dim() == labels.dim() + 1) predictions = argmax(predictions, -1);
  if (predictions.numel() != labels.numel()) throw std::invalid_argument("accuracy: prediction and label sizes differ");
  if (labels.numel() == 0) throw std::invalid_argument("accuracy: no labels");
  const auto p = predictions.to_vector<double>();
  const auto t = labels.to_vector<double>();
  std::size_t correct = 0;
  for (std::size_t i = 0; i < p.size(); ++i) correct += p[i] == t[i];
  return static_cast<double>(correct) / static_cast<double>(p.size());
}

double top_k_accuracy(const Tensor& scores, const Tensor& labels, std::int64_t k) {
  if (scores.dim() != 2) throw std::invalid_argument("top_k_accuracy: scores must be [N, C]");
  const std::int64_t n = scores.shape()[0], classes = scores.shape()[1];
  const auto         truth = integer_values(labels, "top_k_accuracy");
  if (n == 0 || static_cast<std::int64_t>(truth.size()) != n || k <= 0)
    throw std::invalid_argument("top_k_accuracy: invalid inputs");
  if (k > classes) throw std::invalid_argument("top_k_accuracy: k exceeds number of classes");
  const auto  s       = scores.detach().to_vector<double>();
  std::size_t correct = 0;
  for (std::int64_t i = 0; i < n; ++i) {
    const std::int64_t label = truth[static_cast<std::size_t>(i)];
    if (label < 0 || label >= classes) throw std::invalid_argument("top_k_accuracy: label out of range");
    const double*      row    = s.data() + i * classes;
    std::int64_t       higher = 0;
    for (std::int64_t c = 0; c < classes; ++c) {
      if (!std::isfinite(row[c])) throw std::invalid_argument("top_k_accuracy: scores must be finite");
      higher += row[c] > row[label];
    }
    correct += higher < k;
  }
  return static_cast<double>(correct) / static_cast<double>(n);
}

double roc_auc(const std::vector<int>& truth, const std::vector<double>& scores) {
  const auto   ranked    = rank(truth, scores);
  const double positives = static_cast<double>(std::count(truth.begin(), truth.end(), 1));
  const double negatives = static_cast<double>(truth.size()) - positives;
  if (!positives || !negatives) throw std::invalid_argument("roc_auc: needs positive and negative labels");
  double higher_positives = 0, wins = 0;
  for (std::size_t i = 0; i < ranked.size();) {
    std::size_t j = i;
    double      p = 0, neg = 0;
    while (j < ranked.size() && ranked[j].score == ranked[i].score) {
      p += ranked[j].label;
      neg += 1 - ranked[j].label;
      ++j;
    }
    wins += neg * (higher_positives + .5 * p);
    higher_positives += p;
    i = j;
  }
  return wins / (positives * negatives);
}

double average_precision(const std::vector<int>& truth, const std::vector<double>& scores) {
  const auto   ranked    = rank(truth, scores);
  const double positives = static_cast<double>(std::count(truth.begin(), truth.end(), 1));
  if (!positives) return 0;
  double tp = 0, area = 0;
  for (std::size_t i = 0; i < ranked.size();) {
    std::size_t j     = i;
    double      added = 0;
    while (j < ranked.size() && ranked[j].score == ranked[i].score) {
      added += ranked[j].label;
      ++j;
    }
    tp += added;
    area += added / positives * tp / static_cast<double>(j);
    i = j;
  }
  return area;
}

}  // namespace datamunge::nn
