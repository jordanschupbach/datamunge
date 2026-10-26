// Multi-label grouped-tie AP, 80-class confusion metrics and validation-only
// threshold calibration: a port of tflow's common/ml/multilabel_metrics.h.
#pragma once

#include <algorithm>
#include <numeric>

#include "common/multilabel_data.hpp"

namespace multilabel {

/// Noninterpolated AP where equal scores form one group (no optimistic tie order); -1 without positives.
inline double MultilabelAP(const std::vector<double>& scores, const std::vector<int>& labels) {
  check::Require(scores.size() == labels.size(), "AP score/target lengths");
  int                      positives = 0;
  std::vector<std::size_t> order(scores.size());
  std::iota(order.begin(), order.end(), 0);
  for (std::size_t i = 0; i < scores.size(); ++i) {
    check::Require(std::isfinite(scores[i]) && (labels[i] == 0 || labels[i] == 1), "finite AP scores/binary targets");
    positives += labels[i];
  }
  if (!positives) return -1;
  std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) { return scores[a] > scores[b]; });
  double ap = 0;
  int    true_positive = 0, seen = 0;
  for (std::size_t begin = 0; begin < order.size();) {
    std::size_t end    = begin;
    const int   before = true_positive;
    while (end < order.size() && scores[order[end]] == scores[order[begin]]) {
      true_positive += labels[order[end]];
      ++seen;
      ++end;
    }
    ap += double(true_positive - before) / positives * true_positive / seen;
    begin = end;
  }
  return ap;
}

struct MultilabelMetrics {
  double       nll = 0, brier = 0, map = 0, micro_ap = 0, micro_f1 = 0, macro_f1 = 0, hamming = 0, exact = 0;
  int          ap_classes = 0, absent_classes = 0;
  std::int64_t tp = 0, fp = 0, fn = 0, tn = 0;
};

inline MultilabelMetrics ScoreMultilabel(const nn::Tensor& probability, const nn::Tensor& labels, double threshold = .5) {
  check::Require(probability.dim() == 2 && probability.shape() == labels.shape() && probability.numel() > 0, "multi-label probability dimensions");
  MultilabelMetrics   result;
  const int           count = int(labels.size(0)), classes = int(labels.size(1));
  const auto          P = probability.to_vector<float>(), Y = labels.to_vector<float>();
  std::vector<double> all_scores;
  std::vector<int>    all_labels;
  for (int c = 0; c < classes; ++c) {
    std::vector<double> scores;
    std::vector<int>    truth;
    int                 tp = 0, fp = 0, fn = 0, tn = 0;
    for (int n = 0; n < count; ++n) {
      const double p = P[std::size_t(n * classes + c)], y = Y[std::size_t(n * classes + c)];
      check::Require(std::isfinite(p) && p >= 0 && p <= 1 && (y == 0 || y == 1), "binary labels and valid independent probabilities");
      const bool predicted = p >= threshold;
      tp += predicted && y == 1;
      fp += predicted && y == 0;
      fn += !predicted && y == 1;
      tn += !predicted && y == 0;
      result.nll -= y * std::log(std::max(p, 1e-12)) + (1 - y) * std::log(std::max(1 - p, 1e-12));
      result.brier += (p - y) * (p - y);
      scores.push_back(p);
      truth.push_back(int(y));
      all_scores.push_back(p);
      all_labels.push_back(int(y));
    }
    const double ap = MultilabelAP(scores, truth);
    if (ap >= 0) {
      result.map += ap;
      ++result.ap_classes;
    } else
      ++result.absent_classes;
    result.macro_f1 += 2 * tp + fp + fn ? double(2 * tp) / (2 * tp + fp + fn) : 0;
    result.tp += tp, result.fp += fp, result.fn += fn, result.tn += tn;
  }
  const double total = double(probability.numel());
  result.nll /= total;
  result.brier /= total;
  if (result.ap_classes) result.map /= result.ap_classes;
  result.micro_ap = MultilabelAP(all_scores, all_labels);
  if (result.micro_ap < 0) result.micro_ap = 0;
  result.micro_f1 = 2 * result.tp + result.fp + result.fn ? double(2 * result.tp) / double(2 * result.tp + result.fp + result.fn) : 0;
  result.macro_f1 /= classes;
  result.hamming = double(result.fp + result.fn) / total;
  for (int n = 0; n < count; ++n) {
    bool correct = true;
    for (int c = 0; c < classes; ++c) correct &= (P[std::size_t(n * classes + c)] >= threshold) == (Y[std::size_t(n * classes + c)] > .5);
    result.exact += correct;
  }
  result.exact /= count;
  return result;
}

/// One global threshold (.05..95 step .05, or 1.01 = predict none) maximizing validation micro F1; ties take the highest.
inline double CalibrateMultilabel(const nn::Tensor& validation, const nn::Tensor& truth) {
  double best = -1, chosen = 1.01;
  for (int i = 1; i <= 20; ++i) {
    const double threshold = i == 20 ? 1.01 : i * .05, score = ScoreMultilabel(validation, truth, threshold).micro_f1;
    if (score > best + 1e-12 || std::abs(score - best) <= 1e-12) best = score, chosen = threshold;
  }
  return chosen;
}

}  // namespace multilabel
