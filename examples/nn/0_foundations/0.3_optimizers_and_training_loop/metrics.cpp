// CMD: just run-nn metrics
// Classification, ranking, detection and segmentation metrics.
//
// Port of tflow's metrics. Compute softmax probabilities and an InTopK-style
// membership test with nn tensor ops and compare them with nn's metrics
// (nn::ConfusionMatrix, nn::top_k_accuracy) on a small three-class
// prediction set. Print its confusion matrix (true rows, predicted columns),
// per-class precision, recall, F1 and IoU, macro averages and accuracy.
// Validate exact fractions.
//
// Check ROC AUC by an independent pairwise definition, including score ties.
// Binary AP uses noninterpolated recall increments; detection AP uses the
// all-points precision envelope and one-to-one box matching. Report mAP at
// IoU .5 and .75 separately. This small evaluator has no COCO crowd/area rules.
// References: scikit-learn average_precision_score documentation and the
// PASCAL VOC2010 development kit's detection evaluation definition.
//
// Differences from TF: the "TF graph" side (Softmax, ArgMax, InTopK) is
// written with nn tensor ops (InTopK: fewer than k strictly higher scores).
// Box IoU and detection mAP are not in the library yet (plan P3 box
// utilities), so they come from examples/nn/common/detection_metrics.hpp.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <datamunge/nn/nn.hpp>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

#include "common/checks.hpp"
#include "common/detection_metrics.hpp"

namespace nn = datamunge::nn;
using nn::DType;
using nn::Tensor;

namespace {

/// TF InTopK with tensor ops: the label's score has fewer than k strictly higher scores.
Tensor InTopK(const Tensor& scores, const Tensor& labels, double k) {
  const Tensor label_score = nn::sum(scores * nn::one_hot(labels, scores.shape()[1]), {1}, true);
  return nn::less(nn::sum(nn::cast(nn::greater(scores, label_score), DType::Float32), {1}), k);
}

double CountTrue(const Tensor& mask) { return nn::sum(nn::cast(mask, DType::Float64)).item(); }

void Classification() {
  const std::vector<std::int32_t> truth{0, 0, 1, 1, 2, 2};
  const Tensor logits = nn::tensor({{3, 2, 1}, {2, 3, 1}, {1, 3, 2}, {1, 2, 3}, {1, 2, 3}, {1, 2, 3}});
  const Tensor labels = nn::from_vector(truth);
  const Tensor probabilities = nn::softmax(logits, 1);
  const Tensor prediction    = nn::argmax(probabilities, 1);
  const double top1 = CountTrue(InTopK(probabilities, labels, 1)), top2 = CountTrue(InTopK(probabilities, labels, 2));
  const bool   tied = CountTrue(InTopK(nn::tensor({{1, 1, 0}}), nn::from_vector(std::vector<std::int32_t>{1}), 1)) == 1;

  nn::ConfusionMatrix cm(3);
  cm.add(labels, prediction);
  check::Require(cm.cells() == std::vector<std::vector<std::uint64_t>>({{1, 1, 0}, {0, 1, 1}, {0, 0, 2}}),
                 "confusion matrix orientation/counts");
  check::Near(cm.accuracy(), 2. / 3, 1e-12, "accuracy");
  check::Near(nn::accuracy(probabilities, labels), 2. / 3, 1e-12, "tensor accuracy");
  check::Near(nn::top_k_accuracy(probabilities, labels, 1), top1 / 6., 0, "tensor-op top-1 comparison");
  check::Near(nn::top_k_accuracy(probabilities, labels, 2), top2 / 6., 0, "tensor-op top-2 comparison");
  check::Near(nn::top_k_accuracy(nn::tensor({{1, 1, 0}}), nn::from_vector(std::vector<std::int32_t>{1}), 1), 1, 0,
              "ties at top-k boundary included");
  check::Require(tied, "InTopK includes boundary ties");
  const auto macro = cm.macro();
  check::Near(macro.precision, 13. / 18, 1e-12, "macro precision");
  check::Near(macro.recall, 2. / 3, 1e-12, "macro recall");
  check::Near(macro.f1, 59. / 90, 1e-12, "macro F1");
  check::Near(macro.iou, .5, 1e-12, "mean segmentation IoU");
  std::cout << "  confusion matrix (true rows, predicted columns):\n";
  for (const auto& row : cm.cells()) {
    std::cout << "    ";
    for (auto n : row) std::cout << n << ' ';
    std::cout << '\n';
  }
  for (std::size_t c = 0; c < 3; ++c) {
    const auto m = cm.per_class(c);
    std::cout << "  class " << c << ": precision=" << m.precision << ", recall=" << m.recall << ", F1=" << m.f1
              << ", IoU=" << m.iou << '\n';
  }
  std::cout << "  accuracy=" << cm.accuracy() << ", top-2=" << nn::top_k_accuracy(probabilities, labels, 2)
            << ", macro F1=" << macro.f1 << ", mean IoU=" << macro.iou << '\n';
  nn::ConfusionMatrix absent(3);
  absent.add(0, 0);
  check::Require(std::isnan(absent.per_class(1).iou), "empty mask union has undefined IoU");
  check::Near(absent.macro().iou, 1, 0, "mean IoU excludes empty unions");
  check::Near(absent.macro().f1, 1. / 3, 0, "macro F1 includes configured absent classes as zero");
}

void Rankings() {
  check::Near(nn::roc_auc({0, 0, 1, 1}, {.1, .4, .35, .8}), .75, 0, "known ROC AUC");
  check::Near(nn::average_precision({0, 0, 1, 1}, {.1, .4, .35, .8}), 5. / 6, 1e-12, "known AP");
  check::Near(nn::roc_auc({1, 0}, {.5, .5}), .5, 0, "tied scores get half credit");
  check::Near(nn::average_precision({1, 0}, {.5, .5}), .5, 0, "tied AP thresholds processed together");
  check::Near(nn::average_precision({0, 0}, {.1, .9}), 0, 0, "no positives AP convention");
  std::mt19937 rng(7);
  double       worst = 0;
  for (int trial = 0; trial < 100; ++trial) {
    std::vector<int> labels{0, 1};
    for (int i = 2; i < 20; ++i) labels.push_back(static_cast<int>(rng() % 2));
    std::vector<double> scores;
    for (int i = 0; i < 20; ++i) scores.push_back(rng() % 5);  // many ties
    double wins = 0, pairs = 0;
    for (std::size_t p = 0; p < labels.size(); ++p)
      if (labels[p])
        for (std::size_t n = 0; n < labels.size(); ++n)
          if (!labels[n]) {
            wins += scores[p] > scores[n] ? 1 : scores[p] == scores[n] ? .5 : 0;
            ++pairs;
          }
    const double error = std::abs(nn::roc_auc(labels, scores) - wins / pairs);
    worst              = std::max(worst, error);
    check::Near(error, 0, 1e-12, "ranked AUC vs pairwise independent definition");
  }
  bool rejected = false;
  try {
    nn::roc_auc({1, 1}, {.1, .9});
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  check::Require(rejected, "single-class ROC AUC must fail explicitly");
  std::cout << "  known ranking: ROC AUC=0.75, AP=0.833333; all tied: AUC=AP=0.5\n"
            << "  100 randomized tied-score AUC comparisons: max error=" << worst << '\n';
}

void Detection() {
  using namespace detection;
  check::Near(IoU({0, 0, 2, 2}, {1, 1, 3, 3}), 1. / 7, 1e-12, "box IoU");
  check::Near(IoU({0, 0, 1, 1}, {1, 0, 2, 1}), 0, 0, "touching boxes have no overlap");
  check::Near(IoU({0, 0, 0, 0}, {0, 0, 0, 0}), 0, 0, "zero-area boxes");
  const std::vector<GroundTruth> gt{{0, 0, {0, 0, 1, 1}}, {1, 0, {0, 0, 1, 1}}, {0, 1, {2, 2, 4, 4}}};
  const std::vector<detection::Detection> det{
      {0, 0, {0, 0, 1, 1}, .9},   {0, 0, {0, 0, 1, 1}, .8},  // duplicate FP
      {1, 0, {0, 0, 1, 1}, .7},   {0, 1, {2, 2, 4, 3.2}, .6},  // IoU=.6
      {9, 1, {2, 2, 4, 4}, .5},   {0, 2, {0, 0, 1, 1}, .99}};
  const auto low = DetectionMap(gt, det, 3, .5), high = DetectionMap(gt, det, 3, .75);
  check::Near(low.ap[0], 5. / 6, 1e-12, "duplicate detection cannot match twice");
  check::Near(low.ap[1], 1, 0, "class one AP50");
  check::Require(std::isnan(low.ap[2]), "class without ground truth excluded");
  check::Near(low.map, 11. / 12, 1e-12, "mean AP50");
  check::Near(high.map, 5. / 12, 1e-12, "mean AP75");
  check::Near(DetectionMap(gt, {}, 3).map, 0, 0, "no detections has zero recall/AP");
  check::Require(std::isnan(DetectionMap({}, {}, 3).map), "empty evaluation has undefined mAP");
  const auto envelope = DetectionMap({{0, 0, {0, 0, 1, 1}}, {1, 0, {0, 0, 1, 1}}},
                                     {{9, 0, {0, 0, 1, 1}, .9}, {0, 0, {0, 0, 1, 1}, .8}, {1, 0, {0, 0, 1, 1}, .7}}, 1);
  check::Near(envelope.map, 2. / 3, 1e-12, "precision envelope differs from un-interpolated AP");
  std::cout << "  overlapping boxes IoU=1/7; duplicate detection counts as false positive\n"
            << "  AP50 per class=(" << low.ap[0] << ", " << low.ap[1] << ", absent), mAP50=" << low.map
            << ", mAP75=" << high.map << '\n';
}

}  // namespace

int main() {
  try {
    std::cout << std::setprecision(6) << "== Classification and segmentation metrics\n";
    Classification();
    std::cout << "\n== Binary ranking metrics\n";
    Rankings();
    std::cout << "\n== Detection metrics\n";
    Detection();
    std::cout << "All metric checks passed.\n";
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
  return 0;
}
