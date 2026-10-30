#pragma once
/// @file detection_evaluation.hpp
/// @brief Tensor VOC/COCO all-area box AP with explicit difficult/crowd rules.
#include "datamunge/nn/autograd.hpp"
#include "datamunge/nn/boxes.hpp"
#include "datamunge/nn/ops.hpp"
#include <algorithm>
#include <cmath>
#include <map>
#include <set>
#include <stdexcept>
#include <vector>
namespace datamunge::nn::models {
namespace nn = datamunge::nn;
/// @brief Store one annotated box and dataset ignore flags.
///
/// Records own metadata or tensor handles. No input mutation, learned state or fitting.
/// Image/category identifiers and flags are host metadata; overlap and AP numerics
/// use Float64 tensors. Predictions are evaluated without running a detector.
///
/// @par Example
/// @code
/// dm::nn::models::DetectionEvaluationTruth t{1, 3, {0, 0, 10, 10}, false, false};
/// @endcode
/// @par References
/// - Everingham et al. (2010), The PASCAL Visual Object Classes (VOC) Challenge, https://doi.org/10.1007/s11263-009-0275-4; COCO reference evaluator, https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py
/// @see EvaluateDetectionClass(), EvaluateDetectionDataset()
struct DetectionEvaluationTruth {
  int image;   ///< Nonnegative dataset image identifier.
  int label;   ///< Nonnegative category identifier; category IDs need not be contiguous.
  nn::Box box; ///< Finite ordered half-open [ymin,xmin,ymax,xmax] coordinates.
  bool difficult = false; ///< Ignored in VOC mode, ordinary truth in COCO mode.
  bool crowd =
      false; ///< Ignored repeat-match region in COCO mode, ordinary truth in VOC mode.
};
/// @brief Store one scored detection before AP matching.
///
/// Records own metadata or tensor handles. No input mutation, learned state or fitting.
/// Image/category identifiers and flags are host metadata; overlap and AP numerics
/// use Float64 tensors. Predictions are evaluated without running a detector.
///
/// @par Example
/// @code
/// dm::nn::models::DetectionEvaluationPrediction p{1, 3, .9f, {0, 0, 10, 10}};
/// @endcode
/// @par References
/// - Everingham et al. (2010), The PASCAL Visual Object Classes (VOC) Challenge, https://doi.org/10.1007/s11263-009-0275-4; COCO reference evaluator, https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py
/// @see EvaluateDetectionClass(), EvaluateDetectionDataset()
struct DetectionEvaluationPrediction {
  int image;   ///< Nonnegative dataset image identifier.
  int label;   ///< Nonnegative category identifier.
  float score; ///< Finite ranking score; need not be a probability.
  nn::Box box; ///< Finite ordered half-open [ymin,xmin,ymax,xmax] coordinates.
};
/// @brief Retain AP scalars and the precision envelope for one category.
///
/// Records own metadata or tensor handles. No input mutation, learned state or fitting.
/// Image/category identifiers and flags are host metadata; overlap and AP numerics
/// use Float64 tensors. Predictions are evaluated without running a detector.
///
/// @par Example
/// @code
/// const auto r = dm::nn::models::EvaluateDetectionClass({}, {}, {1}, 0, .5, true);
/// @endcode
/// @par References
/// - Everingham et al. (2010), The PASCAL Visual Object Classes (VOC) Challenge, https://doi.org/10.1007/s11263-009-0275-4; COCO reference evaluator, https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py
/// @see EvaluateDetectionClass(), EvaluateDetectionDataset()
struct DetectionClassEvaluation {
  nn::Tensor
      interpolated; ///< Float64 scalar: VOC 11-point or COCO 101-point AP; -1 with no positives.
  nn::Tensor
      integral; ///< Float64 scalar: area under the precision envelope; -1 with no positives.
  nn::Tensor recall; ///< Float64 [D] cumulative recall after ignored detections are removed.
  nn::Tensor
      precision; ///< Float64 [D] nonincreasing precision envelope, in stable descending-score order.
  int positives = 0; ///< Number of selected nonignored truths.
};
/// @brief Retain all-area AP means across categories with selected nonignored truth.
///
/// Records own metadata or tensor handles. No input mutation, learned state or fitting.
/// Image/category identifiers and flags are host metadata; overlap and AP numerics
/// use Float64 tensors. Predictions are evaluated without running a detector.
///
/// @par Example
/// @code
/// const auto r = dm::nn::models::EvaluateDetectionDataset({}, {}, {1}, true);
/// @endcode
/// @par References
/// - Everingham et al. (2010), The PASCAL Visual Object Classes (VOC) Challenge, https://doi.org/10.1007/s11263-009-0275-4; COCO reference evaluator, https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py
/// @see EvaluateDetectionClass(), EvaluateDetectionDataset()
struct DetectionDatasetEvaluation {
  nn::Tensor ap50;  ///< Float64 scalar, VOC integral AP50 or COCO 101-point AP50.
  nn::Tensor ap75;  ///< Float64 scalar COCO AP75; zero in VOC mode.
  nn::Tensor ap;    ///< Float64 scalar COCO AP50:.05:.95; zero in VOC mode.
  nn::Tensor voc07; ///< Float64 scalar VOC 11-point AP50; zero in COCO mode.
  int classes = 0;  ///< Count of categories with at least one selected nonignored truth.
  std::map<int, nn::Tensor>
      per_class50; ///< Category-ID to AP50 tensor, absent categories omitted.
};
/// @brief Internal validation of detection evaluation record metadata and coordinates.
namespace detection_evaluation_detail {
/// @brief Validate finite ordered coordinates and nonnegative metadata for an evaluation record.
/// @param image Nonnegative image ID.
/// @param label Nonnegative category ID.
/// @param box Finite ordered half-open coordinates; zero area is allowed.
/// @throws std::invalid_argument For invalid IDs or box coordinates.
inline void Validate(int image, int label, const nn::Box &box) {
  if (image < 0 || label < 0)
    throw std::invalid_argument("evaluation IDs must be nonnegative");
  for (auto x : box)
    if (!std::isfinite(x))
      throw std::invalid_argument("evaluation coordinates must be finite");
  if (box[0] > box[2] || box[1] > box[3])
    throw std::invalid_argument("evaluation box corners must be ordered");
}
} // namespace detection_evaluation_detail
/// @brief Match one class and compute tensor precision-envelope and sampled AP.
///
/// The optional inclusive_voc switch preserves synthetic-scene source cutoffs;
/// it does not alter the default official VOC strict comparison.
/// Within each image, scores sort stably. VOC chooses maximum IoU even to an
/// already matched truth; duplicates become false positives and difficult hits
/// are ignored. COCO first prefers eligible noncrowd truths, skips matched regular
/// truths, allows repeated crowd hits, and uses intersection/detection-area for
/// crowds. Equal-IoU ties prefer first truth for VOC, last for COCO. COCO caps each
/// image/class at 100 detections before matching. All outcomes then sort stably
/// across images. No area ranges, segmentation, recall summary, or global max100
/// across classes is implied. No gradients or fitting. Tensor overlaps/cumsums,
/// precision envelopes and recall-grid sampling perform all metric arithmetic;
/// host flags and scalar extractions control discrete assignments. Recall grids use
/// k*(1./(S-1)), preserving source floating-point comparisons: 3*.1 lies
/// slightly above stored recall 3/10. Work includes
/// O(D*G) matching and O(D²+D*S) envelope/sampling for S=11 or101.
///
/// @param predictions Finite scored boxes; only selected image IDs and label contribute.
/// @param truth Finite annotated boxes; difficult is ignored for VOC, crowd for COCO.
/// @param images Unique nonnegative selected image IDs; input order breaks cross-image score ties.
/// @param label Nonnegative category identifier.
/// @param threshold Finite IoU cutoff in (0,1]; strict greater for VOC and greater-or-equal for COCO.
/// @param coco True for COCO 101-point/crowd/max100 rules; false for VOC 11-point/difficult rules.
/// @param inclusive_voc Use >= in VOC mode for the synthetic-scene source convention; false by default, ignored for COCO.
/// @return Float64 AP scalars, recall/envelope tensors and positive count; AP -1 if no nonignored truth.
/// @throws std::invalid_argument For malformed coordinates/IDs, nonfinite scores, duplicate selected images or invalid label/cutoff.
/// @par Example
/// @code
/// namespace m = dm::nn::models;
/// const auto r = m::EvaluateDetectionClass({{1, 0, .9f, {0, 0, 10, 20}}},
///                                          {{1, 0, {0, 0, 10, 10}}}, {1}, 0, .5, true);
/// const auto ap = r.interpolated.item(); // 1
/// @endcode
/// @par References
/// - Everingham et al. (2010), The PASCAL Visual Object Classes (VOC) Challenge, https://doi.org/10.1007/s11263-009-0275-4; COCO reference evaluator, https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py
/// @see EvaluateDetectionDataset()
inline DetectionClassEvaluation
EvaluateDetectionClass(const std::vector<DetectionEvaluationPrediction> &predictions,
                       const std::vector<DetectionEvaluationTruth> &truth,
                       const std::vector<int> &images, int label, double threshold, bool coco,
                       bool inclusive_voc = false) {
  if (label < 0 || !std::isfinite(threshold) || threshold <= 0 || threshold > 1)
    throw std::invalid_argument("invalid AP category or threshold");
  std::set<int> ids;
  for (int i : images)
    if (i < 0 || !ids.insert(i).second)
      throw std::invalid_argument("selected image IDs must be unique and nonnegative");
  for (const auto &g : truth)
    detection_evaluation_detail::Validate(g.image, g.label, g.box);
  for (const auto &p : predictions) {
    detection_evaluation_detail::Validate(p.image, p.label, p.box);
    if (!std::isfinite(p.score))
      throw std::invalid_argument("AP scores must be finite");
  }
  nn::NoGradGuard guard;
  const auto type = nn::DType::Float64;
  std::vector<double> scores, tp, fp;
  DetectionClassEvaluation result;
  result.positives = 0;
  for (int image : images) {
    std::vector<nn::Box> gb, pb;
    std::vector<double> ps;
    std::vector<std::int64_t> ignore, crowd;
    for (const auto &g : truth)
      if (g.image == image && g.label == label) {
        gb.push_back(g.box);
        ignore.push_back(coco ? g.crowd : g.difficult);
        crowd.push_back(coco && g.crowd);
        if (!ignore.back())
          ++result.positives;
      }
    for (const auto &p : predictions)
      if (p.image == image && p.label == label) {
        pb.push_back(p.box);
        ps.push_back(p.score);
      }
    if (pb.empty())
      continue;
    const auto ordered = nn::argsort(nn::from_vector(ps), true);
    auto order = ordered.to_vector<std::int64_t>();
    if (coco && order.size() > 100)
      order.resize(100);
    nn::Tensor overlaps;
    if (!gb.empty()) {
      const auto boxes = nn::boxes_from_vector(pb).to(type),
                 targets = nn::boxes_from_vector(gb).to(type);
      overlaps = nn::box_iou(boxes, targets);
      if (coco) {
        const auto area = nn::box_area(boxes).reshape({std::int64_t(pb.size()), 1});
        // IoU*(areaD+areaG)/(1+IoU) equals intersection, including zero intersections.
        const auto intersection =
            overlaps * (area + nn::box_area(targets).reshape({1, std::int64_t(gb.size())})) /
            (1. + overlaps);
        overlaps =
            nn::where(nn::from_vector(crowd).reshape({1, std::int64_t(gb.size())}),
                      intersection / nn::maximum(area, nn::full({}, 1e-12, type)), overlaps);
      }
    }
    auto matched = nn::zeros({std::int64_t(gb.size())}, nn::DType::Bool);
    const auto ignored = nn::from_vector(ignore).to(nn::DType::Bool),
               crowds = nn::from_vector(crowd).to(nn::DType::Bool);
    for (auto index : order) {
      std::int64_t best = -1;
      if (!gb.empty()) {
        const auto row =
            nn::slice_axis(overlaps, 0, index, index + 1).reshape({std::int64_t(gb.size())});
        auto eligible = (coco || inclusive_voc) ? nn::greater_equal(row, threshold)
                                                : nn::greater(row, threshold);
        if (coco) {
          eligible =
              nn::logical_and(eligible, nn::logical_or(nn::logical_not(matched), crowds));
          const auto regular = nn::logical_and(eligible, nn::logical_not(ignored));
          if (nn::any(regular).item<bool>())
            eligible = regular;
        }
        if (nn::any(eligible).item<bool>()) {
          const auto candidates = nn::where(eligible, row, nn::full({}, -1., type));
          best = coco ? std::int64_t(gb.size()) - 1 -
                            nn::argmax(nn::reverse(candidates, {0}), 0).item<std::int64_t>()
                      : nn::argmax(candidates, 0).item<std::int64_t>();
        }
      }
      if (best >= 0 && ignore[best])
        continue;
      const bool hit = best >= 0 && !matched.at({best});
      if (hit)
        matched = nn::logical_or(
            matched, nn::equal(nn::arange(0, std::int64_t(gb.size())), double(best)));
      scores.push_back(ps[index]);
      tp.push_back(hit ? 1. : 0.);
      fp.push_back(hit ? 0. : 1.);
    }
  }
  result.recall = nn::zeros({0}, type);
  result.precision = nn::zeros({0}, type);
  if (!result.positives) {
    result.interpolated = nn::full({}, -1., type);
    result.integral = result.interpolated;
    return result;
  }
  result.interpolated = nn::zeros({}, type);
  result.integral = nn::zeros({}, type);
  if (scores.empty())
    return result;
  const auto order = nn::argsort(nn::from_vector(scores), true);
  const auto cumtp = nn::cumsum(nn::gather(nn::from_vector(tp), order), 0),
             cumfp = nn::cumsum(nn::gather(nn::from_vector(fp), order), 0);
  result.recall = cumtp / double(result.positives);
  const auto precision = cumtp / (cumtp + cumfp);
  std::vector<nn::Tensor> envelope;
  for (std::int64_t i = 0; i < precision.numel(); ++i)
    envelope.push_back(nn::max(nn::slice_axis(precision, 0, i, precision.numel())));
  result.precision = nn::stack(envelope, 0);
  const auto previous = nn::concat(
      {nn::zeros({1}, type), nn::slice_axis(result.recall, 0, 0, result.recall.numel() - 1)},
      0);
  result.integral = nn::sum((result.recall - previous) * result.precision);
  const int samples = coco ? 101 : 11;
  const auto grid = nn::arange(0, samples).to(type) * (1. / double(samples - 1));
  const auto eligible = nn::greater_equal(result.recall.reshape({1, result.recall.numel()}),
                                          grid.reshape({samples, 1}));
  result.interpolated = nn::mean(
      nn::max(nn::where(eligible, result.precision.reshape({1, result.precision.numel()}),
                        nn::zeros({}, type)),
              {1}));
  return result;
}
/// @brief Average dataset box AP over categories with selected nonignored positives.
///
/// Calls EvaluateDetectionClass at .50:.05:.95 in COCO mode, or at .50 in
/// VOC mode. Absent or entirely ignored categories are excluded. Uses exact
/// .75 for the AP75 cutoff. COCO values are a limited all-area evaluator, not
/// the complete official evaluation suite. No parameter mutation or fitting.
///
/// @param predictions Scored boxes; no model inference or score filtering is performed.
/// @param truth Annotated boxes, including difficult and crowd flags.
/// @param images Unique selected nonnegative image IDs, in tie-breaking order.
/// @param coco True for all-area COCO-style box AP; false for VOC integral/11-point AP.
/// @param inclusive_voc Forward the synthetic-scene inclusive VOC cutoff option; false by default.
/// @return Float64 tensor means and per-category AP50; empty eligible categories give zero means.
/// @throws std::invalid_argument For malformed records or selected image IDs as in EvaluateDetectionClass().
/// @par Example
/// @code
/// namespace m = dm::nn::models;
/// const auto r = m::EvaluateDetectionDataset({{1, 0, .9f, {0, 0, 10, 10}}},
///                                            {{1, 0, {0, 0, 10, 10}}}, {1}, true);
/// const auto ap = r.ap.item(); // 1
/// @endcode
/// @par References
/// - Everingham et al. (2010), The PASCAL Visual Object Classes (VOC) Challenge, https://doi.org/10.1007/s11263-009-0275-4; COCO reference evaluator, https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py
/// @see EvaluateDetectionClass()
inline DetectionDatasetEvaluation
EvaluateDetectionDataset(const std::vector<DetectionEvaluationPrediction> &predictions,
                         const std::vector<DetectionEvaluationTruth> &truth,
                         const std::vector<int> &images, bool coco,
                         bool inclusive_voc = false) {
  // Validate even if truth is empty; a sentinel category produces no contributions.
  EvaluateDetectionClass(predictions, truth, images, 0, .5, coco, inclusive_voc);
  DetectionDatasetEvaluation result;
  const auto zero = nn::zeros({}, nn::DType::Float64);
  result.ap50 = zero;
  result.ap75 = zero;
  result.ap = zero;
  result.voc07 = zero;
  std::set<int> labels;
  for (const auto &g : truth)
    labels.insert(g.label);
  std::vector<nn::Tensor> aps50, aps75, aps, vocs;
  for (int label : labels) {
    const auto first =
        EvaluateDetectionClass(predictions, truth, images, label, .5, coco, inclusive_voc);
    if (!first.positives)
      continue;
    ++result.classes;
    result.per_class50[label] = coco ? first.interpolated : first.integral;
    aps50.push_back(result.per_class50[label]);
    if (coco) {
      std::vector<nn::Tensor> thresholds;
      for (int k = 0; k < 10; ++k) {
        const auto r =
            EvaluateDetectionClass(predictions, truth, images, label, .5 + .05 * k, true);
        thresholds.push_back(r.interpolated);
        if (k == 5)
          aps75.push_back(r.interpolated);
      }
      aps.push_back(nn::mean(nn::stack(thresholds, 0)));
    } else
      vocs.push_back(first.interpolated);
  }
  if (result.classes) {
    result.ap50 = nn::mean(nn::stack(aps50, 0));
    if (coco) {
      result.ap75 = nn::mean(nn::stack(aps75, 0));
      result.ap = nn::mean(nn::stack(aps, 0));
    } else
      result.voc07 = nn::mean(nn::stack(vocs, 0));
  }
  return result;
}
} // namespace datamunge::nn::models
