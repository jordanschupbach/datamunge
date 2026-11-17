// VOC-style detection mAP for the nn example ports (tflow
// common/nn/metrics.h). Box IoU comes from the library (nn::box_iou in
// datamunge/nn/boxes.hpp); the mAP rules below are this evaluator's own.
//
// Boxes use continuous, half-open xyxy coordinates, without pixel +1.
// Detections are sorted by score, with input order breaking ties. Each matches
// the highest-IoU same-image, same-class truth; duplicates are false positives.
// Matching requires IoU >= the selected threshold. AP integrates the precision
// envelope over all recall changes (VOC2010). No crowd/difficult rules, area
// ranges or maximum-detection caps: this is not COCO AP. Classes with no truth
// have NaN AP and are excluded from mAP.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

#include <datamunge/nn/boxes.hpp>

namespace detection {

struct Box {
  double x1, y1, x2, y2;
};

inline void Validate(Box b) {
  for (double v : {b.x1, b.y1, b.x2, b.y2})
    if (!std::isfinite(v)) throw std::invalid_argument("box coordinates must be finite");
  if (b.x2 < b.x1 || b.y2 < b.y1) throw std::invalid_argument("inverted box");
}

inline double IoU(Box a, Box b) {
  Validate(a);
  Validate(b);
  // the library's boxes are [y1, x1, y2, x2]
  return datamunge::nn::box_iou(datamunge::nn::BoxT<double>{a.y1, a.x1, a.y2, a.x2},
                                datamunge::nn::BoxT<double>{b.y1, b.x1, b.y2, b.x2});
}

struct GroundTruth {
  int image, label;
  Box box;
};
struct Detection {
  int    image, label;
  Box    box;
  double score;
};
struct DetectionMetrics {
  std::vector<double> ap;
  double              map;
};

inline DetectionMetrics DetectionMap(const std::vector<GroundTruth>& truth, const std::vector<Detection>& detections,
                                     std::size_t classes, double threshold = .5) {
  const double missing = std::numeric_limits<double>::quiet_NaN();
  if (!classes || !std::isfinite(threshold) || threshold <= 0 || threshold > 1)
    throw std::invalid_argument("invalid detection evaluation configuration");
  for (const auto& t : truth) {
    Validate(t.box);
    if (t.image < 0 || t.label < 0 || std::size_t(t.label) >= classes)
      throw std::invalid_argument("invalid truth image/class");
  }
  for (const auto& d : detections) {
    Validate(d.box);
    if (d.image < 0 || d.label < 0 || std::size_t(d.label) >= classes || !std::isfinite(d.score))
      throw std::invalid_argument("invalid detection image/class/score");
  }
  DetectionMetrics result{std::vector<double>(classes, missing), 0};
  std::size_t      present = 0;
  for (std::size_t c = 0; c < classes; ++c) {
    std::vector<GroundTruth> gt;
    std::vector<Detection>   det;
    for (const auto& t : truth)
      if (std::size_t(t.label) == c) gt.push_back(t);
    for (const auto& d : detections)
      if (std::size_t(d.label) == c) det.push_back(d);
    if (gt.empty()) continue;
    std::stable_sort(det.begin(), det.end(), [](const auto& a, const auto& b) { return a.score > b.score; });
    std::vector<bool>   matched(gt.size(), false);
    std::vector<double> recall{0}, precision{0};
    double              tp = 0;
    for (std::size_t i = 0; i < det.size(); ++i) {
      double      best  = -1;
      std::size_t match = gt.size();
      for (std::size_t j = 0; j < gt.size(); ++j)
        if (gt[j].image == det[i].image) {
          const double overlap = IoU(gt[j].box, det[i].box);
          if (overlap > best) best = overlap, match = j;
        }
      if (match < gt.size() && best >= threshold && !matched[match]) matched[match] = true, ++tp;
      recall.push_back(tp / static_cast<double>(gt.size()));
      precision.push_back(tp / static_cast<double>(i + 1));
    }
    recall.push_back(1);
    precision.push_back(0);
    for (std::size_t i = precision.size() - 1; i > 0; --i) precision[i - 1] = std::max(precision[i - 1], precision[i]);
    double ap = 0;
    for (std::size_t i = 1; i < recall.size(); ++i) ap += (recall[i] - recall[i - 1]) * precision[i];
    result.ap[c] = ap;
    result.map += ap;
    ++present;
  }
  result.map = present ? result.map / static_cast<double>(present) : missing;
  return result;
}

}  // namespace detection
