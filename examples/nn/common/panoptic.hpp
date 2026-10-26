// Panoptic overlap resolution and category-averaged PQ/SQ/RQ with void/crowd
// (port of tflow's common/ml/panoptic.h).
// https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/meta_arch/panoptic_fpn.py
// https://github.com/cocodataset/panopticapi/blob/master/panopticapi/evaluation.py
//
// MergePanoptic pastes instances in descending score order (scores below
// `score` dropped), rejecting a mask whose already-claimed fraction exceeds
// `overlap` (or is fully claimed) and trimming the rest; then each stuff class
// of the semantic map claims its unclaimed pixels if they cover at least
// `stuff_area` pixels. Id 0 is void. PanopticStats matches segments of the
// same category with IoU > .5 (void pixels of the prediction excluded from
// the union), skips crowd targets, and ignores unmatched predictions lying
// mostly in void or same-class crowd regions.
#pragma once

#include <algorithm>
#include <cmath>
#include <map>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

#include "common/checks.hpp"
#include "common/instance_segmentation.hpp"

namespace panoptic {

struct PanopticSegment {
  int  label = 0;
  bool thing = false, crowd = false;
};
struct PanopticMap {
  std::vector<int>               ids;
  std::map<int, PanopticSegment> segments;
};
struct PqStat {
  double iou = 0;
  int    tp = 0, fp = 0, fn = 0;
};
using PqStats = std::map<int, PqStat>;
struct PqResult {
  double pq = 0, sq = 0, rq = 0;
  int    classes = 0;
};

/// semantic_categories[c] is the panoptic category of semantic class c (negative: not stuff).
inline PanopticMap MergePanoptic(const std::vector<int>& semantic, const std::vector<instance::MaskDetection>& instances,
                                 const std::vector<int>& semantic_categories, float score = .5f, float overlap = .5f,
                                 int stuff_area = 32) {
  if (semantic_categories.empty() || !(score >= 0 && score <= 1) || !(overlap >= 0 && overlap <= 1) || stuff_area < 1)
    throw std::invalid_argument("invalid panoptic merge configuration");
  PanopticMap out;
  out.ids.resize(semantic.size(), 0);
  std::vector<int> order(instances.size());
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
    return instances[std::size_t(a)].box.score > instances[std::size_t(b)].box.score;
  });
  int id = 0;
  for (int k : order) {
    const auto& inst = instances[std::size_t(k)];
    if (inst.mask.size() != semantic.size() || !std::isfinite(inst.box.score))
      throw std::invalid_argument("invalid panoptic instance mask/score");
    if (inst.box.score < score) continue;
    int area = 0, intersect = 0;
    for (std::size_t i = 0; i < semantic.size(); ++i)
      if (inst.mask[i]) {
        ++area;
        intersect += out.ids[i] != 0;
      }
    if (!area || intersect == area || double(intersect) / area > overlap) continue;
    ++id;
    out.segments[id] = {inst.box.label, true, false};
    for (std::size_t i = 0; i < semantic.size(); ++i)
      if (inst.mask[i] && !out.ids[i]) out.ids[i] = id;
  }
  for (int value : semantic)
    if (value < 0 || value >= int(semantic_categories.size())) throw std::invalid_argument("unknown semantic panoptic class");
  for (std::size_t label = 0; label < semantic_categories.size(); ++label) {
    if (semantic_categories[label] < 0) continue;
    int area = 0;
    for (std::size_t i = 0; i < semantic.size(); ++i) area += semantic[i] == int(label) && out.ids[i] == 0;
    if (area < stuff_area) continue;
    ++id;
    out.segments[id] = {semantic_categories[label], false, false};
    for (std::size_t i = 0; i < semantic.size(); ++i)
      if (semantic[i] == int(label) && !out.ids[i]) out.ids[i] = id;
  }
  return out;
}

/// Pixel area per id (0 = void); rejects pixels without metadata and empty or invalid segments.
inline std::map<int, int> PanopticAreas(const PanopticMap& map) {
  std::map<int, int> area;
  for (int id : map.ids) {
    if (id < 0 || (id && !map.segments.count(id))) throw std::invalid_argument("panoptic pixel lacks segment metadata");
    ++area[id];
  }
  for (const auto& [id, segment] : map.segments)
    if (id <= 0 || !area[id] || segment.label < 0) throw std::invalid_argument("invalid/empty panoptic segment");
  return area;
}

inline PqStats PanopticStats(const PanopticMap& truth, const PanopticMap& prediction) {
  if (truth.ids.size() != prediction.ids.size()) throw std::invalid_argument("panoptic grids differ");
  auto                          ta = PanopticAreas(truth), pa = PanopticAreas(prediction);
  std::map<std::pair<int, int>, int> intersections;
  for (std::size_t i = 0; i < truth.ids.size(); ++i) ++intersections[{truth.ids[i], prediction.ids[i]}];
  std::map<int, bool> matched_truth, matched_prediction;
  PqStats             stats;
  for (const auto& [pair, intersection] : intersections) {
    const auto [gt, pred] = pair;
    if (!gt || !pred) continue;
    const auto& target = truth.segments.at(gt);
    const auto& found  = prediction.segments.at(pred);
    if (target.crowd || target.label != found.label) continue;
    const int    united = ta[gt] + pa[pred] - intersection - intersections[{0, pred}];
    const double iou    = double(intersection) / united;
    if (iou > .5) {
      check::Require(!matched_truth[gt] && !matched_prediction[pred], "panoptic matching uniqueness");
      matched_truth[gt] = matched_prediction[pred] = true;
      ++stats[target.label].tp;
      stats[target.label].iou += iou;
    }
  }
  for (const auto& [id, segment] : truth.segments)
    if (!segment.crowd && !matched_truth[id]) ++stats[segment.label].fn;
  for (const auto& [id, segment] : prediction.segments)
    if (!matched_prediction[id]) {
      int ignored = intersections[{0, id}];
      for (const auto& [gt, target] : truth.segments)
        if (target.crowd && target.label == segment.label) ignored += intersections[{gt, id}];
      if (double(ignored) / pa[id] > .5) continue;
      ++stats[segment.label].fp;
    }
  return stats;
}

inline void AddPanopticStats(PqStats& total, const PqStats& sample) {
  for (const auto& [label, s] : sample) {
    auto& t = total[label];
    t.iou += s.iou;
    t.tp += s.tp;
    t.fp += s.fp;
    t.fn += s.fn;
  }
}

/// Category-averaged PQ/SQ/RQ; subset -1 all, 1 things, 0 stuff.
inline PqResult PanopticQuality(const PqStats& stats, const std::vector<bool>& things, int subset = -1) {
  PqResult out;
  for (const auto& [label, s] : stats) {
    if (label < 0 || label >= int(things.size())) throw std::invalid_argument("unknown panoptic category");
    if (subset >= 0 && things[std::size_t(label)] != bool(subset)) continue;
    const double denominator = s.tp + .5 * s.fp + .5 * s.fn;
    if (!denominator) continue;
    ++out.classes;
    out.pq += s.iou / denominator;
    out.sq += s.tp ? s.iou / s.tp : 0;
    out.rq += s.tp / denominator;
  }
  if (out.classes) {
    out.pq /= out.classes;
    out.sq /= out.classes;
    out.rq /= out.classes;
  }
  return out;
}

}  // namespace panoptic
