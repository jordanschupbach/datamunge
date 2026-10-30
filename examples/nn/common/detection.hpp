// Object-detection helpers shared by the detection example ports: a port of
// tflow's common/ml/detection.h (NMS wrappers, VOC/COCO average precision,
// anchors, box coding, anchor matching, the detector training loop) plus the
// pieces of common/ml/two_stage.h, common/ml/ssd.h and
// common/data/detection_scenes.h that the ports need.
//
// The general box math lives in the library (datamunge/nn/boxes.hpp): IoU,
// Faster R-CNN box coding with variances (0.1, 0.1, 0.2, 0.2), anchor grids,
// argmax anchor matching and TF NonMaxSuppressionV3/V5. This header keeps
// tflow's example-level conventions on top of it: input validation, class-wise
// NMS with a per-image budget, SSD's bipartite matching and the AP rules.
//
// Boxes are [ymin, xmin, ymax, xmax) in pixels (nn::Box, float).
// Average precision (per class, then averaged over classes with ground
// truth): detections are sorted by score and matched in that order. The VOC
// rule matches a detection to the ground truth it overlaps most; a match
// needs IoU >= t and an unmatched target, otherwise the detection is a false
// positive. AP is the area under the precision envelope (VOC 2010+, "all
// points") or the mean of the envelope at recalls 0, 0.1, ..., 1 (VOC 2007).
// The COCO rule matches each detection to the best still-unmatched target
// with IoU >= t and averages the envelope at 101 recall points; COCO AP
// averages that over t = 0.50, 0.55, ..., 0.95. The 11- and 101-point means
// are sums divided once (tflow adds 1/11 eleven times, which can exceed 1 by
// an ulp and trips its own "AP in [0, 1]" check).
//
// The training loop is eager: hooks build parameters, return a training loss
// (updating model state such as batch-norm statistics in place) and return
// detections. Checkpoints are .npz parameter and state stores
// (<prefix>_params.npz, <prefix>_state.npz) instead of TF bundles.
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <datamunge/nn/boxes.hpp>
#include <datamunge/nn/nn.hpp>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "common/checks.hpp"
#include "common/paths.hpp"

namespace detector {

namespace nn = datamunge::nn;
using Box = nn::Box;

/// Synthetic detection scenes (tflow data::DetectionScenes): grey images and
/// per-scene boxes and labels.
struct DetectionScenes {
  int height = 64, width = 64;
  std::vector<std::uint8_t> pixels; // scenes x height x width
  std::vector<std::vector<Box>> boxes;
  std::vector<std::vector<int>> labels;
  [[nodiscard]] int size() const {
    return int(boxes.size());
  }
  [[nodiscard]] const std::uint8_t *image(int i) const {
    return pixels.data() + std::size_t(i) * height * width;
  }
};

struct Detection {
  int image = 0, label = 0;
  float score = 0;
  Box box{};
};

/// Greedy NMS (TF NonMaxSuppressionV3 semantics via nn::non_max_suppression):
/// indices of kept boxes, highest score first. Rejects mismatched inputs,
/// non-finite values, reversed corners and invalid thresholds or limits.
inline std::vector<int> NonMaxSuppression(const std::vector<Box> &boxes,
                                          const std::vector<float> &scores, float iou,
                                          int max_out = 1 << 30) {
  if (boxes.size() != scores.size() || !(iou >= 0 && iou <= 1) || max_out < 0)
    throw std::invalid_argument(
        "NMS requires matching box/score counts, IoU in [0,1] and a nonnegative output limit");
  for (std::size_t i = 0; i < boxes.size(); ++i) {
    if (!std::isfinite(scores[i]))
      throw std::invalid_argument("NMS scores must be finite");
    for (float coordinate : boxes[i])
      if (!std::isfinite(coordinate))
        throw std::invalid_argument("NMS coordinates must be finite");
    if (boxes[i][2] < boxes[i][0] || boxes[i][3] < boxes[i][1])
      throw std::invalid_argument("NMS boxes must have ordered corners");
  }
  const auto kept = nn::non_max_suppression(boxes, scores, max_out, iou);
  return std::vector<int>(kept.begin(), kept.end());
}

/// Per-image, per-class NMS, then the top max_per_image detections of each image.
inline std::vector<Detection> ClassNms(const std::vector<Detection> &detections, float iou,
                                       int max_per_image = 100) {
  if (!(iou >= 0 && iou <= 1) || max_per_image < 0)
    throw std::invalid_argument("invalid class-wise NMS threshold or output limit");
  std::map<std::pair<int, int>, std::vector<Detection>> groups;
  for (const auto &d : detections)
    groups[{d.image, d.label}].push_back(d);
  std::map<int, std::vector<Detection>> per_image;
  for (auto &[key, group] : groups) {
    std::vector<Box> boxes;
    std::vector<float> scores;
    for (const auto &d : group) {
      boxes.push_back(d.box);
      scores.push_back(d.score);
    }
    for (int k : NonMaxSuppression(boxes, scores, iou))
      per_image[key.first].push_back(group[std::size_t(k)]);
  }
  std::vector<Detection> out;
  for (auto &[image, list] : per_image) {
    std::stable_sort(list.begin(), list.end(), [](const Detection &a, const Detection &b) {
      return a.score > b.score;
    });
    if (int(list.size()) > max_per_image)
      list.resize(std::size_t(max_per_image));
    out.insert(out.end(), list.begin(), list.end());
  }
  return out;
}

struct ApResult {
  double ap50 = 0, ap50_voc07 = 0, ap75 = 0, coco = 0;
  std::vector<double> per_class50; // VOC all-point AP at IoU 0.5 (NaN without ground truth)
};

/// AP of one class at one IoU threshold: {all-point, 11-point, 101-point}; `voc` selects the matching rule.
inline std::array<double, 3> ClassAp(const std::vector<Detection> &detections,
                                     const DetectionScenes &scenes,
                                     const std::vector<int> &rows, int label, float threshold,
                                     bool voc) {
  std::map<int, std::vector<Box>> truth;
  int positives = 0;
  for (int r : rows)
    for (std::size_t k = 0; k < scenes.boxes[std::size_t(r)].size(); ++k)
      if (scenes.labels[std::size_t(r)][k] == label) {
        truth[r].push_back(scenes.boxes[std::size_t(r)][k]);
        ++positives;
      }
  if (!positives)
    return {NAN, NAN, NAN};
  std::vector<Detection> dets;
  for (const auto &d : detections)
    if (d.label == label)
      dets.push_back(d);
  std::stable_sort(dets.begin(), dets.end(), [](const Detection &a, const Detection &b) {
    return a.score > b.score;
  });
  std::map<int, std::vector<char>> used;
  for (auto &[image, boxes] : truth)
    used[image].assign(boxes.size(), 0);
  std::vector<double> precision, recall;
  int tp = 0, fp = 0;
  for (const auto &d : dets) {
    int best = -1;
    float best_iou = voc ? -1 : threshold;
    const auto it = truth.find(d.image);
    if (it != truth.end())
      for (std::size_t k = 0; k < it->second.size(); ++k) {
        const float iou = nn::box_iou(d.box, it->second[k]);
        if (voc ? iou > best_iou : (!used[d.image][k] && iou >= best_iou)) {
          best_iou = iou;
          best = int(k);
        }
      }
    const bool hit = best >= 0 && best_iou >= threshold && !used[d.image][std::size_t(best)];
    if (hit) {
      used[d.image][std::size_t(best)] = 1;
      ++tp;
    } else {
      ++fp;
    }
    precision.push_back(double(tp) / (tp + fp));
    recall.push_back(double(tp) / positives);
  }
  std::vector<double> envelope = precision; // precision envelope
  for (int i = int(envelope.size()) - 2; i >= 0; --i)
    envelope[std::size_t(i)] =
        std::max(envelope[std::size_t(i)], envelope[std::size_t(i) + 1]);
  double all = 0, previous = 0;
  for (std::size_t i = 0; i < envelope.size(); ++i) {
    all += (recall[i] - previous) * envelope[i];
    previous = recall[i];
  }
  auto at = [&](double r) {
    for (std::size_t i = 0; i < recall.size(); ++i)
      if (recall[i] >= r)
        return envelope[i];
    return 0.;
  };
  double eleven = 0, hundred = 0;
  for (int i = 0; i <= 10; ++i)
    eleven += at(i / 10.);
  for (int i = 0; i <= 100; ++i)
    hundred += at(i / 100.);
  return {all, eleven / 11, hundred / 101};
}

inline ApResult EvaluateDetections(const std::vector<Detection> &detections,
                                   const DetectionScenes &scenes, const std::vector<int> &rows,
                                   int classes) {
  ApResult r;
  r.per_class50.assign(std::size_t(classes), NAN);
  auto mean = [](const std::vector<double> &v) {
    double s = 0;
    int n = 0;
    for (double x : v)
      if (!std::isnan(x)) {
        s += x;
        ++n;
      }
    return n ? s / n : 0.;
  };
  const auto count = static_cast<std::size_t>(classes);
  std::vector<double> voc50(count), voc07(count), coco75(count), coco(count, 0);
  for (int c = 0; c < classes; ++c) {
    const auto v = ClassAp(detections, scenes, rows, c, .5f, true);
    const auto k = std::size_t(c);
    voc50[k] = v[0];
    voc07[k] = v[1];
    r.per_class50[k] = v[0];
    if (std::isnan(v[0])) {
      coco[k] = coco75[k] = NAN;
      continue;
    }
    for (int t = 0; t < 10; ++t) {
      const auto a = ClassAp(detections, scenes, rows, c, .5f + .05f * t, false)[2];
      coco[k] += a / 10;
      if (t == 5)
        coco75[k] = a;
    }
  }
  r.ap50 = mean(voc50);
  r.ap50_voc07 = mean(voc07);
  r.ap75 = mean(coco75);
  r.coco = mean(coco);
  return r;
}

/// Anchors centred on a stride grid; ratio = height / width (nn::anchor_grid).
inline std::vector<Box> AnchorGrid(int rows, int cols, float stride,
                                   const std::vector<float> &sizes,
                                   const std::vector<float> &ratios) {
  return nn::boxes_to_vector(
      nn::anchor_grid(rows, cols, stride, std::vector<double>(sizes.begin(), sizes.end()),
                      std::vector<double>(ratios.begin(), ratios.end())));
}

/// Faster R-CNN deltas (dy, dx, dh, dw) / (0.1, 0.1, 0.2, 0.2) (nn::encode_box).
inline std::array<float, 4> EncodeBox(const Box &anchor, const Box &box) {
  return nn::encode_box(anchor, box);
}
/// Inverse of EncodeBox, log scales clamped at 4 (nn::decode_box).
inline Box DecodeBox(const Box &anchor, const float *delta) {
  return nn::decode_box(anchor, {delta[0], delta[1], delta[2], delta[3]});
}

/// Anchor assignment: ground-truth index (positive), -1 (negative) or -2
/// (ignored). Every target also claims its best anchor, if that overlaps at all.
inline std::vector<int> MatchAnchors(const std::vector<Box> &anchors,
                                     const std::vector<Box> &truth, float positive,
                                     float negative) {
  const auto m = nn::match_boxes(anchors, truth, positive, negative, true);
  return std::vector<int>(m.begin(), m.end());
}

/// SSD matching: first greedily give every target a distinct anchor (highest
/// remaining IoU pair), then assign the remaining anchors with IoU >= 0.5.
inline std::vector<int> MatchSsd(const std::vector<Box> &anchors,
                                 const std::vector<Box> &truth) {
  const auto matched = nn::models::MatchSsd(
      nn::box_iou(nn::boxes_from_vector(anchors), nn::boxes_from_vector(truth)));
  return matched.to_vector<int>();
}
using nn::models::BuildRpn;
using nn::models::Rpn;
using nn::models::RpnLevel;
using nn::models::RpnLoss;
inline nn::Tensor SmoothL1(const nn::Tensor &difference) {
  return nn::models::DetectorSmoothL1(difference);
}

// ---- training loop -------------------------------------------------------------------

inline std::string ApSummary(const ApResult &r) {
  std::ostringstream s;
  s << std::fixed << std::setprecision(4) << "AP50 " << r.ap50 << " (VOC07 11-point "
    << r.ap50_voc07 << "), AP75 " << r.ap75 << ", COCO AP " << r.coco;
  return s.str();
}

/// Output directory of a training run (output/nn/models/<name>).
inline std::filesystem::path TrainingDirectory(const std::string &name) {
  return paths::OutputRoot() / "models" / name;
}

/// Adam on the hook's loss plus 0.5 * l2 * sum of squared kernels (rank >= 2),
/// the best validation AP50 checkpoint restored for the test evaluation.
struct DetectorTrainingConfig {
  std::string name;
  // Optional parameter/state checkpoint prefix; Adam moments intentionally start fresh.
  std::string warm_start;
  int epochs = 12, batch = 32, fit = 0, validation = 500, classes = 10;
  float learning_rate = 1e-3f, l2 = 1e-4f;
  double minimum_test_ap50 = .5;
  // Existing callers retain accuracy gating; short teaching examples opt out
  // explicitly and must instead demonstrate finite, decreasing epoch loss.
  bool accuracy = true;
};

inline void CheckDetectorDemonstration(const DetectorTrainingConfig &cfg,
                                       const std::vector<double> &epoch_losses,
                                       const ApResult &result) {
  for (double value : epoch_losses)
    check::Require(std::isfinite(value), "finite detector epoch loss");
  for (double value : {result.ap50, result.ap50_voc07, result.ap75, result.coco})
    check::Require(std::isfinite(value) && value >= 0 && value <= 1,
                   "finite detector AP in [0,1]");
  if (cfg.accuracy) {
    check::Require(result.ap50 >= cfg.minimum_test_ap50,
                   "detector demonstration AP50 threshold");
  } else {
    check::Require(epoch_losses.size() >= 2,
                   "short learning demonstration needs at least two epochs");
    check::Require(epoch_losses.back() <
                       epoch_losses.front() -
                           1e-4 * std::max(1., std::abs(epoch_losses.front())),
                   "short detector demonstration must reduce mean training loss");
  }
}

struct DetectorHooks {
  /// Creates every parameter (and model-state entry) before training.
  std::function<void(nn::ParameterStore &params, nn::ParameterStore &state)> build;
  /// Training loss for the given scene rows; may update `state` in place (one call per step).
  std::function<nn::Tensor(nn::ParameterStore &, nn::ParameterStore &, const DetectionScenes &,
                           const std::vector<int> &)>
      loss;
  /// Inference detections (after NMS) for the given rows; `image` fields index scenes.
  std::function<std::vector<Detection>(nn::ParameterStore &, nn::ParameterStore &,
                                       const DetectionScenes &, const std::vector<int> &)>
      detect;
  /// Optional held-out diagnostics after restoring the validation-best checkpoint.
  std::function<void(nn::ParameterStore &, nn::ParameterStore &, const DetectionScenes &,
                     const std::vector<int> &, const std::filesystem::path &)>
      inspect;
};

inline void SaveCheckpoint(const nn::ParameterStore &params, const nn::ParameterStore &state,
                           const std::string &prefix) {
  params.save(prefix + "_params.npz");
  state.save(prefix + "_state.npz");
}
inline void RestoreCheckpoint(nn::ParameterStore &params, nn::ParameterStore &state,
                              const std::string &prefix) {
  params.load(prefix + "_params.npz");
  state.load(prefix + "_state.npz");
}

} // namespace detector
