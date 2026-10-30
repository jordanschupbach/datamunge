#pragma once
// Presentation containers forward all overlap,area,matching and quality arithmetic.
#include "common/checks.hpp"
#include "common/instance_segmentation.hpp"
#include "datamunge/nn/panoptic_models.hpp"
namespace panoptic {
namespace nn = datamunge::nn;
using nn::models::PanopticSegment;
struct PanopticMap {
  std::vector<int> ids;
  std::map<int, PanopticSegment> segments;
};
struct PqStat {
  double iou = 0;
  int tp = 0, fp = 0, fn = 0;
};
using PqStats = std::map<int, PqStat>;
struct PqResult {
  double pq = 0, sq = 0, rq = 0;
  int classes = 0;
};
inline nn::models::TensorPanopticMap TensorMap(const PanopticMap &map) {
  return {nn::from_vector(std::vector<std::int64_t>(map.ids.begin(), map.ids.end())),
          map.segments};
}
inline nn::models::TensorMaskDetections
TensorInstances(const std::vector<instance::MaskDetection> &instances, int height, int width) {
  std::vector<float> scores;
  std::vector<std::int64_t> labels;
  std::vector<int> masks;
  for (const auto &d : instances) {
    if (d.mask.size() != std::size_t(height) * width)
      throw std::invalid_argument("instance mask geometry mismatch");
    scores.push_back(d.box.score);
    labels.push_back(d.box.label);
    masks.insert(masks.end(), d.mask.begin(), d.mask.end());
  }
  const auto D = std::int64_t(instances.size());
  return {{nn::zeros({D, 4}), nn::from_vector(scores), nn::from_vector(labels),
           nn::zeros({D}, nn::DType::Int64)},
          nn::not_equal(nn::from_vector(masks, {D, height, width}), 0.)};
}
inline PanopticMap MergePanoptic(const std::vector<int> &semantic,
                                 const std::vector<instance::MaskDetection> &instances,
                                 const std::vector<int> &semantic_categories,
                                 float score = .5f, float overlap = .5f, int stuff_area = 32) {
  const auto map = nn::models::MergeTensorPanoptic(
      nn::from_vector(std::vector<std::int64_t>(semantic.begin(), semantic.end()),
                      {1, std::int64_t(semantic.size())}),
      TensorInstances(instances, 1, int(semantic.size())), semantic_categories, score, overlap,
      stuff_area);
  return {map.ids.to_vector<int>(), map.segments};
}
inline std::map<int, int> PanopticAreas(const PanopticMap &map) {
  const auto areas = nn::models::PanopticAreas(TensorMap(map)).to_vector<int>();
  std::map<int, int> result;
  if (areas[0])
    result[0] = areas[0];
  std::size_t i = 1;
  for (const auto &[id, segment] : map.segments)
    result[id] = areas[i++];
  return result;
}
inline nn::Tensor DenseStats(const PqStats &stats, int classes) {
  std::vector<double> values(std::size_t(classes) * 4);
  for (const auto &[label, s] : stats) {
    if (label < 0 || label >= classes)
      throw std::invalid_argument("unknown panoptic category");
    values[label * 4] = s.iou;
    values[label * 4 + 1] = s.tp;
    values[label * 4 + 2] = s.fp;
    values[label * 4 + 3] = s.fn;
  }
  return nn::from_vector(values, {classes, 4});
}
inline PqStats LegacyStats(const nn::Tensor &statistics) {
  PqStats result;
  const auto data = statistics.to_vector<double>();
  for (std::int64_t c = 0; c < statistics.shape()[0]; ++c)
    if (data[c * 4 + 1] || data[c * 4 + 2] || data[c * 4 + 3])
      result[int(c)] = {data[c * 4], int(data[c * 4 + 1]), int(data[c * 4 + 2]),
                        int(data[c * 4 + 3])};
  return result;
}
inline PqStats PanopticStats(const PanopticMap &truth, const PanopticMap &prediction) {
  int classes = 0;
  for (const auto *map : {&truth, &prediction})
    for (const auto &[id, segment] : map->segments)
      classes = std::max(classes, segment.label + 1);
  if (!classes) {
    PanopticAreas(truth);
    PanopticAreas(prediction);
    if (truth.ids.size() != prediction.ids.size())
      throw std::invalid_argument("panoptic grids differ");
    return {};
  }
  return LegacyStats(
      nn::models::PanopticStatistics(TensorMap(truth), TensorMap(prediction), classes));
}
inline void AddPanopticStats(PqStats &total, const PqStats &sample) {
  int classes = 0;
  for (const PqStats *stats : std::array<const PqStats *,2>{&total, &sample})
    for (const auto &[label, s] : *stats)
      classes = std::max(classes, label + 1);
  if (classes)
    total = LegacyStats(DenseStats(total, classes) + DenseStats(sample, classes));
}
inline PqResult PanopticQuality(const PqStats &stats, const std::vector<bool> &things,
                                int subset = -1) {
  const auto quality = nn::models::PanopticQuality(DenseStats(stats, int(things.size())),
                                                   nn::from_vector(things), subset);
  return {quality.pq.item(), quality.sq.item(), quality.rq.item(),
          int(quality.classes.item<std::int64_t>())};
}
} // namespace panoptic
