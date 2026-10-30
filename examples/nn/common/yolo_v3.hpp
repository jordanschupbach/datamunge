#pragma once
#include "common/yolo_v2.hpp"
#include "datamunge/nn/yolo_v3_models.hpp"
namespace yolo {
using nn::models::YoloV3;
using nn::models::YoloV3Config;
using nn::models::YoloV3Outputs;
inline nn::Tensor LogisticCrossEntropy(const nn::Tensor &x, const nn::Tensor &y) {
  return nn::models::YoloLogisticCrossEntropy(x, y);
}
struct YoloV3LevelTargets {
  std::vector<float> geometry, classes, positive, objectness_weight, geometry_weight, areas;
  nn::Shape grid;
  int class_count = 0;
};
struct YoloV3Targets {
  std::array<YoloV3LevelTargets, 3> levels;
  int collisions = 0;
};
inline nn::Tensor YoloV3Loss(const std::array<nn::Tensor, 3> &raw, const YoloV3Targets &t,
                             int classes) {
  nn::models::YoloV3Targets target;
  for (int l = 0; l < 3; ++l) {
    const auto &s = raw[l].shape();
    const auto &g = t.levels[l];
    const nn::Shape grid = {s[0], s[1], s[2], 3};
    target.levels[l] = {
        nn::from_vector(g.geometry, {s[0], s[1], s[2], 3, 4}).to(raw[l].dtype()),
        nn::from_vector(g.classes, {s[0], s[1], s[2], 3, classes}).to(raw[l].dtype()),
        nn::from_vector(g.positive, grid).to(raw[l].dtype()),
        nn::from_vector(g.objectness_weight, grid).to(raw[l].dtype()),
        nn::from_vector(g.geometry_weight, grid).to(raw[l].dtype()),
        g.areas.empty() ? nn::zeros(grid, raw[l].dtype())
                        : nn::from_vector(g.areas, grid).to(raw[l].dtype())};
  }
  target.collisions = t.collisions;
  return nn::models::YoloV3Loss(raw, target, classes);
}
inline YoloV3Targets MakeYoloV3Targets(const std::array<std::vector<float>, 3> &raw,
                                       const std::array<nn::Shape, 3> &shapes,
                                       const std::vector<Dimension> &priors,
                                       const DetectionScenes &scenes,
                                       const std::vector<int> &rows) {
  nn::models::TensorDetectionScenes data;
  data.images = nn::zeros({std::int64_t(rows.size()), scenes.height, scenes.width, 1});
  for (auto row : rows) {
    if (row < 0 || row >= scenes.size())
      throw std::invalid_argument("invalid YOLO v3 row");
    data.boxes.push_back(nn::boxes_from_vector(scenes.boxes[row]));
    data.labels.push_back(nn::from_vector(
        std::vector<std::int64_t>(scenes.labels[row].begin(), scenes.labels[row].end())));
  }
  std::vector<std::int64_t> ids(rows.size());
  std::iota(ids.begin(), ids.end(), 0);
  std::array<nn::Tensor, 3> predictions;
  for (int l = 0; l < 3; ++l)
    predictions[l] = nn::from_vector(raw[l], shapes[l]);
  const auto t =
      nn::models::MakeYoloV3Targets(predictions, DimensionTensor(priors), data, ids);
  YoloV3Targets out;
  out.collisions = int(t.collisions);
  for (int l = 0; l < 3; ++l) {
    const auto &g = t.levels[l];
    out.levels[l] = {g.geometry.to_vector<float>(),
                     g.classes.to_vector<float>(),
                     g.positive.to_vector<float>(),
                     g.objectness_weight.to_vector<float>(),
                     g.geometry_weight.to_vector<float>(),
                     g.areas.to_vector<float>(),
                     g.positive.shape(),
                     int(g.classes.shape().back())};
  }
  return out;
}
inline void DecodeYoloV3Level(const nn::Tensor &raw, int level,
                              const std::vector<Dimension> &priors,
                              const DetectionScenes &scenes, const std::vector<int> &rows,
                              std::vector<detector::Detection> &detections) {
  const auto d = nn::models::DecodeYoloV3LevelDetections(
      raw, level, DimensionTensor(priors),
      nn::from_vector(std::vector<std::int64_t>(rows.begin(), rows.end())), scenes.height,
      scenes.width);
  for (const auto &p : nn::models::DetectionEvaluationRecords(d))
    detections.push_back({p.image, p.label, p.score, p.box});
}
template <typename Forward>
std::vector<detector::Detection>
DetectYoloV3(const Forward &forward, const std::vector<Dimension> &priors,
             const DetectionScenes &scenes, const std::vector<int> &rows) {
  nn::NoGradGuard guard;
  std::vector<detector::Detection> detections;
  for (std::size_t start = 0; start < rows.size(); start += 32) {
    const std::vector<int> chunk(rows.begin() + std::ptrdiff_t(start),
                                 rows.begin() +
                                     std::ptrdiff_t(std::min(rows.size(), start + 32)));
    const std::array<nn::Tensor, 3> raw = forward(detector::SceneImages(scenes, chunk));
    for (int level = 0; level < 3; ++level)
      DecodeYoloV3Level(raw[std::size_t(level)], level, priors, scenes, chunk, detections);
  }
  return detector::ClassNms(detections, .45f);
}

} // namespace yolo
