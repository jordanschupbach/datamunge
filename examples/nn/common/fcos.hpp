#pragma once
// Compatibility adapters: numerical implementation is canonical tensor code.
#include "common/retinanet.hpp"
#include "datamunge/nn/fcos_detection_models.hpp"
namespace fcos {
namespace nn = datamunge::nn;
using detector::Box;
using detector::DetectionScenes;
using nn::models::Fcos;
using nn::models::FcosConfig;
using nn::models::FcosIouLoss;
using nn::models::FcosLoss;
using nn::models::FcosOutputs;
using nn::models::FcosTargets;
struct FcosPoint {
  float y, x, stride, minimum, maximum;
};
inline nn::Tensor PointTensor(const std::vector<FcosPoint> &points) {
  std::vector<float> v;
  for (const auto &p : points) {
    v.insert(v.end(), {p.y, p.x, p.stride, p.minimum, p.maximum});
  }
  return nn::from_vector(v, {std::int64_t(points.size()), 5});
}
inline std::vector<FcosPoint> FcosPoints(const std::vector<int> &cells,
                                         float range_scale = 1.f) {
  const auto a = nn::models::FcosPoints(cells, range_scale).to_vector<float>();
  std::vector<FcosPoint> out;
  for (std::size_t i = 0; i < a.size(); i += 5)
    out.push_back({a[i], a[i + 1], a[i + 2], a[i + 3], a[i + 4]});
  return out;
}
inline float FcosCenterness(const std::array<float, 4> &d) {
  return nn::models::FcosCenterness(nn::from_vector(std::vector<float>(d.begin(), d.end())))
      .item<float>();
}
struct FcosTargetArrays {
  std::vector<float> classes, distances, positive, centerness;
};
inline FcosTargets MakeFcosTargets(const std::vector<FcosPoint> &points,
                                   const DetectionScenes &scenes, const std::vector<int> &rows,
                                   int classes, float center_radius = 1.5f) {
  nn::models::TensorDetectionScenes data;
  data.images = nn::zeros({std::int64_t(rows.size()), scenes.height, scenes.width, 1});
  std::vector<std::int64_t> selected;
  for (auto row : rows) {
    if (row < 0 || row >= scenes.size())
      throw std::invalid_argument("invalid FCOS row");
    data.boxes.push_back(nn::boxes_from_vector(scenes.boxes[row]));
    data.labels.push_back(nn::from_vector(
        std::vector<std::int64_t>(scenes.labels[row].begin(), scenes.labels[row].end())));
    selected.push_back(std::int64_t(selected.size()));
  }
  return nn::models::MakeFcosTargets(PointTensor(points), data, selected, classes,
                                     center_radius);
}
inline FcosTargetArrays MakeFcosTargetArrays(const std::vector<FcosPoint> &points,
                                             const DetectionScenes &scenes,
                                             const std::vector<int> &rows, int classes,
                                             float center_radius = 1.5f) {
  const auto t = MakeFcosTargets(points, scenes, rows, classes, center_radius);
  return {t.classes.to_vector<float>(), t.distances.to_vector<float>(),
          t.positive.to_vector<float>(), t.centerness.to_vector<float>()};
}
inline std::vector<detector::Detection> DecodeFcos(const FcosOutputs &net,
                                                   const std::vector<FcosPoint> &points,
                                                   const DetectionScenes &scenes,
                                                   const std::vector<int> &rows, int classes) {
  if (net.logits.shape()[2] != classes)
    throw std::invalid_argument("FCOS class mismatch");
  const auto found = nn::models::DecodeFcosDetections(
      net, PointTensor(points),
      nn::from_vector(std::vector<std::int64_t>(rows.begin(), rows.end())), scenes.height,
      scenes.width, false);
  std::vector<detector::Detection> out;
  for (const auto &d : nn::models::DetectionEvaluationRecords(found))
    out.push_back({d.image, d.label, d.score, d.box});
  return out;
}
inline std::vector<detector::Detection>
DetectFcos(nn::ParameterStore &p, nn::ParameterStore &state, const FcosConfig &cfg,
           const std::vector<FcosPoint> &points, const DetectionScenes &scenes,
           const std::vector<int> &rows, int classes) {
  nn::NoGradGuard guard;
  std::vector<detector::Detection> result;
  for (std::size_t start = 0; start < rows.size(); start += 32) {
    const std::vector<int> chunk(rows.begin() + std::ptrdiff_t(start),
                                 rows.begin() +
                                     std::ptrdiff_t(std::min(rows.size(), start + 32)));
    const auto net = Fcos(p, state, detector::SceneImages(scenes, chunk), cfg, false);
    const auto found = DecodeFcos(net, points, scenes, chunk, classes);
    result.insert(result.end(), found.begin(), found.end());
  }
  return detector::ClassNms(result, .6f, 100);
}

} // namespace fcos
