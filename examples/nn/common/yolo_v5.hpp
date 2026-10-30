#pragma once
#include "common/yolo_v3.hpp"
#include "datamunge/nn/yolo_v5_models.hpp"
namespace yolo {
using nn::models::BoxQuality;
using nn::models::CompleteBoxQuality;
using nn::models::CspBlock;
using nn::models::CspContext;
using nn::models::CspConv;
using nn::models::CspOutputs;
using nn::models::SpatialPyramidFast;
using nn::models::YoloV5;
using nn::models::YoloV5Config;
using nn::models::YoloV5Outputs;
inline nn::Tensor DecodeYoloV5Boxes(const nn::Tensor &raw,
                                    const std::vector<Dimension> &priors, int level) {
  return nn::models::DecodeYoloV5Boxes(raw, DimensionTensor(priors), level);
}
struct YoloV5LevelTargets {
  std::vector<float> geometry, classes, positive, quality;
  int class_count = 0;
};
struct YoloV5Targets {
  std::array<YoloV5LevelTargets, 3> levels;
  int collisions = 0;
};
inline nn::Tensor YoloV5Loss(const std::array<nn::Tensor, 3> &raw, const YoloV5Targets &t,
                             int classes, const std::vector<Dimension> &priors) {
  nn::models::YoloV5Targets target;
  for (int l = 0; l < 3; ++l) {
    const auto &s = raw[l].shape();
    const auto &g = t.levels[l];
    const nn::Shape grid = {s[0], s[1], s[2], 3};
    target.levels[l] = {
        nn::from_vector(g.geometry, {s[0], s[1], s[2], 3, 4}).to(raw[l].dtype()),
        nn::from_vector(g.classes, {s[0], s[1], s[2], 3, classes}).to(raw[l].dtype()),
        nn::from_vector(g.positive, grid).to(raw[l].dtype()),
        g.quality.empty() ? nn::full(grid, -1., raw[l].dtype())
                          : nn::from_vector(g.quality, grid).to(raw[l].dtype())};
  }
  return nn::models::YoloV5Loss(raw, target, classes, DimensionTensor(priors));
}
inline YoloV5Targets MakeYoloV5Targets(const std::array<nn::Shape, 3> &shapes,
                                       const std::vector<Dimension> &priors,
                                       const DetectionScenes &scenes,
                                       const std::vector<int> &rows) {
  nn::models::TensorDetectionScenes data;
  data.images = nn::zeros({std::int64_t(rows.size()), scenes.height, scenes.width, 1});
  for (auto row : rows) {
    if (row < 0 || row >= scenes.size())
      throw std::invalid_argument("invalid YOLO v5 row");
    data.boxes.push_back(nn::boxes_from_vector(scenes.boxes[row]));
    data.labels.push_back(nn::from_vector(
        std::vector<std::int64_t>(scenes.labels[row].begin(), scenes.labels[row].end())));
  }
  std::vector<std::int64_t> ids(rows.size());
  std::iota(ids.begin(), ids.end(), 0);
  std::array<nn::Tensor, 3> raw;
  for (int l = 0; l < 3; ++l)
    raw[l] = nn::zeros(shapes[l]);
  const auto t = nn::models::MakeYoloV5Targets(raw, DimensionTensor(priors), data, ids);
  YoloV5Targets out;
  out.collisions = int(t.collisions);
  for (int l = 0; l < 3; ++l) {
    const auto &g = t.levels[l];
    out.levels[l] = {g.geometry.to_vector<float>(), g.classes.to_vector<float>(),
                     g.positive.to_vector<float>(), g.quality.to_vector<float>(),
                     int(g.classes.shape().back())};
  }
  return out;
}
inline Box DecodeYoloV5(const float *raw, int row, int col, int cells, const Dimension &prior,
                        int stride, int height, int width) {
  const auto v = nn::from_vector(std::vector<float>(raw, raw + 4)), s = nn::sigmoid(v) * 2.;
  const auto centre =
      nn::stack({(nn::slice_axis(s,0,0,1).reshape({}) + nn::full({}, double(row) - .5)) / double(cells) * double(height),
                 (nn::slice_axis(s,0,1,2).reshape({}) + nn::full({}, double(col) - .5)) / double(cells) * double(width),
                 nn::square(nn::slice_axis(s, 0, 2, 3)).reshape({}) * double(prior[0]) /
                     double(stride * cells) * double(height),
                 nn::square(nn::slice_axis(s, 0, 3, 4)).reshape({}) * double(prior[1]) /
                     double(stride * cells) * double(width)},
                0);
  return nn::boxes_to_vector(nn::boxes_from_center(centre.reshape({1, 4})))[0];
}
inline void DecodeYoloV5Level(const nn::Tensor &raw, int level,
                              const std::vector<Dimension> &priors,
                              const DetectionScenes &scenes, const std::vector<int> &rows,
                              std::vector<detector::Detection> &out) {
  const auto d = nn::models::DecodeYoloV5LevelDetections(
      raw, level, DimensionTensor(priors),
      nn::from_vector(std::vector<std::int64_t>(rows.begin(), rows.end())), scenes.height,
      scenes.width);
  for (const auto &p : nn::models::DetectionEvaluationRecords(d))
    out.push_back({p.image, p.label, p.score, p.box});
}
template <typename Forward>
std::vector<detector::Detection>
DetectYoloV5(const Forward &forward, const std::vector<Dimension> &priors,
             const DetectionScenes &scenes, const std::vector<int> &rows) {
  nn::NoGradGuard guard;
  std::vector<detector::Detection> result;
  for (std::size_t start = 0; start < rows.size(); start += 32) {
    const std::vector<int> chunk(rows.begin() + std::ptrdiff_t(start),
                                 rows.begin() +
                                     std::ptrdiff_t(std::min(rows.size(), start + 32)));
    const std::array<nn::Tensor, 3> raw = forward(detector::SceneImages(scenes, chunk));
    for (int level = 0; level < 3; ++level)
      DecodeYoloV5Level(raw[std::size_t(level)], level, priors, scenes, chunk, result);
  }
  return detector::ClassNms(result, .45f);
}

} // namespace yolo
