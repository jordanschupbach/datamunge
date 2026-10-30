#pragma once
#include "common/detection_scenes.hpp"
#include "common/resnet.hpp"
#include "common/word_tree.hpp"
#include "datamunge/nn/yolo_v2_models.hpp"
namespace yolo {
namespace nn = datamunge::nn;
using detector::Box;
using detector::DetectionScenes;
using Dimension = std::array<float, 2>;
using nn::models::DarknetConv;
using nn::models::YoloV2;
using nn::models::YoloV2Config;
using nn::models::YoloV2Outputs;
using nn::models::YoloV2Probabilities;
inline nn::Tensor DimensionTensor(const std::vector<Dimension> &x) {
  std::vector<float> v;
  for (const auto &d : x)
    v.insert(v.end(), d.begin(), d.end());
  return nn::from_vector(v, {std::int64_t(x.size()), 2});
}
inline float DimensionIoU(const Dimension &a, const Dimension &b) {
  return nn::models::YoloDimensionIoU(DimensionTensor({a}), DimensionTensor({b}))
      .item<float>();
}
inline std::vector<Dimension> ClusterDimensions(const std::vector<Dimension> &x, int count) {
  const auto c =
      nn::models::ClusterYoloDimensions(DimensionTensor(x), count).to_vector<float>();
  std::vector<Dimension> out;
  for (std::size_t i = 0; i < c.size(); i += 2)
    out.push_back({c[i], c[i + 1]});
  return out;
}
inline Box DecodeYoloV2(const float *raw, int row, int col, int cells, const Dimension &prior,
                        int height, int width) {
  const auto d = nn::from_vector(std::vector<float>(raw, raw + 4));
  const auto p = nn::from_vector(std::vector<float>(prior.begin(), prior.end()));
  const auto q = nn::unstack(d, 0), a = nn::unstack(p, 0);
  const auto y = (double(row) + nn::sigmoid(q[0])) * (double(height) / cells),
             x = (double(col) + nn::sigmoid(q[1])) * (double(width) / cells),
             h = a[0] * nn::exp(nn::clip(q[2], -10., 10.)) * (double(height) / cells),
             w = a[1] * nn::exp(nn::clip(q[3], -10., 10.)) * (double(width) / cells);
  const auto b =
      nn::stack({y - h * .5, x - w * .5, y + h * .5, x + w * .5}, 0).to_vector<float>();
  return {b[0], b[1], b[2], b[3]};
}
struct YoloV2Targets {
  std::vector<float> geometry, positive, confidence, weights;
  std::vector<int> labels;
  int collisions = 0;
};
inline YoloV2Targets MakeYoloV2Targets(const std::vector<float> &raw, const nn::Shape &shape,
                                       const std::vector<Dimension> &priors,
                                       const DetectionScenes &scenes,
                                       const std::vector<int> &rows, bool coarse) {
  nn::models::TensorDetectionScenes data;
  data.images = nn::zeros({std::int64_t(rows.size()), scenes.height, scenes.width, 1});
  for (auto row : rows) {
    if (row < 0 || row >= scenes.size())
      throw std::invalid_argument("invalid YOLO v2 row");
    data.boxes.push_back(nn::boxes_from_vector(scenes.boxes[row]));
    data.labels.push_back(nn::from_vector(
        std::vector<std::int64_t>(scenes.labels[row].begin(), scenes.labels[row].end())));
  }
  std::vector<std::int64_t> ids(rows.size());
  std::iota(ids.begin(), ids.end(), 0);
  const auto t = nn::models::MakeYoloV2Targets(nn::from_vector(raw, shape),
                                               DimensionTensor(priors), data, ids, coarse);
  return {t.geometry.to_vector<float>(),   t.positive.to_vector<float>(),
          t.confidence.to_vector<float>(), t.weights.to_vector<float>(),
          t.labels.to_vector<int>(),       int(t.collisions)};
}
struct YoloV2Weights {
  double detection = 1, classification = 0;
  std::vector<int> classification_labels;
};
inline nn::Tensor YoloV2Loss(const nn::Tensor &raw, const YoloV2Targets &t, int classes,
                             const YoloV2Weights &w,
                             const wordtree::WordTree *tree = nullptr) {
  const auto s = raw.shape();
  const nn::Shape mask = {s[0], s[1], s[2], s[3]};
  nn::models::YoloV2Targets target{
      nn::from_vector(t.geometry, {s[0], s[1], s[2], s[3], 4}).to(raw.dtype()),
      nn::from_vector(t.positive, mask).to(raw.dtype()),
      nn::from_vector(t.confidence, mask).to(raw.dtype()),
      nn::from_vector(t.weights, mask).to(raw.dtype()),
      nn::from_vector(std::vector<std::int64_t>(t.labels.begin(), t.labels.end()), mask),
      t.collisions};
  nn::models::YoloV2Weights weights;
  weights.detection = w.detection;
  weights.classification = w.classification;
  weights.classification_labels = nn::from_vector(std::vector<std::int64_t>(
      w.classification_labels.begin(), w.classification_labels.end()));
  return nn::models::YoloV2Loss(raw, target, classes, weights, tree);
}
inline void DecodeYoloV2Detections(const std::vector<float> &raw, const nn::Shape &shape,
                                   const std::vector<float> &probabilities, int classes,
                                   const std::vector<Dimension> &priors,
                                   const DetectionScenes &scenes, const std::vector<int> &rows,
                                   bool hierarchy,
                                   std::vector<detector::Detection> &detections) {
  const auto found = nn::models::DecodeYoloV2TensorDetections(
      nn::from_vector(raw, shape), DimensionTensor(priors),
      nn::from_vector(probabilities, {shape[0], shape[1], shape[2], shape[3], classes}),
      nn::from_vector(std::vector<std::int64_t>(rows.begin(), rows.end())), scenes.height,
      scenes.width, hierarchy, false);
  for (const auto &d : nn::models::DetectionEvaluationRecords(found))
    if (d.label < 10)
      detections.push_back({d.image, d.label, d.score, d.box});
}
/// Inference in chunks of 32 with `forward(images)` returning raw predictions; class-wise NMS .45.
template <typename Forward>
std::vector<detector::Detection>
DetectYoloV2(const Forward &forward, const std::vector<Dimension> &priors,
             const DetectionScenes &scenes, const std::vector<int> &rows, int classes,
             const wordtree::WordTree *tree) {
  nn::NoGradGuard guard;
  std::vector<detector::Detection> detections;
  for (std::size_t start = 0; start < rows.size(); start += 32) {
    const std::vector<int> chunk(rows.begin() + std::ptrdiff_t(start),
                                 rows.begin() +
                                     std::ptrdiff_t(std::min(rows.size(), start + 32)));
    const nn::Tensor raw = forward(detector::SceneImages(scenes, chunk));
    DecodeYoloV2Detections(raw.to_vector<float>(), raw.shape(),
                           YoloV2Probabilities(raw, classes, tree).to_vector<float>(), classes,
                           priors, scenes, chunk, tree != nullptr, detections);
  }
  return detector::ClassNms(detections, .45f);
}

} // namespace yolo
