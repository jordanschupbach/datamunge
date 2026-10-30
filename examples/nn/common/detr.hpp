#pragma once
#include "common/detection_scenes.hpp"
#include "common/hungarian.hpp"
#include "common/resnet.hpp"
#include "datamunge/nn/detr_models.hpp"
namespace detr {
namespace nn = datamunge::nn;
using detector::Box;
using detector::DetectionScenes;
using nn::models::Detr;
using nn::models::DetrAttention;
using nn::models::DetrConfig;
using nn::models::DetrLinear;
using nn::models::DetrOutputs;
using nn::models::DetrPositions;
using nn::models::GeneralizedIou;
inline Box DetrCorners(const float *b) {
  const auto p = nn::unstack(nn::from_vector(std::vector<float>(b, b + 4)), 0);
  const auto t =
      nn::stack({p[0] - p[2] * .5, p[1] - p[3] * .5, p[0] + p[2] * .5, p[1] + p[3] * .5}, 0)
          .to_vector<float>();
  return {t[0], t[1], t[2], t[3]};
}
inline double GeneralizedIou(const Box &a, const Box &b) {
  const auto centre = [](const Box &q) {
    const auto t = nn::from_vector(std::vector<double>(q.begin(), q.end()));
    const auto p = nn::unstack(t, 0);
    return nn::stack({(p[0] + p[2]) * .5, (p[1] + p[3]) * .5, p[2] - p[0], p[3] - p[1]}, 0);
  };
  return nn::models::GeneralizedIou(centre(a), centre(b)).item();
}
struct DetrTargets {
  std::vector<float> labels, boxes, positive;
  std::int64_t batch = 0, queries = 0, classes = 0;
};
inline nn::models::TensorDetectionScenes SceneMetadata(const DetectionScenes &scenes,
                                                       const std::vector<int> &rows) {
  nn::models::TensorDetectionScenes data;
  data.images = nn::zeros({std::int64_t(rows.size()), scenes.height, scenes.width, 1});
  for (auto row : rows) {
    if (row < 0 || row >= scenes.size())
      throw std::invalid_argument("invalid DETR scene row");
    data.boxes.push_back(nn::boxes_from_vector(scenes.boxes[row]));
    data.labels.push_back(nn::from_vector(
        std::vector<std::int64_t>(scenes.labels[row].begin(), scenes.labels[row].end())));
  }
  return data;
}
inline nn::Tensor DetrLayerLoss(const nn::Tensor &logits, const nn::Tensor &boxes,
                                const DetrTargets &t) {
  return nn::models::DetrLayerLoss(
      logits, boxes,
      {nn::from_vector(t.labels, {t.batch, t.queries, t.classes + 1}).to(logits.dtype()),
       nn::from_vector(t.boxes, {t.batch, t.queries, 4}).to(logits.dtype()),
       nn::from_vector(t.positive, {t.batch, t.queries}).to(logits.dtype())});
}
inline DetrTargets MatchDetrTargets(const std::vector<float> &logits,
                                    const std::vector<float> &boxes, std::int64_t queries,
                                    const DetectionScenes &scenes,
                                    const std::vector<int> &rows, int classes) {
  const auto N = std::int64_t(rows.size());
  const auto data = SceneMetadata(scenes, rows);
  std::vector<std::int64_t> ids(rows.size());
  std::iota(ids.begin(), ids.end(), 0);
  const auto t = nn::models::MatchDetrTargets(
      nn::from_vector(logits, {N, queries, classes + 1}),
      nn::from_vector(boxes, {N, queries, 4}), data, ids, classes);
  return {t.labels.to_vector<float>(),
          t.boxes.to_vector<float>(),
          t.positive.to_vector<float>(),
          N,
          queries,
          classes};
}
inline nn::Tensor DetrLoss(const DetrOutputs &net, const DetectionScenes &scenes,
                           const std::vector<int> &rows, int classes) {
  const auto data = SceneMetadata(scenes, rows);
  std::vector<std::int64_t> ids(rows.size());
  std::iota(ids.begin(), ids.end(), 0);
  return nn::models::DetrLoss(net, data, ids, classes);
}
inline std::vector<detector::Detection> DecodeDetr(const nn::Tensor &logits,
                                                   const nn::Tensor &boxes,
                                                   const DetectionScenes &scenes,
                                                   const std::vector<int> &rows, int classes) {
  if (logits.shape()[2] != classes + 1)
    throw std::invalid_argument("DETR class count mismatch");
  const auto found = nn::models::DecodeDetrDetections(
      logits, boxes, nn::from_vector(std::vector<std::int64_t>(rows.begin(), rows.end())),
      scenes.height, scenes.width);
  std::vector<detector::Detection> out;
  for (const auto &d : nn::models::DetectionEvaluationRecords(found))
    out.push_back({d.image, d.label, d.score, d.box});
  return out;
}
inline std::vector<detector::Detection>
DetectDetr(nn::ParameterStore &p, nn::ParameterStore &state, const DetrConfig &cfg,
           const DetectionScenes &scenes, const std::vector<int> &rows, int classes) {
  nn::NoGradGuard guard;
  std::vector<detector::Detection> result;
  for (std::size_t start = 0; start < rows.size(); start += 32) {
    const std::vector<int> chunk(rows.begin() + std::ptrdiff_t(start),
                                 rows.begin() +
                                     std::ptrdiff_t(std::min(rows.size(), start + 32)));
    const auto net = Detr(p, state, detector::SceneImages(scenes, chunk), cfg, false);
    const auto found = DecodeDetr(net.logits.back(), net.boxes.back(), scenes, chunk, classes);
    result.insert(result.end(), found.begin(), found.end());
  }
  return result;
}

} // namespace detr
