#pragma once
// Compatibility interop only; canonical tensor implementation lives in nn.
#include "common/detection_scenes.hpp"
#include "common/resnet.hpp"
#include "datamunge/nn/center_detection_models.hpp"
namespace centernet {
namespace nn = datamunge::nn;
using detector::Box;
using detector::DetectionScenes;
using nn::models::CenterHead;
using nn::models::CenterHeatmapFocal;
using nn::models::CenterLoss;
using nn::models::CenterNet;
using nn::models::CenterNetConfig;
using nn::models::CenterNetOutputs;
using nn::models::CenterPeaks;
inline double CenterGaussianRadius(double h, double w, double overlap = .7) {
  return nn::models::CenterGaussianRadius(nn::full({}, h, nn::DType::Float64),
                                          nn::full({}, w, nn::DType::Float64), overlap)
      .item();
}
struct Grid {
  std::int64_t n = 0, h = 0, w = 0, c = 0;
  std::vector<float> v;
  Grid() = default;
  Grid(std::int64_t n_, std::int64_t h_, std::int64_t w_, std::int64_t c_)
      : n(n_), h(h_), w(w_), c(c_), v(std::size_t(n_ * h_ * w_ * c_), 0.f) {
  }
  float &at(std::int64_t b, std::int64_t y, std::int64_t x, std::int64_t k) {
    return v[std::size_t(((b * h + y) * w + x) * c + k)];
  }
  [[nodiscard]] nn::Tensor tensor() const {
    return nn::from_vector(v, {n, h, w, c});
  }
};

inline void DrawCenterGaussian(Grid &heatmap, int batch, int row, int col, int label,
                               int radius) {
  if (batch < 0 || batch >= heatmap.n)
    throw std::invalid_argument("invalid Gaussian batch");
  const auto h = nn::models::CenterGaussianHeatmap(
      heatmap.h, heatmap.w, heatmap.c,
      nn::from_vector(std::vector<std::int64_t>{row, col}, {1, 2}),
      nn::from_vector(std::vector<std::int64_t>{label}),
      nn::from_vector(std::vector<std::int64_t>{radius}));
  auto all = heatmap.tensor();
  const auto index = nn::from_vector(std::vector<std::int64_t>{batch});
  const auto slice =
      nn::maximum(nn::gather(all, index), h.reshape({1, heatmap.h, heatmap.w, heatmap.c}));
  all = nn::scatter_nd(index.reshape({1, 1}), slice, all.shape()) +
        all * (1. - nn::one_hot(index, heatmap.n, 1, 0).reshape({heatmap.n, 1, 1, 1}));
  heatmap.v = all.to_vector<float>();
}
struct CenterTargets {
  Grid heatmap, size, offset, mask;
  int collisions = 0;
};
inline nn::Tensor CenterLoss(const CenterNetOutputs &net, const CenterTargets &t) {
  return nn::models::CenterLoss(net.heatmap, net.size, net.offset, t.heatmap.tensor(),
                                t.size.tensor(), t.offset.tensor(), t.mask.tensor());
}
inline CenterTargets MakeCenterTargets(const DetectionScenes &scenes,
                                       const std::vector<int> &rows, int classes) {
  nn::models::TensorDetectionScenes data;
  data.images = nn::zeros({std::int64_t(rows.size()), scenes.height, scenes.width, 1});
  std::vector<std::int64_t> selected;
  for (auto row : rows) {
    if (row < 0 || row >= scenes.size())
      throw std::invalid_argument("invalid CenterNet row");
    data.boxes.push_back(nn::boxes_from_vector(scenes.boxes[row]));
    data.labels.push_back(nn::from_vector(
        std::vector<std::int64_t>(scenes.labels[row].begin(), scenes.labels[row].end())));
    selected.push_back(std::int64_t(selected.size()));
  }
  const auto t = nn::models::MakeCenterTargets(data, selected, classes);
  const auto make = [](const nn::Tensor &x) {
    Grid g(x.shape()[0], x.shape()[1], x.shape()[2], x.shape()[3]);
    g.v = x.to_vector<float>();
    return g;
  };
  return {make(t.heatmap), make(t.size), make(t.offset), make(t.mask), int(t.collisions)};
}
inline std::vector<detector::Detection>
DecodeCenter(const nn::Tensor &peaks, const nn::Tensor &size, const nn::Tensor &offset,
             const DetectionScenes &scenes, const std::vector<int> &rows) {
  const auto found = nn::models::DecodeCenterDetections(
      peaks, size, offset,
      nn::from_vector(std::vector<std::int64_t>(rows.begin(), rows.end())), scenes.height,
      scenes.width);
  std::vector<detector::Detection> out;
  for (const auto &d : nn::models::DetectionEvaluationRecords(found))
    out.push_back({d.image, d.label, d.score, d.box});
  return out;
}
inline std::vector<detector::Detection>
DetectCenter(nn::ParameterStore &p, nn::ParameterStore &state, const CenterNetConfig &cfg,
             const DetectionScenes &scenes, const std::vector<int> &rows) {
  nn::NoGradGuard guard;
  std::vector<detector::Detection> detections;
  for (std::size_t start = 0; start < rows.size(); start += 32) {
    const std::vector<int> chunk(rows.begin() + std::ptrdiff_t(start),
                                 rows.begin() +
                                     std::ptrdiff_t(std::min(rows.size(), start + 32)));
    const auto net = CenterNet(p, state, detector::SceneImages(scenes, chunk), cfg, false);
    const auto found = DecodeCenter(net.peaks, net.size, net.offset, scenes, chunk);
    detections.insert(detections.end(), found.begin(), found.end());
  }
  return detections;
}

} // namespace centernet
