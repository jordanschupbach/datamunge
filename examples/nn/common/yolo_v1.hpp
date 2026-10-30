#pragma once
#include "common/detection_affine.hpp"
#include "common/resnet.hpp"
#include "datamunge/nn/yolo_v1_models.hpp"
#include <map>
namespace yolo {
namespace nn = datamunge::nn;
using detector::Box;
using detector::DetectionScenes;
using nn::models::YoloV1;
using nn::models::YoloV1Backbone;
using nn::models::YoloV1Config;
using nn::models::YoloV1Layers;
using nn::models::YoloV1LossOutputs;
using nn::models::YoloV1Outputs;
using nn::models::YoloV1PretrainingLogits;
struct YoloV1Targets {
  std::vector<float> object, boxes;
  std::vector<int> labels;
  int discarded = 0;
};
inline nn::models::TensorDetectionScenes YoloSceneMetadata(const DetectionScenes &scenes,
                                                           const std::vector<int> &rows,
                                                           bool pixels = false) {
  nn::models::TensorDetectionScenes data;
  const auto N = std::int64_t(rows.size());
  std::vector<nn::Tensor> images;
  for (auto row : rows) {
    if (row < 0 || row >= scenes.size())
      throw std::invalid_argument("invalid YOLO scene row");
    data.boxes.push_back(nn::boxes_from_vector(scenes.boxes[row]));
    data.labels.push_back(nn::from_vector(
        std::vector<std::int64_t>(scenes.labels[row].begin(), scenes.labels[row].end())));
    if (pixels) {
      const auto *start = scenes.image(row);
      images.push_back(
          nn::from_vector(std::vector<int>(start, start + scenes.height * scenes.width),
                          {scenes.height, scenes.width, 1})
              .to(nn::DType::Float32) /
          255.);
    }
  }
  data.images = pixels && !images.empty() ? nn::stack(images, 0)
                                          : nn::zeros({N, scenes.height, scenes.width, 1});
  return data;
}
inline YoloV1Targets MakeYoloV1Targets(const DetectionScenes &scenes,
                                       const std::vector<int> &rows, int grid = 7) {
  const auto data = YoloSceneMetadata(scenes, rows);
  std::vector<std::int64_t> ids(rows.size());
  std::iota(ids.begin(), ids.end(), 0);
  const auto t =
      nn::models::MakeYoloV1Targets(data, ids, std::numeric_limits<int>::max(), grid);
  return {t.object.to_vector<float>(), t.boxes.to_vector<float>(), t.labels.to_vector<int>(),
          int(t.discarded)};
}
inline YoloV1LossOutputs YoloV1Loss(const nn::Tensor &prediction, const YoloV1Targets &t,
                                    int classes, int grid = 7) {
  const auto N = prediction.shape()[0];
  return nn::models::YoloV1Loss(
      prediction,
      {nn::from_vector(t.object, {N, grid, grid}).to(prediction.dtype()),
       nn::from_vector(t.boxes, {N, grid, grid, 4}).to(prediction.dtype()),
       nn::from_vector(std::vector<std::int64_t>(t.labels.begin(), t.labels.end()),
                       {N, grid, grid}),
       t.discarded},
      classes, grid);
}
inline void DecodeYoloV1(const std::vector<float> &values, const std::vector<int> &rows,
                         const DetectionScenes &scenes, int classes, int grid,
                         std::vector<detector::Detection> &detections) {
  const auto found = nn::models::DecodeYoloV1Detections(
      nn::from_vector(values, {std::int64_t(rows.size()), grid, grid, classes + 10}),
      nn::from_vector(std::vector<std::int64_t>(rows.begin(), rows.end())), scenes.height,
      scenes.width, classes, false);
  for (const auto &d : nn::models::DetectionEvaluationRecords(found))
    detections.push_back({d.image, d.label, d.score, d.box});
}
/// Inference in chunks of 32 with `forward(images)` returning predictions; class-wise NMS .45.
template <typename Forward>
std::vector<detector::Detection>
DetectYoloV1(const Forward &forward, const DetectionScenes &scenes,
             const std::vector<int> &rows, int classes, int grid = 7) {
  nn::NoGradGuard guard;
  std::vector<detector::Detection> detections;
  for (std::size_t start = 0; start < rows.size(); start += 32) {
    const std::vector<int> chunk(rows.begin() + std::ptrdiff_t(start),
                                 rows.begin() +
                                     std::ptrdiff_t(std::min(rows.size(), start + 32)));
    const nn::Tensor prediction = forward(detector::SceneImages(scenes, chunk));
    DecodeYoloV1(prediction.to_vector<float>(), chunk, scenes, classes, grid, detections);
  }
  return detector::ClassNms(detections, .45f);
}

inline DetectionScenes YoloV1PretrainingCrops(const DetectionScenes &scenes, int begin,
                                              int end) {
  if (begin < 0 || end < begin || end > scenes.size() ||
      scenes.labels.size() != scenes.boxes.size() || scenes.height < 1 || scenes.width < 1 ||
      scenes.pixels.size() !=
          std::size_t(scenes.size()) * std::size_t(scenes.height) * std::size_t(scenes.width))
    throw std::invalid_argument("invalid YOLO pretraining split/source");
  std::vector<int> rows(std::size_t(end - begin));
  std::iota(rows.begin(), rows.end(), begin);
  const auto data = YoloSceneMetadata(scenes, rows, true);
  std::vector<std::int64_t> ids(rows.size());
  std::iota(ids.begin(), ids.end(), 0);
  const auto t = nn::models::YoloV1PretrainingCrops(data, ids);
  DetectionScenes out;
  out.height = out.width = 56;
  const auto bytes = nn::floor(t.images.to(nn::DType::Float64) * 255. + .5).to_vector<int>();
  for (int v : bytes)
    out.pixels.push_back(std::uint8_t(v));
  for (std::size_t i = 0; i < t.boxes.size(); ++i) {
    out.boxes.push_back(nn::boxes_to_vector(t.boxes[i]));
    out.labels.push_back(t.labels[i].to_vector<int>());
  }
  return out;
}
/// Images [B, 56*56] in [-1, 1] and int32 labels [B].
inline std::pair<nn::Tensor, nn::Tensor> YoloV1PretrainingBatch(const DetectionScenes &crops,
                                                                const std::vector<int> &rows) {
  check::Require(crops.height == 56 && crops.width == 56 &&
                     crops.labels.size() == crops.boxes.size() &&
                     crops.pixels.size() == std::size_t(crops.size()) * 56 * 56,
                 "valid pretraining crop dimensions");
  std::vector<std::int32_t> labels;
  for (int row : rows) {
    check::Require(
        row >= 0 && row < crops.size() && crops.labels[std::size_t(row)].size() == 1 &&
            crops.labels[std::size_t(row)][0] >= 0 && crops.labels[std::size_t(row)][0] < 10,
        "valid pretraining row/class");
    labels.push_back(crops.labels[std::size_t(row)][0]);
  }
  return {
      nn::reshape(detector::SceneImages(crops, rows), {std::int64_t(rows.size()), 56 * 56}),
      nn::from_vector(labels)};
}

inline bool YoloV1PretrainedName(const std::string &name) {
  for (int layer = 1; layer <= 20; ++layer)
    if (name.rfind("conv" + std::to_string(layer) + "/", 0) == 0)
      return true;
  return false;
}

/// Classifier of the pretraining stage (tflow ml::Classifier): Adam(.001),
/// mean cross-entropy + .5 * 1e-4 * sum of squared kernels.
struct YoloV1Classifier {
  YoloV1Config cfg;
  nn::ParameterStore parameters{17};
  nn::ParameterStore state{17};
  explicit YoloV1Classifier(const YoloV1Config &config) : cfg(config) {
    Init(parameters, state, config);
  }
  static nn::OptimizerConfig AdamConfig() {
    nn::OptimizerConfig o;
    o.kind = nn::OptimizerKind::Adam;
    return o;
  }
  static const nn::ParameterStore &Init(nn::ParameterStore &p, nn::ParameterStore &s,
                                        const YoloV1Config &c) {
    nn::NoGradGuard guard;
    auto frozen = c;
    frozen.update_stats = false;
    YoloV1PretrainingLogits(p, s, nn::zeros({1, 56 * 56}), frozen, true);
    return p;
  }
  /// Mean digit NLL of pretraining logits; the host example minimizes Loss + .5e-4 * WeightPenalty
  /// with Adam (AdamConfig, learning rate 1e-3) and owns that optimizer step.
  static nn::Tensor Loss(const nn::Tensor &logits, const nn::Tensor &y) {
    return nn::mean(-nn::sum(nn::one_hot(y, 10) * nn::log_softmax(logits, -1), {1}));
  }
  /// Sum of squares of every matrix/kernel parameter (rank >= 2).
  static nn::Tensor WeightPenalty(const nn::ParameterStore &p) {
    nn::Tensor penalty;
    for (const auto &t : p.parameters())
      if (t.dim() >= 2)
        penalty =
            penalty.defined() ? penalty + nn::sum(nn::square(t)) : nn::sum(nn::square(t));
    return penalty;
  }
  /// {mean NLL, accuracy} of the inference path.
  std::pair<double, double> Evaluate(const nn::Tensor &x, const nn::Tensor &y) {
    nn::NoGradGuard guard;
    const auto logits = YoloV1PretrainingLogits(parameters, state, x, cfg, false);
    const double loss =
        nn::mean(-nn::sum(nn::one_hot(y, 10) * nn::log_softmax(logits, -1), {1})).item();
    const double accuracy =
        nn::mean(nn::equal(nn::argmax(logits, 1), y.to(nn::DType::Int64)).to(logits.dtype()))
            .item();
    return {loss, accuracy};
  }
};

inline std::pair<double, double> EvaluateYoloV1Pretraining(YoloV1Classifier &model,
                                                           const DetectionScenes &crops) {
  double loss = 0, accuracy = 0;
  for (int start = 0; start < crops.size(); start += 64) {
    std::vector<int> rows(std::size_t(std::min(64, crops.size() - start)));
    std::iota(rows.begin(), rows.end(), start);
    const auto [x, y] = YoloV1PretrainingBatch(crops, rows);
    const auto [l, a] = model.Evaluate(x, y);
    loss += l * double(rows.size());
    accuracy += a * double(rows.size());
  }
  return {loss / crops.size(), accuracy / crops.size()};
}

struct YoloV1PretrainedWeights {
  std::map<std::string, nn::Tensor> parameters, state;
};
inline YoloV1PretrainedWeights CaptureYoloV1Pretrained(const YoloV1Classifier &model) {
  YoloV1PretrainedWeights weights;
  for (const auto &name : model.parameters.names())
    if (YoloV1PretrainedName(name))
      weights.parameters.emplace(name, model.parameters.at(name).detach().clone());
  for (const auto &name : model.state.names())
    if (YoloV1PretrainedName(name))
      weights.state.emplace(name, model.state.at(name).detach().clone());
  return weights;
}

/// Copies the prefix into a fresh 24-convolution detector (seed 17) and saves
/// it as `<prefix>_params.npz` / `<prefix>_state.npz`, verifying every value.
inline void WriteYoloV1Bootstrap(const YoloV1PretrainedWeights &weights,
                                 const YoloV1Config &cfg,
                                 const std::filesystem::path &prefix) {
  check::Require(cfg.compact && cfg.width_divisor == 8 && cfg.classes == 10,
                 "compact digit bootstrap configuration");
  check::Require(weights.parameters.size() == std::size_t(cfg.batch_norm ? 60 : 40) &&
                     weights.state.size() == std::size_t(cfg.batch_norm ? 40 : 0),
                 "complete first-20 prefix weights and BN state");
  nn::ParameterStore p(17), state(17);
  {
    nn::NoGradGuard guard;
    YoloV1(p, &state, nn::zeros({1, 56, 56, 1}), cfg, false);
  }
  auto copy = [&](const std::map<std::string, nn::Tensor> &source,
                  nn::ParameterStore &destination) {
    for (const auto &[name, value] : source) {
      check::Require(YoloV1PretrainedName(name) && destination.contains(name) &&
                         destination.at(name).shape() == value.shape(),
                     "pretrained variable name/type/shape matches");
      check::Require(nn::all(nn::less(nn::abs(value), std::numeric_limits<double>::infinity()))
                         .item<bool>(),
                     "finite pretrained weight");
      destination.set(name, value);
    }
    for (const auto &[name, value] : source) {
      check::Require(nn::all(nn::equal(destination.at(name), value)).item<bool>(),
                     "exact pretrained-prefix transfer");
    }
  };
  copy(weights.parameters, p);
  copy(weights.state, state);
  std::filesystem::create_directories(prefix.parent_path());
  detector::SaveCheckpoint(p, state, prefix.string());
}

} // namespace yolo
