// YOLO v1: the paper's 24-convolution model with a dense head (port of
// tflow's common/nn/yolo_v1.h), its grid assignment, IoU responsibility and
// sum-of-squares loss and decoding (common/ml/yolo_v1.h), and the optional
// first-20-convolution digit-crop classifier pretraining
// (common/ml/yolo_v1_pretrain.h).
// https://arxiv.org/abs/1506.02640
//
// Model: hidden activations are leaky ReLU (slope .1), the output is linear,
// and dropout precedes the output layer during training. This follows the
// paper's dense head, rather than Darknet's later locally connected variant.
// The full paper model has no BN; the compact adaptation can use BN (scale,
// eps 1e-5, momentum .9). Each cell packs [class scores, box0(dy, dx,
// sqrt(h), sqrt(w), confidence), box1(...)]. Centres are offsets within their
// cell and h/w are fractions of the image size. compact=true is for grayscale
// 56-pixel inputs (width_divisor 8, dense 256): it keeps all 24 convolutions,
// but removes the first and 22nd convolution strides and the fourth max pool
// to retain a 7x7 map.
//
// Loss: one target per cell (keep the larger box when centres collide, report
// the number discarded). Both predictors share a cell's class scores. The
// larger current IoU selects the responsible predictor (ties choose index 0).
// Confidence targets are detached IoUs. The other predictor receives the
// no-object confidence penalty. Square-root sizes are direct outputs;
// decoding squares them. Weights 5 (coordinates) and .5 (no-object).
//
// Pretraining: square, aspect-preserving digit crops (centre 28x28 of a 56x56
// canvas) from the caller's fitting scenes; separate validation crops select
// the lowest-NLL checkpoint. Training augments scale/translation/exposure. A
// global average pool and ten-class head train the convolutional prefix; the
// prefix weights and BN state are copied by name into a fresh detector, whose
// last four convolutions and dense head keep their initialization. The
// bootstrap is a <prefix>_params.npz / <prefix>_state.npz pair usable as
// warm_start. This digit-crop pretraining is an adaptation, not ImageNet.
#pragma once

#include <map>

#include "common/detection_affine.hpp"
#include "common/resnet.hpp"

namespace yolo {

namespace nn = datamunge::nn;
using detector::Box;
using detector::DetectionScenes;

struct YoloV1Config {
  int    classes = 20, width_divisor = 1, dense_width = 4096;
  bool   compact = false, batch_norm = false;
  double keep_probability = .5, momentum = .9;
  bool   update_stats = true;  // BN moving averages in training mode
};
struct YoloV1Outputs {
  nn::Tensor              features, prediction;
  std::vector<nn::Tensor> convolutions;
};

inline const std::vector<std::pair<int, int>>& YoloV1Layers() {
  static const std::vector<std::pair<int, int>> layers = {
      {64, 7},   {192, 3}, {128, 1},  {256, 3}, {256, 1},  {512, 3},  {256, 1},  {512, 3},
      {256, 1},  {512, 3}, {256, 1},  {512, 3}, {256, 1},  {512, 3},  {512, 1},  {1024, 3},
      {512, 1},  {1024, 3}, {512, 1}, {1024, 3}, {1024, 3}, {1024, 3}, {1024, 3}, {1024, 3}};
  return layers;
}

/// A prefix of the convolutional stack, with unchanged parameter names.
inline YoloV1Outputs YoloV1Backbone(nn::ParameterStore& p, nn::ParameterStore* state, const nn::Tensor& image,
                                    const YoloV1Config& cfg = {}, bool training = false, int convolutions = 24) {
  if (cfg.classes < 1 || cfg.width_divisor < 1 || 64 % cfg.width_divisor || cfg.dense_width < 1 || (cfg.batch_norm && !state) ||
      !(cfg.keep_probability > 0 && cfg.keep_probability <= 1) || !(cfg.momentum >= 0 && cfg.momentum < 1) || convolutions < 1 ||
      convolutions > 24)
    throw std::invalid_argument("invalid YOLO v1 configuration");
  const auto&            layers = YoloV1Layers();
  YoloV1Outputs          result;
  nn::Tensor             h = image;
  resnet::ConvNormConfig norm;
  norm.training     = training;
  norm.scale        = true;
  norm.epsilon      = 1e-5;
  norm.momentum     = cfg.momentum;
  norm.update_stats = cfg.update_stats;
  for (std::size_t i = 0; i < std::size_t(convolutions); ++i) {
    const int  out = layers[i].first / cfg.width_divisor, kernel = layers[i].second;
    const int  stride = !cfg.compact && (i == 0 || i == 21) ? 2 : 1;
    const auto name   = "conv" + std::to_string(i + 1);
    if (cfg.batch_norm) {
      h = resnet::ResNetConv(p, *state, name, h, out, kernel, norm, stride, false);
    } else {
      nn::ConvLayerOptions conv;
      conv.init    = nn::init::he_uniform();
      conv.padding = nn::Padding::Valid;
      conv.stride  = stride;
      const std::int64_t pad = kernel / 2;
      h = nn::conv2d(p, name, pad ? nn::pad(h, {{0, 0}, {pad, pad}, {pad, pad}, {0, 0}}) : h, out, kernel, conv);
    }
    h = nn::leaky_relu(h, .1);
    result.convolutions.push_back(h);
    if (i == 0 || i == 1 || i == 5 || (i == 15 && !cfg.compact)) h = nn::max_pool2d(h, 2, 2, nn::Padding::Valid);
  }
  result.features = h;
  return result;
}

inline YoloV1Outputs YoloV1(nn::ParameterStore& p, nn::ParameterStore* state, const nn::Tensor& image, const YoloV1Config& cfg = {},
                            bool training = false, nn::Generator* dropout = nullptr) {
  auto               result = YoloV1Backbone(p, state, image, cfg, training);
  const std::int64_t in     = 1024 / cfg.width_divisor;
  auto h = nn::leaky_relu(nn::dense(p, "fc1", nn::reshape(result.features, {-1, 7 * 7 * in}), cfg.dense_width), .1);
  if (training && cfg.keep_probability < 1) {
    static thread_local nn::Generator fallback(37);
    h = nn::dropout(h, 1 - cfg.keep_probability, true, dropout ? *dropout : fallback);
  }
  result.prediction = nn::reshape(nn::dense(p, "output", h, 7 * 7 * (cfg.classes + 10)), {-1, 7, 7, cfg.classes + 10});
  return result;
}

// ---- loss, targets and decoding ---------------------------------------------------

struct YoloV1Targets {
  std::vector<float> object, boxes;  // [N, g, g], [N, g, g, 4]
  std::vector<int>   labels;         // [N, g, g]
  int                discarded = 0;
};
inline YoloV1Targets MakeYoloV1Targets(const DetectionScenes& scenes, const std::vector<int>& rows, int grid = 7) {
  const std::size_t n = rows.size(), cells = std::size_t(grid * grid);
  YoloV1Targets     t;
  t.object.assign(n * cells, 0);
  t.labels.assign(n * cells, 0);
  t.boxes.assign(n * cells * 4, 0);
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t k = 0; k < scenes.boxes[std::size_t(rows[i])].size(); ++k) {
      const auto& b = scenes.boxes[std::size_t(rows[i])][k];
      const float y = (b[0] + b[2]) * .5f / float(scenes.height) * float(grid), x = (b[1] + b[3]) * .5f / float(scenes.width) * float(grid);
      const float h = (b[2] - b[0]) / float(scenes.height), w = (b[3] - b[1]) / float(scenes.width);
      const int   r = std::clamp(int(y), 0, grid - 1), c = std::clamp(int(x), 0, grid - 1);
      const auto  slot = i * cells + std::size_t(r * grid + c);
      if (t.object[slot] != 0) {
        ++t.discarded;
        const float old_h = t.boxes[slot * 4 + 2], old_w = t.boxes[slot * 4 + 3];
        if (h * w <= old_h * old_h * old_w * old_w) continue;
      }
      t.object[slot]        = 1;
      t.labels[slot]        = scenes.labels[std::size_t(rows[i])][k];
      t.boxes[slot * 4]     = y - float(r);
      t.boxes[slot * 4 + 1] = x - float(c);
      t.boxes[slot * 4 + 2] = std::sqrt(h);
      t.boxes[slot * 4 + 3] = std::sqrt(w);
    }
  return t;
}

struct YoloV1LossOutputs {
  nn::Tensor loss, responsible, iou;  // [], [N, g, g, 2], [N, g, g, 2]
};
inline YoloV1LossOutputs YoloV1Loss(const nn::Tensor& prediction, const YoloV1Targets& t, int classes, int grid = 7) {
  const std::int64_t n = prediction.shape()[0], g = grid;
  const auto         object = nn::from_vector(t.object, {n, g, g});
  const auto         scores = nn::slice(prediction, {0, 0, 0, 0}, {n, g, g, classes});
  const auto         parts  = nn::unstack(nn::reshape(nn::slice(prediction, {0, 0, 0, classes}, {n, g, g, 10}), {n, g, g, 2, 5}), 4);
  const auto         target = nn::unstack(nn::expand_dims(nn::from_vector(t.boxes, {n, g, g, 4}), 3), 4);
  std::vector<float> index(static_cast<std::size_t>(grid), 0.f);
  std::iota(index.begin(), index.end(), 0.f);
  const auto rows = nn::from_vector(index, {1, g, 1, 1}), cols = nn::from_vector(index, {1, 1, g, 1});
  const auto py = (rows + parts[0]) / double(grid), px = (cols + parts[1]) / double(grid);
  const auto ty = (rows + target[0]) / double(grid), tx = (cols + target[1]) / double(grid);
  const auto ph = nn::square(parts[2]), pw = nn::square(parts[3]), th = nn::square(target[2]), tw = nn::square(target[3]);
  YoloV1LossOutputs out;
  {
    const auto zero = nn::scalar(0.0);
    const auto ih   = nn::maximum(nn::minimum(py + ph * .5, ty + th * .5) - nn::maximum(py - ph * .5, ty - th * .5), zero);
    const auto iw   = nn::maximum(nn::minimum(px + pw * .5, tx + tw * .5) - nn::maximum(px - pw * .5, tx - tw * .5), zero);
    const auto intersection = ih * iw;
    out.iou = (intersection / nn::maximum(ph * pw + th * tw - intersection, nn::scalar(1e-12))).detach();
  }
  const auto         iou = out.iou.to_vector<float>();
  std::vector<float> responsible(iou.size(), 0);
  for (std::size_t k = 0; k < iou.size() / 2; ++k)
    if (t.object[k] != 0) responsible[2 * k + (iou[2 * k + 1] > iou[2 * k] ? 1 : 0)] = 1;
  out.responsible = nn::from_vector(responsible, {n, g, g, 2});
  nn::Tensor coordinate = nn::square(parts[0] - target[0]);
  for (std::size_t k = 1; k < 4; ++k) coordinate = coordinate + nn::square(parts[k] - target[k]);
  const auto object_confidence     = out.responsible * nn::square(parts[4] - out.iou);
  const auto background_confidence = (1.0 - out.responsible) * nn::square(parts[4]);
  const auto localization = nn::sum(coordinate * (out.responsible * 5.0) + (object_confidence + background_confidence * .5), {1, 2, 3});
  const auto one_hot      = nn::one_hot(nn::from_vector(std::vector<std::int32_t>(t.labels.begin(), t.labels.end()), {n, g, g}), classes);
  const auto classification = nn::sum(nn::square(scores - one_hot) * nn::expand_dims(object, 3), {1, 2, 3});
  out.loss = nn::mean(localization + classification);
  return out;
}

/// Detections from predictions [N, g, g, classes + 10] (host values) of scene rows `chunk`; no NMS.
inline void DecodeYoloV1(const std::vector<float>& values, const std::vector<int>& chunk, const DetectionScenes& scenes, int classes,
                         int grid, std::vector<detector::Detection>& detections) {
  const std::size_t channels = std::size_t(classes + 10);
  const float       H = float(scenes.height), W = float(scenes.width);
  for (std::size_t i = 0; i < chunk.size(); ++i)
    for (int r = 0; r < grid; ++r)
      for (int c = 0; c < grid; ++c) {
        const float* v = values.data() + ((i * std::size_t(grid) + std::size_t(r)) * std::size_t(grid) + std::size_t(c)) * channels;
        for (int b = 0; b < 2; ++b) {
          const int   base = classes + 5 * b;
          const float y = (float(r) + v[base]) / float(grid) * H, x = (float(c) + v[base + 1]) / float(grid) * W;
          const float rh = v[base + 2], rw = v[base + 3], h = rh * rh * H, w = rw * rw * W;
          const Box   box = {std::clamp(y - h / 2, 0.f, H), std::clamp(x - w / 2, 0.f, W), std::clamp(y + h / 2, 0.f, H),
                             std::clamp(x + w / 2, 0.f, W)};
          if (!(detector::BoxArea(box) > 0)) continue;
          const float confidence = std::clamp(v[base + 4], 0.f, 1.f);
          for (int label = 0; label < classes; ++label) {
            const float score = confidence * std::clamp(v[label], 0.f, 1.f);
            if (score > .01f) detections.push_back({chunk[i], label, score, box});
          }
        }
      }
}
/// Inference in chunks of 32 with `forward(images)` returning predictions; class-wise NMS .45.
template <typename Forward>
std::vector<detector::Detection> DetectYoloV1(const Forward& forward, const DetectionScenes& scenes, const std::vector<int>& rows,
                                              int classes, int grid = 7) {
  nn::NoGradGuard                  guard;
  std::vector<detector::Detection> detections;
  for (std::size_t start = 0; start < rows.size(); start += 32) {
    const std::vector<int> chunk(rows.begin() + std::ptrdiff_t(start), rows.begin() + std::ptrdiff_t(std::min(rows.size(), start + 32)));
    const nn::Tensor       prediction = forward(detector::SceneImages(scenes, chunk));
    DecodeYoloV1(prediction.to_vector<float>(), chunk, scenes, classes, grid, detections);
  }
  return detector::ClassNms(detections, .45f);
}

// ---- optional classifier pretraining --------------------------------------------

inline DetectionScenes YoloV1PretrainingCrops(const DetectionScenes& scenes, int begin, int end) {
  if (begin < 0 || end < begin || end > scenes.size() || scenes.labels.size() != scenes.boxes.size() || scenes.height < 1 ||
      scenes.width < 1 || scenes.pixels.size() != std::size_t(scenes.size()) * std::size_t(scenes.height) * std::size_t(scenes.width))
    throw std::invalid_argument("invalid YOLO pretraining source/split");
  DetectionScenes crops;
  crops.height = crops.width = 56;
  for (int row = begin; row < end; ++row) {
    const auto& boxes = scenes.boxes[std::size_t(row)];
    if (boxes.size() != scenes.labels[std::size_t(row)].size()) throw std::invalid_argument("pretraining box/label counts differ");
    for (std::size_t k = 0; k < boxes.size(); ++k) {
      const auto& box = boxes[k];
      for (float coordinate : box)
        if (!std::isfinite(coordinate)) throw std::invalid_argument("nonfinite pretraining box");
      const float side = std::max(box[2] - box[0], box[3] - box[1]);
      if (!(box[2] > box[0] && box[3] > box[1]) || box[0] < 0 || box[1] < 0 || box[2] > scenes.height || box[3] > scenes.width ||
          scenes.labels[std::size_t(row)][k] < 0 || scenes.labels[std::size_t(row)][k] >= 10)
        throw std::invalid_argument("invalid pretraining box/class");
      const float        cy = (box[0] + box[2]) * .5f, cx = (box[1] + box[3]) * .5f;
      const Box          square = {cy - side * .5f, cx - side * .5f, cy + side * .5f, cx + side * .5f};
      std::vector<float> patch(28 * 28);
      detector::CropScene(scenes, row, square, 28, patch.data());
      const std::size_t offset = crops.pixels.size();
      crops.pixels.resize(offset + 56 * 56, 0);
      for (int y = 0; y < 28; ++y)
        for (int x = 0; x < 28; ++x)
          crops.pixels[offset + std::size_t((y + 14) * 56 + x + 14)] =
              std::uint8_t(std::lround(std::clamp((patch[std::size_t(y * 28 + x)] + 1) * 127.5f, 0.f, 255.f)));
      crops.boxes.push_back({{14, 14, 42, 42}});
      crops.labels.push_back({scenes.labels[std::size_t(row)][k]});
    }
  }
  return crops;
}

/// Images [B, 56*56] in [-1, 1] and int32 labels [B].
inline std::pair<nn::Tensor, nn::Tensor> YoloV1PretrainingBatch(const DetectionScenes& crops, const std::vector<int>& rows) {
  check::Require(crops.height == 56 && crops.width == 56 && crops.labels.size() == crops.boxes.size() &&
                     crops.pixels.size() == std::size_t(crops.size()) * 56 * 56,
                 "valid pretraining crop dimensions");
  std::vector<std::int32_t> labels;
  for (int row : rows) {
    check::Require(row >= 0 && row < crops.size() && crops.labels[std::size_t(row)].size() == 1 && crops.labels[std::size_t(row)][0] >= 0 &&
                       crops.labels[std::size_t(row)][0] < 10,
                   "valid pretraining row/class");
    labels.push_back(crops.labels[std::size_t(row)][0]);
  }
  return {nn::reshape(detector::SceneImages(crops, rows), {std::int64_t(rows.size()), 56 * 56}), nn::from_vector(labels)};
}

inline bool YoloV1PretrainedName(const std::string& name) {
  for (int layer = 1; layer <= 20; ++layer)
    if (name.rfind("conv" + std::to_string(layer) + "/", 0) == 0) return true;
  return false;
}

/// First-20-convolution classifier: average-pooled features and a 10-way "pretrain_head".
inline nn::Tensor YoloV1PretrainingLogits(nn::ParameterStore& p, nn::ParameterStore& state, const nn::Tensor& x,
                                          const YoloV1Config& cfg, bool training) {
  const auto features = YoloV1Backbone(p, &state, nn::reshape(x, {-1, 56, 56, 1}), cfg, training, 20).features;
  return nn::dense(p, "pretrain_head", nn::mean(features, {1, 2}), 10);
}

/// Classifier of the pretraining stage (tflow ml::Classifier): Adam(.001),
/// mean cross-entropy + .5 * 1e-4 * sum of squared kernels.
struct YoloV1Classifier {
  YoloV1Config       cfg;
  nn::ParameterStore parameters{17}, state{17};
  nn::Optimizer      optimizer;
  explicit YoloV1Classifier(const YoloV1Config& config)
      : cfg(config), optimizer(Init(parameters, state, config), 1e-3, AdamConfig()) {}
  static nn::OptimizerConfig AdamConfig() {
    nn::OptimizerConfig o;
    o.kind = nn::OptimizerKind::Adam;
    return o;
  }
  static const nn::ParameterStore& Init(nn::ParameterStore& p, nn::ParameterStore& s, const YoloV1Config& c) {
    nn::NoGradGuard guard;
    auto            frozen = c;
    frozen.update_stats    = false;
    YoloV1PretrainingLogits(p, s, nn::zeros({1, 56 * 56}), frozen, true);
    return p;
  }
  void Train(const nn::Tensor& x, const nn::Tensor& y) {
    const auto logits = YoloV1PretrainingLogits(parameters, state, x, cfg, true);
    const auto loss   = nn::mean(-nn::sum(nn::one_hot(y, 10) * nn::log_softmax(logits, -1), {1}));
    nn::Tensor penalty;
    for (const auto& t : parameters.parameters())
      if (t.dim() >= 2) penalty = penalty.defined() ? penalty + nn::sum(nn::square(t)) : nn::sum(nn::square(t));
    optimizer.minimize(loss + .5e-4 * penalty);
  }
  /// {mean NLL, accuracy} of the inference path.
  std::pair<double, double> Evaluate(const nn::Tensor& x, const nn::Tensor& y) {
    nn::NoGradGuard guard;
    const auto      logits = YoloV1PretrainingLogits(parameters, state, x, cfg, false);
    const double    loss   = nn::mean(-nn::sum(nn::one_hot(y, 10) * nn::log_softmax(logits, -1), {1})).item();
    const auto      pred = nn::argmax(logits, 1).to_vector<std::int64_t>();
    const auto      truth = y.to_vector<std::int32_t>();
    double          correct = 0;
    for (std::size_t i = 0; i < pred.size(); ++i) correct += pred[i] == truth[i];
    return {loss, correct / double(pred.size())};
  }
};

inline std::pair<double, double> EvaluateYoloV1Pretraining(YoloV1Classifier& model, const DetectionScenes& crops) {
  double loss = 0, accuracy = 0;
  for (int start = 0; start < crops.size(); start += 64) {
    std::vector<int> rows(std::size_t(std::min(64, crops.size() - start)));
    std::iota(rows.begin(), rows.end(), start);
    const auto [x, y]        = YoloV1PretrainingBatch(crops, rows);
    const auto [l, a]        = model.Evaluate(x, y);
    loss += l * double(rows.size());
    accuracy += a * double(rows.size());
  }
  return {loss / crops.size(), accuracy / crops.size()};
}

struct YoloV1PretrainedWeights {
  std::map<std::string, nn::Tensor> parameters, state;
};
inline YoloV1PretrainedWeights CaptureYoloV1Pretrained(const YoloV1Classifier& model) {
  YoloV1PretrainedWeights weights;
  for (const auto& name : model.parameters.names())
    if (YoloV1PretrainedName(name)) weights.parameters.emplace(name, model.parameters.at(name).detach().clone());
  for (const auto& name : model.state.names())
    if (YoloV1PretrainedName(name)) weights.state.emplace(name, model.state.at(name).detach().clone());
  return weights;
}

/// Copies the prefix into a fresh 24-convolution detector (seed 17) and saves
/// it as `<prefix>_params.npz` / `<prefix>_state.npz`, verifying every value.
inline void WriteYoloV1Bootstrap(const YoloV1PretrainedWeights& weights, const YoloV1Config& cfg, const std::filesystem::path& prefix) {
  check::Require(cfg.compact && cfg.width_divisor == 8 && cfg.classes == 10, "compact digit bootstrap configuration");
  check::Require(weights.parameters.size() == std::size_t(cfg.batch_norm ? 60 : 40) &&
                     weights.state.size() == std::size_t(cfg.batch_norm ? 40 : 0),
                 "complete first-20 prefix weights and BN state");
  nn::ParameterStore p(17), state(17);
  {
    nn::NoGradGuard guard;
    YoloV1(p, &state, nn::zeros({1, 56, 56, 1}), cfg, false);
  }
  auto copy = [&](const std::map<std::string, nn::Tensor>& source, nn::ParameterStore& destination) {
    for (const auto& [name, value] : source) {
      check::Require(YoloV1PretrainedName(name) && destination.contains(name) && destination.at(name).shape() == value.shape(),
                     "pretrained variable name/type/shape matches");
      for (float v : value.to_vector<float>()) check::Require(std::isfinite(v), "finite pretrained weight");
      destination.set(name, value);
    }
    for (const auto& [name, value] : source) {
      const auto a = destination.at(name).to_vector<float>(), b = value.to_vector<float>();
      for (std::size_t i = 0; i < a.size(); ++i) check::Near(a[i], b[i], 0, "exact pretrained-prefix transfer");
    }
  };
  copy(weights.parameters, p);
  copy(weights.state, state);
  std::filesystem::create_directories(prefix.parent_path());
  detector::SaveCheckpoint(p, state, prefix.string());
}

}  // namespace yolo
