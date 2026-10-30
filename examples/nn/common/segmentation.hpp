// End-to-end semantic-segmentation training and held-out mean IoU (port of
// tflow's common/ml/segmentation.h and common/data/segmentation_shapes.h).
//
// Procedural data: circles, rectangles and triangles with identical
// class-independent foreground photometry, random scale/location, occlusion
// and noise. Labels are shape identity, so recognizing class needs spatial
// context rather than a fixed color lookup. Scenes are drawn with
// tensor generators and tensor geometry. Seeds reproduce this adaptation;
// pixels differ from tflow's mt19937_64 draw sequence.
//
// Training: balanced pixel CE with inverse-square-root class frequencies of
// the fit masks, Adam, 0.5 * l2 * sum of squared kernels (rank >= 2), the
// validation-best parameter/state checkpoint (.npz instead of a TF bundle),
// confusion matrices and colored image/truth/prediction previews. Batch-norm
// moving averages update eagerly during training-mode forwards (TF runs the
// collected update ops with the training step).
#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <datamunge/nn/image_io.hpp>
#include <datamunge/nn/nn.hpp>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "common/checks.hpp"
#include "common/paths.hpp"

namespace seg {

namespace nn = datamunge::nn;

// ---- procedural shapes (tflow data/segmentation_shapes.h) ---------------------------------

struct SegmentationData {
  int count = 0, height = 0, width = 0, classes = 0;
  std::vector<float> images; // count x H x W x 3 in [0, 1]
  std::vector<int> labels;   // count x H x W
};

inline SegmentationData SegmentationShapes(int count, int size, std::uint64_t seed) {
  const auto t = nn::models::TensorSegmentationShapes(count, size, seed);
  return {count, size, size, 4, t.images.to_vector<float>(), t.labels.to_vector<int>()};
}

/// Images [B, H, W, 3] (float32) and labels [B, H, W] (int32) of the given rows.
inline std::pair<nn::Tensor, nn::Tensor> SegmentationBatch(const SegmentationData &data,
                                                           const std::vector<int> &rows) {
  const std::size_t pixels = std::size_t(data.height) * data.width;
  std::vector<float> images(rows.size() * pixels * 3);
  std::vector<std::int32_t> labels(rows.size() * pixels);
  for (std::size_t i = 0; i < rows.size(); ++i) {
    if (rows[i] < 0 || rows[i] >= data.count)
      throw std::out_of_range("segmentation batch index");
    std::copy_n(data.images.data() + std::size_t(rows[i]) * pixels * 3, pixels * 3,
                images.data() + i * pixels * 3);
    std::copy_n(data.labels.data() + std::size_t(rows[i]) * pixels, pixels,
                labels.data() + i * pixels);
  }
  const auto B = std::int64_t(rows.size());
  return {nn::from_vector(images, {B, data.height, data.width, 3}),
          nn::from_vector(labels, {B, data.height, data.width})};
}

// ---- loss -----------------------------------------------------------------------------------

/// Class-weighted pixel cross-entropy sum(w_y CE) / sum(w_y); logits [..., classes], labels [...] int.
using nn::models::SegmentationCrossEntropy;

inline void SegmentationLossChecks() {
  nn::Tensor logits = nn::zeros({1, 1, 2, 3});
  logits.requires_grad_(true);
  const auto labels = nn::from_vector(std::vector<std::int32_t>{0, 2}, {1, 1, 2});
  const auto weights = nn::from_vector(std::vector<float>{1, 2, 4}, {3});
  const auto loss = SegmentationCrossEntropy(logits, labels, 3, weights);
  const auto g = nn::grad({loss}, {logits})[0].to_vector<float>();
  check::Near(loss.item<double>(), std::log(3.), 1e-6, "balanced pixel CE value");
  for (int pixel = 0; pixel < 2; ++pixel)
    for (int c = 0; c < 3; ++c)
      check::Near(g[std::size_t(pixel * 3 + c)],
                  (1. / 3 - (c == (pixel ? 2 : 0))) * (pixel ? 4 : 1) / 5, 1e-7,
                  "class weighting and flattened LogSoftmax derivatives");
  std::cout << "  balanced pixel CE value and every analytic logit derivative passed"
            << std::endl;
}

// ---- training ---------------------------------------------------------------------------------

struct SegmentationModel {
  nn::Tensor logits;
  std::vector<nn::Tensor> auxiliary;
  float auxiliary_weight = 1.f;
};
/// builder(params, state, images, training): logits [B, H, W, classes] (+ auxiliary logits).
using SegmentationBuilder = std::function<SegmentationModel(
    nn::ParameterStore &, nn::ParameterStore &, const nn::Tensor &, bool)>;
/// Inference logits of images [B, H, W, 3].
using LogitsFn = std::function<nn::Tensor(const nn::Tensor &)>;

struct SegmentationConfig {
  std::string name, warm_start;
  int epochs = 20, batch = 16;
  float learning_rate = .001f, l2 = .0001f;
  double minimum_test_iou = .55;
  bool accuracy = true;
  std::uint64_t seed = 17; // parameter initialization seed (tflow's Params seed)
  /// Held-out diagnostics after the restore: (logits, test data, output directory).
  std::function<void(const LogitsFn &, const SegmentationData &,
                     const std::filesystem::path &)>
      inspect;
};
struct SegmentationMetrics {
  double mean_iou = 0, pixel_accuracy = 0;
  std::vector<std::int64_t> confusion;
};

inline void CheckSegmentationDemonstration(const SegmentationConfig &cfg,
                                           const std::vector<double> &losses,
                                           const SegmentationMetrics &result) {
  for (double value : losses)
    check::Require(std::isfinite(value), "finite segmentation epoch mean");
  for (double value : {result.mean_iou, result.pixel_accuracy})
    check::Require(std::isfinite(value) && value >= 0 && value <= 1,
                   "finite segmentation metric in [0,1]");
  if (cfg.accuracy)
    check::Require(result.mean_iou >= cfg.minimum_test_iou,
                   "segmentation demonstration mean IoU threshold");
  else {
    check::Require(losses.size() >= 2,
                   "short segmentation demonstration needs at least two epochs");
    check::Require(losses.back() <
                       losses.front() - 1e-4 * std::max(1., std::abs(losses.front())),
                   "short segmentation demonstration must reduce mean training loss");
  }
}

inline void SegmentationTrainingChecks() {
  SegmentationConfig cfg;
  cfg.accuracy = false;
  SegmentationMetrics metrics;
  metrics.mean_iou = .1;
  metrics.pixel_accuracy = .5;
  CheckSegmentationDemonstration(cfg, {3, 2, 1}, metrics);
  auto rejects = [&](const std::vector<double> &losses, const SegmentationMetrics &value) {
    bool rejected = false;
    try {
      CheckSegmentationDemonstration(cfg, losses, value);
    } catch (const std::runtime_error &) {
      rejected = true;
    }
    check::Require(rejected, "segmentation training policy rejects invalid evidence");
  };
  rejects({1, 2, 3}, metrics);
  rejects({1, 1, 1}, metrics);
  rejects({1}, metrics);
  rejects({3, NAN, 1}, metrics);
  rejects({INFINITY, 2, 1}, metrics);
  auto invalid = metrics;
  invalid.mean_iou = NAN;
  rejects({3, 2, 1}, invalid);
  invalid = metrics;
  invalid.pixel_accuracy = 1.1;
  rejects({3, 2, 1}, invalid);
  cfg.accuracy = true;
  rejects({3, 2, 1}, metrics);
  metrics.mean_iou = cfg.minimum_test_iou;
  CheckSegmentationDemonstration(cfg, {3, 2, 1}, metrics);
  std::cout << "  short-segmentation learning and strict-IoU policies: falling loss accepted; "
               "flat/rising/nonfinite "
               "loss, invalid metrics and unmet accuracy rejected"
            << std::endl;
}

/// Mean IoU over classes present in truth or prediction, from a confusion matrix (truth rows).
inline double ConfusionMeanIou(const std::vector<std::int64_t> &confusion, int classes) {
  return nn::models::SegmentationConfusionMetrics(
             nn::from_vector(confusion, {classes, classes}))
      .mean_iou.item();
}

inline SegmentationMetrics SegmentationScore(const LogitsFn& logits_of,const SegmentationData& data){nn::NoGradGuard guard;auto confusion=nn::zeros({data.classes,data.classes},nn::DType::Int64);for(int start=0;start<data.count;start+=16){std::vector<int> rows(std::size_t(std::min(16,data.count-start)));std::iota(rows.begin(),rows.end(),start);const auto batch=SegmentationBatch(data,rows);confusion=confusion+nn::models::SegmentationPredictionMetrics(nn::argmax(logits_of(batch.first),3),batch.second,data.classes).confusion;}const auto metrics=nn::models::SegmentationConfusionMetrics(confusion);return {metrics.mean_iou.item(),metrics.pixel_accuracy.item(),confusion.to_vector<std::int64_t>()};}

inline const unsigned char kSegmentationPalette[4][3] = {
    {24, 24, 32}, {230, 70, 70}, {70, 210, 110}, {80, 120, 240}};

/// First four images: image | truth | prediction rows.
inline void SegmentationPreview(const LogitsFn &logits_of, const SegmentationData &data,
                                const std::filesystem::path &filename) {
  nn::NoGradGuard guard;
  const int count = std::min(4, data.count), h = data.height, w = data.width;
  std::vector<int> rows(static_cast<std::size_t>(count));
  std::iota(rows.begin(), rows.end(), 0);
  const auto batch = SegmentationBatch(data, rows);
  const auto predictions = nn::argmax(logits_of(batch.first), 3).to_vector<std::int64_t>();
  const auto images = batch.first.to_vector<float>();
  const auto targets = batch.second.to_vector<std::int32_t>();
  std::vector<float> rgb(std::size_t(count) * h * 3 * w * 3);
  for (int n = 0; n < count; ++n)
    for (int r = 0; r < h; ++r)
      for (int c = 0; c < w; ++c) {
        const std::size_t index = (std::size_t(n) * h + r) * w + c;
        const int target = targets[index], prediction = int(predictions[index]);
        for (int k = 0; k < 3; ++k) {
          const std::size_t row = std::size_t(n * h + r) * std::size_t(3 * w);
          rgb[(row + c) * 3 + k] =
              float(std::clamp(std::lround(images[index * 3 + k] * 255), 0L, 255L));
          rgb[(row + w + c) * 3 + k] = kSegmentationPalette[target % 4][k];
          rgb[(row + 2 * w + c) * 3 + k] = kSegmentationPalette[prediction % 4][k];
        }
      }
  nn::save_png(nn::from_vector(rgb, {count * h, 3 * w, 3}), filename.string(), 0, 255);
}

inline std::filesystem::path SegmentationDirectory(const std::string &name) {
  return paths::OutputDir("models/" + name);
}

} // namespace seg
