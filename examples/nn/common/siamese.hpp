// Shared four-convolution Siamese encoder, learned weighted-L1 logit and the
// short class-disjoint MNIST training/verification/one-shot harness: a port of
// tflow's common/nn/siamese.h and common/ml/siamese.h (namespace metric).
// TF -> datamunge: the model is eager (nn::ParameterStore "siamese", seed 17,
// datamunge's generator); the checkpoint is best.npz instead of a SaveV2
// bundle; the episode preview PNG is written with nn::save_png.
#pragma once

#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/checks.hpp"
#include "common/siamese_data.hpp"

namespace metric {

// ---- nn/siamese.h --------------------------------------------------------------------

/// Full configuration follows Koch et al.; compact uses smaller kernels,
/// channels and images, preserving every stage and the sigmoid embedding.
struct SiameseConfig {
  int                side = 105, embedding = 4096;
  std::array<int, 4> channels = {64, 128, 128, 256}, kernels = {10, 7, 4, 4};
  static SiameseConfig Compact() {
    SiameseConfig c;
    c.side      = 48;
    c.embedding = 64;
    c.channels  = {8, 16, 16, 32};
    c.kernels   = {5, 3, 3, 2};
    return c;
  }
};

struct SiameseFeatures {
  nn::Tensor              embedding;
  std::vector<nn::Tensor> stages;
};

/// Four VALID He-normal ReLU convolutions ("encoder/conv0..3"), 2x2 max pools
/// after the first three, flatten and a Glorot sigmoid dense embedding.
inline SiameseFeatures SiameseEncoder(nn::ParameterStore& p, const nn::Tensor& images, const SiameseConfig& c) {
  SiameseFeatures      out;
  nn::Tensor           x    = images;
  int                  side = c.side;
  nn::ConvLayerOptions conv;
  conv.padding    = nn::Padding::Valid;
  conv.activation = nn::Activation::Relu;
  conv.init       = nn::init::he_normal();
  for (int i = 0; i < 4; ++i) {
    side -= c.kernels[i] - 1;
    if (side < 1) throw std::invalid_argument("Siamese image too small");
    x = nn::conv2d(p, "encoder/conv" + std::to_string(i), x, c.channels[i], c.kernels[i], conv);
    out.stages.push_back(x);
    if (i < 3) {
      x = nn::max_pool2d(x, 2, 2);
      side /= 2;
    }
  }
  nn::DenseOptions dense;
  dense.activation = nn::Activation::Sigmoid;
  out.embedding    = nn::dense(p, "encoder/embedding", nn::reshape(x, {-1, std::int64_t(side) * side * c.channels[3]}), c.embedding, dense);
  return out;
}

/// |left - right| W + b, flattened to [N]. Negative initial weights (-.1) give
/// distance a useful initial orientation; they remain unconstrained learned
/// coefficients, not a guaranteed metric.
inline nn::Tensor SiameseLogit(nn::ParameterStore& p, const nn::Tensor& left, const nn::Tensor& right, int dimensions) {
  const nn::Tensor weights = p.get("similarity/W", {dimensions, 1}, nn::init::constant(-.1));
  const nn::Tensor bias    = p.get("similarity/b", {1}, nn::init::constant(.1));
  return nn::reshape(nn::bias_add(nn::matmul(nn::abs(left - right), weights), bias), {-1});
}

/// tflow's ml::BinaryCrossEntropy: per-element stable sigmoid cross entropy
/// with soft labels, built from selects exactly as TF does so the derivative
/// at a zero logit is sigmoid(0) - label.
inline nn::Tensor BinaryCrossEntropy(const nn::Tensor& logits, const nn::Tensor& labels) {
  const nn::Tensor positive           = nn::greater_equal(logits, 0.0);
  const nn::Tensor negative_magnitude = nn::where(positive, -logits, logits);
  const nn::Tensor linear             = nn::where(positive, logits * (1.0 - labels), -(logits * labels));
  return linear + nn::log1p(nn::exp(negative_magnitude));
}

// ---- ml/siamese.h --------------------------------------------------------------------

/// Compact twin model on MNIST resized 28 -> 48 (bilinear, TF legacy corners),
/// mean BCE + .00005 sum W^2 (rank > 1), Adam .001.
class SiameseModel {
 public:
  nn::ParameterStore             parameters{17};
  std::unique_ptr<nn::Optimizer> optimizer;
  SiameseModel() {
    {
      nn::NoGradGuard guard;
      Forward(nn::zeros({1, 28, 28, 1}), nn::zeros({1, 28, 28, 1}));
    }
    nn::OptimizerConfig oc;
    oc.kind   = nn::OptimizerKind::Adam;
    optimizer = std::make_unique<nn::Optimizer>(parameters, .001, oc);
  }
  nn::Tensor Forward(const nn::Tensor& left, const nn::Tensor& right) {
    const auto cfg = SiameseConfig::Compact();
    const auto a   = SiameseEncoder(parameters, nn::resize_bilinear(left, {48, 48}), cfg);
    const auto b   = SiameseEncoder(parameters, nn::resize_bilinear(right, {48, 48}), cfg);
    return SiameseLogit(parameters, a.embedding, b.embedding, cfg.embedding);
  }
  /// {mean BCE, logits}; `update` takes one Adam step on BCE + L2.
  std::pair<double, std::vector<float>> Run(const SiameseBatch& b, bool update = false) {
    std::unique_ptr<nn::NoGradGuard> guard;
    if (!update) guard = std::make_unique<nn::NoGradGuard>();
    const nn::Tensor logits = Forward(b.left, b.right);
    const nn::Tensor loss   = nn::mean(BinaryCrossEntropy(logits, b.labels));
    if (update) {
      nn::Tensor penalty;
      for (const auto& v : parameters.parameters())
        if (v.dim() > 1) penalty = penalty.defined() ? penalty + nn::sum(nn::square(v)) : nn::sum(nn::square(v));
      optimizer->step(nn::grad(loss + .00005 * penalty, parameters.parameters()));
    }
    return {loss.item(), logits.contiguous().to_vector<float>()};
  }
  void Save(const std::string& path) const { parameters.save(path); }
  void Restore(const std::string& path) { parameters.load(path); }
};

struct VerificationMetrics {
  double loss = 0, accuracy = 0, positive_probability = 0, negative_probability = 0;
  int    tp = 0, fp = 0, fn = 0, tn = 0;
};

inline VerificationMetrics ScoreSiamesePairs(SiameseModel& model, const nn::ImageDataset& d, const std::vector<VerificationPair>& pairs) {
  VerificationMetrics result;
  int                 positives = 0, negatives = 0;
  for (std::size_t start = 0; start < pairs.size(); start += 64) {
    const auto                          end = std::min(pairs.size(), start + 64);
    const std::vector<VerificationPair> part(pairs.begin() + std::ptrdiff_t(start), pairs.begin() + std::ptrdiff_t(end));
    const auto                          out = model.Run(MakeSiameseBatch(d, part));
    result.loss += out.first * double(part.size());
    for (std::size_t i = 0; i < part.size(); ++i) {
      const double logit = out.second[i];
      check::Require(std::isfinite(logit), "finite verification logit");
      const double probability = 1 / (1 + std::exp(-logit));
      if (part[i].same) {
        result.positive_probability += probability;
        ++positives;
        if (logit >= 0)
          ++result.tp;
        else
          ++result.fn;
      } else {
        result.negative_probability += probability;
        ++negatives;
        if (logit >= 0)
          ++result.fp;
        else
          ++result.tn;
      }
    }
  }
  check::Require(positives > 0 && negatives > 0, "both pair labels evaluated");
  result.loss /= double(pairs.size());
  result.accuracy = double(result.tp + result.tn) / double(pairs.size());
  result.positive_probability /= positives;
  result.negative_probability /= negatives;
  return result;
}

struct OneShotMetrics {
  double                           accuracy = 0, pixel_accuracy = 0;
  std::vector<int>                 predicted;
  std::vector<std::vector<double>> scores;
};

/// uint8 pixels of [0,1] float32 MNIST images (exact: the loader divides by 255).
struct DigitBytes {
  nn::Tensor   images;
  const float* data;
  explicit DigitBytes(const nn::ImageDataset& d) : images(d.images.contiguous()), data(images.data_ptr<float>()) {}
  int operator[](std::size_t i) const { return int(std::lround(data[i] * 255.0)); }
};

inline OneShotMetrics ScoreSiameseTasks(SiameseModel& model, const nn::ImageDataset& d, const std::vector<OneShotTask>& tasks) {
  std::vector<VerificationPair> pairs;
  for (const auto& task : tasks)
    for (std::size_t row : task.support) pairs.push_back({task.query, row, 0});
  std::vector<double> logits;
  for (std::size_t start = 0; start < pairs.size(); start += 64) {
    const auto end = std::min(pairs.size(), start + 64);
    const auto out = model.Run(MakeSiameseBatch(d, {pairs.begin() + std::ptrdiff_t(start), pairs.begin() + std::ptrdiff_t(end)}));
    for (float v : out.second) logits.push_back(v);
  }
  const DigitBytes bytes(d);
  OneShotMetrics result;
  std::size_t    offset = 0;
  int            correct = 0, pixel_correct = 0;
  for (const auto& task : tasks) {
    std::vector<double> scores(logits.begin() + std::ptrdiff_t(offset), logits.begin() + std::ptrdiff_t(offset + task.support.size())), pixels;
    offset += task.support.size();
    for (std::size_t row : task.support) {
      double distance = 0;
      for (int j = 0; j < 784; ++j) {
        const double delta = double(bytes[task.query * 784 + std::size_t(j)]) - bytes[row * 784 + std::size_t(j)];
        distance += delta * delta;
      }
      pixels.push_back(-distance);
    }
    const int predicted = BestOneShot(scores);
    correct += predicted == task.correct;
    pixel_correct += BestOneShot(pixels) == task.correct;
    result.predicted.push_back(predicted);
    result.scores.push_back(scores);
  }
  result.accuracy       = double(correct) / double(tasks.size());
  result.pixel_accuracy = double(pixel_correct) / double(tasks.size());
  return result;
}

inline void ExportSiameseTasks(const nn::ImageDataset& d, const std::vector<OneShotTask>& tasks, const OneShotMetrics& result,
                               const std::filesystem::path& dir) {
  const int          count = std::min<int>(8, int(tasks.size())), ways = int(tasks.front().support.size()), cell = 60;
  const DigitBytes   bytes(d);
  const auto         labels = Labels(d);
  const int          width  = (ways + 1) * cell;
  std::vector<float> rgb(std::size_t(count * cell * width * 3), 0.f);
  std::ofstream      table(dir / "episodes.tsv");
  table << "episode\tquery_digit\tsupport_digits\tpredicted_digit\tcorrect\tlogits\n";
  for (std::size_t i = 0; i < tasks.size(); ++i) {
    const auto& task = tasks[i];
    table << i << '\t' << labels[task.query] << '\t';
    for (int c : task.classes) table << c << ',';
    table << '\t' << task.classes[std::size_t(result.predicted[i])] << '\t' << (result.predicted[i] == task.correct) << '\t';
    for (double value : result.scores[i]) table << value << ',';
    table << '\n';
    if (int(i) >= count) continue;
    for (int col = 0; col <= ways; ++col) {
      const std::size_t row = col == 0 ? task.query : task.support[std::size_t(col - 1)];
      for (int y = 0; y < cell; ++y)
        for (int x = 0; x < cell; ++x)
          for (int c = 0; c < 3; ++c) {
            int value = 25;
            if (y >= 2 && y < 58 && x >= 2 && x < 58)
              value = bytes[row * 784 + std::size_t(((y - 2) / 2) * 28 + (x - 2) / 2)];
            else if (col && col - 1 == result.predicted[i])
              value = c == (result.predicted[i] == task.correct ? 1 : 0) ? 255 : 30;
            rgb[std::size_t(((int(i) * cell + y) * width + col * cell + x) * 3 + c)] = float(value);
          }
    }
  }
  nn::save_png(nn::from_vector(rgb, {count * cell, width, 3}), (dir / "episodes.png").string(), 0, 255);
  table.close();
  check::Require(bool(table), "write one-shot artifacts");
}

}  // namespace metric
