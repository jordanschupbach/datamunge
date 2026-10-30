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
#include "datamunge/nn/siamese_models.hpp"
#include "datamunge/nn/optim.hpp"
#include "datamunge/nn/image_io.hpp"

namespace metric {

// ---- nn/siamese.h --------------------------------------------------------------------

using nn::models::SiameseConfig;
using nn::models::SiameseEncoder;
using nn::models::SiameseFeatures;
using nn::models::SiameseLogit;
inline nn::Tensor BinaryCrossEntropy(const nn::Tensor &logits, const nn::Tensor &labels) {
  return nn::models::SiameseBinaryCrossEntropy(logits, labels);
}

// ---- ml/siamese.h --------------------------------------------------------------------

/// Compact twin model on MNIST resized 28 -> 48 (bilinear, TF legacy corners),
/// mean BCE + .00005 sum W^2 (rank > 1), Adam .001.
class SiameseModel {
public:
  nn::ParameterStore parameters{17};
  std::unique_ptr<nn::Optimizer> optimizer;
  SiameseModel() {
    {
      nn::NoGradGuard guard;
      Forward(nn::zeros({1, 28, 28, 1}), nn::zeros({1, 28, 28, 1}));
    }
    nn::OptimizerConfig oc;
    oc.kind = nn::OptimizerKind::Adam;
    optimizer = std::make_unique<nn::Optimizer>(parameters, .001, oc);
  }
  nn::Tensor Forward(const nn::Tensor &left, const nn::Tensor &right) {
    const auto cfg = SiameseConfig::Compact();
    const auto a = SiameseEncoder(parameters, nn::resize_bilinear(left, {48, 48}), cfg);
    const auto b = SiameseEncoder(parameters, nn::resize_bilinear(right, {48, 48}), cfg);
    return SiameseLogit(parameters, a.embedding, b.embedding, cfg.embedding);
  }
  /// {mean BCE, logits}; `update` takes one Adam step on BCE + L2.
  std::pair<double, std::vector<float>> Run(const SiameseBatch &b, bool update = false) {
    std::unique_ptr<nn::NoGradGuard> guard;
    if (!update)
      guard = std::make_unique<nn::NoGradGuard>();
    const nn::Tensor logits = Forward(b.left, b.right);
    const nn::Tensor loss = nn::mean(BinaryCrossEntropy(logits, b.labels));
    if (update) {
      nn::Tensor penalty;
      for (const auto &v : parameters.parameters())
        if (v.dim() > 1)
          penalty =
              penalty.defined() ? penalty + nn::sum(nn::square(v)) : nn::sum(nn::square(v));
      optimizer->step(nn::grad(loss + .00005 * penalty, parameters.parameters()));
    }
    return {loss.item(), logits.contiguous().to_vector<float>()};
  }
  void Save(const std::string &path) const {
    parameters.save(path);
  }
  void Restore(const std::string &path) {
    parameters.load(path);
  }
};

struct VerificationMetrics {
  double loss = 0, accuracy = 0, positive_probability = 0, negative_probability = 0;
  int tp = 0, fp = 0, fn = 0, tn = 0;
};

inline VerificationMetrics ScoreSiamesePairs(SiameseModel &model, const nn::ImageDataset &d,
                                             const std::vector<VerificationPair> &pairs) {
  std::vector<nn::Tensor> logits, targets;
  for (std::size_t start = 0; start < pairs.size(); start += 64) {
    const auto end = std::min(pairs.size(), start + 64);
    const auto batch = MakeSiameseBatch(
        d, {pairs.begin() + std::ptrdiff_t(start), pairs.begin() + std::ptrdiff_t(end)});
    nn::NoGradGuard guard;
    logits.push_back(model.Forward(batch.left, batch.right));
    targets.push_back(batch.labels);
  }
  const auto values =
      nn::models::SiameseVerificationMetrics(nn::concat(logits, 0), nn::concat(targets, 0))
          .to_vector<double>();
  check::Require(values[4] + values[6] > 0 && values[5] + values[7] > 0,
                 "both pair labels evaluated");
  return {values[0],      values[1],      values[2],      values[3],
          int(values[4]), int(values[5]), int(values[6]), int(values[7])};
}

struct OneShotMetrics {
  double accuracy = 0, pixel_accuracy = 0;
  std::vector<int> predicted;
  std::vector<std::vector<double>> scores;
};

/// uint8 pixels of [0,1] float32 MNIST images (exact: the loader divides by 255).
struct DigitBytes {
  nn::Tensor images;
  const float *data;
  explicit DigitBytes(const nn::ImageDataset &d)
      : images(d.images.contiguous()), data(images.data_ptr<float>()) {
  }
  int operator[](std::size_t i) const {
    return nn::floor(
               nn::slice_axis(images.reshape({-1}), 0, std::int64_t(i), std::int64_t(i + 1))
                       .to(nn::DType::Float64) *
                   255. +
               .5)
        .item<int>();
  }
};

inline OneShotMetrics ScoreSiameseTasks(SiameseModel &model, const nn::ImageDataset &d,
                                        const std::vector<OneShotTask> &tasks) {
  std::vector<VerificationPair> pairs;
  for (const auto &task : tasks)
    for (std::size_t row : task.support)
      pairs.push_back({task.query, row, 0});
  std::vector<double> logits;
  for (std::size_t start = 0; start < pairs.size(); start += 64) {
    const auto end = std::min(pairs.size(), start + 64);
    const auto out = model.Run(MakeSiameseBatch(
        d, {pairs.begin() + std::ptrdiff_t(start), pairs.begin() + std::ptrdiff_t(end)}));
    for (float v : out.second)
      logits.push_back(v);
  }
  if (tasks.empty())
    throw std::invalid_argument("nonempty episodes required");
  const auto ways = tasks.front().support.size();
  const auto scores =
      nn::from_vector(logits, {std::int64_t(tasks.size()), std::int64_t(ways)});
  std::vector<std::int64_t> correct;
  for (const auto &task : tasks) {
    if (task.support.size() != ways)
      throw std::invalid_argument("episode ways mismatch");
    correct.push_back(task.correct);
  }
  const auto targets = nn::from_vector(correct);
  OneShotMetrics result;
  result.accuracy = nn::models::SiameseOneShotAccuracy(scores, targets).item();
  result.pixel_accuracy =
      nn::models::SiameseOneShotAccuracy(nn::models::SiamesePixelScores(d, tasks), targets)
          .item();
  result.predicted = nn::argmax(scores, 1).to_vector<int>();
  for (std::size_t i = 0; i < tasks.size(); ++i)
    result.scores.push_back(nn::slice_axis(scores, 0, i, i + 1).to_vector<double>());
  return result;
}

inline void ExportSiameseTasks(const nn::ImageDataset &d,
                               const std::vector<OneShotTask> &tasks,
                               const OneShotMetrics &result,
                               const std::filesystem::path &dir) {
  const auto labels = Labels(d);
  std::ofstream table(dir / "episodes.tsv");
  table << "episode\tquery_digit\tsupport_digits\tpredicted_digit\tcorrect\tlogits\n";
  for (std::size_t i = 0; i < tasks.size(); ++i) {
    const auto &task = tasks[i];
    table << i << '\t' << labels[task.query] << '\t';
    for (int c : task.classes)
      table << c << ',';
    table << '\t' << task.classes[std::size_t(result.predicted[i])] << '\t'
          << (result.predicted[i] == task.correct) << '\t';
    for (double value : result.scores[i])
      table << value << ',';
    table << '\n';
  }
  std::vector<double> scores;
  for (const auto &row : result.scores)
    scores.insert(scores.end(), row.begin(), row.end());
  const auto values = nn::from_vector(
      scores, {std::int64_t(tasks.size()), std::int64_t(tasks.front().support.size())});
  nn::save_image_grid(nn::models::SiameseEpisodeGallery(d, tasks, values),
                      (dir / "episodes.png").string(), 1, 0, 1, 0);
  table.close();
  check::Require(bool(table), "write one-shot artifacts");
}

} // namespace metric
