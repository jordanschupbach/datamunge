// Image-classifier training with per-image random crop/flip augmentation: a
// port of tflow's common/ml/image_training.h (ImageBatch, EvaluateImages,
// ImageTrainingConfig, TrainImages) and common/ml/cifar10.h (RunCifar10).
//
// Images stay NHWC (tflow flattens them and each architecture reshapes back).
// Pixels map from [0,1] (nn's loaders; tflow's uint8 / 255) to [-1,1].
// Training can zero-pad by four pixels, independently crop back to the
// original size and flip horizontally (nn::augment::random_pad_crop, which
// draws row shift, column shift and coin per image like tflow's ImageBatch,
// from nn's Generator(151) instead of std::mt19937(151)). Evaluation only
// normalizes. Split development rows once (std::mt19937_64(2026), identical to
// tflow), fit on training rows in nn::BatchOrder(fit, batch, 29) order, select
// by validation NLL, restore that model, and evaluate the full test set once.
//
// The model is Adam on mean sparse cross-entropy (optionally label-smoothed
// in training) plus 0.5 * l2 * sum(W^2) over parameters of rank >= 2 (tflow
// Classifier). The builder gets a `training` flag; layers with state (batch
// norm) keep it in the separate `state` store, and checkpoints save both
// stores (best_params.npz, best_state.npz). An optional TrainingBuilder adds
// weighted auxiliary heads to the training objective (tflow's
// ClassifierNetwork::auxiliary, e.g. GoogLeNet/Inception); TrainingObjective
// exposes that objective for unit checks.
#pragma once

#include "common/dataset_assets.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <datamunge/nn/nn.hpp>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "common/checks.hpp"
#include "common/paths.hpp"

namespace image_training {

namespace nn = datamunge::nn;

/// Builds [batch, classes] logits from NHWC images in [-1,1]. `state` holds
/// non-trained model state (batch-norm moving statistics).
using LogitsBuilder =
    std::function<nn::Tensor(nn::ParameterStore& params, nn::ParameterStore& state, const nn::Tensor& x, bool training)>;

/// Normalized NHWC float images and int64 labels for `rows`; with a generator,
/// pad-4 random crops and random horizontal flips per image.
inline std::pair<nn::Tensor, nn::Tensor> ImageBatch(const nn::ImageDataset& dataset,
                                                    const std::vector<std::int64_t>& rows,
                                                    nn::Generator* augmentation = nullptr) {
  if (dataset.images.dim() != 4 || dataset.images.numel() == 0)
    throw std::invalid_argument("image batching requires nonempty NHWC data");
  for (auto row : rows)
    if (row < 0 || row >= dataset.images.shape()[0]) throw std::out_of_range("image batch row");
  nn::Tensor x = nn::take_rows(dataset.images, rows);
  if (augmentation) x = nn::augment::random_pad_crop(x, 4, *augmentation, true, 0.0);
  return {x * 2.0 - 1.0, nn::take_rows(dataset.labels, rows)};
}

struct ImageMetrics {
  nn::WeightedMean    loss, accuracy;
  nn::ConfusionMatrix confusion{10};
};

/// Cross-entropy, accuracy and confusion of fixed (inference-mode) parameters over `rows`, in batches of 256.
inline ImageMetrics EvaluateImages(nn::ParameterStore& params, nn::ParameterStore& state, const LogitsBuilder& model,
                                   const nn::ImageDataset& dataset, const std::vector<std::int64_t>& rows,
                                   int classes) {
  nn::NoGradGuard guard;
  ImageMetrics    result;
  result.confusion = nn::ConfusionMatrix(static_cast<std::size_t>(classes));
  for (std::size_t start = 0; start < rows.size(); start += 256) {
    const std::size_t end = std::min(rows.size(), start + 256);
    const auto [x, y]     = ImageBatch(dataset, std::vector<std::int64_t>(rows.begin() + static_cast<std::ptrdiff_t>(start),
                                                                          rows.begin() + static_cast<std::ptrdiff_t>(end)));
    const nn::Tensor logits     = model(params, state, x, false);
    const nn::Tensor prediction = nn::argmax(logits, 1);
    result.loss.add(nn::sparse_softmax_cross_entropy(logits, y).item(), end - start);
    result.accuracy.add(nn::accuracy(prediction, y), end - start);
    result.confusion.add(y, prediction);
  }
  return result;
}

struct ImageTrainingConfig {
  std::string name;
  int         classes = 10, epochs = 10, batch = 128, validation_rows = 5000;
  int         maximum_fit_rows = 0;  // zero uses every non-validation development row
  double      learning_rate = .001, l2 = .0001;
  double      label_smoothing = 0.;
  bool        augment = true;
  double      minimum_test_accuracy = .55;
  /// Seed of the parameter and state stores (tflow's runs used 17). The
  /// split, batch order and augmentation stream stay fixed, as in tflow.
  std::uint64_t seed = 17;
};

/// Mean sparse cross-entropy with optional label smoothing (tflow Classifier's training loss).
inline nn::Tensor TrainingCrossEntropy(const nn::Tensor& logits, const nn::Tensor& labels, int classes,
                                       double smoothing) {
  if (smoothing <= 0) return nn::sparse_softmax_cross_entropy(logits, labels);
  const nn::Tensor targets = nn::one_hot(labels, classes, 1.0, 0.0, logits.dtype()) * (1 - smoothing) + smoothing / classes;
  return -nn::mean(nn::sum(targets * nn::log_softmax(logits, 1), {1}));
}

/// Training-mode outputs with weighted auxiliary class logits (tflow
/// ClassifierNetwork's `training` and `auxiliary`, e.g. GoogLeNet's heads).
struct TrainingNetwork {
  nn::Tensor                                 logits;
  std::vector<std::pair<nn::Tensor, double>> auxiliary = {};
  /// Extra scalar terms added to the objective as they are (tflow
  /// ClassifierNetwork::penalties, e.g. CondenseNet's group lasso).
  std::vector<nn::Tensor> penalties = {};
  /// Optional training loss from the int64 labels, replacing the
  /// cross-entropy of `logits` (tflow ClassifierNetwork::loss, e.g. a capsule
  /// margin loss with reconstruction). Auxiliary heads, penalties and L2 still apply.
  std::function<nn::Tensor(const nn::Tensor& labels)> loss = {};
};
/// Optional training-path builder. When given, it replaces model(..., true)
/// in the objective; evaluation and checkpoint selection still use `model`.
using TrainingBuilder =
    std::function<TrainingNetwork(nn::ParameterStore& params, nn::ParameterStore& state, const nn::Tensor& x)>;

/// sum(W^2) over the parameters of rank >= 2, in creation order (undefined when there are none).
inline nn::Tensor L2Penalty(const nn::ParameterStore& params) {
  nn::Tensor penalty;
  for (const auto& p : params.parameters())
    if (p.dim() >= 2) penalty = penalty.defined() ? penalty + nn::sum(nn::square(p)) : nn::sum(nn::square(p));
  return penalty;
}

/// objective + 0.5 * l2 * L2Penalty(params) (the objective itself when l2 <= 0 or nothing is penalized).
inline nn::Tensor WithL2(const nn::Tensor& objective, const nn::ParameterStore& params, double l2) {
  if (l2 <= 0) return objective;
  const nn::Tensor penalty = L2Penalty(params);
  return penalty.defined() ? objective + 0.5 * l2 * penalty : objective;
}

/// tflow Classifier's training objective: (smoothed) cross-entropy of the
/// training logits, plus weight * (smoothed) cross-entropy of every auxiliary
/// head, plus 0.5 * l2 * sum(W^2) over parameters of rank >= 2.
inline nn::Tensor TrainingObjective(nn::ParameterStore& params, nn::ParameterStore& state, const LogitsBuilder& model,
                                    const nn::Tensor& x, const nn::Tensor& y, const ImageTrainingConfig& cfg,
                                    const TrainingBuilder& training = {}) {
  TrainingNetwork network = training ? training(params, state, x) : TrainingNetwork{model(params, state, x, true)};
  nn::Tensor objective =
      network.loss ? network.loss(y) : TrainingCrossEntropy(network.logits, y, cfg.classes, cfg.label_smoothing);
  for (const auto& [logits, weight] : network.auxiliary) {
    if (!std::isfinite(weight) || weight <= 0) throw std::invalid_argument("auxiliary loss weight must be positive and finite");
    objective = objective + weight * TrainingCrossEntropy(logits, y, cfg.classes, cfg.label_smoothing);
  }
  for (const auto& penalty : network.penalties) objective = objective + penalty;
  return WithL2(objective, params, cfg.l2);
}

/// Wall-clock seconds per training epoch of the last TrainImages run (for reports).
inline std::vector<double>& EpochSeconds() {
  static std::vector<double> seconds;
  return seconds;
}

/// Parses key=value options shared by the CIFAR-10 runners; returns false for an unknown key.
inline bool ParseCifarOption(const std::string& key, const std::string& value, std::filesystem::path& directory,
                             ImageTrainingConfig& cfg) {
  if (key == "data")
    directory = value;
  else if (key == "epochs")
    cfg.epochs = std::stoi(value);
  else if (key == "batch")
    cfg.batch = std::stoi(value);
  else if (key == "lr")
    cfg.learning_rate = std::stod(value);
  else if (key == "fit")
    cfg.maximum_fit_rows = std::stoi(value);
  else if (key == "augment" && (value == "0" || value == "1"))
    cfg.augment = value == "1";
  else if (key == "threads")
    nn::set_num_threads(std::stoi(value));
  else if (key == "seed")
    cfg.seed = std::stoull(value);
  else
    return false;
  return true;
}

/// The five official development batches and the separate test batch.
inline std::pair<nn::ImageDataset, nn::ImageDataset> LoadCifar10(const std::filesystem::path& directory) {
  auto development = dataset_assets::LoadCifar10(directory.string(), "train");
  auto test        = dataset_assets::LoadCifar10(directory.string(), "test");
  check::Require(development.images.shape() == nn::Shape({50000, 32, 32, 3}) &&
                     test.images.shape() == nn::Shape({10000, 32, 32, 3}),
                 "official CIFAR-10 split sizes");
  return {std::move(development), std::move(test)};
}

}  // namespace image_training
