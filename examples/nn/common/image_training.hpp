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
// stores (best_params.npz, best_state.npz).
#pragma once

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
};

/// Mean sparse cross-entropy with optional label smoothing (tflow Classifier's training loss).
inline nn::Tensor TrainingCrossEntropy(const nn::Tensor& logits, const nn::Tensor& labels, int classes,
                                       double smoothing) {
  if (smoothing <= 0) return nn::sparse_softmax_cross_entropy(logits, labels);
  const nn::Tensor targets = nn::one_hot(labels, classes, 1.0, 0.0, logits.dtype()) * (1 - smoothing) + smoothing / classes;
  return -nn::mean(nn::sum(targets * nn::log_softmax(logits, 1), {1}));
}

/// Wall-clock seconds per training epoch of the last TrainImages run (for reports).
inline std::vector<double>& EpochSeconds() {
  static std::vector<double> seconds;
  return seconds;
}

inline void TrainImages(const nn::ImageDataset& development, const nn::ImageDataset& test,
                        const LogitsBuilder& model, const ImageTrainingConfig& cfg) {
  const std::int64_t development_rows = development.images.shape()[0], test_size = test.images.shape()[0];
  check::Require(development.images.dim() == 4 && test.images.dim() == 4 && cfg.validation_rows > 0 &&
                     cfg.validation_rows < development_rows && cfg.epochs > 0 && cfg.batch > 0 && cfg.classes > 1,
                 "valid image training dimensions/config");
  for (int d = 1; d < 4; ++d)
    check::Require(development.images.shape()[d] == test.images.shape()[d], "training/test image dimensions match");
  const std::size_t available = static_cast<std::size_t>(development_rows - cfg.validation_rows);
  check::Require(cfg.maximum_fit_rows >= 0 && static_cast<std::size_t>(cfg.maximum_fit_rows) <= available,
                 "valid optional fitting-row limit");
  const std::size_t        fit_count = cfg.maximum_fit_rows ? static_cast<std::size_t>(cfg.maximum_fit_rows) : available;
  std::vector<std::size_t> order(static_cast<std::size_t>(development_rows));
  std::iota(order.begin(), order.end(), 0);
  std::mt19937_64 split_rng(2026);
  std::shuffle(order.begin(), order.end(), split_rng);
  // Keep the validation split fixed when limiting the fitting subset.
  const std::vector<std::int64_t> fit(order.begin(), order.begin() + static_cast<std::ptrdiff_t>(fit_count)),
      validation(order.begin() + static_cast<std::ptrdiff_t>(available), order.end());
  std::vector<std::int64_t> test_rows(static_cast<std::size_t>(test_size));
  std::iota(test_rows.begin(), test_rows.end(), 0);

  nn::ParameterStore params(17), state(17);
  {
    nn::NoGradGuard guard;  // create every parameter (and state entry) before building the optimizer
    model(params, state, ImageBatch(development, {fit[0], fit[1]}).first, false);
  }
  nn::OptimizerConfig config;
  config.kind = nn::OptimizerKind::Adam;
  nn::Optimizer           optimizer(params, cfg.learning_rate, config);
  std::vector<nn::Tensor> weights;
  for (const auto& p : params.parameters())
    if (p.dim() >= 2) weights.push_back(p);

  const auto directory = paths::OutputDir("models/" + cfg.name);
  const auto params_path = (directory / "best_params.npz").string(), state_path = (directory / "best_state.npz").string();
  std::ofstream history(directory / "history.tsv");
  check::Require(bool(history), "open image training history");
  history << "epoch\tvalidation_loss\tvalidation_accuracy\n";
  std::cout << "== " << cfg.name << ": " << fit_count << " fit / " << validation.size() << " validation / " << test_size
            << " test\n  parameters=" << params.count() << "; augmentation=" << cfg.augment << "; batch=" << cfg.batch
            << "; lr=" << cfg.learning_rate << "; threads=" << nn::get_num_threads() << std::endl;
  nn::BatchOrder batches(fit_count, static_cast<std::size_t>(cfg.batch), 29);
  nn::Generator  augmentation(151);
  double         best = std::numeric_limits<double>::infinity();
  int            best_epoch = -1;
  EpochSeconds().clear();
  for (int epoch = 1; epoch <= cfg.epochs; ++epoch) {
    const auto epoch_start = std::chrono::steady_clock::now();
    batches.begin_epoch();
    std::size_t used = 0;
    for (;;) {
      auto rows = batches.next();
      if (rows.empty()) break;
      used += rows.size();
      for (auto& row : rows) row = fit[static_cast<std::size_t>(row)];
      const auto [x, y] = ImageBatch(development, rows, cfg.augment ? &augmentation : nullptr);
      nn::Tensor objective = TrainingCrossEntropy(model(params, state, x, true), y, cfg.classes, cfg.label_smoothing);
      if (cfg.l2 > 0 && !weights.empty()) {
        nn::Tensor penalty = nn::sum(nn::square(weights[0]));
        for (std::size_t i = 1; i < weights.size(); ++i) penalty = penalty + nn::sum(nn::square(weights[i]));
        objective = objective + 0.5 * cfg.l2 * penalty;
      }
      optimizer.minimize(objective);
    }
    check::Require(used == fit_count, "each fit image used once per epoch");
    const double train_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - epoch_start).count();
    const auto   result        = EvaluateImages(params, state, model, development, validation, cfg.classes);
    const double loss = result.loss.value(), accuracy = result.accuracy.value();
    if (loss < best) {
      best       = loss;
      best_epoch = epoch;
      params.save(params_path);
      if (state.size()) state.save(state_path);
    }
    history << epoch << '\t' << loss << '\t' << accuracy << '\n';
    history.flush();
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - epoch_start).count();
    EpochSeconds().push_back(seconds);
    std::cout << "  epoch " << epoch << ": validation NLL=" << loss << ", accuracy=" << accuracy << " ("
              << std::setprecision(3) << train_seconds << " s train, " << seconds << " s total)" << std::setprecision(6)
              << std::endl;
  }
  params.load(params_path);
  if (state.size()) state.load(state_path);
  check::Near(EvaluateImages(params, state, model, development, validation, cfg.classes).loss.value(), best, 1e-6,
              "best image checkpoint validation loss");
  const auto result = EvaluateImages(params, state, model, test, test_rows, cfg.classes);
  std::cout << "  selected epoch=" << best_epoch << "; test NLL=" << result.loss.value()
            << "; accuracy=" << result.accuracy.value() << "\n  confusion (truth rows):\n";
  std::uint64_t total = 0, correct = 0;
  const auto&   cells = result.confusion.cells();
  for (int i = 0; i < cfg.classes; ++i) {
    std::cout << "   ";
    for (int j = 0; j < cfg.classes; ++j) {
      std::cout << ' ' << std::setw(4) << cells[i][j];
      total += cells[i][j];
    }
    correct += cells[i][i];
    std::cout << '\n';
  }
  check::Require(total == static_cast<std::uint64_t>(test_size), "all test images evaluated");
  check::Near(result.accuracy.value(), double(correct) / total, 1e-7, "confusion agrees with accuracy");
  check::Require(result.accuracy.value() >= cfg.minimum_test_accuracy, "image classifier demonstration accuracy threshold");
  check::Require(bool(history), "write image training history");
  double train_total = 0;
  for (double s : EpochSeconds()) train_total += s;
  std::printf("  training time %.1f s (%.2f s/epoch)\n", train_total, train_total / cfg.epochs);
  std::cout << "  best checkpoint and history: " << directory.string() << "\nAll " << cfg.name << " checks passed.\n";
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
  else
    return false;
  return true;
}

/// The five official development batches and the separate test batch.
inline std::pair<nn::ImageDataset, nn::ImageDataset> LoadCifar10(const std::filesystem::path& directory) {
  check::Require(std::filesystem::exists(directory / "data_batch_1.bin"),
                 "CIFAR-10 missing in " + directory.string() +
                     "; run python3 scripts/fetch_datasets.py cifar10 --split both in tflow_examples, or set data=");
  auto development = nn::load_cifar10(directory.string(), "train");
  auto test        = nn::load_cifar10(directory.string(), "test");
  check::Require(development.images.shape() == nn::Shape({50000, 32, 32, 3}) &&
                     test.images.shape() == nn::Shape({10000, 32, 32, 3}),
                 "official CIFAR-10 split sizes");
  return {std::move(development), std::move(test)};
}

/// Command-line runner for architecture demonstrations on CIFAR-10 (tflow
/// RunCifar10). Options: data=PATH epochs= batch= lr= fit= augment=0|1 threads=.
/// Returns the process exit code.
inline int RunCifar10(int argc, char** argv, ImageTrainingConfig cfg, const LogitsBuilder& architecture) {
  try {
    std::filesystem::path directory = paths::Dataset("cifar10");
    for (int i = 1; i < argc; ++i) {
      const std::string argument = argv[i];
      const auto        equal    = argument.find('=');
      if (equal == std::string::npos) throw std::invalid_argument("expected key=value option");
      if (!ParseCifarOption(argument.substr(0, equal), argument.substr(equal + 1), directory, cfg))
        throw std::invalid_argument("unknown option: " + argument);
    }
    check::Require(cfg.classes == 10, "CIFAR-10 runner requires ten output classes");
    const auto [development, test] = LoadCifar10(directory);
    TrainImages(development, test, architecture, cfg);
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}

}  // namespace image_training
