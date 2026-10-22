// MNIST training/evaluation for dense (or custom) classifiers: a port of
// tflow's common/ml/mnist.h (RunMnist) and the parts of common/ml/
// classification.h it uses.
//
// The official 60,000 training rows are split once with seed 2026 (the same
// std::mt19937_64 shuffle as tflow, so the split is identical): 55,000 for
// fitting, 5,000 for validation. The official 10,000 test rows are evaluated
// once, after selecting the smallest validation cross entropy. No test labels
// influence updates or checkpoint selection. Pixels map from uint8 to [0,1].
// Every row is used, including short final batches; metrics weight batch
// means by row count. Mini-batches come from nn::BatchOrder(55000, batch, 29),
// which reproduces tflow's batch order exactly.
//
// The model is Adam on mean sparse cross-entropy plus 0.5 * l2 * sum(W^2)
// over parameters of rank >= 2 (tflow Classifier, l2 = 1e-4). Parameter
// initialisation uses datamunge's generator (seed 17), so trajectories match
// tflow only statistically. Checkpoints are .npz files of the parameter
// store (nn::ParameterStore::save/load) instead of TF SaveV2 bundles.
// A custom LogitsBuilder (e.g. a CNN) receives flattened [batch, 784] pixels
// and reshapes them itself, as tflow's architecture lambdas do. threads=N
// sets nn's intra-op thread count (default: all cores).
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
#include <vector>

#include "common/checks.hpp"
#include "common/paths.hpp"

namespace mnist {

namespace nn = datamunge::nn;

/// Builds [batch, 10] logits from flattened [batch, 784] pixels.
using LogitsBuilder = std::function<nn::Tensor(nn::ParameterStore&, const nn::Tensor& x, bool training)>;

struct Metrics {
  nn::WeightedMean    loss, accuracy;
  nn::ConfusionMatrix confusion{10};
};

/// Cross-entropy, accuracy and confusion of fixed parameters over `rows`, in batches of 512.
inline Metrics Evaluate(nn::ParameterStore& store, const LogitsBuilder& logits_of, const nn::Tensor& images,
                        const nn::Tensor& labels, const std::vector<std::int64_t>& rows) {
  nn::NoGradGuard guard;
  Metrics         result;
  for (std::size_t start = 0; start < rows.size(); start += 512) {
    const std::size_t               end = std::min(rows.size(), start + 512);
    const std::vector<std::int64_t> ids(rows.begin() + static_cast<std::ptrdiff_t>(start),
                                        rows.begin() + static_cast<std::ptrdiff_t>(end));
    const nn::Tensor x = nn::take_rows(images, ids), y = nn::take_rows(labels, ids);
    const nn::Tensor logits     = logits_of(store, x, false);
    const nn::Tensor prediction = nn::argmax(logits, 1);
    result.loss.add(nn::sparse_softmax_cross_entropy(logits, y).item(), end - start);
    result.accuracy.add(nn::accuracy(prediction, y), end - start);
    result.confusion.add(y, prediction);
  }
  return result;
}

/// Parses key=value options, trains, selects the best validation checkpoint
/// and evaluates the test set. Returns the process exit code.
inline int Run(int argc, char** argv, const std::string& name, const std::vector<std::int64_t>& hidden,
               int default_epochs, double default_rate, double minimum_test_accuracy, LogitsBuilder builder = {}) {
  try {
    std::filesystem::path dir = paths::Dataset("mnist");
    int                   epochs = default_epochs, batch = 256;
    double                rate = default_rate;
    for (int i = 1; i < argc; ++i) {
      const std::string arg   = argv[i];
      const auto        equal = arg.find('=');
      if (equal == std::string::npos) throw std::invalid_argument("expected key=value argument");
      const auto key = arg.substr(0, equal), value = arg.substr(equal + 1);
      if (key == "data")
        dir = value;
      else if (key == "epochs")
        epochs = std::stoi(value);
      else if (key == "batch")
        batch = std::stoi(value);
      else if (key == "lr")
        rate = std::stod(value);
      else if (key == "threads")
        nn::set_num_threads(std::stoi(value));
      else
        throw std::invalid_argument("unknown argument: " + key);
    }
    check::Require(epochs > 0 && batch > 0 && std::isfinite(rate) && rate > 0,
                   "positive epochs, batch and learning rate required");
    check::Require(std::filesystem::exists(dir / "train-images-idx3-ubyte") ||
                       std::filesystem::exists(dir / "train-images-idx3-ubyte.gz"),
                   "MNIST missing in " + dir.string() +
                       "; run python3 scripts/fetch_datasets.py mnist --split both in tflow_examples, or set data=");
    const auto train = nn::load_mnist(dir.string(), "train");
    const auto test  = nn::load_mnist(dir.string(), "test");
    check::Require(train.images.shape() == nn::Shape({60000, 28, 28, 1}) &&
                       test.images.shape() == nn::Shape({10000, 28, 28, 1}),
                   "official MNIST dimensions required");
    const nn::Tensor train_x = nn::reshape(train.images, {60000, 784}), test_x = nn::reshape(test.images, {10000, 784});

    std::vector<std::size_t> order(60000);
    std::iota(order.begin(), order.end(), 0);
    std::mt19937_64 split_rng(2026);
    std::shuffle(order.begin(), order.end(), split_rng);
    const std::vector<std::int64_t> fit_rows(order.begin(), order.begin() + 55000),
        validation_rows(order.begin() + 55000, order.end());
    std::vector<std::int64_t> test_rows(10000);
    std::iota(test_rows.begin(), test_rows.end(), 0);

    if (!builder)
      builder = [hidden](nn::ParameterStore& store, const nn::Tensor& x, bool) {
        return nn::dense(store, "output", nn::mlp(store, "hidden", x, hidden), 10);
      };
    const double       l2 = 1e-4;
    nn::ParameterStore store(17);
    {
      nn::NoGradGuard guard;  // create every parameter before building the optimizer
      builder(store, nn::slice_axis(train_x, 0, 0, 1), true);
    }
    nn::OptimizerConfig config;
    config.kind = nn::OptimizerKind::Adam;
    nn::Optimizer optimizer(store, rate, config);
    std::vector<nn::Tensor> weights;
    for (const auto& p : store.parameters())
      if (p.dim() >= 2) weights.push_back(p);

    const auto model_dir  = paths::OutputDir("models/" + name);
    const auto checkpoint = (model_dir / "best.npz").string();
    std::ofstream history(model_dir / "history.tsv");
    check::Require(bool(history), "open training history");
    history << "epoch\tvalidation_loss\tvalidation_accuracy\n";
    std::cout << "== " << name << ": 55000 fit / 5000 validation / 10000 test\n"
              << "  parameters=" << store.count() << "; batch=" << batch << "; lr=" << rate << '\n';

    nn::BatchOrder batches(fit_rows.size(), static_cast<std::size_t>(batch), 29);
    double         best       = std::numeric_limits<double>::infinity();
    int            best_epoch = -1;
    const auto     started    = std::chrono::steady_clock::now();
    for (int epoch = 1; epoch <= epochs; ++epoch) {
      const auto epoch_start = std::chrono::steady_clock::now();
      batches.begin_epoch();
      std::size_t samples = 0;
      for (;;) {
        auto ids = batches.next();
        if (ids.empty()) break;
        samples += ids.size();
        for (auto& index : ids) index = fit_rows[static_cast<std::size_t>(index)];
        const nn::Tensor x = nn::take_rows(train_x, ids), y = nn::take_rows(train.labels, ids);
        nn::Tensor objective = nn::sparse_softmax_cross_entropy(builder(store, x, true), y);
        if (l2 > 0) {
          nn::Tensor penalty = nn::sum(nn::square(weights[0]));
          for (std::size_t i = 1; i < weights.size(); ++i) penalty = penalty + nn::sum(nn::square(weights[i]));
          objective = objective + 0.5 * l2 * penalty;
        }
        optimizer.minimize(objective);
      }
      check::Require(samples == 55000, "each training row used once per epoch");
      const auto   validation = Evaluate(store, builder, train_x, train.labels, validation_rows);
      const double loss = validation.loss.value(), accuracy = validation.accuracy.value();
      if (loss < best) {
        best       = loss;
        best_epoch = epoch;
        store.save(checkpoint);
      }
      history << epoch << '\t' << loss << '\t' << accuracy << '\n';
      const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - epoch_start).count();
      std::cout << "  epoch " << epoch << ": validation NLL=" << loss << ", accuracy=" << accuracy << " ("
                << std::setprecision(3) << seconds << std::setprecision(6) << " s)" << std::endl;
    }
    const double train_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    store.load(checkpoint);
    const auto restored = Evaluate(store, builder, train_x, train.labels, validation_rows);
    check::Near(restored.loss.value(), best, 1e-7, "best checkpoint matches selected validation loss");
    const auto result = Evaluate(store, builder, test_x, test.labels, test_rows);
    std::cout << "  selected epoch=" << best_epoch << "; test NLL=" << result.loss.value()
              << "; accuracy=" << result.accuracy.value() << "\n  test confusion (truth rows):\n";
    std::uint64_t total = 0, correct = 0;
    const auto&   cells = result.confusion.cells();
    for (std::size_t i = 0; i < 10; ++i) {
      std::cout << "   ";
      for (std::size_t j = 0; j < 10; ++j) {
        std::cout << ' ' << std::setw(4) << cells[i][j];
        total += cells[i][j];
      }
      correct += cells[i][i];
      std::cout << '\n';
    }
    check::Require(total == 10000, "all test images evaluated");
    check::Near(result.accuracy.value(), correct / 10000., 1e-7, "confusion matches accuracy");
    check::Require(result.accuracy.value() >= minimum_test_accuracy,
                   "classifier did not reach demonstration accuracy threshold");
    check::Require(bool(history), "write training history");
    std::printf("  training time %.1f s (%.2f s/epoch)\n", train_seconds, train_seconds / epochs);
    std::cout << "  best checkpoint and history: " << model_dir.string() << "\nAll " << name << " checks passed.\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}

}  // namespace mnist
