// CMD: just run-nn batch_norm_ablation
// Paired MNIST CNN experiments with and without batch normalization.
//
// Port of tflow's batch_norm_ablation. Use the same 12,000 fitting rows,
// 3,000 validation rows, initial convolution and dense weights, six epochs,
// batch order, Adam(.001) and kernel L2(.0001). Three initialization seeds
// (11,23,47) produce paired runs. Batch norm adds gamma/beta after each
// convolution, before ReLU, with momentum .9 and eps .001. Keep convolution
// biases in both variants to isolate normalization.
// Select checkpoints on validation NLL; evaluate the official 10k test set
// after selection. Report paired differences and sample standard deviations;
// three runs on a small MNIST training subset do not establish a general
// advantage or disadvantage for batch normalization. Do not require BN to win.
//
// Check identical starting weights, one moving-statistics update per batch,
// inference invariance to batch companions, no statistics mutation during
// evaluation, and checkpoint restoration into a fresh model including state.
// The split (std::mt19937_64(2026)) and batch orders (nn::BatchOrder) are
// identical to TF's; weights come from nn's generator per seed, so the paired
// numbers match TF only statistically. Batch norm is nn::batch_norm (moving
// statistics in a separate state store); checkpoints are .npz pairs
// seed<S>_{plain,bn}_{params,state}.npz instead of TF SaveV2 bundles.
// Learning curves, selected checkpoints and summary: output/nn/models/bn_ablation/.
// Preparation: python3 scripts/fetch_datasets.py mnist --split both (in tflow).
// Options: data=PATH epochs=6 threads=N (quick mode runs one epoch unless epochs= is given).

#include <array>
#include <chrono>
#include <cmath>
#include <datamunge/nn/nn.hpp>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>

#include "common/checks.hpp"
#include "common/mnist.hpp"
#include "common/paths.hpp"

namespace nn = datamunge::nn;
using nn::Tensor;

namespace {

nn::BatchNormOptions BnOptions() {
  nn::BatchNormOptions o;
  o.momentum = .9;
  o.eps      = .001;
  return o;
}

/// The paired CNN: conv(16)-[BN]-ReLU-pool, conv(32)-[BN]-ReLU-pool, Dense(64)-ReLU, 10 logits.
Tensor Architecture(bool use_bn, nn::ParameterStore& p, nn::ParameterStore& state, const Tensor& x, bool training) {
  Tensor h = nn::reshape(x, {-1, 28, 28, 1});
  for (int block = 0; block < 2; ++block) {
    const int width = block == 0 ? 16 : 32;
    h               = nn::conv2d(p, "conv" + std::to_string(block + 1), h, width, 3);
    if (use_bn) h = nn::batch_norm(p, state, "bn" + std::to_string(block + 1), h, training, BnOptions());
    h = nn::max_pool2d(nn::relu(h), 2, 2);
  }
  h = nn::dense(p, "dense", nn::reshape(h, {-1, 7 * 7 * 32}), 64, nn::Activation::Relu);
  return nn::dense(p, "output", h, 10);
}

/// Batch mean/variance (unbiased, as TF FusedBatchNorm reports) that a
/// training step on `x` feeds into each moving average, from the current weights.
std::vector<Tensor> Moments(nn::ParameterStore& p, const Tensor& x) {
  nn::NoGradGuard     guard;
  std::vector<Tensor> moments;
  Tensor              h = nn::reshape(x, {-1, 28, 28, 1});
  for (int block = 0; block < 2; ++block) {
    const std::string i = std::to_string(block + 1);
    h                   = nn::conv2d(p, "conv" + i, h, block == 0 ? 16 : 32, 3);
    auto r = nn::batch_norm_train(h, p.at("bn" + i + "/gamma"), p.at("bn" + i + "/beta"), BnOptions().eps);
    moments.push_back(r.mean);
    moments.push_back(r.variance);
    h = nn::max_pool2d(nn::relu(r.y), 2, 2);
  }
  return moments;
}

struct Model {
  bool                           use_bn;
  nn::ParameterStore             params, state;
  std::unique_ptr<nn::Optimizer> optimizer;
  std::vector<Tensor>            weights;
  Model(bool bn, std::uint64_t seed) : use_bn(bn), params(seed), state(0) {
    {
      nn::NoGradGuard guard;
      Architecture(use_bn, params, state, nn::zeros({1, 784}), false);
    }
    nn::OptimizerConfig config;
    config.kind = nn::OptimizerKind::Adam;
    optimizer   = std::make_unique<nn::Optimizer>(params, .001, config);
    for (const auto& w : params.parameters())
      if (w.dim() >= 2) weights.push_back(w);
  }
  mnist::LogitsBuilder Builder() {
    return [this](nn::ParameterStore& p, const Tensor& x, bool training) {
      return Architecture(use_bn, p, state, x, training);
    };
  }
  Tensor Probabilities(const Tensor& x) {
    nn::NoGradGuard guard;
    return nn::softmax(Architecture(use_bn, params, state, x, false));
  }
  void Save(const std::string& prefix) const {
    params.save(prefix + "_params.npz");
    if (use_bn) state.save(prefix + "_state.npz");
  }
  void Load(const std::string& prefix) {
    params.load(prefix + "_params.npz");
    if (use_bn) state.load(prefix + "_state.npz");
  }
};

std::vector<Tensor> Snapshot(const nn::ParameterStore& store) {
  std::vector<Tensor> result;
  for (const auto& t : store.parameters()) result.push_back(t.detach().clone());
  return result;
}

void Same(const Tensor& a, const Tensor& b, double tolerance, const std::string& message) {
  check::Require(a.shape() == b.shape(), message + " shape");
  const auto x = a.contiguous().to_vector<float>(), y = b.contiguous().to_vector<float>();
  for (std::size_t i = 0; i < x.size(); ++i) check::Near(x[i], y[i], tolerance, message);
}

struct Trial {
  double accuracy, nll, seconds;
  int    epoch;
};

}  // namespace

int main(int argc, char** argv) {
  try {
    std::filesystem::path dir    = paths::Dataset("mnist");
    int                   epochs = 0;  // zero: six epochs (one in quick mode)
    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      const auto        eq  = arg.find('=');
      if (eq == std::string::npos) throw std::invalid_argument("expected key=value argument");
      const auto key = arg.substr(0, eq), value = arg.substr(eq + 1);
      if (key == "data")
        dir = value;
      else if (key == "epochs")
        epochs = std::stoi(value);
      else if (key == "threads")
        nn::set_num_threads(std::stoi(value));
      else
        throw std::invalid_argument("unknown argument: " + key);
    }
    check::Require(epochs >= 0, "positive epochs required");
    if (!epochs) epochs = example_runtime::Limit(6, 1);
    const auto training = nn::load_mnist(dir.string(), "train");
    const auto test     = nn::load_mnist(dir.string(), "test");
    check::Require(training.images.size(0) == 60000 && test.images.size(0) == 10000, "official MNIST row counts");
    const Tensor train_x = nn::reshape(training.images, {60000, 784}), test_x = nn::reshape(test.images, {10000, 784});
    std::vector<std::size_t>  order(60000);
    std::vector<std::int64_t> test_rows(example_runtime::Limit<std::size_t>(10000, 32));
    std::iota(order.begin(), order.end(), 0);
    std::iota(test_rows.begin(), test_rows.end(), 0);
    std::mt19937_64 split(2026);
    std::shuffle(order.begin(), order.end(), split);
    std::vector<std::int64_t> fit(order.begin(), order.begin() + 12000),
        validation(order.begin() + 12000, order.begin() + 15000);
    fit.resize(example_runtime::Limit<std::size_t>(fit.size(), 64));
    validation.resize(example_runtime::Limit<std::size_t>(validation.size(), 32));
    example_runtime::Describe();
    const auto    directory = paths::OutputDir("models/bn_ablation");
    std::ofstream curves(directory / "history.tsv"), summary(directory / "summary.tsv");
    curves << "seed\tbatch_norm\tepoch\tvalidation_nll\tvalidation_accuracy\ttrain_seconds\n";
    summary << "seed\tbatch_norm\tselected_epoch\ttest_nll\ttest_accuracy\ttrain_seconds\n";
    std::array<std::vector<Trial>, 2> trials;
    const auto                        started = std::chrono::steady_clock::now();
    for (int seed : {11, 23, 47}) {
      std::array<std::unique_ptr<Model>, 2> models = {std::make_unique<Model>(false, seed),
                                                      std::make_unique<Model>(true, seed)};
      for (const auto& name : models[0]->params.names())
        Same(models[0]->params.at(name), models[1]->params.at(name), 0, "identical paired initialization");
      std::cout << "== seed " << seed << ": parameters " << models[0]->params.count() << " / "
                << models[1]->params.count() << "; identical starting weights\n";
      std::array<double, 2>      best = {INFINITY, INFINITY}, seconds{};
      std::array<int, 2>         selected{};
      std::array<std::string, 2> checkpoints;
      for (int variant = 0; variant < 2; ++variant)
        checkpoints[variant] = (directory / ("seed" + std::to_string(seed) + (variant ? "_bn" : "_plain"))).string();
      nn::BatchOrder batches(fit.size(), example_runtime::Limit(128, 16), static_cast<std::uint64_t>(seed + 1000));
      bool           first_update = true;
      for (int epoch = 1; epoch <= epochs; ++epoch) {
        batches.begin_epoch();
        std::size_t samples = 0;
        for (;;) {
          auto rows = batches.next();
          if (rows.empty()) break;
          samples += rows.size();
          for (auto& row : rows) row = fit[static_cast<std::size_t>(row)];
          const Tensor x = nn::take_rows(train_x, rows), y = nn::take_rows(training.labels, rows);
          std::vector<Tensor> old_state, moments;
          if (first_update) {
            old_state = Snapshot(models[1]->state);
            moments   = Moments(models[1]->params, x);
          }
          for (int variant = 0; variant < 2; ++variant) {
            // One explicit training step: forward, mean cross-entropy plus
            // 0.5 * l2 * sum(W^2) over the kernels, then the Adam update.
            const auto start = std::chrono::steady_clock::now();
            auto&      model = *models[variant];
            Tensor     objective =
                nn::sparse_softmax_cross_entropy(Architecture(model.use_bn, model.params, model.state, x, true), y);
            Tensor penalty = nn::sum(nn::square(model.weights[0]));
            for (std::size_t i = 1; i < model.weights.size(); ++i) penalty = penalty + nn::sum(nn::square(model.weights[i]));
            model.optimizer->minimize(objective + 0.5 * 1e-4 * penalty);
            seconds[variant] += std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
          }
          if (first_update) {
            const auto updated = Snapshot(models[1]->state);
            check::Require(updated.size() == moments.size(), "one mean/variance pair per batch norm");
            for (std::size_t i = 0; i < updated.size(); ++i) {
              const auto u = updated[i].to_vector<float>(), o = old_state[i].to_vector<float>(),
                         m = moments[i].to_vector<float>();
              for (std::size_t c = 0; c < u.size(); ++c)
                check::Near(u[c], .9 * o[c] + .1 * m[c], 2e-6,
                            "exactly one moving-statistics update from pre-update activations");
            }
            first_update = false;
          }
        }
        check::Require(samples == fit.size(), "all fitting rows used once per epoch");
        for (int variant = 0; variant < 2; ++variant) {
          auto&        model   = *models[variant];
          const auto   metrics = mnist::Evaluate(model.params, model.Builder(), train_x, training.labels, validation);
          const double nll = metrics.loss.value(), accuracy = metrics.accuracy.value();
          if (nll < best[variant]) {
            best[variant]     = nll;
            selected[variant] = epoch;
            model.Save(checkpoints[variant]);
          }
          curves << seed << '\t' << variant << '\t' << epoch << '\t' << nll << '\t' << accuracy << '\t'
                 << seconds[variant] << '\n'
                 << std::flush;
          std::cout << "  epoch " << epoch << (variant ? " BN: " : " plain: ") << "validation NLL=" << nll
                    << ", accuracy=" << accuracy << std::endl;
        }
      }
      for (int variant = 0; variant < 2; ++variant) {
        auto& model = *models[variant];
        model.Load(checkpoints[variant]);
        const auto validation_result =
            mnist::Evaluate(model.params, model.Builder(), train_x, training.labels, validation);
        check::Near(validation_result.loss.value(), best[variant], 1e-7, "restored validation loss");
        const auto result = mnist::Evaluate(model.params, model.Builder(), test_x, test.labels, test_rows);
        check::TrainingQuality(result.accuracy.value() >= .95, "both models learn the classification task");
        trials[variant].push_back({result.accuracy.value(), result.loss.value(), seconds[variant], selected[variant]});
        summary << seed << '\t' << variant << '\t' << selected[variant] << '\t' << result.loss.value() << '\t'
                << result.accuracy.value() << '\t' << seconds[variant] << '\n'
                << std::flush;
        std::cout << "  selected " << (variant ? "BN" : "plain") << " epoch=" << selected[variant]
                  << "; test NLL=" << result.loss.value() << ", accuracy=" << result.accuracy.value()
                  << "\n  " << (variant ? "BN" : "plain") << " test confusion (truth rows):\n";
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
        check::Require(total == test_rows.size(), "all selected test images evaluated");
        check::Near(result.accuracy.value(), double(correct) / total, 1e-7, "confusion matches accuracy");
      }
      auto&        bn                = *models[1];
      const auto   before_evaluation = Snapshot(bn.state);
      const Tensor batch_x = nn::take_rows(test_x, {0, 1, 2, 3, 4, 5, 6, 7}), single_x = nn::take_rows(test_x, {0});
      const Tensor batch_result = bn.Probabilities(batch_x), single_result = bn.Probabilities(single_x);
      for (int c = 0; c < 10; ++c)
        check::Near(batch_result.at({0, c}), single_result.at({0, c}), 1e-6,
                    "inference predictions do not depend on batch companions");
      const auto after_evaluation = Snapshot(bn.state);
      for (std::size_t i = 0; i < before_evaluation.size(); ++i)
        Same(before_evaluation[i], after_evaluation[i], 0, "evaluation leaves moments unchanged");
      Model fresh(true, 999);
      fresh.Load(checkpoints[1]);
      Same(fresh.Probabilities(batch_x), batch_result, 1e-7, "fresh model checkpoint predictions");
      const auto fresh_state = Snapshot(fresh.state);
      for (std::size_t i = 0; i < fresh_state.size(); ++i)
        Same(fresh_state[i], before_evaluation[i], 0, "checkpoint contains moving moments");
    }
    std::vector<double> differences;
    for (std::size_t i = 0; i < trials[0].size(); ++i) differences.push_back(trials[1][i].accuracy - trials[0][i].accuracy);
    auto describe = [](const std::vector<double>& values) {
      double mean = std::accumulate(values.begin(), values.end(), 0.) / values.size(), variance = 0;
      for (double x : values) variance += (x - mean) * (x - mean);
      return std::pair<double, double>{mean, std::sqrt(variance / (values.size() - 1))};
    };
    for (int variant = 0; variant < 2; ++variant) {
      std::vector<double> accuracy;
      for (const auto& trial : trials[variant]) accuracy.push_back(trial.accuracy);
      auto [mean, sd] = describe(accuracy);
      std::cout << "  " << (variant ? "BN" : "plain") << " test accuracy mean=" << mean << "; sample SD=" << sd << '\n';
    }
    auto [mean, sd] = describe(differences);
    std::cout << "  paired BN-plain accuracy difference mean=" << mean << "; sample SD=" << sd
              << "\n  six epochs on 12000 fitting rows; descriptive results from three paired seeds\n";
    check::Require(bool(curves) && bool(summary), "write ablation results");
    std::printf("  total time %.1f s\n",
                std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count());
    std::cout << "All batch-normalization ablation checks passed.\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
