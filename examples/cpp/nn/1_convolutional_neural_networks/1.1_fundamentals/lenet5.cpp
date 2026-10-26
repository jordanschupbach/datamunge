// CMD: just run-nn lenet5
// LeNet-5: sparse convolutions, learned subsampling and a fixed RBF head.
//
// Port of tflow's lenet5. Preparation (in ~/projects/tflow_examples):
// python3 scripts/fetch_datasets.py mnist --split both.
// Preserve the 1998 architecture's C1/S2/C3/S4/C5/F6 dimensions, 60 sparse C3
// connections, scaled tanh activations and exactly 60,000 learned parameters.
// Use our own fixed seven-segment 7x12 digit prototypes for the RBF output.
// Adam and cross entropy on negative squared distances are modern training
// adaptations; this is not a reproduction of the original training recipe.
// Reference: https://cognitivemedium.com/assets/rmnist/LeCun1998.pdf
//
// Check stage shapes/counts, C3's absent-channel derivatives and the RBF
// classifier's prototype predictions before fitting the official MNIST data.
// Fit 55k rows, select on 5k validation rows, then evaluate all 10k test rows.
// Default eight epochs, lr=.001, batch=256; require 97.5% test accuracy.
// Weights initialise from nn's generator (seed 17), so the trajectory matches
// TF only statistically. Options: data=PATH epochs=8 batch=256 lr=.001 threads=N.
// Checkpoint and validation history: output/nn/models/lenet5/.

#include <array>
#include <cmath>
#include <iostream>

#include "common/lenet.hpp"
#include "common/mnist.hpp"

namespace nn = datamunge::nn;
using nn::Tensor;

namespace {

void ArchitectureChecks() {
  nn::ParameterStore p(17);
  const Tensor       pixels  = nn::ones({2, 784});
  const auto         network = lenet::LeNet5(p, pixels);
  check::Require(p.count() == 60000, "LeNet-5 has exactly 60000 learned parameters");
  const std::array<std::string, 6> names  = {"c1", "s2", "c3", "s4", "c5", "f6"};
  const std::array<int, 6>         counts = {156, 12, 1516, 32, 48120, 10164};
  for (std::size_t stage = 0; stage < names.size(); ++stage) {
    std::int64_t count = 0;
    for (const auto& name : p.names())
      if (name.rfind(names[stage] + "/", 0) == 0)
        count += p.at(name).numel();
    check::Require(count == counts[stage], "LeNet stage parameter count: " + names[stage]);
  }
  int connections = 0;
  for (const auto& channels : lenet::LeNetC3Connections())
    connections += int(channels.size());
  check::Require(connections == 60, "C3 has 60 channel connections");
  // Map 0 reads channels 0,1,2 only. Its derivative must be exactly zero for
  // channels 3,4,5, while at least one derivative in each connected map lives.
  Tensor c3_input = nn::full({1, 14, 14, 6}, .2);
  c3_input.requires_grad_();
  auto       c3               = lenet::LeNetC3(p, "c3", c3_input);
  auto       probe            = nn::sum(nn::slice_axis(c3, 3, 0, 1));
  const auto gradient         = nn::grad(probe, {c3_input})[0].to_vector<float>();
  const auto prototype_logits = lenet::LeNetRbfLogits(lenet::LeNetDigitCodes());

  const std::vector<std::pair<Tensor, nn::Shape>> stages = {
      {network.image, {2, 32, 32, 1}},
      {network.c1, {2, 28, 28, 6}},
      {network.s2, {2, 14, 14, 6}},
      {network.c3, {2, 10, 10, 16}},
      {network.s4, {2, 5, 5, 16}},
      {network.c5, {2, 1, 1, 120}},
      {network.f6, {2, 84}},
      {network.logits, {2, 10}}};
  for (const auto& [value, shape] : stages)
    check::Require(value.shape() == shape, "LeNet stage shape");
  check::Near(network.image.at({0, 0, 0, 0}), -.1, 1e-7, "padding background normalization");
  check::Near(network.image.at({0, 2, 2, 0}), 1.175, 1e-7, "foreground normalization");
  std::array<double, 6> magnitude{};
  for (int i = 0; i < 14 * 14 * 6; ++i)
    magnitude[i % 6] += std::abs(gradient[static_cast<std::size_t>(i)]);
  for (int channel = 0; channel < 6; ++channel) {
    if (channel < 3)
      check::Require(magnitude[channel] > 0, "connected C3 channel has a gradient");
    else
      check::Near(magnitude[channel], 0, 0, "disconnected C3 channel has no gradient");
  }
  for (int a = 0; a < 10; ++a)
    for (int b = 0; b < 10; ++b) {
      const double logit = prototype_logits.at({a, b});
      if (a == b)
        check::Near(logit, 0, 0, "RBF self distance");
      else
        check::Require(logit < 0, "distinct RBF prototypes classify themselves uniquely");
    }
  std::cout << "  architecture checks: stage shapes/counts, sparse C3 derivatives and RBF "
               "prototypes passed\n";
}

} // namespace

int main(int argc, char** argv) {
  try {
    ArchitectureChecks();
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
  auto builder = [](nn::ParameterStore& p, const Tensor& x, bool) {
    return lenet::LeNet5(p, x).logits;
  };
  auto experiment_builder = builder;
  {
    // Dataset setup, batching, fitting and checkpoint selection are explicit here.
    using namespace mnist;
    const std::string               name                  = "lenet5";
    const std::vector<std::int64_t> hidden                = {};
    const int                       default_epochs        = 8;
    const double                    default_rate          = .001;
    const double                    minimum_test_accuracy = .975;
    LogitsBuilder                   builder               = experiment_builder;

    try {
      std::filesystem::path dir    = paths::Dataset("mnist");
      int                   epochs = default_epochs, batch = 256;
      double                rate = default_rate;
      bool                  explicit_epochs = false;
      for (int i = 1; i < argc; ++i) {
        const std::string arg   = argv[i];
        const auto        equal = arg.find('=');
        if (equal == std::string::npos)
          throw std::invalid_argument("expected key=value argument");
        const auto key = arg.substr(0, equal), value = arg.substr(equal + 1);
        if (key == "data")
          dir = value;
        else if (key == "epochs") {
          epochs          = std::stoi(value);
          explicit_epochs = true;
        }
        else if (key == "batch")
          batch = std::stoi(value);
        else if (key == "lr")
          rate = std::stod(value);
        else if (key == "threads")
          nn::set_num_threads(std::stoi(value));
        else
          throw std::invalid_argument("unknown argument: " + key);
      }
      check::Require(
          epochs > 0 && batch > 0 && std::isfinite(rate) && rate > 0,
          "positive epochs, batch and learning rate required");
      if (!explicit_epochs) epochs = example_runtime::Limit(epochs, 1);  // an explicit budget is honored
      batch  = example_runtime::Limit(batch, 16);
      example_runtime::Describe();
      check::Require(
          std::filesystem::exists(dir / "train-images-idx3-ubyte") ||
              std::filesystem::exists(dir / "train-images-idx3-ubyte.gz"),
          "MNIST missing in " + dir.string() +
              "; run python3 scripts/fetch_datasets.py mnist --split both in tflow_examples, or "
              "set data=");
      const auto train = nn::load_mnist(dir.string(), "train");
      const auto test  = nn::load_mnist(dir.string(), "test");
      check::Require(
          train.images.shape() == nn::Shape({60000, 28, 28, 1}) &&
              test.images.shape() == nn::Shape({10000, 28, 28, 1}),
          "official MNIST dimensions required");
      const nn::Tensor train_x = nn::reshape(train.images, {60000, 784}),
                       test_x  = nn::reshape(test.images, {10000, 784});

      std::vector<std::size_t> order(60000);
      std::iota(order.begin(), order.end(), 0);
      std::mt19937_64 split_rng(2026);
      std::shuffle(order.begin(), order.end(), split_rng);
      std::vector<std::int64_t> fit_rows(order.begin(), order.begin() + 55000),
          validation_rows(order.begin() + 55000, order.end());
      fit_rows.resize(example_runtime::Limit<std::size_t>(fit_rows.size(), 64));
      validation_rows.resize(example_runtime::Limit<std::size_t>(validation_rows.size(), 32));
      std::vector<std::int64_t> test_rows(example_runtime::Limit<std::size_t>(10000, 32));
      std::iota(test_rows.begin(), test_rows.end(), 0);

      if (!builder)
        builder = [hidden](nn::ParameterStore& store, const nn::Tensor& x, bool) {
          return nn::dense(store, "output", nn::mlp(store, "hidden", x, hidden), 10);
        };
      const double       l2 = 1e-4;
      nn::ParameterStore store(17);
      {
        nn::NoGradGuard guard; // create every parameter before building the optimizer
        builder(store, nn::slice_axis(train_x, 0, 0, 1), true);
      }
      nn::OptimizerConfig config;
      config.kind = nn::OptimizerKind::Adam;
      nn::Optimizer           optimizer(store, rate, config);
      std::vector<nn::Tensor> weights;
      for (const auto& p : store.parameters())
        if (p.dim() >= 2)
          weights.push_back(p);

      const auto    model_dir  = paths::OutputDir("models/" + name);
      const auto    checkpoint = (model_dir / "best.npz").string();
      std::ofstream history(model_dir / "history.tsv");
      check::Require(bool(history), "open training history");
      history << "epoch\tvalidation_loss\tvalidation_accuracy\n";
      std::cout << "== " << name << ": " << fit_rows.size() << " fit / " << validation_rows.size()
                << " validation / " << test_rows.size() << " test\n"
                << "  parameters=" << store.count() << "; batch=" << batch << "; lr=" << rate
                << '\n';

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
          if (ids.empty())
            break;
          samples += ids.size();
          for (auto& index : ids)
            index = fit_rows[static_cast<std::size_t>(index)];
          const nn::Tensor x = nn::take_rows(train_x, ids), y = nn::take_rows(train.labels, ids);
          nn::Tensor       objective = nn::sparse_softmax_cross_entropy(builder(store, x, true), y);
          if (l2 > 0) {
            nn::Tensor penalty = nn::sum(nn::square(weights[0]));
            for (std::size_t i = 1; i < weights.size(); ++i)
              penalty = penalty + nn::sum(nn::square(weights[i]));
            objective = objective + 0.5 * l2 * penalty;
          }
          optimizer.minimize(objective);
        }
        check::Require(
            samples == fit_rows.size(), "each selected training row used once per epoch");
        const auto   validation = Evaluate(store, builder, train_x, train.labels, validation_rows);
        const double loss = validation.loss.value(), accuracy = validation.accuracy.value();
        if (loss < best) {
          best       = loss;
          best_epoch = epoch;
          store.save(checkpoint);
        }
        history << epoch << '\t' << loss << '\t' << accuracy << '\n';
        const double seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - epoch_start).count();
        std::cout << "  epoch " << epoch << ": validation NLL=" << loss << ", accuracy=" << accuracy
                  << " (" << std::setprecision(3) << seconds << std::setprecision(6) << " s)"
                  << std::endl;
      }
      const double train_seconds =
          std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
      store.load(checkpoint);
      const auto restored = Evaluate(store, builder, train_x, train.labels, validation_rows);
      check::Near(
          restored.loss.value(), best, 1e-7, "best checkpoint matches selected validation loss");
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
      check::Require(total == test_rows.size(), "all selected test images evaluated");
      check::Near(
          result.accuracy.value(),
          double(correct) / test_rows.size(),
          1e-7,
          "confusion matches accuracy");
      check::TrainingQuality(
          result.accuracy.value() >= minimum_test_accuracy,
          "classifier did not reach demonstration accuracy threshold");
      check::Require(bool(history), "write training history");
      std::printf("  training time %.1f s (%.2f s/epoch)\n", train_seconds, train_seconds / epochs);
      std::cout << "  best checkpoint and history: " << model_dir.string() << "\nAll " << name
                << " checks passed.\n";
      return 0;
    } catch (const std::exception& e) {
      std::cerr << e.what() << '\n';
      return 1;
    }
  }
}
