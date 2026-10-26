// CMD: just run-nn xception
// Xception: residual entry/middle/exit flows with linear separable operations.
//
// Port of tflow's xception. Follow the original Keras layout, including all
// eight middle blocks, the first entry block's special activation order, and
// learned BN gamma/beta. Validate the 299-pixel/1000-class network's
// 22,855,952 learned parameters, 54,528 moving statistics, all 16 stage shapes
// and every parameter gradient. See common/xception.hpp for sources and
// initialization/state conventions.
//
// A signed fixture proves that no ReLU occurs between depthwise and pointwise
// filters: compare the complete separable+BN output and input gradient with
// host arithmetic, and explicitly show the different result when a ReLU is
// inserted between the factors. Batch normalization follows both factors.
//
// Train a compact version with all blocks retained on CIFAR-10: 71-pixel
// resized inputs, quarter widths, 12k fit / 5k validation rows, twelve epochs,
// crop/flip augmentation, Adam(.001), and BN momentum .9. No dropout, label
// smoothing or auxiliary head. Select by validation NLL and evaluate the
// complete 10k test set once. This CPU run is not an ImageNet benchmark.
// CIFAR-10 is read from paths::Dataset("cifar10"). BN moving averages update
// eagerly in training passes; inference passes leave the state store unchanged.
// Options: data=PATH epochs=12 batch=128 lr=.001 fit=12000 augment=0|1 threads=N seed=17.

#include <cmath>
#include <iostream>

#include "common/architecture_checks.hpp"
#include "common/image_training.hpp"
#include "common/xception.hpp"

// Experiment workflow from examples/nn/common/image_training.hpp.
namespace image_training {
using namespace ::image_training;
inline void TrainImages(const nn::ImageDataset& development, const nn::ImageDataset& test,
                        const LogitsBuilder& model, const ImageTrainingConfig& requested_cfg,
                        const TrainingBuilder& training = {}) {
  auto cfg = requested_cfg;
  cfg.epochs = example_runtime::Limit(cfg.epochs, 1);
  cfg.batch = example_runtime::Limit(cfg.batch, 8);
  if (!example_runtime::Full()) cfg.maximum_fit_rows = cfg.maximum_fit_rows ? std::min(cfg.maximum_fit_rows, 32) : 32;
  example_runtime::Describe();
  const std::int64_t development_rows = development.images.shape()[0];
  const std::int64_t test_size = example_runtime::Limit<std::int64_t>(test.images.shape()[0], 32);
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
  const std::vector<std::int64_t> fit(order.begin(), order.begin() + static_cast<std::ptrdiff_t>(fit_count));
  std::vector<std::int64_t>
      validation(order.begin() + static_cast<std::ptrdiff_t>(available), order.end());
  validation.resize(example_runtime::Limit<std::size_t>(validation.size(), 32));
  std::vector<std::int64_t> test_rows(static_cast<std::size_t>(test_size));
  std::iota(test_rows.begin(), test_rows.end(), 0);

  nn::ParameterStore params(cfg.seed), state(cfg.seed);
  {
    nn::NoGradGuard guard;  // create every parameter (and state entry) before building the optimizer
    const nn::Tensor x = ImageBatch(development, {fit[0], fit[1]}).first;
    model(params, state, x, false);
    if (training) {
      // Training-only parameters (auxiliary heads); their batch-norm statistics
      // go to a scratch store so this pass updates no real moving averages.
      nn::ParameterStore scratch(cfg.seed);
      training(params, scratch, x);
    }
  }
  nn::OptimizerConfig config;
  config.kind = nn::OptimizerKind::Adam;
  nn::Optimizer optimizer(params, cfg.learning_rate, config);

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
      optimizer.minimize(TrainingObjective(params, state, model, x, y, cfg, training));
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
  check::TrainingQuality(result.accuracy.value() >= cfg.minimum_test_accuracy, "image classifier demonstration accuracy threshold");
  check::Require(bool(history), "write image training history");
  double train_total = 0;
  for (double s : EpochSeconds()) train_total += s;
  std::printf("  training time %.1f s (%.2f s/epoch)\n", train_total, train_total / cfg.epochs);
  std::cout << "  best checkpoint and history: " << directory.string() << "\nAll " << cfg.name << " checks passed.\n";
}
}  // namespace image_training

// Experiment workflow from examples/nn/common/image_training.hpp.
namespace image_training {
using namespace ::image_training;
inline int RunCifar10(int argc, char** argv, ImageTrainingConfig cfg, const LogitsBuilder& architecture,
                      const TrainingBuilder& training = {}) {
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
    TrainImages(development, test, architecture, cfg, training);
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
}  // namespace image_training



namespace nn = datamunge::nn;

void SeparableActivationChecks() {
  nn::ParameterStore              p(0), state(0);
  normalized_conv::ConvNormConfig cfg;
  const std::vector<float>        values = {-2, 1, 3, -4, 2, -1, 4, -3, 1};
  nn::Tensor                      input  = nn::from_vector(values, {1, 3, 3, 1});
  {
    nn::NoGradGuard guard;
    xception::XceptionSeparable(p, state, "layer", input, 1, cfg);
  }
  check::Require(p.count() == 12 && state.count() == 2, "separable+BN parameters and inference state");
  std::vector<float> kernel(9, 0.f);
  kernel[4] = 1;
  p.set("layer/depthwise/W", nn::from_vector(kernel, {3, 3, 1, 1}));
  p.set("layer/pointwise/W", nn::from_vector(std::vector<float>{-2}, {1, 1, 1, 1}));
  input.requires_grad_(true);
  const auto output   = xception::XceptionSeparable(p, state, "layer", input, 1, cfg);
  const auto gradient = nn::grad(nn::sum(output), {input})[0].to_vector<float>();
  const auto depthwise = nn::depthwise_conv2d(input, p.at("layer/depthwise/W"));
  const auto intervening_relu =
      (nn::conv2d(nn::relu(depthwise), p.at("layer/pointwise/W")) * (1. / std::sqrt(1.001))).to_vector<float>();
  const auto out        = output.to_vector<float>();
  double     difference = 0;
  for (int i = 0; i < 9; ++i) {
    const auto k = static_cast<std::size_t>(i);
    check::Near(out[k], -2 * values[k] / std::sqrt(1.001), 2e-6, "no activation between depthwise and pointwise");
    check::Near(gradient[k], -2 / std::sqrt(1.001), 2e-6, "linear separable input derivative");
    check::Near(intervening_relu[k], -2 * std::max(0.f, values[k]) / std::sqrt(1.001), 2e-6, "intervening ReLU reference");
    difference += std::abs(out[k] - intervening_relu[k]);
  }
  check::Require(difference > 10, "intervening ReLU changes the operation");
  std::cout << "  linear depthwise/pointwise output and derivative passed; inserting ReLU changes summed absolute outputs by "
            << difference << std::endl;
}

void FullArchitectureChecks() {
  nn::ParameterStore p(17), state(17);
  const auto         net = xception::Xception(p, state, architecture::SinePixels({1, 299, 299, 3}, .1, .13));
  check::Require(p.count() == 22855952 && state.count() == 54528, "canonical Xception learned/state counts");
  const auto initial_state = state.flatten();
  std::vector<std::pair<int, int>> expected = {{149, 32}, {147, 64}, {74, 128}, {37, 256}, {19, 728}};
  expected.insert(expected.end(), 8, {19, 728});
  expected.emplace_back(10, 1024);
  expected.emplace_back(10, 1536);
  expected.emplace_back(10, 2048);
  check::Require(net.stages.size() == expected.size(), "all Xception stages");
  for (std::size_t i = 0; i < expected.size(); ++i) {
    const auto [spatial, channels] = expected[i];
    architecture::RequireShape(net.stages[i].second, {1, spatial, spatial, channels},
                               "Xception reference shape: " + net.stages[i].first);
  }
  architecture::RequireShape(net.logits, {1, 1000}, "Xception classifier dimensions");
  const auto [minimum, maximum] = architecture::GradientNormRange(p, nn::mean(nn::square(net.logits)), "Xception parameter");
  check::Require(state.flatten() == initial_state, "frozen inference state");
  std::cout << "  full Xception: learned=" << p.count() << ", running state=" << state.count()
            << "; 16 stage shapes; gradient norms [" << minimum << ',' << maximum << "]" << std::endl;
}

int main(int argc, char** argv) {
  try {
    SeparableActivationChecks();
    FullArchitectureChecks();
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
  const image_training::LogitsBuilder model = [](nn::ParameterStore& p, nn::ParameterStore& state, const nn::Tensor& x,
                                                 bool training) {
    xception::XceptionConfig cfg;
    cfg.image_size    = 71;
    cfg.classes       = 10;
    cfg.width_divisor = 4;
    cfg.momentum      = .9;
    return xception::Xception(p, state, nn::resize_bilinear(x, {71, 71}, false, true), cfg, training).logits;
  };
  image_training::ImageTrainingConfig cfg;
  cfg.name                  = "xception_compact_cifar10";
  cfg.maximum_fit_rows      = 12000;
  cfg.epochs                = 12;
  cfg.minimum_test_accuracy = .35;
  return image_training::RunCifar10(argc, argv, cfg, model);
}
