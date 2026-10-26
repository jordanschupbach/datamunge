// CMD: just run-nn alexnet
// AlexNet: full architecture checks and a compact CIFAR-10 learning example.
//
// Port of tflow's alexnet. Validate the 96/256/384/384/256 convolution widths,
// two 4096-unit dense layers, 1000 logits and 60,965,224 parameters. Use
// explicit 227x227 input geometry, grouped conv2/4/5, local response
// normalization, overlapping pooling and dropout. Check every stage shape and
// finite nonzero gradients through every learned tensor. Independently check
// grouped connectivity and the paper's LRN formula, then verify stochastic
// training/deterministic inference dropout in a compact graph.
// Architecture references:
// https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf
// https://github.com/BVLC/caffe/blob/master/models/bvlc_alexnet/deploy.prototxt
//
// The CPU learning demonstration uses quarter-width convolutions, 256-unit
// dense layers and bilinear 32->67 image resizing while retaining kernel sizes,
// strides, grouping, LRN and dropout. Fit 12k CIFAR-10 rows with crop/flip
// augmentation for ten epochs; select on a fixed 5k validation split and
// evaluate the official 10k test rows. He initialization and Adam are modern
// adaptations. These are compact CIFAR results, not full-model ImageNet results.
// CIFAR-10 is read from paths::Dataset("cifar10") (tflow's download cache).
// LRN is composed from nn ops (common/lrn.hpp); nn has no fused LRN kernel.
// Options: data=PATH epochs=10 batch=128 lr=.001 fit=12000 augment=0|1 threads=N seed=17.

#include <cmath>
#include <iostream>

#include "common/alexnet.hpp"
#include "common/architecture_checks.hpp"
#include "common/image_training.hpp"

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

void BuildingBlockChecks() {
  nn::ParameterStore p(2);
  nn::Tensor         x = nn::ones({1, 5, 5, 4});
  x.requires_grad_(true);
  const auto grouped   = grouped::GroupedConv2D(p, "conv", x, 6, 3, 2);
  const auto first     = nn::sum(nn::slice(grouped, {0, 0, 0, 0}, {1, 5, 5, 3}));
  const auto dx       = nn::grad(first, {x})[0].to_vector<float>();
  double     connected = 0;
  for (int i = 0; i < 100; ++i) {
    if (i % 4 < 2)
      connected += std::abs(dx[static_cast<std::size_t>(i)]);
    else
      check::Near(dx[static_cast<std::size_t>(i)], 0, 0, "grouped convolution cannot read the other partition");
  }
  check::Require(connected > 0, "connected grouped inputs affect output");
  const auto normalized = alexnet::AlexNetLrn(nn::from_vector(std::vector<float>{1, 2, 3, 4, 5, 6, 7}, {1, 1, 1, 7}))
                              .to_vector<float>();
  for (int c = 0; c < 7; ++c) {
    double squares = 0;
    for (int j = std::max(0, c - 2); j <= std::min(6, c + 2); ++j) squares += (j + 1) * (j + 1);
    check::Near(normalized[static_cast<std::size_t>(c)], (c + 1) / std::pow(2 + .0001 * squares, .75), 5e-7,
                "LRN formula and clipped channel window");
  }
  std::cout << "  grouped connectivity and paper LRN formula checks passed\n";
}

void FullArchitectureChecks() {
  nn::ParameterStore p(17);
  const auto         net = alexnet::AlexNet(p, architecture::SinePixels({1, 227, 227, 3}, .5, .13));
  check::Require(p.count() == 60965224, "full AlexNet parameter count");
  const std::vector<nn::Tensor> stages = {net.convolution[0], net.convolution[1], net.convolution[2], net.convolution[3],
                                          net.convolution[4], net.pooling[0],     net.pooling[1],     net.pooling[2],
                                          net.fc6,            net.fc7,            net.logits};
  const std::vector<nn::Shape>  shapes = {{1, 55, 55, 96}, {1, 27, 27, 256}, {1, 13, 13, 384}, {1, 13, 13, 384},
                                          {1, 13, 13, 256}, {1, 27, 27, 96},  {1, 13, 13, 256}, {1, 6, 6, 256},
                                          {1, 4096},        {1, 4096},        {1, 1000}};
  for (std::size_t i = 0; i < shapes.size(); ++i) architecture::RequireShape(stages[i], shapes[i], "full AlexNet stage");
  const auto [minimum, maximum] =
      architecture::GradientNormRange(p, nn::mean(nn::square(net.logits)), "full AlexNet learned tensor");
  std::cout << "  full AlexNet: parameters=" << p.count() << "; conv spatial sizes 55/27/13/13/13; final pool 6x6\n"
            << "  dense widths 4096/4096/1000; parameter gradient norms [" << minimum << ',' << maximum << "]\n";
}

alexnet::AlexNetConfig CompactConfig() {
  alexnet::AlexNetConfig cfg;
  cfg.image_size    = 67;
  cfg.classes       = 10;
  cfg.width_divisor = 4;
  cfg.dense_width   = 256;
  return cfg;
}

image_training::LogitsBuilder CompactArchitecture() {
  return [](nn::ParameterStore& p, nn::ParameterStore&, const nn::Tensor& x, bool training) {
    const auto image = nn::resize_bilinear(x, {67, 67}, false, true);
    return alexnet::AlexNet(p, image, CompactConfig(), training).logits;
  };
}

void DropoutChecks() {
  // Opposite training/inference paths make the update direction diagnostic.
  // At w=1, training CE plus L2 has a positive derivative; inference CE plus
  // L2 has a negative derivative. This catches accidentally replacing the
  // training objective with inference loss when adding regularization.
  {
    nn::ParameterStore probe(17), state(17);
    probe.set("w", nn::ones({1, 1}));
    const image_training::LogitsBuilder model = [](nn::ParameterStore& p, nn::ParameterStore&, const nn::Tensor& x,
                                                   bool training) {
      const auto score = nn::matmul(x, p.at("w"));
      return training ? nn::concat({-score, score}, 1) : nn::concat({score, -score}, 1);
    };
    image_training::ImageTrainingConfig cfg;
    cfg.classes = 2;
    cfg.l2      = .1;
    nn::OptimizerConfig config;
    config.kind = nn::OptimizerKind::Adam;
    nn::Optimizer optimizer(probe, .01, config);
    const auto    x = nn::ones({1, 1}), y = nn::from_vector(std::vector<std::int64_t>{0}, {1});
    optimizer.minimize(image_training::TrainingObjective(probe, state, model, x, y, cfg));
    const double updated = probe.at("w").item();
    check::Require(updated < 1, "L2 preserves training-path objective and update direction");
    std::cout << "  L2 training-path regression: w=1 -> " << updated << " (inference loss would increase w)\n";
  }
  nn::ParameterStore p(17), state(17);
  const auto         model = CompactArchitecture();
  const auto         input = architecture::SinePixels({4, 32, 32, 3}, 1, .11);
  nn::NoGradGuard    guard;
  const auto first = model(p, state, input, false).to_vector<float>(), second = model(p, state, input, false).to_vector<float>();
  const auto training = model(p, state, input, true).to_vector<float>(), repeat = model(p, state, input, true).to_vector<float>();
  double     stochastic_difference = 0;
  for (std::size_t i = 0; i < first.size(); ++i) {
    check::Near(first[i], second[i], 0, "AlexNet deterministic inference");
    stochastic_difference += std::abs(training[i] - repeat[i]);
  }
  check::Require(stochastic_difference > 1e-5, "training uses fresh dropout masks");
  std::cout << "  compact AlexNet parameters=" << p.count() << "; dropout training/inference checks passed\n";
}

int main(int argc, char** argv) {
  try {
    BuildingBlockChecks();
    FullArchitectureChecks();
    DropoutChecks();
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
  image_training::ImageTrainingConfig cfg;
  cfg.name                  = "alexnet_compact_cifar10";
  cfg.maximum_fit_rows      = 12000;
  cfg.minimum_test_accuracy = .35;
  return image_training::RunCifar10(argc, argv, cfg, CompactArchitecture());
}
