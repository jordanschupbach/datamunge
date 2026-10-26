// CMD: just run-nn vgg
// VGG-11, VGG-16 and VGG-19: small kernels, depth and convolutional classifiers.
//
// Port of tflow's vgg. Implement configurations A/D/E from Simonyan/Zisserman
// without batch norm or LRN. Validate every canonical pooling shape,
// learned-layer count and full parameter count, plus finite nonzero parameter
// gradients for all three 224-pixel/1000-class models. Compute their final
// receptive fields and show why three 3x3 convolutions use fewer weights than
// one 7x7 at equal width.
// Reference: https://arxiv.org/abs/1409.1556
//
// Convert the three dense layers to convolutional kernels with shared weights.
// Check equality on each full model's 7x7 feature map and check every sliding
// window of a larger feature map in a small fixture. Feature-window equality
// does not imply equality with raw-image crops, whose padding/context differ.
//
// Separately train compact versions of all three depths on CIFAR-10: 32-pixel
// input, quarter-width convolutions and 256-unit dense layers, preserving the
// five pooling stages and per-block depths. Use the same 12k fit / 5k validation
// split, ten epochs, crop/flip augmentation, dropout, He initialization and
// Adam(.0005). Select by validation NLL and evaluate all 10k test rows. These
// CPU demonstrations are not full-model ImageNet performance measurements.
// CIFAR-10 is read from paths::Dataset("cifar10").
// Options: data=PATH epochs=10 batch=128 lr=.0005 fit=12000 augment=0|1 threads=N seed=17.

#include <cmath>
#include <iostream>

#include "common/architecture_checks.hpp"
#include "common/image_training.hpp"
#include "common/receptive_field.hpp"
#include "common/vgg.hpp"

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

void FullArchitectureChecks(int depth, std::int64_t expected_parameters, int expected_receptive_field) {
  nn::ParameterStore p(17);
  vgg::VggConfig     cfg;
  cfg.depth      = depth;
  const auto net = vgg::Vgg(p, architecture::SinePixels({1, 224, 224, 3}, .1, .13), cfg);
  check::Require(p.count() == expected_parameters, "canonical VGG parameter count");
  check::Require(net.convolution.size() + 3 == std::size_t(depth), "VGG learned-layer depth");
  double head_error = 0;
  {
    nn::NoGradGuard guard;
    const auto      convolutional = vgg::VggConvolutionalHead(p, net.pooling[4], cfg);
    check::Require(net.logits.shape() == nn::Shape({1, 1000}) && convolutional.shape() == nn::Shape({1, 1, 1, 1000}),
                   "VGG classifier dimensions");
    const auto a = net.logits.to_vector<float>(), b = convolutional.to_vector<float>();
    for (std::size_t i = 0; i < a.size(); ++i) head_error = std::max(head_error, double(std::abs(a[i] - b[i])));
  }
  check::Near(head_error, 0, 2e-4, "full dense and convolutional classifiers agree");
  const std::array<int, 5> widths = {64, 128, 256, 512, 512};
  for (int stage = 0; stage < 5; ++stage) {
    const std::int64_t spatial = 224 / (1 << (stage + 1));
    architecture::RequireShape(net.pooling[static_cast<std::size_t>(stage)],
                               {1, spatial, spatial, widths[static_cast<std::size_t>(stage)]}, "VGG pooling stage shape");
  }
  const auto [minimum, maximum] =
      architecture::GradientNormRange(p, nn::mean(nn::square(net.logits)), "VGG parameter tensor");
  receptive::ReceptiveField field;
  for (int layers : vgg::VggBlockDepths(depth)) {
    for (int layer = 0; layer < layers; ++layer) field = field.After(3, 1, 1);
    field = field.After(2, 2, 0);
  }
  check::Require(field.width == expected_receptive_field && field.jump == 32, "VGG final receptive-field geometry");
  std::cout << "  VGG-" << depth << ": parameters=" << p.count() << "; pooling sizes 112/56/28/14/7; receptive field="
            << field.width << "; gradient norms [" << minimum << ',' << maximum << "]; dense/conv error=" << head_error
            << std::endl;
}

void SlidingHeadChecks() {
  nn::ParameterStore p(9);
  vgg::VggConfig     cfg;
  cfg.depth         = 11;
  cfg.image_size    = 64;
  cfg.classes       = 3;
  cfg.width_divisor = 8;
  cfg.dense_width   = 32;
  nn::NoGradGuard guard;
  vgg::Vgg(p, nn::zeros({1, 64, 64, 3}), cfg);
  const auto              features      = architecture::SinePixels({1, 3, 4, 64}, 1, .17);
  const auto              convolutional = vgg::VggConvolutionalHead(p, features, cfg);
  std::vector<nn::Tensor> windows;
  for (int row = 0; row < 2; ++row)
    for (int column = 0; column < 3; ++column) {
      const auto patch = nn::reshape(nn::slice(features, {0, row, column, 0}, {1, 2, 2, 64}), {1, 256});
      auto       h     = nn::dense(p, "fc6", patch, 32, nn::Activation::Relu);
      h                = nn::dense(p, "fc7", h, 32, nn::Activation::Relu);
      windows.push_back(nn::dense(p, "fc8", h, 3));
    }
  const auto manual = nn::concat(windows, 0).to_vector<float>();
  check::Require(convolutional.shape() == nn::Shape({1, 2, 3, 3}), "sliding convolutional class score map");
  const auto scores = convolutional.to_vector<float>();
  for (std::size_t i = 0; i < 18; ++i) check::Near(scores[i], manual[i], 2e-6, "each sliding window equals its dense classifier");
  std::cout << "  convolutional classifier equals all six explicit feature-window classifiers\n";
  std::cout << "  equal-width spatial weights: three 3x3 layers=27*C*C; one 7x7=49*C*C; the stack also has three ReLUs\n";
}

int main(int argc, char** argv) {
  try {
    SlidingHeadChecks();
    FullArchitectureChecks(11, 132863336, 150);
    FullArchitectureChecks(16, 138357544, 212);
    FullArchitectureChecks(19, 143667240, 268);
    for (int depth : {11, 16, 19}) {
      const image_training::LogitsBuilder model = [depth](nn::ParameterStore& p, nn::ParameterStore&, const nn::Tensor& x,
                                                          bool training) {
        vgg::VggConfig cfg;
        cfg.depth         = depth;
        cfg.image_size    = 32;
        cfg.classes       = 10;
        cfg.width_divisor = 4;
        cfg.dense_width   = 256;
        return vgg::Vgg(p, x, cfg, training).logits;
      };
      image_training::ImageTrainingConfig cfg;
      cfg.name                  = "vgg" + std::to_string(depth) + "_compact_cifar10";
      cfg.maximum_fit_rows      = 12000;
      cfg.learning_rate         = .0005;
      cfg.minimum_test_accuracy = .35;
      if (image_training::RunCifar10(argc, argv, cfg, model)) return 1;
    }
    std::cout << "All VGG-11/16/19 checks passed.\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
