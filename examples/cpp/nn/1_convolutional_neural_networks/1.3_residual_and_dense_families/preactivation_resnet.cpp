// CMD: just run-nn preactivation_resnet
// Pre-activation ResNet: identity mappings through 164 and 1001 layers.
//
// Port of tflow's preactivation_resnet. Follow the authors' CIFAR bottleneck
// implementation: raw 3x3 stem, 9n+2 layers, three stages with 64/128/256
// channels, BN/ReLU before convolutions, final BN/ReLU and global pooling. An
// equal-width shortcut carries the raw input; a dimension-changing projection
// uses the first preactivation. There is no ReLU after addition. He-uniform
// fan-in initialization and the TF BN variance convention (biased batch
// variance, unbiased moving update) are documented adaptations.
// https://arxiv.org/abs/1603.05027
// https://github.com/KaimingHe/resnet-1k-layers/blob/master/resnet-pre-act.lua
//
// Check exact signed identity values and derivatives through fifty blocks,
// preactivated projection shortcuts, a controlled nonzero residual and its
// finite-difference derivative. Full-width 164/1001-layer CIFAR models check
// all stage dimensions, parameter counts and every parameter's gradient
// (two-image training-mode batch; moving averages are not updated).
// Separately train the 164-layer architecture at half width on CIFAR-10:
// 12k fit / 5k validation, twelve epochs, crop/flip, Adam(.001), kernel L2.
// Restore the lowest validation-NLL checkpoint and evaluate all 10k test
// images. This compact run does not reproduce the paper's training schedule.
// CIFAR-10 is read from the tflow dataset cache (common/paths.hpp) or data=.
// Options: data=PATH epochs=12 batch=128 lr=.001 fit=12000 augment=0|1 threads=N seed=17.

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>

#include "common/architecture_checks.hpp"
#include "common/checks.hpp"
#include "common/image_training.hpp"
#include "common/preact_resnet.hpp"

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

namespace {

void IdentityChecks() {
  for (bool projection : {false, true}) {
    nn::ParameterStore p(17), state(17);
    const std::int64_t in = projection ? 2 : 4;
    const int          count = projection ? 1 : 50, stride = projection ? 2 : 1;
    std::vector<float> values(static_cast<std::size_t>(25 * in));
    for (std::size_t i = 0; i < values.size(); ++i) values[i] = (i % 2 ? -1.f : 1.f) * (.2f + .01f * i);
    nn::Tensor x = nn::from_vector(values, {1, 5, 5, in});
    auto build = [&](const nn::Tensor& input) {
      nn::Tensor h = input;
      for (int b = 0; b < count; ++b) h = preact::PreActBottleneck(p, state, "block" + std::to_string(b), h, 1, stride).output;
      return h;
    };
    {
      nn::NoGradGuard guard;
      build(x);
      for (int b = 0; b < count; ++b) {
        const auto name = "block" + std::to_string(b);
        p.set(name + "/conv3/W", nn::zeros({1, 1, 1, 4}));
        if (projection) {
          std::vector<float> kernel(static_cast<std::size_t>(in * 4), 0.f);
          for (int c = 0; c < 4; ++c) kernel[static_cast<std::size_t>((c % in) * 4 + c)] = 1;
          p.set(name + "/shortcut/W", nn::from_vector(kernel, {1, 1, in, 4}));
          p.set(name + "/bn1/gamma", nn::full({in}, std::sqrt(1.f + 1e-5f)));
        }
      }
    }
    x.requires_grad_(true);
    const nn::Tensor h        = build(x);
    const auto       gradient = nn::grad(nn::sum(h), {x})[0].to_vector<float>();
    const int        spatial  = projection ? 3 : 5;
    check::Require(h.shape() == nn::Shape({1, spatial, spatial, 4}), "preactivation shortcut dimensions");
    const auto y = h.to_vector<float>();
    for (int r = 0; r < spatial; ++r)
      for (int c = 0; c < spatial; ++c)
        for (int channel = 0; channel < 4; ++channel) {
          const float value = values[static_cast<std::size_t>((stride * r * 5 + stride * c) * in + channel % in)];
          check::Near(y[static_cast<std::size_t>((r * spatial + c) * 4 + channel)], projection ? std::max(0.f, value) : value,
                      2e-6, "identity retains signed inputs; projection consumes BN/ReLU");
        }
    for (int r = 0; r < 5; ++r)
      for (int c = 0; c < 5; ++c)
        for (int channel = 0; channel < in; ++channel) {
          const auto  i          = static_cast<std::size_t>((r * 5 + c) * in + channel);
          const float derivative = projection ? (r % 2 == 0 && c % 2 == 0 && values[i] > 0 ? 2.f : 0.f) : 1.f;
          check::Near(gradient[i], derivative, 2e-6, "raw identity or preactivated projection derivative");
        }
  }
  std::cout << "  fifty zero-residual blocks retain signed inputs and unit derivatives; preactivated projection checks passed\n";
}

void NonzeroResidualCheck() {
  nn::ParameterStore p(17), state(17);
  std::vector<float> values(36);
  for (int i = 0; i < 36; ++i) values[static_cast<std::size_t>(i)] = (i % 5 < 2 ? -1.f : 1.f) * (.3f + .03f * i);
  nn::Tensor x = nn::from_vector(values, {1, 3, 3, 4});
  {
    nn::NoGradGuard guard;
    preact::PreActBottleneck(p, state, "block", x, 1);
    for (int layer = 1; layer <= 3; ++layer) {
      const std::int64_t in = layer == 1 ? 4 : 1, out = layer == 3 ? 4 : 1, kernel = layer == 2 ? 3 : 1;
      std::vector<float> weight(static_cast<std::size_t>(kernel * kernel * in * out), 0.f);
      for (std::int64_t c = 0; c < out; ++c)
        weight[static_cast<std::size_t>((((kernel / 2) * kernel + kernel / 2) * in + 0) * out + c)] = layer == 3 ? .25f : 1.f;
      p.set("block/conv" + std::to_string(layer) + "/W", nn::from_vector(weight, {kernel, kernel, in, out}));
      p.set("block/bn" + std::to_string(layer) + "/gamma", nn::full({in}, std::sqrt(1.f + 1e-5f)));
    }
  }
  auto total = [&](const nn::Tensor& input) { return nn::sum(preact::PreActBottleneck(p, state, "block", input, 1).output); };
  x.requires_grad_(true);
  const auto block      = preact::PreActBottleneck(p, state, "block", x, 1);
  const auto analytical = nn::grad(nn::sum(block.output), {x})[0].to_vector<float>();
  const auto y          = block.output.to_vector<float>();
  for (int position = 0; position < 9; ++position)
    for (int c = 0; c < 4; ++c) {
      const auto  i  = static_cast<std::size_t>(position * 4 + c);
      const float x0 = values[static_cast<std::size_t>(position * 4)];
      check::Near(y[i], values[i] + .25f * std::max(0.f, x0), 2e-6, "raw input plus controlled residual");
      check::Near(analytical[i], 1.f + (c == 0 && x0 > 0 ? 1.f : 0.f), 2e-6, "identity plus residual Jacobian");
    }
  const float     epsilon = .001f;
  double          error   = 0;
  nn::NoGradGuard guard;
  for (int index : {0, 4, 5, 8}) {
    auto plus = values, minus = values;
    plus[static_cast<std::size_t>(index)] += epsilon;
    minus[static_cast<std::size_t>(index)] -= epsilon;
    const float hi = total(nn::from_vector(plus, {1, 3, 3, 4})).item<float>();
    const float lo = total(nn::from_vector(minus, {1, 3, 3, 4})).item<float>();
    const float derivative = (hi - lo) / (2 * epsilon);
    error = std::max(error, double(std::abs(derivative - analytical[static_cast<std::size_t>(index)])));
    check::Near(derivative, analytical[static_cast<std::size_t>(index)], .002, "nonzero preactivation finite difference");
  }
  std::cout << "  nonzero residual matches host values and identity-plus-residual derivatives; finite-difference error="
            << error << '\n';
}

void FullChecks(int depth, std::int64_t parameters, std::int64_t state_parameters) {
  nn::ParameterStore         p(17), state(17);
  preact::PreActResNetConfig cfg;
  cfg.depth        = depth;
  cfg.update_stats = false;
  const auto net   = preact::PreActResNet(p, state, architecture::SinePixels({2, 32, 32, 3}, .1, .13), cfg, true);
  check::Require(p.count() == parameters && state.count() == state_parameters, "canonical preactivation parameter/state counts");
  check::Require(3 * int(net.blocks.size()) + 2 == depth, "9n+2 bottleneck depth");
  for (int stage = 0; stage < 4; ++stage) {
    const std::int64_t spatial = stage < 2 ? 32 : 32 / (1 << (stage - 1)), channels = stage == 0 ? 16 : 64 << (stage - 1);
    check::Require((stage == 0 ? net.stem : net.stages[stage - 1]).shape() == nn::Shape({2, spatial, spatial, channels}),
                   "canonical preactivation stage dimensions");
  }
  check::Require(net.logits.shape() == nn::Shape({2, 10}), "preactivation classifier dimensions");
  const auto [minimum, maximum] = architecture::GradientNormRange(p, nn::mean(nn::square(net.logits)), "preactivation");
  std::cout << "  preactivation ResNet-" << depth << ": parameters=" << parameters << ", state=" << state_parameters
            << "; stage sizes 32/16/8; gradient norms [" << minimum << ',' << maximum << "]" << std::endl;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    for (int i = 1; i < argc; ++i)  // threads= also applies to the checks
      if (std::string(argv[i]).rfind("threads=", 0) == 0) nn::set_num_threads(std::stoi(std::string(argv[i]).substr(8)));
    IdentityChecks();
    NonzeroResidualCheck();
    FullChecks(164, 1703258, 24224);
    FullChecks(1001, 10327706, 149216);
    image_training::ImageTrainingConfig cfg;
    cfg.name                  = "preactivation_resnet164_compact_cifar10";
    cfg.maximum_fit_rows      = 12000;
    cfg.epochs                = 12;
    cfg.minimum_test_accuracy = .4;
    const image_training::LogitsBuilder builder = [](nn::ParameterStore& p, nn::ParameterStore& state,
                                                     const nn::Tensor& image, bool training) {
      preact::PreActResNetConfig model;
      model.base_width = 8;
      return preact::PreActResNet(p, state, image, model, training).logits;
    };
    if (image_training::RunCifar10(argc, argv, cfg, builder)) return 1;
    std::cout << "All preactivation ResNet checks passed.\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
