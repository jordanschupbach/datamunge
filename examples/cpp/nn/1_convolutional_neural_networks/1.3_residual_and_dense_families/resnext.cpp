// CMD: just run-nn resnext
// ResNeXt: grouped bottlenecks and equivalent aggregated transformations.
//
// Port of tflow's resnext. Implement the released Form C: 1x1 reduction,
// grouped 3x3 (with stride), 1x1 expansion and post-addition ReLU. Cardinality
// and per-group width are independently configurable. An explicit
// split/transform/sum implementation shares the same weights and verifies
// outputs plus all input/parameter derivatives, including strided projection
// shortcuts and nontrivial BN state. Compare the residual-kernel budgets of a
// 1x64d bottleneck and 32x4d block. Full ResNeXt-50 32x4d has 25,028,904
// learned parameters; check all stage dimensions and parameter gradients at
// 224 pixels with 1000 output classes.
// https://arxiv.org/abs/1611.05431
// https://github.com/facebookresearch/ResNeXt/blob/master/models/resnext.lua
//
// Train a compact CIFAR 29-layer model with three groups of three blocks,
// eight transformations per block, initial group width four and stage base
// widths 16/32/64 (quarter the release's outer widths). Retain the CIFAR stem
// and global pooling. Use 12k fit / 5k validation, twelve epochs, crop/flip,
// Adam(.001), kernel L2(.0001), and evaluate 10k test images after restoring
// the lowest validation-NLL checkpoint. He-uniform initialization and this
// short training schedule are adaptations, not a reproduction benchmark.
// Deviations: the grouped 3x3 is one nn::conv2d with Conv2DOptions::groups and
// a single filter "<block>/conv2/W" (tflow: per-group "conv2/group<g>/W"); the
// aggregated reference slices that filter per group. Weights come from nn's
// generator (seed=), so trained numbers match TF only statistically. The
// full-model gradient check disables the moving-average updates.
// Options: data=PATH epochs=12 batch=128 lr=.001 fit=12000 augment=0|1 threads=N seed=17.

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>

#include "common/checks.hpp"
#include "common/image_training.hpp"
#include "common/resnext.hpp"

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
void AggregationChecks() {
  for (int stride : {1, 2}) {
    nn::ParameterStore     p(17), state(17);
    resnet::ConvNormConfig cfg;
    cfg.scale   = true;
    cfg.epsilon = 1e-5;
    std::vector<float> values(2 * 5 * 5 * 8);
    for (std::size_t i = 0; i < values.size(); ++i) values[i] = .2f + .3f * std::sin(.17f * static_cast<std::int64_t>(i));
    nn::Tensor input = nn::from_vector(values, {2, 5, 5, 8});
    {
      nn::NoGradGuard guard;  // create the parameters, then set nontrivial normalization state
      resnext::ResNeXtBlock(p, state, "block", input, 2, 2, 64, cfg, stride);
      for (const auto& name : state.names()) {
        const auto         n = state.at(name).numel();
        std::vector<float> v(static_cast<std::size_t>(n));
        for (std::int64_t i = 0; i < n; ++i)
          v[static_cast<std::size_t>(i)] = name.find("moving_var") != std::string::npos ? .7f + .03f * i : .03f * (i - 2);
        state.set(name, nn::from_vector(v, {n}));
      }
      for (const auto& name : p.names())
        if (name.find("/bn/") != std::string::npos) {
          const auto         n = p.at(name).numel();
          std::vector<float> v(static_cast<std::size_t>(n));
          for (std::int64_t i = 0; i < n; ++i)
            v[static_cast<std::size_t>(i)] = name.find("gamma") != std::string::npos ? .8f + .03f * i : .1f + .01f * i;
          p.set(name, nn::from_vector(v, {n}));
        }
    }
    input.requires_grad_(true);
    const auto grouped = resnext::ResNeXtBlock(p, state, "block", input, 2, 2, 64, cfg, stride);
    auto norm = [&](const nn::Tensor& x, const std::string& layer, std::int64_t start, std::int64_t count) {
      const auto name  = "block/" + layer + "/bn/";
      auto       slice = [&](const nn::ParameterStore& store, const std::string& part) {
        return nn::slice_axis(store.at(name + part), 0, start, start + count);
      };
      return (x - slice(state, "moving_mean")) * nn::rsqrt(slice(state, "moving_var") + cfg.epsilon) * slice(p, "gamma") +
             slice(p, "beta");
    };
    std::vector<nn::Tensor> branches;
    for (int group = 0; group < 2; ++group) {
      const nn::Tensor first = nn::slice_axis(p.at("block/conv1/W"), 3, 2 * group, 2 * group + 2);
      nn::Tensor       h     = nn::relu(norm(nn::conv2d(input, first, nn::conv_options(1, nn::Padding::Valid)), "conv1", 2 * group, 2));
      const nn::Tensor middle = nn::slice_axis(p.at("block/conv2/W"), 3, 2 * group, 2 * group + 2);
      h = nn::conv2d(nn::pad(h, {{0, 0}, {1, 1}, {1, 1}, {0, 0}}), middle, nn::conv_options(stride, nn::Padding::Valid));
      h = nn::relu(norm(h, "conv2", 2 * group, 2));
      const nn::Tensor last = nn::slice_axis(p.at("block/conv3/W"), 2, 2 * group, 2 * group + 2);
      branches.push_back(nn::conv2d(h, last, nn::conv_options(1, nn::Padding::Valid)));
    }
    nn::Tensor shortcut = input;
    if (stride == 2)
      shortcut = norm(nn::conv2d(input, p.at("block/shortcut/W"), nn::conv_options(2, nn::Padding::Valid)), "shortcut", 0, 8);
    const nn::Tensor aggregated = nn::relu(shortcut + norm(branches[0] + branches[1], "conv3", 0, 8));
    auto variables = p.parameters();
    variables.insert(variables.begin(), input);
    const auto ga = nn::grad(nn::mean(nn::square(grouped.output)), variables);
    const auto gb = nn::grad(nn::mean(nn::square(aggregated)), variables);
    std::vector<std::pair<nn::Tensor, nn::Tensor>> pairs = {{grouped.output, aggregated}};
    for (std::size_t i = 0; i < ga.size(); ++i) pairs.emplace_back(ga[i], gb[i]);
    double error = 0;
    for (const auto& [x, y] : pairs) {
      check::Require(x.shape() == y.shape(), "equivalent bottleneck tensor shapes");
      const auto a = x.to_vector<float>(), b = y.to_vector<float>();
      for (std::size_t i = 0; i < a.size(); ++i)
        error = std::max(error, std::abs(double(a[i]) - b[i]) / std::max({1., std::abs(double(a[i])), std::abs(double(b[i]))}));
    }
    check::Near(error, 0, 3e-5, "split-transform-sum and grouped outputs/input/all parameter derivatives");
    std::cout << "  grouped versus aggregated, stride " << stride << ": all outputs and derivatives agree; error=" << error << '\n';
  }
  const auto kernel_budget = [](int cardinality, int width) { return 2 * 256 * cardinality * width + 9 * cardinality * width * width; };
  check::Require(kernel_budget(1, 64) == 69632 && kernel_budget(32, 4) == 70144, "matched bottleneck kernel budgets");
  std::cout << "  residual-kernel counts at 256 input/output channels: 1x64d=" << kernel_budget(1, 64)
            << ", 32x4d=" << kernel_budget(32, 4) << " (excludes BN and projection)\n";
}

void FullChecks() {
  nn::ParameterStore     p(17), state(17);
  resnext::ResNeXtConfig cfg;
  cfg.update_stats = false;
  std::vector<float> pixels(2 * 224 * 224 * 3);
  for (std::size_t i = 0; i < pixels.size(); ++i) pixels[i] = .1f * std::sin(static_cast<std::int64_t>(i) * .13f);
  const auto net = resnext::ResNeXt(p, state, nn::from_vector(pixels, {2, 224, 224, 3}), cfg, true);
  check::Require(p.count() == 25028904 && state.count() == 68224, "canonical ResNeXt-50 parameter/state counts");
  check::Require(net.blocks.size() == 16 && net.stages.size() == 4, "canonical ResNeXt-50 block counts");
  for (int stage = 0; stage < 5; ++stage) {
    const std::int64_t spatial = stage < 2 ? 56 : 56 / (1 << (stage - 1)), channels = stage == 0 ? 64 : 256 << (stage - 1);
    const auto&        shape   = (stage == 0 ? net.stem : net.stages[static_cast<std::size_t>(stage - 1)]).shape();
    check::Require(shape == nn::Shape({2, spatial, spatial, channels}), "canonical ResNeXt stage dimensions");
  }
  check::Require(net.logits.shape() == nn::Shape({2, 1000}), "ResNeXt classifier dimensions");
  const auto gradients = nn::grad(nn::mean(nn::square(net.logits)), p.parameters());
  double     minimum = std::numeric_limits<double>::infinity(), maximum = 0;
  for (std::size_t i = 0; i < gradients.size(); ++i) {
    const double norm = nn::sqrt(nn::sum(nn::square(gradients[i]))).item();
    check::Require(std::isfinite(norm) && norm > 0, "finite nonzero ResNeXt derivative: " + p.names()[i]);
    minimum = std::min(minimum, norm);
    maximum = std::max(maximum, norm);
  }
  std::cout << "  ResNeXt-50 32x4d: parameters=" << p.count() << ", state=" << state.count() << "; gradient norms ["
            << minimum << ',' << maximum << "]" << std::endl;
}
}  // namespace

int main(int argc, char** argv) {
  try {
    for (int i = 1; i < argc; ++i)  // threads= also applies to the checks
      if (std::string(argv[i]).rfind("threads=", 0) == 0) nn::set_num_threads(std::stoi(std::string(argv[i]).substr(8)));
    AggregationChecks();
    FullChecks();
    image_training::ImageTrainingConfig cfg;
    cfg.name                  = "resnext29_8x4d_compact_cifar10";
    cfg.maximum_fit_rows      = 12000;
    cfg.epochs                = 12;
    cfg.minimum_test_accuracy = .45;
    const image_training::LogitsBuilder builder = [](nn::ParameterStore& p, nn::ParameterStore& state,
                                                     const nn::Tensor& image, bool training) {
      resnext::ResNeXtConfig model;
      model.cifar_layout = true;
      model.depth        = 29;
      model.cardinality  = 8;
      model.base_width   = 16;
      model.stage_width  = 16;
      model.classes      = 10;
      const auto logits  = resnext::ResNeXt(p, state, image, model, training).logits;
      check::Require(p.count() == 359898 && state.count() == 6304, "compact ResNeXt parameter/state counts");
      return logits;
    };
    if (image_training::RunCifar10(argc, argv, cfg, builder)) return 1;
    std::cout << "All ResNeXt checks passed.\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
