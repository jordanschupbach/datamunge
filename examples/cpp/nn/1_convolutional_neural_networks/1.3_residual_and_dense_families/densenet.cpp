// CMD: just run-nn densenet
// DenseNet: feature concatenation, bottlenecks and compressed transitions.
//
// Port of tflow's densenet. Check preserved feature ordering and direct
// identity derivatives through four dense layers, with and without
// bottlenecks. A controlled transition checks 1x1 compression followed by 2x2
// average pooling against host values and derivatives. Full ImageNet
// DenseNet-121 (7,978,856 parameters) and CIFAR DenseNet-BC-100 k=12 (769,162
// parameters) check every block/transition dimension and finite nonzero
// derivatives through every learned parameter (two-image training-mode batch;
// moving averages are not updated).
// https://arxiv.org/abs/1608.06993
// https://github.com/liuzhuang13/DenseNet/blob/master/models/densenet.lua
//
// Train the smaller DenseNet-BC-40 k=12 (176,122 parameters) on CIFAR-10,
// with 4k bottlenecks, half-width transitions and no dropout. Use 12k fit /
// 5k validation, twelve epochs, crop/flip, Adam(.001), kernel L2(.0001) and
// evaluate all 10k test images after restoring the best validation-NLL model.
// This is a short demonstration with He-uniform initialization, not the
// paper's SGD schedule. The model materializes concatenations; it does not
// implement the authors' memory-saving buffers or activation recomputation.
// CIFAR-10 is read from the tflow dataset cache (common/paths.hpp) or data=.
// Options: data=PATH epochs=12 batch=128 lr=.001 fit=12000 augment=0|1 threads=N seed=17.

#include <cmath>
#include <iostream>
#include <limits>

#include "common/architecture_checks.hpp"
#include "common/checks.hpp"
#include "common/densenet.hpp"
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

namespace {

void FeatureReuseChecks() {
  for (bool bottleneck : {false, true}) {
    nn::ParameterStore p(17), state(17);
    std::vector<float> values(2 * 5 * 5 * 3);
    for (std::size_t i = 0; i < values.size(); ++i) values[i] = std::sin(.17f * static_cast<float>(i));
    nn::Tensor x = nn::from_vector(values, {2, 5, 5, 3});
    std::vector<densenet::DenseGrowthOutputs> outs;
    auto build = [&](const nn::Tensor& input) {
      outs.clear();
      nn::Tensor h = input;
      for (int layer = 0; layer < 4; ++layer) {
        outs.push_back(densenet::DenseGrowthLayer(p, state, "layer" + std::to_string(layer), h, 2, bottleneck));
        h = outs.back().output;
      }
      return h;
    };
    {
      nn::NoGradGuard guard;
      build(x);
      for (int layer = 0; layer < 4; ++layer) {
        const int channels = 3 + 2 * layer;
        check::Require(outs[layer].output.shape() == nn::Shape({2, 5, 5, channels + 2}), "linear DenseNet channel growth");
        const auto out = outs[layer].output.to_vector<float>(), added = outs[layer].added.to_vector<float>();
        const auto previous = layer == 0 ? values : outs[layer - 1].output.to_vector<float>();
        for (int position = 0; position < 50; ++position) {
          for (int c = 0; c < channels; ++c)
            check::Near(out[position * (channels + 2) + c], previous[position * channels + c], 0,
                        "all preceding feature maps retained verbatim");
          for (int c = 0; c < 2; ++c)
            check::Near(out[position * (channels + 2) + channels + c], added[position * 2 + c], 0,
                        "new features appended after existing channels");
        }
      }
      for (int layer = 0; layer < 4; ++layer)  // zero appended features
        p.set("layer" + std::to_string(layer) + "/conv2/W", nn::zeros({3, 3, bottleneck ? 8 : 3 + 2 * layer, 2}));
    }
    x.requires_grad_(true);
    const nn::Tensor h = build(x);
    const auto gradient = nn::grad(nn::sum(h), {x})[0].to_vector<float>();
    const auto y        = h.to_vector<float>();
    for (int position = 0; position < 50; ++position)
      for (int c = 0; c < 11; ++c)
        check::Near(y[position * 11 + c], c < 3 ? values[position * 3 + c] : 0, 0,
                    "signed original features plus zero appended channels");
    for (float g : gradient) check::Near(g, 1, 2e-6, "unit direct input derivative through dense concatenations");
  }
  std::cout << "  basic/bottleneck feature ordering, linear growth and direct identity derivatives passed\n";
}

void TransitionCheck() {
  nn::ParameterStore p(17), state(17);
  std::vector<float> values(64);
  for (int i = 0; i < 64; ++i) values[i] = .1f + .02f * i;
  nn::Tensor x = nn::from_vector(values, {1, 4, 4, 4});
  {
    nn::NoGradGuard guard;
    densenet::DenseTransition(p, state, "transition", x, 2);
    std::vector<float> weights(8, 0.f);
    weights[0 * 2 + 0] = weights[1 * 2 + 1] = 1;
    p.set("transition/conv/W", nn::from_vector(weights, {1, 1, 4, 2}));
    p.set("transition/bn/gamma", nn::full({4}, std::sqrt(1.f + 1e-5f)));
  }
  x.requires_grad_(true);
  const nn::Tensor h = densenet::DenseTransition(p, state, "transition", x, 2);
  const auto gradient = nn::grad(nn::sum(h), {x})[0].to_vector<float>();
  check::Require(h.shape() == nn::Shape({1, 2, 2, 2}), "transition spatial and channel compression");
  const auto y = h.to_vector<float>();
  for (int r = 0; r < 2; ++r)
    for (int c = 0; c < 2; ++c)
      for (int channel = 0; channel < 2; ++channel) {
        double expected = 0;
        for (int dr = 0; dr < 2; ++dr)
          for (int dc = 0; dc < 2; ++dc) expected += .25 * values[((2 * r + dr) * 4 + 2 * c + dc) * 4 + channel];
        check::Near(y[(r * 2 + c) * 2 + channel], expected, 2e-6, "host transition projection and average pooling");
      }
  for (int i = 0; i < 64; ++i) check::Near(gradient[i], i % 4 < 2 ? .25 : 0, 2e-6, "host transition derivative");
  std::cout << "  compressed transition matches host values and pooling derivatives\n";
}

void FullChecks(bool cifar) {
  nn::ParameterStore       p(17), state(17);
  densenet::DenseNetConfig cfg;
  cfg.cifar_layout = cifar;
  cfg.depth        = cifar ? 100 : 121;
  cfg.growth_rate  = cifar ? 12 : 32;
  cfg.classes      = cifar ? 10 : 1000;
  cfg.update_stats = false;
  const int  pixels = cifar ? 32 : 224, initial_spatial = cifar ? 32 : 56;
  const auto channels = cifar ? std::vector<std::int64_t>{216, 300, 342} : std::vector<std::int64_t>{256, 512, 1024, 1024};
  const auto net = densenet::DenseNet(p, state, architecture::SinePixels({2, pixels, pixels, 3}, .1, .13), cfg, true);
  check::Require(p.count() == (cifar ? 769162 : 7978856) && state.count() == (cifar ? 23988 : 83648),
                 "canonical DenseNet parameter/state counts");
  check::Require(net.layers.size() == (cifar ? 48u : 58u) && net.stages.size() == channels.size(),
                 "canonical DenseNet layer counts");
  for (std::size_t i = 0; i < channels.size(); ++i) {
    const std::int64_t spatial = initial_spatial / (1 << i);
    check::Require(net.stages[i].shape() == nn::Shape({2, spatial, spatial, channels[i]}), "canonical dense block dimensions");
    if (i + 1 < channels.size())
      check::Require(net.transitions[i].shape() == nn::Shape({2, spatial / 2, spatial / 2, channels[i] / 2}),
                     "canonical transition dimensions");
  }
  check::Require(net.logits.shape() == nn::Shape({2, cfg.classes}), "DenseNet classifier dimensions");
  const auto [minimum, maximum] = architecture::GradientNormRange(p, nn::mean(nn::square(net.logits)), "DenseNet");
  std::cout << "  DenseNet-BC-" << cfg.depth << " k=" << cfg.growth_rate << ": parameters=" << p.count()
            << ", state=" << state.count() << "; gradient norms [" << minimum << ',' << maximum << "]" << std::endl;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    for (int i = 1; i < argc; ++i)  // threads= also applies to the checks
      if (std::string(argv[i]).rfind("threads=", 0) == 0) nn::set_num_threads(std::stoi(std::string(argv[i]).substr(8)));
    FeatureReuseChecks();
    TransitionCheck();
    FullChecks(false);
    FullChecks(true);
    image_training::ImageTrainingConfig cfg;
    cfg.name                  = "densenet_bc40_k12_cifar10";
    cfg.maximum_fit_rows      = 12000;
    cfg.epochs                = 12;
    cfg.minimum_test_accuracy = .45;
    const image_training::LogitsBuilder builder = [](nn::ParameterStore& p, nn::ParameterStore& state,
                                                     const nn::Tensor& image, bool training) {
      densenet::DenseNetConfig model;
      model.cifar_layout = true;
      model.depth        = 40;
      model.growth_rate  = 12;
      model.classes      = 10;
      const auto logits  = densenet::DenseNet(p, state, image, model, training).logits;
      check::Require(p.count() == 176122 && state.count() == 5088, "training DenseNet parameter/state counts");
      return logits;
    };
    if (image_training::RunCifar10(argc, argv, cfg, builder)) return 1;
    std::cout << "All DenseNet checks passed.\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
