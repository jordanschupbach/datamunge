// CMD: just run-nn wide_resnet
// Wide ResNet: width scaling and dropout inside the residual branch.
//
// Port of tflow's wide_resnet. Implement the authors' CIFAR 6n+4 layout with
// preactivated basic blocks, fixed 16-channel stem and 16k/32k/64k stages.
// Dropout follows the second BN/ReLU and precedes the second convolution;
// shortcuts remain undropped. Check WRN-16-1, WRN-16-2 and the full
// 36,479,194-parameter WRN-28-10, including stage shapes and derivatives
// through every learned parameter. Count convolution/classifier MACs for one
// image; BN, ReLU, additions and pooling are excluded. These operation counts
// are not measured latency.
// https://arxiv.org/abs/1605.07146
// https://github.com/szagoruyko/wide-residual-networks/blob/master/models/wide-resnet.lua
//
// Controlled checks verify inverted dropout values and derivatives, repeatable
// inference and signed identity paths even with training dropout enabled.
// Train three WRN-16 configurations on CIFAR-10: width 1 without dropout,
// width 2 without dropout, and width 2 with dropout .3. Each uses the same
// 12k fit / 5k validation split, seed, twelve epochs, crop/flip, Adam(.001)
// and kernel L2(.0001). Restore each lowest validation-NLL checkpoint and
// evaluate 10k test images. These single-seed short runs illustrate the
// comparison procedure; they do not establish an architecture ranking or
// reproduce the paper's long SGD schedule. Initialization uses He-uniform.
// Deviations: dropout masks come from one nn::Generator per model (tflow
// seeds each block); weights come from nn's generator (seed=), so trained
// numbers match TF only statistically. Fixture and full-model checks run
// training mode with the moving-average updates disabled (tflow does not run
// its update ops there).
// Options: data=PATH epochs=12 batch=128 lr=.001 fit=12000 augment=0|1 threads=N seed=17.

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>

#include "common/checks.hpp"
#include "common/image_training.hpp"
#include "common/wide_resnet.hpp"

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
void DropoutChecks() {
  nn::ParameterStore p(17), state(17);
  nn::Generator      gen(17);
  std::vector<float> values(8 * 8 * 8 * 4);
  for (std::size_t i = 0; i < values.size(); ++i) values[i] = std::sin(.17f * static_cast<std::int64_t>(i));
  nn::Tensor x = nn::from_vector(values, {8, 8, 8, 4});
  x.requires_grad_(true);
  const preact::NormConfig train_norm{true, false, .9, 1e-5}, infer_norm{false, false, .9, 1e-5};
  auto block = [&](const preact::NormConfig& norm, double dropout) {
    return wide_resnet::WideBasicBlock(p, state, "block", x, 4, 1, norm, dropout, gen);
  };
  auto       train      = block(train_norm, .3);
  const auto infer      = block(infer_norm, .3).output.to_vector<float>();
  const auto no_dropout = block(infer_norm, 0).output.to_vector<float>();
  const auto mask_gradient = nn::grad(nn::sum(train.after_dropout), {train.before_dropout})[0].to_vector<float>();
  const auto before = train.before_dropout.to_vector<float>(), after = train.after_dropout.to_vector<float>();
  const auto shortcut = train.shortcut.to_vector<float>();
  int        positive = 0, dropped = 0;
  for (std::size_t i = 0; i < values.size(); ++i) {
    check::Near(infer[i], no_dropout[i], 0, "dropout rate does not affect inference");
    check::Near(shortcut[i], values[i], 0, "dropout leaves raw shortcut untouched");
    if (before[i] <= 1e-6f) continue;
    ++positive;
    const bool removed = after[i] == 0;
    dropped += removed;
    check::Near(after[i], removed ? 0 : before[i] / .7f, 2e-6, "inverted dropout values");
    check::Near(mask_gradient[i], removed ? 0 : 1.f / .7f, 2e-6, "inverted dropout derivatives");
  }
  check::Require(positive > 500 && std::abs(double(dropped) / positive - .3) < .06, "wide dropout empirical mask rate");
  const auto again_train = block(train_norm, .3).after_dropout.to_vector<float>();
  const auto again_infer = block(infer_norm, .3).output.to_vector<float>();
  double     difference  = 0;
  for (std::size_t i = 0; i < values.size(); ++i) {
    check::Near(again_infer[i], infer[i], 0, "repeatable wide inference");
    difference += std::abs(again_train[i] - after[i]);
  }
  check::Require(difference > 1, "training dropout draws fresh masks");
  p.set("block/conv2/W", nn::zeros({3, 3, 4, 4}));  // zero residual
  train                = block(train_norm, .3);
  const auto output    = train.output.to_vector<float>();
  const auto dx        = nn::grad(nn::sum(train.output), {x})[0].to_vector<float>();
  for (std::size_t i = 0; i < values.size(); ++i) {
    check::Near(output[i], values[i], 0, "signed identity survives residual dropout");
    check::Near(dx[i], 1, 2e-6, "unit identity derivative survives residual dropout");
  }
  std::cout << "  dropout values/derivatives, changing masks, deterministic inference and signed identity passed; removed fraction="
            << double(dropped) / positive << '\n';
}

std::int64_t MultiplyAccumulates(int depth, int width) {
  std::int64_t total = 32 * 32 * 9 * 3 * 16;
  std::int64_t in = 16, spatial = 32;
  for (int stage = 0; stage < 3; ++stage) {
    const std::int64_t out = (16 << stage) * width;
    if (stage) spatial /= 2;
    for (int block = 0; block < (depth - 4) / 6; ++block) {
      total += spatial * spatial * (9 * in * out + 9 * out * out + (in != out ? in * out : 0));
      in = out;
    }
  }
  return total + in * 10;
}

void FullChecks(int depth, int width, std::int64_t parameters, std::int64_t state_parameters) {
  nn::ParameterStore            p(17), state(17);
  nn::Generator                 gen(17);
  wide_resnet::WideResNetConfig cfg;
  cfg.depth        = depth;
  cfg.widen_factor = width;
  cfg.dropout      = 0;
  cfg.update_stats = false;
  std::vector<float> pixels(2 * 32 * 32 * 3);
  for (std::size_t i = 0; i < pixels.size(); ++i) pixels[i] = .1f * std::sin(static_cast<std::int64_t>(i) * .13f);
  const auto net = wide_resnet::WideResNet(p, state, nn::from_vector(pixels, {2, 32, 32, 3}), cfg, true, gen);
  check::Require(p.count() == parameters && state.count() == state_parameters, "canonical wide parameter/state counts");
  check::Require(int(net.blocks.size()) == (depth - 4) / 2, "wide residual block count");
  for (int stage = 0; stage < 4; ++stage) {
    const std::int64_t spatial = stage < 2 ? 32 : 32 / (1 << (stage - 1)), channels = stage == 0 ? 16 : (16 << (stage - 1)) * width;
    const auto&        shape   = (stage == 0 ? net.stem : net.stages[static_cast<std::size_t>(stage - 1)]).shape();
    check::Require(shape == nn::Shape({2, spatial, spatial, channels}), "canonical wide stage dimensions");
  }
  check::Require(net.logits.shape() == nn::Shape({2, 10}), "wide classifier dimensions");
  const auto gradients = nn::grad(nn::mean(nn::square(net.logits)), p.parameters());
  double     minimum = std::numeric_limits<double>::infinity(), maximum = 0;
  for (std::size_t i = 0; i < gradients.size(); ++i) {
    const double norm = nn::sqrt(nn::sum(nn::square(gradients[i]))).item();
    check::Require(std::isfinite(norm) && norm > 0, "finite nonzero wide derivative: " + p.names()[i]);
    minimum = std::min(minimum, norm);
    maximum = std::max(maximum, norm);
  }
  std::cout << "  WRN-" << depth << '-' << width << ": parameters=" << parameters << ", state=" << state_parameters
            << ", MACs/image=" << MultiplyAccumulates(depth, width) << "; gradient norms [" << minimum << ',' << maximum << "]"
            << std::endl;
}
}  // namespace

int main(int argc, char** argv) {
  try {
    for (int i = 1; i < argc; ++i)  // threads= also applies to the checks
      if (std::string(argv[i]).rfind("threads=", 0) == 0) nn::set_num_threads(std::stoi(std::string(argv[i]).substr(8)));
    DropoutChecks();
    FullChecks(16, 1, 175066, 928);
    FullChecks(16, 2, 691674, 1824);
    FullChecks(28, 10, 36479194, 17952);
    for (const auto& setting : std::vector<std::pair<int, double>>{{1, 0}, {2, 0}, {2, .3}}) {
      const auto [width, dropout] = setting;
      image_training::ImageTrainingConfig cfg;
      cfg.name = "wide_resnet16_" + std::to_string(width) + (dropout > 0 ? "_dropout" : "") + "_cifar10";
      cfg.maximum_fit_rows      = 12000;
      cfg.epochs                = 12;
      cfg.minimum_test_accuracy = .45;
      auto gen = std::make_shared<nn::Generator>(17);  // one dropout stream per model, persistent across steps
      const image_training::LogitsBuilder builder = [=, w = width, d = dropout](
                                                        nn::ParameterStore& p, nn::ParameterStore& state,
                                                        const nn::Tensor& image, bool training) {
        wide_resnet::WideResNetConfig model;
        model.depth        = 16;
        model.widen_factor = w;
        model.dropout      = d;
        return wide_resnet::WideResNet(p, state, image, model, training, *gen).logits;
      };
      if (image_training::RunCifar10(argc, argv, cfg, builder)) return 1;
    }
    std::cout << "All Wide ResNet checks passed.\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
