// CMD: just run-nn sknet
// Selective kernel networks: attention over 3x3 and dilated (5x5) branches.
//
// Port of tflow's sknet. Check a selective kernel unit against its
// definition: per channel, the branch weights are a softmax (positive,
// summing to one) and the output is their weighted sum of the two branch
// outputs. Check the branches' receptive fields with derivative supports: the
// dilated 3x3 branch reaches nine taps spread over 5x5, the plain branch a
// dense 3x3; strided units sample both at the same centres. Count SKNet-50
// (ResNeXt-50 32x4d with SK units, r = 16, L = 32): 27,479,784 learned
// parameters, ResNeXt-50's 25,028,904 plus 2,450,880 (the paper's 27.5M).
// Check every derivative in training mode (moving averages not updated) at
// 224 pixels with shifted BN betas.
// https://arxiv.org/abs/1903.06586
// https://github.com/implus/SKNet
//
// Train a compact SKNet-50 on CIFAR-10: quarter widths (cardinality 32 with
// group widths 1/2/4/8), a 3x3 stride-one stem and all sixteen blocks. Use
// 12k fit / 5k validation, twelve epochs, crop/flip, Adam(.001), kernel
// L2(.0001) and the best-validation checkpoint before evaluating all 10k
// test images. Deviation: each grouped branch is one grouped, dilated
// nn::conv2d (tflow splits per group and expands the dilated kernel with
// zeros because TF's C++ Conv2D gradient ignores dilation); counts and fans
// are the same. Weights come from nn's generator, so trained numbers match TF
// only statistically. CIFAR-10 is read from the tflow dataset cache
// (common/paths.hpp) or data=.
// Options: data=PATH epochs=12 batch=128 lr=.001 fit=12000 augment=0|1 threads=N seed=17.

#include <cmath>
#include <iostream>
#include <limits>

#include "common/attention_cnn.hpp"
#include "common/checks.hpp"
#include "common/image_training.hpp"
#include "common/mobile_checks.hpp"

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
using mobile_checks::Pattern;

namespace {

void UnitChecks() {
  for (int stride : {1, 2}) {
    const int          channels = 8, size = 9, out_size = (size + stride - 1) / stride, centre = out_size / 2;
    nn::ParameterStore p(17), state(17);
    nn::Tensor         x = Pattern({1, size, size, channels}, 1, .2f);
    x.requires_grad_(true);
    const auto sk = attention_cnn::SelectiveKernel(p, state, "sk", x, channels, 2, stride, 2, 4, false, .9);
    std::vector<std::vector<float>> supports;
    for (const auto& branch : sk.branches) {
      // All channels: ReLU may silence one.
      const nn::Tensor picked = nn::slice(branch, {0, centre, centre, 0}, {1, 1, 1, channels});
      supports.push_back(nn::grad(nn::sum(picked), {x}, nn::GradOptions{true, false, false})[0].to_vector<float>());
    }
    const auto out = sk.output.to_vector<float>(), b0 = sk.branches[0].to_vector<float>(),
               b1 = sk.branches[1].to_vector<float>(), w0 = sk.weights[0].to_vector<float>(),
               w1 = sk.weights[1].to_vector<float>();
    double error = 0, sum = 0, low = 1;
    for (std::size_t i = 0; i < out.size(); ++i) {
      const std::size_t c = i % channels;
      const float       a = w0[c], b = w1[c];
      error               = std::max(error, double(std::abs(out[i] - (a * b0[i] + b * b1[i]))));
      sum                 = std::max(sum, std::abs(a + b - 1.));
      low                 = std::min({low, double(a), double(b)});
    }
    check::Require(sk.output.shape()[1] == out_size && low > 0, "SK output size and positive branch weights");
    check::Near(sum, 0, 1e-6, "SK branch weights sum to one");
    check::Near(error, 0, 1e-6, "SK output is the attention-weighted branch sum");
    std::string text;
    for (int branch = 0; branch < 2; ++branch) {
      int taps = 0, low_r = size, high_r = -1;
      for (int r = 0; r < size; ++r)
        for (int c = 0; c < size; ++c) {
          double mass = 0;
          for (int ch = 0; ch < channels; ++ch) mass += std::abs(supports[branch][(r * size + c) * channels + ch]);
          if (mass > 0) {
            ++taps;
            low_r  = std::min(low_r, r);
            high_r = std::max(high_r, r);
          }
        }
      const int span = high_r - low_r + 1;
      check::Require(taps == 9 && span == (branch ? 5 : 3) && (low_r + high_r) / 2 == centre * stride,
                     "SK branch receptive fields");
      text += (branch ? ", dilated " : "plain ") + std::to_string(taps) + " taps over " + std::to_string(span) + "x" +
              std::to_string(span);
    }
    std::cout << "  SK unit, stride " << stride << ": weights positive and summing to one, output = weighted branch sum (error "
              << error << "); centre-output supports: " << text << std::endl;
  }
}

void FullChecks() {
  nn::ParameterStore                   p(17), state(17);
  attention_cnn::AttentionResNetConfig cfg;
  cfg.cardinality  = 32;
  cfg.base_width   = 4;
  cfg.update_stats = false;
  cfg.conv2        = attention_cnn::SelectiveKernelHook(cfg.cardinality, cfg.reduction, cfg.momentum, false);
  {
    nn::NoGradGuard guard;
    attention_cnn::AttentionResNet(p, state, nn::zeros({1, 64, 64, 3}), cfg);  // create the parameters
  }
  check::Require(p.count() == 27479784, "SKNet-50 parameter count");
  mobile_checks::ShiftBetas(p, false);
  const auto  net     = attention_cnn::AttentionResNet(p, state, Pattern({2, 224, 224, 3}, 1, 0), cfg, true);
  const auto  norms   = mobile_checks::GradientNorms(p, nn::mean(nn::square(net.logits)));
  double      minimum = std::numeric_limits<double>::infinity(), maximum = 0;
  std::string smallest;
  for (std::size_t i = 0; i < norms.size(); ++i) {
    check::Require(std::isfinite(norms[i]), "finite SKNet derivative");
    if (norms[i] < minimum) {
      minimum  = norms[i];
      smallest = p.names()[i];
    }
    maximum = std::max(maximum, norms[i]);
  }
  // The stem's derivative sums over 112x112 positions while a late SK unit's
  // fc sees one pooled vector, so the norms span about seven decades here.
  check::Require(minimum > 1e-7 * maximum, "clearly nonzero SKNet derivatives");
  std::cout << "  SKNet-50: " << p.count() << " parameters; training-mode gradient norms with shifted BN betas [" << minimum
            << ',' << maximum << "] (smallest " << smallest << ")" << std::endl;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    for (int i = 1; i < argc; ++i)  // threads= also applies to the checks
      if (std::string(argv[i]).rfind("threads=", 0) == 0) nn::set_num_threads(std::stoi(std::string(argv[i]).substr(8)));
    UnitChecks();
    FullChecks();
    image_training::ImageTrainingConfig cfg;
    cfg.name                  = "sknet50_compact_cifar10";
    cfg.maximum_fit_rows      = 12000;
    cfg.epochs                = 12;
    cfg.minimum_test_accuracy = .4;
    const image_training::LogitsBuilder builder = [](nn::ParameterStore& p, nn::ParameterStore& state,
                                                     const nn::Tensor& image, bool training) {
      attention_cnn::AttentionResNetConfig model;
      model.classes       = 10;
      model.width_divisor = 4;
      model.small_stem    = true;
      model.cardinality   = 32;
      model.base_width    = 4;
      model.conv2         = attention_cnn::SelectiveKernelHook(model.cardinality, model.reduction, model.momentum);
      return attention_cnn::AttentionResNet(p, state, image, model, training).logits;
    };
    if (image_training::RunCifar10(argc, argv, cfg, builder)) return 1;
    std::cout << "All SKNet checks passed.\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
