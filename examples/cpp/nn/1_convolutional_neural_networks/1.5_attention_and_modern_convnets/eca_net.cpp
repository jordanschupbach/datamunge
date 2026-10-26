// CMD: just run-nn eca_net
// ECA-Net: efficient channel attention by a 1-D convolution across channels.
//
// Port of tflow's eca_net. Check an ECA module against a host computation of
// its global mean, a bias-free 1-D convolution across channels with zero
// padding, and a logistic gate; unlike SE there is no dimensionality
// reduction and only k weights. Check the paper's adaptive kernel size
// k = |log2(C)/2 + 1/2|_odd (3/5/5/5 for 64-512 channels, 5/5/5/7 for
// ResNet-50's 256-2048) and count ECA-ResNet-50: ResNet-50's 25,557,032 plus
// 48 weights with the release's fixed k = 3, or plus 86 with adaptive k.
// Check every derivative in training mode (moving averages not updated) at
// 224 pixels with shifted BN betas.
// https://arxiv.org/abs/1910.03151
// https://github.com/BangguWu/ECANet
//
// Train the compact ResNet-50 of resnet_bottleneck (quarter widths, 3x3
// stride-one stem, stride in the first 1x1) with adaptive-k ECA on CIFAR-10,
// for comparison with the plain, SE and CBAM versions. Use 12k fit / 5k
// validation, twelve epochs, crop/flip, Adam(.001), kernel L2(.0001) and the
// best-validation checkpoint before evaluating all 10k test images. Weights
// come from nn's generator, so trained numbers match TF only statistically.
// CIFAR-10 is read from the tflow dataset cache (common/paths.hpp) or data=.
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

void ModuleChecks() {
  const int          channels = 24, kernel = 5;
  nn::ParameterStore p(17);
  const nn::Tensor   x      = Pattern({2, 4, 4, channels}, 1, .2f);
  const auto         output = attention_cnn::Eca(p, "eca", x, kernel).output.to_vector<float>();
  const auto         w = p.at("eca/W").to_vector<float>(), value = x.to_vector<float>();
  double             error = 0;
  for (int n = 0; n < 2; ++n) {
    std::vector<double> mean(channels, 0);
    for (int i = 0; i < 16; ++i)
      for (int c = 0; c < channels; ++c) mean[c] += value[(n * 16 + i) * channels + c] / 16.;
    for (int c = 0; c < channels; ++c) {
      double z = 0;
      for (int j = 0; j < kernel; ++j) {
        const int source = c + j - kernel / 2;
        if (source >= 0 && source < channels) z += w[j] * mean[source];
      }
      const double gate = 1 / (1 + std::exp(-z));
      for (int i = 0; i < 16; ++i) {
        const std::size_t k = (n * 16 + i) * channels + c;
        error               = std::max(error, std::abs(output[k] - gate * value[k]));
      }
    }
  }
  check::Near(error, 0, 1e-6, "ECA channel attention");
  const std::vector<int> expected = {3, 5, 5, 5, 5, 5, 5, 7};
  std::vector<int>       sizes;
  for (int c : {64, 128, 256, 512, 256, 512, 1024, 2048}) sizes.push_back(attention_cnn::EcaKernel(c));
  check::Require(sizes == expected, "adaptive ECA kernel sizes");
  std::cout << "  ECA module (24 channels, k=5): output equals the host-computed 1-D convolution gate (error " << error
            << "); adaptive k for 64/128/256/512 channels = 3/5/5/5, for 256/512/1024/2048 = 5/5/5/7" << std::endl;
}

void FullChecks() {
  for (bool adaptive : {false, true}) {
    nn::ParameterStore                   p(17), state(17);
    attention_cnn::AttentionResNetConfig cfg;
    cfg.attention    = attention_cnn::ChannelAttention::kEca;
    cfg.adaptive_eca = adaptive;
    cfg.update_stats = false;
    {
      nn::NoGradGuard guard;
      attention_cnn::AttentionResNet(p, state, nn::zeros({1, 64, 64, 3}), cfg);  // create the parameters
    }
    check::Require(p.count() == (adaptive ? 25557118 : 25557080), "ECA-ResNet-50 parameter count");
    if (!adaptive) {
      std::cout << "  ECA-ResNet-50, k=3 (release): " << p.count() << " parameters (+" << p.count() - 25557032 << ")"
                << std::endl;
      continue;
    }
    mobile_checks::ShiftBetas(p, false);
    const auto  net   = attention_cnn::AttentionResNet(p, state, Pattern({2, 224, 224, 3}, 1, 0), cfg, true);
    const auto  norms = mobile_checks::GradientNorms(p, nn::mean(nn::square(net.logits)));
    double      minimum = std::numeric_limits<double>::infinity(), maximum = 0;
    std::string smallest;
    for (std::size_t i = 0; i < norms.size(); ++i) {
      check::Require(std::isfinite(norms[i]), "finite ECA-ResNet derivative");
      if (norms[i] < minimum) {
        minimum  = norms[i];
        smallest = p.names()[i];
      }
      maximum = std::max(maximum, norms[i]);
    }
    check::Require(minimum > 1e-5 * maximum, "clearly nonzero ECA-ResNet derivatives");
    std::cout << "  ECA-ResNet-50, adaptive k: " << p.count() << " parameters (+" << p.count() - 25557032
              << "); training-mode gradient norms with shifted BN betas [" << minimum << ',' << maximum << "] (smallest "
              << smallest << ")" << std::endl;
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    for (int i = 1; i < argc; ++i)  // threads= also applies to the checks
      if (std::string(argv[i]).rfind("threads=", 0) == 0) nn::set_num_threads(std::stoi(std::string(argv[i]).substr(8)));
    ModuleChecks();
    FullChecks();
    image_training::ImageTrainingConfig cfg;
    cfg.name                  = "eca_resnet50_compact_cifar10";
    cfg.maximum_fit_rows      = 12000;
    cfg.epochs                = 12;
    cfg.minimum_test_accuracy = .4;
    const image_training::LogitsBuilder builder = [](nn::ParameterStore& p, nn::ParameterStore& state,
                                                     const nn::Tensor& image, bool training) {
      attention_cnn::AttentionResNetConfig model;
      model.classes       = 10;
      model.width_divisor = 4;
      model.small_stem    = true;
      model.stride_on_3x3 = false;
      model.attention     = attention_cnn::ChannelAttention::kEca;
      model.adaptive_eca  = true;
      const auto logits   = attention_cnn::AttentionResNet(p, state, image, model, training).logits;
      // The plain compact ResNet-50's 1,484,186 plus 74 ECA weights (k = 3/5/5/5 by stage).
      check::Require(p.count() == 1484260, "compact ECA-ResNet-50 parameter count");
      return logits;
    };
    if (image_training::RunCifar10(argc, argv, cfg, builder)) return 1;
    std::cout << "All ECA-Net checks passed.\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
