// CMD: just run-nn cbam
// CBAM: sequential channel and spatial attention in residual blocks.
//
// Port of tflow's cbam. Check a CBAM module against a host computation: the
// shared biased MLP on the mean- and max-pooled features, summed and passed
// through a logistic gate, then the spatial gate from a 7x7 convolution over
// the channel-wise max and mean maps (with zero padding) followed by
// inference BN. Count CBAM-ResNet-50: 28,089,624 learned parameters,
// ResNet-50 plus 2,530,992 in channel MLPs and 1,600 in spatial gates (the
// paper's 28.09M). Check every derivative in training mode (moving averages
// not updated) at 224 pixels with shifted BN betas.
// https://arxiv.org/abs/1807.06521
// https://github.com/Jongchan/attention-module
//
// Train the compact ResNet-50 of resnet_bottleneck (quarter widths, 3x3
// stride-one stem, stride in the first 1x1) with CBAM in every block on
// CIFAR-10, for direct comparison with the plain and SE versions. Use 12k fit
// / 5k validation, twelve epochs, crop/flip, Adam(.001), kernel L2(.0001) and
// the best-validation checkpoint before evaluating all 10k test images.
// Weights come from nn's generator, so trained numbers match TF only
// statistically. CIFAR-10 is read from the tflow dataset cache
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
using mobile_checks::EndsWith;
using mobile_checks::Pattern;

namespace {

void ModuleChecks() {
  const int          channels = 16, reduction = 4, hidden = channels / reduction, size = 9;
  nn::ParameterStore p(17), state(17);
  const nn::Tensor   x = Pattern({1, size, size, channels}, 1, .2f);
  attention_cnn::Cbam(p, state, "cbam", x, reduction, false, .9);  // create the parameters
  {
    nn::NoGradGuard guard;
    for (const auto& name : p.names())
      if (EndsWith(name, "/b") || name.find("/bn/") != std::string::npos)
        p.set(name, Pattern(p.at(name).shape(), .3f, static_cast<float>(name.size())));
    state.set("cbam/spatial/bn/moving_mean", Pattern({1}, .2f, 1));
    state.set("cbam/spatial/bn/moving_var", nn::full({1}, 1.3));
  }
  const auto output = attention_cnn::Cbam(p, state, "cbam", x, reduction, false, .9).output.to_vector<float>();
  auto       get    = [&](nn::ParameterStore& s, const std::string& name) { return s.at(name).to_vector<float>(); };
  const auto w1 = get(p, "cbam/mlp1/W"), b1 = get(p, "cbam/mlp1/b"), w2 = get(p, "cbam/mlp2/W"),
             b2 = get(p, "cbam/mlp2/b"), sw = get(p, "cbam/spatial/W"), gamma = get(p, "cbam/spatial/bn/gamma"),
             beta = get(p, "cbam/spatial/bn/beta"), mm = get(state, "cbam/spatial/bn/moving_mean"),
             mv = get(state, "cbam/spatial/bn/moving_var");
  const auto value = x.to_vector<float>();
  auto       at    = [&](int r, int c, int ch) { return double(value[(r * size + c) * channels + ch]); };
  // Channel gate.
  std::vector<double> mean(channels, 0), peak(channels, -INFINITY), logit(channels, 0);
  for (int r = 0; r < size; ++r)
    for (int c = 0; c < size; ++c)
      for (int ch = 0; ch < channels; ++ch) {
        mean[ch] += at(r, c, ch) / (size * size);
        peak[ch] = std::max(peak[ch], at(r, c, ch));
      }
  for (const auto& pooled : {mean, peak}) {
    std::vector<double> h(hidden);
    for (int j = 0; j < hidden; ++j) {
      h[j] = b1[j];
      for (int ch = 0; ch < channels; ++ch) h[j] += pooled[ch] * w1[ch * hidden + j];
      h[j] = std::max(h[j], 0.);
    }
    for (int ch = 0; ch < channels; ++ch) {
      logit[ch] += b2[ch];
      for (int j = 0; j < hidden; ++j) logit[ch] += h[j] * w2[j * channels + ch];
    }
  }
  std::vector<double> gate(channels);
  for (int ch = 0; ch < channels; ++ch) gate[ch] = 1 / (1 + std::exp(-logit[ch]));
  // Spatial gate on the refined features.
  std::vector<double> maxmap(size * size, -INFINITY), meanmap(size * size, 0);
  for (int r = 0; r < size; ++r)
    for (int c = 0; c < size; ++c)
      for (int ch = 0; ch < channels; ++ch) {
        const double v         = at(r, c, ch) * gate[ch];
        maxmap[r * size + c]   = std::max(maxmap[r * size + c], v);
        meanmap[r * size + c] += v / channels;
      }
  const double scale = gamma[0] / std::sqrt(mv[0] + 1e-5), shift = beta[0] - mm[0] * scale;
  double       error = 0;
  for (int r = 0; r < size; ++r)
    for (int c = 0; c < size; ++c) {
      double z = 0;
      for (int dr = -3; dr <= 3; ++dr)
        for (int dc = -3; dc <= 3; ++dc) {
          const int rr = r + dr, cc = c + dc;
          if (rr < 0 || rr >= size || cc < 0 || cc >= size) continue;
          z += maxmap[rr * size + cc] * sw[((dr + 3) * 7 + dc + 3) * 2] +
               meanmap[rr * size + cc] * sw[((dr + 3) * 7 + dc + 3) * 2 + 1];
        }
      const double spatial = 1 / (1 + std::exp(-(z * scale + shift)));
      for (int ch = 0; ch < channels; ++ch)
        error = std::max(error, std::abs(output[(r * size + c) * channels + ch] - at(r, c, ch) * gate[ch] * spatial));
    }
  check::Near(error, 0, 2e-6, "CBAM channel and spatial attention");
  std::cout << "  CBAM module (16 channels, reduction 4, 9x9): output equals host-computed channel gate x spatial gate x "
               "input (error "
            << error << ")" << std::endl;
}

void FullChecks() {
  nn::ParameterStore                   p(17), state(17);
  const nn::Tensor                     input = Pattern({2, 224, 224, 3}, 1, 0);
  attention_cnn::AttentionResNetConfig cfg;
  cfg.attention    = attention_cnn::ChannelAttention::kCbam;
  cfg.update_stats = false;
  {
    nn::NoGradGuard guard;
    attention_cnn::AttentionResNet(p, state, nn::zeros({1, 64, 64, 3}), cfg);  // create the parameters
  }
  check::Require(p.count() == 28089624, "CBAM-ResNet-50 parameter count");
  mobile_checks::ShiftBetas(p, false);
  const auto  net   = attention_cnn::AttentionResNet(p, state, input, cfg, true);
  const auto  norms = mobile_checks::GradientNorms(p, nn::mean(nn::square(net.logits)));
  double      minimum = std::numeric_limits<double>::infinity(), maximum = 0;
  std::string smallest;
  for (std::size_t i = 0; i < norms.size(); ++i) {
    check::Require(std::isfinite(norms[i]), "finite CBAM-ResNet derivative");
    if (norms[i] < minimum) {
      minimum  = norms[i];
      smallest = p.names()[i];
    }
    maximum = std::max(maximum, norms[i]);
  }
  check::Require(minimum > 1e-5 * maximum, "clearly nonzero CBAM-ResNet derivatives");
  std::cout << "  CBAM-ResNet-50: " << p.count() << " parameters; training-mode gradient norms with shifted BN betas ["
            << minimum << ',' << maximum << "] (smallest " << smallest << ")" << std::endl;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    for (int i = 1; i < argc; ++i)  // threads= also applies to the checks
      if (std::string(argv[i]).rfind("threads=", 0) == 0) nn::set_num_threads(std::stoi(std::string(argv[i]).substr(8)));
    ModuleChecks();
    FullChecks();
    image_training::ImageTrainingConfig cfg;
    cfg.name                  = "cbam_resnet50_compact_cifar10";
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
      model.attention     = attention_cnn::ChannelAttention::kCbam;
      const auto logits   = attention_cnn::AttentionResNet(p, state, image, model, training).logits;
      // The plain compact ResNet-50's 1,484,186 plus 161,196 in channel MLPs and 1,600 in spatial gates.
      check::Require(p.count() == 1646982, "compact CBAM-ResNet-50 parameter count");
      return logits;
    };
    if (image_training::RunCifar10(argc, argv, cfg, builder)) return 1;
    std::cout << "All CBAM checks passed.\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
