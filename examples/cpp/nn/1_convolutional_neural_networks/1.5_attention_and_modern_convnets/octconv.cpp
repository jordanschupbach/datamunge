// CMD: just run-nn octconv
// Octave convolution: half-resolution low-frequency features.
//
// Port of tflow's octconv. Check that an octave convolution with alpha 0 is
// an ordinary convolution, and an alpha 0.25 layer against a host
// computation of its four paths (pooling and nearest upsampling included).
// Its kernels partition one ordinary kernel (same parameter count) while its
// multiply-adds shrink by exactly 1 - 3/4 alpha (2 - alpha) = 0.671875. Count
// Oct-ResNet-50 at alpha 0 / 0.125 / 0.25 / 0.5 / 0.75: always ResNet-50's
// 25,557,032 learned parameters, with multiply-adds falling from ResNet-50's
// 4,089,184,256 (alpha 0, stride in the 3x3 as in torchvision). Check
// derivatives of the compact model (training mode, moving averages not
// updated).
// https://arxiv.org/abs/1904.05049
// https://github.com/facebookresearch/OctConv
//
// Train the compact ResNet-50 of resnet_bottleneck's layout (quarter widths,
// 3x3 stride-one stem; stride here in the 3x3) as an Oct-ResNet with alpha
// 0.25 on CIFAR-10. Use 12k fit / 5k validation, twelve epochs, crop/flip,
// Adam(.001), kernel L2(.0001) and the best-validation checkpoint before
// evaluating all 10k test images. The 224-pixel counts are built on 64-pixel
// inputs (parameters do not depend on the size; multiply-adds are counted at
// the nominal 224 pixels, as tflow does). Weights come from nn's generator,
// so trained numbers match TF only statistically. CIFAR-10 is read from the
// tflow dataset cache (common/paths.hpp) or data=.
// Options: data=PATH epochs=12 batch=128 lr=.001 fit=12000 augment=0|1 threads=N seed=17.

#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>

#include "common/checks.hpp"
#include "common/image_training.hpp"
#include "common/mobile_checks.hpp"
#include "common/octconv.hpp"

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

using Image = std::vector<std::vector<std::vector<double>>>;  // [row][column][channel]
Image Convolve(const Image& x, const std::vector<float>& w, int from, int to) {
  const int size = int(x.size());
  Image     y(size, std::vector<std::vector<double>>(size, std::vector<double>(to, 0)));
  for (int r = 0; r < size; ++r)
    for (int c = 0; c < size; ++c)
      for (int dr = -1; dr <= 1; ++dr)
        for (int dc = -1; dc <= 1; ++dc) {
          const int rr = r + dr, cc = c + dc;
          if (rr < 0 || rr >= size || cc < 0 || cc >= size) continue;
          for (int i = 0; i < from; ++i)
            for (int o = 0; o < to; ++o) y[r][c][o] += x[rr][cc][i] * w[(((dr + 1) * 3 + dc + 1) * from + i) * to + o];
        }
  return y;
}
Image Pool(const Image& x) {
  const int size = int(x.size()) / 2, channels = int(x[0][0].size());
  Image     y(size, std::vector<std::vector<double>>(size, std::vector<double>(channels, 0)));
  for (int r = 0; r < size; ++r)
    for (int c = 0; c < size; ++c)
      for (int ch = 0; ch < channels; ++ch)
        for (int dr = 0; dr < 2; ++dr)
          for (int dc = 0; dc < 2; ++dc) y[r][c][ch] += x[2 * r + dr][2 * c + dc][ch] / 4;
  return y;
}

void LayerChecks() {
  const int    size = 8, channels = 16;
  const double alpha = .25;
  const int    low = octconv::LowChannels(channels, alpha), high = channels - low;
  {
    // alpha 0: an ordinary convolution with the same kernel.
    nn::ParameterStore  p(17);
    const nn::Tensor    x = Pattern({1, size, size, channels}, 1, .2f);
    octconv::OctaveMaps in;
    in.high          = x;
    in.high_channels = channels;
    const auto        y = octconv::OctaveConv(p, "oct", in, channels, 0, 3, 1, size);
    nn::Conv2DOptions o = nn::conv_options(1, nn::Padding::Explicit);
    o.explicit_paddings = {1, 1, 1, 1};
    const auto a = y.high.to_vector<float>(), b = nn::conv2d(x, p.at("oct/hh/W"), o).to_vector<float>();
    double     error = 0;
    for (std::size_t i = 0; i < a.size(); ++i) error = std::max(error, double(std::abs(a[i] - b[i])));
    check::Require(!y.low.defined() && y.low_channels == 0, "alpha 0 has no low-frequency output");
    check::Near(error, 0, 0, "alpha 0 octave convolution is an ordinary convolution");
  }
  nn::ParameterStore  p(17);
  const nn::Tensor    vh = Pattern({1, size, size, high}, 1, .2f), vl = Pattern({1, size / 2, size / 2, low}, 1, .9f);
  octconv::OctaveMaps in{vh, vl, high, low};
  octconv::OctaveCost cost;
  const auto          y  = octconv::OctaveConv(p, "oct", in, channels, alpha, 3, 1, size, &cost);
  const auto          yh = y.high.to_vector<float>(), yl = y.low.to_vector<float>(), h = vh.to_vector<float>(),
             l = vl.to_vector<float>();
  Image H(size, std::vector<std::vector<double>>(size, std::vector<double>(high))),
      L(size / 2, std::vector<std::vector<double>>(size / 2, std::vector<double>(low)));
  for (int r = 0; r < size; ++r)
    for (int c = 0; c < size; ++c)
      for (int ch = 0; ch < high; ++ch) H[r][c][ch] = h[(r * size + c) * high + ch];
  for (int r = 0; r < size / 2; ++r)
    for (int c = 0; c < size / 2; ++c)
      for (int ch = 0; ch < low; ++ch) L[r][c][ch] = l[(r * size / 2 + c) * low + ch];
  const auto hh = Convolve(H, p.at("oct/hh/W").to_vector<float>(), high, high),
             lh = Convolve(L, p.at("oct/lh/W").to_vector<float>(), low, high),
             hl = Convolve(Pool(H), p.at("oct/hl/W").to_vector<float>(), high, low),
             ll = Convolve(L, p.at("oct/ll/W").to_vector<float>(), low, low);
  double error = 0;
  for (int r = 0; r < size; ++r)
    for (int c = 0; c < size; ++c)
      for (int o = 0; o < high; ++o)
        error = std::max(error, std::abs(yh[(r * size + c) * high + o] - (hh[r][c][o] + lh[r / 2][c / 2][o])));
  for (int r = 0; r < size / 2; ++r)
    for (int c = 0; c < size / 2; ++c)
      for (int o = 0; o < low; ++o)
        error = std::max(error, std::abs(yl[(r * size / 2 + c) * low + o] - (hl[r][c][o] + ll[r][c][o])));
  const std::int64_t plain = std::int64_t(size) * size * 9 * channels * channels;
  const double       ratio = double(cost.macs) / plain, formula = 1 - .75 * alpha * (2 - alpha);
  check::Near(error, 0, 1e-5, "octave convolution paths");
  check::Require(p.count() == 9 * channels * channels, "octave kernels partition an ordinary kernel");
  check::Near(ratio, formula, 1e-12, "octave multiply-add ratio");
  std::cout << "  octave convolution 16->16, alpha 0.25, 8x8 (12 high / 4 low channels): paths match the host (error "
            << error << "), " << p.count() << " weights as an ordinary 3x3, multiply-adds x" << ratio
            << " = 1 - 3/4 alpha (2 - alpha); alpha 0 equals Conv2D" << std::endl;
}

void CostChecks() {
  for (double alpha : {0., .125, .25, .5, .75}) {
    nn::ParameterStore       p(17), state(17);
    octconv::OctResNetConfig cfg;
    cfg.alpha = alpha;
    std::int64_t macs;
    {
      nn::NoGradGuard guard;
      macs = octconv::OctResNet(p, state, nn::zeros({1, 64, 64, 3}), cfg).macs;  // MACs at cfg.image_size = 224
    }
    check::Require(p.count() == 25557032 && (alpha > 0 || macs == 4089184256),
                   "Oct-ResNet-50 parameters (and alpha 0 multiply-adds)");
    std::cout << "  Oct-ResNet-50, alpha " << alpha << ": " << p.count() << " parameters, " << std::fixed
              << std::setprecision(3) << macs / 1e9 << "G mult-adds (" << std::setprecision(1) << 100. * macs / 4089184256
              << "% of ResNet-50)" << std::defaultfloat << std::setprecision(6) << std::endl;
  }
}

void GradientChecks() {
  nn::ParameterStore       p(17), state(17);
  octconv::OctResNetConfig cfg;
  cfg.classes       = 10;
  cfg.width_divisor = 4;
  cfg.small_stem    = true;
  cfg.image_size    = 32;
  cfg.update_stats  = false;
  const auto  net     = octconv::OctResNet(p, state, Pattern({2, 32, 32, 3}, 1, 0), cfg, true);
  const auto  norms   = mobile_checks::GradientNorms(p, nn::mean(nn::square(net.logits)));
  double      minimum = std::numeric_limits<double>::infinity(), maximum = 0;
  std::string smallest;
  for (std::size_t i = 0; i < norms.size(); ++i) {
    check::Require(std::isfinite(norms[i]), "finite Oct-ResNet derivative");
    if (norms[i] < minimum) {
      minimum  = norms[i];
      smallest = p.names()[i];
    }
    maximum = std::max(maximum, norms[i]);
  }
  check::Require(minimum > 1e-6 * maximum, "clearly nonzero Oct-ResNet derivatives");
  std::cout << "  compact Oct-ResNet-50 (alpha 0.25): " << p.count() << " parameters, " << net.macs / 1e6
            << "M mult-adds at 32 px; derivatives in [" << minimum << ',' << maximum << "] (smallest " << smallest << ")"
            << std::endl;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    for (int i = 1; i < argc; ++i)  // threads= also applies to the checks
      if (std::string(argv[i]).rfind("threads=", 0) == 0) nn::set_num_threads(std::stoi(std::string(argv[i]).substr(8)));
    LayerChecks();
    CostChecks();
    GradientChecks();
    image_training::ImageTrainingConfig cfg;
    cfg.name                  = "oct_resnet50_compact_cifar10";
    cfg.maximum_fit_rows      = 12000;
    cfg.epochs                = 12;
    cfg.minimum_test_accuracy = .4;
    const image_training::LogitsBuilder builder = [](nn::ParameterStore& p, nn::ParameterStore& state,
                                                     const nn::Tensor& image, bool training) {
      octconv::OctResNetConfig model;
      model.classes       = 10;
      model.width_divisor = 4;
      model.small_stem    = true;
      model.image_size    = 32;
      const auto logits   = octconv::OctResNet(p, state, image, model, training).logits;
      check::Require(p.count() == 1484186, "compact Oct-ResNet-50 has the compact ResNet-50's parameters");
      return logits;
    };
    if (image_training::RunCifar10(argc, argv, cfg, builder)) return 1;
    std::cout << "All octave convolution checks passed.\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
