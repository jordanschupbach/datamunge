// CMD: just run-nn pyramidnet
// PyramidNet: residual units whose widths grow additively, unit by unit.
//
// Port of tflow's pyramidnet. Follow the released Torch code: every unit adds
// alpha/N to a running width (rounded half up), units are BN/conv/BN/ReLU/conv/BN
// (or the bottleneck equivalent), nothing follows the addition, and shortcuts
// are average pooling plus zero channels, so they have no parameters. Check
// the width schedule. With the last BN zeroed, check that basic and
// bottleneck units at stride one and two return exactly the pooled,
// zero-padded input (negative values included, unlike post-activation
// ResNets) and that input derivatives are the identity or the 1/4 pooling
// spread. Count eight CIFAR models and ImageNet PyramidNet-200 (alpha
// 300/450). The paper's table matches these counts once BN scale/shift
// parameters are excluded. Parameter counts do not depend on the image size,
// so the ImageNet count-only models are built on 64-pixel images (tflow only
// builds graphs). Check every derivative of PyramidNet-272 (alpha 200) in
// training mode (moving averages not updated): all are nonzero except the
// shifts of the stem BN and of each unit's first and last BN, which only add
// per-channel constants that a later training-mode BN removes.
// https://arxiv.org/abs/1610.02915
// https://github.com/jhkim89/PyramidNet/blob/master/addpyramidnet.lua
//
// Train the paper's full-size PyramidNet-110 (alpha 48, basic units, 1.77M
// parameters) on CIFAR-10. Use 12k fit / 5k validation, twelve epochs,
// crop/flip, Adam(.001), kernel L2(.0001) and the best-validation checkpoint
// before evaluating all 10k test images. The paper trains 300 epochs of SGD
// on all 50k images; this short CPU run does not reproduce its error rates.
// CIFAR-10 is read from the tflow dataset cache (common/paths.hpp) or data=.
// Options: data=PATH epochs=12 batch=128 lr=.001 fit=12000 augment=0|1 threads=N seed=17.

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>

#include "common/checks.hpp"
#include "common/image_training.hpp"
#include "common/pyramidnet.hpp"

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

std::vector<float> PatternValues(std::int64_t n, float scale, float phase) {
  std::vector<float> v(static_cast<std::size_t>(n));
  for (std::int64_t i = 0; i < n; ++i) v[static_cast<std::size_t>(i)] = scale * std::sin(.37f * static_cast<float>(i) + phase);
  return v;
}
nn::Tensor Pattern(const nn::Shape& shape, float scale, float phase) {
  std::int64_t n = 1;
  for (auto d : shape) n *= d;
  return nn::from_vector(PatternValues(n, scale, phase), shape);
}

void WidthChecks() {
  pyramidnet::PyramidNetConfig cfg;  // 110 layers, alpha 48, basic units
  const auto                   widths = cfg.UnitWidths();
  check::Require(widths.size() == 54 && widths[0] == 17 && widths[1] == 18 && widths[2] == 19 && widths.back() == 64,
                 "PyramidNet-110 (alpha 48) width schedule");
  for (std::size_t i = 1; i < widths.size(); ++i)
    check::Require(widths[i] >= widths[i - 1] && widths[i] - widths[i - 1] <= 1, "additive unit-by-unit widening");
  pyramidnet::PyramidNetConfig deep;
  deep.depth             = 272;
  deep.alpha             = 200;
  deep.bottleneck        = true;
  const auto deep_widths = deep.UnitWidths();
  check::Require(deep_widths.size() == 90 && deep_widths.back() == 216, "PyramidNet-272 (alpha 200) width schedule");
  std::cout << "  PyramidNet-110 alpha 48 widths: " << widths[0] << ',' << widths[1] << ',' << widths[2] << ",...,"
            << widths[17] << " | " << widths[18] << ",...," << widths[35] << " | " << widths[36] << ",...," << widths.back()
            << "; PyramidNet-272 alpha 200 ends at " << deep_widths.back() << "x4=" << 4 * deep_widths.back() << std::endl;
}

// Zero last BN: output is the parameter-free shortcut; derivatives are exact.
void UnitChecks() {
  for (bool bottleneck : {false, true})
    for (int stride : {1, 2}) {
      nn::ParameterStore p(17), state(17);
      const int          planes = bottleneck ? 2 : 5, out_channels = bottleneck ? 8 : 5, spatial = 6 / stride;
      const auto         xv = PatternValues(2 * 6 * 6 * 3, 1, .9f);
      nn::Tensor         x  = nn::from_vector(xv, {2, 6, 6, 3});
      {
        nn::NoGradGuard guard;
        pyramidnet::PyramidBlock(p, state, "unit", x, planes, bottleneck, stride);
        const std::string last = bottleneck ? "unit/bn4/" : "unit/bn3/";
        for (const auto* part : {"gamma", "beta"}) p.set(last + part, nn::zeros({out_channels}));
      }
      const auto weights = Pattern({2, spatial, spatial, out_channels}, 1, .4f);
      x.requires_grad_(true);
      const auto unit     = pyramidnet::PyramidBlock(p, state, "unit", x, planes, bottleneck, stride);
      const auto gradient = nn::grad(nn::sum(unit.output * weights), {x})[0].to_vector<float>();
      check::Require(unit.output.shape() == nn::Shape({2, spatial, spatial, out_channels}), "pyramidal unit output dimensions");
      const auto y = unit.output.to_vector<float>(), w = weights.to_vector<float>();
      auto xt = [&](int b, int r, int c, int ch) { return xv[static_cast<std::size_t>(((b * 6 + r) * 6 + c) * 3 + ch)]; };
      auto at = [&](int b, int r, int c, int ch) {
        return static_cast<std::size_t>(((b * spatial + r) * spatial + c) * out_channels + ch);
      };
      const int k = stride;
      double    output_error = 0, gradient_error = 0;
      int       negatives = 0;
      for (int b = 0; b < 2; ++b)
        for (int r = 0; r < spatial; ++r)
          for (int c = 0; c < spatial; ++c)
            for (int ch = 0; ch < out_channels; ++ch) {
              double expected = 0;
              if (ch < 3)
                for (int dr = 0; dr < k; ++dr)
                  for (int dc = 0; dc < k; ++dc) expected += xt(b, r * k + dr, c * k + dc, ch) / double(k * k);
              if (expected < 0) negatives += y[at(b, r, c, ch)] < 0;
              output_error = std::max(output_error, std::abs(y[at(b, r, c, ch)] - expected));
            }
      for (int b = 0; b < 2; ++b)
        for (int r = 0; r < 6; ++r)
          for (int c = 0; c < 6; ++c)
            for (int ch = 0; ch < 3; ++ch)
              gradient_error = std::max(gradient_error, std::abs(gradient[static_cast<std::size_t>(((b * 6 + r) * 6 + c) * 3 + ch)] -
                                                                 w[at(b, r / k, c / k, ch)] / double(k * k)));
      check::Near(output_error, 0, 1e-6, "zero-residual unit returns the pooled, zero-padded input");
      check::Near(gradient_error, 0, 1e-6, "identity or pooling-spread input derivatives");
      check::Require(negatives > 0, "negative shortcut values pass without post-addition ReLU");
      std::cout << "  " << (bottleneck ? "bottleneck" : "basic") << " unit, stride " << stride << ": 3->" << out_channels
                << " channels, output error=" << output_error << ", derivative error=" << gradient_error
                << ", negative outputs kept=" << negatives << std::endl;
    }
}

void FullChecks() {
  struct Case {
    int          depth;
    double       alpha;
    bool         bottleneck, imagenet;
    std::int64_t params, state;
    double       paper;
  };
  const std::vector<Case> cases = {{110, 48, false, false, 1772706, 13168, 1.7},
                                   {110, 84, false, false, 3904446, 19108, 3.8},
                                   {110, 270, false, false, 28485477, 49798, 28.3},
                                   {164, 48, true, false, 1715442, 43744, 1.7},
                                   {164, 270, true, false, 27216021, 165844, 27.0},
                                   {200, 240, true, false, 26752702, 181984, 26.6},
                                   {236, 220, true, false, 26969046, 198824, 26.8},
                                   {272, 200, true, false, 26210842, 210864, 26.0},
                                   {200, 300, true, true, 62368168, 285736, 62.1},
                                   {200, 450, true, true, 116775213, 386236, 116.4}};
  for (const auto& c : cases) {
    nn::ParameterStore           p(17), state(17);
    pyramidnet::PyramidNetConfig cfg;
    cfg.depth      = c.depth;
    cfg.alpha      = c.alpha;
    cfg.bottleneck = c.bottleneck;
    cfg.imagenet   = c.imagenet;
    cfg.classes    = c.imagenet ? 1000 : 10;
    {
      nn::NoGradGuard guard;
      const int       size = c.imagenet ? 64 : 32;
      pyramidnet::PyramidNet(p, state, Pattern({1, size, size, 3}, .5f, 0), cfg);
    }
    check::Require(p.count() == c.params && state.count() == c.state, "PyramidNet parameter/state counts");
    // Running state holds a mean and variance per BN channel, as many as BN gamma/beta.
    const double without_bn = double(p.count() - state.count()) / 1e6;
    std::cout << "  " << (c.imagenet ? "ImageNet" : "CIFAR-10") << " PyramidNet-" << c.depth << " alpha " << c.alpha
              << (c.bottleneck ? " bottleneck" : " basic") << ": parameters=" << p.count() << ", state=" << state.count()
              << ", without BN scale/shift=" << std::fixed << std::setprecision(2) << without_bn << "M (paper "
              << std::setprecision(1) << c.paper << "M)" << std::defaultfloat << std::setprecision(6) << std::endl;
  }
  nn::ParameterStore           p(17), state(17);
  pyramidnet::PyramidNetConfig cfg;
  cfg.depth        = 272;
  cfg.alpha        = 200;
  cfg.bottleneck   = true;
  cfg.update_stats = false;
  const auto         net        = pyramidnet::PyramidNet(p, state, Pattern({2, 32, 32, 3}, .5f, 0), cfg, true);
  const std::int64_t spatial[3] = {32, 16, 8}, channels[3] = {net.channels[29], net.channels[59], net.channels[89]};
  for (int stage = 0; stage < 3; ++stage)
    check::Require(net.stages[stage].shape() == nn::Shape({2, spatial[stage], spatial[stage], channels[stage]}),
                   "PyramidNet stage dimensions");
  check::Require(channels[2] == 864 && net.logits.shape() == nn::Shape({2, 10}), "PyramidNet-272 final width and classifier");
  // Training-mode BN removes any per-channel constant from its input. The
  // shifts of the stem BN and of every unit's last BN add such constants to
  // the residual stream, whose consumers are all BNs; a bottleneck's first BN
  // feeds an unpadded 1x1 convolution, which keeps its shift constant for the
  // next BN. Those derivatives vanish analytically; every other one must not.
  const auto gradients = nn::grad(nn::mean(nn::square(net.logits)), p.parameters());
  std::vector<double> norms;
  double              minimum = std::numeric_limits<double>::infinity(), maximum = 0, redundant = 0;
  int                 redundant_count = 0;
  for (const auto& g : gradients) {
    norms.push_back(nn::sqrt(nn::sum(nn::square(g))).item());
    maximum = std::max(maximum, norms.back());
  }
  const auto& names = p.names();
  for (std::size_t i = 0; i < norms.size(); ++i) {
    const double norm = norms[i];
    const auto&  name = names[i];
    check::Require(std::isfinite(norm), "finite PyramidNet derivative: " + name);
    auto ends = [&](const std::string& suffix) {
      return name.size() > suffix.size() && name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0;
    };
    if (name == "stem/bn/beta" || ends("/bn1/beta") || ends("/bn4/beta")) {
      redundant = std::max(redundant, norm);
      ++redundant_count;
    } else {
      minimum = std::min(minimum, norm);
    }
  }
  check::Require(redundant_count == 181 && redundant < 1e-6 * maximum, "vanishing derivatives of BN shifts removed downstream");
  check::Require(minimum > 1e-5 * maximum, "nonzero derivatives for every other PyramidNet parameter");
  std::cout << "  PyramidNet-272 alpha 200: stage outputs " << channels[0] << '/' << channels[1] << '/' << channels[2]
            << " channels at 32/16/8; training-mode gradient norms [" << minimum << ',' << maximum << "]; " << redundant_count
            << " stem/first/last BN shift derivatives vanish (largest " << redundant << ")" << std::endl;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    for (int i = 1; i < argc; ++i)  // threads= also applies to the checks
      if (std::string(argv[i]).rfind("threads=", 0) == 0) nn::set_num_threads(std::stoi(std::string(argv[i]).substr(8)));
    WidthChecks();
    UnitChecks();
    FullChecks();
    image_training::ImageTrainingConfig cfg;
    cfg.name                  = "pyramidnet110_a48_cifar10";
    cfg.maximum_fit_rows      = 12000;
    cfg.epochs                = 12;
    cfg.minimum_test_accuracy = .45;
    const image_training::LogitsBuilder builder = [](nn::ParameterStore& p, nn::ParameterStore& state,
                                                     const nn::Tensor& image, bool training) {
      const auto logits = pyramidnet::PyramidNet(p, state, image, {}, training).logits;
      check::Require(p.count() == 1772706 && state.count() == 13168, "PyramidNet-110 parameter/state counts");
      return logits;
    };
    if (image_training::RunCifar10(argc, argv, cfg, builder)) return 1;
    std::cout << "All PyramidNet checks passed.\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
