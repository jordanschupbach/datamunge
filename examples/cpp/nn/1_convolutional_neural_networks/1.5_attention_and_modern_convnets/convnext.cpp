// CMD: just run-nn convnext
// ConvNeXt: a ResNet modernized with patchify stems, 7x7 depthwise kernels, LayerNorm and GELU.
//
// Port of tflow's convnext (model: common/convnext.hpp). Check that a new
// block is nearly the identity (layer scale 1e-6) and that one output
// position of a block depends on the 7x7 input window around it. Measure
// ConvNeXt-T/S/B/L at 224 pixels from the running model's stage sizes (one
// forward pass each): 28,589,128 / 50,223,688 / 88,591,464 / 197,767,336
// learned parameters and 4.46 / 8.68 / 15.35 / 34.36G multiply-adds
// (torchvision's counts and GFLOPs). Check every derivative of ConvNeXt-T in
// training mode with layer scales set to one, as after training (at 1e-6 the
// branch derivatives are a million times smaller).
// https://arxiv.org/abs/2201.03545
// https://github.com/facebookresearch/ConvNeXt
//
// Train the smallest layout (Atto: depths 2-2-6-2, widths 40-320, 3.38M
// parameters) with layer scale on CIFAR-10, with a 2x2/2 patchify stem and
// stochastic depth 0.1; convnext_v2 trains the same layout with GRN. Use
// 12k fit / 5k validation, twelve epochs, crop/flip, Adam(.001), kernel
// L2(.0001) and the best-validation checkpoint before evaluating all 10k
// test images. The paper's recipe (AdamW, 300 epochs, heavy augmentation) is
// not reproduced. Weights and drop-path masks come from nn's generator, so
// trained numbers match TF only statistically. CIFAR-10 is read from the
// tflow dataset cache (common/paths.hpp) or data=.
// Options: data=PATH epochs=12 batch=128 lr=.001 fit=12000 augment=0|1 threads=N seed=17.

#include <cctype>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>

#include "common/checks.hpp"
#include "common/convnext.hpp"
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

void BlockChecks() {
  const int          size = 11, dim = 8;
  nn::ParameterStore p(17);
  nn::Tensor         x = Pattern({1, size, size, dim}, 1, .2f);
  double             change;
  {
    nn::NoGradGuard guard;
    change = nn::max(nn::abs(convnext::ConvNeXtBlock(p, "block", x, false, false).output - x)).item();
    p.set("block/layer_scale", nn::ones({dim}));
  }
  x.requires_grad_(true);
  const auto branch = convnext::ConvNeXtBlock(p, "block", x, false, false).branch;
  const auto g      = nn::grad(nn::sum(nn::slice(branch, {0, 5, 5, 0}, {1, 1, 1, dim})), {x})[0].to_vector<float>();
  int        reached = 0, low = size, high = -1;
  for (int r = 0; r < size; ++r)
    for (int c = 0; c < size; ++c) {
      double m = 0;
      for (int ch = 0; ch < dim; ++ch) m += std::abs(g[(r * size + c) * dim + ch]);
      if (m > 0) {
        ++reached;
        low  = std::min(low, r);
        high = std::max(high, r);
      }
    }
  check::Require(change < 1e-4, "a new block is nearly the identity");
  check::Require(reached == 49 && high - low + 1 == 7, "7x7 receptive field per block");
  std::cout << "  ConvNeXt block (8 channels): a new block changes its input by at most " << change
            << "; one output position depends on " << reached << " inputs in a " << high - low + 1 << "x"
            << high - low + 1 << " window" << std::endl;
}

void CostChecks() {
  struct Case {
    const char*  size;
    std::int64_t params, macs;
  };
  for (const auto& c : {Case{"tiny", 28589128, 4455531264}, Case{"small", 50223688, 8683712256},
                        Case{"base", 88591464, 15354729472}, Case{"large", 197767336, 34361433600}}) {
    nn::ParameterStore       p(17);
    nn::NoGradGuard          guard;
    convnext::ConvNeXtConfig cfg;
    cfg.size         = c.size;
    const auto net   = convnext::ConvNeXt(p, nn::zeros({1, 224, 224, 3}), cfg);
    const auto macs  = convnext::ConvNeXtMacs(net, convnext::ConvNeXtLayout(c.size), 1000, 4);
    check::Require(p.count() == c.params && macs == c.macs, "ConvNeXt parameters and multiply-adds");
    std::cout << "  ConvNeXt-" << char(std::toupper(c.size[0])) << ": " << p.count() << " parameters, " << std::fixed
              << std::setprecision(2) << macs / 1e9 << "G mult-adds" << std::defaultfloat << std::setprecision(6)
              << std::endl;
  }
}

void GradientChecks() {
  nn::ParameterStore p(17);
  const nn::Tensor   input = Pattern({2, 224, 224, 3}, 1, 0);
  {
    nn::NoGradGuard guard;
    convnext::ConvNeXt(p, nn::zeros({1, 32, 32, 3}));  // create the parameters
    for (const auto& name : p.names())
      if (name.find("layer_scale") != std::string::npos) p.set(name, nn::ones(p.at(name).shape()));
  }
  const auto  net   = convnext::ConvNeXt(p, input, {}, true);
  const auto  norms = mobile_checks::GradientNorms(p, nn::mean(nn::square(net.logits)));
  double      minimum = std::numeric_limits<double>::infinity(), maximum = 0;
  std::string smallest;
  for (std::size_t i = 0; i < norms.size(); ++i) {
    check::Require(std::isfinite(norms[i]), "finite ConvNeXt derivative");
    if (norms[i] < minimum) {
      minimum  = norms[i];
      smallest = p.names()[i];
    }
    maximum = std::max(maximum, norms[i]);
  }
  check::Require(minimum > 1e-6 * maximum, "clearly nonzero ConvNeXt derivatives");
  std::cout << "  ConvNeXt-T, layer scales one: " << norms.size() << " derivatives in [" << minimum << ',' << maximum
            << "] (smallest " << smallest << ")" << std::endl;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    for (int i = 1; i < argc; ++i)  // threads= also applies to the checks
      if (std::string(argv[i]).rfind("threads=", 0) == 0) nn::set_num_threads(std::stoi(std::string(argv[i]).substr(8)));
    BlockChecks();
    CostChecks();
    GradientChecks();
    image_training::ImageTrainingConfig cfg;
    cfg.name                  = "convnext_atto_cifar10";
    cfg.maximum_fit_rows      = 12000;
    cfg.epochs                = 12;
    cfg.minimum_test_accuracy = .4;
    const image_training::LogitsBuilder builder = [](nn::ParameterStore& p, nn::ParameterStore& state,
                                                     const nn::Tensor& image, bool training) {
      convnext::ConvNeXtConfig model;
      model.size        = "atto";
      model.classes     = 10;
      model.cifar_stem  = true;
      model.drop_path   = .1;
      const auto logits = convnext::ConvNeXt(p, image, model, training).logits;
      check::Require(p.count() == 3376290 && state.count() == 0, "CIFAR ConvNeXt-Atto parameter count");
      return logits;
    };
    if (image_training::RunCifar10(argc, argv, cfg, builder)) return 1;
    std::cout << "All ConvNeXt checks passed.\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
