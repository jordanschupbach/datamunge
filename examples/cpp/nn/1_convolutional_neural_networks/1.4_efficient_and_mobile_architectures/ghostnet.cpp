// CMD: just run-nn ghostnet
// GhostNet: feature maps from cheap depthwise operations.
//
// Port of tflow's ghostnet. Check the paper's speed-up and compression ratio
// for a ghost module against an ordinary convolution with the same output
// (Eqs. 4 and 5, s*c*k^2 / (c*k^2 + (s-1)*d^2) for a k x k primary and d x d
// cheap kernel), exactly from measured multiply-adds and weights. Check that
// a ghost module's second half is the depthwise image of its first half.
// Measure widths 0.5/1/1.3: 2,587,292 / 5,182,508 / 7,358,820 learned
// parameters and 42.3 / 141.2 / 227.0M multiply-adds (the paper's 2.6 / 5.2 /
// 7.3M and 42 / 141 / 226M). Check every derivative in training mode (moving
// averages not updated) with shifted BN betas and SE biases: a BN shift
// vanishes exactly when everything it reaches linearly is a 1x1 convolution
// followed by BN. That holds for the depthwise BN of every projection
// shortcut, of strided blocks without SE, and for shifts added to the
// residual stream when no later depthwise shortcut reads it; all other
// derivatives are clearly nonzero.
// https://arxiv.org/abs/1911.11907
// https://github.com/huawei-noah/Efficient-AI-Backbones/tree/master/ghostnet_pytorch
//
// Train width 1 on CIFAR-10 with a stride-one stem (all sixteen bottlenecks
// and four remaining downsamplings). Use 12k fit / 5k validation, twelve
// epochs, crop/flip, Adam(.001), kernel L2(.0001) and the best-validation
// checkpoint before evaluating all 10k test images. Initialization and
// dropout use nn's generator, so trained numbers match TF only
// statistically. CIFAR-10 is read from the tflow dataset cache
// (common/paths.hpp) or data=.
// Options: data=PATH epochs=12 batch=128 lr=.001 fit=12000 augment=0|1 threads=N seed=17.

#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <set>

#include "common/checks.hpp"
#include "common/ghostnet.hpp"
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
  const int                           in = 160, out = 960, size = 7;
  nn::ParameterStore                  p(17), state(17);
  std::vector<mobilenet::MobileLayer> ghost_layers, plain_layers;
  mobilenet::MobileContext            ghost_context{p, state, ghost_layers, false}, plain_context{p, state, plain_layers, false};
  nn::NoGradGuard                     guard;
  const nn::Tensor                    x     = Pattern({1, size, size, in}, 1, .2f);
  const auto                          ghost = ghostnet::GhostModule(ghost_context, "ghost", x, out, true);
  ghostnet::GhostConvBN(plain_context, "plain", x, out, 1, 1, false, true);
  // The ghost half recomputed from the intrinsic half with the module's own weights.
  const std::string cheap    = "ghost/cheap";
  const nn::Tensor  filtered = nn::depthwise_conv2d(ghostnet::PadSpatial(ghost.intrinsic, 1), p.at(cheap + "/W"),
                                                    nn::conv_options(1, nn::Padding::Valid));
  const nn::Tensor normalized = (filtered - state.at(cheap + "/bn/moving_mean")) *
                                    (1.0 / nn::sqrt(state.at(cheap + "/bn/moving_var") + 1e-5)) * p.at(cheap + "/bn/gamma") +
                                p.at(cheap + "/bn/beta");
  const auto result = ghost.output.to_vector<float>(), expected = nn::relu(normalized).to_vector<float>();
  double     error  = 0;
  for (std::int64_t pixel = 0; pixel < size * size; ++pixel)
    for (int ch = 0; ch < out / 2; ++ch)
      error = std::max(error, double(std::abs(result[pixel * out + out / 2 + ch] - expected[pixel * out / 2 + ch])));
  std::int64_t ghost_macs = 0, plain_macs = 0, ghost_weights = 0, plain_weights = 0;
  for (const auto& l : ghost_layers) {
    ghost_macs += l.Macs();
    ghost_weights += p.at(l.name + "/W").numel();
  }
  for (const auto& l : plain_layers) {
    plain_macs += l.Macs();
    plain_weights += p.at(l.name + "/W").numel();
  }
  const int    sratio = 2, k = 1, d = 3;
  const double formula = double(sratio) * in * k * k / (in * k * k + (sratio - 1) * d * d);
  check::Near(error, 0, 1e-5, "ghost maps are the depthwise image of the intrinsic maps");
  check::Near(double(plain_macs) / ghost_macs, formula, 1e-12, "ghost module speed-up ratio");
  check::Near(double(plain_weights) / ghost_weights, formula, 1e-12, "ghost module compression ratio");
  std::cout << "  ghost module 160->960 at 7x7: speed-up " << double(plain_macs) / ghost_macs << " and compression "
            << double(plain_weights) / ghost_weights << " = s c k^2 / (c k^2 + (s-1) d^2) = " << formula
            << "; ghost half equals the depthwise image of the intrinsic half (error " << error << ")" << std::endl;
}

void CostChecks() {
  struct Case {
    double       width;
    std::int64_t params, state, macs;
  };
  for (const auto& c : {Case{.5, 2587292, 10984, 42328152}, Case{1, 5182508, 21968, 141150512},
                        Case{1.3, 7358820, 28536, 227003256}}) {
    nn::ParameterStore       p(17), state(17);
    nn::NoGradGuard          guard;
    ghostnet::GhostNetConfig cfg;
    cfg.width       = c.width;
    const auto net  = ghostnet::GhostNet(p, state, nn::zeros({1, 224, 224, 3}), cfg);
    const auto macs = net.Macs();
    check::Require(p.count() == c.params && state.count() == c.state && macs == c.macs, "GhostNet parameters and multiply-adds");
    std::cout << "  GhostNet " << c.width << "x: " << p.count() << " parameters, " << std::fixed << std::setprecision(1)
              << macs / 1e6 << "M mult-adds" << std::defaultfloat << std::setprecision(6) << std::endl;
  }
}

void GradientChecks() {
  nn::ParameterStore       p(17), state(17);
  ghostnet::GhostNetConfig cfg;
  cfg.update_stats   = false;
  const nn::Tensor x = Pattern({2, 224, 224, 3}, 1, 0);
  // BN shifts added to the stream: each block's linear cheap ghost maps and
  // projection shortcut. They vanish if every consumer is 1x1 conv + BN.
  const auto        blocks = ghostnet::GhostNetBlocks();
  std::vector<bool> identity;
  int               width = 16;
  for (const auto& b : blocks) {
    identity.push_back(b.stride == 1 && width == b.out);
    width = b.out;
  }
  std::set<std::string> removable;
  for (std::size_t j = 0; j < blocks.size(); ++j) {
    // Depthwise BN shifts that feed only a 1x1 convolution and BN: every
    // projection shortcut's, and a strided block's when no SE sits between.
    const auto block = "block" + std::to_string(j + 1);
    if (!identity[j]) removable.insert(block + "/shortcut/depthwise/bn/beta");
    if (blocks[j].stride > 1 && blocks[j].se == 0) removable.insert(block + "/depthwise/bn/beta");
    bool only_pointwise = true;
    for (std::size_t k = j + 1; k < blocks.size(); ++k)
      if (!identity[k]) {
        only_pointwise = false;  // a depthwise shortcut reads the stream
        break;
      }
    if (!only_pointwise) continue;
    removable.insert(block + "/ghost2/cheap/bn/beta");
    if (!identity[j]) removable.insert(block + "/shortcut/pointwise/bn/beta");
  }
  {
    nn::NoGradGuard guard;
    ghostnet::GhostNet(p, state, x, cfg, true);
  }
  mobile_checks::ShiftBetas(p);
  const auto  net   = ghostnet::GhostNet(p, state, x, cfg, true);
  const auto  norms = mobile_checks::GradientNorms(p, nn::mean(nn::square(net.logits)));
  double      minimum = std::numeric_limits<double>::infinity(), maximum = 0, removed = 0;
  int         removed_count = 0;
  std::string smallest;
  for (double norm : norms) maximum = std::max(maximum, norm);
  for (std::size_t i = 0; i < norms.size(); ++i) {
    const double norm = norms[i];
    const auto&  name = p.names()[i];
    check::Require(std::isfinite(norm), "finite GhostNet derivative");
    if (removable.count(name)) {
      removed = std::max(removed, norm);
      ++removed_count;
    } else if (norm < minimum) {
      minimum  = norm;
      smallest = name;
    }
  }
  check::Require(removed_count == int(removable.size()) && removed < 1e-6 * maximum,
                 "vanishing BN shifts that only reach 1x1 convolutions");
  check::Require(minimum > 1e-5 * maximum, "clearly nonzero GhostNet derivatives with shifted BN and SE");
  std::cout << "  width 1 with shifted BN betas and SE biases: gradient norms [" << minimum << ',' << maximum
            << "] (smallest " << smallest << "); " << removed_count << " BN shift derivatives vanish (largest " << removed
            << ")" << std::endl;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    for (int i = 1; i < argc; ++i)  // threads= also applies to the checks
      if (std::string(argv[i]).rfind("threads=", 0) == 0) nn::set_num_threads(std::stoi(std::string(argv[i]).substr(8)));
    ModuleChecks();
    CostChecks();
    GradientChecks();
    image_training::ImageTrainingConfig cfg;
    cfg.name                  = "ghostnet_cifar10";
    cfg.maximum_fit_rows      = 12000;
    cfg.epochs                = 12;
    cfg.minimum_test_accuracy = .4;
    const image_training::LogitsBuilder builder = [](nn::ParameterStore& p, nn::ParameterStore& state,
                                                     const nn::Tensor& image, bool training) {
      ghostnet::GhostNetConfig model;
      model.classes     = 10;
      model.small_stem  = true;
      const auto logits = ghostnet::GhostNet(p, state, image, model, training).logits;
      check::Require(p.count() == 3914318, "CIFAR GhostNet parameter count");
      return logits;
    };
    if (image_training::RunCifar10(argc, argv, cfg, builder)) return 1;
    std::cout << "All GhostNet checks passed.\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
