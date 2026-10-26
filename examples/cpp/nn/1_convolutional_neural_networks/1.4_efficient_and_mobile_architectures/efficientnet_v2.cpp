// CMD: just run-nn efficientnet_v2
// EfficientNetV2: Fused-MBConv early stages and searched S/M/L models.
//
// Port of tflow's efficientnet_v2. Decode the reference's V2 block strings
// (single-digit strides, "c1" for Fused-MBConv) and measure every released
// model (one forward pass each): S/M/L have 21,458,488 / 54,139,356 /
// 118,515,272 learned parameters (torchvision's counts) and 8.37 / 24.58 /
// 56.08B multiply-adds at 384/480/480 pixels (the paper reports 8.8 / 24 /
// 53B); the base B0-B3 have 7.14 / 8.14 / 10.10 / 14.38M parameters (the
// paper's 7.1 / 8.1 / 10.1 / 14.4M). Compare one early stage built as
// Fused-MBConv and as MBConv. Check every derivative of base B0 in training
// mode (drop connect off, moving averages not updated) with shifted BN betas
// and SE biases. A projection BN shift vanishes exactly when every block it
// reaches through identity connections starts with a 1x1 convolution (a
// fused block's padded 3x3 does not keep a constant); all other derivatives
// are clearly nonzero.
// https://arxiv.org/abs/2104.00298
// https://github.com/google/automl/tree/master/efficientnetv2
//
// Train base B0 on CIFAR-10 with a stride-one stem (all 21 blocks, four
// remaining downsamplings, drop connect 0.2 and dropout 0.2). Use 12k fit /
// 5k validation, twelve epochs, crop/flip, Adam(.001), kernel L2(.0001) and
// the best-validation checkpoint before evaluating all 10k test images.
// Initialization, dropout and drop connect use nn's generator, so trained
// numbers match TF only statistically. CIFAR-10 is read from the tflow
// dataset cache (common/paths.hpp) or data=.
// Options: data=PATH epochs=12 batch=128 lr=.001 fit=12000 augment=0|1 threads=N seed=17.

#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <set>

#include "common/checks.hpp"
#include "common/image_training.hpp"
#include "common/mbconv.hpp"
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

void CostChecks() {
  struct Case {
    const char*  name;
    std::int64_t params, state, macs;
    int          blocks;
  };
  const std::vector<Case> cases = {{"s", 21458488, 153872, 8365617152, 40},   {"m", 54139356, 292032, 24582460384, 57},
                                   {"l", 118515272, 512576, 56079699008, 79}, {"b0", 7139704, 60608, 718086176, 21},
                                   {"b1", 8141052, 71072, 1196819360, 27},    {"b2", 10096086, 82288, 1697542832, 28},
                                   {"b3", 14377062, 109312, 3213058816, 32}};
  for (const auto& c : cases) {
    const auto         spec = mbconv::EfficientNetV2Spec(c.name);
    const int          r    = spec.resolution;
    nn::ParameterStore p(17), state(17);
    nn::NoGradGuard    guard;
    mbconv::EfficientNetV2Config cfg;
    cfg.variant       = c.name;
    const auto net    = mbconv::EfficientNetV2(p, state, nn::zeros({1, r, r, 3}), cfg);
    const auto macs   = net.Macs();
    int        blocks = 0, fused = 0;
    for (const auto& b : mbconv::EfficientNetV2Blocks(spec)) {
      blocks += b.repeats;
      fused += b.fused ? b.repeats : 0;
    }
    check::Require(p.count() == c.params && state.count() == c.state && macs == c.macs && blocks == c.blocks,
                   "EfficientNetV2 parameters, multiply-adds and depth");
    std::cout << "  EfficientNetV2-" << c.name << " at " << r << " px: " << blocks << " blocks (" << fused << " fused), "
              << p.count() << " parameters, " << std::fixed << std::setprecision(2) << macs / 1e9 << "B mult-adds"
              << std::defaultfloat << std::setprecision(6) << std::endl;
  }
  check::Require(mbconv::DecodeBlock("r4_k3_s2_e4_i24_o48_c1").fused && mbconv::DecodeBlock("r4_k3_s2_e4_i24_o48_c1").stride == 2,
                 "V2 block strings");
}

// The first block of S's second stage (24 -> 48, expansion 4, stride 2) at 96 px.
void FusedChecks() {
  for (bool fused : {true, false}) {
    nn::ParameterStore                  p(17), state(17);
    std::vector<mobilenet::MobileLayer> layers;
    mobilenet::MobileContext            c{p, state, layers, false};
    nn::NoGradGuard                     guard;
    auto                                b = mbconv::DecodeBlock("r1_k3_s2_e4_i24_o48_c1");
    b.fused                               = fused;
    mobilenet::MobileNetOutputs net;
    mbconv::MBConv(c, "block", nn::zeros({1, 96, 96, 24}), 48, b, 2, true);
    net.layers      = layers;
    const auto macs = net.Macs();
    check::Require(macs == (fused ? 58392576 : 33841152) && p.count() == (fused ? 25632 : 8256) &&
                       layers.size() == (fused ? 2u : 3u),
                   "Fused-MBConv and MBConv stage costs");
    std::cout << "  24->48 stride-2 expansion-4 block at 96 px as " << (fused ? "Fused-MBConv" : "MBConv      ") << ": "
              << std::fixed << std::setprecision(1) << macs / 1e6 << "M mult-adds, " << std::defaultfloat
              << std::setprecision(6) << p.count() << " parameters, " << layers.size() << " convolutions" << std::endl;
  }
}

void GradientChecks() {
  nn::ParameterStore           p(17), state(17);
  mbconv::EfficientNetV2Config cfg;
  cfg.variant        = "b0";
  cfg.drop_connect   = 0;
  cfg.update_stats   = false;
  const nn::Tensor x = Pattern({2, 224, 224, 3}, 1, 0);
  // Projection shifts that only reach 1x1 convolutions (then BN).
  std::vector<mbconv::BlockArgs> flat;
  std::vector<int>               strides;
  for (const auto& b : mbconv::EfficientNetV2Blocks(mbconv::EfficientNetV2Spec("b0")))
    for (int r = 0; r < b.repeats; ++r) {
      flat.push_back(b);
      flat.back().input = r ? b.output : b.input;
      strides.push_back(r ? 1 : b.stride);
    }
  std::set<std::string> removable;
  for (std::size_t j = 0; j < flat.size(); ++j) {
    if (flat[j].fused && flat[j].expansion == 1) continue;  // projection followed by an activation
    bool only_pointwise = true;
    for (std::size_t k = j + 1; k <= flat.size(); ++k) {
      if (k == flat.size()) break;  // the head is a 1x1 convolution
      if (flat[k].fused || flat[k].expansion == 1) {
        only_pointwise = false;  // first convolution is not 1x1
        break;
      }
      const bool identity = flat[k].skip && strides[k] == 1 && flat[k].input == flat[k].output;
      if (!identity) break;
    }
    if (only_pointwise) removable.insert("block" + std::to_string(j + 1) + "/project/bn/beta");
  }
  {
    nn::NoGradGuard guard;
    mbconv::EfficientNetV2(p, state, x, cfg, true);
  }
  mobile_checks::ShiftBetas(p);
  const auto  net   = mbconv::EfficientNetV2(p, state, x, cfg, true);
  const auto  norms = mobile_checks::GradientNorms(p, nn::mean(nn::square(net.logits)));
  double      minimum = std::numeric_limits<double>::infinity(), maximum = 0, removed = 0;
  int         removed_count = 0, kept_shifts = 0;
  std::string smallest;
  for (double norm : norms) maximum = std::max(maximum, norm);
  for (std::size_t i = 0; i < norms.size(); ++i) {
    const double norm = norms[i];
    const auto&  name = p.names()[i];
    check::Require(std::isfinite(norm), "finite EfficientNetV2 derivative");
    if (removable.count(name)) {
      removed = std::max(removed, norm);
      ++removed_count;
    } else {
      if (EndsWith(name, "/project/bn/beta")) ++kept_shifts;
      if (norm < minimum) {
        minimum  = norm;
        smallest = name;
      }
    }
  }
  check::Require(removed_count == int(removable.size()) && removed < 1e-6 * maximum,
                 "vanishing projection shifts that reach only 1x1 convolutions");
  check::Require(minimum > 1e-5 * maximum, "clearly nonzero EfficientNetV2 derivatives with shifted BN and SE");
  std::cout << "  base B0 with shifted BN betas and SE biases: gradient norms [" << minimum << ',' << maximum
            << "] (smallest " << smallest << "); " << removed_count << " projection shift derivatives vanish (largest "
            << removed << "), " << kept_shifts << " that reach a fused block do not" << std::endl;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    for (int i = 1; i < argc; ++i)  // threads= also applies to the checks
      if (std::string(argv[i]).rfind("threads=", 0) == 0) nn::set_num_threads(std::stoi(std::string(argv[i]).substr(8)));
    CostChecks();
    FusedChecks();
    GradientChecks();
    image_training::ImageTrainingConfig cfg;
    cfg.name                  = "efficientnet_v2_b0_cifar10";
    cfg.maximum_fit_rows      = 12000;
    cfg.epochs                = 12;
    cfg.minimum_test_accuracy = .4;
    const image_training::LogitsBuilder builder = [](nn::ParameterStore& p, nn::ParameterStore& state,
                                                     const nn::Tensor& image, bool training) {
      mbconv::EfficientNetV2Config model;
      model.variant     = "b0";
      model.classes     = 10;
      model.small_stem  = true;
      const auto logits = mbconv::EfficientNetV2(p, state, image, model, training).logits;
      check::Require(p.count() == 5871514, "CIFAR EfficientNetV2-B0 parameter count");
      return logits;
    };
    if (image_training::RunCifar10(argc, argv, cfg, builder)) return 1;
    std::cout << "All EfficientNetV2 checks passed.\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
