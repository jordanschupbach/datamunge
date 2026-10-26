// CMD: just run-nn mnasnet
// MnasNet: platform-aware searched mobile networks (A1 with SE, B1 without).
//
// Port of tflow's mnasnet. Decode the TensorFlow reference's block strings
// and re-encode them unchanged. Measure MnasNet-A1 (3,887,038 learned
// parameters, 311.6M multiply-adds; the paper's 3.9M / 312M) and MnasNet-B1
// (4,383,312 and 314.4M, torchvision's "mnasnet1_0"), plus B1 at depth
// multipliers 0.5/0.75/1.3 with 2,218,512 / 3,170,208 / 6,282,256
// parameters, which reproduces torchvision's scaled models. Check every
// derivative in training mode (moving averages not updated) with shifted BN
// betas and SE biases: all are clearly nonzero except the projection BN
// shifts, whose per-channel constants only reach 1x1 convolutions followed
// by training-mode BN.
// https://arxiv.org/abs/1807.11626
// https://github.com/tensorflow/tpu/tree/master/models/official/mnasnet
//
// Train MnasNet-A1 on CIFAR-10 with a stride-one stem (all sixteen blocks
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

void BlockStringChecks() {
  for (const char* variant : {"a1", "b1"}) {
    int blocks = 0, se = 0, skips = 0;
    for (const auto& b : mbconv::MnasNetBlocks(variant)) {
      check::Require(mbconv::DecodeBlock(mbconv::EncodeBlock(b)).repeats == b.repeats &&
                         mbconv::EncodeBlock(mbconv::DecodeBlock(mbconv::EncodeBlock(b))) == mbconv::EncodeBlock(b),
                     "block string round trip");
      blocks += b.repeats;
      se += b.se_ratio > 0 ? b.repeats : 0;
      for (int r = 0; r < b.repeats; ++r) skips += b.skip && (r > 0 || (b.input == b.output && b.stride == 1));
    }
    std::cout << "  MnasNet-" << variant << ": " << blocks << " blocks, " << se << " with squeeze-and-excitation, "
              << skips << " identity connections; third block string "
              << mbconv::EncodeBlock(mbconv::MnasNetBlocks(variant)[2]) << std::endl;
  }
  check::Require(mbconv::EncodeBlock(mbconv::DecodeBlock("r3_k5_s22_e6_i112_o160_se0.25")) == "r3_k5_s22_e6_i112_o160_se0.25" &&
                     mbconv::EncodeBlock(mbconv::DecodeBlock("r1_k3_s11_e6_i192_o320_noskip")) ==
                         "r1_k3_s11_e6_i192_o320_noskip",
                 "reference block strings");
}

void CostChecks() {
  struct Case {
    const char*  variant;
    double       multiplier;
    std::int64_t params, state, macs;
  };
  const std::vector<Case> cases = {{"a1", 1, 3887038, 33488, 311579072}, {"b1", 1, 4383312, 37920, 314415872},
                                   {"b1", .5, 2218512, -1, -1},          {"b1", .75, 3170208, -1, -1},
                                   {"b1", 1.3, 6282256, -1, -1}};
  for (const auto& c : cases) {
    nn::ParameterStore    p(17), state(17);
    nn::NoGradGuard       guard;
    mbconv::MnasNetConfig cfg;
    cfg.variant     = c.variant;
    cfg.multiplier  = c.multiplier;
    const auto net  = mbconv::MnasNet(p, state, nn::zeros({1, 224, 224, 3}), cfg);
    const auto macs = net.Macs();
    check::Require(p.count() == c.params && (c.state < 0 || state.count() == c.state) && (c.macs < 0 || macs == c.macs),
                   "MnasNet parameters and multiply-adds");
    std::cout << "  MnasNet-" << c.variant << " x" << c.multiplier << ": " << p.count() << " parameters, " << std::fixed
              << std::setprecision(1) << macs / 1e6 << "M mult-adds" << std::defaultfloat << std::setprecision(6)
              << std::endl;
  }
}

void GradientChecks() {
  nn::ParameterStore    p(17), state(17);
  mbconv::MnasNetConfig cfg;
  cfg.update_stats   = false;
  const nn::Tensor x = Pattern({2, 224, 224, 3}, 1, 0);
  {
    nn::NoGradGuard guard;
    mbconv::MnasNet(p, state, x, cfg, true);
  }
  mobile_checks::ShiftBetas(p);
  const auto  net   = mbconv::MnasNet(p, state, x, cfg, true);
  const auto  norms = mobile_checks::GradientNorms(p, nn::mean(nn::square(net.logits)));
  double      minimum = std::numeric_limits<double>::infinity(), maximum = 0, removed = 0;
  int         removed_count = 0;
  std::string smallest;
  for (double norm : norms) maximum = std::max(maximum, norm);
  for (std::size_t i = 0; i < norms.size(); ++i) {
    const double norm = norms[i];
    const auto&  name = p.names()[i];
    check::Require(std::isfinite(norm), "finite MnasNet derivative");
    if (EndsWith(name, "/project/bn/beta")) {
      removed = std::max(removed, norm);
      ++removed_count;
    } else if (norm < minimum) {
      minimum  = norm;
      smallest = name;
    }
  }
  check::Require(removed_count == 16 && removed < 1e-6 * maximum, "vanishing projection BN shift derivatives");
  check::Require(minimum > 1e-5 * maximum, "clearly nonzero MnasNet derivatives with shifted BN and SE");
  std::cout << "  MnasNet-a1 with shifted BN betas and SE biases: gradient norms [" << minimum << ',' << maximum
            << "] (smallest " << smallest << "); 16 projection BN shift derivatives vanish (largest " << removed << ")"
            << std::endl;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    for (int i = 1; i < argc; ++i)  // threads= also applies to the checks
      if (std::string(argv[i]).rfind("threads=", 0) == 0) nn::set_num_threads(std::stoi(std::string(argv[i]).substr(8)));
    BlockStringChecks();
    CostChecks();
    GradientChecks();
    image_training::ImageTrainingConfig cfg;
    cfg.name                  = "mnasnet_a1_cifar10";
    cfg.maximum_fit_rows      = 12000;
    cfg.epochs                = 12;
    cfg.minimum_test_accuracy = .4;
    const image_training::LogitsBuilder builder = [](nn::ParameterStore& p, nn::ParameterStore& state,
                                                     const nn::Tensor& image, bool training) {
      mbconv::MnasNetConfig model;
      model.classes     = 10;
      model.small_stem  = true;
      const auto logits = mbconv::MnasNet(p, state, image, model, training).logits;
      check::Require(p.count() == 2618848, "CIFAR MnasNet parameter count");
      return logits;
    };
    if (image_training::RunCifar10(argc, argv, cfg, builder)) return 1;
    std::cout << "All MnasNet checks passed.\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
