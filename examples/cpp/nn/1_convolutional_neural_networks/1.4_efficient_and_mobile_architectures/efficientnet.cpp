// CMD: just run-nn efficientnet
// EfficientNet B0-B7: compound scaling of depth, width and resolution.
//
// Port of tflow's efficientnet. Derive every variant from the B0 block
// strings and the released width, depth, resolution and dropout table, and
// measure each at its own resolution (one forward pass each): 5,288,548 /
// 7,794,184 / 9,109,994 / 12,233,232 / 19,341,616 / 30,389,784 / 43,040,704 /
// 66,347,960 learned parameters (torchvision's and Keras's counts) and 0.39 /
// 0.69 / 0.99 / 1.83 / 4.39 / 10.27 / 19.07 / 37.75B multiply-adds. The
// paper's Table 2 lists 0.39 / 0.70 / 1.0 / 1.8 / 4.2 / 9.9 / 19 / 37B, so
// this count is 4% higher for B4 and B5 only. Check drop connect: each
// example's residual branch is either removed or scaled by 1/p, with about a
// fraction p kept (nn's generator draws the mask, so the kept count differs
// from TF's 423 of 512). Check every derivative of B0 in training mode (drop
// connect off, moving averages not updated) with shifted BN betas and SE
// biases: all are clearly nonzero except the projection BN shifts, whose
// per-channel constants only reach 1x1 convolutions followed by
// training-mode BN.
// https://arxiv.org/abs/1905.11946
// https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet
//
// Train B0 on CIFAR-10 with a stride-one stem (all sixteen blocks, four
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
    std::int64_t params, state, macs;
    int          blocks, head;
  };
  const Case cases[8] = {{5288548, 42016, 385814752, 16, 1280},       {7794184, 62048, 686802832, 23, 1280},
                         {9109994, 67568, 993561872, 23, 1408},       {12233232, 87296, 1827141392, 26, 1536},
                         {19341616, 125200, 4393771024, 32, 1792},    {30389784, 172736, 10266384752, 39, 2048},
                         {43040704, 224432, 19067593536, 45, 2304},   {66347960, 310720, 37745884192, 55, 2560}};
  for (int v = 0; v < 8; ++v) {
    const auto         scale = mbconv::EfficientNetCoefficients(v);
    const int          r     = scale.resolution;
    nn::ParameterStore p(17), state(17);
    nn::NoGradGuard    guard;
    mbconv::EfficientNetConfig cfg;
    cfg.variant    = v;
    const auto net = mbconv::EfficientNet(p, state, nn::zeros({1, r, r, 3}), cfg);
    int        blocks = 0;
    for (const auto& b : mbconv::EfficientNetBlocks(v)) blocks += b.repeats;
    const auto& c        = cases[v];
    const auto  features = net.features.shape();
    check::Require(p.count() == c.params && state.count() == c.state && net.Macs() == c.macs && blocks == c.blocks &&
                       features[3] == c.head,
                   "EfficientNet scaling, parameters and multiply-adds");
    std::cout << "  B" << v << ": width " << scale.width << ", depth " << scale.depth << ", " << r << " px, dropout "
              << scale.dropout << "; " << blocks << " blocks, head " << c.head << ", features " << features[1] << "x"
              << features[2] << "; " << p.count() << " parameters, " << std::fixed << std::setprecision(2)
              << net.Macs() / 1e9 << "B mult-adds" << std::defaultfloat << std::setprecision(6) << std::endl;
  }
}

void DropConnectChecks() {
  const nn::Tensor branch = Pattern({512, 2, 2, 3}, 1, .1f);
  nn::Generator    gen(77);
  const auto       out = mbconv::DropConnect(branch, .8, gen).to_vector<float>();
  const auto       value = branch.to_vector<float>();
  int              kept  = 0;
  double           error = 0;
  for (int example = 0; example < 512; ++example) {
    bool keep = false;
    for (int j = 0; j < 12; ++j) keep = keep || out[example * 12 + j] != 0;
    kept += keep;
    for (int j = 0; j < 12; ++j)
      error = std::max(error, double(std::abs(out[example * 12 + j] - (keep ? value[example * 12 + j] / .8f : 0.f))));
  }
  check::Near(error, 0, 1e-6, "drop connect removes or rescales whole examples");
  check::Require(std::abs(kept / 512. - .8) < .06, "drop connect survival rate");
  std::cout << "  drop connect with survival 0.8: " << kept
            << " of 512 residual branches kept, each scaled by 1.25 exactly (error " << error << ")" << std::endl;
}

void GradientChecks() {
  nn::ParameterStore         p(17), state(17);
  mbconv::EfficientNetConfig cfg;
  cfg.drop_connect   = 0;
  cfg.update_stats   = false;
  const nn::Tensor x = Pattern({2, 224, 224, 3}, 1, 0);
  {
    nn::NoGradGuard guard;
    mbconv::EfficientNet(p, state, x, cfg, true);
  }
  mobile_checks::ShiftBetas(p);
  const auto  net   = mbconv::EfficientNet(p, state, x, cfg, true);
  const auto  norms = mobile_checks::GradientNorms(p, nn::mean(nn::square(net.logits)));
  double      minimum = std::numeric_limits<double>::infinity(), maximum = 0, removed = 0;
  int         removed_count = 0;
  std::string smallest;
  for (double norm : norms) maximum = std::max(maximum, norm);
  for (std::size_t i = 0; i < norms.size(); ++i) {
    const double norm = norms[i];
    const auto&  name = p.names()[i];
    check::Require(std::isfinite(norm), "finite EfficientNet derivative");
    if (EndsWith(name, "/project/bn/beta")) {
      removed = std::max(removed, norm);
      ++removed_count;
    } else if (norm < minimum) {
      minimum  = norm;
      smallest = name;
    }
  }
  check::Require(removed_count == 16 && removed < 1e-6 * maximum, "vanishing projection BN shift derivatives");
  check::Require(minimum > 1e-5 * maximum, "clearly nonzero EfficientNet derivatives with shifted BN and SE");
  std::cout << "  B0 with shifted BN betas and SE biases: gradient norms [" << minimum << ',' << maximum << "] (smallest "
            << smallest << "); 16 projection BN shift derivatives vanish (largest " << removed << ")" << std::endl;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    for (int i = 1; i < argc; ++i)  // threads= also applies to the checks
      if (std::string(argv[i]).rfind("threads=", 0) == 0) nn::set_num_threads(std::stoi(std::string(argv[i]).substr(8)));
    CostChecks();
    DropConnectChecks();
    GradientChecks();
    image_training::ImageTrainingConfig cfg;
    cfg.name                  = "efficientnet_b0_cifar10";
    cfg.maximum_fit_rows      = 12000;
    cfg.epochs                = 12;
    cfg.minimum_test_accuracy = .4;
    const image_training::LogitsBuilder builder = [](nn::ParameterStore& p, nn::ParameterStore& state,
                                                     const nn::Tensor& image, bool training) {
      mbconv::EfficientNetConfig model;
      model.classes     = 10;
      model.small_stem  = true;
      const auto logits = mbconv::EfficientNet(p, state, image, model, training).logits;
      check::Require(p.count() == 4020358, "CIFAR EfficientNet-B0 parameter count");
      return logits;
    };
    if (image_training::RunCifar10(argc, argv, cfg, builder)) return 1;
    std::cout << "All EfficientNet checks passed.\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
