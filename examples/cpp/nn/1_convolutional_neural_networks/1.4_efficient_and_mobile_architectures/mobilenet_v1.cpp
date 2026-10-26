// CMD: just run-nn mobilenet_v1
// MobileNet v1: depthwise separable convolutions with width and resolution multipliers.
//
// Port of tflow's mobilenet_v1. Measure every layer's output size from a
// forward pass and tabulate multiply-adds and parameters. Check the paper's
// Tables 6 and 7 exactly as counted here (alpha 1/.75/.5/.25 at 224:
// 569/325/149/41M mult-adds and 4.2/2.6/1.3/0.5M parameters; 224/192/160/128
// pixels: 569/418/290/186M), the full-convolution baseline of Table 4 (29.3M
// parameters, 4866M mult-adds without the classifier) and Table 2's breakdown
// by layer type (1x1: 94.86% of mult-adds, 74.59% of weights). Table 2 prints
// 1.19% for the first convolution, but 1.91% is what makes the shares sum to
// 100%. Check every parameter derivative of the full model in training mode
// (moving averages not updated), with shifted BN betas: at beta=0, ReLU6 is
// positively homogeneous below 6, so a pointwise BN's scale reaches the next
// training-mode BN only as a removable per-channel factor and its derivative
// nearly vanishes.
// https://arxiv.org/abs/1704.04861
// https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.py
//
// Train alpha 1 and alpha 0.5 on CIFAR-10 with a stride-one first convolution
// (all thirteen blocks and their four remaining downsamplings). Use 12k fit /
// 5k validation, twelve epochs for alpha 1 and twenty-four for alpha 0.5,
// crop/flip, Adam(.001), kernel L2(.0001) and the best-validation checkpoint
// before evaluating all 10k test images. These are CIFAR adaptations, not the
// ImageNet benchmark. Initialization uses nn's generator (He-uniform, as
// tflow), so trained numbers match TF only statistically. CIFAR-10 is read
// from the tflow dataset cache (common/paths.hpp) or data=.
// Options: alpha=1|0.5 (default both), data=PATH epochs=N batch=128 lr=.001
// fit=12000 augment=0|1 threads=N seed=17.

#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>

#include "common/checks.hpp"
#include "common/image_training.hpp"
#include "common/mobilenet.hpp"

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

struct Cost {
  std::int64_t                  params = 0, state = 0, macs = 0;
  std::map<std::string, double> mac_share, weight_share;
};
// Multiply-adds from measured output sizes; weight shares exclude BN and the classifier bias.
Cost Measure(const mobilenet::MobileNetV1Config& cfg, int resolution) {
  nn::ParameterStore p(17), state(17);
  nn::NoGradGuard    guard;
  const auto net = mobilenet::MobileNetV1(p, state, nn::zeros({1, resolution, resolution, 3}), cfg);
  Cost       cost;
  cost.params = p.count();
  cost.state  = state.count();
  std::map<std::string, std::int64_t> macs, weights;
  for (const auto& layer : net.layers) {
    const auto& name = layer.name;
    const auto  kind = name == "conv0"                                ? "conv 3x3"
                       : name.find("depthwise") != std::string::npos ? "depthwise 3x3"
                       : name.find("pointwise") != std::string::npos ? "conv 1x1"
                                                                     : "conv 3x3";
    macs[kind] += layer.Macs();
    weights[kind] += p.at(name + "/W").numel();
  }
  macs["classifier"]    = net.classifier_macs;
  weights["classifier"] = p.at("output/W").numel();
  std::int64_t total_weights = 0;
  for (const auto& [kind, value] : macs) cost.macs += value;
  for (const auto& [kind, value] : weights) total_weights += value;
  for (const auto& [kind, value] : macs) cost.mac_share[kind] = 100. * value / cost.macs;
  for (const auto& [kind, value] : weights) cost.weight_share[kind] = 100. * value / total_weights;
  return cost;
}

void CostChecks() {
  struct Case {
    double       alpha;
    int          resolution;
    std::int64_t params, state, macs;
  };
  const std::vector<Case> cases = {{1, 224, 4231976, 21888, 568740352},   {.75, 224, 2585560, 16416, 325400448},
                                   {.5, 224, 1331592, 10944, 149497088},  {.25, 224, 470072, 5472, 41030272},
                                   {1, 192, 4231976, 21888, 418121728},   {1, 160, 4231976, 21888, 290675200},
                                   {1, 128, 4231976, 21888, 186400768}};
  for (const auto& c : cases) {
    mobilenet::MobileNetV1Config cfg;
    cfg.alpha       = c.alpha;
    const auto cost = Measure(cfg, c.resolution);
    check::Require(cost.params == c.params && cost.state == c.state && cost.macs == c.macs,
                   "MobileNet v1 parameters and multiply-adds");
    std::cout << "  alpha " << c.alpha << ", " << c.resolution << " px: " << std::fixed << std::setprecision(1)
              << cost.macs / 1e6 << "M mult-adds, " << std::setprecision(2) << cost.params / 1e6 << "M parameters"
              << std::defaultfloat << std::setprecision(6) << " (" << cost.params << ", state " << cost.state << ")"
              << std::endl;
    if (c.alpha == 1 && c.resolution == 224) {
      const std::map<std::string, std::pair<double, double>> table2 = {{"conv 1x1", {94.86, 74.59}},
                                                                        {"depthwise 3x3", {3.06, 1.06}},
                                                                        {"conv 3x3", {1.91, .02}},
                                                                        {"classifier", {.18, 24.33}}};
      std::cout << "   ";
      for (const auto& [kind, expected] : table2) {
        check::Near(cost.mac_share.at(kind), expected.first, .005, "Table 2 mult-add share: " + kind);
        check::Near(cost.weight_share.at(kind), expected.second, .005, "Table 2 parameter share: " + kind);
        std::cout << ' ' << kind << ' ' << std::fixed << std::setprecision(2) << cost.mac_share.at(kind) << "% / "
                  << cost.weight_share.at(kind) << "%;" << std::defaultfloat << std::setprecision(6);
      }
      std::cout << " (mult-adds / weights)" << std::endl;
    }
  }
  mobilenet::MobileNetV1Config dense;
  dense.full_convolutions        = true;
  const auto         cost        = Measure(dense, 224);
  const std::int64_t classifier  = 1024 * 1000;
  check::Require(cost.params == 29294088 && cost.macs - classifier == 4866269184, "full-convolution MobileNet");
  std::cout << "  full 3x3 convolutions: " << std::fixed << std::setprecision(1) << (cost.macs - classifier) / 1e6
            << "M mult-adds without the classifier, " << cost.params / 1e6 << "M parameters (" << std::setprecision(2)
            << double(cost.macs) / 568740352 << "x and " << double(cost.params) / 4231976 << "x MobileNet)"
            << std::defaultfloat << std::setprecision(6) << std::endl;
}

void GradientChecks() {
  nn::ParameterStore           p(17), state(17);
  mobilenet::MobileNetV1Config cfg;
  cfg.update_stats = false;
  std::vector<float> pixels(2 * 224 * 224 * 3);
  for (std::size_t i = 0; i < pixels.size(); ++i) pixels[i] = std::sin(.13f * static_cast<float>(i));
  const nn::Tensor x = nn::from_vector(pixels, {2, 224, 224, 3});
  {
    nn::NoGradGuard guard;  // create the parameters
    mobilenet::MobileNetV1(p, state, x, cfg, true);
    // At initialization (beta=0) ReLU6 is positively homogeneous below 6, so a
    // pointwise BN's scale passes through ReLU6 and the per-channel depthwise
    // convolution into the next training-mode BN, which removes it: those gamma
    // derivatives nearly vanish. Shifted betas, as after training, avoid that.
    for (const auto& name : p.names())
      if (name.size() > 5 && name.compare(name.size() - 5, 5, "/beta") == 0) {
        std::vector<float> beta(static_cast<std::size_t>(p.at(name).numel()));
        for (std::size_t i = 0; i < beta.size(); ++i)
          beta[i] = .5f * std::sin(.7f * static_cast<float>(i) + static_cast<float>(name.size()));
        p.set(name, nn::from_vector(beta, p.at(name).shape()));
      }
  }
  const auto net = mobilenet::MobileNetV1(p, state, x, cfg, true);
  check::Require(net.features.shape() == nn::Shape({2, 7, 7, 1024}) && net.logits.shape() == nn::Shape({2, 1000}),
                 "MobileNet output dimensions");
  const auto  gradients = nn::grad(nn::mean(nn::square(net.logits)), p.parameters());
  double      minimum = std::numeric_limits<double>::infinity(), maximum = 0;
  std::string smallest;
  for (std::size_t i = 0; i < gradients.size(); ++i) {
    const double norm = nn::sqrt(nn::sum(nn::square(gradients[i]))).item();
    check::Require(std::isfinite(norm), "finite MobileNet derivative");
    if (norm < minimum) {
      minimum  = norm;
      smallest = p.names()[i];
    }
    maximum = std::max(maximum, norm);
  }
  check::Require(minimum > 1e-4 * maximum, "clearly nonzero MobileNet derivatives with shifted BN");
  std::cout << "  alpha 1 at 224 with shifted BN betas: 7x7x1024 features; training-mode gradient norms [" << minimum
            << ',' << maximum << "], smallest " << smallest << std::endl;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    std::vector<double> widths        = {1., .5};
    std::vector<char*>  training_args = {argv[0]};
    for (int i = 1; i < argc; ++i) {
      const std::string argument = argv[i];
      if (argument == "alpha=1")
        widths = {1.};
      else if (argument == "alpha=0.5")
        widths = {.5};
      else if (argument.rfind("alpha=", 0) == 0)
        throw std::invalid_argument("alpha must be 1 or 0.5");
      else {
        if (argument.rfind("threads=", 0) == 0) nn::set_num_threads(std::stoi(argument.substr(8)));  // checks too
        training_args.push_back(argv[i]);
      }
    }
    CostChecks();
    GradientChecks();
    for (double alpha : widths) {
      image_training::ImageTrainingConfig cfg;
      cfg.name                  = alpha == 1 ? "mobilenet_v1_cifar10" : "mobilenet_v1_050_cifar10";
      cfg.maximum_fit_rows      = 12000;
      cfg.epochs                = alpha == 1 ? 12 : 24;
      cfg.minimum_test_accuracy = .4;
      const std::int64_t                  expected = alpha == 1 ? 3217226 : 823722;
      const image_training::LogitsBuilder builder  = [alpha, expected](nn::ParameterStore& p, nn::ParameterStore& state,
                                                                      const nn::Tensor& image, bool training) {
        mobilenet::MobileNetV1Config model;
        model.alpha       = alpha;
        model.classes     = 10;
        model.small_stem  = true;
        const auto logits = mobilenet::MobileNetV1(p, state, image, model, training).logits;
        check::Require(p.count() == expected, "CIFAR MobileNet parameter count");
        return logits;
      };
      if (image_training::RunCifar10(int(training_args.size()), training_args.data(), cfg, builder)) return 1;
    }
    std::cout << "All MobileNet v1 checks passed.\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
