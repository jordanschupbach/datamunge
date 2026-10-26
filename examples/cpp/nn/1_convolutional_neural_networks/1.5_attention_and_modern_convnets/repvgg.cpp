// CMD: just run-nn repvgg
// RepVGG: train three branches, deploy one 3x3 convolution per block.
//
// Port of tflow's repvgg. Fuse identity, strided and grouped training blocks
// (3x3+BN, 1x1+BN and BN branches with nontrivial statistics) into single
// biased 3x3 convolutions and check that inference outputs agree. Fuse every
// block of a RepVGG-A0 the same way and check that the deployed plain network
// reproduces the training network's logits. Count all ten released models in
// both forms: training / deployed parameters from 9,108,968 / 8,309,384 (A0)
// to 123,085,288 / 110,960,872 (B3) and deployed multiply-adds from 1.36G to
// 26.21G at 224 pixels (the paper's Table 4, 9.11/8.31M ... 123.08/110.96M
// and 1.4 ... 26.2 GFLOPs). Check every training-form derivative of A0
// (training mode, moving averages not updated).
// https://arxiv.org/abs/2101.03697
// https://github.com/DingXiaoH/RepVGG
//
// Train RepVGG-A0 (training form) on CIFAR-10 with a stride-one first block,
// all 22 blocks and four downsamplings; the fusion checks above show that
// the deployed plain network would make the same predictions. Use 12k fit /
// 5k validation, twelve epochs, crop/flip, Adam(.001), kernel L2(.0001) and
// the best-validation checkpoint before evaluating all 10k test images.
// Fusion reads the parameter/state stores directly (tflow fetches variables
// from a session and assigns them in a second graph). Training-form counts
// are built on 32-pixel inputs; deployed multiply-adds use the block output
// sizes of a real 224-pixel forward, as in tflow. Weights come from nn's
// generator, so trained numbers match TF only statistically. CIFAR-10 is read
// from the tflow dataset cache (common/paths.hpp) or data=.
// Options: data=PATH epochs=12 batch=128 lr=.001 fit=12000 augment=0|1 threads=N seed=17.

#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>

#include "common/checks.hpp"
#include "common/image_training.hpp"
#include "common/mobile_checks.hpp"
#include "common/repvgg.hpp"

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

// Nontrivial BN parameters and statistics, as after training.
void Perturb(nn::ParameterStore& p, nn::ParameterStore& state) {
  nn::NoGradGuard guard;
  for (const auto& name : p.names())
    if (name.find("/bn/") != std::string::npos) {
      nn::Tensor v = Pattern(p.at(name).shape(), .3f, float(name.size()));
      if (name.find("gamma") != std::string::npos) v = v + 1.0;
      p.set(name, v);
    }
  for (const auto& name : state.names()) {
    nn::Tensor v = Pattern(state.at(name).shape(), .2f, float(name.size()));
    if (name.find("moving_var") != std::string::npos) v = v + 1.0;
    state.set(name, v);
  }
}

double MaxDifference(const nn::Tensor& a, const nn::Tensor& b) {
  const auto x = a.to_vector<float>(), y = b.to_vector<float>();
  double     error = 0, scale = 0;
  for (std::size_t i = 0; i < x.size(); ++i) {
    error = std::max(error, double(std::abs(x[i] - y[i])));
    scale = std::max(scale, double(std::abs(y[i])));
  }
  return error / std::max(1., scale);
}

void BlockChecks() {
  const std::vector<repvgg::RepVGGBlockSpec> specs = {
      {"identity", 8, 8, 1, 1, true}, {"strided", 8, 12, 2, 1, false}, {"grouped", 8, 8, 1, 2, true}};
  nn::NoGradGuard guard;
  for (const auto& spec : specs) {
    nn::ParameterStore p(17), state(17);
    const nn::Tensor   value = Pattern({2, 7, 7, 8}, 1, .3f);
    repvgg::RepVGGBlock(p, state, spec, value, false, false);  // create the parameters
    Perturb(p, state);
    const nn::Tensor   trained = repvgg::RepVGGBlock(p, state, spec, value, false, false);
    nn::ParameterStore dp(17), ds(17);
    repvgg::RepVGGBlock(dp, ds, spec, value, true, false);
    repvgg::DeployRepVGG(dp, {spec}, p, state);
    const nn::Tensor fused = repvgg::RepVGGBlock(dp, ds, spec, value, true, false);
    const double     error = MaxDifference(fused, trained);
    check::Near(error, 0, 1e-5, "fused block equals training block");
    std::cout << "  " << spec.name << " block (" << spec.in << "->" << spec.out << ", stride " << spec.stride << ", "
              << spec.groups << (spec.groups > 1 ? " groups" : " group") << (spec.identity ? ", identity branch" : "")
              << "): fused 3x3 error " << error << std::endl;
  }
}

void NetworkChecks() {
  nn::NoGradGuard    guard;
  nn::ParameterStore p(17), state(17);
  const nn::Tensor   x = Pattern({1, 64, 64, 3}, 1, .1f);
  repvgg::RepVGG(p, state, x, {});
  check::Require(p.count() == 9108968 && state.count() == 23648, "RepVGG-A0 training parameters");
  Perturb(p, state);
  const auto           net = repvgg::RepVGG(p, state, x, {});
  nn::ParameterStore   dp(17), ds(17);
  repvgg::RepVGGConfig cfg;
  cfg.deploy = true;
  repvgg::RepVGG(dp, ds, x, cfg);
  check::Require(dp.count() == 8309384 && ds.count() == 0, "RepVGG-A0 deployed parameters");
  repvgg::DeployRepVGG(dp, net.blocks, p, state);
  const auto   plain = repvgg::RepVGG(dp, ds, x, cfg);
  const double error = MaxDifference(plain.logits, net.logits);
  check::Near(error, 0, 1e-4, "deployed RepVGG-A0 reproduces the training network");
  std::cout << "  RepVGG-A0: " << net.blocks.size() << " blocks fused; deployed plain 3x3 stack (" << dp.count()
            << " parameters, training form " << p.count() << " + " << state.count()
            << " BN statistics) reproduces the logits (relative error " << error << ")" << std::endl;
}

void CostChecks() {
  struct Case {
    const char*  name;
    std::int64_t train, deploy, macs;
  };
  const std::vector<Case> cases = {
      {"A0", 9108968, 8309384, 1361451008},       {"A1", 14092264, 12789864, 2363967488},
      {"A2", 28210600, 25499944, 5116951552},     {"B0", 15817960, 14339048, 3057600512},
      {"B1", 57415016, 51829480, 11815485440},    {"B1g2", 45782376, 41360104, 8809742336},
      {"B1g4", 39966056, 36125416, 7306870784},   {"B2", 89022376, 80315112, 18376609792},
      {"B2g4", 61758376, 55777512, 11331899392},  {"B3", 123085288, 110960872, 26208882688}};
  nn::NoGradGuard guard;
  for (const auto& c : cases) {
    nn::ParameterStore   p(17), state(17);
    repvgg::RepVGGConfig cfg;
    cfg.model = c.name;
    repvgg::RepVGG(p, state, nn::zeros({1, 32, 32, 3}), cfg);
    nn::ParameterStore dp(17), ds(17);
    cfg.deploy       = true;
    const auto plain = repvgg::RepVGG(dp, ds, nn::zeros({1, 224, 224, 3}), cfg);
    std::int64_t macs = std::int64_t(plain.blocks.back().out) * 1000;
    for (std::size_t i = 0; i < plain.blocks.size(); ++i) {
      const auto&        b = plain.blocks[i];
      const std::int64_t n = plain.outputs[i].shape()[1];
      macs += n * n * 9 * b.in * b.out / b.groups;
    }
    check::Require(p.count() == c.train && dp.count() == c.deploy && macs == c.macs, "RepVGG parameters and multiply-adds");
    std::cout << "  RepVGG-" << c.name << ": " << std::fixed << std::setprecision(2) << p.count() / 1e6 << "M training / "
              << dp.count() / 1e6 << "M deployed parameters, " << macs / 1e9 << "G mult-adds" << std::defaultfloat
              << std::setprecision(6) << std::endl;
  }
}

void GradientChecks() {
  nn::ParameterStore   p(17), state(17);
  repvgg::RepVGGConfig cfg;
  cfg.update_stats    = false;
  const auto  net     = repvgg::RepVGG(p, state, Pattern({2, 224, 224, 3}, 1, 0), cfg, true);
  const auto  norms   = mobile_checks::GradientNorms(p, nn::mean(nn::square(net.logits)));
  double      minimum = std::numeric_limits<double>::infinity(), maximum = 0;
  std::string smallest;
  for (std::size_t i = 0; i < norms.size(); ++i) {
    check::Require(std::isfinite(norms[i]), "finite RepVGG derivative");
    if (norms[i] < minimum) {
      minimum  = norms[i];
      smallest = p.names()[i];
    }
    maximum = std::max(maximum, norms[i]);
  }
  check::Require(minimum > 1e-6 * maximum, "clearly nonzero RepVGG derivatives");
  std::cout << "  RepVGG-A0 training form: " << p.names().size() << " derivatives in [" << minimum << ',' << maximum
            << "] (smallest " << smallest << ")" << std::endl;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    for (int i = 1; i < argc; ++i)  // threads= also applies to the checks
      if (std::string(argv[i]).rfind("threads=", 0) == 0) nn::set_num_threads(std::stoi(std::string(argv[i]).substr(8)));
    BlockChecks();
    NetworkChecks();
    CostChecks();
    GradientChecks();
    image_training::ImageTrainingConfig cfg;
    cfg.name                  = "repvgg_a0_cifar10";
    cfg.maximum_fit_rows      = 12000;
    cfg.epochs                = 12;
    cfg.minimum_test_accuracy = .4;
    const image_training::LogitsBuilder builder = [](nn::ParameterStore& p, nn::ParameterStore& state,
                                                     const nn::Tensor& image, bool training) {
      repvgg::RepVGGConfig model;
      model.classes     = 10;
      model.small_stem  = true;
      const auto logits = repvgg::RepVGG(p, state, image, model, training).logits;
      check::Require(p.count() == 7840778, "CIFAR RepVGG-A0 parameter count");
      return logits;
    };
    if (image_training::RunCifar10(argc, argv, cfg, builder)) return 1;
    std::cout << "All RepVGG checks passed.\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
