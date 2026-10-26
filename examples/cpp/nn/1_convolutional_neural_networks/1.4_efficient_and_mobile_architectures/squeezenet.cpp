// CMD: just run-nn squeezenet
// SqueezeNet: AlexNet-level design with 50x fewer parameters via fire modules.
//
// Check that a fire module's concatenated 1x1 and 3x3 expand layers equal a
// single 3x3 convolution whose first filters are the 1x1 kernels placed at
// the centre tap, including input derivatives, and compare its parameters
// with a plain 3x3 convolution of the same width (the paper's strategies 1
// and 2). Check the released spatial geometry, which depends on Caffe's
// ceil-rounded pooling (54/27/13 for v1.0 and 55/27/13 for v1.1 at 224),
// the 1,248,424 / 1,235,496 learned parameters of v1.0 / v1.1 (torchvision's
// counts; simple bypasses add none) and finite nonzero derivatives.
// https://arxiv.org/abs/1602.07360
// https://github.com/forresti/SqueezeNet
//
// Train v1.0, v1.0 with simple bypasses and v1.1 at full width on CIFAR-10
// with a 3x3 stride-one first convolution, keeping all fire modules, pools,
// dropout and the convolutional classifier. Use 12k fit / 5k validation,
// twelve epochs, crop/flip, Adam(.001), kernel L2(.0001) and the
// best-validation checkpoint before evaluating all 10k test images. These are
// CIFAR adaptations, not the paper's ImageNet benchmark.
// Options: data=PATH epochs=12 batch=128 lr=.001 fit=12000 augment=0|1.

// Port: datamunge eager autograd; seeded native initialization gives statistical
// training parity. Dataset cache and training options use image_training.hpp.
// Port default lr=.0001: with native initialization the original .001 kills
// the final ReLU classifier (chance predictions and zero data gradients).
// The original learning rate remains available with lr=.001.
#include "common/squeezenet.hpp"

#include "common/architecture_checks.hpp"
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
namespace {
using mobile_checks::Pattern;
void FireChecks() {
  nn::ParameterStore p(17);
  auto               x = Pattern({2, 6, 6, 5}, 1, .1f).requires_grad_();
  squeezenet::Fire(p, "fire", x, 3, 4, 6);
  for (const auto& name : p.names())
    if (mobile_checks::EndsWith(name, "/b"))
      p.set(name, Pattern(p.at(name).shape(), .2f, name.size()));
  auto              fire    = squeezenet::Fire(p, "fire", x, 3, 4, 6);
  auto              centred = nn::pad(p.at("fire/expand1x1/W"), {{1, 1}, {1, 1}, {0, 0}, {0, 0}});
  auto              kernel  = nn::concat({centred, p.at("fire/expand3x3/W")}, 3);
  auto              bias    = nn::concat({p.at("fire/expand1x1/b"), p.at("fire/expand3x3/b")}, 0);
  nn::Conv2DOptions o;
  o.padding   = nn::Padding::Same;
  auto merged = nn::relu(nn::add(nn::conv2d(fire.squeeze, kernel, o), bias));
  auto weight = Pattern({2, 6, 6, 10}, 1, .3f);
  auto ga     = nn::grad(nn::sum(nn::mul(fire.output, weight)), {x});
  // The first backward releases its graph, so recompute the merged branch.
  fire    = squeezenet::Fire(p, "fire", x, 3, 4, 6);
  merged  = nn::relu(nn::add(nn::conv2d(fire.squeeze, kernel, o), bias));
  auto gb = nn::grad(nn::sum(nn::mul(merged, weight)), {x});
  check::Near(
      nn::max(nn::abs(nn::sub(fire.output, merged))).item(),
      0,
      1e-5,
      "fire module equals centre-embedded convolution");
  check::Near(nn::max(nn::abs(nn::sub(ga[0], gb[0]))).item(), 0, 1e-5, "fire input derivatives");
  const int fire2 = 96 * 16 + 16 + 16 * 64 + 64 + 9 * 16 * 64 + 64, plain = 9 * 96 * 128 + 128;
  std::cout << "fire2 parameters " << fire2 << " versus plain " << plain << std::endl;
}
void FullChecks() {
  struct Case {
    int              version;
    bool             bypass;
    long             params;
    std::vector<int> sizes;
  };
  for (const auto& c :
       {Case{10, false, 1248424, {54, 54, 54, 27, 27, 27, 27, 13}},
        Case{10, true, 1248424, {54, 54, 54, 27, 27, 27, 27, 13}},
        Case{11, false, 1235496, {55, 55, 27, 27, 13, 13, 13, 13}}}) {
    nn::ParameterStore           p(17);
    nn::Generator                gen(101);
    squeezenet::SqueezeNetConfig cfg;
    cfg.version       = c.version;
    cfg.simple_bypass = c.bypass;
    const auto net    = squeezenet::SqueezeNet(p, Pattern({2, 224, 224, 3}, 1, 0), cfg, true, &gen);
    check::Require(
        p.count() == c.params && net.sizes == c.sizes, "SqueezeNet count and ceil geometry");
    const int stem = c.version == 10 ? 109 : 111;
    check::Require(
        net.stem.size(1) == stem && net.fires.back().size(1) == 13 &&
            net.fires.back().size(3) == 512 && net.logits.shape() == nn::Shape({2, 1000}),
        "SqueezeNet stem and classifier dimensions");
    auto norms = architecture::GradientNormRange(p, nn::mean(nn::square(net.logits)), "SqueezeNet");
    std::cout << "SqueezeNet " << c.version << " bypass=" << c.bypass << " parameters=" << p.count()
              << " smallest gradient=" << norms.first << std::endl;
  }
}
} // namespace
int main(int argc, char** argv) {
  try {
    nn::set_num_threads(2);
    FireChecks();
    FullChecks();
    struct Run {
      const char* name;
      int         version;
      bool        bypass;
      long        params;
    };
    for (const auto& run :
         {Run{"squeezenet10_cifar10", 10, false, 729034},
          Run{"squeezenet10_bypass_cifar10", 10, true, 729034},
          Run{"squeezenet11_cifar10", 11, false, 727626}}) {
      image_training::ImageTrainingConfig cfg;
      cfg.name                  = run.name;
      cfg.maximum_fit_rows      = 12000;
      cfg.learning_rate         = .0001;
      cfg.epochs                = 12;
      cfg.minimum_test_accuracy = .4;
      nn::Generator                       generator(101);
      const image_training::LogitsBuilder builder = [run, &generator](
                                                        nn::ParameterStore& p,
                                                        nn::ParameterStore& state,
                                                        const nn::Tensor&   image,
                                                        bool                training) {
        squeezenet::SqueezeNetConfig model;
        model.version       = run.version;
        model.simple_bypass = run.bypass;
        model.classes       = 10;
        model.image_size    = 32;
        model.cifar_stem    = true;
        auto net            = squeezenet::SqueezeNet(p, image, model, training, &generator);
        check::Require(
            p.count() == run.params && state.count() == 0, "CIFAR SqueezeNet parameter count");
        return net.logits;
      };
      if (image_training::RunCifar10(argc, argv, cfg, builder))
        return 1;
    }
    std::cout << "All SqueezeNet checks passed.\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
