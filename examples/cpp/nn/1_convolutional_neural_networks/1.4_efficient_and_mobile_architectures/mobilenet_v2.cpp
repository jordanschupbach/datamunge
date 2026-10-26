// CMD: just run-nn mobilenet_v2
// MobileNet v2: inverted residuals with linear bottlenecks.
//
// Reproduce the paper's Figure 1 argument: embed a 2-D spiral in n
// dimensions with a random matrix T, apply ReLU and recover each point by
// least squares over its still-active coordinates. Few dimensions lose much
// of the manifold (too few coordinates stay positive); many keep it exactly,
// which motivates expanding before the ReLU and projecting linearly. Check
// that an inverted residual with a zeroed projection BN is the identity with
// identity input derivatives (no ReLU after the addition, so negative values
// survive) and that expansion, stride and projection shapes follow Table 2.
// Measure parameters and multiply-adds for widths 1.4/1/.75/.5/.35 and
// resolutions 224-96: 582/300/209/97/59M and 221/154/99/56M mult-adds,
// 6.06/3.47/2.61/1.95/1.66M weights without BN (the TF-slim release table;
// torchvision's 3,504,872 learned parameters for width 1 include BN). Check
// every derivative of the full model in training mode with shifted BN betas:
// all are clearly nonzero except the 17 projection BN shifts, whose constant
// offsets every later 1x1 convolution passes to a training-mode BN.
// https://arxiv.org/abs/1801.04381
// https://github.com/tensorflow/models/tree/master/research/slim/nets/mobilenet
//
// Train width 1 on CIFAR-10 with a stride-one first convolution (all
// seventeen inverted residuals, four remaining downsamplings). Use 12k fit /
// 5k validation, twelve epochs, crop/flip, Adam(.001), kernel L2(.0001) and
// the best-validation checkpoint before evaluating all 10k test images.
// Options: data=PATH epochs=12 batch=128 lr=.001 fit=12000 augment=0|1.

// Eager port; native seeded initialization, statistical training parity.
#include <random>

#include "common/image_training.hpp"
#include "common/mobile_checks.hpp"
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
using mobile_checks::Pattern;
void ManifoldChecks() {
  const int          points = 400, trials = 20;
  std::vector<float> spiral(2 * points);
  for (int i = 0; i < points; ++i) {
    const float angle = 3 * std::acos(-1.) * i / points, radius = .2f + angle;
    spiral[i]          = radius * std::cos(angle);
    spiral[points + i] = radius * std::sin(angle);
  }
  auto                            x = nn::from_vector(spiral, {2, points});
  std::mt19937                    random(17);
  std::normal_distribution<float> normal;
  std::vector<double>             errors;
  nn::NoGradGuard                 guard;
  for (int n : {2, 3, 5, 15, 30}) {
    double total = 0;
    for (int trial = 0; trial < trials; ++trial) {
      std::vector<float> values(n * 2);
      for (auto& v : values)
        v = normal(random);
      auto t      = nn::from_vector(values, {n, 2});
      auto z      = nn::matmul(t, x);
      auto active = nn::expand_dims(
          nn::transpose(
              nn::cast(nn::greater(z, nn::scalar(0, nn::DType::Float32)), nn::DType::Float32),
              {1, 0}),
          2);
      auto rows = nn::mul(nn::expand_dims(t, 0), active);
      auto rhs  = nn::expand_dims(nn::transpose(nn::relu(z), {1, 0}), 2);
      auto recovered =
          nn::transpose(nn::squeeze(nn::matrix_solve_ls(rows, rhs, 1e-6), {2}), {1, 0});
      total += nn::sqrt(nn::sum(nn::square(nn::sub(recovered, x)))).item() /
               nn::sqrt(nn::sum(nn::square(x))).item();
    }
    errors.push_back(total / trials);
  }
  check::Require(
      errors[0] > errors[1] && errors[1] > errors[2] && errors[2] > errors[3] && errors[0] > .3 &&
          errors[3] < 1e-3,
      "ReLU manifold recovery improves in higher dimensions");
  std::cout << "manifold relative errors:";
  for (auto e : errors)
    std::cout << ' ' << e;
  std::cout << std::endl;
}
void BlockChecks() {
  nn::ParameterStore                  p(17), state(17);
  std::vector<mobilenet::MobileLayer> layers;
  mobilenet::MobileContext            c{p, state, layers, false};
  auto                                x = Pattern({2, 8, 8, 4}, 1, .9f).requires_grad_();
  mobilenet::InvertedResidual(c, "kept", x, 4, 6, 1);
  for (const auto& part : {"gamma", "beta"})
    p.set(std::string("kept/project/bn/") + part, nn::zeros({4}));
  auto kept    = mobilenet::InvertedResidual(c, "kept", x, 4, 6, 1);
  auto strided = mobilenet::InvertedResidual(c, "strided", x, 6, 6, 2);
  auto weights = Pattern({2, 8, 8, 4}, 1, .2f);
  auto g       = nn::grad(nn::sum(nn::mul(kept.output, weights)), {x});
  check::Near(nn::max(nn::abs(nn::sub(kept.output, x))).item(), 0, 0, "zero projection identity");
  check::Near(nn::max(nn::abs(nn::sub(g[0], weights))).item(), 0, 0, "identity input derivatives");
  check::Require(nn::min(kept.output).item() < 0, "negative values survive bottleneck");
  check::Require(
      kept.expanded.size(3) == 24 && strided.filtered.shape() == nn::Shape({2, 4, 4, 24}) &&
          strided.output.shape() == nn::Shape({2, 4, 4, 6}),
      "inverted residual shapes");
}
void CostChecks() {
  struct Case {
    double alpha;
    int    resolution;
    long   params, state, macs;
  };
  for (const auto& c :
       {Case{1.4, 224, 6108776, 47936, 582195824},
        Case{1, 224, 3504872, 34112, 300774272},
        Case{.75, 224, 2636424, 26640, 209069792},
        Case{.5, 224, 1968680, 18544, 97131840},
        Case{.35, 224, 1677128, 14080, 59285808},
        Case{1, 192, 3504872, 34112, 221316608},
        Case{1, 160, 3504872, 34112, 154083200},
        Case{1, 128, 3504872, 34112, 99074048},
        Case{1, 96, 3504872, 34112, 56289152}}) {
    nn::NoGradGuard              guard;
    nn::ParameterStore           p(17), state(17);
    mobilenet::MobileNetV2Config cfg;
    cfg.alpha = c.alpha;
    auto net = mobilenet::MobileNetV2(p, state, nn::zeros({1, c.resolution, c.resolution, 3}), cfg);
    long macs = net.classifier_macs;
    for (const auto& l : net.layers)
      macs += l.Macs();
    check::Require(
        p.count() == c.params && state.count() == c.state && macs == c.macs,
        "MobileNet v2 parameters and multiply-adds");
    std::cout << "width " << c.alpha << " resolution " << c.resolution << " params " << p.count()
              << " MACs " << macs << std::endl;
  }
}
void GradientChecks() {
  nn::ParameterStore           p(17), state(17);
  mobilenet::MobileNetV2Config cfg;
  cfg.update_stats = false;
  const auto x     = Pattern({2, 224, 224, 3}, 1, 0);
  {
    nn::NoGradGuard guard;
    mobilenet::MobileNetV2(p, state, x, cfg, true);
    mobile_checks::ShiftBetas(p, false);
  }
  auto net = mobilenet::MobileNetV2(p, state, x, cfg, true);
  check::Require(
      net.features.shape() == nn::Shape({2, 7, 7, 1280}) &&
          net.logits.shape() == nn::Shape({2, 1000}),
      "MobileNet v2 output dimensions");
  auto   norms   = mobile_checks::GradientNorms(p, nn::mean(nn::square(net.logits)));
  double maximum = *std::max_element(norms.begin(), norms.end()), minimum = INFINITY, removed = 0;
  int    count = 0;
  auto   names = p.names();
  for (std::size_t i = 0; i < norms.size(); ++i) {
    check::Require(std::isfinite(norms[i]), "finite MobileNet v2 derivative");
    if (mobile_checks::EndsWith(names[i], "/project/bn/beta")) {
      removed = std::max(removed, norms[i]);
      ++count;
    } else
      minimum = std::min(minimum, norms[i]);
  }
  check::Require(count == 17 && removed < 1e-6 * maximum, "vanishing projection BN shifts");
  check::Require(minimum > 1e-5 * maximum, "nonzero MobileNet v2 derivatives with shifted BN");
  std::cout << "gradient range " << minimum << " to " << maximum << " removed shifts " << removed
            << std::endl;
}
} // namespace
int main(int argc, char** argv) {
  try {
    nn::set_num_threads(2);
    ManifoldChecks();
    BlockChecks();
    CostChecks();
    GradientChecks();
    image_training::ImageTrainingConfig cfg;
    cfg.name                                    = "mobilenet_v2_cifar10";
    cfg.maximum_fit_rows                        = 12000;
    cfg.epochs                                  = 12;
    cfg.minimum_test_accuracy                   = .4;
    const image_training::LogitsBuilder builder = [](nn::ParameterStore& p,
                                                     nn::ParameterStore& state,
                                                     const nn::Tensor&   image,
                                                     bool                training) {
      mobilenet::MobileNetV2Config model;
      model.classes    = 10;
      model.small_stem = true;
      auto net         = mobilenet::MobileNetV2(p, state, image, model, training);
      check::Require(
          p.count() == 2236682 && state.count() == 34112, "CIFAR MobileNet v2 parameter count");
      return net.logits;
    };
    if (image_training::RunCifar10(argc, argv, cfg, builder))
      return 1;
    std::cout << "All MobileNet v2 checks passed.\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
