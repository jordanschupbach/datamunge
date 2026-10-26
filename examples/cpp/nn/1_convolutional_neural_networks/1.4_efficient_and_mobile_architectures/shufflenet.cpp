// CMD: just run-nn shufflenet
// ShuffleNet v1 and v2: grouped pointwise convolutions with channel shuffle, and channel split.
//
// Check the released channel shuffle and the paper's transpose exactly, and
// that they are inverse permutations. Reproduce the paper's Figure 1
// argument with derivative supports: through two stacked group
// convolutions, an output channel depends on one input group, and on every
// group once a shuffle sits between them. Check that v1's strided shortcut
// counts zero padding in its average and that a v2 unit passes its even
// channels through unchanged. Count parameters and multiply-adds for all
// released sizes: v1 (3 groups) 38/137/292/524M, v1 (8 groups)
// 40/138/290/522M and v2 41/146/299/591M at 0.5x/1x/1.5x/2x, matching the
// release's table. Check every derivative with shifted BN betas: all are
// clearly nonzero except the depthwise BN shifts, which only reach 1x1
// convolutions followed by training-mode BN.
// https://arxiv.org/abs/1707.01083
// https://arxiv.org/abs/1807.11164
// https://github.com/megvii-model/ShuffleNet-Series
//
// Train v1 1x (3 groups) and v2 1x on CIFAR-10 without the stem's stride and
// max-pool (all sixteen units per network). Use 12k fit / 5k validation,
// twelve epochs, crop/flip, Adam(.001), kernel L2(.0001) and the
// best-validation checkpoint before evaluating all 10k test images.
// Options: data=PATH epochs=12 batch=128 lr=.001 fit=12000 augment=0|1.

// Eager port with native seeded initialization and statistical training parity.
#include "common/shufflenet.hpp"

#include <set>

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
void ShuffleChecks() {
  constexpr int      C = 12, G = 3, n = C / G;
  std::vector<float> values(48);
  for (int i = 0; i < 48; ++i)
    values[i] = i % C;
  auto x        = nn::from_vector(values, {1, 2, 2, C}).requires_grad_();
  auto release  = shufflenet::ChannelShuffle(x, C, G).to_vector<float>();
  auto paper    = shufflenet::PaperChannelShuffle(x, C, G);
  auto pv       = paper.to_vector<float>(),
       composed = shufflenet::ChannelShuffle(paper, C, G).to_vector<float>();
  for (int c = 0; c < C; ++c) {
    check::Require(release[(c % G) * n + c / G] == c, "release shuffle order");
    check::Require(pv[(c % n) * G + c / n] == c, "paper shuffle order");
    check::Require(composed[c] == c, "inverse shuffles");
  }
  nn::ParameterStore   p(17);
  nn::ConvLayerOptions o;
  o.bias     = false;
  o.init     = nn::init::he_uniform();
  auto build = [&](bool shuffled) {
    auto first = grouped::GroupedConv2D(p, "first", x, C, 1, G, o);
    if (shuffled)
      first = shufflenet::ChannelShuffle(first, C, G);
    auto y = grouped::GroupedConv2D(p, "second", first, C, 1, G, o);
    return nn::grad(nn::sum(nn::slice(y, {0, 0, 0, 0}, {1, 2, 2, 1})), {x})[0].to_vector<float>();
  };
  auto          plain = build(false), shuffled = build(true);
  int           support = 0, shuffled_support = 0;
  std::set<int> reached;
  for (int c = 0; c < C; ++c) {
    double a = 0, b = 0;
    for (int pixel = 0; pixel < 4; ++pixel) {
      a += std::abs(plain[pixel * C + c]);
      b += std::abs(shuffled[pixel * C + c]);
    }
    support += a > 0;
    shuffled_support += b > 0;
    if (b > 0)
      reached.insert(c / n);
  }
  check::Require(support == n && reached.size() == G, "shuffle connects every group");
  std::cout << "derivative supports plain=" << support << " shuffled=" << shuffled_support
            << std::endl;
}
void UnitChecks() {
  nn::ParameterStore         p(17), state(17);
  shufflenet::ShuffleContext c{p, state, false};
  auto                       x      = Pattern({1, 6, 6, 8}, 1, .3f);
  auto                       v      = x.to_vector<float>();
  auto                       pooled = shufflenet::AveragePoolWithPadding(x).to_vector<float>();
  auto                       v2 = shufflenet::ShuffleV2Unit(c, "v2", x, 8, 1).to_vector<float>();
  double                     corner = 0, passed = 0;
  for (int ch = 0; ch < 8; ++ch)
    corner =
        std::max(corner, std::abs(pooled[ch] - (v[ch] + v[8 + ch] + v[48 + ch] + v[56 + ch]) / 9.));
  for (int pixel = 0; pixel < 36; ++pixel)
    for (int ch = 0; ch < 4; ++ch)
      passed = std::max(passed, double(std::abs(v2[pixel * 8 + ch] - v[pixel * 8 + 2 * ch])));
  check::Near(corner, 0, 1e-6, "average shortcut includes zero padding");
  check::Near(passed, 0, 0, "v2 passes even channels unchanged");
}
void CostChecks() {
  struct Case {
    int         version, groups;
    const char* size;
    long        params, state, macs;
  };
  const std::vector<Case> cases = {
      {1, 3, "0.5x", 717324, 12240, 37589856},
      {1, 3, "1.0x", 1864728, 24480, 137460672},
      {1, 3, "1.5x", 3441168, 36720, 292443552},
      {1, 3, "2.0x", 5449776, 48960, 524045184},
      {1, 8, "0.5x", 1010448, 19584, 40469760},
      {1, 8, "1.0x", 2433768, 39168, 138365952},
      {1, 8, "1.5x", 4268952, 58752, 290188800},
      {1, 8, "2.0x", 6521232, 78336, 521942016},
      {2, 0, "0.5x", 1365792, 7952, 40476448},
      {2, 0, "1.0x", 2278760, 16180, 145814296},
      {2, 0, "1.5x", 3506720, 23440, 298970656},
      {2, 0, "2.0x", 7402600, 33716, 590783000}};
  for (const auto& c : cases) {
    nn::NoGradGuard              guard;
    nn::ParameterStore           p(17), state(17);
    shufflenet::ShuffleNetConfig cfg;
    cfg.version = c.version;
    cfg.groups  = c.groups ? c.groups : 1;
    cfg.size    = c.size;
    auto net    = shufflenet::ShuffleNet(p, state, nn::zeros({1, 224, 224, 3}), cfg);
    long macs   = net.classifier_macs;
    for (const auto& [shape, cost] : net.layers)
      macs += shape[1] * shape[2] * cost;
    check::Require(
        net.features.size(1) == 7 && p.count() == c.params && state.count() == c.state &&
            macs == c.macs,
        "ShuffleNet parameters, MACs and 7x7 features");
    std::cout << "ShuffleNet v" << c.version << ' ' << c.size << " groups=" << c.groups
              << " params=" << p.count() << " MACs=" << macs << std::endl;
  }
}
void GradientChecks(int version) {
  nn::ParameterStore           p(17), state(17);
  shufflenet::ShuffleNetConfig cfg;
  cfg.update_stats = false;
  cfg.version      = version;
  const auto x     = Pattern({2, 224, 224, 3}, 1, 0);
  {
    nn::NoGradGuard guard;
    shufflenet::ShuffleNet(p, state, x, cfg, true);
    mobile_checks::ShiftBetas(p, false);
  }
  auto net = shufflenet::ShuffleNet(p, state, x, cfg, true);
  check::Require(net.logits.shape() == nn::Shape({2, 1000}), "ShuffleNet output dimensions");
  auto   norms   = mobile_checks::GradientNorms(p, nn::mean(nn::square(net.logits)));
  double maximum = *std::max_element(norms.begin(), norms.end()), minimum = INFINITY, removed = 0;
  int    count = 0;
  auto   names = p.names();
  for (std::size_t i = 0; i < norms.size(); ++i) {
    check::Require(std::isfinite(norms[i]), "finite ShuffleNet derivative");
    if (mobile_checks::EndsWith(names[i], "/depthwise/bn/beta")) {
      removed = std::max(removed, norms[i]);
      ++count;
    } else
      minimum = std::min(minimum, norms[i]);
  }
  check::Require(
      count == (version == 1 ? 16 : 19) && removed < 1e-6 * maximum,
      "vanishing projection BN shifts");
  check::Require(minimum > 1e-5 * maximum, "nonzero ShuffleNet derivatives with shifted BN");
  std::cout << "gradient range " << minimum << " to " << maximum << " removed shifts " << removed
            << std::endl;
}
} // namespace
int main(int argc, char** argv) {
  try {
    nn::set_num_threads(2);
    ShuffleChecks();
    UnitChecks();
    CostChecks();
    GradientChecks(1);
    GradientChecks(2);
    for (int version : {1, 2}) {
      image_training::ImageTrainingConfig cfg;
      cfg.name             = version == 1 ? "shufflenet_v1_g3_cifar10" : "shufflenet_v2_cifar10";
      cfg.maximum_fit_rows = 12000;
      cfg.epochs           = 12;
      cfg.minimum_test_accuracy                   = .4;
      const image_training::LogitsBuilder builder = [version](
                                                        nn::ParameterStore& p,
                                                        nn::ParameterStore& state,
                                                        const nn::Tensor&   image,
                                                        bool                training) {
        shufflenet::ShuffleNetConfig model;
        model.version    = version;
        model.classes    = 10;
        model.small_stem = true;
        auto net         = shufflenet::ShuffleNet(p, state, image, model, training);
        check::Require(
            p.count() == (version == 1 ? 914328 : 1265000), "CIFAR ShuffleNet parameter count");
        return net.logits;
      };
      if (image_training::RunCifar10(argc, argv, cfg, builder))
        return 1;
    }
    std::cout << "All ShuffleNet checks passed.\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
