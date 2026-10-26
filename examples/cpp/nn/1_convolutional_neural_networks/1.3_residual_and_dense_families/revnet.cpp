// CMD: just run-nn revnet
// RevNet: reversible residual units, trained without storing activations.
//
// Port of tflow's revnet. Follow the authors' release: units split channels
// into halves with y1 = x1' + F(x2), y2 = x2' + G(y1), and every equal-width
// unit is inverted as x2 = y2 - G(y1), x1 = y1 - F(x2). Check inversion of
// basic and bottleneck units in training and inference BN modes. On
// RevNet-38 and RevNet-164, compare every parameter derivative of the
// memory-saving backward pass (reconstruct each unit's input from its
// output, recompute, backpropagate one unit) with ordinary automatic
// differentiation. Run on stored inputs (every unit an nn::checkpoint), the
// unit-by-unit pass must reproduce autodiff exactly; with reconstruction
// (nn::reversible_sequence), report the drift of the rebuilt activations and
// how much it moves the derivatives (float32 drift of ~1e-5 grows to about
// 1% in RevNet-164's earliest units). Measure peak tensor memory during one
// training step for 3/9/18 units per stage: ordinary backpropagation stores
// every unit's activations, while the reversible pass keeps only the stored
// stage inputs and one unit's recomputation at a time. tflow measures the
// process's peak resident set (Linux only); datamunge counts live tensor
// bytes exactly (nn/memory.hpp), on every platform.
// Count RevNet-38/110/164 (0.46M / 1.73M / 1.75M in the paper) and ImageNet
// RevNet-56/104 against an independent count of the released layout (the
// ImageNet models are counted on 64-pixel inputs; counts do not depend on it).
// https://arxiv.org/abs/1707.04585
// https://github.com/renmengye/revnet-public
//
// Train the full RevNet-38 on CIFAR-10 using the reversible backward pass
// for all of its gradients. Use 12k fit / 5k validation, twelve epochs,
// crop/flip, Adam(.001), kernel L2(.0001) and the best-validation checkpoint
// before evaluating all 10k test images. The paper trains with SGD for 80k
// steps on all 50k images; this short run does not reproduce its error rate.
// Options: data=PATH epochs=12 batch=128 lr=.001 fit=12000 augment=0|1 threads=N seed=17.

#include <datamunge/nn/memory.hpp>
#include <iomanip>

#include "common/image_training.hpp"
#include "common/revnet.hpp"

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



namespace {

namespace nn = datamunge::nn;
using nn::Tensor;

Tensor Pattern(const nn::Shape& shape, float scale, float phase) {
  std::vector<float> v(static_cast<std::size_t>(nn::shape_numel(shape)));
  for (std::size_t i = 0; i < v.size(); ++i) v[i] = scale * std::sin(.37f * static_cast<float>(i) + phase);
  return nn::from_vector(v, shape);
}
double RelativeDifference(const Tensor& a, const Tensor& b) {
  check::Require(a.shape() == b.shape(), "matching RevNet comparison shapes");
  const auto x = a.to_vector<float>(), y = b.to_vector<float>();
  double     difference = 0, norm = 0;
  for (std::size_t i = 0; i < x.size(); ++i) {
    const double d = double(x[i]) - y[i];
    difference += d * d;
    norm += double(y[i]) * y[i];
  }
  return std::sqrt(difference / std::max(norm, 1e-30));
}

void InverseChecks() {
  for (bool bottleneck : {false, true})
    for (bool training : {true, false}) {
      // tflow seeds this fixture with 17. With nn's generator, seed 17 leaves
      // the one-channel bottleneck's inference-mode ReLU almost dead (the unit
      // changes its input by only .06), so the fixture uses seed 1 (>.2 for
      // seeds 1-7 except 17's stream).
      nn::ParameterStore p(1), state(1);
      const revnet::RevContext c{p, state, training, .9, bottleneck, false};
      const Tensor             value  = Pattern({4, 5, 5, 8}, 1, .2f);
      auto                     halves = nn::split(value, 2, 3);
      {
        nn::NoGradGuard guard;  // create the parameters, then give the statistics nontrivial values
        revnet::RevBlockForward(c, "unit", {halves[0], halves[1]}, 8, 8, 1, false);
        for (const auto& name : state.names()) {
          Tensor values = Pattern(state.at(name).shape(), .3f, static_cast<float>(name.size()));
          if (name.find("moving_var") != std::string::npos) values = values + .8;
          state.set(name, values);
        }
      }
      nn::NoGradGuard guard;
      const auto      y        = revnet::RevBlockForward(c, "unit", {halves[0], halves[1]}, 8, 8, 1, false);
      const auto      x        = revnet::RevBlockInverse(c, "unit", y, 8);
      const Tensor    restored = nn::concat({x.first, x.second}, 3);
      const double    error    = nn::max(nn::abs(restored - value)).item();
      const double    change   = nn::max(nn::abs(y.first - halves[0])).item();
      check::Require(change > .1, "the unit changes its input");
      check::Near(error, 0, 1e-5, "RevNet unit inverse restores its input");
      std::cout << "  " << (bottleneck ? "bottleneck" : "basic") << " unit, " << (training ? "training" : "inference")
                << " BN: inverse error=" << error << ", change=" << change << std::endl;
    }
}

std::pair<Tensor, Tensor> Batch(int batch) {
  std::vector<std::int64_t> labels(static_cast<std::size_t>(batch));
  for (int i = 0; i < batch; ++i) labels[static_cast<std::size_t>(i)] = (7 * i + 3) % 10;
  return {Pattern({batch, 32, 32, 3}, 1, .5f), nn::from_vector(labels)};
}
Tensor Loss(const Tensor& logits, const Tensor& labels, int classes) {
  return -nn::mean(nn::sum(nn::one_hot(labels, classes, 1.0, 0.0, logits.dtype()) * nn::log_softmax(logits, 1), {1}));
}

void GradientChecks() {
  for (int depth : {38, 164}) {
    const auto         cfg = revnet::RevNetConfig::Cifar(depth);
    nn::ParameterStore p(17), state(17);
    const auto         batch = Batch(8);
    const Tensor       x = batch.first, labels = batch.second;
    auto derivatives       = [&](revnet::Mode mode, revnet::RevNetOutputs* keep) {
      auto net = revnet::RevNet(p, state, x, cfg, true, mode, false);
      auto g   = nn::grad(Loss(net.logits, labels, cfg.classes), p.parameters());
      if (keep) *keep = std::move(net);
      return g;
    };
    revnet::RevNetOutputs net;
    const auto            standard   = derivatives(revnet::Mode::Ordinary, &net);
    const auto            reversible = derivatives(revnet::Mode::Reversible, nullptr);
    const auto            stored     = derivatives(revnet::Mode::StoredUnits, nullptr);
    // The reconstructions the reversible pass sees: walk back from the last
    // output, inverting every invertible unit and taking stored inputs as is.
    double drift = 0;
    int    rebuilt = 0;
    {
      nn::NoGradGuard          guard;
      const revnet::RevContext c{p, state, true, cfg.momentum, cfg.bottleneck, false};
      revnet::RevPair          y{net.units.back().output.first.detach(), net.units.back().output.second.detach()};
      for (std::size_t k = net.units.size(); k-- > 0;) {
        const auto& u = net.units[k];
        if (u.stored) {
          y = {u.input.first.detach(), u.input.second.detach()};
          continue;
        }
        y = revnet::RevBlockInverse(c, u.name, y, u.out);
        drift = std::max(drift, RelativeDifference(nn::concat({y.first, y.second}, 3),
                                                   nn::concat({u.input.first, u.input.second}, 3)));
        ++rebuilt;
      }
    }
    const std::size_t n = standard.size();
    double            gradient = 0, unit = 0;
    std::string       worst;
    for (std::size_t i = 0; i < n; ++i) {
      const double d = RelativeDifference(reversible[i], standard[i]);
      if (d > gradient) {
        gradient = d;
        worst    = p.names()[i];
      }
      unit = std::max(unit, RelativeDifference(stored[i], standard[i]));
    }
    check::Near(unit, 0, 1e-6, "unit-by-unit derivatives from stored inputs equal ordinary backprop");
    check::Near(drift, 0, 1e-4, "reconstructed unit inputs match the forward pass");
    check::Near(gradient, 0, .05, "reversible derivatives stay close to ordinary backprop (worst: " + worst + ")");
    std::cout << "  RevNet-" << depth << ": " << n << " derivatives; unit-by-unit from stored inputs vs ordinary backprop: "
              << unit << "; " << rebuilt << " reconstructed inputs drift up to " << drift
              << ", which moves the reversible derivatives by up to " << gradient << " (" << worst << ")" << std::endl;
  }
}

// Peak live tensor bytes of one training step (forward and gradients) above
// the parameters, after a warm-up step that creates them.
double PeakMiB(const revnet::RevNetConfig& cfg, revnet::Mode mode, int batch) {
  nn::ParameterStore p(17), state(17);
  const auto         data = Batch(batch);
  const Tensor       x = data.first, labels = data.second;
  auto step              = [&] {
    auto net = revnet::RevNet(p, state, x, cfg, true, mode, false);
    return nn::grad(Loss(net.logits, labels, cfg.classes), p.parameters());
  };
  step();
  nn::PeakMemoryScope scope;
  step();
  return scope.peak_above_baseline() / 1048576.0;
}
void MemoryChecks() {
  std::vector<double> ratios;
  for (int n : {3, 9, 18}) {
    auto cfg  = revnet::RevNetConfig::Cifar(38);
    cfg.units = {n, n, n};
    const double a = PeakMiB(cfg, revnet::Mode::Ordinary, 64), b = PeakMiB(cfg, revnet::Mode::Reversible, 64),
                 s = PeakMiB(cfg, revnet::Mode::StoredUnits, 64);
    check::Require(a > 0 && b > 0, "measured peak step memory");
    ratios.push_back(a / b);
    std::cout << "  " << n << " units per stage (" << 12 * n + 2 << " layers), batch 64: peak step memory ordinary="
              << std::fixed << std::setprecision(1) << a << " MiB, reversible=" << b << " MiB (every unit checkpointed="
              << s << " MiB)" << std::defaultfloat << std::setprecision(6) << std::endl;
  }
  check::Require(ratios[0] < ratios[1] && ratios[1] < ratios[2] && ratios[2] > 3, "reversible saving grows with depth");
}

void CountChecks() {
  struct Case {
    const char*          name;
    revnet::RevNetConfig cfg;
    int                  size;
    std::int64_t         params, state;
  };
  const std::vector<Case> cases = {{"RevNet-38", revnet::RevNetConfig::Cifar(38), 32, 464890, 2672},
                                   {"RevNet-110", revnet::RevNetConfig::Cifar(110), 32, 1729194, 8256},
                                   {"RevNet-164", revnet::RevNetConfig::Cifar(164), 32, 1748778, 24768},
                                   {"ImageNet RevNet-56", revnet::RevNetConfig::ImageNet(56), 64, 27553128, 51200},
                                   {"ImageNet RevNet-104", revnet::RevNetConfig::ImageNet(104), 64, 45428072, 100352}};
  for (const auto& c : cases) {
    nn::ParameterStore p(17), state(17);
    nn::NoGradGuard    guard;
    const auto         net = revnet::RevNet(p, state, nn::zeros({1, c.size, c.size, 3}), c.cfg);
    check::Require(p.count() == c.params && state.count() == c.state, std::string(c.name) + " parameter/state counts");
    int stored = 0;
    for (const auto& u : net.units) stored += u.stored;
    std::cout << "  " << c.name << ": parameters=" << p.count() << ", state=" << state.count() << "; "
              << net.units.size() << " units, " << stored << " with stored inputs" << std::endl;
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    InverseChecks();
    GradientChecks();
    MemoryChecks();
    CountChecks();
    image_training::ImageTrainingConfig cfg;
    cfg.name                  = "revnet38_cifar10";
    cfg.maximum_fit_rows      = 12000;
    cfg.epochs                = 12;
    cfg.minimum_test_accuracy = .45;
    const auto model = revnet::RevNetConfig::Cifar(38);
    const auto builder = [model](nn::ParameterStore& p, nn::ParameterStore& state, const Tensor& input, bool training) {
      // Training uses the reversible backward pass for every gradient.
      auto net = revnet::RevNet(p, state, input, model, training,
                                training ? revnet::Mode::Reversible : revnet::Mode::Ordinary);
      check::Require(p.count() == 464890 && state.count() == 2672, "RevNet-38 parameter/state counts");
      return net.logits;
    };
    if (image_training::RunCifar10(argc, argv, cfg, builder)) return 1;
    std::cout << "All RevNet checks passed.\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
