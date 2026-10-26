// CMD: just run-nn condensenet
// CondenseNet: learned group convolutions condensed during training.
//
// Port of tflow's condensenet. Drive one learned group convolution (16 -> 16
// channels, G = C = 4) through the release's schedule: at progress 1/6, 1/3
// and 1/2 every group keeps 12, 8 and finally 4 input channels, and each time
// the dropped channels are exactly those with the smallest summed |w| among
// the group's still-active inputs. Convert the condensed layer to index
// selection (with BN statistics copied), BN/ReLU, a grouped 1x1 convolution
// and a channel shuffle, and check that it reproduces the masked layer
// exactly. Count CondenseNet-86 on CIFAR-10 (1,451,594 parameters while
// training; 516,202 after conversion with the classifier halved, the README's
// 0.52M; 62.4M multiply-adds against its 65.8M FLOPs) and the ImageNet
// CondenseNet-74 with G = C = 4 / 8 (4.77M / 2.94M parameters, 517M / 262M
// multiply-adds; the README's 4.8M / 2.9M and 529M / 274M). Check every
// derivative of CondenseNet-86 in training mode (moving averages and
// condensing state not updated).
// https://arxiv.org/abs/1711.09224
// https://github.com/ShichenLiu/CondenseNet
//
// Train CondenseNet-86 (14-14-14 layers, growth 8-16-32, G = C = 4, group
// lasso 1e-5) on CIFAR-10 with the schedule advancing on every training step
// (tflow runs it in the graph; here each training-mode pass updates the masks
// in the state store), so the network is fully condensed from the first step
// of epoch seven. Use 12k fit / 5k validation, twelve epochs, crop/flip,
// Adam(.001), kernel L2(.0001) and the best-validation checkpoint; the masked
// network it evaluates is the converted one, but with the full classifier
// (the release also drops half of the classifier's inputs at conversion).
// Initialization uses nn's generator, so trained numbers match TF only
// statistically. CIFAR-10 is read from the tflow dataset cache
// (common/paths.hpp) or data=.
// Options: data=PATH epochs=12 batch=128 lr=.001 fit=12000 augment=0|1 threads=N seed=17.

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <set>

#include "common/checks.hpp"
#include "common/condensenet.hpp"
#include "common/image_training.hpp"
#include "common/mobile_checks.hpp"
#include "common/shufflenet.hpp"

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

void LayerChecks() {
  const int                           in = 16, outputs = 16, groups = 4;
  nn::ParameterStore                  p(17), state(17);
  std::vector<mobilenet::MobileLayer> layers;
  const nn::Tensor                    x = Pattern({2, 5, 5, in}, 1, .1f);
  condensenet::CondenseContext        infer{p, state, layers, false};
  {
    nn::NoGradGuard guard;
    condensenet::LearnedGroupConv(infer, "lgc", x, outputs, groups, 4);
  }
  const auto weight = p.at("lgc/W").to_vector<float>();
  std::vector<std::vector<bool>> active(groups, std::vector<bool>(in, true));
  std::string                    schedule;
  for (float value : {0.f, .1f, .17f, .2f, .34f, .4f, .5f, .9f}) {
    // Expected drops from the masked weights before this step.
    std::vector<std::vector<bool>> expected = active;
    const int stage = std::min(3, int(std::floor(value * 6))), kept = in - stage * in / 4;
    int       current = 0;
    for (bool a : active[0]) current += a;
    if (kept < current)
      for (int g = 0; g < groups; ++g) {
        std::vector<std::pair<double, int>> scores;
        for (int i = 0; i < in; ++i)
          if (active[g][i]) {
            double sum = 0;
            for (int k = 0; k < outputs / groups; ++k) sum += std::abs(weight[i * outputs + k * groups + g]);
            scores.push_back({sum, i});
          }
        std::sort(scores.begin(), scores.end());
        for (int d = 0; d < in / 4; ++d) expected[g][scores[d].second] = false;
      }
    // One training-mode pass at this progress (its BN update is harmless here).
    condensenet::CondenseContext train{p, state, layers, true, .9, value};
    {
      nn::NoGradGuard guard;
      condensenet::LearnedGroupConv(train, "lgc", x, outputs, groups, 4);
    }
    const auto mask = state.at("lgc/mask").to_vector<float>();
    for (int g = 0; g < groups; ++g) {
      int count = 0;
      for (int i = 0; i < in; ++i) {
        const bool on = mask[i * outputs + g] > .5f;
        count += on;
        for (int k = 1; k < outputs / groups; ++k)
          check::Require((mask[i * outputs + k * groups + g] > .5f) == on, "group-consistent mask");
        check::Require(on == expected[g][i], "dropped channels have the smallest summed weight magnitude");
      }
      check::Require(count == kept, "channels kept per condensing stage");
    }
    active = expected;
    schedule += (schedule.empty() ? "" : ", ") + std::to_string(value).substr(0, 4) + " -> " + std::to_string(kept);
  }
  // Conversion: gather, BN with gathered statistics, ReLU, grouped 1x1,
  // shuffle, built from the condensed layer's values.
  nn::NoGradGuard guard;
  for (const auto& name : state.names())
    if (name.find("/bn/") != std::string::npos) {
      nn::Tensor values = Pattern(state.at(name).shape(), .3f, static_cast<float>(name.size()));
      if (name.find("moving_var") != std::string::npos) values = values + .8;
      state.set(name, values);
    }
  const nn::Tensor value     = Pattern({2, 5, 5, in}, 1, .3f);
  const auto       reference = condensenet::LearnedGroupConv(infer, "lgc", value, outputs, groups, 4).output.to_vector<float>();
  const auto       condensed = condensenet::Condense(p.at("lgc/W"), state.at("lgc/mask"), groups);
  const auto       width     = std::int64_t(condensed.index.size()) / groups;
  const nn::Tensor index     = nn::from_vector(condensed.index, {std::int64_t(condensed.index.size())});
  auto             pick      = [&](const std::string& name, nn::ParameterStore& store) { return nn::gather(store.at(name), index, 0); };
  const nn::Tensor gathered  = nn::gather(value, index, 3);
  const nn::Tensor normalized =
      nn::relu((gathered - pick("lgc/bn/moving_mean", state)) * (1.0 / nn::sqrt(pick("lgc/bn/moving_var", state) + 1e-5)) *
                   pick("lgc/bn/gamma", p) +
               pick("lgc/bn/beta", p));
  const auto              parts = nn::split(normalized, groups, 3);
  std::vector<nn::Tensor> convolved;
  for (int g = 0; g < groups; ++g)
    convolved.push_back(nn::conv2d(parts[g], nn::slice(condensed.weight, {0, 0, 0, g * outputs / groups}, {1, 1, width, outputs / groups}),
                                   nn::conv_options(1, nn::Padding::Valid)));
  const auto converted =
      shufflenet::PaperChannelShuffle(nn::concat(convolved, 3), outputs, groups).to_vector<float>();
  double error = 0;
  for (std::size_t i = 0; i < reference.size(); ++i) error = std::max(error, double(std::abs(reference[i] - converted[i])));
  std::set<std::int64_t> distinct(condensed.index.begin(), condensed.index.end());
  check::Near(error, 0, 1e-5, "converted group convolution equals the masked layer");
  std::cout << "  learned group convolution 16->16, G=C=4, progress -> kept inputs per group: " << schedule
            << "; drops follow summed |w|; converted: " << groups << " groups of " << width << " gathered channels ("
            << distinct.size() << " distinct inputs), BN/ReLU, grouped 1x1, shuffle; error " << error << std::endl;
}

// Inference-form cost: per LGC in*out/C weights over G groups (with G = C the
// gathered width equals the input width), 3x3 group convolutions, final BN and
// a classifier on half the features.
struct ConvertedCost {
  std::int64_t params, macs;
  int          features;
};
ConvertedCost Converted(const condensenet::CondenseNetConfig& cfg, int resolution) {
  std::int64_t params = 0, macs = 0;
  int          channels = 2 * cfg.growth[0], size = cfg.imagenet ? resolution / 2 : resolution;
  params += 27 * channels;
  macs += std::int64_t(size) * size * 27 * channels;
  for (std::size_t stage = 0; stage < cfg.stages.size(); ++stage) {
    const int k = cfg.growth[stage], wide = cfg.bottleneck * k;
    for (int layer = 0; layer < cfg.stages[stage]; ++layer, channels += k) {
      const int gathered = channels * cfg.groups / cfg.condense;
      params += 2 * gathered + std::int64_t(gathered / cfg.groups) * wide + 2 * wide + std::int64_t(wide / cfg.groups) * k * 9;
      macs += std::int64_t(size) * size * (std::int64_t(gathered / cfg.groups) * wide + std::int64_t(wide / cfg.groups) * k * 9);
    }
    if (stage + 1 < cfg.stages.size()) size /= 2;
  }
  params += 2 * channels + std::int64_t(channels / 2) * cfg.classes + cfg.classes;
  macs += std::int64_t(channels / 2) * cfg.classes;
  return {params, macs, channels};
}

void CostChecks() {
  condensenet::CondenseNetConfig cifar;
  {
    nn::ParameterStore p(17), state(17);
    nn::NoGradGuard    guard;
    const auto         net = condensenet::CondenseNet(p, state, nn::zeros({1, 32, 32, 3}), cifar);
    check::Require(p.count() == 1451594 && net.channels == 800 && state.count() == 1273850,
                   "CondenseNet-86 training parameters (and state)");
    const auto converted = Converted(cifar, 32);
    check::Require(converted.params == 516202 && converted.macs == 62377888, "CondenseNet-86 converted cost");
    std::cout << "  CondenseNet-86 (CIFAR): " << p.count() << " parameters while training (masks and stages in "
              << state.count() << " state values), " << converted.params << " after conversion, " << std::fixed
              << std::setprecision(1) << converted.macs / 1e6 << "M mult-adds; " << net.channels << " final features"
              << std::defaultfloat << std::setprecision(6) << std::endl;
  }
  for (int g : {4, 8}) {
    condensenet::CondenseNetConfig imagenet;
    imagenet.stages = {4, 6, 8, 10, 8};
    imagenet.growth = {8, 16, 32, 64, 128};
    imagenet.groups = imagenet.condense = g;
    imagenet.classes                    = 1000;
    imagenet.imagenet                   = true;
    const auto converted                = Converted(imagenet, 224);
    check::Require(converted.params == (g == 4 ? 4773944 : 2935416) && converted.macs == (g == 4 ? 516640576 : 261545792),
                   "ImageNet CondenseNet cost");
    std::cout << "  CondenseNet-74 (ImageNet, G=C=" << g << "): " << converted.params << " parameters, " << std::fixed
              << std::setprecision(1) << converted.macs / 1e6 << "M mult-adds after conversion" << std::defaultfloat
              << std::setprecision(6) << std::endl;
  }
}

void GradientChecks() {
  nn::ParameterStore             p(17), state(17);
  condensenet::CondenseNetConfig cfg;
  cfg.update_stats = false;
  const auto  net   = condensenet::CondenseNet(p, state, Pattern({2, 32, 32, 3}, 1, 0), cfg, true, 100);
  const auto  norms = mobile_checks::GradientNorms(p, nn::mean(nn::square(net.logits)));
  double      minimum = std::numeric_limits<double>::infinity(), maximum = 0;
  std::string smallest;
  for (std::size_t i = 0; i < norms.size(); ++i) {
    check::Require(std::isfinite(norms[i]), "finite CondenseNet derivative");
    if (norms[i] < minimum) {
      minimum  = norms[i];
      smallest = p.names()[i];
    }
    maximum = std::max(maximum, norms[i]);
  }
  check::Require(minimum > 1e-5 * maximum, "clearly nonzero CondenseNet derivatives");
  std::cout << "  CondenseNet-86 training mode: " << p.names().size() << " parameter derivatives in [" << minimum << ','
            << maximum << "] (smallest " << smallest << ")" << std::endl;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    // The condensing schedule needs the run's total number of training steps.
    int epochs = 12, batch = 128, fit = 12000;
    for (int i = 1; i < argc; ++i) {
      const std::string option = argv[i];
      if (option.rfind("epochs=", 0) == 0)
        epochs = std::stoi(option.substr(7));
      else if (option.rfind("batch=", 0) == 0)
        batch = std::stoi(option.substr(6));
      else if (option.rfind("fit=", 0) == 0)
        fit = std::stoi(option.substr(4));
      else if (option.rfind("threads=", 0) == 0)  // threads= also applies to the checks
        nn::set_num_threads(std::stoi(option.substr(8)));
    }
    LayerChecks();
    CostChecks();
    GradientChecks();
    epochs = example_runtime::Limit(epochs, 1);
    batch = example_runtime::Limit(batch, 8);
    fit = example_runtime::Limit(fit, 32);
    const std::int64_t                  total = std::int64_t(epochs) * ((fit + batch - 1) / batch);
    image_training::ImageTrainingConfig cfg;
    cfg.name                  = "condensenet86_cifar10";
    cfg.maximum_fit_rows      = 12000;
    cfg.epochs                = 12;
    cfg.minimum_test_accuracy = .4;
    const image_training::LogitsBuilder builder = [](nn::ParameterStore& p, nn::ParameterStore& state,
                                                     const nn::Tensor& image, bool training) {
      if (training) throw std::logic_error("CondenseNet training passes go through the training builder");
      const auto logits = condensenet::CondenseNet(p, state, image, {}).logits;
      check::Require(p.count() == 1451594, "CondenseNet-86 parameter count");
      return logits;
    };
    const image_training::TrainingBuilder training = [total](nn::ParameterStore& p, nn::ParameterStore& state,
                                                             const nn::Tensor& image) {
      auto net = condensenet::CondenseNet(p, state, image, {}, true, total);
      return image_training::TrainingNetwork{net.logits, {}, std::move(net.penalties)};
    };
    std::cout << "  condensing schedule over " << total << " training steps; fully condensed from step " << total / 2
              << std::endl;
    if (image_training::RunCifar10(argc, argv, cfg, builder, training)) return 1;
    std::cout << "All CondenseNet checks passed.\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
