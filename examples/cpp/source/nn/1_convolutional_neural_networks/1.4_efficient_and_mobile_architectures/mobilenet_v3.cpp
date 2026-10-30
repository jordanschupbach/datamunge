// CMD: just run-nn mobilenet_v3
// MobileNet v3 Large and Small: searched blocks, hard-swish and squeeze-and-excitation.
//
// Compare hard-swish and the hard sigmoid with swish and the logistic
// function they replace (exact zeros below -3, exact identity / one above 3).
// Check that a squeeze-and-excitation module with a zeroed second layer
// scales every channel by relu6(bias+3)/6. Measure the released Large and
// Small models at 224 pixels: 5,483,032 / 2,542,856 learned parameters
// (torchvision's counts) and 216.6M / 56.5M convolution and dense
// multiply-adds (the paper reports about 219M / 56M). Check every derivative
// in training mode with shifted BN betas and SE biases: all are clearly
// nonzero except the projection BN shifts, whose per-channel constants only
// reach 1x1 convolutions followed by training-mode BN.
// https://arxiv.org/abs/1905.02244
// https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv3.py
//
// Train Large and Small on CIFAR-10 with a stride-one first convolution
// (every block, remaining downsamplings and the efficient last stage). Use
// 12k fit / 5k validation, twelve epochs, crop/flip, Adam(.001), kernel
// L2(.0001) and the best-validation checkpoint before evaluating all 10k
// test images. These are CIFAR adaptations, not the ImageNet benchmark.
// Options: data=PATH epochs=12 batch=128 lr=.001 fit=12000 augment=0|1.

// Port: native seeded initialization; statistical training parity.
#include <chrono>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>

#include "common/image_training.hpp"
#include "common/mobile_checks.hpp"
#include "common/mobilenet.hpp"

namespace nn = datamunge::nn;
namespace {
using mobile_checks::Pattern;
void ActivationChecks() {
  std::vector<float> values(1601);
  for (int i = 0; i < 1601; ++i)
    values[i] = -8 + .01f * i;
  auto x    = nn::from_vector(values, {1601});
  auto hs   = mobilenet::HardSwish(x).to_vector<float>(),
       hsig = mobilenet::HardSigmoid(x).to_vector<float>();
  auto sw = nn::mul(x, nn::sigmoid(x)).to_vector<float>(), sig = nn::sigmoid(x).to_vector<float>();
  double swish = 0, sigmoid = 0, saturation = 0;
  for (int i = 0; i < 1601; ++i) {
    swish   = std::max(swish, double(std::abs(hs[i] - sw[i])));
    sigmoid = std::max(sigmoid, double(std::abs(hsig[i] - sig[i])));
    if (values[i] <= -3)
      saturation = std::max({saturation, double(std::abs(hs[i])), double(std::abs(hsig[i]))});
    if (values[i] >= 3)
      saturation = std::max(
          {saturation, double(std::abs(hs[i] - values[i])), double(std::abs(hsig[i] - 1))});
  }
  check::Near(saturation, 0, 0, "hard activations saturate exactly");
  check::Require(swish < .15 && sigmoid < .1, "hard activation approximation");
  std::cout << "hard activation approximation errors " << swish << ' ' << sigmoid << std::endl;
}
void SqueezeExciteChecks() {
  nn::ParameterStore p(17);
  auto               x = Pattern({2, 5, 5, 8}, 1, .4f);
  mobilenet::SqueezeExcite(p, "se", x, 8);
  p.set("se/expand/W", nn::zeros(p.at("se/expand/W").shape()));
  std::vector<float> bias(8);
  for (int i = 0; i < 8; ++i)
    bias[i] = -4 + i;
  p.set("se/expand/b", nn::from_vector(bias, {8}));
  auto out     = mobilenet::SqueezeExcite(p, "se", x, 8).output.to_vector<float>(),
       in      = x.to_vector<float>();
  double error = 0;
  for (std::size_t i = 0; i < out.size(); ++i)
    error =
        std::max(error, std::abs(out[i] - std::clamp(-4 + int(i % 8) + 3., 0., 6.) / 6 * in[i]));
  check::Near(error, 0, 1e-6, "squeeze-excitation channel scaling");
}
void CostChecks() {
  struct Case {
    bool large;
    long params, state, macs;
  };
  for (const auto& c :
       {Case{true, 5483032, 24400, 216589760}, Case{false, 2542856, 12112, 56510400}}) {
    nn::NoGradGuard              guard;
    nn::ParameterStore           p(17), state(17);
    mobilenet::MobileNetV3Config cfg;
    cfg.large = c.large;
    auto net  = mobilenet::MobileNetV3(p, state, nn::zeros({1, 224, 224, 3}), cfg);
    long macs = net.classifier_macs;
    for (const auto& l : net.layers)
      macs += l.Macs();
    check::Require(
        p.count() == c.params && state.count() == c.state && macs == c.macs,
        "MobileNet v3 parameters and MACs");
    std::cout << "MobileNet v3 large=" << c.large << " parameters=" << p.count() << " MACs=" << macs
              << std::endl;
  }
}
void GradientChecks(bool large) {
  nn::ParameterStore           p(17), state(17);
  mobilenet::MobileNetV3Config cfg;
  cfg.update_stats = false;
  cfg.large        = large;
  const auto x     = Pattern({2, 224, 224, 3}, 1, 0);
  {
    nn::NoGradGuard guard;
    mobilenet::MobileNetV3(p, state, x, cfg, true);
    mobile_checks::ShiftBetas(p, true);
  }
  auto net = mobilenet::MobileNetV3(p, state, x, cfg, true);
  check::Require(net.logits.shape() == nn::Shape({2, 1000}), "MobileNet v3 output dimensions");
  auto   norms   = mobile_checks::GradientNorms(p, nn::mean(nn::square(net.logits)));
  double maximum = *std::max_element(norms.begin(), norms.end()), minimum = INFINITY, removed = 0;
  int    count = 0;
  auto   names = p.names();
  for (std::size_t i = 0; i < norms.size(); ++i) {
    check::Require(std::isfinite(norms[i]), "finite MobileNet v3 derivative");
    if (mobile_checks::EndsWith(names[i], "/project/bn/beta")) {
      removed = std::max(removed, norms[i]);
      ++count;
    } else
      minimum = std::min(minimum, norms[i]);
  }
  check::Require(
      count == int(mobilenet::MobileNetV3Blocks(large).size()) && removed < 1e-6 * maximum,
      "vanishing projection BN shifts");
  check::Require(minimum > 1e-5 * maximum, "nonzero MobileNet v3 derivatives with shifted BN");
  std::cout << "gradient range " << minimum << " to " << maximum << " removed shifts " << removed
            << std::endl;
}

/// Fits one CIFAR MobileNet v3 size: fixed development split, Adam on
/// cross-entropy plus L2, validation-NLL checkpoint selection, one test pass.
/// The whole workflow is written out here so the R and Python examples can
/// follow it line by line over the same bound components.
void TrainCompactMobileNetV3(
    bool                                large,
    const nn::ImageDataset&             development,
    const nn::ImageDataset&             test,
    image_training::ImageTrainingConfig cfg,
    bool                                explicit_epochs) {
  using namespace image_training;
  cfg.name         = large ? "mobilenet_v3_large_cifar10" : "mobilenet_v3_small_cifar10";
  const auto model = [large](
                         nn::ParameterStore& p,
                         nn::ParameterStore& state,
                         const nn::Tensor&   image,
                         bool                training) {
    mobilenet::MobileNetV3Config options;
    options.large      = large;
    options.classes    = 10;
    options.small_stem = true;
    const auto net     = mobilenet::MobileNetV3(p, state, image, options, training);
    check::Require(p.count() == (large ? 4214842 : 1528106), "CIFAR MobileNet v3 parameter count");
    return net.logits;
  };

  // Quick mode bounds the data and batch; an explicit epoch budget is honored.
  if (!explicit_epochs)
    cfg.epochs = example_runtime::Limit(cfg.epochs, 1);
  cfg.batch = example_runtime::Limit(cfg.batch, 8);
  if (!example_runtime::Full())
    cfg.maximum_fit_rows = cfg.maximum_fit_rows ? std::min(cfg.maximum_fit_rows, 32) : 32;
  example_runtime::Describe();
  const std::int64_t development_rows = development.images.shape()[0];
  const std::int64_t test_size = example_runtime::Limit<std::int64_t>(test.images.shape()[0], 32);
  check::Require(
      development.images.dim() == 4 && test.images.dim() == 4 && cfg.validation_rows > 0 &&
          cfg.validation_rows < development_rows && cfg.epochs > 0 && cfg.batch > 0 &&
          cfg.classes > 1,
      "valid image training dimensions/config");
  for (int d = 1; d < 4; ++d)
    check::Require(
        development.images.shape()[d] == test.images.shape()[d],
        "training/test image dimensions match");
  const std::size_t available = static_cast<std::size_t>(development_rows - cfg.validation_rows);
  check::Require(
      cfg.maximum_fit_rows >= 0 && static_cast<std::size_t>(cfg.maximum_fit_rows) <= available,
      "valid optional fitting-row limit");
  const std::size_t fit_count =
      cfg.maximum_fit_rows ? static_cast<std::size_t>(cfg.maximum_fit_rows) : available;
  check::Require(fit_count >= 2, "at least two fitting rows for model initialization");

  // Split the development rows once; keep the validation split fixed when limiting the fitting
  // subset.
  std::vector<std::size_t> order(static_cast<std::size_t>(development_rows));
  std::iota(order.begin(), order.end(), 0);
  std::mt19937_64 split_rng(2026);
  std::shuffle(order.begin(), order.end(), split_rng);
  const std::vector<std::int64_t> fit(
      order.begin(), order.begin() + static_cast<std::ptrdiff_t>(fit_count));
  std::vector<std::int64_t> validation(
      order.begin() + static_cast<std::ptrdiff_t>(available), order.end());
  validation.resize(example_runtime::Limit<std::size_t>(validation.size(), 32));
  std::vector<std::int64_t> test_rows(static_cast<std::size_t>(test_size));
  std::iota(test_rows.begin(), test_rows.end(), 0);

  nn::ParameterStore params(cfg.seed), state(cfg.seed);
  {
    nn::NoGradGuard
        guard; // create every parameter (and batch-norm state) before building the optimizer
    model(params, state, ImageBatch(development, {fit[0], fit[1]}).first, false);
  }
  nn::OptimizerConfig config;
  config.kind = nn::OptimizerKind::Adam;
  nn::Optimizer optimizer(params, cfg.learning_rate, config);

  // Inference-mode cross-entropy, accuracy and confusion over `rows`, in batches of 256.
  auto evaluate = [&](const nn::ImageDataset& dataset, const std::vector<std::int64_t>& rows) {
    nn::NoGradGuard     guard;
    nn::WeightedMean    loss, accuracy;
    nn::ConfusionMatrix confusion(static_cast<std::size_t>(cfg.classes));
    for (std::size_t start = 0; start < rows.size(); start += 256) {
      const std::size_t               end = std::min(rows.size(), start + 256);
      const std::vector<std::int64_t> ids(
          rows.begin() + static_cast<std::ptrdiff_t>(start),
          rows.begin() + static_cast<std::ptrdiff_t>(end));
      const auto [x, y]           = ImageBatch(dataset, ids);
      const nn::Tensor logits     = model(params, state, x, false);
      const nn::Tensor prediction = nn::argmax(logits, 1);
      loss.add(nn::sparse_softmax_cross_entropy(logits, y).item(), end - start);
      accuracy.add(nn::accuracy(prediction, y), end - start);
      confusion.add(y, prediction);
    }
    return std::make_tuple(loss.value(), accuracy.value(), confusion);
  };

  const auto directory   = paths::OutputDir("models/" + cfg.name);
  const auto params_path = (directory / "best_params.npz").string(),
             state_path  = (directory / "best_state.npz").string();
  std::ofstream history(directory / "history.tsv");
  check::Require(bool(history), "open image training history");
  history << "epoch\tvalidation_loss\tvalidation_accuracy\n";
  std::cout << "== " << cfg.name << ": " << fit_count << " fit / " << validation.size()
            << " validation / " << test_size << " test\n  parameters=" << params.count()
            << "; augmentation=" << cfg.augment << "; batch=" << cfg.batch
            << "; lr=" << cfg.learning_rate << "; threads=" << nn::get_num_threads() << std::endl;
  nn::BatchOrder      batches(fit_count, static_cast<std::size_t>(cfg.batch), 29);
  nn::Generator       augmentation(151);
  double              best       = std::numeric_limits<double>::infinity();
  int                 best_epoch = -1;
  std::vector<double> epoch_seconds;
  for (int epoch = 1; epoch <= cfg.epochs; ++epoch) {
    const auto epoch_start = std::chrono::steady_clock::now();
    batches.begin_epoch();
    std::size_t used = 0;
    for (;;) {
      auto rows = batches.next();
      if (rows.empty())
        break;
      used += rows.size();
      for (auto& row : rows)
        row = fit[static_cast<std::size_t>(row)];
      // Augmented batch; training-mode forward; cross-entropy plus 0.5 * l2 * sum(W^2); one Adam
      // step.
      const auto [x, y] = ImageBatch(development, rows, cfg.augment ? &augmentation : nullptr);
      const nn::Tensor logits = model(params, state, x, true);
      const nn::Tensor objective =
          WithL2(TrainingCrossEntropy(logits, y, cfg.classes, cfg.label_smoothing), params, cfg.l2);
      optimizer.minimize(objective);
    }
    check::Require(used == fit_count, "each fit image used once per epoch");
    const double train_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - epoch_start).count();
    const auto [loss, accuracy, confusion] = evaluate(development, validation);
    if (loss < best) {
      best       = loss;
      best_epoch = epoch;
      params.save(params_path);
      if (state.size())
        state.save(state_path);
    }
    history << epoch << '\t' << loss << '\t' << accuracy << '\n';
    history.flush();
    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - epoch_start).count();
    epoch_seconds.push_back(seconds);
    std::cout << "  epoch " << epoch << ": validation NLL=" << loss << ", accuracy=" << accuracy
              << " (" << std::setprecision(3) << train_seconds << " s train, " << seconds
              << " s total)" << std::setprecision(6) << std::endl;
  }
  params.load(params_path);
  if (state.size())
    state.load(state_path);
  check::Near(
      std::get<0>(evaluate(development, validation)),
      best,
      1e-6,
      "best image checkpoint validation loss");
  const auto [test_loss, test_accuracy, confusion] = evaluate(test, test_rows);
  std::cout << "  selected epoch=" << best_epoch << "; test NLL=" << test_loss
            << "; accuracy=" << test_accuracy << "\n  confusion (truth rows):\n";
  std::uint64_t total = 0, correct = 0;
  const auto&   cells = confusion.cells();
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
  check::Near(test_accuracy, double(correct) / total, 1e-7, "confusion agrees with accuracy");
  check::TrainingQuality(
      test_accuracy >= cfg.minimum_test_accuracy,
      "image classifier demonstration accuracy threshold");
  check::Require(bool(history), "write image training history");
  double train_total = 0;
  for (double seconds : epoch_seconds)
    train_total += seconds;
  std::printf("  training time %.1f s (%.2f s/epoch)\n", train_total, train_total / cfg.epochs);
  std::cout << "  best checkpoint and history: " << directory.string() << "\nAll " << cfg.name
            << " checks passed.\n";
}

} // namespace
int main(int argc, char** argv) {
  try {
    nn::set_num_threads(2);
    ActivationChecks();
    SqueezeExciteChecks();
    CostChecks();
    GradientChecks(true);
    GradientChecks(false);
    std::filesystem::path               directory = paths::Dataset("cifar10");
    image_training::ImageTrainingConfig cfg;
    cfg.maximum_fit_rows      = 12000;
    cfg.epochs                = 12;
    cfg.minimum_test_accuracy = .4;
    bool explicit_epochs      = false;
    for (int i = 1; i < argc; ++i) {
      const std::string argument = argv[i];
      const auto        equal    = argument.find('=');
      if (equal == std::string::npos)
        throw std::invalid_argument("expected key=value option");
      explicit_epochs = explicit_epochs || argument.substr(0, equal) == "epochs";
      if (!image_training::ParseCifarOption(
              argument.substr(0, equal), argument.substr(equal + 1), directory, cfg))
        throw std::invalid_argument("unknown option: " + argument);
    }
    const auto [development, test] = image_training::LoadCifar10(directory);
    for (bool large : {true, false})
      TrainCompactMobileNetV3(large, development, test, cfg, explicit_epochs);
    std::cout << "All MobileNet v3 checks passed.\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
