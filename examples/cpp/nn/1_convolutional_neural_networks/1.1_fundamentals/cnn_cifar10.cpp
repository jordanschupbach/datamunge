// CMD: just run-nn cnn_cifar10
// A small CIFAR-10 CNN trained with independent random crops and flips.
//
// Port of tflow's cnn_cifar10. Preparation (in ~/projects/tflow_examples):
// python3 scripts/fetch_datasets.py cifar10 --split both.
// Architecture: Conv(32,3)-ReLU-pool -> Conv(64,3)-ReLU-pool -> Dense(128)-
// ReLU -> 10 logits. Train on 45k original images with four-pixel zero padding,
// random 32x32 crops and horizontal flips; normalize RGB pixels to [-1,1].
// Select by NLL on 5k untouched validation images, restore that checkpoint,
// then evaluate all 10k original test images once. Adam/L2; no pretrained data.
// Options: data=PATH epochs=10 batch=128 lr=.001 augment=0|1 threads=N seed=17.
// The default run requires 55% test accuracy as a training sanity check.
// Checkpoint/history: output/nn/models/cnn_cifar10/ (or cnn_cifar10_no_augmentation).

#include <iostream>
#include <random>

#include "common/image_training.hpp"

namespace nn = datamunge::nn;

void AugmentationChecks() {
  nn::ImageDataset fixture;
  fixture.images = nn::ones({2, 32, 32, 3});
  fixture.labels = nn::from_vector(std::vector<std::int64_t>{2, 7}, {2});
  nn::Generator first(151), repeat(151);
  const auto    plain      = image_training::ImageBatch(fixture, {1, 0});
  const auto    augmented  = image_training::ImageBatch(fixture, {1, 0}, &first);
  const auto    reproduced = image_training::ImageBatch(fixture, {1, 0}, &repeat);
  const auto    p = plain.first.to_vector<float>(), a = augmented.first.to_vector<float>(),
             r = reproduced.first.to_vector<float>();
  std::size_t padded = 0, interior = 0;
  for (std::size_t i = 0; i < p.size(); ++i) {
    check::Near(p[i], 1, 0, "evaluation only normalizes pixels");
    check::Near(a[i], r[i], 0, "seeded per-image augmentation");
    check::Require(a[i] == -1 || a[i] == 1, "crop preserves white image or zero padding");
    padded += a[i] == -1;
    interior += a[i] == 1;
  }
  check::Require(padded > 0 && interior > 0, "random crop includes padded and original pixels");
  const auto labels = augmented.second.to_vector<std::int64_t>();
  check::Require(labels[0] == 7 && labels[1] == 2, "augmentation preserves labels and row order");
  std::cout << "  augmentation checks: reproducible crops, normalized padding and paired labels passed\n";
}

int main(int argc, char** argv) {
  try {
    using namespace image_training;
    std::filesystem::path dir = paths::Dataset("cifar10");
    ImageTrainingConfig   cfg;
    cfg.name             = "cnn_cifar10";
    bool explicit_epochs = false;
    for (int i = 1; i < argc; ++i) {
      const std::string arg   = argv[i];
      const auto        equal = arg.find('=');
      if (equal == std::string::npos) throw std::invalid_argument("expected key=value argument");
      explicit_epochs = explicit_epochs || arg.substr(0, equal) == "epochs";
      if (!ParseCifarOption(arg.substr(0, equal), arg.substr(equal + 1), dir, cfg))
        throw std::invalid_argument("unknown argument: " + arg);
    }
    if (!cfg.augment) cfg.name += "_no_augmentation";
    AugmentationChecks();
    const auto [development, test] = LoadCifar10(dir);
    auto model = [](nn::ParameterStore& p, nn::ParameterStore&, const nn::Tensor& image, bool) {
      nn::ConvLayerOptions options;
      options.activation = nn::Activation::Relu;
      auto first  = nn::max_pool2d(nn::conv2d(p, "conv1", image, 32, 3, options), 2, 2);
      auto second = nn::max_pool2d(nn::conv2d(p, "conv2", first, 64, 3, options), 2, 2);
      auto hidden = nn::dense(p, "dense", nn::reshape(second, {-1, 8 * 8 * 64}), 128, nn::Activation::Relu);
      return nn::dense(p, "output", hidden, 10);
    };

    // Quick mode bounds the data and batch; an explicit epoch budget is honored.
    if (!explicit_epochs) cfg.epochs = example_runtime::Limit(cfg.epochs, 1);
    cfg.batch = example_runtime::Limit(cfg.batch, 8);
    if (!example_runtime::Full()) cfg.maximum_fit_rows = cfg.maximum_fit_rows ? std::min(cfg.maximum_fit_rows, 32) : 32;
    example_runtime::Describe();
    const std::int64_t development_rows = development.images.shape()[0];
    const std::int64_t test_size        = example_runtime::Limit<std::int64_t>(test.images.shape()[0], 32);
    check::Require(development.images.dim() == 4 && test.images.dim() == 4 && cfg.validation_rows > 0 &&
                       cfg.validation_rows < development_rows && cfg.epochs > 0 && cfg.batch > 0 && cfg.classes > 1,
                   "valid image training dimensions/config");
    for (int d = 1; d < 4; ++d)
      check::Require(development.images.shape()[d] == test.images.shape()[d], "training/test image dimensions match");
    const std::size_t available = static_cast<std::size_t>(development_rows - cfg.validation_rows);
    check::Require(cfg.maximum_fit_rows >= 0 && static_cast<std::size_t>(cfg.maximum_fit_rows) <= available,
                   "valid optional fitting-row limit");
    const std::size_t fit_count = cfg.maximum_fit_rows ? static_cast<std::size_t>(cfg.maximum_fit_rows) : available;

    // Split the development rows once; keep the validation split fixed when limiting the fitting subset.
    std::vector<std::size_t> order(static_cast<std::size_t>(development_rows));
    std::iota(order.begin(), order.end(), 0);
    std::mt19937_64 split_rng(2026);
    std::shuffle(order.begin(), order.end(), split_rng);
    const std::vector<std::int64_t> fit(order.begin(), order.begin() + static_cast<std::ptrdiff_t>(fit_count));
    std::vector<std::int64_t>       validation(order.begin() + static_cast<std::ptrdiff_t>(available), order.end());
    validation.resize(example_runtime::Limit<std::size_t>(validation.size(), 32));
    std::vector<std::int64_t> test_rows(static_cast<std::size_t>(test_size));
    std::iota(test_rows.begin(), test_rows.end(), 0);

    nn::ParameterStore params(cfg.seed), state(cfg.seed);
    {
      nn::NoGradGuard guard;  // create every parameter before building the optimizer
      model(params, state, ImageBatch(development, {fit[0], fit[1]}).first, false);
    }
    nn::OptimizerConfig config;
    config.kind = nn::OptimizerKind::Adam;
    nn::Optimizer optimizer(params, cfg.learning_rate, config);

    const auto directory   = paths::OutputDir("models/" + cfg.name);
    const auto params_path = (directory / "best_params.npz").string();
    std::ofstream history(directory / "history.tsv");
    check::Require(bool(history), "open image training history");
    history << "epoch\tvalidation_loss\tvalidation_accuracy\n";
    std::cout << "== " << cfg.name << ": " << fit_count << " fit / " << validation.size() << " validation / " << test_size
              << " test\n  parameters=" << params.count() << "; augmentation=" << cfg.augment << "; batch=" << cfg.batch
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
        if (rows.empty()) break;
        used += rows.size();
        for (auto& row : rows) row = fit[static_cast<std::size_t>(row)];
        // Augmented batch, cross-entropy plus 0.5 * l2 * sum(W^2), one Adam step.
        const auto [x, y] = ImageBatch(development, rows, cfg.augment ? &augmentation : nullptr);
        optimizer.minimize(TrainingObjective(params, state, model, x, y, cfg));
      }
      check::Require(used == fit_count, "each fit image used once per epoch");
      const double train_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - epoch_start).count();
      const auto   result        = EvaluateImages(params, state, model, development, validation, cfg.classes);
      const double loss = result.loss.value(), accuracy = result.accuracy.value();
      if (loss < best) {
        best       = loss;
        best_epoch = epoch;
        params.save(params_path);
      }
      history << epoch << '\t' << loss << '\t' << accuracy << '\n';
      history.flush();
      const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - epoch_start).count();
      epoch_seconds.push_back(seconds);
      std::cout << "  epoch " << epoch << ": validation NLL=" << loss << ", accuracy=" << accuracy << " ("
                << std::setprecision(3) << train_seconds << " s train, " << seconds << " s total)" << std::setprecision(6)
                << std::endl;
    }
    params.load(params_path);
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
    check::TrainingQuality(result.accuracy.value() >= cfg.minimum_test_accuracy,
                           "image classifier demonstration accuracy threshold");
    check::Require(bool(history), "write image training history");
    double train_total = 0;
    for (double seconds : epoch_seconds) train_total += seconds;
    std::printf("  training time %.1f s (%.2f s/epoch)\n", train_total, train_total / cfg.epochs);
    std::cout << "  best checkpoint and history: " << directory.string() << "\nAll " << cfg.name << " checks passed.\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
