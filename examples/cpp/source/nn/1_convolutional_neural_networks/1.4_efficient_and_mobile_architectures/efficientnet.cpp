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

#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>

#include "common/checks.hpp"
#include "common/cnn_validation.hpp"
#include "common/image_training.hpp"
#include "common/mbconv.hpp"
#include "common/mobile_checks.hpp"

namespace nn = datamunge::nn;
using namespace efficientnet_checks;

int main(int argc, char** argv) {
  try {
    using namespace image_training;
    for (int i = 1; i < argc; ++i) // threads= also applies to the checks
      if (std::string(argv[i]).rfind("threads=", 0) == 0)
        nn::set_num_threads(std::stoi(std::string(argv[i]).substr(8)));
    CostChecks();
    DropConnectChecks();
    GradientChecks();
    std::filesystem::path dir = paths::Dataset("cifar10");
    ImageTrainingConfig   cfg;
    cfg.name                  = "efficientnet_b0_cifar10";
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
      if (!ParseCifarOption(argument.substr(0, equal), argument.substr(equal + 1), dir, cfg))
        throw std::invalid_argument("unknown option: " + argument);
    }
    const auto [development, test] = LoadCifar10(dir);
    // CIFAR EfficientNet-B0: stride-one stem and ten classes; the parameter count is checked on
    // every pass.
    const auto model = [](nn::ParameterStore& p,
                          nn::ParameterStore& state,
                          const nn::Tensor&   image,
                          bool                training) {
      mbconv::EfficientNetConfig options;
      options.classes    = 10;
      options.small_stem = true;
      const auto logits  = mbconv::EfficientNet(p, state, image, options, training).logits;
      check::Require(p.count() == 4020358, "CIFAR EfficientNet-B0 parameter count");
      return logits;
    };

    // The training workflow is written out below so the R and Python examples can
    // follow it line by line over the same bound components. Quick mode bounds the
    // data and batch; an explicit epoch budget is honored.
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
        const nn::Tensor logits    = model(params, state, x, true);
        const nn::Tensor objective = WithL2(
            TrainingCrossEntropy(logits, y, cfg.classes, cfg.label_smoothing), params, cfg.l2);
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
    std::cout << "All EfficientNet checks passed.\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
