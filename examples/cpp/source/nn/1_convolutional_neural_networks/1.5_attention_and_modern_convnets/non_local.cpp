// CMD: just run-nn non_local
// Non-local neural networks: self-attention over all positions of a feature map.
//
// Port of tflow's non_local. Check all four pairwise functions (embedded
// Gaussian, Gaussian, dot product, concatenation), with and without 2x2
// subsampling of phi and g, against a host computation of
// y_i = sum_j f(x_i,x_j) g(x_j) / C(x), z = BN(W_z y) + x. Check that a
// freshly built block is the identity (BN scale zero) and that, once active,
// one output position depends on every input position for the smooth
// pairwise functions without subsampling (a 3x3 convolution reaches nine;
// max-pooling and the concatenation ReLU pass derivatives to fewer). Count
// ResNet-50 with the paper's five blocks (two in res3, three in res4):
// 32,915,496 learned parameters. Check every derivative in training mode
// (moving averages not updated) at 224 pixels: all are clearly nonzero except
// the biases of phi, g and W_z, which only shift softmax rows or channels
// that training-mode BN normalizes.
// https://arxiv.org/abs/1711.07971
// https://github.com/facebookresearch/video-nonlocal-net
//
// Train the compact ResNet-50 of resnet_bottleneck with five embedded-Gaussian
// blocks (at 16x16 and 8x8 resolution, subsampled) on CIFAR-10, for
// comparison with the plain and channel-attention versions. Use 12k fit / 5k
// validation, twelve epochs, crop/flip, Adam(.001), kernel L2(.0001) and the
// best-validation checkpoint before evaluating all 10k test images. The
// blocks attach through AttentionResNetConfig::after_block (tflow's
// cfg.non_local). Weights come from nn's generator, so trained numbers match
// TF only statistically. CIFAR-10 is read from the tflow dataset cache
// (common/paths.hpp) or data=.
// Options: data=PATH epochs=12 batch=128 lr=.001 fit=12000 augment=0|1 threads=N seed=17.

#include "common/non_local.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>

#include "common/attention_cnn.hpp"
#include "common/checks.hpp"
#include "common/image_training.hpp"
#include "common/mobile_checks.hpp"
#include "common/residual_validation.hpp"

namespace nn = datamunge::nn;
using namespace non_local_checks;

int main(int argc, char** argv) {
  try {
    for (int i = 1; i < argc; ++i) // threads= also applies to the checks
      if (std::string(argv[i]).rfind("threads=", 0) == 0)
        nn::set_num_threads(std::stoi(std::string(argv[i]).substr(8)));
    BlockChecks();
    FullChecks();
    image_training::ImageTrainingConfig cfg;
    cfg.name                                    = "nonlocal_resnet50_compact_cifar10";
    cfg.maximum_fit_rows                        = 12000;
    cfg.epochs                                  = 12;
    cfg.minimum_test_accuracy                   = .4;
    const image_training::LogitsBuilder builder = [](nn::ParameterStore& p,
                                                     nn::ParameterStore& state,
                                                     const nn::Tensor&   image,
                                                     bool                training) {
      attention_cnn::AttentionResNetConfig model;
      model.classes       = 10;
      model.width_divisor = 4;
      model.small_stem    = true;
      model.stride_on_3x3 = false;
      model.after_block   = NonLocalHook(model.momentum, true);
      return attention_cnn::AttentionResNet(p, state, image, model, training).logits;
    };
    {
      using namespace image_training;
      std::filesystem::path data_directory  = paths::Dataset("cifar10");
      bool                  explicit_epochs = false;
      for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        const auto        equal    = argument.find('=');
        if (equal == std::string::npos)
          throw std::invalid_argument("expected key=value option");
        explicit_epochs = explicit_epochs || argument.substr(0, equal) == "epochs";
        if (!ParseCifarOption(
                argument.substr(0, equal), argument.substr(equal + 1), data_directory, cfg))
          throw std::invalid_argument("unknown option: " + argument);
      }
      check::Require(cfg.classes == 10, "CIFAR-10 requires ten output classes");
      const auto [development, test] = LoadCifar10(data_directory);
      const auto& model              = builder;
      // Quick mode bounds the data and batch; an explicit epoch budget is honored.
      if (!explicit_epochs)
        cfg.epochs = example_runtime::Limit(cfg.epochs, 1);
      cfg.batch = example_runtime::Limit(cfg.batch, 8);
      if (!example_runtime::Full())
        cfg.maximum_fit_rows = cfg.maximum_fit_rows ? std::min(cfg.maximum_fit_rows, 32) : 32;
      example_runtime::Describe();
      const std::int64_t development_rows = development.images.shape()[0];
      const std::int64_t test_size =
          example_runtime::Limit<std::int64_t>(test.images.shape()[0], 32);
      check::Require(
          development.images.dim() == 4 && test.images.dim() == 4 && cfg.validation_rows > 0 &&
              cfg.validation_rows < development_rows && cfg.epochs > 0 && cfg.batch > 0 &&
              cfg.classes > 1,
          "valid image training dimensions/config");
      for (int d = 1; d < 4; ++d)
        check::Require(
            development.images.shape()[d] == test.images.shape()[d],
            "training/test image dimensions match");
      const std::size_t available =
          static_cast<std::size_t>(development_rows - cfg.validation_rows);
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
                << "; lr=" << cfg.learning_rate << "; threads=" << nn::get_num_threads()
                << std::endl;
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
          // Augmented batch; training-mode forward; cross-entropy plus 0.5 * l2 * sum(W^2); one
          // Adam step.
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
    }
    std::cout << "All non-local checks passed.\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
