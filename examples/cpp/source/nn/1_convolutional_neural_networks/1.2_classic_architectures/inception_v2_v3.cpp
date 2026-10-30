// CMD: just run-nn inception_v2_v3
// Inception v2 and v3: batch normalization, factorization and label smoothing.
//
// Port of tflow's inception_v2_v3. Implement full TF-Slim reference layouts:
// v2 at 224 pixels, v3 at 299 pixels, with 1000 classes. Check every stage
// shape, independently counted learned parameters and running statistics, and
// finite nonzero parameter derivatives. v2 includes the separable 7x7 stem and
// ten ordinary/reduction modules. v3 includes asymmetric 1x7/7x1 convolutions,
// split 1x3/3x1 branches and an auxiliary head. Reference definitions and
// initialization differences are documented in common/inception23.hpp.
//
// Demonstrate linear spatial factorization with an outer-product 7x7 kernel:
// two rectangular convolutions match the composed convolution and its input
// and factor gradients. This equivalence needs a factorizable kernel and no
// intervening nonlinearities; real Inception modules include BN/ReLU. Verify
// beta-only BN against host moments, running-statistic updates, and smoothed
// training cross entropy separately from unsmoothed evaluation NLL.
//
// Train compact CIFAR-10 versions, retaining every block: quarter widths,
// 64-pixel v2 / 107-pixel v3 inputs, 12k fit / 5k validation rows, twelve
// epochs, Adam(.001), crop/flip augmentation and .8 dropout retention. Use
// momentum .9 for BN on these short runs; canonical configuration uses .9997.
// v3 adds .1 label smoothing and .4-weight auxiliary cross entropy. Validation
// selection and the final 10k-row test use only unsmoothed primary NLL.
// These demonstrations are not full-model ImageNet benchmark results.
// CIFAR-10 is read from paths::Dataset("cifar10"). Batch-norm moving averages
// update eagerly during training passes (tflow runs collected update ops), so
// "inference requests no updates" becomes "an inference pass leaves the state
// store unchanged".
// Options: data=PATH epochs=12 batch=128 lr=.001 fit=12000 augment=0|1 threads=N seed=17.

#include <cmath>
#include <iostream>

#include "common/architecture_checks.hpp"
#include "common/image_training.hpp"
#include "common/inception23.hpp"

namespace nn = datamunge::nn;

void FactorizationChecks() {
  nn::Tensor x = architecture::SinePixels({1, 11, 11, 1}, 1, .31);
  nn::Tensor vertical =
      nn::from_vector(std::vector<float>{.1f, .2f, -.1f, .3f, .1f, -.2f, .2f}, {7});
  nn::Tensor horizontal =
      nn::from_vector(std::vector<float>{.3f, -.1f, .2f, .1f, .2f, -.3f, .1f}, {7});
  for (auto* t : {&x, &vertical, &horizontal})
    t->requires_grad_(true);
  const auto v = nn::reshape(vertical, {7, 1, 1, 1}), h = nn::reshape(horizontal, {1, 7, 1, 1});
  const auto valid    = nn::conv_options(1, nn::Padding::Valid);
  const auto factored = nn::conv2d(nn::conv2d(x, h, valid), v, valid);
  const auto composed = nn::conv2d(x, v * h, valid);
  nn::GradOptions keep;
  keep.retain_graph  = true; // both graphs share the reshaped factors
  const auto a       = nn::grad(nn::mean(nn::square(factored)), {x, vertical, horizontal}, keep);
  const auto b       = nn::grad(nn::mean(nn::square(composed)), {x, vertical, horizontal});
  double     error   = 0;
  auto       compare = [&](const nn::Tensor& p, const nn::Tensor& q) {
    const auto pv = p.to_vector<float>(), qv = q.to_vector<float>();
    for (std::size_t i = 0; i < pv.size(); ++i)
      error = std::max(error, double(std::abs(pv[i] - qv[i])));
  };
  compare(factored, composed);
  for (std::size_t i = 0; i < a.size(); ++i)
    compare(a[i], b[i]);
  check::Near(error, 0, 2e-6, "factorized output and input/factor gradients");
  std::cout << "  1x7 then 7x1 vs outer-product 7x7: output/gradient max error=" << error
            << "; 14 vs 49 coefficients per channel pair at equal intermediate width\n";
}

void NormalizationChecks() {
  nn::ParameterStore              p(5), state(5);
  normalized_conv::ConvNormConfig cfg;
  cfg.training     = true;
  cfg.momentum     = .9;
  const auto input = architecture::SinePixels({2, 4, 4, 1}, 1, .27);
  const auto output =
      normalized_conv::NormalizedConv(p, state, "rect", input, 2, 1, 3, cfg, 1, "VALID", false)
          .to_vector<float>();
  const auto convolution =
      nn::conv2d(input, p.at("rect/W"), nn::conv_options(1, nn::Padding::Valid)).to_vector<float>();
  check::Require(p.count() == 8 && state.count() == 4, "beta-only BN parameters and running state");
  const auto mean_state = state.at("rect/bn/moving_mean").to_vector<float>(),
             var_state  = state.at("rect/bn/moving_var").to_vector<float>();
  const int rows        = static_cast<int>(output.size()) / 2;
  for (int c = 0; c < 2; ++c) {
    double mean = 0, variance = 0;
    for (int i = 0; i < rows; ++i)
      mean += convolution[static_cast<std::size_t>(2 * i + c)] / rows;
    for (int i = 0; i < rows; ++i)
      variance += std::pow(convolution[static_cast<std::size_t>(2 * i + c)] - mean, 2) / rows;
    for (int i = 0; i < rows; ++i)
      check::Near(
          output[static_cast<std::size_t>(2 * i + c)],
          (convolution[static_cast<std::size_t>(2 * i + c)] - mean) / std::sqrt(variance + .001),
          2e-6,
          "fixed-gamma BN output");
    check::Near(mean_state[static_cast<std::size_t>(c)], .1 * mean, 2e-6, "one BN mean update");
    check::Near(
        var_state[static_cast<std::size_t>(c)],
        .9 + .1 * variance * rows / (rows - 1),
        2e-6,
        "one unbiased-variance update");
  }
  std::cout << "  rectangular convolution and beta-only BN: host output/moments and one EMA update "
               "passed\n";
}

void SmoothingChecks() {
  nn::ParameterStore p(17), state(17);
  p.set("W", nn::from_vector(std::vector<float>{10, 0, 0}, {1, 3}));
  const image_training::LogitsBuilder model =
      [](nn::ParameterStore& q, nn::ParameterStore&, const nn::Tensor& x, bool) {
        return nn::matmul(x, q.at("W"));
      };
  image_training::ImageTrainingConfig cfg;
  cfg.classes         = 3;
  cfg.l2              = 0;
  cfg.label_smoothing = .1;
  const auto x = nn::ones({1, 1}), y = nn::from_vector(std::vector<std::int64_t>{0}, {1});
  double     evaluation = 0, training = 0;
  {
    nn::NoGradGuard guard;
    evaluation = nn::sparse_softmax_cross_entropy(model(p, state, x, false), y).item();
    training   = image_training::TrainingObjective(p, state, model, x, y, cfg).item();
  }
  const double nll = std::log1p(2 * std::exp(-10.));
  check::Near(evaluation, nll, 2e-7, "evaluation retains unsmoothed NLL");
  check::Near(training, nll + 2. / 3, 2e-6, "uniform label smoothing cross entropy");
  nn::OptimizerConfig config;
  config.kind = nn::OptimizerKind::Adam;
  nn::Optimizer optimizer(p, .01, config);
  optimizer.minimize(image_training::TrainingObjective(p, state, model, x, y, cfg));
  check::Require(
      p.at("W").to_vector<float>()[0] < 10,
      "smoothing reduces an overconfident correct-class score");
  std::cout << "  .1 label smoothing: evaluation NLL=" << nll << "; training CE=" << nll + 2. / 3
            << "; correct-class score decreases\n";
}

void FullArchitectureChecks(int version) {
  nn::ParameterStore             p(17), state(17);
  inception23::Inception23Config cfg;
  cfg.version       = version;
  cfg.image_size    = version == 2 ? 224 : 299;
  const auto pixels = architecture::SinePixels({1, cfg.image_size, cfg.image_size, 3}, .1, .13);
  const auto net    = inception23::Inception23(p, state, pixels, cfg, false, true);
  const std::int64_t aux_count  = architecture::CountWithPrefix(p, "aux/"),
                     main_count = p.count() - aux_count;
  check::Require(
      main_count == (version == 2 ? 11178336 : 23817352) &&
          aux_count == (version == 2 ? 0 : 3325800),
      "reference learned parameter counts");
  check::Require(state.count() == (version == 2 ? 19776 : 36224), "reference running-state count");
  const auto                             initial_state = state.flatten();
  const std::vector<std::pair<int, int>> expected =
      version == 2
          ? std::vector<std::pair<
                int,
                int>>{{112, 64}, {56, 64}, {56, 64}, {56, 192}, {28, 192}, {28, 256}, {28, 320}, {14, 576}, {14, 576}, {14, 576}, {14, 576}, {14, 576}, {7, 1024}, {7, 1024}, {7, 1024}}
          : std::vector<std::pair<int, int>>{
                {149, 32},
                {147, 32},
                {147, 64},
                {73, 64},
                {73, 80},
                {71, 192},
                {35, 192},
                {35, 256},
                {35, 288},
                {35, 288},
                {17, 768},
                {17, 768},
                {17, 768},
                {17, 768},
                {17, 768},
                {8, 1280},
                {8, 2048},
                {8, 2048}};
  check::Require(net.stages.size() == expected.size(), "reference stage count");
  for (std::size_t i = 0; i < expected.size(); ++i) {
    const auto [spatial, channels] = expected[i];
    architecture::RequireShape(
        net.stages[i].second,
        {1, spatial, spatial, channels},
        "reference Inception stage shape: " + net.stages[i].first);
  }
  nn::Tensor objective = nn::mean(nn::square(net.logits));
  for (const auto& aux : net.auxiliary)
    objective = objective + .4 * nn::mean(nn::square(aux));
  const auto [minimum, maximum] = architecture::GradientNormRange(p, objective, "Inception");
  check::Require(
      state.flatten() == initial_state, "inference does not update BN running statistics");
  std::cout << "  full Inception v" << version << ": main=" << main_count << ", aux=" << aux_count
            << ", running state=" << state.count() << "; " << expected.size()
            << " stage shapes; gradient norms [" << minimum << ',' << maximum << "]" << std::endl;
}

inception23::Inception23Config CompactConfig(int version) {
  inception23::Inception23Config cfg;
  cfg.version       = version;
  cfg.image_size    = version == 2 ? 64 : 107;
  cfg.classes       = 10;
  cfg.width_divisor = 4;
  cfg.momentum      = .9;
  return cfg;
}

int main(int argc, char** argv) {
  try {
    FactorizationChecks();
    NormalizationChecks();
    SmoothingChecks();
    FullArchitectureChecks(2);
    FullArchitectureChecks(3);
    for (int version : {2, 3}) {
      const auto                          cfg_model = CompactConfig(version);
      const image_training::LogitsBuilder model     = [cfg_model](
                                                      nn::ParameterStore& p,
                                                      nn::ParameterStore& state,
                                                      const nn::Tensor&   x,
                                                      bool                training) {
        const auto image =
            nn::resize_bilinear(x, {cfg_model.image_size, cfg_model.image_size}, false, true);
        return inception23::Inception23(p, state, image, cfg_model, training).logits;
      };
      image_training::TrainingBuilder training;
      if (version == 3)
        training = [cfg_model](
                       nn::ParameterStore& p, nn::ParameterStore& state, const nn::Tensor& x) {
          const auto image =
              nn::resize_bilinear(x, {cfg_model.image_size, cfg_model.image_size}, false, true);
          const auto net = inception23::Inception23(p, state, image, cfg_model, true, true);
          return image_training::TrainingNetwork{net.logits, {{net.auxiliary[0], .4}}};
        };
      image_training::ImageTrainingConfig cfg;
      cfg.name                  = "inception_v" + std::to_string(version) + "_compact_cifar10";
      cfg.maximum_fit_rows      = 12000;
      cfg.epochs                = 12;
      cfg.minimum_test_accuracy = .35;
      cfg.label_smoothing       = version == 3 ? .1 : 0.;
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
            cfg.maximum_fit_rows >= 0 &&
                static_cast<std::size_t>(cfg.maximum_fit_rows) <= available,
            "valid optional fitting-row limit");
        const std::size_t fit_count =
            cfg.maximum_fit_rows ? static_cast<std::size_t>(cfg.maximum_fit_rows) : available;
        check::Require(fit_count >= 2, "at least two fitting rows for model initialization");

        // Split the development rows once; keep the validation split fixed when limiting the
        // fitting subset.
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
          const auto x = ImageBatch(development, {fit[0], fit[1]}).first;
          model(params, state, x, false);
          if (training) {
            nn::ParameterStore scratch(cfg.seed);
            training(params, scratch, x);
          }
        }
        nn::OptimizerConfig config;
        config.kind = nn::OptimizerKind::Adam;
        nn::Optimizer optimizer(params, cfg.learning_rate, config);

        // Inference-mode cross-entropy, accuracy and confusion over `rows`, in batches of 256.
        auto evaluate = [&](const nn::ImageDataset&          dataset,
                            const std::vector<std::int64_t>& rows) {
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
            const auto [x, y] =
                ImageBatch(development, rows, cfg.augment ? &augmentation : nullptr);
            const auto network = training ? training(params, state, x)
                                          : TrainingNetwork{model(params, state, x, true)};
            nn::Tensor objective =
                TrainingCrossEntropy(network.logits, y, cfg.classes, cfg.label_smoothing);
            for (const auto& [logits, weight] : network.auxiliary) {
              check::Require(
                  std::isfinite(weight) && weight > 0, "positive finite auxiliary loss weight");
              objective = objective + weight * TrainingCrossEntropy(
                                                   logits, y, cfg.classes, cfg.label_smoothing);
            }
            for (const auto& penalty : network.penalties)
              objective = objective + penalty;
            objective = WithL2(objective, params, cfg.l2);
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
          std::cout << "  epoch " << epoch << ": validation NLL=" << loss
                    << ", accuracy=" << accuracy << " (" << std::setprecision(3) << train_seconds
                    << " s train, " << seconds << " s total)" << std::setprecision(6) << std::endl;
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
        std::printf(
            "  training time %.1f s (%.2f s/epoch)\n", train_total, train_total / cfg.epochs);
        std::cout << "  best checkpoint and history: " << directory.string() << "\nAll " << cfg.name
                  << " checks passed.\n";
      }
    }
    std::cout << "All Inception v2/v3 checks passed.\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
