// CMD: just run-nn inception_v4_resnet
// Inception v4 and Inception-ResNet-v2: full depth and scaled residual branches.
//
// Port of tflow's inception_v4_resnet. Implement the complete TF-Slim layouts:
// v4's 4/7/3 A/B/C blocks and the residual model's 10/20/10 blocks.
// Inception-ResNet uses linear projections without BN; scales are
// .17/.10/.20, except the last C block is unscaled and has no final ReLU. Both
// include the reference auxiliary head. Validate every canonical stage,
// independent main/auxiliary/state parameter counts, and gradients through all
// learned tensors at 299 pixels and 1000 classes. Architecture and adaptation
// details: common/inception4.hpp.
//
// For each residual block type, compare shared-weight evaluations at scales
// zero, s and 2s. Check the identity path, output scaling, parameter-gradient
// scaling and the identity contribution to input gradients. This isolates the
// residual mechanism without making an unsupported empirical stability claim.
//
// Train compact CIFAR-10 demonstrations with every block retained: 107-pixel
// input, one-eighth widths, 12k fit / 5k validation rows, twelve epochs,
// Adam(.001), crop/flip augmentation, BN momentum .9, .8 dropout retention,
// .1 label smoothing and .4-weight auxiliary CE. Select by unsmoothed primary
// validation NLL and evaluate all 10k test rows once. These short CPU runs
// are not full-model ImageNet benchmarks or a controlled residual ablation.
// CIFAR-10 is read from paths::Dataset("cifar10"). BN moving averages update
// eagerly in training passes; inference passes leave the state store unchanged.
// Options: data=PATH epochs=12 batch=128 lr=.001 fit=12000 augment=0|1 threads=N seed=17.

#include <cmath>
#include <iostream>

#include "common/architecture_checks.hpp"
#include "common/image_training.hpp"
#include "common/inception4.hpp"

namespace nn = datamunge::nn;

void ResidualScaleChecks(char kind, int channels, double scale) {
  nn::ParameterStore             p(17), state(17);
  inception23::Inception23Config cfg;
  cfg.width_divisor = 8;
  inception23::Inception23Outputs result;
  inception23::InceptionBuilder   builder(p, state, result, cfg, false);
  std::vector<float>              values(static_cast<std::size_t>(81 * channels));
  for (std::size_t i = 0; i < values.size(); ++i)
    values[i] = .4f + .2f * std::sin(static_cast<float>(i) * .17f);
  nn::Tensor input = nn::from_vector(values, {1, 9, 9, channels});
  input.requires_grad_(true);
  const inception23::ConvFeature x{input, channels};
  const auto first = inception4::InceptionResidual(builder, "block", x, kind, scale, false).value;
  const auto parameters = p.count();
  const auto twice =
      inception4::InceptionResidual(builder, "block", x, kind, 2 * scale, false).value;
  const auto zero = inception4::InceptionResidual(builder, "block", x, kind, 0, false).value;
  check::Require(
      p.count() == parameters, "residual fixture shares parameters across scale choices");
  auto variables = p.parameters();
  variables.insert(variables.begin(), input);
  const auto g1 = nn::grad(nn::sum(first), variables), g2 = nn::grad(nn::sum(twice), variables);
  const auto a = first.to_vector<float>(), b = twice.to_vector<float>(),
             z = zero.to_vector<float>();
  double error = 0;
  for (std::size_t i = 0; i < values.size(); ++i) {
    check::Near(z[i], values[i], 0, "zero residual scale preserves identity");
    error = std::max(error, std::abs((double(b[i]) - values[i]) - 2 * (double(a[i]) - values[i])));
  }
  for (std::size_t parameter = 0; parameter < g1.size(); ++parameter) {
    const auto   ga = g1[parameter].to_vector<float>(), gb = g2[parameter].to_vector<float>();
    const double skip = parameter == 0 ? 1 : 0;
    for (std::size_t i = 0; i < ga.size(); ++i)
      error = std::max(
          error,
          std::abs((gb[i] - skip) - 2 * (ga[i] - skip)) / std::max(1., std::abs(double(gb[i]))));
  }
  check::Near(error, 0, 2e-5, "residual output, input and parameter derivative scaling");
  std::cout << "  residual " << kind << " at scale " << scale
            << ": identity and scaling checks; max normalized error=" << error << std::endl;
}

void FullArchitectureChecks(bool residual) {
  nn::ParameterStore           p(17), state(17);
  inception4::Inception4Config cfg;
  cfg.residual   = residual;
  const auto net = inception4::Inception4(
      p, state, architecture::SinePixels({1, 299, 299, 3}, .1, .13), cfg, false, true);
  const std::int64_t aux_count  = architecture::CountWithPrefix(p, "aux/"),
                     main_count = p.count() - aux_count;
  check::Require(
      main_count == (residual ? 55813192 : 42648232) && aux_count == (residual ? 3366760 : 3358568),
      "reference main/auxiliary parameter counts");
  check::Require(state.count() == (residual ? 62336 : 64960), "reference BN state count");
  check::Require(net.auxiliary.size() == 1, "canonical auxiliary head");
  const auto                       initial_state = state.flatten();
  std::vector<std::pair<int, int>> expected      = {{149, 32}, {147, 32}, {147, 64}};
  if (residual) {
    for (const auto& value :
         std::vector<std::pair<int, int>>{{73, 64}, {73, 80}, {71, 192}, {35, 192}, {35, 320}})
      expected.push_back(value);
    expected.insert(expected.end(), 10, {35, 320});
    expected.emplace_back(17, 1088);
    expected.insert(expected.end(), 20, {17, 1088});
    expected.emplace_back(8, 2080);
    expected.insert(expected.end(), 10, {8, 2080});
    expected.emplace_back(8, 1536);
  } else {
    expected.emplace_back(73, 160);
    expected.emplace_back(71, 192);
    expected.emplace_back(35, 384);
    expected.insert(expected.end(), 4, {35, 384});
    expected.emplace_back(17, 1024);
    expected.insert(expected.end(), 7, {17, 1024});
    expected.emplace_back(8, 1536);
    expected.insert(expected.end(), 3, {8, 1536});
  }
  check::Require(net.stages.size() == expected.size(), "complete architecture block count");
  for (std::size_t i = 0; i < expected.size(); ++i) {
    const auto [spatial, channels] = expected[i];
    architecture::RequireShape(
        net.stages[i].second,
        {1, spatial, spatial, channels},
        "reference stage shape: " + net.stages[i].first);
  }
  check::Require(
      net.logits.shape() == nn::Shape({1, 1000}) &&
          net.auxiliary[0].shape() == nn::Shape({1, 1000}),
      "main and auxiliary classifier shapes");
  const auto objective =
      nn::mean(nn::square(net.logits)) + .4 * nn::mean(nn::square(net.auxiliary[0]));
  const auto [minimum, maximum] =
      architecture::GradientNormRange(p, objective, "advanced Inception");
  check::Require(
      state.flatten() == initial_state, "inference does not update BN running statistics");
  std::cout << "  " << (residual ? "Inception-ResNet-v2" : "Inception v4")
            << ": main=" << main_count << ", aux=" << aux_count << ", state=" << state.count()
            << "; " << expected.size() << " stage shapes; gradient norms [" << minimum << ','
            << maximum << "]" << std::endl;
}

inception4::Inception4Config CompactConfig(bool residual) {
  inception4::Inception4Config cfg;
  cfg.residual      = residual;
  cfg.image_size    = 107;
  cfg.classes       = 10;
  cfg.width_divisor = 8;
  cfg.momentum      = .9;
  return cfg;
}

int main(int argc, char** argv) {
  try {
    ResidualScaleChecks('A', 40, .17);
    ResidualScaleChecks('B', 136, .1);
    ResidualScaleChecks('C', 260, .2);
    FullArchitectureChecks(false);
    FullArchitectureChecks(true);
    for (bool residual : {false, true}) {
      const auto                          compact = CompactConfig(residual);
      const image_training::LogitsBuilder model   = [compact](
                                                      nn::ParameterStore& p,
                                                      nn::ParameterStore& state,
                                                      const nn::Tensor&   x,
                                                      bool                training) {
        return inception4::Inception4(
                   p, state, nn::resize_bilinear(x, {107, 107}, false, true), compact, training)
            .logits;
      };
      const image_training::TrainingBuilder training =
          [compact](nn::ParameterStore& p, nn::ParameterStore& state, const nn::Tensor& x) {
            const auto net = inception4::Inception4(
                p, state, nn::resize_bilinear(x, {107, 107}, false, true), compact, true, true);
            return image_training::TrainingNetwork{net.logits, {{net.auxiliary[0], .4}}};
          };
      image_training::ImageTrainingConfig cfg;
      cfg.name = residual ? "inception_resnet_v2_compact_cifar10" : "inception_v4_compact_cifar10";
      cfg.maximum_fit_rows      = 12000;
      cfg.epochs                = 12;
      cfg.minimum_test_accuracy = .35;
      cfg.label_smoothing       = .1;
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
    std::cout << "All Inception v4 / Inception-ResNet-v2 checks passed.\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
