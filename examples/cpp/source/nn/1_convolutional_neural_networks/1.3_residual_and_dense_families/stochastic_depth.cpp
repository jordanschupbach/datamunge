// CMD: just run-nn stochastic_depth
// Stochastic depth: survival schedules, masking conventions and learning.
//
// Port of tflow's stochastic_depth. Verify original train=bF/test=pF and
// modern inverted train=bF/p/test=F conventions, whole-batch and per-example
// masks, exact input/weight derivatives, fresh masks, empirical survival rates
// and p=0/1 boundaries. The schedule p_l=1-l/L*(1-p_last) has expected active
// depth (3L-1)/4 when p_last=.5. Check the 110-layer CIFAR architecture with
// all gates open, including its average-pool/zero-pad shortcuts and all
// parameter gradients.
// https://arxiv.org/abs/1603.09382
// https://github.com/pytorch/vision/blob/main/torchvision/ops/stochastic_depth.py
//
// Compare two 20-layer, 269,722-parameter CIFAR models: constant depth and
// inverted whole-batch DropPath with final survival .5. Use the same seed,
// 12k fit / 5k validation split, twelve epochs, crop/flip, Adam(.001), kernel
// L2(.0001), and independent best-validation checkpoints before test scoring.
// Every residual is computed and every BN statistic updated; this does not
// implement the original conditional execution or claim speedup.
// Bias-free convolutions and the short optimizer schedule are adaptations.
// One paired run illustrates the procedure, not a general accuracy claim.
// Deviations: masks are drawn from one nn::Generator per model (tflow seeds
// each block's RandomUniform with 71+index); weights come from nn's generator
// (seed=), so trained numbers match TF only statistically. The 110-layer
// gradient check disables the moving-average updates.
// Options: data=PATH epochs=12 batch=128 lr=.001 fit=12000 augment=0|1 threads=N seed=17.

#include "common/stochastic_depth.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>

#include "common/checks.hpp"
#include "common/image_training.hpp"

namespace nn = datamunge::nn;
namespace sd = stochastic_depth;

namespace {
void MaskChecks(bool per_example, sd::DropPathScaling scaling, float survival) {
  nn::ParameterStore p(17);
  nn::Generator      gen(73);
  const int          rows = per_example ? 8192 : 4,
            repeats       = per_example || survival == 0 || survival == 1 ? 1 : 512;
  std::vector<float> values(static_cast<std::size_t>(rows * 8));
  for (std::size_t i = 0; i < values.size(); ++i)
    values[i] = .2f + .1f * std::sin(.17f * static_cast<std::int64_t>(i));
  nn::Tensor x = nn::from_vector(values, {rows, 2, 2, 2});
  x.requires_grad_(true);
  p.set("weight", nn::scalar(1.5));
  const nn::Tensor weight   = p.at("weight");
  const bool       inverted = scaling == sd::DropPathScaling::kInverted;
  std::int64_t     alive = 0, samples = 0;
  double           error = 0;
  for (int repeat = 0; repeat < repeats; ++repeat) {
    const nn::Tensor branch    = x * weight;
    const auto       train     = sd::DropPath(branch, survival, true, per_example, scaling, gen);
    const auto       infer     = sd::DropPath(branch, survival, false, per_example, scaling, gen);
    const nn::Tensor output    = x + train.value;
    const auto       gradients = nn::grad(nn::sum(output), {x, weight});
    const auto       out = output.to_vector<float>(), gate = train.gate.to_vector<float>();
    const auto dx = gradients[0].to_vector<float>(), inference = infer.value.to_vector<float>();
    const bool scalar_gate       = gate.size() == 1;
    double     weight_derivative = 0;
    for (int row = 0; row < rows; ++row) {
      const float g = gate[static_cast<std::size_t>(scalar_gate ? 0 : row)];
      check::Require(g == 0 || g == 1, "Bernoulli gate");
      if (per_example || row == 0) {
        alive += int(g);
        ++samples;
      }
      const double multiplier = inverted && survival > 0 ? g / survival : g;
      for (int item = 0; item < 8; ++item) {
        const auto   index = static_cast<std::size_t>(row * 8 + item);
        const double value = values[index];
        check::Near(
            out[index],
            value * (1 + 1.5 * multiplier),
            3e-6,
            "identity plus uniformly masked row/whole batch");
        check::Near(
            dx[index], 1 + 1.5 * multiplier, 3e-6, "identity plus masked branch input derivative");
        check::Near(
            inference[index],
            1.5 * value * (inverted ? 1 : survival),
            3e-6,
            "deterministic inference scaling");
        weight_derivative += multiplier * value;
      }
    }
    error = std::max(
        error,
        std::abs(gradients[1].item() - weight_derivative) /
            std::max(1., std::abs(weight_derivative)));
  }
  check::Near(error, 0, 3e-5, "masked parameter derivative including zero branch gradient");
  const double empirical = double(alive) / samples;
  check::Near(
      empirical,
      survival,
      4 * std::sqrt(survival * (1 - survival) / samples) + .002,
      "empirical survival and expected branch multiplier");
  if (survival > 0 && survival < 1)
    check::Require(alive > 0 && alive < samples, "both active and inactive masks observed");
  std::cout << "  " << (per_example ? "row" : "batch") << '/'
            << (inverted ? "inverted" : "original") << ", p=" << survival
            << ": empirical=" << empirical << ", derivative error=" << error << '\n';
}

void ScheduleAndShortcutChecks() {
  const auto survival = sd::SurvivalProbabilities(54, .5f);
  check::Near(survival.front(), 1 - .5 / 54, 1e-7, "first scheduled survival");
  check::Near(survival.back(), .5, 0, "last scheduled survival");
  const double expected = std::accumulate(survival.begin(), survival.end(), 0.);
  check::Near(expected, 40.25, 2e-6, "expected active blocks in 110-layer model");
  std::vector<float> values(32);
  for (int i = 0; i < 32; ++i)
    values[static_cast<std::size_t>(i)] = float(i + 1);
  nn::Tensor x = nn::from_vector(values, {1, 4, 4, 2});
  x.requires_grad_(true);
  const nn::Tensor shortcut = sd::CifarIdentityShortcut(x, 4, 2);
  const auto       gradient = nn::grad(nn::sum(shortcut), {x})[0].to_vector<float>();
  check::Require(
      shortcut.shape() == nn::Shape({1, 2, 2, 4}), "CIFAR shortcut spatial/channel dimensions");
  const auto out = shortcut.to_vector<float>();
  for (int r = 0; r < 2; ++r)
    for (int c = 0; c < 2; ++c)
      for (int channel = 0; channel < 4; ++channel) {
        double value = 0;
        if (channel < 2)
          for (int dr = 0; dr < 2; ++dr)
            for (int dc = 0; dc < 2; ++dc)
              value +=
                  .25 *
                  values[static_cast<std::size_t>(((2 * r + dr) * 4 + 2 * c + dc) * 2 + channel)];
        check::Near(
            out[static_cast<std::size_t>((r * 2 + c) * 4 + channel)],
            value,
            0,
            "host average-pool and channel padding");
      }
  for (int i = 0; i < 32; ++i)
    check::Near(gradient[static_cast<std::size_t>(i)], .25, 0, "average-pool shortcut derivative");
  std::cout << "  schedule: 54 possible blocks, " << expected
            << " expected active; option-A shortcut checks passed\n";
}

void FullChecks() {
  nn::ParameterStore        p(17), state(17);
  nn::Generator             gen(71);
  sd::StochasticCifarConfig cfg;
  cfg.last_survival = 1;
  cfg.update_stats  = false;
  std::vector<float> pixels(2 * 32 * 32 * 3);
  for (std::size_t i = 0; i < pixels.size(); ++i)
    pixels[i] = .1f * std::sin(static_cast<std::int64_t>(i) * .13f);
  const auto net =
      sd::StochasticCifarResNet(p, state, nn::from_vector(pixels, {2, 32, 32, 3}), cfg, true, gen);
  check::Require(
      p.count() == 1727962 && state.count() == 8096 && net.gates.size() == 54,
      "110-layer CIFAR parameter/state/block counts");
  for (int stage = 0; stage < 3; ++stage)
    check::Require(
        net.stages[static_cast<std::size_t>(stage)].shape() ==
            nn::Shape({2, 32 >> stage, 32 >> stage, 16 << stage}),
        "CIFAR residual stage dimensions");
  check::Require(net.logits.shape() == nn::Shape({2, 10}), "CIFAR residual classifier dimensions");
  const auto gradients = nn::grad(nn::mean(nn::square(net.logits)), p.parameters());
  double     minimum = std::numeric_limits<double>::infinity(), maximum = 0;
  for (std::size_t i = 0; i < gradients.size(); ++i) {
    const double norm = nn::sqrt(nn::sum(nn::square(gradients[i]))).item();
    check::Require(
        std::isfinite(norm) && norm > 0, "finite nonzero residual derivative: " + p.names()[i]);
    minimum = std::min(minimum, norm);
    maximum = std::max(maximum, norm);
  }
  std::cout << "  110-layer CIFAR model with open gates: parameters=" << p.count()
            << ", gradient norms [" << minimum << ',' << maximum << "]" << std::endl;
}
} // namespace

int main(int argc, char** argv) {
  try {
    for (int i = 1; i < argc; ++i) // threads= also applies to the checks
      if (std::string(argv[i]).rfind("threads=", 0) == 0)
        nn::set_num_threads(std::stoi(std::string(argv[i]).substr(8)));
    for (const auto scaling : {sd::DropPathScaling::kOriginal, sd::DropPathScaling::kInverted}) {
      MaskChecks(false, scaling, .6f);
      MaskChecks(true, scaling, .6f);
      MaskChecks(false, scaling, 0);
      MaskChecks(true, scaling, 1);
    }
    ScheduleAndShortcutChecks();
    FullChecks();
    for (double survival : {1., .5}) {
      image_training::ImageTrainingConfig cfg;
      cfg.name = survival == 1 ? "constant_depth20_cifar10" : "stochastic_depth20_cifar10";
      cfg.maximum_fit_rows      = 12000;
      cfg.epochs                = 12;
      cfg.minimum_test_accuracy = .45;
      auto gen =
          std::make_shared<nn::Generator>(71); // one mask stream per model, persistent across steps
      const image_training::LogitsBuilder builder = [=](nn::ParameterStore& p,
                                                        nn::ParameterStore& state,
                                                        const nn::Tensor&   image,
                                                        bool                training) {
        sd::StochasticCifarConfig model;
        model.depth         = 20;
        model.last_survival = survival;
        const auto logits =
            sd::StochasticCifarResNet(p, state, image, model, training, *gen).logits;
        check::Require(
            p.count() == 269722 && state.count() == 1376,
            "20-layer comparison model parameter/state counts");
        return logits;
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
          model(params, state, ImageBatch(development, {fit[0], fit[1]}).first, false);
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
    std::cout << "All stochastic-depth checks passed.\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
