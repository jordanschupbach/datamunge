// CMD: just run-nn googlenet
// GoogLeNet / Inception v1: parallel receptive fields and auxiliary supervision.
//
// Port of tflow's googlenet. Implement the nine modules and 224-pixel/1000-class
// widths in Table 1 of https://arxiv.org/abs/1409.4842. Check all stage
// dimensions, 6,998,552 main parameters and 6,379,728 auxiliary parameters,
// and finite nonzero gradients through every learned tensor. Independently
// check branch filtering and concatenation against host arithmetic, including
// padded image boundaries. Measure how the 1x1 reductions change
// spatial-kernel multiply-add counts.
//
// The two auxiliary heads follow modules 4a/4d: 5x5 stride-3 average pooling,
// 1x1 projection, a dense hidden layer, 70% dropout, and class logits. Their
// cross entropies each have weight .3 during training. Verify weighted losses
// against host log-sum-exp, auxiliary parameter updates with L2 enabled, and
// unchanged inference after perturbing auxiliary parameters. Main dropout is
// 40%. See common/googlenet.hpp for padding, LRN and initialization conventions.
//
// Train a separate CPU demonstration on CIFAR-10: resize to 96 pixels, quarter
// convolution widths, 128-unit auxiliary hidden layers, all nine modules,
// 12k fit / 5k validation rows, crop/flip augmentation, Adam(.001), twelve
// epochs. Select by primary validation NLL and evaluate the complete 10k test
// set. This compact demonstration is not an ImageNet benchmark.
// CIFAR-10 is read from paths::Dataset("cifar10"); auxiliary heads train through
// image_training's TrainingBuilder (tflow's ClassifierNetwork::auxiliary).
// Options: data=PATH epochs=12 batch=128 lr=.001 fit=12000 augment=0|1 threads=N seed=17.

#include "common/googlenet.hpp"

#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>

#include "common/architecture_checks.hpp"
#include "common/image_training.hpp"

namespace nn = datamunge::nn;

void BranchChecks() {
  nn::ParameterStore p(0);
  std::vector<float> pixels(50);
  for (int r = 0; r < 5; ++r)
    for (int c = 0; c < 5; ++c) {
      pixels[static_cast<std::size_t>((r * 5 + c) * 2)]     = 1.f + r * .2f + c * .1f;
      pixels[static_cast<std::size_t>((r * 5 + c) * 2 + 1)] = 2.f + r * .1f + c * .3f;
    }
  const auto input = nn::from_vector(pixels, {1, 5, 5, 2});
  {
    nn::NoGradGuard guard;
    inception::InceptionV1(p, "block", input, {2, 1, 1, 1, 1, 1});
    for (std::size_t i = 0; i < p.size(); ++i) {
      const auto name = p.names()[i];
      auto value = std::vector<float>(static_cast<std::size_t>(p.parameters()[i].numel()), 0.f);
      if (name == "block/one/W")
        value[0] = 1, value[3] = 1;
      else if (name == "block/three_reduce/W")
        value[0] = 1, value[1] = 2;
      else if (name == "block/five_reduce/W")
        value[0] = 3, value[1] = 1;
      else if (name == "block/three/W" || name == "block/five/W")
        std::fill(value.begin(), value.end(), .1f);
      else if (name == "block/pool/W")
        value[0] = 5, value[1] = 6;
      p.set(name, nn::from_vector(value, p.parameters()[i].shape()));
    }
  }
  const auto net = inception::InceptionV1(p, "block", input, {2, 1, 1, 1, 1, 1});
  architecture::RequireShape(net.concatenated, {1, 5, 5, 5}, "Inception branch channel sum");
  const auto out   = net.concatenated.to_vector<float>();
  double     error = 0;
  auto       at    = [&](int y, int x, int c) {
    return double(pixels[static_cast<std::size_t>((y * 5 + x) * 2 + c)]);
  };
  for (int r = 0; r < 5; ++r)
    for (int c = 0; c < 5; ++c) {
      std::array<double, 5> expected = {at(r, c, 0), at(r, c, 1), 0, 0, 0};
      double                m0 = 0, m1 = 0;
      for (int dr = -2; dr <= 2; ++dr)
        for (int dc = -2; dc <= 2; ++dc) {
          const int y = r + dr, x = c + dc;
          if (y < 0 || y >= 5 || x < 0 || x >= 5)
            continue;
          const double a = at(y, x, 0), b = at(y, x, 1);
          expected[3] += .1 * (3 * a + b);
          if (std::abs(dr) <= 1 && std::abs(dc) <= 1) {
            expected[2] += .1 * (a + 2 * b);
            m0 = std::max(m0, a);
            m1 = std::max(m1, b);
          }
        }
      expected[4] = 5 * m0 + 6 * m1;
      for (int channel = 0; channel < 5; ++channel)
        error = std::max(
            error,
            std::abs(
                out[static_cast<std::size_t>((r * 5 + c) * 5 + channel)] -
                expected[static_cast<std::size_t>(channel)]));
    }
  check::Near(
      error,
      0,
      1e-5,
      "four branches and concatenation match independent boundary-aware arithmetic");
  const auto         w       = googlenet::GoogLeNetWidths()[0];
  const std::int64_t reduced = w.KernelWeights(192),
                     direct  = 192LL * (w.one + 9 * w.three + 25 * w.five + w.pool);
  check::Require(
      reduced == 163328 && direct == 393216, "first-module kernel MACs per spatial position");
  std::cout << "  branch arithmetic max error=" << error
            << "; Inception 3a MACs/position: reduced=" << reduced
            << ", direct spatial convolutions=" << direct << " (pool comparisons excluded)\n";
}

double CrossEntropy(const nn::Tensor& scores, const std::vector<std::int64_t>& labels) {
  const auto values = scores.to_vector<float>();
  const auto rows = scores.shape()[0], columns = scores.shape()[1];
  double     sum = 0;
  for (std::int64_t r = 0; r < rows; ++r) {
    double largest = -std::numeric_limits<double>::infinity(), z = 0;
    for (std::int64_t c = 0; c < columns; ++c)
      largest = std::max(largest, double(values[static_cast<std::size_t>(r * columns + c)]));
    for (std::int64_t c = 0; c < columns; ++c)
      z += std::exp(values[static_cast<std::size_t>(r * columns + c)] - largest);
    sum += largest + std::log(z) -
           values[static_cast<std::size_t>(r * columns + labels[static_cast<std::size_t>(r)])];
  }
  return sum / rows;
}

void AuxiliaryLossChecks() {
  nn::ParameterStore                  p(17), state(17);
  const image_training::LogitsBuilder model =
      [](nn::ParameterStore& q, nn::ParameterStore&, const nn::Tensor& x, bool) {
        return nn::dense(q, "main", x, 3);
      };
  const image_training::TrainingBuilder training =
      [](nn::ParameterStore& q, nn::ParameterStore&, const nn::Tensor& x) {
        return image_training::TrainingNetwork{
            nn::dense(q, "main", x, 3), {{nn::dense(q, "aux", x, 3), .3}}};
      };
  const auto x = nn::from_vector(std::vector<float>{1, -.2f, -.4f, .7f}, {2, 2});
  const std::vector<std::int64_t>     labels = {0, 2};
  const auto                          y      = nn::from_vector(labels, {2});
  image_training::ImageTrainingConfig cfg;
  cfg.classes = 3;
  cfg.l2      = 0; // tflow's training_loss excludes the L2 term
  {
    nn::NoGradGuard guard;
    training(p, state, x);
  }
  auto losses = [&]() {
    nn::NoGradGuard guard;
    const auto      main = nn::dense(p, "main", x, 3), aux = nn::dense(p, "aux", x, 3);
    const double    loss =
        image_training::TrainingObjective(p, state, model, x, y, cfg, training).item();
    check::Near(
        loss,
        CrossEntropy(main, labels) + .3 * CrossEntropy(aux, labels),
        2e-6,
        "primary plus weighted auxiliary cross entropy");
    return loss;
  };
  auto inference = [&]() {
    nn::NoGradGuard guard;
    return nn::softmax(model(p, state, x, false), 1).to_vector<float>();
  };
  const auto   before   = inference();
  const double old_loss = losses();
  p.set("aux/b", nn::from_vector(std::vector<float>{3, -2, 1}, {3}));
  const auto   after    = inference();
  const double new_loss = losses();
  for (int i = 0; i < 6; ++i)
    check::Near(
        before[static_cast<std::size_t>(i)],
        after[static_cast<std::size_t>(i)],
        0,
        "auxiliary head cannot change inference");
  check::Require(
      std::abs(old_loss - new_loss) > .01, "auxiliary head changes the training objective");
  // Biases are excluded from L2: their movement demonstrates auxiliary CE
  // connectivity even when regularization is enabled.
  cfg.l2 = .1;
  nn::OptimizerConfig config;
  config.kind = nn::OptimizerKind::Adam;
  nn::Optimizer optimizer(p, .01, config);
  optimizer.minimize(image_training::TrainingObjective(p, state, model, x, y, cfg, training));
  check::Require(
      std::abs(p.at("aux/b").to_vector<float>()[0] - 3.f) > .001,
      "auxiliary cross entropy updates its parameters with L2 enabled");
  std::cout << "  auxiliary loss: " << old_loss << " -> " << new_loss
            << " after aux-only perturbation; inference unchanged; training updates aux bias\n";
}

void FullArchitectureChecks() {
  nn::ParameterStore p(17);
  const auto         net =
      googlenet::GoogLeNet(p, architecture::SinePixels({1, 224, 224, 3}, .1, .13), {}, false, true);
  const std::int64_t aux_count  = architecture::CountWithPrefix(p, "aux"),
                     main_count = p.count() - aux_count;
  check::Require(
      main_count == 6998552 && aux_count == 6379728,
      "canonical main and auxiliary parameter counts");
  nn::Tensor objective = nn::mean(nn::square(net.logits));
  for (const auto& aux : net.auxiliary)
    objective = objective + .3 * nn::mean(nn::square(aux));
  std::vector<nn::Tensor> stages(net.stem.begin(), net.stem.end());
  stages.insert(stages.end(), net.inception.begin(), net.inception.end());
  stages.push_back(net.logits);
  stages.insert(stages.end(), net.auxiliary.begin(), net.auxiliary.end());
  const std::vector<nn::Shape> shapes = {
      {1, 112, 112, 64},
      {1, 56, 56, 64},
      {1, 56, 56, 192},
      {1, 28, 28, 256},
      {1, 28, 28, 480},
      {1, 14, 14, 512},
      {1, 14, 14, 512},
      {1, 14, 14, 512},
      {1, 14, 14, 528},
      {1, 14, 14, 832},
      {1, 7, 7, 832},
      {1, 7, 7, 1024},
      {1, 1000},
      {1, 1000},
      {1, 1000}};
  check::Require(stages.size() == shapes.size(), "GoogLeNet stage count");
  for (std::size_t i = 0; i < shapes.size(); ++i)
    architecture::RequireShape(stages[i], shapes[i], "GoogLeNet stage shape");
  const auto [minimum, maximum] =
      architecture::GradientNormRange(p, objective, "GoogLeNet parameter tensor");
  std::cout << "  full GoogLeNet: main=" << main_count << ", auxiliary=" << aux_count
            << ", total=" << p.count() << "; all 15 stage shapes passed; gradient norms ["
            << minimum << ',' << maximum << "]\n";
}

googlenet::GoogLeNetConfig CompactConfig() {
  googlenet::GoogLeNetConfig cfg;
  cfg.image_size       = 96;
  cfg.classes          = 10;
  cfg.width_divisor    = 4;
  cfg.auxiliary_hidden = 128;
  return cfg;
}

int main(int argc, char** argv) {
  try {
    using namespace image_training;
    BranchChecks();
    AuxiliaryLossChecks();
    FullArchitectureChecks();
    std::filesystem::path data = paths::Dataset("cifar10");
    ImageTrainingConfig   cfg;
    cfg.name                  = "googlenet_compact_cifar10";
    cfg.maximum_fit_rows      = 12000;
    cfg.epochs                = 12;
    cfg.minimum_test_accuracy = .35;
    bool explicit_epochs      = false;
    for (int i = 1; i < argc; ++i) {
      const std::string argument = argv[i];
      const auto        equal    = argument.find('=');
      if (equal == std::string::npos)
        throw std::invalid_argument("expected key=value option");
      explicit_epochs = explicit_epochs || argument.substr(0, equal) == "epochs";
      if (!ParseCifarOption(argument.substr(0, equal), argument.substr(equal + 1), data, cfg))
        throw std::invalid_argument("unknown option: " + argument);
    }
    const auto [development, test] = LoadCifar10(data);
    // Compact GoogLeNet on CIFAR-10: fixed development split, Adam on the primary
    // cross-entropy plus the two .3-weight auxiliary heads plus L2, validation-NLL
    // checkpoint selection on the primary logits, one test pass. The whole workflow
    // is written out here so the R and Python examples can follow it line by line
    // over the same bound components.
    const auto model =
        [](nn::ParameterStore& p, nn::ParameterStore&, const nn::Tensor& x, bool training) {
          return googlenet::GoogLeNet(
                     p, nn::resize_bilinear(x, {96, 96}, false, true), CompactConfig(), training)
              .logits;
        };
    // Training path: the same network with its auxiliary heads (tflow's
    // ClassifierNetwork::auxiliary).
    const auto training = [](nn::ParameterStore& p, nn::ParameterStore&, const nn::Tensor& x) {
      const auto net = googlenet::GoogLeNet(
          p, nn::resize_bilinear(x, {96, 96}, false, true), CompactConfig(), true, true);
      return TrainingNetwork{net.logits, {{net.auxiliary[0], .3}, {net.auxiliary[1], .3}}};
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
      const nn::Tensor x = ImageBatch(development, {fit[0], fit[1]}).first;
      model(params, state, x, false);
      // Training-only parameters (auxiliary heads); their batch-norm statistics
      // go to a scratch store so this pass updates no real moving averages.
      nn::ParameterStore scratch(cfg.seed);
      training(params, scratch, x);
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
        // Augmented batch; training-path forward; (smoothed) cross-entropy of the primary logits
        // plus the weighted auxiliary cross-entropies and penalties, plus 0.5 * l2 * sum(W^2); one
        // Adam step.
        const auto [x, y] = ImageBatch(development, rows, cfg.augment ? &augmentation : nullptr);
        const auto net    = training(params, state, x);
        nn::Tensor objective =
            net.loss ? net.loss(y)
                     : TrainingCrossEntropy(net.logits, y, cfg.classes, cfg.label_smoothing);
        for (const auto& [aux, weight] : net.auxiliary) {
          check::Require(
              std::isfinite(weight) && weight > 0,
              "auxiliary loss weight must be positive and finite");
          objective =
              objective + weight * TrainingCrossEntropy(aux, y, cfg.classes, cfg.label_smoothing);
        }
        for (const auto& penalty : net.penalties)
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
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
