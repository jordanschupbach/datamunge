// CMD: just run-nn zfnet
// ZFNet: finer early sampling, dense connectivity and deconvnet visualization.
//
// Port of tflow's zfnet. Build the Figure-3 network from Zeiler/Fergus, ECCV
// 2014. Check full 224-pixel stage shapes, 62,357,608 parameters and finite
// nonzero derivatives through every learned tensor. The shared header
// documents exact padding.
// Reference: https://cs.nyu.edu/~fergus/papers/zeilerECCV2014.pdf
//
// Train a separate compact version on 12k CIFAR-10 rows: quarter channel
// widths, 256-unit dense layers, 67-pixel resized input, ten epochs, crop/flip
// augmentation, He initialization and Adam. Keep dense connectivity, 7x7/5x5
// stride-2 early convolutions, LRN after pooling and .5 dropout. Select on a
// fixed 5k validation split and evaluate all 10k official test rows.
// These compact results are not full-model ImageNet results.
//
// Restore the selected model and project its strongest conv5 activation for
// the first test image back to pixels: unpool at forward switches, rectify the
// backward signal and apply convolution adjoints, omitting LRN. Compare with
// the ordinary input gradient. A switch fixture and a ReLU sign fixture
// distinguish this reconstruction rule from standard or guided backprop.
// CIFAR-10 is read from paths::Dataset("cifar10"). Export:
// output/nn/visualizations/zfnet/features.npz (tflow: output/visualizations/zfnet,
// rendered by tflow's scripts/plot_zfnet.py). This is a single-feature
// diagnostic, not an inverse network or a complete explanation of the class.
// Options: data=PATH epochs=10 batch=128 lr=.001 fit=12000 augment=0|1 threads=N seed=17.

#include "common/dataset_assets.hpp"
#include "common/zfnet.hpp"

#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>

#include "common/architecture_checks.hpp"
#include "common/image_training.hpp"

namespace nn = datamunge::nn;

zfnet::ZFNetConfig CompactConfig() {
  zfnet::ZFNetConfig cfg;
  cfg.image_size    = 67;
  cfg.classes       = 10;
  cfg.width_divisor = 4;
  cfg.dense_width   = 256;
  return cfg;
}

void ReconstructionChecks() {
  const auto values = nn::from_vector(
      std::vector<float>{1, 9, 2, 3, 4, 5, 6, 10, 7, 8, 12, 0, 11, 2, 1, 4}, {1, 4, 4, 1});
  const auto signal = nn::from_vector(std::vector<float>{1, 2, 3, 4}, {1, 2, 2, 1});
  const auto unpooled =
      deconvnet::UnpoolAtSwitches(values, signal, 2, 2, "VALID").to_vector<float>();
  nn::Tensor x = nn::from_vector(std::vector<float>{-1, 2}, {1, 1, 1, 2});
  x.requires_grad_(true);
  const auto backward = nn::from_vector(std::vector<float>{3, -4}, {1, 1, 1, 2});
  const auto identity = nn::from_vector(std::vector<float>{1, 0, 0, 1}, {1, 1, 2, 2});
  const auto ordinary = nn::grad(nn::sum(nn::relu(x) * backward), {x})[0].to_vector<float>();
  const auto projection =
      deconvnet::DeconvLayer(x, identity, backward, 1, "VALID").to_vector<float>();
  std::array<float, 16> expected{};
  expected[1]  = 1;
  expected[7]  = 2;
  expected[12] = 3;
  expected[10] = 4;
  for (std::size_t i = 0; i < 16; ++i)
    check::Near(unpooled[i], expected[i], 0, "unpool uses original maximum locations");
  check::Near(ordinary[0], 0, 0, "ordinary ReLU blocks negative forward input");
  check::Near(ordinary[1], -4, 0, "ordinary ReLU can pass a negative derivative");
  check::Near(projection[0], 3, 0, "deconvnet passes positive backward signal");
  check::Near(projection[1], 0, 0, "deconvnet blocks negative backward signal");
  std::cout << "  unpool switches and deconvnet-vs-backprop rectification checks passed\n";
}

void FullArchitectureChecks() {
  nn::ParameterStore p(17);
  const auto         net = zfnet::ZFNet(p, architecture::SinePixels({1, 224, 224, 3}, .5, .13));
  check::Require(p.count() == 62357608, "full ZFNet parameter count");
  const std::vector<nn::Tensor> stages = {
      net.convolution[0],
      net.convolution[1],
      net.convolution[2],
      net.convolution[3],
      net.convolution[4],
      net.pooling[0],
      net.pooling[1],
      net.pooling[2],
      net.fc6,
      net.fc7,
      net.logits};
  const std::vector<nn::Shape> expected = {
      {1, 110, 110, 96},
      {1, 26, 26, 256},
      {1, 13, 13, 384},
      {1, 13, 13, 384},
      {1, 13, 13, 256},
      {1, 55, 55, 96},
      {1, 13, 13, 256},
      {1, 6, 6, 256},
      {1, 4096},
      {1, 4096},
      {1, 1000}};
  for (std::size_t i = 0; i < expected.size(); ++i)
    architecture::RequireShape(stages[i], expected[i], "ZFNet Figure-3 stage shape");
  const auto [minimum, maximum] = architecture::GradientNormRange(
      p, nn::mean(nn::square(net.logits)), "ZFNet parameter tensor");
  std::cout << "  full ZFNet: " << p.count()
            << " parameters; conv sizes 110/26/13/13/13; pool sizes 55/13/6\n"
            << "  dense widths 4096/4096/1000; parameter gradient norms [" << minimum << ','
            << maximum << "]\n";
}

void Visualize(const std::filesystem::path& directory) {
  nn::ParameterStore p(17);
  const auto         test     = dataset_assets::LoadCifar10(directory.string(), "test");
  const auto [pixels, labels] = image_training::ImageBatch(test, {0});
  nn::Tensor image            = nn::resize_bilinear(pixels, {67, 67}, false, true);
  {
    nn::NoGradGuard guard;
    zfnet::ZFNet(p, image, CompactConfig());
  }
  p.load((paths::OutputDir("models/zfnet_compact_cifar10") / "best_params.npz").string());
  image.requires_grad_(true);
  const auto net            = zfnet::ZFNet(p, image, CompactConfig());
  const auto activation_map = net.convolution[4].to_vector<float>();
  check::Require(net.convolution[4].shape() == nn::Shape({1, 3, 3, 64}), "compact conv5 map");
  std::size_t index = 0;
  for (std::size_t i = 1; i < activation_map.size(); ++i)
    if (activation_map[i] > activation_map[index])
      index = i;
  const float activation = activation_map[index];
  check::Require(activation > 0, "positive selected conv5 activation");
  std::vector<float> seed(activation_map.size(), 0.f);
  seed[index]         = activation;
  const auto signal   = nn::from_vector(seed, {1, 3, 3, 64});
  const auto gradient = nn::grad(nn::sum(net.convolution[4] * signal), {image})[0];
  nn::Tensor deconvolution;
  {
    nn::NoGradGuard guard;
    deconvolution = zfnet::ZFNetDeconvolution(p, net, signal);
  }
  check::Require(
      deconvolution.shape() == nn::Shape({1, 67, 67, 3}) &&
          gradient.shape() == deconvolution.shape(),
      "projection image dimensions");
  const auto a = deconvolution.to_vector<float>(), b = gradient.to_vector<float>();
  double     deconv_energy = 0, gradient_energy = 0, difference = 0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    check::Require(std::isfinite(a[i]) && std::isfinite(b[i]), "finite feature projections");
    deconv_energy += double(a[i]) * a[i];
    gradient_energy += double(b[i]) * b[i];
    difference += (double(a[i]) - b[i]) * (double(a[i]) - b[i]);
  }
  check::Require(
      deconv_energy > 0 && gradient_energy > 0 && difference > 1e-8,
      "nonzero distinct deconvnet and ordinary gradients");
  const std::int64_t unit_row = index / (3 * 64), unit_column = (index / 64) % 3,
                     unit_channel = index % 64;
  const auto prediction           = nn::argmax(net.logits.detach(), 1);
  const auto path                 = paths::OutputDir("visualizations/zfnet") / "features.npz";
  nn::save_npz(
      path.string(),
      {{"image", image.detach()},
       {"label", labels},
       {"prediction", prediction},
       {"unit",
        nn::from_vector(std::vector<std::int64_t>{unit_row, unit_column, unit_channel}, {3})},
       {"deconvolution", deconvolution},
       {"gradient", gradient}});
  std::cout << "  first test image conv5 unit=" << unit_row << ',' << unit_column << ','
            << unit_channel << "; activation=" << activation << "; deconv energy=" << deconv_energy
            << "; ordinary gradient energy=" << gradient_energy << "\n  wrote " << path.string()
            << "\nAll ZFNet architecture and visualization checks passed.\n";
}

int main(int argc, char** argv) {
  try {
    using namespace image_training;
    ReconstructionChecks();
    FullArchitectureChecks();
    std::filesystem::path data = paths::Dataset("cifar10");
    ImageTrainingConfig   cfg;
    cfg.name                  = "zfnet_compact_cifar10";
    cfg.maximum_fit_rows      = 12000;
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
    // Compact ZFNet on CIFAR-10: fixed development split, Adam on cross-entropy
    // plus L2, validation-NLL checkpoint selection, one test pass. The whole
    // workflow is written out here so the R and Python examples can follow it
    // line by line over the same bound components; Visualize then reloads the
    // selected checkpoint.
    const auto model =
        [](nn::ParameterStore& p, nn::ParameterStore&, const nn::Tensor& x, bool training) {
          return zfnet::ZFNet(
                     p, nn::resize_bilinear(x, {67, 67}, false, true), CompactConfig(), training)
              .logits;
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
    Visualize(data);
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
