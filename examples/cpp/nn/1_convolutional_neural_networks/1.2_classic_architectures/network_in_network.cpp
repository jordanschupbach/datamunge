// CMD: just run-nn network_in_network
// Network in Network: shared local MLPs and directly averaged class maps.
//
// Port of tflow's network_in_network. Follow the authors' CIFAR-10 topology:
// 5x5-192/1x1-160/1x1-96, 5x5-192/1x1-192/1x1-192, then 3x3-192/1x1-192/1x1-10.
// Every convolution has ReLU; early blocks have max pooling and dropout.
// Check the full 966,986-parameter graph and gradients. Independently compare
// MLPconv with an MLP on explicit im2col patches, including all parameter
// derivatives. A fixed pointwise MLP implements XOR, demonstrating nonlinear
// channel mixing that a single linear filter cannot separate.
// References: https://arxiv.org/abs/1312.4400
// https://github.com/mavenlin/cuda-convnet/blob/master/NIN/cifar-10_def
//
// Train a quarter-width CPU demonstration on 12k CIFAR-10 fitting rows for
// twelve epochs with crop/flip augmentation, Adam and He initialization.
// Select on 5k validation rows, evaluate all 10k test rows, then restore the
// selected model to export one image per class and its ten 8x8 class maps.
// Verify each spatial map mean equals its logit. The maps are scores, not
// spatial probabilities or ground-truth object masks. Our simple [-1,1]
// normalization omits the original contrast normalization and ZCA whitening;
// this demonstration does not reproduce the paper's training results.
// CIFAR-10 is read from paths::Dataset("cifar10"); the selected model is
// output/nn/models/nin_compact_cifar10/best_params.npz and the export goes to
// output/nn/visualizations/nin/class_maps.npz (tflow: output/visualizations/nin,
// plotted by tflow's scripts/plot_nin.py; images there are NHWC in [-1,1]).
// Options: data=PATH epochs=12 batch=128 lr=.001 fit=12000 augment=0|1 threads=N seed=17.

#include <cmath>
#include <iostream>

#include "common/architecture_checks.hpp"
#include "common/im2col.hpp"
#include "common/image_training.hpp"
#include "common/nin.hpp"

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

void LocalMlpChecks() {
  nn::ParameterStore p(31);
  const auto         pixels      = architecture::SinePixels({2, 5, 4, 3}, 1, .23);
  const auto         convolution = nin::MlpConv(p, "mlp", pixels, 4, 5, 2, 3);
  const auto         columns     = im2col::Im2Col(pixels, im2col::PatchGeometry(pixels.shape(), 3, 3, 1, 1, 1, 1, "SAME"));
  nn::Tensor         h           = columns;
  const std::array<std::string, 3> names   = {"spatial", "mix1", "mix2"};
  const std::array<int, 3>         inputs  = {27, 4, 5}, outputs = {4, 5, 2};
  for (std::size_t layer = 0; layer < 3; ++layer) {
    const auto w = nn::reshape(p.at("mlp/" + names[layer] + "/W"), {inputs[layer], outputs[layer]});
    h            = nn::relu(nn::matmul(h, w) + p.at("mlp/" + names[layer] + "/b"));
  }
  const auto matrix_grad      = nn::grad(nn::mean(nn::square(h)), p.parameters());
  const auto convolution_grad = nn::grad(nn::mean(nn::square(convolution)), p.parameters());
  double     output_error = 0, gradient_error = 0;
  const auto a = convolution.to_vector<float>(), b = h.to_vector<float>();
  for (std::size_t i = 0; i < a.size(); ++i) output_error = std::max(output_error, double(std::abs(a[i] - b[i])));
  for (std::size_t parameter = 0; parameter < p.size(); ++parameter) {
    const auto x = matrix_grad[parameter].to_vector<float>(), y = convolution_grad[parameter].to_vector<float>();
    for (std::size_t i = 0; i < x.size(); ++i) gradient_error = std::max(gradient_error, double(std::abs(x[i] - y[i])));
  }
  check::Near(output_error, 0, 2e-6, "MLPconv equals shared patch MLP");
  check::Near(gradient_error, 0, 2e-6, "all shared local MLP parameter derivatives agree");
  std::cout << "  MLPconv vs explicit patch MLP: output error=" << output_error << "; gradient error=" << gradient_error
            << '\n';
}

void XorCheck() {
  nn::ParameterStore p(0);
  p.set("mlp/spatial/W", nn::ones({1, 1, 2, 2}));
  p.set("mlp/spatial/b", nn::from_vector(std::vector<float>{0, -1}, {2}));
  p.set("mlp/mix1/W", nn::from_vector(std::vector<float>{1, 0, 0, 1}, {1, 1, 2, 2}));
  p.set("mlp/mix1/b", nn::zeros({2}));
  p.set("mlp/mix2/W", nn::from_vector(std::vector<float>{1, -2}, {1, 1, 2, 1}));
  p.set("mlp/mix2/b", nn::zeros({1}));
  const auto inputs = nn::from_vector(std::vector<float>{0, 0, 0, 1, 1, 0, 1, 1}, {1, 1, 4, 2});
  const auto result = nin::MlpConv(p, "mlp", inputs, 2, 2, 1, 1).to_vector<float>();
  const std::array<float, 4> expected = {0, 1, 1, 0};
  for (std::size_t i = 0; i < 4; ++i) check::Near(result[i], expected[i], 0, "shared pointwise MLP computes XOR");
  std::cout << "  pointwise MLP XOR outputs: 0 1 1 0\n";
}

void FullArchitectureChecks() {
  nn::ParameterStore p(17);
  const auto         net = nin::NetworkInNetwork(p, architecture::SinePixels({2, 32, 32, 3}, 1, .13));
  check::Require(p.count() == 966986, "full CIFAR-10 NiN parameter count");
  const std::array<nn::Shape, 4>  shapes = {{{2, 32, 32, 96}, {2, 16, 16, 192}, {2, 8, 8, 10}, {2, 10}}};
  const std::array<nn::Tensor, 4> stages = {net.blocks[0], net.blocks[1], net.blocks[2], net.logits};
  for (std::size_t i = 0; i < 4; ++i) architecture::RequireShape(stages[i], shapes[i], "NiN block output shape");
  architecture::GradientNormRange(p, nn::mean(nn::square(net.logits)), "NiN parameter");
  const auto maps = net.blocks[2].to_vector<float>(), logits = net.logits.to_vector<float>();
  for (int row = 0; row < 2; ++row)
    for (int c = 0; c < 10; ++c) {
      double average = 0;
      for (int i = 0; i < 64; ++i) average += maps[static_cast<std::size_t>((row * 64 + i) * 10 + c)] / 64;
      check::Near(logits[static_cast<std::size_t>(row * 10 + c)], average, 2e-6, "class map averages directly to logit");
    }
  std::cout << "  full NiN: 966986 parameters; three MLPconv blocks and ten directly averaged class maps; gradients passed\n";
}

nin::NinConfig CompactConfig() {
  nin::NinConfig cfg;
  cfg.width_divisor = 4;
  return cfg;
}

void ExportClassMaps(const std::filesystem::path& directory) {
  nn::ParameterStore p(17);
  const auto         test = nn::load_cifar10(directory.string(), "test");
  const auto         labels = test.labels.to_vector<std::int64_t>();
  std::vector<std::int64_t> rows(10, -1);
  for (std::size_t i = 0; i < labels.size(); ++i)
    if (rows[static_cast<std::size_t>(labels[i])] < 0) rows[static_cast<std::size_t>(labels[i])] = static_cast<std::int64_t>(i);
  for (auto row : rows) check::Require(row >= 0, "one test image per class");
  const auto [pixels, y] = image_training::ImageBatch(test, rows);
  nn::NoGradGuard guard;
  nin::NetworkInNetwork(p, pixels, CompactConfig());
  p.load((paths::OutputDir("models/nin_compact_cifar10") / "best_params.npz").string());
  const auto net         = nin::NetworkInNetwork(p, pixels, CompactConfig());
  const auto maps        = net.blocks[2].to_vector<float>(), logits = net.logits.to_vector<float>();
  for (int row = 0; row < 10; ++row)
    for (int c = 0; c < 10; ++c) {
      double mean = 0;
      for (int i = 0; i < 64; ++i) mean += maps[static_cast<std::size_t>((row * 64 + i) * 10 + c)] / 64;
      check::Near(mean, logits[static_cast<std::size_t>(row * 10 + c)], 2e-5, "trained class map means equal logits");
    }
  const auto path = paths::OutputDir("visualizations/nin") / "class_maps.npz";
  nn::save_npz(path.string(), {{"images", pixels},
                               {"labels", y},
                               {"maps", net.blocks[2]},
                               {"logits", net.logits},
                               {"predictions", nn::argmax(net.logits, 1)},
                               {"probabilities", nn::softmax(net.logits, 1)}});
  std::cout << "  ten first-of-class test images exported without prediction filtering; class-map identity passed\n"
            << "  wrote " << path.string() << "\nAll Network-in-Network checks passed.\n";
}

int main(int argc, char** argv) {
  try {
    LocalMlpChecks();
    XorCheck();
    FullArchitectureChecks();
    const image_training::LogitsBuilder model = [](nn::ParameterStore& p, nn::ParameterStore&, const nn::Tensor& x,
                                                   bool training) {
      return nin::NetworkInNetwork(p, x, CompactConfig(), training).logits;
    };
    image_training::ImageTrainingConfig cfg;
    cfg.name                  = "nin_compact_cifar10";
    cfg.maximum_fit_rows      = 12000;
    cfg.epochs                = 12;
    cfg.minimum_test_accuracy = .35;
    if (image_training::RunCifar10(argc, argv, cfg, model)) return 1;
    std::filesystem::path directory = paths::Dataset("cifar10");
    for (int i = 1; i < argc; ++i)
      if (std::string(argv[i]).rfind("data=", 0) == 0) directory = std::string(argv[i]).substr(5);
    ExportClassMaps(directory);
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
