// CMD: just run-nn deformable_conv
// Deformable convolutions v1 / v2: learned sampling offsets (and v2 modulation masks).
//
// Port of tflow's deformable_conv (layer: common/deformable.hpp on
// nn::deformable_conv2d). Check that a new deformable layer (zero offsets)
// equals an ordinary 3x3 convolution with the same weights at strides one and
// two, and that a new modulated layer gives half of it (masks start at
// sigmoid(0)). Check integer offsets (every tap moved one row down reads the
// convolution of the shifted input, with zeros beyond the border) and
// fractional offsets against a host bilinear computation, and derivatives
// with respect to the offsets against central differences. Count the compact
// ResNet-50 with deformable 3x3 convolutions in stages 3 and 4 and check its
// derivatives (training mode, moving averages not updated).
// https://arxiv.org/abs/1703.06211
// https://arxiv.org/abs/1811.11168
//
// Train the compact ResNet-50 of resnet_bottleneck with v1 and with v2
// (modulated) deformable 3x3 convolutions in its last two stages (8x8 and
// 4x4 maps) on CIFAR-10, for comparison with the plain compact ResNet-50. Use
// 12k fit / 5k validation, twelve epochs, crop/flip, Adam(.001), kernel
// L2(.0001) and the best-validation checkpoint before evaluating all 10k test
// images. Weights come from nn's generator, so trained numbers match TF only
// statistically. CIFAR-10 is read from the tflow dataset cache
// (common/paths.hpp) or data=.
// Options: data=PATH epochs=12 batch=128 lr=.001 fit=12000 augment=0|1 threads=N seed=17.

#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>

#include "common/attention_cnn.hpp"
#include "common/checks.hpp"
#include "common/deformable.hpp"
#include "common/image_training.hpp"
#include "common/mobile_checks.hpp"

namespace nn = datamunge::nn;
using mobile_checks::Pattern;

namespace {

const int kSize = 7, kIn = 3, kOut = 4;

// Host bilinear sample of x [1, H, W, C] (flattened) at (y, x), zero outside.
double Sample(const std::vector<float>& x, double y, double xpos, int c) {
  const int y0 = int(std::floor(y)), x0 = int(std::floor(xpos));
  double    v = 0;
  for (int dy = 0; dy <= 1; ++dy)
    for (int dx = 0; dx <= 1; ++dx) {
      const int yy = y0 + dy, xx = x0 + dx;
      if (yy < 0 || yy >= kSize || xx < 0 || xx >= kSize)
        continue;
      v += (dy ? y - y0 : 1 - (y - y0)) * (dx ? xpos - x0 : 1 - (xpos - x0)) *
           x[(yy * kSize + xx) * kIn + c];
    }
  return v;
}

double MaxDifference(const std::vector<float>& a, const std::vector<float>& b) {
  double e = 0;
  for (std::size_t i = 0; i < a.size(); ++i)
    e = std::max(e, double(std::abs(a[i] - b[i])));
  return e;
}

void LayerChecks() {
  const nn::Tensor x = Pattern({1, kSize, kSize, kIn}, 1, .2f);
  for (int stride : {1, 2})
    for (bool modulated : {false, true}) {
      if (stride == 2 && modulated)
        continue;
      nn::ParameterStore p(17);
      const auto layer    = deformable::DeformableConv2D(p, "dcn", x, kOut, 3, stride, modulated);
      nn::Conv2DOptions o = nn::conv_options(stride, nn::Padding::Explicit);
      o.explicit_paddings = {1, 1, 1, 1};
      const nn::Tensor reference = nn::conv2d(x, p.at("dcn/W"), o) * (modulated ? .5 : 1.);
      check::Require(reference.shape() == layer.output.shape(), "deformable output shape");
      const double error =
          MaxDifference(layer.output.to_vector<float>(), reference.to_vector<float>());
      check::Near(
          error,
          0,
          1e-5,
          "a new deformable layer is an ordinary (or half an ordinary) convolution");
      std::cout << "  new " << (modulated ? "modulated " : "") << "deformable layer, stride "
                << stride << ": equals " << (modulated ? "half the " : "the ")
                << "ordinary 3x3 convolution (error " << error << ")" << std::endl;
    }
  // Offsets set through the offset predictor's bias.
  nn::ParameterStore p(17);
  const nn::Tensor   weights = Pattern({1, kSize, kSize, kOut}, 1, .6f);
  auto               loss_of = [&](const std::vector<float>& bias) {
    if (p.contains("dcn/offset/b")) {
      nn::NoGradGuard guard;
      p.set("dcn/offset/b", nn::from_vector(bias, {18}));
    }
    const auto layer = deformable::DeformableConv2D(p, "dcn", x, kOut, 3, 1, false);
    return std::make_pair(layer.output, nn::sum(layer.output * weights));
  };
  loss_of(std::vector<float>(18, 0.f)); // create the parameters
  const auto value = x.to_vector<float>(), w = p.at("dcn/W").to_vector<float>();
  auto       host = [&](const std::vector<float>& b) {
    std::vector<float> y(kSize * kSize * kOut, 0.f);
    for (int i = 0; i < kSize; ++i)
      for (int j = 0; j < kSize; ++j)
        for (int k = 0; k < 9; ++k)
          for (int c = 0; c < kIn; ++c) {
            const double v =
                Sample(value, i - 1 + k / 3 + b[2 * k], j - 1 + k % 3 + b[2 * k + 1], c);
            for (int o = 0; o < kOut; ++o)
              y[(i * kSize + j) * kOut + o] += v * w[(k * kIn + c) * kOut + o];
          }
    return y;
  };
  std::vector<float> down(18, 0.f);
  for (int k = 0; k < 9; ++k)
    down[2 * k] = 1;
  const double       integer = MaxDifference(loss_of(down).first.to_vector<float>(), host(down));
  std::vector<float> fraction(18);
  // Fractional parts in [0.2, 0.8]: central differences then stay on one bilinear piece.
  for (int i = 0; i < 18; ++i) {
    const float v = .37f + .9f * std::sin(1.7f * i) + (i % 3 == 0 ? .25f : 0.f);
    fraction[i]   = std::floor(v) + .2f + .6f * (v - std::floor(v));
  }
  const auto   r          = loss_of(fraction);
  const double fractional = MaxDifference(r.first.to_vector<float>(), host(fraction));
  const auto   symbolic   = nn::grad(r.second, {p.at("dcn/offset/b")})[0].to_vector<float>();
  double       derivative = 0;
  {
    nn::NoGradGuard guard;
    for (int i = 0; i < 18; ++i) {
      auto        plus = fraction, minus = fraction;
      const float h = 1e-2f;
      plus[i] += h;
      minus[i] -= h;
      const double numeric = (static_cast<float>(loss_of(plus).second.item()) -
                              static_cast<float>(loss_of(minus).second.item())) /
                             (2 * h);
      derivative =
          std::max(derivative, std::abs(numeric - symbolic[i]) / std::max(1., std::abs(numeric)));
    }
  }
  check::Near(integer, 0, 1e-5, "integer offsets read the shifted input");
  check::Near(fractional, 0, 1e-5, "fractional offsets interpolate bilinearly");
  check::Near(derivative, 0, 2e-3, "offset derivatives match central differences");
  std::cout << "  offsets: all taps one row down matches the shifted convolution (error " << integer
            << "), fractional offsets match host bilinear sampling (" << fractional
            << "), offset derivatives match central differences (relative " << derivative << ")"
            << std::endl;
}

attention_cnn::AttentionResNetConfig CompactConfig(bool modulated) {
  attention_cnn::AttentionResNetConfig cfg;
  cfg.classes         = 10;
  cfg.width_divisor   = 4;
  cfg.small_stem      = true;
  cfg.stride_on_3x3   = false;
  cfg.deformable_from = 3;
  cfg.modulated       = modulated;
  return cfg;
}

void NetworkChecks() {
  for (bool modulated : {false, true}) {
    nn::ParameterStore p(17), state(17);
    auto               cfg = CompactConfig(modulated);
    cfg.update_stats       = false;
    const auto net =
        attention_cnn::AttentionResNet(p, state, Pattern({2, 32, 32, 3}, 1, 0), cfg, true);
    // The plain compact ResNet-50 (1,484,186) plus offset predictors: 3x3 convolutions from the
    // 64 / 128 channels of the six stage-3 and three stage-4 layers to 18 (v1) or 27 (v2) maps.
    const std::int64_t maps = modulated ? 27 : 18, expected = 1484186 + 6 * (9 * 64 * maps + maps) +
                                                              3 * (9 * 128 * maps + maps);
    check::Require(p.count() == expected, "compact deformable ResNet-50 parameter count");
    const auto  norms   = mobile_checks::GradientNorms(p, nn::mean(nn::square(net.logits)));
    double      minimum = std::numeric_limits<double>::infinity(), maximum = 0;
    std::string smallest;
    for (std::size_t i = 0; i < norms.size(); ++i) {
      check::Require(
          std::isfinite(norms[i]) && norms[i] > 0, "finite nonzero deformable ResNet derivative");
      if (norms[i] < minimum) {
        minimum  = norms[i];
        smallest = p.names()[i];
      }
      maximum = std::max(maximum, norms[i]);
    }
    std::cout << "  compact ResNet-50 with " << (modulated ? "v2 (modulated)" : "v1")
              << " deformable layers in stages 3-4: " << p.count()
              << " parameters; derivatives in [" << minimum << ',' << maximum << "] (smallest "
              << smallest << ")" << std::endl;
  }
}

/// Fits the compact deformable ResNet-50 (plain DCNv1 or modulated DCNv2
/// offsets) on CIFAR-10: fixed development split, Adam on cross-entropy plus
/// L2, validation-NLL checkpoint selection, one test pass. The whole workflow
/// is written out here so the R and Python examples can follow it line by line
/// over the same bound components.
void TrainDeformableResNet(
    bool                                modulated,
    const nn::ImageDataset&             development,
    const nn::ImageDataset&             test,
    image_training::ImageTrainingConfig cfg,
    bool                                explicit_epochs) {
  using namespace image_training;
  cfg.name = modulated ? "dcnv2_resnet50_compact_cifar10" : "dcnv1_resnet50_compact_cifar10";
  const auto model = [modulated](
                         nn::ParameterStore& p,
                         nn::ParameterStore& state,
                         const nn::Tensor&   image,
                         bool                training) {
    return attention_cnn::AttentionResNet(p, state, image, CompactConfig(modulated), training)
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
    for (int i = 1; i < argc; ++i) // threads= also applies to the checks
      if (std::string(argv[i]).rfind("threads=", 0) == 0)
        nn::set_num_threads(std::stoi(std::string(argv[i]).substr(8)));
    LayerChecks();
    NetworkChecks();
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
    for (bool modulated : {false, true})
      TrainDeformableResNet(modulated, development, test, cfg, explicit_epochs);
    std::cout << "All deformable convolution checks passed.\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
