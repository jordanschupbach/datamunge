// CMD: just run-nn convnext_v2
// ConvNeXt V2: global response normalization in place of layer scale.
//
// Port of tflow's convnext_v2 (model: common/convnext.hpp). Check global
// response normalization against its formula (per-channel spatial L2 norms
// divided by their channel mean, gamma * x * N + beta + x), that it is the
// identity at initialization (gamma = beta = 0), that its normalized
// responses are invariant to a global rescaling of the input, and that an
// all-zero input or a dead channel keeps finite, identity input derivatives.
// Measure all eight released sizes at 224 pixels: Atto 3,708,400 learned
// parameters (0.55G multiply-adds), Femto 5,233,240 (0.78G), Pico 9,066,280
// (1.37G), Nano 15,623,800 (2.45G), Tiny 28,635,496 (4.46G), Base 88,717,800
// (15.35G), Large 197,956,840 (34.36G) and Huge 660,289,640 (114.92G), the
// paper's 3.7/5.2/9.1/15.6/28.6/89/198/660M and 0.55/0.78/1.37/2.45/4.47/
// 15.4/34.4/115G. As in tflow the multiply-adds follow from the fixed stage
// sizes 56/28/14/7; the parameters are counted by building each model (here
// eagerly on a 32-pixel input, since counts do not depend on the image size;
// Huge allocates its 2.6 GB of weights briefly). Check every derivative of
// V2-Tiny in training mode with active GRN (nonzero gamma and beta). The
// masked-autoencoder (FCMAE) pretraining that motivated GRN is not
// implemented here.
// https://arxiv.org/abs/2301.00808
// https://github.com/facebookresearch/ConvNeXt-V2
//
// Train V2-Atto on CIFAR-10 exactly as convnext trains V1-Atto (2x2/2 stem,
// stochastic depth 0.1), so the two runs compare layer scale with GRN. Use
// 12k fit / 5k validation, twelve epochs, crop/flip, Adam(.001), kernel
// L2(.0001) and the best-validation checkpoint before evaluating all 10k
// test images. Weights and drop-path masks come from nn's generator, so
// trained numbers match TF only statistically. CIFAR-10 is read from the
// tflow dataset cache (common/paths.hpp) or data=.
// Options: data=PATH epochs=12 batch=128 lr=.001 fit=12000 augment=0|1 threads=N seed=17.

#include <cctype>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>

#include "common/checks.hpp"
#include "common/convnext.hpp"
#include "common/image_training.hpp"
#include "common/mobile_checks.hpp"

namespace nn = datamunge::nn;
using mobile_checks::Pattern;

namespace {

void GrnChecks() {
  const int          size = 5, channels = 12;
  nn::ParameterStore p(17);
  const nn::Tensor   value = Pattern({2, size, size, channels}, 1, .2f);
  double             identity;
  {
    nn::NoGradGuard guard;
    identity =
        nn::max(nn::abs(convnext::GlobalResponseNorm(p, "grn", value).output - value)).item();
    p.set("grn/gamma", Pattern({channels}, .8f, 1));
    p.set("grn/beta", Pattern({channels}, .3f, 2));
  }
  const auto grn = convnext::GlobalResponseNorm(p, "grn", value);
  const auto out = grn.output.to_vector<float>(), response = grn.response.to_vector<float>();
  const auto scaled =
      convnext::GlobalResponseNorm(p, "grn", value * 7.0).response.to_vector<float>();
  const auto gamma = p.at("grn/gamma").to_vector<float>(),
             beta  = p.at("grn/beta").to_vector<float>();
  const auto v     = value.to_vector<float>();
  double     error = 0, invariance = 0;
  for (int n = 0; n < 2; ++n) {
    std::vector<double> norm(channels, 0);
    double              mean = 0;
    for (int i = 0; i < size * size; ++i)
      for (int c = 0; c < channels; ++c)
        norm[c] += std::pow(v[(n * size * size + i) * channels + c], 2);
    for (auto& x : norm) {
      x = std::sqrt(x);
      mean += x / channels;
    }
    for (int i = 0; i < size * size; ++i)
      for (int c = 0; c < channels; ++c) {
        const std::size_t k        = (n * size * size + i) * channels + c;
        const double      expected = gamma[c] * v[k] * norm[c] / (mean + 1e-6) + beta[c] + v[k];
        error                      = std::max(error, std::abs(out[k] - expected));
      }
    for (int c = 0; c < channels; ++c)
      invariance = std::max(
          invariance, double(std::abs(response[n * channels + c] - scaled[n * channels + c])));
  }
  check::Near(identity, 0, 0, "GRN is the identity at initialization");
  check::Near(error, 0, 1e-5, "global response normalization");
  check::Near(invariance, 0, 1e-5, "GRN responses are invariant to input scale");
  auto input_gradient = [&](const nn::Tensor& input) {
    nn::Tensor x = input.detach().requires_grad_(true);
    return nn::grad(nn::sum(convnext::GlobalResponseNorm(p, "grn", x).output), {x})[0]
        .to_vector<float>();
  };
  const auto zero = input_gradient(nn::zeros({2, size, size, channels}));
  for (float g : zero)
    check::Near(g, 1, 0, "all-zero GRN has finite identity input derivative");
  std::vector<float> mixed = v;
  for (int i = 0; i < 2 * size * size; ++i)
    mixed[i * channels] = 0;
  const auto dead = input_gradient(nn::from_vector(mixed, {2, size, size, channels}));
  for (float g : dead)
    check::Require(std::isfinite(g), "finite derivatives with a dead channel");
  for (int i = 0; i < 2 * size * size; ++i)
    check::Near(dead[i * channels], 1, 0, "dead channel retains identity derivative");
  std::cout
      << "  all-zero and single-zero-channel GRN gradients remain finite with identity derivative"
      << std::endl;
  std::cout
      << "  GRN (12 channels): identity at initialization; output matches gamma*x*N+beta+x (error "
      << error << "); responses unchanged when the input is scaled by 7 (difference " << invariance
      << ")" << std::endl;
}

void CostChecks() {
  struct Case {
    const char*  size;
    std::int64_t params, macs;
  };
  const std::vector<Case> cases = {
      {"atto", 3708400, 547332480},
      {"femto", 5233240, 779629824},
      {"pico", 9066280, 1367055360},
      {"nano", 15623800, 2445528320},
      {"tiny", 28635496, 4455531264},
      {"base", 88717800, 15354729472},
      {"large", 197956840, 34361433600},
      {"huge", 660289640, 114922100480}};
  for (const auto& c : cases) {
    const auto   layout = convnext::ConvNeXtLayout(c.size);
    std::int64_t params;
    {
      nn::ParameterStore       p(17);
      nn::NoGradGuard          guard;
      convnext::ConvNeXtConfig cfg;
      cfg.size = c.size;
      cfg.v2   = true;
      convnext::ConvNeXt(p, nn::zeros({1, 32, 32, 3}), cfg);
      params = p.count();
    }
    // Stage sizes are fixed by the stem and stride-two joins: 56/28/14/7.
    std::int64_t macs = std::int64_t(layout.dims.back()) * 1000, n = 56;
    for (std::size_t stage = 0; stage < layout.depths.size(); ++stage, n /= 2) {
      const std::int64_t d = layout.dims[stage];
      macs += layout.depths[stage] * n * n * (49 * d + 8 * d * d) +
              (stage ? n * n * 4 * layout.dims[stage - 1] * d : n * n * 48 * d);
    }
    check::Require(
        params == c.params && macs == c.macs, "ConvNeXt V2 parameters and multiply-adds");
    std::cout << "  ConvNeXt V2-" << char(std::toupper(c.size[0])) << (c.size + 1) << ": " << params
              << " parameters, " << std::fixed << std::setprecision(2) << macs / 1e9
              << "G mult-adds" << std::defaultfloat << std::setprecision(6) << std::endl;
  }
}

void GradientChecks() {
  nn::ParameterStore       p(17);
  convnext::ConvNeXtConfig cfg;
  cfg.v2 = true;
  {
    nn::NoGradGuard guard;
    convnext::ConvNeXt(p, nn::zeros({1, 32, 32, 3}), cfg); // create the parameters
    for (const auto& name : p.names())
      if (name.find("/grn/") != std::string::npos)
        p.set(name, Pattern(p.at(name).shape(), .5f, static_cast<float>(name.size())));
  }
  const auto net = convnext::ConvNeXt(p, Pattern({2, 224, 224, 3}, 1, 0), cfg, true);
  check::Require(
      net.features.shape()[1] == 7 && net.features.shape()[3] == 768,
      "ConvNeXt V2-Tiny 7x7x768 features");
  const auto  norms   = mobile_checks::GradientNorms(p, nn::mean(nn::square(net.logits)));
  double      minimum = std::numeric_limits<double>::infinity(), maximum = 0;
  std::string smallest;
  for (std::size_t i = 0; i < norms.size(); ++i) {
    check::Require(std::isfinite(norms[i]), "finite ConvNeXt V2 derivative");
    if (norms[i] < minimum) {
      minimum  = norms[i];
      smallest = p.names()[i];
    }
    maximum = std::max(maximum, norms[i]);
  }
  check::Require(minimum > 1e-6 * maximum, "clearly nonzero ConvNeXt V2 derivatives");
  std::cout << "  ConvNeXt V2-Tiny with active GRN: " << norms.size() << " derivatives in ["
            << minimum << ',' << maximum << "] (smallest " << smallest << ")" << std::endl;
}

} // namespace

int main(int argc, char** argv) {
  try {
    using namespace image_training;
    for (int i = 1; i < argc; ++i) // threads= also applies to the checks
      if (std::string(argv[i]).rfind("threads=", 0) == 0)
        nn::set_num_threads(std::stoi(std::string(argv[i]).substr(8)));
    GrnChecks();
    CostChecks();
    GradientChecks();
    std::filesystem::path dir = paths::Dataset("cifar10");
    ImageTrainingConfig   cfg;
    cfg.name                  = "convnext_v2_atto_cifar10";
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
    const auto model               = [](nn::ParameterStore& p,
                          nn::ParameterStore& state,
                          const nn::Tensor&   image,
                          bool                training) {
      convnext::ConvNeXtConfig architecture;
      architecture.size       = "atto";
      architecture.v2         = true;
      architecture.classes    = 10;
      architecture.cifar_stem = true;
      architecture.drop_path  = .1;
      const auto logits       = convnext::ConvNeXt(p, image, architecture, training).logits;
      check::Require(
          p.count() == 3389170 && state.count() == 0, "CIFAR ConvNeXt V2-Atto parameter count");
      return logits;
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
    std::cout << "All ConvNeXt V2 checks passed.\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
