// CMD: just run-nn spatial_transformer
// Spatial transformer networks: learned, differentiable affine warping of the input.
//
// Port of tflow's spatial_transformer. Preparation (in ~/projects/tflow_examples):
// dm::datasets::fetch("mnist").
//
// Check the grid generator and bilinear sampler (nn::affine_grid and
// nn::grid_sample with align_corners): the identity transform returns the
// input, a one-pixel translation shifts it (zeros enter at the border), and a
// 90-degree rotation equals the rotated image, all exactly; derivatives with
// respect to theta match central differences. A new transformer (zero
// localization weights, identity bias) is the identity.
// Build the paper's rotated-translated-scaled MNIST: each digit rotated by
// up to 45 degrees, scaled by 0.7-1.2 and placed at random in a 42x42
// canvas (deterministic seeds, the same std::mt19937_64 draws as tflow, so
// the distorted images are identical; the test set is distorted independently).
// https://arxiv.org/abs/1506.02025
//
// Train the same CNN (5x5 conv 32, pool, 5x5 conv 64, pool, dense 128, 10)
// with and without a spatial transformer in front on distorted MNIST: 12k fit
// / 5k validation, twelve epochs, no further augmentation, Adam(.001), kernel
// L2(.0001) and the best-validation checkpoint before evaluating the 10k
// distorted test images. Parameter initialisation uses nn's generator, so
// trajectories match tflow only statistically. The transformer's training is
// seed-sensitive in both frameworks: from the identity start, gradients that
// favour zooming out (zeros enter at the border) can drive theta off the
// canvas, after which the image is blank and stays so. With TF, seeds 20 and
// 21 of the classifier store collapse this way (17, the recorded seed, dips to
// 30% after one epoch and recovers to 95%); here the library's symmetric
// slope at integer coordinates (see common/stn.hpp) keeps seed 17 stable.
// Options: data=PATH epochs=12 batch=128 lr=.001 fit=12000 threads=N seed=17
// (seed of the parameter stores; the transformer is seed-sensitive, see above).

#include "common/dataset_assets.hpp"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <random>

#include "common/checks.hpp"
#include "common/image_training.hpp"
#include "common/paths.hpp"
#include "common/stn.hpp"

namespace nn = datamunge::nn;

namespace {

nn::Tensor Pattern(const nn::Shape& shape, float scale, float phase) {
  std::vector<float> v(static_cast<std::size_t>(nn::shape_numel(shape)));
  for (std::size_t i = 0; i < v.size(); ++i)
    v[i] = scale * std::sin(.37f * i + phase);
  return nn::from_vector(v, shape);
}

void SamplerChecks() {
  const int        size    = 9;
  const nn::Tensor value   = Pattern({1, size, size, 2}, 1, .2f);
  const nn::Tensor weights = Pattern({1, size, size, 2}, 1, .4f);
  auto theta_of            = [](const std::vector<float>& t) { return nn::from_vector(t, {1, 6}); };
  auto sample              = [&](const nn::Tensor& theta) {
    return stn::AffineSample(value, theta, size, size);
  };
  auto loss_of  = [&](const nn::Tensor& theta) { return nn::sum(sample(theta) * weights).item(); };
  const auto v  = value.to_vector<float>();
  auto       at = [&](int r, int c, int ch) {
    return r < 0 || r >= size || c < 0 || c >= size ? 0. : double(v[(r * size + c) * 2 + ch]);
  };
  auto compare = [&](const nn::Tensor& t, auto expected) {
    const auto got = t.to_vector<float>();
    double     e   = 0;
    for (int r = 0; r < size; ++r)
      for (int c = 0; c < size; ++c)
        for (int ch = 0; ch < 2; ++ch)
          e = std::max(e, std::abs(got[(r * size + c) * 2 + ch] - expected(r, c, ch)));
    return e;
  };
  const double identity = compare(
      sample(theta_of({1, 0, 0, 0, 1, 0})), [&](int r, int c, int ch) { return at(r, c, ch); });
  const double shift = compare(
      sample(theta_of({1, 0, 2.f / (size - 1), 0, 1, 0})),
      [&](int r, int c, int ch) { return at(r, c + 1, ch); });
  // x_s = -y_t, y_s = x_t: output (r, c) reads input row c, column size-1-r.
  const double rotation = compare(sample(theta_of({0, -1, 0, 1, 0, 0})), [&](int r, int c, int ch) {
    return at(c, size - 1 - r, ch);
  });
  const std::vector<float> generic = {.93f, .11f, .047f, -.083f, 1.06f, -.031f};
  nn::Tensor               theta   = theta_of(generic).requires_grad_(true);
  const auto symbolic   = nn::grad(nn::sum(sample(theta) * weights), {theta})[0].to_vector<float>();
  double     derivative = 0;
  for (int i = 0; i < 6; ++i) {
    // A small step keeps most of the 81 samples on one bilinear piece; float32
    // rounding then dominates the difference.
    auto        plus = generic, minus = generic;
    const float h = 2e-4f;
    plus[i] += h;
    minus[i] -= h;
    const double numeric = (static_cast<float>(loss_of(theta_of(plus))) -
                            static_cast<float>(loss_of(theta_of(minus)))) /
                           (2 * h);
    derivative =
        std::max(derivative, std::abs(numeric - symbolic[i]) / std::max(1., std::abs(numeric)));
  }
  check::Near(identity, 0, 1e-6, "identity transform");
  check::Near(shift, 0, 1e-6, "one-pixel translation");
  check::Near(rotation, 0, 1e-6, "90-degree rotation");
  check::Near(derivative, 0, 1e-2, "theta derivatives match central differences");
  std::cout << "  sampler on 9x9: identity error " << identity << ", one-pixel shift " << shift
            << ", 90-degree rotation " << rotation
            << "; theta derivatives vs central differences (relative) " << derivative << std::endl;
  // A new transformer is the identity.
  nn::ParameterStore p(17);
  const nn::Tensor   image  = Pattern({2, 42, 42, 1}, 1, .3f);
  const auto         out    = stn::SpatialTransformer(p, "stn", image).output;
  double             change = nn::max(nn::abs(out - image)).item();
  check::Near(change, 0, 1e-5, "a new transformer is the identity");
  std::cout << "  new transformer (" << p.count()
            << " localization parameters) passes 42x42 inputs through unchanged (error " << change
            << ")" << std::endl;
}

// Rotated, scaled and translated digits on a 42x42 canvas (bilinear, host),
// quantised to uint8 like tflow's dataset and returned in [0, 1].
nn::ImageDataset Distort(const nn::ImageDataset& mnist, std::uint64_t seed) {
  const std::int64_t        n = mnist.images.shape()[0], size = 42;
  std::vector<std::uint8_t> source(static_cast<std::size_t>(n * 28 * 28));
  {
    const auto pixels = mnist.images.to_vector<float>();
    for (std::size_t i = 0; i < source.size(); ++i)
      source[i] = static_cast<std::uint8_t>(std::lround(pixels[i] * 255.f));
  }
  std::vector<float>                     target(static_cast<std::size_t>(n * size * size));
  std::mt19937_64                        random(seed);
  std::uniform_real_distribution<double> angle(-M_PI / 4, M_PI / 4), zoom(.7, 1.2), shift(-1, 1);
  for (std::int64_t k = 0; k < n; ++k) {
    const double a = angle(random), z = zoom(random),
                 extent       = 14 * z * (std::abs(std::cos(a)) + std::abs(std::sin(a)));
    const double room         = std::max(0., (size - 1) / 2. - extent),
                 cy           = (size - 1) / 2. + room * shift(random),
                 cx           = (size - 1) / 2. + room * shift(random);
    const std::uint8_t* digit = source.data() + k * 28 * 28;
    for (int r = 0; r < size; ++r)
      for (int c = 0; c < size; ++c) {
        // Inverse map: canvas pixel -> digit coordinates.
        const double dy = (r - cy) / z, dx = (c - cx) / z;
        const double sy = std::cos(a) * dy + std::sin(a) * dx + 13.5,
                     sx = -std::sin(a) * dy + std::cos(a) * dx + 13.5;
        const int y0 = int(std::floor(sy)), x0 = int(std::floor(sx));
        double    v = 0;
        for (int i = 0; i <= 1; ++i)
          for (int j = 0; j <= 1; ++j) {
            const int yy = y0 + i, xx = x0 + j;
            if (yy < 0 || yy > 27 || xx < 0 || xx > 27)
              continue;
            v +=
                (i ? sy - y0 : 1 - (sy - y0)) * (j ? sx - x0 : 1 - (sx - x0)) * digit[yy * 28 + xx];
          }
        target[static_cast<std::size_t>((k * size + r) * size + c)] =
            std::clamp(std::lround(v), 0L, 255L) / 255.f;
      }
  }
  return {nn::from_vector(target, {n, size, size, 1}), mnist.labels};
}

} // namespace

int main(int argc, char** argv) {
  try {
    SamplerChecks();
    std::filesystem::path               directory = paths::Dataset("mnist");
    image_training::ImageTrainingConfig base;
    base.maximum_fit_rows      = 12000;
    base.epochs                = 12;
    base.augment               = false;
    base.minimum_test_accuracy = .8;
    for (int i = 1; i < argc; ++i) {
      const std::string argument = argv[i];
      const auto        equal    = argument.find('=');
      if (equal == std::string::npos)
        throw std::invalid_argument("expected key=value option");
      const auto key = argument.substr(0, equal), value = argument.substr(equal + 1);
      if (key == "data")
        directory = value;
      else if (key == "epochs")
        base.epochs = std::stoi(value);
      else if (key == "batch")
        base.batch = std::stoi(value);
      else if (key == "lr")
        base.learning_rate = std::stod(value);
      else if (key == "fit")
        base.maximum_fit_rows = std::stoi(value);
      else if (key == "threads")
        nn::set_num_threads(std::stoi(value));
      else if (key == "seed")
        base.seed = std::stoull(value);
      else
        throw std::invalid_argument("unknown option " + key);
    }
    dataset_assets::EnsureMnist(directory.string());
    const auto development = Distort(dataset_assets::LoadMnist(directory.string(), "train"), 101);
    const auto test        = Distort(dataset_assets::LoadMnist(directory.string(), "test"), 202);
    std::cout << "  distorted MNIST: " << development.images.shape()[0] << " development and "
              << test.images.shape()[0] << " test images of 42x42" << std::endl;
    for (bool transformer : {false, true}) {
      auto cfg = base;
      cfg.name = transformer ? "st_cnn_rts_mnist" : "cnn_rts_mnist";
      const auto builder =
          [transformer](nn::ParameterStore& p, nn::ParameterStore&, const nn::Tensor& input, bool) {
            nn::Tensor image = nn::reshape(input, {-1, 42, 42, 1});
            if (transformer)
              image = stn::SpatialTransformer(p, "stn", image).output;
            nn::ConvLayerOptions cfg;
            cfg.init       = nn::init::he_uniform();
            cfg.activation = nn::Activation::Relu;
            auto h =
                nn::max_pool2d(nn::conv2d(p, "conv1", image, 32, 5, cfg), 2, 2, nn::Padding::Same);
            h      = nn::max_pool2d(nn::conv2d(p, "conv2", h, 64, 5, cfg), 2, 2, nn::Padding::Same);
            auto f = nn::relu(nn::dense(p, "fc1", nn::reshape(h, {-1, 11 * 11 * 64}), 128));
            return nn::dense(p, "output", f, 10);
          };
      {
        using namespace image_training;
        bool explicit_epochs = false;
        for (int i = 1; i < argc; ++i)
          explicit_epochs = explicit_epochs || std::string(argv[i]).rfind("epochs=", 0) == 0;
        const auto& model = builder;
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
    std::cout << "All spatial transformer checks passed.\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
