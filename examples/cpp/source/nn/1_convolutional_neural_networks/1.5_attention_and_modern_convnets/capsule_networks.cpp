// CMD: just run-nn capsule_networks
// Capsule networks: squashed vector capsules and dynamic routing by agreement.
//
// Port of tflow's capsule_networks. Preparation (in ~/projects/tflow_examples):
// dm::datasets::fetch("mnist") (found via
// common/paths.hpp, or data=).
//
// Check squash against its formula (direction kept, length |s|^2/(1+|s|^2)),
// three routing iterations against a host implementation for a small
// fixture (6 input and 3 output capsules), that couplings are distributions
// over output capsules that concentrate as agreement accumulates, and the
// margin loss. Count the MNIST CapsNet: 8,215,568 learned parameters, or
// 6,804,224 without the reconstruction decoder (the paper's 8.2M and 6.8M),
// with 1152 8-D primary and 10 16-D digit capsules; check every derivative.
// https://arxiv.org/abs/1710.09829
//
// Train it on MNIST with margin loss plus the 0.0005-weighted reconstruction
// loss (decoder masked by the true class). Use 12k fit / 5k validation from
// the official training images, twelve epochs, no augmentation, Adam(.001),
// kernel L2(.0001) and the checkpoint with the lowest validation NLL of the
// normalized capsule lengths, then evaluate the official 10k test images.
// The paper trains on all 60k images with 2-pixel shifts for many more
// epochs; this is not a reproduction of its 0.25% error. Weights come from
// nn's generator, so trained numbers match TF only statistically.
// Options: data=PATH epochs=12 batch=128 lr=.001 fit=12000 threads=N seed=17.

#include "common/dataset_assets.hpp"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <limits>

#include "common/capsnet.hpp"
#include "common/checks.hpp"
#include "common/image_training.hpp"
#include "common/mobile_checks.hpp"
#include "common/paths.hpp"

namespace nn = datamunge::nn;
using mobile_checks::Pattern;

namespace {

void RoutingChecks() {
  const int        inputs = 6, outputs = 3, in_dim = 2, out_dim = 3, iterations = 3;
  const nn::Tensor u        = Pattern({1, inputs, in_dim}, 1.3f, .2f),
                   w        = Pattern({inputs, outputs * out_dim, in_dim}, 1, .7f);
  const nn::Tensor label    = nn::from_vector(std::vector<float>{0, 1, 0}, {1, outputs});
  const nn::Tensor squashed = capsnet::Squash(u);
  const auto routing = capsnet::DynamicRouting(u, w, inputs, outputs, in_dim, out_dim, iterations);
  const double margin =
      capsnet::MarginLoss(capsnet::CapsuleLengths(routing.capsules), label).item();
  const auto uv = u.to_vector<float>(), wv = w.to_vector<float>(), sq = squashed.to_vector<float>();
  const auto caps = routing.capsules.to_vector<float>(),
             last = routing.history.back().to_vector<float>();
  // Squash.
  double squash = 0;
  for (int i = 0; i < inputs; ++i) {
    const double a = uv[2 * i], b = uv[2 * i + 1], n2 = a * a + b * b,
                 scale = n2 / (1 + n2) / std::sqrt(n2);
    squash =
        std::max({squash, std::abs(sq[2 * i] - scale * a), std::abs(sq[2 * i + 1] - scale * b)});
  }
  // Host routing.
  std::vector<std::vector<std::vector<double>>> votes(
      inputs, std::vector<std::vector<double>>(outputs, std::vector<double>(out_dim)));
  for (int i = 0; i < inputs; ++i)
    for (int j = 0; j < outputs; ++j)
      for (int d = 0; d < out_dim; ++d)
        for (int k = 0; k < in_dim; ++k)
          votes[i][j][d] +=
              wv[(i * outputs * out_dim + j * out_dim + d) * in_dim + k] * uv[i * in_dim + k];
  std::vector<std::vector<double>> b(inputs, std::vector<double>(outputs, 0)), c, v;
  std::vector<double>              concentration;
  for (int t = 0; t < iterations; ++t) {
    c.assign(inputs, std::vector<double>(outputs));
    double peak = 0;
    for (int i = 0; i < inputs; ++i) {
      double total = 0;
      for (int j = 0; j < outputs; ++j)
        total += std::exp(b[i][j]);
      for (int j = 0; j < outputs; ++j)
        c[i][j] = std::exp(b[i][j]) / total;
      peak += *std::max_element(c[i].begin(), c[i].end()) / inputs;
    }
    concentration.push_back(peak);
    v.assign(outputs, std::vector<double>(out_dim, 0));
    for (int j = 0; j < outputs; ++j) {
      double n2 = 0;
      for (int d = 0; d < out_dim; ++d) {
        for (int i = 0; i < inputs; ++i)
          v[j][d] += c[i][j] * votes[i][j][d];
        n2 += v[j][d] * v[j][d];
      }
      for (int d = 0; d < out_dim; ++d)
        v[j][d] *= n2 / (1 + n2) / std::sqrt(n2 + 1e-9);
    }
    for (int i = 0; i < inputs; ++i)
      for (int j = 0; j < outputs; ++j)
        for (int d = 0; d < out_dim; ++d)
          b[i][j] += votes[i][j][d] * v[j][d];
  }
  double routing_error = 0, coupling_sum = 0, loss = 0;
  for (int j = 0; j < outputs; ++j) {
    double len = 0;
    for (int d = 0; d < out_dim; ++d) {
      routing_error = std::max(routing_error, std::abs(caps[j * out_dim + d] - v[j][d]));
      len += v[j][d] * v[j][d];
    }
    len = std::sqrt(len);
    loss += j == 1 ? std::pow(std::max(0., .9 - len), 2) : .5 * std::pow(std::max(0., len - .1), 2);
  }
  for (int i = 0; i < inputs; ++i) {
    double total = 0;
    for (int j = 0; j < outputs; ++j)
      total += last[i * outputs + j];
    coupling_sum = std::max(coupling_sum, std::abs(total - 1));
  }
  check::Near(squash, 0, 1e-6, "squash");
  check::Near(routing_error, 0, 1e-5, "dynamic routing");
  check::Near(coupling_sum, 0, 1e-6, "couplings are distributions over output capsules");
  check::Near(margin, loss, 1e-5, "margin loss");
  check::Require(
      concentration[0] < concentration[1] && concentration[1] < concentration[2],
      "routing by agreement concentrates couplings");
  std::cout << "  squash error " << squash
            << "; 3-iteration routing (6 -> 3 capsules) matches the host (error " << routing_error
            << "); mean largest coupling " << concentration[0] << " -> " << concentration[1]
            << " -> " << concentration[2] << "; margin loss " << margin << " (host " << loss << ")"
            << std::endl;
}

/// Margin loss plus 0.0005 x the mean summed squared reconstruction error
/// (decoder masked by `one_hot`); `image` is in [-1, 1].
nn::Tensor CapsuleLoss(
    const capsnet::CapsNetOutputs& net,
    const nn::Tensor&              image,
    const nn::Tensor&              one_hot) {
  const nn::Tensor target  = nn::reshape((image + 1.0) / 2.0, {-1, 784}); // pixels in [0, 1]
  const nn::Tensor squared = nn::sum(nn::square(net.reconstruct(one_hot) - target), {1});
  return capsnet::MarginLoss(net.lengths, one_hot) + nn::mean(squared) * 5e-4;
}

void NetworkChecks() {
  nn::ParameterStore p(17);
  const nn::Tensor   image = Pattern({2, 28, 28, 1}, 1, 0);
  const auto         net   = capsnet::CapsNet(p, image);
  check::Require(p.count() == 8215568, "CapsNet parameter count");
  std::int64_t decoder = 0;
  for (const auto& name : p.names())
    if (name.rfind("decoder/", 0) == 0)
      decoder += p.at(name).numel();
  check::Require(p.count() - decoder == 6804224, "CapsNet without decoder");
  std::vector<float> label(20, 0.f);
  label[3] = label[10 + 7] = 1;
  const nn::Tensor loss    = CapsuleLoss(net, image, nn::from_vector(label, {2, 10}));
  const auto       norms   = mobile_checks::GradientNorms(p, loss);
  check::Require(
      net.primary.shape() == nn::Shape({2, 1152, 8}) &&
          net.digits.shape() == nn::Shape({2, 10, 16}),
      "capsule shapes");
  double      minimum = std::numeric_limits<double>::infinity(), maximum = 0;
  std::string smallest;
  for (std::size_t i = 0; i < norms.size(); ++i) {
    check::Require(std::isfinite(norms[i]) && norms[i] > 0, "finite nonzero CapsNet derivative");
    if (norms[i] < minimum) {
      minimum  = norms[i];
      smallest = p.names()[i];
    }
    maximum = std::max(maximum, norms[i]);
  }
  std::cout << "  CapsNet: " << p.count() << " parameters (" << p.count() - decoder
            << " without the decoder); 1152 x 8-D primary, 10 x 16-D digit capsules; loss "
            << loss.item() << ", derivatives in [" << minimum << ',' << maximum << "] (smallest "
            << smallest << ")" << std::endl;
}

} // namespace

int main(int argc, char** argv) {
  try {
    std::filesystem::path               directory = paths::Dataset("mnist");
    image_training::ImageTrainingConfig cfg;
    cfg.name                  = "capsnet_mnist";
    cfg.maximum_fit_rows      = 12000;
    cfg.epochs                = 12;
    cfg.augment               = false;
    cfg.minimum_test_accuracy = .97;
    for (int i = 1; i < argc; ++i) {
      const std::string argument = argv[i];
      const auto        equal    = argument.find('=');
      if (equal == std::string::npos)
        throw std::invalid_argument("expected key=value option");
      const auto key = argument.substr(0, equal), value = argument.substr(equal + 1);
      if (key == "data")
        directory = value;
      else if (key == "epochs")
        cfg.epochs = std::stoi(value);
      else if (key == "batch")
        cfg.batch = std::stoi(value);
      else if (key == "lr")
        cfg.learning_rate = std::stod(value);
      else if (key == "fit")
        cfg.maximum_fit_rows = std::stoi(value);
      else if (key == "threads")
        nn::set_num_threads(std::stoi(value));
      else if (key == "seed")
        cfg.seed = std::stoull(value);
      else
        throw std::invalid_argument("unknown option " + key);
    }
    RoutingChecks();
    NetworkChecks();
    dataset_assets::EnsureMnist(directory.string());
    const auto                          development = dataset_assets::LoadMnist(directory.string(), "train");
    const auto                          test        = dataset_assets::LoadMnist(directory.string(), "test");
    const image_training::LogitsBuilder model =
        [](nn::ParameterStore& p, nn::ParameterStore& state, const nn::Tensor& input, bool) {
          const auto net = capsnet::CapsNet(p, nn::reshape(input, {-1, 28, 28, 1}));
          check::Require(p.count() == 8215568 && state.count() == 0, "CapsNet parameter count");
          return net.logits;
        };
    {
      using namespace image_training;
      bool explicit_epochs = false;
      for (int i = 1; i < argc; ++i)
        explicit_epochs = explicit_epochs || std::string(argv[i]).rfind("epochs=", 0) == 0;
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
          const auto image  = nn::reshape(x, {-1, 28, 28, 1});
          const auto net    = capsnet::CapsNet(params, image);
          const auto labels = nn::one_hot(y, 10, 1.0, 0.0, net.lengths.dtype());
          const auto objective = WithL2(CapsuleLoss(net, image, labels), params, cfg.l2);
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
    std::cout << "All capsule network checks passed.\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
