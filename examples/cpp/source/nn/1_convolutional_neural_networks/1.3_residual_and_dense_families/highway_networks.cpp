// CMD: just run-nn highway_networks
// Highway networks: learned transform gates and a coupled carry path.
//
// Port of tflow's highway_networks. Implement both dense and convolutional
// highway layers from https://arxiv.org/abs/1505.00387. Check a signed
// two-channel fixture against independent output and full Jacobian
// calculations, including the gate's input-dependent derivative; check every
// parameter derivative and an input finite difference. Verify identical
// mathematics for a 1x1 convolution. Explicit zero/one gate limits recover
// identity/transformation respectively.
//
// A controlled 100-layer scalar example uses H(x)=.5*x and constant gates.
// Compare exact forward/input-gradient attenuation for plain layers and
// highway layers with biases zero and -3. This explains the carry mechanism,
// not a guarantee about arbitrary initialized or trained deep networks.
//
// Train a 20-hidden-layer MNIST classifier: a 784->50 ReLU projection, nineteen
// 50-wide highway layers with gate biases -2, and ten class logits. The model
// has 136,660 parameters. Use twelve epochs, Adam(.001), kernel L2(.0001),
// the shared 55k/5k split and validation-selected checkpoint; evaluate all
// 10k test images. This is an adaptation of the paper's depth-20 experiment,
// without its hyperparameter search or claim of reproducing its comparison.
// MNIST is read from the tflow dataset cache (common/paths.hpp) or data=.
// Options: data=PATH epochs=12 batch=256 lr=.001 threads=N.

#include "common/dataset_assets.hpp"
#include <array>
#include <cmath>
#include <iostream>

#include "common/checks.hpp"
#include "common/highway.hpp"
#include "common/mnist.hpp"

namespace nn = datamunge::nn;

namespace {

void GateChecks(bool convolutional) {
  nn::ParameterStore p(17);
  const nn::Shape    shape = convolutional ? nn::Shape({1, 1, 1, 2}) : nn::Shape({1, 2});
  std::vector<float> xv    = {.8f, -.5f};
  auto               build = [&](const nn::Tensor& input) {
    return convolutional ? highway::HighwayConv2D(p, "layer", input, 1, -1)
                                       : highway::HighwayDense(p, "layer", input, -1);
  };
  {
    nn::NoGradGuard guard;
    build(nn::from_vector(xv, shape));
  }
  const std::vector<float> wh = {.7f, -.2f, .3f, .8f}, bh = {.2f, .1f}, wt = {.4f, -.1f, -.2f, .3f},
                           bt                  = {-1, -1};
  const std::vector<std::vector<float>> values = {wh, bh, wt, bt};
  check::Require(p.count() == 12 && p.size() == 4, "highway fixture parameter count");
  for (std::size_t i = 0; i < p.size(); ++i) {
    const auto& name = p.names()[i];
    p.set(name, nn::from_vector(values[i], p.at(name).shape()));
  }
  nn::Tensor x = nn::from_vector(xv, shape);
  x.requires_grad_(true);
  const auto net       = build(x);
  const auto closed    = highway::HighwayMix(x, net.transformed, nn::zeros({}));
  const auto open      = highway::HighwayMix(x, net.transformed, nn::full({}, 1.0));
  const auto loss      = nn::sum(net.output);
  auto       variables = p.parameters();
  variables.insert(variables.begin(), x);
  const auto                      gradients = nn::grad(loss, variables);
  std::vector<std::vector<float>> out       = {
      net.output.to_vector<float>(),
      net.gate.to_vector<float>(),
      closed.to_vector<float>(),
      open.to_vector<float>()};
  for (const auto& g : gradients)
    out.push_back(g.to_vector<float>());
  std::array<double, 2> h{}, gate{}, dh{}, dt{}, dx{};
  for (int j = 0; j < 2; ++j) {
    double pre = bh[j], logit = bt[j];
    for (int k = 0; k < 2; ++k) {
      pre += xv[k] * wh[2 * k + j];
      logit += xv[k] * wt[2 * k + j];
    }
    h[j]    = std::max(0., pre);
    gate[j] = 1 / (1 + std::exp(-logit));
    dh[j]   = gate[j] * (pre > 0);
    dt[j]   = (h[j] - xv[j]) * gate[j] * (1 - gate[j]);
    check::Near(out[0][j], h[j] * gate[j] + xv[j] * (1 - gate[j]), 1e-6, "highway output");
    check::Near(out[1][j], gate[j], 1e-6, "transform gate");
    check::Near(out[2][j], xv[j], 0, "closed-gate identity limit");
    check::Near(out[3][j], h[j], 1e-6, "open-gate transformation limit");
    check::Near(out[6][j], dh[j], 1e-6, "transform bias derivative");
    check::Near(out[8][j], dt[j], 1e-6, "gate bias derivative");
  }
  nn::NoGradGuard guard;
  for (int k = 0; k < 2; ++k) {
    dx[k] = 1 - gate[k];
    for (int j = 0; j < 2; ++j) {
      dx[k] += dh[j] * wh[2 * k + j] + dt[j] * wt[2 * k + j];
      check::Near(out[5][2 * k + j], xv[k] * dh[j], 1e-6, "transform weight derivative");
      check::Near(out[7][2 * k + j], xv[k] * dt[j], 1e-6, "gate weight derivative");
    }
    check::Near(
        out[4][k], dx[k], 1e-6, "input derivative includes both transformation and gate paths");
    const float step     = .001f;
    auto        evaluate = [&](float value) {
      auto shifted = xv;
      shifted[k]   = value;
      return nn::sum(build(nn::from_vector(shifted, shape)).output).item<float>();
    };
    const double plus = evaluate(xv[k] + step), minus = evaluate(xv[k] - step);
    check::Near((plus - minus) / (2 * step), dx[k], 1e-4, "highway finite difference");
  }
  std::cout
      << "  " << (convolutional ? "1x1 convolutional" : "dense")
      << " highway: gate limits, full analytic derivatives and input finite differences passed\n";
}

void DepthChecks() {
  nn::ParameterStore p;
  nn::Tensor         input = nn::full({1, 1}, 1.0);
  input.requires_grad_(true);
  std::vector<nn::Tensor> outputs;
  std::vector<double>     expected;
  nn::Tensor              plain = input;
  for (int layer = 0; layer < 100; ++layer)
    plain = plain * .5;
  outputs.push_back(plain);
  expected.push_back(std::pow(.5, 100));
  for (float bias : {0.f, -3.f}) {
    nn::Tensor h = input;
    for (int layer = 0; layer < 100; ++layer) {
      const auto name = (bias == 0 ? "neutral" : "carry") + std::to_string(layer);
      p.set(name + "/transform/W", nn::full({1, 1}, .5));
      p.set(name + "/gate/W", nn::zeros({1, 1}));
      h = highway::HighwayDense(p, name, h, bias).output;
    }
    outputs.push_back(h);
    expected.push_back(std::pow(1 - .5 / (1 + std::exp(-double(bias))), 100));
  }
  for (int i = 0; i < 3; ++i) {
    const float value      = outputs[i].item<float>(),
                derivative = nn::grad(outputs[i], {input})[0].item<float>();
    check::Near(value / expected[i], 1, 2e-5, "100-layer exact attenuation");
    check::Near(derivative / expected[i], 1, 2e-5, "100-layer exact input gradient");
    std::cout << "  100-layer "
              << (i == 0   ? "plain"
                  : i == 1 ? "gate bias 0"
                           : "gate bias -3")
              << ": output and input derivative=" << value << std::endl;
  }
}

} // namespace

int main(int argc, char** argv) {
  try {
    for (int i = 1; i < argc; ++i) // threads= also applies to the checks
      if (std::string(argv[i]).rfind("threads=", 0) == 0)
        nn::set_num_threads(std::stoi(std::string(argv[i]).substr(8)));
    GateChecks(false);
    GateChecks(true);
    DepthChecks();
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
  const mnist::LogitsBuilder network = [](nn::ParameterStore& p, const nn::Tensor& input, bool) {
    nn::DenseOptions projection;
    projection.activation = nn::Activation::Relu;
    nn::Tensor h          = nn::dense(p, "projection", input, 50, projection);
    for (int layer = 0; layer < 19; ++layer)
      h = highway::HighwayDense(p, "highway" + std::to_string(layer), h, -2).output;
    const auto logits = nn::dense(p, "output", h, 10);
    check::Require(p.count() == 136660, "20-hidden-layer highway parameter count");
    return logits;
  };
  auto experiment_builder = network;
  {
    // Dataset setup, batching, fitting and checkpoint selection are explicit here.
    using namespace mnist;
    const std::string               name                  = "highway_mnist";
    const std::vector<std::int64_t> hidden                = {};
    const int                       default_epochs        = 12;
    const double                    default_rate          = .001;
    const double                    minimum_test_accuracy = .95;
    LogitsBuilder                   builder               = experiment_builder;

    try {
      std::filesystem::path dir    = paths::Dataset("mnist");
      int                   epochs = default_epochs, batch = 256;
      double                rate = default_rate;
      for (int i = 1; i < argc; ++i) {
        const std::string arg   = argv[i];
        const auto        equal = arg.find('=');
        if (equal == std::string::npos)
          throw std::invalid_argument("expected key=value argument");
        const auto key = arg.substr(0, equal), value = arg.substr(equal + 1);
        if (key == "data")
          dir = value;
        else if (key == "epochs")
          epochs = std::stoi(value);
        else if (key == "batch")
          batch = std::stoi(value);
        else if (key == "lr")
          rate = std::stod(value);
        else if (key == "threads")
          nn::set_num_threads(std::stoi(value));
        else
          throw std::invalid_argument("unknown argument: " + key);
      }
      check::Require(
          epochs > 0 && batch > 0 && std::isfinite(rate) && rate > 0,
          "positive epochs, batch and learning rate required");
      epochs = example_runtime::Limit(epochs, 1);
      batch  = example_runtime::Limit(batch, 16);
      example_runtime::Describe();
      dataset_assets::EnsureMnist(dir.string());
      const auto train = dataset_assets::LoadMnist(dir.string(), "train");
      const auto test  = dataset_assets::LoadMnist(dir.string(), "test");
      check::Require(
          train.images.shape() == nn::Shape({60000, 28, 28, 1}) &&
              test.images.shape() == nn::Shape({10000, 28, 28, 1}),
          "official MNIST dimensions required");
      const nn::Tensor train_x = nn::reshape(train.images, {60000, 784}),
                       test_x  = nn::reshape(test.images, {10000, 784});

      std::vector<std::size_t> order(60000);
      std::iota(order.begin(), order.end(), 0);
      std::mt19937_64 split_rng(2026);
      std::shuffle(order.begin(), order.end(), split_rng);
      std::vector<std::int64_t> fit_rows(order.begin(), order.begin() + 55000),
          validation_rows(order.begin() + 55000, order.end());
      fit_rows.resize(example_runtime::Limit<std::size_t>(fit_rows.size(), 64));
      validation_rows.resize(example_runtime::Limit<std::size_t>(validation_rows.size(), 32));
      std::vector<std::int64_t> test_rows(example_runtime::Limit<std::size_t>(10000, 32));
      std::iota(test_rows.begin(), test_rows.end(), 0);

      if (!builder)
        builder = [hidden](nn::ParameterStore& store, const nn::Tensor& x, bool) {
          return nn::dense(store, "output", nn::mlp(store, "hidden", x, hidden), 10);
        };
      const double       l2 = 1e-4;
      nn::ParameterStore store(17);
      {
        nn::NoGradGuard guard; // create every parameter before building the optimizer
        builder(store, nn::slice_axis(train_x, 0, 0, 1), true);
      }
      nn::OptimizerConfig config;
      config.kind = nn::OptimizerKind::Adam;
      nn::Optimizer           optimizer(store, rate, config);
      std::vector<nn::Tensor> weights;
      for (const auto& p : store.parameters())
        if (p.dim() >= 2)
          weights.push_back(p);

      const auto    model_dir  = paths::OutputDir("models/" + name);
      const auto    checkpoint = (model_dir / "best.npz").string();
      std::ofstream history(model_dir / "history.tsv");
      check::Require(bool(history), "open training history");
      history << "epoch\tvalidation_loss\tvalidation_accuracy\n";
      std::cout << "== " << name << ": " << fit_rows.size() << " fit / " << validation_rows.size()
                << " validation / " << test_rows.size() << " test\n"
                << "  parameters=" << store.count() << "; batch=" << batch << "; lr=" << rate
                << '\n';

      nn::BatchOrder batches(fit_rows.size(), static_cast<std::size_t>(batch), 29);
      double         best       = std::numeric_limits<double>::infinity();
      int            best_epoch = -1;
      const auto     started    = std::chrono::steady_clock::now();
      for (int epoch = 1; epoch <= epochs; ++epoch) {
        const auto epoch_start = std::chrono::steady_clock::now();
        batches.begin_epoch();
        std::size_t samples = 0;
        for (;;) {
          auto ids = batches.next();
          if (ids.empty())
            break;
          samples += ids.size();
          for (auto& index : ids)
            index = fit_rows[static_cast<std::size_t>(index)];
          const nn::Tensor x = nn::take_rows(train_x, ids), y = nn::take_rows(train.labels, ids);
          nn::Tensor       objective = nn::sparse_softmax_cross_entropy(builder(store, x, true), y);
          if (l2 > 0) {
            nn::Tensor penalty = nn::sum(nn::square(weights[0]));
            for (std::size_t i = 1; i < weights.size(); ++i)
              penalty = penalty + nn::sum(nn::square(weights[i]));
            objective = objective + 0.5 * l2 * penalty;
          }
          optimizer.minimize(objective);
        }
        check::Require(
            samples == fit_rows.size(), "each selected training row used once per epoch");
        const auto   validation = Evaluate(store, builder, train_x, train.labels, validation_rows);
        const double loss = validation.loss.value(), accuracy = validation.accuracy.value();
        if (loss < best) {
          best       = loss;
          best_epoch = epoch;
          store.save(checkpoint);
        }
        history << epoch << '\t' << loss << '\t' << accuracy << '\n';
        const double seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - epoch_start).count();
        std::cout << "  epoch " << epoch << ": validation NLL=" << loss << ", accuracy=" << accuracy
                  << " (" << std::setprecision(3) << seconds << std::setprecision(6) << " s)"
                  << std::endl;
      }
      const double train_seconds =
          std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
      store.load(checkpoint);
      const auto restored = Evaluate(store, builder, train_x, train.labels, validation_rows);
      check::Near(
          restored.loss.value(), best, 1e-7, "best checkpoint matches selected validation loss");
      const auto result = Evaluate(store, builder, test_x, test.labels, test_rows);
      std::cout << "  selected epoch=" << best_epoch << "; test NLL=" << result.loss.value()
                << "; accuracy=" << result.accuracy.value() << "\n  test confusion (truth rows):\n";
      std::uint64_t total = 0, correct = 0;
      const auto&   cells = result.confusion.cells();
      for (std::size_t i = 0; i < 10; ++i) {
        std::cout << "   ";
        for (std::size_t j = 0; j < 10; ++j) {
          std::cout << ' ' << std::setw(4) << cells[i][j];
          total += cells[i][j];
        }
        correct += cells[i][i];
        std::cout << '\n';
      }
      check::Require(total == test_rows.size(), "all selected test images evaluated");
      check::Near(
          result.accuracy.value(),
          double(correct) / test_rows.size(),
          1e-7,
          "confusion matches accuracy");
      check::TrainingQuality(
          result.accuracy.value() >= minimum_test_accuracy,
          "classifier did not reach demonstration accuracy threshold");
      check::Require(bool(history), "write training history");
      std::printf("  training time %.1f s (%.2f s/epoch)\n", train_seconds, train_seconds / epochs);
      std::cout << "  best checkpoint and history: " << model_dir.string() << "\nAll " << name
                << " checks passed.\n";
      return 0;
    } catch (const std::exception& e) {
      std::cerr << e.what() << '\n';
      return 1;
    }
  }
}
