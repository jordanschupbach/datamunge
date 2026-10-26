// CMD: just run-nn time_series_cnn
// Three-block temporal FCN:8/5/3 convolutions,BN/ReLU/global average/head.
// Real GunPoint hand-motion sequences,official test split and train-only
// validation;fit-only global normalization saved with all moving BN state.
// Full/compact architecture,independent all40-coordinate host gradient and
// even-kernel SAME/normalization/nearest-baseline checks;three epochs default.
//
// Port of tflow's time_series_cnn. Data: tflow's `just time-series-assets`
// (output/datasets/time_series/time_series.npz, found via paths::Dataset).
// Deviations: .npz checkpoints; tflow's GraphDef audit has no eager
// counterpart. Weights initialise with nn's generator, so trained numbers
// match TF statistically.
// Options: epochs=3 data=PATH checks_only=0|1 threads=4.
// Output: output/nn/models/time_series_cnn/.

#include <array>

#include "common/time_series.hpp"

// Experiment workflow from examples/nn/common/time_series.hpp.
namespace series {
using namespace ::series;
inline void TrainSeries(const std::filesystem::path& root, int epochs) {
  check::Require(epochs > 0, "positive short series epochs");
  const auto  data = LoadSeries(root);
  const auto& fit = data.pools[0];
  const auto& validation = data.pools[1];
  const auto& test = data.pools[2];
  SeriesModel model(data.config);
  const auto  directory = paths::OutputDir("models/time_series_cnn");
  const auto  prefix = (directory / "best.npz").string();
  model.CalibrateFit(fit);
  const double initial = model.Loss(fit);
  const Tensor untrained = model.Predict(test);
  const int    nonzero = sentence::RequireLiveGradients(
      model.parameters,
      nn::grad(sentence::SentenceCE(model.Forward(fit, BnMode::TrainNoUpdate).logits, fit.labels), model.parameters.parameters()),
      "time-series",
      [](const std::string& name) { return name.rfind("conv", 0) == 0 && name.substr(name.size() - 2) == "/b"; }, 1e-18);
  std::cout << "time_series_cnn:" << model.parameters.count()
            << " parameters/14 learned tensors+8 normalization/BN state,channels16/32/16,kernels8/5/3,GAP;" << epochs
            << " epochs,batch8,Adam.001;all14 gradients finite,nonzero=" << nonzero
            << ",pre-BN bias shift invariance allowed,initial calibrated frozen fitCE=" << initial << std::endl;
  std::ofstream history(directory / "history.tsv");
  history << "epoch\ttrain_ce\tvalidation_frozen_ce\n";
  nn::BatchOrder order(fit.size(), 8, 431);
  double         best = INFINITY;
  int            selected = 0;
  for (int epoch = 1; epoch <= epochs; ++epoch) {
    order.begin_epoch();
    double sum = 0;
    int    seen = 0;
    for (;;) {
      const auto rows = order.next();
      if (rows.empty()) break;
      // Batch-statistics BN graph (moving averages updated), CE, gradients and one Adam step.
      const auto   batch = SeriesRows(fit, rows);
      const Tensor loss = sentence::SentenceCE(model.Forward(batch, BnMode::Train).logits, batch.labels);
      model.optimizer->step(nn::grad(loss, model.parameters.parameters()));
      const double value = loss.item<double>();
      check::Require(std::isfinite(value) && value >= 0, "finite time-series CE");
      sum += rows.size() * value;
      seen += static_cast<int>(rows.size());
    }
    check::Require(seen == fit.size(), "every fitting signal once per epoch");
    model.CalibrateFit(fit);
    const double score = model.Loss(validation);
    if (score < best) {
      best = score;
      selected = epoch;
      model.Save(prefix);
    }
    history << epoch << '\t' << sum / seen << '\t' << score << '\n';
    history.flush();
    std::cout << " epoch " << epoch << ":trainCE=" << sum / seen << ",validationCalibratedCE=" << score << std::endl;
  }
  model.Restore(prefix);
  check::Near(model.Loss(validation), best, 1e-7, "restored validation score");
  const double learned = model.Loss(fit);
  check::Require(learned < initial, "short-run fixed fitting frozen-BN CE improves");
  // Fresh inference-only model whose fit normalization comes from the
  // checkpoint (constructed with the neutral {0, 1}); tflow's GraphDef
  // placeholder audit has no eager counterpart.
  SeriesModel deployed({16, 0, 1}, false);
  deployed.Restore(prefix);
  check::Require(deployed.parameters.size() == 14 && deployed.state.size() == 8 && !deployed.optimizer,
                 "raw series model: 14 learned + 8 normalization/BN tensors, no optimizer");
  const auto   before = deployed.Snapshot();
  const Tensor logits = deployed.Predict(test), reference = model.Predict(test), repeat = deployed.Predict(test);
  check::Near(replay::MaxAbsDiff(logits, reference), 0, 1e-6, "independent learned/BN/normalization restore parity");
  check::Require(replay::MaxAbsDiff(logits, repeat) == 0, "deterministic series inference");
  const auto   one = SeriesRows(test, {0});
  const Tensor single = deployed.Predict(one);
  check::Near(replay::MaxAbsDiff(single, nn::slice_axis(logits, 0, 0, 1)), 0, 1e-6, "single/batch frozen BN series inference");
  const double sensitivity = nn::sum(nn::abs(nn::sub(nn::cast(single, nn::DType::Float64),
                                                     nn::cast(deployed.Predict(Reversed(one)), nn::DType::Float64))))
                                 .item<double>();
  check::Require(sensitivity > 1e-7, "temporal order affects restored features");
  check::Require(replay::SameBits(before, deployed.Snapshot()), "learned parameters,BN and fit normalization immutable");
  deployed.Export(directory / "model_arrays.npz");
  const Tensor nearest = SeriesReference(fit, test, false), centroid = SeriesReference(fit, test, true);
  const Tensor majority = nn::zeros({test.size(), 2});
  SeriesMetrics("trained temporal CNN", test, logits);
  SeriesMetrics("untrained", test, untrained);
  SeriesMetrics("fit Euclidean1NN pseudo-logits", test, nearest);
  SeriesMetrics("fit class centroid negative MSE", test, centroid);
  SeriesMetrics("fit majority tie class0", test, majority);
  sentence::PrintConfusion("trained temporal CNN test", sentence::Confusion(test.labels, sentence::Positive(logits), 2), 2);
  std::ofstream predictions(directory / "predictions.tsv");
  predictions << "source_id\tlabel\tprediction\tnearest\tcentroid\n";
  const auto ids = test.source_ids.to_vector<int>(), labels = test.labels.to_vector<int>();
  const auto p = sentence::Positive(logits), q = sentence::Positive(nearest), r = sentence::Positive(centroid);
  for (std::int64_t n = 0; n < test.size(); ++n) predictions << ids[n] << '\t' << labels[n] << '\t' << p[n] << '\t' << q[n] << '\t' << r[n] << '\n';
  const Tensor norm = nn::from_vector(std::vector<float>{data.config.mean, data.config.standard_deviation}, {2});
  nn::save_npz((directory / "test_outputs.npz").string(),
               {{"signals", test.signals}, {"labels", test.labels}, {"source_ids", test.source_ids}, {"logits", logits},
                {"untrained", untrained}, {"nearest", nearest}, {"centroid", centroid}, {"fit_signals", fit.signals},
                {"fit_labels", fit.labels}, {"fit_source_ids", fit.source_ids}, {"normalization", norm}});
  sentence::CopyIfExists(root / "manifest.json", directory / "data_manifest.json");
  check::Require(bool(history) && bool(predictions), "save all series histories/predictions");
  std::cout << " selected=" << selected << ",fixed fitting frozen CE " << initial << " -> " << learned << ",time reversal logit change="
            << sensitivity << ";standalone parity/immutable normalization+BN/single-batch/determinism passed\nArtifacts:" << directory
            << "\nGunPoint small custom training holdout,not actor-disjoint or a tuned benchmark result.\n";
}
}  // namespace series



namespace nn = datamunge::nn;
using nn::Tensor;
using sentence::Weights;

namespace {

std::array<double, 2> Host(const Weights& weights) {
  constexpr double means[3][2] = {{-.2, 0}, {-.1, .05}, {.1, 0}}, variances[3][2] = {{1.3, 1}, {1.2, .7}, {.9, 1}};
  // Constants are rounded to the same float values stored in native state.
  std::vector<std::vector<double>> input(7, std::vector<double>(1));
  for (int t = 0; t < 7; ++t) input[t][0] = (float(.7 * std::cos(.8 * t)) - float(.3)) / float(1.2);
  int incoming = 1;
  for (int layer = 0; layer < 3; ++layer) {
    const int width = layer == 1 ? 2 : 1, kernel = layer == 0 ? 8 : layer == 1 ? 5 : 3, pad = (kernel - 1) / 2;
    std::vector<std::vector<double>> next(7, std::vector<double>(width));
    for (int t = 0; t < 7; ++t)
      for (int c = 0; c < width; ++c) {
        double value = weights[4 * layer + 1][c];
        for (int k = 0; k < kernel; ++k) {
          const int position = t + k - pad;
          if (position < 0 || position >= 7) continue;
          for (int j = 0; j < incoming; ++j) value += input[position][j] * weights[4 * layer][(k * incoming + j) * width + c];
        }
        value = weights[4 * layer + 2][c] * (value - float(means[layer][c])) / std::sqrt(float(variances[layer][c]) + float(.001)) +
                weights[4 * layer + 3][c];
        next[t][c] = std::max(0., value);
      }
    input = next;
    incoming = width;
  }
  double average = 0;
  for (int t = 0; t < 7; ++t) average += input[t][0] / 7;
  return {average * weights[12][0] + weights[13][0], average * weights[12][1] + weights[13][1]};
}
double Objective(const Weights& weights) {
  const auto z = Host(weights);
  return .3 * z[0] - .2 * z[1];
}

void HostChecks() {
  nn::ParameterStore       p(433), state(439);
  const series::SeriesConfig config{1, .3f, 1.2f};
  std::vector<float>       x(7);
  for (int t = 0; t < 7; ++t) x[t] = static_cast<float>(.7 * std::cos(.8 * t));
  const Tensor input = nn::from_vector(x, {1, 7, 1});
  series::SeriesCnn(p, state, input, config);
  const auto weights = sentence::FixParameters(p, [](std::size_t v, std::int64_t i, const nn::Shape&) {
    const bool scale = v < 12 && v % 4 == 2;
    return scale ? 1.f + .1f * static_cast<float>(std::sin(i + v)) : .13f + .04f * static_cast<float>(std::sin(.6 * i + .8 * v));
  });
  const std::vector<std::vector<float>> fixed = {{.3f}, {1.2f}, {-.2f}, {1.3f}, {-.1f, .05f}, {1.2f, .7f}, {.1f}, {.9f}};
  for (std::size_t v = 0; v < fixed.size(); ++v)
    state.set(state.names()[v], nn::reshape(nn::from_vector(fixed[v]), state.parameters()[v].shape()));
  const auto net = series::SeriesCnn(p, state, input, config);
  const auto gradient = nn::grad(nn::sum(nn::mul(net.logits, nn::from_vector(std::vector<float>{.3f, -.2f}, {1, 2}))), p.parameters());
  const auto expected = Host(weights);
  const auto z = net.logits.to_vector<double>(), normalized = net.normalized.to_vector<double>();
  double     value_error = 0;
  for (int c = 0; c < 2; ++c) value_error = std::max(value_error, std::abs(expected[c] - z[c]));
  for (int t = 0; t < 7; ++t) check::Near(normalized[t], (double(x[t]) - float(.3)) / float(1.2), 1e-7, "saved fit-global centering/scaling");
  int          coordinates = 0;
  const double gradient_error = sentence::HostGradientError(weights, gradient, Objective, coordinates);
  check::Require(coordinates == 40 && p.size() == 14 && state.size() == 8, "all40tiny learned coordinates and8 normalization/BN tensors");
  check::Near(value_error, 0, 2e-7, "independent even8/odd5/3 SAME/BN/ReLU/GAP/head values");
  check::Near(gradient_error, 0, 2e-6, "EVERY convolution/BN-affine/head host finite difference");
  std::cout << " independent temporal8/5/3 SAME/BN/GAP/head error=" << value_error << ",ALL40 parameter-coordinate host FD error="
            << gradient_error << ";nontrivial saved mean/std/BN fixtures passed\n";
}

void BaselineChecks() {
  std::vector<float> f(3 * 150), t(2 * 150);
  for (int n = 0; n < 3; ++n)
    for (int i = 0; i < 150; ++i) f[n * 150 + i] = 2.f * n;
  for (int i = 0; i < 150; ++i) {
    t[i] = 1;
    t[150 + i] = 3.8f;
  }
  const series::SeriesData fit{nn::from_vector(f, {3, 150, 1}), nn::from_vector(std::vector<int>{0, 1, 1}, {3}), nn::zeros({3}, nn::DType::Int32)};
  const series::SeriesData test{nn::from_vector(t, {2, 150, 1}), nn::zeros({2}, nn::DType::Int32), nn::zeros({2}, nn::DType::Int32)};
  const auto nearest = series::SeriesReference(fit, test, false).to_vector<double>(),
             centroid = series::SeriesReference(fit, test, true).to_vector<double>();
  check::Require(nearest[0] == 1 && nearest[3] == 1, "1NN exact distances and first fitting-row tie rule");
  check::Near(centroid[0], -1, 0, "known class0 centroid MSE");
  check::Near(centroid[1], -4, 0, "class1 centroid fits ONLY retained training rows");
  check::Near(centroid[3], -.64, 1e-7, "known centroid off-center MSE");
  std::cout << " independent1NN/tie/fitting-centroid fixtures passed\n";
}

void FullChecks() {
  nn::ParameterStore p(443), state(449);
  std::vector<float> x(150);
  for (int t = 0; t < 150; ++t) x[t] = static_cast<float>(std::sin(t * .13) + .2 * std::cos(t * .37));
  const auto network = series::SeriesCnn(p, state, nn::from_vector(x, {1, 150, 1}));
  const auto norms = sentence::SquaredNorms(nn::grad(nn::sum(nn::square(network.logits)), p.parameters()));
  for (int i = 0; i < 3; ++i) check::Require(network.stages[i].shape() == nn::Shape{1, 150, i == 1 ? 256 : 128}, "full SAME temporal lengths/widths");
  check::Require(network.logits.shape() == nn::Shape{1, 2} && p.count() == 264962 && p.size() == 14 && state.size() == 8,
                 "independent full parameter/state count and binary head");
  for (double n : norms) check::Require(std::isfinite(n) && n > 0, "every full inference learned tensor adjoint finite/nonzero");
  std::cout << " full temporal FCN264962 parameters/14 learned tensors+8state,channels128/256/128,T150,kernels8/5/3;all stage shapes/learned gradients passed\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    int                   epochs = 3;
    bool                  checks = false;
    std::filesystem::path root = paths::Dataset("time_series");
    nn::set_num_threads(4);
    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      if (arg.rfind("epochs=", 0) == 0) epochs = std::stoi(arg.substr(7));
      else if (arg.rfind("data=", 0) == 0) root = arg.substr(5);
      else if (arg.rfind("threads=", 0) == 0) nn::set_num_threads(std::stoi(arg.substr(8)));
      else if (arg == "checks_only=1") checks = true;
      else if (arg != "checks_only=0") throw std::invalid_argument("expected epochs=N data=PATH checks_only=0|1 threads=N");
    }
    HostChecks();
    BaselineChecks();
    FullChecks();
    if (!checks) series::TrainSeries(root, epochs);
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
