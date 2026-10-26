// CMD: just run-nn tcn
// TCN:six two-convolution residual blocks,weight normalization,dilations1..32,
// explicit left-only padding,ReLU/dropout and causal endpoint classifier.
// Independent all20-coordinate host derivatives,exact receptive-field/input
// Jacobian/future-perturbation/prefix fixtures,full/compact learned gradients.
// Three compact default epochs on pinned real GunPoint;honest simple baselines.
//
// Port of tflow's tcn. Data: tflow's `just time-series-assets`
// (output/datasets/time_series/time_series.npz, found via paths::Dataset).
// Deviations: dilated taps use nn::conv1d's dilation instead of tflow's
// hand-zero-spaced kernels (same function); .npz checkpoints; tflow's
// GraphDef audit has no eager counterpart. Weights initialise with nn's
// generator, so trained numbers match TF statistically.
// Options: epochs=3 data=PATH checks_only=0|1 threads=4.
// Output: output/nn/models/tcn/.

#include <array>

#include "common/tcn.hpp"

// Experiment workflow from examples/nn/common/tcn.hpp.
namespace tcn {
using namespace ::tcn;
inline void TrainTcn(const std::filesystem::path& root, int epochs) {
  check::Require(epochs > 0, "positive short TCN epochs");
  const auto  data = series::LoadSeries(root);
  const auto& fit = data.pools[0];
  const auto& validation = data.pools[1];
  const auto& test = data.pools[2];
  check::Require(fit.size() == 40 && validation.size() == 10 && test.size() == 150, "configured GunPoint pool sizes");
  const TcnConfig config{8, 6, 3, data.config.mean, data.config.standard_deviation};
  TcnModel        model(config);
  const auto      directory = paths::OutputDir("models/tcn");
  const auto      prefix = (directory / "best.npz").string();
  const double    initial = model.Loss(fit);
  const Tensor    untrained = model.Predict(test);
  sentence::RequireLiveGradients(model.parameters,
                                 nn::grad(sentence::SentenceCE(model.Forward(fit, true).logits, fit.labels), model.parameters.parameters()),
                                 "TCN v/g/b/projection/head");
  std::cout << "TCN:" << model.parameters.count() << " parameters/" << model.parameters.size()
            << " tensors+2fit-normalization state;6two-conv residual blocks,C8,kernel3,dilations1/2/4/8/16/32,RF253,weightnorm,dropout.2;"
            << epochs << " epochs,batch8,Adam.001,global-gradient clip1;all gradients finite/nonzero,initial fitCE=" << initial << std::endl;
  std::ofstream history(directory / "history.tsv");
  history << "epoch\ttrain_dropout_ce\tvalidation_ce\n";
  nn::BatchOrder order(fit.size(), 8, 467);
  double         best = INFINITY;
  int            selected = 0;
  for (int epoch = 1; epoch <= epochs; ++epoch) {
    order.begin_epoch();
    double sum = 0;
    int    seen = 0;
    for (;;) {
      const auto rows = order.next();
      if (rows.empty()) break;
      // Dropout CE, all gradients, global-norm clip to 1 (scale min(1, 1 / (|g| + 1e-6))) and one Adam step.
      const auto   batch = SeriesRows(fit, rows);
      const Tensor loss = sentence::SentenceCE(model.Forward(batch, true).logits, batch.labels);
      auto         g = nn::grad(loss, model.parameters.parameters());
      double       squared = 0;
      for (const auto& x : g) squared += nn::sum(nn::square(x)).item<double>();
      const float scale = std::min(1.f, 1.f / (static_cast<float>(std::sqrt(squared)) + 1e-6f));
      for (auto& x : g) x = nn::mul(x, static_cast<double>(scale));
      model.optimizer->step(g);
      const double value = loss.item<double>();
      check::Require(std::isfinite(value) && value >= 0, "finite TCN CE");
      sum += rows.size() * value;
      seen += static_cast<int>(rows.size());
    }
    check::Require(seen == fit.size(), "every fitting series once per epoch");
    const double score = model.Loss(validation);
    if (score < best) {
      best = score;
      selected = epoch;
      model.Save(prefix);
    }
    history << epoch << '\t' << sum / seen << '\t' << score << '\n';
    history.flush();
    std::cout << " epoch " << epoch << ":trainCE=" << sum / seen << ",validationCE=" << score << std::endl;
  }
  model.Restore(prefix);
  check::Near(model.Loss(validation), best, 1e-6, "restored validation CE");
  const double learned = model.Loss(fit);
  check::Require(learned < initial, "three-epoch fixed fitting inference CE improves");
  // Fresh variable-time inference model (normalization from the checkpoint);
  // tflow's GraphDef placeholder audit has no eager counterpart.
  TcnModel deployed({8, 6, 3, 0, 1}, false);
  deployed.Restore(prefix);
  check::Require(deployed.parameters.size() == 40 && deployed.state.size() == 2 && !deployed.optimizer,
                 "observed-series model: 40 learned tensors + fit normalization, no optimizer");
  const auto   before = deployed.Snapshot();
  const Tensor logits = deployed.Predict(test), sequence = deployed.Predict(test, true), reference = model.Predict(test),
               repeat = deployed.Predict(test);
  check::Near(replay::MaxAbsDiff(logits, reference), 0, 1e-5, "independent restored endpoint parity");
  check::Require(replay::MaxAbsDiff(logits, repeat) == 0, "deterministic TCN inference");
  check::Require(replay::MaxAbsDiff(logits, nn::select(sequence, 1, 149)) == 0, "classification uses last causal time step");
  for (int length : {1, 17, 75, 150})
    check::Near(replay::MaxAbsDiff(deployed.Predict(TcnPrefix(test, length)), nn::select(sequence, 1, length - 1)), 0, 1e-5,
                "whole-batch dynamic prefix equals full-sequence same-time output");
  auto         one = SeriesRows(test, {0});
  const Tensor single = deployed.Predict(one);
  check::Near(replay::MaxAbsDiff(single, nn::slice_axis(logits, 0, 0, 1)), 0, 1e-5, "single/batch TCN logits");
  auto x = one.signals.to_vector<float>();
  for (int t = 75; t < 150; ++t) x[t] += static_cast<float>(3 * std::sin(t * .3));
  one.signals = nn::from_vector(x, {1, 150, 1});
  const auto altered = deployed.Predict(one, true).to_vector<double>(), original = nn::slice_axis(sequence, 0, 0, 1).to_vector<double>();
  double     future_change = 0;
  for (int t = 0; t < 150; ++t)
    for (int c = 0; c < 2; ++c) {
      const double difference = std::abs(altered[2 * t + c] - original[2 * t + c]);
      if (t < 75) check::Near(difference, 0, 0, "changed future cannot affect any prefix logit");
      else future_change += difference;
    }
  check::Require(future_change > 1e-5, "future observations affect later predictions");
  check::Require(replay::SameBits(before, deployed.Snapshot()), "all TCN parameters/gains/fit statistics immutable");
  deployed.Export(directory / "model_arrays.npz");
  const Tensor nearest = series::SeriesReference(fit, test, false), centroid = series::SeriesReference(fit, test, true);
  const Tensor majority = nn::zeros({150, 2});
  series::SeriesMetrics("trained TCN", test, logits);
  series::SeriesMetrics("untrained", test, untrained);
  series::SeriesMetrics("fit1NN pseudo-logits", test, nearest);
  series::SeriesMetrics("fit centroid negative MSE", test, centroid);
  series::SeriesMetrics("fit majority tie class0", test, majority);
  sentence::PrintConfusion("trained TCN test", sentence::Confusion(test.labels, sentence::Positive(logits), 2), 2);
  std::ofstream predictions(directory / "predictions.tsv");
  predictions << "source_id\tlabel\tprediction\tnearest\tcentroid\n";
  const auto ids = test.source_ids.to_vector<int>(), labels = test.labels.to_vector<int>();
  const auto p = sentence::Positive(logits), q = sentence::Positive(nearest), r = sentence::Positive(centroid);
  for (int n = 0; n < 150; ++n) predictions << ids[n] << '\t' << labels[n] << '\t' << p[n] << '\t' << q[n] << '\t' << r[n] << '\n';
  nn::save_npz((directory / "test_outputs.npz").string(),
               {{"signals", test.signals}, {"labels", test.labels}, {"source_ids", test.source_ids}, {"logits", logits},
                {"sequence_logits", sequence}, {"untrained", untrained}, {"nearest", nearest}, {"centroid", centroid},
                {"fit_signals", fit.signals}, {"fit_labels", fit.labels}, {"fit_source_ids", fit.source_ids}});
  sentence::CopyIfExists(root / "manifest.json", directory / "data_manifest.json");
  check::Require(bool(history) && bool(predictions), "save all TCN histories/predictions");
  std::cout << " selected=" << selected << ",fixed fitCE " << initial << " -> " << learned << ",future-only later-logit total change="
            << future_change << ";restored causal prefixes/all150samples/single-batch/immutable inference passed\nArtifacts:" << directory
            << "\nCustom tiny GunPoint holdout,not paper benchmark quality or physically online raw-sensor normalization.\n";
}
}  // namespace tcn



namespace nn = datamunge::nn;
using nn::Tensor;
using sentence::Weights;

namespace {

std::vector<std::array<double, 2>> Host(const Weights& weights) {
  std::vector<double> x(7);
  for (int t = 0; t < 7; ++t) x[t] = (double(float(1 + .3 * std::sin(.7 * t))) - float(.2)) / float(1.3);
  int variable = 0;
  for (int level = 0; level < 2; ++level) {
    const auto residual = x;
    for (int layer = 0; layer < 2; ++layer) {
      const auto&  v = weights[variable];
      const double gain = weights[variable + 1][0], bias = weights[variable + 2][0], norm = std::sqrt(v[0] * v[0] + v[1] * v[1] + 1e-12);
      variable += 3;
      std::vector<double> next(7);
      for (int t = 0; t < 7; ++t) {
        double value = bias;
        for (int k = 0; k < 2; ++k) {
          const int position = t + (k - 1) * (1 << level);
          if (position >= 0) value += x[position] * v[k] * gain / norm;
        }
        next[t] = std::max(0., value);
      }
      x = next;
    }
    for (int t = 0; t < 7; ++t) x[t] = std::max(0., x[t] + residual[t]);
  }
  std::vector<std::array<double, 2>> out(7);
  for (int t = 0; t < 7; ++t)
    for (int c = 0; c < 2; ++c) out[t][c] = x[t] * weights[12][c] + weights[13][c];
  return out;
}
double Objective(const Weights& weights) {
  const auto out = Host(weights);
  return .3 * out.back()[0] - .2 * out.back()[1];
}

void HostChecks() {
  nn::ParameterStore p(479), state(487);
  std::vector<float> x(7);
  for (int t = 0; t < 7; ++t) x[t] = static_cast<float>(1 + .3 * std::sin(.7 * t));
  const Tensor          input = nn::from_vector(x, {1, 7, 1});
  const tcn::TcnConfig config{1, 2, 2, .2f, 1.3f};
  tcn::Tcn(p, state, input, config);
  const auto weights = sentence::FixParameters(p, [](std::size_t v, std::int64_t i, const nn::Shape&) {
    return v < 12 && v % 3 == 1 ? .7f + .03f * static_cast<float>(std::cos(v)) : .15f + .05f * static_cast<float>(std::sin(i * .8 + v * .5));
  });
  const auto net = tcn::Tcn(p, state, input, config);
  const auto gradient = nn::grad(nn::sum(nn::mul(net.logits, nn::from_vector(std::vector<float>{.3f, -.2f}, {1, 2}))), p.parameters());
  const auto expected = Host(weights);
  const auto z = net.sequence.to_vector<double>();
  double     value_error = 0;
  for (int t = 0; t < 7; ++t)
    for (int c = 0; c < 2; ++c) value_error = std::max(value_error, std::abs(expected[t][c] - z[2 * t + c]));
  int          coordinates = 0;
  const double gradient_error = sentence::HostGradientError(weights, gradient, Objective, coordinates);
  check::Require(coordinates == 20 && p.size() == 14, "all20 tiny weightnorm/gain/bias/head coordinates");
  check::Near(value_error, 0, 2e-6, "independent two-level causal weightnorm/ReLU/residual/sequence head");
  check::Near(gradient_error, 0, 3e-6, "EVERY v/g/b/head host finite difference");
  std::cout << " independent TCN every-time forward error=" << value_error << ",ALL20 parameter-coordinate host FD error=" << gradient_error
            << std::endl;
}

void CausalityChecks() {
  nn::ParameterStore   p(491), state(499);
  const tcn::TcnConfig config{1, 3, 2, 0, 1};
  tcn::Tcn(p, state, nn::zeros({1, 20, 1}), config);
  for (const auto& name : p.names()) {
    const double value = name.find("/v") != std::string::npos ? .3 : name.find("/g") != std::string::npos ? 1. : .1;
    p.set(name, nn::full(p.at(name).shape(), value));
  }
  Tensor x = nn::full({1, 20, 1}, 2.0);
  x.requires_grad_(true);
  const auto net = tcn::Tcn(p, state, x, config);
  const auto g = nn::grad(nn::sum(nn::select(net.hidden, 1, 19)), {x})[0].to_vector<double>();
  const auto reference = net.sequence.to_vector<double>();
  check::Require(tcn::TcnReceptiveField(config) == 15, "two convs per dilation give15-position RF");
  for (int t = 0; t < 20; ++t) {
    if (t < 5) check::Near(g[t], 0, 0, "outside causal receptive field Jacobian exactly zero");
    else check::Require(g[t] > 0, "EVERY reachable lag in RF has positive derivative");
  }
  std::vector<float> input(20, 2.f);
  for (int t = 8; t < 20; ++t) input[t] += t * .2f;
  const auto changed = tcn::Tcn(p, state, nn::from_vector(input, {1, 20, 1}), config).sequence.to_vector<double>();
  double     future_change = 0;
  for (int t = 0; t < 20; ++t)
    for (int c = 0; c < 2; ++c) {
      const double difference = std::abs(changed[2 * t + c] - reference[2 * t + c]);
      if (t < 8) check::Near(difference, 0, 0, "future perturbation cannot affect prefix outputs");
      else future_change += difference;
    }
  check::Require(future_change > 0, "future input affects its own later causal output");
  const auto prefix = tcn::Tcn(p, state, nn::full({1, 8, 1}, 2.0), config).logits.to_vector<double>();
  for (int c = 0; c < 2; ++c) check::Near(prefix[c], reference[7 * 2 + c], 1e-6, "prefix last output equals full same-time output");
  std::cout << " exact RF15 input Jacobian/all reachable lags/future zero/prefix equivalence fixtures passed\n";
}

void FullChecks() {
  nn::ParameterStore   p(503), state(509);
  const tcn::TcnConfig config;
  std::vector<float>   x(150);
  for (int t = 0; t < 150; ++t) x[t] = static_cast<float>(.5 + std::sin(t * .11));
  const auto network = tcn::Tcn(p, state, nn::from_vector(x, {1, 150, 1}), config);
  const auto norms = sentence::SquaredNorms(nn::grad(nn::sum(nn::square(network.logits)), p.parameters()));
  for (int i = 0; i < 6; ++i) check::Require(network.blocks[i].shape() == nn::Shape{1, 150, 64}, "causal dilation preserves time at EVERY residual block");
  check::Require(network.sequence.shape() == nn::Shape{1, 150, 2} && p.count() == 137154 && p.size() == 40 && state.size() == 2 &&
                     tcn::TcnReceptiveField(config) == 253,
                 "independent full parameter/state count and RF253");
  for (double n : norms) check::Require(std::isfinite(n) && n > 0, "all40 full learned tensor gradients finite/nonzero");
  std::cout << " full TCN137154 parameters/40 tensors+2state,six levelsC64,k3,RF253;all block shapes and every learned gradient passed\n";
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
    CausalityChecks();
    FullChecks();
    if (!checks) tcn::TrainTcn(root, epochs);
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
