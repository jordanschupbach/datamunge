// CMD: just run-nn wavenet_stack
// WaveNet-style gated causal convolutions,residual/skip paths and mu-law256
// next-audio distributions,with real sampled autoregressive feedback.
// Independent every-coordinate host/gating/causal/shift/warmup tests,full graph
// and compact gradients. Three epochs on pinned speaker-disjoint FSDD audio.
//
// Port of tflow's wavenet_stack. Data: tflow's `just wavenet-stack-assets`
// (output/datasets/wavenet/wavenet_stack.npz, found via paths::Dataset).
// Deviations: dilated taps use nn::conv1d's dilation instead of tflow's
// hand-zero-spaced kernels (same function); WAVs via nn::write_wav (TF
// EncodeWav's 16-bit conversion); .npz checkpoints; tflow's GraphDef audit has
// no eager counterpart. Weights initialise with nn's generator, so trained
// numbers and sampled audio match TF statistically.
// Options: epochs=3 data=PATH checks_only=0|1 threads=4.
// Output: output/nn/models/wavenet_stack/.

#include <array>

#include "common/wavenet_stack.hpp"

// Experiment workflow from examples/nn/common/wavenet_stack.hpp.
namespace wavestack {
using namespace ::wavestack;
inline void TrainWaveStack(const std::filesystem::path& root, int epochs) {
  check::Require(epochs > 0, "positive short WaveNet stack epochs");
  const auto     data = LoadWaveStack(root);
  const auto&    fit = data.pools[0];
  const auto&    validation = data.pools[1];
  const auto&    test = data.pools[2];
  WaveStackModel model;
  const auto     directory = paths::OutputDir("models/wavenet_stack");
  const auto     prefix = (directory / "best.npz").string();
  const double   initial = model.Loss(fit);
  const Tensor   untrained = model.Predict(test.input);
  const auto     probe = WaveRows(fit, Range(0, 8));
  sentence::RequireLiveGradients(model.parameters,
                                 nn::grad(WaveNll(model.Logits(probe.input), probe.target, model.config.classes, model.loss_start),
                                          model.parameters.parameters()),
                                 "gated/skip/residual/head");
  std::cout << "WaveNet-style stack:" << model.parameters.count() << " parameters/" << model.parameters.size()
            << " tensors,12gatedlayers,dilations1..32twice,RF128,R8/S16,mu-law256;" << epochs
            << " epochs,batch8,Adam.001,warmup127;all gradients finite/nonzero,initial fitNLL=" << initial << std::endl;
  std::ofstream history(directory / "history.tsv");
  history << "epoch\ttrain_nll\tvalidation_nll\n";
  nn::BatchOrder order(fit.size(), 8, 557);
  double         best = INFINITY;
  int            selected = 0;
  for (int epoch = 1; epoch <= epochs; ++epoch) {
    order.begin_epoch();
    double sum = 0;
    int    seen = 0;
    for (;;) {
      const auto rows = order.next();
      if (rows.empty()) break;
      // Teacher-forced warmup-masked next-code NLL, all gradients and one Adam step.
      const auto   batch = WaveRows(fit, rows);
      const Tensor loss = WaveNll(model.Logits(batch.input), batch.target, model.config.classes, model.loss_start);
      model.optimizer->step(nn::grad(loss, model.parameters.parameters()));
      const double value = loss.item<double>();
      check::Require(std::isfinite(value) && value >= 0, "finite audio next-code NLL");
      sum += rows.size() * value;
      seen += static_cast<int>(rows.size());
    }
    check::Require(seen == fit.size(), "every real fitting window once per epoch");
    const double score = model.Loss(validation);
    if (score < best) {
      best = score;
      selected = epoch;
      model.Save(prefix);
    }
    history << epoch << '\t' << sum / seen << '\t' << score << '\n';
    history.flush();
    std::cout << " epoch " << epoch << ":trainNLL=" << sum / seen << ",validationNLL=" << score << std::endl;
  }
  model.Restore(prefix);
  check::Near(model.Loss(validation), best, 1e-6, "restored validation NLL");
  const double learned = model.Loss(fit);
  check::Require(learned < initial, "short-run fixed fitting next-audio NLL improves");
  // Fresh previous-code-only generation model; tflow's GraphDef placeholder
  // audit has no eager counterpart (Predict takes only observed codes).
  WaveStackModel deployed({8, 16, 6, 2, 256}, false);
  deployed.Restore(prefix);
  check::Require(deployed.parameters.size() == 76 && !deployed.optimizer, "observed audio codes model: 76 learned tensors, no optimizer");
  const auto   before = deployed.Snapshot();
  const Tensor logits = deployed.Predict(test.input), reference = model.Predict(test.input), repeat = deployed.Predict(test.input);
  check::Near(replay::MaxAbsDiff(logits, reference), 0, 2e-6, "independent restored all-window/all-time logits");
  check::Require(replay::MaxAbsDiff(logits, repeat) == 0, "deterministic next-code inference");
  for (int length : {1, 64, 128, 256}) {
    const Tensor prefix_logits = deployed.Predict(nn::slice_axis(test.input, 1, 0, length).contiguous(), true);
    check::Near(replay::MaxAbsDiff(prefix_logits, nn::select(logits, 1, length - 1)), 0, 2e-6,
                "every held-out prefix equals same-time full-window prediction");
  }
  auto changed = test.input.to_vector<int>();
  for (int n = 0; n < 24; ++n)
    for (int t = 128; t < 256; ++t) changed[n * 256 + t] = (changed[n * 256 + t] + 37) % 256;
  const auto altered = deployed.Predict(nn::from_vector(changed, {24, 256})).to_vector<float>(), original = logits.to_vector<float>();
  double     future_change = 0;
  for (int n = 0; n < 24; ++n)
    for (int t = 0; t < 256; ++t)
      for (int c = 0; c < 256; ++c) {
        const std::size_t i = (static_cast<std::size_t>(n) * 256 + t) * 256 + c;
        const double      difference = std::abs(double(altered[i]) - original[i]);
        if (t < 128) check::Near(difference, 0, 0, "future target/input suffix cannot leak into earlier predictions");
        else future_change += difference;
      }
  check::Require(future_change > 1e-5, "later audio codes affect later logits");
  WaveMetrics("trained gated stack", logits, test);
  WaveMetrics("untrained", untrained, test);
  const auto prior = WavePriors(fit);
  WaveMetrics("fit add-one unigram", WavePriorLogits(test, prior.first), test);
  WaveMetrics("fit add-one transition", WavePriorLogits(test, prior.second), test);
  double     persistence = 0;
  const auto tin = test.input.to_vector<int>(), tout = test.target.to_vector<int>();
  for (int n = 0; n < 24; ++n)
    for (int t = 127; t < 256; ++t) persistence += std::pow(MuLawDecode(tout[n * 256 + t]) - MuLawDecode(tin[n * 256 + t]), 2);
  std::cout << " raw amplitude persistence MSE=" << persistence / (24 * 129) << '\n';
  deployed.Export(directory / "model_arrays.npz");
  nn::save_npz((directory / "test_outputs.npz").string(),
               {{"input", test.input}, {"target", test.target}, {"source_id", test.source}, {"speaker_id", test.speaker},
                {"start", test.start}, {"logits", logits}, {"untrained", untrained}, {"unigram_logcounts", prior.first},
                {"transition_logcounts", prior.second}});
  WaveGenerate(deployed, data, directory);
  check::Require(replay::SameBits(before, deployed.Snapshot()), "inference and all sampled autoregressive feedback leave every model byte unchanged");
  sentence::CopyIfExists(root / "manifest.json", directory / "data_manifest.json");
  sentence::CopyIfExists(root / "ATTRIBUTION.txt", directory / "ATTRIBUTION.txt");
  check::Require(bool(history), "write audio histories");
  std::cout << " selected=" << selected << ",fixed fitNLL " << initial << " -> " << learned << ",future-only later-logit change=" << future_change
            << ";restore/causal-prefix/immutable sampled generation passed\nArtifacts:" << directory
            << "\nTiny48ms unconditional continuations,not intelligible-speech/conditioned-vocoder quality;later WaveNet/vocoder rows remain separate.\n";
}
}  // namespace wavestack



namespace nn = datamunge::nn;
using nn::Tensor;
using sentence::Weights;

namespace {

std::vector<std::array<double, 3>> Host(const Weights& w) {
  const int           ids[] = {0, 1, 2, 0, 2, 1};
  std::vector<double> x(6), skip(6);
  for (int t = 0; t < 6; ++t) {
    x[t] = w[1][0];
    for (int k = 0; k < 2; ++k) {
      const int position = t + k - 1;
      if (position >= 0) x[t] += w[0][3 * k + ids[position]];
    }
  }
  for (int layer = 0; layer < 2; ++layer) {
    const int           first = layer == 0 ? 2 : 8;
    std::vector<double> gate(6);
    for (int t = 0; t < 6; ++t) {
      double joined[2] = {w[first + 1][0], w[first + 1][1]};
      for (int k = 0; k < 2; ++k) {
        const int position = t + (k - 1) * (1 << layer);
        if (position >= 0)
          for (int c = 0; c < 2; ++c) joined[c] += x[position] * w[first][2 * k + c];
      }
      gate[t] = std::tanh(joined[0]) / (1 + std::exp(-joined[1]));
      skip[t] += gate[t] * w[first + 2][0] + w[first + 3][0];
    }
    if (layer == 0)
      for (int t = 0; t < 6; ++t) x[t] += gate[t] * w[6][0] + w[7][0];
  }
  std::vector<std::array<double, 3>> out(6);
  for (int t = 0; t < 6; ++t) {
    const double hidden = std::max(0., std::max(0., skip[t]) * w[12][0] + w[13][0]);
    for (int c = 0; c < 3; ++c) out[t][c] = hidden * w[14][c] + w[15][c];
  }
  return out;
}
double Objective(const Weights& w) {
  const auto out = Host(w);
  return .3 * out.back()[0] - .2 * out.back()[1] + .1 * out.back()[2];
}

void CodecLossChecks() {
  using wavestack::MuLawDecode;
  using wavestack::MuLawEncode;
  check::Require(MuLawEncode(-1) == 0 && MuLawEncode(0) == 128 && MuLawEncode(1) == 255 && MuLawEncode(-.5) == 16 && MuLawEncode(.5) == 239,
                 "independent mu-law endpoint/half/zero values");
  for (int q = 0; q < 256; ++q) check::Require(MuLawEncode(MuLawDecode(q)) == q, "every mu-law quantization-bin inverse roundtrip");
  check::Near(MuLawDecode(0), -1, 1e-12, "inverse negative endpoint");
  check::Near(MuLawDecode(255), 1, 1e-12, "inverse positive endpoint");
  const std::vector<float> values{1, 2, 3, -1, 0, 1, 1000, 999, 998, -2, 0, 2};
  const std::vector<int>   y{0, 1, 1, 2};
  Tensor                   z = nn::from_vector(values, {1, 4, 3});
  z.requires_grad_(true);
  const Tensor loss = wavestack::WaveNll(z, nn::from_vector(y, {1, 4}), 3, 2);
  const auto   g = nn::grad(loss, {z})[0].to_vector<double>();
  double       expected = 0;
  for (int t = 0; t < 4; ++t) {
    double maximum = -INFINITY, sum = 0;
    for (int c = 0; c < 3; ++c) maximum = std::max(maximum, double(values[t * 3 + c]));
    for (int c = 0; c < 3; ++c) sum += std::exp(values[t * 3 + c] - maximum);
    if (t >= 2) expected += (maximum + std::log(sum) - values[t * 3 + y[t]]) / 2;
    for (int c = 0; c < 3; ++c) {
      const double derivative = t < 2 ? 0 : (std::exp(values[t * 3 + c] - maximum) / sum - (c == y[t])) / 2;
      check::Near(g[t * 3 + c], derivative, 1e-7, "every masked stable NLL logit derivative/warmup zero");
    }
  }
  check::Near(loss.item<double>(), expected, 2e-7, "independent mean over ONLY scored positions");
  std::cout << " mu-law256 inverse/all-bin tests and stable warmup-masked NLL/all-logit derivatives passed\n";
}

void HostChecks() {
  nn::ParameterStore p(563);
  std::vector<float> onehot(18, 0.f);
  const int          ids[] = {0, 1, 2, 0, 2, 1};
  for (int t = 0; t < 6; ++t) onehot[t * 3 + ids[t]] = 1;
  const Tensor                     input = nn::from_vector(onehot, {1, 6, 3});
  const wavestack::WaveStackConfig config{1, 1, 2, 1, 3};
  wavestack::WaveStack(p, input, config);
  const auto weights = sentence::FixParameters(p, [](std::size_t v, std::int64_t i, const nn::Shape&) {
    return .13f + .04f * static_cast<float>(std::sin(.7 * i + .9 * v));
  });
  const auto net = wavestack::WaveStack(p, input, config);
  const auto endpoint = nn::select(net.logits, 1, 5);
  const auto gradients = nn::grad(nn::sum(nn::mul(endpoint, nn::from_vector(std::vector<float>{.3f, -.2f, .1f}, {1, 3}))), p.parameters());
  const auto expected = Host(weights);
  const auto z = net.logits.to_vector<double>();
  double     value_error = 0;
  for (int t = 0; t < 6; ++t)
    for (int c = 0; c < 3; ++c) value_error = std::max(value_error, std::abs(expected[t][c] - z[t * 3 + c]));
  int          coordinates = 0;
  const double gradient_error = sentence::HostGradientError(weights, gradients, Objective, coordinates);
  check::Require(coordinates == 33 && p.size() == 16, "all33 tiny coordinates in16 learned tensors");
  check::Near(value_error, 0, 1e-7, "independent causal/tanh-sigmoid/residual/skip/post-head values");
  check::Near(gradient_error, 0, 2e-6, "EVERY tiny initial/gate/residual/skip/post/head host finite difference");
  std::cout << " independent gated-stack forward error=" << value_error << ",ALL33 host parameter-coordinate FD error=" << gradient_error << std::endl;
}

void CausalChecks() {
  nn::ParameterStore               p(569);
  const wavestack::WaveStackConfig config{1, 1, 2, 1, 3};
  wavestack::WaveStack(p, nn::zeros({1, 8, 3}), config);
  for (const auto& name : p.names()) p.set(name, nn::full(p.at(name).shape(), .1));
  Tensor x = nn::full({1, 8, 3}, .2);
  x.requires_grad_(true);
  const auto net = wavestack::WaveStack(p, x, config);
  const auto g = nn::grad(nn::sum(nn::select(net.logits, 1, 7)), {x})[0].to_vector<double>();
  const auto reference = net.logits.to_vector<double>();
  check::Require(wavestack::WaveStackField(config) == 5, "initial2 plus gated dilations1/2 givesRF5");
  for (int t = 0; t < 8; ++t)
    for (int c = 0; c < 3; ++c) {
      if (t < 3) check::Near(g[t * 3 + c], 0, 0, "outside RF exactlyzero Jacobian");
      else check::Require(g[t * 3 + c] > 0, "every reachable lag/channel through gates and skips");
    }
  std::vector<float> input(24, .2f);
  for (int t = 4; t < 8; ++t)
    for (int c = 0; c < 3; ++c) input[t * 3 + c] += .4f;
  const auto changed = wavestack::WaveStack(p, nn::from_vector(input, {1, 8, 3}), config).logits.to_vector<double>();
  double     later = 0;
  for (int t = 0; t < 8; ++t)
    for (int c = 0; c < 3; ++c) {
      const double delta = std::abs(changed[t * 3 + c] - reference[t * 3 + c]);
      if (t < 4) check::Near(delta, 0, 0, "gated prefix independent of future");
      else later += delta;
    }
  check::Require(later > 0, "later predictions depend on changed later inputs");
  std::cout << " exact RF5 input-Jacobian/all reachable lags/future-zero gated fixtures passed\n";
}

void FullChecks() {
  nn::ParameterStore p(571);
  std::vector<int>   ids(160);
  for (int t = 0; t < 160; ++t) ids[t] = (t * 37) % 256;
  const wavestack::WaveStackConfig config;
  const auto net = wavestack::WaveStack(p, nn::one_hot(nn::from_vector(ids, {1, 160}), 256), config);
  const auto norms = sentence::SquaredNorms(nn::grad(nn::mean(nn::square(net.logits)), p.parameters()));
  for (int i = 0; i < 12; ++i) check::Require(net.gates[i].shape() == nn::Shape{1, 160, 32}, "every gated layer preserves160positions/R32");
  check::Require(net.logits.shape() == nn::Shape{1, 160, 256} && p.count() == 124096 && p.size() == 76 && wavestack::WaveStackField(config) == 128,
                 "independent full R32/S64 categorical parameter count/RF128");
  for (double n : norms) check::Require(std::isfinite(n) && n > 0, "all76 full parameter tensor derivatives finite/nonzero");
  std::cout << " full gated stack124096 parameters/76tensors,12layers,R32/S64,mu-law256,RF128;all stage shapes/learned gradients passed\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    int                   epochs = 3;
    bool                  checks = false;
    std::filesystem::path root = paths::Dataset("wavenet");
    nn::set_num_threads(4);
    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      if (arg.rfind("epochs=", 0) == 0) epochs = std::stoi(arg.substr(7));
      else if (arg.rfind("data=", 0) == 0) root = arg.substr(5);
      else if (arg.rfind("threads=", 0) == 0) nn::set_num_threads(std::stoi(arg.substr(8)));
      else if (arg == "checks_only=1") checks = true;
      else if (arg != "checks_only=0") throw std::invalid_argument("expected epochs=N data=PATH checks_only=0|1 threads=N");
    }
    CodecLossChecks();
    HostChecks();
    CausalChecks();
    FullChecks();
    if (!checks) wavestack::TrainWaveStack(root, epochs);
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
