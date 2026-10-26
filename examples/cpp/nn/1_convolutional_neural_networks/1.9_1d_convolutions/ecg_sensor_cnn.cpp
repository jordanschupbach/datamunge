// CMD: just run-nn ecg_sensor_cnn
// Real ECG200 and six-axis smartwatch classification:three strided temporal
// residual stages,fit-only per-channel normalization and weighted softmax CE.
// Independent all171-coordinate multichannel/stride/projection/gradient checks,
// full/compact graphs,known imbalance/ROC/F1 fixtures and observed-input restore.
// Three epochs per dataset;nearest/centroid/prior controls,all-class reporting.
//
// Port of tflow's ecg_sensor_cnn. Data: tflow's `just ecg-sensor-assets`
// (output/datasets/ecg_sensor/{ECG200,BasicMotions}/signals.npz, found via
// paths::Dataset). Deviations: .npz checkpoints; tflow's GraphDef audit has
// no eager counterpart. Weights initialise with nn's generator, so trained
// numbers match TF statistically.
// When tflow's recorded run is present (tf=DIR, default its
// output/models/ordered_ecg_sensor_cnn_darwin_20261020/ecg_sensor_cnn), the
// loaded data is compared with TF's saved arrays and TF's final weights are
// evaluated in this port's network (must reproduce TF's test logits).
// Options: epochs=3 data=PATH checks_only=0|1 seed=587 (nn weight seed) tf=DIR threads=4.
// Output: output/nn/models/ecg_sensor_cnn/{ECG200,BasicMotions}/.

#include "common/ecg_sensor.hpp"

// Experiment workflow from examples/nn/common/ecg_sensor.hpp.
namespace sensor {
using namespace ::sensor;
inline void TrainSensorOne(const std::filesystem::path& root, int epochs, std::uint64_t seed = 587) {
  const auto  data = LoadSensor(root);
  const auto& fit = data.pools[0];
  const auto& validation = data.pools[1];
  const auto& test = data.pools[2];
  const int   channels = data.config.channels, classes = data.config.classes;
  SensorModel model(data.config, data.mean, data.stddev, data.weights, true, seed);
  const auto  directory = paths::OutputDir("models/ecg_sensor_cnn/" + root.filename().string());
  const auto  prefix = (directory / "best.npz").string();
  const double initial = model.Loss(fit);
  const Tensor untrained = model.Predict(test);
  const auto  probe = SeriesRows(fit, Range(0, 8));
  sentence::RequireLiveGradients(model.parameters,
                                 nn::grad(SensorCE(model.Logits(probe), probe.labels, model.weights, classes), model.parameters.parameters()),
                                 "residual/main/head");
  std::cout << root.filename().string() << ":" << model.parameters.count()
            << " parameters/20learnedtensors+2per-channel normalization states;three residual stages8/16/32,weightedCE,batch8,Adam.001,"
            << epochs << " epochs;all20gradients passed,fit weights=";
  for (float w : data.weights.to_vector<float>()) std::cout << w << ',';
  std::cout << ",initial weightedfitCE=" << initial << std::endl;
  nn::BatchOrder order(fit.size(), 8, 599);
  double         best = INFINITY;
  int            selected = 0;
  std::ofstream  history(directory / "history.tsv");
  history << "epoch\ttrain_weightedCE\tvalidation_weightedCE\n";
  for (int epoch = 1; epoch <= epochs; ++epoch) {
    order.begin_epoch();
    double sum = 0;
    int    seen = 0;
    for (;;) {
      const auto r = order.next();
      if (r.empty()) break;
      // Class-weighted CE, all gradients and one Adam step.
      const auto   batch = SeriesRows(fit, r);
      const Tensor loss = SensorCE(model.Logits(batch), batch.labels, model.weights, classes);
      model.optimizer->step(nn::grad(loss, model.parameters.parameters()));
      const double value = loss.item<double>();
      check::Require(std::isfinite(value) && value >= 0, "finite weighted sensor objective");
      sum += r.size() * value;
      seen += static_cast<int>(r.size());
    }
    check::Require(seen == fit.size(), "all fitting ECG/sensor records once per epoch");
    const double value = model.Loss(validation);
    if (value < best) {
      best = value;
      selected = epoch;
      model.Save(prefix);
    }
    history << epoch << '\t' << sum / seen << '\t' << value << '\n';
    history.flush();
    std::cout << " epoch " << epoch << ":trainWeightedCE=" << sum / seen << ",validationWeightedCE=" << value << std::endl;
  }
  model.Restore(prefix);
  check::Near(model.Loss(validation), best, 1e-6, "restored weighted validation CE");
  const double learned = model.Loss(fit);
  check::Require(learned < initial, "short-run fixed weighted fitting CE improves");
  // Fresh raw-waveform-only model; its channel normalization comes from the
  // checkpoint. tflow's GraphDef placeholder audit has no eager counterpart.
  SensorModel deployed(data.config, {}, {}, {}, false);
  deployed.Restore(prefix);
  check::Require(deployed.parameters.size() == 20 && deployed.state.size() == 2 && !deployed.optimizer,
                 "only raw multichannel waveforms: 20 learned + 2 normalization tensors, no optimizer");
  const auto   before = deployed.Snapshot();
  const Tensor logits = deployed.Predict(test), reference = model.Predict(test), repeat = deployed.Predict(test);
  check::Near(replay::MaxAbsDiff(logits, reference), 0, 2e-6, "independent restored learned/channel-normalization parity");
  check::Require(replay::MaxAbsDiff(logits, repeat) == 0, "deterministic ECG/sensor inference");
  const auto   one = SeriesRows(test, {0});
  const Tensor single = deployed.Predict(one);
  check::Near(replay::MaxAbsDiff(single, nn::slice_axis(logits, 0, 0, 1)), 0, 2e-6, "single/batch multichannel inference");
  std::cout << " per-channel replacement mean absolute logit changes=";
  const auto means = data.mean.to_vector<float>();
  for (int channel = 0; channel < channels; ++channel) {
    auto x = one.signals.to_vector<float>();
    for (int t = 0; t < data.time; ++t) x[t * channels + channel] = means[channel];
    SeriesData changed = one;
    changed.signals = nn::from_vector(x, one.signals.shape());
    const double difference = MeanAbsChange(deployed.Predict(changed), single);
    check::Require(difference > 1e-7, "every observed signal channel affects logits");
    std::cout << difference << ',';
  }
  std::cout << '\n';
  const double order_change = MeanAbsChange(deployed.Predict(series::Reversed(one)), single) * classes;
  check::Require(order_change > 1e-7, "waveform order matters beyond channel averages");
  check::Require(replay::SameBits(before, deployed.Snapshot()), "all20weights/two fitting-stat tensors immutable");
  deployed.Export(directory / "model_arrays.npz");
  const Tensor nearest = SensorReference(fit, test, data.stddev, classes, false), centroid = SensorReference(fit, test, data.stddev, classes, true);
  std::vector<int> counts(classes, 1);
  for (int label : fit.labels.to_vector<int>()) ++counts[label];
  std::vector<float> prior_values(static_cast<std::size_t>(test.size() * classes));
  for (std::int64_t n = 0; n < test.size(); ++n)
    for (int c = 0; c < classes; ++c) prior_values[n * classes + c] = static_cast<float>(std::log(double(counts[c]) / (fit.size() + classes)));
  const Tensor prior = nn::from_vector(prior_values, {test.size(), classes});
  PrintSensor("trained residualCNN", logits, test, classes);
  PrintSensor("untrained", untrained, test, classes);
  PrintSensor("fit normalized1NN pseudo-logits", nearest, test, classes);
  PrintSensor("fit normalized centroid", centroid, test, classes);
  PrintSensor("fit class prior", prior, test, classes);
  sentence::PrintConfusion("trained residualCNN " + root.filename().string() + " test", ScoreSensor(logits, test.labels, classes).confusion, classes);
  std::ofstream predictions(directory / "predictions.tsv");
  predictions << "source_id\tlabel\tprediction\tnearest\tcentroid\n";
  const auto ids = test.source_ids.to_vector<int>(), labels = test.labels.to_vector<int>();
  const auto a = nn::argmax(logits, 1).to_vector<int>(), b = nn::argmax(nearest, 1).to_vector<int>(), c = nn::argmax(centroid, 1).to_vector<int>();
  for (std::int64_t n = 0; n < test.size(); ++n) predictions << ids[n] << '\t' << labels[n] << '\t' << a[n] << '\t' << b[n] << '\t' << c[n] << '\n';
  nn::save_npz((directory / "test_outputs.npz").string(),
               {{"signals", test.signals}, {"labels", test.labels}, {"source_ids", test.source_ids}, {"logits", logits},
                {"untrained", untrained}, {"nearest", nearest}, {"centroid", centroid}, {"prior", prior}, {"fit_signals", fit.signals},
                {"fit_labels", fit.labels}, {"fit_source_ids", fit.source_ids}, {"class_weights", data.weights}});
  sentence::CopyIfExists(root / "manifest.json", directory / "data_manifest.json");
  check::Require(bool(history) && bool(predictions), "save ECG/sensor histories/all predictions");
  std::cout << " selected=" << selected << ",weighted fitCE " << initial << " -> " << learned << ",time reversal logit change=" << order_change
            << ";restore/all-channel/single-batch/immutable state passed\nArtifacts:" << directory << std::endl;
}
}  // namespace sensor

// Experiment workflow from examples/nn/common/ecg_sensor.hpp.
namespace sensor {
using namespace ::sensor;
inline void TrainEcgSensor(const std::filesystem::path& root, int epochs, std::uint64_t seed = 587) {
  check::Require(epochs > 0, "positive short ECG/sensor epochs");
  TrainSensorOne(root / "ECG200", epochs, seed);
  TrainSensorOne(root / "BasicMotions", epochs, seed);
  std::cout << "Both real ECG and six-axis sensor demonstrations complete;official record test sets,not verified patient/participant holdouts.\n";
}
}  // namespace sensor



namespace nn = datamunge::nn;
using nn::Tensor;
using sentence::Weights;

namespace {

using Matrix = std::vector<std::vector<double>>;

Matrix Conv(const Matrix& x, const std::vector<double>& w, const std::vector<double>& bias, int kernel, int stride, int outgoing, bool relu) {
  const int incoming = static_cast<int>(x.front().size()), time = static_cast<int>(x.size()), length = (time + stride - 1) / stride,
            pad = std::max(0, (length - 1) * stride + kernel - time) / 2;
  Matrix out(length, std::vector<double>(outgoing));
  for (int t = 0; t < length; ++t)
    for (int c = 0; c < outgoing; ++c) {
      double value = bias[c];
      for (int k = 0; k < kernel; ++k) {
        const int position = t * stride + k - pad;
        if (position < 0 || position >= time) continue;
        for (int i = 0; i < incoming; ++i) value += x[position][i] * w[(k * incoming + i) * outgoing + c];
      }
      out[t][c] = relu ? std::max(0., value) : value;
    }
  return out;
}

std::vector<double> Host(const Weights& w) {
  Matrix x(10, std::vector<double>(2));
  for (int t = 0; t < 10; ++t) {
    x[t][0] = (double(float(.4 * std::cos(.6 * t))) - float(.3)) / float(1.2);
    x[t][1] = (double(float(.3 * std::sin(.8 * t))) - float(-.2)) / float(.7);
  }
  for (int stage = 0; stage < 3; ++stage) {
    const int  first = 6 * stage, width = 1 << stage;
    const auto main = Conv(Conv(x, w[first], w[first + 1], 5, 2, width, true), w[first + 2], w[first + 3], 3, 1, width, false);
    auto       residual = Conv(x, w[first + 4], w[first + 5], 1, 2, width, false);
    for (std::size_t t = 0; t < main.size(); ++t)
      for (int c = 0; c < width; ++c) residual[t][c] = std::max(0., residual[t][c] + main[t][c]);
    x = residual;
  }
  std::vector<double> average(4), out(3);
  for (const auto& row : x)
    for (int c = 0; c < 4; ++c) average[c] += row[c] / x.size();
  for (int c = 0; c < 3; ++c) {
    out[c] = w[19][c];
    for (int i = 0; i < 4; ++i) out[c] += average[i] * w[18][3 * i + c];
  }
  return out;
}
double Objective(const Weights& w) {
  const auto out = Host(w);
  return .3 * out[0] - .2 * out[1] + .1 * out[2];
}

void HostChecks() {
  nn::ParameterStore p(601), state(607);
  std::vector<float> values(20);
  for (int t = 0; t < 10; ++t) {
    values[2 * t] = static_cast<float>(.4 * std::cos(.6 * t));
    values[2 * t + 1] = static_cast<float>(.3 * std::sin(.8 * t));
  }
  const Tensor               input = nn::from_vector(values, {1, 10, 2});
  const sensor::SensorConfig config{2, 3, 1};
  sensor::SensorCnn(p, state, input, config);
  const auto weights = sentence::FixParameters(p, [](std::size_t v, std::int64_t i, const nn::Shape&) {
    return .12f + .04f * static_cast<float>(std::sin(.7 * i + .8 * v));
  });
  state.set("normalization/mean", nn::from_vector(std::vector<float>{.3f, -.2f}, {2}));
  state.set("normalization/std", nn::from_vector(std::vector<float>{1.2f, .7f}, {2}));
  const auto network = sensor::SensorCnn(p, state, input, config);
  const auto g = nn::grad(nn::sum(nn::mul(network.logits, nn::from_vector(std::vector<float>{.3f, -.2f, .1f}, {1, 3}))), p.parameters());
  const auto expected = Host(weights);
  const auto z = network.logits.to_vector<double>();
  double     value_error = 0;
  for (int c = 0; c < 3; ++c) value_error = std::max(value_error, std::abs(expected[c] - z[c]));
  int          coordinates = 0;
  const double gradient_error = sentence::HostGradientError(weights, g, Objective, coordinates);
  check::Require(coordinates == 171 && p.size() == 20 && state.size() == 2, "all171 tiny coordinates across20 parameters plus2 normalization tensors");
  check::Near(value_error, 0, 2e-6, "independent multichannel normalization/stridedSAME/residual/GAP/head forward");
  check::Near(gradient_error, 0, 3e-6, "EVERY conv/projection/head host finite difference");
  std::cout << " independent multichannel strided residual forward error=" << value_error << ",ALL171 host-coordinate FD error=" << gradient_error
            << std::endl;
}

void LossMetricChecks() {
  Tensor z = nn::from_vector(std::vector<float>{1000, 999, -2, 3}, {2, 2});
  z.requires_grad_(true);
  const Tensor loss = sensor::SensorCE(z, nn::from_vector(std::vector<int>{1, 0}, {2}), nn::from_vector(std::vector<float>{1.5f, .5f}, {2}), 2);
  const auto   g = nn::grad(loss, {z})[0].to_vector<double>();
  const auto   logits = z.to_vector<double>();
  double       expected = 0;
  for (int n = 0; n < 2; ++n) {
    const int    label = 1 - n;
    const double weight = label == 0 ? 1.5 : .5, a = logits[2 * n], b = logits[2 * n + 1], maximum = std::max(a, b),
                 sum = std::exp(a - maximum) + std::exp(b - maximum);
    expected += weight * (maximum + std::log(sum) - (label ? b : a)) / 2;
    for (int c = 0; c < 2; ++c)
      check::Near(g[2 * n + c], weight * (std::exp((c ? b : a) - maximum) / sum - (c == label)) / 2, 1e-7, "every weighted mean CE logit derivative");
  }
  check::Near(loss.item<double>(), expected, 1e-6, "weighted mean objective denominator is raw row count");
  std::vector<float> scores(18, 0.f);
  const int          predictions[] = {0, 1, 1, 2, 2, 2};
  for (int n = 0; n < 6; ++n) scores[n * 3 + predictions[n]] = 1;
  const auto metrics = sensor::ScoreSensor(nn::from_vector(scores, {6, 3}), nn::from_vector(std::vector<int>{0, 0, 1, 1, 2, 2}, {6}), 3);
  check::Near(metrics.accuracy, 2. / 3, 1e-12, "known multiclass accuracy");
  check::Near(metrics.balanced, 2. / 3, 1e-12, "known mean class recall");
  check::Near(metrics.macro_f1, (2. / 3 + .5 + .8) / 3, 1e-12, "known macroF1 with different precision/recall");
  check::Require(metrics.confusion == std::vector<int>({1, 1, 0, 0, 1, 1, 0, 0, 2}), "known ALL confusion cells");
  check::Near(sensor::SensorAuc0({.8, .8, .2, .1}, nn::from_vector(std::vector<int>{0, 1, 0, 1}, {4})), .625, 1e-12,
              "sourceclass0 ROC with positive/negative ties gets half credit");
  std::cout << " weighted CE host value error=" << std::abs(loss.item<double>() - expected)
            << ";extreme CE/all logits,ALL confusion/macroF1/balanced recall/tie-aware sourceclass0AUROC fixtures passed\n";
}

void FullChecks() {
  for (int mode = 0; mode < 2; ++mode) {
    const int          channels = mode ? 6 : 1, classes = mode ? 4 : 2, time = mode ? 100 : 96;
    nn::ParameterStore p(613), state(617);
    std::vector<float> input(static_cast<std::size_t>(time * channels));
    for (int t = 0; t < time; ++t)
      for (int c = 0; c < channels; ++c) input[t * channels + c] = static_cast<float>(std::sin(t * .13 + c * .7));
    const auto network = sensor::SensorCnn(p, state, nn::from_vector(input, {1, time, channels}), {channels, classes, 64});
    const auto norms = sentence::SquaredNorms(nn::grad(nn::sum(nn::square(network.logits)), p.parameters()));
    int        length = time;
    for (int stage = 0; stage < 3; ++stage) {
      length = (length + 1) / 2;
      check::Require(network.stages[stage].shape() == nn::Shape{1, length, 64 * (1 << stage)}, "full temporal downsampling/channel stage shapes");
    }
    check::Require(network.logits.shape() == nn::Shape{1, classes} && p.count() == (mode ? 508484 : 506050) && p.size() == 20 && state.size() == 2,
                   "independent full learned/state count");
    for (double n : norms) check::Require(std::isfinite(n) && n > 0, "every full conv/projection/head gradient finite/nonzero");
    std::cout << " full " << (mode ? "six-channel sensor" : "ECG") << ":" << p.count()
              << " parameters/20 tensors+2normstate;all stages and learned gradients passed\n";
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    int                   epochs = 3;
    bool                  checks = false;
    std::uint64_t         seed = 587;
    std::filesystem::path root = paths::Dataset("ecg_sensor");
    std::filesystem::path tf = replay::TfModels("ordered_ecg_sensor_cnn_darwin_20261020") / "ecg_sensor_cnn";
    nn::set_num_threads(4);
    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      if (arg.rfind("epochs=", 0) == 0) epochs = std::stoi(arg.substr(7));
      else if (arg.rfind("data=", 0) == 0) root = arg.substr(5);
      else if (arg.rfind("seed=", 0) == 0) seed = std::stoull(arg.substr(5));
      else if (arg.rfind("tf=", 0) == 0) tf = arg.substr(3);
      else if (arg.rfind("threads=", 0) == 0) nn::set_num_threads(std::stoi(arg.substr(8)));
      else if (arg == "checks_only=1") checks = true;
      else if (arg != "checks_only=0") throw std::invalid_argument("expected epochs=N data=PATH checks_only=0|1 seed=N tf=DIR threads=N");
    }
    HostChecks();
    LossMetricChecks();
    FullChecks();
    for (const char* name : {"ECG200", "BasicMotions"})
      if (std::filesystem::exists(tf / name / "model_arrays.npz")) sensor::CompareWithTflow(root / name, tf / name);
    if (!checks) sensor::TrainEcgSensor(root, epochs, seed);
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
