// CMD: just run-nn midi_lstm
// Stateful LSTM note generation trained on original procedural MIDI melodies.
//
// Port of tflow's midi_lstm. Data: tflow's output/datasets/midi_lstm (data.npz:
// 192 procedural melodies rendered to MIDI and read back as 16-note windows of
// 12 pitch classes, input = start token 12 + notes, target = notes; 128 fit /
// 32 validation / 32 test; written by tflow's scripts/fetch_datasets.py).
// Model (common/midi_lstm.hpp): 13-token one-hot embedding [13, 8], one 8-unit
// LSTM, 12-way head; mean next-note CE, Adam .01, batch 8, three epochs (48
// updates); 756 parameters.
// The port first replays tflow's recorded run (output/models/midi_lstm,
// skipped if absent) from TF's initial weights: initial outputs (logits, all
// states, loss and the loss gradient w.r.t. the one-hot input), first-batch
// gradients, weights after the first update, the validation history and the
// final weights; from TF's final weights it reproduces TF's test outputs,
// final gradients and both seeded samples bit-for-bit (notes). The own run
// then checks restore, the standalone one-step cell against the unroll at
// every position, seeded repeat sampling and immutability.
// Checkpoints are .npz (final.npz); sample MIDI files and piano_roll.png are
// not written (the notes, uniforms, logits and states are in sample08.npz /
// sample12.npz).
// Options: epochs=3 data=PATH (the midi_lstm dataset) tf=PATH (tflow's output/models/midi_lstm).
// Output: output/nn/models/midi_lstm/.

#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>

#include "common/checks.hpp"
#include "common/midi_lstm.hpp"
#include "common/paths.hpp"
#include "common/tf_replay.hpp"

namespace nn = datamunge::nn;
using nn::Tensor;
using replay::Map;
using namespace midi;

namespace {

struct Model {
  nn::ParameterStore p;
  explicit Model(std::uint64_t seed) : p(seed) {
    nn::NoGradGuard guard;
    MidiLstm(p, nn::zeros({1, kSteps, kInputs}));
    check::Require(p.count() == 756, "756parameters");
  }
  static Tensor Loss(const Tensor& logits, const Tensor& target) {
    const Tensor lp = nn::log_softmax(nn::reshape(logits, {-1, kNotes}), -1);
    return nn::neg(nn::mean(nn::sum(nn::mul(lp, nn::one_hot(nn::reshape(target, {-1}), kNotes)), {1})));
  }
  Map Evaluate(const Map& b) {
    Tensor x = nn::one_hot(b.at("input"), kInputs);
    x.requires_grad_(true);
    const Path   path = MidiLstm(p, x);
    const Tensor loss = Loss(path.logits, b.at("target"));
    return {{"logits", path.logits.detach()}, {"loss", loss.detach()}, {"h", path.h.detach()}, {"c", path.c.detach()},
            {"input_gradient", nn::grad(loss, {x})[0]}};
  }
  std::vector<Tensor> Gradients(const Map& b) {
    return nn::grad(Loss(MidiLstm(p, nn::one_hot(b.at("input"), kInputs)).logits, b.at("target")), p.parameters());
  }
  /// The actual previous note only.
  Step StepOnce(const Tensor& token, const Tensor& h, const Tensor& c) {
    nn::NoGradGuard guard;
    return MidiLstmStep(p, nn::one_hot(token, kInputs), {h, c});
  }
  /// 32 notes from the start token, inverse-CDF draws u = (mt19937() + .5) / 2^32.
  Map Sample(double temperature, std::uint32_t seed) {
    Tensor              t = nn::from_vector(std::vector<std::int32_t>{kStart}), h = nn::zeros({1, kUnits}), c = nn::zeros({1, kUnits});
    std::vector<std::int32_t> notes(32);
    std::vector<double>       uniform(32);
    std::vector<float>        scores(32 * kNotes), hs(32 * kUnits), cs(32 * kUnits);
    std::mt19937              rng(seed);
    for (int j = 0; j < 32; ++j) {
      const auto out = StepOnce(t, h, c);
      h              = out.state.h;
      c              = out.state.c;
      const auto lg  = out.logits.to_vector<float>();
      double     maximum = -INFINITY, mass = 0;
      for (int k = 0; k < kNotes; ++k) {
        scores[j * kNotes + k] = lg[k];
        maximum                = std::max(maximum, double(lg[k]) / temperature);
      }
      for (int k = 0; k < kNotes; ++k) mass += std::exp(lg[k] / temperature - maximum);
      const double u = (double(rng()) + .5) / 4294967296.;
      uniform[j]     = u;
      double cumulative = 0;
      int    chosen     = kNotes - 1;
      for (int k = 0; k < kNotes; ++k) {
        cumulative += std::exp(lg[k] / temperature - maximum) / mass;
        if (u < cumulative) {
          chosen = k;
          break;
        }
      }
      notes[j] = chosen;
      t        = nn::from_vector(std::vector<std::int32_t>{chosen});
      const auto hv = h.to_vector<float>(), cv = c.to_vector<float>();
      for (int k = 0; k < kUnits; ++k) hs[j * kUnits + k] = hv[k], cs[j * kUnits + k] = cv[k];
    }
    return {{"notes", nn::from_vector(notes)},
            {"uniform", nn::from_vector(uniform)},
            {"logits", nn::from_vector(scores, {32, kNotes})},
            {"h", nn::from_vector(hs, {32, kUnits})},
            {"c", nn::from_vector(cs, {32, kUnits})}};
  }
};

std::vector<double> Train(Model& m, const Map& fit, const Map& validation, int epochs, std::ofstream* batches,
                          const std::filesystem::path* root, double* first_after = nullptr, const std::filesystem::path* tf = nullptr) {
  nn::Optimizer       opt(m.p, .01, replay::Adam());
  nn::BatchOrder      order(128, 8, 44907);
  std::vector<double> history;
  int                 updates = 0;
  for (int e = 1; e <= epochs; ++e) {
    order.begin_epoch();
    for (;;) {
      const auto rows = order.next();
      if (rows.empty()) break;
      const auto b = replay::Rows(fit, rows);
      if (!updates && root) {
        nn::save_npz((*root / "first_batch.npz").string(), replay::ToNamed(b));
        nn::save_npz((*root / "initial_gradients.npz").string(), replay::ToNamed(replay::Named(m.p, m.Gradients(b))));
      }
      opt.step(m.Gradients(b));
      if (!updates && root) m.p.save((*root / "first_after.npz").string());
      if (!updates && first_after && tf && std::filesystem::exists(*tf / "first_after.npz"))
        *first_after = replay::MaxDiff(replay::Arrays(m.p), *tf / "first_after.npz");
      if (batches)
        for (auto r : rows) *batches << e << '\t' << updates << '\t' << r << '\n';
      ++updates;
    }
    history.push_back(m.Evaluate(validation).at("loss").item<double>());
  }
  return history;
}

void TfParity(const std::filesystem::path& tf, const Map& fit, const Map& validation, const Map& test, int epochs) {
  if (!std::filesystem::exists(tf / "initial_arrays.npz")) {
    std::cout << "(tflow run not found at " << tf << "; TF parity skipped)\n";
    return;
  }
  Model m(0);
  replay::LoadInto(m.p, tf / "initial_arrays.npz");
  const double out = replay::MaxDiff(m.Evaluate(test), tf / "initial_outputs.npz");
  const auto   fb  = replay::ToMap(nn::load_npz((tf / "first_batch.npz").string()));
  const double gd  = replay::MaxDiff(replay::Named(m.p, m.Gradients(fb)), tf / "initial_gradients.npz");
  check::Near(out, 0, 2e-6, "initial outputs and input gradients match TF");
  check::Near(gd, 0, 2e-6, "first-batch parameter gradients match TF");
  double     after = -1;
  const auto hist  = Train(m, fit, validation, epochs, nullptr, nullptr, &after, &tf);
  const auto tfh   = replay::History(tf / "history.tsv");
  double     worst = 0;
  for (std::size_t e = 0; e < hist.size() && e < tfh.size(); ++e) {
    std::cout << "  replay epoch=" << e + 1 << " validationCE=" << hist[e] << " (TF " << tfh[e] << ")\n";
    worst = std::max(worst, std::abs(hist[e] - tfh[e]));
  }
  std::cout << "TF parity from TF's initial weights: initial outputs max diff=" << out << ", gradients max diff=" << gd
            << ", weights after first update max diff=" << after << ", history max diff=" << worst;
  check::Near(after, 0, 1e-4, "weights after the first Adam update match TF");
  if (epochs == 3) {
    const double w = replay::MaxDiff(replay::Arrays(m.p), tf / "model_arrays.npz");
    std::cout << ", final weights max diff=" << w;
    check::Near(worst, 0, 1e-4, "validation history tracks TF's recorded run");
    check::Near(w, 0, 1e-3, "final weights track TF's after 48 Adam updates");
  }
  Model f(0);
  replay::LoadInto(f.p, tf / "model_arrays.npz");
  const double fo = replay::MaxDiff(f.Evaluate(test), tf / "outputs.npz");
  const double fg = replay::MaxDiff(replay::Named(f.p, f.Gradients(fb)), tf / "final_gradients.npz");
  int          same = 0;
  for (auto [file, temperature, seed] : {std::tuple{"sample08.npz", .8, 44911u}, std::tuple{"sample12.npz", 1.2, 44913u}}) {
    const Map s = f.Sample(temperature, seed), ref = replay::ToMap(nn::load_npz((tf / file).string()));
    same += replay::SameBits(s.at("notes"), ref.at("notes")) && replay::SameBits(s.at("uniform"), ref.at("uniform"));
    check::Near(replay::MaxDiff(s, ref), 0, 2e-6, std::string("seeded sample matches TF's ") + file);
  }
  std::cout << "\nfrom TF's final weights: test outputs max diff=" << fo << ", final gradients max diff=" << fg
            << ", seeded samples with identical notes " << same << "/2\n";
  check::Near(fo, 0, 1e-5, "TF's trained weights reproduce TF's test outputs");
  check::Near(fg, 0, 2e-6, "TF's trained weights reproduce TF's final gradients");
  check::Require(same == 2, "TF's trained weights reproduce TF's sampled notes");
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const auto            start  = std::chrono::steady_clock::now();
    int                   epochs = 3;
    std::filesystem::path data = paths::Dataset("midi_lstm"), tf = replay::TfModels("midi_lstm");
    for (int i = 1; i < argc; ++i) {
      const std::string a = argv[i];
      if (a.rfind("epochs=", 0) == 0) epochs = std::stoi(a.substr(7));
      else if (a.rfind("data=", 0) == 0) data = a.substr(5);
      else if (a.rfind("tf=", 0) == 0) tf = a.substr(3);
      else throw std::invalid_argument("epochs=N data=PATH tf=PATH");
    }
    check::Require(epochs > 0 && epochs <= 100, "positive epochs");
    const auto root = paths::OutputDir("models/midi_lstm");
    replay::CopyAssets(data, root, {"data.npz", "manifest.json"});
    const auto all = nn::load_npz((data / "data.npz").string());
    const std::vector<std::string> keys{"input", "target", "source_ids"};
    const Map  fit = replay::Split(all, "fit", keys), validation = replay::Split(all, "validation", keys),
              test = replay::Split(all, "test", keys);
    TfParity(tf, fit, validation, test, epochs);

    Model m(44903);
    m.p.save((root / "initial_arrays.npz").string());
    nn::save_npz((root / "initial_outputs.npz").string(), replay::ToNamed(m.Evaluate(test)));
    std::ofstream history(root / "history.tsv"), batches(root / "batches.tsv");
    history << "epoch\tvalidation_ce\n";
    batches << "epoch\tstep\trow\n";
    const auto hist = Train(m, fit, validation, epochs, &batches, &root);
    for (std::size_t e = 0; e < hist.size(); ++e) {
      history << e + 1 << '\t' << hist[e] << '\n';
      std::cout << "epoch=" << e + 1 << " validationCE=" << hist[e] << std::endl;
    }
    const Map first = replay::ToMap(nn::load_npz((root / "first_batch.npz").string()));
    nn::save_npz((root / "final_gradients.npz").string(), replay::ToNamed(replay::Named(m.p, m.Gradients(first))));
    m.p.save((root / "final.npz").string());
    Model deployed(0);
    deployed.p.load((root / "final.npz").string());
    const Map weights = replay::Arrays(deployed.p);
    check::Require(replay::SameBits(replay::Arrays(m.p), weights), "restore bitexact");
    const Map out = deployed.Evaluate(test);
    for (int t = 0; t < kSteps; ++t) {
      const auto step = deployed.StepOnce(nn::select(test.at("input"), 1, t), nn::select(out.at("h"), 1, t), nn::select(out.at("c"), 1, t));
      check::Near(replay::MaxAbsDiff(step.logits, nn::select(out.at("logits"), 1, t)), 0, 2e-6, "streaming logits");
      check::Near(replay::MaxAbsDiff(step.state.h, nn::select(out.at("h"), 1, t + 1)), 0, 2e-6, "streaming h");
      check::Near(replay::MaxAbsDiff(step.state.c, nn::select(out.at("c"), 1, t + 1)), 0, 2e-6, "streaming c");
    }
    const Map sample = deployed.Sample(.8, 44911);
    check::Require(replay::SameBits(sample.at("notes"), deployed.Sample(.8, 44911).at("notes")), "seeded repeat sampling");
    nn::save_npz((root / "sample08.npz").string(), replay::ToNamed(sample));
    nn::save_npz((root / "sample12.npz").string(), replay::ToNamed(deployed.Sample(1.2, 44913)));
    nn::save_npz((root / "outputs.npz").string(), replay::ToNamed(out));
    nn::save_npz((root / "model_arrays.npz").string(), replay::ToNamed(weights));
    check::Require(replay::SameBits(replay::Arrays(deployed.p), weights), "immutable generation");
    const double  ce = out.at("loss").item<double>();
    std::ofstream metrics(root / "metrics.json");
    metrics << "{\"parameters\":756,\"epochs\":" << epochs << ",\"updates\":" << epochs * 16 << ",\"test_ce\":" << ce << "}\n";
    std::cout << "testCE=" << ce << ";restore/streaming/seeded/immutable PASS\n(TF recorded: testCE=1.4281)\nwall "
              << std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() << " s\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
