// CMD: just run-nn bahdanau
// Additive attention encoder-decoder reversal and addition, three epochs per task.
//
// Port of tflow's bahdanau. Data: tflow's output/datasets/seq2seq/{reversal,
// addition} (data.npz: synthetic 14-token tasks, 128 fit / 32 validation / 32
// test pairs of at most 6 tokens with teacher (BOS-shifted) and target (EOS)
// sequences and their lengths; written by tflow's scripts/fetch_datasets.py).
// A shared 8-D embedding, a length-masked 8-unit encoder LSTM, Bahdanau
// additive attention queried by the previous decoder state and an 8-unit
// decoder LSTM fed [embedding, context]; masked mean token CE, Adam .01,
// batch 8, three epochs (48 updates) per task; 1726 parameters.
// For each task: when tflow's recorded run (output/models/bahdanau/<task>) is
// present, the port first replays it from TF's initial weights (initial
// outputs with all seven BPTT partials, first-batch forward and gradients,
// history, final weights and greedy decodes). The own-initialisation run then
// checks restore, the standalone one-step decoder against the teacher-forced
// unroll, deterministic label-free greedy decoding (teacher/target/lengths
// cannot leak), padding invariance of every output and partial, the empty
// source and immutability. The attention figure (attention.png) is not
// reproduced; the greedy attention weights are saved in decoded.npz.
// Options: epochs=3 data=PATH (the seq2seq dataset root) tf=PATH (tflow's output/models/bahdanau).
// Output: output/nn/models/bahdanau/<task>/.

#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>

#include "common/bahdanau.hpp"
#include "common/checks.hpp"
#include "common/paths.hpp"
#include "common/seq2seq.hpp"
#include "common/tf_replay.hpp"

namespace nn = datamunge::nn;
using nn::Tensor;
using replay::Map;
using namespace bahdanau;

namespace {

Tensor Zero(std::int64_t n) { return nn::zeros({n, kUnits}); }

struct Model {
  nn::ParameterStore p;
  explicit Model(std::uint64_t seed) : p(seed) {
    nn::NoGradGuard guard;
    const Tensor    one = nn::one_hot(nn::zeros({1, kSteps}, nn::DType::Int32), kVocab), len = nn::ones({1}, nn::DType::Int32);
    Bahdanau(p, one, one, len, len, Zero(1), Zero(1));
    check::Require(p.count() == 1726 && p.size() == 13, "1726SharedEmbeddingIndependentEncoderDecoderAndHead");
  }
  Map Evaluate(const Map& b) {
    const auto n = b.at("input").shape()[0];
    Tensor     x = nn::one_hot(b.at("input"), kVocab), d = nn::one_hot(b.at("teacher"), kVocab), ih = Zero(n), ic = Zero(n);
    for (Tensor* t : {&x, &d, &ih, &ic}) t->requires_grad_(true);
    const Path   path = Bahdanau(p, x, d, b.at("input_lengths"), b.at("output_lengths"), ih, ic);
    const Tensor loss = seq2seq::TokenCE(path.logits, b.at("target"), b.at("output_lengths"));
    nn::GradOptions o;
    o.allow_unused = true;
    const std::vector<Tensor> wrt{x, d, ih, ic, path.bridge_h, path.bridge_c, path.memory};
    const auto g = nn::grad(std::vector<Tensor>{loss}, wrt, {}, o);
    Map        out{{"loss", loss.detach()},         {"logits", path.logits.detach()},       {"encoder_h", path.encoder_h.detach()},
                   {"encoder_c", path.encoder_c.detach()}, {"decoder_h", path.decoder_h.detach()}, {"decoder_c", path.decoder_c.detach()},
                   {"attention", path.attention.detach()}, {"contexts", path.contexts.detach()},   {"memory", path.memory.detach()}};
    for (std::size_t i = 0; i < g.size(); ++i)
      out["partial" + std::to_string(i)] = g[i].defined() ? g[i] : nn::zeros_like(wrt[i]).detach();
    return out;
  }
  std::vector<Tensor> Gradients(const Map& b) {
    const auto n    = b.at("input").shape()[0];
    const Path path = Bahdanau(p, nn::one_hot(b.at("input"), kVocab), nn::one_hot(b.at("teacher"), kVocab), b.at("input_lengths"),
                               b.at("output_lengths"), Zero(n), Zero(n));
    return nn::grad(seq2seq::TokenCE(path.logits, b.at("target"), b.at("output_lengths")), p.parameters());
  }
  /// One decoder step from owned state (tflow's step_* placeholders).
  seq2seq::DecoderStep StepOnce(const Tensor& token, const Tensor& h, const Tensor& c, const Tensor& memory, const Tensor& lengths) {
    nn::NoGradGuard guard;
    const Step      s = BahdanauStep(p, seq2seq::Embedding(p, nn::one_hot(token, kVocab), kVocab), {h, c}, memory, lengths);
    return {s.logits, s.state.h, s.state.c, s.weights};
  }
  /// Label-free greedy decoding: only the source ids and lengths are read.
  Tensor Greedy(const Map& b, Tensor* trace = nullptr) {
    nn::NoGradGuard guard;
    const auto      n   = b.at("input").shape()[0];
    const Tensor    len = b.at("input_lengths");
    const auto      enc = seq2seq::Encode(p, seq2seq::Embedding(p, nn::one_hot(b.at("input"), kVocab), kVocab), len, Zero(n), Zero(n));
    const Tensor    memory = enc.memory;
    return seq2seq::Greedy([&](const Tensor& t, const Tensor& h, const Tensor& c) { return StepOnce(t, h, c, memory, len); },
                           enc.h.back(), enc.c.back(), trace);
  }
};

std::vector<double> Train(Model& m, const Map& fit, const Map& validation, int epochs, std::ofstream* batches,
                          const std::filesystem::path* root) {
  nn::Optimizer       opt(m.p, .01, replay::Adam());
  nn::BatchOrder      order(128, 8, 44107);
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
        nn::save_npz((*root / "first_forward.npz").string(), replay::ToNamed(m.Evaluate(b)));
      }
      opt.step(m.Gradients(b));
      if (!updates && root) m.p.save((*root / "first_after.npz").string());
      if (batches)
        for (auto r : rows) *batches << e << '\t' << updates << '\t' << r << '\n';
      ++updates;
    }
    history.push_back(m.Evaluate(validation).at("loss").item<double>());
  }
  return history;
}

void TfParity(const std::string& task, const std::filesystem::path& tf, const std::vector<Map>& pools, int epochs) {
  if (!std::filesystem::exists(tf / "initial_arrays.npz")) {
    std::cout << " (tflow run not found at " << tf << "; TF parity skipped)\n";
    return;
  }
  Model m(0);
  replay::LoadInto(m.p, tf / "initial_arrays.npz");
  const double out = replay::MaxDiff(m.Evaluate(pools[2]), tf / "initial_outputs.npz");
  const auto   fb  = replay::ToMap(nn::load_npz((tf / "first_batch.npz").string()));
  const double fwd = replay::MaxDiff(m.Evaluate(fb), tf / "first_forward.npz");
  const double gd  = replay::MaxDiff(replay::Named(m.p, m.Gradients(fb)), tf / "initial_gradients.npz");
  check::Near(std::max(out, fwd), 0, 2e-6, "initial outputs, attention and all seven BPTT partials match TF");
  check::Near(gd, 0, 2e-6, "first-batch parameter gradients match TF");
  const auto hist = Train(m, pools[0], pools[1], epochs, nullptr, nullptr);
  const auto tfh  = replay::History(tf / "history.tsv");
  double     worst = 0;
  for (std::size_t e = 0; e < hist.size() && e < tfh.size(); ++e) {
    std::cout << "  replay " << task << " epoch=" << e + 1 << " validationTokenCE=" << hist[e] << " (TF " << tfh[e] << ")\n";
    worst = std::max(worst, std::abs(hist[e] - tfh[e]));
  }
  std::cout << " TF parity from TF's initial weights: initial outputs/partials max diff=" << out << ", first-batch forward max diff="
            << fwd << ", gradients max diff=" << gd;
  if (epochs == 3) {
    const double w = replay::MaxDiff(replay::Arrays(m.p), tf / "model_arrays.npz");
    const auto   g = m.Greedy(pools[2]).to_vector<int>(), tg = replay::Get(nn::load_npz((tf / "decoded.npz").string()), "greedy").to_vector<int>();
    int          same = 0;
    for (std::size_t i = 0; i < g.size(); ++i) same += g[i] == tg[i];
    std::cout << ", final weights max diff=" << w << ", greedy tokens identical " << same << "/" << g.size();
    check::Near(worst, 0, 1e-4, "validation history tracks TF's recorded run");
    check::Near(w, 0, 1e-3, "final weights track TF's after 48 Adam updates");
  }
  std::cout << "\n";
}

void Run(const std::string& task, const std::filesystem::path& data, const std::filesystem::path& tf, int epochs) {
  const auto root = paths::OutputDir("models/bahdanau/" + task);
  replay::CopyAssets(data / task, root, {"data.npz", "manifest.json"});
  const auto pools = seq2seq::LoadPools(data / task / "data.npz");
  const Map& test  = pools[2];
  TfParity(task, tf / task, pools, epochs);

  Model m(44203);
  m.p.save((root / "initial_arrays.npz").string());
  nn::save_npz((root / "initial_outputs.npz").string(), replay::ToNamed(m.Evaluate(test)));
  std::ofstream history(root / "history.tsv"), batches(root / "batches.tsv");
  batches << "epoch\tstep\trow\n";
  history << "epoch\tvalidation_token_ce\n";
  const auto hist = Train(m, pools[0], pools[1], epochs, &batches, &root);
  for (std::size_t e = 0; e < hist.size(); ++e) {
    history << e + 1 << '\t' << hist[e] << '\n';
    std::cout << task << " epoch=" << e + 1 << " validationTokenCE=" << hist[e] << std::endl;
  }
  const Map first = replay::ToMap(nn::load_npz((root / "first_batch.npz").string()));
  nn::save_npz((root / "final_gradients.npz").string(), replay::ToNamed(replay::Named(m.p, m.Gradients(first))));
  m.p.save((root / "final.npz").string());
  Model deployed(0);
  deployed.p.load((root / "final.npz").string());
  const Map weights = replay::Arrays(deployed.p);
  check::Require(replay::SameBits(replay::Arrays(m.p), weights), "bahdanauRestoreBitexact");
  const Map out = deployed.Evaluate(test);
  // The standalone one-step decoder reproduces the teacher-forced unroll.
  const auto olen = test.at("output_lengths").to_vector<int>();
  for (int t = 0; t < kSteps; ++t) {
    const auto step = deployed.StepOnce(nn::select(test.at("teacher"), 1, t), nn::select(out.at("decoder_h"), 1, t),
                                        nn::select(out.at("decoder_c"), 1, t), out.at("memory"), test.at("input_lengths"));
    const auto sl = step.logits.to_vector<float>(), sh = step.h.to_vector<float>(), sc = step.c.to_vector<float>();
    const auto ol = nn::select(out.at("logits"), 1, t).to_vector<float>(), oh = nn::select(out.at("decoder_h"), 1, t + 1).to_vector<float>(),
               oc = nn::select(out.at("decoder_c"), 1, t + 1).to_vector<float>();
    for (int b = 0; b < 32; ++b)
      if (t < olen[b]) {
        for (int k = 0; k < kVocab; ++k) check::Near(sl[b * kVocab + k], ol[b * kVocab + k], 2e-6, "nativeStreamingTeacherLogits");
        for (int k = 0; k < kUnits; ++k) {
          check::Near(sh[b * kUnits + k], oh[b * kUnits + k], 2e-6, "nativeStreamingTeacherH");
          check::Near(sc[b * kUnits + k], oc[b * kUnits + k], 2e-6, "nativeStreamingTeacherC");
        }
      }
  }
  Tensor       greedy_attention;
  const Tensor greedy = deployed.Greedy(test, &greedy_attention);
  check::Require(replay::SameBits(greedy, deployed.Greedy(test)), "deterministicGreedy");
  {
    Map corrupt = replay::Rows(test, {0, 1});
    corrupt.at("teacher")        = nn::full_like(corrupt.at("teacher"), 13);
    corrupt.at("target")         = nn::full_like(corrupt.at("target"), 13);
    corrupt.at("output_lengths") = nn::zeros_like(corrupt.at("output_lengths"));
    check::Require(replay::SameBits(deployed.Greedy(replay::Rows(test, {0, 1})), deployed.Greedy(corrupt)),
                   "greedyCannotUseTeacherOrGoldLengthOrTarget");
  }
  Map padded = replay::Rows(test, {0, 1});
  {
    auto       in = padded.at("input").to_vector<std::int32_t>(), te = padded.at("teacher").to_vector<std::int32_t>();
    const auto il = padded.at("input_lengths").to_vector<int>(), ol = padded.at("output_lengths").to_vector<int>();
    for (int b = 0; b < 2; ++b)
      for (int t = 0; t < kSteps; ++t) {
        if (t >= il[b]) in[b * kSteps + t] = 13;
        if (t >= ol[b]) te[b * kSteps + t] = 13;
      }
    padded.at("input")   = nn::from_vector(in, {2, kSteps});
    padded.at("teacher") = nn::from_vector(te, {2, kSteps});
  }
  const Map regular = deployed.Evaluate(replay::Rows(test, {0, 1})), padout = deployed.Evaluate(padded);
  check::Require(replay::SameBits(regular, padout), "allSeqPathsAndLossPartialsPaddingInvariant");
  nn::save_npz((root / "padding_inputs.npz").string(), replay::ToNamed(padded));
  nn::save_npz((root / "padding_outputs.npz").string(), replay::ToNamed(padout));
  Map empty = replay::Rows(test, {0, 1});
  empty.at("input_lengths") = nn::zeros({2}, nn::DType::Int32);
  const Map eo = deployed.Evaluate(empty);
  for (auto k : {"encoder_h", "encoder_c", "partial0"})
    for (float v : eo.at(k).to_vector<float>()) check::Require(v == 0, "emptyEncoderZeroStatesAndPartials");
  check::Require(eo.at("loss").item<double>() > 0, "emptyEncoderStillHasDecoderLoss");
  nn::save_npz((root / "empty_inputs.npz").string(), replay::ToNamed(empty));
  nn::save_npz((root / "empty_outputs.npz").string(), replay::ToNamed(eo));
  nn::save_npz((root / "outputs.npz").string(), replay::ToNamed(out));
  nn::save_npz((root / "model_arrays.npz").string(), replay::ToNamed(weights));
  nn::save_npz((root / "decoded.npz").string(), {{"greedy", greedy}, {"attention", greedy_attention}});
  int        tokens = 0, correct = 0, exact = 0;
  const auto g = greedy.to_vector<int>(), target = test.at("target").to_vector<int>(), input = test.at("input").to_vector<int>(),
             src = test.at("source_ids").to_vector<int>();
  const auto    lg = out.at("logits").to_vector<float>();
  std::ofstream predictions(root / "predictions.tsv");
  predictions << "source_id\tinput_ids\tgold_ids\tgreedy_ids\n";
  for (int b = 0; b < 32; ++b) {
    bool match = true;
    for (int t = 0; t < kSteps; ++t) {
      match &= g[b * kSteps + t] == target[b * kSteps + t];
      if (t < olen[b]) {
        const float* row = lg.data() + (b * kSteps + t) * kVocab;
        correct += std::max_element(row, row + kVocab) - row == target[b * kSteps + t];
        ++tokens;
      }
    }
    exact += match;
    predictions << src[b];
    for (const auto* a : {&input, &target, &g}) {
      predictions << '\t';
      for (int t = 0; t < kSteps; ++t) predictions << (*a)[b * kSteps + t] << ',';
    }
    predictions << '\n';
  }
  const double  ce = out.at("loss").item<double>();
  std::ofstream metrics(root / "metrics.json");
  metrics << "{\"parameters\":1726,\"epochs\":" << epochs << ",\"updates\":" << epochs * 16 << ",\"test_token_ce\":" << ce
          << ",\"teacher_token_accuracy\":" << double(correct) / tokens << ",\"greedy_exact_accuracy\":" << exact / 32.0 << "}\n";
  check::Require(replay::SameBits(replay::Arrays(deployed.p), weights), "bahdanauInferenceImmutable");
  std::cout << task << " testTokenCE=" << ce << " teacherTokenAccuracy=" << double(correct) / tokens << " greedyExact=" << exact
            << "/32; padding/empty/restore/labelFree/immutable PASS\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const auto            start = std::chrono::steady_clock::now();
    int                   epochs = 3;
    std::filesystem::path data = paths::Dataset("seq2seq"), tf = replay::TfModels("bahdanau");
    for (int i = 1; i < argc; ++i) {
      const std::string a = argv[i];
      if (a.rfind("epochs=", 0) == 0) epochs = std::stoi(a.substr(7));
      else if (a.rfind("data=", 0) == 0) data = a.substr(5);
      else if (a.rfind("tf=", 0) == 0) tf = a.substr(3);
      else throw std::invalid_argument("epochs=N data=PATH tf=PATH");
    }
    check::Require(epochs > 0 && epochs <= 100, "positiveEpochs");
    for (const std::string task : {"reversal", "addition"}) Run(task, data, tf, epochs);
    std::cout << "(TF recorded: reversal testTokenCE=2.06084 teacherTokenAccuracy=0.27044 greedyExact=0/32; addition 1.98821 / "
                 "0.40566 / 0)\nwall "
              << std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() << " s\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
