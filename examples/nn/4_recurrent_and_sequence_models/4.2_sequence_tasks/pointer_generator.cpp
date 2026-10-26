// CMD: just run-nn pointer_generator
// Hybrid pointer-generator with repeated source tokens and OOV copying, three epochs.
//
// Port of tflow's pointer_generator. Data: tflow's output/datasets/
// pointer_generator/copy (data.npz: synthetic copy task, 128 fit / 32
// validation / 32 test pairs of at most 6 tokens; ids 14..19 are source-only
// OOVs that can only be produced by copying; written by tflow's
// scripts/fetch_datasets.py). Model (common/pointer_generator.hpp): shared 8-D
// embedding of the base ids min(id, 13), length-masked 8-unit encoder LSTM,
// Bahdanau attention queried by the previous decoder state, decoder LSTM fed
// [embedding, context], a learned gate mixing the 14-way generator (padded to
// the 20-id extended dictionary) with the attention-weighted copy
// distribution, logits = log(p + 1e-9); masked mean token CE, Adam .01, batch
// 8, three epochs (48 updates); 1751 parameters in 15 tensors.
// The port first replays tflow's recorded run (output/models/
// pointer_generator/copy, skipped if absent) from TF's initial weights:
// initial outputs with all seven BPTT partials, first-batch forward and
// gradients, history, final weights and greedy decodes. The
// own-initialisation run then checks restore, the standalone one-step decoder
// against the teacher-forced unroll, deterministic label-free greedy decoding,
// padding invariance of every output and partial, the empty source (generator
// only) and immutability (common/seq2seq_example.hpp). Checkpoints are .npz
// (final.npz); the attention figure is not reproduced (the greedy attention
// weights are saved in decoded.npz).
// Options: epochs=3 data=PATH (the pointer_generator dataset root) tf=PATH (tflow's output/models/pointer_generator).
// Output: output/nn/models/pointer_generator/copy/.

#include "common/pointer_generator.hpp"
#include "common/seq2seq_example.hpp"

// Experiment workflow from examples/nn/common/seq2seq_example.hpp.
namespace seq2seq_example {
using namespace ::seq2seq_example;
template <class Model>
std::vector<double> Train(Model& m, const Map& fit, const Map& validation, int epochs, std::ofstream* batches,
                          const std::filesystem::path* root, const std::function<void()>& after_first = {}) {
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
      if (!updates && after_first) after_first();
      if (batches)
        for (auto r : rows) *batches << e << '\t' << updates << '\t' << r << '\n';
      ++updates;
    }
    history.push_back(m.Evaluate(validation).at("loss").template item<double>());
  }
  return history;
}
}  // namespace seq2seq_example

// Experiment workflow from examples/nn/common/seq2seq_example.hpp.
namespace seq2seq_example {
using namespace ::seq2seq_example;
template <class Model>
Parity TfParity(const Spec& spec, const std::string& task, const std::filesystem::path& tf, const std::vector<Map>& pools,
                int epochs) {
  Parity r;
  if (!std::filesystem::exists(tf / "initial_arrays.npz")) {
    std::cout << " (tflow run not found at " << tf << "; TF parity skipped)\n";
    return r;
  }
  r.ran = true;
  Model m(0);
  replay::LoadInto(m.p, tf / "initial_arrays.npz");
  r.outputs   = replay::MaxDiff(m.Evaluate(pools[2]), tf / "initial_outputs.npz");
  const auto fb = replay::ToMap(nn::load_npz((tf / "first_batch.npz").string()));
  r.forward   = replay::MaxDiff(m.Evaluate(fb), tf / "first_forward.npz");
  r.gradients = replay::MaxDiff(replay::Named(m.p, m.Gradients(fb)), tf / "initial_gradients.npz");
  check::Near(std::max(r.outputs, r.forward), 0, 2e-6, "initial outputs and all BPTT partials match TF");
  check::Near(r.gradients, 0, 2e-6, "first-batch parameter gradients match TF");
  const auto hist = Train(m, pools[0], pools[1], epochs, nullptr, nullptr, [&] {
    if (std::filesystem::exists(tf / "first_after.npz")) {
      r.first_after = replay::MaxDiff(replay::Arrays(m.p), tf / "first_after.npz");
    }
  });
  // Adam's first step is lr g / (|g| + eps): parameters whose gradient is
  // ~1e-8 (comparable to eps) move by a rounding-sensitive fraction of lr.
  if (r.first_after >= 0) check::Near(r.first_after, 0, 1e-4, "weights after the first Adam update match TF");
  const auto tfh  = replay::History(tf / "history.tsv");
  for (std::size_t e = 0; e < hist.size() && e < tfh.size(); ++e) {
    std::cout << "  replay " << task << " epoch=" << e + 1 << " validationTokenCE=" << hist[e] << " (TF " << tfh[e] << ")\n";
    r.history = std::max(r.history, std::abs(hist[e] - tfh[e]));
  }
  std::cout << " TF parity from TF's initial weights: initial outputs/partials max diff=" << r.outputs
            << ", first-batch forward max diff=" << r.forward << ", gradients max diff=" << r.gradients;
  if (epochs == 3) {
    r.weights    = replay::MaxDiff(replay::Arrays(m.p), tf / "model_arrays.npz");
    const auto g = m.Greedy(pools[2], nullptr).template to_vector<int>();
    const auto tg = replay::Get(nn::load_npz((tf / "decoded.npz").string()), "greedy").template to_vector<int>();
    r.greedy_same = 0;
    r.greedy_total = int(g.size());
    for (std::size_t i = 0; i < g.size(); ++i) r.greedy_same += g[i] == tg[i];
    std::cout << ", history max diff=" << r.history << ", final weights max diff=" << r.weights << ", greedy tokens identical "
              << r.greedy_same << "/" << g.size();
    check::Near(r.history, 0, spec.history_tol, "validation history tracks TF's recorded run");
    check::Near(r.weights, 0, spec.weights_tol, "final weights track TF's after 48 Adam updates");
  }
  // Independently of the trajectory: TF's own final weights give TF's test
  // outputs (all partials) and first-batch gradients.
  if (std::filesystem::exists(tf / "model_arrays.npz") && std::filesystem::exists(tf / "final_gradients.npz")) {
    Model f(0);
    replay::LoadInto(f.p, tf / "model_arrays.npz");
    r.final_outputs   = replay::MaxDiff(f.Evaluate(pools[2]), tf / "outputs.npz");
    r.final_gradients = replay::MaxDiff(replay::Named(f.p, f.Gradients(fb)), tf / "final_gradients.npz");
    std::cout << "\n from TF's final weights: test outputs/partials max diff=" << r.final_outputs
              << ", first-batch gradients max diff=" << r.final_gradients << ", weights after first update max diff=" << r.first_after;
    check::Near(r.final_outputs, 0, 1e-5, "TF's trained weights reproduce TF's test outputs and partials");
    check::Near(r.final_gradients, 0, 1e-5, "TF's trained weights reproduce TF's final gradients");
  }
  std::cout << "\n";
  return r;
}
}  // namespace seq2seq_example

// Experiment workflow from examples/nn/common/seq2seq_example.hpp.
namespace seq2seq_example {
using namespace ::seq2seq_example;
template <class Model>
Result Run(const Spec& spec, const std::string& task, const std::filesystem::path& data, const std::filesystem::path& tf,
           int epochs) {
  const auto root  = paths::OutputDir("models/" + spec.name + "/" + task);
  const auto src   = spec.flat_data ? data : data / task;
  replay::CopyAssets(src, root, spec.assets);
  const auto pools = seq2seq::LoadPools(src / "data.npz");
  const Map& test  = pools[2];
  Result     result;
  result.parity = TfParity<Model>(spec, task, tf / task, pools, epochs);

  Model m(spec.seed);
  check::Require(m.p.count() == spec.params && std::int64_t(m.p.size()) == spec.tensors,
                 std::to_string(spec.params) + "SharedEmbeddingIndependentEncoderDecoderAndHead");
  m.p.save((root / "initial_arrays.npz").string());
  nn::save_npz((root / "initial_outputs.npz").string(), replay::ToNamed(m.Evaluate(test)));
  std::ofstream history(root / "history.tsv"), batches(root / "batches.tsv");
  batches << "epoch\tstep\trow\n";
  history << "epoch\tvalidation_token_ce\n";
  result.history = Train(m, pools[0], pools[1], epochs, &batches, &root);
  for (std::size_t e = 0; e < result.history.size(); ++e) {
    history << e + 1 << '\t' << result.history[e] << '\n';
    std::cout << task << " epoch=" << e + 1 << " validationTokenCE=" << result.history[e] << std::endl;
  }
  const Map first = replay::ToMap(nn::load_npz((root / "first_batch.npz").string()));
  nn::save_npz((root / "final_gradients.npz").string(), replay::ToNamed(replay::Named(m.p, m.Gradients(first))));
  m.p.save((root / "final.npz").string());
  Model deployed(0);
  deployed.p.load((root / "final.npz").string());
  const Map weights = replay::Arrays(deployed.p);
  check::Require(replay::SameBits(replay::Arrays(m.p), weights), spec.name + "RestoreBitexact");
  const Map out = deployed.Evaluate(test);
  // The standalone one-step decoder reproduces the teacher-forced unroll.
  const auto olen = test.at("output_lengths").to_vector<int>();
  const int  V    = spec.vocab;
  const auto n    = test.at("teacher").shape()[0];
  for (int t = 0; t < kSteps; ++t) {
    const auto step = deployed.StepAt(test, out, t);
    const auto sl = step.logits.template to_vector<float>(), sh = step.h.template to_vector<float>(),
               sc = step.c.template to_vector<float>();
    const auto ol = nn::select(out.at("logits"), 1, t).template to_vector<float>(),
               oh = nn::select(out.at("decoder_h"), 1, t + 1).template to_vector<float>(),
               oc = nn::select(out.at("decoder_c"), 1, t + 1).template to_vector<float>();
    for (std::int64_t b = 0; b < n; ++b)
      if (t < olen[b]) {
        for (int k = 0; k < V; ++k) check::Near(sl[b * V + k], ol[b * V + k], 2e-6, "nativeStreamingTeacherLogits");
        for (int k = 0; k < kUnits; ++k) {
          check::Near(sh[b * kUnits + k], oh[b * kUnits + k], 2e-6, "nativeStreamingTeacherH");
          check::Near(sc[b * kUnits + k], oc[b * kUnits + k], 2e-6, "nativeStreamingTeacherC");
        }
      }
  }
  Tensor       greedy_attention;
  const Tensor greedy = deployed.Greedy(test, &greedy_attention);
  check::Require(replay::SameBits(greedy, deployed.Greedy(test, nullptr)), "deterministicGreedy");
  {
    Map corrupt = replay::Rows(test, {0, 1});
    corrupt.at("teacher")        = nn::full_like(corrupt.at("teacher"), spec.corrupt);
    corrupt.at("target")         = nn::full_like(corrupt.at("target"), spec.corrupt);
    corrupt.at("output_lengths") = nn::zeros_like(corrupt.at("output_lengths"));
    check::Require(replay::SameBits(deployed.Greedy(replay::Rows(test, {0, 1}), nullptr), deployed.Greedy(corrupt, nullptr)),
                   "greedyCannotUseTeacherOrGoldLengthOrTarget");
  }
  Map padded = replay::Rows(test, {0, 1});
  Pad(spec, padded);
  const Map regular = deployed.Evaluate(replay::Rows(test, {0, 1})), padout = deployed.Evaluate(padded);
  check::Require(replay::SameBits(regular, padout), "allSeqPathsAndLossPartialsPaddingInvariant");
  nn::save_npz((root / "padding_inputs.npz").string(), replay::ToNamed(padded));
  nn::save_npz((root / "padding_outputs.npz").string(), replay::ToNamed(padout));
  Map empty = replay::Rows(test, {0, 1});
  empty.at("input_lengths") = nn::zeros({2}, nn::DType::Int32);
  if (spec.pointer) {
    empty.at("target")         = nn::full_like(empty.at("target"), 5);
    empty.at("teacher")        = nn::full_like(empty.at("teacher"), -1);
    empty.at("output_lengths") = nn::ones({2}, nn::DType::Int32);
  }
  const Map eo = deployed.Evaluate(empty);
  for (auto k : {"encoder_h", "encoder_c", "partial0"})
    for (float v : eo.at(k).template to_vector<float>()) check::Require(v == 0, "emptyEncoderZeroStatesAndPartials");
  if (spec.pointer)
    check::Require(eo.at("loss").template item<double>() == 0, "emptyPointerOnlySTOPHasZeroLoss");
  else
    check::Require(eo.at("loss").template item<double>() > 0, "emptyEncoderStillHasDecoderLoss");
  nn::save_npz((root / "empty_inputs.npz").string(), replay::ToNamed(empty));
  nn::save_npz((root / "empty_outputs.npz").string(), replay::ToNamed(eo));
  nn::save_npz((root / "outputs.npz").string(), replay::ToNamed(out));
  nn::save_npz((root / "model_arrays.npz").string(), replay::ToNamed(weights));
  if (spec.attention)
    nn::save_npz((root / "decoded.npz").string(), {{"greedy", greedy}, {"attention", greedy_attention}});
  else
    nn::save_npz((root / "decoded.npz").string(), {{"greedy", greedy}});
  int        tokens = 0, correct = 0, exact = 0;
  const auto g = greedy.template to_vector<int>(), target = test.at("target").template to_vector<int>(),
             ids = test.at("source_ids").template to_vector<int>();
  const auto    lg = out.at("logits").template to_vector<float>();
  std::ofstream predictions(root / "predictions.tsv");
  predictions << (spec.pointer ? "source_id\tgold_indices\tpredicted_indices\n" : "source_id\tinput_ids\tgold_ids\tgreedy_ids\n");
  std::vector<int> input;
  if (!spec.pointer) input = test.at("input").template to_vector<int>();
  for (std::int64_t b = 0; b < n; ++b) {
    bool match = true;
    for (int t = 0; t < kSteps; ++t) {
      const int gold = spec.pointer && t >= olen[b] ? -1 : target[b * kSteps + t];
      match &= g[b * kSteps + t] == gold;
      if (t < olen[b]) {
        const float* row = lg.data() + (b * kSteps + t) * V;
        correct += std::max_element(row, row + V) - row == target[b * kSteps + t];
        ++tokens;
      }
    }
    exact += match;
    predictions << ids[b];
    std::vector<const std::vector<int>*> columns{&target, &g};
    if (!spec.pointer) columns.insert(columns.begin(), &input);
    for (const auto* a : columns) {
      predictions << '\t';
      for (int t = 0; t < kSteps; ++t) predictions << (*a)[b * kSteps + t] << ',';
    }
    predictions << '\n';
  }
  result.test_ce          = out.at("loss").template item<double>();
  result.teacher_accuracy = double(correct) / tokens;
  result.exact            = exact;
  std::ofstream metrics(root / "metrics.json");
  metrics << "{\"parameters\":" << spec.params << ",\"epochs\":" << epochs << ",\"updates\":" << epochs * 16
          << ",\"test_token_ce\":" << result.test_ce << ",\"teacher_token_accuracy\":" << result.teacher_accuracy << ",\""
          << spec.exact_key << "\":" << exact / double(n) << "}\n";
  check::Require(replay::SameBits(replay::Arrays(deployed.p), weights), spec.name + "InferenceImmutable");
  std::cout << task << " testTokenCE=" << result.test_ce << " teacherTokenAccuracy=" << result.teacher_accuracy << " "
            << spec.exact_label << "=" << exact << "/" << n << "; padding/empty/restore/labelFree/immutable PASS\n";
  return result;
}
}  // namespace seq2seq_example

// Experiment workflow from examples/nn/common/seq2seq_example.hpp.
namespace seq2seq_example {
using namespace ::seq2seq_example;
template <class Model>
int Main(const Spec& spec, int argc, char** argv) {
  try {
    const auto            start  = std::chrono::steady_clock::now();
    int                   epochs = 3;
    std::filesystem::path data = paths::Dataset(spec.dataset), tf = replay::TfModels(spec.name);
    for (int i = 1; i < argc; ++i) {
      const std::string a = argv[i];
      if (a.rfind("epochs=", 0) == 0) epochs = std::stoi(a.substr(7));
      else if (a.rfind("data=", 0) == 0) data = a.substr(5);
      else if (a.rfind("tf=", 0) == 0) tf = a.substr(3);
      else throw std::invalid_argument("epochs=N data=PATH tf=PATH");
    }
    check::Require(epochs > 0 && epochs <= 100, "positiveEpochs");
    for (const auto& task : spec.tasks) Run<Model>(spec, task, data, tf, epochs);
    std::cout << "(TF recorded: " << spec.tf_recorded << ")\nwall "
              << std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() << " s\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
}  // namespace seq2seq_example



namespace nn = datamunge::nn;
using nn::Tensor;
using replay::Map;
using seq2seq_example::Zero;
using namespace pointer_generator;

namespace {

struct Model {
  nn::ParameterStore p;
  explicit Model(std::uint64_t seed) : p(seed) {
    nn::NoGradGuard guard;
    const Tensor ids = nn::zeros({1, kSteps}, nn::DType::Int32), one = nn::one_hot(ids, kBase), len = nn::ones({1}, nn::DType::Int32);
    PointerGenerator(p, one, one, len, len, Zero(1), Zero(1), nn::one_hot(ids, kExtended));
  }
  Map Evaluate(const Map& b) {
    const auto n = b.at("input").shape()[0];
    Tensor     x = nn::one_hot(Base(b.at("input")), kBase), d = nn::one_hot(Base(b.at("teacher")), kBase), ih = Zero(n), ic = Zero(n);
    for (Tensor* t : {&x, &d, &ih, &ic}) t->requires_grad_(true);
    const Path path = PointerGenerator(p, x, d, b.at("input_lengths"), b.at("output_lengths"), ih, ic,
                                       nn::one_hot(b.at("input"), kExtended));
    return seq2seq_example::Fields(seq2seq::TokenCE(path.logits, b.at("target"), b.at("output_lengths")),
                                   {{"logits", path.logits},
                                    {"encoder_h", path.encoder_h},
                                    {"encoder_c", path.encoder_c},
                                    {"decoder_h", path.decoder_h},
                                    {"decoder_c", path.decoder_c},
                                    {"attention", path.attention},
                                    {"contexts", path.contexts},
                                    {"memory", path.memory}},
                                   {x, d, ih, ic, path.bridge_h, path.bridge_c, path.memory});
  }
  std::vector<Tensor> Gradients(const Map& b) {
    const auto n    = b.at("input").shape()[0];
    const Path path = PointerGenerator(p, nn::one_hot(Base(b.at("input")), kBase), nn::one_hot(Base(b.at("teacher")), kBase),
                                       b.at("input_lengths"), b.at("output_lengths"), Zero(n), Zero(n),
                                       nn::one_hot(b.at("input"), kExtended));
    return nn::grad(seq2seq::TokenCE(path.logits, b.at("target"), b.at("output_lengths")), p.parameters());
  }
  /// One step from owned state, the encoder memory and the source ids.
  seq2seq::DecoderStep StepOnce(const Tensor& token, const Tensor& h, const Tensor& c, const Tensor& memory, const Tensor& source,
                                const Tensor& lengths) {
    nn::NoGradGuard guard;
    const Step s = PointerGeneratorStep(p, seq2seq::Embedding(p, nn::one_hot(Base(token), kBase), kBase), {h, c}, memory, lengths,
                                        nn::one_hot(source, kExtended));
    return {s.logits, s.state.h, s.state.c, s.weights};
  }
  seq2seq::DecoderStep StepAt(const Map& b, const Map& out, int t) {
    return StepOnce(nn::select(b.at("teacher"), 1, t), nn::select(out.at("decoder_h"), 1, t), nn::select(out.at("decoder_c"), 1, t),
                    out.at("memory"), b.at("input"), b.at("input_lengths"));
  }
  /// Label-free greedy decoding: only the source ids and lengths are read.
  Tensor Greedy(const Map& b, Tensor* trace) {
    nn::NoGradGuard guard;
    const auto      n   = b.at("input").shape()[0];
    const Tensor    len = b.at("input_lengths"), source = b.at("input");
    const auto enc = seq2seq::Encode(p, seq2seq::Embedding(p, nn::one_hot(Base(source), kBase), kBase), len, Zero(n), Zero(n));
    const Tensor memory = enc.memory;
    return seq2seq::Greedy([&](const Tensor& t, const Tensor& h, const Tensor& c) { return StepOnce(t, h, c, memory, source, len); },
                           enc.h.back(), enc.c.back(), trace);
  }
};

}  // namespace

int main(int argc, char** argv) {
  seq2seq_example::Spec spec;
  spec.name        = "pointer_generator";
  spec.dataset     = "pointer_generator";
  spec.tasks       = {"copy"};
  spec.params      = 1751;
  spec.tensors     = 15;
  spec.vocab       = kExtended;
  spec.seed        = 44603;
  spec.weights_tol = 5e-3;  // log(p + 1e-9) amplifies float32 rounding along the 48 updates
  spec.tf_recorded = "copy testTokenCE=0.711544 teacherTokenAccuracy=0.710059 greedyExact=2/32";
  return seq2seq_example::Main<Model>(spec, argc, argv);
}
