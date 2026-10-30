// CMD: just run-nn luong
// Global general attention encoder-decoder reversal and addition, three epochs per task.
//
// Port of tflow's luong. Data: tflow's output/datasets/seq2seq/{reversal,
// addition} (data.npz: synthetic 14-token tasks, 128 fit / 32 validation / 32
// test pairs of at most 6 tokens with teacher (BOS-shifted) and target (EOS)
// sequences and their lengths; written by tflow's scripts/fetch_datasets.py).
// A shared 8-D embedding, a length-masked 8-unit encoder LSTM, an 8-unit
// decoder LSTM whose CURRENT state queries the encoder memory with Luong
// "general" attention, tanh([context, h] Wc + bc) feeding the head (no input
// feeding); masked mean token CE, Adam .01, batch 8, three epochs (48 updates)
// per task; 1526 parameters in 12 tensors.
// For each task the port first replays tflow's recorded run (output/models/
// luong/<task>, skipped if absent) from TF's initial weights: initial outputs
// with all seven BPTT partials, first-batch forward and gradients, history,
// final weights and greedy decodes. The own-initialisation run then checks
// restore, the standalone one-step decoder against the teacher-forced unroll,
// deterministic label-free greedy decoding, padding invariance of every output
// and partial, the empty source and immutability (common/seq2seq_example.hpp).
// Checkpoints are .npz (final.npz); the attention figure is not reproduced (the
// greedy attention weights are saved in decoded.npz).
// Options: epochs=3 data=PATH (the seq2seq dataset root) tf=PATH (tflow's output/models/luong).
// Output: output/nn/models/luong/<task>/.

#include "common/luong.hpp"

#include "common/seq2seq_example.hpp"

namespace nn = datamunge::nn;
using nn::Tensor;
using replay::Map;
using seq2seq_example::Zero;
using namespace luong;

namespace {

struct Model {
  nn::ParameterStore p;
  explicit Model(std::uint64_t seed) : p(seed) {
    nn::NoGradGuard guard;
    const Tensor    one = nn::one_hot(nn::zeros({1, kSteps}, nn::DType::Int32), kVocab),
                 len    = nn::ones({1}, nn::DType::Int32);
    Luong(p, one, one, len, len, Zero(1), Zero(1));
  }
  Path Forward(const Map& b, const Tensor& x, const Tensor& d, const Tensor& ih, const Tensor& ic) {
    return Luong(p, x, d, b.at("input_lengths"), b.at("output_lengths"), ih, ic);
  }
  Map Evaluate(const Map& b) {
    const auto n = b.at("input").shape()[0];
    Tensor     x = nn::one_hot(b.at("input"), kVocab), d = nn::one_hot(b.at("teacher"), kVocab),
           ih = Zero(n), ic = Zero(n);
    for (Tensor* t : {&x, &d, &ih, &ic})
      t->requires_grad_(true);
    const Path path = Forward(b, x, d, ih, ic);
    return seq2seq_example::Fields(
        seq2seq::TokenCE(path.logits, b.at("target"), b.at("output_lengths")),
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
    const Path path = Forward(
        b,
        nn::one_hot(b.at("input"), kVocab),
        nn::one_hot(b.at("teacher"), kVocab),
        Zero(n),
        Zero(n));
    return nn::grad(
        seq2seq::TokenCE(path.logits, b.at("target"), b.at("output_lengths")), p.parameters());
  }
  seq2seq::DecoderStep StepOnce(
      const Tensor& token,
      const Tensor& h,
      const Tensor& c,
      const Tensor& memory,
      const Tensor& lengths) {
    nn::NoGradGuard guard;
    const Step      s = LuongStep(
        p, seq2seq::Embedding(p, nn::one_hot(token, kVocab), kVocab), {h, c}, memory, lengths);
    return {s.logits, s.state.h, s.state.c, s.weights};
  }
  seq2seq::DecoderStep StepAt(const Map& b, const Map& out, int t) {
    return StepOnce(
        nn::select(b.at("teacher"), 1, t),
        nn::select(out.at("decoder_h"), 1, t),
        nn::select(out.at("decoder_c"), 1, t),
        out.at("memory"),
        b.at("input_lengths"));
  }
  /// Label-free greedy decoding: only the source ids and lengths are read.
  Tensor Greedy(const Map& b, Tensor* trace) {
    nn::NoGradGuard guard;
    const auto      n   = b.at("input").shape()[0];
    const Tensor    len = b.at("input_lengths");
    const auto      enc = seq2seq::Encode(
        p,
        seq2seq::Embedding(p, nn::one_hot(b.at("input"), kVocab), kVocab),
        len,
        Zero(n),
        Zero(n));
    const Tensor memory = enc.memory;
    return seq2seq::Greedy(
        [&](const Tensor& t, const Tensor& h, const Tensor& c) {
          return StepOnce(t, h, c, memory, len);
        },
        enc.h.back(),
        enc.c.back(),
        trace);
  }
};

} // namespace

int main(int argc, char** argv) {
  seq2seq_example::Spec spec;
  spec.name    = "luong";
  spec.dataset = "seq2seq";
  spec.tasks   = {"reversal", "addition"};
  spec.params  = 1526;
  spec.tensors = 12;
  spec.vocab   = kVocab;
  spec.seed    = 44303;
  spec.tf_recorded =
      "reversal testTokenCE=2.10584 teacherTokenAccuracy=0.257862 greedyExact=0/32; addition "
      "1.91345 / 0.40566 / 0";
  using namespace seq2seq_example;
  try {
    const auto            start  = std::chrono::steady_clock::now();
    int                   epochs = 3;
    std::filesystem::path data = paths::Dataset(spec.dataset), tf = replay::TfModels(spec.name);
    for (int i = 1; i < argc; ++i) {
      const std::string a = argv[i];
      if (a.rfind("epochs=", 0) == 0)
        epochs = std::stoi(a.substr(7));
      else if (a.rfind("data=", 0) == 0)
        data = a.substr(5);
      else if (a.rfind("tf=", 0) == 0)
        tf = a.substr(3);
      else
        throw std::invalid_argument("epochs=N data=PATH tf=PATH");
    }
    check::Require(epochs > 0 && epochs <= 100, "positiveEpochs");
    for (const auto& task : spec.tasks) {
      const auto root = paths::OutputDir("models/" + spec.name + "/" + task);
      const auto src  = spec.flat_data ? data : data / task;
      replay::CopyAssets(src, root, spec.assets);
      const auto pools = seq2seq::LoadPools(src / "data.npz");
      const Map& test  = pools[2];
      Result     result;
      result.parity = [&]() -> Parity {
        const auto recorded = tf / task;
        Parity     r;
        if (!std::filesystem::exists(recorded / "initial_arrays.npz")) {
          std::cout << " (tflow run not found at " << recorded << "; TF parity skipped)\n";
          return r;
        }
        r.ran = true;
        Model m(0);
        replay::LoadInto(m.p, recorded / "initial_arrays.npz");
        r.outputs     = replay::MaxDiff(m.Evaluate(pools[2]), recorded / "initial_outputs.npz");
        const auto fb = replay::ToMap(nn::load_npz((recorded / "first_batch.npz").string()));
        r.forward     = replay::MaxDiff(m.Evaluate(fb), recorded / "first_forward.npz");
        r.gradients   = replay::MaxDiff(
            replay::Named(m.p, m.Gradients(fb)), recorded / "initial_gradients.npz");
        check::Near(
            std::max(r.outputs, r.forward),
            0,
            2e-6,
            "initial outputs and all BPTT partials match TF");
        check::Near(r.gradients, 0, 2e-6, "first-batch parameter gradients match TF");
        nn::Optimizer       opt(m.p, .01, replay::Adam());
        nn::BatchOrder      order(128, 8, 44107);
        std::vector<double> hist;
        int                 updates = 0;
        for (int e = 1; e <= epochs; ++e) {
          order.begin_epoch();
          for (;;) {
            const auto rows = order.next();
            if (rows.empty())
              break;
            const auto b = replay::Rows(pools[0], rows);

            const auto n    = b.at("input").shape()[0];
            const Path path = m.Forward(
                b,
                nn::one_hot(b.at("input"), kVocab),
                nn::one_hot(b.at("teacher"), kVocab),
                Zero(n),
                Zero(n));
            const auto gradients = nn::grad(
                seq2seq::TokenCE(path.logits, b.at("target"), b.at("output_lengths")),
                m.p.parameters());

            opt.step(gradients);
            if (!updates && std::filesystem::exists(recorded / "first_after.npz"))
              r.first_after = replay::MaxDiff(replay::Arrays(m.p), recorded / "first_after.npz");
            ++updates;
          }
          hist.push_back(m.Evaluate(pools[1]).at("loss").item<double>());
        }

        // Adam's first step is lr g / (|g| + eps): parameters whose gradient is
        // ~1e-8 (comparable to eps) move by a rounding-sensitive fraction of lr.
        if (r.first_after >= 0)
          check::Near(r.first_after, 0, 1e-4, "weights after the first Adam update match TF");
        const auto tfh = replay::History(recorded / "history.tsv");
        for (std::size_t e = 0; e < hist.size() && e < tfh.size(); ++e) {
          std::cout << "  replay " << task << " epoch=" << e + 1 << " validationTokenCE=" << hist[e]
                    << " (TF " << tfh[e] << ")\n";
          r.history = std::max(r.history, std::abs(hist[e] - tfh[e]));
        }
        std::cout << " TF parity from TF's initial weights: initial outputs/partials max diff="
                  << r.outputs << ", first-batch forward max diff=" << r.forward
                  << ", gradients max diff=" << r.gradients;
        if (epochs == 3) {
          r.weights     = replay::MaxDiff(replay::Arrays(m.p), recorded / "model_arrays.npz");
          const auto g  = m.Greedy(pools[2], nullptr).to_vector<int>();
          const auto tg = replay::Get(nn::load_npz((recorded / "decoded.npz").string()), "greedy")
                              .to_vector<int>();
          r.greedy_same  = 0;
          r.greedy_total = int(g.size());
          for (std::size_t i = 0; i < g.size(); ++i)
            r.greedy_same += g[i] == tg[i];
          std::cout << ", history max diff=" << r.history
                    << ", final weights max diff=" << r.weights << ", greedy tokens identical "
                    << r.greedy_same << "/" << g.size();
          check::Near(
              r.history, 0, spec.history_tol, "validation history tracks TF's recorded run");
          check::Near(
              r.weights, 0, spec.weights_tol, "final weights track TF's after 48 Adam updates");
        }
        // Independently of the trajectory: TF's own final weights give TF's test
        // outputs (all partials) and first-batch gradients.
        if (std::filesystem::exists(recorded / "model_arrays.npz") &&
            std::filesystem::exists(recorded / "final_gradients.npz")) {
          Model f(0);
          replay::LoadInto(f.p, recorded / "model_arrays.npz");
          r.final_outputs   = replay::MaxDiff(f.Evaluate(pools[2]), recorded / "outputs.npz");
          r.final_gradients = replay::MaxDiff(
              replay::Named(f.p, f.Gradients(fb)), recorded / "final_gradients.npz");
          std::cout << "\n from TF's final weights: test outputs/partials max diff="
                    << r.final_outputs << ", first-batch gradients max diff=" << r.final_gradients
                    << ", weights after first update max diff=" << r.first_after;
          check::Near(
              r.final_outputs,
              0,
              1e-5,
              "TF's trained weights reproduce TF's test outputs and partials");
          check::Near(
              r.final_gradients, 0, 1e-5, "TF's trained weights reproduce TF's final gradients");
        }
        std::cout << "\n";
        return r;
      }();

      Model m(spec.seed);
      check::Require(
          m.p.count() == spec.params && std::int64_t(m.p.size()) == spec.tensors,
          std::to_string(spec.params) + "SharedEmbeddingIndependentEncoderDecoderAndHead");
      m.p.save((root / "initial_arrays.npz").string());
      nn::save_npz((root / "initial_outputs.npz").string(), replay::ToNamed(m.Evaluate(test)));
      std::ofstream history(root / "history.tsv"), batches(root / "batches.tsv");
      batches << "epoch\tstep\trow\n";
      history << "epoch\tvalidation_token_ce\n";
      nn::Optimizer  opt(m.p, .01, replay::Adam());
      nn::BatchOrder order(128, 8, 44107);
      int            updates = 0;
      for (int e = 1; e <= epochs; ++e) {
        order.begin_epoch();
        for (;;) {
          const auto rows = order.next();
          if (rows.empty())
            break;
          const auto b = replay::Rows(pools[0], rows);
          if (!updates) {
            nn::save_npz((root / "first_batch.npz").string(), replay::ToNamed(b));
            nn::save_npz(
                (root / "initial_gradients.npz").string(),
                replay::ToNamed(replay::Named(m.p, m.Gradients(b))));
            nn::save_npz((root / "first_forward.npz").string(), replay::ToNamed(m.Evaluate(b)));
          }
          const auto n    = b.at("input").shape()[0];
          const Path path = m.Forward(
              b,
              nn::one_hot(b.at("input"), kVocab),
              nn::one_hot(b.at("teacher"), kVocab),
              Zero(n),
              Zero(n));
          const auto gradients = nn::grad(
              seq2seq::TokenCE(path.logits, b.at("target"), b.at("output_lengths")),
              m.p.parameters());

          opt.step(gradients);
          if (!updates)
            m.p.save((root / "first_after.npz").string());
          for (auto row : rows)
            batches << e << '\t' << updates << '\t' << row << '\n';
          ++updates;
        }
        result.history.push_back(m.Evaluate(pools[1]).at("loss").item<double>());
      }

      for (std::size_t e = 0; e < result.history.size(); ++e) {
        history << e + 1 << '\t' << result.history[e] << '\n';
        std::cout << task << " epoch=" << e + 1 << " validationTokenCE=" << result.history[e]
                  << std::endl;
      }
      const Map first = replay::ToMap(nn::load_npz((root / "first_batch.npz").string()));
      nn::save_npz(
          (root / "final_gradients.npz").string(),
          replay::ToNamed(replay::Named(m.p, m.Gradients(first))));
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
        const auto sl = step.logits.to_vector<float>(), sh = step.h.to_vector<float>(),
                   sc = step.c.to_vector<float>();
        const auto ol = nn::select(out.at("logits"), 1, t).to_vector<float>(),
                   oh = nn::select(out.at("decoder_h"), 1, t + 1).to_vector<float>(),
                   oc = nn::select(out.at("decoder_c"), 1, t + 1).to_vector<float>();
        for (std::int64_t b = 0; b < n; ++b)
          if (t < olen[b]) {
            for (int k = 0; k < V; ++k)
              check::Near(sl[b * V + k], ol[b * V + k], 2e-6, "nativeStreamingTeacherLogits");
            for (int k = 0; k < kUnits; ++k) {
              check::Near(sh[b * kUnits + k], oh[b * kUnits + k], 2e-6, "nativeStreamingTeacherH");
              check::Near(sc[b * kUnits + k], oc[b * kUnits + k], 2e-6, "nativeStreamingTeacherC");
            }
          }
      }
      Tensor       greedy_attention;
      const Tensor greedy = deployed.Greedy(test, &greedy_attention);
      check::Require(
          replay::SameBits(greedy, deployed.Greedy(test, nullptr)), "deterministicGreedy");
      {
        Map corrupt                  = replay::Rows(test, {0, 1});
        corrupt.at("teacher")        = nn::full_like(corrupt.at("teacher"), spec.corrupt);
        corrupt.at("target")         = nn::full_like(corrupt.at("target"), spec.corrupt);
        corrupt.at("output_lengths") = nn::zeros_like(corrupt.at("output_lengths"));
        check::Require(
            replay::SameBits(
                deployed.Greedy(replay::Rows(test, {0, 1}), nullptr),
                deployed.Greedy(corrupt, nullptr)),
            "greedyCannotUseTeacherOrGoldLengthOrTarget");
      }
      Map padded = replay::Rows(test, {0, 1});
      Pad(spec, padded);
      const Map regular = deployed.Evaluate(replay::Rows(test, {0, 1})),
                padout  = deployed.Evaluate(padded);
      check::Require(
          replay::SameBits(regular, padout), "allSeqPathsAndLossPartialsPaddingInvariant");
      nn::save_npz((root / "padding_inputs.npz").string(), replay::ToNamed(padded));
      nn::save_npz((root / "padding_outputs.npz").string(), replay::ToNamed(padout));
      Map empty                 = replay::Rows(test, {0, 1});
      empty.at("input_lengths") = nn::zeros({2}, nn::DType::Int32);
      if (spec.pointer) {
        empty.at("target")         = nn::full_like(empty.at("target"), 5);
        empty.at("teacher")        = nn::full_like(empty.at("teacher"), -1);
        empty.at("output_lengths") = nn::ones({2}, nn::DType::Int32);
      }
      const Map eo = deployed.Evaluate(empty);
      for (auto k : {"encoder_h", "encoder_c", "partial0"})
        for (float v : eo.at(k).to_vector<float>())
          check::Require(v == 0, "emptyEncoderZeroStatesAndPartials");
      if (spec.pointer)
        check::Require(eo.at("loss").item<double>() == 0, "emptyPointerOnlySTOPHasZeroLoss");
      else
        check::Require(eo.at("loss").item<double>() > 0, "emptyEncoderStillHasDecoderLoss");
      nn::save_npz((root / "empty_inputs.npz").string(), replay::ToNamed(empty));
      nn::save_npz((root / "empty_outputs.npz").string(), replay::ToNamed(eo));
      nn::save_npz((root / "outputs.npz").string(), replay::ToNamed(out));
      nn::save_npz((root / "model_arrays.npz").string(), replay::ToNamed(weights));
      if (spec.attention)
        nn::save_npz(
            (root / "decoded.npz").string(), {{"greedy", greedy}, {"attention", greedy_attention}});
      else
        nn::save_npz((root / "decoded.npz").string(), {{"greedy", greedy}});
      std::map<std::pair<int, int>, int> confusion;
      int                                tokens = 0, correct = 0, exact = 0;
      const auto g = greedy.to_vector<int>(), target = test.at("target").to_vector<int>(),
                 ids   = test.at("source_ids").to_vector<int>();
      const auto    lg = out.at("logits").to_vector<float>();
      std::ofstream predictions(root / "predictions.tsv");
      predictions
          << (spec.pointer ? "source_id\tgold_indices\tpredicted_indices\n"
                           : "source_id\tinput_ids\tgold_ids\tgreedy_ids\n");
      std::vector<int> input;
      if (!spec.pointer)
        input = test.at("input").to_vector<int>();
      for (std::int64_t b = 0; b < n; ++b) {
        bool match = true;
        for (int t = 0; t < kSteps; ++t) {
          const int gold = spec.pointer && t >= olen[b] ? -1 : target[b * kSteps + t];
          match &= g[b * kSteps + t] == gold;
          if (t < olen[b]) {
            const float* row        = lg.data() + (b * kSteps + t) * V;
            const int    prediction = static_cast<int>(std::max_element(row, row + V) - row);
            const int    truth      = target[b * kSteps + t];
            ++confusion[{truth, prediction}];
            correct += prediction == truth;
            ++tokens;
          }
        }
        exact += match;
        predictions << ids[b];
        std::vector<const std::vector<int>*> columns{&target, &g};
        if (!spec.pointer)
          columns.insert(columns.begin(), &input);
        for (const auto* a : columns) {
          predictions << '\t';
          for (int t = 0; t < kSteps; ++t)
            predictions << (*a)[b * kSteps + t] << ',';
        }
        predictions << '\n';
      }
      std::cout << "confusion (truth rows, predicted columns; " << V
                << " classes; sparse counts, omitted cells are zero):\ntruth\tprediction\tcount\n";
      int confusion_total = 0, confusion_correct = 0;
      for (const auto& [cell, count] : confusion) {
        std::cout << cell.first << '\t' << cell.second << '\t' << count << '\n';
        confusion_total += count;
        if (cell.first == cell.second)
          confusion_correct += count;
      }
      check::Require(
          confusion_total == tokens && confusion_correct == correct,
          "all valid held-out token predictions counted");
      result.test_ce          = out.at("loss").item<double>();
      result.teacher_accuracy = double(correct) / tokens;
      result.exact            = exact;
      std::ofstream metrics(root / "metrics.json");
      metrics << "{\"parameters\":" << spec.params << ",\"epochs\":" << epochs
              << ",\"updates\":" << epochs * 16 << ",\"test_token_ce\":" << result.test_ce
              << ",\"teacher_token_accuracy\":" << result.teacher_accuracy << ",\""
              << spec.exact_key << "\":" << exact / double(n) << "}\n";
      check::Require(
          replay::SameBits(replay::Arrays(deployed.p), weights), spec.name + "InferenceImmutable");
      std::cout << task << " testTokenCE=" << result.test_ce
                << " teacherTokenAccuracy=" << result.teacher_accuracy << " " << spec.exact_label
                << "=" << exact << "/" << n << "; padding/empty/restore/labelFree/immutable PASS\n";
    }
    std::cout << "(TF recorded: " << spec.tf_recorded << ")\nwall "
              << std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count()
              << " s\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
