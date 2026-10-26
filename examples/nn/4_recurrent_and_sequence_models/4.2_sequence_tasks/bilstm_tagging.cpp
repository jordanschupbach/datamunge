// CMD: just run-nn bilstm_tagging
// Joint POS and entity tagging on real short CoNLL sentences; independent
// bidirectional LSTM, three epochs.
//
// Port of tflow's bilstm_tagging. Data: tflow's output/datasets/bilstm_tagging
// (bilstm_tagging.npz: pinned CoNLL-2003 subset, 128 fit / 32 validation / 32
// test complete sentences of length 2..24, fit-only 512-word vocabulary with
// PAD 0 / UNK 1, 45 POS tags, 9 IOB2 NER tags; vocabulary.txt).
// An 8-D embedding, two independent 8-unit LSTMs (forward, and backward over
// the reversed valid words, re-aligned to original time) give 16-D features;
// linear POS (45) and NER (9) heads, equal-weight masked mean token CE.
// 6102 parameters, Adam .01, batch 8, three epochs (48 updates).
// When tflow's recorded run (output/models/bilstm_tagging) is present, the port
// replays it from TF's initial weights (initial outputs with every BPTT partial,
// first-batch forward and gradients, history, final weights). The
// own-initialisation run then audits restore, label-free prediction,
// single/batch parity, the forward probe's causality and the backward probe's
// real future context, padding and zero-length invariance, the frozen PAD
// embedding row and immutability, and compares with fit-only word-majority
// and all-O baselines (exact entity spans, not just token accuracy).
// Options: epochs=3 data=PATH out=PATH tf=PATH.
// Output: output/nn/models/bilstm_tagging/.

#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>

#include "common/bilstm_tagging.hpp"
#include "common/checks.hpp"
#include "common/paths.hpp"
#include "common/tagging_metrics.hpp"
#include "common/tf_replay.hpp"

namespace nn = datamunge::nn;
using nn::Tensor;
using replay::Map;
using namespace bilstm_tagging;

namespace {

struct Model {
  nn::ParameterStore p;
  explicit Model(std::uint64_t seed) : p(seed) {
    nn::NoGradGuard guard;
    BiLstmTagging(p, nn::zeros({1, kSteps, kVocab}), nn::ones({1}, nn::DType::Int32), Initial(1));
    check::Require(p.count() == 6102 && p.size() == 11, "6102EmbeddingTwoIndependentLSTMsAndPOSNERHeads");
  }
  static std::vector<Tensor> Initial(std::int64_t n) {
    return {nn::zeros({n, kUnits}), nn::zeros({n, kUnits}), nn::zeros({n, kUnits}), nn::zeros({n, kUnits})};
  }
  static Tensor Encode(const Map& x) {
    for (int v : x.at("input").to_vector<int>()) check::Require(v >= 0 && v < kVocab, "validTaggingWordID");
    return nn::one_hot(x.at("input"), kVocab);
  }
  static Map Fields(const Path& path) {
    const auto& c = path.context;
    return {{"pos_logits", path.pos_logits.detach()},           {"ner_logits", path.ner_logits.detach()},
            {"forward_states", c.forward_states.detach()},      {"forward_cells", c.forward_cells.detach()},
            {"backward_processing", c.backward_processing.detach()}, {"backward_cells", c.backward_cells.detach()},
            {"backward_aligned", c.backward_aligned.detach()},  {"features", c.features.detach()},
            {"forward_gates", c.forward_gates.detach()},        {"backward_gates", c.backward_gates.detach()}};
  }
  struct Losses {
    Tensor loss, pos_loss, ner_loss, selected;
  };
  static Losses Loss(const Path& path, const Map& x) {
    const Tensor mask = nn::cast(nn::less(nn::reshape(nn::arange(0, kSteps, 1, nn::DType::Int32), {1, -1}),
                                          nn::reshape(x.at("lengths"), {-1, 1})),
                                 nn::DType::Float32);
    const Tensor pt = nn::one_hot(x.at("pos"), kPos), nt = nn::one_hot(x.at("ner"), kNer);
    const Tensor pos_ce = nn::neg(nn::sum(nn::mul(pt, nn::log_softmax(path.pos_logits, -1)), {2}));
    const Tensor ner_ce = nn::neg(nn::sum(nn::mul(nt, nn::log_softmax(path.ner_logits, -1)), {2}));
    const Tensor denominator = nn::maximum(nn::sum(mask), nn::scalar(1.0));
    Losses       l;
    l.pos_loss = nn::div(nn::sum(nn::mul(mask, pos_ce)), denominator);
    l.ner_loss = nn::div(nn::sum(nn::mul(mask, ner_ce)), denominator);
    l.loss     = nn::mul(nn::add(l.pos_loss, l.ner_loss), .5);
    l.selected = nn::mul(nn::sum(nn::mul(mask, nn::add(nn::sum(nn::mul(pt, path.pos_logits), {2}),
                                                      nn::sum(nn::mul(nt, path.ner_logits), {2})))),
                         .5);
    return l;
  }
  /// Words and lengths only.
  Map Predict(const Map& x) {
    nn::NoGradGuard guard;
    return Fields(BiLstmTagging(p, Encode(x), x.at("lengths"), Initial(x.at("input").shape()[0])));
  }
  Map Evaluate(const Map& x) {
    Tensor enc     = Encode(x);
    auto   initial = Initial(x.at("input").shape()[0]);
    enc.requires_grad_(true);
    for (auto& t : initial) t.requires_grad_(true);
    const Path   path = BiLstmTagging(p, enc, x.at("lengths"), initial);
    const Losses l    = Loss(path, x);
    nn::GradOptions o;
    o.allow_unused = true;
    o.retain_graph = true;
    std::vector<Tensor> wrt{enc};
    wrt.insert(wrt.end(), initial.begin(), initial.end());
    const auto g  = nn::grad(std::vector<Tensor>{l.selected}, wrt, {}, o);
    const auto ff = nn::grad(std::vector<Tensor>{nn::sum(nn::slice(path.context.forward_states, {0, 1, 0}, {-1, 1, 1}))}, {enc}, {}, o);
    const auto bb = nn::grad(std::vector<Tensor>{nn::sum(nn::slice(path.context.backward_aligned, {0, 0, 0}, {-1, 1, 1}))}, {enc}, {}, o);
    auto def = [](const Tensor& t, const Tensor& like) { return t.defined() ? t : nn::zeros_like(like).detach(); };
    Map  out = Fields(path);
    out["loss"]                    = l.loss.detach();
    out["pos_loss"]                = l.pos_loss.detach();
    out["ner_loss"]                = l.ner_loss.detach();
    out["encoded_gradient"]        = def(g[0], enc);
    out["forward_probe_gradient"]  = def(ff[0], enc);
    out["backward_probe_gradient"] = def(bb[0], enc);
    for (int j = 0; j < 4; ++j) out["initial_gradient" + std::to_string(j)] = def(g[j + 1], initial[j]);
    return out;
  }
  std::vector<Tensor> Gradients(const Map& x) {
    const Path path = BiLstmTagging(p, Encode(x), x.at("lengths"), Initial(x.at("input").shape()[0]));
    nn::GradOptions o;
    o.allow_unused = true;  // a batch without some words leaves their embedding rows at zero gradient
    auto g = nn::grad(std::vector<Tensor>{Loss(path, x).loss}, p.parameters(), {}, o);
    for (std::size_t i = 0; i < g.size(); ++i)
      if (!g[i].defined()) g[i] = nn::zeros_like(p.parameters()[i]).detach();
    return g;
  }
};

struct Score {
  tagging::TaggingMetrics ner;
  int                     pos_correct = 0;
  double PosAccuracy() const { return ner.tokens ? double(pos_correct) / ner.tokens : 0; }
};

Score ScoreAll(const Tensor& pos_pred, const Tensor& ner_pred, const Map& x) {
  Score      s;
  s.ner      = tagging::ScoreTagging(ner_pred, x.at("ner"), x.at("lengths"));
  const auto p = pos_pred.to_vector<int>(), g = x.at("pos").to_vector<int>(), n = x.at("lengths").to_vector<int>();
  for (std::size_t b = 0; b < n.size(); ++b)
    for (int t = 0; t < n[b]; ++t) s.pos_correct += p[b * kSteps + t] == g[b * kSteps + t];
  return s;
}
Score ScoreOutputs(const Map& out, const Map& x) {
  return ScoreAll(tagging::TaggingArgmax(out.at("pos_logits"), x.at("lengths")),
                  tagging::TaggingArgmax(out.at("ner_logits"), x.at("lengths")), x);
}

struct Epoch {
  double loss, pos, f1;
};

std::vector<Epoch> Train(Model& m, const Map& fit, const Map& validation, int epochs, std::ofstream* batches,
                         const std::filesystem::path* root) {
  nn::Optimizer      opt(m.p, .01, replay::Adam());
  nn::BatchOrder     order(128, 8, 44019);
  std::vector<Epoch> history;
  int                updates = 0;
  for (int e = 1; e <= epochs; ++e) {
    order.begin_epoch();
    for (;;) {
      const auto rows = order.next();
      if (rows.empty()) break;
      const auto b = replay::Rows(fit, rows);
      if (!updates && root) {
        nn::save_npz((*root / "first_batch.npz").string(), replay::ToNamed(b));
        nn::save_npz((*root / "first_forward.npz").string(), replay::ToNamed(m.Evaluate(b)));
        nn::save_npz((*root / "initial_gradients.npz").string(), replay::ToNamed(replay::Named(m.p, m.Gradients(b))));
      }
      opt.step(m.Gradients(b));
      if (!updates && root) m.p.save((*root / "first_after.npz").string());
      if (batches)
        for (auto r : rows) *batches << e << '\t' << updates << '\t' << r << '\n';
      ++updates;
    }
    const auto val   = m.Evaluate(validation);
    const auto score = ScoreOutputs(val, validation);
    history.push_back({val.at("loss").item<double>(), score.PosAccuracy(), score.ner.EntityF1()});
    if (root) {
      m.p.save((*root / ("epoch" + std::to_string(e) + "_arrays.npz")).string());
      nn::save_npz((*root / ("epoch" + std::to_string(e) + "_validation.npz")).string(), replay::ToNamed(val));
    }
  }
  return history;
}

void TfParity(const std::filesystem::path& tf, const Map& fit, const Map& validation, const Map& test, int epochs) {
  if (!std::filesystem::exists(tf / "initial_arrays.npz")) {
    std::cout << " (tflow run not found at " << tf << "; TF parity skipped)\n";
    return;
  }
  Model m(0);
  replay::LoadInto(m.p, tf / "initial_arrays.npz");
  const double out = replay::MaxDiff(m.Evaluate(test), tf / "initial_outputs.npz");
  const auto   fb  = replay::ToMap(nn::load_npz((tf / "first_batch.npz").string()));
  const double fwd = replay::MaxDiff(m.Evaluate(fb), tf / "first_forward.npz");
  const double gd  = replay::MaxDiff(replay::Named(m.p, m.Gradients(fb)), tf / "initial_gradients.npz");
  check::Near(std::max(out, fwd), 0, 2e-6, "initial outputs, every BPTT partial and first-batch forward match TF");
  check::Near(gd, 0, 2e-6, "first-batch parameter gradients match TF");
  const auto hist = Train(m, fit, validation, epochs, nullptr, nullptr);
  const auto tfh = replay::History(tf / "history.tsv", 1), tfp = replay::History(tf / "history.tsv", 2);
  double     worst = 0;
  for (std::size_t e = 0; e < hist.size() && e < tfh.size(); ++e) {
    std::cout << "  replay epoch" << e + 1 << " validationLoss=" << hist[e].loss << " POS=" << hist[e].pos
              << " entityF1=" << hist[e].f1 << " (TF " << tfh[e] << ", " << tfp[e] << ")\n";
    worst = std::max(worst, std::abs(hist[e].loss - tfh[e]));
  }
  std::cout << " TF parity from TF's initial weights: initial outputs/partials max diff=" << out
            << ", first-batch forward max diff=" << fwd << ", gradients max diff=" << gd;
  if (epochs == 3) {
    const double w = replay::MaxDiff(replay::Arrays(m.p), tf / "model_arrays.npz");
    std::cout << ", final weights max diff=" << w;
    check::Near(worst, 0, 1e-4, "validation history tracks TF's recorded run");
    check::Near(w, 0, 1e-3, "final weights track TF's after 48 Adam updates");
  }
  std::cout << "\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const auto            start = std::chrono::steady_clock::now();
    int                   epochs = 3;
    std::filesystem::path data = paths::Dataset("bilstm_tagging"), root = paths::OutputRoot() / "models/bilstm_tagging",
                          tf = replay::TfModels("bilstm_tagging");
    for (int i = 1; i < argc; ++i) {
      const std::string a = argv[i];
      if (a.rfind("epochs=", 0) == 0) epochs = std::stoi(a.substr(7));
      else if (a.rfind("data=", 0) == 0) data = a.substr(5);
      else if (a.rfind("out=", 0) == 0) root = a.substr(4);
      else if (a.rfind("tf=", 0) == 0) tf = a.substr(3);
      else throw std::invalid_argument("epochs=N data=PATH out=PATH tf=PATH");
    }
    check::Require(epochs > 0 && epochs <= 100, "threeCheapDefaultEpochs");
    std::filesystem::create_directories(root);
    const auto arrays = nn::load_npz((data / "bilstm_tagging.npz").string());
    std::filesystem::copy_file(data / "bilstm_tagging.npz", root / "data.npz", std::filesystem::copy_options::overwrite_existing);
    replay::CopyAssets(data, root, {"manifest.json", "vocabulary.txt"});
    check::Require(tagging::EntitySpans({2, 2, 0, 3, 4, 5}) == std::set<tagging::EntitySpan>{{0, 2, 1}, {3, 5, 2}, {5, 6, 3}},
                   "explicitIstartBIORepairAndExactSpanBoundaries");
    const std::vector<std::string> keys{"input", "lengths", "pos", "ner", "source_ids"};
    const Map fit = replay::Split(arrays, "fit", keys), validation = replay::Split(arrays, "validation", keys),
              test = replay::Split(arrays, "test", keys);
    TfParity(tf, fit, validation, test, epochs);

    Model m(44003);
    m.p.save((root / "initial_arrays.npz").string());
    nn::save_npz((root / "initial_outputs.npz").string(), replay::ToNamed(m.Evaluate(test)));
    std::ofstream history(root / "history.tsv"), batches(root / "batches.tsv");
    history.precision(17);
    history << "epoch\tvalidation_loss\tvalidation_pos_accuracy\tvalidation_entity_f1\n";
    batches << "epoch\tstep\trow\n";
    const auto hist = Train(m, fit, validation, epochs, &batches, &root);
    for (std::size_t e = 0; e < hist.size(); ++e) {
      history << e + 1 << '\t' << hist[e].loss << '\t' << hist[e].pos << '\t' << hist[e].f1 << '\n';
      std::cout << "epoch" << e + 1 << " validationLoss=" << hist[e].loss << " POS=" << hist[e].pos
                << " entityF1=" << hist[e].f1 << std::endl;
    }
    const Map first = replay::ToMap(nn::load_npz((root / "first_batch.npz").string()));
    nn::save_npz((root / "final_gradients.npz").string(), replay::ToNamed(replay::Named(m.p, m.Gradients(first))));
    m.p.save((root / "final.npz").string());
    Model deployed(0);
    deployed.p.load((root / "final.npz").string());
    check::Require(replay::SameBits(replay::Arrays(m.p), replay::Arrays(deployed.p)), "all6102TaggerWeightsRestoreBitexact");
    const Map before = replay::Arrays(deployed.p);
    const Map output = deployed.Evaluate(test), pred = deployed.Predict(test), repeat = deployed.Predict(test);
    for (const auto& [k, v] : pred) {
      check::Require(replay::SameBits(v, output.at(k)), "noPOSorNERlabelsFedToPrediction");
      check::Require(replay::SameBits(v, repeat.at(k)), "repeatTaggerInferenceBitexact");
    }
    nn::save_npz((root / "outputs.npz").string(), replay::ToNamed(output));
    nn::save_npz((root / "model_arrays.npz").string(), replay::ToNamed(before));
    {
      const auto single = deployed.Predict(replay::Rows(test, {0}));
      for (auto key : {"pos_logits", "ner_logits"}) {
        const auto a = single.at(key).to_vector<float>(), b = output.at(key).to_vector<float>();
        for (std::size_t j = 0; j < a.size(); ++j) check::Near(a[j], b[j], 2e-6, "singleBatchTokenLogitParity");
      }
    }
    const auto ids = test.at("input").to_vector<std::int32_t>();
    const auto lengths = test.at("lengths").to_vector<int>();
    Map        padded = test, last = test;
    double     backward_future = 0;
    {
      const auto fpg = output.at("forward_probe_gradient").to_vector<float>(), bpg = output.at("backward_probe_gradient").to_vector<float>(),
                 eg = output.at("encoded_gradient").to_vector<float>();
      auto       pi = ids, li = ids;
      int        pad_positions = 0;
      for (int b = 0; b < 32; ++b) {
        const int length = lengths[b];
        for (int t = 0; t < kSteps; ++t) {
          if (t >= length) {
            ++pad_positions;
            pi[b * kSteps + t] = 2 + (b + t) % 510;
          }
          for (int c = 0; c < kVocab; ++c) {
            const std::size_t i = (static_cast<std::size_t>(b) * kSteps + t) * kVocab + c;
            if (t > 0) check::Require(fpg[i] == 0, "firstForwardStateCannotUseFutureWords");
            if (t >= length) check::Require(eg[i] == 0 && bpg[i] == 0, "allPaddedEncodedAndBackwardContextPartialsZero");
            else if (t > 0) backward_future = std::max(backward_future, double(std::abs(bpg[i])));
          }
        }
        li[b * kSteps + length - 1] = 1;
      }
      check::Require(pad_positions > 100 && backward_future > 1e-4, "nonVacuousPaddingAndActualOfflineBackwardFutureContext");
      padded.at("input") = nn::from_vector(pi, {32, kSteps});
      last.at("input")   = nn::from_vector(li, {32, kSteps});
    }
    const Map padout = deployed.Evaluate(padded), lastout = deployed.Evaluate(last);
    check::Require(replay::SameBits(output, padout), "allTaggerPathsLossesAndPartialsPaddingInvariant");
    {
      const auto f0 = output.at("forward_states").to_vector<float>(), f1 = lastout.at("forward_states").to_vector<float>();
      for (int b = 0; b < 32; ++b)
        for (int t = 0; t < lengths[b]; ++t)
          for (int j = 0; j < kUnits; ++j)
            check::Require(f0[(b * 25 + t) * kUnits + j] == f1[(b * 25 + t) * kUnits + j], "lastWordChangeCannotAffectEarlierFORWARDstates");
    }
    nn::save_npz((root / "padding_inputs.npz").string(), replay::ToNamed(padded));
    nn::save_npz((root / "last_word_inputs.npz").string(), replay::ToNamed(last));
    nn::save_npz((root / "padding_outputs.npz").string(), replay::ToNamed(padout));
    nn::save_npz((root / "last_word_outputs.npz").string(), replay::ToNamed(lastout));
    Map empty = replay::Rows(test, {0, 1});
    empty.at("lengths") = nn::zeros({2}, nn::DType::Int32);
    const Map emptyout = deployed.Evaluate(empty);
    for (const auto& [k, v] : emptyout)
      for (float x : v.to_vector<float>()) check::Require(x == 0, "zeroLengthTaggingHasNoTokenLossUnariesStateOrGradient");
    nn::save_npz((root / "zero_length_inputs.npz").string(), replay::ToNamed(empty));
    nn::save_npz((root / "zero_length_outputs.npz").string(), replay::ToNamed(emptyout));
    {
      const auto ia = nn::select(replay::Get(nn::load_npz((root / "initial_arrays.npz").string()), "embedding/W"), 0, 0);
      check::Require(replay::SameBits(ia, nn::select(deployed.p.at("embedding/W"), 0, 0)), "maskedPADWordEmbeddingNeverUpdated");
    }
    // fit-only word-majority baselines
    std::vector<std::int32_t> pc(kVocab * kPos, 0), nc(kVocab * kNer, 0), gp(kPos, 0), gn(kNer, 0);
    {
      const auto in = fit.at("input").to_vector<int>(), po = fit.at("pos").to_vector<int>(), ne = fit.at("ner").to_vector<int>(),
                 n = fit.at("lengths").to_vector<int>();
      for (int b = 0; b < 128; ++b)
        for (int t = 0; t < n[b]; ++t) {
          const int w = in[b * kSteps + t], p = po[b * kSteps + t], r = ne[b * kSteps + t];
          ++pc[w * kPos + p], ++nc[w * kNer + r], ++gp[p], ++gn[r];
        }
    }
    nn::save_npz((root / "baseline_counts.npz").string(),
                 {{"pos", nn::from_vector(pc, {kVocab, kPos})}, {"ner", nn::from_vector(nc, {kVocab, kNer})},
                  {"global_pos", nn::from_vector(gp)}, {"global_ner", nn::from_vector(gn)}});
    auto baseline = [&](const std::vector<std::int32_t>& count, const std::vector<std::int32_t>& global) {
      const int classes  = static_cast<int>(global.size());
      const int fallback = static_cast<int>(std::max_element(global.begin(), global.end()) - global.begin());
      std::vector<std::int32_t> result(32 * kSteps, -1);
      for (int b = 0; b < 32; ++b)
        for (int t = 0; t < lengths[b]; ++t) {
          const std::int32_t* row = count.data() + classes * ids[b * kSteps + t];
          result[b * kSteps + t]  = std::accumulate(row, row + classes, 0) ? static_cast<int>(std::max_element(row, row + classes) - row) : fallback;
        }
      return nn::from_vector(result, {32, kSteps});
    };
    const Tensor bp = baseline(pc, gp), bn = baseline(nc, gn);
    const Tensor pp = tagging::TaggingArgmax(output.at("pos_logits"), test.at("lengths")),
                 np = tagging::TaggingArgmax(output.at("ner_logits"), test.at("lengths"));
    const Tensor all_o = nn::zeros({32, kSteps}, nn::DType::Int32);
    const Score  score = ScoreAll(pp, np, test), base = ScoreAll(bp, bn, test), zeros = ScoreAll(bp, all_o, test);
    check::Require(score.ner.gold_entities > 10 && zeros.ner.EntityF1() == 0, "allOTokenAccuracyMustNotHideZeroEntityRecall");
    nn::save_npz((root / "decoded.npz").string(), {{"pos", pp}, {"ner", np}, {"baseline_pos", bp}, {"baseline_ner", bn}});
    {
      std::ofstream predictions(root / "predictions.tsv");
      predictions << "source_id\tposition\tword_id\tgold_pos\tpredicted_pos\tgold_ner\tpredicted_ner\n";
      const auto src = test.at("source_ids").to_vector<int>(), gpos = test.at("pos").to_vector<int>(),
                 gner = test.at("ner").to_vector<int>(), ppv = pp.to_vector<int>(), npv = np.to_vector<int>();
      for (int b = 0; b < 32; ++b)
        for (int t = 0; t < lengths[b]; ++t) {
          const int i = b * kSteps + t;
          predictions << src[b] << '\t' << t << '\t' << ids[i] << '\t' << gpos[i] << '\t' << ppv[i] << '\t' << gner[i] << '\t'
                      << npv[i] << '\n';
        }
      // first 16 test sentences: gold then predicted NER tags as colored cells
      const int          colors[9][3] = {{255, 255, 255}, {80, 140, 255}, {160, 200, 255}, {255, 100, 70}, {255, 180, 150},
                                         {70, 190, 100},  {170, 230, 185}, {180, 100, 220}, {225, 180, 240}};
      std::vector<float> rgb(320 * 288 * 3, 220.f);
      for (int b = 0; b < 16; ++b)
        for (int row = 0; row < 2; ++row)
          for (int t = 0; t < lengths[b]; ++t) {
            const int tag = row ? npv[b * kSteps + t] : gner[b * kSteps + t];
            for (int y = 0; y < 9; ++y)
              for (int x = 0; x < 11; ++x)
                for (int c = 0; c < 3; ++c) rgb[((b * 20 + row * 10 + y) * 288 + t * 12 + x) * 3 + c] = colors[tag][c];
          }
      nn::save_png(nn::from_vector(rgb, {320, 288, 3}), (root / "ner_tags.png").string(), 0, 255);
    }
    std::ofstream metrics(root / "metrics.json");
    metrics.precision(17);
    metrics << "{\"parameters\":6102,\"parameter_tensors\":11,\"epochs\":" << epochs << ",\"optimizer_updates\":" << epochs * 16
            << ",\"test_loss\":" << output.at("loss").item<double>() << ",\"pos_accuracy\":" << score.PosAccuracy()
            << ",\"ner_token_accuracy\":" << score.ner.NerAccuracy() << ",\"entity_precision\":" << score.ner.EntityPrecision()
            << ",\"entity_recall\":" << score.ner.EntityRecall() << ",\"entity_f1\":" << score.ner.EntityF1()
            << ",\"true_positive_entities\":" << score.ner.true_positive << ",\"predicted_entities\":" << score.ner.predicted_entities
            << ",\"gold_entities\":" << score.ner.gold_entities << ",\"tokens\":" << score.ner.tokens
            << ",\"word_majority_pos_accuracy\":" << base.PosAccuracy() << ",\"word_majority_entity_f1\":" << base.ner.EntityF1()
            << ",\"all_o_token_accuracy\":" << zeros.ner.NerAccuracy() << ",\"all_o_entity_f1\":0,\"backward_future_derivative_max\":"
            << backward_future << "}\n";
    std::ofstream html(root / "index.html");
    html << "<!doctype html><meta charset='utf-8'><title>BiLSTM POS and NER</title><style>body{font:16px system-ui;margin:30px;"
            "max-width:1100px}img{max-width:100%;image-rendering:pixelated}</style><h1>Offline BiLSTM POS and entity tagging</h1>"
            "<p>Pinned CoNLL-2003 subset (128 fit / 32 validation / 32 test complete sentences of length 2..24), fit-only "
            "512-word vocabulary, 45 POS tags, 9 IOB2 NER tags. Embedding 8, independent 8-unit forward/backward LSTMs, "
            "original-time 16-D features, POS/NER linear heads; 6102 parameters, three epochs of Adam .01. No CRF; exact "
            "entity spans are scored (see metrics.json), next to word-majority and all-O baselines.</p><h2>First 16 held-out "
            "NER sequences: gold then predicted</h2><p>gray padding, white O, blue PER, red ORG, green LOC, purple MISC; "
            "darker B / lighter I.</p><img src='ner_tags.png'>";
    check::Require(replay::SameBits(before, replay::Arrays(deployed.p)), "allJointTaggerControlsWeightsImmutable");
    check::Require(bool(metrics) && bool(html) && bool(history) && bool(batches), "completeJointTaggingRecords");
    std::cout << "testPOS=" << score.PosAccuracy() << " entityF1=" << score.ner.EntityF1() << " wordMajorityPOS=" << base.PosAccuracy()
              << " wordMajorityF1=" << base.ner.EntityF1() << " allOtokenAccuracy=" << zeros.ner.NerAccuracy()
              << " (TF recorded testPOS=0.213777 entityF1=0 wordMajority 0.529691/0.328358)"
              << ";originalTimeBiLSTM/futureContext/masking/restore/labelFree/immutable PASS\nArtifacts:"
              << (root / "index.html").string() << "\nwall "
              << std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() << " s\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
