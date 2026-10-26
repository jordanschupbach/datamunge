// CMD: just run-nn bilstm_crf
// End-to-end BiLSTM-CRF NER with exact partition/marginals/Viterbi and
// exhaustive toy proof; three epochs.
//
// Port of tflow's bilstm_crf. Preparation (in ~/projects/tflow_examples):
// the pinned CoNLL subset in output/datasets/bilstm_tagging (bilstm_tagging.npz,
// 128 fit / 32 validation / 32 test sentences, length 2..24, 512-word
// vocabulary, 9 IOB2 NER tags).
// Two independent 8-unit LSTMs give 16-D offline context, a linear head gives
// NER unaries and a linear-chain CRF (9x9 transition[from,to], 9 start/9 end
// scores) gives the conditional sequence likelihood. 5436 parameters in 12
// tensors; Adam .01, batch 8, three epochs (48 updates) on the mean sequence NLL.
// Checks: a 3-tag CRF fixture against all 121 paths for lengths 0..4 (and
// against TF's native values when tflow's fixture is present); when tflow's
// recorded run is present, the forward pass, every loss derivative and the
// whole 48-update training trajectory are replayed from TF's initial weights
// and compared. Then the port trains from its own initialisation and audits
// padding/empty/causality/restore/label-free decoding like the TF version.
// Options: epochs=3 data=PATH out=PATH tf=PATH (tflow's output/models/bilstm_crf).
// Output: output/nn/models/bilstm_crf/.

#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>

#include "common/bilstm_crf.hpp"
#include "common/checks.hpp"
#include "common/paths.hpp"
#include "common/tagging_metrics.hpp"

namespace nn = datamunge::nn;
using nn::Tensor;
using namespace bilstm_crf;

namespace {

double MaxAbsDiff(const Tensor& a, const Tensor& b) {
  const auto x = a.to_vector<double>(), y = b.to_vector<double>();
  check::Require(x.size() == y.size(), "compared tensors have the same size");
  double m = 0;
  for (std::size_t i = 0; i < x.size(); ++i) m = std::max(m, std::abs(x[i] - y[i]));
  return m;
}

double PathScore(const std::vector<double>& U, std::int64_t b, std::int64_t T, std::int64_t K,
                 const std::vector<int>& y, const std::vector<double>& A, const std::vector<double>& st,
                 const std::vector<double>& en) {
  if (y.empty()) return 0;
  double s = st[y[0]] + en[y.back()];
  for (std::size_t t = 0; t < y.size(); ++t) {
    s += U[(b * T + static_cast<std::int64_t>(t)) * K + y[t]];
    if (t) s += A[y[t - 1] * K + y[t]];
  }
  return s;
}

// Exhaustive 3-tag, length-0..4 CRF fixture (tflow's common/ml/crf_fixture.h).
void RunCrfFixture(const std::filesystem::path& root, const std::filesystem::path& tf_root) {
  const std::vector<float> uv{.2f, 1.2f, 0, .7f, .3f, 1.1f, .9f, .2f, .4f, .4f, 1.f, .3f};
  const std::vector<float> tv{.6f, -.1f, .2f, -.4f, 1.2f, -.6f, .1f, -.2f, .5f}, sv{.1f, .2f, -.1f}, ev{.3f, -.2f, .1f};
  const int                gold[4] = {0, 1, 2, 1};
  std::vector<float>       unary;
  std::vector<std::int32_t> lengths, labels;
  for (int b = 0; b < 5; ++b) {
    lengths.push_back(b);
    for (int t = 0; t < 4; ++t) {
      labels.push_back(gold[t]);
      for (int c = 0; c < 3; ++c) unary.push_back(uv[t * 3 + c]);
    }
  }
  Tensor U = nn::from_vector(unary, {5, 4, 3}), A = nn::from_vector(tv, {3, 3}), S = nn::from_vector(sv),
         E = nn::from_vector(ev);
  const Tensor len = nn::from_vector(lengths), lab = nn::from_vector(labels, {5, 4});
  for (Tensor* t : {&U, &A, &S, &E}) t->requires_grad_(true);
  const Tensor z = nn::crf_log_norm(U, len, A, S, E), gs = nn::crf_sequence_score(U, lab, len, A, S, E);
  const Tensor nll = nn::sub(z, gs), loss = nn::mean(nll);
  const auto   g   = nn::grad(loss, {U, A, S, E}, {true, false, false});
  const auto   pg  = nn::grad(nn::sum(z), {U, A, S, E});
  const Tensor alphas = nn::crf_forward(U, len, A, S).detach();
  const auto   decoded = nn::crf_decode(U.detach(), len, A.detach(), S.detach(), E.detach(), -1);
  const auto   Uv = U.to_vector<double>(), Av = A.to_vector<double>(), Sv = S.to_vector<double>(), Ev = E.to_vector<double>();
  const auto   zv = z.to_vector<double>(), gv = gs.to_vector<double>(), marg = pg[0].to_vector<double>();
  const auto   vt = decoded.tags.to_vector<int>();
  int          total_paths = 0;
  for (int b = 0; b < 5; ++b) {
    int paths = 1;
    for (int t = 0; t < b; ++t) paths *= 3;
    double           maximum = -INFINITY;
    std::vector<int> best_path;
    std::vector<double> scores;
    for (int code = 0; code < paths; ++code) {
      std::vector<int> y(b);
      int              c = code;
      for (int t = b - 1; t >= 0; --t) y[t] = c % 3, c /= 3;
      const double s = PathScore(Uv, b, 4, 3, y, Av, Sv, Ev);
      scores.push_back(s);
      if (s > maximum) maximum = s, best_path = y;
    }
    total_paths += paths;
    double mass = 0;
    for (double s : scores) mass += std::exp(s - maximum);
    check::Near(b ? maximum + std::log(mass) : 0.0, zv[b], 2e-6, "allPathsExactPartitionEnumeration");
    check::Near(b ? maximum : 0.0, decoded.scores.to_vector<double>()[b], 1e-12, "ViterbiMatchesExhaustiveBestPathScore");
    for (int t = 0; t < b; ++t) check::Require(vt[b * 4 + t] == best_path[t], "ViterbiMatchesExhaustiveBestLabels");
    check::Near(PathScore(Uv, b, 4, 3, std::vector<int>(gold, gold + b), Av, Sv, Ev), gv[b], 1e-6,
                "nativeGoldUnariesPlusTransitionAndEndpoints");
    for (int t = 0; t < 4; ++t) {
      double sum = 0;
      for (int c = 0; c < 3; ++c) sum += marg[(b * 4 + t) * 3 + c];
      check::Near(sum, t < b ? 1 : 0, 2e-6, "actualMarginalsSumToOneOnlyOnValidPositions");
    }
  }
  check::Require(total_paths == 121, "enumerateAll121PathsAcrossLengthsZeroThroughFour");
  int differs = 0;
  for (int t = 0; t < 4; ++t) {
    const float* row = unary.data() + 4 * 12 + t * 3;
    differs += vt[4 * 4 + t] != std::max_element(row, row + 3) - row;
  }
  check::Require(differs > 0, "fixtureViterbiIsNotIndependentUnaryArgmax");
  nn::NamedTensors out{{"log_partition", z.detach()}, {"gold_score", gs.detach()}, {"nll", nll.detach()},
                       {"alphas", alphas}, {"loss", loss.detach()}};
  const char* names[4] = {"unaries", "transition", "start", "end"};
  for (int j = 0; j < 4; ++j) {
    out.emplace_back(std::string("gradient_") + names[j], g[j]);
    out.emplace_back(std::string("partition_gradient_") + names[j], pg[j]);
  }
  out.emplace_back("viterbi", decoded.tags);
  out.emplace_back("viterbi_scores", decoded.scores);
  nn::save_npz((root / "crf_fixture_inputs.npz").string(),
               {{"unaries", U.detach()}, {"transition", A.detach()}, {"start", S.detach()}, {"end", E.detach()},
                {"lengths", len}, {"labels", lab}});
  nn::save_npz((root / "crf_fixture_outputs.npz").string(), out);
  std::cout << " CRF fixture: all 121 paths, partition/gold/Viterbi/marginals exact";
  if (std::filesystem::exists(tf_root / "crf_fixture_outputs.npz")) {
    const auto tf = nn::load_npz((tf_root / "crf_fixture_outputs.npz").string());
    double     worst = 0;
    for (const auto& [name, value] : out) {
      if (name == "alphas") {
        worst = std::max(worst, MaxAbsDiff(value, nn::slice_axis(Named(tf, name), 1, 1, 5)));
        continue;
      }
      worst = std::max(worst, MaxAbsDiff(value, Named(tf, name)));
    }
    check::Near(worst, 0, 2e-6, "every CRF fixture value and derivative matches TF's native output");
    std::cout << "; vs TF native fixture max diff=" << worst;
  }
  std::cout << "\n";
}

TaggingBatch Split(const nn::NamedTensors& data, const std::string& split) {
  return {Named(data, split + "_input"), Named(data, split + "_lengths"), Named(data, split + "_pos"),
          Named(data, split + "_ner"), Named(data, split + "_source_ids")};
}

nn::OptimizerConfig AdamConfig() {
  nn::OptimizerConfig c;
  c.kind = nn::OptimizerKind::Adam;
  return c;
}

/// One epoch of Adam updates over the fit rows; returns the visited rows.
void TrainEpoch(nn::ParameterStore& p, nn::Optimizer& opt, nn::BatchOrder& order, const TaggingBatch& fit,
                std::ofstream* batches, int epoch, int& updates) {
  order.begin_epoch();
  for (;;) {
    const auto rows = order.next();
    if (rows.empty()) break;
    const auto o = Evaluate(p, TaggingRows(fit, rows));
    opt.step(nn::grad(o.loss, p.parameters()));
    if (batches)
      for (auto r : rows) *batches << epoch << '\t' << updates << '\t' << r << '\n';
    ++updates;
  }
}

// Replays tflow's recorded run from TF's own initial weights.
void TfParity(const std::filesystem::path& tf_root, const std::vector<TaggingBatch>& pools, int epochs) {
  if (!std::filesystem::exists(tf_root / "initial_arrays.npz") || !std::filesystem::exists(tf_root / "first_batch.npz")) {
    std::cout << " (tflow run not found at " << tf_root << "; TF parity skipped)\n";
    return;
  }
  nn::ParameterStore p(1);
  Build(p);
  for (const auto& [name, value] : nn::load_npz((tf_root / "initial_arrays.npz").string())) p.set(name, value);
  const auto   fb = nn::load_npz((tf_root / "first_batch.npz").string());
  TaggingBatch first{Named(fb, "input"), Named(fb, "lengths"), Named(fb, "pos"), Named(fb, "ner"), Named(fb, "source_ids")};
  const auto   ref = nn::load_npz((tf_root / "first_forward.npz").string());
  auto         o   = Evaluate(p, first);
  const auto   crf = CrfParams(p);
  // marginals and expected counts are derivatives of sum log Z
  const auto marg = nn::grad(nn::sum(o.log_partition), {o.path.unaries, crf.transition, crf.start, crf.end},
                             {true, false, false});
  std::map<std::string, Tensor> ours{{"unaries", o.path.unaries}, {"log_partition", o.log_partition},
                                     {"gold_score", o.gold_score}, {"nll", o.nll}, {"loss", o.loss},
                                     {"features", o.path.features}, {"forward_states", o.path.forward_states},
                                     {"backward_aligned", o.path.backward_aligned}, {"marginals", marg[0]},
                                     {"expected_transition_counts", marg[1]}, {"expected_start_counts", marg[2]},
                                     {"expected_end_counts", marg[3]}};
  double forward = 0;
  for (const auto& [name, value] : ours) forward = std::max(forward, MaxAbsDiff(value, Named(ref, name)));
  check::Near(forward, 0, 2e-5, "first-batch forward, partition, marginals and expected counts match TF");
  const auto grads = nn::grad(o.loss, p.parameters());
  const auto tfg   = nn::load_npz((tf_root / "initial_gradients.npz").string());
  double     gerr  = 0;
  for (std::size_t i = 0; i < grads.size(); ++i) gerr = std::max(gerr, MaxAbsDiff(grads[i], Named(tfg, p.names()[i])));
  check::Near(gerr, 0, 2e-5, "all 12 initial parameter gradients match TF");
  std::cout << " TF parity from TF's initial weights: first batch forward/partition/marginals max diff=" << forward
            << ", all parameter gradients max diff=" << gerr << "\n";
  // Same batch order (BatchOrder reproduces tflow's), same Adam: replay the run.
  std::ifstream history(tf_root / "history.tsv");
  std::string   line;
  std::getline(history, line);
  nn::Optimizer  opt(p, .01, AdamConfig());
  nn::BatchOrder order(128, 8, 44019);
  int            updates = 0;
  double         worst   = 0;
  for (int epoch = 1; epoch <= epochs; ++epoch) {
    TrainEpoch(p, opt, order, pools[0], nullptr, epoch, updates);
    double val;
    {
      nn::NoGradGuard guard;
      val = Evaluate(p, pools[1]).loss.item<double>();
    }
    double tf_val = NAN;
    if (std::getline(history, line)) tf_val = std::stod(line.substr(line.find('\t') + 1));
    std::cout << "  epoch" << epoch << " validationLoss=" << val << " (TF " << tf_val << ")\n";
    if (std::isfinite(tf_val)) worst = std::max(worst, std::abs(val - tf_val));
  }
  if (epochs == 3 && std::filesystem::exists(tf_root / "model_arrays.npz")) {
    double     werr = 0;
    const auto tfw  = nn::load_npz((tf_root / "model_arrays.npz").string());
    for (const auto& name : p.names()) werr = std::max(werr, MaxAbsDiff(p.at(name), Named(tfw, name)));
    nn::NoGradGuard guard;
    const double    test = Evaluate(p, pools[2]).loss.item<double>();
    const auto      dec  = Decode(p, pools[2]);
    const auto      tfd  = nn::load_npz((tf_root / "decoded.npz").string());
    const auto      a = dec.tags.to_vector<int>(), b = Named(tfd, "viterbi").to_vector<int>();
    int             same = 0;
    for (std::size_t i = 0; i < a.size(); ++i) same += a[i] == b[i];
    std::cout << " TF replay after 48 updates: max weight diff=" << werr << ", test NLL=" << test
              << " (TF 10.5985), Viterbi tags identical " << same << "/" << a.size() << "\n";
    check::Near(worst, 0, 2e-3, "validation loss trajectory tracks TF's recorded run");
    check::Near(werr, 0, 2e-3, "trained weights track TF's after 48 Adam updates");
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const auto            start   = std::chrono::steady_clock::now();
    int                   epochs  = 3;
    std::filesystem::path data    = paths::Dataset("bilstm_tagging"), root = paths::OutputRoot() / "models/bilstm_crf";
    std::filesystem::path tf_root = paths::DataRoot().parent_path() / "models" / "bilstm_crf";
    for (int i = 1; i < argc; ++i) {
      const std::string a = argv[i];
      if (a.rfind("epochs=", 0) == 0) epochs = std::stoi(a.substr(7));
      else if (a.rfind("data=", 0) == 0) data = a.substr(5);
      else if (a.rfind("out=", 0) == 0) root = a.substr(4);
      else if (a.rfind("tf=", 0) == 0) tf_root = a.substr(3);
      else throw std::invalid_argument("epochs=N data=PATH out=PATH tf=PATH");
    }
    check::Require(epochs > 0 && epochs <= 100, "threeCheapDefaultEpochs");
    std::filesystem::create_directories(root);
    RunCrfFixture(root, tf_root);
    check::Require(tagging::EntitySpans({2, 2, 0, 3, 4, 5}) == std::set<tagging::EntitySpan>{{0, 2, 1}, {3, 5, 2}, {5, 6, 3}},
                   "explicitIstartBIORepairAndExactSpanBoundaries");
    const auto arrays = nn::load_npz((data / "bilstm_tagging.npz").string());
    std::vector<TaggingBatch> pools{Split(arrays, "fit"), Split(arrays, "validation"), Split(arrays, "test")};
    for (auto f : {"manifest.json", "vocabulary.txt"})
      if (std::filesystem::exists(data / f))
        std::filesystem::copy_file(data / f, root / f, std::filesystem::copy_options::overwrite_existing);
    TfParity(tf_root, pools, epochs);

    // ---- own initialisation -------------------------------------------------------------
    nn::ParameterStore model(45003);
    Build(model);
    check::Require(model.count() == 5436 && model.size() == 12, "6102EmbeddingTwoIndependentLSTMsAndPOSNERHeads");
    model.save((root / "initial_arrays.npz").string());
    const Tensor pad_row0 = nn::select(model.at("embedding/W"), 0, 0).clone();
    nn::Optimizer  opt(model, .01, AdamConfig());
    nn::BatchOrder order(128, 8, 44019);
    std::ofstream  history(root / "history.tsv"), batches(root / "batches.tsv");
    history.precision(17);
    history << "epoch\tvalidation_loss\tvalidation_entity_f1\n";
    batches << "epoch\tstep\trow\n";
    int updates = 0;
    for (int epoch = 1; epoch <= epochs; ++epoch) {
      TrainEpoch(model, opt, order, pools[0], &batches, epoch, updates);
      nn::NoGradGuard guard;
      const double    val   = Evaluate(model, pools[1]).loss.item<double>();
      const auto      score = tagging::ScoreTagging(Decode(model, pools[1]).tags, pools[1].ner, pools[1].lengths);
      history << epoch << '\t' << val << '\t' << score.EntityF1() << '\n';
      std::cout << "epoch" << epoch << " validationLoss=" << val << " entityF1=" << score.EntityF1() << std::endl;
    }
    model.save((root / "final.npz").string());
    nn::ParameterStore deployed(0);
    Build(deployed);
    deployed.load((root / "final.npz").string());
    for (const auto& name : model.names())
      check::Require(MaxAbsDiff(model.at(name), deployed.at(name)) == 0, "all5436TaggerWeightsRestoreBitexact");
    const auto before = deployed.flatten();

    // ---- audits on the test split -----------------------------------------------------------
    const TaggingBatch& test = pools[2];
    auto                output = Evaluate(deployed, test);
    const auto          crf    = CrfParams(deployed);
    const auto          parts  = nn::grad(nn::sum(output.log_partition), {output.path.unaries, crf.transition, crf.start, crf.end},
                                          {true, false, false});
    const auto lengths = test.lengths.to_vector<int>();
    const int  B       = static_cast<int>(lengths.size());
    // padding invariance: arbitrary words in padded positions change nothing
    TaggingBatch padded = test, last = test;
    {
      auto in = test.input.to_vector<std::int32_t>(), lw = in;
      int  pad_positions = 0;
      for (int b = 0; b < B; ++b) {
        for (int t = lengths[b]; t < kSteps; ++t, ++pad_positions) in[b * kSteps + t] = 2 + (b + t) % 510;
        lw[b * kSteps + lengths[b] - 1] = 1;
      }
      padded.input = nn::from_vector(in, {B, kSteps});
      last.input   = nn::from_vector(lw, {B, kSteps});
      check::Require(pad_positions > 100, "nonVacuousPadding");
    }
    nn::NoGradGuard guard;
    const auto      padout = Evaluate(deployed, padded), lastout = Evaluate(deployed, last);
    check::Require(MaxAbsDiff(padout.path.unaries, output.path.unaries) == 0 &&
                       MaxAbsDiff(padout.nll, output.nll) == 0,
                   "allTaggerPathsLossesAndPartialsPaddingInvariant");
    {
      const auto f0 = output.path.forward_states.to_vector<float>(), f1 = lastout.path.forward_states.to_vector<float>();
      for (int b = 0; b < B; ++b)
        for (int t = 0; t < lengths[b]; ++t)  // states 0..length-1 precede the last word
          for (int j = 0; j < kUnits; ++j)
            check::Require(f0[(b * 25 + t) * kUnits + j] == f1[(b * 25 + t) * kUnits + j],
                           "lastWordChangeCannotAffectEarlierFORWARDstates");
      const auto a0 = output.path.backward_aligned.to_vector<float>(), a1 = lastout.path.backward_aligned.to_vector<float>();
      double     future = 0;
      for (int b = 0; b < B; ++b)
        for (int j = 0; j < kUnits; ++j) future = std::max(future, double(std::abs(a0[b * kSteps * kUnits + j] - a1[b * kSteps * kUnits + j])));
      check::Require(future > 0, "backwardContextAtFirstWordSeesTheLastWord");
    }
    TaggingBatch empty = TaggingRows(test, {0, 1});
    empty.lengths      = nn::zeros({2}, nn::DType::Int32);
    const auto emptyout = Evaluate(deployed, empty);
    check::Require(nn::sum(nn::abs(emptyout.path.unaries)).item<double>() == 0 &&
                       nn::sum(nn::abs(emptyout.nll)).item<double>() == 0 &&
                       nn::sum(nn::abs(emptyout.log_partition)).item<double>() == 0,
                   "zeroLengthTaggingHasNoTokenLossUnariesOrState");
    const auto empty_decoded = Decode(deployed, empty);
    for (int v : empty_decoded.tags.to_vector<int>()) check::Require(v == -1, "emptyViterbiHasOnlyPaddingLabels");
    for (double v : empty_decoded.scores.to_vector<double>()) check::Require(v == 0, "emptyViterbiScoreIsZero");
    check::Require(MaxAbsDiff(nn::select(deployed.at("embedding/W"), 0, 0), pad_row0) == 0, "maskedPADWordEmbeddingNeverUpdated");

    const auto viterbi = Decode(deployed, test);
    const auto again   = Decode(deployed, test);
    check::Require(MaxAbsDiff(viterbi.tags, again.tags) == 0 && MaxAbsDiff(viterbi.scores, again.scores) == 0,
                   "repeatTaggerInferenceBitexact");
    const auto greedy = tagging::TaggingArgmax(output.path.unaries, test.lengths);
    const auto U = output.path.unaries.to_vector<double>(), Av = crf.transition.to_vector<double>(),
               Sv = crf.start.to_vector<double>(), Ev = crf.end.to_vector<double>();
    const auto tags = viterbi.tags.to_vector<int>();
    const auto sc   = viterbi.scores.to_vector<double>();
    const auto nll = output.nll.to_vector<double>(), logz = output.log_partition.to_vector<double>();
    const auto marg = parts[0].to_vector<double>();
    int        transitions_expected = 0;
    for (int b = 0; b < B; ++b) {
      transitions_expected += std::max(0, lengths[b] - 1);
      check::Require(nll[b] >= -1e-4 && logz[b] + 1e-4 >= sc[b], "goldAndBestPathIncludedInNormalizedSequenceDistribution");
      std::vector<int> y(tags.begin() + b * kSteps, tags.begin() + b * kSteps + lengths[b]);
      check::Near(PathScore(U, b, kSteps, kTags, y, Av, Sv, Ev), sc[b], 1e-9, "decodedPathRescoresToActualViterbiMaximum");
      for (int t = 0; t < kSteps; ++t) {
        double mass = 0;
        for (int c = 0; c < kTags; ++c) mass += marg[(b * kSteps + t) * kTags + c];
        check::Near(mass, t < lengths[b] ? 1 : 0, 3e-5, "truePosteriorMarginalsNormalizeOnlyOnValidPositions");
      }
    }
    check::Near(nn::sum(parts[1]).item<double>(), transitions_expected, 1e-3, "expectedTransitionCountsSumToActualNumberOfEdges");
    check::Near(nn::sum(parts[2]).item<double>(), B, 2e-4, "expectedStartCountsSumToNonemptySequences");
    check::Near(nn::sum(parts[3]).item<double>(), B, 2e-4, "expectedEndCountsSumToNonemptySequences");

    // word-majority baseline from the fit split
    std::vector<int> counts(kVocab * kTags, 0), global(kTags, 0);
    {
      const auto in = pools[0].input.to_vector<int>(), ner = pools[0].ner.to_vector<int>(), n = pools[0].lengths.to_vector<int>();
      for (int b = 0; b < 128; ++b)
        for (int t = 0; t < n[b]; ++t) ++counts[in[b * kSteps + t] * kTags + ner[b * kSteps + t]], ++global[ner[b * kSteps + t]];
    }
    const int fallback = static_cast<int>(std::max_element(global.begin(), global.end()) - global.begin());
    std::vector<std::int32_t> base(B * kSteps, -1), all_o(B * kSteps, 0);
    {
      const auto in = test.input.to_vector<int>();
      for (int b = 0; b < B; ++b)
        for (int t = 0; t < lengths[b]; ++t) {
          const int* row = counts.data() + kTags * in[b * kSteps + t];
          base[b * kSteps + t] = std::accumulate(row, row + kTags, 0) ? static_cast<int>(std::max_element(row, row + kTags) - row) : fallback;
        }
    }
    const auto score      = tagging::ScoreTagging(viterbi.tags, test.ner, test.lengths);
    const auto greedy_s   = tagging::ScoreTagging(greedy, test.ner, test.lengths);
    const auto base_s     = tagging::ScoreTagging(nn::from_vector(base, {B, kSteps}), test.ner, test.lengths);
    const auto zeros_s    = tagging::ScoreTagging(nn::from_vector(all_o, {B, kSteps}), test.ner, test.lengths);
    nn::save_npz((root / "decoded.npz").string(), {{"viterbi", viterbi.tags}, {"scores", viterbi.scores}, {"greedy", greedy},
                                                    {"baseline", nn::from_vector(base, {B, kSteps})}});
    nn::save_npz((root / "outputs.npz").string(),
                 {{"unaries", output.path.unaries}, {"log_partition", output.log_partition}, {"gold_score", output.gold_score},
                  {"nll", output.nll}, {"loss", output.loss}, {"marginals", parts[0]}, {"features", output.path.features}});
    std::ofstream predictions(root / "predictions.tsv");
    predictions << "source_id\tposition\tword_id\tgold_ner\tgreedy_ner\tviterbi_ner\n";
    {
      const auto src = test.source_ids.to_vector<int>(), in = test.input.to_vector<int>(), ner = test.ner.to_vector<int>(),
                 gr = greedy.to_vector<int>();
      for (int b = 0; b < B; ++b)
        for (int t = 0; t < lengths[b]; ++t)
          predictions << src[b] << '\t' << t << '\t' << in[b * kSteps + t] << '\t' << ner[b * kSteps + t] << '\t'
                      << gr[b * kSteps + t] << '\t' << tags[b * kSteps + t] << '\n';
    }
    std::ofstream metrics(root / "metrics.json");
    metrics.precision(17);
    metrics << "{\"parameters\":5436,\"parameter_tensors\":12,\"epochs\":" << epochs << ",\"optimizer_updates\":" << updates
            << ",\"test_mean_sequence_nll\":" << output.loss.item<double>() << ",\"ner_token_accuracy\":" << score.NerAccuracy()
            << ",\"entity_precision\":" << score.EntityPrecision() << ",\"entity_recall\":" << score.EntityRecall()
            << ",\"entity_f1\":" << score.EntityF1() << ",\"true_positive_entities\":" << score.true_positive
            << ",\"predicted_entities\":" << score.predicted_entities << ",\"gold_entities\":" << score.gold_entities
            << ",\"tokens\":" << score.tokens << ",\"greedy_entity_f1\":" << greedy_s.EntityF1()
            << ",\"word_majority_entity_f1\":" << base_s.EntityF1() << ",\"all_o_token_accuracy\":" << zeros_s.NerAccuracy()
            << ",\"all_o_entity_f1\":0}\n";
    check::Require(before == deployed.flatten(), "allConditionalCrfControlsLeaveWeightsImmutable");
    check::Require(bool(metrics) && bool(history) && bool(batches) && bool(predictions), "completeBiLstmCrfRecords");
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    std::cout << "testMeanSequenceNLL=" << output.loss.item<double>() << " tokenAccuracy=" << score.NerAccuracy()
              << " entityF1=" << score.EntityF1() << " greedyF1=" << greedy_s.EntityF1()
              << " wordMajorityF1=" << base_s.EntityF1() << " allOAccuracy=" << zeros_s.NerAccuracy()
              << ";realCRFpartition/marginals/expectedCounts/Viterbi/empty/padding/restore/labelFree/immutable PASS\n"
              << "Artifacts:" << root << " (" << seconds << " s)" << std::endl;
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
