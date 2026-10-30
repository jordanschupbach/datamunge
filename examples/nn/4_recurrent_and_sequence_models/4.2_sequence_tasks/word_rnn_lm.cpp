// CMD: just run-nn word_rnn_lm
// Actual word-level Elman language model on pinned PTB, full training-only
// vocabulary and three cheap epochs.
//
// Port of tflow's word_rnn_lm. Data: tflow's output/datasets/ptb_word
// (ptb_word.npz: Penn Treebank word ids, 128 fit / 32 validation / 32 test
// disjoint 16-word next-token windows; vocabulary.txt: the 10000-word
// training-only vocabulary, <unk> = 0, newlines -> <eos>; written by tflow's
// scripts/fetch_datasets.py). Model (common/word_rnn_lm.hpp): 8-D one-hot
// embedding, 8 tanh recurrent units from a zero state per window, full
// 10000-way softmax; mean next-word NLL, Adam .003, batch 8, three epochs (48
// updates); 170136 parameters. A tiny teaching subset, not the PTB benchmark.
// The port first replays tflow's recorded run (output/models/word_rnn_lm,
// skipped if absent) from TF's initial weights: initial test outputs, the
// first-batch forward and gradients, weights after the first update, the
// validation history and every epoch's weights; from TF's final weights it
// reproduces TF's test outputs, final gradients, the streaming/Jacobian probe
// (state_parity.npz), the future-word control and all four seeded samples
// (identical word codes). The own run then checks restore, stateful
// one-word steps against the full unroll, exactly-zero future input partials,
// future words not changing earlier logits (bitwise), invalid-id rejection,
// bit-exact repeat sampling, the fit-window unigram/bigram baselines and
// immutability. Checkpoints are .npz (final.npz).
// Options: epochs=3 data=PATH (the ptb_word dataset) out=PATH tf=PATH (tflow's
// output/models/word_rnn_lm). Output: output/nn/models/word_rnn_lm/ (index.html, samples,
// metrics.json ...).

#include "common/word_rnn_lm.hpp"

#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <random>

#include "common/checks.hpp"
#include "common/paths.hpp"
#include "common/tf_replay.hpp"

namespace nn = datamunge::nn;
using nn::Tensor;
using replay::Map;
using namespace word_lm;

namespace {

std::vector<std::string> Vocabulary(const std::filesystem::path& path) {
  std::ifstream            file(path);
  std::vector<std::string> words;
  std::string              word;
  while (std::getline(file, word))
    words.push_back(word);
  check::Require(
      words.size() == 10000 && words[0] == "<unk>", "fullTrainingOnlyPTB10000WordVocabulary");
  return words;
}

std::string Decode(const Tensor& codes, const std::vector<std::string>& words) {
  std::string out;
  const auto  c = codes.to_vector<int>();
  for (std::size_t i = 0; i < c.size(); ++i) {
    if (i)
      out += ' ';
    out += words.at(c[i]);
  }
  return out;
}

struct Model {
  nn::ParameterStore p;
  explicit Model(std::uint64_t seed) : p(seed) {
    Parameters(p);
    check::Require(
        p.count() == 170136 && p.size() == 6, "170136WordEmbeddingRecurrentSoftmaxParameters");
  }
  Map Evaluate(const Tensor& x, const Tensor& y) {
    nn::NoGradGuard guard;
    const Path      path = Sequence(p, nn::one_hot(x, kVocab));
    const Map out{{"logits", path.logits}, {"hidden", path.hidden}, {"loss", Loss(path.logits, y)}};
    for (const auto& [k, v] : out)
      for (float f : v.to_vector<float>())
        check::Require(std::isfinite(f), "finitewordLMfields");
    return out;
  }
  std::vector<Tensor> Gradients(const Tensor& x, const Tensor& y) {
    return nn::grad(Loss(Sequence(p, nn::one_hot(x, kVocab)).logits, y), p.parameters());
  }
  /// One observed word and the state only.
  std::pair<Tensor, Tensor> StepOnce(const Tensor& token, const Tensor& state) {
    check::Require(
        token.dtype() == nn::DType::Int32 && token.dim() == 1 &&
            state.shape() == nn::Shape{token.shape()[0], kUnits},
        "validsingle-wordstatebatch");
    for (int t : token.to_vector<int>())
      check::Require(t >= 0 && t < kVocab, "validwordID");
    nn::NoGradGuard guard;
    const auto      s = Step(p, nn::one_hot(token, kVocab), state);
    return {s.logits, s.hidden};
  }
  /// d sum_b logits[b, 8, 0] / d encoded input [B, 16, 10000].
  Tensor InputGradient(const Tensor& x) {
    Tensor encoded = nn::one_hot(x, kVocab);
    encoded.requires_grad_(true);
    const Path path = Sequence(p, encoded);
    return nn::grad(nn::sum(nn::select(nn::select(path.logits, 1, 8), 1, 0)), {encoded})[0];
  }
  Map Sample(const std::vector<int>& prompt, int count, std::uint32_t seed, double temperature) {
    check::Require(
        !prompt.empty() && count > 0 && temperature > 0 && std::isfinite(temperature),
        "validwordprompt/count/temperature");
    Tensor                    state = nn::zeros({1, kUnits});
    std::vector<std::int32_t> codes(prompt.begin(), prompt.end());
    std::vector<double>       uniform(count);
    std::vector<float>        scores(std::size_t(count) * kVocab), states(count * kUnits);
    std::pair<Tensor, Tensor> output;
    for (int w : prompt) {
      output = StepOnce(nn::from_vector(std::vector<std::int32_t>{w}), state);
      state  = output.second;
    }
    std::mt19937 rng(seed);
    for (int t = 0; t < count; ++t) {
      const double u     = (double(rng()) + .5) / 4294967296.;
      uniform[t]         = u;
      const auto lg      = output.first.to_vector<float>();
      double     maximum = -INFINITY, mass = 0;
      for (int c = 0; c < kVocab; ++c) {
        scores[std::size_t(t) * kVocab + c] = lg[c];
        maximum                             = std::max(maximum, double(lg[c]) / temperature);
      }
      const auto sv = state.to_vector<float>();
      for (int j = 0; j < kUnits; ++j)
        states[t * kUnits + j] = sv[j];
      for (int c = 0; c < kVocab; ++c)
        mass += std::exp(double(lg[c]) / temperature - maximum);
      double cumulative = 0;
      int    chosen     = kVocab - 1;
      for (int c = 0; c < kVocab; ++c) {
        cumulative += std::exp(double(lg[c]) / temperature - maximum) / mass;
        if (u < cumulative) {
          chosen = c;
          break;
        }
      }
      codes.push_back(chosen);
      output = StepOnce(nn::from_vector(std::vector<std::int32_t>{chosen}), state);
      state  = output.second;
    }
    return {
        {"codes", nn::from_vector(codes)},
        {"uniform", nn::from_vector(uniform)},
        {"logits", nn::from_vector(scores, {count, kVocab})},
        {"states", nn::from_vector(states, {count, kUnits})}};
  }
};

std::vector<std::vector<int>> Prompts(const Tensor& single, const std::vector<std::string>& words) {
  std::vector<int> eos;
  for (int i = 0; i < kVocab; ++i)
    if (words[i] == "<eos>")
      eos.push_back(i);
  check::Require(eos.size() == 1, "knownTrainingEOSPrompt");
  const auto s = single.to_vector<int>();
  return {eos, {s[0], s[1]}, {s[0], s[1]}, {s[0], s[1]}};
}
constexpr double kTemperatures[4] = {1, 1, .8, 1.2};

void TfParity(
    const std::filesystem::path&    tf,
    const Map&                      data,
    const std::vector<std::string>& words,
    int                             epochs) {
  if (!std::filesystem::exists(tf / "initial_arrays.npz")) {
    std::cout << "(tflow run not found at " << tf << "; TF parity skipped)\n";
    return;
  }
  Model m(0);
  replay::LoadInto(m.p, tf / "initial_arrays.npz");
  const double out = replay::MaxDiff(
      m.Evaluate(data.at("test_input"), data.at("test_target")), tf / "initial_outputs.npz");
  const auto   fb = replay::ToMap(nn::load_npz((tf / "first_batch.npz").string()));
  const double fwd =
      replay::MaxDiff(m.Evaluate(fb.at("input"), fb.at("target")), tf / "first_forward.npz");
  const double gd = replay::MaxDiff(
      replay::Named(m.p, m.Gradients(fb.at("input"), fb.at("target"))),
      tf / "initial_gradients.npz");
  check::Near(std::max(out, fwd), 0, 2e-6, "initial test outputs and first-batch forward match TF");
  check::Near(gd, 0, 2e-6, "first-batch parameter gradients match TF");
  const auto     tfh   = replay::History(tf / "history.tsv");
  double         after = -1, worst = 0, epoch_w = 0;
  nn::Optimizer  opt(m.p, .003, replay::Adam());
  nn::BatchOrder order(128, 8, 42019);
  int            updates = 0;
  for (int epoch = 1; epoch <= epochs; ++epoch) {
    order.begin_epoch();
    for (;;) {
      const auto rows = order.next();
      if (rows.empty())
        break;
      const Tensor x = nn::take_rows(data.at("fit_input"), rows),
                   y = nn::take_rows(data.at("fit_target"), rows);

      const auto path      = Sequence(m.p, nn::one_hot(x, kVocab));
      const auto loss      = Loss(path.logits, y);
      const auto gradients = nn::grad(loss, m.p.parameters());
      opt.step(gradients);
      if (!updates) {
        after = replay::MaxDiff(replay::Arrays(m.p), tf / "first_after.npz");
      }
      ++updates;
    }
    const Map    valid = m.Evaluate(data.at("validation_input"), data.at("validation_target"));
    const double nll   = valid.at("loss").item<double>();
    std::cout << "  replay epoch" << epoch << " validationNLL=" << nll << " (TF "
              << tfh.at(epoch - 1) << ")\n";
    worst           = std::max(worst, std::abs(nll - tfh.at(epoch - 1)));
    const auto file = tf / ("epoch" + std::to_string(epoch) + "_arrays.npz");
    if (std::filesystem::exists(file))
      epoch_w = std::max(epoch_w, replay::MaxDiff(replay::Arrays(m.p), file));
  }
  std::cout << "TF parity from TF's initial weights: initial outputs max diff=" << out
            << ", first forward max diff=" << fwd << ", gradients max diff=" << gd
            << ", weights after first update max diff=" << after << ", history max diff=" << worst
            << ", epoch weights max diff=" << epoch_w << "\n";
  check::Near(after, 0, 1e-4, "weights after the first Adam update match TF");
  check::Near(worst, 0, 1e-4, "validation history tracks TF's recorded run");
  check::Near(epoch_w, 0, 1e-3, "every epoch's weights track TF's");

  Model f(0);
  replay::LoadInto(f.p, tf / "model_arrays.npz");
  const double fo = replay::MaxDiff(
      f.Evaluate(data.at("test_input"), data.at("test_target")), tf / "outputs.npz");
  const double fg = replay::MaxDiff(
      replay::Named(f.p, f.Gradients(fb.at("input"), fb.at("target"))), tf / "final_gradients.npz");
  const Map    sp    = replay::ToMap(nn::load_npz((tf / "state_parity.npz").string()));
  const double ig    = replay::MaxAbsDiff(f.InputGradient(sp.at("input")), sp.at("input_gradient"));
  Tensor       state = nn::zeros({1, kUnits});
  double       steps = 0;
  const auto   in    = sp.at("input").to_vector<std::int32_t>();
  for (int t = 0; t < kSteps; ++t) {
    const auto s = f.StepOnce(nn::from_vector(std::vector<std::int32_t>{in[t]}), state);
    state        = s.second;
    steps        = std::max(
        {steps,
                replay::MaxAbsDiff(s.first, nn::select(sp.at("step_logits"), 1, t)),
                replay::MaxAbsDiff(s.second, nn::select(sp.at("step_states"), 1, t))});
  }
  const Map    fc  = replay::ToMap(nn::load_npz((tf / "future_control.npz").string()));
  const Tensor tgt = nn::take_rows(data.at("test_target"), {0});
  const double fut =
      replay::MaxAbsDiff(f.Evaluate(fc.at("input"), tgt).at("logits"), fc.at("logits"));
  const auto prompts = Prompts(sp.at("input"), words);
  int        same    = 0;
  double     sample  = 0;
  for (int i = 0; i < 4; ++i) {
    const Map s = f.Sample(prompts[i], 64, 42029, kTemperatures[i]);
    const Map r =
        replay::ToMap(nn::load_npz((tf / ("sample" + std::to_string(i) + ".npz")).string()));
    same += replay::SameBits(s.at("codes"), r.at("codes"));
    sample = std::max(sample, replay::MaxDiff(s, r));
  }
  std::cout << "from TF's final weights: test outputs max diff=" << fo
            << ", final gradients max diff=" << fg << ", step logits/states max diff=" << steps
            << ", input Jacobian max diff=" << ig << ", future control max diff=" << fut
            << ", samples with identical codes " << same << "/4 (values max diff=" << sample
            << ")\n";
  check::Near(
      std::max({fo, steps, fut}),
      0,
      1e-5,
      "TF's trained weights reproduce TF's outputs, steps and future control");
  check::Near(
      std::max(fg, ig),
      0,
      2e-6,
      "TF's trained weights reproduce TF's gradients and input Jacobian");
  check::Require(same == 4, "TF's trained weights reproduce TF's four sampled word sequences");
}

} // namespace

int main(int argc, char** argv) {
  try {
    const auto            start    = std::chrono::steady_clock::now();
    int                   epochs   = 3;
    std::filesystem::path data_dir = paths::Dataset("ptb_word"),
                          root     = paths::OutputRoot() / "models" / "word_rnn_lm",
                          tf       = replay::TfModels("word_rnn_lm");
    for (int i = 1; i < argc; ++i) {
      const std::string a = argv[i];
      if (a.rfind("epochs=", 0) == 0)
        epochs = std::stoi(a.substr(7));
      else if (a.rfind("data=", 0) == 0)
        data_dir = a.substr(5);
      else if (a.rfind("out=", 0) == 0)
        root = a.substr(4);
      else if (a.rfind("tf=", 0) == 0)
        tf = a.substr(3);
      else
        throw std::invalid_argument("epochs=N data=PATH out=PATH tf=PATH");
    }
    check::Require(epochs > 0 && epochs <= 100, "threeCheapDefaultEpochs");
    std::filesystem::create_directories(root);
    const Map  data  = replay::ToMap(nn::load_npz((data_dir / "ptb_word.npz").string()));
    const auto words = Vocabulary(data_dir / "vocabulary.txt");
    std::filesystem::copy_file(
        data_dir / "ptb_word.npz",
        root / "data.npz",
        std::filesystem::copy_options::overwrite_existing);
    replay::CopyAssets(data_dir, root, {"manifest.json", "vocabulary.txt"});
    for (const std::string split : {"fit", "validation", "test"}) {
      const Tensor x = data.at(split + "_input"), y = data.at(split + "_target");
      check::Require(
          x.shape() == nn::Shape{split == "fit" ? 128 : 32, kSteps} && x.shape() == y.shape(),
          "16WordNextTokenWindows");
      const auto xv = x.to_vector<int>(), yv = y.to_vector<int>();
      for (std::size_t i = 0; i < xv.size(); ++i) {
        check::Require(
            xv[i] >= 0 && xv[i] < kVocab && yv[i] >= 0 && yv[i] < kVocab, "validPTBWordIDs");
        if (i % kSteps < kSteps - 1)
          check::Require(yv[i] == xv[i + 1], "exactNextWordTargetShift");
      }
    }
    TfParity(tf, data, words, epochs);

    Model m(42003);
    m.p.save((root / "initial_arrays.npz").string());
    nn::save_npz(
        (root / "initial_outputs.npz").string(),
        replay::ToNamed(m.Evaluate(data.at("test_input"), data.at("test_target"))));
    std::ofstream history(root / "history.tsv"), batches(root / "batches.tsv");
    history.precision(17);
    history << "epoch\tvalidation_nll\tvalidation_perplexity\n";
    batches << "epoch\tstep\trow\n";
    nn::Optimizer  opt(m.p, .003, replay::Adam());
    nn::BatchOrder order(128, 8, 42019);
    int            updates = 0;
    for (int epoch = 1; epoch <= epochs; ++epoch) {
      order.begin_epoch();
      for (;;) {
        const auto rows = order.next();
        if (rows.empty())
          break;
        const Tensor x = nn::take_rows(data.at("fit_input"), rows),
                     y = nn::take_rows(data.at("fit_target"), rows);
        if (!updates) {
          nn::save_npz((root / "first_batch.npz").string(), {{"input", x}, {"target", y}});
          nn::save_npz((root / "first_forward.npz").string(), replay::ToNamed(m.Evaluate(x, y)));
          nn::save_npz(
              (root / "initial_gradients.npz").string(),
              replay::ToNamed(replay::Named(m.p, m.Gradients(x, y))));
        }
        const auto path      = Sequence(m.p, nn::one_hot(x, kVocab));
        const auto loss      = Loss(path.logits, y);
        const auto gradients = nn::grad(loss, m.p.parameters());
        opt.step(gradients);
        if (!updates)
          m.p.save((root / "first_after.npz").string());
        for (auto row : rows)
          batches << epoch << '\t' << updates << '\t' << row << '\n';
        ++updates;
      }
      const Map    valid = m.Evaluate(data.at("validation_input"), data.at("validation_target"));
      const double nll   = valid.at("loss").item<double>();
      history << epoch << '\t' << nll << '\t' << std::exp(nll) << '\n';
      m.p.save((root / ("epoch" + std::to_string(epoch) + "_arrays.npz")).string());
      nn::save_npz(
          (root / ("epoch" + std::to_string(epoch) + "_validation.npz")).string(),
          replay::ToNamed(valid));
      std::cout << "epoch" << epoch << " validationNLL=" << nll << " perplexity=" << std::exp(nll)
                << std::endl;
    }
    const Map batch = replay::ToMap(nn::load_npz((root / "first_batch.npz").string()));
    nn::save_npz(
        (root / "final_gradients.npz").string(),
        replay::ToNamed(replay::Named(m.p, m.Gradients(batch.at("input"), batch.at("target")))));
    m.p.save((root / "final.npz").string());
    Model deployed(0);
    deployed.p.load((root / "final.npz").string());
    check::Require(
        replay::SameBits(replay::Arrays(m.p), replay::Arrays(deployed.p)),
        "allWordRnnArraysRestoreBitexact");
    const Map before = replay::Arrays(deployed.p);
    const Map out    = deployed.Evaluate(data.at("test_input"), data.at("test_target"));
    nn::save_npz((root / "outputs.npz").string(), replay::ToNamed(out));
    nn::save_npz((root / "model_arrays.npz").string(), replay::ToNamed(before));

    // Stateful one-word steps against the full unroll of test window 0.
    const Tensor       single = nn::take_rows(data.at("test_input"), {0});
    const auto         sv     = single.to_vector<std::int32_t>();
    Tensor             state  = nn::zeros({1, kUnits});
    std::vector<float> scores(kSteps * kVocab), states(kSteps * kUnits);
    const auto ol = out.at("logits").to_vector<float>(), oh = out.at("hidden").to_vector<float>();
    double     parity = 0;
    for (int t = 0; t < kSteps; ++t) {
      const auto step = deployed.StepOnce(nn::from_vector(std::vector<std::int32_t>{sv[t]}), state);
      state           = step.second;
      const auto lg = step.first.to_vector<float>(), hv = state.to_vector<float>();
      for (int c = 0; c < kVocab; ++c) {
        scores[t * kVocab + c] = lg[c];
        parity                 = std::max(parity, std::abs(double(lg[c]) - ol[t * kVocab + c]));
      }
      for (int h = 0; h < kUnits; ++h) {
        states[t * kUnits + h] = hv[h];
        parity                 = std::max(parity, std::abs(double(hv[h]) - oh[t * kUnits + h]));
      }
    }
    check::Require(parity < 2e-6, "fullWordUnrollAndStatefulStepParity");
    const Tensor jac = deployed.InputGradient(single);
    const auto   jv  = jac.to_vector<float>();
    for (int t = 9; t < kSteps; ++t)
      for (int c = 0; c < kVocab; ++c)
        check::Require(jv[t * kVocab + c] == 0, "allFutureWordInputPartialsExactlyZero");
    const Tensor step_logits = nn::from_vector(scores, {1, kSteps, kVocab});
    nn::save_npz(
        (root / "state_parity.npz").string(),
        {{"input", single},
         {"step_logits", step_logits},
         {"step_states", nn::from_vector(states, {1, kSteps, kUnits})},
         {"input_gradient", jac}});
    auto changed = sv;
    for (int t = 8; t < kSteps; ++t)
      changed[t] = (changed[t] + 317) % kVocab;
    const Tensor changed_t = nn::from_vector(changed, {1, kSteps});
    const Map    altered = deployed.Evaluate(changed_t, nn::take_rows(data.at("test_target"), {0}));
    const auto   av      = altered.at("logits").to_vector<float>();
    for (int i = 0; i < 8 * kVocab; ++i)
      check::Require(av[i] == scores[i], "futureWordsCannotChangeEarlierLogits");
    nn::save_npz(
        (root / "future_control.npz").string(),
        {{"input", changed_t}, {"logits", altered.at("logits")}});
    bool rejected = false;
    try {
      deployed.StepOnce(nn::from_vector(std::vector<std::int32_t>{kVocab}), state);
    } catch (const std::exception&) {
      rejected = true;
    }
    check::Require(rejected, "invalidWordIDRejected");

    const auto    prompts = Prompts(single, words);
    std::ofstream html(root / "index.html");
    html << "<!doctype html><meta charset='utf-8'><title>PTB word RNN</title><style>body{font:16px "
            "system-ui;margin:30px;max-width:1100px}pre{white-space:pre-wrap}</"
            "style><h1>Word-level PTB language model</h1><p>Pinned "
            "real Penn Treebank word preprocessing, full 10000-word vocabulary fitted on the "
            "training file only. Newlines become EOS; "
            "unknown words map to UNK 0. Compact 8-dimensional embedding, 8 tanh recurrent units, "
            "full 10000-way softmax, 170136 "
            "parameters. Three epochs, 48 Adam .003 updates on 128 disjoint 16-token windows; 32 "
            "each validation/test. This is a "
            "tiny teaching subset, not the full PTB benchmark. Zero state resets per window; "
            "generation carries state "
            "indefinitely. Target tokens never enter Step inference.</p><p>Likelihood is mean "
            "next-token NLL over all 512 test "
            "targets, perplexity = exp(mean NLL). Fitting-window-only add-one unigram/bigram "
            "baselines and unknown-token coverage "
            "are reported. No low-perplexity or language-quality claim.</p>";
    for (int i = 0; i < 4; ++i) {
      const Map sample = deployed.Sample(prompts[i], 64, 42029, kTemperatures[i]),
                repeat = deployed.Sample(prompts[i], 64, 42029, kTemperatures[i]);
      check::Require(replay::SameBits(sample, repeat), "statefulWordSamplingRepeatBitexact");
      nn::save_npz(
          (root / ("sample" + std::to_string(i) + ".npz")).string(), replay::ToNamed(sample));
      const std::string text = Decode(sample.at("codes"), words);
      std::ofstream(root / ("sample" + std::to_string(i) + ".txt")) << text;
      std::string escaped;
      for (char c : text)
        escaped += c == '&' ? "&amp;" : c == '<' ? "&lt;" : c == '>' ? "&gt;" : std::string(1, c);
      html << "<h2>Sample " << i << ", temperature " << kTemperatures[i] << "</h2><pre>" << escaped
           << "</pre>";
    }

    std::vector<int>                   unigram(kVocab, 1), context(kVocab, 0);
    std::map<std::pair<int, int>, int> pairs;
    const auto                         fi = data.at("fit_input").to_vector<int>(),
               ft                         = data.at("fit_target").to_vector<int>();
    for (int i = 0; i < 2048; ++i) {
      ++unigram[ft[i]];
      ++context[fi[i]];
      ++pairs[{fi[i], ft[i]}];
    }
    std::vector<std::int32_t> pair_array;
    for (const auto& [pair, count] : pairs)
      pair_array.insert(pair_array.end(), {pair.first, pair.second, count});
    nn::save_npz(
        (root / "baselines.npz").string(),
        {{"unigram_addone_counts",
          nn::from_vector(std::vector<std::int32_t>(unigram.begin(), unigram.end()))},
         {"context_counts",
          nn::from_vector(std::vector<std::int32_t>(context.begin(), context.end()))},
         {"observed_pair_counts", nn::from_vector(pair_array, {std::int64_t(pairs.size()), 3})}});
    double     unll = 0, bnll = 0;
    int        unknown = 0;
    const auto ti      = data.at("test_input").to_vector<int>(),
               tt      = data.at("test_target").to_vector<int>();
    for (int i = 0; i < 512; ++i) {
      unll -= std::log(double(unigram[tt[i]]) / 12048) / 512;
      bnll -= std::log(double(pairs[{ti[i], tt[i]}] + 1) / (context[ti[i]] + 10000)) / 512;
      unknown += tt[i] == 0;
    }
    // A full 10000x10000 matrix has at most 512 nonzero cells here. Print
    // exact sparse coordinates so every held-out word is represented without
    // allocating or printing 100 million zero entries.
    std::map<std::pair<int, int>, int> confusion;
    const auto predicted_words = nn::argmax(out.at("logits"), -1).to_vector<int>();
    for (std::size_t i = 0; i < tt.size(); ++i)
      ++confusion[{tt[i], predicted_words[i]}];
    std::cout << "confusion (truth rows, predicted columns; 10000 classes; sparse counts, omitted "
                 "cells are zero):\ntruth\tprediction\tcount\n";
    std::size_t counted = 0;
    for (const auto& [cell, count] : confusion) {
      std::cout << cell.first << '\t' << cell.second << '\t' << count << '\n';
      counted += count;
    }
    check::Require(counted == 512, "all held-out next-word targets counted");
    const double  nll = out.at("loss").item<double>();
    std::ofstream meta(root / "metrics.json");
    meta.precision(17);
    meta << "{\"parameters\":170136,\"parameter_tensors\":6,\"vocabulary\":10000,\"embedding\":8,"
            "\"hidden\":8,\"sequence\":16,"
            "\"epochs\":"
         << epochs << ",\"optimizer_updates\":" << updates << ",\"test_nll\":" << nll
         << ",\"test_perplexity\":" << std::exp(nll) << ",\"fit_unigram_nll\":" << unll
         << ",\"fit_bigram_nll\":" << bnll << ",\"uniform_nll\":" << std::log(10000.)
         << ",\"test_unknown_targets\":" << unknown
         << ",\"test_targets\":512,\"streaming_parity_max_error\":" << parity
         << ",\"sampled_words\":256}\n";
    check::Require(
        replay::SameBits(before, replay::Arrays(deployed.p)),
        "allWordLikelihoodGenerationControlsWeightsImmutable");
    check::Require(
        bool(meta) && bool(html) && bool(history) && bool(batches), "completeWordLmRecords");
    std::cout << "testNLL=" << nll << " perplexity=" << std::exp(nll) << " fitUnigram=" << unll
              << " fitBigram=" << bnll << " unknown=" << unknown
              << "/512;wordBPTT/causal/streaming/restore/256draws/immutable PASS\nArtifacts:"
              << root / "index.html"
              << "\n(TF recorded: testNLL=8.28867 perplexity=3978.53 fitUnigram=7.59409 "
                 "fitBigram=8.97219 unknown=18/512)\nwall "
              << std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count()
              << " s\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
