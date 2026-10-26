// CMD: just run-nn beam_search
// State-owning beam search on the trained real translation decoder and a
// non-greedy exhaustive fixture.
//
// Port of tflow's beam_search. Model: the English->French attention model
// trained by the translation example (common/translation.hpp, 5640
// parameters). TF restores tflow's translation checkpoint; this port loads the
// trained weights from model=PATH, by default the first of
//   output/nn/models/translation/english_french/{final.npz,model_arrays.npz}
//   (this repository's translation port) and
//   tflow's output/models/translation/english_french/model_arrays.npz
//   (the weights TF's beam_search restores, written by tflow's translation run).
// Data: tflow's output/datasets/translation (data.npz: 6-token
// English/French pairs over a 256-token vocabulary; vocabulary.txt,
// attribution.tsv, _about.txt).
// Every test sentence is encoded without labels; beams of width 1 and 3 (cap
// 6 tokens, EOS = 2) expand hypotheses that each own their decoder state and
// the sentence's memory. Checks: width 1 equals actual greedy decoding; on the
// 3-token fixture beam search beats greedy, and the best of all 15 terminated
// or capped paths (exhaustive enumeration) equals the beam's best;
// immutability of all 5640 weights. When the weights are TF's, the beams and
// the expansion ledger are compared with tflow's recorded beam_search output.
// Options: model=PATH data=PATH tf=PATH (tflow's output/models/beam_search).
// Output: output/nn/models/beam_search/.

#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>

#include "common/beam_search.hpp"
#include "common/checks.hpp"
#include "common/paths.hpp"
#include "common/seq2seq.hpp"
#include "common/tf_replay.hpp"
#include "common/translation.hpp"

namespace nn = datamunge::nn;
using nn::Tensor;
using replay::Map;
using namespace translation;

namespace {

Tensor Zero(std::int64_t n) { return nn::zeros({n, kUnits}); }

nn::NamedTensors SaveBeams(const std::vector<std::vector<beam::Hypothesis>>& beams, int width) {
  const int                 n = static_cast<int>(beams.size());
  std::vector<std::int32_t> tokens(n * width * 6, 0), lengths(n * width), ended(n * width);
  std::vector<double>       scores(n * width);
  std::vector<float>        h(n * width * 8), c(n * width * 8);
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < width; ++j) {
      const auto& b = beams[i][j];
      const int   k = i * width + j;
      lengths[k]    = static_cast<int>(b.tokens.size());
      ended[k]      = b.ended;
      scores[k]     = b.score;
      for (std::size_t t = 0; t < b.tokens.size(); ++t) tokens[k * 6 + t] = b.tokens[t];
      std::copy(b.h.begin(), b.h.end(), h.begin() + k * 8);
      std::copy(b.c.begin(), b.c.end(), c.begin() + k * 8);
    }
  return {{"tokens", nn::from_vector(tokens, {n, width, 6})}, {"lengths", nn::from_vector(lengths, {n, width})},
          {"ended", nn::from_vector(ended, {n, width})},      {"scores", nn::from_vector(scores, {n, width})},
          {"h", nn::from_vector(h, {n, width, 8})},           {"c", nn::from_vector(c, {n, width, 8})}};
}

std::filesystem::path DefaultModel() {
  const auto ours = paths::OutputRoot() / "models/translation/english_french";
  for (auto f : {"final.npz", "model_arrays.npz"})
    if (std::filesystem::exists(ours / f)) return ours / f;
  return replay::TfModels("translation") / "english_french" / "model_arrays.npz";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const auto            start = std::chrono::steady_clock::now();
    std::filesystem::path model = DefaultModel(), data = paths::Dataset("translation"), tf = replay::TfModels("beam_search");
    for (int i = 1; i < argc; ++i) {
      const std::string a = argv[i];
      if (a.rfind("model=", 0) == 0) model = a.substr(6);
      else if (a.rfind("data=", 0) == 0) data = a.substr(5);
      else if (a.rfind("tf=", 0) == 0) tf = a.substr(3);
      else throw std::invalid_argument("model=PATH data=PATH tf=PATH");
    }
    const auto root = paths::OutputDir("models/beam_search");
    check::Require(std::filesystem::exists(model), "trained translation weights at " + model.string() +
                                                       " (run the translation example first, or pass model=PATH)");
    nn::ParameterStore p(0);
    Build(p);
    check::Require(p.count() == 5640 && p.size() == 12, "5640 translation parameters");
    replay::LoadInto(p, model);
    std::cout << "translation weights: " << model.string() << "\n";
    const Map before = replay::Arrays(p);
    nn::save_npz((root / "model_arrays.npz").string(), replay::ToNamed(before));
    const Map test = seq2seq::LoadPools(data / "data.npz")[2];
    replay::CopyAssets(data, root, {"data.npz", "manifest.json", "vocabulary.txt", "attribution.tsv", "_about.txt"});

    // label-free source encoding
    nn::NoGradGuard guard;
    const Tensor    len = test.at("input_lengths");
    const auto      enc = seq2seq::Encode(p, seq2seq::Embedding(p, nn::one_hot(test.at("input"), kVocab), kVocab), len, Zero(32), Zero(32));
    const Tensor    eh = nn::stack_time(enc.h), ec = nn::stack_time(enc.c);
    nn::save_npz((root / "encoder.npz").string(), {{"h", eh}, {"c", ec}, {"memory", enc.memory}});
    auto step = [&](const Tensor& token, const Tensor& h, const Tensor& c, const Tensor& memory, const Tensor& lengths) {
      const auto s = TranslationStep(p, seq2seq::Embedding(p, nn::one_hot(token, kVocab), kVocab), {h, c}, memory, lengths);
      return seq2seq::DecoderStep{s.logits, s.state.h, s.state.c, s.weights};
    };

    std::vector<std::vector<beam::Hypothesis>> one, three;
    std::vector<std::vector<float>>            logitbank, previous_h, previous_c, next_h, next_c;
    std::vector<std::vector<int>>              prefixes;
    std::vector<int>                           cases, widths;
    const auto                                 lengths = len.to_vector<std::int32_t>();
    const auto                                 hv = eh.to_vector<float>(), cv = ec.to_vector<float>();
    for (int width : {1, 3}) {
      auto& all = width == 1 ? one : three;
      for (int b = 0; b < 32; ++b) {
        const Tensor     memory = nn::slice_axis(enc.memory, 0, b, b + 1);
        const Tensor     length = nn::from_vector(std::vector<std::int32_t>{lengths[b]});
        beam::Hypothesis initial;
        initial.h.assign(hv.begin() + (b * 7 + 6) * 8, hv.begin() + (b * 7 + 7) * 8);
        initial.c.assign(cv.begin() + (b * 7 + 6) * 8, cv.begin() + (b * 7 + 7) * 8);
        auto decoder = [&](const beam::Hypothesis& parent) {
          const Tensor token = nn::from_vector(std::vector<std::int32_t>{parent.tokens.empty() ? seq2seq::kBos : parent.tokens.back()});
          const auto   out   = step(token, nn::from_vector(parent.h, {1, 8}), nn::from_vector(parent.c, {1, 8}), memory, length);
          beam::Step   result{out.logits.to_vector<float>(), out.h.to_vector<float>(), out.c.to_vector<float>()};
          cases.push_back(b);
          widths.push_back(width);
          prefixes.push_back(parent.tokens);
          logitbank.push_back(result.logits);
          previous_h.push_back(parent.h);
          previous_c.push_back(parent.c);
          next_h.push_back(result.h);
          next_c.push_back(result.c);
          return result;
        };
        all.push_back(beam::BeamSearch(initial, width, 6, seq2seq::kEos, decoder));
      }
    }
    const auto w1 = SaveBeams(one, 1), w3 = SaveBeams(three, 3);
    nn::save_npz((root / "width1.npz").string(), w1);
    nn::save_npz((root / "width3.npz").string(), w3);
    {
      const auto greedy = seq2seq::Greedy([&](const Tensor& t, const Tensor& h, const Tensor& c) { return step(t, h, c, enc.memory, len); },
                                          enc.h.back(), enc.c.back())
                              .to_vector<int>();
      for (int b = 0; b < 32; ++b)
        for (int t = 0; t < 6; ++t)
          check::Require(greedy[b * 6 + t] == (t < static_cast<int>(one[b][0].tokens.size()) ? one[b][0].tokens[t] : 0),
                         "widthOneIsActualGreedy");
    }
    const int n = static_cast<int>(cases.size());
    {
      std::vector<std::int32_t> prefix(n * 6, -1), lens(n);
      for (int i = 0; i < n; ++i) {
        lens[i] = static_cast<int>(prefixes[i].size());
        for (std::size_t j = 0; j < prefixes[i].size(); ++j) prefix[i * 6 + j] = prefixes[i][j];
      }
      nn::NamedTensors ledger{{"prefix", nn::from_vector(prefix, {n, 6})}, {"lengths", nn::from_vector(lens)},
                              {"cases", nn::from_vector(std::vector<std::int32_t>(cases.begin(), cases.end()))},
                              {"widths", nn::from_vector(std::vector<std::int32_t>(widths.begin(), widths.end()))}};
      for (const auto& [name, bank] : std::vector<std::pair<std::string, std::vector<std::vector<float>>*>>{
               {"logits", &logitbank}, {"previous_h", &previous_h}, {"previous_c", &previous_c}, {"next_h", &next_h}, {"next_c", &next_c}}) {
        std::vector<float> flat;
        for (const auto& row : *bank) flat.insert(flat.end(), row.begin(), row.end());
        ledger.emplace_back(name, nn::from_vector(flat, {n, static_cast<std::int64_t>((*bank)[0].size())}));
      }
      nn::save_npz((root / "expansions.npz").string(), ledger);
      // Same weights as TF's run: the beams and every expansion must agree.
      const bool tf_weights = std::filesystem::exists(tf / "model_arrays.npz") &&
                              replay::MaxDiff(before, tf / "model_arrays.npz") == 0 && std::filesystem::exists(tf / "width3.npz");
      if (tf_weights) {
        const auto t1 = replay::ToMap(nn::load_npz((tf / "width1.npz").string())), t3 = replay::ToMap(nn::load_npz((tf / "width3.npz").string()));
        const auto te = replay::ToMap(nn::load_npz((tf / "expansions.npz").string()));
        check::Require(replay::SameBits(replay::Get(w1, "tokens"), t1.at("tokens")) && replay::SameBits(replay::Get(w3, "tokens"), t3.at("tokens")),
                       "width1/width3 beam tokens identical to TF's");
        check::Require(te.at("logits").shape()[0] == n, "same number of expansions as TF");
        const double ds = std::max(replay::MaxAbsDiff(replay::Get(w1, "scores"), t1.at("scores")),
                                   replay::MaxAbsDiff(replay::Get(w3, "scores"), t3.at("scores")));
        double       dl = 0;
        for (auto k : {"logits", "next_h", "next_c"}) dl = std::max(dl, replay::MaxAbsDiff(replay::Get(ledger, k), te.at(k)));
        check::Near(ds, 0, 1e-5, "beam scores match TF");
        check::Near(dl, 0, 1e-5, "every expansion's logits and next state match TF");
        std::cout << " vs TF beam_search (same weights): tokens identical, " << n << " expansions, scores max diff=" << ds
                  << ", expansion logits/states max diff=" << dl << "\n";
      }
    }
    beam::Hypothesis initial;
    initial.h = {0};
    initial.c = {0};
    const auto fixture1 = beam::BeamSearch(initial, 1, 3, 2, beam::FixtureStep), fixture3 = beam::BeamSearch(initial, 3, 3, 2, beam::FixtureStep);
    check::Require(fixture1[0].tokens == std::vector<int>{0, 2} && fixture3[0].tokens == std::vector<int>{1, 2} &&
                       fixture3[0].score > fixture1[0].score,
                   "nonGreedyBetterSequenceAndBranchOwnedStates");
    std::vector<beam::Hypothesis>          exhaustive;
    std::function<void(beam::Hypothesis)> enumerate = [&](beam::Hypothesis parent) {
      if (parent.ended || parent.tokens.size() == 3) {
        exhaustive.push_back(parent);
        return;
      }
      const auto s  = beam::FixtureStep(parent);
      const auto lp = beam::LogSoftmax(s.logits);
      for (int token = 0; token < 3; ++token) {
        auto child = parent;
        child.tokens.push_back(token);
        child.score += lp[token];
        child.ended = token == 2;
        child.h     = s.h;
        child.c     = s.c;
        enumerate(child);
      }
    };
    enumerate(initial);
    std::sort(exhaustive.begin(), exhaustive.end(), beam::Better);
    check::Require(exhaustive.size() == 15 && exhaustive[0].tokens == fixture3[0].tokens, "all15TerminatedOrCappedPathsBestMatchesBeam");
    check::Near(fixture3[0].score, exhaustive[0].score, 1e-12, "exhaustiveFixtureScore");
    {
      std::ofstream fixture(root / "fixture.tsv");
      fixture.precision(17);
      fixture << "tokens\tscore\tended\th\tc\n";
      for (const auto& b : exhaustive) {
        for (auto t : b.tokens) fixture << t;
        fixture << '\t' << b.score << '\t' << b.ended << '\t' << b.h[0] << '\t' << b.c[0] << '\n';
      }
    }
    check::Require(replay::SameBits(replay::Arrays(p), before), "beamAll5640WeightsImmutable");
    std::ofstream metrics(root / "metrics.json");
    metrics << "{\"width\":3,\"cap\":6,\"expansions\":" << n << ",\"fixture_paths\":15,\"fixture_greedy_score\":" << fixture1[0].score
            << ",\"fixture_beam_score\":" << fixture3[0].score << "}\n";
    int improved = 0;
    for (int b = 0; b < 32; ++b) improved += three[b][0].score > one[b][0].score + 1e-12;
    std::cout << "realWidth1=greedy, width3 scored per-hypothesis states;expansions=" << n << " (width3 beats greedy on " << improved
              << "/32 sentences);fixture greedy " << fixture1[0].score << " vs beam " << fixture3[0].score
              << ";15exhaustiveFixturePaths/beamBeatsGreedy/EOSonce/cap/restore/immutable PASS\n";
    std::cout << "wall " << std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() << " s\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
