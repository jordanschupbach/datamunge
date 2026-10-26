// CMD: just run-nn token_sampling
// Greedy, beam, top-k, nucleus and temperature decoding with independent
// controls.
//
// Port of tflow's token_sampling (common/ml/token_sampling.h ->
// examples/nn/common/token_sampling.hpp). Part 1: eleven hand-made 4-token
// distributions (temperature, top-k, top-p, combined, ties, k beyond the
// vocabulary, extreme and -inf logits) with five fixed uniforms each; a toy
// predictor on which a width-2 beam escapes greedy's local choice and a beam
// that emits EOS stops growing; twelve rejected arguments. Part 2: the trained
// gpt_lm character model decodes 64 tokens after "First Citizen:\n" with
// greedy / temperature 1 / .5 / 1.5 / top-5 / nucleus .8 (std::mt19937(46211)
// uniforms) and a width-3 beam for 12 steps; the weights must be unchanged.
//
// The trained GPT: gpt=auto (default) loads TF's gpt_lm model_arrays.npz when
// tflow's outputs are present, so every logit and decision can be compared
// with TF's records; gpt=ours loads our gpt_lm run's final.npz (run gpt_lm
// first), gpt=PATH any compatible .npz. Options: out=DIR tf=DIR gpt=auto
// threads=4 verbose=0. Output: output/nn/models/token_sampling/.
#include "common/token_sampling.hpp"
#include "common/transformer_lm.hpp"

namespace {

using lm::Tensor;
namespace nn = datamunge::nn;
namespace fs = std::filesystem;

void SaveBeam(const fs::path& p, const std::vector<lm::TokenBeam>& beam) {
  std::size_t max = 0;
  for (auto& b : beam) max = std::max(max, b.tokens.size());
  const auto                n = static_cast<std::int64_t>(beam.size());
  std::vector<std::int32_t> tokens(beam.size() * max, -1), lengths;
  std::vector<double>       scores;
  std::vector<std::int32_t> finished;
  for (std::size_t i = 0; i < beam.size(); ++i) {
    lengths.push_back(static_cast<std::int32_t>(beam[i].tokens.size()));
    scores.push_back(beam[i].score);
    finished.push_back(beam[i].finished);
    for (std::size_t j = 0; j < beam[i].tokens.size(); ++j) tokens[i * max + j] = beam[i].tokens[j];
  }
  nn::save_npz(p.string(), {{"tokens", nn::from_vector(tokens, {n, static_cast<std::int64_t>(max)})},
                            {"lengths", nn::from_vector(lengths, {n})},
                            {"scores", nn::from_vector(scores, {n})},
                            {"finished", nn::cast(nn::from_vector(finished, {n}), nn::DType::Bool)}});
}

int Run(const lm::Options& o) {
  const auto     started = std::chrono::steady_clock::now();
  const fs::path root    = o.out;
  const std::vector<double> base{std::log(.4), std::log(.3), std::log(.2), std::log(.1)};
  struct Fixture {
    std::string         name;
    std::vector<double> logits;
    double              temperature;
    int                 k;
    double              p;
  };
  std::vector<std::string> fixture_names;
  for (const auto& c : std::vector<Fixture>{{"plain", base, 1, 0, 1},
                                            {"cold", base, .5, 0, 1},
                                            {"hot", base, 2, 0, 1},
                                            {"top2", base, 1, 2, 1},
                                            {"nucleus", base, 1, 0, .6},
                                            {"tiny_nucleus", base, 1, 0, .01},
                                            {"combined", base, 1, 3, .6},
                                            {"ties", {0, 0, 0, 0}, 1, 2, 1},
                                            {"large_k", base, 1, 99, 1},
                                            {"extreme", {10000, 9999, -10000, -INFINITY}, 1, 0, 1},
                                            {"one_finite", {-INFINITY, 0, -INFINITY, -INFINITY}, 1, 0, 1}}) {
    const auto                d = lm::FilterTokens(c.logits, c.temperature, c.k, c.p);
    std::vector<std::int32_t> retained(4, 0), choices;
    for (int i : d.retained) retained[static_cast<std::size_t>(i)] = 1;
    const std::vector<double> uniform{0, .199, .4, .8, std::nextafter(1., 0.)};
    for (double u : uniform) choices.push_back(lm::DrawToken(d.probability, u));
    nn::save_npz((root / (c.name + ".npz")).string(),
                 {{"logits", nn::from_vector(c.logits, {4})},
                  {"probability", nn::from_vector(d.probability, {4})},
                  {"retained", nn::cast(nn::from_vector(retained, {4}), nn::DType::Bool)},
                  {"metadata", nn::from_vector(std::vector<double>{c.temperature, double(c.k), c.p, d.retained_mass}, {4})},
                  {"uniform", nn::from_vector(uniform, {5})},
                  {"choices", nn::from_vector(choices, {5})}});
    fixture_names.push_back(c.name);
  }
  const lm::TokenPredictor toy = [](const std::vector<int>& tokens) {
    std::vector<double> p = tokens.size() == 1   ? std::vector<double>{.6, .39, .01}
                            : tokens.back() == 1 ? std::vector<double>{.01, .01, .98}
                                                 : std::vector<double>{.34, .33, .33};
    for (auto& v : p) v = std::log(v);
    return p;
  };
  const auto one = lm::BeamTokens(toy, {99}, 3, 1, 2), two = lm::BeamTokens(toy, {99}, 3, 2, 2),
             all = lm::BeamTokens(toy, {99}, 3, 27, 2);
  check::Require(two[0].score > one[0].score, "beam escapes greedy local choice");
  check::Require(two[0].tokens == std::vector<int>({99, 1, 2}) && two[0].finished, "finished EOS beam not extended");
  SaveBeam(root / "toy_width1.npz", one);
  SaveBeam(root / "toy_width2.npz", two);
  SaveBeam(root / "toy_exhaustive.npz", all);
  int  rejected = 0;
  auto reject   = [&](auto f) {
    try {
      f();
    } catch (const std::invalid_argument&) {
      ++rejected;
    }
  };
  reject([&] { lm::FilterTokens({}); });
  reject([&] { lm::FilterTokens({-INFINITY, -INFINITY}); });
  reject([&] { lm::FilterTokens({NAN, 0}); });
  reject([&] { lm::FilterTokens({INFINITY, 0}); });
  reject([&] { lm::FilterTokens(base, 0); });
  reject([&] { lm::FilterTokens(base, -1); });
  reject([&] { lm::FilterTokens(base, 1, -1); });
  reject([&] { lm::FilterTokens(base, 1, 0, 0); });
  reject([&] { lm::FilterTokens(base, 1, 0, 1.1); });
  reject([&] { lm::DrawToken({1}, 1); });
  reject([&] { lm::DrawToken({.5}, .1); });
  reject([&] { lm::BeamTokens(toy, {}, 3, 2); });
  check::Require(rejected == 12, "invalid distribution and search arguments rejected");

  const fs::path weights = lm::TrainedGpt(o.gpt);
  std::cout << "trained GPT weights: " << weights.string() << "\n";
  lm::TransformerLM model(false);
  lm::LoadArrays(model.store, weights);
  const auto before = model.Arrays();
  nn::save_npz((root / "model_arrays.npz").string(), before);
  fs::copy_file(lm::GptVocabulary(), root / "vocabulary.txt", fs::copy_options::overwrite_existing);
  const std::string        vocab  = lm::ReadText(root / "vocabulary.txt");
  const std::vector<int>   prefix = lm::Encode(vocab, "First Citizen:\n");
  const lm::TokenPredictor predict = [&](const std::vector<int>& tokens) {
    const int                 length = std::min(64, static_cast<int>(tokens.size()));
    const auto                start  = tokens.size() - static_cast<std::size_t>(length);
    std::vector<std::int32_t> input(64, 0);
    for (int j = 0; j < length; ++j) input[static_cast<std::size_t>(j)] = tokens[start + static_cast<std::size_t>(j)];
    const auto out = model.Predict(nn::from_vector(input, {1, 64})).to_vector<float>();
    return std::vector<double>(out.begin() + (length - 1) * 65, out.begin() + length * 65);
  };
  std::ofstream settings(root / "settings.tsv");
  settings << "name\ttemperature\ttop_k\ttop_p\n";
  struct Setting {
    std::string name;
    double      temperature;
    int         k;
    double      p;
    bool        greedy;
  };
  std::vector<std::string> model_files;
  for (const auto& setting : std::vector<Setting>{{"greedy", 1, 0, 1, true},
                                                  {"temperature", 1, 0, 1, false},
                                                  {"cold", .5, 0, 1, false},
                                                  {"hot", 1.5, 0, 1, false},
                                                  {"top_k", 1, 5, 1, false},
                                                  {"nucleus", 1, 0, .8, false}}) {
    auto                tokens = prefix;
    std::mt19937        rng(46211);
    std::vector<double> logits, prob, u, mass;
    for (int t = 0; t < 64; ++t) {
      const auto   z       = predict(tokens);
      const auto   d       = lm::FilterTokens(z, setting.temperature, setting.k, setting.p);
      const double uniform = (double(rng()) + .5) / 4294967296.;
      tokens.push_back(setting.greedy ? lm::GreedyToken(z) : lm::DrawToken(d.probability, uniform));
      logits.insert(logits.end(), z.begin(), z.end());
      prob.insert(prob.end(), d.probability.begin(), d.probability.end());
      u.push_back(uniform);
      mass.push_back(d.retained_mass);
    }
    std::ofstream text(root / (setting.name + ".txt"));
    for (int c : tokens) text << vocab.at(static_cast<std::size_t>(c));
    nn::save_npz((root / ("model_" + setting.name + ".npz")).string(),
                 {{"codes", nn::from_vector(std::vector<std::int32_t>(tokens.begin(), tokens.end()),
                                            {static_cast<std::int64_t>(tokens.size())})},
                  {"logits", nn::from_vector(logits, {64, 65})},
                  {"probability", nn::from_vector(prob, {64, 65})},
                  {"uniform", nn::from_vector(u, {64})},
                  {"retained_mass", nn::from_vector(mass, {64})}});
    model_files.push_back("model_" + setting.name + ".npz");
    settings << setting.name << '\t' << setting.temperature << '\t' << setting.k << '\t' << setting.p << '\n';
  }
  SaveBeam(root / "model_beam.npz", lm::BeamTokens(predict, prefix, 12, 3));
  model_files.push_back("model_beam.npz");
  check::Require(lm::SameBits(before, model.Arrays()), "sampling and beam do not modify model");
  std::ofstream metrics(root / "metrics.json");
  metrics << "{\"fixture_distributions\":11,\"rejected_inputs\":12,\"model_decisions\":384,\"beam_width\":3,"
             "\"beam_steps\":12,\"toy_greedy_score\":"
          << one[0].score << ",\"toy_beam_score\":" << two[0].score << "}\n";
  std::cout << "11distribution fixtures/12guards; EOS/exhaustive toy beam beats greedy (" << one[0].score << " -> "
            << two[0].score << "); 384 actual trained-GPT greedy/temperature/top-k/nucleus decisions, width3 "
            << "beam12steps, immutable PASS\n";

  // TF comparison: the fixtures must agree exactly (same double arithmetic);
  // the model decodes are compared when the weights are TF's.
  double fixtures = 0;
  for (const auto& n : fixture_names) fixtures = std::max(fixtures, lm::CompareRecord(o, n + ".npz", nn::load_npz((root / (n + ".npz")).string())));
  for (auto f : {"toy_width1.npz", "toy_width2.npz", "toy_exhaustive.npz"})
    fixtures = std::max(fixtures, lm::CompareRecord(o, f, nn::load_npz((root / f).string())));
  check::Require(fixtures <= 1e-12, "distribution fixtures and toy beams equal TF's");
  const double same_weights = lm::CompareRecord(o, "model_arrays.npz", before);
  if (same_weights == 0) {
    double codes = 0;
    for (const auto& f : model_files)
      codes = std::max(codes, lm::CompareRecord(o, f, nn::load_npz((root / f).string()),
                                                [](const std::string& k) { return k == "codes" || k == "tokens"; }));
    for (const auto& f : model_files) lm::CompareRecord(o, f, nn::load_npz((root / f).string()));
    check::Require(codes == 0, "every greedy/sampled/beam token equals TF's with TF's weights");
    std::cout << "all 384 decisions and the beam equal TF's\n";
  }
  std::cout << "wall " << std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count() << " s\n";
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    return Run(lm::ParseOptions(argc, argv, "token_sampling", false));
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
