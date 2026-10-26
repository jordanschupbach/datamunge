// CMD: just run-nn hidden_markov_models
// Categorical HMM forward-backward, Viterbi decoding and Baum-Welch learning.
//
// Port of tflow's hidden_markov_models (model in common/hmm.hpp, float64 nn
// ops). Enumerate all 16 paths of a two-state, four-symbol sequence and
// compare its likelihood, posterior marginals, most likely path and one EM
// update with the nn implementation. Then fit 160 independent length-20
// sequences, check monotone likelihood and evaluate new sequences and their
// latent-state decoding. A one-symbol sequence verifies the no-transition
// update. Observations alone enter Baum-Welch; simulated state labels are used
// only for the final metric. The samples come from the same std::mt19937
// stream as the TF version, so both runs fit identical data. No files are
// written (the TF example writes none either).

#include <array>
#include <cmath>
#include <cstdint>
#include <datamunge/nn/nn.hpp>
#include <iostream>
#include <random>
#include <vector>

#include "common/checks.hpp"
#include "common/hmm.hpp"

namespace nn = datamunge::nn;
using nn::DType;
using nn::Tensor;

namespace {

ml::HmmState State(bool truth) {
  return {nn::tensor({truth ? .6 : .5, truth ? .4 : .5}, {2}, DType::Float64),
          nn::tensor({truth ? .8 : .6, truth ? .2 : .4, truth ? .3 : .4, truth ? .7 : .6}, {2, 2}, DType::Float64),
          nn::tensor({truth ? .65 : .55, .3, truth ? .05 : .15, truth ? .05 : .15, truth ? .25 : .3, truth ? .7 : .55},
                     {2, 3}, DType::Float64)};
}

void Enumerate() {
  const auto p = State(true);
  const auto initial = p.initial.to_vector<double>(), trans = p.transition.to_vector<double>(),
             emit = p.emission.to_vector<double>();
  const int  sequence[]{0, 1, 2, 1};
  const auto observations = nn::from_vector(std::vector<std::int32_t>{0, 1, 2, 1}, {1, 4});
  double     total = 0, best = -1;
  std::array<int, 4> best_path{};
  double posterior[4][2]{}, transition[2][2]{}, emission[2][3]{};
  for (int path_code = 0; path_code < 16; ++path_code) {
    std::array<int, 4> path{};
    for (int t = 0; t < 4; ++t) path[t] = (path_code >> (3 - t)) & 1;
    double probability = initial[path[0]];
    for (int t = 0; t < 4; ++t) {
      probability *= emit[path[t] * 3 + sequence[t]];
      if (t) probability *= trans[path[t - 1] * 2 + path[t]];
    }
    total += probability;
    if (probability > best) best = probability, best_path = path;
    for (int t = 0; t < 4; ++t) {
      posterior[t][path[t]] += probability;
      emission[path[t]][sequence[t]] += probability;
      if (t) transition[path[t - 1]][path[t]] += probability;
    }
  }
  ml::HiddenMarkovModel model(2, 3, 4);
  const auto            result = model.Run(observations, p);
  check::Near(result.log_likelihood.at({0}), std::log(total), 1e-12, "forward likelihood vs all paths");
  check::Near(result.viterbi_score.at({0}), std::log(best), 1e-12, "Viterbi path score vs enumeration");
  const auto path = result.path.to_vector<std::int32_t>();
  for (int t = 0; t < 4; ++t) {
    check::Require(path[t] == best_path[t], "Viterbi path vs enumeration");
    for (int c = 0; c < 2; ++c)
      check::Near(result.posterior.at({0, t, c}), posterior[t][c] / total, 1e-12,
                  "forward-backward marginals vs enumeration");
  }
  for (int c = 0; c < 2; ++c) {
    check::Near(result.next.initial.at({c}), posterior[0][c] / total, 1e-12, "Baum-Welch initial posterior");
    const double transitions = transition[c][0] + transition[c][1],
                 emissions   = emission[c][0] + emission[c][1] + emission[c][2];
    for (int j = 0; j < 2; ++j)
      check::Near(result.next.transition.at({c, j}), transition[c][j] / transitions, 1e-12,
                  "Baum-Welch expected transitions");
    for (int j = 0; j < 3; ++j)
      check::Near(result.next.emission.at({c, j}), emission[c][j] / emissions, 1e-12, "Baum-Welch expected emissions");
  }
  ml::HiddenMarkovModel singleton(2, 3, 1);
  const auto single = singleton.Run(nn::from_vector(std::vector<std::int32_t>{2}, {1, 1}), p);
  check::Near(single.log_likelihood.at({0}), std::log(.6 * .05 + .4 * .7), 1e-12, "single-symbol likelihood");
  const auto kept = single.next.transition.to_vector<double>();
  for (int i = 0; i < 4; ++i) check::Near(kept[i], trans[i], 0, "no-transition row retained");
  std::cout << "  exhaustive 16-path likelihood=" << total << "; Viterbi path=";
  for (int c : best_path) std::cout << c;
  std::cout << "; all posteriors/EM counts and singleton checks passed\n";
}

std::pair<Tensor, Tensor> Sample(int count, int length, const ml::HmmState& model, std::mt19937& rng) {
  std::vector<std::int32_t> observations(count * length), states(count * length);
  const auto initial = model.initial.to_vector<double>(), transition = model.transition.to_vector<double>(),
             emission = model.emission.to_vector<double>();
  auto draw = [&](const double* p, int n) { return std::discrete_distribution<int>(p, p + n)(rng); };
  for (int b = 0; b < count; ++b) {
    int state = draw(initial.data(), 2);
    for (int t = 0; t < length; ++t) {
      states[b * length + t]       = state;
      observations[b * length + t] = draw(emission.data() + state * 3, 3);
      state                        = draw(transition.data() + state * 2, 2);
    }
  }
  return {nn::from_vector(observations, {count, length}), nn::from_vector(states, {count, length})};
}

void Learn() {
  std::mt19937 rng(101);
  const auto [observations, hidden] = Sample(160, 20, State(true), rng);
  const auto [test, test_hidden]    = Sample(60, 20, State(true), rng);
  auto       state                  = State(false);
  ml::HiddenMarkovModel model(2, 3, 20);
  auto sum = [](const Tensor& values) { return nn::sum(values).item(); };
  double previous = -1e100, initial = 0;
  for (int step = 0; step <= 45; ++step) {
    const auto   result     = model.Run(observations, state);
    const double likelihood = sum(result.log_likelihood);
    check::Require(likelihood >= previous - 1e-8, "Baum-Welch likelihood must not decrease");
    if (!step) initial = likelihood;
    if (step % 15 == 0) std::cout << "  EM step " << step << ": training log-likelihood=" << likelihood << '\n';
    previous = likelihood;
    if (step < 45) state = result.next;
  }
  check::Require(previous > initial + 40, "Baum-Welch improves sequence model");
  const auto result = model.Run(test, state);
  const auto path = result.path.to_vector<std::int32_t>(), truth = test_hidden.to_vector<std::int32_t>();
  int        correct = 0;
  for (size_t i = 0; i < path.size(); ++i) correct += path[i] == truth[i];
  const double decoding = std::max<int>(correct, int(path.size()) - correct) / double(path.size());
  check::Require(decoding > .75, "held-out latent-state decoding up to permutation");
  const auto score = result.viterbi_score.to_vector<double>(), ll = result.log_likelihood.to_vector<double>(),
             post  = result.posterior.to_vector<double>();
  for (int b = 0; b < 60; ++b) {
    check::Require(score[b] <= ll[b] + 1e-12, "best path mass bounded by total sequence mass");
    for (int t = 0; t < 20; ++t)
      check::Near(post[(b * 20 + t) * 2] + post[(b * 20 + t) * 2 + 1], 1, 1e-12, "state posterior normalization");
  }
  std::cout << "  held-out mean log-likelihood=" << sum(result.log_likelihood) / 60
            << "; Viterbi state accuracy=" << decoding << '\n';
}

}  // namespace

int main() {
  try {
    std::cout << "== Hidden Markov models\n";
    Enumerate();
    Learn();
    std::cout << "All HMM checks passed.\n";
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
  return 0;
}
