// CMD: just run-nn attention_visualization
// Attention maps, row concentration and measured key-removal effects.
//
// Port of tflow's attention_visualization. Two heads, 4 queries, 5 keys with
// a hand-made validity pattern (query t of the first three sees keys 0..t+1;
// the last row of head 0 is empty, the last row of head 1 sees one key). For
// the actual attention maps it reports per-row entropy, effective support
// exp(entropy) (0 for an empty row, 1 for a singleton) and the max weight,
// then measures each key's influence by removing it from the mask and
// re-running attention: removal_effect[h, t, r] = ||out - out_without_r||.
// Run twice: distinct random values, and equal values (every key carries the
// same value vector), where removing any key from a row that keeps another
// key changes nothing — attention weights alone do not say what a key
// contributes.
//
// Inputs come from TF's std::mt19937 stream, so when TF's fixtures (tflow's
// output/models/attention_visualization/*.npz) are present every array is
// compared with TF's (must agree to 1e-5). The plot (attention_effects.png)
// and index.html are made by tflow's separate report script, not by the TF
// binary, and are not ported.
//
// TF -> datamunge: nn::ScaledDotProductAttention with a constant boolean
// `allowed` is attn::ScaledDotProductAttention with a bool keep mask; the
// TF reductions become nn ops.
//
// Options: out=DIR (default output/nn/attention_visualization), tf=DIR, verbose=1.

#include <cstdio>
#include <iostream>
#include <string>

#include "common/attention.hpp"
#include "common/checks.hpp"
#include "common/paths.hpp"

namespace nn = datamunge::nn;
using nn::Tensor;

namespace {

void Run(const attn::Options& opt, bool equal_values) {
  attn::TfRandom random(45901);
  Tensor         q = random({2, 4, 4}), k = random({2, 5, 4}), v = random({2, 5, 3});
  if (equal_values) {
    float* p = v.data_ptr<float>();
    for (int i = 0; i < 2 * 5; ++i)
      for (int j = 0; j < 3; ++j) p[i * 3 + j] = j - 1.f;
  }
  std::vector<std::uint8_t> keep(2 * 4 * 5, 0);
  auto                      at = [](int h, int t, int j) { return (h * 4 + t) * 5 + j; };
  for (int h = 0; h < 2; ++h)
    for (int t = 0; t < 3; ++t)
      for (int j = 0; j <= t + 1; ++j) keep[at(h, t, j)] = 1;
  keep[at(1, 3, 0)] = 1;  // last row of head 0 is empty; last row of head 1 has one key
  auto to_mask = [](const std::vector<std::uint8_t>& m) {
    Tensor t = nn::zeros({2, 4, 5}, nn::DType::Bool);
    bool*  p = t.data_ptr<bool>();
    for (std::size_t i = 0; i < m.size(); ++i) p[i] = m[i] != 0;
    return t;
  };
  const Tensor allowed = to_mask(keep);
  const auto   a       = attn::ScaledDotProductAttention(q, k, v, 4, allowed);
  const Tensor entropy  = nn::neg(nn::sum(a.weights * nn::log(nn::maximum(a.weights, nn::scalar(1e-30))), {-1}));
  const Tensor nonempty = nn::greater(nn::sum(a.weights, {-1}), 0.0);
  const Tensor support  = nn::exp(entropy) * nn::cast(nonempty, nn::DType::Float32);
  const Tensor maxw     = nn::max(a.weights, {-1});

  nn::NamedTensors saved{{"q", q},           {"k", k},
                         {"v", v},           {"allowed", allowed},
                         {"weights", a.weights}, {"output", a.output},
                         {"entropy", entropy},   {"effective_support", support},
                         {"max_weight", maxw}};
  std::vector<float> effects(2 * 4 * 5);
  const auto         out = a.output.to_vector<float>();
  for (int r = 0; r < 5; ++r) {
    auto m = keep;
    for (int h = 0; h < 2; ++h)
      for (int t = 0; t < 4; ++t) m[at(h, t, r)] = 0;
    const Tensor without = attn::ScaledDotProductAttention(q, k, v, 4, to_mask(m)).output;
    saved.push_back({"without_" + std::to_string(r), without});
    const auto w = without.to_vector<float>();
    for (int h = 0; h < 2; ++h)
      for (int t = 0; t < 4; ++t) {
        double e = 0;
        for (int j = 0; j < 3; ++j) {
          const double d = out[(h * 4 + t) * 3 + j] - w[(h * 4 + t) * 3 + j];
          e += d * d;
        }
        effects[at(h, t, r)] = static_cast<float>(std::sqrt(e));
      }
  }
  saved.push_back({"removal_effect", nn::from_vector(effects, {2, 4, 5})});
  const std::string name = equal_values ? "equal_values" : "distinct_values";
  nn::save_npz((opt.out / (name + ".npz")).string(), saved);

  // Print the maps: weights per head, then entropy / support / max weight / removal effects.
  std::printf("%s:\n", name.c_str());
  for (int h = 0; h < 2; ++h) {
    std::printf("  head %d   weights over keys 0..4          entropy support maxw | removal effect of keys 0..4\n", h);
    for (int t = 0; t < 4; ++t) {
      std::printf("    q%d:", t);
      for (int j = 0; j < 5; ++j) std::printf(" %6.3f", a.weights.at({h, t, j}));
      std::printf("   %6.3f %6.3f %5.3f |", entropy.at({h, t}), support.at({h, t}), maxw.at({h, t}));
      for (int r = 0; r < 5; ++r) std::printf(" %6.3f", effects[at(h, t, r)]);
      std::printf("\n");
    }
  }
  check::Near(support.at({0, 3}), 0, 0, "empty support");
  check::Near(support.at({1, 3}), 1, 1e-6, "singleton support");
  if (equal_values)
    for (int h = 0; h < 2; ++h)
      for (int t = 0; t < 3; ++t)
        for (int r = 0; r < 5; ++r)
          check::Near(effects[at(h, t, r)], 0, 2e-6, "equal values remove-one invariance with surviving keys");
  std::printf("%s: 2 heads, 4 queries, 5 keys; actual maps/entropy/effective support and five native removal "
              "interventions PASS\n",
              name.c_str());
  attn::CheckAgainstTf(opt.tf, name + ".npz", saved, {"q", "k", "v"}, 1e-5, opt.verbose);
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const auto opt = attn::ParseOptions(argc, argv, "attention_visualization", paths::OutputDir("attention_visualization"));
    Run(opt, false);
    Run(opt, true);
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
