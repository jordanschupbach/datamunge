// CMD: just run-nn attention_masks
// Padding, causal and cached-prefix masks with exact zero-gradient isolation.
//
// Port of tflow's attention_masks. A boolean validity mask [batch, Tq, Tk]
// combines key padding, query padding and causality with a query offset (the
// queries of incremental decoding sit after `offset` cached keys). Four cases
// over 5 keys and batch 3 (the third example has no valid key): cross
// attention with padded keys and queries; causal self-attention with a
// probe on one query only, so gradient reaches exactly the keys/values that
// query may see; one cached query at offset 4; and a two-query chunk at
// offset 3. For the cached cases the chunk's output must equal the matching
// rows of full causal attention over all 5 positions. Each case checks 18
// Q/K/V coordinates against float32 central differences (< 5e-4) and
// invalid sizes/offsets are rejected.
//
// Inputs come from TF's std::mt19937 streams, so when TF's fixtures
// (tflow's output/models/attention_masks/*.npz) are present every output and
// gradient is compared with TF's (must agree to 1e-5).
//
// TF -> datamunge: nn::AttentionAllowed is attn::AttentionAllowed
// (common/attention.hpp; the library's causal_keep_mask aligns queries
// bottom-right and has no query-padding form), used as a bool keep mask of
// nn::scaled_dot_product_attention (plain path, attn::ScaledDotProductAttention).
//
// Options: out=DIR (default output/nn/attention_masks), tf=DIR, verbose=1.

#include <cstdio>
#include <iostream>
#include <random>
#include <string>

#include "common/attention.hpp"
#include "common/checks.hpp"
#include "common/paths.hpp"

namespace nn = datamunge::nn;
using nn::Tensor;

namespace {

void Run(const attn::Options& opt, const std::string& name, int queries, bool causal, int offset, int probe_index,
         unsigned seed) {
  const int        keys = 5;
  std::vector<int> klv, qlv;
  if (name == "padding_cross") klv = {4, 2, 0}, qlv = {2, 3, 1};
  else if (name == "causal_square") klv = {5, 3, 0}, qlv = {5, 3, 0};
  else if (queries == 1) klv = {5, 3, 0}, qlv = {1, 1, 1};
  else klv = {5, 4, 0}, qlv = {2, 1, 2};
  const Tensor kl = attn::Ints(klv), ql = attn::Ints(qlv);
  attn::TfRandom random(seed);
  Tensor q = random({3, queries, 4}), k = random({3, keys, 4}),
         v = random({3, keys, 3}), probe = random({3, queries, 3}, .2f);
  if (probe_index >= 0) {
    float* p = probe.data_ptr<float>();
    for (int b = 0; b < 3; ++b)
      for (int t = 0; t < queries; ++t)
        if (t != probe_index)
          for (int j = 0; j < 3; ++j) p[(b * queries + t) * 3 + j] = 0;
  }
  q.requires_grad_(), k.requires_grad_(), v.requires_grad_();
  const Tensor allowed   = attn::AttentionAllowed(kl, ql, queries, keys, causal, offset);
  auto         forward   = [&] { return attn::ScaledDotProductAttention(q, k, v, 4, allowed); };
  const auto   a         = forward();
  const Tensor objective = nn::sum(a.output * probe);
  const auto   g         = nn::grad(objective, {q, k, v});

  nn::NamedTensors saved{{"q", q.detach()},          {"k", k.detach()},
                         {"v", v.detach()},          {"probe", probe},
                         {"key_lengths", kl},        {"query_lengths", ql},
                         {"output", a.output.detach()}, {"weights", a.weights.detach()},
                         {"allowed", allowed},       {"objective", objective.detach()},
                         {"q_gradient", g[0]},       {"k_gradient", g[1]},
                         {"v_gradient", g[2]}};

  // Isolation: keys/values no probed query may see get exactly zero gradient.
  {
    const auto  al = allowed.to_vector<bool>();
    const auto  pr = probe.to_vector<float>();
    const auto  gk = g[1].to_vector<float>(), gv = g[2].to_vector<float>();
    for (int b = 0; b < 3; ++b)
      for (int j = 0; j < keys; ++j) {
        bool seen = false;
        for (int t = 0; t < queries; ++t)
          seen = seen || (al[(b * queries + t) * keys + j] && (pr[(b * queries + t) * 3] != 0));
        if (!seen) {
          for (int c = 0; c < 4; ++c) check::Require(gk[(b * keys + j) * 4 + c] == 0, "unseen key has zero gradient");
          for (int c = 0; c < 3; ++c) check::Require(gv[(b * keys + j) * 3 + c] == 0, "unseen value has zero gradient");
        }
      }
  }

  if (offset > 0) {
    // The query chunk placed at rows offset.. of a full causal pass over all keys.
    Tensor full_q  = nn::zeros({3, keys, 4});
    float* fq      = full_q.data_ptr<float>();
    const auto qv  = q.detach().to_vector<float>();
    std::vector<int> fqlv(3);
    for (int b = 0; b < 3; ++b) {
      fqlv[b] = offset + qlv[b];
      for (int t = 0; t < queries; ++t)
        for (int j = 0; j < 4; ++j) fq[(b * keys + offset + t) * 4 + j] = qv[(b * queries + t) * 4 + j];
    }
    const Tensor full_ql = attn::Ints(fqlv);
    const Tensor full =
        attn::ScaledDotProductAttention(full_q, k, v, 4, attn::AttentionAllowed(kl, full_ql, keys, keys, true)).output;
    saved.push_back({"full_q", full_q});
    saved.push_back({"full_query_lengths", full_ql});
    saved.push_back({"full_output", full.detach()});
    for (int b = 0; b < 3; ++b)
      for (int t = 0; t < queries; ++t)
        for (int j = 0; j < 3; ++j)
          check::Near(a.output.at({b, t, j}), full.at({b, offset + t, j}), 2e-6,
                      "cached query chunk equals full causal slice");
  }

  double error = 0;
  int    count = 0;
  {
    nn::NoGradGuard no_grad;
    auto            f         = [&] { return nn::sum(forward().output * probe).item<float>(); };
    Tensor*         leaves[3] = {&q, &k, &v};
    for (int leaf = 0; leaf < 3; ++leaf)
      error = std::max(error, attn::CentralDifferenceError(*leaves[leaf], g[leaf], 6, f, &count));
  }
  check::Require(error < 5e-4, "masked native finite differences");
  saved.push_back({"metadata", nn::from_vector(std::vector<double>{double(causal), double(offset), double(probe_index),
                                                                  double(count), error},
                                               {5})});
  nn::save_npz((opt.out / (name + ".npz")).string(), saved);
  std::printf("%s queries=%d keys=5 offset=%d nativeFD=%d maxError=%.3g PASS\n", name.c_str(), queries, offset, count,
              error);
  attn::CheckAgainstTf(opt.tf, name + ".npz", saved, {"q", "k", "v", "probe"}, 1e-5, opt.verbose);
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const auto opt = attn::ParseOptions(argc, argv, "attention_masks", paths::OutputDir("attention_masks"));
    Run(opt, "padding_cross", 3, false, 0, -1, 45701);
    Run(opt, "causal_square", 5, true, 0, 2, 45703);
    Run(opt, "cached_one", 1, true, 4, -1, 45705);
    Run(opt, "cached_chunk", 2, true, 3, 0, 45707);
    const Tensor lengths  = attn::Ints({1});
    int          rejected = 0;
    for (auto values : std::vector<std::vector<int>>{{0, 5, 0}, {3, 0, 0}, {3, 5, -1}}) try {
        attn::AttentionAllowed(lengths, lengths, values[0], values[1], true, values[2]);
      } catch (const std::invalid_argument&) {
        ++rejected;
      }
    check::Require(rejected == 3, "invalid physical sizes and offsets rejected");
    std::cout << "Padding/causal/query masks and full/cache equivalence native PASS\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
