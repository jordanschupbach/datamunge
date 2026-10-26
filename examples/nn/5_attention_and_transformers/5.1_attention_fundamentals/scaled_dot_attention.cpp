// CMD: just run-nn scaled_dot_attention
// Scaled dot-product attention with complete input derivatives.
//
// Port of tflow's scaled_dot_attention. Four cases of
// A = softmax(Q K^T / sqrt(d) + bias) V: cross attention [2, 3, 4] x [2, 5, 4];
// two heads with a causal + padding validity mask (one example has no valid
// key, so its rows are all zero); a single key (weights exactly 1); and an
// additive score bias. For each, the objective sum(A * probe) is
// differentiated with respect to Q, K and V by reverse-mode autograd and 18
// coordinates are checked against float32 central differences (|error| < 5e-4,
// as in TF). A zero key width is rejected.
//
// Inputs come from the same std::mt19937 streams as TF, so when TF's fixtures
// (tflow's output/models/scaled_dot_attention/*.npz) are present every output,
// weight and gradient is also compared with TF's (must agree to 1e-5).
//
// TF -> datamunge: nn::ScaledDotProductAttention(s, q, k, v, d, bias, allowed)
// is attn::ScaledDotProductAttention (common/attention.hpp), a thin wrapper of
// nn::scaled_dot_product_attention on its plain path with weights returned;
// tflow's boolean `allowed` is a bool keep mask (common AttentionAllowed).
// Symbolic gradients become nn::grad, session re-runs become eager calls.
//
// Options: out=DIR (default output/nn/scaled_dot_attention), tf=DIR (TF
// fixtures), verbose=1 (per-array TF differences). Writes <case>.npz with
// TF's array names.

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

void Run(const attn::Options& opt, const std::string& name, bool heads, bool strict, bool one_key, unsigned seed) {
  const std::int64_t keys = one_key ? 1 : 5;
  const nn::Shape    qs = heads ? nn::Shape{2, 2, 3, 4} : nn::Shape{2, 3, 4};
  const nn::Shape    ks = heads ? nn::Shape{2, 2, keys, 4} : nn::Shape{2, keys, 4};
  const nn::Shape    vs = heads ? nn::Shape{2, 2, keys, 3} : nn::Shape{2, keys, 3};
  const nn::Shape    os = heads ? nn::Shape{2, 2, 3, 3} : nn::Shape{2, 3, 3};
  attn::TfRandom random(seed);
  Tensor             q = random(qs), k = random(ks), v = random(vs),
         probe = random(os, .2f);
  q.requires_grad_(), k.requires_grad_(), v.requires_grad_();

  Tensor mask, bias, allowed;
  const Tensor kl = attn::Ints({4, 0}), ql = attn::Ints({2, 3});
  if (name == "biased") {
    std::vector<float> b(static_cast<std::size_t>(3 * keys));
    for (std::size_t j = 0; j < b.size(); ++j) b[j] = .3f * std::sin(j * .7f);
    bias = mask = nn::from_vector(b, {3, keys});
  }
  if (strict) {
    allowed = attn::AttentionAllowed(kl, ql, 3, keys, true);
    if (heads) allowed = nn::expand_dims(allowed, 1);
    mask = allowed;
  }
  auto forward = [&] { return attn::ScaledDotProductAttention(q, k, v, 4, mask); };
  const nn::Attended a         = forward();
  const Tensor       objective = nn::sum(a.output * probe);
  const auto         g         = nn::grad(objective, {q, k, v});

  nn::NamedTensors saved{{"q", q.detach()},           {"k", k.detach()},         {"v", v.detach()},
                         {"probe", probe},            {"output", a.output.detach()}, {"weights", a.weights.detach()},
                         {"objective", objective.detach()}, {"q_gradient", g[0]}, {"k_gradient", g[1]},
                         {"v_gradient", g[2]}};
  if (strict) {
    saved.push_back({"allowed", allowed});
    saved.push_back({"key_lengths", kl});
    saved.push_back({"query_lengths", ql});
  }
  if (bias.defined()) saved.push_back({"bias", bias});

  // Native central differences on 6 coordinates of each of Q, K, V.
  double fd_error = 0;
  int    fd_count = 0;
  {
    nn::NoGradGuard no_grad;
    auto            f = [&] { return nn::sum(forward().output * probe).item<float>(); };
    Tensor*         leaves[3] = {&q, &k, &v};
    for (int field = 0; field < 3; ++field)
      fd_error = std::max(fd_error, attn::CentralDifferenceError(*leaves[field], g[field], 6, f, &fd_count));
  }
  check::Require(fd_error < 5e-4, "native Q/K/V central differences");
  if (strict) {
    // Example 1 has no valid key: its weights, outputs and K/V gradients are exactly zero.
    for (std::int64_t i = 0; i < a.weights.numel() / 2; ++i)
      check::Require(a.weights.to_vector<float>()[a.weights.numel() / 2 + i] == 0, "empty example has zero weights");
  }
  if (one_key)
    for (float w : a.weights.to_vector<float>()) check::Near(w, 1, 1e-7, "a single key gets all the weight");
  saved.push_back({"native_fd", nn::from_vector(std::vector<double>{double(fd_count), fd_error}, {2})});
  nn::save_npz((opt.out / (name + ".npz")).string(), saved);
  std::printf("%s Q%s K%s V%s nativeFD=%d maxError=%.3g PASS\n", name.c_str(), nn::shape_str(qs).c_str(),
              nn::shape_str(ks).c_str(), nn::shape_str(vs).c_str(), fd_count, fd_error);
  attn::CheckAgainstTf(opt.tf, name + ".npz", saved, {"q", "k", "v", "probe"}, 1e-5, opt.verbose);
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const auto opt = attn::ParseOptions(argc, argv, "scaled_dot_attention", paths::OutputDir("scaled_dot_attention"));
    Run(opt, "cross", false, false, false, 45501);
    Run(opt, "masked_heads", true, true, false, 45503);
    Run(opt, "one_key", false, false, true, 45505);
    Run(opt, "biased", false, false, false, 45507);
    const Tensor x        = nn::ones({1, 1, 1});
    bool         rejected = false;
    try {
      attn::ScaledDotProductAttention(x, x, x, 0);
    } catch (const std::invalid_argument&) {
      rejected = true;
    }
    check::Require(rejected, "reject zero key width");
    std::cout << "Scaled dot-product: all native derivative checks and dimension guard PASS\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
