// CMD: just run-nn multi_head_attention
// Multi-head attention with projection and shared-input derivatives.
//
// Port of tflow's multi_head_attention. Four cases of the bias-free layer
// concat_h(softmax(Q_h K_h^T / sqrt(dh)) V_h) Wo with d_model 8: cross
// attention with two heads, the same with a causal + padding validity mask
// (one example has no valid key), self attention with four heads (the shared
// input's gradient sums the query, key and value paths), and one head with
// identity projections, which must equal plain scaled dot-product attention.
// The objective sum(output * probe) is differentiated with respect to all
// four projections and the inputs; 3 coordinates of each projection and 6 of
// each input are checked against float32 central differences (< 2e-3, as in
// TF). The layer has 4 x 8 x 8 = 256 parameters, and invalid model/head
// sizes are rejected.
//
// Weights: init=auto (default) replays the projections TF saved in its
// fixtures (tflow's output/models/multi_head_attention/<case>.npz, arrays
// mha/Wq ...) when they exist, init=tf requires them, init=nn draws them from
// nn's generator with TF's distribution U(+-sqrt(3/d_model)). The inputs come
// from TF's std::mt19937 streams, so with TF's weights every output and
// gradient is compared with TF's (must agree to 1e-5).
//
// TF -> datamunge: nn::MultiHeadAttention is attn::MultiHeadAttention
// (common/attention.hpp; explicit d_model and its guard as in tflow) over
// nn::multi_head_attention, whose parameters have tflow's names and layout.
//
// Options: out=DIR (default output/nn/multi_head_attention), tf=DIR,
// init=auto|tf|nn, verbose=1.

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

void Run(const attn::Options& opt, const std::string& name, int heads, bool self, bool strict, bool identity,
         unsigned seed) {
  const std::int64_t tq = self ? 4 : 3, tk = self ? 4 : 5;
  nn::ParameterStore store(seed);
  attn::TfRandom random(seed);
  Tensor             xq  = random({2, tq, 8});
  Tensor             xkv = self ? xq : random({2, tk, 8});
  const Tensor       probe = random({2, tq, 8}, .2f);
  xq.requires_grad_();
  if (!self) xkv.requires_grad_();

  const char* projections[4] = {"Wq", "Wk", "Wv", "Wo"};
  std::string source         = "nn";
  if (identity) {
    for (auto key : projections) store.get(std::string("mha/") + key, {8, 8}, attn::FixedInit(nn::eye(8)));
    source = "identity";
  } else if (opt.init != "nn") {
    const auto file = opt.tf / (name + ".npz");
    if (std::filesystem::exists(file)) {
      const auto ref = nn::load_npz(file.string());
      for (auto key : projections) {
        const std::string n = std::string("mha/") + key;
        store.get(n, {8, 8}, attn::FixedInit(attn::Named(ref, n)));
      }
      source = "TF";
    } else if (opt.init == "tf") {
      throw std::runtime_error("init=tf: " + file.string() + " not found");
    }
  }

  Tensor       mask;
  const Tensor kl = self ? attn::Ints({4, 3}) : attn::Ints({4, 0});
  const Tensor ql = self ? attn::Ints({4, 3}) : attn::Ints({2, 3});
  if (strict) mask = nn::expand_dims(attn::AttentionAllowed(kl, ql, tq, tk, true), 1);
  auto forward = [&] { return attn::MultiHeadAttention(store, "mha", xq, xkv, 8, heads, mask); };
  const nn::Attended a         = forward();
  const Tensor       objective = nn::sum(a.output * probe);
  std::vector<Tensor> leaves;
  for (auto key : projections) leaves.push_back(store.at(std::string("mha/") + key));
  leaves.push_back(xq);
  if (!self) leaves.push_back(xkv);
  const auto g = nn::grad(objective, leaves);

  nn::NamedTensors saved{{"xq", xq.detach()},        {"xkv", xkv.detach()},          {"probe", probe},
                         {"output", a.output.detach()}, {"weights", a.weights.detach()}, {"objective", objective.detach()}};
  for (int j = 0; j < 4; ++j) saved.push_back({std::string("gradient_mha/") + projections[j], g[j]});
  saved.push_back({"xq_gradient", g[4]});
  if (!self) saved.push_back({"xkv_gradient", g[5]});
  if (strict) {
    saved.push_back({"allowed", mask});
    saved.push_back({"key_lengths", kl});
    saved.push_back({"query_lengths", ql});
  }
  if (identity) {
    const Tensor primitive = attn::ScaledDotProductAttention(xq, xkv, xkv, 8).output.detach();
    saved.push_back({"primitive", primitive});
    const double d = attn::MaxAbsDiff(a.output, primitive);
    check::Near(d, 0, 2e-6, "one identity-projected head equals primitive");
  }
  for (int j = 0; j < 4; ++j) saved.push_back({std::string("mha/") + projections[j], leaves[j].detach().clone()});

  double fd_error = 0;
  int    count    = 0;
  {
    nn::NoGradGuard no_grad;
    auto            f = [&] { return nn::sum(forward().output * probe).item<float>(); };
    for (std::size_t leaf = 0; leaf < leaves.size(); ++leaf)
      fd_error = std::max(fd_error, attn::CentralDifferenceError(leaves[leaf], g[leaf], leaf < 4 ? 3 : 6, f, &count));
  }
  check::Require(fd_error < 2e-3, "MHA native projection/input finite differences");
  saved.push_back({"metadata", nn::from_vector(std::vector<double>{double(heads), double(self), double(count), fd_error}, {4})});
  nn::save_npz((opt.out / (name + ".npz")).string(), saved);
  check::Require(store.count() == 256 && store.size() == 4, "four 8x8 bias-free projections");
  std::printf("%s heads=%d sharedInput=%d weights=%s nativeFD=%d maxError=%.3g PASS\n", name.c_str(), heads, int(self),
              source.c_str(), count, fd_error);
  if (source == "TF" || source == "identity")
    attn::CheckAgainstTf(opt.tf, name + ".npz", saved, {"xq", "xkv", "probe", "mha/Wq", "mha/Wk", "mha/Wv", "mha/Wo"},
                         1e-5, opt.verbose);
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const auto opt = attn::ParseOptions(argc, argv, "multi_head_attention", paths::OutputDir("multi_head_attention"));
    Run(opt, "cross_two", 2, false, false, false, 45601);
    Run(opt, "masked_two", 2, false, true, false, 45603);
    Run(opt, "self_four", 4, true, true, false, 45605);
    Run(opt, "identity_one", 1, false, false, true, 45607);
    nn::ParameterStore store;
    const Tensor       x        = nn::ones({1, 1, 1});
    int                rejected = 0;
    for (auto pair : std::vector<std::pair<int, int>>{{8, 0}, {8, 3}, {0, 1}}) try {
        attn::MultiHeadAttention(store, "bad", x, x, pair.first, pair.second);
      } catch (const std::invalid_argument&) {
        ++rejected;
      }
    // The library layer (d_model taken from the input) rejects the same head counts.
    const Tensor x8 = nn::ones({1, 1, 8});
    for (int heads : {0, 3}) try {
        nn::multi_head_attention(store, "bad", x8, x8, heads);
      } catch (const std::invalid_argument&) {
        ++rejected;
      }
    check::Require(rejected == 5, "invalid model/head dimensions rejected");
    std::cout << "Multi-head all projection/input checks and dimension guards PASS\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
