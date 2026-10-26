// Tests for datamunge::nn attention: scaled dot-product attention (plain and
// fused), masks, multi-head attention, the KV cache, additive attention and
// positional encodings.

#include <gtest/gtest.h>

#include <cmath>
#include <datamunge/nn/attention.hpp>
#include <datamunge/nn/autograd.hpp>
#include <datamunge/nn/gradcheck.hpp>
#include <datamunge/nn/ops.hpp>
#include <datamunge/nn/params.hpp>
#include <datamunge/nn/random.hpp>
#include <vector>

namespace nn = datamunge::nn;
using nn::DType;
using nn::Tensor;

namespace {

using Inputs = std::vector<Tensor>;

Tensor randn(const nn::Shape& shape, std::uint64_t seed, DType dtype = DType::Float64) {
  nn::Generator gen(seed);
  return nn::normal(shape, 0.0, 1.0, gen, dtype);
}

double max_abs_diff(const Tensor& a, const Tensor& b) {
  return nn::max(nn::abs(nn::cast(a.detach(), DType::Float64) - nn::cast(b.detach(), DType::Float64))).item();
}

/// Naive attention over [B, Tq, d] x [B, Tk, d] x [B, Tk, dv] with an optional
/// bias [B, Tq, Tk] and keep mask [B, Tq, Tk] (all as flat double vectors).
struct Ref {
  std::vector<double> out, weights;
};
Ref naive(const std::vector<double>& q, const std::vector<double>& k, const std::vector<double>& v, int B, int Tq,
          int Tk, int d, int dv, const std::vector<double>* bias, const std::vector<bool>* keep, bool causal) {
  Ref r;
  r.out.assign(static_cast<std::size_t>(B * Tq * dv), 0.0);
  r.weights.assign(static_cast<std::size_t>(B * Tq * Tk), 0.0);
  for (int b = 0; b < B; ++b)
    for (int i = 0; i < Tq; ++i) {
      std::vector<double> s(static_cast<std::size_t>(Tk));
      std::vector<bool>   ok(static_cast<std::size_t>(Tk), true);
      double              mx = -1e300;
      for (int j = 0; j < Tk; ++j) {
        double acc = 0;
        for (int a = 0; a < d; ++a) acc += q[(b * Tq + i) * d + a] * k[(b * Tk + j) * d + a];
        acc /= std::sqrt(static_cast<double>(d));
        const int idx = (b * Tq + i) * Tk + j;
        if (bias) acc += (*bias)[idx];
        if (keep && !(*keep)[idx]) ok[j] = false;
        if (causal && j > i + (Tk - Tq)) ok[j] = false;
        s[j] = acc;
        if (ok[j]) mx = std::max(mx, acc);
      }
      double z = 0;
      for (int j = 0; j < Tk; ++j) z += ok[j] ? std::exp(s[j] - mx) : 0.0;
      for (int j = 0; j < Tk; ++j) {
        const double w = ok[j] && z > 0 ? std::exp(s[j] - mx) / z : 0.0;
        r.weights[(b * Tq + i) * Tk + j] = w;
        for (int a = 0; a < dv; ++a) r.out[(b * Tq + i) * dv + a] += w * v[(b * Tk + j) * dv + a];
      }
    }
  return r;
}

void expect_vec_near(const Tensor& t, const std::vector<double>& ref, double tol) {
  const auto v = t.to_vector<double>();
  ASSERT_EQ(v.size(), ref.size());
  double worst = 0;
  for (std::size_t i = 0; i < v.size(); ++i) worst = std::max(worst, std::abs(v[i] - ref[i]));
  EXPECT_LT(worst, tol);
}

nn::AttentionOptions opts(nn::AttentionImpl impl, bool causal = false, const Tensor& mask = Tensor()) {
  nn::AttentionOptions o;
  o.impl   = impl;
  o.causal = causal;
  o.mask   = mask;
  return o;
}

}  // namespace

// ---- values ------------------------------------------------------------------------------------

TEST(NnAttention, PlainMatchesNaiveReference) {
  const int    B = 3, Tq = 5, Tk = 7, d = 4, dv = 3;
  const Tensor q = randn({B, Tq, d}, 1), k = randn({B, Tk, d}, 2), v = randn({B, Tk, dv}, 3);
  const Tensor bias = randn({B, Tq, Tk}, 4);
  auto         o    = opts(nn::AttentionImpl::Plain, false, bias);
  o.need_weights    = true;
  const auto a      = nn::scaled_dot_product_attention(q, k, v, o);
  const auto bv     = bias.to_vector<double>();
  const Ref  r = naive(q.to_vector<double>(), k.to_vector<double>(), v.to_vector<double>(), B, Tq, Tk, d, dv, &bv,
                       nullptr, false);
  EXPECT_EQ(a.output.shape(), (nn::Shape{B, Tq, dv}));
  EXPECT_EQ(a.weights.shape(), (nn::Shape{B, Tq, Tk}));
  expect_vec_near(a.output, r.out, 1e-12);
  expect_vec_near(a.weights, r.weights, 1e-12);
}

TEST(NnAttention, TflowFormReturnsWeights) {
  const Tensor q = randn({1, 4, 8}, 5, DType::Float32), k = randn({1, 4, 8}, 6, DType::Float32),
               v = randn({1, 4, 3}, 7, DType::Float32);
  const auto   a = nn::scaled_dot_product_attention(q, k, v, nn::causal_mask(4));
  ASSERT_TRUE(a.weights.defined());
  // Row sums are one; the strict upper triangle is zero.
  expect_vec_near(nn::sum(a.weights, {-1}), std::vector<double>(4, 1.0), 1e-6);
  for (int i = 0; i < 4; ++i)
    for (int j = i + 1; j < 4; ++j) EXPECT_EQ(a.weights.at({0, i, j}), 0.0);
}

TEST(NnAttention, MaskSemantics) {
  const int    B = 2, H = 2, T = 6, d = 4;
  const Tensor q = randn({B, H, T, d}, 8), k = randn({B, H, T, d}, 9), v = randn({B, H, T, d}, 10);
  auto         run = [&](const nn::AttentionOptions& o0) {
    auto o         = o0;
    o.need_weights = true;
    return nn::scaled_dot_product_attention(q, k, v, o);
  };
  // causal flag == bool causal mask == additive causal mask (to rounding from exp(-1e9)).
  const auto flag = run(opts(nn::AttentionImpl::Plain, true));
  const auto keep = run(opts(nn::AttentionImpl::Plain, false, nn::causal_keep_mask(T)));
  const auto add  = run(opts(nn::AttentionImpl::Plain, false, nn::causal_mask(T, DType::Float64)));
  EXPECT_LT(max_abs_diff(flag.output, keep.output), 1e-14);
  EXPECT_LT(max_abs_diff(flag.output, add.output), 1e-14);
  // An integer mask reads nonzero as keep.
  const auto ints = run(opts(nn::AttentionImpl::Plain, false, nn::cast(nn::causal_keep_mask(T), DType::Int32)));
  EXPECT_LT(max_abs_diff(flag.output, ints.output), 1e-14);

  // Padding: example 0 keeps 4 keys, example 1 keeps all 6.
  const Tensor pad = nn::padding_keep_mask(nn::tensor({4, 6}, {}, DType::Int64), T);
  EXPECT_EQ(pad.shape(), (nn::Shape{B, 1, 1, T}));
  const auto padded = run(opts(nn::AttentionImpl::Plain, false, pad));
  for (int h = 0; h < H; ++h)
    for (int i = 0; i < T; ++i)
      for (int j = 4; j < T; ++j) EXPECT_EQ(padded.weights.at({0, h, i, j}), 0.0);
  // Only the first four values matter for example 0.
  const Tensor v2 = nn::concat({nn::slice_axis(v, 2, 0, 4), randn({B, H, 2, d}, 11)}, 2);
  auto         o  = opts(nn::AttentionImpl::Plain, false, pad);
  const auto   a2 = nn::scaled_dot_product_attention(q, k, v2, o);
  EXPECT_LT(max_abs_diff(nn::select(a2.output, 0, 0), nn::select(padded.output, 0, 0)), 1e-14);

  // From ids, pad id 0.
  const Tensor ids = nn::tensor({5, 3, 0, 0, 7, 1, 2, 0}, {2, 4}, DType::Int64);
  const auto   im  = nn::padding_keep_mask_from_ids(ids);
  EXPECT_EQ(im.to_vector<int>(), (std::vector<int>{1, 1, 0, 0, 1, 1, 1, 0}));

  // A query row with nothing to attend to gives zeros, on both paths.
  Tensor none = nn::ones({T, T}, DType::Bool);
  {
    nn::NoGradGuard g;
    nn::slice_axis(none, 0, 2, 3).fill_(0.0);
  }
  const auto plain = run(opts(nn::AttentionImpl::Plain, false, none));
  const auto fused = nn::scaled_dot_product_attention(q, k, v, opts(nn::AttentionImpl::Fused, false, none));
  for (int c = 0; c < d; ++c) {
    EXPECT_EQ(plain.output.at({1, 1, 2, c}), 0.0);
    EXPECT_EQ(fused.output.at({1, 1, 2, c}), 0.0);
  }
  EXPECT_LT(max_abs_diff(plain.output, fused.output), 1e-13);
}

TEST(NnAttention, CausalAlignsBottomRight) {
  const Tensor m = nn::causal_keep_mask(2, 4);
  EXPECT_EQ(m.to_vector<int>(), (std::vector<int>{1, 1, 1, 0, 1, 1, 1, 1}));
  const Tensor a = nn::causal_mask(3, DType::Float32);
  EXPECT_EQ(a.to_vector<double>(), (std::vector<double>{0, -1e9f, -1e9f, 0, 0, -1e9f, 0, 0, 0}));
}

// ---- fused vs plain -----------------------------------------------------------------------------

namespace {

struct FusedCase {
  int  B, H, Tq, Tk, d, dv;
  bool causal;
  int  mask_kind;  // 0 none, 1 bias [H, Tq, Tk], 2 keep [B, 1, 1, Tk], 3 both
};

void compare_paths(const FusedCase& c, DType dt, double tol) {
  const Tensor q = randn({c.B, c.H, c.Tq, c.d}, 20, dt), k = randn({c.B, c.H, c.Tk, c.d}, 21, dt),
               v = randn({c.B, c.H, c.Tk, c.dv}, 22, dt);
  Tensor mask;
  if (c.mask_kind == 1 || c.mask_kind == 3) mask = randn({c.H, c.Tq, c.Tk}, 23, dt);
  Tensor keep;
  if (c.mask_kind >= 2) {
    std::vector<std::int64_t> lens;
    for (int b = 0; b < c.B; ++b) lens.push_back(std::max(1, c.Tk - 5 * b - 3));
    keep = nn::padding_keep_mask(nn::from_vector(lens), c.Tk);
  }
  if (c.mask_kind == 3) mask = nn::where(keep, mask, -1e9);  // additive form of both
  else if (c.mask_kind == 2) mask = keep;

  Inputs leaves = {q.clone().requires_grad_(), k.clone().requires_grad_(), v.clone().requires_grad_()};
  auto   run    = [&](nn::AttentionImpl impl) {
    std::vector<Tensor> ins;
    for (auto& t : leaves) ins.push_back(t);
    const auto   a  = nn::scaled_dot_product_attention(ins[0], ins[1], ins[2], opts(impl, c.causal, mask));
    const Tensor w  = randn(a.output.shape(), 24, dt);
    const auto   gs = nn::grad(nn::sum(a.output * w), ins);
    return std::make_pair(a.output, gs);
  };
  const auto plain = run(nn::AttentionImpl::Plain);
  const auto fused = run(nn::AttentionImpl::Fused);
  EXPECT_EQ(fused.first.grad_fn_name(), "FusedAttention");
  EXPECT_LT(max_abs_diff(plain.first, fused.first), tol) << "output";
  for (int i = 0; i < 3; ++i) EXPECT_LT(max_abs_diff(plain.second[i], fused.second[i]), tol) << "grad " << i;
}

}  // namespace

TEST(NnAttention, FusedMatchesPlainFloat32) {
  const std::vector<FusedCase> cases = {
      {2, 3, 67, 131, 5, 3, false, 0},   // odd lengths spanning several tiles
      {1, 2, 150, 150, 8, 8, true, 0},   // causal square
      {2, 2, 37, 150, 6, 4, true, 1},    // causal, Tq < Tk (decode alignment) + bias
      {3, 1, 129, 257, 4, 7, false, 2},  // padding keep mask
      {2, 2, 70, 70, 16, 16, true, 3},   // additive -1e9 combined with causal
      {1, 1, 1, 300, 8, 8, true, 0},     // one query
  };
  for (const auto& c : cases) compare_paths(c, DType::Float32, 2e-5);
}

TEST(NnAttention, FusedMatchesPlainFloat64) {
  compare_paths({2, 2, 67, 131, 5, 3, true, 3}, DType::Float64, 1e-12);
}

TEST(NnAttention, FusedFewHeadsUsesTwoPassBackward) {
  // A single head forces the key-block / query-block parallel backward.
  compare_paths({1, 1, 200, 300, 8, 5, true, 1}, DType::Float64, 1e-12);
}

TEST(NnAttention, AutoPicksFusedForLongSequences) {
  nn::AttentionOptions o;
  EXPECT_TRUE(nn::attention_would_fuse({1, 1, 512, 64}, {1, 1, 512, 64}, o));
  EXPECT_FALSE(nn::attention_would_fuse({1, 1, 16, 64}, {1, 1, 16, 64}, o));
  o.need_weights = true;
  EXPECT_FALSE(nn::attention_would_fuse({1, 1, 512, 64}, {1, 1, 512, 64}, o));
  const Tensor q = randn({1, 1, 512, 8}, 30, DType::Float32);
  EXPECT_EQ(nn::scaled_dot_product_attention(q.clone().requires_grad_(), q, q).output.grad_fn_name(),
            "FusedAttention");
}

TEST(NnAttention, BroadcastLeadingAxes) {
  // Shared keys/values across heads via a size-1 head axis, on both paths.
  const Tensor q = randn({2, 3, 9, 4}, 31), k = randn({2, 1, 11, 4}, 32), v = randn({2, 1, 11, 5}, 33);
  const auto   p = nn::scaled_dot_product_attention(q, k, v, opts(nn::AttentionImpl::Plain));
  const auto   f = nn::scaled_dot_product_attention(q, k, v, opts(nn::AttentionImpl::Fused));
  EXPECT_EQ(f.output.shape(), (nn::Shape{2, 3, 9, 5}));
  EXPECT_LT(max_abs_diff(p.output, f.output), 1e-13);
}

TEST(NnAttention, RejectsBadArguments) {
  const Tensor q = randn({2, 3, 4}, 1), k = randn({2, 5, 4}, 2), v = randn({2, 5, 2}, 3);
  for (auto implementation : {nn::AttentionImpl::Auto, nn::AttentionImpl::Plain, nn::AttentionImpl::Fused}) {
    nn::AttentionOptions zero_width;
    zero_width.impl = implementation;
    EXPECT_THROW(nn::scaled_dot_product_attention(nn::zeros({1, 1, 0}), nn::zeros({1, 1, 0}),
                                                  nn::ones({1, 1, 1}), zero_width), std::invalid_argument);
  }
  EXPECT_THROW(nn::scaled_dot_product_attention(q, randn({2, 5, 3}, 4), v), std::invalid_argument);
  EXPECT_THROW(nn::scaled_dot_product_attention(q, k, randn({2, 4, 2}, 4)), std::invalid_argument);
  EXPECT_THROW(nn::scaled_dot_product_attention(q, k, v, opts(nn::AttentionImpl::Plain, false, randn({4, 5}, 5))),
               std::invalid_argument);
  auto o         = opts(nn::AttentionImpl::Fused);
  o.need_weights = true;
  EXPECT_THROW(nn::scaled_dot_product_attention(q, k, v, o), std::invalid_argument);
}

// ---- gradients -----------------------------------------------------------------------------------

TEST(NnAttention, GradcheckPlain) {
  const Tensor q = randn({2, 3, 4}, 40).requires_grad_(), k = randn({2, 5, 4}, 41).requires_grad_(),
               v = randn({2, 5, 3}, 42).requires_grad_(), b = randn({3, 5}, 43).requires_grad_();
  const auto f = [](const Inputs& in) {
    return nn::scaled_dot_product_attention(in[0], in[1], in[2], opts(nn::AttentionImpl::Plain, true, in[3])).output;
  };
  const auto r = nn::gradcheck(f, {q, k, v, b});
  EXPECT_TRUE(r.ok) << r.message;
  const auto r2 = nn::gradgradcheck(f, {q, k, v, b});
  EXPECT_TRUE(r2.ok) << r2.message;
}

TEST(NnAttention, GradcheckPlainKeepMask) {
  const Tensor keep = nn::padding_keep_mask(nn::tensor({3, 5}, {}, DType::Int64), 5);
  const Tensor q = randn({2, 1, 3, 4}, 44).requires_grad_(), k = randn({2, 1, 5, 4}, 45).requires_grad_(),
               v = randn({2, 1, 5, 2}, 46).requires_grad_();
  const auto f = [&](const Inputs& in) {
    return nn::scaled_dot_product_attention(in[0], in[1], in[2], opts(nn::AttentionImpl::Plain, false, keep)).output;
  };
  EXPECT_TRUE(nn::gradcheck(f, {q, k, v}).ok);
  EXPECT_TRUE(nn::gradgradcheck(f, {q, k, v}).ok);
}

TEST(NnAttention, GradcheckFusedAcrossTiles) {
  // 66 queries and 130 keys span two tiles each way; causal with Tq < Tk and a keep mask.
  const Tensor keep = nn::padding_keep_mask(nn::tensor({120}, {}, DType::Int64), 130);
  const Tensor q = randn({1, 1, 66, 2}, 47).requires_grad_(), k = randn({1, 1, 130, 2}, 48).requires_grad_(),
               v = randn({1, 1, 130, 2}, 49).requires_grad_();
  const auto f = [&](const Inputs& in) {
    return nn::sum(nn::square(
        nn::scaled_dot_product_attention(in[0], in[1], in[2], opts(nn::AttentionImpl::Fused, true, keep)).output));
  };
  const auto r = nn::gradcheck(f, {q, k, v});
  EXPECT_TRUE(r.ok) << r.message;
}

TEST(NnAttention, GradcheckFusedBiasAndSecondOrder) {
  // A bias that needs a gradient and create_graph both take the recomputing backward.
  const Tensor q = randn({2, 3, 4}, 50).requires_grad_(), k = randn({2, 5, 4}, 51).requires_grad_(),
               v = randn({2, 5, 3}, 52).requires_grad_(), b = randn({3, 5}, 53).requires_grad_();
  const auto f = [](const Inputs& in) {
    return nn::scaled_dot_product_attention(in[0], in[1], in[2], opts(nn::AttentionImpl::Fused, true, in[3])).output;
  };
  const auto r = nn::gradcheck(f, {q, k, v, b});
  EXPECT_TRUE(r.ok) << r.message;
  const auto r2 = nn::gradgradcheck(f, {q, k, v, b});
  EXPECT_TRUE(r2.ok) << r2.message;
}

TEST(NnAttention, DropoutOnWeights) {
  const Tensor  q = randn({1, 64, 8}, 54, DType::Float32), v = randn({1, 64, 4}, 55, DType::Float32);
  nn::Generator g1(7), g2(7);
  auto          o = opts(nn::AttentionImpl::Auto);
  o.dropout       = 0.5;
  o.generator     = &g1;
  const auto a    = nn::scaled_dot_product_attention(q, q, v, o);
  o.generator     = &g2;
  const auto b    = nn::scaled_dot_product_attention(q, q, v, o);
  EXPECT_EQ(max_abs_diff(a.output, b.output), 0.0);  // seeded
  const auto c = nn::scaled_dot_product_attention(q, q, v, opts(nn::AttentionImpl::Plain));
  EXPECT_GT(max_abs_diff(a.output, c.output), 1e-3);
}

// ---- multi-head attention --------------------------------------------------------------------------

TEST(NnAttention, MultiHeadMatchesPerHeadComputation) {
  nn::ParameterStore store(4, DType::Float64);
  const Tensor       x = randn({2, 5, 16}, 60);
  nn::MultiHeadOptions o;
  o.need_weights = true;
  const auto a   = nn::multi_head_attention(store, "self", x, x, 4, o);
  EXPECT_EQ(a.output.shape(), (nn::Shape{2, 5, 16}));
  EXPECT_EQ(a.weights.shape(), (nn::Shape{2, 4, 5, 5}));
  EXPECT_EQ(store.count(), 4 * 16 * 16);
  const Tensor Wq = store.at("self/Wq"), Wk = store.at("self/Wk"), Wv = store.at("self/Wv"), Wo = store.at("self/Wo");
  std::vector<Tensor> heads;
  for (int h = 0; h < 4; ++h) {
    auto cols = [&](const Tensor& W) { return nn::slice_axis(W, 1, 4 * h, 4 * h + 4); };
    const Tensor q = nn::matmul(x, cols(Wq)), k = nn::matmul(x, cols(Wk)), v = nn::matmul(x, cols(Wv));
    heads.push_back(nn::matmul(nn::softmax(nn::matmul(q, k, false, true) / 2.0, -1), v));
  }
  const Tensor ref = nn::matmul(nn::concat(heads, 2), Wo);
  EXPECT_LT(max_abs_diff(a.output, ref), 1e-12);
  // Weights are parameter-initialised U(+-sqrt(3/16)).
  EXPECT_LE(nn::max(nn::abs(Wq)).item(), std::sqrt(3.0 / 16));
}

TEST(NnAttention, MultiHeadGradients) {
  nn::ParameterStore store(5, DType::Float64);
  nn::MultiHeadOptions o;
  o.bias     = true;
  o.causal   = true;
  o.kv_heads = 1;  // multi-query
  o.rotary   = true;
  const Tensor x  = randn({2, 4, 8}, 61).requires_grad_();
  const Tensor xm = randn({2, 6, 6}, 62).requires_grad_();
  // Gradient w.r.t. the inputs (through every projection), self- and cross-attention.
  const auto f = [&](const Inputs& in) { return nn::multi_head_attention(store, "mha", in[0], in[0], 2, o).output; };
  EXPECT_TRUE(nn::gradcheck(f, {x}).ok);
  nn::ParameterStore   cross(8, DType::Float64);
  nn::MultiHeadOptions oc;
  const auto fc = [&](const Inputs& in) { return nn::multi_head_attention(cross, "cross", in[0], in[1], 2, oc).output; };
  EXPECT_TRUE(nn::gradcheck(fc, {x, xm}).ok);
  // Parameter gradients against central differences.
  const Tensor w    = randn({2, 4, 8}, 63);
  auto         loss = [&]() { return nn::sum(nn::multi_head_attention(store, "mha", x, x, 2, o).output * w); };
  const auto   ps   = store.parameters();
  const auto   gs   = nn::grad(loss(), ps);
  for (std::size_t i = 0; i < ps.size(); ++i) {
    Tensor       p   = ps[i];
    const auto   g   = gs[i].to_vector<double>();
    const double eps = 1e-6;
    for (std::int64_t e : {std::int64_t{0}, p.numel() / 2, p.numel() - 1}) {
      auto bump = [&](double delta) {
        nn::NoGradGuard guard;
        Tensor          flat = nn::reshape(p, {-1});
        nn::slice_axis(flat, 0, e, e + 1).add_(nn::full({1}, delta, DType::Float64));
      };
      bump(eps);
      const double up = loss().item();
      bump(-2 * eps);
      const double down = loss().item();
      bump(eps);
      EXPECT_NEAR((up - down) / (2 * eps), g[static_cast<std::size_t>(e)], 1e-6) << store.names()[i] << " " << e;
    }
  }
  EXPECT_EQ(store.at("mha/Wk").shape(), (nn::Shape{8, 4}));  // one kv head of size 4
}

TEST(NnAttention, MultiHeadSecondOrder) {
  nn::ParameterStore store(6, DType::Float64);
  const Tensor       x = randn({1, 3, 4}, 64).requires_grad_();
  const auto f = [&](const Inputs& in) { return nn::multi_head_attention(store, "m", in[0], in[0], 2).output; };
  EXPECT_TRUE(nn::gradgradcheck(f, {x}).ok);
}

TEST(NnAttention, KvCacheDecodingEqualsFullCausal) {
  nn::ParameterStore store(7, DType::Float64);
  const int          T = 9;
  const Tensor       x = randn({2, T, 12}, 65);
  for (const bool rotary : {false, true})
    for (const std::int64_t kvh : {std::int64_t{3}, std::int64_t{1}}) {
      nn::MultiHeadOptions o;
      o.causal      = true;
      o.rotary      = rotary;
      o.kv_heads    = kvh;
      const std::string name = "dec" + std::to_string(rotary) + std::to_string(kvh);
      const Tensor full = nn::multi_head_attention(store, name, x, x, 3, o).output;
      for (const bool grad_mode : {false, true}) {
        nn::KvCache cache;
        o.cache = &cache;
        std::vector<Tensor> steps;
        {
          nn::NoGradGuard guard;
          if (grad_mode) nn::set_grad_enabled(true);
          // One prefill chunk of 4 positions, then one position at a time.
          steps.push_back(nn::multi_head_attention(store, name, nn::slice_axis(x, 1, 0, 4),
                                                   nn::slice_axis(x, 1, 0, 4), 3, o).output);
          for (int t = 4; t < T; ++t) {
            const Tensor xt = nn::slice_axis(x, 1, t, t + 1);
            steps.push_back(nn::multi_head_attention(store, name, xt, xt, 3, o).output);
          }
        }
        EXPECT_EQ(cache.length(), T);
        EXPECT_EQ(cache.keys().shape(), (nn::Shape{2, kvh, T, 4}));
        EXPECT_LT(max_abs_diff(nn::concat(steps, 1), full), 1e-12) << name << " grad " << grad_mode;
      }
    }
}

TEST(NnAttention, KvCacheGradientsFlowThroughConcat) {
  nn::KvCache  cache;
  const Tensor k1 = randn({1, 1, 2, 3}, 70).requires_grad_(), k2 = randn({1, 1, 1, 3}, 71).requires_grad_();
  cache.append(k1, k1);
  const auto kv = cache.append(k2, k2);
  const auto g  = nn::grad(nn::sum(kv.first) + nn::sum(kv.second), {k1, k2});
  EXPECT_EQ(nn::sum(g[0]).item(), 12.0);
  EXPECT_EQ(nn::sum(g[1]).item(), 6.0);
  cache.reset();
  EXPECT_EQ(cache.length(), 0);
  EXPECT_FALSE(cache.keys().defined());
}

TEST(NnAttention, AdditiveAttentionMatchesHand) {
  nn::ParameterStore store(3, DType::Float64);
  const Tensor       q = randn({2, 5}, 80), k = randn({2, 6, 3}, 81), v = randn({2, 6, 2}, 82);
  const auto         a = nn::additive_attention(store, "add", q, k, v, 4);
  EXPECT_EQ(a.weights.shape(), (nn::Shape{2, 6}));
  EXPECT_EQ(a.output.shape(), (nn::Shape{2, 2}));
  EXPECT_EQ(store.count(), 5 * 4 + 3 * 4 + 4);
  const Tensor hidden = nn::tanh(nn::expand_dims(nn::matmul(q, store.at("add/Wq")), 1) +
                                 nn::matmul(k, store.at("add/Wk")));
  const Tensor scores = nn::squeeze(nn::matmul(hidden, store.at("add/v")), {2});
  const Tensor w      = nn::softmax(scores, -1);
  EXPECT_LT(max_abs_diff(a.weights, w), 1e-14);
  EXPECT_LT(max_abs_diff(a.output, nn::sum(nn::expand_dims(w, -1) * v, {1})), 1e-14);
  // Keep mask hides the last two steps.
  const Tensor keep = nn::less(nn::reshape(nn::arange(0, 6, 1, DType::Int64), {1, 6}), 4.0);
  const auto   m    = nn::additive_attention(store, "add", q, k, v, 4, keep);
  EXPECT_EQ(m.weights.at({1, 5}), 0.0);
  const auto f = [&](const Inputs& in) { return nn::additive_attention(store, "add", in[0], in[1], in[2], 4, keep).output; };
  EXPECT_TRUE(nn::gradcheck(f, {q.clone().requires_grad_(), k.clone().requires_grad_(), v.clone().requires_grad_()}).ok);
}

// ---- positional encodings ----------------------------------------------------------------------------

TEST(NnPositional, SinusoidalTable) {
  const Tensor pe = nn::sinusoidal_encoding(50, 16);
  EXPECT_EQ(pe.shape(), (nn::Shape{50, 16}));
  EXPECT_NEAR(pe.at({0, 0}), 0.0, 1e-7);
  EXPECT_NEAR(pe.at({0, 1}), 1.0, 1e-7);
  EXPECT_NEAR(pe.at({3, 2}), std::sin(3 / std::pow(10000.0, 2.0 / 16)), 1e-6);
  EXPECT_NEAR(pe.at({3, 3}), std::cos(3 / std::pow(10000.0, 2.0 / 16)), 1e-6);
  // Every row has norm sqrt(d / 2).
  expect_vec_near(nn::norm(pe, {1}), std::vector<double>(50, std::sqrt(8.0)), 1e-5);
  // An offset continues the table.
  EXPECT_LT(max_abs_diff(nn::sinusoidal_encoding(10, 16, 10000, DType::Float32, 40), nn::slice_axis(pe, 0, 40, 50)), 1e-6);
}

TEST(NnPositional, LearnedTable) {
  nn::ParameterStore store(1);
  const Tensor       t = nn::learned_encoding(store, "enc", 10, 4);
  EXPECT_EQ(t.shape(), (nn::Shape{10, 4}));
  EXPECT_TRUE(store.contains("enc/pos"));
  EXPECT_LT(nn::max(nn::abs(t)).item(), 0.2);
}

TEST(NnPositional, RotaryMatchesHandRotation) {
  const int    T = 12, d = 8;
  const Tensor q = randn({1, T, d}, 90);
  const Tensor r = nn::apply_rotary(q);
  double       err = 0;
  for (int t = 0; t < T; ++t)
    for (int j = 0; j < d / 2; ++j) {
      const double th = t / std::pow(10000.0, 2.0 * j / d);
      const double a = q.at({0, t, 2 * j}), b = q.at({0, t, 2 * j + 1});
      err = std::max(err, std::abs(a * std::cos(th) - b * std::sin(th) - r.at({0, t, 2 * j})));
      err = std::max(err, std::abs(a * std::sin(th) + b * std::cos(th) - r.at({0, t, 2 * j + 1})));
    }
  EXPECT_LT(err, 1e-12);
  // Half-split pairs (j, j + d/2).
  const Tensor h = nn::apply_rotary(q, 0, 10000, nn::RotaryLayout::HalfSplit);
  err            = 0;
  for (int t = 0; t < T; ++t)
    for (int j = 0; j < d / 2; ++j) {
      const double th = t / std::pow(10000.0, 2.0 * j / d);
      const double a = q.at({0, t, j}), b = q.at({0, t, j + d / 2});
      err = std::max(err, std::abs(a * std::cos(th) - b * std::sin(th) - h.at({0, t, j})));
      err = std::max(err, std::abs(a * std::sin(th) + b * std::cos(th) - h.at({0, t, j + d / 2})));
    }
  EXPECT_LT(err, 1e-12);
  // Offset == explicit positions; the score depends only on the offset m - n.
  const Tensor pos = nn::arange(5, 5 + T, 1, DType::Int64);
  EXPECT_LT(max_abs_diff(nn::apply_rotary(q, 5), nn::apply_rotary(q, pos)), 1e-12);
  const Tensor same_q = nn::broadcast_to(nn::slice_axis(q, 1, 0, 1), {1, T, d});
  const Tensor same_k = nn::broadcast_to(nn::slice_axis(randn({1, T, d}, 91), 1, 0, 1), {1, T, d});
  const Tensor s = nn::matmul(nn::apply_rotary(same_q), nn::apply_rotary(same_k), false, true);
  for (int n = 0; n + 3 < T; ++n) EXPECT_NEAR(s.at({0, n + 3, n}), s.at({0, 3, 0}), 1e-12);
}

TEST(NnPositional, RotaryGradcheck) {
  const Tensor x = randn({2, 2, 3, 6}, 92).requires_grad_();
  for (auto layout : {nn::RotaryLayout::Interleaved, nn::RotaryLayout::HalfSplit}) {
    const auto f = [&](const Inputs& in) { return nn::apply_rotary(in[0], 2, 100.0, layout); };
    EXPECT_TRUE(nn::gradcheck(f, {x}).ok);
    const Tensor pos = nn::tensor({0, 4, 9, 1, 2, 3}, {2, 3}, DType::Int64);
    const auto   fp  = [&](const Inputs& in) { return nn::apply_rotary(in[0], pos, 100.0, layout); };
    EXPECT_TRUE(nn::gradcheck(fp, {x}).ok);
  }
  EXPECT_THROW(nn::apply_rotary(randn({2, 5}, 1)), std::invalid_argument);
}

TEST(NnPositional, Alibi) {
  const auto s = nn::alibi_slopes(4);
  ASSERT_EQ(s.size(), 4u);
  EXPECT_DOUBLE_EQ(s[0], 0.25);
  EXPECT_DOUBLE_EQ(s[3], 1.0 / 256);
  const Tensor b = nn::alibi_bias(4, 5);
  EXPECT_EQ(b.shape(), (nn::Shape{4, 5, 5}));
  EXPECT_FLOAT_EQ(static_cast<float>(b.at({0, 4, 1})), -0.75f);
  EXPECT_FLOAT_EQ(static_cast<float>(b.at({0, 1, 4})), -1e9f);
  const Tensor nb = nn::alibi_bias(2, 3, false);
  EXPECT_DOUBLE_EQ(nb.at({0, 0, 2}), -0.0625 * 2);  // two heads: slope 2^-4
  // Rectangular decode rows equal the tail rows of the square bias.
  EXPECT_LT(max_abs_diff(nn::alibi_bias(4, 2, 5, true), nn::slice_axis(b, 1, 3, 5)), 0.0 + 1e-12);
}

TEST(NnPositional, RelativePositionBuckets) {
  const Tensor bk = nn::relative_position_buckets(1, 200, true, 32, 128);
  // Key 199 is the query itself (bottom-right alignment): bucket 0; keys before it count up.
  EXPECT_EQ(bk.at({0, 199}), 0);
  EXPECT_EQ(bk.at({0, 198}), 1);
  EXPECT_EQ(bk.at({0, 192}), 7);
  EXPECT_EQ(bk.at({0, 0}), 15);  // far past saturates at the last bucket of the half
  const Tensor sq = nn::relative_position_buckets(3, 3, true, 32, 128);
  EXPECT_EQ(sq.at({0, 1}), 17);  // future keys use the other half
  const Tensor uni = nn::relative_position_buckets(3, 3, false, 32, 128);
  EXPECT_EQ(uni.at({0, 2}), 0);
  EXPECT_EQ(uni.at({2, 0}), 2);
  nn::ParameterStore store(1, DType::Float64);
  const Tensor       rb = nn::relative_position_bias(store, "rel", 4, 3, 5);
  EXPECT_EQ(rb.shape(), (nn::Shape{4, 3, 5}));
  EXPECT_EQ(store.at("rel/rel_bias").shape(), (nn::Shape{32, 4}));
}
