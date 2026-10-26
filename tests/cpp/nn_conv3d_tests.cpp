// Tests for datamunge::nn 3-D convolution and pooling (NDHWC).

#include <gtest/gtest.h>

#include <cmath>
#include <datamunge/nn/conv.hpp>
#include <datamunge/nn/gradcheck.hpp>
#include <datamunge/nn/layers.hpp>
#include <datamunge/nn/nn.hpp>
#include <datamunge/nn/parallel.hpp>
#include <vector>

namespace nn = datamunge::nn;
using nn::DType;
using nn::Padding;
using nn::Tensor;

namespace {

using Inputs = std::vector<Tensor>;
using I64    = std::int64_t;

Tensor rnd(const nn::Shape& shape, std::uint64_t seed, DType dt = DType::Float64) {
  nn::Generator gen(seed);
  return nn::normal(shape, 0.0, 1.0, gen, dt);
}

Tensor leaf(const Tensor& t) {
  Tensor l = t.detach();
  l.requires_grad_(true);
  return l;
}

/// Output length and front padding of one axis under TF's rules.
I64 axis_len(I64 in, I64 k, I64 s, I64 d, Padding p, I64 eb, I64 ea, I64& before) {
  const I64 eff = d * (k - 1) + 1;
  if (p == Padding::Valid) {
    before = 0;
    return (in - eff) / s + 1;
  }
  if (p == Padding::Explicit) {
    before = eb;
    return (in + eb + ea - eff) / s + 1;
  }
  const I64 out = (in + s - 1) / s;
  before        = std::max<I64>((out - 1) * s + eff - in, 0) / 2;
  return out;
}

/// Direct-loop conv3d reference (float64), straight from the definition.
std::vector<double> conv3d_reference(const Tensor& x, const Tensor& f, const nn::Conv3DOptions& o, nn::Shape& out_shape) {
  const auto xs = x.shape(), fs = f.shape();
  const I64  N = xs[0], D = xs[1], H = xs[2], W = xs[3], C = xs[4];
  const I64  KD = fs[0], KH = fs[1], KW = fs[2], CG = fs[3], O = fs[4], OG = O / o.groups;
  const auto& e = o.explicit_paddings;
  I64        pf = 0, pt = 0, pl = 0;
  const I64  OD = axis_len(D, KD, o.strides[0], o.dilations[0], o.padding, e[0], e[1], pf);
  const I64  OH = axis_len(H, KH, o.strides[1], o.dilations[1], o.padding, e[2], e[3], pt);
  const I64  OW = axis_len(W, KW, o.strides[2], o.dilations[2], o.padding, e[4], e[5], pl);
  out_shape     = {N, OD, OH, OW, O};
  const auto          xv = x.to_vector<double>(), fv = f.to_vector<double>();
  std::vector<double> y(static_cast<std::size_t>(N * OD * OH * OW * O), 0.0);
  for (I64 n = 0; n < N; ++n)
    for (I64 oz = 0; oz < OD; ++oz)
      for (I64 oy = 0; oy < OH; ++oy)
        for (I64 ox = 0; ox < OW; ++ox)
          for (I64 oc = 0; oc < O; ++oc) {
            const I64 g = oc / OG;
            double    s = 0;
            for (I64 a = 0; a < KD; ++a)
              for (I64 i = 0; i < KH; ++i)
                for (I64 j = 0; j < KW; ++j)
                  for (I64 c = 0; c < CG; ++c) {
                    const I64 iz = oz * o.strides[0] - pf + a * o.dilations[0];
                    const I64 iy = oy * o.strides[1] - pt + i * o.dilations[1];
                    const I64 ix = ox * o.strides[2] - pl + j * o.dilations[2];
                    if (iz < 0 || iz >= D || iy < 0 || iy >= H || ix < 0 || ix >= W) continue;
                    s += xv[static_cast<std::size_t>((((n * D + iz) * H + iy) * W + ix) * C + g * CG + c)] *
                         fv[static_cast<std::size_t>((((a * KH + i) * KW + j) * CG + c) * O + oc)];
                  }
            y[static_cast<std::size_t>((((n * OD + oz) * OH + oy) * OW + ox) * O + oc)] = s;
          }
  return y;
}

struct Case {
  nn::Shape          x, f;
  nn::Conv3DOptions  o;
};

std::vector<Case> cases() {
  std::vector<Case> c;
  c.push_back({{2, 4, 5, 3, 2}, {3, 3, 3, 2, 3}, nn::conv3d_options(1, Padding::Same)});
  c.push_back({{2, 4, 5, 3, 2}, {3, 3, 2, 2, 3}, nn::conv3d_options(1, Padding::Valid)});
  c.push_back({{1, 5, 6, 7, 2}, {2, 2, 2, 2, 3}, nn::conv3d_options(2, Padding::Same)});  // even kernel
  c.push_back({{1, 5, 6, 5, 2}, {3, 3, 3, 2, 2}, nn::conv3d_options(2, Padding::Valid)});
  c.push_back({{1, 6, 5, 5, 1}, {2, 3, 2, 1, 2}, nn::conv3d_options(1, Padding::Same, 2)});  // dilation
  nn::Conv3DOptions ex = nn::conv3d_options(1, Padding::Explicit);
  ex.explicit_paddings = {1, 0, 0, 2, 1, 1};
  c.push_back({{2, 3, 4, 4, 2}, {2, 3, 2, 2, 3}, ex});
  nn::Conv3DOptions grp = nn::conv3d_options(1, Padding::Same);
  grp.groups            = 2;
  c.push_back({{1, 3, 4, 4, 4}, {3, 2, 3, 2, 6}, grp});
  c.push_back({{2, 3, 3, 4, 5}, {1, 1, 1, 5, 3}, nn::conv3d_options(1, Padding::Valid)});  // pointwise
  nn::Conv3DOptions rect;
  rect.strides = {2, 1, 3};
  c.push_back({{1, 7, 4, 8, 2}, {3, 2, 3, 2, 2}, rect});
  return c;
}

}  // namespace

// ---- convolution ------------------------------------------------------------------------------

TEST(NnConv3d, MatchesDirectReference) {
  for (const auto& cs : cases()) {
    const Tensor x = rnd(cs.x, 11), f = rnd(cs.f, 12);
    nn::Shape    shape;
    const auto   ref = conv3d_reference(x, f, cs.o, shape);
    const Tensor y   = nn::conv3d(x, f, cs.o);
    ASSERT_EQ(y.shape(), shape) << nn::shape_str(cs.x) << " * " << nn::shape_str(cs.f);
    const auto v = y.to_vector<double>();
    for (std::size_t i = 0; i < v.size(); ++i) ASSERT_NEAR(v[i], ref[i], 1e-10) << "at " << i;
    const auto v32 = nn::conv3d(nn::cast(x, DType::Float32), nn::cast(f, DType::Float32), cs.o).to_vector<double>();
    for (std::size_t i = 0; i < v.size(); ++i) ASSERT_NEAR(v32[i], ref[i], 1e-4);
  }
}

TEST(NnConv3d, MatchesConv2dWhenDepthIsOne) {
  const Tensor x = rnd({2, 1, 6, 5, 3}, 13), f = rnd({1, 3, 3, 3, 4}, 14);
  const Tensor y3 = nn::conv3d(x, f, nn::conv3d_options(2, Padding::Same));
  const Tensor y2 = nn::conv2d(nn::squeeze(x, {1}), nn::squeeze(f, {0}), nn::conv_options(2, Padding::Same));
  const auto   a = y3.to_vector<double>(), b = y2.to_vector<double>();
  ASSERT_EQ(a.size(), b.size());
  for (std::size_t i = 0; i < a.size(); ++i) ASSERT_NEAR(a[i], b[i], 1e-12);
}

TEST(NnConv3d, Gradcheck) {
  for (const auto& cs : cases()) {
    const auto r = nn::gradcheck([&](const Inputs& in) { return nn::conv3d(in[0], in[1], cs.o); },
                                 {leaf(rnd(cs.x, 21)), leaf(rnd(cs.f, 22))});
    EXPECT_TRUE(r.ok) << nn::shape_str(cs.x) << " * " << nn::shape_str(cs.f) << ": " << r.message;
  }
}

TEST(NnConv3d, GradGradcheck) {
  const auto c = cases();
  for (std::size_t k : {std::size_t{2}, std::size_t{5}, std::size_t{6}}) {
    const auto& cs = c[k];
    const auto  r  = nn::gradgradcheck([&](const Inputs& in) { return nn::conv3d(in[0], in[1], cs.o); },
                                       {leaf(rnd(cs.x, 31)), leaf(rnd(cs.f, 32))});
    EXPECT_TRUE(r.ok) << k << ": " << r.message;
  }
}

TEST(NnConv3d, BackpropInputAndFilterGradchecks) {
  const auto  c  = cases();
  const auto& cs = c[2];
  nn::Shape   out;
  conv3d_reference(rnd(cs.x, 1), rnd(cs.f, 2), cs.o, out);
  auto r = nn::gradcheck([&](const Inputs& in) { return nn::conv3d_backprop_input(cs.x, in[0], in[1], cs.o); },
                         {leaf(rnd(cs.f, 41)), leaf(rnd(out, 42))});
  EXPECT_TRUE(r.ok) << r.message;
  r = nn::gradcheck([&](const Inputs& in) { return nn::conv3d_backprop_filter(in[0], cs.f, in[1], cs.o); },
                    {leaf(rnd(cs.x, 43)), leaf(rnd(out, 44))});
  EXPECT_TRUE(r.ok) << r.message;
  r = nn::gradgradcheck([&](const Inputs& in) { return nn::conv3d_backprop_input(cs.x, in[0], in[1], cs.o); },
                        {leaf(rnd(cs.f, 45)), leaf(rnd(out, 46))});
  EXPECT_TRUE(r.ok) << r.message;
}

TEST(NnConv3d, TransposeIsAdjoint) {
  // <conv3d(x, W), y> == <x, conv3d_transpose(y, W)>
  for (auto pad : {Padding::Same, Padding::Valid}) {
    const nn::Conv3DOptions o = nn::conv3d_options(2, pad);
    const Tensor            x = rnd({2, 6, 7, 6, 3}, 51), w = rnd({3, 3, 2, 3, 4}, 52);
    const Tensor            y0 = nn::conv3d(x, w, o);
    const Tensor            y  = rnd(y0.shape(), 53);
    const double            lhs = nn::sum(y0 * y).item<double>();
    const Tensor            xt  = nn::conv3d_transpose(y, w, x.shape(), o);
    ASSERT_EQ(xt.shape(), x.shape());
    EXPECT_NEAR(lhs, nn::sum(x * xt).item<double>(), 1e-9);
  }
  // Default output sizes: SAME in s, VALID (in - 1) s + k.
  const Tensor w = rnd({2, 2, 2, 3, 5}, 54);
  EXPECT_EQ(nn::conv3d_transpose(rnd({1, 2, 3, 4, 5}, 1), w, nn::conv3d_options(2, Padding::Valid)).shape(),
            (nn::Shape{1, 4, 6, 8, 3}));
  EXPECT_EQ(nn::conv3d_transpose(rnd({1, 2, 3, 4, 5}, 1), rnd({3, 3, 3, 3, 5}, 2), nn::conv3d_options(2)).shape(),
            (nn::Shape{1, 4, 6, 8, 3}));
  const auto r = nn::gradcheck(
      [&](const Inputs& in) { return nn::conv3d_transpose(in[0], in[1], nn::conv3d_options(2, Padding::Valid)); },
      {leaf(rnd({1, 2, 2, 3, 5}, 55)), leaf(rnd({2, 2, 2, 3, 5}, 56))});
  EXPECT_TRUE(r.ok) << r.message;
}

TEST(NnConv3d, LargeAndDeterministic) {
  // Big enough to chunk vol2col over the batch; batch-1 results must agree,
  // and repeated runs (input gradient uses phased parallel accumulation) are bitwise equal.
  const Tensor x = rnd({12, 16, 32, 32, 8}, 61, DType::Float32), f = rnd({3, 3, 3, 8, 8}, 62, DType::Float32);
  const Tensor y  = nn::conv3d(x, f);
  const Tensor y0 = nn::conv3d(nn::slice(x, {11, 0, 0, 0, 0}, {1, -1, -1, -1, -1}), f);
  const auto   a = nn::slice(y, {11, 0, 0, 0, 0}, {1, -1, -1, -1, -1}).to_vector<float>(), b = y0.to_vector<float>();
  for (std::size_t i = 0; i < a.size(); ++i) ASSERT_NEAR(a[i], b[i], 1e-4);
  const Tensor g  = rnd({1, 16, 32, 32, 8}, 63, DType::Float32);
  const auto   d1 = nn::conv3d_backprop_input({1, 16, 32, 32, 8}, f, g).to_vector<float>();
  const auto   d2 = nn::conv3d_backprop_input({1, 16, 32, 32, 8}, f, g).to_vector<float>();
  EXPECT_EQ(d1, d2);
  // Chunks split volumes at output-slice boundaries (75 slices per chunk here):
  // the input and filter gradients of a batch agree with per-volume results.
  const Tensor gb  = rnd({12, 16, 32, 32, 8}, 64, DType::Float32);
  const auto   dxb = nn::slice(nn::conv3d_backprop_input(x.shape(), f, gb), {5, 0, 0, 0, 0}, {1, -1, -1, -1, -1}).to_vector<float>();
  const auto   dx5 = nn::conv3d_backprop_input({1, 16, 32, 32, 8}, f, nn::slice(gb, {5, 0, 0, 0, 0}, {1, -1, -1, -1, -1})).to_vector<float>();
  for (std::size_t i = 0; i < dxb.size(); ++i) ASSERT_NEAR(dxb[i], dx5[i], 1e-4);
  const auto dfb = nn::conv3d_backprop_filter(x, f.shape(), gb).to_vector<double>();
  Tensor     dfs = nn::zeros(f.shape(), DType::Float64);
  for (I64 n = 0; n < 12; ++n)
    dfs = dfs + nn::cast(nn::conv3d_backprop_filter(nn::slice(x, {n, 0, 0, 0, 0}, {1, -1, -1, -1, -1}), f.shape(),
                                                    nn::slice(gb, {n, 0, 0, 0, 0}, {1, -1, -1, -1, -1})),
                         DType::Float64);
  const auto dfv = dfs.to_vector<double>();
  for (std::size_t i = 0; i < dfb.size(); ++i) ASSERT_NEAR(dfb[i], dfv[i], 2e-2 + 1e-4 * std::abs(dfv[i]));
}

// ---- pooling ----------------------------------------------------------------------------------

TEST(NnPool3d, MaxAndAvgValues) {
  // 1x4x4x4x1 ramp 0..63
  std::vector<double> v(64);
  for (int i = 0; i < 64; ++i) v[static_cast<std::size_t>(i)] = i;
  const Tensor x = nn::from_vector(v, {1, 4, 4, 4, 1});
  EXPECT_EQ(nn::max_pool3d(x, 2, 2).to_vector<double>(), (std::vector<double>{21, 23, 29, 31, 53, 55, 61, 63}));
  // mean of {0,1,4,5,16,17,20,21} = 10.5
  EXPECT_EQ(nn::avg_pool3d(x, 2, 2).to_vector<double>(), (std::vector<double>{10.5, 12.5, 18.5, 20.5, 42.5, 44.5, 50.5, 52.5}));
  // SAME 3x3x3 stride 2 on 4^3: out 2^3, padding after only; the corner window averages 2x2x2 entries.
  const auto avg = nn::avg_pool3d(x, 3, 2, Padding::Same).to_vector<double>();
  ASSERT_EQ(avg.size(), 8U);
  EXPECT_NEAR(avg[0], 21.0, 1e-12);  // mean over z,y,x in {0,1,2}: 16*1 + 4*1 + 1 = 21
  EXPECT_NEAR(avg[7], (42 + 43 + 46 + 47 + 58 + 59 + 62 + 63) / 8.0, 1e-12);
  // Negative inputs with SAME padding: padding never wins the max.
  const Tensor neg = x * -1.0 - 1.0;
  EXPECT_EQ(nn::max_pool3d(neg, 3, 2, Padding::Same).to_vector<double>(),
            (std::vector<double>{-1, -3, -9, -11, -33, -35, -41, -43}));
  // Anisotropic window (C3D pool1: time window 1).
  EXPECT_EQ(nn::max_pool3d(x, {1, 2, 2}, {1, 2, 2}).shape(), (nn::Shape{1, 4, 2, 2, 1}));
  EXPECT_EQ(nn::global_avg_pool3d(x).shape(), (nn::Shape{1, 1}));
  EXPECT_NEAR(nn::global_avg_pool3d(x).item<double>(), 31.5, 1e-12);
  EXPECT_NEAR(nn::global_max_pool3d(x, true).item<double>(), 63, 1e-12);
  EXPECT_EQ(nn::global_max_pool3d(x, true).shape(), (nn::Shape{1, 1, 1, 1, 1}));
}

TEST(NnPool3d, Gradients) {
  const Tensor x = rnd({2, 5, 4, 5, 2}, 81);
  for (auto pad : {Padding::Valid, Padding::Same}) {
    auto r = nn::gradcheck([&](const Inputs& in) { return nn::max_pool3d(in[0], {3, 2, 2}, {2, 2, 1}, pad); }, {leaf(x)});
    EXPECT_TRUE(r.ok) << r.message;
    r = nn::gradcheck([&](const Inputs& in) { return nn::avg_pool3d(in[0], {2, 3, 2}, {2, 1, 2}, pad); }, {leaf(x)});
    EXPECT_TRUE(r.ok) << r.message;
    r = nn::gradgradcheck([&](const Inputs& in) { return nn::square(nn::avg_pool3d(in[0], 2, 2, pad)); }, {leaf(x)});
    EXPECT_TRUE(r.ok) << r.message;
    r = nn::gradgradcheck([&](const Inputs& in) { return nn::square(nn::max_pool3d(in[0], 2, 2, pad)); }, {leaf(x)});
    EXPECT_TRUE(r.ok) << r.message;
  }
  // Ties route the gradient to the first maximum.
  Tensor ones = nn::ones({1, 2, 2, 2, 1}, DType::Float64);
  ones.requires_grad_(true);
  nn::sum(nn::max_pool3d(ones, 2, 2)).backward();
  EXPECT_EQ(ones.grad().to_vector<double>(), (std::vector<double>{1, 0, 0, 0, 0, 0, 0, 0}));
  // SAME average gradient: each input gets 1 / (in-bounds count) from every window covering it.
  Tensor y = nn::ones({1, 3, 1, 1, 1}, DType::Float64);
  y.requires_grad_(true);
  nn::sum(nn::avg_pool3d(y, {2, 1, 1}, {2, 1, 1}, Padding::Same)).backward();
  EXPECT_EQ(y.grad().to_vector<double>(), (std::vector<double>{0.5, 0.5, 1.0}));
}

// ---- layers -----------------------------------------------------------------------------------

TEST(NnConv3dLayers, ShapesAndParameters) {
  nn::ParameterStore p(1, DType::Float32);
  const Tensor       x = rnd({2, 8, 6, 6, 3}, 91, DType::Float32);
  nn::ConvLayerOptions o;
  o.activation = nn::Activation::Relu;
  const Tensor y = nn::conv3d(p, "c", x, 4, 3, o);
  EXPECT_EQ(y.shape(), (nn::Shape{2, 8, 6, 6, 4}));
  EXPECT_EQ(p.at("c/W").shape(), (nn::Shape{3, 3, 3, 3, 4}));
  EXPECT_EQ(p.at("c/b").shape(), (nn::Shape{4}));
  EXPECT_EQ(p.count(), 3 * 3 * 3 * 3 * 4 + 4);
  // Glorot-uniform bound uses fans 27*3 and 27*4.
  const double bound = std::sqrt(6.0 / (27 * 3 + 27 * 4));
  for (float w : p.at("c/W").to_vector<float>()) ASSERT_LE(std::abs(w), bound + 1e-6);
  nn::ConvLayerOptions s2;
  s2.stride = 2, s2.padding = Padding::Valid;
  EXPECT_EQ(nn::conv3d(p, "d", x, 5, 2, s2).shape(), (nn::Shape{2, 4, 3, 3, 5}));
  // Reuse by name.
  const auto before = p.count();
  nn::conv3d(p, "c", x, 4, 3, o);
  EXPECT_EQ(p.count(), before);
  // Transposed: SAME doubles every axis; W [k, k, k, filters, in].
  nn::ConvLayerOptions up;
  up.stride = 2, up.padding = Padding::Valid;
  const Tensor u = nn::conv3d_transpose(p, "up", x, 2, 2, up);
  EXPECT_EQ(u.shape(), (nn::Shape{2, 16, 12, 12, 2}));
  EXPECT_EQ(p.at("up/W").shape(), (nn::Shape{2, 2, 2, 2, 3}));
  up.padding = Padding::Same;
  EXPECT_EQ(nn::conv3d_transpose(p, "up3", x, 2, 3, up).shape(), (nn::Shape{2, 16, 12, 12, 2}));
}
