// Tests for datamunge::nn GEMM, threading, convolution, pooling and batch norm.

#include <gtest/gtest.h>

#include <atomic>
#include <cmath>
#include <datamunge/nn/conv.hpp>
#include <datamunge/nn/nn.hpp>
#include <datamunge/nn/parallel.hpp>
#include <limits>
#include <thread>
#include <vector>

namespace nn = datamunge::nn;
using nn::DType;
using nn::Padding;
using nn::Tensor;

namespace {

using Inputs = std::vector<Tensor>;

Tensor rnd(const nn::Shape& shape, std::uint64_t seed, DType dt = DType::Float64) {
  nn::Generator gen(seed);
  return nn::normal(shape, 0.0, 1.0, gen, dt);
}

Tensor leaf(const Tensor& t) {
  Tensor l = t.detach();
  l.requires_grad_(true);
  return l;
}

/// Direct-loop reference for conv2d (float64), straight from the definition.
std::vector<double> conv_reference(const Tensor& x, const Tensor& f, const nn::Conv2DOptions& o, nn::Shape& out_shape) {
  const auto xs = x.shape(), fs = f.shape();
  const auto N = xs[0], H = xs[1], W = xs[2], C = xs[3], KH = fs[0], KW = fs[1], CG = fs[2], O = fs[3];
  const auto G = o.groups, OG = O / G;
  auto       axis = [&](std::int64_t in, std::int64_t k, std::int64_t s, std::int64_t d, std::int64_t eb,
                  std::int64_t ea, std::int64_t& before) {
    const std::int64_t eff = d * (k - 1) + 1;
    if (o.padding == Padding::Valid) {
      before = 0;
      return (in - eff) / s + 1;
    }
    if (o.padding == Padding::Explicit) {
      before = eb;
      return (in + eb + ea - eff) / s + 1;
    }
    const std::int64_t out = (in + s - 1) / s;
    before                 = std::max<std::int64_t>((out - 1) * s + eff - in, 0) / 2;
    return out;
  };
  std::int64_t pt = 0, pl = 0;
  const auto   OH = axis(H, KH, o.strides[0], o.dilations[0], o.explicit_paddings[0], o.explicit_paddings[1], pt);
  const auto   OW = axis(W, KW, o.strides[1], o.dilations[1], o.explicit_paddings[2], o.explicit_paddings[3], pl);
  out_shape       = {N, OH, OW, O};
  const auto           xv = x.to_vector<double>(), fv = f.to_vector<double>();
  std::vector<double> y(static_cast<std::size_t>(N * OH * OW * O), 0.0);
  for (std::int64_t n = 0; n < N; ++n)
    for (std::int64_t oy = 0; oy < OH; ++oy)
      for (std::int64_t ox = 0; ox < OW; ++ox)
        for (std::int64_t oc = 0; oc < O; ++oc) {
          const std::int64_t g = oc / OG;
          double             s = 0;
          for (std::int64_t i = 0; i < KH; ++i)
            for (std::int64_t j = 0; j < KW; ++j)
              for (std::int64_t c = 0; c < CG; ++c) {
                const std::int64_t iy = oy * o.strides[0] - pt + i * o.dilations[0];
                const std::int64_t ix = ox * o.strides[1] - pl + j * o.dilations[1];
                if (iy < 0 || iy >= H || ix < 0 || ix >= W) continue;
                s += xv[static_cast<std::size_t>(((n * H + iy) * W + ix) * C + g * CG + c)] *
                     fv[static_cast<std::size_t>(((i * KW + j) * CG + c) * O + oc)];
              }
          y[static_cast<std::size_t>(((n * OH + oy) * OW + ox) * O + oc)] = s;
        }
  return y;
}

struct ConvCase {
  nn::Shape          x, f;
  nn::Conv2DOptions  o;
};

std::vector<ConvCase> conv_cases() {
  std::vector<ConvCase> cases;
  cases.push_back({{2, 5, 6, 3}, {3, 3, 3, 4}, nn::conv_options(1, Padding::Same)});
  cases.push_back({{2, 5, 6, 3}, {3, 3, 3, 4}, nn::conv_options(1, Padding::Valid)});
  cases.push_back({{1, 7, 8, 2}, {3, 2, 2, 3}, nn::conv_options(2, Padding::Same)});
  cases.push_back({{1, 7, 8, 2}, {4, 4, 2, 3}, nn::conv_options(2, Padding::Same)});  // even kernel
  cases.push_back({{1, 7, 8, 2}, {3, 3, 2, 2}, nn::conv_options(2, Padding::Valid)});
  cases.push_back({{1, 8, 7, 2}, {3, 3, 2, 2}, nn::conv_options(1, Padding::Same, 2)});  // dilation
  nn::Conv2DOptions ex = nn::conv_options(1, Padding::Explicit);
  ex.explicit_paddings = {1, 0, 2, 1};
  cases.push_back({{2, 4, 5, 2}, {2, 3, 2, 3}, ex});
  nn::Conv2DOptions grp = nn::conv_options(1, Padding::Same);
  grp.groups             = 2;
  cases.push_back({{2, 5, 5, 4}, {3, 3, 2, 6}, grp});
  cases.push_back({{2, 4, 4, 5}, {1, 1, 5, 3}, nn::conv_options(1, Padding::Valid)});  // pointwise fast path
  nn::Conv2DOptions rect;
  rect.strides = {1, 2};
  cases.push_back({{1, 6, 7, 2}, {2, 3, 2, 2}, rect});
  return cases;
}

}  // namespace

// ---- GEMM and threading -----------------------------------------------------------------------

TEST(NnParallel, CoversRangeOnce) {
  std::vector<std::atomic<int>> hits(10007);
  nn::parallel_for(0, 10007, 7, [&](std::int64_t lo, std::int64_t hi) {
    for (auto i = lo; i < hi; ++i) hits[static_cast<std::size_t>(i)]++;
  });
  for (auto& h : hits) EXPECT_EQ(h.load(), 1);
}

TEST(NnParallel, NestedAndExceptions) {
  std::atomic<long> total{0};
  nn::parallel_for(0, 64, 1, [&](std::int64_t lo, std::int64_t hi) {
    for (auto i = lo; i < hi; ++i)
      nn::parallel_for(0, 100, 1, [&](std::int64_t a, std::int64_t b) { total += b - a; });
  });
  EXPECT_EQ(total.load(), 6400);
  EXPECT_THROW(nn::parallel_for(0, 1000, 1,
                                [&](std::int64_t lo, std::int64_t) {
                                  if (lo == 0) throw std::runtime_error("boom");
                                }),
               std::runtime_error);
}

TEST(NnParallel, ManySmallJobsStress) {
  // Back-to-back jobs exercise workers that wake late from a previous job.
  for (int rep = 0; rep < 20000; ++rep) {
    std::atomic<int> count{0};
    nn::parallel_for(0, 64, 1, [&](std::int64_t lo, std::int64_t hi) { count += static_cast<int>(hi - lo); });
    ASSERT_EQ(count.load(), 64) << "at job " << rep;
  }
}

TEST(NnParallel, ThreadCountAndBackend) {
  const int before = nn::get_num_threads();
  nn::set_num_threads(1);
  EXPECT_EQ(nn::get_num_threads(), 1);
  const Tensor a = rnd({33, 17}, 1), b = rnd({17, 9}, 2);
  const auto   serial = nn::matmul(a, b).to_vector<double>();
  nn::set_num_threads(4);
  const auto par = nn::matmul(a, b).to_vector<double>();
  for (std::size_t i = 0; i < serial.size(); ++i) EXPECT_NEAR(serial[i], par[i], 1e-12);
  nn::set_num_threads(before);
  EXPECT_FALSE(nn::gemm_backend().empty());
}

TEST(NnGemm, MatmulTransposesAndBatchesMatchNaive) {
  for (DType dt : {DType::Float64, DType::Float32}) {
    const double tol = dt == DType::Float64 ? 1e-10 : 1e-3;
    for (bool ta : {false, true})
      for (bool tb : {false, true}) {
        const Tensor a = rnd(ta ? nn::Shape{3, 70, 65} : nn::Shape{3, 65, 70}, 3, dt);
        const Tensor b = rnd(tb ? nn::Shape{40, 70} : nn::Shape{70, 40}, 4, dt);
        const Tensor c = nn::matmul(a, b, ta, tb);
        ASSERT_EQ(c.shape(), (nn::Shape{3, 65, 40}));
        const auto av = a.to_vector<double>(), bv = b.to_vector<double>(), cv = c.to_vector<double>();
        for (int p = 0; p < 3; ++p)
          for (int i = 0; i < 65; i += 7)
            for (int j = 0; j < 40; j += 3) {
              double s = 0;
              for (int l = 0; l < 70; ++l) {
                const double x = ta ? av[p * 70 * 65 + l * 65 + i] : av[p * 65 * 70 + i * 70 + l];
                const double y = tb ? bv[j * 70 + l] : bv[l * 40 + j];
                s += x * y;
              }
              EXPECT_NEAR(cv[static_cast<std::size_t>(p * 65 * 40 + i * 40 + j)], s, tol);
            }
      }
  }
}

// ---- convolution ------------------------------------------------------------------------------

TEST(NnConv, MatchesDirectReference) {
  for (const auto& cs : conv_cases()) {
    const Tensor x = rnd(cs.x, 11), f = rnd(cs.f, 12);
    nn::Shape    shape;
    const auto   ref = conv_reference(x, f, cs.o, shape);
    const Tensor y   = nn::conv2d(x, f, cs.o);
    ASSERT_EQ(y.shape(), shape) << nn::shape_str(cs.x) << " * " << nn::shape_str(cs.f);
    const auto v = y.to_vector<double>();
    for (std::size_t i = 0; i < v.size(); ++i) ASSERT_NEAR(v[i], ref[i], 1e-10) << "at " << i;
    // float32 agrees too
    const auto v32 = nn::conv2d(nn::cast(x, DType::Float32), nn::cast(f, DType::Float32), cs.o).to_vector<double>();
    for (std::size_t i = 0; i < v.size(); ++i) ASSERT_NEAR(v32[i], ref[i], 1e-4);
  }
}

TEST(NnConv, OutputSizes) {
  EXPECT_EQ(nn::conv_output_size(28, 3, 1, 1, Padding::Same), 28);
  EXPECT_EQ(nn::conv_output_size(28, 3, 2, 1, Padding::Same), 14);
  EXPECT_EQ(nn::conv_output_size(7, 3, 2, 1, Padding::Same), 4);
  EXPECT_EQ(nn::conv_output_size(28, 5, 1, 1, Padding::Valid), 24);
  EXPECT_EQ(nn::conv_output_size(28, 3, 1, 2, Padding::Valid), 24);
  EXPECT_EQ(nn::conv_output_size(10, 3, 3, 1, Padding::Valid), 3);
}

TEST(NnConv, Gradcheck) {
  for (const auto& cs : conv_cases()) {
    const auto r = nn::gradcheck([&](const Inputs& in) { return nn::conv2d(in[0], in[1], cs.o); },
                                 {leaf(rnd(cs.x, 21)), leaf(rnd(cs.f, 22))});
    EXPECT_TRUE(r.ok) << nn::shape_str(cs.x) << " * " << nn::shape_str(cs.f) << ": " << r.message;
  }
}

TEST(NnConv, GradGradcheck) {
  const auto cases = conv_cases();
  for (std::size_t k : {std::size_t{2}, std::size_t{6}, std::size_t{7}}) {
    const auto& cs = cases[k];
    const auto  r  = nn::gradgradcheck([&](const Inputs& in) { return nn::conv2d(in[0], in[1], cs.o); },
                                       {leaf(rnd(cs.x, 31)), leaf(rnd(cs.f, 32))});
    EXPECT_TRUE(r.ok) << k << ": " << r.message;
  }
}

TEST(NnConv, BackpropInputAndFilterGradchecks) {
  const auto  cases = conv_cases();
  const auto& cs    = cases[3];
  nn::Shape   out;
  conv_reference(rnd(cs.x, 1), rnd(cs.f, 2), cs.o, out);
  const auto r1 = nn::gradcheck(
      [&](const Inputs& in) { return nn::conv2d_backprop_input(cs.x, in[0], in[1], cs.o); },
      {leaf(rnd(cs.f, 41)), leaf(rnd(out, 42))});
  EXPECT_TRUE(r1.ok) << r1.message;
  const auto r2 = nn::gradcheck(
      [&](const Inputs& in) { return nn::conv2d_backprop_filter(in[0], cs.f, in[1], cs.o); },
      {leaf(rnd(cs.x, 43)), leaf(rnd(out, 44))});
  EXPECT_TRUE(r2.ok) << r2.message;
}

TEST(NnConv, TransposeIsAdjoint) {
  // <conv(x, W), y> == <x, conv_transpose(y, W)>
  const nn::Conv2DOptions o = nn::conv_options(2, Padding::Same);
  const Tensor            x = rnd({2, 8, 8, 3}, 51), w = rnd({4, 4, 3, 5}, 52);
  const Tensor            y = rnd({2, 4, 4, 5}, 53);
  const double            lhs = nn::sum(nn::conv2d(x, w, o) * y).item<double>();
  const Tensor            xt  = nn::conv2d_transpose(y, w, o);
  ASSERT_EQ(xt.shape(), (nn::Shape{2, 8, 8, 3}));
  EXPECT_NEAR(lhs, nn::sum(x * xt).item<double>(), 1e-9);
  // VALID default output size (H - 1) s + k
  EXPECT_EQ(nn::conv2d_transpose(rnd({1, 7, 7, 5}, 1), w, nn::conv_options(2, Padding::Valid)).shape(),
            (nn::Shape{1, 16, 16, 3}));
  const auto r = nn::gradcheck([&](const Inputs& in) { return nn::conv2d_transpose(in[0], in[1], o); },
                               {leaf(rnd({1, 3, 3, 5}, 54)), leaf(rnd({4, 4, 3, 5}, 55))});
  EXPECT_TRUE(r.ok) << r.message;
}

TEST(NnConv, DepthwiseSeparableAndConv1d) {
  const Tensor x = rnd({2, 6, 5, 3}, 61), f = rnd({3, 3, 3, 2}, 62);
  const auto   o = nn::conv_options(1, Padding::Same);
  const Tensor y = nn::depthwise_conv2d(x, f, o);
  ASSERT_EQ(y.shape(), (nn::Shape{2, 6, 5, 6}));
  // Output channel c * M + m is input channel c filtered by f[..., c, m].
  for (int c = 0; c < 3; ++c)
    for (int m = 0; m < 2; ++m) {
      const Tensor xc  = nn::slice(x, {0, 0, 0, c}, {-1, -1, -1, 1});
      const Tensor fc  = nn::slice(f, {0, 0, c, m}, {-1, -1, 1, 1});
      const auto   ref = nn::conv2d(xc, fc, o).to_vector<double>();
      const auto   got = nn::slice(y, {0, 0, 0, c * 2 + m}, {-1, -1, -1, 1}).to_vector<double>();
      for (std::size_t i = 0; i < ref.size(); ++i) ASSERT_NEAR(got[i], ref[i], 1e-10);
    }
  const Tensor pw = rnd({1, 1, 6, 4}, 63);
  EXPECT_EQ(nn::separable_conv2d(x, f, pw, o).shape(), (nn::Shape{2, 6, 5, 4}));
  const auto r = nn::gradcheck([&](const Inputs& in) { return nn::depthwise_conv2d(in[0], in[1], nn::conv_options(2)); },
                               {leaf(x), leaf(f)});
  EXPECT_TRUE(r.ok) << r.message;

  const Tensor seq = rnd({2, 9, 3}, 64), k = rnd({3, 3, 4}, 65);
  const Tensor c1  = nn::conv1d(seq, k, 2, Padding::Valid);
  ASSERT_EQ(c1.shape(), (nn::Shape{2, 4, 4}));
  const auto via2d = nn::conv2d(nn::expand_dims(seq, 1), nn::reshape(k, {1, 3, 3, 4}),
                                [] {
                                  nn::Conv2DOptions o2;
                                  o2.strides = {1, 2};
                                  o2.padding = Padding::Valid;
                                  return o2;
                                }())
                         .to_vector<double>();
  const auto got = c1.to_vector<double>();
  for (std::size_t i = 0; i < got.size(); ++i) EXPECT_NEAR(got[i], via2d[i], 1e-12);
}

TEST(NnConv, LargeBatchChunkingMatchesSmall) {
  // Big enough that im2col is chunked over images.
  const Tensor x = rnd({40, 32, 32, 64}, 71, DType::Float32), f = rnd({3, 3, 64, 8}, 72, DType::Float32);
  const Tensor y = nn::conv2d(x, f);
  const Tensor y0 = nn::conv2d(nn::slice(x, {39, 0, 0, 0}, {1, -1, -1, -1}), f);
  const auto   a = nn::slice(y, {39, 0, 0, 0}, {1, -1, -1, -1}).to_vector<double>(), b = y0.to_vector<double>();
  for (std::size_t i = 0; i < a.size(); ++i) ASSERT_NEAR(a[i], b[i], 1e-4);
}

// ---- pooling ----------------------------------------------------------------------------------

TEST(NnPool, MaxAndAvgValues) {
  // 1x4x4x1 ramp 0..15
  std::vector<double> v(16);
  for (int i = 0; i < 16; ++i) v[static_cast<std::size_t>(i)] = i;
  const Tensor x = nn::from_vector(v, {1, 4, 4, 1});
  EXPECT_EQ(nn::max_pool2d(x, 2, 2).to_vector<double>(), (std::vector<double>{5, 7, 13, 15}));
  EXPECT_EQ(nn::avg_pool2d(x, 2, 2).to_vector<double>(), (std::vector<double>{2.5, 4.5, 10.5, 12.5}));
  // SAME 3x3 stride 2 on 4x4: out 2x2, pad total 1 -> before 0, after 1; edge windows average only valid entries.
  const auto avg = nn::avg_pool2d(x, 3, 2, Padding::Same).to_vector<double>();
  EXPECT_NEAR(avg[0], (0 + 1 + 2 + 4 + 5 + 6 + 8 + 9 + 10) / 9.0, 1e-12);
  EXPECT_NEAR(avg[3], (10 + 11 + 14 + 15) / 4.0, 1e-12);
  EXPECT_EQ(nn::max_pool2d(x, 3, 2, Padding::Same).to_vector<double>(), (std::vector<double>{10, 11, 14, 15}));
  // negative inputs with SAME padding: padding must not win the max
  const Tensor neg = x * -1.0 - 1.0;
  EXPECT_EQ(nn::max_pool2d(neg, 3, 2, Padding::Same).to_vector<double>(), (std::vector<double>{-1, -3, -9, -11}));
  EXPECT_EQ(nn::global_avg_pool2d(x).shape(), (nn::Shape{1, 1}));
  EXPECT_NEAR(nn::global_avg_pool2d(x).item<double>(), 7.5, 1e-12);
  EXPECT_NEAR(nn::global_max_pool2d(x, true).item<double>(), 15, 1e-12);
}

TEST(NnPool, Gradients) {
  const Tensor x = rnd({2, 5, 6, 3}, 81);
  for (auto pad : {Padding::Valid, Padding::Same}) {
    auto r = nn::gradcheck([&](const Inputs& in) { return nn::max_pool2d(in[0], {3, 2}, {2, 2}, pad); }, {leaf(x)});
    EXPECT_TRUE(r.ok) << r.message;
    r = nn::gradcheck([&](const Inputs& in) { return nn::avg_pool2d(in[0], {3, 2}, {2, 1}, pad); }, {leaf(x)});
    EXPECT_TRUE(r.ok) << r.message;
    r = nn::gradgradcheck([&](const Inputs& in) { return nn::square(nn::avg_pool2d(in[0], 2, 2, pad)); }, {leaf(x)});
    EXPECT_TRUE(r.ok) << r.message;
    r = nn::gradgradcheck([&](const Inputs& in) { return nn::square(nn::max_pool2d(in[0], 2, 2, pad)); }, {leaf(x)});
    EXPECT_TRUE(r.ok) << r.message;
  }
  // Ties route the gradient to the first maximum.
  Tensor ones = nn::ones({1, 2, 2, 1}, DType::Float64);
  ones.requires_grad_(true);
  nn::sum(nn::max_pool2d(ones, 2, 2)).backward();
  EXPECT_EQ(ones.grad().to_vector<double>(), (std::vector<double>{1, 0, 0, 0}));
  const Tensor seq = rnd({2, 7, 3}, 82);
  EXPECT_EQ(nn::max_pool1d(seq, 2, 2).shape(), (nn::Shape{2, 3, 3}));
  EXPECT_EQ(nn::avg_pool1d(seq, 3, 1, Padding::Same).shape(), (nn::Shape{2, 7, 3}));
}

// ---- batch norm -------------------------------------------------------------------------------

TEST(NnBatchNorm, TrainMatchesComposition) {
  const Tensor x = rnd({4, 3, 3, 5}, 91), gamma = rnd({5}, 92), beta = rnd({5}, 93);
  const auto   r = nn::batch_norm_train(x, gamma, beta, 1e-3);
  const Tensor mu = nn::mean(x, {0, 1, 2}), var = nn::variance(x, {0, 1, 2});
  const Tensor ref = (x - mu) * nn::rsqrt(var + 1e-3) * gamma + beta;
  const auto   a = r.y.to_vector<double>(), b = ref.to_vector<double>();
  for (std::size_t i = 0; i < a.size(); ++i) ASSERT_NEAR(a[i], b[i], 1e-10);
  const auto m = r.mean.to_vector<double>(), mv = mu.to_vector<double>();
  const auto v = r.variance.to_vector<double>(), vv = var.to_vector<double>();
  for (std::size_t k = 0; k < 5; ++k) {
    EXPECT_NEAR(m[k], mv[k], 1e-12);
    EXPECT_NEAR(v[k], vv[k] * 36.0 / 35.0, 1e-12);  // unbiased
  }
  const Tensor inf = nn::batch_norm_infer(x, gamma, beta, mu, var, 1e-3);
  const auto   c   = inf.to_vector<double>();
  for (std::size_t i = 0; i < a.size(); ++i) ASSERT_NEAR(a[i], c[i], 1e-10);
}

TEST(NnBatchNorm, Gradients) {
  const Tensor x = rnd({3, 2, 2, 4}, 101), gamma = rnd({4}, 102), beta = rnd({4}, 103);
  const auto   f = [](const Inputs& in) { return nn::batch_norm_train(in[0], in[1], in[2], 1e-3).y; };
  auto         r = nn::gradcheck(f, {leaf(x), leaf(gamma), leaf(beta)});
  EXPECT_TRUE(r.ok) << r.message;
  r = nn::gradgradcheck(f, {leaf(x), leaf(gamma), leaf(beta)});
  EXPECT_TRUE(r.ok) << r.message;
  // 2-D input [batch, features]
  r = nn::gradcheck(f, {leaf(rnd({6, 3}, 104)), leaf(rnd({3}, 105)), leaf(rnd({3}, 106))});
  EXPECT_TRUE(r.ok) << r.message;
}

// ---- layers -----------------------------------------------------------------------------------

TEST(NnConvLayers, ShapesAndReuse) {
  nn::ParameterStore p(7);
  const Tensor       x = rnd({2, 8, 8, 3}, 111, DType::Float32);
  nn::ConvLayerOptions o;
  o.activation = nn::Activation::Relu;
  const Tensor y = nn::conv2d(p, "c1", x, 5, 3, o);
  EXPECT_EQ(y.shape(), (nn::Shape{2, 8, 8, 5}));
  EXPECT_EQ(p.at("c1/W").shape(), (nn::Shape{3, 3, 3, 5}));
  nn::conv2d(p, "c1", x, 5, 3, o);  // reuse
  EXPECT_EQ(p.size(), 2u);
  o.stride  = 2;
  o.padding = Padding::Valid;
  EXPECT_EQ(nn::conv2d(p, "c2", x, 4, 3, o).shape(), (nn::Shape{2, 3, 3, 4}));
  nn::ConvLayerOptions up;
  up.stride = 2;
  EXPECT_EQ(nn::conv2d_transpose(p, "up", x, 6, 4, up).shape(), (nn::Shape{2, 16, 16, 6}));
  EXPECT_EQ(p.at("up/W").shape(), (nn::Shape{4, 4, 6, 3}));
  EXPECT_EQ(nn::depthwise_conv2d(p, "dw", x, 2, 3).shape(), (nn::Shape{2, 8, 8, 6}));
  EXPECT_EQ(nn::separable_conv2d(p, "sep", x, 7, 3).shape(), (nn::Shape{2, 8, 8, 7}));
  EXPECT_TRUE(p.contains("sep/depthwise/W") && p.contains("sep/pointwise/W") && p.contains("sep/pointwise/b"));
  EXPECT_EQ(nn::conv1d(p, "c1d", rnd({2, 10, 3}, 112, DType::Float32), 4, 3).shape(), (nn::Shape{2, 10, 4}));
}

TEST(NnConvLayers, BatchNormMovingAverages) {
  nn::ParameterStore p(1), stats(2);
  const Tensor       x = rnd({16, 4}, 121, DType::Float32) * 3.0 + 5.0;
  nn::BatchNormOptions o;
  o.momentum   = 0.9;
  const Tensor y = nn::batch_norm(p, stats, "bn", x, true, o);
  const auto   bm = nn::mean(x, {0}).to_vector<double>();
  const auto   bv = (nn::variance(x, {0}) * (16.0 / 15.0)).to_vector<double>();
  const auto   mm = stats.at("bn/moving_mean").to_vector<double>();
  const auto   mv = stats.at("bn/moving_var").to_vector<double>();
  for (std::size_t k = 0; k < 4; ++k) {
    EXPECT_NEAR(mm[k], 0.1 * bm[k], 1e-4);
    EXPECT_NEAR(mv[k], 0.9 + 0.1 * bv[k], 1e-3);
  }
  EXPECT_NEAR(nn::mean(y).item<double>(), 0.0, 1e-5);
  EXPECT_EQ(p.size(), 2u);
  const Tensor yi = nn::batch_norm(p, stats, "bn", x, false, o);
  EXPECT_EQ(yi.shape(), x.shape());
  // Gradients reach gamma and beta.
  nn::sum(nn::square(nn::batch_norm(p, stats, "bn", x, true, o))).backward();
  EXPECT_TRUE(p.at("bn/gamma").grad().defined());
}

TEST(NnConvLayers, GroupNormAndNoisyDense) {
  nn::ParameterStore p(3);
  const Tensor       x = rnd({2, 3, 3, 4}, 131);
  const Tensor       g = nn::group_norm(p, "gn", nn::cast(x, DType::Float32), 2);
  // Each (example, group) is standardized.
  const Tensor grouped = nn::reshape(g, {2, 3, 3, 2, 2});
  for (double m : nn::mean(grouped, {1, 2, 4}).to_vector<double>()) EXPECT_NEAR(m, 0.0, 1e-5);
  EXPECT_EQ(nn::instance_norm(p, "in", nn::cast(x, DType::Float32)).shape(), x.shape());

  nn::Generator noise(5);
  const Tensor  in = rnd({3, 6}, 132, DType::Float32);
  const Tensor  a  = nn::noisy_dense(p, "nd", in, 4, noise, 0.0);
  const Tensor  b  = nn::noisy_dense(p, "nd", in, 4, noise, 0.0);
  EXPECT_EQ(a.to_vector<double>(), b.to_vector<double>());  // scale 0 = mean weights
  const Tensor c = nn::noisy_dense(p, "nd", in, 4, noise, 1.0);
  EXPECT_NE(a.to_vector<double>(), c.to_vector<double>());
  nn::sum(c).backward();
  EXPECT_TRUE(p.at("nd/sigma_W").grad().defined());
}

// ---- direct depthwise kernels ---------------------------------------------------------------

TEST(NnConv, DepthwiseKernelMatchesPerChannelConv) {
  struct Case {
    nn::Shape         x, f;
    nn::Conv2DOptions o;
  };
  nn::Conv2DOptions dil = nn::conv_options(1, Padding::Same, 2);
  nn::Conv2DOptions ex  = nn::conv_options(2, Padding::Explicit);
  ex.explicit_paddings  = {1, 0, 2, 1};
  nn::Conv2DOptions rect;
  rect.strides = {2, 1};
  rect.padding = Padding::Valid;
  const std::vector<Case> cases{{{2, 7, 6, 3}, {3, 3, 3, 1}, nn::conv_options(1)},
                                {{2, 7, 6, 3}, {3, 3, 3, 2}, nn::conv_options(2)},
                                {{1, 9, 8, 4}, {3, 3, 4, 1}, dil},
                                {{2, 6, 7, 2}, {2, 3, 2, 3}, ex},
                                {{1, 8, 7, 5}, {3, 2, 5, 2}, rect},
                                {{3, 5, 5, 20}, {3, 3, 20, 1}, nn::conv_options(1, Padding::Valid)}};  // > one channel block
  for (const auto& cs : cases) {
    const Tensor x = rnd(cs.x, 141), f = rnd(cs.f, 142);
    const Tensor y = nn::depthwise_conv2d(x, f, cs.o);
    const auto   C = cs.x[3], M = cs.f[3];
    for (std::int64_t c = 0; c < C; ++c)
      for (std::int64_t m = 0; m < M; ++m) {
        const auto ref = nn::conv2d(nn::slice(x, {0, 0, 0, c}, {-1, -1, -1, 1}),
                                    nn::slice(f, {0, 0, c, m}, {-1, -1, 1, 1}), cs.o)
                             .to_vector<double>();
        const auto got = nn::slice(y, {0, 0, 0, c * M + m}, {-1, -1, -1, 1}).to_vector<double>();
        ASSERT_EQ(got.size(), ref.size());
        for (std::size_t i = 0; i < ref.size(); ++i) ASSERT_NEAR(got[i], ref[i], 1e-11) << nn::shape_str(cs.x);
      }
    const auto r = nn::gradcheck([&](const Inputs& in) { return nn::depthwise_conv2d(in[0], in[1], cs.o); },
                                 {leaf(x), leaf(f)});
    EXPECT_TRUE(r.ok) << nn::shape_str(cs.x) << ": " << r.message;
  }
  const auto r2 = nn::gradgradcheck([&](const Inputs& in) { return nn::depthwise_conv2d(in[0], in[1], nn::conv_options(2)); },
                                    {leaf(rnd({1, 5, 6, 3}, 143)), leaf(rnd({3, 3, 3, 2}, 144))});
  EXPECT_TRUE(r2.ok) << r2.message;
}

TEST(NnConv, DepthwiseKernelIsDeterministicAcrossThreadCounts) {
  const Tensor x = rnd({4, 16, 16, 40}, 151, DType::Float32), f = rnd({3, 3, 40, 1}, 152, DType::Float32);
  const Tensor g = rnd({4, 16, 16, 40}, 153, DType::Float32);
  const int    before = nn::get_num_threads();
  std::vector<std::vector<double>> runs;
  for (int threads : {1, 3, 8}) {
    nn::set_num_threads(threads);
    runs.push_back(nn::depthwise_conv2d(x, f).to_vector<double>());
    auto dx = nn::conv2d_backprop_input(x.shape(), nn::reshape(f, {3, 3, 1, 40}), g, [] {
      nn::Conv2DOptions o;
      o.groups = 40;
      return o;
    }());
    runs.push_back(dx.to_vector<double>());
  }
  nn::set_num_threads(before);
  for (std::size_t k = 2; k < runs.size(); ++k) EXPECT_EQ(runs[k], runs[k % 2]) << "run " << k;
}

TEST(NnParallel, ConcurrentSubmittersWaitForTheirOwnJobs) {
  const int before = nn::get_num_threads();
  nn::set_num_threads(4);
  std::atomic<int> failures{0};
  std::vector<std::thread> callers;
  for (int user = 0; user < 4; ++user) {
    callers.emplace_back([&] {
      for (int rep = 0; rep < 250; ++rep) {
        std::vector<int> hits(257, 0);
        nn::parallel_for(0, 257, 3, [&](std::int64_t lo, std::int64_t hi) {
          for (auto i = lo; i < hi; ++i) ++hits[static_cast<std::size_t>(i)];
        });
        for (int hit : hits) if (hit != 1) ++failures;
      }
    });
  }
  for (auto& caller : callers) caller.join();
  nn::set_num_threads(before);
  EXPECT_EQ(failures.load(), 0);
}

TEST(NnBatchNorm, ParallelPartialsMatchSerialAndRepeatDeterministically) {
  const int original_threads = nn::get_num_threads();
  struct RestoreThreads {
    int count;
    ~RestoreThreads() { nn::set_num_threads(count); }
  } restore{original_threads};
  nn::NoGradGuard guard;
  for (const auto channels : {1, 3, 129}) {
    const std::int64_t rows = channels == 129 ? 1027 : 20003;
    std::vector<double> data(static_cast<std::size_t>(rows * channels));
    for (std::size_t i = 0; i < data.size(); ++i) data[i] = (static_cast<int>(i % 251) - 125) / 128.0;
    auto x = nn::from_vector(data, {rows, channels});
    auto gamma = nn::ones({channels}, DType::Float64), beta = nn::zeros({channels}, DType::Float64);
    nn::set_num_threads(1);
    const auto serial = nn::batch_norm_train(x, gamma, beta);
    nn::set_num_threads(4);
    const auto parallel = nn::batch_norm_train(x, gamma, beta);
    const auto repeat = nn::batch_norm_train(x, gamma, beta);
    EXPECT_EQ(parallel.mean.to_vector<double>(), repeat.mean.to_vector<double>());
    EXPECT_EQ(parallel.variance.to_vector<double>(), repeat.variance.to_vector<double>());
    EXPECT_EQ(parallel.y.to_vector<double>(), repeat.y.to_vector<double>());
    EXPECT_LT(nn::max(nn::abs(nn::sub(serial.y, parallel.y))).item(), 1e-12);
    EXPECT_LT(nn::max(nn::abs(nn::sub(serial.variance, parallel.variance))).item(), 1e-12);
  }
}

namespace {

void expect_close_fused(const Tensor& a, const Tensor& b, double rel) {
  ASSERT_EQ(a.shape(), b.shape());
  const auto av = a.to_vector<double>(), bv = b.to_vector<double>();
  for (std::size_t i = 0; i < av.size(); ++i) EXPECT_NEAR(av[i], bv[i], rel * (1.0 + std::abs(bv[i]))) << "at " << i;
}

}  // namespace

TEST(NnConv, FusedConvBiasReluMatchesCompositionInValuesAndGradients) {
  nn::Generator gen(3);
  struct Case {
    nn::Shape    x;
    std::int64_t k, out, stride;
    nn::Padding  pad;
  };
  // 3x3 same; pointwise; strided valid; many small images (more than the
  // partial count); one image whose rows do not fit one tile (several tiles).
  for (const Case& c : std::vector<Case>{{{2, 9, 7, 5}, 3, 6, 1, nn::Padding::Same},
                                         {{3, 8, 8, 6}, 1, 5, 1, nn::Padding::Same},
                                         {{2, 11, 10, 4}, 3, 7, 2, nn::Padding::Valid},
                                         {{18, 5, 5, 3}, 3, 4, 1, nn::Padding::Same},
                                         {{1, 40, 64, 48}, 3, 8, 1, nn::Padding::Same}}) {
    Tensor x = nn::normal(c.x, 0.0, 1.0, gen), w = nn::normal({c.k, c.k, c.x[3], c.out}, 0.0, 0.3, gen),
           b = nn::normal({c.out}, 0.0, 1.0, gen);
    x.requires_grad_(true);
    w.requires_grad_(true);
    b.requires_grad_(true);
    const nn::Conv2DOptions o     = nn::conv_options(c.stride, c.pad);
    const Tensor            fused = nn::conv2d_bias_relu(x, w, b, o), plain = nn::relu(nn::bias_add(nn::conv2d(x, w, o), b));
    expect_close_fused(fused, plain, 1e-5);
    const Tensor    seed = nn::normal(fused.shape(), 0.0, 1.0, gen);
    nn::GradOptions keep;
    keep.retain_graph = true;
    const auto gf = nn::grad({fused}, {x, w, b}, {seed}, keep), gp = nn::grad({plain}, {x, w, b}, {seed}, keep);
    for (int i = 0; i < 3; ++i) expect_close_fused(gf[i], gp[i], 1e-4);
    // The gradient of a split of a concatenation is a strided view.
    const Tensor other = nn::normal(fused.shape(), 0.0, 1.0, gen);
    const Tensor cat_f = nn::concat({fused, other}, 3), cat_p = nn::concat({plain, other}, 3);
    const Tensor seed2 = nn::normal(cat_f.shape(), 0.0, 1.0, gen);
    const auto   hf = nn::grad({cat_f}, {x, w, b}, {seed2}, keep), hp = nn::grad({cat_p}, {x, w, b}, {seed2}, keep);
    for (int i = 0; i < 3; ++i) expect_close_fused(hf[i], hp[i], 1e-4);
    // Recording the backward pass takes the differentiable composition.
    Tensor s3 = nn::normal(fused.shape(), 0.0, 1.0, gen);
    s3.requires_grad_(true);
    nn::GradOptions create;
    create.create_graph = true;
    const auto   df = nn::grad({fused}, {x, w, b}, {s3}, create), dp = nn::grad({plain}, {x, w, b}, {s3}, create);
    const Tensor u0 = nn::normal(x.shape(), 0.0, 1.0, gen), u1 = nn::normal(w.shape(), 0.0, 1.0, gen),
                 u2 = nn::normal(b.shape(), 0.0, 1.0, gen);
    const Tensor of = nn::sum(nn::mul(df[0], u0)) + nn::sum(nn::mul(df[1], u1)) + nn::sum(nn::mul(df[2], u2)),
                 op = nn::sum(nn::mul(dp[0], u0)) + nn::sum(nn::mul(dp[1], u1)) + nn::sum(nn::mul(dp[2], u2));
    expect_close_fused(nn::grad({of}, {s3})[0], nn::grad({op}, {s3})[0], 1e-4);
  }
  // Shapes the fused kernels do not take fall back to the composition.
  Tensor x = nn::normal({2, 6, 6, 4}, 0.0, 1.0, gen), w = nn::normal({3, 3, 2, 6}, 0.0, 0.3, gen),
         b = nn::normal({6}, 0.0, 1.0, gen);
  nn::Conv2DOptions grouped = nn::conv_options(1, nn::Padding::Same);
  grouped.groups            = 2;
  expect_close_fused(nn::conv2d_bias_relu(x, w, b, grouped), nn::relu(nn::bias_add(nn::conv2d(x, w, grouped), b)), 1e-5);
  EXPECT_THROW((void)nn::conv2d_bias_relu(x, w, nn::normal({5}, 0.0, 1.0, gen)), std::invalid_argument);
}

TEST(NnConv, FusedConvBiasReluConcatMatchesCompositionInValuesAndGradients) {
  nn::Generator gen(4);
  struct Branch {
    std::int64_t k, out, stride;
    nn::Padding  pad;
  };
  struct Case {
    nn::Shape           x;
    std::vector<Branch> branches;
  };
  // Fire expand (1x1 + SAME 3x3); three branches; strided VALID branches of
  // equal output size; one image too large for one tile (tiled 3x3 path).
  for (const Case& c : std::vector<Case>{
           {{2, 7, 6, 5}, {{1, 4, 1, nn::Padding::Same}, {3, 6, 1, nn::Padding::Same}}},
           {{3, 5, 5, 3}, {{3, 2, 1, nn::Padding::Same}, {1, 5, 1, nn::Padding::Same}, {3, 3, 1, nn::Padding::Same}}},
           {{2, 11, 10, 4}, {{3, 3, 2, nn::Padding::Valid}, {3, 5, 2, nn::Padding::Valid}}},
           {{1, 40, 64, 16}, {{1, 8, 1, nn::Padding::Same}, {3, 8, 1, nn::Padding::Same}}}}) {
    Tensor x = nn::normal(c.x, 0.0, 1.0, gen);
    x.requires_grad_(true);
    std::vector<Tensor>            ws, bs, leaves{x};
    std::vector<nn::Conv2DOptions> os;
    for (const Branch& br : c.branches) {
      ws.push_back(nn::normal({br.k, br.k, c.x[3], br.out}, 0.0, 0.3, gen).requires_grad_(true));
      bs.push_back(nn::normal({br.out}, 0.0, 1.0, gen).requires_grad_(true));
      os.push_back(nn::conv_options(br.stride, br.pad));
    }
    leaves.insert(leaves.end(), ws.begin(), ws.end());
    leaves.insert(leaves.end(), bs.begin(), bs.end());
    std::vector<Tensor> parts;
    for (std::size_t i = 0; i < ws.size(); ++i) parts.push_back(nn::relu(nn::bias_add(nn::conv2d(x, ws[i], os[i]), bs[i])));
    const Tensor fused = nn::conv2d_bias_relu_concat(x, ws, bs, os), plain = nn::concat(parts, 3);
    expect_close_fused(fused, plain, 1e-5);
    nn::GradOptions keep;
    keep.retain_graph = true;
    const Tensor seed = nn::normal(fused.shape(), 0.0, 1.0, gen);
    const auto   gf = nn::grad({fused}, leaves, {seed}, keep), gp = nn::grad({plain}, leaves, {seed}, keep);
    for (std::size_t i = 0; i < leaves.size(); ++i) expect_close_fused(gf[i], gp[i], 1e-4);
    // A gradient arriving as a strided view (the output is concatenated again).
    const Tensor other = nn::normal(fused.shape(), 0.0, 1.0, gen);
    const Tensor cat_f = nn::concat({other, fused}, 3), cat_p = nn::concat({other, plain}, 3);
    const Tensor seed2 = nn::normal(cat_f.shape(), 0.0, 1.0, gen);
    const auto   hf = nn::grad({cat_f}, leaves, {seed2}, keep), hp = nn::grad({cat_p}, leaves, {seed2}, keep);
    for (std::size_t i = 0; i < leaves.size(); ++i) expect_close_fused(hf[i], hp[i], 1e-4);
    // Recording the backward pass takes the differentiable composition.
    Tensor s3 = nn::normal(fused.shape(), 0.0, 1.0, gen);
    s3.requires_grad_(true);
    nn::GradOptions create;
    create.create_graph = true;
    const auto df = nn::grad({fused}, leaves, {s3}, create), dp = nn::grad({plain}, leaves, {s3}, create);
    Tensor     of = nn::zeros({}), op = nn::zeros({});
    for (std::size_t i = 0; i < leaves.size(); ++i) {
      const Tensor u = nn::normal(leaves[i].shape(), 0.0, 1.0, gen);
      of             = of + nn::sum(nn::mul(df[i], u));
      op             = op + nn::sum(nn::mul(dp[i], u));
    }
    expect_close_fused(nn::grad({of}, {s3})[0], nn::grad({op}, {s3})[0], 1e-4);
  }
  // Grouped branches use the composition; mismatched lists are rejected.
  Tensor            x = nn::normal({2, 6, 6, 4}, 0.0, 1.0, gen), w = nn::normal({3, 3, 2, 6}, 0.0, 0.3, gen),
         b = nn::normal({6}, 0.0, 1.0, gen);
  nn::Conv2DOptions grouped = nn::conv_options(1, nn::Padding::Same);
  grouped.groups            = 2;
  expect_close_fused(nn::conv2d_bias_relu_concat(x, {w, w}, {b, b}, {grouped, grouped}),
                     nn::concat({nn::conv2d_bias_relu(x, w, b, grouped), nn::conv2d_bias_relu(x, w, b, grouped)}, 3),
                     1e-5);
  EXPECT_THROW((void)nn::conv2d_bias_relu_concat(x, {w}, {b, b}, {grouped}), std::invalid_argument);
  EXPECT_THROW((void)nn::conv2d_bias_relu_concat(x, {}, {}, {}), std::invalid_argument);
}

TEST(NnConv, MaxPoolTapPathMatchesTheGlobalArgmaxPath) {
  nn::Generator gen(12);
  struct Case {
    nn::Shape    x;
    std::int64_t window, stride;
    nn::Padding  pad;
  };
  // Ties (rounded values), SAME and VALID, overlapping and disjoint windows,
  // channel counts below and across the 64-channel block.
  for (const Case& c : std::vector<Case>{{{2, 8, 8, 5}, 3, 2, nn::Padding::Same},
                                         {{3, 9, 7, 70}, 3, 2, nn::Padding::Valid},
                                         {{2, 6, 6, 130}, 2, 2, nn::Padding::Valid},
                                         {{1, 7, 5, 3}, 3, 1, nn::Padding::Same},
                                         {{2, 5, 5, 4}, 4, 3, nn::Padding::Same}}) {
    Tensor x = nn::round(nn::mul(nn::normal(c.x, 0.0, 1.0, gen), 2.0));
    x.requires_grad_(true);
    const Tensor fast = nn::max_pool2d(x, c.window, c.stride, c.pad);
    const Tensor old  = nn::max_pool2d_with_argmax(x, {c.window, c.window}, {c.stride, c.stride}, c.pad).output;
    ASSERT_EQ(fast.shape(), old.shape());
    EXPECT_EQ(fast.to_vector<double>(), old.to_vector<double>());
    nn::GradOptions keep;
    keep.retain_graph = true;
    const Tensor seed = nn::normal(fast.shape(), 0.0, 1.0, gen);
    EXPECT_EQ(nn::grad({fast}, {x}, {seed}, keep)[0].to_vector<double>(),
              nn::grad({old}, {x}, {seed}, keep)[0].to_vector<double>());
    // Higher order: the gradient of <dx, u> with respect to the seed.
    Tensor s2 = nn::normal(fast.shape(), 0.0, 1.0, gen);
    s2.requires_grad_(true);
    nn::GradOptions create;
    create.create_graph = true;
    const Tensor u  = nn::normal(c.x, 0.0, 1.0, gen);
    const Tensor hf = nn::sum(nn::mul(nn::grad({fast}, {x}, {s2}, create)[0], u));
    const Tensor ho = nn::sum(nn::mul(nn::grad({old}, {x}, {s2}, create)[0], u));
    EXPECT_EQ(nn::grad({hf}, {s2})[0].to_vector<double>(), nn::grad({ho}, {s2})[0].to_vector<double>());
  }
  // A NaN first tap stays the maximum; a later NaN is ignored.
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const Tensor x   = nn::from_vector<double>({nan, 1, 2, 3, 4, nan, 6, 7, 8}, {1, 3, 3, 1});
  const auto   f = nn::max_pool2d(x, 2, 1, nn::Padding::Valid).to_vector<double>(),
             o = nn::max_pool2d_with_argmax(x, {2, 2}, {1, 1}, nn::Padding::Valid).output.to_vector<double>();
  ASSERT_EQ(f.size(), o.size());
  for (std::size_t i = 0; i < f.size(); ++i) EXPECT_TRUE(f[i] == o[i] || (std::isnan(f[i]) && std::isnan(o[i]))) << i;
  EXPECT_TRUE(std::isnan(f[0]));
}

TEST(NnConv, ConcatLayerCreatesTheSameParametersAsSeparateLayers) {
  nn::ConvLayerOptions o;
  o.init       = nn::init::he_uniform();
  o.activation = nn::Activation::Relu;
  nn::Generator gen(9);
  const Tensor  x = nn::normal({2, 5, 5, 3}, 0.0, 1.0, gen);
  nn::ParameterStore fused_store(17), plain_store(17);
  const Tensor fused = nn::conv2d_concat(fused_store, {"e1", "e3"}, x, {4, 6}, {1, 3}, o);
  const Tensor plain = nn::concat({nn::conv2d(plain_store, "e1", x, 4, 1, o), nn::conv2d(plain_store, "e3", x, 6, 3, o)}, 3);
  EXPECT_EQ(fused_store.names(), plain_store.names());
  for (const auto& name : plain_store.names())
    EXPECT_EQ(fused_store.at(name).to_vector<double>(), plain_store.at(name).to_vector<double>()) << name;
  expect_close_fused(fused, plain, 1e-5);
  // Without ReLU the layer is the composition.
  o.activation = nn::Activation::None;
  nn::ParameterStore linear(17);
  EXPECT_EQ(nn::conv2d_concat(linear, {"e1", "e3"}, x, {4, 6}, {1, 3}, o).shape(), (nn::Shape{2, 5, 5, 10}));
}
