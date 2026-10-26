// Tests for library gaps reported by the tflow example ports (P5): ALiBi
// slopes for any head count, argument guards of the positional encodings and
// multi-head attention, nn::grad with allow_unused and history-less outputs,
// from_scalar, depth_to_space modes, max_pool2d_with_argmax / max_unpool2d,
// local response normalization, Adadelta and the max-norm constraint, mu-law
// companding, in-memory .npz and equal_bits.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <datamunge/nn/nn.hpp>
#include <fstream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
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

/// The ALiBi paper's reference get_slopes (Press et al., attention_with_linear_biases).
std::vector<double> paper_slopes(int n) {
  const auto power_of_2 = [](int m) {
    const double        start = std::pow(2.0, -std::pow(2.0, -(std::log2(m) - 3)));
    std::vector<double> s;
    for (int i = 0; i < m; ++i) s.push_back(start * std::pow(start, i));
    return s;
  };
  if (std::exp2(std::floor(std::log2(n))) == n) return power_of_2(n);
  const int  closest = static_cast<int>(std::exp2(std::floor(std::log2(n))));
  auto       out     = power_of_2(closest);
  const auto extra   = paper_slopes(2 * closest);
  for (int i = 0; static_cast<int>(out.size()) < n; i += 2) out.push_back(extra[static_cast<std::size_t>(i)]);
  return out;
}

}  // namespace

// ---- 1. ALiBi --------------------------------------------------------------------------------

TEST(NnLibGapsAlibi, SlopesMatchPaperForOneToTwelveHeads) {
  for (int h = 1; h <= 12; ++h) {
    const auto got = nn::alibi_slopes(h), want = paper_slopes(h);
    ASSERT_EQ(got.size(), want.size()) << h;
    for (std::size_t i = 0; i < got.size(); ++i) EXPECT_NEAR(got[i], want[i], 1e-15) << "heads " << h << " slot " << i;
  }
  const auto three = nn::alibi_slopes(3);
  EXPECT_DOUBLE_EQ(three[0], std::pow(2.0, -4));
  EXPECT_DOUBLE_EQ(three[1], std::pow(2.0, -8));
  EXPECT_DOUBLE_EQ(three[2], std::pow(2.0, -2));
  const auto six = nn::alibi_slopes(6);  // P = 4: 2^-2, 2^-4, 2^-6, 2^-8, then 2^-1, 2^-3
  const std::vector<double> six_want = {0.25, 0.0625, 0.015625, 0.00390625, 0.5, 0.125};
  for (std::size_t i = 0; i < 6; ++i) EXPECT_DOUBLE_EQ(six[i], six_want[i]);
}

TEST(NnLibGapsAlibi, BiasUsesTheSameSlopes) {
  const auto   slopes = nn::alibi_slopes(5);
  const Tensor b      = nn::alibi_bias(5, 4, false, DType::Float64);
  for (int h = 0; h < 5; ++h)
    EXPECT_DOUBLE_EQ(b.to_vector<double>()[static_cast<std::size_t>((h * 4 + 3) * 4 + 0)], -3 * slopes[static_cast<std::size_t>(h)]);
}

// ---- 2. argument guards ----------------------------------------------------------------------------

TEST(NnLibGapsGuards, PositionalAndAttentionArguments) {
  EXPECT_THROW(nn::sinusoidal_encoding(4, 0), std::invalid_argument);
  EXPECT_THROW(nn::sinusoidal_encoding(4, 5), std::invalid_argument);
  EXPECT_THROW(nn::sinusoidal_encoding(-1, 4), std::invalid_argument);
  EXPECT_THROW(nn::sinusoidal_encoding(4, 4, 0.0), std::invalid_argument);
  EXPECT_NO_THROW(nn::sinusoidal_encoding(4, 4));
  EXPECT_THROW(nn::apply_rotary(randn({1, 3, 4}, 1), -1), std::invalid_argument);
  EXPECT_THROW(nn::rotary_tables(3, 4, 10000.0, DType::Float32, -2), std::invalid_argument);
  EXPECT_THROW(nn::apply_rotary(randn({1, 3, 4}, 1), 0, -5.0), std::invalid_argument);
  EXPECT_NO_THROW(nn::apply_rotary(randn({1, 3, 4}, 1), 3));
  EXPECT_THROW(nn::alibi_slopes(0), std::invalid_argument);
  EXPECT_THROW(nn::alibi_slopes(-3), std::invalid_argument);
  EXPECT_THROW(nn::alibi_bias(0, 4), std::invalid_argument);
  EXPECT_THROW(nn::alibi_bias(2, -1, 3, true), std::invalid_argument);
  EXPECT_THROW(nn::relative_position_buckets(3, 3, true, 2, 128), std::invalid_argument);
  EXPECT_THROW(nn::relative_position_buckets(3, 3, true, 7, 128), std::invalid_argument);
  EXPECT_THROW(nn::relative_position_buckets(3, 3, true, 32, 8), std::invalid_argument);  // exact range is 8
  EXPECT_THROW(nn::relative_position_buckets(3, 3, false, 32, 16), std::invalid_argument);
  EXPECT_NO_THROW(nn::relative_position_buckets(3, 3, true, 32, 9));
  EXPECT_NO_THROW(nn::relative_position_buckets(3, 3, false, 7, 4));
  nn::ParameterStore store(1);
  EXPECT_THROW(nn::multi_head_attention(store, "mha", randn({1, 2, 0}, 2), randn({1, 2, 0}, 2), 2),
               std::invalid_argument);
  EXPECT_THROW(nn::multi_head_attention(store, "mha", randn({1, 2, 6}, 2), randn({1, 2, 6}, 2), 4),
               std::invalid_argument);
  EXPECT_THROW(nn::multi_head_attention(store, "mha", randn({1, 2, 6}, 2), randn({1, 2, 6}, 2), 0),
               std::invalid_argument);
}

// ---- 3. grad with allow_unused ----------------------------------------------------------------------

TEST(NnLibGapsGrad, AllowUnusedToleratesOutputsWithoutHistory) {
  Tensor       x        = nn::tensor({1.0, 2.0}, {}, DType::Float64).requires_grad_();
  const Tensor constant = nn::scalar(3.0, DType::Float64);
  nn::GradOptions o;
  o.allow_unused = true;
  const auto none = nn::grad({constant}, {x}, {}, o);
  ASSERT_EQ(none.size(), 1u);
  EXPECT_FALSE(none[0].defined());
  EXPECT_FALSE(nn::grad(Tensor(), {x}, o)[0].defined());
  // Mixed: the history-less output (and its seed) is ignored.
  const Tensor y     = nn::sum(nn::square(x));
  const auto   mixed = nn::grad({constant, y}, {x}, {nn::scalar(5.0, DType::Float64), nn::scalar(1.0, DType::Float64)}, o);
  ASSERT_TRUE(mixed[0].defined());
  EXPECT_EQ(mixed[0].to_vector<double>(), (std::vector<double>{2.0, 4.0}));
  // Without allow_unused it still throws.
  EXPECT_THROW(nn::grad({constant}, {x}), std::invalid_argument);
  EXPECT_THROW(nn::grad({constant, nn::sum(x)}, {x}), std::invalid_argument);
  // Seeds must still align with the outputs.
  EXPECT_THROW(nn::grad({constant, nn::sum(x)}, {x}, {nn::scalar(1.0, DType::Float64)}, o), std::invalid_argument);
}

// ---- 4. 0-d tensors from values ------------------------------------------------------------------

TEST(NnLibGapsScalar, FromScalarIsZeroDimensional) {
  const Tensor f = nn::from_scalar(2.5f);
  EXPECT_EQ(f.dim(), 0);
  EXPECT_EQ(f.dtype(), DType::Float32);
  EXPECT_EQ(f.item<float>(), 2.5f);
  const Tensor big = nn::from_scalar(std::int64_t{9007199254740993});  // not representable as a double
  EXPECT_EQ(big.dtype(), DType::Int64);
  EXPECT_EQ(big.data_ptr<std::int64_t>()[0], std::int64_t{9007199254740993});
  EXPECT_EQ(nn::from_scalar(true).dtype(), DType::Bool);
  // Documented: from_vector with an empty shape is 1-D even for one value.
  EXPECT_EQ(nn::from_vector(std::vector<double>{1.0}).shape(), nn::Shape{1});
  EXPECT_EQ(nn::from_vector(std::vector<double>{1.0}, {}).shape(), nn::Shape{1});
}

// ---- 5. depth_to_space / space_to_depth modes ------------------------------------------------------

TEST(NnLibGapsDepthToSpace, CrdLayoutAndRoundTrip) {
  const std::int64_t N = 2, H = 2, W = 3, C = 2, b = 2;
  const Tensor x = nn::arange(0, static_cast<double>(N * H * W * C * b * b), 1, DType::Float64).reshape({N, H, W, C * b * b});
  const Tensor crd = nn::depth_to_space(x, b, nn::DepthToSpaceMode::CRD);
  const Tensor dcr = nn::depth_to_space(x, b, nn::DepthToSpaceMode::DCR);
  ASSERT_EQ(crd.shape(), (nn::Shape{N, H * b, W * b, C}));
  EXPECT_TRUE(nn::equal_bits(dcr, nn::depth_to_space(x, b)));  // default unchanged (DCR)
  const auto xv = x.to_vector<double>(), cv = crd.to_vector<double>(), dv = dcr.to_vector<double>();
  for (std::int64_t n = 0; n < N; ++n)
    for (std::int64_t y = 0; y < H * b; ++y)
      for (std::int64_t xx = 0; xx < W * b; ++xx)
        for (std::int64_t c = 0; c < C; ++c) {
          const std::int64_t dy = y % b, dx = xx % b, base = ((n * H + y / b) * W + xx / b) * C * b * b;
          const auto         o  = static_cast<std::size_t>(((n * H * b + y) * W * b + xx) * C + c);
          EXPECT_EQ(cv[o], xv[static_cast<std::size_t>(base + c * b * b + dy * b + dx)]);
          EXPECT_EQ(dv[o], xv[static_cast<std::size_t>(base + (dy * b + dx) * C + c)]);
        }
  for (auto mode : {nn::DepthToSpaceMode::CRD, nn::DepthToSpaceMode::DCR})
    EXPECT_TRUE(nn::equal_bits(nn::space_to_depth(nn::depth_to_space(x, b, mode), b, mode), x));
  EXPECT_FALSE(nn::equal_bits(nn::space_to_depth(crd, b, nn::DepthToSpaceMode::DCR), x));
}

TEST(NnLibGapsDepthToSpace, CrdGradients) {
  const Tensor x = randn({1, 2, 2, 8}, 5).requires_grad_();
  const auto   f = [](const Inputs& in) { return nn::depth_to_space(in[0], 2, nn::DepthToSpaceMode::CRD); };
  EXPECT_TRUE(nn::gradcheck(f, {x}).ok);
  EXPECT_TRUE(nn::gradgradcheck(f, {x}).ok);
  const Tensor z = randn({1, 4, 4, 2}, 6).requires_grad_();
  const auto   g = [](const Inputs& in) { return nn::square(nn::space_to_depth(in[0], 2, nn::DepthToSpaceMode::CRD)); };
  EXPECT_TRUE(nn::gradcheck(g, {z}).ok);
  EXPECT_TRUE(nn::gradgradcheck(g, {z}).ok);
}

// ---- 6. max_pool2d_with_argmax / max_unpool2d --------------------------------------------------------

TEST(NnLibGapsMaxPoolArgmax, ValuesAndTfIndices) {
  // [2, 2, 3, 2]: distinct values, so every window has a unique maximum.
  const Tensor x  = randn({2, 3, 4, 2}, 11);
  for (auto padding : {nn::Padding::Valid, nn::Padding::Same})
    for (bool with_batch : {false, true}) {
      const auto r = nn::max_pool2d_with_argmax(x, {2, 2}, {2, 1}, padding, with_batch);
      EXPECT_TRUE(nn::equal_bits(r.output, nn::max_pool2d(x, {2, 2}, {2, 1}, padding)));
      ASSERT_EQ(r.argmax.dtype(), DType::Int64);
      ASSERT_EQ(r.argmax.shape(), r.output.shape());
      const auto xv = x.to_vector<double>(), ov = r.output.to_vector<double>();
      const auto av = r.argmax.to_vector<std::int64_t>();
      const std::int64_t per_out = r.output.numel() / 2, per_image = x.numel() / 2;
      for (std::size_t i = 0; i < av.size(); ++i) {
        const std::int64_t image = static_cast<std::int64_t>(i) / per_out;
        const std::int64_t flat  = av[i] + (with_batch ? 0 : image * per_image);
        if (with_batch) EXPECT_GE(av[i], image * per_image);
        else EXPECT_LT(av[i], per_image);
        EXPECT_EQ(xv[static_cast<std::size_t>(flat)], ov[i]);
        EXPECT_EQ(flat % 2, static_cast<std::int64_t>(i) % 2);  // same channel
      }
    }
  // Hand example: [1, 2, 2, 1] -> argmax of 7 at (1, 0) is (1 * 2 + 0) * 1 + 0 = 2.
  const auto small = nn::max_pool2d_with_argmax(nn::tensor({1, 3, 7, 2}, {1, 2, 2, 1}), 2, 2);
  EXPECT_EQ(small.argmax.to_vector<std::int64_t>(), std::vector<std::int64_t>{2});
  EXPECT_EQ(small.output.item(), 7.0);
}

TEST(NnLibGapsMaxPoolArgmax, OutputGradientMatchesMaxPool) {
  const Tensor x = randn({2, 4, 4, 3}, 12).requires_grad_();
  const auto   f = [](const Inputs& in) { return nn::max_pool2d_with_argmax(in[0], {2, 2}, {2, 2}).output; };
  EXPECT_TRUE(nn::gradcheck(f, {x}).ok);
  EXPECT_TRUE(nn::gradgradcheck(f, {x}).ok);
  const Tensor up = randn({2, 2, 2, 3}, 13);
  const auto   g1 = nn::grad({f({x})}, {x}, {up})[0];
  const auto   g2 = nn::grad({nn::max_pool2d(x, {2, 2}, {2, 2})}, {x}, {up})[0];
  EXPECT_TRUE(nn::equal_bits(g1, g2));
}

TEST(NnLibGapsMaxUnpool, ScatterAndGradients) {
  const Tensor x = nn::abs(randn({2, 4, 4, 3}, 14));  // positive, so the re-pooled zeros never win
  for (bool with_batch : {false, true}) {
    const auto   r  = nn::max_pool2d_with_argmax(x, 2, 2, nn::Padding::Valid, with_batch);
    const Tensor up = nn::max_unpool2d(r.output, r.argmax, x.shape(), with_batch);
    ASSERT_EQ(up.shape(), x.shape());
    // Each pooled value lands back at its argmax; everything else is zero.
    EXPECT_TRUE(nn::equal_bits(nn::max_pool2d(up, 2, 2), r.output));
    EXPECT_NEAR(nn::sum(nn::abs(up)).item(), nn::sum(nn::abs(r.output)).item(), 1e-12);
    // Equals the max_pool2d gradient routed with the pooled values as seeds.
    Tensor xr = x.clone().requires_grad_();
    EXPECT_TRUE(nn::equal_bits(up, nn::grad({nn::max_pool2d(xr, 2, 2)}, {xr}, {r.output})[0]));
    const Tensor v = randn(r.output.shape(), 15).requires_grad_();
    const auto   f = [&](const Inputs& in) { return nn::square(nn::max_unpool2d(in[0], r.argmax, x.shape(), with_batch)); };
    EXPECT_TRUE(nn::gradcheck(f, {v}).ok);
    EXPECT_TRUE(nn::gradgradcheck(f, {v}).ok);
  }
  // Overlapping windows: repeated indices add.
  const auto ov = nn::max_pool2d_with_argmax(nn::tensor({0, 0, 0, 0, 9, 0, 0, 0, 0}, {1, 3, 3, 1}), 2, 1);
  EXPECT_EQ(nn::max_unpool2d(ov.output, ov.argmax, {1, 3, 3, 1}).to_vector<double>()[4], 36.0);
  // Indices outside the image (or into another image) are rejected.
  const Tensor bad = nn::tensor({16}, {1, 1, 1, 1}, DType::Int64);
  EXPECT_THROW(nn::max_unpool2d(nn::tensor({1.0}, {1, 1, 1, 1}), bad, {1, 2, 2, 4}), std::invalid_argument);
  EXPECT_THROW(nn::max_unpool2d(nn::tensor({1.0}, {1, 1, 1, 1}), bad, {1, 2, 2}), std::invalid_argument);
  const Tensor other_image = nn::tensor({0, 0}, {2, 1, 1, 1}, DType::Int64);
  EXPECT_THROW(nn::max_unpool2d(nn::tensor({1.0, 2.0}, {2, 1, 1, 1}), other_image, {2, 1, 1, 1}, true),
               std::invalid_argument);
}

// ---- 7. local response normalization ----------------------------------------------------------------

TEST(NnLibGapsLrn, MatchesTfFormulaAndGradients) {
  const Tensor x = randn({2, 3, 2, 7}, 21);
  const std::int64_t r = 2;
  const double bias = 2.0, alpha = 1e-1, beta = 0.75;
  const Tensor y  = nn::local_response_normalization(x, r, bias, alpha, beta);
  const auto   xv = x.to_vector<double>(), yv = y.to_vector<double>();
  const std::int64_t C = 7;
  for (std::size_t i = 0; i < xv.size(); ++i) {
    const std::int64_t c = static_cast<std::int64_t>(i) % C, row = static_cast<std::int64_t>(i) - c;
    double             s = 0;
    for (std::int64_t j = std::max<std::int64_t>(0, c - r); j <= std::min(C - 1, c + r); ++j)
      s += xv[static_cast<std::size_t>(row + j)] * xv[static_cast<std::size_t>(row + j)];
    EXPECT_NEAR(yv[i], xv[i] / std::pow(bias + alpha * s, beta), 1e-13);
  }
  // TF defaults (radius 5 exceeds the channel count here) and float32 input.
  const Tensor xf = nn::cast(x, DType::Float32);
  EXPECT_EQ(nn::local_response_normalization(xf).dtype(), DType::Float32);
  EXPECT_LT(max_abs_diff(nn::local_response_normalization(xf), nn::local_response_normalization(x)), 1e-6);
  const Tensor xs = randn({1, 2, 2, 5}, 22).requires_grad_();
  const auto   f  = [&](const Inputs& in) { return nn::local_response_normalization(in[0], 1, 1.5, 0.3, 0.75); };
  EXPECT_TRUE(nn::gradcheck(f, {xs}).ok);
  EXPECT_TRUE(nn::gradgradcheck(f, {xs}).ok);
  EXPECT_THROW(nn::local_response_normalization(x, -1), std::invalid_argument);
}

// ---- 8. Adadelta and max-norm -------------------------------------------------------------------------

TEST(NnLibGapsOptim, AdadeltaFollowsTfUpdate) {
  Tensor              w = nn::tensor({0.5, -1.0, 2.0}, {}, DType::Float64).requires_grad_();
  nn::OptimizerConfig c;
  c.kind    = nn::OptimizerKind::Adadelta;
  c.rho     = 0.95;
  c.epsilon = 1e-6;
  nn::Optimizer opt({w}, 0.5, c);
  EXPECT_EQ(nn::optimizer_name(nn::OptimizerKind::Adadelta), "Adadelta");
  std::vector<double> ref = {0.5, -1.0, 2.0}, accum(3, 0.0), accum_update(3, 0.0);
  for (int step = 0; step < 4; ++step) {
    const std::vector<double> g = {0.3 * (step + 1), -0.2, 1.5 - step};
    opt.step({nn::tensor({g[0], g[1], g[2]}, {}, DType::Float64)});
    for (std::size_t i = 0; i < 3; ++i) {
      accum[i]           = 0.95 * accum[i] + 0.05 * g[i] * g[i];
      const double d     = std::sqrt(accum_update[i] + 1e-6) / std::sqrt(accum[i] + 1e-6) * g[i];
      accum_update[i]    = 0.95 * accum_update[i] + 0.05 * d * d;
      ref[i]            -= 0.5 * d;
    }
    const auto got = w.to_vector<double>();
    for (std::size_t i = 0; i < 3; ++i) EXPECT_NEAR(got[i], ref[i], 1e-14);
  }
  std::vector<std::string> names;
  for (const auto& entry : opt.state()) names.push_back(entry.first);
  EXPECT_NE(std::find(names.begin(), names.end(), "accum/0"), names.end());
  EXPECT_NE(std::find(names.begin(), names.end(), "accum_update/0"), names.end());
}

TEST(NnLibGapsOptim, MaxNormConstraint) {
  const Tensor w = nn::tensor({{3.0, 0.1}, {4.0, 0.2}}, DType::Float64);  // column norms 5 and ~0.22
  const Tensor c = nn::max_norm_constraint(w, 3.0);
  const auto   v = c.to_vector<double>();
  EXPECT_NEAR(v[0], 1.8, 1e-14);
  EXPECT_NEAR(v[2], 2.4, 1e-14);
  EXPECT_DOUBLE_EQ(v[1], 0.1);
  EXPECT_DOUBLE_EQ(v[3], 0.2);
  const Tensor rows = nn::max_norm_constraint(w, 1.0, {1});  // per row
  EXPECT_NEAR(nn::norm(nn::slice_axis(rows, 0, 0, 1)).item(), 1.0, 1e-14);
  EXPECT_THROW(nn::max_norm_constraint(w, 0.0), std::invalid_argument);
  const Tensor x = randn({3, 2}, 31).requires_grad_();
  EXPECT_TRUE(nn::gradcheck([](const Inputs& in) { return nn::max_norm_constraint(in[0], 0.5); }, {x}).ok);
}

// ---- 9. mu-law --------------------------------------------------------------------------------------------

TEST(NnLibGapsAudio, MuLawCodec) {
  const Tensor codes = nn::mu_law_encode(nn::tensor({-1.0, 0.0, 1.0, -0.5, 0.5, 3.0, -7.0}));
  EXPECT_EQ(codes.dtype(), DType::Int32);
  EXPECT_EQ(codes.to_vector<int>(), (std::vector<int>{0, 128, 255, 16, 239, 255, 0}));
  const Tensor all     = nn::arange(0, 256, 1, DType::Int32);
  const Tensor decoded = nn::mu_law_decode(all, 256, DType::Float64);
  EXPECT_TRUE(nn::equal_bits(nn::mu_law_encode(decoded), all));
  EXPECT_NEAR(decoded.to_vector<double>()[0], -1.0, 1e-12);
  EXPECT_NEAR(decoded.to_vector<double>()[255], 1.0, 1e-12);
  EXPECT_EQ(nn::mu_law_decode(all).dtype(), DType::Float32);
  // Other channel counts: mu = 15.
  EXPECT_EQ(nn::mu_law_encode(nn::tensor({-1.0, 1.0}), 16).to_vector<int>(), (std::vector<int>{0, 15}));
  EXPECT_THROW(nn::mu_law_encode(nn::tensor({0.0}), 1), std::invalid_argument);
  const Tensor q = nn::tensor({10.5, 100.25, 200.0}, {}, DType::Float64).requires_grad_();
  EXPECT_TRUE(nn::gradcheck([](const Inputs& in) { return nn::mu_law_decode(in[0]); }, {q}).ok);
}

// ---- 10. in-memory npz ------------------------------------------------------------------------------------

TEST(NnLibGapsIo, NpzInMemory) {
  const nn::NamedTensors arrays = {{"a", randn({2, 3}, 41)},
                                   {"dir/b", nn::tensor({1, 2, 3}, {}, DType::Int32)},
                                   {"s", nn::from_scalar(7.0)}};
  const auto bytes = nn::encode_npz(arrays);
  const auto back  = nn::decode_npz(bytes);
  ASSERT_EQ(back.size(), 3u);
  for (std::size_t i = 0; i < 3; ++i) {
    EXPECT_EQ(back[i].first, arrays[i].first);
    EXPECT_TRUE(nn::equal_bits(back[i].second, arrays[i].second));
  }
  const std::string path = ::testing::TempDir() + "nn_libgaps.npz";
  nn::save_npz(path, arrays);
  std::ifstream                    in(path, std::ios::binary);
  const std::vector<unsigned char> file((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  EXPECT_EQ(file, bytes);
  EXPECT_THROW(nn::decode_npz(std::vector<unsigned char>(10, 0)), std::runtime_error);
  auto corrupt = bytes;
  corrupt[40] ^= 0xff;  // inside the first member's data: CRC mismatch
  EXPECT_THROW(nn::decode_npz(corrupt), std::runtime_error);
}

// ---- 11. equal_bits -------------------------------------------------------------------------------------

TEST(NnLibGapsEqualBits, DtypeShapeAndBytes) {
  const Tensor a = nn::tensor({1.0, 2.0, 3.0, 4.0}, {2, 2});
  EXPECT_TRUE(nn::equal_bits(a, a.clone()));
  EXPECT_FALSE(nn::equal_bits(a, nn::cast(a, DType::Float64)));
  EXPECT_FALSE(nn::equal_bits(a, a.reshape({4})));
  EXPECT_FALSE(nn::equal_bits(nn::tensor({0.0}), nn::tensor({-0.0})));
  const double nan = std::numeric_limits<double>::quiet_NaN();
  EXPECT_TRUE(nn::equal_bits(nn::tensor({nan}, {}, DType::Float64), nn::tensor({nan}, {}, DType::Float64)));
  EXPECT_TRUE(nn::equal_bits(nn::transpose(a, {1, 0}), nn::tensor({1.0, 3.0, 2.0, 4.0}, {2, 2})));
  EXPECT_FALSE(nn::equal_bits(a, nn::tensor({1.0, 2.0, 3.0, 4.5}, {2, 2})));
  EXPECT_TRUE(nn::equal_bits(Tensor(), Tensor()));
  EXPECT_FALSE(nn::equal_bits(a, Tensor()));
  EXPECT_TRUE(nn::equal_bits(nn::zeros({0, 3}), nn::zeros({0, 3})));
}
