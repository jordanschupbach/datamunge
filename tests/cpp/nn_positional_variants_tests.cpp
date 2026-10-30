/// @file nn_positional_variants_tests.cpp
/// @brief Tests for RoPE frequency scaling, rotation at real positions, 2-D sinusoidal tables and Fourier features.

#include <gtest/gtest.h>

#include <datamunge/nn/nn.hpp>

#include <cmath>
#include <numbers>

namespace nn = datamunge::nn;
using nn::Tensor;

namespace {
const auto F64 = nn::DType::Float64;

double max_abs_diff(const Tensor& a, const Tensor& b) { return nn::max(nn::abs(a - b)).item(); }
}  // namespace

TEST(NnPositionalVariants, RopeFrequencyScaling) {
  const auto t = nn::rope_inverse_frequencies(8, 10000.0);
  ASSERT_EQ(t.size(), 4u);
  EXPECT_DOUBLE_EQ(t[0], 1.0);
  EXPECT_NEAR(t[3], std::pow(10000.0, -6.0 / 8), 1e-15);
  const auto pi = nn::rope_inverse_frequencies(8, 10000.0, nn::RopeScaling::PositionInterpolation, 4.0);
  for (std::size_t j = 0; j < 4; ++j) EXPECT_NEAR(pi[j], t[j] / 4, 1e-15);
  const auto ntk = nn::rope_inverse_frequencies(8, 10000.0, nn::RopeScaling::Ntk, 4.0);
  EXPECT_DOUBLE_EQ(ntk[0], 1.0);
  EXPECT_NEAR(ntk[3], t[3] / 4, 1e-15);
  EXPECT_GT(ntk[1], pi[1]);
  EXPECT_THROW((void)nn::rope_inverse_frequencies(7), std::invalid_argument);
  EXPECT_THROW((void)nn::rope_inverse_frequencies(8, 10000.0, nn::RopeScaling::Ntk, 0.5), std::invalid_argument);
}

TEST(NnPositionalVariants, RotaryAtRealPositions) {
  nn::Generator g(1);
  const Tensor x = nn::normal({2, 5, 8}, 0.0, 1.0, g, F64);
  const auto th = nn::rope_inverse_frequencies(8);
  EXPECT_LT(max_abs_diff(nn::apply_rotary_at(x, {0, 1, 2, 3, 4}, th), nn::apply_rotary(x)), 1e-14);
  EXPECT_LT(max_abs_diff(nn::apply_rotary_at(x, {3, 4, 5, 6, 7}, th), nn::apply_rotary(x, 3)), 1e-14);
  // Scores depend only on the offset, also at fractional positions; norms are preserved.
  const Tensor q = nn::normal({1, 8}, 0.0, 1.0, g, F64), k = nn::normal({1, 8}, 0.0, 1.0, g, F64);
  auto score = [&](double m, double n) { return nn::sum(nn::apply_rotary_at(q, {m}, th) * nn::apply_rotary_at(k, {n}, th)).item(); };
  EXPECT_NEAR(score(2.5, 1.0), score(10.5, 9.0), 1e-12);
  EXPECT_NEAR(nn::sum(nn::square(nn::apply_rotary_at(q, {3.7}, th))).item(), nn::sum(nn::square(q)).item(), 1e-12);
  EXPECT_THROW((void)nn::apply_rotary_at(x, {0, 1}, th), std::invalid_argument);
}

TEST(NnPositionalVariants, TwoDimensionalTableAndFourierFeatures) {
  const Tensor t = nn::sinusoidal_encoding_2d(3, 4, 8);
  ASSERT_EQ(t.size(0), 12);
  const Tensor rows = nn::sinusoidal_encoding(3, 4, 10000.0, F64), cols = nn::sinusoidal_encoding(4, 4, 10000.0, F64);
  for (std::int64_t j = 0; j < 4; ++j) {
    EXPECT_DOUBLE_EQ(t.at({2 * 4 + 1, j}), rows.at({2, j}));
    EXPECT_DOUBLE_EQ(t.at({2 * 4 + 1, 4 + j}), cols.at({1, j}));
  }
  EXPECT_THROW((void)nn::sinusoidal_encoding_2d(3, 4, 6), std::invalid_argument);
  const Tensor x = nn::from_vector(std::vector<double>{0.25, -0.5}, {1, 2});
  const Tensor f = nn::fourier_features(x, 3);
  ASSERT_EQ(f.size(1), 14);
  EXPECT_DOUBLE_EQ(f.at({0, 1}), -0.5);
  EXPECT_NEAR(f.at({0, 2 + 4 + 0}), std::sin(2 * std::numbers::pi * 0.25), 1e-15);  // octave 1, sin, coordinate 0
  EXPECT_NEAR(f.at({0, 2 + 4 + 2 + 1}), std::cos(2 * std::numbers::pi * -0.5), 1e-15);  // octave 1, cos, coordinate 1
  EXPECT_EQ(nn::fourier_features(x, 2, false).size(1), 8);
  // Kernel identity: gamma(x) . gamma(x') = sum_j cos(2 pi b_j . (x - x')).
  nn::Generator g(2);
  const Tensor B = nn::normal({2, 16}, 0.0, 3.0, g, F64);
  const Tensor p = nn::from_vector(std::vector<double>{0.1, 0.2, -0.3, 0.05}, {2, 2});
  const Tensor gp = nn::gaussian_fourier_features(p, B);
  ASSERT_EQ(gp.size(1), 32);
  double k = 0;
  for (std::int64_t j = 0; j < 16; ++j) k += std::cos(2 * std::numbers::pi * (B.at({0, j}) * 0.4 + B.at({1, j}) * 0.15));
  EXPECT_NEAR(nn::sum(nn::slice(gp, {0, 0}, {1, 32}) * nn::slice(gp, {1, 0}, {1, 32})).item(), k, 1e-12);
  EXPECT_THROW((void)nn::gaussian_fourier_features(p, nn::zeros({3, 4}, F64)), std::invalid_argument);
}
