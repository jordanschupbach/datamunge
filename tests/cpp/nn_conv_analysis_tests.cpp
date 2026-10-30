/// @file nn_conv_analysis_tests.cpp
/// @brief Tests for receptive fields, convolution matrices and circular-convolution singular values.

#include <gtest/gtest.h>

#include <datamunge/nn/nn.hpp>

namespace nn = datamunge::nn;
using nn::Tensor;

TEST(NnConvAnalysis, ReceptiveFields) {
  const auto three = nn::receptive_fields({{3}, {3}, {3}});
  EXPECT_EQ(three.back().size, 7);
  const auto dil = nn::receptive_fields({{3, 1, 1}, {3, 1, 2}, {3, 1, 4}});
  EXPECT_EQ(dil.back().size, 15);
  const auto pooled = nn::receptive_fields({{3, 1, 1}, {2, 2, 1}, {3, 1, 1}});
  EXPECT_EQ(pooled[0].size, 3);
  EXPECT_EQ(pooled[1].size, 4);
  EXPECT_EQ(pooled[1].jump, 2);
  EXPECT_EQ(pooled[2].size, 8);
  EXPECT_DOUBLE_EQ(pooled[2].start, 3.5);  // centre of the first unit's 8 inputs, positions 0..7
  EXPECT_THROW((void)nn::receptive_fields({{0}}), std::invalid_argument);
}

TEST(NnConvAnalysis, MatrixMatchesConvolution) {
  nn::Generator g(1);
  const auto F64 = nn::DType::Float64;
  const Tensor x = nn::normal({1, 7, 6, 4}, 0, 1, g, F64);
  struct Case { nn::Conv2DOptions o; std::int64_t cig; };
  std::vector<Case> cases;
  cases.push_back({nn::conv_options(1, nn::Padding::Valid), 4});
  cases.push_back({nn::conv_options(1, nn::Padding::Same), 4});
  cases.push_back({nn::conv_options(2, nn::Padding::Same), 4});
  cases.push_back({nn::conv_options(1, nn::Padding::Same, 2), 4});
  nn::Conv2DOptions grouped = nn::conv_options(1, nn::Padding::Same);
  grouped.groups = 2;
  cases.push_back({grouped, 2});
  nn::Conv2DOptions expl;
  expl.padding = nn::Padding::Explicit;
  expl.explicit_paddings = {1, 0, 2, 1};
  cases.push_back({expl, 4});
  for (const auto& c : cases) {
    const Tensor W = nn::normal({3, 3, c.cig, 6}, 0, 1, g, F64);
    const Tensor y = nn::conv2d(x, W, c.o);
    const Tensor M = nn::conv2d_matrix(7, 6, W, c.o);
    const Tensor My = nn::matmul(M, nn::reshape(x, {-1, 1}));
    ASSERT_EQ(M.size(0), y.numel());
    EXPECT_LT(nn::max(nn::abs(nn::reshape(y, {-1, 1}) - My)).item(), 1e-12);
    // The transpose is the input gradient (the transposed convolution).
    const Tensor r = nn::normal(y.shape(), 0, 1, g, F64);
    const Tensor back = nn::conv2d_backprop_input(x.shape(), W, r, c.o);
    EXPECT_LT(nn::max(nn::abs(nn::reshape(back, {-1, 1}) - nn::matmul(nn::swapaxes(M, 0, 1), nn::reshape(r, {-1, 1})))).item(), 1e-12);
  }
  EXPECT_THROW((void)nn::conv2d_matrix(4, 4, nn::ones({3, 3}, F64)), std::invalid_argument);
}

TEST(NnConvAnalysis, CircularSingularValuesMatchSvd) {
  nn::Generator g(2);
  const auto F64 = nn::DType::Float64;
  const Tensor W = nn::normal({3, 3, 2, 3}, 0, 1, g, F64);
  const Tensor fast = nn::conv2d_circular_singular_values(W, 6, 5);
  const Tensor M = nn::conv2d_matrix(6, 5, W, nn::conv_options(1), true);
  const Tensor slow = nn::svd(M, false).s;
  ASSERT_EQ(fast.numel(), 6 * 5 * 2);
  EXPECT_LT(nn::max(nn::abs(fast - nn::slice(slow, {0}, {60}))).item(), 1e-10);
  EXPECT_THROW((void)nn::conv2d_circular_singular_values(W, 2, 5), std::invalid_argument);
}
