/// @file nn_batch_statistics_tests.cpp
/// @brief Tests for batch renormalization, exact batch statistics and batch-norm folding.

#include <gtest/gtest.h>

#include <datamunge/nn/nn.hpp>

namespace nn = datamunge::nn;
using nn::Tensor;

TEST(NnBatchStatistics, RenormWithoutCorrectionIsBatchNorm) {
  nn::Generator g(1);
  const auto F64 = nn::DType::Float64;
  nn::ParameterStore p1(1, F64), s1(1, F64), p2(1, F64), s2(1, F64);
  for (int step = 0; step < 3; ++step) {
    const Tensor x = nn::normal({16, 3}, 2.0, 3.0, g, F64);
    const Tensor a = nn::batch_norm(p1, s1, "bn", x, true, {.momentum = 0.9});
    const Tensor b = nn::batch_renorm(p2, s2, "bn", x, true, {.momentum = 0.9, .r_max = 1.0, .d_max = 0.0});
    EXPECT_LT(nn::max(nn::abs(a - b)).item(), 1e-12);
  }
  EXPECT_LT(nn::max(nn::abs(s1.at("bn/moving_mean") - s2.at("bn/moving_mean"))).item(), 1e-12);
  EXPECT_LT(nn::max(nn::abs(s1.at("bn/moving_var") - s2.at("bn/moving_var"))).item(), 1e-12);
  EXPECT_THROW((void)nn::batch_renorm(p2, s2, "bn", nn::normal({4, 3}, 0, 1, g, F64), true, {.r_max = 0.5}), std::invalid_argument);
}

TEST(NnBatchStatistics, UnclippedRenormMatchesInferenceForward) {
  nn::Generator g(2);
  const auto F64 = nn::DType::Float64;
  nn::ParameterStore p(1, F64), s(1, F64);
  s.get("bn/moving_mean", {2}, nn::init::zeros());
  s.get("bn/moving_var", {2}, nn::init::ones());
  s.set("bn/moving_mean", nn::tensor({1.0, -2.0}, {2}, F64));
  s.set("bn/moving_var", nn::tensor({4.0, 0.25}, {2}, F64));
  const Tensor x = nn::normal({8, 2}, 0.0, 1.0, g, F64).requires_grad_(true);
  const Tensor infer = nn::batch_norm(p, s, "bn", x, false, {.eps = 1e-3});
  const Tensor train = nn::batch_renorm(p, s, "bn", x, true, {.momentum = 0.5, .r_max = 100, .d_max = 100});
  EXPECT_LT(nn::max(nn::abs(infer - train)).item(), 1e-12);
  // The batch statistics stay in the gradient: shifting a whole channel changes nothing, so columns sum to zero.
  const Tensor r = nn::normal({8, 2}, 0, 1, g, F64);
  const Tensor gx = nn::grad({nn::sum(train * r)}, {x})[0];
  EXPECT_LT(nn::max(nn::abs(nn::sum(gx, {0}))).item(), 1e-12);
  // Momentum 0.5 update with the unbiased batch variance.
  const Tensor m = nn::mean(x.detach(), {0});
  EXPECT_NEAR(s.at("bn/moving_mean").at({0}), 0.5 * 1.0 + 0.5 * m.at({0}), 1e-12);
}

TEST(NnBatchStatistics, AccumulatorMatchesConcatenation) {
  nn::Generator g(3);
  const auto F64 = nn::DType::Float64;
  const Tensor a = nn::normal({5, 2, 3}, 1.0, 2.0, g, F64), b = nn::normal({7, 2, 3}, -1.0, 0.5, g, F64), c = nn::normal({1, 2, 3}, 4.0, 1.0, g, F64);
  nn::BatchStatsAccumulator acc;
  for (const Tensor* t : {&a, &b, &c}) acc.add(*t);
  EXPECT_EQ(acc.count(), 26);
  const Tensor all = nn::reshape(nn::concat({a, b, c}, 0), {26, 3});
  const Tensor mu = nn::mean(all, {0});
  const Tensor var = nn::sum(nn::square(all - mu), {0}) / 25.0;
  EXPECT_LT(nn::max(nn::abs(acc.mean() - mu)).item(), 1e-12);
  EXPECT_LT(nn::max(nn::abs(acc.variance() - var)).item(), 1e-12);
  EXPECT_LT(nn::max(nn::abs(acc.variance(false) - var * (25.0 / 26.0))).item(), 1e-12);
  nn::ParameterStore stats(0, F64);
  acc.write(stats, "bn");
  EXPECT_LT(nn::max(nn::abs(stats.at("bn/moving_var") - var)).item(), 1e-12);
  EXPECT_THROW(acc.add(nn::normal({2, 4}, 0, 1, g, F64)), std::invalid_argument);
  acc.reset();
  EXPECT_THROW((void)acc.mean(), std::logic_error);
}

TEST(NnBatchStatistics, FoldingMatchesInferenceBatchNorm) {
  nn::Generator g(4);
  const auto F64 = nn::DType::Float64;
  const Tensor x = nn::normal({6, 4}, 0, 1, g, F64), W = nn::normal({4, 3}, 0, 1, g, F64), b = nn::normal({3}, 0, 1, g, F64);
  const Tensor gamma = nn::tensor({1.5, 0.5, -1.0}, {3}, F64), beta = nn::tensor({0.1, 0.2, 0.3}, {3}, F64);
  const Tensor mu = nn::tensor({0.4, -0.3, 1.0}, {3}, F64), var = nn::tensor({2.0, 0.5, 1.5}, {3}, F64);
  const Tensor want = nn::batch_norm_infer(nn::matmul(x, W) + b, gamma, beta, mu, var, 1e-3);
  const auto f = nn::fold_batch_norm(W, b, gamma, beta, mu, var, 1e-3);
  EXPECT_LT(nn::max(nn::abs(nn::matmul(x, f.weight) + f.bias - want)).item(), 1e-12);
  const auto nob = nn::fold_batch_norm(W, Tensor(), gamma, beta, mu, var, 1e-3);
  EXPECT_LT(nn::max(nn::abs(nn::matmul(x, nob.weight) + nob.bias - nn::batch_norm_infer(nn::matmul(x, W), gamma, beta, mu, var, 1e-3))).item(), 1e-12);
  EXPECT_THROW((void)nn::fold_batch_norm(W, b, nn::ones({4}, F64), beta, mu, var), std::invalid_argument);
}
