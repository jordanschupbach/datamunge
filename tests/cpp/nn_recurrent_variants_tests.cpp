/// @file nn_recurrent_variants_tests.cpp
/// @brief Tests for linear_scan(), minGRU, minLSTM and indrnn_step().

#include <gtest/gtest.h>

#include <datamunge/nn/nn.hpp>

#include <cstdint>
#include <vector>

namespace nn = datamunge::nn;
using nn::Tensor;

namespace {
const auto F64 = nn::DType::Float64;

double max_abs_diff(const Tensor& a, const Tensor& b) { return nn::max(nn::abs(a - b)).item(); }

/// Sequential reference for h_t = a_t h_{t-1} + b_t.
Tensor loop_scan(const Tensor& a, const Tensor& b, const Tensor& h0) {
  const std::int64_t B = a.size(0), T = a.size(1), H = a.size(2);
  Tensor h = h0;
  std::vector<Tensor> out;
  for (std::int64_t t = 0; t < T; ++t) {
    h = nn::reshape(nn::slice(a, {0, t, 0}, {B, 1, H}), {B, H}) * h + nn::reshape(nn::slice(b, {0, t, 0}, {B, 1, H}), {B, H});
    out.push_back(nn::reshape(h, {B, 1, H}));
  }
  return nn::concat(out, 1);
}
}  // namespace

TEST(NnRecurrentVariants, LinearScanMatchesLoopAndGradients) {
  nn::Generator g(3);
  for (std::int64_t T : {1, 2, 5, 8, 13}) {
    Tensor a = nn::uniform({2, T, 3}, -1.0, 1.0, g, F64), b = nn::normal({2, T, 3}, 0.0, 1.0, g, F64);
    Tensor h0 = nn::normal({2, 3}, 0.0, 1.0, g, F64);
    EXPECT_LT(max_abs_diff(nn::linear_scan(a, b, h0), loop_scan(a, b, h0)), 1e-13) << "T = " << T;
    EXPECT_LT(max_abs_diff(nn::linear_scan(a, b), loop_scan(a, b, nn::zeros({2, 3}, F64))), 1e-13) << "T = " << T;
    a.requires_grad_(), b.requires_grad_(), h0.requires_grad_();
    const Tensor w = nn::normal({2, T, 3}, 0.0, 1.0, g, F64);
    const auto gs = nn::grad(nn::sum(nn::linear_scan(a, b, h0) * w), {a, b, h0});
    const auto gl = nn::grad(nn::sum(loop_scan(a, b, h0) * w), {a, b, h0});
    for (int k = 0; k < 3; ++k) EXPECT_LT(max_abs_diff(gs[k], gl[k]), 1e-12) << "T = " << T << ", input " << k;
  }
  const Tensor a = nn::ones({1, 4, 2}, F64);
  EXPECT_THROW((void)nn::linear_scan(a, nn::ones({1, 3, 2}, F64)), std::invalid_argument);
  EXPECT_THROW((void)nn::linear_scan(a, a, nn::ones({1, 3}, F64)), std::invalid_argument);
}

TEST(NnRecurrentVariants, MinimalCellsParallelEqualsSequential) {
  nn::Generator g(5);
  const Tensor x = nn::normal({3, 7, 4}, 0.0, 1.0, g, F64), h0 = nn::normal({3, 5}, 0.0, 1.0, g, F64);
  nn::ParameterStore p(11, F64);
  const Tensor hg = nn::mingru(p, "g", x, 5, h0, -1.0), hl = nn::minlstm(p, "l", x, 5, h0, 2.0);
  Tensor sg = h0, sl = h0;
  for (std::int64_t t = 0; t < 7; ++t) {
    const Tensor xt = nn::reshape(nn::slice(x, {0, t, 0}, {3, 1, 4}), {3, 4});
    sg = nn::mingru_step(p, "g", xt, sg), sl = nn::minlstm_step(p, "l", xt, sl);
    EXPECT_LT(max_abs_diff(sg, nn::reshape(nn::slice(hg, {0, t, 0}, {3, 1, 5}), {3, 5})), 1e-13);
    EXPECT_LT(max_abs_diff(sl, nn::reshape(nn::slice(hl, {0, t, 0}, {3, 1, 5}), {3, 5})), 1e-13);
  }
  // Bias initialisation reaches the gates.
  EXPECT_DOUBLE_EQ(p.get("g/z/b", {5}, nn::init::zeros()).to_vector<double>()[0], -1.0);
  EXPECT_DOUBLE_EQ(p.get("l/f/b", {5}, nn::init::zeros()).to_vector<double>()[0], 2.0);
  // minLSTM's normalised gates keep the state inside the hull of h0 and the candidates: with x = 0 the candidate is
  // the bias 0, so |h_t| never exceeds |h0|.
  const Tensor hz = nn::minlstm(p, "l", nn::zeros({3, 7, 4}, F64), 5, h0);
  EXPECT_LE(nn::max(nn::abs(hz)).item(), nn::max(nn::abs(h0)).item() + 1e-15);
  EXPECT_THROW((void)nn::mingru(p, "g", nn::zeros({3, 4}, F64), 5), std::invalid_argument);
}

TEST(NnRecurrentVariants, MinGruGradientIsProductOfRetainGates) {
  // dh_T/dh_0 for minGRU is exactly prod_t (1 - z_t), elementwise.
  nn::Generator g(7);
  const Tensor x = nn::normal({2, 6, 3}, 0.0, 1.0, g, F64);
  Tensor h0 = nn::zeros({2, 4}, F64);
  h0.requires_grad_();
  nn::ParameterStore p(2, F64);
  const Tensor h = nn::mingru(p, "g", x, 4, h0, 0.5);
  const Tensor gh = nn::grad(nn::sum(nn::slice(h, {0, 5, 0}, {2, 1, 4})), {h0})[0];
  nn::NoGradGuard ng;
  const Tensor z = nn::sigmoid(nn::reshape(nn::matmul(nn::reshape(x, {12, 3}), p.get("g/z/W", {3, 4}, nn::init::zeros())) +
                                               p.get("g/z/b", {4}, nn::init::zeros()), {2, 6, 4}));
  const Tensor expected = nn::exp(nn::sum(nn::log(1.0 - z), {1}));
  EXPECT_LT(max_abs_diff(gh, expected), 1e-14);
}

TEST(NnRecurrentVariants, IndRnnStep) {
  nn::Generator g(9);
  const Tensor x = nn::normal({2, 3}, 0.0, 1.0, g, F64), h = nn::normal({2, 4}, 0.0, 1.0, g, F64);
  nn::ParameterStore p(4, F64);
  const Tensor out = nn::indrnn_step(p, "c", x, h, nn::Activation::None);
  const Tensor u = p.get("c/u", {4}, nn::init::zeros()), W = p.get("c/W", {3, 4}, nn::init::zeros()), b = p.get("c/b", {4}, nn::init::zeros());
  EXPECT_LT(max_abs_diff(out, nn::matmul(x, W) + b + u * h), 1e-15);
  for (double v : u.to_vector<double>()) EXPECT_TRUE(v >= 0.0 && v <= 1.0);
  const Tensor relu = nn::indrnn_step(p, "c", x, h);
  EXPECT_LT(max_abs_diff(relu, nn::relu(out)), 1e-15);
}
