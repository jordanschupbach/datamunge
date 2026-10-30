/// @file nn_adaptive_variants_tests.cpp
/// @brief Tests for RAdam, AdaBelief, Yogi, radam_rectification() and adam_step_bound().

#include <gtest/gtest.h>

#include <datamunge/nn/nn.hpp>

#include <cmath>
#include <stdexcept>
#include <vector>

namespace nn = datamunge::nn;
using nn::Tensor;

namespace {
const auto F64 = nn::DType::Float64;
}  // namespace

TEST(NnAdaptiveVariants, RulesMatchScalarReference) {
  // A changing gradient stream on three coordinates against the update rules written out in double arithmetic.
  for (auto rule : {nn::AdaptiveRule::RAdam, nn::AdaptiveRule::AdaBelief, nn::AdaptiveRule::Yogi}) {
    Tensor w = nn::from_vector(std::vector<double>{1.0, -2.0, 0.5}, {3});
    w.requires_grad_();
    nn::AdaptiveConfig cfg{.rule = rule, .beta1 = 0.9, .beta2 = 0.99, .epsilon = 1e-6, .weight_decay = 0.01};
    nn::AdaptiveOptimizer opt({w}, 0.05, cfg);
    std::vector<double> ref{1.0, -2.0, 0.5}, m(3, 0.0), v(3, 0.0);
    for (int t = 1; t <= 30; ++t) {
      const std::vector<double> g{t % 3 == 0 ? 2.0 : -0.2, t % 4 == 0 ? 0.0 : 0.005, std::sin(t)};
      opt.step({nn::from_vector(g, {3})});
      const double c1 = 1 - std::pow(0.9, t), c2 = 1 - std::pow(0.99, t);
      const double r = rule == nn::AdaptiveRule::RAdam ? nn::radam_rectification(t, 0.99) : 1.0;
      for (int i = 0; i < 3; ++i) {
        m[i] = 0.9 * m[i] + 0.1 * g[i];
        if (rule == nn::AdaptiveRule::RAdam) v[i] = 0.99 * v[i] + 0.01 * g[i] * g[i];
        else if (rule == nn::AdaptiveRule::AdaBelief) v[i] = 0.99 * v[i] + 0.01 * (g[i] - m[i]) * (g[i] - m[i]) + 1e-6;
        else v[i] = v[i] - 0.01 * ((v[i] - g[i] * g[i]) > 0 ? 1.0 : (v[i] - g[i] * g[i]) < 0 ? -1.0 : 0.0) * g[i] * g[i];
        ref[i] -= 0.05 * 0.01 * ref[i];
        if (rule == nn::AdaptiveRule::RAdam && r == 0.0) ref[i] -= 0.05 * m[i] / c1;
        else ref[i] -= 0.05 * r * (m[i] / c1) / (std::sqrt(v[i] / c2) + 1e-6);
      }
      const auto got = w.to_vector<double>();
      for (int i = 0; i < 3; ++i) EXPECT_NEAR(got[i], ref[i], 1e-12) << nn::adaptive_rule_name(rule) << " t = " << t << " i = " << i;
    }
    EXPECT_EQ(opt.iterations(), 30);
    EXPECT_EQ(opt.state().size(), 3u);
  }
  EXPECT_THROW(nn::AdaptiveConfig{.beta2 = 1.0}.validate(), std::invalid_argument);
}

TEST(NnAdaptiveVariants, RadamRectification) {
  // rho_1 = 1, so the first steps are not adaptive; r_t rises towards 1.
  EXPECT_NEAR(nn::radam_rho(1, 0.999), 1.0, 1e-9);
  EXPECT_DOUBLE_EQ(nn::radam_rectification(1, 0.999), 0.0);
  int first = 0;
  double prev = 0;
  for (int t = 1; t <= 10000; ++t) {
    const double r = nn::radam_rectification(t, 0.999);
    if (r > 0 && first == 0) first = t;
    EXPECT_GE(r, prev - 1e-15);
    EXPECT_LT(r, 1.0);
    prev = r;
  }
  EXPECT_EQ(first, 6);  // rho_t first exceeds 5 at t = 6 for beta2 = 0.999
  EXPECT_GT(prev, 0.99);
  EXPECT_THROW((void)nn::radam_rho(0, 0.999), std::invalid_argument);
}

TEST(NnAdaptiveVariants, AdamStepBoundIsAttained) {
  // The worst gradient history g_i ~ (beta1 / beta2)^(t - i) makes Adam's step equal the bound (with tiny epsilon).
  for (int T : {1, 5, 50, 400}) {
    Tensor w = nn::zeros({1}, F64);
    w.requires_grad_();
    nn::OptimizerConfig cfg;
    cfg.kind = nn::OptimizerKind::Adam;
    cfg.epsilon = 1e-300;
    nn::Optimizer opt({w}, 1.0, cfg);
    double before = 0;
    for (int i = 1; i <= T; ++i) {
      before = w.to_vector<double>()[0];
      opt.step({nn::from_vector(std::vector<double>{std::pow(0.9 / 0.999, T - i) * 1e-3}, {1})});
    }
    const double step = before - w.to_vector<double>()[0];
    EXPECT_NEAR(step, nn::adam_step_bound(0.9, 0.999, T), 1e-6 * step) << "T = " << T;
  }
  EXPECT_NEAR(nn::adam_step_bound(0.9, 0.999, 1), 1.0, 1e-12);
  EXPECT_NEAR(nn::adam_step_bound(0.0, 0.5, 7), std::sqrt((1 - std::pow(0.5, 7)) / 0.5), 1e-12);
  EXPECT_THROW((void)nn::adam_step_bound(0.9, 1.0, 3), std::invalid_argument);
}
