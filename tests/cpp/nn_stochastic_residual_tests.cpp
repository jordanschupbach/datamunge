#include <datamunge/nn/nn.hpp>
#include <gtest/gtest.h>
#include <print>
namespace n = datamunge::nn;
TEST(NnStochasticModels, MaskScalingEndpointsGranularityAndAllDerivatives) {
  namespace n = datamunge::nn;
  using Scaling = n::models::DropPathScaling;
  for (bool row : {false, true})
    for (auto scaling : {Scaling::kOriginal, Scaling::kInverted})
      for (double survival : {0., .6, 1.}) {
        const bool inverted = scaling == Scaling::kInverted;
        const int rows = row ? 8192 : 4,
                  repeats = row || survival == 0 || survival == 1 ? 1 : 512;
        n::Generator gen(73);
        const auto x = n::full({rows, 2, 2, 2}, .25).requires_grad_(),
                   weight = n::scalar(1.5).requires_grad_();
        int64_t alive = 0, samples = 0;
        double error = 0;
        for (int repeat = 0; repeat < repeats; ++repeat) {
          const auto branch = x * weight;
          const auto train = n::models::DropPath(branch, survival, true, row, scaling, gen),
                     infer = n::models::DropPath(branch, survival, false, row, scaling, gen);
          const auto gradients = n::grad(n::sum(x + train.value), {x, weight});
          const auto multiplier =
              inverted && survival > 0 ? train.gate / survival : train.gate;
          error = std::max(
              error, n::max(n::abs((x + train.value) - x * (1 + 1.5 * multiplier))).item());
          error = std::max(
              error,
              n::max(n::abs(gradients[0] - (n::ones_like(x) + 1.5 * multiplier))).item());
          const double expected_weight = n::sum(x.detach() * multiplier).item();
          error = std::max(error, std::abs(gradients[1].item() - expected_weight) /
                                      std::max(1., std::abs(expected_weight)));
          error = std::max(
              error, n::max(n::abs(infer.value - x * 1.5 * (inverted ? 1 : survival))).item());
          if (n::sum(n::abs(train.gate * (1 - train.gate))).item() != 0)
            throw std::runtime_error("mask values, gradients or rate mismatch");
          alive += static_cast<int64_t>(n::sum(train.gate).item());
          samples += train.gate.numel();
        }
        const double empirical = double(alive) / samples,
                     tolerance = 4 * std::sqrt(survival * (1 - survival) / samples) + .002;
        if (error > 3e-5 || std::abs(empirical - survival) > tolerance ||
            (survival > 0 && survival < 1 && (alive == 0 || alive == samples)))
          throw std::runtime_error("mask values, gradients or rate mismatch");
        std::print("{} / {}, survival {}: observed {:.6f}, {} independent masks; "
                   "value/input/weight derivative max error {:.9f}.\n",
                   row ? "per row" : "whole batch", inverted ? "inverted" : "original",
                   survival, empirical, samples, error);
      }
  std::print(
      "At survival zero, both training values are zero; original inference returns zero and "
      "inverted inference returns the full branch. Endpoints never divide by zero.\n");
}

TEST(NnStochasticModels, LinearScheduleAndPooledShortcutDerivatives) {
  const auto p = n::models::SurvivalProbabilities(54, .5f);
  double total = 0;
  for (float v : p)
    total += v;
  EXPECT_NEAR(total, 40.25, 2e-6);
  EXPECT_EQ(p.back(), .5f);
  EXPECT_LT(p[0], 1);
  const auto x =
      n::arange(1, 33).to(n::DType::Float32).reshape({1, 4, 4, 2}).detach().requires_grad_();
  const auto y = n::models::CifarIdentityShortcut(x, 4, 2);
  EXPECT_EQ(y.shape(), n::Shape({1, 2, 2, 4}));
  EXPECT_EQ(y.at({0, 0, 0, 0}), 6);
  EXPECT_EQ(y.at({0, 0, 0, 1}), 7);
  EXPECT_EQ(n::max(n::abs(n::slice_axis(y, 3, 2, 4))).item(), 0);
  EXPECT_EQ(n::max(n::abs(n::grad(n::sum(y), {x})[0] - .25)).item(), 0);
}
TEST(NnStochasticModels, CompleteTwentyLayerCountsShapesGradientsAndMaskedBnUpdates) {
  n::set_num_threads(2);
  n::ParameterStore p(17), state(17);
  n::Generator gen(71), rng(29);
  n::models::StochasticCifarConfig cfg;
  cfg.depth = 20;
  cfg.last_survival = 1;
  cfg.update_stats = false;
  const auto x = n::normal({2, 16, 16, 3}, 0, .2, rng);
  const auto out = n::models::StochasticCifarResNet(p, state, x, cfg, true, gen);
  EXPECT_EQ(p.count(), 269722);
  EXPECT_EQ(state.count(), 1376);
  EXPECT_EQ(out.gates.size(), 9);
  for (const auto &g : out.gates)
    EXPECT_EQ(g.item(), 1);
  for (int i = 0; i < 3; ++i)
    EXPECT_EQ(out.stages[i].shape(), n::Shape({2, 16 / (1 << i), 16 / (1 << i), 16 << i}));
  for (const auto &g : n::grad(n::mean(n::square(out.logits)), p.parameters()))
    EXPECT_TRUE(std::isfinite(n::sum(n::abs(g)).item()));
  const auto saved = state.flatten();
  n::NoGradGuard guard;
  n::models::StochasticCifarResNet(p, state, x, cfg, false, gen);
  EXPECT_EQ(saved, state.flatten());
  cfg.update_stats = true;
  cfg.last_survival = 0;
  const auto masked = n::models::StochasticCifarResNet(p, state, x, cfg, true, gen);
  EXPECT_EQ(masked.gates.back().item(), 0);
  EXPECT_NE(saved, state.flatten());
}
TEST(NnStochasticModels, InferenceDoesNotConsumeMaskGenerator) {
  n::Generator a(71), b(71);
  const auto x = n::ones({4, 3});
  n::models::DropPath(x, .5, false, true, n::models::DropPathScaling::kInverted, a);
  n::models::DropPath(x, 0, true, true, n::models::DropPathScaling::kInverted, a);
  n::models::DropPath(x, 1, true, true, n::models::DropPathScaling::kOriginal, a);
  const auto ga = n::models::DropPath(x, .5, true, true, n::models::DropPathScaling::kInverted,
                                      a)
                      .gate,
             gb = n::models::DropPath(x, .5, true, true, n::models::DropPathScaling::kInverted,
                                      b)
                      .gate;
  EXPECT_EQ(n::max(n::abs(ga - gb)).item(), 0);
}
TEST(NnStochasticModels, RejectsInvalidMaskAndNetworkSettings) {
  n::Generator gen(71);
  using S = n::models::DropPathScaling;
  EXPECT_THROW(n::models::DropPath(n::scalar(1), .5, true, true, S::kOriginal, gen),
               std::invalid_argument);
  EXPECT_THROW(n::models::DropPath(n::ones({2, 3}), -1, true, false, S::kOriginal, gen),
               std::invalid_argument);
  EXPECT_THROW(n::models::DropPath(n::ones({2, 3}), .5, true, false, static_cast<S>(4), gen),
               std::invalid_argument);
  EXPECT_THROW(n::models::SurvivalProbabilities(0, .5f), std::invalid_argument);
  n::models::StochasticCifarConfig cfg;
  cfg.depth = 12;
  EXPECT_THROW(cfg.Validate(), std::invalid_argument);
  n::ParameterStore p(17), state(17);
  cfg = {};
  cfg.depth = 8;
  EXPECT_THROW(
      n::models::StochasticCifarResNet(p, state, n::ones({1, 7, 7, 3}), cfg, false, gen),
      std::invalid_argument);
}
