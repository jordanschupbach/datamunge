#include <gtest/gtest.h>

#include <dm/dm.hpp>
namespace {
namespace n = dm::nn;
namespace m = n::models;
namespace r = dm::rl::returns;
namespace e = dm::rl::environments;
TEST(PolicyGradientComponents, DiscountedReturnsHandEpisodeAndBootstrap) {
  auto a = r::DiscountedReturns({1, 1, 1}, {0, 0, 1}, .5);
  EXPECT_EQ(a, (std::vector<float>{1.75, 1.5, 1}));
  auto b = r::DiscountedReturns({1, 2}, {0, 0}, .5, 4);
  EXPECT_EQ(b, (std::vector<float>{3, 4}));
  auto c = r::DiscountedReturns({1, 2}, {1, 0}, .5, 4);
  EXPECT_EQ(c, (std::vector<float>{1, 4}));
  EXPECT_TRUE(r::DiscountedReturns({}, {}, .9).empty());
}
TEST(PolicyGradientComponents, GaeLambdaZeroAndOneIdentities) {
  std::vector<float> reward{1, 2, 3}, value{.2, .4, .6}, done{0, 0, 1}, adv, ret;
  r::Gae(reward, value, done, 0, .9, 0, &adv, &ret);
  for (int i = 0; i < 3; ++i) {
    float target = reward[i] + (i < 2 ? .9f * value[i + 1] : 0);
    EXPECT_NEAR(adv[i], target - value[i], 1e-6);
    EXPECT_NEAR(ret[i], target, 1e-6);
  }
  r::Gae(reward, value, done, 0, .9, 1, &adv, &ret);
  auto mc = r::DiscountedReturns(reward, done, .9);
  for (int i = 0; i < 3; ++i) {
    EXPECT_NEAR(ret[i], mc[i], 1e-6);
    EXPECT_NEAR(adv[i], mc[i] - value[i], 1e-6);
  }
}
TEST(PolicyGradientComponents, GaeIntermediateLambdaAndTerminalBoundary) {
  std::vector<float> a, b;
  r::Gae({1, 2, 3}, {.2, .4, .6}, {0, 1, 0}, 4, .9, .5, &a, &b);
  double d0 = 1 + .9 * .4 - .2, d1 = 2 - .4, d2 = 3 + .9 * 4 - .6;
  EXPECT_NEAR(a[0], d0 + .45 * d1, 1e-6);
  EXPECT_NEAR(a[1], d1, 1e-6);
  EXPECT_NEAR(a[2], d2, 1e-6);
  EXPECT_NEAR(b[2], 6.6, 1e-6);
}
TEST(PolicyGradientComponents, ReturnValidationPrecedesOutputMutation) {
  std::vector<float> a{7}, b{9};
  EXPECT_THROW(r::Gae({1}, {.2}, {0}, 0, .9, 2, &a, &b), std::invalid_argument);
  EXPECT_EQ(a, (std::vector<float>{7}));
  EXPECT_EQ(b, (std::vector<float>{9}));
  EXPECT_THROW(r::Gae({1}, {.2}, {0}, 0, .9, .5, &a, &a), std::invalid_argument);
  EXPECT_THROW(r::DiscountedReturns({1}, {}, .9), std::invalid_argument);
  EXPECT_THROW(r::DiscountedReturns({NAN}, {0}, .9), std::invalid_argument);
  EXPECT_THROW(r::DiscountedReturns({1}, {.5}, .9), std::invalid_argument);
}
TEST(PolicyGradientComponents, RolloutFlattenOwnsEnvMajorCopies) {
  r::Rollout ro;
  ro.n_envs  = 2;
  ro.n_steps = 2;
  ro.obs     = {{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}};
  ro.act     = {{0, 1}, {2, 3}};
  auto x     = ro.FlatObs();
  auto a     = ro.FlatAct();
  EXPECT_EQ(a, (std::vector<int>{0, 1, 2, 3}));
  EXPECT_EQ(x[2], (e::Obs{5, 6}));
  x[0][0] = 99;
  EXPECT_EQ(ro.obs[0][0][0], 1);
  auto copy         = ro;
  copy.obs[0][0][0] = 88;
  EXPECT_EQ(ro.obs[0][0][0], 1);
}
TEST(PolicyGradientComponents, ActorCriticMatchesIndependentSourceGraphAndParameterOrder) {
  n::ParameterStore a(13), b(13);
  auto x      = n::from_vector(std::vector<float>{.2, -.1, .3, .4}, {1, 4}).requires_grad_();
  auto ac     = m::BuildActorCritic(a, x, 2, 64);
  auto h      = n::mlp(b, "actor", x, {64, 64}, n::Activation::Tanh);
  auto logits = n::dense(b, "actor/logits", h, 2);
  auto hv     = n::mlp(b, "critic", x, {64, 64}, n::Activation::Tanh);
  auto value  = n::squeeze(n::dense(b, "critic/value", hv, 1), {1});
  EXPECT_EQ(a.count(), 9155);
  EXPECT_EQ(a.names(), b.names());
  for (const auto& name : a.names())
    EXPECT_EQ(a.at(name).to_vector<float>(), b.at(name).to_vector<float>());
  EXPECT_EQ(ac.logits.to_vector<float>(), logits.to_vector<float>());
  EXPECT_EQ(ac.value.to_vector<float>(), value.to_vector<float>());
  EXPECT_EQ(
      n::grad(n::sum(ac.logits) + n::sum(ac.value), {x})[0].to_vector<float>(),
      n::grad(n::sum(logits) + n::sum(value), {x})[0].to_vector<float>());
}
TEST(PolicyGradientComponents, ActorOnlyAndUniformEntropyEndpoints) {
  n::ParameterStore p(0);
  auto              ac = m::BuildActorCritic(p, n::zeros({2, 4}), 2, 8, false);
  EXPECT_FALSE(ac.value.defined());
  for (const auto& name : p.names())
    EXPECT_EQ(name.find("critic"), std::string::npos);
  for (float v : ac.probs.to_vector<float>())
    EXPECT_FLOAT_EQ(v, .5);
  for (float v : ac.entropy.to_vector<float>())
    EXPECT_NEAR(v, std::log(2.), 1e-7);
}
TEST(PolicyGradientComponents, SmoothPolicyRetainsSecondDerivativeForTrustRegions) {
  n::ParameterStore p(4);
  auto           x  = n::from_vector(std::vector<float>{.2, -.1, .3, .4}, {1, 4}).requires_grad_();
  auto           ac = m::BuildActorCritic(p, x, 2, 8, false);
  auto           objective = n::sum(ac.logp);
  n::GradOptions options;
  options.create_graph = true;
  auto first           = n::grad(objective, {x}, options)[0];
  auto second          = n::grad(n::sum(first), {x})[0];
  for (float v : second.to_vector<float>())
    EXPECT_TRUE(std::isfinite(v));
  EXPECT_GT(n::sum(n::abs(second)).item(), 0);
  float eps  = .001;
  auto  plus = x.detach().clone(), minus = x.detach().clone();
  for (int i = 0; i < 4; ++i) {
    plus.data_ptr<float>()[i] += eps;
    minus.data_ptr<float>()[i] -= eps;
  }
  plus.requires_grad_();
  minus.requires_grad_();
  auto gp       = n::grad(n::sum(m::BuildActorCritic(p, plus, 2, 8, false).logp), {plus})[0],
       gm       = n::grad(n::sum(m::BuildActorCritic(p, minus, 2, 8, false).logp), {minus})[0];
  auto expected = ((gp - gm) / (2 * eps)).to_vector<float>(), actual = second.to_vector<float>();
  for (int i = 0; i < 4; ++i)
    EXPECT_NEAR(actual[i], expected[i], 2e-3);
}
TEST(PolicyGradientComponents, SeparateCriticUsesExactSourceNames) {
  n::ParameterStore a(1), b(1);
  auto              x      = n::ones({2, 4});
  auto              actual = m::StandaloneValue(a, x, 8);
  auto              expected =
      n::squeeze(n::dense(b, "v", n::mlp(b, "mlp", x, {8, 8}, n::Activation::Tanh), 1), {1});
  EXPECT_EQ(a.names(), b.names());
  EXPECT_EQ(actual.to_vector<float>(), expected.to_vector<float>());
}
TEST(PolicyGradientComponents, ManualKfacActorExposesLiveIntermediateGradients) {
  n::ParameterStore p(17);
  m::KfacActorParameters(p, 4, 8, 2);
  auto names = p.names();
  m::KfacActorParameters(p, 4, 8, 2);
  EXPECT_EQ(p.names(), names);
  auto x    = n::ones({3, 4});
  auto path = m::KfacActor(p, x);
  ASSERT_EQ(path.inputs.size(), 3);
  ASSERT_EQ(path.pre.size(), 3);
  auto h = x;
  for (int l = 0; l < 3; ++l) {
    auto pre =
        n::bias_add(n::matmul(h, p.at("W" + std::to_string(l))), p.at("b" + std::to_string(l)));
    EXPECT_EQ(path.pre[l].to_vector<float>(), pre.to_vector<float>());
    h = l < 2 ? n::tanh(pre) : pre;
  }
  auto grads = n::grad(n::sum(path.logp), path.pre);
  for (auto& g : grads) {
    EXPECT_TRUE(g.defined());
    for (float v : g.to_vector<float>())
      EXPECT_TRUE(std::isfinite(v));
  }
}
TEST(PolicyGradientComponents, KfacDampedSolveHandDiagonalAndResidual) {
  auto result = m::KfacNaturalGradient(n::eye(2), n::eye(1), n::ones({1, 1}), n::ones({1}), .01);
  EXPECT_FLOAT_EQ(result.pi, 1);
  for (float x : result.natural.to_vector<float>())
    EXPECT_NEAR(x, 1 / 1.21, 1e-6);
  auto residual =
      n::matmul(n::matmul(result.A_damped, result.natural), result.G_damped) - result.gradient;
  EXPECT_LT(n::max(n::abs(residual)).item(), 1e-6);
  auto A       = n::from_vector(std::vector<float>{2, .2, .2, 1}, {2, 2}),
       G       = n::from_vector(std::vector<float>{3, .1, .1, 2}, {2, 2});
  auto general = m::KfacNaturalGradient(A, G, n::ones({1, 2}), n::ones({2}), .01);
  EXPECT_LT(
      n::max(
          n::abs(
              n::matmul(n::matmul(general.A_damped, general.natural), general.G_damped) -
              general.gradient))
          .item(),
      2e-6);
  EXPECT_THROW(
      m::KfacNaturalGradient(A, G, n::ones({1, 2}), n::ones({2}), 0), std::invalid_argument);
}
TEST(PolicyGradientComponents, NormalizedMountainCarMatchesSourcePhysicsAndScale) {
  e::NormalizedMountainCar scaled;
  e::MountainCar           raw;
  auto                     a = scaled.reset(19), b = raw.reset(19);
  EXPECT_FLOAT_EQ(a[0], (b[0] + .3f) / .9f);
  EXPECT_FLOAT_EQ(a[1], b[1] / .07f);
  for (int i = 0; i < 200; ++i) {
    auto x = scaled.step(i % 3), y = raw.step(i % 3);
    EXPECT_FLOAT_EQ(x.obs[0], (y.obs[0] + .3f) / .9f);
    EXPECT_FLOAT_EQ(x.obs[1], y.obs[1] / .07f);
    EXPECT_EQ(x.reward, y.reward);
    EXPECT_EQ(x.terminated, y.terminated);
    EXPECT_EQ(x.truncated, y.truncated);
    if (x.done())
      break;
  }
  EXPECT_EQ(e::NormalizedMountainCar::Scale({-.3, 0}), (e::Obs{0, 0}));
  EXPECT_THROW(e::NormalizedMountainCar::Scale({1}), std::invalid_argument);
}
} // namespace
