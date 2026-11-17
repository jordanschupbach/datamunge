#include <gtest/gtest.h>

#include <dm/dm.hpp>

#include "../fixtures/rl_sequence_replay_reference.hpp"
namespace {
namespace n = dm::nn;
namespace m = n::models;
namespace c = dm::rl::continuous;
namespace r = dm::rl::replay;
namespace o = dm::rl::off_policy;
namespace e = dm::rl::environments;
TEST(ContinuousOffPolicy, SequenceReplayMatchesFrozenWindowsMasksAndStream) {
  r::SequenceReplay                  actual(5, 4, 2);
  sequence_reference::SequenceReplay oracle(5, 4, 2);
  std::vector<e::Obs>                obs{{0, 1}, {2, 3}, {4, 5}, {6, 7}, {8, 9}, {10, 11}};
  std::vector<int>                   a{0, 1, 0, 1, 0};
  std::vector<float>                 reward{1, 2, 3, 4, 5}, done{0, 0, 0, 0, 1};
  actual.AddEpisode(obs, a, reward, done);
  oracle.AddEpisode(obs, a, reward, done);
  EXPECT_EQ(actual.size(), 3);
  std::mt19937_64 x(19), y(19);
  auto            batch    = actual.Sample(50, x);
  auto            expected = oracle.Sample(50, y);
  EXPECT_EQ(batch.obs.to_vector<float>(), expected.obs.to_vector<float>());
  EXPECT_EQ(batch.mask.to_vector<float>(), expected.mask.to_vector<float>());
  EXPECT_EQ(batch.a.to_vector<int64_t>(), expected.a.to_vector<int64_t>());
  EXPECT_EQ(batch.done.to_vector<float>(), expected.done.to_vector<float>());
  EXPECT_EQ(x(), y());
}
TEST(ContinuousOffPolicy, SequencePaddingAndInvalidEpisodeTransaction) {
  r::SequenceReplay buffer(3, 4, 1);
  buffer.AddEpisode({{0}, {1}, {2}}, {0, 1}, {3, 4}, {0, 1});
  std::mt19937_64 rng(1);
  auto            sample = buffer.Sample(1, rng);
  EXPECT_EQ(sample.obs.to_vector<float>(), (std::vector<float>{0, 1, 2, 2, 2}));
  EXPECT_EQ(sample.mask.to_vector<float>(), (std::vector<float>{1, 1, 0, 0}));
  EXPECT_EQ(sample.done.to_vector<float>(), (std::vector<float>{0, 1, 1, 1}));
  EXPECT_THROW(buffer.AddEpisode({{0}, {1}}, {0}, {NAN}, {1}), std::invalid_argument);
  EXPECT_EQ(buffer.size(), 1);
  EXPECT_THROW(r::SequenceReplay(1, 0, 1), std::invalid_argument);
  r::SequenceReplay empty(2, 2, 1);
  EXPECT_THROW(empty.Sample(1, rng), std::invalid_argument);
}
TEST(ContinuousOffPolicy, ContinuousActionScalingAndOwnedNoiseStream) {
  auto box = e::ActionSpace::Continuous({-2, -4}, {2, 8});
  auto a   = c::ScaleAction({.5f, -2}, box);
  EXPECT_EQ(a.continuous, (std::vector<float>{1, -4}));
  EXPECT_THROW(c::ScaleAction({NAN, 0}, box), std::invalid_argument);
  std::mt19937_64 x(71), y(71);
  for (int call = 0; call < 2; ++call) {
    std::normal_distribution<float> normal(0, 1);
    std::vector<float>              expected(3);
    for (auto& v : expected)
      v = normal(y);
    EXPECT_EQ(c::NormalNoise(1, 3, x).to_vector<float>(), expected);
  }
  EXPECT_EQ(x(), y());
}
TEST(ContinuousOffPolicy, SquashedDensityClosedFormAndLiveMeanDerivative) {
  auto mean = n::full({1, 1}, .3).requires_grad_(), ls = n::full({1, 1}, -.5).requires_grad_(),
       eps      = n::full({1, 1}, .7);
  auto   sample = m::SquashedSample({mean, ls}, eps);
  double u = .3 + std::exp(-.5) * .7, a = std::tanh(u),
         logp = -.5 * .49 + .5 - .5 * std::log(2 * 3.14159265) - std::log(1 - a * a + 1e-6);
  EXPECT_NEAR(sample.action.item(), a, 1e-7);
  EXPECT_NEAR(sample.logp.item(), logp, 1e-6);
  auto gradient = n::grad(n::sum(sample.action), {mean})[0];
  EXPECT_NEAR(gradient.item(), 1 - a * a, 1e-7);
}
TEST(ContinuousOffPolicy, GaussianLogDensityBroadcastAndAnalyticPartials) {
  auto mean   = n::from_vector(std::vector<float>{.2, .3}, {2, 1}).requires_grad_(),
       ls     = n::zeros({1}).requires_grad_(),
       action = n::from_vector(std::vector<float>{.5, -.2}, {2, 1});
  auto lp     = m::GaussianLogProb(action, mean, ls);
  auto g      = n::grad(n::sum(lp), {mean, ls});
  auto gm     = g[0].to_vector<float>();
  EXPECT_NEAR(gm[0], .3, 1e-7);
  EXPECT_NEAR(gm[1], -.5, 1e-7);
  EXPECT_NEAR(g[1].item(), .09 + .25 - 2, 2e-7);
  EXPECT_THROW(m::GaussianLogProb(n::zeros({2, 2}), mean, ls), std::invalid_argument);
}
TEST(ContinuousOffPolicy, DeterministicActorAndCriticMatchSourceNamesValuesAndActionGradients) {
  n::ParameterStore actor(0), critic(1), reference(0);
  auto              state = n::ones({2, 3});
  auto              a     = m::DeterministicActor(actor, "pi", state, 1, 128);
  auto              expected =
      n::tanh(n::dense(reference, "pi/out", n::mlp(reference, "pi/mlp", state, {128, 128}), 1));
  EXPECT_EQ(actor.count(), 17153);
  EXPECT_EQ(actor.names(), reference.names());
  EXPECT_EQ(a.to_vector<float>(), expected.to_vector<float>());
  auto q = m::QNetwork(critic, "q", state, a, 128);
  EXPECT_EQ(critic.count(), 17281);
  auto gradient = n::grad(n::sum(q), {a})[0];
  EXPECT_EQ(gradient.shape(), (n::Shape{2, 1}));
  EXPECT_GT(n::sum(n::abs(gradient)).item(), 0);
}
TEST(ContinuousOffPolicy, GaussianActorClipsLogStdAndPpoRetainsIndependentScale) {
  n::ParameterStore p(3);
  auto              g = m::GaussianActor(p, "pi", n::ones({2, 3}), 1, 8, -.1, .1);
  for (float v : g.log_std.to_vector<float>())
    EXPECT_LE(std::abs(v), .100001);
  n::ParameterStore q(4);
  auto              path = m::GaussianValuePolicy(q, n::ones({2, 3}), 1, 8);
  EXPECT_EQ(path.log_std.shape(), (n::Shape{1}));
  EXPECT_FLOAT_EQ(path.log_std.item(), -.5);
  EXPECT_EQ(path.value.shape(), (n::Shape{2}));
}
TEST(ContinuousOffPolicy, CategoricalActionCriticMassAndExpectedReturn) {
  n::ParameterStore p(1);
  auto              path = m::DistributionalActionCritic(
      p, n::ones({2, 3}), n::zeros({2, 1}), 8, n::from_vector(std::vector<float>{-2, -1, 0}));
  auto probs = path.probs.to_vector<float>(), q = path.q.to_vector<float>();
  for (int i = 0; i < 2; ++i) {
    EXPECT_NEAR(probs[i * 3] + probs[i * 3 + 1] + probs[i * 3 + 2], 1, 1e-7);
    EXPECT_FLOAT_EQ(q[i], -2 * probs[i * 3] - probs[i * 3 + 1]);
  }
}
TEST(ContinuousOffPolicy, AcerStateValueStopsPolicyExpectationGradient) {
  n::ParameterStore p(2);
  auto              path     = m::AcerQPolicy(p, n::ones({2, 4}), 2, 8);
  auto              gradient = n::grad(n::sum(path.v), {path.q})[0];
  EXPECT_EQ(gradient.to_vector<float>(), path.probs.to_vector<float>());
  auto direct = n::sum(path.probs.detach() * path.q, {1});
  EXPECT_EQ(path.v.to_vector<float>(), direct.to_vector<float>());
}
TEST(ContinuousOffPolicy, RecurrentQMatchesSourceStepAndRetainsStateGradient) {
  n::ParameterStore p(0), reference(0);
  auto              x        = n::ones({2, 2}).requires_grad_();
  auto              state    = n::lstm_zero_state(2, 8);
  auto              actual   = m::RecurrentQStep(p, x, state, 8, 2);
  auto              embed    = n::dense(reference, "embed", x, 8, n::Activation::Relu);
  auto              expected = n::lstm_step(reference, "lstm", embed, state);
  auto              q        = n::dense(reference, "q", expected.h, 2);
  EXPECT_EQ(actual.q.to_vector<float>(), q.to_vector<float>());
  EXPECT_EQ(p.names(), reference.names());
  auto next     = m::RecurrentQStep(p, x, actual.state, 8, 2);
  auto gradient = n::grad(n::sum(next.q), {actual.state.h})[0];
  EXPECT_GT(n::sum(n::abs(gradient)).item(), 0);
}
TEST(ContinuousOffPolicy, MaskedCartPolePreservesPhysicsAndTerminality) {
  e::MaskedCartPole masked;
  e::CartPole       full;
  auto              x = masked.reset(17), y = full.reset(17);
  EXPECT_EQ(x, (e::Obs{y[0], y[2]}));
  for (int t = 0; t < 100; ++t) {
    auto a = masked.step(t % 2), b = full.step(t % 2);
    EXPECT_EQ(a.obs, (e::Obs{b.obs[0], b.obs[2]}));
    EXPECT_EQ(a.reward, b.reward);
    EXPECT_EQ(a.terminated, b.terminated);
    EXPECT_EQ(a.truncated, b.truncated);
    if (a.done())
      break;
  }
}
TEST(ContinuousOffPolicy, RetraceHandOnAndOffPolicyRecurrences) {
  std::vector<float> reward{1, 2}, done{0, 1}, behavior{.5, .5, .5, .5}, p{.5, .5, .5, .5, .5, .5},
      q(6), v(3);
  std::vector<int> a{0, 0};
  auto             target = o::Retrace(reward, done, a, behavior, p, q, v, 2, .5);
  EXPECT_EQ(target, (std::vector<float>{2, 2}));
  p[2]   = .25;
  p[3]   = .75;
  target = o::Retrace(reward, done, a, behavior, p, q, v, 2, .5);
  EXPECT_EQ(target, (std::vector<float>{1.5, 2}));
  EXPECT_THROW(
      o::Retrace(reward, done, std::vector<int>{0, 3}, behavior, p, q, v, 2, .5),
      std::invalid_argument);
}
TEST(ContinuousOffPolicy, VTraceImportanceClippingTerminalResetAndBootstrap) {
  std::vector<float> lp{0, 0}, values{0, 0, 0}, reward{1, 2}, done{0, 1};
  auto               on = o::VTrace(lp, values, reward, done, lp, .5);
  EXPECT_EQ(on.values, (std::vector<float>{2, 2, 0}));
  EXPECT_EQ(on.advantages, (std::vector<float>{2, 2}));
  std::vector<float> offlog{float(std::log(.5)), float(std::log(.5))};
  auto               off = o::VTrace(offlog, values, reward, done, lp, .5);
  EXPECT_NEAR(off.values[0], .75, 1e-7);
  EXPECT_NEAR(off.values[1], 1, 1e-7);
  EXPECT_NEAR(off.advantages[0], .75, 1e-7);
  auto high = o::VTrace(std::vector<float>{1000, 1000}, values, reward, done, lp, .5);
  EXPECT_EQ(high.values, on.values);
}
TEST(ContinuousOffPolicy, HybridReachPolicyDimensionsAndExactOracleReward) {
  n::ParameterStore p(0);
  auto              path = m::HybridReachPolicy(p, n::zeros({2, 2}));
  EXPECT_EQ(path.means.shape(), (n::Shape{2, 2}));
  EXPECT_EQ(path.log_std.shape(), (n::Shape{2}));
  for (float v : path.log_std.to_vector<float>())
    EXPECT_EQ(v, -1);
  EXPECT_FLOAT_EQ(m::HybridReachReward(0, .5, 1, .5), 0);
  EXPECT_FLOAT_EQ(m::HybridReachReward(0, .5, 0, .5), -1);
  EXPECT_FLOAT_EQ(m::HybridReachReward(0, 1, 1, 3), 0);
  EXPECT_THROW(m::HybridReachReward(0, 0, 2, 1), std::invalid_argument);
}
} // namespace
