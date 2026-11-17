#include <gtest/gtest.h>

#include <dm/dm.hpp>

#include "../fixtures/rl_replay_reference.hpp"
namespace {
namespace n = dm::nn;
namespace m = n::models;
namespace r = dm::rl::replay;
r::Experience Experience(int i) {
  return {{float(i), float(i + 1)}, i % 2, float(i), {float(i + 1), float(i + 2)}, 0, {}};
}
TEST(DeepQComponents, UniformRingMatchesFrozenStreamAndOwnsSamples) {
  r::ReplayBuffer                actual(7, 2);
  replay_reference::ReplayBuffer oracle(7, 2);
  for (int i = 0; i < 13; ++i) {
    auto t = Experience(i);
    actual.Add(t);
    oracle.Add({t.s, t.a, t.r, t.s2, t.done, t.act_c});
  }
  std::mt19937_64 a(37), b(37);
  auto sample=actual.Sample(30,a);auto expected=oracle.Sample(30,b);
  EXPECT_EQ(sample.indices, expected.indices);
  EXPECT_EQ(sample.s.to_vector<float>(), expected.s.to_vector<float>());
  EXPECT_EQ(a(), b());
  auto saved = sample.s.to_vector<float>();
  for (int i = 0; i < 20; ++i)
    actual.Add(Experience(i + 20));
  EXPECT_EQ(saved, sample.s.to_vector<float>());
  auto clone = actual;
  clone.Add(Experience(99));
  std::mt19937_64 c(1), d(1);
  EXPECT_NE(clone.Sample(100, c).s.to_vector<float>(), actual.Sample(100, d).s.to_vector<float>());
}
TEST(DeepQComponents, PrioritizedNonPowerOfTwoMatchesFrozenIndicesWeightsAndEngine) {
  r::PrioritizedReplay                actual(7, 2, .6);
  replay_reference::PrioritizedReplay oracle(7, 2, .6);
  for (int i = 0; i < 13; ++i) {
    auto t = Experience(i);
    actual.Add(t);
    oracle.Add({t.s, t.a, t.r, t.s2, t.done, t.act_c});
  }
  std::vector<int>   idx{0, 1, 2, 3, 4, 5, 6, 2};
  std::vector<float> td{.1, .2, .3, 1, 2, 3, 10, .7};
  actual.UpdatePriorities(idx, td);
  oracle.UpdatePriorities(idx, td);
  EXPECT_EQ(actual.total(), oracle.total());
  EXPECT_EQ(actual.max_priority(), oracle.max_priority());
  std::mt19937_64 a(17), b(17);
  for (float beta : {0.f, .4f, 1.f}) {
    auto x = actual.Sample(1500, beta, a);auto y = oracle.Sample(1500, beta, b);
    EXPECT_EQ(x.indices, y.indices);
    EXPECT_EQ(x.weights.to_vector<float>(), y.weights.to_vector<float>());
  }
  EXPECT_EQ(a(), b());
}
TEST(DeepQComponents, ReplayValidationIsTransactional) {
  r::ReplayBuffer empty(3, 2);
  std::mt19937_64 rng(1);
  EXPECT_THROW(empty.Sample(1, rng), std::invalid_argument);
  EXPECT_THROW(empty.Add({{0}, 0, 1, {1}, 0, {}}), std::invalid_argument);
  EXPECT_EQ(empty.size(), 0);
  r::PrioritizedReplay p(3, 2);
  p.Add(Experience(0));
  float mass = p.total();
  EXPECT_THROW(p.UpdatePriorities({0, 2}, {3, 1}), std::invalid_argument);
  EXPECT_EQ(mass, p.total());
  EXPECT_THROW(p.Sample(3, 1.2, rng), std::invalid_argument);
  EXPECT_THROW(r::ReplayBuffer(0, 2), std::invalid_argument);
  EXPECT_THROW(r::PrioritizedReplay(3, 2, NAN), std::invalid_argument);
}
TEST(DeepQComponents, BatchPackOwnsContinuousActionsAndRejectsRaggedRows) {
  auto t      = Experience(1);
  t.act_c     = {.2, .3};
  auto sample = r::MakeBatch({&t}, 2, {1}, {0});
  t.act_c[0]  = 9;
  t.s[0]      = 9;
  EXPECT_FLOAT_EQ(sample.act_c.to_vector<float>()[0], .2);
  EXPECT_EQ(sample.s.to_vector<float>()[0], 1);
  auto bad = Experience(2);
  EXPECT_THROW(r::MakeBatch({&t, &bad}, 2, {1, 1}, {0, 1}), std::invalid_argument);
  EXPECT_THROW(r::MakeBatch({nullptr}, 2, {1}, {0}), std::invalid_argument);
}
TEST(DeepQComponents, NStepRewardsAndTerminalPrefixesMatchArithmetic) {
  r::NStepBuffer buffer(3, .9);
  EXPECT_TRUE(buffer.Push(Experience(1)).empty());
  EXPECT_TRUE(buffer.Push(Experience(2)).empty());
  auto ready = buffer.Push(Experience(3));
  ASSERT_EQ(ready.size(), 1);
  EXPECT_NEAR(ready[0].r, 1 + .9 * 2 + .81 * 3, 1e-6);
  auto end  = Experience(4);
  end.done  = 1;
  auto tail = buffer.Push(end);
  ASSERT_EQ(tail.size(), 3);
  EXPECT_NEAR(tail[0].r, 2 + .9 * 3 + .81 * 4, 1e-6);
  EXPECT_NEAR(tail[1].r, 3 + .9 * 4, 1e-6);
  EXPECT_EQ(tail[2].r, 4);
  for (auto& t : tail) {
    EXPECT_EQ(t.done, 1);
    EXPECT_EQ(t.s2, end.s2);
  }
  EXPECT_TRUE(buffer.Flush().empty());
}
TEST(DeepQComponents, TruncationFlushRetainsNonterminalShortPrefixes) {
  r::NStepBuffer buffer(3, .9);
  buffer.Push(Experience(1));
  buffer.Push(Experience(2));
  auto tail = buffer.Flush();
  ASSERT_EQ(tail.size(), 2);
  EXPECT_NEAR(tail[0].r, 2.8, 1e-6);
  EXPECT_EQ(tail[1].r, 2);
  EXPECT_EQ(tail[0].done, 0);
  EXPECT_EQ(tail[1].done, 0);
  EXPECT_THROW(r::NStepBuffer(0, .9), std::invalid_argument);
}
void SameStore(const n::ParameterStore& a, const n::ParameterStore& b) {
  ASSERT_EQ(a.names(), b.names());
  for (auto& name : a.names())
    EXPECT_EQ(a.at(name).to_vector<float>(), b.at(name).to_vector<float>()) << name;
}
TEST(DeepQComponents, PlainGraphMatchesSourceInitializationAndInputDerivative) {
  n::ParameterStore a(3), b(3);
  auto x        = n::from_vector(std::vector<float>{.2, -.1, .3, .7}, {1, 4}).requires_grad_();
  auto actual   = m::DeepQ(a, x, 128, 2);
  auto expected = n::dense(b, "q", n::mlp(b, "mlp", x, {128, 128}), 2);
  SameStore(a, b);
  EXPECT_EQ(a.count(), 17410);
  EXPECT_EQ(actual.to_vector<float>(), expected.to_vector<float>());
  auto ga = n::grad(n::sum(actual), {x})[0], gb = n::grad(n::sum(expected), {x})[0];
  EXPECT_EQ(ga.to_vector<float>(), gb.to_vector<float>());
}
TEST(DeepQComponents, DuelingCenteringAndValueGradient) {
  n::ParameterStore p(7);
  auto              path = m::DuelingQ(p, n::ones({3, 4}), 8, 2);
  EXPECT_LT(n::max(n::abs(n::mean(path.advantage, {1}))).item(), 1e-6);
  EXPECT_LT(n::max(n::abs(n::mean(path.q, {1}) - n::squeeze(path.value, {1}))).item(), 1e-6);
  auto g = n::grad(n::sum(path.q), {path.value})[0];
  for (float v : g.to_vector<float>())
    EXPECT_EQ(v, 2);
}
TEST(DeepQComponents, CategoricalExpectationAndProjectionMass) {
  n::ParameterStore p(2);
  auto              z     = n::from_vector(std::vector<float>{0, 1, 2});
  auto              path  = m::CategoricalQ(p, n::ones({2, 4}), 8, 2, z);
  auto              probs = path.probs.to_vector<float>(), q = path.q.to_vector<float>();
  for (int i = 0; i < 4; ++i) {
    EXPECT_NEAR(probs[i * 3] + probs[i * 3 + 1] + probs[i * 3 + 2], 1, 2e-7);
    EXPECT_FLOAT_EQ(q[i], probs[i * 3 + 1] + 2 * probs[i * 3 + 2]);
  }
  float pn[] = {.2, .3, .5}, reward[] = {.5}, done[] = {1}, out[3];
  m::ProjectDistribution(pn, reward, done, 1, 3, 0, 2, .9, out);
  EXPECT_FLOAT_EQ(out[0], .5);
  EXPECT_FLOAT_EQ(out[1], .5);
  EXPECT_FLOAT_EQ(out[2], 0);
  done[0] = 0;
  m::ProjectDistribution(pn, reward, done, 1, 3, 0, 2, 1, out);
  EXPECT_NEAR(out[0] + out[1] + out[2], 1, 1e-7);
  EXPECT_NEAR(out[0], .1, 1e-7);
  EXPECT_NEAR(out[1], .25, 1e-7);
  EXPECT_NEAR(out[2], .65, 1e-7);
}
TEST(DeepQComponents, ProjectionInvalidInputLeavesOutputUnchanged) {
  float p[] = {.5, .5}, reward[] = {0}, done[] = {0}, out[] = {7, 9};
  EXPECT_THROW(m::ProjectDistribution(p, reward, done, 1, 2, 1, 1, .9, out), std::invalid_argument);
  EXPECT_EQ(out[0], 7);
  p[0] = -.5;
  EXPECT_THROW(m::ProjectDistribution(p, reward, done, 1, 2, 0, 1, .9, out), std::invalid_argument);
  EXPECT_EQ(out[1], 9);
}
TEST(DeepQComponents, QuantileAxisOrderAndImplicitConditioning) {
  n::ParameterStore p(1), q(1);
  auto              fixed = m::QuantileQ(p, n::ones({2, 4}), 8, 2, 3);
  EXPECT_EQ(fixed.theta.shape(), (n::Shape{2, 2, 3}));
  EXPECT_EQ(fixed.q.to_vector<float>(), n::mean(fixed.theta, {2}).to_vector<float>());
  auto tau = n::from_vector(std::vector<float>{.1, .5, .9, .1, .5, .9}, {2, 3}).requires_grad_();
  auto implicit = m::ImplicitQuantileQ(
      q,
      n::ones({2, 4}),
      tau,
      8,
      2,
      n::from_vector(std::vector<float>{0, 3.1415927f, 6.2831855f}, {1, 1, 3}));
  EXPECT_EQ(implicit.theta.shape(), (n::Shape{2, 3, 2}));
  auto g = n::grad(n::sum(implicit.q), {tau})[0];
  for (float v : g.to_vector<float>())
    EXPECT_TRUE(std::isfinite(v));
  EXPECT_GT(n::sum(n::abs(g)).item(), 0);
}
TEST(DeepQComponents, NoisyGraphPreservesSourceGeneratorAndQuietOutput) {
  n::ParameterStore a(2), b(2);
  n::Generator      noise_a(19), noise_b(19);
  auto              x        = n::ones({2, 4});
  auto              actual   = m::NoisyQ(a, x, 8, 2, noise_a, 1);
  auto              h1       = n::noisy_dense(b, "noisy1", x, 8, noise_b, 1, n::Activation::Relu);
  auto              h2       = n::noisy_dense(b, "noisy2", h1, 8, noise_b, 1, n::Activation::Relu);
  auto              expected = n::noisy_dense(b, "noisy_q", h2, 2, noise_b, 1);
  EXPECT_EQ(actual.to_vector<float>(), expected.to_vector<float>());
  SameStore(a, b);
  EXPECT_EQ(
      n::normal({10}, 0, 1, noise_a).to_vector<float>(),
      n::normal({10}, 0, 1, noise_b).to_vector<float>());
  EXPECT_EQ(
      m::NoisyQ(a, x, 8, 2, noise_a, 0).to_vector<float>(),
      m::NoisyQ(a, x, 8, 2, noise_a, 0).to_vector<float>());
}
TEST(DeepQComponents, RainbowDistributionAndLiveSigmaPaths) {
  n::ParameterStore p(5);
  n::Generator      noise(77);
  auto              path =
      m::RainbowQ(p, n::ones({2, 4}), 8, 2, n::from_vector(std::vector<float>{0, 1, 2}), noise, 1);
  EXPECT_EQ(path.logits.shape(), (n::Shape{2, 2, 3}));
  auto mass = n::sum(path.probs, {-1}).to_vector<float>();
  for (float v : mass)
    EXPECT_NEAR(v, 1, 2e-7);
  auto gradients = n::grad(n::sum(path.q), p.parameters());
  bool sigma     = false;
  auto names     = p.names();
  for (size_t i = 0; i < names.size(); ++i)
    if (names[i].find("sigma") != std::string::npos) {
      sigma = true;
      EXPECT_TRUE(gradients[i].defined());
    }
  EXPECT_TRUE(sigma);
}
} // namespace
