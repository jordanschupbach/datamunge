#include <gtest/gtest.h>

#include <dm/dm.hpp>
namespace {
namespace n = dm::nn;
namespace m = n::models;
namespace e = dm::rl::environments;
namespace r = dm::rl::returns;
TEST(PixelEnvironmentComponents, PoolingRejectsMisalignedFramesBeforeMutation) {
  std::vector<unsigned char> cur{1, 4};
  e::AtariFrameStack::MaxPool(cur, {3, 2});
  EXPECT_EQ(cur, (std::vector<unsigned char>{3, 4}));
  EXPECT_THROW(e::AtariFrameStack::MaxPool(cur, {1}), std::invalid_argument);
  EXPECT_EQ(cur, (std::vector<unsigned char>{3, 4}));
}
TEST(PixelEnvironmentComponents, BoxResizeUpsamplingAndOwningHistory) {
  e::AtariFrameStack frames(2, 2);
  EXPECT_THROW(frames.StackedObs(), std::logic_error);
  EXPECT_THROW(frames.last_frame(), std::logic_error);
  for (int i = 0; i < 4; ++i)
    frames.PushFrame({0, 64, 128, 255});
  auto obs = frames.StackedObs();
  EXPECT_EQ(obs[0], 0);
  EXPECT_EQ(obs[83], 64 / 255.f);
  EXPECT_EQ(obs[83 * 84], 128 / 255.f);
  EXPECT_EQ(obs[84 * 84 - 1], 1);
  auto copy = frames;
  frames.Clear();
  EXPECT_THROW(frames.StackedObs(), std::logic_error);
  EXPECT_EQ(copy.StackedObs(), obs);
  EXPECT_THROW(copy.PushFrame({0}), std::invalid_argument);
  EXPECT_EQ(copy.StackedObs(), obs);
}
TEST(PixelEnvironmentComponents, NonintegralBoxResizeMatchesIndependentIntegerAreas) {
  const int                  w = 173, h = 211;
  std::vector<unsigned char> raw(w * h);
  for (size_t i = 0; i < raw.size(); ++i)
    raw[i] = (i * 19 + i / 17) % 256;
  e::AtariFrameStack frames(w, h);
  frames.PushFrame(raw);
  const auto& out = frames.last_frame();
  for (int y = 0; y < 84; ++y)
    for (int x = 0; x < 84; ++x) {
      int sum = 0, count = 0;
      for (int yy = y * h / 84; yy < std::max(y * h / 84 + 1, (y + 1) * h / 84); ++yy)
        for (int xx = x * w / 84; xx < std::max(x * w / 84 + 1, (x + 1) * w / 84); ++xx) {
          sum += raw[yy * w + xx];
          ++count;
        }
      EXPECT_EQ(out[y * 84 + x], sum / count);
    }
  EXPECT_THROW(e::AtariFrameStack(0, 2), std::invalid_argument);
}
TEST(PixelEnvironmentComponents, LanderNoopMatchesSemiImplicitGravityAndPotentialReward) {
  e::LunarLander env;
  auto           before = env.reset(3);
  auto           after  = env.step(0);
  const float    vy     = before[3] - 1.6f * .02f;
  EXPECT_EQ(after.obs[3], vy);
  EXPECT_EQ(after.obs[1], before[1] + vy * .02f);
  auto potential = [](const e::Obs& o) {
    return -100.f * std::sqrt(o[0] * o[0] + o[1] * o[1]) -
           100.f * std::sqrt(o[2] * o[2] + o[3] * o[3]) - 100.f * std::abs(o[4]);
  };
  EXPECT_EQ(after.reward, potential(after.obs) - potential(before));
  EXPECT_FALSE(after.done());
}
TEST(PixelEnvironmentComponents, LanderOwnsSeededStateAndRejectsInvalidAction) {
  e::LunarLander env(1);
  env.reset(4);
  auto copy = env;
  EXPECT_EQ(env.step(0).obs, copy.step(0).obs);
  auto before = env.state().x;
  EXPECT_THROW(env.step(7), std::invalid_argument);
  EXPECT_EQ(env.state().x, before);
  EXPECT_EQ(e::LunarLander(0).max_steps(), 0);
  EXPECT_THROW(e::LunarLander(-1), std::invalid_argument);
  EXPECT_EQ(env.max_steps(), 1);
}
TEST(PixelEnvironmentComponents, NativeSingleAdapterMatchesRealTaskStateAndTimeLimit) {
  auto               task = dm::sim::Task::InvertedPendulum;
  e::NativeTaskEnv   env(task, 1);
  dm::sim::TaskBatch reference(task, 1, dm::compute::Context{}, 0, 1);
  reference.seed(13);
  auto expected = reference.reset().observations;
  EXPECT_EQ(env.reset(13), e::NativeObservation(expected, 0, 4));
  auto        output = env.step(e::Action(std::vector<float>{0}));
  const auto& next   = reference.step({0});
  EXPECT_EQ(output.obs, e::NativeObservation(next.observations, 0, 4));
  EXPECT_EQ(output.reward, float(next.rewards[0]));
  EXPECT_EQ(output.truncated, next.truncated[0] != 0);
  EXPECT_TRUE(output.truncated);
  EXPECT_THROW(env.step(0), std::invalid_argument);
}
TEST(PixelEnvironmentComponents, NativeVectorRetainsFinalRowsBeforeMaskedReset) {
  auto                task = dm::sim::Task::InvertedPendulum;
  e::NativeVectorTask env(task, 2, 7, 2, 1);
  dm::sim::TaskBatch  reference(task, 2, dm::compute::Context({}, 2), 7, 1);
  reference.reset();
  env.reset();
  const auto& transition = reference.step({0, 0});
  auto        final      = transition.final_observations;
  auto        output     = env.step(std::vector<e::Action>(2, e::Action(std::vector<float>{0})));
  EXPECT_EQ(output.final_obs[0], e::NativeObservation(final, 0, 4));
  EXPECT_EQ(output.final_obs[1], e::NativeObservation(final, 1, 4));
  reference.reset(std::vector<std::uint8_t>{1, 1});
  EXPECT_EQ(output.obs[0], e::NativeObservation(reference.output().observations, 0, 4));
  EXPECT_EQ(output.truncated, (std::vector<std::uint8_t>{1, 1}));
  EXPECT_THROW(env.step({}), std::invalid_argument);
}
TEST(PixelEnvironmentComponents, ExplicitBootstrapsSeparateTimeLimitsFromPhysicalTerminals) {
  auto limited = r::NativeGae({1, 2}, {.5, .5}, {4, 0}, {1, 1}, .9, .95);
  EXPECT_NEAR(limited.first[0], 4.1, 1e-6);
  EXPECT_EQ(limited.first[1], 1.5);
  auto continuing = r::NativeGae({1, 2}, {.5, .5}, {.5, 0}, {0, 1}, .9, .95);
  EXPECT_NEAR(continuing.first[0], .95 + .9 * .95 * 1.5, 1e-6);
  EXPECT_THROW(r::NativeGae({1}, {0}, {1}, {.5}, .9, .95), std::invalid_argument);
  EXPECT_TRUE(r::NativeGae({}, {}, {}, {}, 0, 0).first.empty());
}
TEST(PixelEnvironmentComponents, NativeRowAndTaskValidation) {
  EXPECT_EQ(e::NativeObservation({1, 2, 3, 4}, 1, 2), (std::vector<float>{3, 4}));
  EXPECT_THROW(e::NativeObservation({1, 2}, 1, 2), std::invalid_argument);
  EXPECT_THROW(e::NativeObservation({1, 2}, 0, 0), std::invalid_argument);
  EXPECT_THROW(e::NativeTask("typo"), std::invalid_argument);
  EXPECT_EQ(e::NativeSpace(dm::sim::Task::Hopper).dim, 3);
}
TEST(PixelEnvironmentComponents, SnakeForwardMatchesSourceLeavesAndValues) {
  n::ParameterStore   p(0), ref(0);
  auto                x      = n::ones({1, 108});
  auto                actual = m::SnakePixelQ(p, x);
  n::ConvLayerOptions c;
  c.activation  = n::Activation::Relu;
  auto image    = n::transpose(n::reshape(x, {-1, 3, 6, 6}), {0, 2, 3, 1});
  auto c1       = n::conv2d(ref, "conv1", image, 16, 3, c);
  auto c2       = n::conv2d(ref, "conv2", c1, 32, 3, c);
  auto hidden   = n::dense(ref, "fc", n::reshape(c2, {-1, 1152}), 128, n::Activation::Relu);
  auto expected = n::dense(ref, "q", hidden, 4);
  EXPECT_EQ(p.names(), ref.names());
  EXPECT_EQ(actual.to_vector<float>(), expected.to_vector<float>());
  EXPECT_EQ(p.count(), 153188);
}
TEST(PixelEnvironmentComponents, NatureGraphCountAndLiveFinalHeadDerivative) {
  n::ParameterStore p(1);
  auto              out = m::NaturePixelQ(p, n::zeros({1, 28224}), 5);
  EXPECT_EQ(out.shape(), (n::Shape{1, 5}));
  EXPECT_EQ(p.count(), m::NaturePixelQParameterCount(5));
  auto gradient = n::grad(n::sum(out), {p.at("q/b")})[0];
  EXPECT_EQ(gradient.to_vector<float>(), std::vector<float>(5, 1));
  EXPECT_THROW(m::NaturePixelQ(p, n::zeros({1, 108}), 5), std::invalid_argument);
}
TEST(PixelEnvironmentComponents, NativePolicyUsesSourceNamesAndLiveDensityPartials) {
  n::ParameterStore p(5), ref(5);
  auto              x     = n::ones({2, 4});
  auto              path  = m::NativeGaussianValuePolicy(p, x, 1, 32);
  auto              h     = n::mlp(ref, "actor", x, {32, 32}, n::Activation::Tanh);
  auto              mean  = n::dense(ref, "actor/mean", h, 1);
  auto              ls    = ref.get("actor/log_std", {1}, n::init::constant(-.5));
  auto              hv    = n::mlp(ref, "critic", x, {32, 32}, n::Activation::Tanh);
  auto              value = n::squeeze(n::dense(ref, "critic/value", hv, 1), {1});
  EXPECT_EQ(p.names(), ref.names());
  EXPECT_EQ(path.mean.to_vector<float>(), mean.to_vector<float>());
  EXPECT_EQ(path.value.to_vector<float>(), value.to_vector<float>());
  auto loss =
      n::sum(m::GaussianLogProb(n::zeros({2, 1}), path.mean, path.log_std)) + n::sum(path.value);
  auto grads = n::grad(loss, p.parameters());
  EXPECT_GT(n::global_norm(grads), 0);
}
} // namespace
