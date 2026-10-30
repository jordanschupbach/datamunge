#include <gtest/gtest.h>

#include "common/rl/native_tasks.hpp"
namespace s = datamunge::sim;
TEST(NativeRlAdapters, SingleTaskPreservesFinalStateAndExplicitReset) {
  rl::NativeTaskEnv env(s::Task::InvertedPendulum, 1);
  const auto        initial = env.reset(42);
  EXPECT_EQ(env.obs_size(), 4);
  EXPECT_EQ(env.action_space().high, (std::vector<float>{3}));
  const auto out = env.step(rl::Action{0.f});
  EXPECT_TRUE(out.truncated);
  EXPECT_FALSE(out.terminated);
  EXPECT_EQ(out.reward, 1);
  EXPECT_THROW(env.step(rl::Action{0.f}), std::logic_error);
  EXPECT_EQ(env.reset(42), initial);
  EXPECT_THROW(env.step(rl::Action{0.f, 1.f}), std::invalid_argument);
  EXPECT_THROW(rl::NativeTask("unknown"), std::invalid_argument);
  char  executable[] = "example", device[] = "device=cuda";
  char* arguments[] = {executable, device};
  EXPECT_THROW(rl::ValidateNativeDeviceArguments(2, arguments), std::invalid_argument);
  EXPECT_THROW(rl::NativeTaskEnv(s::Task::Hopper, 0), std::invalid_argument);
}
TEST(NativeRlAdapters, BatchAutoResetRetainsActualBootstrapStates) {
  rl::NativeVectorTask env(s::Task::InvertedPendulum, 2, 42, 2, 1);
  s::TaskBatch reference(s::Task::InvertedPendulum, 2, datamunge::compute::Context({}, 2), 42, 1);
  env.reset();
  reference.reset();
  const auto expected = reference.step({0, 0});
  const auto first    = env.step({rl::Action{0.f}, rl::Action{0.f}});
  for (std::size_t i = 0; i < 2; ++i) {
    EXPECT_TRUE(first.truncated[i]);
    EXPECT_FALSE(first.terminated[i]);
    EXPECT_EQ(first.final_obs[i], rl::NativeObservation(expected.final_observations, i, 4));
  }
  const auto& next = reference.reset();
  for (std::size_t i = 0; i < 2; ++i)
    EXPECT_EQ(first.obs[i], rl::NativeObservation(next.observations, i, 4));
  EXPECT_NO_THROW(env.step({rl::Action{0.f}, rl::Action{0.f}}));
  EXPECT_EQ(first.final_obs[0], rl::NativeObservation(expected.final_observations, 0, 4));
  EXPECT_THROW(env.step({rl::Action{0.f}}), std::invalid_argument);
  EXPECT_THROW(rl::NativeVectorTask(s::Task::Hopper, 0, 0), std::invalid_argument);
}
TEST(NativeRlAdapters, GaeBootstrapsTruncationAndCutsResetTraces) {
  const auto truncated = rl::NativeGae({1, 1, 1}, {10, 20, 100}, {20, 30, 200}, {0, 1, 0}, .9f, 1);
  EXPECT_NEAR(truncated.first[0], 16.2, 1e-5);
  EXPECT_NEAR(truncated.first[1], 8, 1e-5);
  EXPECT_NEAR(truncated.first[2], 81, 1e-5);
  EXPECT_NEAR(truncated.second[1], 28, 1e-5);
  const auto terminal = rl::NativeGae({1, 1, 1}, {10, 20, 100}, {20, 0, 200}, {0, 1, 0}, .9f, 1);
  EXPECT_NEAR(terminal.first[0], -8.1, 1e-5);
  EXPECT_NEAR(terminal.second[1], 1, 1e-5);
  EXPECT_THROW(rl::NativeGae({1}, {}, {}, {}, .9f, 1), std::invalid_argument);
  EXPECT_THROW(rl::NativeGae({1}, {1}, {1}, {2}, .9f, 1), std::invalid_argument);
}
