/// @file rl_discrete_tests.cpp
/// @brief Analytic masked targets, variable-duration advantages and end-to-end native learners.
#include <gtest/gtest.h>

#include <datamunge/nn/io.hpp>
#include <datamunge/rl/discrete.hpp>
#include <datamunge/sim/fruit_stream.hpp>
#include <filesystem>
#include <limits>

#include "../../src/datamunge/rl/detail.hpp"
namespace r = datamunge::rl;
namespace n = datamunge::nn;
TEST(DiscreteRl, DoubleTargetUsesOnlineChoiceAndRespectsTerminalAndMask) {
  EXPECT_DOUBLE_EQ(
      r::detail::double_target(1, .5, false, {2, 1, 100}, {3, 10, 999}, {1, 1, 0}), 2.5);
  EXPECT_DOUBLE_EQ(r::detail::double_target(1, .25, false, {1, 2}, {9, 4}, {1, 1}), 2);
  EXPECT_DOUBLE_EQ(r::detail::double_target(1, .5, true, {}, {}, {}), 1);
  EXPECT_THROW(r::detail::double_target(1, .5, false, {1}, {2}, {0}), std::invalid_argument);
}
TEST(DiscreteRl, GaeBootstrapsTimeLimitsAndCutsResetTraces) {
  auto adv = r::detail::gae({{1, .5, 2, 4, true}, {100, .9, 0, 0, true}}, 1);
  ASSERT_EQ(adv.size(), 2);
  EXPECT_FLOAT_EQ(adv[0], 1);
  EXPECT_FLOAT_EQ(adv[1], 100);
  auto terminal = r::detail::gae({{1, .5, 2, 0, true}}, 1);
  EXPECT_FLOAT_EQ(terminal[0], -1);
  auto elapsed = r::detail::gae({{1, .25, 0, 2, false}, {1, .5, 2, 4, true}}, .8);
  EXPECT_NEAR(elapsed[0], 1 + .25 * 2 + .25 * .8 * 1, 1e-6);
}
TEST(DiscreteRl, PpoClipsPositiveAndNegativeAdvantagesAndKeepsFiniteEntropy) {
  auto logp = n::from_vector(std::vector<float>{std::log(1.5f), std::log(.5f)});
  logp.requires_grad_(true);
  auto loss =
      r::detail::ppo_loss(logp, n::zeros({2}), n::from_vector(std::vector<float>{1, -1}), .2);
  EXPECT_NEAR(loss.item<float>(), -.2, 1e-6);
  auto grads = n::grad(loss, {logp});
  for (auto v : grads[0].to_vector<float>())
    EXPECT_NEAR(v, 0, 1e-7);
}
TEST(DiscreteRl, PoliciesMaskAndRoundTripAllArchitectures) {
  for (auto method :
       {r::DiscreteMethod::QLambda,
        r::DiscreteMethod::DoubleDqn,
        r::DiscreteMethod::DuelingDoubleDqn,
        r::DiscreteMethod::Ppo}) {
    r::DiscreteModel m;
    m.method   = method;
    m.features = 2;
    m.actions  = 3;
    m.hidden   = 8;
    m.states   = 4;
    r::DiscretePolicy      policy(m);
    r::DiscreteObservation obs{{.2f, .7f}, {0, 1, 0}, 2};
    EXPECT_EQ(policy.action(obs), 1);
    auto scores = policy.scores(obs);
    auto copy   = policy.clone();
    EXPECT_EQ(scores, copy.scores(obs));
    auto path =
        (std::filesystem::temp_directory_path() / ("dm-rl-" + std::to_string(int(method)) + ".npz"))
            .string();
    policy.save(path);
    auto restored = r::DiscretePolicy::load(path);
    EXPECT_EQ(scores, restored.scores(obs));
    std::filesystem::remove(path);
    obs.allowed = {0, 0, 0};
    EXPECT_THROW(policy.action(obs), std::invalid_argument);
    obs.allowed     = {1, 1, 1};
    obs.features[0] = std::numeric_limits<float>::quiet_NaN();
    EXPECT_THROW(policy.scores(obs), std::invalid_argument);
  }
}
TEST(DiscreteRl, LearnersImproveAOneStepBanditWithoutChoosingMaskedActions) {
  for (auto method :
       {r::DiscreteMethod::QLambda,
        r::DiscreteMethod::DoubleDqn,
        r::DiscreteMethod::DuelingDoubleDqn,
        r::DiscreteMethod::Ppo}) {
    r::DiscreteEnvironment env;
    env.reset = [](std::uint64_t) { return r::DiscreteObservation{{1.f}, {1, 1, 0}, 0}; };
    env.step  = [](std::size_t a) {
      EXPECT_LT(a, 2);
      return r::DiscreteTransition{{{1.f}, {0, 0, 0}, 0}, a == 1 ? 1. : -1., .5, true, false};
    };
    r::DiscreteTrainingOptions o;
    o.model.method    = method;
    o.model.features  = 1;
    o.model.actions   = 3;
    o.model.hidden    = 8;
    o.model.states    = 1;
    o.windows         = 384;
    o.batch           = 16;
    o.warmup          = 16;
    o.replay_capacity = 512;
    o.train_every     = 1;
    o.target_every    = 16;
    o.learning_rate   = .003;
    o.window_seconds  = 1;
    auto learned      = r::train_discrete(env, o);
    EXPECT_EQ(learned.windows.size(), 384);
    EXPECT_EQ(learned.policy.action({{1.f}, {1, 1, 0}, 0}), 1) << int(method);
    EXPECT_EQ(learned.checkpoint_windows.back(), 384);
  }
}
TEST(DiscreteRl, ForcedChainsPreserveDiscountedRewardsAndTimeAccounting) {
  int                    step = 0;
  r::DiscreteEnvironment env;
  env.reset = [&](std::uint64_t) {
    step = 0;
    return r::DiscreteObservation{{0.f}, {1, 0}, 0};
  };
  env.step = [&](std::size_t a) {
    EXPECT_EQ(a, 0);
    ++step;
    return r::DiscreteTransition{{{0.f}, {1, 0}, 0}, 1., 1., step == 3, false};
  };
  r::DiscreteTrainingOptions o;
  o.model.method        = r::DiscreteMethod::QLambda;
  o.model.features      = 1;
  o.model.actions       = 2;
  o.model.states        = 1;
  o.windows             = 1;
  o.discount_per_second = .5;
  o.table_learning_rate = 1;
  o.forced_chain        = 8;
  auto trained          = r::train_discrete(env, o);
  EXPECT_EQ(trained.windows[0].environment_steps, 3);
  EXPECT_EQ(trained.windows[0].decisions, 1);
  EXPECT_DOUBLE_EQ(trained.windows[0].seconds, 3);
  EXPECT_DOUBLE_EQ(trained.windows[0].reward, 3);
  EXPECT_FLOAT_EQ(trained.policy.scores({{0.f}, {1, 0}, 0})[0], 1.75);
}
TEST(DiscreteRl, InvalidOptionsAreRejectedBeforeReset) {
  r::DiscreteEnvironment env;
  int                    resets = 0;
  env.reset                     = [&](std::uint64_t) {
    ++resets;
    return r::DiscreteObservation{};
  };
  env.step = [](std::size_t) { return r::DiscreteTransition{}; };
  r::DiscreteTrainingOptions o;
  o.replay_capacity = 64;
  o.warmup          = 65;
  EXPECT_THROW(r::train_discrete(env, o), std::invalid_argument);
  EXPECT_EQ(resets, 0);
}
TEST(DiscreteRl, WorkcellAdapterIsOwnedBoundedAndDoesNotAdvanceTheWorld) {
  const auto urdf = (std::filesystem::path(__FILE__).parent_path().parent_path() /
                     "fixtures/devarm_pick/robot.urdf")
                        .string();
  datamunge::sim::FruitStreamOptions config;
  config.training_proxy     = true;
  config.watch_infeed       = true;
  config.synchronized_grasp = true;
  datamunge::sim::FruitStreamEnvironment env(urdf, config);
  auto callbacks = datamunge::sim::stream_learning_environment(env, .12, .12);
  auto first     = callbacks.reset(123);
  auto before    = env.observation();
  EXPECT_EQ(first.features.size(), 25);
  EXPECT_EQ(first.allowed.size(), 5);
  EXPECT_LT(first.state, 1680);
  for (float x : first.features)
    EXPECT_TRUE(std::isfinite(x) && x >= -1 && x <= 1);
  auto second = env.learning_observation();
  EXPECT_EQ(first.features, second.features);
  EXPECT_EQ(first.allowed, second.allowed);
  EXPECT_DOUBLE_EQ(env.observation().time, before.time);
  EXPECT_EQ(env.observation().arm.joints, before.arm.joints);
  auto step = callbacks.step(0);
  EXPECT_GT(step.seconds, 0);
  EXPECT_FALSE(step.terminated);
  EXPECT_FALSE(step.truncated);
  auto again = callbacks.reset(123);
  EXPECT_EQ(first.features, again.features);
  EXPECT_EQ(first.allowed, again.allowed);
  EXPECT_THROW(datamunge::sim::stream_learning_environment(env, .3, .1), std::invalid_argument);
  EXPECT_THROW(callbacks.step(5), std::invalid_argument);
}

TEST(DiscreteRl, CheckpointsRejectMissingAndNonfiniteParameters) {
  const auto        path = (std::filesystem::temp_directory_path() / "dm-rl-invalid.npz").string();
  r::DiscretePolicy policy;
  policy.save(path);
  auto values     = n::load_npz(path);
  auto incomplete = values;
  incomplete.pop_back();
  n::save_npz(path, incomplete);
  EXPECT_THROW(r::DiscretePolicy::load(path), std::runtime_error);
  for (auto& entry : values)
    if (entry.first != "architecture") {
      auto data    = entry.second.to_vector<float>();
      data[0]      = std::numeric_limits<float>::infinity();
      entry.second = n::from_vector(data, entry.second.shape());
      break;
    }
  n::save_npz(path, values);
  EXPECT_THROW(r::DiscretePolicy::load(path), std::runtime_error);
  std::filesystem::remove(path);
}
TEST(DiscreteLearning, WindowStopPreservesActualHistoryAndFinalCheckpoint) {
  namespace r = datamunge::rl;
  r::DiscreteEnvironment env;
  env.reset = [](std::uint64_t) { return r::DiscreteObservation{{0.f}, {1, 1}, 0}; };
  env.step  = [](std::size_t action) {
    return r::DiscreteTransition{{{0.f}, {1, 1}, 0}, action == 1 ? 1. : 0., 1., true, false};
  };
  r::DiscreteTrainingOptions options;
  options.model.method   = r::DiscreteMethod::QLambda;
  options.model.features = 1;
  options.model.actions  = 2;
  options.model.states   = 1;
  options.windows        = 20;
  options.window_seconds = 1;
  int progress           = 0;
  options.progress       = [&](auto, const auto&, const auto&) { ++progress; };
  options.stop           = [](auto completed, const auto&, const auto&) { return completed == 3; };
  auto trained           = r::train_discrete(env, options);
  EXPECT_EQ(progress, 3);
  EXPECT_EQ(trained.windows.size(), 3u);
  ASSERT_EQ(trained.checkpoint_windows.size(), 1u);
  EXPECT_EQ(trained.checkpoint_windows[0], 3u);
  EXPECT_EQ(
      trained.checkpoints[0].scores({{0.f}, {1, 1}, 0}), trained.policy.scores({{0.f}, {1, 1}, 0}));
}
