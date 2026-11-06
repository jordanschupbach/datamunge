#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <datamunge/sim/devarm_camera.hpp>
#include <datamunge/sim/fruit_stream.hpp>
#include <filesystem>
#include <limits>
#include <map>
#include <numeric>
namespace s = datamunge::sim;
namespace {
std::string fixture() {
  return (std::filesystem::path(__FILE__).parent_path().parent_path() /
          "fixtures/devarm_pick/robot.urdf")
      .string();
}
std::uint64_t unresolved(const s::StreamFrame& f) {
  std::uint64_t n = 0;
  for (auto& p : f.fruit)
    if (p.status == s::StreamFruitStatus::Belt || p.status == s::StreamFruitStatus::Held ||
        p.status == s::StreamFruitStatus::Released)
      ++n;
  return n;
}
} // namespace
TEST(FruitStream, PoissonArrivalsAreIndependentOfBusyArmAndContinueWithoutReset) {
  s::FruitStreamOptions options;
  options.arrival_rate   = .5;
  options.training_error = {0, 0};
  s::FruitStreamEnvironment                                                env(fixture(), options);
  std::array<std::map<std::uint64_t, std::pair<double, std::uint64_t>>, 2> arrivals;
  std::array<s::StreamFrame, 2>                                            endpoint;
  for (int method = 0; method < 2; ++method) {
    env.record(
        [&](const auto& f) {
          if (f.time > 60 + 1e-9)
            return;
          endpoint[method] = f;
          for (auto& p : f.fruit)
            arrivals[method][p.id] = {p.arrival, p.seed};
        },
        15);
    auto f = env.reset(20261131);
    EXPECT_TRUE(f.fruit.empty());
    EXPECT_EQ(f.time, 0);
    while (f.time < 60)
      f = env.step(method == 0 ? s::StreamAction::Wait : s::StreamAction::LateFront).frame;
    EXPECT_DOUBLE_EQ(endpoint[method].time, 60);
    EXPECT_LE(f.fruit.size(), 32);
    EXPECT_EQ(
        f.counts.arrivals, f.counts.correct + f.counts.wrong + f.counts.missed + unresolved(f));
    const auto previous = f.counts.arrivals;
    while (f.time < 90)
      f = env.step(s::StreamAction::Wait).frame;
    EXPECT_GE(f.counts.arrivals, previous);
    EXPECT_GT(f.time, 89);
  }
  EXPECT_EQ(arrivals[0], arrivals[1]);
  EXPECT_GT(arrivals[0].size(), 15);
  EXPECT_EQ(endpoint[0].counts.arrivals, endpoint[1].counts.arrivals);
  EXPECT_GT(endpoint[0].counts.missed, 0);
  std::vector<double> gaps;
  double              previous = 0;
  for (auto& [id, value] : arrivals[0]) {
    gaps.push_back(value.first - previous);
    previous = value.first;
  }
  double mean = std::accumulate(gaps.begin(), gaps.end(), 0.) / gaps.size(), variance = 0;
  for (double x : gaps)
    variance += (x - mean) * (x - mean);
  variance /= gaps.size();
  EXPECT_GT(variance, .4 * mean * mean);
  EXPECT_LT(variance, 2.5 * mean * mean);
}
TEST(FruitStream, CapacityCountsOverflowAndUnknownVisionNeverRoutesUsingTruth) {
  s::FruitStreamOptions o;
  o.arrival_rate    = 5;
  o.active_capacity = 1;
  s::FruitStreamEnvironment env(fixture(), o);
  auto                      f = env.reset(74);
  while (f.time < 20)
    f = env.step(s::StreamAction::Wait).frame;
  EXPECT_GT(f.counts.arrivals, 50);
  EXPECT_GT(f.counts.overflow, 0);
  EXPECT_LE(f.fruit.size(), 1);
  EXPECT_EQ(f.counts.arrivals, f.counts.missed + unresolved(f));
  env.set_perception([](const auto&, std::uint64_t) { return s::StreamPrediction{-1, 0}; });
  f = env.reset(123, .2);
  while (f.time < 60)
    f = env.step(s::StreamAction::LateFront).frame;
  EXPECT_FALSE(f.synthetic_perception);
  EXPECT_GT(f.counts.unknown, 0);
  EXPECT_EQ(f.counts.grasped, 0);
  EXPECT_EQ(f.counts.correct, 0);
  EXPECT_GT(f.counts.missed, 0);
  auto before = f.time;
  EXPECT_THROW(env.reset(0, -1), std::invalid_argument);
  EXPECT_DOUBLE_EQ(env.observation().time, before);
  EXPECT_THROW(env.record({}, 7), std::invalid_argument);
  EXPECT_THROW(env.step(static_cast<s::StreamAction>(99)), std::invalid_argument);
}
TEST(FruitStream, RecordedServosAreContinuousAndBodiesOnlyScoreAfterPhysicalRelease) {
  s::FruitStreamOptions o;
  o.arrival_rate   = .14;
  o.training_error = {0, 0};
  s::FruitStreamEnvironment env(fixture(), o);
  s::StreamFrame            previous;
  bool                      seen = false, arrived_while_busy = false;
  env.record(
      [&](const auto& f) {
        if (seen) {
          double dt = f.time - previous.time;
          if (dt > 0) {
            for (int k = 0; k < 6; ++k)
              EXPECT_LE(std::abs(f.arm.joints[k] - previous.arm.joints[k]) / dt, 1.50001);
            EXPECT_LE(std::abs(f.arm.finger - previous.arm.finger) / dt, .08001);
          }
          if (f.target != 0 && f.counts.arrivals > previous.counts.arrivals)
            arrived_while_busy = true;
        }
        for (const auto& p : f.fruit)
          if (p.status == s::StreamFruitStatus::Correct) {
            EXPECT_LT(p.body.position[1], .57);
            EXPECT_GT(p.body.position[1], .44);
            EXPECT_GT(std::abs(p.body.position[2]), .30);
          }
        previous = f;
        seen     = true;
      },
      24);
  auto f = env.reset(2026110901);
  while (f.time < 90)
    f = env.step(s::StreamAction::LateFront).frame;
  EXPECT_GT(f.counts.grasped, 3);
  EXPECT_GT(f.counts.correct, 2);
  EXPECT_TRUE(arrived_while_busy);
}
TEST(FruitStream, NewControllerTrainsAndCheckpointRestoresAllFiniteValues) {
  s::FruitStreamEnvironment env(fixture());
  s::StreamTrainingOptions  o;
  o.rollouts   = 8;
  o.seconds    = 10;
  auto trained = s::train_fruit_stream(env, o);
  EXPECT_EQ(trained.counts.size(), 8);
  ASSERT_EQ(trained.policy.q.size(), 1680);
  auto path = (std::filesystem::temp_directory_path() / "datamunge-stream-test.txt").string();
  trained.policy.save(path);
  auto restored = s::StreamPolicy::load(path);
  EXPECT_EQ(restored.q, trained.policy.q);
  EXPECT_THROW(restored.action(1680), std::out_of_range);
  std::filesystem::remove(path);
  auto before = env.observation().time;
  o.seconds   = 0;
  EXPECT_THROW(s::train_fruit_stream(env, o), std::invalid_argument);
  EXPECT_DOUBLE_EQ(before, env.observation().time);
}
TEST(FruitStream, MultiActorCameraUsesActualRgbAndTrackedFocusWithoutChangingArmPose) {
  s::FruitStreamOptions options;
  options.arrival_rate = .5;
  s::FruitStreamEnvironment env(fixture(), options);
  s::DevArmWristCamera      camera(env.arm_model(), 128, 95);
  auto                      f = env.reset(20261131);
  while (f.time < 8)
    f = env.step(s::StreamAction::Wait).frame;
  auto actors  = env.fruit_scene(f);
  auto raw     = camera.capture_actors(f.arm, actors);
  auto outside = camera.inspect_actors(f.arm, actors, {100, 100, 100});
  EXPECT_EQ(raw.rgba, outside.rgb.rgba);
  EXPECT_TRUE(outside.crop.rgba.empty());
  auto initial = env.arm_model().observation().joints;
  auto target  = env.arm_model().joint_target(0, 0, 4);
  EXPECT_EQ(initial, env.arm_model().observation().joints);
  EXPECT_EQ(target.size(), 6);
  EXPECT_THROW(env.arm_model().joint_target(3, 0, 0), std::invalid_argument);
}

TEST(FruitStream, TrainingProxyIsExplicitAndCannotProduceAnEvaluationRecording) {
  s::FruitStreamOptions o;
  o.training_proxy = true;
  s::FruitStreamEnvironment env(fixture(), o);
  EXPECT_TRUE(env.reset(71).training_proxy);
  EXPECT_THROW(env.record([](const auto&) {}), std::logic_error);
  auto f = env.observation();
  while (f.time < 20)
    f = env.step(s::StreamAction::EarlyFront).frame;
  EXPECT_EQ(f.counts.arrivals, f.counts.correct + f.counts.wrong + f.counts.missed + unresolved(f));
}
TEST(FruitStream, NativeVideoTextClipsPixelsAndPreservesGeometryBuffers) {
  datamunge::cad::RenderResult image;
  image.width  = 32;
  image.height = 20;
  image.rgba.resize(32 * 20 * 4);
  image.depth.assign(32 * 20, 4);
  const auto depth = image.depth;
  datamunge::cad::draw_text(image, "A 1", -2, 2);
  EXPECT_TRUE(std::any_of(image.rgba.begin(), image.rgba.end(), [](auto v) { return v > 0; }));
  EXPECT_EQ(image.depth, depth);
  const auto before = image.rgba;
  EXPECT_THROW(datamunge::cad::draw_text(image, "B", 0, 0, 0), std::invalid_argument);
  EXPECT_EQ(before, image.rgba);
}
TEST(FruitStream, AddedVisualProfilePreservesLegacyConfigurationDefaults) {
  s::FruitStreamOptions legacy{false, .14, 32, {.02, .1}};
  EXPECT_EQ(legacy.visual_style, s::WorkcellVisualStyle::Simple);
  EXPECT_DOUBLE_EQ(legacy.arrival_rate, .14);
  EXPECT_FALSE(legacy.watch_infeed);
  EXPECT_FALSE(legacy.synchronized_grasp);
  EXPECT_DOUBLE_EQ(legacy.queued_wait_cost, 0);
  EXPECT_DOUBLE_EQ(legacy.miss_cost, 3);
  EXPECT_DOUBLE_EQ(legacy.wrong_cost, 5);
  EXPECT_DOUBLE_EQ(legacy.grasp_alignment_tolerance, .012);
  EXPECT_FALSE(legacy.hold_grasp_target);
}
TEST(FruitStream, GraspAlignmentToleranceIsValidatedBeforeConstruction) {
  s::FruitStreamOptions o;
  for (double bad : {.004, .051, std::nan("")}) {
    o.grasp_alignment_tolerance = bad;
    EXPECT_THROW(s::FruitStreamEnvironment(fixture(), o), std::invalid_argument);
  }
  o.grasp_alignment_tolerance = .02;
  o.hold_grasp_target         = true;
  EXPECT_NO_THROW(s::FruitStreamEnvironment(fixture(), o));
}

TEST(FruitStream, FeasibleMaskIgnoresMissingTargetsAndExpiresCachedDeadlinesWithoutSideEffects) {
  s::FruitStreamOptions o;
  o.watch_infeed       = true;
  o.synchronized_grasp = true;
  o.arrival_rate       = .12;
  s::FruitStreamEnvironment env(fixture(), o), control(fixture(), o);
  auto                      f     = env.reset(2026113001);
  auto                      other = control.reset(2026113001);
  EXPECT_EQ(env.action_mask(), (std::array<bool, 5>{true, false, false, false, false}));
  while (f.fruit.empty()) {
    const auto before = env.observation();
    EXPECT_EQ(env.action_mask(), env.action_mask());
    EXPECT_EQ(before.time, env.observation().time);
    EXPECT_EQ(before.arm.joints, env.observation().arm.joints);
    f     = env.step(s::StreamAction::Wait).frame;
    other = control.step(s::StreamAction::Wait).frame;
  }
  ASSERT_EQ(f.fruit.size(), 1);
  EXPECT_EQ(f.fruit.front().seed, other.fruit.front().seed);
  EXPECT_EQ(f.fruit.front().body.position, other.fruit.front().body.position);
  auto mask = env.action_mask();
  EXPECT_TRUE(mask[1] || mask[2]);
  EXPECT_FALSE(mask[3]);
  EXPECT_FALSE(mask[4]);
  while (f.time < 11.8)
    f = env.step(s::StreamAction::Wait).frame;
  EXPECT_EQ(env.action_mask(), (std::array<bool, 5>{true, false, false, false, false}));
  env.reset(2026113001);
  EXPECT_EQ(env.action_mask(), (std::array<bool, 5>{true, false, false, false, false}));
}

TEST(FruitStream, SynchronizedWristKeepsInspectionCloseToTheGraspAndUsesRealServoLimits) {
  s::FruitStreamOptions o;
  o.watch_infeed       = true;
  o.synchronized_grasp = true;
  o.arrival_rate       = .12;
  s::FruitStreamEnvironment env(fixture(), o);
  bool                      inspected = false, latched = false;
  double                    roll = 0;
  env.set_perception([&](const auto& frame, std::uint64_t id) {
    const auto& p = *std::find_if(
        frame.fruit.begin(), frame.fruit.end(), [&](const auto& p) { return p.id == id; });
    EXPECT_LT(std::abs(p.body.position[0] - frame.arm.tool.x), .08);
    inspected = true;
    roll      = frame.arm.joints[5];
    return s::StreamPrediction{1, 1};
  });
  s::StreamFrame previous;
  bool           seen = false;
  env.record(
      [&](const auto& frame) {
        if (seen && frame.time > previous.time)
          for (int i = 0; i < 6; ++i)
            EXPECT_LE(
                std::abs(frame.arm.joints[i] - previous.arm.joints[i]) /
                    (frame.time - previous.time),
                1.50001);
        if (inspected && frame.arm.grasped && !latched) {
          EXPECT_NEAR(frame.arm.joints[5], roll, 1e-8);
          latched = true;
        }
        previous = frame;
        seen     = true;
      },
      60);
  auto f = env.reset(2026113001);
  while (f.time < 20 && !f.counts.grasped)
    f = env.step(s::StreamAction::LateFront).frame;
  EXPECT_TRUE(inspected);
  EXPECT_TRUE(latched);
  EXPECT_EQ(f.counts.grasped, 1);
}

TEST(FruitStream, MaskedTraceLearningBootstrapsOnlyAllowedActionsAndRejectsInvalidSettings) {
  s::FruitStreamOptions config;
  config.training_proxy = true;
  s::FruitStreamEnvironment env(fixture(), config);
  s::StreamTrainingOptions  o;
  o.rollouts = 8;
  o.seconds  = 10;
  o.rate_min = o.rate_max = 0;
  o.mask_actions          = true;
  o.trace_decay           = .8;
  auto trained            = s::train_fruit_stream(env, o);
  auto repeated           = s::train_fruit_stream(env, o);
  EXPECT_EQ(trained.policy.q, repeated.policy.q);
  const auto empty = env.reset(42, 0).state;
  EXPECT_LT(trained.policy.q[empty][0], -.03);
  for (int a = 1; a < 5; ++a)
    EXPECT_DOUBLE_EQ(trained.policy.q[empty][a], 0);
  const auto& p = trained.policy;
  EXPECT_EQ(p.action(empty, {true, false, false, false, false}), s::StreamAction::Wait);
  EXPECT_THROW(p.action(empty, {false, false, false, false, false}), std::invalid_argument);
  const double time = env.observation().time;
  o.trace_decay     = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(s::train_fruit_stream(env, o), std::invalid_argument);
  EXPECT_DOUBLE_EQ(env.observation().time, time);
}

TEST(FruitStream, QueuedWaitingHasConfigurableCostAndInvalidRewardsFailConstruction) {
  s::FruitStreamOptions o;
  o.arrival_rate = .12;
  s::FruitStreamEnvironment baseline(fixture(), o);
  o.queued_wait_cost = .5;
  s::FruitStreamEnvironment penalized(fixture(), o);
  auto                      f = baseline.reset(2026113001);
  penalized.reset(2026113001);
  while (f.fruit.empty()) {
    f = baseline.step(s::StreamAction::Wait).frame;
    penalized.step(s::StreamAction::Wait);
  }
  auto a = baseline.step(s::StreamAction::Wait), b = penalized.step(s::StreamAction::Wait);
  EXPECT_NEAR(a.reward - b.reward, .5 * .125, 1e-12);
  EXPECT_EQ(a.frame.fruit.front().body.position, b.frame.fruit.front().body.position);
  o.miss_cost = -1;
  EXPECT_THROW(s::FruitStreamEnvironment(fixture(), o), std::invalid_argument);
  o.miss_cost  = 6;
  o.wrong_cost = 101;
  EXPECT_THROW(s::FruitStreamEnvironment(fixture(), o), std::invalid_argument);
}

TEST(FruitStream, MaskQueriesDuringUnknownPerceptionDoNotLeaveARejectedTargetFeasible) {
  s::FruitStreamOptions o;
  o.arrival_rate = .12;
  o.watch_infeed = o.synchronized_grasp = true;
  s::FruitStreamEnvironment env(fixture(), o);
  bool                      queried = false;
  env.set_perception([&](const auto&, std::uint64_t) {
    env.action_mask();
    queried = true;
    return s::StreamPrediction{-1, 0};
  });
  auto f = env.reset(2026113001);
  while (f.time < 20 && !queried)
    f = env.step(s::StreamAction::LateFront).frame;
  ASSERT_TRUE(queried);
  ASSERT_EQ(f.counts.unknown, 1);
  EXPECT_EQ(env.action_mask(), (std::array<bool, 5>{true, false, false, false, false}));
}

TEST(FruitStream, InfeedWatchUsesRigidCameraAndPreservesWaitClockAndArrivalSeeds) {
  std::array<std::map<std::uint64_t, std::pair<double, std::uint64_t>>, 2> births;
  for (int watch = 0; watch < 2; ++watch) {
    s::FruitStreamOptions o;
    o.watch_infeed = watch;
    o.arrival_rate = .12;
    s::FruitStreamEnvironment env(fixture(), o);
    s::DevArmWristCamera      camera(env.arm_model(), 128, 95);
    auto                      f      = env.reset(2026113001);
    const auto                joints = f.arm.joints;
    if (watch) {
      const auto view = camera.pose(f.arm);
      const auto axis = (view.target - view.position).Normalized();
      EXPECT_GT(f.arm.tool.z, .95);
      for (double x : {-1., -.7})
        for (double y : {-.065, .065})
          for (double z : {.745, .84})
            EXPECT_GT(axis.Dot((datamunge::cad::Vec3d{x, y, z} - view.position).Normalized()), .97);
    }
    while (f.time < 40) {
      const double before = f.time;
      // Missing-target choices must also keep watch, without a long blocking return.
      f = env.step(f.fruit.empty() ? s::StreamAction::EarlyFront : s::StreamAction::Wait).frame;
      EXPECT_DOUBLE_EQ(f.time - before, .125);
      EXPECT_EQ(f.arm.joints, joints);
      for (const auto& p : f.fruit)
        births[watch][p.id] = {p.arrival, p.seed};
    }
    EXPECT_EQ(f.arm.joints, env.reset(2026113001).arm.joints);
  }
  EXPECT_EQ(births[0], births[1]);
  EXPECT_EQ(births[1].size(), 2);
}

TEST(FruitStream, InfeedWatchReturnsFromBothBinsWithContinuousServosAndFixedWaitBudget) {
  for (bool proxy : {false, true}) {
    for (int destination : {0, 1}) {
      s::FruitStreamOptions o;
      o.training_proxy     = proxy;
      o.synchronized_grasp = true;
      o.watch_infeed       = true;
      o.arrival_rate       = .12;
      s::FruitStreamEnvironment env(fixture(), o);
      env.set_perception([&](const auto&, auto) { return s::StreamPrediction{destination, 1}; });
      s::StreamFrame previous;
      bool           seen = false;
      if (!proxy)
        env.record(
            [&](const auto& f) {
              if (seen && f.time > previous.time) {
                const double dt = f.time - previous.time;
                for (int i = 0; i < 6; ++i)
                  EXPECT_LE(std::abs(f.arm.joints[i] - previous.arm.joints[i]) / dt, 1.50001);
                EXPECT_LE(std::abs(f.arm.finger - previous.arm.finger) / dt, .08001);
              }
              previous = f;
              seen     = true;
            },
            60);
      auto       f     = env.reset(2026113001);
      const auto watch = f.arm.joints;
      while (f.time < 25 && !f.counts.grasped)
        f = env.step(s::StreamAction::LateFront).frame;
      ASSERT_EQ(f.counts.grasped, 1);
      EXPECT_EQ(f.target, 0);
      EXPECT_GT((destination ? 1 : -1) * f.arm.tool.y, .3);
      EXPECT_NE(f.arm.joints, watch);
      for (int k = 0; k < 60; ++k) {
        const auto transition = env.step(s::StreamAction::Wait);
        EXPECT_NEAR(transition.seconds, .125, 1e-12);
        f = transition.frame;
      }
      EXPECT_EQ(f.arm.joints, watch);
      EXPECT_EQ(f.counts.correct + f.counts.wrong, 1);
      EXPECT_FALSE(f.arm.grasped);
      // A subsequent fruit interrupts surveillance and runs the requested pick option.
      while (f.time < 44 && f.counts.grasped < 2)
        f = env.step(s::StreamAction::LateFront).frame;
      EXPECT_GE(f.counts.grasped, 2);
    }
  }
}
