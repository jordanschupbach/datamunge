#include <gtest/gtest.h>

#include <cmath>
#include <datamunge/sim/devarm_pick.hpp>
#include <filesystem>
#include <fstream>
#include <limits>
namespace s = datamunge::sim;
namespace {
std::string fixture() {
  return (std::filesystem::path(__FILE__).parent_path().parent_path() /
          "fixtures/devarm_pick/robot.urdf")
      .string();
}
void descend(s::DevArmPickEnvironment& e) {
  for (int i = 0; i < 4; ++i)
    e.step(s::PickAction::ZMinus);
}
void transport(s::DevArmPickEnvironment& e) {
  for (int i = 0; i < 2; ++i)
    e.step(s::PickAction::ZPlus);
  for (int i = 0; i < 14; ++i)
    e.step(s::PickAction::YMinus);
}
} // namespace
TEST(DevArmPick, DeterministicResetUsesRealJointLimitsAndNoAutomaticGrasp) {
  s::DevArmPickEnvironment e(fixture());
  for (unsigned seed : {0, 3}) {
    auto a = e.reset(seed);
    e.step(s::PickAction::Close);
    EXPECT_FALSE(e.observation().grasped);
    auto b = e.reset(seed);
    EXPECT_EQ(a.joints, b.joints);
    EXPECT_DOUBLE_EQ(a.fruit.x, b.fruit.x);
    EXPECT_DOUBLE_EQ(a.fruit.y, b.fruit.y);
    EXPECT_EQ(a.kind, b.kind);
    EXPECT_FALSE(a.success);
    EXPECT_LT((a.tool - a.fruit).Length(), .16);
  }
  auto before = e.observation();
  EXPECT_THROW(e.step(static_cast<s::PickAction>(99)), std::invalid_argument);
  EXPECT_EQ(before.joints, e.observation().joints);
  EXPECT_EQ(before.time, e.observation().time);
  EXPECT_THROW(s::DevArmPickEnvironment(fixture(), 0), std::invalid_argument);
}
TEST(DevArmPick, GraspLiftReleaseAndRedFloorContactRequiredForBothFruitClasses) {
  s::DevArmPickEnvironment e(fixture());
  bool                     apple = false, banana = false;
  for (unsigned seed : {0, 3}) {
    auto start = e.reset(seed);
    apple |= start.kind == s::FruitKind::Apple;
    banana |= start.kind == s::FruitKind::Banana;
    descend(e);
    auto captured = e.step(s::PickAction::Close);
    ASSERT_TRUE(captured.grasped);
    EXPECT_FALSE(captured.success);
    // This fixed sequence is an environment regression oracle, never a training policy.
    transport(e);
    auto above = e.observation();
    EXPECT_FALSE(above.success);
    EXPECT_TRUE(above.grasped);
    EXPECT_GT(above.fruit.z, .72);
    EXPECT_NEAR(above.fruit.y, -.42, .015);
    auto released = e.step(s::PickAction::Open);
    EXPECT_TRUE(released.released);
    EXPECT_FALSE(released.grasped);
    EXPECT_TRUE(released.terminated);
    EXPECT_TRUE(released.success);
    EXPECT_LT(released.fruit.z, .55);
    EXPECT_GT(released.reward, 8);
    EXPECT_THROW(e.step(s::PickAction::Wait), std::logic_error);
  }
  EXPECT_TRUE(apple);
  EXPECT_TRUE(banana);
}
TEST(DevArmPick, BlockedPayloadNeedsLiftAndWrongReleaseFails) {
  s::DevArmPickEnvironment e(fixture());
  e.reset(0);
  descend(e);
  ASSERT_TRUE(e.step(s::PickAction::Close).grasped);
  bool blocked = false;
  for (int i = 0; i < 10; ++i)
    blocked |= e.step(s::PickAction::YMinus).collision;
  EXPECT_TRUE(blocked);
  EXPECT_GT(e.observation().fruit.y, -.20);
  auto failed = e.step(s::PickAction::Open);
  EXPECT_TRUE(failed.terminated);
  EXPECT_FALSE(failed.success);
  EXPECT_LT(failed.reward, 0);
}
TEST(DevArmPick, RecordedJointAndFingerRatesAndTruncationAreExplicit) {
  s::DevArmPickEnvironment        e(fixture(), 1);
  std::vector<s::PickObservation> frames;
  e.record([&](const auto& f) { frames.push_back(f); });
  e.reset(42);
  auto last = e.step(s::PickAction::ZMinus);
  EXPECT_TRUE(last.truncated);
  EXPECT_FALSE(last.terminated);
  ASSERT_GT(frames.size(), 1);
  for (size_t i = 1; i < frames.size(); ++i) {
    double dt = frames[i].time - frames[i - 1].time;
    ASSERT_GT(dt, 0);
    for (int j = 0; j < 6; ++j)
      EXPECT_LE(std::abs(frames[i].joints[j] - frames[i - 1].joints[j]) / dt, 1.500001);
    EXPECT_LE(std::abs(frames[i].finger - frames[i - 1].finger) / dt, .080001);
  }
  EXPECT_THROW(e.step(s::PickAction::Wait), std::logic_error);
}
TEST(DevArmPick, CheckpointRoundTripRejectsMalformedAndNonfinitePolicies) {
  auto path = (std::filesystem::temp_directory_path() / "datamunge-pick-policy-test.txt").string();
  s::PickPolicy p;
  p.q.assign(s::DevArmPickEnvironment::state_count(), std::vector<double>(9));
  p.q[5][7] = 1.23456789012345;
  p.save(path);
  auto loaded = s::PickPolicy::load(path);
  EXPECT_EQ(p.q, loaded.q);
  EXPECT_EQ(loaded.action(5), s::PickAction::Open);
  EXPECT_THROW(loaded.action(loaded.q.size()), std::out_of_range);
  {
    std::ofstream out(path);
    out << "DATAMUNGE_DEVARM_Q 1 99999999 9\n";
  }
  EXPECT_THROW(s::PickPolicy::load(path), std::runtime_error);
  p.q[5][7] = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(p.save(path), std::invalid_argument);
  EXPECT_THROW(p.action(5), std::invalid_argument);
  std::filesystem::remove(path);
}
TEST(DevArmPick, TrainingUsesTransitionsAndProducesPortableFiniteValues) {
  s::DevArmPickEnvironment e(fixture(), 25);
  auto                     trained = s::train_red_bin(e, 12, 42);
  EXPECT_EQ(trained.returns.size(), 12);
  EXPECT_EQ(trained.successes.size(), 12);
  EXPECT_EQ(trained.policy.q.size(), e.state_count());
  bool changed = false;
  for (auto& row : trained.policy.q)
    for (double v : row) {
      EXPECT_TRUE(std::isfinite(v));
      changed |= v != 0;
    }
  EXPECT_TRUE(changed);
  EXPECT_THROW(s::train_red_bin(e, 0), std::invalid_argument);
}
