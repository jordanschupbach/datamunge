#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <datamunge/sim/tasks.hpp>
namespace s = datamunge::sim;
TEST(SimTasks, PendulumShapesMasksAndExplicitTimeLimit) {
  s::TaskBatch task(s::Task::InvertedPendulum, 3, datamunge::compute::Context({}, 2), 42, 2);
  EXPECT_EQ(task.observation_size(), 4);
  EXPECT_EQ(task.action_size(), 1);
  const auto initial = task.output().observations;
  EXPECT_EQ(initial.size(), 12);
  task.step({0, 0, 0}, {1, 0, 0});
  EXPECT_EQ(task.output().observations[4], initial[4]);
  task.step({0, 0, 0}, {1, 0, 0});
  const auto result = task.output();
  EXPECT_EQ(result.terminated[0], 0);
  EXPECT_EQ(result.truncated[0], 1);
  EXPECT_EQ(result.final_valid[0], 1);
  EXPECT_EQ(result.final_valid[1], 0);
  EXPECT_EQ(result.rewards[0], 1);
  EXPECT_TRUE(
      std::equal(
          result.observations.begin(),
          result.observations.begin() + 4,
          result.final_observations.begin()));
  EXPECT_THROW(task.step({0, 0, 0}), std::logic_error);
  task.reset({1, 0, 0});
  EXPECT_EQ(task.output().truncated[0], 0);
  EXPECT_EQ(task.output().rewards[0], 0);
  EXPECT_EQ(task.output().final_observations, result.final_observations);
  EXPECT_EQ(task.output().observations[4], initial[4]);
}
TEST(SimTasks, ActionsAreValidatedBeforeAnyForcesAndReseedingIsRepeatable) {
  s::TaskBatch task(s::Task::InvertedPendulum, 2), reference(s::Task::InvertedPendulum, 2);
  EXPECT_THROW(task.step({1, NAN}), std::invalid_argument);
  EXPECT_THROW(task.step({0}), std::invalid_argument);
  EXPECT_THROW(task.step({0, 4}), std::invalid_argument);
  EXPECT_THROW(task.reset({2, 0}), std::invalid_argument);
  EXPECT_EQ(task.output().observations, reference.output().observations);
  EXPECT_EQ(task.step({0, 0}).observations, reference.step({0, 0}).observations);
  task.seed(123);
  reference.seed(123);
  EXPECT_EQ(task.reset().observations, reference.reset().observations);
}
TEST(SimTasks, CartForceMovesRealArticulatedBodiesAndFailureIsTerminal) {
  s::TaskBatch task(s::Task::InvertedPendulum, 1);
  const auto   initial = task.world(0).body(1).position[0];
  bool         done    = false;
  for (int i = 0; i < 100 && !done; ++i) {
    const auto& r = task.step({3});
    done          = r.terminated[0];
  }
  EXPECT_TRUE(done);
  EXPECT_GT(task.world(0).body(1).position[0], initial + .01);
  EXPECT_EQ(task.output().truncated[0], 0);
  EXPECT_EQ(task.output().final_valid[0], 1);
}
TEST(SimTasks, HopperHasRealFloorContactsJointsAndTorqueResponse) {
  s::TaskBatch zero(s::Task::Hopper, 1), torque(s::Task::Hopper, 1);
  EXPECT_EQ(zero.observation_size(), 11);
  EXPECT_EQ(zero.action_size(), 3);
  EXPECT_EQ(zero.world(0).body_count(), 5);
  EXPECT_EQ(zero.world(0).joint_count(), 3);
  zero.step({0, 0, 0});
  torque.step({.1, 0, 0});
  EXPECT_NE(zero.output().observations[2], torque.output().observations[2]);
  bool contact = false;
  for (int i = 0; i < 100; ++i) {
    if (zero.output().terminated[0] || zero.output().truncated[0])
      break;
    zero.step({0, 0, 0});
    contact = contact || !zero.world(0).contacts().empty();
    for (double value : zero.output().observations)
      EXPECT_TRUE(std::isfinite(value));
  }
  EXPECT_TRUE(contact);
}
TEST(SimTasks, FullRangeHopperActionsRetainBoundedSpeedsAcrossResets) {
  s::TaskBatch              batch(s::Task::Hopper, 4, datamunge::compute::Context({}, 2), 777, 100);
  std::uint64_t             random = 41;
  std::vector<double>       actions(12);
  std::vector<std::uint8_t> resets(4);
  for (int step = 0; step < 200; ++step) {
    for (auto& value : actions) {
      random = random * 6364136223846793005ULL + 1442695040888963407ULL;
      value  = 2. * double(random >> 11) * 0x1.0p-53 - 1.;
    }
    ASSERT_NO_THROW(batch.step(actions)) << "transition " << step;
    for (std::size_t i = 0; i < 4; ++i) {
      for (s::BodyId id = 1; id <= 4; ++id) {
        const auto body = batch.world(i).body(id);
        EXPECT_NO_THROW(body.validate());
        double linear = 0, angular = 0;
        for (int axis = 0; axis < 3; ++axis) {
          linear += body.linear_velocity[axis] * body.linear_velocity[axis];
          angular += body.angular_velocity[axis] * body.angular_velocity[axis];
        }
        EXPECT_LE(linear, 500. * 500. + 1e-8);
        EXPECT_LE(angular, 100. * 100. + 1e-8);
      }
      resets[i] = batch.output().terminated[i] || batch.output().truncated[i];
    }
    ASSERT_NO_THROW(batch.reset(resets));
  }
}
