#include <gtest/gtest.h>

#include <cmath>
#include <datamunge/sim/batch.hpp>
namespace s = datamunge::sim;
namespace {
s::WorldOptions limits() {
  s::WorldOptions o;
  o.body_capacity    = 8;
  o.pair_capacity    = 1;
  o.contact_capacity = 32;
  return o;
}
s::Body moving() {
  s::Body b;
  b.motion         = s::Motion::Dynamic;
  b.position       = {0, 4, 0};
  b.linear_damping = 0;
  return b;
}
} // namespace
TEST(SimBatch, MaskedStepsAndResetsPreserveOtherWorlds) {
  s::World                  prototype(limits());
  const auto                id = prototype.add_body(moving());
  s::BatchWorld             batch(prototype, 35, datamunge::compute::Context({}, 4));
  std::vector<std::uint8_t> mask(35);
  mask[0] = mask[34] = 1;
  batch.step(1. / 60, mask);
  EXPECT_EQ(batch.steps(0), 1);
  EXPECT_EQ(batch.steps(1), 0);
  EXPECT_EQ(batch.steps(34), 1);
  EXPECT_LT(batch.world(0).body(id).position[1], 4);
  EXPECT_EQ(batch.world(1).body(id).position[1], 4);
  const auto untouched = batch.world(34).body(id);
  mask[34]             = 0;
  batch.reset(mask);
  EXPECT_EQ(batch.steps(0), 0);
  EXPECT_EQ(batch.steps(34), 1);
  EXPECT_EQ(batch.world(0).body(id).position[1], 4);
  EXPECT_EQ(batch.world(34).body(id).position, untouched.position);
  EXPECT_EQ(prototype.body(id).position[1], 4);
}
TEST(SimBatch, RandomStreamsIgnoreSchedulingAndPartialResets) {
  s::World      prototype(limits());
  s::BatchWorld a(prototype, 3, datamunge::compute::Context({}, 1), 17);
  s::BatchWorld b(prototype, 3, datamunge::compute::Context({}, 3), 17);
  for (int i = 0; i < 100; ++i) {
    EXPECT_DOUBLE_EQ(a.uniform(0), b.uniform(0));
    a.uniform(1);
    a.uniform(1);
    b.uniform(2);
    a.reset({0, 1, 0});
  }
  a.seed(41);
  b.seed(41);
  for (int i = 0; i < 100; ++i)
    for (int j = 0; j < 3; ++j) {
      auto v = a.uniform(j);
      EXPECT_GE(v, 0);
      EXPECT_LT(v, 1);
      EXPECT_EQ(v, b.uniform(j));
    }
  EXPECT_THROW(a.uniform(3), std::out_of_range);
}
TEST(SimBatch, FailedParallelStepRewindsSuccessfulWorldsAndPendingForces) {
  s::World      prototype(limits());
  const auto    id = prototype.add_body(moving());
  s::BatchWorld batch(prototype, 35, datamunge::compute::Context({}, 4));
  for (std::size_t i = 0; i < batch.size(); ++i)
    batch.world(i).apply_force(id, {10, 0, 0});
  auto overlap = moving();
  batch.world(17).add_body(overlap);
  batch.world(17).add_body(overlap);
  EXPECT_THROW(batch.step(1. / 60), std::runtime_error);
  for (std::size_t i = 0; i < batch.size(); ++i) {
    EXPECT_EQ(batch.steps(i), 0);
    EXPECT_EQ(batch.world(i).body(id).position[1], 4);
  }
  std::vector<std::uint8_t> mask(35, 1);
  mask[17] = 0;
  batch.step(1. / 60, mask);
  EXPECT_NEAR(batch.world(0).body(id).linear_velocity[0], 10. / 60, 1e-12);
}
TEST(SimBatch, InvalidMasksAndStaleResetTargetsFailBeforeAnyMutation) {
  s::World      prototype(limits());
  const auto    id = prototype.add_body(moving());
  s::BatchWorld batch(prototype, 2);
  EXPECT_THROW(batch.step(1. / 60, {1}), std::invalid_argument);
  EXPECT_THROW(batch.step(1. / 60, {1, 2}), std::invalid_argument);
  EXPECT_THROW(batch.step(NAN), std::invalid_argument);
  batch.step(1. / 60);
  const auto before = batch.world(0).body(id);
  batch.world(1).add_body({});
  EXPECT_THROW(batch.reset(), std::invalid_argument);
  EXPECT_EQ(batch.world(0).body(id).position, before.position);
  EXPECT_EQ(batch.steps(0), 1);
  batch.capture_reset(1);
  EXPECT_NO_THROW(batch.reset());
  EXPECT_EQ(batch.steps(0), 0);
  EXPECT_THROW(batch.capture_reset(2), std::out_of_range);
  EXPECT_THROW(s::BatchWorld(prototype, 0), std::invalid_argument);
}
