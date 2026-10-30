#include <gtest/gtest.h>

#include <cmath>
#include <datamunge/sim/world.hpp>
#include <limits>
namespace s = datamunge::sim;
namespace {
s::WorldOptions small() {
  s::WorldOptions o;
  o.body_capacity    = 32;
  o.pair_capacity    = 256;
  o.contact_capacity = 256;
  return o;
}
s::Body dynamic(s::Shape shape = s::Shape::Sphere) {
  s::Body b;
  b.motion          = s::Motion::Dynamic;
  b.shape           = shape;
  b.position        = {0, 3, 0};
  b.linear_damping  = 0;
  b.angular_damping = 0;
  return b;
}
s::Body floor_body() {
  s::Body b;
  b.dimensions  = {20, 1, 20};
  b.position    = {0, -.5, 0};
  b.restitution = 0;
  return b;
}
void steps(s::World& w, int count) {
  for (int i = 0; i < count; ++i)
    w.step(1. / 60);
}
} // namespace
TEST(SimWorld, ValidationAndIdsAreTransactional) {
  s::World w(small());
  auto     b  = dynamic();
  auto     id = w.add_body(b);
  b.mass      = 0;
  EXPECT_THROW(w.add_body(b), std::invalid_argument);
  EXPECT_EQ(w.body_count(), 1);
  EXPECT_DOUBLE_EQ(w.body(id).mass, 1);
  EXPECT_THROW(w.set_body(id, b), std::invalid_argument);
  w.remove_body(id);
  EXPECT_THROW(w.body(id), std::out_of_range);
  EXPECT_GT(w.add_body(dynamic()), id);
  EXPECT_THROW(w.step(0), std::invalid_argument);
  EXPECT_THROW(w.step(std::numeric_limits<double>::quiet_NaN()), std::invalid_argument);
}
TEST(SimWorld, FreeFallMatchesDiscreteSemiImplicitSolution) {
  auto     o = small();
  s::World w(o);
  auto     b  = dynamic();
  auto     id = w.add_body(b);
  steps(w, 30);
  double dt = 1. / 60 / o.substeps, n = 30. * o.substeps;
  EXPECT_NEAR(w.body(id).position[1], 3 - .5 * 9.81 * dt * dt * n * (n + 1), 1e-11);
  EXPECT_NEAR(w.body(id).linear_velocity[1], -9.81 * .5, 1e-12);
}
TEST(SimWorld, ImpulseAndAccumulatedForceRespectMassAndLocks) {
  auto o    = small();
  o.gravity = {0, 0, 0};
  s::World w(o);
  auto     b            = dynamic();
  b.mass                = 2;
  b.allowed_translation = {true, false, false};
  auto id               = w.add_body(b);
  w.apply_impulse(id, {2, 4, 0});
  EXPECT_DOUBLE_EQ(w.body(id).linear_velocity[0], 1);
  EXPECT_DOUBLE_EQ(w.body(id).linear_velocity[1], 0);
  w.apply_force(id, {12, 0, 0});
  w.step(1. / 60);
  EXPECT_NEAR(w.body(id).linear_velocity[0], 1.1, 1e-12);
  w.step(1. / 60);
  EXPECT_NEAR(w.body(id).linear_velocity[0], 1.1, 1e-12);
  auto before = w.body(id);
  EXPECT_THROW(w.apply_impulse(id, {100000, 0, 0}), std::invalid_argument);
  EXPECT_EQ(w.body(id).linear_velocity, before.linear_velocity);
}
TEST(SimWorld, EveryPrimitiveRestsOnABox) {
  for (auto shape : {s::Shape::Box, s::Shape::Sphere, s::Shape::Capsule}) {
    s::World w(small());
    w.add_body(floor_body());
    auto b = dynamic(shape);
    if (shape == s::Shape::Capsule)
      b.dimensions = {.5, 1.5, .5};
    b.restitution = 0;
    auto id       = w.add_body(b);
    steps(w, 300);
    EXPECT_NEAR(w.body(id).position[1], b.dimensions[1] / 2, .015) << int(shape);
    EXPECT_LT(std::abs(w.body(id).linear_velocity[1]), .05) << int(shape);
    EXPECT_TRUE(w.grounded(id)) << int(shape);
  }
}
TEST(SimWorld, SensorsAndSleepingProximityQueries) {
  s::World w(small());
  w.add_body(floor_body());
  auto b             = dynamic();
  b.restitution      = 0;
  auto id            = w.add_body(b);
  auto trigger       = floor_body();
  trigger.sensor     = true;
  trigger.position   = {0, 1.5, 0};
  trigger.dimensions = {4, .2, 4};
  auto sensor        = w.add_body(trigger);
  steps(w, 180);
  EXPECT_NEAR(w.body(id).position[1], .5, .02);
  EXPECT_FALSE(w.touching(id, sensor));
  auto overlap     = w.body(id);
  overlap.position = {0, 1.5, 0};
  w.set_body(id, overlap);
  EXPECT_TRUE(w.touching(id, sensor));
}
TEST(SimWorld, ContinuousCollisionDoesNotTunnelThroughThinWall) {
  auto o     = small();
  o.substeps = 1;
  o.gravity  = {0, 0, 0};
  s::World w(o);
  s::Body  wall;
  wall.dimensions = {.02, 10, 10};
  w.add_body(wall);
  auto b            = dynamic();
  b.position        = {-2, 0, 0};
  b.linear_velocity = {500, 0, 0};
  b.restitution     = 0;
  auto id           = w.add_body(b);
  w.step(1. / 30);
  EXPECT_LT(w.body(id).position[0], 0);
  EXPECT_LT(w.body(id).linear_velocity[0], .01);
}
TEST(SimWorld, SnapshotReplaysAndRejectsForeignOrChangedTopology) {
  s::World w(small()), foreign(small());
  auto     id = w.add_body(dynamic());
  w.add_body(floor_body());
  auto snapshot = w.save_state();
  steps(w, 90);
  auto result = w.body(id);
  w.restore_state(snapshot);
  steps(w, 90);
  EXPECT_EQ(w.body(id).position, result.position);
  EXPECT_EQ(w.body(id).linear_velocity, result.linear_velocity);
  EXPECT_THROW(foreign.restore_state(snapshot), std::invalid_argument);
  auto change = w.body(id);
  change.mass = 2;
  w.set_body(id, change);
  EXPECT_THROW(w.restore_state(snapshot), std::invalid_argument);
}
TEST(SimWorld, CapacityFailuresRewindTheWholeStep) {
  auto o             = small();
  o.contact_capacity = 1;
  o.continuous       = false;
  s::World w(o);
  w.add_body(floor_body());
  auto b      = dynamic(s::Shape::Box);
  b.position  = {0, .49, 0};
  auto id     = w.add_body(b);
  auto before = w.body(id);
  EXPECT_THROW(w.step(1. / 60), std::runtime_error);
  EXPECT_EQ(w.body(id).position, before.position);
  EXPECT_EQ(w.body(id).linear_velocity, before.linear_velocity);
  EXPECT_EQ(w.contacts().size(), 0);
}
TEST(SimWorld, HingeCoordinateTeleportAndTorque) {
  auto o    = small();
  o.gravity = {0, 0, 0};
  s::World w(o);
  auto     b   = dynamic(s::Shape::Box);
  b.position   = {0, 0, 0};
  b.dimensions = {1, 2, 1};
  auto     id  = w.add_body(b);
  s::Joint j;
  j.child = id;
  j.axis  = {0, 0, 1};
  w.add_joint(j);
  EXPECT_NEAR(w.joint_state(id).position, 0, 1e-15);
  w.set_joint_state(id, .3, .2);
  EXPECT_NEAR(w.joint_state(id).position, .3, 1e-12);
  EXPECT_NEAR(w.joint_state(id).velocity, .2, 1e-12);
  w.apply_joint_force(id, 1);
  w.step(1. / 60);
  EXPECT_GT(w.joint_state(id).velocity, .2);
  EXPECT_NEAR(w.body(id).position[0], 0, 1e-6);
  EXPECT_THROW(w.remove_body(id), std::invalid_argument);
}
TEST(SimWorld, SliderLimitsAndSubtreeTeleport) {
  auto o    = small();
  o.gravity = {0, 0, 0};
  s::World w(o);
  auto     b     = dynamic(s::Shape::Box);
  b.position     = {0, 0, 0};
  auto parent    = w.add_body(b);
  b.position     = {0, 2, 0};
  auto     child = w.add_body(b);
  s::Joint slider;
  slider.child     = parent;
  slider.type      = s::JointType::Slider;
  slider.axis      = {1, 0, 0};
  slider.limited   = true;
  slider.limit_min = -1;
  slider.limit_max = 1;
  w.add_joint(slider);
  s::Joint hinge;
  hinge.child  = child;
  hinge.parent = parent;
  w.add_joint(hinge);
  w.set_joint_state(parent, .5, .2);
  EXPECT_NEAR(w.body(parent).position[0], .5, 1e-12);
  EXPECT_NEAR(w.body(child).position[0], .5, 1e-12);
  EXPECT_NEAR(w.joint_state(child).position, 0, 1e-12);
  auto before = w.body(parent);
  EXPECT_THROW(w.set_joint_state(parent, 2, 0), std::invalid_argument);
  EXPECT_EQ(w.body(parent).position, before.position);
  for (int i = 0; i < 120; ++i) {
    w.apply_joint_force(parent, 5);
    w.step(1. / 60);
  }
  EXPECT_LE(w.joint_state(parent).position, 1.01);
}
TEST(SimWorld, ConvexQueriesDistinguishOverlappingBounds) {
  s::World w(small());
  s::Body  a;
  a.shape     = s::Shape::Sphere;
  auto first  = w.add_body(a);
  a.position  = {.9, .9, 0};
  auto second = w.add_body(a);
  EXPECT_FALSE(w.touching(first, second, 0));
  a.position = {.5, .5, 0};
  w.set_body(second, a);
  EXPECT_TRUE(w.touching(first, second, 0));
}

TEST(SimWorld, TeleportPreservesMovingParentFrameVelocities) {
  auto options    = small();
  options.gravity = {0, 0, 0};
  for (auto kind : {s::JointType::Hinge, s::JointType::Slider}) {
    s::World world(options);
    s::Body  parent;
    parent.motion           = s::Motion::Kinematic;
    parent.linear_velocity  = {2, 3, 0};
    parent.angular_velocity = {0, 0, 1};
    const auto p            = world.add_body(parent);
    auto       child        = dynamic();
    child.position          = {1, 0, 0};
    child.linear_velocity   = {2, 4, 0};
    child.angular_velocity  = {0, 0, 1};
    const auto c            = world.add_body(child);
    s::Joint   j;
    j.child  = c;
    j.parent = p;
    j.type   = kind;
    j.anchor = {-1, 0, 0};
    if (kind == s::JointType::Slider)
      j.axis = {1, 0, 0};
    world.add_joint(j);
    world.set_joint_state(c, kind == s::JointType::Hinge ? std::acos(-1.) / 2 : .5, 0);
    const auto result = world.body(c);
    if (kind == s::JointType::Hinge) {
      EXPECT_NEAR(result.linear_velocity[0], 1, 1e-12);
      EXPECT_NEAR(result.linear_velocity[1], 3, 1e-12);
    } else {
      EXPECT_NEAR(result.linear_velocity[0], 2, 1e-12);
      EXPECT_NEAR(result.linear_velocity[1], 4.5, 1e-12);
    }
    EXPECT_NEAR(world.joint_state(c).velocity, 0, 1e-12);
  }
}
TEST(SimWorld, LimitDoesNotSlowAChildAlreadyReturningInsideTheAllowedRange) {
  auto options    = small();
  options.gravity = {0, 0, 0};
  s::World world(options);
  auto     value = dynamic();
  value.position = {0, 0, 0};
  const auto id  = world.add_body(value);
  s::Joint   joint;
  joint.type      = s::JointType::Slider;
  joint.child     = id;
  joint.axis      = {1, 0, 0};
  joint.limited   = true;
  joint.limit_min = -1;
  joint.limit_max = 1;
  world.add_joint(joint);
  value.position        = {1.01, 0, 0};
  value.linear_velocity = {-3, 0, 0};
  world.set_body(id, value);
  world.step(1. / 60);
  EXPECT_NEAR(world.body(id).linear_velocity[0], -3, 1e-12);
  EXPECT_LT(world.joint_state(id).position, 1);
}
TEST(SimWorld, JointWarmStartsAndPendingActuationReplayExactly) {
  auto     options = small();
  s::World world(options);
  auto     value = dynamic();
  value.position = {0, 1, 0};
  const auto id  = world.add_body(value);
  s::Joint   joint;
  joint.child  = id;
  joint.anchor = {0, -.5, 0};
  world.add_joint(joint);
  world.set_joint_state(id, .1, 0);
  for (int i = 0; i < 10; ++i)
    world.step(1. / 60);
  world.apply_joint_force(id, .3);
  const auto saved = world.save_state();
  for (int i = 0; i < 20; ++i)
    world.step(1. / 60);
  const auto expected = world.body(id);
  world.restore_state(saved);
  for (int i = 0; i < 20; ++i)
    world.step(1. / 60);
  const auto actual = world.body(id);
  EXPECT_EQ(actual.position, expected.position);
  EXPECT_EQ(actual.rotation, expected.rotation);
  EXPECT_EQ(actual.linear_velocity, expected.linear_velocity);
  EXPECT_EQ(actual.angular_velocity, expected.angular_velocity);
}
