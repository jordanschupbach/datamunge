#include <gtest/gtest.h>

#include <cmath>
#include <datamunge/sim/fruit_stream.hpp>
#include <filesystem>

#include "../../src/datamunge/sim/stream_dynamics.hpp"
namespace s = datamunge::sim;
namespace {
std::string fixture() {
  return (std::filesystem::path(__FILE__).parent_path().parent_path() /
          "fixtures/devarm_physics/robot.urdf")
      .string();
}
const std::array<double, 6> pose{0, -2.4, -2.4, 1.2, 0, 0};
s::ConveyorObservation      apple() {
  s::ConveyorObservation f;
  f.kind                 = s::FruitKind::Apple;
  f.appearance.scale     = 1;
  f.appearance.prototype = 0;
  f.position             = {-.5, 2, 0};
  f.dimensions           = {.078, .078, .078};
  return f;
}
} // namespace
TEST(ArticulatedPhysics, MeasuredEncodersDoNotChangeStrictJointCommands) {
  auto robot = datamunge::cad::Robot::load_urdf(fixture());
  EXPECT_THROW(robot.set_joint("joint2", .001), std::invalid_argument);
  robot.set_measured_joint("joint2", .001);
  EXPECT_DOUBLE_EQ(robot.joint_position("joint2"), .001);
  EXPECT_TRUE(std::isfinite(robot.link_transform("end_link").TransformPoint({0, 0, 0}).z));
  EXPECT_THROW(robot.set_measured_joint("joint2", NAN), std::invalid_argument);
  EXPECT_DOUBLE_EQ(robot.joint_position("joint2"), .001);
  robot.set_measured_joint("finger_left", .0501);
  EXPECT_DOUBLE_EQ(robot.joint_position("finger_right"), -.0501);
}
TEST(ArticulatedPhysics, FreeFlightConvergesAndRetainsMomentumAndRotation) {
  if (s::articulated_physics_version().empty())
    GTEST_SKIP() << "MuJoCo not compiled";
  double previous = 1;
  for (int divisor : {240, 480, 960}) {
    s::World                  world;
    s::detail::StreamDynamics dynamics(fixture(), world, 1. / divisor, 2);
    dynamics.reset(pose, .05);
    auto f             = apple();
    f.velocity         = {1, 0, 0};
    f.angular_velocity = {0, 1, 0};
    dynamics.spawn(42, f);
    s::Body base;
    base.motion     = s::Motion::Dynamic;
    base.shape      = s::Shape::Sphere;
    base.dimensions = f.dimensions;
    for (int i = 0; i < 24; ++i)
      dynamics.step(pose, .05);
    auto   b     = dynamics.body(42, base);
    double exact = 2 - 9.81 * .1 * .1 / 2, error = std::abs(b.position[1] - exact);
    EXPECT_LT(error, previous);
    previous = error;
    EXPECT_NEAR(b.position[0], -.4, 1e-9);
    EXPECT_NEAR(b.linear_velocity[0], 1, 1e-9);
    EXPECT_NEAR(b.linear_velocity[1], -.981, 1e-9);
    EXPECT_GT(std::abs(b.rotation[2]), .04);
    EXPECT_FALSE(dynamics.gripped(42));
    EXPECT_FALSE(dynamics.bin_supported(42));
    EXPECT_LE(dynamics.effort_ratio(), 1 + 1e-12);
  }
}
TEST(ArticulatedPhysics, NewMassPropertiesNeverTeleportExistingBodies) {
  if (s::articulated_physics_version().empty())
    GTEST_SKIP() << "MuJoCo not compiled";
  s::World                  world;
  s::detail::StreamDynamics dynamics(fixture(), world, 1. / 480, 2);
  dynamics.reset(pose, .05);
  auto first     = apple();
  first.velocity = {1, 0, 0};
  dynamics.spawn(10, first);
  for (int i = 0; i < 12; ++i)
    dynamics.step(pose, .05);
  s::Body base;
  auto    before    = dynamics.body(10, base);
  auto    second    = apple();
  second.kind       = s::FruitKind::Banana;
  second.position   = {-.8, 2, 0};
  second.dimensions = {.046, .16, .046};
  dynamics.spawn(11, second);
  auto after = dynamics.body(10, base);
  EXPECT_EQ(before.position, after.position);
  EXPECT_EQ(before.linear_velocity, after.linear_velocity);
  EXPECT_EQ(before.rotation, after.rotation);
  dynamics.remove(10);
  EXPECT_THROW(dynamics.body(10, base), std::out_of_range);
  dynamics.reset(pose, .05);
  EXPECT_NO_THROW(dynamics.step(pose, .05));
  dynamics.spawn(12, second);
  EXPECT_NO_THROW(dynamics.step(pose, .05));
}
TEST(ArticulatedPhysics, BeltActsThroughContactAndBinsRequireActualFloorForces) {
  if (s::articulated_physics_version().empty())
    GTEST_SKIP() << "MuJoCo not compiled";
  s::World world;
  s::Body  belt;
  belt.position   = {-.3, .68, 0};
  belt.dimensions = {1.5, .08, .34};
  world.add_body(belt);
  s::Body floor;
  floor.position   = {.1, .43, -.42};
  floor.dimensions = {.34, .025, .30};
  world.add_body(floor);
  s::detail::StreamDynamics dynamics(fixture(), world, 1. / 480, 2);
  dynamics.reset(pose, .05);
  auto f     = apple();
  f.position = {-.7, .78, 0};
  dynamics.spawn(1, f);
  auto bin     = apple();
  bin.position = {.1, .8, -.42};
  dynamics.spawn(2, bin);
  EXPECT_FALSE(dynamics.bin_supported(2));
  for (int i = 0; i < 480; ++i)
    dynamics.step(pose, .05);
  s::Body base;
  auto    transported = dynamics.body(1, base), settled = dynamics.body(2, base);
  EXPECT_GT(transported.position[0], -.4);
  EXPECT_NEAR(transported.linear_velocity[0], .22, .015);
  EXPECT_NEAR(transported.position[1], .759, .002);
  EXPECT_TRUE(dynamics.bin_supported(2));
  EXPECT_NEAR(settled.position[1], .4815, .002);
  auto diagnostics = dynamics.diagnostics();
  EXPECT_LT(diagnostics[0], .004);
  EXPECT_LE(diagnostics[1], 1 + 1e-12);
}
TEST(ArticulatedPhysics, DetailedTrainingRejectsAnalyticTransport) {
  s::FruitStreamOptions options;
  options.articulated_physics = true;
  options.training_proxy      = true;
  EXPECT_THROW(s::FruitStreamEnvironment(fixture(), options), std::invalid_argument);
}

TEST(ArticulatedPhysics, BatchMeasurementsAreAtomicAndKinematicsMatchTheRenderer) {
  auto robot = datamunge::cad::Robot::load_urdf(fixture());
  robot.set_measured_joints({{"joint1", .1}, {"joint2", -2.4}});
  EXPECT_THROW(robot.set_measured_joints({{"joint1", .2}, {"absent", 0}}), std::invalid_argument);
  EXPECT_DOUBLE_EQ(robot.joint_position("joint1"), .1);
  if (s::articulated_physics_version().empty())
    GTEST_SKIP() << "MuJoCo not compiled";
  s::World                  world;
  s::detail::StreamDynamics dynamics(fixture(), world, 1. / 480, 2);
  dynamics.reset(pose, .05);
  for (int i = 0; i < 20; ++i)
    dynamics.step(pose, .05);
  s::PickObservation measured;
  dynamics.observe(measured);
  std::vector<std::pair<std::string, double>> values;
  for (int i = 0; i < 6; ++i)
    values.emplace_back("joint" + std::to_string(i + 1), measured.joints[i]);
  robot.set_measured_joints(values);
  auto p      = robot.link_transform("end_link").TransformPoint({-.02, 0, 0});
  auto actual = dynamics.tool_point();
  EXPECT_NEAR(actual.x, .62 - p.x, 1e-12);
  EXPECT_NEAR(actual.y, -p.y, 1e-12);
  EXPECT_NEAR(actual.z, .72 + p.z, 1e-12);
}

TEST(ArticulatedPhysics, RapidAsymmetricTumblingDoesNotCreateRotationalEnergy) {
  if (s::articulated_physics_version().empty())
    GTEST_SKIP() << "MuJoCo not compiled";
  s::World                  world;
  s::detail::StreamDynamics dynamics(fixture(), world, 1. / 960, 1);
  dynamics.reset(pose, .05);
  auto fruit                 = apple();
  fruit.appearance.prototype = 1;
  fruit.angular_velocity     = {150, 200, 100};
  dynamics.spawn(1, fruit);
  s::Body      base;
  const double r = .039, a = .8 * r;
  auto         energy = [&](const s::Body& body) {
    auto q = body.rotation;
    // Transform world angular velocity into the native body coordinate frame.
    s::Vec3 v     = body.angular_velocity, xyz{-q[1], -q[2], -q[3]};
    auto    cross = [](auto x, auto y) {
      return s::Vec3{
          x[1] * y[2] - x[2] * y[1], x[2] * y[0] - x[0] * y[2], x[0] * y[1] - x[1] * y[0]};
    };
    auto t = cross(xyz, v);
    for (auto& x : t)
      x *= 2;
    auto u = cross(xyz, t);
    for (int i = 0; i < 3; ++i)
      v[i] += q[0] * t[i] + u[i];
    return .5 * .16 / 5 * ((a * a + r * r) * (v[0] * v[0] + v[2] * v[2]) + 2 * a * a * v[1] * v[1]);
  };
  double initial = energy(dynamics.body(1, base));
  for (int i = 0; i < 480; ++i) {
    ASSERT_NO_THROW(dynamics.step(pose, .05));
    auto body = dynamics.body(1, base);
    EXPECT_LE(energy(body), initial * 1.01);
  }
}

TEST(ArticulatedPhysics, CloseArrivalsStartAboveExistingFruitWithoutMovingIt) {
  if (s::articulated_physics_version().empty())
    GTEST_SKIP() << "MuJoCo not compiled";
  s::World                  world;
  s::detail::StreamDynamics dynamics(fixture(), world, 1. / 960, 2);
  dynamics.reset(pose, .05);
  auto fruit = apple();
  dynamics.spawn(1, fruit);
  s::Body base;
  auto    before = dynamics.body(1, base);
  dynamics.spawn(2, fruit);
  auto first = dynamics.body(1, base), second = dynamics.body(2, base);
  EXPECT_EQ(first.position, before.position);
  EXPECT_EQ(first.linear_velocity, before.linear_velocity);
  EXPECT_NEAR(second.position[0], fruit.position[0], 1e-12);
  EXPECT_NEAR(second.position[2], fruit.position[2], 1e-12);
  EXPECT_GE(second.position[1] - first.position[1], .080 - 1e-12);
  EXPECT_NO_THROW(dynamics.step(pose, .05));
  EXPECT_LT(dynamics.diagnostics()[0], 1e-9);
}
