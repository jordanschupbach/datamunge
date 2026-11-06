/// @file sim_native_dynamics_tests.cpp
/// @brief Analytic native rigid dynamics, geometry, import and no-MuJoCo workcell oracles.
#include <gtest/gtest.h>

#include <datamunge/sim/articulation.hpp>
#include <datamunge/sim/fruit_stream.hpp>
#include <filesystem>
#include <fstream>

#include "../../src/datamunge/sim/collision.hpp"
#include "../../src/datamunge/sim/rigid_math.hpp"
#include "../../src/datamunge/sim/stream_dynamics.hpp"
namespace s = datamunge::sim;
namespace d = datamunge::sim::detail;
namespace {
s::WorldOptions settings() {
  s::WorldOptions o;
  o.gravity               = {0, 0, 0};
  o.substeps              = 1;
  o.continuous            = false;
  o.gyroscopic            = true;
  o.clamp_velocities      = false;
  o.sleeping              = false;
  o.solver_iterations     = 80;
  o.persistent_contacts   = true;
  o.contact_stabilization = true;
  o.contact_slop          = .0002;
  return o;
}
s::Body moving() {
  s::Body b;
  b.motion         = s::Motion::Dynamic;
  b.linear_damping = b.angular_damping = 0;
  b.collision_group                    = 0;
  return b;
}
std::string fixture() {
  return (std::filesystem::path(__FILE__).parent_path().parent_path() /
          "fixtures/devarm_physics/robot.urdf")
      .string();
}
} // namespace
TEST(NativeDynamics, AsymmetricFreeRotationConservesEnergyAndWorldMomentum) {
  for (double rate : {5., 269.}) {
    s::World w(settings());
    auto     b            = moving();
    b.mass                = .12;
    b.dimensions          = {.05, .15, .04};
    b.angular_speed_limit = 10000;
    b.inertia_tensor      = {.00021, .000012, 0, .000012, .00008, 0, 0, 0, .00017};
    b.center_of_mass      = {0, .02, -.01};
    b.angular_velocity    = {rate, .41 * rate, -.27 * rate};
    auto id               = w.add_body(b);
    auto momentum         = [&](s::Body a) {
      return d::rotate(
          a.rotation,
          d::apply_tensor(
              a.inertia_tensor, d::rotate(d::conjugate(a.rotation), a.angular_velocity)));
    };
    auto   L   = momentum(b);
    double E   = .5 * d::dot(b.angular_velocity, L);
    auto   com = d::center(b);
    for (int k = 0; k < 1920; ++k)
      w.step(1. / 960);
    auto out = w.body(id);
    EXPECT_NEAR(.5 * d::dot(out.angular_velocity, momentum(out)) / E, 1, 2e-8);
    EXPECT_LT(d::length(d::sub(momentum(out), L)) / d::length(L), 2e-8);
    EXPECT_LT(d::length(d::sub(d::center(out), com)), 1e-12);
  }
}
TEST(NativeDynamics, ImplicitServoHonorsEffortAndArmature) {
  s::World w(settings());
  auto     b  = moving();
  b.mass      = 2;
  auto     id = w.add_body(b);
  s::Joint j;
  j.type         = s::JointType::Slider;
  j.axis         = {1, 0, 0};
  j.child        = id;
  j.armature     = 3;
  j.effort_limit = 2;
  w.add_joint(j);
  w.set_joint_motor(id, 10, 1e9, 2);
  w.step(.01);
  EXPECT_NEAR(w.joint_state(id).velocity, .004, 1e-10);
  EXPECT_NEAR(w.joint_motor_effort(id), 2, 1e-12);
  EXPECT_THROW(w.set_joint_motor(id, 1, 10, 3), std::invalid_argument);
  auto snap = w.save_state();
  w.step(.01);
  auto expected = w.joint_state(id);
  w.restore_state(snap);
  w.step(.01);
  EXPECT_DOUBLE_EQ(w.joint_state(id).velocity, expected.velocity);
}
TEST(NativeDynamics, MimicSharesDynamicLoadAndSnapshots) {
  s::World w(settings());
  auto     b = moving();
  auto     a = w.add_body(b);
  b.position = {0, 2, 0};
  auto     c = w.add_body(b);
  s::Joint j;
  j.type  = s::JointType::Slider;
  j.axis  = {1, 0, 0};
  j.child = a;
  w.add_joint(j);
  j.child = c;
  w.add_joint(j);
  w.add_gear(a, c, -1);
  EXPECT_THROW(w.remove_joint(a), std::invalid_argument);
  for (int k = 0; k < 100; ++k) {
    w.apply_joint_force(a, 2);
    w.step(.001);
  }
  EXPECT_NEAR(w.joint_state(a).velocity, .1, 1e-9);
  EXPECT_NEAR(w.joint_state(c).velocity, -.1, 1e-9);
  EXPECT_NEAR(w.joint_state(a).position, -w.joint_state(c).position, 1e-10);
  auto copy = w.fork();
  w.step(.001);
  copy->step(.001);
  EXPECT_EQ(w.body(a).position, copy->body(a).position);
  w.remove_gear(c);
  w.remove_joint(a);
}
TEST(NativeDynamics, GravityCompensationUsesOffsetCom) {
  auto o    = settings();
  o.gravity = {0, -9.81, 0};
  s::World w(o);
  auto     b       = moving();
  b.mass           = 2;
  b.center_of_mass = {.25, 0, 0};
  auto     id      = w.add_body(b);
  s::Joint j;
  j.child = id;
  j.axis  = {0, 0, 1};
  w.add_joint(j);
  EXPECT_NEAR(w.joint_gravity_compensation(id), 4.905, 1e-12);
  for (int k = 0; k < 100; ++k) {
    w.set_joint_motor(id, 0, 100, 10, w.joint_gravity_compensation(id));
    w.step(.001);
  }
  EXPECT_NEAR(w.joint_state(id).position, 0, 1e-6);
}
TEST(NativeDynamics, FixedJointCarriesChildLoad) {
  s::World w(settings());
  auto     b = moving();
  auto     a = w.add_body(b);
  b.position = {1, 0, 0};
  auto     c = w.add_body(b);
  s::Joint j;
  j.type   = s::JointType::Fixed;
  j.parent = a;
  j.child  = c;
  w.add_joint(j);
  for (int i = 0; i < 100; ++i) {
    w.apply_force(c, {0, 1, 0});
    w.step(.001);
  }
  auto aa = w.body(a), cc = w.body(c);
  auto local = d::rotate(d::conjugate(aa.rotation), d::sub(cc.position, aa.position));
  EXPECT_NEAR(local[0], 1, 5e-5);
  EXPECT_NEAR(local[1], 0, 5e-5);
  EXPECT_GT(aa.linear_velocity[1], 0);
}
TEST(NativeGeometry, HullDistancePenetrationAndOrder) {
  s::Body              a, b;
  std::vector<s::Vec3> v;
  for (double x : {-.5, .5})
    for (double y : {-.5, .5})
      for (double z : {-.5, .5})
        v.push_back({x, y, z});
  a.shape    = s::Shape::ConvexHull;
  a.vertices = std::make_shared<const std::vector<s::Vec3>>(v);
  b.position = {2, 0, 0};
  EXPECT_NEAR(d::distance(a, b).value, 1, 1e-7);
  b.position = {.8, 0, 0};
  auto ab = d::collide(a, b, .0002), ba = d::collide(b, a, .0002);
  ASSERT_GT(ab.count, 0u);
  ASSERT_GT(ba.count, 0u);
  EXPECT_NEAR(ab.points[0].separation, -.2, 1e-5);
  EXPECT_GT(ab.points[0].normal[0], .999);
  EXPECT_LT(ba.points[0].normal[0], -.999);
}
TEST(NativeGeometry, CompoundAndEllipsoidSettleWithForceFeedback) {
  auto o    = settings();
  o.gravity = {0, -9.81, 0};
  s::World w(o);
  s::Body  floor;
  floor.dimensions  = {10, 1, 10};
  floor.position    = {0, -.5, 0};
  floor.restitution = 0;
  w.add_body(floor);
  auto b            = moving();
  b.collision_group = 1;
  b.shape           = s::Shape::Ellipsoid;
  b.dimensions      = {.12, .08, .10};
  b.position        = {0, .2, 0};
  b.mass            = .2;
  b.restitution     = 0;
  auto id           = w.add_body(b);
  for (int k = 0; k < 1920; ++k)
    w.step(1. / 960);
  EXPECT_NEAR(w.body(id).position[1], .04, .0004);
  double force = 0;
  for (auto c : w.contacts())
    if (c.first == id || c.second == id)
      force += c.impulse * 960;
  EXPECT_NEAR(force, 1.962, .02);
  s::Collider left, right;
  left.position  = {-2, 0, 0};
  right.position = {2, 0, 0};
  s::Body compound;
  compound.colliders =
      std::make_shared<const std::vector<s::Collider>>(std::vector<s::Collider>{left, right});
  double  t = 0;
  s::Vec3 normal;
  EXPECT_TRUE(d::ray_hit(compound, {-2, 2, 0}, {0, -1, 0}, 10, t, normal));
  EXPECT_NEAR(t, 1.5, 1e-10);
  EXPECT_FALSE(d::ray_hit(compound, {0, 2, 0}, {0, -1, 0}, 10, t, normal));
}
TEST(NativeImport, StrictInertiasAndTransactionalCapacity) {
  auto     model = s::UrdfModel::load(fixture(), [] {
    s::UrdfPhysicsOptions o;
    o.missing_mass = .04;
    return o;
  }());
  s::World w(settings());
  auto     r = model.instantiate(w);
  EXPECT_EQ(r.links.size(), 10u);
  EXPECT_EQ(r.joints.size(), 9u);
  EXPECT_NEAR(w.body(r.links.at("base_link")).mass, .8366, 1e-12);
  EXPECT_NEAR(w.body(r.links.at("base_link")).center_of_mass[2], .029841, 1e-12);
  auto b = w.body(r.links.at("link1"));
  EXPECT_NEAR(b.mass, .1613, 1e-12);
  EXPECT_NEAR(b.center_of_mass[0], .000113614552951627, 1e-16);
  EXPECT_NEAR(b.inertia_tensor[2], -.00002832, 1e-12);
  auto o          = settings();
  o.body_capacity = 3;
  s::World small(o);
  EXPECT_THROW(model.instantiate(small), std::length_error);
  EXPECT_EQ(small.body_count(), 0u);
  EXPECT_EQ(small.joint_count(), 0u);
}
TEST(NativeWorkcell, NoExternalEngineFreeFlightAndPhysicalBelt) {
  s::World source;
  s::Body  belt;
  belt.position   = {-.3, .68, 0};
  belt.dimensions = {1.5, .08, .34};
  source.add_body(belt);
  s::Body floor;
  floor.position   = {.1, .43, -.42};
  floor.dimensions = {.34, .025, .30};
  source.add_body(floor);
  s::detail::StreamDynamics physics(fixture(), source, 1. / 960, 3, s::ArticulatedBackend::Native);
  std::array<double, 6>     q{0, -2.4, -2.4, 1.2, 0, 0};
  physics.reset(q, .05);
  s::ConveyorObservation apple;
  apple.kind             = s::FruitKind::Apple;
  apple.appearance.scale = 1;
  apple.position         = {-.7, .78, 0};
  physics.spawn(1, apple);
  auto bin     = apple;
  bin.position = {.1, .8, -.42};
  physics.spawn(2, bin);
  auto free     = apple;
  free.position = {-3, 2, 0};
  free.velocity = {1, 0, 0};
  physics.spawn(3, free);
  for (int i = 0; i < 24; ++i)
    physics.step(q, .05);
  auto f = physics.body(3, {});
  EXPECT_NEAR(f.position[0], -2.9, 1e-10);
  EXPECT_NEAR(f.linear_velocity[1], -.981, 1e-10);
  physics.remove(3);
  for (int i = 24; i < 480; ++i)
    physics.step(q, .05);
  auto b = physics.body(1, {});
  EXPECT_GT(b.position[0], -.4);
  EXPECT_NEAR(b.linear_velocity[0], .22, .02);
  EXPECT_NEAR(b.position[1], .759, .003);
  EXPECT_TRUE(physics.bin_supported(2));
  EXPECT_LT(physics.diagnostics()[0], .005);
  EXPECT_LE(physics.diagnostics()[1], 1 + 1e-12);
  s::PickObservation a;
  physics.observe(a);
  EXPECT_TRUE(a.measured_joints);
  for (int i = 0; i < 6; ++i)
    EXPECT_NEAR(a.joints[i], q[i], .02);
  auto geometry = datamunge::cad::Robot::load_urdf(fixture());
  std::vector<std::pair<std::string, double>> measured;
  for (int i = 0; i < 6; ++i)
    measured.emplace_back("joint" + std::to_string(i + 1), a.joints[i]);
  geometry.set_measured_joints(measured);
  auto rendered = geometry.link_transform("end_link").TransformPoint({-.02, 0, 0});
  auto physical = physics.tool_point();
  EXPECT_NEAR(physical.x, .62 - rendered.x, 2e-5);
  EXPECT_NEAR(physical.y, -rendered.y, 2e-5);
  EXPECT_NEAR(physical.z, .72 + rendered.z, 2e-5);
}
TEST(NativeDynamics, CoupledServoUsesReflectedInertiaAndFullChainMass) {
  auto o         = settings();
  o.block_joints = true;
  s::World w(o);
  auto     b = moving();
  b.mass     = 2;
  auto a     = w.add_body(b);
  b.position = {0, 2, 0};
  auto     c = w.add_body(b);
  s::Joint j;
  j.type         = s::JointType::Slider;
  j.axis         = {1, 0, 0};
  j.child        = a;
  j.armature     = 3;
  j.effort_limit = 2;
  w.add_joint(j);
  j.type     = s::JointType::Fixed;
  j.child    = c;
  j.parent   = a;
  j.armature = 0;
  w.add_joint(j);
  w.set_joint_motor(a, 10, 1e9, 2);
  w.step(.01);
  EXPECT_NEAR(w.joint_state(a).velocity, 2 * .01 / 7, 1e-10);
  EXPECT_NEAR(w.body(c).linear_velocity[0], w.body(a).linear_velocity[0], 1e-12);
  EXPECT_NEAR(w.joint_motor_effort(a), 2, 1e-12);
  auto state = w.save_state();
  w.step(.01);
  auto expected = w.body(c);
  w.restore_state(state);
  w.step(.01);
  EXPECT_EQ(expected.position, w.body(c).position);
}
TEST(NativeGeometry, DeepConcentricHullsAndCompoundSweeps) {
  std::vector<s::Vec3> vertices;
  for (double x : {-.5, .5})
    for (double y : {-.5, .5})
      for (double z : {-.5, .5})
        vertices.push_back({x, y, z});
  s::Body a, b;
  a.shape = b.shape = s::Shape::ConvexHull;
  a.vertices = b.vertices = std::make_shared<const std::vector<s::Vec3>>(vertices);
  auto manifold           = d::collide(a, b, .0002);
  ASSERT_GT(manifold.count, 0u);
  EXPECT_NEAR(manifold.points[0].separation, -1, 1e-5);
  s::Collider arm;
  arm.position     = {10, 0, 0};
  a                = moving();
  a.center_of_mass = {-1, 0, 0};
  a.colliders = std::make_shared<const std::vector<s::Collider>>(std::vector<s::Collider>{arm});
  a.angular_velocity = {0, 0, 10};
  auto swept         = d::swept_bounds(a, .02, true);
  auto moved         = a;
  d::integrate_pose(moved, .02, true);
  auto end = d::bounds(moved);
  for (int i = 0; i < 3; ++i) {
    EXPECT_LE(swept.lo[i], end.lo[i]);
    EXPECT_GE(swept.hi[i], end.hi[i]);
  }
}
TEST(NativeImport, MissingAndInvalidInertiasAreExplicit) {
  auto root = std::filesystem::temp_directory_path() / "datamunge-native-urdf-tests";
  std::filesystem::create_directories(root);
  auto file = root / "missing.urdf";
  std::ofstream(file)
      << R"(<robot name="minimal"><link name="base"/><link name="tip"/><joint name="slide" type="prismatic"><parent link="base"/><child link="tip"/><axis xyz="1 0 0"/><limit lower="0" upper="1" effort="2" velocity="1"/></joint></robot>)";
  EXPECT_THROW(s::UrdfModel::load(file.string()), std::invalid_argument);
  s::UrdfPhysicsOptions options;
  options.missing_mass = .04;
  auto     model       = s::UrdfModel::load(file.string(), options);
  s::World w(settings());
  auto     robot = model.instantiate(w);
  EXPECT_DOUBLE_EQ(w.body(robot.links.at("tip")).mass, .04);
  EXPECT_THROW(w.set_joint_motor(robot.joints.at("slide"), 2, 1, 1), std::invalid_argument);
  std::filesystem::remove_all(root);
}
TEST(NativeDynamics, CoupledGearsIncludeRotorAndSharedParentMass) {
  auto o              = settings();
  o.block_joints      = true;
  o.solver_iterations = 12;
  s::World w(o);
  auto     b = moving();
  b.mass     = .04;
  auto a     = w.add_body(b);
  b.position = {0, 2, 0};
  auto     c = w.add_body(b);
  s::Joint j;
  j.type     = s::JointType::Slider;
  j.axis     = {1, 0, 0};
  j.child    = a;
  j.armature = 1;
  w.add_joint(j);
  j.child = c;
  w.add_joint(j);
  w.add_gear(a, c, -1);
  for (int k = 0; k < 100; ++k) {
    w.apply_joint_force(a, 2);
    w.step(.001);
  }
  EXPECT_NEAR(w.joint_state(a).velocity, 2 * .1 / (2 * 1.04), 1e-9);
  EXPECT_NEAR(w.joint_state(c).velocity, -w.joint_state(a).velocity, 1e-9);
}
TEST(NativeContacts, SpinAndRollingTorquesAreBoundedByMeasuredNormalLoad) {
  auto o    = settings();
  o.gravity = {0, -9.81, 0};
  s::World w(o);
  s::Body  floor;
  floor.position    = {0, -.5, 0};
  floor.dimensions  = {10, 1, 10};
  floor.friction    = 0;
  floor.restitution = 0;
  w.add_body(floor);
  auto b              = moving();
  b.collision_group   = 1;
  b.shape             = s::Shape::Sphere;
  b.dimensions        = {.2, .2, .2};
  b.position          = {0, .1, 0};
  b.friction          = 0;
  b.restitution       = 0;
  b.spinning_friction = .01;
  b.rolling_friction  = .02;
  b.angular_velocity  = {0, 4, 3};
  auto id             = w.add_body(b);
  w.step(.001);
  auto   after = w.body(id);
  double load  = 0;
  for (auto c : w.contacts())
    if (c.first == id || c.second == id)
      load += c.impulse;
  EXPECT_NEAR(load, .00981, 1e-10);
  EXPECT_NEAR(after.angular_velocity[1], 4 - .01 * load / .004, 1e-9);
  EXPECT_NEAR(after.angular_velocity[2], 3 - .02 * load / .004, 1e-9);
}
TEST(NativeImport, RotatedAuthoredInertiaAndLocalStlRemainInLinkFrame) {
  auto root = std::filesystem::temp_directory_path() / "datamunge-native-inertia-mesh-test";
  std::filesystem::create_directories(root);
  std::ofstream(root / "tetra.stl")
      << "solid tetra\nfacet normal 0 0 0\nouter loop\nvertex 0 0 0\nvertex 1 0 0\nvertex 0 1 "
         "0\nendloop\nendfacet\nfacet normal 0 0 0\nouter loop\nvertex 0 0 0\nvertex 0 1 0\nvertex "
         "0 0 1\nendloop\nendfacet\nfacet normal 0 0 0\nouter loop\nvertex 0 0 0\nvertex 0 0 "
         "1\nvertex 1 0 0\nendloop\nendfacet\nfacet normal 0 0 0\nouter loop\nvertex 1 0 0\nvertex "
         "0 0 1\nvertex 0 1 0\nendloop\nendfacet\nendsolid tetra\n";
  std::ofstream(root / "robot.urdf")
      << R"(<robot name="tensor"><link name="base"/><link name="tip"><inertial><origin xyz=".1 .02 .03" rpy="0 0 1.5707963267948966"/><mass value=".2"/><inertia ixx=".0001" iyy=".0002" izz=".0002"/></inertial><collision><origin xyz=".02 0 0"/><geometry><mesh filename="tetra.stl" scale=".1 .2 .3"/></geometry></collision></link><joint name="fixed" type="fixed"><parent link="base"/><child link="tip"/><origin xyz=".3 0 0"/></joint></robot>)";
  auto     model = s::UrdfModel::load((root / "robot.urdf").string());
  s::World w(settings());
  auto     ids = model.instantiate(w, {1, 2, 3}, d::axis_angle({0, 0, 1}, std::acos(-1.) / 2));
  auto     b   = w.body(ids.links.at("tip"));
  EXPECT_NEAR(b.position[0], 1, 1e-12);
  EXPECT_NEAR(b.position[1], 2.3, 1e-12);
  EXPECT_NEAR(b.inertia_tensor[0], .0002, 1e-14);
  EXPECT_NEAR(b.inertia_tensor[4], .0001, 1e-14);
  EXPECT_EQ(b.center_of_mass, (s::Vec3{.1, .02, .03}));
  ASSERT_TRUE(b.colliders);
  ASSERT_EQ(b.colliders->size(), 1u);
  auto c = b.colliders->at(0);
  EXPECT_EQ(c.shape, s::Shape::ConvexHull);
  ASSERT_TRUE(c.vertices);
  EXPECT_EQ(c.vertices->size(), 4u);
  auto bounds = d::bounds(b);
  EXPECT_NEAR(bounds.lo[0], .8, 1e-12);
  EXPECT_NEAR(bounds.hi[1], 2.42, 1e-12);
  std::filesystem::remove_all(root);
}
TEST(NativeWorkcell, ExplicitReferenceRequestNeverSilentlyUsesNative) {
  EXPECT_EQ(s::articulated_physics_version(s::ArticulatedBackend::Native), "datamunge-vsim-1");
  if (s::articulated_physics_version().empty()) {
    s::World world;
    EXPECT_THROW(
        s::detail::StreamDynamics(fixture(), world, 1. / 960, 2, s::ArticulatedBackend::MuJoCo),
        std::runtime_error);
  }
  EXPECT_THROW(s::articulated_physics_version(s::ArticulatedBackend(99)), std::invalid_argument);
}
TEST(NativeContacts, GearedFiniteForceJawsLiftAFreePayloadByFriction) {
  auto o              = settings();
  o.block_joints      = true;
  o.gravity           = {0, -9.81, 0};
  o.solver_iterations = 12;
  s::World w(o);
  s::Body  wrist;
  wrist.motion          = s::Motion::Kinematic;
  wrist.position        = {0, .5, 0};
  wrist.collision_group = 0;
  auto parent           = w.add_body(wrist);
  auto jaw              = moving();
  jaw.collision_group   = 2;
  jaw.dimensions        = {.02, .12, .10};
  jaw.mass              = .04;
  jaw.position          = {-.08, .5, 0};
  jaw.friction          = .8;
  jaw.restitution       = 0;
  auto left             = w.add_body(jaw);
  jaw.position[0]       = .08;
  auto     right        = w.add_body(jaw);
  s::Joint joint;
  joint.type         = s::JointType::Slider;
  joint.axis         = {1, 0, 0};
  joint.parent       = parent;
  joint.child        = left;
  joint.armature     = 1;
  joint.effort_limit = 8;
  w.add_joint(joint);
  joint.child = right;
  w.add_joint(joint);
  w.add_gear(left, right, -1);
  s::Body floor;
  floor.position    = {0, .44, 0};
  floor.dimensions  = {1, .04, 1};
  floor.restitution = 0;
  w.add_body(floor);
  auto fruit            = moving();
  fruit.shape           = s::Shape::Sphere;
  fruit.dimensions      = {.078, .078, .078};
  fruit.position        = {0, .5, 0};
  fruit.mass            = .16;
  fruit.collision_group = 4;
  fruit.friction        = .8;
  fruit.restitution     = 0;
  auto payload          = w.add_body(fruit);
  for (int k = 0; k < 2880; ++k) {
    if (k == 960) {
      auto b            = w.body(parent);
      b.linear_velocity = {0, .1, 0};
      w.set_body(parent, b);
    }
    w.set_joint_motor(left, .08, 80, 8);
    w.set_joint_motor(right, -.08, 80, 8);
    w.step(1. / 960);
  }
  double loads[2]{};
  for (auto c : w.contacts())
    for (int i = 0; i < 2; ++i) {
      auto id = i ? right : left;
      if ((c.first == id && c.second == payload) || (c.second == id && c.first == payload))
        loads[i] += c.impulse * 960;
    }
  // A sphere can seat below the pad centres. Require a 10 cm free lift,
  // sustained upward motion and opposing loads rather than a rigid attachment.
  EXPECT_GT(w.body(payload).position[1], .60);
  for (auto contact : w.contacts())
    EXPECT_GT(contact.separation, -.003);
  EXPECT_NEAR(w.body(payload).linear_velocity[1], .1, .015);
  EXPECT_GT(loads[0], .05);
  EXPECT_GT(loads[1], .05);
  EXPECT_LE(std::abs(w.joint_motor_effort(left)), 8 + 1e-10);
  EXPECT_NEAR(w.joint_state(left).position, -w.joint_state(right).position, .0001);
}

TEST(NativeDynamics, ProjectedDriveCacheInvalidatesForInitialSnapshotsAndBodyEdits) {
  auto o         = settings();
  o.block_joints = true;
  s::World w(o);
  auto     b  = moving();
  b.mass      = 2;
  auto     id = w.add_body(b);
  s::Joint j;
  j.child    = id;
  j.type     = s::JointType::Slider;
  j.axis     = {1, 0, 0};
  j.armature = 3;
  w.add_joint(j);
  auto initial = w.save_state();
  w.step(.001);
  w.restore_state(initial);
  w.set_joint_motor(id, 10, 1e9, 2);
  w.step(.001);
  EXPECT_NEAR(w.joint_state(id).velocity, .0004, 1e-10);
  auto edited = w.body(id);
  edited.mass = 7;
  w.set_body(id, edited);
  w.set_joint_motor(id, 10, 1e9, 2);
  auto before = w.joint_state(id).velocity;
  w.step(.001);
  EXPECT_NEAR(w.joint_state(id).velocity - before, .0002, 1e-10);
}
