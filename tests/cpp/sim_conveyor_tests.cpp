#include <gtest/gtest.h>

#include <cmath>
#include <datamunge/sim/conveyor.hpp>
#include <limits>
namespace s = datamunge::sim;
namespace {
void run(s::ConveyorEnvironment& env, int ticks) {
  for (int i = 0; i < ticks; ++i)
    env.step(1. / 120);
}
} // namespace
TEST(SimConveyor, SeededArrivalStopsAndWaitsForBothFruitClasses) {
  bool                   apple = false, banana = false;
  s::ConveyorEnvironment env;
  for (unsigned seed = 0; seed < 8; ++seed) {
    env.reset(seed);
    auto start = env.observation();
    apple |= start.kind == s::FruitKind::Apple;
    banana |= start.kind == s::FruitKind::Banana;
    const auto count = env.world().body_count();
    EXPECT_EQ(start.position[0], -.9);
    EXPECT_THROW(env.next_fruit(), std::logic_error);
    run(env, 720);
    auto arrived = env.observation();
    EXPECT_EQ(arrived.stage, s::ConveyorStage::Waiting);
    EXPECT_GT(arrived.arrival_time, 4.4);
    EXPECT_LT(arrived.arrival_time, 5.5);
    EXPECT_NEAR(arrived.position[0], .10, .012);
    EXPECT_LT(std::abs(arrived.velocity[0]), .002);
    run(env, 360);
    auto held = env.observation();
    EXPECT_NEAR(held.position[0], arrived.position[0], .001);
    EXPECT_DOUBLE_EQ(held.belt_travel, arrived.belt_travel);
    EXPECT_EQ(env.world().body_count(), count);
    EXPECT_THROW(env.next_fruit(), std::logic_error);
    env.reset(seed);
    run(env, 720);
    auto replay = env.observation();
    EXPECT_EQ(replay.kind, start.kind);
    EXPECT_EQ(replay.position, arrived.position);
    EXPECT_DOUBLE_EQ(replay.arrival_time, arrived.arrival_time);
  }
  EXPECT_TRUE(apple);
  EXPECT_TRUE(banana);
}
TEST(SimConveyor, BinDetectionAndNextPayloadRequireCompletedPlacement) {
  s::ConveyorEnvironment env;
  for (bool correct : {true, false}) {
    env.reset(42);
    auto s0           = env.observation();
    auto b            = env.world().body(s0.fruit);
    bool negative     = (s0.kind == s::FruitKind::Apple) == correct;
    b.position        = {.1, .52, negative ? -.42 : .42};
    b.linear_velocity = {0, 0, 0};
    env.world().set_body(s0.fruit, b);
    run(env, 120);
    EXPECT_EQ(
        env.observation().stage, correct ? s::ConveyorStage::Sorted : s::ConveyorStage::Missorted);
    auto count = env.world().body_count();
    env.next_fruit();
    EXPECT_EQ(env.observation().stage, s::ConveyorStage::Travelling);
    EXPECT_EQ(env.world().body_count(), count);
    EXPECT_NE(env.observation().fruit, s0.fruit);
  }
}
TEST(SimConveyor, InvalidDeltaDoesNotMutateAndScenesAreIndependent) {
  s::ConveyorEnvironment env;
  auto                   before = env.observation();
  for (double dt : {0., -1., 1., std::numeric_limits<double>::quiet_NaN()})
    EXPECT_THROW(env.step(dt), std::invalid_argument);
  EXPECT_EQ(before.position, env.observation().position);
  EXPECT_EQ(before.time, env.observation().time);
  auto scene  = env.static_scene();
  auto actors = env.dynamic_scene();
  EXPECT_GT(scene.objects.size(), 20);
  EXPECT_GT(actors.objects.size(), 13);
  scene.objects.clear();
  EXPECT_GT(env.static_scene().objects.size(), 20);
  auto b     = env.world().body(before.fruit);
  b.position = {2, .8, 0};
  env.world().set_body(before.fruit, b);
  env.step(1. / 120);
  EXPECT_EQ(env.observation().stage, s::ConveyorStage::Lost);
  EXPECT_NO_THROW(env.next_fruit());
}
TEST(SimWorld, ConveyorSurfaceDrivesAndBrakesWithoutMovingSupport) {
  s::WorldOptions o;
  o.body_capacity    = 8;
  o.pair_capacity    = 64;
  o.contact_capacity = 64;
  s::World world(o);
  s::Body  belt;
  belt.dimensions       = {4, .1, 1};
  belt.friction         = 1;
  belt.surface_velocity = {.25, 50, 0}; // Normal surface velocity must not launch fruit.
  auto    belt_id       = world.add_body(belt);
  s::Body fruit;
  fruit.motion           = s::Motion::Dynamic;
  fruit.position         = {-1, .105, 0};
  fruit.shape            = s::Shape::Sphere;
  fruit.dimensions       = {.1, .1, .1};
  fruit.friction         = 1;
  fruit.restitution      = 0;
  fruit.allowed_rotation = {false, false, false};
  fruit.linear_damping   = 0;
  auto id                = world.add_body(fruit);
  for (int i = 0; i < 120; ++i)
    world.step(1. / 120);
  EXPECT_NEAR(world.body(id).linear_velocity[0], .25, .002);
  EXPECT_NEAR(world.body(id).position[1], .10, .003);
  EXPECT_GT(world.body(id).position[0], -.8);
  EXPECT_EQ(world.body(belt_id).position, belt.position);
  belt.surface_velocity = {0, 0, 0};
  world.set_body(belt_id, belt);
  for (int i = 0; i < 120; ++i)
    world.step(1. / 120);
  EXPECT_NEAR(world.body(id).linear_velocity[0], 0, .002);
  // Restart after the payload has gone to sleep: material motion must wake it.
  const auto stopped    = world.body(id).position[0];
  belt.surface_velocity = {.25, 0, 0};
  world.set_body(belt_id, belt);
  for (int i = 0; i < 120; ++i)
    world.step(1. / 120);
  EXPECT_GT(world.body(id).position[0], stopped + .20);
  belt.surface_velocity = {std::numeric_limits<double>::quiet_NaN(), 0, 0};
  EXPECT_THROW(world.set_body(belt_id, belt), std::invalid_argument);
}

TEST(Conveyor, DiverseSeedsVaryShapePalettePoseAndPhysicalEntryReproducibly) {
  s::ConveyorEnvironment             env(s::FruitVariation::Diverse);
  std::array<std::array<bool, 6>, 2> seen{};
  double                             minimum = 1, maximum = -1, first_yaw = -1;
  bool                               rotated = false;
  for (unsigned seed = 0; seed < 160; ++seed) {
    env.reset(seed);
    auto a = env.observation();
    EXPECT_EQ(a.variation, s::FruitVariation::Diverse);
    ASSERT_LT(a.appearance.prototype, 6);
    seen[int(a.kind)][a.appearance.prototype] = true;
    EXPECT_GE(a.position[0], -1.);
    EXPECT_LE(a.position[0], -.7);
    EXPECT_GE(a.position[2], -.065);
    EXPECT_LE(a.position[2], .065);
    minimum = std::min(minimum, a.position[2]);
    maximum = std::max(maximum, a.position[2]);
    if (first_yaw < 0)
      first_yaw = a.appearance.yaw;
    else
      rotated |= std::abs(first_yaw - a.appearance.yaw) > 1;
    auto scene = env.dynamic_scene();
    ASSERT_EQ(scene.textures.size(), 1);
    env.reset(seed);
    auto b = env.observation();
    EXPECT_EQ(a.rotation, b.rotation);
    EXPECT_EQ(a.position, b.position);
    EXPECT_EQ(a.appearance.skin, b.appearance.skin);
    EXPECT_EQ(scene.textures[0].pixels, env.dynamic_scene().textures[0].pixels);
  }
  EXPECT_GT(maximum - minimum, .11);
  EXPECT_TRUE(rotated);
  for (auto row : seen)
    for (bool present : row)
      EXPECT_TRUE(present);
}

TEST(Conveyor, RealisticAssetsAreDeterministicAndDoNotChangeThePhysicalTask) {
  s::ConveyorEnvironment simple(s::FruitVariation::Diverse), real(s::FruitVariation::Diverse);
  real.set_visual_style(s::WorkcellVisualStyle::Realistic);
  simple.reset(17);
  real.reset(17);
  EXPECT_EQ(simple.observation().position, real.observation().position);
  auto plain = simple.static_scene(), decorated = real.static_scene();
  EXPECT_GT(decorated.objects.size(), plain.objects.size() + 50);
  EXPECT_GT(decorated.textures.size(), 10);
  auto fruit = real.dynamic_scene();
  EXPECT_EQ(fruit.textures.size(), 3);
  EXPECT_EQ(fruit.textures[0].pixels, real.dynamic_scene().textures[0].pixels);
  EXPECT_GT(fruit.meshes.back().positions.size(), 0);
  for (int i = 0; i < 720; ++i) {
    simple.step(1. / 120);
    real.step(1. / 120);
  }
  EXPECT_EQ(simple.observation().position, real.observation().position);
  EXPECT_EQ(simple.observation().rotation, real.observation().rotation);
  EXPECT_EQ(simple.observation().stage, real.observation().stage);
  EXPECT_EQ(real.visual_style(), s::WorkcellVisualStyle::Realistic);
  real.reset(28);
  EXPECT_EQ(real.visual_style(), s::WorkcellVisualStyle::Realistic);
  EXPECT_THROW(
      real.set_visual_style(static_cast<s::WorkcellVisualStyle>(8)), std::invalid_argument);
}
