#include <gtest/gtest.h>

#include <datamunge/game/clock.hpp>
#include <datamunge/game/simulation.hpp>
#include <limits>
namespace g = datamunge::game;
namespace s = datamunge::sim;
namespace {
s::WorldOptions options() {
  s::WorldOptions o;
  o.body_capacity = 32;
  o.pair_capacity = 128;
  o.contact_capacity = 128;
  return o;
}
g::Entity physical() {
  g::Entity e;
  e.physics_enabled = true;
  e.spatial.motion = s::Motion::Dynamic;
  e.spatial.position = {0, 3, 0};
  return e;
}
struct Counter : g::Behavior {
  int& starts;
  int& updates;
  double& input;
  Counter(int& s, int& u, double& i) : starts(s), updates(u), input(i) {}
  void on_start(g::Simulation&, g::EntityId, double) override { ++starts; }
  void on_update(g::Simulation& sim, g::EntityId, double) override {
    ++updates;
    input = sim.input("move");
  }
};
}  // namespace
TEST(GameScene, InvalidGraphEditsAreAtomicAndIdsNeverReuse) {
  g::Scene scene;
  auto parent = scene.create(physical());
  auto child = physical();
  s::Joint j;
  j.parent = parent;
  child.joint = j;
  auto id = scene.create(child);
  auto before = scene.revision();
  auto invalid = scene.get(parent);
  invalid.physics_enabled = false;
  EXPECT_THROW(scene.set(parent, invalid), std::invalid_argument);
  EXPECT_EQ(scene.revision(), before);
  EXPECT_THROW(scene.destroy(parent), std::invalid_argument);
  invalid = scene.get(parent);
  j.parent = id;
  invalid.joint = j;
  EXPECT_THROW(scene.set(parent, invalid), std::invalid_argument);
  EXPECT_EQ(scene.revision(), before);
  scene.destroy(id);
  scene.destroy(parent);
  EXPECT_GT(scene.create({}), id);
}
TEST(GameScene, PrefabsAndCameraMaterialValidation) {
  g::Scene scene;
  g::Entity e;
  e.name = "Template";
  scene.define_prefab("box", e);
  auto a = scene.spawn("box"), b = scene.spawn("box");
  e = scene.get(a);
  e.name = "Changed";
  scene.set(a, e);
  EXPECT_EQ(scene.get(b).name, "Template");
  g::Mesh mesh;
  mesh.roughness = 0;
  EXPECT_THROW(mesh.validate(), std::invalid_argument);
  g::Camera camera;
  camera.near_plane = camera.far_plane;
  EXPECT_THROW(camera.validate(), std::invalid_argument);
}
TEST(GameClock, FixedTicksDroppedTimeAndInvalidInput) {
  g::FixedClock clock(.01, 4);
  auto first = clock.advance(.025);
  EXPECT_EQ(first.steps, 2);
  EXPECT_NEAR(first.alpha, .5, 1e-14);
  auto second = clock.advance(.1);
  EXPECT_EQ(second.steps, 4);
  EXPECT_NEAR(second.dropped_seconds, .06, 1e-14);
  EXPECT_NEAR(second.alpha, .5, 1e-14);
  EXPECT_THROW(clock.advance(-1), std::invalid_argument);
  EXPECT_THROW(clock.advance(std::numeric_limits<double>::quiet_NaN()),
               std::invalid_argument);
  EXPECT_NEAR(clock.advance(0).alpha, .5, 1e-14);
  clock.reset();
  EXPECT_EQ(clock.advance(0).steps, 0);
}
TEST(GameSimulation, NativeStatePersistsAndInputsDoNotLatch) {
  int starts = 0, updates = 0;
  double input = 0;
  g::BehaviorRegistry registry;
  registry.add("counter", [&] {
    return std::make_unique<Counter>(starts, updates, input);
  });
  g::Scene scene;
  g::Entity e;
  e.behavior = "counter";
  scene.create(e);
  g::Simulation sim(scene, registry, options());
  sim.tick(1. / 60, {{"move", .5}});
  EXPECT_EQ(starts, 1);
  EXPECT_EQ(updates, 1);
  EXPECT_EQ(input, .5);
  sim.tick(1. / 60);
  EXPECT_EQ(starts, 1);
  EXPECT_EQ(updates, 2);
  EXPECT_EQ(input, 0);
  EXPECT_THROW(sim.tick(1. / 60, {{"move", 2}}), std::invalid_argument);
  EXPECT_EQ(updates, 2);
}
TEST(GameSimulation, SelfDestructionSkipsUpdateAndSpawnsStartNextTick) {
  int starts = 0, updates = 0;
  double input = 0;
  g::BehaviorRegistry registry;
  registry.add("counter", [&] {
    return std::make_unique<Counter>(starts, updates, input);
  });
  struct Spawner : g::Behavior {
    void on_start(g::Simulation& sim, g::EntityId self, double) override {
      g::Entity e;
      e.behavior = "counter";
      sim.create(e);
      sim.destroy(self);
    }
    void on_update(g::Simulation&, g::EntityId, double) override {
      throw std::runtime_error("destroyed behavior executed");
    }
  };
  registry.add("spawn", [] { return std::make_unique<Spawner>(); });
  g::Scene scene;
  g::Entity e;
  e.behavior = "spawn";
  auto id = scene.create(e);
  g::Simulation sim(scene, registry, options());
  sim.tick(1. / 60);
  EXPECT_FALSE(sim.scene().alive(id));
  EXPECT_EQ(starts, 0);
  sim.tick(1. / 60);
  EXPECT_EQ(starts, 1);
  EXPECT_EQ(updates, 1);
}
TEST(GameSimulation, MotionAndSceneEditsPreservePendingImpulses) {
  g::Scene scene;
  auto id = scene.create(physical());
  g::Simulation sim(scene, {}, options());
  sim.apply_impulse(id, {1, 0, 0});
  auto e = sim.scene().get(id);
  e.name = "Renamed";
  sim.set(id, e);
  g::Entity obstacle;
  obstacle.physics_enabled = true;
  obstacle.spatial.position = {20, 0, 0};
  sim.create(obstacle);
  sim.tick(1. / 60);
  EXPECT_GT(sim.scene().get(id).spatial.position[0], 0);
  EXPECT_GT(sim.scene().get(id).spatial.linear_velocity[0], .99);
  auto before = sim.scene().size();
  e.behavior = "unknown";
  EXPECT_THROW(sim.create(e), std::invalid_argument);
  EXPECT_EQ(sim.scene().size(), before);
}
TEST(GameSimulation, RemovingBehaviorsAndFactoryFailuresAreExplicit) {
  g::BehaviorRegistry registry;
  registry.add("bad", [] { return std::unique_ptr<g::Behavior>{}; });
  EXPECT_THROW(registry.create("bad"), std::runtime_error);
  EXPECT_THROW(registry.create("absent"), std::out_of_range);
  g::Scene scene;
  g::Entity e;
  e.behavior = "missing";
  scene.create(e);
  EXPECT_THROW((g::Simulation(scene, {}, options())), std::invalid_argument);
}
TEST(SimWorld, ForkCopiesPendingForcesWithoutSharingMutableState) {
  s::World world(options());
  auto b = physical().spatial;
  auto id = world.add_body(b);
  world.apply_force(id, {2, 0, 0});
  auto copy = world.fork();
  auto snapshot = world.save_state();
  world.step(1. / 60);
  copy->step(1. / 60);
  EXPECT_EQ(world.body(id).position, copy->body(id).position);
  EXPECT_THROW(copy->restore_state(snapshot), std::invalid_argument);
  world.apply_impulse(id, {1, 0, 0});
  EXPECT_NE(world.body(id).linear_velocity, copy->body(id).linear_velocity);
}
