#include <gtest/gtest.h>

#include <array>
#include <datamunge/game/courier.hpp>
#include <filesystem>
#include <map>
#include <queue>
#include <set>
#include <tuple>
namespace g = datamunge::game;
namespace {
g::Entity named(const std::string& name, datamunge::sim::Vec3 position = {0, 0, 0}) {
  g::Entity e;
  e.name             = name;
  e.spatial.position = position;
  return e;
}
g::Scene grid() {
  g::Scene scene;
  auto     controller = named("Controller");
  controller.behavior = "courier_grid";
  scene.create(controller);
  scene.create(named("Status: Ready"));
  scene.create(named("Player"));
  scene.create(named("Crystal", {1, 0, 0}));
  scene.create(named("Exit", {2, 0, 0}));
  scene.create(named("Wall", {0, 0, 1}));
  scene.create(named("Trap", {0, 0, -1}));
  scene.define_prefab("Trophy", named("Trophy"));
  return scene;
}
g::Scene physical() {
  g::Scene scene;
  auto     controller = named("Controller");
  controller.behavior = "courier_3d";
  scene.create(controller);
  scene.create(named("Status: Ready"));
  auto player                     = named("Player", {0, .5, 0});
  player.physics_enabled          = true;
  player.spatial.motion           = datamunge::sim::Motion::Dynamic;
  player.spatial.shape            = datamunge::sim::Shape::Sphere;
  player.spatial.allowed_rotation = {false, false, false};
  scene.create(player);
  auto exit            = named("Exit", {10, .5, 0});
  exit.physics_enabled = true;
  exit.spatial.sensor  = true;
  scene.create(exit);
  auto floor               = named("Floor", {0, -.5, 0});
  floor.physics_enabled    = true;
  floor.spatial.dimensions = {30, 1, 30};
  scene.create(floor);
  scene.define_prefab("Trophy", named("Trophy"));
  return scene;
}
std::string fixture(const std::string& name) {
  return (std::filesystem::path(__FILE__).parent_path().parent_path() / "fixtures/game" /
          (name + ".json"))
      .string();
}
} // namespace
TEST(GameCourier, GridRulesBlockWallsCollectCrystalsAndReachVictory) {
  g::Simulation simulation(grid(), g::courier_behaviors());
  simulation.tick(1. / 60);
  simulation.tick(1. / 60, {{"move_z", 1}});
  EXPECT_EQ(simulation.scene().get(3).spatial.position, (datamunge::sim::Vec3{0, 0, 0}));
  EXPECT_NE(simulation.scene().get(2).name.find("wall blocks"), std::string::npos);
  simulation.tick(1. / 60, {{"move_x", 1}, {"move_z", 1}});
  EXPECT_EQ(simulation.scene().get(3).spatial.position[0], 0);
  simulation.tick(1. / 60, {{"move_x", 1}});
  EXPECT_FALSE(simulation.scene().alive(4));
  simulation.tick(1. / 60, {{"move_x", 1}});
  EXPECT_NE(simulation.scene().get(2).name.find("Victory!"), std::string::npos);
  const auto final = simulation.scene().get(3).spatial.position;
  simulation.tick(1. / 60, {{"move_x", -1}});
  EXPECT_EQ(simulation.scene().get(3).spatial.position, final);
}
TEST(GameCourier, GridTrapEndsOnlyThisBehaviorInstance) {
  g::Simulation failed(grid(), g::courier_behaviors()), other(grid(), g::courier_behaviors());
  failed.tick(1. / 60, {{"move_z", -1}});
  EXPECT_FALSE(failed.scene().alive(3));
  EXPECT_NE(failed.scene().get(2).name.find("Game over"), std::string::npos);
  EXPECT_NO_THROW(failed.tick(1. / 60));
  other.tick(1. / 60, {{"move_x", 1}});
  EXPECT_TRUE(other.scene().alive(3));
  EXPECT_FALSE(other.scene().alive(4));
}
TEST(GameCourier, PhysicalJumpUsesAGroundedInputEdgeAndBoundedAcceleration) {
  g::Simulation simulation(physical(), g::courier_behaviors());
  simulation.tick(1. / 60, {{"jump", 1}, {"move_x", 1}});
  const auto first = simulation.scene().get(3).spatial;
  EXPECT_GT(first.linear_velocity[1], 10);
  EXPECT_LE(first.linear_velocity[0], .51);
  simulation.tick(1. / 60, {{"jump", 1}, {"move_x", 1}});
  const auto second = simulation.scene().get(3).spatial;
  EXPECT_LT(second.linear_velocity[1], first.linear_velocity[1]);
  EXPECT_LE(second.linear_velocity[0], 1.01);
}
TEST(GameCourier, PhysicalPickupVictoryAndHazardUseNativeProximity) {
  auto scene            = physical();
  auto exit             = scene.get(4);
  exit.spatial.position = {0, .5, 0};
  scene.set(4, exit);
  auto crystal            = named("Crystal 1", {0, .5, 0});
  crystal.physics_enabled = true;
  crystal.spatial.sensor  = true;
  const auto    pickup    = scene.create(crystal);
  g::Simulation win(scene, g::courier_behaviors());
  win.tick(1. / 60);
  EXPECT_FALSE(win.scene().alive(pickup));
  EXPECT_NE(win.scene().get(2).name.find("VICTORY!"), std::string::npos);
  auto danger            = physical();
  auto hazard            = named("Hazard", {0, .5, 0});
  hazard.physics_enabled = true;
  hazard.spatial.sensor  = true;
  danger.create(hazard);
  g::Simulation lose(danger, g::courier_behaviors());
  lose.tick(1. / 60);
  EXPECT_NE(lose.scene().get(2).name.find("Lava!"), std::string::npos);
}
TEST(GameCourier, BothFrozenScenesRunUsingOnlyNativeBehaviorFactories) {
  for (bool three : {false, true}) {
    const auto    name   = three ? "crystal_courier_3d" : "crystal_courier";
    auto          scene  = g::load_scene(fixture(name), g::courier_import(three));
    const auto    camera = scene.get(3).spatial.position;
    g::Simulation simulation(scene, g::courier_behaviors());
    for (int i = 0; i < 30; ++i)
      simulation.tick(1. / 60);
    EXPECT_NE(simulation.scene().get(2).name, "Status: Ready");
    if (three) {
      EXPECT_NE(simulation.scene().get(3).spatial.position, camera);
      EXPECT_GT(simulation.scene().get(110).spatial.linear_velocity[1], .6);
    }
  }
}
TEST(GameCourier, FrozenGridLevelCanBeSolvedThroughTheNativeEngine) {
  auto scene  = g::load_scene(fixture("crystal_courier"), g::courier_import());
  using Tile  = std::pair<int, int>;
  using State = std::tuple<int, int, unsigned>;
  std::set<Tile>           walls, traps;
  std::map<Tile, unsigned> crystals;
  Tile                     start{}, goal{};
  unsigned                 all       = 0;
  int                      maximum_x = 0, maximum_z = 0;
  for (auto id : scene.entities()) {
    const auto& e = scene.get(id);
    const Tile  tile{int(e.spatial.position[0]), int(e.spatial.position[2])};
    if (e.name == "Wall") {
      walls.insert(tile);
      maximum_x = std::max(maximum_x, tile.first);
      maximum_z = std::max(maximum_z, tile.second);
    } else if (e.name == "Trap")
      traps.insert(tile);
    else if (e.name == "Player")
      start = tile;
    else if (e.name == "Exit")
      goal = tile;
    else if (e.name == "Crystal") {
      const unsigned bit = 1u << crystals.size();
      crystals[tile]     = bit;
      all |= bit;
    }
  }
  ASSERT_LE(crystals.size(), 16);
  const State       initial{start.first, start.second, 0};
  std::queue<State> frontier;
  frontier.push(initial);
  std::map<State, std::pair<State, int>> visited;
  visited[initial] = {initial, -1};
  const std::array<Tile, 4> moves{{{1, 0}, {-1, 0}, {0, 1}, {0, -1}}};
  State                     final{};
  bool                      found = false;
  while (!frontier.empty()) {
    const auto state = frontier.front();
    frontier.pop();
    const auto [x, z, collected] = state;
    if (Tile{x, z} == goal && collected == all) {
      final = state;
      found = true;
      break;
    }
    for (int action = 0; action < 4; ++action) {
      const Tile next{x + moves[action].first, z + moves[action].second};
      if (next.first < 0 || next.second < 0 || next.first > maximum_x || next.second > maximum_z ||
          walls.count(next) || traps.count(next))
        continue;
      const auto  pickup = crystals.find(next);
      const State target{
          next.first, next.second, collected | (pickup == crystals.end() ? 0 : pickup->second)};
      if (visited.emplace(target, std::make_pair(state, action)).second)
        frontier.push(target);
    }
  }
  ASSERT_TRUE(found);
  std::vector<int> route;
  for (auto state = final; state != initial; state = visited.at(state).first)
    route.push_back(visited.at(state).second);
  g::Simulation simulation(scene, g::courier_behaviors());
  for (auto action = route.rbegin(); action != route.rend(); ++action)
    simulation.tick(
        1. / 60,
        {{"move_x", double(moves[*action].first)}, {"move_z", double(moves[*action].second)}});
  EXPECT_NE(simulation.scene().get(2).name.find("Victory!"), std::string::npos);
}
