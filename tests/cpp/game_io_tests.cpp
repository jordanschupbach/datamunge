#include <gtest/gtest.h>

#include <datamunge/game/io.hpp>
#include <datamunge/game/simulation.hpp>
#include <filesystem>
#include <limits>
namespace g = datamunge::game;
TEST(GameIO, ExhaustedSignedIdCounterRoundTripsWithoutMutation) {
  auto scene = g::parse_scene(R"({"version":1,"next_id":9223372036854775806,"entities":[]})");
  EXPECT_EQ(scene.create(), 9223372036854775806ULL);
  const auto encoded = g::encode_scene(scene);
  auto       copy    = g::parse_scene(encoded);
  EXPECT_EQ(g::encode_scene(copy), encoded);
  EXPECT_THROW(copy.create(), std::overflow_error);
  EXPECT_EQ(g::encode_scene(copy), encoded);
}
TEST(GameIO, ExactIdsForwardParentsAndDeterministicRoundTrips) {
  auto scene = g::parse_scene(R"({"version":1,"next_id":9007199254740997,"entities":[
    {"id":9007199254740995,"components":{"name":"Child","rigid_body":{"motion":"dynamic"},"joint":{"parent":3}}},
    {"id":3,"components":{"name":"Parent","rigid_body":{}}}],"prefabs":{"Box":{"name":"Spawn"}}})");
  EXPECT_EQ(scene.get(9007199254740995ULL).joint->parent, 3);
  const auto encoded = g::encode_scene(scene);
  const auto copy    = g::parse_scene(encoded);
  EXPECT_EQ(g::encode_scene(copy), encoded);
  EXPECT_EQ(scene.spawn("Box"), 9007199254740997ULL);
  g::Simulation simulation(copy);
  EXPECT_NO_THROW(simulation.tick(1. / 60));
}
TEST(GameIO, NativeMaterialsMotionMasksAndFiniteDoubleValuesRoundTrip) {
  g::Scene  scene;
  g::Entity e;
  e.name                     = "Snowman \xe2\x98\x83";
  e.physics_enabled          = true;
  e.spatial.motion           = datamunge::sim::Motion::Dynamic;
  e.spatial.position         = {.12345678901234567, 2, 3};
  e.spatial.linear_damping   = .23456789012345678;
  e.spatial.allowed_rotation = {false, true, false};
  e.mesh                     = g::Mesh{};
  e.mesh->primitive          = g::Primitive::Crystal;
  e.camera                   = g::Camera{};
  e.behavior                 = "native";
  const auto id              = scene.create(e);
  const auto copy            = g::parse_scene(g::encode_scene(scene));
  EXPECT_EQ(copy.get(id).spatial.position, e.spatial.position);
  EXPECT_EQ(copy.get(id).spatial.linear_damping, e.spatial.linear_damping);
  EXPECT_EQ(copy.get(id).spatial.allowed_rotation, e.spatial.allowed_rotation);
  EXPECT_EQ(copy.get(id).mesh->primitive, e.mesh->primitive);
  EXPECT_EQ(copy.get(id).behavior, "native");
  EXPECT_EQ(copy.get(id).name, e.name);
}
TEST(GameIO, LegacySourceRequiresExplicitReplacementAndUnusedMappingsFail) {
  const std::string input =
      R"json({"version":1,"entities":[{"id":1,"components":{"script":{"source":"error('never execute')"}}}]})json";
  EXPECT_THROW(g::parse_scene(input), std::invalid_argument);
  g::SceneImport mapping;
  mapping.entities[1] = "idle";
  const auto scene    = g::parse_scene(input, mapping);
  EXPECT_EQ(scene.get(1).behavior, "idle");
  EXPECT_EQ(g::encode_scene(scene).find("never execute"), std::string::npos);
  g::BehaviorRegistry registry;
  registry.add("idle", [] { return std::make_unique<g::Behavior>(); });
  g::Simulation simulation(scene, registry);
  simulation.tick(1. / 60);
  mapping.entities[2] = "unused";
  EXPECT_THROW(g::parse_scene(input, mapping), std::invalid_argument);
  mapping.entities.erase(2);
  mapping.entities[1] = "";
  EXPECT_THROW(g::parse_scene(input, mapping), std::invalid_argument);
  mapping.entities[1] = "idle";
  EXPECT_THROW(
      g::parse_scene(R"({"version":1,"entities":[{"id":1,"components":{}}]})", mapping),
      std::invalid_argument);
}
TEST(GameIO, StrictSyntaxKeysUnicodeAndGraphFailuresAreRejected) {
  const std::vector<std::string> invalid = {
      "",
      "{}",
      R"({"version":1,"version":1,"entities":[]})",
      R"({"version":1,"entities":[],"extra":0})",
      R"({"version":1,"entities":[{"id":1.0,"components":{}}]})",
      R"({"version":1,"entities":[{"id":0,"components":{}}]})",
      R"({"version":1,"entities":[],})",
      R"({"version":01,"entities":[]})",
      R"({"version":1e,"entities":[]})",
      R"({"version":1,"entities":[{"id":1,"components":{}},{"id":1,"components":{}}]})",
      R"({"version":1,"entities":[{"id":1,"components":{"name":"\ud800"}}]})",
      R"({"version":1,"entities":[{"id":1,"components":{"name":"\udfff"}}]})",
      R"({"version":1,"entities":[{"id":1,"components":{"transform":{"position":[1e999,0,0]}}}]})",
      R"({"version":1,"entities":[{"id":1,"components":{"rigid_body":{"motion":"dynamic"},"joint":{"parent":2}}}]})",
      R"({"version":1,"entities":[{"id":1,"components":{"rigid_body":{"motion":"dynamic"},"joint":{"parent":2}}},{"id":2,"components":{"rigid_body":{"motion":"dynamic"},"joint":{"parent":1}}}]})"};
  for (const auto& input : invalid)
    EXPECT_THROW(g::parse_scene(input), std::invalid_argument) << input;
  EXPECT_THROW(g::parse_scene(std::string(16 * 1024 * 1024 + 1, ' ')), std::length_error);
  EXPECT_THROW(g::parse_scene(std::string(66, '[') + std::string(66, ']')), std::length_error);
  auto invalid_utf8 = std::string(R"({"version":1,"entities":[{"id":1,"components":{"name":")") +
                      "\xc0\x80" + R"("}}]})";
  EXPECT_THROW(g::parse_scene(invalid_utf8), std::invalid_argument);
  const auto unicode =
      g::parse_scene(R"({"version":1,"entities":[{"id":1,"components":{"name":"\ud83d\ude80"}}]})");
  EXPECT_EQ(unicode.get(1).name, "\xf0\x9f\x9a\x80");
}
TEST(GameIO, RealReferenceScenesImportWithoutExecutingLegacyScripts) {
  const auto root = std::filesystem::path(__FILE__).parent_path().parent_path() / "fixtures/game";
  for (const auto& name : {"crystal_courier", "crystal_courier_3d"}) {
    const auto path = (root / (std::string(name) + ".json")).string();
    EXPECT_THROW(g::load_scene(path), std::invalid_argument);
    g::SceneImport mapping;
    mapping.entities[1] = "courier";
    if (std::string(name) == "crystal_courier_3d") {
      mapping.entities[3]   = "camera";
      mapping.entities[110] = "lift";
    }
    const auto scene = g::load_scene(path, mapping);
    EXPECT_GT(scene.size(), 20);
    EXPECT_EQ(scene.get(1).behavior, "courier");
    EXPECT_EQ(g::encode_scene(g::parse_scene(g::encode_scene(scene))), g::encode_scene(scene));
  }
}
TEST(GameIO, PrefabScriptMappingsAndLegacyGeometryToleranceAreExplicit) {
  const std::string json =
      R"({"version":1,"entities":[],"prefabs":{"Ball":{"transform":{"scale":[1,1.000000001,1]},"rigid_body":{"shape":"sphere"},"script":{"source":"legacy"}}}})";
  g::SceneImport mapping;
  mapping.prefabs["Ball"] = "ball";
  auto       scene        = g::parse_scene(json, mapping);
  const auto id           = scene.spawn("Ball");
  EXPECT_EQ(scene.get(id).behavior, "ball");
  EXPECT_EQ(scene.get(id).spatial.dimensions, (datamunge::sim::Vec3{1, 1, 1}));
  EXPECT_THROW(g::parse_scene(json), std::invalid_argument);
  EXPECT_THROW(g::load_scene(""), std::invalid_argument);
  EXPECT_THROW(g::load_scene(".datamunge-missing-scene.json"), std::runtime_error);
}
