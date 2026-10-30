#pragma once
/// @file io.hpp
/// @brief Bounded native scene serialization and explicit legacy-script behavior mapping.
#include <datamunge/game/scene.hpp>
namespace datamunge::game {
/// @brief Explicit replacements for legacy Lua-bearing entities and prefabs.
/// Keys are original scene IDs or prefab names. Every nonempty legacy script
/// needs one nonempty native factory name; unused mappings are rejected.
/// Mapping never executes, translates or retains Lua source. Simulation later
/// validates mapped factory names against its BehaviorRegistry.
/// @par Example
/// @code
/// dm::game::SceneImport mapping;mapping.entities[1]="crystal_courier";
/// @endcode
struct SceneImport {
  std::map<EntityId, std::string>
      entities; ///< Original entity IDs to native behavior factory names.
  std::map<std::string, std::string> prefabs; ///< Prefab names to native behavior factory names.
};
/// @brief Parse a validated native/VSim version-one scene while preserving exact IDs and next_id.
/// Accepts strict scalar UTF-8 JSON, at most 16 MiB, 64 nested containers,
/// 10000 entities and 10000 prefabs. Duplicate/unknown keys, missing parents and
/// joint cycles fail before any scene is returned. Live IDs are in [1,INT64_MAX);
/// next_id may equal INT64_MAX for an exhausted scene. Legacy scripts require explicit
/// replacements. Native behavior and damping fields extend the original schema.
/// Sphere/capsule dimension differences within the original 1e-5 tolerance are
/// canonicalized to exact equal diameters. No source script is executed.
/// @param document Complete owned/borrowed JSON text; empty is invalid.
/// @param mapping Explicit legacy behavior replacements; empty by default.
/// @return Independent validated scene with fresh revision and original numeric IDs.
/// @throws std::invalid_argument Invalid JSON, schema, components, graph or script mapping.
/// @throws std::length_error Document, nesting, value or scene capacity limit exceeded.
/// @par Example
/// @code
/// const auto
/// scene=dm::game::parse_scene(R"({"version":1,"entities":[{"id":7,"components":{"name":"Player"}}]})");
/// std::print("{}\n",scene.get(7).name); // Player
/// @endcode
/// @par References
/// - T. Bray (2017), RFC 8259, https://www.rfc-editor.org/rfc/rfc8259.
/// - VSim version-one scene schema in the frozen reference manifest.
/// @see encode_scene(), load_scene(), SceneImport
Scene parse_scene(const std::string& document, const SceneImport& mapping = {});
/// @brief Encode a validated scene deterministically as version-one UTF-8 JSON.
/// Preserves live IDs, next_id, prefabs, authored components and native behavior
/// names. Writes no file and includes no solver state or Lua. Finite doubles use
/// sufficient decimal precision for exact round trips; keys and entity IDs are ordered.
/// @param scene Owned native scene to serialize read-only.
/// @return New JSON document ending with a newline, at most 16 MiB.
/// @throws std::invalid_argument Invalid UTF-8/component data.
/// @throws std::length_error Encoded document exceeds the document limit.
/// @par Example
/// @code
/// dm::game::Scene scene;scene.create({});const auto json=dm::game::encode_scene(scene);
/// const auto copy=dm::game::parse_scene(json);
/// std::print("{}\n",copy.size()); // 1
/// @endcode
/// @see parse_scene(), sim::World::save_state()
std::string encode_scene(const Scene& scene);
/// @brief Read a bounded scene file and parse it with explicit legacy replacements.
/// Reads at most 16 MiB before invoking the same transactional parser as parse_scene.
/// @param path Filesystem path encoded by the platform's narrow-path convention.
/// @param mapping Explicit native replacement names, empty by default.
/// @return Independent validated scene.
/// @throws std::runtime_error File cannot be opened/read.
/// @throws std::invalid_argument Invalid path, scene or script mapping.
/// @throws std::length_error File or parser capacity limit exceeded.
/// @par Example
/// @code
/// try { const auto scene=dm::game::load_scene(".datamunge-missing-scene.json"); }
/// catch(const std::runtime_error&) { /* Report an unavailable scene to the caller. */ }
/// @endcode
/// @see parse_scene()
Scene load_scene(const std::string& path, const SceneImport& mapping = {});
} // namespace datamunge::game
