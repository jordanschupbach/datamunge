#pragma once
/// @file scene.hpp
/// @brief Validated native game entities, prefabs and snapshot-scoped identifiers.
#include <cstdint>
#include <datamunge/sim/types.hpp>
#include <map>
#include <optional>
#include <string>
#include <vector>
namespace datamunge::game {
struct SceneImport;
/// @brief Scene-snapshot scoped entity identifier, positive and signed-64-bit compatible.
/// @par Example
/// @code
/// const dm::game::EntityId invalid=0;
/// @endcode
using EntityId = std::uint64_t;
/// @brief Procedural mesh primitive independent of collision shape.
/// @par Example
/// @code
/// const auto kind=dm::game::Primitive::Crystal;
/// @endcode
enum class Primitive {
  Box,    ///< Unit box with flat faces.
  Sphere, ///< Unit sphere with smooth normals.
  Crystal ///< Unit octahedron with flat faces.
};
/// @brief Linear-RGB metallic/roughness mesh appearance.
/// @par Example
/// @code
/// dm::game::Mesh mesh;mesh.validate();
/// @endcode
struct Mesh {
  Primitive primitive{Primitive::Box}; ///< Unit procedural primitive.
  sim::Vec3 color{.6, .6, .6};         ///< Linear RGB in [0,1].
  double    metallic{0};               ///< Metal fraction in [0,1].
  double    roughness{.6};             ///< Surface roughness in [.04,1].
  double    emissive{0};               ///< Color emission multiplier in [0,10].

  /// @brief Validate primitive, finite color and material ranges.
  /// @throws std::invalid_argument Invalid material values.
  /// @par Example
  /// @code
  /// const dm::game::Mesh mesh;mesh.validate();
  /// @endcode
  void validate() const;
};
/// @brief Perspective camera with right-handed local -Z view and zero-to-one depth.
/// @par Example
/// @code
/// const dm::game::Camera camera;camera.validate();
/// @endcode
struct Camera {
  double vertical_fov{60}; ///< Degrees strictly between zero and 180.
  double near_plane{.1};   ///< Positive near distance in metres.
  double far_plane{1000};  ///< Far distance greater than near_plane.

  /// @brief Validate finite FOV and clip-plane ordering.
  /// @throws std::invalid_argument Invalid camera projection settings.
  /// @par Example
  /// @code
  /// const dm::game::Camera camera;camera.validate();
  /// @endcode
  void validate() const;
};
/// @brief Owned scene components; spatial is the single source of transform and velocities.
/// spatial.position/rotation/dimensions describe the entity even when physics is
/// disabled. Enabling physics uses all spatial fields as the rigid-body settings.
/// No Lua source is executed: behavior names refer to explicitly registered native factories.
/// @par Example
/// @code
/// dm::game::Entity entity;entity.name="Player";entity.physics_enabled=true;
/// entity.spatial.motion=dm::sim::Motion::Dynamic;entity.validate();
/// @endcode
struct Entity {
  std::string name{"Entity"}; ///< Human-readable scalar UTF-8 name, 1..128 bytes without NUL.
  sim::Body   spatial;        ///< Pose/scale and optional physical settings in SI units.
  bool        physics_enabled{false}; ///< Whether a rigid body is compiled for this entity.
  std::optional<Mesh>       mesh;     ///< Optional appearance.
  std::optional<Camera>     camera;   ///< Optional camera using spatial pose.
  std::optional<sim::Joint> joint;    ///< Optional constraint; child is filled from the entity ID.
  std::string behavior; ///< Native scalar UTF-8 factory name, at most 128 bytes; empty means none.

  /// @brief Validate component fields before scene-level parent checks.
  /// @throws std::invalid_argument Invalid spatial/material/camera data or a joint without physics.
  /// @par Example
  /// @code
  /// const dm::game::Entity entity;entity.validate();
  /// @endcode
  void validate() const;
};
/// @brief Owned authoring scene with validated, transactional entity edits and prefabs.
/// IDs are never reused within a snapshot. Scene copies are separate snapshots;
/// references must not be retained across replacement. Methods are single-writer.
/// @par Example
/// @code
/// dm::game::Scene scene;const auto id=scene.create({});
/// std::print("{}\n",scene.alive(id)); // true
/// @endcode
class Scene {
 public:
  /// @brief Create an empty scene with fresh snapshot identity.
  /// @par Example
  /// @code
  /// dm::game::Scene scene;
  /// std::print("{}\n",scene.size()); // 0
  /// @endcode
  Scene();
  /// @brief Allocate a never-reused positive ID for a validated entity.
  /// @param entity Owned component value, default empty entity.
  /// @return Fresh entity ID.
  /// @throws std::invalid_argument Invalid components, missing parent or joint cycle.
  /// @throws std::length_error Scene capacity of 10000 reached.
  /// @throws std::overflow_error No signed-64-compatible next ID remains for serialization.
  /// @par Example
  /// @code
  /// dm::game::Scene scene;std::print("{}\n",scene.create({})); // 1
  /// @endcode
  EntityId create(Entity entity = {});
  /// @brief Remove a live entity only when no remaining joint references it.
  /// @param id Live snapshot-scoped ID.
  /// @throws std::out_of_range Unknown entity.
  /// @throws std::invalid_argument A child joint still names this parent.
  /// @par Example
  /// @code
  /// dm::game::Scene scene;const auto id=scene.create({});scene.destroy(id);
  /// @endcode
  void destroy(EntityId id);
  /// @brief Check a snapshot-scoped identifier without throwing.
  /// @param id Possible entity ID; zero is invalid.
  /// @return Whether the ID is live.
  bool alive(EntityId id) const noexcept;
  /// @brief Borrow one entity without granting unvalidated mutation.
  /// @param id Live snapshot-scoped entity ID.
  /// @return Reference valid until edit, erase or scene replacement.
  /// @throws std::out_of_range Unknown entity.
  /// @par Example
  /// @code
  /// dm::game::Scene scene;const auto id=scene.create({});
  /// std::print("{}\n",scene.get(id).name); // Entity
  /// @endcode
  const Entity& get(EntityId id) const;
  /// @brief Replace a live entity after validating the complete parent graph.
  /// @param id Live entity.
  /// @param entity Complete replacement, copied/moved.
  /// @throws std::out_of_range Unknown entity.
  /// @throws std::invalid_argument Invalid components or joint graph; scene unchanged.
  /// @par Example
  /// @code
  /// dm::game::Scene scene;const auto id=scene.create({});
  /// auto entity=scene.get(id);entity.name="Renamed";scene.set(id,entity);
  /// @endcode
  void set(EntityId id, Entity entity);
  /// @brief List IDs in ascending order.
  /// @return Owned stable-order snapshot.
  /// @par Example
  /// @code
  /// dm::game::Scene scene;std::print("{}\n",scene.entities().size()); // 0
  /// @endcode
  std::vector<EntityId> entities() const;
  /// @brief Return the number of live entities.
  /// @return Live entity count.
  std::size_t size() const noexcept { return entities_.size(); }
  /// @brief Read the scene revision for incremental consumers.
  /// @return Revision changes on every successful mutation.
  std::uint64_t revision() const noexcept { return revision_; }
  /// @brief Define or replace a validated component template.
  /// @param name Nonempty scalar UTF-8 prefab name, at most 128 bytes and without NUL.
  /// @param entity Template components; joints are not allowed because IDs are snapshot-scoped.
  /// @throws std::invalid_argument Empty name, invalid components or jointed prefab.
  /// @throws std::length_error More than 10000 distinct prefabs.
  /// @par Example
  /// @code
  /// dm::game::Scene scene;scene.define_prefab("Box",{});
  /// @endcode
  void define_prefab(std::string name, Entity entity);
  /// @brief Create an independent copy of a named prefab.
  /// @param name Existing template name.
  /// @return Fresh entity ID.
  /// @throws std::out_of_range Unknown prefab.
  /// @throws std::length_error Scene is full.
  /// @par Example
  /// @code
  /// dm::game::Scene scene;scene.define_prefab("Box",{});
  /// std::print("{}\n",scene.spawn("Box")); // 1
  /// @endcode
  EntityId spawn(const std::string& name);

 private:
  /// @brief Validated codec restores exact IDs and checks the complete graph before publication.
  friend Scene parse_scene(const std::string&, const SceneImport&);
  /// @brief Codec reads immutable authoring components, prefabs and the never-reused ID counter.
  friend std::string encode_scene(const Scene&);
  /// @brief Runtime reconciles already validated scene/physics transactions.
  friend class Simulation;
  std::map<EntityId, Entity>    entities_;    ///< Owned components, stable ordered IDs.
  std::map<std::string, Entity> prefabs_;     ///< Owned independent authoring templates.
  EntityId                      next_id_{1};  ///< Never reused during this snapshot's lifetime.
  std::uint64_t                 revision_{0}; ///< Last successful edit revision.

  /// @brief Validate the current graph using a proposed addition or replacement.
  /// @param id Proposed live or fresh entity identifier.
  /// @param entity Complete proposed components.
  void validate_graph(EntityId id, const Entity& entity) const;
  /// @brief Publish a globally fresh edit revision.
  void touch() noexcept;
};
} // namespace datamunge::game
