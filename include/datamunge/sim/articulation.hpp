#pragma once
/// @file articulation.hpp
/// @brief Renderer-independent URDF rigid mechanisms instantiated in the native solver.
#include <datamunge/sim/world.hpp>
#include <map>
#include <string>
namespace datamunge::sim {
/// @brief Explicit physical import approximations for URDF collision meshes and missing inertias.
/// Source units are metres/radians/kg. No external physics engine is used.
/// @par Example
/// @code
/// dm::sim::UrdfPhysicsOptions options; options.missing_mass=.04;
/// @endcode
struct UrdfPhysicsOptions {
  double missing_mass{
      0}; ///< Zero rejects missing dynamic inertias; positive explicitly opts into nominal data.
  Vec3 missing_inertia{
      .00002,
      .00002,
      .00002};             ///< Nominal diagonal COM inertia when missing_mass is positive.
  double joint_damping{0}; ///< Default only when URDF has no dynamics damping.
  double hinge_armature{
      0}; ///< Nominal reflected arm inertia, kg m²; not inferred from URDF transmissions.
  double slider_armature{0}; ///< Nominal reflected slide inertia, kg.
  double slider_damping{0};  ///< Default slide damping if absent from URDF.
  std::map<std::string, std::vector<std::array<double, 2>>>
      mesh_x_slices; ///< Optional per-link convex slab decomposition in mesh-local X metres.
  std::size_t hull_directions{
      128}; ///< Deterministic support sampling; zero retains every source vertex.
};
/// @brief Body identifiers for one model instance, scoped to its borrowed world.
/// Copies contain identifiers only; removing bodies invalidates those identifiers.
/// @par Example
/// @code
/// dm::sim::Articulation instance;
/// @endcode
struct Articulation {
  std::map<std::string, BodyId> links;  ///< One rigid body per URDF link, including fixed links.
  std::map<std::string, BodyId> joints; ///< Joint name to constrained child body.
};
/// @brief Immutable native URDF dynamics/collision model, independent of rendering and MuJoCo.
/// Supports rooted trees, fixed/revolute/continuous/prismatic joints, affine mimic
/// gearing, full rotated COM tensors, and box/sphere/cylinder/STL collision geometry.
/// Cylinders use 32-sided hulls; meshes use deterministic sampled convex hulls
/// (concavities require multiple collision elements). Missing inertias are rejected
/// unless explicitly given nominal import options. Copies share immutable geometry.
/// @par Example
/// @code
/// auto model=dm::sim::UrdfModel::load("robot.urdf");
/// dm::sim::World world;
/// auto robot=model.instantiate(world);
/// @endcode
/// @par References
/// - URDF specification: https://wiki.ros.org/urdf/XML
/// @see World::set_joint_motor(), World::add_gear()
class UrdfModel {
 public:
  /// @brief Parse a complete physical URDF and load its collision assets.
  /// @param path Existing URDF; mesh paths are relative to its directory.
  /// @param options Explicit approximation/nominal-parameter controls.
  /// @return Shared immutable native model.
  /// @throws std::invalid_argument Invalid graph, unsupported geometry or missing physical data.
  /// @throws std::runtime_error Asset/XML I/O failure.
  /// @par Example
  /// See the class example.
  static UrdfModel load(const std::string& path, UrdfPhysicsOptions options = {});
  /// @brief Add the model to an empty or populated native world with a fixed root.
  /// Validate capacity beforehand; on failure no partially built bodies are retained.
  /// @param world Borrowed destination world; must outlive the returned identifiers.
  /// @param position Root reference origin in the world's axes, default zero.
  /// @param rotation Root WXYZ orientation, default identity.
  /// @return Link/joint identifiers for this independent instance.
  /// @throws std::invalid_argument Invalid root transform.
  /// @throws std::length_error Insufficient destination capacity.
  /// @par Example
  /// See the class example.
  Articulation instantiate(World& world, Vec3 position = {}, Quaternion rotation = {1, 0, 0, 0})
      const;

 private:
  struct Impl;                       ///< Immutable physical link tree and shared convex geometry.
  std::shared_ptr<const Impl> impl_; ///< Shared imported definition.
};
} // namespace datamunge::sim
