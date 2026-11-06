#pragma once
/// @file types.hpp
/// @brief Native simulation coordinates and validated rigid-body descriptions.
#include <array>
#include <cstdint>
#include <memory>
#include <vector>
namespace datamunge::sim {
/// @brief Three Cartesian components, in metres or SI units specified by the field.
/// @par Example
/// @code
/// const dm::sim::Vec3 position{1,2,3};
/// std::print("{}\n",position[1]); // 2
/// @endcode
using Vec3 = std::array<double, 3>;
/// @brief Unit quaternion in w,x,y,z order, right-handed with Y up.
/// @par Example
/// @code
/// const dm::sim::Quaternion identity{1,0,0,0};
/// std::print("{}\n",identity[0]); // 1
/// @endcode
using Quaternion = std::array<double, 4>;
/// @brief Body identifier scoped to one world's lifetime; zero is reserved for the world.
/// @par Example
/// @code
/// const dm::sim::BodyId world=0;
/// std::print("{}\n",world); // 0
/// @endcode
using BodyId = std::uint64_t;
/// @brief Body motion classification.
/// @par Example
/// @code
/// const auto motion=dm::sim::Motion::Dynamic;
/// @endcode
enum class Motion {
  Static,   ///< Immovable, zero velocities.
  Dynamic,  ///< Integrated from forces, impulses and contacts.
  Kinematic ///< Prescribed velocities, infinite response mass.
};
/// @brief Convex collision primitive; all dimensions are full extents.
/// @par Example
/// @code
/// const auto shape=dm::sim::Shape::Sphere;
/// @endcode
enum class Shape {
  Box,       ///< Oriented box, one extent per local axis.
  Sphere,    ///< Uniform full diameter.
  Capsule,   ///< Local-Y capsule: equal X/Z diameter and Y tip-to-tip length.
  Ellipsoid, ///< Three independent full axis diameters.
  ConvexHull ///< Convex hull of an immutable vertex cloud.
};
/// @brief Immutable local convex geometry and its transform within a rigid body.
/// Dimensions are full extents in metres. ConvexHull uses vertices instead.
/// Shared vertex storage must remain immutable for the lifetime of bodies/snapshots.
/// @par Example
/// @code
/// dm::sim::Collider pad; pad.dimensions={.05,.01,.02};
/// @endcode
struct Collider {
  Shape      shape{Shape::Box};    ///< Primitive or convex hull.
  Vec3       dimensions{1, 1, 1};  ///< Positive full extents; capsule Y is tip-to-tip length.
  Vec3       position{};           ///< Offset from the body reference origin, metres.
  Quaternion rotation{1, 0, 0, 0}; ///< Local WXYZ orientation.
  std::shared_ptr<const std::vector<Vec3>>
      vertices; ///< Hull points; at least four noncoplanar points.
};
/// @brief Owned rigid-body creation and edit parameters; no dependency on a renderer.
/// Coordinates use SI units. Validation matches the implemented VSim component
/// ranges. Quaternion input must already be unit length within 1e-6.
/// @par Example
/// @code
/// dm::sim::Body body; body.motion=dm::sim::Motion::Dynamic; body.validate();
/// @endcode
struct Body {
  Motion     motion{Motion::Static}; ///< Static by default.
  Shape      shape{Shape::Box};      ///< Oriented box by default.
  Vec3       position{0, 0, 0};      ///< Reference origin in metres, each axis within +/-100000.
  Quaternion rotation{1, 0, 0, 0};   ///< Unit w,x,y,z rotation.
  Vec3       dimensions{1, 1, 1}; ///< Full extents in [0.02,1000]; sphere uniform, capsule Y > X=Z.
  Vec3       linear_velocity{0, 0, 0}; ///< COM world velocity, each component within +/-500 m/s.
  Vec3       angular_velocity{
      0,
      0,
      0}; ///< World angular velocity, each component within +/-angular_speed_limit rad/s.
  double              mass{1};              ///< Dynamic mass in [0.001,1000000] kg.
  double              friction{.6};         ///< Coulomb coefficient in [0,1].
  double              restitution{.1};      ///< Restitution in [0,1].
  double              gravity_scale{1};     ///< Gravity multiplier in [0,4].
  double              linear_damping{.05};  ///< Nonnegative inverse-second linear damping.
  double              angular_damping{.05}; ///< Nonnegative inverse-second angular damping.
  bool                sensor{false};        ///< Detect overlap without solid response.
  std::array<bool, 3> allowed_translation{
      true,
      true,
      true};                                              ///< World translation degrees of freedom.
  std::array<bool, 3> allowed_rotation{true, true, true}; ///< World rotation degrees of freedom.

  /// @brief World-space material velocity in m/s, each component within +/-500.
  /// Adds tangential contact slip without translating the collision geometry.
  /// The normal component is ignored. Useful for stationary conveyor bodies;
  /// leave zero for ordinary rigid bodies. Updating it wakes contacting bodies.
  Vec3 surface_velocity{0, 0, 0};

  Vec3 center_of_mass{}; ///< Body-local COM offset; position denotes the body reference origin.
  std::array<double, 9> inertia_tensor{}; ///< Symmetric positive-definite body-frame COM inertia,
                                          ///< kg m²; all zero uses the legacy primitive formula.
  std::shared_ptr<const std::vector<Vec3>>
      vertices; ///< ConvexHull vertex cloud in body coordinates.
  std::shared_ptr<const std::vector<Collider>>
         colliders; ///< Optional immutable compound; explicit COM inertia is recommended.
  double spinning_friction{0}; ///< Torsional Coulomb radius, metres.
  double rolling_friction{0};  ///< Rolling Coulomb radius, metres.
  double angular_speed_limit{
      100}; ///< Finite validation/optional clamp limit, rad/s, in [1,100000].
  std::uint32_t collision_group{1}; ///< Collision category bits; zero disables collisions.
  std::uint32_t collision_mask{
      0xffffffffu}; ///< Categories accepted by this body (mutual matching).

  /// @brief Check finite values, physical ranges and primitive/motion restrictions.
  /// @throws std::invalid_argument Invalid values, static velocity or zero dynamic degrees of
  /// freedom.
  /// @par Example
  /// @code
  /// dm::sim::Body body; body.validate();
  /// @endcode
  void validate() const;
};
/// @brief Joint type with one unconstrained generalized coordinate.
/// @par Example
/// @code
/// const auto kind=dm::sim::JointType::Hinge;
/// @endcode
enum class JointType {
  Hinge,  ///< Right-handed relative rotation in radians.
  Slider, ///< Relative displacement in metres.
  Fixed   ///< All six relative degrees constrained.
};
/// @brief A joint from a dynamic child to a parent body or the fixed world.
/// Anchor/axis are in the child's rotated local frame; coordinate zero is its
/// authored pose. Connected bodies do not collide. Each child has at most one joint.
/// @par Example
/// @code
/// dm::sim::Joint joint; joint.child=1; joint.validate();
/// @endcode
struct Joint {
  JointType type{JointType::Hinge}; ///< Hinge by default.
  BodyId    child{0};               ///< Dynamic child; must be nonzero.
  BodyId    parent{0};              ///< Parent body, or zero for the world.
  Vec3      anchor{0, 0, 0};        ///< Child-local offset in metres, ignoring shape dimensions.
  Vec3      axis{0, 0, 1};  ///< Nonzero child-local generalized axis, normalized when attached.
  bool      limited{false}; ///< Whether hard coordinate limits apply.
  double    limit_min{0};   ///< Lower limit, at most zero; hinges within -pi, sliders within -1000.
  double    limit_max{0};   ///< Upper limit, at least zero; hinges within pi, sliders within 1000.
  double    damping{0};     ///< Nonnegative generalized viscous damping, at most 1000000.

  double effort_limit{
      100000}; ///< Maximum motor force/torque; external impulses remain independent.
  double velocity_limit{
      100000};        ///< Maximum requested motor rate; contact backdrive is not clamped.
  double armature{0}; ///< Reflected generalized inertia, kg m² (hinge) or kg (slider).
  double friction{0}; ///< Passive dry joint friction, N m or N.

  /// @brief Check finite fields, identifier relations and coordinate limits.
  /// @throws std::invalid_argument Invalid type, axes, bounds or identifiers.
  /// @par Example
  /// @code
  /// dm::sim::Joint joint; joint.child=1; joint.validate();
  /// @endcode
  void validate() const;
};
/// @brief Joint coordinate/rate pair in radians/radians per second or metres/metres per second.
/// @par Example
/// @code
/// const dm::sim::JointState state;
/// std::print("{}\n",state.position); // 0
/// @endcode
struct JointState {
  double position{0}; ///< Coordinate relative to the authored pose.
  double velocity{0}; ///< Relative generalized rate.
};
} // namespace datamunge::sim
