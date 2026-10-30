#pragma once
/// @file types.hpp
/// @brief Native simulation coordinates and validated rigid-body descriptions.
#include <array>
#include <cstdint>
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
  Static,    ///< Immovable, zero velocities.
  Dynamic,   ///< Integrated from forces, impulses and contacts.
  Kinematic  ///< Prescribed velocities, infinite response mass.
};
/// @brief Convex collision primitive; all dimensions are full extents.
/// @par Example
/// @code
/// const auto shape=dm::sim::Shape::Sphere;
/// @endcode
enum class Shape {
  Box,     ///< Oriented box, one extent per local axis.
  Sphere,  ///< Uniform full diameter.
  Capsule  ///< Local-Y capsule: equal X/Z diameter and Y tip-to-tip length.
};
/// @brief Owned rigid-body creation and edit parameters; no dependency on a renderer.
/// Coordinates use SI units. Validation matches the implemented VSim component
/// ranges. Quaternion input must already be unit length within 1e-6.
/// @par Example
/// @code
/// dm::sim::Body body; body.motion=dm::sim::Motion::Dynamic; body.validate();
/// @endcode
struct Body {
  Motion motion{Motion::Static};  ///< Static by default.
  Shape shape{Shape::Box};        ///< Oriented box by default.
  Vec3 position{0, 0,
                0};  ///< Centre of mass in metres, each axis within +/-100000.
  Quaternion rotation{1, 0, 0, 0};  ///< Unit w,x,y,z rotation.
  Vec3 dimensions{
      1, 1,
      1};  ///< Full extents in [0.02,1000]; sphere uniform, capsule Y > X=Z.
  Vec3 linear_velocity{
      0, 0, 0};  ///< World velocity, each component within +/-500 m/s.
  Vec3 angular_velocity{
      0, 0,
      0};  ///< World angular velocity, each component within +/-100 rad/s.
  double mass{1};               ///< Dynamic mass in [0.001,1000000] kg.
  double friction{.6};          ///< Coulomb coefficient in [0,1].
  double restitution{.1};       ///< Restitution in [0,1].
  double gravity_scale{1};      ///< Gravity multiplier in [0,4].
  double linear_damping{.05};   ///< Nonnegative inverse-second linear damping.
  double angular_damping{.05};  ///< Nonnegative inverse-second angular damping.
  bool sensor{false};           ///< Detect overlap without solid response.
  std::array<bool, 3> allowed_translation{
      true, true, true};  ///< World translation degrees of freedom.
  std::array<bool, 3> allowed_rotation{
      true, true, true};  ///< World rotation degrees of freedom.

  /// @brief Check finite values, physical ranges and primitive/motion restrictions.
  /// @throws std::invalid_argument Invalid values, static velocity or zero dynamic degrees of freedom.
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
  Slider  ///< Relative displacement in metres.
};
/// @brief A joint from a dynamic child to a parent body or the fixed world.
/// Anchor/axis are in the child's rotated local frame; coordinate zero is its
/// authored pose. Connected bodies do not collide. Each child has at most one joint.
/// @par Example
/// @code
/// dm::sim::Joint joint; joint.child=1; joint.validate();
/// @endcode
struct Joint {
  JointType type{JointType::Hinge};  ///< Hinge by default.
  BodyId child{0};                   ///< Dynamic child; must be nonzero.
  BodyId parent{0};                  ///< Parent body, or zero for the world.
  Vec3 anchor{0, 0,
              0};  ///< Child-local offset in metres, ignoring shape dimensions.
  Vec3 axis{
      0, 0,
      1};  ///< Nonzero child-local generalized axis, normalized when attached.
  bool limited{false};  ///< Whether hard coordinate limits apply.
  double limit_min{
      0};  ///< Lower limit, at most zero; hinges within -pi, sliders within -1000.
  double limit_max{
      0};  ///< Upper limit, at least zero; hinges within pi, sliders within 1000.
  double damping{
      0};  ///< Nonnegative generalized viscous damping, at most 1000000.

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
  double position{0};  ///< Coordinate relative to the authored pose.
  double velocity{0};  ///< Relative generalized rate.
};
}  // namespace datamunge::sim
