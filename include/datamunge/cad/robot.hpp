#pragma once
/// @file robot.hpp
/// @brief URDF visual asset loading and joint-driven robot forward kinematics.
#include <datamunge/cad/math.hpp>
#include <datamunge/cad/scene.hpp>
#include <map>
#include <memory>

namespace datamunge::cad {
/// @brief URDF joint motion type supported by the visual robot loader.
enum class JointType {
  Fixed,      ///< No relative motion.
  Revolute,   ///< Bounded rotation in radians.
  Continuous, ///< Unbounded rotation in radians.
  Prismatic   ///< Bounded translation in metres.
};
/// @brief An owned summary of a URDF joint and its limits.
struct RobotJoint {
  std::string name;                     ///< Unique joint name.
  std::string parent;                   ///< Parent link name.
  std::string child;                    ///< Child link name.
  JointType   type  = JointType::Fixed; ///< Motion type.
  double      lower = 0;      ///< Revolute radians/prismatic metres; continuous has no limits.
  double      upper = 0;      ///< Upper motion limit.
  std::string mimic;          ///< Source joint, or empty for independent motion.
  double      multiplier = 1; ///< Mimic scale.
  double      offset     = 0; ///< Mimic offset in this joint's units.
};
/// @brief Load real robot visual meshes and compute poses from URDF joint values.
/// Supports a single rooted link tree, fixed/revolute/continuous/prismatic
/// joints, mimic relations, visual origins, mesh scale, materials and relative
/// or package:// asset paths. Coordinates are URDF metres/radians; CAD STEP
/// files keep their own file units. Inertial/collision/dynamics/transmissions
/// are not evaluated by this visual loader. Xacro must be expanded first.
/// Geometry is owned; copies share immutable source meshes and have independent
/// joint positions. Concurrent const scene generation is safe.
/// @par Example
/// @code
/// auto robot = dm::cad::Robot::load_urdf("ReBot_Arm_DM.urdf");
/// robot.set_joint("joint2", -.6);
/// auto scene = robot.scene();
/// dm::cad::render(scene, dm::cad::fit_camera(scene)).save("arm.png");
/// @endcode
/// @par References
/// - Seeed DevArm URDF used for the real-model integration example:
///   https://github.com/Seeed-Projects/reBot-DevArm/blob/922c5fa3d8c8919db1847b49166f02fc26c8fdad/Rebot_Arm_description/DM/urdf/ReBot_Arm_DM.urdf
class Robot {
 public:
  /// @brief Load a URDF tree and every referenced visual asset synchronously.
  /// @param path Existing plain URDF file.
  /// @param package_root Optional directory containing ROS package directories;
  /// empty searches ancestors of the URDF for a matching package name.
  /// @return Owned model with zero independent coordinates clamped into their limits.
  /// @throws std::invalid_argument Invalid/duplicate/cyclic graph, unsupported geometry/joint,
  /// malformed numeric values, limits or unresolved package URI.
  /// @throws std::runtime_error XML or asset I/O/import failure.
  /// @par Example
  /// See the class example.
  static Robot load_urdf(const std::string& path, const std::string& package_root = "");
  /// @brief Set one independent movable joint coordinate transactionally.
  /// @param name Existing independent joint name; fixed/mimic joints cannot be set.
  /// @param value Finite radians for rotation or metres for translation, within limits.
  /// @throws std::invalid_argument Unknown joint, nonfinite/out-of-range value,
  /// or invalid resulting mimic coordinate.
  /// @par Example
  /// See the class example.
  void set_joint(const std::string& name, double value);
  /// @brief Set a finite measured joint coordinate without enforcing authored motion limits.
  /// For rendering physical encoder states, including small compliant-limit excursions.
  /// Does not simulate or clamp motion. Mimic relations still apply; this copy's FK
  /// subsequently accepts measured coordinates. set_joint() remains strictly limited.
  /// @param name Existing independent movable joint.
  /// @param value Finite radians/metres, absolute magnitude at most 1000.
  /// @throws std::invalid_argument Invalid name, fixed/mimic joint or invalid value.
  /// @par Example
  /// @code
  /// auto robot=dm::cad::Robot::load_urdf("ReBot_Arm_DM.urdf");
  /// robot.set_measured_joint("joint2", .0001); // measured compliant-stop excursion
  /// auto scene=robot.scene();
  /// @endcode
  /// @see set_joint(), link_transform()
  void set_measured_joint(const std::string& name,double value);
  /// @brief Set a group of measured encoder coordinates in one transaction.
  /// Uses the same finite measured-state semantics as set_measured_joint(), with
  /// one map copy and validation pass. Empty input leaves this model unchanged.
  /// @param values Unique independent joint names and finite coordinates, absolute value <=1000.
  /// @throws std::invalid_argument Duplicate/unknown names, fixed/mimic joints or invalid values.
  /// @par Example
  /// @code
  /// auto robot=dm::cad::Robot::load_urdf("ReBot_Arm_DM.urdf");
  /// robot.set_measured_joints({{"joint1",.1},{"joint2",-.5}});
  /// auto pose=robot.link_transform("end_link");
  /// @endcode
  /// @see set_measured_joint()
  void set_measured_joints(const std::vector<std::pair<std::string,double>>& values);
  /// @brief Read the effective coordinate, including mimic multiplier/offset.
  /// @param name Existing joint name; fixed joints return zero.
  /// @return Radians or metres according to the joint type.
  /// @throws std::invalid_argument Unknown joint.
  /// @par Example
  /// See the class example and set_joint().
  double joint_position(const std::string& name) const;
  /// @brief Copy all joint names/types/limits in URDF order.
  /// @return Owned joint summaries, including fixed and mimic joints.
  std::vector<RobotJoint> joints() const;
  /// @brief Copy names of independent movable joints in URDF order.
  /// @return Owned names, suitable for set_joint().
  std::vector<std::string> joint_names() const;
  /// @brief Compute a link's placement by forward kinematics.
  /// @param name Existing link name.
  /// @return Local-link to world transform at the current coordinates.
  /// @throws std::invalid_argument Unknown link.
  /// @par Example
  /// @code
  /// auto robot = dm::cad::Robot::load_urdf("robot.urdf");
  /// auto tip = robot.link_transform("tool").TransformPoint({0,0,0});
  /// @endcode
  Mat4d link_transform(const std::string& name) const;
  /// @brief Build an independent renderable scene at the current joint coordinates.
  /// Mesh placements are baked in world coordinates; material and UV values are
  /// preserved. Subsequent set_joint() calls do not change an earlier scene.
  /// @return Owned scene named by link/visual with URDF source_path.
  /// @throws std::runtime_error Internal asset transformation failure.
  /// @par Example
  /// See the class example.
  Scene scene() const;

 private:
  struct Impl; ///< Immutable parsed URDF tree, joint origins and local visual scenes.
  std::shared_ptr<const Impl>   impl_;      ///< Shared immutable geometry/model ownership.
  bool measured_{false}; ///< Whether FK accepts finite measured limit excursions.
  std::map<std::string, double> positions_; ///< This copy's independent coordinates.

  /// @brief Adopt a parsed immutable model with independent zero/clamped coordinates.
  /// @param impl Shared nonnull model constructed by load_urdf().
  explicit Robot(std::shared_ptr<const Impl> impl);
};
} // namespace datamunge::cad
