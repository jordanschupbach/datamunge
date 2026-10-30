#pragma once
/// @file world.hpp
/// @brief Native rigid-body worlds with bounded capacities and topology-checked snapshots.
#include <cstddef>
#include <datamunge/compute/context.hpp>
#include <datamunge/sim/types.hpp>
#include <memory>
#include <vector>
namespace datamunge::sim {
/// @brief Explicit fixed-capacity physics controls; capacities reserve reusable working memory.
/// @par Example
/// @code
/// dm::sim::WorldOptions options; options.gravity={0,0,0};
/// @endcode
struct WorldOptions {
  Vec3        gravity{0, -9.81, 0};    ///< World gravity in m/s squared.
  std::size_t body_capacity{10000};    ///< Maximum simultaneously live bodies.
  std::size_t pair_capacity{65536};    ///< Maximum broadphase pairs; overflow is an error.
  std::size_t contact_capacity{20000}; ///< Maximum contact points; overflow is an error.
  std::size_t solver_iterations{20};   ///< Positive velocity constraint sweeps per substep.
  double      contact_slop{.001}; ///< Allowed penetration in metres before positional correction.
  double      position_correction{
      .2};            ///< Fraction of excess penetration corrected per substep, in (0,1].
  std::size_t substeps{4}; ///< Fixed positive integration/constraint substeps per requested step.
  bool        continuous{true}; ///< Enable swept convex time-of-impact subdivision.

  /// @brief Check finite gravity, positive capacities/iterations and correction ranges.
  /// @throws std::invalid_argument Invalid settings.
  /// @par Example
  /// @code
  /// const dm::sim::WorldOptions options; options.validate();
  /// @endcode
  void validate() const;
};
/// @brief One solved contact point between two body identifiers.
/// Normal points from first to second; separation is signed (negative penetration).
/// Sensor contacts are observations only and carry no response impulse.
/// @par Example
/// @code
/// const dm::sim::Contact contact;
/// std::print("{}\n",contact.sensor); // false
/// @endcode
struct Contact {
  BodyId first{0};        ///< First body.
  BodyId second{0};       ///< Second body.
  Vec3   point{0, 0, 0};  ///< World-space contact point in metres.
  Vec3   normal{1, 0, 0}; ///< Unit normal pointing first to second.
  double separation{0};   ///< Signed distance along normal, metres.
  double impulse{0};      ///< Accumulated normal impulse, N s.
  bool   sensor{false};   ///< Whether either body is a nonresponding sensor.
};
/// @brief Private solver state retained by immutable snapshots.
namespace detail {
struct WorldState;
}
/// @brief Owned solver snapshot including contacts, velocities, sleeping state and joint warm
/// starts. It can be restored only into the same world with unchanged topology/configuration.
/// Default construction is empty. Copies share immutable snapshot storage.
/// @par Example
/// @code
/// const dm::sim::PhysicsState state;
/// std::print("{}\n",state.empty()); // true
/// @endcode
class PhysicsState {
 public:
  /// @brief Construct an empty state.
  /// @par Example
  /// @code
  /// const dm::sim::PhysicsState state;
  /// @endcode
  PhysicsState() = default;
  /// @brief Test whether a world snapshot has been captured.
  /// @return True when no snapshot is held.
  bool empty() const noexcept { return !state_; }

 private:
  /// @brief World alone captures and restores owned solver storage.
  friend class World;
  std::shared_ptr<const detail::WorldState> state_; ///< Immutable captured state.
};
/// @brief A library-owned, noncopyable native physics world; no window or external physics runtime.
/// World methods are single-writer; use independent worlds for parallel rollouts.
/// Bodies are world-scoped positive IDs, never reused. External input is validated
/// before mutation. A failed solve restores the pre-step state. Snapshots retain
/// solver state and must not be confused with authored scene serialization.
/// @par Example
/// @code
/// dm::sim::World world;
/// dm::sim::Body body; body.motion=dm::sim::Motion::Dynamic; body.position={0,3,0};
/// const auto id=world.add_body(body); world.step(1./60);
/// std::print("{}\n",world.body(id).position[1]<3); // true
/// @endcode
/// @par References
/// - Erin Catto (2006), Fast and Simple Physics using Sequential Impulses,
///   https://box2d.org/files/ErinCatto_SequentialImpulses_GDC2006.pdf.
///   This implementation uses native 3D primitive constraints and conservative sweeps.
class World {
 public:
  /// @brief Own an empty world and reserve its configured working capacities.
  /// @param options Validated native solver controls; defaults above.
  /// @param context Explicit execution context, CPU by default.
  /// @throws std::invalid_argument Invalid controls.
  /// @throws std::runtime_error Requested simulation backend unavailable.
  /// @par Example
  /// @code
  /// dm::sim::World world;
  /// std::print("{}\n",world.body_count()); // 0
  /// @endcode
  explicit World(WorldOptions options = {}, compute::Context context = compute::Context{});
  /// @brief Destroy all native world state and owned resources.
  ~World();
  /// @brief Disable implicit copying of exclusive mutable runtime state.
  World(const World&)            = delete;
  /// @brief Disable copy assignment of exclusive mutable runtime state.
  World& operator=(const World&) = delete;
  /// @brief Add a validated rigid body and allocate a fresh identifier.
  /// @param body Owned initial component settings, copied.
  /// @return Fresh positive body ID.
  /// @throws std::invalid_argument Invalid body.
  /// @throws std::length_error Body capacity exhausted.
  /// @par Example
  /// @code
  /// dm::sim::World world; const auto id=world.add_body(dm::sim::Body{});
  /// std::print("{}\n",id); // 1
  /// @endcode
  BodyId add_body(const Body& body);
  /// @brief Remove an existing body, retaining all other IDs.
  /// @param id Live world-scoped body ID.
  /// @throws std::out_of_range Unknown ID.
  /// @throws std::invalid_argument Body remains referenced by a joint.
  /// @par Example
  /// @code
  /// dm::sim::World world; const auto id=world.add_body({}); world.remove_body(id);
  /// std::print("{}\n",world.body_count()); // 0
  /// @endcode
  void remove_body(BodyId id);
  /// @brief Return a copy of current component settings, simulated pose and velocities.
  /// @param id Live world-scoped body ID.
  /// @return Owned current body state.
  /// @throws std::out_of_range Unknown ID.
  /// @par Example
  /// @code
  /// dm::sim::World world; const auto id=world.add_body({});
  /// std::print("{}\n",world.body(id).mass); // 1
  /// @endcode
  Body body(BodyId id) const;
  /// @brief Replace validated body settings; pose/velocity-only edits preserve snapshot topology.
  /// @param id Live body ID.
  /// @param body Complete valid replacement.
  /// @throws std::out_of_range Unknown ID.
  /// @throws std::invalid_argument Invalid body or non-dynamic joint child.
  /// @par Example
  /// @code
  /// dm::sim::World world; const auto id=world.add_body({});
  /// auto b=world.body(id); b.position={0,2,0}; world.set_body(id,b);
  /// @endcode
  void set_body(BodyId id, const Body& body);
  /// @brief List live body identifiers in ascending order.
  /// @return Owned ID vector, empty for an empty world.
  /// @par Example
  /// @code
  /// dm::sim::World world;
  /// std::print("{}\n",world.bodies().size()); // 0
  /// @endcode
  std::vector<BodyId> bodies() const;
  /// @brief Return live body count.
  /// @return Number of live bodies.
  std::size_t body_count() const noexcept;
  /// @brief Apply a finite world-space impulse immediately to a dynamic body.
  /// @param id Live dynamic body.
  /// @param impulse Impulse in N s, each component within +/-100000.
  /// @throws std::invalid_argument Non-dynamic body, invalid impulse or excessive resulting
  /// velocity.
  /// @throws std::out_of_range Unknown ID.
  /// @par Example
  /// @code
  /// dm::sim::World world;dm::sim::Body b;b.motion=dm::sim::Motion::Dynamic;
  /// const auto id=world.add_body(b);world.apply_impulse(id,{1,0,0});
  /// std::print("{}\n",world.body(id).linear_velocity[0]); // 1
  /// @endcode
  void apply_impulse(BodyId id, const Vec3& impulse);
  /// @brief Accumulate a world-space force until the next successful step.
  /// @param id Live dynamic body.
  /// @param force Finite force in N, each component within +/-100000.
  /// @throws std::invalid_argument Invalid force or non-dynamic body.
  /// @throws std::out_of_range Unknown ID.
  /// @par Example
  /// @code
  /// dm::sim::World world;dm::sim::Body b;b.motion=dm::sim::Motion::Dynamic;
  /// const auto id=world.add_body(b);world.apply_force(id,{1,0,0});
  /// @endcode
  void apply_force(BodyId id, const Vec3& force);
  /// @brief Add a joint at its authored zero coordinate, rejecting parent cycles.
  /// @param joint Validated child/parent constraint.
  /// @throws std::invalid_argument Invalid joint, duplicate child, cycle or non-dynamic child.
  /// @throws std::out_of_range Missing body.
  /// @par Example
  /// @code
  /// dm::sim::World world;dm::sim::Body b;b.motion=dm::sim::Motion::Dynamic;
  /// dm::sim::Joint j;j.child=world.add_body(b);world.add_joint(j);
  /// @endcode
  void add_joint(const Joint& joint);
  /// @brief Remove the joint attached to a child.
  /// @param child Live body with a joint.
  /// @throws std::out_of_range No joint for child.
  /// @par Example
  /// @code
  /// dm::sim::World world;dm::sim::Body b;b.motion=dm::sim::Motion::Dynamic;
  /// dm::sim::Joint j;j.child=world.add_body(b);world.add_joint(j);world.remove_joint(j.child);
  /// @endcode
  void remove_joint(BodyId child);
  /// @brief Return the number of attached joints.
  /// @return Joint count.
  std::size_t joint_count() const noexcept;
  /// @brief Read a joint generalized coordinate and rate.
  /// @param child Live joint child.
  /// @return Angle/rate for a hinge or displacement/rate for a slider.
  /// @throws std::out_of_range No joint for child.
  /// @par Example
  /// @code
  /// dm::sim::World world;dm::sim::Body b;b.motion=dm::sim::Motion::Dynamic;
  /// dm::sim::Joint j;j.child=world.add_body(b);world.add_joint(j);
  /// std::print("{:.0f}\n",world.joint_state(j.child).position); // 0
  /// @endcode
  JointState joint_state(BodyId child) const;
  /// @brief Accumulate generalized torque or force for the next successful step.
  /// @param child Live joint child.
  /// @param force Finite N m (hinge) or N (slider), within +/-100000.
  /// @throws std::invalid_argument Invalid force.
  /// @throws std::out_of_range No joint for child.
  /// @par Example
  /// @code
  /// dm::sim::World world;dm::sim::Body b;b.motion=dm::sim::Motion::Dynamic;
  /// dm::sim::Joint
  /// j;j.child=world.add_body(b);world.add_joint(j);world.apply_joint_force(j.child,1);
  /// @endcode
  void apply_joint_force(BodyId child, double force);
  /// @brief Teleport a jointed subtree while preserving descendant coordinates.
  /// @param child Live joint child.
  /// @param position Finite angle/displacement within configured limits.
  /// @param velocity Finite generalized rate.
  /// @throws std::invalid_argument Invalid target or resulting body state.
  /// @throws std::out_of_range No joint for child.
  /// @par Example
  /// @code
  /// dm::sim::World world;dm::sim::Body b;b.motion=dm::sim::Motion::Dynamic;
  /// dm::sim::Joint
  /// j;j.child=world.add_body(b);world.add_joint(j);world.set_joint_state(j.child,.1,0);
  /// @endcode
  void set_joint_state(BodyId child, double position, double velocity);
  /// @brief Advance native physics by a finite positive delta of at most 1/30 second.
  /// Dynamic linear/angular speed magnitudes are capped at 500 m/s and 100 rad/s
  /// after force integration and every constraint solve, before swept motion.
  /// @param delta Simulation seconds; forces persist across configured substeps then clear.
  /// @throws std::invalid_argument Invalid delta.
  /// @throws std::runtime_error Solver or capacity failure; pre-step physics state is restored.
  /// @par Example
  /// @code
  /// dm::sim::World world; world.step(1./60);
  /// @endcode
  void step(double delta);
  /// @brief Test current shape proximity, including sensors and sleeping bodies.
  /// @param first Live body.
  /// @param second Different live body.
  /// @param distance Nonnegative proximity tolerance in metres; default .02.
  /// @return True within the requested surface distance.
  /// @throws std::invalid_argument Invalid distance or identical bodies.
  /// @throws std::out_of_range Unknown body.
  /// @par Example
  /// @code
  /// dm::sim::World world;auto a=world.add_body({});auto b=world.add_body({});
  /// std::print("{}\n",world.touching(a,b)); // true
  /// @endcode
  bool touching(BodyId first, BodyId second, double distance = .02) const;
  /// @brief Probe downward from the body centre for a solid support surface.
  /// @param id Live body.
  /// @param distance Nonnegative distance below body bounds; default .08 m.
  /// @return True for a solid surface with upward normal Y > .6.
  /// @throws std::invalid_argument Invalid distance.
  /// @throws std::out_of_range Unknown body.
  /// @par Example
  /// @code
  /// dm::sim::World world;auto id=world.add_body({});
  /// std::print("{}\n",world.grounded(id)); // false
  /// @endcode
  bool grounded(BodyId id, double distance = .08) const;
  /// @brief Borrow the final substep contact manifold.
  /// @return Contacts valid until the next step, body edit or restore.
  const std::vector<Contact>& contacts() const noexcept;
  /// @brief Capture physics and solver caches without rebuilding bodies.
  /// @return Owned immutable snapshot tied to this world/topology.
  /// @par Example
  /// @code
  /// dm::sim::World world; const auto state=world.save_state();
  /// std::print("{}\n",state.empty()); // false
  /// @endcode
  PhysicsState save_state() const;
  /// @brief Restore a compatible snapshot atomically.
  /// @param state Nonempty snapshot captured by this world with unchanged topology.
  /// @throws std::invalid_argument Empty, foreign or topology-incompatible state.
  /// @par Example
  /// @code
  /// dm::sim::World world; const auto state=world.save_state();
  /// world.step(1./60);world.restore_state(state);
  /// @endcode
  void restore_state(const PhysicsState& state);

  /// @brief Deep-copy live solver state into an independent world, retaining the compute context.
  /// Captures forces, contacts and joint frames. IDs remain numerically equal but
  /// are scoped to the new world; snapshots from either world do not cross-restore.
  /// @return Exclusive new world with independent mutable state.
  /// @throws std::bad_alloc Copy allocation failure; the source is unchanged.
  /// @par Example
  /// @code
  /// dm::sim::World world; const auto id=world.add_body({});
  /// const auto copy=world.fork();
  /// std::print("{}\n",copy->body_count()); // 1
  /// @endcode
  std::unique_ptr<World> fork() const;

 private:
  /// @brief Batch coordinator validates resets and rewinds successful parallel steps.
  friend class BatchWorld;
  /// @brief Check snapshot compatibility without mutating state.
  /// @param state Snapshot with this world's owner and topology.
  /// @throws std::invalid_argument Incompatible or empty state.
  void validate_state(const PhysicsState& state) const;
  /// @brief Restore the most recent successful step's reserved backup for batch rollback.
  void rewind_step() noexcept;
  struct Impl;                 ///< Native solver storage and validated mutation implementation.
  std::unique_ptr<Impl> impl_; ///< Exclusive world owner.
};
} // namespace datamunge::sim
