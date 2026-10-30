#pragma once
/// @file simulation.hpp
/// @brief Native behavior lifecycle and canonical behavior-before-physics game updates.
#include <datamunge/game/scene.hpp>
#include <datamunge/sim/world.hpp>
#include <functional>
#include <map>
#include <memory>
namespace datamunge::game {
/// @brief Named normalized controls for one tick; absent controls are zero and never latch.
/// @par Example
/// @code
/// const dm::game::InputFrame input{{"move_x",1}};
/// @endcode
using InputFrame = std::map<std::string, double>;
class Simulation;
/// @brief Per-entity native behavior; instances retain state between ticks.
/// A factory supplies one exclusive instance per entity. Lifecycle order is
/// ascending entity ID, start once then update (including the first tick).
/// Native callbacks are trusted caller code and may propagate exceptions.
/// @par Example
/// @code
/// const dm::game::Behavior behavior;
/// @endcode
class Behavior {
 public:
  /// @brief Release behavior-owned state through the native polymorphic interface.
  virtual ~Behavior() = default;
  /// @brief Initialize this entity's behavior once, on its first visit.
  /// @param simulation Owning simulation, valid for this callback.
  /// @param self Live entity identifier.
  /// @param delta Current fixed tick seconds.
  /// @par Example
  /// @code
  /// dm::game::Simulation simulation(dm::game::Scene{});
  /// dm::game::Behavior behavior;behavior.on_start(simulation,simulation.create({}),1./60);
  /// @endcode
  virtual void on_start(Simulation& simulation, EntityId self, double delta) {}
  /// @brief Update behavior state before the tick's final physics step.
  /// @param simulation Owning simulation, valid for this callback.
  /// @param self Live entity identifier.
  /// @param delta Current fixed tick seconds.
  /// @par Example
  /// @code
  /// dm::game::Simulation simulation(dm::game::Scene{});
  /// dm::game::Behavior behavior;behavior.on_update(simulation,simulation.create({}),1./60);
  /// @endcode
  virtual void on_update(Simulation& simulation, EntityId self, double delta) {}
};
/// @brief Explicit name-to-native-factory mapping; no script interpreter or global registry.
/// @par Example
/// @code
/// dm::game::BehaviorRegistry registry;
/// @endcode
class BehaviorRegistry {
 public:
  /// @brief Factory returning one exclusive stateful behavior instance.
  /// @par Example
  /// @code
  /// dm::game::BehaviorRegistry::Factory factory=[]{return
  /// std::make_unique<dm::game::Behavior>();}; const auto instance=factory();
  /// @endcode
  using Factory = std::function<std::unique_ptr<Behavior>()>;
  /// @brief Register or replace a factory used by subsequently constructed instances.
  /// @param name Nonempty scalar UTF-8 behavior name, at most 128 bytes without NUL.
  /// @param factory Nonempty callable returning a nonnull behavior.
  /// @throws std::invalid_argument Invalid name or empty callable.
  /// @par Example
  /// @code
  /// dm::game::BehaviorRegistry registry;
  /// registry.add("idle",[]{return std::make_unique<dm::game::Behavior>();});
  /// @endcode
  void add(std::string name, Factory factory);
  /// @brief Test whether a behavior name is registered.
  /// @param name Possible factory name.
  /// @return Whether a factory exists.
  bool contains(const std::string& name) const noexcept;
  /// @brief Construct a fresh stateful instance using a registered factory.
  /// @param name Registered native behavior name.
  /// @return Exclusive instance; not shared with other entities.
  /// @throws std::out_of_range Unknown factory.
  /// @throws std::runtime_error Factory returned null.
  /// @throws std::exception Factory errors propagate.
  /// @par Example
  /// @code
  /// dm::game::BehaviorRegistry registry;
  /// registry.add("idle",[]{return std::make_unique<dm::game::Behavior>();});
  /// const auto behavior=registry.create("idle");
  /// @endcode
  std::unique_ptr<Behavior> create(const std::string& name) const;

 private:
  std::map<std::string, Factory> factories_; ///< Owned factory callbacks.
};
/// @brief Owned native scene, behavior instances and physics world with deterministic visit order.
/// Single-writer, library-only game update. New entities first run on the next
/// tick; destroyed entities skip later callbacks. Renaming a behavior replaces
/// its instance on the next visit. Errors propagate with entity context; earlier
/// successful callbacks are not rolled back. Invalid input fails before callbacks.
/// @par Example
/// @code
/// dm::game::Simulation simulation(dm::game::Scene{});
/// simulation.tick(1./60);
/// @endcode
class Simulation {
 public:
  /// @brief Own and compile a scene; each behavior requires an explicit registered factory.
  /// @param scene Owned starting scene, moved or copied.
  /// @param behaviors Owned native behavior factories, empty by default.
  /// @param physics Native solver controls.
  /// @throws std::invalid_argument Invalid/unregistered components or physics controls.
  /// @par Example
  /// @code
  /// dm::game::Simulation simulation(dm::game::Scene{});
  /// @endcode
  explicit Simulation(Scene scene, BehaviorRegistry behaviors = {}, sim::WorldOptions physics = {});
  /// @brief Destroy native behaviors before releasing world/scene state.
  ~Simulation();
  /// @brief Disable implicit copying of exclusive mutable runtime state.
  Simulation(const Simulation&) = delete;
  /// @brief Disable copy assignment of exclusive mutable runtime state.
  Simulation& operator=(const Simulation&) = delete;
  /// @brief Borrow the authoritative current scene read-only.
  /// @return Reference valid until simulation replacement/destruction.
  const Scene& scene() const noexcept;
  /// @brief Advance ordered native callbacks, then physics, then scene write-back.
  /// @param delta Finite positive seconds at most 1/30.
  /// @param input At most 32 nonempty named controls in [-1,1], empty by default.
  /// @throws std::invalid_argument Invalid delta or control frame, before mutation.
  /// @throws std::logic_error Recursive tick call.
  /// @throws std::exception Behavior/physics errors propagate; earlier callbacks may have changed
  /// state.
  /// @par Example
  /// @code
  /// dm::game::Simulation simulation(dm::game::Scene{});simulation.tick(1./60);
  /// @endcode
  void tick(double delta, const InputFrame& input = {});
  /// @brief Read a normalized control from the current tick.
  /// @param name Named game action.
  /// @return Copied finite value; zero when absent.
  /// @par Example
  /// @code
  /// dm::game::Simulation simulation(dm::game::Scene{});
  /// std::print("{}\n",simulation.input("move_x")); // 0
  /// @endcode
  double input(const std::string& name) const noexcept;
  /// @brief Create an entity after compiling the proposed scene transactionally.
  /// @param entity Owned valid native components.
  /// @return Fresh scene entity ID.
  /// @throws std::exception Invalid components/factory or compile failure; scene/world unchanged.
  /// @par Example
  /// @code
  /// dm::game::Simulation simulation(dm::game::Scene{});
  /// std::print("{}\n",simulation.create({})); // 1
  /// @endcode
  EntityId create(Entity entity = {});
  /// @brief Remove an entity and native physics after validating remaining references.
  /// @param id Live entity ID.
  /// @throws std::exception Invalid references or failed compile; scene/world unchanged.
  /// @par Example
  /// @code
  /// dm::game::Simulation simulation(dm::game::Scene{});auto
  /// id=simulation.create({});simulation.destroy(id);
  /// @endcode
  void destroy(EntityId id);
  /// @brief Replace one entity transactionally, preserving unchanged live solver state.
  /// @param id Live entity ID.
  /// @param entity Validated full component replacement.
  /// @throws std::exception Invalid entity, unknown factory or allocation/compile failure.
  /// @par Example
  /// @code
  /// dm::game::Simulation simulation(dm::game::Scene{});auto id=simulation.create({});
  /// auto e=simulation.scene().get(id);e.name="Player";simulation.set(id,e);
  /// @endcode
  void set(EntityId id, Entity entity);
  /// @brief Spawn an independent named prefab transactionally.
  /// @param name Existing scene prefab.
  /// @return Fresh entity ID.
  /// @throws std::exception Unknown prefab, invalid factory or failed compile.
  /// @par Example
  /// @code
  /// dm::game::Scene scene;scene.define_prefab("Box",{});
  /// dm::game::Simulation simulation(scene);const auto id=simulation.spawn("Box");
  /// @endcode
  EntityId spawn(const std::string& name);
  /// @brief Set validated spatial state without rebuilding native bodies or behavior instances.
  /// @param id Live entity ID.
  /// @param spatial Complete pose/settings with unchanged physical configuration.
  /// @throws std::invalid_argument Invalid state or physical settings changed; use set for
  /// configuration edits.
  /// @throws std::out_of_range Unknown entity.
  /// @par Example
  /// @code
  /// dm::game::Simulation simulation(dm::game::Scene{});auto id=simulation.create({});
  /// auto b=simulation.scene().get(id).spatial;b.position={1,2,3};simulation.set_motion(id,b);
  /// @endcode
  void set_motion(EntityId id, const sim::Body& spatial);
  /// @brief Apply a world-space impulse to an entity's dynamic body.
  /// @param id Live physical entity.
  /// @param impulse Finite N s, matching native world limits.
  /// @throws std::exception Unknown/nonphysical entity or invalid impulse.
  /// @par Example
  /// @code
  /// dm::game::Entity e;e.physics_enabled=true;e.spatial.motion=dm::sim::Motion::Dynamic;
  /// dm::game::Simulation simulation(dm::game::Scene{});auto
  /// id=simulation.create(e);simulation.apply_impulse(id,{0,1,0});
  /// @endcode
  void apply_impulse(EntityId id, const sim::Vec3& impulse);
  /// @brief Query native primitive proximity using scene entity IDs.
  /// @param first Live physical entity.
  /// @param second Different live physical entity.
  /// @return Whether the surfaces are within .02 m.
  /// @throws std::exception Unknown/nonphysical entities.
  /// @par Example
  /// @code
  /// dm::game::Scene scene;dm::game::Entity e;e.physics_enabled=true;
  /// auto a=scene.create(e),b=scene.create(e);dm::game::Simulation simulation(scene);
  /// std::print("{}\n",simulation.touching(a,b)); // true
  /// @endcode
  bool touching(EntityId first, EntityId second) const;
  /// @brief Query a downward native support probe for an entity.
  /// @param id Live physical entity.
  /// @return Whether a nonsensor support lies within .08 m below its bounds.
  /// @throws std::exception Unknown/nonphysical entity.
  /// @par Example
  /// @code
  /// dm::game::Scene scene;dm::game::Entity e;e.physics_enabled=true;
  /// auto id=scene.create(e);dm::game::Simulation simulation(scene);
  /// std::print("{}\n",simulation.grounded(id)); // false
  /// @endcode
  bool grounded(EntityId id) const;

 private:
  struct Impl;                 ///< Ordered native instances and world/entity mapping.
  std::unique_ptr<Impl> impl_; ///< Exclusive runtime state.
};
} // namespace datamunge::game
