#pragma once
/// @file conveyor.hpp
/// @brief Resettable fruit-conveyor physics and native CAD scenes for sorting experiments.
#include <cstdint>
#include <datamunge/cad/scene.hpp>
#include <datamunge/sim/workcell_visuals.hpp>
#include <datamunge/sim/world.hpp>
#include <memory>

namespace datamunge::sim {
/// @brief Payload identity; default sorting maps apples to red and bananas to yellow.
enum class FruitKind {
  Apple, ///< Apple, with class-independent appearance variation in Diverse mode.
  Banana ///< Banana, with class-independent appearance variation in Diverse mode.
};
/// @brief Reproducible fruit-domain protocol; Basic preserves earlier checkpoints/examples.
/// @par Example
/// @code
/// auto domain=dm::sim::FruitVariation::Diverse;
/// @endcode
enum class FruitVariation {
  Basic,  ///< Original single shape/color and 8 mm pick-position jitter.
  Diverse ///< Six prototypes per class, shared skin palettes, varied size, pose and belt entry.
};
/// @brief Seeded rendering/shape parameters, independent of camera or policy predictions.
/// Prototype IDs 0..5 have separate apple/banana geometries. The skin palette is
/// shared by both classes to prevent a red-versus-yellow color shortcut.
/// @par Example
/// @code
/// dm::sim::ConveyorEnvironment env(dm::sim::FruitVariation::Diverse);
/// auto appearance=env.observation().appearance;
/// @endcode
struct FruitAppearance {
  std::size_t          prototype{0}; ///< Per-class geometry prototype, 0..5.
  double               scale{1};     ///< Uniform size multiplier; Diverse samples [.85,1.15].
  std::array<float, 3> skin{
      .8f,
      .3f,
      .1f};                      ///< RGB skin tint from a shared class-independent palette.
  double        mottling{0};     ///< Dark mottling strength, 0..0.35.
  double        roughness{.35};  ///< Surface roughness, .25..0.75.
  double        yaw{0};          ///< Sampled CAD Z-up yaw, radians; Diverse spans a full turn.
  double        pitch{0};        ///< Sampled tilt about CAD Y, radians.
  double        roll{0};         ///< Sampled tilt about CAD X, radians.
  std::uint64_t texture_seed{0}; ///< Independent deterministic texture realization.
  bool          leaf{true};      ///< Whether this apple has a visible leaf; ignored for bananas.
};
/// @brief Latched arrival and placement status of the single active payload.
enum class ConveyorStage {
  Travelling, ///< Belt carries the fruit toward the optical pick sensor.
  Waiting,    ///< Sensor has stopped the belt; an external controller may pick.
  Sorted,     ///< Fruit settled in the matching bin.
  Missorted,  ///< Fruit settled in the other bin.
  Lost        ///< Fruit left the work area or fell to the floor.
};
/// @brief Owned observation, using physics coordinates X along belt, Y up, Z across belt.
/// CAD scene methods convert to X along belt, Z up, Y across belt.
/// @par Example
/// @code
/// dm::sim::ConveyorEnvironment env;
/// auto state = env.observation();
/// std::print("{}\n", state.position[0]); // -0.9
/// @endcode
struct ConveyorObservation {
  FruitVariation  variation{FruitVariation::Basic}; ///< Selected domain protocol.
  FruitAppearance appearance;                       ///< Shape, palette and material realization.
  Quaternion      rotation{1, 0, 0, 0}; ///< Actual Y-up physics body orientation, w,x,y,z.
  Vec3            dimensions{}; ///< Legacy primitive sizes; bounds metadata for articulated compound fruit.
  FruitKind       kind{FruitKind::Apple};           ///< Seeded payload class.
  ConveyorStage   stage{ConveyorStage::Travelling}; ///< Current latched stage.
  BodyId          fruit{0};                         ///< Active dynamic body's world-scoped ID.
  Vec3            position{};                       ///< Rigid-body reference origin in metres; articulated COM can be offset.
  double mass{0}; ///< Actual rigid-body mass, kg.
  Vec3            angular_velocity{}; ///< Actual Y-up angular velocity, rad/s, for articulated frames.
  Vec3            velocity{};                       ///< Payload linear velocity in m/s.
  double          time{0};                          ///< Seconds since the most recent reset().
  double          belt_travel{0};                   ///< Integrated tread travel in metres.
  double          arrival_time{-1};                 ///< Sensor trigger time, or -1 before arrival.
  std::uint64_t   seed{0};                          ///< Reset seed, retained for replay.
  bool physical_origin{false}; ///< Render geometry about the measured rigid-body origin, without
                               ///< legacy bottom-alignment correction; true for articulated fruit.
};
/// @brief A headless, single-payload conveyor and two physical sorting bins.
/// Uses the native rigid-body solver at 240 Hz, SI units and Y-up coordinates.
/// A friction-driven belt stops when the centre crosses X=0.10 m; the next
/// fruit cannot spawn until the current one is sorted, missorted or lost.
/// Basic apples use a 78 mm sphere proxy; bananas use a 160 by 46 mm capsule proxy.
/// Diverse selects six meshes per class and scales/rotates the corresponding proxy;
/// entry X is [-1,-.7] m, across-belt Z is [-.065,.065] m, yaw is [0,2*pi),
/// and pitch/roll are [-.12,.12] rad. Appearance draws are shared across classes.
/// Payload rotation is locked as a stable initial manipulation approximation.
/// Bins are centred at (0.10, 0.44, +/-0.42) m; the negative-Z bin is for apples.
/// The workcell includes a pedestal at X=0.62 m, ready for a Z-up URDF visual
/// translated to CAD (0.62,0,0.72). Robot actuation, grasping and arm collision
/// geometry belong to the external controller; this class does not infer them
/// from visual CAD. world() exposes forces/body edits for later control tasks.
/// Instances are single-writer; reset(seed) repeats the same fruit sequence.
/// @par Example
/// @code
/// dm::sim::ConveyorEnvironment env;
/// env.reset(42);
/// for (int i=0; i<720; ++i) env.step(1./120);
/// auto state = env.observation();
/// std::print("{}\n", state.stage == dm::sim::ConveyorStage::Waiting); // true
/// @endcode
class ConveyorEnvironment {
 public:
  /// @brief Construct the workcell and start the sequence for seed zero.
  /// @throws std::bad_alloc Insufficient memory for the world or meshes.
  /// @par Example
  /// See the class example.
  ConveyorEnvironment();
  /// @brief Construct a workcell using a specified fruit-domain protocol.
  /// @param variation Basic or Diverse; retained across reset() and next_fruit().
  /// @throws std::invalid_argument Unknown protocol.
  /// @par Example
  /// See FruitAppearance.
  explicit ConveyorEnvironment(FruitVariation variation);
  /// @brief Select rendering assets without changing physics, seeds or task state.
  /// Retained across resets; returned scenes are independent owned copies.
  /// @param style Simple or Realistic.
  /// @throws std::invalid_argument Unknown profile, before mutation.
  /// @par Example
  /// @code
  /// dm::sim::ConveyorEnvironment env(dm::sim::FruitVariation::Diverse);
  /// env.set_visual_style(dm::sim::WorkcellVisualStyle::Realistic);
  /// auto scene=env.static_scene();
  /// @endcode
  void set_visual_style(WorkcellVisualStyle style);
  /// @brief Read the rendering-only workcell profile.
  /// @return Current style, Simple by default.
  WorkcellVisualStyle visual_style() const noexcept;
  /// @brief Destroy the owned world and meshes.
  ~ConveyorEnvironment();
  /// @brief Copying a live physics environment is disabled; use reset(seed) for replay.
  /// @param other Environment that cannot be copied.
  ConveyorEnvironment(const ConveyorEnvironment& other) = delete;
  /// @brief Assigning a live physics environment is disabled.
  /// @param other Environment that cannot be assigned.
  /// @return No value; deleted operation.
  ConveyorEnvironment& operator=(const ConveyorEnvironment& other) = delete;

  /// @brief Rebuild an empty workcell and spawn one seeded random fruit.
  /// @param seed Full-width reproducibility seed; the low PRNG bit selects fruit.
  /// @throws std::bad_alloc Insufficient world storage.
  /// @par Example
  /// See the class example.
  void reset(std::uint64_t seed);
  /// @brief Start a manipulation episode with one stationary fruit at the pick sensor.
  /// Randomizes lateral/longitudinal position by at most 8 mm using the seed.
  /// This initial-state placement occurs only during reset; subsequent movement
  /// uses the native solver. The belt stays stopped and arrival_time is zero.
  /// @param seed Reproducibility seed for fruit class and position.
  /// @throws std::bad_alloc Insufficient world storage.
  /// @par Example
  /// @code
  /// dm::sim::ConveyorEnvironment env; env.reset_for_pick(42);
  /// std::print("{}\n", env.observation().stage==dm::sim::ConveyorStage::Waiting); // true
  /// @endcode
  void reset_for_pick(std::uint64_t seed);
  /// @brief Integrate physics, stop the sensor-controlled belt, and detect placement.
  /// @param seconds Finite elapsed time in (0,1/30]; subdivided into <=1/240 s ticks.
  /// @throws std::invalid_argument Invalid elapsed time, before mutation.
  /// @throws std::runtime_error Physics capacity or numerical solve failure.
  /// @par Example
  /// See the class example.
  void step(double seconds);
  /// @brief Replace a completed payload with the next seeded fruit; retain both bins.
  /// @throws std::logic_error Current fruit is still travelling or waiting.
  /// @par Example
  /// @code
  /// dm::sim::ConveyorEnvironment env;
  /// auto s = env.observation();
  /// auto b = env.world().body(s.fruit);
  /// b.position={0.1,0.5,s.kind==dm::sim::FruitKind::Apple ? -0.42 : 0.42};
  /// env.world().set_body(s.fruit,b);
  /// for(int i=0;i<120;++i) env.step(1./120);
  /// env.next_fruit();
  /// @endcode
  void next_fruit();
  /// @brief Read the current dynamic payload and latched task state.
  /// @return Owned observation; body edits through world() are visible immediately.
  /// @throws std::out_of_range External code removed the active fruit body.
  ConveyorObservation observation() const;
  /// @brief Borrow the mutable physics world for an external controller.
  /// Keep the active fruit and workcell bodies; reset() invalidates this reference.
  /// @return Borrowed world with native forces, impulses, contacts and snapshots.
  World& world();
  /// @brief Borrow the read-only physics world; reset() invalidates this reference.
  /// @return Borrowed world, valid until reset() or destruction.
  const World& world() const;
  /// @brief Copy the static workcell in Z-up CAD metres, without a robot visual.
  /// @return Owned floor, frame, belt bed, sensor, pedestal and labelled-color bins.
  /// @par Example
  /// @code
  /// dm::sim::ConveyorEnvironment env;
  /// auto scene=env.static_scene();
  /// dm::cad::render(scene,dm::cad::fit_camera(scene)).save("workcell.png");
  /// @endcode
  cad::Scene static_scene() const;
  /// @brief Copy the active fruit, moving tread marks and sensor lamp in Z-up metres.
  /// @return Owned scene; combine with static_scene() using the same camera/depth.
  /// @par Example
  /// @code
  /// dm::sim::ConveyorEnvironment env;
  /// auto scene=env.dynamic_scene();
  /// dm::cad::render(scene,dm::cad::fit_camera(scene)).save("fruit.png");
  /// @endcode
  cad::Scene dynamic_scene() const;
  /// @brief Render a recorded fruit state, including its shape, material and actual orientation.
  /// @param state Valid recorded observation with prototype 0..5 and positive scale.
  /// @return Owned Z-up moving-actor scene, independent of the current body placement.
  /// @throws std::invalid_argument Invalid recorded appearance or protocol.
  /// @par Example
  /// @code
  /// dm::sim::ConveyorEnvironment env(dm::sim::FruitVariation::Diverse);
  /// auto scene=env.dynamic_scene(env.observation());
  /// @endcode
  cad::Scene dynamic_scene(const ConveyorObservation& state) const;

 private:
  struct Impl;                 ///< Physics storage, seeded generator and owned workcell scene.
  std::unique_ptr<Impl> impl_; ///< Exclusive state; no shared mutable simulation.
};
} // namespace datamunge::sim
