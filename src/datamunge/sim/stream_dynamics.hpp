#pragma once
/// @file stream_dynamics.hpp
/// @brief Private explicit native/reference articulated workcell, sharing the public recorded-pose
/// protocol.
#include <datamunge/sim/conveyor.hpp>
#include <datamunge/sim/devarm_pick.hpp>
#include <datamunge/sim/physics_backend.hpp>
#include <memory>
#include <string>
namespace datamunge::sim::detail {
/// @brief Finite-effort robot and freely moving compound fruit; never attaches payloads.
/// Owns an independent native world or optional MuJoCo model/data. The supplied world provides
/// static fixtures and stable payload IDs only; native stepping is disabled by the caller.
class StreamDynamics {
 public:
  /// @brief Compile real URDF inertias/collision hulls and supplied static fixtures.
  /// @param urdf Full DevArm URDF; collision files must exist.
  /// @param world Static native workcell to mirror in SI coordinates.
  /// @param delta Internal timestep dividing 1/240 second.
  /// @param capacity Maximum simultaneously retained payloads, 1..32.
  /// @param backend Explicit engine; legacy private direct construction defaults to the comparison
  /// backend.
  /// @throws std::runtime_error Backend absent, invalid dynamics or missing asset.
  StreamDynamics(
      const std::string& urdf,
      const World&       world,
      double             delta,
      std::size_t        capacity,
      ArticulatedBackend backend = ArticulatedBackend::MuJoCo);
  /// @brief Free the independent model/data without changing any other world.
  ~StreamDynamics();
  /// @brief Reset joints/velocities, controls and unused payload slots.
  /// @param q Six URDF coordinates.
  /// @param finger Half-aperture, metres.
  void reset(const std::array<double, 6>& q, double finger);
  /// @brief Initialize a new free payload using its sampled appearance and SI state.
  /// @param id Live native body identifier.
  /// @param observation Sampled kind, geometry and pose.
  void spawn(BodyId id, const ConveyorObservation& observation);
  /// @brief Disable collisions and recycle a completed payload slot.
  /// @param id Retained native identifier.
  void remove(BodyId id);
  /// @brief Integrate exactly one 240 Hz tick, retaining all release velocities.
  /// @param target Six bounded desired URDF coordinates.
  /// @param finger Desired half-aperture, metres; finite-force physical fingers may stop short.
  void step(const std::array<double, 6>& target, double finger);
  /// @brief Read actual articulated positions for rendering and future control.
  /// @param arm Updated six joints and symmetric measured finger aperture.
  void observe(PickObservation& arm) const;
  /// @brief Read the grasp reference point directly from articulated forward kinematics.
  /// @return CAD Z-up world point, metres, at end_link local (-.02,0,0).
  cad::Vec3d tool_point() const;
  /// @brief Copy the measured free-body state in the public Y-up convention.
  /// @param id Retained identifier.
  /// @param base Native shape/material metadata to retain.
  /// @return Actual pose, linear/angular velocities with all rotational degrees free.
  Body body(BodyId id, Body base) const;
  /// @brief Detect simultaneous nontrivial force contacts with both real finger bodies.
  /// @param id Retained payload.
  /// @return True only if each finger supplies at least 0.05 N normal force.
  bool gripped(BodyId id) const;
  /// @brief Sum normal forces per finger and return the smaller opposing load.
  /// @param id Retained payload.
  /// @return Nonnegative normal force, N; zero unless both fingers contact.
  double grip_force(BodyId id) const;
  /// @brief Detect actual upward support specifically from a bin floor, not a nearby ray hit.
  /// @param id Retained payload.
  /// @return True for an upward load-bearing floor contact exceeding 0.05 N.
  bool bin_supported(BodyId id) const;
  /// @brief Read maximum absolute servo torque normalized by its authored effort limit.
  /// @return Ratio in [0,1] for the last tick.
  double effort_ratio() const;
  /// @brief Read maximum contact penetration and actuator fraction since reset.
  /// @return Penetration metres, bounded effort fraction, maximum rad/s, maximum finger m/s.
  std::array<double, 4> diagnostics() const;

 private:
  struct Impl;                 ///< Independent compiled articulated world.
  std::unique_ptr<Impl> impl_; ///< Exclusive owner.
};
} // namespace datamunge::sim::detail
