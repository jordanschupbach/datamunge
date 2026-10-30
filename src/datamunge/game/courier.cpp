/// @file courier.cpp
/// @brief Native equivalents of the frozen 2D/3D Courier, camera and lift callbacks.
#include <algorithm>
#include <cmath>
#include <datamunge/game/courier.hpp>
#include <stdexcept>
namespace datamunge::game {
/// @brief Reference-game rules and private per-entity native callback state.
namespace {
/// @brief Replace a live status entity's visible name without changing physical configuration.
/// @param simulation Current native runtime.
/// @param id Live status entity.
/// @param value Visible message without the Status prefix.
void message(Simulation& simulation, EntityId id, const std::string& value) {
  auto entity = simulation.scene().get(id);
  entity.name = "Status: " + value;
  simulation.set(id, std::move(entity));
}
/// @brief Test whether one remaining entity lies on an exact integer grid tile.
/// @param simulation Current native runtime.
/// @param id Possible live grid entity.
/// @param x World X tile coordinate.
/// @param z World Z tile coordinate.
/// @return True for a live entity at this tile.
bool at(const Simulation& simulation, EntityId id, double x, double z) {
  return simulation.scene().alive(id) && simulation.scene().get(id).spatial.position[0] == x &&
         simulation.scene().get(id).spatial.position[2] == z;
}
/// @brief Require exactly one matching named gameplay entity.
/// @param slot Existing ID or zero.
/// @param id Newly discovered ID.
void unique(EntityId& slot, EntityId id) {
  if (slot)
    throw std::invalid_argument("duplicate Courier gameplay entity");
  slot = id;
}
/// @brief Stateful turn-based source-compatible Crystal Courier rules.
class GridCourier final : public Behavior {
  EntityId              player_{0};       ///< Unique player.
  EntityId              status_{0};       ///< Visible status entity.
  EntityId              exit_{0};         ///< Goal tile entity.
  std::vector<EntityId> walls_;           ///< Authored blocking tile IDs.
  std::vector<EntityId> crystals_;        ///< Remaining pickup IDs.
  std::vector<EntityId> traps_;           ///< Authored terminal trap IDs.
  std::size_t           moves_{0};        ///< Successful cardinal move count.
  std::size_t           collected_{0};    ///< Collected pickup count.
  std::size_t           total_{0};        ///< Initial pickup count.
  bool                  finished_{false}; ///< Terminal game state.
 public:
  /// @brief Discover source-named components once and publish the initial objective.
  void on_start(Simulation& simulation, EntityId, double) override {
    for (auto id : simulation.scene().entities()) {
      const auto& name = simulation.scene().get(id).name;
      if (name == "Player")
        unique(player_, id);
      else if (name.compare(0, 8, "Status: ") == 0)
        unique(status_, id);
      else if (name == "Exit")
        unique(exit_, id);
      else if (name == "Wall")
        walls_.push_back(id);
      else if (name == "Crystal")
        crystals_.push_back(id);
      else if (name == "Trap")
        traps_.push_back(id);
    }
    if (!player_ || !status_ || !exit_)
      throw std::invalid_argument("Courier needs Player, Status and Exit entities");
    total_ = crystals_.size();
    message(
        simulation,
        status_,
        "Collect all " + std::to_string(total_) + " crystals, then reach E. Avoid X.");
  }
  /// @brief Apply one cardinal tile move, pickup/trap rules and terminal exit behavior.
  void on_update(Simulation& simulation, EntityId, double) override {
    if (finished_)
      return;
    const double dx = simulation.input("move_x"), dz = simulation.input("move_z");
    if (dx == 0 && dz == 0)
      return;
    if ((dx != 0 && dz != 0) || std::trunc(dx) != dx || std::trunc(dz) != dz) {
      message(simulation, status_, "Move one tile along one axis.");
      return;
    }
    auto         body = simulation.scene().get(player_).spatial;
    const double x = body.position[0] + dx, z = body.position[2] + dz;
    for (auto id : walls_)
      if (at(simulation, id, x, z)) {
        message(
            simulation,
            status_,
            "A wall blocks the way. Crystals: " + std::to_string(collected_) + "/" +
                std::to_string(total_));
        return;
      }
    ++moves_;
    body.position[0] = x;
    body.position[2] = z;
    simulation.set_motion(player_, body);
    for (auto id : traps_)
      if (at(simulation, id, x, z)) {
        simulation.destroy(player_);
        message(
            simulation,
            status_,
            "Game over! You stepped on a trap after " + std::to_string(moves_) + " moves.");
        finished_ = true;
        return;
      }
    for (auto it = crystals_.begin(); it != crystals_.end();)
      if (at(simulation, *it, x, z)) {
        simulation.destroy(*it);
        it = crystals_.erase(it);
        ++collected_;
      } else
        ++it;
    if (at(simulation, exit_, x, z)) {
      if (crystals_.empty()) {
        const auto trophy = simulation.spawn("Trophy");
        auto       pose   = simulation.scene().get(trophy).spatial;
        pose.position     = {x, body.position[1], z};
        simulation.set_motion(trophy, pose);
        message(
            simulation,
            status_,
            "Victory! Delivered all " + std::to_string(total_) + " crystals in " +
                std::to_string(moves_) + " moves.");
        finished_ = true;
      } else
        message(
            simulation,
            status_,
            "Exit locked: " + std::to_string(crystals_.size()) + " crystals remain.");
    } else
      message(
          simulation,
          status_,
          "Crystals: " + std::to_string(collected_) + "/" + std::to_string(total_) +
              " | Moves: " + std::to_string(moves_));
  }
};
/// @brief Recognize the original Crystal followed by space and decimal digits naming rule.
/// @param name Possible authored pickup name.
/// @return True for a numbered crystal.
bool numbered_crystal(const std::string& name) {
  return name.size() > 8 && name.compare(0, 8, "Crystal ") == 0 &&
         std::all_of(name.begin() + 8, name.end(), [](char c) { return c >= '0' && c <= '9'; });
}
/// @brief Native continuous-movement Courier with physical proximity and grounded jumping.
class Courier3D final : public Behavior {
  EntityId              player_{0};        ///< Unique player entity.
  EntityId              status_{0};        ///< Visible status entity.
  EntityId              exit_{0};          ///< Physical exit trigger.
  std::vector<EntityId> crystals_;         ///< Authored numbered pickup IDs.
  std::vector<EntityId> hazards_;          ///< Authored physical hazard IDs.
  std::size_t           total_{0};         ///< Initial pickup count.
  std::size_t           collected_{0};     ///< Collected pickup count.
  bool                  done_{false};      ///< Terminal gameplay state.
  bool                  jump_held_{false}; ///< Previous jump control for edge detection.

  /// @brief Clear linear velocity while preserving the player's physical settings.
  void stop(Simulation& simulation) {
    auto body            = simulation.scene().get(player_).spatial;
    body.linear_velocity = {0, 0, 0};
    simulation.set_motion(player_, body);
  }

 public:
  /// @brief Discover authored gameplay objects and publish initial progress.
  void on_start(Simulation& simulation, EntityId, double) override {
    for (auto id : simulation.scene().entities()) {
      const auto& name = simulation.scene().get(id).name;
      if (name == "Player")
        unique(player_, id);
      else if (name.compare(0, 8, "Status: ") == 0)
        unique(status_, id);
      else if (name == "Exit")
        unique(exit_, id);
      else if (numbered_crystal(name))
        crystals_.push_back(id);
      else if (name == "Hazard")
        hazards_.push_back(id);
    }
    if (!player_ || !status_ || !exit_)
      throw std::invalid_argument("3D Courier needs Player, Status and Exit entities");
    total_ = crystals_.size();
    message(
        simulation,
        status_,
        "0 / " + std::to_string(total_) + " crystals - explore, jump, and push crates");
  }
  /// @brief Resolve physical win/loss and apply bounded planar acceleration plus jump edges.
  void on_update(Simulation& simulation, EntityId, double delta) override {
    if (done_)
      return;
    auto body = simulation.scene().get(player_).spatial;
    if (body.position[1] < -5) {
      done_ = true;
      message(simulation, status_, "You fell! Press R to restart");
      return;
    }
    for (auto id : hazards_)
      if (simulation.scene().alive(id) && simulation.touching(player_, id)) {
        done_ = true;
        stop(simulation);
        message(simulation, status_, "Lava! Press R to restart");
        return;
      }
    for (auto id : crystals_)
      if (simulation.scene().alive(id) && simulation.touching(player_, id)) {
        simulation.destroy(id);
        ++collected_;
        message(
            simulation,
            status_,
            std::to_string(collected_) + " / " + std::to_string(total_) + " crystals - " +
                (collected_ == total_ ? "reach the golden portal!" : "keep exploring"));
      }
    if (collected_ == total_ && simulation.touching(player_, exit_)) {
      done_ = true;
      stop(simulation);
      const auto trophy = simulation.spawn("Trophy");
      auto       pose   = simulation.scene().get(trophy).spatial;
      pose.position     = simulation.scene().get(exit_).spatial.position;
      pose.position[1] += 2;
      simulation.set_motion(trophy, pose);
      message(simulation, status_, "VICTORY! All crystals delivered - R to play again");
      return;
    }
    double       dx = simulation.input("move_x"), dz = simulation.input("move_z");
    const double length = std::hypot(dx, dz);
    if (length > 1) {
      dx /= length;
      dz /= length;
    }
    body = simulation.scene().get(player_).spatial;
    body.linear_velocity[0] +=
        std::clamp(dx * 4.5 - body.linear_velocity[0], -30 * delta, 30 * delta);
    body.linear_velocity[2] +=
        std::clamp(dz * 4.5 - body.linear_velocity[2], -30 * delta, 30 * delta);
    simulation.set_motion(player_, body);
    const bool jump = simulation.input("jump") > 0;
    if (jump && !jump_held_ && simulation.grounded(player_))
      simulation.apply_impulse(player_, {0, 12, 0});
    jump_held_ = jump;
  }
};
/// @brief Stateful reference orbit camera around the world origin, looking along local -Z.
class OrbitCamera final : public Behavior {
  double angle_{0};  ///< Accumulated azimuth in radians.
  double radius_{0}; ///< Authored horizontal radius in metres.
  double height_{0}; ///< Authored height above the origin.
  double pitch_{0};  ///< Downward local pitch in radians.
 public:
  /// @brief Initialize orbit radius/height and reject a camera on the vertical axis.
  void on_start(Simulation& simulation, EntityId self, double) override {
    const auto p = simulation.scene().get(self).spatial.position;
    radius_      = std::hypot(p[0], p[2]);
    if (radius_ <= 0)
      throw std::invalid_argument("orbit camera needs nonzero horizontal radius");
    height_ = p[1];
    angle_  = std::atan2(p[0], p[2]);
    pitch_  = -std::atan2(height_, radius_);
  }
  /// @brief Advance a six-degree-per-second orbit and update its native camera quaternion.
  void on_update(Simulation& simulation, EntityId self, double delta) override {
    const double pi = std::acos(-1.);
    angle_          = std::fmod(angle_ + pi / 30 * delta, 2 * pi);
    auto body       = simulation.scene().get(self).spatial;
    body.position   = {radius_ * std::sin(angle_), height_, radius_ * std::cos(angle_)};
    const double cy = std::cos(angle_ / 2), sy = std::sin(angle_ / 2), cp = std::cos(pitch_ / 2),
                 sp = std::sin(pitch_ / 2);
    body.rotation   = {cy * cp, cy * sp, sy * cp, -sy * sp};
    simulation.set_motion(self, body);
  }
};
/// @brief Reference kinematic lift driven by a bounded sinusoidal vertical velocity.
class OscillatingLift final : public Behavior {
  double time_{0}; ///< Elapsed native tick seconds for this entity.
 public:
  /// @brief Accumulate time and set the physical kinematic velocity without teleporting.
  void on_update(Simulation& simulation, EntityId self, double delta) override {
    time_ += delta;
    auto body            = simulation.scene().get(self).spatial;
    body.linear_velocity = {0, .8 * std::cos(time_), 0};
    simulation.set_motion(self, body);
  }
};
} // namespace
BehaviorRegistry courier_behaviors() {
  BehaviorRegistry registry;
  registry.add("courier_grid", [] { return std::make_unique<GridCourier>(); });
  registry.add("courier_3d", [] { return std::make_unique<Courier3D>(); });
  registry.add("orbit_camera", [] { return std::make_unique<OrbitCamera>(); });
  registry.add("oscillating_lift", [] { return std::make_unique<OscillatingLift>(); });
  return registry;
}
SceneImport courier_import(bool three_dimensional) {
  SceneImport result;
  result.entities[1] = three_dimensional ? "courier_3d" : "courier_grid";
  if (three_dimensional) {
    result.entities[3]   = "orbit_camera";
    result.entities[110] = "oscillating_lift";
  }
  return result;
}
} // namespace datamunge::game
