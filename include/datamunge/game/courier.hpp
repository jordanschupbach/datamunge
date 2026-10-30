#pragma once
/// @file courier.hpp
/// @brief Native C++ Crystal Courier rules and explicit mappings for the reference scenes.
#include <datamunge/game/io.hpp>
#include <datamunge/game/simulation.hpp>
namespace datamunge::game {
/// @brief Build independently owned native factories for both Crystal Courier reference games.
/// Registers courier_grid (turn-based tile movement/pickups/traps/exit), courier_3d
/// (accelerated planar movement, grounded edge-triggered jump, hazards and pickups),
/// orbit_camera and oscillating_lift. Per-entity factory instances own all mutable
/// state. Names/components/prefabs follow the frozen VSim reference scenes.
/// No Lua interpreter or renderer is required. Restart by constructing a fresh
/// Simulation from the authored scene, which also recreates behavior state.
/// @return Owned native registry; caller may add other factories before Simulation construction.
/// @par Example
/// @code
/// const auto behaviors=dm::game::courier_behaviors();
/// std::print("{}\n",behaviors.contains("courier_3d")); // true
/// @endcode
/// @par References
/// - Frozen VSim examples/crystal_courier/game.lua and
///   examples/crystal_courier_3d/{game.lua,camera.lua,scene.json}.
/// @see courier_import(), Simulation
BehaviorRegistry courier_behaviors();
/// @brief Explicitly map known script-bearing IDs in one frozen Courier reference scene.
/// This opt-in mapping does not inspect or execute source. Grid scene ID 1 maps to
/// courier_grid. The 3D scene maps IDs 1/3/110 to courier_3d/orbit_camera/oscillating_lift.
/// Use a custom SceneImport for scenes with other IDs or different game rules.
/// @param three_dimensional False for grid Courier (default), true for the 3D reference.
/// @return Owned mappings accepted by parse_scene/load_scene.
/// @par Example
/// @code
/// const auto mapping=dm::game::courier_import(true);
/// std::print("{}\n",mapping.entities.at(3)); // orbit_camera
/// @endcode
/// @see courier_behaviors(), load_scene()
SceneImport courier_import(bool three_dimensional = false);
} // namespace datamunge::game
