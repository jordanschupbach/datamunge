#pragma once
/// @file physics_backend.hpp
/// @brief Explicit native/reference selection for articulated workcell physics.
#include <string>
namespace datamunge::sim {
/// @brief Articulated implementation; no silent fallback between engines.
/// @par Example
/// @code
/// auto engine=dm::sim::ArticulatedBackend::Native;
/// @endcode
enum class ArticulatedBackend {
  Native, ///< In-house C++ solver: no external physics runtime.
  MuJoCo  ///< Optional comparison engine, requiring DATAMUNGE_ENABLE_MUJOCO.
};
/// @brief Identify the selected engine without creating a world.
/// @param backend Explicit native or optional reference backend.
/// @return Native version tag or MuJoCo runtime version; empty if reference unavailable.
/// @throws std::invalid_argument Unknown backend enumeration.
/// @par Example
/// @code
/// auto version=dm::sim::articulated_physics_version(dm::sim::ArticulatedBackend::Native);
/// @endcode
std::string articulated_physics_version(ArticulatedBackend backend);
} // namespace datamunge::sim
