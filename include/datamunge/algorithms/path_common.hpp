#pragma once

#include <cstddef>

namespace datamunge::algorithms {

/// @brief Sentinel marking "no predecessor" in a shortest-path predecessor array (used by the
///        single-source shortest-path algorithms, e.g. Dijkstra and Bellman-Ford).
inline constexpr std::size_t kNoPredecessor = static_cast<std::size_t>(-1);

} // namespace datamunge::algorithms
