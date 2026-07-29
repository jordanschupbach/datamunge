#pragma once

#include <cstddef>
#include <utility>
#include <vector>

namespace datamunge::algorithms {

inline constexpr std::size_t kGeneralMatchingUnmatched = static_cast<std::size_t>(-1);

struct GeneralMatchingResult {
  std::vector<std::size_t> mate;
  std::size_t              size{0};
};

/// Edmonds' blossom algorithm for maximum-cardinality matching in an undirected graph.
///
/// Vertices are 0..vertex_count-1. Self-loops are ignored, duplicate edges are accepted,
/// and an out-of-range endpoint throws std::invalid_argument. Runs in O(V^3) time.
GeneralMatchingResult blossom_maximum_matching(
    std::size_t vertex_count,
    const std::vector<std::pair<std::size_t, std::size_t>>& edges);

} // namespace datamunge::algorithms
