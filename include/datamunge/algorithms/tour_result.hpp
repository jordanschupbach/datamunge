#pragma once
/// @file tour_result.hpp
/// @brief Shared owned closed-tour result, preserving the original TSP result layout.
#include <cstddef>
#include <vector>
namespace datamunge::algorithms {
/// @brief Owned ordered vertex cycle and its producing algorithm's cost.
/// Nonempty TSP outputs repeat their starting vertex at the end, including [0,0]
/// for a singleton. Empty problems return an empty tour and zero cost. This plain
/// record does not validate manually assigned tours or certify optimality.
/// @par Example
/// @code
/// const auto r=dm::algorithms::christofides_tsp({{0,2},{2,0}});
/// std::print("{} {}\n",r.tour.size(),r.cost); // 3 4
/// @endcode
/// @see christofides_tsp(), nearest_neighbor_tsp()
struct TourResult {
  std::vector<std::size_t> tour; ///< Vertex order, with closing start for nonempty TSP results.
  double cost{0}; ///< Producer's sum along the returned order; not an optimality certificate.
};
}
