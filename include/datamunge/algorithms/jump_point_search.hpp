#pragma once
/// @file jump_point_search.hpp
/// @brief Checked four-neighbor corridor-jump A* under the historical search entry point.
#include <datamunge/algorithms/d_star.hpp>
namespace datamunge::algorithms {
/// @brief Find an optimal unit-cost four-neighbor route, compressing only straight corridors.
/// The historical name denotes this corridor-jump A* variant, NOT the full
/// symmetry-pruning rules of published Jump Point Search. Coordinates are
/// (row,column), true cells blocked. Each ray stops at the goal, a possible
/// perpendicular turn/junction, or its last cell before a wall/boundary. Thus
/// intermediate junctions are retained, correcting the old ray-to-wall shortcut
/// that could miss paths. A shortest path cannot turn inside a degree-two
/// straight corridor, so replacing it by a length-weighted edge preserves cost.
/// Manhattan is consistent on these macro edges. Lazy heap ties prefer row-major
/// cell id; directions are down,up,right,left, and equal-cost parents are retained.
/// Output restores every unit step with both endpoints. Blocked/out-of-bounds
/// endpoints return empty/zero/false; empty or ragged grids throw. Start=goal on
/// a free cell returns one coordinate and zero cost. Caller data are unchanged;
/// output is owned, all state local, concurrent calls independent. Dimensions and
/// total cells must fit int; arithmetic/priorities use checked bounds and uint64.
/// Worst-case O(N*L+N*log(N+1)) work for N cells and maximum ray length L,
/// O(N) labels/heap/output storage. Open rooms retain every turn opportunity,
/// so this variant promises neither open-room symmetry pruning nor a speedup.
/// @param blocked Borrowed nonempty rectangular occupancy grid.
/// @param start Starting row/column, possibly outside or blocked.
/// @param goal Target row/column, possibly outside or blocked.
/// @return Owned shortest unit route, or empty/zero/false if unreachable/invalid terminals.
/// @throws std::invalid_argument On empty or ragged grid.
/// @throws std::length_error On dimension/cell/container representability limits.
/// @throws std::bad_alloc On allocation failure.
/// @par Example
/// @code
/// const auto r=dm::algorithms::jump_point_search({{true,false,true},{false,false,false},{true,true,true}},{1,0},{0,1});
/// std::print("{} {}\n",r.found,r.cost); // true 2
/// @endcode
/// @par References
/// - Hart,Nilsson,Raphael(1968), A Formal Basis for the Heuristic Determination
///   of Minimum Cost Paths, https://doi.org/10.1109/TSSC.1968.300136 (A*).
/// - Harabor,Grastien(2011), https://doi.org/10.1609/aaai.v25i1.7994;
///   published JPS symmetry pruning is not implemented by this corridor variant.
/// @see GridPathResult, plot_grid_route()
GridPathResult jump_point_search(
    const std::vector<std::vector<bool>>& blocked,
    std::pair<int, int>                   start,
    std::pair<int, int>                   goal);
} // namespace datamunge::algorithms
