#pragma once

/// \file flood_fill.hpp
/// \brief Flood fill: recolor a connected region of a grid sharing a starting value
///        (the "paint bucket" of image editors).
///
/// Given a grid, a start cell, and a new value, flood fill changes the start cell and
/// every cell reachable from it through neighbors of the *same original value* to the
/// new value -- the connected monochromatic region containing the seed. It is a graph
/// traversal (BFS or DFS) over the implicit grid graph where edges join same-valued
/// neighbors, so it visits each cell of the region once. This implementation uses an
/// explicit stack (iterative DFS) to avoid recursion-depth limits on large regions and
/// supports 4- or 8-connectivity. Flood fill is the paint-bucket tool, the core of
/// connected-component labeling and maze/region analysis, and the boundary case of
/// region-growing segmentation.

#include <cstddef>
#include <utility>
#include <vector>

namespace datamunge::algorithms {

/// \brief Flood-fill the region of \p grid connected to (\p sr, \p sc) and sharing its value.
///
/// \param grid          Mutable grid of integer labels/colors (row-major).
/// \param sr, sc        Seed cell.
/// \param new_value     Value to paint the region.
/// \param eight_connected  Use 8-connectivity if true, else 4-connectivity.
/// \return the number of cells filled.
inline std::size_t flood_fill(std::vector<std::vector<int>>& grid, std::size_t sr, std::size_t sc,
                              int new_value, bool eight_connected = false) {
    const std::size_t rows = grid.size();
    if (rows == 0) return 0;
    const std::size_t cols = grid.front().size();
    if (sr >= rows || sc >= cols) return 0;

    const int target = grid[sr][sc];
    if (target == new_value) return 0;  // nothing to do

    const int dr4[] = {-1, 1, 0, 0}, dc4[] = {0, 0, -1, 1};
    const int dr8[] = {-1, -1, -1, 0, 0, 1, 1, 1}, dc8[] = {-1, 0, 1, -1, 1, -1, 0, 1};
    const int* dr = eight_connected ? dr8 : dr4;
    const int* dc = eight_connected ? dc8 : dc4;
    const int  nd = eight_connected ? 8 : 4;

    std::vector<std::pair<std::size_t, std::size_t>> stack = {{sr, sc}};
    grid[sr][sc]      = new_value;
    std::size_t count = 0;
    while (!stack.empty()) {
        const auto [r, c] = stack.back();
        stack.pop_back();
        ++count;
        for (int k = 0; k < nd; ++k) {
            const int nr = static_cast<int>(r) + dr[k], nc = static_cast<int>(c) + dc[k];
            if (nr < 0 || nr >= static_cast<int>(rows) || nc < 0 || nc >= static_cast<int>(cols)) continue;
            if (grid[static_cast<std::size_t>(nr)][static_cast<std::size_t>(nc)] == target) {
                grid[static_cast<std::size_t>(nr)][static_cast<std::size_t>(nc)] = new_value;
                stack.push_back({static_cast<std::size_t>(nr), static_cast<std::size_t>(nc)});
            }
        }
    }
    return count;
}

}  // namespace datamunge::algorithms
