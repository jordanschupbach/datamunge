// Minimum-cost rectangular one-to-one assignment for DETR set matching (port
// of tflow's common/ml/hungarian.h). Rows (targets) must not outnumber
// columns (queries). The primal/dual augmenting-path method supports negative
// finite costs, O(rows^2*columns). Returns the selected column for each row;
// ties resolve deterministically. (The library's algorithms::hungarian solves
// square problems only.)
#pragma once

#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace hungarian {

inline std::vector<int> HungarianAssignment(const std::vector<std::vector<double>>& cost) {
  const int rows = int(cost.size());
  if (!rows) return {};
  const int cols = int(cost[0].size());
  if (rows > cols) throw std::invalid_argument("assignment needs at least as many columns as rows");
  for (const auto& row : cost) {
    if (int(row.size()) != cols) throw std::invalid_argument("ragged assignment matrix");
    for (double c : row)
      if (!std::isfinite(c)) throw std::invalid_argument("nonfinite assignment cost");
  }
  const auto          R = std::size_t(rows), C = std::size_t(cols);
  std::vector<double> row_potential(R + 1), col_potential(C + 1);
  std::vector<int>    owner(C + 1), previous(C + 1);
  for (int row = 1; row <= rows; ++row) {
    owner[0]                   = row;
    std::size_t         current = 0;
    std::vector<double> distance(C + 1, std::numeric_limits<double>::infinity());
    std::vector<bool>   visited(C + 1, false);
    do {
      visited[current]   = true;
      const int    source = owner[current];
      std::size_t  next   = 0;
      double       delta  = std::numeric_limits<double>::infinity();
      for (std::size_t col = 1; col <= C; ++col)
        if (!visited[col]) {
          const double reduced = cost[std::size_t(source - 1)][col - 1] - row_potential[std::size_t(source)] - col_potential[col];
          if (reduced < distance[col]) {
            distance[col] = reduced;
            previous[col] = int(current);
          }
          if (distance[col] < delta) {
            delta = distance[col];
            next  = col;
          }
        }
      for (std::size_t col = 0; col <= C; ++col) {
        if (visited[col]) {
          row_potential[std::size_t(owner[col])] += delta;
          col_potential[col] -= delta;
        } else {
          distance[col] -= delta;
        }
      }
      current = next;
    } while (owner[current] != 0);
    do {
      const auto parent = std::size_t(previous[current]);
      owner[current]    = owner[parent];
      current           = parent;
    } while (current != 0);
  }
  std::vector<int> assignment(R, -1);
  for (std::size_t col = 1; col <= C; ++col)
    if (owner[col]) assignment[std::size_t(owner[col] - 1)] = int(col - 1);
  return assignment;
}

}  // namespace hungarian
