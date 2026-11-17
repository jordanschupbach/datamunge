/// @file search_more.cpp
/// @brief Complete four-neighbor corridor-jump A* retaining every possible turn.
#include <algorithm>
#include <cstdint>
#include <datamunge/algorithms/jump_point_search.hpp>
#include <limits>
#include <queue>
#include <stdexcept>
namespace datamunge::algorithms {
GridPathResult jump_point_search(
    const std::vector<std::vector<bool>>& blocked,
    std::pair<int, int>                   start,
    std::pair<int, int>                   goal) {
  if (blocked.empty() || blocked.front().empty())
    throw std::invalid_argument("jump search: empty grid");
  const auto rows = blocked.size(), cols = blocked.front().size();
  for (const auto& row : blocked)
    if (row.size() != cols)
      throw std::invalid_argument("jump search: ragged grid");
  const auto limit = static_cast<std::size_t>(std::numeric_limits<int>::max());
  if (rows > limit || cols > limit || rows > limit / cols)
    throw std::length_error("jump search: cell ids exceed int");
  const auto count = rows * cols;
  /// @brief Test original bounds and occupancy without flattening invalid coordinates.
  const auto open = [&](int r, int c) {
    return r >= 0 && c >= 0 && std::size_t(r) < rows && std::size_t(c) < cols && !blocked[r][c];
  };
  if (!open(start.first, start.second) || !open(goal.first, goal.second))
    return {};
  /// @brief Map an already checked valid coordinate to its row-major cell id.
  const auto id = [&](int r, int c) { return std::size_t(r) * cols + std::size_t(c); };
  /// @brief Exact consistent Manhattan priority component on the bounded integer grid.
  const auto heuristic = [&](int r, int c) {
    return std::uint64_t(r > goal.first ? r - goal.first : goal.first - r) +
           std::uint64_t(c > goal.second ? c - goal.second : goal.second - c);
  };
  const auto                 src = id(start.first, start.second), dst = id(goal.first, goal.second);
  const auto                 infinity = std::numeric_limits<std::uint64_t>::max();
  std::vector<std::uint64_t> distance(count, infinity);
  std::vector<std::size_t>   parent(count, count);
  using Key = std::pair<std::uint64_t, std::size_t>;
  std::priority_queue<Key, std::vector<Key>, std::greater<Key>> queue;
  distance[src] = 0;
  queue.push({heuristic(start.first, start.second), src});
  const int dr[] = {1, -1, 0, 0}, dc[] = {0, 0, 1, -1};
  while (!queue.empty()) {
    const auto [priority, u] = queue.top();
    queue.pop();
    const int r = static_cast<int>(u / cols), c = static_cast<int>(u % cols);
    if (priority != distance[u] + heuristic(r, c))
      continue;
    if (u == dst)
      break;
    for (int direction = 0; direction < 4; ++direction) {
      int           nr = r, nc = c;
      std::uint64_t steps = 0;
      while (open(nr + dr[direction], nc + dc[direction])) {
        nr += dr[direction];
        nc += dc[direction];
        ++steps;
        if (std::pair{nr, nc} == goal)
          break;
        const bool turn = dr[direction] != 0 ? (open(nr, nc - 1) || open(nr, nc + 1))
                                             : (open(nr - 1, nc) || open(nr + 1, nc));
        if (turn)
          break;
      }
      if (!steps)
        continue;
      const auto v = id(nr, nc);
      const auto candidate = distance[u] + steps;
      if (candidate < distance[v]) {
        distance[v] = candidate;
        parent[v]   = u;
        queue.push({candidate + heuristic(nr, nc), v});
      }
    }
  }
  if (distance[dst] == infinity)
    return {};
  std::vector<std::size_t> vertices;
  for (auto v = dst; v != count; v = parent[v])
    vertices.push_back(v);
  std::reverse(vertices.begin(), vertices.end());
  GridPathResult result;
  result.found = true;
  result.cost  = static_cast<double>(distance[dst]);
  result.path.reserve(static_cast<std::size_t>(distance[dst]) + 1);
  result.path.push_back(start);
  for (std::size_t i = 1; i < vertices.size(); ++i) {
    const int r = static_cast<int>(vertices[i] / cols), c = static_cast<int>(vertices[i] % cols);
    auto [rr, cc] = result.path.back();
    while (rr != r || cc != c) {
      rr += (r > rr) - (r < rr);
      cc += (c > cc) - (c < cc);
      result.path.push_back({rr, cc});
    }
  }
  return result;
}
} // namespace datamunge::algorithms
