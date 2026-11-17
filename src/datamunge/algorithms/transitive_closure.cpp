/// @file transitive_closure.cpp
/// @brief Actual checked Warshall row gating, path updates and phase capture.
#include <datamunge/algorithms/transitive_closure.hpp>
#include <limits>
#include <stdexcept>
namespace datamunge::algorithms {
/// @brief Private per-call reachability producer.
namespace {
/// @brief Validate complete geometry and execute the actual gated inner updates.
ClosureRun execute(
    std::size_t                                             n,
    const std::vector<std::pair<std::size_t, std::size_t>>& edges,
    bool                                                    reflexive,
    const ClosureOptions&                                   o,
    bool                                                    capture) {
  for (auto [u, v] : edges)
    if (u >= n || v >= n)
      throw std::invalid_argument("closure endpoint");
  auto cap = std::numeric_limits<std::size_t>::max();
  if (n && n > cap / n)
    throw std::length_error("closure square capacity");
  auto cells = n * n;
  if (o.max_cells && cells > o.max_cells)
    throw std::length_error("closure matrix");
  if (capture && (cells == cap || (n && cells + 1 > cap / n) ||
                  (o.max_history_values && n * (cells + 1) > o.max_history_values)))
    throw std::length_error("closure complete phase history");
  ClosureRun r;
  r.reachable.assign(n, std::vector<bool>(n));
  for (auto [u, v] : edges) {
    r.reachable[u][v] = true;
    ++r.seeded_edges;
  }
  if (reflexive)
    for (std::size_t i = 0; i < n; ++i)
      r.reachable[i][i] = true;
  for (std::size_t k = 0; k < n; ++k) {
    for (std::size_t i = 0; i < n; ++i) {
      ++r.row_tests;
      if (r.reachable[i][k])
        for (std::size_t j = 0; j < n; ++j) {
          if (r.candidate_updates == cap || (o.max_updates && r.candidate_updates >= o.max_updates))
            throw std::length_error("closure actual updates");
          ++r.candidate_updates;
          bool before       = r.reachable[i][j];
          r.reachable[i][j] = before || r.reachable[k][j];
          if (!before && r.reachable[i][j])
            ++r.new_paths;
        }
    }
    if (capture)
      r.phases.push_back({k, r.candidate_updates, r.new_paths, r.reachable});
  }
  return r;
}
} // namespace
ClosureRun transitive_closure_run(
    std::size_t                                             n,
    const std::vector<std::pair<std::size_t, std::size_t>>& edges,
    bool                                                    reflexive,
    const ClosureOptions&                                   options) {
  return execute(n, edges, reflexive, options, false);
}
ClosureRun transitive_closure_trace(
    std::size_t                                             n,
    const std::vector<std::pair<std::size_t, std::size_t>>& edges,
    bool                                                    reflexive,
    const ClosureOptions&                                   options) {
  return execute(n, edges, reflexive, options, true);
}
std::vector<std::vector<bool>> transitive_closure(
    std::size_t                                             n,
    const std::vector<std::pair<std::size_t, std::size_t>>& edges,
    bool                                                    reflexive) {
  return transitive_closure_run(n, edges, reflexive).reachable;
}
} // namespace datamunge::algorithms
