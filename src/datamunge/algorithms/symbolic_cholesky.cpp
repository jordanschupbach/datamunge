/// @file symbolic_cholesky.cpp
/// @brief Actual symmetric structural elimination and new-fill capture.
#include <datamunge/algorithms/symbolic_cholesky.hpp>
#include <limits>
#include <set>
#include <stdexcept>
namespace datamunge::algorithms {
/// @brief Private graph normalization and producing state.
namespace {
/// @brief Normalize, eliminate and capture actual higher-neighbor fill insertions.
SymbolicCholeskyRun execute(
    const std::vector<std::vector<int>>& input,
    const SymbolicCholeskyOptions&       o,
    bool                                 capture) {
  auto n = input.size();
  if (n > std::size_t(std::numeric_limits<int>::max()) || (o.max_vertices && n > o.max_vertices))
    throw std::length_error("symbolic Cholesky vertices");
  for (const auto& row : input)
    for (int v : row)
      if (v < 0 || std::size_t(v) >= n)
        throw std::invalid_argument("symbolic Cholesky neighbor");
  std::vector<std::set<int>> adj(n);
  SymbolicCholeskyRun        r;
  r.lower.resize(n);
  std::size_t edges = 0, retained = 0;
  /// @brief Bound a newly inserted undirected edge before growing either adjacency row.
  auto edge = [&]() {
    if (edges == std::numeric_limits<std::size_t>::max() || (o.max_edges && edges >= o.max_edges))
      throw std::length_error("symbolic Cholesky edges");
    ++edges;
  };
  for (std::size_t i = 0; i < n; ++i)
    for (int v : input[i])
      if (v != int(i) && !adj[i].contains(v)) {
        edge();
        adj[i].insert(v);
        adj[v].insert(int(i));
      }
  r.input_edges = edges;
  for (std::size_t k = 0; k < n; ++k) {
    std::vector<int> higher;
    for (int v : adj[k])
      if (v > int(k))
        higher.push_back(v);
    for (int v : higher) {
      r.lower[v].push_back(int(k));
      ++r.factor_entries;
    }
    SymbolicCholeskyStep step;
    step.column = int(k);
    if (capture)
      step.higher = higher;
    for (std::size_t a = 0; a < higher.size(); ++a)
      for (std::size_t b = a + 1; b < higher.size(); ++b) {
        if (r.pair_tests == std::numeric_limits<std::size_t>::max() ||
            (o.max_pair_tests && r.pair_tests >= o.max_pair_tests))
          throw std::length_error("symbolic Cholesky pair work");
        ++r.pair_tests;
        int u = higher[a], v = higher[b];
        if (!adj[u].contains(v)) {
          edge();
          adj[u].insert(v);
          adj[v].insert(u);
          r.fill.emplace_back(u, v);
          if (capture)
            step.fill.emplace_back(u, v);
        }
      }
    if (capture) {
      auto limit = std::numeric_limits<std::size_t>::max();
      if (step.fill.size() > (limit - 1 - higher.size()) / 2)
        throw std::length_error("symbolic Cholesky history capacity");
      auto count = 1 + higher.size() + 2 * step.fill.size();
      if (retained > limit - count ||
          (o.max_history_values &&
           (retained > o.max_history_values || count > o.max_history_values - retained)))
        throw std::length_error("symbolic Cholesky history");
      retained += count;
      step.cumulative_pairs = r.pair_tests;
      r.steps.push_back(std::move(step));
    }
  }
  return r;
}
} // namespace
SymbolicCholeskyRun symbolic_cholesky_run(
    const std::vector<std::vector<int>>& adjacency,
    const SymbolicCholeskyOptions&       options) {
  return execute(adjacency, options, false);
}
SymbolicCholeskyRun symbolic_cholesky_trace(
    const std::vector<std::vector<int>>& adjacency,
    const SymbolicCholeskyOptions&       options) {
  return execute(adjacency, options, true);
}
std::vector<std::vector<int>> symbolic_cholesky(const std::vector<std::vector<int>>& adjacency) {
  return symbolic_cholesky_run(adjacency).lower;
}
std::vector<std::vector<int>> symbolic_cholesky_grid(std::size_t side) {
  if (side > 1000)
    throw std::length_error("symbolic grid side");
  std::vector<std::vector<int>> r(side * side);
  for (std::size_t row = 0; row < side; ++row)
    for (std::size_t col = 0; col < side; ++col) {
      int u = int(row * side + col);
      if (col + 1 < side) {
        r[u].push_back(u + 1);
        r[u + 1].push_back(u);
      }
      if (row + 1 < side) {
        int v = u + int(side);
        r[u].push_back(v);
        r[v].push_back(u);
      }
    }
  return r;
}
} // namespace datamunge::algorithms
