/// @file reverse_delete.cpp
/// @brief One actual reverse-delete connectivity producer, retaining original edge identities.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <datamunge/algorithms/reverse_delete.hpp>
#include <limits>
#include <numeric>
#include <stdexcept>
/// @brief Canonical undirected forest implementation.
namespace datamunge::algorithms {
/// @brief Private reverse-delete execution.
namespace {
/// @brief Owned actual connectivity labels with path compression and union by size.
struct Sets {
  std::vector<std::size_t> parent;     ///< Owned parent labels.
  std::vector<std::size_t> size;       ///< Actual root component sizes.
  std::size_t              components; ///< Actual current connected component count.

  /// @brief Construct singleton components.
  /// @param n Vertex count.
  explicit Sets(std::size_t n) : parent(n), size(n, 1), components(n) {
    std::iota(parent.begin(), parent.end(), 0);
  }
  /// @brief Find an actual root with path halving.
  /// @param x Valid vertex id.
  /// @return Actual root id.
  std::size_t root(std::size_t x) {
    while (parent[x] != x) {
      parent[x] = parent[parent[x]];
      x         = parent[x];
    }
    return x;
  }
  /// @brief Join two actual components when distinct.
  /// @param a First endpoint.
  /// @param b Second endpoint.
  void join(std::size_t a, std::size_t b) {
    a = root(a);
    b = root(b);
    if (a == b)
      return;
    if (size[a] < size[b])
      std::swap(a, b);
    parent[b] = a;
    size[a] += size[b];
    --components;
  }
};
/// @brief Execute one checked actual omission stream.
/// @param n Vertex count.
/// @param edges Borrowed complete graph.
/// @param o Policies.
/// @param capture Retain actual masks.
/// @return Owned same-pass result.
ReverseDeleteRun solve(
    std::size_t                      n,
    const std::vector<WeightedEdge>& edges,
    const ReverseDeleteOptions&      o,
    bool                             capture) {
  const auto m = edges.size();
  if ((o.max_vertices && n > o.max_vertices) || (o.max_edges && m > o.max_edges))
    throw std::length_error("reverse delete: input policy");
  for (const auto& [u, v, w] : edges)
    if (u >= n || v >= n || !std::isfinite(w))
      throw std::invalid_argument("reverse delete: invalid edge");
  ReverseDeleteRun out;
  out.order.resize(m);
  std::iota(out.order.begin(), out.order.end(), 0);
  std::sort(out.order.begin(), out.order.end(), [&](auto i, auto j) {
    auto a = std::get<2>(edges[i]), b = std::get<2>(edges[j]);
    return a > b || (a == b && i < j);
  });
  out.kept.assign(m, 1);
  std::size_t retained = 0;
  /// @brief Count actual components after omitting one original edge id.
  /// @param skip Omitted id; m means no omission.
  /// @return Actual connected component count.
  const auto components = [&](std::size_t skip) {
    Sets sets(n);
    if (out.connectivity_passes == SIZE_MAX)
      throw std::length_error("reverse delete: pass capacity");
    ++out.connectivity_passes;
    for (auto id : out.order) {
      if (out.edge_visits == SIZE_MAX ||
          (o.max_edge_visits && out.edge_visits >= o.max_edge_visits))
        throw std::length_error("reverse delete: work policy");
      ++out.edge_visits;
      if (out.kept[id] && id != skip) {
        if (out.union_attempts == SIZE_MAX)
          throw std::length_error("reverse delete: union capacity");
        ++out.union_attempts;
        sets.join(std::get<0>(edges[id]), std::get<1>(edges[id]));
      }
    }
    return sets.components;
  };
  out.initial_components = components(m);
  for (auto id : out.order) {
    if (capture) {
      if (m == SIZE_MAX || m + 1 > SIZE_MAX - retained ||
          (o.max_trace_entries &&
           (retained > o.max_trace_entries || m + 1 > o.max_trace_entries - retained)))
        throw std::length_error("reverse delete: history policy");
      retained += m + 1;
    }
    const auto count   = components(id);
    const bool removed = count == out.initial_components;
    if (removed)
      out.kept[id] = 0;
    if (capture)
      out.steps.push_back({id, count, removed, out.kept});
  }
  long double total = 0;
  for (auto id : out.order)
    if (out.kept[id]) {
      out.forest.edges.push_back(edges[id]);
      total += std::get<2>(edges[id]);
    }
  if (!std::isfinite(total) || std::fabs(total) > std::numeric_limits<double>::max())
    throw std::overflow_error("reverse delete: total weight");
  out.forest.total_weight = static_cast<double>(total);
  out.forest.is_connected = n <= 1 || out.initial_components == 1;
  return out;
}
} // namespace
ReverseDeleteRun reverse_delete_run(
    std::size_t                      n,
    const std::vector<WeightedEdge>& e,
    const ReverseDeleteOptions&      o) {
  return solve(n, e, o, false);
}
ReverseDeleteRun reverse_delete_trace(
    std::size_t                      n,
    const std::vector<WeightedEdge>& e,
    const ReverseDeleteOptions&      o) {
  return solve(n, e, o, true);
}
MinimumSpanningTree reverse_delete(std::size_t n, const std::vector<WeightedEdge>& e) {
  return solve(n, e, {}, false).forest;
}
std::vector<WeightedEdge> reverse_delete_cycle(std::size_t n) {
  if (n < 3)
    throw std::invalid_argument("reverse delete cycle: need three vertices");
  if (n > 100000)
    throw std::length_error("reverse delete cycle: vertex policy");
  std::vector<WeightedEdge> edges;
  edges.reserve(n);
  for (std::size_t i = 0; i + 1 < n; ++i)
    edges.emplace_back(i, i + 1, static_cast<double>(i + 1));
  edges.emplace_back(n - 1, 0, static_cast<double>(n));
  return edges;
}
} // namespace datamunge::algorithms
