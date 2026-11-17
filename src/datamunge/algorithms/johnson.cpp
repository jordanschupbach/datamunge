/// @file johnson.cpp
/// @brief Shared checked Johnson kernel with actual scalar event recording.
#include <cmath>
#include <datamunge/algorithms/johnson.hpp>
#include <limits>
#include <queue>
#include <stdexcept>
#include <utility>
namespace datamunge::algorithms {
/// @brief Checked admission, arithmetic and same-execution Johnson implementation details.
namespace johnson_detail {
/// @brief Increment one logical work counter with representability checking.
/// @param n Counter mutated only when increment is representable.
/// @throws std::length_error On counter overflow.
void increment(std::size_t& n) {
  if (n == std::numeric_limits<std::size_t>::max())
    throw std::length_error("Johnson counter overflow");
  ++n;
}
/// @brief Reject nonfinite produced arithmetic without conflating it with unreachable infinity.
/// @param x Actual intermediate value.
/// @return The same finite value.
/// @throws std::overflow_error On a nonfinite intermediate.
long double finite(long double x) {
  if (!std::isfinite(x))
    throw std::overflow_error("Johnson arithmetic overflow");
  return x;
}
/// @brief Validate the entire graph and admit complete input/output record counts.
/// @param n Vertex count.
/// @param edges Borrowed original directed edges.
/// @param o Optional complete record ceilings.
/// @return Representable matrix cell count n*n.
/// @throws std::invalid_argument On invalid endpoint/weight.
/// @throws std::length_error On count overflow or enabled complete ceiling.
std::size_t validate(std::size_t n, const std::vector<WeightedEdge>& edges, JohnsonOptions o) {
  for (const auto& [u, v, w] : edges)
    if (u >= n || v >= n || !std::isfinite(w))
      throw std::invalid_argument("Johnson invalid edge");
  const auto max = std::numeric_limits<std::size_t>::max();
  if (edges.size() > max - n || (n && n > max / n))
    throw std::length_error("Johnson shape overflow");
  const auto cells = n * n;
  if ((o.max_input_records && n + edges.size() > o.max_input_records) ||
      (o.max_distance_cells && cells > o.max_distance_cells))
    throw std::length_error("Johnson complete admission");
  return cells;
}
/// @brief Execute checked Johnson once, optionally recording actual operations.
/// @param n Vertex count.
/// @param edges Borrowed original directed edges in input order.
/// @param o Complete input/output and online event ceilings.
/// @param trace Whether to capture graph and scalar events.
/// @return Owned producing result/history; no partial result on exceptions.
/// @throws std::invalid_argument On invalid input edges.
/// @throws std::length_error On enabled ceilings or size/counter/container overflow.
/// @throws std::overflow_error On produced arithmetic/output overflow.
/// @throws std::domain_error On negative rounded reduced weight.
/// @throws std::bad_alloc On allocation failure.
JohnsonTrace
execute(std::size_t n, const std::vector<WeightedEdge>& edges, JohnsonOptions o, bool trace) {
  const auto   cells = validate(n, edges, o);
  JohnsonTrace t;
  t.vertex_count = n;
  if (trace)
    t.edges = edges;
  auto& a             = t.analysis;
  auto& w             = a.work;
  w.matrix_cells      = cells;
  const auto infinity = std::numeric_limits<long double>::infinity();
  a.result.distance.assign(n, std::vector<double>(n, std::numeric_limits<double>::infinity()));
  a.potentials.assign(n, 0);
  /// @brief Admit the next actual operation before arithmetic, queue edits or state mutation.
  const auto admit = [&]() {
    if (trace && ((o.max_events && t.events.size() >= o.max_events) ||
                  t.events.size() == t.events.max_size()))
      throw std::length_error("Johnson actual event ceiling");
  };
  /// @brief Append an admitted scalar actual event only in recording mode.
  /// @param e Owned scalar actual operation to append.
  const auto record = [&](JohnsonEvent e) {
    if (trace)
      t.events.push_back(e);
  };
  for (std::size_t pass = 0; pass < n; ++pass) {
    bool changed = false;
    for (std::size_t i = 0; i < edges.size(); ++i) {
      admit();
      const auto [u, v, cost] = edges[i];
      const auto before       = a.potentials[v];
      const auto candidate    = finite(a.potentials[u] + cost);
      increment(w.potential_checks);
      if (candidate < before) {
        a.potentials[v] = candidate;
        changed         = true;
        increment(w.potential_updates);
      }
      record(
          {JohnsonAction::Potential, pass, 0, i, u, v, before, candidate, a.potentials[v], false});
    }
    increment(w.potential_passes);
    if (!changed)
      break;
    if (pass + 1 == n) {
      a.result.has_negative_cycle = true;
      return t;
    }
  }
  std::vector<std::vector<std::size_t>> adjacency(n);
  a.reduced_weights.reserve(edges.size());
  for (std::size_t i = 0; i < edges.size(); ++i) {
    admit();
    const auto [u, v, cost] = edges[i];
    const auto reduced =
        finite(finite(static_cast<long double>(cost) + a.potentials[u]) - a.potentials[v]);
    if (reduced < 0)
      throw std::domain_error("Johnson negative rounded reduced weight");
    a.reduced_weights.push_back(reduced);
    adjacency[u].push_back(i);
    increment(w.reweighted_edges);
    record({JohnsonAction::Reweight, 0, 0, i, u, v, cost, reduced, reduced, false});
  }
  for (std::size_t s = 0; s < n; ++s) {
    std::vector<long double> d(n, infinity);
    using Key = std::pair<long double, std::size_t>;
    std::priority_queue<Key, std::vector<Key>, std::greater<Key>> q;
    admit();
    d[s] = 0;
    q.push({0, s});
    increment(w.heap_pushes);
    record({JohnsonAction::Seed, 0, s, 0, s, s, infinity, 0, 0, false});
    while (!q.empty()) {
      admit();
      const auto [key, u] = q.top();
      q.pop();
      const bool stale = key != d[u];
      increment(w.heap_pops);
      if (stale)
        increment(w.stale_pops);
      record({JohnsonAction::Pop, 0, s, 0, u, u, d[u], key, d[u], stale});
      if (stale)
        continue;
      for (auto i : adjacency[u]) {
        admit();
        const auto v      = std::get<1>(edges[i]);
        const auto before = d[v], candidate = finite(key + a.reduced_weights[i]);
        increment(w.dijkstra_checks);
        if (candidate < before) {
          d[v] = candidate;
          q.push({candidate, v});
          increment(w.dijkstra_updates);
          increment(w.heap_pushes);
        }
        record({JohnsonAction::Relax, 0, s, i, u, v, before, candidate, d[v], false});
      }
    }
    for (std::size_t v = 0; v < n; ++v)
      if (std::isfinite(d[v])) {
        admit();
        const auto restored = finite(d[v] + finite(-a.potentials[s] + a.potentials[v]));
        if (std::fabs(restored) > std::numeric_limits<double>::max())
          throw std::overflow_error("Johnson finite distance exceeds double");
        a.result.distance[s][v] = static_cast<double>(restored);
        increment(w.restored_cells);
        record({JohnsonAction::Restore, 0, s, 0, s, v, d[v], restored, restored, false});
      }
  }
  return t;
}
} // namespace johnson_detail
JohnsonAllPairsResult johnson_all_pairs_shortest_paths(
    std::size_t                      n,
    const std::vector<WeightedEdge>& edges) {
  return johnson_detail::execute(n, edges, {0, 0, 0}, false).analysis.result;
}
JohnsonAnalysis
johnson_analyze(std::size_t n, const std::vector<WeightedEdge>& edges, JohnsonOptions o) {
  return johnson_detail::execute(n, edges, o, false).analysis;
}
JohnsonTrace
johnson_trace(std::size_t n, const std::vector<WeightedEdge>& edges, JohnsonOptions o) {
  return johnson_detail::execute(n, edges, o, true);
}
} // namespace datamunge::algorithms
