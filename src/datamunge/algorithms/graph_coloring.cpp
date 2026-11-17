/// @file graph_coloring.cpp
/// @brief Shared checked degree-ordered first-fit sweep with actual optional constant-size events.
#include <algorithm>
#include <datamunge/algorithms/graph_coloring.hpp>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace datamunge::algorithms {
/// @brief Private checked counts and producing first-fit execution.
namespace coloring_detail {
/// @brief Sentinel used only in the internal not-yet-colored result vector.
constexpr std::size_t uncolored = std::numeric_limits<std::size_t>::max();
/// @brief Add logical operation/storage counts without unsigned wrapping.
/// @param a Current count.
/// @param b Increment.
/// @return Exact representable sum.
/// @throws std::length_error On size_t overflow.
std::size_t add(std::size_t a, std::size_t b) {
  if (b > std::numeric_limits<std::size_t>::max() - a)
    throw std::length_error("coloring count overflow");
  return a + b;
}
/// @brief Validate every endpoint before caps or adjacency allocation.
/// @param n Vertex count.
/// @param edges Borrowed original records, loops included.
/// @param o Complete input caps.
/// @throws std::invalid_argument On any endpoint outside range.
/// @throws std::length_error On complete vertex/edge caps or outer adjacency capacity.
void validate(
    std::size_t                                             n,
    const std::vector<std::pair<std::size_t, std::size_t>>& edges,
    const ColoringOptions&                                  o) {
  for (auto [u, v] : edges)
    if (u >= n || v >= n)
      throw std::invalid_argument("greedy_coloring: edge endpoint out of range (must be < n)");
  if ((o.max_vertices && n > o.max_vertices) || (o.max_edges && edges.size() > o.max_edges) ||
      n > std::vector<std::vector<std::size_t>>{}.max_size())
    throw std::length_error("coloring complete input limit");
}
/// @brief Admit a whole optional event before its producing sweep operation.
/// @param events Optional borrowed owned-history sink; null disables history.
/// @param cap Whole-event policy cap; zero disables it.
/// @throws std::length_error On next-record policy/container limit.
void admit(const std::vector<ColoringEvent>* events, std::size_t cap) {
  if (events && ((cap && events->size() >= cap) || events->size() >= events->max_size()))
    throw std::length_error("coloring event quota");
}
/// @brief Append one actual already-admitted constant-size event.
/// @param events Optional borrowed history sink.
/// @param e Producing event, owning primitive values.
/// @throws std::bad_alloc On record allocation failure.
void record(std::vector<ColoringEvent>* events, ColoringEvent e) {
  if (events)
    events->push_back(e);
}
/// @brief Prepare original multigraph degree order and execute first fit exactly once.
/// @param n Vertex count, including isolates.
/// @param edges Borrowed original edge records, never mutated or retained.
/// @param o Complete input and optional online event caps.
/// @param events Optional borrowed capture sink, initially empty.
/// @return Owned actual result, degree/order preparation and selected work.
/// @throws std::invalid_argument On endpoints, before caps/allocation.
/// @throws std::length_error On caps, counters or container exhaustion.
/// @throws std::bad_alloc On allocation failure.
ColoringAnalysis execute(
    std::size_t                                             n,
    const std::vector<std::pair<std::size_t, std::size_t>>& edges,
    ColoringOptions                                         o,
    std::vector<ColoringEvent>*                             events) {
  validate(n, edges, o);
  ColoringAnalysis a;
  auto&            work = a.work;
  work.vertices         = n;
  work.edge_records     = edges.size();
  std::vector<std::vector<std::size_t>> adj(n);
  for (auto [u, v] : edges) {
    if (u == v) {
      work.ignored_loops = add(work.ignored_loops, 1);
      continue;
    }
    work.adjacency_entries = add(work.adjacency_entries, 2);
    adj[u].push_back(v);
    adj[v].push_back(u);
  }
  a.degree.resize(n);
  for (std::size_t v = 0; v < n; ++v) {
    a.degree[v]     = adj[v].size();
    work.max_degree = std::max(work.max_degree, a.degree[v]);
  }
  a.order.resize(n);
  std::iota(a.order.begin(), a.order.end(), std::size_t{0});
  /// @brief Actual total degree/index comparator, with one selected comparison count.
  /// @param u First valid vertex ID.
  /// @param v Second valid vertex ID.
  /// @return Whether u precedes v by descending multiplicity degree and ascending ID.
  /// @throws std::length_error On selected comparison-count overflow.
  const auto before = [&](std::size_t u, std::size_t v) {
    work.sort_comparisons = add(work.sort_comparisons, 1);
    if (a.degree[u] != a.degree[v])
      return a.degree[u] > a.degree[v];
    return u < v;
  };
  std::sort(a.order.begin(), a.order.end(), before);
  a.result.color.assign(n, uncolored);
  for (const auto v : a.order) {
    const auto slots  = add(a.degree[v], 1);
    work.marker_slots = add(work.marker_slots, slots);
    std::vector<bool> used(slots, false);
    for (std::size_t j = 0; j < adj[v].size(); ++j) {
      admit(events, o.max_events);
      const auto w         = adj[v][j];
      work.neighbor_visits = add(work.neighbor_visits, 1);
      const auto cw        = a.result.color[w];
      const bool colored   = cw != uncolored;
      const bool marked    = colored && cw < used.size();
      if (marked) {
        work.marker_writes = add(work.marker_writes, 1);
        used[cw]           = true;
      }
      record(events, {ColoringAction::neighbor, v, w, colored ? cw : 0, colored, marked});
    }
    std::size_t c = 0;
    while (c < used.size()) {
      admit(events, o.max_events);
      work.color_probes  = add(work.color_probes, 1);
      const bool blocked = used[c];
      record(events, {ColoringAction::probe, v, 0, c, false, blocked});
      if (!blocked)
        break;
      ++c;
    }
    admit(events, o.max_events);
    work.assignments  = add(work.assignments, 1);
    a.result.color[v] = c;
    record(events, {ColoringAction::assign, v, 0, c, false, false});
  }
  if (n)
    a.result.num_colors = add(*std::max_element(a.result.color.begin(), a.result.color.end()), 1);
  return a;
}
} // namespace coloring_detail
ColoringResult greedy_coloring(
    std::size_t                                             n,
    const std::vector<std::pair<std::size_t, std::size_t>>& edges) {
  return coloring_detail::execute(n, edges, {0, 0, 0}, nullptr).result;
}
ColoringAnalysis greedy_coloring_analyze(
    std::size_t                                             n,
    const std::vector<std::pair<std::size_t, std::size_t>>& edges,
    ColoringOptions                                         options) {
  return coloring_detail::execute(n, edges, options, nullptr);
}
ColoringTrace greedy_coloring_trace(
    std::size_t                                             n,
    const std::vector<std::pair<std::size_t, std::size_t>>& edges,
    ColoringOptions                                         options) {
  ColoringTrace t;
  t.analysis = coloring_detail::execute(n, edges, options, &t.events);
  return t;
}
bool is_proper_coloring(
    std::size_t                                             n,
    const std::vector<std::pair<std::size_t, std::size_t>>& edges,
    const std::vector<std::size_t>&                         color) {
  if (color.size() != n)
    return false;
  for (auto [u, v] : edges) {
    if (u >= n || v >= n)
      return false;
    if (u == v)
      continue;
    if (color[u] == color[v])
      return false;
  }
  return true;
}
} // namespace datamunge::algorithms
