#pragma once
/// @file johnson_analysis.hpp
/// @brief Original Johnson fixture, bounded path controls and an actual reweighting diagram.
#include <algorithm>
#include <cmath>
#include <datamunge/algorithms/johnson.hpp>
#include <datamunge/plot/diagram.hpp>
#include <format>
#include <stdexcept>
namespace datamunge::algorithms {
/// @brief Return the original three directed edges in historical input order.
/// Owned fixed finite data, no randomness, references or shared mutation.
/// @return Edges 0->1 weight1,1->2 weight-2,0->2 weight4.
/// @throws std::bad_alloc On allocation failure.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::johnson_demo_edges().size()); // 3
/// @endcode
/// @see johnson_analyze()
inline std::vector<WeightedEdge> johnson_demo_edges() {
  return {{0, 1, 1}, {1, 2, -2}, {0, 2, 4}};
}
/// @brief Generate a directed negative-unit path in ascending or reversed edge order.
/// Vertices0..n-1, edge i->i+1 weight-1; no negative cycles. Ascending order needs
/// two potential passes for n>=2; reversed order needs n passes. Both have the
/// same all-pairs answers and n(n-1)/2 Dijkstra edge checks. O(n) owned storage;
/// complete edge count admitted before allocation; n<=1 produces no edges.
/// @param n Vertex count including empty/singleton.
/// @param reversed False(default) ascending origin, true reverse input order.
/// @param max_edges Complete edge ceiling, default million, zero disables.
/// @return Owned complete edge list without RNG or shared state.
/// @throws std::length_error On enabled ceiling or vector capacity overflow.
/// @throws std::bad_alloc On allocation failure.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::johnson_analyze(4,dm::algorithms::johnson_path_edges(4,true)).work.potential_passes); // 4
/// @endcode
/// @see johnson_trace()
inline std::vector<WeightedEdge>
johnson_path_edges(std::size_t n, bool reversed = false, std::size_t max_edges = 1000000) {
  const auto                m = n ? n - 1 : 0;
  std::vector<WeightedEdge> edges;
  if ((max_edges && m > max_edges) || m > edges.max_size())
    throw std::length_error("Johnson complete path edge ceiling");
  edges.reserve(m);
  for (std::size_t i = 0; i < m; ++i)
    edges.emplace_back(i, i + 1, -1);
  if (reversed)
    std::reverse(edges.begin(), edges.end());
  return edges;
}
/// @brief Diagram one actual reweight event from the producing trace, without solver replay.
/// Shows original endpoints/weight, terminal h values and recorded reduced weight.
/// Checks local identity/finite shape only, not provenance of caller-edited histories.
/// Constant primitive count, no caller mutation or retained references.
/// @param trace Borrowed same-execution trace with a completed reweight phase.
/// @param edge Original edge index, default0.
/// @return Owned fitted four-node native concept diagram.
/// @throws std::out_of_range If the original edge is absent.
/// @throws std::invalid_argument On absent reweight event, bad endpoint/potential shape or nonfinite shown values.
/// @throws std::bad_alloc On graphics allocation failure.
/// @par Example
/// @code
/// const auto t=dm::algorithms::johnson_trace(3,dm::algorithms::johnson_demo_edges());
/// std::print("{}\n",dm::algorithms::johnson_reweight_diagram(t,1).to_svg().empty()); // false
/// @endcode
/// @see JohnsonEvent, johnson_trace()
inline graphics::Diagram johnson_reweight_diagram(const JohnsonTrace& trace, std::size_t edge = 0) {
  if (edge >= trace.edges.size())
    throw std::out_of_range("Johnson diagram edge");
  const auto [u, v, w] = trace.edges[edge];
  const auto& h        = trace.analysis.potentials;
  const auto  it =
      std::find_if(trace.events.begin(), trace.events.end(), [&](const JohnsonEvent& e) {
        return e.action == JohnsonAction::Reweight && e.edge == edge;
      });
  if (u >= trace.vertex_count || v >= trace.vertex_count || h.size() != trace.vertex_count ||
      it == trace.events.end() || it->u != u || it->v != v || !std::isfinite(w) ||
      !std::isfinite(h[u]) || !std::isfinite(h[v]) || !std::isfinite(it->after))
    throw std::invalid_argument("Johnson diagram actual shape");
  graphics::Diagram          d;
  graphics::DiagramNodeStyle style;
  style.width = 4.2;
  d.node("edge", {0, 0}, std::format("Edge {} -> {}\nw = {:.6g}", u, v, w), style)
      .node("origin", {6, 0}, std::format("Origin potential\nh[{}] = {:.6g}", u, h[u]), style)
      .node("target", {12, 0}, std::format("Target potential\nh[{}] = {:.6g}", v, h[v]), style)
      .node("reduced", {18, 0}, std::format("Recorded w'\n{:.6g}", it->after), style)
      .edge("edge", "origin")
      .edge("origin", "target")
      .edge("target", "reduced");
  return d;
}
} // namespace datamunge::algorithms
