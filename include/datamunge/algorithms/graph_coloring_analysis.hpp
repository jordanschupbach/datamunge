#pragma once
/// @file graph_coloring_analysis.hpp
/// @brief Checked tiny graph families, crown witnesses and native coloring diagrams.
#include <algorithm>
#include <cmath>
#include <datamunge/algorithms/graph_coloring.hpp>
#include <datamunge/plot/diagram.hpp>
#include <format>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <string>

namespace datamunge::algorithms {
/// @brief Owned undirected graph fixture with explicit vertex count, including isolates.
/// Edges retain their order and may be edited; family builders produce simple
/// loopless graphs. No coloring or optimality certificate is implied by storage.
/// @par Example
/// @code
/// const auto g=dm::algorithms::coloring_cycle(5);
/// std::print("{} {}\n",g.vertices,g.edges.size()); // 5 5
/// @endcode
/// @see coloring_cycle(), coloring_complete(), coloring_crown()
struct ColoringGraph {
  std::size_t vertices{0}; ///< Complete vertex count, including isolates.
  std::vector<std::pair<std::size_t, std::size_t>> edges; ///< Owned undirected edge records.
};
/// @brief Checked geometry for report family builders.
namespace coloring_analysis_detail {
/// @brief Checked size product for complete family generation.
/// @param a First nonnegative count.
/// @param b Second nonnegative count.
/// @return Exact size_t product.
/// @throws std::length_error On overflow.
inline std::size_t product(std::size_t a, std::size_t b) {
  if (a && b > std::numeric_limits<std::size_t>::max() / a)
    throw std::length_error("coloring family size overflow");
  return a * b;
}
/// @brief Enforce a known complete edge count before allocation.
/// @param count Required whole records.
/// @param cap Optional complete edge cap; zero disables it.
/// @throws std::length_error On policy or container limit.
inline void edges(std::size_t count, std::size_t cap) {
  if ((cap && count > cap) || count > std::vector<std::pair<std::size_t, std::size_t>>{}.max_size())
    throw std::length_error("coloring family edge limit");
}
} // namespace coloring_analysis_detail
/// @brief Generate the simple cycle 0–1–...–(n−1)–0.
/// No randomness or input references. O(n) output/work. n must be at least three.
/// @param n Cycle size.
/// @param max_edges Complete known edge cap, default one million; zero disables it.
/// @return Owned n-vertex/n-edge fixture.
/// @throws std::invalid_argument If n<3.
/// @throws std::length_error On complete edge cap/container limit.
/// @throws std::bad_alloc On allocation failure.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::coloring_cycle(5).edges.size()); // 5
/// @endcode
/// @see coloring_complete()
inline ColoringGraph coloring_cycle(std::size_t n, std::size_t max_edges = 1000000) {
  if (n < 3)
    throw std::invalid_argument("cycle needs at least three vertices");
  coloring_analysis_detail::edges(n, max_edges);
  ColoringGraph g;
  g.vertices = n;
  g.edges.reserve(n);
  for (std::size_t i = 0; i < n; ++i)
    g.edges.emplace_back(i, (i + 1) % n);
  return g;
}
/// @brief Generate each unordered edge of the complete loopless graph once.
/// Computes n(n−1)/2 without overflowing an intermediate when the result fits.
/// Empty/singleton graphs have no edges. O(n²) output/work for n>=2.
/// @param n Vertex count.
/// @param max_edges Complete known edge cap, default one million; zero disables it.
/// @return Owned complete graph in lexicographic edge order.
/// @throws std::length_error On size overflow, complete edge cap or container limit.
/// @throws std::bad_alloc On allocation failure.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::coloring_complete(4).edges.size()); // 6
/// @endcode
/// @see coloring_cycle()
inline ColoringGraph coloring_complete(std::size_t n, std::size_t max_edges = 1000000) {
  const auto count =
      n < 2 ? 0 : coloring_analysis_detail::product(n % 2 ? n : n / 2, n % 2 ? (n - 1) / 2 : n - 1);
  coloring_analysis_detail::edges(count, max_edges);
  ColoringGraph g;
  g.vertices = n;
  g.edges.reserve(count);
  for (std::size_t u = 0; u < n; ++u)
    for (std::size_t v = u + 1; v < n; ++v)
      g.edges.emplace_back(u, v);
  return g;
}
/// @brief Generate K(k,k) minus its matched pairs, in paired or grouped vertex numbering.
/// Paired IDs are a_i=2i,b_i=2i+1; grouped IDs a_i=i,b_i=k+i.
/// All degrees are k−1. For k>=2 the graph has an edge and is bipartite, hence
/// chromatic number two. Paired ID ties make this native greedy use k colors;
/// grouped ID ties use two. The underlying abstract graphs are isomorphic.
/// O(k²) output/work, no random state or caller references.
/// @param k Number of matched pairs, at least two.
/// @param paired Interleave pair IDs if true (default), group sides if false.
/// @param max_edges Complete k(k−1) edge cap, default one million; zero disables it.
/// @return Owned 2k-vertex simple crown fixture.
/// @throws std::invalid_argument If k<2.
/// @throws std::length_error On size overflow, edge cap or container limit.
/// @throws std::bad_alloc On allocation failure.
/// @par Example
/// @code
/// const auto g=dm::algorithms::coloring_crown(4);
/// std::print("{}\n",dm::algorithms::greedy_coloring(g.vertices,g.edges).num_colors); // 4
/// @endcode
/// @see coloring_crown_witness()
inline ColoringGraph
coloring_crown(std::size_t k, bool paired = true, std::size_t max_edges = 1000000) {
  if (k < 2)
    throw std::invalid_argument("crown needs at least two pairs");
  const auto n = coloring_analysis_detail::product(2, k),
             m = coloring_analysis_detail::product(k, k - 1);
  coloring_analysis_detail::edges(m, max_edges);
  ColoringGraph g;
  g.vertices = n;
  g.edges.reserve(m);
  for (std::size_t i = 0; i < k; ++i)
    for (std::size_t j = 0; j < k; ++j)
      if (i != j)
        g.edges.emplace_back(paired ? 2 * i : i, paired ? 2 * j + 1 : k + j);
  return g;
}
/// @brief Return the explicit two-color bipartition for a generated crown numbering.
/// This is a constructive feasible witness, not a general exact-coloring solver.
/// Labels own their storage; both colors occur for every permitted k.
/// @param k Pair count, at least two.
/// @param paired Same numbering choice as coloring_crown(), default true.
/// @return Owned 2k labels with every a-side zero and b-side one; num_colors=2.
/// @throws std::invalid_argument If k<2.
/// @throws std::length_error On 2k overflow/container limit.
/// @throws std::bad_alloc On allocation failure.
/// @par Example
/// @code
/// const auto c=dm::algorithms::coloring_crown_witness(3);
/// std::print("{} {}\n",c.color[0],c.color[1]); // 0 1
/// @endcode
/// @see coloring_crown()
inline ColoringResult coloring_crown_witness(std::size_t k, bool paired = true) {
  if (k < 2)
    throw std::invalid_argument("crown needs at least two pairs");
  const auto     n = coloring_analysis_detail::product(2, k);
  ColoringResult r;
  r.color.resize(n);
  r.num_colors = 2;
  for (std::size_t i = 0; i < k; ++i) {
    r.color[paired ? 2 * i : i]         = 0;
    r.color[paired ? 2 * i + 1 : k + i] = 1;
  }
  return r;
}
/// @brief Draw a small supplied coloring on an equal-scale circular native graph diagram.
/// Validates endpoints and label vector length; loops are omitted to match the
/// library projection. Parallel records are drawn separately on coincident links.
/// Does not certify properness or optimality. Node labels include vertex ID and
/// actual color number, so palette reuse cannot hide distinct numeric labels.
/// O(n+edges) retained geometry plus diagram node-lookup costs. Empty input gives
/// a text-only diagram; at most 32 vertices are permitted for legibility.
/// @param graph Borrowed graph, never mutated or retained.
/// @param coloring Borrowed labels, one per vertex; arbitrary size_t labels allowed.
/// @param title Owned display title, default generic.
/// @return Owned editable native diagram with actual supplied labels.
/// @throws std::invalid_argument On endpoints, label length, more than 32 vertices,
/// or diagram geometry rejected after unusually wide labels expand adjacent nodes.
/// @throws std::bad_alloc On allocation failure.
/// @throws std::length_error On diagram/string/container limits.
/// @par Example
/// @code
/// const auto g=dm::algorithms::coloring_cycle(5);
/// dm::algorithms::coloring_diagram(g,dm::algorithms::greedy_coloring(g.vertices,g.edges)).save_svg("build/coloring.svg");
/// @endcode
/// @see greedy_coloring_trace()
inline graphics::Diagram coloring_diagram(
    const ColoringGraph&  graph,
    const ColoringResult& coloring,
    std::string           title = "Actual supplied graph coloring") {
  if (graph.vertices > 32 || coloring.color.size() != graph.vertices)
    throw std::invalid_argument("diagram needs at most 32 matching labels");
  for (auto [u, v] : graph.edges)
    if (u >= graph.vertices || v >= graph.vertices)
      throw std::invalid_argument("diagram endpoint out of range");
  graphics::Diagram d(80, 1050);
  if (!graph.vertices) {
    d.text({0, 0}, title + "\nEmpty graph: zero vertices", 24);
    return d;
  }
  const graphics::RGB palette[] = {
      {191, 219, 254},
      {254, 202, 202},
      {187, 247, 208},
      {254, 240, 138},
      {233, 213, 255},
      {254, 215, 170}};
  const double               radius = std::max(3., .42 * double(graph.vertices));
  graphics::DiagramNodeStyle style;
  style.circle  = true;
  style.width     = 1.15;
  style.height    = 1.15;
  style.font_size = 20;
  for (std::size_t v = 0; v < graph.vertices; ++v) {
    const double angle = std::numbers::pi / 2 - 2 * std::numbers::pi * double(v) / graph.vertices;
    style.fill         = palette[coloring.color[v] % 6];
    d.node(
        std::to_string(v),
        {radius * std::cos(angle), radius * std::sin(angle)},
        std::format("v{}\nc{}", v, coloring.color[v]),
        style);
  }
  for (auto [u, v] : graph.edges)
    if (u != v)
      d.link(std::to_string(u), std::to_string(v));
  d.text({0, radius + 1.3}, std::move(title), 25);
  return d;
}
} // namespace datamunge::algorithms
