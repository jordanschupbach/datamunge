#pragma once
/// @file hopcroft_karp_analysis.hpp
/// @brief Public bipartite matching fixtures, bounded long-path graphs and native matching views.
#include <algorithm>
#include <datamunge/algorithms/hopcroft_karp.hpp>
#include <datamunge/plot/plot.hpp>
#include <limits>
#include <stdexcept>
#include <string>
namespace datamunge::algorithms {
/// @brief Owned bipartite graph descriptor with separate zero-based side labels.
/// Repeated edges and order are retained; algorithms validate endpoints on use.
/// Default construction gives the empty graph. No references or shared state.
/// @par Example
/// @code
/// const dm::algorithms::HopcroftKarpGraph g;
/// std::print("{} {}\n",g.n_left,g.n_right); // 0 0
/// @endcode
struct HopcroftKarpGraph {
  std::size_t n_left{0};               ///< Complete left-side size, including isolated vertices.
  std::size_t n_right{0};              ///< Complete right-side size, including isolated vertices.
  std::vector<HopcroftKarpEdge> edges; ///< Owned ordered left/right endpoint entries.
};
/// @brief Return the original five-applicant/four-job eligibility fixture.
/// The eight supplied edges have a maximum matching of size4. Returned graph
/// owns fixed bounded storage and retains original endpoint order.
/// @return Owned5-by4 public graph.
/// @throws std::bad_alloc On allocation failure.
/// @par Example
/// @code
/// const auto g=dm::algorithms::hopcroft_karp_example_graph();
/// std::print("{} {} {}\n",g.n_left,g.n_right,g.edges.size()); // 5 4 8
/// @endcode
/// @see hopcroft_karp()
inline HopcroftKarpGraph hopcroft_karp_example_graph() {
  return {5, 4, {{0, 0}, {0, 1}, {1, 0}, {2, 1}, {2, 2}, {3, 2}, {3, 3}, {4, 3}}};
}
/// @brief Return the six-by-six graph exposing the old missing shortest-path cutoff.
/// The second phase's shortest path has3 forward pairs; the older unconstrained
/// DFS could choose4. Both reach cardinality6, but only the corrected execution
/// implements the stated shortest blocking phases. Fixed owned public input.
/// @return Owned6-by6 regression graph in its original witnessing edge order.
/// @throws std::bad_alloc On allocation failure.
/// @par Example
/// @code
/// const auto g=dm::algorithms::hopcroft_karp_cutoff_graph();
/// std::print("{}\n",g.edges.size()); // 15
/// @endcode
/// @see hopcroft_karp_trace()
inline HopcroftKarpGraph hopcroft_karp_cutoff_graph() {
  return {
      6,
      6,
      {{0, 4},
       {1, 0},
       {1, 1},
       {1, 4},
       {1, 5},
       {2, 0},
       {2, 2},
       {2, 3},
       {3, 2},
       {3, 4},
       {4, 3},
       {5, 0},
       {5, 1},
       {5, 3},
       {5, 5}}};
}
/// @brief Build an ordered graph requiring a long alternating second-phase path.
/// For n>=2, edges(i,i),(i,i+1) for i<n-1 precede(n-1,0). Greedy first-phase
/// choices match n-1 pairs; the next shortest augmentation has n forward pairs.
/// n=1 is a single edge; n=0 is empty. O(n) work/owned storage, no randomness.
/// Full2n vertex/count/container admission precedes edge allocation; no truncation.
/// @param side_size Complete size n of each side, zero allowed.
/// @param max_vertices Complete combined2n vertex cap, default1m; zero disables.
/// @return Owned graph with2n-1 supplied edges when n>0, otherwise empty.
/// @throws std::length_error On combined vertex cap or count/container limits.
/// @throws std::bad_alloc On allocation failure.
/// @par Example
/// @code
/// const auto g=dm::algorithms::hopcroft_karp_chain_graph(3);
/// std::print("{} {}\n",g.n_left,g.edges.size()); // 3 5
/// @endcode
/// @see hopcroft_karp_run()
inline HopcroftKarpGraph hopcroft_karp_chain_graph(
    std::size_t side_size,
    std::size_t max_vertices = 1000000) {
  if (side_size > std::numeric_limits<std::size_t>::max() / 2)
    throw std::length_error("matching chain vertex overflow");
  const auto total = 2 * side_size;
  if (max_vertices && total > max_vertices)
    throw std::length_error("matching chain vertex cap");
  HopcroftKarpGraph g{side_size, side_size, {}};
  if (side_size == 0)
    return g;
  const auto count = total - 1;
  if (count > g.edges.max_size())
    throw std::length_error("matching chain edge count");
  g.edges.reserve(count);
  for (std::size_t u = 0; u + 1 < side_size; ++u) {
    g.edges.emplace_back(u, u);
    g.edges.emplace_back(u, u + 1);
  }
  g.edges.emplace_back(side_size - 1, 0);
  return g;
}
/// @brief Plot eligible edges and any valid matching on two labelled-by-axis vertex columns.
/// Grey lines show supplied eligibility, blue lines matched pairs; left/right
/// vertex indices use y coordinates on x=0/1. Duplicates are overdrawn. Validates
/// all endpoint ranges, matching dimensions/inverses/edge presence/cardinality
/// before display policies and plot arrays. Matching need not be maximum.
/// Validation O(V+E+V*E), no allocation; plot O(V+E) owned series. Display caps
/// count complete vertices/edge entries, not bytes; zero disables each. Empty
/// sides are valid. No input mutation or retained input references.
/// @param graph Borrowed complete descriptor, arbitrary edge order/duplicates.
/// @param matching Borrowed same-size valid matching, including unmatched sentinels.
/// @param title Owned plot title, default Bipartite matching.
/// @param max_vertices Complete display-vertex cap, default100; zero disables.
/// @param max_edges Complete display-edge cap, default2000; zero disables.
/// @return Owned native RPlot with explicit side/index axes and fixed count-based limits.
/// @throws std::invalid_argument On invalid graph/matching before display policy checks.
/// @throws std::length_error On combined count/display/container limits.
/// @throws std::bad_alloc On plot allocation failure.
/// @par Example
/// @code
/// const auto g=dm::algorithms::hopcroft_karp_example_graph();
/// const auto m=dm::algorithms::hopcroft_karp(g.n_left,g.n_right,g.edges);
/// const auto p=dm::algorithms::hopcroft_karp_graph_plot(g,m);
/// std::print("{}\n",m.size); // 4
/// @endcode
/// @see hopcroft_karp_example_graph(), HopcroftKarpGraph
inline graphics::RPlot hopcroft_karp_graph_plot(
    const HopcroftKarpGraph&       graph,
    const BipartiteMatchingResult& matching,
    std::string                    title        = "Bipartite matching",
    std::size_t                    max_vertices = 100,
    std::size_t                    max_edges    = 2000) {
  for (auto [u, v] : graph.edges)
    if (u >= graph.n_left || v >= graph.n_right)
      throw std::invalid_argument("matching plot endpoints");
  if (matching.match_left.size() != graph.n_left || matching.match_right.size() != graph.n_right)
    throw std::invalid_argument("matching plot dimensions");
  std::size_t pairs = 0;
  for (std::size_t u = 0; u < graph.n_left; ++u) {
    const auto v = matching.match_left[u];
    if (v == kBipartiteUnmatched)
      continue;
    if (v >= graph.n_right || matching.match_right[v] != u ||
        std::find(graph.edges.begin(), graph.edges.end(), HopcroftKarpEdge{u, v}) ==
            graph.edges.end())
      throw std::invalid_argument("matching plot left consistency");
    ++pairs;
  }
  for (std::size_t v = 0; v < graph.n_right; ++v) {
    const auto u = matching.match_right[v];
    if (u != kBipartiteUnmatched && (u >= graph.n_left || matching.match_left[u] != v))
      throw std::invalid_argument("matching plot right consistency");
  }
  if (pairs != matching.size)
    throw std::invalid_argument("matching plot cardinality");
  if (graph.n_right > std::numeric_limits<std::size_t>::max() - graph.n_left)
    throw std::length_error("matching plot vertex overflow");
  const auto total = graph.n_left + graph.n_right;
  if ((max_vertices && total > max_vertices) || (max_edges && graph.edges.size() > max_edges))
    throw std::length_error("matching plot display cap");
  auto plot  = graphics::RPlot::create();
  bool first = true;
  for (auto [u, v] : graph.edges) {
    plot.line({0., 1.}, {double(u), double(v)}, first ? "eligible" : "", {203, 213, 225}, 1.);
    first = false;
  }
  first = true;
  for (std::size_t u = 0; u < graph.n_left; ++u)
    if (matching.match_left[u] != kBipartiteUnmatched) {
      plot.line(
          {0., 1.},
          {double(u), double(matching.match_left[u])},
          first ? "matched" : "",
          {37, 99, 235},
          4.);
      first = false;
    }
  std::vector<double> lx(graph.n_left, 0.), ly, rx(graph.n_right, 1.), ry;
  for (std::size_t u = 0; u < graph.n_left; ++u)
    ly.push_back(u);
  for (std::size_t v = 0; v < graph.n_right; ++v)
    ry.push_back(v);
  if (!lx.empty())
    plot.points(lx, ly, "left", {30, 41, 59}, 8.);
  if (!rx.empty())
    plot.points(rx, ry, "right", {220, 38, 38}, 8.);
  plot.title(std::move(title))
      .x_label("side: left=0, right=1")
      .y_label("vertex index")
      .x_limits(-.3, 1.3)
      .y_limits(-1., double(std::max(graph.n_left, graph.n_right)) + 2.);
  return plot;
}
} // namespace datamunge::algorithms
