#pragma once
/// @file iterative_deepening_analysis.hpp
/// @brief Bounded complete-tree controls and actual-pass method diagrams for IDDFS.
#include <datamunge/algorithms/iterative_deepening.hpp>
#include <datamunge/plot/diagram.hpp>
#include <format>
#include <limits>
#include <stdexcept>
namespace datamunge::algorithms {
/// @brief Owned complete rooted b-ary directed tree with the last leaf as goal.
/// Vertices have breadth-first integer ids, root zero. Children retain increasing
/// id order. Depth zero is the singleton root/goal. No external data or RNG.
/// @par Example
/// @code
/// const auto t=dm::algorithms::deepening_tree(2,2);
/// std::print("{} {}\n",t.n,t.goal); // 7 6
/// @endcode
/// @see deepening_tree()
struct DeepeningTree {
  std::size_t n{0}; ///< Complete tree vertex count.
  std::vector<std::pair<std::size_t, std::size_t>>
              edges;    ///< Owned root-to-child arcs in increasing child-id order.
  std::size_t goal{0};  ///< Rightmost deepest leaf id, n-1 after successful construction.
  std::size_t depth{0}; ///< Complete tree edge depth.
};
/// @brief Construct a complete finite b-ary tree without truncating a requested level.
/// Complete count arithmetic and optional vertex cap precede any output allocation.
/// b=1 gives a chain; b=0 is rejected. O(n) time/owned storage; no shared state,
/// retained references or caller mutation. This deterministic control is not a
/// model of arbitrary finite-graph search work or elapsed runtime.
/// @param branching Positive child count per internal node.
/// @param depth Complete edge depth, zero for a singleton.
/// @param max_vertices Complete output count cap, default million; zero disables optional cap.
/// @return Owned full tree, root zero, goal last id.
/// @throws std::invalid_argument On zero branching.
/// @throws std::length_error On count overflow, enabled complete cap or exhausted capacity.
/// @throws std::bad_alloc On allocation failure.
/// @par Example
/// @code
/// const auto t=dm::algorithms::deepening_tree(2,3);
/// const auto r=dm::algorithms::iterative_deepening_dfs_run(t.n,t.edges,0,t.goal,t.depth);
/// std::print("{} {}\n",t.n,r.result.expanded); // 15 26
/// @endcode
/// @see DeepeningTree, iterative_deepening_dfs_run()
inline DeepeningTree
deepening_tree(std::size_t branching, std::size_t depth, std::size_t max_vertices = 1000000) {
  if (!branching)
    throw std::invalid_argument("IDDFS positive branching required");
  const auto  max   = std::numeric_limits<std::size_t>::max();
  std::size_t count = 1;
  if (branching == 1) {
    if (depth == max)
      throw std::length_error("IDDFS tree count");
    count = depth + 1;
  } else {
    std::size_t level = 1;
    for (std::size_t d = 0; d < depth; ++d) {
      if (level > max / branching)
        throw std::length_error("IDDFS tree level count");
      level *= branching;
      if (count > max - level)
        throw std::length_error("IDDFS tree total count");
      count += level;
      if (max_vertices && count > max_vertices)
        throw std::length_error("IDDFS complete tree cap");
    }
  }
  if (max_vertices && count > max_vertices)
    throw std::length_error("IDDFS complete tree cap");
  DeepeningTree out;
  out.n     = count;
  out.goal  = count - 1;
  out.depth = depth;
  out.edges.reserve(count - 1);
  for (std::size_t child = 1; child < count; ++child)
    out.edges.push_back({(child - 1) / branching, child});
  return out;
}
/// @brief Draw actual pass summaries without rerunning depth-limited searches.
/// One box per supplied pass shows limit, entries, cutoffs and found status.
/// Checks consecutive limits, basic scalar consistency, terminal-found placement
/// and the sum of entries against result.expanded; not a proof of an edited
/// record's graph origin. O(P+path length) work/label storage, no retained
/// references or caller mutation. Complete pass display quota precedes layout.
/// @param run Borrowed actual final result/pass summaries.
/// @param max_passes Complete displayed-pass cap, default 12; zero disables.
/// @return Owned fitted native Diagram with chronological pass arrows.
/// @throws std::invalid_argument On empty/inconsistent summaries or found/path shape.
/// @throws std::length_error On count overflow or exceeded nonzero display quota.
/// @throws std::bad_alloc On allocation failure.
/// @par Example
/// @code
/// const auto r=dm::algorithms::iterative_deepening_dfs_run(2,{{0,1}},0,1,1);
/// std::print("{}\n",dm::algorithms::deepening_pass_diagram(r).to_svg().empty()); // false
/// @endcode
/// @see DeepeningRun, iterative_deepening_dfs_trace()
inline graphics::Diagram deepening_pass_diagram(
    const DeepeningRun& run,
    std::size_t         max_passes = 12) {
  if (run.passes.empty())
    throw std::invalid_argument("IDDFS empty pass diagram");
  if (max_passes && run.passes.size() > max_passes)
    throw std::length_error("IDDFS complete pass display cap");
  std::size_t total = 0;
  const auto  max   = std::numeric_limits<std::size_t>::max();
  for (std::size_t i = 0; i < run.passes.size(); ++i) {
    const auto& p = run.passes[i];
    if (p.limit != i || !p.entered || !p.maximum_stack || p.cutoffs > p.entered ||
        p.cycle_skips > p.edges_examined || (p.found && i + 1 != run.passes.size()))
      throw std::invalid_argument("IDDFS inconsistent pass diagram");
    if (total > max - p.entered)
      throw std::length_error("IDDFS diagram entry count");
    total += p.entered;
  }
  if (total != run.result.expanded || run.passes.back().found != run.result.found ||
      (run.result.found ? run.result.path.empty() : !run.result.path.empty()))
    throw std::invalid_argument("IDDFS inconsistent result diagram");
  graphics::Diagram d;
  for (std::size_t i = 0; i < run.passes.size(); ++i) {
    const auto& p = run.passes[i];
    d.node(
        std::to_string(i),
        {6. * static_cast<double>(i), 0},
        std::format(
            "limit {}\nentries {}\ncutoffs {}\n{}",
            p.limit,
            p.entered,
            p.cutoffs,
            p.found ? "goal found" : "no goal"));
    if (i)
      d.edge(std::to_string(i - 1), std::to_string(i));
  }
  return d;
}
} // namespace datamunge::algorithms
