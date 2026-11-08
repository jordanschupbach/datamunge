#pragma once
/// @file brute_force_search.hpp
/// @brief Directed shortest simple-path backtracking with actual incumbent-bound diagnostics.
#include <datamunge/algorithms/basic_path_result.hpp>
#include <cstddef>
#include <utility>
#include <vector>
namespace datamunge::algorithms {
/// @brief Optional complete-search visit budget and retained-path choice for diagnostics.
/// Zero max_visits is unlimited. A positive budget throws before entering an
/// excess prefix; no partial optimum is returned. It does not avoid adjacency
/// allocation. record_paths=false omits per-visit path copies while retaining
/// visit ids, parents, vertices, depths, decisions and counters.
/// @par Example
/// @code
/// const dm::algorithms::BruteForceTraceOptions options{20,false};
/// std::print("{} {}\n",options.max_visits,options.record_paths); // 20 false
/// @endcode
/// @see brute_force_search_trace()
struct BruteForceTraceOptions {
    std::size_t max_visits{0}; ///< Maximum entered path prefixes, or zero for no caller limit.
    bool record_paths{true}; ///< Copy each actual current path; false retains metadata only.
};

/// @brief One actual entered simple-path prefix and its local decision/probe counts.
/// Entry fields describe the incumbent before and after this visit's own goal
/// decision; descendant improvements do not rewrite those historical values.
/// Local edge/cycle counts accumulate while the visit remains active. Records
/// are in DFS preorder, and parent ids identify the actual prefix search tree.
/// @par Example
/// @code
/// const auto trace=dm::algorithms::brute_force_search_trace(3,{{0,1},{1,2}},0,2);
/// std::print("{} {}\n",trace.steps.back().depth,trace.steps.back().improved); // 2 true
/// @endcode
/// @see BruteForceTrace
struct BruteForceStep {
    std::size_t id{0}; ///< Preorder visit index, matching the owning trace's vector index.
    std::size_t parent{0}; ///< Parent visit index; root zero uses parent zero.
    std::size_t vertex{0}; ///< Last vertex of this actual current path.
    std::size_t depth{0}; ///< Number of edges in the current path, starting at zero.
    std::vector<std::size_t> path; ///< Owned actual prefix, or empty when path recording was disabled.
    std::size_t incumbent_vertices_before{0}; ///< Incumbent vertex count at entry; zero means none yet.
    std::size_t incumbent_vertices_after{0}; ///< Count after this visit's own goal decision; zero means none yet.
    std::size_t edges_examined{0}; ///< Adjacency entries inspected while this visit is active.
    std::size_t cycle_rejections{0}; ///< Those entries targeting a vertex already on the current path.
    bool goal{false}; ///< This visit reaches the requested goal and has no outgoing search.
    bool improved{false}; ///< The goal visit replaces the incumbent with a strictly shorter path or the first path.
    bool pruned{false}; ///< Non-goal prefix has at least the incumbent's vertex count; no successors inspected.
};

/// @brief Owned shortest-path result and actual DFS incumbent-bound diagnostics.
/// Only vertices on the current path are marked, so the search can revisit a
/// vertex along another prefix. Duplicate input arcs can repeat subtrees. Ordinary
/// and traced calls share one explicit-stack kernel and preserve legacy tie order.
/// Retaining full prefixes costs their total length; omitting copies retains O(V)
/// records for V visits, in addition to the graph, live stack and final result.
/// @par Example
/// @code
/// const auto trace=dm::algorithms::brute_force_search_trace(3,{{0,1},{1,2}},0,2);
/// std::print("{} {}\n",trace.result.expanded,trace.edges_examined); // 3 2
/// @endcode
/// @see brute_force_search_trace(), BasicPathResult
struct BruteForceTrace {
    BasicPathResult result; ///< Same selected shortest path, visit count and success as the ordinary call.
    std::vector<BruteForceStep> steps; ///< Actual preorder entered-prefix records; one per result.expanded visit.
    std::size_t edges_examined{0}; ///< All adjacency entries actually inspected, including loops/duplicates.
    std::size_t cycle_rejections{0}; ///< Entries rejected because their endpoint was on the live path.
    std::size_t bound_prunes{0}; ///< Non-goal entered prefixes cut off by the incumbent vertex-count bound.
    std::size_t goal_visits{0}; ///< Entered goal prefixes, including non-improving equal/longer paths.
    std::size_t improvements{0}; ///< Strict incumbent replacements, including the first found path.
    std::size_t max_depth{0}; ///< Maximum entered path edge count; at most n-1.
    bool paths_recorded{true}; ///< Whether each record owns its actual entered prefix.
};

/// @brief Find a minimum-edge directed path by DFS over simple prefixes with an incumbent bound.
/// Preserves the original signature, result layout, input adjacency order and
/// strict-improvement ties. Each input pair is one directed arc; loops and parallel
/// arcs are allowed. Mark only vertices on the live path and unmark on backtracking.
/// Goals are terminal. After finding an incumbent of L vertices, a non-goal prefix
/// with at least L vertices is terminal too. Thus this does not enumerate every
/// simple goal path after finding an incumbent. Equal shortest alternatives retain
/// the first discovered path. start==goal returns [start] with one visit; failure
/// returns an empty path. Validate all terminals/edges before graph allocation.
/// Uses an explicit owned stack rather than native recursion, without trace records
/// or input/shared-state mutation. With V visits, A examined entries and I total
/// copied incumbent vertex ids, work is O(n+m+V+A+I), live storage O(n+m).
/// The prefix search can have factorially many visits on simple dense graphs;
/// parallel arcs can repeat even more equivalent prefixes. BFS is preferable when
/// only a minimum-edge path is required.
/// @param n Vertex count; both terminals must be in [0,n), so n=0 is invalid.
/// @param edges Directed arcs, inspected in their per-source input order.
/// @param start Initial vertex.
/// @param goal Requested goal vertex.
/// @return Owned minimum-edge path, actual entered-prefix count and success flag.
/// @throws std::invalid_argument If a terminal or edge endpoint is out of range.
/// @throws std::overflow_error Before logical counters wrap; allocation failures propagate.
/// @par Example
/// @code
/// const auto result=dm::algorithms::brute_force_search(3,{{0,1},{1,2}},0,2);
/// std::print("{}\n",result.path); // [0, 1, 2]
/// @endcode
/// @par References
/// - NIST DADS, exhaustive search, https://xlinux.nist.gov/dads/HTML/exhaustiveSearch.html,
///   terminology; this custom path-prefix search additionally uses the stated length bound.
/// - A. H. Land and A. G. Doig (1960), An Automatic Method of Solving Discrete
///   Programming Problems, Econometrica 28(3), 497-520, doi:10.2307/1910129,
///   general branch-and-bound background, not this particular graph implementation.
/// @see brute_force_search_trace(), breadth_first_search()
BasicPathResult brute_force_search(std::size_t n,const std::vector<std::pair<std::size_t,std::size_t>>& edges,
                                   std::size_t start,std::size_t goal);

/// @brief Record the ordinary path-prefix search's actual incumbents, decisions and work.
/// Uses the same kernel, directed arcs, validation, visit order and tie rule as
/// brute_force_search(). Adds O(V) record storage, plus O(total retained prefix
/// lengths) when record_paths=true (default); the ordinary call records nothing.
/// Positive max_visits stops before an excess visit and throws without returning
/// a partial optimum. Input graph allocation still precedes search. Zero means
/// unlimited; recording paths can be disabled independently of the budget.
/// @param n Vertex count, with both terminals in [0,n).
/// @param edges Directed arcs in original per-source order, including loops/duplicates.
/// @param start Initial vertex.
/// @param goal Goal vertex.
/// @param options Visit limit and full-prefix capture; defaults are unlimited and true.
/// @return Owned result and actual preorder visit records/counters.
/// @throws std::invalid_argument If a terminal or edge endpoint is invalid.
/// @throws std::runtime_error If a positive complete-search visit budget is exhausted.
/// @throws std::overflow_error Before counter wrap; allocation failures propagate.
/// @par Example
/// @code
/// const auto trace=dm::algorithms::brute_force_search_trace(3,{{0,1},{1,2}},0,2,{10,false});
/// std::print("{} {}\n",trace.steps.size(),trace.steps.back().path.empty()); // 3 true
/// @endcode
/// @par References
/// - NIST DADS, https://xlinux.nist.gov/dads/HTML/exhaustiveSearch.html;
///   diagnostics of this library's actual incumbent-bound DFS, not an independent replay.
/// @see BruteForceTrace, BruteForceTraceOptions, brute_force_search()
BruteForceTrace brute_force_search_trace(std::size_t n,const std::vector<std::pair<std::size_t,std::size_t>>& edges,
                                         std::size_t start,std::size_t goal,BruteForceTraceOptions options={});

/// @brief Build every directed non-loop arc between n labelled vertices.
/// Rows are represented as pairs in lexicographic (source,destination) order.
/// Zero/one vertex produces no arcs. O(n^2) work and O(n*(n-1)) owned output
/// storage, without randomness or shared/input mutation. The count is checked
/// before multiplication/allocation; large valid output can still fail to allocate.
/// @param n Number of vertices, labelled 0..n-1.
/// @return Owned unique ordered arc pairs; both orientations are present for each distinct pair.
/// @throws std::overflow_error If n*(n-1) cannot fit size_t.
/// @throws std::length_error If the arc count exceeds vector capacity; allocation failures propagate.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::complete_directed_graph_edges(2)); // [(0, 1), (1, 0)]
/// @endcode
/// @see brute_force_search_trace(), complete_graph_edges()
std::vector<std::pair<std::size_t,std::size_t>> complete_directed_graph_edges(std::size_t n);
} // namespace datamunge::algorithms
