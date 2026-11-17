#pragma once

/// @file max_clique.hpp
/// @brief Canonical max clique API, producing histories and checked native contracts.

#include <cstddef>
#include <vector>
namespace datamunge::algorithms {
/// @brief Actual colouring-bound DFS work on the canonical simple undirected model.
/// @details Counts include actual adjacency tests, not a timing estimate.
/// @par Example
/// @code
/// auto a=dm::algorithms::max_clique_analyze(1,{{}});
/// std::print("{}\n",a.work.nodes); // 2
/// @endcode
/// @see max_clique_dyn, max_clique_analyze, max_clique_trace
/// @par Reference
/// Konc and Janezic (2007), An improved branch and bound algorithm for the maximum clique problem, MATCH58,569–590, sections2.2–2.3 describe the basic search/colouring. Native fixed-order search omits their later Dyn refinements.
struct MaxCliqueWork {
  std::size_t nodes{0};            ///< Actual entered search frames, including leaves.
  std::size_t colour_checks{0};    ///< Actual candidate/class-member adjacency tests.
  std::size_t candidate_checks{0}; ///< Actual child-intersection adjacency tests.
  std::size_t branches{0};         ///< Actual chosen vertices scheduled as child frames.
  std::size_t prunes{0};           ///< Actual colour-bound frame cutoffs.
  std::size_t improvements{0};     ///< Actual strict best-clique replacements at leaves.
  std::size_t max_depth{0};        ///< Actual entered frame depth, root one.
};

/// @brief Actual best clique, completion flag and certified bound under an optional node budget.
/// @details Complete=true means the full bounded DFS exhausted or pruned all frames. Otherwise clique is a valid lower-bound witness and upper_bound is the actual root greedy-colour bound. No optimality claim is made for incomplete search.
/// @par Example
/// @code
/// auto a=dm::algorithms::max_clique_analyze(1,{{}},1);
/// std::print("{} {}\n",a.complete,a.upper_bound); // false 1
/// @endcode
/// @see max_clique_dyn, max_clique_analyze, max_clique_trace
/// @par Reference
/// Konc and Janezic (2007), An improved branch and bound algorithm for the maximum clique problem, MATCH58,569–590, sections2.2–2.3 describe the basic search/colouring. Native fixed-order search omits their later Dyn refinements.
struct MaxCliqueAnalysis {
  std::vector<int>
       clique; ///< Actual best found clique, sorted ascending; ties keep the first strict maximum.
  bool complete{
      true}; ///< Whether the exact search completed rather than hitting its optional node cap.
  std::size_t   upper_bound{0}; ///< Exact size if complete, otherwise actual root colour count.
  std::size_t   edges{0};       ///< Distinct non-self undirected edges after symmetric union.
  MaxCliqueWork work;           ///< Actual producing operations.
};

/// @brief One actual entered search frame after its greedy colouring.
/// @details Vectors are owned snapshots of actual active clique, candidate order, colour-sorted order and corresponding one-based bounds. Leaf frames have empty candidate/order/colour vectors.
/// @par Example
/// @code
/// auto t=dm::algorithms::max_clique_trace(1,{{}});
/// std::print("{}\n",t.nodes[0].colours[0]); // 1
/// @endcode
/// @see max_clique_dyn, max_clique_analyze, max_clique_trace
/// @par Reference
/// Konc and Janezic (2007), An improved branch and bound algorithm for the maximum clique problem, MATCH58,569–590, sections2.2–2.3 describe the basic search/colouring. Native fixed-order search omits their later Dyn refinements.
struct MaxCliqueNode {
  std::size_t      id{0};          ///< Actual zero-based entered frame id.
  std::vector<int> clique;         ///< Actual chosen active clique, in DFS order.
  std::vector<int> candidates;     ///< Actual original candidate sequence at entry.
  std::vector<int> order;          ///< Actual vertices grouped by increasing greedy colour.
  std::vector<int> colours;        ///< Actual one-based colour bound paired with each order vertex.
  std::size_t      best_before{0}; ///< Actual best size before entering this frame.
};

/// @brief One actual candidate versus existing colour-class member adjacency test.
/// @details A conflict causes testing the next colour class; a conflict-free class receives the candidate.
/// @par Example
/// @code
/// auto t=dm::algorithms::max_clique_trace(2,{{1},{0}});
/// std::print("{}\n",t.colour_checks[0].conflict); // true
/// @endcode
/// @see max_clique_dyn, max_clique_analyze, max_clique_trace
/// @par Reference
/// Konc and Janezic (2007), An improved branch and bound algorithm for the maximum clique problem, MATCH58,569–590, sections2.2–2.3 describe the basic search/colouring. Native fixed-order search omits their later Dyn refinements.
struct MaxCliqueColourCheck {
  std::size_t node{0};         ///< Actual owning entered frame id.
  int         vertex{0};       ///< Actual candidate being placed.
  int         against{0};      ///< Actual previously placed member checked.
  std::size_t colour{0};       ///< Actual one-based tested colour class.
  bool        conflict{false}; ///< Actual normalized adjacency value.
};

/// @brief One actual leaf, branch choice or colour-bound cutoff.
/// @details Kind0 is leaf,1 branch,2 prune. Bounds use the actual active clique size plus current one-based colour. Leaf/branch/prune records are producing observations rather than reconstructed search trees.
/// @par Example
/// @code
/// auto t=dm::algorithms::max_clique_trace(1,{{}});
/// std::print("{}\n",t.decisions[0].kind); // 1
/// @endcode
/// @see max_clique_dyn, max_clique_analyze, max_clique_trace
/// @par Reference
/// Konc and Janezic (2007), An improved branch and bound algorithm for the maximum clique problem, MATCH58,569–590, sections2.2–2.3 describe the basic search/colouring. Native fixed-order search omits their later Dyn refinements.
struct MaxCliqueDecision {
  unsigned    kind{0};        ///< Actual event:0 leaf,1 branch,2 prune.
  std::size_t node{0};        ///< Actual owning entered frame id.
  int         vertex{-1};     ///< Actual selected/cutoff vertex, -1 at a leaf.
  std::size_t active_size{0}; ///< Actual active clique size at event.
  std::size_t bound{0};       ///< Actual local colour bound, or active leaf size.
  std::size_t best_before{0}; ///< Actual best size immediately before event.
  std::size_t best_after{0};  ///< Actual best size immediately after event.
};

/// @brief Owned original adjacency lists and actual colouring/search history.
/// @details Logical quota counts N vertex slots plus every listed input neighbour, every node snapshot record and its four vector scalar entries, every actual colour-check record and every actual decision record. Fields, row headers, capacities, runtime dense matrix/frames and bytes are excluded.
/// @par Example
/// @code
/// auto t=dm::algorithms::max_clique_trace(0,{});
/// std::print("{}\n",t.captured_entries); // 2
/// @endcode
/// @see max_clique_dyn, max_clique_analyze, max_clique_trace
/// @par Reference
/// Konc and Janezic (2007), An improved branch and bound algorithm for the maximum clique problem, MATCH58,569–590, sections2.2–2.3 describe the basic search/colouring. Native fixed-order search omits their later Dyn refinements.
struct MaxCliqueTrace {
  int n{0}; ///< Owned original vertex count.
  std::vector<std::vector<int>>
      input; ///< Owned original lists, preserving duplicates, self entries and asymmetric listings.
  MaxCliqueAnalysis                 analysis; ///< Actual best witness, completion/bound and work.
  std::vector<MaxCliqueNode>        nodes;    ///< Actual entered frame/colour snapshots.
  std::vector<MaxCliqueColourCheck> colour_checks; ///< Actual tested class-member conflicts.
  std::vector<MaxCliqueDecision> decisions; ///< Actual leaf/branch/prune events in producing order.
  std::size_t                    captured_entries{0}; ///< Complete logical capture count.
};

/// @brief One original seeded forty-vertex density experiment.
/// @details The nine rows continue the same mt19937_64 seed31 stream, preserving every original upper-triangle draw and modulo100 threshold. These are separate sampled graphs, not a nested edge-addition trajectory.
/// @par Example
/// @code
/// auto rows=dm::algorithms::max_clique_density_study();
/// std::print("{}\n",rows.size()); // 9
/// @endcode
/// @see max_clique_dyn, max_clique_analyze, max_clique_trace
/// @par Reference
/// Konc and Janezic (2007), An improved branch and bound algorithm for the maximum clique problem, MATCH58,569–590, sections2.2–2.3 describe the basic search/colouring. Native fixed-order search omits their later Dyn refinements.
struct MaxCliqueDensityRow {
  int                           percent{0}; ///< Actual prescribed threshold10..90.
  std::vector<std::vector<int>> graph;      ///< Actual generated symmetric forty-vertex graph.
  MaxCliqueAnalysis             analysis;   ///< Actual exact native search on this graph.
};

/// @brief Return an exact maximum clique using the retained greedy-colouring DFS order.
/// @details Validate all input first, ignore self entries and take symmetric union of listed edges; duplicates do not multiply edges. Greedy first-fit colouring processes actual candidate order; reverse colour-group order drives branches. Strict larger leaves replace best; equality bounds prune. An explicit frame stack preserves legacy DFS/tie results without recursive call-stack use. Dense matrix and active frame vectors need O(N^2) storage; worst-case time is exponential. This fixed-order native variant is not a claim to implement every published Dyn refinement.
/// @param n Nonnegative vertex count.
/// @param adj Exactly n original adjacency lists.
/// @return Sorted first strictly largest clique found by completed exact search.
/// @throws std::invalid_argument Before capture on negative n, wrong row count or any neighbour outside0..n-1.
/// @throws std::length_error On nonrepresentable dense/input/count/capture geometry.
/// @par Example
/// @code
/// auto c=dm::algorithms::max_clique_dyn(2,{{1},{0}});
/// std::print("{}\n",c.size()); // 2
/// @endcode
/// @see max_clique_dyn, max_clique_analyze, max_clique_trace
/// @par Reference
/// Konc and Janezic (2007), An improved branch and bound algorithm for the maximum clique problem, MATCH58,569–590, sections2.2–2.3 describe the basic search/colouring. Native fixed-order search omits their later Dyn refinements.
std::vector<int> max_clique_dyn(int n, const std::vector<std::vector<int>>& adj);

/// @brief Run the canonical colouring-bound producer with an optional actual entered-node cap.
/// @details Zero disables the node cap. Before entering a frame, a positive exhausted cap stops search with complete=false, retaining the actual best clique and root colouring upper bound. This is a work cap, not a wall-clock cap; one colouring can itself be expensive. Complete results have upper_bound=clique.size().
/// @param n Nonnegative vertex count.
/// @param adj Exactly n original lists, interpreted by symmetric union.
/// @param max_nodes Optional actual entered-frame cap, zero unlimited.
/// @return Actual lower-bound witness, valid upper bound, completion and selected work.
/// @throws std::invalid_argument Before capture on negative n, wrong row count or any neighbour outside0..n-1.
/// @throws std::length_error On nonrepresentable dense/input/count/capture geometry.
/// @par Example
/// @code
/// auto a=dm::algorithms::max_clique_analyze(1,{{}});
/// std::print("{} {}\n",a.complete,a.clique.size()); // true 1
/// @endcode
/// @see max_clique_dyn, max_clique_analyze, max_clique_trace
/// @par Reference
/// Konc and Janezic (2007), An improved branch and bound algorithm for the maximum clique problem, MATCH58,569–590, sections2.2–2.3 describe the basic search/colouring. Native fixed-order search omits their later Dyn refinements.
MaxCliqueAnalysis
max_clique_analyze(int n, const std::vector<std::vector<int>>& adj, std::size_t max_nodes = 0);

/// @brief Capture actual entered frames, greedy conflicts and search decisions under separate work/capture caps.
/// @details Complete input geometry is validated/admitted before copying. Actual node vectors, colour-test records and decisions are admitted before capture copies. Default logical entry cap one million; zero disables it. A work-cap stop returns the complete actual bounded history with complete=false; a capture-cap failure throws and never returns a prefix. Captured work can be exponential and is not bounded by the ordinary O(N^2) runtime matrix/frame storage.
/// @param n Nonnegative vertex count.
/// @param adj Exactly n original adjacency lists.
/// @param max_nodes Optional actual entered-frame work cap, zero unlimited.
/// @param max_entries Complete logical capture cap, zero off.
/// @return Owned original input, complete actual bounded history and same-run best/bounds.
/// @throws std::invalid_argument Before capture on negative n, wrong row count or any neighbour outside0..n-1.
/// @throws std::length_error On nonrepresentable dense/input/count/capture geometry.
/// @par Example
/// @code
/// auto t=dm::algorithms::max_clique_trace(1,{{}},1);
/// std::print("{}\n",t.analysis.complete); // false
/// @endcode
/// @see max_clique_dyn, max_clique_analyze, max_clique_trace
/// @par Reference
/// Konc and Janezic (2007), An improved branch and bound algorithm for the maximum clique problem, MATCH58,569–590, sections2.2–2.3 describe the basic search/colouring. Native fixed-order search omits their later Dyn refinements.
MaxCliqueTrace max_clique_trace(
    int                                  n,
    const std::vector<std::vector<int>>& adj,
    std::size_t                          max_nodes   = 0,
    std::size_t                          max_entries = 1000000);

/// @brief Return the checked simple symmetric union graph used by the canonical solver.
/// @details Each output row is sorted and unique; self loops are omitted. This helper shares validation/matrix construction with the search and is useful for truthful graph diagrams of asymmetric listed input.
/// @param n Nonnegative vertex count.
/// @param adj Exactly n original lists.
/// @return Owned canonical simple symmetric adjacency lists.
/// @throws std::invalid_argument Before capture on negative n, wrong row count or any neighbour outside0..n-1.
/// @throws std::length_error On nonrepresentable dense/input/count/capture geometry.
/// @par Example
/// @code
/// auto g=dm::algorithms::max_clique_normalize(2,{{},{0}});
/// std::print("{}\n",g[0][0]); // 1
/// @endcode
/// @see max_clique_dyn, max_clique_analyze, max_clique_trace
/// @par Reference
/// Konc and Janezic (2007), An improved branch and bound algorithm for the maximum clique problem, MATCH58,569–590, sections2.2–2.3 describe the basic search/colouring. Native fixed-order search omits their later Dyn refinements.
std::vector<std::vector<int>> max_clique_normalize(int n, const std::vector<std::vector<int>>& adj);

/// @brief Return the original eight-vertex clique/triangle/pendant listed fixture.
/// @details Retains the original asymmetric pendant entry7->6; the solver's symmetric union supplies6->7. The original maximum clique is0,1,2,3.
/// @return Owned original eight input lists.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::max_clique_dyn(8,dm::algorithms::max_clique_demo_graph()).size()); // 4
/// @endcode
/// @see max_clique_dyn, max_clique_analyze, max_clique_trace
/// @par Reference
/// Konc and Janezic (2007), An improved branch and bound algorithm for the maximum clique problem, MATCH58,569–590, sections2.2–2.3 describe the basic search/colouring. Native fixed-order search omits their later Dyn refinements.
std::vector<std::vector<int>> max_clique_demo_graph();

/// @brief Reproduce the original nine forty-vertex seeded density experiments.
/// @details One mt19937_64 seed31 stream; each pct10,20,...90 draws rng()%100 once per upper-triangle pair and includes it if below pct. Each actual graph runs the exact canonical producer without node cap. Nominal probabilities do not imply monotone clique sizes across these separately sampled graphs.
/// @return Original actual graphs and complete exact search results, in threshold order.
/// @par Example
/// @code
/// auto rows=dm::algorithms::max_clique_density_study();
/// std::print("{}\n",rows.front().percent); // 10
/// @endcode
/// @see max_clique_dyn, max_clique_analyze, max_clique_trace
/// @par Reference
/// Konc and Janezic (2007), An improved branch and bound algorithm for the maximum clique problem, MATCH58,569–590, sections2.2–2.3 describe the basic search/colouring. Native fixed-order search omits their later Dyn refinements.
std::vector<MaxCliqueDensityRow> max_clique_density_study();

} // namespace datamunge::algorithms
