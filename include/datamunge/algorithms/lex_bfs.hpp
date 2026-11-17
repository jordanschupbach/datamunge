#pragma once
/// @file lex_bfs.hpp
/// @brief Checked dense Lex-BFS with actual ordered partitions/classifications and separate perfect-elimination verification.
#include <cstddef>
#include <vector>
namespace datamunge::algorithms {
/// @brief Actual selected matrix/refinement work, excluding guards/copies/allocation/time.
/// @par Example
/// @code
/// const auto a=dm::algorithms::lex_bfs_analyze(3,{{1},{0,2},{1}});
/// std::print("{}\n",a.work.neighbor_tests); // 3
/// @endcode
/// @par References
/// - Rose, Tarjan and Lueker(1976), Algorithmic Aspects of Vertex Elimination on Graphs, SIAM J.Comput.5(2),266–283, https://doi.org/10.1137/0205021; this dense scan implementation is not the paper's linear-time data structure.
/// @see lex_bfs_analyze(), lex_bfs_trace()
struct LexBFSWork {
  std::size_t matrix_cells{0}; ///< Actual zero-initialized dense adjacency cells, n*n.
  std::size_t adjacency_entries{
      0}; ///< Actual original input entries scanned to build the symmetric matrix, including loops/duplicates.
  std::size_t emitted{0};        ///< Actual selected/output vertices.
  std::size_t blocks_scanned{0}; ///< Actual nonempty remaining blocks scanned during refinement.
  std::size_t neighbor_tests{0}; ///< Actual classified remaining vertices, n*(n-1)/2.
};
/// @brief Owned actual deterministic Lex-BFS ordering and same-execution selected work.
/// @par Example
/// @code
/// const auto a=dm::algorithms::lex_bfs_analyze(3,{{1},{0,2},{1}});
/// std::print("{} {} {}\n",a.order[0],a.order[1],a.order[2]); // 2 1 0
/// @endcode
/// @par References
/// - Rose, Tarjan and Lueker(1976), Algorithmic Aspects of Vertex Elimination on Graphs, SIAM J.Comput.5(2),266–283, https://doi.org/10.1137/0205021; this dense scan implementation is not the paper's linear-time data structure.
/// @see lex_bfs_analyze(), lex_bfs_trace()
struct LexBFSAnalysis {
  std::vector<int> order; ///< Actual selected vertex identities in forward output order.
  LexBFSWork       work;  ///< Actual dense construction/refinement operations.
};
/// @brief One actual membership test made while refining a real remaining block.
/// @par Example
/// @code
/// const auto t=dm::algorithms::lex_bfs_trace(2,{{1},{0}});
/// std::print("{}\n",t.classifications[0].neighbor); // true
/// @endcode
/// @par References
/// - Rose, Tarjan and Lueker(1976), Algorithmic Aspects of Vertex Elimination on Graphs, SIAM J.Comput.5(2),266–283, https://doi.org/10.1137/0205021; this dense scan implementation is not the paper's linear-time data structure.
/// @see lex_bfs_analyze(), lex_bfs_trace()
struct LexBFSClassification {
  std::size_t step{0};         ///< Actual zero-based emitted-vertex step.
  std::size_t block{0};        ///< Actual remaining block index after selected vertex removal.
  int         selected{-1};    ///< Actual selected vertex whose neighborhood is tested.
  int         vertex{-1};      ///< Actual unvisited vertex being classified.
  bool        neighbor{false}; ///< Actual normalized adjacency matrix test result.
};
/// @brief Owned actual initial or post-refinement ordered partition snapshot.
/// @par Example
/// @code
/// const auto t=dm::algorithms::lex_bfs_trace(2,{{1},{0}});
/// std::print("{} {}\n",t.stages.front().selected,t.stages.back().blocks.size()); // -1 0
/// @endcode
/// @par References
/// - Rose, Tarjan and Lueker(1976), Algorithmic Aspects of Vertex Elimination on Graphs, SIAM J.Comput.5(2),266–283, https://doi.org/10.1137/0205021; this dense scan implementation is not the paper's linear-time data structure.
/// @see lex_bfs_analyze(), lex_bfs_trace()
struct LexBFSStage {
  int selected{-1}; ///< Actual just-selected vertex, or-1 for original initial partition.
  std::vector<std::vector<int>> blocks; ///< Actual ordered remaining blocks and member order.
};
/// @brief Owned actual normalized matrix, original adjacency, every partition and real membership tests.
/// Exact complete capture admission is E+2*n*n+n+1 before copies/matrix work.
/// Counts original adjacency entries, n*n matrix cells, n+1 stage records,
/// n*(n+1)/2 member slots and n*(n-1)/2 classifications. Excludes row/block
/// container headers, fields/capacities/output/runtime copies and bytes/RAM.
/// Default million logical records;0 disables only optional cap. No prefix reruns.
/// @par Example
/// @code
/// const auto t=dm::algorithms::lex_bfs_trace(2,{{1},{0}});
/// std::print("{}\n",t.stages.size()); // 3
/// @endcode
/// @par References
/// - Rose, Tarjan and Lueker(1976), Algorithmic Aspects of Vertex Elimination on Graphs, SIAM J.Comput.5(2),266–283, https://doi.org/10.1137/0205021; this dense scan implementation is not the paper's linear-time data structure.
/// @see lex_bfs_analyze(), lex_bfs_trace()
struct LexBFSTrace {
  std::vector<std::vector<int>>
      input_adjacency; ///< Owned original caller rows, including asymmetric entries/loops/duplicates.
  std::vector<std::vector<char>>
      matrix; ///< Owned actual simple undirected matrix used by the producer.
  std::vector<LexBFSStage>
      stages; ///< Actual initial partition then every real post-refinement partition, including final empty.
  std::vector<LexBFSClassification>
                 classifications; ///< Actual chronological step/block/member membership decisions.
  LexBFSAnalysis analysis;        ///< Owned actual final order and selected work.
  std::size_t    captured_entries{
      0}; ///< Complete original-entry/matrix-cell/stage/member/classification logical count.
};
/// @brief Actual clique checks for later neighbors of a supplied candidate elimination order.
/// Separate O(n cubed+E) diagnostic, not work done by ordinary Lex-BFS and
/// not a claim of a linear-time chordality implementation. An actual witness
/// disproves this candidate order; arbitrary failed candidates alone do not
/// prove nonchordality. A reverse Lex-BFS candidate has the chordal equivalence.
/// @par Example
/// @code
/// const auto p=dm::algorithms::check_perfect_elimination_order(3,{{1},{0,2},{1}},{2,1,0});
/// std::print("{}\n",p.valid); // true
/// @endcode
/// @par References
/// - Rose, Tarjan and Lueker(1976), Algorithmic Aspects of Vertex Elimination on Graphs, SIAM J.Comput.5(2),266–283, https://doi.org/10.1137/0205021; this dense scan implementation is not the paper's linear-time data structure.
/// @see lex_bfs_analyze(), lex_bfs_trace()
struct PerfectEliminationCheck {
  bool        valid{true}; ///< True exactly when every later-neighbor pair is adjacent.
  int         vertex{-1};  ///< Actual earliest candidate vertex with a failed pair, or-1 if valid.
  int         left{-1};    ///< Actual first later nonadjacent neighbor witness, or-1 if valid.
  int         right{-1};   ///< Actual second later nonadjacent neighbor witness, or-1 if valid.
  std::size_t neighbor_pairs{
      0}; ///< Actual later-neighbor adjacency pairs checked until first failure.
};
/// @brief Order an undirected normalized graph using the historical deterministic dense partition refinement.
/// Initialize the one block[0,...,n-1], repeatedly pop the last member of the
/// first block, then split each remaining block with neighbors first and its
/// member order preserved. Ties therefore pick the largest remaining member
/// in the first block; this does not start at0 unless n<=1. Disconnected
/// components are valid. Dense matrix and scans cost O(n squared+E) time and
/// O(n squared+n) runtime storage, explicitly not a linear-time implementation.
/// No input mutation, borrowed output lifetime or shared state.
/// @param n Nonnegative vertex count, identities0 through n-1.
/// @param adj Borrowed exactly n rows with all endpoints in range. Asymmetric entries symmetrize; loops ignored, duplicates coalesced.
/// @return Owned complete actual forward Lex-BFS permutation; empty when n0.
/// @throws std::invalid_argument On negative n, wrong complete row count or any endpoint outside range.
/// @throws std::length_error On complete dense/work/container geometry.
/// @throws std::bad_alloc On runtime/output allocation failure.
/// @par Example
/// @code
/// const auto o=dm::algorithms::lex_bfs(3,{{1},{0,2},{1}});
/// std::print("{}\n",o[0]); // 2
/// @endcode
/// @par References
/// - Rose, Tarjan and Lueker(1976), Algorithmic Aspects of Vertex Elimination on Graphs, SIAM J.Comput.5(2),266–283, https://doi.org/10.1137/0205021; this dense scan implementation is not the paper's linear-time data structure.
/// @see LexBFSTrace, check_perfect_elimination_order()
std::vector<int> lex_bfs(int n, const std::vector<std::vector<int>>& adj);
/// @brief Return the same actual dense Lex-BFS ordering and selected construction/refinement work.
/// Same complete graph normalization/tie/empty/geometry contract as lex_bfs().
/// Counts n*(n-1)/2 actual membership tests even for empty/complete graphs.
/// @param n Nonnegative vertex count, identities0 through n-1.
/// @param adj Borrowed exactly n rows with all endpoints in range. Asymmetric entries symmetrize; loops ignored, duplicates coalesced.
/// @return Owned actual order/work without optional history.
/// @throws std::invalid_argument On negative n, wrong complete row count or any endpoint outside range.
/// @throws std::length_error On complete dense/work/container geometry.
/// @throws std::bad_alloc On runtime/output allocation failure.
/// @par Example
/// @code
/// const auto a=dm::algorithms::lex_bfs_analyze(3,{{1},{0,2},{1}});
/// std::print("{} {}\n",a.work.matrix_cells,a.work.emitted); // 9 3
/// @endcode
/// @par References
/// - Rose, Tarjan and Lueker(1976), Algorithmic Aspects of Vertex Elimination on Graphs, SIAM J.Comput.5(2),266–283, https://doi.org/10.1137/0205021; this dense scan implementation is not the paper's linear-time data structure.
/// @see LexBFSTrace, check_perfect_elimination_order()
LexBFSAnalysis lex_bfs_analyze(int n, const std::vector<std::vector<int>>& adj);
/// @brief Capture the actual dense matrix, ordered partitions and every producing membership decision once.
/// Same shared kernel as ordinary/count-only calls, no display reconstruction.
/// Full E+2*n*n+n+1 capture geometry/quota admitted before any input copies or
/// matrix evaluation; guards still apply with cap0. Captures own their data.
/// @param n Nonnegative vertex count, identities0 through n-1.
/// @param adj Borrowed exactly n rows with all endpoints in range. Asymmetric entries symmetrize; loops ignored, duplicates coalesced.
/// @param max_entries Complete logical record cap, default1000000;0 disables optional cap.
/// @return Owned original rows and every actual producing snapshot/test/order/work.
/// @throws std::invalid_argument On negative n, wrong complete row count or any endpoint outside range.
/// @throws std::length_error On complete dense/work/container geometry.
/// @throws std::bad_alloc On runtime/output allocation failure.
/// @par Example
/// @code
/// const auto t=dm::algorithms::lex_bfs_trace(2,{{1},{0}},13);
/// std::print("{}\n",t.captured_entries); // 13
/// @endcode
/// @par References
/// - Rose, Tarjan and Lueker(1976), Algorithmic Aspects of Vertex Elimination on Graphs, SIAM J.Comput.5(2),266–283, https://doi.org/10.1137/0205021; this dense scan implementation is not the paper's linear-time data structure.
/// @see LexBFSTrace, check_perfect_elimination_order()
LexBFSTrace
lex_bfs_trace(int n, const std::vector<std::vector<int>>& adj, std::size_t max_entries = 1000000);
/// @brief Verify an explicit full candidate order by checking every later-neighbor clique until first failure.
/// Complete graph/permutation guards precede pair checks. O(n cubed+E) worst
/// time/O(n squared+n) runtime storage, separate from any Lex-BFS execution.
/// Candidate failure is not by itself a matrix/graph structural certificate.
/// @param n Nonnegative vertex count, identities0 through n-1.
/// @param adj Borrowed exactly n rows with all endpoints in range. Asymmetric entries symmetrize; loops ignored, duplicates coalesced.
/// @param order Borrowed complete candidate permutation, first eliminated first; reverse a forward Lex-BFS result explicitly.
/// @return Actual valid status or earliest failed later-neighbor pair and real pair count.
/// @throws std::invalid_argument On negative n, wrong complete row count or any endpoint outside range.
/// @throws std::length_error On complete dense/work/container geometry.
/// @throws std::bad_alloc On runtime/output allocation failure.
/// @throws std::invalid_argument On incomplete/duplicate/out-of-range permutation.
/// @par Example
/// @code
/// const auto p=dm::algorithms::check_perfect_elimination_order(4,{{1,3},{0,2},{1,3},{0,2}},{0,1,2,3});
/// std::print("{}\n",p.valid); // false
/// @endcode
/// @par References
/// - Rose, Tarjan and Lueker(1976), Algorithmic Aspects of Vertex Elimination on Graphs, SIAM J.Comput.5(2),266–283, https://doi.org/10.1137/0205021; this dense scan implementation is not the paper's linear-time data structure.
/// @see LexBFSTrace, check_perfect_elimination_order()
PerfectEliminationCheck check_perfect_elimination_order(
    int                                  n,
    const std::vector<std::vector<int>>& adj,
    const std::vector<int>&              order);
} // namespace datamunge::algorithms
