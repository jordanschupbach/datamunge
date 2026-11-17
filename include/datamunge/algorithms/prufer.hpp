#pragma once
/// @file prufer.hpp
/// @brief Checked smallest-leaf Prüfer bijection with actual leaf-set transitions.
#include <cstddef>
#include <utility>
#include <vector>
namespace datamunge::algorithms {
/// @brief Vertex and retained-entry policies for native tree coding.
/// Zero independently disables a cap; default limits count vertices and logical records/leaf ids, not bytes.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::PruferOptions{}.max_vertices); // 1000000
/// @endcode
struct PruferOptions {
  std::size_t max_vertices{1000000}; ///< Maximum admitted complete vertex count.
  std::size_t max_trace_entries{
      1000000}; ///< Initial leaf ids plus records and copied before/after leaf ids.
};
/// @brief One actual smallest-leaf removal or final decoding connection.
/// Leaf sets are copied from the actual ordered set before and after its updates; records own all values.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::tree_to_prufer_trace(3,{{0,1},{1,2}}).steps[0].leaf); // 0
/// @endcode
struct PruferStep {
  std::size_t leaf;     ///< Actual smallest chosen leaf.
  std::size_t neighbor; ///< Actual emitted neighbor or consumed code entry.
  std::size_t
      neighbor_degree_before; ///< Actual stored neighbor degree immediately before decrement.
  std::size_t neighbor_degree_after; ///< Actual stored neighbor degree immediately after decrement.
  bool        final_edge;            ///< True only for decoding's final two-leaf connection.
  std::vector<std::size_t> leaves_before; ///< Actual ordered available leaves before removal.
  std::vector<std::size_t> leaves_after;  ///< Actual ordered available leaves after updates.
};
/// @brief Owned produced sequence, final survivors and optional actual leaf transitions.
/// Ordinary output is the sequence; trace records expose producing decisions rather than a reconstructed deletion run.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::tree_to_prufer_trace(3,{{0,1},{1,2}}).sequence); // [1]
/// @endcode
struct PruferEncoding {
  std::vector<std::size_t> sequence;  ///< Actual smallest-leaf neighbor code, length n-2.
  std::vector<std::size_t> survivors; ///< Actual two unremoved vertices in ascending order.
  std::vector<std::size_t>
      initial_leaves;            ///< Actual initial ordered set, retained only with capture.
  std::vector<PruferStep> steps; ///< Actual producing removals, retained only with capture.
  std::size_t             adjacency_entries_examined{
      0}; ///< Actual encoding neighbor-list entries tested, excluding validation traversal.
};
/// @brief Owned decoded edges and optional actual leaf transitions.
/// The final edge is emitted by the same two-leaf producer and marked separately in history.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::prufer_to_tree_trace({1}).steps.back().final_edge); // true
/// @endcode
struct PruferDecoding {
  std::vector<std::pair<std::size_t, std::size_t>>
      edges; ///< Actual n-1 emitted leaf/neighbor edges, final pair last.
  std::vector<std::size_t>
      initial_leaves; ///< Actual initial ordered leaf set, retained only with capture.
  std::vector<PruferStep>
      steps; ///< Actual n-2 consumptions and final connection, retained only with capture.
};
/// @brief Encode a labeled tree by actual smallest-leaf removals.
/// Labels are0..n-1; n>=2. Ordinary entry points use a one-million-vertex policy. Complete endpoint/count and connectivity validation precedes leaf deletion, so cycles, disconnected graphs, parallel-edge cycles and self-loops cannot produce a partial code or invalid set access. The ordered-set algorithm takes O(n log n) ordinary time and O(n) storage. Inputs are borrowed unchanged and no shared mutable state is used.
/// @param n Number of labeled vertices, at least2.
/// @param tree_edges Borrowed n-1 undirected edges forming one connected tree.
/// @return Owned code of length n-2; n=2 gives empty.
/// @throws std::invalid_argument If vertices/edges/code are malformed, the encoding graph is not a tree, or a code label is out of range.
/// @throws std::length_error If vertex/history policy, length arithmetic or representable storage is exceeded.
/// @throws std::runtime_error If a checked internal leaf invariant fails.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::tree_to_prufer(3,{{0,1},{1,2}})); // [1]
/// @endcode
/// @par References
/// - H. Prüfer (1918), Neuer Beweis eines Satzes über Permutationen, Archiv der Mathematik und Physik27,142–144.
std::vector<std::size_t> tree_to_prufer(
    std::size_t                                             n,
    const std::vector<std::pair<std::size_t, std::size_t>>& tree_edges);
/// @brief Encode a labeled tree by actual smallest-leaf removals.
/// Labels are0..n-1; n>=2. Ordinary entry points use a one-million-vertex policy. Complete endpoint/count and connectivity validation precedes leaf deletion, so cycles, disconnected graphs, parallel-edge cycles and self-loops cannot produce a partial code or invalid set access. The ordered-set algorithm takes O(n log n) ordinary time and O(n) storage. Inputs are borrowed unchanged and no shared mutable state is used.
/// @param n Number of labeled vertices, at least2.
/// @param tree_edges Borrowed n-1 undirected edges forming one connected tree.
/// @param options Vertex/history policies; actual copied leaf-set histories can use O(n*n) entries.
/// @return Owned code and actual leaf-set histories.
/// @throws std::invalid_argument If vertices/edges/code are malformed, the encoding graph is not a tree, or a code label is out of range.
/// @throws std::length_error If vertex/history policy, length arithmetic or representable storage is exceeded.
/// @throws std::runtime_error If a checked internal leaf invariant fails.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::tree_to_prufer_trace(3,{{0,1},{1,2}}).survivors); // [1, 2]
/// @endcode
/// @par References
/// - H. Prüfer (1918), Neuer Beweis eines Satzes über Permutationen, Archiv der Mathematik und Physik27,142–144.
PruferEncoding tree_to_prufer_trace(
    std::size_t                                             n,
    const std::vector<std::pair<std::size_t, std::size_t>>& tree_edges,
    PruferOptions                                           options = {});
/// @brief Decode a Prüfer code through actual smallest-leaf connections.
/// Every valid code represents one tree; the inverse uses degree frequencies and an ordered set. O(n log n) ordinary work/O(n) storage; optional copied leaf sets can require O(n*n) history. Complete code validation precedes leaf processing, inputs remain unchanged and state is local.
/// @param sequence Borrowed code of length n-2 with every entry<n; empty decodes to the2-vertex tree.
/// @return Owned n-1 edges.
/// @throws std::invalid_argument If vertices/edges/code are malformed, the encoding graph is not a tree, or a code label is out of range.
/// @throws std::length_error If vertex/history policy, length arithmetic or representable storage is exceeded.
/// @throws std::runtime_error If a checked internal leaf invariant fails.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::prufer_to_tree({}).size()); // 1
/// @endcode
/// @par References
/// - H. Prüfer (1918), Neuer Beweis eines Satzes über Permutationen, Archiv der Mathematik und Physik27,142–144.
std::vector<std::pair<std::size_t, std::size_t>> prufer_to_tree(
    const std::vector<std::size_t>& sequence);
/// @brief Decode a Prüfer code through actual smallest-leaf connections.
/// Every valid code represents one tree; the inverse uses degree frequencies and an ordered set. O(n log n) ordinary work/O(n) storage; optional copied leaf sets can require O(n*n) history. Complete code validation precedes leaf processing, inputs remain unchanged and state is local.
/// @param sequence Borrowed code of length n-2 with every entry<n; empty decodes to the2-vertex tree.
/// @param options Vertex and actual retained-entry policies.
/// @return Owned n-1 edges and actual histories.
/// @throws std::invalid_argument If vertices/edges/code are malformed, the encoding graph is not a tree, or a code label is out of range.
/// @throws std::length_error If vertex/history policy, length arithmetic or representable storage is exceeded.
/// @throws std::runtime_error If a checked internal leaf invariant fails.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::prufer_to_tree_trace({1}).edges.size()); // 2
/// @endcode
/// @par References
/// - H. Prüfer (1918), Neuer Beweis eines Satzes über Permutationen, Archiv der Mathematik und Physik27,142–144.
PruferDecoding prufer_to_tree_trace(
    const std::vector<std::size_t>& sequence,
    PruferOptions                   options = {});
/// @brief Actual exhaustive Prüfer enumeration counts and checked Cayley reference.
/// Only the summary is returned; exact canonical edge sets are compared during enumeration without hash collisions.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::prufer_enumeration(4).distinct_trees); // 16
/// @endcode
struct PruferEnumeration {
  std::size_t vertices;  ///< Requested complete labeled vertex count.
  std::size_t sequences; ///< Actual codes decoded.
  std::size_t
      distinct_trees; ///< Actual distinct sorted undirected edge lists retained in the exact set.
  std::size_t cayley_count; ///< Checked n^(n-2) arithmetic reference, not an observed count.
};
/// @brief Exhaust all original labeled codes and count exact distinct decoded trees.
/// Enumerates base-n code digits in lexicographic order, calls the same checked native decoder, canonicalizes undirected edge endpoints and compares complete sorted edge lists. Default one-million-code admission occurs before enumeration. Work/storage grow exponentially with n; set comparisons and stored edge lists also cost O(n) per code. This finite exercise checks the bijection on the requested size, not a new proof of Cayley's theorem.
/// @param n At least2 labeled vertices; ordinary one-million-vertex policy applies.
/// @param max_sequences Admitted checked n^(n-2), defaultone million; zero disables the policy.
/// @return Owned actual enumeration/distinct counts and checked reference.
/// @throws std::invalid_argument If n<2.
/// @throws std::length_error If requested count, integer geometry, vertex policy or storage is exceeded.
/// @throws std::runtime_error If the shared decoder detects an internal invariant failure.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::prufer_enumeration(4).sequences); // 16
/// @endcode
/// @par References
/// - H. Prüfer (1918), Neuer Beweis eines Satzes über Permutationen, Archiv der Mathematik und Physik27,142–144.
/// @see prufer_to_tree()
PruferEnumeration prufer_enumeration(std::size_t n, std::size_t max_sequences = 1000000);
} // namespace datamunge::algorithms
