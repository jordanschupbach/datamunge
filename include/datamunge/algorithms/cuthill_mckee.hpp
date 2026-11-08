#pragma once
/// @file
/// @brief Checked deterministic Cuthill–McKee traversal and symmetric-pattern diagnostics.
#include <cstddef>
#include <vector>

namespace datamunge::algorithms {
/// @brief One actual CM queue removal and the neighbours it discovers.
/// Owned values; steps are in forward BFS order even for a reversed result.
/// Newly enqueued vertices use increasing (degree, original id), separately
/// for each removed vertex, rather than sorting a whole BFS level.
/// @par Example
/// @code
/// const auto t=dm::algorithms::cuthill_mckee_trace({{1},{0,2},{1}});
/// std::print("{} {}\n",t.steps[0].vertex,t.steps[0].newly_enqueued); // 0 [1]
/// @endcode
/// @see CuthillMcKeeTrace
struct CuthillMcKeeStep {
  int vertex{0}; ///< Original id removed from the queue.
  std::vector<int> newly_enqueued; ///< Owned discoveries, marked before later removals.
};

/// @brief Owned complete output and actual queue history of one CM traversal.
/// No input references survive. Arrays degree, parent and level use original ids;
/// roots parent themselves at level zero. Each new component resets its level.
/// Steps and forward_order always describe the forward run; order and position
/// describe the requested final direction. Empty graphs have empty arrays and
/// zero work. Adjacency scans count normalized directed entries, not raw duplicates.
/// @par Example
/// @code
/// const auto t=dm::algorithms::cuthill_mckee_trace({{1},{0,2},{1}},true);
/// std::print("{} {} {}\n",t.order,t.parent,t.adjacency_scans); // [2, 1, 0] [0, 0, 1] 4
/// @endcode
/// @par References
/// - E. Cuthill and J. McKee (1969), Reducing the bandwidth of sparse symmetric matrices, doi:10.1145/800195.805928.
/// @see cuthill_mckee_trace(), CuthillMcKeeStep
struct CuthillMcKeeTrace {
  std::vector<int> order; ///< New position to original vertex, reversed if requested.
  std::vector<int> position; ///< Original vertex to requested new position.
  std::vector<int> forward_order; ///< Original ids in actual queue-removal order.
  std::vector<int> degree; ///< Distinct off-diagonal degree for each original vertex.
  std::vector<int> parent; ///< Discovering original vertex; roots parent themselves.
  std::vector<int> level; ///< BFS distance from the current component root.
  std::vector<int> roots; ///< Component starts in forward traversal order.
  std::vector<CuthillMcKeeStep> steps; ///< Exactly one actual removal per vertex.
  std::size_t adjacency_scans{0}; ///< Exactly twice the normalized undirected edge count.
  std::size_t neighbor_enqueues{0}; ///< Non-root discoveries: n minus component count.
};

/// @brief Symmetric-pattern bandwidth and lower/upper envelope widths under an ordering.
/// Width arrays use new row positions. The diagonal is treated as structurally
/// present: isolated rows have zero widths. Profiles exclude diagonal entries;
/// n+lower_profile counts lower skyline slots, not actual factor nonzeros.
/// All values are owned; metrics describe the input pattern, not numeric values,
/// Cholesky fill, optimality or a speed guarantee.
/// @par Example
/// @code
/// const auto m=dm::algorithms::symmetric_pattern_metrics({{1,2,3},{0},{0},{0}},{0,1,2,3});
/// std::print("{} {} {}\n",m.bandwidth,m.lower_profile,m.upper_profile); // 3 6 3
/// @endcode
/// @see symmetric_pattern_metrics(), cuthill_mckee()
struct SymmetricPatternMetrics {
  std::size_t bandwidth{0}; ///< Maximum off-diagonal distance; zero for no edges.
  std::size_t lower_profile{0}; ///< Sum of row minus leftmost structural column.
  std::size_t upper_profile{0}; ///< Sum of rightmost structural column minus row.
  std::size_t edges{0}; ///< Distinct undirected off-diagonal edges.
  std::vector<std::size_t> lower_width; ///< Lower envelope distance in each new row.
  std::vector<std::size_t> upper_width; ///< Upper envelope distance in each new row.
};

/// @brief Owned row/column coordinates of a normalized symmetric sparsity pattern.
/// Zero-based integral coordinates stored exactly as doubles (ids fit int).
/// Row-major order, sorted columns; the optional diagonal is included once.
/// No numeric matrix values or references to inputs are retained.
/// @par Example
/// @code
/// const auto p=dm::algorithms::symmetric_pattern_coordinates({{1},{0}},{1,0});
/// std::print("{} {}\n",p.row,p.column); // [0, 0, 1, 1] [0, 1, 0, 1]
/// @endcode
/// @see symmetric_pattern_coordinates()
struct SymmetricPatternCoordinates {
  std::vector<double> row; ///< New row for every structural entry.
  std::vector<double> column; ///< Matching new column for every structural entry.
};

/// @brief Return a deterministic minimum-degree-start Cuthill–McKee ordering.
/// Input is an undirected adjacency list; duplicate entries and diagonal loops
/// are ignored. Off-diagonal symmetry is required after normalization. Components
/// start at the smallest (degree,id) unvisited vertex. Each removed vertex's
/// undiscovered neighbours are enqueued in that same order and marked immediately.
/// Ties are explicit, independent of input row order. This is a heuristic: it may
/// increase bandwidth. Reversing the whole result preserves bandwidth exactly;
/// it can change profile and fill. No pseudo-peripheral-root search is performed.
/// Read-only inputs, owned output, no shared mutable state. Empty input returns
/// empty. Normalization and traversal take O(n log(n+1)+sum r_i log(r_i+1)) time
/// and O(n+m_raw) auxiliary storage, including a sorted component-start list;
/// duplicate-free row sizes are bounded by raw sizes r_i. No optional size cap.
/// @param adjacency Raw rows indexed by original vertex; ids must fit [0,n).
/// @param reverse Reverse the entire forward order when true; default false.
/// @return Permutation with order[new_position]=original_vertex.
/// @throws std::invalid_argument Negative/out-of-range id or missing reverse edge.
/// @throws std::length_error Vertex count exceeds INT_MAX or a size sum overflows.
/// @throws std::bad_alloc Allocation failure, as for all owning APIs in this file.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::cuthill_mckee({{1},{0,2},{1}},true)); // [2, 1, 0]
/// @endcode
/// @par References
/// - E. Cuthill and J. McKee (1969), Reducing the bandwidth of sparse symmetric matrices, pp. 157–172, doi:10.1145/800195.805928.
/// - SciPy reverse_cuthill_mckee: https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csgraph.reverse_cuthill_mckee.html (tie/start policy may differ).
/// @see cuthill_mckee_trace(), symmetric_pattern_metrics()
std::vector<int> cuthill_mckee(const std::vector<std::vector<int>>& adjacency, bool reverse=false);

/// @brief Capture the complete actual canonical CM traversal without replay.
/// Same input, tie, component and reverse semantics as cuthill_mckee(). All
/// endpoint ids and raw size sums are checked before optional limits and before
/// normalized graph/history allocation. Symmetry is checked after normalization.
/// Limits count raw vertices/entries (including loops/duplicates); zero disables
/// only that optional cap. No silent truncation. Trace storage is O(n+m_raw).
/// @param adjacency Read-only symmetric raw adjacency rows.
/// @param reverse Reverse final order/position only; default false.
/// @param max_vertices Complete vertex cap, default 100000; zero disables.
/// @param max_adjacency_entries Complete raw directed-entry cap, default 1000000; zero disables.
/// @return Owned traversal, inverse map and full history.
/// @throws std::invalid_argument Invalid endpoint or asymmetric normalized graph.
/// @throws std::length_error Integer/size representation or optional limit exceeded.
/// @throws std::bad_alloc Allocation failure.
/// @par Example
/// @code
/// const auto t=dm::algorithms::cuthill_mckee_trace({{}, {2}, {1}},false,3,2);
/// std::print("{} {}\n",t.order,t.roots); // [0, 1, 2] [0, 1]
/// @endcode
/// @par References
/// - E. Cuthill and J. McKee (1969), doi:10.1145/800195.805928; deterministic low-degree-start variant.
/// @see cuthill_mckee(), CuthillMcKeeTrace
CuthillMcKeeTrace cuthill_mckee_trace(const std::vector<std::vector<int>>& adjacency, bool reverse=false, std::size_t max_vertices=100000, std::size_t max_adjacency_entries=1000000);

/// @brief Validate and invert a new-position-to-original permutation.
/// Empty input returns empty. Read-only input, owned result, O(n) time/storage.
/// @param order Permutation of [0,n), n at most INT_MAX.
/// @return position[order[k]]=k for every k.
/// @throws std::invalid_argument Missing, repeated, negative or out-of-range id.
/// @throws std::length_error More than INT_MAX entries.
/// @throws std::bad_alloc Allocation failure.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::inverse_ordering({2,0,1})); // [1, 2, 0]
/// @endcode
/// @see permute_symmetric_pattern()
std::vector<int> inverse_ordering(const std::vector<int>& order);

/// @brief Normalize and symmetrically permute a graph pattern to new row/column ids.
/// B[k,l]=A[order[k],order[l]] structurally, with loops removed and duplicate
/// entries collapsed. Sorted output rows; inputs are unchanged, output owned.
/// Empty graph/order return empty. Normalization as cuthill_mckee(); sorting
/// output takes O(sum d_i log(d_i+1)), auxiliary storage O(n+m_raw).
/// @param adjacency Raw symmetric pattern with valid ids.
/// @param order New-position-to-original permutation of all graph vertices.
/// @return Owned simple symmetric adjacency in new coordinates.
/// @throws std::invalid_argument Invalid graph, permutation, or size mismatch.
/// @throws std::length_error Integer/size representation exceeded.
/// @throws std::bad_alloc Allocation failure.
/// @par Example
/// @code
/// const auto g=dm::algorithms::permute_symmetric_pattern({{1,2},{0},{0}},{1,0,2});
/// std::print("{}\n",g); // [[1], [0, 2], [1]]
/// @endcode
/// @see inverse_ordering(), symmetric_pattern_coordinates()
std::vector<std::vector<int>> permute_symmetric_pattern(const std::vector<std::vector<int>>& adjacency, const std::vector<int>& order);

/// @brief Measure bandwidth and both row-envelope profiles of a symmetric pattern.
/// Normalizes as cuthill_mckee(), validates the full permutation, then scans
/// every distinct directed edge once. Profiles use checked size_t sums. No
/// numerical factorization; metrics do not count fill. Inputs read-only, output
/// owned. O(n+m_raw) storage; normalization plus linear edge scan time.
/// @param adjacency Raw undirected pattern, loops/duplicates ignored.
/// @param order New-position-to-original permutation, including empty for n=0.
/// @return Owned metrics and widths in new row coordinates.
/// @throws std::invalid_argument Invalid graph, permutation, or size mismatch.
/// @throws std::length_error Integer/size representation or profile sum overflow.
/// @throws std::bad_alloc Allocation failure.
/// @par Example
/// @code
/// const auto m=dm::algorithms::symmetric_pattern_metrics({{1},{0,2},{1}},{0,2,1});
/// std::print("{} {}\n",m.bandwidth,m.lower_width); // 2 [0, 0, 2]
/// @endcode
/// @see SymmetricPatternMetrics, cuthill_mckee()
SymmetricPatternMetrics symmetric_pattern_metrics(const std::vector<std::vector<int>>& adjacency, const std::vector<int>& order);

/// @brief Extract sorted structural plotting coordinates after symmetric permutation.
/// Uses permute_symmetric_pattern() and emits each directed off-diagonal entry
/// once; optionally emits all n diagonal entries regardless of raw loops. No
/// external plotting state or numeric matrix data. Time as permutation plus
/// O(n+m) extraction; storage O(n+m_raw). Empty graph yields empty coordinates.
/// @param adjacency Raw symmetric graph pattern.
/// @param order New-position-to-original permutation of its vertices.
/// @param include_diagonal Include all diagonal positions; default true.
/// @return Owned matching row/column arrays in new row-major order.
/// @throws std::invalid_argument Invalid graph, permutation, or size mismatch.
/// @throws std::length_error Integer/size representation or output count overflow.
/// @throws std::bad_alloc Allocation failure.
/// @par Example
/// @code
/// const auto p=dm::algorithms::symmetric_pattern_coordinates({{1},{0}},{0,1},false);
/// std::print("{} {}\n",p.row,p.column); // [0, 1] [1, 0]
/// @endcode
/// @see SymmetricPatternCoordinates, permute_symmetric_pattern()
SymmetricPatternCoordinates symmetric_pattern_coordinates(const std::vector<std::vector<int>>& adjacency, const std::vector<int>& order, bool include_diagonal=true);

/// @brief Generate the simple symmetric pattern joining all ids at distance at most a width.
/// n zero or one gives empty rows; width zero gives n isolated vertices. Width
/// beyond n-1 is clamped, giving a complete graph. Counts are checked before graph
/// allocation; optional limit counts directed entries and zero disables it.
/// O(n+m) time/storage, read-only arguments, owned output, no random state.
/// @param n Vertex count at most INT_MAX.
/// @param half_bandwidth Maximum positive id distance; any size_t accepted.
/// @param max_adjacency_entries Complete directed-entry cap, default 1000000; zero disables.
/// @return Sorted symmetric loop-free adjacency rows.
/// @throws std::length_error Id/count representation or optional cap exceeded.
/// @throws std::bad_alloc Allocation failure.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::symmetric_band_pattern(3,1)); // [[1], [0, 2], [1]]
/// @endcode
/// @see permute_symmetric_pattern(), cuthill_mckee()
std::vector<std::vector<int>> symmetric_band_pattern(std::size_t n, std::size_t half_bandwidth, std::size_t max_adjacency_entries=1000000);
} // namespace datamunge::algorithms
