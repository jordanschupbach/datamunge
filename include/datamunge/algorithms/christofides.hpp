#pragma once
/// @file christofides.hpp
/// @brief Bounded exact-matching Christofides construction, actual stage records and metric checks.
#include <datamunge/algorithms/tour_result.hpp>
#include <datamunge/algorithms/mst_result.hpp>
#include <utility>
namespace datamunge::algorithms {
/// @brief Complete-call storage limits for the exact odd-vertex matching subproblem.
/// max_odd_vertices may be 0..22, defaults to the legacy limit 22, and is a hard
/// vertex cap (zero permits only an empty odd set). max_matching_states counts
/// allocated table entries, including odd masks; zero is unlimited. Limits are
/// checked after the actual MST and before matching-table allocation. No partial
/// result escapes a cap failure. These do not bound the complete graph's storage.
/// @par Example
/// @code
/// dm::algorithms::ChristofidesOptions options;options.max_matching_states=4;
/// const auto r=dm::algorithms::christofides_tsp_with_options({{0,2},{2,0}},0,options);
/// std::print("{}\n",r.cost); // 4
/// @endcode
/// @see christofides_trace(), ChristofidesWork
struct ChristofidesOptions {
  std::size_t max_odd_vertices{22}; ///< Inclusive hard cap, at most 22; zero is a real zero cap.
  std::size_t max_matching_states{0}; ///< Maximum allocated 2^k table entries; zero is unlimited.
};
/// @brief Actual named construction counts, distinct from timing or bit complexity.
/// Every nonempty even mask is evaluated once; matching_candidates counts partner
/// relaxations for its least set bit. Odd masks occupy table storage but are never
/// evaluated. An empty odd set needs one table entry and no relaxations. Empty
/// input bypasses all graph/matching work. No sort comparisons are counted.
/// @par Example
/// @code
/// const auto t=dm::algorithms::christofides_trace({{0,2},{2,0}});
/// std::print("{} {} {}\n",t.work.matching_table_entries,t.work.matching_masks,t.work.matching_candidates); // 4 1 1
/// @endcode
/// @see ChristofidesTrace
struct ChristofidesWork {
  std::size_t input_entries{0}; ///< Validated finite nonnegative matrix entries.
  std::size_t complete_edges{0}; ///< Materialized upper-triangle edges, n(n-1)/2.
  std::size_t kruskal_candidates{0}; ///< Actual sorted edges inspected before the tree was complete.
  std::size_t matching_table_entries{0}; ///< Actual allocated cost/choice slots, 2^k for nonempty input.
  std::size_t matching_masks{0}; ///< Actual nonempty even masks evaluated.
  std::size_t matching_candidates{0}; ///< Actual candidate partner relaxations.
  std::size_t euler_edge_visits{0}; ///< Multigraph edge identities consumed once each.
  std::size_t shortcut_vertices{0}; ///< Euler positions inspected, including final closing start.
};
/// @brief Owned actual tree, exact-matching choices, Euler circuit and shortcut result.
/// All stages come from one execution without replay. tree edges are in Kruskal
/// acceptance order, odd_vertices in ascending order, and matching in bitmask
/// reconstruction order. Euler edge ids index tree followed by matching, retaining
/// parallel copies as distinct identities. kept_positions indexes each first visit
/// in euler_vertices; the final closing start is appended separately to result.
/// Costs use long double accumulation for diagnostics and double for the legacy
/// result. They describe rounded arithmetic, not an exact-real certificate.
/// @par Example
/// @code
/// const auto t=dm::algorithms::christofides_trace({{0,2},{2,0}});
/// std::print("{} {} {}\n",t.tree.size(),t.matching.size(),t.euler_edge_ids.size()); // 1 1 2
/// @endcode
/// @see christofides_trace(), ChristofidesWork
struct ChristofidesTrace {
  TourResult result; ///< Same closed tour and cost as the ordinary call with the same options.
  std::vector<WeightedEdge> tree; ///< Selected upper-triangle (u,v,weight) MST edges.
  std::vector<std::size_t> odd_vertices; ///< Actual odd-degree tree vertices, ascending.
  std::vector<WeightedEdge> matching; ///< Actual minimum-cost perfect matching on odd_vertices.
  std::vector<std::size_t> euler_vertices; ///< Actual closed Euler vertex walk, one more than its edge count.
  std::vector<std::size_t> euler_edge_ids; ///< Edge identity for each consecutive Euler vertex pair.
  std::vector<std::size_t> kept_positions; ///< Euler positions retained as first visits, in order.
  long double tree_cost{0}; ///< Accumulated selected tree weights.
  long double matching_cost{0}; ///< Accumulated selected matching weights.
  long double euler_cost{0}; ///< Directed input costs along the actual Euler walk; equals tree+matching for symmetric input.
  ChristofidesWork work; ///< Actual named operation/storage counts.
};
/// @brief Test symmetry, zero diagonal and all triangle inequalities within an absolute tolerance.
/// First validates the whole square finite nonnegative matrix and tolerance.
/// Empty input is metric by convention; off-diagonal zeros are allowed, so the
/// accepted class includes pseudometrics. Tolerance applies independently to each
/// comparison and is not a proof of an exact metric or an unchanged 3/2 bound.
/// Long-double comparisons avoid overflow in d(i,k)+d(k,j). O(n^3) worst-case
/// checks, constant auxiliary space, no input mutation or shared state.
/// @param distance Borrowed complete finite nonnegative square matrix.
/// @param tolerance Finite nonnegative absolute distance tolerance, default zero.
/// @return True iff diagonal, symmetry and triangle tests all pass.
/// @throws std::invalid_argument If matrix shape/entries or tolerance are invalid.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::is_tsp_metric({{0,1,3},{1,0,1},{3,1,0}})); // false
/// @endcode
/// @see christofides_tsp(), euclidean_tsp_distances()
bool is_tsp_metric(const std::vector<std::vector<double>>& distance,double tolerance=0);
/// @brief Build a symmetric complete Euclidean distance matrix from finite planar points.
/// Diagonal is exactly zero; repeated points are allowed. Coordinate differences
/// and hypot use long double before checked conversion to double. O(n^2) work and
/// owned storage. Floating-point rounding may require a small tolerance when
/// checking triangle inequalities; this does not assert exact-real certification.
/// @param points Borrowed (x,y) pairs in vertex order; empty input is supported.
/// @return Owned n by n matrix with finite nonnegative distances.
/// @throws std::invalid_argument If a coordinate is nonfinite.
/// @throws std::overflow_error If a distance cannot be represented as finite double.
/// @throws std::length_error If matrix storage exceeds container capacity.
/// @throws std::bad_alloc If allocation fails.
/// @par Example
/// @code
/// const auto d=dm::algorithms::euclidean_tsp_distances({{0,0},{3,4}});
/// std::print("{} {}\n",d[0][1],d[1][0]); // 5 5
/// @endcode
/// @see is_tsp_metric(), christofides_tsp()
std::vector<std::vector<double>> euclidean_tsp_distances(const std::vector<std::pair<double,double>>& points);
/// @brief Build controlled complete costs with one hub and interchangeable leaves.
/// Vertex zero is the hub; leaves are 1..leaves. Hub/leaf distances are spoke,
/// distinct leaf/leaf distances are rim, and the diagonal is zero. For at least
/// two leaves this is a pseudometric iff rim<=2*spoke; rim<spoke may change the
/// MST away from a star. These are synthetic costs, not necessarily planar
/// Euclidean distances. O((leaves+1)^2) work and owned storage.
/// @param leaves Number of leaf vertices, including zero for a singleton hub.
/// @param spoke Finite nonnegative hub/leaf cost, default one.
/// @param rim Finite nonnegative distinct leaf/leaf cost, default 1.5.
/// @return Owned symmetric (leaves+1) by (leaves+1) matrix.
/// @throws std::invalid_argument If spoke/rim are negative or nonfinite.
/// @throws std::length_error If vertex/storage dimensions exceed container capacity.
/// @throws std::bad_alloc If allocation fails.
/// @par Example
/// @code
/// const auto t=dm::algorithms::christofides_trace(dm::algorithms::hub_tsp_distances(4));
/// std::print("{} {}\n",t.odd_vertices.size(),t.result.cost); // 4 6.5
/// @endcode
/// @see christofides_trace(), is_tsp_metric()
std::vector<std::vector<double>> hub_tsp_distances(std::size_t leaves,double spoke=1,double rim=1.5);
/// @brief Construct a closed TSP tour using an MST, exact bounded matching and Euler shortcutting.
/// Preserves the legacy call/result layout and finite nonnegative square-matrix
/// acceptance. The 3/2 theorem requires symmetric zero-diagonal triangle costs;
/// these assumptions are NOT checked here (use is_tsp_metric()). Upper-triangle
/// weights choose the undirected tree/matching even for asymmetric inputs; final
/// cost uses directed input entries. Empty input ignores start and returns {};
/// singleton returns [start,start] with its diagonal cost. Tree ties now prefer
/// (weight,u,v); matching ties prefer the smallest partner of the least set bit.
/// Euler traversal consumes reverse insertion adjacency order. This makes ties
/// explicit; a previously unspecified tied tour can change. Nonempty odd sets
/// above 22 reject before matching allocation. Exact combinatorial matching uses
/// long-double sums, not a polynomial weighted-blossom implementation or exact
/// real arithmetic. No input mutation or shared state.
/// For k odd vertices: O(n^2 log(n+1)+k*2^k) work, O(n^2+2^k) storage.
/// @param distance Borrowed complete finite nonnegative square distance matrix.
/// @param start First/last vertex, default zero; ignored for empty input.
/// @return Owned closed Hamiltonian cycle and checked finite double cost.
/// @throws std::invalid_argument If shape/entries/start are invalid or the odd set exceeds 22.
/// @throws std::overflow_error If an accumulated cost is nonfinite or the final cost exceeds double range.
/// @throws std::length_error If required container storage cannot be represented.
/// @throws std::bad_alloc If allocation fails.
/// @par Example
/// @code
/// const auto r=dm::algorithms::christofides_tsp({{0,1,1.414,1},{1,0,1,1.414},{1.414,1,0,1},{1,1.414,1,0}});
/// std::print("{} {}\n",r.tour.size(),r.cost); // 5 4
/// @endcode
/// @par References
/// - N. Christofides (1976), Worst-Case Analysis of a New Heuristic for the
///   Travelling Salesman Problem, Management Sciences Research Report 388,
///   https://www.ibr.cs.tu-bs.de/courses/ss21/na/uebungen/ADA025602.pdf.
///   This implementation substitutes bounded subset DP for weighted matching.
/// @see christofides_trace(), christofides_tsp_with_options(), is_tsp_metric()
TourResult christofides_tsp(const std::vector<std::vector<double>>& distance,std::size_t start=0);
/// @brief Execute the same Christofides construction with explicit matching storage caps.
/// Same validation, deterministic ties, numerical limits and conditional metric
/// guarantee as christofides_tsp(). Cap failure returns no partial tour.
/// @param distance Borrowed finite nonnegative square matrix.
/// @param start First/last vertex, ignored for empty input.
/// @param options Matching caps; default legacy odd limit and unlimited table entries.
/// @return Owned closed tour and checked finite double cost.
/// @throws std::invalid_argument If input/start/options are invalid or the odd cap is exceeded.
/// @throws std::length_error If the positive table cap or container capacity is exceeded.
/// @throws std::overflow_error If an accumulated cost is nonfinite or the final cost exceeds double range.
/// @throws std::bad_alloc If allocation fails.
/// @par Example
/// @code
/// dm::algorithms::ChristofidesOptions options;options.max_odd_vertices=2;
/// std::print("{}\n",dm::algorithms::christofides_tsp_with_options({{0,2},{2,0}},0,options).cost); // 4
/// @endcode
/// @see christofides_tsp(), ChristofidesOptions
TourResult christofides_tsp_with_options(const std::vector<std::vector<double>>& distance,std::size_t start=0,ChristofidesOptions options={});
/// @brief Retain actual Christofides stages and stable multigraph edge identities without replay.
/// Same kernel, validation and caps as the ordinary APIs. Records own their values
/// and do not borrow inputs. No partial trace escapes a failure. Trace recording
/// adds O(n) stage/identity storage; no full DP table is retained in the result.
/// The metric guarantee is conditional, not inferred from a successful return.
/// @param distance Borrowed finite nonnegative square matrix.
/// @param start First/last vertex, default zero; ignored for empty input.
/// @param options Matching caps, default legacy odd limit and unlimited table entries.
/// @return Owned actual tree, matching, Euler walk, shortcut positions, result and counts.
/// @throws std::invalid_argument If input/start/options are invalid or the odd cap is exceeded.
/// @throws std::length_error If the positive table cap or container capacity is exceeded.
/// @throws std::overflow_error If an accumulated cost is nonfinite or the final cost exceeds double range.
/// @throws std::bad_alloc If allocation fails.
/// @par Example
/// @code
/// const auto t=dm::algorithms::christofides_trace({{0,2},{2,0}});
/// std::print("{} {}\n",t.euler_vertices.size(),t.kept_positions.size()); // 3 2
/// @endcode
/// @par References
/// - N. Christofides (1976), Report 388,
///   https://www.ibr.cs.tu-bs.de/courses/ss21/na/uebungen/ADA025602.pdf.
/// @see christofides_tsp(), ChristofidesTrace
ChristofidesTrace christofides_trace(const std::vector<std::vector<double>>& distance,std::size_t start=0,ChristofidesOptions options={});
}
