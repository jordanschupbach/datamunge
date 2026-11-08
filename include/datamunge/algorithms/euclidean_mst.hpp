#pragma once
/// @file euclidean_mst.hpp
/// @brief Complete planar Euclidean graphs and shared actual Kruskal minimum-spanning trajectories.
#include <datamunge/algorithms/layout_geometry.hpp>
#include <datamunge/algorithms/kruskal.hpp>
namespace datamunge::algorithms {

/// @brief Actual distance and stable Kruskal work from a complete planar graph.
/// Each unordered point pair is measured once with hypot, before sorting.
/// Logical counts exclude coordinate validation, allocation, sorting and CPU time.
///
///
/// @par Example
/// @code
/// const auto r=dm::algorithms::euclidean_minimum_spanning_tree_work({{0,0},{1,0},{0,1}});
/// std::print("{} {}\n",r.work.distance_evaluations,r.work.kruskal.accepted_edges); // 3 2
/// @endcode
///
/// @see euclidean_minimum_spanning_tree_work()
struct EuclideanMstWork {
    std::size_t distance_evaluations=0; ///< Complete pair count n(n-1)/2, including coincident point pairs.
    KruskalWork kruskal; ///< Actual generic forest execution counts.
};

/// @brief Owned planar minimum spanning tree and actual complete-graph work.
/// Point indices are preserved, including separate vertices at identical coordinates.
/// The output owns its edge records; no inputs or optional trajectory are retained.
///
///
/// @par Example
/// @code
/// const auto r=dm::algorithms::euclidean_minimum_spanning_tree_work({{0,0},{3,4}});
/// std::print("{} {}\n",r.result.total_weight,r.work.distance_evaluations); // 5 1
/// @endcode
///
/// @see euclidean_minimum_spanning_tree_work()
struct EuclideanMstOutcome {
    MinimumSpanningTree result; ///< Minimum tree with checked finite total; empty input is connected.
    EuclideanMstWork work; ///< Actual pair-distance and Kruskal counts.
};

/// @brief Owned points and actual Kruskal trace on their complete Euclidean graph.
/// Pair edges are generated lexicographically (i,j), i<j, with stable ties in that
/// order. Captured entries C=N+M+S count N point records plus the nested M input-edge
/// and S actual scan records. N+M is checked before any generated graph/copy; nested
/// steps reserve online before scanning. Default one million; zero disables only quota.
/// Output tree, runtime graph/sort/DSU state, capacities and allocator bytes are excluded,
/// so this is not a peak-memory bound. Points, nested trace and output are all owned;
/// exceptions expose no partial result. No independent optimization replay is performed.
///
///
/// @par Example
/// @code
/// const auto t=dm::algorithms::euclidean_minimum_spanning_tree_trace({{0,0},{3,4}});
/// std::print("{} {} {}\n",t.points.size(),t.kruskal.outcome.result.total_weight,t.captured_entries); // 2 5 4
/// @endcode
///
/// @see euclidean_minimum_spanning_tree_trace(), KruskalTrace
struct EuclideanMstTrace {
    std::vector<Point2D> points; ///< Owned validated input coordinates, in original order.
    KruskalTrace kruskal; ///< Owned complete edges and actual shared-kernel decisions/outcome.
    std::size_t distance_evaluations=0; ///< Actual complete-pair distance calls.
    std::size_t captured_entries=0; ///< points.size()+kruskal.captured_entries logical records.
};

/// @brief Check complete-pair geometry without allocating a graph.
/// Computes n(n-1)/2 by dividing an even factor before checked multiplication.
/// Also checks the point, edge and DSU container geometries required by these APIs.
/// Zero and one yield zero; this is a count check, not a promise of available RAM.
///
/// @param point_count Number of labeled planar point vertices.
/// @return Representable number of unordered distinct-index pairs.
/// @throws std::length_error If arithmetic or required container limits are exceeded.
///
/// @par Example
/// @code
/// std::print("{} {}\n",dm::algorithms::euclidean_mst_pair_count(4),dm::algorithms::euclidean_mst_pair_count(0)); // 6 0
/// @endcode
///
/// @see euclidean_minimum_spanning_tree_work()
std::size_t euclidean_mst_pair_count(std::size_t point_count);

/// @brief Return the complete-graph Euclidean minimum spanning tree of planar points.
/// Constructs every unordered pair in lexicographic input-index order, weighted by
/// Euclidean hypot distance; then calls the shared stable Kruskal kernel. Equal weights
/// retain pair order. Coincident points remain distinct vertices joined by zero edges.
/// Empty/singleton input is connected with zero total. Coordinates are arbitrary
/// Cartesian units and outputs use those same units; this is not geographic distance,
/// a tour or a Steiner tree. The complete graph always connects finite valid inputs.
/// O(n² log(n+1)) time and O(n²+n) runtime memory; no Delaunay sparsification is used.
/// All coordinates and pair/container geometry are checked before generated allocation.
/// Independent calls share no mutable state; concurrent caller mutation is unsafe.
/// Ordinary/work calls retain no optional history or capture quota. The original
/// function remains available through advanced_graph.hpp.
///
/// @param points Borrowed finite planar vertices; empty, singleton and duplicate coordinates are supported.
/// @return Owned minimum tree, with edges oriented i<j and stable accepted order.
/// @throws std::invalid_argument If any coordinate is nonfinite, including a singleton.
/// @throws std::length_error If pair/container/count geometry exceeds limits.
/// @throws std::overflow_error If a difference, pair distance or accepted-edge total is nonfinite.
/// @throws std::bad_alloc If required storage cannot be allocated.
///
/// @par Example
/// @code
/// const auto r=dm::algorithms::euclidean_minimum_spanning_tree({{0,0},{1,0},{1,1},{0,1}});
/// std::print("{} {}\n",r.edges.size(),r.total_weight); // 3 3
/// @endcode
///
/// @par References
/// - J. B. Kruskal, Jr. (1956). On the shortest spanning subtree of a graph and the traveling salesman problem. Proceedings of the AMS 7(1), 48-50. https://doi.org/10.1090/S0002-9939-1956-0078686-7
///
/// @see euclidean_minimum_spanning_tree_work(), euclidean_minimum_spanning_tree_trace()
MinimumSpanningTree euclidean_minimum_spanning_tree(const std::vector<Point2D>& points);

/// @brief Return a planar minimum tree and actual pair-distance/union-find work.
/// Constructs every unordered pair in lexicographic input-index order, weighted by
/// Euclidean hypot distance; then calls the shared stable Kruskal kernel. Equal weights
/// retain pair order. Coincident points remain distinct vertices joined by zero edges.
/// Empty/singleton input is connected with zero total. Coordinates are arbitrary
/// Cartesian units and outputs use those same units; this is not geographic distance,
/// a tour or a Steiner tree. The complete graph always connects finite valid inputs.
/// O(n² log(n+1)) time and O(n²+n) runtime memory; no Delaunay sparsification is used.
/// All coordinates and pair/container geometry are checked before generated allocation.
/// Independent calls share no mutable state; concurrent caller mutation is unsafe.
///
/// @param points Borrowed finite planar vertices; empty, singleton and duplicate coordinates are supported.
/// @return Owned minimum tree and work, without optional trajectory or quota.
/// @throws std::invalid_argument If any coordinate is nonfinite, including a singleton.
/// @throws std::length_error If pair/container/count geometry exceeds limits.
/// @throws std::overflow_error If a difference, pair distance or accepted-edge total is nonfinite.
/// @throws std::bad_alloc If required storage cannot be allocated.
///
/// @par Example
/// @code
/// const auto r=dm::algorithms::euclidean_minimum_spanning_tree_work({{0,0},{0,0},{1,0}});
/// std::print("{} {}\n",r.result.total_weight,r.work.distance_evaluations); // 1 3
/// @endcode
///
/// @par References
/// - J. B. Kruskal, Jr. (1956). On the shortest spanning subtree of a graph and the traveling salesman problem. Proceedings of the AMS 7(1), 48-50. https://doi.org/10.1090/S0002-9939-1956-0078686-7
///
/// @see EuclideanMstOutcome, euclidean_minimum_spanning_tree()
EuclideanMstOutcome euclidean_minimum_spanning_tree_work(const std::vector<Point2D>& points);

/// @brief Capture the actual complete-graph Euclidean Kruskal execution with a record quota.
/// Constructs every unordered pair in lexicographic input-index order, weighted by
/// Euclidean hypot distance; then calls the shared stable Kruskal kernel. Equal weights
/// retain pair order. Coincident points remain distinct vertices joined by zero edges.
/// Empty/singleton input is connected with zero total. Coordinates are arbitrary
/// Cartesian units and outputs use those same units; this is not geographic distance,
/// a tour or a Steiner tree. The complete graph always connects finite valid inputs.
/// O(n² log(n+1)) time and O(n²+n) runtime memory; no Delaunay sparsification is used.
/// All coordinates and pair/container geometry are checked before generated allocation.
/// Independent calls share no mutable state; concurrent caller mutation is unsafe.
/// C=N+M+S follows EuclideanMstTrace. Known N+M capture is checked before
/// coordinate copies/edge generation; each actual scan is checked before capture.
/// The nested result is the result of the shared kernel, not reconstructed from steps.
///
/// @param points Borrowed finite planar vertices; empty, singleton and duplicate coordinates are supported.
/// @param max_entries Maximum captured records (default 1000000); zero disables only quota.
/// @return Owned points, actual edge/scan trace and nested result/work.
/// @throws std::invalid_argument If any coordinate is nonfinite, including a singleton.
/// @throws std::length_error If pair/container/count geometry exceeds limits.
/// @throws std::overflow_error If a difference, pair distance or accepted-edge total is nonfinite.
/// @throws std::bad_alloc If required storage cannot be allocated.
/// @throws std::length_error If known input capture or any next actual step exceeds max_entries.
///
/// @par Example
/// @code
/// const auto t=dm::algorithms::euclidean_minimum_spanning_tree_trace({{0,0},{3,4}},4);
/// std::print("{} {}\n",t.distance_evaluations,t.captured_entries); // 1 4
/// @endcode
///
/// @par References
/// - J. B. Kruskal, Jr. (1956). On the shortest spanning subtree of a graph and the traveling salesman problem. Proceedings of the AMS 7(1), 48-50. https://doi.org/10.1090/S0002-9939-1956-0078686-7
///
/// @see EuclideanMstTrace, kruskal_trace()
EuclideanMstTrace euclidean_minimum_spanning_tree_trace(const std::vector<Point2D>& points,std::size_t max_entries=1000000);

} // namespace datamunge::algorithms
