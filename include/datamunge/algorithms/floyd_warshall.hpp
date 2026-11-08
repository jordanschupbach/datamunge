#pragma once
/// @file floyd_warshall.hpp
/// @brief Checked in-place all-pairs dynamic programming and bounded path reconstruction.
#include <cstddef>
#include <tuple>
#include <vector>
namespace datamunge::algorithms {
/// @brief No-next-hop sentinel, equal to SIZE_MAX; normally used on diagonal/unreachable pairs.
/// Native negative-cycle raw matrices may contain diagonal next hops.
/// @par Example
/// @code
/// std::cout << (dm::algorithms::kNoVertex == std::size_t(-1)); // 1
/// @endcode
/// @see AllPairsShortestPaths
inline constexpr std::size_t kNoVertex=static_cast<std::size_t>(-1);
/// @brief Owned native distance/next-hop matrices with the historical member layout.
/// Native results have two n-by-n matrices. Unreachable distances are positive
/// infinity. A zero-length diagonal path costs0 unless affected by a negative cycle.
/// After negative-cycle detection, affected raw entries are not shortest distances;
/// they are retained to preserve the actual in-place computation. Independent
/// classification uses the matrices, not an externally supplied boolean flag.
/// @par Example
/// @code
/// const auto a=dm::algorithms::floyd_warshall(2,{{0,1,3.}});
/// std::cout << a.distance[0][1]; // 3
/// @endcode
/// @see floyd_warshall(), reconstruct_path()
struct AllPairsShortestPaths {
    std::vector<std::vector<double>> distance{}; ///< Raw distances; finite shortest costs only for unaffected reachable pairs.
    std::vector<std::vector<std::size_t>> next{}; ///< First hop, or kNoVertex; raw affected hops can cycle.
    bool has_negative_cycle{false}; ///< Whether the final native scan found any negative diagonal.
};
/// @brief Actual seeding, guarded sweep and final diagonal-scan counts.
/// Counts refer to the executing kernel, excluding input validation, allocations
/// and optional snapshot copying. Each finite candidate costs one addition and one
/// strict comparison; equal-cost candidates keep the existing next hop. All counts
/// are checked by representable full sweep bounds before matrix allocation.
/// @par Example
/// @code
/// dm::algorithms::FloydWarshallWork w{};
/// std::cout << w.candidates; // 0
/// @endcode
/// @see floyd_warshall_work()
struct FloydWarshallWork {
    std::size_t vertices{}; ///< Vertex count.
    std::size_t edges{}; ///< Input edge records, including parallels.
    std::size_t matrix_cells{}; ///< Cells in one n-by-n matrix.
    std::size_t seed_improvements{}; ///< Strict edge-seeding improvements.
    std::size_t row_checks{}; ///< Actual outer i-row reachability guards.
    std::size_t skipped_rows{}; ///< Rows skipped because distance[i][k] is positive infinity.
    std::size_t column_checks{}; ///< Actual inner j reachability guards.
    std::size_t skipped_columns{}; ///< Columns skipped because distance[k][j] is positive infinity.
    std::size_t candidates{}; ///< Finite candidate additions and strict comparisons.
    std::size_t relaxations{}; ///< Strict successful sweep improvements, excluding seeds.
    std::size_t diagonal_checks{}; ///< Final negative-diagonal tests, stopping at the first negative.
};
/// @brief Owned ordinary matrices and actual work without history.
/// Uses the same seeding and k/i/j kernel as the original API; no replay.
/// @par Example
/// @code
/// dm::algorithms::FloydWarshallOutcome o{};
/// std::cout << o.work.vertices; // 0
/// @endcode
/// @see floyd_warshall_work()
struct FloydWarshallOutcome {
    AllPairsShortestPaths result{}; ///< Owned raw result and global negative-cycle flag.
    FloydWarshallWork work{}; ///< Actual kernel measurements.
};
/// @brief Owned actual matrices at one admitted-intermediate boundary.
/// Admitted0 is the seeded state. Admitted k+1 follows the complete k sweep.
/// Snapshots are raw execution states; affected negative-cycle entries do not have
/// shortest-path meaning. No negative-cycle flag is fabricated before its final scan.
/// @par Example
/// @code
/// dm::algorithms::FloydWarshallStage s{};
/// std::cout << s.admitted; // 0
/// @endcode
/// @see floyd_warshall_trace()
struct FloydWarshallStage {
    std::size_t admitted{}; ///< Number of admitted intermediates,0..n.
    std::vector<std::vector<double>> distance{}; ///< Owned actual distance matrix at this boundary.
    std::vector<std::vector<std::size_t>> next{}; ///< Owned actual next-hop matrix at this boundary.
};
/// @brief One actual strict improvement in k/i/j execution order.
/// The left/right entries are read immediately before the update; in a negative
/// cycle sweep they can change within that same k stage. A record is not a failed
/// comparison or an edge seed. The selected hop is the actual next[from][k].
/// @par Example
/// @code
/// dm::algorithms::FloydWarshallRelaxation r{};
/// std::cout << (r.next==dm::algorithms::kNoVertex); // 1
/// @endcode
/// @see FloydWarshallTrace
struct FloydWarshallRelaxation {
    std::size_t k{}; ///< Currently admitted intermediate vertex.
    std::size_t from{}; ///< Source row.
    std::size_t to{}; ///< Destination column.
    double previous{}; ///< Replaced distance, possibly positive infinity.
    double left{}; ///< Actual distance[from][k] used in the addition.
    double right{}; ///< Actual distance[k][to] used in the addition.
    double candidate{}; ///< Actual finite double sum, the new distance.
    std::size_t previous_next{kNoVertex}; ///< Replaced first hop.
    std::size_t next{kNoVertex}; ///< Actual newly selected first hop.
};
/// @brief Owned graph, raw matrices and evidence captured during the executing sweep.
/// Contains seeded and every post-k matrix, plus only strict successful improvements.
/// No input views, rerun or independent alternative algorithm is retained.
/// @par Example
/// @code
/// const auto t=dm::algorithms::floyd_warshall_trace(2,{{0,1,3.}});
/// std::cout << t.stages.size(); // 3
/// @endcode
/// @see floyd_warshall_trace()
struct FloydWarshallTrace {
    std::vector<std::tuple<std::size_t,std::size_t,double>> edges{}; ///< Owned original directed edge records in order.
    AllPairsShortestPaths result{}; ///< Owned final raw matrices and cycle flag.
    FloydWarshallWork work{}; ///< Actual kernel counts.
    std::vector<FloydWarshallStage> stages{}; ///< Exactly n+1 owned actual stage boundaries.
    std::vector<FloydWarshallRelaxation> relaxations{}; ///< Actual strict improvements; at most n cubed.
};
/// @brief Meaning of one ordered pair after the negative-cycle reachability test.
/// This classification does not rewrite raw matrices to negative infinity.
/// @par Example
/// @code
/// std::cout << (dm::algorithms::FloydWarshallPairStatus::Finite == dm::algorithms::FloydWarshallPairStatus::Finite); // 1
/// @endcode
/// @see floyd_warshall_pair_status()
enum class FloydWarshallPairStatus {
    Finite, ///< Reachable and unaffected by a detected negative diagonal.
    Unreachable, ///< Raw distance is positive infinity and no negative cycle connects the pair.
    Unbounded ///< A reachable negative-cycle vertex can also reach the destination.
};
/// @brief Compute all-pairs directed distances and next hops using Floyd's native recurrence.
/// Validates all endpoints and finite edge weights before allocating. Keeps the
/// lightest parallel edge using strict comparisons; equal weights retain the first
/// hop. Diagonal initialization is zero, positive self-edges are ignored, and
/// negative self-edges are retained. Sweep order is k/i/j with infinity guards.
/// Every attempted finite sum must remain finite, including non-improving ones.
/// Floating additions/comparisons and strict negative-diagonal tests follow double
/// arithmetic; they do not certify exact-real shortest costs near rounding limits.
/// After all sweeps, the first negative diagonal sets the global cycle flag.
/// Raw matrices remain unchanged for affected pairs; use pair classification.
/// No mutation of inputs or shared state. O(n squared+edges+n cubed) worst-case
/// work and O(n squared) owned matrix storage. No ordinary history quota.
/// @param n Vertex count; empty zero-vertex graph is valid.
/// @param edges Directed(from,to,weight) records; endpoints below n, weights finite.
/// @return Owned historical result layout; affected negative-cycle entries are raw.
/// @throws std::invalid_argument Endpoint out of range or any nonfinite weight.
/// @throws std::length_error Matrix/work product or container capacity exhausted.
/// @throws std::overflow_error An attempted double sum becomes nonfinite.
/// @throws std::bad_alloc Matrix allocation fails.
/// @par Example
/// @code
/// const auto a=dm::algorithms::floyd_warshall(3,{{0,1,2.},{1,2,3.},{0,2,9.}});
/// std::cout << a.distance[0][2]; // 5
/// @endcode
/// @par References
/// - R.W.Floyd (1962), Algorithm97: Shortest path, Communications of the ACM5(6),345,
///   doi:10.1145/367766.368168; weighted in-place intermediate-vertex recurrence.
/// - S.Warshall (1962), A Theorem on Boolean Matrices, Journal of the ACM9(1),11-12,
///   doi:10.1145/321105.321107; related Boolean reachability recurrence.
/// @see floyd_warshall_work(), floyd_warshall_trace(), reconstruct_path()
AllPairsShortestPaths floyd_warshall(std::size_t n,const std::vector<std::tuple<std::size_t,std::size_t,double>>& edges);
/// @brief Return the ordinary matrices with actual guarded-kernel work counts.
/// Validates all endpoints and finite edge weights before allocating. Keeps the
/// lightest parallel edge using strict comparisons; equal weights retain the first
/// hop. Diagonal initialization is zero, positive self-edges are ignored, and
/// negative self-edges are retained. Sweep order is k/i/j with infinity guards.
/// Every attempted finite sum must remain finite, including non-improving ones.
/// Floating additions/comparisons and strict negative-diagonal tests follow double
/// arithmetic; they do not certify exact-real shortest costs near rounding limits.
/// After all sweeps, the first negative diagonal sets the global cycle flag.
/// Raw matrices remain unchanged for affected pairs; use pair classification.
/// No mutation of inputs or shared state. O(n squared+edges+n cubed) worst-case
/// work and O(n squared) owned matrix storage. No ordinary history quota.
/// @param n Vertex count, including zero.
/// @param edges Borrowed finite directed edge records with valid endpoints.
/// @return Independent owned ordinary result and actual seeding/sweep/scan counts.
/// @throws std::invalid_argument Invalid endpoints or nonfinite weights.
/// @throws std::length_error Matrix/work product or container capacity exhausted.
/// @throws std::overflow_error An attempted sum becomes nonfinite.
/// @throws std::bad_alloc Matrix allocation fails.
/// @par Example
/// @code
/// const auto o=dm::algorithms::floyd_warshall_work(1,{});
/// std::cout << o.work.candidates << ' ' << o.work.diagonal_checks; // 1 1
/// @endcode
/// @par References
/// - R.W.Floyd (1962), Algorithm97: Shortest path, Communications of the ACM5(6),345,
///   doi:10.1145/367766.368168; weighted in-place intermediate-vertex recurrence.
/// - S.Warshall (1962), A Theorem on Boolean Matrices, Journal of the ACM9(1),11-12,
///   doi:10.1145/321105.321107; related Boolean reachability recurrence.
/// @see floyd_warshall(), FloydWarshallWork
FloydWarshallOutcome floyd_warshall_work(std::size_t n,const std::vector<std::tuple<std::size_t,std::size_t,double>>& edges);
/// @brief Compute the complete upper entry bound of an owned full-stage trace.
/// With A=2n+2n squared, returns edges+A+(n+1)*(1+A)+n cubed: owned edge tuples,
/// final matrix row/cell entries, stage objects and their row/cell entries, and
/// at most n cubed improvement records. This is not bytes or peak RAM. Pure
/// checked arithmetic, no allocation or graph access; empty graph bound is1.
/// @param n Vertex count used for matrix/stage dimensions.
/// @param edge_count Number of owned edge-tuple entries.
/// @return Representable whole upper entry count.
/// @throws std::length_error Any intermediate sum/product is unrepresentable.
/// @par Example
/// @code
/// std::cout << dm::algorithms::floyd_warshall_trace_entries(2,1); // 60
/// @endcode
/// @see floyd_warshall_trace()
std::size_t floyd_warshall_trace_entries(std::size_t n,std::size_t edge_count);
/// @brief Record actual seeded/post-k matrices and strict improvements without replay.
/// Validates all endpoints and finite edge weights before allocating. Keeps the
/// lightest parallel edge using strict comparisons; equal weights retain the first
/// hop. Diagonal initialization is zero, positive self-edges are ignored, and
/// negative self-edges are retained. Sweep order is k/i/j with infinity guards.
/// Every attempted finite sum must remain finite, including non-improving ones.
/// Floating additions/comparisons and strict negative-diagonal tests follow double
/// arithmetic; they do not certify exact-real shortest costs near rounding limits.
/// After all sweeps, the first negative diagonal sets the global cycle flag.
/// Raw matrices remain unchanged for affected pairs; use pair classification.
/// No mutation of inputs or shared state. O(n squared+edges+n cubed) worst-case
/// work and O(n squared) owned matrix storage. No ordinary history quota.
/// Full input validation and the complete upper bound from trace_entries() precede
/// copies and kernel execution. Default cap is one million entries; zero disables
/// only that cap. Checked arithmetic/container limits remain. This conservative
/// bound can reject a graph with few actual improvements. Actual records contain
/// every stage and only successful updates, without truncation or partial results.
/// O(edges+n squared+n cubed) work/storage for owned full-stage evidence.
/// @param n Vertex count, including zero.
/// @param edges Borrowed finite directed edge records; copied only after admission.
/// @param max_entries Complete upper entry cap, default one million; zero unlimited.
/// @return Independent graph, actual stages/improvements, raw result and counts.
/// @throws std::invalid_argument Invalid endpoints or nonfinite weights, before quota checks.
/// @throws std::length_error Whole bound/cap or a container limit exhausted.
/// @throws std::overflow_error Any attempted double candidate becomes nonfinite.
/// @throws std::bad_alloc Owned matrices or records fail allocation.
/// @par Example
/// @code
/// const auto t=dm::algorithms::floyd_warshall_trace(2,{{0,1,3.}},60);
/// std::cout << t.result.distance[0][1] << ' ' << t.stages.size(); // 3 3
/// @endcode
/// @par References
/// - R.W.Floyd (1962), Algorithm97: Shortest path, Communications of the ACM5(6),345,
///   doi:10.1145/367766.368168; weighted in-place intermediate-vertex recurrence.
/// - S.Warshall (1962), A Theorem on Boolean Matrices, Journal of the ACM9(1),11-12,
///   doi:10.1145/321105.321107; related Boolean reachability recurrence.
/// @see floyd_warshall_trace_entries(), FloydWarshallTrace
FloydWarshallTrace floyd_warshall_trace(std::size_t n,const std::vector<std::tuple<std::size_t,std::size_t,double>>& edges,
    std::size_t max_entries=1000000);
/// @brief Classify a pair as finite, unreachable or negative-cycle unbounded.
/// Validates complete square distance/next matrices, finite-or-positive-infinity
/// distances and every hop ID before querying. Unbounded means some k has a
/// negative diagonal and finite i-to-k and k-to-j entries. The supplied global
/// boolean is not trusted as a substitute. Unaffected pairs remain meaningful
/// despite another component's cycle. External matrices are not certified shortest
/// paths by shape/value validation. O(n squared) validation plus O(n) querying;
/// no allocation, mutation or shared state.
/// @param apsp Read-only native result or externally supplied matrix record.
/// @param i Source vertex, below matrix dimension.
/// @param j Destination vertex, below matrix dimension.
/// @return Pair classification from validated raw matrices.
/// @throws std::invalid_argument Invalid matrices, distance/hop values or endpoints.
/// @par Example
/// @code
/// const auto a=dm::algorithms::floyd_warshall(2,{});
/// std::cout << (dm::algorithms::floyd_warshall_pair_status(a,0,1)==dm::algorithms::FloydWarshallPairStatus::Unreachable); // 1
/// @endcode
/// @see floyd_warshall_unbounded_pairs(), reconstruct_path()
FloydWarshallPairStatus floyd_warshall_pair_status(const AllPairsShortestPaths& apsp,std::size_t i,std::size_t j);
/// @brief Mark all source/destination pairs connected through a negative-cycle vertex.
/// Complete matrix validation matches pair_status(), performed once. Computes an
/// owned n-by-n Boolean mask in O(n cubed) worst-case work and O(n squared) storage.
/// Does not modify raw costs or trust the supplied global boolean flag.
/// @param apsp Read-only native/external result with valid matrix shapes/values.
/// @return True iff a negative diagonal can be reached from the row and reach the column.
/// @throws std::invalid_argument Invalid matrix shapes, distance values or hop IDs.
/// @throws std::length_error Container capacity exhausted.
/// @throws std::bad_alloc Mask allocation fails.
/// @par Example
/// @code
/// const auto a=dm::algorithms::floyd_warshall(1,{{0,0,-1.}});
/// std::cout << dm::algorithms::floyd_warshall_unbounded_pairs(a)[0][0]; // 1
/// @endcode
/// @see floyd_warshall_pair_status()
std::vector<std::vector<bool>> floyd_warshall_unbounded_pairs(const AllPairsShortestPaths& apsp);
/// @brief Follow validated next hops with a vertex-count bound and negative-cycle checks.
/// Full matrix validation matches pair_status() before diagonal/no-path shortcuts.
/// An affected pair throws even on the diagonal; unrelated negative cycles do not
/// prevent finite/unreachable queries. Unreachable or a broken sentinel chain
/// returns empty; a finite diagonal returns its single vertex. Cyclic malformed
/// hop chains throw after at most n moves, preventing unbounded walks. Validation
/// does not certify an external record's weights or optimality. O(n squared+path
/// length) work due to full validation; O(path length) returned storage.
/// @param apsp Read-only native/external result with valid shapes and values.
/// @param i Source vertex, below the distance dimension.
/// @param j Destination vertex, below the distance dimension.
/// @return Owned vertex sequence, or empty for unreachable/broken-sentinel chains.
/// @throws std::invalid_argument Invalid matrices/values/endpoints or cyclic hop chain.
/// @throws std::domain_error Pair is unbounded through a reachable negative cycle.
/// @throws std::length_error Path container capacity exhausted.
/// @throws std::bad_alloc Path allocation fails.
/// @par Example
/// @code
/// const auto a=dm::algorithms::floyd_warshall(3,{{0,1,2.},{1,2,3.}});
/// const auto p=dm::algorithms::reconstruct_path(a,0,2);
/// std::cout << p[0] << ' ' << p[1] << ' ' << p[2]; // 0 1 2
/// @endcode
/// @see floyd_warshall(), floyd_warshall_pair_status()
[[nodiscard]] std::vector<std::size_t> reconstruct_path(const AllPairsShortestPaths& apsp,std::size_t i,std::size_t j);
/// @brief Return the original nine-edge five-vertex report graph with negative edges.
/// Fresh owned data; no random state/dataset. No negative cycle, and0-to-2 has
/// cost-3 along0,4,3,2. Vertex count is five, supplied separately to the detector.
/// @return Owned(from,to,weight) tuples in original order.
/// @throws std::bad_alloc Edge allocation fails.
/// @par Example
/// @code
/// std::cout << dm::algorithms::floyd_warshall(5,dm::algorithms::floyd_warshall_example_edges()).distance[0][2]; // -3
/// @endcode
/// @see floyd_warshall()
std::vector<std::tuple<std::size_t,std::size_t,double>> floyd_warshall_example_edges();
} // namespace datamunge::algorithms
