#pragma once
/// @file force_directed_layout.hpp
/// @brief Direct Fruchterman–Reingold variant, actual work, owned histories and diagnostics.
#include <datamunge/algorithms/layout_geometry.hpp>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>
namespace datamunge::algorithms {
/// @brief Actual kernel counts and final normalization metadata.
/// Counts exclude validation, allocation, capture copies and optional metrics. Uniform values count distribution invocations, not raw engine words. Inactive n<2 returns without normalization.
/// @par Example
/// @code
/// dm::algorithms::ForceDirectedWork x{};
/// std::cout << x.vertices; // 0
/// @endcode
/// @see force_directed_layout_trace()
struct ForceDirectedWork {
    std::size_t vertices{}; ///< Vertex count.
    std::size_t edge_records{}; ///< Input records including loops and duplicates.
    std::size_t nonloop_edges{}; ///< Input records with distinct endpoints, including repetitions.
    std::size_t iterations_requested{}; ///< Requested sweeps, even when n<2.
    std::size_t completed_iterations{}; ///< Actual complete sweeps; zero when n<2.
    std::size_t pair_visits{}; ///< Actual unordered repulsion pair visits.
    std::size_t edge_visits{}; ///< Actual edge loop visits, including ignored self loops.
    std::size_t attraction_visits{}; ///< Actual nonloop attraction updates.
    std::size_t vertex_moves{}; ///< Actual vertex update visits.
    std::size_t uniform_values{}; ///< Actual uniform variate invocations: 2n seeded, zero supplied.
    bool seeded{}; ///< Whether initialization used a local random engine.
    std::uint64_t seed{}; ///< Local engine seed; zero placeholder for supplied positions.
    double k{}; ///< 1/sqrt(n) for n>=2, zero placeholder otherwise.
    bool normalized{}; ///< Whether final centering/scaling ran, including zero iterations.
    Point2D mean{}; ///< Actual double mean subtracted; zero placeholder if inactive.
    double scale{}; ///< Maximum centered radius before division; zero if all centered points vanish or inactive.
};
/// @brief Owned ordinary coordinates with actual work.
/// No history is allocated; the shared kernel performs each sweep once.
/// @par Example
/// @code
/// dm::algorithms::ForceDirectedOutcome x{};
/// std::cout << x.work.vertices; // 0
/// @endcode
/// @see force_directed_layout_trace()
struct ForceDirectedOutcome {
    std::vector<Point2D> positions{}; ///< Owned final coordinates.
    ForceDirectedWork work{}; ///< Actual counts and normalization metadata.
};
/// @brief Owned state of one actual sweep.
/// Repulsion is captured before attraction; displacement is the actual net accumulator after attraction. Positions follow all vertex moves. No replay or separately rounded force sums.
/// @par Example
/// @code
/// dm::algorithms::ForceDirectedFrame x{};
/// std::cout << x.iteration; // 0
/// @endcode
/// @see force_directed_layout_trace()
struct ForceDirectedFrame {
    std::size_t iteration{}; ///< Zero-based sweep index.
    double temperature{}; ///< Actual 0.1*(1-double(iteration)/iterations).
    std::vector<Point2D> positions{}; ///< Owned raw coordinates after the sweep.
    std::vector<Point2D> repulsion{}; ///< Owned actual displacement immediately after pair repulsion.
    std::vector<Point2D> displacement{}; ///< Owned actual net displacement before moves.
};
/// @brief Owned initial, raw, normalized and per-sweep states.
/// All arrays own storage; caller mutation is independent. There are no frames for n<2 or zero sweeps. Final centering/scaling changes units and potential.
/// @par Example
/// @code
/// dm::algorithms::ForceDirectedTrace x{};
/// std::cout << x.frames.size(); // 0
/// @endcode
/// @see force_directed_layout_trace()
struct ForceDirectedTrace {
    std::vector<std::pair<std::size_t,std::size_t>> edges{}; ///< Owned input edge records in order.
    std::vector<Point2D> initial{}; ///< Owned initialized raw positions.
    std::vector<Point2D> raw_final{}; ///< Owned positions before final normalization.
    std::vector<Point2D> positions{}; ///< Owned ordinary output coordinates.
    std::vector<ForceDirectedFrame> frames{}; ///< Owned actual sweep states in order.
    ForceDirectedWork work{}; ///< Actual counts and normalization metadata.
};
/// @brief Independent widened geometry and regularized force potential.
/// Diagnostics use long double, O(n squared + m) work and O(1) auxiliary storage. They do not run the layout or mutate input. Potential is an integral of this implementation’s regularized law, not a monotonicity or optimality certificate.
/// @par Example
/// @code
/// dm::algorithms::ForceDirectedMetrics x{};
/// std::cout << x.vertices; // 0
/// @endcode
/// @see force_directed_layout_trace()
struct ForceDirectedMetrics {
    std::size_t vertices{}; ///< Number of points.
    std::size_t edge_records{}; ///< Input edge records.
    std::size_t active_edges{}; ///< Nonloop records counted with multiplicity.
    std::size_t pair_visits{}; ///< Unordered pairs measured.
    std::size_t clamped_pairs{}; ///< Pairs with raw distance below 1e-9.
    long double centroid_x{}; ///< Mean x; zero for empty input.
    long double centroid_y{}; ///< Mean y; zero for empty input.
    long double max_radius{}; ///< Maximum distance from the centroid; zero when empty.
    long double min_pair_distance{}; ///< Minimum raw pair distance; zero with fewer than two points.
    long double mean_edge_length{}; ///< Mean raw nonloop record distance; zero without active edges.
    long double repulsion_potential{}; ///< Sum of regularized repulsion integrals over all pairs.
    long double attraction_potential{}; ///< Sum of regularized attraction integrals over nonloop records.
    long double potential{}; ///< Repulsion plus attraction; raw and normalized units differ.
};
/// @brief Compute the historical seeded force-directed drawing.
/// The direct unit-area variant uses k=1/sqrt(n), distance floor 1e-9,
/// ordered u<v repulsion and input-order attraction (duplicates count; loops ignored).
/// All forces precede all moves: F/s*min(s,t), s=max(1e-9,hypot(F)),
/// t=0.1*(1-double(i)/T). Coincident points receive no jitter. There is no
/// bounding box, convergence test, velocity or energy line search. For n>=2,
/// final double centering and maximum-radius scaling run even for T=0.
/// For n<2, return initialized positions unchanged and perform no sweeps.
/// All endpoints and supplied finite coordinates are validated before allocation
/// or optional capture admission. Actual nonfinite arithmetic throws without
/// a partial result. No caller mutation or shared state. Counts and capacities
/// are checked; ordinary auxiliary storage is O(n), time O(n+m+T*(n*n+m+n))
/// for n>=2. Seeded coordinates use a local mt19937_64 and two uniform(-.5,.5)
/// variates per vertex; exact seeded bits also depend on the standard library.
/// @param vertex_count Number of vertices, possibly zero.
/// @param edges Undirected endpoint records; repetitions retained.
/// @param iterations Requested sweeps, default500.
/// @param seed Local engine seed, default42.
/// @return Owned coordinates in vertex order.
/// @throws std::invalid_argument Invalid endpoint or nonfinite supplied coordinate.
/// @throws std::length_error Unrepresentable work/container count or exceeded capture cap.
/// @throws std::overflow_error Nonfinite actual double arithmetic.
/// @throws std::bad_alloc Storage allocation fails.
/// @par Example
/// @code
/// const auto p=dm::algorithms::force_directed_layout(0,{});
/// std::cout << p.size(); // 0
/// @endcode
/// @par References
/// - Fruchterman and Reingold (1991), Graph Drawing by Force-directed Placement, doi:10.1002/spe.4380211102; https://www.reingold.co/force-directed.pdf; direct variant without paper frame clamping or jitter.
/// @see force_directed_layout_metrics(), force_directed_layout_trace_entries()
std::vector<Point2D> force_directed_layout(std::size_t vertex_count, const std::vector<std::pair<std::size_t,std::size_t>>& edges, std::size_t iterations=500, std::uint64_t seed=42);
/// @brief Start a fresh cooling schedule from supplied raw positions.
/// The direct unit-area variant uses k=1/sqrt(n), distance floor 1e-9,
/// ordered u<v repulsion and input-order attraction (duplicates count; loops ignored).
/// All forces precede all moves: F/s*min(s,t), s=max(1e-9,hypot(F)),
/// t=0.1*(1-double(i)/T). Coincident points receive no jitter. There is no
/// bounding box, convergence test, velocity or energy line search. For n>=2,
/// final double centering and maximum-radius scaling run even for T=0.
/// For n<2, return initialized positions unchanged and perform no sweeps.
/// All endpoints and supplied finite coordinates are validated before allocation
/// or optional capture admission. Actual nonfinite arithmetic throws without
/// a partial result. No caller mutation or shared state. Counts and capacities
/// are checked; ordinary auxiliary storage is O(n), time O(n+m+T*(n*n+m+n))
/// for n>=2. Seeded coordinates use a local mt19937_64 and two uniform(-.5,.5)
/// variates per vertex; exact seeded bits also depend on the standard library.
/// @param initial Finite raw coordinates; never mutated.
/// @param edges Undirected endpoint records in order.
/// @param iterations Fresh schedule length, default500; not continuation of an old trace.
/// @return Owned final coordinates.
/// @throws std::invalid_argument Invalid endpoint or nonfinite supplied coordinate.
/// @throws std::length_error Unrepresentable work/container count or exceeded capture cap.
/// @throws std::overflow_error Nonfinite actual double arithmetic.
/// @throws std::bad_alloc Storage allocation fails.
/// @par Example
/// @code
/// const auto p=dm::algorithms::force_directed_layout_from({{3,4}},{});
/// std::cout << p[0].x; // 3
/// @endcode
/// @par References
/// - Fruchterman and Reingold (1991), Graph Drawing by Force-directed Placement, doi:10.1002/spe.4380211102; https://www.reingold.co/force-directed.pdf; direct variant without paper frame clamping or jitter.
/// @see force_directed_layout_metrics(), force_directed_layout_trace_entries()
std::vector<Point2D> force_directed_layout_from(const std::vector<Point2D>& initial, const std::vector<std::pair<std::size_t,std::size_t>>& edges, std::size_t iterations=500);
/// @brief Measure the executing seeded layout without history.
/// The direct unit-area variant uses k=1/sqrt(n), distance floor 1e-9,
/// ordered u<v repulsion and input-order attraction (duplicates count; loops ignored).
/// All forces precede all moves: F/s*min(s,t), s=max(1e-9,hypot(F)),
/// t=0.1*(1-double(i)/T). Coincident points receive no jitter. There is no
/// bounding box, convergence test, velocity or energy line search. For n>=2,
/// final double centering and maximum-radius scaling run even for T=0.
/// For n<2, return initialized positions unchanged and perform no sweeps.
/// All endpoints and supplied finite coordinates are validated before allocation
/// or optional capture admission. Actual nonfinite arithmetic throws without
/// a partial result. No caller mutation or shared state. Counts and capacities
/// are checked; ordinary auxiliary storage is O(n), time O(n+m+T*(n*n+m+n))
/// for n>=2. Seeded coordinates use a local mt19937_64 and two uniform(-.5,.5)
/// variates per vertex; exact seeded bits also depend on the standard library.
/// @param vertex_count Vertex count.
/// @param edges Endpoint records.
/// @param iterations Sweep count, default500.
/// @param seed Local seed, default42.
/// @return Owned coordinates and actual work.
/// @throws std::invalid_argument Invalid endpoint or nonfinite supplied coordinate.
/// @throws std::length_error Unrepresentable work/container count or exceeded capture cap.
/// @throws std::overflow_error Nonfinite actual double arithmetic.
/// @throws std::bad_alloc Storage allocation fails.
/// @par Example
/// @code
/// const auto o=dm::algorithms::force_directed_layout_work(2,{{0,1}},2);
/// std::cout << o.work.pair_visits; // 2
/// @endcode
/// @par References
/// - Fruchterman and Reingold (1991), Graph Drawing by Force-directed Placement, doi:10.1002/spe.4380211102; https://www.reingold.co/force-directed.pdf; direct variant without paper frame clamping or jitter.
/// @see force_directed_layout_metrics(), force_directed_layout_trace_entries()
ForceDirectedOutcome force_directed_layout_work(std::size_t vertex_count, const std::vector<std::pair<std::size_t,std::size_t>>& edges, std::size_t iterations=500, std::uint64_t seed=42);
/// @brief Measure a fresh supplied-position run without history.
/// The direct unit-area variant uses k=1/sqrt(n), distance floor 1e-9,
/// ordered u<v repulsion and input-order attraction (duplicates count; loops ignored).
/// All forces precede all moves: F/s*min(s,t), s=max(1e-9,hypot(F)),
/// t=0.1*(1-double(i)/T). Coincident points receive no jitter. There is no
/// bounding box, convergence test, velocity or energy line search. For n>=2,
/// final double centering and maximum-radius scaling run even for T=0.
/// For n<2, return initialized positions unchanged and perform no sweeps.
/// All endpoints and supplied finite coordinates are validated before allocation
/// or optional capture admission. Actual nonfinite arithmetic throws without
/// a partial result. No caller mutation or shared state. Counts and capacities
/// are checked; ordinary auxiliary storage is O(n), time O(n+m+T*(n*n+m+n))
/// for n>=2. Seeded coordinates use a local mt19937_64 and two uniform(-.5,.5)
/// variates per vertex; exact seeded bits also depend on the standard library.
/// @param initial Finite raw coordinates, not mutated.
/// @param edges Endpoint records.
/// @param iterations Fresh schedule length, default500.
/// @return Owned coordinates and actual work.
/// @throws std::invalid_argument Invalid endpoint or nonfinite supplied coordinate.
/// @throws std::length_error Unrepresentable work/container count or exceeded capture cap.
/// @throws std::overflow_error Nonfinite actual double arithmetic.
/// @throws std::bad_alloc Storage allocation fails.
/// @par Example
/// @code
/// const auto o=dm::algorithms::force_directed_layout_work_from({{0,0},{1,0}},{{0,1}},2);
/// std::cout << o.work.uniform_values; // 0
/// @endcode
/// @par References
/// - Fruchterman and Reingold (1991), Graph Drawing by Force-directed Placement, doi:10.1002/spe.4380211102; https://www.reingold.co/force-directed.pdf; direct variant without paper frame clamping or jitter.
/// @see force_directed_layout_metrics(), force_directed_layout_trace_entries()
ForceDirectedOutcome force_directed_layout_work_from(const std::vector<Point2D>& initial, const std::vector<std::pair<std::size_t,std::size_t>>& edges, std::size_t iterations=500);
/// @brief Capture every actual seeded sweep without replay.
/// The direct unit-area variant uses k=1/sqrt(n), distance floor 1e-9,
/// ordered u<v repulsion and input-order attraction (duplicates count; loops ignored).
/// All forces precede all moves: F/s*min(s,t), s=max(1e-9,hypot(F)),
/// t=0.1*(1-double(i)/T). Coincident points receive no jitter. There is no
/// bounding box, convergence test, velocity or energy line search. For n>=2,
/// final double centering and maximum-radius scaling run even for T=0.
/// For n<2, return initialized positions unchanged and perform no sweeps.
/// All endpoints and supplied finite coordinates are validated before allocation
/// or optional capture admission. Actual nonfinite arithmetic throws without
/// a partial result. No caller mutation or shared state. Counts and capacities
/// are checked; ordinary auxiliary storage is O(n), time O(n+m+T*(n*n+m+n))
/// for n>=2. Seeded coordinates use a local mt19937_64 and two uniform(-.5,.5)
/// variates per vertex; exact seeded bits also depend on the standard library.
/// The whole upper entry count is checked before RNG or input copies; see force_directed_layout_trace_entries().
/// @param vertex_count Vertex count.
/// @param edges Endpoint records copied after admission.
/// @param iterations Requested sweeps, default500.
/// @param seed Local seed, default42.
/// @param max_entries Whole entry cap, default1million; zero disables only the cap.
/// @return Fully owned trace without truncation or partial results.
/// @throws std::invalid_argument Invalid endpoint or nonfinite supplied coordinate.
/// @throws std::length_error Unrepresentable work/container count or exceeded capture cap.
/// @throws std::overflow_error Nonfinite actual double arithmetic.
/// @throws std::bad_alloc Storage allocation fails.
/// @par Example
/// @code
/// const auto t=dm::algorithms::force_directed_layout_trace(2,{{0,1}},2);
/// std::cout << t.frames.size(); // 2
/// @endcode
/// @par References
/// - Fruchterman and Reingold (1991), Graph Drawing by Force-directed Placement, doi:10.1002/spe.4380211102; https://www.reingold.co/force-directed.pdf; direct variant without paper frame clamping or jitter.
/// @see force_directed_layout_metrics(), force_directed_layout_trace_entries()
ForceDirectedTrace force_directed_layout_trace(std::size_t vertex_count, const std::vector<std::pair<std::size_t,std::size_t>>& edges, std::size_t iterations=500, std::uint64_t seed=42, std::size_t max_entries=1000000);
/// @brief Capture a fresh schedule from supplied raw coordinates.
/// The direct unit-area variant uses k=1/sqrt(n), distance floor 1e-9,
/// ordered u<v repulsion and input-order attraction (duplicates count; loops ignored).
/// All forces precede all moves: F/s*min(s,t), s=max(1e-9,hypot(F)),
/// t=0.1*(1-double(i)/T). Coincident points receive no jitter. There is no
/// bounding box, convergence test, velocity or energy line search. For n>=2,
/// final double centering and maximum-radius scaling run even for T=0.
/// For n<2, return initialized positions unchanged and perform no sweeps.
/// All endpoints and supplied finite coordinates are validated before allocation
/// or optional capture admission. Actual nonfinite arithmetic throws without
/// a partial result. No caller mutation or shared state. Counts and capacities
/// are checked; ordinary auxiliary storage is O(n), time O(n+m+T*(n*n+m+n))
/// for n>=2. Seeded coordinates use a local mt19937_64 and two uniform(-.5,.5)
/// variates per vertex; exact seeded bits also depend on the standard library.
/// The whole upper entry count is checked before any input copy.
/// @param initial Finite raw coordinates, fully validated before admission.
/// @param edges Endpoint records copied after admission.
/// @param iterations Fresh schedule length, default500.
/// @param max_entries Whole entry cap, default1million; zero disables only cap.
/// @return Fully owned actual trace.
/// @throws std::invalid_argument Invalid endpoint or nonfinite supplied coordinate.
/// @throws std::length_error Unrepresentable work/container count or exceeded capture cap.
/// @throws std::overflow_error Nonfinite actual double arithmetic.
/// @throws std::bad_alloc Storage allocation fails.
/// @par Example
/// @code
/// const auto t=dm::algorithms::force_directed_layout_trace_from({{0,0},{1,0}},{{0,1}},1);
/// std::cout << t.frames.size(); // 1
/// @endcode
/// @par References
/// - Fruchterman and Reingold (1991), Graph Drawing by Force-directed Placement, doi:10.1002/spe.4380211102; https://www.reingold.co/force-directed.pdf; direct variant without paper frame clamping or jitter.
/// @see force_directed_layout_metrics(), force_directed_layout_trace_entries()
ForceDirectedTrace force_directed_layout_trace_from(const std::vector<Point2D>& initial, const std::vector<std::pair<std::size_t,std::size_t>>& edges, std::size_t iterations=500, std::size_t max_entries=1000000);
/// @brief Forecast whole trace entry records without allocation or RNG.
/// E+3n+S*(1+3n), S=T if n>=2 and0 otherwise. Counts edge, point and frame
/// records, not bytes or peak RAM; temporary kernel scratch is excluded.
/// Endpoint validity cannot be checked from counts alone. Zero optional cap on
/// trace calls still retains these representability/container checks.
/// @param vertex_count Number of vertices.
/// @param edge_count Number of input records, including loops.
/// @param iterations Requested sweeps, default500.
/// @return Whole owned entry count upper bound.
/// @throws std::length_error Arithmetic or required container capacity is unrepresentable.
/// @par Example
/// @code
/// std::cout << dm::algorithms::force_directed_layout_trace_entries(2,1,2); // 21
/// @endcode
/// @see force_directed_layout_trace()
std::size_t force_directed_layout_trace_entries(std::size_t vertex_count, std::size_t edge_count, std::size_t iterations=500);
/// @brief Measure geometry and a regularized potential in the supplied coordinate units.
/// For delta=double(1e-9), k=double(1/sqrt(n)), repulsion integral is
/// -k²log(d) for d>=delta, otherwise -k²log(delta)+k²/2*(1-(d/delta)²).
/// Edge integral is d³/(3k) above delta, otherwise delta*d²/(2k)-delta³/(6k).
/// Both are C1 at delta and yield this kernel's underlying regularized forces.
/// Finite arithmetic is approximate; no exact gradient, descent or drawing-quality
/// certificate is asserted. Query raw frames for a trajectory; final normalization
/// changes fixed-k potential. Empty/singleton interactions contribute zero.
/// All coordinates/endpoints are checked before measurement. Long-double queries
/// are independent, O(n²+m) work/O(1) auxiliary space, thread safe, no mutation.
/// @param points Finite raw or normalized coordinates in vertex order.
/// @param edges Endpoint records; loops ignored, repetitions count.
/// @return Owned geometry, counts and potential components.
/// @throws std::invalid_argument Nonfinite coordinate or endpoint out of range.
/// @throws std::length_error Unrepresentable pair count.
/// @throws std::overflow_error Nonfinite widened diagnostic arithmetic.
/// @par Example
/// @code
/// const auto m=dm::algorithms::force_directed_layout_metrics({{0,0},{1,0}},{{0,1}});
/// std::cout << m.mean_edge_length; // 1
/// @endcode
/// @par References
/// - Fruchterman and Reingold (1991), Graph Drawing by Force-directed Placement, doi:10.1002/spe.4380211102; https://www.reingold.co/force-directed.pdf; potential integrates this library's distance-floor variant.
/// @see ForceDirectedMetrics, force_directed_layout_trace()
ForceDirectedMetrics force_directed_layout_metrics(const std::vector<Point2D>& points, const std::vector<std::pair<std::size_t,std::size_t>>& edges);
/// @brief Return the original report's four-vertex path records.
/// Owned deterministic fixture: 0--1--2--3; no random engine or shared state.
/// @return Three independent endpoint records.
/// @throws std::bad_alloc Allocation fails.
/// @par Example
/// @code
/// std::cout << dm::algorithms::force_directed_layout_example_edges().size(); // 3
/// @endcode
/// @see force_directed_layout()
std::vector<std::pair<std::size_t,std::size_t>> force_directed_layout_example_edges();
} // namespace datamunge::algorithms
