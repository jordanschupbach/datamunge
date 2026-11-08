#pragma once
/// @file circle_packing.hpp
/// @brief Equal-centre relaxation, actual snapshots and measured disk contacts; no circle-packing theorem construction.
#include <datamunge/algorithms/layout_geometry.hpp>
#include <utility>
#include <vector>
namespace datamunge::algorithms {
/// @brief Explicit fixed-sweep controls and preflight work/record limits for the legacy relaxation.
/// Finite nonnegative step_size scales simultaneous pair displacements. There is
/// no adaptive step, line search or early convergence test. Empty/singleton inputs
/// execute no sweeps. snapshot_stride is positive and applies only to traced calls:
/// record the initial state, every stride, and the final state without duplication.
/// Zero caps mean unlimited. Positive caps reject the complete forecast before
/// adjacency/snapshot allocation; validation precedes those caps. Snapshot controls
/// do not limit untraced storage. The dense adjacency matrix remains O(V^2).
/// @par Example
/// @code
/// dm::algorithms::CirclePackingOptions o;o.iterations=1;o.snapshot_stride=1;
/// const auto t=dm::algorithms::circle_packing_trace(2,{{0,1}},o);
/// std::print("{} {}\n",t.work.completed_iterations,t.frames.size()); // 1 2
/// @endcode
/// @see circle_packing_trace(), circle_packing_layout_with_options()
struct CirclePackingOptions {
  std::size_t iterations{1000}; ///< Requested simultaneous sweeps; zero returns normalized initial positions.
  double step_size{0.08}; ///< Finite nonnegative fixed displacement factor; zero still executes requested sweeps.
  std::size_t max_pair_visits{0}; ///< Complete relaxation pair-inspection cap, excluding diagnostics; zero unlimited.
  std::size_t snapshot_stride{100}; ///< Positive recording interval; initial and final states are always included in a trace.
  std::size_t max_snapshots{0}; ///< Positive complete frame-count cap for tracing only; zero unlimited.
};
/// @brief Owned tolerance-based contact, overlap and edge-error diagnostics for supplied disks.
/// For each pair, gap=distance-radius_i-radius_j. Overlap means gap<-tolerance;
/// contact means |gap|<=tolerance. Contacts can therefore include tiny tolerated
/// overlaps. Edge stress is the sum of squared gaps on distinct nonloop edges;
/// overlap penalty is the sum of squared negative gaps on every pair, independent
/// of the overlap-count tolerance. Empty/singleton min_gap is zero; no-edge maximum
/// error/stress is zero. Zero radii are counted separately and are not disk-packing
/// certificates. These measurements do not establish planarity, optimality or an
/// exact-real contact representation. All numeric aggregates use long double.
/// @par Example
/// @code
/// const auto m=dm::algorithms::circle_packing_metrics({{{0,0},1},{{2,0},1}},{{0,1}});
/// std::print("{} {} {}\n",m.edge_contacts,m.overlapping_pairs,m.min_gap); // 1 0 0
/// @endcode
/// @see circle_packing_metrics(), CirclePackingTrace
struct CirclePackingMetrics {
  std::size_t circle_count{0}; ///< Number of validated disks, including zero-radius records.
  std::size_t unique_edges{0}; ///< Distinct unordered nonloop input edges.
  std::size_t pair_visits{0}; ///< Actual unordered pair measurements.
  std::size_t overlapping_pairs{0}; ///< Pairs with gap strictly below negative tolerance.
  std::size_t edge_contacts{0}; ///< Input edges whose absolute gap is within tolerance.
  std::size_t nonedge_contacts{0}; ///< Nonedges whose absolute gap is within tolerance.
  std::size_t zero_radii{0}; ///< Degenerate disks with radius exactly zero.
  long double min_gap{0}; ///< Minimum signed pair gap; zero by convention with fewer than two disks.
  long double max_edge_error{0}; ///< Maximum absolute gap on input edges; zero for no edges.
  long double edge_stress{0}; ///< Sum of squared edge gaps, counted once per distinct nonloop edge.
  long double overlap_penalty{0}; ///< Sum of squared negative gaps on all pairs, without tolerance clipping.
};
/// @brief Owned actual unnormalised working disks and measured relaxation energy at one snapshot.
/// Working disks have radius one. Raw edge target is two; nonedge repulsion uses
/// clearance target 2.15. energy=(edge_stress+nonedge_penalty)/2, where the latter
/// squares only positive clearance deficits. These are diagnostic measurements of
/// the captured state, not a replay or a promised decreasing objective. The force
/// distance floor 1e-9 can modify the gradient interpretation near coincidence.
/// max_displacement is the largest actual committed vertex movement in the last
/// sweep, zero for the initial snapshot. Postprocessed disks are stored separately.
/// @par Example
/// @code
/// dm::algorithms::CirclePackingOptions o;o.iterations=1;o.snapshot_stride=1;
/// const auto t=dm::algorithms::circle_packing_trace(2,{{0,1}},o);
/// std::print("{:.4f}\n",t.frames[1].energy); // 5.6448
/// @endcode
/// @see CirclePackingTrace
struct CirclePackingFrame {
  std::size_t iteration{0}; ///< Number of actual completed sweeps at capture; initial is zero.
  std::vector<Circle2D> circles; ///< Actual owned raw centres with unit working radii.
  long double edge_stress{0}; ///< Sum of (raw distance-2)^2 on distinct edges.
  long double nonedge_penalty{0}; ///< Sum of max(0,2.15-raw distance)^2 on nonedges.
  long double energy{0}; ///< Half the sum of the two measured penalties.
  long double max_displacement{0}; ///< Maximum actual movement in the last sweep; zero for initialization.
};
/// @brief Actual named validation, relaxation and diagnostic work, excluding sort/allocator costs.
/// Pair visits count all unordered pairs in every requested sweep for V>=2, even
/// when their force is zero. Position updates count actual vertex commits, including
/// unchanged values. Diagnostic scans are separate from the relaxation work cap.
/// Graph input is validated in full; loops are ignored and duplicate/reversed edges
/// collapse to one unweighted adjacency. Counts are not elapsed time.
/// @par Example
/// @code
/// dm::algorithms::CirclePackingOptions o;o.iterations=2;o.snapshot_stride=1;
/// const auto w=dm::algorithms::circle_packing_trace(2,{{0,1}},o).work;
/// std::print("{} {} {}\n",w.pair_visits,w.position_updates,w.frame_pair_visits); // 2 4 3
/// @endcode
/// @see CirclePackingTrace, CirclePackingOptions
struct CirclePackingWork {
  std::size_t validated_edges{0}; ///< Actual original endpoint-validation visits.
  std::size_t unique_edges{0}; ///< Distinct unordered nonloop adjacencies installed.
  std::size_t ignored_self_loops{0}; ///< Original loop inputs ignored after validation.
  std::size_t collapsed_duplicates{0}; ///< Repeated/reversed nonloop inputs collapsed by adjacency.
  std::size_t completed_iterations{0}; ///< Actual complete simultaneous sweeps; zero for V<2.
  std::size_t pair_visits{0}; ///< Actual relaxation pair inspections.
  std::size_t position_updates{0}; ///< Actual per-vertex committed updates.
  std::size_t frame_pair_visits{0}; ///< Actual raw snapshot-energy pair measurements.
  std::size_t output_pair_visits{0}; ///< Actual final disk-diagnostic pair measurements in traced calls.
};
/// @brief Owned actual relaxation snapshots, normalization map and returned fixed-radius disks.
/// Ordinary and traced entry points share the executing kernel without replay.
/// Raw snapshots precede postprocessing: subtract normalization_center, divide by
/// normalization_scale, then RESET every radius to 0.45. The radius is not scaled
/// with the centres. This can destroy tangencies or produce overlapping disks.
/// The name is legacy: no Koebe packing, planarity test or convergence guarantee
/// is supplied. Default storage alone performs no layout. Calls share no state.
/// @par Example
/// @code
/// dm::algorithms::CirclePackingOptions o;o.iterations=1;
/// const auto t=dm::algorithms::circle_packing_trace(2,{{0,1}},o);
/// std::print("{:.2f} {:.2f}\n",t.normalization_scale,t.circles[0].radius); // 2.68 0.45
/// @endcode
/// @see circle_packing_trace(), CirclePackingMetrics
struct CirclePackingTrace {
  CirclePackingOptions options; ///< Requested controls, including the fixed sweep count.
  std::vector<Circle2D> circles; ///< Returned normalized centres and reset radius 0.45.
  std::vector<CirclePackingFrame> frames; ///< Actual initial/stride/final raw states.
  Point2D normalization_center; ///< Rounded mean actually subtracted before scaling.
  long double normalization_scale{1}; ///< Maximum centred raw radius; one when that maximum is zero.
  CirclePackingMetrics output_metrics; ///< Actual returned-disk diagnostics at tolerance 1e-9.
  CirclePackingWork work; ///< Actual named work; diagnostic scans remain separate.
};
/// @brief Measure supplied disks' overlaps and graph contacts with an absolute distance tolerance.
/// Validates every finite centre, finite nonnegative radius, endpoint and tolerance
/// before measuring. Negative radii/nonfinite data reject; zero radii are allowed
/// and reported separately. Undirected loops are ignored; duplicate/reversed edges
/// count once. Empty input is supported if edges are empty. O(V^2+E) work and
/// O(V^2) dense adjacency storage. Widened distances/squares are checked for range.
/// No input mutation or shared state; results own all values.
/// @param circles Borrowed disks in original vertex order.
/// @param edges Borrowed unordered endpoint pairs, each endpoint in range.
/// @param tolerance Finite nonnegative absolute gap tolerance, default 1e-9.
/// @return Owned measurements, without a geometric/theorem certificate.
/// @throws std::invalid_argument If coordinates/radii/endpoints/tolerance are invalid.
/// @throws std::length_error If supported pair/matrix storage dimensions are exceeded.
/// @throws std::overflow_error If widened diagnostic arithmetic or counts overflow.
/// @throws std::bad_alloc If allocation fails.
/// @par Example
/// @code
/// const auto m=dm::algorithms::circle_packing_metrics({{{0,0},1},{{1,0},1}},{{0,1}});
/// std::print("{} {}\n",m.overlapping_pairs,m.min_gap); // 1 -1
/// @endcode
/// @see CirclePackingMetrics, circle_packing_trace()
CirclePackingMetrics circle_packing_metrics(const std::vector<Circle2D>& circles,const std::vector<std::pair<std::size_t,std::size_t>>& edges,double tolerance=1e-9);
/// @brief Run the legacy equal-centre pair relaxation and return normalized radius-0.45 disks.
/// Preserves the call/default controls and Point2D/Circle2D layouts. Initializes
/// unit working disks on a radius-three ring; each simultaneous sweep attracts
/// edge centre distances toward two and repels nonedges below 2.15, with fixed
/// factor 0.08 and force-distance floor 1e-9. Loops are ignored, duplicates collapse.
/// Then subtract the mean, scale centre extent to one, and RESET radii to 0.45.
/// This is a heuristic drawing, not a construction of a tangency packing: overlaps,
/// missed edge contacts, extra contacts and nondecreasing stress are possible.
/// No planarity test, adaptive step or convergence certificate is supplied. Empty
/// input returns empty; a singleton returns centre zero/radius 0.45 without sweeps.
/// Zero iterations returns postprocessed initial positions. Working arithmetic is
/// checked/widened, so last bits can differ from the old unchecked double path.
/// O(E+V^2+T*V^2) work and O(V^2) auxiliary storage, with actual T=0 for V<2.
/// No input mutation or shared state; returned disks own their values.
/// @param vertex_count Original vertex count, including zero.
/// @param edges Borrowed undirected endpoint pairs, validated before allocation.
/// @param iterations Requested sweeps, default 1000; no early convergence stop.
/// @return Owned heuristic disks; contact feasibility must be measured separately.
/// @throws std::invalid_argument If an endpoint is outside the vertex range.
/// @throws std::length_error If matrix or logical work dimensions exceed supported size_t/container range.
/// @throws std::overflow_error If numeric state or counters cannot be represented.
/// @throws std::bad_alloc If allocation fails.
/// @par Example
/// @code
/// const auto c=dm::algorithms::circle_packing_layout(3,{{0,1},{1,2},{2,0}});
/// std::print("{} {}\n",c.size(),c[0].radius); // 3 0.45
/// @endcode
/// @par References
/// - K. Stephenson (2005), Introduction to Circle Packing, Cambridge University Press;
///   author's overview https://www.circlepack.com/.
///   The circle-packing theorem is background; this equal-centre heuristic is NOT
///   the theorem's variable-radius construction.
/// @see circle_packing_trace(), circle_packing_metrics(), circle_packing_layout_with_options()
std::vector<Circle2D> circle_packing_layout(std::size_t vertex_count,const std::vector<std::pair<std::size_t,std::size_t>>& edges,std::size_t iterations=1000);
/// @brief Run the same checked heuristic with a configurable fixed step and preflight pair cap.
/// Same geometry/postprocessing as circle_packing_layout(); a zero step still
/// executes every requested sweep for V>=2. Pair/position-count forecasts are
/// checked before adjacency allocation. Snapshot controls are validated but the
/// positive snapshot cap applies only to traced calls. No partial output escapes
/// a cap/range/allocation failure. O(V^2) auxiliary storage, no shared state.
/// @param vertex_count Original vertex count, including zero.
/// @param edges Borrowed undirected endpoint pairs.
/// @param options Fixed sweeps/step and work limits; defaults match the legacy call.
/// @return Owned normalized centres and fixed-radius disks, without contact guarantees.
/// @throws std::invalid_argument If an endpoint, fixed step or snapshot stride is invalid.
/// @throws std::length_error If a positive work cap or supported count/container capacity is exceeded.
/// @throws std::overflow_error If numeric state or counters overflow.
/// @throws std::bad_alloc If allocation fails.
/// @par Example
/// @code
/// dm::algorithms::CirclePackingOptions o;o.iterations=0;
/// const auto c=dm::algorithms::circle_packing_layout_with_options(2,{{0,1}},o);
/// std::print("{}\n",c.size()); // 2
/// @endcode
/// @see CirclePackingOptions, circle_packing_layout()
std::vector<Circle2D> circle_packing_layout_with_options(std::size_t vertex_count,const std::vector<std::pair<std::size_t,std::size_t>>& edges,CirclePackingOptions options={});
/// @brief Capture actual unnormalised relaxation states and measure the returned disk geometry.
/// Uses the same kernel without replay. Includes initial, every positive stride
/// and final raw frames; the final frame is not duplicated on a stride boundary.
/// Snapshot energy scans and final disk diagnostics are counted separately from
/// relaxation visits. Complete frame/visit caps preflight after validation, before
/// graph/state allocation. For F frames, adds O(F*V) owned storage and O(F*V^2)
/// measurement work to the ordinary dense relaxation. No partial trace escapes.
/// @param vertex_count Original vertex count, including zero.
/// @param edges Borrowed undirected endpoint pairs, with duplicate/loop semantics as above.
/// @param options Fixed work/record controls; default stride 100 plus initial/final frames.
/// @return Owned raw states, actual normalization map, final disks, metrics and counts.
/// @throws std::invalid_argument If endpoints, fixed step or positive stride are invalid.
/// @throws std::length_error If a work/frame cap or supported count/storage capacity is exceeded.
/// @throws std::overflow_error If numeric state, diagnostics or actual counts overflow.
/// @throws std::bad_alloc If allocation fails.
/// @par Example
/// @code
/// dm::algorithms::CirclePackingOptions o;o.iterations=2;o.snapshot_stride=1;
/// const auto t=dm::algorithms::circle_packing_trace(2,{{0,1}},o);
/// std::print("{} {}\n",t.frames.size(),t.frames.back().iteration); // 3 2
/// @endcode
/// @see CirclePackingTrace, circle_packing_metrics()
CirclePackingTrace circle_packing_trace(std::size_t vertex_count,const std::vector<std::pair<std::size_t,std::size_t>>& edges,CirclePackingOptions options={});
}
