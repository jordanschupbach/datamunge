#pragma once
/// @file gjk.hpp
/// @brief Checked planar GJK distance, actual execution records and regular polygons.
#include <datamunge/geometry/point2d.hpp>
#include <array>
#include <cstddef>
#include <vector>

namespace datamunge::geometry {
/// @brief The actual stopping branch of the floating point planar GJK kernel.
/// A branch name is not an exact geometric certificate. In particular near_origin
/// uses an absolute squared cutoff, and progress_tolerance can stop too early for
/// small shapes. iteration_limit returns the last approximation. empty_input is
/// the historical zero convention, not distance to a nonempty geometric set.
/// @par Example
/// @code
/// const auto s=dm::geometry::GjkStopReason::iteration_limit;
/// std::cout << (s==dm::geometry::GjkStopReason::iteration_limit); // 1
/// @endcode
/// @see gjk_distance_run(), gjk_stop_name()
enum class GjkStopReason {
    empty_input, ///< Either validated input cloud is empty; return zero by convention.
    near_origin, ///< Actual squared closest norm is strictly below 1e-20; return zero.
    progress_tolerance, ///< Actual support improvement is strictly below 1e-10; return current norm.
    triangle_contains_origin, ///< The floating triangle routine takes its final inside branch; return zero.
    iteration_limit ///< Requested loop visits exhausted; return the current norm.
};
/// @brief Selected actual operations of one GJK execution, initially all zero.
/// Counts exclude input validation, arithmetic FLOPs, allocation and wall time.
/// Each Minkowski query scans both complete clouds, including duplicates, with
/// first-index ties: support_dots equals the scanned vertex count and
/// support_comparisons equals support_dots-support_calls. All counters are owned;
/// overflow throws length_error. Concurrent independent calls share no state.
/// @par Example
/// @code
/// const dm::geometry::GjkWork w{};
/// std::cout << w.support_calls << " " << w.iterations; // 0 0
/// @endcode
/// @see GjkRun
struct GjkWork {
    std::size_t support_calls{0}; ///< Individual cloud scans, two per Minkowski query.
    std::size_t support_dots{0}; ///< Vertex-direction dot products used in support scans.
    std::size_t support_comparisons{0}; ///< Strict greater-than support comparisons after each first vertex.
    std::size_t iterations{0}; ///< Loop visits, including a visit that stops before querying support.
    std::size_t progress_checks{0}; ///< Support improvement comparisons, excluding near-origin stops.
    std::size_t segment_reductions{0}; ///< Actual two-point closest-feature calls.
    std::size_t triangle_reductions{0}; ///< Actual three-point closest-feature calls, including inside stops.
};
/// @brief An owned GJK approximation, final working feature and actual stop reason.
/// distance is in input coordinate units. closest/simplex describe the final
/// working Minkowski feature, with at most three points; they are zero/empty for
/// empty input. A near-origin return reports distance zero even if closest is
/// nonzero. A triangle-inside return stores the routine's zero and three points.
/// No closest pair on the original shapes, exact containment, penetration depth
/// or rigorous floating error bound is supplied. Copies own every value.
/// @par Example
/// @code
/// const auto r=dm::geometry::gjk_distance_run({{0,0}},{{3,4}});
/// std::cout << r.distance << " " << r.work.support_calls; // 5 4
/// @endcode
/// @see gjk_distance_run(), GjkStopReason
struct GjkRun {
    double distance{0}; ///< Returned nonnegative approximation, or legacy empty-input zero.
    Point2D closest{}; ///< Last working closest point, in Minkowski difference coordinates.
    std::array<Point2D,3> simplex{}; ///< Copied final feature; only simplex_size elements are active.
    std::size_t simplex_size{0}; ///< Number of active final feature entries, zero through three.
    GjkStopReason stop{GjkStopReason::empty_input}; ///< Actual return branch, not a collision certificate.
    GjkWork work{}; ///< Selected operations accumulated during this execution.
};
/// @brief A point in the actual GJK control flow captured without replay.
/// Initial and empty records precede loop iteration zero; all other iterations
/// are zero based, except limit which uses max_iterations. Records count as
/// complete fixed-size objects even when their simplex has fewer than three points.
/// @par Example
/// @code
/// const auto k=dm::geometry::GjkEventKind::support;
/// std::cout << (k==dm::geometry::GjkEventKind::support); // 1
/// @endcode
/// @see GjkEvent, gjk_distance_trace()
enum class GjkEventKind {
    initial, ///< Initial support point/feature, initial direction, value zero.
    direction, ///< Before support: current closest/feature, negative closest direction, squared norm.
    support, ///< Newly selected support point and improvement; simplex is still the old feature.
    segment, ///< Actual returned two-point reduction, new closest and feature, value zero.
    triangle, ///< Actual returned three-point outside reduction, new closest/feature, value zero.
    triangle_inside, ///< Actual inside branch, zero closest, three-point feature and zero value.
    near, ///< Near-origin exit, current closest/feature, value zero.
    progress, ///< Progress-tolerance exit, current closest/feature, value returned norm.
    limit, ///< Iteration-budget exit, current closest/feature, value returned norm.
    empty ///< Validated empty-input exit; zero points/direction/value and no feature.
};
/// @brief One owned fixed-size GJK state record emitted during the actual kernel.
/// point is the new support point only for support records; otherwise it is the
/// current/returned closest point. direction is the search direction (initial
/// direction for initial, zero for empty). value is squared norm for direction,
/// actual support improvement for support, returned norm for progress/limit and
/// zero otherwise. support_a/b are the most recent selected input indices, not
/// simplex provenance for all retained points; they are zero placeholders for
/// empty input. At each support query strict greater-than retains the first tie.
/// Unused simplex array slots are zero. Copies own data and retain no input aliases.
/// @par Example
/// @code
/// const dm::geometry::GjkEvent e{};
/// std::cout << e.iteration << " " << e.simplex_size; // 0 0
/// @endcode
/// @see GjkEventKind, GjkTrace
struct GjkEvent {
    GjkEventKind kind{GjkEventKind::initial}; ///< Emission point defining all field meanings.
    std::size_t iteration{0}; ///< Zero-based loop index; zero for initial/empty, budget for limit.
    std::size_t support_a{0}; ///< Most recent winning index in the first cloud; empty placeholder zero.
    std::size_t support_b{0}; ///< Most recent winning index in the second cloud; empty placeholder zero.
    Point2D point{}; ///< Actual closest or support point as specified by kind.
    Point2D direction{}; ///< Actual search direction; zero for empty input.
    double value{0}; ///< Actual squared norm, improvement, returned norm or zero as specified by kind.
    std::array<Point2D,3> simplex{}; ///< Copied active feature, with unused slots zero.
    std::size_t simplex_size{0}; ///< Active feature size, zero through three.
};
/// @brief An owned ordinary GJK result and its complete online state records.
/// Input vectors, root object, ordinary result/final feature, constant working
/// simplex/feature storage and allocator overhead are outside the record quota.
/// A quota counts complete GjkEvent objects, not scalar entries, bytes or peak RAM.
/// All metadata and stopping records count. No replay, partial return or truncation
/// occurs. Copies own event storage; independent calls share no mutable state.
/// @par Example
/// @code
/// const auto t=dm::geometry::gjk_distance_trace({{0,0}},{{3,4}});
/// std::cout << t.run.distance << " " << t.events.size(); // 5 4
/// @endcode
/// @see gjk_distance_trace()
struct GjkTrace {
    GjkRun run{}; ///< Ordinary result from this same execution.
    std::vector<GjkEvent> events; ///< Complete actual state records in execution order.
};
/// @brief Return a stable spelling for a GJK stopping branch.
/// @param reason Stop reason; an unrecognized enum value maps to "unknown".
/// @return Static string literal, with process lifetime and no owned allocation.
/// @par Example
/// @code
/// std::cout << dm::geometry::gjk_stop_name(dm::geometry::GjkStopReason::near_origin); // near_origin
/// @endcode
/// @see GjkStopReason
const char* gjk_stop_name(GjkStopReason reason) noexcept;
/// @brief Approximate distance between the convex hulls of two finite planar point clouds.
/// Input order may be arbitrary; duplicate points, singletons and collinear hulls
/// are supported. Concave clouds represent their convex hulls. All coordinates
/// are checked before the legacy empty-input zero convention. Nonempty execution
/// preserves the original ordered double support/simplex arithmetic, first support
/// ties, absolute squared-norm cutoff 1e-20, absolute improvement cutoff 1e-10 and
/// maximum 100 loop visits. Zero is not an exact contact certificate; small shapes
/// can stop prematurely, and rounding/underflow/cancellation remain. No signed
/// penetration depth or callable support-function/3D interface is provided.
/// Inputs are not modified. Validation is O(n+m); with K loop visits the bound is
/// O((K+1)(n+m)) and ordinary auxiliary space O(1), excluding inputs/results.
/// Thread-safe for independent calls and const inputs without caller mutation.
/// @param A First finite stored point cloud; empty is accepted by legacy convention.
/// @param B Second finite stored point cloud; empty is accepted by legacy convention.
/// @return Nonnegative approximation in coordinate units; zero for either empty cloud.
/// @throws std::invalid_argument Any coordinate is nonfinite, including in an otherwise unused vertex.
/// @throws std::overflow_error An evaluated arithmetic intermediate is nonfinite.
/// @throws std::length_error A selected operation counter cannot be represented.
/// @throws std::bad_alloc Constant-sized working storage allocation fails.
/// @par Example
/// @code
/// std::cout << dm::geometry::gjk_distance({{0,0}},{{3,4}}); // 5
/// @endcode
/// @par References
/// - E. G. Gilbert, D. W. Johnson and S. S. Keerthi (1988), "A Fast Procedure for
///   Computing the Distance Between Complex Objects in Three-Dimensional Space",
///   IEEE J. Robotics and Automation 4(2),193–203. doi:10.1109/56.2083, SectionsIII–VI.
///   This planar variant uses fixed absolute cutoffs and a fixed loop cap, without
///   the paper's scaled criterion or backup procedure.
/// @see gjk_distance_run(), gjk_distance_trace()
double gjk_distance(const std::vector<Point2D>& A,const std::vector<Point2D>& B);
/// @brief Execute the same planar GJK kernel with selected work and a stopping reason.
/// Geometry, floating limitations, complete validation and ownership are those of
/// gjk_distance(). max_iterations changes only the loop budget; zero skips every
/// loop visit but still obtains the initial support point and returns its norm
/// with iteration_limit. This budget is not an accuracy or convergence guarantee.
/// Validation O(n+m), kernel O((K+1)(n+m)), ordinary auxiliary storage O(1).
/// @param A First finite point cloud, without mutation.
/// @param B Second finite point cloud, without mutation.
/// @param max_iterations Maximum loop visits, default100; zero performs only initialization.
/// @return Owned approximation/final working feature, reason and selected counters.
/// @throws std::invalid_argument Any input coordinate is nonfinite, before budget/empty handling.
/// @throws std::overflow_error Any evaluated arithmetic intermediate is nonfinite.
/// @throws std::length_error A selected counter cannot fit size_t.
/// @throws std::bad_alloc Constant-sized working allocation fails.
/// @par Example
/// @code
/// const auto r=dm::geometry::gjk_distance_run({{0,0}},{{3,4}},0);
/// std::cout << r.distance << " " << dm::geometry::gjk_stop_name(r.stop); // 5 iteration_limit
/// @endcode
/// @par References
/// - Gilbert, Johnson and Keerthi (1988), doi:10.1109/56.2083, SectionsIII–VI;
///   legacy planar absolute-cutoff variant described in gjk_distance().
/// @see gjk_distance(), GjkRun
GjkRun gjk_distance_run(const std::vector<Point2D>& A,const std::vector<Point2D>& B,std::size_t max_iterations=100);
/// @brief Capture complete owned state records online while executing the shared GJK kernel.
/// Ordinary geometry/budget semantics are those of gjk_distance_run(). Before
/// each whole append, max_records and vector capacity are checked. The default
/// limit is one million complete records; zero disables only that optional quota.
/// All initial/empty/stop records count. A separated singleton pair takes four
/// records (reject3/admit4); a near-origin pair takes three (reject2/admit3).
/// Empty input takes one record; quota zero is off and therefore admits it too.
/// No partial history is returned and input is unchanged on failure. Storage is
/// O(H) additional complete records; this is not a numeric-entry, byte or peak RAM
/// limit. Input/root/ordinary result/final feature/working storage/allocator are
/// excluded. Records are actual kernel states, never numerical replay.
/// @param A First finite input cloud, without mutation or retained aliases.
/// @param B Second finite input cloud, without mutation or retained aliases.
/// @param max_iterations Loop visits, default100; zero still records initialization and limit.
/// @param max_records Optional complete-record quota, default1000000; zero turns it off.
/// @return Owned run plus every emitted state record from the same execution.
/// @throws std::invalid_argument Any coordinate is nonfinite, before quota/empty handling.
/// @throws std::overflow_error Any evaluated arithmetic intermediate is nonfinite.
/// @throws std::length_error The next record exceeds quota/container capacity or a counter overflows.
/// @throws std::bad_alloc Working or retained history allocation fails.
/// @par Example
/// @code
/// const auto t=dm::geometry::gjk_distance_trace({{0,0}},{{3,4}},100,4);
/// std::cout << t.events.size() << " " << t.run.distance; // 4 5
/// @endcode
/// @par References
/// - Gilbert, Johnson and Keerthi (1988), doi:10.1109/56.2083; actual states of
///   the planar variant, with no replay or exact collision certificate.
/// @see GjkTrace, gjk_distance_run()
GjkTrace gjk_distance_trace(const std::vector<Point2D>& A,const std::vector<Point2D>& B,std::size_t max_iterations=100,std::size_t max_records=1000000);
/// @brief Generate owned regular polygon vertices using the original report's ordered formula.
/// Vertex i uses angle rotation+2*pi*i/sides, then center+radius*(cos,sin).
/// Mathematically the order is counterclockwise; finite rounding may coincide
/// vertices. Radius zero gives repeated center points. This is a point generator,
/// not an exact convexity/orientation certificate. All scalar arguments and the
/// whole vertex count/quota/capacity are checked before allocation. Computed
/// angles/coordinates are checked before append. O(sides) time/output storage;
/// no caller mutation, retained aliases or shared mutable state.
/// @param center Finite center in coordinate units.
/// @param radius Finite nonnegative circumradius, in coordinate units.
/// @param sides Number of generated vertices, at least3.
/// @param rotation Finite angle in radians, default0; not normalized.
/// @param max_vertices Optional whole output-vertex quota, default1000000; zero disables it.
/// @return Owned vector of exactly sides vertices, beginning at rotation.
/// @throws std::invalid_argument Nonfinite scalar, negative radius or fewer than three sides.
/// @throws std::overflow_error Computed angle or coordinate is nonfinite.
/// @throws std::length_error Whole count exceeds the optional quota or vector capacity.
/// @throws std::bad_alloc Output allocation fails.
/// @par Example
/// @code
/// const auto p=dm::geometry::regular_polygon({2,3},1,4);
/// std::cout << p.size() << " " << p[0].x << " " << p[0].y; // 4 3 3
/// @endcode
/// @see gjk_distance(), Point2D
std::vector<Point2D> regular_polygon(Point2D center,double radius,std::size_t sides,double rotation=0,std::size_t max_vertices=1000000);
} // namespace datamunge::geometry
