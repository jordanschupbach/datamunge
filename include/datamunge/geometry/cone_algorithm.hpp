#pragma once
/// @file cone_algorithm.hpp
/// @brief Checked planar empty-sector classification, actual angular witnesses and bounded disk sampling.
#include <datamunge/geometry/point2d.hpp>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace datamunge::geometry {

/// @brief Complete pair-work and optional actual-direction storage limits.
/// All finite coordinates and controls validate before the n(n-1) forecast.
/// Zero means unlimited. The direction cap is online and applies only to traces;
/// analysis/output-only calls do not retain all directions. Failures expose no
/// partial return. No spatial index is implied by a small geometric radius.
/// @par Example
/// @code
/// dm::geometry::ConeSurfaceLimits limits;limits.max_pair_visits=6;
/// const auto a=dm::geometry::cone_surface_analyze({{0,0},{1,0},{0,1}},2,3,limits);
/// std::print("{}\n",a.work.pair_visits); // 6
/// @endcode
/// @see cone_surface_analyze(), cone_surface_trace()
struct ConeSurfaceLimits {
  std::size_t max_pair_visits=10000000; ///< Positive complete ordered-pair cap, default ten million; zero unlimited.
  std::size_t max_directions=1000000; ///< Positive actual retained-direction cap, tracing only; zero unlimited.
};

/// @brief Owned original neighbour identity, location, distance and computed direction.
/// Angles use atan2 in [-pi,pi]; equal computed angles sort by original index.
/// Repeated nonzero directions remain separate records. Coincident query/other
/// locations are excluded, since their direction is undefined. No input reference.
/// @par Example
/// @code
/// const auto t=dm::geometry::cone_surface_trace({{0,0},{1,0}},1,3);
/// std::print("{} {}\n",t.directions[0][0].index,t.directions[0][0].distance); // 1 1
/// @endcode
/// @see ConeSurfaceTrace
struct ConeDirection {
  std::size_t index=0; ///< Original neighbour index, different from the query index.
  Point2D point; ///< Owned finite neighbour-coordinate copy.
  long double distance=0; ///< Actual widened hypot distance, positive and at most radius.
  long double angle=0; ///< Actual widened atan2 direction in radians, [-pi,pi].
};

/// @brief Owned actual largest-gap witness and classification for one original point.
/// At least two directions select the first strict largest gap in sorted adjacent
/// order, then the wrap gap last. Empty/single-direction queries use a full-turn
/// score by convention and are flagged for every valid threshold. For no
/// directions, gap endpoint ids are n (the input size), not valid point ids.
/// gap_start and largest_gap describe an open angular sector; its bounding
/// neighbour rays are allowed. Wrap arcs can extend beyond pi. Rounded gaps are
/// confined to [0,2pi]; this is floating geometry, not an exact predicate.
/// @par Example
/// @code
/// const auto a=dm::geometry::cone_surface_analyze({{0,0},{1,0},{0,1}},2,3);
/// std::print("{} {}\n",a.points[0].neighbors,a.points[0].surface); // 2 true
/// @endcode
/// @see ConeSurfaceAnalysis, cone_surface_analyze()
struct ConePointSummary {
  std::size_t index=0; ///< Original query index.
  Point2D point; ///< Owned query-coordinate copy.
  std::size_t neighbors=0; ///< Noncoincident neighbour records within the inclusive radius.
  std::size_t coincident=0; ///< Other original records exactly at the query's location, excluded from directions.
  long double largest_gap=0; ///< Actual maximum circular gap; full turn for fewer than two directions.
  long double gap_start=0; ///< First boundary direction of the selected counterclockwise open sector.
  std::size_t gap_from=0; ///< Original neighbour at the starting ray, or n when none.
  std::size_t gap_to=0; ///< Original neighbour at the ending ray, or n when none.
  bool insufficient=false; ///< True when fewer than two noncoincident direction records exist.
  bool surface=false; ///< largest_gap >= validated threshold, with no hidden tolerance.
};

/// @brief Actual all-pairs and angular work, excluding allocator, sort-comparator and rendering costs.
/// Pair visits include coincident records; distance evaluations exclude them.
/// Angle evaluations count admitted noncoincident directions. sort_calls counts
/// std::sort invocations with at least two records, not sort comparisons.
/// gap_tests counts k circular gaps for each query with k>=2; sparse conventions
/// add no gap tests. Complete pair preflight and checked increments prevent wrap.
/// @par Example
/// @code
/// const auto w=dm::geometry::cone_surface_analyze({{0,0},{1,0},{0,1}},2,3).work;
/// std::print("{} {}\n",w.pair_visits,w.gap_tests); // 6 6
/// @endcode
/// @see ConeSurfaceAnalysis
struct ConeSurfaceWork {
  std::size_t validated_points=0; ///< Complete finite-coordinate validation visits.
  std::size_t pair_visits=0; ///< Actual ordered i!=j visits; always n(n-1).
  std::size_t coincident_visits=0; ///< Exact-location duplicates skipped before distance/direction evaluation.
  std::size_t distance_evaluations=0; ///< Actual widened nonzero-displacement hypot calls.
  std::size_t angle_evaluations=0; ///< Actual atan2 calls for admitted nonzero displacements.
  std::size_t sort_calls=0; ///< Actual angle/id sorts, only for queries with at least two directions.
  std::size_t gap_tests=0; ///< Actual adjacent/wrap gap candidates tested.
};

/// @brief Owned binary mask, per-point witnesses and actual work from one executing scan.
/// Mask entries align with original indices, including coincident queries.
/// No references or shared state survive. The radius criterion is local, but this
/// implementation visits every ordered pair: O(n^2+sum(k log k)) time, bounded
/// by O(n^2 log n), and O(n) scratch/output storage without retained directions.
/// @par Example
/// @code
/// const auto a=dm::geometry::cone_surface_analyze({{0,0},{1,0},{0,1}},2,3);
/// std::print("{} {}\n",a.surface.size(),a.points[0].surface); // 3 true
/// @endcode
/// @see cone_surface_analyze()
struct ConeSurfaceAnalysis {
  std::vector<char> surface; ///< Owned zero/one mask in original point order.
  std::vector<ConePointSummary> points; ///< Owned actual query summaries in original order.
  ConeSurfaceWork work; ///< Actual complete scan and angular work.
};

/// @brief Owned per-query sorted direction records from the same scan as the analysis.
/// Total direction storage is O(sum k), at most n(n-1), in addition to O(n)
/// analysis/scratch storage. No candidate distance/angle is replayed. Sorted
/// records retain original identities and copied coordinates.
/// @par Example
/// @code
/// const auto t=dm::geometry::cone_surface_trace({{0,0},{1,0},{0,1}},2,3);
/// std::print("{} {}\n",t.analysis.points[0].gap_from,t.analysis.points[0].gap_to); // 2 1
/// @endcode
/// @see cone_surface_trace()
struct ConeSurfaceTrace {
  ConeSurfaceAnalysis analysis; ///< Same output/work/witnesses as the untraced analysis.
  std::vector<std::vector<ConeDirection>> directions; ///< Owned sorted admitted directions for each original query.
};

/// @brief Apply the checked planar largest-angular-gap rule and return the legacy zero/one mask.
/// Radius is inclusive, self indices and coincident locations are excluded, and
/// at least two admitted directions are sorted to measure all circular gaps.
/// Fewer than two directions receive a full-turn score. A point is flagged when
/// its computed largest gap is at least the threshold. This is a local sampling
/// diagnostic, not a guarantee of an underlying physical boundary. It is a 2D
/// angular-gap analogue of the published 3D empty-cone idea, not that 3D algorithm.
/// Widened differences/hypot avoid squared-distance underflow/overflow. Arithmetic
/// and angular comparisons remain rounded; no exact-predicate claim or tolerance.
/// No work cap is imposed by this legacy call. O(n^2+sum(k log k)) time, O(n)
/// scratch/output storage, no input mutation or shared state.
/// @param pts Borrowed finite planar records, possibly empty or coincident.
/// @param radius Finite strictly positive inclusive distance cutoff.
/// @param gap_threshold Finite angle in radians, greater than zero and no greater than 2pi.
/// @return Owned char mask with entries zero or one in original index order.
/// @throws std::invalid_argument If any coordinate or control is invalid.
/// @throws std::length_error If complete pair count or container capacity cannot fit.
/// @throws std::overflow_error If widened arithmetic or actual counters cannot fit.
/// @throws std::bad_alloc If storage cannot be allocated.
/// @par Example
/// @code
/// const auto f=dm::geometry::cone_surface_points({{0,0},{1,0}},1,3);
/// std::print("{} {}\n",int(f[0]),int(f[1])); // 1 1
/// @endcode
/// @par References
/// - Y. Wang, S. Teitel and C. Dellago (2005), Melting of icosahedral gold nanoclusters
///   from molecular dynamics simulations, J. Chem. Phys. 122, 214722,
///   doi:10.1063/1.1917756 (3D empty-cone idea; this API is a planar angular-gap variant).
/// @see cone_surface_analyze(), cone_surface_trace()
std::vector<char> cone_surface_points(const std::vector<Point2D>& pts,double radius,double gap_threshold);

/// @brief Return actual planar gap witnesses and work without retaining all neighbour directions.
/// Uses the same validated kernel and rounding semantics as cone_surface_points().
/// Full point/control validation precedes complete pair-work preflight/allocation.
/// Threshold is long double so an actual returned gap can be reused exactly as
/// an inclusive threshold. max_directions is ignored; no directions are retained.
/// No partial result escapes a failed cap/numeric/allocation check.
/// @param pts Borrowed finite original planar records.
/// @param radius Finite strictly positive inclusive cutoff.
/// @param gap_threshold Finite angle in (0,2pi], in radians.
/// @param limits Complete ordered-pair cap and ignored trace-only direction cap.
/// @return Owned actual summaries/mask/work; no input references are retained.
/// @throws std::invalid_argument If any point, radius or threshold is invalid.
/// @throws std::length_error If complete work cap/count or storage capacity is exceeded.
/// @throws std::overflow_error If widened arithmetic/counters cannot fit.
/// @throws std::bad_alloc If allocation fails.
/// @par Example
/// @code
/// const auto a=dm::geometry::cone_surface_analyze({{0,0},{1,0},{-1,0},{0,1},{0,-1}},1,3);
/// std::print("{} {}\n",a.points[0].neighbors,a.points[0].surface); // 4 false
/// @endcode
/// @par References
/// - Y. Wang, S. Teitel and C. Dellago (2005), doi:10.1063/1.1917756
///   (3D empty-cone idea; see cone_surface_points() for the planar distinction).
/// @see ConeSurfaceAnalysis, cone_surface_points()
ConeSurfaceAnalysis cone_surface_analyze(const std::vector<Point2D>& pts,double radius,long double gap_threshold,ConeSurfaceLimits limits={});

/// @brief Retain actual sorted admitted directions without replaying the planar classifier.
/// Same result, witnesses and work as cone_surface_analyze(). The trace-only cap
/// applies online before each admitted direction is stored; coincident/far records
/// do not consume it. Zero means unlimited. Ties use computed angle then original
/// id, and first strict largest adjacent gap wins before the wrap gap is tested.
/// @param pts Borrowed finite original planar records.
/// @param radius Finite strictly positive inclusive cutoff.
/// @param gap_threshold Finite radians in (0,2pi].
/// @param limits Complete pair cap and actual direction storage cap; default ten million/one million.
/// @return Owned analysis and per-query actual direction records.
/// @throws std::invalid_argument If any point/control is invalid.
/// @throws std::length_error If complete pair or online direction cap/count/capacity is exceeded.
/// @throws std::overflow_error If widened arithmetic/counters cannot fit.
/// @throws std::bad_alloc If allocation fails.
/// @par Example
/// @code
/// const auto t=dm::geometry::cone_surface_trace({{0,0},{1,0},{0,1}},2,3);
/// std::print("{}\n",t.directions[0].size()); // 2
/// @endcode
/// @par References
/// - Y. Wang, S. Teitel and C. Dellago (2005), doi:10.1063/1.1917756
///   (3D empty-cone idea; this trace records the planar angular-gap variant).
/// @see ConeSurfaceTrace, cone_surface_analyze()
ConeSurfaceTrace cone_surface_trace(const std::vector<Point2D>& pts,double radius,long double gap_threshold,ConeSurfaceLimits limits={});

/// @brief Generate the original square-rejection disk workload with a local mt19937 engine.
/// Each trial consumes an x draw then a y draw from uniform_real(-R,R); accept
/// when widened hypot(x,y)<=R. Trial count, not accepted count, is fixed.
/// Preserves the original seed-one/2500-trial draw stream on a given standard
/// library. Distribution mappings are not promised bitwise portable. No global
/// random state. O(trials) work/storage, with complete trial preflight.
/// @param trials Number of square candidates, possibly zero.
/// @param radius Finite positive disk radius; double interval width 2R must be finite.
/// @param seed Local mt19937 seed, default one.
/// @param max_trials Positive complete-work cap, default one million.
/// @return Owned accepted finite points in candidate encounter order.
/// @throws std::invalid_argument If radius/interval width or cap is invalid.
/// @throws std::length_error If the trial cap or vector capacity is exceeded.
/// @throws std::overflow_error If generated coordinates/arithmetic are nonfinite.
/// @throws std::bad_alloc If allocation fails.
/// @par Example
/// @code
/// const auto p=dm::geometry::sample_disk_candidates(100,1);
/// std::print("{}\n",p.size()<=100); // true
/// @endcode
/// @see cone_surface_analyze()
std::vector<Point2D> sample_disk_candidates(std::size_t trials,double radius,std::uint32_t seed=1,std::size_t max_trials=1000000);

/// @brief Owned radial diagnostics for a binary mask, with explicit absent-flag state.
/// Inner points have distance strictly below inner_radius; outer points strictly
/// above outer_radius. Equality belongs to neither named region. These are chosen
/// diagnostic bands, not physical ground-truth surface labels. Min/max flagged
/// radii are zero placeholders when has_flagged=false. No input references.
/// @par Example
/// @code
/// const auto s=dm::geometry::radial_flag_summary({{0,0},{3,4}},{0,1},1,4);
/// std::print("{} {}\n",s.flagged_points,s.min_flagged_radius); // 1 5
/// @endcode
/// @see radial_flag_summary()
struct RadialFlagSummary {
  std::size_t point_count=0; ///< Original point count.
  std::size_t flagged_points=0; ///< Number of mask entries equal to one.
  std::size_t inner_points=0; ///< Points with radius strictly below the inner cutoff.
  std::size_t inner_flagged=0; ///< Flagged points in that strict inner region.
  std::size_t outer_points=0; ///< Points with radius strictly above the outer cutoff.
  std::size_t outer_flagged=0; ///< Flagged points in that strict outer region.
  bool has_flagged=false; ///< Whether flagged radius extrema are meaningful.
  long double min_flagged_radius=0; ///< Minimum flagged radius, or zero if absent.
  long double max_flagged_radius=0; ///< Maximum flagged radius, or zero if absent.
};

/// @brief Measure a mask's chosen inner/outer bands and flagged radii in widened arithmetic.
/// Fully validates all points, centre, mask length/binary values and finite ordered
/// nonnegative cutoffs before measurement. Uses widened differences/hypot, not
/// squared double distances. O(n) work, constant extra storage, no mutation/shared state.
/// @param pts Borrowed finite planar records.
/// @param flags Same-length mask containing only char zero or one.
/// @param inner_radius Finite nonnegative strict inner cutoff.
/// @param outer_radius Finite cutoff at least inner_radius; outer uses strict greater-than.
/// @param center Finite measurement centre, default origin.
/// @return Owned radial counts and explicit optional-extrema state.
/// @throws std::invalid_argument If coordinates, mask or cutoffs are invalid.
/// @throws std::overflow_error If widened radial arithmetic cannot fit.
/// @par Example
/// @code
/// const auto s=dm::geometry::radial_flag_summary({{0,0},{3,4}},{0,1},1,4);
/// std::print("{} {} {}\n",s.inner_flagged,s.outer_flagged,s.has_flagged); // 0 1 true
/// @endcode
/// @see RadialFlagSummary
RadialFlagSummary radial_flag_summary(const std::vector<Point2D>& pts,const std::vector<char>& flags,double inner_radius,double outer_radius,Point2D center={});

} // namespace datamunge::geometry
