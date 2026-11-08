#pragma once
/// @file
/// @brief Exact-sign symbolic-boundary Delaunay triangulation, actual cavities and mesh measurements.
#include <datamunge/geometry/incircle.hpp>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>
namespace datamunge::geometry {
/// @brief Three original point indices in counterclockwise order for finite mesh faces.
/// Original field layout/defaults are retained. A trace ghost uses c=points.size()
/// as a symbolic infinity marker, with a,b the reversed finite hull edge; it is
/// never returned by delaunay_triangulation(). No coordinate is constructed for it.
/// @par Example
/// @code
/// const dm::geometry::Triangle t{0,1,2};std::print("{} {} {}\n",t.a,t.b,t.c); // 0 1 2
/// @endcode
/// @see delaunay_triangulation(), DelaunayInsertion
struct Triangle {
  std::size_t a{0}; ///< First original index, finite also for a trace ghost.
  std::size_t b{0}; ///< Second original index, finite also for a trace ghost.
  std::size_t c{0}; ///< Third original index; points.size() denotes a trace ghost only.
};
/// @brief Actual seed and cavity operations, excluding validation, sorting and copies.
/// A conflict test is either an incircle test on a finite face or an exact
/// half-plane test on a ghost. Counters exclude CCW checks on newly created faces,
/// between-endpoint comparisons, map operations, allocation, trace copies and final
/// sorting. Initial seed plus three ghosts are excluded from creation/removal counts.
/// No counts are estimates of time or fixed-predicate internal arithmetic.
/// @par Example
/// @code
/// const auto t=dm::geometry::delaunay_trace({{0,0},{4,0},{4,4},{0,4},{2,2}});
/// std::print("{} {}\n",t.work.insertions,t.work.incircle_tests+t.work.hull_side_tests); // 2 10
/// @endcode
/// @see DelaunayTrace
struct DelaunayWork {
  std::size_t seed_orientation_tests{0}; ///< Exact tests finding the first noncollinear seed.
  std::size_t insertions{0}; ///< Actual non-seed distinct sites inserted.
  std::size_t incircle_tests{0}; ///< Finite live faces tested against an inserted site.
  std::size_t hull_side_tests{0}; ///< Symbolic boundary faces tested by orientation.
  std::size_t removed_faces{0}; ///< All conflicting real/ghost faces removed.
  std::size_t boundary_edges{0}; ///< Actual directed cavity edges retained once.
  std::size_t created_faces{0}; ///< New real/ghost faces, one per boundary edge.
};
/// @brief Owned actual cavity removal, directed boundary and replacement for one site.
/// removed/added may contain ghosts, identified by c=the original input count.
/// boundary is ordered by canonical undirected endpoint key but retains the first
/// directed occurrence; shared interior edges are cancelled. Actual face scans
/// precede this event. No full before/after mesh snapshot or geometric replay is
/// retained. All vectors own their entries and have no input lifetime dependency.
/// @par Example
/// @code
/// const auto t=dm::geometry::delaunay_trace({{0,0},{4,0},{4,4},{0,4},{2,2}});
/// std::print("{} {} {}\n",t.steps.back().input_index,t.steps.back().removed.size(),t.steps.back().added.size()); // 4 2 4
/// @endcode
/// @see DelaunayTrace
struct DelaunayInsertion {
  std::size_t input_index{0}; ///< Earliest original id of the inserted distinct coordinate.
  std::size_t faces_before{0}; ///< Live real plus ghost faces before this insertion.
  std::size_t faces_after{0}; ///< Live real plus ghost faces after this insertion.
  std::vector<Triangle> removed; ///< Exact actual conflicting faces, in scan order.
  std::vector<std::pair<std::size_t,std::size_t>> boundary; ///< Actual directed nonshared cavity edges.
  std::vector<Triangle> added; ///< Actual replacement real/ghost faces, in boundary-key order.
};
/// @brief Complete owned sites, earliest-identity mapping, finite mesh and actual events.
/// Numeric coordinate equality coalesces duplicates and signed zeros. Representatives
/// retain earliest input indices in input order. Seed is the first two distinct
/// sites and first later noncollinear site; remaining distinct sites are inserted
/// in input order. Strict positive incircle conflicts retain an existing diagonal
/// on an exact cocircular tie; permutations may therefore select another valid mesh.
/// Final finite triangles rotate their smallest id first and are lexicographically
/// sorted, independent of map/hash traversal. All-collinear/fewer than three
/// distinct sites have no seed, faces or insertion events. All fields own data;
/// public edits are unchecked and independent calls share no mutable state.
/// @par Example
/// @code
/// const auto t=dm::geometry::delaunay_trace({{0,0},{1,0},{0,1},{0,0}});
/// std::print("{} {} {}\n",t.representatives.size(),t.representative_for_input.back(),t.triangles.size()); // 3 0 1
/// @endcode
/// @see delaunay_trace(), delaunay_triangulation()
struct DelaunayTrace {
  std::vector<Point2D> points; ///< Complete copied original coordinates, including duplicates.
  std::vector<std::size_t> representatives; ///< Earliest unique ids in original input order.
  std::vector<std::size_t> representative_for_input; ///< Earliest equal-coordinate id per original record.
  bool has_seed{false}; ///< Whether three distinct noncollinear sites exist.
  Triangle seed; ///< Initial CCW finite seed, meaningful only when has_seed.
  std::vector<Triangle> triangles; ///< Complete finite final CCW triangles, using earliest original ids.
  std::vector<DelaunayInsertion> steps; ///< Actual non-seed cavity transitions, without replay.
  DelaunayWork work; ///< Checked actual operations from the executing kernel.
};
/// @brief Coordinate-deduplicated topology counts and exact local empty-circle checks.
/// Counts are independent of triangle order. Duplicate-coordinate ids are mapped
/// to earliest originals. Triangle indices must be valid, distinct in coordinate,
/// exactly CCW, with no duplicate face and at most two incident faces per edge.
/// These checks do not certify global nonoverlap, hull coverage or planar embedding
/// of an arbitrary supplied mesh. minimum_angle_degrees is a scaled floating
/// diagnostic, NaN for no faces and potentially zero after extreme scale loss;
/// it never controls topology. Empty-circle signs are exact for stored binary64.
/// @par Example
/// @code
/// const std::vector<dm::geometry::Point2D> p{{0,0},{4,0},{4,4},{0,4},{2,2}};
/// const auto a=dm::geometry::delaunay_analyze(p,dm::geometry::delaunay_triangulation(p));
/// std::print("{} {} {} {}\n",a.distinct_points,a.triangles,a.boundary_edges,a.interior_violations); // 5 4 4 0
/// @endcode
/// @see delaunay_analyze()
struct DelaunayAnalysis {
  std::size_t distinct_points{0}; ///< Number of unique coordinates, including unused sites.
  std::size_t used_points{0}; ///< Unique sites incident to at least one supplied face.
  std::size_t triangles{0}; ///< Number of supplied distinct CCW faces.
  std::size_t edges{0}; ///< Distinct undirected coordinate edges.
  std::size_t boundary_edges{0}; ///< Edges incident to exactly one face.
  std::size_t circle_tests{0}; ///< Actual nonvertex distinct-site queries for every face.
  std::size_t interior_violations{0}; ///< Strictly positive exact incircle queries.
  double minimum_angle_degrees{0}; ///< Approximate minimum interior angle, NaN for no faces.
};
/// @brief Construct a complete unconstrained Delaunay triangulation with exact signs.
/// Retains the original one-vector signature and Triangle layout. Finite coordinate
/// validation precedes even trivial results. Duplicates coalesce to earliest ids;
/// collinear/fewer-than-three distinct sites return no faces. Uses Bowyer-Watson
/// cavity replacement with symbolic hull ghosts, avoiding any finite super-triangle
/// whose proximity could remove valid narrow faces. All orientation/incircle signs
/// are exact for finite stored binary64, including subnormals/extreme scales.
/// Cocircular ties retain existing faces under the ordering in DelaunayTrace.
/// Faces are strictly CCW, contain no ghosts and reference original ids. Finite
/// output order is canonical as documented on DelaunayTrace. No boundary constraints,
/// holes, Steiner refinement or minimum-angle guarantee is supplied. Input unchanged;
/// O(n log(n+1)+u^2 log(u+1)) worst-case work for n records/u distinct sites with
/// ordered cavity maps and full live scans; O(n+u) working/output storage, no shared
/// state or optional resource caps. Exact-predicate cost is bounded per binary64 query.
/// @param points Borrowed finite planar sites, possibly empty/duplicate/collinear.
/// @return Owned complete finite mesh of the distinct sites' convex hull, or no faces for collinearity.
/// @throws std::invalid_argument Any nonfinite coordinate, even on short input.
/// @throws std::length_error Counter arithmetic or container capacity exceeded.
/// @throws std::logic_error Internal cavity topology invariant fails; no partial mesh returned.
/// @throws std::bad_alloc Allocation fails.
/// @par Example
/// @code
/// std::print("{}\n",dm::geometry::delaunay_triangulation({{0,0},{1,0},{.5,.000001}}).size()); // 1
/// @endcode
/// @par References
/// - A. Bowyer (1981), Computing Dirichlet tessellations, Computer Journal 24(2),162–166, doi:10.1093/comjnl/24.2.162.
/// - D. F. Watson (1981), Computing the n-dimensional Delaunay tessellation with application to Voronoi polytopes, Computer Journal 24(2),167–172, doi:10.1093/comjnl/24.2.167. This variant uses symbolic boundary faces, exact decoded-dyadic signs, earliest duplicate identities and ordered cavity edges.
/// @see delaunay_trace(), incircle_sign()
[[nodiscard]] std::vector<Triangle> delaunay_triangulation(const std::vector<Point2D>& points);
/// @brief Capture actual symbolic-boundary cavity transitions from the same solver.
/// No replay. Full finite validation precedes the original-record cap and all
/// allocation. max_tests bounds cumulative incircle plus hull-side calls, checked
/// before each complete live-face scan. Trace entries charge one per insertion
/// plus removed faces, boundary edges and added faces, including ghosts, checked
/// before retaining the event. Points/maps/final output and temporary cavities are
/// excluded from that entry count; caps do not bound bytes or wall time. Zero
/// disables only an optional cap. O(sum cavity/event sizes) additional history,
/// at worst O(u^2), beyond the ordinary solver. Failure returns no partial result.
/// @param points Finite original coordinates, unchanged.
/// @param max_points Complete original-record cap, default 1000000; zero disables.
/// @param max_tests Cumulative complete-scan conflict cap, default 10000000; zero disables.
/// @param max_entries Cumulative event-entry cap, default 1000000; zero disables.
/// @return Owned points, original-id maps, finite mesh, actual events and work.
/// @throws std::invalid_argument Any nonfinite coordinate.
/// @throws std::length_error Optional cap, count arithmetic or container capacity exceeded.
/// @throws std::logic_error Internal cavity invariant fails.
/// @throws std::bad_alloc Allocation fails.
/// @par Example
/// @code
/// const auto t=dm::geometry::delaunay_trace({{0,0},{4,0},{4,4},{0,4},{2,2}},5,10);
/// std::print("{} {}\n",t.triangles.size(),t.work.incircle_tests+t.work.hull_side_tests); // 4 10
/// @endcode
/// @par References
/// - Bowyer (1981), doi:10.1093/comjnl/24.2.162; Watson (1981), doi:10.1093/comjnl/24.2.167; records the symbolic-boundary variant rather than a geometric replay.
/// @see DelaunayTrace, delaunay_triangulation()
DelaunayTrace delaunay_trace(const std::vector<Point2D>& points,std::size_t max_points=1000000,
                             std::size_t max_tests=10000000,std::size_t max_entries=1000000);
/// @brief Measure a supplied mesh and test every nonvertex site against every circle.
/// Performs the structural checks on DelaunayAnalysis first. The complete prospective
/// circle count faces*(unique_sites-3) is checked before circle queries; zero cap
/// disables only that limit. O(n log(n+1)+f log(f+1)+fu) work and O(n+f) storage;
/// minimum angles are floating diagnostics and are not certified bounds. This is a
/// local property analysis, not a global embedding/coverage certificate.
/// @param points Finite original sites, unchanged.
/// @param triangles Finite CCW faces with valid original indices, unchanged.
/// @param max_tests Complete exact-circle query cap, default 1000000; zero disables.
/// @return Owned scalar topology, actual query/violation counts and approximate angle.
/// @throws std::invalid_argument Nonfinite coordinates, bad/degenerate/non-CCW/duplicate faces or nonmanifold edges.
/// @throws std::length_error Circle-count arithmetic or optional cap exceeded.
/// @throws std::bad_alloc Allocation fails.
/// @par Example
/// @code
/// const std::vector<dm::geometry::Point2D> p{{0,0},{1,0},{1,1},{0,1}};
/// const auto a=dm::geometry::delaunay_analyze(p,dm::geometry::delaunay_triangulation(p));
/// std::print("{} {} {}\n",a.edges,a.boundary_edges,a.circle_tests); // 5 4 2
/// @endcode
/// @see DelaunayAnalysis, delaunay_triangulation()
DelaunayAnalysis delaunay_analyze(const std::vector<Point2D>& points,const std::vector<Triangle>& triangles,
                                 std::size_t max_tests=1000000);
/// @brief Generate the report's seeded uniform square-cloud control in the core.
/// Uses mt19937(seed) and uniform_real_distribution<double>(0,extent), x then y
/// for each point. Same-library runs reproduce the original report's seed20250728
/// and 30-site square when defaults are used. Standard distribution algorithms
/// need not match across C++ libraries, so the seed is not a universal byte fixture.
/// Coincident draws are allowed and later triangulation deduplicates them.
/// O(count) work/storage, owned output, no global RNG or shared mutation.
/// @param count Number of original generated records, zero allowed.
/// @param seed mt19937 seed, default 20250728.
/// @param extent Positive finite square extent, default 10; checked even for count zero.
/// @param max_points Optional complete output-record cap, default 1000000; zero disables.
/// @return Owned generated points in [0,extent) on each axis.
/// @throws std::invalid_argument Nonpositive or nonfinite extent.
/// @throws std::length_error Optional count cap or container capacity exceeded before allocation.
/// @throws std::bad_alloc Allocation fails.
/// @par Example
/// @code
/// std::print("{}\n",dm::geometry::delaunay_uniform_cloud(30).size()); // 30
/// @endcode
/// @see delaunay_trace()
std::vector<Point2D> delaunay_uniform_cloud(std::size_t count,std::uint32_t seed=20250728,double extent=10,
                                           std::size_t max_points=1000000);
} // namespace datamunge::geometry
