#pragma once
/// @file segment_intersection.hpp
/// @brief Exact binary64 contact classification and approximate unique-point construction.
#include <array>
#include <cstddef>
#include <datamunge/geometry/orientation.hpp>
/// @brief Canonical checked native producers and owned algorithm values.
namespace datamunge::geometry {
/// @brief Exact orientation of three stored finite binary64 points.
/// @par Example
/// @code
/// const auto value=dm::geometry::Orientation::Collinear;
/// std::print("{}\n",static_cast<int>(value)); // 0
/// @endcode
enum class Orientation {
  Collinear,       ///< Exactly zero determinant.
  Clockwise,       ///< Exactly negative determinant.
  CounterClockwise ///< Exactly positive determinant.
};
/// @brief Exact intersection-set type for two closed finite segments.
/// @par Example
/// @code
/// const auto value=dm::geometry::SegmentContact::None;
/// std::print("{}\n",static_cast<int>(value)); // 0
/// @endcode
enum class SegmentContact {
  None,   ///< Empty intersection.
  Point,  ///< Exactly one point, including endpoint and zero-length-segment contact.
  Overlap ///< Nonzero collinear overlap, with two distinct represented endpoint bounds.
};
/// @brief Owned actual exact predicates and optional approximate unique point.
/// @par Example
/// @code
/// const dm::geometry::SegmentIntersectionRun value{};
/// std::print("{}\n",value.orientation_tests); // 0
/// @endcode
struct SegmentIntersectionRun {
  SegmentContact contact{
      SegmentContact::None};              ///< Exact represented-coordinate contact classification.
  std::array<int, 4> orientations{};      ///< Actual exact signs for p-q-r, p-q-s, r-s-p, r-s-q.
  std::size_t        orientation_tests{}; ///< Actual sign calls, exactly four.
  std::size_t
      bounding_tests{}; ///< Actual closed bounding-box checks for zero-orientation candidates.
  Point2D
      first{}; ///< Point or lexicographically lower overlap endpoint; zero sentinel for None/unconstructed proper contact.
  Point2D second{};            ///< Same unique point or lexicographically upper overlap endpoint.
  bool    point_constructed{}; ///< Actual point/overlap coordinates populated.
  long double
      parameter{}; ///< Actual approximate first-segment parameter for proper-crossing construction; zero otherwise.
};
/// @brief Return the exact orientation enum of three stored finite binary64 points.
/// Delegates to the canonical bounded dyadic integer predicate, without epsilon or overflow-prone floating products. Constant bounded work/storage; inputs unchanged.
/// @param a First finite point.
/// @param b Second finite point.
/// @param c Third finite point.
/// @return Exact collinear, clockwise or counterclockwise enum.
/// @throws std::invalid_argument Nonfinite coordinate.
/// @par Example
/// @code
/// std::print("{}\n",int(dm::geometry::orientation({0,0},{1,0},{0,1}))); // 2
/// @endcode
/// @par References
/// - Exact determinant motivation: Shewchuk (1997), https://www.cs.cmu.edu/~quake/robust.html ; standard closed-segment orientation and collinear bounding-box test.
/// @see orientation_sign()
Orientation orientation(const Point2D& a, const Point2D& b, const Point2D& c);
/// @brief Classify exact closed-segment contact and construct its unique point or overlap bounds.
/// Evaluates four canonical exact orientation signs. Opposite nonzero signs on both baselines identify a proper crossing; zero signs use closed coordinate bounds. Collinear endpoint/point-segment cases return Point; nonzero interval overlaps return Overlap with lexicographically ordered endpoint bounds. Proper crossing coordinates use a widened quotient of exactly accumulated determinants, then convex interpolation and final double narrowing. The contact decision is exact for represented finite coordinates; a constructed interior point is approximate, not a correctly rounded rational certificate. Trace and run use the same fixed-size producing record, with no replay or dynamic history. Constant bounded work/storage, no shared state; borrowed inputs unchanged.
/// @param p First segment start.
/// @param q First segment end.
/// @param r Second segment start.
/// @param s Second segment end.
/// @return Owned exact contact, actual predicates, and approximate point/exact overlap endpoint coordinates.
/// @throws std::invalid_argument Nonfinite coordinate.
/// @throws std::overflow_error A nonzero construction quotient or final point is outside the finite represented range.
/// @par Example
/// @code
/// auto t=dm::geometry::segment_intersection_run({0,0},{2,2},{0,2},{2,0});
/// std::print("{} {} {}\n",int(t.contact),t.first.x,t.first.y); // 1 1 1
/// @endcode
/// @par References
/// - Exact determinant motivation: Shewchuk (1997), https://www.cs.cmu.edu/~quake/robust.html ; standard closed-segment orientation and collinear bounding-box test.
/// @see segments_intersect(), cross_difference_ratio()
SegmentIntersectionRun
segment_intersection_run(const Point2D& p, const Point2D& q, const Point2D& r, const Point2D& s);
/// @brief Classify exact closed-segment contact and construct its unique point or overlap bounds.
/// Evaluates four canonical exact orientation signs. Opposite nonzero signs on both baselines identify a proper crossing; zero signs use closed coordinate bounds. Collinear endpoint/point-segment cases return Point; nonzero interval overlaps return Overlap with lexicographically ordered endpoint bounds. Proper crossing coordinates use a widened quotient of exactly accumulated determinants, then convex interpolation and final double narrowing. The contact decision is exact for represented finite coordinates; a constructed interior point is approximate, not a correctly rounded rational certificate. Trace and run use the same fixed-size producing record, with no replay or dynamic history. Constant bounded work/storage, no shared state; borrowed inputs unchanged.
/// @param p First segment start.
/// @param q First segment end.
/// @param r Second segment start.
/// @param s Second segment end.
/// @return Owned exact contact, actual predicates, and approximate point/exact overlap endpoint coordinates.
/// @throws std::invalid_argument Nonfinite coordinate.
/// @throws std::overflow_error A nonzero construction quotient or final point is outside the finite represented range.
/// @par Example
/// @code
/// auto t=dm::geometry::segment_intersection_trace({0,0},{2,2},{0,2},{2,0});
/// std::print("{} {} {}\n",int(t.contact),t.first.x,t.first.y); // 1 1 1
/// @endcode
/// @par References
/// - Exact determinant motivation: Shewchuk (1997), https://www.cs.cmu.edu/~quake/robust.html ; standard closed-segment orientation and collinear bounding-box test.
/// @see segments_intersect(), cross_difference_ratio()
SegmentIntersectionRun
segment_intersection_trace(const Point2D& p, const Point2D& q, const Point2D& r, const Point2D& s);
/// @brief Decide exact nonempty closed-segment contact through the preserved signature.
/// Uses the same four-sign classifier without constructing any approximate coordinates; endpoint touching, collinear overlap and zero-length-segment contact are included. Constant bounded work/storage; no epsilon.
/// @param p1 First segment start.
/// @param q1 First segment end.
/// @param p2 Second segment start.
/// @param q2 Second segment end.
/// @return True exactly when the represented closed segments have nonempty intersection.
/// @throws std::invalid_argument Any nonfinite coordinate.
/// @par Example
/// @code
/// std::print("{}\n",dm::geometry::segments_intersect({0,0},{2,0},{2,0},{3,0})); // true
/// @endcode
/// @par References
/// - Exact determinant motivation: Shewchuk (1997), https://www.cs.cmu.edu/~quake/robust.html ; standard closed-segment orientation and collinear bounding-box test.
/// @see segment_intersection_run()
bool segments_intersect(const Point2D& p1, const Point2D& q1, const Point2D& p2, const Point2D& q2);
/// @brief Construct the approximate unique intersection of two infinite supporting lines.
/// Uses exact determinant parallelism and a widened exact-determinant quotient before final coordinate construction. Returns false for parallel, coincident or degenerate supporting lines, even when finite segments overlap. Does not restrict line parameters to [0,1]. On false/error leaves out unchanged. Constructed coordinates are approximate and must be finite doubles; classification is exact for represented inputs. Constant bounded work/storage.
/// @param p1 First line first point.
/// @param p2 First line second point.
/// @param p3 Second line first point.
/// @param p4 Second line second point.
/// @param out Borrowed output, assigned only on successful construction.
/// @return True when a finite approximate unique intersection was constructed.
/// @throws std::invalid_argument Nonfinite coordinate.
/// @throws std::overflow_error Nonzero quotient or final coordinate exceeds finite represented range.
/// @par Example
/// @code
/// dm::geometry::Point2D out;bool ok=dm::geometry::line_intersection_point({0,0},{2,2},{0,2},{2,0},out);
/// std::print("{} {}\n",ok,out.x); // true 1
/// @endcode
/// @par References
/// - Exact determinant motivation: Shewchuk (1997), https://www.cs.cmu.edu/~quake/robust.html ; standard closed-segment orientation and collinear bounding-box test.
/// @see segment_intersection_point()
bool line_intersection_point(
    const Point2D& p1,
    const Point2D& p2,
    const Point2D& p3,
    const Point2D& p4,
    Point2D&       out);
/// @brief Construct the unique point of closed-segment contact through the preserved signature.
/// Returns false for disjoint or nonzero-overlap intersections and leaves out unchanged. Correctly returns true for a single collinear endpoint or zero-length contact, fixing the historical determinant-zero shortcut. Proper crossing construction is approximate after exact classification. Constant bounded work/storage.
/// @param p1 First segment start.
/// @param q1 First segment end.
/// @param p2 Second segment start.
/// @param q2 Second segment end.
/// @param out Borrowed output, assigned only for unique-point success.
/// @return True exactly for a successfully constructed unique-point contact.
/// @throws std::invalid_argument Nonfinite coordinate.
/// @throws std::overflow_error Construction quotient or final point outside finite represented range.
/// @par Example
/// @code
/// dm::geometry::Point2D out;bool ok=dm::geometry::segment_intersection_point({0,0},{2,0},{2,0},{3,0},out);
/// std::print("{} {}\n",ok,out.x); // true 2
/// @endcode
/// @par References
/// - Exact determinant motivation: Shewchuk (1997), https://www.cs.cmu.edu/~quake/robust.html ; standard closed-segment orientation and collinear bounding-box test.
/// @see segment_intersection_run()
bool segment_intersection_point(
    const Point2D& p1,
    const Point2D& q1,
    const Point2D& p2,
    const Point2D& q2,
    Point2D&       out);
} // namespace datamunge::geometry
