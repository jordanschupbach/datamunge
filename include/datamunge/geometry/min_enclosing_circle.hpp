#pragma once
/// @file min_enclosing_circle.hpp
/// @brief Seeded incremental enclosing circles with actual containment/rebuild events.
#include <cstddef>
#include <datamunge/geometry/point2d.hpp>
#include <vector>
namespace datamunge::geometry {
/// @brief Owned planar center and nonnegative radius in coordinate units.
/// Storage alone does not validate geometry; checked producers require finiteness.
/// @par Example
/// @code
/// auto c=dm::geometry::min_enclosing_circle({{0,0},{2,0}});
/// std::print("{} {}\n",c.center.x,c.radius); // 1 1
/// @endcode
/// @see min_enclosing_circle()
struct Circle {
  Point2D center;    ///< Owned circle center.
  double  radius{0}; ///< Nonnegative radius for valid native results.
};
/// @brief Actual containment test and optional immediate boundary rebuild.
/// Depth0 tests the next outer point,1 tests a preceding point under one fixed
/// boundary point,2 tests under two. A failed test immediately rebuilds from
/// one/two/three points respectively. Later nested events can rebuild again.
/// @par Example
/// @code
/// auto t=dm::geometry::min_enclosing_circle_trace({{0,0},{2,0}});
/// std::print("{}\n",t.events.size()); // 2
/// @endcode
/// @see EnclosingCircleTrace
struct EnclosingCircleEvent {
  unsigned    depth{0};         ///< Actual nested-loop depth,0 through2.
  std::size_t input_index{0};   ///< Original identity of the tested point.
  bool        contained{false}; ///< Actual tolerance-aware containment result.
  Circle      before;           ///< Actual circle before this test.
  Circle      after;            ///< Circle immediately after this test/rebuild.
};
/// @brief Owned original/order data and actual seeded incremental history.
/// Capture units count original points, shuffled point copies, permutation IDs
/// and event records. Scalar final/initial circle fields are not additional units.
/// @par Example
/// @code
/// auto t=dm::geometry::min_enclosing_circle_trace({{0,0},{2,0}});
/// std::print("{}\n",t.result.radius); // 1
/// @endcode
/// @see min_enclosing_circle_trace()
struct EnclosingCircleTrace {
  std::vector<Point2D>              input;       ///< Owned original finite coordinates.
  std::vector<Point2D>              shuffled;    ///< Actual MT19937-64 seed42 processing order.
  std::vector<std::size_t>          permutation; ///< Original identity at every shuffled position.
  Circle                            initial;     ///< Singleton circle at the first shuffled point.
  Circle                            result;      ///< Actual final tolerance-aware enclosing circle.
  std::vector<EnclosingCircleEvent> events;      ///< Actual tests and immediate rebuilds.
  std::size_t                       captured_entries{0}; ///< Actual admitted capture units.
};
/// @brief Tolerance-aware distances and containment diagnostics for a supplied circle.
/// Residual is hypot(point-center)-radius. A point passes when residual is at
/// most1e-9*max(1,radius). Counts describe tolerance, not exact support identities.
/// @par Example
/// @code
/// auto a=dm::geometry::enclosing_circle_check({{1,0},1},{{0,0},{2,0}});
/// std::print("{} {}\n",a.outside,a.near_boundary); // 0 2
/// @endcode
/// @see enclosing_circle_check()
struct EnclosingCircleCheck {
  std::vector<double> residuals;  ///< Owned signed radial residuals, original input order.
  std::size_t         outside{0}; ///< Residuals greater than the same containment tolerance.
  std::size_t         near_boundary{
      0}; ///< Absolute residuals within tolerance; not necessarily essential support.
};
/// @brief Compute a seeded incremental enclosing circle using three nested loops.
/// Fixed MT19937-64 seed42 and std::shuffle preserve ordinary processing order
/// on the same standard library. One/two/three-point boundary rebuilds use
/// translated long-double geometry, exact orientation for collinearity, and
/// overflow checks. Collinear triples use the farthest-pair diameter.
/// Containment tolerance is1e-9*max(1,radius), so the returned circle is a
/// numerical approximation, not an exact enclosure/minimality certificate.
/// O(n^3) worst-case tests, O(n) copied driver storage. No expected-linear
/// guarantee for this particular loop implementation or timing claim.
/// @param points Nonempty finite points; duplicates and collinearity allowed.
/// @return Owned finite center/radius.
/// @throws std::invalid_argument If input is empty or nonfinite.
/// @throws std::overflow_error If rebuilt center/radius is unrepresentable.
/// @par Example
/// @code
/// auto c=dm::geometry::min_enclosing_circle({{0,0},{2,0},{1,0}});
/// std::print("{} {}\n",c.center.x,c.radius); // 1 1
/// @endcode
/// @par References
/// - E. Welzl (1991), Smallest enclosing disks (balls and ellipsoids), LNCS555,359–370, doi:10.1007/BFb0038202 (boundary-set framework; explicit nested-loop variant here).
/// @see min_enclosing_circle_trace(), enclosing_circle_check()
Circle min_enclosing_circle(std::vector<Point2D> points);
/// @brief Capture actual containment/rebuild events of the ordinary producer.
/// No replay. Three copies/identity units per input are pre-admitted, then one
/// per actual test. No partial trace escapes a quota failure. History storage
/// is proportional to admitted entries; zero disables the optional cap.
/// @param points Nonempty finite points, retained by value in original order.
/// @param max_entries Capture cap, default1000000; zero disables.
/// @return Owned actual order, events and final circle.
/// @throws std::invalid_argument If input is empty or nonfinite.
/// @throws std::overflow_error If circle geometry is unrepresentable.
/// @throws std::length_error If quota/count geometry exceeds its limit.
/// @throws std::bad_alloc If allocation fails.
/// @par Example
/// @code
/// auto t=dm::geometry::min_enclosing_circle_trace({{0,0},{2,0}});
/// std::print("{}\n",t.captured_entries); // 8
/// @endcode
/// @see EnclosingCircleTrace
EnclosingCircleTrace min_enclosing_circle_trace(
    std::vector<Point2D> points,
    std::size_t          max_entries = 1000000);
/// @brief Check supplied points using the producer's radial containment tolerance.
/// O(n) work/storage. No mutation, fitting or exact support/minimality assertion.
/// @param circle Finite center and finite nonnegative radius.
/// @param points Finite points, empty allowed.
/// @return Owned original-order residuals and tolerance-aware counts.
/// @throws std::invalid_argument If any geometry is invalid/nonfinite.
/// @throws std::overflow_error If a residual cannot be represented as double.
/// @par Example
/// @code
/// std::print("{}\n",dm::geometry::enclosing_circle_check({{1,0},1},{{0,0},{2,0}}).outside); // 0
/// @endcode
/// @see min_enclosing_circle()
EnclosingCircleCheck enclosing_circle_check(Circle circle, const std::vector<Point2D>& points);
/// @brief Generate the original eighty-point normal cloud.
/// Uses std::mt19937 seed2718281 and normal_distribution(0,2), drawing x then y.
/// Distribution implementation can differ across standard libraries.
/// @return Owned original80 finite coordinates.
/// @par Example
/// @code
/// std::print("{}\n",dm::geometry::enclosing_circle_demo().size()); // 80
/// @endcode
/// @see min_enclosing_circle()
std::vector<Point2D> enclosing_circle_demo();
/// @brief Sample a supplied circle's boundary for plotting, with a closing duplicate.
/// @param circle Finite center/nonnegative finite radius.
/// @param segments Positive segment count, default100, at most1000000.
/// @return Owned segments+1 points, last equal to first.
/// @throws std::invalid_argument If circle is invalid or segment count is zero.
/// @throws std::length_error If segment count exceeds1000000.
/// @throws std::overflow_error If a boundary coordinate is unrepresentable.
/// @par Example
/// @code
/// std::print("{}\n",dm::geometry::circle_boundary({{0,0},1},4).size()); // 5
/// @endcode
/// @see Circle
std::vector<Point2D> circle_boundary(Circle circle, std::size_t segments = 100);
/// @brief Historical helper names retained for existing generated bindings.
namespace detail {
/// @brief Construct a singleton zero-radius circle with finite-point validation.
/// @param p Finite point.
/// @return Owned singleton circle.
/// @throws std::invalid_argument If p is nonfinite.
/// @par Example
/// @code
/// std::print("{}\n",dm::geometry::detail::circle_from_one_point({1,2}).radius); // 0
/// @endcode
/// @see min_enclosing_circle()
Circle circle_from_one_point(const Point2D& p);
/// @brief Construct a two-point diameter circle using checked wide midpoint arithmetic.
/// @param a First finite endpoint.
/// @param b Second finite endpoint.
/// @return Owned midpoint circle; duplicates give zero radius.
/// @throws std::invalid_argument If a coordinate is nonfinite.
/// @throws std::overflow_error If radius/center is unrepresentable.
/// @par Example
/// @code
/// std::print("{}\n",dm::geometry::detail::circle_from_two_points({0,0},{2,0}).radius); // 1
/// @endcode
/// @see min_enclosing_circle()
Circle circle_from_two_points(const Point2D& a, const Point2D& b);
/// @brief Construct a translated circumcircle or a farthest-pair collinear diameter.
/// @param a First finite boundary point.
/// @param b Second finite boundary point.
/// @param c Third finite boundary point.
/// @return Owned checked circle through noncollinear boundary points, or collinear diameter.
/// @throws std::invalid_argument If a coordinate is nonfinite.
/// @throws std::overflow_error If circle arithmetic is unrepresentable.
/// @par Example
/// @code
/// std::print("{}\n",dm::geometry::detail::circle_from_three_points({0,0},{4,0},{0,3}).radius); // 2.5
/// @endcode
/// @see min_enclosing_circle()
Circle circle_from_three_points(const Point2D& a, const Point2D& b, const Point2D& c);
/// @brief Test a point with the same tolerance used by the incremental producer.
/// @param circle Finite center and finite nonnegative radius.
/// @param p Finite query point.
/// @return Distance<=radius+1e-9*max(1,radius).
/// @throws std::invalid_argument If circle or point is invalid.
/// @par Example
/// @code
/// std::print("{}\n",dm::geometry::detail::circle_contains({{0,0},1},{1,0})); // true
/// @endcode
/// @see enclosing_circle_check()
bool circle_contains(const Circle& circle, const Point2D& p);
} // namespace detail
} // namespace datamunge::geometry
