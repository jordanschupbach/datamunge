#pragma once
/// @file newell_normal.hpp
/// @brief Checked translated Newell vector areas, unit directions and actual cyclic-edge histories.
#include <array>
#include <cstddef>
#include <string>
#include <vector>
namespace datamunge::algorithms {
/// @brief Owned three-dimensional binary64 Cartesian point, x/y/z order.
/// @par Example
/// @code
/// dm::algorithms::Point3D p{1,2,3};std::print("{}\n",p[2]); // 3
/// @endcode
/// @see newell_analyze()
using Point3D = std::array<double, 3>;
/// @brief Three long-double native arithmetic components, x/y/z order.
/// @par Example
/// @code
/// dm::algorithms::NewellWidePoint p{};std::print("{}\n",double(p[0])); // 0
/// @endcode
/// @see NewellEdge
using NewellWidePoint = std::array<long double, 3>;
/// @brief Actual vector area, normalized direction and centroid-plane residual diagnostics.
/// Nonplanar magnitude is a vector-area magnitude, not a chosen triangulated
/// surface area. Self-intersecting boundaries can cancel; no simplicity/planarity
/// certificate or fitted least-squares normal is supplied.
/// @par Example
/// @code
/// auto a=dm::algorithms::newell_analyze({{0,0,0},{2,0,0},{2,2,0},{0,2,0}});std::print("{}\n",a.vector_area); // 4
/// @endcode
/// @see newell_trace()
struct NewellAnalysis {
  Point3D raw{}; ///< Rounded twice-vector-area sum, sum of actual translated cyclic cross products.
  Point3D unit{}; ///< Actual rounded unit direction; zero placeholder when has_direction=false.
  double  vector_area{
      0}; ///< Half length of wide raw sum; equals area only for a simple planar boundary.
  NewellWidePoint plane_origin{}; ///< Actual wide arithmetic centroid used for diagnostic plane.
  std::vector<double>
      signed_plane_distances; ///< Actual signed distances to the centroid plane using returned unit; zero placeholders if undefined.
  double max_plane_distance{
      0}; ///< Actual maximum absolute diagnostic distance, zero if no direction.
  bool has_direction{
      false}; ///< Whether the computed wide vector sum has nonzero norm; not a validity certificate.
  std::size_t edges{0}; ///< Actual cyclic edges processed, equal to supplied vertex count.
};
/// @brief Actual translated endpoint cross product and running wide sum.
/// @par Example
/// @code
/// auto t=dm::algorithms::newell_trace({{0,0,0},{1,0,0},{0,1,0}});std::print("{}\n",double(t.edges[1].cross[2])); // 1
/// @endcode
/// @see newell_trace()
struct NewellEdge {
  std::size_t     index{0};    ///< Actual original edge start index.
  std::size_t     next{0};     ///< Actual next original vertex index, cyclic.
  NewellWidePoint from{};      ///< Actual original start minus first-vertex origin.
  NewellWidePoint to{};        ///< Actual original end minus first-vertex origin.
  NewellWidePoint cross{};     ///< Actual from cross to contribution.
  NewellWidePoint sum_after{}; ///< Actual wide running sum after this edge.
};
/// @brief Owned complete original polygon and actual translated edge observations.
/// Logical units count original coordinates3N, actual edge records N, returned
/// signed distances N and origin3. Fixed analysis metadata/record fields are
/// excluded from that policy; it is not a capacity/byte/peak-memory estimate.
/// @par Example
/// @code
/// auto t=dm::algorithms::newell_trace({{0,0,0},{1,0,0},{0,1,0}});std::print("{}\n",t.captured_entries); // 18
/// @endcode
/// @see newell_trace()
struct NewellTrace {
  std::vector<Point3D> polygon; ///< Owned original finite vertices in caller order.
  NewellWidePoint origin{};     ///< Actual first vertex used for translation, zero for empty input.
  NewellAnalysis  analysis;     ///< Actual same-producer result and plane diagnostics.
  std::vector<NewellEdge> edges;               ///< Actual original cyclic order, no replay.
  std::size_t             captured_entries{0}; ///< Complete logical history units, exactly3+5N.
};
/// @brief Return the rounded twice-vector-area of the original cyclic boundary.
/// Complete coordinates are checked finite before arithmetic/policy checks. Empty,
/// singleton, two-point, repeated and self-intersecting inputs are accepted;
/// a zero computed vector returns zero unit/area placeholders. This does not
/// certify zero physical surface area, collinearity, planarity or a simple polygon.
/// Translating every vertex by the first vertex makes sum(p_i cross p_next)
/// algebraically equal to Newell's original cyclic formula while reducing large
/// translation cancellation. Computation uses long double with finite checks;
/// raw/area/residual outputs must remain finite binary64, underflow may round zero.
/// Unit normalization uses the wide norm, so tiny nonzero vectors can still have
/// a direction even when rounded raw/area are zero. Winding reversal negates
/// direction; nonplanar vector area is not a unique surface area or best-fit plane.
/// O(N) work, O(1) driver storage, no mutation/retained input references.
/// @param poly Complete original finite three-coordinate vertices in cyclic order.
/// @return Owned native Point3D result, with the stated degenerate conventions.
/// @throws std::invalid_argument If any complete input coordinate is nonfinite.
/// @throws std::overflow_error If wide arithmetic or finite binary64 output is unrepresentable.
/// @throws std::length_error If count geometry is exceeded.
/// @throws std::bad_alloc If diagnostic/history allocation fails.
/// @par Example
/// @code
/// auto n=dm::algorithms::newell_normal_raw({{0,0,0},{1,0,0},{0,1,0}});std::print("{}\n",n[2]); // 1
/// @endcode
/// @par References
/// - F. Tampieri(1992), Newell's Method for Computing the Plane Equation of a Polygon, Graphics Gems III,231–232; https://github.com/erich666/GraphicsGems/blob/master/gemsiii/newell.c.
/// @see NewellAnalysis, NewellTrace
Point3D newell_normal_raw(const std::vector<Point3D>& poly);
/// @brief Return the unit direction of the native wide vector-area sum.
/// Complete coordinates are checked finite before arithmetic/policy checks. Empty,
/// singleton, two-point, repeated and self-intersecting inputs are accepted;
/// a zero computed vector returns zero unit/area placeholders. This does not
/// certify zero physical surface area, collinearity, planarity or a simple polygon.
/// Translating every vertex by the first vertex makes sum(p_i cross p_next)
/// algebraically equal to Newell's original cyclic formula while reducing large
/// translation cancellation. Computation uses long double with finite checks;
/// raw/area/residual outputs must remain finite binary64, underflow may round zero.
/// Unit normalization uses the wide norm, so tiny nonzero vectors can still have
/// a direction even when rounded raw/area are zero. Winding reversal negates
/// direction; nonplanar vector area is not a unique surface area or best-fit plane.
/// O(N) work, O(1) driver storage, no mutation/retained input references.
/// @param poly Complete original finite three-coordinate vertices in cyclic order.
/// @return Owned native Point3D result, with the stated degenerate conventions.
/// @throws std::invalid_argument If any complete input coordinate is nonfinite.
/// @throws std::overflow_error If wide arithmetic or finite binary64 output is unrepresentable.
/// @throws std::length_error If count geometry is exceeded.
/// @throws std::bad_alloc If diagnostic/history allocation fails.
/// @par Example
/// @code
/// auto n=dm::algorithms::newell_normal({{0,0,0},{1,0,0},{0,1,0}});std::print("{}\n",n[2]); // 1
/// @endcode
/// @par References
/// - F. Tampieri(1992), Newell's Method for Computing the Plane Equation of a Polygon, Graphics Gems III,231–232; https://github.com/erich666/GraphicsGems/blob/master/gemsiii/newell.c.
/// @see NewellAnalysis, NewellTrace
Point3D newell_normal(const std::vector<Point3D>& poly);
/// @brief Return half the vector-area norm, a planar-boundary area under stated conditions.
/// Complete coordinates are checked finite before arithmetic/policy checks. Empty,
/// singleton, two-point, repeated and self-intersecting inputs are accepted;
/// a zero computed vector returns zero unit/area placeholders. This does not
/// certify zero physical surface area, collinearity, planarity or a simple polygon.
/// Translating every vertex by the first vertex makes sum(p_i cross p_next)
/// algebraically equal to Newell's original cyclic formula while reducing large
/// translation cancellation. Computation uses long double with finite checks;
/// raw/area/residual outputs must remain finite binary64, underflow may round zero.
/// Unit normalization uses the wide norm, so tiny nonzero vectors can still have
/// a direction even when rounded raw/area are zero. Winding reversal negates
/// direction; nonplanar vector area is not a unique surface area or best-fit plane.
/// O(N) work, O(1) driver storage, no mutation/retained input references.
/// @param poly Complete original finite three-coordinate vertices in cyclic order.
/// @return Owned native double result, with the stated degenerate conventions.
/// @throws std::invalid_argument If any complete input coordinate is nonfinite.
/// @throws std::overflow_error If wide arithmetic or finite binary64 output is unrepresentable.
/// @throws std::length_error If count geometry is exceeded.
/// @throws std::bad_alloc If diagnostic/history allocation fails.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::newell_polygon_area({{0,0,0},{1,0,0},{0,1,0}})); // 0.5
/// @endcode
/// @par References
/// - F. Tampieri(1992), Newell's Method for Computing the Plane Equation of a Polygon, Graphics Gems III,231–232; https://github.com/erich666/GraphicsGems/blob/master/gemsiii/newell.c.
/// @see NewellAnalysis, NewellTrace
double newell_polygon_area(const std::vector<Point3D>& poly);
/// @brief Return actual Newell vector area and centroid-plane diagnostics.
/// Complete coordinates are checked finite before arithmetic/policy checks. Empty,
/// singleton, two-point, repeated and self-intersecting inputs are accepted;
/// a zero computed vector returns zero unit/area placeholders. This does not
/// certify zero physical surface area, collinearity, planarity or a simple polygon.
/// Translating every vertex by the first vertex makes sum(p_i cross p_next)
/// algebraically equal to Newell's original cyclic formula while reducing large
/// translation cancellation. Computation uses long double with finite checks;
/// raw/area/residual outputs must remain finite binary64, underflow may round zero.
/// Unit normalization uses the wide norm, so tiny nonzero vectors can still have
/// a direction even when rounded raw/area are zero. Winding reversal negates
/// direction; nonplanar vector area is not a unique surface area or best-fit plane.
/// O(N) work, O(N) diagnostic storage, no mutation/retained input references.
/// @param poly Complete original finite three-coordinate vertices in cyclic order.
/// @return Owned native NewellAnalysis result, with the stated degenerate conventions.
/// @throws std::invalid_argument If any complete input coordinate is nonfinite.
/// @throws std::overflow_error If wide arithmetic or finite binary64 output is unrepresentable.
/// @throws std::length_error If count geometry is exceeded.
/// @throws std::bad_alloc If diagnostic/history allocation fails.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::newell_analyze({}).has_direction); // false
/// @endcode
/// @par References
/// - F. Tampieri(1992), Newell's Method for Computing the Plane Equation of a Polygon, Graphics Gems III,231–232; https://github.com/erich666/GraphicsGems/blob/master/gemsiii/newell.c.
/// @see NewellAnalysis, NewellTrace
NewellAnalysis newell_analyze(const std::vector<Point3D>& poly);
/// @brief Capture every actual translated cyclic-edge contribution.
/// Complete coordinates are checked finite before arithmetic/policy checks. Empty,
/// singleton, two-point, repeated and self-intersecting inputs are accepted;
/// a zero computed vector returns zero unit/area placeholders. This does not
/// certify zero physical surface area, collinearity, planarity or a simple polygon.
/// Translating every vertex by the first vertex makes sum(p_i cross p_next)
/// algebraically equal to Newell's original cyclic formula while reducing large
/// translation cancellation. Computation uses long double with finite checks;
/// raw/area/residual outputs must remain finite binary64, underflow may round zero.
/// Unit normalization uses the wide norm, so tiny nonzero vectors can still have
/// a direction even when rounded raw/area are zero. Winding reversal negates
/// direction; nonplanar vector area is not a unique surface area or best-fit plane.
/// O(N) work, O(N) diagnostic storage, no mutation/retained input references.
/// @param poly Complete original finite three-coordinate vertices in cyclic order.
/// @param max_entries Complete3+5N logical history cap, default1000000; zero disables, before copies/arithmetic.
/// @return Owned actual history and result.
/// @throws std::invalid_argument If any complete input coordinate is nonfinite.
/// @throws std::overflow_error If wide arithmetic or finite binary64 output is unrepresentable.
/// @throws std::length_error If count geometry or complete capture quota is exceeded.
/// @throws std::bad_alloc If diagnostic/history allocation fails.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::newell_trace({}).captured_entries); // 3
/// @endcode
/// @par References
/// - F. Tampieri(1992), Newell's Method for Computing the Plane Equation of a Polygon, Graphics Gems III,231–232; https://github.com/erich666/GraphicsGems/blob/master/gemsiii/newell.c.
/// @see NewellAnalysis, NewellTrace
NewellTrace newell_trace(const std::vector<Point3D>& poly, std::size_t max_entries = 1000000);
/// @brief Named original or explicitly corrected nonplanar polygon fixture.
/// @par Example
/// @code
/// auto v=dm::algorithms::newell_demo_polygons();std::print("{}\n",v.size()); // 4
/// @endcode
/// @see newell_trace()
struct NewellPolygonFixture {
  std::string          name;    ///< Owned fixture identity.
  std::vector<Point3D> polygon; ///< Owned original cyclic vertices.
};
/// @brief Preserve the original square, tilted triangle and purportedly warped quad, plus an actual warped control.
/// Original quad z values0,.1,0,-.1 lie exactly on z=.05x-.05y up to input
/// rounding. Corrected control changes its fourth z to.3, making it nonplanar.
/// @return Four owned fixtures in original square/triangle/quad/warped order.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::newell_demo_polygons()[0].polygon.size()); // 4
/// @endcode
/// @see newell_analyze()
std::vector<NewellPolygonFixture> newell_demo_polygons();
} // namespace datamunge::algorithms
