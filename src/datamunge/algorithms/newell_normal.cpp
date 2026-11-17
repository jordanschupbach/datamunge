/// @file newell_normal.cpp
/// @brief Translated wide cyclic Newell producer and actual centroid-plane diagnostics.
#include <algorithm>
#include <cmath>
#include <datamunge/algorithms/newell_normal.hpp>
#include <limits>
#include <stdexcept>
namespace datamunge::algorithms {
/// @brief Private finite conversion and shared actual edge producer.
namespace {
/// @brief Convert one wide finite result to a checked finite binary64, allowing underflow.
/// @param value Actual native wide arithmetic result.
/// @return Rounded finite double.
/// @throws std::overflow_error If either representation is nonfinite.
double narrow(long double value) {
  double result = double(value);
  if (!std::isfinite(value) || !std::isfinite(result))
    throw std::overflow_error("Newell: nonfinite output");
  return result;
}
/// @brief Execute complete finite admission, translated cyclic sum and centroid-plane residuals.
/// @param poly Original complete vertices, any count.
/// @param out Optional owned actual-history sink.
/// @param cap Complete logical history policy, zero disables.
/// @param diagnostics Whether to allocate/produce per-vertex plane residuals.
/// @return Actual vector-area result and signed diagnostic distances.
/// @throws std::invalid_argument If a complete original coordinate is nonfinite.
/// @throws std::length_error If count geometry/capture policy is exceeded.
/// @throws std::overflow_error If finite arithmetic/output representation fails.
NewellAnalysis
run(const std::vector<Point3D>& poly, NewellTrace* out, std::size_t cap, bool diagnostics) {
  for (auto& p : poly)
    for (double x : p)
      if (!std::isfinite(x))
        throw std::invalid_argument("Newell: finite complete coordinates required");
  const auto n = poly.size();
  if (n > (std::numeric_limits<std::size_t>::max() - 3) / 5)
    throw std::length_error("Newell: complete history geometry");
  NewellWidePoint origin{};
  if (n)
    for (int k = 0; k < 3; ++k)
      origin[k] = poly[0][k];
  if (out) {
    out->captured_entries = 3 + 5 * n;
    if (cap && out->captured_entries > cap)
      throw std::length_error("Newell: complete capture policy");
    out->polygon = poly;
    out->origin  = origin;
    out->edges.reserve(n);
  }
  NewellAnalysis  a;
  NewellWidePoint sum{}, centroid{};
  for (std::size_t i = 0; i < n; ++i) {
    NewellEdge e;
    e.index = i;
    e.next  = (i + 1) % n;
    for (int k = 0; k < 3; ++k) {
      e.from[k] = static_cast<long double>(poly[i][k]) - origin[k];
      e.to[k]   = static_cast<long double>(poly[e.next][k]) - origin[k];
      centroid[k] += e.from[k];
      if (!std::isfinite(e.from[k]) || !std::isfinite(e.to[k]) || !std::isfinite(centroid[k]))
        throw std::overflow_error("Newell: wide translation/centroid overflow");
    }
    for (int k = 0; k < 3; ++k) {
      e.cross[k] =
          e.from[(k + 1) % 3] * e.to[(k + 2) % 3] - e.from[(k + 2) % 3] * e.to[(k + 1) % 3];
      sum[k] += e.cross[k];
      if (!std::isfinite(e.cross[k]) || !std::isfinite(sum[k]))
        throw std::overflow_error("Newell: wide edge/sum overflow");
    }
    e.sum_after = sum;
    ++a.edges;
    if (out)
      out->edges.push_back(e);
  }
  const long double length = std::hypot(sum[0], sum[1], sum[2]);
  if (!std::isfinite(length))
    throw std::overflow_error("Newell: wide norm overflow");
  a.has_direction = length > 0;
  a.vector_area   = narrow(length / 2);
  for (int k = 0; k < 3; ++k) {
    a.raw[k]          = narrow(sum[k]);
    a.unit[k]         = length ? narrow(sum[k] / length) : 0;
    a.plane_origin[k] = origin[k] + (n ? centroid[k] / n : 0);
    if (!std::isfinite(a.plane_origin[k]))
      throw std::overflow_error("Newell: plane origin overflow");
  }
  if (diagnostics) {
    a.signed_plane_distances.reserve(n);
    for (auto& p : poly) {
      long double distance = 0;
      if (a.has_direction)
        for (int k = 0; k < 3; ++k)
          distance += (static_cast<long double>(p[k]) - a.plane_origin[k]) * a.unit[k];
      double value = narrow(distance);
      a.signed_plane_distances.push_back(value);
      a.max_plane_distance = std::max(a.max_plane_distance, std::abs(value));
    }
  }
  return a;
}
} // namespace
Point3D newell_normal_raw(const std::vector<Point3D>& p) {
  return run(p, nullptr, 0, false).raw;
}
Point3D newell_normal(const std::vector<Point3D>& p) {
  return run(p, nullptr, 0, false).unit;
}
double newell_polygon_area(const std::vector<Point3D>& p) {
  return run(p, nullptr, 0, false).vector_area;
}
NewellAnalysis newell_analyze(const std::vector<Point3D>& p) {
  return run(p, nullptr, 0, true);
}
NewellTrace newell_trace(const std::vector<Point3D>& p, std::size_t cap) {
  NewellTrace t;
  t.analysis = run(p, &t, cap, true);
  return t;
}
std::vector<NewellPolygonFixture> newell_demo_polygons() {
  return {
      {"original square", {{0, 0, 0}, {2, 0, 0}, {2, 2, 0}, {0, 2, 0}}},
      {"original triangle", {{0, 0, 0}, {1, 0, 1}, {0, 1, 1}}},
      {"original tilted quad", {{0, 0, 0}, {2, 0, .1}, {2, 2, 0}, {0, 2, -.1}}},
      {"actual warped quad", {{0, 0, 0}, {2, 0, .1}, {2, 2, 0}, {0, 2, .3}}}};
}
} // namespace datamunge::algorithms
