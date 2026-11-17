/// @file min_enclosing_circle.cpp
/// @brief Shared incremental circle producer with translated wide-precision boundary geometry.
#include <algorithm>
#include <cmath>
#include <datamunge/geometry/min_enclosing_circle.hpp>
#include <datamunge/geometry/orientation.hpp>
#include <limits>
#include <numbers>
#include <numeric>
#include <random>
#include <stdexcept>
namespace datamunge::geometry {
/// @brief Private checked geometric and capture helpers.
namespace {
/// @brief Validate one stored circle before diagnostics or boundary sampling.
/// @param c Circle to check.
/// @throws std::invalid_argument If geometry is nonfinite or radius negative.
void validate(Circle c) {
  if (!std::isfinite(c.center.x) || !std::isfinite(c.center.y) || !std::isfinite(c.radius) ||
      c.radius < 0)
    throw std::invalid_argument("circle: finite nonnegative geometry required");
}
/// @brief Compute distance in translated long-double arithmetic.
/// @param a First finite point.
/// @param b Second finite point.
/// @return Wide Euclidean distance.
long double dist(Point2D a, Point2D b) {
  return std::hypot(static_cast<long double>(a.x) - b.x, static_cast<long double>(a.y) - b.y);
}
/// @brief Construct a finite stored circle from wide arithmetic.
/// @param x Wide center x.
/// @param y Wide center y.
/// @param radius Wide radius.
/// @return Representable stored circle.
/// @throws std::overflow_error If center/radius cannot be represented.
Circle checked(long double x, long double y, long double radius) {
  Circle c{{double(x), double(y)}, double(radius)};
  if (!std::isfinite(c.center.x) || !std::isfinite(c.center.y) || !std::isfinite(c.radius))
    throw std::overflow_error("circle: unrepresentable boundary geometry");
  return c;
}
/// @brief Form a two-point diameter circle without overflowing endpoint sums.
/// @param a First finite endpoint.
/// @param b Second finite endpoint.
/// @return Checked midpoint and half-distance circle.
Circle two(Point2D a, Point2D b) {
  return checked(
      (static_cast<long double>(a.x) + b.x) / 2,
      (static_cast<long double>(a.y) + b.y) / 2,
      dist(a, b) / 2);
}
/// @brief Form a translated circumcircle or farthest-pair collinear diameter.
/// @param a First finite boundary point.
/// @param b Second finite boundary point.
/// @param c Third finite boundary point.
/// @return Checked boundary circle.
Circle three(Point2D a, Point2D b, Point2D c) {
  if (orientation_sign(a, b, c) == 0) {
    auto ab = dist(a, b), bc = dist(b, c), ac = dist(a, c);
    if (ab >= bc && ab >= ac)
      return two(a, b);
    if (bc >= ab && bc >= ac)
      return two(b, c);
    return two(a, c);
  }
  const long double bx   = static_cast<long double>(b.x) - a.x,
                    by   = static_cast<long double>(b.y) - a.y,
                    cx   = static_cast<long double>(c.x) - a.x,
                    cy   = static_cast<long double>(c.y) - a.y;
  const auto determinant = 2 * (bx * cy - by * cx);
  if (determinant == 0)
    throw std::overflow_error("circle: unresolved circumcircle determinant");
  const auto b2 = bx * bx + by * by, c2 = cx * cx + cy * cy;
  const auto ux = (b2 * cy - c2 * by) / determinant, uy = (bx * c2 - cx * b2) / determinant;
  return checked(
      static_cast<long double>(a.x) + ux, static_cast<long double>(a.y) + uy, std::hypot(ux, uy));
}
/// @brief Execute the actual three-loop circle producer, optionally capturing tests.
/// @param points Owned finite input, validated before shuffling.
/// @param out Optional actual-state sink.
/// @param cap Capture quota, zero disables.
/// @return Actual final circle.
/// @throws std::invalid_argument If input is empty/nonfinite.
/// @throws std::overflow_error If boundary geometry is unrepresentable.
/// @throws std::length_error If capture quota is exceeded.
Circle run(std::vector<Point2D> points, EnclosingCircleTrace* out, std::size_t cap) {
  if (points.empty())
    throw std::invalid_argument("min_enclosing_circle: nonempty input required");
  for (auto p : points)
    if (!std::isfinite(p.x) || !std::isfinite(p.y))
      throw std::invalid_argument("min_enclosing_circle: finite points required");
  /// @brief Admit actual owned entries before copy or append.
  auto admit = [&](std::size_t amount) {
    if (!out)
      return;
    if (amount > std::numeric_limits<std::size_t>::max() - out->captured_entries ||
        (cap && (out->captured_entries > cap || amount > cap - out->captured_entries)))
      throw std::length_error("min_enclosing_circle: quota exceeded");
    out->captured_entries += amount;
  };
  if (out) {
    admit(points.size());
    admit(points.size());
    admit(points.size());
    out->input = points;
  }
  std::vector<std::size_t> order(points.size());
  std::iota(order.begin(), order.end(), 0);
  std::mt19937_64 rng(42);
  std::shuffle(order.begin(), order.end(), rng);
  auto original = std::move(points);
  points.clear();
  points.reserve(original.size());
  for (auto index : order)
    points.push_back(original[index]);
  if (out) {
    out->permutation = order;
    out->shuffled    = points;
  }
  Circle circle{points[0], 0};
  if (out)
    out->initial = circle;
  /// @brief Test the current circle and capture its immediate actual rebuild.
  auto test = [&](std::size_t index, unsigned depth, std::size_t i, std::size_t j) {
    const auto before = circle;
    const bool contained =
        dist(circle.center, points[index]) <=
        static_cast<long double>(circle.radius) + 1e-9L * std::max(1.0, circle.radius);
    if (out)
      admit(1);
    if (!contained) {
      if (depth == 0)
        circle = {points[index], 0};
      else if (depth == 1)
        circle = two(points[i], points[index]);
      else
        circle = three(points[i], points[j], points[index]);
    }
    if (out)
      out->events.push_back({depth, order[index], contained, before, circle});
    return contained;
  };
  for (std::size_t i = 1; i < points.size(); ++i) {
    if (test(i, 0, i, 0))
      continue;
    for (std::size_t j = 0; j < i; ++j) {
      if (test(j, 1, i, j))
        continue;
      for (std::size_t k = 0; k < j; ++k)
        test(k, 2, i, j);
    }
  }
  return circle;
}
} // namespace
Circle min_enclosing_circle(std::vector<Point2D> points) {
  return run(std::move(points), nullptr, 0);
}
EnclosingCircleTrace min_enclosing_circle_trace(std::vector<Point2D> points, std::size_t cap) {
  EnclosingCircleTrace t;
  t.result = run(std::move(points), &t, cap);
  return t;
}
EnclosingCircleCheck enclosing_circle_check(Circle c, const std::vector<Point2D>& points) {
  validate(c);
  EnclosingCircleCheck a;
  for (auto p : points) {
    if (!std::isfinite(p.x) || !std::isfinite(p.y))
      throw std::invalid_argument("enclosing_circle_check: finite points required");
    auto residual = dist(c.center, p) - c.radius;
    auto value    = double(residual);
    if (!std::isfinite(value))
      throw std::overflow_error("enclosing_circle_check: residual overflow");
    a.residuals.push_back(value);
    auto eps = 1e-9L * std::max(1.0, c.radius);
    a.outside += residual > eps;
    a.near_boundary += std::abs(residual) <= eps;
  }
  return a;
}
std::vector<Point2D> enclosing_circle_demo() {
  std::mt19937                     rng(2718281);
  std::normal_distribution<double> normal(0, 2);
  std::vector<Point2D>             points;
  for (int i = 0; i < 80; ++i) {
    auto x = normal(rng), y = normal(rng);
    points.push_back({x, y});
  }
  return points;
}
std::vector<Point2D> circle_boundary(Circle c, std::size_t segments) {
  validate(c);
  if (!segments)
    throw std::invalid_argument("circle_boundary: positive segments required");
  if (segments > 1000000)
    throw std::length_error("circle_boundary: too many segments");
  std::vector<Point2D> p;
  p.reserve(segments + 1);
  for (std::size_t i = 0; i < segments; ++i) {
    long double a = 2 * std::numbers::pi_v<long double> * i / segments;
    auto        q = checked(
        c.center.x + static_cast<long double>(c.radius) * std::cos(a),
        c.center.y + static_cast<long double>(c.radius) * std::sin(a),
        0);
    p.push_back(q.center);
  }
  p.push_back(p.front());
  return p;
}
namespace detail {
Circle circle_from_one_point(const Point2D& p) {
  if (!std::isfinite(p.x) || !std::isfinite(p.y))
    throw std::invalid_argument("circle_from_one_point: finite point required");
  return {p, 0};
}
Circle circle_from_two_points(const Point2D& a, const Point2D& b) {
  circle_from_one_point(a);
  circle_from_one_point(b);
  return two(a, b);
}
Circle circle_from_three_points(const Point2D& a, const Point2D& b, const Point2D& c) {
  circle_from_one_point(a);
  circle_from_one_point(b);
  circle_from_one_point(c);
  return three(a, b, c);
}
bool circle_contains(const Circle& circle, const Point2D& p) {
  validate(circle);
  circle_from_one_point(p);
  return dist(circle.center, p) <=
         static_cast<long double>(circle.radius) + 1e-9L * std::max(1.0, circle.radius);
}
} // namespace detail
} // namespace datamunge::geometry
