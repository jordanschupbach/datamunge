/// @file gjk.cpp
/// @brief Shared checked legacy-ordered planar GJK kernel and online state capture.
#include <cmath>
#include <datamunge/geometry/gjk.hpp>
#include <limits>
#include <numbers>
#include <stdexcept>

namespace datamunge::geometry {
/// @brief Private arithmetic, feature reduction and per-call execution state.
namespace {
/// @brief Reject a nonfinite evaluated arithmetic result without changing finite values.
/// @param value Actual double result.
/// @return The unchanged finite value, including signed zero.
/// @throws std::overflow_error The result is not finite.
double finite_result(double value) {
  if (!std::isfinite(value))
    throw std::overflow_error("GJK: nonfinite arithmetic intermediate");
  return value;
}
/// @brief Check a complete cloud before any budget, history or empty-input handling.
/// @param points Input coordinates, without mutation.
/// @throws std::invalid_argument A coordinate is nonfinite.
void validate_points(const std::vector<Point2D>& points) {
  for (const auto& p : points)
    if (!std::isfinite(p.x) || !std::isfinite(p.y))
      throw std::invalid_argument("GJK: finite coordinates required");
}
/// @brief Increment one selected-operation counter with a representability check.
/// @param count Counter updated in place.
/// @throws std::length_error The increment cannot be represented.
void increment(std::size_t& count) {
  if (count == std::numeric_limits<std::size_t>::max())
    throw std::length_error("GJK: operation counter overflow");
  ++count;
}
/// @brief Evaluate the original ordered double dot product with finite-result checks.
/// @param a First vector.
/// @param b Second vector.
/// @return Unchanged finite product sum.
/// @throws std::overflow_error A product or sum is nonfinite.
double dot(const Point2D& a, const Point2D& b) {
  return finite_result(finite_result(a.x * b.x) + finite_result(a.y * b.y));
}
/// @brief Evaluate checked component differences in their original double arithmetic.
/// @param a First point.
/// @param b Point to subtract.
/// @return Component difference.
/// @throws std::overflow_error A difference is nonfinite.
Point2D sub(const Point2D& a, const Point2D& b) {
  return {finite_result(a.x - b.x), finite_result(a.y - b.y)};
}
/// @brief Evaluate the original a+t*d formula with checked products and sums.
/// @param a Starting point.
/// @param d Displacement vector.
/// @param t Actual interpolation coefficient.
/// @return Actual rounded interpolation result.
/// @throws std::overflow_error An intermediate or result is nonfinite.
Point2D add_scaled(const Point2D& a, const Point2D& d, double t) {
  return {finite_result(a.x + finite_result(t * d.x)), finite_result(a.y + finite_result(t * d.y))};
}
/// @brief One actual closest-triangle return and its owned supporting feature.
struct TriangleResult {
  Point2D              closest;       ///< Actual closest point, or zero for the inside branch.
  std::vector<Point2D> feature;       ///< One, two or three copied retained points.
  bool                 inside{false}; ///< Whether the original final triangle branch was taken.
};
/// @brief Apply the original ordered triangle Voronoi-region tests to the origin.
/// Double predicates are approximate and no exact containment certificate is made.
/// @param a First point.
/// @param b Second point.
/// @param c Third point.
/// @return Actual retained feature and closest point, with the final inside flag.
/// @throws std::overflow_error Any evaluated intermediate is nonfinite.
/// @throws std::bad_alloc Small feature allocation fails.
/// @par References
/// - C. Ericson (2005), Real-Time Collision Detection, Morgan Kaufmann,
///   Section5.1.5. The original planar region tests are retained.
///   https://realtimecollisiondetection.net/books/rtcd/toc/index.html
TriangleResult closest_triangle(const Point2D& a, const Point2D& b, const Point2D& c) {
  const Point2D O{0, 0};
  const Point2D ab = sub(b, a), ac = sub(c, a), ap = sub(O, a);
  const double  d1 = dot(ab, ap), d2 = dot(ac, ap);
  if (d1 <= 0 && d2 <= 0)
    return {a, {a}, false};
  const Point2D bp = sub(O, b);
  const double  d3 = dot(ab, bp), d4 = dot(ac, bp);
  if (d3 >= 0 && d4 <= d3)
    return {b, {b}, false};
  const double vc = finite_result(finite_result(d1 * d4) - finite_result(d3 * d2));
  if (vc <= 0 && d1 >= 0 && d3 <= 0) {
    const double v = finite_result(d1 / finite_result(d1 - d3));
    return {add_scaled(a, ab, v), {a, b}, false};
  }
  const Point2D cp = sub(O, c);
  const double  d5 = dot(ab, cp), d6 = dot(ac, cp);
  if (d6 >= 0 && d5 <= d6)
    return {c, {c}, false};
  const double vb = finite_result(finite_result(d5 * d2) - finite_result(d1 * d6));
  if (vb <= 0 && d2 >= 0 && d6 <= 0) {
    const double w = finite_result(d2 / finite_result(d2 - d6));
    return {add_scaled(a, ac, w), {a, c}, false};
  }
  const double va = finite_result(finite_result(d3 * d6) - finite_result(d5 * d4));
  if (va <= 0 && finite_result(d4 - d3) >= 0 && finite_result(d5 - d6) >= 0) {
    const double w = finite_result(
        finite_result(d4 - d3) / finite_result(finite_result(d4 - d3) + finite_result(d5 - d6)));
    const Point2D bc = sub(c, b);
    return {add_scaled(b, bc, w), {b, c}, false};
  }
  return {O, {a, b, c}, true};
}
/// @brief Return the original closest point on a segment and its retained feature.
/// @param a First segment endpoint.
/// @param b Second segment endpoint.
/// @param feature Replaced with one or two copied retained points.
/// @return Actual double interpolation, including the coincident-endpoint case.
/// @throws std::overflow_error Any evaluated intermediate is nonfinite.
/// @throws std::bad_alloc Small feature allocation fails.
/// @par References
/// - C. Ericson (2005), Real-Time Collision Detection, Section5.1.2.
///   https://realtimecollisiondetection.net/books/rtcd/toc/index.html
Point2D closest_segment(const Point2D& a, const Point2D& b, std::vector<Point2D>& feature) {
  const Point2D O{0, 0};
  const Point2D ab  = sub(b, a);
  const double  den = dot(ab, ab);
  const double  t   = den > 0 ? finite_result(dot(sub(O, a), ab) / den) : 0.;
  if (t <= 0) {
    feature = {a};
    return a;
  }
  if (t >= 1) {
    feature = {b};
    return b;
  }
  feature = {a, b};
  return add_scaled(a, ab, t);
}
/// @brief Per-call kernel state shared by ordinary, diagnostic and history APIs.
/// Input references live only for the call; no state escapes except owned results.
class Kernel {
  const std::vector<Point2D>& a_; ///< First validated cloud, borrowed for this call only.
  const std::vector<Point2D>& b_; ///< Second validated cloud, borrowed for this call only.
  std::vector<GjkEvent>*
              events_; ///< Optional caller-owned output history, or null for ordinary execution.
  std::size_t max_records_;  ///< Optional complete-record quota; zero turns it off.
  std::size_t support_a_{0}; ///< Most recent selected index of the first cloud.
  std::size_t support_b_{0}; ///< Most recent selected index of the second cloud.
  GjkRun      result_{};     ///< Owned result and selected counters for this execution.

  /// @brief Find the first maximum support vertex while counting actual scans.
  /// @param points Nonempty validated cloud.
  /// @param direction Actual finite query vector.
  /// @param selected Updated to the winning first-tie index.
  /// @return Copied support point.
  /// @throws std::overflow_error A support dot intermediate is nonfinite.
  /// @throws std::length_error A counter overflows.
  Point2D
  support(const std::vector<Point2D>& points, const Point2D& direction, std::size_t& selected) {
    increment(result_.work.support_calls);
    increment(result_.work.support_dots);
    selected    = 0;
    double best = dot(points[0], direction);
    for (std::size_t i = 1; i < points.size(); ++i) {
      increment(result_.work.support_dots);
      const double candidate = dot(points[i], direction);
      increment(result_.work.support_comparisons);
      if (candidate > best) {
        best     = candidate;
        selected = i;
      }
    }
    return points[selected];
  }
  /// @brief Compute one actual Minkowski support point using both complete scans.
  /// @param direction Finite query vector.
  /// @return Actual rounded difference of selected points.
  /// @throws std::overflow_error A dot or coordinate difference is nonfinite.
  /// @throws std::length_error A selected counter overflows.
  Point2D minkowski(const Point2D& direction) {
    return sub(
        support(a_, direction, support_a_), support(b_, {-direction.x, -direction.y}, support_b_));
  }
  /// @brief Append one whole fixed-size actual-state record without numerical replay.
  /// @param kind Emission point.
  /// @param iteration Actual loop index, or zero for initial/empty.
  /// @param point Actual closest or selected support point according to kind.
  /// @param direction Actual search direction.
  /// @param value Actual evaluated scalar, or zero according to kind.
  /// @param simplex Current or returned feature to copy.
  /// @throws std::length_error The next complete record exceeds quota or capacity.
  /// @throws std::bad_alloc History allocation fails.
  void emit(
      GjkEventKind                kind,
      std::size_t                 iteration,
      Point2D                     point,
      Point2D                     direction,
      double                      value,
      const std::vector<Point2D>& simplex) {
    if (!events_)
      return;
    if ((max_records_ && events_->size() >= max_records_) || events_->size() == events_->max_size())
      throw std::length_error("GJK: complete record quota/capacity exceeded");
    GjkEvent e;
    e.kind         = kind;
    e.iteration    = iteration;
    e.support_a    = support_a_;
    e.support_b    = support_b_;
    e.point        = point;
    e.direction    = direction;
    e.value        = value;
    e.simplex_size = simplex.size();
    for (std::size_t i = 0; i < simplex.size(); ++i)
      e.simplex[i] = simplex[i];
    events_->push_back(e);
  }
  /// @brief Store the final actual approximation and copied working feature.
  /// @param value Returned norm or conventional zero.
  /// @param closest Actual final working closest point.
  /// @param simplex Actual final feature, with at most three points.
  /// @param reason Actual return branch.
  /// @return Owned ordinary result.
  GjkRun
  finish(double value, Point2D closest, const std::vector<Point2D>& simplex, GjkStopReason reason) {
    result_.distance     = value;
    result_.closest      = closest;
    result_.simplex_size = simplex.size();
    result_.stop         = reason;
    for (std::size_t i = 0; i < simplex.size(); ++i)
      result_.simplex[i] = simplex[i];
    return result_;
  }

 public:
  /// @brief Bind per-call input/history state without performing kernel work.
  /// @param a First input cloud, validated by execute().
  /// @param b Second input cloud, validated by execute().
  /// @param events Optional owned history to append, or null.
  /// @param max_records Optional complete-record quota; zero turns it off.
  Kernel(
      const std::vector<Point2D>& a,
      const std::vector<Point2D>& b,
      std::vector<GjkEvent>*      events,
      std::size_t                 max_records)
      : a_(a), b_(b), events_(events), max_records_(max_records) {}
  /// @brief Execute complete validation and the original ordered planar loop.
  /// @param max_iterations Maximum loop visits; zero still initializes support.
  /// @return Owned final approximation, reason, feature and counters.
  /// @throws std::invalid_argument An input coordinate is nonfinite.
  /// @throws std::overflow_error An evaluated arithmetic intermediate is nonfinite.
  /// @throws std::length_error A history quota/capacity or counter is exceeded.
  /// @throws std::bad_alloc Working or history allocation fails.
  GjkRun execute(std::size_t max_iterations) {
    validate_points(a_);
    validate_points(b_);
    if (a_.empty() || b_.empty()) {
      emit(GjkEventKind::empty, 0, {0, 0}, {0, 0}, 0, {});
      return finish(0, {0, 0}, {}, GjkStopReason::empty_input);
    }
    Point2D direction = sub(a_[0], b_[0]);
    if (direction.x == 0 && direction.y == 0)
      direction = {1, 0};
    std::vector<Point2D> simplex{minkowski(direction)};
    Point2D              closest = simplex[0];
    emit(GjkEventKind::initial, 0, closest, direction, 0, simplex);
    for (std::size_t iteration = 0; iteration < max_iterations; ++iteration) {
      increment(result_.work.iterations);
      direction          = {-closest.x, -closest.y};
      const double norm2 = dot(direction, direction);
      emit(GjkEventKind::direction, iteration, closest, direction, norm2, simplex);
      if (norm2 < 1e-20) {
        emit(GjkEventKind::near, iteration, closest, direction, 0, simplex);
        return finish(0, closest, simplex, GjkStopReason::near_origin);
      }
      const Point2D point       = minkowski(direction);
      const double  improvement = finite_result(dot(point, direction) - dot(closest, direction));
      increment(result_.work.progress_checks);
      emit(GjkEventKind::support, iteration, point, direction, improvement, simplex);
      if (improvement < 1e-10) {
        const double answer = std::sqrt(dot(closest, closest));
        emit(GjkEventKind::progress, iteration, closest, direction, answer, simplex);
        return finish(answer, closest, simplex, GjkStopReason::progress_tolerance);
      }
      simplex.push_back(point);
      if (simplex.size() == 2) {
        increment(result_.work.segment_reductions);
        std::vector<Point2D> feature;
        closest = closest_segment(simplex[0], simplex[1], feature);
        simplex = feature;
        emit(GjkEventKind::segment, iteration, closest, direction, 0, simplex);
      } else {
        increment(result_.work.triangle_reductions);
        const auto reduced = closest_triangle(simplex[0], simplex[1], simplex[2]);
        if (reduced.inside) {
          emit(
              GjkEventKind::triangle_inside,
              iteration,
              reduced.closest,
              direction,
              0,
              reduced.feature);
          return finish(
              0, reduced.closest, reduced.feature, GjkStopReason::triangle_contains_origin);
        }
        closest = reduced.closest;
        simplex = reduced.feature;
        emit(GjkEventKind::triangle, iteration, closest, direction, 0, simplex);
      }
    }
    const double answer = std::sqrt(dot(closest, closest));
    emit(GjkEventKind::limit, max_iterations, closest, direction, answer, simplex);
    return finish(answer, closest, simplex, GjkStopReason::iteration_limit);
  }
};
} // namespace

const char* gjk_stop_name(GjkStopReason reason) noexcept {
  switch (reason) {
    case GjkStopReason::empty_input:
      return "empty_input";
    case GjkStopReason::near_origin:
      return "near_origin";
    case GjkStopReason::progress_tolerance:
      return "progress_tolerance";
    case GjkStopReason::triangle_contains_origin:
      return "triangle_contains_origin";
    case GjkStopReason::iteration_limit:
      return "iteration_limit";
  }
  return "unknown";
}
double gjk_distance(const std::vector<Point2D>& A, const std::vector<Point2D>& B) {
  return Kernel(A, B, nullptr, 0).execute(100).distance;
}
GjkRun gjk_distance_run(
    const std::vector<Point2D>& A,
    const std::vector<Point2D>& B,
    std::size_t                 max_iterations) {
  return Kernel(A, B, nullptr, 0).execute(max_iterations);
}
GjkTrace gjk_distance_trace(
    const std::vector<Point2D>& A,
    const std::vector<Point2D>& B,
    std::size_t                 max_iterations,
    std::size_t                 max_records) {
  GjkTrace result;
  result.run = Kernel(A, B, &result.events, max_records).execute(max_iterations);
  return result;
}
std::vector<Point2D> regular_polygon(
    Point2D     center,
    double      radius,
    std::size_t sides,
    double      rotation,
    std::size_t max_vertices) {
  if (!std::isfinite(center.x) || !std::isfinite(center.y) || !std::isfinite(radius) ||
      !std::isfinite(rotation) || radius < 0 || sides < 3)
    throw std::invalid_argument(
        "regular_polygon: finite center/radius/rotation, nonnegative radius and at least three "
        "sides required");
  std::vector<Point2D> points;
  if ((max_vertices && sides > max_vertices) || sides > points.max_size())
    throw std::length_error("regular_polygon: whole vertex count exceeds quota/capacity");
  points.reserve(sides);
  for (std::size_t i = 0; i < sides; ++i) {
    const double angle = finite_result(
        rotation + finite_result(
                       finite_result(2 * std::numbers::pi * static_cast<double>(i)) /
                       static_cast<double>(sides)));
    points.push_back(
        {finite_result(center.x + finite_result(radius * std::cos(angle))),
         finite_result(center.y + finite_result(radius * std::sin(angle)))});
  }
  return points;
}
} // namespace datamunge::geometry
