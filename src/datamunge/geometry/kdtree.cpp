/// @file kdtree.cpp
/// @brief Deterministic balanced build and shared widened nearest/k/radius traversal kernels.
#include <algorithm>
#include <cmath>
#include <datamunge/geometry/kdtree.hpp>
#include <limits>
#include <stdexcept>
#include <tuple>
namespace datamunge::geometry {
/// @brief Finite validation, widened metric and actual record admission details.
namespace kd_detail {
/// @brief Validate one finite Cartesian coordinate pair.
/// @param q Borrowed point.
/// @throws std::invalid_argument On any nonfinite coordinate.
void validate(const Point2D& q) {
  if (!std::isfinite(q.x) || !std::isfinite(q.y))
    throw std::invalid_argument("KDTree finite point");
}
/// @brief Compute widened subtraction/hypot distance without raw double squaring.
/// @param a First finite point.
/// @param b Second finite point.
/// @return Actual rounded long-double Euclidean distance.
/// @throws std::overflow_error On nonfinite widened result.
long double distance(const Point2D& a, const Point2D& b) {
  const auto d =
      std::hypot(static_cast<long double>(a.x) - b.x, static_cast<long double>(a.y) - b.y);
  if (!std::isfinite(d))
    throw std::overflow_error("KDTree widened distance");
  return d;
}
/// @brief Read one coordinate as a widened scalar.
/// @param p Finite point.
/// @param axis Split axis0/1.
/// @return Selected coordinate, exactly widened from double.
long double coordinate(const Point2D& p, int axis) {
  return axis ? p.y : p.x;
}
/// @brief Admit an actual trace event before its producing observation/decision.
/// @param t Optional actual trace, null for constant-storage query.
/// @param cap Optional actual event ceiling, zero disables.
/// @throws std::length_error On enabled ceiling or vector capacity.
void admit(KDNearestTrace* t, std::size_t cap) {
  if (t && ((cap && t->probes.size() >= cap) || t->probes.size() == t->probes.max_size()))
    throw std::length_error("KDTree actual record ceiling");
}
} // namespace kd_detail
KDTree2D::KDTree2D(std::vector<Point2D> points, std::size_t cap) : size_(points.size()) {
  for (const auto& p : points)
    kd_detail::validate(p);
  if ((cap && points.size() > cap) ||
      points.size() > std::size_t(std::numeric_limits<std::ptrdiff_t>::max()))
    throw std::length_error("KDTree complete points");
  std::vector<Item> items;
  items.reserve(points.size());
  for (std::size_t i = 0; i < points.size(); ++i)
    items.push_back({points[i], i});
  root_ = build(items, 0, items.size(), 0);
}
std::unique_ptr<KDTree2D::Node>
KDTree2D::build(std::vector<Item>& items, std::size_t lo, std::size_t hi, std::size_t depth) {
  if (lo == hi)
    return {};
  const int  axis = depth % 2;
  const auto mid  = lo + (hi - lo) / 2;
  std::nth_element(
      items.begin() + static_cast<std::ptrdiff_t>(lo),
      items.begin() + static_cast<std::ptrdiff_t>(mid),
      items.begin() + static_cast<std::ptrdiff_t>(hi),
      [axis](const Item& a, const Item& b) {
        if (axis)
          return std::tie(a.point.y, a.point.x, a.index) < std::tie(b.point.y, b.point.x, b.index);
        return std::tie(a.point.x, a.point.y, a.index) < std::tie(b.point.x, b.point.y, b.index);
      });
  auto node   = std::make_unique<Node>();
  node->item  = items[mid];
  node->axis  = axis;
  node->left  = build(items, lo, mid, depth + 1);
  node->right = build(items, mid + 1, hi, depth + 1);
  return node;
}
bool KDTree2D::Compare::operator()(const Entry& a, const Entry& b) const {
  return a.distance < b.distance ||
         (a.distance == b.distance && a.node->item.index < b.node->item.index);
}
void KDTree2D::nearest_recurse(
    const Node*     node,
    const Point2D&  q,
    Entry&          best,
    KDNearestTrace* t,
    std::size_t     cap) {
  if (!node)
    return;
  kd_detail::admit(t, cap);
  const auto before  = best;
  const auto measure = kd_detail::distance(q, node->item.point);
  if (!best.node || measure < best.distance ||
      (measure == best.distance && node->item.index < best.node->item.index))
    best = {measure, node};
  const auto missing = std::numeric_limits<std::size_t>::max();
  if (t) {
    ++t->work.visited_nodes;
    t->probes.push_back(
        {KDProbeAction::Visit,
         node->item.index,
         missing,
         node->axis,
         measure,
         before.node ? before.node->item.index : missing,
         best.node->item.index,
         before.distance,
         best.distance,
         false});
  }
  const auto diff =
      kd_detail::coordinate(q, node->axis) - kd_detail::coordinate(node->item.point, node->axis);
  const auto near = diff < 0 ? node->left.get() : node->right.get(),
             far  = diff < 0 ? node->right.get() : node->left.get();
  nearest_recurse(near, q, best, t, cap);
  if (far) {
    kd_detail::admit(t, cap);
    const auto bound   = std::fabs(diff);
    const bool explore = bound <= best.distance;
    if (t) {
      ++t->work.plane_tests;
      t->work.pruned_subtrees += !explore;
      t->probes.push_back(
          {KDProbeAction::FarPlane,
           node->item.index,
           far->item.index,
           node->axis,
           bound,
           best.node->item.index,
           best.node->item.index,
           best.distance,
           best.distance,
           explore});
    }
    if (explore)
      nearest_recurse(far, q, best, t, cap);
  }
}
Point2D KDTree2D::nearest(const Point2D& q) const {
  kd_detail::validate(q);
  if (!root_)
    throw std::logic_error("KDTree nearest on empty");
  Entry best{std::numeric_limits<long double>::infinity(), nullptr};
  nearest_recurse(root_.get(), q, best, nullptr, 0);
  return best.node->item.point;
}
void KDTree2D::original_points(const Node* node, std::vector<Point2D>& points) {
  if (!node)
    return;
  points[node->item.index] = node->item.point;
  original_points(node->left.get(), points);
  original_points(node->right.get(), points);
}
KDNearestTrace KDTree2D::nearest_trace(const Point2D& q, std::size_t cap) const {
  kd_detail::validate(q);
  if (!root_)
    throw std::logic_error("KDTree nearest on empty");
  KDNearestTrace t;
  t.query = q;
  t.points.resize(size_);
  original_points(root_.get(), t.points);
  Entry best{std::numeric_limits<long double>::infinity(), nullptr};
  nearest_recurse(root_.get(), q, best, &t, cap);
  t.point          = best.node->item.point;
  t.original_index = best.node->item.index;
  t.distance       = best.distance;
  return t;
}
void KDTree2D::k_recurse(const Node* node, const Point2D& q, std::size_t k, Heap& heap) {
  if (!node)
    return;
  const Entry candidate{kd_detail::distance(q, node->item.point), node};
  if (heap.size() < k)
    heap.push(candidate);
  else if (Compare{}(candidate, heap.top())) {
    heap.pop();
    heap.push(candidate);
  }
  const auto diff =
      kd_detail::coordinate(q, node->axis) - kd_detail::coordinate(node->item.point, node->axis);
  const auto near = diff < 0 ? node->left.get() : node->right.get(),
             far  = diff < 0 ? node->right.get() : node->left.get();
  k_recurse(near, q, k, heap);
  if (heap.size() < k || std::fabs(diff) <= heap.top().distance)
    k_recurse(far, q, k, heap);
}
std::vector<Point2D> KDTree2D::k_nearest(const Point2D& q, std::size_t k) const {
  kd_detail::validate(q);
  if (!root_ || !k)
    return {};
  k = std::min(k, size_);
  Heap heap;
  k_recurse(root_.get(), q, k, heap);
  std::vector<Point2D> out;
  out.reserve(heap.size());
  while (!heap.empty()) {
    out.push_back(heap.top().node->item.point);
    heap.pop();
  }
  std::reverse(out.begin(), out.end());
  return out;
}
void KDTree2D::radius_recurse(
    const Node*           node,
    const Point2D&        q,
    long double           radius,
    std::vector<Point2D>& out) {
  if (!node)
    return;
  if (std::isinf(radius) || kd_detail::distance(q, node->item.point) <= radius)
    out.push_back(node->item.point);
  const auto diff =
      kd_detail::coordinate(q, node->axis) - kd_detail::coordinate(node->item.point, node->axis);
  const auto near = diff < 0 ? node->left.get() : node->right.get(),
             far  = diff < 0 ? node->right.get() : node->left.get();
  radius_recurse(near, q, radius, out);
  if (std::fabs(diff) <= radius)
    radius_recurse(far, q, radius, out);
}
std::vector<Point2D> KDTree2D::points_in_radius(const Point2D& q, double radius) const {
  kd_detail::validate(q);
  if (std::isnan(radius) || radius < 0)
    throw std::invalid_argument("KDTree radius");
  std::vector<Point2D> out;
  radius_recurse(root_.get(), q, radius, out);
  return out;
}
} // namespace datamunge::geometry
