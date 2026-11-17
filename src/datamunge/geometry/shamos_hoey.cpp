/// @file shamos_hoey.cpp
/// @brief Actual checked endpoint sweep, native AVL status and exact orientation searches.
#include <algorithm>
#include <cmath>
#include <datamunge/geometry/shamos_hoey.hpp>
#include <stdexcept>
#include <tuple>
/// @brief Canonical checked native producers and owned algorithm values.
namespace datamunge::geometry {
/// @brief Private invocation-local checked helpers.
namespace {
/// @brief Owned normalized finite segment retaining its caller id by vector address.
struct Segment {
  Point2D p;          ///< Left or lower endpoint.
  Point2D q;          ///< Right or upper endpoint.
  bool    vertical{}; ///< Equal represented x coordinates.
};
/// @brief One actual pending native endpoint/vertical event.
struct Event {
  double          x{};    ///< Actual x coordinate.
  ShamosEventKind kind{}; ///< Equal-x category order.
  std::size_t     id{};   ///< Original segment identity.
};
/// @brief Native AVL node with stable arena address and mutable original-identity payload.
struct Node {
  int         left{-1};   ///< Left child arena index.
  int         right{-1};  ///< Right child arena index.
  int         parent{-1}; ///< Parent arena index.
  int         height{1};  ///< Actual AVL subtree height.
  std::size_t id{};       ///< Original segment currently occupying this tree position.
};
/// @brief One independent producing sweep and its owned actual work/history.
struct Sweep {
  const ShamosOptions& options;  ///< Borrowed policies.
  bool                 capture;  ///< Retain actual optional records.
  std::vector<Segment> segments; ///< Owned normalized complete geometry.
  std::vector<Node>    nodes;    ///< Stable-address integer-index arena.
  std::vector<int>     where;    ///< Original identity to its currently active node, or minus one.
  int                  root{-1}; ///< Actual AVL root arena address.
  std::size_t          active{}; ///< Actual active nonvertical count.
  std::size_t          retained{}; ///< Actual admitted history records.
  ShamosRun            out;        ///< Owned actual result/work/history.

  /// @brief Admit actual incremental policy units without overflow.
  /// @param used Existing count.
  /// @param n Next units.
  /// @param cap Zero or enabled cap.
  void admit(std::size_t used, std::size_t n, std::size_t cap) {
    if (n > std::numeric_limits<std::size_t>::max() - used ||
        (cap && (used > cap || n > cap - used)))
      throw std::length_error("Shamos-Hoey: capacity/policy");
  }
  /// @brief Admit one actual optional history record before its corresponding operation.
  void record() {
    if (capture) {
      admit(retained, 1, options.max_trace_entries);
      ++retained;
    }
  }
  /// @brief Read actual subtree height with empty-node convention.
  /// @param x Arena address or minus one.
  /// @return Actual subtree height, zero when empty.
  int height(int x) const { return x < 0 ? 0 : nodes[x].height; }
  /// @brief Refresh actual cached subtree height from its children.
  /// @param x Nonempty arena address.
  void update(int x) {
    nodes[x].height = 1 + std::max(height(nodes[x].left), height(nodes[x].right));
  }
  /// @brief Rotate one actual AVL subtree and reconnect its parent.
  /// @param x Actual imbalanced subtree root.
  /// @param left Whether this is a left rotation, otherwise right.
  /// @return Actual new subtree root arena address.
  int rotate(int x, bool left) {
    int y = left ? nodes[x].right : nodes[x].left, p = nodes[x].parent,
        middle = left ? nodes[y].left : nodes[y].right;
    if (p < 0)
      root = y;
    else if (nodes[p].left == x)
      nodes[p].left = y;
    else
      nodes[p].right = y;
    nodes[y].parent = p;
    if (left) {
      nodes[x].right = middle;
      nodes[y].left  = x;
    } else {
      nodes[x].left  = middle;
      nodes[y].right = x;
    }
    if (middle >= 0)
      nodes[middle].parent = x;
    nodes[x].parent = y;
    update(x);
    update(y);
    ++out.rotations;
    return y;
  }
  /// @brief Restore actual AVL height balance iteratively up the changed ancestor chain.
  /// @param x First changed ancestor, or minus one.
  void balance(int x) {
    while (x >= 0) {
      update(x);
      const auto delta = height(nodes[x].left) - height(nodes[x].right);
      if (delta > 1) {
        int child = nodes[x].left;
        if (height(nodes[child].right) > height(nodes[child].left))
          rotate(child, true);
        x = rotate(x, false);
      } else if (delta < -1) {
        int child = nodes[x].right;
        if (height(nodes[child].left) > height(nodes[child].right))
          rotate(child, false);
        x = rotate(x, true);
      }
      x = nodes[x].parent;
    }
  }
  /// @brief Count and execute one actual exact active-height query.
  /// @param id Incoming or vertical identity.
  /// @param other Actual active identity.
  /// @param point Actual finite query point at current event x.
  /// @return Exact active-line/query orientation sign.
  int compare(std::size_t id, std::size_t other, const Point2D& point) {
    admit(out.height_tests, 1, options.max_height_tests);
    record();
    ++out.height_tests;
    const int sign = orientation_sign(segments[other].p, segments[other].q, point);
    if (capture)
      out.heights.push_back({id, other, sign});
    return sign;
  }
  /// @brief Count and execute one actual exact complete segment test, recording a discovered witness.
  /// @param a First original identity.
  /// @param b Second original identity.
  /// @return Actual nonempty contact decision.
  bool hit(std::size_t a, std::size_t b) {
    admit(out.pair_tests, 1, options.max_pair_tests);
    record();
    ++out.pair_tests;
    const bool value =
        segments_intersect(segments[a].p, segments[a].q, segments[b].p, segments[b].q);
    if (capture)
      out.pairs.push_back({a, b, value});
    if (value) {
      out.found  = true;
      out.first  = a;
      out.second = b;
    }
    return value;
  }
  /// @brief Read an actual inorder predecessor/successor by child/parent links.
  /// @param x Active arena address.
  /// @param next True for successor, false for predecessor.
  /// @return Actual neighbor arena address or minus one.
  int neighbor(int x, bool next) const {
    int child = next ? nodes[x].right : nodes[x].left;
    if (child >= 0) {
      x = child;
      while ((next ? nodes[x].left : nodes[x].right) >= 0)
        x = next ? nodes[x].left : nodes[x].right;
      return x;
    }
    int parent = nodes[x].parent;
    while (parent >= 0 && (next ? nodes[parent].right : nodes[parent].left) == x) {
      x      = parent;
      parent = nodes[x].parent;
    }
    return parent;
  }
  /// @brief Insert by actual exact incoming-endpoint comparisons, then test its newly adjacent pairs.
  /// @param id Actual starting nonvertical identity.
  void insert(std::size_t id) {
    int  current = root, parent = -1;
    bool right = false;
    while (current >= 0) {
      parent         = current;
      const int sign = compare(id, nodes[current].id, segments[id].p);
      if (!sign) {
        hit(id, nodes[current].id);
        return;
      }
      right   = sign > 0;
      current = right ? nodes[current].right : nodes[current].left;
    }
    const int index = static_cast<int>(nodes.size());
    nodes.push_back({-1, -1, parent, 1, id});
    where[id] = index;
    if (parent < 0)
      root = index;
    else if (right)
      nodes[parent].right = index;
    else
      nodes[parent].left = index;
    ++active;
    balance(parent);
    const int below = neighbor(index, false), above = neighbor(index, true);
    if (below >= 0 && hit(nodes[below].id, id))
      return;
    if (above >= 0)
      hit(id, nodes[above].id);
  }
  /// @brief Test newly adjacent neighbors and remove the exact identified AVL node.
  /// @param id Actual ending nonvertical identity.
  void remove(std::size_t id) {
    int index = where[id];
    if (index < 0)
      throw std::logic_error("Shamos-Hoey: missing active identity");
    const int below = neighbor(index, false), above = neighbor(index, true);
    if (below >= 0 && above >= 0 && hit(nodes[below].id, nodes[above].id))
      return;
    if (nodes[index].left >= 0 && nodes[index].right >= 0) {
      const int successor = neighbor(index, true);
      std::swap(nodes[index].id, nodes[successor].id);
      where[nodes[index].id]     = index;
      where[nodes[successor].id] = successor;
      index                      = successor;
    }
    const int child  = nodes[index].left >= 0 ? nodes[index].left : nodes[index].right,
              parent = nodes[index].parent;
    if (child >= 0)
      nodes[child].parent = parent;
    if (parent < 0)
      root = child;
    else if (nodes[parent].left == index)
      nodes[parent].left = child;
    else
      nodes[parent].right = child;
    where[id] = -1;
    --active;
    balance(parent);
  }
  /// @brief Query the first active height at/above a vertical segment's lower endpoint.
  /// @param id Actual vertical identity.
  void vertical(std::size_t id) {
    int current = root, candidate = -1;
    while (current >= 0) {
      const int sign = compare(id, nodes[current].id, segments[id].p);
      if (sign > 0)
        current = nodes[current].right;
      else {
        candidate = current;
        current   = nodes[current].left;
      }
    }
    if (candidate >= 0)
      hit(id, nodes[candidate].id);
  }
  /// @brief Validate complete geometry and process actual start/vertical/end events once.
  /// @param input Borrowed complete original segments.
  /// @return Owned actual sweep result and optional histories.
  ShamosRun run(const std::vector<std::pair<Point2D, Point2D>>& input) {
    const auto n = input.size();
    admit(0, n, options.max_segments);
    if (n > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        n > std::numeric_limits<std::size_t>::max() / 2)
      throw std::length_error("Shamos-Hoey: complete event/node geometry");
    segments.reserve(n);
    std::vector<Event> events;
    events.reserve(2 * n);
    where.assign(n, -1);
    nodes.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
      auto [p, q] = input[i];
      for (double value : {p.x, p.y, q.x, q.y})
        if (!std::isfinite(value))
          throw std::invalid_argument("Shamos-Hoey: finite coordinates required");
      if (q.x < p.x || (q.x == p.x && q.y < p.y))
        std::swap(p, q);
      const bool vertical = p.x == q.x;
      segments.push_back({p, q, vertical});
      if (vertical)
        events.push_back({p.x, ShamosEventKind::Vertical, i});
      else {
        events.push_back({p.x, ShamosEventKind::Start, i});
        events.push_back({q.x, ShamosEventKind::End, i});
      }
    }
    /// @brief Sort actual endpoint events by x, closed-contact phase, vertical lower y and original identity.
    /// @param a First event.
    /// @param b Second event.
    /// @return Actual deterministic event ordering.
    const auto earlier = [&](const Event& a, const Event& b) {
      if (a.x != b.x)
        return a.x < b.x;
      if (a.kind != b.kind)
        return a.kind < b.kind;
      if (a.kind == ShamosEventKind::Vertical && segments[a.id].p.y != segments[b.id].p.y)
        return segments[a.id].p.y < segments[b.id].p.y;
      return a.id < b.id;
    };
    std::sort(events.begin(), events.end(), earlier);
    std::size_t previous_vertical = std::numeric_limits<std::size_t>::max();
    double      previous_x        = 0;
    bool        first             = true;
    for (const auto& event : events) {
      if (first || event.x != previous_x) {
        previous_vertical = std::numeric_limits<std::size_t>::max();
        previous_x        = event.x;
        first             = false;
      }
      record();
      ++out.events_processed;
      if (event.kind == ShamosEventKind::Start)
        insert(event.id);
      else if (event.kind == ShamosEventKind::End)
        remove(event.id);
      else {
        if (previous_vertical != std::numeric_limits<std::size_t>::max())
          hit(previous_vertical, event.id);
        if (!out.found)
          vertical(event.id);
        previous_vertical = event.id;
      }
      if (capture)
        out.events.push_back({event.x, event.kind, event.id, active, height(root)});
      if (out.found)
        return std::move(out);
    }
    return std::move(out);
  }
};
} // namespace
ShamosRun shamos_hoey_run(
    const std::vector<std::pair<Point2D, Point2D>>& input,
    const ShamosOptions&                            options) {
  Sweep sweep{options, false};
  return sweep.run(input);
}
ShamosRun shamos_hoey_trace(
    const std::vector<std::pair<Point2D, Point2D>>& input,
    const ShamosOptions&                            options) {
  Sweep sweep{options, true};
  return sweep.run(input);
}
bool any_segments_intersect(const std::vector<std::pair<Point2D, Point2D>>& input) {
  return shamos_hoey_run(input).found;
}
std::vector<std::pair<Point2D, Point2D>>
shamos_horizontal_segments(std::size_t count, double gap, double length, std::size_t cap) {
  if (!std::isfinite(gap) || gap <= 0 || !std::isfinite(length) || length <= 0)
    throw std::invalid_argument("horizontal segments: positive finite dimensions");
  if ((cap && count > cap) || count > std::vector<std::pair<Point2D, Point2D>>().max_size())
    throw std::length_error("horizontal segments: complete geometry");
  std::vector<std::pair<Point2D, Point2D>> out;
  out.reserve(count);
  double previous = -1;
  for (std::size_t i = 0; i < count; ++i) {
    const long double wide = static_cast<long double>(i) * gap;
    if (!std::isfinite(wide) || wide > std::numeric_limits<double>::max())
      throw std::overflow_error("horizontal segments: y range");
    const double y = static_cast<double>(wide);
    if (i && !(y > previous))
      throw std::overflow_error("horizontal segments: distinct y range");
    out.push_back({{0, y}, {length, y}});
    previous = y;
  }
  return out;
}
} // namespace datamunge::geometry
