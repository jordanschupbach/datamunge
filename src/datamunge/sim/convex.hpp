#pragma once
/// @file convex.hpp
/// @brief Native three-dimensional support-map GJK distance and EPA penetration.
/// @par References
/// - E. G. Gilbert, D. W. Johnson and S. S. Keerthi (1988),
///   A Fast Procedure for Computing the Distance Between Complex Objects in Three-Dimensional
///   Space, doi:10.1109/56.2083. Uses enumerated simplex faces for closest-point minimization.
/// - G. van den Bergen (2001), Proximity Queries and Penetration Depth Computation on 3D Game
/// Objects,
///   https://graphics.stanford.edu/courses/cs468-01-fall/Papers/van-den-bergen.pdf .
#include <vector>

#include "rigid_math.hpp"
namespace datamunge::sim::detail {
/// @brief One Minkowski point and its original-shape witnesses.
struct SupportVertex {
  Vec3 v{}; ///< Minkowski difference.
  Vec3 a{}; ///< First-shape world witness.
  Vec3 b{}; ///< Second-shape world witness.
};
/// @brief Witnessed nearest simplex point and active barycentric coordinates.
struct SimplexPoint {
  Vec3                       v{};    ///< Closest difference.
  Vec3                       a{};    ///< First-shape interpolated witness.
  Vec3                       b{};    ///< Second-shape interpolated witness.
  std::vector<SupportVertex> active; ///< Positive-weight face vertices, at most four.
};
/// @brief Small pivoted solve used for closest simplex barycentrics.
/// @param a Augmented row-major matrix, modified in place.
/// @param n Dimension in 1..3.
/// @param out Solution on success.
/// @return False for a numerically singular system.
inline bool small_solve(double a[3][4], int n, double out[3]) {
  for (int col = 0; col < n; ++col) {
    int pivot = col;
    for (int r = col + 1; r < n; ++r)
      if (std::abs(a[r][col]) > std::abs(a[pivot][col]))
        pivot = r;
    if (std::abs(a[pivot][col]) < 1e-24)
      return false;
    for (int k = col; k <= n; ++k)
      std::swap(a[col][k], a[pivot][k]);
    double d = a[col][col];
    for (int k = col; k <= n; ++k)
      a[col][k] /= d;
    for (int r = 0; r < n; ++r)
      if (r != col) {
        double q = a[r][col];
        for (int k = col; k <= n; ++k)
          a[r][k] -= q * a[col][k];
      }
  }
  for (int i = 0; i < n; ++i)
    out[i] = a[i][n];
  return true;
}
/// @brief Minimum-norm point on every simplex face, retaining original witnesses.
/// @param points One to four Minkowski support vertices.
/// @return Closest convex combination with original witnesses.
/// @throws std::runtime_error No valid finite simplex face.
inline SimplexPoint closest_simplex(const std::vector<SupportVertex>& points) {
  SimplexPoint result;
  double       best = INFINITY;
  for (unsigned mask = 1; mask < (1u << points.size()); ++mask) {
    int ids[4], n = 0;
    for (int i = 0; i < int(points.size()); ++i)
      if (mask & (1u << i))
        ids[n++] = i;
    double weights[4]{1, 0, 0, 0}, matrix[3][4]{}, answer[3]{};
    auto   origin = points[ids[0]].v;
    for (int i = 1; i < n; ++i) {
      auto edge = sub(points[ids[i]].v, origin);
      for (int j = 1; j < n; ++j)
        matrix[i - 1][j - 1] = dot(edge, sub(points[ids[j]].v, origin));
      matrix[i - 1][n - 1] = -dot(edge, origin);
    }
    if (n > 1 && !small_solve(matrix, n - 1, answer))
      continue;
    for (int i = 1; i < n; ++i) {
      weights[i] = answer[i - 1];
      weights[0] -= weights[i];
    }
    bool valid = true;
    for (int i = 0; i < n; ++i)
      valid &= weights[i] >= -1e-10;
    if (!valid)
      continue;
    Vec3 v{}; ///< Minkowski difference.
    Vec3 a{}; ///< First-shape world witness.
    Vec3 b{}; ///< Second-shape world witness.
    for (int i = 0; i < n; ++i) {
      v = add(v, mul(points[ids[i]].v, weights[i]));
      a = add(a, mul(points[ids[i]].a, weights[i]));
      b = add(b, mul(points[ids[i]].b, weights[i]));
    }
    double d = dot(v, v);
    if (d < best) {
      best   = d;
      result = {v, a, b, {}};
      for (int i = 0; i < n; ++i)
        if (weights[i] > 1e-12)
          result.active.push_back(points[ids[i]]);
    }
  }
  if (!std::isfinite(best))
    throw std::runtime_error("native GJK degenerate simplex");
  return result;
}
/// @brief GJK result with witnesses and an enclosing/near-contact simplex.
struct GjkResult {
  SimplexPoint closest;         ///< Closest witnessed simplex.
  Vec3         normal{1, 0, 0}; ///< First-to-second search direction.
  bool         overlap{false};  ///< Whether the difference contains the origin.
};
/// @brief Bounded convex distance solve; caller supplies an exact support function.
/// @tparam Support Callable returning a SupportVertex for a nonzero direction.
/// @param support Borrowed support-map callback.
/// @param direction Initial search direction; zero selects the X axis.
/// @return Witnessed closest point or overlapping simplex.
/// @throws std::runtime_error Iteration capacity or degenerate simplex.
template<class Support>
GjkResult gjk(Support support, Vec3 direction) {
  auto                       first = support(unit(direction));
  std::vector<SupportVertex> simplex{first};
  auto                       point = closest_simplex(simplex);
  Vec3                       last  = unit(mul(point.v, -1));
  for (int k = 0; k < 96; ++k) {
    double d = length(point.v);
    if (d < 1e-10)
      return {point, last, true};
    direction   = mul(point.v, -1);
    last        = unit(direction);
    auto vertex = support(last);
    if (dot(vertex.v, direction) - dot(point.v, direction) < 1e-11 * std::max(1., d))
      return {point, last, false};
    bool duplicate = false;
    for (auto p : simplex)
      duplicate |= length(sub(p.v, vertex.v)) < 1e-12;
    if (duplicate)
      return {point, last, d < 1e-8};
    simplex = point.active;
    if (simplex.size() == 4)
      return {point, last, true};
    simplex.push_back(vertex);
    point = closest_simplex(simplex);
  }
  throw std::runtime_error("native GJK iteration capacity exhausted");
}
/// @brief Oriented triangular boundary face of an expanding polytope.
struct EpaFace {
  int    a; ///< First face vertex.
  int    b; ///< Second face vertex.
  int    c; ///< Third face vertex.
  Vec3   n; ///< Outward unit face normal.
  double d; ///< Nonnegative plane distance from the origin.
};
/// @brief Contact witnesses, first-to-second normal and signed separation.
struct ConvexContact {
  Vec3   a{};           ///< First-shape witness.
  Vec3   b{};           ///< Second-shape witness.
  Vec3   normal{};      ///< First-to-second separation normal.
  double separation{0}; ///< Signed unpadded separation, metres.
  bool   hit{false};    ///< Whether padded convex sets overlap.
};
/// @brief Penetration of a padded support difference, using native EPA.
/// @tparam Support Callable returning padded Minkowski support vertices.
/// @param support Borrowed support-map callback.
/// @param result Overlapping GJK simplex.
/// @param margin Padding to remove from the final penetration.
/// @return Witnesses and signed separation; boundary degeneracies use the GJK normal.
/// @throws std::runtime_error Invalid (empty) polytope; capacity limits return the closest face.
template<class Support>
ConvexContact epa(Support support, const GjkResult& result, double margin) {
  auto points    = result.closest.active;
  auto add_point = [&](Vec3 dir) {
    auto p = support(unit(dir));
    for (auto v : points)
      if (length(sub(v.v, p.v)) < 1e-10)
        return;
    points.push_back(p);
  };
  for (int i = 0; i < 3; ++i) {
    add_point(basis(i));
    add_point(mul(basis(i), -1));
  }
  add_point(result.normal);
  add_point(mul(result.normal, -1));
  for (auto d : std::array<Vec3, 4>{{{1, 1, 1}, {-1, -1, 1}, {-1, 1, -1}, {1, -1, -1}}})
    add_point(d);
  std::array<int, 4> tetra{};
  bool               found = false;
  for (int a = 0; a < int(points.size()) && !found; ++a)
    for (int b = a + 1; b < int(points.size()) && !found; ++b)
      for (int c = b + 1; c < int(points.size()) && !found; ++c)
        for (int d = c + 1; d < int(points.size()); ++d) {
          double matrix[3][4]{}, w[3]{};
          for (int row = 0; row < 3; ++row) {
            matrix[row][0] = points[b].v[row] - points[a].v[row];
            matrix[row][1] = points[c].v[row] - points[a].v[row];
            matrix[row][2] = points[d].v[row] - points[a].v[row];
            matrix[row][3] = -points[a].v[row];
          }
          if (small_solve(matrix, 3, w) &&
              std::min({w[0], w[1], w[2], 1 - w[0] - w[1] - w[2]}) > 1e-10) {
            tetra = {a, b, c, d};
            found = true;
            break;
          }
        }
  if (!found)
    return {result.closest.a, result.closest.b, result.normal, margin, true};
  std::vector<SupportVertex> vertices;
  for (auto i : tetra)
    vertices.push_back(points[i]);
  std::vector<EpaFace> faces;
  auto                 face = [&](int a, int b, int c) {
    auto   n = cross(sub(vertices[b].v, vertices[a].v), sub(vertices[c].v, vertices[a].v));
    double size = length(n);
    if (size < 1e-18)
      return;
    n        = mul(n, 1 / size);
    double d = dot(n, vertices[a].v);
    if (d < 0) {
      std::swap(b, c);
      n = mul(n, -1);
      d = -d;
    }
    faces.push_back({a, b, c, n, d});
  };
  face(0, 1, 2);
  face(0, 3, 1);
  face(0, 2, 3);
  face(1, 3, 2);
  for (int k = 0; k < 128; ++k) {
    if (faces.empty())
      throw std::runtime_error("native EPA empty polytope");
    auto it =
        std::min_element(faces.begin(), faces.end(), [](auto& a, auto& b) { return a.d < b.d; });
    auto f         = *it;
    auto p         = support(f.n);
    bool duplicate = false;
    for (auto v : vertices)
      duplicate |= length(sub(v.v, p.v)) < 1e-10;
    if (dot(p.v, f.n) - f.d < 2e-7 || duplicate) {
      auto q = closest_simplex({vertices[f.a], vertices[f.b], vertices[f.c]});
      return {q.a, q.b, f.n, margin - f.d, true};
    }
    // Smooth (rounded) supports can refine indefinitely below the absolute
    // tolerance. At capacity, the closest face is already a lower bound on the
    // penetration within the last support gap, so use it instead of aborting.
    if (vertices.size() > 256 || faces.size() > 512) {
      auto q = closest_simplex({vertices[f.a], vertices[f.b], vertices[f.c]});
      return {q.a, q.b, f.n, margin - f.d, true};
    }
    std::vector<std::pair<int, int>> boundary;
    auto                             edge = [&](int a, int b) {
      auto q = std::find(boundary.begin(), boundary.end(), std::pair{b, a});
      if (q == boundary.end())
        boundary.push_back({a, b});
      else
        boundary.erase(q);
    };
    for (auto q = faces.begin(); q != faces.end();)
      if (dot(q->n, sub(p.v, vertices[q->a].v)) > 1e-10) {
        edge(q->a, q->b);
        edge(q->b, q->c);
        edge(q->c, q->a);
        q = faces.erase(q);
      } else
        ++q;
    int index = vertices.size();
    vertices.push_back(p);
    for (auto [a, b] : boundary)
      face(a, b, index);
  }
  if (faces.empty())
    throw std::runtime_error("native EPA empty polytope");
  auto f = *std::min_element(faces.begin(), faces.end(), [](auto& a, auto& b) { return a.d < b.d; });
  auto q = closest_simplex({vertices[f.a], vertices[f.b], vertices[f.c]});
  return {q.a, q.b, f.n, margin - f.d, true};
}
} // namespace datamunge::sim::detail
