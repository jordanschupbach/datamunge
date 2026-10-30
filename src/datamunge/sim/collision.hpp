#pragma once
/// @file collision.hpp
/// @brief Private convex support, bounded feature distances and analytic primitive contacts.
#include <array>
#include <limits>
#include <stdexcept>

#include "math.hpp"
namespace datamunge::sim::detail {
/// @brief Axis-aligned bounds for broadphase and swept candidate generation.
struct Bounds {
  Vec3 lo; ///< Lower corner.
  Vec3 hi; ///< Upper corner.
};
/// @brief A contact generated in a pair-local manifold.
struct PointContact {
  Vec3   point;      ///< World-space midpoint.
  Vec3   normal;     ///< First-to-second normal.
  double separation; ///< Signed surface distance.
};
/// @brief Fixed-size contact manifold, avoiding pair-local allocations.
struct Manifold {
  std::array<PointContact, 8> points;   ///< Contact storage.
  std::size_t                 count{0}; ///< Valid entries.
};
/// @brief Return three world-space unit local axes.
inline std::array<Vec3, 3> axes(const Body& b) {
  return {rotate(b.rotation, basis(0)), rotate(b.rotation, basis(1)), rotate(b.rotation, basis(2))};
}
/// @brief Exact support point of a native convex primitive.
inline Vec3 support(const Body& b, const Vec3& direction) {
  auto local = rotate(conjugate(b.rotation), direction);
  Vec3 point{};
  if (b.shape == Shape::Box) {
    for (int i = 0; i < 3; ++i)
      point[i] = std::copysign(.5 * b.dimensions[i], local[i]);
  } else {
    point = mul(unit(local), .5 * b.dimensions[0]);
    if (b.shape == Shape::Capsule)
      point[1] += std::copysign(.5 * (b.dimensions[1] - b.dimensions[0]), local[1]);
  }
  return add(b.position, rotate(b.rotation, point));
}
/// @brief Exact world-space AABB from the primitive support function.
inline Bounds bounds(const Body& b) {
  Bounds out;
  for (int i = 0; i < 3; ++i) {
    auto d    = basis(i);
    out.hi[i] = support(b, d)[i];
    out.lo[i] = support(b, mul(d, -1))[i];
  }
  return out;
}
/// @brief Conservative swept AABB including angular displacement about the centre.
inline Bounds swept_bounds(const Body& b, double dt) {
  auto   box        = bounds(b);
  double rotational = length(b.angular_velocity) * dt * .5 * length(b.dimensions);
  for (int i = 0; i < 3; ++i) {
    double change = b.linear_velocity[i] * dt;
    box.lo[i] += std::min(0., change) - rotational;
    box.hi[i] += std::max(0., change) + rotational;
  }
  return box;
}
/// @brief Whether two AABBs overlap within tolerance.
inline bool overlap(const Bounds& a, const Bounds& b, double tolerance = 0) {
  for (int i = 0; i < 3; ++i)
    if (a.hi[i] + tolerance < b.lo[i] || b.hi[i] + tolerance < a.lo[i])
      return false;
  return true;
}
/// @brief Closest points on two convex primitives and their nonnegative distance.
struct Distance {
  Vec3   first{};  ///< Closest first-shape witness.
  Vec3   second{}; ///< Closest second-shape witness.
  double value{0}; ///< Euclidean distance; zero for intersecting shapes.
};
/// @brief Segment endpoints defining a sphere/capsule's swept centreline.
inline std::array<Vec3, 2> segment(const Body& b) {
  Vec3 d = rotate(
      b.rotation, {0, b.shape == Shape::Capsule ? .5 * (b.dimensions[1] - b.dimensions[0]) : 0, 0});
  return {sub(b.position, d), add(b.position, d)};
}
/// @brief Closest points of two closed segments, including zero-length sphere segments.
inline std::array<Vec3, 2>
segments_closest(const Vec3& p1, const Vec3& q1, const Vec3& p2, const Vec3& q2) {
  auto   d1 = sub(q1, p1), d2 = sub(q2, p2), r = sub(p1, p2);
  double a = dot(d1, d1), e = dot(d2, d2), f = dot(d2, r), s = 0, t = 0;
  if (a <= 1e-20 && e <= 1e-20)
    return {p1, p2};
  if (a <= 1e-20)
    t = std::clamp(f / e, 0., 1.);
  else {
    double c = dot(d1, r);
    if (e <= 1e-20)
      s = std::clamp(-c / a, 0., 1.);
    else {
      double b = dot(d1, d2), den = a * e - b * b;
      if (den > 1e-20)
        s = std::clamp((b * f - c * e) / den, 0., 1.);
      t = (b * s + f) / e;
      if (t < 0) {
        t = 0;
        s = std::clamp(-c / a, 0., 1.);
      } else if (t > 1) {
        t = 1;
        s = std::clamp((b - c) / a, 0., 1.);
      }
    }
  }
  return {add(p1, mul(d1, s)), add(p2, mul(d2, t))};
}
/// @brief Closest segment/AABB points by exact piecewise-quadratic minimization.
inline std::array<Vec3, 2> segment_box(const Vec3& a, const Vec3& b, const Vec3& half) {
  auto                  d = sub(b, a);
  std::array<double, 8> cuts{};
  std::size_t           count = 2;
  cuts[0]                     = 0;
  cuts[1]                     = 1;
  for (int i = 0; i < 3; ++i)
    if (std::abs(d[i]) > 1e-15)
      for (double side : {-half[i], half[i]}) {
        double t = (side - a[i]) / d[i];
        if (t > 0 && t < 1)
          cuts[count++] = t;
      }
  std::sort(cuts.begin(), cuts.begin() + count);
  double              best = std::numeric_limits<double>::infinity();
  std::array<Vec3, 2> out{};
  auto                consider = [&](double t) {
    auto p = add(a, mul(d, t)), q = p;
    for (int i = 0; i < 3; ++i)
      q[i] = std::clamp(q[i], -half[i], half[i]);
    double v = dot(sub(p, q), sub(p, q));
    if (v < best) {
      best = v;
      out  = {p, q};
    }
  };
  for (std::size_t k = 0; k < count; ++k)
    consider(cuts[k]);
  for (std::size_t k = 1; k < count; ++k) {
    double middle = .5 * (cuts[k - 1] + cuts[k]), den = 0, num = 0;
    for (int i = 0; i < 3; ++i) {
      double v = a[i] + middle * d[i];
      if (v < -half[i] || v > half[i]) {
        double target = v < 0 ? -half[i] : half[i];
        den += d[i] * d[i];
        num += d[i] * (a[i] - target);
      }
    }
    if (den > 0)
      consider(std::clamp(-num / den, cuts[k - 1], cuts[k]));
  }
  return out;
}
/// @brief Exact box feature distance, defined after the SAT manifold generator.
inline Distance boxes_distance(const Body& a, const Body& b);
/// @brief Exact primitive distance using segment and box feature witnesses.
inline Distance distance(const Body& a, const Body& b) {
  if (a.shape == Shape::Box && b.shape == Shape::Box)
    return boxes_distance(a, b);
  if (a.shape == Shape::Box) {
    auto result = distance(b, a);
    std::swap(result.first, result.second);
    return result;
  }
  const auto          ae = segment(a);
  std::array<Vec3, 2> nearest;
  double              radius = .5 * a.dimensions[0];
  if (b.shape == Shape::Box) {
    const auto inverse = conjugate(b.rotation);
    nearest            = segment_box(
        rotate(inverse, sub(ae[0], b.position)),
        rotate(inverse, sub(ae[1], b.position)),
        mul(b.dimensions, .5));
    for (auto& p : nearest)
      p = add(b.position, rotate(b.rotation, p));
  } else {
    const auto be = segment(b);
    nearest       = segments_closest(ae[0], ae[1], be[0], be[1]);
    radius += .5 * b.dimensions[0];
  }
  const auto   direction = sub(nearest[1], nearest[0]);
  const double length_   = length(direction);
  if (length_ <= radius) {
    const auto point = b.shape == Shape::Box ? nearest[1] : mul(add(nearest[0], nearest[1]), .5);
    return {point, point, 0};
  }
  const auto normal = mul(direction, 1 / length_);
  return {
      add(nearest[0], mul(normal, .5 * a.dimensions[0])),
      b.shape == Shape::Box ? nearest[1] : sub(nearest[1], mul(normal, .5 * b.dimensions[0])),
      length_ - radius};
}
/// @brief Emit a sphere/capsule against an oriented box; normal points rounded shape to box.
inline Manifold rounded_box(const Body& rounded, const Body& box, double slop) {
  auto ends    = segment(rounded);
  auto inv     = conjugate(box.rotation);
  auto half    = mul(box.dimensions, .5);
  auto nearest = segment_box(
      rotate(inv, sub(ends[0], box.position)), rotate(inv, sub(ends[1], box.position)), half);
  auto     offset = sub(nearest[1], nearest[0]);
  double   d = length(offset), r = .5 * rounded.dimensions[0];
  Manifold m;
  if (d > r + slop)
    return m;
  Vec3   normal;
  double separation = d - r;
  if (d > 1e-12)
    normal = mul(offset, 1 / d);
  else {
    auto   centre = rotate(inv, sub(rounded.position, box.position));
    int    axis   = 0;
    double depth  = half[0] - std::abs(nearest[0][0]);
    for (int i = 1; i < 3; ++i) {
      double candidate = half[i] - std::abs(nearest[0][i]);
      if (candidate < depth) {
        depth = candidate;
        axis  = i;
      }
    }
    normal           = {0, 0, 0};
    normal[axis]     = centre[axis] >= 0 ? -1 : 1;
    nearest[1]       = nearest[0];
    nearest[1][axis] = -normal[axis] * half[axis];
    separation       = -r - depth;
  }
  normal              = rotate(box.rotation, normal);
  auto p              = add(box.position, rotate(box.rotation, nearest[0])),
       q              = add(box.position, rotate(box.rotation, nearest[1]));
  m.points[m.count++] = {mul(add(add(p, mul(normal, r)), q), .5), normal, separation};
  return m;
}
/// @brief Clip an incident contact polygon against one reference-face side plane.
inline std::size_t clip_polygon(
    const std::array<Vec3, 16>& in,
    std::size_t                 count,
    std::array<Vec3, 16>&       out,
    Vec3                        normal,
    double                      boundary) {
  std::size_t used = 0;
  if (!count)
    return 0;
  for (std::size_t i = 0; i < count; ++i) {
    auto   a = in[i], b = in[(i + 1) % count];
    double da = dot(normal, a) - boundary, db = dot(normal, b) - boundary;
    if (da <= 0)
      out[used++] = a;
    if ((da < 0 && db > 0) || (da > 0 && db < 0))
      out[used++] = add(a, mul(sub(b, a), da / (da - db)));
  }
  return used;
}
/// @brief Exact oriented-box SAT with clipped face contacts and edge witnesses.
inline Manifold boxes(const Body& a, const Body& b, double slop) {
  auto ax = axes(a), bx = axes(b);
  auto ah = mul(a.dimensions, .5), bh = mul(b.dimensions, .5), delta = sub(b.position, a.position);
  double minimum = std::numeric_limits<double>::infinity();
  Vec3   normal{};
  int    feature = -1;
  for (int k = 0; k < 15; ++k) {
    Vec3   d = k < 3 ? ax[k] : (k < 6 ? bx[k - 3] : cross(ax[(k - 6) / 3], bx[(k - 6) % 3]));
    double n = length(d);
    if (n < 1e-10)
      continue;
    d         = mul(d, 1 / n);
    double ra = 0, rb = 0;
    for (int j = 0; j < 3; ++j) {
      ra += ah[j] * std::abs(dot(d, ax[j]));
      rb += bh[j] * std::abs(dot(d, bx[j]));
    }
    double penetration = ra + rb - std::abs(dot(delta, d));
    if (penetration < -slop)
      return {};
    if (penetration < minimum) {
      minimum = penetration;
      normal  = dot(delta, d) >= 0 ? d : mul(d, -1);
      feature = k;
    }
  }
  Manifold m;
  if (feature < 6) {
    bool        reference_a = feature < 3;
    const auto& ref         = reference_a ? a : b;
    const auto& incident    = reference_a ? b : a;
    auto        ra = reference_a ? ax : bx, ia = reference_a ? bx : ax;
    auto        rh = reference_a ? ah : bh, ih = reference_a ? bh : ah;
    int         axis          = reference_a ? feature : feature - 3;
    auto        outward       = reference_a ? normal : mul(normal, -1);
    auto        face          = add(ref.position, mul(outward, rh[axis]));
    int         incident_axis = 0;
    for (int j = 1; j < 3; ++j)
      if (std::abs(dot(ia[j], outward)) > std::abs(dot(ia[incident_axis], outward)))
        incident_axis = j;
    auto ic =
        add(incident.position,
            mul(ia[incident_axis],
                dot(ia[incident_axis], outward) > 0 ? -ih[incident_axis] : ih[incident_axis]));
    int                  u = (incident_axis + 1) % 3, v = (incident_axis + 2) % 3;
    std::array<Vec3, 16> polygon{}, temp{};
    std::size_t          count = 4;
    for (int k = 0; k < 4; ++k) {
      double su = (k == 0 || k == 3) ? -1 : 1, sv = k < 2 ? -1 : 1;
      polygon[k] = add(ic, add(mul(ia[u], su * ih[u]), mul(ia[v], sv * ih[v])));
    }
    for (int j = 0; j < 3; ++j)
      if (j != axis)
        for (double sign : {-1., 1.}) {
          auto d  = mul(ra[j], sign);
          count   = clip_polygon(polygon, count, temp, d, dot(d, ref.position) + rh[j]);
          polygon = temp;
        }
    for (std::size_t k = 0; k < count && m.count < m.points.size(); ++k) {
      double separation = dot(outward, sub(polygon[k], face));
      if (separation <= slop)
        m.points[m.count++] = {sub(polygon[k], mul(outward, .5 * separation)), normal, separation};
    }
  } else {
    int  ai = (feature - 6) / 3, bi = (feature - 6) % 3;
    Vec3 ac = a.position, bc = b.position;
    for (int j = 0; j < 3; ++j) {
      if (j != ai)
        ac = add(ac, mul(ax[j], dot(ax[j], normal) >= 0 ? ah[j] : -ah[j]));
      if (j != bi)
        bc = add(bc, mul(bx[j], dot(bx[j], normal) >= 0 ? -bh[j] : bh[j]));
    }
    auto points = segments_closest(
        sub(ac, mul(ax[ai], ah[ai])),
        add(ac, mul(ax[ai], ah[ai])),
        sub(bc, mul(bx[bi], bh[bi])),
        add(bc, mul(bx[bi], bh[bi])));
    m.points[m.count++] = {mul(add(points[0], points[1]), .5), normal, -minimum};
  }
  if (!m.count)
    m.points[m.count++] = {
        mul(add(support(a, normal), support(b, mul(normal, -1))), .5), normal, -minimum};
  return m;
}
/// @brief Minimum box distance from vertex/face and edge/edge witnesses, bounded independently of
/// aspect ratio.
inline Distance boxes_distance(const Body& a, const Body& b) {
  const auto manifold = boxes(a, b, 0);
  if (manifold.count) {
    const auto p = manifold.points[0].point;
    return {p, p, 0};
  }
  std::array<Vec3, 8> av{}, bv{};
  const auto          ah = mul(a.dimensions, .5), bh = mul(b.dimensions, .5);
  for (int mask = 0; mask < 8; ++mask) {
    Vec3 ap{}, bp{};
    for (int axis = 0; axis < 3; ++axis) {
      ap[axis] = (mask & (1 << axis)) ? ah[axis] : -ah[axis];
      bp[axis] = (mask & (1 << axis)) ? bh[axis] : -bh[axis];
    }
    av[mask] = add(a.position, rotate(a.rotation, ap));
    bv[mask] = add(b.position, rotate(b.rotation, bp));
  }
  Distance result;
  result.value = std::numeric_limits<double>::infinity();
  /// @brief Retain the nearest feature pair without allocating temporary storage.
  const auto consider = [&](const Vec3& p, const Vec3& q) {
    const auto d = length(sub(p, q));
    if (d < result.value)
      result = {p, q, d};
  };
  for (int i = 0; i < 8; ++i) {
    auto q = rotate(conjugate(b.rotation), sub(av[i], b.position));
    auto p = rotate(conjugate(a.rotation), sub(bv[i], a.position));
    for (int axis = 0; axis < 3; ++axis) {
      q[axis] = std::clamp(q[axis], -bh[axis], bh[axis]);
      p[axis] = std::clamp(p[axis], -ah[axis], ah[axis]);
    }
    consider(av[i], add(b.position, rotate(b.rotation, q)));
    consider(add(a.position, rotate(a.rotation, p)), bv[i]);
  }
  for (int i = 0; i < 8; ++i)
    for (int axis = 0; axis < 3; ++axis)
      if (!(i & (1 << axis)))
        for (int j = 0; j < 8; ++j)
          for (int other = 0; other < 3; ++other)
            if (!(j & (1 << other))) {
              const auto pair =
                  segments_closest(av[i], av[i | (1 << axis)], bv[j], bv[j | (1 << other)]);
              consider(pair[0], pair[1]);
            }
  return result;
}
/// @brief Native exact primitive manifold dispatch.
inline Manifold collide(const Body& a, const Body& b, double slop) {
  if (a.shape == Shape::Box && b.shape == Shape::Box)
    return boxes(a, b, slop);
  if (b.shape == Shape::Box)
    return rounded_box(a, b, slop);
  if (a.shape == Shape::Box) {
    auto m = rounded_box(b, a, slop);
    for (std::size_t i = 0; i < m.count; ++i)
      m.points[i].normal = mul(m.points[i].normal, -1);
    return m;
  }
  auto   ae = segment(a), be = segment(b);
  auto   points = segments_closest(ae[0], ae[1], be[0], be[1]);
  auto   delta  = sub(points[1], points[0]);
  double d = length(delta), ar = .5 * a.dimensions[0], br = .5 * b.dimensions[0];
  if (d > ar + br + slop)
    return {};
  auto     normal = unit(delta, unit(sub(b.position, a.position)));
  Manifold m;
  m.points[m.count++] = {
      mul(add(add(points[0], mul(normal, ar)), sub(points[1], mul(normal, br))), .5),
      normal,
      d - ar - br};
  return m;
}
/// @brief First exact ray intersection with a convex primitive; ray direction is unit length.
inline bool
ray_hit(const Body& b, Vec3 origin, Vec3 direction, double maximum, double& time, Vec3& normal) {
  auto inverse = conjugate(b.rotation);
  origin       = rotate(inverse, sub(origin, b.position));
  direction    = rotate(inverse, direction);
  double best  = maximum + 1;
  Vec3   best_normal{};
  auto   sphere = [&](Vec3 centre) {
    auto   o      = sub(origin, centre);
    double radius = .5 * b.dimensions[0], c = dot(o, o) - radius * radius, q = dot(o, direction),
           disc = q * q - c;
    if (disc < 0)
      return;
    double t = -q - std::sqrt(disc);
    if (t >= 0 && t <= maximum && t < best) {
      best        = t;
      best_normal = unit(sub(add(origin, mul(direction, t)), centre));
    }
  };
  if (b.shape == Shape::Box) {
    double enter = 0, exit = maximum;
    Vec3   entering{};
    for (int i = 0; i < 3; ++i) {
      double half = .5 * b.dimensions[i];
      if (std::abs(direction[i]) < 1e-15) {
        if (origin[i] < -half || origin[i] > half)
          return false;
        continue;
      }
      double a = (-half - origin[i]) / direction[i], z = (half - origin[i]) / direction[i];
      Vec3   n = basis(i);
      n[i]     = -1;
      if (a > z) {
        std::swap(a, z);
        n[i] = 1;
      }
      if (a > enter) {
        enter    = a;
        entering = n;
      }
      exit = std::min(exit, z);
      if (enter > exit)
        return false;
    }
    if (enter >= 0 && enter <= maximum) {
      best        = enter;
      best_normal = entering;
    }
  } else if (b.shape == Shape::Sphere)
    sphere({0, 0, 0});
  else {
    double half = .5 * (b.dimensions[1] - b.dimensions[0]), r = .5 * b.dimensions[0];
    sphere({0, -half, 0});
    sphere({0, half, 0});
    double a = direction[0] * direction[0] + direction[2] * direction[2],
           q = origin[0] * direction[0] + origin[2] * direction[2],
           c = origin[0] * origin[0] + origin[2] * origin[2] - r * r, disc = q * q - a * c;
    if (a > 1e-20 && disc >= 0) {
      double t = (-q - std::sqrt(disc)) / a, y = origin[1] + t * direction[1];
      if (t >= 0 && t <= maximum && t < best && y >= -half && y <= half) {
        best        = t;
        best_normal = unit(Vec3{origin[0] + t * direction[0], 0, origin[2] + t * direction[2]});
      }
    }
  }
  if (best > maximum)
    return false;
  time   = best;
  normal = rotate(b.rotation, best_normal);
  return true;
}
} // namespace datamunge::sim::detail
