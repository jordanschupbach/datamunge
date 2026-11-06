/// @file types.cpp
/// @brief Strict physical component validation before world mutation.
#include <datamunge/sim/types.hpp>
#include <stdexcept>

#include "rigid_math.hpp"
namespace datamunge::sim {
/// @brief Translation-unit-local validation and construction helpers.
namespace {
/// @brief Require a finite scalar within a closed interval.
void range(double x, double lo, double hi) {
  if (!std::isfinite(x) || x < lo || x > hi)
    throw std::invalid_argument("simulation component out of range");
}
} // namespace
void Body::validate() const {
  if (motion != Motion::Static && motion != Motion::Dynamic && motion != Motion::Kinematic)
    throw std::invalid_argument("unknown body motion");
  if (shape != Shape::Box && shape != Shape::Sphere && shape != Shape::Capsule &&
      shape != Shape::Ellipsoid && shape != Shape::ConvexHull)
    throw std::invalid_argument("unknown collision shape");
  for (int i = 0; i < 3; ++i) {
    range(position[i], -100000, 100000);
    range(dimensions[i], .02, 1000);
    range(linear_velocity[i], -500, 500);
    range(surface_velocity[i], -500, 500);
    range(angular_velocity[i], -angular_speed_limit, angular_speed_limit);
    range(center_of_mass[i], -1000, 1000);
  }
  range(angular_speed_limit, 1, 100000);
  range(spinning_friction, 0, 1000);
  range(rolling_friction, 0, 1000);
  if (inertia_tensor != detail::Tensor3{}) {
    for (auto v : inertia_tensor)
      if (!std::isfinite(v))
        throw std::invalid_argument("nonfinite full inertia");
    auto&  I     = inertia_tensor;
    double scale = std::max({std::abs(I[0]), std::abs(I[4]), std::abs(I[8]), 1e-30});
    if (std::abs(I[1] - I[3]) + std::abs(I[2] - I[6]) + std::abs(I[5] - I[7]) > 1e-12 * scale ||
        I[0] <= 0 || I[0] * I[4] - I[1] * I[3] <= 0)
      throw std::invalid_argument("COM inertia must be symmetric positive definite");
    double det = I[0] * (I[4] * I[8] - I[5] * I[7]) - I[1] * (I[3] * I[8] - I[5] * I[6]) +
                 I[2] * (I[3] * I[7] - I[4] * I[6]);
    if (det <= 0)
      throw std::invalid_argument("COM inertia must be positive definite");
    auto inv = detail::invert_tensor(I);
    if (inv[8] <= 0)
      throw std::invalid_argument("COM inertia must be positive definite");
    // Principal moments must obey the rigid mass-distribution triangle inequalities.
    double          tr = I[0] + I[4] + I[8];
    detail::Tensor3 B  = I;
    for (auto& v : B)
      v = -v;
    B[0] += tr / 2;
    B[4] += tr / 2;
    B[8] += tr / 2;
    double detB = B[0] * (B[4] * B[8] - B[5] * B[7]) - B[1] * (B[3] * B[8] - B[5] * B[6]) +
                  B[2] * (B[3] * B[7] - B[4] * B[6]);
    if (B[0] < -1e-12 * scale || B[4] < -1e-12 * scale || B[8] < -1e-12 * scale ||
        B[0] * B[4] - B[1] * B[3] < -1e-12 * scale * scale ||
        B[0] * B[8] - B[2] * B[6] < -1e-12 * scale * scale ||
        B[4] * B[8] - B[5] * B[7] < -1e-12 * scale * scale || detB < -1e-12 * scale * scale * scale)
      throw std::invalid_argument("inertia violates physical triangle inequalities");
  }
  auto validate_geometry =
      [](Shape sh, const Vec3& dim, const std::shared_ptr<const std::vector<Vec3>>& points) {
        if (sh != Shape::Box && sh != Shape::Sphere && sh != Shape::Capsule &&
            sh != Shape::Ellipsoid && sh != Shape::ConvexHull)
          throw std::invalid_argument("invalid compound shape");
        for (auto v : dim)
          range(v, 1e-6, 1000);
        if (sh == Shape::Sphere && (dim[0] != dim[1] || dim[0] != dim[2]))
          throw std::invalid_argument("invalid sphere");
        if (sh == Shape::Capsule && (dim[0] != dim[2] || dim[1] <= dim[0]))
          throw std::invalid_argument("invalid capsule");
        if (sh == Shape::ConvexHull) {
          if (!points || points->size() < 4 || points->size() > 100000)
            throw std::invalid_argument("invalid convex vertex cloud");
          for (auto p : *points)
            for (auto v : p)
              range(v, -1000, 1000);
          auto   a = (*points)[0];
          Vec3   edge{}, normal{};
          double best = 0;
          for (auto p : *points)
            if (detail::length(detail::sub(p, a)) > best) {
              edge = detail::sub(p, a);
              best = detail::length(edge);
            }
          best = 0;
          for (auto p : *points) {
            auto c = detail::cross(edge, detail::sub(p, a));
            if (detail::length(c) > best) {
              normal = c;
              best   = detail::length(c);
            }
          }
          best = 0;
          for (auto p : *points)
            best = std::max(best, std::abs(detail::dot(normal, detail::sub(p, a))));
          if (best < 1e-16)
            throw std::invalid_argument("degenerate convex hull");
        }
      };
  validate_geometry(shape, dimensions, vertices);
  if (colliders) {
    if (colliders->empty() || colliders->size() > 256)
      throw std::invalid_argument("invalid compound size");
    for (auto& c : *colliders) {
      validate_geometry(c.shape, c.dimensions, c.vertices);
      for (auto v : c.position)
        range(v, -1000, 1000);
      double n = 0;
      for (auto v : c.rotation) {
        range(v, -1, 1);
        n += v * v;
      }
      if (std::abs(n - 1) > 1e-6)
        throw std::invalid_argument("invalid collider rotation");
    }
  }
  double qnorm = 0;
  for (double q : rotation) {
    range(q, -1, 1);
    qnorm += q * q;
  }
  if (std::abs(qnorm - 1) > 1e-6)
    throw std::invalid_argument("body rotation must be unit length");
  range(mass, .001, 1000000);
  range(friction, 0, 1);
  range(restitution, 0, 1);
  range(gravity_scale, 0, 4);
  range(linear_damping, 0, 1000000);
  range(angular_damping, 0, 1000000);
  if (shape == Shape::Sphere && (dimensions[0] != dimensions[1] || dimensions[0] != dimensions[2]))
    throw std::invalid_argument("sphere diameter must be uniform");
  if (shape == Shape::Capsule && (dimensions[0] != dimensions[2] || dimensions[1] <= dimensions[0]))
    throw std::invalid_argument("capsule requires Y > X = Z");
  if (motion == Motion::Static &&
      (detail::length(linear_velocity) > 0 || detail::length(angular_velocity) > 0))
    throw std::invalid_argument("static body velocities must be zero");
  if (motion == Motion::Dynamic && !allowed_translation[0] && !allowed_translation[1] &&
      !allowed_translation[2] && !allowed_rotation[0] && !allowed_rotation[1] &&
      !allowed_rotation[2])
    throw std::invalid_argument("dynamic body needs a degree of freedom");
}
void Joint::validate() const {
  if (type != JointType::Hinge && type != JointType::Slider && type != JointType::Fixed)
    throw std::invalid_argument("unknown joint type");
  if (!child || child == parent)
    throw std::invalid_argument("joint child/parent identifiers");
  for (double x : anchor)
    range(x, -1000, 1000);
  for (double x : axis)
    range(x, -1e6, 1e6);
  if (detail::length(axis) < 1e-12)
    throw std::invalid_argument("joint axis must be nonzero");
  double bound = type == JointType::Hinge ? std::acos(-1.) : 1000;
  range(limit_min, -bound, 0);
  range(limit_max, 0, bound);
  range(damping, 0, 1000000);
  range(armature, 0, 1000000);
  range(friction, 0, 1000000);
  range(effort_limit, 0, 100000);
  range(velocity_limit, 0, 100000);
}
} // namespace datamunge::sim
