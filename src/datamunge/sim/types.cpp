/// @file types.cpp
/// @brief Strict physical component validation before world mutation.
#include <datamunge/sim/types.hpp>
#include <stdexcept>

#include "math.hpp"
namespace datamunge::sim {
/// @brief Translation-unit-local validation and construction helpers.
namespace {
/// @brief Require a finite scalar within a closed interval.
void range(double x, double lo, double hi) {
  if (!std::isfinite(x) || x < lo || x > hi)
    throw std::invalid_argument("simulation component out of range");
}
}  // namespace
void Body::validate() const {
  if (motion != Motion::Static && motion != Motion::Dynamic &&
      motion != Motion::Kinematic)
    throw std::invalid_argument("unknown body motion");
  if (shape != Shape::Box && shape != Shape::Sphere && shape != Shape::Capsule)
    throw std::invalid_argument("unknown collision shape");
  for (int i = 0; i < 3; ++i) {
    range(position[i], -100000, 100000);
    range(dimensions[i], .02, 1000);
    range(linear_velocity[i], -500, 500);
    range(angular_velocity[i], -100, 100);
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
  if (shape == Shape::Sphere &&
      (dimensions[0] != dimensions[1] || dimensions[0] != dimensions[2]))
    throw std::invalid_argument("sphere diameter must be uniform");
  if (shape == Shape::Capsule &&
      (dimensions[0] != dimensions[2] || dimensions[1] <= dimensions[0]))
    throw std::invalid_argument("capsule requires Y > X = Z");
  if (motion == Motion::Static && (detail::length(linear_velocity) > 0 ||
                                   detail::length(angular_velocity) > 0))
    throw std::invalid_argument("static body velocities must be zero");
  if (motion == Motion::Dynamic && !allowed_translation[0] &&
      !allowed_translation[1] && !allowed_translation[2] &&
      !allowed_rotation[0] && !allowed_rotation[1] && !allowed_rotation[2])
    throw std::invalid_argument("dynamic body needs a degree of freedom");
}
void Joint::validate() const {
  if (type != JointType::Hinge && type != JointType::Slider)
    throw std::invalid_argument("unknown joint type");
  if (!child || child == parent)
    throw std::invalid_argument("joint child/parent identifiers");
  for (double x : anchor) range(x, -1000, 1000);
  for (double x : axis) range(x, -1e6, 1e6);
  if (detail::length(axis) < 1e-12)
    throw std::invalid_argument("joint axis must be nonzero");
  double bound = type == JointType::Hinge ? std::acos(-1.) : 1000;
  range(limit_min, -bound, 0);
  range(limit_max, 0, bound);
  range(damping, 0, 1000000);
}
}  // namespace datamunge::sim
