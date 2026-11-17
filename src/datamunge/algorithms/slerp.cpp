/// @file slerp.cpp
/// @brief Single scale-safe quaternion interpolation producer.
#include <algorithm>
#include <cmath>
#include <datamunge/algorithms/slerp.hpp>
#include <stdexcept>
/// @brief Canonical checked native producers and owned values.
namespace datamunge::algorithms {
/// @brief Private invocation-local checked producing helpers.
namespace {
/// @brief Validate finite raw components.
/// @param q Borrowed raw quaternion.
void finite(const Quaternion& q) {
  for (double x : q)
    if (!std::isfinite(x))
      throw std::invalid_argument("quaternion: finite components");
}
} // namespace
double quat_dot(const Quaternion& a, const Quaternion& b) {
  finite(a);
  finite(b);
  const double x = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
  if (!std::isfinite(x))
    throw std::overflow_error("quaternion: dot range");
  return x;
}
double quat_norm(const Quaternion& q) {
  finite(q);
  const double x = std::hypot(std::hypot(q[0], q[1]), std::hypot(q[2], q[3]));
  if (!std::isfinite(x))
    throw std::overflow_error("quaternion: norm range");
  return x;
}
Quaternion quat_normalize(const Quaternion& q) {
  finite(q);
  double scale = 0;
  for (double x : q)
    scale = std::max(scale, std::abs(x));
  if (scale == 0)
    throw std::invalid_argument("quaternion: zero rotation");
  Quaternion value;
  for (std::size_t i = 0; i < 4; ++i)
    value[i] = q[i] / scale;
  const double norm = quat_norm(value);
  for (double& x : value)
    x /= norm;
  return value;
}
double quat_angle_between(const Quaternion& a, const Quaternion& b) {
  auto x = quat_normalize(a), y = quat_normalize(b);
  if (quat_dot(x, y) < 0)
    for (double& v : y)
      v = -v;
  Quaternion diff, sum;
  for (std::size_t i = 0; i < 4; ++i) {
    diff[i] = x[i] - y[i];
    sum[i]  = x[i] + y[i];
  }
  return 4 * std::atan2(quat_norm(diff), quat_norm(sum));
}
Quaternion quat_z_rotation(double angle) {
  if (!std::isfinite(angle))
    throw std::invalid_argument("quaternion: finite angle");
  return {std::cos(angle / 2), 0, 0, std::sin(angle / 2)};
}
SlerpRun slerp_run(Quaternion q0, Quaternion q1, double t) {
  if (!std::isfinite(t) || t < 0 || t > 1)
    throw std::invalid_argument("slerp: fraction");
  SlerpRun r;
  r.start    = quat_normalize(q0);
  r.target   = quat_normalize(q1);
  double dot = quat_dot(r.start, r.target);
  if (dot < 0) {
    r.sign_flipped = true;
    for (double& v : r.target)
      v = -v;
    dot = -dot;
  }
  r.dot               = std::clamp(dot, 0., 1.);
  r.omega             = std::acos(r.dot);
  r.normalized_linear = r.dot > 0.9995;
  if (r.normalized_linear) {
    r.first_weight  = 1 - t;
    r.second_weight = t;
  } else {
    r.first_weight  = std::sin((1 - t) * r.omega) / std::sin(r.omega);
    r.second_weight = std::sin(t * r.omega) / std::sin(r.omega);
  }
  for (std::size_t i = 0; i < 4; ++i)
    r.value[i] = r.first_weight * r.start[i] + r.second_weight * r.target[i];
  r.value = quat_normalize(r.value);
  return r;
}
SlerpRun slerp_trace(Quaternion q0, Quaternion q1, double t) {
  return slerp_run(q0, q1, t);
}
Quaternion slerp(Quaternion q0, Quaternion q1, double t) {
  return slerp_run(q0, q1, t).value;
}
} // namespace datamunge::algorithms
