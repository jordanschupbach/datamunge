#pragma once
/// @file math.hpp
/// @brief Private allocation-free vector/quaternion arithmetic for native rigid dynamics.
#include <algorithm>
#include <cmath>
#include <datamunge/sim/types.hpp>
namespace datamunge::sim::detail {
/// @brief Componentwise sum of SI vectors.
inline Vec3 add(Vec3 a, const Vec3& b) {
  for (int i = 0; i < 3; ++i) a[i] += b[i];
  return a;
}
/// @brief Componentwise difference.
inline Vec3 sub(Vec3 a, const Vec3& b) {
  for (int i = 0; i < 3; ++i) a[i] -= b[i];
  return a;
}
/// @brief Scale each component.
inline Vec3 mul(Vec3 a, double s) {
  for (auto& v : a) v *= s;
  return a;
}
/// @brief Euclidean inner product.
inline double dot(const Vec3& a, const Vec3& b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
/// @brief Right-handed vector cross product.
inline Vec3 cross(const Vec3& a, const Vec3& b) {
  return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
          a[0] * b[1] - a[1] * b[0]};
}
/// @brief Stable vector norm.
inline double length(const Vec3& a) { return std::hypot(a[0], a[1], a[2]); }
/// @brief Unit vector with a caller-selected zero-vector fallback.
inline Vec3 unit(const Vec3& a, Vec3 fallback = {1, 0, 0}) {
  double n = length(a);
  return n > 1e-15 ? mul(a, 1 / n) : fallback;
}
/// @brief Conjugate of a unit quaternion.
inline Quaternion conjugate(Quaternion q) {
  for (int i = 1; i < 4; ++i) q[i] = -q[i];
  return q;
}
/// @brief Quaternion Hamilton product, in w,x,y,z order.
inline Quaternion product(const Quaternion& a, const Quaternion& b) {
  return {a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
          a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
          a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
          a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0]};
}
/// @brief Renormalize a finite, nonzero quaternion after integration.
inline Quaternion normalized(Quaternion q) {
  double n = std::hypot(std::hypot(q[0], q[1]), std::hypot(q[2], q[3]));
  for (auto& v : q) v /= n;
  return q;
}
/// @brief Rotate a vector by a unit quaternion.
inline Vec3 rotate(const Quaternion& q, const Vec3& v) {
  Vec3 xyz{q[1], q[2], q[3]};
  auto t = mul(cross(xyz, v), 2);
  return add(v, add(mul(t, q[0]), cross(xyz, t)));
}
/// @brief Unit quaternion for a right-handed axis-angle rotation.
inline Quaternion axis_angle(const Vec3& axis, double angle) {
  auto u = unit(axis);
  double s = std::sin(angle / 2);
  return {std::cos(angle / 2), s * u[0], s * u[1], s * u[2]};
}
/// @brief Integrate constant world angular velocity over delta seconds.
inline Quaternion advance(const Quaternion& q, const Vec3& w, double delta) {
  double speed = length(w);
  return speed > 1e-15 ? normalized(product(
                             axis_angle(mul(w, 1 / speed), speed * delta), q))
                       : q;
}
/// @brief World-space rotation error taking current to target with shortest quaternion sign.
inline Vec3 rotation_error(const Quaternion& target,
                           const Quaternion& current) {
  auto q = product(target, conjugate(current));
  if (q[0] < 0)
    for (auto& v : q) v = -v;
  double n = std::hypot(q[1], q[2], q[3]);
  return n > 1e-15 ? mul(Vec3{q[1], q[2], q[3]}, 2 * std::atan2(n, q[0]) / n)
                   : Vec3{0, 0, 0};
}
/// @brief Principal Cartesian basis vector.
inline Vec3 basis(int i) {
  Vec3 v{};
  v[i] = 1;
  return v;
}
/// @brief Clip vector magnitude while preserving direction.
inline Vec3 capped(Vec3 v, double limit) {
  double n = length(v);
  return n > limit ? mul(v, limit / n) : v;
}
}  // namespace datamunge::sim::detail
