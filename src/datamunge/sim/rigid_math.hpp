#pragma once
/// @file rigid_math.hpp
/// @brief Full COM inertia and momentum/energy-preserving free rigid rotation.
#include <limits>
#include <stdexcept>

#include "math.hpp"
namespace datamunge::sim::detail {
/// @brief Row-major three-dimensional linear operator.
using Tensor3 = std::array<double, 9>;
/// @brief Multiply a full tensor by a vector.
/// @param a Row-major finite operator.
/// @param v Input vector.
/// @return Matrix-vector product.
inline Vec3 apply_tensor(const Tensor3& a, Vec3 v) {
  return {
      a[0] * v[0] + a[1] * v[1] + a[2] * v[2],
      a[3] * v[0] + a[4] * v[1] + a[5] * v[2],
      a[6] * v[0] + a[7] * v[1] + a[8] * v[2]};
}
/// @brief Invert a nonsingular full tensor; reject invalid numerical solves.
/// @param a Row-major finite nonsingular operator.
/// @return Inverse operator.
/// @throws std::runtime_error Nonfinite or singular determinant.
inline Tensor3 invert_tensor(const Tensor3& a) {
  Tensor3 c{
      a[4] * a[8] - a[5] * a[7],
      a[2] * a[7] - a[1] * a[8],
      a[1] * a[5] - a[2] * a[4],
      a[5] * a[6] - a[3] * a[8],
      a[0] * a[8] - a[2] * a[6],
      a[2] * a[3] - a[0] * a[5],
      a[3] * a[7] - a[4] * a[6],
      a[1] * a[6] - a[0] * a[7],
      a[0] * a[4] - a[1] * a[3]};
  double det = a[0] * c[0] + a[1] * c[3] + a[2] * c[6];
  if (!std::isfinite(det) || std::abs(det) < std::numeric_limits<double>::min())
    throw std::runtime_error("singular native inertia/rotation solve");
  for (auto& x : c)
    x /= det;
  return c;
}
/// @brief Authored full inertia or uniform legacy primitive COM inertia.
/// @param b Valid body with optional authored COM tensor.
/// @return Row-major COM inertia, kg m²; hull/compound fallback uses dimensions.
inline Tensor3 body_tensor(const Body& b) {
  if (b.inertia_tensor != Tensor3{})
    return b.inertia_tensor;
  double m = b.mass, r = b.dimensions[0] / 2;
  Vec3   v{};
  if (b.shape == Shape::Sphere)
    v = {.4 * m * r * r, .4 * m * r * r, .4 * m * r * r};
  else if (b.shape == Shape::Ellipsoid) {
    for (int i = 0; i < 3; ++i)
      v[i] = m * (std::pow(b.dimensions[(i + 1) % 3], 2) + std::pow(b.dimensions[(i + 2) % 3], 2)) /
             20;
  } else if (b.shape == Shape::Capsule) {
    double l = b.dimensions[1] - 2 * r, mc = m * l / (l + 4 * r / 3), ms = m - mc;
    double a = mc * (3 * r * r + l * l) / 12 + ms * (.4 * r * r + .375 * l * r + .25 * l * l);
    v        = {a, .5 * mc * r * r + .4 * ms * r * r, a};
  } else
    for (int i = 0; i < 3; ++i)
      v[i] = m * (std::pow(b.dimensions[(i + 1) % 3], 2) + std::pow(b.dimensions[(i + 2) % 3], 2)) /
             12;
  return {v[0], 0, 0, 0, v[1], 0, 0, 0, v[2]};
}
/// @brief World COM of a reference-origin rigid body.
/// @param b Valid reference-origin body.
/// @return World COM position in metres.
inline Vec3 center(const Body& b) {
  return add(b.position, rotate(b.rotation, b.center_of_mass));
}
/// @brief Implicit midpoint Euler equations with matching Cayley attitude update.
/// Free motion preserves kinetic energy and world angular momentum to solve tolerance.
/// External forces/contact impulses are split before this rotation. Fast rotations
/// subdivide to bound each nonlinear solve; failure throws instead of resetting.
/// @param b Mutable body, with COM linear and world angular velocity.
/// @param dt Positive integration interval, seconds.
/// @param gyro Whether to integrate free Euler equations; kinematic/locked rotation stays
/// prescribed.
/// @par References
/// - J. C. Simo and K. K. Wong (1991), doi:10.1002/nme.1620310103.
inline void integrate_pose(Body& b, double dt, bool gyro) {
  auto com = add(center(b), mul(b.linear_velocity, dt));
  if (gyro && b.motion == Motion::Dynamic &&
      b.allowed_rotation == std::array<bool, 3>{true, true, true}) {
    auto I = body_tensor(b);
    auto w = rotate(conjugate(b.rotation), b.angular_velocity);
    int  n = std::max(1, int(std::ceil(length(w) * dt / .25)));
    if (n > 4096)
      throw std::runtime_error("native rotation subdivision exhausted");
    double h = dt / n;
    for (int k = 0; k < n; ++k) {
      auto old       = w;
      bool converged = false;
      for (int it = 0; it < 20; ++it) {
        auto mid = mul(add(old, w), .5), L = apply_tensor(I, mid);
        auto f = add(apply_tensor(I, sub(w, old)), mul(cross(mid, L), h));
        if (length(f) < 1e-13 * (1 + length(L))) {
          converged = true;
          break;
        }
        auto J = I;
        for (int col = 0; col < 3; ++col) {
          auto v = mul(add(cross(basis(col), L), cross(mid, apply_tensor(I, basis(col)))), h / 2);
          for (int row = 0; row < 3; ++row)
            J[3 * row + col] += v[row];
        }
        w = sub(w, apply_tensor(invert_tensor(J), f));
      }
      if (!converged)
        throw std::runtime_error("native gyroscopic midpoint did not converge");
      auto mid   = mul(add(old, w), h / 4);
      auto dq    = normalized(Quaternion{1, mid[0], mid[1], mid[2]});
      b.rotation = normalized(product(b.rotation, dq));
    }
    b.angular_velocity = rotate(b.rotation, w);
  } else
    b.rotation = advance(b.rotation, b.angular_velocity, dt);
  b.position = sub(com, rotate(b.rotation, b.center_of_mass));
}
} // namespace datamunge::sim::detail
