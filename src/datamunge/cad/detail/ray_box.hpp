#pragma once
/// @file ray_box.hpp
/// @brief Reciprocal ray slabs with division fallback at uncertain floating-point boundaries.
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <datamunge/cad/render.hpp>
#include <limits>
namespace datamunge::cad::detail {
/// @brief Per-query ray state; reciprocal directions are shared by all visited bounds.
struct BoundsRay {
  const Ray&          ray;        ///< Borrowed world ray, valid for the synchronous query.
  Vec3d               inverse;    ///< Reciprocal direction on nonparallel axes.
  std::array<bool, 3> parallel{}; ///< Original abs(direction)<1e-300 classification.

  /// @brief Cache exactly the original parallel classification and three reciprocals.
  /// @param source Borrowed ray, normally normalized by the public render/pick API.
  explicit BoundsRay(const Ray& source) : ray(source) {
    for (int i = 0; i < 3; ++i) {
      parallel[i] = std::abs(ray.direction[i]) < 1e-300;
      inverse[i]  = parallel[i] ? 0 : 1 / ray.direction[i];
    }
  }
};
/// @brief Original division-based slab predicate, retained as the boundary/fuzz reference.
/// @param lo Lower world bounds.
/// @param hi Upper world bounds.
/// @param r Borrowed ray.
/// @param near Inclusive ray minimum.
/// @param far Inclusive ray maximum.
/// @return Exactly the renderer's original overlap decision, including flat/parallel slabs.
static inline bool
box_division(const Vec3d& lo, const Vec3d& hi, const Ray& r, double near, double far) {
  for (int i = 0; i < 3; ++i) {
    if (std::abs(r.direction[i]) < 1e-300) {
      if (r.origin[i] < lo[i] || r.origin[i] > hi[i])
        return false;
      continue;
    }
    double a = (lo[i] - r.origin[i]) / r.direction[i], b = (hi[i] - r.origin[i]) / r.direction[i];
    if (a > b)
      std::swap(a, b);
    near = std::max(a, near);
    far  = std::min(b, far);
    if (far < near)
      return false;
  }
  return true;
}
/// @brief Map finite IEEE doubles to an increasing integer rank, including signed zero.
/// @param value Finite slab/clip endpoint.
/// @return Monotonic bit rank used only to detect a narrow rounding-uncertainty interval.
static inline std::uint64_t ordered_endpoint(double value) {
  constexpr std::uint64_t sign = std::uint64_t{1} << 63;
  auto                    bits = std::bit_cast<std::uint64_t>(value);
  return bits & sign ? ~bits : bits | sign;
}
/// @brief Detect slab endpoints within sixteen representable doubles of each other.
/// @param a First endpoint.
/// @param b Second endpoint.
/// @return True when the original division predicate should resolve overlap.
static inline bool uncertain_endpoints(double a, double b) {
  auto x = ordered_endpoint(a), y = ordered_endpoint(b);
  return (x > y ? x - y : y - x) <= 16;
}
/// @brief Fast slabs with the original predicate for near-touching or nonfinite intervals.
/// A reciprocal multiply has extra rounding relative to division. A sixteen-ULP
/// guard covers the narrow uncertain interval; ambiguous cases use the original arithmetic
/// and original clip bounds. Thus this does not expand bounds or reorder hits.
/// @param lo Lower world bounds.
/// @param hi Upper world bounds.
/// @param q Shared per-ray reciprocals/parallel flags.
/// @param near Inclusive ray minimum.
/// @param far Inclusive ray maximum.
/// @return Original slab-overlap decision, with fewer divisions for ordinary intervals.
static inline bool
box_reciprocal(const Vec3d& lo, const Vec3d& hi, const BoundsRay& q, double near, double far) {
  const double initial_near = near, initial_far = far;
  for (int i = 0; i < 3; ++i) {
    if (q.parallel[i]) {
      if (q.ray.origin[i] < lo[i] || q.ray.origin[i] > hi[i])
        return false;
      continue;
    }
    double a = (lo[i] - q.ray.origin[i]) * q.inverse[i],
           b = (hi[i] - q.ray.origin[i]) * q.inverse[i];
    if (!std::isfinite(a) || !std::isfinite(b))
      return box_division(lo, hi, q.ray, initial_near, initial_far);
    if (a > b)
      std::swap(a, b);
    near = std::max(a, near);
    far  = std::min(b, far);
    if (far < near) {
      return uncertain_endpoints(near, far) ? box_division(lo, hi, q.ray, initial_near, initial_far)
                                            : false;
    }
  }
  return uncertain_endpoints(near, far) ? box_division(lo, hi, q.ray, initial_near, initial_far)
                                        : true;
}
} // namespace datamunge::cad::detail
