/// @file cad_ray_bounds_test.cpp
/// @brief Differential slab tests against the original division predicate, including grazing
/// extremes.
#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>

#include "detail/ray_box.hpp"
using namespace datamunge::cad;
namespace {
/// @brief Require identical overlap decisions without relying on disabled Release assertions.
void compare(Vec3d lo, Vec3d hi, Ray ray, double near, double far) {
  bool old  = detail::box_division(lo, hi, ray, near, far);
  bool fast = detail::box_reciprocal(lo, hi, detail::BoundsRay(ray), near, far);
  if (old != fast)
    throw std::runtime_error("Reciprocal slabs differ from division reference");
}
} // namespace
int main() {
  std::mt19937_64                        random(20261102);
  std::uniform_real_distribution<double> value(-1, 1);
  constexpr int                          count = 1000000;
  for (int k = 0; k < count; ++k) {
    double scale = std::ldexp(1., int(random() % 1800) - 900);
    Vec3d  lo{value(random) * scale, value(random) * scale, value(random) * scale};
    Vec3d  hi = lo + Vec3d{
                        std::abs(value(random)) * scale,
                        std::abs(value(random)) * scale,
                        std::abs(value(random)) * scale};
    Ray ray{
        {value(random) * scale, value(random) * scale, value(random) * scale},
        {value(random), value(random), value(random)}};
    if (k % 7 == 0)
      ray.direction[k % 3] = 0;
    if (k % 11 == 0)
      ray.direction[k % 3] = -0.;
    if (k % 13 == 0)
      ray.direction[k % 3] = 1e-301;
    if (k % 17 == 0)
      hi[k % 3] = lo[k % 3];
    double near = std::abs(value(random)) * scale, far = near + std::abs(value(random)) * scale;
    compare(lo, hi, ray, near, far);
    // Put an endpoint within a few ulps of the clip interval; the guard must
    // retain the original arithmetic at these signed/flat/grazing boundaries.
    int axis         = k % 3;
    ray.origin[axis] = lo[axis] - near * ray.direction[axis];
    compare(
        lo,
        hi,
        ray,
        std::nextafter(near, 0.),
        std::nextafter(far, std::numeric_limits<double>::infinity()));
  }
  for (double d : {0., -0., 1e-310, 1e-300, 1e-299, 1., -1., 1e308})
    for (double x : {-1e308, -1., 0., 1., 1e308})
      compare({-1, -1, -1}, {1, 1, 1}, {{x, 0, 0}, {d, 0, 0}}, 0, 1e100);
  std::cout << "2,000,040 slab comparisons match the original division predicate\n";
}
