/// @file tricubic_example.cpp
/// @brief Canonical producing kernels and native report diagnostics.
#include <datamunge/algorithms/tricubic_interpolation.hpp>
namespace datamunge::algorithms {
/// @brief Implement the checked canonical tricubic_example producer; contracts are declared in the matching header.
TricubicExample tricubic_example() {
  auto f = [](double x, double y, double z) {
    return std::exp(-.08 * (x * x + y * y + z * z)) * std::cos(.6 * x + .4 * y - .3 * z);
  };
  std::vector<double> axis;
  for (int i = 0; i < 9; ++i)
    axis.push_back(i * .7 - 2.8);
  std::vector<double> v(729);
  for (int i = 0; i < 9; ++i)
    for (int j = 0; j < 9; ++j)
      for (int k = 0; k < 9; ++k)
        v[(i * 9 + j) * 9 + k] = f(axis[i], axis[j], axis[k]);
  TricubicExample r;
  for (int p = 0; p <= 100; ++p) {
    double x = -2.5 + .05 * p, y = .35, z = -.5;
    long   ix = long((x - axis[0]) / .7), iy = long((y - axis[0]) / .7),
         iz   = long((z - axis[0]) / .7);
    double tx = (x - axis[ix]) / .7, ty = (y - axis[iy]) / .7, tz = (z - axis[iz]) / .7, linear = 0;
    for (int i = 0; i < 2; ++i)
      for (int j = 0; j < 2; ++j)
        for (int k = 0; k < 2; ++k)
          linear += (i ? tx : 1 - tx) * (j ? ty : 1 - ty) * (k ? tz : 1 - tz) *
                    v[((ix + i) * 9 + iy + j) * 9 + iz + k];
    r.x.push_back(x);
    r.truth.push_back(f(x, y, z));
    r.cubic.push_back(tricubic_interpolate(axis, axis, axis, v, x, y, z));
    r.linear.push_back(linear);
  }
  return r;
}
} // namespace datamunge::algorithms
