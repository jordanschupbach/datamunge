#include <gtest/gtest.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <datamunge/algorithms/filtered_back_projection.hpp>
#include <limits>
#include <numbers>
#include <numeric>
#include <random>
#include <stdexcept>
namespace {
using namespace datamunge::algorithms;
constexpr double pi = std::numbers::pi_v<double>;
// Explicit dense hat-basis operator: no floor, two-bin selection or shared helper.
std::vector<std::vector<double>> matrix(int n, int d, const std::vector<double>& a) {
  std::vector<std::vector<double>> A(a.size() * d, std::vector<double>(n * n));
  for (std::size_t v = 0; v < a.size(); ++v)
    for (int detector = 0; detector < d; ++detector)
      for (int r = 0; r < n; ++r)
        for (int c = 0; c < n; ++c) {
          const double s = (c - (n - 1) / 2.) * std::cos(a[v]) +
                           (r - (n - 1) / 2.) * std::sin(a[v]) + (d - 1) / 2.;
          A[v * d + detector][r * n + c] = std::max(0., 1 - std::abs(s - detector));
        }
  return A;
}
std::vector<double> flatten(const std::vector<std::vector<double>>& s) {
  std::vector<double> out;
  for (const auto& r : s)
    out.insert(out.end(), r.begin(), r.end());
  return out;
}
void near(const std::vector<double>& a, const std::vector<double>& b, double tol = 2e-12) {
  ASSERT_EQ(a.size(), b.size());
  for (std::size_t i = 0; i < a.size(); ++i)
    EXPECT_NEAR(a[i], b[i], tol * (1 + std::abs(a[i]) + std::abs(b[i]))) << i;
}
TEST(FilteredBackProjectionNative, TinyActualSplatAndFilter) {
  std::vector<double> img(9);
  img[5]       = 1;
  const auto a = parallel_projection_angles(4);
  const auto s = radon_transform(img, 3, a, 3);
  EXPECT_EQ(s[0], (std::vector<double>{0, 0, 1}));
  EXPECT_NEAR(s[1][1], 1 - std::sqrt(.5), 1e-15);
  EXPECT_NEAR(s[1][2], std::sqrt(.5), 1e-15);
  for (const auto& r : s)
    EXPECT_NEAR(std::accumulate(r.begin(), r.end(), 0.), 1, 1e-15);
  const auto f = ram_lak_filter({{0, 0, 1, 0, 0}});
  near(f.front(), {0, -1 / (pi * pi), .25, -1 / (pi * pi), 0}, 1e-15);
  EXPECT_NEAR(back_projection({{2}}, 1, {0}, 1)[0], 2 * pi, 1e-15);
  EXPECT_NEAR(filtered_back_projection({{4}}, 1, {0}, 1)[0], pi, 1e-15);
}
TEST(FilteredBackProjectionNative, IndependentAxisAlignedSums) {
  for (int n = 1; n <= 16; ++n) {
    std::vector<double> img(n * n), expected(n);
    for (int r = 0; r < n; ++r)
      for (int c = 0; c < n; ++c) {
        img[r * n + c] = (r * n + c) % 7 - 3;
        expected[c] += img[r * n + c];
      }
    EXPECT_EQ(radon_transform(img, n, {0}, n).front(), expected);
    std::vector<double> row(n);
    std::iota(row.begin(), row.end(), -3);
    const auto b = back_projection({row}, n, {0}, n);
    for (int r = 0; r < n; ++r)
      for (int c = 0; c < n; ++c)
        EXPECT_DOUBLE_EQ(b[r * n + c], pi * row[c]);
  }
}
TEST(FilteredBackProjectionNative, IndependentDenseHatMatrix) {
  std::mt19937 rng(110);
  for (int fixture = 0; fixture < 300; ++fixture) {
    const int           n = 1 + rng() % 7, d = 1 + rng() % 13, m = 1 + rng() % 9;
    const auto          angles = parallel_projection_angles(m);
    const auto          A      = matrix(n, d, angles);
    std::vector<double> x(n * n), y(m * d);
    for (auto& v : x)
      v = double(int(rng() % 21) - 10) / 10;
    for (auto& v : y)
      v = double(int(rng() % 21) - 10) / 10;
    const auto          sx = flatten(radon_transform(x, n, angles, d));
    std::vector<double> expected(m * d);
    for (std::size_t k = 0; k < A.size(); ++k)
      expected[k] = std::inner_product(A[k].begin(), A[k].end(), x.begin(), 0.);
    near(sx, expected);
    std::vector<std::vector<double>> s(m, std::vector<double>(d));
    for (int v = 0; v < m; ++v)
      std::copy_n(y.begin() + v * d, d, s[v].begin());
    std::vector<double> bt(n * n);
    for (int p = 0; p < n * n; ++p)
      for (int k = 0; k < m * d; ++k)
        bt[p] += (pi / m) * A[k][p] * y[k];
    near(back_projection(s, n, angles, d), bt);
    const auto          actual = flatten(ram_lak_filter(s));
    std::vector<double> filter(m * d);
    for (int v = 0; v < m; ++v)
      for (int out = 0; out < d; ++out)
        for (int in = 0; in < d; ++in) {
          const int    lag = out - in;
          const double h   = lag ? (std::cos(pi * lag) - 1) / (2 * pi * pi * lag * lag) : .25;
          filter[v * d + out] += h * s[v][in];
        }
    near(actual, filter);
    for (int p = 0; p < n * n; ++p) {
      bt[p] = 0;
      for (int k = 0; k < m * d; ++k)
        bt[p] += (pi / m) * A[k][p] * filter[k];
    }
    near(filtered_back_projection(s, n, angles, d), bt);
  }
}
TEST(FilteredBackProjectionNative, ScaledAdjointAndLinearity) {
  std::mt19937 rng(111);
  for (int fixture = 0; fixture < 300; ++fixture) {
    const int           n = 1 + rng() % 9, d = 1 + rng() % 17, m = 1 + rng() % 12;
    const auto          angles = parallel_projection_angles(m);
    std::vector<double> x(n * n), z(n * n), sum(n * n);
    for (std::size_t i = 0; i < x.size(); ++i) {
      x[i]   = double(int(rng() % 17) - 8) / 8;
      z[i]   = double(int(rng() % 17) - 8) / 8;
      sum[i] = x[i] + z[i];
    }
    std::vector<std::vector<double>> y(m, std::vector<double>(d));
    for (auto& r : y)
      for (auto& v : r)
        v = double(int(rng() % 17) - 8) / 8;
    const auto ax = flatten(radon_transform(x, n, angles, d)), ay = flatten(y),
               by    = back_projection(y, n, angles, d);
    const double lhs = std::inner_product(ax.begin(), ax.end(), ay.begin(), 0.),
                 rhs = (m / pi) * std::inner_product(x.begin(), x.end(), by.begin(), 0.);
    EXPECT_NEAR(lhs, rhs, 3e-12 * (1 + std::abs(lhs) + std::abs(rhs)));
    auto az = flatten(radon_transform(z, n, angles, d));
    for (std::size_t i = 0; i < az.size(); ++i)
      az[i] += ax[i];
    near(flatten(radon_transform(sum, n, angles, d)), az);
    const auto f = ram_lak_filter(y);
    EXPECT_EQ(back_projection(f, n, angles, d), filtered_back_projection(y, n, angles, d));
  }
}
TEST(FilteredBackProjectionNative, PhantomsAndRoundedAngles) {
  for (int n = 1; n <= 31; ++n)
    for (double radius : {0., 1., 1.5, n / 3.}) {
      const auto img = disk_phantom(n, radius, -2);
      for (int r = 0; r < n; ++r)
        for (int c = 0; c < n; ++c) {
          const double dx = c - (n - 1) / 2., dy = r - (n - 1) / 2.;
          EXPECT_EQ(img[r * n + c], dx * dx + dy * dy <= radius * radius ? -2 : 0);
        }
    }
  EXPECT_EQ(disk_phantom(2, 0), std::vector<double>(4));
  EXPECT_EQ(disk_phantom(8, std::numeric_limits<double>::max()), std::vector<double>(64, 1));
  for (std::size_t m = 1; m <= 360; ++m) {
    const auto a = parallel_projection_angles(m);
    ASSERT_EQ(a.size(), m);
    for (std::size_t i = 0; i < m; ++i) {
      EXPECT_DOUBLE_EQ(a[i], pi * i / m);
      EXPECT_GE(a[i], 0);
      EXPECT_LT(a[i], pi);
      if (i)
        EXPECT_GT(a[i], a[i - 1]);
    }
  }
}
TEST(FilteredBackProjectionNative, ClippingAndFiniteKernelEdges) {
  std::vector<double> impulse(9);
  impulse[5] = 1;
  EXPECT_EQ(radon_transform(impulse, 3, {0}, 1).front(), std::vector<double>{0});
  EXPECT_EQ(radon_transform(impulse, 3, {0}, 3).front(), (std::vector<double>{0, 0, 1}));
  EXPECT_DOUBLE_EQ(radon_transform({1, 1, 1, 1}, 2, {0}, 1)[0][0], 2);
  const auto constant = ram_lak_filter({std::vector<double>(9, 1)});
  EXPECT_GT(constant[0][4], 0);
  EXPECT_GT(constant[0][0], constant[0][4]);
  EXPECT_TRUE(ram_lak_filter({}).empty());
  const auto y = radon_transform({-0.}, 1, {0}, 1);
  EXPECT_EQ(y[0][0], 0);
}
TEST(FilteredBackProjectionNative, ShapesAndNonfiniteControls) {
  const double nan = std::numeric_limits<double>::quiet_NaN(),
               inf = std::numeric_limits<double>::infinity();
  for (int n : {INT_MIN, -1, 0}) {
    EXPECT_THROW(disk_phantom(n, 1), std::invalid_argument);
    EXPECT_THROW(radon_transform({}, n, {0}, 1), std::invalid_argument);
    EXPECT_THROW(back_projection({{1}}, n, {0}, 1), std::invalid_argument);
    EXPECT_THROW(filtered_back_projection({{1}}, n, {0}, 1), std::invalid_argument);
  }
  EXPECT_THROW(disk_phantom(INT_MAX, 1), std::length_error);
  EXPECT_THROW(parallel_projection_angles(0), std::invalid_argument);
  EXPECT_THROW(
      parallel_projection_angles(std::numeric_limits<std::size_t>::max()), std::length_error);
  for (double radius : {-1., nan, inf})
    EXPECT_THROW(disk_phantom(1, radius), std::invalid_argument);
  EXPECT_THROW(disk_phantom(1, 1, nan), std::invalid_argument);
  for (int d : {INT_MIN, -1, 0})
    EXPECT_THROW(radon_transform({1}, 1, {0}, d), std::invalid_argument);
  EXPECT_THROW(radon_transform({}, 2, {0}, 3), std::invalid_argument);
  EXPECT_THROW(radon_transform({1, 2}, 1, {0}, 1), std::invalid_argument);
  EXPECT_THROW(radon_transform({1}, 1, {}, 1), std::invalid_argument);
  for (double v : {nan, inf, -inf}) {
    EXPECT_THROW(radon_transform({v}, 1, {0}, 1), std::invalid_argument);
    EXPECT_THROW(radon_transform({1}, 1, {v}, 1), std::invalid_argument);
    EXPECT_THROW(ram_lak_filter({{v}}), std::invalid_argument);
    EXPECT_THROW(back_projection({{v}}, 1, {0}, 1), std::invalid_argument);
    EXPECT_THROW(filtered_back_projection({{1}}, 1, {v}, 1), std::invalid_argument);
  }
  EXPECT_THROW(ram_lak_filter({{}}), std::invalid_argument);
  EXPECT_THROW(ram_lak_filter({{1}, {1, 2}}), std::invalid_argument);
  EXPECT_THROW(filtered_back_projection({}, 1, {}, 1), std::invalid_argument);
  EXPECT_THROW(filtered_back_projection({}, 1, {0}, 1), std::invalid_argument);
  EXPECT_THROW(filtered_back_projection({{1}}, 1, {0, 1}, 1), std::invalid_argument);
  EXPECT_THROW(back_projection({{1}}, 1, {0}, 2), std::invalid_argument);
}
TEST(FilteredBackProjectionNative, FiniteArithmeticOverflowAndHugeAngles) {
  const auto max = std::numeric_limits<double>::max();
  EXPECT_THROW(radon_transform({max, max, max, max}, 2, {0}, 2), std::overflow_error);
  EXPECT_THROW(back_projection({{max}}, 1, {0}, 1), std::overflow_error);
  EXPECT_THROW(filtered_back_projection({{-max, max, -max}}, 1, {0}, 3), std::overflow_error);
  EXPECT_TRUE(std::isfinite(filtered_back_projection({{max}}, 1, {0}, 1).front()));
  EXPECT_NO_THROW(radon_transform({1}, 1, {max, -max}, 1));
  EXPECT_NO_THROW(back_projection({{1}, {1}}, 1, {max, -max}, 1));
}
TEST(FilteredBackProjectionNative, OriginalDiskAndCorrelationScale) {
  const int  n   = 96;
  const auto img = disk_phantom(n, 22), angles = parallel_projection_angles(120);
  const auto s = radon_transform(img, n, angles, 144), f = ram_lak_filter(s);
  const auto rec = back_projection(f, n, angles, 144);
  double     sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0;
  for (std::size_t i = 0; i < img.size(); ++i) {
    sx += img[i];
    sy += rec[i];
    sxx += img[i] * img[i];
    syy += rec[i] * rec[i];
    sxy += img[i] * rec[i];
  }
  const double count = img.size(),
               corr  = (count * sxy - sx * sy) /
                      std::sqrt((count * sxx - sx * sx) * (count * syy - sy * sy));
  EXPECT_NEAR(corr, .9941, .00005);
  EXPECT_NEAR(rec[(n / 2) * n + n / 2], .9860, .00005);
  EXPECT_EQ(rec, filtered_back_projection(s, n, angles, 144));
}
TEST(FilteredBackProjectionNative, OwnedStageResultsAndCallableSignatures) {
  auto       img    = disk_phantom(3, 0);
  auto       s      = radon_transform(img, 3, parallel_projection_angles(4), 5);
  const auto s_copy = s;
  auto       f      = ram_lak_filter(s);
  const auto f_copy = f;
  const auto rec    = back_projection(f, 3, parallel_projection_angles(4), 5);
  img[4]            = 99;
  s[0][2]           = 99;
  f[0][2]           = 99;
  EXPECT_EQ(s_copy[0][2], 1);
  EXPECT_NE(f_copy[0][2], 99);
  EXPECT_TRUE(std::isfinite(rec[4]));
  std::vector<std::vector<double>> (*forward)(
      const std::vector<double>&, int, const std::vector<double>&, int) = &radon_transform;
  std::vector<double> (*inverse)(
      const std::vector<std::vector<double>>&, int, const std::vector<double>&, int) =
      &filtered_back_projection;
  EXPECT_EQ(forward({2}, 1, {0}, 1)[0][0], 2);
  EXPECT_NEAR(inverse({{4}}, 1, {0}, 1)[0], pi, 1e-15);
}
} // namespace
