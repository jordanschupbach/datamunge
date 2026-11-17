/// @file embedding_diagnostics.cpp
/// @brief Canonical producing kernels and native report diagnostics.
#include <cmath>
#include <datamunge/stats/embedding_diagnostics.hpp>
#include <limits>
#include <numbers>
#include <random>
#include <stdexcept>
namespace datamunge::stats {
/// @brief Implement the checked canonical embedding_swiss_roll producer; contracts are declared in the matching header.
dstruct::DataFrame embedding_swiss_roll(std::size_t n, std::uint64_t seed) {
  if (!n || n > 2000)
    throw std::invalid_argument("swiss roll: n");
  std::mt19937_64                        rng(seed);
  std::uniform_real_distribution<double> u(0, 1);
  std::vector<double>                    t(n), h(n), x(n), y(n), z(n);
  for (std::size_t i = 0; i < n; ++i) {
    t[i] = 1.5 * std::numbers::pi * (1 + 2 * u(rng));
    h[i] = 21 * u(rng);
    x[i] = t[i] * std::cos(t[i]);
    y[i] = h[i];
    z[i] = t[i] * std::sin(t[i]);
  }
  dstruct::DataFrame d;
  d.add_column("x", x);
  d.add_column("y", y);
  d.add_column("z", z);
  d.add_column("t", t);
  d.add_column("height", h);
  return d;
}
/// @brief Implement the checked canonical embedding_group_distances producer; contracts are declared in the matching header.
dstruct::DataFrame embedding_group_distances(
    const linalg::DenseMatrix<double>& y,
    const std::vector<std::string>&    labels) {
  if (y.rows() != labels.size())
    throw std::invalid_argument("embedding labels shape");
  if (y.rows() > 2000)
    throw std::length_error("embedding diagnostics row cap");
  for (std::size_t i = 0; i < y.rows(); ++i)
    for (std::size_t c = 0; c < y.cols(); ++c)
      if (!std::isfinite(y(i, c)))
        throw std::invalid_argument("nonfinite coordinates");
  double      sums[2]   = {};
  std::size_t counts[2] = {};
  for (std::size_t i = 0; i < y.rows(); ++i)
    for (std::size_t j = i + 1; j < y.rows(); ++j) {
      double d = 0;
      for (std::size_t c = 0; c < y.cols(); ++c)
        d = std::hypot(d, y(i, c) - y(j, c));
      int k = labels[i] == labels[j] ? 0 : 1;
      sums[k] += d;
      ++counts[k];
    }
  dstruct::DataFrame d;
  d.add_column(
      "within",
      std::vector<double>{
          counts[0] ? sums[0] / counts[0] : std::numeric_limits<double>::quiet_NaN()});
  d.add_column(
      "between",
      std::vector<double>{
          counts[1] ? sums[1] / counts[1] : std::numeric_limits<double>::quiet_NaN()});
  d.add_column("within_pairs", std::vector<double>{double(counts[0])});
  d.add_column("between_pairs", std::vector<double>{double(counts[1])});
  return d;
}
/// @brief Implement the checked canonical embedding_abs_correlation producer; contracts are declared in the matching header.
double embedding_abs_correlation(const std::vector<double>& a, const std::vector<double>& b) {
  if (a.size() != b.size())
    throw std::invalid_argument("correlation shape");
  long double ma = 0, mb = 0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    if (!std::isfinite(a[i]) || !std::isfinite(b[i]))
      throw std::invalid_argument("nonfinite correlation");
    ma += a[i];
    mb += b[i];
  }
  if (a.size() < 2)
    return std::numeric_limits<double>::quiet_NaN();
  ma /= a.size();
  mb /= a.size();
  long double aa = 0, bb = 0, ab = 0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    long double x = a[i] - ma, y = b[i] - mb;
    aa += x * x;
    bb += y * y;
    ab += x * y;
  }
  return aa == 0 || bb == 0 ? std::numeric_limits<double>::quiet_NaN()
                            : double(std::abs(ab) / std::sqrt(aa * bb));
}
} // namespace datamunge::stats
