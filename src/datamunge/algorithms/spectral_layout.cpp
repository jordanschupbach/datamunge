/// @file spectral_layout.cpp
/// @brief One actual Laplacian/Jacobi producer, with explicit finite/cap checks.
#include <algorithm>
#include <cmath>
#include <datamunge/algorithms/spectral_layout.hpp>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>
/// @brief Canonical checked native producers and owned values.
namespace datamunge::algorithms {
/// @brief Private invocation-local checked producing helpers.
namespace {
/// @brief Check actual finite floating arithmetic.
/// @param x Actual value.
/// @return Same finite value.
double finite(double x) {
  if (!std::isfinite(x))
    throw std::overflow_error("spectral: arithmetic");
  return x;
}
/// @brief Execute actual rotations and select the historical coordinate columns.
/// @param n Complete order.
/// @param edges Undirected endpoints.
/// @param o Policies.
/// @param capture Retain actual rotations.
/// @return Actual owned layout and eigensystem.
SpectralLayoutRun run(
    std::size_t                                             n,
    const std::vector<std::pair<std::size_t, std::size_t>>& edges,
    const SpectralLayoutOptions&                            o,
    bool                                                    capture) {
  const auto limit = std::numeric_limits<std::size_t>::max();
  if (!std::isfinite(o.tolerance) || o.tolerance <= 0)
    throw std::invalid_argument("spectral: tolerance");
  if ((o.max_vertices && n > o.max_vertices) || (n && n > limit / n))
    throw std::length_error("spectral: vertices/cells");
  const auto cells = n * n;
  if (cells > limit / 50)
    throw std::length_error("spectral: rotation capacity");
  for (const auto& [u, v] : edges)
    if (u >= n || v >= n)
      throw std::invalid_argument("spectral: endpoint");
  SpectralLayoutRun r;
  if (!n) {
    r.converged = true;
    return r;
  }
  std::vector<std::vector<double>> a(n, std::vector<double>(n)), vectors = a;
  for (const auto& [u, v] : edges)
    if (u != v) {
      a[u][u] = finite(a[u][u] + 1);
      a[v][v] = finite(a[v][v] + 1);
      a[u][v] = finite(a[u][v] - 1);
      a[v][u] = finite(a[v][u] - 1);
    }
  r.laplacian = a;
  for (std::size_t i = 0; i < n; ++i)
    vectors[i][i] = 1;
  const auto  cap      = o.max_rotations ? o.max_rotations : 50 * cells;
  std::size_t retained = 0;
  for (;;) {
    std::size_t p = 0, q = 0;
    double      largest = 0;
    for (std::size_t i = 0; i < n; ++i)
      for (std::size_t j = i + 1; j < n; ++j) {
        if (r.scan_visits == limit || (o.max_scan_visits && r.scan_visits >= o.max_scan_visits))
          throw std::length_error("spectral: scans");
        ++r.scan_visits;
        if (std::abs(a[i][j]) > largest) {
          largest = std::abs(a[i][j]);
          p       = i;
          q       = j;
        }
      }
    r.remaining_off_diagonal = largest;
    if (largest < o.tolerance) {
      r.converged = true;
      break;
    }
    if (r.rotations == cap)
      break;
    const double angle = .5 * std::atan2(2 * a[p][q], a[q][q] - a[p][p]), c = std::cos(angle),
                 s = std::sin(angle);
    for (std::size_t k = 0; k < n; ++k) {
      const double apk = a[p][k], aqk = a[q][k];
      a[p][k] = finite(c * apk - s * aqk);
      a[q][k] = finite(s * apk + c * aqk);
    }
    for (std::size_t k = 0; k < n; ++k) {
      const double akp = a[k][p], akq = a[k][q];
      a[k][p]          = finite(c * akp - s * akq);
      a[k][q]          = finite(s * akp + c * akq);
      const double vkp = vectors[k][p], vkq = vectors[k][q];
      vectors[k][p] = finite(c * vkp - s * vkq);
      vectors[k][q] = finite(s * vkp + c * vkq);
    }
    ++r.rotations;
    if (capture) {
      if (n == limit || retained > limit - n - 1)
        throw std::length_error("spectral: history capacity");
      retained += n + 1;
      if (o.max_history_values && retained > o.max_history_values)
        throw std::length_error("spectral: history");
      std::vector<double> diagonal(n);
      for (std::size_t i = 0; i < n; ++i)
        diagonal[i] = a[i][i];
      r.history.push_back({p, q, largest, angle, std::move(diagonal)});
    }
  }
  std::vector<std::size_t> order(n);
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(), [&](auto i, auto j) { return a[i][i] < a[j][j]; });
  r.eigenvalues.resize(n);
  r.eigenvectors.assign(n, std::vector<double>(n));
  for (std::size_t j = 0; j < n; ++j) {
    r.eigenvalues[j] = a[order[j]][order[j]];
    for (std::size_t i = 0; i < n; ++i)
      r.eigenvectors[i][j] = vectors[i][order[j]];
  }
  r.positions.resize(n);
  const auto x = n > 1 ? 1 : 0, y = n > 2 ? 2 : x;
  Point2D    mean;
  for (std::size_t i = 0; i < n; ++i) {
    r.positions[i] = {r.eigenvectors[i][x], r.eigenvectors[i][y]};
    mean.x += r.positions[i].x;
    mean.y += r.positions[i].y;
  }
  mean.x /= n;
  mean.y /= n;
  double scale = 0;
  for (auto& p : r.positions) {
    p.x   = finite(p.x - mean.x);
    p.y   = finite(p.y - mean.y);
    scale = std::max(scale, std::hypot(p.x, p.y));
  }
  if (scale)
    for (auto& p : r.positions) {
      p.x /= scale;
      p.y /= scale;
    }
  return r;
}
} // namespace
SpectralLayoutRun spectral_layout_run(
    std::size_t                                             n,
    const std::vector<std::pair<std::size_t, std::size_t>>& e,
    const SpectralLayoutOptions&                            o) {
  return run(n, e, o, false);
}
SpectralLayoutRun spectral_layout_trace(
    std::size_t                                             n,
    const std::vector<std::pair<std::size_t, std::size_t>>& e,
    const SpectralLayoutOptions&                            o) {
  return run(n, e, o, true);
}
std::vector<Point2D> spectral_layout(
    std::size_t                                             n,
    const std::vector<std::pair<std::size_t, std::size_t>>& e) {
  auto r = spectral_layout_run(n, e);
  if (!r.converged)
    throw std::runtime_error("spectral: unresolved eigenmodes");
  return r.positions;
}
std::vector<std::pair<std::size_t, std::size_t>> spectral_path_edges(std::size_t n) {
  if (n > 1000000)
    throw std::length_error("spectral: path fixture");
  std::vector<std::pair<std::size_t, std::size_t>> e;
  for (std::size_t i = 1; i < n; ++i)
    e.emplace_back(i - 1, i);
  return e;
}
} // namespace datamunge::algorithms
