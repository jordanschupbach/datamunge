#include <datamunge/algorithms/graph_layout.hpp>

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>

namespace datamunge::algorithms {
namespace {

using Edge = std::pair<std::size_t, std::size_t>;

void validate(std::size_t n, const std::vector<Edge>& edges) {
  for (const auto& [u, v] : edges)
    if (u >= n || v >= n) throw std::invalid_argument("graph layout: endpoint out of range");
}

void center_and_scale(std::vector<Point2D>& points) {
  if (points.empty()) return;
  Point2D mean;
  for (const auto& p : points) { mean.x += p.x; mean.y += p.y; }
  mean.x /= points.size(); mean.y /= points.size();
  double scale = 0.0;
  for (auto& p : points) {
    p.x -= mean.x; p.y -= mean.y;
    scale = std::max(scale, std::hypot(p.x, p.y));
  }
  if (scale > 0.0) for (auto& p : points) { p.x /= scale; p.y /= scale; }
}

void jacobi(std::vector<std::vector<double>>& a, std::vector<std::vector<double>>& vectors) {
  const std::size_t n = a.size();
  vectors.assign(n, std::vector<double>(n));
  for (std::size_t i = 0; i < n; ++i) vectors[i][i] = 1.0;
  for (std::size_t step = 0; step < 50 * n * n; ++step) {
    std::size_t p = 0, q = 0; double largest = 0.0;
    for (std::size_t i = 0; i < n; ++i)
      for (std::size_t j = i + 1; j < n; ++j)
        if (std::abs(a[i][j]) > largest) { largest = std::abs(a[i][j]); p = i; q = j; }
    if (largest < 1e-12) break;
    const double angle = 0.5 * std::atan2(2.0 * a[p][q], a[q][q] - a[p][p]);
    const double c = std::cos(angle), s = std::sin(angle);
    for (std::size_t k = 0; k < n; ++k) {
      const double apk = a[p][k], aqk = a[q][k];
      a[p][k] = c * apk - s * aqk; a[q][k] = s * apk + c * aqk;
    }
    for (std::size_t k = 0; k < n; ++k) {
      const double akp = a[k][p], akq = a[k][q];
      a[k][p] = c * akp - s * akq; a[k][q] = s * akp + c * akq;
      const double vkp = vectors[k][p], vkq = vectors[k][q];
      vectors[k][p] = c * vkp - s * vkq; vectors[k][q] = s * vkp + c * vkq;
    }
  }
}

} // namespace

std::vector<Point2D> force_directed_layout(
    std::size_t n, const std::vector<Edge>& edges, std::size_t iterations, std::uint64_t seed) {
  validate(n, edges);
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<double> uniform(-0.5, 0.5);
  std::vector<Point2D> p(n), displacement(n);
  for (auto& point : p) point = {uniform(rng), uniform(rng)};
  if (n < 2) return p;
  const double k = 1.0 / std::sqrt(static_cast<double>(n));
  for (std::size_t iteration = 0; iteration < iterations; ++iteration) {
    std::fill(displacement.begin(), displacement.end(), Point2D{});
    for (std::size_t u = 0; u < n; ++u) for (std::size_t v = u + 1; v < n; ++v) {
      double dx = p[u].x - p[v].x, dy = p[u].y - p[v].y;
      double d = std::max(1e-9, std::hypot(dx, dy)), f = k * k / d;
      displacement[u].x += dx / d * f; displacement[u].y += dy / d * f;
      displacement[v].x -= dx / d * f; displacement[v].y -= dy / d * f;
    }
    for (const auto& [u, v] : edges) if (u != v) {
      double dx = p[u].x - p[v].x, dy = p[u].y - p[v].y;
      double d = std::max(1e-9, std::hypot(dx, dy)), f = d * d / k;
      displacement[u].x -= dx / d * f; displacement[u].y -= dy / d * f;
      displacement[v].x += dx / d * f; displacement[v].y += dy / d * f;
    }
    const double temperature = 0.1 * (1.0 - static_cast<double>(iteration) / iterations);
    for (std::size_t u = 0; u < n; ++u) {
      const double d = std::max(1e-9, std::hypot(displacement[u].x, displacement[u].y));
      const double step = std::min(d, temperature);
      p[u].x += displacement[u].x / d * step; p[u].y += displacement[u].y / d * step;
    }
  }
  center_and_scale(p);
  return p;
}

std::vector<Circle2D> circle_packing_layout(
    std::size_t n, const std::vector<Edge>& edges, std::size_t iterations) {
  validate(n, edges);
  std::vector<Circle2D> circles(n);
  if (n == 0) return circles;
  std::vector<std::vector<bool>> adjacent(n, std::vector<bool>(n));
  for (const auto& [u, v] : edges) adjacent[u][v] = adjacent[v][u] = true;
  constexpr double pi = 3.14159265358979323846;
  for (std::size_t i = 0; i < n; ++i)
    circles[i].center = {3.0 * std::cos(2 * pi * i / n), 3.0 * std::sin(2 * pi * i / n)};
  for (std::size_t step = 0; step < iterations; ++step) {
    std::vector<Point2D> delta(n);
    for (std::size_t u = 0; u < n; ++u) for (std::size_t v = u + 1; v < n; ++v) {
      double dx = circles[v].center.x - circles[u].center.x;
      double dy = circles[v].center.y - circles[u].center.y;
      double d = std::max(1e-9, std::hypot(dx, dy));
      const bool is_adjacent = adjacent[u][v];
      double target = is_adjacent ? 2.0 : 2.15;
      double force = (d - target) * (is_adjacent ? 0.08 : (d < target ? 0.08 : 0.0));
      delta[u].x += dx / d * force; delta[u].y += dy / d * force;
      delta[v].x -= dx / d * force; delta[v].y -= dy / d * force;
    }
    for (std::size_t i = 0; i < n; ++i) {
      circles[i].center.x += delta[i].x; circles[i].center.y += delta[i].y;
    }
  }
  std::vector<Point2D> centers;
  for (const auto& c : circles) centers.push_back(c.center);
  center_and_scale(centers);
  for (std::size_t i = 0; i < n; ++i) { circles[i].center = centers[i]; circles[i].radius = 0.45; }
  return circles;
}

std::vector<Point2D> spectral_layout(std::size_t n, const std::vector<Edge>& edges) {
  validate(n, edges);
  if (n == 0) return {};
  std::vector<std::vector<double>> laplacian(n, std::vector<double>(n));
  for (const auto& [u, v] : edges) if (u != v) {
    laplacian[u][u] += 1; laplacian[v][v] += 1;
    laplacian[u][v] -= 1; laplacian[v][u] -= 1;
  }
  std::vector<std::vector<double>> eigenvectors;
  jacobi(laplacian, eigenvectors);
  std::vector<std::size_t> order(n);
  for (std::size_t i = 0; i < n; ++i) order[i] = i;
  std::sort(order.begin(), order.end(), [&](auto i, auto j) { return laplacian[i][i] < laplacian[j][j]; });
  std::vector<Point2D> result(n);
  const std::size_t xcol = n > 1 ? order[1] : order[0], ycol = n > 2 ? order[2] : xcol;
  for (std::size_t i = 0; i < n; ++i) result[i] = {eigenvectors[i][xcol], eigenvectors[i][ycol]};
  center_and_scale(result);
  return result;
}

} // namespace datamunge::algorithms
