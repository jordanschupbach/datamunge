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
