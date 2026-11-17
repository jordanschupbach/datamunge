/// @file laplacian_analysis.cpp
/// @brief Original curve generator and explicit widened matched-point MSE diagnostic.
#include <cmath>
#include <datamunge/geometry/laplacian_analysis.hpp>
#include <random>
#include <stdexcept>
namespace datamunge::geometry {
LaplacianCurveStudy laplacian_demo_curve() {
  LaplacianCurveStudy out;
  out.vertices.resize(41);
  out.neighbors.resize(41);
  out.fixed.assign(41, 0);
  out.reference.resize(41);
  std::mt19937                           rng(42);
  std::uniform_real_distribution<double> noise(-3, 3);
  for (int i = 0; i < 41; ++i) {
    const double x = i, y = 8. * std::sin(x / 6.);
    out.reference[i] = {x, y};
    out.vertices[i]  = {x, y + (i == 0 || i == 40 ? 0. : noise(rng))};
  }
  for (int i = 0; i < 40; ++i) {
    out.neighbors[i].push_back(i + 1);
    out.neighbors[i + 1].push_back(i);
  }
  out.fixed.front() = out.fixed.back() = 1;
  return out;
}
long double laplacian_point_mse(
    const std::vector<Point2D>& observed,
    const std::vector<Point2D>& reference) {
  if (observed.empty() || observed.size() != reference.size())
    throw std::invalid_argument("Laplacian point MSE shape");
  for (std::size_t i = 0; i < observed.size(); ++i)
    if (!std::isfinite(observed[i].x) || !std::isfinite(observed[i].y) ||
        !std::isfinite(reference[i].x) || !std::isfinite(reference[i].y))
      throw std::invalid_argument("Laplacian point MSE finite data");
  long double sum = 0;
  for (std::size_t i = 0; i < observed.size(); ++i) {
    const long double dx = static_cast<long double>(observed[i].x) - reference[i].x,
                      dy = static_cast<long double>(observed[i].y) - reference[i].y;
    sum += dx * dx + dy * dy;
    if (!std::isfinite(sum))
      throw std::overflow_error("Laplacian point MSE range");
  }
  return sum / observed.size();
}
} // namespace datamunge::geometry
