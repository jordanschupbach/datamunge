#include <datamunge/fda/hierarchical_bspline.hpp>

#include <cmath>
#include <iostream>
#include <vector>

// Adaptive THB-spline fitting: refine only the quadrants whose residual is large, then refit
// every active function on every level. The coarse basis cannot follow a sharp bump;
// the hierarchy adds detail locally instead of refining the whole tensor grid.
int main() {
  namespace fda = datamunge::fda;

  auto bump = [](const std::vector<double>& p) {
    const double dx = p[0] - 0.3, dy = p[1] - 0.6;
    return std::exp(-80.0 * (dx * dx + dy * dy)) + 0.2 * p[0];
  };

  std::vector<std::vector<double>> points;
  std::vector<double>              values;
  for (int i = 0; i <= 40; ++i)
    for (int j = 0; j <= 40; ++j) {
      points.push_back({i / 40.0, j / 40.0});
      values.push_back(bump(points.back()));
    }

  const auto                        cubic = fda::BSplineBasis::open_uniform(3, 6);
  fda::TruncatedHierarchicalBSpline thb(fda::TensorBSpline({cubic, cubic}, std::vector<double>(36, 0.0)));

  auto rmse = [&] {
    double sse = 0.0;
    for (std::size_t k = 0; k < points.size(); ++k) sse += std::pow(thb.evaluate(points[k]) - values[k], 2);
    return std::sqrt(sse / static_cast<double>(points.size()));
  };

  thb.fit(points, values);
  std::cout << "level 0: " << thb.basis_size() << " functions, RMSE " << rmse() << "\n";

  for (int round = 1; round <= 3; ++round) {
    // For every finest-level cell, create only the children (quadrants) whose worst
    // residual exceeds the tolerance.
    int        refined = 0;
    const int  cells   = thb.cell_count();
    for (int cell = 0; cell < cells; ++cell) {
      if (thb.depth(cell) != round - 1) continue;
      for (int slot = 0; slot < 4; ++slot) {
        std::vector<double> lo = thb.cell_min(cell), hi = thb.cell_max(cell);
        for (int axis = 0; axis < 2; ++axis) {
          const double mid = 0.5 * (lo[axis] + hi[axis]);
          ((slot >> axis) & 1 ? lo[axis] : hi[axis]) = mid;
        }
        double worst = 0.0;
        for (std::size_t k = 0; k < points.size(); ++k) {
          const auto& p = points[k];
          if (p[0] >= lo[0] && p[0] <= hi[0] && p[1] >= lo[1] && p[1] <= hi[1])
            worst = std::max(worst, std::abs(thb.evaluate(p) - values[k]));
        }
        if (worst > 0.01) {
          thb.refine(cell, slot);
          ++refined;
        }
      }
    }
    thb.fit(points, values);
    std::cout << "level " << round << ": refined " << refined << " quadrants, " << thb.basis_size()
              << " functions, RMSE " << rmse() << "\n";
  }

  // For comparison: one uniform tensor grid with as many functions as the hierarchy.
  const int  n       = static_cast<int>(std::ceil(std::sqrt(static_cast<double>(thb.basis_size()))));
  const auto uniform = fda::BSplineBasis::open_uniform(3, static_cast<std::size_t>(n));
  const auto flat    = fda::fit_tensor_bspline({uniform, uniform}, points, values);
  double     sse     = 0.0;
  for (std::size_t k = 0; k < points.size(); ++k) sse += std::pow(flat.evaluate(points[k]) - values[k], 2);
  std::cout << "uniform " << n << "x" << n << " grid (" << n * n << " functions): RMSE "
            << std::sqrt(sse / static_cast<double>(points.size())) << "\n";
}
