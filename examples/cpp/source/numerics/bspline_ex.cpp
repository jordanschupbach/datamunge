#include <datamunge/dstruct/dstruct.hpp>
#include <datamunge/fda/bspline.hpp>
#include <datamunge/stats/stats.hpp>

#include <cmath>
#include <iostream>
#include <vector>

int main() {
  using datamunge::dstruct::DataFrame;
  using datamunge::fda::BSpline;
  using datamunge::stats::LM;

  // A sparse tensor-product design matrix: two linear bases, each with three functions.
  const auto basis = BSpline::open_uniform({1, 1}, {3, 3}, {0.0, 0.0}, {1.0, 1.0});
  const auto design = basis.evaluate({{0.2, 0.3}, {0.7, 0.8}});
  std::cout << "2-D sparse B-spline design: " << design.rows() << " x " << design.cols()
            << ", " << design.nnz() << " non-zeros\n";

  // Formula terms choose cubic bases with six functions per axis.  The model stores the
  // fitted basis, so predict() evaluates new x/y values against exactly the same knots.
  std::vector<double> x, y, z;
  for (std::size_t iy = 0; iy < 8; ++iy) {
    for (std::size_t ix = 0; ix < 8; ++ix) {
      const double xv = static_cast<double>(ix) / 7.0;
      const double yv = static_cast<double>(iy) / 7.0;
      x.push_back(xv);
      y.push_back(yv);
      z.push_back(std::sin(xv) + yv * yv);
    }
  }
  DataFrame data;
  data.add_column("x", x);
  data.add_column("y", y);
  data.add_column("z", z);

  LM surface(data, "z ~ bs(x, y)");
  std::cout << "Fitted z ~ bs(x, y): R^2=" << surface.r_squared()
            << ", coefficients=" << surface.coefficients().size() << "\n";

  DataFrame new_points;
  new_points.add_column("x", std::vector<double>{0.25, 0.75});
  new_points.add_column("y", std::vector<double>{0.50, 0.25});
  const auto fitted = surface.predict(new_points);
  for (double value : fitted) std::cout << "  prediction: " << value << "\n";
}
