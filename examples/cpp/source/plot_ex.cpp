#include <datamunge/linalg/regression.hpp>
#include <datamunge/plot/plot.hpp>

#include <filesystem>
#include <iomanip>
#include <iostream>
#include <vector>

using datamunge::linalg::DenseMatrix;
using datamunge::linalg::linear_regression;
using namespace datamunge::plot;

int main() {
  const auto out_dir = std::filesystem::path("build/debug/examples/plot_output");
  std::filesystem::create_directories(out_dir);

  const std::vector<double> scatter_x = {0.3, 0.8, 1.1, 1.7, 2.0, 2.6, 3.0, 3.4};
  const std::vector<double> scatter_y = {1.4, 1.8, 2.5, 2.7, 3.1, 3.9, 4.2, 4.8};

  DenseMatrix<double> design(scatter_x.size(), 2, 0.0);
  for (std::size_t i = 0; i < scatter_x.size(); ++i) {
    design(i, 0) = 1.0;
    design(i, 1) = scatter_x[i];
  }
  const auto fit = linear_regression(design, scatter_y);
  const double intercept = fit.coefficients[0];
  const double slope = fit.coefficients[1];

  auto scatter = ScatterPlot::create();
  scatter.points(scatter_x,
                 scatter_y,
                 "samples",
                 {37, 99, 235},
                 6.0)
      .line({scatter_x.front(), scatter_x.back()},
            {intercept + slope * scatter_x.front(), intercept + slope * scatter_x.back()},
            "best fit",
            {220, 38, 38},
            3.0)
      .title("Scatter With Linear Fit")
      .x_label("feature x")
      .y_label("feature y")
      .size(960, 640)
      .background({250, 250, 252});

  auto line = LinePlot::create();
  line.line({0, 1, 2, 3, 4, 5},
            {0.2, 0.9, 1.4, 1.1, 1.9, 2.4},
            "trend",
            {220, 38, 38},
            3.0)
      .title("Line Plot")
      .x_label("step")
      .y_label("value");

  auto bars = BarChart::create();
  bars.bars({1, 2, 3, 4, 5},
            {4, 7, 3, 9, 6},
            "counts",
            {22, 163, 74},
            0.7)
      .title("Bar Chart")
      .x_label("bucket")
      .y_label("count");

  scatter.save((out_dir / "scatter.svg").string());
  scatter.save((out_dir / "scatter.pdf").string());
  scatter.save((out_dir / "scatter.png").string());
  line.save((out_dir / "line.svg").string());
  line.save((out_dir / "line.png").string());
  bars.save((out_dir / "bars.svg").string());
  bars.save((out_dir / "bars.pdf").string());

  scatter.show("datamunge-scatter");
  std::cout << "Wrote plots to " << out_dir << "\n";
  std::cout << std::fixed << std::setprecision(4)
            << "Best fit: y = " << intercept << " + " << slope << "x"
            << " (R^2 = " << fit.r_squared << ")\n";
  std::cout << "Closed scatter plot window\n";

  return 0;
}
