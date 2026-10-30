// CMD: just run-plot
// Plot the built-in Iris dataset, with one scatter series per species.
#include <datamunge/datasets/datasets.hpp>
#include <datamunge/plot/plot.hpp>

#include <array>
#include <exception>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

using datamunge::graphics::RGB;
using datamunge::graphics::RPlot;

int main() {
  const auto iris = datamunge::datasets::iris();
  const auto& petal_length = iris.numeric_column("Petal.Length");
  const auto& petal_width = iris.numeric_column("Petal.Width");
  const auto& species = iris.string_column("Species");
  const std::array<std::string, 3> species_names = {"setosa", "versicolor", "virginica"};
  const std::array<RGB, 3> colors = {RGB{37, 99, 235}, RGB{22, 163, 74}, RGB{220, 38, 38}};

  auto plot = RPlot::create();
  plot.title("Iris Petal Dimensions")
      .x_label("Petal length (cm)")
      .y_label("Petal width (cm)")
      .background({250, 250, 252});

  std::cout << "iris: " << iris.nrows() << " rows x " << iris.ncols() << " cols\n";
  for (std::size_t group = 0; group < species_names.size(); ++group) {
    std::vector<double> x;
    std::vector<double> y;
    for (std::size_t row = 0; row < iris.nrows(); ++row) {
      if (species.at(row) == species_names[group]) {
        x.push_back(petal_length.at(row));
        y.push_back(petal_width.at(row));
      }
    }
    std::cout << species_names[group] << ": " << x.size() << " flowers\n";
    plot.points(x, y, species_names[group], colors[group], 6.0);
  }

  const auto out_dir = std::filesystem::path("build/debug/examples");
  std::filesystem::create_directories(out_dir);
  const auto svg_path = out_dir / "iris_scatter.svg";
  plot.save_svg(svg_path.string());

  try {
    plot.show("datamunge-plot-ex");
    std::cout << "viewer = shown\n";
  } catch (const std::exception& ex) {
    std::cout << "viewer = skipped (" << ex.what() << ")\n";
  }

  std::cout << "svg = " << svg_path.string() << "\n";
  return 0;
}
