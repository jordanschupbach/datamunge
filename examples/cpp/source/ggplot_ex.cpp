// Demonstrates datamunge::plot::GGPlot -- the ggplot2-style grammar-of-graphics library.
#include <datamunge/datamunge.hpp>
#include <datamunge/datasets/datasets.hpp>
#include <datamunge/plot/ggplot.hpp>

#include <filesystem>
#include <iostream>

using datamunge::plot::Aes;
using datamunge::plot::GGPlot;
using datamunge::plot::RGB;

int main() {
  const auto out_dir = std::filesystem::path("build/debug/examples");
  std::filesystem::create_directories(out_dir);

  const auto iris = datamunge::datasets::iris();

  // ggplot(iris, aes(x = Sepal.Length, y = Sepal.Width, color = Species)) + geom_point()
  auto scatter = GGPlot::create(iris, Aes{"Sepal.Length", "Sepal.Width", "Species", "", ""});
  scatter.geom_point();
  scatter.labs("Iris Sepal Dimensions", "Sepal Length", "Sepal Width");
  scatter.theme_minimal();
  scatter.save_svg((out_dir / "ggplot_point.svg").string());

  // + geom_smooth(): an lm() fit line per group, reusing stats::LM internally.
  auto smooth = GGPlot::create(iris, Aes{"Sepal.Length", "Petal.Length", "", "", ""});
  smooth.geom_point(RGB{156, 163, 175}, 2.5);
  smooth.geom_smooth();
  smooth.labs("Petal Length vs Sepal Length With a Linear Fit", "Sepal Length", "Petal Length");
  smooth.save_svg((out_dir / "ggplot_smooth.svg").string());

  // geom_bar(): counts a discrete column (stat = "count").
  auto bar = GGPlot::create(iris, Aes{"Species", "", "", "", ""});
  bar.geom_bar();
  bar.labs("Observations per Species", "Species", "Count");
  bar.theme_bw();
  bar.save_svg((out_dir / "ggplot_bar.svg").string());

  // geom_boxplot(): grouped by a discrete x column.
  auto box = GGPlot::create(iris, Aes{"Species", "Petal.Width", "", "", ""});
  box.geom_boxplot();
  box.labs("Petal Width by Species", "Species", "Petal Width");
  box.save_svg((out_dir / "ggplot_boxplot.svg").string());

  // geom_histogram() + geom_density(): distribution of a single numeric column.
  auto hist = GGPlot::create(iris, Aes{"Sepal.Length", "", "", "", ""});
  hist.geom_histogram(20);
  hist.labs("Distribution of Sepal Length", "Sepal Length", "Count");
  hist.save_svg((out_dir / "ggplot_histogram.svg").string());

  auto density = GGPlot::create(iris, Aes{"Sepal.Length", "", "", "", ""});
  density.geom_density();
  density.labs("Density of Sepal Length", "Sepal Length", "Density");
  density.save_svg((out_dir / "ggplot_density.svg").string());

  // facet_wrap(): one panel per Species, composed via RLayout under the hood.
  auto faceted = GGPlot::create(iris, Aes{"Petal.Length", "Petal.Width", "", "", ""});
  faceted.geom_point();
  faceted.facet_wrap("Species");
  faceted.labs("Petal Dimensions", "Petal Length", "Petal Width");
  faceted.save_svg((out_dir / "ggplot_facet.svg").string());

  // scale_color_manual(): override the default discrete palette.
  auto custom_colors = GGPlot::create(iris, Aes{"Sepal.Length", "Sepal.Width", "Species", "", ""});
  custom_colors.geom_point();
  custom_colors.scale_color_manual({RGB{16, 185, 129}, RGB{245, 158, 11}, RGB{99, 102, 241}});
  custom_colors.theme_classic();
  custom_colors.save_svg((out_dir / "ggplot_custom_colors.svg").string());

  std::cout << "Wrote 7 SVGs to " << out_dir.string() << "\n";
  return 0;
}
