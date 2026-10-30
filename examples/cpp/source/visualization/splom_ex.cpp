// Demonstrates datamunge::graphics::Splom -- a generalized pairs plot: densities / stacked bars on the
// diagonal, smooths with prediction bands above it and bivariate density contours below it (numeric
// pairs), violins (numeric vs string) and chi-squared-shaded mosaics (string vs string).
#include <datamunge/datamunge.hpp>
#include <datamunge/datasets/datasets.hpp>
#include <datamunge/plot/splom.hpp>

#include <filesystem>
#include <iostream>

using datamunge::graphics::Splom;

int main() {
  const auto out_dir = std::filesystem::path("build/debug/examples");
  std::filesystem::create_directories(out_dir);

  const auto iris = datamunge::datasets::iris();

  // Every numeric column plus Species: violins in the Species row and column.
  auto plain = Splom::create(iris);
  plain.title("Iris measurements");
  plain.save_svg((out_dir / "splom_iris.svg").string());

  // ggpairs(iris, aes(colour = Species)): every numeric panel split by species
  auto grouped = Splom::create(iris, {"Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"});
  grouped.title("Iris by species").group_by("Species");
  grouped.save_svg((out_dir / "splom_iris_species.svg").string());

  std::cout << "variables:";
  for (const auto& name : grouped.variables()) std::cout << ' ' << name;
  std::cout << "\ngroups:";
  for (const auto& name : grouped.groups()) std::cout << ' ' << name;
  std::cout << "\nwrote " << (out_dir / "splom_iris.svg") << " and " << (out_dir / "splom_iris_species.svg")
            << "\n";
  // dm::splom(iris, {}, "Species") would additionally open the matrix in the interactive viewer.
  return 0;
}
