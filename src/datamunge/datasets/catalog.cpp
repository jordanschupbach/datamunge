/// @file catalog.cpp
/// @brief The built-in and downloaded dataset accessors, load(), catalog() and to_dataframe().

#include <datamunge/datasets/datasets.hpp>

#include "embedded_csv.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <utility>
#include <string>
#include <vector>

namespace datamunge::datasets {

dstruct::DataFrame wine() { return parse_csv(detail::wine_csv()); }
dstruct::DataFrame titanic() { return parse_csv(detail::titanic_csv()); }
dstruct::DataFrame eight_schools() { return parse_csv(detail::eight_schools_csv()); }
dstruct::DataFrame orthodont() { return parse_csv(detail::orthodont_csv()); }
dstruct::DataFrame faithful() { return parse_csv(detail::faithful_csv()); }
dstruct::DataFrame misra1a() { return parse_csv(detail::misra1a_csv()); }
dstruct::DataFrame mtcars() { return parse_csv(detail::mtcars_csv()); }
dstruct::DataFrame airquality() { return parse_csv(detail::airquality_csv()); }
dstruct::DataFrame air_passengers() { return parse_csv(detail::air_passengers_csv()); }
dstruct::DataFrame breast_cancer() { return parse_csv(detail::breast_cancer_csv()); }
dstruct::DataFrame diabetes() { return parse_csv(detail::diabetes_csv()); }
dstruct::DataFrame digits() { return parse_csv(detail::digits_csv()); }
dstruct::DataFrame volcano() { return parse_csv(detail::volcano_csv()); }

dstruct::DataFrame diamonds() { return read_csv(fetch("diamonds")); }
dstruct::DataFrame california_housing() { return read_csv(fetch("california_housing")); }

dstruct::DataFrame sudoku() {
  const std::filesystem::path dir = fetch("sudoku");
  std::vector<std::string> collection, puzzles;
  std::vector<double> ids, givens;
  for (const auto& [name, expected] :
       std::vector<std::pair<std::string, int>>{{"easy50", 50}, {"top95", 95}, {"hardest", 11}}) {
    std::ifstream in(dir / (name + ".txt"));
    if (!in) throw std::runtime_error("cannot read sudoku collection " + (dir / (name + ".txt")).string());
    // Grids are 81 cells on one line or nine lines of nine; other lines are separators.
    std::string line, cells;
    int count = 0;
    while (std::getline(in, line)) {
      while (!line.empty() && (line.back() == '\r' || line.back() == ' ')) line.pop_back();
      if (line.empty() || line.find_first_not_of("0123456789.") != std::string::npos) continue;
      cells += line;
      while (cells.size() >= 81) {
        std::string grid = cells.substr(0, 81);
        cells.erase(0, 81);
        std::replace(grid.begin(), grid.end(), '0', '.');
        collection.push_back(name);
        ids.push_back(++count);
        givens.push_back(static_cast<double>(81 - std::count(grid.begin(), grid.end(), '.')));
        puzzles.push_back(std::move(grid));
      }
    }
    if (count != expected || !cells.empty())
      throw std::runtime_error("sudoku collection " + name + " holds " + std::to_string(count) + " puzzles, expected " +
                               std::to_string(expected));
  }
  dstruct::DataFrame frame;
  frame.add_column("collection", collection);
  frame.add_column("id", ids);
  frame.add_column("puzzle", puzzles);
  frame.add_column("givens", givens);
  return frame;
}

/// @brief The catalog() table's rows.
namespace {

/// @brief One catalog() row.
struct Entry {
  const char* name;         ///< Dataset name, as load()/ImageLoader take it.
  double rows;              ///< Rows (image datasets: train + test images).
  double columns;           ///< Columns (image datasets: values per image).
  const char* storage;      ///< "built-in", "download" or "stream".
  double size_mb;           ///< Approximate download size.
  const char* description;  ///< One-line summary.
};

/// @brief Every dataset, sorted by name.
const std::vector<Entry> kEntries = {
    {"air_passengers", 144, 3, "built-in", 0, "Monthly airline passengers, 1949-1960 (Box & Jenkins, R)"},
    {"airquality", 153, 6, "built-in", 0, "New York daily air quality, May-Sep 1973 (R)"},
    {"breast_cancer", 569, 31, "built-in", 0, "Wisconsin diagnostic breast cancer, malignant vs benign"},
    {"california_housing", 20640, 10, "download", 1.4, "1990 California census housing by block group"},
    {"cifar10", 60000, 3072, "stream", 163, "32x32 color images in 10 classes"},
    {"cifar100", 60000, 3072, "stream", 161, "32x32 color images in 100 fine / 20 coarse classes"},
    {"diabetes", 442, 11, "built-in", 0, "Diabetes progression after one year (Efron et al. 2004)"},
    {"diamonds", 53940, 10, "download", 2.7, "Price, carat, cut, color and clarity of diamonds (ggplot2)"},
    {"digits", 1797, 65, "built-in", 0, "8x8 handwritten digit images (UCI)"},
    {"eight_schools", 8, 3, "built-in", 0, "SAT coaching effects in 8 schools (Rubin 1981)"},
    {"faithful", 272, 2, "built-in", 0, "Old Faithful geyser eruption and waiting times (R)"},
    {"fashion_mnist", 70000, 784, "stream", 30, "28x28 grayscale clothing images in 10 classes"},
    {"iris", 150, 5, "built-in", 0, "Fisher's iris flower measurements, 3 species (R)"},
    {"mnist", 70000, 784, "stream", 11, "28x28 grayscale handwritten digits"},
    {"misra1a", 14, 2, "built-in", 0, "NIST Misra1a monomolecular adsorption, observed pressure and volume"},
    {"mtcars", 32, 12, "built-in", 0, "Motor Trend 1974 road tests of 32 cars (R)"},
    {"orthodont", 108, 4, "built-in", 0, "Orthodontic growth of 27 children, ages 8-14 (Potthoff & Roy 1964)"},
    {"penguins", 344, 8, "built-in", 0, "Palmer Archipelago penguin measurements, 3 species"},
    {"sudoku", 156, 4, "download", 0.014, "Norvig's Sudoku puzzles: easy, hard and hardest"},
    {"titanic", 891, 12, "built-in", 0, "Titanic passengers and survival (Kaggle training set)"},
    {"tooth_growth", 60, 3, "built-in", 0, "Guinea pig tooth growth vs vitamin C dose (R)"},
    {"volcano", 5307, 3, "built-in", 0, "Maunga Whau (Mt Eden) elevations on a 10 m grid (R)"},
    {"wine", 178, 14, "built-in", 0, "Chemical analysis of wines from 3 cultivars (UCI)"},
};

} // namespace

dstruct::DataFrame load(const std::string& name) {
  if (name == "air_passengers") return air_passengers();
  if (name == "airquality") return airquality();
  if (name == "breast_cancer") return breast_cancer();
  if (name == "california_housing") return california_housing();
  if (name == "diabetes") return diabetes();
  if (name == "diamonds") return diamonds();
  if (name == "digits") return digits();
  if (name == "eight_schools") return eight_schools();
  if (name == "faithful") return faithful();
  if (name == "iris") return iris();
  if (name == "misra1a") return misra1a();
  if (name == "mtcars") return mtcars();
  if (name == "orthodont") return orthodont();
  if (name == "penguins") return penguins();
  if (name == "sudoku") return sudoku();
  if (name == "titanic") return titanic();
  if (name == "tooth_growth") return tooth_growth();
  if (name == "volcano") return volcano();
  if (name == "wine") return wine();
  if (name == "mnist" || name == "fashion_mnist" || name == "cifar10" || name == "cifar100")
    throw std::invalid_argument("\"" + name + "\" is an image dataset; read it in batches with ImageLoader(\"" +
                                name + "\")");
  throw std::invalid_argument("unknown dataset \"" + name + "\"; see datasets::catalog() for the list");
}

dstruct::DataFrame to_dataframe(const ImageBatch& batch) {
  const int n = batch.size();
  const int f = batch.features();
  dstruct::DataFrame frame;
  std::vector<double> column(static_cast<std::size_t>(n));
  for (int j = 0; j < f; ++j) {
    for (int i = 0; i < n; ++i)
      column[static_cast<std::size_t>(i)] = batch.pixels[static_cast<std::size_t>(i) * f + j];
    frame.add_column("pixel_" + std::to_string(j), column);
  }
  frame.add_column("label", std::vector<double>(batch.labels.begin(), batch.labels.end()));
  frame.add_column("index", std::vector<double>(batch.indices.begin(), batch.indices.end()));
  return frame;
}

dstruct::DataFrame catalog() {
  std::vector<std::string> names, storage, descriptions;
  std::vector<double> rows, columns, sizes;
  for (const auto& e : kEntries) {
    names.emplace_back(e.name);
    rows.push_back(e.rows);
    columns.push_back(e.columns);
    storage.emplace_back(e.storage);
    sizes.push_back(e.size_mb);
    descriptions.emplace_back(e.description);
  }
  dstruct::DataFrame frame;
  frame.add_column("name", names);
  frame.add_column("rows", rows);
  frame.add_column("columns", columns);
  frame.add_column("storage", storage);
  frame.add_column("size_mb", sizes);
  frame.add_column("description", descriptions);
  return frame;
}

} // namespace datamunge::datasets
