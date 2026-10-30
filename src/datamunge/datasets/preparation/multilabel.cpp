#ifdef DATAMUNGE_ENABLE_DATA_ASSETS
#include "preparation.hpp"
/// @file multilabel.cpp
/// @brief Tensor COCO image/presence preprocessing from discrete checked source metadata.
#include <dm/dm.hpp>
#include <fstream>
#include <iostream>
#include <map>

/// @brief Prepare observed RGB and independent presence targets without host pixel arithmetic.
/// @param argc Exactly two arguments are required.
/// @param argv Program name and source/cache directory.
/// @return Zero after saving and validating all128 unique source rows.
int datamunge::datasets::detail::prepare_multilabel(int argc, char** argv) {
  try {
    if (argc != 2)
      throw std::invalid_argument("expected COCO128 source/cache directory");
    namespace n                      = dm::nn;
    const std::filesystem::path root = argv[1];
    std::ifstream               rows(root / "prepared_rows.tsv");
    if (!rows)
      throw std::runtime_error("prepared row metadata missing");
    std::map<std::string, std::vector<std::int32_t>>     ids;
    std::map<std::string, std::vector<std::vector<int>>> categories;
    std::string                                          split;
    std::int32_t                                         id;
    int                                                  count;
    while (rows >> split >> id >> count) {
      if ((split != "fit" && split != "validation" && split != "test") || count < 0 || count > 80)
        throw std::invalid_argument("invalid split/category-count metadata");
      std::vector<int> present;
      for (int i = 0; i < count; ++i) {
        int category;
        if (!(rows >> category))
          throw std::invalid_argument("truncated category metadata");
        present.push_back(category);
      }
      ids[split].push_back(id);
      categories[split].push_back(present);
    }
    if (!rows.eof())
      throw std::invalid_argument("invalid row metadata stream");
    n::NamedTensors arrays;
    for (const std::string name : {"fit", "validation", "test"}) {
      const auto data = n::models::ReadMultilabelImages(
          root / "coco128/images/train2017", ids.at(name), categories.at(name));
      arrays.push_back({name + "_rgb", data.rgb});
      arrays.push_back({name + "_labels", data.labels});
      arrays.push_back({name + "_ids", data.ids});
    }
    n::save_npz((root / "multilabel.npz").string(), arrays);
    n::models::LoadMultilabel(root);
    std::cout << "Validated80/24/24 unique source images and tensor OR-presence targets.\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}

#endif // DATAMUNGE_ENABLE_DATA_ASSETS
