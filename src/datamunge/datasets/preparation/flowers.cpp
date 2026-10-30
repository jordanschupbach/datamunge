#ifdef DATAMUNGE_ENABLE_DATA_ASSETS
#include "preparation.hpp"
/// @file flowers.cpp
/// @brief Metadata-only flower row ingress delegates all pixel preprocessing to tensor APIs.
#include <dm/dm.hpp>
#include <fstream>
#include <iostream>
#include <map>

/// @brief Build the prepared archive from checksummed official split-row metadata.
/// @param argc Exactly two arguments are required.
/// @param argv Program name and source/cache directory.
/// @return Zero after validating and saving all three balanced tensor subsets.
int datamunge::datasets::detail::prepare_flowers(int argc, char** argv) {
  try {
    if (argc != 2)
      throw std::invalid_argument("expected official Flowers102 cache directory");
    namespace n                      = dm::nn;
    const std::filesystem::path root = argv[1];
    std::ifstream               rows(root / "prepared_rows.tsv");
    if (!rows)
      throw std::runtime_error("prepared row metadata missing");
    std::map<std::string, std::vector<int>> ids, labels;
    std::string                             split;
    int                                     id, label;
    while (rows >> split >> id >> label) {
      if (split != "fit" && split != "validation" && split != "test")
        throw std::invalid_argument("unknown official split metadata");
      ids[split].push_back(id);
      labels[split].push_back(label);
    }
    if (!rows.eof())
      throw std::runtime_error("invalid row metadata stream");
    n::NamedTensors arrays;
    for (const std::string name : {"fit", "validation", "test"}) {
      const auto data = n::models::ReadFlowerImageRows(root, ids.at(name), labels.at(name));
      arrays.push_back({name + "_rgb", data.rgb});
      arrays.push_back({name + "_labels", data.labels});
      arrays.push_back({name + "_ids", data.ids});
    }
    n::save_npz((root / "flowers.npz").string(), arrays);
    const auto data = n::models::LoadFlowers(root);
    std::cout << "Validated48/48/48 officially disjoint flower images after tensor resizing.\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}

#endif // DATAMUNGE_ENABLE_DATA_ASSETS
