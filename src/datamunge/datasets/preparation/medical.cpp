/// @file medical.cpp
/// @brief Native medical tensor preparation from verified source metadata.
#ifdef DATAMUNGE_ENABLE_DATA_ASSETS
#include <dm/dm.hpp>
#include <fstream>
#include <iostream>
#include <map>
#include <set>

#include "preparation.hpp"
int datamunge::datasets::detail::prepare_medical(int argc, char** argv) {
  try {
    namespace n = dm::nn;
    if (argc != 2)
      return 1;
    n::set_num_threads(2);
    const auto                       root = std::filesystem::path(argv[1]);
    std::map<std::string, n::Tensor> source;
    for (const auto& [key, value] : n::load_npz((root / "nodulemnist3d.npz").string()))
      source[key] = value;
    n::NamedTensors            arrays;
    std::set<std::vector<int>> used;
    std::ofstream              metadata(root / "selected_rows.tsv");
    int                        offset = 0;
    for (const std::string split : {"train", "val", "test"}) {
      const auto images = source.at(split + "_images"),
                 labels = source.at(split + "_labels").reshape({-1}).to(n::DType::Int32);
      const auto                classes = labels.to_vector<int>();
      const int                 needed  = split == "val" ? 8 : 32;
      std::array<int, 2>        count{};
      std::vector<std::int64_t> rows, ids;
      for (int row = 0; row < labels.numel(); ++row) {
        const int label = classes[row];
        if (label < 0 || label >= 2)
          throw std::invalid_argument("binary original CT labels required");
        if (count[label] >= needed)
          continue;
        const auto bytes = n::select(images, 0, row).to_vector<int>();
        if (!used.insert(bytes).second)
          continue;
        rows.push_back(row);
        ids.push_back(offset + row);
        ++count[label];
        metadata << split << '\t' << row << '\t' << offset + row << '\t' << label << '\n';
        if (count[0] == needed && count[1] == needed)
          break;
      }
      if (count[0] != needed || count[1] != needed)
        throw std::invalid_argument("insufficient unique original CT subset");
      const std::string name = split == "train" ? "fit" : split == "val" ? "validation" : "test";
      arrays.push_back(
          {name + "_voxels", n::take_rows(images, rows).reshape({2 * needed, 28, 28, 28, 1})});
      arrays.push_back({name + "_labels", n::take_rows(labels, rows)});
      arrays.push_back({name + "_source_ids", n::from_vector(ids).to(n::DType::Int32)});
      std::cout << name << ':' << 2 * needed << "original cubes,source offset" << offset << '\n';
      offset += labels.numel();
    }
    if (offset != 1633)
      throw std::invalid_argument("pinned official source split lengths required");
    n::save_npz((root / "medical_volume.npz").string(), arrays);
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}

#endif // DATAMUNGE_ENABLE_DATA_ASSETS
