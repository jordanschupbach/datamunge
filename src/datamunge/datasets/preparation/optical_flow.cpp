#ifdef DATAMUNGE_ENABLE_DATA_ASSETS
#include "preparation.hpp"
/// @file optical_flow.cpp
/// @brief Tensor original Middlebury ingress and predetermined family-disjoint64px cropping.
#include <dm/dm.hpp>
#include <fstream>
#include <iostream>
#include <map>

/// @brief Prepare64 aligned observed/GT crops and source-coordinate metadata.
/// @param argc Exactly two arguments are required.
/// @param argv Program name and prepared raw-scene/cache directory.
/// @return Zero after validating32/8/24 image-family-isolated crop tensors.
int datamunge::datasets::detail::prepare_optical_flow(int argc, char** argv) {
  try {
    if (argc != 2)
      throw std::invalid_argument("expected Middlebury scene/cache directory");
    namespace n = dm::nn;
    namespace m = n::models;
    n::set_num_threads(2);
    const std::filesystem::path root = argv[1];
    std::ifstream               input(root / "scene_rows.tsv");
    std::ofstream               coordinates(root / "crop_rows.tsv");
    if (!input || !coordinates)
      throw std::runtime_error("scene/crop metadata stream unavailable");
    std::map<std::string, std::vector<m::FlowData>> parts;
    std::string                                     split, name;
    std::int32_t                                    scene, family;
    while (input >> split >> name >> scene >> family) {
      if (split != "fit" && split != "validation" && split != "test")
        throw std::invalid_argument("invalid flow scene split");
      const auto raw     = m::ReadMiddleburyScene(root / "scenes" / name, scene, family);
      const auto origins = m::FlowCropOrigins(raw).to_vector<std::int64_t>();
      parts[split].push_back(m::FlowSceneCrops(raw));
      for (int i = 0; i < 8; ++i)
        coordinates << split << '\t' << name << '\t' << scene << '\t' << family << '\t'
                    << origins[2 * i] << '\t' << origins[2 * i + 1] << '\t' << raw.first.size(1)
                    << '\t' << raw.first.size(2) << '\n';
    }
    if (!input.eof())
      throw std::invalid_argument("truncated flow scene metadata");
    n::NamedTensors arrays;
    for (const std::string key : {"fit", "validation", "test"}) {
      std::vector<n::Tensor> first, second, flow, valid, scenes, families;
      for (const auto& d : parts.at(key)) {
        first.push_back(d.first);
        second.push_back(d.second);
        flow.push_back(d.flow);
        valid.push_back(d.valid);
        scenes.push_back(d.scene_ids);
        families.push_back(d.family_ids);
      }
      arrays.push_back({key + "_first", n::concat(first, 0)});
      arrays.push_back({key + "_second", n::concat(second, 0)});
      arrays.push_back({key + "_flow", n::concat(flow, 0)});
      arrays.push_back({key + "_valid", n::concat(valid, 0)});
      arrays.push_back({key + "_scene_ids", n::concat(scenes, 0)});
      arrays.push_back({key + "_family_ids", n::concat(families, 0)});
    }
    n::save_npz((root / "pairs.npz").string(), arrays);
    m::LoadFlowPairs(root);
    std::cout << "Validated32/8/24 published-GT crops with disjoint scene families.\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}

#endif // DATAMUNGE_ENABLE_DATA_ASSETS
