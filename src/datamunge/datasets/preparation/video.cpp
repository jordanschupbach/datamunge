/// @file video.cpp
/// @brief Native video tensor preparation from verified source metadata.
#ifdef DATAMUNGE_ENABLE_DATA_ASSETS
#include <datamunge/image/read_image.hpp>
#include <dm/dm.hpp>
#include <fstream>
#include <iostream>
#include <map>

#include "preparation.hpp"
int datamunge::datasets::detail::prepare_video(int argc, char** argv) {
  try {
    namespace n = dm::nn;
    n::set_num_threads(2);
    if (argc != 2)
      return 1;
    const auto    root = std::filesystem::path(argv[1]);
    std::ifstream input(root / "clip_rows.tsv");
    std::string   split, path;
    int           label, person, video, segment;
    std::map<std::string, std::map<std::string, std::vector<n::Tensor>>> parts;
    while (input >> split >> label >> person >> video >> segment) {
      std::vector<n::Tensor> frames;
      for (int i = 0; i < 16; ++i) {
        input >> path;
        const auto rgb =
            n::from_image(dm::image::read_image((root / path).string())).to(n::DType::Float32);
        frames.push_back(n::resize_bilinear(n::expand_dims(rgb, 0), {32, 32}) / 255.);
      }
      parts[split]["rgb"].push_back(n::expand_dims(n::concat(frames, 0), 0));
      for (const auto& [key, value] : std::vector<std::pair<std::string, int>>{
               {"label", label}, {"person", person}, {"video_id", video}, {"segment", segment}})
        parts[split][key].push_back(n::full({1}, value, n::DType::Int32));
    }
    n::NamedTensors arrays;
    for (const auto& [name, fields] : parts)
      for (const auto& [key, value] : fields)
        arrays.push_back({name + "_" + key, n::concat(value, 0)});
    n::save_npz((root / "action_video.npz").string(), arrays);
    std::cout << "Tensor RGB ingress/32x32bilinear/scaling saved for36original clips.\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}

#endif // DATAMUNGE_ENABLE_DATA_ASSETS
