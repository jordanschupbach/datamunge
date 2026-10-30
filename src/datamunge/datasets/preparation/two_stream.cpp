/// @file two_stream.cpp
/// @brief Native two_stream tensor preparation from verified source metadata.
#ifdef DATAMUNGE_ENABLE_DATA_ASSETS
#include <datamunge/image/read_image.hpp>
#include <dm/dm.hpp>
#include <fstream>
#include <map>
#include <print>

#include "preparation.hpp"
int datamunge::datasets::detail::prepare_two_stream(int argc, char** argv) {
  try {
    namespace n = dm::nn;
    namespace m = n::models;
    n::set_num_threads(2);
    if (argc != 2)
      return 1;
    const auto    root = std::filesystem::path(argv[1]);
    std::ifstream input(root / "clip_rows.tsv");
    std::string   split, path;
    int           label, person, video, segment;
    std::map<std::string, std::map<std::string, std::vector<n::Tensor>>> parts;
    n::NamedTensors                                                      audit;
    int                                                                  count = 0;
    while (input >> split >> label >> person >> video >> segment) {
      std::vector<n::Tensor> frames;
      for (int i = 0; i < 11; ++i) {
        input >> path;
        const auto rgb =
            n::from_image(dm::image::read_image((root / path).string())).to(n::DType::Float32);
        frames.push_back(n::resize_bilinear(n::expand_dims(rgb, 0), {96, 96}) / 255.);
      }
      const auto rgb  = n::concat(frames, 0);
      const auto gray = n::sum(
          rgb * n::from_vector(std::vector<float>{.299f, .587f, .114f}, {1, 1, 1, 3}), {3}, true);
      const auto derivatives =
          m::TwoStreamFlowDerivatives(n::slice_axis(gray, 0, 0, 10), n::slice_axis(gray, 0, 1, 11));
      n::ParameterStore displacement;
      const auto        u = displacement.get("u", {10, 96, 96, 1}, n::init::zeros()),
                 v        = displacement.get("v", {10, 96, 96, 1}, n::init::zeros());
      n::OptimizerConfig options;
      options.kind = n::OptimizerKind::Sgd;
      n::Optimizer sgd(displacement, .2, options);
      const auto   initial = m::TwoStreamFlowEnergy(u, v, derivatives).item();
      for (int step = 0; step < 400; ++step)
        sgd.minimize(m::TwoStreamFlowEnergy(u, v, derivatives, .05));
      const auto final = m::TwoStreamFlowEnergy(u, v, derivatives).item();
      if (!std::isfinite(final) || final >= initial)
        throw std::runtime_error("actual flow energy must decrease");
      const auto flow = n::concat({u.detach(), v.detach()}, 3).reshape({1, 10, 96, 96, 2});
      parts[split]["rgb"].push_back(n::slice_axis(rgb, 0, 5, 6));
      parts[split]["flow"].push_back(flow);
      for (const auto& [key, value] : std::vector<std::pair<std::string, int>>{
               {"label", label}, {"person", person}, {"video_id", video}, {"segment", segment}})
        parts[split][key].push_back(n::full({1}, value, n::DType::Int32));
      const auto prefix = std::to_string(video) + "/" + std::to_string(segment) + "/";
      audit.push_back({prefix + "rgb", rgb});
      audit.push_back({prefix + "energy", n::from_vector(std::vector<double>{initial, final})});
      std::print(
          "Original clip video{},segment{}:quadratic flow energy {:.9g}->{:.9g}.\n",
          video,
          segment,
          initial,
          final);
      ++count;
    }
    n::NamedTensors arrays;
    for (const auto& [split, fields] : parts)
      for (const auto& [key, values] : fields) {
        const auto value = n::concat(values, 0);
        arrays.push_back({split + "_" + key, value});
        if (split == "test" && key == "rgb")
          n::save_image_grid(value, (root / "test_contact.png").string(), 3, 0, 1, 2, 0, 2);
      }
    n::save_npz((root / "two_stream.npz").string(), arrays);
    n::save_npz((root / "flow_audit.npz").string(), audit);
    std::print(
        "Prepared{}actual RGB/ten estimated dense flow pairs;all400-step tensorSGD "
        "fits explicit.\n",
        count);
    return 0;
  } catch (const std::exception& e) {
    std::print(stderr, "{}\n", e.what());
    return 1;
  }
}

#endif // DATAMUNGE_ENABLE_DATA_ASSETS
