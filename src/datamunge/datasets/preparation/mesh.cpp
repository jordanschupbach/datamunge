/// @file mesh.cpp
/// @brief Native mesh tensor preparation from verified source metadata.
#ifdef DATAMUNGE_ENABLE_DATA_ASSETS
#include <dm/dm.hpp>
#include <fstream>
#include <iostream>
#include <map>

#include "preparation.hpp"
int datamunge::datasets::detail::prepare_mesh(int argc, char** argv) {
  try {
    namespace n = dm::nn;
    if (argc != 2)
      return 1;
    n::set_num_threads(2);
    const auto    root = std::filesystem::path(argv[1]);
    std::ifstream metadata(root / "mesh_rows.tsv");
    std::string   split, file;
    int           source, label;
    std::map<std::string, std::map<std::string, std::vector<n::Tensor>>> parts;
    n::NamedTensors                                                      audit;
    while (metadata >> split >> source >> label >> file) {
      std::ifstream input(root / file);
      int           nv, nf;
      input >> nv >> nf;
      std::vector<double>       values(nv * 3);
      std::vector<std::int64_t> topology(nf * 3);
      for (auto& v : values)
        input >> v;
      for (auto& i : topology)
        input >> i;
      const auto vertices  = n::from_vector(values, {nv, 3});
      const auto faces     = n::from_vector(topology, {nf, 3});
      const auto triangles = n::gather(vertices, faces, 0);
      const auto a         = n::select(triangles, 1, 1) - n::select(triangles, 1, 0),
                 b         = n::select(triangles, 1, 2) - n::select(triangles, 1, 0);
      const auto cross     = n::stack(
          {n::select(a, 1, 1) * n::select(b, 1, 2) - n::select(a, 1, 2) * n::select(b, 1, 1),
               n::select(a, 1, 2) * n::select(b, 1, 0) - n::select(a, 1, 0) * n::select(b, 1, 2),
               n::select(a, 1, 0) * n::select(b, 1, 1) - n::select(a, 1, 1) * n::select(b, 1, 0)},
          1);
      const auto area = n::sqrt(n::sum(n::square(cross), {1})) * .5;
      if (n::sum(area).item() <= 0 || !n::all(n::is_finite(area)).item<bool>())
        throw std::invalid_argument("finite nondegenerate original mesh required");
      n::Generator gen(941 + source);
      const auto   selected =
          n::categorical(
              n::where(
                  n::greater(area, 0.),
                  n::log(n::maximum(area, n::full(area.shape(), 1e-300, n::DType::Float64))),
                  -1e30)
                  .reshape({1, nf}),
              1024,
              gen)
              .reshape({1024});
      const auto sampled = n::gather(triangles, selected, 0);
      const auto draws   = n::uniform({1024, 2}, 0, 1, gen, n::DType::Float64);
      const auto u = n::sqrt(n::slice_axis(draws, 1, 0, 1)), v = n::slice_axis(draws, 1, 1, 2);
      const auto raw = n::select(sampled, 1, 0) * (1. - u) +
                       n::select(sampled, 1, 1) * u * (1. - v) + n::select(sampled, 1, 2) * u * v;
      const auto center   = n::mean(raw, {0});
      const auto centered = raw - center;
      const auto radius   = n::max(n::sqrt(n::sum(n::square(centered), {1})));
      const auto xyz      = (centered / radius).to(n::DType::Float32);
      parts[split]["xyz"].push_back(n::expand_dims(xyz, 0));
      parts[split]["label"].push_back(n::full({1}, label, n::DType::Int32));
      parts[split]["source_id"].push_back(n::full({1}, source, n::DType::Int32));
      const auto prefix = std::to_string(source) + "/";
      audit.push_back({prefix + "triangles", selected});
      audit.push_back({prefix + "uniforms", draws});
      audit.push_back({prefix + "center", center});
      audit.push_back({prefix + "radius", radius});
    }
    n::NamedTensors arrays;
    for (const auto& [name, fields] : parts)
      for (const auto& [key, value] : fields)
        arrays.push_back({name + "_" + key, n::concat(value, 0)});
    n::save_npz((root / "point_clouds.npz").string(), arrays);
    n::save_npz((root / "point_sampling.npz").string(), audit);
    std::cout << "Tensor area/barycentric sampling and unit-ball normalization "
                 "saved42original surfaces.\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}

#endif // DATAMUNGE_ENABLE_DATA_ASSETS
