#ifdef DATAMUNGE_ENABLE_DATA_ASSETS
#include "preparation.hpp"
/// @file depth.cpp
/// @brief HDF5 byte/metric ingress followed by canonical tensor NYU sensor sampling.
#include <H5Cpp.h>

#include <cstdint>
#include <dm/dm.hpp>
#include <fstream>
#include <iostream>
#include <map>

/// @brief Private source ingress and tensor preparation helpers.
namespace {
/// @brief Decode original NYU scene-name references as metadata and retain each first frame.
/// @param file Open original NYU HDF5/Matlab archive.
/// @return Sorted mapping from canonical room name to first zero-based source frame.
std::map<std::string, int> Inventory(H5::H5File& file) {
  auto       scenes = file.openDataSet("scenes");
  const auto count  = scenes.getSpace().getSimpleExtentNpoints();
  if (count != 1449)
    throw std::runtime_error("NYU scene inventory must contain1449 aligned frames");
  std::vector<hobj_ref_t> refs(count);
  scenes.read(refs.data(), H5::PredType::STD_REF_OBJ);
  std::map<std::string, int> first;
  for (int i = 0; i < count; ++i) {
    H5::DataSet room(H5Rdereference2(file.getId(), H5P_DEFAULT, H5R_OBJECT, &refs[i]));
    const auto  size = room.getSpace().getSimpleExtentNpoints();
    std::vector<std::uint16_t> chars(size);
    room.read(chars.data(), H5::PredType::NATIVE_UINT16);
    std::string name;
    for (auto c : chars) {
      if (c > 127)
        throw std::runtime_error("non-ASCII original NYU room metadata");
      name += char(c);
    }
    first.try_emplace(name, i);
  }
  if (first.size() != 464)
    throw std::runtime_error("expected464 original canonical rooms");
  return first;
}
} // namespace

/// @brief Inventory source rooms or prepare selected56 original RGB/raw-sensor frames.
/// @param argc Program accepts source/cache directory and optional --inventory.
/// @param argv Program name,source/cache directory,optional inventory mode.
/// @return Zero after metadata inventory or validated32/8/16 tensor sensor preparation.
int datamunge::datasets::detail::prepare_depth(int argc, char** argv) {
  try {
    if (argc < 2 || argc > 3)
      throw std::invalid_argument("expected NYU cache directory and optional --inventory");
    namespace n = dm::nn;
    namespace m = n::models;
    n::set_num_threads(2);
    const std::filesystem::path root = argv[1];
    H5::H5File                  file((root / "nyu_depth_v2_labeled.mat").string(), H5F_ACC_RDONLY);
    const auto                  rooms = Inventory(file);
    if (argc == 3) {
      if (std::string(argv[2]) != "--inventory")
        throw std::invalid_argument("unknown NYU prepare option");
      std::ofstream out(root / "room_rows.tsv");
      for (const auto& [room, row] : rooms)
        out << room << '\t' << row << '\n';
      if (!out)
        throw std::runtime_error("NYU room inventory write failed");
      std::cout << "Decoded464 original room identities and first-frame metadata.\n";
      return 0;
    }
    auto images = file.openDataSet("images"), depth = file.openDataSet("rawDepths");
    auto rgb_space = images.getSpace(), depth_space = depth.getSpace();
    if (rgb_space.getSimpleExtentNdims() != 4 || depth_space.getSimpleExtentNdims() != 3)
      throw std::runtime_error("NYU images/rawDepths rank mismatch");
    hsize_t rgb_shape[4], depth_shape[3];
    rgb_space.getSimpleExtentDims(rgb_shape);
    depth_space.getSimpleExtentDims(depth_shape);
    if (rgb_space.getSimpleExtentNdims() != 4 || depth_space.getSimpleExtentNdims() != 3 ||
        std::vector<hsize_t>(rgb_shape, rgb_shape + 4) != std::vector<hsize_t>{1449, 3, 640, 480} ||
        std::vector<hsize_t>(depth_shape, depth_shape + 3) != std::vector<hsize_t>{1449, 640, 480})
      throw std::runtime_error("unexpected original Matlab/HDF5 sensor layouts");
    std::ifstream input(root / "selected_rows.tsv");
    if (!input)
      throw std::runtime_error("selected room metadata missing");
    std::map<std::string, std::vector<m::DepthData>> parts;
    std::string                                      split, room;
    std::int32_t                                     room_id, source_row;
    std::set<std::string>                            unique;
    while (input >> split >> room >> room_id >> source_row) {
      if ((split != "fit" && split != "validation" && split != "test") ||
          !unique.insert(room).second || rooms.at(room) != source_row)
        throw std::runtime_error("source room/frame metadata mismatch");
      const hsize_t rgb_start[] = {hsize_t(source_row), 0, 0, 0}, rgb_count[] = {1, 3, 640, 480},
                    depth_start[] = {hsize_t(source_row), 0, 0}, depth_count[] = {1, 640, 480};
      rgb_space.selectHyperslab(H5S_SELECT_SET, rgb_count, rgb_start);
      depth_space.selectHyperslab(H5S_SELECT_SET, depth_count, depth_start);
      const H5::DataSpace       rgb_memory(4, rgb_count), depth_memory(3, depth_count);
      std::vector<std::uint8_t> bytes(3 * 640 * 480);
      std::vector<float>        sensor(640 * 480);
      images.read(bytes.data(), H5::PredType::NATIVE_UINT8, rgb_memory, rgb_space);
      depth.read(sensor.data(), H5::PredType::NATIVE_FLOAT, depth_memory, depth_space);
      std::vector<std::int32_t> ingress(bytes.begin(), bytes.end());
      const auto rgb = n::transpose(n::from_vector(ingress, {3, 640, 480}), {2, 1, 0}),
                 raw = n::transpose(n::from_vector(sensor, {640, 480}), {1, 0});
      parts[split].push_back(m::PrepareDepthSensorFrame(rgb, raw, room_id, source_row));
    }
    if (!input.eof())
      throw std::runtime_error("truncated selected room metadata");
    n::NamedTensors arrays;
    for (const std::string name : {"fit", "validation", "test"}) {
      std::vector<n::Tensor> rgb, sensor, valid, ids, rows;
      for (const auto& d : parts.at(name)) {
        rgb.push_back(d.rgb);
        sensor.push_back(d.depth);
        valid.push_back(d.valid);
        ids.push_back(d.room_ids);
        rows.push_back(d.source_rows);
      }
      arrays.push_back({name + "_rgb", n::concat(rgb, 0)});
      arrays.push_back({name + "_depth", n::concat(sensor, 0)});
      arrays.push_back({name + "_valid", n::concat(valid, 0)});
      arrays.push_back({name + "_room_ids", n::concat(ids, 0)});
      arrays.push_back({name + "_source_rows", n::concat(rows, 0)});
    }
    n::save_npz((root / "pairs.npz").string(), arrays);
    m::LoadDepthPairs(root);
    std::cout << "Validated32/8/16 source-room-disjoint RGB/rawDepth sensor pairs.\n";
    return 0;
  } catch (const H5::Exception& e) {
    std::cerr << e.getDetailMsg() << '\n';
    return 1;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}

#endif // DATAMUNGE_ENABLE_DATA_ASSETS
