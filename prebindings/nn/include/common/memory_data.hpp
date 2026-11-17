// Deterministic local datasets for the synthetic memory-network examples.
// Same task dimensions/protocol as the external assets, with an explicitly
// independent local selection. All languages call this C++ generator.
#pragma once
#include <filesystem>
#include <fstream>
#include <random>

#include "common/dataset_assets.hpp"
#include "common/memory_task.hpp"

namespace memory_data {
namespace nn = datamunge::nn;

inline nn::NamedTensors Make(bool copy_task) {
  std::mt19937     generator(copy_task ? 45101 : 45301);
  nn::NamedTensors records;
  for (int split = 0; split < 3; ++split) {
    const std::string prefix = split == 0 ? "fit_" : split == 1 ? "validation_" : "test_";
    const int         count  = split == 0 ? 128 : 32;
    std::vector<int>  lengths(count), ids(count), facts(count * 12), queries(count), answers(count);
    std::vector<float> input(count * 36), target(count * 27);
    for (int row = 0; row < count; ++row) {
      ids[row]         = split * 10000 + row;
      const int length = 1 + generator() % (copy_task ? 4 : 6);
      lengths[row]     = length;
      if (copy_task) {
        for (int step = 0; step < length; ++step)
          for (int bit = 0; bit < 3; ++bit) {
            const float value                               = float(generator() & 1);
            input[(row * 9 + step) * 4 + bit]               = value;
            target[(row * 9 + length + 1 + step) * 3 + bit] = value;
          }
        input[(row * 9 + length) * 4 + 3] = 1;
      } else {
        int entities[7] = {1, 2, 3, 4, 5, 6, 7};
        for (int i = 6; i > 0; --i)
          std::swap(entities[i], entities[generator() % (i + 1)]);
        const int selected = generator() % length;
        for (int slot = 0; slot < length; ++slot) {
          const int answer                = generator() % 4;
          facts[(row * 6 + slot) * 2]     = entities[slot];
          facts[(row * 6 + slot) * 2 + 1] = 8 + answer;
          if (slot == selected) {
            queries[row] = entities[slot];
            answers[row] = answer;
          }
        }
      }
    }
    if (copy_task) {
      records.emplace_back(prefix + "input", nn::from_vector(input, {count, 9, 4}));
      records.emplace_back(prefix + "target", nn::from_vector(target, {count, 9, 3}));
    } else {
      records.emplace_back(prefix + "facts", nn::from_vector(facts, {count, 6, 2}));
      records.emplace_back(prefix + "query", nn::from_vector(queries, {count}));
      records.emplace_back(prefix + "target", nn::from_vector(answers, {count}));
    }
    records.emplace_back(prefix + "lengths", nn::from_vector(lengths, {count}));
    records.emplace_back(prefix + "source_ids", nn::from_vector(ids, {count}));
  }
  return records;
}

inline void PrepareDataset(const std::string& directory, bool copy_task, bool overwrite = false) {
  namespace fs    = std::filesystem;
  const auto root = fs::path(directory);
  const auto data = root / "data.npz";
  if (fs::exists(data) && !overwrite)
    return;
  dataset_assets::detail::Staging stage(root);
  const auto                      temporary = stage.path / "data.npz";
  nn::save_npz(temporary.string(), Make(copy_task));
  std::ofstream manifest(stage.path / "manifest.json");
  manifest << "{\"generator\":\"datamunge local C++ memory task v1\",\"external_selection\":false,"
           << "\"seed\":" << (copy_task ? 45101 : 45301)
           << ",\"fit\":128,\"validation\":32,\"test\":32,\"task\":\""
           << (copy_task ? "binary copy with delimiter and recall-only targets"
                         : "unambiguous fact/query question answering")
           << "\"}\n";
  if (!manifest)
    throw std::runtime_error("cannot write memory dataset manifest");
  manifest.close();
  if (overwrite) {
    fs::rename(stage.path / "manifest.json", root / "manifest.json");
    fs::rename(temporary, data);
  } else {
    dataset_assets::detail::Publish(stage.path / "manifest.json", root / "manifest.json");
    dataset_assets::detail::Publish(temporary, data);
  }
}

inline void Prepare(const std::string& directory, bool overwrite = false) {
  for (bool copy_task : {false, true}) {
    PrepareDataset(
        (std::filesystem::path(directory) / (copy_task ? "ntm" : "memnet")).string(),
        copy_task,
        overwrite);
  }
}
} // namespace memory_data
