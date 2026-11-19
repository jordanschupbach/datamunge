/// @file
/// @brief Verified Cora citation graph and stratified classification split.
#pragma once

#include <datamunge/datasets/assets.hpp>
#include <datamunge/datasets/download.hpp>
#include <datamunge/nn/tensor.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <map>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace datamunge::datasets {

/// @brief Own normalized features, sparse graph entries, and label splits for Cora.
///
/// Edges are sorted by receiver then source, with a single undirected
/// connection per pair and one self-loop per node.
/// @par Example
/// @code
/// const auto data = dm::datasets::load_cora();
/// std::println("Training labels {}", data.training.size(0)); // 140
/// @endcode
/// @par References
/// - Kipf and Welling, Cora fixture for pygcn,
///   https://github.com/tkipf/pygcn/tree/1600b5b748b3976413d1e307540ccc62605b4d6d/data/cora
/// @see load_cora()
struct CoraCitationGraph {
  nn::Tensor features; ///< Float32 [2708,1433] row-normalized bag-of-words features.
  nn::Tensor labels; ///< Int32 [2708] class IDs, numbered in first-occurrence order.
  nn::Tensor coordinates; ///< Int64 [E,2] receiver and source indices.
  nn::Tensor weights; ///< Float32 [E] symmetrically degree-normalized edge weights.
  nn::Tensor training; ///< Int32 [140] training node indices, 20 per class.
  nn::Tensor test; ///< Int32 [2568] remaining node indices.
};

/// @brief Load the checksum-verified Cora graph and its reproducible label split.
///
/// Fetches the two original Cora tables if absent. Feature rows sum to one
/// when nonempty. Citations are symmetrized and deduplicated, then unit
/// self-loops are added. Each edge receives weight 1/sqrt(degree[i]*degree[j]).
/// Class groups are shuffled with MT19937 seed 49301 and the first 20
/// members of each group provide training labels.
/// @param directory Dataset cache path. Empty uses data_home()/cora.
/// @return Owned CPU tensors with the normalized graph, labels and split.
/// @throws std::runtime_error For download, checksum, I/O or malformed dataset errors.
/// @par Example
/// @code
/// const auto data = dm::datasets::load_cora();
/// std::println("Held-out labels {}", data.test.size(0)); // 2568
/// @endcode
/// @par References
/// - Kipf and Welling, Semi-Supervised Classification with Graph Convolutional Networks,
///   https://arxiv.org/abs/1609.02907
/// - Pinned original Cora files, https://github.com/tkipf/pygcn/tree/1600b5b748b3976413d1e307540ccc62605b4d6d/data/cora
/// @see CoraCitationGraph, fetch_file()
inline CoraCitationGraph load_cora(const std::filesystem::path &directory = {}) {
  const auto root = directory.empty() ? std::filesystem::path(data_home()) / "cora" : directory;
  const std::string base =
      "https://raw.githubusercontent.com/tkipf/pygcn/1600b5b748b3976413d1e307540ccc62605b4d6d/data/cora/";
  const auto content_path = (root / "cora.content").string();
  const auto citations_path = (root / "cora.cites").string();
  fetch_file(base + "cora.content", content_path,
             "b8ebed0fd45e53995abafd1b70cf6a9fcc8f86abc5826d69b97fe6ca46e9c8ee", 7823427, "cora");
  fetch_file(base + "cora.cites", citations_path,
             "ec1a372391b7f0f60a6aff0084e8abd8f19f0faa7e1f2441a41c492042d5945e", 69928, "cora");

  constexpr int nodes = 2708, words = 1433, classes = 7, labels_per_class = 20;
  std::ifstream content(content_path), citations(citations_path);
  if (!content || !citations)
    throw std::runtime_error("cannot open verified Cora tables");
  std::map<std::string, int> node_ids, category_ids;
  std::array<std::vector<int>, classes> groups;
  std::vector<float> features(nodes * words);
  std::vector<int> labels;
  std::string line, id, category;
  for (int row = 0; row < nodes; ++row) {
    if (!std::getline(content, line))
      throw std::runtime_error("missing Cora content row");
    std::istringstream fields(line);
    if (!(fields >> id) || !node_ids.emplace(id, row).second)
      throw std::runtime_error("invalid Cora node ID");
    auto first = features.begin() + row * words;
    int total = 0;
    for (int word = 0; word < words; ++word) {
      int bit;
      if (!(fields >> bit) || (bit != 0 && bit != 1))
        throw std::runtime_error("invalid Cora word indicator");
      first[word] = static_cast<float>(bit);
      total += bit;
    }
    if (total > 0)
      for (auto value = first; value != first + words; ++value)
        *value /= static_cast<float>(total);
    if (!(fields >> category) || (fields >> id))
      throw std::runtime_error("invalid Cora class label");
    const int label = category_ids.try_emplace(category, category_ids.size()).first->second;
    if (label >= classes)
      throw std::runtime_error("unexpected Cora class label");
    groups[label].push_back(row);
    labels.push_back(label);
  }
  if (content >> id)
    throw std::runtime_error("unexpected extra Cora content row");

  std::vector<std::set<int>> neighbors(nodes);
  for (int row = 0; row < nodes; ++row)
    neighbors[row].insert(row);
  std::string source, receiver;
  while (citations >> source) {
    if (!(citations >> receiver) || !node_ids.contains(source) || !node_ids.contains(receiver))
      throw std::runtime_error("invalid Cora citation");
    const int a = node_ids.at(source), b = node_ids.at(receiver);
    neighbors[a].insert(b);
    neighbors[b].insert(a);
  }

  std::vector<std::int64_t> coordinates;
  std::vector<float> weights;
  for (int a = 0; a < nodes; ++a)
    for (const int b : neighbors[a]) {
      coordinates.insert(coordinates.end(), {a, b});
      weights.push_back(1.f / std::sqrt(static_cast<float>(neighbors[a].size() * neighbors[b].size())));
    }
  std::mt19937 rng(49301);
  std::vector<int> training, test;
  for (auto &group : groups) {
    if (group.size() <= labels_per_class)
      throw std::runtime_error("insufficient Cora class members");
    std::shuffle(group.begin(), group.end(), rng);
    training.insert(training.end(), group.begin(), group.begin() + labels_per_class);
    test.insert(test.end(), group.begin() + labels_per_class, group.end());
  }
  return {nn::from_vector(features, {nodes, words}), nn::from_vector(labels),
          nn::from_vector(coordinates, {static_cast<std::int64_t>(weights.size()), 2}),
          nn::from_vector(weights), nn::from_vector(training), nn::from_vector(test)};
}

} // namespace datamunge::datasets
