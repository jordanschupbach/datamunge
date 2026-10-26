// Exact descriptor index and noninterpolated average precision with self
// exclusion: a port of tflow's common/ml/retrieval_index.h. Descriptors are
// host float matrices (HostEmbeddings); distances are accumulated in double.
#pragma once

#include <algorithm>
#include <cmath>
#include <set>
#include <vector>

#include "common/retrieval_data.hpp"

namespace metric {

struct RetrievalNeighbor {
  std::size_t  row;
  std::int64_t id;
  double       distance;
};

class DescriptorIndex {
 public:
  HostEmbeddings            features;
  std::vector<std::int64_t> ids;
  DescriptorIndex(HostEmbeddings values, std::vector<std::int64_t> source_ids) : features(std::move(values)), ids(std::move(source_ids)) {
    check::Require(features.rows == std::int64_t(ids.size()) && features.dims > 0, "descriptor index dimensions");
    std::set<std::int64_t> unique;
    for (std::int64_t id : ids) check::Require(unique.insert(id).second, "unique gallery source IDs");
    for (float v : features.values) check::Require(std::isfinite(v), "finite indexed coordinates");
  }
  DescriptorIndex(const nn::Tensor& values, std::vector<std::int64_t> source_ids) : DescriptorIndex(HostEmbeddings(values), std::move(source_ids)) {}
  /// Ascending squared distance, ties by source ID; the gallery item with ID `exclude` is skipped.
  [[nodiscard]] std::vector<RetrievalNeighbor> Rank(const HostEmbeddings& query, int row, std::int64_t exclude) const {
    check::Require(query.dims == features.dims && row >= 0 && row < query.rows, "query descriptor dimensions");
    std::vector<RetrievalNeighbor> ranking;
    for (std::size_t gallery = 0; gallery < ids.size(); ++gallery)
      if (ids[gallery] != exclude) {
        double distance = 0;
        for (std::int64_t c = 0; c < features.dims; ++c) {
          const double difference = double(query.at(row, c)) - features.at(std::int64_t(gallery), c);
          check::Require(std::isfinite(difference), "finite query coordinates");
          distance += difference * difference;
        }
        ranking.push_back({gallery, ids[gallery], distance});
      }
    std::sort(ranking.begin(), ranking.end(),
              [](const auto& a, const auto& b) { return a.distance < b.distance || (a.distance == b.distance && a.id < b.id); });
    return ranking;
  }
};

struct RetrievalMetrics {
  double map = 0, hit1 = 0, hit5 = 0;
  int    eligible = 0, without_positive = 0;
};

inline RetrievalMetrics ScoreRetrieval(const HostEmbeddings& queries, const RetrievalPool& query, const DescriptorIndex& index,
                                       const std::vector<int>& gallery_labels) {
  check::Require(queries.rows == std::int64_t(query.labels.size()) && query.ids.size() == query.labels.size() &&
                     index.ids.size() == gallery_labels.size(),
                 "query/gallery metric labels and IDs");
  RetrievalMetrics result;
  for (int n = 0; n < queries.rows; ++n) {
    const auto ranking  = index.Rank(queries, n, query.ids[std::size_t(n)]);
    int        relevant = 0;
    for (const auto& item : ranking) relevant += gallery_labels[item.row] == query.labels[std::size_t(n)];
    if (!relevant) {
      ++result.without_positive;
      continue;
    }
    ++result.eligible;
    int    found         = 0;
    double precision_sum = 0;
    for (int rank = 0; rank < int(ranking.size()); ++rank)
      if (gallery_labels[ranking[std::size_t(rank)].row] == query.labels[std::size_t(n)]) {
        ++found;
        precision_sum += double(found) / (rank + 1);
        if (rank == 0) ++result.hit1;
        if (found == 1 && rank < 5) ++result.hit5;
      }
    result.map += precision_sum / relevant;
  }
  if (result.eligible) {
    result.map /= result.eligible;
    result.hit1 /= result.eligible;
    result.hit5 /= result.eligible;
  }
  return result;
}

/// Unit-normalized (32/block)^2 x 3 block means of RGB - .5 (block 32: mean RGB).
inline HostEmbeddings RetrievalPixels(const RetrievalPool& data, int block = 4) {
  check::Require(block > 0 && 32 % block == 0, "RGB baseline pooling block");
  const int      side = 32 / block, dimensions = side * side * 3;
  HostEmbeddings values;
  values.rows = std::int64_t(data.size()), values.dims = dimensions;
  values.values.resize(data.size() * std::size_t(dimensions));
  for (std::size_t n = 0; n < data.size(); ++n) {
    double norm = 0;
    for (int y = 0; y < side; ++y)
      for (int x = 0; x < side; ++x)
        for (int c = 0; c < 3; ++c) {
          double sum = 0;
          for (int dy = 0; dy < block; ++dy)
            for (int dx = 0; dx < block; ++dx) sum += data.pixels[n * 3072 + std::size_t(((y * block + dy) * 32 + x * block + dx) * 3 + c)] - .5;
          const float value = float(sum / (block * block));
          values.values[n * std::size_t(dimensions) + std::size_t((y * side + x) * 3 + c)] = value;
          norm += double(value) * value;
        }
    norm = std::sqrt(std::max(norm, 1e-12));
    for (int j = 0; j < dimensions; ++j) values.values[n * std::size_t(dimensions) + std::size_t(j)] = float(values.values[n * std::size_t(dimensions) + std::size_t(j)] / norm);
  }
  return values;
}

}  // namespace metric
