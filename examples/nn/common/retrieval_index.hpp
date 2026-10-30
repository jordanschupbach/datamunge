#pragma once
#include "common/retrieval_data.hpp"
#include "datamunge/nn/retrieval_index.hpp"
namespace metric {
struct RetrievalNeighbor {
  std::size_t row;
  std::int64_t id;
  double distance;
};

class DescriptorIndex {
public:
  HostEmbeddings features;
  std::vector<std::int64_t> ids;
  DescriptorIndex(HostEmbeddings values, std::vector<std::int64_t> source_ids)
      : features(std::move(values)), ids(std::move(source_ids)) {
    Canonical();
  }
  DescriptorIndex(const nn::Tensor &values, std::vector<std::int64_t> source_ids)
      : DescriptorIndex(HostEmbeddings(values), std::move(source_ids)) {
  }
  nn::models::TensorDescriptorIndex Canonical() const {
    return {nn::from_vector(features.values, {features.rows, features.dims}), ids};
  }
  std::vector<RetrievalNeighbor> Rank(const HostEmbeddings &query, int row,
                                      std::int64_t exclude) const {
    const auto result = Canonical().Rank(
        nn::from_vector(query.values, {query.rows, query.dims}), row, exclude);
    const auto rows = result.rows.to_vector<std::int64_t>(),
               ids = result.ids.to_vector<std::int64_t>();
    const auto distances = result.distances.to_vector<double>();
    std::vector<RetrievalNeighbor> ranking;
    for (std::size_t i = 0; i < rows.size(); ++i)
      ranking.push_back({std::size_t(rows[i]), ids[i], distances[i]});
    return ranking;
  }
};
struct RetrievalMetrics {
  double map = 0, hit1 = 0, hit5 = 0;
  int eligible = 0, without_positive = 0;
};

inline RetrievalMetrics ScoreRetrieval(const HostEmbeddings &features,
                                       const RetrievalPool &query,
                                       const DescriptorIndex &index,
                                       const std::vector<int> &labels) {
  const auto result = nn::models::ScoreTensorRetrieval(
      nn::from_vector(features.values, {features.rows, features.dims}), query.labels,
      query.ids, index.Canonical(), labels);
  return {result.map.item(), result.hit1.item(), result.hit5.item(),
          int(result.eligible.item<std::int64_t>()),
          int(result.without_positive.item<std::int64_t>())};
}
inline HostEmbeddings RetrievalPixels(const RetrievalPool &data, int block = 4) {
  return HostEmbeddings(
      nn::models::RetrievalPixelDescriptors(CanonicalRetrievalPool(data), block));
}
} // namespace metric
