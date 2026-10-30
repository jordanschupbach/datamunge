#pragma once
#include "common/binary.hpp"
#include "common/checks.hpp"
#include "common/triplet.hpp"
#include "datamunge/nn/retrieval_data.hpp"
namespace metric {
/// Images float32 [N, 32, 32, 3] in [0, 1] (host copy in `pixels`), category labels and source IDs.
struct RetrievalPool {
  nn::Tensor images;
  std::vector<float> pixels;
  std::vector<int> labels;
  std::vector<int64_t> ids;
  [[nodiscard]] std::size_t size() const {
    return labels.size();
  }
};
struct RetrievalSplits {
  RetrievalPool fit, validation_query, validation_gallery, test_query, test_gallery;
};

/// Convert only container ownership;pixel values are tensor ingress/presentation data.
inline nn::models::TensorRetrievalPool CanonicalRetrievalPool(const RetrievalPool &data) {
  return {data.images.defined()
              ? data.images
              : nn::from_vector(data.pixels, {std::int64_t(data.size()), 32, 32, 3}),
          data.labels, data.ids};
}
inline RetrievalPool LegacyRetrievalPool(const nn::models::TensorRetrievalPool &data) {
  return {data.images, data.images.to_vector<float>(), data.labels, data.ids};
}
inline RetrievalPool RetrievalRows(const RetrievalPool &data,
                                   const std::vector<std::size_t> &rows) {
  return LegacyRetrievalPool(
      nn::models::TensorRetrievalRows(CanonicalRetrievalPool(data), rows));
}
/// CIFAR-10 binary records concatenated in file order (label byte + 3072 planar RGB bytes).
struct CifarRecords {
  std::vector<data::Bytes> files;
  std::vector<int> labels;
  [[nodiscard]] const std::uint8_t *Record(std::size_t i) const {
    return files[i / 10000].data() + (i % 10000) * 3073;
  }
};
inline CifarRecords ReadCifarRecords(const std::vector<std::filesystem::path> &paths) {
  CifarRecords out;
  for (const auto &path : paths) {
    out.files.push_back(data::ReadBinary(path));
    check::Require(out.files.back().size() == 10000 * 3073,
                   "complete 10000-record CIFAR-10 batch");
    for (std::size_t start = 0; start < out.files.back().size(); start += 3073) {
      const int label = out.files.back()[start];
      check::Require(label < 10, "CIFAR label out of range");
      out.labels.push_back(label);
    }
  }
  return out;
}

inline RetrievalSplits LoadRetrievalData(const std::filesystem::path &root) {
  const auto data = nn::models::LoadTensorRetrievalData(root);
  return {LegacyRetrievalPool(data.fit), LegacyRetrievalPool(data.validation_query),
          LegacyRetrievalPool(data.validation_gallery), LegacyRetrievalPool(data.test_query),
          LegacyRetrievalPool(data.test_gallery)};
}
inline std::vector<std::vector<std::size_t>> RetrievalBatches(const RetrievalPool &fitting,
                                                              int epoch) {
  nn::models::TensorRetrievalPool metadata;
  metadata.labels = fitting.labels;
  return nn::models::RetrievalBatches(metadata, epoch);
}
inline std::vector<TripletIndex> RetrievalHardTriplets(const nn::Tensor &features,
                                                       const std::vector<int> &labels) {
  return nn::models::RetrievalHardTriplets(features, labels);
}
inline std::vector<TripletIndex> RetrievalHardTriplets(const HostEmbeddings &features,
                                                       const std::vector<int> &labels) {
  return nn::models::RetrievalHardTriplets(
      nn::from_vector(features.values, {features.rows, features.dims}), labels);
}
} // namespace metric
