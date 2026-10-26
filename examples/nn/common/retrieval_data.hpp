// Deterministic class/image-disjoint CIFAR-10 retrieval pools and balanced
// hard-triplet batches: a port of tflow's common/ml/retrieval_data.h.
// Buckets are shuffled with std::mt19937(211) and batches with
// std::mt19937(223 + epoch) exactly as tflow does, so with the same standard
// library (libc++ on macOS) every pool and batch row is identical to TF's.
// The CIFAR binaries are read as raw bytes and only the selected rows are
// decoded (data::LoadCifar would hold 50000 images as int32).
#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <datamunge/nn/nn.hpp>
#include <filesystem>
#include <random>
#include <set>
#include <string>
#include <vector>

#include "common/binary.hpp"
#include "common/checks.hpp"
#include "common/triplet.hpp"

namespace metric {

/// Images float32 [N, 32, 32, 3] in [0, 1] (host copy in `pixels`), category labels and source IDs.
struct RetrievalPool {
  nn::Tensor           images;
  std::vector<float>   pixels;
  std::vector<int>     labels;
  std::vector<int64_t> ids;
  [[nodiscard]] std::size_t size() const { return labels.size(); }
};
struct RetrievalSplits {
  RetrievalPool fit, validation_query, validation_gallery, test_query, test_gallery;
};

inline RetrievalPool RetrievalRows(const RetrievalPool& input, const std::vector<std::size_t>& rows) {
  RetrievalPool out;
  out.pixels.resize(rows.size() * 3072);
  for (std::size_t i = 0; i < rows.size(); ++i) {
    check::Require(rows[i] < input.labels.size(), "retrieval row bounds");
    std::copy_n(input.pixels.begin() + std::ptrdiff_t(rows[i] * 3072), 3072, out.pixels.begin() + std::ptrdiff_t(i * 3072));
    out.labels.push_back(input.labels[rows[i]]);
    out.ids.push_back(input.ids[rows[i]]);
  }
  out.images = nn::from_vector(out.pixels, {std::int64_t(rows.size()), 32, 32, 3});
  return out;
}

/// CIFAR-10 binary records concatenated in file order (label byte + 3072 planar RGB bytes).
struct CifarRecords {
  std::vector<data::Bytes> files;
  std::vector<int>   labels;
  [[nodiscard]] const std::uint8_t* Record(std::size_t i) const { return files[i / 10000].data() + (i % 10000) * 3073; }
};
inline CifarRecords ReadCifarRecords(const std::vector<std::filesystem::path>& paths) {
  CifarRecords out;
  for (const auto& path : paths) {
    out.files.push_back(data::ReadBinary(path));
    check::Require(out.files.back().size() == 10000 * 3073, "complete 10000-record CIFAR-10 batch");
    for (std::size_t start = 0; start < out.files.back().size(); start += 3073) {
      const int label = out.files.back()[start];
      check::Require(label < 10, "CIFAR label out of range");
      out.labels.push_back(label);
    }
  }
  return out;
}

inline RetrievalSplits LoadRetrievalData(const std::filesystem::path& root) {
  std::vector<std::filesystem::path> paths;
  for (int i = 1; i <= 5; ++i) paths.push_back(root / ("data_batch_" + std::to_string(i) + ".bin"));
  const auto development = ReadCifarRecords(paths), test = ReadCifarRecords({root / "test_batch.bin"});
  check::Require(development.labels.size() == 50000 && test.labels.size() == 10000, "official CIFAR-10 source sizes");
  std::array<std::vector<std::size_t>, 10> dev_rows, test_rows;
  for (std::size_t i = 0; i < development.labels.size(); ++i) dev_rows[std::size_t(development.labels[i])].push_back(i);
  for (std::size_t i = 0; i < test.labels.size(); ++i) test_rows[std::size_t(test.labels[i])].push_back(i);
  std::mt19937 random(211);
  for (auto& rows : dev_rows) std::shuffle(rows.begin(), rows.end(), random);
  for (auto& rows : test_rows) std::shuffle(rows.begin(), rows.end(), random);
  auto make = [](const CifarRecords& source, const std::array<std::vector<std::size_t>, 10>& buckets, const std::vector<int>& classes,
                 int begin, int each, std::int64_t id_offset) {
    RetrievalPool out;
    out.pixels.resize(classes.size() * std::size_t(each) * 3072);
    std::size_t row = 0;
    for (int label : classes)
      for (int i = begin; i < begin + each; ++i, ++row) {
        const std::size_t index = buckets[std::size_t(label)].at(std::size_t(i));
        out.labels.push_back(label);
        out.ids.push_back(id_offset + std::int64_t(index));
        const std::uint8_t* record = source.Record(index) + 1;
        for (int pixel = 0; pixel < 1024; ++pixel)
          for (int c = 0; c < 3; ++c) out.pixels[row * 3072 + std::size_t(pixel * 3 + c)] = record[c * 1024 + pixel] / 255.f;
      }
    out.images = nn::from_vector(out.pixels, {std::int64_t(row), 32, 32, 3});
    return out;
  };
  RetrievalSplits s{make(development, dev_rows, {0, 1, 2, 3, 4}, 0, 32, 0), make(development, dev_rows, {5, 6}, 0, 8, 0),
                    make(development, dev_rows, {5, 6}, 8, 16, 0), make(test, test_rows, {7, 8, 9}, 0, 16, 100000),
                    make(test, test_rows, {7, 8, 9}, 16, 32, 100000)};
  std::set<std::int64_t> used;
  for (const auto* pool : {&s.fit, &s.validation_query, &s.validation_gallery, &s.test_query, &s.test_gallery})
    for (std::int64_t id : pool->ids) check::Require(used.insert(id).second, "no repeated source images across any fitting/query/gallery pools");
  return s;
}

/// 8 batches of 4 images from each of the 5 fitting classes, reshuffled per epoch.
inline std::vector<std::vector<std::size_t>> RetrievalBatches(const RetrievalPool& fitting, int epoch) {
  std::array<std::vector<std::size_t>, 5> buckets;
  for (std::size_t i = 0; i < fitting.labels.size(); ++i) {
    check::Require(fitting.labels[i] >= 0 && fitting.labels[i] < 5, "fit-only category labels");
    buckets[std::size_t(fitting.labels[i])].push_back(i);
  }
  std::mt19937 random(std::uint32_t(223 + epoch));
  for (auto& rows : buckets) {
    check::Require(rows.size() == 32, "32 fitting images per class");
    std::shuffle(rows.begin(), rows.end(), random);
  }
  std::vector<std::vector<std::size_t>> result;
  for (int batch = 0; batch < 8; ++batch) {
    std::vector<std::size_t> rows;
    for (const auto& bucket : buckets) rows.insert(rows.end(), bucket.begin() + 4 * batch, bucket.begin() + 4 * batch + 4);
    std::shuffle(rows.begin(), rows.end(), random);
    result.push_back(rows);
  }
  return result;
}

/// Batch-hard mining: farthest positive and nearest negative per anchor (self excluded).
inline std::vector<TripletIndex> RetrievalHardTriplets(const HostEmbeddings& features, const std::vector<int>& labels) {
  check::Require(features.rows == std::int64_t(labels.size()), "hard mining aligned descriptor labels");
  std::vector<TripletIndex> result;
  for (int anchor = 0; anchor < int(labels.size()); ++anchor) {
    int    positive = -1, negative = -1;
    double farthest = -1, nearest = INFINITY;
    for (int other = 0; other < int(labels.size()); ++other)
      if (anchor != other) {
        const double distance = EmbeddingDistance(features, anchor, other);
        if (labels[anchor] == labels[other] && distance > farthest) farthest = distance, positive = other;
        if (labels[anchor] != labels[other] && distance < nearest) nearest = distance, negative = other;
      }
    check::Require(positive >= 0 && negative >= 0, "every anchor has a distinct positive and negative");
    result.push_back({anchor, positive, negative});
  }
  return result;
}
inline std::vector<TripletIndex> RetrievalHardTriplets(const nn::Tensor& features, const std::vector<int>& labels) {
  return RetrievalHardTriplets(HostEmbeddings(features), labels);
}

}  // namespace metric
