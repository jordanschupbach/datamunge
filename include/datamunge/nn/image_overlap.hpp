#pragma once

/// @file image_overlap.hpp
/// @brief Exact decoded-image overlap audits across labelled dataset partitions.

#include <cstring>
#include <datamunge/nn/autograd.hpp>
#include <datamunge/nn/datasets.hpp>
#include <datamunge/nn/ops.hpp>
#include <stdexcept>
#include <unordered_map>

namespace datamunge::nn::study {

/// @brief Exact matching-image pair counts and later rows shared with the first partition.
///
/// Counts are upper triangular: pairs[i][j], i<=j, counts unordered matching
/// image pairs. Within-partition groups of k identical images contribute k(k-1)/2.
/// Hash collisions are checked with a full byte comparison, so never count as matches.
/// Only decoded Float32 bytes are compared; near duplicates are outside this audit.
/// @par Example
/// @code
/// namespace n=dm::nn;
/// const n::ImageDataset x{n::zeros({2,2,2,1}),n::zeros({2},n::DType::Int64)};
/// const auto audit=n::study::audit_image_overlap({x});
/// const auto pairs=audit.pairs[0][0]; // 1
/// @endcode
/// @see audit_image_overlap()
struct ImageOverlapAudit {
  std::vector<std::vector<std::uint64_t>>
                pairs; ///< Upper-triangular unordered exact-image pair counts.
  std::uint64_t conflicting_label_pairs = 0; ///< Matching-image pairs whose integer labels differ.
  std::vector<std::vector<std::int64_t>>
      rows_matching_first; ///< Unique row IDs in each later partition matching any first-partition
                           ///< image; first entry empty.
};

/// @brief Audit bit-identical decoded Float32 images within and between labelled partitions.
///
/// Uses FNV-1a only to find candidate buckets, then memcmp to confirm every
/// match. Signed zero has distinct bytes; all nonfinite images are rejected.
/// Copies noncontiguous inputs to contiguous storage without changing caller data.
/// No model is fitted, no split is changed and no source files are acquired.
/// Expected O(total pixel bytes + confirmed/colliding pair comparisons) work;
/// highly repeated images require quadratic pair enumeration. Host metadata
/// scales with image count; contiguous tensor handles retain their image storage.
/// @param partitions Nonempty list of nonempty CPU Float32 NHWC batches with identical HWC
///                   geometry and aligned Int64 [N] labels. Pixel range is unrestricted.
/// @return Upper-triangular exact pair counts, label conflicts and later row IDs matching partition
/// zero.
/// @throws std::invalid_argument For empty, nonfinite, non-Float32 or inconsistent image/label
/// geometry.
/// @throws std::logic_error For device storage that cannot be read directly on the CPU.
/// @par Example
/// @code
/// namespace n=dm::nn;
/// const n::ImageDataset x{n::zeros({2,2,2,1}),n::zeros({2},n::DType::Int64)};
/// const auto audit=n::study::audit_image_overlap({x,x});
/// const auto between=audit.pairs[0][1]; // 4
/// @endcode
/// @par References
/// - Fowler, Noll and Vo, FNV hash parameters and FNV-1a update order.
///   https://www.isthe.com/chongo/tech/comp/fnv/
/// @see ImageOverlapAudit
inline ImageOverlapAudit audit_image_overlap(const std::vector<ImageDataset>& partitions) {
  if (partitions.empty())
    throw std::invalid_argument("at least one image partition required");
  NoGradGuard                            guard;
  std::vector<Tensor>                    images;
  std::vector<std::vector<std::int64_t>> labels;
  Shape                                  geometry;
  for (const auto& partition : partitions) {
    const auto& x = partition.images;
    const auto& y = partition.labels;
    if (!x.defined() || x.dim() != 4 || x.size(0) < 1 || x.numel() < 1 ||
        x.dtype() != DType::Float32 || !y.defined() || y.dtype() != DType::Int64 ||
        y.shape() != Shape{x.size(0)} || !all(is_finite(x)).item<bool>())
      throw std::invalid_argument("finite Float32 NHWC images with aligned Int64 labels required");
    const Shape hwc{x.size(1), x.size(2), x.size(3)};
    if (!geometry.empty() && geometry != hwc)
      throw std::invalid_argument("partition image geometries differ");
    geometry = hwc;
    images.push_back(x.contiguous());
    labels.push_back(y.to_vector<std::int64_t>());
  }
  ImageOverlapAudit out;
  out.pairs.assign(images.size(), std::vector<std::uint64_t>(images.size(), 0));
  out.rows_matching_first.resize(images.size());
  /// @brief A candidate's partition index and local row ID, retaining no additional pixel storage.
  using Row = std::pair<std::size_t, std::int64_t>;
  std::unordered_map<std::uint64_t, std::vector<Row>> buckets;
  const auto                                          bytes =
      static_cast<std::size_t>(images[0].numel() / images[0].size(0)) * sizeof(float);
  for (std::size_t part = 0; part < images.size(); ++part) {
    const auto* data = reinterpret_cast<const unsigned char*>(images[part].data_ptr<float>());
    for (std::int64_t row = 0; row < images[part].size(0); ++row) {
      const auto*   pixel = data + row * bytes;
      std::uint64_t hash  = 14695981039346656037ull;
      for (std::size_t i = 0; i < bytes; ++i) {
        hash ^= pixel[i];
        hash *= 1099511628211ull;
      }
      auto& candidates    = buckets[hash];
      bool  matches_first = false;
      for (const auto& [old_part, old_row] : candidates) {
        const auto* previous =
            reinterpret_cast<const unsigned char*>(images[old_part].data_ptr<float>()) +
            old_row * bytes;
        if (std::memcmp(pixel, previous, bytes) == 0) {
          ++out.pairs[old_part][part];
          out.conflicting_label_pairs += labels[old_part][old_row] != labels[part][row];
          matches_first |= old_part == 0;
        }
      }
      if (part > 0 && matches_first)
        out.rows_matching_first[part].push_back(row);
      candidates.emplace_back(part, row);
    }
  }
  return out;
}
} // namespace datamunge::nn::study
