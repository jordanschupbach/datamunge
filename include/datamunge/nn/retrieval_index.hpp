#pragma once
/// @file retrieval_index.hpp
/// @brief Tensor exact descriptor search,noninterpolated AP,pixel references and image-only galleries.
#include "datamunge/nn/image_ops.hpp"
#include "datamunge/nn/retrieval_data.hpp"
#include <optional>
namespace datamunge::nn::models {
namespace nn = datamunge::nn;
/// @brief Own one detached exact gallery ranking in sorted order.
///
/// All result tensors own sorted output storage. No index/query gradients
/// or learned state. A matching excluded source ID is omitted entirely.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto gallery = n::tensor({1., 0., 0., 1., -1., 0., 1., 0.}, {4, 2});
/// n::models::TensorDescriptorIndex index(gallery, {9, 3, 7, 2});
/// const auto query = n::tensor({1., 0., 1., 0.}, {2, 2});
/// const auto ranking = index.Rank(query, 0, 9);
/// @endcode
/// @par References
/// - Radenovic,Tolias and Chum(2018),Fine-tuning CNN Image Retrieval with No Human Annotation, https://arxiv.org/abs/1711.02512. Exact squared-distance ranking,source-ID tie breaking,self exclusion and noninterpolated AP follow the source category-retrieval teaching protocol.
/// @see TensorDescriptorIndex, ScoreTensorRetrieval()
struct TensorDescriptorRanking {
  nn::Tensor rows;      ///< Int64[R] original gallery row indices.
  nn::Tensor ids;       ///< Int64[R] source IDs,sorted by distance then ID.
  nn::Tensor distances; ///< Float64[R] squared Euclidean distances.
};
/// @brief Own a detached immutable descriptor index for exact squared-distance search.
///
/// Construction clones/promotes finite features to Float64 and owns source-ID
/// metadata. Rank preserves original coordinate precision,sorts ties by source
/// ID and optionally excludes one matching source ID. Concurrent const Rank
/// calls are independent;no fitting,approximate search or whitening.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto gallery = n::tensor({1., 0., 0., 1., -1., 0., 1., 0.}, {4, 2});
/// n::models::TensorDescriptorIndex index(gallery, {9, 3, 7, 2});
/// const auto query = n::tensor({1., 0., 1., 0.}, {2, 2});
/// const auto ranking = index.Rank(query, 0, 9);
/// @endcode
/// @par References
/// - Radenovic,Tolias and Chum(2018),Fine-tuning CNN Image Retrieval with No Human Annotation, https://arxiv.org/abs/1711.02512. Exact squared-distance ranking,source-ID tie breaking,self exclusion and noninterpolated AP follow the source category-retrieval teaching protocol.
/// @see TensorDescriptorIndex, ScoreTensorRetrieval()
class TensorDescriptorIndex {
  nn::Tensor features_;           ///< Detached owned Float64[G,D] gallery coordinates.
  std::vector<std::int64_t> ids_; ///< Owned unique gallery source-ID metadata.
public:
  /// @brief Clone finite descriptor coordinates and own unique source IDs.
  ///
  /// No numerical normalization is applied. Tensor conversion preserves
  /// Float32 coordinate values exactly in Float64;Float64 inputs stay Float64.
  /// Future mutation of the input tensor cannot alter this index.
  ///
  /// @param features Float32/64[G,D],D positive;empty G allowed.
  /// @param ids G unique Int64 source IDs,any signed values.
  /// @throws std::invalid_argument For invalid descriptor/pool shapes,dtypes,metadata,nonfinite values,duplicate IDs or requested preview rows.
  /// @par Example
  /// @code
  /// namespace n = dm::nn;
  /// const auto gallery = n::tensor({1., 0., 0., 1., -1., 0., 1., 0.}, {4, 2});
  /// n::models::TensorDescriptorIndex index(gallery, {9, 3, 7, 2});
  /// const auto query = n::tensor({1., 0., 1., 0.}, {2, 2});
  /// @endcode
  /// @par References
  /// - Radenovic,Tolias and Chum(2018),Fine-tuning CNN Image Retrieval with No Human Annotation, https://arxiv.org/abs/1711.02512. Exact squared-distance ranking,source-ID tie breaking,self exclusion and noninterpolated AP follow the source category-retrieval teaching protocol.
  /// @see TensorDescriptorIndex, ScoreTensorRetrieval()
  TensorDescriptorIndex(const nn::Tensor &features, std::vector<std::int64_t> ids)
      : ids_(std::move(ids)) {
    nn::NoGradGuard guard;
    if (features.dim() != 2 || features.shape()[1] < 1 ||
        features.shape()[0] != std::int64_t(ids_.size()) ||
        (features.dtype() != nn::DType::Float32 && features.dtype() != nn::DType::Float64) ||
        !nn::all(nn::less(nn::abs(features), std::numeric_limits<double>::infinity()))
             .item<bool>() ||
        std::set<std::int64_t>(ids_.begin(), ids_.end()).size() != ids_.size())
      throw std::invalid_argument("finite aligned descriptors and unique source IDs required");
    features_ = features.detach().to(nn::DType::Float64).clone();
  }
  /// @brief Return the number of gallery descriptors.
  /// @return Owned source-ID count.
  std::size_t size() const {
    return ids_.size();
  }
  /// @brief Return the positive descriptor width shared by gallery and queries.
  /// @return Original descriptor column count.
  std::int64_t dimensions() const {
    return features_.shape()[1];
  }
  /// @brief Expose read-only source-ID metadata in original gallery-row order.
  /// @return Reference valid until index destruction;caller must not cast away constness.
  const std::vector<std::int64_t> &source_ids() const {
    return ids_;
  }
  /// @brief Rank one query against every eligible gallery row by exact squared distance.
  ///
  /// Subtract coordinates in Float64 before summing squares;there is no
  /// matrix-product cancellation. Stable distance sorting after integer-ID
  /// sorting makes exact distance ties deterministic by source ID. Host loops
  /// filter only source-ID metadata. No gradient,mutation,optimizer or fitting.
  ///
  /// @param queries Finite Float32/64[Q,D] query matrix,matching gallery width.
  /// @param row Valid query-row index.
  /// @param exclude Optional source ID to omit,default none.
  /// @return Detached sorted gallery rows/IDs/Float64 distances,possibly empty.
  /// @throws std::invalid_argument For invalid descriptor/pool shapes,dtypes,metadata,nonfinite values,duplicate IDs or requested preview rows.
  /// @par Example
  /// @code
  /// namespace n = dm::nn;
  /// const auto gallery = n::tensor({1., 0., 0., 1., -1., 0., 1., 0.}, {4, 2});
  /// n::models::TensorDescriptorIndex index(gallery, {9, 3, 7, 2});
  /// const auto query = n::tensor({1., 0., 1., 0.}, {2, 2});
  /// const auto ranking = index.Rank(query, 0, 9);
  /// @endcode
  /// @par References
  /// - Radenovic,Tolias and Chum(2018),Fine-tuning CNN Image Retrieval with No Human Annotation, https://arxiv.org/abs/1711.02512. Exact squared-distance ranking,source-ID tie breaking,self exclusion and noninterpolated AP follow the source category-retrieval teaching protocol.
  /// @see TensorDescriptorIndex, ScoreTensorRetrieval()
  TensorDescriptorRanking Rank(const nn::Tensor &queries, std::int64_t row,
                               std::optional<std::int64_t> exclude = std::nullopt) const {
    nn::NoGradGuard guard;
    if (queries.dim() != 2 || queries.shape()[1] != features_.shape()[1] || row < 0 ||
        row >= queries.shape()[0] ||
        (queries.dtype() != nn::DType::Float32 && queries.dtype() != nn::DType::Float64))
      throw std::invalid_argument("invalid query descriptor geometry");
    const auto query = nn::select(queries, 0, row).to(nn::DType::Float64);
    if (!nn::all(nn::less(nn::abs(query), std::numeric_limits<double>::infinity()))
             .item<bool>())
      throw std::invalid_argument("finite query coordinates required");
    std::vector<std::int64_t> rows;
    for (std::size_t i = 0; i < ids_.size(); ++i)
      if (!exclude || ids_[i] != *exclude)
        rows.push_back(std::int64_t(i));
    const auto candidates = nn::from_vector(rows), source_ids = nn::from_vector(ids_);
    const auto ids = nn::gather(source_ids, candidates);
    const auto distances = nn::sum(nn::square(nn::gather(features_, candidates) - query), {1});
    const auto by_id = nn::argsort(ids);
    const auto by_distance = nn::argsort(nn::gather(distances, by_id));
    const auto order = nn::gather(by_id, by_distance);
    return {nn::gather(candidates, order), nn::gather(ids, order),
            nn::gather(distances, order)};
  }
};
/// @brief Own mean AP,hit rates and query-eligibility tensor counts.
///
/// AP and hit metrics average eligible queries equally. Queries with no
/// relevant gallery rows after self exclusion are counted separately and
/// excluded from metric denominators. No eligible query yields zero metrics.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto gallery = n::tensor({1., 0., 0., 1., -1., 0., 1., 0.}, {4, 2});
/// n::models::TensorDescriptorIndex index(gallery, {9, 3, 7, 2});
/// const auto query = n::tensor({1., 0., 1., 0.}, {2, 2});
/// const auto score =
///     n::models::ScoreTensorRetrieval(query, {1, 99}, {9, 90}, index, {1, 0, 1, 1});
/// @endcode
/// @par References
/// - Radenovic,Tolias and Chum(2018),Fine-tuning CNN Image Retrieval with No Human Annotation, https://arxiv.org/abs/1711.02512. Exact squared-distance ranking,source-ID tie breaking,self exclusion and noninterpolated AP follow the source category-retrieval teaching protocol.
/// @see TensorDescriptorIndex, ScoreTensorRetrieval()
struct TensorRetrievalMetrics {
  nn::Tensor map;  ///< Float64 scalar mean noninterpolated average precision.
  nn::Tensor hit1; ///< Float64 scalar eligible-query hit fraction at rank1.
  nn::Tensor hit5; ///< Float64 scalar eligible-query hit fraction within first five.
  nn::Tensor
      eligible; ///< Int64 scalar queries with at least one relevant eligible gallery row.
  nn::Tensor without_positive; ///< Int64 scalar queries excluded for no relevant eligible row.
};
/// @brief Score exact category retrieval with self-excluded noninterpolated AP.
///
/// For each query,AP=sum(precision at relevant ranks)/number relevant after
/// self exclusion. Exact ties use source-ID order. Only queries with positives
/// enter mean AP/hit rates. Empty/no-eligible inputs return zero. No threshold,
/// interpolation,model forward,optimizer or query updates.
///
/// @param queries Float32/64[Q,D] finite query descriptors,Q may be zero.
/// @param query_labels Q category IDs as relevance metadata.
/// @param query_ids Q original source IDs for self exclusion.
/// @param index Exact gallery descriptor index.
/// @param gallery_labels One relevance category per original gallery row.
/// @return Detached tensor mAP,hit@1,hit@5,eligible and no-positive counts.
/// @throws std::invalid_argument For invalid descriptor/pool shapes,dtypes,metadata,nonfinite values,duplicate IDs or requested preview rows.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto gallery = n::tensor({1., 0., 0., 1., -1., 0., 1., 0.}, {4, 2});
/// n::models::TensorDescriptorIndex index(gallery, {9, 3, 7, 2});
/// const auto query = n::tensor({1., 0., 1., 0.}, {2, 2});
/// const auto score =
///     n::models::ScoreTensorRetrieval(query, {1, 99}, {9, 90}, index, {1, 0, 1, 1});
/// @endcode
/// @par References
/// - Radenovic,Tolias and Chum(2018),Fine-tuning CNN Image Retrieval with No Human Annotation, https://arxiv.org/abs/1711.02512. Exact squared-distance ranking,source-ID tie breaking,self exclusion and noninterpolated AP follow the source category-retrieval teaching protocol.
/// @see TensorDescriptorIndex, ScoreTensorRetrieval()
inline TensorRetrievalMetrics ScoreTensorRetrieval(const nn::Tensor &queries,
                                                   const std::vector<int> &query_labels,
                                                   const std::vector<std::int64_t> &query_ids,
                                                   const TensorDescriptorIndex &index,
                                                   const std::vector<int> &gallery_labels) {
  nn::NoGradGuard guard;
  if (queries.dim() != 2 || queries.shape()[1] != index.dimensions() ||
      (queries.dtype() != nn::DType::Float32 && queries.dtype() != nn::DType::Float64) ||
      queries.shape()[0] != std::int64_t(query_labels.size()) ||
      query_ids.size() != query_labels.size() || gallery_labels.size() != index.size())
    throw std::invalid_argument("aligned retrieval query/gallery metadata required");
  const auto zero = nn::zeros({}, nn::DType::Float64),
             count_zero = nn::zeros({}, nn::DType::Int64);
  if (query_labels.empty())
    return {zero, zero, zero, count_zero, count_zero};
  std::vector<nn::Tensor> aps, hits1, hits5, included;
  const auto labels =
      nn::from_vector(std::vector<std::int64_t>(gallery_labels.begin(), gallery_labels.end()));
  for (std::size_t i = 0; i < query_labels.size(); ++i) {
    const auto ranking = index.Rank(queries, std::int64_t(i), query_ids[i]);
    const auto relevant = nn::equal(nn::gather(labels, ranking.rows), double(query_labels[i]));
    const auto rel = relevant.to(nn::DType::Float64);
    const auto count = nn::count_nonzero(relevant);
    const auto precision =
        nn::cumsum(rel, 0) / nn::arange(1, rel.numel() + 1, 1, nn::DType::Float64);
    aps.push_back(nn::sum(precision * rel) /
                  nn::maximum(count.to(nn::DType::Float64), nn::ones({}, nn::DType::Float64)));
    hits1.push_back(
        nn::sum(nn::slice_axis(rel, 0, 0, std::min<std::int64_t>(1, rel.numel()))));
    hits5.push_back(
        nn::any(nn::slice_axis(relevant, 0, 0, std::min<std::int64_t>(5, rel.numel())))
            .to(nn::DType::Float64));
    included.push_back(nn::greater(count, 0.));
  }
  const auto valid = nn::stack(included).to(nn::DType::Float64);
  const auto eligible = nn::count_nonzero(valid);
  const auto denominator =
      nn::maximum(eligible.to(nn::DType::Float64), nn::ones({}, nn::DType::Float64));
  return {nn::sum(nn::stack(aps) * valid) / denominator,
          nn::sum(nn::stack(hits1) * valid) / denominator,
          nn::sum(nn::stack(hits5) * valid) / denominator, eligible,
          nn::full({}, double(query_labels.size()), nn::DType::Int64) - eligible};
}
/// @brief Form unit-normalized block-mean RGB reference descriptors without learning.
///
/// Subtract.5 and pool in Float64,cast block means to Float32,then normalize
/// using a Float64 squared-norm floor1e-12 and return Float32. This retains the
/// source reference precision/order. No learned parameters or optimizer.
///
/// @param data Aligned Float32[N,32,32,3] RGB pool.
/// @param block Positive exact divisor of32,default4;32 gives mean RGB.
/// @return Detached Float32[N,(32/block)^2*3] centered block-mean unit descriptors.
/// @throws std::invalid_argument For invalid descriptor/pool shapes,dtypes,metadata,nonfinite values,duplicate IDs or requested preview rows.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::models::TensorRetrievalPool data{n::zeros({4, 32, 32, 3}), {0, 0, 1, 1}, {10, 11, 12, 13}};
/// const auto features = n::models::RetrievalPixelDescriptors(data, 4);
/// @endcode
/// @par References
/// - Radenovic,Tolias and Chum(2018),Fine-tuning CNN Image Retrieval with No Human Annotation, https://arxiv.org/abs/1711.02512. Exact squared-distance ranking,source-ID tie breaking,self exclusion and noninterpolated AP follow the source category-retrieval teaching protocol.
/// @see TensorRetrievalRows(), TensorDescriptorIndex
inline nn::Tensor RetrievalPixelDescriptors(const TensorRetrievalPool &data, int block = 4) {
  nn::NoGradGuard guard;
  if (block < 1 || 32 % block ||
      data.images.shape() != nn::Shape{std::int64_t(data.size()), 32, 32, 3} ||
      data.images.dtype() != nn::DType::Float32 || data.ids.size() != data.size())
    throw std::invalid_argument("valid RGB pool and exact32px block divisor required");
  const auto dimensions = (32 / block) * (32 / block) * 3;
  if (data.size() == 0)
    return nn::zeros({0, dimensions});
  const auto means =
      nn::avg_pool2d(data.images.to(nn::DType::Float64) - .5, block, block, nn::Padding::Valid)
          .reshape({std::int64_t(data.size()), dimensions})
          .to(nn::DType::Float32)
          .to(nn::DType::Float64);
  return (means / nn::sqrt(nn::maximum(nn::sum(nn::square(means), {1}, true),
                                       nn::full({}, 1e-12, nn::DType::Float64))))
      .to(nn::DType::Float32);
}
/// @brief Render observed queries and exact ranked gallery neighbors with tensor borders.
///
/// Each row is query then ranked neighbors. Blue query borders,green relevant
/// and red irrelevant neighbor borders replace two-pixel edges. Images are
/// nearest-upsampled2x and byte-rounded with tensor arithmetic. Source IDs
/// exclude the same image before ranking. No fitting,forward or host pixel loops.
///
/// @param query Float32 RGB query pool with category/source-ID metadata.
/// @param gallery Float32 RGB gallery pool aligned exactly to index source IDs.
/// @param descriptors Query descriptor matrix for index ranking.
/// @param index Exact gallery descriptor index.
/// @param rows Valid query rows to display,default source eight-row selection.
/// @param top Positive neighbors per query,default5;enough eligible gallery rows required.
/// @return Detached Float32[rows.size(),64,64*(top+1),3] RGB[0,1] tiles.
/// @throws std::invalid_argument For invalid descriptor/pool shapes,dtypes,metadata,nonfinite values,duplicate IDs or requested preview rows.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::models::TensorRetrievalPool data{
///     n::zeros({6, 32, 32, 3}), {0, 0, 1, 1, 2, 2}, {10, 11, 12, 13, 14, 15}};
/// const auto features = n::ones({6, 2});
/// n::models::TensorDescriptorIndex index(features, data.ids);
/// const auto tiles = n::models::RetrievalPreviewTiles(data, data, features, index, {0, 2}, 3);
/// @endcode
/// @par References
/// - Radenovic,Tolias and Chum(2018),Fine-tuning CNN Image Retrieval with No Human Annotation, https://arxiv.org/abs/1711.02512. Exact squared-distance ranking,source-ID tie breaking,self exclusion and noninterpolated AP follow the source category-retrieval teaching protocol.
/// @see TensorDescriptorIndex, ScoreTensorRetrieval()
inline nn::Tensor
RetrievalPreviewTiles(const TensorRetrievalPool &query, const TensorRetrievalPool &gallery,
                      const nn::Tensor &descriptors, const TensorDescriptorIndex &index,
                      const std::vector<std::size_t> &rows = {0, 16, 32, 1, 17, 33, 2, 18},
                      int top = 5) {
  nn::NoGradGuard guard;
  if (rows.empty() || top < 1 || gallery.ids != index.source_ids())
    throw std::invalid_argument("nonempty retrieval preview and aligned gallery IDs required");
  TensorRetrievalRows(query, {});
  TensorRetrievalRows(gallery, {});
  const auto x = nn::arange(0, 64).reshape({1, 1, 64, 1}),
             y = nn::arange(0, 64).reshape({1, 64, 1, 1});
  const auto border =
      nn::logical_or(nn::logical_or(nn::less(x, 2.), nn::greater_equal(x, 62.)),
                     nn::logical_or(nn::less(y, 2.), nn::greater_equal(y, 62.)));
  const auto red = nn::tensor({1.f, 0.f, 0.f}).reshape({1, 1, 1, 3}),
             green = nn::tensor({0.f, 1.f, 0.f}).reshape({1, 1, 1, 3}),
             blue = nn::tensor({0.f, 0.f, 1.f}).reshape({1, 1, 1, 3});
  std::vector<nn::Tensor> result;
  for (const auto row : rows) {
    if (row >= query.size())
      throw std::invalid_argument("retrieval preview row out of bounds");
    const auto ranking = index.Rank(descriptors, std::int64_t(row), query.ids[row]);
    if (ranking.rows.numel() < top)
      throw std::invalid_argument("too few eligible gallery neighbors");
    const auto selected = nn::slice_axis(ranking.rows, 0, 0, top).to_vector<std::int64_t>();
    std::vector<nn::Tensor> tiles;
    for (int col = 0; col <= top; ++col) {
      const bool first = col == 0;
      const auto source = first ? std::int64_t(row) : selected[std::size_t(col - 1)];
      const auto &pool = first ? query : gallery;
      auto image =
          nn::resize_nearest(nn::slice_axis(pool.images, 0, source, source + 1), {64, 64});
      image = nn::floor(nn::clip(image, 0., 1.) * 255. + .5) / 255.;
      const auto colour = first                                                   ? blue
                          : pool.labels[std::size_t(source)] == query.labels[row] ? green
                                                                                  : red;
      tiles.push_back(nn::where(border, colour, image));
    }
    result.push_back(nn::concat(tiles, 2));
  }
  return nn::concat(result, 0);
}
} // namespace datamunge::nn::models
