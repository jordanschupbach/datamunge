#pragma once
/// @file retrieval_data.hpp
/// @brief Tensor CIFAR image ingress and metadata-only class-disjoint retrieval sampling.
#include "datamunge/nn/metric_learning.hpp"
#include <filesystem>
#include <fstream>
#include <iterator>
namespace datamunge::nn::models {
namespace nn = datamunge::nn;
/// @brief Own CIFAR RGB tensors and aligned category/source-row metadata.
///
/// Images retain shared tensor storage. Host vectors carry only discrete
/// labels and IDs;there is no host pixel mirror,optimizer or learned state.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::models::TensorRetrievalPool data{n::zeros({4, 32, 32, 3}), {0, 0, 1, 1}, {10, 11, 12, 13}};
/// @endcode
/// @par References
/// - Krizhevsky(2009),Learning Multiple Layers of Features from Tiny Images, https://www.cs.toronto.edu/~kriz/cifar.html. Class/image-disjoint CIFAR pools,index-only mt19937 sampling and hard-triplet teaching protocol follow the source example.
/// @see LoadTensorRetrievalData(), TensorRetrievalRows()
struct TensorRetrievalPool {
  nn::Tensor images;             ///< Float32[N,32,32,3] RGB in[0,1].
  std::vector<int> labels;       ///< N category IDs as sampling/relevance metadata.
  std::vector<std::int64_t> ids; ///< N unique original source IDs when loaded.

  /// @brief Return the number of metadata rows without reading pixels.
  /// @return Category-vector size;aligned tensors are a caller precondition.
  [[nodiscard]] std::size_t size() const {
    return labels.size();
  }
};
/// @brief Own disjoint fitting and validation/test query/gallery retrieval pools.
///
/// This value container does not itself validate disjointness. The loader
/// checks original IDs across every pool.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::models::TensorRetrievalPool data{n::zeros({4, 32, 32, 3}), {0, 0, 1, 1}, {10, 11, 12, 13}};
/// n::models::TensorRetrievalSplits splits{data, data, data, data, data};
/// @endcode
/// @par References
/// - Krizhevsky(2009),Learning Multiple Layers of Features from Tiny Images, https://www.cs.toronto.edu/~kriz/cifar.html. Class/image-disjoint CIFAR pools,index-only mt19937 sampling and hard-triplet teaching protocol follow the source example.
/// @see LoadTensorRetrievalData(), TensorRetrievalRows()
struct TensorRetrievalSplits {
  TensorRetrievalPool fit;                ///<160 development images,classes0..4,32/class.
  TensorRetrievalPool validation_query;   ///<16 development images,classes5..6,8/class.
  TensorRetrievalPool validation_gallery; ///<32 other development images,classes5..6,16/class.
  TensorRetrievalPool test_query;         ///<48 official test images,classes7..9,16/class.
  TensorRetrievalPool test_gallery; ///<96 other official test images,classes7..9,32/class.
};
/// @brief Gather aligned retrieval RGB,label and source-ID rows.
///
/// Image gathering uses tensors and retains input gradients. Host copies
/// select only row/class/ID metadata;no pixel arithmetic or fitting.
///
/// @param input Float32[N,32,32,3] images and N labels/IDs.
/// @param rows Valid requested pool-row metadata,duplicates/empty allowed.
/// @return Selected images and aligned labels/IDs,in requested order.
/// @throws std::invalid_argument For invalid geometry,dtype,metadata,row indices,class balance or missing/corrupt CIFAR files.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::models::TensorRetrievalPool data{n::zeros({4, 32, 32, 3}), {0, 0, 1, 1}, {10, 11, 12, 13}};
/// const auto selected = n::models::TensorRetrievalRows(data, {2, 0, 2});
/// @endcode
/// @par References
/// - Krizhevsky(2009),Learning Multiple Layers of Features from Tiny Images, https://www.cs.toronto.edu/~kriz/cifar.html. Class/image-disjoint CIFAR pools,index-only mt19937 sampling and hard-triplet teaching protocol follow the source example.
/// @see LoadTensorRetrievalData(), TensorRetrievalRows()
inline TensorRetrievalPool TensorRetrievalRows(const TensorRetrievalPool &input,
                                               const std::vector<std::size_t> &rows) {
  if (input.images.shape() != nn::Shape{std::int64_t(input.size()), 32, 32, 3} ||
      input.images.dtype() != nn::DType::Float32 || input.ids.size() != input.size())
    throw std::invalid_argument("invalid retrieval pool alignment");
  TensorRetrievalPool result;
  std::vector<std::int64_t> indices;
  for (const auto row : rows) {
    if (row >= input.size())
      throw std::invalid_argument("retrieval row out of bounds");
    indices.push_back(std::int64_t(row));
    result.labels.push_back(input.labels[row]);
    result.ids.push_back(input.ids[row]);
  }
  result.images = nn::gather(input.images, nn::from_vector(indices));
  return result;
}
/// @brief Private raw-byte ingress for original CIFAR binary records.
namespace retrieval_data_detail {
/// @brief Own raw source bytes and class metadata before selected-row tensor conversion.
struct CifarRecords {
  std::vector<std::vector<std::uint8_t>> files; ///< Complete10000-record source files.
  std::vector<int> labels;                      ///< One original category per source record.

  /// @brief Access a valid raw record with byte lifetime tied to files storage.
  /// @param row Valid source record index.
  /// @return Pointer to label plus3072 planar channel bytes.
  const std::uint8_t *Record(std::size_t row) const {
    return files.at(row / 10000).data() + (row % 10000) * 3073;
  }
};
/// @brief Read whole CIFAR batch bytes and validate integer label metadata.
/// @param paths Original binary files,each exactly10000 records.
/// @return Byte storage and aligned original category metadata.
/// @throws std::runtime_error For unreadable files.
/// @throws std::invalid_argument For incomplete records or category outside0..9.
inline CifarRecords Read(const std::vector<std::filesystem::path> &paths) {
  CifarRecords result;
  for (const auto &path : paths) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream)
      throw std::runtime_error("cannot read CIFAR retrieval source " + path.string());
    result.files.emplace_back(std::istreambuf_iterator<char>(stream),
                              std::istreambuf_iterator<char>());
    const auto &bytes = result.files.back();
    if (bytes.size() != 10000 * 3073)
      throw std::invalid_argument("complete10000-record CIFAR source required");
    for (std::size_t start = 0; start < bytes.size(); start += 3073) {
      if (bytes[start] >= 10)
        throw std::invalid_argument("CIFAR category out of range");
      result.labels.push_back(bytes[start]);
    }
  }
  return result;
}
} // namespace retrieval_data_detail
/// @brief Load reproducible class/image-disjoint pools from official CIFAR-10 binaries.
///
/// Read original byte records and class metadata;shuffle every development
/// class bucket then every test bucket using std::mt19937(211). Preserve the
/// source index draw order within the same C++ standard library. Selected planar
/// bytes enter Int32 tensors before transpose,Float32 conversion and /255.
/// Source IDs are train row IDs or100000+test row IDs;all pools are disjoint.
/// No downloading,augmentation,fitting,host pixel normalization or full float
/// source-dataset materialization occurs here.
///
/// @param root Directory containing data_batch_1..5.bin and test_batch.bin.
/// @return Five pools:160 fitting,16/32 validation query/gallery,48/96 test query/gallery.
/// @throws std::invalid_argument For invalid geometry,dtype,metadata,row indices,class balance or missing/corrupt CIFAR files.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// std::filesystem::path cache;
/// if (const auto *p = std::getenv("DATAMUNGE_DATA_DIR"))
///   cache = p;
/// else if (const auto *p = std::getenv("XDG_CACHE_HOME"))
///   cache = std::filesystem::path(p) / "datamunge/datasets";
/// else
///   cache = std::filesystem::path(std::getenv("HOME")) / ".cache/datamunge/datasets";
/// const auto data = n::models::LoadTensorRetrievalData(cache / "cifar10");
/// @endcode
/// @par References
/// - Krizhevsky(2009),Learning Multiple Layers of Features from Tiny Images, https://www.cs.toronto.edu/~kriz/cifar.html. Class/image-disjoint CIFAR pools,index-only mt19937 sampling and hard-triplet teaching protocol follow the source example.
/// @see LoadTensorRetrievalData(), TensorRetrievalRows()
inline TensorRetrievalSplits LoadTensorRetrievalData(const std::filesystem::path &root) {
  std::vector<std::filesystem::path> paths;
  for (int i = 1; i <= 5; ++i)
    paths.push_back(root / ("data_batch_" + std::to_string(i) + ".bin"));
  const auto development = retrieval_data_detail::Read(paths),
             test = retrieval_data_detail::Read({root / "test_batch.bin"});
  std::array<std::vector<std::size_t>, 10> dev_rows, test_rows;
  for (std::size_t i = 0; i < development.labels.size(); ++i)
    dev_rows[std::size_t(development.labels[i])].push_back(i);
  for (std::size_t i = 0; i < test.labels.size(); ++i)
    test_rows[std::size_t(test.labels[i])].push_back(i);
  std::mt19937 random(211);
  for (auto &rows : dev_rows)
    std::shuffle(rows.begin(), rows.end(), random);
  for (auto &rows : test_rows)
    std::shuffle(rows.begin(), rows.end(), random);
  /// @brief Select discrete raw records before tensor planar-to-RGB normalization.
  /// @param source Original checked binary records.
  /// @param buckets Seeded shuffled class-row metadata.
  /// @param classes Included original categories.
  /// @param begin First within-class bucket offset.
  /// @param each Positive rows per included category.
  /// @param id_offset Zero for development,100000 for official test.
  /// @return Selected Float32 RGB tensors and aligned metadata.
  auto make = [](const retrieval_data_detail::CifarRecords &source,
                 const std::array<std::vector<std::size_t>, 10> &buckets,
                 const std::vector<int> &classes, int begin, int each,
                 std::int64_t id_offset) {
    TensorRetrievalPool result;
    std::vector<std::int32_t> bytes;
    bytes.reserve(classes.size() * std::size_t(each) * 3072);
    for (const auto label : classes)
      for (int i = begin; i < begin + each; ++i) {
        const auto row = buckets[std::size_t(label)].at(std::size_t(i));
        result.labels.push_back(label);
        result.ids.push_back(id_offset + std::int64_t(row));
        const auto record = source.Record(row);
        bytes.insert(bytes.end(), record + 1, record + 3073);
      }
    const auto planar = nn::from_vector(bytes, {std::int64_t(result.size()), 3, 32, 32});
    result.images = nn::transpose(planar, {0, 2, 3, 1}).to(nn::DType::Float32) / 255.;
    return result;
  };
  TensorRetrievalSplits result{make(development, dev_rows, {0, 1, 2, 3, 4}, 0, 32, 0),
                               make(development, dev_rows, {5, 6}, 0, 8, 0),
                               make(development, dev_rows, {5, 6}, 8, 16, 0),
                               make(test, test_rows, {7, 8, 9}, 0, 16, 100000),
                               make(test, test_rows, {7, 8, 9}, 16, 32, 100000)};
  std::set<std::int64_t> used;
  for (const auto *pool : {&result.fit, &result.validation_query, &result.validation_gallery,
                           &result.test_query, &result.test_gallery})
    for (const auto id : pool->ids)
      if (!used.insert(id).second)
        throw std::invalid_argument("retrieval source IDs overlap");
  return result;
}
/// @brief Create eight class-balanced20-image fitting batches with no repeated epoch rows.
///
/// Metadata-only sampling. Buckets shuffle in class order,then each batch
/// shuffles;source std::mt19937 draw order is preserved in one standard library.
/// No pixel access,fitting or source mutation.
///
/// @param fitting Pool metadata with32 rows in each category0..4.
/// @param epoch Integer epoch tag;seed is uint32(223)+uint32(epoch).
/// @return Eight shuffled batches,each four rows per fitting class.
/// @throws std::invalid_argument For invalid geometry,dtype,metadata,row indices,class balance or missing/corrupt CIFAR files.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::models::TensorRetrievalPool data;
/// for (int c = 0; c < 5; ++c)
///   for (int i = 0; i < 32; ++i)
///     data.labels.push_back(c);
/// const auto batches = n::models::RetrievalBatches(data, 1);
/// @endcode
/// @par References
/// - Krizhevsky(2009),Learning Multiple Layers of Features from Tiny Images, https://www.cs.toronto.edu/~kriz/cifar.html. Class/image-disjoint CIFAR pools,index-only mt19937 sampling and hard-triplet teaching protocol follow the source example.
/// @see LoadTensorRetrievalData(), TensorRetrievalRows()
inline std::vector<std::vector<std::size_t>>
RetrievalBatches(const TensorRetrievalPool &fitting, int epoch) {
  std::array<std::vector<std::size_t>, 5> buckets;
  for (std::size_t i = 0; i < fitting.labels.size(); ++i) {
    if (fitting.labels[i] < 0 || fitting.labels[i] >= 5)
      throw std::invalid_argument("fit-only category labels required");
    buckets[std::size_t(fitting.labels[i])].push_back(i);
  }
  std::mt19937 random(std::uint32_t(223) + std::uint32_t(epoch));
  for (auto &rows : buckets) {
    if (rows.size() != 32)
      throw std::invalid_argument("32 fitting images per class required");
    std::shuffle(rows.begin(), rows.end(), random);
  }
  std::vector<std::vector<std::size_t>> result;
  for (int batch = 0; batch < 8; ++batch) {
    std::vector<std::size_t> rows;
    for (const auto &bucket : buckets)
      rows.insert(rows.end(), bucket.begin() + 4 * batch, bucket.begin() + 4 * batch + 4);
    std::shuffle(rows.begin(), rows.end(), random);
    result.push_back(rows);
  }
  return result;
}

/// @brief Select a farthest positive and nearest negative for every anchor.
///
/// Frozen Float64 tensor distances select strict argmax/argmin after self
/// exclusion. No host distance arithmetic,fallback,gradient or fitting.
/// The caller recomputes its selected triplet objective before optimization.
///
/// @param features Finite Float32/64[B,D],D positive,empty B allowed.
/// @param labels B integer class IDs;every row needs another same-class row and a different class.
/// @return B anchor-major triplets,first row index wins all distance ties.
/// @throws std::invalid_argument For invalid geometry,dtype,metadata,row indices,class balance or missing/corrupt CIFAR files.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto triplets =
///     n::models::RetrievalHardTriplets(n::tensor({0., .3, .1, .9}, {4, 1}), {0, 0, 1, 1});
/// @endcode
/// @par References
/// - Krizhevsky(2009),Learning Multiple Layers of Features from Tiny Images, https://www.cs.toronto.edu/~kriz/cifar.html. Class/image-disjoint CIFAR pools,index-only mt19937 sampling and hard-triplet teaching protocol follow the source example.
/// @see MetricDistanceMatrix(), MeanTriplet()
inline std::vector<TripletIndex> RetrievalHardTriplets(const nn::Tensor &features,
                                                       const std::vector<int> &labels) {
  nn::NoGradGuard guard;
  if (features.dim() != 2 || features.shape()[0] != std::int64_t(labels.size()))
    throw std::invalid_argument("aligned hard-mining matrix required");
  const auto distances = MetricDistanceMatrix(features);
  if (labels.empty())
    return {};
  const auto n = features.shape()[0];
  const auto y = nn::from_vector(std::vector<std::int64_t>(labels.begin(), labels.end()));
  const auto index = nn::arange(0, n, 1, nn::DType::Int64);
  const auto same = nn::equal(nn::expand_dims(y, 1), nn::expand_dims(y, 0));
  const auto positive = nn::logical_and(
      same, nn::not_equal(nn::expand_dims(index, 1), nn::expand_dims(index, 0)));
  const auto negative = nn::logical_not(same);
  if (!nn::all(nn::logical_and(nn::any(positive, {1}), nn::any(negative, {1}))).item<bool>())
    throw std::invalid_argument("each anchor needs a positive and negative");
  const auto p = nn::argmax(
                     nn::where(positive, distances, nn::full({}, -1., nn::DType::Float64)), 1)
                     .to_vector<std::int64_t>(),
             q = nn::argmin(nn::where(negative, distances,
                                      nn::full({}, std::numeric_limits<double>::infinity(),
                                               nn::DType::Float64)),
                            1)
                     .to_vector<std::int64_t>();
  std::vector<TripletIndex> result;
  for (std::int64_t a = 0; a < n; ++a)
    result.push_back({int(a), int(p[std::size_t(a)]), int(q[std::size_t(a)])});
  return result;
}
} // namespace datamunge::nn::models
