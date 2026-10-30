#pragma once
/// @file time_series_data.hpp
/// @brief Checked original GunPoint observations,tensor fitting statistics and reference parity helpers.
#include "datamunge/nn/io.hpp"
#include "datamunge/nn/sequence_classification.hpp"
#include "datamunge/nn/time_series_models.hpp"
#include <filesystem>
#include <map>
#include <set>
namespace datamunge::nn::models {
namespace nn = datamunge::nn;
/// @brief Own raw temporal signals,original labels and exact source-record identities.
///
/// Generic[N,T,C] temporal container shared with sensor/causal examples.
/// Only signals enter forward models;labels/source IDs remain metadata.
/// Copying shares tensor storage and retained graph handles.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// m::SeriesData d{n::zeros({2, 150, 1}), n::zeros({2}, n::DType::Int32),
///                 n::tensor({0, 1}).to(n::DType::Int32)};
/// @endcode
/// @par References
/// - Official GunPoint source, https://www.timeseriesclassification.com/description.php?Dataset=GunPoint. Custom class-stratified20% holdout only within official50-row training pool;official150-row test remains unchanged. Source-record separation does not establish actor separation.
/// @see SeriesRows(), LoadSeries()
struct SeriesData {
  nn::Tensor signals;    ///< Float32[N,T,C] original observed signal values.
  nn::Tensor labels;     ///< Int32[N] zero-based true class metadata.
  nn::Tensor source_ids; ///< Int32[N] exact original source-record identities.

  /// @brief Return the number of aligned label observations.
  /// @return Leading label dimension;requires initialized labels.
  [[nodiscard]] std::int64_t size() const {
    return labels.size(0);
  }
};
/// @brief Own original training-holdout/test signals and compact fitting normalization metadata.
///
/// Pool order is fitting,validation,test. Compact width16 follows source.
/// Mean/std are population statistics of fitting raw values only,roundedFloat32.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto d = m::LoadSeries("build/cnn-datasets/time_series");
/// @endcode
/// @par References
/// - Official GunPoint source, https://www.timeseriesclassification.com/description.php?Dataset=GunPoint. Custom class-stratified20% holdout only within official50-row training pool;official150-row test remains unchanged. Source-record separation does not establish actor separation.
/// @see LoadSeries()
struct SeriesSplits {
  std::vector<SeriesData> pools; ///< Three ordered fitting/validation/test tensor containers.
  SeriesConfig config; ///< Compact width16 and fitting-only mean/std creation defaults.
};
/// @brief Gather aligned raw temporal values,labels and source IDs with tensor indexing.
///
/// No statistics,target transformations or numerical host copies.
///
/// @param d Generic aligned temporal observation container.
/// @param rows Bounded leading-row metadata.
/// @return Requested tensor rows in supplied order,retaining dtype and graph.
/// @throws std::invalid_argument For invalid leading axes or out-of-range row.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// m::SeriesData d{n::zeros({2, 150, 1}), n::zeros({2}, n::DType::Int32),
///                 n::tensor({0, 1}).to(n::DType::Int32)};
/// const auto row = m::SeriesRows(d, {1});
/// @endcode
/// @see SeriesData

inline SeriesData SeriesRows(const SeriesData &d, const std::vector<std::int64_t> &rows) {
  std::vector<std::int64_t> ids;
  for (auto row : rows) {
    if (row <0 || row >= d.size())
      throw std::invalid_argument("bounded temporal row required");
    ids.push_back(row);
  }
  return {nn::take_rows(d.signals, ids), nn::take_rows(d.labels, ids),
          nn::take_rows(d.source_ids, ids)};
}
/// @brief Load raw original GunPoint rows and compute only fitting-population normalization.
///
/// Require finite Float32[N,150,1],Int32 labels/source IDs,binary labels and
/// unique source records across splits. Fit/validation IDs<50,test IDs>=50.
/// Tensor Double population mean/variance over fitting observations only;
/// positive nonconstant variance required. No downloads,augmentation or fitting.
///
/// @param root Directory containing time_series.npz and original-source manifest.
/// @return Three checked pools and compact width16 with Float32 fitting mean/std.
/// @throws std::invalid_argument For invalid source shapes/values/IDs or constant fitting population.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto d = m::LoadSeries("build/cnn-datasets/time_series");
/// @endcode
/// @par References
/// - Official GunPoint source, https://www.timeseriesclassification.com/description.php?Dataset=GunPoint. Custom class-stratified20% holdout only within official50-row training pool;official150-row test remains unchanged. Source-record separation does not establish actor separation.
/// @see SeriesCnn(), SeriesRows()

inline SeriesSplits LoadSeries(const std::filesystem::path &root) {
  nn::NoGradGuard guard;
  std::map<std::string, nn::Tensor> arrays;
  for (const auto &[name, value] : nn::load_npz((root / "time_series.npz").string()))
    if (!arrays.emplace(name, value).second)
      throw std::invalid_argument("unique temporal array names required");
  SeriesSplits out;
  std::set<int> seen;
  for (const std::string name : {"fit", "validation", "test"}) {
    SeriesData d{arrays.at(name + "_signals"), arrays.at(name + "_labels"),
                 arrays.at(name + "_source_ids")};
    if (d.signals.dim() != 3 || d.size() < 1 || d.signals.dtype() != nn::DType::Float32 ||
        d.signals.shape() != nn::Shape{d.size(), 150, 1} ||
        d.labels.dtype() != nn::DType::Int32 || d.labels.shape() != nn::Shape{d.size()} ||
        d.source_ids.dtype() != nn::DType::Int32 || d.source_ids.shape() != d.labels.shape() ||
        !nn::all(nn::is_finite(d.signals)).item<bool>() ||
        !nn::all(nn::logical_and(nn::greater_equal(d.labels, 0.), nn::less(d.labels, 2.)))
             .item<bool>())
      throw std::invalid_argument("finite aligned original GunPoint values required");
    for (int id : d.source_ids.to_vector<int>())
      if (id < 0 || id >= 200 || (name == "test" ? id < 50 : id >= 50) ||
          !seen.insert(id).second)
        throw std::invalid_argument(
            "unique original official GunPoint source boundaries required");
    out.pools.push_back(d);
  }
  const auto x = out.pools[0].signals.to(nn::DType::Float64), mean = nn::mean(x),
             variance = nn::mean(nn::square(x - mean));
  if (variance.item() <= 1e-12)
    throw std::invalid_argument("nonconstant fitting temporal population required");
  out.config = {16, float(mean.item()), float(nn::sqrt(variance).item())};
  return out;
}
/// @brief Compute source raw-distance1NN or a class-centroid parity oracle using tensors.
///
/// Nearest memorizes fitting observations and has no optimized numeric leaves.
/// Source nearest differences roundFloat32 before Double square reduction,
/// first fitting row wins distance ties. Centroids use Double population sums
/// and negative mean squared distances,roundedFloat32 at output. Reports fit
/// centroid fields with explicit SGD;closed-form branch is a parity oracle only.
/// No global signal normalization or test annotations enter either reference.
///
/// @param fit Fitting-only Float32[N,150,1] raw observations with both binary classes.
/// @param test Float32[M,150,1] observations;test labels are ignored.
/// @param centroid False returns parameter-free1NN pseudo-logits;true is closed-form centroid oracle.
/// @return Float32[M,2] onehot nearest labels or negative Double class-mean squared distances.
/// @throws std::invalid_argument For invalid raw signal geometry,label bounds or missing fitting class.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// m::SeriesData f{n::concat({n::zeros({1, 150, 1}), n::ones({1, 150, 1})}, 0),
///                 n::tensor({0, 1}).to(n::DType::Int32), n::zeros({2}, n::DType::Int32)};
/// const auto z = m::SeriesReference(f, f, false);
/// @endcode
/// @par References
/// - Official GunPoint source, https://www.timeseriesclassification.com/description.php?Dataset=GunPoint. Custom class-stratified20% holdout only within official50-row training pool;official150-row test remains unchanged. Source-record separation does not establish actor separation.
/// @see SeriesRows(), ScoreSequenceClassifier()

inline nn::Tensor SeriesReference(const SeriesData &fit, const SeriesData &test,
                                  bool centroid) {
  nn::NoGradGuard guard;
  if (fit.signals.dim() != 3 || test.signals.dim() != 3 ||
      fit.signals.shape() != nn::Shape{fit.size(), 150, 1} ||
      test.signals.shape() != nn::Shape{test.signals.size(0), 150, 1} || fit.size() < 1 ||
      test.signals.size(0) < 1)
    throw std::invalid_argument("aligned150sample raw temporal reference inputs required");
  const auto labels = nn::one_hot(fit.labels, 2, 1., 0., nn::DType::Float64),
             counts = nn::sum(labels, {0});
  if (!nn::all(nn::greater(counts, 0.)).item<bool>() ||
      !nn::all(nn::logical_and(nn::greater_equal(fit.labels, 0.), nn::less(fit.labels, 2.)))
           .item<bool>())
    throw std::invalid_argument("both bounded fitting temporal classes required");
  if (centroid) {
    const auto weights = labels.reshape({fit.size(), 2, 1, 1}),
               centers =
                   nn::sum(
                       weights *
                           nn::expand_dims(
                               fit.signals.to(nn::DType::Float32).to(nn::DType::Float64), 1),
                       {0}) /
                   counts.reshape({2, 1, 1}),
               diff = nn::expand_dims(
                          test.signals.to(nn::DType::Float32).to(nn::DType::Float64), 1) -
                      nn::expand_dims(centers, 0);
    return (-nn::mean(nn::square(diff), {2, 3})).to(nn::DType::Float32);
  }
  const auto diff = (nn::expand_dims(test.signals.to(nn::DType::Float32), 1) -
                     nn::expand_dims(fit.signals.to(nn::DType::Float32), 0))
                        .to(nn::DType::Float64),
             distance = nn::sum(nn::square(diff), {2, 3}), nearest = nn::argmin(distance, 1);
  return nn::one_hot(nn::gather(fit.labels, nearest, 0), 2);
}
/// @brief Return temporal observations with all raw time samples reversed.
///
/// A declared observed-input diagnostic,not a fitted augmentation or new source.
///
/// @param d Original temporal container;labels/source IDs are preserved.
/// @return Copy sharing metadata with signals axis1 reversed/contiguous.
/// @throws std::invalid_argument For signals without a temporal axis.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// m::SeriesData d{n::zeros({2, 150, 1}), n::zeros({2}, n::DType::Int32),
///                 n::tensor({0, 1}).to(n::DType::Int32)};
/// const auto reversed = m::Reversed(d);
/// @endcode
/// @see SeriesCnn()

inline SeriesData Reversed(const SeriesData &d) {
  SeriesData out = d;
  out.signals = nn::reverse(d.signals, {1}).contiguous();
  return out;
}
} // namespace datamunge::nn::models
