#pragma once

/// @file feature_preprocessing.hpp
/// @brief Frozen column normalization, boundary buckets and stable categorical fingerprints.

#include <algorithm>
#include <cstdint>
#include <functional>
#include <cmath>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "datamunge/nn/detail/farmhash_fingerprint.hpp"
#include "datamunge/nn/ops.hpp"

namespace datamunge::nn {

/// @brief Frozen population moments for column standardization.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto scale = n::fit_feature_standardizer(n::ones({2,1})).scale.item(); // 1
/// @endcode
/// @see fit_feature_standardizer(), standardize_features()
struct FeatureStandardizer {
  Tensor mean; ///< Training column means [D].
  Tensor scale; ///< Training population standard deviations [D], with zero replaced by one.
};

/// @brief Fit frozen per-column population moments.
/// @param rows Finite nonempty floating matrix [N,D].
/// @return Detached means and positive scales.
/// @throws std::invalid_argument Invalid geometry or nonfinite inputs.
/// @throws std::overflow_error A moment exceeds the input dtype range.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// auto fit = n::fit_feature_standardizer(n::tensor({1.,3.},{2,1}));
/// auto mean = fit.mean.item(); // 2
/// @endcode
/// @see standardize_features()
inline FeatureStandardizer fit_feature_standardizer(const Tensor& rows) {
  if (!rows.defined() || rows.dim()!=2 || rows.size(0)==0 || rows.size(1)==0 ||
      (rows.dtype()!=DType::Float32 && rows.dtype()!=DType::Float64) ||
      !all(is_finite(rows)).item<bool>())
    throw std::invalid_argument("feature standardizer requires finite nonempty floating rows");
  const auto x=rows.detach();
  const auto center=mean(x,{0});
  const auto variance=mean(square(x-center),{0});
  if (!all(is_finite(center)).item<bool>() || !all(is_finite(variance)).item<bool>())
    throw std::overflow_error("feature standardizer moments overflowed");
  return {center,where(greater(variance,0.),sqrt(variance),1.)};
}

/// @brief Transform rows using the supplied frozen column moments.
/// @param rows Feature matrix [N,D].
/// @param fit Moments returned by fit_feature_standardizer().
/// @return Standardized values [N,D].
/// @throws std::invalid_argument Incompatible feature width.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// auto train=n::tensor({1.,3.},{2,1});
/// auto y=n::standardize_features(train,n::fit_feature_standardizer(train)); // [-1,1]
/// @endcode
/// @see fit_feature_standardizer()
inline Tensor standardize_features(const Tensor& rows,const FeatureStandardizer& fit) {
  if (!rows.defined() || rows.dim()!=2 || !fit.mean.defined() || !fit.scale.defined() ||
      fit.mean.shape()!=Shape{rows.size(1)} || fit.scale.shape()!=fit.mean.shape())
    throw std::invalid_argument("incompatible feature standardization geometry");
  return (rows-fit.mean)/fit.scale;
}

/// @brief Assign numeric values to intervals with left-closed boundaries.
/// @param values Finite CPU tensor of any shape.
/// @param boundaries Finite strictly increasing interval edges.
/// @return CPU Int64 IDs matching the input shape. Values equal to an edge enter the higher bucket.
/// @throws std::invalid_argument Invalid values or boundaries.
/// @par Example
/// @code
/// auto ids=dm::nn::bucketize_features(dm::nn::tensor({-1.,0.,1.}),{0.}); // [0,1,1]
/// @endcode
/// @see hash_feature_buckets()
inline Tensor bucketize_features(const Tensor& values,const std::vector<double>& boundaries) {
  const auto finite=[](double value){return std::isfinite(value);};
  if (!values.defined() || !std::ranges::all_of(boundaries,finite) ||
      std::adjacent_find(boundaries.begin(),boundaries.end(),std::greater_equal<double>{})!=boundaries.end())
    throw std::invalid_argument("invalid feature values or bucket boundaries");
  std::vector<std::int64_t> ids;
  for (double value : values.to_vector<double>()) {
    if (!finite(value)) throw std::invalid_argument("nonfinite bucket value");
    ids.push_back(std::upper_bound(boundaries.begin(),boundaries.end(),value)-boundaries.begin());
  }
  return from_vector(ids,values.shape());
}

/// @brief Compute Google's portable FarmHash Fingerprint64 on the exact input bytes.
/// @param value Bytes to fingerprint, including any embedded NUL.
/// @return Stable unsigned 64-bit fingerprint.
/// @par Example
/// @code
/// auto hash=dm::nn::feature_fingerprint64(""); // 11160318154034397263
/// @endcode
/// @par References
/// - Geoff Pike, Google FarmHash. https://github.com/google/farmhash
/// @see hash_feature_buckets()
inline std::uint64_t feature_fingerprint64(std::string_view value) {
  return farmhash_detail::Hash64(value.data(),value.size());
}

/// @brief Hash string categories into a fixed number of buckets.
/// @param values Raw strings, including values unseen in the fit data.
/// @param buckets Positive number of buckets.
/// @return CPU Int64 vector containing Fingerprint64 modulo buckets.
/// @throws std::invalid_argument Nonpositive bucket count.
/// @par Example
/// @code
/// auto ids=dm::nn::hash_feature_buckets({"oak","oak"},8); // repeated IDs
/// @endcode
/// @see feature_fingerprint64()
inline Tensor hash_feature_buckets(const std::vector<std::string>& values,std::int64_t buckets) {
  if (buckets<=0) throw std::invalid_argument("hash buckets must be positive");
  std::vector<std::int64_t> ids;
  ids.reserve(values.size());
  for (const auto& value : values)
    ids.push_back(static_cast<std::int64_t>(feature_fingerprint64(value)%static_cast<std::uint64_t>(buckets)));
  return from_vector(ids);
}
}  // namespace datamunge::nn
