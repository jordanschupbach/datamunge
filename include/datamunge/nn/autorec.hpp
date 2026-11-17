#pragma once
/// @file autorec.hpp
/// @brief Explicit tensor AutoRec components and original MovieLens warm-user ingress.
#include <datamunge/nn/io.hpp>
#include <datamunge/nn/layers.hpp>
#include <datamunge/nn/ops.hpp>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <map>
namespace datamunge::nn::models {
/// @brief Private validation and checked archive lookup for rating records.
namespace autorec_detail {
/// @brief Reject invalid rating/archive invariants.
/// @param ok Required predicate.
/// @param message Diagnostic.
/// @throws std::invalid_argument If ok is false.
inline void Require(bool ok, const std::string& message) {
  if (!ok)
    throw std::invalid_argument(message);
}
/// @brief Check one finite archive scalar against its original source value.
/// @param value Prepared value.
/// @param expected Original rating or input.
/// @param tolerance Absolute bound.
/// @param message Diagnostic.
/// @throws std::invalid_argument For nonfinite or inconsistent values.
inline void Near(double value, double expected, double tolerance, const std::string& message) {
  Require(std::isfinite(value) && std::abs(value - expected) <= tolerance, message);
}
/// @brief Borrow one required NPZ tensor until its archive container is destroyed.
/// @param arrays Owned archive entries.
/// @param key Unique required name.
/// @return A tensor handle retaining the archive storage.
/// @throws std::invalid_argument If missing or duplicated.
inline Tensor Find(const NamedTensors& arrays, const std::string& key) {
  Tensor result;
  for (const auto& [name, value] : arrays)
    if (name == key) {
      Require(!result.defined(), "duplicate AutoRec array");
      result = value;
    }
  Require(result.defined(), "missing AutoRec array: " + key);
  return result;
}
} // namespace autorec_detail
/// @brief Compute observed-entry squared error with an observed-only denominator.
///
/// Floating CPU tensors must have identical nonempty shapes and dtypes. Mask is
/// finite binary with positive support. Unobserved target/prediction values are
/// selected away before subtraction, so hidden NaNs cannot contaminate the loss.
/// Observed predictions/targets must be finite. No mutation or optimizer; first
/// and higher derivatives remain live, with exactly zero hidden-coordinate paths.
/// @param prediction Model output in caller-selected rating units.
/// @param target Aligned targets in those same units.
/// @param mask Binary observation indicator in the prediction dtype.
/// @return Live scalar in the prediction dtype.
/// @throws std::invalid_argument For geometry, dtype, mask or observed-finiteness errors.
/// @par Example
/// @code
/// namespace n=dm::nn;
/// const auto p=n::tensor({.2,.9,.4}).requires_grad_();
/// const auto loss=n::models::ObservedRatingMse(p,n::tensor({.1,99.,.5}),n::tensor({1.,0.,1.}));
/// @endcode
/// @par References
/// - Sedhain et al. (2015), AutoRec, https://doi.org/10.1145/2740908.2742726 .
/// @see AutoRec(), RatingErrors()
inline Tensor
ObservedRatingMse(const Tensor& prediction, const Tensor& target, const Tensor& mask) {
  autorec_detail::Require(
      prediction.defined() && target.defined() && mask.defined(),
      "defined rating tensors required");
  autorec_detail::Require(
      prediction.device().backend == compute::Backend::CPU &&
          target.device().backend == compute::Backend::CPU &&
          mask.device().backend == compute::Backend::CPU && prediction.numel() > 0 &&
          prediction.shape() == target.shape() && prediction.shape() == mask.shape() &&
          prediction.dtype() == target.dtype() && prediction.dtype() == mask.dtype() &&
          (prediction.dtype() == DType::Float32 || prediction.dtype() == DType::Float64),
      "aligned floating rating tensors required");
  autorec_detail::Require(
      all(logical_or(equal(mask, 0.), equal(mask, 1.))).item<bool>() && sum(mask).item() > 0,
      "positive binary observation support required");
  const auto active = equal(mask, 1.);
  const auto p = where(active, prediction, 0.), t = where(active, target, 0.);
  autorec_detail::Require(
      all(is_finite(p)).item<bool>() && all(is_finite(t)).item<bool>(),
      "finite observed ratings required");
  return sum(square(p - t)) / sum(mask);
}
/// @brief Own live hidden-code and normalized rating-prediction tensor handles.
///
/// Independent handles retain their storage/derivatives beyond the input scope.
/// No fitting state, mutable cache or optimizer is owned by this record.
/// @par Example
/// @code
/// dm::nn::models::AutoRecOutput output;
/// @endcode
/// @par References
/// - Sedhain et al. (2015), AutoRec, https://doi.org/10.1145/2740908.2742726 .
/// @see AutoRec()
struct AutoRecOutput {
  Tensor code;       ///< Live CPU[N,32] sigmoid encoding.
  Tensor prediction; ///< Live CPU[N,256] unconstrained normalized ratings.
};
/// @brief Apply the original 256→32 sigmoid→256 linear warm-user AutoRec preset.
///
/// Input is finite floating CPU[N,256], N positive, containing training ratings
/// divided by five and zero for withheld/unknown entries. Parameters are created
/// lazily as encoder/W,b and decoder/W,b in the caller's store; predictions need
/// no targets or masks. All four tensors stay live. Caller owns fitting and L2.
/// @param parameters Caller-owned store, seeded before first use.
/// @param input Training-only normalized rows.
/// @return Live hidden code and unconstrained normalized predictions.
/// @throws std::invalid_argument For invalid shape/dtype/nonfinite input.
/// @par Example
/// @code
/// namespace n=dm::nn;n::ParameterStore p(4603);
/// const auto out=n::models::AutoRec(p,n::zeros({2,256}));
/// @endcode
/// @par References
/// - Sedhain et al. (2015), AutoRec, https://doi.org/10.1145/2740908.2742726 .
/// @see ObservedRatingMse(), LoadAutoRec()
inline AutoRecOutput AutoRec(ParameterStore& parameters, const Tensor& input) {
  autorec_detail::Require(
      input.defined() && input.device().backend == compute::Backend::CPU && input.dim() == 2 &&
          input.size(0) > 0 && input.size(1) == 256 &&
          (input.dtype() == DType::Float32 || input.dtype() == DType::Float64) &&
          all(is_finite(input)).item<bool>(),
      "finite AutoRec CPU[N,256] input required");
  AutoRecOutput out;
  out.code       = dense(parameters, "encoder", input, 32, Activation::Sigmoid);
  out.prediction = dense(parameters, "decoder", out.code, 256);
  return out;
}
/// @brief Own the original 199-user/256-item MovieLens teaching archive.
///
/// Loaded handles own detached CPU storage beyond the archive's lifetime. Masks
/// partition observed ratings; zero unknown targets never become negatives.
/// @par Example
/// @code
/// dm::nn::models::AutoRecData data;
/// @endcode
/// @par References
/// - GroupLens, MovieLens 100K, https://grouplens.org/datasets/movielens/100k/ .
/// @see LoadAutoRec(), AutoRec()
struct AutoRecData {
  Tensor input;      ///< Float32[199,256], training stars /5, held-out/unknown zero.
  Tensor ratings;    ///< Float32[199,256], original stars1..5 or unknown0.
  Tensor train;      ///< Float32[199,256],8079 binary fitting observations.
  Tensor validation; ///< Float32[199,256],898 binary validation observations.
  Tensor test;       ///< Float32[199,256],898 binary test observations.
  Tensor users;      ///< Int32[199], ascending original warm-user IDs.
  Tensor items;      ///< Int32[256], original item IDs1..256.
};
/// @brief Load and verify every prepared rating against original MovieLens u.data.
///
/// Requires ratings.npz and all100000 original unique records. Selects users1..256
/// with at least10 ratings on items1..256; retains199 rows. Validates every source
/// ID, rating, disjoint binary mask, per-user10% holdout count, training-only input
/// and each item's fitting support. Never prepares/downloads data implicitly.
/// @param root Native autorec dataset directory.
/// @return Owning detached CPU record with the documented fixed geometry.
/// @throws std::invalid_argument For missing files, malformed/changed source data or schema.
/// @par Example
/// @code
/// const auto root=std::filesystem::path(dm::datasets::data_home())/"autorec";
/// // After dm::datasets::prepare_example_data("autorec",root.string()):
/// // const auto data=dm::nn::models::LoadAutoRec(root);
/// @endcode
/// @par References
/// - GroupLens, MovieLens 100K, https://grouplens.org/datasets/movielens/100k/ .
/// @see AutoRecData, ObservedRatingMse()
inline AutoRecData LoadAutoRec(const std::filesystem::path& root) {
  const auto file = root / "ratings.npz";
  autorec_detail::Require(
      std::filesystem::exists(file),
      file.string() + " missing; prepare the native autorec dataset");
  const auto  a = nn::load_npz(file.string());
  AutoRecData d{
      autorec_detail::Find(a, "input").contiguous(),
      autorec_detail::Find(a, "ratings").contiguous(),
      autorec_detail::Find(a, "train_mask").contiguous(),
      autorec_detail::Find(a, "validation_mask").contiguous(),
      autorec_detail::Find(a, "test_mask").contiguous(),
      autorec_detail::Find(a, "user_ids").contiguous(),
      autorec_detail::Find(a, "item_ids").contiguous()};
  for (const auto& t : {d.input, d.ratings, d.train, d.validation, d.test})
    autorec_detail::Require(
        t.dtype() == nn::DType::Float32 && t.shape() == nn::Shape({199, 256}),
        "exact199users256itemsratingsschema");
  autorec_detail::Require(
      d.users.dtype() == nn::DType::Int32 && d.users.shape() == nn::Shape({199}) &&
          d.items.dtype() == nn::DType::Int32 && d.items.shape() == nn::Shape({256}),
      "originaluser/itemidentityvectors");
  std::ifstream                      raw(root / "u.data");
  std::map<std::pair<int, int>, int> source;
  int                                user, item, rating, timestamp, total = 0, window[257] = {};
  while (raw >> user >> item >> rating >> timestamp) {
    autorec_detail::Require(
        user >= 1 && user <= 943 && item >= 1 && item <= 1682 && rating >= 1 && rating <= 5 &&
            source.emplace(std::make_pair(user, item), rating).second,
        "all100Kuniquevalidoriginalratings");
    ++total;
    if (user <= 256 && item <= 256)
      ++window[user];
  }
  autorec_detail::Require(raw.eof() && total == 100000, "all100000originalMovieLensratings");
  std::vector<int> expected;
  for (int u = 1; u <= 256; ++u)
    if (window[u] >= 10)
      expected.push_back(u);
  autorec_detail::Require(expected.size() == 199, "deterministicuserwindowfilter");
  const auto users = d.users.to_vector<int>(), items = d.items.to_vector<int>();
  const auto in = d.input.to_vector<float>(), ratings = d.ratings.to_vector<float>(),
             tr = d.train.to_vector<float>(), va = d.validation.to_vector<float>(),
             te = d.test.to_vector<float>();
  int train = 0, val = 0, test = 0, item_fit[256] = {};
  for (int u = 0; u < 199; ++u) {
    autorec_detail::Require(users[u] == expected[u], "originaluserIDsortedwindow");
    int observed = 0, v = 0, t = 0, f = 0;
    for (int i = 0; i < 256; ++i) {
      autorec_detail::Require(items[i] == i + 1, "originalitemIDs1..256");
      auto      found = source.find({expected[u], i + 1});
      const int value = found == source.end() ? 0 : found->second;
      const int k     = u * 256 + i;
      autorec_detail::Near(ratings[k], value, 0, "everyselectedratingagainstoriginalsource");
      const float a1 = tr[k], b1 = va[k], c1 = te[k];
      autorec_detail::Require(
          (a1 == 0 || a1 == 1) && (b1 == 0 || b1 == 1) && (c1 == 0 || c1 == 1) &&
              a1 + b1 + c1 == (value ? 1 : 0),
          "disjointobserved-onlymasks,unknownneverzero-negatives");
      autorec_detail::Near(
          in[k], a1 ? float(value) / 5.f : 0, 0, "traininginputEXCLUDESallheldout/unknownratings");
      observed += value != 0;
      f += int(a1);
      v += int(b1);
      t += int(c1);
      item_fit[i] += int(a1);
    }
    autorec_detail::Require(
        v == std::max(1, observed / 10) && t == v && f > 0, "peruserholdoutcountsandwarmtraining");
    train += f;
    val += v;
    test += t;
  }
  for (int c : item_fit)
    autorec_detail::Require(c > 0, "everyitemhastrainingobservations");
  autorec_detail::Require(
      train == 8079 && val == 898 && test == 898,
      "exactsourcepinnedtraining/validation/testcounts");
  return d;
}
/// @brief Compute raw or clipped held-out rating RMSE/MAE in explicit star units.
///
/// Uses Float64 tensor reductions over binary positive-support masks, retaining
/// live paths to predictions. Ratings outside the mask are selected away before
/// arithmetic. Predictions and targets are already stars; no dtype-based scaling.
/// No mutation or inference/training state is owned.
/// @param prediction Aligned floating CPU star-valued predictions.
/// @param ratings Floating targets in stars, same shape.
/// @param mask Binary indicator, same shape.
/// @param clip Clamp predictions to1..5 for metric-only comparison.
/// @return Live Float64[2] RMSE and MAE.
/// @throws std::invalid_argument For empty/misaligned data, invalid masks or observed nonfinite
/// values.
/// @par Example
/// @code
/// namespace n=dm::nn;
/// const auto errors=n::models::RatingErrors(n::tensor({2.,4.}),n::tensor({3.,4.}),n::ones({2}));
/// @endcode
/// @par References
/// - GroupLens, MovieLens 100K, https://grouplens.org/datasets/movielens/100k/ .
/// @see ObservedRatingMse(), AutoRec()
inline Tensor RatingErrors(
    const Tensor& prediction,
    const Tensor& ratings,
    const Tensor& mask,
    bool          clip = false) {
  autorec_detail::Require(
      prediction.defined() && ratings.defined() && mask.defined() &&
          prediction.device().backend == compute::Backend::CPU &&
          ratings.device().backend == compute::Backend::CPU &&
          mask.device().backend == compute::Backend::CPU &&
          (prediction.dtype() == DType::Float32 || prediction.dtype() == DType::Float64) &&
          (ratings.dtype() == DType::Float32 || ratings.dtype() == DType::Float64),
      "floating CPU star-valued rating tensors required");
  const auto p = prediction.to(DType::Float64), t = ratings.to(DType::Float64),
             m          = mask.to(DType::Float64);
  const auto estimate   = clip ? nn::clip(p, 1., 5.) : p;
  const auto mse        = ObservedRatingMse(estimate, t, m);
  const auto difference = where(equal(m, 1.), estimate, 0.) - where(equal(m, 1.), t, 0.);
  return stack({sqrt(mse), sum(abs(difference)) / sum(m)}, 0);
}
} // namespace datamunge::nn::models
