#pragma once
/// @file depth_data.hpp
/// @brief Tensor raw-sensor sampling,room-isolated depth data,fit-only prior oracles and metric diagnostics.
#include "datamunge/nn/checkpoint.hpp"
#include "datamunge/nn/depth_models.hpp"
#include "datamunge/nn/training.hpp"
#include <filesystem>
#include <map>
#include <set>
namespace datamunge::nn::models {
namespace nn = datamunge::nn;
/// @brief Own observed RGB,metric sensor depth,validity and room provenance.
///
/// Tensor handles own/share storage. Sensor holes are explicit mask0/depth0,
/// not filled or inferred labels. Room/source metadata never enters forward.
///
/// @throws std::invalid_argument For invalid geometry,dtype,metadata,range or missing valid support.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const n::models::DepthData d{n::zeros({1, 48, 64, 3}), n::ones({1, 48, 64, 1}),
///                              n::ones({1, 48, 64, 1}), n::zeros({1}, n::DType::Int32),
///                              n::zeros({1}, n::DType::Int32)};
/// @endcode
/// @par References
/// - Silberman et al.(2012),Indoor Segmentation and Support Inference from RGBD Images,NYU Depth V2, https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html;rawDepths are aligned metric sensor data before hole filling. Eigen et al.(2014), https://arxiv.org/abs/1406.2283. Custom room-disjoint56-frame sampled teaching split,not benchmark evaluation.
/// @see LoadDepthPairs(), PrepareDepthSensorFrame(), ScoreTensorDepth()
struct DepthData {
  nn::Tensor rgb;         ///< Float32[N,H,W,3] observed RGB[0,1].
  nn::Tensor depth;       ///< Float32[N,H,W,1] metric sensor depth,invalid zeros.
  nn::Tensor valid;       ///< Float32 matching binary observed sensor support.
  nn::Tensor room_ids;    ///< Int32[N] canonical source-room identity,never a network input.
  nn::Tensor source_rows; ///< Int32[N] original zero-based aligned source frame indices.
};
/// @brief Own custom56-room fitting,validation and test sensor pairs.
///
/// LoadDepthPairs checks all56 room and source-frame identities are distinct.
///
/// @throws std::invalid_argument For invalid geometry,dtype,metadata,range or missing valid support.
/// @par Example
/// @code
/// const dm::nn::models::DepthSplits splits{};
/// @endcode
/// @par References
/// - Silberman et al.(2012),Indoor Segmentation and Support Inference from RGBD Images,NYU Depth V2, https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html;rawDepths are aligned metric sensor data before hole filling. Eigen et al.(2014), https://arxiv.org/abs/1406.2283. Custom room-disjoint56-frame sampled teaching split,not benchmark evaluation.
/// @see LoadDepthPairs(), PrepareDepthSensorFrame(), ScoreTensorDepth()
struct DepthSplits {
  DepthData fit;        ///< Thirty-two original frames from distinct fitting rooms.
  DepthData validation; ///< Eight frames from different validation rooms.
  DepthData test;       ///< Sixteen held-out rooms,one frame each.
};
/// @brief Load and validate one original RGB/raw-depth frame from each of56 disjoint rooms.
///
/// Requires Float32 RGB48x64,metric sensor depths/masks and Int32 provenance.
/// All56 canonical rooms AND source rows are distinct across the entire archive.
/// Every frame has>307 true sensor samples;eligible depth is finite,>.1f,<=10m,
/// invalid placeholders are exact0. Tensor reductions validate all pixel values.
/// Numeric/provenance checks do not prove annotation origin;no training is hidden.
///
/// @param root Directory with prepared pairs.npz.
/// @return Validated32/8/16 room-isolated splits.
/// @throws std::invalid_argument For invalid geometry,dtype,metadata,range or missing valid support.
/// @par Example
/// @code
/// try {
///   dm::nn::models::LoadDepthPairs(std::filesystem::temp_directory_path() /
///                                  "dm-missing-depth-example");
/// } catch (const std::exception &) {
/// }
/// @endcode
/// @par References
/// - Silberman et al.(2012),Indoor Segmentation and Support Inference from RGBD Images,NYU Depth V2, https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html;rawDepths are aligned metric sensor data before hole filling. Eigen et al.(2014), https://arxiv.org/abs/1406.2283. Custom room-disjoint56-frame sampled teaching split,not benchmark evaluation.
/// @see LoadDepthPairs(), PrepareDepthSensorFrame(), ScoreTensorDepth()
inline DepthSplits LoadDepthPairs(const std::filesystem::path &root) {
  std::map<std::string, nn::Tensor> arrays;
  for (const auto &[name, value] : nn::load_npz((root / "pairs.npz").string()))
    arrays.emplace(name, value);
  std::set<int> rooms, sources;
  /// @brief Load one fixed-size sensor split and extend room/source registries.
  /// @param name Exact split archive prefix.
  /// @param count Required original-room count.
  /// @return Validated aligned sensor handles.
  auto read = [&](const std::string &name, std::int64_t count) {
    DepthData d{arrays.at(name + "_rgb"), arrays.at(name + "_depth"),
                arrays.at(name + "_valid"), arrays.at(name + "_room_ids"),
                arrays.at(name + "_source_rows")};
    if (d.rgb.shape() != nn::Shape{count, 48, 64, 3} ||
        d.depth.shape() != nn::Shape{count, 48, 64, 1} || d.valid.shape() != d.depth.shape() ||
        d.room_ids.shape() != nn::Shape{count} || d.source_rows.shape() != nn::Shape{count})
      throw std::invalid_argument("aligned48x64 NYU sensor data required");
    for (const auto *t : {&d.rgb, &d.depth, &d.valid})
      if (t->dtype() != nn::DType::Float32)
        throw std::invalid_argument("Float32 RGB/sensor/mask required");
    for (const auto *t : {&d.room_ids, &d.source_rows})
      if (t->dtype() != nn::DType::Int32)
        throw std::invalid_argument("Int32 room/source provenance required");
    if (!nn::all(nn::logical_and(nn::greater_equal(d.rgb, 0.), nn::less_equal(d.rgb, 1.)))
             .item<bool>() ||
        !nn::all(nn::logical_or(nn::equal(d.valid, 0.), nn::equal(d.valid, 1.))).item<bool>())
      throw std::invalid_argument("RGB unit range or binary sensor mask invalid");
    const auto observed = nn::logical_and(
        nn::is_finite(d.depth),
        nn::logical_and(nn::greater(d.depth, double(.1f)), nn::less_equal(d.depth, 10.)));
    if (!nn::all(nn::where(nn::greater(d.valid, 0.), observed, nn::equal(d.depth, 0.)))
             .item<bool>() ||
        !nn::all(nn::greater(nn::sum(d.valid, {1, 2, 3}), 307.)).item<bool>())
      throw std::invalid_argument("metric sensor depth/invalid zeros/support invalid");
    const auto ids = d.room_ids.to_vector<int>(), rows = d.source_rows.to_vector<int>();
    for (std::int64_t i = 0; i < count; ++i)
      if (ids[i] < 0 || rows[i] < 0 || !rooms.insert(ids[i]).second ||
          !sources.insert(rows[i]).second)
        throw std::invalid_argument("duplicate or negative canonical room/source frame");
    return d;
  };
  auto fit = read("fit", 32), validation = read("validation", 8), test = read("test", 16);
  return {fit, validation, test};
}
/// @brief Gather an arbitrary-dtype sensor/provenance tensor by row metadata.
///
/// Host loops validate/serialize metadata only;no host pixel arithmetic.
///
/// @param tensor Positive leading batch axis,any remaining shape/dtype.
/// @param rows Bounded source rows;duplicates and empty output allowed.
/// @return Contiguous gathered tensor,preserving dtype and input graph.
/// @throws std::invalid_argument For invalid geometry,dtype,metadata,range or missing valid support.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto rows =
///     n::models::DepthBatch(n::tensor({7, 13, 19, 29}, {4}, n::DType::Int32), {3, 1});
/// @endcode
/// @par References
/// - Silberman et al.(2012),Indoor Segmentation and Support Inference from RGBD Images,NYU Depth V2, https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html;rawDepths are aligned metric sensor data before hole filling. Eigen et al.(2014), https://arxiv.org/abs/1406.2283. Custom room-disjoint56-frame sampled teaching split,not benchmark evaluation.
/// @see LoadDepthPairs(), PrepareDepthSensorFrame(), ScoreTensorDepth()
inline nn::Tensor DepthBatch(const nn::Tensor &tensor, const std::vector<std::size_t> &rows) {
  if (tensor.dim() < 1 || tensor.size(0) < 1)
    throw std::invalid_argument("positive depth batch axis required");
  std::vector<std::int64_t> ids;
  for (auto r : rows) {
    if (r >= static_cast<std::size_t>(tensor.size(0)))
      throw std::invalid_argument("depth paired row bounds");
    ids.push_back(static_cast<std::int64_t>(r));
  }
  return nn::take_rows(tensor, ids).contiguous();
}
/// @brief Gather observed RGB,sensor targets,masks and provenance together.
///
/// No sensor resampling,target regeneration or fitting.
///
/// @param data Aligned sensor/provenance tensors.
/// @param rows Requested bounded row metadata.
/// @return DepthData in caller-specified order.
/// @throws std::invalid_argument For invalid geometry,dtype,metadata,range or missing valid support.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const n::models::DepthData d{n::zeros({2, 48, 64, 3}), n::ones({2, 48, 64, 1}),
///                              n::ones({2, 48, 64, 1}), n::zeros({2}, n::DType::Int32),
///                              n::zeros({2}, n::DType::Int32)};
/// const auto batch = n::models::DepthBatch(d, {1});
/// @endcode
/// @par References
/// - Silberman et al.(2012),Indoor Segmentation and Support Inference from RGBD Images,NYU Depth V2, https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html;rawDepths are aligned metric sensor data before hole filling. Eigen et al.(2014), https://arxiv.org/abs/1406.2283. Custom room-disjoint56-frame sampled teaching split,not benchmark evaluation.
/// @see LoadDepthPairs(), PrepareDepthSensorFrame(), ScoreTensorDepth()
inline DepthData DepthBatch(const DepthData &data, const std::vector<std::size_t> &rows) {
  return {DepthBatch(data.rgb, rows), DepthBatch(data.depth, rows),
          DepthBatch(data.valid, rows), DepthBatch(data.room_ids, rows),
          DepthBatch(data.source_rows, rows)};
}
/// @brief Own closed-form fitting-only geometric and spatial prior oracles.
///
/// These are independent tensor reduction oracles. Reports fit trainable
/// constant/spatial log-depth parameters with visible optimizers,then check parity.
///
/// @throws std::invalid_argument For invalid geometry,dtype,metadata,range or missing valid support.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::models::DepthData d;
/// d.depth = n::ones({2, 48, 64, 1});
/// d.valid = n::ones({2, 48, 64, 1});
/// const auto prior = n::models::FitDepthPriors(d);
/// @endcode
/// @par References
/// - Silberman et al.(2012),Indoor Segmentation and Support Inference from RGBD Images,NYU Depth V2, https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html;rawDepths are aligned metric sensor data before hole filling. Eigen et al.(2014), https://arxiv.org/abs/1406.2283. Custom room-disjoint56-frame sampled teaching split,not benchmark evaluation.
/// @see LoadDepthPairs(), PrepareDepthSensorFrame(), ScoreTensorDepth()
struct DepthPriors {
  double geometric = 1.; ///< Fitting-only pooled geometric metric depth,meters.
  nn::Tensor spatial;    ///< Float32[1,48,64,1] per-position geometric depth,pooled fallback.
};
/// @brief Compute fitting-only pooled/per-position geometric depth as an oracle.
///
/// Tensor Float64 log sums/counts exclude invalid entries BEFORE log. Positions
/// with no observations use pooled geometric depth. All visible report fitting
/// must use explicit tensor optimizer loops;this helper provides independent
/// closed-form parity only,not the demonstrated fit.
///
/// @param fitting Float32[N,48,64,1] positive valid depths and matching positive-valid mask.
/// @return Pooled Double geometric depth and Float32 per-position geometric image.
/// @throws std::invalid_argument For invalid geometry,dtype,metadata,range or missing valid support.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::models::DepthData d;
/// d.depth = n::ones({2, 48, 64, 1});
/// d.valid = n::ones({2, 48, 64, 1});
/// const auto prior = n::models::FitDepthPriors(d);
/// @endcode
/// @par References
/// - Silberman et al.(2012),Indoor Segmentation and Support Inference from RGBD Images,NYU Depth V2, https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html;rawDepths are aligned metric sensor data before hole filling. Eigen et al.(2014), https://arxiv.org/abs/1406.2283. Custom room-disjoint56-frame sampled teaching split,not benchmark evaluation.
/// @see LoadDepthPairs(), PrepareDepthSensorFrame(), ScoreTensorDepth()
inline DepthPriors FitDepthPriors(const DepthData &fitting) {
  if (fitting.depth.dim() != 4 || fitting.depth.size(0) < 1 ||
      fitting.depth.shape() != nn::Shape{fitting.depth.size(0), 48, 64, 1} ||
      fitting.valid.shape() != fitting.depth.shape())
    throw std::invalid_argument("aligned48x64 fitting depth priors required");
  const auto observed = nn::greater(fitting.valid, 0.);
  const auto mask = observed.to(nn::DType::Float64);
  const auto count = nn::sum(mask);
  if (count.item() <= 0)
    throw std::invalid_argument("fit-only priors require observed sensor support");
  const auto safe = nn::where(observed, fitting.depth, 1.).to(nn::DType::Float64);
  if (!nn::all(nn::logical_and(nn::is_finite(safe), nn::greater(safe, 0.))).item<bool>())
    throw std::invalid_argument("positive finite fitting prior targets required");
  const auto log = nn::log(safe), pooled = nn::sum(mask * log) / count,
             counts = nn::sum(mask, {0}, true);
  const auto spatial = nn::exp(nn::where(nn::greater(counts, 0.),
                                         nn::sum(mask * log, {0}, true) /
                                             nn::maximum(counts, nn::ones_like(counts)),
                                         pooled))
                           .to(nn::DType::Float32);
  return {nn::exp(pooled).item(), spatial};
}
/// @brief Broadcast supplied fitting-only depth priors to requested image geometry.
///
/// No fitting or test-derived scale alignment. Prior values must originate
/// from fitting data and a visible optimizer loop in the report.
///
/// @param data Supplies48x64 target batch geometry only.
/// @param prior Supplied positive pooled/spatial metric depths.
/// @param spatial True uses per-position image,false uses pooled constant.
/// @return Float32[N,48,64,1] predictions without reading target values.
/// @throws std::invalid_argument For invalid geometry,dtype,metadata,range or missing valid support.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::models::DepthData d;
/// d.depth = n::zeros({2, 48, 64, 1});
/// const n::models::DepthPriors p{2., n::full({1, 48, 64, 1}, 2.)};
/// const auto baseline = n::models::DepthPriorImages(d, p, true);
/// @endcode
/// @par References
/// - Silberman et al.(2012),Indoor Segmentation and Support Inference from RGBD Images,NYU Depth V2, https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html;rawDepths are aligned metric sensor data before hole filling. Eigen et al.(2014), https://arxiv.org/abs/1406.2283. Custom room-disjoint56-frame sampled teaching split,not benchmark evaluation.
/// @see LoadDepthPairs(), PrepareDepthSensorFrame(), ScoreTensorDepth()
inline nn::Tensor DepthPriorImages(const DepthData &data, const DepthPriors &prior,
                                   bool spatial) {
  if (data.depth.dim() != 4 || data.depth.size(0) < 1 ||
      data.depth.shape() != nn::Shape{data.depth.size(0), 48, 64, 1} ||
      prior.spatial.shape() != nn::Shape{1, 48, 64, 1} || !std::isfinite(prior.geometric) ||
      prior.geometric <= 0)
    throw std::invalid_argument("valid positive depth prior geometry required");
  return spatial ? nn::ones(data.depth.shape()) * prior.spatial
                 : nn::full(data.depth.shape(), prior.geometric, nn::DType::Float32);
}
/// @brief Own unaligned metric-depth diagnostics on true sensor support.
///
/// Numeric metrics are Float64 tensors before optional presentation extraction.
/// No GT median scale alignment,prediction clipping or error normalization is hidden.
///
/// @throws std::invalid_argument For invalid geometry,dtype,metadata,range or missing valid support.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto score = n::models::ScoreTensorDepth(n::full({1, 1, 1, 1}, 2.),
///                                                n::ones({1, 1, 1, 1}), n::ones({1, 1, 1, 1}));
/// @endcode
/// @par References
/// - Silberman et al.(2012),Indoor Segmentation and Support Inference from RGBD Images,NYU Depth V2, https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html;rawDepths are aligned metric sensor data before hole filling. Eigen et al.(2014), https://arxiv.org/abs/1406.2283. Custom room-disjoint56-frame sampled teaching split,not benchmark evaluation.
/// @see LoadDepthPairs(), PrepareDepthSensorFrame(), ScoreTensorDepth()
struct TensorDepthMetrics {
  nn::Tensor abs_rel;  ///< Equal-image mean absolute error divided by true depth.
  nn::Tensor sq_rel;   ///< Equal-image mean squared error divided by true depth,meters.
  nn::Tensor rmse;     ///< Mean per-image metric RMSE,meters.
  nn::Tensor log_rmse; ///< Mean per-image natural-log RMSE.
  nn::Tensor log10;    ///< Mean per-image absolute log10 error.
  nn::Tensor silog;    ///< Mean per-image centered log-residual standard deviation.
  nn::Tensor delta1;   ///< Mean per-image fraction with strict ratio<1.25.
  nn::Tensor delta2;   ///< Mean per-image fraction with strict ratio<1.25².
  nn::Tensor delta3;   ///< Mean per-image fraction with strict ratio<1.25³.
  nn::Tensor images;   ///< Int64 scalar supported frame count.
  nn::Tensor pixels;   ///< Int64 scalar eligible sensor sample count.
};
/// @brief Own unaligned metric-depth diagnostics on true sensor support. Host presentation view.
///
/// Numeric metrics are Float64 tensors before optional presentation extraction.
/// No GT median scale alignment,prediction clipping or error normalization is hidden.
///
/// @throws std::invalid_argument For invalid geometry,dtype,metadata,range or missing valid support.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto score = n::models::ScoreDepth(n::full({1, 1, 1, 1}, 2.), n::ones({1, 1, 1, 1}),
///                                          n::ones({1, 1, 1, 1}));
/// @endcode
/// @par References
/// - Silberman et al.(2012),Indoor Segmentation and Support Inference from RGBD Images,NYU Depth V2, https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html;rawDepths are aligned metric sensor data before hole filling. Eigen et al.(2014), https://arxiv.org/abs/1406.2283. Custom room-disjoint56-frame sampled teaching split,not benchmark evaluation.
/// @see LoadDepthPairs(), PrepareDepthSensorFrame(), ScoreTensorDepth()
struct DepthMetrics {
  double abs_rel = 0;      ///< Equal-image mean absolute error divided by true depth.
  double sq_rel = 0;       ///< Equal-image mean squared error divided by true depth,meters.
  double rmse = 0;         ///< Mean per-image metric RMSE,meters.
  double log_rmse = 0;     ///< Mean per-image natural-log RMSE.
  double log10 = 0;        ///< Mean per-image absolute log10 error.
  double silog = 0;        ///< Mean per-image centered log-residual standard deviation.
  double delta1 = 0;       ///< Mean per-image fraction with strict ratio<1.25.
  double delta2 = 0;       ///< Mean per-image fraction with strict ratio<1.25².
  double delta3 = 0;       ///< Mean per-image fraction with strict ratio<1.25³.
  int images = 0;          ///< Supported room/frame count.
  std::int64_t pixels = 0; ///< Total eligible sensor samples.
};
/// @brief Score metric RGB-only depth without post-hoc GT alignment or clipping.
///
/// All images are weighted equally after their own valid-pixel reductions.
/// SILog uses centered residuals to avoid cancellation in mean(e²)-mean(e)^2.
/// Strict delta thresholds use max(p/t,t/p)<1.25^k. Invalid locations select
/// safe1 before logs/division,so their NaNs cannot leak. Unsupported images are
/// excluded;all-empty input throws. No estimated scale normalization or fitting.
///
/// @param prediction Finite positive Float32[N,H,W,1] estimates at eligible positions.
/// @param truth Same-shaped Float32 sensor meters;finite,>.1f,<=10 eligible.
/// @param mask Matching numeric support,used when>.5.
/// @return Detached Float64 nine metrics and Int64 supported image/pixel counts.
/// @throws std::invalid_argument For invalid geometry,dtype,metadata,range or missing valid support.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto score = n::models::ScoreTensorDepth(n::full({1, 1, 1, 1}, 2.),
///                                                n::ones({1, 1, 1, 1}), n::ones({1, 1, 1, 1}));
/// @endcode
/// @par References
/// - Silberman et al.(2012),Indoor Segmentation and Support Inference from RGBD Images,NYU Depth V2, https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html;rawDepths are aligned metric sensor data before hole filling. Eigen et al.(2014), https://arxiv.org/abs/1406.2283. Custom room-disjoint56-frame sampled teaching split,not benchmark evaluation.
/// @see LoadDepthPairs(), PrepareDepthSensorFrame(), ScoreTensorDepth()
inline TensorDepthMetrics ScoreTensorDepth(const nn::Tensor &prediction,
                                           const nn::Tensor &truth, const nn::Tensor &mask) {
  nn::NoGradGuard guard;
  if (prediction.dim() != 4 || prediction.size(0) < 1 || prediction.numel() < 1 ||
      prediction.size(3) != 1 || prediction.shape() != truth.shape() ||
      mask.shape() != truth.shape() || prediction.dtype() != nn::DType::Float32 ||
      truth.dtype() != nn::DType::Float32)
    throw std::invalid_argument("aligned positive-batch Float32 scalar depth scores required");
  const auto valid = nn::logical_and(
      nn::greater(mask.to(nn::DType::Float64), .5),
      nn::logical_and(nn::is_finite(truth), nn::logical_and(nn::greater(truth, double(.1f)),
                                                            nn::less_equal(truth, 10.))));
  const auto p = nn::where(valid, prediction, 1.).to(nn::DType::Float64),
             t = nn::where(valid, truth, 1.).to(nn::DType::Float64);
  if (!nn::all(nn::logical_and(nn::is_finite(p), nn::greater(p, 0.))).item<bool>())
    throw std::invalid_argument("positive finite unaligned depth estimates required");
  const auto weight = valid.to(nn::DType::Float64), count = nn::sum(weight, {1, 2, 3}),
             denom = nn::maximum(count, nn::ones_like(count));
  const auto images = nn::count_nonzero(nn::greater(count, 0.)).to(nn::DType::Int64);
  if (images.item<std::int64_t>() < 1)
    throw std::invalid_argument("depth scores require eligible sensor samples");
  /// @brief Reduce one full scalar error field to its within-image mean.
  /// @param values Error tensor matching observed scalar depth geometry.
  /// @return One Float64 mean per image,zero for unsupported images.
  auto per = [&](const nn::Tensor &values) {
    return nn::sum(weight * values, {1, 2, 3}) / denom;
  };
  /// @brief Average already normalized supported-image values equally.
  /// @param values One Float64 value per source frame.
  /// @return Detached Float64 supported-image mean.
  auto average = [&](const nn::Tensor &values) {
    return nn::sum(nn::where(nn::greater(count, 0.), values, 0.)) /
           images.to(nn::DType::Float64);
  };
  const auto diff = p - t, log = nn::log(p) - nn::log(t), ratio = nn::maximum(p / t, t / p),
             log_mean = per(log);
  const auto centered = log - log_mean.reshape({prediction.size(0), 1, 1, 1});
  return {average(per(nn::abs(diff) / t)),
          average(per(nn::square(diff) / t)),
          average(nn::sqrt(per(nn::square(diff)))),
          average(nn::sqrt(per(nn::square(log)))),
          average(per(nn::abs(log) / std::log(10.))),
          average(nn::sqrt(per(nn::square(centered)))),
          average(per(nn::less(ratio, 1.25).to(nn::DType::Float64))),
          average(per(nn::less(ratio, 1.25 * 1.25).to(nn::DType::Float64))),
          average(per(nn::less(ratio, 1.25 * 1.25 * 1.25).to(nn::DType::Float64))),
          images,
          nn::count_nonzero(valid).to(nn::DType::Int64)};
}
/// @brief Present canonical tensor metric-depth diagnostics as host scalars.
///
/// Compatibility egress only;all numerical work delegates to ScoreTensorDepth.
///
/// @param prediction Positive-batch Float32 scalar metric estimates.
/// @param truth Matching numeric GT,converted to Float32 like the legacy source.
/// @param mask Matching numeric sensor support.
/// @return Nine presentation doubles and supported image/pixel counts.
/// @throws std::invalid_argument For invalid geometry,dtype,metadata,range or missing valid support.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto s = n::models::ScoreDepth(n::full({1, 1, 1, 1}, 5.), n::full({1, 1, 1, 1}, 4.),
///                                      n::ones({1, 1, 1, 1}));
/// @endcode
/// @par References
/// - Silberman et al.(2012),Indoor Segmentation and Support Inference from RGBD Images,NYU Depth V2, https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html;rawDepths are aligned metric sensor data before hole filling. Eigen et al.(2014), https://arxiv.org/abs/1406.2283. Custom room-disjoint56-frame sampled teaching split,not benchmark evaluation.
/// @see LoadDepthPairs(), PrepareDepthSensorFrame(), ScoreTensorDepth()
inline DepthMetrics ScoreDepth(const nn::Tensor &prediction, const nn::Tensor &truth,
                               const nn::Tensor &mask) {
  const auto s = ScoreTensorDepth(prediction, truth.to(nn::DType::Float32), mask);
  return {s.abs_rel.item(),
          s.sq_rel.item(),
          s.rmse.item(),
          s.log_rmse.item(),
          s.log10.item(),
          s.silog.item(),
          s.delta1.item(),
          s.delta2.item(),
          s.delta3.item(),
          s.images.item<int>(),
          s.pixels.item<std::int64_t>()};
}
/// @brief Sample the same original aligned RGB/sensor coordinates without filling depth holes.
///
/// Tensor gather selects exact rows/columns5,15,...475/635 on BOTH modalities.
/// RGB converts to Float32/255;depth stays in meters and casts to Float32 once.
/// Eligible raw sensor depths are finite,>.1f,<=10m;others become mask0/depth0.
/// No resize,interpolation,filling,GT alignment,learned preprocessing or optimizer.
/// Dataset decoders should handle file layout/endian ingress before this call.
///
/// @param rgb_bytes Int32[480,640,3] original encoded RGB byte values0..255.
/// @param raw_depth Float32/64[480,640] aligned rawDepths meters,including invalid sensor values.
/// @param room_id Nonnegative canonical source-room metadata.
/// @param source_row Nonnegative zero-based original NYU frame metadata.
/// @return One Float32 RGB/depth/mask48x64 pair with Int32 provenance.
/// @throws std::invalid_argument For invalid geometry,dtype,metadata,range or missing valid support.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto d = n::models::PrepareDepthSensorFrame(n::zeros({480, 640, 3}, n::DType::Int32),
///                                                   n::ones({480, 640}), 7, 13);
/// @endcode
/// @par References
/// - Silberman et al.(2012),Indoor Segmentation and Support Inference from RGBD Images,NYU Depth V2, https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html;rawDepths are aligned metric sensor data before hole filling. Eigen et al.(2014), https://arxiv.org/abs/1406.2283. Custom room-disjoint56-frame sampled teaching split,not benchmark evaluation.
/// @see LoadDepthPairs(), PrepareDepthSensorFrame(), ScoreTensorDepth()
inline DepthData PrepareDepthSensorFrame(const nn::Tensor &rgb_bytes,
                                         const nn::Tensor &raw_depth, std::int32_t room_id,
                                         std::int32_t source_row) {
  nn::NoGradGuard guard;
  if (rgb_bytes.shape() != nn::Shape{480, 640, 3} || rgb_bytes.dtype() != nn::DType::Int32 ||
      raw_depth.shape() != nn::Shape{480, 640} ||
      (raw_depth.dtype() != nn::DType::Float32 && raw_depth.dtype() != nn::DType::Float64) ||
      room_id < 0 || source_row < 0 ||
      !nn::all(
           nn::logical_and(nn::greater_equal(rgb_bytes, 0), nn::less_equal(rgb_bytes, 255)))
           .item<bool>())
    throw std::invalid_argument(
        "aligned480x640 RGB bytes/raw metric depth and provenance required");
  const auto rows = nn::arange(5, 480, 10, nn::DType::Int64),
             cols = nn::arange(5, 640, 10, nn::DType::Int64);
  const auto rgb =
      nn::gather(nn::gather(rgb_bytes, rows, 0), cols, 1).to(nn::DType::Float32) / 255.;
  const auto sensor = nn::gather(nn::gather(raw_depth, rows, 0), cols, 1)
                          .to(nn::DType::Float32)
                          .reshape({1, 48, 64, 1});
  const auto valid =
      nn::logical_and(nn::is_finite(sensor), nn::logical_and(nn::greater(sensor, double(.1f)),
                                                             nn::less_equal(sensor, 10.)));
  return {rgb.reshape({1, 48, 64, 3}), nn::where(valid, sensor, 0.),
          valid.to(nn::DType::Float32), nn::full({1}, room_id, nn::DType::Int32),
          nn::full({1}, source_row, nn::DType::Int32)};
}
/// @brief Map scalar metric depth to the source display-only color ramp.
///
/// All color arithmetic uses tensors. Out-of-display-range values saturate
/// colors only;original sensor/model tensors and numeric scores remain raw.
/// The ramp is a teaching jet-style visualization,not semantic class color.
///
/// @param depth Finite numeric[...,1] metric depths,meters.
/// @return Float64[...,3] RGB clipped to the source.1..10m display scale.
/// @throws std::invalid_argument For invalid geometry or nonfinite display values.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto colors = n::models::DepthColors(n::tensor({.1, 2., 5., 10.}, {4, 1}));
/// @endcode
/// @par References
/// - NYU RGB/raw metric sensor depth, https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html. The source teaching piecewise jet-style.1..10m color ramp is display-only,not a depth transform.
/// @see DepthColors(), DepthPreviewTiles(), ScoreTensorDepth()
inline nn::Tensor DepthColors(const nn::Tensor &depth) {
  if (depth.dim() < 1 || depth.shape().back() != 1 || depth.numel() < 1 ||
      !nn::all(nn::is_finite(depth)).item<bool>())
    throw std::invalid_argument("finite scalar depth color field required");
  const auto t = nn::clip((depth.to(nn::DType::Float64) - .1) / 9.9, 0., 1.);
  return nn::concat({nn::clip(1.5 - nn::abs(4. * t - 3.), 0., 1.),
                     nn::clip(1.5 - nn::abs(4. * t - 2.), 0., 1.),
                     nn::clip(1.5 - nn::abs(4. * t - 1.), 0., 1.)},
                    depth.dim() - 1);
}
/// @brief Present one tensor metric-depth display color as host scalars.
///
/// Compatibility ingress/egress only;no host numerical color implementation.
///
/// @param depth Finite metric depth in meters.
/// @return Three Double RGB values from DepthColors.
/// @throws std::invalid_argument For invalid geometry or nonfinite display values.
/// @par Example
/// @code
/// const auto rgb = dm::nn::models::DepthColor(2.);
/// @endcode
/// @par References
/// - NYU RGB/raw metric sensor depth, https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html. The source teaching piecewise jet-style.1..10m color ramp is display-only,not a depth transform.
/// @see DepthColors(), DepthPreviewTiles(), ScoreTensorDepth()
inline std::array<double, 3> DepthColor(double depth) {
  const auto rgb =
      DepthColors(nn::from_vector(std::vector<double>{depth}, {1})).to_vector<double>();
  return {rgb[0], rgb[1], rgb[2]};
}
/// @brief Render observed RGB,sensor depth,prediction,valid error and sensor mask.
///
/// Columns:RGB,observed sensor colors,predicted colors,absolute metric error/5
/// in red,white valid mask. Sensor holes have black GT/error;prediction colors
/// are shown everywhere without inventing sensor targets. Colors clip.1..10m
/// for display only. Float64 quantization preserves source lround(255*x).
/// Use save_image_grid for layout/scaling;no optimizer or metric clipping.
///
/// @param data Aligned48x64 observed RGB,raw sensor metric depth and validity.
/// @param prediction Matching finite raw metric depth,not GT-scale aligned.
/// @return Float32[N,48,320,3] display-clipped/byte-quantized five-panel tiles.
/// @throws std::invalid_argument For invalid geometry or nonfinite display values.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const n::models::DepthData d{n::zeros({1, 48, 64, 3}), n::ones({1, 48, 64, 1}),
///                              n::ones({1, 48, 64, 1}), n::zeros({1}, n::DType::Int32),
///                              n::zeros({1}, n::DType::Int32)};
/// const auto tile = n::models::DepthPreviewTiles(d, d.depth);
/// @endcode
/// @par References
/// - NYU RGB/raw metric sensor depth, https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html. The source teaching piecewise jet-style.1..10m color ramp is display-only,not a depth transform.
/// @see DepthColors(), DepthPreviewTiles(), ScoreTensorDepth()
inline nn::Tensor DepthPreviewTiles(const DepthData &data, const nn::Tensor &prediction) {
  if (data.rgb.dim() != 4 || data.rgb.size(0) < 1 || data.rgb.size(1) != 48 ||
      data.rgb.size(2) != 64 || data.rgb.size(3) != 3 ||
      data.depth.shape() != nn::Shape{data.rgb.size(0), 48, 64, 1} ||
      data.valid.shape() != data.depth.shape() || prediction.shape() != data.depth.shape())
    throw std::invalid_argument("aligned48x64 RGB/sensor/prediction display required");
  const auto observed = nn::greater(data.valid, 0.);
  const auto truth = nn::where(observed, data.depth, 0.).to(nn::DType::Float64),
             estimate = prediction.to(nn::DType::Float64);
  const auto gt = nn::where(observed, DepthColors(truth), 0.), pred = DepthColors(estimate);
  const auto error = nn::where(observed, nn::abs(estimate - truth) / 5., 0.) *
                     nn::from_vector(std::vector<double>{1., 0., 0.}, {1, 1, 1, 3});
  const auto mask =
      observed.to(nn::DType::Float64) * nn::ones({1, 1, 1, 3}, nn::DType::Float64);
  const auto tiles = nn::concat({data.rgb.to(nn::DType::Float64), gt, pred, error, mask}, 2);
  return (nn::floor(255. * nn::clip(tiles, 0., 1.) + .5) / 255.).to(nn::DType::Float32);
}
} // namespace datamunge::nn::models
