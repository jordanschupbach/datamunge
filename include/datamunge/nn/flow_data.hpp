#pragma once
/// @file flow_data.hpp
/// @brief Tensor published-flow ingress,family-isolated crop validation,pyramid targets and flow metrics.
#include "datamunge/image/read_image.hpp"
#include "datamunge/nn/checkpoint.hpp"
#include "datamunge/nn/flow_components.hpp"
#include "datamunge/nn/image_io.hpp"
#include "datamunge/nn/training.hpp"
#include <bit>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
namespace datamunge::nn::models {
namespace nn = datamunge::nn;
/// @brief Own paired observed frames,published flow and provenance metadata.
///
/// Tensor handles own/share storage. Unknown ground truth uses valid0 and zero
/// flow placeholders. Source scene/family IDs are never model inputs.
///
/// @throws std::invalid_argument For invalid geometry,dtype,metadata,mask support or unreadable source data.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const n::models::FlowData d{n::zeros({1, 64, 64, 3}),       n::zeros({1, 64, 64, 3}),
///                             n::zeros({1, 64, 64, 2}),       n::ones({1, 64, 64, 1}),
///                             n::zeros({1}, n::DType::Int32), n::zeros({1}, n::DType::Int32)};
/// @endcode
/// @par References
/// - Middlebury public optical-flow data and .flo specification, https://vision.middlebury.edu/flow/data/ and https://vision.middlebury.edu/flow/code/flow-code/README.txt;Baker et al.(2011),A Database and Evaluation Methodology for Optical Flow, https://vision.middlebury.edu/flow/floweval-ijcv2011.pdf. Custom64px family-disjoint crops and source smooth multiscale objective are teaching choices,not official benchmark evaluation.
/// @see LoadFlowPairs(), PrepareFlowSupervision(), ScoreTensorFlow()
struct FlowData {
  nn::Tensor first;      ///< Float32[N,H,W,3] observed frame10 RGB[0,1].
  nn::Tensor second;     ///< Matching observed frame11 RGB[0,1].
  nn::Tensor flow;       ///< Float32[N,H,W,2] published first->second dx/dy,input pixels.
  nn::Tensor valid;      ///< Float32[N,H,W,1] binary published-ground-truth validity.
  nn::Tensor scene_ids;  ///< Int32[N] original scene metadata IDs.
  nn::Tensor family_ids; ///< Int32[N] related-scene family metadata,isolated between splits.
};
/// @brief Own fitting,validation and held-out scene-family splits.
///
/// LoadFlowPairs validates complete data and family isolation before fitting.
///
/// @throws std::invalid_argument For invalid geometry,dtype,metadata,mask support or unreadable source data.
/// @par Example
/// @code
/// const dm::nn::models::FlowSplits splits{};
/// @endcode
/// @par References
/// - Middlebury public optical-flow data and .flo specification, https://vision.middlebury.edu/flow/data/ and https://vision.middlebury.edu/flow/code/flow-code/README.txt;Baker et al.(2011),A Database and Evaluation Methodology for Optical Flow, https://vision.middlebury.edu/flow/floweval-ijcv2011.pdf. Custom64px family-disjoint crops and source smooth multiscale objective are teaching choices,not official benchmark evaluation.
/// @see LoadFlowPairs(), PrepareFlowSupervision(), ScoreTensorFlow()
struct FlowSplits {
  FlowData fit;        ///< Fitting families,32 canonical64px crops.
  FlowData validation; ///< Validation families,eight canonical64px crops.
  FlowData test;       ///< Held-out families,24 canonical64px crops.
};
/// @brief Load fixed32/8/24 published-flow crops with scene-family separation.
///
/// Requires aligned64px Float32 RGB/flow/masks,Int32 scene/family metadata,
/// binary masks,finite targets and zero invalid placeholders. Every crop must
/// have>409 valid pixels. Related families may repeat within one split but may
/// never cross splits. Checks numeric/provenance structure,not annotation origin.
/// All pixel validation uses tensor reductions;no hidden fitting.
///
/// @param root Directory containing prepared pairs.npz.
/// @return Validated fitting,validation and test handles.
/// @throws std::invalid_argument For invalid geometry,dtype,metadata,mask support or unreadable source data.
/// @par Example
/// @code
/// try {
///   dm::nn::models::LoadFlowPairs(std::filesystem::temp_directory_path() /
///                                 "dm-missing-flow-example");
/// } catch (const std::exception &) {
/// }
/// @endcode
/// @par References
/// - Middlebury public optical-flow data and .flo specification, https://vision.middlebury.edu/flow/data/ and https://vision.middlebury.edu/flow/code/flow-code/README.txt;Baker et al.(2011),A Database and Evaluation Methodology for Optical Flow, https://vision.middlebury.edu/flow/floweval-ijcv2011.pdf. Custom64px family-disjoint crops and source smooth multiscale objective are teaching choices,not official benchmark evaluation.
/// @see LoadFlowPairs(), PrepareFlowSupervision(), ScoreTensorFlow()
inline FlowSplits LoadFlowPairs(const std::filesystem::path &root) {
  std::map<std::string, nn::Tensor> arrays;
  for (const auto &[name, value] : nn::load_npz((root / "pairs.npz").string()))
    arrays.emplace(name, value);
  std::set<int> prior;
  /// @brief Load one fixed-size split and extend the family-isolation registry.
  /// @param split Exact archive prefix.
  /// @param count Required number of64px crops.
  /// @return Validated aligned split handles.
  auto read = [&](const std::string &split, std::int64_t count) {
    FlowData d{arrays.at(split + "_first"),     arrays.at(split + "_second"),
               arrays.at(split + "_flow"),      arrays.at(split + "_valid"),
               arrays.at(split + "_scene_ids"), arrays.at(split + "_family_ids")};
    if (d.first.shape() != nn::Shape{count, 64, 64, 3} ||
        d.second.shape() != d.first.shape() || d.flow.shape() != nn::Shape{count, 64, 64, 2} ||
        d.valid.shape() != nn::Shape{count, 64, 64, 1} ||
        d.scene_ids.shape() != nn::Shape{count} || d.family_ids.shape() != nn::Shape{count})
      throw std::invalid_argument("aligned64px published flow split required");
    for (const auto *t : {&d.first, &d.second, &d.flow, &d.valid})
      if (t->dtype() != nn::DType::Float32)
        throw std::invalid_argument("floating flow data required");
    for (const auto *t : {&d.scene_ids, &d.family_ids})
      if (t->dtype() != nn::DType::Int32)
        throw std::invalid_argument("Int32 flow provenance required");
    for (const auto &rgb : {d.first, d.second})
      if (!nn::all(nn::logical_and(nn::greater_equal(rgb, 0.), nn::less_equal(rgb, 1.)))
               .item<bool>())
        throw std::invalid_argument("observed RGB unit range required");
    if (!nn::all(nn::logical_or(nn::equal(d.valid, 0.), nn::equal(d.valid, 1.)))
             .item<bool>() ||
        !nn::all(nn::is_finite(d.flow)).item<bool>() ||
        !nn::all(nn::where(nn::equal(d.valid, 0.), nn::equal(d.flow, 0.),
                           nn::ones_like(d.flow).to(nn::DType::Bool)))
             .item<bool>() ||
        !nn::all(nn::greater(nn::sum(d.valid, {1, 2, 3}), 409.)).item<bool>())
      throw std::invalid_argument("invalid published flow mask or support");
    std::set<int> current;
    for (int family : d.family_ids.to_vector<int>()) {
      if (prior.contains(family))
        throw std::invalid_argument("related scene family crosses flow splits");
      current.insert(family);
    }
    prior.insert(current.begin(), current.end());
    return d;
  };
  auto fit = read("fit", 32), validation = read("validation", 8), test = read("test", 24);
  return {fit, validation, test};
}
/// @brief Gather arbitrary tensor batch rows for aligned optical-flow data.
///
/// Host loops validate row metadata only;no host pixel arithmetic.
///
/// @param input Positive leading batch axis,any remaining shape/dtype.
/// @param rows Bounded metadata indices,duplicates/empty allowed.
/// @return Contiguous gathered tensor with dtype/graph preserved.
/// @throws std::invalid_argument For invalid geometry,dtype,metadata,mask support or unreadable source data.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto batch = n::models::FlowBatch(n::ones({2, 3, 4, 2}), {1, 0, 1});
/// @endcode
/// @par References
/// - Middlebury public optical-flow data and .flo specification, https://vision.middlebury.edu/flow/data/ and https://vision.middlebury.edu/flow/code/flow-code/README.txt;Baker et al.(2011),A Database and Evaluation Methodology for Optical Flow, https://vision.middlebury.edu/flow/floweval-ijcv2011.pdf. Custom64px family-disjoint crops and source smooth multiscale objective are teaching choices,not official benchmark evaluation.
/// @see LoadFlowPairs(), PrepareFlowSupervision(), ScoreTensorFlow()
inline nn::Tensor FlowBatch(const nn::Tensor &input, const std::vector<std::size_t> &rows) {
  if (input.dim() < 1 || input.size(0) < 1)
    throw std::invalid_argument("positive flow batch axis required");
  std::vector<std::int64_t> ids;
  for (auto r : rows) {
    if (r >= static_cast<std::size_t>(input.size(0)))
      throw std::invalid_argument("flow batch row bounds");
    ids.push_back(static_cast<std::int64_t>(r));
  }
  return nn::take_rows(input, ids).contiguous();
}
/// @brief Gather observed pair,targets,masks and provenance together.
///
/// No target regeneration,flow-unit change or fitting.
///
/// @param d Aligned observed/target/provenance tensors.
/// @param rows Requested bounded metadata indices.
/// @return Aligned FlowData in caller order.
/// @throws std::invalid_argument For invalid geometry,dtype,metadata,mask support or unreadable source data.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const n::models::FlowData d{n::zeros({2, 64, 64, 3}),       n::zeros({2, 64, 64, 3}),
///                             n::zeros({2, 64, 64, 2}),       n::ones({2, 64, 64, 1}),
///                             n::zeros({2}, n::DType::Int32), n::zeros({2}, n::DType::Int32)};
/// const auto batch = n::models::FlowBatch(d, {1});
/// @endcode
/// @par References
/// - Middlebury public optical-flow data and .flo specification, https://vision.middlebury.edu/flow/data/ and https://vision.middlebury.edu/flow/code/flow-code/README.txt;Baker et al.(2011),A Database and Evaluation Methodology for Optical Flow, https://vision.middlebury.edu/flow/floweval-ijcv2011.pdf. Custom64px family-disjoint crops and source smooth multiscale objective are teaching choices,not official benchmark evaluation.
/// @see LoadFlowPairs(), PrepareFlowSupervision(), ScoreTensorFlow()
inline FlowData FlowBatch(const FlowData &d, const std::vector<std::size_t> &rows) {
  return {FlowBatch(d.first, rows),     FlowBatch(d.second, rows),
          FlowBatch(d.flow, rows),      FlowBatch(d.valid, rows),
          FlowBatch(d.scene_ids, rows), FlowBatch(d.family_ids, rows)};
}
/// @brief Own five tensor pyramid targets in fixed input-image displacement units.
///
/// All five resolutions share input-pixel/20 values. Empty blocks have zero
/// flow and zero validity;there is no per-level displacement rescaling.
///
/// @throws std::invalid_argument For invalid geometry,dtype,metadata,mask support or unreadable source data.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::models::FlowData d;
/// d.flow = n::ones({1, 64, 64, 2});
/// d.valid = n::ones({1, 64, 64, 1});
/// const auto targets = n::models::PrepareFlowSupervision(d);
/// @endcode
/// @par References
/// - Middlebury public optical-flow data and .flo specification, https://vision.middlebury.edu/flow/data/ and https://vision.middlebury.edu/flow/code/flow-code/README.txt;Baker et al.(2011),A Database and Evaluation Methodology for Optical Flow, https://vision.middlebury.edu/flow/floweval-ijcv2011.pdf. Custom64px family-disjoint crops and source smooth multiscale objective are teaching choices,not official benchmark evaluation.
/// @see LoadFlowPairs(), PrepareFlowSupervision(), ScoreTensorFlow()
struct FlowSupervision {
  std::array<nn::Tensor, 5> flow;  ///< Levels2..6 Float32 valid-block mean,input pixels/20.
  std::array<nn::Tensor, 5> valid; ///< Matching one-channel Float32 block-support flags.
};
/// @brief Downsample published GT by valid-block mean without changing input-pixel units.
///
/// Tensor reshape/transpose groups disjoint2^level blocks. Float64 masked
/// sums/counts precede /20 and final Float32 cast,matching source precision.
/// Invalid entries are selected out before reduction;empty blocks stay zero.
/// Targets are prepared once outside the visible model-optimizer loop.
///
/// @param data Float32 flow[N,64,64,2] and matching positive-valid mask.
/// @return Five Float32 target/mask tensors at16,8,4,2,1 spatial side.
/// @throws std::invalid_argument For invalid geometry,dtype,metadata,mask support or unreadable source data.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::models::FlowData d;
/// d.flow = n::ones({1, 64, 64, 2}) * 4.;
/// d.valid = n::ones({1, 64, 64, 1});
/// const auto target = n::models::PrepareFlowSupervision(d);
/// @endcode
/// @par References
/// - Middlebury public optical-flow data and .flo specification, https://vision.middlebury.edu/flow/data/ and https://vision.middlebury.edu/flow/code/flow-code/README.txt;Baker et al.(2011),A Database and Evaluation Methodology for Optical Flow, https://vision.middlebury.edu/flow/floweval-ijcv2011.pdf. Custom64px family-disjoint crops and source smooth multiscale objective are teaching choices,not official benchmark evaluation.
/// @see LoadFlowPairs(), PrepareFlowSupervision(), ScoreTensorFlow()
inline FlowSupervision PrepareFlowSupervision(const FlowData &data) {
  if (data.flow.dim() != 4 || data.flow.size(0) < 1 ||
      data.flow.shape() != nn::Shape{data.flow.size(0), 64, 64, 2} ||
      data.valid.shape() != nn::Shape{data.flow.size(0), 64, 64, 1})
    throw std::invalid_argument("64px aligned pyramid supervision required");
  const auto count = data.flow.size(0);
  FlowSupervision result;
  const auto valid = nn::greater(data.valid, 0.).to(nn::DType::Float64),
             flow =
                 nn::where(nn::greater(data.valid, 0.), data.flow, 0.).to(nn::DType::Float64);
  for (int level = 2; level <= 6; ++level) {
    const std::int64_t factor = std::int64_t(1) << level, side = 64 / factor;
    const auto blocks = nn::transpose(flow.reshape({count, side, factor, side, factor, 2}),
                                      {0, 1, 3, 2, 4, 5}),
               mask = nn::transpose(valid.reshape({count, side, factor, side, factor, 1}),
                                    {0, 1, 3, 2, 4, 5});
    const auto support = nn::sum(mask, {3, 4}), present = nn::greater(support, 0.);
    result.flow[level - 2] =
        nn::where(present,
                  nn::sum(blocks, {3, 4}) / nn::maximum(support, nn::ones_like(support)) / 20.,
                  0.)
            .to(nn::DType::Float32);
    result.valid[level - 2] = present.to(nn::DType::Float32);
  }
  return result;
}
/// @brief Own observed flow data and its cached multiscale supervision.
///
/// Caching targets performs no training or target-dependent image preprocessing.
///
/// @throws std::invalid_argument For invalid geometry,dtype,metadata,mask support or unreadable source data.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::models::FlowData d;
/// d.flow = n::ones({1, 64, 64, 2});
/// d.valid = n::ones({1, 64, 64, 1});
/// const n::models::FlowPrepared prepared{d, n::models::PrepareFlowSupervision(d)};
/// @endcode
/// @par References
/// - Middlebury public optical-flow data and .flo specification, https://vision.middlebury.edu/flow/data/ and https://vision.middlebury.edu/flow/code/flow-code/README.txt;Baker et al.(2011),A Database and Evaluation Methodology for Optical Flow, https://vision.middlebury.edu/flow/floweval-ijcv2011.pdf. Custom64px family-disjoint crops and source smooth multiscale objective are teaching choices,not official benchmark evaluation.
/// @see LoadFlowPairs(), PrepareFlowSupervision(), ScoreTensorFlow()
struct FlowPrepared {
  FlowData data;          ///< Original paired observations and input-pixel GT.
  FlowSupervision target; ///< Precomputed five-scale normalized target/mask tensors.
};
/// @brief Gather cached flow supervision in the same order as observed frames.
///
/// All pixel gathering delegates to tensors;no optimizer or recomputed targets.
///
/// @param input Aligned raw data and five target/mask levels.
/// @param rows Requested bounded row metadata.
/// @return Gathered observations and cached normalized supervision.
/// @throws std::invalid_argument For invalid geometry,dtype,metadata,mask support or unreadable source data.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const n::models::FlowData d{n::zeros({2, 64, 64, 3}),       n::zeros({2, 64, 64, 3}),
///                             n::zeros({2, 64, 64, 2}),       n::ones({2, 64, 64, 1}),
///                             n::zeros({2}, n::DType::Int32), n::zeros({2}, n::DType::Int32)};
/// const n::models::FlowPrepared prepared{d, n::models::PrepareFlowSupervision(d)};
/// const auto batch = n::models::FlowBatch(prepared, {1});
/// @endcode
/// @par References
/// - Middlebury public optical-flow data and .flo specification, https://vision.middlebury.edu/flow/data/ and https://vision.middlebury.edu/flow/code/flow-code/README.txt;Baker et al.(2011),A Database and Evaluation Methodology for Optical Flow, https://vision.middlebury.edu/flow/floweval-ijcv2011.pdf. Custom64px family-disjoint crops and source smooth multiscale objective are teaching choices,not official benchmark evaluation.
/// @see LoadFlowPairs(), PrepareFlowSupervision(), ScoreTensorFlow()
inline FlowPrepared FlowBatch(const FlowPrepared &input,
                              const std::vector<std::size_t> &rows) {
  FlowPrepared out;
  out.data = FlowBatch(input.data, rows);
  for (std::size_t i = 0; i < 5; ++i) {
    out.target.flow[i] = FlowBatch(input.target.flow[i], rows);
    out.target.valid[i] = FlowBatch(input.target.valid[i], rows);
  }
  return out;
}
/// @brief Own endpoint,angular and strict optical-flow outlier diagnostics.
///
/// Numeric metrics are Float64 tensor reductions. Images are weighted equally,
/// not by valid-pixel count. Unsupported images are excluded,not assigned zero.
///
/// @throws std::invalid_argument For invalid geometry,dtype,metadata,mask support or unreadable source data.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto metrics = n::models::ScoreTensorFlow(
///     n::zeros({1, 1, 1, 2}), n::tensor({3., 4.}, {1, 1, 1, 2}), n::ones({1, 1, 1, 1}));
/// @endcode
/// @par References
/// - Middlebury public optical-flow data and .flo specification, https://vision.middlebury.edu/flow/data/ and https://vision.middlebury.edu/flow/code/flow-code/README.txt;Baker et al.(2011),A Database and Evaluation Methodology for Optical Flow, https://vision.middlebury.edu/flow/floweval-ijcv2011.pdf. Custom64px family-disjoint crops and source smooth multiscale objective are teaching choices,not official benchmark evaluation.
/// @see LoadFlowPairs(), PrepareFlowSupervision(), ScoreTensorFlow()
struct TensorFlowMetrics {
  nn::Tensor epe;    ///< Mean per-image endpoint error,input-image pixels.
  nn::Tensor aae;    ///< Mean per-image augmented-vector angular error,degrees.
  nn::Tensor bad1;   ///< Mean per-image fraction with strict endpoint error>1px.
  nn::Tensor bad3;   ///< Mean per-image fraction with strict endpoint error>3px.
  nn::Tensor fl_all; ///< Mean per-image strict error>3px AND>.05*GT magnitude.
  nn::Tensor images; ///< Int64 scalar count of supported images.
  nn::Tensor pixels; ///< Int64 scalar total eligible true pixels.
};
/// @brief Own endpoint,angular and strict optical-flow outlier diagnostics. Host presentation view.
///
/// Numeric metrics are Float64 tensor reductions. Images are weighted equally,
/// not by valid-pixel count. Unsupported images are excluded,not assigned zero.
///
/// @throws std::invalid_argument For invalid geometry,dtype,metadata,mask support or unreadable source data.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto metrics = n::models::ScoreFlow(
///     n::zeros({1, 1, 1, 2}), n::tensor({3., 4.}, {1, 1, 1, 2}), n::ones({1, 1, 1, 1}));
/// @endcode
/// @par References
/// - Middlebury public optical-flow data and .flo specification, https://vision.middlebury.edu/flow/data/ and https://vision.middlebury.edu/flow/code/flow-code/README.txt;Baker et al.(2011),A Database and Evaluation Methodology for Optical Flow, https://vision.middlebury.edu/flow/floweval-ijcv2011.pdf. Custom64px family-disjoint crops and source smooth multiscale objective are teaching choices,not official benchmark evaluation.
/// @see LoadFlowPairs(), PrepareFlowSupervision(), ScoreTensorFlow()
struct FlowMetrics {
  double epe = 0;          ///< Mean per-image endpoint error,input-image pixels.
  double aae = 0;          ///< Mean per-image augmented-vector angular error,degrees.
  double bad1 = 0;         ///< Mean per-image fraction with strict endpoint error>1px.
  double bad3 = 0;         ///< Mean per-image fraction with strict endpoint error>3px.
  double fl_all = 0;       ///< Mean per-image strict error>3px AND>.05*GT magnitude.
  int images = 0;          ///< Images with at least one eligible true pixel.
  std::int64_t pixels = 0; ///< Total eligible true pixels.
};
/// @brief Score input-pixel flow with equal-image endpoint,angular and outlier means.
///
/// Angular error uses augmented vectors(u,v,1),with cosine clamped to[-1,1].
/// Outlier tests are strict,including exactly3px and5% boundaries. Invalid
/// locations are selected out before any arithmetic. Empty images are excluded;
/// all-empty input throws. No optimizer or model state is read or changed.
///
/// @param prediction Float32[N,H,W,2] estimated dx/dy,finite at eligible true pixels.
/// @param truth Same-shaped Float32 published GT;nonfinite or absolute components>=1e9 excluded.
/// @param valid Numeric mask with N*H*W elements,included when>.5.
/// @return Detached Float64 metrics and Int64 supported image/pixel counts.
/// @throws std::invalid_argument For invalid geometry,dtype,metadata,mask support or unreadable source data.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto m = n::models::ScoreTensorFlow(
///     n::zeros({1, 1, 1, 2}), n::tensor({3., 4.}, {1, 1, 1, 2}), n::ones({1, 1, 1, 1}));
/// @endcode
/// @par References
/// - Middlebury public optical-flow data and .flo specification, https://vision.middlebury.edu/flow/data/ and https://vision.middlebury.edu/flow/code/flow-code/README.txt;Baker et al.(2011),A Database and Evaluation Methodology for Optical Flow, https://vision.middlebury.edu/flow/floweval-ijcv2011.pdf. Custom64px family-disjoint crops and source smooth multiscale objective are teaching choices,not official benchmark evaluation.
/// @see LoadFlowPairs(), PrepareFlowSupervision(), ScoreTensorFlow()
inline TensorFlowMetrics ScoreTensorFlow(const nn::Tensor &prediction, const nn::Tensor &truth,
                                         const nn::Tensor &valid) {
  nn::NoGradGuard guard;
  if (truth.dim() != 4 || truth.size(0) < 1 || truth.numel() < 1 || truth.size(3) != 2 ||
      prediction.shape() != truth.shape() || valid.numel() != truth.numel() / 2 ||
      prediction.dtype() != nn::DType::Float32 || truth.dtype() != nn::DType::Float32)
    throw std::invalid_argument("aligned floating dx/dy flow score required");
  const auto mask = nn::logical_and(
      nn::greater(valid.reshape({truth.size(0), truth.size(1), truth.size(2), 1})
                      .to(nn::DType::Float64),
                  .5),
      nn::all(nn::logical_and(nn::is_finite(truth), nn::less(nn::abs(truth), 1e9)), {3},
              true));
  const auto p = nn::where(mask, prediction, 0.).to(nn::DType::Float64),
             t = nn::where(mask, truth, 0.).to(nn::DType::Float64);
  if (!nn::all(nn::is_finite(p)).item<bool>())
    throw std::invalid_argument("finite estimated flow at eligible GT required");
  const auto weight = mask.to(nn::DType::Float64), count = nn::sum(weight, {1, 2, 3}),
             present = nn::greater(count, 0.).to(nn::DType::Float64);
  const auto images = nn::count_nonzero(nn::greater(count, 0.)).to(nn::DType::Int64);
  if (images.item<std::int64_t>() < 1)
    throw std::invalid_argument("flow scores require eligible true pixels");
  const auto epe = nn::sqrt(nn::sum(nn::square(p - t), {3}, true)),
             tm = nn::sqrt(nn::sum(nn::square(t), {3}, true));
  const auto cosine =
      (nn::sum(p * t, {3}, true) + 1.) / nn::sqrt((nn::sum(nn::square(p), {3}, true) + 1.) *
                                                  (nn::sum(nn::square(t), {3}, true) + 1.));
  const auto angle = nn::acos(nn::clip(cosine, -1., 1.)) * 180. / 3.14159265358979323846;
  /// @brief Reduce one eligible error field as an equal-image-weighted mean.
  /// @param values Matching one-channel error/outlier tensor.
  /// @return Detached Float64 scalar mean over supported images.
  auto reduce = [&](const nn::Tensor &values) {
    return nn::sum((nn::sum(values * weight, {1, 2, 3}) /
                    nn::maximum(count, nn::ones_like(count))) *
                   present) /
           images.to(nn::DType::Float64);
  };
  return {reduce(epe),
          reduce(angle),
          reduce(nn::greater(epe, 1.).to(nn::DType::Float64)),
          reduce(nn::greater(epe, 3.).to(nn::DType::Float64)),
          reduce(nn::logical_and(nn::greater(epe, 3.), nn::greater(epe, .05 * tm))
                     .to(nn::DType::Float64)),
          images,
          nn::count_nonzero(mask).to(nn::DType::Int64)};
}
/// @brief Present canonical tensor optical-flow diagnostics as host scalars.
///
/// Compatibility egress only;all numerical work delegates to ScoreTensorFlow.
///
/// @param prediction Matching numeric estimated dx/dy field,converted to Float32 for legacy compatibility.
/// @param truth Numeric published true field,converted to Float32 with unknown sentinel policy.
/// @param valid Matching eligible GT mask.
/// @return Five doubles and supported image/pixel counts.
/// @throws std::invalid_argument For invalid geometry,dtype,metadata,mask support or unreadable source data.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto m = n::models::ScoreFlow(n::zeros({1, 1, 1, 2}), n::tensor({3., 4.}, {1, 1, 1, 2}),
///                                     n::ones({1, 1, 1, 1}));
/// @endcode
/// @par References
/// - Middlebury public optical-flow data and .flo specification, https://vision.middlebury.edu/flow/data/ and https://vision.middlebury.edu/flow/code/flow-code/README.txt;Baker et al.(2011),A Database and Evaluation Methodology for Optical Flow, https://vision.middlebury.edu/flow/floweval-ijcv2011.pdf. Custom64px family-disjoint crops and source smooth multiscale objective are teaching choices,not official benchmark evaluation.
/// @see LoadFlowPairs(), PrepareFlowSupervision(), ScoreTensorFlow()
inline FlowMetrics ScoreFlow(const nn::Tensor &prediction, const nn::Tensor &truth,
                             const nn::Tensor &valid) {
  const auto s = ScoreTensorFlow(prediction.to(nn::DType::Float32), truth.to(nn::DType::Float32), valid);
  return {s.epe.item(),
          s.aae.item(),
          s.bad1.item(),
          s.bad3.item(),
          s.fl_all.item(),
          s.images.item<int>(),
          s.pixels.item<std::int64_t>()};
}
/// @brief Compute a pooled fitting-only mean-flow oracle with tensor reductions.
///
/// Every eligible pixel has equal weight. Reports should fit the trainable
/// constant baseline with an explicit optimizer and use this only as an oracle.
/// Invalid entries are selected out before Float64 accumulation;no optimizer here.
///
/// @param data Fitting GT and positive-valid mask in input pixels.
/// @return Two Double mean dx/dy presentation values.
/// @throws std::invalid_argument For invalid geometry,dtype,metadata,mask support or unreadable source data.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::models::FlowData d;
/// d.flow = n::ones({1, 3, 4, 2});
/// d.valid = n::ones({1, 3, 4, 1});
/// const auto mean = n::models::FitMeanFlow(d);
/// @endcode
/// @par References
/// - Middlebury public optical-flow data and .flo specification, https://vision.middlebury.edu/flow/data/ and https://vision.middlebury.edu/flow/code/flow-code/README.txt;Baker et al.(2011),A Database and Evaluation Methodology for Optical Flow, https://vision.middlebury.edu/flow/floweval-ijcv2011.pdf. Custom64px family-disjoint crops and source smooth multiscale objective are teaching choices,not official benchmark evaluation.
/// @see LoadFlowPairs(), PrepareFlowSupervision(), ScoreTensorFlow()
inline std::array<double, 2> FitMeanFlow(const FlowData &data) {
  if (data.flow.dim() != 4 || data.flow.size(3) != 2 ||
      data.valid.shape() !=
          nn::Shape{data.flow.size(0), data.flow.size(1), data.flow.size(2), 1})
    throw std::invalid_argument("aligned fit-only flow mean required");
  const auto mask = nn::greater(data.valid, 0.).to(nn::DType::Float64), count = nn::sum(mask);
  if (count.item() <= 0)
    throw std::invalid_argument("fit-only flow mean requires support");
  const auto flow =
      nn::where(nn::greater(data.valid, 0.), data.flow, 0.).to(nn::DType::Float64);
  const auto values = (nn::sum(flow, {0, 1, 2}) / count).to_vector<double>();
  return {values[0], values[1]};
}
/// @brief Broadcast a supplied constant dx/dy without reading target values.
///
/// No fitting. Zero flow is a parameter-free reference;fitted constants must
/// come from fitting data and a visible optimizer loop in the report.
///
/// @param data Supplies NHW2 geometry only.
/// @param values Finite input-pixel dx/dy values.
/// @return Float32 matching-shape constant flow.
/// @throws std::invalid_argument For invalid geometry,dtype,metadata,mask support or unreadable source data.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::models::FlowData d;
/// d.flow = n::zeros({2, 3, 4, 2});
/// const auto zero = n::models::ConstantFlow(d, {0., 0.});
/// @endcode
/// @par References
/// - Middlebury public optical-flow data and .flo specification, https://vision.middlebury.edu/flow/data/ and https://vision.middlebury.edu/flow/code/flow-code/README.txt;Baker et al.(2011),A Database and Evaluation Methodology for Optical Flow, https://vision.middlebury.edu/flow/floweval-ijcv2011.pdf. Custom64px family-disjoint crops and source smooth multiscale objective are teaching choices,not official benchmark evaluation.
/// @see LoadFlowPairs(), PrepareFlowSupervision(), ScoreTensorFlow()
inline nn::Tensor ConstantFlow(const FlowData &data, const std::array<double, 2> &values) {
  if (data.flow.dim() != 4 || data.flow.size(3) != 2 || !std::isfinite(values[0]) ||
      !std::isfinite(values[1]))
    throw std::invalid_argument("finite constant dx/dy and aligned geometry required");
  return nn::ones(data.flow.shape()) *
         nn::from_vector(std::vector<double>{values[0], values[1]}, {2})
             .to(nn::DType::Float32);
}
/// @brief Read original frame10/11 images and published Middlebury .flo ground truth.
///
/// Validates little-endian PIEH header,positive dimensions,exact payload size
/// and observed frame geometry. Byte/bit decoding is ingress only. Tensor finite
/// and absolute<1e9 checks remove unknown GT to zero placeholders. Image decode
/// is byte ingress;RGB normalization/replication uses tensors. No resizing,flow
/// rescaling,ground-truth synthesis,optimizer or model inputs beyond observed RGB.
///
/// @param directory Contains frame10.png,frame11.png,flow10.flo.
/// @param scene_id Nonnegative caller-assigned original scene metadata.
/// @param family_id Nonnegative caller-assigned related-scene family metadata.
/// @return One full-resolution observed pair,Float32 input-pixel flow and validity.
/// @throws std::invalid_argument For invalid geometry,dtype,metadata,mask support or unreadable source data.
/// @par Example
/// @code
/// try {
///   dm::nn::models::ReadMiddleburyScene(
///       std::filesystem::temp_directory_path() / "dm-missing-flow-scene", 0, 0);
/// } catch (const std::exception &) {
/// }
/// @endcode
/// @par References
/// - Middlebury public optical-flow data and .flo specification, https://vision.middlebury.edu/flow/data/ and https://vision.middlebury.edu/flow/code/flow-code/README.txt;Baker et al.(2011),A Database and Evaluation Methodology for Optical Flow, https://vision.middlebury.edu/flow/floweval-ijcv2011.pdf. Custom64px family-disjoint crops and source smooth multiscale objective are teaching choices,not official benchmark evaluation.
/// @see LoadFlowPairs(), PrepareFlowSupervision(), ScoreTensorFlow()
inline FlowData ReadMiddleburyScene(const std::filesystem::path &directory,
                                    std::int32_t scene_id, std::int32_t family_id) {
  nn::NoGradGuard guard;
  if (scene_id < 0 || family_id < 0)
    throw std::invalid_argument("nonnegative flow provenance required");
  std::ifstream in(directory / "flow10.flo", std::ios::binary);
  if (!in)
    throw std::runtime_error("Middlebury flow10.flo missing");
  const std::vector<unsigned char> bytes((std::istreambuf_iterator<char>(in)), {});
  /// @brief Decode one little-endian32-bit word from raw file ingress bytes.
  /// @param offset Byte offset with four bytes available.
  /// @return Unsigned32-bit bit pattern,without floating arithmetic.
  auto word = [&](std::size_t offset) {
    return std::uint32_t(bytes[offset]) | (std::uint32_t(bytes[offset + 1]) << 8) |
           (std::uint32_t(bytes[offset + 2]) << 16) | (std::uint32_t(bytes[offset + 3]) << 24);
  };
  if (bytes.size() < 12 || word(0) != 0x48454950)
    throw std::invalid_argument("invalid PIEH flow header");
  const auto width = std::bit_cast<std::int32_t>(word(4)),
             height = std::bit_cast<std::int32_t>(word(8));
  if (width < 1 || height < 1 ||
      std::uint64_t(width) * std::uint64_t(height) >
          (std::numeric_limits<std::size_t>::max() - 12) / 8 ||
      bytes.size() != 12 + std::size_t(width) * height * 8)
    throw std::invalid_argument("invalid flow dimensions/payload length");
  std::vector<float> values(std::size_t(width) * height * 2);
  for (std::size_t i = 0; i < values.size(); ++i)
    values[i] = std::bit_cast<float>(word(12 + i * 4));
  const auto raw = nn::from_vector(values, {1, height, width, 2}),
             valid = nn::all(nn::logical_and(nn::is_finite(raw), nn::less(nn::abs(raw), 1e9)),
                             {3}, true)
                         .to(nn::DType::Float32);
  const auto flow = nn::where(nn::greater(valid, 0.), raw, 0.);
  std::vector<nn::Tensor> frames;
  for (const std::string file : {"frame10.png", "frame11.png"}) {
    auto pixels = nn::from_image(datamunge::image::read_image((directory / file).string()));
    if (pixels.dim() != 3 || pixels.size(0) != height || pixels.size(1) != width ||
        (pixels.size(2) != 1 && pixels.size(2) != 3))
      throw std::invalid_argument("observed frames disagree with flow geometry");
    if (pixels.size(2) == 1)
      pixels = pixels * nn::ones({1, 1, 3});
    frames.push_back(nn::expand_dims(pixels / 255., 0));
  }
  return {frames[0],
          frames[1],
          flow,
          valid,
          nn::full({1}, scene_id, nn::DType::Int32),
          nn::full({1}, family_id, nn::DType::Int32)};
}
/// @brief Expose deterministic selected64px crop origins as Int64 metadata.
///
/// Uses the declared5x5 tensor geometry grid,row-major order and>409 valid
/// sample rule. Host iteration only serializes selected coordinate metadata.
/// No model predictions,optimization,randomness or target-vector selection.
///
/// @param scene One observed scene>=64px with aligned one-channel valid mask.
/// @param count Number of unique origins1..25,default8.
/// @return Int64[count,2] y/x original-image crop origins.
/// @throws std::invalid_argument For invalid dimensions/count or insufficient unique supported origins.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::models::FlowData scene;
/// scene.first = n::zeros({1, 96, 96, 3});
/// scene.valid = n::ones({1, 96, 96, 1});
/// const auto origins = n::models::FlowCropOrigins(scene);
/// @endcode
/// @par References
/// - Middlebury public optical flow, https://vision.middlebury.edu/flow/data/. Source teaching64px validity-only crop policy is explicit.
/// @see FlowSceneCrops()
inline nn::Tensor FlowCropOrigins(const FlowData &scene, int count = 8) {
  if (scene.first.dim() != 4 || scene.first.size(0) != 1 || scene.first.size(1) < 64 ||
      scene.first.size(2) < 64 ||
      scene.valid.shape() != nn::Shape{1, scene.first.size(1), scene.first.size(2), 1} ||
      count < 1 || count > 25)
    throw std::invalid_argument("one aligned scene>=64px and1..25 origins required");
  const auto axis = nn::arange(0, 5, 1, nn::DType::Float64) / 4.;
  const auto ys = nn::floor(axis * double(scene.first.size(1) - 64)).to_vector<std::int64_t>(),
             xs = nn::floor(axis * double(scene.first.size(2) - 64)).to_vector<std::int64_t>();
  std::set<std::pair<std::int64_t, std::int64_t>> seen;
  std::vector<std::int64_t> selected;
  for (auto y : ys)
    for (auto x : xs) {
      if (static_cast<int>(selected.size() / 2) >= count || !seen.emplace(y, x).second)
        continue;
      const auto mask =
          nn::slice_axis(nn::slice_axis(scene.valid, 1, y, y + 64), 2, x, x + 64);
      if (nn::sum(mask).item() <= 409.)
        continue;
      selected.push_back(y);
      selected.push_back(x);
    }
  if (static_cast<int>(selected.size() / 2) != count)
    throw std::invalid_argument("not enough valid unique flow crop origins");
  return nn::from_vector(selected, {count, 2});
}
/// @brief Select fixed-grid64px crops while preserving original published vector units.
///
/// Tensor origins floor([0,.25,.5,.75,1]*(dimension-64)). Iterate row-major
/// 5x5 grid,skip duplicate origins and masks with<=409 valid samples,take first
/// count. This is a declared validity-only preparation rule,no prediction/test
/// score selection. Crop both frames identically;do not resize or alter dx/dy.
/// Related scene families must be assigned before calling LoadFlowPairs.
///
/// @param scene One original aligned observed/GT scene,side at least64.
/// @param count Number of crops1..25,default8.
/// @return Count observed/target/mask crops and repeated source provenance.
/// @throws std::invalid_argument For invalid geometry,dtype,metadata,mask support or unreadable source data.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const n::models::FlowData d{n::zeros({1, 96, 96, 3}),       n::zeros({1, 96, 96, 3}),
///                             n::zeros({1, 96, 96, 2}),       n::ones({1, 96, 96, 1}),
///                             n::zeros({1}, n::DType::Int32), n::zeros({1}, n::DType::Int32)};
/// const auto crops = n::models::FlowSceneCrops(d);
/// @endcode
/// @par References
/// - Middlebury public optical-flow data and .flo specification, https://vision.middlebury.edu/flow/data/ and https://vision.middlebury.edu/flow/code/flow-code/README.txt;Baker et al.(2011),A Database and Evaluation Methodology for Optical Flow, https://vision.middlebury.edu/flow/floweval-ijcv2011.pdf. Custom64px family-disjoint crops and source smooth multiscale objective are teaching choices,not official benchmark evaluation.
/// @see LoadFlowPairs(), PrepareFlowSupervision(), ScoreTensorFlow()
inline FlowData FlowSceneCrops(const FlowData &scene, int count = 8) {
  if (scene.first.dim() != 4 || scene.first.size(0) != 1 || scene.first.size(1) < 64 ||
      scene.first.size(2) < 64 || scene.first.size(3) != 3 ||
      scene.second.shape() != scene.first.shape() ||
      scene.flow.shape() != nn::Shape{1, scene.first.size(1), scene.first.size(2), 2} ||
      scene.valid.shape() != nn::Shape{1, scene.first.size(1), scene.first.size(2), 1} ||
      count < 1 || count > 25)
    throw std::invalid_argument("one aligned scene>=64px and1..25 crops required");
  const auto origins = FlowCropOrigins(scene, count).to_vector<std::int64_t>();
  std::vector<nn::Tensor> first, second, flow, valid;
  for (int i = 0; i < count; ++i) {
    const auto y = origins[2 * i], x = origins[2 * i + 1];
    first.push_back(nn::slice_axis(nn::slice_axis(scene.first, 1, y, y + 64), 2, x, x + 64));
    second.push_back(nn::slice_axis(nn::slice_axis(scene.second, 1, y, y + 64), 2, x, x + 64));
    flow.push_back(nn::slice_axis(nn::slice_axis(scene.flow, 1, y, y + 64), 2, x, x + 64));
    valid.push_back(nn::slice_axis(nn::slice_axis(scene.valid, 1, y, y + 64), 2, x, x + 64));
  }
  if (static_cast<int>(first.size()) != count)
    throw std::invalid_argument("not enough predetermined valid published-flow crops");
  return {nn::concat(first, 0),
          nn::concat(second, 0),
          nn::concat(flow, 0),
          nn::concat(valid, 0),
          nn::tile(scene.scene_ids, {count}),
          nn::tile(scene.family_ids, {count})};
}
} // namespace datamunge::nn::models
