#pragma once
/// @file occlusion.hpp
/// @brief Tensor finite-patch raw-score interventions and explicit overlap projection.
#include "datamunge/nn/ops.hpp"
#include <algorithm>
#include <functional>
#include <stdexcept>
namespace datamunge::nn::models {
namespace nn = datamunge::nn;
using nn::Tensor;
/// @brief Configure rectangular finite-patch replacement and full-coverage strides.
///
/// Integer geometry metadata. Strides cannot exceed patch dimensions;edge
/// anchors append the final legal start if the regular grid misses the boundary.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// m::OcclusionSpec spec{7, 7, 4, 4};
/// @endcode
/// @par References
/// - Zeiler and Fergus(2014),Visualizing and Understanding Convolutional Networks, https://arxiv.org/abs/1311.2901. This implementation measures finite PATCH replacement effects on fixed RAWlogits and projects them by explicit overlap averaging. A patch intervention,its replacement and its stride define the experiment;the projected map is not a pixel gradient,segmentation or causal attribution.
/// @see OcclusionSensitivity(),OcclusionAnchors()
struct OcclusionSpec {
  int height = 7;   ///< Positive patch height<=image height.
  int width = 7;    ///< Positive patch width<=image width.
  int stride_y = 4; ///< Positive vertical stride<=patch height.
  int stride_x = 4; ///< Positive horizontal stride<=patch width.
};
/// @brief Choose full-coverage patch start indices including the final image edge.
///
/// Pure index metadata;no observation arithmetic. Guarantee every image
/// coordinate belongs to at least one patch.
///
/// @param length Positive axis length.
/// @param patch Positive patch length<=axis length.
/// @param stride Positive stride<=patch length.
/// @return Ordered host index metadata0,stride,...plus final length-patch anchor ifneeded.
/// @throws std::invalid_argument For invalid image/patch/stride or stride>patch holes.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto starts = m::OcclusionAnchors(5, 2, 2);
/// @endcode
/// @par References
/// - Zeiler and Fergus(2014),Visualizing and Understanding Convolutional Networks, https://arxiv.org/abs/1311.2901. This implementation measures finite PATCH replacement effects on fixed RAWlogits and projects them by explicit overlap averaging. A patch intervention,its replacement and its stride define the experiment;the projected map is not a pixel gradient,segmentation or causal attribution.
/// @see OcclusionSensitivity(),OcclusionAnchors()
inline std::vector<int> OcclusionAnchors(int length, int patch, int stride) {
  if (length < 1 || patch < 1 || patch > length || stride < 1 || stride > patch)
    throw std::invalid_argument("valid patch/stride with complete axis coverage required");
  std::vector<int> starts;
  for (int i = 0; i <= length - patch; i += stride)
    starts.push_back(i);
  if (starts.back() != length - patch)
    starts.push_back(length - patch);
  return starts;
}
/// @brief Own every finite-patch input,prediction,score change and overlap projection.
///
/// All observation/score/coverage/projection arrays are tensors. Host integers
/// only record dimensions. Drops retain both signs;negative means the patch
/// replacement increased the selected raw score.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto x = n::ones({2, 4, 5, 1});
/// const auto out =
///     m::OcclusionSensitivity(x, n::zeros(x.shape()), n::zeros({2}, n::DType::Int32),
///                             {2, 2, 2, 2}, 3, [](const n::Tensor &p) {
///                               const auto s = n::sum(p, {1, 2, 3});
///                               return n::stack({s, -s}, 1);
///                             });
/// @endcode
/// @par References
/// - Zeiler and Fergus(2014),Visualizing and Understanding Convolutional Networks, https://arxiv.org/abs/1311.2901. This implementation measures finite PATCH replacement effects on fixed RAWlogits and projects them by explicit overlap averaging. A patch intervention,its replacement and its stride define the experiment;the projected map is not a pixel gradient,segmentation or causal attribution.
/// @see OcclusionSensitivity(),OcclusionAnchors()
struct OcclusionOutput {
  int patches = 0;        ///< Number of actual rectangular patch locations.
  int n = 0;              ///< Original image count.
  int h = 0;              ///< Original image height.
  int w = 0;              ///< Original image width.
  int c = 0;              ///< Original image channels.
  int classes = 0;        ///< Actual predictor class count.
  Tensor rectangles;      ///< Int32[P,4]top,left,height,width index metadata.
  Tensor coverage;        ///< Int32[H,W]number of covering patches at each coordinate.
  Tensor original_logits; ///< Float32[N,K]unchanged predictor raw scores.
  Tensor occluded_logits; ///< Float32[P,N,K]all actual counterfactual raw scores.
  Tensor original_score;  ///< Float32[N]fixed target raw score.
  Tensor occluded_score;  ///< Float32[P,N]same fixed target counterfactual score.
  Tensor drops;           ///< Float64[P,N]original minus counterfactual raw score,unclipped.
  Tensor pixel_mean; ///< Float64[N,H,W]coverage-normalized sum of overlapping patch drops.
  Tensor occluded_inputs; ///< Float32[P,N,H,W,C]all actual replacement inputs.
};
/// @brief Represent a frozen independent-row categorical prediction callable.
///
/// Callable(Float32[count,H,W,C])->Float32[count,K]RAWlogits. Callback may
/// batch internally but must not fit/update weights or use query labels.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// m::OcclusionPrediction predict = [](const n::Tensor &x) {
///   const auto s = n::sum(x, {1, 2, 3});
///   return n::stack({s, -s}, 1);
/// };
/// @endcode
/// @par References
/// - Zeiler and Fergus(2014),Visualizing and Understanding Convolutional Networks, https://arxiv.org/abs/1311.2901. This implementation measures finite PATCH replacement effects on fixed RAWlogits and projects them by explicit overlap averaging. A patch intervention,its replacement and its stride define the experiment;the projected map is not a pixel gradient,segmentation or causal attribution.
/// @see OcclusionSensitivity(),OcclusionAnchors()
using OcclusionPrediction = std::function<Tensor(const Tensor &)>;
/// @brief Measure signed finite-patch replacement effects and explicit overlap means with tensors.
///
/// Patchindex order is vertical-major/horizontal-minor,case axis follows each
/// patch. Tensor coordinate comparisons construct masks,where inserts replacement,
/// and callback batching preserves alignment. Raw Float32scores subtract inDouble.
/// Tensor mask-matrix multiplication sums overlapping Doublepatch drops and
/// divides by positive coverage. This is a patch measurement projected to pixels,
/// not additive pixel contributions or an infinitesimal derivative. Original
/// inputs/replacements/targets and frozen model remain unchanged;no optimizer.
///
/// @param input Nonempty finite Float32[N,H,W,C]actual observations.
/// @param replacement Matching Float32observations providing patch contents,not just a scalar.
/// @param target Int32[N]fixed explicit categories,held unchanged for all patches.
/// @param spec Positive rectangular patch/stride geometry guaranteeing full coverage.
/// @param batch Positive maximum callback batch of actual occluded observations.
/// @param predict Frozen Float32raw-logit callback,independent across rows.
/// @return Every actual input/logit/score/drop/coverage and projected overlap-mean tensor.
/// @throws std::invalid_argument For invalid shape/dtype/nonfinite data,targets,geometry,batching or prediction outputs.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto x = n::ones({2, 4, 5, 1});
/// const auto out =
///     m::OcclusionSensitivity(x, n::zeros(x.shape()), n::zeros({2}, n::DType::Int32),
///                             {2, 2, 2, 2}, 3, [](const n::Tensor &p) {
///                               const auto s = n::sum(p, {1, 2, 3});
///                               return n::stack({s, -s}, 1);
///                             });
/// @endcode
/// @par References
/// - Zeiler and Fergus(2014),Visualizing and Understanding Convolutional Networks, https://arxiv.org/abs/1311.2901. This implementation measures finite PATCH replacement effects on fixed RAWlogits and projects them by explicit overlap averaging. A patch intervention,its replacement and its stride define the experiment;the projected map is not a pixel gradient,segmentation or causal attribution.
/// @see OcclusionSensitivity(),OcclusionAnchors()
inline OcclusionOutput OcclusionSensitivity(const Tensor &input, const Tensor &replacement,
                                            const Tensor &target, const OcclusionSpec &spec,
                                            int batch, const OcclusionPrediction &predict) {
  if (input.dtype() != nn::DType::Float32 || input.dim() != 4 || input.numel() < 1 ||
      input.shape() != replacement.shape() || replacement.dtype() != input.dtype() ||
      target.dtype() != nn::DType::Int32 || target.shape() != nn::Shape{input.size(0)} ||
      batch < 1 || !predict || !nn::all(nn::is_finite(input)).item<bool>() ||
      !nn::all(nn::is_finite(replacement)).item<bool>())
    throw std::invalid_argument(
        "finite alignedFloat32NHWCinput/replacement,fixedtargets and positive batch required");
  OcclusionOutput out;
  out.n = input.size(0);
  out.h = input.size(1);
  out.w = input.size(2);
  out.c = input.size(3);
  const auto ys = OcclusionAnchors(out.h, spec.height, spec.stride_y),
             xs = OcclusionAnchors(out.w, spec.width, spec.stride_x);
  std::vector<int> rectangles;
  std::vector<Tensor> masks;
  const auto y = nn::arange(0, out.h, 1, nn::DType::Int64).reshape({out.h, 1}),
             x = nn::arange(0, out.w, 1, nn::DType::Int64).reshape({1, out.w});
  for (int top : ys)
    for (int left : xs) {
      rectangles.insert(rectangles.end(), {top, left, spec.height, spec.width});
      masks.push_back(nn::logical_and(
          nn::logical_and(nn::greater_equal(y, top), nn::less(y, top + spec.height)),
          nn::logical_and(nn::greater_equal(x, left), nn::less(x, left + spec.width))));
    }
  out.patches = masks.size();
  out.rectangles = nn::from_vector(rectangles, {out.patches, 4});
  const auto mask = nn::stack(masks, 0);
  out.coverage = nn::sum(mask.to(nn::DType::Int32), {0});
  out.original_logits = predict(input);
  if (out.original_logits.dim() != 2 || out.original_logits.size(0) != out.n ||
      out.original_logits.dtype() != nn::DType::Float32 ||
      !nn::all(nn::is_finite(out.original_logits)).item<bool>())
    throw std::invalid_argument("finiteFloat32rawlogit prediction required");
  out.classes = out.original_logits.size(1);
  if (out.classes < 2 || nn::min(target).item() < 0 || nn::max(target).item() >= out.classes)
    throw std::invalid_argument("valid fixed target class required");
  const auto selector = nn::one_hot(target, out.classes, 1, 0, nn::DType::Float32);
  out.original_score = nn::sum(out.original_logits * selector, {1});
  out.occluded_inputs = nn::where(mask.reshape({out.patches, 1, out.h, out.w, 1}),
                                  nn::expand_dims(replacement, 0), nn::expand_dims(input, 0));
  const auto flat = out.occluded_inputs.reshape({out.patches * out.n, out.h, out.w, out.c});
  std::vector<Tensor> logits;
  for (int start = 0; start < out.patches * out.n; start += batch) {
    const int end = std::min(start + batch, out.patches * out.n);
    const auto scores = predict(nn::slice_axis(flat, 0, start, end));
    if (scores.shape() != nn::Shape{end - start, out.classes} ||
        scores.dtype() != nn::DType::Float32 || !nn::all(nn::is_finite(scores)).item<bool>())
      throw std::invalid_argument("finite aligned counterfactual logits required");
    logits.push_back(scores);
  }
  out.occluded_logits = nn::concat(logits, 0).reshape({out.patches, out.n, out.classes});
  out.occluded_score = nn::sum(out.occluded_logits * nn::expand_dims(selector, 0), {2});
  out.drops = nn::expand_dims(out.original_score.to(nn::DType::Float64), 0) -
              out.occluded_score.to(nn::DType::Float64);
  out.pixel_mean =
      nn::matmul(nn::transpose(out.drops, {1, 0}),
                 mask.to(nn::DType::Float64).reshape({out.patches, out.h * out.w}))
          .reshape({out.n, out.h, out.w}) /
      out.coverage.to(nn::DType::Float64);
  return out;
}
} // namespace datamunge::nn::models
