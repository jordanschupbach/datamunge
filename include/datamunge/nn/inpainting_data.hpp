#pragma once
/// @file inpainting_data.hpp
/// @brief Tensor center-hole preparation,observed-context reference,hole metrics and image previews.
#include "datamunge/nn/inpainting_models.hpp"
#include "datamunge/nn/autograd.hpp"
#include "datamunge/nn/training.hpp"
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>
namespace datamunge::nn::models {
namespace nn = datamunge::nn;
/// @brief Own aligned clean targets,masked observations and center patch targets.
///
/// Tensor handles preserve their own storage/graphs. Learning and deployment
/// must use masked observations as generator inputs;truth/patch are loss targets.
///
/// @throws std::invalid_argument For invalid shape,dtype,rows,nonfinite values or missing required mask support.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto d = n::models::ContextPairs(n::zeros({2, 32, 32, 3}, n::DType::Int32), {1, 0});
/// @endcode
/// @par References
/// - Pathak et al.(2016),Context Encoders:Feature Learning by Inpainting, https://arxiv.org/abs/1604.07379. Source teaching CIFAR32/center16 split,ImageNet fill and mean per-image hole PSNR are explicit custom choices.
/// @see ContextPairs(), ContextFillMean(), InpaintScore()
struct InpaintData {
  nn::Tensor truth;  ///< Original normalized RGB Float32[N,32,32,3] in[0,1].
  nn::Tensor masked; ///< Observed/mean-filled RGB Float32[N,32,32,3] in[-1,1].
  nn::Tensor valid;  ///< Float32[N,32,32,1],one observed,zero missing.
  nn::Tensor patch;  ///< Center target Float32[N,16,16,3] in[-1,1].
};
/// @brief Create observed-validity mask for the center16px hole.
///
/// Pure tensor coordinate comparisons. No targets,learned parameters or optimizer.
///
/// @return Float32[1,32,32,1],zero at rows/columns8..23,one elsewhere.
/// @throws std::invalid_argument For invalid shape,dtype,rows,nonfinite values or missing required mask support.
/// @par Example
/// @code
/// const auto mask = dm::nn::models::ContextMask();
/// @endcode
/// @par References
/// - Pathak et al.(2016),Context Encoders:Feature Learning by Inpainting, https://arxiv.org/abs/1604.07379. Source teaching CIFAR32/center16 split,ImageNet fill and mean per-image hole PSNR are explicit custom choices.
/// @see ContextPairs(), ContextFillMean(), InpaintScore()
inline nn::Tensor ContextMask() {
  const auto y = nn::arange(0, 32, 1, nn::DType::Int32).reshape({1, 32, 1, 1});
  const auto x = nn::arange(0, 32, 1, nn::DType::Int32).reshape({1, 1, 32, 1});
  const auto hole = nn::logical_and(nn::logical_and(nn::greater_equal(y, 8), nn::less(y, 24)),
                                    nn::logical_and(nn::greater_equal(x, 8), nn::less(x, 24)));
  return nn::logical_not(hole).to(nn::DType::Float32);
}
/// @brief Gather observed-image batch rows in caller-specified order.
///
/// Host iteration only validates/serializes row metadata;pixel gathering is tensor based.
///
/// @param x Nonempty-batch Float32 rank4 input.
/// @param rows Metadata indices;duplicates and empty output allowed.
/// @return Float32[rows.size(),H,W,C],graph retained.
/// @throws std::invalid_argument For invalid shape,dtype,rows,nonfinite values or missing required mask support.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto batch = n::models::InpaintBatch(n::ones({2, 32, 32, 3}), {1, 0, 1});
/// @endcode
/// @par References
/// - Pathak et al.(2016),Context Encoders:Feature Learning by Inpainting, https://arxiv.org/abs/1604.07379. Source teaching CIFAR32/center16 split,ImageNet fill and mean per-image hole PSNR are explicit custom choices.
/// @see ContextPairs(), ContextFillMean(), InpaintScore()
inline nn::Tensor InpaintBatch(const nn::Tensor &x, const std::vector<std::size_t> &rows) {
  if (x.dtype() != nn::DType::Float32 || x.dim() != 4 || x.size(0) < 1)
    throw std::invalid_argument("float NHWC inpainting batch required");
  std::vector<std::int64_t> ids;
  for (auto row : rows) {
    if (row >= static_cast<std::size_t>(x.size(0)))
      throw std::invalid_argument("inpainting row bounds");
    ids.push_back(static_cast<std::int64_t>(row));
  }
  return nn::take_rows(x, ids).contiguous();
}
/// @brief Gather all inpainting inputs and targets in the same row order.
///
/// Delegates pixel operations to the tensor overload;no fitting.
///
/// @param data Aligned Float32 data tensor handles.
/// @param rows Requested metadata indices;duplicates/empty allowed.
/// @return Aligned independently gathered truth,masked,valid and patch tensors.
/// @throws std::invalid_argument For invalid shape,dtype,rows,nonfinite values or missing required mask support.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto d = n::models::ContextPairs(n::zeros({2, 32, 32, 3}, n::DType::Int32), {0, 1});
/// const auto batch = n::models::InpaintBatch(d, {1});
/// @endcode
/// @par References
/// - Pathak et al.(2016),Context Encoders:Feature Learning by Inpainting, https://arxiv.org/abs/1604.07379. Source teaching CIFAR32/center16 split,ImageNet fill and mean per-image hole PSNR are explicit custom choices.
/// @see ContextPairs(), ContextFillMean(), InpaintScore()
inline InpaintData InpaintBatch(const InpaintData &data,
                                const std::vector<std::size_t> &rows) {
  return {InpaintBatch(data.truth, rows), InpaintBatch(data.masked, rows),
          InpaintBatch(data.valid, rows), InpaintBatch(data.patch, rows)};
}
/// @brief Prepare central-hole context inputs directly from raw CIFAR tensors.
///
/// Only selected input pixels enter computation. Fill is(117,104,123)/255,
/// then scaled to[-1,1];normalization uses Float32 like the source. Masked
/// hole values never depend on clean target pixels. No optimizer or target leakage.
///
/// @param images Int32[N,32,32,3] RGB byte values,source batch positive.
/// @param rows Nonempty bounded source row metadata in desired order.
/// @return Normalized truth,mean-filled input,valid mask and target patch.
/// @throws std::invalid_argument For invalid shape,dtype,rows,nonfinite values or missing required mask support.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto d = n::models::ContextPairs(n::zeros({2, 32, 32, 3}, n::DType::Int32), {1, 0});
/// @endcode
/// @par References
/// - Pathak et al.(2016),Context Encoders:Feature Learning by Inpainting, https://arxiv.org/abs/1604.07379. Source teaching CIFAR32/center16 split,ImageNet fill and mean per-image hole PSNR are explicit custom choices.
/// @see ContextPairs(), ContextFillMean(), InpaintScore()
inline InpaintData ContextPairs(const nn::Tensor &images,
                                const std::vector<std::size_t> &rows) {
  if (images.dim() != 4 || images.size(0) < 1 || images.size(1) != 32 ||
      images.size(2) != 32 || images.size(3) != 3 || images.dtype() != nn::DType::Int32 ||
      rows.empty())
    throw std::invalid_argument("original32px Int32 RGB context images required");
  std::vector<std::int64_t> ids;
  for (auto row : rows) {
    if (row >= static_cast<std::size_t>(images.size(0)))
      throw std::invalid_argument("context row bounds");
    ids.push_back(static_cast<std::int64_t>(row));
  }
  const auto truth = nn::take_rows(images, ids).to(nn::DType::Float32) / 255.;
  const auto mask =
      nn::ones({static_cast<std::int64_t>(rows.size()), 1, 1, 1}) * ContextMask();
  const auto fill =
      (nn::from_vector(std::vector<double>{117., 104., 123.}, {1, 1, 1, 3}) / 255. * 2. - 1.)
          .to(nn::DType::Float32);
  const auto masked = nn::where(nn::greater(mask, .5), 2. * truth - 1., fill);
  const auto patch =
      nn::slice_axis(nn::slice_axis(2. * truth - 1., 1, 8, 24), 2, 8, 24).contiguous();
  return {truth, masked, mask, patch};
}
/// @brief Fill missing pixels with each image's observed-context mean color.
///
/// Mean computation reads only masked observations where valid>0,using Float64
/// accumulation after source Float32 inverse scaling. Observed output uses truth
/// for exact preservation;changing hidden truth never changes predictions. This
/// is a per-image deterministic statistic,with no fitted parameters or optimizer.
///
/// @param data Aligned32px image/valid tensors;every image has observed pixels.
/// @return Float32[N,32,32,3] known truth pixels and missing-region observed mean.
/// @throws std::invalid_argument For invalid shape,dtype,rows,nonfinite values or missing required mask support.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto d = n::models::ContextPairs(n::zeros({2, 32, 32, 3}, n::DType::Int32), {0, 1});
/// const auto mean = n::models::ContextFillMean(d);
/// @endcode
/// @par References
/// - Pathak et al.(2016),Context Encoders:Feature Learning by Inpainting, https://arxiv.org/abs/1604.07379. Source teaching CIFAR32/center16 split,ImageNet fill and mean per-image hole PSNR are explicit custom choices.
/// @see ContextPairs(), ContextFillMean(), InpaintScore()
inline nn::Tensor ContextFillMean(const InpaintData &data) {
  if (data.truth.dim() != 4 || data.truth.shape() != data.masked.shape() ||
      data.truth.shape() != nn::Shape{data.truth.size(0), 32, 32, 3} ||
      data.valid.shape() != nn::Shape{data.truth.size(0), 32, 32, 1} || data.truth.size(0) < 1)
    throw std::invalid_argument("aligned context mean images required");
  const auto valid = nn::greater(data.valid, 0.).to(nn::DType::Float64);
  const auto count = nn::sum(valid, {1, 2}, true);
  if (!nn::all(nn::greater(count, 0.)).item<bool>())
    throw std::invalid_argument("context mean requires observed support");
  const auto observed = ((data.masked + 1.) / 2.).to(nn::DType::Float64);
  const auto mean = (nn::sum(observed * valid, {1, 2}, true) / count).to(nn::DType::Float32);
  return nn::where(nn::greater(data.valid, 0.), data.truth, mean);
}
/// @brief Own hole-focused and full-image tensor reconstruction diagnostics.
///
/// Perfect-image PSNR is positive infinity. Average per-image PSNR differs
/// from PSNR of aggregate MSE;these fields retain the source protocol.
///
/// @throws std::invalid_argument For invalid shape,dtype,rows,nonfinite values or missing required mask support.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto d = n::models::ContextPairs(n::zeros({1, 32, 32, 3}, n::DType::Int32), {0});
/// const auto score = n::models::ScoreTensorInpaint(d, d.truth);
/// @endcode
/// @par References
/// - Pathak et al.(2016),Context Encoders:Feature Learning by Inpainting, https://arxiv.org/abs/1604.07379. Source teaching CIFAR32/center16 split,ImageNet fill and mean per-image hole PSNR are explicit custom choices.
/// @see ContextPairs(), ContextFillMean(), InpaintScore()
struct TensorInpaintMetrics {
  nn::Tensor hole_mse;    ///< Detached Float64 scalar mean per-image hole MSE.
  nn::Tensor hole_psnr;   ///< Detached Float64 scalar mean per-image hole PSNR,dB.
  nn::Tensor full_psnr;   ///< Detached Float64 scalar mean per-image full-image PSNR,dB.
  nn::Tensor known_error; ///< Detached Float64 scalar maximum observed absolute error.
};
/// @brief Convert unit-range MSE to PSNR with explicit perfect-image infinity.
///
/// No model/optimizer;unit peak intensity1 is assumed.
///
/// @param mse Nonempty numeric finite nonnegative mean-squared errors.
/// @return Detached Float64 matching-shape10*log10(1/MSE),positive infinity for zero.
/// @throws std::invalid_argument For invalid shape,dtype,rows,nonfinite values or missing required mask support.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto db = n::models::InpaintPSNR(n::tensor({.01, 0.}));
/// @endcode
/// @par References
/// - Pathak et al.(2016),Context Encoders:Feature Learning by Inpainting, https://arxiv.org/abs/1604.07379. Source teaching CIFAR32/center16 split,ImageNet fill and mean per-image hole PSNR are explicit custom choices.
/// @see ContextPairs(), ContextFillMean(), InpaintScore()
inline nn::Tensor InpaintPSNR(const nn::Tensor &mse) {
  const auto x = mse.detach().to(nn::DType::Float64);
  if (x.numel() < 1 ||
      !nn::all(nn::logical_and(nn::greater_equal(x, 0.),
                               nn::less(x, std::numeric_limits<double>::infinity())))
           .item<bool>())
    throw std::invalid_argument("finite nonnegative MSE required");
  return nn::where(
      nn::equal(x, 0.),
      nn::full(x.shape(), std::numeric_limits<double>::infinity(), nn::DType::Float64),
      -10. *
          nn::log(nn::maximum(x, nn::full(x.shape(), std::numeric_limits<double>::min(),
                                          nn::DType::Float64))) /
          std::log(10.));
}
/// @brief Return scalar unit-range reconstruction PSNR.
///
/// Compatibility scalar ingress delegates all numerical computation to tensors.
///
/// @param mse Finite nonnegative MSE;zero is a perfect image.
/// @return Double PSNR,positive infinity when mse is zero.
/// @throws std::invalid_argument For invalid shape,dtype,rows,nonfinite values or missing required mask support.
/// @par Example
/// @code
/// const double db = dm::nn::models::InpaintPSNR(.01); // 20
/// @endcode
/// @par References
/// - Pathak et al.(2016),Context Encoders:Feature Learning by Inpainting, https://arxiv.org/abs/1604.07379. Source teaching CIFAR32/center16 split,ImageNet fill and mean per-image hole PSNR are explicit custom choices.
/// @see ContextPairs(), ContextFillMean(), InpaintScore()
inline double InpaintPSNR(double mse) {
  return InpaintPSNR(nn::full({}, mse, nn::DType::Float64)).item();
}
/// @brief Score missing pixels separately from preserved context.
///
/// Every image must have missing support(valid==0). Difference/squared error
/// use Float64 before reduction. Full-image PSNR includes known pixels and can
/// look inflated. Perfect images contribute positive infinity to PSNR averages.
///
/// @param data Clean truth and valid mask of aligned32px inputs.
/// @param output Finite[N,32,32,3] RGB prediction,in the same[0,1] value scale.
/// @return Detached Float64 mean hole MSE,mean per-image hole/full PSNR and known-pixel max error.
/// @throws std::invalid_argument For invalid shape,dtype,rows,nonfinite values or missing required mask support.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto d = n::models::ContextPairs(n::zeros({1, 32, 32, 3}, n::DType::Int32), {0});
/// const auto score = n::models::ScoreTensorInpaint(d, d.truth);
/// @endcode
/// @par References
/// - Pathak et al.(2016),Context Encoders:Feature Learning by Inpainting, https://arxiv.org/abs/1604.07379. Source teaching CIFAR32/center16 split,ImageNet fill and mean per-image hole PSNR are explicit custom choices.
/// @see ContextPairs(), ContextFillMean(), InpaintScore()
inline TensorInpaintMetrics ScoreTensorInpaint(const InpaintData &data,
                                               const nn::Tensor &output) {
  nn::NoGradGuard guard;
  if (output.dim() != 4 || output.size(0) < 1 || output.shape() != data.truth.shape() ||
      output.shape() != nn::Shape{output.size(0), 32, 32, 3} ||
      data.valid.shape() != nn::Shape{output.size(0), 32, 32, 1} ||
      !nn::all(nn::less(nn::abs(output), std::numeric_limits<double>::infinity()))
           .item<bool>())
    throw std::invalid_argument("aligned finite32px inpainting output required");
  const auto holes = nn::equal(data.valid, 0.).to(nn::DType::Float64), known = 1. - holes;
  const auto counts = nn::sum(holes, {1, 2, 3}) * 3.;
  if (!nn::all(nn::greater(counts, 0.)).item<bool>())
    throw std::invalid_argument("hole metrics require missing pixels");
  const auto diff = output.to(nn::DType::Float64) - data.truth.to(nn::DType::Float64);
  const auto square = nn::square(diff);
  const auto hole = nn::sum(square * holes, {1, 2, 3}) / counts;
  const auto full = nn::mean(square, {1, 2, 3});
  return {nn::mean(hole), nn::mean(InpaintPSNR(hole)), nn::mean(InpaintPSNR(full)),
          nn::max(nn::abs(diff) * known)};
}
/// @brief Present tensor inpainting metrics as legacy host scalars.
///
/// Presentation only;all numerical work is performed by ScoreTensorInpaint.
///
/// @throws std::invalid_argument For invalid shape,dtype,rows,nonfinite values or missing required mask support.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto d = n::models::ContextPairs(n::zeros({1, 32, 32, 3}, n::DType::Int32), {0});
/// const auto score = n::models::InpaintScore(d, d.truth);
/// @endcode
/// @par References
/// - Pathak et al.(2016),Context Encoders:Feature Learning by Inpainting, https://arxiv.org/abs/1604.07379. Source teaching CIFAR32/center16 split,ImageNet fill and mean per-image hole PSNR are explicit custom choices.
/// @see ContextPairs(), ContextFillMean(), InpaintScore()
struct InpaintMetrics {
  double hole_mse = 0;    ///< Mean per-image missing-region MSE.
  double hole_psnr = 0;   ///< Mean per-image missing-region PSNR,dB.
  double full_psnr = 0;   ///< Mean per-image full-image PSNR,dB.
  double known_error = 0; ///< Maximum observed-pixel absolute error.
};
/// @brief Extract tensor reconstruction diagnostics as host presentation scalars.
///
/// Compatibility-only conversion,no host-array error arithmetic.
///
/// @param data Aligned truth/valid tensors with missing support.
/// @param output Finite matching32px RGB predictions.
/// @return Four doubles with exactly the tensor scorer's units and perfect-image policy.
/// @throws std::invalid_argument For invalid shape,dtype,rows,nonfinite values or missing required mask support.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto d = n::models::ContextPairs(n::zeros({1, 32, 32, 3}, n::DType::Int32), {0});
/// const auto m = n::models::InpaintScore(d, d.truth);
/// @endcode
/// @par References
/// - Pathak et al.(2016),Context Encoders:Feature Learning by Inpainting, https://arxiv.org/abs/1604.07379. Source teaching CIFAR32/center16 split,ImageNet fill and mean per-image hole PSNR are explicit custom choices.
/// @see ContextPairs(), ContextFillMean(), InpaintScore()
inline InpaintMetrics InpaintScore(const InpaintData &data, const nn::Tensor &output) {
  const auto m = ScoreTensorInpaint(data, output);
  return {m.hole_mse.item(), m.hole_psnr.item(), m.full_psnr.item(), m.known_error.item()};
}
/// @brief Render observed input,mean reference,prediction,truth and red error tiles.
///
/// Tensor concatenation and Float64 quantization preserve source lround(255*x).
/// Error is four times absolute RGB error in red channel only;no optimizer.
/// Use save_image_grid for layout/scaling;values are detached only by the writer.
///
/// @param data Aligned32px truth/masked inputs.
/// @param learned Matching predicted RGB image tensor.
/// @param mean Matching visible-mean RGB reference.
/// @return Float32[N,32,160,3] five-panel tiles,unit range,byte quantized.
/// @throws std::invalid_argument For invalid shape,dtype,rows,nonfinite values or missing required mask support.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto d = n::models::ContextPairs(n::zeros({1, 32, 32, 3}, n::DType::Int32), {0});
/// const auto mean = n::models::ContextFillMean(d);
/// const auto tile = n::models::ContextPreviewTiles(d, d.truth, mean);
/// @endcode
/// @par References
/// - Pathak et al.(2016),Context Encoders:Feature Learning by Inpainting, https://arxiv.org/abs/1604.07379. Source teaching CIFAR32/center16 split,ImageNet fill and mean per-image hole PSNR are explicit custom choices.
/// @see ContextPairs(), ContextFillMean(), InpaintScore()
inline nn::Tensor ContextPreviewTiles(const InpaintData &data, const nn::Tensor &learned,
                                      const nn::Tensor &mean) {
  if (learned.shape() != data.truth.shape() || mean.shape() != data.truth.shape() ||
      data.masked.shape() != data.truth.shape() || data.truth.dim() != 4 ||
      data.truth.size(0) < 1 || data.truth.size(1) != 32 || data.truth.size(2) != 32 ||
      data.truth.size(3) != 3)
    throw std::invalid_argument("aligned context preview images required");
  const auto red = nn::tensor({1., 0., 0.}).reshape({1, 1, 1, 3});
  const auto error =
      4. * nn::abs(learned.to(nn::DType::Float64) - data.truth.to(nn::DType::Float64)) *
      red.to(nn::DType::Float64);
  const auto masked = ((data.masked.to(nn::DType::Float64) + 1.) / 2.);
  const auto tiles =
      nn::concat({masked, mean.to(nn::DType::Float64), learned.to(nn::DType::Float64),
                  data.truth.to(nn::DType::Float64), error},
                 2);
  return (nn::floor(nn::clip(tiles, 0., 1.) * 255. + .5) / 255.).to(nn::DType::Float32);
}
} // namespace datamunge::nn::models
