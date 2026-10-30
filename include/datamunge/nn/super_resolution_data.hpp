#pragma once
/// @file super_resolution_data.hpp
/// @brief Tensor antialiased bicubic pairs,studio-range Y,common-crop PSNR/SSIM and image diagnostics.
#include "datamunge/nn/image_ops.hpp"
#include "datamunge/nn/super_resolution_models.hpp"
#include "datamunge/nn/training.hpp"
#include <filesystem>
namespace datamunge::nn::models {
namespace nn = datamunge::nn;
using nn::Tensor;
/// @brief Evaluate the Keys a=-.5 cubic reconstruction kernel.
///
/// Tensor arithmetic retains coordinate derivatives;piecewise boundaries follow source.
///
/// @param coordinate Floating signed offsets.
/// @return Matching-shape cubic values,zero at absolute offsets>=2.
/// @throws std::invalid_argument For invalid dimensions,dtype,nonfinite metric predictions or source row bounds.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto w = m::SRCubicTensor(n::tensor({-2., -1., 0., 1., 2.}));
/// @endcode
/// @par References
/// - Keys(1981),Cubic Convolution Interpolation for Digital Image Processing,IEEE TASSP29(6),1153-1160,doi:10.1109/TASSP.1981.1163711;Wang et al.(2004),Image quality assessment:From error visibility to structural similarity,IEEE TIP13(4),600-612, https://ece.uwaterloo.ca/~z70wang/research/ssim/. Source choices:Keys a=-.5,antialias widening,half-pixel centers,symmetric edges,column-then-row Float32 rounding,studio-range BT.601 Y and common central20x20 CIFAR crop. No benchmark claim.
/// @see SRBicubic(), ScoreTensorSR(), SRPreviewTiles()
inline Tensor SRCubicTensor(const Tensor &coordinate) {
  const auto x = nn::abs(coordinate);
  return nn::where(
      nn::less_equal(x, 1.), 1.5 * x * x * x - 2.5 * x * x + 1.,
      nn::where(nn::less(x, 2.), -.5 * x * x * x + 2.5 * x * x - 4. * x + 2., 0.));
}
/// @brief Evaluate one Keys kernel value through the tensor implementation.
///
/// Scalar presentation compatibility;numeric implementation is tensor-based.
///
/// @param x Signed finite offset.
/// @return Double kernel value.
/// @throws std::invalid_argument For invalid dimensions,dtype,nonfinite metric predictions or source row bounds.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto w = m::SRCubic(.5);
/// @endcode
/// @par References
/// - Keys(1981),Cubic Convolution Interpolation for Digital Image Processing,IEEE TASSP29(6),1153-1160,doi:10.1109/TASSP.1981.1163711;Wang et al.(2004),Image quality assessment:From error visibility to structural similarity,IEEE TIP13(4),600-612, https://ece.uwaterloo.ca/~z70wang/research/ssim/. Source choices:Keys a=-.5,antialias widening,half-pixel centers,symmetric edges,column-then-row Float32 rounding,studio-range BT.601 Y and common central20x20 CIFAR crop. No benchmark claim.
/// @see SRBicubic(), ScoreTensorSR(), SRPreviewTiles()
inline double SRCubic(double x) {
  return SRCubicTensor(nn::full({}, x, nn::DType::Float64)).item();
}
/// @brief Reflect a resampling index with duplicated edge samples.
///
/// Integer indexing metadata only. Uses64-bit period to avoid overflow.
///
/// @param index Integer coordinate metadata,possibly outside the image.
/// @param length Positive axis extent.
/// @return Bounded reflected coordinate.
/// @throws std::invalid_argument For invalid dimensions,dtype,nonfinite metric predictions or source row bounds.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto i = m::SRSymmetric(-1, 3);
/// @endcode
/// @par References
/// - Keys(1981),Cubic Convolution Interpolation for Digital Image Processing,IEEE TASSP29(6),1153-1160,doi:10.1109/TASSP.1981.1163711;Wang et al.(2004),Image quality assessment:From error visibility to structural similarity,IEEE TIP13(4),600-612, https://ece.uwaterloo.ca/~z70wang/research/ssim/. Source choices:Keys a=-.5,antialias widening,half-pixel centers,symmetric edges,column-then-row Float32 rounding,studio-range BT.601 Y and common central20x20 CIFAR crop. No benchmark claim.
/// @see SRBicubic(), ScoreTensorSR(), SRPreviewTiles()
inline int SRSymmetric(int index, int length) {
  if (length < 1)
    throw std::invalid_argument("positive symmetric extent required");
  const std::int64_t period = 2LL * length;
  std::int64_t i = index;
  i = (i % period + period) % period;
  return int(i < length ? i : period - 1 - i);
}
/// @brief Build normalized antialiased reflected interpolation weights as a tensor.
///
/// Centers are(i+.5)/scale-.5. Downsampling widens support to2/min(1,scale).
/// All cubic coefficients,normalization and duplicate-edge folding use tensors;
/// host loops serialize reflected index metadata only. Dense O(input*output) storage.
///
/// @param input Positive original sample count.
/// @param output Positive requested sample count.
/// @return Float64[output,input] matrix,each row sums to one.
/// @throws std::invalid_argument For invalid dimensions,dtype,nonfinite metric predictions or source row bounds.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto a = m::SRContributionMatrix(5, 10);
/// @endcode
/// @par References
/// - Keys(1981),Cubic Convolution Interpolation for Digital Image Processing,IEEE TASSP29(6),1153-1160,doi:10.1109/TASSP.1981.1163711;Wang et al.(2004),Image quality assessment:From error visibility to structural similarity,IEEE TIP13(4),600-612, https://ece.uwaterloo.ca/~z70wang/research/ssim/. Source choices:Keys a=-.5,antialias widening,half-pixel centers,symmetric edges,column-then-row Float32 rounding,studio-range BT.601 Y and common central20x20 CIFAR crop. No benchmark claim.
/// @see SRBicubic(), ScoreTensorSR(), SRPreviewTiles()
inline Tensor SRContributionMatrix(int input, int output) {
  if (input < 1 || output < 1)
    throw std::invalid_argument("positive bicubic dimensions required");
  const double scale = double(output) / input, filter = std::min(1., scale),
               radius = 2 / filter;
  const int margin = int(std::ceil(radius)) + 1;
  const auto centers = (nn::arange(0, output, 1, nn::DType::Float64) + .5) / scale - .5,
             positions = nn::arange(-margin, input + margin + 1, 1, nn::DType::Float64);
  auto weights =
      filter *
      SRCubicTensor(filter * (centers.reshape({output, 1}) - positions.reshape({1, -1})));
  weights = weights / nn::sum(weights, {1}, true);
  std::vector<std::int64_t> indices;
  for (int j = -margin; j <= input + margin; ++j)
    indices.push_back(SRSymmetric(j, input));
  const auto fold =
      nn::one_hot(nn::from_vector(indices, {std::int64_t(indices.size())}), input)
          .to(nn::DType::Float64);
  return nn::matmul(weights, fold);
}
/// @brief Name legacy host interpolation-weight metadata.
///
/// Compatibility presentation only;SRBicubic consumes tensor matrices directly.
///
/// @throws std::invalid_argument For invalid dimensions,dtype,nonfinite metric predictions or source row bounds.
/// @par Example
/// @code
/// const dm::nn::models::SRWeights weights{};
/// @endcode
/// @par References
/// - Keys(1981),Cubic Convolution Interpolation for Digital Image Processing,IEEE TASSP29(6),1153-1160,doi:10.1109/TASSP.1981.1163711;Wang et al.(2004),Image quality assessment:From error visibility to structural similarity,IEEE TIP13(4),600-612, https://ece.uwaterloo.ca/~z70wang/research/ssim/. Source choices:Keys a=-.5,antialias widening,half-pixel centers,symmetric edges,column-then-row Float32 rounding,studio-range BT.601 Y and common central20x20 CIFAR crop. No benchmark claim.
/// @see SRBicubic(), ScoreTensorSR(), SRPreviewTiles()
using SRWeights = std::vector<std::vector<std::pair<int, double>>>;
/// @brief Present normalized tensor interpolation weights as legacy index/value metadata.
///
/// Repeated reflected taps are folded. No image computation uses this host view.
///
/// @param input Positive original extent.
/// @param output Positive target extent.
/// @return Per-output reflected index/weight pairs.
/// @throws std::invalid_argument For invalid dimensions,dtype,nonfinite metric predictions or source row bounds.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto a = m::SRContributions(5, 10);
/// @endcode
/// @par References
/// - Keys(1981),Cubic Convolution Interpolation for Digital Image Processing,IEEE TASSP29(6),1153-1160,doi:10.1109/TASSP.1981.1163711;Wang et al.(2004),Image quality assessment:From error visibility to structural similarity,IEEE TIP13(4),600-612, https://ece.uwaterloo.ca/~z70wang/research/ssim/. Source choices:Keys a=-.5,antialias widening,half-pixel centers,symmetric edges,column-then-row Float32 rounding,studio-range BT.601 Y and common central20x20 CIFAR crop. No benchmark claim.
/// @see SRBicubic(), ScoreTensorSR(), SRPreviewTiles()
inline SRWeights SRContributions(int input, int output) {
  const auto matrix = SRContributionMatrix(input, output).to_vector<double>();
  SRWeights result(output);
  for (int i = 0; i < output; ++i)
    for (int j = 0; j < input; ++j)
      if (matrix[std::size_t(i) * input + j] != 0)
        result[i].push_back({j, matrix[std::size_t(i) * input + j]});
  return result;
}
/// @brief Resize NHWC images with separable antialiased Keys bicubic tensor operations.
///
/// Column resize uses Float64 sums then Float32 intermediate;row resize repeats
/// that policy. Negative cubic lobes and overshoot remain raw. No target fitting,
/// image decoder,optimizer or external resize is hidden. Dense weight matrices
/// are intended for teaching-sized images.
///
/// @param image Numeric positive-batch NHWC image,converted to Float32.
/// @param oh Positive output height.
/// @param ow Positive output width.
/// @return Float32[N,oh,ow,C] unclipped interpolation,graph retained.
/// @throws std::invalid_argument For invalid dimensions,dtype,nonfinite metric predictions or source row bounds.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto x = n::ones({1, 5, 7, 3});
/// const auto y = m::SRBicubic(x, 3, 4);
/// @endcode
/// @par References
/// - Keys(1981),Cubic Convolution Interpolation for Digital Image Processing,IEEE TASSP29(6),1153-1160,doi:10.1109/TASSP.1981.1163711;Wang et al.(2004),Image quality assessment:From error visibility to structural similarity,IEEE TIP13(4),600-612, https://ece.uwaterloo.ca/~z70wang/research/ssim/. Source choices:Keys a=-.5,antialias widening,half-pixel centers,symmetric edges,column-then-row Float32 rounding,studio-range BT.601 Y and common central20x20 CIFAR crop. No benchmark claim.
/// @see SRBicubic(), ScoreTensorSR(), SRPreviewTiles()
inline Tensor SRBicubic(const Tensor &image, int oh, int ow) {
  if (image.dim() != 4 || image.size(0) < 1 || image.size(1) < 1 || image.size(2) < 1 ||
      image.size(3) < 1 || oh < 1 || ow < 1)
    throw std::invalid_argument("positive bicubic NHWC geometry required");
  const auto b = image.size(0), h = image.size(1), w = image.size(2), c = image.size(3);
  const auto input = image.to(nn::DType::Float32).to(nn::DType::Float64),
             cols = SRContributionMatrix(int(w), ow), rows = SRContributionMatrix(int(h), oh);
  const auto horizontal =
      nn::transpose(nn::matmul(nn::transpose(input, {0, 1, 3, 2}).reshape({-1, w}),
                               nn::transpose(cols, {1, 0}))
                        .reshape({b, h, c, ow}),
                    {0, 1, 3, 2})
          .to(nn::DType::Float32);
  return nn::transpose(
             nn::matmul(nn::transpose(horizontal.to(nn::DType::Float64), {0, 2, 3, 1})
                            .reshape({-1, h}),
                        nn::transpose(rows, {1, 0}))
                 .reshape({b, ow, c, oh}),
             {0, 3, 1, 2})
      .to(nn::DType::Float32)
      .contiguous();
}
/// @brief Convert encoded RGB to studio-range BT.601 luminance.
///
/// Preserves out-of-range interpolation values;no gamma linearization or clipping.
/// Coefficients match source double arithmetic,not full-range Rec.709 luminance.
///
/// @param rgb Numeric[...,3] encoded RGB unit values.
/// @return Float64[...,1] Y,black16/255,white235/255.
/// @throws std::invalid_argument For invalid dimensions,dtype,nonfinite metric predictions or source row bounds.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto y = m::SRLumaTensor(n::ones({1, 2, 3, 3}));
/// @endcode
/// @par References
/// - Keys(1981),Cubic Convolution Interpolation for Digital Image Processing,IEEE TASSP29(6),1153-1160,doi:10.1109/TASSP.1981.1163711;Wang et al.(2004),Image quality assessment:From error visibility to structural similarity,IEEE TIP13(4),600-612, https://ece.uwaterloo.ca/~z70wang/research/ssim/. Source choices:Keys a=-.5,antialias widening,half-pixel centers,symmetric edges,column-then-row Float32 rounding,studio-range BT.601 Y and common central20x20 CIFAR crop. No benchmark claim.
/// @see SRBicubic(), ScoreTensorSR(), SRPreviewTiles()
inline Tensor SRLumaTensor(const Tensor &rgb) {
  if (rgb.dim() < 1 || rgb.shape().back() != 3)
    throw std::invalid_argument("RGB final axis required");
  const auto x = rgb.to(nn::DType::Float64);
  return 16. / 255. + (65.481 * nn::slice_axis(x, x.dim() - 1, 0, 1) +
                       128.553 * nn::slice_axis(x, x.dim() - 1, 1, 2) +
                       24.966 * nn::slice_axis(x, x.dim() - 1, 2, 3)) /
                          255.;
}
/// @brief Present one studio-range tensor luminance as a Double.
///
/// Legacy scalar compatibility delegates all numeric work to tensors.
///
/// @param r Encoded red unit value.
/// @param g Encoded green unit value.
/// @param b Encoded blue unit value.
/// @return Studio-range Y.
/// @throws std::invalid_argument For invalid dimensions,dtype,nonfinite metric predictions or source row bounds.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto y = m::SRLuma(1, 1, 1);
/// @endcode
/// @par References
/// - Keys(1981),Cubic Convolution Interpolation for Digital Image Processing,IEEE TASSP29(6),1153-1160,doi:10.1109/TASSP.1981.1163711;Wang et al.(2004),Image quality assessment:From error visibility to structural similarity,IEEE TIP13(4),600-612, https://ece.uwaterloo.ca/~z70wang/research/ssim/. Source choices:Keys a=-.5,antialias widening,half-pixel centers,symmetric edges,column-then-row Float32 rounding,studio-range BT.601 Y and common central20x20 CIFAR crop. No benchmark claim.
/// @see SRBicubic(), ScoreTensorSR(), SRPreviewTiles()
inline double SRLuma(double r, double g, double b) {
  return SRLumaTensor(nn::from_vector(std::vector<double>{r, g, b}, {3})).item();
}
/// @brief Prepare Float32 studio-range Y for reconstruction training.
///
/// Matches source input Float32 then Double coefficient arithmetic and one final
/// Float32 cast. Metrics call SRLumaTensor directly to avoid extra rounding.
///
/// @param rgb Numeric[N,H,W,3] encoded RGB.
/// @return Float32[N,H,W,1],graph retained.
/// @throws std::invalid_argument For invalid dimensions,dtype,nonfinite metric predictions or source row bounds.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto y = m::SRY(n::ones({1, 2, 3, 3}));
/// @endcode
/// @par References
/// - Keys(1981),Cubic Convolution Interpolation for Digital Image Processing,IEEE TASSP29(6),1153-1160,doi:10.1109/TASSP.1981.1163711;Wang et al.(2004),Image quality assessment:From error visibility to structural similarity,IEEE TIP13(4),600-612, https://ece.uwaterloo.ca/~z70wang/research/ssim/. Source choices:Keys a=-.5,antialias widening,half-pixel centers,symmetric edges,column-then-row Float32 rounding,studio-range BT.601 Y and common central20x20 CIFAR crop. No benchmark claim.
/// @see SRBicubic(), ScoreTensorSR(), SRPreviewTiles()
inline Tensor SRY(const Tensor &rgb) {
  if (rgb.dim() != 4)
    throw std::invalid_argument("SRY NHWC required");
  return SRLumaTensor(rgb.to(nn::DType::Float32)).to(nn::DType::Float32);
}
/// @brief Crop an equal border from both spatial axes.
///
/// SRCNN target alignment uses border6;common metrics use central20x20.
///
/// @param x NHWC tensor,any dtype.
/// @param border Nonnegative border,strictly less than half each extent.
/// @return Contiguous centered NHWC crop,graph retained.
/// @throws std::invalid_argument For invalid dimensions,dtype,nonfinite metric predictions or source row bounds.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto y = m::SRCrop(n::ones({1, 32, 34, 3}), 6);
/// @endcode
/// @par References
/// - Keys(1981),Cubic Convolution Interpolation for Digital Image Processing,IEEE TASSP29(6),1153-1160,doi:10.1109/TASSP.1981.1163711;Wang et al.(2004),Image quality assessment:From error visibility to structural similarity,IEEE TIP13(4),600-612, https://ece.uwaterloo.ca/~z70wang/research/ssim/. Source choices:Keys a=-.5,antialias widening,half-pixel centers,symmetric edges,column-then-row Float32 rounding,studio-range BT.601 Y and common central20x20 CIFAR crop. No benchmark claim.
/// @see SRBicubic(), ScoreTensorSR(), SRPreviewTiles()
inline Tensor SRCrop(const Tensor &x, int border) {
  if (x.dim() != 4 || border < 0 || x.size(1) <= 2LL * border || x.size(2) <= 2LL * border)
    throw std::invalid_argument("positive center-crop geometry required");
  return nn::slice(x, {0, border, border, 0},
                   {x.size(0), x.size(1) - 2 * border, x.size(2) - 2 * border, x.size(3)})
      .contiguous();
}
/// @brief Gather aligned reconstruction data by leading-row metadata.
///
/// No pixel arithmetic or host image copies.
///
/// @param x Tensor with leading batch axis.
/// @param rows Bounded row indices;reordering and duplicates allowed.
/// @return Gathered tensor,preserving graph and dtype.
/// @throws std::invalid_argument For invalid dimensions,dtype,nonfinite metric predictions or source row bounds.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto b = m::SRBatch(n::ones({3, 2, 2, 1}), {2, 0});
/// @endcode
/// @par References
/// - Keys(1981),Cubic Convolution Interpolation for Digital Image Processing,IEEE TASSP29(6),1153-1160,doi:10.1109/TASSP.1981.1163711;Wang et al.(2004),Image quality assessment:From error visibility to structural similarity,IEEE TIP13(4),600-612, https://ece.uwaterloo.ca/~z70wang/research/ssim/. Source choices:Keys a=-.5,antialias widening,half-pixel centers,symmetric edges,column-then-row Float32 rounding,studio-range BT.601 Y and common central20x20 CIFAR crop. No benchmark claim.
/// @see SRBicubic(), ScoreTensorSR(), SRPreviewTiles()
inline Tensor SRBatch(const Tensor &x, const std::vector<std::int64_t> &rows) {
  return nn::take_rows(x, rows);
}
/// @brief Own aligned high-resolution,low-resolution and bicubic reference tensors.
///
/// Tensor handles share/own storage. Original CIFAR32x32 is the HR truth.
///
/// @throws std::invalid_argument For invalid dimensions,dtype,nonfinite metric predictions or source row bounds.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto d = m::SRPairs(n::zeros({2, 32, 32, 3}, n::DType::Int32), {1});
/// @endcode
/// @par References
/// - Keys(1981),Cubic Convolution Interpolation for Digital Image Processing,IEEE TASSP29(6),1153-1160,doi:10.1109/TASSP.1981.1163711;Wang et al.(2004),Image quality assessment:From error visibility to structural similarity,IEEE TIP13(4),600-612, https://ece.uwaterloo.ca/~z70wang/research/ssim/. Source choices:Keys a=-.5,antialias widening,half-pixel centers,symmetric edges,column-then-row Float32 rounding,studio-range BT.601 Y and common central20x20 CIFAR crop. No benchmark claim.
/// @see SRBicubic(), ScoreTensorSR(), SRPreviewTiles()
struct SRData {
  Tensor hr;      ///< Raw normalized RGB[N,32,32,3].
  Tensor lr;      ///< True x2 downsampled RGB[N,16,16,3].
  Tensor bicubic; ///< Raw LR-to32x32 interpolation,not the HR target.
};
/// @brief Prepare real x2 degradation pairs from selected raw CIFAR image tensors.
///
/// Tensor normalization and antialiased two-stage resizing only. Source row
/// splits are caller policy;no labels,learned weights or hidden optimizer.
///
/// @param images Int32[N,32,32,3] original encoded bytes0..255.
/// @param rows Nonempty bounded original row metadata.
/// @return Aligned normalized HR32,LR16 and bicubic32 tensors.
/// @throws std::invalid_argument For invalid dimensions,dtype,nonfinite metric predictions or source row bounds.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto d = m::SRPairs(n::zeros({2, 32, 32, 3}, n::DType::Int32), {1});
/// @endcode
/// @par References
/// - Keys(1981),Cubic Convolution Interpolation for Digital Image Processing,IEEE TASSP29(6),1153-1160,doi:10.1109/TASSP.1981.1163711;Wang et al.(2004),Image quality assessment:From error visibility to structural similarity,IEEE TIP13(4),600-612, https://ece.uwaterloo.ca/~z70wang/research/ssim/. Source choices:Keys a=-.5,antialias widening,half-pixel centers,symmetric edges,column-then-row Float32 rounding,studio-range BT.601 Y and common central20x20 CIFAR crop. No benchmark claim.
/// @see SRBicubic(), ScoreTensorSR(), SRPreviewTiles()
inline SRData SRPairs(const Tensor &images, const std::vector<std::int64_t> &rows) {
  if (images.dim() != 4 || images.shape() != nn::Shape{images.size(0), 32, 32, 3} ||
      images.dtype() != nn::DType::Int32 || rows.empty())
    throw std::invalid_argument("nonempty source Int32 CIFAR RGB rows required");
  for (auto r : rows)
    if (r < 0 || r >= images.size(0))
      throw std::invalid_argument("SR source row bounds");
  const auto hr = SRBatch(images, rows).to(nn::DType::Float32) / 255.;
  if (!nn::all(nn::logical_and(nn::greater_equal(hr, 0.), nn::less_equal(hr, 1.)))
           .item<bool>())
    throw std::invalid_argument("source bytes outside0..255");
  const auto lr = SRBicubic(hr, 16, 16);
  return {hr, lr, SRBicubic(lr, 32, 32)};
}
/// @brief Convert unit-range MSE to PSNR in decibels.
///
/// No epsilon flooring or clipping;perfect reconstruction remains infinite.
///
/// @param mse Numeric finite nonnegative errors.
/// @return Float64 matching shape,-10log10(MSE),positive infinity at zero.
/// @throws std::invalid_argument For invalid dimensions,dtype,nonfinite metric predictions or source row bounds.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto p = m::SRPSNRTensor(n::tensor({.01, 1., 0.}));
/// @endcode
/// @par References
/// - Keys(1981),Cubic Convolution Interpolation for Digital Image Processing,IEEE TASSP29(6),1153-1160,doi:10.1109/TASSP.1981.1163711;Wang et al.(2004),Image quality assessment:From error visibility to structural similarity,IEEE TIP13(4),600-612, https://ece.uwaterloo.ca/~z70wang/research/ssim/. Source choices:Keys a=-.5,antialias widening,half-pixel centers,symmetric edges,column-then-row Float32 rounding,studio-range BT.601 Y and common central20x20 CIFAR crop. No benchmark claim.
/// @see SRBicubic(), ScoreTensorSR(), SRPreviewTiles()
inline Tensor SRPSNRTensor(const Tensor &mse) {
  if (!nn::all(nn::logical_and(nn::is_finite(mse), nn::greater_equal(mse, 0.))).item<bool>())
    throw std::invalid_argument("finite nonnegative MSE required");
  return -10. * nn::log(mse.to(nn::DType::Float64)) / std::log(10.);
}
/// @brief Present one tensor PSNR as a Double.
///
/// Legacy scalar compatibility;all numerical work delegates to tensors.
///
/// @param mse Finite nonnegative unit-range error.
/// @return PSNR in dB,infinite if zero.
/// @throws std::invalid_argument For invalid dimensions,dtype,nonfinite metric predictions or source row bounds.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto p = m::SRPSNR(.01);
/// @endcode
/// @par References
/// - Keys(1981),Cubic Convolution Interpolation for Digital Image Processing,IEEE TASSP29(6),1153-1160,doi:10.1109/TASSP.1981.1163711;Wang et al.(2004),Image quality assessment:From error visibility to structural similarity,IEEE TIP13(4),600-612, https://ece.uwaterloo.ca/~z70wang/research/ssim/. Source choices:Keys a=-.5,antialias widening,half-pixel centers,symmetric edges,column-then-row Float32 rounding,studio-range BT.601 Y and common central20x20 CIFAR crop. No benchmark claim.
/// @see SRBicubic(), ScoreTensorSR(), SRPreviewTiles()
inline double SRPSNR(double mse) {
  return SRPSNRTensor(nn::full({}, mse, nn::DType::Float64)).item();
}
/// @brief Compute one valid-window Gaussian SSIM per image using tensor moments.
///
/// Gaussian sigma1.5,normalized11x11 window,C1=.0001,C2=.0009,unit range.
/// Population variance is clamped at0 against roundoff,covariance is signed.
/// No border padding,scale averaging or hidden clipping.
///
/// @param a Numeric[N,H,W,1],H/W>=11.
/// @param b Matching reference images.
/// @return Float64[N] mean SSIM over valid11x11 centers.
/// @throws std::invalid_argument For invalid dimensions,dtype,nonfinite metric predictions or source row bounds.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto x = n::full({2, 20, 20, 1}, .2);
/// const auto s = m::SRSSIMTensor(x, x);
/// @endcode
/// @par References
/// - Keys(1981),Cubic Convolution Interpolation for Digital Image Processing,IEEE TASSP29(6),1153-1160,doi:10.1109/TASSP.1981.1163711;Wang et al.(2004),Image quality assessment:From error visibility to structural similarity,IEEE TIP13(4),600-612, https://ece.uwaterloo.ca/~z70wang/research/ssim/. Source choices:Keys a=-.5,antialias widening,half-pixel centers,symmetric edges,column-then-row Float32 rounding,studio-range BT.601 Y and common central20x20 CIFAR crop. No benchmark claim.
/// @see SRBicubic(), ScoreTensorSR(), SRPreviewTiles()
inline Tensor SRSSIMTensor(const Tensor &a, const Tensor &b) {
  if (a.dim() != 4 || a.shape() != b.shape() || a.size(0) < 1 || a.size(1) < 11 ||
      a.size(2) < 11 || a.size(3) != 1)
    throw std::invalid_argument("aligned NHW1 SSIM with11x11 support required");
  const auto x = a.to(nn::DType::Float64), y = b.to(nn::DType::Float64),
             offset = nn::arange(-5, 6, 1, nn::DType::Float64);
  auto k = nn::exp(-offset * offset / 4.5);
  k = k / nn::sum(k);
  const auto kernel = (k.reshape({11, 1}) * k.reshape({1, 11})).reshape({11, 11, 1, 1});
  /// @brief Compute valid11x11 Gaussian local expectations.
  /// @param z Float64 scalar image field.
  /// @return Float64 valid local expectation map.
  auto average = [&](const Tensor &z) {
    return nn::conv2d(z, kernel, nn::conv_options(1, nn::Padding::Valid));
  };
  const auto mx = average(x), my = average(y),
             vx = nn::maximum(average(x * x) - mx * mx, nn::zeros_like(mx)),
             vy = nn::maximum(average(y * y) - my * my, nn::zeros_like(my)),
             cov = average(x * y) - mx * my;
  return nn::mean(((2. * mx * my + .0001) * (2. * cov + .0009)) /
                      ((mx * mx + my * my + .0001) * (vx + vy + .0009)),
                  {1, 2, 3});
}
/// @brief Present a legacy pair of scalar images through tensor SSIM.
///
/// Host vectors are ingress only;all moments and Gaussian weighting use tensors.
///
/// @param a First row-major unit-range image.
/// @param b Matching second image.
/// @param h Height>=11.
/// @param w Width>=11.
/// @return Mean valid-window SSIM.
/// @throws std::invalid_argument For invalid dimensions,dtype,nonfinite metric predictions or source row bounds.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto s = m::SRSSIM(std::vector<double>(400, .2), std::vector<double>(400, .3), 20, 20);
/// @endcode
/// @par References
/// - Keys(1981),Cubic Convolution Interpolation for Digital Image Processing,IEEE TASSP29(6),1153-1160,doi:10.1109/TASSP.1981.1163711;Wang et al.(2004),Image quality assessment:From error visibility to structural similarity,IEEE TIP13(4),600-612, https://ece.uwaterloo.ca/~z70wang/research/ssim/. Source choices:Keys a=-.5,antialias widening,half-pixel centers,symmetric edges,column-then-row Float32 rounding,studio-range BT.601 Y and common central20x20 CIFAR crop. No benchmark claim.
/// @see SRBicubic(), ScoreTensorSR(), SRPreviewTiles()
inline double SRSSIM(const std::vector<double> &a, const std::vector<double> &b, int h,
                     int w) {
  if (h < 11 || w < 11 || a.size() != std::size_t(h) * w || b.size() != a.size())
    throw std::invalid_argument("SSIM vector shape mismatch");
  return SRSSIMTensor(nn::from_vector(a, {1, h, w, 1}), nn::from_vector(b, {1, h, w, 1}))
      .item();
}
/// @brief Own common-coordinate Y reconstruction metrics. Tensor values.
///
/// Means weight images equally;mean PSNR differs from PSNR of pooled MSE.
///
/// @throws std::invalid_argument For invalid dimensions,dtype,nonfinite metric predictions or source row bounds.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto d = m::SRPairs(n::zeros({1, 32, 32, 3}, n::DType::Int32), {0});
/// const auto s = m::ScoreTensorSR(d.hr, d);
/// @endcode
/// @par References
/// - Keys(1981),Cubic Convolution Interpolation for Digital Image Processing,IEEE TASSP29(6),1153-1160,doi:10.1109/TASSP.1981.1163711;Wang et al.(2004),Image quality assessment:From error visibility to structural similarity,IEEE TIP13(4),600-612, https://ece.uwaterloo.ca/~z70wang/research/ssim/. Source choices:Keys a=-.5,antialias widening,half-pixel centers,symmetric edges,column-then-row Float32 rounding,studio-range BT.601 Y and common central20x20 CIFAR crop. No benchmark claim.
/// @see SRBicubic(), ScoreTensorSR(), SRPreviewTiles()
struct TensorSRMetrics {
  Tensor psnr;          ///< Mean per-image prediction PSNR,dB.
  Tensor ssim;          ///< Mean per-image prediction valid Gaussian SSIM.
  Tensor mse;           ///< Pooled unit-Y mean squared error.
  Tensor baseline_psnr; ///< Mean per-image bicubic PSNR,dB.
  Tensor baseline_ssim; ///< Mean per-image bicubic SSIM.
};
/// @brief Own common-coordinate Y reconstruction metrics. Host presentation.
///
/// Means weight images equally;mean PSNR differs from PSNR of pooled MSE.
///
/// @throws std::invalid_argument For invalid dimensions,dtype,nonfinite metric predictions or source row bounds.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto d = m::SRPairs(n::zeros({1, 32, 32, 3}, n::DType::Int32), {0});
/// const auto s = m::SRScore(d.hr, d);
/// @endcode
/// @par References
/// - Keys(1981),Cubic Convolution Interpolation for Digital Image Processing,IEEE TASSP29(6),1153-1160,doi:10.1109/TASSP.1981.1163711;Wang et al.(2004),Image quality assessment:From error visibility to structural similarity,IEEE TIP13(4),600-612, https://ece.uwaterloo.ca/~z70wang/research/ssim/. Source choices:Keys a=-.5,antialias widening,half-pixel centers,symmetric edges,column-then-row Float32 rounding,studio-range BT.601 Y and common central20x20 CIFAR crop. No benchmark claim.
/// @see SRBicubic(), ScoreTensorSR(), SRPreviewTiles()
struct SRMetrics {
  double psnr = 0;          ///< Mean per-image prediction PSNR,dB.
  double ssim = 0;          ///< Mean per-image prediction valid Gaussian SSIM.
  double mse = 0;           ///< Pooled unit-Y mean squared error.
  double baseline_psnr = 0; ///< Mean per-image bicubic PSNR,dB.
  double baseline_ssim = 0; ///< Mean per-image bicubic SSIM.
};
/// @brief Align estimates,truth and bicubic to identical central studio-range Y coordinates.
///
/// Preserves source Float32 image ingress. RGB estimates convert to Y before
/// clipping;one-channel estimates already represent Y. Only evaluation/display
/// clips estimates;training tensors remain raw.
///
/// @param prediction Float prediction[N,20/32,20/32,1/3].
/// @param data Aligned original HR32 RGB and bicubic32 RGB reference.
/// @return Float64 array{clipped prediction,raw truth,clipped bicubic},NHW1.
/// @throws std::invalid_argument For invalid dimensions,dtype,nonfinite metric predictions or source row bounds.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto d = m::SRPairs(n::zeros({1, 32, 32, 3}, n::DType::Int32), {0});
/// const auto a = m::SRScoreImages(d.hr, d);
/// @endcode
/// @par References
/// - Keys(1981),Cubic Convolution Interpolation for Digital Image Processing,IEEE TASSP29(6),1153-1160,doi:10.1109/TASSP.1981.1163711;Wang et al.(2004),Image quality assessment:From error visibility to structural similarity,IEEE TIP13(4),600-612, https://ece.uwaterloo.ca/~z70wang/research/ssim/. Source choices:Keys a=-.5,antialias widening,half-pixel centers,symmetric edges,column-then-row Float32 rounding,studio-range BT.601 Y and common central20x20 CIFAR crop. No benchmark claim.
/// @see SRBicubic(), ScoreTensorSR(), SRPreviewTiles()
inline std::array<Tensor, 3> SRScoreImages(const Tensor &prediction, const SRData &data) {
  if (data.hr.dim() != 4 || data.hr.size(0) < 1 ||
      data.hr.shape() != nn::Shape{data.hr.size(0), 32, 32, 3} ||
      data.bicubic.shape() != data.hr.shape() || prediction.dim() != 4 ||
      prediction.size(0) != data.hr.size(0) ||
      (prediction.size(1) != 20 && prediction.size(1) != 32) ||
      prediction.size(2) != prediction.size(1) ||
      (prediction.size(3) != 1 && prediction.size(3) != 3))
    throw std::invalid_argument("common SR central20 geometry required");
  auto p =
      (prediction.size(1) == 20 ? prediction : SRCrop(prediction, 6)).to(nn::DType::Float32);
  auto estimated = p.size(3) == 1 ? p.to(nn::DType::Float64) : SRLumaTensor(p);
  if (!nn::all(nn::is_finite(estimated)).item<bool>())
    throw std::invalid_argument("finite raw central SR estimates required");
  const auto truth = SRLumaTensor(SRCrop(data.hr.to(nn::DType::Float32), 6)),
             baseline = nn::clip(SRLumaTensor(SRCrop(data.bicubic.to(nn::DType::Float32), 6)),
                                 0., 1.);
  return {nn::clip(estimated, 0., 1.), truth, baseline};
}
/// @brief Score all images with common-crop Y PSNR/SSIM against bicubic.
///
/// Central coordinates6..25 of HR are compared with SRCNN20 output directly,
/// or with the same crop of32-grid outputs. Mean per-image PSNR/SSIM and pooled
/// MSE use identical images. Perfect prediction gives infinite PSNR;negative
/// SSIM remains meaningful. No hidden fitting,test tuning or target alignment.
///
/// @param prediction Aligned raw Float32 NHWC estimates,20/32 grid,one Y or three RGB channels.
/// @param data Original32x32 HR/bicubic RGB pairs.
/// @return Five detached Float64 scalar metrics.
/// @throws std::invalid_argument For invalid dimensions,dtype,nonfinite metric predictions or source row bounds.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto d = m::SRPairs(n::zeros({1, 32, 32, 3}, n::DType::Int32), {0});
/// const auto s = m::ScoreTensorSR(d.hr, d);
/// @endcode
/// @par References
/// - Keys(1981),Cubic Convolution Interpolation for Digital Image Processing,IEEE TASSP29(6),1153-1160,doi:10.1109/TASSP.1981.1163711;Wang et al.(2004),Image quality assessment:From error visibility to structural similarity,IEEE TIP13(4),600-612, https://ece.uwaterloo.ca/~z70wang/research/ssim/. Source choices:Keys a=-.5,antialias widening,half-pixel centers,symmetric edges,column-then-row Float32 rounding,studio-range BT.601 Y and common central20x20 CIFAR crop. No benchmark claim.
/// @see SRBicubic(), ScoreTensorSR(), SRPreviewTiles()
inline TensorSRMetrics ScoreTensorSR(const Tensor &prediction, const SRData &data) {
  nn::NoGradGuard guard;
  const auto image = SRScoreImages(prediction, data);
  const auto mse = nn::mean(nn::square(image[0] - image[1]), {1, 2, 3}),
             bmse = nn::mean(nn::square(image[2] - image[1]), {1, 2, 3});
  return {nn::mean(SRPSNRTensor(mse)), nn::mean(SRSSIMTensor(image[0], image[1])),
          nn::mean(mse), nn::mean(SRPSNRTensor(bmse)),
          nn::mean(SRSSIMTensor(image[2], image[1]))};
}
/// @brief Extract legacy host presentation scalars from tensor reconstruction scores.
///
/// All numerical computation is in ScoreTensorSR.
///
/// @param prediction Aligned raw reconstruction.
/// @param data Original HR/bicubic pairs.
/// @return Five host presentation scalars.
/// @throws std::invalid_argument For invalid dimensions,dtype,nonfinite metric predictions or source row bounds.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto d = m::SRPairs(n::zeros({1, 32, 32, 3}, n::DType::Int32), {0});
/// const auto s = m::SRScore(d.hr, d);
/// @endcode
/// @par References
/// - Keys(1981),Cubic Convolution Interpolation for Digital Image Processing,IEEE TASSP29(6),1153-1160,doi:10.1109/TASSP.1981.1163711;Wang et al.(2004),Image quality assessment:From error visibility to structural similarity,IEEE TIP13(4),600-612, https://ece.uwaterloo.ca/~z70wang/research/ssim/. Source choices:Keys a=-.5,antialias widening,half-pixel centers,symmetric edges,column-then-row Float32 rounding,studio-range BT.601 Y and common central20x20 CIFAR crop. No benchmark claim.
/// @see SRBicubic(), ScoreTensorSR(), SRPreviewTiles()
inline SRMetrics SRScore(const Tensor &prediction, const SRData &data) {
  const auto r = ScoreTensorSR(prediction, data);
  return {r.psnr.item(), r.ssim.item(), r.mse.item(), r.baseline_psnr.item(),
          r.baseline_ssim.item()};
}
/// @brief Own architecture-aligned training input and target handles.
///
/// No optimizer or fitted preprocessing. Targets never enter inference forward.
///
/// @throws std::invalid_argument For invalid dimensions,dtype,nonfinite metric predictions or source row bounds.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto d = m::SRPairs(n::zeros({1, 32, 32, 3}, n::DType::Int32), {0});
/// const auto a = m::PrepareSR(d, m::SRConfig::Compact(m::SRKind::kSrcnn));
/// @endcode
/// @par References
/// - Keys(1981),Cubic Convolution Interpolation for Digital Image Processing,IEEE TASSP29(6),1153-1160,doi:10.1109/TASSP.1981.1163711;Wang et al.(2004),Image quality assessment:From error visibility to structural similarity,IEEE TIP13(4),600-612, https://ece.uwaterloo.ca/~z70wang/research/ssim/. Source choices:Keys a=-.5,antialias widening,half-pixel centers,symmetric edges,column-then-row Float32 rounding,studio-range BT.601 Y and common central20x20 CIFAR crop. No benchmark claim.
/// @see SRBicubic(), ScoreTensorSR(), SRPreviewTiles()
struct SRPrepared {
  Tensor x; ///< Observed input,HR bicubic or LR,Y or RGB.
  Tensor y; ///< Original HR target,Y cropped for SRCNN only.
};
/// @brief Choose observed image grid and align source targets for one architecture.
///
/// SRCNN/VDSR consume bicubic32 Y;ESPCN consumes LR16 Y. EDSR/RCAN consume
/// LR16 RGB. Only SRCNN target loses six pixels on each side.
///
/// @param data Genuine32/16 x2 RGB pairs.
/// @param cfg Architecture metadata.
/// @return Inputs/targets with model-specific channels and valid-convolution crop.
/// @throws std::invalid_argument For invalid dimensions,dtype,nonfinite metric predictions or source row bounds.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto d = m::SRPairs(n::zeros({1, 32, 32, 3}, n::DType::Int32), {0});
/// const auto a = m::PrepareSR(d, m::SRConfig::Compact(m::SRKind::kSrcnn));
/// @endcode
/// @par References
/// - Keys(1981),Cubic Convolution Interpolation for Digital Image Processing,IEEE TASSP29(6),1153-1160,doi:10.1109/TASSP.1981.1163711;Wang et al.(2004),Image quality assessment:From error visibility to structural similarity,IEEE TIP13(4),600-612, https://ece.uwaterloo.ca/~z70wang/research/ssim/. Source choices:Keys a=-.5,antialias widening,half-pixel centers,symmetric edges,column-then-row Float32 rounding,studio-range BT.601 Y and common central20x20 CIFAR crop. No benchmark claim.
/// @see SRBicubic(), ScoreTensorSR(), SRPreviewTiles()
inline SRPrepared PrepareSR(const SRData &data, const SRConfig &cfg) {
  Tensor x = cfg.PreUpsampled() ? data.bicubic : data.lr, y = data.hr;
  if (cfg.Channels() == 1)
    x = SRY(x), y = SRY(y);
  if (cfg.kind == SRKind::kSrcnn)
    y = SRCrop(y, 6);
  return {x, y};
}
/// @brief Render all central-crop Y comparisons as four-panel image tiles.
///
/// Tensor quantization reproduces source half-up byte display. Metric clipping
/// policy is shared. No SVG markup or encoded image payload is printed.
///
/// @param data Original HR/bicubic pairs.
/// @param prediction Aligned raw reconstruction.
/// @return Float32[N,20,80,3] bicubic|estimate|truth|8x absolute red error.
/// @throws std::invalid_argument For invalid dimensions,dtype,nonfinite metric predictions or source row bounds.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto d = m::SRPairs(n::zeros({1, 32, 32, 3}, n::DType::Int32), {0});
/// const auto tiles = m::SRPreviewTiles(d, d.hr);
/// @endcode
/// @par References
/// - Keys(1981),Cubic Convolution Interpolation for Digital Image Processing,IEEE TASSP29(6),1153-1160,doi:10.1109/TASSP.1981.1163711;Wang et al.(2004),Image quality assessment:From error visibility to structural similarity,IEEE TIP13(4),600-612, https://ece.uwaterloo.ca/~z70wang/research/ssim/. Source choices:Keys a=-.5,antialias widening,half-pixel centers,symmetric edges,column-then-row Float32 rounding,studio-range BT.601 Y and common central20x20 CIFAR crop. No benchmark claim.
/// @see SRBicubic(), ScoreTensorSR(), SRPreviewTiles()
inline Tensor SRPreviewTiles(const SRData &data, const Tensor &prediction) {
  const auto a = SRScoreImages(prediction, data);
  const auto error = nn::clip(8. * nn::abs(a[0] - a[1]), 0., 1.), zero = nn::zeros_like(error);
  const auto tiles =
      nn::concat({nn::concat({a[2], a[2], a[2]}, 3), nn::concat({a[0], a[0], a[0]}, 3),
                  nn::concat({a[1], a[1], a[1]}, 3), nn::concat({error, zero, zero}, 3)},
                 2);
  return (nn::floor(255. * tiles + .5) / 255.).to(nn::DType::Float32);
}
} // namespace datamunge::nn::models
