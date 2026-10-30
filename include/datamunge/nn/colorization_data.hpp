#pragma once
/// @file colorization_data.hpp
/// @brief Tensor sRGB/CIELAB transforms,lightness-only training pairs and chroma diagnostics.
#include "datamunge/nn/autograd.hpp"
#include "datamunge/nn/ops.hpp"
#include "datamunge/nn/training.hpp"
#include <array>
#include <cmath>
#include <limits>
#include <vector>
namespace datamunge::nn::models {
namespace nn = datamunge::nn;
/// @brief Represent one RGB or Lab triple at the compatibility boundary.
///
/// Three host presentation scalars;canonical numerical transforms use Float64 tensors.
///
/// @throws std::invalid_argument For invalid shape,dtype,rows,range or nonfinite color values.
/// @par Example
/// @code
/// const dm::nn::models::ColorTriple rgb{1., 0., 0.};
/// @endcode
/// @par References
/// - Scikit-image v0.25.2 colorconv.py, https://github.com/scikit-image/scikit-image/blob/v0.25.2/skimage/color/colorconv.py. sRGB matrices,transfer thresholds and D65/2degree white follow the source. This example performs Lab chroma regression,not Zhang classification.
/// @see ColorPairs(), ColorRGBToLabTensor(), ScoreTensorColor()
using ColorTriple = std::array<double, 3>;
/// @brief Internal shape,dtype and finite-value validation for color triples.
namespace color_detail {
/// @brief Require nonempty floating values with a last dimension of three.
/// @param x RGB/Lab tensor with floating channel values.
inline void ValidateTriples(const nn::Tensor &x) {
  if (x.dim() < 1 || x.shape().back() != 3 || x.numel() < 1 ||
      (x.dtype() != nn::DType::Float32 && x.dtype() != nn::DType::Float64) ||
      !nn::all(nn::less(nn::abs(x), std::numeric_limits<double>::infinity())).item<bool>())
    throw std::invalid_argument("finite floating color triples required");
}
} // namespace color_detail
/// @brief Decode the sRGB transfer function to linear light.
///
/// Threshold.04045;linear branch x/12.92,otherwise((x+.055)/1.055)^2.4.
/// Inactive nonlinear branches use a positive floor only outside their active domain.
/// No fitting,parameter mutation or display clipping.
///
/// @param value Numeric tensor;finite values are the supported contract.
/// @return Float64 matching-shape transform,unclipped,graph retained.
/// @throws std::invalid_argument For invalid shape,dtype,rows,range or nonfinite color values.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto y = n::models::ColorLinearTensor(n::tensor({0., .01, .5, 1.}));
/// @endcode
/// @par References
/// - Scikit-image v0.25.2 colorconv.py, https://github.com/scikit-image/scikit-image/blob/v0.25.2/skimage/color/colorconv.py. sRGB matrices,transfer thresholds and D65/2degree white follow the source. This example performs Lab chroma regression,not Zhang classification.
/// @see ColorPairs(), ColorRGBToLabTensor(), ScoreTensorColor()
inline nn::Tensor ColorLinearTensor(const nn::Tensor &value) {
  const auto x = value.to(nn::DType::Float64);
  return nn::where(
      nn::less_equal(x, .04045), x / 12.92,
      nn::pow(nn::maximum((x + .055) / 1.055, nn::full(x.shape(), 1e-30, nn::DType::Float64)),
              2.4));
}
/// @brief Decode the sRGB transfer function to linear light. Scalar compatibility overload.
///
/// Threshold.04045;linear branch x/12.92,otherwise((x+.055)/1.055)^2.4. No fitted parameters.
///
/// @param value Scalar color-channel coordinate.
/// @return Transformed double,computed by the tensor implementation.
/// @throws std::invalid_argument For invalid shape,dtype,rows,range or nonfinite color values.
/// @par Example
/// @code
/// const auto y = dm::nn::models::ColorLinear(.5);
/// @endcode
/// @par References
/// - Scikit-image v0.25.2 colorconv.py, https://github.com/scikit-image/scikit-image/blob/v0.25.2/skimage/color/colorconv.py. sRGB matrices,transfer thresholds and D65/2degree white follow the source. This example performs Lab chroma regression,not Zhang classification.
/// @see ColorPairs(), ColorRGBToLabTensor(), ScoreTensorColor()
inline double ColorLinear(double value) {
  return ColorLinearTensor(nn::full({}, value, nn::DType::Float64)).item();
}
/// @brief Encode linear light with the sRGB transfer function.
///
/// Threshold.0031308;negative values use the linear branch and remain unclipped.
/// Inactive nonlinear branches use a positive floor only outside their active domain.
/// No fitting,parameter mutation or display clipping.
///
/// @param value Numeric tensor;finite values are the supported contract.
/// @return Float64 matching-shape transform,unclipped,graph retained.
/// @throws std::invalid_argument For invalid shape,dtype,rows,range or nonfinite color values.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto y = n::models::ColorGammaTensor(n::tensor({0., .01, .5, 1.}));
/// @endcode
/// @par References
/// - Scikit-image v0.25.2 colorconv.py, https://github.com/scikit-image/scikit-image/blob/v0.25.2/skimage/color/colorconv.py. sRGB matrices,transfer thresholds and D65/2degree white follow the source. This example performs Lab chroma regression,not Zhang classification.
/// @see ColorPairs(), ColorRGBToLabTensor(), ScoreTensorColor()
inline nn::Tensor ColorGammaTensor(const nn::Tensor &value) {
  const auto x = value.to(nn::DType::Float64);
  return nn::where(
      nn::less_equal(x, .0031308), 12.92 * x,
      1.055 * nn::pow(nn::maximum(x, nn::full(x.shape(), 1e-30, nn::DType::Float64)),
                      1. / 2.4) -
          .055);
}
/// @brief Encode linear light with the sRGB transfer function. Scalar compatibility overload.
///
/// Threshold.0031308;negative values use the linear branch and remain unclipped. No fitted parameters.
///
/// @param value Scalar color-channel coordinate.
/// @return Transformed double,computed by the tensor implementation.
/// @throws std::invalid_argument For invalid shape,dtype,rows,range or nonfinite color values.
/// @par Example
/// @code
/// const auto y = dm::nn::models::ColorGamma(.5);
/// @endcode
/// @par References
/// - Scikit-image v0.25.2 colorconv.py, https://github.com/scikit-image/scikit-image/blob/v0.25.2/skimage/color/colorconv.py. sRGB matrices,transfer thresholds and D65/2degree white follow the source. This example performs Lab chroma regression,not Zhang classification.
/// @see ColorPairs(), ColorRGBToLabTensor(), ScoreTensorColor()
inline double ColorGamma(double value) {
  return ColorGammaTensor(nn::full({}, value, nn::DType::Float64)).item();
}
/// @brief Apply the CIELAB piecewise cube-root transform.
///
/// Threshold(6/29)^3;linear continuation below the threshold avoids singularities.
/// Inactive nonlinear branches use a positive floor only outside their active domain.
/// No fitting,parameter mutation or display clipping.
///
/// @param value Numeric tensor;finite values are the supported contract.
/// @return Float64 matching-shape transform,unclipped,graph retained.
/// @throws std::invalid_argument For invalid shape,dtype,rows,range or nonfinite color values.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto y = n::models::ColorLabFTensor(n::tensor({0., .01, .5, 1.}));
/// @endcode
/// @par References
/// - Scikit-image v0.25.2 colorconv.py, https://github.com/scikit-image/scikit-image/blob/v0.25.2/skimage/color/colorconv.py. sRGB matrices,transfer thresholds and D65/2degree white follow the source. This example performs Lab chroma regression,not Zhang classification.
/// @see ColorPairs(), ColorRGBToLabTensor(), ScoreTensorColor()
inline nn::Tensor ColorLabFTensor(const nn::Tensor &value) {
  const auto x = value.to(nn::DType::Float64);
  return nn::where(
      nn::greater(x, (6. / 29) * (6. / 29) * (6. / 29)),
      nn::pow(nn::maximum(x, nn::full(x.shape(), 1e-30, nn::DType::Float64)), 1. / 3),
      x / (3 * (6. / 29) * (6. / 29)) + 4. / 29);
}
/// @brief Apply the CIELAB piecewise cube-root transform. Scalar compatibility overload.
///
/// Threshold(6/29)^3;linear continuation below the threshold avoids singularities. No fitted parameters.
///
/// @param value Scalar color-channel coordinate.
/// @return Transformed double,computed by the tensor implementation.
/// @throws std::invalid_argument For invalid shape,dtype,rows,range or nonfinite color values.
/// @par Example
/// @code
/// const auto y = dm::nn::models::ColorLabF(.5);
/// @endcode
/// @par References
/// - Scikit-image v0.25.2 colorconv.py, https://github.com/scikit-image/scikit-image/blob/v0.25.2/skimage/color/colorconv.py. sRGB matrices,transfer thresholds and D65/2degree white follow the source. This example performs Lab chroma regression,not Zhang classification.
/// @see ColorPairs(), ColorRGBToLabTensor(), ScoreTensorColor()
inline double ColorLabF(double value) {
  return ColorLabFTensor(nn::full({}, value, nn::DType::Float64)).item();
}
/// @brief Invert the CIELAB piecewise cube-root transform.
///
/// Threshold6/29;cube above,linear continuation below.
/// Inactive nonlinear branches use a positive floor only outside their active domain.
/// No fitting,parameter mutation or display clipping.
///
/// @param value Numeric tensor;finite values are the supported contract.
/// @return Float64 matching-shape transform,unclipped,graph retained.
/// @throws std::invalid_argument For invalid shape,dtype,rows,range or nonfinite color values.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto y = n::models::ColorLabInverseTensor(n::tensor({0., .01, .5, 1.}));
/// @endcode
/// @par References
/// - Scikit-image v0.25.2 colorconv.py, https://github.com/scikit-image/scikit-image/blob/v0.25.2/skimage/color/colorconv.py. sRGB matrices,transfer thresholds and D65/2degree white follow the source. This example performs Lab chroma regression,not Zhang classification.
/// @see ColorPairs(), ColorRGBToLabTensor(), ScoreTensorColor()
inline nn::Tensor ColorLabInverseTensor(const nn::Tensor &value) {
  const auto x = value.to(nn::DType::Float64);
  return nn::where(nn::greater(x, 6. / 29), x * x * x,
                   3 * (6. / 29) * (6. / 29) * (x - 4. / 29));
}
/// @brief Invert the CIELAB piecewise cube-root transform. Scalar compatibility overload.
///
/// Threshold6/29;cube above,linear continuation below. No fitted parameters.
///
/// @param value Scalar color-channel coordinate.
/// @return Transformed double,computed by the tensor implementation.
/// @throws std::invalid_argument For invalid shape,dtype,rows,range or nonfinite color values.
/// @par Example
/// @code
/// const auto y = dm::nn::models::ColorLabInverse(.5);
/// @endcode
/// @par References
/// - Scikit-image v0.25.2 colorconv.py, https://github.com/scikit-image/scikit-image/blob/v0.25.2/skimage/color/colorconv.py. sRGB matrices,transfer thresholds and D65/2degree white follow the source. This example performs Lab chroma regression,not Zhang classification.
/// @see ColorPairs(), ColorRGBToLabTensor(), ScoreTensorColor()
inline double ColorLabInverse(double value) {
  return ColorLabInverseTensor(nn::full({}, value, nn::DType::Float64)).item();
}
/// @brief Convert sRGB triples to D65/2degree CIELAB with tensor arithmetic.
///
/// Gamma decoding precedes the scikit-image RGB->XYZ matrix and D65 normalization.
/// No trainable state,optimizer or input mutation;color conversion is not fitting.
///
/// @param rgb Finite Float32/64[...,3] RGB in[0,1],nonempty.
/// @return Float64[...,3] physical L,a,b coordinates,graph retained.
/// @throws std::invalid_argument For invalid shape,dtype,rows,range or nonfinite color values.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto lab = n::models::ColorRGBToLabTensor(n::tensor({1., 0., 0.}));
/// @endcode
/// @par References
/// - Scikit-image v0.25.2 colorconv.py, https://github.com/scikit-image/scikit-image/blob/v0.25.2/skimage/color/colorconv.py. sRGB matrices,transfer thresholds and D65/2degree white follow the source. This example performs Lab chroma regression,not Zhang classification.
/// @see ColorPairs(), ColorRGBToLabTensor(), ScoreTensorColor()
inline nn::Tensor ColorRGBToLabTensor(const nn::Tensor &rgb) {
  color_detail::ValidateTriples(rgb);
  const auto x = rgb.to(nn::DType::Float64);
  if (!nn::all(nn::logical_and(nn::greater_equal(x, 0.), nn::less_equal(x, 1.))).item<bool>())
    throw std::invalid_argument("sRGB unit range required");
  const auto matrix =
      nn::from_vector(std::vector<double>{.412453, .212671, .019334, .357580, .715160, .119193,
                                          .180423, .072169, .950227},
                      {3, 3});
  const auto xyz =
      nn::matmul(ColorLinearTensor(x).reshape({-1, 3}), matrix).reshape(x.shape());
  const auto f =
      ColorLabFTensor(xyz / nn::from_vector(std::vector<double>{.95047, 1., 1.08883}, {3}));
  const auto axis = x.dim() - 1;
  const auto fx = nn::slice_axis(f, axis, 0, 1), fy = nn::slice_axis(f, axis, 1, 2),
             fz = nn::slice_axis(f, axis, 2, 3);
  return nn::concat({116. * fy - 16., 500. * (fx - fy), 200. * (fy - fz)}, axis);
}
/// @brief Convert D65 Lab triples to unclipped sRGB with tensor arithmetic.
///
/// XYZ inverse and gamma retain out-of-gamut values. Clip only for presentation
/// and specified RGB metrics;no optimizer or model mutation.
///
/// @param lab Finite Float32/64[...,3] physical L,a,b coordinates,nonempty.
/// @return Float64[...,3] sRGB,possibly outside[0,1],graph retained.
/// @throws std::invalid_argument For invalid shape,dtype,rows,range or nonfinite color values.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto rgb = n::models::ColorLabToRGBTensor(n::tensor({50., 128., -128.}));
/// @endcode
/// @par References
/// - Scikit-image v0.25.2 colorconv.py, https://github.com/scikit-image/scikit-image/blob/v0.25.2/skimage/color/colorconv.py. sRGB matrices,transfer thresholds and D65/2degree white follow the source. This example performs Lab chroma regression,not Zhang classification.
/// @see ColorPairs(), ColorRGBToLabTensor(), ScoreTensorColor()
inline nn::Tensor ColorLabToRGBTensor(const nn::Tensor &lab) {
  color_detail::ValidateTriples(lab);
  const auto x = lab.to(nn::DType::Float64);
  const auto axis = x.dim() - 1;
  const auto y = (nn::slice_axis(x, axis, 0, 1) + 16.) / 116.,
             a = nn::slice_axis(x, axis, 1, 2), b = nn::slice_axis(x, axis, 2, 3);
  const auto xyz =
      nn::concat({.95047 * ColorLabInverseTensor(y + a / 500.), ColorLabInverseTensor(y),
                  1.08883 * ColorLabInverseTensor(y - b / 200.)},
                 axis);
  const auto matrix = nn::from_vector(
      std::vector<double>{3.2404813432005266, -.9692549499965684, .0556466391351772,
                          -1.5371515162713185, 1.8759900014898907, -.2040413383665112,
                          -.4985363261688878, .0415559265582928, 1.0573110696453443},
      {3, 3});
  return ColorGammaTensor(nn::matmul(xyz.reshape({-1, 3}), matrix).reshape(x.shape()));
}
/// @brief Convert one sRGB color to Lab.
///
/// Compatibility ingress/egress only;all channel arithmetic uses tensors.
///
/// @param value Three finite RGB[0,1] or physical Lab presentation coordinates.
/// @return Three doubles from the canonical tensor conversion.
/// @throws std::invalid_argument For invalid shape,dtype,rows,range or nonfinite color values.
/// @par Example
/// @code
/// const auto value = dm::nn::models::ColorRGBToLab({1., 0., 0.});
/// @endcode
/// @par References
/// - Scikit-image v0.25.2 colorconv.py, https://github.com/scikit-image/scikit-image/blob/v0.25.2/skimage/color/colorconv.py. sRGB matrices,transfer thresholds and D65/2degree white follow the source. This example performs Lab chroma regression,not Zhang classification.
/// @see ColorPairs(), ColorRGBToLabTensor(), ScoreTensorColor()
inline ColorTriple ColorRGBToLab(const ColorTriple &value) {
  const auto out = ColorRGBToLabTensor(
                       nn::from_vector(std::vector<double>(value.begin(), value.end()), {3}))
                       .to_vector<double>();
  return {out[0], out[1], out[2]};
}
/// @brief Convert one Lab color to unclipped sRGB.
///
/// Compatibility ingress/egress only;all channel arithmetic uses tensors.
///
/// @param value Three finite RGB[0,1] or physical Lab presentation coordinates.
/// @return Three doubles from the canonical tensor conversion.
/// @throws std::invalid_argument For invalid shape,dtype,rows,range or nonfinite color values.
/// @par Example
/// @code
/// const auto value = dm::nn::models::ColorLabToRGB({50., 0., 0.});
/// @endcode
/// @par References
/// - Scikit-image v0.25.2 colorconv.py, https://github.com/scikit-image/scikit-image/blob/v0.25.2/skimage/color/colorconv.py. sRGB matrices,transfer thresholds and D65/2degree white follow the source. This example performs Lab chroma regression,not Zhang classification.
/// @see ColorPairs(), ColorRGBToLabTensor(), ScoreTensorColor()
inline ColorTriple ColorLabToRGB(const ColorTriple &value) {
  const auto out = ColorLabToRGBTensor(
                       nn::from_vector(std::vector<double>(value.begin(), value.end()), {3}))
                       .to_vector<double>();
  return {out[0], out[1], out[2]};
}
/// @brief Own lightness-only observations,chroma targets and original RGB.
///
/// Tensor handles share owned storage. Preparing pairs does not fit a model.
///
/// @throws std::invalid_argument For invalid shape,dtype,rows,range or nonfinite color values.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto data = n::models::ColorPairs(n::zeros({1, 32, 32, 3}, n::DType::Int32), {0});
/// @endcode
/// @par References
/// - Scikit-image v0.25.2 colorconv.py, https://github.com/scikit-image/scikit-image/blob/v0.25.2/skimage/color/colorconv.py. sRGB matrices,transfer thresholds and D65/2degree white follow the source. This example performs Lab chroma regression,not Zhang classification.
/// @see ColorPairs(), ColorRGBToLabTensor(), ScoreTensorColor()
struct ColorData {
  nn::Tensor lightness; ///< Float32[N,H,W,1],L/50-1;model observation.
  nn::Tensor chroma;    ///< Float32[N,H,W,2],a/128,b/128;fitting target.
  nn::Tensor rgb;       ///< Float32[N,H,W,3],original RGB in[0,1].
};
/// @brief Prepare original-resolution grayscale/chroma pairs using tensor Lab conversion.
///
/// Selected bytes convert to Float64 before/255,color science and normalization;
/// outputs cast once to Float32,matching the source preparation precision.
/// No color labels enter the one-channel observation beyond its measured L.
///
/// @param images Int32[N,32,32,3] CIFAR RGB byte values0..255.
/// @param rows Nonempty bounded source indices,duplicates/reordering allowed.
/// @return Float32 lightness,chroma and RGB pairs in requested row order.
/// @throws std::invalid_argument For invalid shape,dtype,rows,range or nonfinite color values.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto data = n::models::ColorPairs(n::zeros({2, 32, 32, 3}, n::DType::Int32), {1, 0});
/// @endcode
/// @par References
/// - Scikit-image v0.25.2 colorconv.py, https://github.com/scikit-image/scikit-image/blob/v0.25.2/skimage/color/colorconv.py. sRGB matrices,transfer thresholds and D65/2degree white follow the source. This example performs Lab chroma regression,not Zhang classification.
/// @see ColorPairs(), ColorRGBToLabTensor(), ScoreTensorColor()
inline ColorData ColorPairs(const nn::Tensor &images, const std::vector<std::size_t> &rows) {
  if (images.dim() != 4 || images.size(0) < 1 || images.size(1) != 32 ||
      images.size(2) != 32 || images.size(3) != 3 || images.dtype() != nn::DType::Int32 ||
      rows.empty())
    throw std::invalid_argument("original32px Int32 RGB images required");
  std::vector<std::int64_t> ids;
  for (auto r : rows) {
    if (r >= static_cast<std::size_t>(images.size(0)))
      throw std::invalid_argument("color row bounds");
    ids.push_back(static_cast<std::int64_t>(r));
  }
  const auto rgb = nn::take_rows(images, ids).to(nn::DType::Float64) / 255.;
  const auto lab = ColorRGBToLabTensor(rgb);
  return {(nn::slice_axis(lab, 3, 0, 1) / 50. - 1.).to(nn::DType::Float32),
          (nn::slice_axis(lab, 3, 1, 3) / 128.).to(nn::DType::Float32),
          rgb.to(nn::DType::Float32)};
}
/// @brief Gather color-image rows with the input dtype and graph preserved.
///
/// No host pixel arithmetic or fitting.
///
/// @param x Nonempty-batch rank4 input.
/// @param rows Bounded row metadata,duplicates/empty output allowed.
/// @return Contiguous[rows.size(),H,W,C] tensor.
/// @throws std::invalid_argument For invalid shape,dtype,rows,range or nonfinite color values.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto x = n::models::ColorBatch(n::ones({2, 32, 32, 1}), {1, 0, 1});
/// @endcode
/// @par References
/// - Scikit-image v0.25.2 colorconv.py, https://github.com/scikit-image/scikit-image/blob/v0.25.2/skimage/color/colorconv.py. sRGB matrices,transfer thresholds and D65/2degree white follow the source. This example performs Lab chroma regression,not Zhang classification.
/// @see ColorPairs(), ColorRGBToLabTensor(), ScoreTensorColor()
inline nn::Tensor ColorBatch(const nn::Tensor &x, const std::vector<std::size_t> &rows) {
  if (x.dim() != 4 || x.size(0) < 1)
    throw std::invalid_argument("positive NHWC color batch required");
  std::vector<std::int64_t> ids;
  for (auto r : rows) {
    if (r >= static_cast<std::size_t>(x.size(0)))
      throw std::invalid_argument("color batch row bounds");
    ids.push_back(static_cast<std::int64_t>(r));
  }
  return nn::take_rows(x, ids).contiguous();
}
/// @brief Compute the fitting-only normalized mean chroma as an independent oracle.
///
/// Float64 tensor reduction;host array only presents two values. Reports should
/// fit this trainable baseline with an explicit optimizer and use this as an oracle.
///
/// @param fit Fitting chroma Float32[N,H,W,2],nonempty.
/// @return Double[2] means in normalized a/128,b/128 units.
/// @throws std::invalid_argument For invalid shape,dtype,rows,range or nonfinite color values.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const n::models::ColorData data{n::zeros({1, 1, 1, 1}), n::ones({1, 1, 1, 2}),
///                                 n::zeros({1, 1, 1, 3})};
/// const auto mean = n::models::ColorMeanChroma(data);
/// @endcode
/// @par References
/// - Scikit-image v0.25.2 colorconv.py, https://github.com/scikit-image/scikit-image/blob/v0.25.2/skimage/color/colorconv.py. sRGB matrices,transfer thresholds and D65/2degree white follow the source. This example performs Lab chroma regression,not Zhang classification.
/// @see ColorPairs(), ColorRGBToLabTensor(), ScoreTensorColor()
inline std::array<double, 2> ColorMeanChroma(const ColorData &fit) {
  if (fit.chroma.dim() != 4 || fit.chroma.shape().back() != 2 || fit.chroma.numel() < 1)
    throw std::invalid_argument("nonempty two-channel chroma required");
  const auto mean = nn::mean(fit.chroma.to(nn::DType::Float64), {0, 1, 2}).to_vector<double>();
  return {mean[0], mean[1]};
}
/// @brief Broadcast a supplied fixed chroma to every image pixel.
///
/// No fitting;zero chroma is a neutral-gray reference. Use fitting-only values
/// for a learned constant baseline and show its optimizer loop in the report.
///
/// @param data Supplies target[N,H,W,2] geometry only.
/// @param mean Finite normalized a/128,b/128 constants.
/// @return Float32 matching-shape constant prediction,no target values read.
/// @throws std::invalid_argument For invalid shape,dtype,rows,range or nonfinite color values.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const n::models::ColorData data{n::zeros({1, 1, 1, 1}), n::zeros({1, 1, 1, 2}),
///                                 n::zeros({1, 1, 1, 3})};
/// const auto gray = n::models::ColorConstantChroma(data, {0., 0.});
/// @endcode
/// @par References
/// - Scikit-image v0.25.2 colorconv.py, https://github.com/scikit-image/scikit-image/blob/v0.25.2/skimage/color/colorconv.py. sRGB matrices,transfer thresholds and D65/2degree white follow the source. This example performs Lab chroma regression,not Zhang classification.
/// @see ColorPairs(), ColorRGBToLabTensor(), ScoreTensorColor()
inline nn::Tensor ColorConstantChroma(const ColorData &data,
                                      const std::array<double, 2> &mean) {
  if (data.chroma.dim() != 4 || data.chroma.shape().back() != 2 || !std::isfinite(mean[0]) ||
      !std::isfinite(mean[1]))
    throw std::invalid_argument("finite two-channel constant chroma required");
  return nn::ones(data.chroma.shape()) *
         nn::from_vector(std::vector<double>{mean[0], mean[1]}, {2}).to(nn::DType::Float32);
}
/// @brief Own color-regression diagnostics and decoded RGB.
///
/// Numerical work uses Float64 tensors;RGB presentation casts once to Float32.
/// Metrics are detached and never update a model or optimizer.
///
/// @throws std::invalid_argument For invalid shape,dtype,rows,range or nonfinite color values.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto d = n::models::ColorPairs(n::zeros({1, 32, 32, 3}, n::DType::Int32), {0});
/// const auto score = n::models::ScoreTensorColor(d, d.chroma);
/// @endcode
/// @par References
/// - Scikit-image v0.25.2 colorconv.py, https://github.com/scikit-image/scikit-image/blob/v0.25.2/skimage/color/colorconv.py. sRGB matrices,transfer thresholds and D65/2degree white follow the source. This example performs Lab chroma regression,not Zhang classification.
/// @see ColorPairs(), ColorRGBToLabTensor(), ScoreTensorColor()
struct TensorColorMetrics {
  nn::Tensor ab_mse;    ///< Mean normalized a/b squared error,averaged over both channels.
  nn::Tensor delta_e76; ///< Mean Euclidean Lab error,preserved L cancels.
  nn::Tensor rgb_psnr;  ///< Mean per-image clipped-RGB PSNR,dB;perfect image is infinity.
  nn::Tensor clipped_pixels; ///< Fraction of pixels with any out-of-gamut raw decoded channel.
  nn::Tensor chroma;         ///< Mean physical predicted sqrt(a²+b²).
  nn::Tensor rgb;            ///< Float32[N,H,W,3] clipped decoded RGB for display.
};
/// @brief Own color-regression diagnostics and decoded RGB. Host presentation view.
///
/// Numerical work uses Float64 tensors;RGB presentation casts once to Float32.
/// Metrics are detached and never update a model or optimizer.
///
/// @throws std::invalid_argument For invalid shape,dtype,rows,range or nonfinite color values.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto d = n::models::ColorPairs(n::zeros({1, 32, 32, 3}, n::DType::Int32), {0});
/// const auto score = n::models::ColorScore(d, d.chroma);
/// @endcode
/// @par References
/// - Scikit-image v0.25.2 colorconv.py, https://github.com/scikit-image/scikit-image/blob/v0.25.2/skimage/color/colorconv.py. sRGB matrices,transfer thresholds and D65/2degree white follow the source. This example performs Lab chroma regression,not Zhang classification.
/// @see ColorPairs(), ColorRGBToLabTensor(), ScoreTensorColor()
struct ColorMetrics {
  double ab_mse = 0;    ///< Mean normalized a/b squared error,averaged over both channels.
  double delta_e76 = 0; ///< Mean Euclidean Lab error,preserved L cancels.
  double rgb_psnr = 0;  ///< Mean per-image clipped-RGB PSNR,dB;perfect image is infinity.
  double clipped_pixels = 0; ///< Fraction of pixels with any out-of-gamut raw decoded channel.
  double chroma = 0;         ///< Mean physical predicted sqrt(a²+b²).
  nn::Tensor rgb;            ///< Float32[N,H,W,3] clipped decoded RGB for display.
};
/// @brief Score chroma predictions while preserving measured lightness.
///
/// RGB decoding uses given L and predicted ab. Raw gamut fraction counts pixels,
/// not channels. Clip only decoded RGB for PSNR/display;chroma/DeltaE see original
/// predictions. Mean per-image PSNR differs from PSNR of pooled error.
///
/// @param data Aligned nonempty lightness/chroma/RGB data;H/W may be1.
/// @param prediction Finite matching[N,H,W,2] normalized a/128,b/128 estimates.
/// @return Detached Float64 scalar metrics and Float32 clipped RGB.
/// @throws std::invalid_argument For invalid shape,dtype,rows,range or nonfinite color values.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto d = n::models::ColorPairs(n::zeros({1, 32, 32, 3}, n::DType::Int32), {0});
/// const auto score = n::models::ScoreTensorColor(d, n::zeros_like(d.chroma));
/// @endcode
/// @par References
/// - Scikit-image v0.25.2 colorconv.py, https://github.com/scikit-image/scikit-image/blob/v0.25.2/skimage/color/colorconv.py. sRGB matrices,transfer thresholds and D65/2degree white follow the source. This example performs Lab chroma regression,not Zhang classification.
/// @see ColorPairs(), ColorRGBToLabTensor(), ScoreTensorColor()
inline TensorColorMetrics ScoreTensorColor(const ColorData &data,
                                           const nn::Tensor &prediction) {
  nn::NoGradGuard guard;
  if (prediction.dim() != 4 || prediction.size(0) < 1 || prediction.size(1) < 1 ||
      prediction.size(2) < 1 || prediction.size(3) != 2 ||
      prediction.shape() != data.chroma.shape() ||
      data.lightness.shape() !=
          nn::Shape{prediction.size(0), prediction.size(1), prediction.size(2), 1} ||
      data.rgb.shape() !=
          nn::Shape{prediction.size(0), prediction.size(1), prediction.size(2), 3})
    throw std::invalid_argument("aligned color prediction required");
  const auto p = prediction.to(nn::DType::Float64),
             target = data.chroma.to(nn::DType::Float64), diff = 128. * (p - target);
  const auto lab =
      nn::concat({50. * (data.lightness.to(nn::DType::Float64) + 1.), 128. * p}, 3);
  const auto raw = ColorLabToRGBTensor(lab), rgb = nn::clip(raw, 0., 1.);
  const auto mse = nn::mean(nn::square(rgb - data.rgb.to(nn::DType::Float64)), {1, 2, 3});
  const auto psnr = nn::where(
      nn::equal(mse, 0.),
      nn::full(mse.shape(), std::numeric_limits<double>::infinity(), nn::DType::Float64),
      -10. *
          nn::log(nn::maximum(mse, nn::full(mse.shape(), std::numeric_limits<double>::min(),
                                            nn::DType::Float64))) /
          std::log(10.));
  const auto outside = nn::any(nn::logical_or(nn::less(raw, 0.), nn::greater(raw, 1.)), {3});
  return {nn::mean(nn::square(p - target)),
          nn::mean(nn::sqrt(nn::sum(nn::square(diff), {3}))),
          nn::mean(psnr),
          nn::mean(outside.to(nn::DType::Float64)),
          nn::mean(nn::sqrt(nn::sum(nn::square(128. * p), {3}))),
          rgb.to(nn::DType::Float32)};
}
/// @brief Present canonical tensor color diagnostics as host scalars.
///
/// Compatibility egress only;all numerical work delegates to ScoreTensorColor.
///
/// @param data Aligned lightness/chroma/RGB data.
/// @param prediction Matching normalized chroma predictions.
/// @return Five presentation doubles plus decoded Float32 RGB.
/// @throws std::invalid_argument For invalid shape,dtype,rows,range or nonfinite color values.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto d = n::models::ColorPairs(n::zeros({1, 32, 32, 3}, n::DType::Int32), {0});
/// const auto score = n::models::ColorScore(d, d.chroma);
/// @endcode
/// @par References
/// - Scikit-image v0.25.2 colorconv.py, https://github.com/scikit-image/scikit-image/blob/v0.25.2/skimage/color/colorconv.py. sRGB matrices,transfer thresholds and D65/2degree white follow the source. This example performs Lab chroma regression,not Zhang classification.
/// @see ColorPairs(), ColorRGBToLabTensor(), ScoreTensorColor()
inline ColorMetrics ColorScore(const ColorData &data, const nn::Tensor &prediction) {
  const auto m = ScoreTensorColor(data, prediction);
  return {m.ab_mse.item(),         m.delta_e76.item(), m.rgb_psnr.item(),
          m.clipped_pixels.item(), m.chroma.item(),    m.rgb};
}
/// @brief Render grayscale,fit-mean,learned and target RGB tiles.
///
/// Float32 multiplication by255 then Float64 half-byte rounding preserves
/// source lround(255*Float32). Use save_image_grid for layout/scaling;no fitting.
///
/// @param data Original RGB tensor supplies target column.
/// @param learned Matching decoded learned RGB.
/// @param gray Matching neutral-chroma decoded RGB.
/// @param mean Matching fitting-only constant-chroma decoded RGB.
/// @return Float32[N,H,4W,3] clipped byte-quantized tiles in[0,1].
/// @throws std::invalid_argument For invalid shape,dtype,rows,range or nonfinite color values.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto d = n::models::ColorPairs(n::zeros({1, 32, 32, 3}, n::DType::Int32), {0});
/// const auto gray = n::models::ColorScore(d, n::zeros_like(d.chroma));
/// const auto tiles = n::models::ColorPreviewTiles(d, gray.rgb, gray.rgb, gray.rgb);
/// @endcode
/// @par References
/// - Scikit-image v0.25.2 colorconv.py, https://github.com/scikit-image/scikit-image/blob/v0.25.2/skimage/color/colorconv.py. sRGB matrices,transfer thresholds and D65/2degree white follow the source. This example performs Lab chroma regression,not Zhang classification.
/// @see ColorPairs(), ColorRGBToLabTensor(), ScoreTensorColor()
inline nn::Tensor ColorPreviewTiles(const ColorData &data, const nn::Tensor &learned,
                                    const nn::Tensor &gray, const nn::Tensor &mean) {
  if (learned.shape() != data.rgb.shape() || gray.shape() != data.rgb.shape() ||
      mean.shape() != data.rgb.shape() || data.rgb.dim() != 4 || data.rgb.size(0) < 1 ||
      data.rgb.size(3) != 3)
    throw std::invalid_argument("aligned color preview RGB required");
  const auto columns = nn::concat({gray, mean, learned, data.rgb}, 2);
  const auto bytes = nn::clip(columns, 0., 1.) * 255.;
  return (nn::floor(bytes.to(nn::DType::Float64) + .5) / 255.).to(nn::DType::Float32);
}
} // namespace datamunge::nn::models
