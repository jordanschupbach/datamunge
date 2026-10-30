#pragma once
/// @file colorization_models.hpp
/// @brief Lightness-only tensor colorizer with spatial skips and global context.
#include "datamunge/nn/colorization_data.hpp"
#include "datamunge/nn/image_ops.hpp"
#include "datamunge/nn/layers.hpp"
namespace datamunge::nn::models {
namespace nn = datamunge::nn;
/// @brief Resize context maps to a reference spatial shape.
///
/// align_corners=false,half_pixel_centers=true. Batch/channel dimensions preserved;
/// 1x1 context broadcasts without mixing images. No learned parameters.
///
/// @param input Floating[N,H,W,C] source.
/// @param like Rank4 reference supplies positive target height/width only.
/// @return Half-pixel bilinear resized input,graph retained.
/// @throws std::invalid_argument For invalid geometry,dtype or incompatible stored parameters.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto x = n::models::ColorResize(n::ones({2, 1, 1, 3}), n::zeros({2, 3, 5, 1}));
/// @endcode
/// @par References
/// - Scikit-image v0.25.2 colorconv.py, https://github.com/scikit-image/scikit-image/blob/v0.25.2/skimage/color/colorconv.py. sRGB matrices,transfer thresholds and D65/2degree white follow the source. This example performs Lab chroma regression,not Zhang classification.
/// @see Colorization(), ColorizationLoss()
inline nn::Tensor ColorResize(const nn::Tensor &input, const nn::Tensor &like) {
  return nn::resize_bilinear(input, {like.size(1), like.size(2)}, false, true);
}

/// @brief Own predicted chroma and all colorizer stages.
///
/// Tensor handles preserve the graph;ten stages expose complete connectivity.
///
/// @throws std::invalid_argument For invalid geometry,dtype or incompatible stored parameters.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(31);
/// const auto result = n::models::Colorization(p, n::zeros({1, 32, 32, 1}));
/// @endcode
/// @par References
/// - Scikit-image v0.25.2 colorconv.py, https://github.com/scikit-image/scikit-image/blob/v0.25.2/skimage/color/colorconv.py. sRGB matrices,transfer thresholds and D65/2degree white follow the source. This example performs Lab chroma regression,not Zhang classification.
/// @see Colorization(), ColorizationLoss()
struct ColorizationResult {
  nn::Tensor chroma;              ///< Float32[N,H,W,2] tanh normalized a/b prediction.
  std::vector<nn::Tensor> stages; ///< All ten encoder/context/decoder outputs in order.
};

/// @brief Predict normalized Lab chroma from one lightness channel.
///
/// Two stride2 encoders,three low-resolution context convolutions including
/// fusion,global average branch and bilinear decoder with spatial skips.
/// 140082 learned scalars/20 tensors;no BN or target/color inputs. Parameter
/// creation only,no implicit fitting. Odd rectangular shapes retain exact input size.
///
/// @param p Mutable learned store,warm all20 leaves before optimizer construction.
/// @param lightness Floating[N,H,W,1] L/50-1,positive dimensions,matching p dtype.
/// @return Tanh normalized a/128,b/128[N,H,W,2] and ten stages.
/// @throws std::invalid_argument For invalid geometry,dtype or incompatible stored parameters.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(31);
/// const auto result = n::models::Colorization(p, n::zeros({2, 33, 35, 1}));
/// @endcode
/// @par References
/// - Scikit-image v0.25.2 colorconv.py, https://github.com/scikit-image/scikit-image/blob/v0.25.2/skimage/color/colorconv.py. sRGB matrices,transfer thresholds and D65/2degree white follow the source. This example performs Lab chroma regression,not Zhang classification.
/// @see Colorization(), ColorizationLoss()
inline ColorizationResult Colorization(nn::ParameterStore &p, const nn::Tensor &lightness) {
  if (lightness.dim() != 4 || lightness.size(0) < 1 || lightness.size(1) < 1 ||
      lightness.size(2) < 1 || lightness.size(3) != 1 || lightness.dtype() != p.dtype())
    throw std::invalid_argument("colorizer requires positive lightness-only NHWC input");
  ColorizationResult result;
  /// @brief Create one named ReLU convolution and record its activation.
  /// @param name Parameter prefix.
  /// @param x Floating NHWC input.
  /// @param out Positive output width.
  /// @param stride Spatial stride,default1.
  /// @param kernel Square kernel,default3.
  /// @return ReLU activation recorded in result.stages.
  auto conv = [&](const std::string &name, const nn::Tensor &x, std::int64_t out,
                  std::int64_t stride = 1, std::int64_t kernel = 3) {
    nn::ConvLayerOptions o;
    o.activation = nn::Activation::Relu;
    o.stride = stride;
    auto y = nn::conv2d(p, name, x, out, kernel, o);
    result.stages.push_back(y);
    return y;
  };
  auto shallow = conv("encode1", lightness, 16);
  auto middle = conv("encode2", shallow, 32, 2);
  auto deep = conv("encode3", middle, 64, 2);
  auto context = conv("bottleneck1", deep, 64);
  context = conv("bottleneck2", context, 64);
  auto pooled = nn::mean(context, {1, 2}, true);
  auto global = conv("global", pooled, 32, 1, 1);
  auto fused =
      conv("fusion", nn::concat({context, ColorResize(global, context)}, 3), 64, 1, 1);
  auto decoded = conv("decode2", nn::concat({ColorResize(fused, middle), middle}, 3), 32);
  decoded = conv("decode1", nn::concat({ColorResize(decoded, shallow), shallow}, 3), 16);
  result.chroma = nn::tanh(nn::conv2d(p, "chroma", decoded, 2, 1));
  result.stages.push_back(result.chroma);
  return result;
}

/// @brief Compute mean normalized chroma regression error.
///
/// Gradient2*(prediction-target)/numel. Ambiguous colors converge toward
/// the conditional mean under squared error,which can desaturate predictions.
///
/// @param prediction Nonempty floating normalized a/128,b/128 predictions.
/// @param target Same-shaped normalized targets.
/// @return Scalar mean over every image/pixel/chroma channel.
/// @throws std::invalid_argument For invalid geometry,dtype or incompatible stored parameters.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto loss = n::models::ColorizationLoss(n::zeros({2, 1, 1, 2}), n::ones({2, 1, 1, 2}));
/// @endcode
/// @par References
/// - Scikit-image v0.25.2 colorconv.py, https://github.com/scikit-image/scikit-image/blob/v0.25.2/skimage/color/colorconv.py. sRGB matrices,transfer thresholds and D65/2degree white follow the source. This example performs Lab chroma regression,not Zhang classification.
/// @see Colorization(), ColorizationLoss()
inline nn::Tensor ColorizationLoss(const nn::Tensor &prediction, const nn::Tensor &target) {
  if (prediction.shape() != target.shape() || prediction.numel() < 1)
    throw std::invalid_argument("aligned nonempty chroma targets required");
  return nn::mean(nn::square(prediction - target));
}

} // namespace datamunge::nn::models
