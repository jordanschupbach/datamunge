#pragma once
/// @file partial_unet_models.hpp
/// @brief Complete mask-propagating partial-convolution U-Net with explicit BN policy.
#include "datamunge/nn/image_ops.hpp"
#include "datamunge/nn/inpainting_models.hpp"
namespace datamunge::nn::models {
namespace nn = datamunge::nn;
/// @brief Describe partial-convolution encoder/decoder depth and channel widths.
///
/// Full8x64 width progression64/128/256/512/512/512/512/512.
/// Compact5x8 retains encoder/decoder mechanisms with8/16/32/64/64 channels.
///
/// @throws std::invalid_argument For invalid configuration,shapes,dtype or incompatible stored parameters.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto c = m::PartialConfig::Compact();
/// @endcode
/// @par References
/// - Liu et al.(2018),Image Inpainting for Irregular Holes Using Partial Convolutions, https://arxiv.org/abs/1804.07723;official reference https://github.com/NVIDIA/partialconv. Source adaptations:NHWC,TensorFlow SAME stride phase,compact5-stage32px training and Keras VGG feature units. No paper/pretrained Torch quality claim.
/// @see PartialUNet(), PartialConv(), PartialComposite()
struct PartialConfig {
  int depth = 8;  ///< Encoder/decoder depth,3..8.
  int width = 64; ///< Positive initial channel count,doubling up to8*width.

  /// @brief Return the five-level width8 teaching configuration.
  ///
  /// No weights or optimizer are allocated.
  ///
  /// @return Depth5,width8 metadata.
  /// @throws std::invalid_argument For invalid configuration,shapes,dtype or incompatible stored parameters.
  /// @par Example
  /// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto c = m::PartialConfig::Compact();
/// @endcode
  /// @par References
  /// - Liu et al.(2018),Image Inpainting for Irregular Holes Using Partial Convolutions, https://arxiv.org/abs/1804.07723;official reference https://github.com/NVIDIA/partialconv. Source adaptations:NHWC,TensorFlow SAME stride phase,compact5-stage32px training and Keras VGG feature units. No paper/pretrained Torch quality claim.
  /// @see PartialUNet(), PartialConv(), PartialComposite()
  static PartialConfig Compact() {
    return {5, 8};
  }
  /// @brief Validate configuration and return each encoder channel width.
  ///
  /// Depth3..8 and positive width required. Integer architecture metadata only.
  ///
  /// @return depth integer widths,width*2^min(level,3).
  /// @throws std::invalid_argument For invalid configuration,shapes,dtype or incompatible stored parameters.
  /// @par Example
  /// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto widths = m::PartialConfig::Compact().Widths();
/// @endcode
  /// @par References
  /// - Liu et al.(2018),Image Inpainting for Irregular Holes Using Partial Convolutions, https://arxiv.org/abs/1804.07723;official reference https://github.com/NVIDIA/partialconv. Source adaptations:NHWC,TensorFlow SAME stride phase,compact5-stage32px training and Keras VGG feature units. No paper/pretrained Torch quality claim.
  /// @see PartialUNet(), PartialConv(), PartialComposite()
  [[nodiscard]] std::vector<int> Widths() const {
    if (depth < 3 || depth > 8 || width < 1)
      throw std::invalid_argument("partial UNet depth3..8 and positive width required");
    std::vector<int> result;
    for (int i = 0; i < depth; ++i)
      result.push_back(width * (1 << std::min(i, 3)));
    return result;
  }
};

/// @brief Own raw prediction,propagated mask and all feature/mask stages.
///
/// Tensor handles retain graph/storage;output RGB is not clipped or bounded.
///
/// @throws std::invalid_argument For invalid configuration,shapes,dtype or incompatible stored parameters.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// n::ParameterStore p(83), state(84);
/// const auto out = m::PartialUNet(p, state, n::zeros({1, 32, 32, 3}), n::ones({1, 32, 32, 3}),
///                                 m::PartialConfig::Compact());
/// @endcode
/// @par References
/// - Liu et al.(2018),Image Inpainting for Irregular Holes Using Partial Convolutions, https://arxiv.org/abs/1804.07723;official reference https://github.com/NVIDIA/partialconv. Source adaptations:NHWC,TensorFlow SAME stride phase,compact5-stage32px training and Keras VGG feature units. No paper/pretrained Torch quality claim.
/// @see PartialUNet(), PartialConv(), PartialComposite()
struct PartialUNetResult {
  nn::Tensor image;               ///< Raw unconstrained predicted RGB,NHWC.
  nn::Tensor mask;                ///< Final propagated multichannel binary support,same shape.
  std::vector<nn::Tensor> stages; ///< Encoder then decoder feature activations.
  std::vector<nn::Tensor> masks;  ///< Matching per-stage multichannel binary masks.
};

/// @brief Resize features or masks to a skip activation’s spatial geometry.
///
/// No value interpolation. Observed skip geometry supports odd rectangular inputs.
///
/// @param input NHWC features or multichannel mask.
/// @param like NHWC tensor supplying requested height/width.
/// @return Legacy nearest-neighbor resized NHWC tensor.
/// @throws std::invalid_argument For invalid configuration,shapes,dtype or incompatible stored parameters.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto y = m::PartialResize(n::ones({1, 2, 3, 4}), n::ones({1, 5, 7, 2}));
/// @endcode
/// @par References
/// - Liu et al.(2018),Image Inpainting for Irregular Holes Using Partial Convolutions, https://arxiv.org/abs/1804.07723;official reference https://github.com/NVIDIA/partialconv. Source adaptations:NHWC,TensorFlow SAME stride phase,compact5-stage32px training and Keras VGG feature units. No paper/pretrained Torch quality claim.
/// @see PartialUNet(), PartialConv(), PartialComposite()
inline nn::Tensor PartialResize(const nn::Tensor &input, const nn::Tensor &like) {
  return nn::resize_nearest(input, {like.size(1), like.size(2)});
}

/// @brief Select the private partial-layer activation policy.
///
/// Architecture metadata;final RGB remains linear.
///
/// @throws std::invalid_argument For invalid configuration,shapes,dtype or incompatible stored parameters.

/// @par References
/// - Liu et al.(2018),Image Inpainting for Irregular Holes Using Partial Convolutions, https://arxiv.org/abs/1804.07723;official reference https://github.com/NVIDIA/partialconv. Source adaptations:NHWC,TensorFlow SAME stride phase,compact5-stage32px training and Keras VGG feature units. No paper/pretrained Torch quality claim.
/// @see PartialUNet(), PartialConv(), PartialComposite()
enum class PartialAct {
  kNone,     ///< Linear RGB output.
  kRelu,     ///< ReLU encoder features.
  kLeakyRelu ///< LeakyReLU.2 decoder features.
};

/// @brief Predict raw RGB with a complete mask-propagating partial-convolution U-Net.
///
/// Every layer multiplies input by detached mask,normalizes by valid kernel
/// count and updates validity. Encoder stride2 kernels7/5/5/3... use ReLU;
/// decoder nearest-upsamples BOTH features/masks,concatenates BOTH skips,
/// then partial3x3. Final skip is original observed RGB/mask. BN except first
/// encoder/final decoder;LeakyReLU.2 except linear final RGB. Full32865236
/// learned/10880 BN values;compact183684/592. No truth,VGG or optimizer
/// enters inference. Finite hole fill cannot affect output;NaN fill is outside
/// contract because masked convolution uses multiplication.
///
/// @param p Mutable learned store;warm every leaf before optimizer construction.
/// @param state Separate mutable moving BN mean/variance store.
/// @param image Floating finite[N,H,W,3] observed values,holes may contain arbitrary finite fill.
/// @param valid Matching-dtype multichannel binary validity0/1,NHWC.
/// @param config Default full8x64;Compact is5x8.
/// @param training Use BN batch moments if true,default false.
/// @param update Mutate moving BN iff training&&update,default true.
/// @return Raw RGB,final propagated mask and2*depth stage/mask handles.
/// @throws std::invalid_argument For invalid configuration,shapes,dtype or incompatible stored parameters.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// n::ParameterStore p(83), state(84);
/// const auto out = m::PartialUNet(p, state, n::zeros({1, 32, 32, 3}), n::ones({1, 32, 32, 3}),
///                                 m::PartialConfig::Compact());
/// @endcode
/// @par References
/// - Liu et al.(2018),Image Inpainting for Irregular Holes Using Partial Convolutions, https://arxiv.org/abs/1804.07723;official reference https://github.com/NVIDIA/partialconv. Source adaptations:NHWC,TensorFlow SAME stride phase,compact5-stage32px training and Keras VGG feature units. No paper/pretrained Torch quality claim.
/// @see PartialUNet(), PartialConv(), PartialComposite()
inline PartialUNetResult PartialUNet(nn::ParameterStore &p, nn::ParameterStore &state,
                                     const nn::Tensor &image, const nn::Tensor &valid,
                                     const PartialConfig &config = {}, bool training = false,
                                     bool update = true) {
  if (image.dim() != 4 || image.size(0) < 1 || image.size(1) < 1 || image.size(2) < 1 ||
      image.size(3) != 3 || valid.shape() != image.shape() || image.dtype() != p.dtype() ||
      valid.dtype() != image.dtype())
    throw std::invalid_argument("aligned floating RGB/multichannel validity required");
  const auto widths = config.Widths();
  PartialUNetResult result;
  std::vector<nn::Tensor> features{image}, masks{valid};
  std::int64_t channels = 3;
  /// @brief Create or reuse one masked convolution,explicit BN and activation.
  /// @param name Unique learned-layer prefix.
  /// @param x Input floating NHWC features.
  /// @param m Matching detached mask values.
  /// @param in Expected input channels.
  /// @param out Output channels.
  /// @param kernel Square convolution width.
  /// @param stride Spatial stride.
  /// @param normalize Apply BN when true.
  /// @param activation Linear,ReLU or LeakyReLU.2.
  /// @return Partial-convolution values,mask and counts;append stage handles.
  auto layer = [&](const std::string &name, const nn::Tensor &x, const nn::Tensor &m,
                   std::int64_t in, std::int64_t out, std::int64_t kernel, std::int64_t stride,
                   bool normalize, PartialAct activation) {
    const nn::Tensor weights =
        p.get(name + "/W", {kernel, kernel, in, out}, nn::init::he_normal());
    const nn::Tensor bias = p.get(name + "/b", {out}, nn::init::zeros());
    auto value = PartialConv(x, m, weights, bias, in, out, kernel, stride);
    if (normalize)
      value.value = InpaintNorm(p, state, name + "/bn", value.value, training, update);
    if (activation == PartialAct::kRelu)
      value.value = nn::relu(value.value);
    if (activation == PartialAct::kLeakyRelu)
      value.value = nn::leaky_relu(value.value, .2);
    result.stages.push_back(value.value);
    result.masks.push_back(value.mask);
    return value;
  };
  for (int i = 0; i < config.depth; ++i) {
    const int kernel = i == 0 ? 7 : i <= 2 ? 5 : 3;
    const auto value =
        layer("encode" + std::to_string(i + 1), features.back(), masks.back(), channels,
              widths[std::size_t(i)], kernel, 2, i > 0, PartialAct::kRelu);
    features.push_back(value.value);
    masks.push_back(value.mask);
    channels = widths[std::size_t(i)];
  }
  nn::Tensor x = features.back(), mask = masks.back();
  for (int i = config.depth - 1; i >= 0; --i) {
    const std::int64_t skip = i == 0 ? 3 : widths[std::size_t(i - 1)], out = i == 0 ? 3 : skip;
    const nn::Tensor combined =
        nn::concat({PartialResize(x, features[std::size_t(i)]), features[std::size_t(i)]}, 3);
    const nn::Tensor combined_mask =
        nn::concat({PartialResize(mask, masks[std::size_t(i)]), masks[std::size_t(i)]}, 3);
    const auto value = layer("decode" + std::to_string(config.depth - i), combined,
                             combined_mask, channels + skip, out, 3, 1, i > 0,
                             i > 0 ? PartialAct::kLeakyRelu : PartialAct::kNone);
    x = value.value, mask = value.mask, channels = out;
  }
  result.image = x;
  result.mask = mask;
  return result;
}

/// @brief Preserve observed values and use predictions only in holes.
///
/// Converts threshold comparisons to Double so Bool/integer masks obey.5.
/// No clipping,GT lookup or fitting. Caller must supply original observed input.
///
/// @param observed Observed tensor matching prediction geometry.
/// @param output Raw prediction,same shape as observed.
/// @param mask Same-shaped numeric validity,known when>.5.
/// @return Graph-preserving where(mask>.5,observed,output).
/// @throws std::invalid_argument For invalid configuration,shapes,dtype or incompatible stored parameters.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto y = m::PartialComposite(n::ones({1, 2, 3, 3}), n::zeros({1, 2, 3, 3}),
///                                    n::ones({1, 2, 3, 3}, n::DType::Bool));
/// @endcode
/// @par References
/// - Liu et al.(2018),Image Inpainting for Irregular Holes Using Partial Convolutions, https://arxiv.org/abs/1804.07723;official reference https://github.com/NVIDIA/partialconv. Source adaptations:NHWC,TensorFlow SAME stride phase,compact5-stage32px training and Keras VGG feature units. No paper/pretrained Torch quality claim.
/// @see PartialUNet(), PartialConv(), PartialComposite()
inline nn::Tensor PartialComposite(const nn::Tensor &observed, const nn::Tensor &output,
                                   const nn::Tensor &mask) {
  if (observed.shape() != output.shape() || observed.shape() != mask.shape())
    throw std::invalid_argument("aligned observed/output/composite mask required");
  return nn::where(nn::greater(mask.to(nn::DType::Float64), .5), observed, output);
}

} // namespace datamunge::nn::models
