#pragma once
/// @file partial_inpainting_losses.hpp
/// @brief Full tensor pixel/perceptual/Gram/dilated-hole TV objective with genuine frozen VGG16 units.
#include "datamunge/nn/johnson_models.hpp"
#include "datamunge/nn/partial_unet_models.hpp"
namespace datamunge::nn::models {
namespace nn = datamunge::nn;
/// @brief Own three frozen VGG pool feature handles.
///
/// Order pool1/2/3. Handles retain generated-image graphs or detached truth targets.
///
/// @throws std::invalid_argument For incompatible tensor geometry,dtype or invalid imported feature inventory.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// m::PartialFeatures f{n::ones({1, 4, 4, 2}), n::ones({1, 2, 2, 3}), n::ones({1, 1, 1, 4})};
/// @endcode
/// @par References
/// - Liu et al.(2018),Image Inpainting for Irregular Holes Using Partial Convolutions, https://arxiv.org/abs/1804.07723;official implementation https://github.com/NVIDIA/partialconv. Source feature-unit adaptation uses genuine Keras centered BGR255 VGG16 pool1/2/3 divided by255,not a claim of Torch feature parity. Coefficients valid+6hole+.05perceptual+120(style_raw+style_composite)+.1TV;pixel losses divide by full image size.
/// @see PartialObjective(), PartialVgg(), PartialUNet()
using PartialFeatures = std::array<nn::Tensor, 3>;

/// @brief Extract genuine frozen VGG16 pool1/2/3 with explicit source feature units.
///
/// Extract last ReLU of first three blocks with max pools between blocks,
/// then apply each block’s max pool. Divide all activations by255 explicitly.
/// No fitting,optimizer or pretrained Torch compatibility promise. See
/// FrozenVgg16 for complete genuine-weight construction;missing inventory throws.
///
/// @param vgg Imported frozen VGG16 convolution extractor.
/// @param image Float32 encoded RGB[N,H,W,3],H/W>=8.
/// @return Three generated-image-differentiable pooled feature tensors divided by255.
/// @throws std::invalid_argument For incompatible tensor geometry,dtype or invalid imported feature inventory.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// m::TensorMap weights;
/// const int depths[] = {2, 2, 3, 3, 3}, widths[] = {64, 128, 256, 512, 512};
/// int input = 3;
/// // Zero weights are a shape-only documentation fixture,not pretrained features.
/// for (int b = 0; b < 5; ++b)
///   for (int l = 0; l < depths[b]; ++l) {
///     const auto name = "block" + std::to_string(b + 1) + "_conv" + std::to_string(l + 1);
///     weights.emplace(name + "_W", n::zeros({3, 3, input, widths[b]}));
///     weights.emplace(name + "_b", n::zeros({widths[b]}));
///     input = widths[b];
///   }
/// m::FrozenVgg16 vgg(weights);
/// const auto features=vgg.Extract(n::ones({1,16,16,3}),3);
/// const auto pooled=m::PartialVgg(vgg,n::ones({1,16,16,3}));
/// @endcode
/// @par References
/// - Liu et al.(2018),Image Inpainting for Irregular Holes Using Partial Convolutions, https://arxiv.org/abs/1804.07723;official implementation https://github.com/NVIDIA/partialconv. Source feature-unit adaptation uses genuine Keras centered BGR255 VGG16 pool1/2/3 divided by255,not a claim of Torch feature parity. Coefficients valid+6hole+.05perceptual+120(style_raw+style_composite)+.1TV;pixel losses divide by full image size.
/// @see PartialObjective(), PartialVgg(), PartialUNet()
inline PartialFeatures PartialVgg(const FrozenVgg16 &vgg, const nn::Tensor &image) {
  const auto features = vgg.Extract(image, 3);
  PartialFeatures pooled;
  for (std::size_t i = 0; i < 3; ++i)
    pooled[i] = nn::max_pool2d(features.style[i], 2, 2, nn::Padding::Valid) / 255.0;
  return pooled;
}

/// @brief Compute per-image channel/spatial-normalized feature Grams.
///
/// No centering,batch mixing or fitted statistics.
///
/// @param value Floating nonempty NHWC feature tensor.
/// @return Graph-preserving[N,C,C] F^T F/(H*W*C).
/// @throws std::invalid_argument For incompatible tensor geometry,dtype or invalid imported feature inventory.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto g = m::PartialGram(n::ones({2, 3, 4, 5}));
/// @endcode
/// @par References
/// - Liu et al.(2018),Image Inpainting for Irregular Holes Using Partial Convolutions, https://arxiv.org/abs/1804.07723;official implementation https://github.com/NVIDIA/partialconv. Source feature-unit adaptation uses genuine Keras centered BGR255 VGG16 pool1/2/3 divided by255,not a claim of Torch feature parity. Coefficients valid+6hole+.05perceptual+120(style_raw+style_composite)+.1TV;pixel losses divide by full image size.
/// @see PartialObjective(), PartialVgg(), PartialUNet()
inline nn::Tensor PartialGram(const nn::Tensor &value) {
  const nn::Tensor flat = nn::reshape(value, {value.size(0), -1, value.size(3)});
  return nn::matmul(flat, flat, true) / double(value.size(1) * value.size(2) * value.size(3));
}

/// @brief Compute mean absolute reconstruction error over all elements.
///
/// Graph retained;zero subgradient at exact equality follows tensor abs semantics.
///
/// @param a Generated tensor.
/// @param b Compatible fixed reference,same shape recommended.
/// @return Scalar mean(abs(a-b)).
/// @throws std::invalid_argument For incompatible tensor geometry,dtype or invalid imported feature inventory.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto l = m::PartialL1(n::ones({2, 3}), n::zeros({2, 3}));
/// @endcode
/// @par References
/// - Liu et al.(2018),Image Inpainting for Irregular Holes Using Partial Convolutions, https://arxiv.org/abs/1804.07723;official implementation https://github.com/NVIDIA/partialconv. Source feature-unit adaptation uses genuine Keras centered BGR255 VGG16 pool1/2/3 divided by255,not a claim of Torch feature parity. Coefficients valid+6hole+.05perceptual+120(style_raw+style_composite)+.1TV;pixel losses divide by full image size.
/// @see PartialObjective(), PartialVgg(), PartialUNet()
inline nn::Tensor PartialL1(const nn::Tensor &a, const nn::Tensor &b) {
  return nn::mean(nn::abs(a - b));
}

/// @brief Sum raw/composite perceptual and Gram L1 discrepancies over three pools.
///
/// Perceptual sums L1(raw,truth)+L1(comp,truth);style compares channel/spatial
/// Grams with L1. Generated branches retain gradients;truth must be cached
/// without gradients by the caller. No coefficients or optimizer hidden.
///
/// @param raw Three generated raw-output features.
/// @param composite Three observed-preserving composite features.
/// @param truth Three detached ground-truth features.
/// @return Array{perceptual,style_raw,style_composite}.
/// @throws std::invalid_argument For incompatible tensor geometry,dtype or invalid imported feature inventory.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// m::PartialFeatures f{n::ones({1, 4, 4, 2}), n::ones({1, 2, 2, 3}), n::ones({1, 1, 1, 4})};
/// const auto l = m::PartialFeatureLosses(f, f, f);
/// @endcode
/// @par References
/// - Liu et al.(2018),Image Inpainting for Irregular Holes Using Partial Convolutions, https://arxiv.org/abs/1804.07723;official implementation https://github.com/NVIDIA/partialconv. Source feature-unit adaptation uses genuine Keras centered BGR255 VGG16 pool1/2/3 divided by255,not a claim of Torch feature parity. Coefficients valid+6hole+.05perceptual+120(style_raw+style_composite)+.1TV;pixel losses divide by full image size.
/// @see PartialObjective(), PartialVgg(), PartialUNet()
inline std::array<nn::Tensor, 3> PartialFeatureLosses(const PartialFeatures &raw,
                                                      const PartialFeatures &composite,
                                                      const PartialFeatures &truth) {
  nn::Tensor perceptual, style_raw, style_composite;
  for (std::size_t i = 0; i < 3; ++i) {
    const nn::Tensor target = PartialGram(truth[i]);
    const nn::Tensor p = PartialL1(raw[i], truth[i]) + PartialL1(composite[i], truth[i]),
                     sr = PartialL1(PartialGram(raw[i]), target),
                     sc = PartialL1(PartialGram(composite[i]), target);
    perceptual = i ? perceptual + p : p;
    style_raw = i ? style_raw + sr : sr;
    style_composite = i ? style_composite + sc : sc;
  }
  return {perceptual, style_raw, style_composite};
}

/// @brief Measure hole-region L1 neighbor variation with a full-image denominator.
///
/// One-pixel max-pool dilation of holes defines the region. Each neighbor pair
/// counts only if BOTH endpoints lie in that dilated region. Denominator is
/// full RGB image size,not eligible-neighbor count. This is L1 TV,not Johnson TV.
///
/// @param composite Floating[N,H,W,C] observed-preserving RGB.
/// @param valid Matching floating binary validity,one value per channel.
/// @return Scalar masked horizontal+vertical absolute differences/numel(composite).
/// @throws std::invalid_argument For incompatible tensor geometry,dtype or invalid imported feature inventory.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto x = n::ones({1, 4, 5, 3});
/// const auto tv = m::PartialTV(x, n::ones_like(x));
/// @endcode
/// @par References
/// - Liu et al.(2018),Image Inpainting for Irregular Holes Using Partial Convolutions, https://arxiv.org/abs/1804.07723;official implementation https://github.com/NVIDIA/partialconv. Source feature-unit adaptation uses genuine Keras centered BGR255 VGG16 pool1/2/3 divided by255,not a claim of Torch feature parity. Coefficients valid+6hole+.05perceptual+120(style_raw+style_composite)+.1TV;pixel losses divide by full image size.
/// @see PartialObjective(), PartialVgg(), PartialUNet()
inline nn::Tensor PartialTV(const nn::Tensor &composite, const nn::Tensor &valid) {
  const nn::Tensor region = nn::max_pool2d(1.0 - valid, 3, 1, nn::Padding::Same);
  const std::int64_t b = composite.size(0), h = composite.size(1), w = composite.size(2),
                     c = composite.size(3);
  const nn::Shape hs{b, h, w - 1, c}, vs{b, h - 1, w, c};
  const nn::Tensor left = nn::slice(composite, {0, 0, 0, 0}, hs),
                   right = nn::slice(composite, {0, 0, 1, 0}, hs),
                   top = nn::slice(composite, {0, 0, 0, 0}, vs),
                   bottom = nn::slice(composite, {0, 1, 0, 0}, vs);
  const nn::Tensor horizontal =
      nn::slice(region, {0, 0, 0, 0}, hs) * nn::slice(region, {0, 0, 1, 0}, hs);
  const nn::Tensor vertical =
      nn::slice(region, {0, 0, 0, 0}, vs) * nn::slice(region, {0, 1, 0, 0}, vs);
  const nn::Tensor hsum = nn::sum(horizontal * nn::abs(left - right)),
                   vsum = nn::sum(vertical * nn::abs(top - bottom));
  return (hsum + vsum) / double(composite.numel());
}

/// @brief Compute every component of the complete source partial-inpainting objective.
///
/// Total=valid+6hole+.05perceptual+120(style_raw+style_composite)+.1TV.
/// Pixel valid/hole L1 components use full-image means,so larger holes carry
/// more pixel loss. TV is evaluated on the observed-preserving composite.
/// Caller explicitly owns the tensor optimizer and training loops.
///
/// @param raw Unclipped generated RGB matching truth.
/// @param truth Original RGB targets.
/// @param valid Matching binary observed validity.
/// @param rf Raw generated VGG pool features.
/// @param cf Composite generated VGG pool features.
/// @param target_features Detached original-truth pool features.
/// @return Array{total,valid,hole,perceptual,style_raw,style_composite,TV}.
/// @throws std::invalid_argument For incompatible tensor geometry,dtype or invalid imported feature inventory.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto x = n::ones({1, 4, 4, 3});
/// m::PartialFeatures f{n::ones({1, 4, 4, 2}), n::ones({1, 2, 2, 3}), n::ones({1, 1, 1, 4})};
/// const auto l = m::PartialObjective(x, x, n::ones_like(x), f, f, f);
/// @endcode
/// @par References
/// - Liu et al.(2018),Image Inpainting for Irregular Holes Using Partial Convolutions, https://arxiv.org/abs/1804.07723;official implementation https://github.com/NVIDIA/partialconv. Source feature-unit adaptation uses genuine Keras centered BGR255 VGG16 pool1/2/3 divided by255,not a claim of Torch feature parity. Coefficients valid+6hole+.05perceptual+120(style_raw+style_composite)+.1TV;pixel losses divide by full image size.
/// @see PartialObjective(), PartialVgg(), PartialUNet()
inline std::array<nn::Tensor, 7>
PartialObjective(const nn::Tensor &raw, const nn::Tensor &truth, const nn::Tensor &valid,
                 const PartialFeatures &rf, const PartialFeatures &cf,
                 const PartialFeatures &target_features) {
  const nn::Tensor error = nn::abs(raw - truth);
  const nn::Tensor observed = nn::mean(valid * error), hole = nn::mean((1.0 - valid) * error);
  const auto features = PartialFeatureLosses(rf, cf, target_features);
  const nn::Tensor tv = PartialTV(PartialComposite(truth, raw, valid), valid);
  const nn::Tensor total = observed + 6.0 * hole + .05 * features[0] +
                           120.0 * (features[1] + features[2]) + .1 * tv;
  return {total, observed, hole, features[0], features[1], features[2], tv};
}

} // namespace datamunge::nn::models
