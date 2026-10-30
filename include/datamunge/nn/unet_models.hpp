#pragma once
/// @file unet_models.hpp
/// @brief Valid/SAME U-Net with floor-centred cropped skips and learned transposed up-convolutions.
#include "datamunge/nn/autograd.hpp"
#include "datamunge/nn/ops.hpp"
#include "datamunge/nn/layers.hpp"
#include <stdexcept>
#include <string>
#include <vector>
namespace datamunge::nn::models {
namespace nn = datamunge::nn;
/// @brief Compute the square output side of the source U-Net geometry.
///
/// Returns388 for the original572 input. This architecture-shape calculation
/// uses integer metadata only,not tensor numerics or fitting.
///
/// @param side Positive input side.
/// @param levels Pooling levels1..6,default4.
/// @param valid True(default) subtracts4 for every convolution pair; false uses SAME.
/// @return Final positive output side; odd pooling uses floor.
/// @throws std::invalid_argument For invalid geometry, configuration, dtype or incompatible parameter shapes.
/// @par Example
/// @code
/// const auto side = dm::nn::models::UnetOutputSide(572);
/// @endcode
/// @par References
/// - Ronneberger, Fischer and Brox(2015), U-Net: Convolutional Networks for Biomedical Image Segmentation, https://arxiv.org/abs/1505.04597.
/// @see Unet()
inline int UnetOutputSide(int side, int levels = 4, bool valid = true) {
  if (levels < 1 || levels > 6 || side < 1)
    throw std::invalid_argument("invalid U-Net geometry");
  for (int level = 0; level < levels; ++level) {
    if (valid)
      side -= 4;
    if (side < 2)
      throw std::invalid_argument("U-Net input too small");
    side /= 2;
  }
  if (valid)
    side -= 4;
  for (int level = 0; level < levels; ++level) {
    side *= 2;
    if (valid)
      side -= 4;
    if (side < 1)
      throw std::invalid_argument("U-Net input too small");
  }
  return side;
}

/// @brief Crop a U-Net skip using floor-centred spatial offsets.
///
/// Odd excess pixels remain on bottom/right. Only source values receive
/// gradients; reference values are unused. No fitting or mutation.
///
/// @param source Positive rank4 NHWC tensor.
/// @param reference Rank4 tensor supplying target H/W, not values or gradients.
/// @return View of source with target spatial size and source batch/channels.
/// @throws std::invalid_argument For invalid geometry, configuration, dtype or incompatible parameter shapes.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto crop = n::models::UnetCenterCropTo(n::ones({1, 7, 9, 2}), n::zeros({1, 4, 6, 1}));
/// @endcode
/// @par References
/// - Ronneberger, Fischer and Brox(2015), U-Net: Convolutional Networks for Biomedical Image Segmentation, https://arxiv.org/abs/1505.04597.
/// @see Unet()
inline nn::Tensor UnetCenterCropTo(const nn::Tensor &source, const nn::Tensor &reference) {
  if (source.dim() != 4 || reference.dim() != 4 || reference.shape()[1] < 1 ||
      reference.shape()[2] < 1)
    throw std::invalid_argument("invalid U-Net crop rank/size");
  const auto &s = source.shape();
  const std::int64_t th = reference.shape()[1], tw = reference.shape()[2];
  if (th > s[1] || tw > s[2])
    throw std::invalid_argument("centre crop larger than source");
  return nn::slice(source, {0, (s[1] - th) / 2, (s[2] - tw) / 2, 0}, {s[0], th, tw, s[3]});
}

/// @brief Apply two named He-initialized3x3 ReLU convolutions.
///
/// Biases are learned; no BN, hidden fitting or external state.
///
/// @param p Borrowed learned store, creates/reuses conv1/conv2 leaves.
/// @param name Unique convolution-pair prefix.
/// @param x Positive same-dtype NHWC input.
/// @param out Positive output channels.
/// @param padding SAME(default) or VALID spatial geometry.
/// @return Second ReLU output,with gradients through both convolutions.
/// @throws std::invalid_argument For invalid geometry, configuration, dtype or incompatible parameter shapes.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(1);
/// const auto y = n::models::UnetConvPair(p, "pair", n::ones({1, 8, 8, 2}), 4);
/// @endcode
/// @par References
/// - Ronneberger, Fischer and Brox(2015), U-Net: Convolutional Networks for Biomedical Image Segmentation, https://arxiv.org/abs/1505.04597.
/// @see Unet()
inline nn::Tensor UnetConvPair(nn::ParameterStore &p, const std::string &name,
                               const nn::Tensor &x, std::int64_t out,
                               nn::Padding padding = nn::Padding::Same) {
  nn::ConvLayerOptions conv;
  conv.padding = padding;
  conv.activation = nn::Activation::Relu;
  conv.init = nn::init::he_uniform();
  return nn::conv2d(p, name + "/conv2", nn::conv2d(p, name + "/conv1", x, out, 3, conv), out,
                    3, conv);
}

/// @brief Double spatial size with a learned2x2 stride2 transposed convolution.
///
/// Kernel and bias are learned; this is not fixed bilinear interpolation.
/// No fitting or moving state.
///
/// @param p Learned same-dtype store.
/// @param name Unique weight/bias prefix.
/// @param x Positive NHWC input.
/// @param out Positive output channels.
/// @return He-initialized VALID transpose-convolution output after ReLU.
/// @throws std::invalid_argument For invalid geometry, configuration, dtype or incompatible parameter shapes.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(1);
/// const auto y = n::models::UnetUpConv(p, "up", n::ones({1, 4, 4, 2}), 4);
/// @endcode
/// @par References
/// - Ronneberger, Fischer and Brox(2015), U-Net: Convolutional Networks for Biomedical Image Segmentation, https://arxiv.org/abs/1505.04597.
/// @see Unet()
inline nn::Tensor UnetUpConv(nn::ParameterStore &p, const std::string &name,
                             const nn::Tensor &x, std::int64_t out) {
  nn::ConvLayerOptions up;
  up.stride = 2;
  up.padding = nn::Padding::Valid;
  up.activation = nn::Activation::Relu;
  up.init = nn::init::he_uniform();
  return nn::conv2d_transpose(p, name, x, out, 2, up);
}

/// @brief Configure original valid U-Net or an explicitly compact SAME variant.
///
/// Default grayscale/two-class64-base/four-level architecture uses valid pairs
/// and dropout.5 on deepest encoder and bottleneck during training.
///
/// @throws std::invalid_argument For invalid geometry, configuration, dtype or incompatible parameter shapes.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17);
/// n::models::UnetConfig cfg;
/// cfg.classes = 4;
/// cfg.input_channels = 3;
/// cfg.base_width = 8;
/// cfg.valid = false;
/// cfg.dropout = 0;
/// const auto net = n::models::Unet(p, n::ones({1, 32, 32, 3}), cfg);
/// @endcode
/// @par References
/// - Ronneberger, Fischer and Brox(2015), U-Net: Convolutional Networks for Biomedical Image Segmentation, https://arxiv.org/abs/1505.04597.
/// @see
struct UnetConfig {
  int classes = 2;        ///< At least two output classes.
  int input_channels = 1; ///< Positive input channel count.
  int base_width = 64;    ///< Positive first-stage width,doubles at every level.
  int levels = 4;         ///< Pooling/decoder levels1..6.
  bool valid = true;      ///< True(default) VALID pairs; false SAME compact geometry.
  float dropout = .5f; ///< Drop probability[0,1), training only on deepest encoder/bottleneck.
};
/// @brief Own U-Net logits, bottleneck, skips and decoder activations.
///
/// No independent learned storage, BN state or optimizer. Returned tensors
/// retain forward graphs borrowing parameter leaves.
///
/// @throws std::invalid_argument For invalid geometry, configuration, dtype or incompatible parameter shapes.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17);
/// n::models::UnetConfig cfg;
/// cfg.classes = 4;
/// cfg.input_channels = 3;
/// cfg.base_width = 8;
/// cfg.valid = false;
/// cfg.dropout = 0;
/// const auto net = n::models::Unet(p, n::ones({1, 32, 32, 3}), cfg);
/// @endcode
/// @par References
/// - Ronneberger, Fischer and Brox(2015), U-Net: Convolutional Networks for Biomedical Image Segmentation, https://arxiv.org/abs/1505.04597.
/// @see
struct UnetOutputs {
  nn::Tensor logits;               ///< Final linear class logits NHWC.
  nn::Tensor bottleneck;           ///< Deepest convolution-pair output before decoding.
  std::vector<nn::Tensor> skips;   ///< Encoder activations before pooling,shallow to deep.
  std::vector<nn::Tensor> decoder; ///< Decoder activations,deep to shallow.
};
/// @brief Build symmetric U-Net with learned up-convolutions and cropped concatenated skips.
///
/// Original572 grayscale maps to388 pixels,31,030,658 learned scalars and46
/// parameter tensors. Full188->4 is a smaller executable geometry probe.
/// Compact base8,SAME,RGB/four classes,no dropout retains spatial size for
/// sides divisible by16. No fitting loop or BN state. Warm all leaves before
/// constructing an optimizer; do not mutate leaves until backward.
///
/// @param p Borrowed floating learned store,creates/reuses every named leaf.
/// @param image Positive floating NHWC matching p dtype and cfg channels.
/// @param cfg Valid/SAME architecture settings,default original model.
/// @param training False(default) disables dropout; true draws through p.generator().
/// @return Linear class logits and encoder/decoder feature tensors.
/// @throws std::invalid_argument For invalid geometry, configuration, dtype or incompatible parameter shapes.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17);
/// n::models::UnetConfig cfg;
/// cfg.classes = 4;
/// cfg.input_channels = 3;
/// cfg.base_width = 8;
/// cfg.valid = false;
/// cfg.dropout = 0;
/// const auto net = n::models::Unet(p, n::ones({1, 32, 32, 3}), cfg);
/// @endcode
/// @par References
/// - Ronneberger, Fischer and Brox(2015), U-Net: Convolutional Networks for Biomedical Image Segmentation, https://arxiv.org/abs/1505.04597.
/// @see UnetOutputSide(), UnetCenterCropTo()
inline UnetOutputs Unet(nn::ParameterStore &p, const nn::Tensor &image,
                        const UnetConfig &cfg = {}, bool training = false) {
  if (cfg.classes < 2 || cfg.input_channels < 1 || cfg.base_width < 1 || cfg.levels < 1 ||
      cfg.levels > 6 || !(cfg.dropout >= 0 && cfg.dropout < 1) || image.dim() != 4 ||
      image.shape()[0] < 1 || image.shape()[1] < 1 || image.shape()[2] < 1 ||
      image.shape().back() != cfg.input_channels || image.dtype() != p.dtype())
    throw std::invalid_argument("invalid U-Net configuration");
  UnetOutputSide(int(image.shape()[1]), cfg.levels, cfg.valid);
  UnetOutputSide(int(image.shape()[2]), cfg.levels, cfg.valid);
  const nn::Padding padding = cfg.valid ? nn::Padding::Valid : nn::Padding::Same;
  UnetOutputs result;
  nn::Tensor h = image;
  for (int level = 0; level < cfg.levels; ++level) {
    h = UnetConvPair(p, "encoder" + std::to_string(level), h,
                     std::int64_t(cfg.base_width) << level, padding);
    if (training && cfg.dropout > 0 && level == cfg.levels - 1)
      h = nn::dropout(h, cfg.dropout, true, p.generator());
    result.skips.push_back(h);
    h = nn::max_pool2d(h, 2, 2, nn::Padding::Valid);
  }
  h = UnetConvPair(p, "bottleneck", h, 2 * h.shape().back(), padding);
  if (training && cfg.dropout > 0)
    h = nn::dropout(h, cfg.dropout, true, p.generator());
  result.bottleneck = h;
  for (int level = cfg.levels - 1; level >= 0; --level) {
    const std::int64_t width = std::int64_t(cfg.base_width) << level;
    const auto name = "decoder" + std::to_string(level);
    h = UnetUpConv(p, name + "/up", h, width);
    h = nn::concat({UnetCenterCropTo(result.skips[std::size_t(level)], h), h}, 3);
    h = UnetConvPair(p, name, h, width, padding);
    result.decoder.push_back(h);
  }
  nn::ConvLayerOptions output; // linear 1x1, Glorot uniform
  result.logits = nn::conv2d(p, "output", h, cfg.classes, 1, output);
  return result;
}

} // namespace datamunge::nn::models
