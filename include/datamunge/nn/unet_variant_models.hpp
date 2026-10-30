#pragma once
/// @file unet_variant_models.hpp
/// @brief Tensor UNet++ dense skips/deep supervision and additive attention U-Net.
#include "datamunge/nn/image_ops.hpp"
#include "datamunge/nn/unet_models.hpp"
namespace datamunge::nn::models {
namespace nn = datamunge::nn;
/// @brief Configure nested dense-skip U-Net and one supervised head per decoder depth.
///
/// Defaults base32,four levels,two classes,RGB. No BN,dropout or hidden fitting.
///
/// @throws std::invalid_argument For invalid settings,geometry,dtype or incompatible learned shapes.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17);
/// n::models::NestedUnetConfig c;
/// c.classes = 4;
/// c.base_width = 8;
/// const auto net = n::models::NestedUnet(p, n::ones({1, 32, 32, 3}), c);
/// @endcode
/// @par References
/// - Zhou et al.(2018), UNet++: A Nested U-Net Architecture for Medical Image Segmentation, https://arxiv.org/abs/1807.10165; Oktay et al.(2018), Attention U-Net: Learning Where to Look for the Pancreas, https://arxiv.org/abs/1804.03999. Convolution pairs and additive gates omit BN in this adaptation.
/// @see NestedUnet(), AttentionUnet()
struct NestedUnetConfig {
  int classes = 2;        ///< At least two pixel classes.
  int input_channels = 3; ///< Positive input channels,defaultRGB.
  int base_width = 32;    ///< Positive shallow width,doubles per encoder level.
  int levels = 4;         ///< Pooling/nested depths1..5.
};
/// @brief Own UNet++ final/deep-supervision logits and every dense nested node.
///
/// No separate parameter ownership or moving state. Returned graphs borrow
/// learned leaves until backward.
///
/// @throws std::invalid_argument For invalid settings,geometry,dtype or incompatible learned shapes.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17);
/// n::models::NestedUnetConfig c;
/// c.classes = 4;
/// c.base_width = 8;
/// const auto net = n::models::NestedUnet(p, n::ones({1, 32, 32, 3}), c);
/// @endcode
/// @par References
/// - Zhou et al.(2018), UNet++: A Nested U-Net Architecture for Medical Image Segmentation, https://arxiv.org/abs/1807.10165; Oktay et al.(2018), Attention U-Net: Learning Where to Look for the Pancreas, https://arxiv.org/abs/1804.03999. Convolution pairs and additive gates omit BN in this adaptation.
/// @see NestedUnet(), AttentionUnet()
struct NestedUnetOutputs {
  nn::Tensor logits;             ///< Deepest full-resolution head,inference output.
  std::vector<nn::Tensor> heads; ///< Full-resolution heads1..levels,all are learned.
  std::vector<nn::Tensor>
      nodes; ///< First encoder nodes x[i,0],then nested nodes in j-outer/i-inner order.
};
/// @brief Build UNet++ dense nested skip paths and all deep-supervision heads.
///
/// Each x[i,j] concatenates x[i,0..j-1] with corner-aligned bilinear
/// upsampling of x[i+1,j-1] to the actual encoder shape,then two3x3 ReLUs.
/// All configured heads are created even if inference uses only the deepest.
/// Supervise every head to fit all leaves. Full base32/two-class count9156264,
/// 68 learned tensors; deepest node joins192 input channels. No optimizer,
/// pretraining,BN or dropout. Warm all leaves before optimizer creation.
///
/// @param p Same-dtype floating learned store,creates/reuses convolution/score leaves.
/// @param images Positive floating NHWC matching configured channels and p dtype.
/// @param cfg Valid class,width and depth settings.
/// @return Deepest logits,all level heads and ordered nested feature nodes.
/// @throws std::invalid_argument For invalid settings,geometry,dtype or incompatible learned shapes.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17);
/// n::models::NestedUnetConfig c;
/// c.classes = 4;
/// c.base_width = 8;
/// const auto net = n::models::NestedUnet(p, n::ones({1, 32, 32, 3}), c);
/// @endcode
/// @par References
/// - Zhou et al.(2018), UNet++: A Nested U-Net Architecture for Medical Image Segmentation, https://arxiv.org/abs/1807.10165; Oktay et al.(2018), Attention U-Net: Learning Where to Look for the Pancreas, https://arxiv.org/abs/1804.03999. Convolution pairs and additive gates omit BN in this adaptation.
/// @see UnetConvPair()
inline NestedUnetOutputs NestedUnet(nn::ParameterStore &p, const nn::Tensor &images,
                                    const NestedUnetConfig &cfg = {}) {
  if (cfg.classes < 2 || cfg.input_channels < 1 || cfg.base_width < 1 || cfg.levels < 1 ||
      cfg.levels > 5 || images.dim() != 4 || images.shape()[0] < 1 ||
      images.shape()[1] < (1 << cfg.levels) || images.shape()[2] < (1 << cfg.levels) ||
      images.shape().back() != cfg.input_channels || images.dtype() != p.dtype())
    throw std::invalid_argument("invalid nested U-Net configuration");
  std::vector<std::vector<nn::Tensor>> grid(
      std::size_t(cfg.levels + 1), std::vector<nn::Tensor>(std::size_t(cfg.levels + 1)));
  NestedUnetOutputs result;
  /// @brief Name architecture grid nodes from integer depth/column metadata.
  auto node = [](int i, int j) {
    return "x" + std::to_string(i) + "_" + std::to_string(j);
  };
  for (int i = 0; i <= cfg.levels; ++i) {
    const std::int64_t width = std::int64_t(cfg.base_width) << i;
    grid[std::size_t(i)][0] = UnetConvPair(
        p, node(i, 0),
        i ? nn::max_pool2d(grid[std::size_t(i - 1)][0], 2, 2, nn::Padding::Valid) : images,
        width);
    result.nodes.push_back(grid[std::size_t(i)][0]);
  }
  for (int j = 1; j <= cfg.levels; ++j)
    for (int i = 0; i + j <= cfg.levels; ++i) {
      const std::int64_t width = std::int64_t(cfg.base_width) << i;
      std::vector<nn::Tensor> inputs(grid[std::size_t(i)].begin(),
                                     grid[std::size_t(i)].begin() + j);
      const auto &reference = grid[std::size_t(i)][0];
      inputs.push_back(nn::resize_bilinear(grid[std::size_t(i + 1)][std::size_t(j - 1)],
                                           {reference.shape()[1], reference.shape()[2]},
                                           true));
      grid[std::size_t(i)][std::size_t(j)] =
          UnetConvPair(p, node(i, j), nn::concat(inputs, 3), width);
      result.nodes.push_back(grid[std::size_t(i)][std::size_t(j)]);
    }
  nn::ConvLayerOptions output; // linear 1x1, Glorot uniform
  for (int j = 1; j <= cfg.levels; ++j)
    result.heads.push_back(nn::conv2d(p, "head" + std::to_string(j), grid[0][std::size_t(j)],
                                      cfg.classes, 1, output));
  result.logits = result.heads.back();
  return result;
}

/// @brief Own additive attention coefficients and gated skip features.
///
/// Coefficients are shared over skip channels; returned tensors preserve both
/// input and learned projection gradients.
///
/// @throws std::invalid_argument For invalid settings,geometry,dtype or incompatible learned shapes.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17);
/// const auto gate =
///     n::models::AttentionGate(p, "gate", n::ones({1, 4, 4, 2}), n::ones({1, 4, 4, 4}), 1);
/// @endcode
/// @par References
/// - Zhou et al.(2018), UNet++: A Nested U-Net Architecture for Medical Image Segmentation, https://arxiv.org/abs/1807.10165; Oktay et al.(2018), Attention U-Net: Learning Where to Look for the Pancreas, https://arxiv.org/abs/1804.03999. Convolution pairs and additive gates omit BN in this adaptation.
/// @see NestedUnet(), AttentionUnet()
struct AttentionGateOutputs {
  nn::Tensor coefficients; ///< [N,H,W,1] sigmoid coefficients in[0,1].
  nn::Tensor output;       ///< [N,H,W,skip_channels] skip*coefficients.
};
/// @brief Apply an additive sigmoid gate to spatially aligned skip features.
///
/// coefficients=sigmoid(psi(ReLU(Wx*skip+Wg*gate))). Projections use
/// Glorot initialization with biases,no BN. Both inputs receive gradients.
/// No resizing,random augmentation or fitting is hidden.
///
/// @param p Same-dtype learned store for x/g/psi1x1 projections and biases.
/// @param name Unique gate parameter prefix.
/// @param skip Positive floating NHWC skip features.
/// @param gate Same-dtype NHWC context with matching N/H/W.
/// @param hidden Positive projection channel width.
/// @return Sigmoid coefficients and gated skip tensor.
/// @throws std::invalid_argument For invalid settings,geometry,dtype or incompatible learned shapes.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17);
/// const auto gate =
///     n::models::AttentionGate(p, "gate", n::ones({1, 4, 4, 2}), n::ones({1, 4, 4, 4}), 1);
/// @endcode
/// @par References
/// - Zhou et al.(2018), UNet++: A Nested U-Net Architecture for Medical Image Segmentation, https://arxiv.org/abs/1807.10165; Oktay et al.(2018), Attention U-Net: Learning Where to Look for the Pancreas, https://arxiv.org/abs/1804.03999. Convolution pairs and additive gates omit BN in this adaptation.
/// @see AttentionUnet()
inline AttentionGateOutputs AttentionGate(nn::ParameterStore &p, const std::string &name,
                                          const nn::Tensor &skip, const nn::Tensor &gate,
                                          std::int64_t hidden) {
  if (hidden < 1 || skip.dim() != 4 || gate.dim() != 4 || skip.shape()[0] < 1 ||
      skip.shape()[1] < 1 || skip.shape()[2] < 1 || skip.shape()[3] < 1 ||
      gate.shape()[3] < 1 || skip.shape()[0] != gate.shape()[0] ||
      skip.shape()[1] != gate.shape()[1] || skip.shape()[2] != gate.shape()[2] ||
      skip.dtype() != p.dtype() || gate.dtype() != p.dtype())
    throw std::invalid_argument("positive attention gate widths required");
  nn::ConvLayerOptions projection;
  const auto x = nn::conv2d(p, name + "/x", skip, hidden, 1, projection);
  const auto g = nn::conv2d(p, name + "/g", gate, hidden, 1, projection);
  AttentionGateOutputs r;
  r.coefficients =
      nn::sigmoid(nn::conv2d(p, name + "/psi", nn::relu(x + g), 1, 1, projection));
  r.output = skip * r.coefficients;
  return r;
}

/// @brief Configure an additive attention-gated SAME U-Net.
///
/// Defaults base64,four pooling levels,RGB/two classes; no BN or dropout.
///
/// @throws std::invalid_argument For invalid settings,geometry,dtype or incompatible learned shapes.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17);
/// n::models::AttentionUnetConfig c;
/// c.classes = 4;
/// c.base_width = 8;
/// const auto net = n::models::AttentionUnet(p, n::ones({1, 32, 32, 3}), c);
/// @endcode
/// @par References
/// - Zhou et al.(2018), UNet++: A Nested U-Net Architecture for Medical Image Segmentation, https://arxiv.org/abs/1807.10165; Oktay et al.(2018), Attention U-Net: Learning Where to Look for the Pancreas, https://arxiv.org/abs/1804.03999. Convolution pairs and additive gates omit BN in this adaptation.
/// @see NestedUnet(), AttentionUnet()
struct AttentionUnetConfig {
  int classes = 2;        ///< At least two class logits.
  int input_channels = 3; ///< Positive image channels,defaultRGB.
  int base_width = 64;    ///< Shallow width>=2,doubles per level.
  int levels = 4;         ///< Pooling/decoder depths1..5.
};
/// @brief Own attention U-Net class logits and decoder gate coefficient maps.
///
/// No independent parameters,moving state or optimizer. Learned leaves remain
/// borrowed by the returned graph until backward.
///
/// @throws std::invalid_argument For invalid settings,geometry,dtype or incompatible learned shapes.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17);
/// n::models::AttentionUnetConfig c;
/// c.classes = 4;
/// c.base_width = 8;
/// const auto net = n::models::AttentionUnet(p, n::ones({1, 32, 32, 3}), c);
/// @endcode
/// @par References
/// - Zhou et al.(2018), UNet++: A Nested U-Net Architecture for Medical Image Segmentation, https://arxiv.org/abs/1807.10165; Oktay et al.(2018), Attention U-Net: Learning Where to Look for the Pancreas, https://arxiv.org/abs/1804.03999. Convolution pairs and additive gates omit BN in this adaptation.
/// @see NestedUnet(), AttentionUnet()
struct AttentionUnetOutputs {
  nn::Tensor logits; ///< Final linear class logits at the restored spatial size.
  std::vector<nn::Tensor> coefficients; ///< Decoder gates,deepest to shallowest.
};
/// @brief Build learned-upsample U-Net with an additive attention gate on each cropped skip.
///
/// Each decoder uses learned2x2 stride2 up-convolution,a floor-centred encoder
/// trace crop,additive gate and concatenated two3x3 ReLU pair. Full base64/
/// two-class count31381414,70 learned tensors. Sides divisible by2^levels
/// retain their size; other valid sides round down through pooling/decoding.
/// No pretrained weights,optimizer,BN or dropout. Warm every leaf before
/// optimizer creation and keep learned leaves unchanged until backward.
///
/// @param p Same-dtype learned store,creates/reuses every named leaf.
/// @param images Positive floating NHWC matching p dtype and configured channels.
/// @param cfg Valid class,width and depth settings.
/// @return Final logits and per-decoder sigmoid coefficient maps.
/// @throws std::invalid_argument For invalid settings,geometry,dtype or incompatible learned shapes.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17);
/// n::models::AttentionUnetConfig c;
/// c.classes = 4;
/// c.base_width = 8;
/// const auto net = n::models::AttentionUnet(p, n::ones({1, 32, 32, 3}), c);
/// @endcode
/// @par References
/// - Zhou et al.(2018), UNet++: A Nested U-Net Architecture for Medical Image Segmentation, https://arxiv.org/abs/1807.10165; Oktay et al.(2018), Attention U-Net: Learning Where to Look for the Pancreas, https://arxiv.org/abs/1804.03999. Convolution pairs and additive gates omit BN in this adaptation.
/// @see AttentionGate(), UnetCenterCropTo()
inline AttentionUnetOutputs AttentionUnet(nn::ParameterStore &p, const nn::Tensor &images,
                                          const AttentionUnetConfig &cfg = {}) {
  if (cfg.classes < 2 || cfg.input_channels < 1 || cfg.base_width < 2 || cfg.levels < 1 ||
      cfg.levels > 5 || images.dim() != 4 || images.shape()[0] < 1 ||
      images.shape()[1] < (1 << cfg.levels) || images.shape()[2] < (1 << cfg.levels) ||
      images.shape().back() != cfg.input_channels || images.dtype() != p.dtype())
    throw std::invalid_argument("invalid attention U-Net configuration");
  std::vector<nn::Tensor> skips;
  nn::Tensor h = images;
  for (int i = 0; i < cfg.levels; ++i) {
    h = UnetConvPair(p, "encoder" + std::to_string(i), h, std::int64_t(cfg.base_width) << i);
    skips.push_back(h);
    h = nn::max_pool2d(h, 2, 2, nn::Padding::Valid);
  }
  h = UnetConvPair(p, "bottleneck", h, 2 * h.shape().back());
  AttentionUnetOutputs result;
  for (int i = cfg.levels - 1; i >= 0; --i) {
    const std::int64_t width = std::int64_t(cfg.base_width) << i;
    const auto name = "decoder" + std::to_string(i);
    h = UnetUpConv(p, name + "/up", h, width);
    const auto gate = AttentionGate(p, name + "/gate",
                                    UnetCenterCropTo(skips[std::size_t(i)], h), h, width / 2);
    result.coefficients.push_back(gate.coefficients);
    h = UnetConvPair(p, name, nn::concat({gate.output, h}, 3), width);
  }
  nn::ConvLayerOptions output;
  result.logits = nn::conv2d(p, "output", h, cfg.classes, 1, output);
  return result;
}

} // namespace datamunge::nn::models
