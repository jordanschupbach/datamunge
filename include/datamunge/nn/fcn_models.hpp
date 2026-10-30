#pragma once
/// @file fcn_models.hpp
/// @brief VGG FCN32s/16s/8s with frozen tensor bilinear deconvolutions and released crop geometry.
#include "datamunge/nn/autograd.hpp"
#include "datamunge/nn/ops.hpp"
#include "datamunge/nn/layers.hpp"
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
namespace datamunge::nn::models {
namespace nn = datamunge::nn;
/// @brief Crop FCN spatial features at an explicit top-left offset.
///
/// Reference values are unused. Only source receives value gradients.
/// Bounds are checked before slicing; no fitting or mutation.
///
/// @param source Positive NHWC source tensor.
/// @param reference Rank4 tensor supplying output H/W.
/// @param offset Nonnegative row/column offset,default0.
/// @return Source view with reference H/W and original batch/channels.
/// @throws std::invalid_argument For invalid geometry, configuration, dtype or incompatible parameter shapes.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto y = n::models::FcnSpatialCrop(n::ones({1, 7, 9, 2}), n::zeros({1, 4, 6, 1}), 1);
/// @endcode
/// @par References
/// - Long, Shelhamer and Darrell(2015), Fully Convolutional Networks for Semantic Segmentation, https://arxiv.org/abs/1411.4038; released model geometry https://github.com/shelhamer/fcn.berkeleyvision.org.
/// @see FcnFuseSkip()
inline nn::Tensor FcnSpatialCrop(const nn::Tensor &source, const nn::Tensor &reference,
                                 std::int64_t offset = 0) {
  if (source.dim() != 4 || reference.dim() != 4 || offset < 0 || reference.shape()[1] < 1 ||
      reference.shape()[2] < 1 || offset + reference.shape()[1] > source.shape()[1] ||
      offset + reference.shape()[2] > source.shape()[2])
    throw std::invalid_argument("invalid FCN crop");
  const auto &s = source.shape();
  return nn::slice(source, {0, offset, offset, 0},
                   {s[0], reference.shape()[1], reference.shape()[2], s[3]});
}

/// @brief Create a constant diagonal2s bilinear transpose-convolution filter using tensors.
///
/// The separable tent is centred at s-.5 and multiplied by channel identity.
/// No trainable parameters, fitting or random draws. Cast to source dtype before
/// use for Float64 features.
///
/// @param stride Positive integer upsampling stride s.
/// @param channels Positive class/channel count.
/// @return Float32[2s,2s,channels,channels] frozen filter,zero cross-class mixing.
/// @throws std::invalid_argument For invalid geometry, configuration, dtype or incompatible parameter shapes.
/// @par Example
/// @code
/// const auto kernel = dm::nn::models::FcnBilinearKernel(2, 3);
/// @endcode
/// @par References
/// - Long, Shelhamer and Darrell(2015), Fully Convolutional Networks for Semantic Segmentation, https://arxiv.org/abs/1411.4038; released model geometry https://github.com/shelhamer/fcn.berkeleyvision.org.
/// @see FcnBilinearDeconvolution()
inline nn::Tensor FcnBilinearKernel(int stride, int channels) {
  if (stride < 1 || stride > std::numeric_limits<int>::max() / 2 || channels < 1)
    throw std::invalid_argument("invalid bilinear filter stride/channels");
  const auto K = std::int64_t(stride) * 2;
  const auto axis =
      1. - nn::abs(nn::arange(0, double(K)).to(nn::DType::Float64) - (double(stride) - .5)) /
               double(stride);
  return ((axis.reshape({K, 1, 1, 1}) * axis.reshape({1, K, 1, 1})) *
          nn::eye(channels, nn::DType::Float64).reshape({1, 1, channels, channels}))
      .to(nn::DType::Float32);
}
/// @brief Upsample with a frozen diagonal bilinear transposed convolution.
///
/// Filters are constants,excluded from ParameterStore and optimizer.
/// No fitting, random draws or source mutation.
///
/// @param source Positive Float32/64 NHWC features with expected channel count.
/// @param stride Positive upsampling factor.
/// @param channels Positive channel count matching source.
/// @param valid False(default) SAME yields H*s; true VALID yields(H-1)*s+2*s.
/// @return Bilinear output preserving gradients to source only.
/// @throws std::invalid_argument For invalid geometry, configuration, dtype or incompatible parameter shapes.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto y = n::models::FcnBilinearDeconvolution(n::ones({1, 3, 4, 2}), 2, 2);
/// @endcode
/// @par References
/// - Long, Shelhamer and Darrell(2015), Fully Convolutional Networks for Semantic Segmentation, https://arxiv.org/abs/1411.4038; released model geometry https://github.com/shelhamer/fcn.berkeleyvision.org.
/// @see FcnBilinearKernel()
inline nn::Tensor FcnBilinearDeconvolution(const nn::Tensor &source, int stride, int channels,
                                           bool valid = false) {
  if (source.dim() != 4 || source.shape()[0] < 1 || source.shape()[1] < 1 ||
      source.shape()[2] < 1 || source.shape()[3] != channels || stride < 1)
    throw std::invalid_argument("invalid FCN upsample input");
  const auto &s = source.shape();
  const std::int64_t h = valid ? (s[1] - 1) * stride + 2 * stride : s[1] * stride;
  const std::int64_t w = valid ? (s[2] - 1) * stride + 2 * stride : s[2] * stride;
  return nn::conv2d_transpose(
      source, FcnBilinearKernel(stride, channels).to(source.dtype()), {s[0], h, w, channels},
      nn::conv_options(stride, valid ? nn::Padding::Valid : nn::Padding::Same));
}

/// @brief Upsample a coarse score map and add an aligned skip score map.
///
/// Both input score maps receive gradients. Released Caffe geometry crops
/// skips by5/9; compact mode crops upsampled maps by0. No learned parameters
/// or hidden fitting in this fusion.
///
/// @param coarse NHWC coarse scores.
/// @param skip Same-dtype NHWC skip scores.
/// @param classes Positive channel count matching both tensors.
/// @param valid False(default) SAME, true VALID bilinear upsampling.
/// @param offset Nonnegative skip crop offset,default0,used unless crop_upsampled.
/// @param crop_upsampled False(default) crops skip to upsampled map; true crops upsampled map to skip.
/// @return Sum of upsampled and aligned skip scores.
/// @throws std::invalid_argument For invalid geometry, configuration, dtype or incompatible parameter shapes.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto y = n::models::FcnFuseSkip(n::ones({1, 2, 2, 2}), n::ones({1, 4, 4, 2}), 2);
/// @endcode
/// @par References
/// - Long, Shelhamer and Darrell(2015), Fully Convolutional Networks for Semantic Segmentation, https://arxiv.org/abs/1411.4038; released model geometry https://github.com/shelhamer/fcn.berkeleyvision.org.
/// @see Fcn()
inline nn::Tensor FcnFuseSkip(const nn::Tensor &coarse, const nn::Tensor &skip, int classes,
                              bool valid = false, int offset = 0,
                              bool crop_upsampled = false) {
  const auto up = FcnBilinearDeconvolution(coarse, 2, classes, valid);
  if (crop_upsampled)
    return FcnSpatialCrop(up, skip) + skip;
  return up + FcnSpatialCrop(skip, up, offset);
}

/// @brief Configure released FCN stride variants or a compact all-stage VGG adaptation.
///
/// Full defaults21 classes,width divisor1,fc4096 and dropout keep.5.
/// Compact teaching mode uses SAME fc6 kernel3 and zero crop offsets.
///
/// @throws std::invalid_argument For invalid geometry, configuration, dtype or incompatible parameter shapes.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17);
/// n::models::FcnConfig cfg;
/// cfg.compact = true;
/// cfg.width_divisor = 8;
/// cfg.fc_width = 64;
/// cfg.classes = 4;
/// cfg.keep_probability = 1;
/// const auto net = n::models::Fcn(p, n::ones({1, 32, 32, 3}), cfg);
/// @endcode
/// @par References
/// - Long, Shelhamer and Darrell(2015), Fully Convolutional Networks for Semantic Segmentation, https://arxiv.org/abs/1411.4038; released model geometry https://github.com/shelhamer/fcn.berkeleyvision.org.
/// @see
struct FcnConfig {
  int stride = 8;        ///< Final decoder factor8/16/32,default8.
  int classes = 21;      ///< At least two class logits.
  int width_divisor = 1; ///< Positive divisor of64 for all thirteen VGG widths.
  int fc_width = 4096;   ///< Positive spatial fc6/fc7 channel width.
  bool compact = false;  ///< False released padding/crops; true SAME fc6/zero-offset crops.
  float keep_probability = .5f; ///< Dropout keep probability in(0,1],training fc6/fc7 only.
};
/// @brief Own FCN linear logits and coarse/skip/fused score features.
///
/// Returned tensors retain parameter graphs; no separate state or optimizer.
///
/// @throws std::invalid_argument For invalid geometry, configuration, dtype or incompatible parameter shapes.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17);
/// n::models::FcnConfig cfg;
/// cfg.compact = true;
/// cfg.width_divisor = 8;
/// cfg.fc_width = 64;
/// cfg.classes = 4;
/// cfg.keep_probability = 1;
/// const auto net = n::models::Fcn(p, n::ones({1, 32, 32, 3}), cfg);
/// @endcode
/// @par References
/// - Long, Shelhamer and Darrell(2015), Fully Convolutional Networks for Semantic Segmentation, https://arxiv.org/abs/1411.4038; released model geometry https://github.com/shelhamer/fcn.berkeleyvision.org.
/// @see
struct FcnOutputs {
  nn::Tensor logits;             ///< Linear NHWC logits cropped exactly to original input H/W.
  nn::Tensor coarse;             ///< Coarsest score_fr class logits.
  nn::Tensor pool3;              ///< VGG third-stage pooled features.
  nn::Tensor pool4;              ///< VGG fourth-stage pooled features.
  std::vector<nn::Tensor> fused; ///< After pool4 and optional pool3 fusion,in execution order.
};
/// @brief Build FCN32s/16s/8s from all thirteen VGG convolutions and dense prediction skips.
///
/// Full geometry adds99 pixels per edge before SAME conv1,ceil/SAME pooling,
/// VALID7x7 fc6,frozen bilinear upsampling,skip crops5/9 and final crops19/27/31.
/// Full learned counts32s/16s/8s are134346581/134357354/134362751; filters
/// are excluded. Compact divisor8/fc64/four classes retain all stages and
/// exact odd-grid alignment. No fitting or BN state. Warm every leaf before
/// optimizer creation,and do not mutate learned leaves before backward.
///
/// @param p Same-dtype learned store,creates/reuses VGG/fc/score leaves.
/// @param image Positive Float32/64 NHWC RGB tensor matching p dtype.
/// @param cfg Valid stride,width,fc,dropout and full/compact settings.
/// @param training False(default) disables dropout; true draws through p.generator().
/// @return Logits exactly aligned to input and intermediate skip/score features.
/// @throws std::invalid_argument For invalid geometry, configuration, dtype or incompatible parameter shapes.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17);
/// n::models::FcnConfig cfg;
/// cfg.compact = true;
/// cfg.width_divisor = 8;
/// cfg.fc_width = 64;
/// cfg.classes = 4;
/// cfg.keep_probability = 1;
/// const auto net = n::models::Fcn(p, n::ones({1, 32, 32, 3}), cfg);
/// @endcode
/// @par References
/// - Long, Shelhamer and Darrell(2015), Fully Convolutional Networks for Semantic Segmentation, https://arxiv.org/abs/1411.4038; released model geometry https://github.com/shelhamer/fcn.berkeleyvision.org.
/// @see FcnFuseSkip(), FcnBilinearDeconvolution()
inline FcnOutputs Fcn(nn::ParameterStore &p, const nn::Tensor &image,
                      const FcnConfig &cfg = {}, bool training = false) {
  if ((cfg.stride != 8 && cfg.stride != 16 && cfg.stride != 32) || cfg.classes < 2 ||
      cfg.width_divisor < 1 || 64 % cfg.width_divisor || cfg.fc_width < 1 ||
      !(cfg.keep_probability > 0 && cfg.keep_probability <= 1) || image.dim() != 4 ||
      image.shape()[0] < 1 || image.shape()[1] < 1 || image.shape()[2] < 1 ||
      image.shape()[3] != 3 || image.dtype() != p.dtype())
    throw std::invalid_argument("invalid FCN configuration");
  FcnOutputs result;
  nn::Tensor h = cfg.compact ? image : nn::pad(image, {{0, 0}, {99, 99}, {99, 99}, {0, 0}});
  const int repeats[] = {2, 2, 3, 3, 3};
  nn::ConvLayerOptions conv;
  conv.init = nn::init::he_uniform();
  conv.activation = nn::Activation::Relu;
  for (int stage = 0; stage < 5; ++stage) {
    const std::int64_t width = (64 << std::min(stage, 3)) / cfg.width_divisor;
    for (int block = 0; block < repeats[stage]; ++block)
      h = nn::conv2d(p, "conv" + std::to_string(stage + 1) + "_" + std::to_string(block + 1),
                     h, width, 3, conv);
    h = nn::max_pool2d(h, 2, 2, nn::Padding::Same);
    if (stage == 2)
      result.pool3 = h;
    if (stage == 3)
      result.pool4 = h;
  }
  conv.padding = cfg.compact ? nn::Padding::Same : nn::Padding::Valid;
  h = nn::conv2d(p, "fc6", h, cfg.fc_width, cfg.compact ? 3 : 7, conv);
  if (training && cfg.keep_probability < 1)
    h = nn::dropout(h, 1 - cfg.keep_probability, true, p.generator());
  conv.padding = nn::Padding::Same;
  h = nn::conv2d(p, "fc7", h, cfg.fc_width, 1, conv);
  if (training && cfg.keep_probability < 1)
    h = nn::dropout(h, 1 - cfg.keep_probability, true, p.generator());
  conv.activation = nn::Activation::None;
  result.coarse = nn::conv2d(p, "score_fr", h, cfg.classes, 1, conv);
  h = result.coarse;
  if (cfg.stride < 32) {
    const auto skip = nn::conv2d(p, "score_pool4", result.pool4, cfg.classes, 1, conv);
    h = FcnFuseSkip(h, skip, cfg.classes, !cfg.compact, cfg.compact ? 0 : 5, cfg.compact);
    result.fused.push_back(h);
  }
  if (cfg.stride == 8) {
    const auto skip = nn::conv2d(p, "score_pool3", result.pool3, cfg.classes, 1, conv);
    h = FcnFuseSkip(h, skip, cfg.classes, !cfg.compact, cfg.compact ? 0 : 9, cfg.compact);
    result.fused.push_back(h);
  }
  const auto up = FcnBilinearDeconvolution(h, cfg.stride, cfg.classes, !cfg.compact);
  result.logits = FcnSpatialCrop(up, image,
                                 cfg.compact        ? 0
                                 : cfg.stride == 32 ? 19
                                 : cfg.stride == 16 ? 27
                                                    : 31);
  return result;
}

} // namespace datamunge::nn::models
