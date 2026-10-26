// FCN-32s/16s/8s: VGG-16 converted to dense spatial prediction and skip
// fusion (port of tflow's common/nn/fcn.h).
// https://arxiv.org/abs/1411.4038
// https://github.com/shelhamer/fcn.berkeleyvision.org
// Full mode follows the released Caffe geometry: conv1_1 pad 100 (99 extra
// plus SAME), ceil-mode (SAME) pooling, valid 7x7 fc6, fc7 and score_fr;
// fixed bilinear deconvolutions, skip pool4 crop 5 and pool3 crop 9, final
// crop 19/27/31 for 32s/16s/8s. Compact mode explicitly uses SAME padding,
// width/8, fc6 3x3/64 channels, and crop 0 to avoid excessive padding in
// small teaching images. It retains all thirteen VGG convolutions and the
// same skip-fusion/deconvolution paths. The bilinear filters are constants
// (frozen upsampling), applied with nn::conv2d_transpose (TF Conv2DBackpropInput).
#pragma once

#include <cmath>
#include <datamunge/nn/nn.hpp>
#include <stdexcept>
#include <string>
#include <vector>

namespace fcn {

namespace nn = datamunge::nn;

/// source[:, offset:offset+H_ref, offset:offset+W_ref, :].
inline nn::Tensor SpatialCrop(const nn::Tensor& source, const nn::Tensor& reference, std::int64_t offset = 0) {
  const auto& s = source.shape();
  return nn::slice(source, {0, offset, offset, 0}, {s[0], reference.shape()[1], reference.shape()[2], s[3]});
}

/// Diagonal bilinear deconvolution filter [2s, 2s, channels, channels] (no cross-class mixing).
inline nn::Tensor BilinearKernel(int stride, int channels) {
  if (stride < 1 || channels < 1) throw std::invalid_argument("positive bilinear stride/channels required");
  const int          kernel = 2 * stride;
  const double       centre = stride - .5;
  std::vector<float> values(std::size_t(kernel) * kernel * channels * channels, 0.f);
  for (int r = 0; r < kernel; ++r)
    for (int c = 0; c < kernel; ++c)
      for (int k = 0; k < channels; ++k)
        values[((std::size_t(r) * kernel + c) * channels + k) * channels + k] =
            float((1 - std::abs(r - centre) / stride) * (1 - std::abs(c - centre) / stride));
  return nn::from_vector(values, {kernel, kernel, channels, channels});
}

/// Fixed bilinear upsampling by `stride`: H s (SAME) or (H - 1) s + 2 s (VALID).
inline nn::Tensor BilinearDeconvolution(const nn::Tensor& source, int stride, int channels, bool valid = false) {
  const auto&        s = source.shape();
  const std::int64_t h = valid ? (s[1] - 1) * stride + 2 * stride : s[1] * stride;
  const std::int64_t w = valid ? (s[2] - 1) * stride + 2 * stride : s[2] * stride;
  return nn::conv2d_transpose(source, BilinearKernel(stride, channels), {s[0], h, w, channels},
                              nn::conv_options(stride, valid ? nn::Padding::Valid : nn::Padding::Same));
}

inline nn::Tensor FcnFuseSkip(const nn::Tensor& coarse, const nn::Tensor& skip, int classes, bool valid = false,
                              int offset = 0, bool crop_upsampled = false) {
  const auto up = BilinearDeconvolution(coarse, 2, classes, valid);
  if (crop_upsampled) return SpatialCrop(up, skip) + skip;
  return up + SpatialCrop(skip, up, offset);
}

struct FcnConfig {
  int   stride = 8, classes = 21, width_divisor = 1, fc_width = 4096;
  bool  compact          = false;
  float keep_probability = .5f;
};
struct FcnOutputs {
  nn::Tensor              logits, coarse, pool3, pool4;
  std::vector<nn::Tensor> fused;
};

inline FcnOutputs Fcn(nn::ParameterStore& p, const nn::Tensor& image, const FcnConfig& cfg = {}, bool training = false) {
  if ((cfg.stride != 8 && cfg.stride != 16 && cfg.stride != 32) || cfg.classes < 2 || cfg.width_divisor < 1 ||
      64 % cfg.width_divisor || cfg.fc_width < 1 || !(cfg.keep_probability > 0 && cfg.keep_probability <= 1))
    throw std::invalid_argument("invalid FCN configuration");
  FcnOutputs result;
  nn::Tensor h = cfg.compact ? image : nn::pad(image, {{0, 0}, {99, 99}, {99, 99}, {0, 0}});
  const int  repeats[] = {2, 2, 3, 3, 3};
  nn::ConvLayerOptions conv;
  conv.init       = nn::init::he_uniform();
  conv.activation = nn::Activation::Relu;
  for (int stage = 0; stage < 5; ++stage) {
    const std::int64_t width = (64 << std::min(stage, 3)) / cfg.width_divisor;
    for (int block = 0; block < repeats[stage]; ++block)
      h = nn::conv2d(p, "conv" + std::to_string(stage + 1) + "_" + std::to_string(block + 1), h, width, 3, conv);
    h = nn::max_pool2d(h, 2, 2, nn::Padding::Same);
    if (stage == 2) result.pool3 = h;
    if (stage == 3) result.pool4 = h;
  }
  conv.padding = cfg.compact ? nn::Padding::Same : nn::Padding::Valid;
  h            = nn::conv2d(p, "fc6", h, cfg.fc_width, cfg.compact ? 3 : 7, conv);
  if (training && cfg.keep_probability < 1) h = nn::dropout(h, 1 - cfg.keep_probability, true, p.generator());
  conv.padding = nn::Padding::Same;
  h            = nn::conv2d(p, "fc7", h, cfg.fc_width, 1, conv);
  if (training && cfg.keep_probability < 1) h = nn::dropout(h, 1 - cfg.keep_probability, true, p.generator());
  conv.activation = nn::Activation::None;
  result.coarse   = nn::conv2d(p, "score_fr", h, cfg.classes, 1, conv);
  h               = result.coarse;
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
  const auto up = BilinearDeconvolution(h, cfg.stride, cfg.classes, !cfg.compact);
  result.logits = SpatialCrop(up, image, cfg.compact ? 0 : cfg.stride == 32 ? 19 : cfg.stride == 16 ? 27 : 31);
  return result;
}

}  // namespace fcn
