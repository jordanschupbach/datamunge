// Johnson's reflection-padded, cropped-residual, BN image transformation net
// and a frozen ImageNet VGG-16 feature extractor: a port of tflow's
// common/nn/johnson.h (johnson_fast_style, partial_inpainting).
// Explicit symmetric stride padding and transpose cropping follow Torch's
// k3/s2/p1/adjustment1 coordinates, which differ from TensorFlow SAME.
// Parameters live in nn::ParameterStore (weights, tflow's names) plus a
// separate store for the moving BN statistics. VGG-16 weights are Keras NHWC
// kernels from vgg16.npz (tflow's `just johnson-assets`: "block<b>_conv<l>_W"
// [3,3,in,out] and "_b"); they are plain tensors that never require grad.
#pragma once

#include <array>
#include <cmath>
#include <cstdint>
#include <datamunge/nn/nn.hpp>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "common/checks.hpp"
#include "common/perceptual.hpp"

namespace style {

/// REFLECT mirror padding of H and W by `pad` (TF MirrorPad "REFLECT").
inline nn::Tensor JohnsonReflect(const nn::Tensor& x, std::int64_t pad = 40) { return nn::mirror_pad(x, pad, pad, nn::PadMode::Reflect); }

/// Drops two pixels on every spatial side (the shortcut of a VALID residual block).
inline nn::Tensor JohnsonCrop(const nn::Tensor& x) {
  return nn::slice(x, {0, 2, 2, 0}, {x.size(0), x.size(1) - 4, x.size(2) - 4, x.size(3)});
}

/// He-normal k x k convolution with zero bias ("<name>/W", "<name>/b").
inline nn::Tensor JohnsonConv(nn::ParameterStore& p, const std::string& name, const nn::Tensor& x, std::int64_t out, std::int64_t kernel,
                              std::int64_t stride = 1, nn::Padding padding = nn::Padding::Same) {
  const nn::Tensor w = p.get(name + "/W", {kernel, kernel, x.size(3), out}, nn::init::he_normal());
  const nn::Tensor b = p.get(name + "/b", {out}, nn::init::zeros());
  return nn::bias_add(nn::conv2d(x, w, nn::conv_options(stride, padding)), b);
}

/// Torch k3/s2/p1: one pixel of zero padding each side, then a VALID stride-2 convolution.
inline nn::Tensor JohnsonDown(nn::ParameterStore& p, const std::string& name, const nn::Tensor& x, std::int64_t out) {
  return JohnsonConv(p, name, nn::pad(x, {{0, 0}, {1, 1}, {1, 1}, {0, 0}}), out, 3, 2, nn::Padding::Valid);
}

/// Torch k3/s2/p1/adjustment1 transpose: VALID transpose to 2h+1, cropped from 1 to 2h.
inline nn::Tensor JohnsonUp(nn::ParameterStore& p, const std::string& name, const nn::Tensor& x, std::int64_t out) {
  const std::int64_t b = x.size(0), h = x.size(1), w = x.size(2);
  // He normal over the true fan-in 9 * in (nn::init::he_normal would read [3,3,out,in] as fan-in 9 * out).
  const nn::Tensor   weights    = p.get(name + "/W", {3, 3, out, x.size(3)}, nn::init::normal(std::sqrt(2.0 / double(9 * x.size(3)))));
  const nn::Tensor   transposed = nn::conv2d_transpose(x, weights, {b, 2 * h + 1, 2 * w + 1, out}, nn::conv_options(2, nn::Padding::Valid));
  const nn::Tensor   cropped    = nn::slice(transposed, {0, 1, 1, 0}, {b, 2 * h, 2 * w, out});
  return nn::bias_add(cropped, p.get(name + "/b", {out}, nn::init::zeros()));
}

/// BN momentum .9, eps 1e-5 (moving statistics updated only when `training && update`), optional ReLU.
inline nn::Tensor JohnsonNorm(nn::ParameterStore& p, nn::ParameterStore& state, const std::string& name, const nn::Tensor& x, bool training,
                              bool update, bool relu = true) {
  nn::BatchNormOptions o;
  o.momentum     = .9;
  o.eps          = 1e-5;
  o.update_stats = update;
  const nn::Tensor y = nn::batch_norm(p, state, name, x, training, o);
  return relu ? nn::relu(y) : y;
}

/// Two VALID 3x3 convolutions (BN/ReLU, BN) added to the 2-pixel-cropped input, no post-add ReLU.
inline nn::Tensor JohnsonResidual(nn::ParameterStore& p, nn::ParameterStore& state, const std::string& name, const nn::Tensor& x,
                                  bool training, bool update = true) {
  const std::int64_t channels = x.size(3);
  const nn::Tensor   a        = JohnsonConv(p, name + "/conv1", x, channels, 3, 1, nn::Padding::Valid);
  const nn::Tensor   h        = JohnsonNorm(p, state, name + "/bn1", a, training, update);
  const nn::Tensor   b        = JohnsonConv(p, name + "/conv2", h, channels, 3, 1, nn::Padding::Valid);
  return JohnsonCrop(x) + JohnsonNorm(p, state, name + "/bn2", b, training, update, false);
}

struct JohnsonOutputs {
  nn::Tensor              image;
  std::vector<nn::Tensor> stages;  // reflect, enc1..3, res1..5, up1..2
};

/// RGB [0, 1] -> reflect 40 -> 9x9 conv -> 2 Torch downs -> 5 cropped residuals
/// -> 2 learned Torch transposes -> 9x9 conv, .5 (tanh + 1). BN after every conv but the last.
inline JohnsonOutputs JohnsonTransform(nn::ParameterStore& p, nn::ParameterStore& state, const nn::Tensor& rgb, std::int64_t base = 32,
                                       bool training = false, bool update = true) {
  if (base < 1) throw std::invalid_argument("positive Johnson width required");
  JohnsonOutputs net;
  nn::Tensor     x = JohnsonReflect(rgb * 2.0 - 1.0);
  net.stages.push_back(x);
  x = JohnsonNorm(p, state, "enc1/bn", JohnsonConv(p, "enc1", x, base, 9), training, update);
  net.stages.push_back(x);
  x = JohnsonNorm(p, state, "enc2/bn", JohnsonDown(p, "enc2", x, base * 2), training, update);
  net.stages.push_back(x);
  x = JohnsonNorm(p, state, "enc3/bn", JohnsonDown(p, "enc3", x, base * 4), training, update);
  net.stages.push_back(x);
  for (int i = 0; i < 5; ++i) {
    x = JohnsonResidual(p, state, "res" + std::to_string(i + 1), x, training, update);
    net.stages.push_back(x);
  }
  x = JohnsonNorm(p, state, "up1/bn", JohnsonUp(p, "up1", x, base * 2), training, update);
  net.stages.push_back(x);
  x = JohnsonNorm(p, state, "up2/bn", JohnsonUp(p, "up2", x, base), training, update);
  net.stages.push_back(x);
  net.image = .5 * (nn::tanh(JohnsonConv(p, "output", x, 3, 9)) + 1.0);
  return net;
}

struct JohnsonFeatures {
  nn::Tensor                content;  // relu2_2
  std::array<nn::Tensor, 4> style;    // relu1_2, relu2_2, relu3_3, relu4_3 (unset beyond `blocks`)
};

/// Frozen Keras ImageNet VGG-16 blocks 1..4 with max pools.
class FrozenVgg16 {
 public:
  explicit FrozenVgg16(const TensorMap& tensors) : source(tensors) {
    check::Require(tensors.size() == 26, "all26 VGG-16 weight/bias tensors required");
    std::int64_t in = 3, count = 0;
    const int    depth[] = {2, 2, 3, 3, 3}, width[] = {64, 128, 256, 512, 512};
    for (int b = 0; b < 5; ++b) {
      for (int l = 0; l < depth[b]; ++l) {
        const auto name = Name(b, l);
        for (const auto& entry : std::vector<std::pair<std::string, nn::Shape>>{{name + "_W", {3, 3, in, width[b]}}, {name + "_b", {width[b]}}}) {
          const auto it = tensors.find(entry.first);
          check::Require(it != tensors.end() && it->second.dtype() == nn::DType::Float32 && it->second.shape() == entry.second,
                         "VGG-16 tensor inventory/dtype/shape: " + entry.first);
          for (float v : it->second.to_vector<float>())
            if (!std::isfinite(v)) throw std::runtime_error("nonfinite VGG-16 tensor");
          count += it->second.numel();
          constants.emplace(entry.first, it->second.detach().clone());
        }
        in = width[b];
      }
    }
    check::Require(count == 14714688, "VGG-16 convolution parameter census");
  }
  /// relu2_2 content and the last ReLU of the first `blocks` (<= 4) blocks; max pools between blocks.
  [[nodiscard]] JohnsonFeatures Extract(const nn::Tensor& rgb, int blocks = 4) const {
    JohnsonFeatures net;
    nn::Tensor      x       = VggImageNetInput(rgb);
    const int       depth[] = {2, 2, 3, 3};
    for (int b = 0; b < blocks; ++b) {
      for (int l = 0; l < depth[b]; ++l) {
        const auto name = Name(b, l);
        x               = nn::relu(nn::bias_add(nn::conv2d(x, constants.at(name + "_W")), constants.at(name + "_b")));
      }
      net.style[std::size_t(b)] = x;
      if (b == 1) net.content = x;
      if (b < 3 && b + 1 < blocks) x = nn::max_pool2d(x, 2, 2, nn::Padding::Valid);
    }
    return net;
  }
  /// Throws unless every frozen tensor still equals the imported values bitwise.
  void CheckFrozen() const {
    for (const auto& [name, value] : source) {
      const auto it = constants.find(name);
      check::Require(it != constants.end() && !it->second.requires_grad() && it->second.shape() == value.shape() &&
                         it->second.to_vector<float>() == value.to_vector<float>(),
                     "all pretrained VGG-16 values unchanged bitwise");
    }
  }
  std::map<std::string, nn::Tensor> constants;

 private:
  static std::string Name(int b, int l) { return "block" + std::to_string(b + 1) + "_conv" + std::to_string(l + 1); }
  const TensorMap&   source;
};

}  // namespace style
