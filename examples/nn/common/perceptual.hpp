// Fixed pretrained VGG-19 convolutional features for perceptual objectives:
// a port of tflow's common/nn/perceptual.h (gatys_style_transfer,
// johnson_fast_style, ...). Keras NHWC kernels from vgg19.npz (tflow's
// `just style-assets`: "block<b>_conv<l>_W" [3,3,in,out] and "_b"), RGB [0,1]
// -> ImageNet-centred BGR [0,255], average pools instead of max pools.
// The weights are plain tensors that never require grad (TF: Const nodes).
#pragma once

#include <array>
#include <cmath>
#include <cstdint>
#include <datamunge/nn/nn.hpp>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "common/checks.hpp"

namespace style {

namespace nn = datamunge::nn;

using TensorMap = std::map<std::string, nn::Tensor>;

/// .npz members by name.
inline TensorMap LoadTensorMap(const std::string& path) {
  TensorMap out;
  for (auto& [name, value] : nn::load_npz(path)) out[name] = value;
  return out;
}

/// tflow VggImageNetInput: RGB [0, 1] -> BGR [0, 255] minus the ImageNet means.
inline nn::Tensor VggImageNetInput(const nn::Tensor& rgb) {
  const nn::Tensor bgr = nn::gather(rgb, nn::tensor({2, 1, 0}, {3}, nn::DType::Int64), 3);
  return bgr * 255.0 - nn::tensor({103.939f, 116.779f, 123.68f}, {3}).to(rgb.dtype());
}

struct PerceptualFeatures {
  nn::Tensor                content;  // block4_conv2
  std::array<nn::Tensor, 5> style;    // block1..5_conv1
};

class FrozenVgg19 {
 public:
  static constexpr std::array<int, 5> Depths = {2, 2, 4, 4, 4}, Widths = {64, 128, 256, 512, 512};
  explicit FrozenVgg19(const TensorMap& tensors) : source(tensors) {
    check::Require(tensors.size() == 32, "exact16 VGG-19 kernel/bias pairs required");
    std::int64_t channels = 3, count = 0;
    for (int block = 0; block < 5; ++block)
      for (int layer = 0; layer < Depths[std::size_t(block)]; ++layer) {
        const auto         name = Name(block, layer);
        const std::int64_t out  = Widths[std::size_t(block)];
        for (const auto& entry : std::vector<std::pair<std::string, nn::Shape>>{{name + "_W", {3, 3, channels, out}}, {name + "_b", {out}}}) {
          const auto it = tensors.find(entry.first);
          check::Require(it != tensors.end() && it->second.dtype() == nn::DType::Float32 && it->second.shape() == entry.second,
                         "VGG-19 pretrained tensor inventory/dtype/shape: " + entry.first);
          for (float v : it->second.to_vector<float>())
            if (!std::isfinite(v)) throw std::runtime_error("non-finite pretrained VGG-19 tensor");
          count += it->second.numel();
          constants.emplace(entry.first, it->second.detach().clone());
        }
        channels = out;
      }
    check::Require(count == 20024384, "full pretrained VGG-19 convolution parameter census");
  }
  /// Features up to block5_conv1 (the deeper layers are never evaluated).
  [[nodiscard]] PerceptualFeatures Extract(const nn::Tensor& rgb) const {
    PerceptualFeatures output;
    nn::Tensor         x = VggImageNetInput(rgb);
    for (int block = 0; block < 5; ++block) {
      for (int layer = 0; layer < Depths[std::size_t(block)]; ++layer) {
        const auto name = Name(block, layer);
        x = nn::relu(nn::bias_add(nn::conv2d(x, constants.at(name + "_W")), constants.at(name + "_b")));
        if (layer == 0) output.style[std::size_t(block)] = x;
        if (block == 3 && layer == 1) output.content = x;
        if (block == 4 && layer == 0) return output;
      }
      x = nn::avg_pool2d(x, 2, 2, nn::Padding::Valid);
    }
    return output;
  }
  /// Throws unless every frozen tensor still equals the imported values bitwise.
  void CheckFrozen() const {
    for (const auto& [name, value] : source) {
      const auto it = constants.find(name);
      check::Require(it != constants.end() && !it->second.requires_grad() && it->second.shape() == value.shape() &&
                         it->second.to_vector<float>() == value.to_vector<float>(),
                     "all pretrained VGG-19 values remain bitwise unchanged");
    }
  }
  std::map<std::string, nn::Tensor> constants;

 private:
  static std::string Name(int block, int layer) { return "block" + std::to_string(block + 1) + "_conv" + std::to_string(layer + 1); }
  const TensorMap&   source;
};

}  // namespace style
