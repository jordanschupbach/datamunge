// HRNetV2: parallel high-resolution streams with repeated all-to-all fusion
// (port of tflow's common/nn/hrnet.h).
// https://arxiv.org/abs/1908.07919
// https://github.com/HRNet/HRNet-Semantic-Segmentation/blob/HRNet-OCR/lib/models/seg_hrnet.py
// Upsampling paths project with 1x1 conv/BN and corner-aligned bilinear
// resize; downsampling paths chain 3x3 stride-2 conv/BN (ReLU except the
// last). Each fused output sums all paths, then ReLU. The final streams are
// resized to the stride-4 branch, concatenated, projected (1x1/BN/ReLU) and
// scored linearly, then resized to the image.
#pragma once

#include <array>
#include <datamunge/nn/nn.hpp>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/resnet.hpp"

namespace hrnet {

namespace nn = datamunge::nn;
using resnet::ConvNormConfig;

inline nn::Tensor HRResize(const nn::Tensor& input, const nn::Tensor& target) {
  return nn::resize_bilinear(input, {target.shape()[1], target.shape()[2]}, true);
}

/// Exposed separately so identity-projection routing can be checked independently.
inline std::vector<nn::Tensor> HRFuse(nn::ParameterStore& p, nn::ParameterStore& state, const std::string& name,
                                      const std::vector<nn::Tensor>& inputs, const std::vector<int>& widths,
                                      const ConvNormConfig& norm) {
  if (inputs.empty() || inputs.size() != widths.size())
    throw std::invalid_argument("HRNet fusion requires matching branches/widths");
  std::vector<nn::Tensor> outputs;
  for (std::size_t target = 0; target < inputs.size(); ++target) {
    nn::Tensor sum;
    for (std::size_t source = 0; source < inputs.size(); ++source) {
      nn::Tensor h    = inputs[source];
      const auto path = name + "/to" + std::to_string(target) + "/from" + std::to_string(source);
      if (source > target)
        h = HRResize(resnet::ResNetConv(p, state, path, h, widths[target], 1, norm, 1, false), inputs[target]);
      else if (source < target) {
        for (std::size_t step = source; step < target; ++step) {
          const bool last = step + 1 == target;
          const int  out  = last ? widths[target] : int(h.shape().back());
          h = resnet::ResNetConv(p, state, path + "/step" + std::to_string(step), h, out, 3, norm, 2, !last);
        }
      }
      sum = source == 0 ? h : sum + h;
    }
    outputs.push_back(nn::relu(sum));
  }
  return outputs;
}

struct HRNetConfig {
  int                classes = 21, width = 32, stem_width = 64, stage1_width = 64, stage1_blocks = 4, blocks = 4;
  std::array<int, 3> modules = {1, 4, 3};
};
struct HRNetOutputs {
  nn::Tensor              logits, coarse;
  std::vector<nn::Tensor> branches;
  int                     fusion_modules = 0;
};

inline HRNetOutputs HRNet(nn::ParameterStore& p, nn::ParameterStore& state, const nn::Tensor& images,
                          const HRNetConfig& cfg = {}, bool training = false) {
  if (cfg.classes < 2 || cfg.width < 1 || cfg.stem_width < 1 || cfg.stage1_width < 1 || cfg.stage1_blocks < 1 ||
      cfg.blocks < 1)
    throw std::invalid_argument("invalid HRNet configuration");
  for (int modules : cfg.modules)
    if (modules < 1) throw std::invalid_argument("HRNet stages need a fusion module");
  HRNetOutputs   r;
  ConvNormConfig norm;
  norm.training = training;
  norm.scale    = true;
  norm.momentum = .9;
  norm.epsilon  = 1e-5;
  auto conv     = [&](const std::string& name, const nn::Tensor& h, int out, int kernel, int stride = 1) {
    return resnet::ResNetConv(p, state, name, h, out, kernel, norm, stride);
  };
  nn::Tensor h = conv("stem/conv1", images, cfg.stem_width, 3, 2);
  h            = conv("stem/conv2", h, cfg.stem_width, 3, 2);
  int in       = cfg.stem_width;
  for (int block = 0; block < cfg.stage1_blocks; ++block) {
    h  = resnet::ResNetBlock(p, state, "stage1/block" + std::to_string(block), h, cfg.stage1_width, norm, 1, true).output;
    in = 4 * cfg.stage1_width;
  }
  std::vector<nn::Tensor> branches = {h};
  std::vector<int>        widths   = {in};
  for (int stage = 0; stage < 3; ++stage) {
    const int               count = stage + 2;
    std::vector<nn::Tensor> next;
    std::vector<int>        next_widths;
    for (int branch = 0; branch < count; ++branch) {
      const int  width = cfg.width * (1 << branch);
      const auto name  = "transition" + std::to_string(stage + 1) + "/branch" + std::to_string(branch);
      if (branch < int(branches.size()))
        next.push_back(width == widths[std::size_t(branch)] ? branches[std::size_t(branch)]
                                                             : conv(name, branches[std::size_t(branch)], width, 3));
      else
        next.push_back(conv(name, branches.back(), width, 3, 2));
      next_widths.push_back(width);
    }
    branches = next;
    widths   = next_widths;
    for (int module = 0; module < cfg.modules[std::size_t(stage)]; ++module) {
      const auto prefix = "stage" + std::to_string(stage + 2) + "/module" + std::to_string(module);
      for (int branch = 0; branch < count; ++branch)
        for (int block = 0; block < cfg.blocks; ++block)
          branches[std::size_t(branch)] =
              resnet::ResNetBlock(p, state, prefix + "/branch" + std::to_string(branch) + "/block" + std::to_string(block),
                                  branches[std::size_t(branch)], widths[std::size_t(branch)], norm)
                  .output;
      branches = HRFuse(p, state, prefix + "/fuse", branches, widths, norm);
      ++r.fusion_modules;
    }
  }
  r.branches = branches;
  for (std::size_t k = 1; k < branches.size(); ++k) branches[k] = HRResize(branches[k], branches[0]);
  const int channels = 15 * cfg.width;
  h                  = conv("head/project", nn::concat(branches, 3), channels, 1);
  r.coarse           = nn::conv2d(p, "classifier", h, cfg.classes, 1);
  r.logits           = HRResize(r.coarse, images);
  return r;
}

}  // namespace hrnet
