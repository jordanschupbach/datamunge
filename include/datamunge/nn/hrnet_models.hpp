#pragma once
/// @file hrnet_models.hpp
/// @brief HRNetV2 persistent high-resolution streams and repeated all-to-all exchange fusion.
#include "datamunge/nn/image_ops.hpp"
#include "datamunge/nn/resnet_models.hpp"
#include <limits>
namespace datamunge::nn::models {
namespace nn = datamunge::nn;
/// @brief Resize floating HRNet features by aligned-corner bilinear sampling to another spatial shape.
///
/// No learned parameters,fitting or source mutation. Reference values receive
/// no gradient;aligned-corner interpolation preserves endpoint coordinates.
///
/// @param input Positive floating NHWC features.
/// @param target Rank4 tensor supplying positive target H/W;values unused.
/// @return Resized input with source gradients and unchanged batch/channels.
/// @throws std::invalid_argument For invalid settings,geometry,dtypes or incompatible learned shapes.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto y = n::models::HRResize(n::ones({1, 3, 4, 2}), n::zeros({1, 5, 7, 1}));
/// @endcode
/// @par References
/// - Wang et al., Deep High-Resolution Representation Learning for Visual Recognition, https://arxiv.org/abs/1908.07919; authors semantic-segmentation implementation https://github.com/HRNet/HRNet-Semantic-Segmentation/blob/HRNet-OCR/lib/models/seg_hrnet.py.
/// @see HRFuse()
inline nn::Tensor HRResize(const nn::Tensor &input, const nn::Tensor &target) {
  if (input.dim() != 4 || target.dim() != 4 || target.shape()[1] < 1 || target.shape()[2] < 1)
    throw std::invalid_argument("invalid HRNet resize reference");
  return nn::resize_bilinear(input, {target.shape()[1], target.shape()[2]}, true);
}

/// @brief Exchange every HRNet branch with all others and sum/ReLU at each target resolution.
///
/// Lower-resolution sources project1x1/BN without ReLU,then aligned-corner
/// bilinear resize. Higher-resolution sources chain3x3 stride2 convolutions,
/// ReLU except the final step. Identity paths preserve their own stream. Every
/// target sums all routes then ReLU. Spatial sizes must match repeated ceil-half
/// downsampling. No optimizer or hidden fitting.
///
/// @param p Same-dtype learned projection/downsampling store.
/// @param state Same-dtype BN moving state,controlled by norm.
/// @param name Unique exchange prefix.
/// @param inputs Nonempty ordered high-to-low resolution positive NHWC feature tensors,common N/dtype.
/// @param widths Positive channels for each branch,matching inputs.
/// @param norm Borrowed ResNet normalization/training/update settings.
/// @return One fused tensor per input resolution/channel shape.
/// @throws std::invalid_argument For invalid settings,geometry,dtypes or incompatible learned shapes.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(17);
/// n::models::ResNetNormConfig norm;
/// norm.scale = true;
/// const auto fused = n::models::HRFuse(
///     p, state, "exchange", {n::ones({1, 5, 7, 2}), n::ones({1, 3, 4, 4})}, {2, 4}, norm);
/// @endcode
/// @par References
/// - Wang et al., Deep High-Resolution Representation Learning for Visual Recognition, https://arxiv.org/abs/1908.07919; authors semantic-segmentation implementation https://github.com/HRNet/HRNet-Semantic-Segmentation/blob/HRNet-OCR/lib/models/seg_hrnet.py.
/// @see HRResize(), HRNet()
inline std::vector<nn::Tensor> HRFuse(nn::ParameterStore &p, nn::ParameterStore &state,
                                      const std::string &name,
                                      const std::vector<nn::Tensor> &inputs,
                                      const std::vector<int> &widths,
                                      const ResNetNormConfig &norm) {
  if (inputs.empty() || inputs.size() != widths.size())
    throw std::invalid_argument("HRNet fusion requires matching branches/widths");
  for (std::size_t i = 0; i < inputs.size(); ++i) {
    const auto &x = inputs[i];
    if (x.dim() != 4 || x.shape()[0] < 1 || x.shape()[1] < 1 || x.shape()[2] < 1 ||
        widths[i] < 1 || x.shape()[3] != widths[i] || x.shape()[0] != inputs[0].shape()[0] ||
        x.dtype() != p.dtype() || state.dtype() != p.dtype())
      throw std::invalid_argument("invalid HRNet fusion branch");
    if (i && (x.shape()[1] != (inputs[i - 1].shape()[1] + 1) / 2 ||
              x.shape()[2] != (inputs[i - 1].shape()[2] + 1) / 2))
      throw std::invalid_argument("HRNet branch spatial resolutions must ceil-half");
  }
  std::vector<nn::Tensor> outputs;
  for (std::size_t target = 0; target < inputs.size(); ++target) {
    nn::Tensor sum;
    for (std::size_t source = 0; source < inputs.size(); ++source) {
      nn::Tensor h = inputs[source];
      const auto path =
          name + "/to" + std::to_string(target) + "/from" + std::to_string(source);
      if (source > target)
        h = HRResize(ResNetConv(p, state, path, h, widths[target], 1, norm, 1, false),
                     inputs[target]);
      else if (source < target) {
        for (std::size_t step = source; step < target; ++step) {
          const bool last = step + 1 == target;
          const int out = last ? widths[target] : int(h.shape().back());
          h = ResNetConv(p, state, path + "/step" + std::to_string(step), h, out, 3, norm, 2,
                         !last);
        }
      }
      sum = source == 0 ? h : sum + h;
    }
    outputs.push_back(nn::relu(sum));
  }
  return outputs;
}

/// @brief Configure HRNetV2 widths,residual depths and repeated fusion module counts.
///
/// Full defaults W32,two64-channel stems,four64-width bottlenecks,then
/// 2/3/4 parallel branches with1/4/3 modules and four basic blocks each.
///
/// @throws std::invalid_argument For invalid settings,geometry,dtypes or incompatible learned shapes.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(17);
/// n::models::HRNetConfig c;
/// c.classes = 4;
/// c.width = 8;
/// c.stem_width = 16;
/// c.stage1_width = 16;
/// c.stage1_blocks = 2;
/// c.blocks = 2;
/// c.modules = {1, 1, 1};
/// const auto net = n::models::HRNet(p, state, n::ones({1, 33, 33, 3}), c);
/// @endcode
/// @par References
/// - Wang et al., Deep High-Resolution Representation Learning for Visual Recognition, https://arxiv.org/abs/1908.07919; authors semantic-segmentation implementation https://github.com/HRNet/HRNet-Semantic-Segmentation/blob/HRNet-OCR/lib/models/seg_hrnet.py.
/// @see HRNet()
struct HRNetConfig {
  int classes = 21; ///< At least two linear class scores.
  int width = 32;   ///< Positive highest-resolution width,lower streams double progressively.
  int stem_width = 64;                    ///< Positive two-convolution stem channel width.
  int stage1_width = 64;                  ///< Positive bottleneck inner width,output4*width.
  int stage1_blocks = 4;                  ///< Positive initial bottleneck count.
  int blocks = 4;                         ///< Positive basic-block count per branch/module.
  std::array<int, 3> modules = {1, 4, 3}; ///< Positive module counts for2/3/4-branch stages.
};
/// @brief Own HRNet logits,fused coarse scores and persistent final branch maps.
///
/// No independent learned or BN stores. Returned tensors retain borrowed
/// parameter graphs until backward.
///
/// @throws std::invalid_argument For invalid settings,geometry,dtypes or incompatible learned shapes.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(17);
/// n::models::HRNetConfig c;
/// c.classes = 4;
/// c.width = 8;
/// c.stem_width = 16;
/// c.stage1_width = 16;
/// c.stage1_blocks = 2;
/// c.blocks = 2;
/// c.modules = {1, 1, 1};
/// const auto net = n::models::HRNet(p, state, n::ones({1, 33, 33, 3}), c);
/// @endcode
/// @par References
/// - Wang et al., Deep High-Resolution Representation Learning for Visual Recognition, https://arxiv.org/abs/1908.07919; authors semantic-segmentation implementation https://github.com/HRNet/HRNet-Semantic-Segmentation/blob/HRNet-OCR/lib/models/seg_hrnet.py.
/// @see HRNet()
struct HRNetOutputs {
  nn::Tensor logits; ///< Linear class logits resized to exact original image H/W.
  nn::Tensor coarse; ///< Linear class scores on highest-resolution stride4 branch grid.
  std::vector<nn::Tensor>
      branches; ///< Final four maps before resizing/concatenation,high to low resolution.
  int fusion_modules = 0; ///< Actual executed exchange module count,sum(cfg.modules).
};
/// @brief Build persistent-resolution HRNetV2 with repeated all-to-all exchange and fused class head.
///
/// Two stride2 stems precede bottlenecks. Stages add one lower stream at a
/// time and keep every earlier stream active. Final maps resize to stride4,
/// concatenate15*width channels,project1x1/BN/ReLU and classify linearly.
/// BN epsilon1e-5/momentum.9. No pretrained weights or optimizer. Warm every
/// leaf before fitting,checkpoint learned/BN stores together,and keep learned
/// leaves unchanged until backward.
///
/// @param p Same-dtype floating learned store,creates/reuses every residual/projection/classifier leaf.
/// @param state Same-dtype BN moving moments,updated during training.
/// @param images Positive floating RGB NHWC matching both stores.
/// @param cfg Valid class,width,depth and positive module settings.
/// @param training False(default) moving moments;true batch moments and moving updates.
/// @return Input-aligned logits,coarse scores,four final branches and exchange count.
/// @throws std::invalid_argument For invalid settings,geometry,dtypes or incompatible learned shapes.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(17);
/// n::models::HRNetConfig c;
/// c.classes = 4;
/// c.width = 8;
/// c.stem_width = 16;
/// c.stage1_width = 16;
/// c.stage1_blocks = 2;
/// c.blocks = 2;
/// c.modules = {1, 1, 1};
/// const auto net = n::models::HRNet(p, state, n::ones({1, 33, 33, 3}), c);
/// @endcode
/// @par References
/// - Wang et al., Deep High-Resolution Representation Learning for Visual Recognition, https://arxiv.org/abs/1908.07919; authors semantic-segmentation implementation https://github.com/HRNet/HRNet-Semantic-Segmentation/blob/HRNet-OCR/lib/models/seg_hrnet.py.
/// @see HRFuse(), HRResize()
inline HRNetOutputs HRNet(nn::ParameterStore &p, nn::ParameterStore &state,
                          const nn::Tensor &images, const HRNetConfig &cfg = {},
                          bool training = false) {
  if (cfg.classes < 2 || cfg.width < 1 || cfg.stem_width < 1 || cfg.stage1_width < 1 ||
      cfg.stage1_blocks < 1 || cfg.blocks < 1 ||
      cfg.width > std::numeric_limits<int>::max() / 15 ||
      cfg.stage1_width > std::numeric_limits<int>::max() / 4 || images.dim() != 4 ||
      images.shape()[0] < 1 || images.shape()[1] < 1 || images.shape()[2] < 1 ||
      images.shape()[3] != 3 || images.dtype() != p.dtype() || state.dtype() != p.dtype())
    throw std::invalid_argument("invalid HRNet configuration");
  for (int modules : cfg.modules)
    if (modules < 1)
      throw std::invalid_argument("HRNet stages need a fusion module");
  HRNetOutputs r;
  ResNetNormConfig norm;
  norm.training = training;
  norm.scale = true;
  norm.momentum = .9;
  norm.epsilon = 1e-5;
  /// @brief Borrow current stores/BN settings during architecture construction.
  auto conv = [&](const std::string &name, const nn::Tensor &h, int out, int kernel,
                  int stride = 1) {
    return ResNetConv(p, state, name, h, out, kernel, norm, stride);
  };
  nn::Tensor h = conv("stem/conv1", images, cfg.stem_width, 3, 2);
  h = conv("stem/conv2", h, cfg.stem_width, 3, 2);
  int in = cfg.stem_width;
  for (int block = 0; block < cfg.stage1_blocks; ++block) {
    h = ResNetBlock(p, state, "stage1/block" + std::to_string(block), h, cfg.stage1_width,
                    norm, 1, true)
            .output;
    in = 4 * cfg.stage1_width;
  }
  std::vector<nn::Tensor> branches = {h};
  std::vector<int> widths = {in};
  for (int stage = 0; stage < 3; ++stage) {
    const int count = stage + 2;
    std::vector<nn::Tensor> next;
    std::vector<int> next_widths;
    for (int branch = 0; branch < count; ++branch) {
      const int width = cfg.width * (1 << branch);
      const auto name =
          "transition" + std::to_string(stage + 1) + "/branch" + std::to_string(branch);
      if (branch < int(branches.size()))
        next.push_back(width == widths[std::size_t(branch)]
                           ? branches[std::size_t(branch)]
                           : conv(name, branches[std::size_t(branch)], width, 3));
      else
        next.push_back(conv(name, branches.back(), width, 3, 2));
      next_widths.push_back(width);
    }
    branches = next;
    widths = next_widths;
    for (int module = 0; module < cfg.modules[std::size_t(stage)]; ++module) {
      const auto prefix =
          "stage" + std::to_string(stage + 2) + "/module" + std::to_string(module);
      for (int branch = 0; branch < count; ++branch)
        for (int block = 0; block < cfg.blocks; ++block)
          branches[std::size_t(branch)] =
              ResNetBlock(p, state,
                          prefix + "/branch" + std::to_string(branch) + "/block" +
                              std::to_string(block),
                          branches[std::size_t(branch)], widths[std::size_t(branch)], norm)
                  .output;
      branches = HRFuse(p, state, prefix + "/fuse", branches, widths, norm);
      ++r.fusion_modules;
    }
  }
  r.branches = branches;
  for (std::size_t k = 1; k < branches.size(); ++k)
    branches[k] = HRResize(branches[k], branches[0]);
  const int channels = 15 * cfg.width;
  h = conv("head/project", nn::concat(branches, 3), channels, 1);
  r.coarse = nn::conv2d(p, "classifier", h, cfg.classes, 1);
  r.logits = HRResize(r.coarse, images);
  return r;
}

} // namespace datamunge::nn::models
