#pragma once
/// @file deeplab_models.hpp
/// @brief Atrous output-stride-controlled ResNet,DeepLab LargeFOV v1 and summed-score ASPP v2.
#include "datamunge/nn/image_ops.hpp"
#include "datamunge/nn/resnet_models.hpp"
#include <array>
#include <limits>
namespace datamunge::nn::models {
namespace nn = datamunge::nn;
/// @brief Apply native dilated no-bias convolution,BN and optional ReLU with symmetric padding.
///
/// Explicit padding is floor(kernel/2)*dilation for non-pointwise kernels.
/// Kernels use He initialization; BN settings are borrowed. No optimizer or
/// fitting. Learned and moving stores cannot be concurrently mutated.
///
/// @param p Borrowed floating learned store,creates/reuses kernel and BN affine leaves.
/// @param state Same-dtype moving BN moments.
/// @param name Unique kernel/BN prefix.
/// @param input Positive floating NHWC matching p dtype.
/// @param out Positive output channels.
/// @param kernel Positive square kernel side.
/// @param cfg BN training,scale,momentum,epsilon and update settings.
/// @param dilation Positive atrous rate;1x1 kernels always use dilation1.
/// @param stride Positive spatial stride,default1.
/// @param relu True(default) applies ReLU after BN.
/// @return Normalized optionally activated feature tensor.
/// @throws std::invalid_argument For invalid settings,geometry,dtype or incompatible learned shapes.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(17);
/// n::models::ResNetNormConfig c;
/// const auto y = n::models::DenseResNetConv(p, state, "conv", n::ones({1, 9, 9, 2}), 4, 3, c, 2);
/// @endcode
/// @par References
/// - Chen,Papandreou,Kokkinos,Murphy and Yuille(2015), Semantic Image Segmentation with Deep Convolutional Nets and Fully Connected CRFs, https://arxiv.org/abs/1412.7062; Chen et al.(2017), DeepLab: Semantic Image Segmentation with Deep Convolutional Nets,Atrous Convolution,and Fully Connected CRFs, https://arxiv.org/abs/1606.00915.
/// @see DenseResNetBlock()
inline nn::Tensor DenseResNetConv(nn::ParameterStore &p, nn::ParameterStore &state,
                                  const std::string &name, const nn::Tensor &input,
                                  std::int64_t out, std::int64_t kernel,
                                  const ResNetNormConfig &cfg, int dilation,
                                  std::int64_t stride = 1, bool relu = true) {
  if (input.dim() != 4 || input.shape()[0] < 1 || input.shape()[1] < 1 ||
      input.shape()[2] < 1 || input.shape()[3] < 1 || input.dtype() != p.dtype() ||
      state.dtype() != p.dtype() || out < 1 || kernel < 1 || dilation < 1 || stride < 1)
    throw std::invalid_argument("invalid dense ResNet convolution");
  const std::int64_t in = input.shape().back();
  const nn::Tensor w = p.get(name + "/W", {kernel, kernel, in, out}, nn::init::he_uniform());
  nn::Conv2DOptions o =
      nn::conv_options(stride, nn::Padding::Valid, kernel > 1 ? dilation : 1);
  if (kernel > 1) {
    const std::int64_t pad = (kernel / 2) * dilation;
    o.padding = nn::Padding::Explicit;
    o.explicit_paddings = {pad, pad, pad, pad};
  }
  nn::BatchNormOptions bn;
  bn.momentum = cfg.momentum;
  bn.eps = cfg.epsilon;
  bn.scale = cfg.scale;
  bn.update_stats = cfg.update_stats;
  nn::Tensor y =
      nn::batch_norm(p, state, name + "/bn", nn::conv2d(input, w, o), cfg.training, bn);
  return relu ? nn::relu(y) : y;
}

/// @brief Build a basic or bottleneck dilated residual unit with a projection when needed.
///
/// Final branch and shortcut projections are unactivated before addition.
/// No hidden optimizer or fitting; parameter names match the source dense model.
///
/// @param p Learned store.
/// @param state Moving BN store.
/// @param name Unique unit prefix.
/// @param input Positive same-dtype NHWC features.
/// @param width Positive basic/bottleneck inner width; bottleneck output is4*width.
/// @param cfg Borrowed normalization settings.
/// @param dilation Positive3x3 atrous rate.
/// @param stride Positive first-unit stride.
/// @param bottleneck True uses1x1/3x3/1x1;false uses two3x3 layers.
/// @param stride_on_3x3 True places bottleneck stride on3x3;false on initial1x1.
/// @return ReLU of residual branch plus identity/projected shortcut.
/// @throws std::invalid_argument For invalid settings,geometry,dtype or incompatible learned shapes.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(17);
/// n::models::ResNetNormConfig c;
/// const auto y = n::models::DenseResNetBlock(p, state, "block", n::ones({1, 9, 9, 2}), 4, c, 2,
///                                            1, false, true);
/// @endcode
/// @par References
/// - Chen,Papandreou,Kokkinos,Murphy and Yuille(2015), Semantic Image Segmentation with Deep Convolutional Nets and Fully Connected CRFs, https://arxiv.org/abs/1412.7062; Chen et al.(2017), DeepLab: Semantic Image Segmentation with Deep Convolutional Nets,Atrous Convolution,and Fully Connected CRFs, https://arxiv.org/abs/1606.00915.
/// @see DenseResNet()
inline nn::Tensor DenseResNetBlock(nn::ParameterStore &p, nn::ParameterStore &state,
                                   const std::string &name, const nn::Tensor &input,
                                   std::int64_t width, const ResNetNormConfig &cfg,
                                   int dilation, int stride, bool bottleneck,
                                   bool stride_on_3x3) {
  const std::int64_t in = input.shape().back();
  const std::int64_t out = width * (bottleneck ? 4 : 1);
  /// @brief Borrow stores,normalization and dilation only during this residual forward.
  auto conv = [&](const std::string &part, const nn::Tensor &x, std::int64_t co,
                  std::int64_t k, int step, bool relu) {
    return DenseResNetConv(p, state, name + "/" + part, x, co, k, cfg, dilation, step, relu);
  };
  nn::Tensor h;
  if (bottleneck) {
    h = conv("conv1", input, width, 1, stride_on_3x3 ? 1 : stride, true);
    h = conv("conv2", h, width, 3, stride_on_3x3 ? stride : 1, true);
    h = conv("conv3", h, out, 1, 1, false);
  } else {
    h = conv("conv1", input, width, 3, stride, true);
    h = conv("conv2", h, out, 3, 1, false);
  }
  const nn::Tensor shortcut =
      in == out && stride == 1 ? input : conv("shortcut", input, out, 1, stride, false);
  return nn::relu(h + shortcut);
}

/// @brief Build source ResNet with controlled8/16/32 output stride and native atrous stages.
///
/// Once running stride reaches output_stride,later stages suppress their stride
/// and double dilation. Stage-first units use current dilation consistently;
/// 3x3 layers pad symmetrically. Small stem starts at stride1; standard stem
/// starts at4. No optimizer or pretrained weights. Checkpoint learned/moving
/// state together,and keep learned leaves unchanged until backward.
///
/// @param p Floating learned store.
/// @param state Same-dtype moving BN store.
/// @param image Positive same-dtype RGB NHWC.
/// @param cfg Supported ResNet depth/width/stem/BN settings; optional classifier honored.
/// @param output_stride Requested nominal encoder stride8,16 or32.
/// @param training Use batch moments/update policy when true;moving moments when false.
/// @return Stem,four stage maps,every residual output and optional classifier logits.
/// @throws std::invalid_argument For invalid settings,geometry,dtype or incompatible learned shapes.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(17);
/// n::models::ResNetConfig c;
/// c.depth = 18;
/// c.width_divisor = 8;
/// c.include_classifier = false;
/// const auto net = n::models::DenseResNet(p, state, n::ones({1, 33, 33, 3}), c, 8, false);
/// @endcode
/// @par References
/// - Chen,Papandreou,Kokkinos,Murphy and Yuille(2015), Semantic Image Segmentation with Deep Convolutional Nets and Fully Connected CRFs, https://arxiv.org/abs/1412.7062; Chen et al.(2017), DeepLab: Semantic Image Segmentation with Deep Convolutional Nets,Atrous Convolution,and Fully Connected CRFs, https://arxiv.org/abs/1606.00915.
/// @see ResNet(), DenseResNetBlock()
inline ResNetOutputs DenseResNet(nn::ParameterStore &p, nn::ParameterStore &state,
                                 const nn::Tensor &image, const ResNetConfig &cfg,
                                 int output_stride, bool training) {
  cfg.Validate();
  if (image.dim() != 4 || image.shape()[0] < 1 || image.shape()[1] < 1 ||
      image.shape()[2] < 1 || image.shape()[3] != 3 || image.dtype() != p.dtype() ||
      state.dtype() != p.dtype())
    throw std::invalid_argument("invalid dense ResNet image");
  if (output_stride != 8 && output_stride != 16 && output_stride != 32)
    throw std::invalid_argument("invalid ResNet configuration");
  ResNetOutputs out;
  ResNetNormConfig norm;
  norm.training = training;
  norm.scale = true;
  norm.momentum = cfg.momentum;
  norm.epsilon = 1e-5;
  norm.update_stats = cfg.update_stats;
  auto h = DenseResNetConv(p, state, "stem", image, 64 / cfg.width_divisor,
                           cfg.small_stem ? 3 : 7, norm, 1, cfg.small_stem ? 1 : 2);
  if (!cfg.small_stem)
    h = nn::max_pool2d(nn::pad(h, {{0, 0}, {1, 1}, {1, 1}, {0, 0}}, nn::PadMode::Constant,
                               -std::numeric_limits<double>::infinity()),
                       3, 2, nn::Padding::Valid);
  out.stem = h;
  const bool bottleneck = cfg.depth >= 50;
  const auto counts = ResNetBlockCounts(cfg.depth);
  int current = cfg.small_stem ? 1 : 4, dilation = 1;
  for (int stage = 0; stage < 4; ++stage) {
    int stage_stride = stage ? 2 : 1;
    if (stage && current >= output_stride) {
      stage_stride = 1;
      dilation *= 2;
    } else
      current *= stage_stride;
    const std::int64_t width = (64 << stage) / cfg.width_divisor;
    for (int block = 0; block < counts[std::size_t(stage)]; ++block) {
      const auto name =
          "stage" + std::to_string(stage + 1) + "/block" + std::to_string(block + 1);
      h = DenseResNetBlock(p, state, name, h, width, norm, dilation,
                           block == 0 ? stage_stride : 1, bottleneck, cfg.stride_on_3x3);
      out.blocks.push_back(h);
    }
    out.stages[std::size_t(stage)] = h;
  }
  if (cfg.include_classifier)
    out.logits = nn::dense(p, "output", nn::global_avg_pool2d(h), cfg.classes);
  return out;
}

/// @brief Apply a biased He-initialized native atrous SAME convolution with optional ReLU.
///
/// Native dilation differentiates inputs/kernels directly; no expanded zero
/// filter is needed. Bias starts at0. No optimizer,BN state or fitting.
///
/// @param p Same-dtype learned store.
/// @param name Unique weight/bias prefix.
/// @param input Positive floating NHWC features.
/// @param out Positive output channels.
/// @param kernel Positive square kernel size.
/// @param rate Positive atrous rate.
/// @param relu True(default) applies ReLU;false returns linear scores.
/// @return Same-spatial-size atrous features/scores.
/// @throws std::invalid_argument For invalid settings,geometry,dtype or incompatible learned shapes.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17);
/// const auto y = n::models::AtrousLayer(p, "atrous", n::ones({1, 9, 9, 2}), 4, 3, 2);
/// @endcode
/// @par References
/// - Chen,Papandreou,Kokkinos,Murphy and Yuille(2015), Semantic Image Segmentation with Deep Convolutional Nets and Fully Connected CRFs, https://arxiv.org/abs/1412.7062; Chen et al.(2017), DeepLab: Semantic Image Segmentation with Deep Convolutional Nets,Atrous Convolution,and Fully Connected CRFs, https://arxiv.org/abs/1606.00915.
/// @see AtrousScorePyramid()
inline nn::Tensor AtrousLayer(nn::ParameterStore &p, const std::string &name,
                              const nn::Tensor &input, std::int64_t out, std::int64_t kernel,
                              int rate, bool relu = true) {
  if (input.dim() != 4 || input.shape()[0] < 1 || input.shape()[1] < 1 ||
      input.shape()[2] < 1 || input.shape()[3] < 1 || input.dtype() != p.dtype() || out < 1 ||
      kernel < 1 || rate < 1)
    throw std::invalid_argument("invalid atrous layer");
  const std::int64_t in = input.shape().back();
  const nn::Tensor w = p.get(name + "/W", {kernel, kernel, in, out}, nn::init::he_uniform());
  const nn::Tensor b = p.get(name + "/b", {out}, nn::init::zeros());
  const nn::Tensor y =
      nn::bias_add(nn::conv2d(input, w, nn::conv_options(1, nn::Padding::Same, rate)), b);
  return relu ? nn::relu(y) : y;
}

/// @brief Own DeepLab features,coarse/final logits and optional ASPP score branches.
///
/// No independent learned state or fitting; returned tensors retain graphs
/// borrowing parameter leaves until backward.
///
/// @throws std::invalid_argument For invalid settings,geometry,dtype or incompatible learned shapes.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17);
/// n::models::DeepLabV1Config c;
/// c.classes = 4;
/// c.width_divisor = 8;
/// c.fc_width = 128;
/// c.fc_rate = 4;
/// c.keep_probability = 1;
/// const auto net = n::models::DeepLabV1(p, n::ones({1, 33, 33, 3}), c);
/// @endcode
/// @par References
/// - Chen,Papandreou,Kokkinos,Murphy and Yuille(2015), Semantic Image Segmentation with Deep Convolutional Nets and Fully Connected CRFs, https://arxiv.org/abs/1412.7062; Chen et al.(2017), DeepLab: Semantic Image Segmentation with Deep Convolutional Nets,Atrous Convolution,and Fully Connected CRFs, https://arxiv.org/abs/1606.00915.
/// @see DenseResNet(), DeepLabV1(), DeepLabV2()
struct DeepLabOutputs {
  nn::Tensor features; ///< Deepest dense VGG/ResNet feature map.
  nn::Tensor coarse;   ///< Coarse class logits before corner-aligned resize.
  nn::Tensor logits;   ///< Linear NHWC logits at exact input H/W.
  std::vector<nn::Tensor>
      branches; ///< V2 four linear atrous scores,in requested rate order;empty for v1.
};
/// @brief Build four independent linear atrous score branches for summed-score ASPP.
///
/// Each3x3 branch has its own He kernel/bias and no activation. This v2
/// pyramid sums class scores rather than concatenating normalized features.
/// No fitting,BN state or random augmentation.
///
/// @param p Same-dtype learned store.
/// @param name Unique pyramid prefix.
/// @param feature Positive floating NHWC features.
/// @param classes Positive output channels.
/// @param rates Four positive distinct atrous rates,each used in its parameter name.
/// @return Four same-spatial class-score tensors in requested rate order.
/// @throws std::invalid_argument For invalid settings,geometry,dtype or incompatible learned shapes.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17);
/// const auto scores =
///     n::models::AtrousScorePyramid(p, "aspp", n::ones({1, 9, 9, 2}), 4, {1, 2, 3, 4});
/// @endcode
/// @par References
/// - Chen,Papandreou,Kokkinos,Murphy and Yuille(2015), Semantic Image Segmentation with Deep Convolutional Nets and Fully Connected CRFs, https://arxiv.org/abs/1412.7062; Chen et al.(2017), DeepLab: Semantic Image Segmentation with Deep Convolutional Nets,Atrous Convolution,and Fully Connected CRFs, https://arxiv.org/abs/1606.00915.
/// @see DeepLabV2()
inline std::vector<nn::Tensor>
AtrousScorePyramid(nn::ParameterStore &p, const std::string &name, const nn::Tensor &feature,
                   std::int64_t classes, const std::array<int, 4> &rates) {
  for (std::size_t i = 0; i < rates.size(); ++i)
    for (std::size_t j = 0; j < i; ++j)
      if (rates[i] == rates[j])
        throw std::invalid_argument("ASPP rates require distinct parameter names");
  std::vector<nn::Tensor> outputs;
  for (int rate : rates) {
    if (rate < 1)
      throw std::invalid_argument("ASPP rates must be positive");
    outputs.push_back(AtrousLayer(p, name + "/rate" + std::to_string(rate), feature, classes,
                                  3, rate, false));
  }
  return outputs;
}

/// @brief Resize features bilinearly with aligned corners to another tensor spatial shape.
///
/// No gradient to reference values. Single-pixel source dimensions broadcast
/// with the library aligned-corner convention. No fitting or mutation.
///
/// @param x Positive floating NHWC source.
/// @param like Rank4 tensor supplying positive target H/W;values unused.
/// @return Resized source features preserving source gradients.
/// @throws std::invalid_argument For invalid settings,geometry,dtype or incompatible learned shapes.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto y = n::models::DeepLabResizeLike(n::ones({1, 3, 3, 2}), n::zeros({1, 9, 9, 1}));
/// @endcode
/// @par References
/// - Chen,Papandreou,Kokkinos,Murphy and Yuille(2015), Semantic Image Segmentation with Deep Convolutional Nets and Fully Connected CRFs, https://arxiv.org/abs/1412.7062; Chen et al.(2017), DeepLab: Semantic Image Segmentation with Deep Convolutional Nets,Atrous Convolution,and Fully Connected CRFs, https://arxiv.org/abs/1606.00915.
/// @see DeepLabV1(), DeepLabV2()
inline nn::Tensor DeepLabResizeLike(const nn::Tensor &x, const nn::Tensor &like) {
  if (x.dim() != 4 || like.dim() != 4 || like.shape()[1] < 1 || like.shape()[2] < 1)
    throw std::invalid_argument("invalid DeepLab resize reference");
  return nn::resize_bilinear(x, {like.shape()[1], like.shape()[2]}, true);
}

/// @brief Configure VGG LargeFOV DeepLab v1.
///
/// Defaults21 classes,VGG widths unchanged,fc1024/rate12,dropout keep.5.
/// No BN or pretrained weights;compact settings are explicit in the example.
///
/// @throws std::invalid_argument For invalid settings,geometry,dtype or incompatible learned shapes.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17);
/// n::models::DeepLabV1Config c;
/// c.classes = 4;
/// c.width_divisor = 8;
/// c.fc_width = 128;
/// c.fc_rate = 4;
/// c.keep_probability = 1;
/// const auto net = n::models::DeepLabV1(p, n::ones({1, 33, 33, 3}), c);
/// @endcode
/// @par References
/// - Chen,Papandreou,Kokkinos,Murphy and Yuille(2015), Semantic Image Segmentation with Deep Convolutional Nets and Fully Connected CRFs, https://arxiv.org/abs/1412.7062; Chen et al.(2017), DeepLab: Semantic Image Segmentation with Deep Convolutional Nets,Atrous Convolution,and Fully Connected CRFs, https://arxiv.org/abs/1606.00915.
/// @see DenseResNet(), DeepLabV1(), DeepLabV2()
struct DeepLabV1Config {
  int classes = 21;             ///< At least two classes.
  int width_divisor = 1;        ///< Positive exact divisor of64 for every VGG width.
  int fc_width = 1024;          ///< Positive fc6/fc7 channel width.
  int fc_rate = 12;             ///< Positive3x3 fc6 dilation.
  float keep_probability = .5f; ///< Training fc6/fc7 dropout keep probability in(0,1].
};
/// @brief Build output-stride8 VGG LargeFOV with native atrous conv5/fc6.
///
/// First three pools stride2,later pools stride1;conv5 rate2 and fc6 rate12
/// by default. Full count20505429,no BN state,32 learned tensors. Native
/// bilinear aligned-corner resizing restores exact odd/even input shapes.
/// No optimizer or pretrained weights. Warm leaves before optimizer creation.
///
/// @param p Same-dtype learned store,creates/reuses VGG/fc/score leaves.
/// @param image Positive floating RGB NHWC matching p dtype.
/// @param cfg Valid classes,width,fc,dilation and dropout settings.
/// @param training False(default) disables dropout;true draws through p.generator().
/// @return Dense features,coarse logits and input-aligned logits.
/// @throws std::invalid_argument For invalid settings,geometry,dtype or incompatible learned shapes.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17);
/// n::models::DeepLabV1Config c;
/// c.classes = 4;
/// c.width_divisor = 8;
/// c.fc_width = 128;
/// c.fc_rate = 4;
/// c.keep_probability = 1;
/// const auto net = n::models::DeepLabV1(p, n::ones({1, 33, 33, 3}), c);
/// @endcode
/// @par References
/// - Chen,Papandreou,Kokkinos,Murphy and Yuille(2015), Semantic Image Segmentation with Deep Convolutional Nets and Fully Connected CRFs, https://arxiv.org/abs/1412.7062; Chen et al.(2017), DeepLab: Semantic Image Segmentation with Deep Convolutional Nets,Atrous Convolution,and Fully Connected CRFs, https://arxiv.org/abs/1606.00915.
/// @see AtrousLayer(), DeepLabV2()
inline DeepLabOutputs DeepLabV1(nn::ParameterStore &p, const nn::Tensor &image,
                                const DeepLabV1Config &cfg = {}, bool training = false) {
  if (cfg.classes < 2 || cfg.width_divisor < 1 || 64 % cfg.width_divisor || cfg.fc_width < 1 ||
      cfg.fc_rate < 1 || !(cfg.keep_probability > 0 && cfg.keep_probability <= 1) ||
      image.dim() != 4 || image.shape()[0] < 1 || image.shape()[1] < 1 ||
      image.shape()[2] < 1 || image.shape()[3] != 3 || image.dtype() != p.dtype())
    throw std::invalid_argument("invalid DeepLab v1 configuration");
  DeepLabOutputs r;
  nn::Tensor h = image;
  const int repeats[] = {2, 2, 3, 3, 3};
  for (int stage = 0; stage < 5; ++stage) {
    const std::int64_t width = (64 << std::min(stage, 3)) / cfg.width_divisor;
    for (int block = 0; block < repeats[stage]; ++block)
      h = AtrousLayer(p, "conv" + std::to_string(stage + 1) + "_" + std::to_string(block + 1),
                      h, width, 3, stage == 4 ? 2 : 1);
    h = nn::max_pool2d(h, 3, stage < 3 ? 2 : 1, nn::Padding::Same);
  }
  r.features = h;
  h = AtrousLayer(p, "fc6", h, cfg.fc_width, 3, cfg.fc_rate);
  if (training && cfg.keep_probability < 1)
    h = nn::dropout(h, 1 - cfg.keep_probability, true, p.generator());
  h = AtrousLayer(p, "fc7", h, cfg.fc_width, 1, 1);
  if (training && cfg.keep_probability < 1)
    h = nn::dropout(h, 1 - cfg.keep_probability, true, p.generator());
  r.coarse = AtrousLayer(p, "score", h, cfg.classes, 1, 1, false);
  r.logits = DeepLabResizeLike(r.coarse, image);
  return r;
}

/// @brief Configure dense ResNet DeepLab v2 and summed class-score ASPP.
///
/// Default ResNet101,21 classes,output stride8 and rates6/12/18/24.
/// BN follows the dense ResNet;score branches are biased linear convolutions.
///
/// @throws std::invalid_argument For invalid settings,geometry,dtype or incompatible learned shapes.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(17);
/// n::models::DeepLabV2Config c;
/// c.classes = 4;
/// c.backbone_depth = 18;
/// c.width_divisor = 8;
/// c.rates = {1, 2, 3, 4};
/// const auto net = n::models::DeepLabV2(p, state, n::ones({1, 33, 33, 3}), c);
/// @endcode
/// @par References
/// - Chen,Papandreou,Kokkinos,Murphy and Yuille(2015), Semantic Image Segmentation with Deep Convolutional Nets and Fully Connected CRFs, https://arxiv.org/abs/1412.7062; Chen et al.(2017), DeepLab: Semantic Image Segmentation with Deep Convolutional Nets,Atrous Convolution,and Fully Connected CRFs, https://arxiv.org/abs/1606.00915.
/// @see DenseResNet(), DeepLabV1(), DeepLabV2()
struct DeepLabV2Config {
  int classes = 21;                           ///< At least two output classes.
  int backbone_depth = 101;                   ///< Supported ResNet depth18/34/50/101/152.
  int width_divisor = 1;                      ///< Positive divisor of64 for backbone widths.
  std::array<int, 4> rates = {6, 12, 18, 24}; ///< Distinct positive class-score atrous rates.
};
/// @brief Build output-stride8 dense ResNet and summed four-rate ASPP class scores.
///
/// Bottleneck stage strides sit on3x3 layers. Later strides are suppressed
/// and dilation doubles to preserve stride8 features. Full101/21 count44048532
/// and105344 moving scalars. No optimizer or pretrained weights. Warm leaves
/// before fitting,and checkpoint learned/BN stores together.
///
/// @param p Same-dtype learned store.
/// @param state Same-dtype BN moving state,updated during training.
/// @param image Positive floating RGB NHWC matching stores.
/// @param cfg Supported depth/width/classes and four distinct positive rates.
/// @param training False(default) moving moments;true batch moments and moving updates.
/// @return Dense features,four atrous branches,their sum and input-aligned logits.
/// @throws std::invalid_argument For invalid settings,geometry,dtype or incompatible learned shapes.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(17);
/// n::models::DeepLabV2Config c;
/// c.classes = 4;
/// c.backbone_depth = 18;
/// c.width_divisor = 8;
/// c.rates = {1, 2, 3, 4};
/// const auto net = n::models::DeepLabV2(p, state, n::ones({1, 33, 33, 3}), c);
/// @endcode
/// @par References
/// - Chen,Papandreou,Kokkinos,Murphy and Yuille(2015), Semantic Image Segmentation with Deep Convolutional Nets and Fully Connected CRFs, https://arxiv.org/abs/1412.7062; Chen et al.(2017), DeepLab: Semantic Image Segmentation with Deep Convolutional Nets,Atrous Convolution,and Fully Connected CRFs, https://arxiv.org/abs/1606.00915.
/// @see DenseResNet(), AtrousScorePyramid()
inline DeepLabOutputs DeepLabV2(nn::ParameterStore &p, nn::ParameterStore &state,
                                const nn::Tensor &image, const DeepLabV2Config &cfg = {},
                                bool training = false) {
  if (cfg.classes < 2)
    throw std::invalid_argument("DeepLab v2 requires at least two classes");
  ResNetConfig backbone;
  backbone.depth = cfg.backbone_depth;
  backbone.width_divisor = cfg.width_divisor;
  backbone.include_classifier = false;
  backbone.stride_on_3x3 = true;
  const auto net = DenseResNet(p, state, image, backbone, 8, training);
  DeepLabOutputs r;
  r.features = net.stages[3];
  r.branches = AtrousScorePyramid(p, "aspp", r.features, cfg.classes, cfg.rates);
  r.coarse = r.branches[0];
  for (std::size_t i = 1; i < r.branches.size(); ++i)
    r.coarse = r.coarse + r.branches[i];
  r.logits = DeepLabResizeLike(r.coarse, image);
  return r;
}

} // namespace datamunge::nn::models
