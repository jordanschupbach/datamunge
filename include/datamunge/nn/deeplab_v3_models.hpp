#pragma once
/// @file deeplab_v3_models.hpp
/// @brief Five-branch DeepLab v3 ASPP and v3+ separable atrous/low-level decoder.
#include "datamunge/nn/deeplab_models.hpp"
#include "datamunge/nn/vision_architecture_components.hpp"
namespace datamunge::nn::models {
namespace nn = datamunge::nn;
/// @brief Apply atrous depthwise convolution,BN/ReLU,and a pointwise normalized projection.
///
/// Depthwise kernel uses U(+-sqrt(6/9)),one-channel He fan convention.
/// Depthwise BN always learns gamma;pointwise BN follows cfg.scale. No hidden
/// fitting or pretrained weights.
///
/// @param p Same-dtype learned store.
/// @param state Moving BN store.
/// @param name Unique depthwise/pointwise prefix.
/// @param input Positive same-dtype floating NHWC.
/// @param out Positive output channels.
/// @param rate Positive3x3 depthwise dilation.
/// @param cfg Borrowed BN settings;pointwise dilation is forced1.
/// @return Same-spatial-size separable feature output.
/// @throws std::invalid_argument For invalid settings,geometry,dtype or incompatible learned shapes.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(17);
/// n::models::ConvNormConfig c;
/// c.scale = true;
/// const auto y =
///     n::models::AtrousSeparableBlock(p, state, "sep", n::ones({1, 7, 7, 2}), 4, 2, c);
/// @endcode
/// @par References
/// - Chen,Papandreou,Schroff and Adam(2017), Rethinking Atrous Convolution for Semantic Image Segmentation, https://arxiv.org/abs/1706.05587; Chen,Zhu,Papandreou,Schroff and Adam(2018), Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation, https://arxiv.org/abs/1802.02611. Uses a supported ResNet encoder variant.
/// @see DeepLabV3()
inline nn::Tensor AtrousSeparableBlock(nn::ParameterStore &p, nn::ParameterStore &state,
                                       const std::string &name, const nn::Tensor &input,
                                       std::int64_t out, int rate, const ConvNormConfig &cfg) {
  if (input.dim() != 4 || input.shape()[0] < 1 || input.shape()[1] < 1 ||
      input.shape()[2] < 1 || input.shape()[3] < 1 || input.dtype() != p.dtype() ||
      state.dtype() != p.dtype() || out < 1 || rate < 1)
    throw std::invalid_argument("invalid separable atrous block");
  const std::int64_t in = input.shape().back();
  // tflow passes fans 9/9 (one depthwise channel), He uniform: U(+-sqrt(6/9)).
  const nn::Tensor kernel =
      p.get(name + "/depthwise/W", {3, 3, in, 1}, nn::init::uniform(std::sqrt(6.0 / 9.0)));
  const nn::Tensor depthwise =
      nn::depthwise_conv2d(input, kernel, nn::conv_options(1, nn::Padding::Same, rate));
  nn::BatchNormOptions bn;
  bn.momentum = cfg.momentum;
  bn.eps = cfg.epsilon;
  bn.scale = true;
  bn.update_stats = cfg.update_stats;
  const nn::Tensor normalized =
      nn::batch_norm(p, state, name + "/depthwise/bn", depthwise, cfg.training, bn);
  auto pointwise = cfg;
  pointwise.dilation = 1;
  return NormalizedConv(p, state, name + "/pointwise", nn::relu(normalized), out, 1, 1,
                        pointwise);
}

/// @brief Project global image mean and broadcast it by aligned-corner bilinear resizing.
///
/// Input spatial mean preserves gradients. In training,N=1 and1x1 BN has
/// zero batch variance;initialized gamma/input derivatives may be zero.
/// No optimizer or fitting. Shared stores cannot mutate concurrently.
///
/// @param p Same-dtype learned projection store.
/// @param state Moving BN state.
/// @param name Unique projection prefix.
/// @param input Positive same-dtype floating NHWC features.
/// @param out Positive output channels.
/// @param cfg Borrowed normalization settings;projection dilation forced1.
/// @return Image-sized normalized/ReLU global context feature map.
/// @throws std::invalid_argument For invalid settings,geometry,dtype or incompatible learned shapes.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(17);
/// n::models::ConvNormConfig c;
/// c.scale = true;
/// const auto y = n::models::ImagePoolBranch(p, state, "context", n::ones({2, 7, 9, 2}), 4, c);
/// @endcode
/// @par References
/// - Chen,Papandreou,Schroff and Adam(2017), Rethinking Atrous Convolution for Semantic Image Segmentation, https://arxiv.org/abs/1706.05587; Chen,Zhu,Papandreou,Schroff and Adam(2018), Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation, https://arxiv.org/abs/1802.02611. Uses a supported ResNet encoder variant.
/// @see DeepLabV3()
inline nn::Tensor ImagePoolBranch(nn::ParameterStore &p, nn::ParameterStore &state,
                                  const std::string &name, const nn::Tensor &input,
                                  std::int64_t out, const ConvNormConfig &cfg) {
  auto one = cfg;
  one.dilation = 1;
  const auto pooled = nn::mean(input, {1, 2}, true);
  const auto projected = NormalizedConv(p, state, name + "/project", pooled, out, 1, 1, one);
  return DeepLabResizeLike(projected, input);
}

/// @brief Configure ResNet-based DeepLab v3 or a separable v3+ decoder.
///
/// Defaults ResNet50/output stride16,head256,rates6/12/18,keep.9.
/// Decoder false(default) uses ordinary atrous ASPP;true enables separable
/// ASPP and low-level C2 fusion. No pretrained weights.
///
/// @throws std::invalid_argument For invalid settings,geometry,dtype or incompatible learned shapes.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(17);
/// n::models::DeepLabV3Config c;
/// c.classes = 4;
/// c.backbone_depth = 18;
/// c.width_divisor = 8;
/// c.head_width = 32;
/// c.decoder_width = 32;
/// c.low_width = 8;
/// c.rates = {1, 2, 3};
/// c.keep_probability = 1;
/// const auto net = n::models::DeepLabV3(p, state, n::ones({1, 33, 33, 3}), c);
/// @endcode
/// @par References
/// - Chen,Papandreou,Schroff and Adam(2017), Rethinking Atrous Convolution for Semantic Image Segmentation, https://arxiv.org/abs/1706.05587; Chen,Zhu,Papandreou,Schroff and Adam(2018), Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation, https://arxiv.org/abs/1802.02611. Uses a supported ResNet encoder variant.
/// @see
struct DeepLabV3Config {
  int classes = 21;        ///< At least two class logits.
  int backbone_depth = 50; ///< Supported ResNet depth18/34/50/101/152.
  int width_divisor = 1;   ///< Positive exact divisor of64 for backbone widths.
  int head_width = 256;    ///< Positive ASPP branch/projection width.
  int decoder_width = 256; ///< Positive v3+ decoder width.
  int low_width = 48;      ///< Positive v3+ C2 projection width.
  int output_stride = 16;  ///< Dense encoder stride8 or16.
  bool decoder = false;    ///< True enables separable v3+ ASPP and low-level decoder.
  std::array<int, 3> rates = {6, 12,
                              18}; ///< Positive distinct base rates;double at output stride8.
  float keep_probability = .9f; ///< ASPP projection training dropout keep probability in(0,1].
};
/// @brief Own DeepLab v3/v3+ context branches,low-level projection and logits.
///
/// Returned graphs borrow learned leaves;moving state belongs to the caller.
///
/// @throws std::invalid_argument For invalid settings,geometry,dtype or incompatible learned shapes.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(17);
/// n::models::DeepLabV3Config c;
/// c.classes = 4;
/// c.backbone_depth = 18;
/// c.width_divisor = 8;
/// c.head_width = 32;
/// c.decoder_width = 32;
/// c.low_width = 8;
/// c.rates = {1, 2, 3};
/// c.keep_probability = 1;
/// const auto net = n::models::DeepLabV3(p, state, n::ones({1, 33, 33, 3}), c);
/// @endcode
/// @par References
/// - Chen,Papandreou,Schroff and Adam(2017), Rethinking Atrous Convolution for Semantic Image Segmentation, https://arxiv.org/abs/1706.05587; Chen,Zhu,Papandreou,Schroff and Adam(2018), Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation, https://arxiv.org/abs/1802.02611. Uses a supported ResNet encoder variant.
/// @see
struct DeepLabV3Outputs {
  nn::Tensor features; ///< Deepest dense ResNet map.
  nn::Tensor low;      ///< V3+ projected C2 map;undefined without decoder.
  nn::Tensor coarse;   ///< Linear classifier logits before final resize.
  nn::Tensor logits;   ///< NHWC logits at exact original image H/W.
  std::vector<nn::Tensor>
      branches; ///< Pointwise,three atrous,image-global branches,in concatenation order.
};
/// @brief Build five-branch ASPP with optional separable low-level v3+ decoding.
///
/// ASPP concatenates pointwise,three atrous and image-global features,then
/// projects1x1 and optionally drops out. V3+ projects C2,fuses at stride4 and
/// applies two separable3x3 blocks;class scores remain linear. Full50/21
/// v3/v3+ learned counts39048533/26696133,state56192/70720. No optimizer,
/// pretraining or benchmark implication. Warm all leaves before optimizer
/// creation;checkpoint learned and moving stores together.
///
/// @param p Same-dtype floating learned store,creates/reuses all encoder/head leaves.
/// @param state Same-dtype BN moving state,updated during training.
/// @param images Positive floating RGB NHWC matching stores.
/// @param cfg Valid architecture/rates/dropout settings.
/// @param training False(default) moving moments/no dropout;true batch moments/updates/dropout.
/// @return Dense features,five context branches,optional low map and input-aligned logits.
/// @throws std::invalid_argument For invalid settings,geometry,dtype or incompatible learned shapes.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(17);
/// n::models::DeepLabV3Config c;
/// c.classes = 4;
/// c.backbone_depth = 18;
/// c.width_divisor = 8;
/// c.head_width = 32;
/// c.decoder_width = 32;
/// c.low_width = 8;
/// c.rates = {1, 2, 3};
/// c.keep_probability = 1;
/// const auto net = n::models::DeepLabV3(p, state, n::ones({1, 33, 33, 3}), c);
/// @endcode
/// @par References
/// - Chen,Papandreou,Schroff and Adam(2017), Rethinking Atrous Convolution for Semantic Image Segmentation, https://arxiv.org/abs/1706.05587; Chen,Zhu,Papandreou,Schroff and Adam(2018), Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation, https://arxiv.org/abs/1802.02611. Uses a supported ResNet encoder variant.
/// @see DenseResNet(), AtrousSeparableBlock(), ImagePoolBranch()
inline DeepLabV3Outputs DeepLabV3(nn::ParameterStore &p, nn::ParameterStore &state,
                                  const nn::Tensor &images, const DeepLabV3Config &cfg = {},
                                  bool training = false) {
  if (cfg.classes < 2 || cfg.head_width < 1 || cfg.decoder_width < 1 || cfg.low_width < 1 ||
      (cfg.output_stride != 8 && cfg.output_stride != 16) ||
      !(cfg.keep_probability > 0 && cfg.keep_probability <= 1))
    throw std::invalid_argument("invalid DeepLab v3 configuration");
  for (std::size_t i = 0; i < cfg.rates.size(); ++i) {
    if (cfg.rates[i] < 1 ||
        cfg.rates[i] > std::numeric_limits<int>::max() / (16 / cfg.output_stride))
      throw std::invalid_argument("invalid ASPP rate");
    for (std::size_t j = 0; j < i; ++j)
      if (cfg.rates[i] == cfg.rates[j])
        throw std::invalid_argument("ASPP rates must be distinct");
  }
  ResNetConfig backbone;
  backbone.depth = cfg.backbone_depth;
  backbone.width_divisor = cfg.width_divisor;
  backbone.include_classifier = false;
  backbone.stride_on_3x3 = true;
  const auto net = DenseResNet(p, state, images, backbone, cfg.output_stride, training);
  DeepLabV3Outputs result;
  result.features = net.stages[3];
  ConvNormConfig norm;
  norm.training = training;
  norm.scale = true;
  norm.momentum = .9;
  norm.epsilon = 1e-5;
  result.branches.push_back(
      NormalizedConv(p, state, "aspp/pointwise", result.features, cfg.head_width, 1, 1, norm));
  for (int base_rate : cfg.rates) {
    const int rate = base_rate * (16 / cfg.output_stride);
    if (rate < 1)
      throw std::invalid_argument("ASPP rate must be positive");
    const auto name = "aspp/rate" + std::to_string(base_rate);
    if (cfg.decoder)
      result.branches.push_back(
          AtrousSeparableBlock(p, state, name, result.features, cfg.head_width, rate, norm));
    else {
      auto atrous = norm;
      atrous.dilation = rate;
      result.branches.push_back(
          NormalizedConv(p, state, name, result.features, cfg.head_width, 3, 3, atrous));
    }
  }
  result.branches.push_back(
      ImagePoolBranch(p, state, "aspp/image_pool", result.features, cfg.head_width, norm));
  nn::Tensor h = NormalizedConv(p, state, "aspp/project", nn::concat(result.branches, 3),
                                cfg.head_width, 1, 1, norm);
  if (training && cfg.keep_probability < 1)
    h = nn::dropout(h, 1 - cfg.keep_probability, true, p.generator());
  if (cfg.decoder) {
    result.low =
        NormalizedConv(p, state, "decoder/low", net.stages[0], cfg.low_width, 1, 1, norm);
    const auto resized = DeepLabResizeLike(h, result.low);
    h = AtrousSeparableBlock(p, state, "decoder/conv1", nn::concat({resized, result.low}, 3),
                             cfg.decoder_width, 1, norm);
    h = AtrousSeparableBlock(p, state, "decoder/conv2", h, cfg.decoder_width, 1, norm);
  }
  result.coarse = nn::conv2d(p, "classifier", h, cfg.classes, 1);
  result.logits = DeepLabResizeLike(result.coarse, images);
  return result;
}

} // namespace datamunge::nn::models
