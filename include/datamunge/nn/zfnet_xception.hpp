#pragma once
/// @file zfnet_xception.hpp
/// @brief ZFNet with deconvnet reconstruction, and the full-depth Xception tensor model.
#include "datamunge/nn/alexnet_googlenet.hpp"
#include "datamunge/nn/vision_architecture_components.hpp"
#include <utility>
namespace datamunge::nn::models {
namespace nn = datamunge::nn;
/// @brief Reference or explicitly compact settings for ZFNet.
///
/// Value-only architecture metadata; creating it does not fit or allocate weights.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17);
/// n::models::ZFNetConfig cfg;
/// cfg.image_size = 67;
/// cfg.classes = 3;
/// cfg.width_divisor = 8;
/// cfg.dense_width = 16;
/// cfg.Validate();
/// @endcode
/// @par References
/// - Zeiler and Fergus (2014), https://cs.nyu.edu/~fergus/papers/zeilerECCV2014.pdf
/// @see ZFNet()
struct ZFNetConfig {
  int image_size = 224;         ///< Square RGB input side, at least 67; reference 224.
  int classes = 1000;           ///< At least two output classes.
  int width_divisor = 1;        ///< One, two, four or eight; divides all convolution widths.
  int dense_width = 4096;       ///< Positive width of the two hidden dense layers.
  double keep_probability = .5; ///< Hidden training dropout retention in (0,1].

  /// @brief Validate the architecture dimensions and probability settings.
  ///
  /// @throws std::invalid_argument For settings outside the field domains.
  /// @par Example
  /// @code
  /// dm::nn::models::ZFNetConfig{}.Validate();
  /// @endcode
  /// @see ZFNetConfig
  void Validate() const {
    if (image_size < 67 || classes < 2 || dense_width < 1 ||
        (width_divisor != 1 && width_divisor != 2 && width_divisor != 4 &&
         width_divisor != 8) ||
        !(keep_probability > 0 && keep_probability <= 1))
      throw std::invalid_argument("invalid ZFNet configuration");
  }
  /// @brief Calculate the side after the final VALID stride-two pool.
  ///
  /// @return Positive final map side; six for the reference 224-pixel input.
  /// @throws std::invalid_argument For invalid configuration.
  /// @par Example
  /// @code
  /// const auto side = dm::nn::models::ZFNetConfig{}.FinalSpatialSize();
  /// @endcode
  /// @see ZFNet()
  int FinalSpatialSize() const {
    Validate();
    int size = (image_size + 2 - 7) / 2 + 1;
    size = (size + 1) / 2;
    size = (size - 5) / 2 + 1;
    size = (size + 1) / 2;
    return (size - 3) / 2 + 1;
  }
};

/// @brief Named forward activations and classifier scores for ZFNet.
///
/// Tensor handles retain storage and graphs; fitted parameters remain in p.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17);
/// n::models::ZFNetConfig cfg;
/// cfg.image_size = 67;
/// cfg.classes = 3;
/// cfg.width_divisor = 8;
/// cfg.dense_width = 16;
/// const auto out = n::models::ZFNet(p, n::zeros({1, cfg.image_size, cfg.image_size, 3}), cfg);
/// @endcode
/// @see ZFNet()
struct ZFNetOutputs {
  nn::Tensor image;                      ///< Original NHWC input handle.
  nn::Tensor padded_image;               ///< Input with one zero pixel on each spatial edge.
  std::array<nn::Tensor, 5> convolution; ///< Five post-ReLU maps.
  std::array<nn::Tensor, 3> pooling;     ///< Three max-pooling outputs.
  std::array<nn::Tensor, 2> normalized;  ///< LRN after the first two pools.
  nn::Tensor fc6;                        ///< First hidden ReLU before dropout.
  nn::Tensor fc7;                        ///< Second hidden ReLU before dropout.
  nn::Tensor logits;                     ///< Unnormalised [N,classes] scores.
};

/// @brief Build the complete ZFNet topology without performing optimizer updates.
///
/// Uses dense channel connectivity, a one-pixel padded 7x7 stride-two stem,
/// VALID 5x5 stride-two conv2, SAME first two pools, LRN after those pools,
/// three SAME 3x3 convolutions and a VALID final pool. He/Glorot initialization
/// and inverted dropout are adaptations. Reference model learns 62357608 scalars.
///
/// @param p Mutable learnable store; creates/reuses every named floating leaf.
/// @param image Floating [N,cfg.image_size,cfg.image_size,3], matching store dtype.
/// @param cfg Valid reference or explicitly compact dimensions.
/// @param training Enable dropout when true; false by default.
/// @return All recorded stages and [N,classes] unnormalised logits, with autograd history.
/// @throws std::invalid_argument For invalid cfg, RGB geometry, tensor dtype or parameter/state entries.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17);
/// n::models::ZFNetConfig cfg;
/// cfg.image_size = 67;
/// cfg.classes = 3;
/// cfg.width_divisor = 8;
/// cfg.dense_width = 16;
/// const auto out = n::models::ZFNet(p, n::zeros({1, cfg.image_size, cfg.image_size, 3}), cfg);
/// @endcode
/// @par References
/// - Zeiler and Fergus (2014), https://cs.nyu.edu/~fergus/papers/zeilerECCV2014.pdf
/// @see ZFNetConfig
inline ZFNetOutputs ZFNet(nn::ParameterStore &p, const nn::Tensor &image,
                          const ZFNetConfig &cfg = {}, bool training = false) {
  cfg.Validate();
  if (image.dim() != 4 || image.size(1) != cfg.image_size || image.size(2) != cfg.image_size ||
      image.size(3) != 3)
    throw std::invalid_argument("image does not match model configuration");
  ZFNetOutputs out;
  out.image = image;
  out.padded_image = nn::pad(image, {{0, 0}, {1, 1}, {1, 1}, {0, 0}});
  const std::int64_t c1 = 96 / cfg.width_divisor, c2 = 256 / cfg.width_divisor,
                     c3 = 384 / cfg.width_divisor;
  nn::ConvLayerOptions options;
  options.activation = nn::Activation::Relu;
  options.init = nn::init::he_uniform();
  options.stride = 2;
  options.padding = nn::Padding::Valid;
  out.convolution[0] = nn::conv2d(p, "conv1", out.padded_image, c1, 7, options);
  out.pooling[0] = nn::max_pool2d(out.convolution[0], 3, 2, nn::Padding::Same);
  out.normalized[0] = AlexNetLrn(out.pooling[0]);
  out.convolution[1] = nn::conv2d(p, "conv2", out.normalized[0], c2, 5, options);
  out.pooling[1] = nn::max_pool2d(out.convolution[1], 3, 2, nn::Padding::Same);
  out.normalized[1] = AlexNetLrn(out.pooling[1]);
  options.stride = 1;
  options.padding = nn::Padding::Same;
  out.convolution[2] = nn::conv2d(p, "conv3", out.normalized[1], c3, 3, options);
  out.convolution[3] = nn::conv2d(p, "conv4", out.convolution[2], c3, 3, options);
  out.convolution[4] = nn::conv2d(p, "conv5", out.convolution[3], c2, 3, options);
  out.pooling[2] = nn::max_pool2d(out.convolution[4], 3, 2);
  const std::int64_t features =
      std::int64_t(cfg.FinalSpatialSize()) * cfg.FinalSpatialSize() * c2;
  nn::DenseOptions dense;
  dense.init = nn::init::he_uniform();
  dense.activation = nn::Activation::Relu;
  out.fc6 =
      nn::dense(p, "fc6", nn::reshape(out.pooling[2], {-1, features}), cfg.dense_width, dense);
  out.fc7 =
      nn::dense(p, "fc7", training ? KeepDropout(p, out.fc6, cfg.keep_probability) : out.fc6,
                cfg.dense_width, dense);
  out.logits =
      nn::dense(p, "fc8", training ? KeepDropout(p, out.fc7, cfg.keep_probability) : out.fc7,
                cfg.classes);
  return out;
}

// Reconstruct from a signal shaped like conv5's activation map. Reuse each
// forward pooling switch; omit LRN, bias subtraction and inverse nonlinearities.
/// @brief Project a conv5 signal back to the resized RGB input grid using deconvnet rules.
///
/// At every convolution, rectify the backward signal before applying its adjoint.
/// At pooling, reuse the forward argmax switches. Omit LRN inversion, bias
/// subtraction and inverse nonlinearities, then remove only the explicit stem pad.
/// No fitting, and no claim that the result is a gradient or an input reconstruction.
///
/// @param p Existing ZFNet store with conv1 through conv5 HWIO kernels.
/// @param net Forward stage values used for shapes and pooling switches.
/// @param signal Floating tensor with the same shape as net.convolution[4].
/// @return RGB tensor shaped like net.image; switch unpooling breaks the autograd graph.
/// @throws std::invalid_argument For incompatible stage/signal geometry or kernel entries.
/// @throws std::out_of_range If a required convolution kernel is absent from p.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17);
/// n::models::ZFNetConfig cfg;
/// cfg.image_size = 67;
/// cfg.classes = 3;
/// cfg.width_divisor = 8;
/// cfg.dense_width = 16;
/// const auto out = n::models::ZFNet(p, n::zeros({1, cfg.image_size, cfg.image_size, 3}), cfg);
/// const auto reconstruction = n::models::ZFNetDeconvolution(p, out, out.convolution[4]);
/// @endcode
/// @par References
/// - Zeiler and Fergus (2014), https://cs.nyu.edu/~fergus/papers/zeilerECCV2014.pdf
/// @see UnpoolAtSwitches(), DeconvLayer()
inline nn::Tensor ZFNetDeconvolution(const nn::ParameterStore &p, const ZFNetOutputs &net,
                                     const nn::Tensor &signal) {
  auto h = DeconvLayer(net.convolution[3], p.at("conv5/W"), signal, 1, "SAME");
  h = DeconvLayer(net.convolution[2], p.at("conv4/W"), h, 1, "SAME");
  h = DeconvLayer(net.normalized[1], p.at("conv3/W"), h, 1, "SAME");
  h = UnpoolAtSwitches(net.convolution[1], h, 3, 2, "SAME");
  h = DeconvLayer(net.normalized[0], p.at("conv2/W"), h, 2, "VALID");
  h = UnpoolAtSwitches(net.convolution[0], h, 3, 2, "SAME");
  h = DeconvLayer(net.padded_image, p.at("conv1/W"), h, 2, "VALID");
  // Remove only the explicit stem padding, retaining the resized image grid.
  const auto &s = net.image.shape();
  return nn::slice(h, {0, 1, 1, 0}, {s[0], s[1], s[2], s[3]});
}

/// @brief Reference or explicitly compact settings for Xception.
///
/// Value-only architecture metadata; creating it does not fit or allocate weights.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(17);
/// n::models::XceptionConfig cfg;
/// cfg.image_size = 71;
/// cfg.classes = 3;
/// cfg.width_divisor = 8;
/// cfg.Validate();
/// @endcode
/// @par References
/// - Chollet (2017), https://arxiv.org/abs/1610.02357; Keras reference topology, https://github.com/keras-team/keras/blob/master/keras/src/applications/xception.py
/// @see Xception()
struct XceptionConfig {
  int image_size = 299;  ///< Square RGB input side, at least 71; reference 299.
  int classes = 1000;    ///< At least two output classes.
  int width_divisor = 1; ///< One, two, four or eight; divides every convolution width.
  double momentum = .99; ///< Finite batch-normalisation EMA retention in [0,1).

  /// @brief Validate the architecture dimensions and probability settings.
  ///
  /// @throws std::invalid_argument For settings outside the field domains.
  /// @par Example
  /// @code
  /// dm::nn::models::XceptionConfig{}.Validate();
  /// @endcode
  /// @see XceptionConfig
  void Validate() const {
    if (image_size < 71 || classes < 2 ||
        (width_divisor != 1 && width_divisor != 2 && width_divisor != 4 &&
         width_divisor != 8) ||
        !(momentum >= 0 && momentum < 1))
      throw std::invalid_argument("invalid Xception configuration");
  }
};

/// @brief Named forward activations and classifier scores for Xception.
///
/// Tensor handles retain storage and graphs; fitted parameters remain in p.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(17);
/// n::models::XceptionConfig cfg;
/// cfg.image_size = 71;
/// cfg.classes = 3;
/// cfg.width_divisor = 8;
/// const auto out =
///     n::models::Xception(p, state, n::zeros({1, cfg.image_size, cfg.image_size, 3}), cfg);
/// @endcode
/// @see Xception()
struct XceptionOutputs {
  std::vector<std::pair<std::string, nn::Tensor>>
      stages;        ///< Sixteen named feature maps in execution order.
  nn::Tensor logits; ///< Unnormalised [N,classes] scores after global averaging.
};

/// @brief Apply a linear depthwise-pointwise 3x3 convolution, then learned-scale BN.
///
/// There is no ReLU or BN between the two filters, and no final ReLU in this
/// helper. Glorot initializes both filters; BN always learns scale irrespective of
/// cfg.scale. No fitting occurs; state updates are the only eager training side effect.
///
/// @param p Mutable learnable store; no depthwise/pointwise biases; gamma and beta learned.
/// @param state Separate moving-moment store, updated when cfg requests training updates.
/// @param name Prefix for separable filter and /bn names.
/// @param input Floating NHWC feature map, matching store dtype.
/// @param out Positive output-channel width.
/// @param cfg BN mode, momentum, epsilon and update flag; dilation is unused here.
/// @return Same-resolution NHWC activation with out channels, retaining its gradient graph.
/// @throws std::invalid_argument For invalid widths, BN settings, input or stored shapes.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(17);
/// n::models::ConvNormConfig cfg;
/// const auto y = n::models::XceptionSeparable(p, state, "sep", n::ones({2, 5, 5, 3}), 4, cfg);
/// @endcode
/// @par References
/// - Chollet (2017), https://arxiv.org/abs/1610.02357; Keras reference topology, https://github.com/keras-team/keras/blob/master/keras/src/applications/xception.py
/// @see Xception(), NormalizedConv()
inline nn::Tensor XceptionSeparable(nn::ParameterStore &p, nn::ParameterStore &state,
                                    const std::string &name, const nn::Tensor &input,
                                    std::int64_t out, const ConvNormConfig &cfg) {
  nn::ConvLayerOptions options;
  options.bias = false;
  options.init = nn::init::glorot_uniform();
  const auto h = nn::separable_conv2d(p, name, input, out, 3, options);
  nn::BatchNormOptions bn;
  bn.momentum = cfg.momentum;
  bn.eps = cfg.epsilon;
  bn.scale = true;
  bn.update_stats = cfg.update_stats;
  return nn::batch_norm(p, state, name + "/bn", h, cfg.training, bn);
}

/// @brief Build the complete Xception topology without performing optimizer updates.
///
/// Preserves entry flow, all eight three-separable middle residual blocks and exit
/// flow. No intermediate activation between depthwise and pointwise filters.
/// BN learns gamma/beta and uses unbiased EMA variance per nn::batch_norm.
/// No dropout or auxiliary heads. Reference model learns 22855952 scalars and
/// tracks 54528 moving-statistic scalars. No pretrained weights are imported.
///
/// @param p Mutable learnable store; creates/reuses every named floating leaf.
/// @param state Separate BN moving-moment store, updated eagerly only during training.
/// @param image Floating [N,cfg.image_size,cfg.image_size,3], matching store dtype.
/// @param cfg Valid reference or explicitly compact dimensions.
/// @param training Enable batch moments and moving-state updates when true; false by default.
/// @return All recorded stages and [N,classes] unnormalised logits, with autograd history.
/// @throws std::invalid_argument For invalid cfg, RGB geometry, tensor dtype or parameter/state entries.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(17);
/// n::models::XceptionConfig cfg;
/// cfg.image_size = 71;
/// cfg.classes = 3;
/// cfg.width_divisor = 8;
/// const auto out =
///     n::models::Xception(p, state, n::zeros({1, cfg.image_size, cfg.image_size, 3}), cfg);
/// @endcode
/// @par References
/// - Chollet (2017), https://arxiv.org/abs/1610.02357; Keras reference topology, https://github.com/keras-team/keras/blob/master/keras/src/applications/xception.py
/// @see XceptionConfig
inline XceptionOutputs Xception(nn::ParameterStore &p, nn::ParameterStore &state,
                                const nn::Tensor &image, const XceptionConfig &cfg = {},
                                bool training = false) {
  cfg.Validate();
  if (image.dim() != 4 || image.size(1) != cfg.image_size || image.size(2) != cfg.image_size ||
      image.size(3) != 3)
    throw std::invalid_argument("image does not match model configuration");
  XceptionOutputs out;
  ConvNormConfig norm;
  norm.training = training;
  norm.scale = true;
  norm.momentum = cfg.momentum;
  const int divisor = cfg.width_divisor;
  // Ordinary convolution -> BN; the skip projection omits the final ReLU.
  auto convolution = [&](const std::string &name, const nn::Tensor &x, int channels,
                         int kernel, int stride, const std::string &padding, bool relu) {
    return NormalizedConv(p, state, name, x, channels, kernel, kernel, norm, stride, padding,
                          relu);
  };
  // Preserve the linear depthwise -> pointwise ordering before a single BN.
  auto separable = [&](const std::string &name, const nn::Tensor &x, int channels) {
    return XceptionSeparable(p, state, name, x, channels, norm);
  };
  nn::Tensor h = convolution("stem1", image, 32 / divisor, 3, 2, "VALID", true);
  out.stages.emplace_back("stem1", h);
  h = convolution("stem2", h, 64 / divisor, 3, 1, "VALID", true);
  out.stages.emplace_back("stem2", h);
  int channels = 64 / divisor;
  for (int block = 0; block < 3; ++block) {
    const int width = (block == 0 ? 128 : block == 1 ? 256 : 728) / divisor;
    const auto name = "entry" + std::to_string(block + 1);
    const auto skip = convolution(name + "/skip", h, width, 1, 2, "SAME", false);
    if (block > 0)
      h = nn::relu(h);
    h = separable(name + "/sep1", h, width);
    h = separable(name + "/sep2", nn::relu(h), width);
    h = nn::max_pool2d(h, 3, 2, nn::Padding::Same) + skip;
    channels = width;
    out.stages.emplace_back(name, h);
  }
  for (int block = 0; block < 8; ++block) {
    const auto name = "middle" + std::to_string(block + 1);
    const auto skip = h;
    for (int layer = 1; layer <= 3; ++layer)
      h = separable(name + "/sep" + std::to_string(layer), nn::relu(h), channels);
    h = h + skip;
    out.stages.emplace_back(name, h);
  }
  const auto skip = convolution("exit/skip", h, 1024 / divisor, 1, 2, "SAME", false);
  h = separable("exit/sep1", nn::relu(h), channels);
  h = separable("exit/sep2", nn::relu(h), 1024 / divisor);
  h = nn::max_pool2d(h, 3, 2, nn::Padding::Same) + skip;
  out.stages.emplace_back("exit", h);
  h = nn::relu(separable("final/sep1", h, 1536 / divisor));
  out.stages.emplace_back("final1", h);
  h = nn::relu(separable("final/sep2", h, 2048 / divisor));
  out.stages.emplace_back("final2", h);
  out.logits = nn::dense(p, "output", nn::global_avg_pool2d(h), cfg.classes);
  return out;
}
} // namespace datamunge::nn::models
