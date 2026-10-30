#pragma once
/// @file super_resolution_models.hpp
/// @brief Five tensor super-resolution architectures,CRD shuffle,channel attention and explicit-gradient clipping.
#include "datamunge/nn/autograd.hpp"
#include "datamunge/nn/layers.hpp"
#include "datamunge/nn/ops.hpp"
#include <array>
#include <cmath>
#include <string>
#include <vector>
namespace datamunge::nn::models {
namespace nn = datamunge::nn;
using nn::Tensor;
/// @brief Select one of five genuine single-image reconstruction architectures.
///
/// No hidden fit or optimizer;selection only.
///
/// @throws std::invalid_argument For invalid architecture,shape,configuration,channels or incompatible parameters.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto k = m::SRKind::kEdsr;
/// @endcode
/// @par References
/// - Dong et al.(2015),Image Super-Resolution Using Deep Convolutional Networks, https://arxiv.org/abs/1501.00092;Shi et al.(2016),Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network, https://arxiv.org/abs/1609.05158;Kim et al.(2016),Accurate Image Super-Resolution Using Very Deep Convolutional Networks, https://cv.snu.ac.kr/research/VDSR/;Lim et al.(2017),Enhanced Deep Residual Networks for Single Image Super-Resolution, https://arxiv.org/abs/1707.02921;Zhang et al.(2018),Image Super-Resolution Using Very Deep Residual Channel Attention Networks, https://arxiv.org/abs/1807.02758. Compact from-scratch teaching training,not pretrained benchmark results.
/// @see SuperResolution(), SRConfig, SRPixelShuffle()
enum class SRKind {
  kSrcnn, ///< Three valid convolutions on bicubic studio-range luma.
  kEspcn, ///< Two tanh low-resolution convolutions and CRD shuffle.
  kVdsr,  ///< Twenty convolutions and an image-space bicubic residual.
  kEdsr,  ///< Scaled residual blocks,feature skip and CRD RGB reconstruction.
  kRcan   ///< Nested residual channel-attention groups and CRD RGB reconstruction.
};
/// @brief Return the stable lowercase architecture name.
///
/// Throws for an invalid enum value.
///
/// @param k Supported SR architecture.
/// @return Static string with process lifetime.
/// @throws std::invalid_argument For invalid architecture,shape,configuration,channels or incompatible parameters.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto name = m::SRName(m::SRKind::kSrcnn);
/// @endcode
/// @par References
/// - Dong et al.(2015),Image Super-Resolution Using Deep Convolutional Networks, https://arxiv.org/abs/1501.00092;Shi et al.(2016),Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network, https://arxiv.org/abs/1609.05158;Kim et al.(2016),Accurate Image Super-Resolution Using Very Deep Convolutional Networks, https://cv.snu.ac.kr/research/VDSR/;Lim et al.(2017),Enhanced Deep Residual Networks for Single Image Super-Resolution, https://arxiv.org/abs/1707.02921;Zhang et al.(2018),Image Super-Resolution Using Very Deep Residual Channel Attention Networks, https://arxiv.org/abs/1807.02758. Compact from-scratch teaching training,not pretrained benchmark results.
/// @see SuperResolution(), SRConfig, SRPixelShuffle()
inline const char *SRName(SRKind k) {
  switch (k) {
  case SRKind::kSrcnn:
    return "srcnn";
  case SRKind::kEspcn:
    return "espcn";
  case SRKind::kVdsr:
    return "vdsr";
  case SRKind::kEdsr:
    return "edsr";
  case SRKind::kRcan:
    return "rcan";
  }
  throw std::invalid_argument("SR kind");
}

/// @brief Describe full or compact super-resolution graph metadata.
///
/// SRCNN/ESPCN widths remain64/32. Compact VDSR has20 layers/16 channels;
/// EDSR has8 blocks/16 channels;RCAN has2 groups/3 blocks/16 channels/reduction4.
/// Full EDSR has32 blocks/256 channels;full RCAN10 groups/20 blocks/64 channels.
/// No BN in any graph. Scale2 by default;mean shift is fixed RGB metadata.
///
/// @throws std::invalid_argument For invalid architecture,shape,configuration,channels or incompatible parameters.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto c = m::SRConfig::Compact(m::SRKind::kRcan);
/// @endcode
/// @par References
/// - Dong et al.(2015),Image Super-Resolution Using Deep Convolutional Networks, https://arxiv.org/abs/1501.00092;Shi et al.(2016),Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network, https://arxiv.org/abs/1609.05158;Kim et al.(2016),Accurate Image Super-Resolution Using Very Deep Convolutional Networks, https://cv.snu.ac.kr/research/VDSR/;Lim et al.(2017),Enhanced Deep Residual Networks for Single Image Super-Resolution, https://arxiv.org/abs/1707.02921;Zhang et al.(2018),Image Super-Resolution Using Very Deep Residual Channel Attention Networks, https://arxiv.org/abs/1807.02758. Compact from-scratch teaching training,not pretrained benchmark results.
/// @see SuperResolution(), SRConfig, SRPixelShuffle()
struct SRConfig {
  SRKind kind = SRKind::kSrcnn; ///< Architecture selector.
  int scale = 2;                ///< Integer spatial enlargement,2 by default.
  int width = 64;               ///< Hidden feature channels,architecture-dependent.
  int blocks = 20;              ///< EDSR residual blocks or RCAN blocks per group.
  int groups = 10;              ///< RCAN residual groups.
  int reduction = 16;           ///< RCAN gate bottleneck divisor.

                                /// @brief Construct paper-scale architecture metadata.
                                ///
                                /// No weights allocated and no optimizer constructed.
                                ///
                                /// @param kind Supported architecture.
                                /// @return Full configuration with default scale2.
  /// @throws std::invalid_argument For invalid architecture,shape,configuration,channels or incompatible parameters.
  /// @par Example
  /// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto c = m::SRConfig::Full(m::SRKind::kEdsr);
/// @endcode
  /// @par References
  /// - Dong et al.(2015),Image Super-Resolution Using Deep Convolutional Networks, https://arxiv.org/abs/1501.00092;Shi et al.(2016),Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network, https://arxiv.org/abs/1609.05158;Kim et al.(2016),Accurate Image Super-Resolution Using Very Deep Convolutional Networks, https://cv.snu.ac.kr/research/VDSR/;Lim et al.(2017),Enhanced Deep Residual Networks for Single Image Super-Resolution, https://arxiv.org/abs/1707.02921;Zhang et al.(2018),Image Super-Resolution Using Very Deep Residual Channel Attention Networks, https://arxiv.org/abs/1807.02758. Compact from-scratch teaching training,not pretrained benchmark results.
  /// @see SuperResolution(), SRConfig, SRPixelShuffle()
  static SRConfig Full(SRKind kind) {
    SRConfig c;
    c.kind = kind;
    if (kind == SRKind::kEdsr)
      c.width = 256, c.blocks = 32;
    return c;
  }
  /// @brief Construct the declared teaching-scale architecture metadata.
  ///
  /// See SRConfig for exact widths/depths;SRCNN/ESPCN remain full width.
  ///
  /// @param kind Supported architecture.
  /// @return Compact configuration,retaining all model mechanisms.
  /// @throws std::invalid_argument For invalid architecture,shape,configuration,channels or incompatible parameters.
  /// @par Example
  /// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto c = m::SRConfig::Compact(m::SRKind::kRcan);
/// @endcode
  /// @par References
  /// - Dong et al.(2015),Image Super-Resolution Using Deep Convolutional Networks, https://arxiv.org/abs/1501.00092;Shi et al.(2016),Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network, https://arxiv.org/abs/1609.05158;Kim et al.(2016),Accurate Image Super-Resolution Using Very Deep Convolutional Networks, https://cv.snu.ac.kr/research/VDSR/;Lim et al.(2017),Enhanced Deep Residual Networks for Single Image Super-Resolution, https://arxiv.org/abs/1707.02921;Zhang et al.(2018),Image Super-Resolution Using Very Deep Residual Channel Attention Networks, https://arxiv.org/abs/1807.02758. Compact from-scratch teaching training,not pretrained benchmark results.
  /// @see SuperResolution(), SRConfig, SRPixelShuffle()
  static SRConfig Compact(SRKind kind) {
    auto c = Full(kind);
    if (kind == SRKind::kVdsr)
      c.width = 16;
    if (kind == SRKind::kEdsr)
      c.width = 16, c.blocks = 8;
    if (kind == SRKind::kRcan)
      c.width = 16, c.groups = 2, c.blocks = 3, c.reduction = 4;
    return c;
  }
  /// @brief State whether the network consumes a bicubic HR-grid input.
  ///
  /// Pure architecture metadata.
  ///
  /// @return True for SRCNN/VDSR;false for ESPCN/EDSR/RCAN.
  /// @throws std::invalid_argument For invalid architecture,shape,configuration,channels or incompatible parameters.

  /// @par References
  /// - Dong et al.(2015),Image Super-Resolution Using Deep Convolutional Networks, https://arxiv.org/abs/1501.00092;Shi et al.(2016),Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network, https://arxiv.org/abs/1609.05158;Kim et al.(2016),Accurate Image Super-Resolution Using Very Deep Convolutional Networks, https://cv.snu.ac.kr/research/VDSR/;Lim et al.(2017),Enhanced Deep Residual Networks for Single Image Super-Resolution, https://arxiv.org/abs/1707.02921;Zhang et al.(2018),Image Super-Resolution Using Very Deep Residual Channel Attention Networks, https://arxiv.org/abs/1807.02758. Compact from-scratch teaching training,not pretrained benchmark results.
  /// @see SuperResolution(), SRConfig, SRPixelShuffle()
  bool PreUpsampled() const {
    return kind == SRKind::kSrcnn || kind == SRKind::kVdsr;
  }
  /// @brief Return the required input and output channel count.
  ///
  /// No conversion occurs here.
  ///
  /// @return Three for RGB EDSR/RCAN;one studio-range Y for the other models.
  /// @throws std::invalid_argument For invalid architecture,shape,configuration,channels or incompatible parameters.

  /// @par References
  /// - Dong et al.(2015),Image Super-Resolution Using Deep Convolutional Networks, https://arxiv.org/abs/1501.00092;Shi et al.(2016),Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network, https://arxiv.org/abs/1609.05158;Kim et al.(2016),Accurate Image Super-Resolution Using Very Deep Convolutional Networks, https://cv.snu.ac.kr/research/VDSR/;Lim et al.(2017),Enhanced Deep Residual Networks for Single Image Super-Resolution, https://arxiv.org/abs/1707.02921;Zhang et al.(2018),Image Super-Resolution Using Very Deep Residual Channel Attention Networks, https://arxiv.org/abs/1807.02758. Compact from-scratch teaching training,not pretrained benchmark results.
  /// @see SuperResolution(), SRConfig, SRPixelShuffle()
  int Channels() const {
    return kind == SRKind::kEdsr || kind == SRKind::kRcan ? 3 : 1;
  }
};

/// @brief Rearrange CRD-packed channels into a larger spatial grid.
///
/// Input index is c*r*r+dy*r+dx. This differs from TensorFlow DCR ordering.
/// No interpolation,clipping,parameters or hidden fitting.
///
/// @param x NHWC tensor with channels*scale*scale final channels.
/// @param channels Positive output channels.
/// @param scale Integer enlargement>=2.
/// @return Graph-preserving[N,H*scale,W*scale,channels].
/// @throws std::invalid_argument For invalid architecture,shape,configuration,channels or incompatible parameters.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto y = m::SRPixelShuffle(n::arange(0, 8).reshape({1, 1, 1, 8}), 2, 2);
/// @endcode
/// @par References
/// - Dong et al.(2015),Image Super-Resolution Using Deep Convolutional Networks, https://arxiv.org/abs/1501.00092;Shi et al.(2016),Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network, https://arxiv.org/abs/1609.05158;Kim et al.(2016),Accurate Image Super-Resolution Using Very Deep Convolutional Networks, https://cv.snu.ac.kr/research/VDSR/;Lim et al.(2017),Enhanced Deep Residual Networks for Single Image Super-Resolution, https://arxiv.org/abs/1707.02921;Zhang et al.(2018),Image Super-Resolution Using Very Deep Residual Channel Attention Networks, https://arxiv.org/abs/1807.02758. Compact from-scratch teaching training,not pretrained benchmark results.
/// @see SuperResolution(), SRConfig, SRPixelShuffle()
inline Tensor SRPixelShuffle(const Tensor &x, int channels, int scale) {
  if (channels < 1 || scale < 2)
    throw std::invalid_argument("pixel shuffle channels/scale");
  if (x.dim() != 4 || x.size(3) != channels * scale * scale)
    throw std::invalid_argument("CRD shuffle NHWC packed channels");
  const auto b = x.size(0), h = x.size(1), w = x.size(2);
  return nn::transpose(x.reshape({b, h, w, channels, scale, scale}), {0, 1, 4, 2, 5, 3})
      .reshape({b, h * scale, w * scale, channels});
}

/// @brief Apply fan-in-uniform SAME convolution with zero bias.
///
/// Weights initialize U(+-1/sqrt(float(kernel*kernel*in))),matching the source
/// fan-in convention. Repeated names reuse leaves;caller warms before optimizer.
///
/// @param p Mutable learned store.
/// @param name Unique layer prefix.
/// @param x Floating NHWC input matching p dtype.
/// @param in Positive expected input channels.
/// @param out Positive output channels.
/// @param kernel Positive square kernel size.
/// @param relu Whether to apply ReLU,default false.
/// @return Graph-preserving convolution result.
/// @throws std::invalid_argument For invalid architecture,shape,configuration,channels or incompatible parameters.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// n::ParameterStore p(17);
/// const auto y = m::SRConv(p, "a", n::ones({1, 3, 4, 2}), 2, 3, 3);
/// @endcode
/// @par References
/// - Dong et al.(2015),Image Super-Resolution Using Deep Convolutional Networks, https://arxiv.org/abs/1501.00092;Shi et al.(2016),Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network, https://arxiv.org/abs/1609.05158;Kim et al.(2016),Accurate Image Super-Resolution Using Very Deep Convolutional Networks, https://cv.snu.ac.kr/research/VDSR/;Lim et al.(2017),Enhanced Deep Residual Networks for Single Image Super-Resolution, https://arxiv.org/abs/1707.02921;Zhang et al.(2018),Image Super-Resolution Using Very Deep Residual Channel Attention Networks, https://arxiv.org/abs/1807.02758. Compact from-scratch teaching training,not pretrained benchmark results.
/// @see SuperResolution(), SRConfig, SRPixelShuffle()
inline Tensor SRConv(nn::ParameterStore &p, const std::string &name, const Tensor &x, int in,
                     int out, int kernel, bool relu = false) {
  if (in < 1 || out < 1 || kernel < 1 || x.dim() != 4 || x.size(3) != in)
    throw std::invalid_argument("SR convolution NHWC channels");
  p.get(name + "/W", {kernel, kernel, in, out},
        nn::init::uniform(1 / std::sqrt(float(kernel * kernel * in))));
  const Tensor y = nn::conv2d(p, name, x, out, kernel);
  return relu ? nn::relu(y) : y;
}

/// @brief Apply a Glorot/default or explicitly initialized convolution.
///
/// Zero biases and explicit activation/padding. Parameters persist in p. No optimizer.
///
/// @param p Mutable learned store.
/// @param name Unique layer prefix.
/// @param x Floating NHWC input.
/// @param out Positive output channels.
/// @param kernel Positive square kernel size.
/// @param act Activation,default None.
/// @param padding Spatial padding,default Same.
/// @param init Kernel initializer,default Glorot uniform.
/// @return Differentiable convolution result.
/// @throws std::invalid_argument For invalid architecture,shape,configuration,channels or incompatible parameters.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// n::ParameterStore p(17);
/// const auto y = m::SRLayer(p, "a", n::ones({1, 3, 4, 1}), 2, 3);
/// @endcode
/// @par References
/// - Dong et al.(2015),Image Super-Resolution Using Deep Convolutional Networks, https://arxiv.org/abs/1501.00092;Shi et al.(2016),Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network, https://arxiv.org/abs/1609.05158;Kim et al.(2016),Accurate Image Super-Resolution Using Very Deep Convolutional Networks, https://cv.snu.ac.kr/research/VDSR/;Lim et al.(2017),Enhanced Deep Residual Networks for Single Image Super-Resolution, https://arxiv.org/abs/1707.02921;Zhang et al.(2018),Image Super-Resolution Using Very Deep Residual Channel Attention Networks, https://arxiv.org/abs/1807.02758. Compact from-scratch teaching training,not pretrained benchmark results.
/// @see SuperResolution(), SRConfig, SRPixelShuffle()
inline Tensor SRLayer(nn::ParameterStore &p, const std::string &name, const Tensor &x, int out,
                      int kernel, nn::Activation act = nn::Activation::None,
                      nn::Padding padding = nn::Padding::Same,
                      const nn::Initializer &init = nn::init::glorot_uniform()) {
  nn::ConvLayerOptions o;
  o.activation = act;
  o.padding = padding;
  o.init = init;
  return nn::conv2d(p, name, x, out, kernel, o);
}

/// @brief Apply a two-convolution scaled feature residual.
///
/// No batch normalization or image-space bicubic addition. Four learned leaves.
///
/// @param p Mutable learned store.
/// @param name Unique block prefix.
/// @param input Floating NHWC width-channel features.
/// @param width Positive feature channels.
/// @param scaling Residual multiplier,.1 for EDSR.
/// @return input+scaling*conv2(relu(conv1(input))).
/// @throws std::invalid_argument For invalid architecture,shape,configuration,channels or incompatible parameters.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// n::ParameterStore p(17);
/// const auto y = m::SRResidual(p, "b", n::ones({1, 3, 4, 2}), 2, .1);
/// @endcode
/// @par References
/// - Dong et al.(2015),Image Super-Resolution Using Deep Convolutional Networks, https://arxiv.org/abs/1501.00092;Shi et al.(2016),Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network, https://arxiv.org/abs/1609.05158;Kim et al.(2016),Accurate Image Super-Resolution Using Very Deep Convolutional Networks, https://cv.snu.ac.kr/research/VDSR/;Lim et al.(2017),Enhanced Deep Residual Networks for Single Image Super-Resolution, https://arxiv.org/abs/1707.02921;Zhang et al.(2018),Image Super-Resolution Using Very Deep Residual Channel Attention Networks, https://arxiv.org/abs/1807.02758. Compact from-scratch teaching training,not pretrained benchmark results.
/// @see SuperResolution(), SRConfig, SRPixelShuffle()
inline Tensor SRResidual(nn::ParameterStore &p, const std::string &name, const Tensor &input,
                         int width, double scaling) {
  const Tensor h = SRConv(p, name + "/conv1", input, width, width, 3, true);
  const Tensor residual = SRConv(p, name + "/conv2", h, width, width, 3);
  return input + scaling * residual;
}

/// @brief Own channel-attended output and differentiable gate activations.
///
/// Gate gradients flow through global spatial means,ReLU bottleneck and sigmoid.
///
/// @throws std::invalid_argument For invalid architecture,shape,configuration,channels or incompatible parameters.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// n::ParameterStore p(17);
/// const auto g = m::SRChannelAttention(p, "ca", n::ones({1, 3, 4, 4}), 4, 2);
/// @endcode
/// @par References
/// - Dong et al.(2015),Image Super-Resolution Using Deep Convolutional Networks, https://arxiv.org/abs/1501.00092;Shi et al.(2016),Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network, https://arxiv.org/abs/1609.05158;Kim et al.(2016),Accurate Image Super-Resolution Using Very Deep Convolutional Networks, https://cv.snu.ac.kr/research/VDSR/;Lim et al.(2017),Enhanced Deep Residual Networks for Single Image Super-Resolution, https://arxiv.org/abs/1707.02921;Zhang et al.(2018),Image Super-Resolution Using Very Deep Residual Channel Attention Networks, https://arxiv.org/abs/1807.02758. Compact from-scratch teaching training,not pretrained benchmark results.
/// @see SuperResolution(), SRConfig, SRPixelShuffle()
struct SRChannelGate {
  Tensor output; ///< Input multiplied by differentiable channel gates.
  Tensor gates;  ///< Sigmoid weights[N,1,1,width].
};
/// @brief Gate features using global pooled channel dependencies.
///
/// Two learned1x1 convolutions reduce then expand pooled features. No detach
/// or fitted summaries;global spatial pooling remains differentiable.
///
/// @param p Mutable learned store.
/// @param name Unique gate prefix.
/// @param input Floating[N,H,W,width] feature tensor.
/// @param width Positive feature channels.
/// @param reduction Positive exact divisor of width.
/// @return Attended output and sigmoid gates[N,1,1,width].
/// @throws std::invalid_argument For invalid architecture,shape,configuration,channels or incompatible parameters.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// n::ParameterStore p(17);
/// const auto g = m::SRChannelAttention(p, "ca", n::ones({1, 3, 4, 4}), 4, 2);
/// @endcode
/// @par References
/// - Dong et al.(2015),Image Super-Resolution Using Deep Convolutional Networks, https://arxiv.org/abs/1501.00092;Shi et al.(2016),Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network, https://arxiv.org/abs/1609.05158;Kim et al.(2016),Accurate Image Super-Resolution Using Very Deep Convolutional Networks, https://cv.snu.ac.kr/research/VDSR/;Lim et al.(2017),Enhanced Deep Residual Networks for Single Image Super-Resolution, https://arxiv.org/abs/1707.02921;Zhang et al.(2018),Image Super-Resolution Using Very Deep Residual Channel Attention Networks, https://arxiv.org/abs/1807.02758. Compact from-scratch teaching training,not pretrained benchmark results.
/// @see SuperResolution(), SRConfig, SRPixelShuffle()
inline SRChannelGate SRChannelAttention(nn::ParameterStore &p, const std::string &name,
                                        const Tensor &input, int width, int reduction) {
  if (reduction < 1 || width % reduction)
    throw std::invalid_argument("channel attention reduction");
  const Tensor mean = nn::mean(input, {1, 2}, true);
  const Tensor narrow = SRConv(p, name + "/reduce", mean, width, width / reduction, 1, true);
  const Tensor gate =
      nn::sigmoid(SRConv(p, name + "/expand", narrow, width / reduction, width, 1));
  return {input * gate, gate};
}
/// @brief Apply a residual channel-attention block.
///
/// Eight learned leaves;no BN. Channel gates see the residual feature branch.
///
/// @param p Mutable learned store.
/// @param name Unique block prefix.
/// @param input Floating NHWC width-channel input.
/// @param width Positive feature channels.
/// @param reduction Positive exact divisor for channel attention.
/// @return input+attention(conv2(relu(conv1(input)))).
/// @throws std::invalid_argument For invalid architecture,shape,configuration,channels or incompatible parameters.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// n::ParameterStore p(17);
/// const auto y = m::SRRcab(p, "b", n::ones({1, 3, 4, 4}), 4, 2);
/// @endcode
/// @par References
/// - Dong et al.(2015),Image Super-Resolution Using Deep Convolutional Networks, https://arxiv.org/abs/1501.00092;Shi et al.(2016),Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network, https://arxiv.org/abs/1609.05158;Kim et al.(2016),Accurate Image Super-Resolution Using Very Deep Convolutional Networks, https://cv.snu.ac.kr/research/VDSR/;Lim et al.(2017),Enhanced Deep Residual Networks for Single Image Super-Resolution, https://arxiv.org/abs/1707.02921;Zhang et al.(2018),Image Super-Resolution Using Very Deep Residual Channel Attention Networks, https://arxiv.org/abs/1807.02758. Compact from-scratch teaching training,not pretrained benchmark results.
/// @see SuperResolution(), SRConfig, SRPixelShuffle()
inline Tensor SRRcab(nn::ParameterStore &p, const std::string &name, const Tensor &input,
                     int width, int reduction) {
  const Tensor a = SRConv(p, name + "/conv1", input, width, width, 3, true);
  const Tensor b = SRConv(p, name + "/conv2", a, width, width, 3);
  return input + SRChannelAttention(p, name + "/ca", b, width, reduction).output;
}
/// @brief Learn low-resolution features before CRD spatial enlargement.
///
/// Scale4 uses two conv/shuffle x2 stages;scale3 uses one x3 stage.
/// Layer names upsample0/1 must belong to one model store.
///
/// @param p Mutable learned store.
/// @param input Floating NHWC width-channel features.
/// @param width Positive feature channels.
/// @param scale Enlargement2,3 or4.
/// @return Upsampled width-channel features.
/// @throws std::invalid_argument For invalid architecture,shape,configuration,channels or incompatible parameters.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// n::ParameterStore p(17);
/// const auto y = m::SRUpsample(p, n::ones({1, 2, 3, 2}), 2, 2);
/// @endcode
/// @par References
/// - Dong et al.(2015),Image Super-Resolution Using Deep Convolutional Networks, https://arxiv.org/abs/1501.00092;Shi et al.(2016),Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network, https://arxiv.org/abs/1609.05158;Kim et al.(2016),Accurate Image Super-Resolution Using Very Deep Convolutional Networks, https://cv.snu.ac.kr/research/VDSR/;Lim et al.(2017),Enhanced Deep Residual Networks for Single Image Super-Resolution, https://arxiv.org/abs/1707.02921;Zhang et al.(2018),Image Super-Resolution Using Very Deep Residual Channel Attention Networks, https://arxiv.org/abs/1807.02758. Compact from-scratch teaching training,not pretrained benchmark results.
/// @see SuperResolution(), SRConfig, SRPixelShuffle()
inline Tensor SRUpsample(nn::ParameterStore &p, const Tensor &input, int width, int scale) {
  if (scale != 2 && scale != 3 && scale != 4)
    throw std::invalid_argument("SR scale2,3,4 required");
  Tensor x = input;
  const int stages = scale == 4 ? 2 : 1, r = scale == 3 ? 3 : 2;
  for (int i = 0; i < stages; ++i)
    x = SRPixelShuffle(SRConv(p, "upsample" + std::to_string(i), x, width, width * r * r, 3),
                       width, r);
  return x;
}

/// @brief Own raw reconstructed image and retained intermediate graph stages.
///
/// Unclipped floating pixels,not a constrained image probability. Graph is retained.
///
/// @throws std::invalid_argument For invalid architecture,shape,configuration,channels or incompatible parameters.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// n::ParameterStore p(17);
/// const auto y =
///     m::SuperResolution(p, n::ones({1, 2, 3, 3}), m::SRConfig::Compact(m::SRKind::kEdsr));
/// @endcode
/// @par References
/// - Dong et al.(2015),Image Super-Resolution Using Deep Convolutional Networks, https://arxiv.org/abs/1501.00092;Shi et al.(2016),Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network, https://arxiv.org/abs/1609.05158;Kim et al.(2016),Accurate Image Super-Resolution Using Very Deep Convolutional Networks, https://cv.snu.ac.kr/research/VDSR/;Lim et al.(2017),Enhanced Deep Residual Networks for Single Image Super-Resolution, https://arxiv.org/abs/1707.02921;Zhang et al.(2018),Image Super-Resolution Using Very Deep Residual Channel Attention Networks, https://arxiv.org/abs/1807.02758. Compact from-scratch teaching training,not pretrained benchmark results.
/// @see SuperResolution(), SRConfig, SRPixelShuffle()
struct SROutputs {
  Tensor image;               ///< Raw unclipped reconstruction NHWC.
  std::vector<Tensor> stages; ///< Architecture-specific intermediate activations.
};
/// @brief Store fixed RGB mean-shift constants used by EDSR/RCAN.
///
/// Fixed source constants(.4488,.4371,.4040);never fitted to evaluation targets.
///
/// @throws std::invalid_argument For invalid architecture,shape,configuration,channels or incompatible parameters.
/// @par References
/// - Dong et al.(2015),Image Super-Resolution Using Deep Convolutional Networks, https://arxiv.org/abs/1501.00092;Shi et al.(2016),Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network, https://arxiv.org/abs/1609.05158;Kim et al.(2016),Accurate Image Super-Resolution Using Very Deep Convolutional Networks, https://cv.snu.ac.kr/research/VDSR/;Lim et al.(2017),Enhanced Deep Residual Networks for Single Image Super-Resolution, https://arxiv.org/abs/1707.02921;Zhang et al.(2018),Image Super-Resolution Using Very Deep Residual Channel Attention Networks, https://arxiv.org/abs/1807.02758. Compact from-scratch teaching training,not pretrained benchmark results.
/// @see SuperResolution(), SRConfig, SRPixelShuffle()
inline const std::array<float, 3> kSRMean = {.4488f, .4371f, .4040f};

/// @brief Run SRCNN,ESPCN,VDSR,EDSR or RCAN without hidden fitting.
///
/// SRCNN uses VALID9-1-5 convolutions and shrinks both dimensions by12.
/// ESPCN keeps source SAME padding and tanh5-3-3,then CRD shuffle. VDSR has
/// 20 layers;its final residual kernel uses normal(.001) for this short source
/// experiment. Only VDSR adds a bicubic image-space skip. EDSR scales blocks.1;
/// RCAN uses nested groups and channel gates. Both RGB models apply fixed mean
/// shift and a feature skip,then learned CRD upsampling. No BN,GT or optimizer
/// inputs. Warm every leaf before constructing the explicitly owned optimizer.
///
/// @param p Mutable learned store with one architecture per store.
/// @param input Floating NHWC observed LR or pre-upsampled image.
/// @param cfg Architecture metadata;default scale2,valid dimensions/channels.
/// @return Raw image and all declared intermediate activations.
/// @throws std::invalid_argument For invalid architecture,shape,configuration,channels or incompatible parameters.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// n::ParameterStore p(17);
/// const auto y =
///     m::SuperResolution(p, n::ones({1, 2, 3, 3}), m::SRConfig::Compact(m::SRKind::kEdsr));
/// @endcode
/// @par References
/// - Dong et al.(2015),Image Super-Resolution Using Deep Convolutional Networks, https://arxiv.org/abs/1501.00092;Shi et al.(2016),Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network, https://arxiv.org/abs/1609.05158;Kim et al.(2016),Accurate Image Super-Resolution Using Very Deep Convolutional Networks, https://cv.snu.ac.kr/research/VDSR/;Lim et al.(2017),Enhanced Deep Residual Networks for Single Image Super-Resolution, https://arxiv.org/abs/1707.02921;Zhang et al.(2018),Image Super-Resolution Using Very Deep Residual Channel Attention Networks, https://arxiv.org/abs/1807.02758. Compact from-scratch teaching training,not pretrained benchmark results.
/// @see SuperResolution(), SRConfig, SRPixelShuffle()
inline SROutputs SuperResolution(nn::ParameterStore &p, const Tensor &input,
                                 const SRConfig &cfg) {
  if (input.dim() != 4 || input.size(0) < 1 || input.size(1) < 1 || input.size(2) < 1 ||
      input.size(3) != cfg.Channels() || input.dtype() != p.dtype() ||
      (cfg.kind == SRKind::kSrcnn && (input.size(1) < 13 || input.size(2) < 13)) ||
      cfg.width < 1 || cfg.blocks < 1 || cfg.groups < 1 ||
      (cfg.scale != 2 && cfg.scale != 3 && cfg.scale != 4))
    throw std::invalid_argument("SR configuration");
  SROutputs net;
  if (cfg.kind == SRKind::kSrcnn) {
    const Tensor a =
        SRLayer(p, "extract", input, 64, 9, nn::Activation::Relu, nn::Padding::Valid);
    const Tensor b = SRLayer(p, "mapping", a, 32, 1, nn::Activation::Relu, nn::Padding::Valid);
    net.image = SRLayer(p, "reconstruct", b, 1, 5, nn::Activation::None, nn::Padding::Valid);
    net.stages = {a, b};
    return net;
  }
  if (cfg.kind == SRKind::kEspcn) {
    const Tensor a = SRLayer(p, "extract", input, 64, 5, nn::Activation::Tanh);
    const Tensor b = SRLayer(p, "mapping", a, 32, 3, nn::Activation::Tanh);
    const Tensor packed = SRLayer(p, "subpixel", b, cfg.scale * cfg.scale, 3);
    net.image = SRPixelShuffle(packed, 1, cfg.scale);
    net.stages = {a, b, packed};
    return net;
  }
  if (cfg.kind == SRKind::kVdsr) {
    Tensor x = SRLayer(p, "input", input, cfg.width, 3, nn::Activation::Relu,
                       nn::Padding::Same, nn::init::he_normal());
    net.stages.push_back(x);
    for (int i = 0; i < 18; ++i) {
      x = SRLayer(p, "hidden" + std::to_string(i), x, cfg.width, 3, nn::Activation::Relu,
                  nn::Padding::Same, nn::init::he_normal());
      net.stages.push_back(x);
    }
    // Small final initialization starts the short demonstration near bicubic;
    // this is explicit, not an extra shortcut absent from the architecture.
    const Tensor w = p.get("residual/W", {3, 3, cfg.width, 1}, nn::init::normal(.001));
    const Tensor residual =
        nn::bias_add(nn::conv2d(x, w, nn::conv_options(1, nn::Padding::Same)),
                     p.get("residual/b", {1}, nn::init::zeros()));
    net.image = input + residual;
    return net;
  }
  const Tensor mean = nn::from_vector(std::vector<float>(kSRMean.begin(), kSRMean.end()), {3})
                          .to(input.dtype());
  const Tensor head = SRConv(p, "head", input - mean, 3, cfg.width, 3);
  Tensor x = head;
  net.stages.push_back(head);
  if (cfg.kind == SRKind::kEdsr) {
    for (int i = 0; i < cfg.blocks; ++i) {
      x = SRResidual(p, "block" + std::to_string(i), x, cfg.width, .1);
      net.stages.push_back(x);
    }
  } else {
    for (int g = 0; g < cfg.groups; ++g) {
      const Tensor group = x;
      for (int b = 0; b < cfg.blocks; ++b)
        x = SRRcab(p, "group" + std::to_string(g) + "/block" + std::to_string(b), x, cfg.width,
                   cfg.reduction);
      x = group + SRConv(p, "group" + std::to_string(g) + "/tail", x, cfg.width, cfg.width, 3);
      net.stages.push_back(x);
    }
  }
  x = head + SRConv(p, "body_tail", x, cfg.width, cfg.width, 3);
  net.stages.push_back(x);
  x = SRUpsample(p, x, cfg.width, cfg.scale);
  net.stages.push_back(x);
  net.image = SRConv(p, "output", x, cfg.width, 3, 3) + mean;
  return net;
}

/// @brief Clip each VDSR derivative to its adjustable theta/rate bound.
///
/// Caller explicitly computes derivatives and calls optimizer.step. This helper
/// does not update weights or optimizer state. Float32 bound matches source.
///
/// @param gradient Floating derivative tensor,graph preserved.
/// @param learning_rate Positive finite optimizer rate.
/// @param theta Finite nonnegative update bound,default.01.
/// @return Elementwise clip to+-float(theta)/float(learning_rate).
/// @throws std::invalid_argument For invalid architecture,shape,configuration,channels or incompatible parameters.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto g = m::VdsrClippedGradient(n::tensor({-1., 0., 1.}), .1);
/// @endcode
/// @par References
/// - Dong et al.(2015),Image Super-Resolution Using Deep Convolutional Networks, https://arxiv.org/abs/1501.00092;Shi et al.(2016),Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network, https://arxiv.org/abs/1609.05158;Kim et al.(2016),Accurate Image Super-Resolution Using Very Deep Convolutional Networks, https://cv.snu.ac.kr/research/VDSR/;Lim et al.(2017),Enhanced Deep Residual Networks for Single Image Super-Resolution, https://arxiv.org/abs/1707.02921;Zhang et al.(2018),Image Super-Resolution Using Very Deep Residual Channel Attention Networks, https://arxiv.org/abs/1807.02758. Compact from-scratch teaching training,not pretrained benchmark results.
/// @see SuperResolution(), SRConfig, SRPixelShuffle()
inline Tensor VdsrClippedGradient(const Tensor &gradient, double learning_rate,
                                  double theta = .01) {
  if (!std::isfinite(learning_rate) || learning_rate <= 0 || !std::isfinite(theta) ||
      theta < 0)
    throw std::invalid_argument("positive finite VDSR learning rate/nonnegative theta");
  const double bound = double(float(theta) / float(learning_rate));
  return nn::clip(gradient, -bound, bound);
}

} // namespace datamunge::nn::models
