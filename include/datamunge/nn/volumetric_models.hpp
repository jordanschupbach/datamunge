#pragma once
/// @file volumetric_models.hpp
/// @brief Tensor 3D U-Net,V-Net,learned up-convolutions,PReLU and sparse/dense volumetric objectives.
#include "datamunge/nn/autograd.hpp"
#include "datamunge/nn/layers.hpp"
#include "datamunge/nn/ops.hpp"
#include <algorithm>
#include <limits>
#include <stdexcept>
namespace datamunge::nn::models {
namespace nn = datamunge::nn;
/// @brief Apply a learned2x2x2 stride2 VALID volumetric transpose convolution.
///
/// He-uniform kernel;linear output. No optimizer or hidden fitting.
///
/// @param p Floating learned parameter store.
/// @param name Layer prefix for W[2,2,2,out,in] and optional bias.
/// @param x Positive same-dtype NDHWC features.
/// @param out Positive output channel count.
/// @param bias False(default) omits bias;true creates out bias entries.
/// @return Graph-connected[N,2D,2H,2W,out] output.
/// @throws std::invalid_argument For invalid dimensions,geometry,dtypes,targets or incompatible stored leaves.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17);
/// const auto y = n::models::VolumeTranspose(p, "up", n::ones({1, 2, 3, 4, 2}), 3, true);
/// @endcode
/// @par References
/// - Çiçek,Abdulkadir,Lienkamp,Brox and Ronneberger(2016),3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation, https://arxiv.org/abs/1606.06650; Milletari,Navab and Ahmadi(2016),V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation, https://arxiv.org/abs/1606.04797. Initializers,explicit BN affine leaves and compact dimensions follow the source teaching example.
/// @see VolumeCrop()
inline nn::Tensor VolumeTranspose(nn::ParameterStore &p, const std::string &name,
                                  const nn::Tensor &x, std::int64_t out, bool bias = false) {
  if (x.dim() != 5 || x.shape()[0] < 1 || x.shape()[1] < 1 || x.shape()[2] < 1 ||
      x.shape()[3] < 1 || x.shape()[4] < 1 || out < 1 || x.dtype() != p.dtype())
    throw std::invalid_argument("invalid volume transpose inputs");
  nn::ConvLayerOptions o;
  o.stride = 2, o.padding = nn::Padding::Valid, o.bias = bias, o.init = nn::init::he_uniform();
  return nn::conv3d_transpose(p, name, x, out, 2, o);
}

/// @brief Floor-centre crop source spatial axes to a reference volume geometry.
///
/// Offsets floor((source-reference)/2) choose the earlier centre for odd
/// differences. No fitting,parameter creation or input mutation.
///
/// @param source Positive floating[N,D,H,W,C] features.
/// @param reference Rank5 tensor supplying positive target D/H/W no larger than source;other values/axes unused.
/// @return Source batch/channels with reference spatial sizes and gradients only in retained voxels.
/// @throws std::invalid_argument For invalid dimensions,geometry,dtypes,targets or incompatible stored leaves.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto crop = n::models::VolumeCrop(n::ones({1, 5, 7, 9, 2}), n::zeros({1, 3, 4, 6, 1}));
/// @endcode
/// @par References
/// - Çiçek,Abdulkadir,Lienkamp,Brox and Ronneberger(2016),3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation, https://arxiv.org/abs/1606.06650; Milletari,Navab and Ahmadi(2016),V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation, https://arxiv.org/abs/1606.04797. Initializers,explicit BN affine leaves and compact dimensions follow the source teaching example.
/// @see Unet3D()
inline nn::Tensor VolumeCrop(const nn::Tensor &source, const nn::Tensor &reference) {
  if (source.dim() != 5 || reference.dim() != 5 || source.shape()[0] < 1 ||
      source.shape()[4] < 1 ||
      (source.dtype() != nn::DType::Float32 && source.dtype() != nn::DType::Float64))
    throw std::invalid_argument("invalid volume crop tensors");
  for (int axis = 1; axis <= 3; ++axis)
    if (reference.shape()[axis] < 1 || reference.shape()[axis] > source.shape()[axis])
      throw std::invalid_argument("volume crop reference exceeds source");
  const auto &a = source.shape();
  const auto &b = reference.shape();
  return nn::slice(source, {0, (a[1] - b[1]) / 2, (a[2] - b[2]) / 2, (a[3] - b[3]) / 2, 0},
                   {a[0], b[1], b[2], b[3], a[4]});
}

/// @brief Calculate three-pool3D U-Net output side with SAME or cropped VALID convolution pairs.
///
/// Three floor stride2 pools and three2x learned up-convolutions determine
/// geometry. SAME produces8*floor(side/8). No tensors,training or mutation.
///
/// @param side Positive input side.
/// @param valid True(default) removes four voxels at each double convolution;false retains convolution sizes.
/// @return Positive output side;132->44 and116->28 with VALID,16->16 with SAME.
/// @throws std::invalid_argument For invalid dimensions,geometry,dtypes,targets or incompatible stored leaves.
/// @par Example
/// @code
/// const auto output = dm::nn::models::Unet3DOutputSide(132);
/// @endcode
/// @par References
/// - Çiçek,Abdulkadir,Lienkamp,Brox and Ronneberger(2016),3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation, https://arxiv.org/abs/1606.06650; Milletari,Navab and Ahmadi(2016),V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation, https://arxiv.org/abs/1606.04797. Initializers,explicit BN affine leaves and compact dimensions follow the source teaching example.
/// @see Unet3D()
inline int Unet3DOutputSide(int side, bool valid = true) {
  for (int i = 0; i < 3; ++i) {
    if (valid)
      side -= 4;
    if (side < 2)
      throw std::invalid_argument("3D U-Net input too small");
    side /= 2;
  }
  if (valid)
    side -= 4;
  for (int i = 0; i < 3; ++i) {
    side *= 2;
    if (valid)
      side -= 4;
    if (side < 1)
      throw std::invalid_argument("3D U-Net input too small");
  }
  return side;
}

/// @brief Configure sparse-label3D U-Net widths,padding and batch-normalization inference policy.
///
/// Full defaults reproduce three cropped skips and doubled encoder widths
/// before pooling;compact settings retain the entire topology.
///
/// @throws std::invalid_argument For invalid dimensions,geometry,dtypes,targets or incompatible stored leaves.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(18);
/// n::models::Unet3DConfig cfg;
/// cfg.base = 2;
/// cfg.input_channels = 1;
/// cfg.classes = 2;
/// cfg.valid = false;
/// const auto net = n::models::Unet3D(p, state, n::ones({1, 16, 16, 16, 1}), cfg);
/// @endcode
/// @par References
/// - Çiçek,Abdulkadir,Lienkamp,Brox and Ronneberger(2016),3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation, https://arxiv.org/abs/1606.06650; Milletari,Navab and Ahmadi(2016),V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation, https://arxiv.org/abs/1606.04797. Initializers,explicit BN affine leaves and compact dimensions follow the source teaching example.
/// @see Unet3D(), VNet()
struct Unet3DConfig {
  int input_channels = 3; ///< Positive input channel count.
  int classes = 3;        ///< At least two linear voxel classes.
  int base = 32;          ///< Positive first encoder width;each depth doubles.
  bool valid = true;      ///< True uses VALID convolution pairs/cropped skips;false SAME.
  bool batch_norm =
      true; ///< True adds learned affine BN after every pair member,no convolution biases there.
  bool batch_stats_at_test =
      true; ///< True uses current-volume/batch moments even when training=false,without automatic updates.
};
/// @brief Own linear voxel logits and encoder/decoder skip-stage activations.
///
/// Tensor handles retain caller-store graphs;no independent optimizer or
/// learned state is copied into this result.
///
/// @throws std::invalid_argument For invalid dimensions,geometry,dtypes,targets or incompatible stored leaves.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(18);
/// n::models::Unet3DConfig cfg;
/// cfg.base = 2;
/// cfg.input_channels = 1;
/// cfg.classes = 2;
/// cfg.valid = false;
/// const auto net = n::models::Unet3D(p, state, n::ones({1, 16, 16, 16, 1}), cfg);
/// @endcode
/// @par References
/// - Çiçek,Abdulkadir,Lienkamp,Brox and Ronneberger(2016),3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation, https://arxiv.org/abs/1606.06650; Milletari,Navab and Ahmadi(2016),V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation, https://arxiv.org/abs/1606.04797. Initializers,explicit BN affine leaves and compact dimensions follow the source teaching example.
/// @see Unet3D(), VNet()
struct VolumeOutputs {
  nn::Tensor logits; ///< Floating[N,Dout,Hout,Wout,classes] unnormalized voxel scores.
  std::vector<nn::Tensor> encoder; ///< High-to-low-resolution stage maps before pooling.
  std::vector<nn::Tensor> decoder; ///< Low-to-high-resolution stage maps after skip fusion.
};
/// @brief Build three-skip3D U-Net with doubled encoder widths and channel-preserving learned up-convolutions.
///
/// Each encoder stage has3x3x3 layers at base*2^i and twice that width;
/// three2x2x2 VALID pools. Up-convolutions preserve current channels,concatenate
/// centre-cropped skips,then two3x3x3 layers. BN momentum.9/epsilon1e-5,ReLU;
/// linear1x1x1 classifier. Full VALID132x132x116->44x44x28. Full convolution
/// census19069955;explicit affine BN adds2336. No hidden fitting. Warm every
/// leaf before optimizer;checkpoint learned/BN stores together. Default test
/// batch statistics make predictions depend on inference batch composition;
/// the source scores one volume at a time and never updates test moments.
///
/// @param p Same-dtype floating learned store,creates/reuses every convolution and BN affine leaf.
/// @param state Same-dtype BN moving moments,only updated if training and update_stats.
/// @param image Positive same-dtype[N,D,H,W,cfg.input_channels] volume.
/// @param cfg Valid widths/padding/BN policies,full defaults base32/RGB-like3 input/3 classes/VALID.
/// @param training False(default) uses the configured inference policy;true uses current batch moments.
/// @param update_stats False(default) prevents moving updates;true permits updates only when training.
/// @return Logits,four encoder maps and three decoder maps.
/// @throws std::invalid_argument For invalid dimensions,geometry,dtypes,targets or incompatible stored leaves.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(18);
/// n::models::Unet3DConfig cfg;
/// cfg.base = 2;
/// cfg.input_channels = 1;
/// cfg.classes = 2;
/// cfg.valid = false;
/// const auto net = n::models::Unet3D(p, state, n::ones({1, 16, 16, 16, 1}), cfg);
/// @endcode
/// @par References
/// - Çiçek,Abdulkadir,Lienkamp,Brox and Ronneberger(2016),3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation, https://arxiv.org/abs/1606.06650; Milletari,Navab and Ahmadi(2016),V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation, https://arxiv.org/abs/1606.04797. Initializers,explicit BN affine leaves and compact dimensions follow the source teaching example.
/// @see Unet3D(), VNet()
inline VolumeOutputs Unet3D(nn::ParameterStore &p, nn::ParameterStore &state,
                            const nn::Tensor &image, const Unet3DConfig &cfg = {},
                            bool training = false, bool update_stats = false) {
  if (cfg.input_channels < 1 || cfg.classes < 2 || cfg.base < 1)
    throw std::invalid_argument("invalid 3D U-Net widths");
  if (image.dim() != 5 || image.shape()[0] < 1 || image.shape()[4] != cfg.input_channels ||
      image.dtype() != p.dtype() || state.dtype() != p.dtype())
    throw std::invalid_argument("invalid3D U-Net volume/stores");
  for (int axis = 1; axis <= 3; ++axis)
    Unet3DOutputSide(int(image.shape()[axis]), cfg.valid);
  VolumeOutputs out;
  nn::ConvLayerOptions conv;
  conv.padding = cfg.valid ? nn::Padding::Valid : nn::Padding::Same;
  conv.bias = !cfg.batch_norm;
  conv.init = nn::init::he_uniform();
  /// @brief Borrow stores and apply the current padding/BN policy to one ReLU layer.
  const auto layer = [&](const std::string &name, const nn::Tensor &x, std::int64_t channels) {
    nn::Tensor h = nn::conv3d(p, name, x, channels, 3, conv);
    if (cfg.batch_norm) {
      nn::BatchNormOptions bn;
      bn.momentum = .9;
      bn.eps = 1e-5;
      bn.update_stats = training && update_stats;
      h = nn::batch_norm(p, state, name + "/bn", h, training || cfg.batch_stats_at_test, bn);
    }
    return nn::relu(h);
  };
  nn::Tensor h = image;
  std::int64_t channels = cfg.input_channels;
  for (int i = 0; i < 4; ++i) {
    const std::int64_t first = std::int64_t{cfg.base} << i, second = 2 * first;
    h = layer("encoder" + std::to_string(i) + "/first", h, first);
    h = layer("encoder" + std::to_string(i) + "/second", h, second);
    channels = second;
    out.encoder.push_back(h);
    if (i < 3)
      h = nn::max_pool3d(h, 2, 2, nn::Padding::Valid);
  }
  for (int i = 2; i >= 0; --i) {
    h = VolumeTranspose(p, "decoder" + std::to_string(i) + "/up", h, channels);
    const std::int64_t skip = std::int64_t{cfg.base} << (i + 1);
    h = nn::concat({VolumeCrop(out.encoder[static_cast<std::size_t>(i)], h), h}, 4);
    h = layer("decoder" + std::to_string(i) + "/first", h, skip);
    h = layer("decoder" + std::to_string(i) + "/second", h, skip);
    channels = skip;
    out.decoder.push_back(h);
  }
  out.logits = nn::conv3d(p, "score", h, cfg.classes, 1);
  return out;
}

/// @brief Apply learned per-channel PReLU with slope.25 initially and source derivative1 at zero.
///
/// No hidden fit. Negative-input gradients include alpha,while zero takes the
/// identity branch. Store alpha initialization is scalar.25 per channel.
///
/// @param p Same-dtype learned store,creates/reuses name/alpha.
/// @param name Nonempty parameter prefix.
/// @param x Floating tensor with positive last-axis channels.
/// @return Same-shape/type output,x for x>=0 and alpha*x below zero.
/// @throws std::invalid_argument For invalid dimensions,geometry,dtypes,targets or incompatible stored leaves.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17);
/// const auto y = n::models::VolumePRelu(p, "act", n::tensor({-2., 0., 2.}));
/// @endcode
/// @par References
/// - Çiçek,Abdulkadir,Lienkamp,Brox and Ronneberger(2016),3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation, https://arxiv.org/abs/1606.06650; Milletari,Navab and Ahmadi(2016),V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation, https://arxiv.org/abs/1606.04797. Initializers,explicit BN affine leaves and compact dimensions follow the source teaching example.
/// @see VolumeResidual(), VNet()
inline nn::Tensor VolumePRelu(nn::ParameterStore &p, const std::string &name,
                              const nn::Tensor &x) {
  if (x.dim() < 1 || x.shape().back() < 1 || x.dtype() != p.dtype())
    throw std::invalid_argument("invalid volume PReLU input");
  const nn::Tensor alpha = p.get(name + "/alpha", {x.shape().back()}, nn::init::constant(.25));
  return nn::where(nn::greater_equal(x, 0.0), x, x * alpha);
}

/// @brief Apply5x5x5 convolution/PReLU layers with the input added before the final activation.
///
/// Residual addition precedes the last PReLU. No normalization,dropout or
/// hidden fitting.
///
/// @param p Same-dtype learned store.
/// @param name Unique residual-block prefix.
/// @param input Positive floating NDHWC features.
/// @param channels Positive hidden/output channels,matching input channels.
/// @param layers Positive number of biased He5x5x5 SAME convolutions.
/// @return Same-shape residual output with all convolution/slope/input gradients.
/// @throws std::invalid_argument For invalid dimensions,geometry,dtypes,targets or incompatible stored leaves.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17);
/// const auto y = n::models::VolumeResidual(p, "block", n::ones({1, 2, 3, 4, 2}), 2, 2);
/// @endcode
/// @par References
/// - Çiçek,Abdulkadir,Lienkamp,Brox and Ronneberger(2016),3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation, https://arxiv.org/abs/1606.06650; Milletari,Navab and Ahmadi(2016),V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation, https://arxiv.org/abs/1606.04797. Initializers,explicit BN affine leaves and compact dimensions follow the source teaching example.
/// @see VolumePRelu(), VNet()
inline nn::Tensor VolumeResidual(nn::ParameterStore &p, const std::string &name,
                                 const nn::Tensor &input, std::int64_t channels, int layers) {
  if (channels < 1 || layers < 1 || input.dim() != 5 || input.shape()[4] != channels ||
      input.dtype() != p.dtype())
    throw std::invalid_argument("positive volume residual dimensions required");
  nn::ConvLayerOptions conv;
  conv.init = nn::init::he_uniform();
  nn::Tensor h = input;
  for (int i = 0; i < layers; ++i) {
    h = nn::conv3d(p, name + "/conv" + std::to_string(i), h, channels, 5, conv);
    if (i == layers - 1)
      h = h + input;
    h = VolumePRelu(p, name + "/act" + std::to_string(i), h);
  }
  return h;
}

/// @brief Configure four-level residual V-Net channel width and voxel class count.
///
/// Metadata only,no learned parameters or optimizer.
///
/// @throws std::invalid_argument For invalid dimensions,geometry,dtypes,targets or incompatible stored leaves.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17);
/// n::models::VNetConfig cfg;
/// cfg.base = 2;
/// const auto net = n::models::VNet(p, n::ones({1, 16, 16, 16, 1}), cfg);
/// @endcode
/// @par References
/// - Çiçek,Abdulkadir,Lienkamp,Brox and Ronneberger(2016),3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation, https://arxiv.org/abs/1606.06650; Milletari,Navab and Ahmadi(2016),V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation, https://arxiv.org/abs/1606.04797. Initializers,explicit BN affine leaves and compact dimensions follow the source teaching example.
/// @see Unet3D(), VNet()
struct VNetConfig {
  int base = 16;   ///< Positive initial width,doubles over four downsampling stages.
  int classes = 2; ///< At least two linear voxel classes.
};
/// @brief Build full four-level residual V-Net with5x5x5 stacks and learned per-channel PReLU.
///
/// Input5x5x5 projection adds a channel-tiled image before PReLU. Four
/// biased2x2x2 stride2 VALID downsamples precede residual stacks of2/3/3/3
/// 5x5x5 layers. Decoder2x up-convolutions concatenate skips and use3/3/2/1
/// residual layers. No BN,moving state or hidden fitting. Full65175314
/// learned scalars;compact base2 has1019348. Warm every leaf before optimizer.
///
/// @param p Same-dtype learned store for every convolution,bias and PReLU slope.
/// @param image Positive floating[N,D,H,W,1],each spatial side divisible by16 and at least16.
/// @param cfg Valid base/classes,full defaults16/two.
/// @return Logits aligned to input,five encoder and four decoder maps.
/// @throws std::invalid_argument For invalid dimensions,geometry,dtypes,targets or incompatible stored leaves.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17);
/// n::models::VNetConfig cfg;
/// cfg.base = 2;
/// const auto net = n::models::VNet(p, n::ones({1, 16, 16, 16, 1}), cfg);
/// @endcode
/// @par References
/// - Çiçek,Abdulkadir,Lienkamp,Brox and Ronneberger(2016),3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation, https://arxiv.org/abs/1606.06650; Milletari,Navab and Ahmadi(2016),V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation, https://arxiv.org/abs/1606.04797. Initializers,explicit BN affine leaves and compact dimensions follow the source teaching example.
/// @see Unet3D(), VNet()
inline VolumeOutputs VNet(nn::ParameterStore &p, const nn::Tensor &image,
                          const VNetConfig &cfg = {}) {
  if (cfg.base < 1 || cfg.classes < 2)
    throw std::invalid_argument("invalid V-Net widths");
  if (image.dim() != 5 || image.shape()[0] < 1 || image.shape()[4] != 1 ||
      image.dtype() != p.dtype())
    throw std::invalid_argument("V-Net needs floating single-channel volume");
  for (int axis = 1; axis <= 3; ++axis)
    if (image.shape()[axis] < 16 || image.shape()[axis] % 16)
      throw std::invalid_argument("V-Net spatial axes must be positive multiples of16");
  VolumeOutputs out;
  nn::ConvLayerOptions conv;
  conv.init = nn::init::he_uniform();
  nn::Tensor h = nn::conv3d(p, "input", image, cfg.base, 5, conv);
  h = VolumePRelu(p, "input/act", h + nn::tile(image, {1, 1, 1, 1, cfg.base}));
  out.encoder.push_back(h);
  const int counts[] = {2, 3, 3, 3};
  std::int64_t channels = cfg.base;
  for (int i = 0; i < 4; ++i) {
    nn::ConvLayerOptions down = conv;
    down.stride = 2, down.padding = nn::Padding::Valid;
    const std::int64_t width = 2 * channels;
    h = nn::conv3d(p, "down" + std::to_string(i), h, width, 2, down);
    h = VolumePRelu(p, "down" + std::to_string(i) + "/act", h);
    h = VolumeResidual(p, "encoder" + std::to_string(i + 1), h, width, counts[i]);
    channels = width;
    out.encoder.push_back(h);
  }
  const int decoder_layers[] = {3, 3, 2, 1};
  for (int i = 3; i >= 0; --i) {
    const std::int64_t skip = std::int64_t{cfg.base} << i;
    h = VolumeTranspose(p, "up" + std::to_string(i), h, skip, true);
    h = VolumePRelu(p, "up" + std::to_string(i) + "/act", h);
    h = nn::concat({h, out.encoder[static_cast<std::size_t>(i)]}, 4);
    channels = 2 * skip;
    h = VolumeResidual(p, "decoder" + std::to_string(i), h, channels, decoder_layers[3 - i]);
    out.decoder.push_back(h);
  }
  out.logits = nn::conv3d(p, "score", h, cfg.classes, 1);
  return out;
}

/// @brief Average stable class-weighted softmax CE over annotated volume voxels.
///
/// Every unannotated logit has exactly zero gradient. No hidden fit or source
/// mutation. Weight normalization uses annotated voxel weights,not voxel count.
///
/// @param logits Floating[N,D,H,W,classes] voxel scores,positive dimensions.
/// @param labels Int32/64[N,D,H,W];values outside[0,classes) are unannotated and ignored.
/// @param classes At least two output channels,matching logits.
/// @param class_weights Same-dtype[classes] finite nonnegative weights,detached.
/// @return Scalar sum(weight*CE)/max(sum(weight),1e-6);all void returns differentiable zero.
/// @throws std::invalid_argument For invalid dimensions,geometry,dtypes,targets or incompatible stored leaves.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto loss = n::models::SparseVolumeCE(
///     n::zeros({1, 1, 1, 3, 3}),
///     n::from_vector(std::vector<std::int64_t>{0, 2, 255}, {1, 1, 1, 3}), 3,
///     n::tensor({1., 2., 4.}));
/// @endcode
/// @par References
/// - Çiçek,Abdulkadir,Lienkamp,Brox and Ronneberger(2016),3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation, https://arxiv.org/abs/1606.06650; Milletari,Navab and Ahmadi(2016),V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation, https://arxiv.org/abs/1606.04797. Initializers,explicit BN affine leaves and compact dimensions follow the source teaching example.
/// @see SquaredVolumeDice(), Unet3D()
inline nn::Tensor SparseVolumeCE(const nn::Tensor &logits, const nn::Tensor &labels,
                                 int classes, const nn::Tensor &class_weights) {
  if (logits.dim() != 5 || logits.shape()[0] < 1 || classes < 2 ||
      logits.shape()[4] != classes ||
      (logits.dtype() != nn::DType::Float32 && logits.dtype() != nn::DType::Float64) ||
      labels.shape() != nn::Shape{logits.shape()[0], logits.shape()[1], logits.shape()[2],
                                  logits.shape()[3]} ||
      (labels.dtype() != nn::DType::Int32 && labels.dtype() != nn::DType::Int64) ||
      class_weights.shape() != nn::Shape{classes} || class_weights.dtype() != logits.dtype() ||
      !nn::all(
           nn::logical_and(nn::greater_equal(class_weights, 0.),
                           nn::less(class_weights, std::numeric_limits<double>::infinity())))
           .item<bool>())
    throw std::invalid_argument("invalid sparse volume CE inputs");
  for (int axis = 1; axis <= 3; ++axis)
    if (logits.shape()[axis] < 1)
      throw std::invalid_argument("positive spatial loss axes required");
  const auto type = logits.dtype();
  const auto truth = labels.reshape({-1});
  const auto valid =
      nn::logical_and(nn::greater_equal(truth, 0.), nn::less(truth, double(classes))).to(type);
  const auto target = nn::one_hot(nn::clip(truth, 0, classes - 1), classes).to(type);
  const auto weights = nn::sum(target * class_weights.detach(), {1}) * valid;
  const auto ce = -nn::sum(nn::log_softmax(logits.reshape({-1, classes})) * target, {1});
  return nn::sum(ce * weights) / nn::maximum(nn::sum(weights), nn::full({}, 1e-6, type));
}
/// @brief Average V-Net squared-denominator foreground Dice loss over volumes with any valid labels.
///
/// Void voxels have zero probability derivative;entirely void volumes are
/// excluded. No valid volume gives graph-connected zero. Perfect predictions
/// have zero loss. This uses a squared denominator,not linear Dice or a
/// thresholded metric. No fitting or input mutation.
///
/// @param foreground Floating[N,D,H,W] finite probabilities in[0,1],positive dimensions.
/// @param labels Same-shape Int32/64 binary0/1 truth;other labels are ignored.
/// @return Scalar mean1-(2*sum(p*t)+1e-6)/(sum(p²+t²)+1e-6) over annotated volumes.
/// @throws std::invalid_argument For invalid dimensions,geometry,dtypes,targets or incompatible stored leaves.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto loss = n::models::SquaredVolumeDice(
///     n::tensor({.2, .5, .9}, {1, 1, 1, 3}),
///     n::from_vector(std::vector<std::int64_t>{0, 1, 1}, {1, 1, 1, 3}));
/// @endcode
/// @par References
/// - Çiçek,Abdulkadir,Lienkamp,Brox and Ronneberger(2016),3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation, https://arxiv.org/abs/1606.06650; Milletari,Navab and Ahmadi(2016),V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation, https://arxiv.org/abs/1606.04797. Initializers,explicit BN affine leaves and compact dimensions follow the source teaching example.
/// @see SparseVolumeCE(), VNet()
inline nn::Tensor SquaredVolumeDice(const nn::Tensor &foreground, const nn::Tensor &labels) {
  if (foreground.dim() != 4 || foreground.shape()[0] < 1 || foreground.shape()[1] < 1 ||
      foreground.shape()[2] < 1 || foreground.shape()[3] < 1 ||
      (foreground.dtype() != nn::DType::Float32 && foreground.dtype() != nn::DType::Float64) ||
      foreground.shape() != labels.shape() ||
      (labels.dtype() != nn::DType::Int32 && labels.dtype() != nn::DType::Int64) ||
      !nn::all(
           nn::logical_and(nn::greater_equal(foreground, 0.), nn::less_equal(foreground, 1.)))
           .item<bool>())
    throw std::invalid_argument("invalid squared volume Dice inputs");
  const auto type = foreground.dtype();
  const auto valid =
                 nn::logical_and(nn::greater_equal(labels, 0.), nn::less(labels, 2.)).to(type),
             prediction = foreground * valid, truth = nn::equal(labels, 1.).to(type);
  const auto numerator =
                 2. * nn::sum(prediction * truth, {1, 2, 3}) + nn::full({}, 1e-6, type),
             denominator = nn::sum(nn::square(prediction) + nn::square(truth), {1, 2, 3}) +
                           nn::full({}, 1e-6, type);
  const auto has = nn::greater(nn::sum(valid, {1, 2, 3}), 0.).to(type);
  return nn::sum((1. - numerator / denominator) * has) /
         nn::maximum(nn::sum(has), nn::ones({}, type));
}
/// @brief Compute fit-only inverse-square-root voxel class weights with one pseudocount.
///
/// Counts use exact Int64 tensor reductions,ratios/square roots Float64 before
/// Float32 output. Empty/all-void labels return sqrt(classes) for each class.
/// No fitting,input mutation or held-out information;pass fitting labels only.
///
/// @param labels Int32/64 tensor of any shape;values outside[0,classes) are ignored.
/// @param classes At least two categories,default2.
/// @return Detached Float32[classes],1/sqrt((count_c+1)/sum_j(count_j+1)).
/// @throws std::invalid_argument For invalid label dtype or class count.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto weights = n::models::SparseVolumeClassWeights(
///     n::from_vector(std::vector<std::int64_t>{0, 0, 1, 255}), 2);
/// @endcode
/// @par References
/// - Source inverse-square-root sparse-voxel weighting convention from examples/cpp/source/nn/1_convolutional_neural_networks/1.7_segmentation/volumetric_segmentation.cpp.
/// @see SparseVolumeCE()
inline nn::Tensor SparseVolumeClassWeights(const nn::Tensor &labels, int classes = 2) {
  if (classes < 2 ||
      (labels.dtype() != nn::DType::Int32 && labels.dtype() != nn::DType::Int64))
    throw std::invalid_argument("invalid volume class weights");
  nn::NoGradGuard guard;
  std::vector<nn::Tensor> counts;
  for (int c = 0; c < classes; ++c)
    counts.push_back(nn::count_nonzero(nn::equal(labels, double(c))) +
                     nn::ones({}, nn::DType::Int64));
  const auto frequency = nn::stack(counts, 0).to(nn::DType::Float64);
  return (nn::ones({}, nn::DType::Float64) / nn::sqrt(frequency / nn::sum(frequency)))
      .to(nn::DType::Float32);
}
} // namespace datamunge::nn::models
