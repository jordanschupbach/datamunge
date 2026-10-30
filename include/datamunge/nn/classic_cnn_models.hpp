#pragma once
/// @file classic_cnn_models.hpp
/// @brief Network in Network and VGG tensor model definitions, without fitting loops.
#include "datamunge/nn/layers.hpp"
#include "datamunge/nn/ops.hpp"
#include <array>
#include <stdexcept>
#include <string>
#include <vector>
namespace datamunge::nn::models {
namespace nn = datamunge::nn;

/// @brief Apply a spatial ReLU convolution and two pointwise ReLU channel mixers.
///
/// SAME padding preserves NHWC spatial dimensions. Each convolution creates/reuses
/// its own named kernel/bias with He initialization. No parameters are fitted.
/// Gradients reach inputs and every learned tensor; work is dominated by spatial
/// convolution plus the two local channel matrix products.
/// @param p Mutable learnable floating parameter store.
/// @param name Prefix; suffixes /spatial, /mix1 and /mix2 identify the layers.
/// @param input Floating [N,H,W,C], same dtype as p.
/// @param spatial_width Positive width of the spatial convolution output.
/// @param first_width Positive width of the first pointwise hidden mixer.
/// @param out Positive output-channel width of the second pointwise mixer.
/// @param kernel Positive square spatial-kernel width.
/// @return ReLU feature map [N,H,W,out], retaining its graph.
/// @throws std::invalid_argument For incompatible shapes/store entries or invalid widths.
/// @par Example
/// @code
/// namespace n=dm::nn; n::ParameterStore p(17);
/// const auto y=n::models::MlpConv(p,"mlp",n::ones({1,4,4,2}),4,3,2,3);
/// @endcode
/// @par References
/// - Lin, Chen and Yan (2014), Network In Network, https://arxiv.org/abs/1312.4400
/// @see NetworkInNetwork()
inline nn::Tensor MlpConv(nn::ParameterStore &p, const std::string &name,
                          const nn::Tensor &input, std::int64_t spatial_width,
                          std::int64_t first_width, std::int64_t out, std::int64_t kernel) {
  nn::ConvLayerOptions options;
  options.activation = nn::Activation::Relu;
  options.init = nn::init::he_uniform();
  auto h = nn::conv2d(p, name + "/spatial", input, spatial_width, kernel, options);
  h = nn::conv2d(p, name + "/mix1", h, first_width, 1, options);
  return nn::conv2d(p, name + "/mix2", h, out, 1, options);
}

/// @brief NiN class count, exact width scaling and training dropout retention.
/// @par Example
/// @code
/// dm::nn::models::NinConfig cfg; cfg.width_divisor=4; cfg.Validate();
/// @endcode
/// @see NetworkInNetwork()
struct NinConfig {
  int classes = 10;             ///< Number of class maps/logits, at least two.
  int width_divisor = 1;        ///< Positive divisor of 32; one preserves reference widths.
  double keep_probability = .5; ///< Training dropout retention in (0,1].

  /// @brief Reject invalid class, exact-width or dropout settings.
  /// @throws std::invalid_argument For classes below two, invalid divisor or retention.
  /// @par Example
  /// See the NinConfig example.
  /// @see NinConfig
  void Validate() const {
    if (classes < 2 || width_divisor <= 0 || 32 % width_divisor ||
        !(keep_probability > 0 && keep_probability <= 1))
      throw std::invalid_argument("invalid NiN configuration");
  }
};

/// @brief NiN intermediate class/feature maps, including their gradient graphs.
/// @par Example
/// @code
/// namespace n=dm::nn; n::ParameterStore p(17);
/// n::models::NinConfig cfg; cfg.width_divisor=8;
/// const auto out=n::models::NetworkInNetwork(p,n::ones({1,8,8,3}),cfg);
/// @endcode
/// @see NetworkInNetwork()
struct NinOutputs {
  std::array<nn::Tensor, 3>
      blocks; ///< Maps before each pooling/readout, final channels=classes.
  std::array<nn::Tensor, 2> pooling; ///< First two pooled maps, with SAME edge handling.
  nn::Tensor logits;                 ///< Spatial mean of final class maps [N,classes].
};

/// @brief Build the authors' three-MLPconv CIFAR NiN topology, with optional width scaling.
///
/// Full widths are 192/160/96, 192/192/192, 192/192/classes; spatial kernels
/// are 5/5/3. ReLU follows every convolution, including the final class maps.
/// SAME 3x3 stride-two max-pooling follows the first two blocks. Global mean
/// gives logits directly. He initialization and the caller's optimiser/input
/// preprocessing are adaptations; no original ZCA or contrast normalization.
/// @param p Mutable parameter store, named block1/2/3.
/// @param image Floating NHWC image tensor with three RGB input channels.
/// @param cfg Class count, width divisor, dropout retention; default reference widths.
/// @param training Apply dropout after each pool if true; default deterministic false.
/// @return Intermediate maps/pools and [N,classes] logits; parameters are not fitted.
/// @throws std::invalid_argument For invalid configuration, shapes or store entries.
/// @par Example
/// @code
/// namespace n=dm::nn; n::ParameterStore p(17);
/// n::models::NinConfig cfg; cfg.width_divisor=8;
/// const auto y=n::models::NetworkInNetwork(p,n::ones({1,8,8,3}),cfg).logits;
/// @endcode
/// @par References
/// - Lin, Chen and Yan (2014), Network In Network, https://arxiv.org/abs/1312.4400
/// - Authors' topology, https://github.com/mavenlin/cuda-convnet/blob/master/NIN/cifar-10_def
/// @see MlpConv(), NinConfig
inline NinOutputs NetworkInNetwork(nn::ParameterStore &p, const nn::Tensor &image,
                                   const NinConfig &cfg = {}, bool training = false) {
  cfg.Validate();
  NinOutputs out;
  const std::int64_t a = 192 / cfg.width_divisor, b = 160 / cfg.width_divisor,
                     c = 96 / cfg.width_divisor;
  // Draw only in training mode, using the store-owned reproducible random stream.
  auto drop = [&](const nn::Tensor &x) {
    return training ? nn::dropout(x, 1 - cfg.keep_probability, true, p.generator()) : x;
  };
  out.blocks[0] = MlpConv(p, "block1", image, a, b, c, 5);
  out.pooling[0] = nn::max_pool2d(out.blocks[0], 3, 2, nn::Padding::Same);
  out.blocks[1] = MlpConv(p, "block2", drop(out.pooling[0]), a, a, a, 5);
  out.pooling[1] = nn::max_pool2d(out.blocks[1], 3, 2, nn::Padding::Same);
  out.blocks[2] = MlpConv(p, "block3", drop(out.pooling[1]), a, a, cfg.classes, 3);
  out.logits = nn::global_avg_pool2d(out.blocks[2]);
  return out;
}

/// @brief Return convolution counts for VGG A/D/E (11/16/19 total learned layers).
/// @param depth Total learned depth, exactly 11, 16 or 19 including three dense layers.
/// @return Five block depths in input-to-output order.
/// @throws std::invalid_argument For any other depth.
/// @par Example
/// @code
/// const auto blocks=dm::nn::models::VggBlockDepths(16); // 2,2,3,3,3
/// @endcode
/// @par References
/// - Simonyan and Zisserman (2015), Very Deep Convolutional Networks,
/// https://arxiv.org/abs/1409.1556
/// @see Vgg()
inline std::array<int, 5> VggBlockDepths(int depth) {
  if (depth == 11)
    return {1, 1, 2, 2, 2};
  if (depth == 16)
    return {2, 2, 3, 3, 3};
  if (depth == 19)
    return {2, 2, 4, 4, 4};
  throw std::invalid_argument("VGG depth must be 11, 16 or 19");
}

/// @brief VGG depth and reference/compact dimensions, with dropout retention.
/// @par Example
/// @code
/// dm::nn::models::VggConfig cfg; cfg.image_size=32; cfg.width_divisor=8;
/// cfg.dense_width=32; cfg.classes=10; cfg.Validate();
/// @endcode
/// @see Vgg()
struct VggConfig {
  int depth = 16;         ///< Total learned layers; 11,16,19 select A,D,E.
  int image_size = 224;   ///< Square input side, at least 32, used to size FC6.
  int classes = 1000;     ///< Positive output width, at least two.
  int width_divisor = 1;  ///< Positive exact divisor of 64; scales all convolution widths.
  int dense_width = 4096; ///< Positive width of both hidden dense layers.
  double keep_probability = .5; ///< Dropout retention in (0,1], applied only in training.

  /// @brief Validate depth, dimensions, exact width scaling and dropout retention.
  /// @throws std::invalid_argument For invalid values; does not check an input tensor.
  /// @par Example
  /// See the VggConfig example.
  /// @see VggBlockDepths()
  void Validate() const {
    VggBlockDepths(depth);
    if (image_size < 32 || classes < 2 || dense_width < 1 || width_divisor <= 0 ||
        64 % width_divisor || !(keep_probability > 0 && keep_probability <= 1))
      throw std::invalid_argument("invalid VGG configuration");
  }
  /// @brief Return floor(image_size/32) after five VALID stride-two pools.
  /// @return Positive final feature-map width.
  /// @throws std::invalid_argument For invalid configuration.
  /// @par Example
  /// @code
  /// const auto width=dm::nn::models::VggConfig{}.FinalSpatialSize(); // 7
  /// @endcode
  /// @see Validate()
  int FinalSpatialSize() const {
    Validate();
    return image_size / 32;
  }
};

/// @brief All VGG convolution/pool activations and the three classifier outputs.
/// @par Example
/// @code
/// namespace n=dm::nn; n::ParameterStore p(17);
/// n::models::VggConfig cfg; cfg.image_size=32; cfg.width_divisor=8;
/// cfg.dense_width=16; cfg.classes=10;
/// const auto out=n::models::Vgg(p,n::zeros({1,32,32,3}),cfg);
/// @endcode
/// @see Vgg()
struct VggOutputs {
  std::vector<nn::Tensor> convolution; ///< Maps after each 3x3 ReLU, in execution order.
  std::array<nn::Tensor, 5> pooling;   ///< Maps after five 2x2 stride-two pools.
  nn::Tensor fc6;                      ///< First dense ReLU output, before its dropout.
  nn::Tensor fc7;                      ///< Second dense ReLU output, before its dropout.
  nn::Tensor logits;                   ///< Unnormalised [N,classes] classifier outputs.
};

/// @brief Build VGG A/D/E with five spatial blocks and a three-layer dense classifier.
///
/// Only SAME 3x3 convolutions, ReLU and VALID 2x2 max-pooling occur in the
/// feature extractor; no BN/LRN. Widths 64/128/256/512/512 scale exactly by
/// cfg.width_divisor. He/Glorot initialization is a modern adaptation. Parameters
/// are created/reused but never fitted; dropout uses p's generator only when
/// training. The input must match cfg.image_size and the store dtype.
/// @param p Mutable learnable parameter store.
/// @param image Floating [N,image_size,image_size,3].
/// @param cfg Valid depth, image/class/width/dense dimensions and retention.
/// @param training Enable hidden-layer dropout; false by default.
/// @return Every stage and [N,classes] logits, retaining gradient graphs.
/// @throws std::invalid_argument For invalid cfg, tensor shapes or parameter entries.
/// @par Example
/// @code
/// namespace n=dm::nn; n::ParameterStore p(17);
/// n::models::VggConfig cfg; cfg.image_size=32; cfg.width_divisor=8;
/// cfg.dense_width=16; cfg.classes=10;
/// const auto y=n::models::Vgg(p,n::zeros({1,32,32,3}),cfg).logits;
/// @endcode
/// @par References
/// - Simonyan and Zisserman (2015), Very Deep Convolutional Networks,
/// https://arxiv.org/abs/1409.1556
/// @see VggConvolutionalHead(), VggConfig
inline VggOutputs Vgg(nn::ParameterStore &p, const nn::Tensor &image,
                      const VggConfig &cfg = {}, bool training = false) {
  cfg.Validate();
  VggOutputs out;
  const auto depths = VggBlockDepths(cfg.depth);
  const std::array<int, 5> widths = {64, 128, 256, 512, 512};
  nn::ConvLayerOptions options;
  options.activation = nn::Activation::Relu;
  options.init = nn::init::he_uniform();
  nn::Tensor h = image;
  std::int64_t channels = 3;
  for (int block = 0; block < 5; ++block) {
    const int width = widths[static_cast<std::size_t>(block)] / cfg.width_divisor;
    for (int layer = 0; layer < depths[static_cast<std::size_t>(block)]; ++layer) {
      h = nn::conv2d(p,
                     "block" + std::to_string(block + 1) + "/conv" + std::to_string(layer + 1),
                     h, width, 3, options);
      out.convolution.push_back(h);
      channels = width;
    }
    h = nn::max_pool2d(h, 2, 2);
    out.pooling[static_cast<std::size_t>(block)] = h;
  }
  const std::int64_t features =
      std::int64_t(cfg.FinalSpatialSize()) * cfg.FinalSpatialSize() * channels;
  nn::DenseOptions dense;
  dense.init = nn::init::he_uniform();
  dense.activation = nn::Activation::Relu;
  // Store-owned dropout randomness is consumed only by training calls.
  auto drop = [&](const nn::Tensor &x) {
    return training ? nn::dropout(x, 1 - cfg.keep_probability, true, p.generator()) : x;
  };
  out.fc6 = nn::dense(p, "fc6", nn::reshape(h, {-1, features}), cfg.dense_width, dense);
  out.fc7 = nn::dense(p, "fc7", drop(out.fc6), cfg.dense_width, dense);
  out.logits = nn::dense(p, "fc8", drop(out.fc7), cfg.classes);
  return out;
}

// Reuse a previously built VGG classifier as a fully convolutional head.
// FC6 becomes a spatial kernel and FC7/8 become 1x1 kernels. An F x F feature
// map produces a 1x1 score map; larger maps produce sliding classifier scores.
// These are sliding windows of feature maps, not equivalent raw-image crops
// (convolution padding and surrounding image context can differ).
/// @brief Reinterpret fitted dense VGG kernels as a sliding convolutional head.
///
/// FC6 becomes a spatial VALID filter; FC7/8 become pointwise filters. For
/// a matching final feature window, inference logits agree with the dense
/// classifier. Larger feature maps give sliding scores, not necessarily the
/// scores of independently cropped raw images with different context/padding.
/// No parameters or input tensors are mutated. Gradients remain available.
/// @param p Existing VGG parameter store containing fc6/7/8 kernels and biases.
/// @param features Floating [N,H,W,512/width_divisor], H,W at least final spatial size.
/// @param cfg Same dimensions used to create the dense head.
/// @return Floating score map [N,H-F+1,W-F+1,classes], F=FinalSpatialSize().
/// @throws std::invalid_argument For invalid cfg or incompatible tensor shapes.
/// @throws std::out_of_range If required classifier parameters are absent.
/// @par Example
/// @code
/// namespace n=dm::nn; n::ParameterStore p(17);
/// n::models::VggConfig cfg; cfg.image_size=32; cfg.width_divisor=8;
/// cfg.dense_width=16; cfg.classes=10;
/// const auto out=n::models::Vgg(p,n::zeros({1,32,32,3}),cfg);
/// const auto scores=n::models::VggConvolutionalHead(p,out.pooling[4],cfg);
/// @endcode
/// @par References
/// - Simonyan and Zisserman (2015), section 3, https://arxiv.org/abs/1409.1556
/// @see Vgg()
inline nn::Tensor VggConvolutionalHead(const nn::ParameterStore &p, const nn::Tensor &features,
                                       const VggConfig &cfg) {
  cfg.Validate();
  const std::int64_t spatial = cfg.FinalSpatialSize(), channels = 512 / cfg.width_divisor;
  const auto valid = nn::conv_options(1, nn::Padding::Valid);
  const auto first = nn::reshape(p.at("fc6/W"), {spatial, spatial, channels, cfg.dense_width});
  nn::Tensor h = nn::relu(nn::conv2d(features, first, valid) + p.at("fc6/b"));
  const auto second = nn::reshape(p.at("fc7/W"), {1, 1, cfg.dense_width, cfg.dense_width});
  h = nn::relu(nn::conv2d(h, second, valid) + p.at("fc7/b"));
  const auto last = nn::reshape(p.at("fc8/W"), {1, 1, cfg.dense_width, cfg.classes});
  return nn::conv2d(h, last, valid) + p.at("fc8/b");
}

} // namespace datamunge::nn::models
